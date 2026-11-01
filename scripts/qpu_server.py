#!/usr/bin/env python3
"""Local helper server so the static 3D tour can submit ONE real IBM Quantum job.

Mirrors qmed/ui/tab_ibm_hardware.py: circuit fm(x_p) o fm(x_q)^-1 + measure_all,
least-busy operational non-simulator backend, transpile optimization_level=2,
SamplerV2 with 1024 shots. Binds 127.0.0.1 only. Never returns/logs credentials.

Free-tier safeguards: explicit confirm flag, max_execution_time cap, one job in
flight, per-run job budget, persistent result cache keyed by (slug, patient).

Also serves the tour's live mode (scripts/live_api.py: retrain with settings, custom-patient
predict; never contacts IBM) and GET /api/qpu/backends (lists IBM backends, no QPU time).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCRIPTS_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT / "web" / "public" / "data"
CACHE_PATH = ROOT / "web" / ".qpu-cache.json"
QISKIT_ACCOUNTS_FILE = Path.home() / ".qiskit" / "qiskit-ibm.json"
DEFAULT_ACCOUNT = "default-ibm-quantum-platform"
SHOTS = 1024
MAX_EXECUTION_TIME_S = 60
FINAL_STATES = {"DONE", "ERROR", "CANCELLED"}

CONFIG = {"account": DEFAULT_ACCOUNT, "max_jobs": 3}
STATE = {"lock": threading.RLock(), "jobs": {}, "services": {}, "jobs_used": 0, "active": None}


# ---------------------------------------------------------------- credentials
def get_token() -> str:
    try:
        from qmed import config
        return (config.QISKIT_IBM_TOKEN or "").strip()
    except Exception:
        return os.getenv("QISKIT_IBM_TOKEN", "").strip()


def has_saved_account() -> bool:
    """True if a saved account with channel ibm_quantum_platform exists. Contents discarded."""
    try:
        with open(QISKIT_ACCOUNTS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        ok = any(isinstance(v, dict) and v.get("channel") == "ibm_quantum_platform" for v in data.values())
        del data
        return ok
    except Exception:
        return False


def has_credentials() -> bool:
    return bool(get_token()) or has_saved_account()


_TOKENISH = re.compile(r"[A-Za-z0-9_\-]{24,}")


def scrub(text: str) -> str:
    text = str(text)
    tok = get_token()
    if tok:
        text = text.replace(tok, "[redacted]")
    for secret in list(STATE.get("secrets", [])):
        if secret:
            text = text.replace(secret, "[redacted]")
    return _TOKENISH.sub("[redacted]", text)


def runtime_available() -> bool:
    try:
        import importlib.util
        return importlib.util.find_spec("qiskit_ibm_runtime") is not None
    except Exception:
        return False


# ---------------------------------------------------------- injectable factories
def make_service():
    """Create the IBM service following the documented credential order."""
    from qiskit_ibm_runtime import QiskitRuntimeService
    attempts = []
    tok = get_token()
    if tok:
        attempts.append(lambda: QiskitRuntimeService(channel="ibm_quantum_platform", token=tok))
    attempts.append(lambda: QiskitRuntimeService(name=CONFIG["account"]))
    attempts.append(lambda: QiskitRuntimeService(channel="ibm_quantum_platform"))
    attempts.append(lambda: QiskitRuntimeService())
    last = None
    for fn in attempts:
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError(scrub(f"Could not create IBM Quantum service: {last}"))


def transpile_circuit(qc, backend):
    from qiskit import transpile
    return transpile(qc, backend=backend, optimization_level=2)


def submit_job(backend, tqc):
    from qiskit_ibm_runtime import SamplerV2
    sampler = SamplerV2(mode=backend)
    sampler.options.max_execution_time = MAX_EXECUTION_TIME_S
    return sampler.run([tqc], shots=SHOTS)


# ----------------------------------------------------------------------- logic
class ApiError(Exception):
    def __init__(self, code, message, **extra):
        super().__init__(message)
        self.code, self.message, self.extra = code, message, extra


def load_cache() -> dict:
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def save_cache(cache: dict) -> None:
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f)
    except Exception:
        pass


def status_name(job) -> str:
    s = job.status()
    name = getattr(s, "name", None) or str(s)
    return name.split(".")[-1].upper()


def load_dataset(slug: str) -> dict:
    if not isinstance(slug, str) or not re.fullmatch(r"[A-Za-z0-9_\-]+", slug):
        raise ApiError(404, "Unknown dataset.")
    path = DATA_DIR / f"{slug}.json"
    if not path.is_file() or slug == "index":
        raise ApiError(404, f"Unknown dataset '{slug}'.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def health() -> dict:
    with STATE["lock"]:
        used = STATE["jobs_used"]
    return {
        "ok": True,
        "credentials": has_credentials(),
        "runtime": runtime_available(),
        "jobs_used": used,
        "jobs_max": CONFIG["max_jobs"],
        "max_execution_time_s": MAX_EXECUTION_TIME_S,
        "shots": SHOTS,
        "live": True,
    }


def backends() -> list:
    """List IBM backends (mirrors tab_ibm_hardware.py 'Query Active IBM Quantum Backends'). No QPU time."""
    if not has_credentials():
        raise ApiError(503, "No IBM credentials found. Set QISKIT_IBM_TOKEN in .env or save a Qiskit account "
                            "for channel ibm_quantum_platform.")
    try:
        service = make_service()
        out = []
        for b in service.backends():
            st = b.status()
            out.append({"name": str(b.name), "operational": bool(st.operational),
                        "num_qubits": int(b.num_qubits), "pending_jobs": int(st.pending_jobs)})
        return out
    except ApiError:
        raise
    except Exception as e:  # noqa: BLE001
        raise ApiError(502, f"Could not list IBM Quantum backends: {scrub(e)}")


# ------------------------------------------------------------------ live mode
def _live_api():
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    import live_api
    return live_api


def live_call(name: str, *args):
    la = _live_api()
    try:
        out = getattr(la, name)(*args)
    except la.LiveError as e:
        raise ApiError(e.code, e.message, **e.extra)
    if isinstance(out, dict) and isinstance(out.get("error"), str):
        out = {**out, "error": scrub(out["error"])}
    return out


def submit(body: dict) -> dict:
    if not isinstance(body, dict):
        raise ApiError(400, "Body must be a JSON object.")
    if body.get("confirm") is not True:
        raise ApiError(400, 'Submitting uses real QPU minutes: send "confirm": true to proceed.')
    slug, patient = body.get("slug"), body.get("patient")
    if isinstance(patient, bool) or not isinstance(patient, int):
        raise ApiError(400, "'patient' must be an integer index.")
    data = load_dataset(slug)
    tests = data["test"]
    if not 0 <= patient < len(tests):
        raise ApiError(400, f"patient must be between 0 and {len(tests) - 1}.")
    kt = data["qsvc"]["kernel_test"][patient]
    partner = max(range(len(kt)), key=lambda i: kt[i])
    key = f"{slug}|{patient}"

    with STATE["lock"]:
        cached = load_cache().get(key)
        if cached:
            return {"cached": True, **cached}
        active = STATE["active"]
        if active:
            try:
                st = status_name(STATE["jobs"][active]["job"])
            except Exception:
                st = "ERROR"
            if st not in FINAL_STATES:
                raise ApiError(409, "Another QPU job is still in flight; wait for it to finish.", job_id=active)
            STATE["active"] = None
        if STATE["jobs_used"] >= CONFIG["max_jobs"]:
            raise ApiError(429, f"Job budget reached ({CONFIG['max_jobs']} per server run) to protect QPU minutes. "
                                "Restart the server with a higher --max-jobs to allow more.")
        if not has_credentials():
            raise ApiError(503, "No IBM credentials found. Set QISKIT_IBM_TOKEN in .env or save a Qiskit account "
                                "for channel ibm_quantum_platform.")

        try:
            from qiskit import QuantumCircuit
            from qiskit.quantum_info import Statevector
            from qmed.circuits import build_feature_map
            x_p = tests[patient]["angles"]
            x_q = data["train"]["angles"][partner]
            fm = build_feature_map("ZZFeatureMap", 3, 2, "linear")
            bound_p, bound_q = fm.assign_parameters(x_p), fm.assign_parameters(x_q)
            qc = QuantumCircuit(3)
            qc.compose(bound_p, inplace=True)
            qc.compose(bound_q.inverse(), inplace=True)
            qc.measure_all()
            ideal = float(abs(Statevector(bound_p).inner(Statevector(bound_q))) ** 2)
            ideal = min(1.0, max(0.0, ideal))

            service = make_service()
            cands = [b for b in service.backends() if b.status().operational and not b.simulator]
            if not cands:
                raise ApiError(503, "No operational IBM quantum hardware backend is available right now.")
            backend = min(cands, key=lambda b: b.status().pending_jobs)
            pending = int(backend.status().pending_jobs)
            tqc = transpile_circuit(qc, backend)
            job = submit_job(backend, tqc)
            job_id = job.job_id()
        except ApiError:
            raise
        except Exception as e:  # noqa: BLE001
            raise ApiError(502, f"IBM Quantum submission failed: {scrub(e)}")

        submitted_at = time.time()
        STATE["jobs_used"] += 1
        STATE["active"] = job_id
        STATE["services"][job_id] = service
        STATE["jobs"][job_id] = {
            "job": job, "slug": slug, "patient": patient, "partner": partner, "ideal": ideal,
            "backend": backend.name, "pending_jobs": pending, "submitted_at": submitted_at, "result": None,
        }
        return {"job_id": job_id, "backend": backend.name, "pending_jobs": pending, "partner": partner,
                "ideal_fidelity": ideal, "submitted_at": submitted_at}


def extract_counts(result) -> dict:
    pub = result[0]
    bits = getattr(pub.data, "meas", getattr(pub.data, "c", None))
    return dict(bits.get_counts()) if bits is not None else {}


def status(job_id: str) -> dict:
    with STATE["lock"]:
        rec = STATE["jobs"].get(job_id)
        if rec is None:
            raise ApiError(404, "Unknown job id (the server only knows jobs submitted during this run).")
        elapsed = round(time.time() - rec["submitted_at"], 1)
        if rec["result"] is not None:
            return {"job_id": job_id, "status": "DONE", "elapsed_s": elapsed, **rec["result"]}
        try:
            st = status_name(rec["job"])
        except Exception as e:  # noqa: BLE001
            raise ApiError(502, f"Could not read job status: {scrub(e)}")
        out = {"job_id": job_id, "status": st, "elapsed_s": elapsed}
        if st in FINAL_STATES and STATE["active"] == job_id:
            STATE["active"] = None
        if st == "DONE":
            try:
                counts = extract_counts(rec["job"].result())
            except Exception as e:  # noqa: BLE001
                raise ApiError(502, f"Job finished but results could not be read: {scrub(e)}")
            shots = sum(counts.values()) or SHOTS
            hw = counts.get("000", 0) / shots
            res = {"hw_fidelity": hw, "counts": counts, "noise_gap": rec["ideal"] - hw,
                   "backend": rec["backend"], "partner": rec["partner"], "ideal_fidelity": rec["ideal"],
                   "job_id_ibm": job_id, "elapsed_s_total": elapsed}
            rec["result"] = res
            cache = load_cache()
            cache[f"{rec['slug']}|{rec['patient']}"] = {"job_id": job_id, "submitted_at": rec["submitted_at"], **res}
            save_cache(cache)
            out.update(res)
            out.pop("job_id_ibm", None)
        elif st in ("ERROR", "CANCELLED"):
            out["error"] = f"Job ended with status {st}."
        return out


# ------------------------------------------------------------------------ HTTP
class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # no request logging (URLs/bodies stay out of logs)
        pass

    def _send(self, code, obj):
        raw = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def _run(self, fn):
        try:
            out = fn()
            code, obj = out if isinstance(out, tuple) else (200, out)
            self._send(code, obj)
        except ApiError as e:
            self._send(e.code, {"error": scrub(e.message), **e.extra})
        except Exception as e:  # noqa: BLE001
            self._send(500, {"error": scrub(f"Internal error: {e}")})

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/api/qpu/health":
            return self._run(health)
        if u.path == "/api/qpu/status":
            job = (parse_qs(u.query).get("job") or [""])[0]
            return self._run(lambda: status(job))
        if u.path == "/api/qpu/backends":
            return self._run(backends)
        if u.path == "/api/live/train/status":
            run = (parse_qs(u.query).get("run") or [""])[0]
            return self._run(lambda: live_call("train_status", run))
        m = re.fullmatch(r"/api/live/run/([0-9a-f]{1,64})\.json", u.path)
        if m:
            return self._run(lambda: live_call("run_json", m.group(1)))
        self._send(404, {"error": "Not found."})

    def do_POST(self):
        u = urlparse(self.path)
        routes = {"/api/qpu/submit": submit,
                  "/api/live/train": lambda b: live_call("start_train", b),
                  "/api/live/predict": lambda b: live_call("predict", b)}
        fn = routes.get(u.path)
        if fn is None:
            return self._send(404, {"error": "Not found."})
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(min(n, 65536)) or b"{}")
        except Exception:
            return self._send(400, {"error": "Invalid JSON body."})
        self._run(lambda: fn(body))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--max-jobs", type=int, default=3)
    ap.add_argument("--account", default=DEFAULT_ACCOUNT)
    a = ap.parse_args(argv)
    CONFIG["account"], CONFIG["max_jobs"] = a.account, a.max_jobs
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"QPU helper listening on http://127.0.0.1:{a.port}  "
          f"(budget: {a.max_jobs} job(s) this run, {MAX_EXECUTION_TIME_S}s execution cap, {SHOTS} shots, "
          f"1 in flight). Health does not contact IBM. Live mode: /api/live/*.", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
