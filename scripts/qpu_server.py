#!/usr/bin/env python3
"""Local helper server so the static 3D tour can submit ONE real IBM Quantum job.

Mirrors qmed/ui/tab_ibm_hardware.py: circuit fm(x_p) o fm(x_q)^-1 + measure_all,
least-busy operational non-simulator backend, transpile optimization_level=2,
SamplerV2 with 1024 shots. Binds 127.0.0.1 only. Never returns/logs credentials.

The circuit uses the ACTIVE pipeline config (static <slug>.json settings, or a live Custom run's via
run_id): feature map, qubits, reps, entanglement and the same kernel-bandwidth scaling the exported
QSVC kernel uses (qmed.circuits.scale_feature_map). Partner = argmax of that data's kernel_test row.

Free-tier safeguards: explicit confirm flag, one-time click token (GET /api/qpu/token, single use,
short-lived), browser-only clients (UA / Origin / Referer / Sec-Fetch-Site / Content-Type checks),
audit log of every submit attempt (web/.qpu-submit.log), max_execution_time cap, one job in flight,
per-run job budget, persistent result cache keyed by (slug, patient, settings hash).

Also serves the tour's live mode (scripts/live_api.py: retrain with settings, custom-patient
predict; never contacts IBM) and GET /api/qpu/backends (lists IBM backends, no QPU time).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
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
SUBMIT_LOG_PATH = ROOT / "web" / ".qpu-submit.log"
QISKIT_ACCOUNTS_FILE = Path.home() / ".qiskit" / "qiskit-ibm.json"
DEFAULT_ACCOUNT = "default-ibm-quantum-platform"
SHOTS = 1024
MAX_EXECUTION_TIME_S = 60
FINAL_STATES = {"DONE", "ERROR", "CANCELLED"}
TOKEN_TTL_S = 30
MAX_TOKENS = 16
CACHE_VERSION = 2
LEGACY_LABEL = "old config (s = 1.0)"
RUN_ID_RE = re.compile(r"[0-9a-f]{1,64}")

CONFIG = {"account": DEFAULT_ACCOUNT, "max_jobs": 3, "allow_scripted": False}
STATE = {"lock": threading.RLock(), "jobs": {}, "services": {}, "jobs_used": 0, "active": None, "tokens": {}}


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


def migrate_cache(d) -> dict:
    """v2 layout: {"version": 2, "entries": {slug|patient|hash: rec}, "legacy": {slug|patient: rec}}.
    Pre-v2 files were flat {slug|patient: rec} made with the unscaled (s = 1.0) 3-qubit ZZ/reps 2/linear
    circuit: they move to "legacy", labelled, and are never served as current results."""
    if not isinstance(d, dict):
        return {"version": CACHE_VERSION, "entries": {}, "legacy": {}}
    if d.get("version") == CACHE_VERSION:
        return {"version": CACHE_VERSION, "entries": dict(d.get("entries") or {}), "legacy": dict(d.get("legacy") or {})}
    legacy = {}
    for k, v in d.items():
        if isinstance(v, dict):
            legacy[k] = {**v, "legacy": True, "label": LEGACY_LABEL,
                         "settings": {"feature_map": "ZZFeatureMap", "n_qubits": 3, "reps": 2,
                                      "entanglement": "linear", "kernel_bandwidth": 1.0}}
    return {"version": CACHE_VERSION, "entries": {}, "legacy": legacy}


def load_cache() -> dict:
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return migrate_cache(json.load(f))
    except Exception:
        return migrate_cache(None)


def save_cache(cache: dict) -> None:
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=1)
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
    used = STATE["jobs_used"]  # plain int read: never waits on a submit holding the lock
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


def resolve_data(slug, run_id):
    """Active data for a submit: a live Custom run's JSON (run_id) or the static <slug>.json."""
    if run_id is None:
        return load_dataset(slug)
    if not isinstance(run_id, str) or not RUN_ID_RE.fullmatch(run_id):
        raise ApiError(400, "'run_id' must be a live run id.")
    data = live_call("run_json", run_id)
    if data.get("slug") != slug:
        raise ApiError(400, "run_id belongs to a different dataset.")
    return data


def run_settings(data: dict) -> dict:
    s = data.get("settings") or {}
    return {"feature_map": str(s.get("feature_map", "ZZFeatureMap")), "n_qubits": int(s.get("n_qubits", 3)),
            "reps": int(s.get("reps", 2)), "entanglement": str(s.get("entanglement", "linear")),
            "kernel_bandwidth": float(s.get("kernel_bandwidth", 1.0))}


def feature_map_for(settings: dict):
    """Same circuit (incl. bandwidth scaling) as qmed.models.build_quantum_kernel used for the exported kernel."""
    from qmed.circuits import build_feature_map, scale_feature_map
    fm = build_feature_map(settings["feature_map"], settings["n_qubits"], settings["reps"], settings["entanglement"])
    return scale_feature_map(fm, settings["kernel_bandwidth"])


def settings_hash(settings: dict, partner: int, x_p, x_q) -> str:
    blob = {"fm": settings["feature_map"], "n": settings["n_qubits"], "reps": settings["reps"],
            "ent": settings["entanglement"], "s": settings["kernel_bandwidth"], "partner": partner,
            "xp": [float(v) for v in x_p], "xq": [float(v) for v in x_q]}
    return hashlib.sha1(json.dumps(blob, sort_keys=True).encode()).hexdigest()[:12]


def plan_run(body: dict) -> dict:
    """Validate the request and work out pair, settings, cache key and ideal fidelity. Never contacts IBM."""
    slug, patient, run_id = body.get("slug"), body.get("patient"), body.get("run_id")
    if isinstance(patient, bool) or not isinstance(patient, int):
        raise ApiError(400, "'patient' must be an integer index.")
    data = resolve_data(slug, run_id)
    tests = data["test"]
    if not 0 <= patient < len(tests):
        raise ApiError(400, f"patient must be between 0 and {len(tests) - 1}.")
    if tests[patient].get("label") is None:
        raise ApiError(400, "Custom (slider) patients cannot be sent to the QPU.")
    st = run_settings(data)
    kt = data["qsvc"]["kernel_test"][patient]
    partner = max(range(len(kt)), key=lambda i: kt[i])
    x_p = tests[patient]["angles"]
    x_q = data["train"]["angles"][partner]
    if len(x_p) != st["n_qubits"] or len(x_q) != st["n_qubits"]:
        raise ApiError(500, "kernel/settings mismatch (angle count != n_qubits).")
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    fm = feature_map_for(st)
    bound_p, bound_q = fm.assign_parameters(x_p), fm.assign_parameters(x_q)
    ideal = float(abs(Statevector(bound_p).inner(Statevector(bound_q))) ** 2)
    ideal = min(1.0, max(0.0, ideal))
    if abs(ideal - float(kt[partner])) > 1e-3:
        raise ApiError(500, f"kernel/settings mismatch: ideal {ideal:.4f} vs exported K {float(kt[partner]):.4f}.")
    qc = QuantumCircuit(st["n_qubits"])
    qc.compose(bound_p, inplace=True)
    qc.compose(bound_q.inverse(), inplace=True)
    qc.measure_all()
    h = settings_hash(st, partner, x_p, x_q)
    return {"slug": slug, "patient": patient, "run_id": run_id, "settings": st, "partner": partner,
            "ideal": ideal, "qc": qc, "hash": h, "key": f"{slug}|{patient}|{h}"}


def submit(body: dict, audit: dict | None = None) -> dict:
    audit = audit if audit is not None else {}
    if not isinstance(body, dict):
        raise ApiError(400, "Body must be a JSON object.")
    if body.get("confirm") is not True:
        raise ApiError(400, 'Submitting uses real QPU minutes: send "confirm": true to proceed.')
    plan = plan_run(body)
    audit["settings_hash"] = plan["hash"]
    key, st = plan["key"], plan["settings"]
    meta = {"settings": st, "cache_key": key, "run_id": plan["run_id"]}

    with STATE["lock"]:
        cached = load_cache()["entries"].get(key)
        if cached:
            return {"cached": True, **cached, **meta}
        active = STATE["active"]
        if active:
            try:
                s = status_name(STATE["jobs"][active]["job"])
            except Exception:
                s = "ERROR"
            if s not in FINAL_STATES:
                raise ApiError(409, "Another QPU job is still in flight; wait for it to finish.", job_id=active)
            STATE["active"] = None
        if STATE["jobs_used"] >= CONFIG["max_jobs"]:
            raise ApiError(429, f"Job budget reached ({CONFIG['max_jobs']} per server run) to protect QPU minutes. "
                                "Restart the server with a higher --max-jobs to allow more.")
        if not has_credentials():
            raise ApiError(503, "No IBM credentials found. Set QISKIT_IBM_TOKEN in .env or save a Qiskit account "
                                "for channel ibm_quantum_platform.")

        try:
            service = make_service()
            cands = [b for b in service.backends() if b.status().operational and not b.simulator]
            if not cands:
                raise ApiError(503, "No operational IBM quantum hardware backend is available right now.")
            backend = min(cands, key=lambda b: b.status().pending_jobs)
            pending = int(backend.status().pending_jobs)
            tqc = transpile_circuit(plan["qc"], backend)
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
            "job": job, "slug": plan["slug"], "patient": plan["patient"], "partner": plan["partner"],
            "ideal": plan["ideal"], "n_qubits": st["n_qubits"], "key": key, "meta": meta,
            "backend": backend.name, "pending_jobs": pending, "submitted_at": submitted_at, "result": None,
        }
        audit["job_id"] = job_id
        return {"job_id": job_id, "backend": backend.name, "pending_jobs": pending, "partner": plan["partner"],
                "ideal_fidelity": plan["ideal"], "submitted_at": submitted_at, **meta}


# ------------------------------------------------------------ submit hardening
LOCAL_HOSTS = {"localhost", "127.0.0.1", "[::1]", "::1"}


def _host_of(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower()
    except Exception:
        return ""


def check_client(headers, *, need_json: bool, browser_only: bool = True) -> None:
    """Only the tour page in a real (non-headless) local browser may arm or submit a QPU run.
    browser_only=False (backend list: no QPU time) keeps only the cross-site checks."""
    ua = headers.get("User-Agent") or ""
    if browser_only and not CONFIG["allow_scripted"]:
        if "headless" in ua.lower():
            raise ApiError(403, "Refused: headless browsers cannot submit QPU jobs.")
        if not ua.startswith("Mozilla/"):
            raise ApiError(403, "Refused: QPU jobs can only be submitted from the tour in a browser.")
    if (headers.get("Sec-Fetch-Site") or "").lower() == "cross-site":
        raise ApiError(403, "Refused: cross-site request.")
    for h in ("Origin", "Referer"):
        v = headers.get(h)
        if v is not None and _host_of(v) not in LOCAL_HOSTS:
            raise ApiError(403, f"Refused: {h} must be localhost / 127.0.0.1.")
    if need_json and (headers.get("Content-Type") or "").split(";")[0].strip().lower() != "application/json":
        raise ApiError(415, "Content-Type must be application/json.")


def issue_token() -> dict:
    now = time.time()
    with STATE["lock"]:
        toks = STATE["tokens"]
        for k in [k for k, exp in toks.items() if exp <= now]:
            toks.pop(k, None)
        while len(toks) >= MAX_TOKENS:
            toks.pop(min(toks, key=toks.get), None)
        tok = secrets.token_hex(16)
        toks[tok] = now + TOKEN_TTL_S
    return {"token": tok, "expires_in": TOKEN_TTL_S}


def consume_token(tok) -> bool:
    """Single use: the token is deleted on first presentation, whatever happens next."""
    if not isinstance(tok, str) or not tok:
        return False
    with STATE["lock"]:
        exp = STATE["tokens"].pop(tok, None)
    return exp is not None and exp > time.time()


def write_audit(entry: dict) -> None:
    try:
        SUBMIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(SUBMIT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass
    print(f"[qpu submit] {entry.get('time')} {entry.get('slug')}#{entry.get('patient')} "
          f"run={entry.get('run_id')} hash={entry.get('settings_hash')} -> {entry.get('code')} "
          f"{entry.get('reason') or entry.get('job_id') or ''} ua={str(entry.get('user_agent'))[:60]!r}",
          file=sys.stderr, flush=True)


def handle_submit(headers, raw: bytes, remote: str) -> tuple:
    """POST /api/qpu/submit with audit logging. Never auto-retries; one call = at most one IBM job."""
    entry = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "slug": None, "patient": None, "run_id": None,
             "settings_hash": None, "confirm": None, "token_present": False, "token_valid": False,
             "user_agent": headers.get("User-Agent"), "origin": headers.get("Origin"),
             "referer": headers.get("Referer"), "host": headers.get("Host"),
             "sec_fetch_site": headers.get("Sec-Fetch-Site"), "content_type": headers.get("Content-Type"),
             "remote": remote}
    try:
        try:
            body = json.loads(raw or b"{}")
        except Exception:
            body = None
        if isinstance(body, dict):
            entry.update(slug=str(body.get("slug"))[:80], patient=body.get("patient") if isinstance(body.get("patient"), int) else None,
                         run_id=str(body.get("run_id"))[:64] if body.get("run_id") is not None else None,
                         confirm=body.get("confirm") is True)
        tok = headers.get("X-QPU-Token")
        entry["token_present"] = bool(tok)
        valid = consume_token(tok)  # consumed before any other check: one token, one attempt
        entry["token_valid"] = valid
        check_client(headers, need_json=True)
        if body is None:
            raise ApiError(400, "Invalid JSON body.")
        if not valid:
            raise ApiError(403, "Missing or expired click token: press Run and confirm again.")
        code, obj = 200, submit(body, entry)
        entry["reason"] = "cached" if obj.get("cached") else "submitted"
    except ApiError as e:
        code, obj = e.code, {"error": scrub(e.message), **e.extra}
        entry["reason"] = scrub(e.message)[:200]
    except Exception as e:  # noqa: BLE001
        code, obj = 500, {"error": scrub(f"Internal error: {e}")}
        entry["reason"] = obj["error"][:200]
    entry["code"] = code
    write_audit(entry)
    return code, obj


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
            hw = counts.get("0" * int(rec.get("n_qubits", 3)), 0) / shots
            res = {"hw_fidelity": hw, "counts": counts, "noise_gap": rec["ideal"] - hw,
                   "backend": rec["backend"], "partner": rec["partner"], "ideal_fidelity": rec["ideal"],
                   "job_id_ibm": job_id, "elapsed_s_total": elapsed, **rec.get("meta", {})}
            rec["result"] = res
            cache = load_cache()
            cache["entries"][rec["key"]] = {"job_id": job_id, "submitted_at": rec["submitted_at"],
                                            "key_version": CACHE_VERSION, **res}
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
            return self._run(lambda: (check_client(self.headers, need_json=False, browser_only=False), backends())[1])
        if u.path == "/api/qpu/token":
            return self._run(lambda: (check_client(self.headers, need_json=False), issue_token())[1])
        if u.path == "/api/live/train/status":
            run = (parse_qs(u.query).get("run") or [""])[0]
            return self._run(lambda: live_call("train_status", run))
        m = re.fullmatch(r"/api/live/run/([0-9a-f]{1,64})\.json", u.path)
        if m:
            return self._run(lambda: live_call("run_json", m.group(1)))
        self._send(404, {"error": "Not found."})

    def do_POST(self):
        u = urlparse(self.path)
        if u.path == "/api/qpu/submit":
            try:
                n = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(min(n, 65536))
            except Exception:
                raw = b""
            code, obj = handle_submit(self.headers, raw, self.client_address[0])
            return self._send(code, obj)
        routes = {"/api/live/train": lambda b: live_call("start_train", b),
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
    ap.add_argument("--allow-scripted-submit", action="store_true",
                    help="allow non-browser / headless clients to arm and submit (off by default)")
    a = ap.parse_args(argv)
    CONFIG["account"], CONFIG["max_jobs"] = a.account, a.max_jobs
    CONFIG["allow_scripted"] = bool(a.allow_scripted_submit)
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
