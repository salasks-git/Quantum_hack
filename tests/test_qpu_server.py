"""Tests for scripts/qpu_server.py. Fakes only: IBM is never contacted."""
import importlib.util
import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib import request as urlreq, error as urlerr

import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("qpu_server", ROOT / "scripts" / "qpu_server.py")
qs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qs)

SECRET = "SECRET-TOKEN-123"
SLUG = "heart_disease_statlog"
BROWSER_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"


class FakeStatus:
    operational = True
    pending_jobs = 4


class FakeBackend:
    def __init__(self, name="fake_qpu", pending=4, sim=False):
        self.name, self.simulator, self._p = name, sim, pending

    def status(self):
        s = FakeStatus()
        s.pending_jobs = self._p
        return s


class FakeBits:
    def get_counts(self):
        return {"000": 512, "001": 256, "111": 256}


class FakeData:
    meas = FakeBits()


class FakeJob:
    def __init__(self):
        self.state = "QUEUED"

    def job_id(self):
        return "fakejob1"

    def status(self):
        return self.state

    def result(self):
        class P:
            data = FakeData()
        return [P()]


class FakeService:
    def __init__(self):
        self.job = FakeJob()
        self.circuits = []

    def backends(self):
        return [FakeBackend("busy", 50), FakeBackend("quiet", 2), FakeBackend("sim", 0, sim=True)]


@pytest.fixture
def env(tmp_path, monkeypatch):
    calls = {"service": 0, "submit": 0}
    svc = FakeService()

    def fake_make_service():
        calls["service"] += 1
        return svc

    def fake_submit(backend, tqc):
        calls["submit"] += 1
        calls["backend"] = backend.name
        return svc.job

    monkeypatch.setattr(qs, "make_service", fake_make_service)
    def fake_transpile(qc, b):
        svc.circuits.append(qc)
        return qc

    monkeypatch.setattr(qs, "transpile_circuit", fake_transpile)
    monkeypatch.setattr(qs, "submit_job", fake_submit)
    monkeypatch.setattr(qs, "get_token", lambda: SECRET)
    monkeypatch.setattr(qs, "CACHE_PATH", tmp_path / "cache.json")
    monkeypatch.setattr(qs, "SUBMIT_LOG_PATH", tmp_path / "submit.log")
    monkeypatch.setitem(qs.CONFIG, "allow_scripted", False)
    monkeypatch.setattr(qs, "QISKIT_ACCOUNTS_FILE", tmp_path / "none.json")
    monkeypatch.setitem(qs.CONFIG, "max_jobs", 3)
    for k, v in (("jobs", {}), ("services", {}), ("jobs_used", 0), ("active", None), ("tokens", {})):
        monkeypatch.setitem(qs.STATE, k, v)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), qs.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    bodies = []

    def call(method, path, body=None, headers=None, raw=None):
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        hdrs = {"Content-Type": "application/json", "User-Agent": BROWSER_UA, "Origin": "http://localhost:5180"}
        for k, v in (headers or {}).items():
            if v is None:
                hdrs.pop(k, None)
            else:
                hdrs[k] = v
        req = urlreq.Request(base + path, data=data, method=method, headers=hdrs)
        try:
            with urlreq.urlopen(req) as r:
                raw, code = r.read(), r.status
        except urlerr.HTTPError as e:
            raw, code = e.read(), e.code
        bodies.append(raw.decode())
        return code, json.loads(raw)

    yield type("E", (), {"call": call, "calls": calls, "svc": svc, "bodies": bodies, "mp": monkeypatch})
    srv.shutdown()
    assert all(SECRET not in b for b in bodies)


def expected_partner(patient):
    d = json.load(open(ROOT / "web/public/data" / f"{SLUG}.json"))
    kt = d["qsvc"]["kernel_test"][patient]
    return max(range(len(kt)), key=lambda i: kt[i])


def token(env):
    code, r = env.call("GET", "/api/qpu/token")
    assert code == 200 and r["expires_in"] > 0
    return r["token"]


def sub(env, patient=0, headers=None, slug=SLUG, **kw):
    h = {"X-QPU-Token": token(env), **(headers or {})}
    return env.call("POST", "/api/qpu/submit", {"slug": slug, "patient": patient, "confirm": True, **kw}, headers=h)


def test_health_no_credentials_and_never_calls_service(env):
    env.mp.setattr(qs, "get_token", lambda: "")
    code, h = env.call("GET", "/api/qpu/health")
    assert code == 200 and h["ok"] is True and h["credentials"] is False
    assert h["jobs_used"] == 0 and h["jobs_max"] == 3 and isinstance(h["runtime"], bool)
    assert env.calls["service"] == 0


def test_health_credentials_from_saved_account_file(env, tmp_path):
    env.mp.setattr(qs, "get_token", lambda: "")
    f = tmp_path / "acc.json"
    f.write_text(json.dumps({"x": {"channel": "ibm_quantum_platform", "token": SECRET}}))
    env.mp.setattr(qs, "QISKIT_ACCOUNTS_FILE", f)
    _, h = env.call("GET", "/api/qpu/health")
    assert h["credentials"] is True and env.calls["service"] == 0


def test_submit_no_credentials_503(env):
    env.mp.setattr(qs, "get_token", lambda: "")
    code, r = sub(env)
    assert code == 503 and ".env" in r["error"] and "QISKIT_IBM_TOKEN" in r["error"]
    assert env.calls["service"] == 0


def test_submit_requires_confirm(env):
    code, _ = env.call("POST", "/api/qpu/submit", {"slug": SLUG, "patient": 0}, headers={"X-QPU-Token": token(env)})
    assert code == 400 and env.calls["service"] == 0


def test_unknown_slug_404(env):
    code, _ = sub(env, 0, slug="nope")
    assert code == 404


def test_submit_ok_and_status_done(env):
    code, r = sub(env, 2)
    assert code == 200 and r["partner"] == expected_partner(2)
    assert 0 <= r["ideal_fidelity"] <= 1 and r["backend"] == "quiet" and r["pending_jobs"] == 2
    assert r["job_id"] == "fakejob1"
    _, s = env.call("GET", "/api/qpu/status?job=fakejob1")
    assert s["status"] == "QUEUED" and "hw_fidelity" not in s
    env.svc.job.state = "DONE"
    _, s = env.call("GET", "/api/qpu/status?job=fakejob1")
    assert s["status"] == "DONE" and s["hw_fidelity"] == 0.5
    assert s["noise_gap"] == pytest.approx(r["ideal_fidelity"] - 0.5) and s["counts"]["000"] == 512
    cache = json.load(open(qs.CACHE_PATH))
    assert cache["version"] == 2 and r["cache_key"].startswith(f"{SLUG}|2|") and len(r["cache_key"].split("|")[2]) == 12
    assert cache["entries"][r["cache_key"]]["hw_fidelity"] == 0.5 and f"{SLUG}|2" not in cache["entries"]


def test_409_while_in_flight(env):
    sub(env, 0)
    code, r = sub(env, 1)
    assert code == 409 and r["job_id"] == "fakejob1" and env.calls["submit"] == 1


def test_429_after_max_jobs(env):
    env.mp.setitem(qs.CONFIG, "max_jobs", 1)
    sub(env, 0)
    env.svc.job.state = "ERROR"
    code, r = sub(env, 1)
    assert code == 429 and "budget" in r["error"].lower() and env.calls["submit"] == 1
    _, h = env.call("GET", "/api/qpu/health")
    assert h["jobs_used"] == 1 and h["jobs_max"] == 1


def test_cached_pair_skips_service(env):
    sub(env, 3)
    env.svc.job.state = "DONE"
    env.call("GET", "/api/qpu/status?job=fakejob1")
    n_service, n_submit = env.calls["service"], env.calls["submit"]
    code, r = sub(env, 3, force=True)
    assert code == 200 and r["cached"] is True and r["hw_fidelity"] == 0.5
    assert env.calls["service"] == n_service and env.calls["submit"] == n_submit


def test_token_never_leaks_from_exceptions(env):
    def boom():
        raise RuntimeError(f"auth failed for token {SECRET} and key abcdefghijklmnopqrstuvwxyz0123456789")
    env.mp.setattr(qs, "make_service", boom)
    code, r = sub(env)
    assert code == 502 and SECRET not in r["error"] and "abcdefghijklmnopqrstuvwxyz" not in r["error"]
    assert env.calls["submit"] == 0
    _, h = env.call("GET", "/api/qpu/health")
    assert h["jobs_used"] == 0


# ------------------------------------------------- config-aware circuit + cache (C1 / C1-ext)
def load_static(slug=SLUG):
    return json.load(open(ROOT / "web/public/data" / f"{slug}.json"))


def scaled_state(settings, x):
    from qiskit.quantum_info import Statevector
    from qmed.circuits import build_feature_map, scale_feature_map
    fm = scale_feature_map(build_feature_map(settings["feature_map"], settings["n_qubits"], settings["reps"],
                                             settings["entanglement"]), settings["kernel_bandwidth"])
    return Statevector(fm.assign_parameters(x))


def unmeasured(qc):
    from qiskit import QuantumCircuit
    out = QuantumCircuit(qc.num_qubits)
    for inst in qc.data:
        if inst.operation.name not in ("measure", "barrier"):
            out.append(inst.operation, [out.qubits[qc.find_bit(q).index] for q in inst.qubits])
    return out


def test_static_circuit_follows_settings_and_bandwidth(env):
    from qiskit.quantum_info import Statevector
    d = load_static()
    st = d["settings"]
    assert st["kernel_bandwidth"] == 0.1
    p = 16
    code, r = sub(env, p)
    assert code == 200
    partner = expected_partner(p)
    assert r["partner"] == partner
    qc = env.svc.circuits[-1]
    assert qc.num_qubits == st["n_qubits"]
    # ideal fidelity == the exported (bandwidth-scaled) kernel entry for the same pair
    assert abs(r["ideal_fidelity"] - d["qsvc"]["kernel_test"][p][partner]) < 1e-3
    # circuit = fm_s(x_p) . fm_s(x_q)^-1 with the scaled map: applied to fm_s(x_q)|0> it gives fm_s(x_p)|0>... via state check
    sv = Statevector(unmeasured(qc))
    psi_p = scaled_state(st, d["test"][p]["angles"])
    psi_q = scaled_state(st, d["train"]["angles"][partner])
    # <0|U U'^dag ... > : amplitude of |0..0> equals <psi_q|psi_p>
    assert abs(abs(sv.data[0]) ** 2 - abs(psi_q.inner(psi_p)) ** 2) < 1e-9
    # and it is NOT the unscaled (s = 1.0) circuit
    s1 = {**st, "kernel_bandwidth": 1.0}
    unscaled = abs(scaled_state(s1, d["train"]["angles"][partner]).inner(scaled_state(s1, d["test"][p]["angles"]))) ** 2
    assert abs(unscaled - r["ideal_fidelity"]) > 0.05
    assert r["settings"]["kernel_bandwidth"] == 0.1 and r["settings"]["n_qubits"] == 3


def fake_run(n=4, fm="PauliFeatureMap", ent="full", reps=1, s=0.3, n_train=6, n_test=3, slug=SLUG):
    """A synthetic live-run JSON whose kernel_test is computed with its own scaled map."""
    import numpy as np
    rng = np.random.default_rng(7)
    st = {"n_qubits": n, "feature_map": fm, "reps": reps, "entanglement": ent, "kernel_bandwidth": s,
          "vqc_iter": 15, "seed": 42, "max_samples": 60}
    tr = rng.uniform(0, np.pi, (n_train, n)).round(4).tolist()
    te = rng.uniform(0, np.pi, (n_test, n)).round(4).tolist()
    kt = [[round(float(abs(scaled_state(st, b).inner(scaled_state(st, a))) ** 2), 4) for b in tr] for a in te]
    return {"slug": slug, "settings": st, "custom_run": True, "run_id": "abc123",
            "train": {"angles": tr}, "test": [{"label": i % 2, "angles": a} for i, a in enumerate(te)],
            "qsvc": {"kernel_test": kt}}


def test_run_id_uses_that_runs_settings_and_pair(env):
    run = fake_run()
    seen = []
    env.mp.setattr(qs, "live_call", lambda name, *a: (seen.append((name, a)), run)[1])
    code, r = sub(env, 1, run_id="abc123")
    assert code == 200, r
    assert seen == [("run_json", ("abc123",))]
    kt = run["qsvc"]["kernel_test"][1]
    partner = max(range(len(kt)), key=lambda i: kt[i])
    assert r["partner"] == partner and abs(r["ideal_fidelity"] - kt[partner]) < 1e-3
    assert env.svc.circuits[-1].num_qubits == 4
    assert r["settings"] == {"feature_map": "PauliFeatureMap", "n_qubits": 4, "reps": 1, "entanglement": "full",
                             "kernel_bandwidth": 0.3}
    assert r["run_id"] == "abc123"
    # different run => different cache key than the static data for the same slug/patient
    plan_static = qs.plan_run({"slug": SLUG, "patient": 1})
    assert plan_static["key"] != r["cache_key"]


def test_run_id_wrong_slug_400_unknown_404_no_service(env):
    env.mp.setattr(qs, "live_call", lambda name, *a: fake_run(slug="parkinsons_vocal"))
    code, _ = sub(env, 0, run_id="abc123")
    assert code == 400

    def unknown(name, *a):
        raise qs.ApiError(404, "Unknown run id")
    env.mp.setattr(qs, "live_call", unknown)
    code, _ = sub(env, 0, run_id="deadbeef")
    assert code == 404
    code, _ = sub(env, 0, run_id="../etc")
    assert code == 400
    assert env.calls["service"] == 0


def test_kernel_settings_mismatch_refused_without_ibm(env):
    run = fake_run()
    run["settings"]["kernel_bandwidth"] = 1.0  # kernel was computed at s = 0.3
    env.mp.setattr(qs, "live_call", lambda name, *a: run)
    code, r = sub(env, 0, run_id="abc123")
    assert code == 500 and "mismatch" in r["error"] and env.calls["service"] == 0


def test_cache_key_includes_settings_hash():
    d = load_static()
    a = qs.settings_hash(qs.run_settings(d), 3, [0.1, 0.2, 0.3], [0.3, 0.2, 0.1])
    d2 = json.loads(json.dumps(d))
    d2["settings"]["kernel_bandwidth"] = 0.5
    b = qs.settings_hash(qs.run_settings(d2), 3, [0.1, 0.2, 0.3], [0.3, 0.2, 0.1])
    c = qs.settings_hash(qs.run_settings(d), 3, [0.1, 0.2, 0.3], [0.3, 0.2, 0.2])
    assert len({a, b, c}) == 3 and all(len(x) == 12 for x in (a, b, c))


def test_legacy_cache_entries_are_not_served(env):
    old = {f"{SLUG}|16": {"job_id": "dbjgsgpffo6c73b9kkeg", "hw_fidelity": 0.509, "ideal_fidelity": 0.542,
                          "partner": 13, "backend": "ibm_old"}}
    qs.CACHE_PATH.write_text(json.dumps(old))
    code, r = sub(env, 16)
    assert code == 200 and not r.get("cached") and r["job_id"] == "fakejob1" and env.calls["submit"] == 1
    assert r["partner"] == expected_partner(16) != 13
    cache = qs.load_cache()
    assert cache["entries"] == {}
    leg = cache["legacy"][f"{SLUG}|16"]
    assert leg["legacy"] is True and leg["label"] == "old config (s = 1.0)" and leg["settings"]["kernel_bandwidth"] == 1.0


def test_changing_bandwidth_is_a_cache_miss(env, monkeypatch):
    sub(env, 3)
    env.svc.job.state = "DONE"
    env.call("GET", "/api/qpu/status?job=fakejob1")
    code, r = sub(env, 3)
    assert r["cached"] is True
    real = qs.load_dataset

    def other_bw(slug):
        d = real(slug)
        d["settings"]["kernel_bandwidth"] = 1.0
        return d
    monkeypatch.setattr(qs, "load_dataset", other_bw)
    plan = None
    try:
        plan = qs.plan_run({"slug": SLUG, "patient": 3})
    except qs.ApiError as e:  # kernel was exported at 0.1, so s = 1.0 is refused before any key is served
        assert "mismatch" in e.message
    assert plan is None or plan["key"] != r["cache_key"]


@pytest.mark.parametrize("n", [2, 4])
def test_status_counts_all_zeros_for_n_qubits(env, n):
    class Bits:
        def get_counts(self):
            return {"0" * n: 300, "1" * n: 724}

    class Job(FakeJob):
        def result(self):
            class P:
                class data:
                    meas = Bits()
            return [P()]
    env.svc.job = Job()
    env.svc.job.state = "DONE"
    qs.STATE["jobs"]["J"] = {"job": env.svc.job, "slug": SLUG, "patient": 0, "partner": 1, "ideal": 0.9,
                             "n_qubits": n, "key": f"{SLUG}|0|abc", "meta": {}, "backend": "b",
                             "pending_jobs": 0, "submitted_at": 0, "result": None}
    _, s = env.call("GET", "/api/qpu/status?job=J")
    assert s["hw_fidelity"] == pytest.approx(300 / 1024)


# ------------------------------------------------------------------ submit hardening (I3)
def test_no_token_no_submit(env):
    code, r = env.call("POST", "/api/qpu/submit", {"slug": SLUG, "patient": 0, "confirm": True})
    assert code == 403 and "token" in r["error"].lower() and env.calls["service"] == 0


def test_token_is_single_use(env):
    t = token(env)
    body = {"slug": SLUG, "patient": 0, "confirm": True}
    code, _ = env.call("POST", "/api/qpu/submit", body, headers={"X-QPU-Token": t})
    assert code == 200
    env.svc.job.state = "ERROR"
    code, _ = env.call("POST", "/api/qpu/submit", body, headers={"X-QPU-Token": t})
    assert code == 403 and env.calls["submit"] == 1


def test_token_consumed_even_when_request_fails(env):
    t = token(env)
    code, _ = env.call("POST", "/api/qpu/submit", {"slug": SLUG, "patient": 0}, headers={"X-QPU-Token": t})
    assert code == 400  # no confirm
    code, _ = env.call("POST", "/api/qpu/submit", {"slug": SLUG, "patient": 0, "confirm": True},
                       headers={"X-QPU-Token": t})
    assert code == 403 and env.calls["submit"] == 0


def test_token_expires(env, monkeypatch):
    t = token(env)
    real = qs.time.time
    monkeypatch.setattr(qs.time, "time", lambda: real() + qs.TOKEN_TTL_S + 1)
    code, _ = env.call("POST", "/api/qpu/submit", {"slug": SLUG, "patient": 0, "confirm": True},
                       headers={"X-QPU-Token": t})
    assert code == 403 and env.calls["service"] == 0


@pytest.mark.parametrize("headers,code", [
    ({"Origin": "https://evil.example"}, 403),
    ({"Origin": None, "Referer": "https://evil.example/page"}, 403),
    ({"Sec-Fetch-Site": "cross-site"}, 403),
    ({"Content-Type": "text/plain"}, 415),
    ({"User-Agent": BROWSER_UA.replace("Chrome/", "HeadlessChrome/")}, 403),
    ({"User-Agent": "curl/8.4.0"}, 403),
    ({"User-Agent": "Python-urllib/3.13"}, 403),
])
def test_submit_refusals(env, headers, code):
    t = token(env)
    c, r = env.call("POST", "/api/qpu/submit", {"slug": SLUG, "patient": 0, "confirm": True},
                    headers={"X-QPU-Token": t, **headers})
    assert c == code, r
    assert env.calls["service"] == 0 and env.calls["submit"] == 0


def test_token_endpoint_refuses_headless_and_foreign_origin(env):
    c, _ = env.call("GET", "/api/qpu/token", headers={"User-Agent": "Mozilla/5.0 HeadlessChrome/156"})
    assert c == 403
    c, _ = env.call("GET", "/api/qpu/token", headers={"Origin": "https://evil.example"})
    assert c == 403
    c, _ = env.call("GET", "/api/qpu/backends", headers={"Origin": "https://evil.example"})
    assert c == 403 and env.calls["service"] == 0


def test_local_origins_accepted(env):
    for origin in ("http://127.0.0.1:4173", "http://localhost:5180"):
        env.svc.job.state = "QUEUED"
        qs.STATE["active"] = None
        code, _ = sub(env, 0, headers={"Origin": origin, "Referer": origin + "/?step=6"})
        assert code in (200,), origin


def test_allow_scripted_flag(env):
    env.mp.setitem(qs.CONFIG, "allow_scripted", True)
    code, _ = sub(env, 0, headers={"User-Agent": "curl/8.4.0"})
    assert code == 200


def test_every_attempt_is_audited_without_secrets(env):
    t = token(env)
    env.call("POST", "/api/qpu/submit", {"slug": SLUG, "patient": 0, "confirm": True},
             headers={"X-QPU-Token": t, "User-Agent": "curl/8.4.0"})
    code, r = sub(env, 2)
    assert code == 200
    lines = [json.loads(x) for x in qs.SUBMIT_LOG_PATH.read_text().splitlines()]
    assert len(lines) == 2
    assert lines[0]["code"] == 403 and lines[0]["user_agent"] == "curl/8.4.0" and lines[0]["token_present"] is True
    assert lines[1]["code"] == 200 and lines[1]["slug"] == SLUG and lines[1]["patient"] == 2
    assert lines[1]["settings_hash"] == r["cache_key"].split("|")[2] and lines[1]["job_id"] == "fakejob1"
    assert lines[1]["user_agent"] == BROWSER_UA and lines[1]["token_valid"] is True
    raw = qs.SUBMIT_LOG_PATH.read_text()
    assert t not in raw and SECRET not in raw


def test_default_max_jobs_is_3_and_health_offline(env):
    import inspect
    assert 'ap.add_argument("--max-jobs", type=int, default=3)' in inspect.getsource(qs.main)
    _, h = env.call("GET", "/api/qpu/health", headers={"User-Agent": None, "Origin": None})
    assert h["ok"] is True and env.calls["service"] == 0
