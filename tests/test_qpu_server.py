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
    monkeypatch.setattr(qs, "transpile_circuit", lambda qc, b: qc)
    monkeypatch.setattr(qs, "submit_job", fake_submit)
    monkeypatch.setattr(qs, "get_token", lambda: SECRET)
    monkeypatch.setattr(qs, "CACHE_PATH", tmp_path / "cache.json")
    monkeypatch.setattr(qs, "QISKIT_ACCOUNTS_FILE", tmp_path / "none.json")
    monkeypatch.setitem(qs.CONFIG, "max_jobs", 3)
    for k, v in (("jobs", {}), ("services", {}), ("jobs_used", 0), ("active", None)):
        monkeypatch.setitem(qs.STATE, k, v)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), qs.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    bodies = []

    def call(method, path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urlreq.Request(base + path, data=data, method=method, headers={"Content-Type": "application/json"})
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


def sub(env, patient=0, **kw):
    return env.call("POST", "/api/qpu/submit", {"slug": SLUG, "patient": patient, "confirm": True, **kw})


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
    code, _ = env.call("POST", "/api/qpu/submit", {"slug": SLUG, "patient": 0})
    assert code == 400 and env.calls["service"] == 0


def test_unknown_slug_404(env):
    code, _ = env.call("POST", "/api/qpu/submit", {"slug": "nope", "patient": 0, "confirm": True})
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
    assert json.load(open(qs.CACHE_PATH))[f"{SLUG}|2"]["hw_fidelity"] == 0.5


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
