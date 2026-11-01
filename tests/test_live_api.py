"""Tests for the tour's live mode (scripts/live_api.py via scripts/qpu_server.py routes).

Fakes only: IBM is never contacted. Training tests use the small settings
(vertebral_column, max_samples=60, vqc_iter=15, grid_size=6). With the exact statevector
kernel every training here takes well under 60 s, so no test is marked slow.
"""
import importlib.util
import json
import math
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib import error as urlerr, request as urlreq

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("qpu_server_live", ROOT / "scripts" / "qpu_server.py")
qs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qs)

import live_api as la  # noqa: E402

SECRET = "SECRET-TOKEN-456"
SLUG = "vertebral_column"
SMALL = {"slug": SLUG, "max_samples": 60, "vqc_iter": 15}
MODELS = ("linear", "rbf", "qsvc", "vqc", "hybrid")


@pytest.fixture(scope="module")
def srv():
    mp = pytest.MonkeyPatch()
    mp.setattr(qs, "get_token", lambda: SECRET)
    mp.setitem(la.CONFIG, "grid_size", 6)
    mp.setattr(la, "RUNS", {})
    server = ThreadingHTTPServer(("127.0.0.1", 0), qs.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    bodies = []

    def call(method, path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urlreq.Request(base + path, data=data, method=method, headers={"Content-Type": "application/json"})
        try:
            with urlreq.urlopen(req, timeout=120) as r:
                raw, code = r.read(), r.status
        except urlerr.HTTPError as e:
            raw, code = e.read(), e.code
        bodies.append(raw.decode())
        return code, json.loads(raw)

    def wait(run_id, timeout=180):
        t0 = time.time()
        seen = set()
        while time.time() - t0 < timeout:
            code, s = call("GET", f"/api/live/train/status?run={run_id}")
            assert code == 200 and 0 <= s["progress"] <= 1 and s["elapsed_s"] >= 0
            seen.add(s["stage"])
            if s["state"] in ("done", "error"):
                return s, seen
            time.sleep(0.5)
        raise AssertionError(f"run {run_id} did not finish within {timeout}s")

    yield type("S", (), {"call": call, "wait": wait, "bodies": bodies, "mp": mp})
    server.shutdown()
    mp.undo()
    assert all(SECRET not in b for b in bodies)


def train(srv, **body):
    code, r = srv.call("POST", "/api/live/train", body)
    assert code == 200, r
    s, _ = srv.wait(r["run_id"])
    assert s["state"] == "done", s
    code, js = srv.call("GET", f"/api/live/run/{r['run_id']}.json")
    assert code == 200
    return r["run_id"], js


@pytest.fixture(scope="module")
def small_run(srv):
    """Small n=3 run, also checking 404-before-done, dedupe and 409 while it trains."""
    code, r = srv.call("POST", "/api/live/train", SMALL)
    assert code == 200 and r["state"] == "queued"
    rid = r["run_id"]
    assert rid == la.make_run_id(SLUG, {**la.DEFAULTS, "max_samples": 60, "vqc_iter": 15})
    code, _ = srv.call("GET", f"/api/live/run/{rid}.json")
    assert code == 404
    code, again = srv.call("POST", "/api/live/train", dict(SMALL))
    assert code == 200 and again["run_id"] == rid
    code, busy = srv.call("POST", "/api/live/train", {**SMALL, "reps": 1})
    assert code == 409 and busy["run_id"] == rid and "error" in busy
    s, stages = srv.wait(rid)
    assert s["state"] == "done" and s["progress"] == 1.0 and s["stage"] == "done"
    assert stages & {"QSVC kernel", "VQC training"}
    code, js = srv.call("GET", f"/api/live/run/{rid}.json")
    assert code == 200
    return rid, js


# ------------------------------------------------------------ fast tests
def test_health_has_live(srv):
    code, h = srv.call("GET", "/api/qpu/health")
    assert code == 200 and h["live"] is True


@pytest.mark.parametrize("patch", [
    {"max_samples": 70}, {"max_samples": 200}, {"max_samples": 40}, {"max_samples": "100"},
    {"n_qubits": 5}, {"n_qubits": 1}, {"n_qubits": True},
    {"feature_map": "PauliFeatureMap"}, {"reps": 0}, {"reps": 4}, {"reps": 2.0},
    {"entanglement": "ring"}, {"vqc_iter": 16}, {"vqc_iter": 80}, {"vqc_iter": 10}, {"bogus": 1},
    {"kernel_bandwidth": 0.15}, {"kernel_bandwidth": 2.0}, {"kernel_bandwidth": "0.1"},
    {"kernel_bandwidth": True}, {"kernel_bandwidth": None},
])
def test_invalid_settings_400(srv, patch):
    code, r = srv.call("POST", "/api/live/train", {"slug": SLUG, **patch})
    assert code == 400 and "error" in r


def test_unknown_slug_and_unknown_run(srv):
    assert srv.call("POST", "/api/live/train", {"slug": "nope"})[0] == 404
    assert srv.call("GET", "/api/live/train/status?run=abc123")[0] == 404
    assert srv.call("GET", "/api/live/run/abc123.json")[0] == 404
    assert srv.call("POST", "/api/live/predict", {"run_id": "abc123", "angles": [1, 1, 1]})[0] == 404
    assert srv.call("POST", "/api/live/predict", {"angles": [1, 1, 1]})[0] == 400
    assert srv.call("POST", "/api/live/predict", {"slug": "nope", "angles": [1, 1, 1]})[0] == 404


def test_settings_defaults_and_bandwidth_options():
    from qmed.config import BANDWIDTH_OPTIONS, DEFAULT_BANDWIDTH
    slug, st = la.validate_settings({"slug": SLUG})
    assert st["kernel_bandwidth"] == DEFAULT_BANDWIDTH and st["n_qubits"] == 3 and st["max_samples"] == 100
    for b in BANDWIDTH_OPTIONS:
        assert la.validate_settings({"slug": SLUG, "kernel_bandwidth": b})[1]["kernel_bandwidth"] == b
    assert la.validate_settings({"slug": SLUG, "kernel_bandwidth": 1})[1]["kernel_bandwidth"] == 1.0
    assert la.STATIC_SETTINGS["kernel_bandwidth"] == 1.0  # shipped JSON = train_all_models' default
    assert la.default_run_id(SLUG) == la.make_run_id(SLUG, {**st, "kernel_bandwidth": 1.0})


def test_run_id_stable_and_distinct():
    a = la.make_run_id(SLUG, la.DEFAULTS)
    assert a == la.make_run_id(SLUG, dict(reversed(list(la.DEFAULTS.items()))))
    assert a != la.make_run_id(SLUG, {**la.DEFAULTS, "reps": 1})
    assert a != la.make_run_id(SLUG, {**la.DEFAULTS, "kernel_bandwidth": 0.3})
    assert a != la.default_run_id("diabetes_pima")
    assert len(a) == 12


# ------------------------------------------------------------ backends (fake IBM)
class _St:
    def __init__(self, op, p):
        self.operational, self.pending_jobs = op, p


class _B:
    def __init__(self, name, n, op, p):
        self.name, self.num_qubits, self._s = name, n, _St(op, p)

    def status(self):
        return self._s


class _Svc:
    def backends(self):
        return [_B("ibm_fake_a", 127, True, 12), _B("ibm_fake_b", 156, False, 0)]


def test_backends_fake_service_shape(srv):
    srv.mp.setattr(qs, "make_service", lambda: _Svc())
    code, r = srv.call("GET", "/api/qpu/backends")
    assert code == 200
    assert r == [{"name": "ibm_fake_a", "operational": True, "num_qubits": 127, "pending_jobs": 12},
                 {"name": "ibm_fake_b", "operational": False, "num_qubits": 156, "pending_jobs": 0}]


def test_backends_errors_scrub_token(srv):
    def boom():
        raise RuntimeError(f"401 for token {SECRET} / abcdefghijklmnopqrstuvwxyz0123456789")
    srv.mp.setattr(qs, "make_service", boom)
    code, r = srv.call("GET", "/api/qpu/backends")
    assert code == 502 and SECRET not in r["error"] and "abcdefghijklmnopqrstuvwxyz" not in r["error"]
    srv.mp.setattr(qs, "get_token", lambda: "")
    srv.mp.setattr(qs, "QISKIT_ACCOUNTS_FILE", ROOT / "does-not-exist.json")
    try:
        code, r = srv.call("GET", "/api/qpu/backends")
        assert code == 503 and "QISKIT_IBM_TOKEN" in r["error"]
    finally:
        srv.mp.setattr(qs, "get_token", lambda: SECRET)


# ------------------------------------------------------------ training (small)
def test_small_run_json(small_run):
    rid, js = small_run
    assert js["custom_run"] is True and js["run_id"] == rid and js["slug"] == SLUG
    assert js["settings"] == {"n_qubits": 3, "feature_map": "ZZFeatureMap", "reps": 2, "entanglement": "linear",
                              "vqc_iter": 15, "seed": 42, "max_samples": 60, "kernel_bandwidth": 0.1}
    assert len(js["train"]["angles"]) == 45 and len(js["test"]) == 15
    assert all(len(a) == 3 for a in js["train"]["angles"])
    assert len(js["rbf"]["values"]) == 6 ** 3 and "grid_dims" not in js["rbf"]
    assert np.array(js["qsvc"]["kernel_test"]).shape == (15, 45)
    assert len(js["vqc"]["loss_history"]) <= 15
    for p in js["test"]:
        assert set(p["models"]) == set(MODELS)


def test_predict_matches_exported_models(srv, small_run):
    rid, js = small_run
    bundle = la.RUNS[rid]["bundle"]
    n_train = len(js["train"]["angles"])
    checked = 0
    for i, p in enumerate(js["test"]):
        x = bundle.data.X_test_q[i]
        if np.any(x < 0) or np.any(x > np.pi):
            continue  # live predict clamps to [0, pi]; exported test angles can lie slightly outside
        code, r = srv.call("POST", "/api/live/predict", {"run_id": rid, "angles": [float(v) for v in x]})
        assert code == 200 and r["clamped"] is False
        assert len(r["kernel_row"]) == n_train
        assert r["kernel_row"] == pytest.approx(js["qsvc"]["kernel_test"][i], abs=1e-6)
        assert r["partner"] == int(np.argmax(js["qsvc"]["kernel_test"][i]))
        for m in MODELS:
            assert 0 <= r["models"][m]["prob"] <= 1
            assert r["models"][m]["prob"] == pytest.approx(p["models"][m]["prob"], abs=1e-6), (i, m)
            assert r["models"][m]["pred"] == p["models"][m]["pred"], (i, m)
        checked += 1
    assert checked >= 5


def test_predict_reconstruction_clamp_and_validation(srv, small_run):
    rid, js = small_run
    i = 0
    code, r = srv.call("POST", "/api/live/predict", {"run_id": rid, "angles": js["test"][i]["angles"]})
    assert code == 200
    rec = r["reconstructed"]
    nf = len(js["feature_names"])
    assert len(rec["pca"]) == 3 and len(rec["standardized"]) == nf and len(rec["raw"]) == nf
    assert rec["pca"] == pytest.approx(js["test"][i]["pca"], abs=2e-3)
    h = r["models"]["hybrid"]
    assert h["pred"] == int(0.5 * r["models"]["rbf"]["prob"] + 0.5 * r["models"]["qsvc"]["prob"] >= 0.5)

    code, r = srv.call("POST", "/api/live/predict", {"run_id": rid, "angles": [-1.0, 0.5, 9.0]})
    assert code == 200 and r["clamped"] is True
    assert r["angles"] == pytest.approx([0.0, 0.5, round(math.pi, 4)])
    for m in MODELS:
        assert 0 <= r["models"][m]["prob"] <= 1 and r["models"][m]["pred"] in (0, 1)

    # Python's json accepts NaN/Infinity, so they must be rejected explicitly.
    for bad in ([1.0, 2.0], [1, 2, 3, 4], [1, "a", 2], [1, True, 2], "1,2,3", None,
                [1.0, float("nan"), 2.0], [1.0, float("inf"), 2.0]):
        code, r = srv.call("POST", "/api/live/predict", {"run_id": rid, "angles": bad})
        assert code == 400, bad


@pytest.mark.parametrize("n, dims, size", [(2, 2, 6 * 6), (4, 3, 6 ** 3)])
def test_grid_shapes_other_qubit_counts(srv, small_run, n, dims, size):
    rid, js = train(srv, **SMALL, n_qubits=n)
    assert js["settings"]["n_qubits"] == n and all(len(a) == n for a in js["train"]["angles"])
    assert len(js["linear"]["coef"]) == n
    assert js["rbf"]["grid_dims"] == dims and len(js["rbf"]["values"]) == size
    if n == 4:
        x4 = float(np.mean(np.array(js["train"]["angles"])[:, 3]))
        assert js["rbf"]["slice"]["dim"] == 3
        assert js["rbf"]["slice"]["value"] == pytest.approx(x4, abs=1e-3)
    else:
        assert "slice" not in js["rbf"]
    code, r = srv.call("POST", "/api/live/predict", {"run_id": rid, "angles": [1.0] * n})
    assert code == 200 and len(r["reconstructed"]["pca"]) == n


def test_train_models_matches_qmed_train_all_models():
    """Drift guard: export_tour_data.train_models replicates qmed.models.train_all_models exactly."""
    import export_tour_data as etd
    from qmed.circuits import build_ansatz, build_feature_map
    from qmed.config import DATASETS
    from qmed.data import prepare_data
    from qmed.models import train_all_models
    d = prepare_data(DATASETS["Vertebral Column (Orthopedics)"], 60, 2)
    fm, an = build_feature_map("ZZFeatureMap", 2, 1, "linear"), build_ansatz(2, 1, "linear")
    ref = train_all_models(d.X_train_q, d.y_train, d.X_test_q, fm, an, 15, seed=42, kernel_bandwidth=0.3)
    new, _, _ = etd.train_models(d.X_train_q, d.y_train, d.X_test_q, fm, an, 15, seed=42, kernel_bandwidth=0.3)
    assert set(ref) == set(new)
    for name in ref:
        assert np.array_equal(ref[name]["pred"], new[name]["pred"]), name
        assert np.allclose(ref[name]["prob"], new[name]["prob"], atol=1e-12), name
    assert np.allclose(ref[etd.QSVC_MODEL]["kernel_matrix"], new[etd.QSVC_MODEL]["kernel_matrix"])
    assert ref[etd.VQC_MODEL]["loss_history"] == new[etd.VQC_MODEL]["loss_history"]


def test_bandwidth_predict_uses_scaled_kernel(srv, small_run):
    """QSVC bandwidth lives inside the kernel circuit: predict on raw angles == exported, and
    kernel_row == exact |<psi(s*x)|psi(s*x')>|^2 (not the unscaled fidelity)."""
    from qiskit.quantum_info import Statevector
    from qmed.circuits import build_feature_map
    rid, js = train(srv, **SMALL, kernel_bandwidth=0.3)
    assert js["settings"]["kernel_bandwidth"] == 0.3
    bundle = la.RUNS[rid]["bundle"]
    fm = build_feature_map("ZZFeatureMap", 3, 2, "linear")
    i = next(k for k, x in enumerate(bundle.data.X_test_q) if np.all((x >= 0) & (x <= np.pi)))
    x = bundle.data.X_test_q[i]
    code, r = srv.call("POST", "/api/live/predict", {"run_id": rid, "angles": [float(v) for v in x]})
    assert code == 200
    for m in MODELS:
        assert r["models"][m]["prob"] == pytest.approx(js["test"][i]["models"][m]["prob"], abs=1e-6), m
        assert r["models"][m]["pred"] == js["test"][i]["models"][m]["pred"], m
    assert r["kernel_row"] == pytest.approx(js["qsvc"]["kernel_test"][i], abs=1e-6)
    t = bundle.data.X_train_q[0]

    def fid(a, b):
        return abs(Statevector(fm.assign_parameters(a)).inner(Statevector(fm.assign_parameters(b)))) ** 2
    assert r["kernel_row"][0] == pytest.approx(fid(0.3 * x, 0.3 * t), abs=1e-4)
    assert abs(fid(x, t) - fid(0.3 * x, 0.3 * t)) > 1e-3  # the scaling really matters for this pair


def test_default_settings_lazy_predict_reproduces_static_export(srv, small_run):
    """predict({slug}) lazily trains the static-file settings; results equal a fresh export with
    the current code (the committed JSON may predate the exact-kernel change on main, so it is only
    compared when its kernel matches the current code)."""
    import export_tour_data as etd
    from qmed.config import DATASETS
    static_path = ROOT / "web/public/data" / f"{SLUG}.json"
    committed = json.load(open(static_path))
    angles = committed["test"][0]["angles"]
    code, r = srv.call("POST", "/api/live/predict", {"slug": SLUG, "angles": angles})
    assert code == 202 and r["state"] == "training" and r["run_id"] == la.default_run_id(SLUG)
    rid = r["run_id"]
    code, again = srv.call("POST", "/api/live/predict", {"slug": SLUG, "angles": angles})
    assert code in (200, 202) and again["run_id"] == rid
    s, _ = srv.wait(rid, timeout=300)
    assert s["state"] == "done"
    code, js = srv.call("GET", f"/api/live/run/{rid}.json")
    assert code == 200 and js["custom_run"] is True
    assert js["settings"]["max_samples"] == 100 and js["settings"]["kernel_bandwidth"] == 1.0

    fresh = etd.export_dataset("Vertebral Column (Orthopedics)", DATASETS["Vertebral Column (Orthopedics)"],
                               grid_size=6)
    refs = [fresh]
    if committed["qsvc"]["kernel_test"] == fresh["qsvc"]["kernel_test"]:
        refs.append(committed)  # committed JSON already regenerated with the current code
    for ref in refs:
        assert js["test"][0]["models"]["qsvc"]["prob"] == pytest.approx(ref["test"][0]["models"]["qsvc"]["prob"],
                                                                          abs=1e-6)
        for a, b in zip(js["test"], ref["test"]):
            for m in MODELS:
                assert a["models"][m]["prob"] == pytest.approx(b["models"][m]["prob"], abs=1e-6)
                assert a["models"][m]["pred"] == b["models"][m]["pred"]
        for key in ("train", "qsvc", "vqc", "linear", "qpu", "feature_ranges"):
            assert js[key] == ref[key], key
    assert {k: v for k, v in js["settings"].items() if k not in ("max_samples", "kernel_bandwidth")} \
        == fresh["settings"]

    bundle = la.RUNS[rid]["bundle"]
    for i in range(len(fresh["test"])):
        x = bundle.data.X_test_q[i]
        if np.any(x < 0) or np.any(x > np.pi):
            continue
        code, p = srv.call("POST", "/api/live/predict", {"slug": SLUG, "angles": [float(v) for v in x]})
        assert code == 200 and p["run_id"] == rid and p["clamped"] is False
        for m in MODELS:
            assert p["models"][m]["prob"] == pytest.approx(fresh["test"][i]["models"][m]["prob"], abs=1e-6)
            assert p["models"][m]["pred"] == fresh["test"][i]["models"][m]["pred"]
