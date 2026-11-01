"""Tour extras export (Kernel Bandwidth Lab + QuHaMA) on vertebral_column."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import export_tour_extras as ex  # noqa: E402
from qmed.config import BANDWIDTH_OPTIONS, DATASETS  # noqa: E402
from qmed.data import prepare_data  # noqa: E402

CSV = next(p for p in DATASETS.values() if p.stem == "vertebral_column")


@pytest.fixture(scope="module")
def exported():
    return ex.export_dataset(CSV)


def test_bandwidth_kernels_and_sweep(exported):
    bw, _ = exported
    assert bw["options"] == BANDWIDTH_OPTIONS
    n = len(bw["order"])
    assert 0 < n <= 40 and len(bw["labels"]) == n
    assert bw["labels"] == sorted(bw["labels"])
    for s in BANDWIDTH_OPTIONS:
        K = np.array(bw["kernels"][str(s)])
        assert K.shape == (n, n)
        assert np.allclose(np.diag(K), 1.0, atol=1e-3)
        assert np.allclose(K, K.T, atol=1e-3)
        assert str(s) in bw["diagnostics"] and str(s) in bw["regime"]
    assert len(bw["sweep"]["rows"]) == len(BANDWIDTH_OPTIONS)
    assert [r["bandwidth"] for r in bw["sweep"]["rows"]] == BANDWIDTH_OPTIONS
    assert "accuracy" in bw["sweep"]["rbf"]


def test_quhama_breakdown(exported):
    _, qh = exported
    data = prepare_data(CSV, ex.MAX_SAMPLES, ex.N_QUBITS)
    assert len(qh["test"]) == len(data.y_test)
    for i, p in enumerate(qh["test"]):
        assert p["index"] == i and p["label"] == int(data.y_test[i])
        assert abs(p["w_classical"] + p["w_quantum"] - 1.0) <= 1e-6
        for k in ("p_classical", "p_quantum", "p_final", "w_classical", "w_quantum"):
            assert 0.0 <= p[k] <= 1.0
        assert p["pred"] == int(p["p_final"] >= 0.5)
        assert len(p["z_latent"]) == ex.N_QUBITS
    assert len(qh["history"]["encoder"]) == ex.EPOCHS_ENC


def test_test_order_matches_committed_json():
    committed = ROOT / "web/public/data/vertebral_column.json"
    if not committed.exists():
        pytest.skip("no committed tour json")
    test = json.loads(committed.read_text())["test"]
    data = prepare_data(CSV, ex.MAX_SAMPLES, ex.N_QUBITS)
    assert [t["label"] for t in test] == [int(v) for v in data.y_test]
    assert np.allclose([t["standardized"] for t in test], data.X_test_std, atol=1e-3)


def test_deterministic(exported):
    again = ex.export_dataset(CSV)
    strip = lambda o: {k: v for k, v in o.items() if k != "timings"}  # noqa: E731
    assert json.dumps(strip(again[0])) == json.dumps(strip(exported[0]))
    assert json.dumps(strip(again[1])) == json.dumps(strip(exported[1]))
