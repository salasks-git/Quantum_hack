import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from export_tour_data import export_dataset  # noqa: E402
from qmed.config import DATASETS  # noqa: E402
from qmed.data import prepare_data  # noqa: E402

LABEL = "Vertebral Column (Orthopedics)"


@pytest.fixture(scope="module")
def result():
    return export_dataset(LABEL, DATASETS[LABEL], grid_size=8)


def test_shapes(result):
    assert len(result["train"]["angles"]) == 75
    assert all(len(a) == 3 for a in result["train"]["angles"])
    assert len(result["test"]) == 25
    kt = np.array(result["qsvc"]["kernel_train"])
    ks = np.array(result["qsvc"]["kernel_test"])
    assert kt.shape == (75, 75)
    assert ks.shape == (25, 75)
    assert len(result["rbf"]["values"]) == 8 ** 3
    assert len(result["feature_names"]) == 6 == len(result["test"][0]["raw"])


def test_grid_bounds(result):
    assert result["rbf"]["min"] == -0.3
    assert result["rbf"]["max"] == pytest.approx(np.pi + 0.3)


def test_preds_and_probs_valid(result):
    for p in result["test"]:
        for m in ("linear", "rbf", "qsvc", "vqc"):
            assert p["models"][m]["pred"] in (0, 1)
            assert 0 <= p["models"][m]["prob"] <= 1


def test_kernel_diagonal_near_one(result):
    kt = np.array(result["qsvc"]["kernel_train"])
    assert all(kt[i][i] > 0.95 for i in range(len(kt)))


def test_json_serialisable(result):
    json.dumps(result)


def test_prepare_data_backward_compatible():
    d = prepare_data(DATASETS[LABEL], 100, 3)
    for f in ("X_train_q", "X_test_q", "y_train", "y_test", "pca", "prep_time"):
        assert hasattr(d, f)


def test_hybrid_timings_qpu(result):
    for p in result["test"]:
        h = p["models"]["hybrid"]
        assert h["pred"] in (0, 1) and 0 <= h["prob"] <= 1
        assert h["pred"] == int(h["prob"] >= 0.5)
    assert set(result["timings"]) == {"linear", "rbf", "qsvc", "vqc", "hybrid"}
    assert all(t["total_s"] > 0 for t in result["timings"].values())
    assert result["qpu"]["qsvc_qpu_s"] > 0
