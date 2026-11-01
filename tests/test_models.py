import numpy as np
import pytest

from qmed.circuits import build_ansatz, build_feature_map
from qmed.config import DATASETS, QSVC_MODEL, VQC_MODEL
from qmed.data import prepare_data
from qmed.models import train_all_models

LABEL = "Statlog Heart Disease (Cardiology)"


@pytest.fixture(scope="module")
def setup():
    d = prepare_data(DATASETS[LABEL], 60, 2)
    fm = build_feature_map("ZZFeatureMap", 2, 2, "linear")
    an = build_ansatz(2, 1, "linear")
    return d, fm, an


def _run(setup, **kw):
    d, fm, an = setup
    return train_all_models(d.X_train_q, d.y_train, d.X_test_q, fm, an, 5, **kw), an


def test_seeded_runs_are_identical(setup):
    a, _ = _run(setup)
    b, _ = _run(setup)
    assert np.allclose(a[QSVC_MODEL]["prob"], b[QSVC_MODEL]["prob"])
    assert np.allclose(a[VQC_MODEL]["prob"], b[VQC_MODEL]["prob"])
    assert np.allclose(a[VQC_MODEL]["loss_history"], b[VQC_MODEL]["loss_history"])


def test_weights_history_matches_loss_history(setup):
    r, an = _run(setup)
    v = r[VQC_MODEL]
    assert len(v["weights_history"]) == len(v["loss_history"]) > 0
    assert all(len(w) == an.num_parameters for w in v["weights_history"])


def test_seed_none_still_runs(setup):
    r, _ = _run(setup, seed=None)
    assert len(r) == 4
