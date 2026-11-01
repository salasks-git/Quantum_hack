import numpy as np
import pytest

from qmed.circuits import build_feature_map
from qmed.config import DATASETS
from qmed.data import prepare_data
from qmed.quhama import QuHaMAEncoder, QuHaMAModel, train_quhama

LABEL = "Statlog Heart Disease (Cardiology)"


@pytest.fixture(scope="module")
def setup():
    d = prepare_data(DATASETS[LABEL], 50, 2)
    fm = build_feature_map("ZZFeatureMap", 2, 1, "linear")
    return d, fm


def test_quhama_trains_and_predicts(setup):
    d, fm = setup
    res = train_quhama(
        d.X_train_std, d.y_train, d.X_test_std, d.y_test,
        n_qubits=2, feature_map=fm, epochs_enc=15, epochs_gate=10
    )
    assert res.metrics["Accuracy"] >= 0.5
    assert 0.0 <= res.metrics["F2-Score"] <= 1.0
    assert len(res.loss_history_encoder) == 15
    assert len(res.loss_history_gating) == 10

    # Test breakdown fields
    bk = res.breakdown_test
    assert len(bk["p_classical"]) == len(d.y_test)
    assert len(bk["p_quantum"]) == len(d.y_test)
    assert len(bk["w_classical"]) == len(d.y_test)
    assert len(bk["w_quantum"]) == len(d.y_test)
    assert np.allclose(bk["w_classical"] + bk["w_quantum"], 1.0, atol=1e-4)


def test_quhama_predict_proba(setup):
    d, fm = setup
    res = train_quhama(
        d.X_train_std, d.y_train, d.X_test_std, d.y_test,
        n_qubits=2, feature_map=fm, epochs_enc=10, epochs_gate=8
    )
    probs = res.model.predict_proba(d.X_test_std)
    assert probs.shape == (len(d.y_test), 2)
    assert np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-5)
