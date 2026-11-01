import pytest

from qmed.circuits import build_ansatz, build_feature_map
from qmed.config import FEATURE_MAP_ZZ
from qmed.hardware import qpu_profile


@pytest.fixture(scope="module")
def prof():
    fm = build_feature_map(FEATURE_MAP_ZZ, 3, 2, "linear")
    an = build_ansatz(3, 2, "linear")
    return qpu_profile(fm, an, 3, 75, 25, 35)


def test_gate_counts_and_times(prof):
    assert prof["n_2q"] == 2 * (prof["fm_ops"].get("ecr", 0) + prof["fm_ops"].get("cx", 0))
    assert prof["t_eval_ms"] == pytest.approx(1024 * (prof["t_circuit_us"] + 250.0) / 1000)
    assert prof["n_qsvc_evals"] == 75 * 74 // 2 + 75 * 25
    assert prof["n_vqc_evals"] == 75 * 35
    assert prof["qsvc_qpu_s"] > 0


def test_gates_sequence(prof):
    assert len(prof["gates"]) > 0
    allowed = set(prof["basis"]) | {"measure"}
    assert all(g["name"] in allowed for g in prof["gates"])
