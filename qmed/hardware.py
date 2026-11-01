"""IBM QPU hardware profile (Streamlit-free). Mirrors qmed/ui/tab_ibm_hardware.py formulas."""
from qiskit.compiler import transpile

T_1Q_NS = 35.0
T_RZ_NS = 0.0
T_2Q_NS = 300.0
T_MEAS_NS = 700.0
T_RESET_US = 250.0
SHOTS = 1024
QUEUE_REF_S = 600.0
BASIS = ["rz", "sx", "x", "ecr"]


def qpu_profile(feature_map, ansatz, n_qubits: int, n_train: int, n_test: int, vqc_iter: int) -> dict:
    tfm = transpile(feature_map, basis_gates=BASIS, optimization_level=2)
    tans = transpile(ansatz, basis_gates=BASIS, optimization_level=2)
    fm_ops = {k: int(v) for k, v in tfm.count_ops().items()}
    ans_ops = {k: int(v) for k, v in tans.count_ops().items()}

    n_1q = (fm_ops.get("sx", 0) + fm_ops.get("x", 0)) * 2
    n_rz = fm_ops.get("rz", 0) * 2
    n_2q = (fm_ops.get("ecr", 0) + fm_ops.get("cx", 0)) * 2
    n_meas = int(n_qubits)

    t_circuit_ns = n_1q * T_1Q_NS + n_rz * T_RZ_NS + n_2q * T_2Q_NS + n_meas * T_MEAS_NS
    t_circuit_us = t_circuit_ns / 1000.0
    t_eval_ms = SHOTS * (t_circuit_us + T_RESET_US) / 1000.0

    n_qsvc_evals = (n_train * (n_train - 1)) // 2 + n_train * n_test
    n_vqc_evals = n_train * vqc_iter

    gates = [
        {"name": inst.operation.name, "qubits": [tfm.find_bit(q).index for q in inst.qubits]}
        for inst in tfm.data if inst.operation.name != "barrier"
    ]
    return {
        "basis": list(BASIS), "fm_ops": fm_ops, "ans_ops": ans_ops,
        "fm_depth": int(tfm.depth()), "ans_depth": int(tans.depth()),
        "n_1q": int(n_1q), "n_rz": int(n_rz), "n_2q": int(n_2q), "n_meas": n_meas,
        "t_circuit_us": float(t_circuit_us), "t_eval_ms": float(t_eval_ms),
        "n_qsvc_evals": int(n_qsvc_evals), "n_vqc_evals": int(n_vqc_evals),
        "qsvc_qpu_s": float(n_qsvc_evals * t_eval_ms / 1000.0),
        "vqc_qpu_s": float(n_vqc_evals * t_eval_ms / 1000.0),
        "queue_s": QUEUE_REF_S, "shots": SHOTS, "gates": gates,
    }
