"""Train and evaluate the four classifiers on the same angle-encoded data."""
import time

import numpy as np
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.algorithms import QSVC, VQC
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from sklearn.svm import SVC

# qiskit-machine-learning >= 0.8 ships these; older installs use qiskit-algorithms
try:
    from qiskit_machine_learning.optimizers import COBYLA
    from qiskit_machine_learning.state_fidelities import ComputeUncompute
except ImportError:
    from qiskit_algorithms.optimizers import COBYLA
    from qiskit_algorithms.state_fidelities import ComputeUncompute

from qmed.config import HYBRID_ENSEMBLE, LINEAR_SVM, QSVC_MODEL, RANDOM_STATE, RBF_SVM, VQC_MODEL


class HybridEnsembleModel:
    """Ensemble model blending Classical RBF SVM and Quantum Kernel QSVC probabilities."""
    def __init__(self, rbf_model, qsvc_model):
        self.rbf = rbf_model
        self.qsvc = qsvc_model

    def predict_proba(self, X):
        p_rbf = np.asarray(self.rbf.predict_proba(X))
        p_qsvc = np.asarray(self.qsvc.predict_proba(X))
        return 0.5 * p_rbf + 0.5 * p_qsvc

    def predict(self, X):
        prob = self.predict_proba(X)[:, 1]
        return (prob >= 0.5).astype(int)


def fit_and_evaluate(model, X_train, y_train, X_test) -> dict:
    """Fit, predict, and time both phases.

    Returns a result dict with keys: model, pred (0/1 labels), prob (P(disease)),
    train_time, inf_time, total_time (seconds).
    """
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    t0 = time.time()
    pred = np.asarray(model.predict(X_test)).ravel().astype(int)
    # Real probabilities (not hard labels) so ROC-AUC measures ranking fairly
    prob = np.asarray(model.predict_proba(X_test))[:, 1]
    inf_time = time.time() - t0

    return {
        "model": model, "pred": pred, "prob": prob,
        "train_time": train_time, "inf_time": inf_time,
        "total_time": train_time + inf_time,
    }


def train_all_models(X_train_q, y_train, X_test_q, feature_map, ansatz, vqc_iter: int) -> dict:
    """Return {model name: result dict}. QSVC adds ``kernel_matrix``; VQC adds ``loss_history``."""
    results = {}

    results[LINEAR_SVM] = fit_and_evaluate(
        SVC(kernel="linear", probability=True, random_state=RANDOM_STATE), X_train_q, y_train, X_test_q
    )
    results[RBF_SVM] = fit_and_evaluate(
        SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE), X_train_q, y_train, X_test_q
    )

    # Quantum kernel: similarity = overlap of the two patients' quantum states
    sampler = StatevectorSampler()
    q_kernel = FidelityQuantumKernel(feature_map=feature_map, fidelity=ComputeUncompute(sampler=sampler))
    qsvc = QSVC(quantum_kernel=q_kernel, probability=True, random_state=RANDOM_STATE)
    results[QSVC_MODEL] = fit_and_evaluate(qsvc, X_train_q, y_train, X_test_q)
    results[QSVC_MODEL]["kernel_matrix"] = q_kernel.evaluate(x_vec=X_train_q)

    # Variational classifier: COBYLA tunes ansatz weights; record loss each step
    loss_history = []
    vqc = VQC(
        sampler=sampler,
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=COBYLA(maxiter=vqc_iter),
        callback=lambda weights, loss: loss_history.append(loss),
    )
    results[VQC_MODEL] = fit_and_evaluate(vqc, X_train_q, y_train, X_test_q)
    results[VQC_MODEL]["loss_history"] = loss_history

    # 5. Hybrid Quantum-Classical Ensemble (QSVC + RBF Soft Voting)
    prob_rbf = results[RBF_SVM]["prob"]
    prob_qsvc = results[QSVC_MODEL]["prob"]
    prob_ens = 0.5 * prob_rbf + 0.5 * prob_qsvc
    pred_ens = (prob_ens >= 0.5).astype(int)
    t_train_ens = results[RBF_SVM]["train_time"] + results[QSVC_MODEL]["train_time"]
    t_inf_ens = results[RBF_SVM]["inf_time"] + results[QSVC_MODEL]["inf_time"]

    results[HYBRID_ENSEMBLE] = {
        "model": HybridEnsembleModel(results[RBF_SVM]["model"], results[QSVC_MODEL]["model"]),
        "pred": pred_ens,
        "prob": prob_ens,
        "train_time": t_train_ens,
        "inf_time": t_inf_ens,
        "total_time": t_train_ens + t_inf_ens,
    }

    return results
