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

from qmed.config import LINEAR_SVM, QSVC_MODEL, RANDOM_STATE, RBF_SVM, VQC_MODEL


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


def train_all_models(X_train_q, y_train, X_test_q, feature_map, ansatz, vqc_iter: int,
                     seed: int | None = RANDOM_STATE) -> dict:
    """Return {model name: result dict}. QSVC adds ``kernel_matrix``; VQC adds ``loss_history`` and ``weights_history``.

    ``seed`` makes the sampler and VQC start point repeatable (None = unseeded).
    """
    results = {}

    results[LINEAR_SVM] = fit_and_evaluate(
        SVC(kernel="linear", probability=True, random_state=RANDOM_STATE), X_train_q, y_train, X_test_q
    )
    results[RBF_SVM] = fit_and_evaluate(
        SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE), X_train_q, y_train, X_test_q
    )

    # Quantum kernel: similarity = overlap of the two patients' quantum states
    sampler = StatevectorSampler(seed=seed)
    q_kernel = FidelityQuantumKernel(feature_map=feature_map, fidelity=ComputeUncompute(sampler=sampler))
    qsvc = QSVC(quantum_kernel=q_kernel, probability=True, random_state=RANDOM_STATE)
    results[QSVC_MODEL] = fit_and_evaluate(qsvc, X_train_q, y_train, X_test_q)
    results[QSVC_MODEL]["kernel_matrix"] = q_kernel.evaluate(x_vec=X_train_q)

    # Variational classifier: COBYLA tunes ansatz weights; record loss each step
    loss_history = []
    weights_history = []
    initial_point = (
        np.random.default_rng(seed).uniform(-np.pi, np.pi, ansatz.num_parameters)
        if seed is not None else None
    )

    def _record(weights, loss):
        weights_history.append(list(map(float, weights)))
        loss_history.append(float(loss))

    vqc = VQC(
        sampler=sampler,
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=COBYLA(maxiter=vqc_iter),
        callback=_record,
        initial_point=initial_point,
    )
    results[VQC_MODEL] = fit_and_evaluate(vqc, X_train_q, y_train, X_test_q)
    results[VQC_MODEL]["loss_history"] = loss_history
    results[VQC_MODEL]["weights_history"] = weights_history

    return results
