"""Quantum kernel bandwidth sweep: cross-validated QSVC scores and kernel diagnostics vs s.

QSVC encodes s * angles. With s = 1 the full [0, pi] range makes the fidelity kernel
very "sharp": every patient looks dissimilar to every other one (near-identity Gram
matrix), so the SVM memorizes. As s -> 0 every patient looks the same (all-ones Gram
matrix). The useful regime is in between (Shaydulin & Wild 2022, arXiv:2111.05451).

No Streamlit imports: usable from notebooks and tests.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from qmed.circuits import build_feature_map
from qmed.config import BANDWIDTH_OPTIONS, CV_FOLDS, RANDOM_STATE
from qmed.data import angle_transform, load_dataset, subsample
from qmed.models import build_quantum_kernel


@dataclass(frozen=True)
class BandwidthSweep:
    # One row per bandwidth: bandwidth, accuracy(_std), roc_auc(_std), recall(_std),
    # kernel_mean, kernel_var, alignment (all averaged over the CV folds).
    table: pd.DataFrame
    rbf: dict          # classical RBF SVM CV baseline: accuracy(_std), roc_auc(_std), recall(_std)
    n_patients: int
    n_folds: int


def kernel_diagnostics(K: np.ndarray, y: np.ndarray) -> dict:
    """Concentration and label alignment of a square Gram matrix.

    - kernel_mean / kernel_var: mean and variance of the off-diagonal entries. Mean
      near 0 = near-identity (memorizing); mean near 1 with tiny variance = all-ones
      (exponential concentration: every patient looks the same).
    - alignment: centered kernel-target alignment (Cortes, Mohri & Rostamizadeh 2012),
      cosine similarity between the centered K and the ideal kernel y y^T with
      y in {-1, +1}. Higher = kernel geometry already separates the two classes.
    """
    n = len(K)
    off = K[~np.eye(n, dtype=bool)]
    y_pm = np.where(np.asarray(y) == 1, 1.0, -1.0)
    H = np.eye(n) - 1.0 / n
    Kc = H @ K @ H
    Yc = H @ np.outer(y_pm, y_pm) @ H
    denom = np.linalg.norm(Kc) * np.linalg.norm(Yc)
    return {
        "kernel_mean": float(off.mean()),
        "kernel_var": float(off.var()),
        # An all-ones kernel centers to zero: alignment is 0, not NaN.
        "alignment": float(np.sum(Kc * Yc) / denom) if denom > 1e-12 else 0.0,
    }


def kernel_matrices(X_q: np.ndarray, feature_map, bandwidths) -> dict:
    """{s: Gram matrix of X_q under the quantum kernel at bandwidth s}."""
    return {s: build_quantum_kernel(feature_map, s).evaluate(x_vec=X_q) for s in bandwidths}


def _scores(y_true, pred, score) -> tuple:
    # ROC-AUC from the SVM margin (decision_function), not Platt probabilities.
    return (accuracy_score(y_true, pred), roc_auc_score(y_true, score),
            recall_score(y_true, pred, zero_division=0))


def _summarize(fold_scores: list) -> dict:
    a = np.asarray(fold_scores)
    out = {}
    for i, name in enumerate(["accuracy", "roc_auc", "recall"]):
        out[name] = float(a[:, i].mean())
        out[f"{name}_std"] = float(a[:, i].std())
    return out


def bandwidth_sweep(
    csv_path: Path,
    n_samples: int,
    n_qubits: int,
    feature_map_name: str,
    reps: int,
    entanglement: str,
    bandwidths=tuple(BANDWIDTH_OPTIONS),
    n_folds: int = CV_FOLDS,
) -> BandwidthSweep:
    """Stratified k-fold CV of an exact-kernel QSVC at each bandwidth, plus an RBF SVM baseline.

    Uses the same stratified cohort of ``n_samples`` patients as the app (train + test
    together). Preprocessing (standardize -> PCA -> [0, pi]) is refit inside every fold,
    so held-out patients never leak into the scaler or PCA. The QSVC is an SVC on the
    precomputed quantum Gram matrix, which is what qiskit's QSVC does internally.
    """
    _, X, y, _ = load_dataset(csv_path)
    X, y = subsample(X, y, n_samples)
    feature_map = build_feature_map(feature_map_name, n_qubits, reps, entanglement)
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    q_scores = {s: [] for s in bandwidths}
    q_diag = {s: [] for s in bandwidths}
    rbf_scores = []
    for tr, te in folds.split(X, y):
        A, B, _ = angle_transform(X[tr], X[te], n_qubits)
        y_tr, y_te = y[tr], y[te]

        rbf = SVC(kernel="rbf").fit(A, y_tr)
        rbf_scores.append(_scores(y_te, rbf.predict(B), rbf.decision_function(B)))

        for s in bandwidths:
            kernel = build_quantum_kernel(feature_map, s)
            K_tr = kernel.evaluate(x_vec=A)
            K_te = kernel.evaluate(x_vec=B, y_vec=A)
            svm = SVC(kernel="precomputed").fit(K_tr, y_tr)
            q_scores[s].append(_scores(y_te, svm.predict(K_te), svm.decision_function(K_te)))
            q_diag[s].append(kernel_diagnostics(K_tr, y_tr))

    rows = []
    for s in bandwidths:
        diag = pd.DataFrame(q_diag[s]).mean().to_dict()
        rows.append({"bandwidth": s, **_summarize(q_scores[s]), **diag})
    return BandwidthSweep(
        table=pd.DataFrame(rows),
        rbf=_summarize(rbf_scores),
        n_patients=len(y),
        n_folds=n_folds,
    )
