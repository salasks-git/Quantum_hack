"""Approach 3: Geometric Composite Kernel Fusion (Mercer Convex Combination)
========================================================================

Architecture & Mathematical Details:
-------------------------------------
Instead of combining models at the probabilistic decision level (soft voting), Approach 3 
unifies classical and quantum geometries directly at the foundational kernel level 
using Mercer's Theorem.

1. Mercer's Theorem & Direct Sum Hilbert Space:
   Under Mercer's Theorem, if K_RBF and K_Quantum are positive semi-definite (PSD) kernels, 
   any convex linear combination:
       K_Composite(x, x') = (1 - λ) · K_RBF(x, x') + λ · K_Quantum(x, x'),   λ ∈ [0, 1]
   is guaranteed to be a mathematically valid Mercer kernel.
   This constructs a unified direct sum Hilbert space:
       ℋ_Composite = ℋ_RBF ⊕ ℋ_Quantum

2. Gram Matrix Formulation:
   During training on cohort {x_1, ..., x_N}:
       G_RBF[i, j] = exp(-γ ||x_i - x_j||²)
       G_Quantum[i, j] = |⟨ψ(x_i) | ψ(x_j)⟩|²
       G_Composite = (1 - λ) · G_RBF + λ · G_Quantum

3. Dual Optimization in Composite Space:
   A standard Support Vector Machine solves the convex quadratic dual problem on G_Composite:
       max_α  ∑_i α_i - (1/2) ∑_{i,j} α_i α_j y_i y_j G_Composite[i, j]
       subject to  0 ≤ α_i ≤ C,   ∑_i α_i y_i = 0
   The resulting support vectors simultaneously satisfy local Euclidean metric continuity 
   and non-local quantum phase interference.

4. Test Inference:
   For test cohort {x*_1, ..., x*_M}:
       G_test[m, i] = (1 - λ) · K_RBF(x*_m, x_i) + λ · K_Quantum(x*_m, x_i)
       y_pred = sign(∑_{i ∈ SV} α_i y_i G_test[m, i] + b)
"""
import sys
from pathlib import Path
import time
import numpy as np
from sklearn.metrics import accuracy_score, fbeta_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qmed.circuits import build_feature_map
from qmed.config import DATASETS
from qmed.data import prepare_data


class CompositeMercerSVM:
    """Precomputed SVM operating on a convex combination of Classical RBF and Quantum Gram matrices."""
    def __init__(self, fmap, gamma=None, lam=0.5, C=1.0):
        self.fmap = fmap
        self.kernel_q = FidelityQuantumKernel(feature_map=fmap)
        self.gamma = gamma
        self.lam = lam
        self.C = C
        self.svm = SVC(kernel="precomputed", probability=True, C=C, random_state=42)
        self.X_train_q = None

    def fit(self, X_train_q: np.ndarray, y_train: np.ndarray):
        self.X_train_q = X_train_q
        gamma = self.gamma if self.gamma is not None else 1.0 / X_train_q.shape[1]

        # 1. Classical RBF Gram Matrix
        G_rbf = rbf_kernel(X_train_q, X_train_q, gamma=gamma)

        # 2. Quantum Fidelity Gram Matrix
        G_quantum = self.kernel_q.evaluate(X_train_q, X_train_q)

        # 3. Composite Gram Matrix (PSD)
        G_composite = (1.0 - self.lam) * G_rbf + self.lam * G_quantum

        # Fit precomputed SVM
        self.svm.fit(G_composite, y_train)
        return self

    def _compute_test_gram(self, X_test_q: np.ndarray) -> np.ndarray:
        gamma = self.gamma if self.gamma is not None else 1.0 / self.X_train_q.shape[1]
        G_rbf_test = rbf_kernel(X_test_q, self.X_train_q, gamma=gamma)
        G_quantum_test = self.kernel_q.evaluate(X_test_q, self.X_train_q)
        return (1.0 - self.lam) * G_rbf_test + self.lam * G_quantum_test

    def predict_proba(self, X_test_q: np.ndarray) -> np.ndarray:
        G_test = self._compute_test_gram(X_test_q)
        return self.svm.predict_proba(G_test)

    def predict(self, X_test_q: np.ndarray) -> np.ndarray:
        G_test = self._compute_test_gram(X_test_q)
        return self.svm.predict(G_test)


def run_approach_3(X_train_q, y_train, X_test_q, y_test, n_qubits=3, lam=0.5, C=1.0):
    t_start = time.time()
    np.random.seed(42)

    fmap = build_feature_map("ZZFeatureMap", n_qubits, 2, "linear")
    model = CompositeMercerSVM(fmap=fmap, lam=lam, C=C)
    model.fit(X_train_q, y_train)

    total_time = time.time() - t_start

    probs = model.predict_proba(X_test_q)[:, 1]
    preds = model.predict(X_test_q)

    return {
        "Accuracy": float(accuracy_score(y_test, preds)),
        "F2-Score": float(fbeta_score(y_test, preds, beta=2, zero_division=0)),
        "Precision": float(precision_score(y_test, preds, zero_division=0)),
        "Recall": float(recall_score(y_test, preds, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_test, probs)) if len(np.unique(y_test)) > 1 else 0.5,
        "Execution_Time": total_time,
        "Model": model,
    }


if __name__ == "__main__":
    print("Testing Approach 3: Composite Mercer Kernel on Wisconsin Breast Cancer...")
    d = prepare_data(DATASETS["Wisconsin Breast Cancer (Oncology)"], max_samples=100, n_qubits=3)
    res = run_approach_3(d.X_train_q, d.y_train, d.X_test_q, d.y_test, n_qubits=3, lam=0.5)
    print(f"Results: Acc={res['Accuracy']:.3f}, F2={res['F2-Score']:.3f}, Recall={res['Recall']:.3f}, Time={res['Execution_Time']:.2f}s")
