"""Approach 6: Bandwidth-Scaled Centered Kernel-Target Alignment (Q-CKTA) Hybrid
=============================================================================

Architecture & Mathematical Details:
-------------------------------------
Approach 6 tackles the fundamental "quantum kernel concentration problem" (Huang et al., 
Science 2021; Shaydulin & Wild 2022). Unscaled quantum kernels (s = 1.0) often suffer from 
near-identity Gram matrices (memorization) or exponential concentration in Hilbert space.

1. Quantum Kernel Bandwidth Parameterization:
   We scale the feature map input angles by a tunable bandwidth factor s > 0:
       θ_s = s · x ∈ [0, s · π]^n
       K_Quantum(x, x'; s) = |⟨ψ(s · x) | ψ(s · x')⟩|²
   - Large s (s = 1.0): Kernel becomes overly sharp (approaching identity matrix).
   - Small s (s → 0): Kernel approaches all-ones matrix (vanishing gradient).
   - Optimal s* (s ∈ [0.05, 0.3]): Preserves non-local wave contrast without exponential concentration.

2. Centered Kernel-Target Alignment (CKTA) Metric:
   CKTA (Cortes et al., 2012) measures the Frobenius inner product between the centered 
   kernel matrix K_c and the centered ideal label agreement matrix Y_c = H (y y^T) H:
       CKTA(K, Y) = ⟨K_c, Y_c⟩_F / (||K_c||_F · ||Y_c||_F)
   Higher alignment indicates the kernel geometry naturally clusters identical diagnoses 
   before any classifier optimization begins.

3. Data-Driven Alignment-Weighted Hybrid Kernel:
   We evaluate CKTA across a candidate sweep of quantum bandwidths s ∈ {0.05, 0.1, 0.2, 0.5, 1.0}:
       s* = argmax_s CKTA(K_Quantum(s), Y)
       A_q = CKTA(K_Quantum(s*), Y)
       A_c = CKTA(K_RBF, Y)
   The fusion weights are set strictly by mathematical alignment:
       w_c = max(A_c, 1e-4) / (max(A_c, 1e-4) + max(A_q, 1e-4))
       w_q = 1.0 - w_c
       K_Hybrid(s*) = w_c · K_RBF + w_q · K_Quantum(s*)

4. Optimal SVM Classification:
   An SVM with kernel="precomputed" is trained on the alignment-optimal hybrid Gram matrix, 
   guaranteeing a data-driven balance between Euclidean proximity and quantum phase coherence.
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

from qmed.bandwidth import kernel_diagnostics
from qmed.circuits import build_feature_map
from qmed.config import DATASETS
from qmed.data import prepare_data
from qmed.models import build_quantum_kernel


class CKTAHybridSVM:
    """Hybrid SVM whose combination weights are derived from Centered Kernel-Target Alignment."""
    def __init__(self, fmap, bandwidth_candidates=(0.05, 0.1, 0.2, 0.5, 1.0)):
        self.fmap = fmap
        self.bandwidth_candidates = bandwidth_candidates
        self.best_s = 0.1
        self.w_c = 0.5
        self.w_q = 0.5
        self.best_quantum_kernel = None
        self.svm = SVC(kernel="precomputed", probability=True, C=1.0, random_state=42)
        self.X_train_q = None

    def fit(self, X_train_q: np.ndarray, y_train: np.ndarray):
        self.X_train_q = X_train_q
        gamma = 1.0 / X_train_q.shape[1]

        # 1. Classical RBF Gram Matrix & CKTA
        G_rbf = rbf_kernel(X_train_q, X_train_q, gamma=gamma)
        diag_rbf = kernel_diagnostics(G_rbf, y_train)
        A_c = max(diag_rbf["alignment"], 1e-4)

        # 2. Sweep Quantum Bandwidths to find s* maximizing CKTA
        best_align = -1.0
        best_G_q = None
        best_kernel = None

        for s in self.bandwidth_candidates:
            q_ker = build_quantum_kernel(self.fmap, bandwidth=s)
            G_q = q_ker.evaluate(X_train_q, X_train_q)
            diag_q = kernel_diagnostics(G_q, y_train)
            align = diag_q["alignment"]
            if align > best_align:
                best_align = align
                self.best_s = s
                best_G_q = G_q
                best_kernel = q_ker

        A_q = max(best_align, 1e-4)
        self.best_quantum_kernel = best_kernel

        # 3. Compute Data-Driven Alignment Weights
        self.w_c = float(A_c / (A_c + A_q))
        self.w_q = float(1.0 - self.w_c)

        # 4. Form optimal hybrid Gram matrix and fit SVM
        G_hybrid = self.w_c * G_rbf + self.w_q * best_G_q
        self.svm.fit(G_hybrid, y_train)
        return self

    def _compute_test_gram(self, X_test_q: np.ndarray) -> np.ndarray:
        gamma = 1.0 / self.X_train_q.shape[1]
        G_rbf_test = rbf_kernel(X_test_q, self.X_train_q, gamma=gamma)
        G_q_test = self.best_quantum_kernel.evaluate(X_test_q, self.X_train_q)
        return self.w_c * G_rbf_test + self.w_q * G_q_test

    def predict_proba(self, X_test_q: np.ndarray) -> np.ndarray:
        G_test = self._compute_test_gram(X_test_q)
        return self.svm.predict_proba(G_test)

    def predict(self, X_test_q: np.ndarray) -> np.ndarray:
        G_test = self._compute_test_gram(X_test_q)
        return self.svm.predict(G_test)


def run_approach_6(X_train_q, y_train, X_test_q, y_test, n_qubits=3):
    t_start = time.time()
    np.random.seed(42)

    fmap = build_feature_map("ZZFeatureMap", n_qubits, 2, "linear")
    model = CKTAHybridSVM(fmap=fmap)
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
        "Best_Bandwidth_s": model.best_s,
        "Weight_Classical": model.w_c,
        "Weight_Quantum": model.w_q,
        "Execution_Time": total_time,
        "Model": model,
    }


if __name__ == "__main__":
    print("Testing Approach 6: Q-CKTA Hybrid on Wisconsin Breast Cancer...")
    d = prepare_data(DATASETS["Wisconsin Breast Cancer (Oncology)"], max_samples=100, n_qubits=3)
    res = run_approach_6(d.X_train_q, d.y_train, d.X_test_q, d.y_test, n_qubits=3)
    print(f"Results: Acc={res['Accuracy']:.3f}, F2={res['F2-Score']:.3f}, Best_s={res['Best_Bandwidth_s']}, w_q={res['Weight_Quantum']:.2f}, Time={res['Execution_Time']:.2f}s")
