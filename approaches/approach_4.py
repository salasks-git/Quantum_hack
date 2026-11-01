"""Approach 4: Quantum-Classical Residual Learning (ResNet-Style Quantum Error Corrector)
======================================================================================

Architecture & Mathematical Details:
-------------------------------------
Inspired by deep Residual Networks (ResNets) and Gradient Boosting, Approach 4 partitions 
the diagnostic task into a coarse classical foundation and a fine quantum phase correction.

1. Classical Base Learner:
   A robust classical model (Logistic Regression or Linear SVM with Platt scaling) is fit on 
   the entire training cohort to capture the dominant, linearly separable disease features:
       f_classical(x) = σ(w^T x + b) ∈ [0, 1]

2. Residual Error Formulation:
   Rather than asking the quantum computer to learn the entire classification task from scratch 
   (which exposes it to NISQ hardware noise and barren plateaus), we compute the diagnostic residual:
       Δ_i = y_i - f_classical(x_i) ∈ [-1.0, +1.0]
   - Δ_i ≈ 0: The classical model is already highly confident and correct.
   - |Δ_i| >> 0: Borderline or deceptive clinical edge cases where classical Euclidean math failed.

3. Quantum Residual Estimator:
   A Quantum Kernel Support Vector Regressor (QSVR) with a ZZFeatureMap is trained strictly 
   to predict the continuous residual errors:
       g_quantum(x) = ∑_{i ∈ SV} (α_i - α*_i) K_Quantum(x_i, x) + b_q

4. Additive Composite Prediction:
   The final diagnosis is computed as the residual-corrected probability:
       P_final(x) = clip(f_classical(x) + α · g_quantum(x), 0.0, 1.0)
       y_pred = 𝟙[P_final ≥ 0.5]
   where α ∈ [0, 1] is the quantum correction shrinkage rate (default α = 0.5).

Noise Insulation Benefit:
-------------------------
If physical quantum hardware experiences thermal drift or gate dephasing, the classical 
foundation guarantees a stable baseline prediction. The quantum processor is only queried 
for the difficult, non-local residual discrepancies.
"""
import sys
from pathlib import Path
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, fbeta_score, precision_score, recall_score, roc_auc_score
from sklearn.svm import SVR

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qmed.circuits import build_feature_map
from qmed.config import DATASETS
from qmed.data import prepare_data


class QuantumResidualLearner:
    """Additive residual model: Classical Base Learner + Quantum Kernel Residual Corrector."""
    def __init__(self, fmap, alpha=0.5):
        self.fmap = fmap
        self.alpha = alpha
        self.base_classical = LogisticRegression(random_state=42)
        self.kernel_q = FidelityQuantumKernel(feature_map=fmap)
        self.quantum_svr = SVR(kernel="precomputed", C=1.0, epsilon=0.05)
        self.X_train_q = None

    def fit(self, X_train_q: np.ndarray, y_train: np.ndarray):
        self.X_train_q = X_train_q
        # 1. Fit classical base learner
        self.base_classical.fit(X_train_q, y_train)
        p_base = self.base_classical.predict_proba(X_train_q)[:, 1]

        # 2. Compute residual
        residuals = y_train.astype(float) - p_base

        # 3. Fit Quantum SVR on precomputed quantum Gram matrix
        G_train = self.kernel_q.evaluate(X_train_q, X_train_q)
        self.quantum_svr.fit(G_train, residuals)
        return self

    def predict_proba(self, X_test_q: np.ndarray) -> np.ndarray:
        p_base = self.base_classical.predict_proba(X_test_q)[:, 1]
        G_test = self.kernel_q.evaluate(X_test_q, self.X_train_q)
        delta_q = self.quantum_svr.predict(G_test)
        p_final = np.clip(p_base + self.alpha * delta_q, 0.0, 1.0)
        return np.column_stack([1.0 - p_final, p_final])

    def predict(self, X_test_q: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X_test_q)[:, 1] >= 0.5).astype(int)


def run_approach_4(X_train_q, y_train, X_test_q, y_test, n_qubits=3, alpha=0.5):
    t_start = time.time()
    np.random.seed(42)

    fmap = build_feature_map("ZZFeatureMap", n_qubits, 2, "linear")
    model = QuantumResidualLearner(fmap=fmap, alpha=alpha)
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
    print("Testing Approach 4: Quantum Residual Learning on Wisconsin Breast Cancer...")
    d = prepare_data(DATASETS["Wisconsin Breast Cancer (Oncology)"], max_samples=100, n_qubits=3)
    res = run_approach_4(d.X_train_q, d.y_train, d.X_test_q, d.y_test, n_qubits=3, alpha=0.5)
    print(f"Results: Acc={res['Accuracy']:.3f}, F2={res['F2-Score']:.3f}, Recall={res['Recall']:.3f}, Time={res['Execution_Time']:.2f}s")
