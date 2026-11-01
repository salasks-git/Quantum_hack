"""Approach 1: QuHaMA (Quantum-Hybrid Adaptive Multi-branch Architecture)
======================================================================

Architecture & Mathematical Details:
-------------------------------------
QuHaMA is an end-to-end adaptive multi-branch hybrid architecture designed to solve the 
dimensional bottleneck and hardware noise of NISQ quantum computing:

1. Feature Compression Backbone:
   A PyTorch non-linear neural encoder E(x) compresses high-dimensional clinical 
   biomarkers x ∈ ℝ^d (e.g., 30 features from fine needle aspirates) into k latent 
   dimensions z ∈ [-1, 1]^k, where k equals the physical qubit count (k = n_qubits):
       z = tanh(W_2 · ReLU(W_1 x + b_1) + b_2)

2. Dual-Branch Processing:
   - Branch A (Classical Deep Head):
     Latent features z proceed through deep non-linear layers to output classical 
     posterior disease probability P_classical(y=1 | x):
         P_classical = Softmax(W_c2 · ReLU(W_c1 z + b_c1) + b_c2)
   - Branch B (Parameterized Quantum Circuit):
     Latent features z are mapped to rotation angles on the Bloch sphere:
         θ = (z + 1) · (π / 2) ∈ [0, π]^k
     A quantum circuit evaluates non-local state fidelity in 2^k-dimensional Hilbert 
     space via a ZZFeatureMap, outputting quantum phase probability P_quantum(y=1 | x).

3. Adaptive Gating Meta-Network:
   An auxiliary neural network G(z, P_c, P_q) dynamically learns patient-specific 
   gating weights [w_classical, w_quantum] through supervised backpropagation:
       [w_classical, w_quantum] = Softmax(W_g · [z, P_c, P_q] + b_g)
       w_classical + w_quantum = 1.0

4. Final Decision Fusion:
   The final diagnosis is computed as the dynamically weighted soft consensus:
       P_final = w_classical · P_classical + w_quantum · P_quantum
       y_pred = 𝟙[P_final ≥ 0.5]
"""
import sys
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, fbeta_score, precision_score, recall_score, roc_auc_score

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qmed.circuits import build_feature_map
from qmed.config import DATASETS
from qmed.data import prepare_data


class QuHaMAEncoder(nn.Module):
    """Compresses raw clinical measurements into k latent dimensions (k = n_qubits)."""
    def __init__(self, in_features: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, latent_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QuHaMAClassicalHead(nn.Module):
    """Deep classical classification head operating on latent features."""
    def __init__(self, latent_dim: int, hidden_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(z), dim=-1)


class QuHaMAGatingNetwork(nn.Module):
    """Adaptive meta-network learning patient-specific weights for Classical vs Quantum branches."""
    def __init__(self, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 2, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )
        with torch.no_grad():
            self.net[2].bias.fill_(0.0)

    def forward(self, z: torch.Tensor, p_class: torch.Tensor, p_quant: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([z, p_class.unsqueeze(-1), p_quant.unsqueeze(-1)], dim=-1)
        return torch.softmax(self.net(inputs), dim=-1)


class QuHaMAModel:
    """Trained QuHaMA composite model."""
    def __init__(self, encoder, classical_head, quantum_model, gating_net, latent_dim):
        self.encoder = encoder
        self.classical_head = classical_head
        self.quantum_model = quantum_model
        self.gating_net = gating_net
        self.latent_dim = latent_dim
        self.encoder.eval()
        self.classical_head.eval()
        self.gating_net.eval()

    def predict_proba(self, X_std: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x_t = torch.tensor(X_std, dtype=torch.float32)
            z_latent = self.encoder(x_t).numpy()
            z_angles = (z_latent + 1.0) * (np.pi / 2.0)
            p_class = self.classical_head(torch.tensor(z_latent, dtype=torch.float32)).numpy()[:, 1]

        dec_q = self.quantum_model.decision_function(z_angles)
        p_quant = 1.0 / (1.0 + np.exp(-dec_q))

        with torch.no_grad():
            w = self.gating_net(
                torch.tensor(z_latent, dtype=torch.float32),
                torch.tensor(p_class, dtype=torch.float32),
                torch.tensor(p_quant, dtype=torch.float32),
            ).numpy()

        p_final = w[:, 0] * p_class + w[:, 1] * p_quant
        return np.column_stack([1.0 - p_final, p_final])

    def predict(self, X_std: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X_std)[:, 1] >= 0.5).astype(int)


def run_approach_1(X_train_std, y_train, X_test_std, y_test, n_qubits=3):
    t_start = time.time()
    torch.manual_seed(42)
    np.random.seed(42)

    in_features = X_train_std.shape[1]
    latent_dim = n_qubits

    encoder = QuHaMAEncoder(in_features, latent_dim)
    class_head = QuHaMAClassicalHead(latent_dim)
    optimizer_enc = torch.optim.Adam(
        list(encoder.parameters()) + list(class_head.parameters()), lr=0.015, weight_decay=1e-4
    )
    criterion_ce = nn.CrossEntropyLoss()

    X_tr_t = torch.tensor(X_train_std, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train, dtype=torch.long)
    X_te_t = torch.tensor(X_test_std, dtype=torch.float32)

    encoder.train()
    class_head.train()
    for _ in range(60):
        optimizer_enc.zero_grad()
        z = encoder(X_tr_t)
        loss = criterion_ce(class_head(z), y_tr_t)
        loss.backward()
        optimizer_enc.step()

    encoder.eval()
    class_head.eval()
    with torch.no_grad():
        z_tr = encoder(X_tr_t).numpy()
        z_te = encoder(X_te_t).numpy()
        z_tr_angles = (z_tr + 1.0) * (np.pi / 2.0)
        z_te_angles = (z_te + 1.0) * (np.pi / 2.0)
        p_class_tr = class_head(torch.tensor(z_tr, dtype=torch.float32)).numpy()[:, 1]
        p_class_te = class_head(torch.tensor(z_te, dtype=torch.float32)).numpy()[:, 1]

    # Quantum Kernel on latent angles
    fmap = build_feature_map("ZZFeatureMap", latent_dim, 2, "linear")
    kernel = FidelityQuantumKernel(feature_map=fmap)
    qsvc = QSVC(quantum_kernel=kernel)
    qsvc.fit(z_tr_angles, y_train)

    dec_tr = qsvc.decision_function(z_tr_angles)
    dec_te = qsvc.decision_function(z_te_angles)
    p_quant_tr = 1.0 / (1.0 + np.exp(-dec_tr))
    p_quant_te = 1.0 / (1.0 + np.exp(-dec_te))

    # Gating Network
    gating_net = QuHaMAGatingNetwork(latent_dim)
    opt_gate = torch.optim.Adam(gating_net.parameters(), lr=0.01)
    crit_bce = nn.BCELoss()

    y_tr_f = torch.tensor(y_train, dtype=torch.float32)
    p_c_tr_t = torch.tensor(p_class_tr, dtype=torch.float32)
    p_q_tr_t = torch.tensor(p_quant_tr, dtype=torch.float32)
    z_tr_t = torch.tensor(z_tr, dtype=torch.float32)

    gating_net.train()
    for _ in range(45):
        opt_gate.zero_grad()
        w = gating_net(z_tr_t, p_c_tr_t, p_q_tr_t)
        p_fused = w[:, 0] * p_c_tr_t + w[:, 1] * p_q_tr_t
        entropy = -torch.mean(torch.sum(w * torch.log(w + 1e-8), dim=-1))
        loss = crit_bce(p_fused, y_tr_f) - 0.05 * entropy
        loss.backward()
        opt_gate.step()

    total_time = time.time() - t_start

    model = QuHaMAModel(encoder, class_head, qsvc, gating_net, latent_dim)
    probs = model.predict_proba(X_test_std)[:, 1]
    preds = (probs >= 0.5).astype(int)

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
    print("Testing Approach 1: QuHaMA on Wisconsin Breast Cancer...")
    d = prepare_data(DATASETS["Wisconsin Breast Cancer (Oncology)"], max_samples=100, n_qubits=3)
    res = run_approach_1(d.X_train_std, d.y_train, d.X_test_std, d.y_test, n_qubits=3)
    print(f"Results: Acc={res['Accuracy']:.3f}, F2={res['F2-Score']:.3f}, Recall={res['Recall']:.3f}, Time={res['Execution_Time']:.2f}s")
