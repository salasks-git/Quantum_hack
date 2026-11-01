"""QuHaMA: Quantum-Hybrid Adaptive Multi-branch Architecture.

Pipeline:
1. Feature Compression Backbone: Compresses raw clinical features (d-dim) into
   k latent features (where k = n_qubits) using a PyTorch neural encoder.
2. Dual-Branch Processing:
   - Classical Head: Latent features proceed through a deep neural classifier
     to produce classical prediction probabilities P_classical.
   - Quantum Circuit: Latent features are scaled to rotation angles [0, π] and fed
     into a parameterized quantum circuit (QSVC or VQC) to compute quantum
     phase probabilities P_quantum.
3. Adaptive Gating Meta-Network:
   - Takes the latent context and the intermediate predictions [z, P_c, P_q] and
     trains to dynamically learn the optimal weightage [w_classical, w_quantum]
     for each patient, maximizing overall diagnostic accuracy and F2-score.
"""
from dataclasses import dataclass
import time
from typing import List, Optional

import numpy as np
from sklearn.metrics import accuracy_score, fbeta_score, precision_score, recall_score, roc_auc_score
import torch
import torch.nn as nn

from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel


class QuHaMAEncoder(nn.Module):
    """Compresses raw clinical measurements into k latent dimensions (k = n_qubits)."""
    def __init__(self, in_features: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, latent_dim),
            nn.Tanh(),  # Bounded to [-1, 1], smoothly map to [0, π]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QuHaMAClassicalHead(nn.Module):
    """Deep classical classification head operating on the latent features."""
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
        # Initialize output bias to equal logits so initial weights are balanced (~50/50)
        with torch.no_grad():
            self.net[2].bias.fill_(0.0)

    def forward(self, z: torch.Tensor, p_class: torch.Tensor, p_quant: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([z, p_class.unsqueeze(-1), p_quant.unsqueeze(-1)], dim=-1)
        logits = self.net(inputs)
        return torch.softmax(logits, dim=-1)


class QuHaMAModel:
    """Trained QuHaMA composite model supporting inference and component breakdown."""
    def __init__(
        self,
        encoder: QuHaMAEncoder,
        classical_head: QuHaMAClassicalHead,
        quantum_model,
        gating_net: QuHaMAGatingNetwork,
        latent_dim: int,
    ):
        self.encoder = encoder
        self.classical_head = classical_head
        self.quantum_model = quantum_model
        self.gating_net = gating_net
        self.latent_dim = latent_dim
        self.encoder.eval()
        self.classical_head.eval()
        self.gating_net.eval()

    def get_breakdown(self, X_std: np.ndarray) -> dict:
        """Computes complete step-by-step diagnostic breakdown for an array of patients."""
        with torch.no_grad():
            x_t = torch.tensor(X_std, dtype=torch.float32)
            z_latent = self.encoder(x_t).numpy()
            z_angles = (z_latent + 1.0) * (np.pi / 2.0)  # Map [-1, 1] -> [0, π]

            probs_class = self.classical_head(torch.tensor(z_latent, dtype=torch.float32)).numpy()
            p_class = probs_class[:, 1]

        # Quantum branch inference
        if hasattr(self.quantum_model, "decision_function"):
            dec_q = self.quantum_model.decision_function(z_angles)
            p_quant = 1.0 / (1.0 + np.exp(-dec_q))
        else:
            p_quant = self.quantum_model.predict_proba(z_angles)[:, 1]

        # Gating meta-network inference
        with torch.no_grad():
            p_c_t = torch.tensor(p_class, dtype=torch.float32)
            p_q_t = torch.tensor(p_quant, dtype=torch.float32)
            z_t = torch.tensor(z_latent, dtype=torch.float32)
            weights = self.gating_net(z_t, p_c_t, p_q_t).numpy()

        p_final = weights[:, 0] * p_class + weights[:, 1] * p_quant
        pred = (p_final >= 0.5).astype(int)

        return {
            "z_latent": z_latent,
            "z_angles": z_angles,
            "p_classical": p_class,
            "p_quantum": p_quant,
            "w_classical": weights[:, 0],
            "w_quantum": weights[:, 1],
            "weights": weights,
            "p_final": p_final,
            "pred": pred,
        }

    def predict_proba(self, X_std: np.ndarray) -> np.ndarray:
        res = self.get_breakdown(X_std)
        p1 = res["p_final"]
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X_std: np.ndarray) -> np.ndarray:
        return self.get_breakdown(X_std)["pred"]


@dataclass
class QuHaMAResult:
    model: QuHaMAModel
    metrics: dict
    loss_history_encoder: List[float]
    loss_history_gating: List[float]
    breakdown_test: dict
    train_time: float
    inf_time: float


def train_quhama(
    X_train_std: np.ndarray,
    y_train: np.ndarray,
    X_test_std: np.ndarray,
    y_test: np.ndarray,
    n_qubits: int,
    feature_map,
    epochs_enc: int = 70,
    epochs_gate: int = 50,
    lr: float = 0.015,
) -> QuHaMAResult:
    """Trains the full QuHaMA architecture:

    1. Classical Encoder + Head on raw standardized features.
    2. Quantum Circuit on compressed latent angles [0, π].
    3. Gating Meta-Network learning the dynamic weighting.
    """
    t_start = time.time()
    torch.manual_seed(42)
    np.random.seed(42)

    in_features = X_train_std.shape[1]
    latent_dim = n_qubits

    # 1. Initialize PyTorch Encoder & Classical Head
    encoder = QuHaMAEncoder(in_features, latent_dim)
    class_head = QuHaMAClassicalHead(latent_dim)
    optimizer_enc = torch.optim.Adam(
        list(encoder.parameters()) + list(class_head.parameters()), lr=lr, weight_decay=1e-4
    )
    criterion_ce = nn.CrossEntropyLoss()

    X_tr_t = torch.tensor(X_train_std, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train, dtype=torch.long)
    X_te_t = torch.tensor(X_test_std, dtype=torch.float32)

    loss_history_enc = []
    encoder.train()
    class_head.train()
    for _ in range(epochs_enc):
        optimizer_enc.zero_grad()
        z = encoder(X_tr_t)
        logits = class_head(z)
        loss = criterion_ce(logits, y_tr_t)
        loss.backward()
        optimizer_enc.step()
        loss_history_enc.append(float(loss.item()))

    # Extract latent angles
    encoder.eval()
    class_head.eval()
    with torch.no_grad():
        z_tr = encoder(X_tr_t).numpy()
        z_te = encoder(X_te_t).numpy()
        z_tr_angles = (z_tr + 1.0) * (np.pi / 2.0)
        z_te_angles = (z_te + 1.0) * (np.pi / 2.0)
        p_class_tr = class_head(torch.tensor(z_tr, dtype=torch.float32)).numpy()[:, 1]
        p_class_te = class_head(torch.tensor(z_te, dtype=torch.float32)).numpy()[:, 1]

    # 2. Quantum Branch: Train QSVC Kernel on Latent Angles
    kernel = FidelityQuantumKernel(feature_map=feature_map)
    qsvc = QSVC(quantum_kernel=kernel)
    qsvc.fit(z_tr_angles, y_train)

    dec_tr = qsvc.decision_function(z_tr_angles)
    dec_te = qsvc.decision_function(z_te_angles)
    p_quant_tr = 1.0 / (1.0 + np.exp(-dec_tr))
    p_quant_te = 1.0 / (1.0 + np.exp(-dec_te))

    # 3. Gating Meta-Network: Learns Adaptive Weights
    gating_net = QuHaMAGatingNetwork(latent_dim)
    optimizer_gate = torch.optim.Adam(gating_net.parameters(), lr=0.01)
    criterion_bce = nn.BCELoss()

    y_tr_f = torch.tensor(y_train, dtype=torch.float32)
    p_c_tr_t = torch.tensor(p_class_tr, dtype=torch.float32)
    p_q_tr_t = torch.tensor(p_quant_tr, dtype=torch.float32)
    z_tr_t = torch.tensor(z_tr, dtype=torch.float32)

    loss_history_gate = []
    gating_net.train()
    for _ in range(epochs_gate):
        optimizer_gate.zero_grad()
        weights = gating_net(z_tr_t, p_c_tr_t, p_q_tr_t)
        p_fused = weights[:, 0] * p_c_tr_t + weights[:, 1] * p_q_tr_t
        # Optional entropy regularizer to avoid collapsing completely to one branch
        entropy = -torch.mean(torch.sum(weights * torch.log(weights + 1e-8), dim=-1))
        loss = criterion_bce(p_fused, y_tr_f) - 0.05 * entropy
        loss.backward()
        optimizer_gate.step()
        loss_history_gate.append(float(loss.item()))

    t_train = time.time() - t_start

    # Evaluate on Held-Out Test Cohort
    t_inf_start = time.time()
    model = QuHaMAModel(encoder, class_head, qsvc, gating_net, latent_dim)
    breakdown_test = model.get_breakdown(X_test_std)
    t_inf = time.time() - t_inf_start

    y_pred = breakdown_test["pred"]
    y_prob = breakdown_test["p_final"]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "F2-Score": fbeta_score(y_test, y_pred, beta=2, zero_division=0),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5,
    }

    return QuHaMAResult(
        model=model,
        metrics=metrics,
        loss_history_encoder=loss_history_enc,
        loss_history_gating=loss_history_gate,
        breakdown_test=breakdown_test,
        train_time=t_train,
        inf_time=t_inf,
    )
