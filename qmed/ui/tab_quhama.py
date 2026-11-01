"""Tab 8: QuHaMA - Quantum-Hybrid Adaptive Multi-branch Architecture.

Features:
- Dual-branch Neural Encoder + Quantum Circuit + Adaptive Gating Meta-Network.
- Architecture flow visualizer.
- Apples-to-apples comparison: Classical Branch vs Quantum Branch vs QuHaMA Fusion.
- Loss convergence curves and per-patient learned gating distribution.
- Live patient diagnostic tester with dynamic gating weight allocation.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from qmed.circuits import build_feature_map
from qmed.data import PreparedData
from qmed.quhama import QuHaMAResult, train_quhama
from qmed.ui import plots
from qmed.ui.sidebar import PipelineConfig


@st.cache_resource(show_spinner=False)
def _cached_quhama_train(
    dataset_label: str,
    n_samples: int,
    n_qubits: int,
    feature_map_name: str,
    reps: int,
    entanglement: str,
    X_train_std: np.ndarray,
    y_train: np.ndarray,
    X_test_std: np.ndarray,
    y_test: np.ndarray,
) -> QuHaMAResult:
    fmap = build_feature_map(feature_map_name, n_qubits, reps, entanglement)
    return train_quhama(
        X_train_std=X_train_std,
        y_train=y_train,
        X_test_std=X_test_std,
        y_test=y_test,
        n_qubits=n_qubits,
        feature_map=fmap,
        epochs_enc=60,
        epochs_gate=45,
    )


def render(cfg: PipelineConfig, data: PreparedData, results: dict) -> None:
    st.subheader("🧠 QuHaMA: Quantum-Hybrid Adaptive Multi-branch Architecture")
    st.markdown(
        """
        **QuHaMA** is a custom multi-branch architecture designed to solve the feature-dimension and hardware-noise 
        trade-off in NISQ computing:
        1. **Classical Feature Compression Backbone:** Raw clinical measurements ($d$ dimensions) are compressed into $k$ latent features ($k = n_{\\text{qubits}}$) via a PyTorch neural encoder.
        2. **Dual-Branch Execution:**
           - **Classical Deep Pathway:** Latent features $\\vec{z}$ pass through deep linear/ReLU layers to compute $P_{\\text{classical}}$.
           - **Quantum Circuit Pathway:** Latent features $\\vec{z}$ are mapped to rotation angles $[0, \\pi]$ and fed into a parameterized quantum circuit in $2^k$-dimensional Hilbert space to compute $P_{\\text{quantum}}$.
        3. **Adaptive Gating Meta-Network:** A specialized neural network trains to dynamically learn the **optimal patient-specific weightage** $[w_{\\text{classical}}(\\vec{x}), w_{\\text{quantum}}(\\vec{x})]$ that maximizes predictive accuracy and clinical $F_2$-score.
        """
    )

    with st.expander("📐 View QuHaMA Architectural Diagram", expanded=False):
        st.markdown(
            """
            ```
                     Raw Patient Clinical Biomarkers x⃗ ∈ ℝᵈ
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │   Neural Encoder E(x⃗)     │
                        │  Linear(d, 16) → Tanh     │
                        └─────────────┬─────────────┘
                                      │ Compressed Latent Vector z⃗ ∈ ℝᵏ (k = n_qubits)
                        ┌─────────────┴─────────────┐
                        ▼                           ▼
            ┌───────────────────────┐   ┌───────────────────────┐
            │ Classical Deep Head   │   │ Latent Angle Scaling  │
            │ Linear(k, 8) → ReLU   │   │ θ⃗ = (z⃗ + 1) · (π / 2) │
            │ Linear(8, 2) → Softmax│   └───────────┬───────────┘
            └───────────┬───────────┘               ▼
                        │               ┌───────────────────────┐
                        │               │ Quantum Circuit (PQC) │
                        │               │ ZZFeatureMap in 2ᵏ-D  │
                        │               └───────────┬───────────┘
                        │                           │
                   P_classical                 P_quantum
                        │                           │
                        └─────────────┬─────────────┘
                                      ▼
                      ┌───────────────────────────────┐
                      │  Adaptive Gating Meta-Network │
                      │       G(z⃗, P_c, P_q)          │
                      │   [w_classical, w_quantum]    │
                      └───────────────┬───────────────┘
                                      ▼
                     P_QuHaMA = w_c · P_c + w_q · P_q
                                      │
                                      ▼
                             🩺 Final Diagnosis
            ```
            """
        )

    st.divider()

    # Train QuHaMA
    with st.spinner(f"Training QuHaMA Dual-Branch Architecture on {cfg.dataset_label}..."):
        quhama_res = _cached_quhama_train(
            dataset_label=cfg.dataset_label,
            n_samples=cfg.n_samples,
            n_qubits=cfg.n_qubits,
            feature_map_name=cfg.feature_map_name,
            reps=cfg.reps,
            entanglement=cfg.entanglement,
            X_train_std=data.X_train_std,
            y_train=data.y_train,
            X_test_std=data.X_test_std,
            y_test=data.y_test,
        )

    bk = quhama_res.breakdown_test
    m = quhama_res.metrics

    # 1. Performance Overview KPIs
    st.subheader("🏆 QuHaMA Evaluation: Adaptive Fusion vs Standalone Branches")
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.metric("QuHaMA Accuracy", f"{m['Accuracy'] * 100:.1f}%")
    with kpi2:
        st.metric("Clinical F2-Score", f"{m['F2-Score']:.3f}", help="Weighs Recall 2x over Precision")
    with kpi3:
        st.metric("Sensitivity (Recall)", f"{m['Recall'] * 100:.1f}%")
    with kpi4:
        st.metric("Precision", f"{m['Precision'] * 100:.1f}%")
    with kpi5:
        st.metric("ROC-AUC", f"{m['ROC-AUC']:.3f}")

    # Comparative branch performance
    acc_class = float(np.mean((bk["p_classical"] >= 0.5).astype(int) == data.y_test))
    acc_quant = float(np.mean((bk["p_quantum"] >= 0.5).astype(int) == data.y_test))
    mean_w_c = float(np.mean(bk["w_classical"]))
    mean_w_q = float(np.mean(bk["w_quantum"]))

    comp_df = pd.DataFrame([
        {
            "Branch / Model": "Classical Neural Head (Standalone)",
            "Test Accuracy": f"{acc_class * 100:.1f}%",
            "Mean Gating Weight Allocation": f"{mean_w_c * 100:.1f}%",
            "Role": "Local Euclidean representation in compressed space",
        },
        {
            "Branch / Model": "Quantum Circuit Branch (Standalone)",
            "Test Accuracy": f"{acc_quant * 100:.1f}%",
            "Mean Gating Weight Allocation": f"{mean_w_q * 100:.1f}%",
            "Role": f"Non-local phase interference in 2^{cfg.n_qubits}-D Hilbert space",
        },
        {
            "Branch / Model": "🧬 QuHaMA Adaptive Gated Fusion",
            "Test Accuracy": f"{m['Accuracy'] * 100:.1f}%",
            "Mean Gating Weight Allocation": "100.0% (Adaptive per patient)",
            "Role": "Dynamic meta-network consensus fusion",
        },
    ])
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    st.divider()

    # 2. Training Convergence & Gating Weight Distribution Plots
    st.subheader("📈 Training Convergence & Gating Weight Distribution")
    col_p1, col_p2 = st.columns(2)

    with col_p1:
        fig1, ax1 = plt.subplots(figsize=(6, 3.8))
        ax1.plot(quhama_res.loss_history_encoder, label="Encoder + Head CE Loss", color="#4f46e5", lw=2)
        ax1.plot(quhama_res.loss_history_gating, label="Gating Meta-Net Loss", color="#059669", lw=2)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("QuHaMA Training Convergence")
        ax1.grid(alpha=0.3)
        ax1.legend()
        fig1.tight_layout()
        plots.show(fig1)

    with col_p2:
        fig2, ax2 = plt.subplots(figsize=(6, 3.8))
        ax2.hist(bk["w_quantum"], bins=10, color="#d946ef", edgecolor="black", alpha=0.85)
        ax2.axvline(mean_w_q, color="#4f46e5", ls="--", lw=2, label=f"Mean Quantum Weight ({mean_w_q:.2f})")
        ax2.set_xlabel("Quantum Branch Weight Allocated ($w_{\\text{quantum}}$)")
        ax2.set_ylabel("Number of Patients")
        ax2.set_title("Distribution of Learned Quantum Weights on Test Cohort")
        ax2.grid(alpha=0.3)
        ax2.legend()
        fig2.tight_layout()
        plots.show(fig2)

    st.divider()

    # 3. Interactive Patient Tester with Dynamic Gating
    st.subheader("🩺 Interactive Patient Diagnosis & Dynamic Gating Inspection")
    st.caption("Select any patient from the held-out test cohort to observe how QuHaMA's meta-network dynamically assigns weights.")

    n_test = len(data.y_test)
    patient_idx = st.selectbox(
        "Select Test Cohort Patient:",
        options=list(range(n_test)),
        format_func=lambda i: f"Patient #{i+1:02d} — True Diagnosis: {'🚨 DISEASE (Positive)' if data.y_test[i] == 1 else '✅ HEALTHY (Negative)'}",
    )

    p_c_val = bk["p_classical"][patient_idx]
    p_q_val = bk["p_quantum"][patient_idx]
    w_c_val = bk["w_classical"][patient_idx]
    w_q_val = bk["w_quantum"][patient_idx]
    p_final_val = bk["p_final"][patient_idx]
    true_label = data.y_test[patient_idx]

    c_box1, c_box2, c_box3 = st.columns(3)
    with c_box1:
        st.markdown("**1. Classical Head Branch**")
        st.progress(float(p_c_val), text=f"Classical Risk: {p_c_val * 100:.1f}%")
        st.caption(f"Status: {'🚨 Flagged' if p_c_val >= 0.5 else '✅ Normal'}")

    with c_box2:
        st.markdown("**2. Quantum Circuit Branch**")
        st.progress(float(p_q_val), text=f"Quantum Phase Risk: {p_q_val * 100:.1f}%")
        st.caption(f"Status: {'🚨 Flagged' if p_q_val >= 0.5 else '✅ Normal'}")

    with c_box3:
        st.markdown("**3. Learned Gating Weightage**")
        st.write(f"Classical Weight ($w_c$): **{w_c_val * 100:.1f}%**")
        st.write(f"Quantum Weight ($w_q$): **{w_q_val * 100:.1f}%**")

    # Consensus Verdict Banner
    is_correct = (int(p_final_val >= 0.5) == true_label)
    verdict_badge = "🚨 DISEASE DETECTED" if p_final_val >= 0.5 else "✅ HEALTHY / BENIGN"
    st.markdown(
        f"""
        <div style="border: 2px solid {'#10b981' if is_correct else '#ef4444'}; border-radius: 8px; padding: 16px; background-color: {'#f0fdf4' if is_correct else '#fef2f2'}; margin-top: 10px;">
            <h4 style="margin: 0; color: {'#065f46' if is_correct else '#991b1b'};">
                QuHaMA Final Consensus Diagnosis: {verdict_badge}
            </h4>
            <p style="margin: 6px 0 0 0; color: #374151;">
                <strong>Consensus Disease Probability:</strong> {p_final_val * 100:.1f}% &nbsp;|&nbsp;
                <strong>Formula:</strong> {w_c_val:.2f} × ({p_c_val * 100:.1f}%) + {w_q_val:.2f} × ({p_q_val * 100:.1f}%) = <strong>{p_final_val * 100:.1f}%</strong> &nbsp;|&nbsp;
                <strong>Concordance:</strong> {'🎯 MATCHES TRUE DIAGNOSIS' if is_correct else '⚠️ MISCLASSIFIED'}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
