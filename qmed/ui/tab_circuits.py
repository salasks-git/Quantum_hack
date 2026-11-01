"""Tab 2: circuit diagrams, quantum kernel matrix, VQC loss curve."""
import streamlit as st

from qmed.config import QSVC_MODEL, VQC_MODEL
from qmed.ui import plots
from qmed.ui.sidebar import PipelineConfig


def render(cfg: PipelineConfig, feature_map, ansatz, results: dict) -> None:
    st.subheader("Quantum Circuit Architecture & Hilbert Space Geometry")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Feature Map (`{cfg.feature_map_name}`)** — Encodes $\\vec{{x}} \\in [0, \\pi]^{cfg.n_qubits}$ via non-linear $ZZ$ phase entanglement:")
        plots.show(plots.circuit_figure(feature_map))
    with c2:
        st.markdown(f"**Variational Ansatz (`RealAmplitudes`)** — Trainable $R_y(\\theta)$ + CNOT entanglement (`{cfg.entanglement}`):")
        plots.show(plots.circuit_figure(ansatz))

    st.divider()
    k1, k2 = st.columns(2)
    with k1:
        st.markdown("**Quantum Kernel Matrix $K(\\vec{x}_i, \\vec{x}_j) = |\\langle \\psi(\\vec{x}_i) | \\psi(\\vec{x}_j) \\rangle|^2$**")
        plots.show(plots.kernel_heatmap(results[QSVC_MODEL]["kernel_matrix"]))
    with k2:
        st.markdown("**VQC Training Convergence (COBYLA Loss Curve)**")
        plots.show(plots.loss_curve(results[VQC_MODEL]["loss_history"]))
