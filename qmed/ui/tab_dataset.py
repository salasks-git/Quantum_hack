"""Tab 1: dataset summary and patients plotted as qubit angles."""
import numpy as np
import streamlit as st

from qmed.data import PreparedData
from qmed.ui import plots
from qmed.ui.sidebar import PipelineConfig


def render(cfg: PipelineConfig, data: PreparedData) -> None:
    st.subheader(f"Dataset: {cfg.dataset_label}")
    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown(f"""
        - **Original Clinical Features:** `{len(data.feature_names)}` continuous measurements
        - **Active Cohort Size:** `{len(data.X_train_q) + len(data.X_test_q)}` patients (`{len(data.X_train_q)}` train / `{len(data.X_test_q)}` test)
        - **Variance Captured by {cfg.n_qubits} PCA Components:** `{np.sum(data.pca.explained_variance_ratio_):.1%}`
        - **Quantum Angle Mapping:** Standardized $\\to$ PCA({cfg.n_qubits}) $\\to$ Scaled to $[0, \\pi]$ radians to prevent $2\\pi$ Bloch sphere phase aliasing.
        - **Preprocessing Computation Time:** `{data.prep_time*1000:.2f} ms`
        """)
        st.dataframe(data.df.head(8), use_container_width=True)
    with col2:
        plots.show(plots.patient_scatter(data.X_train_q, data.y_train))
