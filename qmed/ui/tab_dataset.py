"""Tab 1: dataset summary, patients plotted as qubit angles, and full dataset table viewer."""
import os
import numpy as np
import pandas as pd
import streamlit as st

from qmed.config import DATASETS
from qmed.data import PreparedData
from qmed.ui import plots
from qmed.ui.sidebar import PipelineConfig


def render(cfg: PipelineConfig, data: PreparedData) -> None:
    st.subheader(f"Active Cohort: {cfg.dataset_label}")
    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown(f"""
        - **Original Clinical Features:** `{len(data.feature_names)}` continuous measurements
        - **Active Subsampled Cohort:** `{len(data.X_train_q) + len(data.X_test_q)}` patients (`{len(data.X_train_q)}` train / `{len(data.X_test_q)}` test)
        - **Variance Captured by {cfg.n_qubits} PCA Components:** `{np.sum(data.pca.explained_variance_ratio_):.1%}`
        - **Quantum Angle Mapping:** Standardized $\\to$ PCA({cfg.n_qubits}) $\\to$ Scaled to $[0, \\pi]$ radians to prevent $2\\pi$ Bloch sphere phase aliasing.
        - **Preprocessing Computation Time:** `{data.prep_time*1000:.2f} ms`
        """)
        st.caption(f"Preview of active cohort records (showing 8 of {len(data.df)}):")
        st.dataframe(data.df.head(8), use_container_width=True)
    with col2:
        plots.show(plots.patient_scatter(data.X_train_q, data.y_train))

    st.divider()

    # --- INDIVIDUAL DATASETS TABLE EXPLORER ---
    st.subheader("📋 Clinical Datasets Table Explorer")
    st.markdown("Inspect any of the 5 clinical benchmark datasets as a full interactive table with feature statistics and class balances.")

    selected_view_label = st.selectbox(
        "Select Dataset to Inspect as Table:",
        list(DATASETS.keys()),
        index=list(DATASETS.keys()).index(cfg.dataset_label) if cfg.dataset_label in DATASETS else 0,
        key="dataset_table_explorer_select"
    )

    csv_path = DATASETS[selected_view_label]
    if os.path.exists(csv_path):
        table_df = pd.read_csv(csv_path)

        # Dataset metrics
        n_rows, n_cols = table_df.shape
        n_features = n_cols - 1 if "target" in table_df.columns else n_cols
        disease_count = int(table_df["target"].sum()) if "target" in table_df.columns else 0
        healthy_count = n_rows - disease_count

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("Total Patient Records", f"{n_rows:,}")
        with k2:
            st.metric("Raw Clinical Features", n_features)
        with k3:
            st.metric("Positive Diagnoses (Disease)", f"{disease_count} ({disease_count/n_rows:.1%})")
        with k4:
            st.metric("Negative Diagnoses (Healthy)", f"{healthy_count} ({healthy_count/n_rows:.1%})")

        # Full Interactive Table
        st.dataframe(table_df, use_container_width=True, height=350)

        # Expandable Descriptive Statistics
        with st.expander("📊 View Summary Statistics (Mean, Std, Min, Max, Quartiles)"):
            st.dataframe(table_df.describe().T, use_container_width=True)
    else:
        st.warning(f"File {csv_path} not found. Run `python3 scripts/download_datasets.py` to populate.")
