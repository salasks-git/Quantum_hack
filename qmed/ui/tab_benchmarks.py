"""Tab 3: score table and confusion matrices for all four models."""
import streamlit as st

from qmed.metrics import benchmark_table
from qmed.ui import plots
from qmed.ui.sidebar import PipelineConfig


def render(cfg: PipelineConfig, results: dict, y_test) -> None:
    st.subheader("Apples-to-Apples Benchmark: Quantum vs Classical Models")
    st.caption(f"All models evaluated on the exact same {cfg.n_qubits}-dimensional compressed feature representation.")
    st.table(benchmark_table(results, y_test))

    st.markdown("#### Test Cohort Confusion Matrices")
    for col, (name, res) in zip(st.columns(len(results)), results.items()):
        with col:
            plots.show(plots.confusion_heatmap(y_test, res["pred"], name))
