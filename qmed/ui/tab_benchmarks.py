"""Tab 3: score table and confusion matrices for all four models."""
import streamlit as st

from qmed.metrics import benchmark_table
from qmed.ui import plots
from qmed.ui.sidebar import PipelineConfig


def render(cfg: PipelineConfig, results: dict, y_test) -> None:
    st.subheader("Apples-to-Apples Benchmark: Quantum vs Classical Models")
    st.caption(f"All models evaluated on the exact same {cfg.n_qubits}-dimensional compressed feature representation.")

    # Calculate the clinical leader based on F2-score
    from sklearn.metrics import fbeta_score
    best_name = None
    best_f2 = -1.0
    for name, res in results.items():
        score = fbeta_score(y_test, res["pred"], beta=2, zero_division=0)
        if score > best_f2:
            best_f2 = score
            best_name = name

    st.success(
        f"🎯 **Clinical Optimization Target: $F_2$-Score (Recall-Weighted)** — "
        f"Top Model: **{best_name}** with $F_2 = {best_f2:.3f}$. "
        f"(Weighs Recall 2× higher than Precision to minimize dangerous False Negatives in disease screening)."
    )

    st.dataframe(benchmark_table(results, y_test), use_container_width=True)

    st.markdown("#### Test Cohort Confusion Matrices")
    for col, (name, res) in zip(st.columns(len(results)), results.items()):
        with col:
            plots.show(plots.confusion_heatmap(y_test, res["pred"], name))
