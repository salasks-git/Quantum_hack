"""Tab 3: score table, step-by-step execution times, efficiency frontier, and confusion matrices."""
import pandas as pd
import streamlit as st
from sklearn.metrics import fbeta_score

from qmed.config import RBF_SVM
from qmed.metrics import benchmark_table, fmt_duration
from qmed.ui import plots
from qmed.ui.sidebar import PipelineConfig


def render(cfg: PipelineConfig, data_or_results, results_or_y_test=None) -> None:
    # Support both render(cfg, data, results) and render(cfg, results, y_test)
    if hasattr(data_or_results, "y_test"):
        data = data_or_results
        results = results_or_y_test
        y_test = data.y_test
    else:
        data = None
        results = data_or_results
        y_test = results_or_y_test

    st.subheader("🏆 Apples-to-Apples Benchmark: Quantum vs Classical Models")
    st.caption(f"All models evaluated on the exact same {cfg.n_qubits}-dimensional compressed feature representation.")

    # Calculate the clinical leader based on F2-score
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

    # 1. Classification Performance Metrics Table
    st.dataframe(benchmark_table(results, y_test), use_container_width=True)

    st.divider()

    # 2. Step-by-Step and Overall Execution Time Breakdown
    st.subheader("⏱️ Execution Time by Technique: Step-by-Step & Overall")
    st.caption("Detailed timing breakdown across Step 1 (Model Training / Quantum Kernel Gram Matrix) and Step 2 (Test Set Inference).")

    # Metrics row for quick comparison
    n_test = len(y_test)
    time_cols = st.columns(len(results) + (1 if data is not None else 0))
    idx = 0
    if data is not None:
        with time_cols[0]:
            st.metric("Step 0: Preprocessing", f"{data.prep_time * 1000:.1f} ms", help="StandardScaler + PCA + Angle Scaling")
        idx = 1

    for name, res in results.items():
        with time_cols[idx]:
            short_name = name.split("(")[-1].replace(")", "").strip() if "(" in name else name
            st.metric(f"{short_name} Total", fmt_duration(res["total_time"]), delta=f"Train: {fmt_duration(res['train_time'])}")
        idx += 1

    # Visual Timing Breakdown
    t_col1, t_col2 = st.columns(2)
    with t_col1:
        st.markdown("#### Step-by-Step Breakdown (Train vs. Test Inference)")
        plots.show(plots.train_vs_inference_bars(results))

    with t_col2:
        st.markdown("#### Clinical Efficiency Frontier ($F_2$ vs. Latency)")
        plots.show(plots.pareto_frontier_scatter(results, y_test))

    # Step-by-Step Timing Table
    step_rows = []
    baseline_tot = results[RBF_SVM]["total_time"] if RBF_SVM in results else 1.0
    for name, res in results.items():
        t_tr = res["train_time"]
        t_inf = res["inf_time"]
        t_tot = res["total_time"]
        per_patient_ms = (t_inf / n_test) * 1000 if n_test > 0 else 0
        ratio = t_tot / baseline_tot if baseline_tot > 0 else 1.0

        step_rows.append({
            "Technique": name,
            "Step 1: Training / Kernel Evaluation": fmt_duration(t_tr),
            "Step 2: Test Cohort Inference": fmt_duration(t_inf),
            "Per-Patient Query Latency": f"{per_patient_ms:.2f} ms",
            "Overall Execution Time": fmt_duration(t_tot),
            "Throughput": f"{n_test / t_inf:.1f} patients/s" if t_inf > 0 else "N/A",
            "Time vs Classical RBF": f"{ratio:.1f}x slower" if ratio >= 1.0 else f"{1/ratio:.1f}x faster",
        })

    st.dataframe(pd.DataFrame(step_rows), use_container_width=True)

    st.divider()

    # 3. Test Cohort Confusion Matrices
    st.markdown("#### 🩺 Test Cohort Confusion Matrices")
    for col, (name, res) in zip(st.columns(len(results)), results.items()):
        with col:
            plots.show(plots.confusion_heatmap(y_test, res["pred"], name))
