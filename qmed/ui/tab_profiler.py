"""Tab 4: execution time and latency profiler."""
import streamlit as st

from qmed.config import LINEAR_SVM, QSVC_MODEL, RBF_SVM, VQC_MODEL
from qmed.data import PreparedData
from qmed.metrics import timing_table
from qmed.ui import plots
from qmed.ui.sidebar import PipelineConfig

LATENCY_EXPLAINER = r"""
- **Why are Classical SVMs so fast ($<5\text{ ms}$)?** Classical SVMs use optimized C++ libraries (`libsvm`) with cache-friendly matrix operations in Euclidean space.
- **Where is the Quantum Time Spent?**
  - **QSVC:** Must evaluate $\frac{N(N-1)}{2}$ pairwise state fidelities $|\langle \psi(\vec{x}_i) | \psi(\vec{x}_j) \rangle|^2$. On a classical CPU simulator, each fidelity requires multiplying $2^n \times 2^n$ state vectors.
  - **VQC:** Evaluates quantum circuits at every iteration of the `COBYLA` optimizer. For 35 iterations across the training cohort, this requires hundreds of circuit simulations.
- **Real Quantum Hardware (QPU) Perspective:** On physical superconducting QPUs (like IBM Quantum Heron or Eagle), gate execution takes $\approx 20\text{--}100\text{ ns}$. The quantum state evolution occurs naturally in hardware in $O(1)$ depth rather than classical $2^n$ statevector matrix manipulation.
"""


def render(cfg: PipelineConfig, data: PreparedData, results: dict) -> None:
    n_train, n_test = len(data.X_train_q), len(data.X_test_q)
    st.subheader("⏱️ Execution Time & Computational Latency Profiler")
    st.markdown(
        f"Granular timing breakdown of each technique across **{n_train}** training samples "
        f"and **{n_test}** unseen test samples in a **{cfg.n_qubits}**-qubit ($2^{{{cfg.n_qubits}}} = {2**cfg.n_qubits}$-dimensional Hilbert space)."
    )

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Quantum Preprocessing", f"{data.prep_time*1000:.1f} ms", help="StandardScaler + PCA + Angle Scaling")
    m2.metric("Linear SVM Total", f"{results[LINEAR_SVM]['total_time']*1000:.1f} ms")
    m3.metric("RBF SVM Total", f"{results[RBF_SVM]['total_time']*1000:.1f} ms")
    m4.metric("QSVC (Quantum Kernel)", f"{results[QSVC_MODEL]['total_time']:.2f} s")
    m5.metric("VQC (Variational)", f"{results[VQC_MODEL]['total_time']:.2f} s")

    st.divider()
    col_plot1, col_plot2 = st.columns(2)
    with col_plot1:
        st.markdown("#### Total End-to-End Execution Time by Technique")
        plots.show(plots.total_time_bars(results))
    with col_plot2:
        st.markdown("#### Training vs. Test Inference Time Breakdown")
        plots.show(plots.train_vs_inference_bars(results))

    st.divider()
    st.markdown("#### Granular Computational Profiling & Scaling Metrics")
    st.dataframe(timing_table(results, n_test), use_container_width=True)

    with st.expander("💡 Understanding Quantum Simulation Latency vs. Physical QPU Execution"):
        st.markdown(LATENCY_EXPLAINER)
