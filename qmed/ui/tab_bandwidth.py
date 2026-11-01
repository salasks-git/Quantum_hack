"""Tab 7: Kernel Bandwidth Lab - why QSVC needs a scaled-down feature map."""
import pandas as pd
import streamlit as st

from qmed.bandwidth import BandwidthSweep, bandwidth_sweep, kernel_diagnostics, kernel_matrices
from qmed.circuits import build_feature_map
from qmed.config import BANDWIDTH_OPTIONS, DATASETS
from qmed.data import PreparedData
from qmed.ui import plots
from qmed.ui.sidebar import PipelineConfig

S_MAX, S_MIN, S_MID = max(BANDWIDTH_OPTIONS), min(BANDWIDTH_OPTIONS), 0.3

EXPLAINER = r"""
**What is the bandwidth?** Each patient is turned into rotation angles $\vec{x} \in [0, \pi]^n$ and loaded
into a quantum state $|\psi(\vec{x})\rangle$. The quantum kernel scores two patients by the overlap of their
states, $K(\vec{x}, \vec{x}') = |\langle\psi(s\vec{x})|\psi(s\vec{x}')\rangle|^2$. The bandwidth $s$ shrinks the
angles before encoding, the same role $\gamma$ plays in a classical RBF kernel.

- **$s$ too large (e.g. 1.0): the kernel memorizes.** The states spread over the whole $2^n$-dimensional
  Hilbert space, so any two different patients look almost unrelated ($K \approx 0$). The Gram matrix is close
  to the identity: the SVM can fit the training set but has nothing to generalize from.
- **$s$ too small (e.g. 0.05): the kernel blurs.** All states sit next to each other, $K \approx 1$ for every
  pair, and the differences between patients get very small.
- **In between, the kernel is informative.** Patients with the same diagnosis are more similar to each other
  than to patients with the other diagnosis.

Scaling quantum feature maps this way follows Shaydulin & Wild, *Importance of Kernel Bandwidth in Quantum
Machine Learning*, Phys. Rev. A 106, 042407 (2022). It costs nothing on hardware: it only changes rotation angles.
"""


@st.cache_data(show_spinner=False)
def _cached_sweep(dataset_label: str, n_samples: int, n_qubits: int, feature_map_name: str,
                  reps: int, entanglement: str) -> BandwidthSweep:
    return bandwidth_sweep(DATASETS[dataset_label], n_samples, n_qubits, feature_map_name, reps, entanglement)


@st.cache_data(show_spinner=False)
def _cached_kernels(X_train_q, feature_map_name: str, n_qubits: int, reps: int, entanglement: str,
                    bandwidths: tuple) -> dict:
    feature_map = build_feature_map(feature_map_name, n_qubits, reps, entanglement)
    return kernel_matrices(X_train_q, feature_map, bandwidths)


def _regime(kernel_mean: float) -> str:
    if kernel_mean < 0.15:
        return "Too sharp: near-identity, every patient looks unique, so the SVM memorizes."
    if kernel_mean > 0.85:
        return "Too blurred: near all-ones, every patient looks the same."
    return "Balanced: similar patients score high, different patients score low."


def _panel_bandwidths(current_s: float) -> list:
    """Largest, current, and smallest s; if current is an extreme, show a middle value instead."""
    middle = current_s if current_s not in (S_MAX, S_MIN) else S_MID
    return sorted({S_MAX, middle, S_MIN}, reverse=True)


def _results_table(sweep: BandwidthSweep, current_s: float) -> pd.DataFrame:
    t = sweep.table
    rows = [{
        "Bandwidth s": f"{r.bandwidth:g}" + ("  ◀ your setting" if r.bandwidth == current_s else ""),
        "CV Accuracy": f"{r.accuracy:.3f} ± {r.accuracy_std:.3f}",
        "CV ROC-AUC": f"{r.roc_auc:.3f} ± {r.roc_auc_std:.3f}",
        "CV Recall": f"{r.recall:.3f}",
        "Kernel Mean (off-diag)": f"{r.kernel_mean:.3f}",
        "Kernel Variance (off-diag)": f"{r.kernel_var:.4f}",
        "Kernel-Target Alignment": f"{r.alignment:.3f}",
    } for r in t.itertuples()]
    rbf = sweep.rbf
    rows.append({
        "Bandwidth s": "Classical RBF SVM (baseline)",
        "CV Accuracy": f"{rbf['accuracy']:.3f} ± {rbf['accuracy_std']:.3f}",
        "CV ROC-AUC": f"{rbf['roc_auc']:.3f} ± {rbf['roc_auc_std']:.3f}",
        "CV Recall": f"{rbf['recall']:.3f}",
        "Kernel Mean (off-diag)": "—", "Kernel Variance (off-diag)": "—", "Kernel-Target Alignment": "—",
    })
    return pd.DataFrame(rows)


def render(cfg: PipelineConfig, data: PreparedData) -> None:
    current_s = cfg.kernel_bandwidth
    st.subheader("🎚️ Kernel Bandwidth Lab: Tuning the Quantum Kernel")
    st.markdown(
        "The quantum kernel has one setting that matters more than circuit depth or entanglement: "
        "how far apart it pushes patients in Hilbert space. Here that setting is the **bandwidth $s$** "
        f"(sidebar option 7, currently **s = {current_s:g}**)."
    )
    with st.expander("💡 How the bandwidth works (plain-language explanation)", expanded=True):
        st.markdown(EXPLAINER)

    # --- 1. Small multiples: what the Gram matrix looks like at three bandwidths ---
    st.markdown("#### What the Quantum Kernel Sees at Three Bandwidths")
    st.caption(
        f"Gram matrices of the {len(data.y_train)} training patients, sorted by diagnosis "
        "(healthy, then disease). Bright = similar. A useful kernel shows two bright blocks on the diagonal."
    )
    panels = _panel_bandwidths(current_s)
    with st.spinner("Computing quantum kernel matrices..."):
        kernels = _cached_kernels(data.X_train_q, cfg.feature_map_name, cfg.n_qubits, cfg.reps,
                                  cfg.entanglement, tuple(panels))
    titles = {s: f"s = {s:g}" + (" (your setting)" if s == current_s else "") for s in panels}
    plots.show(plots.kernel_small_multiples(kernels, data.y_train, titles))
    for col, s in zip(st.columns(len(panels)), panels):
        diag = kernel_diagnostics(kernels[s], data.y_train)
        col.caption(f"**s = {s:g}:** {_regime(diag['kernel_mean'])} "
                    f"(mean off-diagonal K = {diag['kernel_mean']:.2f})")

    st.divider()

    # --- 2. Cross-validated bandwidth sweep ---
    with st.spinner(f"Running {len(BANDWIDTH_OPTIONS)}-bandwidth × 5-fold cross-validation sweep..."):
        sweep = _cached_sweep(cfg.dataset_label, cfg.n_samples, cfg.n_qubits, cfg.feature_map_name,
                              cfg.reps, cfg.entanglement)
    table = sweep.table.set_index("bandwidth")
    best_s = float(table["accuracy"].idxmax())

    st.markdown("#### Does the Bandwidth Change Diagnostic Accuracy?")
    st.caption(
        f"Stratified {sweep.n_folds}-fold cross-validation on all {sweep.n_patients} patients of the active cohort. "
        "Scaling and PCA are refit inside every fold, so held-out patients never leak into preprocessing. "
        "ROC-AUC is computed from the SVM decision margin. Exact (noise-free) statevector kernel."
    )
    m1, m2, m3, m4 = st.columns(4)
    if current_s in table.index:
        m1.metric(f"QSVC CV Accuracy (s = {current_s:g})", f"{table.loc[current_s, 'accuracy']:.1%}",
                  delta=f"{table.loc[current_s, 'accuracy'] - table.loc[S_MAX, 'accuracy']:+.1%} vs s = {S_MAX:g}")
    m2.metric(f"QSVC CV Accuracy (s = {S_MAX:g}, unscaled)", f"{table.loc[S_MAX, 'accuracy']:.1%}")
    m3.metric("Best s by CV Accuracy", f"{best_s:g}", delta=f"{table.loc[best_s, 'accuracy']:.1%}",
              delta_color="off")
    m4.metric("Classical RBF SVM (CV)", f"{sweep.rbf['accuracy']:.1%}",
              delta=f"AUC {sweep.rbf['roc_auc']:.3f}", delta_color="off")

    c1, c2 = st.columns([1, 1.35])
    with c1:
        plots.show(plots.bandwidth_scores(sweep.table, sweep.rbf, current_s))
    with c2:
        plots.show(plots.bandwidth_diagnostics(sweep.table, current_s))
    st.caption(
        "**Kernel concentration:** mean kernel value between two different patients (band = spread). "
        "Near 0 = near-identity matrix, near 1 with no spread = all-ones matrix. "
        "**Kernel-target alignment:** how closely the kernel's similarity pattern matches the true diagnoses "
        "(centered, Cortes et al. 2012)."
    )

    st.markdown("#### Full Sweep Results")
    st.dataframe(_results_table(sweep, current_s), use_container_width=True, hide_index=True)
    st.caption(
        "Caveat: picking the best s from this same table is itself a tuning step, so the best row is slightly "
        "optimistic. A fully unbiased estimate would need nested cross-validation."
    )
