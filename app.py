"""Q-Med Streamlit entry point: `streamlit run app.py`.

Wires sidebar config -> cached data prep -> cached model training -> tabs.
Pipeline logic lives in qmed/ (no Streamlit); each tab lives in qmed/ui/.
"""
import streamlit as st

from qmed.circuits import build_ansatz, build_feature_map
from qmed.config import DATASETS
from qmed.data import PreparedData, prepare_data
from qmed.models import train_all_models
from qmed.ui import (
    tab_bandwidth, tab_benchmarks, tab_circuits, tab_dataset, tab_ibm_hardware, tab_profiler, tab_quhama, tab_simulator,
)
from qmed.ui.sidebar import render_sidebar

st.set_page_config(page_title="Q-Med: Hybrid Quantum Disease Classifier", layout="wide")

st.title("🧬 Q-Med: Multi-Domain Hybrid Quantum Disease Classifier")
st.markdown(
    "**Qiskit Fall Fest 2026 · Quantum Hack Day Kochi | Challenge 02: Healthcare QML**  \n"
    "Benchmarking Quantum Kernel Support Vector Machines (`QSVC`) and Variational Quantum Classifiers (`VQC`) "
    "against classical baselines across Oncology, Neurology, Cardiology, Endocrinology, Orthopedics, Nephrology, and Obstetrics cohorts."
)


# Streamlit reruns this script on every interaction; caching keeps reruns instant.
# Cache keys are the function arguments, so all of them are plain hashable values.
@st.cache_data
def cached_prepare(dataset_label: str, n_samples: int, n_qubits: int) -> PreparedData:
    return prepare_data(DATASETS[dataset_label], n_samples, n_qubits)


@st.cache_resource
def cached_train(X_train_q, y_train, X_test_q, feature_map_name, n_qubits, reps, entanglement, vqc_iter,
                 kernel_bandwidth):
    # Circuits are rebuilt here from plain settings (rather than passed in) so every
    # circuit setting is part of the cache key and stale models can't be reused.
    return train_all_models(
        X_train_q, y_train, X_test_q,
        feature_map=build_feature_map(feature_map_name, n_qubits, reps, entanglement),
        ansatz=build_ansatz(n_qubits, reps, entanglement),
        vqc_iter=vqc_iter,
        kernel_bandwidth=kernel_bandwidth,
    )


cfg = render_sidebar()
data = cached_prepare(cfg.dataset_label, cfg.n_samples, cfg.n_qubits)
feature_map = build_feature_map(cfg.feature_map_name, cfg.n_qubits, cfg.reps, cfg.entanglement)
ansatz = build_ansatz(cfg.n_qubits, cfg.reps, cfg.entanglement)

with st.spinner(f"Simulating {cfg.n_qubits}-Qubit Hilbert Space on {cfg.dataset_label}..."):
    results = cached_train(
        data.X_train_q, data.y_train, data.X_test_q,
        cfg.feature_map_name, cfg.n_qubits, cfg.reps, cfg.entanglement, cfg.vqc_iter,
        cfg.kernel_bandwidth,
    )

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📊 1. Dataset & Quantum Preprocessing",
    "⚛️ 2. Quantum Circuits & Kernel Matrix",
    "🏆 3. Classical vs Quantum Benchmarks",
    "⏱️ 4. Execution Time & Quantum Latency Profiler",
    "🔬 5. IBM Quantum Hardware Latency (Real QPU)",
    "🩺 6. Live Patient Diagnostic Simulator",
    "🎚️ 7. Kernel Bandwidth Lab",
    "🧠 8. QuHaMA & Elite Hybrid Architectures",
])
with tab1:
    tab_dataset.render(cfg, data)
with tab2:
    tab_circuits.render(cfg, feature_map, ansatz, results)
with tab3:
    tab_benchmarks.render(cfg, data, results)
with tab4:
    tab_profiler.render(cfg, data, results)
with tab5:
    tab_ibm_hardware.render(cfg, data, feature_map, ansatz, results)
with tab6:
    tab_simulator.render(cfg, data, results)
with tab7:
    tab_bandwidth.render(cfg, data)
with tab8:
    tab_quhama.render(cfg, data, results)
