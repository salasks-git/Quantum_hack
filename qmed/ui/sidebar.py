"""Sidebar widgets -> PipelineConfig."""
from dataclasses import dataclass

import streamlit as st

from qmed.config import (
    BANDWIDTH_OPTIONS, DATASETS, DEFAULT_BANDWIDTH, ENTANGLEMENT_OPTIONS, FEATURE_MAP_OPTIONS, QUBIT_OPTIONS,
)


@dataclass(frozen=True)
class PipelineConfig:
    dataset_label: str
    n_samples: int
    n_qubits: int
    feature_map_name: str
    reps: int
    entanglement: str
    kernel_bandwidth: float
    vqc_iter: int


def render_sidebar() -> PipelineConfig:
    sb = st.sidebar
    sb.header("⚙️ Quantum Pipeline Config")
    return PipelineConfig(
        dataset_label=sb.selectbox("1. Select Clinical Dataset", list(DATASETS.keys())),
        n_samples=sb.slider("2. Cohort Sample Size (Stratified)", min_value=60, max_value=180, value=100, step=20),
        n_qubits=sb.selectbox("3. Number of Qubits (PCA Features)", options=QUBIT_OPTIONS, index=2),
        feature_map_name=sb.selectbox("4. Quantum Feature Map", options=FEATURE_MAP_OPTIONS),
        reps=sb.slider("5. Circuit Depth (Repetitions)", min_value=1, max_value=3, value=2),
        entanglement=sb.selectbox("6. Entanglement Topology", options=ENTANGLEMENT_OPTIONS, index=0),
        kernel_bandwidth=sb.select_slider(
            "7. Quantum Kernel Bandwidth (s)", options=BANDWIDTH_OPTIONS, value=DEFAULT_BANDWIDTH,
            help="QSVC encodes s × angle instead of the raw [0, π] angle. s = 1 is the unscaled "
                 "feature map; smaller s makes the quantum kernel smoother. Classical models and "
                 "VQC are unaffected. See the Kernel Bandwidth Lab tab.",
        ),
        vqc_iter=sb.slider("8. VQC Max Iterations (COBYLA)", min_value=15, max_value=75, value=35, step=5),
    )
