"""Tab 5: live QSVC vs RBF SVM diagnosis from qubit-angle sliders."""
import numpy as np
import streamlit as st

from qmed.config import QSVC_MODEL, RBF_SVM
from qmed.data import PreparedData
from qmed.ui.sidebar import PipelineConfig


def _diagnosis_metric(label: str, model, patient_vec) -> None:
    pred = model.predict(patient_vec)[0]
    prob = model.predict_proba(patient_vec)[0][1]
    st.metric(
        label=label,
        value="🚨 DISEASE DETECTED" if pred == 1 else "✅ HEALTHY",
        delta=f"Disease Probability: {prob:.1%}",
    )


def render(cfg: PipelineConfig, data: PreparedData, results: dict) -> None:
    st.subheader("🩺 Real-Time Quantum Patient Diagnostic Inference")
    st.markdown("Adjust the qubit rotation angles $\\vec{x} \\in [0, \\pi]$ to compare live inference between **QSVC** and **Classical RBF SVM**:")
    st.caption(
        f"Each slider is a compressed PCA summary score (a blend of all {len(data.feature_names)} clinical "
        "measurements) mapped to a rotation angle — not a raw value like blood pressure or glucose. "
        "Load a real held-out test patient below to start from genuine data."
    )

    patient_options = ["Custom (manual sliders)"] + [
        f"Test patient #{i} — actual: {'Disease' if label == 1 else 'Healthy'}"
        for i, label in enumerate(data.y_test)
    ]
    patient_choice = st.selectbox("Load patient", range(len(patient_options)), format_func=lambda k: patient_options[k])

    if patient_choice == 0:
        default_angles = np.full(cfg.n_qubits, np.pi / 2)
    else:
        # Test angles come from a scaler fit on training data, so they can fall slightly outside [0, pi]
        default_angles = np.clip(data.X_test_q[patient_choice - 1], 0.0, np.pi)

    angles = []
    for i, col in enumerate(st.columns(cfg.n_qubits)):
        with col:
            # Key includes patient choice so sliders reset when a different patient is loaded
            angles.append(st.slider(
                f"Qubit {i} Angle (PC{i+1})", 0.0, float(np.pi), float(default_angles[i]), 0.05,
                key=f"angle_{i}_{patient_choice}_{cfg.n_qubits}",
            ))

    patient_vec = np.array([angles])
    rcol1, rcol2 = st.columns(2)
    with rcol1:
        _diagnosis_metric("Quantum Kernel (QSVC) Diagnosis", results[QSVC_MODEL]["model"], patient_vec)
    with rcol2:
        _diagnosis_metric("Classical RBF SVM Diagnosis", results[RBF_SVM]["model"], patient_vec)
