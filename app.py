import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)

# Qiskit 1.x+ Imports (Compatible with qiskit-machine-learning 0.7 - 0.9+)
from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap, RealAmplitudes
from qiskit.primitives import StatevectorSampler
try:
    from qiskit_machine_learning.state_fidelities import ComputeUncompute
    from qiskit_machine_learning.optimizers import COBYLA
except ImportError:
    from qiskit_algorithms.state_fidelities import ComputeUncompute
    from qiskit_algorithms.optimizers import COBYLA

from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC, VQC

st.set_page_config(page_title="Q-Med: Hybrid Quantum Disease Classifier", layout="wide")

st.title("🧬 Q-Med: Multi-Domain Hybrid Quantum Disease Classifier")
st.markdown(
    "**Qiskit Fall Fest 2026 · Quantum Hack Day Kochi | Challenge 02: Healthcare QML**  \n"
    "Benchmarking Quantum Kernel Support Vector Machines (`QSVC`) and Variational Quantum Classifiers (`VQC`) "
    "against classical baselines across Oncology, Neurology, Cardiology, and Endocrinology datasets."
)

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("⚙️ Quantum Pipeline Config")

DATASET_MAP = {
    "Parkinson's Vocal Biomarkers (Neurology)": "data/parkinsons_vocal.csv",
    "Statlog Heart Disease (Cardiology)": "data/heart_disease_statlog.csv",
    "Wisconsin Breast Cancer (Oncology)": "data/breast_cancer_wisconsin.csv",
    "Pima Indians Diabetes (Endocrinology)": "data/diabetes_pima.csv",
}

selected_dataset_label = st.sidebar.selectbox("1. Select Clinical Dataset", list(DATASET_MAP.keys()))
n_samples = st.sidebar.slider("2. Cohort Sample Size (Stratified)", min_value=60, max_value=180, value=100, step=20)
n_qubits = st.sidebar.selectbox("3. Number of Qubits (PCA Features)", options=[2, 3, 4], index=2)
feature_map_type = st.sidebar.selectbox("4. Quantum Feature Map", options=["ZZFeatureMap", "PauliFeatureMap (Z, YY, ZX)"])
reps = st.sidebar.slider("5. Circuit Depth (Repetitions)", min_value=1, max_value=3, value=2)
entanglement = st.sidebar.selectbox("6. Entanglement Topology", options=["linear", "full", "circular"], index=0)
vqc_iter = st.sidebar.slider("7. VQC Max Iterations (COBYLA)", min_value=15, max_value=75, value=35, step=5)

# --- 1. OFFLINE DATA LOADING & PREPROCESSING ---
@st.cache_data
def load_and_preprocess(dataset_label: str, max_samples: int, n_qubits: int):
    csv_path = DATASET_MAP[dataset_label]
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        X = df.drop(columns=["target"]).values
        y = df["target"].values.astype(int)
        feature_names = [c for c in df.columns if c != "target"]
    else:
        # Automatic offline fallback if scripts/download_datasets.py wasn't run yet
        raw = load_breast_cancer()
        df = pd.DataFrame(raw.data, columns=raw.feature_names)
        df["target"] = 1 - raw.target
        X = raw.data
        y = (1 - raw.target).astype(int)
        feature_names = list(raw.feature_names)

    if len(X) > max_samples:
        X_sub, _, y_sub, _ = train_test_split(
            X, y, train_size=max_samples, random_state=42, stratify=y
        )
    else:
        X_sub, y_sub = X, y

    X_train, X_test, y_train, y_test = train_test_split(
        X_sub, y_sub, test_size=0.25, random_state=42, stratify=y_sub
    )

    # 3-Stage Quantum Preprocessing: Standardize -> PCA(n_qubits) -> Scale [0, pi]
    std_scaler = StandardScaler()
    pca = PCA(n_components=n_qubits, random_state=42)
    angle_scaler = MinMaxScaler(feature_range=(0, np.pi))

    X_train_std = std_scaler.fit_transform(X_train)
    X_test_std = std_scaler.transform(X_test)

    X_train_pca = pca.fit_transform(X_train_std)
    X_test_pca = pca.transform(X_test_std)

    X_train_q = angle_scaler.fit_transform(X_train_pca)
    X_test_q = angle_scaler.transform(X_test_pca)

    return df, feature_names, X_train_q, X_test_q, y_train, y_test, pca

df, feature_names, X_train_q, X_test_q, y_train, y_test, pca = load_and_preprocess(
    selected_dataset_label, n_samples, n_qubits
)

# --- 2. QUANTUM CIRCUIT CONSTRUCTION ---
if feature_map_type == "ZZFeatureMap":
    feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=reps, entanglement=entanglement)
else:
    feature_map = PauliFeatureMap(feature_dimension=n_qubits, reps=reps, paulis=["Z", "YY", "ZX"], entanglement=entanglement)

ansatz = RealAmplitudes(num_qubits=n_qubits, reps=reps, entanglement=entanglement)

# --- 3. HYBRID MODEL TRAINING & EVALUATION ---
@st.cache_resource
def train_pipeline(dataset_label, X_train_q, X_test_q, y_train, y_test, _feature_map, _ansatz, fm_name, reps, ent, vqc_iter):
    results = {}

    # 1. Classical Linear SVM
    t0 = time.time()
    svm_lin = SVC(kernel="linear", probability=True, random_state=42)
    svm_lin.fit(X_train_q, y_train)
    y_pred_lin = svm_lin.predict(X_test_q)
    y_prob_lin = svm_lin.predict_proba(X_test_q)[:, 1]
    results["Classical SVM (Linear)"] = {
        "model": svm_lin, "pred": y_pred_lin, "prob": y_prob_lin, "time": time.time() - t0
    }

    # 2. Classical RBF SVM
    t0 = time.time()
    svm_rbf = SVC(kernel="rbf", probability=True, random_state=42)
    svm_rbf.fit(X_train_q, y_train)
    y_pred_rbf = svm_rbf.predict(X_test_q)
    y_prob_rbf = svm_rbf.predict_proba(X_test_q)[:, 1]
    results["Classical SVM (RBF)"] = {
        "model": svm_rbf, "pred": y_pred_rbf, "prob": y_prob_rbf, "time": time.time() - t0
    }

    # 3. Quantum Kernel SVM (QSVC)
    t0 = time.time()
    sampler = StatevectorSampler()
    fidelity = ComputeUncompute(sampler=sampler)
    q_kernel = FidelityQuantumKernel(feature_map=_feature_map, fidelity=fidelity)
    qsvc = QSVC(quantum_kernel=q_kernel, probability=True, random_state=42)
    qsvc.fit(X_train_q, y_train)
    y_pred_qsvc = qsvc.predict(X_test_q)
    y_prob_qsvc = qsvc.predict_proba(X_test_q)[:, 1]
    kernel_matrix = q_kernel.evaluate(x_vec=X_train_q)
    results["Quantum Kernel SVM (QSVC)"] = {
        "model": qsvc, "pred": y_pred_qsvc, "prob": y_prob_qsvc, "time": time.time() - t0,
        "kernel_matrix": kernel_matrix
    }

    # 4. Variational Quantum Classifier (VQC)
    t0 = time.time()
    loss_history = []
    def callback(weights, obj_func_eval):
        loss_history.append(obj_func_eval)

    vqc = VQC(
        sampler=sampler,
        feature_map=_feature_map,
        ansatz=_ansatz,
        optimizer=COBYLA(maxiter=vqc_iter),
        callback=callback
    )
    vqc.fit(X_train_q, y_train)
    y_pred_vqc = np.asarray(vqc.predict(X_test_q)).ravel().astype(int)
    results["Variational Quantum (VQC)"] = {
        "model": vqc, "pred": y_pred_vqc, "prob": y_pred_vqc, "time": time.time() - t0,
        "loss_history": loss_history
    }

    return results

with st.spinner(f"Simulating {n_qubits}-Qubit Hilbert Space on {selected_dataset_label}..."):
    results = train_pipeline(
        selected_dataset_label, X_train_q, X_test_q, y_train, y_test,
        feature_map, ansatz, feature_map_type, reps, entanglement, vqc_iter
    )

# --- TABS FOR DEMO ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 1. Dataset & Quantum Preprocessing",
    "⚛️ 2. Quantum Circuits & Kernel Matrix",
    "🏆 3. Classical vs Quantum Benchmarks",
    "🩺 4. Live Patient Diagnostic Simulator"
])

with tab1:
    st.subheader(f"Dataset: {selected_dataset_label}")
    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown(f"""
        - **Original Clinical Features:** `{len(feature_names)}` continuous measurements
        - **Active Cohort Size:** `{len(X_train_q) + len(X_test_q)}` patients (`{len(X_train_q)}` train / `{len(X_test_q)}` test)
        - **Variance Captured by {n_qubits} PCA Components:** `{np.sum(pca.explained_variance_ratio_):.1%}`
        - **Quantum Angle Mapping:** Standardized $\\to$ PCA({n_qubits}) $\\to$ Scaled to $[0, \\pi]$ radians to prevent $2\\pi$ Bloch sphere phase aliasing.
        """)
        st.dataframe(df.head(8), use_container_width=True)
    with col2:
        fig, ax = plt.subplots(figsize=(5, 3.8))
        scatter = ax.scatter(X_train_q[:, 0], X_train_q[:, 1], c=y_train, cmap="coolwarm", edgecolors="k", s=65)
        ax.set_xlabel("Qubit 0 Angle (PC1 in [0, π] rad)")
        ax.set_ylabel("Qubit 1 Angle (PC2 in [0, π] rad)")
        ax.set_title("Patients Mapped to Quantum Rotation Angles")
        plt.colorbar(scatter, ax=ax, label="0: Healthy | 1: Disease")
        st.pyplot(fig)

with tab2:
    st.subheader("Quantum Circuit Architecture & Hilbert Space Geometry")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Feature Map (`{feature_map_type}`)** — Encodes $\\vec{{x}} \\in [0, \\pi]^{{{n_qubits}}}$ via non-linear $ZZ$ phase entanglement:")
        fig_fm = feature_map.decompose().draw(output="mpl", style="iqp")
        st.pyplot(fig_fm)
    with c2:
        st.markdown(f"**Variational Ansatz (`RealAmplitudes`)** — Trainable $R_y(\\theta)$ + CNOT entanglement (`{entanglement}`):")
        fig_ans = ansatz.decompose().draw(output="mpl", style="iqp")
        st.pyplot(fig_ans)

    st.divider()
    k1, k2 = st.columns(2)
    with k1:
        st.markdown("**Quantum Kernel Matrix $K(\\vec{x}_i, \\vec{x}_j) = |\\langle \\psi(\\vec{x}_i) | \\psi(\\vec{x}_j) \\rangle|^2$**")
        fig_k, ax_k = plt.subplots(figsize=(5, 4))
        sns.heatmap(results["Quantum Kernel SVM (QSVC)"]["kernel_matrix"], cmap="viridis", ax=ax_k)
        ax_k.set_title("Pairwise State Fidelity in 2ⁿ Hilbert Space")
        ax_k.set_xlabel("Patient Index i")
        ax_k.set_ylabel("Patient Index j")
        st.pyplot(fig_k)
    with k2:
        st.markdown("**VQC Training Convergence (COBYLA Loss Curve)**")
        loss_hist = results["Variational Quantum (VQC)"]["loss_history"]
        fig_l, ax_l = plt.subplots(figsize=(5, 4))
        ax_l.plot(range(1, len(loss_hist) + 1), loss_hist, marker="o", color="#7c3aed", linewidth=2)
        ax_l.set_title("Cross-Entropy Objective vs COBYLA Iterations")
        ax_l.set_xlabel("Iteration")
        ax_l.set_ylabel("Objective Loss")
        ax_l.grid(True, alpha=0.3)
        st.pyplot(fig_l)

with tab3:
    st.subheader("Apples-to-Apples Benchmark: Quantum vs Classical Models")
    st.caption(f"All models evaluated on the exact same {n_qubits}-dimensional compressed feature representation.")
    rows = []
    for name, res in results.items():
        y_p = res["pred"]
        rows.append({
            "Model Architecture": name,
            "Test Accuracy": f"{accuracy_score(y_test, y_p):.2%}",
            "F1-Score (Disease)": f"{f1_score(y_test, y_p, zero_division=0):.3f}",
            "Precision": f"{precision_score(y_test, y_p, zero_division=0):.3f}",
            "Recall (Sensitivity)": f"{recall_score(y_test, y_p, zero_division=0):.3f}",
            "ROC-AUC": f"{roc_auc_score(y_test, res['prob']):.3f}",
            "Compute Time": f"{res['time']:.2f}s"
        })
    st.table(pd.DataFrame(rows))

    st.markdown("#### Test Cohort Confusion Matrices")
    cm_cols = st.columns(4)
    for idx, (name, res) in enumerate(results.items()):
        with cm_cols[idx]:
            cm = confusion_matrix(y_test, res["pred"])
            fig_cm, ax_cm = plt.subplots(figsize=(3.2, 2.8))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax_cm,
                        xticklabels=["Healthy", "Disease"], yticklabels=["Healthy", "Disease"])
            ax_cm.set_title(name, fontsize=9)
            ax_cm.set_ylabel("Actual")
            ax_cm.set_xlabel("Predicted")
            st.pyplot(fig_cm)

with tab4:
    st.subheader("🩺 Real-Time Quantum Patient Diagnostic Inference")
    st.markdown("Adjust the qubit rotation angles $\\vec{x} \\in [0, \\pi]$ to compare live inference between **QSVC** and **Classical RBF SVM**:")
    sliders = []
    scols = st.columns(n_qubits)
    for i in range(n_qubits):
        with scols[i]:
            val = st.slider(f"Qubit {i} Angle (PC{i+1})", 0.0, float(np.pi), float(np.pi / 2), 0.05)
            sliders.append(val)

    patient_vec = np.array([sliders])
    qsvc_model = results["Quantum Kernel SVM (QSVC)"]["model"]
    rbf_model = results["Classical SVM (RBF)"]["model"]

    q_pred = qsvc_model.predict(patient_vec)[0]
    q_prob = qsvc_model.predict_proba(patient_vec)[0][1]
    c_pred = rbf_model.predict(patient_vec)[0]
    c_prob = rbf_model.predict_proba(patient_vec)[0][1]

    rcol1, rcol2 = st.columns(2)
    with rcol1:
        st.metric(
            label="Quantum Kernel (QSVC) Diagnosis",
            value="🚨 DISEASE DETECTED" if q_pred == 1 else "✅ HEALTHY",
            delta=f"Disease Probability: {q_prob:.1%}"
        )
    with rcol2:
        st.metric(
            label="Classical RBF SVM Diagnosis",
            value="🚨 DISEASE DETECTED" if c_pred == 1 else "✅ HEALTHY",
            delta=f"Disease Probability: {c_prob:.1%}"
        )
