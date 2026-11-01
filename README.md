# 🧬 Q-Med: Multi-Domain Hybrid Quantum Disease Classifier

[![Qiskit](https://img.shields.io/badge/Qiskit-1.x-6929C4.svg?style=flat&logo=qiskit)](https://qiskit.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Event:** Qiskit Fall Fest 2026 · Quantum Hack Day Kochi (Sunday, Nov 1, 2026)  
> **Track:** Challenge 02 — Quantum Machine Learning (QML) for Healthcare Disease Classification

---

## 📌 Executive Summary

**Q-Med** is a reproducible, multi-domain hybrid quantum-classical diagnostic platform built on **Qiskit 1.x** and **Streamlit**. It delivers an empirical, head-to-head benchmark comparing state-of-the-art Quantum Machine Learning algorithms against standard classical models across four clinical disease domains:

- **Oncology:** Wisconsin Breast Cancer Diagnosis (Malignant vs. Benign)
- **Neurology:** Parkinson's Disease Vocal Biomarker Detection
- **Cardiology:** Statlog Heart Disease Prediction
- **Endocrinology:** Pima Indians Diabetes Screening

The platform implements a complete 3-stage quantum preprocessing pipeline, parameterizable quantum circuits (Feature Maps & Variational Ansätze), fidelity-based quantum kernels, and a real-time clinical inference simulator.

---

## 🔬 Mathematical & Architectural Framework

### 1. Quantum State Encoding & Hilbert Space Mapping
To represent high-dimensional continuous clinical data on Near-Term Intermediate-Scale Quantum (NISQ) devices without phase ambiguity, Q-Med applies a 3-stage transformation:
1. **Z-Score Standardization:** Feature scaling to $\mu=0, \sigma=1$.
2. **Principal Component Analysis (PCA):** Dimensionality reduction to $N \in \{2, 3, 4\}$ orthogonal principal components capturing maximal cohort variance.
3. **Angle Scaling $[0, \pi]$:** Features are mapped to rotation angles $\vec{x} \in [0, \pi]^N$ on the Bloch sphere, purposefully avoiding $2\pi$ periodic phase aliasing.

$$\vec{x} \mapsto |\psi(\vec{x})\rangle = U_{\Phi}(\vec{x}) |0\rangle^{\otimes n}$$

### 2. Quantum Feature Maps
- **`ZZFeatureMap`:** Non-linear second-order Pauli expansion encoding pairwise feature interactions via $R_z$ rotations and entangling $R_{zz}(\theta)$ gates:
  $$U_{\Phi}(\vec{x}) = \exp\left(i \sum_j x_j Z_j + \sum_{j < k} (\pi - x_j)(\pi - x_k) Z_j Z_k \right)$$
- **`PauliFeatureMap`:** Higher-order multi-axis entanglement utilizing $\{Z, YY, ZX\}$ operator combinations with linear, full, or circular entanglement topologies.

### 3. Evaluated Model Architectures

| Model Type | Algorithm | Optimization / Method | Hilbert / Feature Space |
| :--- | :--- | :--- | :--- |
| **Quantum Kernel** | **QSVC** (`FidelityQuantumKernel`) | Dual convex quadratic program with Statevector Fidelity | $2^n$-dimensional complex projective Hilbert space |
| **Variational Quantum** | **VQC** | `RealAmplitudes` ansatz ($R_y$ + CNOT) optimized via `COBYLA` | Parameterized Quantum Circuit (PQC) |
| **Classical Baseline** | **Linear SVM** | $L_2$-regularized linear separating hyperplane | Compressed $\mathbb{R}^n$ PCA feature space |
| **Classical Baseline** | **RBF Kernel SVM** | Radial Basis Function $\exp(-\gamma \|\vec{x} - \vec{x}'\|^2)$ | Infinite-dimensional Reproducing Kernel Hilbert Space (RKHS) |

#### Quantum Kernel Evaluation:
$$K(\vec{x}_i, \vec{x}_j) = |\langle \psi(\vec{x}_i) | \psi(\vec{x}_j) \rangle|^2$$
Computed via the `ComputeUncompute` fidelity primitive with `StatevectorSampler`.

---

## 📂 Clinical Datasets (Standardized & Offline-Ready)

All datasets are automatically cleaned, labeled into standardized binary targets (`0: Healthy / Benign`, `1: Disease / Malignant`), and cached locally under `./data/`:

| Dataset | Clinical Domain | Patients ($M$) | Raw Features | Target Definition |
| :--- | :--- | :--- | :--- | :--- |
| **Wisconsin Breast Cancer** | Oncology | 569 | 30 | Fine Needle Aspirate nuclear characteristics (Malignant=1) |
| **Parkinson's Vocal** | Neurology | 195 | 22 | Biomedical voice frequency and amplitude jitter/shimmer (PD=1) |
| **Statlog Heart Disease** | Cardiology | 270 | 13 | Resting ECG, blood pressure, thallium heart scan (Heart Disease=1) |
| **Pima Indians Diabetes** | Endocrinology | 768 | 8 | Glucose, insulin, BMI, age (Tested Positive=1) |

---

## 🖥️ Streamlit Interactive UI Overview

Launch the diagnostic dashboard (`streamlit run app.py`) to explore four specialized modules:

1. **📊 Tab 1 — Dataset & Quantum Preprocessing:**
   - Raw data inspection, class distribution metrics, and cumulative explained variance ratios.
   - Interactive 2D scatter plots of clinical cohorts mapped onto quantum rotation angles $[0, \pi]^2$.
2. **⚛️ Tab 2 — Quantum Circuits & Kernel Geometry:**
   - Visual circuit gate decompositions of both the Feature Map and Variational Ansatz (`iqp` / `mpl` rendered).
   - Pairwise Quantum Kernel matrix heatmap visualizing state fidelities across the cohort.
   - VQC training convergence loss trajectory across COBYLA optimization iterations.
3. **🏆 Tab 3 — Apples-to-Apples Benchmarks:**
   - Rigorous comparison metrics: Test Accuracy, F1-Score (Disease), Precision, Sensitivity (Recall), ROC-AUC, and Execution Latency.
   - Side-by-side $2 \times 2$ confusion matrices across all 4 classical and quantum models on identical test partitions.
4. **🩺 Tab 4 — Live Patient Diagnostic Simulator:**
   - Real-time clinical inference slider module allowing practitioners to manipulate patient principal feature angles.
   - Live side-by-side disease prediction and calibrated confidence metrics between **QSVC** and **Classical RBF SVM**.

---

## 🚀 Quickstart & Setup Guide

### 1. Clone the Repository
```bash
git clone https://github.com/salasks-git/Quantum_hack.git
cd Quantum_hack
```

### 2. Environment Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Download & Standardize Clinical Datasets
```bash
python3 scripts/download_datasets.py
```
*(Pre-downloads and verifies all 4 clinical datasets into `./data/` for full offline execution).*

### 4. Launch the Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📁 Repository Structure

```text
Quantum_hack/
├── AGENTS.md                  # Autonomous agent setup instructions & project context
├── README.md                  # Comprehensive project documentation
├── app.py                     # Full-featured Streamlit hybrid QML dashboard
├── requirements.txt           # Pinned dependencies (Qiskit 1.x, Torch, Scikit-Learn, Streamlit)
├── .gitignore                 # Excludes venvs, caches, checkpoints; tracks clean CSVs
├── data/                      # Cleaned clinical benchmark datasets
│   ├── breast_cancer_wisconsin.csv
│   ├── diabetes_pima.csv
│   ├── heart_disease_statlog.csv
│   └── parkinsons_vocal.csv
├── scripts/                   # Automated data pipelines
│   └── download_datasets.py   # Dataset ingestion and preprocessing script
└── src/                       # Custom modular source components
```

---

## 🛠️ Technology Stack & Dependencies

- **Quantum Computing:** `qiskit>=1.0.0`, `qiskit-aer>=0.14.0`, `qiskit-machine-learning>=0.7.2`, `qiskit-algorithms>=0.3.0`
- **Machine Learning & Baselines:** `scikit-learn>=1.4.0`, `torch>=2.2.0`
- **Data Engineering:** `pandas>=2.2.0`, `numpy>=1.26.0`
- **Visualization & Web App:** `streamlit>=1.32.0`, `matplotlib>=3.8.0`, `seaborn>=0.13.0`, `pylatexenc>=2.10`

---

## 👥 Contributors & Acknowledgements

Developed for **Qiskit Fall Fest 2026 · Quantum Hack Day Kochi**.  
Special thanks to the **IBM Quantum** and **Qiskit Community** teams for supporting open-source quantum computing education and research.
