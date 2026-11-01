"""Static configuration: dataset registry, UI option lists, model names."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

RANDOM_STATE = 42
TEST_SIZE = 0.25

# Label shown in the UI -> CSV produced by scripts/download_datasets.py
DATASETS = {
    "Parkinson's Vocal Biomarkers (Neurology)": DATA_DIR / "parkinsons_vocal.csv",
    "Statlog Heart Disease (Cardiology)": DATA_DIR / "heart_disease_statlog.csv",
    "Wisconsin Breast Cancer (Oncology)": DATA_DIR / "breast_cancer_wisconsin.csv",
    "Pima Indians Diabetes (Endocrinology)": DATA_DIR / "diabetes_pima.csv",
    "Vertebral Column (Orthopedics)": DATA_DIR / "vertebral_column.csv",
}

QUBIT_OPTIONS = [2, 3, 4]
FEATURE_MAP_ZZ = "ZZFeatureMap"
FEATURE_MAP_PAULI = "PauliFeatureMap (Z, YY, ZX)"
FEATURE_MAP_OPTIONS = [FEATURE_MAP_ZZ, FEATURE_MAP_PAULI]
ENTANGLEMENT_OPTIONS = ["linear", "full", "circular"]

# Model names double as result-dict keys and table labels.
LINEAR_SVM = "Classical SVM (Linear)"
RBF_SVM = "Classical SVM (RBF)"
QSVC_MODEL = "Quantum Kernel SVM (QSVC)"
VQC_MODEL = "Variational Quantum (VQC)"

MODEL_COMPLEXITY = {
    LINEAR_SVM: "O(N_train · d)",
    RBF_SVM: "O(N_train² · d)",
    QSVC_MODEL: "O(N_train² · 2ⁿ)",
    VQC_MODEL: "O(N_train · Iters · Depth)",
}
