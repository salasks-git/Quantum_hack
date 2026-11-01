import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ENV_FILE = PROJECT_ROOT / ".env"

# Auto-load .env if present (works with or without python-dotenv)
def _load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(path)
    except ImportError:
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip().strip("'\"")
                    if k and k not in os.environ:
                        os.environ[k] = v
        except Exception:
            pass

_load_env_file(ENV_FILE)

# IBM Quantum Platform API Token
QISKIT_IBM_TOKEN = os.getenv("QISKIT_IBM_TOKEN", "")


RANDOM_STATE = 42
TEST_SIZE = 0.25

# Label shown in the UI -> CSV produced by scripts/download_datasets.py
DATASETS = {
    "Parkinson's Vocal Biomarkers (Neurology)": DATA_DIR / "parkinsons_vocal.csv",
    "Statlog Heart Disease (Cardiology)": DATA_DIR / "heart_disease_statlog.csv",
    "Wisconsin Breast Cancer (Oncology)": DATA_DIR / "breast_cancer_wisconsin.csv",
    "Pima Indians Diabetes (Endocrinology)": DATA_DIR / "diabetes_pima.csv",
    "Vertebral Column (Orthopedics)": DATA_DIR / "vertebral_column.csv",
    "Acute Inflammations (Nephrology / Urology)": DATA_DIR / "acute_inflammations.csv",
}


QUBIT_OPTIONS = [2, 3, 4]
FEATURE_MAP_ZZ = "ZZFeatureMap"
FEATURE_MAP_PAULI = "PauliFeatureMap (Z, YY, ZX)"
FEATURE_MAP_OPTIONS = [FEATURE_MAP_ZZ, FEATURE_MAP_PAULI]
ENTANGLEMENT_OPTIONS = ["linear", "full", "circular"]

# Quantum kernel bandwidth s: QSVC encodes s * angles instead of the raw [0, pi]
# angles. Small s widens the kernel (Shaydulin & Wild 2022); s = 1 is the unscaled map.
BANDWIDTH_OPTIONS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
DEFAULT_BANDWIDTH = 0.1
CV_FOLDS = 5

# Model names double as result-dict keys and table labels.
LINEAR_SVM = "Classical SVM (Linear)"
RBF_SVM = "Classical SVM (RBF)"
QSVC_MODEL = "Quantum Kernel SVM (QSVC)"
VQC_MODEL = "Variational Quantum (VQC)"
HYBRID_ENSEMBLE = "Hybrid Ensemble (QSVC + RBF)"

MODEL_COMPLEXITY = {
    LINEAR_SVM: "O(N_train · d)",
    RBF_SVM: "O(N_train² · d)",
    QSVC_MODEL: "O(N_train² · 2ⁿ)",
    VQC_MODEL: "O(N_train · Iters · Depth)",
    HYBRID_ENSEMBLE: "O(N_train² · (d + 2ⁿ))",
}
