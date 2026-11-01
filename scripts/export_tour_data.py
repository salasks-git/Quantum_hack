"""Export per-disease JSON (models, kernels, decision surfaces) for the 3D diagnosis tour."""
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from qmed.circuits import build_ansatz, build_feature_map  # noqa: E402
from qmed.config import (  # noqa: E402
    DATASETS, FEATURE_MAP_ZZ, LINEAR_SVM, QSVC_MODEL, RANDOM_STATE, RBF_SVM, VQC_MODEL,
)
from qmed.data import prepare_data  # noqa: E402
from qmed.models import train_all_models  # noqa: E402

N_QUBITS, REPS, ENTANGLEMENT, VQC_ITER, MAX_SAMPLES = 3, 2, "linear", 35, 100
GRID_MIN, GRID_MAX = -0.3, float(np.pi + 0.3)


def _r(a):
    """Round to 4 dp and convert to plain Python floats (nested lists)."""
    return np.round(np.asarray(a, dtype=float), 4).tolist()


def export_dataset(label: str, csv_path: Path, grid_size: int = 24) -> dict:
    csv_path = Path(csv_path)
    data = prepare_data(csv_path, MAX_SAMPLES, N_QUBITS)
    fm = build_feature_map(FEATURE_MAP_ZZ, N_QUBITS, REPS, ENTANGLEMENT)
    ansatz = build_ansatz(N_QUBITS, REPS, ENTANGLEMENT)
    res = train_all_models(data.X_train_q, data.y_train, data.X_test_q, fm, ansatz,
                           VQC_ITER, seed=RANDOM_STATE)

    keys = {"linear": LINEAR_SVM, "rbf": RBF_SVM, "qsvc": QSVC_MODEL, "vqc": VQC_MODEL}
    test = []
    for i in range(len(data.y_test)):
        models = {k: {"pred": int(res[n]["pred"][i]), "prob": float(np.round(res[n]["prob"][i], 4))}
                  for k, n in keys.items()}
        test.append({
            "index": i, "label": int(data.y_test[i]),
            "raw": _r(data.X_test_raw[i]), "standardized": _r(data.X_test_std[i]),
            "pca": _r(data.X_test_pca[i]), "angles": _r(data.X_test_q[i]),
            "models": models,
        })

    lin = res[LINEAR_SVM]["model"]
    rbf = res[RBF_SVM]["model"]
    axis = np.linspace(GRID_MIN, GRID_MAX, grid_size)
    gx, gy, gz = np.meshgrid(axis, axis, axis, indexing="ij")
    grid = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    rbf_values = rbf.decision_function(grid)

    q_kernel = res[QSVC_MODEL]["model"].quantum_kernel
    kernel_test = q_kernel.evaluate(x_vec=data.X_test_q, y_vec=data.X_train_q)

    full = data.df.drop(columns=["target"]).values
    return {
        "label": label, "slug": csv_path.stem,
        "feature_names": list(data.feature_names),
        "feature_ranges": {"min": _r(full.min(axis=0)), "max": _r(full.max(axis=0))},
        "settings": {"n_qubits": N_QUBITS, "feature_map": FEATURE_MAP_ZZ, "reps": REPS,
                     "entanglement": ENTANGLEMENT, "vqc_iter": VQC_ITER, "seed": RANDOM_STATE},
        "train": {"angles": _r(data.X_train_q), "labels": [int(v) for v in data.y_train]},
        "test": test,
        "linear": {"coef": _r(lin.coef_[0]), "intercept": float(np.round(lin.intercept_[0], 4))},
        "rbf": {"grid_size": grid_size, "min": GRID_MIN, "max": GRID_MAX, "values": _r(rbf_values)},
        "qsvc": {"kernel_train": _r(res[QSVC_MODEL]["kernel_matrix"]), "kernel_test": _r(kernel_test)},
        "vqc": {"loss_history": _r(res[VQC_MODEL]["loss_history"]),
                "weights_history": _r(res[VQC_MODEL]["weights_history"])},
    }


def main(out_dir: Path = PROJECT_ROOT / "web/public/data") -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for label, csv_path in DATASETS.items():
        result = export_dataset(label, csv_path)
        path = out_dir / f"{result['slug']}.json"
        path.write_text(json.dumps(result, separators=(",", ":")))
        print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)", flush=True)
        index.append({"slug": result["slug"], "label": label,
                      "n_features": len(result["feature_names"])})
    path = out_dir / "index.json"
    path.write_text(json.dumps(index, indent=2))
    print(f"wrote {path} ({path.stat().st_size} B)")


if __name__ == "__main__":
    main()
