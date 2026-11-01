"""Export per-disease JSON (models, kernels, decision surfaces) for the 3D diagnosis tour.

``train_bundle`` trains everything and keeps the fitted objects (needed by the live API's
custom-patient predict); ``bundle_to_json`` turns a bundle into the tour JSON schema.
With the default settings the output is identical to the shipped web/public/data files
(apart from wall-clock timings).
"""
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from qiskit.primitives import StatevectorSampler  # noqa: E402
from qiskit_machine_learning.algorithms import QSVC, VQC  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from qmed.circuits import build_ansatz, build_feature_map  # noqa: E402
from qmed.config import (  # noqa: E402
    DATASETS, DEFAULT_BANDWIDTH, FEATURE_MAP_ZZ, HYBRID_ENSEMBLE, LINEAR_SVM, QSVC_MODEL, RANDOM_STATE, RBF_SVM, TEST_SIZE,
    VQC_MODEL,
)
from qmed.data import PreparedData, load_dataset, prepare_data  # noqa: E402
from qmed.hardware import qpu_profile  # noqa: E402
from qmed.models import COBYLA, HybridEnsembleModel, build_quantum_kernel, fit_and_evaluate  # noqa: E402

N_QUBITS, REPS, ENTANGLEMENT, VQC_ITER, MAX_SAMPLES = 3, 2, "linear", 35, 100
# Streamlit parity: the tour uses the Streamlit sidebar's default kernel bandwidth.
KERNEL_BANDWIDTH = float(DEFAULT_BANDWIDTH)
GRID_MIN, GRID_MAX = -0.3, float(np.pi + 0.3)
MODEL_KEYS = {"linear": LINEAR_SVM, "rbf": RBF_SVM, "qsvc": QSVC_MODEL, "vqc": VQC_MODEL}

ProgressFn = Optional[Callable[[str, float], None]]


def _r(a):
    """Round to 4 dp and convert to plain Python floats (nested lists)."""
    return np.round(np.asarray(a, dtype=float), 4).tolist()


@dataclass
class Bundle:
    label: str
    slug: str
    settings: dict          # n_qubits, feature_map, reps, entanglement, vqc_iter, seed, max_samples,
                            # kernel_bandwidth
    data: PreparedData
    std_scaler: StandardScaler
    pca: PCA
    angle_scaler: MinMaxScaler
    feature_map: object
    ansatz: object
    q_kernel: object
    res: dict               # {model name: result dict} like qmed.models.train_all_models
    kernel_test: np.ndarray


def refit_transforms(csv_path: Path, max_samples: int, n_qubits: int, data: PreparedData):
    """Re-fit prepare_data's StandardScaler / PCA / MinMaxScaler (qmed/data.py keeps them private).

    Same splits and random_state as qmed.data.prepare_data; asserts the angles match.
    """
    _, X, y, _ = load_dataset(csv_path)
    if len(X) > max_samples:
        X, _, y, _ = train_test_split(X, y, train_size=max_samples, random_state=RANDOM_STATE, stratify=y)
    X_train, X_test, _, _ = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    std_scaler = StandardScaler()
    pca = PCA(n_components=n_qubits, random_state=RANDOM_STATE)
    angle_scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_train_q = angle_scaler.fit_transform(pca.fit_transform(std_scaler.fit_transform(X_train)))
    X_test_q = angle_scaler.transform(pca.transform(std_scaler.transform(X_test)))
    if not (np.allclose(X_train_q, data.X_train_q) and np.allclose(X_test_q, data.X_test_q)):
        raise AssertionError("re-fitted transforms differ from qmed.data.prepare_data")
    return std_scaler, pca, angle_scaler


def train_models(X_train_q, y_train, X_test_q, feature_map, ansatz, vqc_iter: int,
                 seed=RANDOM_STATE, kernel_bandwidth: float = KERNEL_BANDWIDTH, progress: ProgressFn = None):
    """Same steps, objects and seeds as qmed.models.train_all_models, plus stage progress.

    Returns (results, q_kernel, kernel_test). tests/test_live_api.py checks it matches
    train_all_models exactly.
    """
    report = progress or (lambda stage, frac: None)
    results = {}

    report("classical SVMs", 0.02)
    results[LINEAR_SVM] = fit_and_evaluate(
        SVC(kernel="linear", probability=True, random_state=RANDOM_STATE), X_train_q, y_train, X_test_q
    )
    results[RBF_SVM] = fit_and_evaluate(
        SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE), X_train_q, y_train, X_test_q
    )

    report("QSVC kernel", 0.03)
    # Exact statevector kernel with the bandwidth baked into the circuit, so q_kernel / QSVC.predict
    # on raw angles (test set, live custom patient) are scaled consistently.
    q_kernel = build_quantum_kernel(feature_map, kernel_bandwidth)
    qsvc = QSVC(quantum_kernel=q_kernel, probability=True, random_state=RANDOM_STATE)
    results[QSVC_MODEL] = fit_and_evaluate(qsvc, X_train_q, y_train, X_test_q)
    report("QSVC kernel", 0.05)
    results[QSVC_MODEL]["kernel_matrix"] = q_kernel.evaluate(x_vec=X_train_q)
    report("QSVC kernel", 0.07)
    kernel_test = q_kernel.evaluate(x_vec=X_test_q, y_vec=X_train_q)

    report("VQC training", 0.10)
    loss_history, weights_history = [], []
    initial_point = (
        np.random.default_rng(seed).uniform(-np.pi, np.pi, ansatz.num_parameters)
        if seed is not None else None
    )

    def _record(weights, loss):
        weights_history.append(list(map(float, weights)))
        loss_history.append(float(loss))
        report("VQC training", 0.10 + 0.85 * min(1.0, len(loss_history) / max(1, vqc_iter)))

    vqc = VQC(sampler=StatevectorSampler(seed=seed), feature_map=feature_map, ansatz=ansatz,
              optimizer=COBYLA(maxiter=vqc_iter), callback=_record, initial_point=initial_point)
    results[VQC_MODEL] = fit_and_evaluate(vqc, X_train_q, y_train, X_test_q)
    results[VQC_MODEL]["loss_history"] = loss_history
    results[VQC_MODEL]["weights_history"] = weights_history

    report("hybrid", 0.96)
    prob_ens = 0.5 * results[RBF_SVM]["prob"] + 0.5 * results[QSVC_MODEL]["prob"]
    t_train = results[RBF_SVM]["train_time"] + results[QSVC_MODEL]["train_time"]
    t_inf = results[RBF_SVM]["inf_time"] + results[QSVC_MODEL]["inf_time"]
    results[HYBRID_ENSEMBLE] = {
        "model": HybridEnsembleModel(results[RBF_SVM]["model"], results[QSVC_MODEL]["model"]),
        "pred": (prob_ens >= 0.5).astype(int), "prob": prob_ens,
        "train_time": t_train, "inf_time": t_inf, "total_time": t_train + t_inf,
    }
    return results, q_kernel, kernel_test


def train_bundle(label: str, csv_path: Path, *, max_samples: int = MAX_SAMPLES, n_qubits: int = N_QUBITS,
                 feature_map: str = FEATURE_MAP_ZZ, reps: int = REPS, entanglement: str = ENTANGLEMENT,
                 vqc_iter: int = VQC_ITER, kernel_bandwidth: float = KERNEL_BANDWIDTH,
                 progress: ProgressFn = None) -> Bundle:
    csv_path = Path(csv_path)
    if progress:
        progress("preparing data", 0.0)
    data = prepare_data(csv_path, max_samples, n_qubits)
    std_scaler, pca, angle_scaler = refit_transforms(csv_path, max_samples, n_qubits, data)
    fm = build_feature_map(feature_map, n_qubits, reps, entanglement)
    ansatz = build_ansatz(n_qubits, reps, entanglement)
    res, q_kernel, kernel_test = train_models(data.X_train_q, data.y_train, data.X_test_q, fm, ansatz,
                                              vqc_iter, seed=RANDOM_STATE, kernel_bandwidth=kernel_bandwidth,
                                              progress=progress)
    settings = {"n_qubits": n_qubits, "feature_map": feature_map, "reps": reps,
                "entanglement": entanglement, "vqc_iter": vqc_iter, "seed": RANDOM_STATE,
                "max_samples": max_samples, "kernel_bandwidth": float(kernel_bandwidth)}
    return Bundle(label=label, slug=csv_path.stem, settings=settings, data=data, std_scaler=std_scaler,
                  pca=pca, angle_scaler=angle_scaler, feature_map=fm, ansatz=ansatz, q_kernel=q_kernel,
                  res=res, kernel_test=kernel_test)


def rbf_grid(rbf_model, X_train_q, n_qubits: int, grid_size: int) -> dict:
    """RBF decision values on a grid in angle space.

    n=3: G^3 over (x1, x2, x3). n=2: G x G x 1 over (x1, x2) drawn at z=0 (``grid_dims: 2``).
    n=4: G^3 over (x1, x2, x3) with x4 fixed at the training mean (``grid_dims: 3``, ``slice``).
    Values are flattened in C order with indexing="ij" (x1 slowest).
    """
    axis = np.linspace(GRID_MIN, GRID_MAX, grid_size)
    out = {"grid_size": grid_size, "min": GRID_MIN, "max": GRID_MAX}
    extra = {}
    if n_qubits == 2:
        gx, gy = np.meshgrid(axis, axis, indexing="ij")
        grid = np.column_stack([gx.ravel(), gy.ravel()])
        extra = {"grid_dims": 2}
    else:
        gx, gy, gz = np.meshgrid(axis, axis, axis, indexing="ij")
        cols = [gx.ravel(), gy.ravel(), gz.ravel()]
        if n_qubits == 4:
            x4 = float(np.asarray(X_train_q, dtype=float)[:, 3].mean())
            cols.append(np.full(gx.size, x4))
            extra = {"grid_dims": 3, "slice": {"dim": 3, "value": float(np.round(x4, 4))}}
        grid = np.column_stack(cols)
    out["values"] = _r(rbf_model.decision_function(grid))
    out.update(extra)
    return out


def bundle_to_json(bundle: Bundle, grid_size: int = 24) -> dict:
    data, res, s = bundle.data, bundle.res, bundle.settings
    n_qubits, vqc_iter = s["n_qubits"], s["vqc_iter"]
    test = []
    for i in range(len(data.y_test)):
        models = {k: {"pred": int(res[n]["pred"][i]), "prob": float(np.round(res[n]["prob"][i], 4))}
                  for k, n in MODEL_KEYS.items()}
        hp = float(np.round(res[HYBRID_ENSEMBLE]["prob"][i], 4))
        models["hybrid"] = {"pred": int(res[HYBRID_ENSEMBLE]["pred"][i]), "prob": hp}
        test.append({
            "index": i, "label": int(data.y_test[i]),
            "raw": _r(data.X_test_raw[i]), "standardized": _r(data.X_test_std[i]),
            "pca": _r(data.X_test_pca[i]), "angles": _r(data.X_test_q[i]),
            "models": models,
        })

    settings = {"n_qubits": n_qubits, "feature_map": s["feature_map"], "reps": s["reps"],
                "entanglement": s["entanglement"], "vqc_iter": vqc_iter, "seed": s["seed"]}
    settings["kernel_bandwidth"] = float(s.get("kernel_bandwidth", KERNEL_BANDWIDTH))
    lin = res[LINEAR_SVM]["model"]
    full = data.df.drop(columns=["target"]).values
    return {
        "label": bundle.label, "slug": bundle.slug,
        "feature_names": list(data.feature_names),
        "feature_ranges": {"min": _r(full.min(axis=0)), "max": _r(full.max(axis=0))},
        "settings": settings,
        "train": {"angles": _r(data.X_train_q), "labels": [int(v) for v in data.y_train]},
        "test": test,
        "timings": {k: {"train_s": float(res[n]["train_time"]), "inf_s": float(res[n]["inf_time"]),
                        "total_s": float(res[n]["total_time"])}
                    for k, n in {**MODEL_KEYS, "hybrid": HYBRID_ENSEMBLE}.items()},
        "qpu": qpu_profile(bundle.feature_map, bundle.ansatz, n_qubits, len(data.X_train_q),
                           len(data.X_test_q), vqc_iter),
        "linear": {"coef": _r(lin.coef_[0]), "intercept": float(np.round(lin.intercept_[0], 4))},
        "rbf": rbf_grid(res[RBF_SVM]["model"], data.X_train_q, n_qubits, grid_size),
        "qsvc": {"kernel_train": _r(res[QSVC_MODEL]["kernel_matrix"]), "kernel_test": _r(bundle.kernel_test)},
        "vqc": {"loss_history": _r(res[VQC_MODEL]["loss_history"]),
                "weights_history": _r(res[VQC_MODEL]["weights_history"])},
    }


def export_dataset(label: str, csv_path: Path, grid_size: int = 24, *, max_samples: int = MAX_SAMPLES,
                   n_qubits: int = N_QUBITS, feature_map: str = FEATURE_MAP_ZZ, reps: int = REPS,
                   entanglement: str = ENTANGLEMENT, vqc_iter: int = VQC_ITER,
                   kernel_bandwidth: float = KERNEL_BANDWIDTH) -> dict:
    bundle = train_bundle(label, csv_path, max_samples=max_samples, n_qubits=n_qubits,
                          feature_map=feature_map, reps=reps, entanglement=entanglement, vqc_iter=vqc_iter,
                          kernel_bandwidth=kernel_bandwidth)
    return bundle_to_json(bundle, grid_size)


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
