"""Export Kernel Bandwidth Lab and QuHaMA data per disease for the 3D diagnosis tour.

Writes web/public/data/<slug>.bandwidth.json and <slug>.quhama.json. Read-only use of the
pipeline: calls qmed.bandwidth and qmed.quhama exactly like qmed/ui/tab_bandwidth.py and
qmed/ui/tab_quhama.py do (tour settings: 100 patients, 3 qubits, ZZFeatureMap, reps 2, linear).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from qmed.bandwidth import bandwidth_sweep, kernel_diagnostics, kernel_matrices  # noqa: E402
from qmed.circuits import build_feature_map  # noqa: E402
from qmed.config import BANDWIDTH_OPTIONS, DATASETS, DEFAULT_BANDWIDTH, FEATURE_MAP_ZZ  # noqa: E402
from qmed.data import prepare_data  # noqa: E402
from qmed.quhama import train_quhama  # noqa: E402

N_QUBITS, REPS, ENTANGLEMENT, MAX_SAMPLES = 3, 2, "linear", 100
EPOCHS_ENC, EPOCHS_GATE = 60, 45  # the QuHaMA tab's training call
MAX_KERNEL_N = 40


def _regime(kernel_mean: float) -> dict:
    """Same thresholds and text as qmed.ui.tab_bandwidth._regime (copied; that module imports streamlit)."""
    if kernel_mean < 0.15:
        return {"key": "sharp", "text": "Too sharp: near-identity, every patient looks unique, so the SVM memorizes."}
    if kernel_mean > 0.85:
        return {"key": "blurred", "text": "Too blurred: near all-ones, every patient looks the same."}
    return {"key": "balanced", "text": "Balanced: similar patients score high, different patients score low."}


def _r(a, nd=3):
    return np.round(np.asarray(a, dtype=float), nd).tolist()


def _downsample_order(y: np.ndarray, n_max: int = MAX_KERNEL_N) -> list:
    """Indices into the training set, grouped by label (healthy then disease), at most n_max.

    Takes classes in proportion to their size (first patients of each class, in training
    order) so both diagonal blocks stay visible; all patients if n_train <= n_max.
    """
    idx = {c: [int(i) for i in np.flatnonzero(y == c)] for c in (0, 1)}
    if len(y) <= n_max:
        return idx[0] + idx[1]
    n1 = int(round(n_max * len(idx[1]) / len(y)))
    n1 = min(max(n1, 1), len(idx[1]))
    n0 = min(n_max - n1, len(idx[0]))
    return idx[0][:n0] + idx[1][:n1]


def export_bandwidth(csv_path: Path, data) -> dict:
    fm = build_feature_map(FEATURE_MAP_ZZ, N_QUBITS, REPS, ENTANGLEMENT)
    bws = tuple(BANDWIDTH_OPTIONS)
    kernels = kernel_matrices(data.X_train_q, fm, bws)
    order = _downsample_order(data.y_train)
    sweep = bandwidth_sweep(csv_path, MAX_SAMPLES, N_QUBITS, FEATURE_MAP_ZZ, REPS, ENTANGLEMENT)

    diagnostics, regime, kern = {}, {}, {}
    for s in bws:
        d = kernel_diagnostics(kernels[s], data.y_train)  # on the full training Gram matrix, like the tab
        diagnostics[str(s)] = {k: round(v, 6) for k, v in d.items()}
        regime[str(s)] = _regime(d["kernel_mean"])
        kern[str(s)] = _r(kernels[s][np.ix_(order, order)], 3)
    rows = [{k: (round(float(v), 6)) for k, v in rec.items()} for rec in sweep.table.to_dict("records")]
    best = max(rows, key=lambda r: r["accuracy"])  # first max, like DataFrame.idxmax
    return {
        "slug": csv_path.stem,
        "settings": {"n_samples": MAX_SAMPLES, "n_qubits": N_QUBITS, "feature_map": FEATURE_MAP_ZZ,
                     "reps": REPS, "entanglement": ENTANGLEMENT, "n_folds": sweep.n_folds,
                     "n_patients": sweep.n_patients, "n_train": int(len(data.y_train))},
        "options": list(bws),
        "default": DEFAULT_BANDWIDTH,
        "order": order,
        "labels": [int(data.y_train[i]) for i in order],
        "kernels": kern,
        "diagnostics": diagnostics,
        "regime": regime,
        "sweep": {"rows": rows, "rbf": {k: round(float(v), 6) for k, v in sweep.rbf.items()},
                  "best_s": best["bandwidth"], "best_accuracy": best["accuracy"]},
    }


def export_quhama(csv_path: Path, data) -> dict:
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    fm = build_feature_map(FEATURE_MAP_ZZ, N_QUBITS, REPS, ENTANGLEMENT)
    res = train_quhama(X_train_std=data.X_train_std, y_train=data.y_train, X_test_std=data.X_test_std,
                       y_test=data.y_test, n_qubits=N_QUBITS, feature_map=fm,
                       epochs_enc=EPOCHS_ENC, epochs_gate=EPOCHS_GATE)
    bk = res.breakdown_test
    m = res.model
    enc_lin = [l for l in m.encoder.net if isinstance(l, torch.nn.Linear)]
    head_lin = [l for l in m.classical_head.net if isinstance(l, torch.nn.Linear)]
    gate_lin = [l for l in m.gating_net.net if isinstance(l, torch.nn.Linear)]
    arch = {
        "in_features": int(data.X_train_std.shape[1]), "latent_dim": int(m.latent_dim),
        "encoder_hidden": int(enc_lin[0].out_features), "head_hidden": int(head_lin[0].out_features),
        "gate_hidden": int(gate_lin[0].out_features),
        "branches": ["classical", "quantum"],
        "encoder": f"Linear({enc_lin[0].in_features},{enc_lin[0].out_features}) - BatchNorm - ReLU - "
                   f"Linear({enc_lin[1].in_features},{enc_lin[1].out_features}) - Tanh",
        "classical_head": f"Linear({head_lin[0].in_features},{head_lin[0].out_features}) - ReLU - "
                          f"Linear({head_lin[1].in_features},{head_lin[1].out_features}) - Softmax",
        "quantum_branch": f"QSVC, {FEATURE_MAP_ZZ} reps {REPS} {ENTANGLEMENT}, on angles (z+1)*pi/2",
        "gating": f"Linear({gate_lin[0].in_features},{gate_lin[0].out_features}) - ReLU - "
                  f"Linear({gate_lin[1].in_features},{gate_lin[1].out_features}) - Softmax over [classical, quantum]",
        "fusion": "p_final = w_classical*p_classical + w_quantum*p_quantum; pred = p_final >= 0.5",
    }
    y_test = data.y_test
    patients = []
    for i in range(len(y_test)):
        patients.append({
            "index": i, "label": int(y_test[i]),
            "z_latent": _r(bk["z_latent"][i], 6), "z_angles": _r(bk["z_angles"][i], 6),
            "p_classical": float(np.round(bk["p_classical"][i], 8)),
            "p_quantum": float(np.round(bk["p_quantum"][i], 8)),
            "w_classical": float(np.round(bk["w_classical"][i], 8)),
            "w_quantum": float(np.round(bk["w_quantum"][i], 8)),
            "p_final": float(np.round(bk["p_final"][i], 8)),
            "pred": int(bk["pred"][i]),
        })
    acc_c = float(np.mean((bk["p_classical"] >= 0.5).astype(int) == y_test))
    acc_q = float(np.mean((bk["p_quantum"] >= 0.5).astype(int) == y_test))
    return {
        "slug": csv_path.stem,
        "settings": {"n_samples": MAX_SAMPLES, "n_qubits": N_QUBITS, "feature_map": FEATURE_MAP_ZZ, "reps": REPS,
                     "entanglement": ENTANGLEMENT, "epochs_encoder": EPOCHS_ENC, "epochs_gating": EPOCHS_GATE,
                     "seed": 42},
        "architecture": arch,
        "history": {"encoder": _r(res.loss_history_encoder, 6), "gating": _r(res.loss_history_gating, 6)},
        "metrics": {k: round(float(v), 6) for k, v in res.metrics.items()},
        "branch_summary": {"classical_accuracy": round(acc_c, 6), "quantum_accuracy": round(acc_q, 6),
                           "mean_w_classical": round(float(np.mean(bk["w_classical"])), 6),
                           "mean_w_quantum": round(float(np.mean(bk["w_quantum"])), 6)},
        "timings": {"train_s": round(float(res.train_time), 3), "inf_s": round(float(res.inf_time), 4)},
        "test": patients,
    }


def export_dataset(csv_path: Path) -> tuple:
    csv_path = Path(csv_path)
    data = prepare_data(csv_path, MAX_SAMPLES, N_QUBITS)
    return export_bandwidth(csv_path, data), export_quhama(csv_path, data)


def main(out_dir: Path = PROJECT_ROOT / "web/public/data", only: str | None = None) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for csv_path in DATASETS.values():
        if only and csv_path.stem != only:
            continue
        t = time.time()
        bw, qh = export_dataset(csv_path)
        for kind, obj in (("bandwidth", bw), ("quhama", qh)):
            p = out_dir / f"{csv_path.stem}.{kind}.json"
            p.write_text(json.dumps(obj, separators=(",", ":")))
            print(f"wrote {p} ({p.stat().st_size / 1024:.0f} KB)", flush=True)
        print(f"  {csv_path.stem}: {time.time() - t:.0f}s", flush=True)
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(only=sys.argv[1] if len(sys.argv) > 1 else None)
