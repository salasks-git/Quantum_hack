"""Export the Streamlit Tab 8 "Elite Hybrid Architectures" data per disease for the 3D tour.

Mirrors qmed/ui/tab_quhama.py (read-only): QuHaMA via train_quhama(epochs_enc=50, epochs_gate=40),
Sandwich via run_approach_2(X_*_std, n_qubits, epochs=30), CKTA via run_approach_3(X_*_q, n_qubits),
and the same five-row empirical leaderboard (QuHaMA, Sandwich, CKTA, RBF SVM, Linear SVM).
Writes web/public/data/<slug>.elite.json. Test order equals <slug>.json test[].
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, fbeta_score, precision_score, recall_score, roc_auc_score
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from approaches.approach_2 import run_approach_2  # noqa: E402
from approaches.approach_3 import run_approach_3  # noqa: E402
from qmed.bandwidth import kernel_diagnostics  # noqa: E402
from qmed.circuits import build_feature_map  # noqa: E402
from qmed.config import DATASETS, DEFAULT_BANDWIDTH, RANDOM_STATE  # noqa: E402
from qmed.data import prepare_data  # noqa: E402
from qmed.models import build_quantum_kernel, fit_and_evaluate  # noqa: E402
from qmed.quhama import train_quhama  # noqa: E402
from sklearn.metrics.pairwise import rbf_kernel  # noqa: E402

N_QUBITS, REPS, ENTANGLEMENT, VQC_ITER, MAX_SAMPLES = 3, 2, "linear", 35, 100
FEATURE_MAP = "ZZFeatureMap"
QUHAMA_EPOCHS = (50, 40)   # tab: epochs_enc=50, epochs_gate=40
SANDWICH_EPOCHS = 30       # tab: epochs=30

DESCRIPTIONS = {
    "quhama": "Dual-branch hybrid: a neural encoder compresses the biomarkers, a classical head and a quantum "
              "kernel each estimate disease risk, and a gating network learns a per-patient mix [w_c, w_q].",
    "sandwich": "Neural-Quantum Sandwich HQNN: a classical Pre-Net compresses biomarkers into qubit angles, a "
                "quantum circuit (ZZFeatureMap + ansatz) returns Z expectations, and a classical Post-Net "
                "classifies, absorbing systematic QPU readout noise.",
    "ckta": "Bandwidth-Scaled Q-CKTA: sweeps quantum-kernel bandwidths, keeps the one with the best centered "
            "kernel-target alignment, and fuses it with an RBF kernel using alignment-derived weights for an SVM.",
}
ADVANTAGE = {
    "quhama": "Patient-specific dynamic weight allocation [w_c, w_q]",
    "sandwich": "Post-Net naturally absorbs systematic QPU readout noise",
    "ckta": "Optimal bandwidth s* avoids kernel concentration",
    "rbf": "Strong local Euclidean baseline; lacks non-local parity detection",
    "linear": "Fast convex baseline; fails on complex non-linear disease boundaries",
}


def _f(x):
    return float(x)


def metrics_of(y_true, pred, prob) -> dict:
    """Same metric definitions as the tab / run_approach_*.

    Deviation from the tab's _safe_m (which does prob[:, 1] on a 1-D array and raises IndexError):
    accept either a 1-D P(disease) vector or an (n, 2) matrix.
    """
    prob = np.asarray(prob)
    if prob.ndim == 2:
        prob = prob[:, 1]
    return {
        "accuracy": _f(accuracy_score(y_true, pred)),
        "f2": _f(fbeta_score(y_true, pred, beta=2, zero_division=0)),
        "precision": _f(precision_score(y_true, pred, zero_division=0)),
        "recall": _f(recall_score(y_true, pred, zero_division=0)),
        "roc_auc": _f(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) > 1 else 0.5,
    }


def _test_rows(y_test, pred, prob, **extra):
    rows = []
    for i in range(len(y_test)):
        r = {"pred": int(pred[i]), "prob": _f(prob[i]), "label": int(y_test[i])}
        for k, v in extra.items():
            r[k] = v[i]
        rows.append(r)
    return rows


def export_dataset(label: str, csv_path: Path) -> dict:
    # Determinism (script-level only; run_approach_* / train_quhama also seed 42 themselves).
    torch.manual_seed(42)
    np.random.seed(42)
    csv_path = Path(csv_path)
    data = prepare_data(csv_path, MAX_SAMPLES, N_QUBITS)
    y = data.y_test

    # --- classical baselines (same SVCs as the tab's results[...]) ---
    lin = fit_and_evaluate(SVC(kernel="linear", probability=True, random_state=RANDOM_STATE),
                           data.X_train_q, data.y_train, data.X_test_q)
    rbf = fit_and_evaluate(SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
                           data.X_train_q, data.y_train, data.X_test_q)

    # --- QuHaMA (tab: _cached_quhama_train) ---
    fmap = build_feature_map(FEATURE_MAP, N_QUBITS, REPS, ENTANGLEMENT)
    qh = train_quhama(data.X_train_std, data.y_train, data.X_test_std, y, n_qubits=N_QUBITS,
                      feature_map=fmap, epochs_enc=QUHAMA_EPOCHS[0], epochs_gate=QUHAMA_EPOCHS[1])
    bk = qh.breakdown_test
    qh_time = qh.train_time + qh.inf_time  # tab latency: fmt_duration(quhama_res.execution_time)

    # --- Sandwich (tab: _cached_sandwich_train) ---
    sw = run_approach_2(data.X_train_std, data.y_train, data.X_test_std, y, n_qubits=N_QUBITS,
                        epochs=SANDWICH_EPOCHS)
    sw_model = sw["Model"]
    sw_prob = sw_model.predict_proba(data.X_test_std)[:, 1]
    sw_pred = (sw_prob >= 0.5).astype(int)
    with torch.no_grad():
        z = sw_model.pre_net(torch.tensor(data.X_test_std, dtype=torch.float32)).numpy()
    e = np.asarray(sw_model.qnn.forward((z + 1.0) * (np.pi / 2.0), sw_model.weights_q))
    qc = sw_model.qnn.circuit
    sandwich = {
        "description": DESCRIPTIONS["sandwich"],
        "architecture": {
            "n_qubits": N_QUBITS, "in_features": int(data.X_train_std.shape[1]),
            "pre_net": [int(data.X_train_std.shape[1]), 16, N_QUBITS],
            "post_net": [2 * N_QUBITS, 8, 2],
            "quantum_layer": {"feature_map": FEATURE_MAP, "reps": 1, "entanglement": "linear",
                              "ansatz": "RealAmplitudes-style (qmed.circuits.build_ansatz), reps 1",
                              "n_weights": int(len(sw_model.weights_q)),
                              "weights": [_f(w) for w in sw_model.weights_q],
                              "observables": [f"Z{i}" for i in range(N_QUBITS)],
                              "circuit_depth": int(qc.decompose().decompose().depth()),
                              "circuit_size": int(qc.decompose().decompose().size())},
            "epochs": SANDWICH_EPOCHS, "lr": 0.02,
            "angle_map": "theta = (z + 1) * pi / 2",
        },
        "test": _test_rows(y, sw_pred, sw_prob,
                           latent=[[_f(v) for v in row] for row in z],
                           expectation=[[_f(v) for v in row] for row in e]),
        "metrics": {**metrics_of(y, sw_pred, sw_prob), "time_s": _f(sw["Execution_Time"])},
    }
    assert abs(sandwich["metrics"]["f2"] - sw["F2-Score"]) < 1e-9

    # --- CKTA (tab: _cached_ckta_train) ---
    ck = run_approach_3(data.X_train_q, data.y_train, data.X_test_q, y, n_qubits=N_QUBITS)
    ck_model = ck["Model"]
    ck_prob = ck_model.predict_proba(data.X_test_q)[:, 1]
    ck_pred = np.asarray(ck_model.predict(data.X_test_q)).astype(int)  # tab/approach use svm.predict
    # Alignment sweep (re-evaluated read-only with the model's own helpers).
    g_rbf = rbf_kernel(data.X_train_q, data.X_train_q, gamma=1.0 / data.X_train_q.shape[1])
    a_c = _f(kernel_diagnostics(g_rbf, data.y_train)["alignment"])
    sweep = []
    for s in ck_model.bandwidth_candidates:
        kq = build_quantum_kernel(ck_model.fmap, bandwidth=s).evaluate(data.X_train_q, data.X_train_q)
        sweep.append({"s": _f(s), "alignment": _f(kernel_diagnostics(kq, data.y_train)["alignment"])})
    ckta = {
        "description": DESCRIPTIONS["ckta"],
        "architecture": {
            "n_qubits": N_QUBITS, "feature_map": FEATURE_MAP, "reps": 2, "entanglement": "linear",
            "bandwidth_candidates": [_f(s) for s in ck_model.bandwidth_candidates],
            "bandwidth_sweep": sweep, "best_bandwidth_s": _f(ck["Best_Bandwidth_s"]),
            "alignment_classical_rbf": a_c,
            "alignment_quantum_best": max(s["alignment"] for s in sweep),
            "weight_classical": _f(ck["Weight_Classical"]), "weight_quantum": _f(ck["Weight_Quantum"]),
            "rbf_gamma": _f(1.0 / data.X_train_q.shape[1]),
        },
        "test": _test_rows(y, ck_pred, ck_prob),
        "metrics": {**metrics_of(y, ck_pred, ck_prob), "time_s": _f(ck["Execution_Time"])},
    }
    assert abs(ckta["metrics"]["f2"] - ck["F2-Score"]) < 1e-9

    quhama = {
        "description": DESCRIPTIONS["quhama"],
        "architecture": {
            "n_qubits": N_QUBITS, "feature_map": FEATURE_MAP, "reps": REPS, "entanglement": ENTANGLEMENT,
            "epochs_encoder": QUHAMA_EPOCHS[0], "epochs_gating": QUHAMA_EPOCHS[1],
            "loss_history_encoder": [_f(v) for v in qh.loss_history_encoder],
            "loss_history_gating": [_f(v) for v in qh.loss_history_gating],
            "mean_w_quantum": _f(np.mean(bk["w_quantum"])), "mean_w_classical": _f(np.mean(bk["w_classical"])),
        },
        "test": _test_rows(y, np.asarray(bk["pred"]).astype(int), np.asarray(bk["p_final"]),
                           p_classical=[_f(v) for v in bk["p_classical"]],
                           p_quantum=[_f(v) for v in bk["p_quantum"]],
                           w_classical=[_f(v) for v in bk["w_classical"]],
                           w_quantum=[_f(v) for v in bk["w_quantum"]]),
        "metrics": {**metrics_of(y, np.asarray(bk["pred"]).astype(int), np.asarray(bk["p_final"])),
                    "time_s": _f(qh_time)},
    }

    def row(model, family, key, m):
        return {"model": model, "key": key, "family": family, **{k: m[k] for k in
                ("accuracy", "f2", "precision", "recall", "roc_auc", "time_s")},
                "core_advantage": ADVANTAGE[key]}

    rb = {**metrics_of(y, rbf["pred"], rbf["prob"]), "time_s": _f(rbf["total_time"])}
    ln = {**metrics_of(y, lin["pred"], lin["prob"]), "time_s": _f(lin["total_time"])}
    leaderboard = [
        row("QuHaMA", "elite", "quhama", quhama["metrics"]),
        row("Sandwich", "elite", "sandwich", sandwich["metrics"]),
        row("CKTA", "elite", "ckta", ckta["metrics"]),
        row("RBF SVM", "classical", "rbf", rb),
        row("Linear SVM", "classical", "linear", ln),
    ]
    ckta["architecture"]["core_advantage"] = f"Optimal bandwidth s*={ck['Best_Bandwidth_s']} avoids concentration"
    return {
        "slug": csv_path.stem, "label": label,
        "settings": {"n_qubits": N_QUBITS, "feature_map": FEATURE_MAP, "reps": REPS,
                     "entanglement": ENTANGLEMENT, "vqc_iter": VQC_ITER, "seed": RANDOM_STATE,
                     "max_samples": MAX_SAMPLES, "kernel_bandwidth": float(DEFAULT_BANDWIDTH)},
        "leaderboard": leaderboard,
        "approaches": {"quhama": quhama, "sandwich": sandwich, "ckta": ckta},
    }


def main(out_dir: Path = PROJECT_ROOT / "web/public/data") -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for label, csv_path in DATASETS.items():
        t0 = time.time()
        result = export_dataset(label, csv_path)
        path = out_dir / f"{result['slug']}.elite.json"
        path.write_text(json.dumps(result, separators=(",", ":")))
        print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB) in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
