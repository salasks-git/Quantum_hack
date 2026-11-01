"""Benchmark and timing tables built from ``train_all_models`` results."""
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from qmed.config import MODEL_COMPLEXITY, RBF_SVM


def fmt_duration(seconds: float) -> str:
    """'1.234 s' for >= 10 ms, else '4.2 ms'."""
    return f"{seconds:.3f} s" if seconds >= 0.01 else f"{seconds * 1000:.1f} ms"


def benchmark_table(results: dict, y_test) -> pd.DataFrame:
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
            "Compute Time": fmt_duration(res["total_time"]),
        })
    return pd.DataFrame(rows)


def timing_table(results: dict, n_test: int) -> pd.DataFrame:
    baseline_total = results[RBF_SVM]["total_time"]
    rows = []
    for name, res in results.items():
        t_inf, t_tot = res["inf_time"], res["total_time"]
        throughput = n_test / t_inf if t_inf > 0 else float("inf")
        ratio_vs_rbf = t_tot / baseline_total if baseline_total > 0 else 1.0
        rows.append({
            "Technique": name,
            "Training Time": fmt_duration(res["train_time"]),
            "Inference Time (Test)": fmt_duration(t_inf),
            "Total Process Time": fmt_duration(t_tot),
            "Latency / Patient": f"{t_inf / n_test * 1000:.2f} ms",
            "Throughput": f"{throughput:.1f} patients/s",
            "Overhead vs Classical RBF": f"{ratio_vs_rbf:.1f}x" if ratio_vs_rbf >= 1.0 else f"{1 / ratio_vs_rbf:.1f}x faster",
            "Complexity": MODEL_COMPLEXITY.get(name, "N/A"),
        })
    return pd.DataFrame(rows)
