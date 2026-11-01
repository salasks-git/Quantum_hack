"""Matplotlib figure builders. Each returns a Figure; ``show`` renders and frees it."""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st
from sklearn.metrics import confusion_matrix


def show(fig) -> None:
    """Render in Streamlit, then close so figures don't pile up across reruns."""
    st.pyplot(fig)
    plt.close(fig)


def circuit_figure(circuit):
    return circuit.decompose().draw(output="mpl", style="iqp")


def patient_scatter(X_q, y):
    fig, ax = plt.subplots(figsize=(5, 3.8))
    scatter = ax.scatter(X_q[:, 0], X_q[:, 1], c=y, cmap="coolwarm", edgecolors="k", s=65)
    ax.set_xlabel("Qubit 0 Angle (PC1 in [0, π] rad)")
    ax.set_ylabel("Qubit 1 Angle (PC2 in [0, π] rad)")
    ax.set_title("Patients Mapped to Quantum Rotation Angles")
    fig.colorbar(scatter, ax=ax, label="0: Healthy | 1: Disease")
    return fig


def kernel_heatmap(kernel_matrix):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(kernel_matrix, cmap="viridis", ax=ax)
    ax.set_title("Pairwise State Fidelity in 2ⁿ Hilbert Space")
    ax.set_xlabel("Patient Index i")
    ax.set_ylabel("Patient Index j")
    return fig


def loss_curve(loss_history):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(range(1, len(loss_history) + 1), loss_history, marker="o", color="#7c3aed", linewidth=2)
    ax.set_title("Cross-Entropy Objective vs COBYLA Iterations")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective Loss")
    ax.grid(True, alpha=0.3)
    return fig


def confusion_heatmap(y_true, y_pred, title: str):
    fig, ax = plt.subplots(figsize=(3.2, 2.8))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Healthy", "Disease"], yticklabels=["Healthy", "Disease"])
    ax.set_title(title, fontsize=9)
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    return fig


def total_time_bars(results: dict):
    names = list(results.keys())
    totals = [results[m]["total_time"] for m in names]
    colors = ["#2563eb", "#0284c7", "#7c3aed", "#d946ef", "#10b981"]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    bars = ax.barh(names, totals, color=colors[:len(names)], edgecolor="black", alpha=0.85)
    ax.set_xlabel("Total Execution Time (Seconds)")
    ax.set_title("End-to-End Wall-Clock Latency (Train + Test)")
    for bar in bars:
        w = bar.get_width()
        ax.text(w + max(totals) * 0.02, bar.get_y() + bar.get_height() / 2,
                f"{w:.3f}s" if w >= 0.01 else f"{w * 1000:.1f}ms",
                va="center", ha="left", fontsize=9, fontweight="bold")
    ax.set_xlim(0, max(totals) * 1.3)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def train_vs_inference_bars(results: dict):
    names = list(results.keys())
    train_times = [results[m]["train_time"] for m in names]
    inf_times = [results[m]["inf_time"] for m in names]
    y_pos = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.barh(y_pos, train_times, label="Training / Kernel Eval", color="#4f46e5", alpha=0.85, edgecolor="black")
    ax.barh(y_pos, inf_times, left=train_times, label="Test Inference", color="#06b6d4", alpha=0.85, edgecolor="black")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("Time (Seconds)")
    ax.set_title("Train vs Inference Breakdown")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def pareto_frontier_scatter(results: dict, y_test):
    from sklearn.metrics import fbeta_score
    names = list(results.keys())
    times = [results[m]["total_time"] for m in names]
    f2_scores = [fbeta_score(y_test, results[m]["pred"], beta=2, zero_division=0) for m in names]
    colors = ["#2563eb", "#0284c7", "#7c3aed", "#d946ef"]

    fig, ax = plt.subplots(figsize=(6, 3.8))
    for i, (name, t, f2) in enumerate(zip(names, times, f2_scores)):
        ax.scatter(t, f2, s=160, color=colors[i % len(colors)], edgecolors="black", zorder=5, label=name)
        short_name = name.split("(")[-1].replace(")", "").strip() if "(" in name else name
        ax.annotate(f" {short_name}", (t, f2), fontsize=9, fontweight="bold", xytext=(5, 4), textcoords="offset points")

    ax.set_xlabel("Total Wall-Clock Time (Seconds, log scale)")
    ax.set_xscale("log")
    ax.set_ylabel("Clinical Target (F2-Score)")
    ax.set_ylim(-0.05, 1.1)
    ax.set_title("Efficiency Frontier: Clinical Target (F2) vs. Latency")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    return fig


# --- Kernel Bandwidth Lab -------------------------------------------------------------
ACC_COLOR = "#2a78d6"
AUC_COLOR = "#eb6834"
MARKER_COLOR = "#52514e"


def kernel_small_multiples(kernels: dict, y, titles: dict):
    """Side-by-side Gram matrices on one shared [0, 1] color scale.

    Patients are sorted by diagnosis (healthy first) so a useful kernel shows two
    bright diagonal blocks; the white lines mark the class boundary.
    """
    order = np.argsort(y, kind="stable")
    boundary = int(np.sum(np.asarray(y) == 0))
    fig, axes = plt.subplots(1, len(kernels), figsize=(4.2 * len(kernels), 3.9), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, (s, K) in zip(axes, kernels.items()):
        im = ax.imshow(K[np.ix_(order, order)], cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
        ax.axhline(boundary - 0.5, color="white", linewidth=1)
        ax.axvline(boundary - 0.5, color="white", linewidth=1)
        ax.set_title(titles[s], fontsize=10)
        ax.set_xticks([boundary / 2, (boundary + len(y)) / 2], ["Healthy", "Disease"], fontsize=8)
        ax.set_yticks([boundary / 2, (boundary + len(y)) / 2], ["Healthy", "Disease"], fontsize=8, rotation=90, va="center")
        ax.tick_params(length=0)
    fig.colorbar(im, ax=axes, shrink=0.85, label="Kernel value K(x, x′)")
    return fig


def _mark_current(ax, current_s) -> None:
    ax.axvline(current_s, color=MARKER_COLOR, linestyle=":", linewidth=1.5)
    ax.annotate(f"your s = {current_s:g}", (current_s, 1.0), xycoords=("data", "axes fraction"),
                xytext=(4, -12), textcoords="offset points", fontsize=8, color=MARKER_COLOR)


def _log_bandwidth_axis(ax, bandwidths) -> None:
    ax.set_xscale("log")
    ax.set_xticks(bandwidths, [f"{s:g}" for s in bandwidths])
    ax.minorticks_off()
    ax.set_xlabel("Kernel bandwidth s (log scale)")
    ax.grid(True, alpha=0.3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def bandwidth_scores(table, rbf: dict, current_s: float):
    """CV accuracy and ROC-AUC of QSVC vs bandwidth, RBF SVM as dashed reference lines."""
    s = table["bandwidth"].to_numpy()
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for col, color, name in [("accuracy", ACC_COLOR, "Accuracy"), ("roc_auc", AUC_COLOR, "ROC-AUC")]:
        ax.plot(s, table[col], marker="o", markersize=6, linewidth=2, color=color, label=f"QSVC {name}")
        ax.axhline(rbf[col], color=color, linestyle="--", linewidth=1.2, alpha=0.8, label=f"Classical RBF {name}")
        ax.annotate(f"RBF {name} {rbf[col]:.2f}", (s.max(), rbf[col]), xytext=(0, 3), textcoords="offset points",
                    fontsize=8, ha="right", va="bottom", color="#333333")
    _mark_current(ax, current_s)
    _log_bandwidth_axis(ax, s)
    ax.set_ylabel("Mean over CV folds")
    lo = min(table["accuracy"].min(), table["roc_auc"].min(), rbf["accuracy"], rbf["roc_auc"])
    ax.set_ylim(max(0.0, lo - 0.08), 1.0)
    ax.set_title("QSVC Cross-Validated Score vs Kernel Bandwidth")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), fontsize=8, ncol=2, frameon=False)
    fig.tight_layout()
    return fig


def bandwidth_diagnostics(table, current_s: float):
    """Two panels (separate y-axes, no twin axis): kernel concentration and kernel-target alignment."""
    s = table["bandwidth"].to_numpy()
    mean = table["kernel_mean"].to_numpy()
    std = np.sqrt(table["kernel_var"].to_numpy())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))

    ax1.fill_between(s, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1), color=ACC_COLOR, alpha=0.18,
                     linewidth=0, label="± 1 std (spread)")
    ax1.plot(s, mean, marker="o", markersize=6, linewidth=2, color=ACC_COLOR, label="Off-diagonal mean")
    ax1.set_ylim(0, 1.02)
    ax1.set_ylabel("Kernel value between two patients")
    ax1.set_title("Kernel Concentration", fontsize=10)
    ax1.legend(loc="upper right", fontsize=8, frameon=False)

    ax2.plot(s, table["alignment"], marker="o", markersize=6, linewidth=2, color=ACC_COLOR)
    ax2.set_ylim(0, max(0.3, table["alignment"].max() * 1.25))
    ax2.set_ylabel("Centered alignment with labels")
    ax2.set_title("Kernel-Target Alignment (higher = better)", fontsize=10)

    for ax in (ax1, ax2):
        _mark_current(ax, current_s)
        _log_bandwidth_axis(ax, s)
    fig.tight_layout()
    return fig
