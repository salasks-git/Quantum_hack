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
    fig, ax = plt.subplots(figsize=(6, 3.8))
    bars = ax.barh(names, totals, color=["#2563eb", "#0284c7", "#7c3aed", "#d946ef"], edgecolor="black", alpha=0.85)
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
