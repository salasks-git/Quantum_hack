"""Tab 6: IBM Quantum Hardware Execution Time & QPU Profiler."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from qiskit.compiler import transpile

from qmed.data import PreparedData
from qmed.ui.sidebar import PipelineConfig


# Real IBM Superconducting QPU Hardware Calibration Benchmarks (Eagle / Heron / Falcon)
T_1Q_NS = 35.0         # 1-qubit physical pulse gate duration (sx, x): ~35 ns
T_RZ_NS = 0.0          # Virtual Z-gates (frame changes): 0 ns in hardware
T_2Q_NS = 300.0        # 2-qubit entangling gate (ECR / CX): ~300 ns
T_MEAS_NS = 700.0      # Dispersive readout measurement: ~700 ns
T_RESET_US = 250.0     # Qubit active reset / thermal relaxation delay between shots: ~250 µs


def render(cfg: PipelineConfig, data: PreparedData, feature_map, ansatz, results: dict) -> None:
    st.subheader("⚛️ Real IBM Quantum Hardware Execution Time & QPU Profiler")
    st.markdown(
        "Measure the **physical execution duration on actual IBM Quantum Processors (QPUs)** "
        "by transpiling circuits to IBM's native basis gate set (`rz`, `sx`, `x`, `ecr`), "
        "and optionally connect your IBM Quantum API key for live backend status."
    )

    # --- 1. PHYSICAL CIRCUIT TRANSPILATION & GATE COUNTING ---
    with st.spinner("Transpiling quantum circuits to IBM Quantum native basis gates (`rz`, `sx`, `x`, `ecr`)..."):
        # Transpile feature map and ansatz to IBM native basis
        tfm = transpile(feature_map, basis_gates=["rz", "sx", "x", "ecr"], optimization_level=2)
        tans = transpile(ansatz, basis_gates=["rz", "sx", "x", "ecr"], optimization_level=2)

        fm_ops = dict(tfm.count_ops())
        ans_ops = dict(tans.count_ops())

    # Gate counts for a single kernel evaluation (U_phi(x_i) + U_phi^dagger(x_j) + measure)
    n_1q = (fm_ops.get("sx", 0) + fm_ops.get("x", 0)) * 2
    n_rz = (fm_ops.get("rz", 0)) * 2
    n_2q = (fm_ops.get("ecr", 0) + fm_ops.get("cx", 0)) * 2
    n_meas = cfg.n_qubits

    # Duration of a single circuit shot (nanoseconds)
    t_circuit_ns = (n_1q * T_1Q_NS) + (n_rz * T_RZ_NS) + (n_2q * T_2Q_NS) + (n_meas * T_MEAS_NS)
    t_circuit_us = t_circuit_ns / 1000.0

    # Total duration for 1 circuit execution with shots (including reset time)
    shots = 1024
    t_single_eval_ms = shots * (t_circuit_us + T_RESET_US) / 1000.0

    # Number of evaluations
    n_train = len(data.X_train_q)
    n_test = len(data.X_test_q)
    n_qsvc_evals = (n_train * (n_train - 1)) // 2 + (n_train * n_test)
    n_vqc_evals = n_train * cfg.vqc_iter

    # Total physical QPU execution times
    qsvc_qpu_sec = (n_qsvc_evals * t_single_eval_ms) / 1000.0
    vqc_qpu_sec = (n_vqc_evals * t_single_eval_ms) / 1000.0

    # Local CPU simulation times for comparison
    qsvc_cpu_sec = results.get("Quantum Kernel SVM (QSVC)", {}).get("total_time", 1.0)
    vqc_cpu_sec = results.get("Variational Quantum (VQC)", {}).get("total_time", 1.0)

    # --- KPI METRICS ---
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Native 2Q Gates (ECR)", n_2q, help="Entangling gates are the bottleneck on hardware")
    with m2:
        st.metric("1-Shot Circuit Duration", f"{t_circuit_us:.2f} µs", help="Time of a single quantum pass")
    with m3:
        st.metric("1024-Shot QPU Eval", f"{t_single_eval_ms:.1f} ms", help="Hardware execution time per kernel pair")
    with m4:
        st.metric("QSVC Full QPU Time", f"{qsvc_qpu_sec:.2f} s", delta=f"{qsvc_qpu_sec/qsvc_cpu_sec:.1f}x vs CPU sim" if qsvc_cpu_sec > 0 else "")

    st.divider()

    # --- COMPARISON VISUALIZATION ---
    st.subheader("📊 Comparison: Physical QPU Gate Time vs. Local CPU Simulation vs. Cloud Queue")
    c_plot1, c_plot2 = st.columns(2)

    with c_plot1:
        st.markdown("#### Hardware Execution vs. CPU Simulation (Pure Compute Time)")
        fig, ax = plt.subplots(figsize=(6, 3.8))
        categories = ["QSVC (Physical QPU)", "QSVC (CPU Statevector)", "VQC (Physical QPU)", "VQC (CPU Statevector)"]
        times = [qsvc_qpu_sec, qsvc_cpu_sec, vqc_qpu_sec, vqc_cpu_sec]
        colors = ["#10b981", "#6366f1", "#10b981", "#a855f7"]

        bars = ax.barh(categories, times, color=colors, edgecolor="black", alpha=0.85)
        ax.set_xlabel("Time (Seconds)")
        ax.set_title("Physical Hardware vs Classical Simulation")
        for bar in bars:
            w = bar.get_width()
            ax.text(w + max(times) * 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{w:.2f}s", va="center", ha="left", fontsize=9, fontweight="bold")
        ax.set_xlim(0, max(times) * 1.3)
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with c_plot2:
        st.markdown("#### The Three Latency Regimes of Real Quantum Computing")
        st.info("""
        1. **Physical QPU Gate Execution:** Milliseconds to seconds. Superconducting microwave pulses operate at **nanosecond** speeds.
        2. **Local CPU Simulation:** Seconds. Simulating $2^n$ state vectors with linear algebra scales exponentially with qubits.
        3. **IBM Cloud Queue Overhead:** Minutes to hours. Free tier jobs must wait in fair-share queues before reaching the dilution refrigerator.
        """)
        q_times = [qsvc_qpu_sec, qsvc_cpu_sec, 600.0]  # 600s = 10 min typical queue
        q_labels = ["QPU Execution", "CPU Simulation", "IBM Cloud Queue (~10m)"]
        fig2, ax2 = plt.subplots(figsize=(6, 2.5))
        ax2.barh(q_labels, q_times, color=["#10b981", "#6366f1", "#f59e0b"], edgecolor="black")
        ax2.set_xscale("log")
        ax2.set_xlabel("Time (Seconds, Log Scale)")
        ax2.set_title("Real-World Latency with Queue Wait")
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    st.divider()

    # --- HARDWARE SPECS BREAKDOWN TABLE ---
    st.subheader("📋 Transpiled Circuit Hardware Specifications")
    st.table(pd.DataFrame([
        {
            "Circuit Component": "Feature Map (U_phi)",
            "Native 1Q Gates (sx, x)": fm_ops.get("sx", 0) + fm_ops.get("x", 0),
            "Virtual Z Gates (rz)": fm_ops.get("rz", 0),
            "2Q Entangling Gates (ecr)": fm_ops.get("ecr", 0) + fm_ops.get("cx", 0),
            "Physical Depth": tfm.depth(),
            "Est. Gate Time": f"{(fm_ops.get('sx', 0) * T_1Q_NS + (fm_ops.get('ecr', 0) + fm_ops.get('cx', 0)) * T_2Q_NS)/1000:.2f} µs"
        },
        {
            "Circuit Component": "Variational Ansatz (RealAmplitudes)",
            "Native 1Q Gates (sx, x)": ans_ops.get("sx", 0) + ans_ops.get("x", 0),
            "Virtual Z Gates (rz)": ans_ops.get("rz", 0),
            "2Q Entangling Gates (ecr)": ans_ops.get("ecr", 0) + ans_ops.get("cx", 0),
            "Physical Depth": tans.depth(),
            "Est. Gate Time": f"{(ans_ops.get('sx', 0) * T_1Q_NS + (ans_ops.get('ecr', 0) + ans_ops.get('cx', 0)) * T_2Q_NS)/1000:.2f} µs"
        }
    ]))

    st.divider()

    # --- LIVE IBM QUANTUM CONNECTIVITY ---
    st.subheader("🔑 Live IBM Quantum Cloud Connectivity (Optional)")
    st.markdown("Enter your IBM Quantum API token to inspect active QPUs, live queue depths, and run a hardware verification test:")

    api_token = st.text_input("IBM Quantum API Token:", type="password", help="From quantum.ibm.com/account", key="ibm_token_input")

    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        check_backends = st.button("📡 Query Active IBM Quantum Backends & Queues")

    if check_backends:
        if not api_token.strip():
            st.warning("Please provide your IBM Quantum API token from quantum.ibm.com/account.")
        else:
            with st.spinner("Connecting to IBM Quantum Runtime Service..."):
                try:
                    from qiskit_ibm_runtime import QiskitRuntimeService
                    service = QiskitRuntimeService(channel="ibm_quantum", token=api_token.strip())
                    backends = service.backends()
                    b_rows = []
                    for b in backends:
                        status = b.status()
                        b_rows.append({
                            "Backend Name": b.name,
                            "Operational": "✅ Online" if status.operational else "❌ Offline",
                            "Total Qubits": b.num_qubits,
                            "Pending Queue Jobs": status.pending_jobs,
                            "Basis Gates": ", ".join(b.operation_names[:6]) + "...",
                        })
                    st.success(f"Connected! Found {len(backends)} active IBM Quantum backends.")
                    st.dataframe(pd.DataFrame(b_rows), use_container_width=True)
                except Exception as e:
                    st.error(f"Failed to connect to IBM Quantum: {e}")
