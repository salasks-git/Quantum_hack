"""Tab 6: IBM Quantum Hardware Execution Time & QPU Profiler."""
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from qiskit import QuantumCircuit
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
        "and optionally dispatch a live patient fidelity test to real hardware using your IBM Quantum API key."
    )

    # --- 1. PHYSICAL CIRCUIT TRANSPILATION & GATE COUNTING ---
    with st.spinner("Transpiling quantum circuits to IBM Quantum native basis gates (`rz`, `sx`, `x`, `ecr`)..."):
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

    # --- LIVE IBM QUANTUM CONNECTIVITY & REAL QPU DISPATCH ---
    st.subheader("🔑 Live IBM Quantum Cloud Execution (Real Hardware Test)")
    st.markdown(
        "To test actual hardware execution on IBM Quantum computers, enter your API token from "
        "[quantum.ibm.com/account](https://quantum.ibm.com/account). "
        "We solve the queue problem by dispatching a **targeted single-job test** (using $<1$ second of your free quota) "
        "to measure real QPU latency and noise impact."
    )

    from qmed.config import QISKIT_IBM_TOKEN
    default_token = QISKIT_IBM_TOKEN if QISKIT_IBM_TOKEN and "your_ibm_quantum" not in QISKIT_IBM_TOKEN else ""
    api_token = st.text_input(
        "IBM Quantum API Token:",
        value=default_token,
        type="password",
        help="Preloaded from .env (or enter from quantum.ibm.com/account)",
        key="ibm_token_input",
    )


    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        check_backends = st.button("📡 Query Active IBM Quantum Backends & Queues")
    with col_btn2:
        run_real_qpu = st.button("🚀 Run Live Patient Test on Least-Busy IBM QPU")

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

    if run_real_qpu:
        if not api_token.strip():
            st.warning("Please enter your IBM Quantum API token above first.")
        else:
            with st.status("Executing live diagnostic test on IBM Quantum Hardware...", expanded=True) as status_box:
                try:
                    st.write("1️⃣ Authenticating with IBM Quantum Runtime Service...")
                    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
                    service = QiskitRuntimeService(channel="ibm_quantum", token=api_token.strip())

                    st.write("2️⃣ Finding least-busy operational quantum processor...")
                    candidate_backends = [b for b in service.backends() if b.status().operational and not b.simulator]
                    if not candidate_backends:
                        candidate_backends = service.backends()
                    target_backend = min(candidate_backends, key=lambda b: b.status().pending_jobs)
                    st.write(f"🎯 Target QPU Selected: **{target_backend.name}** ({target_backend.num_qubits} qubits, {target_backend.status().pending_jobs} jobs in queue)")

                    st.write("3️⃣ Synthesizing and transpiling patient diagnostic circuit...")
                    x_0 = data.X_test_q[0]
                    x_1 = data.X_test_q[1] if len(data.X_test_q) > 1 else data.X_test_q[0]

                    qc = QuantumCircuit(cfg.n_qubits)
                    qc.compose(feature_map.assign_parameters(x_0), inplace=True)
                    qc.compose(feature_map.assign_parameters(x_1).inverse(), inplace=True)
                    qc.measure_all()

                    transpiled_qc = transpile(qc, backend=target_backend, optimization_level=2)

                    st.write(f"4️⃣ Submitting job to {target_backend.name} with 1,024 shots...")
                    sampler = SamplerV2(mode=target_backend)
                    t_submit = time.time()
                    job = sampler.run([transpiled_qc], shots=1024)
                    job_id = job.job_id()
                    st.info(f"📋 **IBM Quantum Job ID:** `{job_id}`  \nTrack in real-time on: [quantum.ibm.com/jobs](https://quantum.ibm.com/jobs)")

                    st.write("⏳ Waiting for IBM queue and QPU physical execution...")
                    result = job.result()
                    t_complete = time.time() - t_submit

                    st.write("5️⃣ Processing raw quantum bitstring measurements...")
                    # Extract counts from SamplerV2 DataBin
                    pub_result = result[0]
                    # c is typically the BitArray containing measurement outcomes
                    bit_data = getattr(pub_result.data, "meas", getattr(pub_result.data, "c", None))
                    counts = bit_data.get_counts() if bit_data is not None else {}

                    zero_key = "0" * cfg.n_qubits
                    shots_measured = sum(counts.values()) if counts else 1024
                    hw_fidelity = counts.get(zero_key, 0) / shots_measured

                    # Ideal simulated fidelity for comparison
                    from qiskit.quantum_info import Statevector
                    sv_0 = Statevector(feature_map.assign_parameters(x_0))
                    sv_1 = Statevector(feature_map.assign_parameters(x_1))
                    ideal_fidelity = abs(sv_0.inner(sv_1)) ** 2
                    noise_gap = ideal_fidelity - hw_fidelity

                    status_box.update(label="✅ Real Quantum Hardware Execution Completed!", state="complete")

                    st.success(f"🎉 Job `{job_id}` successfully executed on physical processor `{target_backend.name}`!")

                    r1, r2, r3 = st.columns(3)
                    with r1:
                        st.metric("Total Cloud Turnaround", f"{t_complete:.1f} s", help="Queue wait + network transfer + QPU execution")
                    with r2:
                        st.metric("Real QPU Hardware Fidelity", f"{hw_fidelity:.3f}", delta=f"{hw_fidelity*100:.1f}% state overlap")
                    with r3:
                        st.metric("Quantum Noise Impact (ΔF)", f"{noise_gap:.3f}", delta="Fidelity reduction from decoherence/gate error", delta_color="inverse")

                    with st.expander("🔬 View Measured Physical Bitstring Histogram"):
                        st.write(counts)
                except Exception as e:
                    status_box.update(label=f"❌ Execution failed: {e}", state="error")
                    st.error(f"Error during hardware execution: {e}")
