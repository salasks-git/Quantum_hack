# Tour v2: Streamlit Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Bring the latest Streamlit additions into the 3D tour:
- (A) Hybrid Ensemble column in the finale.
- (B) Clinical scoreboard (F2, missed diagnoses, leader) in the finale.
- (C) New scene 6 "From simulator to real QPU", with transpiled native gates and a QPU vs CPU vs queue time race.
- (D) Optional live IBM QPU run from scene 6 via a local helper server.

**Architecture:** The export gains `hybrid` predictions, per-model timings and a `qpu` profile, computed by a new Streamlit-free `qmed/hardware.py`. A stdlib local server `scripts/qpu_server.py` (127.0.0.1:8765) does the live IBM submission; the browser reaches it through a Vite `/api` proxy and never sees the token. The finale and a new scene 6 consume the new data.

**Tech Stack:** Python venv (qiskit 1.x/2.x <2.5, qiskit-ibm-runtime, qiskit-machine-learning 0.9.1, pytest), Vite + three@0.160 + vitest.

**Spec:** `docs/superpowers/specs/2026-11-01-threejs-diagnosis-tour-design.md` (base tour). The design for A–D was approved in chat on 2026-11-01:
- A: the RBF and QSVC columns visibly pour into a Hybrid column, 0.5 + 0.5.
- B: same F2 formula as Streamlit (beta=2, zero_division=0), plus missed diagnoses (FN) over the 25 test patients, plus a "clinical leader" badge.
- C: gates flow along 3 qubit wires as 3D blocks with real counts and depth, then a 3-way race of QPU estimate vs actual CPU time vs ~10 min queue. Captions label the chip times as estimates from typical IBM calibration.
- D: a "Run on real IBM QPU" button runs the chosen patient against their most-similar training patient from scene 3. The token stays server-side in `.env`. States: queued → running → done, then hardware vs ideal fidelity and the noise gap. It degrades gracefully when the server, token or network is missing. A slow queue never blocks the tour.

## Global Constraints

- Every number shown comes from the exported JSON or the live server's response; nothing invented. QPU times are labelled "estimate (typical IBM calibration)".
- Calibration constants are copied verbatim from `qmed/ui/tab_ibm_hardware.py`: `T_1Q_NS = 35.0`, `T_RZ_NS = 0.0`, `T_2Q_NS = 300.0`, `T_MEAS_NS = 700.0`, `T_RESET_US = 250.0`, shots `1024`, queue reference `600.0` s. Gate counting is the same as that tab: transpile with `basis_gates=["rz","sx","x","ecr"]`, `optimization_level=2`; per kernel evaluation, counts ×2 (U and U†); `n_qsvc_evals = n_train*(n_train-1)//2 + n_train*n_test`; `n_vqc_evals = n_train*vqc_iter`.
- Do not modify `app.py` or `qmed/ui/*` (teammates are actively changing them; a bandwidth feature is mid-merge on main). New Python logic goes in new files; `qmed/models.py` and `qmed/data.py` are untouched.
- **IBM token:** read only server-side from `qmed.config.QISKIT_IBM_TOKEN` (`.env`). It never appears in any HTTP response, log line, JSON file or the browser. The server binds `127.0.0.1` only.
- The tour must keep working fully offline: if the QPU server is down, the token is missing or the network is down, scene 6 shows a clear inline message and every other feature works.
- Tour settings are unchanged: 3 qubits, ZZFeatureMap, reps 2, linear, 100 patients (75/25), VQC 35, seed 42.
- Model keys in the web app: `linear, rbf, qsvc, vqc, hybrid`. Labels: `Linear SVM, RBF SVM, QSVC, VQC, Hybrid (QSVC+RBF)`.
- Python is the shared interpreter `/Users/sabinsalamp/Documents/Quantum_hack/venv/bin/python`, run from the worktree root. Work only in the worktree `/Users/sabinsalamp/Documents/Quantum_hack-tour2` (branch `feat/tour-v2`).

## Review Focus

1. **Token leakage:** no endpoint, error message or log echoes the token, including in exception text. Test: a fake token string never appears in any response body (Task 13).
2. **A QPU job that takes minutes or hours:** the browser polls without blocking; navigating away or switching patient doesn't break anything; a stale job's result isn't shown for a different patient (Task 14).
3. **Server not running** (proxy 502/ECONNREFUSED) or **no token:** the scene 6 panel shows a specific message, not a spinner forever (Tasks 13, 14).
4. **Hybrid probability exactly 0.5:** the verdict follows `pred` (`prob >= 0.5` → 1), consistent with Streamlit (Task 11 test).
5. **Scoreboard math:** F2 with zero true positives/predicted positives returns 0, not NaN (Task 12 test).

---

### Task 11: Export hybrid, timings, QPU profile

**Files:** Create `qmed/hardware.py`, `tests/test_hardware.py`. Modify `scripts/export_tour_data.py`, `tests/test_export.py`. Regenerate `web/public/data/*.json`.

**Interfaces:**
- `qmed/hardware.py`: the constants above, plus `qpu_profile(feature_map, ansatz, n_qubits: int, n_train: int, n_test: int, vqc_iter: int) -> dict`, which returns `{"basis": ["rz","sx","x","ecr"], "fm_ops": {name:int}, "ans_ops": {name:int}, "fm_depth": int, "ans_depth": int, "n_1q": int, "n_rz": int, "n_2q": int, "n_meas": int, "t_circuit_us": float, "t_eval_ms": float, "n_qsvc_evals": int, "n_vqc_evals": int, "qsvc_qpu_s": float, "vqc_qpu_s": float, "queue_s": 600.0, "shots": 1024, "gates": [{"name": str, "qubits": [int,...]}]}`. Here `gates` is the transpiled feature map's instruction sequence in order, excluding barriers, for the wires animation.
- Export additions:
  - `test[i].models.hybrid = {pred, prob}` from `results[HYBRID_ENSEMBLE]`.
  - Top-level `timings = {linear|rbf|qsvc|vqc|hybrid: {train_s, inf_s, total_s}}`, the CPU times measured during export.
  - Top-level `qpu = qpu_profile(...)`.
- [ ] **Step 1: Tests.**
  - `test_hardware.py`: for a 3-qubit ZZ reps=2 linear map and a RealAmplitudes reps=2 ansatz, `n_2q == 2 * (fm_ops.get("ecr",0) + fm_ops.get("cx",0))`; `t_eval_ms == pytest.approx(1024 * (t_circuit_us + 250.0) / 1000)`; `n_qsvc_evals == 75*74//2 + 75*25`; `len(gates) > 0` and every gate name is in the basis set ∪ `{"measure"}`.
  - Extend `test_export.py`: every test patient has `models.hybrid` with `pred in (0,1)` and `0 <= prob <= 1`, and `pred == int(prob >= 0.5)`; `timings` has all 5 keys with `total_s > 0`; the `qpu` key exists with `qsvc_qpu_s > 0`.
- [ ] **Step 2: Run** the tests. Expected: FAIL.
- [ ] **Step 3: Implement.** Then run `scripts/export_tour_data.py`. Each JSON stays < 1 MB.
- [ ] **Step 4: Run** `python -m pytest tests -q`. Expected: all pass.
- [ ] **Step 5: Commit** `feat: export hybrid ensemble, timings and QPU profile for tour`.

### Task 12: Finale: Hybrid column + clinical scoreboard

**Files:** Create `web/src/clinical.js`, `web/src/clinical.test.js`. Modify `web/src/verdict.js` (`MODEL_KEYS`/`MODEL_LABELS` gain `hybrid`) and `web/src/scenes/scene5.js`.

**Interfaces:** `clinicalScores(test: Patient[], key: string) -> {tp, fp, fn, tn, recall, precision, f2, missed}` with F2 = (1+4)·P·R / (4·P + R), and 0 when the denominator is 0. `clinicalLeader(test, keys) -> key` = max F2, ties broken by fewer `missed`, then key order.
- [ ] **Step 1: vitest tests.**
  - A hand-built 6-patient array gives exact tp/fp/fn/tn and `f2` (`toBeCloseTo`).
  - All-negative predictions give `f2 === 0` and no NaN.
  - The leader tie-break works as specified.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.**
  - Scene 5 shows 5 columns. After the RBF and QSVC columns finish, two streams of particles flow from them into the Hybrid column, which then rises to its real prob. The caption explains "Hybrid = ½·RBF + ½·QSVC".
  - The scoreboard is an HTML overlay, or 3D text, under the columns: per model F2 (3 dp) and missed diagnoses (n of total sick), across all test patients of the disease, plus a "🏆 Clinical leader (F2)" badge on the leader. The patient-specific verdict and borderline flags are unchanged.
  - Update the scene's `stops` so a 3s hold lands after the hybrid column resolves and after the scoreboard appears.
  - Extend `duration` as needed.
- [ ] **Step 4: Run** `npm test` and `bash web/scripts/smoke.sh; echo exit=$?`, and look at the step 5 screenshots for 2 diseases.
- [ ] **Step 5: Commit** `feat: tour finale adds hybrid ensemble and clinical scoreboard`.

### Task 13: Local QPU helper server + Vite proxy

**Files:** Create `scripts/qpu_server.py`, `tests/test_qpu_server.py`, `web/vite.config.js`.

**Interfaces (JSON over HTTP, 127.0.0.1:8765, stdlib `http.server` + `ThreadingHTTPServer`):**
- `GET /api/qpu/health` → `{ok: true, token: bool, runtime: bool}`. `runtime` = `qiskit_ibm_runtime` importable.
- `POST /api/qpu/submit` with body `{slug, patient}` → 200 `{job_id, backend, pending_jobs, partner, ideal_fidelity, submitted_at}`, or 4xx/5xx `{error: "<human message>"}`.
  - `partner` = argmax of `kernel_test[patient]` from `web/public/data/<slug>.json`.
  - The angles come from the same JSON (`test[patient].angles`, `train.angles[partner]`).
  - Circuit = `fm.assign_parameters(x_p)` ∘ `fm.assign_parameters(x_q).inverse()` + `measure_all`, with `fm = build_feature_map("ZZFeatureMap", 3, 2, "linear")`.
  - `ideal_fidelity` comes from `Statevector`.
  - Backend = the least-busy operational non-simulator backend.
  - Transpile with `optimization_level=2`; submit with `SamplerV2(mode=backend).run([qc], shots=1024)`.
  - Service: `QiskitRuntimeService(channel="ibm_quantum_platform", token=...)`. On failure, try `channel="ibm_quantum"`, then the saved default account. Exceptions are reported as messages **with the token scrubbed**.
- `GET /api/qpu/status?job=<id>` → `{job_id, status, elapsed_s}` plus, when done, `{hw_fidelity, counts, noise_gap}`. `hw_fidelity` = count of `"000"` / shots; `noise_gap` = ideal − hw.
- Jobs and services are cached in memory by `job_id`. Submits run in the request thread, but any `job.result()` waiting happens only in status calls via a non-blocking `job.status()` check.
- `web/vite.config.js`: `server.proxy` and `preview.proxy` map `/api` to `http://127.0.0.1:8765`.
- Make the service and backend factory injectable, e.g. a module-level `make_service` function the tests monkeypatch.
- [ ] **Step 1: pytest tests** (no network; fake service, backend and job):
  - health with an empty token → `token: false`;
  - submit with no token → 503, with an error mentioning `.env` and `QISKIT_IBM_TOKEN`;
  - submit with the fake service → 200 and `partner` equal to the argmax from the JSON, `ideal_fidelity` within [0, 1];
  - status → done with `hw_fidelity` computed from the fake counts;
  - a fake token `"SECRET-TOKEN-123"` never appears in any response body, including when the fake service raises an exception containing it;
  - an unknown slug → 404.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** pytest (all pass). Then a manual check: start the server, `curl -s 127.0.0.1:8765/api/qpu/health`, and through `npx vite preview` call `curl -s localhost:4173/api/qpu/health` (the proxy works). Stop both.
- [ ] **Step 5: Commit** `feat: local IBM QPU helper server and /api proxy for tour`.

### Task 14: Scene 6: from simulator to real QPU (+ live run panel)

**Files:** Create `web/src/scenes/scene6.js`, `web/src/qpu.js` (client), `web/src/qpu.test.js`. Modify `web/src/main.js` (step 6 everywhere: dots, Next/Prev bounds, Play all, URL `step` 1..6), `web/index.html`, `web/src/style.css`, and `web/scripts/smoke.sh` (steps 1..6).

**Interfaces:** `qpu.js`:
- `health() -> Promise<{ok, token, runtime} | {ok:false, reason}>`, where `reason` ∈ `"server-down"` (fetch or proxy failure) | `"error"`.
- `submit(slug, patient) -> Promise<SubmitResult>`.
- `pollStatus(jobId, {onUpdate, signal, intervalMs=5000}) -> Promise<final>`. Abortable via `AbortSignal`.

**Scene (duration ~14 s):**
- Part 1: 3 horizontal qubit wires. The transpiled `qpu.gates` flow left to right as 3D blocks (1Q `sx`/`x` small cubes, `rz` thin discs, `ecr` tall connectors spanning their two wires). The caption gives the real counts (1Q, rz, 2Q ×2 per kernel evaluation), the depth, and 1-shot duration µs.
- Part 2: 3 bars race on a log axis: "QSVC on IBM QPU (estimate)" `qsvc_qpu_s`, "QSVC on this CPU (measured)" `timings.qsvc.total_s`, "IBM cloud queue (typical)" `queue_s`. The same for VQC is optional. A caption gives the ratios.
- `stops` after each part.

**Live panel** (HTML overlay, bottom-right, visible on step 6 only):
- It shows a health check result.
- The button "🚀 Run on real IBM QPU" is enabled only when `token && runtime`. Otherwise it shows a specific message: "Start the helper: `python scripts/qpu_server.py`" / "Add QISKIT_IBM_TOKEN to .env" / "Install qiskit-ibm-runtime".
- On submit it shows the backend, the queue length and the job id, with an animated status (queued → running → done) and elapsed time.
- On done it shows hardware fidelity, ideal fidelity, the noise gap, and scene 3's K for the same pair (`kernel_test[patient][partner]`).
- Switching patient or disease, or leaving step 6, aborts polling for the UI. A result is only shown if its `{slug, patient}` matches the current selection.
- "Play all" never submits jobs.

- [ ] **Step 1: vitest tests** with stubbed fetch:
  - `health` maps a network error → `{ok:false, reason:"server-down"}`;
  - `pollStatus` calls `onUpdate` until done and resolves with the final result;
  - an aborted signal stops polling, with no further fetches.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `npm test` and `bash web/scripts/smoke.sh; echo exit=$?` (now 6 steps). View the step 6 screenshots (part 1 and part 2 moments) and the panel's "server-down" message (smoke runs without the server).
- [ ] **Step 5: Commit** `feat: tour scene 6 — simulator to real IBM QPU with live run panel`.

### Task 15: Docs + verification

- [ ] **Step 1: README updates.**
  - Scene 5 additions and scene 6.
  - The live QPU run: `.env` with `QISKIT_IBM_TOKEN`, then `python scripts/qpu_server.py` alongside `npm run dev`/`preview`; the token stays local; queue times can be long.
  - The `step=6` deep link.
- [ ] **Step 2: Verify:**
  - `python -m pytest tests -q`
  - `cd web && npm test`
  - `bash web/scripts/smoke.sh; echo exit=$?`
  - `npm run build` and `grep` dist for the absence of `QISKIT_IBM_TOKEN` values and of `http(s)://` CDN URLs.
- [ ] **Step 3: Commit** `docs: README for tour v2 (hybrid, scoreboard, QPU scene, live run)`.
