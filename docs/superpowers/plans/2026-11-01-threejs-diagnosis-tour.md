# Q-Med 3D Diagnosis Tour Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A standalone Three.js page that walks one chosen patient of a chosen disease through 5 animated 3D scenes (encoding, classical, QSVC, VQC, finale), using real exported model outputs. Streamlit becomes repeatable too.

**Architecture:** `scripts/export_tour_data.py` trains all four models once per disease, using the existing `qmed` pipeline with fixed tour settings, and writes static JSON to `web/public/data/`. `web/` is a Vite + vanilla Three.js app with one shared renderer. A scene manager swaps between 5 scene modules that share one interface, driven by a top bar (disease picker, patient picker, step nav). A URL deep link (`?disease=&patient=&step=`) enables rehearsal and headless smoke tests.

**Tech Stack:** Python 3.13 venv (qiskit-machine-learning 0.9.1, scikit-learn, pytest), Node 20 + Vite 5 + three@0.160 + vitest.

**Spec:** `docs/superpowers/specs/2026-11-01-threejs-diagnosis-tour-design.md`

## Global Constraints

- Do **not** modify `app.py`, `qmed/ui/*`, `qmed/metrics.py` (teammates are editing them). Changes to `qmed/models.py` and `qmed/data.py` must be backward-compatible (new optional params/fields only).
- Tour settings, fixed: 3 qubits, `ZZFeatureMap`, reps=2, entanglement `linear`, `max_samples=100` (gives 75 train / 25 test), VQC `maxiter=35`, seed `RANDOM_STATE` (42).
- Every number shown in the page comes from the exported JSON. No invented values.
- Verdict, colour and right/wrong follow `pred`; `prob` is shown as confidence. If `(prob > 0.5) != (pred == 1)`, the caption says "borderline".
- Bloch-sphere captions state they show the **first encoding layer** only.
- No runtime CDN or network access: `three` is installed via npm and bundled. Data is loaded with `fetch()` from the same origin. Run via `npm run dev` or `npm run preview`.
- Palette: background `#070b16`, quantum `#a78bfa`/`#f0abfc`, classical `#38bdf8`, healthy `#4ade80`, sick `#fb7185` (as in the approved samples).
- Priority if time runs out: Tasks 1–6 give a complete demo (scenes 1, 3, 5). Tasks 7–8 (scenes 2, 4) come after.

## Review Focus

1. **30-feature disease (breast cancer):** scene 1 must stay legible at 30 bars (labels may abbreviate or show on hover), not overlap into noise. Smoke test: screenshot of `?disease=breast_cancer_wisconsin&step=1` is reviewed by eye (Task 4).
2. **Test patients whose angles fall outside `[0, π]`** (the test set is transformed by the training-fitted scaler). Positions, arrows and the RBF grid must not clip or crash. The export grid spans `[-0.3, π + 0.3]` per axis (Task 2 test), and scenes plot true values.
3. **`prob` and `pred` disagree:** the finale must not show a red column with a "healthy" verdict unexplained (Task 3 `verdictFor` tests).
4. **Fast switching of disease, patient or step mid-animation:** no duplicated scenes, leaked geometries or stacked render loops. The manager disposes the old scene before building the new one (Task 3 test: `SceneManager` calls `dispose` exactly once per switch).
5. **Missing or unreadable data file** (export not run, wrong slug in URL): the page shows a visible error banner naming the slug, not a blank canvas (Task 3 test on `loadDisease`).

---

### Task 1: Repeatable training + VQC weight history

**Files:**
- Modify: `qmed/models.py` (`train_all_models`)
- Modify: `README.md` ("Known limitations" bullet about quantum scores varying)
- Modify: `requirements.txt` (add `pytest>=8.0`)
- Create: `tests/__init__.py` (empty), `tests/test_models.py`

**Interfaces:**
- Produces: `train_all_models(X_train_q, y_train, X_test_q, feature_map, ansatz, vqc_iter: int, seed: int | None = RANDOM_STATE) -> dict`. VQC result gains `"weights_history": list[list[float]]` (one entry per callback, same length as `loss_history`, each of length `ansatz.num_parameters`).

- [ ] **Step 1: Write failing tests** in `tests/test_models.py`, using a small fixture: `prepare_data(DATASETS["Statlog Heart Disease (Cardiology)"], 60, 2)`, ZZ map and ansatz with reps=1, linear, `vqc_iter=5`.
  - `test_seeded_runs_are_identical`: two calls with the default seed → `np.allclose` on QSVC `prob`, VQC `prob`, and VQC `loss_history`.
  - `test_weights_history_matches_loss_history`: `len(weights_history) == len(loss_history) > 0`, and every entry has length `ansatz.num_parameters`.
  - `test_seed_none_still_runs`: `seed=None` returns all 4 model keys.
- [ ] **Step 2: Run** `venv/bin/pip install pytest && venv/bin/python -m pytest tests/test_models.py -v`. Expected: FAIL (`weights_history` KeyError / unexpected kwarg `seed`).
- [ ] **Step 3: Implement.** `StatevectorSampler(seed=seed)`. VQC `initial_point = np.random.default_rng(seed).uniform(-np.pi, np.pi, ansatz.num_parameters)` when `seed is not None`, else `None`. Callback appends `list(map(float, weights))` and `float(loss)`.
- [ ] **Step 4: Run tests.** Expected: 3 passed.
- [ ] **Step 5: Update the README limitation** to say QSVC/VQC are seeded (`seed=42`) and repeatable, and that Streamlit at 3 qubits / 100 patients / ZZ / reps 2 / linear / 35 iterations matches the tour.
- [ ] **Step 6: Commit** `feat: seed quantum models for repeatable results; record VQC weights`.

### Task 2: Tour data export

**Files:**
- Modify: `qmed/data.py`: add optional fields to `PreparedData` (defaults `None`, appended after existing fields): `X_test_raw`, `X_test_std`, `X_test_pca` (np.ndarray). `prepare_data` fills them.
- Create: `scripts/export_tour_data.py`, `tests/test_export.py`
- Create (generated, committed): `web/public/data/index.json`, `web/public/data/<slug>.json` ×5

**Interfaces:**
- Consumes: `prepare_data`, `build_feature_map`, `build_ansatz`, `train_all_models(..., seed=RANDOM_STATE)` (Task 1).
- Produces: `export_dataset(label: str, csv_path: Path, grid_size: int = 24) -> dict` and `main(out_dir: Path = PROJECT_ROOT / "web/public/data") -> None`. `slug = csv_path.stem` (e.g. `heart_disease_statlog`). The JSON schema (floats rounded to 4 dp, ints for labels/preds):

```text
{ label, slug, feature_names[F], feature_ranges: {min[F], max[F]},
  settings: {n_qubits:3, feature_map:"ZZFeatureMap", reps:2, entanglement:"linear", vqc_iter:35, seed:42},
  train: {angles[N][3], labels[N]},
  test: [{index, label, raw[F], standardized[F], pca[3], angles[3],
          models: {linear:{pred,prob}, rbf:{pred,prob}, qsvc:{pred,prob}, vqc:{pred,prob}}}],
  linear: {coef[3], intercept},
  rbf: {grid_size:G, min:-0.3, max:π+0.3, values[G*G*G]},   // index = (i*G + j)*G + k for (x_i, y_j, z_k)
  qsvc: {kernel_train[N][N], kernel_test[T][N]},
  vqc: {loss_history[E], weights_history[E][9]} }
index.json: [{slug, label, n_features}]
```

- [ ] **Step 1: Write failing tests** in `tests/test_export.py`. Use one module-scoped fixture `export_dataset("Vertebral Column (Orthopedics)", DATASETS[...], grid_size=8)` (the smallest dataset):
  - `test_shapes`: `len(train.angles) == 75`, each length 3; `len(test) == 25`; `kernel_train` is 75×75; `kernel_test` is 25×75; `len(rbf.values) == 8**3`; `len(feature_names) == 6 == len(test[0].raw)`.
  - `test_grid_bounds`: `rbf.min == -0.3` and `rbf.max == pytest.approx(np.pi + 0.3)`.
  - `test_preds_and_probs_valid`: for every test patient and every model in `linear, rbf, qsvc, vqc`, `pred in (0, 1)` and `0 <= prob <= 1`.
  - `test_kernel_diagonal_near_one`: `kernel_train[i][i] > 0.95` for all i (fidelity of a state with itself; sampled with shots).
  - `test_json_serialisable`: `json.dumps(result)` succeeds.
  - `test_prepare_data_backward_compatible`: `prepare_data(...)` result still has `X_train_q, X_test_q, y_train, y_test, pca, prep_time`.
- [ ] **Step 2: Run** `venv/bin/python -m pytest tests/test_export.py -v`. Expected: FAIL (module not found).
- [ ] **Step 3: Implement.** Linear plane from the Linear SVM model's `coef_[0]` / `intercept_[0]`. RBF grid from `decision_function` on a `np.linspace(min, max, G)` meshgrid (`indexing="ij"`, flattened C-order to match the index formula). `kernel_test` via `q_kernel.evaluate(x_vec=X_test_q, y_vec=X_train_q)`, where the kernel comes from `results[QSVC_MODEL]["model"].quantum_kernel`. Labels are the `y_test` order, so the patient index matches Streamlit's simulator. `main()` loops over `DATASETS`, writes one file per slug plus `index.json`, and prints each written path with its file size.
- [ ] **Step 4: Run tests.** Expected: 6 passed.
- [ ] **Step 5: Run the full export** `venv/bin/python scripts/export_tour_data.py`. Expected: 6 files written, each disease file < 1 MB.
- [ ] **Step 6: Commit** `feat: export per-disease tour data for 3D diagnosis tour` (including the generated JSON).

### Task 3: Web app shell, data layer, verdict logic, scene manager

**Files:**
- Create: `web/` via `npm create vite@5 web -- --template vanilla`, then remove the template's counter/demo files. `npm i three@0.160.0`, `npm i -D vitest@1`. Add `"test": "vitest run"` to `package.json`.
- Create: `web/index.html` (top bar: disease `<select id="disease">`, patient `<select id="patient">`, step dots 1–5, Prev, Next, Replay, Play all; `<div id="stage">`; `<div id="caption">`; `<div id="error" hidden>`)
- Create: `web/src/main.js`, `web/src/style.css`, `web/src/data.js`, `web/src/verdict.js`, `web/src/sceneManager.js`, `web/src/scenes/common.js`, `web/src/scenes/placeholder.js`
- Test: `web/src/verdict.test.js`, `web/src/data.test.js`, `web/src/sceneManager.test.js`
- Create: `web/scripts/smoke.sh` (headless Chrome smoke)

**Interfaces:**
- `data.js`: `loadIndex(): Promise<{slug,label,n_features}[]>`, `loadDisease(slug): Promise<DiseaseData>`. Throws `Error("Could not load data for '<slug>'")` on a non-OK response.
- `verdict.js`: `verdictFor({pred, prob}) -> {disease: boolean, borderline: boolean}` (`disease = pred === 1`, `borderline = (prob > 0.5) !== disease`). `MODEL_KEYS = ["linear","rbf","qsvc","vqc"]`, `MODEL_LABELS = {linear:"Linear SVM", rbf:"RBF SVM", qsvc:"QSVC", vqc:"VQC"}`.
- Scene interface (every `scenes/sceneN.js` default-exports a factory): `createScene(ctx) -> { title: string, scene: THREE.Scene, cameraPos: [x,y,z], update(elapsedSec): string /* caption HTML */, dispose(): void }`, where `ctx = { data: DiseaseData, patient: number }`.
- `sceneManager.js`: `class SceneManager { constructor(renderer, camera, controls); show(factory, ctx): void /* disposes current, builds new, resets clock and camera */; tick(): string /* renders, returns caption */ }`.
- `common.js`: `PALETTE`, `ease(t)`, `seg(p,a,b)`, `blochSphere(radius) -> THREE.Group` with `userData.set(theta, phi)`, `firstLayerState(angle) -> {theta: π/2, phi: 2*angle}` (H then P(2x)), `disposeTree(object3d)`, `addLights(scene)`.
- `main.js` URL params: `?disease=<slug>&patient=<int>&step=<1..5>`. Missing → first disease, patient 0, step 1. The step→factory table uses `placeholder.js` for steps not yet built.

- [ ] **Step 1: Write failing vitest tests.**
  - `verdict.test.js`: `{pred:1,prob:0.8}` → `{disease:true,borderline:false}`; `{pred:0,prob:0.62}` → `{disease:false,borderline:true}`; `{pred:1,prob:0.4}` → `{disease:true,borderline:true}`; `{pred:0,prob:0.1}` → `{disease:false,borderline:false}`.
  - `data.test.js` (stub `globalThis.fetch`): an OK response returns the parsed JSON; a 404 for slug `nope` rejects with message `Could not load data for 'nope'`.
  - `sceneManager.test.js` (stub renderer/camera/controls with no-op methods; fake factories record calls): `show` twice → the first scene's `dispose` is called exactly once and the second's not at all; `tick()` returns the current scene's `update` string.
- [ ] **Step 2: Run** `cd web && npm test`. Expected: FAIL (modules missing).
- [ ] **Step 3: Implement** the modules above. Use one `WebGLRenderer` and `OrbitControls` (`enableZoom: false`, gentle `autoRotate`) in `main.js`. Disease/patient change → `show(currentStepFactory, newCtx)` (keep the step, replay). "Play all" auto-advances when a scene's `update` has run past its duration (export `duration` seconds from each scene; placeholder 3 s). Patient options read `Test patient #i — actual: Disease|Healthy`. Errors from `loadDisease` show in `#error`.
- [ ] **Step 4: Run** `npm test`. Expected: all pass.
- [ ] **Step 5: Write `web/scripts/smoke.sh`.** It runs `npm run build && npx vite preview --port 4317 &`, then for each slug in `public/data/index.json` and step 1–5: headless Chrome (`--headless=new --use-angle=swiftshader --enable-unsafe-swiftshader --enable-logging=stderr --virtual-time-budget=6000 --screenshot=out/<slug>-<step>.png`). It greps stderr for `Uncaught|TypeError|ReferenceError` and exits non-zero on any match. It also loads `?disease=nope` and asserts the screenshot exists (manual look: error banner). The script kills preview at the end.
- [ ] **Step 6: Run** `bash web/scripts/smoke.sh`. Expected: exit 0; 25 screenshots in `web/out/` (add `web/out/` and `web/node_modules/` and `web/dist/` to `.gitignore`).
- [ ] **Step 7: Commit** `feat: 3D tour shell with data loading, verdicts, scene manager`.

### Task 4: Scene 1 — patient becomes qubits

**Files:** Create `web/src/scenes/scene1.js`; register it as step 1 in `main.js`.

**Interfaces:** Consumes `ctx.data.feature_names`, `feature_ranges`, `test[patient].{raw, standardized, pca, angles}`, `common.blochSphere`, `firstLayerState`. Exports `duration = 10`.

- [ ] **Step 1: Implement.** Stages over `duration`:
  - A: bars at raw values, with height `= raw / max(|feature_ranges.max|)` across features (shows the scale differences).
  - B: lerp to standardised heights (`0.5 + 0.35 * standardized`, clamped to ≥ 0.05).
  - C: bars merge into 3 orbs sized by `|pca|`.
  - D: 3 Bloch spheres; each arrow goes |0⟩ → equator → phase `2·angle` (`firstLayerState`).
  
  Captions name the stage, the feature count, and the 3 real angles (2 dp), plus "first encoding layer only; later layers entangle the qubits". With 30 bars, show labels only for the 5 tallest standardised values.
- [ ] **Step 2: Run** `bash web/scripts/smoke.sh`. Expected: exit 0. Open `out/breast_cancer_wisconsin-1.png` and `out/vertebral_column-1.png` and check that the bars are legible and the spheres are visible.
- [ ] **Step 3: Commit** `feat: tour scene 1 — patient encoded as qubits`.

### Task 5: Scene 3 — QSVC quantum similarity

**Files:** Create `web/src/scenes/scene3.js`; register it as step 3.

**Interfaces:** Consumes `qsvc.kernel_test[patient]`, `qsvc.kernel_train`, `train.{angles, labels}`, `test[patient].{angles, models.qsvc}`, `common.*`, `verdictFor`. Exports `duration = 12`.

- [ ] **Step 1: Implement.**
  - Part 1: two Bloch spheres (the chosen patient and the training patient with the max `kernel_test` value), using first-layer states. The caption shows the real `K` (3 dp) and that patient's true label.
  - Part 2: the web. Training dots sit at their angles, coloured by label. Threads join each patient to its top-3 `kernel_train` neighbours (excluding itself), with opacity ∝ K. The chosen patient (white, at its angles) gets threads to its top-5 `kernel_test` neighbours, lit last. The final caption shows the QSVC verdict and prob, plus "borderline" if flagged.
- [ ] **Step 2: Run smoke.** Expected: exit 0; check `out/heart_disease_statlog-3.png` by eye.
- [ ] **Step 3: Commit** `feat: tour scene 3 — QSVC similarity`.

### Task 6: Scene 5 — finale

**Files:** Create `web/src/scenes/scene5.js`; register it as step 5.

**Interfaces:** Consumes `test[patient].{label, models}`, `MODEL_KEYS`, `MODEL_LABELS`, `verdictFor`. Exports `duration = 7`.

- [ ] **Step 1: Implement.** Four columns (classical cyan, quantum violet) rise in sequence to `prob × H` against a translucent 50% sheet. On completion, the colour follows `verdictFor(...).disease` (sick colour if true). The caption lists each model with DISEASE/HEALTHY, prob %, ✓/✗ vs the true label, and "borderline" flags, then "Actual diagnosis: Disease|Healthy".
- [ ] **Step 2: Run smoke.** Expected: exit 0. Find a test patient where any model has `borderline` true (`node -e` over the JSON). Load that patient's step 5 and check that the caption shows "borderline".
- [ ] **Step 3: Commit** `feat: tour scene 5 — four-model finale`.

> **Checkpoint:** the demo is complete with steps 1, 3, 5 (placeholder for 2 and 4). If time is short, skip to Task 9.

### Task 7: Scene 2 — classical doctors

**Files:** Create `web/src/scenes/scene2.js`; register it as step 2.

**Interfaces:** Consumes `train.{angles, labels}`, `test[patient].{angles, models.linear, models.rbf}`, `linear.{coef, intercept}`, `rbf.{grid_size, min, max, values}`. Uses `MarchingCubes` from `three/addons/objects/MarchingCubes.js` (or a point shell of grid cells where the sign changes, if MarchingCubes field mapping proves fiddly). Exports `duration = 11`.

- [ ] **Step 1: Implement.** Dots at angle coordinates, the chosen patient highlighted. Linear plane `coef·x + intercept = 0`: build a large `PlaneGeometry`, orient its normal to `coef`, and position it at `-intercept·coef/|coef|²`. It slides in along its normal, with a caption showing the linear verdict and prob. Then it fades, and the RBF surface (decision value 0 over the grid) grows in, with a caption showing the RBF verdict and prob.
- [ ] **Step 2: Run smoke.** Expected: exit 0; check `out/diabetes_pima-2.png` by eye (the plane and surface are visible and cut through the dot cloud).
- [ ] **Step 3: Commit** `feat: tour scene 2 — classical decision boundaries`.

### Task 8: Scene 4 — VQC learning

**Files:** Create `web/src/scenes/scene4.js`; register it as step 4.

**Interfaces:** Consumes `vqc.{loss_history, weights_history}`, `test[patient].models.vqc`. Exports `duration = 9`.

- [ ] **Step 1: Implement.** Nine knobs (3×3 grid of tori with a pointer); pointer angle = the current `weights_history[e][k]`. The loss tube is drawn progressively over `E` points (any `E ≥ 2`; COBYLA may stop before `maxiter`). The caption reads "evaluation e/E · loss x.xxx". The final caption shows the VQC verdict and prob for the patient.
- [ ] **Step 2: Run smoke.** Expected: exit 0; check `out/parkinsons_vocal-4.png` by eye.
- [ ] **Step 3: Commit** `feat: tour scene 4 — VQC training`.

### Task 9: Docs + final verification

**Files:** Modify `README.md` (add a "🎬 3D Diagnosis Tour" section: what it is, `venv/bin/python scripts/export_tour_data.py`, `cd web && npm install && npm run dev`, demo URL deep-link format, `npm run build && npm run preview` for the venue). Modify the Repository Structure tree to add `web/`, `tests/`, `docs/`.

- [ ] **Step 1: Update the README.**
- [ ] **Step 2: Run** `venv/bin/python -m pytest -v` (expected: all pass), `cd web && npm test` (all pass), `bash web/scripts/smoke.sh` (exit 0).
- [ ] **Step 3: Run the Streamlit headless check:** AppTest run of `app.py` twice with defaults, asserting the QSVC/VQC ROC-AUC strings are identical across the two runs (repeatability) and that there's no exception.
- [ ] **Step 4: Commit** `docs: README section for 3D diagnosis tour`.
