# Q-Med 3D Diagnosis Tour — Design

**Date:** 2026-11-01 (hackathon day) · **Time budget:** ~2 hours to demo-ready

## Intent

The Streamlit app works but feels jumpy (full-page reruns) and can't animate. For the demo, the team wants a fluid, animated, **3D (Three.js)** walkthrough that shows **each step of the pipeline for every technique** (Linear SVM, RBF SVM, QSVC, VQC) for one patient. The goal is to make the quantum part feel alive and understandable to judges.

**Selecting a disease re-drives everything:** every scene shows that disease's real data and real model outputs.

Streamlit stays untouched as the interactive backup and the place for detailed tables and charts.

## Goals

1. A standalone web page with a **disease picker** (all 5 datasets) and a **patient picker** (that disease's held-out test patients, each labelled with their true diagnosis).
2. Five animated 3D scenes, played step by step for the chosen patient:
   1. **Patient becomes qubits:** raw measurements → standardised → compressed (PCA) → encoded as qubit states on Bloch spheres.
   2. **Classical doctors:** training patients as 3D dots; Linear SVM's plane slides in, then RBF SVM's curved boundary.
   3. **QSVC:** the chosen patient's quantum similarity to other patients, then a web of similarity across patients.
   4. **VQC:** trainable circuit weights ("knobs") turning while the real loss curve descends.
   5. **Finale:** four disease-probability columns for the chosen patient vs the 50% line, next to the true diagnosis.
3. **Every number shown is real**, taken from models trained by the existing `qmed` pipeline.
4. **Works offline at the venue:** no CDN or network dependency at demo time.

## Non-goals

- Training models in the browser, or any live Python backend or API server.
- Replacing Streamlit, or reproducing its tables and charts.
- Configurable qubits, feature maps or depth in the tour (fixed settings, below).
- Mobile polish beyond "doesn't break".

## Anti-goals (failure even if it "works")

- **Fake or illustrative numbers presented as real.** If a value isn't from the export, don't show a number.
- **Physically misleading quantum visuals** presented as exact. Where a visual simplifies (for example per-qubit Bloch spheres for an entangled state), the caption says so.
- **Breaking or changing the Streamlit app.** Teammates are actively editing `qmed/ui/*`, `qmed/metrics.py` and `app.py`; the tour must not modify those files.
- **A demo-time failure mode:** network fetches, long loading spinners, or results that differ between rehearsal and demo.

## Constraints

- Node 20.18 / npm 11 available. Python venv has qiskit-machine-learning 0.9.1 and the `qmed` package.
- `qmed/models.py` may be changed **only backward-compatibly** (Streamlit calls `train_all_models(X_train_q, y_train, X_test_q, feature_map=..., ansatz=..., vqc_iter=...)`).
- Visual style reference: the approved samples (dark navy background, violet/pink quantum accents, cyan classical, green healthy, rose sick). See `.superpowers/brainstorm/*/content/threejs-scenes.html` (gitignored, local only).

## Decisions

### Decided with the user
- Three.js, 3D throughout.
- Five scenes as listed above; selecting a disease drives all scenes with that disease's data.
- Streamlit kept as backup.
- Priority if time runs short: **scenes 1, 3, 5 first** (a complete demo on their own), then 2, then 4.

### Claude's calls (user may override)
- **Stack:** Vite + vanilla Three.js (+ OrbitControls from `three/addons`) in a new top-level `web/` folder. No React or Next.js, because there's no routing or SSR need, the scenes are imperative Three.js, and it has fewer moving parts. Dependencies are installed locally and bundled, so it works offline.
- **Data flow:** a Python script `scripts/export_tour_data.py` trains everything once per disease and writes static JSON to `web/public/data/`. The page only loads JSON. Switching disease is instant.
- **Fixed tour settings:** 3 qubits, `ZZFeatureMap`, reps=2, linear entanglement, 100-patient cohort (the existing `prepare_data` split: 75 train / 25 test), VQC 35 COBYLA iterations. With 3 qubits, each patient's three angles are their exact `(x, y, z)` position, so dots and decision boundaries in scene 2 are a true picture.
- **Determinism (user decision):** seed the sampler (`StatevectorSampler(seed=...)`) and VQC's initial point (fixed RNG) via a new `seed` parameter on `train_all_models` with **default `RANDOM_STATE` (42)**. This makes **both** the tour export and Streamlit repeatable. Streamlit set to 3 qubits / 100 patients / ZZ / reps 2 / linear / 35 iterations then shows exactly the tour's numbers. The call signature stays backward-compatible; this file is the only Streamlit-side change.
- **How it's opened at the demo:** served locally (`npm run dev`, or `npm run preview` after `npm run build`), not by double-clicking `dist/index.html`. Under `file://`, browsers block ES modules and `fetch()`, so the page loads JSON with `fetch()` from its own origin.
- **Verdict vs probability:** for each model, the verdict (DISEASE/HEALTHY, colour, right/wrong mark) follows `pred`, the model's actual output and what Streamlit's tables use. `prob` is shown as confidence. SVC probabilities come from a separate calibration step and can disagree with `pred` near the boundary. When that happens, the finale column is coloured by `pred` and the caption flags it as "borderline". Scene 2's RBF surface is the true `decision_function = 0` boundary, which matches `pred`.
- **Navigation:** a top bar with disease picker, patient picker, a 1–5 step indicator, Prev/Next, Replay, and "Play all" (auto-advance). Changing disease or patient keeps the current step and replays it with the new data.

## Data export (`scripts/export_tour_data.py`)

Run once with `venv/bin/python scripts/export_tour_data.py`. It writes `web/public/data/index.json` (a list of `{slug, label, n_features}`) and one `web/public/data/<slug>.json` per disease. The JSON files are committed so the page works without running Python.

Per-disease JSON must contain at least:

| Key | Content | Used by |
|---|---|---|
| `label`, `slug`, `feature_names` | Dataset identity and real measurement names | top bar, scene 1 |
| `train.angles` (N×3), `train.labels` | Training patients in angle space | scenes 2, 3 |
| `test[]` | Per test patient: `raw` (original units), `standardized`, `pca` (3), `angles` (3), `label`, and `models.{linear,rbf,qsvc,vqc}.{pred, prob}` | scenes 1, 5 |
| `linear.plane` | `coef` (3) + `intercept` in angle space | scene 2 |
| `rbf.grid` | `decision_function` sampled on a G×G×G grid over `[0, π]³` (G≈24), plus `G` and bounds | scene 2 (isosurface at 0) |
| `qsvc.kernel_train` | Train×train kernel matrix (75×75) | scene 3 web |
| `qsvc.kernel_test` | Per test patient: kernel row vs all training patients | scene 3 |
| `vqc.loss_history` | Objective value per COBYLA evaluation | scene 4 |
| `vqc.weights_history` | Ansatz weights per evaluation (9 params for 3 qubits, reps=2) | scene 4 knobs |
| `feature_ranges` | Per-feature min/max over the dataset (original units) | scene 1 bar scaling |

`train_all_models` currently records only the loss in its VQC callback. Capturing `weights_history` is a backward-compatible addition: an extra key in the VQC result dict.

## Scene details

1. **Patient becomes qubits.** One bar per real measurement (6–30), labelled with the feature name and its value in original units. Stage A shows raw heights to make the very different scales visible. Stage B: bars even out to standardised values. Stage C: bars merge into 3 glowing PCA values. Stage D: 3 Bloch spheres; each arrow animates the **first encoding layer** of the ZZ feature map (H takes |0⟩ to the equator, then the phase rotates by 2·angle). The caption notes that later entangling layers can't be drawn per qubit.
2. **Classical doctors.** Training patients are dots at their angle coordinates, coloured by true label; the chosen patient is a highlighted white dot. Linear SVM's real plane slides in, with a caption showing the Linear SVM probability. Then the RBF SVM real decision surface (isosurface of `rbf.grid` at 0) grows in, with a caption showing the RBF probability.
3. **QSVC.** Two Bloch spheres: the chosen patient and their **most similar** training patient (first-layer states, as in scene 1), with the real kernel value `K` shown. Then the web: training patients at their angle coordinates; threads between pairs with kernel above a threshold (or top-k per patient), brightness proportional to K. The chosen patient's threads light up last, with the QSVC probability.
4. **VQC.** Nine knobs (one per ansatz weight) rotate to the real `weights_history` values per iteration, while the real `loss_history` draws as a 3D tube descending over a grid floor. The caption shows the iteration and loss.
5. **Finale.** Four columns rise to each model's real disease probability for the chosen patient; a column turns red if it crosses the 50% sheet. The true diagnosis and which models got it right are shown.

## Done means

- `cd web && npm install && npm run dev` serves the tour; `npm run build` produces a static `web/dist/` that runs offline.
- All 5 diseases load. Every scene runs for any test patient with no console errors, and switching disease or patient mid-tour works.
- Streamlit app still works (`streamlit run app.py`), and its QSVC and VQC scores are now identical across reruns. README "Known limitations" is updated to match (it currently says quantum scores vary between runs).
- README gets a short "3D Diagnosis Tour" section with the run commands.

## Left to the builder

- Module and file layout inside `web/`, animation timing and easing, camera framing, exact colours within the sample palette.
- Isosurface technique for the RBF boundary (e.g. `MarchingCubes` from `three/addons`, or a point/voxel shell).
- Kernel-web thread threshold or top-k value, and the bar height scaling in scene 1 (log or per-feature normalisation), as long as stage A visibly shows the scale differences.
- Whether to use a tween library (e.g. GSAP) or hand-rolled easing.
