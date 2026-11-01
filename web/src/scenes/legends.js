// Plain-language legend entries per scene (shown in the Legend panel).
// Shape: { symbol, text, swatch? } where swatch is a CSS colour for colour keys.
// Scenes 1-3 depend on the number of qubits n (2, 3 or 4); LEGEND_k = legendk(3) is the shipped tour.

const SUB = ["₁", "₂", "₃", "₄", "₅", "₆", "₇", "₈"];
export function legend1(n = 3) {
  const xs = Array.from({ length: n }, (_, k) => `x${SUB[k]}`).join(", ");
  return [
  { symbol: "raw", text: "The patient's real measurements in their original units (years, mmHg, mg/dL…). Very different scales." },
  { symbol: "σ (sigma)", text: "Standard deviation: how far a value is from the average training patient. 0σ = typical, +1σ = one \"typical spread\" above average, −2σ = well below average. Values beyond ±2σ are unusual." },
  { symbol: "standardise", text: "Rescale every measurement to σ units so no measurement dominates just because of its units." },
  { symbol: "PCA", text: `Principal Component Analysis: squeezes all measurements into ${n} summary scores that keep most of the variation between patients.` },
  { symbol: xs, text: `The ${n} summary scores turned into rotation angles in radians, between 0 and π (≈3.14).` },
  { symbol: "qubit", text: "A quantum bit. It can point anywhere on a sphere, not just 0 or 1." },
  { symbol: "Bloch sphere", text: "The sphere that shows a single qubit's state; the arrow is the state." },
  { symbol: "|0⟩", text: "The starting state of every qubit (arrow pointing straight up, north pole)." },
  { symbol: "H (Hadamard)", text: "A quantum gate that tips the arrow from |0⟩ down to the equator." },
  { symbol: "phase 2x", text: "A rotation around the equator by twice the patient's angle: this is how the data is written into the qubit." },
  { symbol: "first layer only", text: "The feature map repeats and entangles the qubits; entangled qubits can't be drawn as separate arrows, so only the first layer is shown." },
  ];
}
export const LEGEND_1 = legend1(3);

export function legend2(n = 3) {
  const axes = n === 2
    ? { symbol: "axes x₁, x₂", text: "Each patient's 2 angles from scene 1, used as a position on a flat plane (2 qubits → 2D)." }
    : n === 3
      ? { symbol: "axes x₁, x₂, x₃", text: "Each patient's 3 angles from scene 1, used as a position in 3D." }
      : { symbol: "axes x₁, x₂, x₃", text: `The first 3 of each patient's ${n} angles from scene 1, used as a position in 3D. The other angle${n > 4 ? "s" : ""} can't be drawn.` };
  const slice = n > 3
    ? [{ symbol: "3D slice", text: `With ${n} qubits the boundaries live in ${n}D. The surfaces shown are the 3D slice where x₄ is fixed at the training patients' average; patients are drawn by x₁–x₃ only.` }]
    : [];
  return [
    { symbol: "healthy", swatch: "#4ade80", text: "A training patient whose true diagnosis is healthy." },
    { symbol: "sick", swatch: "#fb7185", text: "A training patient whose true diagnosis is disease." },
    { symbol: "your patient", swatch: "#ffffff", text: "The chosen test patient (never seen in training)." },
    axes,
    ...slice,
    n === 2
      ? { symbol: "Linear SVM", text: "Separates sick from healthy with one straight line." }
      : { symbol: "Linear SVM", text: "Separates sick from healthy with one flat sheet (a plane)." },
    n === 2
      ? { symbol: "RBF SVM", text: "Separates them with a curved line (a contour) that can wrap around clusters." }
      : { symbol: "RBF SVM", text: "Separates them with a curved surface that can wrap around clusters." },
    { symbol: "decision boundary (0)", text: "Where the model's score is exactly 0; one side is predicted disease, the other healthy." },
    { symbol: "arrow", text: "Points to the side the model calls \"disease\"." },
    { symbol: "prob", text: "The model's confidence that the patient has the disease (0–100%)." },
  ];
}
export const LEGEND_2 = legend2(3);

export function legend3(n = 3) {
  const pos = n === 3
    ? []
    : [{ symbol: "positions", text: n === 2
      ? "With 2 qubits each patient is placed by its 2 angles on a flat plane. K is unaffected."
      : `Patients are placed by their first 3 angles; ${n === 4 ? "x₄ isn't" : "the others aren't"} drawn. K always uses all ${n} qubits.` }];
  return [
    { symbol: "|ψ⟩ (psi)", text: `The quantum state a patient is encoded into (all ${n} qubits together).` },
    { symbol: "K = |⟨ψa|ψb⟩|²", text: "Quantum similarity (kernel) between two patients: 1 = identical states, 0 = completely different." },
    { symbol: "QSVC", text: "Quantum-kernel Support Vector Classifier: an SVM that uses K as its similarity measure." },
    { symbol: "threads", text: "Lines between patients; brighter = higher K (more alike to the quantum computer)." },
    ...pos,
    { symbol: "most similar patient", text: "The training patient with the highest K to your patient." },
  ];
}
export const LEGEND_3 = legend3(3);

export const LEGEND_4 = [
  { symbol: "VQC", text: "Variational Quantum Classifier: a quantum circuit with adjustable settings, trained like a neural network." },
  { symbol: "knobs θ₁…θ₉ (theta)", text: "The 9 trainable rotation angles (radians) of the circuit (RealAmplitudes ansatz)." },
  { symbol: "loss", text: "How wrong the model is on the training patients (cross-entropy). Lower is better." },
  { symbol: "evaluation", text: "One test of a knob setting by the optimiser." },
  { symbol: "COBYLA", text: "The optimiser that chooses the next knob setting without needing gradients." },
  { symbol: "lowest-loss marker", text: "The best setting found during training." },
];

export const LEGEND_5 = [
  { symbol: "prob", text: "Each model's confidence that this patient has the disease." },
  { symbol: "50% line", text: "The decision threshold: above → disease, below → healthy." },
  { symbol: "verdict", text: "The model's actual yes/no answer (it can differ slightly from prob near 50%)." },
  { symbol: "borderline", text: "The model's yes/no answer and its probability disagree, or prob is right on 50%: treat with caution." },
  { symbol: "✓ / ✗", text: "The model's verdict matches / doesn't match the patient's true diagnosis." },
  { symbol: "Hybrid (QSVC+RBF)", text: "Averages the two: ½ × RBF prob + ½ × QSVC prob." },
  { symbol: "F2 score", text: "A 0–1 accuracy score that counts missed sick patients twice as heavily as false alarms (recall-weighted). Higher is better for screening." },
  { symbol: "missed diagnoses", text: "Sick test patients the model called healthy (false negatives). The most dangerous error in screening." },
  { symbol: "🏆 Clinical leader", text: "The model with the best F2 on this disease's 25 test patients." },
];

export const LEGEND_6 = [
  { symbol: "QPU", text: "Quantum Processing Unit: a real IBM quantum chip." },
  { symbol: "transpile", text: "Rewrite the circuit into the only gates the chip can run natively." },
  { symbol: "sx, x", text: "Native single-qubit pulses (~35 ns each)." },
  { symbol: "rz", text: "Single-qubit rotation done in software (\"virtual\"), 0 ns on the chip." },
  { symbol: "ECR", text: "Echoed Cross-Resonance: the chip's native two-qubit (entangling) gate, ~300 ns. The slowest and noisiest gate." },
  { symbol: "depth", text: "Number of gate layers that must run one after another." },
  { symbol: "µs / ns", text: "Microseconds / nanoseconds (millionths / billionths of a second)." },
  { symbol: "shots", text: "How many times the circuit is run to estimate probabilities (1,024)." },
  { symbol: "estimate", text: "Chip times computed from typical IBM gate timings, not measured." },
  { symbol: "CPU (measured)", text: "How long this laptop's simulator actually took." },
  { symbol: "queue", text: "Typical wait for a shared IBM machine on the free tier (~10 min); can be much longer." },
  { symbol: "fidelity (hardware)", text: "Fraction of shots that returned 000: the chip's measured similarity K." },
  { symbol: "fidelity (ideal)", text: "The same similarity computed perfectly on the simulator." },
  { symbol: "noise gap", text: "Ideal − hardware: how much real-chip noise changed the answer." },
  { symbol: "train / inference", text: "Training = fitting the model on the 75 training patients; inference = predicting the 25 test patients. Bars: solid = train, light = inference." },
  { symbol: "log scale", text: "Each axis step is 10× longer, so a millisecond and a minute fit on one chart." },
  { symbol: "F2 score", text: "0–1 score that counts a missed sick patient twice as heavily as a false alarm. Higher is better." },
  { symbol: "efficiency frontier", text: "Pareto-optimal models (gold line): no other model is both faster and has a higher F2." },
];

export const LEGEND_BANDWIDTH = [
  { symbol: "s (bandwidth)", text: "A dial that shrinks each patient's angles before they are loaded into the qubits: K = |⟨ψ(s·x)|ψ(s·x′)⟩|². It plays the same role as γ in a classical RBF kernel and costs nothing on hardware." },
  { symbol: "K", text: "Quantum similarity (kernel) between two training patients: 1 = identical states, 0 = completely different." },
  { symbol: "kernel landscape", text: "The kernel matrix in 3D: one column per pair of training patients, height and brightness = K. Patients are sorted healthy first, then disease." },
  { symbol: "healthy block", swatch: "#4ade80", text: "Pairs of two healthy patients. A useful kernel makes this block (and the disease block) higher than the mixed pairs." },
  { symbol: "disease block", swatch: "#fb7185", text: "Pairs of two sick patients." },
  { symbol: "too sharp", text: "s too large: every patient looks unique (K ≈ 0 off the diagonal, a near-identity matrix), so the SVM memorises the training set (overfits)." },
  { symbol: "too flat / too blurred", text: "s too small: everyone looks alike (K ≈ 1 for every pair, an all-ones matrix), so the kernel can't separate sick from healthy." },
  { symbol: "balanced", text: "In between: similar patients score high, different patients score low." },
  { symbol: "kernel-target alignment", text: "How closely the kernel's similarity pattern matches the true diagnoses (centered). Higher = better." },
  { symbol: "cross-validation (5-fold)", text: "All patients of the cohort split into 5 parts; train on 4, test on the 5th, rotate, and average. Scaling and PCA are refit inside each fold." },
  { symbol: "CV accuracy bars", swatch: "#a78bfa", text: "QSVC cross-validated accuracy at each s (whisker = ± 1 std over folds)." },
  { symbol: "ROC-AUC dots", swatch: "#f0abfc", text: "QSVC cross-validated ROC-AUC at each s (from the SVM margin)." },
  { symbol: "RBF reference", swatch: "#38bdf8", text: "The classical RBF SVM's cross-validated accuracy (plane) and ROC-AUC (dashed line) on the same folds." },
  { symbol: "best s", swatch: "#fbbf24", text: "The s with the highest CV accuracy. Picking it from the same table is itself tuning, so it is slightly optimistic." },
  { symbol: "default s", text: "The app's default bandwidth (the value is shown on screen from the exported data)." },
];
// --- QuHaMA scene legend ---
export const LEGEND_QUHAMA = [
  { symbol: "QuHaMA", text: "Quantum-Hybrid Adaptive Multi-branch Architecture: a small neural network with a classical branch and a quantum branch, blended per patient." },
  { symbol: "inputs (σ)", swatch: "#38bdf8", text: "The patient's standardised measurements (σ units). Brighter = further from the average patient. For many features only the 12 most extreme are drawn; the encoder still reads all of them." },
  { symbol: "encoder", swatch: "#38bdf8", text: "A small neural network (Linear → BatchNorm → ReLU → Linear → Tanh) that compresses all measurements into a few numbers." },
  { symbol: "latent space (k = qubits)", text: "The encoder's output z: k = 3 numbers between −1 and +1, one per qubit. Brightness = |z|." },
  { symbol: "classical head", swatch: "#38bdf8", text: "A tiny neural network that reads z and gives the classical disease probability P_c." },
  { symbol: "quantum branch", swatch: "#a78bfa", text: "z is turned into qubit angles θ = (z+1)·π/2 (0 to π) and classified by a quantum-kernel SVM (QSVC, ZZFeatureMap) on 3 qubits, giving P_q. The Bloch arrows show only the first encoding layer." },
  { symbol: "gating network (per-patient adaptive weights)", text: "A third small network that looks at z, P_c and P_q and decides, for this patient, how much to trust each branch (the two weights add up to 100%)." },
  { symbol: "blended prob", text: "P = w_classical × P_c + w_quantum × P_q. At or above 50% the verdict is disease." },
  { symbol: "particles", text: "Show the order the numbers flow through the network; they carry no extra information." },
  { symbol: "ROC-AUC", text: "How well the model ranks sick above healthy patients across all thresholds: 1 = perfect, 0.5 = coin flip. Below 0.6 the model is not reliable on that dataset." },
  { symbol: "F2 score", text: "A 0–1 score that counts missed sick patients twice as heavily as false alarms." },
  // Elite Hybrid Architectures part (Streamlit Tab 8 parity)
  { symbol: "Elite Hybrid Architectures", text: "The three hybrid quantum-classical models Streamlit's Tab 8 compares with the classical baselines: QuHaMA, Sandwich HQNN and Q-CKTA." },
  { symbol: "Sandwich HQNN", swatch: "#a78bfa", text: "Classical → quantum → classical: a Pre-Net squeezes the measurements into 3 qubit angles, a small quantum circuit returns one number per qubit, and a Post-Net makes the call from both. The circuit's own weights are fixed (0.5); only the two classical networks are trained." },
  { symbol: "⟨Z⟩ (Z expectation)", text: "The quantum circuit's output for one qubit, between −1 (always measured 1) and +1 (always measured 0). The Bloch arrow's height shows it." },
  { symbol: "Q-CKTA", swatch: "#f0abfc", text: "Bandwidth-scaled quantum kernel with Centered Kernel-Target Alignment: tries several bandwidths s, keeps the best-aligned one, mixes it with an RBF kernel and trains an SVM on the mix." },
  { symbol: "bandwidth s", text: "Scales the patient angles before encoding (s = 1 is the unscaled kernel). Too large and every patient looks different from every other (kernel concentration)." },
  { symbol: "kernel-target alignment", text: "How well a kernel's similarities match the diagnoses (same diagnosis = similar). Higher is better. The sharpening tile is a schematic; the bars are the real alignment for each s." },
  { symbol: "fusion weights", text: "Q-CKTA's classical/quantum kernel mix, set by the two alignments: w_c = A_c / (A_c + A_q)." },
  { symbol: "leaderboard / F2", text: "The tab's table: the 3 elite hybrids and the RBF / Linear SVM on the same test patients. Bars are F2 (the clinical target); ★ marks the best F2 (ties share it). Violet bars with gold edges = elite hybrids, cyan = classical." },
  { symbol: "elite verdicts", text: "This patient's DISEASE/HEALTHY call and probability from each elite model, with ✓/✗ against the true diagnosis. A custom (slider) patient has no precomputed elite predictions." },
];
// --- end QuHaMA scene legend ---
// --- Sandwich scene legend (step 9) ---
export const LEGEND_SANDWICH = [
  { symbol: "Sandwich HQNN", text: "A hybrid neural network built like a sandwich: a classical network, then a small quantum circuit, then another classical network. Streamlit Tab 8's \"layer-embedded\" hybrid." },
  { symbol: "inputs (σ)", swatch: "#38bdf8", text: "The patient's standardised measurements (σ units). Brighter = further from the average patient. For many features only the 12 most extreme are drawn; the network still reads all of them." },
  { symbol: "classical encoder (Pre-Net)", swatch: "#38bdf8", text: "A small neural network that squeezes all measurements into 3 numbers z between −1 and +1, one per qubit. It is trained." },
  { symbol: "z (latent values)", swatch: "#e9d5ff", text: "The encoder's 3 output numbers for this patient. They become qubit angles θ = (z+1)·π/2 (0 to π)." },
  { symbol: "quantum layer", swatch: "#a78bfa", text: "A small quantum circuit on 3 qubits (ZZFeatureMap + a RealAmplitudes-style ansatz, run on a noiseless simulator). Its own weights are fixed at 0.5; it is not trained." },
  { symbol: "expectation value ⟨Z⟩", text: "What the quantum layer outputs for one qubit: the average measurement on a −1 … +1 scale (+1 = always measured 0, −1 = always 1). The Bloch arrow's height shows it." },
  { symbol: "classical head (Post-Net)", swatch: "#38bdf8", text: "A second small neural network that reads both z and ⟨Z⟩ and outputs the disease probability. It is trained." },
  { symbol: "P(disease)", text: "The Post-Net's probability that this patient has the disease. At or above 50% the verdict is disease." },
  { symbol: "✓ / ✗", text: "The verdict matches / doesn't match the patient's true diagnosis." },
  { symbol: "particles", text: "Show the order the numbers flow through the network; they carry no extra information." },
  { symbol: "F2 🎯", swatch: "#fbbf24", text: "The clinical target: a 0–1 score that counts missed sick patients twice as heavily as false alarms. Higher is better." },
  { symbol: "ROC-AUC", text: "How well the model ranks sick above healthy patients: 1 = perfect, 0.5 = coin flip. Below 0.6 it is not reliable." },
  { symbol: "training time", text: "How long Sandwich took to train in this run (as Tab 8's \"Latency\" measures it)." },
  { symbol: "custom patient", text: "A slider-made patient has no precomputed Sandwich values, so only the architecture and the test metrics are shown." },
];
// --- end Sandwich scene legend ---
// --- CKTA + Elite leaderboard scene legend (step 10) ---
export const LEGEND_CKTA = [
  { symbol: "Q-CKTA", swatch: "#f0abfc", text: "Bandwidth-scaled quantum kernel with Centered Kernel-Target Alignment: tries several bandwidths s, keeps the best-aligned one, mixes it with a classical RBF kernel and trains an SVM on the mix." },
  { symbol: "quantum kernel K(s)", text: "The quantum similarity of every pair of training patients (1 = identical, 0 = completely different), drawn as a 3D landscape sorted healthy first, then disease. Height and brightness = K." },
  { symbol: "bandwidth s", text: "Shrinks the patient angles before they go into the qubits. Too large and everyone looks different from everyone; too small and everyone looks alike." },
  { symbol: "ideal pattern", text: "What a perfect kernel would look like: patients with the same diagnosis fully similar (1), different diagnoses not similar (0)." },
  { symbol: "kernel-target alignment", text: "How closely the kernel's pattern of similarities matches the ideal pattern (centered, so the overall level doesn't count). It is computed before any classifier is trained." },
  { symbol: "alignment score A_q(s)", swatch: "#a78bfa", text: "The kernel-target alignment of the quantum kernel at each s (bars). Higher is better; the gold bar is the best one, s*." },
  { symbol: "RBF A_c", swatch: "#38bdf8", text: "The alignment of the classical RBF kernel on the same patients (reference line)." },
  { symbol: "fusion weights", text: "CKTA mixes the two kernels by their alignments: RBF weight w_c = A_c / (A_c + A_q(s*)), quantum weight w_q = 1 − w_c." },
  { symbol: "schematic", text: "Shown only if no real kernel matrix matching CKTA's alignments was exported: the tile then illustrates the idea, and only the bars are real values." },
  { symbol: "F2 leaderboard", text: "Streamlit Tab 8's table: QuHaMA, Sandwich, CKTA and the RBF / Linear SVM on the same test patients, sorted by F2 (the clinical target). The other metrics are in the table under the chart." },
  { symbol: "elite hybrid", swatch: "#a78bfa", text: "Violet bars with gold edges: the three hybrid quantum-classical models." },
  { symbol: "classical", swatch: "#38bdf8", text: "Cyan bars: the classical RBF and Linear SVM baselines." },
  { symbol: "★ clinical leader", swatch: "#fbbf24", text: "The best F2 on this disease (ties share it)." },
  { symbol: "latency", text: "Each model's measured run time (Tab 8's Latency column)." },
];
// --- end CKTA + Elite leaderboard scene legend ---
