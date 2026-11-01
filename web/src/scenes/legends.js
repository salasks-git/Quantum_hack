// Plain-language legend entries per scene (shown in the Legend panel).
// Shape: { symbol, text, swatch? } where swatch is a CSS colour for colour keys.
export const LEGEND_1 = [
  { symbol: "raw", text: "The patient's real measurements in their original units (years, mmHg, mg/dL…). Very different scales." },
  { symbol: "σ (sigma)", text: "Standard deviation: how far a value is from the average training patient. 0σ = typical, +1σ = one \"typical spread\" above average, −2σ = well below average. Values beyond ±2σ are unusual." },
  { symbol: "standardise", text: "Rescale every measurement to σ units so no measurement dominates just because of its units." },
  { symbol: "PCA", text: "Principal Component Analysis: squeezes all measurements into 3 summary scores that keep most of the variation between patients." },
  { symbol: "x₁, x₂, x₃", text: "The 3 summary scores turned into rotation angles in radians, between 0 and π (≈3.14)." },
  { symbol: "qubit", text: "A quantum bit. It can point anywhere on a sphere, not just 0 or 1." },
  { symbol: "Bloch sphere", text: "The sphere that shows a single qubit's state; the arrow is the state." },
  { symbol: "|0⟩", text: "The starting state of every qubit (arrow pointing straight up, north pole)." },
  { symbol: "H (Hadamard)", text: "A quantum gate that tips the arrow from |0⟩ down to the equator." },
  { symbol: "phase 2x", text: "A rotation around the equator by twice the patient's angle: this is how the data is written into the qubit." },
  { symbol: "first layer only", text: "The feature map repeats and entangles the qubits; entangled qubits can't be drawn as separate arrows, so only the first layer is shown." },
];

export const LEGEND_2 = [
  { symbol: "healthy", swatch: "#4ade80", text: "A training patient whose true diagnosis is healthy." },
  { symbol: "sick", swatch: "#fb7185", text: "A training patient whose true diagnosis is disease." },
  { symbol: "your patient", swatch: "#ffffff", text: "The chosen test patient (never seen in training)." },
  { symbol: "axes x₁, x₂, x₃", text: "Each patient's 3 angles from scene 1, used as a position in 3D." },
  { symbol: "Linear SVM", text: "Separates sick from healthy with one flat sheet (a plane)." },
  { symbol: "RBF SVM", text: "Separates them with a curved surface that can wrap around clusters." },
  { symbol: "decision boundary (0)", text: "Where the model's score is exactly 0; one side is predicted disease, the other healthy." },
  { symbol: "arrow", text: "Points to the side the model calls \"disease\"." },
  { symbol: "prob", text: "The model's confidence that the patient has the disease (0–100%)." },
];

export const LEGEND_3 = [
  { symbol: "|ψ⟩ (psi)", text: "The quantum state a patient is encoded into (all 3 qubits together)." },
  { symbol: "K = |⟨ψa|ψb⟩|²", text: "Quantum similarity (kernel) between two patients: 1 = identical states, 0 = completely different." },
  { symbol: "QSVC", text: "Quantum-kernel Support Vector Classifier: an SVM that uses K as its similarity measure." },
  { symbol: "threads", text: "Lines between patients; brighter = higher K (more alike to the quantum computer)." },
  { symbol: "most similar patient", text: "The training patient with the highest K to your patient." },
];

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
];
// --- end QuHaMA scene legend ---
