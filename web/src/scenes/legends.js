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
];
