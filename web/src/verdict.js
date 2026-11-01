export const MODEL_KEYS = ["linear", "rbf", "qsvc", "vqc", "hybrid"];
export const MODEL_LABELS = {
  linear: "Linear SVM", rbf: "RBF SVM", qsvc: "QSVC", vqc: "VQC", hybrid: "Hybrid (QSVC+RBF)",
};

export function verdictFor({ pred, prob }) {
  const disease = pred === 1;
  return { disease, borderline: (prob > 0.5) !== disease };
}
