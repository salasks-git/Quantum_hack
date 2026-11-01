// Pure benchmark metrics matching Streamlit's benchmark_table (sklearn, zero_division=0).
import { clinicalScores } from "./clinical.js";

export function confusion(labels, preds) {
  let tp = 0, fp = 0, fn = 0, tn = 0;
  labels.forEach((y, i) => {
    const sick = y === 1, flagged = preds[i] === 1;
    if (sick && flagged) tp++; else if (sick) fn++; else if (flagged) fp++; else tn++;
  });
  return { tp, fp, fn, tn };
}
const div = (a, b) => (b ? a / b : 0);
export const precision = ({ tp, fp }) => div(tp, tp + fp);
export const recall = ({ tp, fn }) => div(tp, tp + fn);
export const fbeta = (beta) => ({ tp, fp, fn }) => {
  const b2 = beta * beta;
  return div((1 + b2) * tp, (1 + b2) * tp + b2 * fn + fp);
};
export const f1 = fbeta(1);
export const accuracy = ({ tp, fp, fn, tn }) => div(tp + tn, tp + fp + fn + tn);

// Rank-based AUC (Mann-Whitney U) with average ranks for ties; null if one class only.
export function rocAuc(labels, probs) {
  const n = labels.length;
  const nPos = labels.filter((y) => y === 1).length, nNeg = n - nPos;
  if (!nPos || !nNeg) return null;
  const idx = probs.map((_, i) => i).sort((a, b) => probs[a] - probs[b]);
  const rank = new Array(n);
  for (let i = 0; i < n;) {
    let j = i;
    while (j + 1 < n && probs[idx[j + 1]] === probs[idx[i]]) j++;
    const avg = (i + j) / 2 + 1;
    for (let k = i; k <= j; k++) rank[idx[k]] = avg;
    i = j + 1;
  }
  let sumPos = 0;
  labels.forEach((y, i) => { if (y === 1) sumPos += rank[i]; });
  return (sumPos - (nPos * (nPos + 1)) / 2) / (nPos * nNeg);
}

export function modelMetrics(test, key) {
  const labels = test.map((p) => p.label);
  const c = clinicalScores(test, key);
  const cm = { tp: c.tp, fp: c.fp, fn: c.fn, tn: c.tn };
  return {
    f2: c.f2, recall: c.recall, missed: c.missed, f1: f1(cm), precision: c.precision,
    accuracy: accuracy(cm), rocAuc: rocAuc(labels, test.map((p) => p.models[key].prob)), cm,
  };
}
