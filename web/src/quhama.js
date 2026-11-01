// Pure helpers for the QuHaMA tour scene (no three.js / DOM), unit-tested in quhama.test.js.

/** Below this test ROC-AUC the caption says plainly that QuHaMA is not reliable on the dataset. */
export const AUC_UNRELIABLE = 0.6;
/** At most this many input nodes are drawn; the rest are indicated as "+N more". */
export const MAX_INPUTS = 12;

/**
 * Indices of the input features to draw: all of them if there are at most `max`,
 * otherwise the `max` with the largest |standardized value|, kept in feature order.
 */
export function pickInputs(values, max = MAX_INPUTS) {
  const idx = values.map((_, i) => i);
  if (values.length <= max) return idx;
  return idx
    .sort((a, b) => Math.abs(values[b]) - Math.abs(values[a]) || a - b)
    .slice(0, max)
    .sort((a, b) => a - b);
}

/**
 * The QuHaMA breakdown for test patient `i`, checked against the main tour data
 * (same index as `data.test[]`). Returns { ok: true, row, ... } or { ok: false, msg }.
 */
export function quhamaPatient(q, data, i) {
  if (!q || !Array.isArray(q.test) || !q.metrics || !q.architecture) {
    return { ok: false, msg: "QuHaMA data is missing or malformed." };
  }
  const row = q.test[i];
  const main = data && data.test && data.test[i];
  if (!row || !main || q.test.length !== data.test.length || row.label !== main.label) {
    return { ok: false, msg: "QuHaMA data does not match this disease's test patients (re-export needed)." };
  }
  const auc = q.metrics["ROC-AUC"];
  return {
    ok: true,
    row,
    disease: row.pred === 1,
    actualSick: row.label === 1,
    correct: row.pred === row.label,
    unreliable: typeof auc === "number" && auc < AUC_UNRELIABLE,
    nTest: q.test.length,
  };
}

/** "68.5" style percentage with one decimal. */
export const pct1 = (p) => (p * 100).toFixed(1);
