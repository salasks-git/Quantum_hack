// Pure helpers for the Kernel Bandwidth Lab scene (data from <slug>.bandwidth.json, see scripts/export_tour_extras.py).

/** Key used for per-s objects in the JSON (Python str(s): 1 -> "1.0", 0.05 -> "0.05"). */
export function sKey(s) {
  return Number.isInteger(s) ? s.toFixed(1) : String(s);
}

/** Display form of s like Python's f"{s:g}" (1.0 -> "1", 0.05 -> "0.05"). */
export const fmtS = (s) => String(+s);

/** Short regime label: the exported text up to its colon ("Too sharp", "Balanced", "Too blurred"). */
export function regimeShort(text) {
  const t = String(text || "");
  const i = t.indexOf(":");
  return i > 0 ? t.slice(0, i) : t;
}

/**
 * Morph schedule over the bandwidths, largest s first.
 * Step i starts at t0 + i*(morph+hold): the landscape morphs from the previous s over `morph`
 * seconds, then holds for `hold` seconds. Returns {from, to, mix, index} (s values, mix in [0,1]).
 */
export function morphState(t, sDesc, { t0 = 0, morph = 0.7, hold = 1.0 } = {}) {
  const step = morph + hold;
  const u = t - t0;
  if (u <= 0) return { from: sDesc[0], to: sDesc[0], mix: 1, index: 0 };
  const i = Math.min(sDesc.length - 1, Math.floor(u / step));
  const local = u - i * step;
  if (i === 0) return { from: sDesc[0], to: sDesc[0], mix: 1, index: 0 };
  const mix = Math.min(1, local / morph);
  return { from: sDesc[i - 1], to: sDesc[i], mix, index: i };
}

/** Summary of the CV sweep: best row (first max accuracy, like pandas idxmax), default row, RBF. */
export function sweepSummary(bw) {
  const rows = bw.sweep.rows;
  let best = rows[0];
  for (const r of rows) if (r.accuracy > best.accuracy) best = r;
  const bestS = bw.sweep.best_s !== undefined ? bw.sweep.best_s : best.bandwidth;
  const bestRow = rows.find((r) => r.bandwidth === bestS) || best;
  const defRow = rows.find((r) => r.bandwidth === bw.default) || null;
  return { rows, best: bestRow, def: defRow, rbf: bw.sweep.rbf };
}

/** Lower bound of the score axis, as in plots.bandwidth_scores: max(0, min(all scores, RBF) - 0.08). */
export function scoreAxisMin(rows, rbf) {
  const lo = Math.min(...rows.map((r) => r.accuracy), ...rows.map((r) => r.roc_auc), rbf.accuracy, rbf.roc_auc);
  return Math.max(0, lo - 0.08);
}

/** Number of healthy (label 0) patients before the class boundary of the sorted kernel rows. */
export function classBoundary(labels) {
  return labels.filter((l) => l === 0).length;
}
