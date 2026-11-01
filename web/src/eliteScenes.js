// Pure helpers for the Sandwich (step 9) and CKTA + Elite leaderboard (step 10) tour scenes.
// No three.js / DOM: unit-tested in eliteScenes.test.js. Data: <slug>.elite.json (scripts/export_tour_elite.py),
// with QuHaMA's leaderboard row taken from <slug>.quhama.json (the run scene 8 shows) and the real CKTA
// kernel matrices from <slug>.bandwidth.json when its alignments match CKTA's sweep.
import { leaderboardView, F2_LOW, fmtTime } from "./elite.js";
import { AUC_UNRELIABLE, pct1 } from "./quhama.js";
import { sKey } from "./bandwidth.js";

export { F2_LOW };

/**
 * Can the scene show this patient's precomputed elite values?
 * A custom (slider) patient has no precomputed values; a Custom run (other settings) has a different test set.
 * Returns { ok: true } or { ok: false, kind: "custom" | "customRun" | "missing", msg }.
 */
export function patientGuard(data, i, model) {
  const p = data && data.test && data.test[i];
  if (!p) return { ok: false, kind: "missing", msg: "No such test patient." };
  if (p.label == null || p.index === "custom") {
    return { ok: false, kind: "custom", msg: `${model} needs a precomputed test patient` };
  }
  if (data.custom_run) {
    return { ok: false, kind: "customRun", msg: `${model} is precomputed for the default settings: Reset to precomputed to see this patient` };
  }
  return { ok: true };
}

/** The approach's block in elite.json is usable (architecture, metrics, test[]). */
export function approachOk(e, key) {
  const a = e && e.approaches && e.approaches[key];
  return !!(a && a.architecture && a.metrics && Array.isArray(a.test));
}

const num = (v) => typeof v === "number" && isFinite(v);

/**
 * One model's test metrics as display rows (F2 first: the clinical target).
 * `warn` marks a low F2 (< F2_LOW) or a coin-flip ROC-AUC (< AUC_UNRELIABLE). Time has no bar.
 */
export function metricRows(m) {
  const g = (k) => (m && num(m[k]) ? m[k] : null);
  const fx = (v) => (v === null ? "—" : v.toFixed(3));
  const pc = (v) => (v === null ? "—" : `${pct1(v)}%`);
  const f2 = g("f2"), auc = g("roc_auc");
  return [
    { key: "f2", label: "F2 🎯", value: f2, text: fx(f2), warn: f2 !== null && f2 < F2_LOW, bar: true },
    { key: "accuracy", label: "Accuracy", value: g("accuracy"), text: pc(g("accuracy")), warn: false, bar: true },
    { key: "precision", label: "Precision", value: g("precision"), text: pc(g("precision")), warn: false, bar: true },
    { key: "recall", label: "Recall", value: g("recall"), text: pc(g("recall")), warn: false, bar: true },
    { key: "roc_auc", label: "ROC-AUC", value: auc, text: fx(auc), warn: auc !== null && auc < AUC_UNRELIABLE, bar: true },
    { key: "time_s", label: "Time", value: g("time_s"), text: fmtTime(g("time_s")), warn: false, bar: false },
  ];
}

/** Plain-language warnings for one model's metrics (low F2, coin-flip ROC-AUC). Empty when both are fine. */
export function metricHonesty(name, m) {
  const out = [];
  if (m && num(m.f2) && m.f2 < F2_LOW) {
    out.push(`${name}: F2 ${m.f2.toFixed(3)} is low: it misses many sick patients${num(m.recall) ? ` (recall ${pct1(m.recall)}%)` : ""}.`);
  }
  if (m && num(m.roc_auc) && m.roc_auc < AUC_UNRELIABLE) {
    out.push(`${name}: ROC-AUC ${m.roc_auc.toFixed(3)} is close to a coin flip (0.5): not reliable on this dataset.`);
  }
  return out;
}

const QH = [["accuracy", "Accuracy"], ["f2", "F2-Score"], ["precision", "Precision"], ["recall", "Recall"], ["roc_auc", "ROC-AUC"]];

/**
 * The leaderboard with QuHaMA's row taken from <slug>.quhama.json (the run scene 8 shows), so the tour shows
 * ONE set of QuHaMA numbers. If the two files already agree nothing changes. Returns { rows, replaced }.
 */
export function withQuhamaRun(leaderboard, q) {
  const rows = (leaderboard || []).map((r) => ({ ...r }));
  const m = q && q.metrics;
  const i = rows.findIndex((r) => r.key === "quhama");
  if (i < 0 || !m || !QH.every(([, k]) => num(m[k]))) return { rows, replaced: false };
  const differs = QH.some(([a, k]) => !num(rows[i][a]) || Math.abs(rows[i][a] - m[k]) > 1e-6);
  if (!differs) return { rows, replaced: false };
  for (const [a, k] of QH) rows[i][a] = m[k];
  const t = q.timings;
  if (t && num(t.train_s)) rows[i].time_s = t.train_s + (num(t.inf_s) ? t.inf_s : 0);
  return { rows, replaced: true };
}

/**
 * Leaderboard sorted by F2 (best first; ties keep the tab's order, a missing F2 goes last), with the
 * leader(s) (ties all lead), rank, and the honesty flags of leaderboardView (elite, f2Low, aucLow).
 * Same shape as leaderboardView, so honestyLines() works on it.
 */
const TIE = 1e-6;
export function rankLeaderboard(rows) {
  const clean = (rows || []).filter((r) => r && num(r.f2));
  if (!clean.length) return { rows: [], best: NaN, leaders: [] };
  const v = leaderboardView({ leaderboard: clean });
  // ties within TIE (QuHaMA's row from quhama.json is rounded to 6 decimals)
  const same = (a, b) => Math.abs(a - b) <= TIE;
  const sorted = v.rows.map((r, i) => ({ r, i }))
    .sort((a, b) => (same(a.r.f2, b.r.f2) ? 0 : b.r.f2 - a.r.f2) || a.i - b.i).map((x) => x.r);
  let rank = 0, prev = null;
  sorted.forEach((r, i) => {
    if (prev === null || !same(r.f2, prev)) rank = i + 1;
    r.rank = rank;
    r.leader = same(r.f2, v.best);
    prev = r.f2;
  });
  return { rows: sorted, best: v.best, leaders: sorted.filter((r) => r.leader) };
}

/**
 * Real CKTA kernel matrices: <slug>.bandwidth.json holds the quantum kernel of 40 of the training patients
 * (sorted healthy first) at each bandwidth s. They are the CKTA kernels when, for every s of CKTA's sweep,
 * the bandwidth export's kernel-target alignment equals CKTA's (same kernel, same training set).
 * Returns { real: true, N, labels, kernels: [matrix per sweep row], nTrain } or { real: false, reason }.
 */
export function kernelSource(bw, sweep, tol = 1e-4) {
  if (!bw || !bw.kernels || !bw.diagnostics || !Array.isArray(bw.labels)) return { real: false, reason: "no kernel export" };
  if (!Array.isArray(sweep) || !sweep.length) return { real: false, reason: "no CKTA sweep" };
  const kernels = [];
  for (const r of sweep) {
    const k = sKey(r.s);
    const K = bw.kernels[k], d = bw.diagnostics[k];
    if (!Array.isArray(K) || K.length !== bw.labels.length || !d || !num(d.alignment) || !num(r.alignment)) {
      return { real: false, reason: `no kernel for s = ${r.s}` };
    }
    if (Math.abs(d.alignment - r.alignment) > tol) return { real: false, reason: `alignment differs at s = ${r.s}` };
    kernels.push(K);
  }
  return { real: true, N: bw.labels.length, labels: bw.labels.slice(), kernels, nTrain: bw.settings && bw.settings.n_train };
}

/**
 * Schematic kernel (used only when no real matrix matches): N patients, first half healthy. Diagonal 1,
 * same-diagnosis pairs get sharper (higher) as `sharp` (= alignment / max alignment, 0..1) grows.
 */
export function schematicKernel(N, sharp) {
  const s = Math.max(0, Math.min(1, sharp));
  return Array.from({ length: N }, (_, i) => Array.from({ length: N }, (_, j) => {
    if (i === j) return 1;
    return (i < N / 2) === (j < N / 2) ? 0.12 + 0.7 * s : 0.12 + 0.06 * s;
  }));
}

/** The ideal label pattern the alignment compares with: 1 for same diagnosis, 0 otherwise. */
export function idealPattern(labels) {
  return labels.map((a) => labels.map((b) => (a === b ? 1 : 0)));
}

// ---------------- timelines (seconds after the data has loaded) ----------------

/** Metric / leaderboard bars: each starts this long after the previous one and grows for BAR_GROW s. */
export const BAR_STAGGER = 0.3, BAR_GROW = 0.8;

/**
 * Sandwich scene. Part 1 architecture + particle flow (stop after it), part 2 this patient's values
 * (stop after the verdict; replaced by a short message for a custom patient), part 3 the metric bars
 * (one stop after all bars have grown).
 */
export function sandwichTimeline(known) {
  const p3 = known ? 12.0 : 10.4;
  const bars = [p3 + 0.6, p3 + 0.6 + 4 * BAR_STAGGER + BAR_GROW];
  const stops = known ? [6.2, 11.6, bars[1] + 0.5] : [6.2, bars[1] + 0.5];
  return { p3, bars, stops, duration: p3 + 9 };
}


/**
 * CKTA scene for a sweep of n bandwidths. Part 1: the kernel morphs through each s while its alignment bar
 * grows (no stop inside), settles on s*, then the fused-kernel weights appear (stop). Part 2: this patient's
 * CKTA probability and CKTA's metrics (stop). Part 3: the leaderboard bars grow (one stop after all of them).
 */
export function cktaTimeline(n, nRows = 5) {
  const sweep0 = 1.0, morph = 0.6, hold = 1.0, step = morph + hold;
  const sweepEnd = sweep0 + Math.max(1, n) * step;
  const settle = [sweepEnd, sweepEnd + 0.7];
  const fuse = [settle[1] + 0.1, settle[1] + 0.9];
  const p2 = fuse[1] + 0.9;
  const p2Stop = p2 + 3.2;
  const p3 = p2Stop + 0.6;
  const bars = [p3 + 0.4, p3 + 0.4 + Math.max(0, nRows - 1) * BAR_STAGGER + BAR_GROW];
  const leader = [bars[1], bars[1] + 0.6];
  const stops = [fuse[1] + 0.4, p2Stop, leader[1] + 0.4];
  return { sweep0, morph, hold, step, sweepEnd, settle, fuse, p2, p3, bars, leader, stops, duration: p3 + 10 };
}

/**
 * Which kernel the landscape shows at time t during the sweep: morphing from sweep row `from` to `to`
 * with weight `mix` (0..1); `cur` is the row whose alignment bar is current; `grown[i]` (0..1) is how far
 * bar i has grown. After the sweep the landscape settles on `best`.
 */
export function sweepState(t, n, best, tl) {
  const grown = Array.from({ length: n }, (_, i) => {
    const a = tl.sweep0 + i * tl.step;
    return Math.min(1, Math.max(0, (t - a) / (tl.morph + 0.2)));
  });
  if (t < tl.sweep0) return { from: 0, to: 0, mix: 1, cur: -1, grown };
  if (t < tl.sweepEnd) {
    const i = Math.min(n - 1, Math.floor((t - tl.sweep0) / tl.step));
    const mix = i === 0 ? 1 : Math.min(1, (t - tl.sweep0 - i * tl.step) / tl.morph);
    return { from: Math.max(0, i - 1), to: i, mix, cur: i, grown };
  }
  const b = best >= 0 ? best : n - 1;
  const mix = Math.min(1, Math.max(0, (t - tl.settle[0]) / (tl.settle[1] - tl.settle[0])));
  return { from: n - 1, to: b, mix, cur: b, grown };
}
