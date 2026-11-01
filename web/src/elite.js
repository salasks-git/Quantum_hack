// Pure helpers + lazy loader for the "Elite Hybrid Architectures" part of the QuHaMA scene
// (mirrors Streamlit Tab 8's leaderboard: QuHaMA, Sandwich HQNN, Q-CKTA vs RBF / Linear SVM).
// Data: web/public/data/<slug>.elite.json (scripts/export_tour_elite.py). Unit-tested in elite.test.js.
import { AUC_UNRELIABLE } from "./quhama.js";

export const ELITE_KEYS = ["quhama", "sandwich", "ckta"];
export const ELITE_NAMES = { quhama: "QuHaMA", sandwich: "Sandwich HQNN", ckta: "Q-CKTA" };
/** Below this F2 the caption says plainly that the model misses many sick patients. */
export const F2_LOW = 0.5;

// ---- lazy loader (cached per disease; a failed load is not cached) ----
const base = () => (import.meta.env && import.meta.env.BASE_URL) || "/";
const cache = new Map();
export function loadElite(slug) {
  if (!cache.has(slug)) {
    const msg = `Could not load Elite Hybrid data for '${slug}'`;
    const p = (async () => {
      const res = await fetch(base() + `data/${slug}.elite.json`);
      if (!res.ok) throw new Error(msg);
      try { return await res.json(); } catch { throw new Error(msg); }
    })();
    p.catch(() => cache.delete(slug));
    cache.set(slug, p);
  }
  return cache.get(slug);
}

/** Real (precomputed) test patients: a custom patient (label null) is appended at the end and excluded. */
const realTest = (data) => ((data && data.test) || []).filter((p) => p.label != null);

/**
 * Checks the elite export against the main tour data: three approaches, a leaderboard, and
 * test[] rows in the same order (same labels) as the real test patients of <slug>.json.
 */
export function eliteCheck(e, data) {
  if (!e || !Array.isArray(e.leaderboard) || !e.leaderboard.length || !e.approaches) {
    return { ok: false, msg: "Elite Hybrid data is missing or malformed." };
  }
  for (const k of ELITE_KEYS) {
    const a = e.approaches[k];
    if (!a || !Array.isArray(a.test) || !a.architecture || !a.metrics) {
      return { ok: false, msg: "Elite Hybrid data is missing or malformed." };
    }
  }
  if (data && data.custom_run) return { ok: false, msg: "Elite Hybrid results are precomputed for the default settings: Reset to precomputed." };
  const real = realTest(data);
  for (const k of ELITE_KEYS) {
    const t = e.approaches[k].test;
    if (t.length !== real.length || t.some((r, i) => r.label !== real[i].label)) {
      return { ok: false, msg: "Elite Hybrid data does not match this disease's test patients (re-export needed)." };
    }
  }
  return { ok: true };
}

/**
 * Each elite model's verdict for test patient `i` (index into data.test).
 * A custom patient (label null) has no precomputed elite predictions: { ok: true, custom: true }.
 * The verdict follows `pred` (as Streamlit); `borderline` = pred and prob disagree or prob is exactly 50%.
 */
export function elitePatient(e, data, i) {
  const p = data && data.test && data.test[i];
  if (!p) return { ok: false, msg: "No such test patient." };
  if (p.label == null) return { ok: true, custom: true };
  const rows = ELITE_KEYS.map((k) => {
    const r = e.approaches[k].test[i];
    return {
      key: k, name: ELITE_NAMES[k], pred: r.pred, prob: r.prob,
      disease: r.pred === 1,
      correct: r.pred === r.label,
      borderline: r.prob === 0.5 || (r.prob >= 0.5) !== (r.pred === 1),
    };
  });
  return { ok: true, custom: false, actualSick: p.label === 1, rows, sandwich: e.approaches.sandwich.test[i] };
}

/** Leaderboard rows in the tab's order, with the F2 leader(s) (ties all lead) and honesty flags. */
export function leaderboardView(e) {
  const best = Math.max(...e.leaderboard.map((r) => r.f2));
  const rows = e.leaderboard.map((r) => ({
    ...r,
    elite: r.family === "elite",
    leader: Math.abs(r.f2 - best) < 1e-9,
    f2Low: r.f2 < F2_LOW,
    aucLow: typeof r.roc_auc === "number" && r.roc_auc < AUC_UNRELIABLE,
  }));
  return { rows, best, leaders: rows.filter((r) => r.leader) };
}

/** Plain-language warnings for the caption (elite models with a low F2 / ROC-AUC, classical leader). */
export function honestyLines(view) {
  const out = [];
  for (const r of view.rows) {
    if (!r.elite) continue;
    if (r.f2Low) out.push(`${r.model}: F2 ${r.f2.toFixed(3)} is low: it misses many sick patients (recall ${(r.recall * 100).toFixed(1)}%).`);
    if (r.aucLow) out.push(`${r.model}: ROC-AUC ${r.roc_auc.toFixed(3)} is close to a coin flip (0.5): not reliable on this dataset.`);
  }
  if (!view.leaders.some((r) => r.elite)) {
    out.push(`A classical baseline (${view.leaders.map((r) => r.model).join(", ")}) has the best F2 here: no elite hybrid beats it on this disease.`);
  } else if (view.leaders.some((r) => !r.elite)) {
    out.push(`The F2 lead is a tie with a classical baseline: no elite hybrid beats it outright on this disease.`);
  }
  return out;
}

/** "131 ms" / "14.3 s" style latency. */
export function fmtTime(s) {
  if (typeof s !== "number" || !isFinite(s)) return "—";
  if (s < 1) return `${(s * 1000).toFixed(s < 0.01 ? 1 : 0)} ms`;
  return `${s.toFixed(1)} s`;
}
