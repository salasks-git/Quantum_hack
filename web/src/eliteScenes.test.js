import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import {
  patientGuard, approachOk, metricRows, metricHonesty, withQuhamaRun, rankLeaderboard, kernelSource,
  schematicKernel, idealPattern, sandwichTimeline, cktaTimeline, sweepState, BAR_STAGGER, BAR_GROW,
} from "./eliteScenes.js";
import { honestyLines } from "./elite.js";

const read = (f) => JSON.parse(readFileSync(new URL(`../public/data/${f}`, import.meta.url), "utf8"));
const SLUGS = ["vertebral_column", "breast_cancer_wisconsin", "diabetes_pima", "heart_disease_statlog", "parkinsons_vocal"];

describe("patientGuard (custom-patient guard)", () => {
  const d = read("vertebral_column.json");
  it("accepts a real test patient", () => {
    expect(patientGuard(d, 0, "Sandwich")).toEqual({ ok: true });
  });
  it("refuses a custom (slider) patient with the scene's message", () => {
    const withCustom = { ...d, test: [...d.test, { index: "custom", label: null }] };
    const g = patientGuard(withCustom, d.test.length, "Sandwich");
    expect(g.ok).toBe(false);
    expect(g.kind).toBe("custom");
    expect(g.msg).toBe("Sandwich needs a precomputed test patient");
  });
  it("refuses a patient of a Custom run (other settings) and a missing patient", () => {
    expect(patientGuard({ ...d, custom_run: true }, 0, "CKTA").kind).toBe("customRun");
    expect(patientGuard(d, 999, "CKTA").kind).toBe("missing");
    expect(patientGuard(null, 0, "CKTA").ok).toBe(false);
  });
});

describe("metricRows / metricHonesty (metric formatting)", () => {
  it("formats F2 first, percentages, ROC-AUC and time from the JSON", () => {
    const m = read("vertebral_column.elite.json").approaches.sandwich.metrics;
    const rows = metricRows(m);
    expect(rows.map((r) => r.key)).toEqual(["f2", "accuracy", "precision", "recall", "roc_auc", "time_s"]);
    expect(rows.map((r) => r.text)).toEqual(["0.862", "76.0%", "78.9%", "88.2%", "0.846", "131 ms"]);
    expect(rows.filter((r) => r.bar).length).toBe(5);
    expect(rows.some((r) => r.warn)).toBe(false);
    expect(metricHonesty("Sandwich", m)).toEqual([]);
  });
  it("flags a low F2 and a coin-flip AUC, and tolerates missing values", () => {
    const rows = metricRows({ f2: 0.132, accuracy: 0.7, precision: 0.5, recall: 0.111, roc_auc: 0.55, time_s: 2 });
    expect(rows[0].warn).toBe(true);
    expect(rows[4].warn).toBe(true);
    expect(rows[5].text).toBe("2.0 s");
    const lines = metricHonesty("CKTA", { f2: 0.132, recall: 0.111, roc_auc: 0.55 });
    expect(lines[0]).toBe("CKTA: F2 0.132 is low: it misses many sick patients (recall 11.1%).");
    expect(lines[1]).toContain("coin flip");
    expect(metricRows({}).map((r) => r.text)).toEqual(["—", "—", "—", "—", "—", "—"]);
  });
  it("diabetes CKTA gets the low-F2 line from the real export", () => {
    const m = read("diabetes_pima.elite.json").approaches.ckta.metrics;
    expect(metricHonesty("CKTA", m)[0]).toMatch(/^CKTA: F2 0\.13\d is low/);
  });
});

describe("withQuhamaRun (QuHaMA row = scene 8's run)", () => {
  it.each(SLUGS)("%s: QuHaMA's leaderboard row equals <slug>.quhama.json's metrics", (slug) => {
    const e = read(`${slug}.elite.json`), q = read(`${slug}.quhama.json`);
    const { rows } = withQuhamaRun(e.leaderboard, q);
    const r = rows.find((x) => x.key === "quhama");
    expect(r.f2).toBeCloseTo(q.metrics["F2-Score"], 6);
    expect(r.roc_auc).toBeCloseTo(q.metrics["ROC-AUC"], 6);
    expect(r.accuracy).toBeCloseTo(q.metrics.Accuracy, 6);
    // the other rows are untouched
    rows.filter((x) => x.key !== "quhama").forEach((x) => expect(x).toEqual(e.leaderboard.find((y) => y.key === x.key)));
  });
  it("changes nothing when the files agree or quhama.json is missing", () => {
    const e = read("vertebral_column.elite.json");
    expect(withQuhamaRun(e.leaderboard, null)).toEqual({ rows: e.leaderboard, replaced: false });
    const lb = e.leaderboard.map((r) => (r.key === "quhama" ? { ...r, accuracy: 0.5, f2: 0.6, precision: 0.7, recall: 0.8, roc_auc: 0.9 } : r));
    const q = { metrics: { Accuracy: 0.5, "F2-Score": 0.6, Precision: 0.7, Recall: 0.8, "ROC-AUC": 0.9 } };
    expect(withQuhamaRun(lb, q).replaced).toBe(false);
  });
  it("replaces a stale row (and its time from the run's timings)", () => {
    const lb = [{ key: "quhama", model: "QuHaMA", family: "elite", accuracy: 0.1, f2: 0.1, precision: 0.1, recall: 0.1, roc_auc: 0.1, time_s: 99 }];
    const q = { metrics: { Accuracy: 0.5, "F2-Score": 0.6, Precision: 0.7, Recall: 0.8, "ROC-AUC": 0.9 }, timings: { train_s: 10, inf_s: 2 } };
    const out = withQuhamaRun(lb, q);
    expect(out.replaced).toBe(true);
    expect(out.rows[0]).toMatchObject({ accuracy: 0.5, f2: 0.6, precision: 0.7, recall: 0.8, roc_auc: 0.9, time_s: 12 });
    expect(lb[0].f2).toBe(0.1); // input not mutated
  });
});

describe("rankLeaderboard (sorting and leader pick)", () => {
  it("sorts by F2, best first, and picks the leader (vertebral: Linear SVM)", () => {
    const v = rankLeaderboard(read("vertebral_column.elite.json").leaderboard);
    expect(v.rows.map((r) => r.model)).toEqual(["Linear SVM", "CKTA", "RBF SVM", "Sandwich", "QuHaMA"]);
    expect(v.leaders.map((r) => r.model)).toEqual(["Linear SVM"]);
    expect(v.rows.map((r) => r.rank)).toEqual([1, 2, 2, 4, 5]);
    expect(honestyLines(v).some((l) => l.includes("classical baseline (Linear SVM)"))).toBe(true);
  });
  it("ties share the lead and keep the tab's order (parkinsons: CKTA = RBF SVM)", () => {
    const v = rankLeaderboard(read("parkinsons_vocal.elite.json").leaderboard);
    expect(v.leaders.map((r) => r.model)).toEqual(["CKTA", "RBF SVM"]);
    expect(v.rows.slice(0, 2).every((r) => r.rank === 1)).toBe(true);
    expect(honestyLines(v).some((l) => l.includes("tie"))).toBe(true);
  });
  it("treats F2 values equal to 6 decimals as a tie (diabetes: QuHaMA from quhama.json = Sandwich)", () => {
    const e = read("diabetes_pima.elite.json"), q = read("diabetes_pima.quhama.json");
    const v = rankLeaderboard(withQuhamaRun(e.leaderboard, q).rows);
    expect(v.leaders.map((r) => r.model).sort()).toEqual(["QuHaMA", "Sandwich"]);
    expect(v.rows[0].model).toBe("QuHaMA"); // tab order kept within the tie
    expect(v.rows.slice(0, 2).map((r) => r.rank)).toEqual([1, 1]);
  });
  it("marks families and honesty flags; drops rows without an F2", () => {
    const v = rankLeaderboard([
      { model: "A", key: "a", family: "elite", f2: 0.3, roc_auc: 0.55 },
      { model: "B", key: "b", family: "classical", f2: 0.9, roc_auc: 0.9 },
      { model: "C", key: "c", family: "classical", f2: null },
    ]);
    expect(v.rows.map((r) => r.model)).toEqual(["B", "A"]);
    expect(v.rows[1]).toMatchObject({ elite: true, f2Low: true, aucLow: true, leader: false });
    expect(rankLeaderboard([]).rows).toEqual([]);
  });
});

describe("kernelSource / schematic kernel", () => {
  it.each(SLUGS)("%s: the bandwidth export's kernels are CKTA's (alignments match at every s)", (slug) => {
    const sweep = read(`${slug}.elite.json`).approaches.ckta.architecture.bandwidth_sweep;
    const k = kernelSource(read(`${slug}.bandwidth.json`), sweep);
    expect(k.real).toBe(true);
    expect(k.kernels.length).toBe(sweep.length);
    expect(k.kernels[0].length).toBe(k.N);
  });
  it("falls back to a schematic when the alignments differ or data is missing", () => {
    const bw = read("vertebral_column.bandwidth.json");
    const sweep = read("vertebral_column.elite.json").approaches.ckta.architecture.bandwidth_sweep;
    expect(kernelSource(bw, sweep.map((r) => ({ ...r, alignment: r.alignment + 0.01 }))).real).toBe(false);
    expect(kernelSource(null, sweep).real).toBe(false);
    expect(kernelSource(bw, [{ s: 0.33, alignment: 0.1 }]).real).toBe(false);
  });
  it("schematic same-diagnosis blocks sharpen with the alignment; ideal pattern is yyᵀ", () => {
    const lo = schematicKernel(4, 0), hi = schematicKernel(4, 1);
    expect(hi[0][1]).toBeGreaterThan(lo[0][1]);
    expect(hi[0][3]).toBeLessThan(hi[0][1]);
    expect(hi[2][2]).toBe(1);
    expect(idealPattern([0, 0, 1])).toEqual([[1, 1, 0], [1, 1, 0], [0, 0, 1]]);
  });
});

describe("timelines (pacing)", () => {
  it("Sandwich: stops after each formed stage, the last one after all metric bars have grown", () => {
    for (const known of [true, false]) {
      const tl = sandwichTimeline(known);
      expect(tl.stops.length).toBe(known ? 3 : 2);
      const last = tl.stops[tl.stops.length - 1];
      expect(tl.bars[1]).toBe(tl.bars[0] + 4 * BAR_STAGGER + BAR_GROW);
      expect(last).toBeGreaterThan(tl.bars[1]);
      expect(tl.stops.some((s) => s > tl.bars[0] && s < tl.bars[1])).toBe(false); // no stop mid-growth
      expect(tl.duration - last).toBeGreaterThanOrEqual(5); // time to read the metrics
      expect([...tl.stops].sort((a, b) => a - b)).toEqual(tl.stops);
    }
    expect(sandwichTimeline(false).p3).toBeLessThan(sandwichTimeline(true).p3);
  });
  it("CKTA: no stop during the sweep or the leaderboard growth; one after each", () => {
    const tl = cktaTimeline(5);
    for (const s of tl.stops) {
      expect(s > tl.sweep0 && s < tl.settle[1]).toBe(false);
      expect(s > tl.bars[0] && s < tl.leader[1]).toBe(false);
    }
    expect(tl.stops[0]).toBeGreaterThan(tl.fuse[1]);
    expect(tl.stops[2]).toBeGreaterThan(tl.leader[1]);
    expect(tl.duration - tl.stops[2]).toBeGreaterThanOrEqual(5);
  });
  it("sweepState walks the bandwidths, grows each bar once, then settles on s*", () => {
    const tl = cktaTimeline(5);
    expect(sweepState(0, 5, 1, tl).cur).toBe(-1);
    const mid = sweepState(tl.sweep0 + 2 * tl.step + tl.morph / 2, 5, 1, tl);
    expect(mid).toMatchObject({ from: 1, to: 2, cur: 2 });
    expect(mid.mix).toBeCloseTo(0.5, 5);
    expect(mid.grown[0]).toBe(1);
    expect(mid.grown[4]).toBe(0);
    const end = sweepState(tl.settle[1] + 1, 5, 1, tl);
    expect(end).toMatchObject({ from: 4, to: 1, mix: 1, cur: 1 });
    expect(end.grown.every((g) => g === 1)).toBe(true);
  });
});

describe("approachOk", () => {
  it("checks the approach block", () => {
    const e = read("vertebral_column.elite.json");
    expect(approachOk(e, "sandwich")).toBe(true);
    expect(approachOk(e, "nope")).toBe(false);
    expect(approachOk(null, "ckta")).toBe(false);
  });
});
