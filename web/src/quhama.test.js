import { describe, it, expect, vi, afterEach } from "vitest";
import { readFileSync } from "node:fs";
import { pickInputs, quhamaPatient, AUC_UNRELIABLE, pct1 } from "./quhama.js";
import { loadQuhama } from "./data.js";

const read = (f) => JSON.parse(readFileSync(new URL(`../public/data/${f}`, import.meta.url), "utf8"));

describe("pickInputs", () => {
  it("keeps every feature when there are few", () => {
    expect(pickInputs([0.1, -2, 3])).toEqual([0, 1, 2]);
  });
  it("keeps the largest |value| features in feature order", () => {
    expect(pickInputs([0.1, -5, 0.2, 4, -0.3], 3)).toEqual([1, 3, 4]);
  });
});

describe("quhamaPatient", () => {
  const q = read("vertebral_column.quhama.json");
  const data = read("vertebral_column.json");
  it("matches the main tour data patient by patient", () => {
    data.test.forEach((p, i) => {
      const r = quhamaPatient(q, data, i);
      expect(r.ok).toBe(true);
      expect(r.row.label).toBe(p.label);
      expect(r.disease).toBe(r.row.p_final >= 0.5);
      expect(r.correct).toBe(r.row.pred === r.row.label);
      expect(r.row.w_classical + r.row.w_quantum).toBeCloseTo(1, 5);
    });
  });
  it("flags vertebral (test ROC-AUC 0.507) as unreliable, heart disease not", () => {
    expect(quhamaPatient(q, data, 0).unreliable).toBe(true);
    const qh = read("heart_disease_statlog.quhama.json");
    const dh = read("heart_disease_statlog.json");
    expect(qh.metrics["ROC-AUC"]).toBeGreaterThanOrEqual(AUC_UNRELIABLE);
    expect(quhamaPatient(qh, dh, 0).unreliable).toBe(false);
  });
  it("refuses mismatched or missing data instead of showing wrong numbers", () => {
    expect(quhamaPatient(null, data, 0).ok).toBe(false);
    const bad = { ...q, test: q.test.slice(1) };
    expect(quhamaPatient(bad, data, 0).ok).toBe(false);
  });
  it("formats percentages with one decimal", () => {
    expect(pct1(0.91523)).toBe("91.5");
  });
});

describe("loadQuhama", () => {
  afterEach(() => vi.unstubAllGlobals());
  it("rejects on 404 with a readable message", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => ({ ok: false, status: 404 })));
    await expect(loadQuhama("nope")).rejects.toThrow("Could not load QuHaMA data for 'nope'");
  });
});
