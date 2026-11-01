import { describe, it, expect, vi, afterEach } from "vitest";
import { readFileSync } from "node:fs";
import { eliteCheck, elitePatient, leaderboardView, honestyLines, fmtTime, loadElite, ELITE_KEYS } from "./elite.js";

const read = (f) => JSON.parse(readFileSync(new URL(`../public/data/${f}`, import.meta.url), "utf8"));
const SLUGS = ["vertebral_column", "breast_cancer_wisconsin", "diabetes_pima", "heart_disease_statlog", "parkinsons_vocal"];

describe("eliteCheck", () => {
  it.each(SLUGS)("%s: the elite export matches the tour's test patients", (slug) => {
    expect(eliteCheck(read(`${slug}.elite.json`), read(`${slug}.json`)).ok).toBe(true);
  });
  it("refuses missing, malformed or mismatched data", () => {
    const e = read("vertebral_column.elite.json"), d = read("vertebral_column.json");
    expect(eliteCheck(null, d).ok).toBe(false);
    expect(eliteCheck({ ...e, approaches: { quhama: e.approaches.quhama } }, d).ok).toBe(false);
    const short = { ...e, approaches: { ...e.approaches, ckta: { ...e.approaches.ckta, test: e.approaches.ckta.test.slice(1) } } };
    expect(eliteCheck(short, d).ok).toBe(false);
  });
  it("still matches when a custom patient (label null) is appended", () => {
    const e = read("vertebral_column.elite.json"), d = read("vertebral_column.json");
    const withCustom = { ...d, test: [...d.test, { index: "custom", label: null }] };
    expect(eliteCheck(e, withCustom).ok).toBe(true);
  });
});

describe("elitePatient", () => {
  const e = read("parkinsons_vocal.elite.json"), d = read("parkinsons_vocal.json");
  it("gives each elite model's verdict from pred, compared with the true label", () => {
    d.test.forEach((p, i) => {
      const r = elitePatient(e, d, i);
      expect(r.ok && !r.custom).toBe(true);
      expect(r.actualSick).toBe(p.label === 1);
      expect(r.rows.map((x) => x.key)).toEqual(ELITE_KEYS);
      r.rows.forEach((x) => {
        const src = e.approaches[x.key].test[i];
        expect(x.disease).toBe(src.pred === 1);
        expect(x.correct).toBe(src.pred === p.label);
        expect(x.borderline).toBe((src.prob >= 0.5) !== (src.pred === 1) || src.prob === 0.5);
      });
      expect(r.sandwich.latent.length).toBe(3);
    });
  });
  it("flags a CKTA verdict whose prob disagrees with pred as borderline", () => {
    const i = e.approaches.ckta.test.findIndex((r) => (r.prob >= 0.5) !== (r.pred === 1));
    expect(i).toBeGreaterThanOrEqual(0);
    expect(elitePatient(e, d, i).rows[2].borderline).toBe(true);
  });
  it("reports a custom patient instead of inventing predictions", () => {
    const withCustom = { ...d, test: [...d.test, { index: "custom", label: null }] };
    expect(elitePatient(e, withCustom, d.test.length)).toEqual({ ok: true, custom: true });
  });
});

describe("leaderboardView / honestyLines", () => {
  it("keeps the tab's five rows in order and picks the F2 leader (vertebral: Linear SVM)", () => {
    const v = leaderboardView(read("vertebral_column.elite.json"));
    expect(v.rows.map((r) => r.model)).toEqual(["QuHaMA", "Sandwich", "CKTA", "RBF SVM", "Linear SVM"]);
    expect(v.leaders.map((r) => r.model)).toEqual(["Linear SVM"]);
    expect(honestyLines(v).some((l) => l.includes("classical baseline (Linear SVM)"))).toBe(true);
  });
  it("ties share the lead (parkinsons: CKTA = RBF SVM) and the tie is called out", () => {
    const v = leaderboardView(read("parkinsons_vocal.elite.json"));
    expect(v.leaders.map((r) => r.model)).toEqual(["CKTA", "RBF SVM"]);
    expect(honestyLines(v).some((l) => l.includes("tie"))).toBe(true);
  });
  it("says plainly when an elite model's F2 is low (diabetes: CKTA)", () => {
    const v = leaderboardView(read("diabetes_pima.elite.json"));
    const ckta = v.rows.find((r) => r.key === "ckta");
    expect(ckta.f2Low).toBe(true);
    expect(honestyLines(v).some((l) => l.startsWith("CKTA: F2"))).toBe(true);
  });
});

describe("fmtTime / loadElite", () => {
  afterEach(() => vi.unstubAllGlobals());
  it("formats latency", () => {
    expect(fmtTime(0.1309)).toBe("131 ms");
    expect(fmtTime(0.00114)).toBe("1.1 ms");
    expect(fmtTime(14.325)).toBe("14.3 s");
    expect(fmtTime(null)).toBe("—");
  });
  it("rejects on 404 with a readable message", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => ({ ok: false, status: 404 })));
    await expect(loadElite("nope")).rejects.toThrow("Could not load Elite Hybrid data for 'nope'");
  });
});
