import { describe, it, expect } from "vitest";
import { clinicalScores, clinicalLeader } from "./clinical.js";

const pt = (label, preds) => ({
  label,
  models: Object.fromEntries(Object.entries(preds).map(([k, pred]) => [k, { pred, prob: pred ? 0.9 : 0.1 }])),
});

describe("clinicalScores", () => {
  it("counts a hand-built 6-patient confusion matrix and F2 exactly", () => {
    // labels: 1 1 1 0 0 0 ; preds: 1 1 0 1 0 0 -> tp=2 fn=1 fp=1 tn=2
    const test = [pt(1, { a: 1 }), pt(1, { a: 1 }), pt(1, { a: 0 }), pt(0, { a: 1 }), pt(0, { a: 0 }), pt(0, { a: 0 })];
    const s = clinicalScores(test, "a");
    expect(s).toMatchObject({ tp: 2, fp: 1, fn: 1, tn: 2, missed: 1 });
    expect(s.recall).toBeCloseTo(2 / 3, 10);
    expect(s.precision).toBeCloseTo(2 / 3, 10);
    // F2 = 5PR / (4P + R) = 5*(4/9) / (8/3 + 2/3) = (20/9)/(10/3) = 2/3
    expect(s.f2).toBeCloseTo(2 / 3, 10);
  });

  it("uneven precision/recall matches sklearn fbeta(beta=2)", () => {
    // tp=1 fp=3 fn=0 -> P=0.25 R=1 -> F2 = 5*0.25/(1+1) = 0.625
    const test = [pt(1, { a: 1 }), pt(0, { a: 1 }), pt(0, { a: 1 }), pt(0, { a: 1 }), pt(0, { a: 0 }), pt(0, { a: 0 })];
    const s = clinicalScores(test, "a");
    expect(s).toMatchObject({ tp: 1, fp: 3, fn: 0, tn: 2, missed: 0 });
    expect(s.f2).toBeCloseTo(0.625, 10);
  });

  it("all-negative predictions give f2 === 0 and no NaN", () => {
    const test = [pt(1, { a: 0 }), pt(1, { a: 0 }), pt(0, { a: 0 })];
    const s = clinicalScores(test, "a");
    expect(s.f2).toBe(0);
    expect(s.precision).toBe(0);
    expect(s.recall).toBe(0);
    expect(s.missed).toBe(2);
    expect(Object.values(s).some(Number.isNaN)).toBe(false);
  });

  it("no sick patients and no positive predictions gives 0, not NaN", () => {
    const s = clinicalScores([pt(0, { a: 0 }), pt(0, { a: 0 })], "a");
    expect(s.f2).toBe(0);
    expect(Object.values(s).some(Number.isNaN)).toBe(false);
  });
});

describe("clinicalLeader", () => {
  // 2 sick + 8 healthy. a: catches 1 of 2 sick, no false alarms -> tp1 fn1 fp0 -> F2 = 5/9, missed 1.
  // c: catches both sick, flags all 8 healthy -> tp2 fn0 fp8 -> F2 = 10/18 = 5/9, missed 0.
  // b: same predictions as a. d: predicts nobody -> F2 0.
  const test = [
    pt(1, { a: 1, b: 1, c: 1, d: 0 }),
    pt(1, { a: 0, b: 0, c: 1, d: 0 }),
    ...Array.from({ length: 8 }, () => pt(0, { a: 0, b: 0, c: 1, d: 0 })),
  ];

  it("fixture really is an F2 tie", () => {
    expect(clinicalScores(test, "a").f2).toBeCloseTo(5 / 9, 12);
    expect(clinicalScores(test, "c").f2).toBeCloseTo(5 / 9, 12);
    expect(clinicalScores(test, "a").missed).toBe(1);
    expect(clinicalScores(test, "c").missed).toBe(0);
  });

  it("picks the max F2", () => {
    expect(clinicalLeader(test, ["d", "a"])).toBe("a");
  });

  it("breaks an F2 tie by fewer missed diagnoses, regardless of key order", () => {
    expect(clinicalLeader(test, ["a", "c"])).toBe("c");
    expect(clinicalLeader(test, ["c", "a"])).toBe("c");
  });

  it("breaks a full tie (same F2, same missed) by key order", () => {
    expect(clinicalLeader(test, ["b", "a", "d"])).toBe("b");
    expect(clinicalLeader(test, ["a", "b", "d"])).toBe("a");
  });
});
