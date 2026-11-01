import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { confusion, precision, recall, f1, fbeta, accuracy, rocAuc, modelMetrics } from "./metrics.js";
import { MODEL_KEYS } from "./verdict.js";

describe("rocAuc", () => {
  it("known example", () => expect(rocAuc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])).toBeCloseTo(0.75, 10));
  it("ties use average ranks", () => {
    expect(rocAuc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5])).toBeCloseTo(0.5, 10);
    expect(rocAuc([0, 1, 1], [0.2, 0.2, 0.9])).toBeCloseTo(0.75, 10);
  });
  it("perfect, inverted, single class", () => {
    expect(rocAuc([0, 1], [0.1, 0.9])).toBe(1);
    expect(rocAuc([0, 1], [0.9, 0.1])).toBe(0);
    expect(rocAuc([1, 1], [0.1, 0.9])).toBeNull();
    expect(rocAuc([0, 0], [0.1, 0.9])).toBeNull();
  });
});

describe("basic metrics", () => {
  const cm = confusion([1, 1, 1, 0, 0, 0, 0], [1, 1, 0, 1, 0, 0, 0]);
  it("confusion", () => expect(cm).toEqual({ tp: 2, fp: 1, fn: 1, tn: 3 }));
  it("hand values", () => {
    expect(precision(cm)).toBeCloseTo(2 / 3, 10);
    expect(recall(cm)).toBeCloseTo(2 / 3, 10);
    expect(f1(cm)).toBeCloseTo(2 / 3, 10);
    expect(fbeta(2)(cm)).toBeCloseTo(2 / 3, 10);
    expect(accuracy(cm)).toBeCloseTo(5 / 7, 10);
    expect(fbeta(2)({ tp: 3, fp: 1, fn: 2 })).toBeCloseTo((5 * 3) / (5 * 3 + 4 * 2 + 1), 10);
  });
  it("zero denominators give 0", () => {
    const z = { tp: 0, fp: 0, fn: 0, tn: 4 };
    expect([precision(z), recall(z), f1(z), fbeta(2)(z)]).toEqual([0, 0, 0, 0]);
  });
});

describe("modelMetrics", () => {
  const m = (pred, prob) => ({ pred, prob });
  const hand = [
    { label: 1, models: { a: m(1, 0.9) } }, { label: 1, models: { a: m(0, 0.45) } },
    { label: 0, models: { a: m(1, 0.7) } }, { label: 0, models: { a: m(0, 0.55) } },
  ];
  it("hand set", () => {
    const r = modelMetrics(hand, "a");
    expect(r).toMatchObject({ missed: 1, cm: { tp: 1, fp: 1, fn: 1, tn: 1 } });
    expect([r.f2, r.recall, r.f1, r.precision, r.accuracy, r.rocAuc]).toEqual([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]);
  });
});

// Expected values computed with sklearn (venv python) from heart_disease_statlog.json.
const SK = {
  linear: { f2: 0.818182, recall: 0.818182, missed: 2, f1: 0.818182, precision: 0.818182, accuracy: 0.84, rocAuc: 0.941558, cm: { tp: 9, fp: 2, fn: 2, tn: 12 } },
  rbf: { f2: 0.833333, recall: 0.818182, missed: 2, f1: 0.857143, precision: 0.9, accuracy: 0.88, rocAuc: 0.954545, cm: { tp: 9, fp: 1, fn: 2, tn: 13 } },
  qsvc: { f2: 0.462963, recall: 0.454545, missed: 6, f1: 0.47619, precision: 0.5, accuracy: 0.56, rocAuc: 0.600649, cm: { tp: 5, fp: 5, fn: 6, tn: 9 } },
  vqc: { f2: 0.283019, recall: 0.272727, missed: 8, f1: 0.3, precision: 0.333333, accuracy: 0.44, rocAuc: 0.422078, cm: { tp: 3, fp: 6, fn: 8, tn: 8 } },
  hybrid: { f2: 0.849057, recall: 0.818182, missed: 2, f1: 0.9, precision: 1.0, accuracy: 0.92, rocAuc: 0.954545, cm: { tp: 9, fp: 0, fn: 2, tn: 14 } },
};
describe("sklearn cross-check (heart_disease_statlog)", () => {
  const d = JSON.parse(readFileSync("public/data/heart_disease_statlog.json", "utf8"));
  MODEL_KEYS.forEach((k) => it(k, () => {
    const r = modelMetrics(d.test, k), e = SK[k];
    for (const f of ["f2", "recall", "f1", "precision", "accuracy", "rocAuc"]) expect(r[f]).toBeCloseTo(e[f], 3);
    expect(r.missed).toBe(e.missed);
    expect(r.cm).toEqual(e.cm);
  }));
});
