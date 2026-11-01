import { describe, it, expect, vi, afterEach } from "vitest";
import { sKey, fmtS, regimeShort, morphState, sweepSummary, scoreAxisMin, classBoundary } from "./bandwidth.js";
import { loadBandwidth } from "./data.js";

afterEach(() => vi.unstubAllGlobals());

describe("bandwidth helpers", () => {
  it("formats s like Python str()/:g", () => {
    expect(sKey(1)).toBe("1.0");
    expect(sKey(0.05)).toBe("0.05");
    expect(sKey(0.75)).toBe("0.75");
    expect(fmtS(1)).toBe("1");
    expect(fmtS(0.1)).toBe("0.1");
  });
  it("short regime label is the text before the colon", () => {
    expect(regimeShort("Too sharp: near-identity, every patient looks unique")).toBe("Too sharp");
    expect(regimeShort("Balanced")).toBe("Balanced");
    expect(regimeShort(undefined)).toBe("");
  });
  it("morphs largest to smallest and holds on each s", () => {
    const S = [1, 0.5, 0.1];
    const o = { t0: 1, morph: 1, hold: 1 };
    expect(morphState(0, S, o)).toMatchObject({ to: 1, mix: 1, index: 0 });
    expect(morphState(3.5, S, o)).toMatchObject({ from: 1, to: 0.5, index: 1 });
    expect(morphState(3.5, S, o).mix).toBeCloseTo(0.5);
    expect(morphState(4.5, S, o)).toMatchObject({ to: 0.5, mix: 1 });
    expect(morphState(99, S, o)).toMatchObject({ from: 0.5, to: 0.1, mix: 1, index: 2 });
  });
  it("summarises the sweep (best = exported best_s, first max otherwise)", () => {
    const rows = [
      { bandwidth: 0.05, accuracy: 0.7, roc_auc: 0.8 },
      { bandwidth: 0.1, accuracy: 0.76, roc_auc: 0.85 },
      { bandwidth: 0.2, accuracy: 0.76, roc_auc: 0.6 },
    ];
    const rbf = { accuracy: 0.74, roc_auc: 0.81 };
    const s = sweepSummary({ default: 0.1, sweep: { rows, rbf } });
    expect(s.best.bandwidth).toBe(0.1);
    expect(s.def.bandwidth).toBe(0.1);
    expect(sweepSummary({ default: 0.1, sweep: { rows, rbf, best_s: 0.2 } }).best.bandwidth).toBe(0.2);
    expect(scoreAxisMin(rows, rbf)).toBeCloseTo(0.52);
  });
  it("finds the healthy/disease boundary", () => {
    expect(classBoundary([0, 0, 0, 1, 1])).toBe(3);
  });
});

describe("loadBandwidth", () => {
  it("fetches <slug>.bandwidth.json", async () => {
    const f = vi.fn(async () => ({ ok: true, json: async () => ({ slug: "a" }) }));
    vi.stubGlobal("fetch", f);
    expect(await loadBandwidth("a")).toEqual({ slug: "a" });
    expect(f.mock.calls[0][0]).toMatch(/data\/a\.bandwidth\.json$/);
  });
  it("rejects with a clear message and does not cache the failure", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => ({ ok: false, status: 404 })));
    await expect(loadBandwidth("zz")).rejects.toThrow("Could not load Kernel Bandwidth Lab data for 'zz'");
    vi.stubGlobal("fetch", vi.fn(async () => ({ ok: true, json: async () => ({ slug: "zz" }) })));
    expect(await loadBandwidth("zz")).toEqual({ slug: "zz" });
  });
});
