import { describe, it, expect } from "vitest";
import { fmtDuration, paretoFront, logScale, pyFixed, tickLabel } from "./timing.js";

describe("fmtDuration (matches qmed/metrics.py fmt_duration)", () => {
  it("formats ms and s", () => {
    expect(fmtDuration(0.0042)).toBe("4.2 ms");
    expect(fmtDuration(1.2345)).toBe("1.234 s"); // Python f"{1.2345:.3f}" -> 1.234
    expect(fmtDuration(17.400967359542847)).toBe("17.401 s");
    expect(fmtDuration(0.0011150836944580078)).toBe("1.1 ms");
  });
  it("switches at 10 ms", () => {
    expect(fmtDuration(0.01)).toBe("0.010 s");
    expect(fmtDuration(0.00999)).toBe("10.0 ms");
  });
  it("rounds exact ties half-to-even like Python", () => {
    expect(pyFixed(0.0625, 3)).toBe("0.062"); // Python: 0.062
    expect(pyFixed(0.1875, 3)).toBe("0.188"); // Python: 0.188
    expect(fmtDuration(0.5)).toBe("0.500 s");
  });
});

describe("paretoFront", () => {
  const P = (key, time, score) => ({ key, time, score });
  it("keeps non-dominated points sorted by time", () => {
    const pts = [P("slow", 10, 0.9), P("fast", 1, 0.5), P("bad", 5, 0.4), P("mid", 3, 0.7)];
    expect(paretoFront(pts)).toEqual(["fast", "mid", "slow"]);
  });
  it("drops a point that is both slower and no better", () => {
    expect(paretoFront([P("a", 1, 0.8), P("b", 2, 0.8)])).toEqual(["a"]); // same score, slower
    expect(paretoFront([P("a", 1, 0.8), P("b", 1, 0.6)])).toEqual(["a"]); // same time, lower score
  });
  it("keeps exact ties (neither dominates)", () => {
    expect(paretoFront([P("a", 1, 0.8), P("b", 1, 0.8), P("c", 5, 0.2)])).toEqual(["a", "b"]);
  });
  it("single and empty", () => {
    expect(paretoFront([P("a", 1, 0)])).toEqual(["a"]);
    expect(paretoFront([])).toEqual([]);
  });
});

describe("logScale", () => {
  it("maps endpoints and midpoint", () => {
    expect(logScale(0.001, 0.001, 100, 6)).toBe(0);
    expect(logScale(100, 0.001, 100, 6)).toBeCloseTo(6, 12);
    expect(logScale(1, 0.001, 100, 6)).toBeCloseTo(3.6, 12);
  });
  it("clamps outside the range", () => {
    expect(logScale(1e-9, 0.001, 100, 6)).toBe(0);
    expect(logScale(1e9, 0.001, 100, 6)).toBe(6);
  });
});

describe("tickLabel", () => {
  it("labels powers of ten", () => {
    expect([-4, -3, -2, -1, 0, 1, 2].map(tickLabel)).toEqual(["100 µs", "1 ms", "10 ms", "100 ms", "1 s", "10 s", "100 s"]);
  });
});
