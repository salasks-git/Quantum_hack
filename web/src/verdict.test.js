import { describe, it, expect } from "vitest";
import { verdictFor } from "./verdict.js";
describe("verdictFor", () => {
  it("cases", () => {
    expect(verdictFor({ pred: 1, prob: 0.8 })).toEqual({ disease: true, borderline: false });
    expect(verdictFor({ pred: 0, prob: 0.62 })).toEqual({ disease: false, borderline: true });
    expect(verdictFor({ pred: 1, prob: 0.4 })).toEqual({ disease: true, borderline: true });
    expect(verdictFor({ pred: 0, prob: 0.1 })).toEqual({ disease: false, borderline: false });
  });
});
