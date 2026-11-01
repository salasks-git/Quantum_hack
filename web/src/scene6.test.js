import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { layerGates, logRange, fmtTime } from "./scenes/scene6.js";

const data = JSON.parse(readFileSync(new URL("../public/data/heart_disease_statlog.json", import.meta.url)));

describe("scene 6 helpers", () => {
  it("ASAP layering of the exported gates reproduces the exported depth", () => {
    const { depth, layer } = layerGates(data.qpu.gates, 3);
    expect(depth).toBe(data.qpu.fm_depth);
    expect(layer.length).toBe(data.qpu.gates.length);
  });
  it("log range covers the three racers", () => {
    const { lo, hi } = logRange([data.qpu.qsvc_qpu_s, data.timings.qsvc.total_s, data.qpu.queue_s]);
    expect(lo).toBeLessThanOrEqual(data.timings.qsvc.total_s);
    expect(hi).toBeGreaterThanOrEqual(data.qpu.qsvc_qpu_s);
  });
  it("formats times", () => {
    expect(fmtTime(17.4)).toBe("17.4 s");
    expect(fmtTime(600)).toBe("10.0 min");
  });
});
