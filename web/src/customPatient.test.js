// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { buildSyntheticPatient, withCustomPatient, createPredictor, MSG } from "./customPatient.js";
import { clinicalScores } from "./clinical.js";
import { modelMetrics } from "./metrics.js";
import { buildPredictionTable } from "./allPatients.js";
import { createQpuController, renderQpuPanel } from "./qpuPanel.js";

const m = (pred, prob) => ({ pred, prob });
const resp = () => ({
  run_id: "r1", angles: [1, 2, 3], clamped: false,
  models: { linear: m(1, 0.6), rbf: m(0, 0.4), qsvc: m(1, 0.7), vqc: m(0, 0.3), hybrid: m(1, 0.55) },
  kernel_row: [0.1, 0.9],
  reconstructed: { pca: [1, 2, 3], standardized: [0.1, 0.2], raw: [10, 20] },
});
const data = () => ({
  settings: { n_qubits: 3 },
  test: [
    { index: 0, label: 1, angles: [0, 0, 0], models: { linear: m(1, 0.9) } },
    { index: 1, label: 0, angles: [1, 1, 1], models: { linear: m(1, 0.8) } },
  ],
  qsvc: { kernel_test: [[0.5, 0.5], [0.2, 0.8]] },
});

describe("synthetic patient", () => {
  it("has the expected shape", () => {
    const p = buildSyntheticPatient(resp());
    expect(p).toMatchObject({ index: "custom", label: null, angles: [1, 2, 3], raw: [10, 20], standardized: [0.1, 0.2], pca: [1, 2, 3], reconstructed: true });
    expect(p.models.qsvc).toEqual(m(1, 0.7));
  });
  it("appends to copies without mutating the original", () => {
    const d = data();
    const before = JSON.stringify(d);
    const { data: w, index } = withCustomPatient(d, resp());
    expect(JSON.stringify(d)).toBe(before);
    expect(index).toBe(2);
    expect(w.test).toHaveLength(3);
    expect(w.test[2].label).toBeNull();
    expect(w.qsvc.kernel_test[2]).toEqual([0.1, 0.9]);
    expect(w.qsvc.kernel_test).not.toBe(d.qsvc.kernel_test);
    expect(d.qsvc.kernel_test).toHaveLength(2);
  });
});

describe("label null is excluded from metrics", () => {
  const { data: w } = withCustomPatient(data(), resp());
  it("clinicalScores / modelMetrics ignore it", () => {
    w.test[2].models.linear = m(1, 0.99);
    expect(clinicalScores(w.test, "linear")).toEqual(clinicalScores(data().test, "linear"));
    expect(modelMetrics(w.test, "linear")).toEqual(modelMetrics(data().test, "linear"));
  });
  it("prediction table: no correctness for the custom row, counts use real patients", () => {
    const t = buildPredictionTable(w, ["linear"]);
    expect(t.rows[2].cells.linear.correct).toBeNull();
    expect(t.summary.linear).toEqual({ accuracy: 0.5, missed: 0, sick: 1 });
  });
});

describe("predictor debounce + 202 retry", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());
  const json = (status, body) => ({ status, ok: status < 300, json: async () => body });

  it("debounces to one call with the last body", async () => {
    const f = vi.fn(async () => json(200, resp()));
    const onResult = vi.fn();
    const p = createPredictor({ fetchImpl: f, onResult });
    p.request({ a: 1 }); p.request({ a: 2 }); p.request({ a: 3 });
    await vi.advanceTimersByTimeAsync(249);
    expect(f).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(2);
    expect(f).toHaveBeenCalledTimes(1);
    expect(JSON.parse(f.mock.calls[0][1].body)).toEqual({ a: 3 });
    expect(onResult).toHaveBeenCalledTimes(1);
  });

  it("retries every 1 s on 202 then succeeds", async () => {
    const f = vi.fn().mockResolvedValueOnce(json(202, { state: "training" })).mockResolvedValueOnce(json(202, { state: "training" })).mockResolvedValue(json(200, resp()));
    const onResult = vi.fn(), onWarming = vi.fn();
    createPredictor({ fetchImpl: f, onResult, onWarming }).request({});
    await vi.advanceTimersByTimeAsync(250);
    expect(f).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(1000);
    expect(f).toHaveBeenCalledTimes(2);
    await vi.advanceTimersByTimeAsync(1000);
    expect(f).toHaveBeenCalledTimes(3);
    expect(onWarming).toHaveBeenCalledTimes(2);
    expect(onResult).toHaveBeenCalledTimes(1);
  });

  it("gives up after 30 s of 202s", async () => {
    const f = vi.fn(async () => json(202, { state: "training" }));
    const onError = vi.fn();
    createPredictor({ fetchImpl: f, onError }).request({});
    await vi.advanceTimersByTimeAsync(250 + 40000);
    expect(onError).toHaveBeenCalledWith(MSG.warmTimeout, "timeout");
    expect(f.mock.calls.length).toBeLessThanOrEqual(32);
  });

  it("a newer request cancels an older retry loop; fetch failure -> server-down", async () => {
    const f = vi.fn().mockResolvedValueOnce(json(202, {})).mockRejectedValue(new Error("x"));
    const onError = vi.fn(), onResult = vi.fn();
    const p = createPredictor({ fetchImpl: f, onError, onResult });
    p.request({ n: 1 });
    await vi.advanceTimersByTimeAsync(250);
    p.request({ n: 2 });
    await vi.advanceTimersByTimeAsync(5000);
    expect(onResult).not.toHaveBeenCalled();
    expect(onError).toHaveBeenCalledWith(MSG.serverDown, "server-down");
    expect(f).toHaveBeenCalledTimes(2);
  });
});

describe("IBM backends button", () => {
  it("fetches only on click", async () => {
    const client = {
      health: vi.fn(async () => ({ ok: false, reason: "server-down" })),
      pollStatus: vi.fn(), submit: vi.fn(),
      backends: vi.fn(async () => ({ ok: true, rows: [{ name: "ibm_a", num_qubits: 127, operational: true, pending_jobs: 3 }] })),
    };
    const ctl = createQpuController({ client });
    ctl.activate({ slug: "s", patient: 0, kRow: [0.5] });
    await Promise.resolve();
    expect(client.backends).not.toHaveBeenCalled();
    expect(renderQpuPanel(ctl.view())).toContain("Show IBM backends");
    await ctl.showBackends();
    expect(client.backends).toHaveBeenCalledTimes(1);
    const html = renderQpuPanel(ctl.view());
    expect(html).toContain("ibm_a");
    expect(html).toContain("Listing uses no QPU time");
    expect(client.submit).not.toHaveBeenCalled();
  });
});
