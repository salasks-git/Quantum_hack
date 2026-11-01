// @vitest-environment jsdom
import { describe, it, expect } from "vitest";
import { qpuSelection, adoptData, choosePatient, FIXTURE_QPU_MSG, CUSTOM_QPU_MSG } from "./tourState.js";
import { mountCustomPatient, withCustomPatient } from "./customPatient.js";
import { quhamaPatient, QUHAMA_MSG } from "./quhama.js";
import { eliteCheck } from "./elite.js";

const m = (pred, prob) => ({ pred, prob });
const mk = (nTest, extra = {}) => ({
  slug: "heart", label: "Heart Disease", settings: { n_qubits: 3, kernel_bandwidth: 0.1 },
  test: Array.from({ length: nTest }, (_, i) => ({ index: i, label: i % 2, angles: [i * 0.1, 0.2, 0.3], models: {} })),
  qsvc: { kernel_test: Array.from({ length: nTest }, () => [0.1, 0.9]) },
  ...extra,
});
const resp = () => ({
  run_id: "r1", angles: [1, 2, 3], clamped: false,
  models: { linear: m(1, 0.6), rbf: m(0, 0.4), qsvc: m(1, 0.7), vqc: m(0, 0.3), hybrid: m(1, 0.55) },
  kernel_row: [0.1, 0.9], reconstructed: { pca: [1, 2, 3], standardized: [0.1], raw: [10] },
});

describe("C2: new data while a custom patient is active", () => {
  function setup() {
    document.body.innerHTML = `<button id="b"></button><div id="p"></div>`;
    const base = mk(25);
    const st = { index: [], slug: "heart", base, data: base, patient: 3, realPatient: 3 };
    const cp = mountCustomPatient({
      button: document.getElementById("b"), panel: document.getElementById("p"),
      getCtx: () => ({ data: st.base, slug: st.slug, patient: st.data === st.base ? st.patient : st.realPatient }),
      onPatient: () => {}, onRevert: () => {}, healthFn: async () => ({ ok: true }),
    });
    // the user invented a patient: synthetic index 25 on a copy of the data
    const c = withCustomPatient(st.base, resp());
    st.realPatient = st.patient; st.data = c.data; st.patient = c.index;
    return { st, cp };
  }

  it("Train & replay with a 15-patient run does not throw and lands on the real patient", async () => {
    const { st, cp } = setup();
    await cp.setOpen(true);
    expect(st.patient).toBe(25);
    const run = mk(15, { custom_run: true, run_id: "abc" });
    expect(() => adoptData(st, run, { resetCustom: () => cp.reset() })).not.toThrow();
    expect(st.patient).toBe(3);
    expect(st.realPatient).toBe(3);
    expect(st.data).toBe(run);
    expect(st.base).toBe(run);
  });

  it("clamps when the real patient does not exist in the new data, and the panel still builds", async () => {
    const { st, cp } = setup();
    st.realPatient = 20;
    await cp.setOpen(true);
    expect(() => adoptData(st, mk(15), { resetCustom: () => cp.reset() })).not.toThrow();
    expect(st.patient).toBe(14);
    expect(document.querySelectorAll("#p input[type=range]").length).toBe(3);
  });

  it("customPatient.build guards an out-of-range index", async () => {
    document.body.innerHTML = `<button id="b"></button><div id="p"></div>`;
    const cp = mountCustomPatient({
      button: document.getElementById("b"), panel: document.getElementById("p"),
      getCtx: () => ({ data: mk(15), slug: "heart", patient: 25 }),
      onPatient: () => {}, onRevert: () => {}, healthFn: async () => ({ ok: true }),
    });
    await expect(cp.setOpen(true)).resolves.toBeUndefined();
    const s = document.querySelector("#p input[type=range]");
    expect(s.max).toBe("3.15"); // M6: the top of the slider reaches pi
  });
});

describe("I1: picking a real patient while a custom one is active", () => {
  it("goes to the chosen patient, not the previous real one", () => {
    const base = mk(25);
    const c = withCustomPatient(base, resp());
    const st = { base, data: c.data, patient: c.index, realPatient: 3 };
    const leaveCustom = () => { st.data = st.base; st.patient = st.realPatient; };
    expect(choosePatient(st, 7, { leaveCustom })).toBe(7);
    expect(st.patient).toBe(7);
    expect(st.data).toBe(base);
  });
});

describe("QPU selection follows the active data", () => {
  const index = [{ slug: "heart", label: "Heart Disease" }];
  it("static data: no runId, static settings", () => {
    const d = mk(25);
    const s = qpuSelection({ index, slug: "heart", base: d, data: d, patient: 4 });
    expect(s).toMatchObject({ slug: "heart", patient: 4, runId: null, disabled: null, label: "Heart Disease" });
    expect(s.settings.kernel_bandwidth).toBe(0.1);
    expect(s.kRow).toBe(d.qsvc.kernel_test[4]);
  });
  it("Custom run: passes run_id and the run's settings", () => {
    const d = mk(15, { custom_run: true, run_id: "abc123", settings: { n_qubits: 4, kernel_bandwidth: 0.3 } });
    const s = qpuSelection({ index, slug: "heart", base: d, data: d, patient: 2 });
    expect(s.runId).toBe("abc123");
    expect(s.settings).toEqual({ n_qubits: 4, kernel_bandwidth: 0.3 });
    expect(s.disabled).toBeNull();
  });
  it("fixtures and custom slider patients are disabled with a clear message", () => {
    const f = mk(15, { custom_run: true, run_id: "fixture-q2" });
    expect(qpuSelection({ index, slug: "heart", base: f, data: f, patient: 0 })).toMatchObject({ disabled: FIXTURE_QPU_MSG, runId: null });
    const base = mk(25);
    const c = withCustomPatient(base, resp());
    expect(qpuSelection({ index, slug: "heart", base, data: c.data, patient: c.index })).toMatchObject({ disabled: CUSTOM_QPU_MSG });
  });
});

describe("I5: scene 8 messages for custom patients / Custom runs", () => {
  const q = { metrics: { "ROC-AUC": 0.9 }, architecture: {}, test: Array.from({ length: 25 }, (_, i) => ({ label: i % 2, pred: 0 })) };
  it("custom patient -> pick a real patient", () => {
    const c = withCustomPatient(mk(25), resp());
    expect(quhamaPatient(q, c.data, c.index)).toEqual({ ok: false, msg: QUHAMA_MSG.custom });
  });
  it("Custom run -> Reset to precomputed (not 're-export needed')", () => {
    const r = quhamaPatient(q, mk(15, { custom_run: true }), 0);
    expect(r).toEqual({ ok: false, msg: QUHAMA_MSG.customRun });
    expect(r.msg).not.toMatch(/re-export/);
    const e = { leaderboard: [{}], approaches: Object.fromEntries(["quhama", "sandwich", "ckta"].map((k) => [k, { test: [], architecture: {}, metrics: {} }])) };
    expect(eliteCheck(e, mk(15, { custom_run: true })).msg).toMatch(/Reset to precomputed/);
  });
  it("genuine mismatch on static data still says re-export", () => {
    expect(quhamaPatient(q, mk(20), 0).msg).toMatch(/re-export/);
  });
});
