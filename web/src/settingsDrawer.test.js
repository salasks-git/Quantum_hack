// @vitest-environment jsdom
import { describe, it, expect, vi } from "vitest";
import { mountSettingsDrawer, OPTIONS, defaultsFrom, badgeText } from "./settingsDrawer.js";

const staticData = { settings: { n_qubits: 3, feature_map: "ZZFeatureMap", reps: 2, entanglement: "linear", vqc_iter: 35, seed: 42, kernel_bandwidth: 0.1 } };
const optVals = (sel) => [...sel.options].map((o) => o.value);

function setup(over = {}) {
  document.body.innerHTML = '<button id="b"></button><div id="c"></div><span id="g" hidden></span>';
  const live = {
    liveHealth: vi.fn(async () => ({ up: true, live: true })),
    startTrain: vi.fn(async () => ({ run_id: "r1", state: "queued" })),
    pollTrain: vi.fn(async (id, { onUpdate }) => { onUpdate({ stage: "VQC training", progress: 0.5, elapsed_s: 1.2 }); return { state: "done", elapsed_s: 2.5 }; }),
    loadRun: vi.fn(async () => ({ custom_run: true, settings: { n_qubits: 4, feature_map: "ZZFeatureMap", reps: 2, entanglement: "linear", vqc_iter: 15, kernel_bandwidth: 0.3, max_samples: 60 } })),
    ...(over.live || {}),
  };
  const onApply = vi.fn(), onReset = vi.fn(), onToast = vi.fn();
  const d = mountSettingsDrawer({ button: document.getElementById("b"), container: document.getElementById("c"), badge: document.getElementById("g"), getData: () => staticData, getSlug: () => "vertebral_column", getLabel: () => "Vertebral", onApply, onReset, onToast, live });
  return { d, live, onApply, onReset, onToast };
}
const tick = () => new Promise((r) => setTimeout(r, 0));

describe("settings drawer", () => {
  it("renders the exact option lists", async () => {
    const { d } = setup(); await tick();
    const c = d._ctrls;
    expect(optVals(c.n_qubits)).toEqual(["2", "3", "4"]);
    expect(optVals(c.feature_map)).toEqual(OPTIONS.feature_map);
    expect(optVals(c.entanglement)).toEqual(["linear", "full", "circular"]);
    expect([c.max_samples.min, c.max_samples.max, c.max_samples.step]).toEqual(["60", "180", "20"]);
    expect([c.reps.min, c.reps.max]).toEqual(["1", "3"]);
    expect([c.vqc_iter.min, c.vqc_iter.max, c.vqc_iter.step]).toEqual(["15", "75", "5"]);
    expect(OPTIONS.kernel_bandwidth).toEqual([0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]);
    expect(c.kernel_bandwidth.max).toBe("6");
  });
  it("defaults come from the loaded data (bandwidth 1.0 if absent)", () => {
    expect(defaultsFrom(staticData).kernel_bandwidth).toBe(0.1);
    expect(defaultsFrom({ settings: { n_qubits: 3 } }).kernel_bandwidth).toBe(1.0);
  });
  it("submit payload has every setting with the right types", async () => {
    const { d, live, onApply } = setup(); await tick();
    d._ctrls.n_qubits.value = "4"; d._ctrls.n_qubits.dispatchEvent(new Event("change"));
    d._ctrls.max_samples.value = "60"; d._ctrls.max_samples.dispatchEvent(new Event("input"));
    d._ctrls.kernel_bandwidth.value = "3"; d._ctrls.kernel_bandwidth.dispatchEvent(new Event("input"));
    d._ctrls.vqc_iter.value = "15"; d._ctrls.vqc_iter.dispatchEvent(new Event("input"));
    await d.run();
    expect(live.startTrain).toHaveBeenCalledWith({ slug: "vertebral_column", max_samples: 60, n_qubits: 4, feature_map: "ZZFeatureMap", reps: 2, entanglement: "linear", kernel_bandwidth: 0.3, vqc_iter: 15 });
    expect(onApply).toHaveBeenCalledTimes(1);
    expect(document.getElementById("g").hidden).toBe(false);
    expect(document.getElementById("g").textContent).toContain("Custom run · 4 qubits · s=0.3");
  });
  it("toasts for non-3-qubit runs", async () => {
    const { d, onToast } = setup(); await tick();
    await d.run();
    expect(onToast).toHaveBeenCalledWith(expect.stringContaining("scenes 1"));
  });
  it("Reset restores static data via callback and clears badge", async () => {
    const { d, onReset } = setup(); await tick();
    await d.run();
    d._resetBtn.click(); await tick();
    expect(onReset).toHaveBeenCalledTimes(1);
    expect(document.getElementById("g").hidden).toBe(true);
    expect(d.isCustom()).toBe(false);
  });
  it("409 shows message; errors show their text", async () => {
    let e = Object.assign(new Error("x"), { status: 409 });
    const { d } = setup({ live: { startTrain: async () => { throw e; } } }); await tick();
    await d.run();
    expect(d._msg.textContent).toBe("Another training is running");
    e = Object.assign(new Error("Unknown dataset slug."), { status: 404 });
    await d.run();
    expect(d._msg.textContent).toBe("Unknown dataset slug.");
  });
  it("disabled with how-to message when helper down", async () => {
    const { d } = setup({ live: { liveHealth: async () => ({ up: false, live: false }) } }); await tick(); await tick();
    expect(d._goBtn.disabled).toBe(true);
    expect(d._ctrls.n_qubits.disabled).toBe(true);
    expect(d._msg.textContent).toContain("scripts/qpu_server.py");
  });
  it("badgeText is compact", () => {
    expect(badgeText({ n_qubits: 2, kernel_bandwidth: 0.5, feature_map: "PauliFeatureMap (Z, YY, ZX)", entanglement: "full", reps: 1, max_samples: 80, vqc_iter: 25 })).toContain("2 qubits · s=0.5 · Pauli");
  });
});
