import { describe, it, expect } from "vitest";
import { renderLegend, mountLegendToggle } from "./legend.js";
import { LEGEND_1, LEGEND_2, LEGEND_3, LEGEND_4, LEGEND_5, LEGEND_6, LEGEND_BANDWIDTH, LEGEND_QUHAMA, LEGEND_SANDWICH, LEGEND_CKTA } from "./scenes/legends.js";

const fakeEl = () => ({ innerHTML: "", hidden: false, _l: {}, classList: { toggle() {} }, setAttribute() {}, addEventListener(n, f) { this._l[n] = f; } });

describe("renderLegend", () => {
  it("escapes HTML", () => {
    const c = fakeEl();
    renderLegend(c, [{ symbol: "<b>x</b>", text: "a <i>b</i> & c" }]);
    expect(c.innerHTML).not.toContain("<b>");
    expect(c.innerHTML).toContain("&lt;b&gt;x&lt;/b&gt;");
    expect(c.innerHTML).toContain("&amp;");
  });
  it("renders a swatch dot only when given", () => {
    const c = fakeEl();
    renderLegend(c, [{ symbol: "a", text: "t", swatch: "#4ade80" }, { symbol: "b", text: "u" }]);
    expect(c.innerHTML.match(/lg-dot/g).length).toBe(1);
    expect(c.innerHTML).toContain("background:#4ade80");
  });
  it("rejects unsafe swatch values", () => {
    const c = fakeEl();
    renderLegend(c, [{ symbol: "a", text: "t", swatch: "red;}</style><script>" }]);
    expect(c.innerHTML).not.toContain("lg-dot");
  });
});

describe("scene legends (scenes 1-6 + Kernel Bandwidth Lab)", () => {
  it.each([[LEGEND_1], [LEGEND_2], [LEGEND_3], [LEGEND_4], [LEGEND_5], [LEGEND_6], [LEGEND_BANDWIDTH]])("is non-empty with string symbol/text", (L) => {
    expect(L.length).toBeGreaterThan(0);
    for (const it of L) { expect(typeof it.symbol).toBe("string"); expect(typeof it.text).toBe("string"); }
  });
});

describe("scene 6 legend", () => {
  it("has the verbatim QPU entries", () => {
    expect(LEGEND_6.map((x) => x.symbol)).toEqual([
      "QPU", "transpile", "sx, x", "rz", "ECR", "depth", "µs / ns", "shots", "estimate",
      "CPU (measured)", "queue", "fidelity (hardware)", "fidelity (ideal)", "noise gap",
      "train / inference", "log scale", "F2 score", "efficiency frontier",
    ]);
    expect(LEGEND_6.find((x) => x.symbol === "noise gap").text).toBe("Ideal − hardware: how much real-chip noise changed the answer.");
  });
});

describe("QuHaMA scene legend", () => {
  it("covers the architecture terms with string symbol/text", () => {
    const syms = LEGEND_QUHAMA.map((it) => it.symbol);
    for (const want of ["encoder", "latent space (k = qubits)", "classical head", "quantum branch", "gating network (per-patient adaptive weights)", "blended prob"]) {
      expect(syms).toContain(want);
    }
    for (const it of LEGEND_QUHAMA) { expect(typeof it.symbol).toBe("string"); expect(typeof it.text).toBe("string"); }
  });
});

describe("Sandwich and CKTA scene legends (steps 9-10)", () => {
  it.each([[LEGEND_SANDWICH], [LEGEND_CKTA]])("is non-empty with string symbol/text", (L) => {
    expect(L.length).toBeGreaterThan(0);
    for (const it of L) { expect(typeof it.symbol).toBe("string"); expect(typeof it.text).toBe("string"); }
  });
  it("Sandwich covers the plain-language architecture terms", () => {
    const syms = LEGEND_SANDWICH.map((it) => it.symbol).join(" | ");
    for (const want of ["classical encoder", "quantum layer", "expectation value", "classical head"]) expect(syms).toContain(want);
  });
  it("CKTA covers alignment and the F2 leaderboard", () => {
    const syms = LEGEND_CKTA.map((it) => it.symbol).join(" | ");
    for (const want of ["kernel-target alignment", "alignment score", "F2 leaderboard"]) expect(syms).toContain(want);
  });
});

describe("mountLegendToggle", () => {
  it("defaults open, toggles, tolerates missing storage", () => {
    const b = fakeEl(), p = fakeEl();
    const t = mountLegendToggle(b, p, { storageKey: "k" });
    expect(t.isOpen()).toBe(true);
    t.setOpen(false);
    expect(t.isOpen()).toBe(false);
    expect(p.hidden).toBe(true);
    b._l.click();
    expect(t.isOpen()).toBe(true);
  });
});
