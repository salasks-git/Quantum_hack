import { describe, it, expect } from "vitest";
import { renderLegend, mountLegendToggle } from "./legend.js";
import { LEGEND_1, LEGEND_2, LEGEND_3, LEGEND_4, LEGEND_5, LEGEND_BANDWIDTH } from "./scenes/legends.js";

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

describe("scene legends (scenes 1-5 + Kernel Bandwidth Lab)", () => {
  it.each([[LEGEND_1], [LEGEND_2], [LEGEND_3], [LEGEND_4], [LEGEND_5], [LEGEND_BANDWIDTH]])("is non-empty with string symbol/text", (L) => {
    expect(L.length).toBeGreaterThan(0);
    for (const it of L) { expect(typeof it.symbol).toBe("string"); expect(typeof it.text).toBe("string"); }
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
