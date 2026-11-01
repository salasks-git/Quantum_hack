// @vitest-environment jsdom
// Tour pacing: dwell after every finished chart, PC orbs linger, bars grow at their own pace.
import { describe, it, expect, beforeAll } from "vitest";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import scene1, { T as T1 } from "./scene1.js";
import scene2 from "./scene2.js";
import scene3 from "./scene3.js";
import scene4 from "./scene4.js";
import scene5 from "./scene5.js";
import scene6 from "./scene6.js";
import { E, E_STOPS, E_DURATION } from "./eliteParts.js";

beforeAll(() => {
  const ctx2d = new Proxy({}, {
    get: (_, k) => (k === "measureText" ? (s) => ({ width: 30 * String(s).length })
      : k === "createRadialGradient" || k === "createLinearGradient" ? () => ({ addColorStop() {} }) : () => {}),
    set: () => true,
  });
  HTMLCanvasElement.prototype.getContext = () => ctx2d;
});

const HERE = dirname(fileURLToPath(import.meta.url));
const data = JSON.parse(readFileSync(join(HERE, "../../public/data/heart_disease_statlog.json"), "utf8"));

describe("scene 1: PC orbs linger", () => {
  it("standardised bars stay >= 2 s before merging; orbs fully formed >= 3 s before the Bloch morph", () => {
    expect(T1.merge[0] - T1.std[1]).toBeGreaterThanOrEqual(2);
    expect(T1.bloch[0] - T1.orbs[1]).toBeGreaterThanOrEqual(3);
  });
  it("stops right after the orbs form, and the orbs (with PC labels) are visible through the gap", () => {
    const s = scene1({ data, patient: 0 });
    expect(s.stops).toContain(T1.orbs[1]);
    expect(s.stops.at(-1)).toBeLessThan(s.duration);
    // find orb groups: children carrying a "PC" label sprite are created per component
    for (const t of [T1.orbs[1], T1.orbs[1] + 1.5, T1.bloch[0] - 0.01]) {
      s.update(t);
      const visibleOrbs = s.scene.children.filter((o) => o.visible && o.scale && Math.abs(o.scale.x - 1) < 1e-6 && o.position.y === 0.6);
      expect(visibleOrbs.length).toBeGreaterThanOrEqual(data.settings.n_qubits);
    }
    s.dispose();
  });
});

describe("stops land after a chart is complete, before the next morph", () => {
  it.each([["scene2", scene2], ["scene3", scene3], ["scene4", scene4], ["scene5", scene5], ["scene6", scene6]])(
    "%s: stops sorted, unique and inside the timeline", (_, f) => {
      const s = f({ data, patient: 0 });
      const st = s.stops;
      expect(st.length).toBeGreaterThan(0);
      expect([...st].sort((a, b) => a - b)).toEqual(st);
      expect(new Set(st).size).toBe(st.length);
      expect(st.at(-1)).toBeLessThan(s.duration);
      s.dispose();
    });
  it("scene 5: no stop in the middle of the column growth (one dwell once all columns are grown)", () => {
    const s = scene5({ data, patient: 0 });
    expect(s.stops).toHaveLength(2);
    s.dispose();
  });
  it("scene 8 elite: stops after cards, leaderboard and patient verdicts, never inside a growth window", () => {
    expect(E_STOPS).toEqual([...E_STOPS].sort((a, b) => a - b));
    expect(E_STOPS[0]).toBeGreaterThanOrEqual(E.cVals[1]);
    expect(E_STOPS[0]).toBeLessThan(E.cardsOut[0]);
    expect(E_STOPS[1]).toBeGreaterThanOrEqual(E.leader[1]);
    expect(E_STOPS[2]).toBeGreaterThanOrEqual(E.patient[1]);
    for (const s of E_STOPS) expect(s > E.bars[0] && s < E.bars[1]).toBe(false);
    expect(E_STOPS.at(-1)).toBeLessThan(E_DURATION);
  });
});
