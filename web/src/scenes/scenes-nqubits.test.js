// @vitest-environment jsdom
// Builds scenes 1-3 for 2, 3 and 4 qubits (real fixtures + shipped static data) and steps through their timelines.
import { describe, it, expect, beforeAll } from "vitest";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import scene1, { T as T1 } from "./scene1.js";
import scene2 from "./scene2.js";
import scene3 from "./scene3.js";
import { legend1, legend2, legend3 } from "./legends.js";

beforeAll(() => { // jsdom has no 2D canvas: a minimal stub is enough for text/glow sprites
  const ctx2d = new Proxy({}, {
    get: (_, k) => (k === "measureText" ? (s) => ({ width: 30 * String(s).length })
      : k === "createRadialGradient" ? () => ({ addColorStop() {} }) : () => {}),
    set: () => true,
  });
  HTMLCanvasElement.prototype.getContext = () => ctx2d;
});

const HERE = dirname(fileURLToPath(import.meta.url));
const load = (rel) => JSON.parse(readFileSync(join(HERE, rel), "utf8"));
const DATA = {
  2: load("../../test-fixtures/vertebral_q2.json"),
  3: load("../../public/data/vertebral_column.json"),
  4: load("../../test-fixtures/vertebral_q4.json"),
};
const TIMES = [0, 0.5, 1, 2, 3, 4, 4.6, 5, 6, 7, 7.5, 8, 9, 10, 12];
const captions = (factory, data, patient) => {
  const s = factory({ data, patient });
  const out = TIMES.map((t) => s.update(t));
  s.dispose();
  return { s, out };
};
const meshCount = (obj, pred) => { let n = 0; obj.traverse((o) => { if (pred(o)) n++; }); return n; };

describe.each([2, 3, 4])("%i qubits", (n) => {
  const data = DATA[n];
  it("fixture/static data has n angles per patient", () => {
    expect(data.test[0].angles).toHaveLength(n);
  });
  it.each([0, 1, 2])("scenes 1-3 build and run for patient %i", (pi) => {
    for (const f of [scene1, scene2, scene3]) {
      const { out } = captions(f, data, pi);
      out.forEach((c) => { expect(typeof c).toBe("string"); expect(c.length).toBeGreaterThan(10); });
      out.forEach((c) => expect(c).not.toMatch(/undefined|NaN/));
    }
  });
  it("scene 1: n Bloch spheres, captions list n angles and n components", () => {
    const s = scene1({ data, patient: 0 });
    // one equator torus per Bloch sphere
    expect(meshCount(s.scene, (o) => o.geometry && o.geometry.type === "TorusGeometry")).toBe(n);
    const enc = s.update(T1.phase[1]), pca = s.update((T1.orbs[1] + T1.bloch[0]) / 2); // timeline from scene1 (pacing)
    for (let k = 0; k < n; k++) expect(enc).toContain(`x${"₁₂₃₄"[k]} = `);
    expect(enc).not.toContain(`x${"₁₂₃₄₅"[n]} = `);
    expect(pca).toContain(`into ${n} principal components`);
    expect(s.legend).toEqual(legend1(n));
    s.dispose();
  });
  it("scene 3: n spheres per row (2 rows)", () => {
    const s = scene3({ data, patient: 0 });
    expect(meshCount(s.scene, (o) => o.geometry && o.geometry.type === "TorusGeometry")).toBe(2 * n);
    expect(s.legend).toEqual(legend3(n));
    s.dispose();
  });
  it("scene 2 legend follows n", () => {
    const s = scene2({ data, patient: 0 });
    expect(s.legend).toEqual(legend2(n));
    s.dispose();
  });
});

describe("honest captions", () => {
  it("2 qubits: scene 2 says 2 qubits → 2D and draws everything at z = 0", () => {
    const s = scene2({ data: DATA[2], patient: 0 });
    const caps = TIMES.map((t) => s.update(t)).join(" ");
    expect(caps).toContain("2 qubits → 2D");
    expect(caps).toContain(`${DATA[2].rbf.grid_size}×${DATA[2].rbf.grid_size} grid`);
    expect(caps).not.toContain("x₃");
    expect(caps).toContain("the curve grows"); // this fixture's RBF boundary crosses 0 inside the plane
    let dots = 0;
    s.scene.traverse((o) => {
      if (o.geometry && o.geometry.type === "SphereGeometry") { dots++; expect(Math.abs(o.position.z)).toBeLessThan(1e-9); }
    });
    expect(dots).toBe(DATA[2].train.angles.length + 1); // training dots + your patient
    s.scene.updateMatrixWorld(true);
    s.scene.traverse((o) => {
      const pos = o.geometry && o.geometry.attributes && o.geometry.attributes.position;
      if (!pos || o.type === "Sprite" || !o.visible || o.geometry.type !== "BufferGeometry") return;
      for (let i = 0; i < pos.count; i++) expect(Math.abs(pos.getZ(i))).toBeLessThan(1e-6);
    });
    s.dispose();
  });
  it("4 qubits: scene 2 states the x₄ slice value and that the 4th dimension isn't drawn", () => {
    const s = scene2({ data: DATA[4], patient: 0 });
    const caps = TIMES.map((t) => s.update(t)).join(" ");
    expect(caps).toContain(`showing a 3D slice at x₄ = ${DATA[4].rbf.slice.value.toFixed(2)}`);
    expect(caps).toContain("the 4th dimension isn't drawn");
    expect(caps).toMatch(/·x₄/); // the full 4D equation is shown
    s.dispose();
  });
  it("2/4 qubits: scene 3 notes the dropped/flat dimension; K values are the real ones", () => {
    for (const n of [2, 4]) {
      const d = DATA[n];
      const s = scene3({ data: d, patient: 0 });
      const caps = TIMES.map((t) => s.update(t)).join(" ");
      expect(caps).toContain(n === 2 ? "2 qubits → 2D" : "x₄ isn't drawn");
      const K = Math.max(...d.qsvc.kernel_test[0]);
      expect(caps).toContain(`<b>${K.toFixed(3)}</b>`);
      s.dispose();
    }
  });
});
