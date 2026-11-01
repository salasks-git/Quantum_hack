import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { rowLayout, pos3, sectionPlane, contourSegments, clipLineToSquare, xName } from "./common.js";
import { legend1, legend2, legend3, LEGEND_1, LEGEND_2, LEGEND_3 } from "./legends.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const fixture = (n) => JSON.parse(readFileSync(join(HERE, `../../test-fixtures/vertebral_q${n}.json`), "utf8"));
const close = (a, b) => a.forEach((v, i) => expect(v).toBeCloseTo(b[i], 9));

describe("rowLayout (sphere positions for n = 2..4)", () => {
  it("3 qubits is exactly the shipped layout", () => {
    expect(rowLayout(3, 2.5, 0.95)).toEqual({ xs: [-2.5, 0, 2.5], scale: 1 });
    expect(rowLayout(3, 1.9, 0.62)).toEqual({ xs: [-1.9, 0, 1.9], scale: 1 });
  });
  it("2 qubits keep full size, centred", () => {
    expect(rowLayout(2, 2.5, 0.95)).toEqual({ xs: [-1.25, 1.25], scale: 1 });
  });
  it("4 qubits shrink to the 3-qubit width, evenly spaced and symmetric", () => {
    const { xs, scale } = rowLayout(4, 2.5, 0.95);
    expect(scale).toBeCloseTo(6.9 / 9.4, 12);
    expect(xs[3] - xs[0] + 2 * 0.95 * scale).toBeCloseTo(6.9, 12);
    close(xs, [-3.75 * scale, -1.25 * scale, 1.25 * scale, 3.75 * scale]);
    // spheres never overlap: gap between centres > 2 radii
    expect(xs[1] - xs[0]).toBeGreaterThan(2 * 0.95 * scale);
  });
});

describe("pos3 (drawn positions drop / pad dimensions)", () => {
  it("3 angles unchanged", () => expect(pos3([0.1, 0.2, 0.3])).toEqual([0.1, 0.2, 0.3]));
  it("2 angles -> flat at the centre (world z = 0)", () => expect(pos3([0.1, 0.2])).toEqual([0.1, 0.2, Math.PI / 2]));
  it("custom fill", () => expect(pos3([0.1, 0.2], 7)).toEqual([0.1, 0.2, 7]));
  it("4 angles -> the first 3", () => expect(pos3([0.1, 0.2, 0.3, 0.4])).toEqual([0.1, 0.2, 0.3]));
});

describe("sectionPlane (4D -> 3D plane section)", () => {
  it("hand-computed: 1·x1 + 2·x2 + 3·x3 + 4·x4 + 5 = 0 at x4 = 0.5 -> x1 + 2x2 + 3x3 + 7 = 0", () => {
    expect(sectionPlane([1, 2, 3, 4], 5, { 3: 0.5 })).toEqual({ coef: [1, 2, 3], intercept: 7 });
  });
  it("a point on the 4D plane with x4 = slice lies on the section, and vice versa", () => {
    const coef = [0.7, -1.3, 0.4, 2.1], b = -0.9, x4 = 1.35;
    const s = sectionPlane(coef, b, { 3: x4 });
    const x = [0.2, 1.1, 0]; // solve x3 so the 3D section holds
    x[2] = -(s.coef[0] * x[0] + s.coef[1] * x[1] + s.intercept) / s.coef[2];
    const full = coef[0] * x[0] + coef[1] * x[1] + coef[2] * x[2] + coef[3] * x4 + b;
    expect(full).toBeCloseTo(0, 12);
  });
  it("no fixed dims = identity (3 qubits)", () => {
    expect(sectionPlane([1, 2, 3], -4)).toEqual({ coef: [1, 2, 3], intercept: -4 });
  });
});

describe("contourSegments (2D marching squares)", () => {
  const G = 5, axis = [0, 1, 2, 3, 4];
  const grid = (f) => Array.from({ length: G * G }, (_, n) => f(axis[Math.floor(n / G)], axis[n % G]));
  it("v = x1 − 1.5: a straight zero line at x1 = 1.5 covering the whole x2 range", () => {
    const segs = contourSegments(grid((x) => x - 1.5), G, axis);
    expect(segs).toHaveLength(4);
    segs.forEach(([x1, , x2]) => { expect(x1).toBeCloseTo(1.5, 12); expect(x2).toBeCloseTo(1.5, 12); });
    const ys = segs.flatMap(([, y1, , y2]) => [y1, y2]);
    expect(Math.min(...ys)).toBe(0);
    expect(Math.max(...ys)).toBe(4);
  });
  it("v = x1 + x2 − 4.5: every endpoint lies on x1 + x2 = 4.5", () => {
    const segs = contourSegments(grid((x, y) => x + y - 4.5), G, axis);
    expect(segs.length).toBeGreaterThan(0);
    segs.forEach(([a, b, c, d]) => { expect(a + b).toBeCloseTo(4.5, 12); expect(c + d).toBeCloseTo(4.5, 12); });
  });
  it("level shifts the line; no crossing -> no segments", () => {
    const segs = contourSegments(grid((x) => x), G, axis, 2.5);
    segs.forEach(([x1, , x2]) => { expect(x1).toBeCloseTo(2.5, 12); expect(x2).toBeCloseTo(2.5, 12); });
    expect(contourSegments(grid(() => 1), G, axis)).toEqual([]);
  });
  it("saddle cell gives two segments", () => {
    expect(contourSegments([1, -1, -1, 1], 2, [0, 1])).toHaveLength(2);
  });
});

describe("clipLineToSquare (2-qubit linear boundary)", () => {
  it("vertical line x = 0.5", () => {
    const seg = clipLineToSquare([1, 0], 0.5, -1, 1);
    expect(seg.map((p) => p[0])).toEqual([0.5, 0.5]);
    expect(seg.map((p) => p[1]).sort()).toEqual([-1, 1]);
  });
  it("diagonal through two corners", () => {
    const s = Math.SQRT1_2;
    const seg = clipLineToSquare([s, -s], 0, -1, 1);
    expect(seg.map((p) => p.join()).sort()).toEqual(["-1,-1", "1,1"]);
  });
  it("misses the square -> null", () => expect(clipLineToSquare([1, 0], 3, -1, 1)).toBeNull());
});

describe("legends per qubit count", () => {
  it("3 qubits = the shipped legends", () => {
    expect(legend1(3)).toEqual(LEGEND_1);
    expect(legend2(3)).toEqual(LEGEND_2);
    expect(legend3(3)).toEqual(LEGEND_3);
  });
  it("2 and 4 qubits name the right number of angles", () => {
    expect(legend1(2).map((e) => e.symbol)).toContain("x₁, x₂");
    expect(legend1(4).map((e) => e.symbol)).toContain("x₁, x₂, x₃, x₄");
    expect(JSON.stringify(legend2(2))).toContain("2 qubits → 2D");
    expect(JSON.stringify(legend2(4))).toContain("3D slice");
    expect(JSON.stringify(legend3(4))).toContain("all 4 qubits");
    expect(xName(3)).toBe("x₄");
  });
});

describe("2/4-qubit fixtures (real live-run exports) match the data contract", () => {
  it("q2: G×G grid, 2 angles, 2 coefs", () => {
    const d = fixture(2);
    expect(d.settings.n_qubits).toBe(2);
    expect(d.rbf.grid_dims).toBe(2);
    expect(d.rbf.values).toHaveLength(d.rbf.grid_size ** 2);
    expect(d.linear.coef).toHaveLength(2);
    expect(d.train.angles.every((a) => a.length === 2)).toBe(true);
    expect(d.test.every((p) => p.angles.length === 2 && p.pca.length === 2)).toBe(true);
  });
  it("q4: G³ slice at x4 = slice.value, 4 angles, 4 coefs", () => {
    const d = fixture(4);
    expect(d.settings.n_qubits).toBe(4);
    expect(d.rbf.grid_dims).toBe(3);
    expect(d.rbf.slice.dim).toBe(3);
    expect(typeof d.rbf.slice.value).toBe("number");
    expect(d.rbf.values).toHaveLength(d.rbf.grid_size ** 3);
    expect(d.linear.coef).toHaveLength(4);
    expect(d.train.angles.every((a) => a.length === 4)).toBe(true);
  });
});
