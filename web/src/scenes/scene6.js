import * as THREE from "three";
import { LEGEND_6 } from "./legends.js";
import { PALETTE, ease, seg, textSprite, glowSprite, setOpacity, disposeTree, addLights } from "./common.js";
import { MODEL_KEYS, MODEL_LABELS } from "../verdict.js";
import { clinicalScores } from "../clinical.js";
import { fmtDuration, paretoFront, logScale, tickLabel } from "../timing.js";

// Scene 6: from simulator to real QPU.
// Part 1: the transpiled feature map U(x) (IBM native gates rz / sx / x / ecr from data.qpu.gates) slides in
//         along 3 qubit wires as 3D blocks, then an execution cursor sweeps through it.
// Part 2: log-scale time race: QSVC on IBM QPU (estimate) vs this CPU (measured) vs IBM cloud queue (typical).
// Part 3: measured compute time of all 5 models (train solid + inference lighter, log axis).
// Part 4: efficiency frontier: F2 vs total time, Pareto-optimal models connected in gold.

const GATE_COL = { sx: PALETTE.classical, x: PALETTE.quantum, rz: 0x94a3b8, ecr: 0xfbbf24 };
const WIRE_Y = [1.55, 0.75, -0.05]; // q0 top
const X0 = -2.3, X1 = 4.4; // first / last gate layer
const ENTRY = X0 - 0.35; // gates fade in as they cross this x
const CUBE = 0.13;

// Timeline (seconds)
const WIRES_IN = 0.5;
const SLIDE0 = 0.3, SLIDE1 = 3.3; // circuit train slides in
const SWEEP0 = 3.6, SWEEP1 = 6.3; // execution cursor
const MEAS_AT = SWEEP1 + 0.05;
const P1_STOP = 6.75;
const P1_OUT0 = 7.0, P1_OUT1 = 7.6;
const P2_IN0 = 7.4, P2_IN1 = 7.9;
const RACE0 = 8.1, RACE1 = 12.5; // log clock sweeps lo -> hi at constant log speed
const RESULT_AT = RACE1 + 0.15;
const P2_OUT0 = RESULT_AT + 0.55, P2_OUT1 = P2_OUT0 + 0.6;
const P3_IN0 = P2_OUT0 + 0.4, P3_IN1 = P3_IN0 + 0.5;
const TB0 = P3_IN1 + 0.2, TB1 = TB0 + 2.4; // log clock for the timing bars
const P3_RESULT = TB1 + 0.15;
const P3_STOP = P3_RESULT + 0.4;
const P3_OUT0 = P3_STOP + 0.25, P3_OUT1 = P3_OUT0 + 0.6;
const P4_IN0 = P3_OUT0 + 0.4, P4_IN1 = P4_IN0 + 0.5;
const DOT0 = P4_IN1 + 0.1, DOT_STEP = 0.35, DOT_FADE = 0.4; // markers appear fastest -> slowest
const LINE0 = DOT0 + 4 * DOT_STEP + DOT_FADE + 0.15, LINE1 = LINE0 + 1.2;
const P4_RESULT = LINE1 + 0.1;
const P4_STOP = P4_RESULT + 0.4;
export const duration = P4_RESULT + 1.7;

const AX0 = -2.0, AX1 = 4.1, AXIS_Y = -0.75;
const P2_LIFT = 0.3; // keeps the axis labels clear of the live panel (bottom-right) at 1280 px
const BAR_Y = [1.55, 0.75, -0.05], BAR_H = 0.42, BAR_D = 0.3;

// parts 3 and 4 share one log time axis and the same screen area as part 2
const T_BAR_Y = [1.55, 1.05, 0.55, 0.05, -0.45], T_BAR_H = 0.26, T_BAR_D = 0.2;
const SC_TOP = 1.85; // scatter: y = 0 at AXIS_Y, y = 1 at SC_TOP
const MODEL_COL = { linear: 0x38bdf8, rbf: 0x22d3ee, qsvc: 0xa78bfa, vqc: 0xf0abfc, hybrid: 0x4ade80 };
const FRONT_COL = 0xfbbf24;

const hex = (c) => "#" + c.toString(16).padStart(6, "0");
const int = (n) => Math.round(n).toLocaleString("en-US");

/** ASAP layering of a gate list on n wires: layer[i] for each gate and the number of layers (= depth). */
export function layerGates(gates, nQubits = 3) {
  const free = new Array(nQubits).fill(0);
  const layer = gates.map((g) => {
    const L = Math.max(...g.qubits.map((q) => free[q]));
    g.qubits.forEach((q) => { free[q] = L + 1; });
    return L;
  });
  return { layer, depth: Math.max(0, ...free) };
}

/** Log-axis range covering all values: [10^floor(log10 min) capped at 1 s, 10^ceil(log10 max)]. */
export function logRange(values) {
  const lo = Math.min(1, 10 ** Math.floor(Math.log10(Math.min(...values))));
  const hi = 10 ** Math.ceil(Math.log10(Math.max(...values) * 1.0001));
  return { lo, hi };
}

export function fmtTime(s) {
  if (s < 10) return `${s.toFixed(2)} s`;
  if (s < 60) return `${s.toFixed(1)} s`;
  if (s < 3600) return `${(s / 60).toFixed(1)} min`;
  return `${(s / 3600).toFixed(1)} h`;
}
const fmtRatio = (r) => (r >= 10 ? int(r) : r.toFixed(1)) + "×";

export default function createScene(ctx) {
  const { data } = ctx;
  const q = data.qpu;
  const tm = data.timings;
  const scene = new THREE.Scene();
  addLights(scene);

  // ======================= Part 1: native gates on 3 wires =======================
  const p1 = new THREE.Group();
  scene.add(p1);
  const p1static = new THREE.Group(); // wires + labels (faded as one)
  p1.add(p1static);

  const wireMat = new THREE.MeshBasicMaterial({ color: 0x64748b, transparent: true, opacity: 0.8 });
  WIRE_Y.forEach((y, k) => {
    const len = X1 - ENTRY + 1.05;
    const w = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.012, len, 8), wireMat);
    w.rotation.z = Math.PI / 2;
    w.position.set(ENTRY + len / 2 - 0.1, y, 0);
    p1static.add(w);
    const lab = textSprite(`q${k}`, { height: 0.2, color: "#c4b5fd" });
    lab.position.set(ENTRY - 0.45, y, 0);
    p1static.add(lab);
  });
  const title = textSprite("U(x): ZZFeatureMap transpiled to IBM native gates", { height: 0.2, color: "#e5e7eb" });
  title.position.set((X0 + X1) / 2, WIRE_Y[0] + 0.62, 0);
  p1static.add(title);

  const { layer, depth: nLayers } = layerGates(q.gates, 3);
  const dx = (X1 - X0) / Math.max(1, nLayers - 1);
  const cubeGeo = new THREE.BoxGeometry(CUBE, CUBE, CUBE);
  const xGeo = new THREE.BoxGeometry(CUBE * 1.15, CUBE * 1.15, CUBE * 1.15);
  const discGeo = new THREE.CylinderGeometry(0.075, 0.075, 0.022, 24);
  discGeo.rotateZ(Math.PI / 2); // disc faces along the wire
  const gates = q.gates.map((g, i) => {
    const col = GATE_COL[g.name] ?? 0xffffff;
    const mat = new THREE.MeshPhongMaterial({
      color: col, emissive: col, emissiveIntensity: 0.15, transparent: true, opacity: 0, shininess: 50,
    });
    let mesh, y;
    if (g.name === "ecr") {
      const ys = g.qubits.map((k) => WIRE_Y[k]);
      const top = Math.max(...ys), bot = Math.min(...ys);
      mesh = new THREE.Mesh(new THREE.BoxGeometry(CUBE, top - bot + 0.2, CUBE), mat);
      y = (top + bot) / 2;
    } else {
      mesh = new THREE.Mesh(g.name === "rz" ? discGeo : g.name === "x" ? xGeo : cubeGeo, mat);
      y = WIRE_Y[g.qubits[0]];
    }
    const slot = X0 + layer[i] * dx;
    mesh.position.set(slot, y, 0);
    p1.add(mesh);
    return { g, mesh, mat, slot, base: g.name === "rz" ? 0.55 : 0.95 };
  });

  // measurement boxes at the end of each wire (n_meas from the export)
  const meas = [];
  for (let k = 0; k < Math.min(q.n_meas, 3); k++) {
    const m = new THREE.Mesh(
      new THREE.BoxGeometry(0.26, 0.26, 0.1),
      new THREE.MeshPhongMaterial({ color: 0x1e293b, emissive: 0xffffff, emissiveIntensity: 0, transparent: true, opacity: 0 })
    );
    m.position.set(X1 + 0.5, WIRE_Y[k], 0);
    const lab = textSprite("M", { height: 0.15, color: "#e5e7eb" });
    lab.position.set(0, 0, 0.08);
    m.add(lab);
    p1.add(m);
    meas.push({ m, lab });
  }

  // execution cursor
  const cursor = new THREE.Group();
  const cursorBar = new THREE.Mesh(
    new THREE.PlaneGeometry(0.035, WIRE_Y[0] - WIRE_Y[2] + 0.7),
    new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.7, depthWrite: false, side: THREE.DoubleSide })
  );
  cursor.add(cursorBar);
  const cursorGlow = glowSprite(PALETTE.quantum2, 1.0, 0.7);
  cursorGlow.scale.set(0.7, WIRE_Y[0] - WIRE_Y[2] + 1.0, 1);
  cursor.add(cursorGlow);
  cursor.position.y = WIRE_Y[1];
  cursor.visible = false;
  p1.add(cursor);

  // ======================= Part 2: log-scale time race =======================
  const p2 = new THREE.Group();
  p2.position.y = P2_LIFT;
  scene.add(p2);
  const racers = [
    { label: "QSVC on IBM QPU (estimate)", v: q.qsvc_qpu_s, color: PALETTE.quantum },
    { label: "QSVC on this CPU (measured)", v: tm.qsvc.total_s, color: PALETTE.classical },
    { label: "IBM cloud queue (typical)", v: q.queue_s, color: 0x94a3b8 },
  ];
  const { lo, hi } = logRange(racers.map((r) => r.v));
  const L = AX1 - AX0;
  const frac = (v) => Math.min(1, Math.max(0, Math.log(v / lo) / Math.log(hi / lo)));

  const axis = new THREE.Mesh(new THREE.BoxGeometry(L, 0.02, 0.02), new THREE.MeshBasicMaterial({ color: 0x64748b }));
  axis.position.set(AX0 + L / 2, AXIS_Y, 0);
  p2.add(axis);
  for (let e = Math.round(Math.log10(lo)); e <= Math.round(Math.log10(hi)); e++) {
    const x = AX0 + frac(10 ** e) * L;
    const grid = new THREE.Mesh(
      new THREE.PlaneGeometry(0.012, BAR_Y[0] - AXIS_Y + 0.45),
      new THREE.MeshBasicMaterial({ color: 0x334155, transparent: true, opacity: 0.7, depthWrite: false })
    );
    grid.position.set(x, (BAR_Y[0] + 0.45 + AXIS_Y) / 2, -0.2);
    p2.add(grid);
    const t = textSprite(`${int(10 ** e)} s`, { height: 0.15, color: "#94a3b8" });
    t.position.set(x, AXIS_Y - 0.22, 0);
    p2.add(t);
  }
  [[60, "1 min"], [600, "10 min"], [3600, "1 h"]].forEach(([s, name]) => {
    if (s <= lo || s >= hi) return;
    const x = AX0 + frac(s) * L;
    const tick = new THREE.Mesh(new THREE.BoxGeometry(0.015, 0.14, 0.015), new THREE.MeshBasicMaterial({ color: 0x64748b }));
    tick.position.set(x, AXIS_Y, 0);
    p2.add(tick);
    const t = textSprite(name, { height: 0.13, color: "#64748b" });
    t.position.set(x, AXIS_Y - 0.45, 0);
    p2.add(t);
  });
  const axTitle = textSprite("time, log scale (each step = 10× longer)", { height: 0.14, color: "#64748b" });
  axTitle.center.set(0, 0.5);
  axTitle.position.set(AX0, AXIS_Y - 0.72, 0);
  p2.add(axTitle);

  const barGeo = new THREE.BoxGeometry(1, BAR_H, BAR_D);
  barGeo.translate(0.5, 0, 0); // grows from its left end
  const bars = racers.map((r, i) => {
    const mat = new THREE.MeshPhongMaterial({ color: r.color, emissive: r.color, emissiveIntensity: 0.3, transparent: true, opacity: 0.92 });
    const mesh = new THREE.Mesh(barGeo, mat);
    mesh.position.set(AX0, BAR_Y[i], 0);
    mesh.scale.x = 0.001;
    p2.add(mesh);
    const name = textSprite(r.label, { height: 0.17, color: hex(r.color) });
    name.center.set(0, 0.5);
    name.position.set(AX0, BAR_Y[i] + BAR_H / 2 + 0.15, 0);
    p2.add(name);
    const val = textSprite(fmtTime(r.v), { height: 0.2, color: "#ffffff" });
    val.center.set(0, 0.5);
    val.position.set(AX0 + frac(r.v) * L + 0.12, BAR_Y[i], 0);
    p2.add(val);
    const glow = glowSprite(r.color, 0.9, 0.9);
    glow.position.set(AX0 + frac(r.v) * L, BAR_Y[i], 0.1);
    p2.add(glow);
    return { ...r, mesh, mat, val, glow, f: frac(r.v) };
  });
  const clock = new THREE.Mesh(
    new THREE.PlaneGeometry(0.03, BAR_Y[0] - AXIS_Y + 0.5),
    new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.55, depthWrite: false })
  );
  clock.position.y = (BAR_Y[0] + 0.5 + AXIS_Y) / 2;
  p2.add(clock);
  setOpacity(p2, 0);


  // ======================= Parts 3 + 4: timing for all 5 models =======================
  const test = data.test;
  const rows = MODEL_KEYS.map((k) => ({
    key: k, label: MODEL_LABELS[k], color: MODEL_COL[k],
    train: tm[k].train_s, inf: tm[k].inf_s, total: tm[k].total_s, f2: clinicalScores(test, k).f2,
  }));
  const allT = rows.flatMap((r) => [r.train, r.total]);
  const tLo = 10 ** Math.floor(Math.log10(Math.min(...allT) * 0.7));
  const tHi = 10 ** Math.ceil(Math.log10(Math.max(...allT) * 1.0001));
  const tx = (v) => AX0 + logScale(v, tLo, tHi, L); // world x on the shared log axis
  const tFrac = (v) => logScale(v, tLo, tHi, 1);

  function addLogAxis(group, gridTop) {
    const ax = new THREE.Mesh(new THREE.BoxGeometry(L, 0.02, 0.02), new THREE.MeshBasicMaterial({ color: 0x64748b }));
    ax.position.set(AX0 + L / 2, AXIS_Y, 0);
    group.add(ax);
    for (let e = Math.round(Math.log10(tLo)); e <= Math.round(Math.log10(tHi)); e++) {
      const x = tx(10 ** e);
      const grid = new THREE.Mesh(
        new THREE.PlaneGeometry(0.012, gridTop - AXIS_Y),
        new THREE.MeshBasicMaterial({ color: 0x334155, transparent: true, opacity: 0.7, depthWrite: false })
      );
      grid.position.set(x, (gridTop + AXIS_Y) / 2, -0.2);
      group.add(grid);
      const t = textSprite(tickLabel(e), { height: 0.15, color: "#94a3b8" });
      t.position.set(x, AXIS_Y - 0.22, 0);
      group.add(t);
    }
  }

  // ---- Part 3: horizontal bars, train (solid) + inference (lighter) ----
  const p3 = new THREE.Group();
  p3.position.y = P2_LIFT;
  scene.add(p3);
  addLogAxis(p3, T_BAR_Y[0] + 0.5);
  const p3title = textSprite("measured on this laptop · log scale (each step = 10× longer)", { height: 0.14, color: "#64748b" });
  p3title.center.set(0, 0.5);
  p3title.position.set(AX0, AXIS_Y - 0.5, 0);
  p3.add(p3title);
  const unitBar = new THREE.BoxGeometry(1, T_BAR_H, T_BAR_D);
  unitBar.translate(0.5, 0, 0);
  const tbars = rows.map((r, i) => {
    const y = T_BAR_Y[i];
    const trainMat = new THREE.MeshPhongMaterial({ color: r.color, emissive: r.color, emissiveIntensity: 0.3, transparent: true, opacity: 0.95 });
    const infMat = new THREE.MeshPhongMaterial({ color: r.color, emissive: r.color, emissiveIntensity: 0.12, transparent: true, opacity: 0.38 });
    const trainM = new THREE.Mesh(unitBar, trainMat);
    const infM = new THREE.Mesh(unitBar, infMat);
    trainM.position.set(AX0, y, 0);
    infM.position.set(AX0, y, 0);
    trainM.scale.x = infM.scale.x = 0.001;
    p3.add(trainM, infM);
    const name = textSprite(r.label, { height: 0.16, color: hex(r.color) });
    name.center.set(0, 0.5);
    name.position.set(AX0, y + T_BAR_H / 2 + 0.13, 0);
    p3.add(name);
    const val = textSprite(fmtDuration(r.total), { height: 0.19, color: "#ffffff" });
    val.center.set(0, 0.5);
    val.position.set(tx(r.total) + 0.12, y, 0);
    p3.add(val);
    const split = textSprite(`train ${fmtDuration(r.train)} · inference ${fmtDuration(r.inf)}`, { height: 0.11, color: "#94a3b8", weight: 500 });
    split.center.set(0, 0.5);
    split.position.set(tx(r.total) + 0.12, y - 0.19, 0);
    p3.add(split);
    return { ...r, trainM, infM, trainMat, infMat, val, split, xTrain: tx(r.train), xTot: tx(r.total), f: tFrac(r.total) };
  });
  const tclock = new THREE.Mesh(
    new THREE.PlaneGeometry(0.03, T_BAR_Y[0] - AXIS_Y + 0.5),
    new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.55, depthWrite: false })
  );
  tclock.position.y = (T_BAR_Y[0] + 0.4 + AXIS_Y) / 2;
  p3.add(tclock);
  setOpacity(p3, 0);

  const byTotal = [...rows].sort((a, b) => a.total - b.total);
  const fastest = byTotal[0], slowest = byTotal[byTotal.length - 1];

  // ---- Part 4: F2 vs time scatter + Pareto frontier ----
  const p4 = new THREE.Group();
  p4.position.y = P2_LIFT;
  scene.add(p4);
  addLogAxis(p4, SC_TOP + 0.1);
  const yOf = (f2) => AXIS_Y + f2 * (SC_TOP - AXIS_Y);
  [0, 0.5, 1].forEach((v) => {
    const g = new THREE.Mesh(
      new THREE.PlaneGeometry(L, 0.012),
      new THREE.MeshBasicMaterial({ color: 0x334155, transparent: true, opacity: 0.7, depthWrite: false })
    );
    g.position.set(AX0 + L / 2, yOf(v), -0.2);
    p4.add(g);
    const t = textSprite(v.toFixed(1), { height: 0.15, color: "#94a3b8" });
    t.center.set(1, 0.5);
    t.position.set(AX0 - 0.1, yOf(v), 0);
    p4.add(t);
  });
  const yTitle = textSprite("F2 (higher = better)", { height: 0.14, color: "#64748b" });
  yTitle.center.set(0, 0.5);
  yTitle.position.set(AX0 - 0.1, SC_TOP + 0.28, 0);
  p4.add(yTitle);
  const xTitle = textSprite("total time, log scale (left = faster)", { height: 0.14, color: "#64748b" });
  xTitle.center.set(0, 0.5);
  xTitle.position.set(AX0, AXIS_Y - 0.5, 0);
  p4.add(xTitle);

  const front = paretoFront(rows.map((r) => ({ key: r.key, time: r.total, score: r.f2 })));
  const onFront = new Set(front);
  const dots = byTotal.map((r, order) => {
    const px = tx(r.total), py = yOf(r.f2), fr = onFront.has(r.key);
    const g = new THREE.Group();
    g.position.set(px, py, 0.05);
    const mat = new THREE.MeshPhongMaterial({ color: r.color, emissive: r.color, emissiveIntensity: fr ? 0.7 : 0.25, transparent: true, opacity: fr ? 1 : 0.55, shininess: 60 });
    g.add(new THREE.Mesh(new THREE.SphereGeometry(fr ? 0.1 : 0.075, 20, 14), mat));
    const glow = glowSprite(fr ? FRONT_COL : r.color, fr ? 0.85 : 0.5, fr ? 1 : 0.6);
    g.add(glow);
    if (fr) {
      const ring = new THREE.Mesh(new THREE.TorusGeometry(0.15, 0.012, 8, 40), new THREE.MeshBasicMaterial({ color: FRONT_COL, transparent: true, opacity: 0.95 }));
      g.add(ring);
    }
    p4.add(g);
    return { ...r, px, py, fr, g, order };
  });
  // labels: greedy placement over candidate offsets so close points (e.g. Linear/RBF, QSVC/Hybrid) don't collide
  const LH = 0.15, LW = (txt) => Math.max(...txt.split("\n").map((l) => l.length)) * LH * 0.56;
  const placed = [];
  const hit = (a, b) => a.x0 < b.x1 && b.x0 < a.x1 && a.y0 < b.y1 && b.y0 < a.y1;
  dots.forEach((d) => {
    d.labelText = `${d.label}${d.fr ? "  ★" : ""}\nF2 ${d.f2.toFixed(3)} · ${fmtDuration(d.total)}`;
    const w = LW(d.labelText), h = LH * 2.5;
    const cands = [];
    for (const dy of [0.3, -0.3, 0.68, -0.68, 1.06, -1.06]) for (const side of ["R", "L"]) cands.push({ dy, side });
    let best = null;
    for (const c of cands) {
      const x0 = c.side === "R" ? d.px + 0.1 : d.px - 0.1 - w;
      const box = { x0, x1: x0 + w, y0: d.py + c.dy - h / 2, y1: d.py + c.dy + h / 2 };
      const ok = box.x0 >= AX0 - 0.05 && box.x1 <= AX1 + 0.35 && box.y1 <= SC_TOP + 0.45 && box.y0 >= AXIS_Y + 0.02
        && !placed.some((p) => hit(box, p))
        && !dots.some((o) => hit(box, { x0: o.px - 0.14, x1: o.px + 0.14, y0: o.py - 0.14, y1: o.py + 0.14 }));
      if (ok) { best = { ...c, box }; break; }
    }
    if (!best) best = { dy: 0.3, side: "R", box: { x0: d.px, x1: d.px + w, y0: d.py, y1: d.py + h } };
    placed.push(best.box);
    const lab = textSprite(d.labelText, { height: LH, color: hex(d.color), align: best.side === "R" ? "left" : "right" });
    lab.center.set(best.side === "R" ? 0 : 1, 0.5);
    lab.position.set(best.side === "R" ? d.px + 0.1 : d.px - 0.1, d.py + best.dy, 0.05);
    p4.add(lab);
    d.lab = lab;
    if (Math.abs(best.dy) > 0.4) { // thin leader line from the dot to its label
      const ay = d.py + best.dy + (best.dy > 0 ? -LH * 1.3 : LH * 1.3);
      const ax = best.side === "R" ? d.px + 0.1 : d.px - 0.1;
      const lg = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(d.px, d.py, 0.03), new THREE.Vector3(ax, ay, 0.03)]);
      const ll = new THREE.Line(lg, new THREE.LineBasicMaterial({ color: d.color, transparent: true, opacity: 0.6 }));
      p4.add(ll);
      d.leader = ll;
    }
  });
  // frontier line (gold) connecting the Pareto-optimal models in time order
  const frontDots = front.map((k) => dots.find((d) => d.key === k));
  const lineMat = new THREE.MeshBasicMaterial({ color: FRONT_COL, transparent: true, opacity: 0.95 });
  const segs = [];
  for (let i = 0; i + 1 < frontDots.length; i++) {
    const a = frontDots[i], b = frontDots[i + 1];
    const len = Math.hypot(b.px - a.px, b.py - a.py);
    const geo = new THREE.CylinderGeometry(0.018, 0.018, 1, 8);
    geo.translate(0, 0.5, 0); // from base up
    geo.rotateZ(-Math.PI / 2); // along +x
    const m = new THREE.Mesh(geo, lineMat.clone());
    m.position.set(a.px, a.py, 0.02);
    m.rotation.z = Math.atan2(b.py - a.py, b.px - a.px);
    m.scale.x = 0.001;
    p4.add(m);
    segs.push({ m, len });
  }
  const totalLen = segs.reduce((s, x) => s + x.len, 0);
  setOpacity(p4, 0);

  // ======================= captions =======================
  const key = `<b style="color:${hex(GATE_COL.sx)}">sx</b> · <b style="color:${hex(GATE_COL.x)}">x</b> (pulses) · <b style="color:${hex(GATE_COL.rz)}">rz</b> (virtual) · <b style="color:${hex(GATE_COL.ecr)}">ECR</b> (2-qubit)`;
  const ops = q.fm_ops || {};
  const n1U = (ops.sx || 0) + (ops.x || 0);
  const resetUs = (q.t_eval_ms * 1000) / q.shots - q.t_circuit_us;
  const est = `<span style="color:#94a3b8">estimate (typical IBM calibration)</span>`;
  const cpu = tm.qsvc.total_s, qpu = q.qsvc_qpu_s, queue = q.queue_s;
  const capPart1a = `<b>From simulator to real QPU</b> · an IBM chip runs only a few native gates, so the feature-map circuit U(x) is <b>transpiled</b> (optimization level 2) to ${key}`;
  const capPart1b = `<b>One run of U(x)</b>: ${n1U} pulses (sx, x) · ${ops.rz || 0} rz · ${ops.ecr || 0} ECR · depth <b>${q.fm_depth}</b> layers · ${key}`;
  const capPart1c = `<b>Per kernel evaluation</b> the chip runs U(x) then U†(x′) and measures: <b>${q.n_1q}</b> single-qubit pulses · <b>${q.n_rz}</b> rz (0 ns) · <b>${q.n_2q}</b> ECR · <b>${q.n_meas}</b> measurements · depth <b>${q.fm_depth}</b> per U → <b>${q.t_circuit_us.toFixed(1)} µs</b> per shot · ${est}`;
  const racePrefix = `<b>Time race</b> · the whole QSVC job for this disease: <b>${int(q.n_qsvc_evals)}</b> kernel evaluations × <b>${int(q.shots)}</b> shots × (${q.t_circuit_us.toFixed(1)} µs circuit + ${resetUs.toFixed(0)} µs reset) on the chip`;
  const vqcLine = `VQC: ${fmtTime(q.vqc_qpu_s)} on the QPU (estimate) vs ${fmtTime(tm.vqc.total_s)} on this CPU (${fmtRatio(q.vqc_qpu_s / tm.vqc.total_s)})`;
  const capResult = `<b>QSVC on IBM QPU ≈ ${fmtTime(qpu)}</b> <span style="color:#94a3b8">(estimate)</span> vs <b>${fmtTime(cpu)}</b> on this CPU (measured): <b>${fmtRatio(qpu / cpu)}</b> longer · a typical free-tier queue alone is <b>${fmtTime(queue)}</b> (<b>${fmtRatio(queue / cpu)}</b> the CPU run)`
    + `<div style="font-size:13px;color:#cbd5e1;margin-top:3px">${vqcLine} · at 3 qubits the laptop simulator wins; a QPU pays off only for circuits too large to simulate</div>`
    + `<div style="font-size:12px;color:#94a3b8;margin-top:2px">Live: the panel on the right can send this patient's kernel circuit to a real IBM QPU (it asks first)</div>`;

  const ratioTT = slowest.total / fastest.total;
  const nTest = test.length;
  const capBarsRun = `<b>Every model's compute time</b> · measured on this laptop (train + inference) · <span style="color:#94a3b8">solid = training, light = inference · log scale</span>`;
  const capBarsDone = `<b>Fastest: ${fastest.label}</b> ${fmtDuration(fastest.total)} · <b>slowest: ${slowest.label}</b> ${fmtDuration(slowest.total)} · the slowest takes <b>${fmtRatio(ratioTT)}</b> longer`
    + `<div style="font-size:13px;color:#cbd5e1;margin-top:3px">The Hybrid's time includes QSVC's (${fmtDuration(tm.qsvc.total_s)}): it computes the QSVC kernel and then also fits the RBF SVM · all numbers measured on this CPU, not estimated</div>`;
  const label = (k) => MODEL_LABELS[k];
  const frontNames = front.map(label).join(" → ");
  const dominated = rows.filter((r) => !onFront.has(r.key)).map((r) => r.label);
  const capFrontRun = `<b>Efficiency frontier</b> · each dot: F2 on the ${nTest} test patients (up = better) vs total time (left = faster) · <span style="color:#fbbf24">gold = no other model is both faster and scores higher</span>`;
  const capFrontDone = `<b>Frontier (Pareto-optimal): <span style="color:#fbbf24">${frontNames}</span></b>`
    + (dominated.length ? ` · beaten on both axes: ${dominated.join(", ")}` : " · no model is beaten on both axes")
    + `<div style="font-size:13px;color:#cbd5e1;margin-top:3px">Pick from the gold line: moving right costs time, and is worth it only if F2 rises</div>`;

  return {
    title: "Simulator → real QPU",
    legend: LEGEND_6,
    scene,
    cameraPos: [0, 0.45, 8.5],
    duration,
    stops: [P1_STOP, RESULT_AT + 0.4, P3_STOP, P4_STOP],
    update(t) {
      // ---------- part 1 ----------
      const out = 1 - ease(seg(t, P1_OUT0, P1_OUT1));
      p1.visible = out > 0.005;
      if (p1.visible) {
        setOpacity(p1static, ease(seg(t, 0, WIRES_IN)) * out);
        const off = -(X1 - ENTRY + 1.2) * (1 - ease(seg(t, SLIDE0, SLIDE1)));
        const sweeping = t >= SWEEP0;
        const cx = X0 - 0.2 + (X1 - X0 + 0.4) * seg(t, SWEEP0, SWEEP1);
        cursor.visible = sweeping && t < SWEEP1 + 0.35;
        cursor.position.x = cx;
        cursorBar.material.opacity = 0.7 * (1 - seg(t, SWEEP1, SWEEP1 + 0.35)) * out;
        cursorGlow.material.opacity = cursorBar.material.opacity;
        for (const G of gates) {
          const x = G.slot + off;
          G.mesh.position.x = x;
          const op = seg(x, ENTRY - 0.4, ENTRY) * out;
          G.mat.opacity = G.base * op;
          G.mesh.visible = op > 0.005;
          let glow = 0.15, s = 1;
          if (sweeping && cx >= G.slot) {
            const f = Math.max(0, 1 - (cx - G.slot) / 0.6);
            glow = 0.45 + 0.9 * f;
            s = 1 + 0.4 * f;
          }
          G.mat.emissiveIntensity = glow;
          G.mesh.scale.setScalar(s);
        }
        const mOp = ease(seg(t, SLIDE1 - 0.4, SLIDE1)) * out;
        const mGlow = ease(seg(t, MEAS_AT, MEAS_AT + 0.3)) * (1 - 0.6 * seg(t, MEAS_AT + 0.3, MEAS_AT + 0.9));
        meas.forEach(({ m, lab }) => {
          m.material.opacity = 0.95 * mOp;
          m.material.emissiveIntensity = 0.6 * mGlow;
          lab.material.opacity = mOp;
          m.visible = mOp > 0.005;
        });
      }

      // ---------- part 2 ----------
      const inP2 = ease(seg(t, P2_IN0, P2_IN1)) * (1 - ease(seg(t, P2_OUT0, P2_OUT1)));
      const p = seg(t, RACE0, RACE1);
      const T = lo * (hi / lo) ** p;
      if (inP2 > 0) {
        setOpacity(p2, inP2);
        for (const b of bars) {
          const len = Math.min(b.f, p) * L;
          b.mesh.scale.x = Math.max(0.001, len);
          const doneAt = RACE0 + b.f * (RACE1 - RACE0);
          const fin = ease(seg(t, doneAt, doneAt + 0.25));
          b.val.material.opacity = fin * inP2;
          b.val.visible = fin > 0.005;
          const flash = t >= doneAt ? Math.max(0, 1 - (t - doneAt) / 0.8) : 0;
          b.glow.material.opacity = (0.25 + 0.75 * flash) * fin * inP2;
          b.glow.visible = fin > 0.005;
          b.mat.emissiveIntensity = 0.3 + 0.6 * flash;
        }
        clock.position.x = AX0 + p * L;
        clock.visible = t >= RACE0 && t < RACE1 + 0.3;
        clock.material.opacity = 0.55 * (1 - seg(t, RACE1, RACE1 + 0.3));
      } else {
        p2.visible = false;
      }

      // ---------- part 3: timing bars ----------
      const inP3 = ease(seg(t, P3_IN0, P3_IN1)) * (1 - ease(seg(t, P3_OUT0, P3_OUT1)));
      if (inP3 > 0) {
        setOpacity(p3, inP3);
        const pt = seg(t, TB0, TB1);
        for (const b of tbars) {
          const reach = pt * L; // clock position along the axis
          const endTot = b.xTot - AX0, endTr = b.xTrain - AX0;
          const trLen = Math.min(endTr, reach), totLen = Math.min(endTot, reach);
          b.trainM.scale.x = Math.max(0.001, trLen);
          b.infM.position.x = AX0 + trLen;
          b.infM.scale.x = Math.max(0.001, totLen - trLen);
          const doneAt = TB0 + b.f * (TB1 - TB0);
          const fin = ease(seg(t, doneAt, doneAt + 0.25));
          b.val.material.opacity = fin * inP3;
          b.split.material.opacity = fin * inP3;
          b.val.visible = b.split.visible = fin > 0.005;
          const flash = t >= doneAt ? Math.max(0, 1 - (t - doneAt) / 0.8) : 0;
          b.trainMat.emissiveIntensity = 0.3 + 0.6 * flash;
        }
        tclock.position.x = AX0 + pt * L;
        tclock.visible = t >= TB0 && t < TB1 + 0.3;
        tclock.material.opacity = 0.55 * (1 - seg(t, TB1, TB1 + 0.3)) * inP3;
      } else {
        p3.visible = false;
      }

      // ---------- part 4: efficiency frontier ----------
      const inP4 = ease(seg(t, P4_IN0, P4_IN1));
      if (inP4 > 0) {
        setOpacity(p4, inP4);
        const pulse = t >= LINE1 ? 0.5 + 0.5 * Math.sin((t - LINE1) * 3) : 0;
        for (const d of dots) {
          const a = ease(seg(t, DOT0 + d.order * DOT_STEP, DOT0 + d.order * DOT_STEP + DOT_FADE));
          const s = (0.3 + 0.7 * a) * (d.fr ? 1 + 0.08 * pulse : 1);
          d.g.scale.setScalar(s);
          d.g.visible = a > 0.005;
          d.lab.visible = a > 0.3;
          if (d.leader) { d.leader.visible = a > 0.3; d.leader.material.opacity = 0.6 * Math.max(0, (a - 0.3) / 0.7) * inP4; }
          d.lab.material.opacity = Math.max(0, (a - 0.3) / 0.7) * inP4;
          d.g.traverse((o) => { if (o.material) o.material.opacity = (o.material.userData.baseOp ?? 1) * a * inP4; });
        }
        let drawn = ease(seg(t, LINE0, LINE1)) * totalLen;
        for (const sg of segs) {
          const l = Math.min(sg.len, Math.max(0, drawn));
          drawn -= sg.len;
          sg.m.scale.x = Math.max(0.001, l);
          sg.m.visible = l > 0.002;
        }
      } else {
        p4.visible = false;
      }

      // ---------- caption ----------
      if (t >= P3_OUT0 + 0.2) return t < P4_RESULT ? capFrontRun : capFrontDone;
      if (t >= P3_IN0) return t < P3_RESULT ? capBarsRun : capBarsDone;
      if (t < SWEEP0) return capPart1a;
      if (t < SWEEP1) return capPart1b;
      if (t < P2_IN0) return capPart1c;
      if (t < RESULT_AT) return `${racePrefix} · clock <b>${fmtTime(Math.min(T, hi))}</b> · ${est}`;
      return capResult;
    },
    dispose() {
      disposeTree(scene);
    },
  };
}
