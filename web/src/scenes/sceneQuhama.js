import * as THREE from "three";
import { LEGEND_QUHAMA } from "./legends.js";
import {
  PALETTE, ease, seg, blochSphere, firstLayerState, textSprite, glowSprite,
  setOpacity, disposeTree, addLights,
} from "./common.js";
import { loadQuhama } from "../data.js";
import { quhamaPatient, pickInputs, pct1 } from "../quhama.js";

// QuHaMA (mirrors qmed/ui/tab_quhama.py): the chosen test patient flows through
// inputs -> encoder -> latent z -> {classical head, quantum branch} -> gating network -> blended verdict,
// then the model's test metrics. Every number comes from <slug>.quhama.json (and the patient's
// standardized inputs from <slug>.json). The JSON is fetched lazily; the timeline below starts once it
// has loaded (stops / duration are shifted by that moment).

// Timeline (seconds after the data has loaded)
const T = {
  skeleton: [0, 0.7], // faint network fades in
  inputs: [0.1, 0.8], // input nodes light up
  toEnc: [0.7, 2.3], // particles: inputs -> encoder
  enc: [1.4, 2.4], // encoder units light up
  toLat: [2.2, 3.4], // encoder -> latent
  lat: [3.0, 3.6], // latent nodes + values
  toBr: [4.1, 5.5], // latent -> classical head / qubits
  head: [4.9, 5.7],
  bloch: [4.7, 5.8], // Bloch arrows |0> -> first-layer state
  toP: [5.5, 6.6], // head -> P_c, qubits -> P_q
  probs: [6.1, 6.9], // branch gauges fill
  toGate: [7.4, 8.7], // z, P_c, P_q -> gate
  ring: [8.3, 9.2], // weight ring grows
  toFinal: [9.8, 11.2], // weighted streams -> blend
  final: [10.7, 11.6], // blended gauge fills, verdict
  metrics: [12.0, 12.5], // test-metrics panel
};
const STOPS = [3.9, 7.1, 9.5]; // after the latent, after the branches, after the gate
const DURATION = 16.5;
const TRAVEL = 0.7; // seconds per particle flight

// Layout (root-local)
const IN_X = -4.3;
const ENC = new THREE.Vector3(-2.6, 0, 0);
const ENC_W = 0.8, ENC_H = 2.3;
const LAT_X = -0.95;
const LAT_Y = [0.8, 0, -0.8];
const HEAD = new THREE.Vector3(0.95, 1.45, 0);
const HEAD_W = 0.85, HEAD_H = 0.7;
const PC = new THREE.Vector3(2.4, 1.45, 0);
const QB_Y = -1.45, QB_X = [0.27, 0.95, 1.63], QB_R = 0.27;
const PQ = new THREE.Vector3(2.4, -1.45, 0);
const GATE = new THREE.Vector3(3.8, 0, 0);
const FINAL = new THREE.Vector3(5.35, 0, 0);
const G_H = 0.95, F_H = 1.7; // gauge heights (branch, final)

const LATENT = 0xe9d5ff;
const GATE_C = 0xfbbf24;
const SUB = ["₁", "₂", "₃", "₄", "₅", "₆", "₇", "₈"];
const hex = (c) => "#" + c.toString(16).padStart(6, "0");
const n2 = (v) => v.toFixed(2).replace("-", "−");
const sgn = (v) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(2);
const pretty = (name) => name.replace(/_/g, " ");
const short = (name, n = 15) => (name.length > n ? name.slice(0, n - 1) + "…" : name);
const span = (c, x) => `<b style="color:${hex(c)}">${x}</b>`;

function dotTexture() {
  const c = document.createElement("canvas");
  c.width = c.height = 64;
  const g = c.getContext("2d");
  const grad = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  grad.addColorStop(0, "rgba(255,255,255,1)");
  grad.addColorStop(0.3, "rgba(255,255,255,0.7)");
  grad.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = grad;
  g.fillRect(0, 0, 64, 64);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

/** Translucent box with glowing edges and a grid of unit dots (a neural-network layer). */
function layerBox(w, h, color, rows, cols) {
  const g = new THREE.Group();
  const box = new THREE.Mesh(
    new THREE.BoxGeometry(w, h, 0.45),
    new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.07, depthWrite: false })
  );
  g.add(box);
  g.add(new THREE.LineSegments(
    new THREE.EdgesGeometry(box.geometry),
    new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.6 })
  ));
  const units = [];
  const geo = new THREE.SphereGeometry(0.05, 12, 8);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const m = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 1 }));
      m.position.set((c - (cols - 1) / 2) * (w / (cols + 0.6)), ((rows - 1) / 2 - r) * (h / (rows + 0.6)), 0);
      g.add(m);
      units.push(m);
    }
  }
  return { group: g, units };
}

/** Vertical probability gauge: open frame, fill from the bottom, a ring at 50%. */
function gauge(h, r, color) {
  const g = new THREE.Group();
  const frame = new THREE.Mesh(
    new THREE.CylinderGeometry(r, r, h, 28, 1, true),
    new THREE.MeshBasicMaterial({ color: 0x94a3b8, transparent: true, opacity: 0.16, side: THREE.DoubleSide, depthWrite: false })
  );
  g.add(frame);
  const fillGeo = new THREE.CylinderGeometry(r * 0.82, r * 0.82, 1, 28);
  fillGeo.translate(0, 0.5, 0);
  const mat = new THREE.MeshPhongMaterial({ color, emissive: color, emissiveIntensity: 0.45, transparent: true, opacity: 0.95 });
  const fill = new THREE.Mesh(fillGeo, mat);
  fill.position.y = -h / 2;
  fill.scale.y = 0.001;
  g.add(fill);
  const tick = new THREE.Mesh(
    new THREE.TorusGeometry(r * 1.12, 0.012, 6, 40),
    new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.55 })
  );
  tick.rotation.x = Math.PI / 2;
  g.add(tick);
  const glow = glowSprite(color, r * 5, 0.8);
  g.add(glow);
  return {
    group: g, mat,
    set(p) {
      fill.scale.y = Math.max(0.001, p * h);
      glow.position.y = -h / 2 + p * h;
      glow.visible = p > 0.002;
    },
  };
}

export default function createScene(ctx) {
  const { data } = ctx;
  const scene = new THREE.Scene();
  addLights(scene);
  const root = new THREE.Group();
  root.position.set(2.2, 1.0, 0); // right of the legend panel, above the caption
  scene.add(root);

  let status = "loading"; // loading | ready | error
  let errMsg = "";
  let q = null, info = null, built = null;
  let t0 = null; // scene time at which the data had loaded and the timeline started
  let lastT = 0; // last scene time rendered while loading (frames can be sparse, e.g. headless)
  let disposed = false;

  loadQuhama(data.slug).then(
    (json) => {
      if (disposed) return;
      const r = quhamaPatient(json, data, ctx.patient);
      if (!r.ok) { status = "error"; errMsg = r.msg; return; }
      q = json; info = r; status = "ready";
      t0 = lastT;
    },
    (e) => { if (!disposed) { status = "error"; errMsg = e.message; } }
  );

  function build() {
    const row = info.row;
    const arch = q.architecture;
    const p = data.test[ctx.patient];
    const k = row.z_latent.length;
    const d = arch.in_features;
    const latY = k === 3 ? LAT_Y : row.z_latent.map((_, j) => 0.8 - (1.6 * j) / Math.max(1, k - 1));
    const qbX = k === 3 ? QB_X : row.z_latent.map((_, j) => 0.95 + (j - (k - 1) / 2) * 0.68);

    const flows = []; // {a, b, win, color, n}
    const stageObjs = []; // {obj, win, dim} fade from dim to 1 over win
    const labels = []; // {obj, win} fade from 0 to 1 over win

    // ---------- inputs ----------
    const shown = pickInputs(p.standardized);
    const more = d - shown.length;
    const nRows = shown.length + (more > 0 ? 1 : 0);
    const dy = Math.min(0.42, 4.3 / Math.max(1, nRows - 1));
    const inputs = new THREE.Group();
    root.add(inputs);
    const nodeGeo = new THREE.SphereGeometry(0.085, 16, 12);
    shown.forEach((fi, r) => {
      const z = p.standardized[fi];
      const b = 0.3 + 0.7 * Math.min(1, Math.abs(z) / 2.5);
      const y = ((nRows - 1) / 2 - r) * dy;
      const col = new THREE.Color(PALETTE.classical).multiplyScalar(b);
      const node = new THREE.Mesh(nodeGeo, new THREE.MeshBasicMaterial({ color: col, transparent: true }));
      node.position.set(IN_X, y, 0);
      inputs.add(node);
      const gl = glowSprite(PALETTE.classical, 0.2 + 0.35 * b, b);
      gl.position.copy(node.position);
      inputs.add(gl);
      const lab = textSprite(`${short(pretty(data.feature_names[fi]))}  ${sgn(z)}σ`, { height: 0.16, color: "#cbd5e1", align: "right", weight: 500 });
      lab.position.set(IN_X - 0.16 - lab.scale.x / 2, y, 0);
      inputs.add(lab);
      flows.push({ a: node.position.clone(), b: new THREE.Vector3(ENC.x - ENC_W / 2, ENC.y + y * 0.45, 0), win: T.toEnc, color: PALETTE.classical, n: 6 });
    });
    if (more > 0) {
      const y = ((nRows - 1) / 2 - shown.length) * dy;
      const lab = textSprite(`+${more} more (all ${d} go into the encoder)`, { height: 0.14, color: "#94a3b8", align: "right", weight: 500 });
      lab.position.set(IN_X + 0.1 - lab.scale.x / 2, y, 0);
      inputs.add(lab);
    }
    stageObjs.push({ obj: inputs, win: T.inputs, dim: 0.25 });
    const inTitle = textSprite(`Inputs · ${d} measurements (σ)`, { height: 0.20, color: hex(PALETTE.classical) });
    inTitle.position.set(IN_X - 0.55, (nRows - 1) / 2 * dy + 0.45, 0);
    root.add(inTitle);

    // ---------- encoder ----------
    const enc = layerBox(ENC_W, ENC_H, PALETTE.classical, Math.min(8, arch.encoder_hidden / 2 || 8), 2);
    enc.group.position.copy(ENC);
    root.add(enc.group);
    stageObjs.push({ obj: enc.group, win: T.enc, dim: 0.3 });
    const encT = textSprite(`Encoder\n${d} → ${arch.encoder_hidden} → ${k}`, { height: 0.19, color: hex(PALETTE.classical) });
    encT.position.set(ENC.x, ENC.y + ENC_H / 2 + 0.36, 0);
    root.add(encT);

    // ---------- latent ----------
    const latent = row.z_latent.map((z, j) => {
      const a = Math.abs(z);
      const g = new THREE.Group();
      const col = new THREE.Color(LATENT).multiplyScalar(0.25 + 0.75 * a);
      const m = new THREE.Mesh(new THREE.SphereGeometry(0.15, 24, 16), new THREE.MeshBasicMaterial({ color: col, transparent: true }));
      g.add(m);
      const gl = glowSprite(LATENT, 0.3 + 0.75 * a, 0.3 + 0.7 * a);
      g.add(gl);
      g.position.set(LAT_X, latY[j], 0);
      root.add(g);
      stageObjs.push({ obj: g, win: T.lat, dim: 0.2 });
      const lab = textSprite(`z${SUB[j] || j + 1} = ${n2(z)}`, { height: 0.18, color: "#f5f3ff" });
      lab.position.set(LAT_X, latY[j] + 0.3, 0);
      root.add(lab);
      labels.push({ obj: lab, win: T.lat });
      flows.push({ a: new THREE.Vector3(ENC.x + ENC_W / 2, ENC.y + latY[j] * 1.1, 0), b: g.position.clone(), win: T.toLat, color: LATENT, n: 10 });
      return g;
    });
    const latT = textSprite(`Latent z · k = ${k} = qubits`, { height: 0.19, color: hex(LATENT) });
    latT.position.set(LAT_X, latY[0] + 0.72, 0);
    root.add(latT);

    // ---------- classical head ----------
    const head = layerBox(HEAD_W, HEAD_H, PALETTE.classical, 2, Math.min(4, arch.head_hidden / 2 || 4));
    head.group.position.copy(HEAD);
    root.add(head.group);
    stageObjs.push({ obj: head.group, win: T.head, dim: 0.3 });
    const headT = textSprite("Classical head", { height: 0.19, color: hex(PALETTE.classical) });
    headT.position.set(HEAD.x, HEAD.y + HEAD_H / 2 + 0.22, 0);
    root.add(headT);
    latent.forEach((g) => flows.push({ a: g.position.clone(), b: new THREE.Vector3(HEAD.x - HEAD_W / 2, HEAD.y, 0), win: T.toBr, color: PALETTE.classical, n: 8 }));
    flows.push({ a: new THREE.Vector3(HEAD.x + HEAD_W / 2, HEAD.y, 0), b: PC.clone(), win: T.toP, color: PALETTE.classical, n: 12 });

    // ---------- quantum branch ----------
    const qubits = row.z_angles.map((ang, j) => {
      const s = blochSphere(QB_R);
      s.position.set(qbX[j], QB_Y, 0);
      root.add(s);
      stageObjs.push({ obj: s, win: T.bloch, dim: 0.3 });
      flows.push({ a: latent[j].position.clone(), b: s.position.clone().add(new THREE.Vector3(0, QB_R, 0)), win: T.toBr, color: PALETTE.quantum, n: 8 });
      flows.push({ a: s.position.clone(), b: PQ.clone(), win: T.toP, color: PALETTE.quantum, n: 5 });
      const lab = textSprite(`θ${SUB[j] || j + 1} ${ang.toFixed(2)}`, { height: 0.14, color: "#e9d5ff", weight: 500 });
      lab.position.set(qbX[j], QB_Y - QB_R - 0.16, 0);
      root.add(lab);
      labels.push({ obj: lab, win: T.bloch });
      return { s, target: firstLayerState(ang) };
    });
    const qT = textSprite(`Quantum branch · QSVC · ${k} qubits`, { height: 0.18, color: hex(PALETTE.quantum) });
    qT.position.set(qbX[Math.floor(k / 2)], QB_Y - QB_R - 0.42, 0);
    root.add(qT);

    // ---------- branch gauges ----------
    const gc = gauge(G_H, 0.13, PALETTE.classical);
    gc.group.position.copy(PC);
    root.add(gc.group);
    const gq = gauge(G_H, 0.13, PALETTE.quantum);
    gq.group.position.copy(PQ);
    root.add(gq.group);
    stageObjs.push({ obj: gc.group, win: T.probs, dim: 0.35 }, { obj: gq.group, win: T.probs, dim: 0.35 });
    const pcL = textSprite(`P_c ${pct1(row.p_classical)}%`, { height: 0.19, color: hex(PALETTE.classical) });
    pcL.position.set(PC.x, PC.y + G_H / 2 + 0.22, 0);
    const pqL = textSprite(`P_q ${pct1(row.p_quantum)}%`, { height: 0.19, color: hex(PALETTE.quantum) });
    pqL.position.set(PQ.x, PQ.y + G_H / 2 + 0.22, 0);
    root.add(pcL, pqL);
    labels.push({ obj: pcL, win: T.probs }, { obj: pqL, win: T.probs });

    // ---------- gating network ----------
    const gate = new THREE.Group();
    gate.position.copy(GATE);
    root.add(gate);
    const core = new THREE.Mesh(
      new THREE.IcosahedronGeometry(0.2, 1),
      new THREE.MeshPhongMaterial({ color: GATE_C, emissive: GATE_C, emissiveIntensity: 0.35, flatShading: true, transparent: true })
    );
    gate.add(core);
    const back = new THREE.Mesh(
      new THREE.RingGeometry(0.4, 0.55, 96, 1),
      new THREE.MeshBasicMaterial({ color: 0x334155, transparent: true, opacity: 0.6, side: THREE.DoubleSide, depthWrite: false })
    );
    gate.add(back);
    const wc = row.w_classical, wq = row.w_quantum;
    const SEGS = 128;
    const arcC = new THREE.Mesh(
      new THREE.RingGeometry(0.39, 0.56, SEGS, 1, Math.PI / 2, 2 * Math.PI * wc),
      new THREE.MeshBasicMaterial({ color: PALETTE.classical, transparent: true, side: THREE.DoubleSide, depthWrite: false })
    );
    const arcQ = new THREE.Mesh(
      new THREE.RingGeometry(0.39, 0.56, SEGS, 1, Math.PI / 2 + 2 * Math.PI * wc, 2 * Math.PI * wq),
      new THREE.MeshBasicMaterial({ color: PALETTE.quantum, transparent: true, side: THREE.DoubleSide, depthWrite: false })
    );
    arcC.position.z = arcQ.position.z = 0.01;
    gate.add(arcC, arcQ);
    stageObjs.push({ obj: gate, win: T.toGate, dim: 0.35 });
    const gateT = textSprite("Gating network", { height: 0.19, color: hex(GATE_C) });
    gateT.position.set(GATE.x, GATE.y + 0.85, 0);
    root.add(gateT);
    const wcL = textSprite(`Classical ${pct1(wc)}%`, { height: 0.18, color: hex(PALETTE.classical) });
    wcL.position.set(GATE.x, GATE.y - 0.78, 0);
    const wqL = textSprite(`Quantum ${pct1(wq)}%`, { height: 0.18, color: hex(PALETTE.quantum) });
    wqL.position.set(GATE.x, GATE.y - 1.0, 0);
    root.add(wcL, wqL);
    labels.push({ obj: wcL, win: T.ring }, { obj: wqL, win: T.ring });
    latent.forEach((g) => flows.push({ a: g.position.clone(), b: GATE.clone(), win: T.toGate, color: LATENT, n: 4 }));
    flows.push({ a: PC.clone().add(new THREE.Vector3(0, G_H / 2, 0)), b: GATE.clone(), win: T.toGate, color: PALETTE.classical, n: 10 });
    flows.push({ a: PQ.clone().add(new THREE.Vector3(0, G_H / 2, 0)), b: GATE.clone(), win: T.toGate, color: PALETTE.quantum, n: 10 });

    // ---------- blended verdict ----------
    const verdictC = info.disease ? PALETTE.sick : PALETTE.healthy;
    const gf = gauge(F_H, 0.2, verdictC);
    gf.group.position.copy(FINAL);
    root.add(gf.group);
    stageObjs.push({ obj: gf.group, win: T.final, dim: 0.35 });
    const fT = textSprite(`Blended P ${pct1(row.p_final)}%`, { height: 0.20, color: "#ffffff" });
    fT.position.set(FINAL.x, FINAL.y + F_H / 2 + 0.3, 0);
    root.add(fT);
    labels.push({ obj: fT, win: T.final });
    const vL = textSprite(
      `${info.disease ? "DISEASE" : "HEALTHY"} ${info.correct ? "✓" : "✗"}\nactual: ${info.actualSick ? "Disease" : "Healthy"}`,
      { height: 0.18, color: hex(verdictC) }
    );
    vL.position.set(FINAL.x, FINAL.y - F_H / 2 - 0.3, 0);
    root.add(vL);
    labels.push({ obj: vL, win: [T.final[1] - 0.2, T.final[1] + 0.2] });
    // particle counts proportional to the gate's weights (at least one so both paths stay visible)
    const NF = 40;
    flows.push({ a: PC.clone(), b: FINAL.clone(), win: T.toFinal, color: PALETTE.classical, n: Math.max(1, Math.round(NF * wc)), bendY: 0.35 });
    flows.push({ a: PQ.clone(), b: FINAL.clone(), win: T.toFinal, color: PALETTE.quantum, n: Math.max(1, Math.round(NF * wq)), bendY: -0.35 });

    // ---------- edges + particles ----------
    const edges = flows.map((f, i) => {
      const mid = f.a.clone().lerp(f.b, 0.5);
      mid.z += 0.3 + 0.1 * Math.sin(i * 1.7);
      if (f.bendY) mid.y += f.bendY;
      f.curve = new THREE.QuadraticBezierCurve3(f.a, mid, f.b);
      const line = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(f.curve.getPoints(20)),
        new THREE.LineBasicMaterial({ color: f.color, transparent: true, opacity: 0.08, depthWrite: false })
      );
      root.add(line);
      return line;
    });
    const parts = [];
    flows.forEach((f, fi) => {
      for (let i = 0; i < f.n; i++) {
        const emit = Math.max(0.01, f.win[1] - f.win[0] - TRAVEL);
        parts.push({ f: fi, te: f.win[0] + (f.n === 1 ? emit / 2 : (i / (f.n - 1)) * emit) });
      }
    });
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(parts.length * 3), 3));
    geo.setAttribute("color", new THREE.BufferAttribute(new Float32Array(parts.length * 3), 3));
    const pts = new THREE.Points(geo, new THREE.PointsMaterial({
      size: 0.17, map: dotTexture(), vertexColors: true, transparent: true, depthWrite: false,
      blending: THREE.AdditiveBlending, sizeAttenuation: true,
    }));
    pts.frustumCulled = false;
    root.add(pts);
    const flowCol = flows.map((f) => new THREE.Color(f.color));
    const P = new THREE.Vector3();

    return function animate(t) {
      const sk = ease(seg(t, ...T.skeleton));
      for (const s of stageObjs) setOpacity(s.obj, sk * (s.dim + (1 - s.dim) * ease(seg(t, ...s.win))));
      for (const l of labels) setOpacity(l.obj, ease(seg(t, ...l.win)));
      enc.units.forEach((u, i) => {
        const on = ease(seg(t, T.enc[0] + (i / enc.units.length) * 0.6, T.enc[0] + (i / enc.units.length) * 0.6 + 0.3));
        u.material.opacity = sk * (0.2 + 0.8 * on);
      });
      head.units.forEach((u, i) => {
        const on = ease(seg(t, T.head[0] + (i / head.units.length) * 0.4, T.head[0] + (i / head.units.length) * 0.4 + 0.3));
        u.material.opacity = sk * (0.2 + 0.8 * on);
      });
      const bq = ease(seg(t, ...T.bloch));
      qubits.forEach(({ s, target }) => s.userData.set(target.theta * bq, target.phi * bq));
      gc.set(row.p_classical * ease(seg(t, ...T.probs)));
      gq.set(row.p_quantum * ease(seg(t, ...T.probs)));
      gf.set(row.p_final * ease(seg(t, ...T.final)));
      const rg = ease(seg(t, ...T.ring));
      arcC.geometry.setDrawRange(0, 6 * Math.round(SEGS * rg));
      arcQ.geometry.setDrawRange(0, 6 * Math.round(SEGS * rg));
      core.rotation.y = t * 0.6;
      core.rotation.x = t * 0.35;
      edges.forEach((line, i) => {
        const w = flows[i].win;
        const active = seg(t, w[0], w[0] + 0.3) * (1 - seg(t, w[1], w[1] + 0.6));
        const after = t > w[1] ? 0.22 : 0.08;
        line.material.opacity = sk * Math.max(after, 0.55 * active);
      });
      const pos = geo.attributes.position, col = geo.attributes.color;
      parts.forEach((pt, i) => {
        const u = (t - pt.te) / TRAVEL;
        if (u <= 0 || u >= 1) { pos.setXYZ(i, 0, -100, 0); col.setXYZ(i, 0, 0, 0); return; }
        flows[pt.f].curve.getPoint(u, P);
        pos.setXYZ(i, P.x, P.y, P.z);
        const fade = Math.min(1, u * 6, (1 - u) * 5);
        const c = flowCol[pt.f];
        col.setXYZ(i, c.r * fade, c.g * fade, c.b * fade);
      });
      pos.needsUpdate = true;
      col.needsUpdate = true;
    };
  }

  function caption(t) {
    const row = info.row;
    const k = row.z_latent.length;
    const d = q.architecture.in_features;
    const m = q.metrics, bs = q.branch_summary;
    const muted = (x) => `<span style="color:#94a3b8">${x}</span>`;
    let html = `<div style="font-size:14px"><b>QuHaMA</b> · Quantum-Hybrid Adaptive Multi-branch Architecture · test patient #${ctx.patient}${t < T.metrics[0] ? ` ${muted("· one patient flows through the trained network")}` : ""}`;
    let line;
    if (t < T.toBr[0]) {
      line = `① The <b style="color:${hex(PALETTE.classical)}">encoder</b> (small neural network) compresses all ${d} standardised measurements into k = ${k} latent numbers, one per qubit.`;
      if (t >= T.lat[0]) line += ` z = [${row.z_latent.map(n2).join(", ")}] (tanh, between −1 and +1; brightness = |z|).`;
    } else if (t < T.toGate[0]) {
      line = `② Two branches read z. ${span(PALETTE.classical, "Classical head")}: ${t >= T.probs[0] ? `P_c = <b>${pct1(row.p_classical)}%</b>` : "…"} · ${span(PALETTE.quantum, "Quantum branch")}: angles θ = (z+1)·π/2 = [${row.z_angles.map((a) => a.toFixed(2)).join(", ")}] → QSVC (ZZFeatureMap, ${k} qubits): ${t >= T.probs[0] ? `P_q = <b>${pct1(row.p_quantum)}%</b>` : "…"}`;
    } else if (t < T.toFinal[0]) {
      line = `③ The ${span(GATE_C, "gating network")} reads [z, P_c, P_q] and picks this patient's weights: ${t >= T.ring[0] ? `${span(PALETTE.classical, `Classical ${pct1(row.w_classical)}%`)} · ${span(PALETTE.quantum, `Quantum ${pct1(row.w_quantum)}%`)}` : "…"}`;
    } else {
      line = `④ Blended P = ${row.w_classical.toFixed(2)} × ${pct1(row.p_classical)}% + ${row.w_quantum.toFixed(2)} × ${pct1(row.p_quantum)}% = <b>${pct1(row.p_final)}%</b>`;
      if (t >= T.final[1] - 0.2) {
        const v = info.disease ? span(PALETTE.sick, "DISEASE") : span(PALETTE.healthy, "HEALTHY");
        const a = info.actualSick ? span(PALETTE.sick, "Disease") : span(PALETTE.healthy, "Healthy");
        line += ` → ${v} · actual: ${a} ${info.correct ? span(PALETTE.healthy, "✓ correct") : span(PALETTE.sick, "✗ misclassified")}`;
      }
    }
    html += `<div style="margin-top:3px">${line}</div>`;
    if (t >= T.metrics[0]) {
      const op = ease(seg(t, ...T.metrics)).toFixed(2);
      const auc = m["ROC-AUC"];
      const kpi = (label, val, warn) => `<div style="background:rgba(255,255,255,.04);border:1px solid #2a3350;border-radius:6px;padding:1px 8px;text-align:center;line-height:1.25"><div style="font-size:11px;color:#94a3b8">${label}</div><div style="font-weight:700;${warn ? "color:#fbbf24" : ""}">${val}</div></div>`;
      html += `<div style="opacity:${op};margin-top:6px"><div style="font-size:12px;color:#94a3b8">QuHaMA on all ${info.nTest} test patients of this disease</div>`;
      html += `<div style="display:grid;grid-template-columns:repeat(5,auto);gap:6px;margin-top:3px">${[
        kpi("Accuracy", `${pct1(m.Accuracy)}%`),
        kpi("F2-Score", m["F2-Score"].toFixed(3)),
        kpi("Recall", `${pct1(m.Recall)}%`),
        kpi("Precision", `${pct1(m.Precision)}%`),
        kpi("ROC-AUC", auc.toFixed(3), info.unreliable),
      ].join("")}</div>`;
      html += `<div style="font-size:12px;margin-top:4px;color:#cbd5e1">Each branch alone: ${span(PALETTE.classical, `classical ${pct1(bs.classical_accuracy)}%`)} · ${span(PALETTE.quantum, `quantum ${pct1(bs.quantum_accuracy)}%`)} accuracy · mean gate weight: classical ${pct1(bs.mean_w_classical)}% / quantum ${pct1(bs.mean_w_quantum)}%</div>`;
      if (info.unreliable) {
        html += `<div style="margin-top:4px;color:#fbbf24;font-weight:600">⚠ Test ROC-AUC ${auc.toFixed(3)} is close to a coin flip (0.5): QuHaMA is not reliable on this dataset. This verdict illustrates the architecture; it is not a diagnosis.</div>`;
      } else {
        html += `<div style="font-size:12px;margin-top:2px;color:#94a3b8">Small test set (${info.nTest} patients): treat these numbers as indicative.</div>`;
      }
      html += `</div>`;
    }
    return html + `</div>`;
  }

  return {
    title: "QuHaMA: hybrid classical + quantum network",
    legend: LEGEND_QUHAMA,
    scene,
    cameraPos: [0, 0.9, 11.6],
    // Timeline starts when the data has loaded: stops/duration are shifted by that moment.
    get duration() {
      if (status === "error") return 4;
      return t0 === null ? Infinity : t0 + DURATION;
    },
    get stops() { return t0 === null ? [] : STOPS.map((s) => s + t0); },
    update(t) {
      if (status === "loading") {
        lastT = t;
        return `<div style="font-size:14px"><b>QuHaMA</b> · loading the hybrid model's data…</div>`;
      }
      if (status === "error") {
        return `<div style="font-size:14px"><b>QuHaMA</b> · <span style="color:#fb7185">${errMsg}${/[.!?]$/.test(errMsg) ? "" : "."}</span> Other steps still work.</div>`;
      }
      if (!built) built = build();
      const rt = t - t0;
      built(rt);
      return caption(rt);
    },
    dispose() {
      disposed = true;
      disposeTree(scene);
    },
  };
}
