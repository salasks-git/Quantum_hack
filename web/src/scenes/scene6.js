import * as THREE from "three";
import { LEGEND_6 } from "./legends.js";
import { PALETTE, ease, seg, textSprite, glowSprite, setOpacity, disposeTree, addLights } from "./common.js";

// Scene 6: from simulator to real QPU.
// Part 1: the transpiled feature map U(x) (IBM native gates rz / sx / x / ecr from data.qpu.gates) slides in
//         along 3 qubit wires as 3D blocks, then an execution cursor sweeps through it.
// Part 2: log-scale time race: QSVC on IBM QPU (estimate) vs this CPU (measured) vs IBM cloud queue (typical).

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
export const duration = RESULT_AT + 1.7;

const AX0 = -2.0, AX1 = 4.1, AXIS_Y = -0.75;
const P2_LIFT = 0.3; // keeps the axis labels clear of the live panel (bottom-right) at 1280 px
const BAR_Y = [1.55, 0.75, -0.05], BAR_H = 0.42, BAR_D = 0.3;

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

  return {
    title: "Simulator → real QPU",
    legend: LEGEND_6,
    scene,
    cameraPos: [0, 0.45, 8.5],
    duration,
    stops: [P1_STOP, RESULT_AT + 0.4],
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
      const inP2 = ease(seg(t, P2_IN0, P2_IN1));
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

      // ---------- caption ----------
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
