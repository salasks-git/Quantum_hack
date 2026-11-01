import * as THREE from "three";
import { PALETTE, ease, seg, textSprite, glowSprite, setOpacity, disposeTree, addLights } from "./common.js";
import { verdictFor } from "../verdict.js";

export const duration = 9;

// Timeline (seconds)
const T = {
  intro: [0.0, 0.7], // knobs + chart fade in
  train: [0.7, 6.9], // evaluations play back
  verdict: 6.9,
};

// Layout (world units)
const KR = 0.42; // knob radius
const KX = [-3.55, -2.45, -1.35]; // columns: qubit 1..3
const KY = [1.2, 0.0, -1.2]; // rows: RY layer 1..3
const CX0 = 0.15, CW = 3.9; // loss chart x range
const CY0 = -1.0, CH = 2.4; // loss chart y range
const FLOOR_Y = CY0 - 0.2;
const RADIAL = 8, PER_STEP = 12; // tube resolution
const ARC_SEGS = 128;

const hex = (c) => "#" + c.toString(16).padStart(6, "0");
const n3 = (v) => v.toFixed(3);
const pct = (p) => {
  const v = p * 100;
  return Math.round(v) === 50 ? v.toFixed(1) : String(Math.round(v));
};

/** Camera-facing label whose text can change (redraws its own canvas). */
function liveText({ color = "#e5e7eb", height = 0.16, chars = 8, weight = 600 } = {}) {
  const px = 64, c = document.createElement("canvas");
  c.width = Math.ceil(px * 0.62 * chars) + 24;
  c.height = Math.ceil(px * 1.3);
  const g = c.getContext("2d");
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false, depthTest: false }));
  const worldH = (height * c.height) / (px * 1.2);
  s.scale.set((worldH * c.width) / c.height, worldH, 1);
  s.renderOrder = 10;
  let last = null;
  s.userData.set = (text) => {
    if (text === last) return;
    last = text;
    g.clearRect(0, 0, c.width, c.height);
    g.font = `${weight} ${px}px system-ui, -apple-system, Segoe UI, sans-serif`;
    g.fillStyle = color;
    g.textAlign = "center";
    g.textBaseline = "middle";
    g.fillText(text, c.width / 2, c.height / 2);
    tex.needsUpdate = true;
  };
  return s;
}

/** One knob: dial, ring, 4 ticks, swept arc from 12 o'clock, needle. Positive angles turn clockwise. */
function makeKnob() {
  const k = new THREE.Group();
  const dial = new THREE.Mesh(
    new THREE.CircleGeometry(KR * 0.94, 48),
    new THREE.MeshBasicMaterial({ color: 0x0e1426, transparent: true, opacity: 0.85 })
  );
  dial.position.z = -0.02;
  const ring = new THREE.Mesh(
    new THREE.TorusGeometry(KR, 0.035, 12, 72),
    new THREE.MeshPhongMaterial({ color: PALETTE.quantum, emissive: PALETTE.quantum, emissiveIntensity: 0.45, shininess: 90 })
  );
  const tp = [];
  for (let i = 0; i < 4; i++) {
    const a = (i * Math.PI) / 2;
    const s = Math.sin(a), c = Math.cos(a);
    tp.push(s * KR * 0.78, c * KR * 0.78, 0, s * KR * 0.92, c * KR * 0.92, 0);
  }
  const ticks = new THREE.LineSegments(
    new THREE.BufferGeometry().setAttribute("position", new THREE.Float32BufferAttribute(tp, 3)),
    new THREE.LineBasicMaterial({ color: 0x94a3b8, transparent: true, opacity: 0.7 })
  );
  const arc = new THREE.Mesh(
    new THREE.RingGeometry(KR * 0.22, KR * 0.86, ARC_SEGS, 1, Math.PI / 2, Math.PI * 2),
    new THREE.MeshBasicMaterial({ color: PALETTE.quantum, transparent: true, opacity: 0.32, side: THREE.DoubleSide, depthWrite: false })
  );
  arc.position.z = -0.01;
  const needle = new THREE.Group();
  const shaft = new THREE.Mesh(
    new THREE.BoxGeometry(0.035, KR * 0.86, 0.035),
    new THREE.MeshBasicMaterial({ color: PALETTE.quantum2 })
  );
  shaft.position.y = KR * 0.43;
  const tip = glowSprite(PALETTE.quantum2, 0.22, 1);
  tip.position.y = KR * 0.86;
  needle.add(shaft, tip);
  const cap = new THREE.Mesh(new THREE.SphereGeometry(0.05, 16, 12), new THREE.MeshBasicMaterial({ color: 0xffffff }));
  const pulse = glowSprite(0xffffff, KR * 3.2, 0.8);
  pulse.material.color.setHex(PALETTE.quantum2);
  pulse.material.opacity = 0;
  pulse.position.z = -0.05;
  k.add(pulse, dial, arc, ring, ticks, needle, cap);
  k.userData.set = (theta) => {
    needle.rotation.z = -theta;
    const a = Math.min(Math.abs(theta), Math.PI * 2);
    arc.scale.x = theta >= 0 ? -1 : 1; // mirror so a positive sweep runs clockwise
    arc.geometry.setDrawRange(0, Math.round((a / (Math.PI * 2)) * ARC_SEGS) * 6);
  };
  k.userData.pulse = pulse;
  return k;
}

export default function createScene(ctx) {
  const { data } = ctx;
  const pi = ctx.patient;
  const p = data.test[pi];
  const loss = data.vqc.loss_history;
  const W = data.vqc.weights_history;
  const E = Math.min(loss.length, W.length);
  const NK = W[0].length; // 9 for 3 qubits, reps=2
  const lMax = Math.max(...loss.slice(0, E)), lMin = Math.min(...loss.slice(0, E));
  const lRange = lMax - lMin;
  const best = loss.indexOf(lMin);

  const scene = new THREE.Scene();
  addLights(scene);

  // ---------------- knobs (RealAmplitudes: θ[k], layer = floor(k/3), qubit = k mod 3) ----------------
  const knobs = new THREE.Group();
  scene.add(knobs);
  const knobList = [];
  for (let k = 0; k < NK; k++) {
    const row = Math.floor(k / 3), col = k % 3;
    const kn = makeKnob();
    kn.position.set(KX[col] ?? KX[2] + (col - 2) * 1.1, KY[row] ?? KY[2] - (row - 2) * 1.2, 0);
    knobs.add(kn);
    const name = textSprite(`θ[${k}]`, { height: 0.13, color: "#c4b5fd" });
    name.position.set(kn.position.x - KR * 0.95, kn.position.y + KR * 0.95, 0.05);
    knobs.add(name);
    const val = liveText({ height: 0.15, color: "#f0abfc", chars: 6 });
    val.position.set(kn.position.x, kn.position.y - KR - 0.15, 0.05);
    knobs.add(val);
    knobList.push({ kn, val });
  }
  [0, 1, 2].forEach((c) => {
    const s = textSprite(`qubit ${c + 1}`, { height: 0.16, color: "#a78bfa" });
    s.position.set(KX[c], KY[0] + KR + 0.42, 0);
    knobs.add(s);
  });
  [0, 1, 2].forEach((r) => {
    const s = textSprite(`RY layer ${r + 1}`, { height: 0.14, color: "#a78bfa" });
    s.position.set(KX[0] - KR - 0.62, KY[r], 0);
    knobs.add(s);
  });

  // ---------------- loss chart ----------------
  const chart = new THREE.Group();
  scene.add(chart);
  const xOf = (e) => CX0 + (E > 1 ? (CW * e) / (E - 1) : 0);
  const yOf = (l) => CY0 + (lRange > 1e-12 ? (CH * (l - lMin)) / lRange : CH / 2);
  const pts = loss.slice(0, E).map((l, e) => new THREE.Vector3(xOf(e), yOf(l), 0));

  const floor = new THREE.GridHelper(CW + 0.8, 12, 0x334155, 0x1e293b);
  floor.scale.z = 0.35;
  floor.position.set(CX0 + CW / 2, FLOOR_Y, 0);
  chart.add(floor);
  const axisMat = new THREE.LineBasicMaterial({ color: 0x64748b, transparent: true, opacity: 0.8 });
  chart.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(CX0 - 0.15, CY0 + CH + 0.15, 0), new THREE.Vector3(CX0 - 0.15, FLOOR_Y, 0),
    new THREE.Vector3(CX0 + CW + 0.2, FLOOR_Y, 0),
  ]), axisMat));
  // dashed guide at the lowest loss
  const guide = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(CX0 - 0.15, yOf(lMin), 0), new THREE.Vector3(CX0 + CW, yOf(lMin), 0)]),
    new THREE.LineDashedMaterial({ color: 0x64748b, dashSize: 0.08, gapSize: 0.06, transparent: true, opacity: 0.6 })
  );
  guide.computeLineDistances();
  chart.add(guide);
  const lab = (text, x, y, opts) => { const s = textSprite(text, opts); s.position.set(x, y, 0); chart.add(s); return s; };
  lab(n3(lMax), CX0 - 0.5, yOf(lMax), { height: 0.14, color: "#cbd5e1" });
  lab(n3(lMin), CX0 - 0.5, yOf(lMin) + 0.06, { height: 0.14, color: "#cbd5e1" });
  lab("loss", CX0 - 0.15, CY0 + CH + 0.4, { height: 0.17, color: "#e5e7eb" });
  lab("1", xOf(0), FLOOR_Y - 0.24, { height: 0.15, color: "#cbd5e1" });
  lab(String(E), xOf(E - 1), FLOOR_Y - 0.24, { height: 0.15, color: "#cbd5e1" });
  lab("objective evaluation", CX0 + CW / 2, FLOOR_Y - 0.26, { height: 0.14, color: "#cbd5e1" });

  // tube through the real losses, revealed progressively (index-uniform parameterisation)
  const curve = new THREE.CatmullRomCurve3(pts.length > 1 ? pts : [pts[0], pts[0].clone().add(new THREE.Vector3(1e-3, 0, 0))], false, "centripetal");
  curve.getPointAt = (u, out) => curve.getPoint(u, out);
  curve.getTangentAt = (u, out) => curve.getTangent(u, out);
  const SEGS = Math.max(1, E - 1) * PER_STEP;
  const tubeGeo = new THREE.TubeGeometry(curve, SEGS, 0.04, RADIAL, false);
  const cols = new Float32Array(tubeGeo.attributes.position.count * 3);
  const cA = new THREE.Color(PALETTE.quantum), cB = new THREE.Color(PALETTE.quantum2), cT = new THREE.Color();
  for (let v = 0; v < tubeGeo.attributes.position.count; v++) {
    cT.copy(cA).lerp(cB, Math.floor(v / (RADIAL + 1)) / SEGS).toArray(cols, v * 3);
  }
  tubeGeo.setAttribute("color", new THREE.BufferAttribute(cols, 3));
  tubeGeo.setDrawRange(0, 0);
  const tube = new THREE.Mesh(tubeGeo, new THREE.MeshPhongMaterial({
    vertexColors: true, emissive: PALETTE.quantum, emissiveIntensity: 0.35, shininess: 90,
  }));
  chart.add(tube);
  // drop lines from each evaluation to the floor, and a dot per evaluation
  const dl = [];
  pts.forEach((v) => dl.push(v.x, v.y, 0, v.x, FLOOR_Y, 0));
  const dropGeo = new THREE.BufferGeometry().setAttribute("position", new THREE.Float32BufferAttribute(dl, 3));
  dropGeo.setDrawRange(0, 0);
  chart.add(new THREE.LineSegments(dropGeo, new THREE.LineBasicMaterial({ color: PALETTE.quantum, transparent: true, opacity: 0.22 })));
  const dotGeo = new THREE.SphereGeometry(0.058, 12, 10);
  const evalDots = pts.map((v) => {
    const m = new THREE.Mesh(dotGeo, new THREE.MeshBasicMaterial({ color: 0xfae8ff, transparent: true, opacity: 0.9 }));
    m.position.copy(v);
    m.visible = false;
    chart.add(m);
    return m;
  });
  const head = new THREE.Group();
  head.add(new THREE.Mesh(new THREE.SphereGeometry(0.08, 16, 12), new THREE.MeshBasicMaterial({ color: 0xffffff })));
  head.add(glowSprite(PALETTE.quantum2, 0.7, 1));
  const headLab = liveText({ height: 0.14, color: "#ffffff", chars: 7 });
  headLab.position.set(0, 0.25, 0);
  head.add(headLab);
  chart.add(head);
  // lowest-loss marker (shown after training)
  const bestMark = new THREE.Group();
  bestMark.position.copy(pts[best]);
  const bestRing = new THREE.Mesh(
    new THREE.TorusGeometry(0.13, 0.015, 8, 40),
    new THREE.MeshBasicMaterial({ color: PALETTE.healthy, transparent: true })
  );
  bestMark.add(bestRing, glowSprite(PALETTE.healthy, 0.55, 0.8));
  const bestLab = textSprite(`lowest ${n3(lMin)}\nevaluation ${best + 1}`, { height: 0.14, color: hex(PALETTE.healthy), bg: "rgba(7,11,22,0.8)" });
  bestLab.position.set(0, 0.48, 0);
  bestMark.add(bestLab);
  chart.add(bestMark);

  // ---------------- captions ----------------
  const v = verdictFor(p.models.vqc);
  const sickSpan = (txt) => `<span style="color:${hex(PALETTE.sick)}">${txt}</span>`;
  const okSpan = (txt) => `<span style="color:${hex(PALETTE.healthy)}">${txt}</span>`;
  const violet = (txt) => `<span style="color:${hex(PALETTE.quantum)}">${txt}</span>`;
  const cap = {
    intro: `<b>${violet("VQC")} learning</b> · ${NK} trainable rotation angles θ[0]–θ[${NK - 1}] (the knobs: RY gates, 3 layers × 3 qubits) · the COBYLA optimiser turns them to lower the training loss`,
    train: (e) => `<b>${violet("VQC")} training</b> · evaluation ${e}/${E} · loss ${n3(loss[e - 1])} · each evaluation runs the circuit on all ${data.train.labels.length} training patients with the knob settings shown`,
    verdict: `<b>${violet("VQC")} verdict: ${v.disease ? sickSpan("DISEASE") : okSpan("HEALTHY")}</b> · disease probability <b>${pct(p.models.vqc.prob)}%</b>${
      v.borderline ? " · <b>borderline</b>: the probability estimate and the decision disagree; the verdict follows the decision" : ""
    } · after ${E} objective evaluations · lowest loss ${n3(lMin)} at evaluation ${best + 1}`,
  };

  let lastE = -1;
  return {
    title: "VQC learning",
    scene,
    cameraPos: [0.1, 0.45, 6.9],
    duration,
    update(t) {
      const pIn = ease(seg(t, ...T.intro));
      setOpacity(knobs, pIn);
      setOpacity(floor, pIn);
      knobs.scale.setScalar(0.9 + 0.1 * pIn);

      // fractional evaluation index: ef in [0, E-1]
      const u = seg(t, ...T.train);
      const ef = u * Math.max(0, E - 1);
      const e0 = Math.min(E - 1, Math.floor(ef));
      const e1 = Math.min(E - 1, e0 + 1);
      const f = ease((ef - e0) * 1.6); // knobs settle in the first part of each step
      knobList.forEach(({ kn, val }, k) => {
        const a = W[e0][k], b = W[e1][k];
        kn.userData.set(a + (b - a) * f);
        const prev = e0 > 0 ? W[e0 - 1][k] : a;
        const moved = Math.min(1, Math.abs(a - prev) / 0.4);
        kn.userData.pulse.material.opacity = pIn * moved * (1 - Math.min(1, ef - e0)) * (u > 0 && u < 1 ? 1 : 0);
        if (e0 !== lastE) val.userData.set(W[e0][k].toFixed(2));
      });
      lastE = e0;

      tubeGeo.setDrawRange(0, Math.round((ef / Math.max(1, E - 1)) * SEGS) * RADIAL * 6);
      dropGeo.setDrawRange(0, (e0 + 1) * 2);
      evalDots.forEach((m, i) => { m.visible = u > 0 && i <= e0; });
      curve.getPoint(Math.min(1, ef / Math.max(1, E - 1)), head.position);
      headLab.userData.set(n3(loss[e0]));
      headLab.visible = t < T.verdict;
      setOpacity(head, pIn);

      const pB = ease(seg(t, T.verdict, T.verdict + 0.5));
      setOpacity(bestMark, pB);
      bestRing.rotation.y = t * 1.5;

      if (t < T.train[0]) return cap.intro;
      if (t < T.verdict) return cap.train(e0 + 1);
      return cap.verdict;
    },
    dispose() {
      disposeTree(scene);
    },
  };
}
