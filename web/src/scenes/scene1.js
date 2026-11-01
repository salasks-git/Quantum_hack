import * as THREE from "three";
import { LEGEND_1 } from "./legends.js";
import {
  PALETTE, ease, seg, blochSphere, firstLayerState, textSprite, glowSprite,
  setOpacity, disposeTree, addLights,
} from "./common.js";

export const duration = 10;

// Timeline (seconds)
const T = {
  grow: [0.1, 0.8], // raw bars rise
  std: [1.6, 2.3], // lerp to standardised heights
  labels: [2.2, 2.5], // top-5 labels fade in, held until the merge
  merge: [3.5, 4.35], // bars fly into the centre
  orbs: [4.1, 4.55], // 3 PCA orbs emerge
  bloch: [4.7, 5.05], // orbs turn into Bloch spheres
  tilt: [5.05, 5.45], // |0> -> equator (Hadamard)
  phase: [5.45, 7.2], // phase 2*angle
};

const BASE_Y = -0.95; // bar baseline
const H0 = 1.55; // world units per height unit (shrunk per patient if a bar would crowd the frame)
const MAX_BAR_TOP = 1.7; // tallest standardised bar may reach this y
const LABEL_CEIL = 2.65; // labels never extend above this y
const QX = [-2.5, 0, 2.5]; // qubit positions
const QY = 0.6;
const SUB = ["₁", "₂", "₃"];

const fmt = (v) => {
  const a = Math.abs(v);
  if (a >= 1000) return Math.round(v).toLocaleString("en-US");
  if (a >= 100) return v.toFixed(0);
  if (a >= 10) return v.toFixed(1);
  if (a >= 1) return v.toFixed(2);
  return v.toPrecision(2);
};
const pretty = (name) => name.replace(/_/g, " ");
const short = (name, n = 18) => (name.length > n ? name.slice(0, n - 1) + "…" : name);
const n2 = (v) => v.toFixed(2).replace("-", "\u2212");
const sgn = (v) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(2);

export default function createScene(ctx) {
  const { data } = ctx;
  const p = data.test[ctx.patient];
  const names = data.feature_names;
  const N = names.length;
  const scene = new THREE.Scene();
  addLights(scene);

  // ---------- bars ----------
  const maxAbs = Math.max(...data.feature_ranges.max.map(Math.abs));
  const hRaw = p.raw.map((v) => {
    const h = v / maxAbs;
    return Math.abs(h) < 0.012 ? (h < 0 ? -0.012 : 0.012) : h; // keep a visible sliver
  });
  const hStd = p.standardized.map((z) => Math.max(0.05, 0.5 + 0.35 * z));
  const H = Math.min(H0, (MAX_BAR_TOP - BASE_Y) / Math.max(...hStd));
  const spacing = Math.min(0.9, 7.2 / N);
  const width = Math.min(0.42, spacing * 0.62);
  const barX = (i) => (i - (N - 1) / 2) * spacing;

  const barGroup = new THREE.Group();
  scene.add(barGroup);
  const barGeo = new THREE.BoxGeometry(1, 1, 1);
  barGeo.translate(0, 0.5, 0); // base at origin
  const bars = names.map((_, i) => {
    const m = new THREE.Mesh(
      barGeo,
      new THREE.MeshPhongMaterial({ color: PALETTE.classical, emissive: 0x0c4a6e, transparent: true })
    );
    m.position.set(barX(i), BASE_Y, 0);
    barGroup.add(m);
    return m;
  });

  const floorW = N * spacing + 0.8;
  const floor = new THREE.GridHelper(floorW, Math.max(8, Math.round(floorW / 0.45)), 0x334155, 0x1e293b);
  floor.scale.z = 1.6 / floorW; // a narrow strip under the bars
  floor.position.y = BASE_Y;
  scene.add(floor);

  // labels: all when few features, else the 5 tallest standardised values
  const order = names.map((_, i) => i).sort((a, b) => p.standardized[b] - p.standardized[a]);
  const labelled = (N <= 8 ? [...order] : order.slice(0, 5)).sort((a, b) => a - b);
  const labelGroup = new THREE.Group();
  scene.add(labelGroup);
  const placed = [];
  labelled.forEach((i) => {
    const s = textSprite(`${short(pretty(names[i]))}\n${sgn(p.standardized[i])}σ`, {
      height: 0.17, color: "#e0f2fe", bg: "rgba(14,20,38,0.85)",
    });
    const w = s.scale.x, h = s.scale.y;
    const top = BASE_Y + hStd[i] * H;
    const clash = (yy) => placed.some((q) =>
      Math.abs(q.x - barX(i)) < (q.w + w) / 2 + 0.04 && Math.abs(q.y - yy) < (q.h + h) / 2 + 0.03);
    // greedy de-overlap: first free slot going up, but never above LABEL_CEIL;
    // if the ceiling is hit, fall back to the first free slot going down instead.
    const y0 = Math.min(top + 0.22 + h / 2, LABEL_CEIL - h / 2);
    let y = y0;
    for (let k = 0; clash(y) && y + 0.04 <= LABEL_CEIL - h / 2; k++) y += 0.04;
    if (clash(y)) {
      y = y0;
      while (clash(y) && y > BASE_Y + h / 2) y -= 0.04;
    }
    placed.push({ x: barX(i), y, w, h });
    s.position.set(barX(i), y, 0.05);
    labelGroup.add(s);
    const lead = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(barX(i), top + 0.02, 0), new THREE.Vector3(barX(i), y - h / 2, 0),
      ]),
      new THREE.LineBasicMaterial({ color: 0x7dd3fc, transparent: true, opacity: 0.6 })
    );
    labelGroup.add(lead);
  });
  setOpacity(labelGroup, 0);

  // ---------- PCA orbs ----------
  const maxPca = Math.max(...p.pca.map(Math.abs), 1e-6);
  const CENTER = new THREE.Vector3(0, QY, 0);
  const orbs = p.pca.map((v, k) => {
    const g = new THREE.Group();
    const r = 0.2 + 0.5 * (Math.abs(v) / maxPca);
    const ball = new THREE.Mesh(
      new THREE.SphereGeometry(r, 32, 24),
      new THREE.MeshStandardMaterial({
        color: PALETTE.quantum, emissive: PALETTE.quantum, emissiveIntensity: 0.3, roughness: 0.35, metalness: 0.1,
        transparent: true,
      })
    );
    g.add(ball, glowSprite(PALETTE.quantum, r * 4.2, 0.9));
    const lab = textSprite(`PC${k + 1} = ${n2(v)}`, { height: 0.2, color: "#ddd6fe" });
    lab.position.y = -r - 0.32;
    g.add(lab);
    g.visible = false;
    scene.add(g);
    return g;
  });

  // ---------- Bloch spheres ----------
  const R = 0.95;
  const qubits = p.angles.map((a, k) => {
    const g = new THREE.Group();
    g.position.set(QX[k], QY, 0);
    const b = blochSphere(R);
    g.add(b);
    const ket = textSprite("|0⟩", { height: 0.2, color: "#f0abfc" });
    ket.position.y = R + 0.28;
    g.add(ket);
    const lab = textSprite(`qubit ${k + 1} · x${SUB[k]} = ${n2(a)}`, { height: 0.19, color: "#e9d5ff" });
    lab.position.y = -R - 0.32;
    g.add(lab);
    // phase wedge in the equatorial plane, swept from phi = 0 to phi = 2*angle
    const phiEnd = firstLayerState(a).phi;
    const n = Math.max(8, Math.ceil((Math.abs(phiEnd) / (2 * Math.PI)) * 96));
    const trail = new THREE.Mesh(
      new THREE.CircleGeometry(R * 0.98, n, 0, phiEnd),
      new THREE.MeshBasicMaterial({
        color: PALETTE.quantum2, transparent: true, opacity: 0.22, side: THREE.DoubleSide, depthWrite: false,
      })
    );
    trail.rotation.x = -Math.PI / 2; // circle (cos a, sin a, 0) -> Bloch (cos phi, 0, -sin phi)
    trail.geometry.setDrawRange(0, 0);
    g.add(trail);
    g.userData = { b, trail, n, phiEnd };
    g.visible = false;
    scene.add(g);
    return g;
  });

  // ---------- captions ----------
  let iMax = 0, iMin = 0;
  p.raw.forEach((v, i) => {
    if (Math.abs(v) > Math.abs(p.raw[iMax])) iMax = i;
    if (Math.abs(v) < Math.abs(p.raw[iMin]) && v !== 0) iMin = i;
  });
  const top = names.reduce((b, _, i) => (Math.abs(p.standardized[i]) > Math.abs(p.standardized[b]) ? i : b), 0);
  const pcs = p.pca.map((v, k) => `PC${k + 1} ${n2(v)}`).join(" · ");
  const xs = p.angles.map((a, k) => `x${SUB[k]} = ${n2(a)}`).join(", ");
  const cap = {
    raw: `<b>1 · Patient record</b> · ${N} raw measurements${p.reconstructed ? " reconstructed from the 3 scores (approximate)" : ""} on very different scales — largest: <b>${pretty(names[iMax])} = ${fmt(p.raw[iMax])}</b>, smallest: ${pretty(names[iMin])} = ${fmt(p.raw[iMin])}`,
    std: `<b>2 · Standardise</b> · all ${N} measurements rescaled by the training patients' mean and spread (0σ = typical) · most unusual: <b>${pretty(names[top])} ${sgn(p.standardized[top])}σ</b>`,
    pca: `<b>3 · Compress (PCA)</b> · ${N} measurements squeezed into 3 principal components: <b>${pcs}</b> (orb size = |value|)`,
    enc: `<b>4 · Encode</b> · each component becomes a qubit angle: <b>${xs}</b> rad · Hadamard tips |0⟩ to the equator, then a phase turn of 2x · <i>first encoding layer only; later layers entangle the qubits</i>`,
  };

  const tmp = new THREE.Vector3();
  return {
    title: "Patient becomes qubits",
    legend: LEGEND_1,
    scene,
    cameraPos: [0, 2.1, 6.6],
    duration,
    stops: [T.grow[1], T.labels[1], T.orbs[1]],
    update(t) {
      const pGrow = ease(seg(t, ...T.grow));
      const pStd = ease(seg(t, ...T.std));
      const pLab = ease(seg(t, ...T.labels)) * (1 - ease(seg(t, T.merge[0] - 0.3, T.merge[0])));
      const pMerge = seg(t, ...T.merge);

      bars.forEach((b, i) => {
        const h = THREE.MathUtils.lerp(hRaw[i], hStd[i], pStd) * H * Math.max(0.001, pGrow);
        // stagger the flight to the centre so it reads as a stream
        const local = ease(seg(pMerge, (i / N) * 0.45, (i / N) * 0.45 + 0.55));
        tmp.set(barX(i), BASE_Y, 0).lerp(CENTER, local);
        b.position.copy(tmp);
        const sh = 1 - local;
        b.scale.set(width * (0.15 + 0.85 * sh), Math.max(0.001, Math.abs(h) * sh), width * (0.15 + 0.85 * sh));
        b.rotation.x = h < 0 ? Math.PI : 0; // negative raw values hang below the baseline
        b.visible = local < 0.995;
      });
      setOpacity(floor, 1 - ease(seg(t, T.merge[0], T.merge[1])));
      setOpacity(labelGroup, pLab);

      const pOrb = ease(seg(t, ...T.orbs));
      const pBloch = ease(seg(t, ...T.bloch));
      orbs.forEach((o, k) => {
        o.position.set(QX[k] * pOrb, QY, 0);
        o.scale.setScalar(Math.max(0.001, pOrb * (1 - pBloch)));
        o.visible = pOrb > 0 && pBloch < 0.99;
      });

      const pTilt = ease(seg(t, ...T.tilt));
      const pPhase = ease(seg(t, ...T.phase));
      qubits.forEach((q, k) => {
        q.visible = pBloch > 0;
        q.scale.setScalar(Math.max(0.001, pBloch));
        const { theta, phi } = firstLayerState(p.angles[k]);
        q.userData.b.userData.set(theta * pTilt, phi * pPhase);
        q.userData.trail.geometry.setDrawRange(0, pTilt >= 1 ? 3 * Math.round(q.userData.n * pPhase) : 0);
      });

      if (t < T.std[0]) return cap.raw;
      if (t < T.merge[0]) return cap.std;
      if (t < T.bloch[0]) return cap.pca;
      return cap.enc;
    },
    dispose() {
      disposeTree(scene);
    },
  };
}
