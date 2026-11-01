import * as THREE from "three";
import { LEGEND_3 } from "./legends.js";
import {
  PALETTE, ease, seg, blochSphere, firstLayerState, textSprite, glowSprite, setOpacity, disposeTree, addLights,
} from "./common.js";
import { verdictFor } from "../verdict.js";

export const duration = 12;

// Timeline (seconds)
const T = {
  grow: [0.2, 0.7], // sphere pairs appear
  tilt: [0.7, 1.3], // |0> -> equator
  phase: [1.3, 2.7], // phase 2*angle
  kernel: [2.7, 3.1], // K badge
  pairOut: [4.2, 4.7],
  webIn: [4.5, 5.0],
  threads: [4.8, 6.8], // training threads, strongest first
  you: [6.8, 7.3], // chosen patient appears
  yourThreads: [7.3, 8.6], // its top-5 threads, one by one
  verdict: 8.6,
};

const S = 0.9; // world units per radian in the web
const WEB_Y = 0.5;
const COLS = [-1.9, 0, 1.9];
const ROWS = [1.3, -0.55];
const R = 0.62;
const hex = (c) => "#" + c.toString(16).padStart(6, "0");
const n3 = (v) => v.toFixed(3);
const pct = (p) => {
  const v = p * 100;
  return Math.round(v) === 50 ? v.toFixed(1) : String(Math.round(v));
};

function topK(row, k, exclude = -1) {
  return row
    .map((v, j) => [v, j])
    .filter(([, j]) => j !== exclude)
    .sort((a, b) => b[0] - a[0])
    .slice(0, k)
    .map(([, j]) => j);
}

export default function createScene(ctx) {
  const { data } = ctx;
  const pi = ctx.patient;
  const p = data.test[pi];
  const kTest = data.qsvc.kernel_test[pi];
  const kTrain = data.qsvc.kernel_train;
  const tAngles = data.train.angles;
  const tLabels = data.train.labels;
  const nTrain = tAngles.length;
  const scene = new THREE.Scene();
  addLights(scene);

  // ================= Part 1: two encoded states, qubit by qubit =================
  const match = topK(kTest, 1)[0];
  const K = kTest[match];
  const matchSick = tLabels[match] === 1;
  const pair = new THREE.Group();
  scene.add(pair);
  const rows = [p.angles, tAngles[match]].map((angles, r) => {
    const spheres = angles.map((a, k) => {
      const b = blochSphere(R);
      b.position.set(COLS[k], ROWS[r], 0);
      pair.add(b);
      if (r === 0) {
        const q = textSprite(`qubit ${k + 1}`, { height: 0.15, color: "#c4b5fd" });
        q.position.set(COLS[k], ROWS[0] + R + 0.22, 0);
        pair.add(q);
      }
      return { b, a };
    });
    return spheres;
  });
  const youLab = textSprite(`Your patient\n#${pi}`, { height: 0.18, color: "#ffffff" });
  youLab.position.set(-3.35, ROWS[0], 0);
  const matchLab = textSprite(`Training patient #${match}\n${matchSick ? "Disease" : "Healthy"}`, {
    height: 0.18, color: hex(matchSick ? PALETTE.sick : PALETTE.healthy),
  });
  matchLab.position.set(-3.35, ROWS[1], 0);
  pair.add(youLab, matchLab);
  const kBadge = textSprite(`K = ${n3(K)}`, { height: 0.3, color: "#f0abfc", bg: "rgba(14,20,38,0.85)" });
  kBadge.position.set(3.35, (ROWS[0] + ROWS[1]) / 2, 0);
  pair.add(kBadge);
  const kSub = textSprite("quantum similarity", { height: 0.13, color: "#c4b5fd" });
  kSub.position.set(3.35, (ROWS[0] + ROWS[1]) / 2 - 0.36, 0);
  pair.add(kSub);

  // ================= Part 2: the kernel web =================
  const web = new THREE.Group();
  web.position.y = WEB_Y;
  scene.add(web);
  const toPos = (a) => new THREE.Vector3((a[0] - Math.PI / 2) * S, (a[1] - Math.PI / 2) * S, (a[2] - Math.PI / 2) * S);

  const box = new THREE.LineSegments(
    new THREE.EdgesGeometry(new THREE.BoxGeometry(Math.PI * S, Math.PI * S, Math.PI * S)),
    new THREE.LineBasicMaterial({ color: PALETTE.quantum, transparent: true, opacity: 0.18 })
  );
  web.add(box);
  const h = (Math.PI * S) / 2;
  [["x₁", [h + 0.25, -h, -h]], ["x₂", [-h, h + 0.22, -h]], ["x₃", [-h - 0.2, -h, h + 0.25]]].forEach(([t, xyz]) => {
    const s = textSprite(t, { height: 0.18, color: "#a78bfa" });
    s.material.sizeAttenuation = false; // same on-screen size at the front and back of the box
    s.scale.multiplyScalar(0.034 / s.scale.y);
    s.position.set(...xyz);
    web.add(s);
  });

  const dotGeo = new THREE.SphereGeometry(0.065, 12, 10);
  const trainPos = tAngles.map(toPos);
  const dots = trainPos.map((v, j) => {
    const col = tLabels[j] === 1 ? PALETTE.sick : PALETTE.healthy;
    const m = new THREE.Mesh(dotGeo, new THREE.MeshBasicMaterial({ color: col, transparent: true }));
    m.position.copy(v);
    const gl = glowSprite(col, 0.32, 0.7);
    m.add(gl);
    web.add(m);
    return m;
  });

  // training threads: each patient to its top-3 kernel neighbours (excluding itself), deduped
  const seen = new Map();
  for (let i = 0; i < nTrain; i++) {
    for (const j of topK(kTrain[i], 3, i)) {
      const key = i < j ? `${i}-${j}` : `${j}-${i}`;
      if (!seen.has(key)) seen.set(key, [i, j, kTrain[i][j]]);
    }
  }
  const links = [...seen.values()].sort((a, b) => b[2] - a[2]);
  const lp = new Float32Array(links.length * 6);
  const lc = new Float32Array(links.length * 6);
  const pink = new THREE.Color(PALETTE.quantum2);
  links.forEach(([i, j, k], n) => {
    trainPos[i].toArray(lp, n * 6);
    trainPos[j].toArray(lp, n * 6 + 3);
    const w = Math.min(1, Math.max(0, k)) * 0.6; // additive blend: brightness (opacity) proportional to K
    for (const o of [0, 3]) {
      lc[n * 6 + o] = pink.r * w;
      lc[n * 6 + o + 1] = pink.g * w;
      lc[n * 6 + o + 2] = pink.b * w;
    }
  });
  const lgeo = new THREE.BufferGeometry();
  lgeo.setAttribute("position", new THREE.BufferAttribute(lp, 3));
  lgeo.setAttribute("color", new THREE.BufferAttribute(lc, 3));
  lgeo.setDrawRange(0, 0);
  const threads = new THREE.LineSegments(
    lgeo,
    new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false })
  );
  web.add(threads);

  // the chosen patient and its top-5 kernel_test threads
  const youPos = toPos(p.angles);
  const you = new THREE.Group();
  you.position.copy(youPos);
  const youBall = new THREE.Mesh(new THREE.SphereGeometry(0.12, 20, 16), new THREE.MeshBasicMaterial({ color: 0xffffff }));
  const youGlow = glowSprite(0xffffff, 0.75, 0.9);
  const youTag = textSprite(`#${pi}`, { height: 0.17, color: "#ffffff" });
  youTag.position.y = 0.3;
  you.add(youBall, youGlow, youTag);
  web.add(you);

  const near = topK(kTest, 5);
  const beamGeo = new THREE.CylinderGeometry(0.03, 0.03, 1, 8);
  beamGeo.translate(0, 0.5, 0);
  const up = new THREE.Vector3(0, 1, 0);
  const beams = near.map((j) => {
    const d = trainPos[j].clone().sub(youPos);
    const len = d.length();
    const m = new THREE.Mesh(
      beamGeo,
      new THREE.MeshBasicMaterial({
        color: PALETTE.quantum2, transparent: true, opacity: 0.25 + 0.75 * Math.min(1, Math.max(0, kTest[j])),
        blending: THREE.AdditiveBlending, depthWrite: false,
      })
    );
    m.position.copy(youPos);
    m.quaternion.setFromUnitVectors(up, d.normalize());
    m.userData.len = len;
    m.visible = false;
    web.add(m);
    return { m, j };
  });

  // ================= captions =================
  const v = verdictFor(p.models.qsvc);
  const sickSpan = (txt) => `<span style="color:${hex(PALETTE.sick)}">${txt}</span>`;
  const okSpan = (txt) => `<span style="color:${hex(PALETTE.healthy)}">${txt}</span>`;
  const lab = (sick) => (sick ? sickSpan("Disease") : okSpan("Healthy"));
  const firstLayer = "<i>spheres show the first encoding layer only; K comes from the full entangling circuit</i>";
  const nearSick = near.filter((j) => tLabels[j] === 1).length;
  const cap = {
    pairA: `<b>Quantum similarity</b> · your patient #${pi} vs training patient #${match}, the most similar of ${nTrain} · comparing their encoded states… · ${firstLayer}`,
    pairB: `<b>Quantum similarity</b> · your patient #${pi} vs training patient #${match}, the most similar of ${nTrain} · K = |⟨ψ<sub>${pi}</sub>|ψ<sub>${match}</sub>⟩|² = <b>${n3(K)}</b> · #${match}'s actual diagnosis: <b>${lab(matchSick)}</b> · ${firstLayer}`,
    web: `<b>The quantum kernel web</b> · ${nTrain} training patients placed by their 3 qubit angles (${okSpan("healthy")} / ${sickSpan("disease")}) · each is joined to its 3 most similar · brighter thread = higher K`,
    you: `<b>Your patient #${pi}</b> (white) joins the web · its 5 most similar training patients: K = ${near.map((j) => n3(kTest[j])).join(", ")} (${nearSick} ${sickSpan("disease")}, ${5 - nearSick} ${okSpan("healthy")})`,
    verdict: `<b>QSVC verdict: ${v.disease ? sickSpan("DISEASE") : okSpan("HEALTHY")}</b> · disease probability <b>${pct(p.models.qsvc.prob)}%</b>${
      v.borderline ? " · <b>borderline</b>: the probability estimate and the decision disagree; the verdict follows the decision" : ""
    } · the QSVC weighs similarity to all ${nTrain} training patients; the 5 strongest are lit`,
  };

  return {
    title: "QSVC quantum similarity",
    legend: LEGEND_3,
    scene,
    cameraPos: [0, 1.5, 7.2],
    duration,
    stops: [T.kernel[1], T.verdict],
    update(t) {
      // ---- part 1
      const pGrow = ease(seg(t, ...T.grow));
      const pTilt = ease(seg(t, ...T.tilt));
      const pPhase = ease(seg(t, ...T.phase));
      rows.forEach((row) => row.forEach(({ b, a }) => {
        const { theta, phi } = firstLayerState(a);
        b.userData.set(theta * pTilt, phi * pPhase);
      }));
      const pOut = ease(seg(t, ...T.pairOut));
      pair.scale.setScalar(Math.max(0.001, pGrow * (1 - 0.25 * pOut)));
      setOpacity(pair, 1 - pOut);
      const pK = ease(seg(t, ...T.kernel));
      kBadge.material.opacity = pK * (1 - pOut);
      kSub.material.opacity = pK * (1 - pOut);

      // ---- part 2
      const pIn = ease(seg(t, ...T.webIn));
      setOpacity(web, pIn);
      web.scale.setScalar(0.75 + 0.25 * pIn);
      threads.visible = pIn > 0;
      const pT = seg(t, ...T.threads);
      lgeo.setDrawRange(0, 2 * Math.round(links.length * pT));

      const pYou = ease(seg(t, ...T.you));
      threads.material.opacity = pIn * (1 - 0.6 * pYou); // dim the web so the patient's own threads stand out
      you.visible = pYou > 0 && pIn > 0;
      you.scale.setScalar(Math.max(0.001, pYou * (1 + 0.12 * Math.sin(t * 5))));

      const span = T.yourThreads[1] - T.yourThreads[0];
      beams.forEach(({ m, j }, n) => {
        const s0 = T.yourThreads[0] + (n * span) / 6;
        const g = ease(seg(t, s0, s0 + span / 3));
        m.visible = g > 0;
        m.scale.set(1, Math.max(0.001, m.userData.len * g), 1);
        const d = dots[j];
        d.scale.setScalar(1 + 0.9 * g);
      });

      if (t < T.kernel[0]) return cap.pairA;
      if (t < T.pairOut[1]) return cap.pairB;
      if (t < T.you[0]) return cap.web;
      if (t < T.verdict) return cap.you;
      return cap.verdict;
    },
    dispose() {
      disposeTree(scene);
    },
  };
}
