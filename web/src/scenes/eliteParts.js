import * as THREE from "three";
import { PALETTE, ease, seg, blochSphere, textSprite, glowSprite, setOpacity } from "./common.js";
import { ELITE_NAMES, elitePatient, leaderboardView, honestyLines, fmtTime } from "../elite.js";
import { pct1 } from "../quhama.js";

// "Elite Hybrid Architectures" part of the QuHaMA scene (Streamlit Tab 8 parity):
// (a) three architecture cards (QuHaMA, Sandwich HQNN, Q-CKTA), (b) the tab's leaderboard as 3D bars on F2,
// (c) the current patient's elite verdicts. Every number comes from <slug>.elite.json; the kernel tile on the
// Q-CKTA card is a schematic whose sharpness follows the real exported alignment values.

// Timeline (seconds after the part starts)
export const E = {
  cardQ: [0.5, 1.4], // QuHaMA card
  cardS: [1.4, 2.3], // Sandwich card
  slabs: [1.6, 2.8], // slabs stack: Pre-Net, quantum, Post-Net
  sVals: [2.7, 3.4], // this patient's latent z / <Z> / P on the slabs
  cardC: [3.1, 4.0], // Q-CKTA card
  sweep: [3.6, 5.8], // bandwidth sweep: alignment bars, the tile sharpens
  cVals: [5.8, 6.3], // s*, alignments, fusion weights
  cardsOut: [7.0, 7.7],
  bars: [7.4, 9.0], // leaderboard bars grow (staggered)
  leader: [9.0, 9.6],
  patient: [10.4, 11.3], // this patient's elite verdicts
};
export const E_STOPS = [6.6, 10.0]; // after the cards, after the leaderboard
export const E_DURATION = 16;

const GOLD = 0xfbbf24;
const LATENT = 0xe9d5ff;
const AMBER = "#fbbf24";
const CARD_X = [-3.9, 0, 3.9], CARD_Y = 0.6, CARD_W = 3.4, CARD_H = 3.9;
const BAR_X = [-4.4, -2.2, 0, 2.2, 4.4], BAR_Y = -0.9, BAR_H = 3.0, BAR_W = 0.9;
const SUB = ["₁", "₂", "₃", "₄", "₅", "₆", "₇", "₈"];
const hex = (c) => "#" + c.toString(16).padStart(6, "0");
const n2 = (v) => v.toFixed(2).replace("-", "−");
const sgn = (v) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(2);
const span = (c, x) => `<b style="color:${hex(c)}">${x}</b>`;

function label(parent, text, x, y, opts = {}) {
  const s = textSprite(text, { height: 0.15, ...opts });
  s.position.set(x, y, 0.05);
  parent.add(s);
  return s;
}

function line(parent, a, b, color, opacity = 0.45) {
  const l = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints([a, b]),
    new THREE.LineBasicMaterial({ color, transparent: true, opacity, depthWrite: false })
  );
  parent.add(l);
  return l;
}

/** Translucent card panel with a glowing outline and a two-line title above it. */
function cardPanel(title, sub, color) {
  const g = new THREE.Group();
  const plane = new THREE.PlaneGeometry(CARD_W, CARD_H);
  g.add(new THREE.Mesh(plane, new THREE.MeshBasicMaterial({ color: 0x0e1426, transparent: true, opacity: 0.6, depthWrite: false })));
  g.add(new THREE.LineSegments(new THREE.EdgesGeometry(plane), new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.7 })));
  label(g, title, 0, CARD_H / 2 + 0.42, { height: 0.21, color: hex(color) });
  label(g, sub, 0, CARD_H / 2 + 0.17, { height: 0.13, color: "#94a3b8", weight: 500 });
  return g;
}

/** Thin slab (a network layer seen edge-on) with outline. */
function slab(w, h, d, color, op = 0.22) {
  const g = new THREE.Group();
  const box = new THREE.BoxGeometry(w, h, d);
  g.add(new THREE.Mesh(box, new THREE.MeshPhongMaterial({ color, emissive: color, emissiveIntensity: 0.25, transparent: true, opacity: op, depthWrite: false })));
  g.add(new THREE.LineSegments(new THREE.EdgesGeometry(box), new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.8 })));
  return g;
}

/** Split ring: cyan arc = classical weight, violet arc = quantum weight. */
function weightRing(wc, wq, r0, r1) {
  const g = new THREE.Group();
  const mk = (start, len, color) => new THREE.Mesh(
    new THREE.RingGeometry(r0, r1, 64, 1, start, len),
    new THREE.MeshBasicMaterial({ color, transparent: true, side: THREE.DoubleSide, depthWrite: false })
  );
  g.add(mk(Math.PI / 2, 2 * Math.PI * wc, PALETTE.classical), mk(Math.PI / 2 + 2 * Math.PI * wc, 2 * Math.PI * wq, PALETTE.quantum));
  return g;
}

export function buildElitePart(parent, { elite, data, patient, quhamaMetrics }) {
  const root = new THREE.Group();
  root.position.set(1.9, 1.3, 0); // right of the legend panel, above the caption
  parent.add(root);
  const A = elite.approaches;
  const view = leaderboardView(elite);
  const pat = elitePatient(elite, data, patient);
  const known = pat.ok && !pat.custom;
  const nTest = A.quhama.test.length;
  const fades = []; // {obj, win, parentOp?} sub-elements that fade in inside a card

  // ================= (a) architecture cards =================
  const cards = new THREE.Group();
  root.add(cards);

  // ---- QuHaMA: mini network (latent -> classical box / Bloch sphere -> gate with mean weights) ----
  const aq = A.quhama.architecture;
  const cQ = cardPanel("QuHaMA", "learned gating · per-patient [w_c, w_q]", GOLD);
  cQ.position.set(CARD_X[0], CARD_Y, 0);
  cards.add(cQ);
  const icoY = 0.45;
  const latPos = [0.42, 0, -0.42].map((dy) => new THREE.Vector3(-1.05, icoY + dy, 0));
  const boxPos = new THREE.Vector3(0, icoY + 0.45, 0), blochPos = new THREE.Vector3(0, icoY - 0.42, 0);
  const gatePos = new THREE.Vector3(1.0, icoY, 0);
  latPos.forEach((p) => {
    const m = new THREE.Mesh(new THREE.SphereGeometry(0.075, 16, 12), new THREE.MeshBasicMaterial({ color: LATENT, transparent: true }));
    m.position.copy(p);
    cQ.add(m);
    line(cQ, p, boxPos, PALETTE.classical);
    line(cQ, p, blochPos, PALETTE.quantum);
  });
  const miniBox = slab(0.5, 0.32, 0.3, PALETTE.classical, 0.15);
  miniBox.position.copy(boxPos);
  cQ.add(miniBox);
  const miniBloch = blochSphere(0.22);
  miniBloch.position.copy(blochPos);
  cQ.add(miniBloch);
  line(cQ, boxPos, gatePos, PALETTE.classical);
  line(cQ, blochPos, gatePos, PALETTE.quantum);
  const ico = new THREE.Mesh(
    new THREE.IcosahedronGeometry(0.14, 1),
    new THREE.MeshPhongMaterial({ color: GOLD, emissive: GOLD, emissiveIntensity: 0.35, flatShading: true, transparent: true })
  );
  ico.position.copy(gatePos);
  cQ.add(ico);
  const ring = weightRing(aq.mean_w_classical, aq.mean_w_quantum, 0.22, 0.3);
  ring.position.copy(gatePos);
  cQ.add(ring);
  label(cQ, "z", -1.05, icoY + 0.72, { height: 0.13, color: hex(LATENT) });
  label(cQ, "gate", 1.0, icoY + 0.45, { height: 0.13, color: hex(GOLD) });
  label(cQ, "mean gate weight (test patients)", 0, -0.55, { height: 0.13, color: "#cbd5e1", weight: 500 });
  label(cQ, `classical ${pct1(aq.mean_w_classical)}% · quantum ${pct1(aq.mean_w_quantum)}%`, 0, -0.8, { height: 0.15, color: "#ffffff" });
  label(cQ, `${aq.n_qubits} qubits · ${aq.feature_map} reps ${aq.reps}`, 0, -1.15, { height: 0.13, color: "#94a3b8", weight: 500 });
  label(cQ, `encoder ${aq.epochs_encoder} epochs · gating ${aq.epochs_gating} epochs`, 0, -1.4, { height: 0.13, color: "#94a3b8", weight: 500 });

  // ---- Sandwich: classical Pre-Net -> quantum layer (Bloch spheres) -> classical Post-Net ----
  const as = A.sandwich.architecture;
  const ql = as.quantum_layer || {};
  const cS = cardPanel("Sandwich HQNN", "layer-embedded · classical → quantum → classical", PALETTE.quantum);
  cS.position.set(CARD_X[1], CARD_Y, 0);
  cards.add(cS);
  const SL_Y = [1.25, 0.25, -0.75];
  const nq = as.n_qubits || 3;
  const pre = slab(2.6, 0.2, 0.8, PALETTE.classical);
  pre.position.y = SL_Y[0];
  const mid = slab(2.6, 0.2, 0.8, PALETTE.quantum);
  mid.position.y = SL_Y[1];
  const post = slab(2.6, 0.2, 0.8, PALETTE.classical);
  post.position.y = SL_Y[2];
  cS.add(pre, mid, post);
  const qbX = Array.from({ length: nq }, (_, j) => (j - (nq - 1) / 2) * 0.8);
  const sBloch = qbX.map((x) => {
    const b = blochSphere(0.2);
    b.position.set(x, SL_Y[1], 0);
    cS.add(b);
    return b;
  });
  const arrows = [line(cS, new THREE.Vector3(0, SL_Y[0] - 0.12, 0), new THREE.Vector3(0, SL_Y[1] + 0.22, 0), LATENT, 0.6),
    line(cS, new THREE.Vector3(0, SL_Y[1] - 0.22, 0), new THREE.Vector3(0, SL_Y[2] + 0.12, 0), PALETTE.quantum, 0.6)];
  const preL = label(cS, `Pre-Net  ${(as.pre_net || []).join(" → ")}  (classical)`, 0, SL_Y[0] + 0.27, { height: 0.14, color: hex(PALETTE.classical) });
  const midL = label(cS, `${ql.feature_map || "feature map"} + ansatz · ${nq} qubits → ⟨Z⟩`, 0, SL_Y[1] + 0.4, { height: 0.13, color: hex(PALETTE.quantum) });
  const postL = label(cS, `Post-Net  [z, ⟨Z⟩] ${(as.post_net || []).join(" → ")}  (classical)`, 0, SL_Y[2] + 0.27, { height: 0.14, color: hex(PALETTE.classical) });
  fades.push({ obj: pre, win: [E.slabs[0], E.slabs[0] + 0.5], card: cS }, { obj: preL, win: [E.slabs[0], E.slabs[0] + 0.5], card: cS });
  fades.push({ obj: mid, win: [E.slabs[0] + 0.35, E.slabs[0] + 0.85], card: cS }, { obj: midL, win: [E.slabs[0] + 0.35, E.slabs[0] + 0.85], card: cS });
  sBloch.forEach((b) => fades.push({ obj: b, win: [E.slabs[0] + 0.35, E.slabs[0] + 0.85], card: cS }));
  fades.push({ obj: arrows[0], win: [E.slabs[0] + 0.3, E.slabs[0] + 0.7], card: cS });
  fades.push({ obj: post, win: [E.slabs[1] - 0.5, E.slabs[1]], card: cS }, { obj: postL, win: [E.slabs[1] - 0.5, E.slabs[1]], card: cS });
  fades.push({ obj: arrows[1], win: [E.slabs[1] - 0.6, E.slabs[1] - 0.2], card: cS });
  let sTargets = null;
  if (known) {
    const st = pat.sandwich;
    if (Array.isArray(st.latent)) {
      const l = label(cS, `this patient: z = ${st.latent.map(sgn).join(" · ")}`, 0, SL_Y[0] - 0.26, { height: 0.13, color: hex(LATENT) });
      fades.push({ obj: l, win: E.sVals, card: cS });
    }
    if (Array.isArray(st.expectation)) {
      sTargets = st.expectation.map((v) => Math.acos(Math.max(-1, Math.min(1, v))));
      st.expectation.forEach((v, j) => {
        if (j >= qbX.length) return;
        const l = label(cS, `⟨Z${SUB[j] || j + 1}⟩ ${n2(v)}`, qbX[j], SL_Y[1] - 0.33, { height: 0.12, color: "#e9d5ff", weight: 500 });
        fades.push({ obj: l, win: E.sVals, card: cS });
      });
    }
    const l = label(cS, `→ P(disease) ${pct1(st.prob)}%`, 0, SL_Y[2] - 0.27, { height: 0.14, color: hex(st.pred === 1 ? PALETTE.sick : PALETTE.healthy) });
    fades.push({ obj: l, win: E.sVals, card: cS });
  } else {
    const l = label(cS, "custom patient: no precomputed values", 0, SL_Y[2] - 0.27, { height: 0.12, color: "#94a3b8", weight: 500 });
    fades.push({ obj: l, win: E.sVals, card: cS });
  }
  label(cS, `circuit depth ${ql.circuit_depth ?? "—"} · ${ql.circuit_size ?? "—"} gates`, 0, -1.45, { height: 0.13, color: "#94a3b8", weight: 500 });
  label(cS, `ansatz weights fixed at ${Array.isArray(ql.weights) && ql.weights.length ? ql.weights[0] : "—"} · Pre/Post-Net trained`, 0, -1.68, { height: 0.13, color: "#94a3b8", weight: 500 });

  // ---- Q-CKTA: kernel tile that sharpens with the alignment + the bandwidth sweep bars ----
  const ac = A.ckta.architecture;
  const cC = cardPanel("Q-CKTA", "kernel alignment · bandwidth-scaled quantum kernel", PALETTE.quantum2);
  cC.position.set(CARD_X[2], CARD_Y, 0);
  cards.add(cC);
  const sweep = (ac.bandwidth_sweep || []).filter((r) => typeof r.alignment === "number");
  const maxA = Math.max(1e-6, ...sweep.map((r) => r.alignment));
  const bestI = sweep.findIndex((r) => r.s === ac.best_bandwidth_s);
  const N = 10, CELL = 0.12, GAP = 0.016;
  const tileX = -0.72, tileY = 0.62;
  const cellGeo = new THREE.PlaneGeometry(CELL, CELL);
  const cells = [];
  const DARK = new THREE.Color(0x141a33), VIO = new THREE.Color(PALETTE.quantum), WHITE = new THREE.Color(0xf5f3ff);
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      const m = new THREE.Mesh(cellGeo, new THREE.MeshBasicMaterial({ color: DARK.clone(), transparent: true }));
      m.position.set(tileX + (j - (N - 1) / 2) * (CELL + GAP), tileY + ((N - 1) / 2 - i) * (CELL + GAP), 0);
      cC.add(m);
      cells.push({ m, diag: i === j, same: (i < N / 2) === (j < N / 2) });
    }
  }
  label(cC, "kernel K (schematic)", tileX, tileY + 0.95, { height: 0.12, color: "#cbd5e1", weight: 500 });
  label(cC, "healthy | sick", tileX, tileY - 0.88, { height: 0.11, color: "#94a3b8", weight: 500 });
  const BX0 = 0.32, BDX = 0.26, BMAX = 1.1, BBASE = tileY - 0.62;
  const aBars = sweep.map((r, k) => {
    const g = new THREE.BoxGeometry(0.17, 1, 0.12);
    g.translate(0, 0.5, 0);
    const color = k === bestI ? GOLD : PALETTE.quantum;
    const m = new THREE.Mesh(g, new THREE.MeshPhongMaterial({ color, emissive: color, emissiveIntensity: 0.35, transparent: true }));
    m.position.set(BX0 + k * BDX, BBASE, 0);
    m.scale.y = 0.001;
    cC.add(m);
    label(cC, `${r.s}`, BX0 + k * BDX, BBASE - 0.12, { height: 0.1, color: k === bestI ? hex(GOLD) : "#94a3b8", weight: 500 });
    return { m, h: Math.max(0.02, (Math.max(0, r.alignment) / maxA) * BMAX), a: r.alignment };
  });
  label(cC, "alignment vs s", BX0 + ((sweep.length - 1) * BDX) / 2, BBASE + BMAX + 0.3, { height: 0.12, color: "#cbd5e1", weight: 500 });
  const sweepCursor = glowSprite(PALETTE.quantum2, 0.45, 0.9);
  cC.add(sweepCursor);
  const cv1 = label(cC, `s* = ${ac.best_bandwidth_s} · A_q(s*) = ${typeof ac.alignment_quantum_best === "number" ? ac.alignment_quantum_best.toFixed(3) : "—"} · A_c(RBF) = ${typeof ac.alignment_classical_rbf === "number" ? ac.alignment_classical_rbf.toFixed(3) : "—"}`, 0, -0.75, { height: 0.13, color: "#ffffff" });
  const cv2 = label(cC, `fused kernel: classical ${pct1(ac.weight_classical)}% · quantum ${pct1(ac.weight_quantum)}%`, 0, -1.02, { height: 0.13, color: "#ffffff" });
  const fuse = weightRing(ac.weight_classical, ac.weight_quantum, 0.12, 0.18);
  fuse.position.set(0, -1.4, 0);
  cC.add(fuse);
  label(cC, `RBF γ = ${typeof ac.rbf_gamma === "number" ? ac.rbf_gamma.toFixed(3) : "—"} · precomputed-kernel SVM`, 0, -1.72, { height: 0.12, color: "#94a3b8", weight: 500 });
  fades.push({ obj: cv1, win: E.cVals, card: cC }, { obj: cv2, win: E.cVals, card: cC }, { obj: fuse, win: E.cVals, card: cC });

  const cardWins = [[cQ, E.cardQ], [cS, E.cardS], [cC, E.cardC]];

  // ================= (b) leaderboard bars =================
  const board = new THREE.Group();
  root.add(board);
  const x0 = BAR_X[0] - 0.8, x1 = BAR_X[BAR_X.length - 1] + 0.8;
  line(board, new THREE.Vector3(x0, BAR_Y, 0), new THREE.Vector3(x1, BAR_Y, 0), 0x94a3b8, 0.6);
  line(board, new THREE.Vector3(x0, BAR_Y + BAR_H, 0), new THREE.Vector3(x1, BAR_Y + BAR_H, 0), 0x94a3b8, 0.18);
  line(board, new THREE.Vector3(x0, BAR_Y + BAR_H / 2, 0), new THREE.Vector3(x1, BAR_Y + BAR_H / 2, 0), 0x94a3b8, 0.1);
  label(board, "F2 = 1", x0 - 0.42, BAR_Y + BAR_H, { height: 0.12, color: "#94a3b8", weight: 500 });
  label(board, "0.5", x0 - 0.3, BAR_Y + BAR_H / 2, { height: 0.12, color: "#94a3b8", weight: 500 });
  label(board, `Leaderboard · clinical F2 (recall-weighted) on the ${nTest} test patients`, 0, BAR_Y + BAR_H + 1.05, { height: 0.2, color: "#ffffff" });
  const bars = view.rows.slice(0, BAR_X.length).map((r, i) => {
    const color = r.elite ? PALETTE.quantum : PALETTE.classical;
    const geo = new THREE.BoxGeometry(BAR_W, 1, 0.6);
    geo.translate(0, 0.5, 0);
    const mesh = new THREE.Mesh(geo, new THREE.MeshPhongMaterial({ color, emissive: color, emissiveIntensity: r.leader ? 0.55 : 0.3, transparent: true, opacity: 0.9 }));
    mesh.position.set(BAR_X[i], BAR_Y, 0);
    mesh.scale.y = 0.001;
    board.add(mesh);
    if (r.elite) {
      mesh.add(new THREE.LineSegments(new THREE.EdgesGeometry(geo), new THREE.LineBasicMaterial({ color: GOLD, transparent: true, opacity: 0.9 })));
    }
    const val = label(board, r.f2.toFixed(3), BAR_X[i], BAR_Y + 0.2, { height: 0.19, color: "#ffffff" });
    const name = label(board, r.model, BAR_X[i], BAR_Y - 0.24, { height: 0.17, color: hex(r.elite ? GOLD : PALETTE.classical) });
    const fam = label(board, r.elite ? "elite hybrid" : "classical", BAR_X[i], BAR_Y - 0.46, { height: 0.12, color: "#94a3b8", weight: 500 });
    const warnTxt = [r.f2Low ? "⚠ low F2" : "", r.aucLow ? "⚠ AUC ≈ coin flip" : ""].filter(Boolean).join(" · ");
    const warn = warnTxt ? label(board, warnTxt, BAR_X[i], BAR_Y - 0.66, { height: 0.12, color: AMBER, weight: 600 }) : null;
    let glow = null, lead = null;
    if (r.leader) {
      glow = glowSprite(GOLD, 1.6, 0.8);
      board.add(glow);
      lead = label(board, view.leaders.length > 1 ? "★ tied leader" : "★ clinical leader", BAR_X[i], BAR_Y + 0.5, { height: 0.15, color: hex(GOLD) });
    }
    return { r, i, mesh, val, name, fam, warn, glow, lead, h: Math.max(0.001, r.f2 * BAR_H) };
  });

  // ================= (c) this patient's elite verdicts =================
  const pills = [];
  if (known) {
    pat.rows.forEach((pr) => {
      const i = view.rows.findIndex((r) => r.key === pr.key);
      if (i < 0 || i >= BAR_X.length) return;
      const c = pr.disease ? PALETTE.sick : PALETTE.healthy;
      const s = label(board, `${pr.disease ? "DISEASE" : "HEALTHY"} ${pct1(pr.prob)}% ${pr.correct ? "✓" : "✗"}${pr.borderline ? " ⚠" : ""}`,
        BAR_X[i], BAR_Y - 0.95, { height: 0.15, color: hex(c), bg: "rgba(14,20,38,0.85)" });
      pills.push(s);
    });
    const actual = label(board, `this patient · actual: ${pat.actualSick ? "Disease" : "Healthy"}`, (BAR_X[0] + BAR_X[2]) / 2, BAR_Y - 1.25, { height: 0.13, color: hex(pat.actualSick ? PALETTE.sick : PALETTE.healthy), weight: 500 });
    pills.push(actual);
  } else {
    pills.push(label(board, "elite models need a precomputed patient", (BAR_X[0] + BAR_X[2]) / 2, BAR_Y - 0.95, { height: 0.15, color: "#cbd5e1", bg: "rgba(14,20,38,0.85)" }));
  }

  // ================= animation =================
  const animate = (t) => {
    const out = 1 - ease(seg(t, ...E.cardsOut));
    for (const [c, w] of cardWins) setOpacity(c, out * ease(seg(t, ...w)));
    for (const f of fades) {
      const parentOn = cardWins.find(([c]) => c === f.card);
      setOpacity(f.obj, out * ease(seg(t, ...parentOn[1])) * ease(seg(t, ...f.win)));
    }
    ico.rotation.y = t * 0.6;
    ico.rotation.x = t * 0.35;
    miniBloch.userData.set(Math.PI / 2 * ease(seg(t, ...E.cardQ)), t * 0.8);
    const sv = ease(seg(t, ...E.sVals));
    sBloch.forEach((b, j) => b.userData.set(sTargets && sTargets[j] !== undefined ? sTargets[j] * sv : 0, 0));
    // bandwidth sweep: bars appear left to right; the tile's block contrast follows the current alignment,
    // then settles on the best bandwidth s*.
    const n = aBars.length;
    const u = seg(t, ...E.sweep);
    let sharp = 0;
    if (n) {
      const k = Math.min(n - 1, Math.floor(u * n));
      aBars.forEach((b, j) => {
        const grow = ease(seg(u * n - j, 0, 0.8));
        b.m.scale.y = Math.max(0.001, b.h * grow);
        b.m.material.emissiveIntensity = u < 1 && j === k ? 0.8 : j === bestI && u >= 1 ? 0.7 : 0.3;
      });
      const cur = u >= 1 ? (bestI >= 0 ? bestI : k) : k;
      const local = u >= 1 ? 1 : ease(Math.min(1, (u * n - k) * 2));
      const prevA = k > 0 ? Math.max(0, aBars[k - 1].a) : 0;
      sharp = u >= 1 ? Math.max(0, aBars[cur].a) / maxA : (prevA + (Math.max(0, aBars[k].a) - prevA) * local) / maxA;
      sweepCursor.position.set(BX0 + cur * BDX, BBASE + aBars[cur].h * (u > 0 ? 1 : 0) + 0.08, 0.05);
      sweepCursor.visible = u > 0 && out > 0.01;
    }
    for (const c of cells) {
      const v = c.diag ? 1 : c.same ? 0.08 + 0.72 * sharp : 0.08 + 0.05 * sharp;
      if (v < 0.6) c.m.material.color.lerpColors(DARK, VIO, v / 0.6);
      else c.m.material.color.lerpColors(VIO, WHITE, (v - 0.6) / 0.4);
    }
    // leaderboard
    const bOn = ease(seg(t, E.cardsOut[0] + 0.3, E.bars[0] + 0.4));
    setOpacity(board, bOn);
    bars.forEach((b) => {
      const st = E.bars[0] + b.i * 0.22;
      const g = ease(seg(t, st, st + 0.8));
      b.mesh.scale.y = Math.max(0.001, b.h * g);
      b.val.position.y = BAR_Y + b.h * g + 0.2;
      setOpacity(b.val, bOn * g);
      if (b.warn) setOpacity(b.warn, bOn * ease(seg(t, ...E.leader)));
      if (b.glow) {
        const lo = ease(seg(t, ...E.leader));
        b.glow.position.set(BAR_X[b.i], BAR_Y + b.h, 0.1);
        setOpacity(b.glow, bOn * lo * (0.75 + 0.25 * Math.sin(t * 3)));
        b.lead.position.y = BAR_Y + b.h + 0.48;
        setOpacity(b.lead, bOn * lo);
      }
    });
    const pOn = ease(seg(t, ...E.patient));
    pills.forEach((s) => setOpacity(s, bOn * pOn));
  };

  // ================= caption =================
  const qRow = view.rows.find((r) => r.key === "quhama");
  const caption = (t) => {
    const muted = (x) => `<span style="color:#94a3b8">${x}</span>`;
    let html = `<div style="font-size:14px"><b>Elite Hybrid Architectures</b> · Streamlit Tab 8's top 3 hybrids vs classical baselines ${muted(`· ${elite.label}`)}`;
    if (t < E.cardsOut[0]) {
      const items = [["quhama", GOLD, E.cardQ], ["sandwich", PALETTE.quantum, E.cardS], ["ckta", PALETTE.quantum2, E.cardC]];
      html += `<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:4px;font-size:12px;line-height:1.35">${items.map(([k, c, w]) =>
        `<div style="opacity:${ease(seg(t, ...w)).toFixed(2)}">${span(c, ELITE_NAMES[k])}: ${A[k].description || ""}</div>`).join("")}</div>`;
      if (t >= E.sweep[0]) {
        html += `<div style="font-size:12px;margin-top:3px;color:#94a3b8">Q-CKTA tries bandwidths s, keeps the one whose kernel best matches the diagnoses (centered kernel-target alignment), then mixes it with an RBF kernel by alignment. The tile is a schematic; the bars are the real alignments.</div>`;
      }
    } else {
      if (t >= E.patient[0]) {
        if (known) {
          const a = pat.actualSick ? span(PALETTE.sick, "Disease") : span(PALETTE.healthy, "Healthy");
          html += `<div style="margin-top:3px">This patient (#${patient}, actual ${a}): ${pat.rows.map((pr) =>
            `${pr.name} ${span(pr.disease ? PALETTE.sick : PALETTE.healthy, `${pr.disease ? "DISEASE" : "HEALTHY"} ${pct1(pr.prob)}%`)} ${pr.correct ? span(PALETTE.healthy, "✓") : span(PALETTE.sick, "✗")}${pr.borderline ? ` <span style="color:${AMBER}">(borderline: verdict and prob disagree)</span>` : ""}`).join(" · ")}</div>`;
        } else {
          html += `<div style="margin-top:3px">This patient: custom (slider) patient. <b>Elite models need a precomputed patient</b>, so no elite verdicts are shown.</div>`;
        }
      }
      const th = (x) => `<th style="text-align:right;font-weight:600;color:#94a3b8;padding:0 6px">${x}</th>`;
      const td = (x, warn) => `<td style="text-align:right;padding:0 6px;${warn ? `color:${AMBER}` : ""}">${x}</td>`;
      html += `<table style="margin-top:5px;font-size:12px;line-height:1.3;border-collapse:collapse;opacity:${ease(seg(t, ...E.bars)).toFixed(2)}"><tr><th style="text-align:left;color:#94a3b8;padding:0 6px 0 0">Model</th>${th("F2 🎯")}${th("Accuracy")}${th("Precision")}${th("Recall")}${th("ROC-AUC")}${th("Latency")}</tr>${view.rows.map((r) =>
        `<tr style="${r.leader ? `color:${hex(GOLD)};font-weight:700` : ""}"><td style="padding:0 6px 0 0">${r.leader ? "★ " : ""}${r.model} ${muted(r.elite ? "· elite" : "· classical")}</td>${td(r.f2.toFixed(3), r.elite && r.f2Low)}${td(`${pct1(r.accuracy)}%`)}${td(`${pct1(r.precision)}%`)}${td(`${pct1(r.recall)}%`)}${td(typeof r.roc_auc === "number" ? r.roc_auc.toFixed(3) : "—", r.elite && r.aucLow)}${td(fmtTime(r.time_s))}</tr>`).join("")}</table>`;
      if (t >= E.leader[0]) {
        const lines = honestyLines(view);
        lines.forEach((l) => { html += `<div style="font-size:12px;margin-top:2px;color:${AMBER};font-weight:600">⚠ ${l}</div>`; });
        if (quhamaMetrics && qRow && typeof quhamaMetrics["F2-Score"] === "number" && Math.abs(quhamaMetrics["F2-Score"] - qRow.f2) > 1e-6) {
          html += `<div style="font-size:12px;margin-top:2px;color:#94a3b8">QuHaMA's leaderboard row is a separate training run, so it differs from the network above (F2 ${quhamaMetrics["F2-Score"].toFixed(3)} there). Small test set (${nTest} patients): treat these numbers as indicative.</div>`;
        } else {
          html += `<div style="font-size:12px;margin-top:2px;color:#94a3b8">Small test set (${nTest} patients): treat these numbers as indicative.</div>`;
        }
      }
    }
    return html + `</div>`;
  };

  return { root, animate, caption };
}
