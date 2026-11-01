// Step 10 · Q-CKTA + Elite leaderboard (mirrors Streamlit Tab 8: the Q-CKTA rationale and the
// "Empirical Leaderboard"; approaches/approach_3.py).
// Part 1: kernel-target alignment. The quantum kernel K(s) morphs through CKTA's bandwidth sweep while each
//         s's real alignment bar grows next to the ideal label pattern; it settles on s*, then the
//         alignment-weighted fusion with the RBF kernel appears.
// Part 2: this patient's CKTA probability and verdict vs the true label; CKTA's test metrics.
// Part 3: the Elite leaderboard: 3D bars on F2 for the tab's five rows (sorted, leader highlighted).
// Numbers come from <slug>.elite.json; QuHaMA's row from <slug>.quhama.json (the run scene 8 shows); the
// kernel matrices from <slug>.bandwidth.json when its alignments equal CKTA's (else a labelled schematic).
import * as THREE from "three";
import { LEGEND_CKTA } from "./legends.js";
import { PALETTE, ease, seg, textSprite, glowSprite, setOpacity, disposeTree, addLights } from "./common.js";
import { GOLD, AMBER, hex, span, muted, label, line, weightRing, gauge, barChart } from "./eliteViz.js";
import { loadQuhama, loadBandwidth } from "../data.js";
import { loadElite, eliteCheck, elitePatient, honestyLines, fmtTime } from "../elite.js";
import { pct1 } from "../quhama.js";
import { fmtS } from "../bandwidth.js";
import {
  patientGuard, approachOk, metricRows, metricHonesty, withQuhamaRun, rankLeaderboard, kernelSource,
  schematicKernel, idealPattern, cktaTimeline, sweepState,
} from "../eliteScenes.js";

const ERR_LEN = 4;
const LAND_W = 2.7, HMAX = 0.95, TILT = 0.62;
const IDEAL_W = 1.35;
const SCHEMATIC_N = 10;

// kernel colour ramp (as the Kernel Bandwidth Lab): low K = deep indigo, high K = bright pink-white
const RAMP = [
  [0.0, new THREE.Color(0x1e1b4b)],
  [0.35, new THREE.Color(0x4c1d95)],
  [0.65, new THREE.Color(PALETTE.quantum)],
  [1.0, new THREE.Color(0xfdf4ff)],
];
function rampColour(k, out) {
  k = Math.min(1, Math.max(0, k));
  for (let i = 1; i < RAMP.length; i++) {
    if (k <= RAMP[i][0]) {
      const [a, ca] = RAMP[i - 1], [b, cb] = RAMP[i];
      return out.copy(ca).lerp(cb, (k - a) / (b - a));
    }
  }
  return out.copy(RAMP[RAMP.length - 1][1]);
}

/** N×N instanced columns on a tilted floor (rows/cols sorted healthy first) + class strips on two edges. */
function kernelTile(parent, N, W, labels) {
  const g = new THREE.Group();
  g.rotation.x = TILT;
  parent.add(g);
  const cell = W / N;
  const geo = new THREE.BoxGeometry(cell * 0.86, 1, cell * 0.86);
  geo.translate(0, 0.5, 0);
  const mat = new THREE.MeshPhongMaterial({ color: 0xffffff, shininess: 40, transparent: true, opacity: 1 });
  const cols = new THREE.InstancedMesh(geo, mat, N * N);
  cols.frustumCulled = false;
  const tmp = new THREE.Color(0, 0, 0);
  for (let i = 0; i < N * N; i++) cols.setColorAt(i, tmp);
  g.add(cols);
  const cx = (j) => -W / 2 + (j + 0.5) * cell;
  const nH = labels.filter((l) => l === 0).length;
  const strip = (i0, i1, colour, alongX) => {
    const len = (i1 - i0) * cell, c = -W / 2 + ((i0 + i1) / 2) * cell;
    const m = new THREE.Mesh(new THREE.BoxGeometry(alongX ? len - 0.02 : 0.06, 0.02, alongX ? 0.06 : len - 0.02),
      new THREE.MeshBasicMaterial({ color: colour, transparent: true, opacity: 0.9 }));
    if (alongX) m.position.set(c, 0.01, W / 2 + 0.08); else m.position.set(-W / 2 - 0.08, 0.01, c);
    g.add(m);
  };
  for (const alongX of [true, false]) {
    if (nH > 0) strip(0, nH, PALETTE.healthy, alongX);
    if (nH < N) strip(nH, N, PALETTE.sick, alongX);
  }
  const M = new THREE.Matrix4();
  return {
    group: g, nH,
    set(Kf, Kt, mix, hScale = 1) {
      for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
          const k = Kf[i][j] + (Kt[i][j] - Kf[i][j]) * mix;
          const n = i * N + j;
          M.makeScale(1, Math.max(0.004, k * HMAX * hScale), 1).setPosition(cx(j), 0, cx(i));
          cols.setMatrixAt(n, M);
          cols.setColorAt(n, rampColour(k, tmp));
        }
      }
      cols.instanceMatrix.needsUpdate = true;
      cols.instanceColor.needsUpdate = true;
    },
  };
}

export default function createScene(ctx) {
  const { data } = ctx;
  const scene = new THREE.Scene();
  addLights(scene);
  const root1 = new THREE.Group(); // part 1: alignment
  root1.position.set(1.4, 0.5, 0);
  const root2 = new THREE.Group(); // part 2: this patient
  root2.position.set(1.4, 0.5, 0);
  const root3 = new THREE.Group(); // part 3: leaderboard
  root3.position.set(1.7, -0.6, 0);
  scene.add(root1, root2, root3);

  const guard = patientGuard(data, ctx.patient, "CKTA");
  let status = "loading", errMsg = "", e = null, q = null, bw = null, known = false, note = guard.ok ? "" : guard.msg;
  let tl = cktaTimeline(5);
  let t0 = null, lastT = 0, disposed = false, built = null;

  Promise.allSettled([loadElite(data.slug), loadQuhama(data.slug), loadBandwidth(data.slug)]).then(([re, rq, rb]) => {
    if (disposed) return;
    if (re.status === "rejected") { status = "error"; errMsg = re.reason.message; }
    else if (!approachOk(re.value, "ckta") || !Array.isArray(re.value.leaderboard) || !re.value.leaderboard.length) {
      status = "error"; errMsg = "CKTA / leaderboard data is missing or malformed.";
    } else {
      e = re.value;
      q = rq.status === "fulfilled" ? rq.value : null;
      bw = rb.status === "fulfilled" ? rb.value : null;
      if (guard.ok) {
        const c = eliteCheck(e, data);
        if (c.ok) known = true; else note = c.msg;
      }
      const sweep = (e.approaches.ckta.architecture.bandwidth_sweep || []).filter((r) => typeof r.alignment === "number");
      tl = cktaTimeline(sweep.length, e.leaderboard.length);
      status = "ready";
    }
    t0 = lastT;
  });

  function build() {
    const A = e.approaches.ckta;
    const ac = A.architecture;
    const sweep = (ac.bandwidth_sweep || []).filter((r) => typeof r.alignment === "number");
    const n = sweep.length;
    const maxA = Math.max(1e-6, ...sweep.map((r) => r.alignment));
    const best = sweep.findIndex((r) => r.s === ac.best_bandwidth_s);
    const Ac = typeof ac.alignment_classical_rbf === "number" ? ac.alignment_classical_rbf : null;

    // ================= part 1: kernel-target alignment =================
    const ks = kernelSource(bw, sweep);
    const N = ks.real ? ks.N : SCHEMATIC_N;
    const labels = ks.real ? ks.labels : Array.from({ length: N }, (_, i) => (i < N / 2 ? 0 : 1));
    const Ks = ks.real ? ks.kernels : sweep.map((r) => schematicKernel(N, Math.max(0, r.alignment) / maxA));
    const land = new THREE.Group();
    land.position.set(-2.45, -1.0, 0);
    root1.add(land);
    const tile = n ? kernelTile(land, N, LAND_W, labels) : null;
    const landT = label(root1, ks.real
      ? `Quantum kernel K(s) · ${N} of the ${ks.nTrain || "?"} training patients`
      : "Quantum kernel K(s) · schematic", -2.45, 1.75, { height: 0.18, color: hex(PALETTE.quantum) });
    label(root1, ks.real ? "real values · healthy first, then disease" : "illustration: sharpness follows the real alignment", -2.45, 1.5, { height: 0.12, color: "#94a3b8", weight: 500 });
    label(root1, `Healthy (${tile ? tile.nH : 0})`, -2.45 - LAND_W / 4, -2.15, { height: 0.13, color: hex(PALETTE.healthy) });
    label(root1, `Disease (${N - (tile ? tile.nH : 0)})`, -2.45 + LAND_W / 4, -2.15, { height: 0.13, color: hex(PALETTE.sick) });
    // the ideal pattern the alignment compares with
    const idealG = new THREE.Group();
    idealG.position.set(0.3, -1.0, 0);
    root1.add(idealG);
    const ideal = kernelTile(idealG, N, IDEAL_W, labels);
    const Y = idealPattern(labels);
    ideal.set(Y, Y, 1, 0.55);
    label(root1, "ideal pattern", 0.3, 0.55, { height: 0.16, color: "#e5e7eb" });
    label(root1, "same diagnosis = 1, different = 0", 0.3, 0.33, { height: 0.11, color: "#94a3b8", weight: 500 });
    // current bandwidth readout above the landscape
    const sLab = sweep.map((r, i) => {
      const s = textSprite(`s = ${fmtS(r.s)} · A_q = ${r.alignment.toFixed(3)}`, { height: 0.2, color: i === best ? hex(GOLD) : "#ffffff", weight: 700 });
      s.position.set(-2.45, 1.08, 0.05);
      s.visible = false;
      root1.add(s);
      return s;
    });
    // alignment bars per s (+ the RBF kernel's alignment as a reference line)
    const BX0 = 1.75, BDX = 0.42, BBASE = -1.45, BMAX = 2.0;
    const top = Math.max(maxA, Ac || 0);
    const aBars = sweep.map((r, i) => {
      const geo = new THREE.BoxGeometry(0.26, 1, 0.2);
      geo.translate(0, 0.5, 0);
      const colour = i === best ? GOLD : PALETTE.quantum;
      const m = new THREE.Mesh(geo, new THREE.MeshPhongMaterial({ color: colour, emissive: colour, emissiveIntensity: 0.35, transparent: true }));
      m.position.set(BX0 + i * BDX, BBASE, 0);
      m.scale.y = 0.001;
      root1.add(m);
      label(root1, fmtS(r.s), BX0 + i * BDX, BBASE - 0.15, { height: 0.12, color: i === best ? hex(GOLD) : "#94a3b8", weight: 500 });
      const h = Math.max(0.02, (Math.max(0, r.alignment) / top) * BMAX);
      const v = label(root1, r.alignment.toFixed(3), BX0 + i * BDX, BBASE + h + 0.14, { height: 0.11, color: "#e5e7eb", weight: 500 });
      return { m, h, v };
    });
    const bMid = BX0 + ((n - 1) * BDX) / 2;
    label(root1, "alignment A_q(s)", bMid, BBASE + BMAX + 0.55, { height: 0.16, color: "#e5e7eb" });
    label(root1, "bandwidth s", bMid, BBASE - 0.36, { height: 0.11, color: "#94a3b8", weight: 500 });
    let rbfLine = null, rbfLab = null;
    if (Ac !== null) {
      const y = BBASE + (Math.max(0, Ac) / top) * BMAX;
      rbfLine = line(root1, new THREE.Vector3(BX0 - 0.25, y, 0.12), new THREE.Vector3(BX0 + (n - 1) * BDX + 0.25, y, 0.12), PALETTE.classical, 0.9);
      rbfLab = label(root1, `RBF A_c ${Ac.toFixed(3)}`, BX0 + (n - 1) * BDX + 0.75, y, { height: 0.11, color: hex(PALETTE.classical) });
    }
    const cursor = glowSprite(PALETTE.quantum2, 0.5, 0.9);
    root1.add(cursor);
    // fusion with the RBF kernel (weights from the two alignments)
    const fuse = new THREE.Group();
    fuse.position.set(bMid, 2.2, 0);
    root1.add(fuse);
    fuse.add(weightRing(ac.weight_classical ?? 0.5, ac.weight_quantum ?? 0.5, 0.2, 0.3));
    label(fuse, `s* = ${fmtS(ac.best_bandwidth_s)}`, 0, 0.55, { height: 0.18, color: hex(GOLD) });
    label(fuse, `fused kernel: RBF ${pct1(ac.weight_classical ?? 0)}% + quantum ${pct1(ac.weight_quantum ?? 0)}%`, 0, -0.48, { height: 0.13, color: "#ffffff" });
    label(fuse, "w_c = A_c / (A_c + A_q(s*)) → precomputed-kernel SVM", 0, -0.7, { height: 0.11, color: "#94a3b8", weight: 500 });

    // ================= part 2: this patient =================
    const pat = known ? elitePatient(e, data, ctx.patient) : null;
    const row = pat ? pat.rows.find((r) => r.key === "ckta") : null;
    const vC = row ? (row.disease ? PALETTE.sick : PALETTE.healthy) : PALETTE.quantum;
    const G = new THREE.Vector3(-1.4, 0, 0), GH = 1.9;
    const gP = gauge(GH, 0.22, vC);
    gP.group.position.copy(G);
    root2.add(gP.group);
    label(root2, "Q-CKTA · this patient", G.x, GH / 2 + 0.75, { height: 0.19, color: hex(PALETTE.quantum2) });
    const pLab = label(root2, row ? `P(disease) ${pct1(row.prob)}%` : "P(disease)", G.x, GH / 2 + 0.35, { height: 0.19, color: "#ffffff" });
    let vLab = null;
    if (row) {
      vLab = label(root2, `${row.disease ? "DISEASE" : "HEALTHY"} ${row.correct ? "✓" : "✗"}${row.borderline ? " ⚠" : ""}\nactual: ${pat.actualSick ? "Disease" : "Healthy"}`, G.x, -GH / 2 - 0.38, { height: 0.18, color: hex(vC) });
    } else {
      vLab = label(root2, note || "CKTA needs a precomputed test patient", G.x + 1.6, -GH / 2 - 0.38, { height: 0.16, color: AMBER, bg: "rgba(14,20,38,0.88)" });
    }
    const info = [
      `fused kernel at s* = ${fmtS(ac.best_bandwidth_s)}: RBF ${pct1(ac.weight_classical ?? 0)}% · quantum ${pct1(ac.weight_quantum ?? 0)}%`,
      `K(patient, training patients) → SVM → verdict`,
    ];
    info.forEach((t, i) => label(root2, t, 1.6, 0.35 - i * 0.3, { height: i ? 0.13 : 0.15, color: i ? "#94a3b8" : "#e5e7eb", weight: i ? 500 : 600 }));
    const pFill = [tl.p2 + 0.6, tl.p2 + 1.8], pVerdict = [tl.p2 + 1.9, tl.p2 + 2.4];

    // ================= part 3: Elite leaderboard =================
    const lb = withQuhamaRun(e.leaderboard, q);
    const view = rankLeaderboard(lb.rows);
    const nTest = A.test.length;
    const board = barChart(root3, view.rows.map((r) => ({
      name: r.model, value: r.f2, text: r.f2.toFixed(3),
      color: r.elite ? PALETTE.quantum : PALETTE.classical, edge: r.elite ? GOLD : undefined,
      nameColor: r.elite ? hex(GOLD) : hex(PALETTE.classical), highlight: r.leader,
      sub: `#${r.rank} · ${r.elite ? "elite hybrid" : "classical"}`,
      warn: [r.f2Low ? "⚠ low F2" : "", r.aucLow ? "⚠ AUC ≈ coin flip" : ""].filter(Boolean).join(" · "),
      lead: r.leader ? (view.leaders.length > 1 ? "★ tied leader" : "★ clinical leader") : "",
    })), { x0: -0.2, gap: 1.75, h: 2.7, w: 0.85, t0: tl.bars[0], title: `Elite leaderboard · F2 🎯 (recall-weighted) on the ${nTest} test patients` });

    return {
      view, lb, ks,
      animate(t) {
        // ---- part 1 ----
        const on1 = 1 - ease(seg(t, tl.p2, tl.p2 + 0.7));
        setOpacity(root1, on1 * ease(seg(t, 0, 0.8)));
        if (on1 > 0.005 && tile) {
          const sw = sweepState(t, n, best, tl);
          tile.set(Ks[sw.from], Ks[sw.to], sw.mix, ease(seg(t, 0.1, 1.0)));
          aBars.forEach((b, i) => {
            b.m.scale.y = Math.max(0.001, b.h * ease(sw.grown[i]));
            b.m.material.emissiveIntensity = i === sw.cur ? 0.85 : 0.3;
            setOpacity(b.v, on1 * ease(seg(sw.grown[i], 0.6, 1)));
          });
          sLab.forEach((s, i) => { s.visible = i === sw.cur; });
          if (sw.cur >= 0) cursor.position.set(BX0 + sw.cur * BDX, BBASE + aBars[sw.cur].h * ease(sw.grown[sw.cur]) + 0.1, 0.1);
          cursor.visible = sw.cur >= 0;
          if (rbfLine) {
            const r = ease(seg(t, tl.sweep0, tl.sweep0 + 0.6));
            setOpacity(rbfLine, on1 * r);
            setOpacity(rbfLab, on1 * r);
          }
          setOpacity(fuse, on1 * ease(seg(t, ...tl.fuse)));
        }
        setOpacity(landT, on1 * ease(seg(t, 0, 0.8)));
        // ---- part 2 ----
        const on2 = ease(seg(t, tl.p2 + 0.3, tl.p2 + 0.9)) * (1 - ease(seg(t, tl.p3, tl.p3 + 0.5)));
        setOpacity(root2, on2);
        if (on2 > 0.005) {
          gP.set(row ? row.prob * ease(seg(t, ...pFill)) : 0);
          setOpacity(pLab, on2 * ease(seg(t, ...pFill)));
          setOpacity(vLab, on2 * ease(seg(t, ...pVerdict)));
        }
        // ---- part 3 ----
        board.animate(t, ease(seg(t, tl.p3 + 0.2, tl.p3 + 0.6)));
      },
    };
  }

  function caption(t) {
    const A = e.approaches.ckta;
    const ac = A.architecture;
    const custom = data.test[ctx.patient] && data.test[ctx.patient].label == null;
    const who = custom ? "custom patient" : `test patient #${ctx.patient}`;
    const ks = built.ks;
    let html = `<div style="font-size:14px"><b>Q-CKTA &amp; Elite leaderboard</b> · ${who} ${muted(`· ${e.label || data.slug}`)}`;
    if (t < tl.p2) {
      let l = `① ${A.description || ""}`;
      if (t >= tl.sweep0) {
        l += ` ${muted(`Kernel-target alignment scores how well K's similarities match the diagnoses (same diagnosis = similar): the closer K's blocks look to the ideal pattern, the higher the score. ${ks.real ? `The landscape shows the real kernel of ${ks.N} training patients at each s (same alignments as CKTA).` : "The landscape is a schematic (no real matrix exported); the bars are the real alignments."}`)}`;
      }
      if (t >= tl.fuse[0]) {
        const Aq = ac.alignment_quantum_best, Ac = ac.alignment_classical_rbf;
        l += ` <div style="margin-top:2px">s* = <b style="color:${hex(GOLD)}">${fmtS(ac.best_bandwidth_s)}</b> has the best alignment A_q = ${typeof Aq === "number" ? Aq.toFixed(3) : "—"} vs the RBF kernel's A_c = ${typeof Ac === "number" ? Ac.toFixed(3) : "—"}, so the fused kernel is ${span(PALETTE.classical, `RBF ${pct1(ac.weight_classical ?? 0)}%`)} + ${span(PALETTE.quantum, `quantum ${pct1(ac.weight_quantum ?? 0)}%`)} (RBF γ = ${typeof ac.rbf_gamma === "number" ? ac.rbf_gamma.toFixed(3) : "—"}), fed to an SVM.</div>`;
      }
      html += `<div style="margin-top:3px">${l}</div>`;
    } else if (t < tl.p3) {
      if (known) {
        const pat = elitePatient(e, data, ctx.patient);
        const row = pat.rows.find((r) => r.key === "ckta");
        let l = `② CKTA's fused-kernel SVM on this patient: P(disease) <b>${pct1(row.prob)}%</b>`;
        if (t >= tl.p2 + 1.9) {
          const v = row.disease ? span(PALETTE.sick, "DISEASE") : span(PALETTE.healthy, "HEALTHY");
          const a = pat.actualSick ? span(PALETTE.sick, "Disease") : span(PALETTE.healthy, "Healthy");
          l += ` → ${v} · actual: ${a} ${row.correct ? span(PALETTE.healthy, "✓ correct") : span(PALETTE.sick, "✗ misclassified")}`;
          if (row.borderline) l += ` <span style="color:${AMBER}">(borderline: the SVM's verdict and its probability disagree)</span>`;
        }
        html += `<div style="margin-top:3px">${l}</div>`;
      } else {
        html += `<div style="margin-top:3px"><span style="color:${AMBER};font-weight:600">${note || "CKTA needs a precomputed test patient"}.</span> ${custom ? "A custom (slider) patient has no exported CKTA prediction." : ""}</div>`;
      }
      if (t >= tl.p2 + 1.0) {
        const m = A.metrics;
        const op = ease(seg(t, tl.p2 + 1.0, tl.p2 + 1.6)).toFixed(2);
        const kpi = (r) => `<div style="background:rgba(255,255,255,.04);border:1px solid #2a3350;border-radius:6px;padding:1px 8px;text-align:center;line-height:1.25"><div style="font-size:11px;color:#94a3b8">${r.label}</div><div style="font-weight:700;${r.warn ? `color:${AMBER}` : ""}">${r.text}</div></div>`;
        html += `<div style="opacity:${op};margin-top:5px"><div style="font-size:12px;color:#94a3b8">CKTA on all ${A.test.length} test patients of this disease${data.custom_run ? " (precomputed at the default settings)" : ""}</div>`;
        html += `<div style="display:grid;grid-template-columns:repeat(6,auto);gap:6px;margin-top:3px">${metricRows(m).map(kpi).join("")}</div>`;
        metricHonesty("CKTA", m).forEach((l) => { html += `<div style="font-size:12px;margin-top:2px;color:${AMBER};font-weight:600">⚠ ${l}</div>`; });
        html += `<div style="font-size:12px;margin-top:2px;color:#94a3b8">Small test set (${A.test.length} patients): treat these numbers as indicative.</div></div>`;
      }
    } else {
      const view = built.view;
      const th = (x) => `<th style="text-align:right;font-weight:600;color:#94a3b8;padding:0 6px">${x}</th>`;
      const td = (x, warn) => `<td style="text-align:right;padding:0 6px;${warn ? `color:${AMBER}` : ""}">${x}</td>`;
      html += `<table style="margin-top:5px;font-size:12px;line-height:1.3;border-collapse:collapse;opacity:${ease(seg(t, ...tl.bars)).toFixed(2)}"><tr><th style="text-align:left;color:#94a3b8;padding:0 6px 0 0">#</th><th style="text-align:left;color:#94a3b8;padding:0 6px 0 0">Model</th>${th("F2 🎯")}${th("Accuracy")}${th("Precision")}${th("Recall")}${th("ROC-AUC")}${th("Latency")}</tr>${view.rows.map((r) =>
        `<tr style="${r.leader ? `color:${hex(GOLD)};font-weight:700` : ""}"><td style="padding:0 6px 0 0">${r.leader ? "★" : r.rank}</td><td style="padding:0 6px 0 0">${r.model} ${muted(r.elite ? "· elite" : "· classical")}</td>${td(r.f2.toFixed(3), r.elite && r.f2Low)}${td(typeof r.accuracy === "number" ? `${pct1(r.accuracy)}%` : "—")}${td(typeof r.precision === "number" ? `${pct1(r.precision)}%` : "—")}${td(typeof r.recall === "number" ? `${pct1(r.recall)}%` : "—")}${td(typeof r.roc_auc === "number" ? r.roc_auc.toFixed(3) : "—", r.elite && r.aucLow)}${td(fmtTime(r.time_s))}</tr>`).join("")}</table>`;
      if (t >= tl.leader[0]) {
        honestyLines(view).forEach((l) => { html += `<div style="font-size:12px;margin-top:2px;color:${AMBER};font-weight:600">⚠ ${l}</div>`; });
        const notes = [];
        if (built.lb.replaced) notes.push("QuHaMA's row uses the run scene 8 shows (quhama.json).");
        else if (!q) notes.push("QuHaMA's row is from elite.json (scene 8's QuHaMA data could not be loaded).");
        notes.push(`Latency = each model's measured run time (Tab 8's Latency column). Small test set (${A.test.length} patients): treat these numbers as indicative.`);
        html += `<div style="font-size:12px;margin-top:2px;color:#94a3b8">${notes.join(" ")}</div>`;
      }
    }
    return html + `</div>`;
  }

  return {
    title: "Q-CKTA kernel alignment & Elite leaderboard",
    legend: LEGEND_CKTA,
    scene,
    cameraPos: [0, 1.4, 11.6],
    // The timeline starts when the data has loaded: stops/duration are shifted by that moment.
    get duration() {
      if (t0 === null) return Infinity;
      return t0 + (status === "ready" ? tl.duration : ERR_LEN);
    },
    get stops() {
      return t0 === null || status !== "ready" ? [] : tl.stops.map((s) => s + t0);
    },
    update(t) {
      if (t0 === null) {
        lastT = t;
        return `<div style="font-size:14px"><b>Q-CKTA &amp; Elite leaderboard</b> · loading the models' data…</div>`;
      }
      if (status === "error") {
        return `<div style="font-size:14px"><b>Q-CKTA &amp; Elite leaderboard</b> · <span style="color:#fb7185">${errMsg}${/[.!?]$/.test(errMsg) ? "" : "."}</span> Other steps still work.</div>`;
      }
      const rt = t - t0;
      if (!built) built = build();
      built.animate(rt);
      return caption(rt);
    },
    dispose() {
      disposed = true;
      disposeTree(scene);
    },
  };
}
