// Step 9 · Sandwich HQNN (mirrors Streamlit Tab 8's "Sandwich HQNN" rationale; approaches/approach_2.py).
// Part 1: the architecture, classical Pre-Net (encoder) -> quantum layer (one Bloch sphere per qubit) ->
//         classical Post-Net (head); particles carry the chosen patient's measurements through it.
// Part 2: this patient's real latent z (Pre-Net output) and ⟨Z⟩ expectations (quantum layer), then the
//         Post-Net's P(disease) and the verdict vs the true label.
// Part 3: Sandwich's test metrics for this disease as 3D bars (F2 first), with an honest caption.
// Every number comes from <slug>.elite.json (QuHaMA's leaderboard row, used only for Sandwich's rank,
// from <slug>.quhama.json like scene 8). Data loads lazily; the timeline starts once it has loaded.
import * as THREE from "three";
import { LEGEND_SANDWICH } from "./legends.js";
import { PALETTE, ease, seg, blochSphere, glowSprite, textSprite, setOpacity, disposeTree, addLights } from "./common.js";
import {
  GOLD, LATENT, AMBER, SUB, hex, n2, sgn, span, muted, label, slab, layerBox, gauge, particleFlows, barChart,
} from "./eliteViz.js";
import { loadQuhama } from "../data.js";
import { loadElite, eliteCheck, elitePatient } from "../elite.js";
import { pickInputs, pct1 } from "../quhama.js";
import {
  patientGuard, approachOk, metricRows, metricHonesty, withQuhamaRun, rankLeaderboard, sandwichTimeline,
} from "../eliteScenes.js";

// Timeline (seconds after the data has loaded). Part 3 starts at tl.p3 (earlier for a custom patient).
const T = {
  inputs: [0.2, 0.9], pre: [0.6, 1.4], lat: [1.1, 1.7], q: [1.5, 2.3], post: [2.1, 2.8], out: [2.5, 3.1],
  // part 1: one pass of particles through the whole sandwich
  inPre: [3.0, 4.0], preLat: [3.6, 4.5], latQ: [4.1, 5.0], latPost: [4.3, 5.3], qPost: [4.8, 5.6], postOut: [5.2, 5.9],
  // part 2: this patient's values
  zVals: [6.6, 7.3], latQ2: [7.0, 7.9], bloch: [7.7, 8.6], toPost2: [8.7, 9.7], postOut2: [9.4, 10.2],
  fill: [9.9, 10.8], verdict: [10.8, 11.2],
  note: [6.6, 7.2], // custom patient: message instead of part 2
};
const ERR_LEN = 4;

// Layout (root-local)
const IN_X = -4.3, PRE = new THREE.Vector3(-2.85, 0, 0), PRE_W = 0.75, PRE_H = 2.5;
const LAT_X = -1.5, Q = new THREE.Vector3(0.15, 0, 0), Q_W = 1.0, Q_H = 2.75, QB_R = 0.3;
const POST = new THREE.Vector3(1.95, 0, 0), POST_W = 0.7, POST_H = 1.9;
const OUT = new THREE.Vector3(3.45, 0, 0), OUT_H = 1.7;
const pretty = (name) => name.replace(/_/g, " ");
const short = (name, n = 15) => (name.length > n ? name.slice(0, n - 1) + "…" : name);

export default function createScene(ctx) {
  const { data } = ctx;
  const scene = new THREE.Scene();
  addLights(scene);
  const root = new THREE.Group(); // architecture (parts 1-2)
  root.position.set(2.2, 0.9, 0);
  scene.add(root);
  const root3 = new THREE.Group(); // metrics (part 3)
  root3.position.set(1.6, -0.5, 0);
  scene.add(root3);

  const guard = patientGuard(data, ctx.patient, "Sandwich");
  let status = "loading", errMsg = "", e = null, q = null, known = false, note = guard.ok ? "" : guard.msg;
  let tl = sandwichTimeline(guard.ok);
  let t0 = null, lastT = 0, disposed = false, built = null;

  Promise.allSettled([loadElite(data.slug), loadQuhama(data.slug)]).then(([re, rq]) => {
    if (disposed) return;
    if (re.status === "rejected") { status = "error"; errMsg = re.reason.message; }
    else if (!approachOk(re.value, "sandwich")) { status = "error"; errMsg = "Sandwich data is missing or malformed."; }
    else {
      e = re.value;
      q = rq.status === "fulfilled" ? rq.value : null;
      if (guard.ok) {
        const c = eliteCheck(e, data);
        if (c.ok) known = true; else note = c.msg;
      }
      tl = sandwichTimeline(known);
      status = "ready";
    }
    t0 = lastT;
  });

  function build() {
    const A = e.approaches.sandwich;
    const arch = A.architecture, ql = arch.quantum_layer || {};
    const p = data.test[ctx.patient] || {};
    const pat = known ? elitePatient(e, data, ctx.patient) : null;
    const row = pat ? pat.rows.find((r) => r.key === "sandwich") : null;
    const st = pat ? pat.sandwich : null;
    const k = arch.n_qubits || 3;
    const d = arch.in_features || (arch.pre_net || [])[0] || (p.standardized || []).length;
    const latY = k === 3 ? [0.8, 0, -0.8] : Array.from({ length: k }, (_, j) => 0.9 - (1.8 * j) / Math.max(1, k - 1));
    const flows = [];
    const stage = []; // {obj, win, dim}
    const labels = []; // {obj, win}

    // ---------- inputs (the patient's standardised measurements) ----------
    const inputs = new THREE.Group();
    root.add(inputs);
    const vals = Array.isArray(p.standardized) ? p.standardized : null;
    const shown = vals ? pickInputs(vals) : Array.from({ length: Math.min(d, 12) }, (_, i) => i);
    const more = d - shown.length;
    const nRows = shown.length + (more > 0 ? 1 : 0);
    const dy = Math.min(0.4, 4.0 / Math.max(1, nRows - 1));
    const nodeGeo = new THREE.SphereGeometry(0.08, 16, 12);
    shown.forEach((fi, r) => {
      const z = vals ? vals[fi] : 0;
      const b = vals ? 0.3 + 0.7 * Math.min(1, Math.abs(z) / 2.5) : 0.6;
      const y = ((nRows - 1) / 2 - r) * dy;
      const node = new THREE.Mesh(nodeGeo, new THREE.MeshBasicMaterial({ color: new THREE.Color(PALETTE.classical).multiplyScalar(b), transparent: true }));
      node.position.set(IN_X, y, 0);
      inputs.add(node);
      const gl = glowSprite(PALETTE.classical, 0.2 + 0.35 * b, b);
      gl.position.copy(node.position);
      inputs.add(gl);
      if (vals && data.feature_names && data.feature_names[fi]) {
        const lab = textSprite(`${short(pretty(data.feature_names[fi]))}  ${sgn(z)}σ`, { height: 0.15, color: "#cbd5e1", align: "right", weight: 500 });
        lab.position.set(IN_X - 0.15 - lab.scale.x / 2, y, 0);
        inputs.add(lab);
      }
      flows.push({ a: node.position.clone(), b: new THREE.Vector3(PRE.x - PRE_W / 2, y * 0.45, 0), win: T.inPre, color: PALETTE.classical, n: 5 });
    });
    if (more > 0) {
      const lab = textSprite(`+${more} more (all ${d} go into the Pre-Net)`, { height: 0.13, color: "#94a3b8", align: "right", weight: 500 });
      lab.position.set(IN_X + 0.1 - lab.scale.x / 2, ((nRows - 1) / 2 - shown.length) * dy, 0);
      inputs.add(lab);
    }
    stage.push({ obj: inputs, win: T.inputs, dim: 0.2 });
    const inT = label(root, `Inputs · ${d} measurements (σ)`, IN_X - 0.5, (nRows - 1) / 2 * dy + 0.42, { height: 0.19, color: hex(PALETTE.classical) });
    labels.push({ obj: inT, win: T.inputs });

    // ---------- Pre-Net: the classical encoder ----------
    const hid = (arch.pre_net || [])[1] || 16;
    const pre = layerBox(PRE_W, PRE_H, PALETTE.classical, Math.min(8, Math.ceil(hid / 2)), 2);
    pre.group.position.copy(PRE);
    root.add(pre.group);
    stage.push({ obj: pre.group, win: T.pre, dim: 0 });
    const preT = label(root, `Pre-Net · classical encoder\n${(arch.pre_net || [d, hid, k]).join(" → ")} · tanh`, PRE.x, PRE_H / 2 + 0.45, { height: 0.17, color: hex(PALETTE.classical) });
    labels.push({ obj: preT, win: T.pre });

    // ---------- latent z (encoder output) ----------
    const latent = latY.map((y, j) => {
      const g = new THREE.Group();
      const m = new THREE.Mesh(new THREE.SphereGeometry(0.14, 24, 16), new THREE.MeshBasicMaterial({ color: LATENT, transparent: true }));
      g.add(m);
      const gl = glowSprite(LATENT, 0.55, 0.6);
      g.add(gl);
      g.position.set(LAT_X, y, 0);
      root.add(g);
      flows.push({ a: new THREE.Vector3(PRE.x + PRE_W / 2, y * 1.05, 0), b: g.position.clone(), win: T.preLat, color: LATENT, n: 6 });
      return { g, m, gl };
    });
    latent.forEach((l) => stage.push({ obj: l.g, win: T.lat, dim: 0 }));
    const latT = label(root, `z · ${k} numbers`, LAT_X, latY[0] + 0.62, { height: 0.16, color: hex(LATENT) });
    labels.push({ obj: latT, win: T.lat });
    if (known && Array.isArray(st.latent)) st.latent.slice(0, k).forEach((z, j) => {
      const l = label(root, `z${SUB[j] || j + 1} ${sgn(z)}`, LAT_X, latY[j] + 0.3, { height: 0.15, color: "#f5f3ff" });
      labels.push({ obj: l, win: T.zVals });
    });

    // ---------- quantum layer ----------
    const qSlab = slab(Q_W, Q_H, 0.9, PALETTE.quantum, 0.1);
    qSlab.position.copy(Q);
    root.add(qSlab);
    stage.push({ obj: qSlab, win: T.q, dim: 0 });
    const blochs = latY.map((y, j) => {
      const s = blochSphere(QB_R);
      s.position.set(Q.x, y, 0);
      root.add(s);
      stage.push({ obj: s, win: T.q, dim: 0 });
      flows.push({ a: latent[j].g.position.clone(), b: s.position.clone().add(new THREE.Vector3(-QB_R, 0, 0)), win: T.latQ, color: PALETTE.quantum, n: 6 });
      return s;
    });
    const qT = label(root, `Quantum layer · ${k} qubits\n${ql.feature_map || "feature map"} + ansatz → ⟨Z⟩`, Q.x, Q_H / 2 + 0.75, { height: 0.17, color: hex(PALETTE.quantum) });
    labels.push({ obj: qT, win: T.q });
    const qSub = label(root, `depth ${ql.circuit_depth ?? "—"} · ${ql.circuit_size ?? "—"} gates · ansatz weights fixed at ${Array.isArray(ql.weights) && ql.weights.length ? ql.weights[0] : "—"}`, Q.x, -Q_H / 2 - 0.22, { height: 0.12, color: "#94a3b8", weight: 500 });
    labels.push({ obj: qSub, win: T.q });
    const expT = known && Array.isArray(st.expectation) ? st.expectation.slice(0, k).map((v) => Math.acos(Math.max(-1, Math.min(1, v)))) : null;
    if (expT) {
      st.expectation.slice(0, k).forEach((v, j) => {
        const l = label(root, `⟨Z${SUB[j] || j + 1}⟩ ${n2(v)}`, Q.x + 0.9, latY[j] + 0.02, { height: 0.14, color: "#e9d5ff" });
        labels.push({ obj: l, win: T.bloch });
      });
    }

    // ---------- Post-Net: the classical head ----------
    const ph = (arch.post_net || [])[1] || 8;
    const post = layerBox(POST_W, POST_H, PALETTE.classical, Math.min(4, Math.ceil(ph / 2)), 2);
    post.group.position.copy(POST);
    root.add(post.group);
    stage.push({ obj: post.group, win: T.post, dim: 0 });
    const postT = label(root, `Post-Net · classical head\n[z, ⟨Z⟩] ${(arch.post_net || [2 * k, ph, 2]).join(" → ")}`, POST.x, POST_H / 2 + 0.45, { height: 0.17, color: hex(PALETTE.classical) });
    labels.push({ obj: postT, win: T.post });
    const postIn = (y) => new THREE.Vector3(POST.x - POST_W / 2, y * 0.5, 0);
    latent.forEach((l, j) => {
      const a = l.g.position.clone(), b = postIn(latY[j]);
      const bend = 1.75 - (a.y + b.y) / 2; // z skips over the quantum layer into the Post-Net
      flows.push({ a, b, win: T.latPost, color: LATENT, n: 4, bendY: bend });
      if (known) flows.push({ a: a.clone(), b: b.clone(), win: T.toPost2, color: LATENT, n: 5, bendY: bend });
    });
    blochs.forEach((s, j) => {
      flows.push({ a: s.position.clone().add(new THREE.Vector3(QB_R, 0, 0)), b: postIn(latY[j]), win: T.qPost, color: PALETTE.quantum, n: 5 });
      if (known) flows.push({ a: s.position.clone().add(new THREE.Vector3(QB_R, 0, 0)), b: postIn(latY[j]), win: T.toPost2, color: PALETTE.quantum, n: 6 });
    });
    if (known) latent.forEach((l, j) => flows.push({ a: l.g.position.clone(), b: blochs[j].position.clone().add(new THREE.Vector3(-QB_R, 0, 0)), win: T.latQ2, color: PALETTE.quantum, n: 5 }));
    flows.push({ a: new THREE.Vector3(POST.x + POST_W / 2, 0, 0), b: OUT.clone().add(new THREE.Vector3(0, -OUT_H / 2, 0)), win: T.postOut, color: PALETTE.classical, n: 8 });
    if (known) flows.push({ a: new THREE.Vector3(POST.x + POST_W / 2, 0, 0), b: OUT.clone().add(new THREE.Vector3(0, -OUT_H / 2, 0)), win: T.postOut2, color: PALETTE.classical, n: 10 });

    // ---------- output gauge + verdict ----------
    const vC = row ? (row.disease ? PALETTE.sick : PALETTE.healthy) : PALETTE.quantum;
    const gOut = gauge(OUT_H, 0.2, vC);
    gOut.group.position.copy(OUT);
    root.add(gOut.group);
    stage.push({ obj: gOut.group, win: T.out, dim: 0 });
    const outT = label(root, row ? `P(disease) ${pct1(row.prob)}%` : "P(disease)", OUT.x, OUT_H / 2 + 0.3, { height: 0.19, color: "#ffffff" });
    labels.push({ obj: outT, win: T.out });
    if (row) {
      const vL = label(root, `${row.disease ? "DISEASE" : "HEALTHY"} ${row.correct ? "✓" : "✗"}${row.borderline ? " ⚠" : ""}\nactual: ${pat.actualSick ? "Disease" : "Healthy"}`, OUT.x, -OUT_H / 2 - 0.36, { height: 0.17, color: hex(vC) });
      labels.push({ obj: vL, win: T.verdict });
    }
    let noteL = null;
    if (!known) {
      noteL = label(root, note || "Sandwich needs a precomputed test patient", Q.x, -Q_H / 2 - 0.62, { height: 0.17, color: "#fbbf24", bg: "rgba(14,20,38,0.88)" });
    }
    const flowAnim = particleFlows(root, flows);

    // ---------- part 3: metric bars ----------
    const rows = metricRows(A.metrics);
    const bars = rows.filter((r) => r.bar).map((r) => ({
      name: r.label, value: r.value, text: r.text,
      color: r.key === "f2" ? GOLD : PALETTE.quantum, edge: r.key === "f2" ? 0xffffff : GOLD,
      nameColor: r.key === "f2" ? hex(GOLD) : "#e5e7eb", highlight: r.key === "f2",
      warn: r.warn ? (r.key === "f2" ? "⚠ low F2" : "⚠ ≈ coin flip") : "",
    }));
    const nTest = A.test.length;
    const chart = barChart(root3, bars, { x0: -0.6, gap: 1.25, h: 2.5, w: 0.7, t0: tl.bars[0], title: `Sandwich HQNN · test metrics on the ${nTest} test patients`, titleColor: hex(PALETTE.quantum) });
    const tRow = rows.find((r) => r.key === "time_s");
    const clock = label(root3, `⏱ ${tRow.text}\ntraining time`, -0.6 + 2.5 * 1.25 + 1.0, 1.25, { height: 0.19, color: "#e5e7eb" });

    return function animate(t) {
      const archOn = 1 - ease(seg(t, tl.p3, tl.p3 + 0.8));
      setOpacity(root, archOn);
      if (archOn > 0.005) {
        for (const s of stage) setOpacity(s.obj, archOn * (s.dim + (1 - s.dim) * ease(seg(t, ...s.win))));
        for (const l of labels) setOpacity(l.obj, archOn * ease(seg(t, ...l.win)));
        pre.units.forEach((u, i) => {
          const a = T.inPre[1] - 0.4 + (i / pre.units.length) * 0.6;
          u.material.opacity = archOn * ease(seg(t, ...T.pre)) * (0.25 + 0.75 * ease(seg(t, a, a + 0.3)));
        });
        post.units.forEach((u, i) => {
          const a = T.qPost[1] - 0.4 + (i / post.units.length) * 0.4;
          u.material.opacity = archOn * ease(seg(t, ...T.post)) * (0.25 + 0.75 * ease(seg(t, a, a + 0.3)));
        });
        // latent nodes: brightness = |z| once the values are shown
        if (known && Array.isArray(st.latent)) {
          const zv = ease(seg(t, ...T.zVals));
          latent.forEach((l, j) => {
            const a = Math.min(1, Math.abs(st.latent[j] ?? 0));
            l.gl.scale.setScalar(0.55 + zv * 0.5 * a);
          });
        }
        const bq = expT ? ease(seg(t, ...T.bloch)) : 0;
        blochs.forEach((s, j) => s.userData.set(expT ? expT[j] * bq : 0, 0));
        gOut.set(row ? row.prob * ease(seg(t, ...T.fill)) : 0);
        if (noteL) setOpacity(noteL, archOn * ease(seg(t, ...T.note)));
        flowAnim(t, archOn);
      }
      const on = ease(seg(t, tl.p3 + 0.2, tl.p3 + 0.7));
      chart.animate(t, on);
      setOpacity(clock, on * ease(seg(t, chart.end, chart.end + 0.6)));
    };
  }

  function caption(t) {
    const A = e.approaches.sandwich;
    const arch = A.architecture, ql = arch.quantum_layer || {};
    const k = arch.n_qubits || 3;
    const d = arch.in_features || (arch.pre_net || [])[0];
    const custom = data.test[ctx.patient] && data.test[ctx.patient].label == null;
    const who = custom ? "custom patient" : `test patient #${ctx.patient}`;
    let html = `<div style="font-size:14px"><b>Sandwich HQNN</b> · classical → quantum → classical · ${who} ${muted(`· ${e.label || data.slug}`)}`;
    let line = "";
    if (t < 6.4) {
      line = `① ${A.description || ""}`;
      if (t >= T.inPre[0]) line += ` ${muted(`The ${span(PALETTE.classical, "Pre-Net")} (${(arch.pre_net || []).join(" → ")}) and ${span(PALETTE.classical, "Post-Net")} (${(arch.post_net || []).join(" → ")}) are trained; the quantum layer's ansatz weights stay fixed at ${Array.isArray(ql.weights) && ql.weights.length ? ql.weights[0] : "—"}. The Post-Net reads both z and ⟨Z⟩. Here the quantum layer runs on a noiseless simulator.`)}`;
    } else if (t < tl.p3) {
      if (!known) {
        line = `<span style="color:${AMBER};font-weight:600">${note || "Sandwich needs a precomputed test patient"}.</span> ${custom ? "A custom (slider) patient has no exported Sandwich values; pick a real test patient to see z and ⟨Z⟩." : ""} The test metrics follow.`;
      } else {
        const pat = elitePatient(e, data, ctx.patient);
        const row = pat.rows.find((r) => r.key === "sandwich"), st = pat.sandwich;
        const zs = (st.latent || []).slice(0, k);
        line = `② ${span(PALETTE.classical, "Pre-Net")} squeezes all ${d} measurements into z = [${zs.map(n2).join(", ")}] (tanh, −1…+1), angles θ = (z+1)·π/2 = [${zs.map((z) => ((z + 1) * Math.PI / 2).toFixed(2)).join(", ")}]`;
        if (t >= T.bloch[0]) line += ` · ③ ${span(PALETTE.quantum, "quantum layer")}: ⟨Z⟩ = [${(st.expectation || []).slice(0, k).map(n2).join(", ")}] (arrow height; −1 = always 1, +1 = always 0)`;
        if (t >= T.fill[0]) line += ` · ④ ${span(PALETTE.classical, "Post-Net")} on [z, ⟨Z⟩] → P(disease) <b>${pct1(row.prob)}%</b>`;
        if (t >= T.verdict[0]) {
          const v = row.disease ? span(PALETTE.sick, "DISEASE") : span(PALETTE.healthy, "HEALTHY");
          const a = pat.actualSick ? span(PALETTE.sick, "Disease") : span(PALETTE.healthy, "Healthy");
          line += ` → ${v} · actual: ${a} ${row.correct ? span(PALETTE.healthy, "✓ correct") : span(PALETTE.sick, "✗ misclassified")}`;
          if (row.borderline) line += ` <span style="color:${AMBER}">(borderline: verdict and prob disagree)</span>`;
        }
      }
    }
    if (line) html += `<div style="margin-top:3px">${line}</div>`;
    if (t >= tl.p3) {
      const m = A.metrics;
      const op = ease(seg(t, tl.p3 + 0.2, tl.p3 + 0.8)).toFixed(2);
      const nTest = A.test.length;
      const kpi = (r) => `<div style="background:rgba(255,255,255,.04);border:1px solid #2a3350;border-radius:6px;padding:1px 8px;text-align:center;line-height:1.25"><div style="font-size:11px;color:#94a3b8">${r.label}</div><div style="font-weight:700;${r.warn ? `color:${AMBER}` : ""}">${r.text}</div></div>`;
      html += `<div style="opacity:${op};margin-top:5px"><div style="font-size:12px;color:#94a3b8">Sandwich on all ${nTest} test patients of this disease${data.custom_run ? " (precomputed at the default settings)" : ""}</div>`;
      html += `<div style="display:grid;grid-template-columns:repeat(6,auto);gap:6px;margin-top:3px">${metricRows(m).map(kpi).join("")}</div>`;
      if (t >= tl.bars[1]) {
        metricHonesty("Sandwich", m).forEach((l) => { html += `<div style="font-size:12px;margin-top:2px;color:${AMBER};font-weight:600">⚠ ${l}</div>`; });
        const v = rankLeaderboard(withQuhamaRun(e.leaderboard, q).rows);
        const me = v.rows.find((r) => r.key === "sandwich");
        const claims = [];
        if (me) {
          const pos = me.leader ? (v.leaders.length > 1 ? "it shares the best F2" : "it has the best F2") : `it ranks #${me.rank} of ${v.rows.length} on F2`;
          claims.push(`Tab 8 calls Sandwich "the undisputed overall winner": on this disease ${pos}.`);
        }
        if (typeof m.time_s === "number") claims.push(m.time_s < 0.2 ? `Tab 8's "&lt; 0.2 s" holds here (trained in ${metricRows(m)[5].text}).` : `Tab 8 says "&lt; 0.2 s"; this run took ${metricRows(m)[5].text}.`);
        claims.push("Tab 8 also says the Post-Net absorbs QPU readout noise; this run used a noiseless simulator, so that is not tested here.");
        html += `<div style="font-size:12px;margin-top:3px;color:#cbd5e1">${claims.join(" ")}</div>`;
        html += `<div style="font-size:12px;margin-top:2px;color:#94a3b8">Small test set (${nTest} patients): treat these numbers as indicative.</div>`;
      }
      html += `</div>`;
    }
    return html + `</div>`;
  }

  return {
    title: "Sandwich HQNN: classical → quantum → classical",
    legend: LEGEND_SANDWICH,
    scene,
    cameraPos: [0, 0.9, 11.6],
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
        return `<div style="font-size:14px"><b>Sandwich HQNN</b> · loading the model's data…</div>`;
      }
      if (status === "error") {
        return `<div style="font-size:14px"><b>Sandwich HQNN</b> · <span style="color:#fb7185">${errMsg}${/[.!?]$/.test(errMsg) ? "" : "."}</span> Other steps still work.</div>`;
      }
      const rt = t - t0;
      if (!built) built = build();
      built(rt);
      return caption(rt);
    },
    dispose() {
      disposed = true;
      disposeTree(scene);
    },
  };
}
