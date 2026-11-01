// Kernel Bandwidth Lab (mirrors qmed/ui/tab_bandwidth.py).
// Part 1: the quantum kernel matrix of 40 training patients (sorted healthy -> disease) as a 3D landscape,
//         morphing through every bandwidth s from largest to smallest.
// Part 2: the 5-fold cross-validated sweep: QSVC CV accuracy bars (+ ROC-AUC dots) per s vs the RBF reference.
// Every number comes from web/public/data/<slug>.bandwidth.json (scripts/export_tour_extras.py).
import * as THREE from "three";
import { LEGEND_BANDWIDTH } from "./legends.js";
import { PALETTE, ease, seg, textSprite, glowSprite, setOpacity, disposeTree, addLights } from "./common.js";
import { loadBandwidth } from "../data.js";
import { sKey, fmtS, regimeShort, morphState, sweepSummary, scoreAxisMin, classBoundary } from "../bandwidth.js";

const GOLD = 0xfbbf24;
const CELL = 0.088; // world size of one kernel cell
const HMAX = 1.25; // column height at K = 1
const LAND_Y = -0.1;

// ---- timeline (seconds, relative to the moment the data is ready) ----
const RISE0 = 0.2, RISE1 = 1.3; // landscape rises at the largest s
const MORPH0 = 1.3, MORPH = 0.75, HOLD = 1.15; // then morphs to each next-smaller s and holds
const STEP = MORPH + HOLD;
// part 2 timings are computed from the number of options (see build())
const BAR_STAGGER = 0.3, BAR_RISE = 0.7;

const REGIME_COLOUR = { sharp: "#fb923c", balanced: "#a7f3d0", blurred: "#94a3b8" };
const hex = (c) => "#" + c.toString(16).padStart(6, "0");
const pct1 = (v) => (v * 100).toFixed(1) + "%";

// kernel colour ramp: low K = deep indigo, high K = bright pink-white ("bright = similar")
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

function regimeOf(bw, s) {
  const r = (bw.regime && bw.regime[sKey(s)]) || {};
  return { key: r.key || "", text: r.text || "", short: regimeShort(r.text) };
}
const regimeHtml = (r) => `<b style="color:${REGIME_COLOUR[r.key] || "#e5e7eb"}">${r.short}</b>`;

export default function createScene(ctx) {
  const slug = ctx.slug || (ctx.data && ctx.data.slug);
  const scene = new THREE.Scene();
  addLights(scene);
  const floor = new THREE.GridHelper(9, 18, 0x334155, 0x1e293b);
  floor.position.y = LAND_Y - 0.01;
  scene.add(floor);

  let bw = null, err = null, built = null, disposed = false, tReady = null;
  loadBandwidth(slug).then(
    (d) => { if (!disposed) { bw = d; try { built = build(d); } catch (e) { err = e.message || String(e); } } },
    (e) => { err = e.message; }
  );

  function build(d) {
    const N = d.order.length;
    const W = N * CELL;
    const b = classBoundary(d.labels);
    const sDesc = [...d.options].sort((x, y) => y - x);
    const sAsc = [...d.options].sort((x, y) => x - y);
    const K = Object.fromEntries(d.options.map((s) => [s, d.kernels[sKey(s)]]));
    const nTrain = (d.settings && d.settings.n_train) || "?";
    const nPatients = (d.settings && d.settings.n_patients) || "?";
    const nFolds = (d.settings && d.settings.n_folds) || 5;

    // ================= Part 1: kernel landscape =================
    const land = new THREE.Group();
    land.position.set(0, LAND_Y, -1.0);
    scene.add(land);
    const geo = new THREE.BoxGeometry(CELL * 0.86, 1, CELL * 0.86);
    geo.translate(0, 0.5, 0);
    const mat = new THREE.MeshPhongMaterial({ color: 0xffffff, shininess: 40, transparent: true, opacity: 1 });
    mat.onBeforeCompile = (sh) => { // self-glow from the instance colour, like scene 5's hybrid column
      sh.fragmentShader = sh.fragmentShader.replace(
        "vec3 totalEmissiveRadiance = emissive;",
        "vec3 totalEmissiveRadiance = emissive + vColor.rgb * 0.35;"
      );
    };
    const cols = new THREE.InstancedMesh(geo, mat, N * N);
    cols.frustumCulled = false;
    const tmpC = new THREE.Color();
    for (let i = 0; i < N * N; i++) cols.setColorAt(i, tmpC.setRGB(0, 0, 0));
    land.add(cols);
    const cx = (j) => -W / 2 + (j + 0.5) * CELL;
    const M = new THREE.Matrix4();

    // diagonal blocks: tinted floor + glowing outline (healthy = green, disease = rose)
    const block = (i0, i1, colour) => {
      const w = (i1 - i0) * CELL, c = -W / 2 + ((i0 + i1) / 2) * CELL;
      const g = new THREE.Group();
      const tint = new THREE.Mesh(
        new THREE.PlaneGeometry(w, w),
        new THREE.MeshBasicMaterial({ color: colour, transparent: true, opacity: 0.16, side: THREE.DoubleSide, depthWrite: false })
      );
      tint.rotation.x = -Math.PI / 2;
      tint.position.set(c, 0.002, c);
      const pts = [[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]].map(([a, z]) => new THREE.Vector3(c + (a * w) / 2, 0.006, c + (z * w) / 2));
      const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),
        new THREE.LineBasicMaterial({ color: colour, transparent: true, opacity: 0.95 }));
      // the same outline floating just above the block's tallest column
      const top = line.clone();
      top.material = line.material.clone();
      top.material.opacity = 0.45;
      top.userData.top = true;
      g.add(tint, line, top);
      land.add(g);
      return { g, top };
    };
    const blocks = [];
    if (b > 0) blocks.push({ ...block(0, b, PALETTE.healthy), i0: 0, i1: b });
    if (b < N) blocks.push({ ...block(b, N, PALETTE.sick), i0: b, i1: N });

    // class strips along the front and left edges (like the Healthy/Disease ticks of the Gram plot)
    const strip = (i0, i1, colour, alongX) => {
      const len = (i1 - i0) * CELL, c = -W / 2 + ((i0 + i1) / 2) * CELL;
      const m = new THREE.Mesh(new THREE.BoxGeometry(alongX ? len - 0.03 : 0.07, 0.03, alongX ? 0.07 : len - 0.03),
        new THREE.MeshBasicMaterial({ color: colour, transparent: true, opacity: 0.9 }));
      if (alongX) m.position.set(c, 0.015, W / 2 + 0.12); else m.position.set(-W / 2 - 0.12, 0.015, c);
      land.add(m);
    };
    const edgeLabel = (text, colour, i0, i1, alongX) => {
      const s = textSprite(text, { height: 0.17, color: hex(colour) });
      const c = -W / 2 + ((i0 + i1) / 2) * CELL;
      if (alongX) s.position.set(c, 0.05, W / 2 + 0.38); else s.position.set(-W / 2 - 0.62, 0.05, c);
      land.add(s);
    };
    for (const alongX of [true, false]) {
      if (b > 0) { strip(0, b, PALETTE.healthy, alongX); edgeLabel(`Healthy (${b})`, PALETTE.healthy, 0, b, alongX); }
      if (b < N) { strip(b, N, PALETTE.sick, alongX); edgeLabel(`Disease (${N - b})`, PALETTE.sick, b, N, alongX); }
    }

    // floating s / regime label per bandwidth (only the current one is shown)
    const sLabels = Object.fromEntries(d.options.map((s) => {
      const r = regimeOf(d, s);
      const g = new THREE.Group();
      const a = textSprite(`s = ${fmtS(s)}`, { height: 0.3, color: "#ffffff", weight: 700 });
      const c = textSprite(r.short, { height: 0.19, color: REGIME_COLOUR[r.key] || "#e5e7eb" });
      a.position.y = 0.2; c.position.y = -0.1;
      g.add(a, c);
      g.position.set(0, HMAX + 0.55, -W / 2 - 0.2);
      g.visible = false;
      land.add(g);
      return [s, g];
    }));

    const kLerp = new Float32Array(N * N);
    function setLandscape(from, to, mix, grow) {
      const A = K[from], B = K[to];
      for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
          const k = A[i][j] + (B[i][j] - A[i][j]) * mix;
          const n = i * N + j;
          kLerp[n] = k;
          M.makeScale(1, Math.max(0.004, k * HMAX * grow), 1).setPosition(cx(j), 0, cx(i));
          cols.setMatrixAt(n, M);
          cols.setColorAt(n, rampColour(k * (0.35 + 0.65 * grow), tmpC));
        }
      }
      cols.instanceMatrix.needsUpdate = true;
      cols.instanceColor.needsUpdate = true;
      for (const bl of blocks) { // top outline at the block's tallest off-diagonal column
        let mx = 0;
        for (let i = bl.i0; i < bl.i1; i++) for (let j = bl.i0; j < bl.i1; j++) if (i !== j && kLerp[i * N + j] > mx) mx = kLerp[i * N + j];
        bl.top.position.y = mx * HMAX * grow;
      }
    }

    // ================= Part 2: cross-validated sweep =================
    const sw = sweepSummary(d);
    const yMin = scoreAxisMin(sw.rows, sw.rbf);
    const CH = 2.3, CY = -0.45; // chart height for [yMin, 1], chart floor y
    const yOf = (v) => CY + ((Math.min(1, Math.max(yMin, v)) - yMin) / (1 - yMin)) * CH;
    const GAPX = 0.74, ZA = 0.3, ZU = -0.4; // bar spacing, accuracy row z, ROC-AUC row z
    const xOf = (i) => (i - (sAsc.length - 1) / 2) * GAPX;
    const chart = new THREE.Group();
    chart.rotation.x = -0.42; // lean back like an easel so the elevated camera sees the bars more frontally
    chart.position.set(0.35, 0.45, -0.2);
    scene.add(chart);
    const XL = xOf(0) - 0.55, XR = xOf(sAsc.length - 1) + 0.55;

    // back wall gridlines + tick labels
    const ticks = [];
    for (let v = Math.ceil(yMin * 10 - 1e-9) / 10; v <= 1.0001; v += 0.1) ticks.push(+v.toFixed(1));
    const gridMat = new THREE.LineBasicMaterial({ color: 0x334155, transparent: true, opacity: 0.8 });
    for (const v of ticks) {
      const l = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(XL, yOf(v), ZU - 0.35), new THREE.Vector3(XR, yOf(v), ZU - 0.35)]), gridMat);
      chart.add(l);
      const s = textSprite(v.toFixed(1), { height: 0.13, color: "#94a3b8" });
      s.position.set(XL - 0.22, yOf(v), ZU - 0.35);
      chart.add(s);
    }
    const axisTitle = textSprite(`CV score (mean of ${nFolds} folds)\naxis starts at ${yMin.toFixed(2)}`, { height: 0.12, color: "#94a3b8" });
    axisTitle.position.set(XR - axisTitle.scale.x / 2, CY + CH + 0.32, ZU - 0.35);
    chart.add(axisTitle);
    const base = new THREE.Mesh(new THREE.PlaneGeometry(XR - XL, ZA - ZU + 0.9),
      new THREE.MeshBasicMaterial({ color: 0x0f172a, transparent: true, opacity: 0.7, side: THREE.DoubleSide, depthWrite: false }));
    base.rotation.x = -Math.PI / 2;
    base.position.set((XL + XR) / 2, CY - 0.005, (ZA + ZU) / 2);
    chart.add(base);

    const barGeo = new THREE.BoxGeometry(0.36, 1, 0.36);
    barGeo.translate(0, 0.5, 0);
    const bars = sAsc.map((s, i) => {
      const row = sw.rows.find((r) => r.bandwidth === s);
      const x = xOf(i);
      const isBest = row === sw.best, isDef = sw.def && row === sw.def;
      const colour = isBest ? GOLD : PALETTE.quantum;
      const m = new THREE.MeshPhongMaterial({ color: colour, emissive: colour, emissiveIntensity: 0.3, transparent: true, opacity: 0.9, shininess: 60 });
      const mesh = new THREE.Mesh(barGeo, m);
      mesh.position.set(x, CY, ZA);
      chart.add(mesh);
      // +-1 std whisker
      const lo = yOf(row.accuracy - row.accuracy_std), hi = yOf(row.accuracy + row.accuracy_std);
      const whisk = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(x, lo, ZA), new THREE.Vector3(x, hi, ZA),
        new THREE.Vector3(x - 0.07, lo, ZA), new THREE.Vector3(x + 0.07, lo, ZA),
        new THREE.Vector3(x - 0.07, hi, ZA), new THREE.Vector3(x + 0.07, hi, ZA),
      ]), new THREE.LineBasicMaterial({ color: 0xe2e8f0, transparent: true, opacity: 0.55 }));
      chart.add(whisk);
      const val = textSprite(pct1(row.accuracy), { height: 0.13, color: "#ffffff", weight: 700, bg: "rgba(7,11,22,0.78)" });
      val.position.set(x, yOf(row.accuracy) + 0.13, ZA + 0.3);
      chart.add(val);
      const r = regimeOf(d, s);
      const name = textSprite(`s = ${fmtS(s)}${isDef ? "\ndefault" : ""}`, { height: 0.13, color: isDef ? "#ffffff" : "#cbd5e1", weight: isDef ? 700 : 600 });
      name.position.set(x, CY - 0.16 - (isDef ? 0.08 : 0), ZA + 0.5);
      chart.add(name);
      const reg = textSprite(r.short, { height: 0.12, color: REGIME_COLOUR[r.key] || "#e5e7eb" });
      reg.position.set(x, CY - 0.4 - (isDef ? 0.16 : 0), ZA + 0.5);
      chart.add(reg);
      // ROC-AUC dot (back row)
      // drawn on top (depthTest off) so low AUC dots behind the bars stay visible
      const dot = new THREE.Mesh(new THREE.SphereGeometry(0.07, 16, 12), new THREE.MeshBasicMaterial({ color: PALETTE.quantum2, transparent: true, depthTest: false }));
      dot.renderOrder = 5;
      dot.position.set(x, yOf(row.roc_auc), ZU);
      chart.add(dot);
      return { s, row, x, mesh, mat: m, whisk, val, name, reg, dot, isBest, isDef };
    });
    const aucLine = new THREE.Line(new THREE.BufferGeometry().setFromPoints(bars.map((q) => q.dot.position.clone())),
      new THREE.LineBasicMaterial({ color: PALETTE.quantum2, transparent: true, opacity: 0.85, depthTest: false }));
    aucLine.renderOrder = 5;
    chart.add(aucLine);

    // RBF reference: translucent plane at the RBF CV accuracy across the accuracy row, dashed line at RBF ROC-AUC
    const rbfPlane = new THREE.Mesh(new THREE.PlaneGeometry(XR - XL, 0.75),
      new THREE.MeshBasicMaterial({ color: PALETTE.classical, transparent: true, opacity: 0.22, side: THREE.DoubleSide, depthWrite: false }));
    rbfPlane.rotation.x = -Math.PI / 2;
    rbfPlane.add(new THREE.LineSegments(new THREE.EdgesGeometry(rbfPlane.geometry),
      new THREE.LineBasicMaterial({ color: PALETTE.classical, transparent: true, opacity: 0.8 })));
    rbfPlane.position.set((XL + XR) / 2, yOf(sw.rbf.accuracy), ZA);
    chart.add(rbfPlane);
    const rbfLabel = textSprite(`RBF SVM\naccuracy ${pct1(sw.rbf.accuracy)}`, { height: 0.13, color: hex(PALETTE.classical), weight: 700 });
    rbfLabel.position.set(XR + 0.08 + rbfLabel.scale.x / 2, yOf(sw.rbf.accuracy), ZA);
    chart.add(rbfLabel);
    const dashed = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(XL, yOf(sw.rbf.roc_auc), ZU), new THREE.Vector3(XR, yOf(sw.rbf.roc_auc), ZU)]),
      new THREE.LineDashedMaterial({ color: PALETTE.classical, dashSize: 0.12, gapSize: 0.08, transparent: true, opacity: 0.9 }));
    dashed.computeLineDistances();
    chart.add(dashed);
    const rbfAucLabel = textSprite(`RBF ROC-AUC ${sw.rbf.roc_auc.toFixed(3)}`, { height: 0.11, color: hex(PALETTE.classical) });
    rbfAucLabel.position.set(XR + 0.08 + rbfAucLabel.scale.x / 2, yOf(sw.rbf.roc_auc), ZU);
    chart.add(rbfAucLabel);

    // best s: gold ring + glow + label; default s: dotted pole + label
    const bestBar = bars.find((q) => q.isBest) || bars[0];
    const ring = new THREE.Mesh(new THREE.RingGeometry(0.3, 0.4, 40),
      new THREE.MeshBasicMaterial({ color: GOLD, transparent: true, opacity: 0.9, side: THREE.DoubleSide, depthWrite: false }));
    ring.rotation.x = -Math.PI / 2;
    ring.position.set(bestBar.x, CY + 0.01, ZA);
    chart.add(ring);
    const bestGlow = glowSprite(GOLD, 1.1, 0.9);
    bestGlow.position.set(bestBar.x, yOf(bestBar.row.accuracy), ZA);
    chart.add(bestGlow);
    const defBar = bars.find((q) => q.isDef);
    const same = defBar === bestBar;
    const bestLabel = textSprite(`best s = ${fmtS(bestBar.s)}${same ? " (default)" : ""}`, { height: 0.16, color: hex(GOLD), weight: 700 });
    bestLabel.position.set(bestBar.x, Math.max(yOf(bestBar.row.accuracy + bestBar.row.accuracy_std), yOf(bestBar.row.accuracy) + 0.3) + 0.22, ZA + 0.3);
    chart.add(bestLabel);
    let defPole = null, defLabel = null;
    if (defBar && !same) {
      defPole = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(defBar.x, CY, ZA + 0.3), new THREE.Vector3(defBar.x, CY + CH + 0.25, ZA + 0.3)]),
        new THREE.LineDashedMaterial({ color: 0xffffff, dashSize: 0.05, gapSize: 0.06, transparent: true, opacity: 0.8 }));
      defPole.computeLineDistances();
      chart.add(defPole);
      defLabel = textSprite(`default s = ${fmtS(defBar.s)}`, { height: 0.14, color: "#ffffff" });
      defLabel.position.set(defBar.x, CY + CH + 0.38, ZA + 0.3);
      chart.add(defLabel);
    }

    // ---- part 2 timeline ----
    const P1_END = MORPH0 + sDesc.length * STEP;
    const T_OUT = P1_END, T_OUT_END = P1_END + 1.0; // landscape flattens and fades
    const BAR0 = P1_END + 0.6;
    const BARS_DONE = BAR0 + (sAsc.length - 1) * BAR_STAGGER + BAR_RISE;
    const AUC0 = BARS_DONE, AUC1 = AUC0 + 0.8;
    const RBF0 = BARS_DONE + 0.3, RBF1 = RBF0 + 0.8;
    const MARK = RBF1 + 0.2; // best + default highlighted
    const DUR = MARK + 3.4;
    const iDef = sDesc.indexOf(d.default);
    const settled = (i) => MORPH0 + i * STEP + (i === 0 ? 0.25 : MORPH + 0.15);
    const stops = [...new Set([settled(0), iDef > 0 ? settled(iDef) : null, settled(sDesc.length - 1), MARK + 0.6].filter((x) => x !== null))].sort((a, z) => a - z);

    let lastKey = "";
    function update(t) {
      // ---- part 1 ----
      const grow = ease(seg(t, RISE0, RISE1));
      const ms = morphState(t, sDesc, { t0: MORPH0, morph: MORPH, hold: HOLD });
      const mix = ease(ms.mix);
      const out = ease(seg(t, T_OUT, T_OUT_END));
      const key = `${ms.from}|${ms.to}|${mix.toFixed(4)}|${grow.toFixed(4)}`;
      if (key !== lastKey && out < 1) { setLandscape(ms.from, ms.to, mix, grow); lastKey = key; }
      land.scale.set(1 - 0.25 * out, Math.max(0.001, 1 - out), 1 - 0.25 * out);
      setOpacity(land, 1 - out);
      const cur = mix > 0.5 ? ms.to : ms.from;
      for (const s of d.options) sLabels[s].visible = s === cur && grow > 0.5 && out < 0.6;

      // ---- part 2 ----
      const cin = ease(seg(t, T_OUT + 0.3, T_OUT_END + 0.3));
      chart.visible = cin > 0;
      if (chart.visible) {
        setOpacity(chart, cin);
        bars.forEach((q, i) => {
          const g = ease(seg(t, BAR0 + i * BAR_STAGGER, BAR0 + i * BAR_STAGGER + BAR_RISE));
          q.mesh.scale.y = Math.max(0.001, (yOf(q.row.accuracy) - CY) * g);
          q.mesh.visible = g > 0;
          setOpacity(q.whisk, cin * seg(g, 0.8, 1));
          setOpacity(q.val, cin * seg(g, 0.6, 1));
          const dg = seg(t, AUC0 + (i / sAsc.length) * 0.6, AUC0 + (i / sAsc.length) * 0.6 + 0.2);
          setOpacity(q.dot, cin * dg);
          const mk = ease(seg(t, MARK, MARK + 0.4));
          const c = q.isBest ? tmpC.setHex(PALETTE.quantum).lerp(new THREE.Color(GOLD), mk) : tmpC.setHex(PALETTE.quantum);
          q.mat.color.copy(c); q.mat.emissive.copy(c);
          q.mat.emissiveIntensity = q.isBest ? 0.3 + 0.25 * mk : 0.3;
        });
        aucLine.geometry.setDrawRange(0, Math.round(seg(t, AUC0, AUC1) * sAsc.length));
        const rb = ease(seg(t, RBF0, RBF1));
        rbfPlane.position.y = CY + (yOf(sw.rbf.accuracy) - CY) * rb;
        setOpacity(rbfPlane, cin * rb);
        setOpacity(rbfLabel, cin * seg(rb, 0.7, 1));
        setOpacity(dashed, cin * rb);
        setOpacity(rbfAucLabel, cin * seg(rb, 0.7, 1));
        const mk = ease(seg(t, MARK, MARK + 0.5));
        setOpacity(ring, cin * mk);
        setOpacity(bestLabel, cin * mk);
        bestGlow.visible = mk > 0;
        bestGlow.scale.setScalar(0.9 + 0.15 * Math.sin(t * 3));
        bestGlow.material.opacity = mk;
        if (defPole) { setOpacity(defPole, cin * mk); setOpacity(defLabel, cin * mk); }
      }

      // ---- caption ----
      if (t < T_OUT + 0.3) return caption1(t, ms, mix, grow);
      return caption2(t);
    }

    function chips(cur) {
      return sDesc.map((s) => s === cur
        ? `<b style="color:#fff;background:#4c1d95;border-radius:4px;padding:0 5px">${fmtS(s)}</b>`
        : `<span style="color:#64748b">${fmtS(s)}</span>`).join(" ");
    }
    function caption1(t, ms, mix, grow) {
      const cur = mix > 0.5 ? ms.to : ms.from;
      const r = regimeOf(d, cur);
      const dg = (d.diagnostics && d.diagnostics[sKey(cur)]) || {};
      const f = (v, n) => (typeof v === "number" ? v.toFixed(n) : "—");
      let html = `<div style="font-size:14px"><b>Kernel Bandwidth Lab</b> · Part 1: the quantum kernel matrix as a landscape · `
        + `${N} of ${nTrain} training patients, sorted <b style="color:${hex(PALETTE.healthy)}">healthy</b> → <b style="color:${hex(PALETTE.sick)}">disease</b> · height &amp; brightness = K`;
      html += `<div style="margin-top:3px">bandwidth s (largest → smallest): ${chips(cur)}${ms.mix < 1 && ms.index > 0 ? ` <span style="color:#94a3b8">morphing ${fmtS(ms.from)} → ${fmtS(ms.to)}</span>` : ""}</div>`;
      if (grow > 0.3) {
        html += `<div style="margin-top:3px"><b>s = ${fmtS(cur)}</b>${cur === d.default ? ` <span style="color:#94a3b8">(default)</span>` : ""} · ${regimeHtml(r)}${r.text.slice(r.short.length)}</div>`;
        html += `<div style="margin-top:2px">mean off-diagonal K = <b>${f(dg.kernel_mean, 3)}</b> · variance <b>${f(dg.kernel_var, 4)}</b> · kernel-target alignment <b>${f(dg.alignment, 3)}</b> <span style="color:#94a3b8">(all ${nTrain} training patients; too sharp below 0.15, too blurred above 0.85)</span></div>`;
      }
      html += `<div style="margin-top:2px;font-size:12px;color:#94a3b8">A useful kernel shows two raised blocks on the diagonal: same-diagnosis patients look more alike than mixed pairs. K = |⟨ψ(s·x)|ψ(s·x′)⟩|²; s shrinks the angles before encoding (like γ in an RBF kernel).</div></div>`;
      return html;
    }
    function caption2(t) {
      const rows = sw.rows;
      const sMax = Math.max(...d.options);
      const rowMax = rows.find((r) => r.bandwidth === sMax);
      let html = `<div style="font-size:14px"><b>Kernel Bandwidth Lab</b> · Part 2: does the bandwidth change diagnostic accuracy? `
        + `<span style="color:#94a3b8">Stratified ${nFolds}-fold cross-validation on all ${nPatients} patients · scaling and PCA refit inside every fold · exact (noise-free) statevector kernel</span>`;
      const parts = [];
      if (sw.def) parts.push(`QSVC CV accuracy at default s = ${fmtS(sw.def.bandwidth)}: <b>${pct1(sw.def.accuracy)}</b> ± ${pct1(sw.def.accuracy_std)}`);
      if (rowMax) parts.push(`at s = ${fmtS(sMax)} (unscaled): <b>${pct1(rowMax.accuracy)}</b>`);
      html += `<div style="margin-top:3px">${parts.join(" · ")}</div>`;
      html += `<div style="margin-top:2px;font-size:12px;color:#94a3b8"><span style="color:${hex(PALETTE.quantum)}">■</span> bars = QSVC CV accuracy (whisker ± 1 std)`
        + (t >= AUC0 ? ` · <span style="color:${hex(PALETTE.quantum2)}">●</span> dots = QSVC CV ROC-AUC` : "")
        + (t >= RBF0 ? ` · <span style="color:${hex(PALETTE.classical)}">▬</span> plane / dashed line = RBF SVM accuracy / ROC-AUC` : "") + `</div>`;
      if (t >= RBF0) {
        html += `<div style="margin-top:2px"><b style="color:${hex(PALETTE.classical)}">Classical RBF SVM (CV)</b>: accuracy <b>${pct1(sw.rbf.accuracy)}</b> ± ${pct1(sw.rbf.accuracy_std)} · ROC-AUC <b>${sw.rbf.roc_auc.toFixed(3)}</b></div>`;
      }
      if (t >= MARK) {
        const diff = (sw.best.accuracy - sw.rbf.accuracy) * 100;
        const vs = Math.abs(diff) < 0.05 ? "ties the RBF SVM" : `${diff > 0 ? "+" : "−"}${Math.abs(diff).toFixed(1)} pts vs the RBF SVM`;
        html += `<div style="margin-top:2px"><b style="color:${hex(GOLD)}">Best s by CV accuracy = ${fmtS(sw.best.bandwidth)}</b>: <b>${pct1(sw.best.accuracy)}</b> ± ${pct1(sw.best.accuracy_std)} · ROC-AUC ${sw.best.roc_auc.toFixed(3)} · ${vs}</div>`;
        html += `<div style="margin-top:2px;font-size:12px;color:#94a3b8">Caveat: picking the best s from this same table is itself a tuning step, so the best row is slightly optimistic (an unbiased estimate needs nested cross-validation).</div>`;
      }
      html += `</div>`;
      return html;
    }

    return { update, stops, DUR };
  }

  const STATUS = (msg, colour) => `<div style="font-size:14px"><b>Kernel Bandwidth Lab</b> · <span style="color:${colour}">${msg}</span></div>`;
  return {
    title: "Kernel Bandwidth Lab: tuning the quantum kernel",
    legend: LEGEND_BANDWIDTH,
    scene,
    cameraPos: [0, 4.9, 5.6],
    // timeline runs from the moment the data is ready (normally a few ms after the scene starts)
    get duration() { return built ? built.DUR + (tReady || 0) : 6; },
    get stops() { return built ? built.stops.map((s) => s + (tReady || 0)) : []; },
    update(t) {
      if (err) return STATUS(`${err}. The rest of the tour still works; run scripts/export_tour_extras.py to create it.`, hex(PALETTE.sick));
      if (!built) return STATUS("Loading kernel bandwidth data…", "#94a3b8");
      if (tReady === null) tReady = t;
      else if (t < tReady) tReady = 0; // clock moved back (e.g. ?pauseAt= snapping): use the scene clock as is
      return built.update(t - tReady);
    },
    dispose() {
      disposed = true;
      disposeTree(scene);
    },
  };
}
