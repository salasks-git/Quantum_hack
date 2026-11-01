import * as THREE from "three";
import { PALETTE, ease, seg, textSprite, glowSprite, disposeTree, addLights } from "./common.js";
import { MODEL_KEYS, MODEL_LABELS, verdictFor } from "../verdict.js";
import { clinicalScores, clinicalLeader } from "../clinical.js";

const FLOOR_Y = -0.4;
const H = 2.45; // column height at prob = 1 (shorter than v1 so 99% columns + labels stay on screen above the scoreboard)
const GAP = 1.6;
const RADIUS = 0.33;
const QUANTUM = new Set(["qsvc", "vqc"]);
const HYBRID_LO = PALETTE.classical, HYBRID_HI = PALETTE.quantum; // hybrid column: cyan base -> violet top
const HYBRID_MID = new THREE.Color(HYBRID_LO).lerp(new THREE.Color(HYBRID_HI), 0.5).getHex();
const GOLD = 0xfbbf24;
const LIFT = 0.95; // world-y lift of the whole set
const ARC_MAX = 2.55; // local y cap for the particle arcs (stays on screen)

// Timeline (seconds). The four single models rise one after another: column i rises over
// [RISE0 + i*STAGGER, RISE0 + i*STAGGER + RISE], then flips colour over FLIP.
const RISE0 = 0.4, STAGGER = 0.65, RISE = 1.1, FLIP = 0.35;
const SINGLES_DONE = RISE0 + 3 * STAGGER + RISE + FLIP; // 3.8
// Then RBF and QSVC pour into the hybrid column, which fills as the particles land.
const POUR0 = SINGLES_DONE + 0.25; // first particles leave RBF / QSVC
const EMIT = 1.3; // particles leave over this window
const TRAVEL = 0.9; // seconds each particle is in flight
const HYB_RISE0 = POUR0 + TRAVEL * 0.6;
const HYB_RISE1 = POUR0 + EMIT + TRAVEL; // last particle lands as the column reaches its height
const HYB_DONE = HYB_RISE1 + FLIP;
const ACTUAL_AT = HYB_DONE + 0.25;
const SCORE_AT = ACTUAL_AT + 0.5; // scoreboard rows appear
const LEADER_AT = SCORE_AT + 0.6; // leader badge + floor ring
export const duration = LEADER_AT + 2.6;
const N_PER_STREAM = 60;

const hex = (c) => "#" + c.toString(16).padStart(6, "0");
const pct = (p) => {
  const v = p * 100;
  return Math.round(v) === 50 ? v.toFixed(1) : String(Math.round(v));
};
const pct1 = (p) => (p * 100).toFixed(1);
const GRAD_TEXT = `background:linear-gradient(90deg,${hex(HYBRID_LO)},${hex(HYBRID_HI)});-webkit-background-clip:text;background-clip:text;color:transparent`;
const nameHtml = (key) =>
  key === "hybrid"
    ? `<b style="${GRAD_TEXT}">${MODEL_LABELS[key]}</b>`
    : `<b style="color:${hex(QUANTUM.has(key) ? PALETTE.quantum : PALETTE.classical)}">${MODEL_LABELS[key]}</b>`;

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

export default function createScene(ctx) {
  const test = ctx.data.test;
  const p = test[ctx.patient];
  const actualSick = p.label === 1;
  const nSick = test.filter((q) => q.label === 1).length;
  const scores = Object.fromEntries(MODEL_KEYS.map((k) => [k, clinicalScores(test, k)]));
  const leader = clinicalLeader(test, MODEL_KEYS);
  const scene = new THREE.Scene();
  addLights(scene);
  const root = new THREE.Group(); // lifted so the columns clear the scoreboard caption
  root.position.y = LIFT;
  scene.add(root);

  const floor = new THREE.GridHelper(10, 20, 0x334155, 0x1e293b);
  floor.scale.z = 0.32;
  floor.position.y = FLOOR_Y;
  root.add(floor);

  // translucent 50% decision sheet
  const sheetY = FLOOR_Y + 0.5 * H;
  const sheet = new THREE.Mesh(
    new THREE.PlaneGeometry(8.6, 2.0),
    new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.07, side: THREE.DoubleSide, depthWrite: false })
  );
  sheet.rotation.x = -Math.PI / 2;
  sheet.position.y = sheetY;
  sheet.add(new THREE.LineSegments(
    new THREE.EdgesGeometry(sheet.geometry),
    new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.35 })
  ));
  root.add(sheet);
  const fifty = textSprite("50%", { height: 0.2, color: "#cbd5e1" });
  fifty.position.set(-4.7, sheetY, 0);
  root.add(fifty);

  const colGeo = new THREE.CylinderGeometry(RADIUS, RADIUS, 1, 40);
  colGeo.translate(0, 0.5, 0); // base at origin

  // Hybrid column: own geometry, vertex-coloured as a twisted half-cyan / half-violet braid
  // (literally ½ RBF + ½ QSVC), blending smoothly where the halves meet.
  const hybGeo = colGeo.clone();
  const hybPos = hybGeo.attributes.position;
  const hybBase = new Float32Array(hybPos.count * 3);
  const cLo = new THREE.Color(HYBRID_LO), cHi = new THREE.Color(HYBRID_HI), cTmp = new THREE.Color();
  for (let i = 0; i < hybPos.count; i++) {
    const a = Math.atan2(hybPos.getZ(i), hybPos.getX(i)) + hybPos.getY(i) * 2.2;
    cTmp.copy(cLo).lerp(cHi, ease(seg(0.5 + 0.5 * Math.cos(a), 0.2, 0.8)));
    cTmp.toArray(hybBase, i * 3);
  }
  hybGeo.setAttribute("color", new THREE.BufferAttribute(hybBase.slice(), 3));

  const cols = MODEL_KEYS.map((key, i) => {
    const m = p.models[key];
    const v = verdictFor(m);
    const hybrid = key === "hybrid";
    const base = hybrid ? HYBRID_MID : QUANTUM.has(key) ? PALETTE.quantum : PALETTE.classical;
    const final = v.disease ? PALETTE.sick : base;
    const x = (i - 2) * GAP;
    const mat = new THREE.MeshPhongMaterial({
      color: hybrid ? 0xffffff : base, emissive: hybrid ? 0x000000 : base, emissiveIntensity: 0.3,
      transparent: true, opacity: 0.92, shininess: 60, vertexColors: hybrid,
    });
    if (hybrid) {
      // same self-glow as the other columns (emissive = colour at 0.3), but per vertex
      mat.onBeforeCompile = (sh) => {
        sh.fragmentShader = sh.fragmentShader.replace(
          "vec3 totalEmissiveRadiance = emissive;",
          "vec3 totalEmissiveRadiance = emissive + vColor.rgb * 0.3;"
        );
      };
    }
    const mesh = new THREE.Mesh(hybrid ? hybGeo : colGeo, mat);
    mesh.position.set(x, FLOOR_Y, 0);
    root.add(mesh);
    const glow = glowSprite(0xffffff, 1.3, 0.8); // white texture, tinted per frame via material.color
    glow.material.color.setHex(base);
    root.add(glow);
    const name = textSprite(hybrid ? "Hybrid\n(QSVC+RBF)" : MODEL_LABELS[key], { height: 0.19, color: hex(base) });
    name.position.set(x, FLOOR_Y - (hybrid ? 0.42 : 0.3), 0.75);
    root.add(name);
    const val = textSprite(`${pct(m.prob)}%${v.borderline ? "\nborderline" : ""}`, {
      height: 0.24, color: "#ffffff",
    });
    if (v.borderline) val.scale.multiplyScalar(0.9);
    root.add(val);
    const correct = v.disease === actualSick;
    const start = hybrid ? HYB_RISE0 : RISE0 + i * STAGGER;
    const rise = hybrid ? HYB_RISE1 - HYB_RISE0 : RISE;
    return { key, m, v, x, mesh, mat, glow, val, base, final, correct, start, rise, hybrid };
  });
  const byKey = Object.fromEntries(cols.map((c) => [c.key, c]));
  const hyb = byKey.hybrid;

  // Two particle streams: RBF (cyan) and QSVC (violet) pour into the hybrid column.
  const streams = ["rbf", "qsvc"].map((key, s) => {
    const src = byKey[key];
    const n = N_PER_STREAM;
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(n * 3), 3));
    geo.setAttribute("color", new THREE.BufferAttribute(new Float32Array(n * 3), 3));
    const pts = new THREE.Points(geo, new THREE.PointsMaterial({
      size: 0.26, map: dotTexture(), vertexColors: true, transparent: true, depthWrite: false,
      blending: THREE.AdditiveBlending, sizeAttenuation: true,
    }));
    pts.frustumCulled = false;
    root.add(pts);
    // deterministic per-particle jitter
    const jit = Array.from({ length: n }, (_, i) => {
      const a = Math.sin((i + 1) * 12.9898 + s * 78.233) * 43758.5453;
      const b = Math.sin((i + 1) * 39.3468 + s * 11.135) * 24634.6345;
      return [(a - Math.floor(a) - 0.5) * 0.22, (b - Math.floor(b) - 0.5) * 0.22];
    });
    return { src, geo, pts, n, jit, color: new THREE.Color(src.base), lift: 0.7 + s * 0.3 };
  });

  // golden ring on the floor under the clinical leader
  const ring = new THREE.Mesh(
    new THREE.RingGeometry(RADIUS + 0.08, RADIUS + 0.2, 48),
    new THREE.MeshBasicMaterial({ color: GOLD, transparent: true, opacity: 0, side: THREE.DoubleSide, depthWrite: false })
  );
  ring.rotation.x = -Math.PI / 2;
  ring.position.set(byKey[leader].x, FLOOR_Y + 0.01, 0);
  root.add(ring);

  const tmpA = new THREE.Color(), tmpB = new THREE.Color();
  const sick = new THREE.Color(PALETTE.sick);
  const anyBorderline = cols.some((c) => c.v.borderline);
  const nCorrect = cols.filter((c) => c.correct).length;
  const P0 = new THREE.Vector3(), P1 = new THREE.Vector3(), P2 = new THREE.Vector3(), Q = new THREE.Vector3();
  let lastHybFlip = -1;

  const verdictCell = (c, done) => {
    if (!done) return `<span style="color:#64748b">…</span>`;
    const verdict = c.v.disease
      ? `<b style="color:${hex(PALETTE.sick)}">DISEASE</b>`
      : `<b style="color:${hex(PALETTE.healthy)}">HEALTHY</b>`;
    const mark = c.correct
      ? `<span style="color:${hex(PALETTE.healthy)}">✓</span>`
      : `<span style="color:${hex(PALETTE.sick)}">✗</span>`;
    const bl = c.v.borderline ? ` <i style="color:#f0abfc">borderline</i>` : "";
    return `${verdict} · ${pct(c.m.prob)}% ${mark}${bl}`;
  };

  return {
    title: "Finale: five verdicts + clinical scoreboard",
    scene,
    cameraPos: [0, 1.4, 8.6],
    duration,
    stops: [
      ...[0, 1, 2, 3].map((i) => RISE0 + i * STAGGER + RISE + FLIP),
      HYB_DONE, // hybrid column resolved
      LEADER_AT + 0.4, // scoreboard + leader badge on screen
    ],
    update(t) {
      const doneFlags = cols.map((c) => {
        const g = ease(seg(t, c.start, c.start + c.rise));
        const h = Math.max(0.01, c.m.prob * H * g);
        c.mesh.scale.y = h;
        c.mesh.visible = !c.hybrid || t >= POUR0;
        const top = FLOOR_Y + h;
        c.glow.position.set(c.x, top, 0);
        c.glow.visible = g > 0;
        c.val.position.set(c.x, top + 0.42 + (c.v.borderline ? 0.12 : 0), 0);
        c.val.material.opacity = ease(seg(t, c.start + c.rise - 0.2, c.start + c.rise + 0.1));
        const f = ease(seg(t, c.start + c.rise, c.start + c.rise + FLIP));
        tmpA.setHex(c.base).lerp(tmpB.setHex(c.final), f);
        c.glow.material.color.copy(tmpA);
        if (c.hybrid) {
          // blend the gradient itself towards rose when the hybrid says disease
          const k = c.v.disease ? f : 0;
          if (k !== lastHybFlip) {
            const arr = c.mesh.geometry.attributes.color;
            for (let i = 0; i < arr.count; i++) {
              tmpB.fromArray(hybBase, i * 3).lerp(sick, k);
              arr.setXYZ(i, tmpB.r, tmpB.g, tmpB.b);
            }
            arr.needsUpdate = true;
            lastHybFlip = k;
          }
        } else {
          c.mat.color.copy(tmpA);
          c.mat.emissive.copy(tmpA);
        }
        return f >= 1;
      });

      // particle streams
      const hTop = FLOOR_Y + Math.max(0.01, hyb.m.prob * H * ease(seg(t, hyb.start, hyb.start + hyb.rise)));
      for (const st of streams) {
        const pos = st.geo.attributes.position, col = st.geo.attributes.color;
        const srcTop = FLOOR_Y + st.src.m.prob * H;
        for (let i = 0; i < st.n; i++) {
          const te = POUR0 + (i / st.n) * EMIT;
          const u = (t - te) / TRAVEL;
          if (u <= 0 || u >= 1) { pos.setXYZ(i, 0, -100, 0); col.setXYZ(i, 0, 0, 0); continue; }
          const [jx, jz] = st.jit[i];
          P0.set(st.src.x + jx * 0.6, srcTop + 0.05, jz * 0.6);
          P2.set(hyb.x + jx * 0.5, hTop + 0.05, jz * 0.5);
          P1.set((P0.x + P2.x) / 2 + jx, Math.min(ARC_MAX, Math.max(P0.y, P2.y) + st.lift), jz * 2);
          Q.copy(P0).multiplyScalar((1 - u) * (1 - u))
            .addScaledVector(P1, 2 * (1 - u) * u)
            .addScaledVector(P2, u * u);
          pos.setXYZ(i, Q.x, Q.y, Q.z);
          const fade = Math.min(1, u * 6, (1 - u) * 5);
          col.setXYZ(i, st.color.r * fade, st.color.g * fade, st.color.b * fade);
        }
        pos.needsUpdate = true;
        col.needsUpdate = true;
      }
      ring.material.opacity = 0.85 * ease(seg(t, LEADER_AT, LEADER_AT + 0.4));

      // caption: per-patient verdicts + clinical scoreboard (one row per model)
      let html = `<div style="font-size:14px"><b>Five verdicts</b> · height = disease probability · sheet = <b>50% line</b> · <span style="color:${hex(PALETTE.sick)}">red</span> = model says disease`;
      if (t >= POUR0 - 0.1) {
        html += `<div style="margin-top:3px"><b style="${GRAD_TEXT}">Hybrid</b> = ½·RBF + ½·QSVC = ½·${pct1(byKey.rbf.m.prob)}% + ½·${pct1(byKey.qsvc.m.prob)}% → <b>${pct1(hyb.m.prob)}%</b></div>`;
      }
      const showScore = t >= SCORE_AT;
      const showLeader = t >= LEADER_AT;
      const op = ease(seg(t, SCORE_AT, SCORE_AT + 0.4)).toFixed(2);
      const muted = (x) => `<span style="color:#64748b">${x}</span>`;
      const th = (x) => `<div style="color:#94a3b8;font-size:12px;align-self:end;white-space:nowrap;padding:0 4px">${x}</div>`;
      const grid = [th("Model"), th("This patient"), th(`F2 · all ${test.length} patients`), th(`Missed · of ${nSick} sick`), th("")];
      cols.forEach((c, i) => {
        const s = scores[c.key];
        const lead = showLeader && c.key === leader;
        const bg = lead ? "background:rgba(251,191,36,.12);" : "";
        const cell = (x, extra = "") => `<div style="${bg}padding:1px 4px;border-radius:4px;white-space:nowrap;${extra}">${x}</div>`;
        grid.push(cell(nameHtml(c.key)));
        grid.push(cell(verdictCell(c, doneFlags[i])));
        grid.push(cell(showScore ? `<b${lead ? ` style="color:${hex(GOLD)}"` : ""}>${s.f2.toFixed(3)}</b>` : muted("…"), `opacity:${showScore ? op : 1}`));
        grid.push(cell(showScore
          ? `<b style="color:${hex(s.missed === 0 ? PALETTE.healthy : PALETTE.sick)}">${s.missed}</b> of ${nSick}`
          : muted("…"), `opacity:${showScore ? op : 1}`));
        grid.push(cell(lead
          ? `<span style="display:inline-block;background:${hex(GOLD)};color:#070b16;font-weight:700;font-size:12px;border-radius:999px;padding:0 8px">🏆 Clinical leader (F2)</span>`
          : ""));
      });
      html += `<div style="display:grid;grid-template-columns:auto auto auto auto 1fr;gap:1px 10px;margin-top:6px;line-height:1.4">${grid.join("")}</div>`;
      if (t >= ACTUAL_AT) {
        const actual = actualSick
          ? `<b style="color:${hex(PALETTE.sick)}">Disease</b>`
          : `<b style="color:${hex(PALETTE.healthy)}">Healthy</b>`;
        html += `<div style="margin-top:5px">Actual diagnosis: ${actual} · ${nCorrect} of 5 models correct for this patient${
          anyBorderline ? ` · <i style="color:#f0abfc">borderline</i>: probability and decision disagree; verdict follows the decision` : ""
        }</div>`;
      }
      if (showScore) {
        html += `<div style="margin-top:2px;font-size:12px;color:#94a3b8;opacity:${op}">Scoreboard: every test patient of this disease · F2 weighs recall 2× precision · missed = sick patients called healthy</div>`;
      }
      html += `</div>`;
      return html;
    },
    dispose() {
      disposeTree(scene);
    },
  };
}
