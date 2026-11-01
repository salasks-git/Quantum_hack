import * as THREE from "three";
import { PALETTE, ease, seg, textSprite, glowSprite, disposeTree, addLights } from "./common.js";
import { MODEL_KEYS, MODEL_LABELS, verdictFor } from "../verdict.js";

export const duration = 7;

const FLOOR_Y = -0.4;
const H = 2.8; // column height at prob = 1
const GAP = 1.75;
const RADIUS = 0.36;
const QUANTUM = new Set(["qsvc", "vqc"]);
// column i rises over [RISE0 + i*STAGGER, RISE0 + i*STAGGER + RISE]
const RISE0 = 0.4, STAGGER = 0.65, RISE = 1.1, FLIP = 0.35;
const ACTUAL_AT = RISE0 + 3 * STAGGER + RISE + FLIP + 0.25;

const hex = (c) => "#" + c.toString(16).padStart(6, "0");
const pct = (p) => {
  const v = p * 100;
  return Math.round(v) === 50 ? v.toFixed(1) : String(Math.round(v));
};

export default function createScene(ctx) {
  const p = ctx.data.test[ctx.patient];
  const actualSick = p.label === 1;
  const scene = new THREE.Scene();
  addLights(scene);

  const floor = new THREE.GridHelper(8, 16, 0x334155, 0x1e293b);
  floor.scale.z = 0.4;
  floor.position.y = FLOOR_Y;
  scene.add(floor);

  // translucent 50% decision sheet
  const sheetY = FLOOR_Y + 0.5 * H;
  const sheet = new THREE.Mesh(
    new THREE.PlaneGeometry(7.6, 2.0),
    new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.07, side: THREE.DoubleSide, depthWrite: false })
  );
  sheet.rotation.x = -Math.PI / 2;
  sheet.position.y = sheetY;
  sheet.add(new THREE.LineSegments(
    new THREE.EdgesGeometry(sheet.geometry),
    new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.35 })
  ));
  scene.add(sheet);
  const fifty = textSprite("50%", { height: 0.2, color: "#cbd5e1" });
  fifty.position.set(-4.2, sheetY, 0);
  scene.add(fifty);

  const colGeo = new THREE.CylinderGeometry(RADIUS, RADIUS, 1, 40);
  colGeo.translate(0, 0.5, 0); // base at origin
  const cols = MODEL_KEYS.map((key, i) => {
    const m = p.models[key];
    const v = verdictFor(m);
    const base = QUANTUM.has(key) ? PALETTE.quantum : PALETTE.classical;
    const final = v.disease ? PALETTE.sick : base;
    const x = (i - 1.5) * GAP;
    const mat = new THREE.MeshPhongMaterial({
      color: base, emissive: base, emissiveIntensity: 0.3, transparent: true, opacity: 0.92, shininess: 60,
    });
    const mesh = new THREE.Mesh(colGeo, mat);
    mesh.position.set(x, FLOOR_Y, 0);
    scene.add(mesh);
    const glow = glowSprite(0xffffff, 1.4, 0.8); // white texture, tinted per frame via material.color
    glow.material.color.setHex(base);
    scene.add(glow);
    const name = textSprite(MODEL_LABELS[key], { height: 0.2, color: hex(base) });
    name.position.set(x, FLOOR_Y - 0.3, 0.75);
    scene.add(name);
    const val = textSprite(`${pct(m.prob)}%${v.borderline ? "\nborderline" : ""}`, {
      height: 0.24, color: "#ffffff",
    });
    if (v.borderline) val.scale.multiplyScalar(0.9);
    scene.add(val);
    const correct = v.disease === actualSick;
    return { key, m, v, x, mesh, mat, glow, val, base, final, correct, start: RISE0 + i * STAGGER };
  });

  const tmpA = new THREE.Color(), tmpB = new THREE.Color();
  const anyBorderline = cols.some((c) => c.v.borderline);
  const nCorrect = cols.filter((c) => c.correct).length;
  const cell = (c, done) => {
    const nm = `<b style="color:${hex(c.base)}">${MODEL_LABELS[c.key]}</b>`;
    if (!done) return `<div>${nm}<br><span style="color:#64748b">…</span></div>`;
    const verdict = c.v.disease
      ? `<b style="color:${hex(PALETTE.sick)}">DISEASE</b>`
      : `<b style="color:${hex(PALETTE.healthy)}">HEALTHY</b>`;
    const mark = c.correct
      ? `<span style="color:${hex(PALETTE.healthy)}">✓</span>`
      : `<span style="color:${hex(PALETTE.sick)}">✗</span>`;
    const bl = c.v.borderline ? ` <i style="color:#f0abfc">borderline</i>` : "";
    return `<div>${nm}<br>${verdict} · ${pct(c.m.prob)}% ${mark}${bl}</div>`;
  };

  return {
    title: "Finale: four verdicts",
    scene,
    cameraPos: [0, 1.4, 7.6],
    duration,
    update(t) {
      const doneFlags = cols.map((c) => {
        const g = ease(seg(t, c.start, c.start + RISE));
        const h = Math.max(0.01, c.m.prob * H * g);
        c.mesh.scale.y = h;
        const top = FLOOR_Y + h;
        c.glow.position.set(c.x, top, 0);
        c.glow.visible = g > 0;
        c.val.position.set(c.x, top + 0.42 + (c.v.borderline ? 0.12 : 0), 0);
        c.val.material.opacity = ease(seg(t, c.start + RISE - 0.2, c.start + RISE + 0.1));
        const f = ease(seg(t, c.start + RISE, c.start + RISE + FLIP));
        tmpA.setHex(c.base).lerp(tmpB.setHex(c.final), f);
        c.mat.color.copy(tmpA);
        c.mat.emissive.copy(tmpA);
        c.glow.material.color.copy(tmpA);
        return f >= 1;
      });
      let html = `<b>Four verdicts</b> · height = disease probability · sheet = <b>50% line</b> · <span style="color:${hex(PALETTE.sick)}">red</span> = model says disease`;
      html += `<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:4px 18px;margin-top:8px">${cols.map((c, i) => cell(c, doneFlags[i])).join("")}</div>`;
      if (t >= ACTUAL_AT) {
        const actual = actualSick
          ? `<b style="color:${hex(PALETTE.sick)}">Disease</b>`
          : `<b style="color:${hex(PALETTE.healthy)}">Healthy</b>`;
        html += `<div style="margin-top:8px">Actual diagnosis: ${actual} · ${nCorrect} of 4 models correct${
          anyBorderline ? " · <i>borderline</i>: probability and decision disagree; verdict follows the decision" : ""
        }</div>`;
      }
      return html;
    },
    dispose() {
      disposeTree(scene);
    },
  };
}
