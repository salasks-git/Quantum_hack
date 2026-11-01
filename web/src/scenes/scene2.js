import * as THREE from "three";
import { legend2 } from "./legends.js";
import {
  PALETTE, ease, seg, textSprite, glowSprite, setOpacity, disposeTree, addLights,
  pos3, sectionPlane, contourSegments, clipLineToSquare, xName,
} from "./common.js";
import { verdictFor } from "../verdict.js";

export const duration = 11;

// Timeline (seconds)
const T = {
  webIn: [0.0, 0.8], // box + training dots
  you: [0.8, 1.3], // chosen patient
  planeIn: [1.4, 2.9], // linear plane slides in along its normal
  linVerdict: 2.9,
  planeOut: [5.0, 6.3], // plane fades to a faint ghost
  rbfIn: [4.9, 7.4], // RBF surface grows: level set sweeps down to decision value 0
  rbfVerdict: 7.4,
};

const S = 0.85; // world units per radian
const WEB_Y = 0.45;
const SLIDE = 2.4; // world units the plane travels along its normal
const LEVEL_STEPS = 40; // quantised levels for the growth animation (meshes are cached)
const GHOST = 0.25; // opacity of the linear plane while the RBF surface shows
const LINE_W = 0.07; // world width of the boundary lines drawn for 2 qubits (flat layout)

const hex = (c) => "#" + c.toString(16).padStart(6, "0");
const n2 = (v) => v.toFixed(2);
const pct = (p) => {
  const v = p * 100;
  return Math.round(v) === 50 ? v.toFixed(1) : String(Math.round(v));
};

/** Convex polygon where the plane n·x = d cuts the axis-aligned box [lo, hi]³ (CCW about n). */
function planeBoxPolygon(n, d, lo, hi) {
  const corners = [];
  for (let i = 0; i < 8; i++) corners.push(new THREE.Vector3(i & 1 ? hi : lo, i & 2 ? hi : lo, i & 4 ? hi : lo));
  const pts = [];
  for (let a = 0; a < 8; a++) {
    for (const bit of [1, 2, 4]) {
      if (a & bit) continue;
      const b = a | bit;
      const da = n.dot(corners[a]) - d, db = n.dot(corners[b]) - d;
      if ((da > 0) === (db > 0) || da === db) continue;
      pts.push(corners[a].clone().lerp(corners[b], da / (da - db)));
    }
  }
  if (pts.length < 3) return [];
  const c = pts.reduce((s, p) => s.add(p), new THREE.Vector3()).divideScalar(pts.length);
  const u = pts[0].clone().sub(c).normalize();
  const v = new THREE.Vector3().crossVectors(n, u);
  return pts
    .map((p) => { const q = p.clone().sub(c); return [Math.atan2(q.dot(v), q.dot(u)), p]; })
    .sort((a, b) => a[0] - b[0])
    .map(([, p]) => p);
}

/**
 * Surface nets over the exported RBF grid: the level set `values = level`.
 * values[(i*G + j)*G + k] is the decision value at (axis[i], axis[j], axis[k]).
 * Triangles are wound so their normals point towards larger values (the disease side).
 */
function surfaceNets(values, G, axis, level, toWorld) {
  const val = (i, j, k) => values[(i * G + j) * G + k] - level;
  const cellIdx = new Int32Array((G - 1) ** 3).fill(-1);
  const cid = (i, j, k) => (i * (G - 1) + j) * (G - 1) + k;
  const pos = [];
  const EDGES = [];
  for (let a = 0; a < 8; a++) for (const bit of [1, 2, 4]) if (!(a & bit)) EDGES.push([a, a | bit]);
  const cv = new Float64Array(8);
  for (let i = 0; i < G - 1; i++) for (let j = 0; j < G - 1; j++) for (let k = 0; k < G - 1; k++) {
    let mask = 0;
    for (let c = 0; c < 8; c++) {
      cv[c] = val(i + (c & 1), j + ((c >> 1) & 1), k + ((c >> 2) & 1));
      if (cv[c] > 0) mask |= 1 << c;
    }
    if (mask === 0 || mask === 255) continue;
    let sx = 0, sy = 0, sz = 0, n = 0;
    for (const [a, b] of EDGES) {
      if ((cv[a] > 0) === (cv[b] > 0)) continue;
      const t = cv[a] / (cv[a] - cv[b]);
      const ax = (a & 1), ay = (a >> 1) & 1, az = (a >> 2) & 1;
      const bx = (b & 1), by = (b >> 1) & 1, bz = (b >> 2) & 1;
      sx += ax + (bx - ax) * t; sy += ay + (by - ay) * t; sz += az + (bz - az) * t;
      n++;
    }
    const h = axis[1] - axis[0];
    const w = toWorld([axis[i] + (sx / n) * h, axis[j] + (sy / n) * h, axis[k] + (sz / n) * h]);
    cellIdx[cid(i, j, k)] = pos.length / 3;
    pos.push(w.x, w.y, w.z);
  }
  const idx = [];
  const A = new THREE.Vector3(), B = new THREE.Vector3(), C = new THREE.Vector3();
  const quad = (cells, dir) => { // cells: the 4 cells around an edge; dir: desired normal
    let q = cells.map((c) => cellIdx[c]);
    if (q.some((x) => x < 0)) return;
    A.fromArray(pos, q[0] * 3); B.fromArray(pos, q[1] * 3); C.fromArray(pos, q[2] * 3);
    const nrm = B.sub(A).cross(C.sub(A));
    if (nrm.dot(dir) < 0) q = [q[0], q[3], q[2], q[1]];
    idx.push(q[0], q[1], q[2], q[0], q[2], q[3]);
  };
  const dirs = [new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 1)];
  const neg = new THREE.Vector3();
  for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) for (let k = 0; k < G; k++) {
    const v0 = val(i, j, k);
    // x-edge (i,j,k)-(i+1,j,k), shared by cells (i, j-1..j, k-1..k)
    if (i < G - 1 && j > 0 && k > 0 && j < G - 1 && k < G - 1 && (v0 > 0) !== (val(i + 1, j, k) > 0)) {
      const up = val(i + 1, j, k) > v0 ? dirs[0] : neg.copy(dirs[0]).negate();
      quad([cid(i, j - 1, k - 1), cid(i, j, k - 1), cid(i, j, k), cid(i, j - 1, k)], up.clone());
    }
    if (j < G - 1 && i > 0 && k > 0 && i < G - 1 && k < G - 1 && (v0 > 0) !== (val(i, j + 1, k) > 0)) {
      const up = val(i, j + 1, k) > v0 ? dirs[1] : neg.copy(dirs[1]).negate();
      quad([cid(i - 1, j, k - 1), cid(i, j, k - 1), cid(i, j, k), cid(i - 1, j, k)], up.clone());
    }
    if (k < G - 1 && i > 0 && j > 0 && i < G - 1 && j < G - 1 && (v0 > 0) !== (val(i, j, k + 1) > 0)) {
      const up = val(i, j, k + 1) > v0 ? dirs[2] : neg.copy(dirs[2]).negate();
      quad([cid(i - 1, j - 1, k), cid(i, j - 1, k), cid(i, j, k), cid(i - 1, j, k)], up.clone());
    }
  }
  if (!idx.length) return null;
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  g.setIndex(idx);
  g.computeVertexNormals();
  return g;
}

/** Translucent classical-cyan skin: front faces (towards the disease side) slightly brighter than back faces. */
function skinMaterials(opacity) {
  const mk = (side, glow) => new THREE.MeshPhongMaterial({
    color: PALETTE.classical, emissive: PALETTE.classical, emissiveIntensity: glow, transparent: true, opacity,
    side, depthWrite: false, shininess: 80, specular: 0x7dd3fc,
  });
  return { front: mk(THREE.FrontSide, 0.4), back: mk(THREE.BackSide, 0.22) };
}

/** 2 qubits: flat ribbon (width `w`, in the z = 0 plane) along 2D segments [x1, y1, x2, y2] given in world units. */
function ribbonGeometry(segs, w) {
  const pos = new Float32Array(segs.length * 18);
  segs.forEach(([x1, y1, x2, y2], s) => {
    const len = Math.hypot(x2 - x1, y2 - y1) || 1;
    const ox = (-(y2 - y1) / len) * (w / 2), oy = ((x2 - x1) / len) * (w / 2);
    // extend each piece by half a width so neighbouring segments overlap at the joints
    const ex = ((x2 - x1) / len) * (w / 2), ey = ((y2 - y1) / len) * (w / 2);
    const a = [x1 - ex, y1 - ey], b = [x2 + ex, y2 + ey];
    const q = [[a[0] - ox, a[1] - oy], [b[0] - ox, b[1] - oy], [b[0] + ox, b[1] + oy], [a[0] + ox, a[1] + oy]];
    [0, 1, 2, 0, 2, 3].forEach((c, k) => pos.set([q[c][0], q[c][1], 0], s * 18 + k * 3));
  });
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  return g;
}

/** Sick-coloured arrow at `pos` pointing along unit `dir`, labelled "disease side". */
function diseaseArrow(pos, dir) {
  const a = new THREE.Group();
  const mat = new THREE.MeshBasicMaterial({ color: PALETTE.sick, transparent: true });
  const shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.022, 0.022, 0.5, 8), mat);
  shaft.position.y = 0.25;
  const tip = new THREE.Mesh(new THREE.ConeGeometry(0.075, 0.2, 14), mat.clone());
  tip.position.y = 0.6;
  const lab = textSprite("disease side", { height: 0.13, color: hex(PALETTE.sick), bg: "rgba(7,11,22,0.7)" });
  lab.position.y = 0.88;
  a.add(shaft, tip, lab);
  a.position.copy(pos);
  a.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
  return a;
}

export default function createScene(ctx) {
  const { data } = ctx;
  const pi = ctx.patient;
  const p = data.test[pi];
  const youP = p.label == null ? "your custom patient" : `your patient #${pi}`; // M3
  const tAngles = data.train.angles;
  const tLabels = data.train.labels;
  const nTrain = tAngles.length;
  const { coef, intercept } = data.linear;
  const rbf = data.rbf;
  const G = rbf.grid_size;
  const axis = Array.from({ length: G }, (_, i) => rbf.min + ((rbf.max - rbf.min) * i) / (G - 1));
  // Number of qubits: 3 = the shipped tour; 2 = flat (x₁, x₂ at z = 0); 4 = 3D slice at x₄ = rbf.slice.value.
  const nQ = p.angles.length;
  const flat = nQ === 2;
  const x4 = nQ > 3
    ? (rbf.slice ? rbf.slice.value : tAngles.reduce((s, a) => s + a[3], 0) / nTrain)
    : null;
  const fixed = nQ > 3 ? { 3: x4 } : {};
  const sub4 = xName(3);

  const scene = new THREE.Scene();
  addLights(scene);
  const web = new THREE.Group();
  web.position.y = WEB_Y;
  scene.add(web);
  const C0 = Math.PI / 2;
  const toWorld = (a) => { const q = pos3(a, C0); return new THREE.Vector3((q[0] - C0) * S, (q[1] - C0) * S, (q[2] - C0) * S); };
  const lo = (rbf.min - C0) * S, hi = (rbf.max - C0) * S;

  // ---------------- box, axes, dots ----------------
  let box;
  if (flat) { // 2 qubits: a square frame with a faint fill on the z = 0 plane
    box = new THREE.Group();
    box.add(new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.PlaneGeometry(hi - lo, hi - lo)),
      new THREE.LineBasicMaterial({ color: PALETTE.classical, transparent: true, opacity: 0.35 })
    ));
    box.add(new THREE.Mesh(
      new THREE.PlaneGeometry(hi - lo, hi - lo),
      new THREE.MeshBasicMaterial({ color: PALETTE.classical, transparent: true, opacity: 0.05, side: THREE.DoubleSide, depthWrite: false })
    ));
  } else {
    box = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(hi - lo, hi - lo, hi - lo)),
      new THREE.LineBasicMaterial({ color: PALETTE.classical, transparent: true, opacity: 0.2 })
    );
  }
  web.add(box);
  (flat
    ? [["x₁", [hi + 0.25, lo, 0]], ["x₂", [lo, hi + 0.22, 0]]]
    : [["x₁", [hi + 0.25, lo, lo]], ["x₂", [lo, hi + 0.22, lo]], ["x₃", [lo - 0.2, lo, hi + 0.25]]]
  ).forEach(([t, xyz]) => {
    const s = textSprite(t, { height: 0.18, color: "#7dd3fc" });
    s.material.sizeAttenuation = false;
    s.scale.multiplyScalar(0.034 / s.scale.y);
    s.position.set(...xyz);
    web.add(s);
  });

  const dotGeo = new THREE.SphereGeometry(0.06, 12, 10);
  const dots = new THREE.Group();
  tAngles.forEach((a, j) => {
    const col = tLabels[j] === 1 ? PALETTE.sick : PALETTE.healthy;
    const m = new THREE.Mesh(dotGeo, new THREE.MeshBasicMaterial({ color: col, transparent: true }));
    m.position.copy(toWorld(a));
    m.add(glowSprite(col, 0.28, 0.6));
    dots.add(m);
  });
  web.add(dots);

  const youPos = toWorld(p.angles);
  const you = new THREE.Group();
  you.position.copy(youPos);
  const youBall = new THREE.Mesh(new THREE.SphereGeometry(0.11, 20, 16), new THREE.MeshBasicMaterial({ color: 0xffffff }));
  const youGlow = glowSprite(0xffffff, 0.6, 0.9);
  const halo = glowSprite(0xffffff, 1.25, 1.0); // verdict-coloured halo, tinted per phase
  halo.material.opacity = 0;
  const ring = new THREE.Mesh(
    new THREE.RingGeometry(0.19, 0.235, 48),
    new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0, depthWrite: false, side: THREE.DoubleSide })
  );
  ring.onBeforeRender = (r, sc, cam) => ring.quaternion.copy(cam.quaternion); // always faces the viewer
  const youTag = textSprite(`#${pi}`, { height: 0.17, color: "#ffffff" });
  youTag.position.y = 0.28;
  you.add(halo, ring, youBall, youGlow, youTag);
  web.add(you);

  // ---------------- linear plane: coef·x + intercept = 0 ----------------
  // In world coordinates x = w/S + π/2, so the plane is n·w = dW with n = coef/|coef|.
  // 4 qubits: the drawn plane is the 3D section at x₄ = slice value (x₄ term folded into the intercept).
  // 2 qubits: the "plane" is a line on z = 0, drawn as a flat ribbon.
  const sec = sectionPlane(coef, intercept, fixed);
  const lc = sec.coef;
  const cLen = Math.hypot(...lc);
  const hasPlane = cLen > 1e-9;
  const nrm = hasPlane ? new THREE.Vector3(lc[0], lc[1], flat ? 0 : lc[2]).divideScalar(cLen) : new THREE.Vector3(0, 1, 0);
  const dW = hasPlane ? (-(lc.reduce((s, w) => s + w * C0, 0) + sec.intercept) * S) / cLen : 0;
  const plane = new THREE.Group();
  web.add(plane);
  const planeSkin = skinMaterials(flat ? 0.9 : 0.32);
  const MAXV = 8;
  const pPos = new THREE.BufferAttribute(new Float32Array(MAXV * 3), 3);
  const pGeo = new THREE.BufferGeometry();
  pGeo.setAttribute("position", pPos);
  pGeo.setIndex(Array.from({ length: MAXV - 2 }, (_, i) => [0, i + 1, i + 2]).flat());
  const nAttr = new THREE.BufferAttribute(new Float32Array(MAXV * 3), 3);
  for (let i = 0; i < MAXV; i++) (flat ? nAttr.setXYZ(i, 0, 0, 1) : nAttr.setXYZ(i, nrm.x, nrm.y, nrm.z));
  pGeo.setAttribute("normal", nAttr);
  plane.add(new THREE.Mesh(pGeo, planeSkin.front), new THREE.Mesh(pGeo, planeSkin.back));
  const rimGeo = new THREE.BufferGeometry();
  const rimPos = new THREE.BufferAttribute(new Float32Array(MAXV * 3), 3);
  rimGeo.setAttribute("position", rimPos);
  const rim = new THREE.LineLoop(rimGeo, new THREE.LineBasicMaterial({ color: PALETTE.classical, transparent: true, opacity: 0.95 }));
  plane.add(rim);
  const lineRibbon = (d) => { // the boundary line clipped to the square, as a 4-vertex ribbon
    const ends = clipLineToSquare([nrm.x, nrm.y], d, lo, hi);
    if (!ends) return [];
    const o = new THREE.Vector3(nrm.x, nrm.y, 0).multiplyScalar(LINE_W / 2);
    const [a, b] = ends.map(([x, y]) => new THREE.Vector3(x, y, 0));
    return [a.clone().sub(o), b.clone().sub(o), b.clone().add(o), a.clone().add(o)];
  };
  const setPlane = (offset) => {
    const poly = !hasPlane ? [] : flat ? lineRibbon(dW + offset) : planeBoxPolygon(nrm, dW + offset, lo, hi);
    poly.forEach((v, i) => { pPos.setXYZ(i, v.x, v.y, v.z); rimPos.setXYZ(i, v.x, v.y, v.z); });
    pPos.needsUpdate = rimPos.needsUpdate = true;
    pGeo.setDrawRange(0, Math.max(0, poly.length - 2) * 3);
    rimGeo.setDrawRange(0, poly.length);
    pGeo.computeBoundingSphere(); rimGeo.computeBoundingSphere();
    return poly;
  };
  const finalPoly = setPlane(0);
  const planeMid = finalPoly.length
    ? finalPoly.reduce((s, v) => s.add(v), new THREE.Vector3()).divideScalar(finalPoly.length)
    : null;
  // arrow towards the disease side (direction of coef)
  const arrow = planeMid ? diseaseArrow(planeMid, nrm) : new THREE.Group();
  web.add(arrow);
  // dashed drop line from the patient to the plane
  const foot = youPos.clone().addScaledVector(nrm, -(nrm.dot(youPos) - dW));
  const drop = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints([youPos, foot]),
    new THREE.LineDashedMaterial({ color: 0xffffff, dashSize: 0.07, gapSize: 0.05, transparent: true, opacity: 0.8 })
  );
  drop.computeLineDistances();
  web.add(drop);

  // ---------------- RBF surface: decision value 0 over the exported grid ----------------
  const V = rbf.values;
  let vMax = -Infinity, vMin = Infinity;
  for (const x of V) { if (x > vMax) vMax = x; if (x < vMin) vMin = x; }
  const hasCrossing = vMax > 0 && vMin < 0;
  const SKIN_O = 0.3, WIRE_O = 0.14;
  const rbfSkin = skinMaterials(SKIN_O);
  const wireMat = new THREE.MeshBasicMaterial({ color: 0x7dd3fc, wireframe: true, transparent: true, opacity: WIRE_O, depthWrite: false });
  const surf = new THREE.Group();
  web.add(surf);
  const levelCache = new Map();
  // Growth: the level set sweeps from 85% of the peak decision value down to 0 (the real boundary),
  // so the disease region inflates until it reaches the model's boundary. Only level 0 is captioned.
  const peak = hasCrossing ? 0.85 * vMax : 0;
  const CONTOUR_O = 0.9;
  const contourMat = new THREE.MeshBasicMaterial({ color: PALETTE.classical, transparent: true, opacity: CONTOUR_O, side: THREE.DoubleSide, depthWrite: false });
  const contourGeo = (level) => { // 2 qubits: the level line of the G×G grid as a flat ribbon
    const segs = contourSegments(V, G, axis, level).map((sg) => sg.map((x) => (x - C0) * S));
    return segs.length ? ribbonGeometry(segs, LINE_W) : null;
  };
  const levelMeshes = (step) => {
    if (!levelCache.has(step)) {
      const level = peak * (1 - step / LEVEL_STEPS);
      const geo = flat ? contourGeo(level) : surfaceNets(V, G, axis, level, toWorld);
      let grp = null;
      if (geo && flat) {
        grp = new THREE.Group();
        grp.add(new THREE.Mesh(geo, contourMat));
        grp.visible = false;
        surf.add(grp);
      } else if (geo) {
        grp = new THREE.Group();
        grp.add(new THREE.Mesh(geo, rbfSkin.front), new THREE.Mesh(geo, rbfSkin.back), new THREE.Mesh(geo, wireMat));
        grp.visible = false;
        surf.add(grp);
      }
      levelCache.set(step, grp);
    }
    return levelCache.get(step);
  };
  if (hasCrossing) levelMeshes(LEVEL_STEPS); // the real boundary, built up front
  let shownStep = -1;
  const showStep = (step) => {
    if (step === shownStep) return;
    if (shownStep >= 0 && levelCache.get(shownStep)) levelCache.get(shownStep).visible = false;
    const g = levelMeshes(step);
    if (g) g.visible = true;
    shownStep = step;
  };
  // disease-side arrow on the real boundary, at the surface point nearest the patient,
  // along the gradient of the decision value (central differences on the grid)
  let rbfArrow = new THREE.Group();
  const zeroGrp = hasCrossing ? levelCache.get(LEVEL_STEPS) : null;
  if (zeroGrp) {
    const pa = zeroGrp.children[0].geometry.attributes.position;
    const q = new THREE.Vector3(), best = new THREE.Vector3();
    let bd = Infinity;
    for (let i = 0; i < pa.count; i++) {
      q.fromBufferAttribute(pa, i);
      const d = q.distanceToSquared(youPos);
      if (d < bd) { bd = d; best.copy(q); }
    }
    const h = axis[1] - axis[0];
    const gi = [best.x, best.y, best.z].map((w) => Math.min(G - 2, Math.max(1, Math.round((w / S + C0 - rbf.min) / h))));
    let grad;
    if (flat) { // G×G grid: values[i*G + j]
      const at2 = (i, j) => V[i * G + j];
      const [i, j] = gi;
      grad = new THREE.Vector3(at2(i + 1, j) - at2(i - 1, j), at2(i, j + 1) - at2(i, j - 1), 0);
    } else {
      const at = (i, j, k) => V[(i * G + j) * G + k];
      const [i, j, k] = gi;
      grad = new THREE.Vector3(
        at(i + 1, j, k) - at(i - 1, j, k), at(i, j + 1, k) - at(i, j - 1, k), at(i, j, k + 1) - at(i, j, k - 1)
      );
    }
    if (grad.lengthSq() > 1e-12) rbfArrow = diseaseArrow(best, grad.normalize());
  }
  web.add(rbfArrow);

  // ---------------- captions ----------------
  const vLin = verdictFor(p.models.linear);
  const vRbf = verdictFor(p.models.rbf);
  const sickSpan = (txt) => `<span style="color:${hex(PALETTE.sick)}">${txt}</span>`;
  const okSpan = (txt) => `<span style="color:${hex(PALETTE.healthy)}">${txt}</span>`;
  const cyan = (txt) => `<span style="color:${hex(PALETTE.classical)}">${txt}</span>`;
  const verdictWord = (v) => (v.disease ? sickSpan("DISEASE") : okSpan("HEALTHY"));
  const borderline = (v) => (v.borderline ? " · <b>borderline</b>: the probability estimate and the decision disagree; the verdict follows the decision" : "");
  // side of the real (all-qubit) linear boundary; 4 qubits: also the side in the drawn x₄ slice
  const linSide = coef.reduce((s, w, k) => s + w * p.angles[k], 0) + intercept > 0;
  const drawnSide = lc.reduce((s, w, k) => s + w * p.angles[k], 0) + sec.intercept > 0;
  const sliceWarn = nQ > 3 && drawnSide !== linSide
    ? ` · in this ${sub4} slice #${pi} appears on the other side, because its own ${sub4} = ${n2(p.angles[3])} differs from the slice value ${n2(x4)}`
    : "";
  const sideTxt = linSide === vLin.disease
    ? ` · #${pi} sits on the ${linSide ? sickSpan("disease") : okSpan("healthy")} side of the cut${sliceWarn}`
    : "";
  const fmt = (x) => (x < 0 ? "−" : "") + n2(Math.abs(x));
  const eq = `${fmt(coef[0])}·x₁${coef.slice(1).map((w, k) => ` ${w < 0 ? "−" : "+"} ${n2(Math.abs(w))}·${xName(k + 1)}`).join("")} ${intercept < 0 ? "−" : "+"} ${n2(Math.abs(intercept))} = 0`;
  const allSick = vMin > 0;
  const sliceTxt = nQ > 3 ? `showing a 3D slice at ${sub4} = ${n2(x4)} (the training average); the 4th dimension isn't drawn` : "";
  const where = flat ? "plane" : nQ > 3 ? "slice" : "region";
  const cap = {
    intro: flat
      ? `<b>The classical doctors</b> · 2 qubits → 2D: ${nTrain} training patients placed by their 2 angles on a flat plane (${okSpan("healthy")} / ${sickSpan("disease")}) · ${youP} in white · two classical SVMs each draw a boundary across this plane`
      : nQ > 3
        ? `<b>The classical doctors</b> · ${nTrain} training patients placed by their first 3 of ${nQ} angles (${okSpan("healthy")} / ${sickSpan("disease")}) · ${sliceTxt} · ${youP} in white (its own ${sub4} = ${n2(p.angles[3])}) · two classical SVMs each draw a boundary`
        : `<b>The classical doctors</b> · ${nTrain} training patients placed by their 3 angles (${okSpan("healthy")} / ${sickSpan("disease")}) · ${youP} in white · two classical SVMs each draw a boundary through this space`,
    linear: hasPlane
      ? flat
        ? `<b>Doctor 1: ${cyan("Linear SVM")}</b> · one straight line across the plane: ${eq} · the arrow points to the side it calls disease · 2 qubits → 2D`
        : nQ > 3
          ? `<b>Doctor 1: ${cyan("Linear SVM")}</b> · one flat cut through the ${nQ}D space: ${eq} · drawn as its 3D section at ${sub4} = ${n2(x4)} · the arrow points to the side it calls disease`
          : `<b>Doctor 1: ${cyan("Linear SVM")}</b> · one flat cut through the space: ${eq} · the arrow points to the side it calls disease`
      : `<b>Doctor 1: ${cyan("Linear SVM")}</b> · ${nQ > 3 ? `in this ${sub4} slice its cut has no x₁–x₃ part, so none is drawn` : "its weights are all zero, so it draws no cut and labels every patient the same way"}`,
    linVerdict: `<b>${cyan("Linear SVM")} verdict: ${verdictWord(vLin)}</b> · disease probability <b>${pct(p.models.linear.prob)}%</b>${sideTxt}${borderline(vLin)}`,
    rbf: hasCrossing
      ? flat
        ? `<b>Doctor 2: ${cyan("RBF SVM")}</b> · bends its boundary around the clusters · the curve grows until the model's decision value reaches 0 (evaluated on a ${G}×${G} grid) · faint straight line: the linear cut for comparison · 2 qubits → 2D`
        : nQ > 3
          ? `<b>Doctor 2: ${cyan("RBF SVM")}</b> · bends its boundary around the clusters · the surface grows until the model's decision value reaches 0 (evaluated on a ${G}×${G}×${G} grid at ${sub4} = ${n2(x4)}) · ${sliceTxt} · faint flat sheet: the linear cut for comparison`
          : `<b>Doctor 2: ${cyan("RBF SVM")}</b> · bends its boundary around the clusters · the surface grows until the model's decision value reaches 0 (evaluated on a ${G}×${G}×${G} grid) · faint flat sheet: the linear cut for comparison`
      : `<b>Doctor 2: ${cyan("RBF SVM")}</b> · its decision value never crosses 0 inside this ${where}, so it labels the whole ${where} ${allSick ? sickSpan("disease") : okSpan("healthy")}`,
    rbfVerdict: `<b>${cyan("RBF SVM")} verdict: ${verdictWord(vRbf)}</b> · disease probability <b>${pct(p.models.rbf.prob)}%</b>${borderline(vRbf)} · Linear SVM said ${verdictWord(vLin)} (${pct(p.models.linear.prob)}%)${rbfArrow.children.length ? " · arrow: the side RBF calls disease" : ""}`,
  };
  const tint = (v) => (v.disease ? PALETTE.sick : PALETTE.healthy);

  return {
    title: "Classical doctors",
    legend: legend2(nQ),
    scene,
    cameraPos: flat ? [0.9, 2.3, 6.6] : [0.6, 1.7, 7.0], // 2 qubits: tilted to show the plane
    duration,
    stops: [T.linVerdict + 0.45, T.rbfVerdict + 0.45],
    update(t) {
      const pIn = ease(seg(t, ...T.webIn));
      setOpacity(box, pIn);
      setOpacity(dots, pIn);
      web.scale.setScalar(0.8 + 0.2 * pIn);

      const pYou = ease(seg(t, ...T.you));
      you.visible = pYou > 0;
      you.scale.setScalar(Math.max(0.001, pYou * (1 + 0.1 * Math.sin(t * 5))));

      // linear plane slides in along its normal, then fades to a ghost
      const pPlane = ease(seg(t, ...T.planeIn));
      const pOut = ease(seg(t, ...T.planeOut));
      if (pPlane > 0 && hasPlane) {
        setPlane(SLIDE * (1 - pPlane));
        plane.visible = true;
        setOpacity(plane, Math.min(1, pPlane * 2.5) * (1 - (1 - GHOST) * pOut));
      } else plane.visible = false;
      const pArrow = ease(seg(t, T.planeIn[1] - 0.2, T.planeIn[1] + 0.3)) * (1 - pOut);
      setOpacity(arrow, pArrow);
      drop.material.opacity = 0.8 * pArrow;
      drop.visible = hasPlane && pArrow > 0.01;

      // RBF surface grows (cached level sets), real boundary from rbfIn end
      const pR = seg(t, ...T.rbfIn);
      if (hasCrossing && pR > 0) {
        showStep(Math.round(ease(pR) * LEVEL_STEPS));
        surf.visible = true;
        const o = Math.min(1, pR * 3);
        rbfSkin.front.opacity = rbfSkin.back.opacity = SKIN_O * o;
        wireMat.opacity = WIRE_O * o;
        contourMat.opacity = CONTOUR_O * o;
      } else surf.visible = false;

      setOpacity(rbfArrow, ease(seg(t, T.rbfVerdict - 0.3, T.rbfVerdict + 0.3)));

      // verdict halo around the patient
      let haloCol = null, haloO = 0;
      if (t >= T.rbfVerdict) { haloCol = tint(vRbf); haloO = ease(seg(t, T.rbfVerdict, T.rbfVerdict + 0.4)); }
      else if (t >= T.linVerdict && t < T.planeOut[0] + 0.3) { haloCol = tint(vLin); haloO = ease(seg(t, T.linVerdict, T.linVerdict + 0.4)) * (1 - seg(t, T.planeOut[0], T.planeOut[0] + 0.3)); }
      if (haloCol !== null) { halo.material.color.setHex(haloCol); ring.material.color.setHex(haloCol); }
      halo.material.opacity = haloO * (0.85 + 0.15 * Math.sin(t * 6));
      ring.material.opacity = haloO;

      if (t < T.planeIn[0]) return cap.intro;
      if (t < T.linVerdict) return cap.linear;
      if (t < T.rbfIn[0]) return cap.linVerdict;
      if (t < T.rbfVerdict) return cap.rbf;
      return hasCrossing ? cap.rbfVerdict : `${cap.rbf}<br>${cap.rbfVerdict}`;
    },
    dispose() {
      disposeTree(scene);
    },
  };
}
