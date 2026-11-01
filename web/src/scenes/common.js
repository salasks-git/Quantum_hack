import * as THREE from "three";

export const PALETTE = {
  bg: 0x070b16, quantum: 0xa78bfa, quantum2: 0xf0abfc,
  classical: 0x38bdf8, healthy: 0x4ade80, sick: 0xfb7185,
};

export const ease = (t) => { t = Math.min(1, Math.max(0, t)); return t * t * (3 - 2 * t); };
export const seg = (p, a, b) => Math.min(1, Math.max(0, (p - a) / (b - a)));

export function firstLayerState(angle) {
  return { theta: Math.PI / 2, phi: 2 * angle };
}

export function blochSphere(radius = 1) {
  const g = new THREE.Group();
  const shell = new THREE.Mesh(
    new THREE.SphereGeometry(radius, 32, 24),
    new THREE.MeshBasicMaterial({ color: PALETTE.quantum, wireframe: true, transparent: true, opacity: 0.15 })
  );
  g.add(shell);
  const inner = new THREE.Mesh(
    new THREE.SphereGeometry(radius * 0.98, 32, 24),
    new THREE.MeshPhongMaterial({ color: 0x1e1b4b, transparent: true, opacity: 0.35, shininess: 80, depthWrite: false })
  );
  g.add(inner);
  const ring = new THREE.Mesh(
    new THREE.TorusGeometry(radius, radius * 0.008, 6, 64),
    new THREE.MeshBasicMaterial({ color: PALETTE.quantum, transparent: true, opacity: 0.5 })
  );
  ring.rotation.x = Math.PI / 2;
  g.add(ring);
  const axis = new THREE.Mesh(
    new THREE.CylinderGeometry(radius * 0.005, radius * 0.005, radius * 2.2, 6),
    new THREE.MeshBasicMaterial({ color: PALETTE.quantum, transparent: true, opacity: 0.35 })
  );
  g.add(axis);
  const arrowGeo = new THREE.ConeGeometry(radius * 0.06, radius * 0.2, 12);
  const arrow = new THREE.Group();
  const shaft = new THREE.Mesh(
    new THREE.CylinderGeometry(radius * 0.015, radius * 0.015, radius * 0.9, 8),
    new THREE.MeshBasicMaterial({ color: PALETTE.quantum2 })
  );
  shaft.position.y = radius * 0.45;
  const tip = new THREE.Mesh(arrowGeo, new THREE.MeshBasicMaterial({ color: PALETTE.quantum2 }));
  tip.position.y = radius * 0.95;
  arrow.add(shaft, tip);
  g.add(arrow);
  const glow = glowSprite(PALETTE.quantum2, radius * 0.45);
  g.add(glow);
  g.userData.set = (theta, phi) => {
    // Bloch convention: z up (theta from +z), mapped to three.js y-up
    const dir = new THREE.Vector3(
      Math.sin(theta) * Math.cos(phi), Math.cos(theta), -Math.sin(theta) * Math.sin(phi)
    ).normalize();
    arrow.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
    glow.position.copy(dir).multiplyScalar(radius);
  };
  g.userData.set(0, 0);
  return g;
}

/** Point on a Bloch sphere of given radius (same mapping as blochSphere's arrow). */
export function blochPoint(theta, phi, radius = 1) {
  return new THREE.Vector3(
    Math.sin(theta) * Math.cos(phi), Math.cos(theta), -Math.sin(theta) * Math.sin(phi)
  ).multiplyScalar(radius);
}

const FONT = "system-ui, -apple-system, Segoe UI, sans-serif";

/** Camera-facing text label. `height` is the world height of one text line. Lines split on "\n". */
export function textSprite(text, { color = "#e5e7eb", height = 0.2, weight = 600, bg = null, align = "center" } = {}) {
  const px = 64, pad = 18, lines = String(text).split("\n");
  const c = document.createElement("canvas");
  const g = c.getContext("2d");
  g.font = `${weight} ${px}px ${FONT}`;
  const w = Math.ceil(Math.max(...lines.map((l) => g.measureText(l).width))) + pad * 2;
  const lh = px * 1.2;
  c.width = w;
  c.height = Math.ceil(lh * lines.length + pad);
  if (bg) {
    g.fillStyle = bg;
    const r = 22;
    g.beginPath();
    g.roundRect ? g.roundRect(0, 0, c.width, c.height, r) : g.rect(0, 0, c.width, c.height);
    g.fill();
  }
  g.font = `${weight} ${px}px ${FONT}`;
  g.fillStyle = color;
  g.textBaseline = "middle";
  g.textAlign = align;
  const x = align === "center" ? c.width / 2 : align === "right" ? c.width - pad : pad;
  lines.forEach((l, i) => g.fillText(l, x, pad / 2 + lh * (i + 0.5)));
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 4;
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false, depthTest: false }));
  const worldH = (height * c.height) / lh;
  s.scale.set((worldH * c.width) / c.height, worldH, 1);
  s.renderOrder = 10;
  return s;
}

/** Soft additive glow (radial gradient sprite) of the given world diameter. */
export function glowSprite(color, size = 1, strength = 1) {
  const c = document.createElement("canvas");
  c.width = c.height = 128;
  const g = c.getContext("2d");
  const grad = g.createRadialGradient(64, 64, 0, 64, 64, 64);
  const col = new THREE.Color(color);
  const rgb = `${Math.round(col.r * 255)},${Math.round(col.g * 255)},${Math.round(col.b * 255)}`;
  grad.addColorStop(0, `rgba(${rgb},${0.9 * strength})`);
  grad.addColorStop(0.35, `rgba(${rgb},${0.35 * strength})`);
  grad.addColorStop(1, `rgba(${rgb},0)`);
  g.fillStyle = grad;
  g.fillRect(0, 0, 128, 128);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  const s = new THREE.Sprite(new THREE.SpriteMaterial({
    map: tex, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
  }));
  s.scale.set(size, size, 1);
  return s;
}

/** Fade every material under `obj` to `o` times its original opacity. */
export function setOpacity(obj, o) {
  obj.traverse((m) => {
    if (!m.material) return;
    const mats = Array.isArray(m.material) ? m.material : [m.material];
    mats.forEach((mat) => {
      if (mat.userData.baseOp === undefined) mat.userData.baseOp = mat.opacity;
      mat.transparent = true;
      mat.opacity = mat.userData.baseOp * o;
    });
  });
  obj.visible = o > 0.005;
}

export function disposeTree(obj) {
  obj.traverse((o) => {
    if (o.geometry) o.geometry.dispose();
    if (o.material) {
      const mats = Array.isArray(o.material) ? o.material : [o.material];
      mats.forEach((m) => {
        for (const k of Object.keys(m)) if (m[k] && m[k].isTexture) m[k].dispose();
        m.dispose();
      });
    }
  });
}

export function addLights(scene) {
  scene.add(new THREE.AmbientLight(0xffffff, 0.6));
  const d = new THREE.DirectionalLight(0xffffff, 0.8);
  d.position.set(3, 5, 4);
  scene.add(d);
}

// ---------------- n-qubit helpers (2, 3 or 4 qubits; pure, unit-tested) ----------------

export const SUBSCRIPTS = ["₁", "₂", "₃", "₄", "₅", "₆", "₇", "₈"];
/** "x₁", "x₂", … for 0-based dimension k. */
export const xName = (k) => `x${SUBSCRIPTS[k]}`;

/**
 * Centres for a row of n items (e.g. Bloch spheres) spaced `gap` apart, with item radius `radius`.
 * The 3-item row is the reference width; rows that would be wider are shrunk uniformly
 * (`scale` < 1 applies to gap, radius and labels). n <= 3 keeps scale 1 (3 -> exactly the old layout).
 */
export function rowLayout(n, gap, radius) {
  const ref = 2 * gap + 2 * radius;
  const scale = Math.min(1, ref / ((n - 1) * gap + 2 * radius));
  const xs = Array.from({ length: n }, (_, k) => (k - (n - 1) / 2) * gap * scale);
  return { xs, scale };
}

/**
 * Angles -> the 3 coordinates that are drawn: the first <= 3 angles; a missing 3rd (n = 2)
 * is filled with `fill` (the centre of the box, i.e. world z = 0 -> a flat layout). Extra dims are dropped.
 */
export function pos3(angles, fill = Math.PI / 2) {
  return [0, 1, 2].map((k) => (k < angles.length ? angles[k] : fill));
}

/**
 * Section of the hyperplane coef·x + intercept = 0 at fixed coordinates `fixed` = {dim: value}:
 * the fixed terms fold into the intercept, the remaining coefficients keep their order.
 * e.g. 4D -> 3D slice at x₄ = v: sectionPlane(coef, b, {3: v}) = {coef: coef[0..2], intercept: b + coef[3]·v}.
 */
export function sectionPlane(coef, intercept, fixed = {}) {
  let b = intercept;
  const c = [];
  coef.forEach((w, d) => { if (d in fixed) b += w * fixed[d]; else c.push(w); });
  return { coef: c, intercept: b };
}

/**
 * Marching squares over a G×G grid: values[i*G + j] is the value at (axis[i], axis[j]).
 * Returns segments [x1, y1, x2, y2] (axis units) where the values cross `level`.
 * Saddle cells are resolved with the cell-centre average.
 */
export function contourSegments(values, G, axis, level = 0) {
  const segs = [];
  const v = (i, j) => values[i * G + j] - level;
  for (let i = 0; i < G - 1; i++) for (let j = 0; j < G - 1; j++) {
    // corners counter-clockwise: c0 (i,j), c1 (i+1,j), c2 (i+1,j+1), c3 (i,j+1)
    const ci = [[i, j], [i + 1, j], [i + 1, j + 1], [i, j + 1]];
    const cv = ci.map(([a, b]) => v(a, b));
    const pts = []; // crossing point per edge e_k = c_k -> c_{k+1}, or null
    for (let k = 0; k < 4; k++) {
      const a = cv[k], b = cv[(k + 1) % 4];
      if ((a > 0) === (b > 0)) { pts.push(null); continue; }
      const t = a / (a - b);
      const [ia, ja] = ci[k], [ib, jb] = ci[(k + 1) % 4];
      pts.push([axis[ia] + (axis[ib] - axis[ia]) * t, axis[ja] + (axis[jb] - axis[ja]) * t]);
    }
    const hit = pts.map((p, k) => (p ? k : -1)).filter((k) => k >= 0);
    if (hit.length === 2) segs.push([...pts[hit[0]], ...pts[hit[1]]]);
    else if (hit.length === 4) {
      const centre = (cv[0] + cv[1] + cv[2] + cv[3]) / 4;
      const pairs = (centre > 0) === (cv[0] > 0) ? [[0, 1], [2, 3]] : [[3, 0], [1, 2]];
      for (const [a, b] of pairs) segs.push([...pts[a], ...pts[b]]);
    }
  }
  return segs;
}

/**
 * The line n·p = d clipped to the square [lo, hi]² -> [[x, y], [x, y]], or null if it misses.
 * (2D counterpart of the plane/box section used for 3 and 4 qubits.)
 */
export function clipLineToSquare(n, d, lo, hi) {
  const C = [[lo, lo], [hi, lo], [hi, hi], [lo, hi]];
  const pts = [];
  for (let k = 0; k < 4; k++) {
    const a = C[k], b = C[(k + 1) % 4];
    const da = n[0] * a[0] + n[1] * a[1] - d, db = n[0] * b[0] + n[1] * b[1] - d;
    if (da === 0) pts.push(a);
    if ((da > 0) === (db > 0) || da === 0 || db === 0) continue;
    const t = da / (da - db);
    pts.push([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]);
  }
  const uniq = pts.filter((p, i) => pts.findIndex((q) => Math.hypot(q[0] - p[0], q[1] - p[1]) < 1e-9) === i);
  return uniq.length >= 2 ? [uniq[0], uniq[1]] : null;
}
