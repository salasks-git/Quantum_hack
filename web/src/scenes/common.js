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
