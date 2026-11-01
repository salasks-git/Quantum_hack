// three.js helpers shared by the Sandwich (step 9) and CKTA + Elite leaderboard (step 10) scenes.
// COPIED (not moved) from sceneQuhama.js (dotTexture, layerBox, gauge, the particle-flow system) and
// eliteParts.js (label, line, slab, weightRing) so those files can be trimmed independently.
import * as THREE from "three";
import { PALETTE, ease, seg, textSprite, glowSprite, setOpacity } from "./common.js";
import { BAR_STAGGER, BAR_GROW } from "../eliteScenes.js";

export const GOLD = 0xfbbf24;
export const LATENT = 0xe9d5ff;
export const AMBER = "#fbbf24";
export const SUB = ["₁", "₂", "₃", "₄", "₅", "₆", "₇", "₈"];
export const hex = (c) => "#" + c.toString(16).padStart(6, "0");
export const n2 = (v) => v.toFixed(2).replace("-", "−");
export const sgn = (v) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(2);
export const span = (c, x) => `<b style="color:${hex(c)}">${x}</b>`;
export const muted = (x) => `<span style="color:#94a3b8">${x}</span>`;

/** Camera-facing label at (x, y) in `parent`. */
export function label(parent, text, x, y, opts = {}) {
  const s = textSprite(text, { height: 0.15, ...opts });
  s.position.set(x, y, 0.05);
  parent.add(s);
  return s;
}

export function line(parent, a, b, color, opacity = 0.45) {
  const l = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints([a, b]),
    new THREE.LineBasicMaterial({ color, transparent: true, opacity, depthWrite: false })
  );
  parent.add(l);
  return l;
}

/** Thin slab (a network layer) with outline. */
export function slab(w, h, d, color, op = 0.22) {
  const g = new THREE.Group();
  const box = new THREE.BoxGeometry(w, h, d);
  g.add(new THREE.Mesh(box, new THREE.MeshPhongMaterial({ color, emissive: color, emissiveIntensity: 0.25, transparent: true, opacity: op, depthWrite: false })));
  g.add(new THREE.LineSegments(new THREE.EdgesGeometry(box), new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.8 })));
  return g;
}

/** Split ring: cyan arc = classical weight, violet arc = quantum weight. */
export function weightRing(wc, wq, r0, r1) {
  const g = new THREE.Group();
  const mk = (start, len, color) => new THREE.Mesh(
    new THREE.RingGeometry(r0, r1, 64, 1, start, len),
    new THREE.MeshBasicMaterial({ color, transparent: true, side: THREE.DoubleSide, depthWrite: false })
  );
  g.add(mk(Math.PI / 2, 2 * Math.PI * wc, PALETTE.classical), mk(Math.PI / 2 + 2 * Math.PI * wc, 2 * Math.PI * wq, PALETTE.quantum));
  return g;
}

export function dotTexture() {
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

/** Translucent box with glowing edges and a grid of unit dots (a neural-network layer). */
export function layerBox(w, h, color, rows, cols, depth = 0.45) {
  const g = new THREE.Group();
  const box = new THREE.Mesh(
    new THREE.BoxGeometry(w, h, depth),
    new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.07, depthWrite: false })
  );
  g.add(box);
  g.add(new THREE.LineSegments(
    new THREE.EdgesGeometry(box.geometry),
    new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.6 })
  ));
  const units = [];
  const geo = new THREE.SphereGeometry(0.05, 12, 8);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const m = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 1 }));
      m.position.set((c - (cols - 1) / 2) * (w / (cols + 0.6)), ((rows - 1) / 2 - r) * (h / (rows + 0.6)), 0);
      g.add(m);
      units.push(m);
    }
  }
  return { group: g, units };
}

/** Vertical probability gauge: open frame, fill from the bottom, a ring at 50%. */
export function gauge(h, r, color) {
  const g = new THREE.Group();
  const frame = new THREE.Mesh(
    new THREE.CylinderGeometry(r, r, h, 28, 1, true),
    new THREE.MeshBasicMaterial({ color: 0x94a3b8, transparent: true, opacity: 0.16, side: THREE.DoubleSide, depthWrite: false })
  );
  g.add(frame);
  const fillGeo = new THREE.CylinderGeometry(r * 0.82, r * 0.82, 1, 28);
  fillGeo.translate(0, 0.5, 0);
  const mat = new THREE.MeshPhongMaterial({ color, emissive: color, emissiveIntensity: 0.45, transparent: true, opacity: 0.95 });
  const fill = new THREE.Mesh(fillGeo, mat);
  fill.position.y = -h / 2;
  fill.scale.y = 0.001;
  g.add(fill);
  const tick = new THREE.Mesh(
    new THREE.TorusGeometry(r * 1.12, 0.012, 6, 40),
    new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.55 })
  );
  tick.rotation.x = Math.PI / 2;
  g.add(tick);
  const glow = glowSprite(color, r * 5, 0.8);
  g.add(glow);
  return {
    group: g, mat,
    set(p) {
      fill.scale.y = Math.max(0.001, p * h);
      glow.position.y = -h / 2 + p * h;
      glow.visible = p > 0.002;
    },
  };
}

/**
 * Particle flows along curved edges (from sceneQuhama.js). flows: [{a, b, win: [t0, t1], color, n, bendY?}].
 * Returns animate(t, sk) where sk scales the edge opacity (0..1).
 */
export function particleFlows(root, flows, travel = 0.7) {
  const edges = flows.map((f, i) => {
    const mid = f.a.clone().lerp(f.b, 0.5);
    mid.z += 0.3 + 0.1 * Math.sin(i * 1.7);
    if (f.bendY) mid.y += f.bendY;
    f.curve = new THREE.QuadraticBezierCurve3(f.a, mid, f.b);
    const l = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(f.curve.getPoints(20)),
      new THREE.LineBasicMaterial({ color: f.color, transparent: true, opacity: 0.08, depthWrite: false })
    );
    root.add(l);
    return l;
  });
  const parts = [];
  flows.forEach((f, fi) => {
    for (let i = 0; i < f.n; i++) {
      const emit = Math.max(0.01, f.win[1] - f.win[0] - travel);
      parts.push({ f: fi, te: f.win[0] + (f.n === 1 ? emit / 2 : (i / (f.n - 1)) * emit) });
    }
  });
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(Math.max(1, parts.length) * 3), 3));
  geo.setAttribute("color", new THREE.BufferAttribute(new Float32Array(Math.max(1, parts.length) * 3), 3));
  const pts = new THREE.Points(geo, new THREE.PointsMaterial({
    size: 0.17, map: dotTexture(), vertexColors: true, transparent: true, depthWrite: false,
    blending: THREE.AdditiveBlending, sizeAttenuation: true,
  }));
  pts.frustumCulled = false;
  root.add(pts);
  const flowCol = flows.map((f) => new THREE.Color(f.color));
  const P = new THREE.Vector3();
  return function animate(t, sk = 1) {
    edges.forEach((l, i) => {
      const w = flows[i].win;
      const active = seg(t, w[0], w[0] + 0.3) * (1 - seg(t, w[1], w[1] + 0.6));
      const after = t > w[1] ? 0.22 : 0.08;
      l.material.opacity = sk * Math.max(after, 0.55 * active);
      l.visible = sk > 0.005;
    });
    const pos = geo.attributes.position, col = geo.attributes.color;
    parts.forEach((pt, i) => {
      const u = (t - pt.te) / travel;
      if (u <= 0 || u >= 1 || sk <= 0.005) { pos.setXYZ(i, 0, -100, 0); col.setXYZ(i, 0, 0, 0); return; }
      flows[pt.f].curve.getPoint(u, P);
      pos.setXYZ(i, P.x, P.y, P.z);
      const fade = Math.min(1, u * 6, (1 - u) * 5) * sk;
      const c = flowCol[pt.f];
      col.setXYZ(i, c.r * fade, c.g * fade, c.b * fade);
    });
    pos.needsUpdate = true;
    col.needsUpdate = true;
  };
}

/**
 * 3D bar chart of 0..1 values (metrics or the leaderboard's F2). bars: [{name, sub?, value, text, color,
 * edge?, warn?, highlight?, lead?}]. Bars grow one after another from `t0` (BAR_STAGGER apart, BAR_GROW each);
 * warnings and the leader glow appear once all bars have grown. Returns { group, animate(t, on) }.
 */
export function barChart(parent, bars, { x0 = 0, gap = 1.2, y = 0, h = 2.6, w = 0.75, t0 = 0, title = "", titleColor = "#ffffff" } = {}) {
  const group = new THREE.Group();
  group.position.set(x0, y, 0);
  parent.add(group);
  const xs = bars.map((_, i) => (i - (bars.length - 1) / 2) * gap);
  const xl = (xs[0] ?? 0) - gap / 2 - 0.2, xr = (xs[xs.length - 1] ?? 0) + gap / 2 + 0.2;
  line(group, new THREE.Vector3(xl, 0, 0), new THREE.Vector3(xr, 0, 0), 0x94a3b8, 0.6);
  line(group, new THREE.Vector3(xl, h, 0), new THREE.Vector3(xr, h, 0), 0x94a3b8, 0.18);
  line(group, new THREE.Vector3(xl, h / 2, 0), new THREE.Vector3(xr, h / 2, 0), 0x94a3b8, 0.1);
  label(group, "1", xl - 0.15, h, { height: 0.12, color: "#94a3b8", weight: 500 });
  label(group, "0.5", xl - 0.22, h / 2, { height: 0.12, color: "#94a3b8", weight: 500 });
  if (title) label(group, title, 0, h + 0.75, { height: 0.2, color: titleColor });
  const end = t0 + Math.max(0, bars.length - 1) * BAR_STAGGER + BAR_GROW;
  const items = bars.map((b, i) => {
    const geo = new THREE.BoxGeometry(w, 1, 0.55);
    geo.translate(0, 0.5, 0);
    const mesh = new THREE.Mesh(geo, new THREE.MeshPhongMaterial({ color: b.color, emissive: b.color, emissiveIntensity: b.highlight ? 0.55 : 0.3, transparent: true, opacity: 0.9 }));
    mesh.position.set(xs[i], 0, 0);
    mesh.scale.y = 0.001;
    group.add(mesh);
    if (b.edge !== undefined) mesh.add(new THREE.LineSegments(new THREE.EdgesGeometry(geo), new THREE.LineBasicMaterial({ color: b.edge, transparent: true, opacity: 0.9 })));
    const val = label(group, b.text, xs[i], 0.2, { height: 0.19, color: "#ffffff" });
    label(group, b.name, xs[i], -0.24, { height: 0.17, color: b.nameColor || "#e5e7eb" });
    if (b.sub) label(group, b.sub, xs[i], -0.46, { height: 0.12, color: "#94a3b8", weight: 500 });
    const warn = b.warn ? label(group, b.warn, xs[i], b.sub ? -0.66 : -0.46, { height: 0.12, color: AMBER, weight: 600 }) : null;
    let glow = null, lead = null;
    if (b.lead) {
      glow = glowSprite(GOLD, 1.5, 0.8);
      group.add(glow);
      lead = label(group, b.lead, xs[i], 0.5, { height: 0.15, color: hex(GOLD) });
    }
    const hh = Math.max(0.001, Math.max(0, Math.min(1, b.value ?? 0)) * h);
    return { i, mesh, val, warn, glow, lead, hh };
  });
  const animate = (t, on = 1) => {
    setOpacity(group, on);
    const after = ease(seg(t, end, end + 0.6));
    items.forEach((b) => {
      const st = t0 + b.i * BAR_STAGGER;
      const g = ease(seg(t, st, st + BAR_GROW));
      b.mesh.scale.y = Math.max(0.001, b.hh * g);
      b.val.position.y = b.hh * g + 0.2;
      setOpacity(b.val, on * g);
      if (b.warn) setOpacity(b.warn, on * after);
      if (b.glow) {
        b.glow.position.set(xs[b.i], b.hh, 0.1);
        setOpacity(b.glow, on * after * (0.75 + 0.25 * Math.sin(t * 3)));
        b.lead.position.y = b.hh + 0.48;
        setOpacity(b.lead, on * after);
      }
    });
  };
  return { group, animate, end };
}
