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
  g.userData.set = (theta, phi) => {
    // Bloch convention: z up (theta from +z), mapped to three.js y-up
    const dir = new THREE.Vector3(
      Math.sin(theta) * Math.cos(phi), Math.cos(theta), -Math.sin(theta) * Math.sin(phi)
    ).normalize();
    arrow.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
  };
  g.userData.set(0, 0);
  return g;
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
