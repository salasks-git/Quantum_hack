import * as THREE from "three";
import { PALETTE, addLights, disposeTree } from "./common.js";

export function makePlaceholder(n) {
  return function createScene(ctx) {
    const scene = new THREE.Scene();
    addLights(scene);
    const mesh = new THREE.Mesh(
      new THREE.IcosahedronGeometry(1, 1),
      new THREE.MeshStandardMaterial({ color: PALETTE.quantum, wireframe: true })
    );
    scene.add(mesh);
    return {
      title: `Scene ${n}`,
      scene,
      cameraPos: [0, 1, 4],
      duration: 3,
      update(t) {
        mesh.rotation.y = t;
        return `Scene ${n} coming soon`;
      },
      dispose() { disposeTree(scene); },
    };
  };
}

export default makePlaceholder(0);
