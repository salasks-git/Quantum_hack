import { describe, it, expect, vi } from "vitest";
import { SceneManager } from "./sceneManager.js";
const stubs = () => {
  const camera = { position: { set: vi.fn() } };
  const controls = { target: { set: vi.fn() }, update: vi.fn() };
  const renderer = { render: vi.fn() };
  return { camera, controls, renderer };
};
const fake = (name) => {
  const s = { dispose: vi.fn(), update: vi.fn(() => `cap-${name}`), scene: {}, cameraPos: [0, 1, 2], duration: 3 };
  return { s, factory: () => s };
};
describe("SceneManager", () => {
  it("disposes the old scene exactly once per switch", () => {
    const { camera, controls, renderer } = stubs();
    const m = new SceneManager(renderer, camera, controls);
    const a = fake("a"), b = fake("b");
    m.show(a.factory, {});
    m.show(b.factory, {});
    expect(a.s.dispose).toHaveBeenCalledTimes(1);
    expect(b.s.dispose).not.toHaveBeenCalled();
    expect(camera.position.set).toHaveBeenLastCalledWith(0, 1, 2);
  });
  it("tick returns current caption and renders", () => {
    const { camera, controls, renderer } = stubs();
    const m = new SceneManager(renderer, camera, controls);
    const a = fake("a");
    m.show(a.factory, {});
    expect(m.tick()).toBe("cap-a");
    expect(renderer.render).toHaveBeenCalledWith(a.s.scene, camera);
  });
});
