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

describe("SceneManager holds and pause", () => {
  const setup = (holds) => {
    const { camera, controls, renderer } = stubs();
    let t = 0;
    const m = new SceneManager(renderer, camera, controls, { now: () => t * 1000 });
    const seen = [];
    const s = { dispose: vi.fn(), update: vi.fn((e) => { seen.push(e); return ""; }), scene: {}, cameraPos: [0, 0, 1], duration: 10, stops: [2, 5] };
    m.show(() => s, {});
    m.setHolds(holds);
    const adv = (d) => { t += d; m.tick(); };
    return { m, adv, last: () => seen[seen.length - 1] };
  };
  it("holds on: time clamps at 2 for 3s then continues", () => {
    const { m, adv, last } = setup(true);
    adv(2.5);
    expect(last()).toBe(2);
    expect(m.holding).toBeGreaterThan(0);
    adv(3.0);
    expect(last()).toBeGreaterThan(2);
    expect(last()).toBeLessThan(5);
  });
  it("holds off: passes through stops", () => {
    const { m, adv, last } = setup(false);
    adv(2.5);
    expect(last()).toBe(2.5);
    expect(m.holding).toBe(0);
  });
  it("pause freezes time and hold countdown", () => {
    const { m, adv, last } = setup(true);
    adv(2.2);
    expect(m.holding).toBeGreaterThan(0);
    m.setPaused(true);
    adv(10);
    expect(last()).toBe(2);
    expect(m.holding).toBeGreaterThan(0);
    m.setPaused(false);
    adv(3.1);
    expect(last()).toBeGreaterThan(2);
  });
  it("show() resets stops", () => {
    const { m, adv, last } = setup(true);
    adv(2.5); adv(3.5);
    expect(last()).toBeGreaterThan(2);
    m.show(() => ({ dispose() {}, update: () => "", scene: {}, cameraPos: [0, 0, 1], duration: 10, stops: [2, 5] }), {});
    adv(2.5);
    expect(m.elapsed).toBe(2);
    expect(m.holding).toBeGreaterThan(0);
  });
});
