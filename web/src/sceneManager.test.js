import { describe, it, expect, vi } from "vitest";
import { HOLD_SECONDS, SceneManager } from "./sceneManager.js";
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
  it("holds are ON by default (no toggle needed) with a 2.5 s dwell", () => {
    expect(HOLD_SECONDS).toBe(2.5);
    const { camera, controls, renderer } = stubs();
    let t = 0;
    const m = new SceneManager(renderer, camera, controls, { now: () => t * 1000 });
    expect(m.holds).toBe(true);
    const seen = [];
    m.show(() => ({ dispose() {}, update: (e) => { seen.push(e); return ""; }, scene: {}, cameraPos: [0, 0, 1], duration: 10, stops: [2] }), {});
    t = 2.5; m.tick();
    expect(seen.at(-1)).toBe(2);
    expect(m.holding).toBeCloseTo(2.5 - 0.5, 6);
    t = 4.4; m.tick();
    expect(seen.at(-1)).toBe(2); // still dwelling (0.5 + 1.9 < 2.5)
    t = 5.5; m.tick();
    expect(seen.at(-1)).toBeCloseTo(3, 6); // dwell over: 3.5 s wall - 2.5 s dwell = scene time 3
  });
  it("holds on: time clamps at 2 for the dwell then continues", () => {
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

describe("SceneManager camera drive", () => {
  const setup = () => {
    let az = 0;
    const controls = {
      target: { set: vi.fn() }, autoRotate: true, autoRotateSpeed: 1,
      update: vi.fn((dt) => { if (controls.autoRotate && dt) az += controls.autoRotateSpeed * dt * 0.25; }),
      getAzimuthalAngle: () => az,
    };
    const camera = { position: { set: vi.fn() } };
    let t = 0;
    const m = new SceneManager({ render: vi.fn() }, camera, controls, { now: () => t * 1000 });
    const s = { dispose: vi.fn(), update: () => "", scene: {}, cameraPos: [0, 0, 1], duration: 100, stops: [2] };
    m.show(() => s, {});
    return { m, controls, adv: (d) => { t += d; m.tick(); }, az: () => az };
  };
  it("passes delta seconds and stops rotating while paused or holding", () => {
    const { m, controls, adv } = setup();
    adv(0.05);
    expect(controls.update).toHaveBeenLastCalledWith(0.05);
    expect(controls.autoRotate).toBe(true);
    m.setPaused(true); adv(0.05);
    expect(controls.autoRotate).toBe(false);
    m.setPaused(false); m.setHolds(true); adv(2.5);
    expect(m.holding).toBeGreaterThan(0);
    adv(0.1);
    expect(controls.autoRotate).toBe(false);
  });
  it("swing stays bounded near +-0.5 rad over a long run", () => {
    const { adv, az } = setup();
    let lo = 0, hi = 0;
    for (let i = 0; i < 6000; i++) { adv(1 / 60); lo = Math.min(lo, az()); hi = Math.max(hi, az()); }
    expect(hi).toBeLessThan(0.55);
    expect(lo).toBeGreaterThan(-0.55);
    expect(hi - lo).toBeGreaterThan(0.4);
  });
});
