export const HOLD_SECONDS = 3;

export class SceneManager {
  constructor(renderer, camera, controls, opts = {}) {
    this.renderer = renderer;
    this.camera = camera;
    this.controls = controls;
    this.now = opts.now || (() => performance.now());
    this.current = null;
    this.elapsed = 0;
    this._paused = false;
    this._holds = false;
    this._holdLeft = 0;
    this._passed = new Set();
    this._last = this.now();
  }

  get paused() { return this._paused; }
  get holds() { return this._holds; }
  get holding() { return this._holdLeft; }

  setPaused(v) { this._paused = !!v; }
  togglePause() { this._paused = !this._paused; return this._paused; }
  setHolds(v) {
    this._holds = !!v;
    if (!this._holds) this._holdLeft = 0;
  }

  show(factory, ctx) {
    if (this.current) {
      this.current.dispose();
      this.current = null;
    }
    this.current = factory(ctx);
    this._last = this.now();
    this.elapsed = 0;
    this._holdLeft = 0;
    this._passed = new Set();
    const p = this.current.cameraPos;
    this.camera.position.set(p[0], p[1], p[2]);
    if (this.controls && this.controls.target) this.controls.target.set(0, 0, 0);
    if (this.controls) this.controls.update();
  }

  _advance(dt) {
    if (this._holdLeft > 0) {
      if (dt < this._holdLeft) { this._holdLeft -= dt; return; }
      dt -= this._holdLeft;
      this._holdLeft = 0;
    }
    const target = this.elapsed + dt;
    if (this._holds) {
      const stops = this.current.stops || [];
      const next = stops.find((s) => !this._passed.has(s) && s > this.elapsed);
      if (next !== undefined && target >= next) {
        this._passed.add(next);
        this.elapsed = next;
        this._holdLeft = HOLD_SECONDS;
        this._advance(target - next); // time past the stop already counts toward the hold
        return;
      }
    }
    this.elapsed = target;
  }

  tick() {
    const n = this.now();
    const dt = Math.max(0, (n - this._last) / 1000);
    this._last = n;
    if (!this.current) return "";
    if (!this._paused) this._advance(dt);
    const caption = this.current.update(this.elapsed);
    if (this.controls) this.controls.update();
    this.renderer.render(this.current.scene, this.camera);
    return caption;
  }

  get duration() {
    return this.current ? this.current.duration : 0;
  }
}
