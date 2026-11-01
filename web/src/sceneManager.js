export class SceneManager {
  constructor(renderer, camera, controls) {
    this.renderer = renderer;
    this.camera = camera;
    this.controls = controls;
    this.current = null;
    this.start = performance.now();
    this.elapsed = 0;
  }

  show(factory, ctx) {
    if (this.current) {
      this.current.dispose();
      this.current = null;
    }
    this.current = factory(ctx);
    this.start = performance.now();
    this.elapsed = 0;
    const p = this.current.cameraPos;
    this.camera.position.set(p[0], p[1], p[2]);
    if (this.controls && this.controls.target) this.controls.target.set(0, 0, 0);
    if (this.controls) this.controls.update();
  }

  tick() {
    if (!this.current) return "";
    this.elapsed = (performance.now() - this.start) / 1000;
    const caption = this.current.update(this.elapsed);
    if (this.controls) this.controls.update();
    this.renderer.render(this.current.scene, this.camera);
    return caption;
  }

  get duration() {
    return this.current ? this.current.duration : 0;
  }
}
