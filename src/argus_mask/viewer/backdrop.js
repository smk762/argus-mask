// Backdrops: an optional scene with a little life in it behind the face (?scene=ftl). Cosmetic and the
// viewer's own, like idle life at rest: never in a take, so the engine's determinism is untouched.
//
// A scene is a module under scenes/ (see scenes/ftl.js for the interface). The host draws it into a render
// target of capped size and rate, and the viewer uses that as scene.background: three.js draws it behind the
// head, and the look's post-processing (depth of field on high) softens it like any far background. A hidden
// tab draws nothing (requestAnimationFrame stops), so a backdrop costs nothing there.
//
// Time: live, the page's own clock; in a recording, the posed frame's t_ms, so two recordings of one take
// show the same backdrop in the same place.

// The built-in scenes. A name is looked up here before any import, so ?scene= never names a file.
export const SCENES = {
  ftl: () => import('./scenes/ftl.js'),
};

// Per profile: the longest side the backdrop draws at, and how often it redraws. It is soft and far, so
// low and medium draw it smaller and the GPU upscales it; high (and record) draw it at full size every frame.
const BUDGET = {
  low: { maxPx: 480, maxFps: 30 },
  medium: { maxPx: 960, maxFps: 60 },
  high: { maxPx: 4096, maxFps: Infinity },
};

export class Backdrop {
  static async create(THREE, renderer, name, { quality = 'medium', options = {} } = {}) {
    const load = Object.hasOwn(SCENES, name) ? SCENES[name] : null;
    if (!load) throw new Error(`Unknown scene "${name}". Scenes: ${Object.keys(SCENES).join(', ')}.`);
    const module = await load();
    return new Backdrop(THREE, renderer, name, module, BUDGET[quality] || BUDGET.medium, options);
  }

  constructor(THREE, renderer, name, module, budget, options) {
    this.name = name;
    this.meta = module.meta;
    this.renderer = renderer;
    this.budget = budget;
    this.material = module.material(THREE, options);
    // Half float: the scene writes linear light, and 8 bits of dark linear values band into rings.
    this.target = new THREE.WebGLRenderTarget(1, 1, { depthBuffer: false, type: THREE.HalfFloatType });
    this.scene = new THREE.Scene();
    const quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.material);
    quad.frustumCulled = false;
    this.scene.add(quad);
    this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    this.drawnAt = -Infinity;   // page time of the last draw, for the rate cap
  }

  get texture() { return this.target.texture; }

  setSize(width, height) {
    const scale = Math.min(1, this.budget.maxPx / Math.max(1, width, height));
    const w = Math.max(1, Math.round(width * scale));
    const h = Math.max(1, Math.round(height * scale));
    this.target.setSize(w, h);
    this.material.uniforms.uRes.value.set(w, h);
    this.drawnAt = -Infinity;   // a resize clears the target: draw again now
  }

  // Draw the scene at time *t* (seconds). *exact*: a recording, where every frame must show its own time.
  update(t, { exact = false } = {}) {
    const now = performance.now();
    if (!exact && now - this.drawnAt < 1000 / this.budget.maxFps) return;
    this.drawnAt = now;
    this.material.uniforms.uTime.value = t;
    const previous = this.renderer.getRenderTarget();
    this.renderer.setRenderTarget(this.target);
    this.renderer.render(this.scene, this.camera);
    this.renderer.setRenderTarget(previous);
  }

  get summary() {
    const { maxFps } = this.budget;
    return { name: this.name, title: this.meta.title, size: [this.target.width, this.target.height], maxFps: Number.isFinite(maxFps) ? maxFps : 'every frame' };
  }
}
