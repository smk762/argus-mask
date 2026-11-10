// A face on a three.js stage: load a model, pose it, show it in any box, or take a still of it.
// Shared by the editor (index.html) and the saved-faces page (faces.html).

import * as THREE from 'three';
import { GLTFLoader } from './vendor/loaders/GLTFLoader.js';
import { Look, lookOptions } from './look.js';

export const POSES = {
  rest: () => ({}),
  open: () => ({ jawOpen: 0.6, mouthFunnel: 0.1 }),
  smile: () => ({ mouthSmileLeft: 0.7, mouthSmileRight: 0.7, jawOpen: 0.12 }),
  // A rough syllable rhythm, to judge the mouth in motion. The real motion comes from Audio2Face.
  talk: (t) => {
    const s = Math.max(0, Math.sin(t * 9.0)) * (0.55 + 0.45 * Math.sin(t * 2.3));
    return { jawOpen: 0.45 * s, mouthFunnel: 0.25 * Math.max(0, Math.sin(t * 4.1)), mouthClose: 0.1 * (1 - s),
             mouthPucker: 0.3 * Math.max(0, Math.sin(t * 3.3 + 1)), mouthStretchLeft: 0.2 * s, mouthStretchRight: 0.2 * s };
  },
};

function dispose(root) {
  root.traverse((node) => {
    if (!node.isMesh) return;
    node.geometry?.dispose();
    for (const m of [].concat(node.material || [])) {
      for (const v of Object.values(m)) if (v && v.isTexture && !v.userData.shared) v.dispose();  // look.js's skin detail is shared
      m.dispose();
    }
  });
}

export class Stage {
  constructor({ animate = true } = {}) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color('#101216');
    this.camera = new THREE.PerspectiveCamera(30, 1, 0.01, 100);
    // The viewer's own look (look.js), so a face looks here as it will on a screen. Medium by default;
    // the page's ?quality= and single switches apply as in the viewer.
    this.look = new Look(this.renderer, this.scene, this.camera, lookOptions(new URLSearchParams(location.search), 'medium'));
    this.pivot = new THREE.Group();
    this.scene.add(this.pivot);
    this.targets = new Map();
    this.pose = 'rest';
    this.turn = false;
    this.loaded = false;
    this.host = null;
    window.addEventListener('resize', () => this.resize());
    if (animate) {
      const tick = (now) => { requestAnimationFrame(tick); if (this.loaded && this.host) this.frame(now / 1000); };
      requestAnimationFrame(tick);
    }
  }

  async load(modelUrl, framing = {}) {
    const gltf = await new GLTFLoader().loadAsync(modelUrl);
    for (const child of [...this.pivot.children]) dispose(child);
    this.pivot.clear();
    this.targets = new Map();
    this.pivot.add(gltf.scene);
    this.look.apply(gltf.scene);
    gltf.scene.traverse((node) => {
      if (node.isMesh && node.morphTargetDictionary) {
        for (const [name, index] of Object.entries(node.morphTargetDictionary)) {
          if (!this.targets.has(name)) this.targets.set(name, []);
          this.targets.get(name).push({ influences: node.morphTargetInfluences, index });
        }
      }
    });
    this.camera.fov = framing.fov || 30;
    this.camera.position.set(0, framing.height || 0, framing.distance || 0.6);
    this.camera.lookAt(0, framing.height || 0, 0);
    this.loaded = true;
    this.resize();
    this.look.focus(gltf.scene);
  }

  attach(host) {
    this.host = host;
    host.prepend(this.renderer.domElement);
    this.resize();
  }

  resize() {
    if (!this.host) return;
    const box = this.host.getBoundingClientRect();
    if (!box.width || !box.height) return;
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setSize(box.width, box.height, false);
    this.camera.aspect = box.width / box.height;
    this.camera.updateProjectionMatrix();
    this.look.setSize(box.width, box.height);
  }

  weigh(name, value) {
    for (const { influences, index } of this.targets.get(name) || []) influences[index] = value;
  }

  frame(t) {
    const weights = POSES[this.pose](t);
    const blink = (t % 4.2) < 0.12 ? 1 : 0;
    for (const name of this.targets.keys()) this.weigh(name, weights[name] ?? 0);
    this.weigh('eyeBlinkLeft', blink);
    this.weigh('eyeBlinkRight', blink);
    this.pivot.rotation.y += ((this.turn ? 0.6 : 0) - this.pivot.rotation.y) * 0.08;
    this.look.render();
  }

  // A still of the loaded face at rest, as a data URL: for a card, without a live canvas per card.
  snapshot(width = 240, height = 288) {
    this.renderer.setPixelRatio(1);
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    for (const name of this.targets.keys()) this.weigh(name, 0);
    this.pivot.rotation.y = 0;
    this.look.setSize(width, height);
    this.look.render();
    return this.renderer.domElement.toDataURL('image/jpeg', 0.85);
  }

  // Pose buttons for this stage, and for any linked ones: a comparison moves both faces together.
  poseButtons(bar, linked = []) {
    const all = [this, ...linked];
    bar.replaceChildren();
    for (const name of Object.keys(POSES)) {
      const b = Object.assign(document.createElement('button'), { type: 'button', textContent: name });
      b.dataset.pose = name;
      b.classList.toggle('on', name === this.pose);
      b.onclick = () => {
        for (const st of all) st.pose = name;
        for (const x of bar.children) if (x.dataset.pose) x.classList.toggle('on', x === b);
      };
      bar.append(b);
    }
    const spin = Object.assign(document.createElement('button'), { type: 'button', textContent: 'turn' });
    spin.classList.toggle('on', this.turn);
    spin.onclick = () => {
      const turn = !this.turn;
      for (const st of all) st.turn = turn;
      spin.classList.toggle('on', turn);
    };
    bar.append(spin);
  }
}
