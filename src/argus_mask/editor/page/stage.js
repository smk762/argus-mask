// A face on a three.js stage: load a model, pose it, show it in any box, or take a still of it.
// Shared by the editor (index.html) and the saved-faces page (faces.html).

import * as THREE from 'three';
import { GLTFLoader } from './vendor/loaders/GLTFLoader.js';
import { Look, lookOptions } from './look.js';
import { ARKIT, Motion, lipBudget } from './motion.js';

// Real speech for the preview: 12 s of Audio2Face weights (talk.json, no audio), looped with a pause, so a
// face is judged speaking as it will on a screen: the same lip budget, rest set and idle life (motion.js).
const TALK_GAP_S = 1.5;
let talk = null;
const talkReady = fetch('talk.json').then((r) => (r.ok ? r.json() : null)).then((data) => {
  if (!data) return;
  const at = data.names.map((n) => ARKIT.indexOf(n));
  talk = { fps: data.fps, frames: data.frames.map((f) => { const w = new Float32Array(ARKIT.length); f.forEach((v, i) => { if (at[i] >= 0) w[at[i]] = v; }); return w; }) };
}).catch(() => {});

const NONE = new Float32Array(ARKIT.length);
const fixed = (weights) => { const w = new Float32Array(ARKIT.length); for (const [n, v] of Object.entries(weights)) w[ARKIT.indexOf(n)] = v; return () => ({ weights: w, live: true }); };

// Each pose gives the frame weights (ARKit order) at t seconds, and whether it is speech (no idle life).
export const POSES = {
  rest: () => ({ weights: NONE, live: false }),
  speak: (t) => {
    if (!talk) return { weights: NONE, live: false };
    const loop = talk.frames.length / talk.fps + TALK_GAP_S;
    const i = Math.floor((t % loop) * talk.fps);
    return i < talk.frames.length ? { weights: talk.frames[i], live: true } : { weights: NONE, live: false };
  },
  open: fixed({ jawOpen: 0.6, mouthFunnel: 0.1 }),
  smile: fixed({ mouthSmileLeft: 0.7, mouthSmileRight: 0.7, jawOpen: 0.12 }),
};
export { talkReady };

// The rest sets and each face's choice (GET /expressions), fetched once per page; fresh() after a change.
let setsCache = null;
export function restSets(fresh = false) {
  if (!setsCache || fresh) setsCache = fetch('/expressions').then((r) => (r.ok ? r.json() : null)).catch(() => null);
  return setsCache;
}

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
    this.pose = 'speak';
    this.turn = false;
    this.motion = new Motion();
    this.restWeights = {};
    this.restLife = 1;
    this.shown = new Float32Array(ARKIT.length);
    this.lastT = null;
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
    this.motion = new Motion({ budget: lipBudget(gltf.scene), rest: this.restWeights, life: this.restLife });
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

  // The rest set this stage shows (a set's weights and life, from GET /expressions).
  setRest(set) {
    this.restWeights = (set && set.weights) || {};
    this.restLife = set && Number.isFinite(set.life) ? set.life : 1;
    this.motion.setRest(this.restWeights);
    this.motion.idle.life = this.restLife;
  }

  frame(t) {
    const dt = this.lastT === null ? 0 : Math.min(0.1, t - this.lastT);
    this.lastT = t;
    const pose = POSES[this.pose](t);
    // Relax towards the frame's weights (a cut between speech and rest would snap the mouth).
    const k = 1 - Math.exp(-14 * dt);
    for (let i = 0; i < this.shown.length; i += 1) this.shown[i] += (pose.weights[i] - this.shown[i]) * (pose.live ? 1 : k);
    const { weights, head } = this.motion.present(this.shown, ARKIT, { live: pose.live, t, dt });
    for (let i = 0; i < ARKIT.length; i += 1) this.weigh(ARKIT[i], weights[i]);
    const rad = Math.PI / 180;
    const idle = head || { yaw: 0, pitch: 0, roll: 0 };
    this.headTurn = this.headTurn || { yaw: 0, pitch: 0, roll: 0 };
    for (const axis of ['yaw', 'pitch', 'roll']) this.headTurn[axis] += (idle[axis] - this.headTurn[axis]) * (1 - Math.exp(-6 * dt));
    this.spin = (this.spin || 0) + ((this.turn ? 0.6 : 0) - (this.spin || 0)) * 0.08;
    this.pivot.rotation.set(this.headTurn.pitch * rad, this.spin + this.headTurn.yaw * rad, this.headTurn.roll * rad);
    this.look.render();
  }

  // A still of the loaded face at rest, as a data URL: for a card, without a live canvas per card.
  snapshot(width = 240, height = 288) {
    this.renderer.setPixelRatio(1);
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    for (const name of this.targets.keys()) this.weigh(name, this.restWeights[name] ?? 0);   // at rest, in its rest set
    this.pivot.rotation.set(0, 0, 0);
    this.look.setSize(width, height);
    this.look.render();
    return this.renderer.domElement.toDataURL('image/jpeg', 0.85);
  }

  // Pose buttons for this stage, and for any linked ones: a comparison moves both faces together. With
  // *faceId*, a rest-set picker too, starting at that face's own set; *onRest(name)* is told of a change
  // (the saved-faces page keeps it; elsewhere the choice only previews).
  poseButtons(bar, linked = [], { faceId = null, onRest = null } = {}) {
    const all = [this, ...linked];
    restSets().then((sets) => {
      if (!sets || !bar.isConnected) return;
      const pick = Object.assign(document.createElement('select'), { title: 'Rest set: the expression the face holds at rest and speaks from' });
      for (const name of Object.keys(sets.sets)) pick.append(new Option(`rest: ${name}`, name));
      pick.value = (faceId && sets.faces[faceId]) || sets.default;
      const show = () => { for (const st of all) st.setRest(sets.sets[pick.value]); };
      pick.onchange = () => { show(); if (onRest) onRest(pick.value); };
      show();
      bar.append(pick);
    });
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
