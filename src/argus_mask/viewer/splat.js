// LAM Gaussian-splat heads (a face whose manifest.model.format is "lam-splat"): everything splat-specific, for
// the viewer (viewer.js) and the editor's previews (stage.js).
//
// The archive (a .zip: skin.glb, offset.ply, animation.glb, vertex_order.json) is drawn by Myned AI's
// gsplat-flame-avatar-renderer (MIT, vendor/), which deforms the splats with FLAME's bones and ARKit shapes.
// mask is the only thing that moves the head: the weights come through the page's own pipeline (frames,
// interpolation, motion.js's rest set, idle life and lip budget), the head turn through the skeleton's neck and
// head bones, and LAM's own animation clips are held still, so a screen and a recording show the same thing.
//
// - The lip budget is measured from the archive's skin.glb at unit scale: the FBX round trip leaves a 0.01
//   scale on its node while the vertices are already metres, and the renderer draws the vertices as they are.
// - The jaw is capped (knee 0.35, slope 0.35 on jawOpen and mouthLowerDown, ?jaw= moves the knee): the splat
//   mouth's inside is weak, and wide openings show it. The cap runs before the lip budget, so the budget
//   works against the jaw actually shown.
// - The renderer's own requestAnimationFrame loop is stopped: the page calls step() once per frame it draws,
//   or exact() in record mode, which resolves only when this pose, sorted for this camera, is on the canvas.
// - The canvas is sized here, before each draw. The renderer's own ResizeObserver is disconnected: its
//   callback runs after requestAnimationFrame callbacks, and resizing a canvas clears it, so a frame drawn
//   in that tick (a recorded one included) would be presented blank.
// - The head turn: LAM's skin.glb carries a full body rig (neckLower, neckUpper, head; not FLAME's single
//   "neck", so the renderer's getNeckPose({neck}) moves nothing on it). The turn is shared down the chain and
//   applied about world axes at each bone, as the mesh viewer turns its head: yaw, pitch, roll in degrees.
// - The camera is fixed. The renderer builds no orbit controls (its setupControls is commented out); if a
//   later version does, they are told to stop listening to the window's keys (W A S D would be lost to text boxes).

import * as THREE from 'three';
import JSZip from 'jszip';
import { GLTFLoader } from './vendor/loaders/GLTFLoader.js';
import { GaussianSplatRenderer } from './vendor/gsplat-flame-avatar-renderer.esm.min.js';
import { ARKIT, lipBudget } from './motion.js';

// Where the camera sits when the manifest does not say (faces.py SPLAT_FRAMING): LAM's heads stand at about
// 1.65 m. The camera aims at `height` from a tenth of its distance above it, as a portrait is taken.
export const SPLAT_FRAMING = { distance: 0.7, height: 1.65, fov: 50 };
const MESH_DEFAULT = { distance: 0.6, height: 0, fov: 30 };   // models.Framing's defaults: "not set" for a splat head
const LIFT = 0.1;

export function splatFraming(framing) {
  const f = framing || {};
  const unset = !framing || (f.distance === MESH_DEFAULT.distance && (f.height || 0) === MESH_DEFAULT.height && f.fov === MESH_DEFAULT.fov);
  return unset ? { ...SPLAT_FRAMING } : { distance: f.distance || SPLAT_FRAMING.distance, height: f.height || 0, fov: f.fov || SPLAT_FRAMING.fov };
}

export const JAW = { knee: 0.35, slope: 0.35 };

// Past the knee, jawOpen (and the lower lip with it) opens at `slope` of the rate. A knee of 1 or more is no cap.
// Returns a Motion `cap`: (weights, index) => void, in place.
export function jawCap(knee = JAW.knee, slope = JAW.slope) {
  const capped = (x) => (knee >= 1 || x <= knee ? x : knee + (x - knee) * slope);
  return (w, index) => {
    const jaw = index.get('jawOpen');
    if (jaw !== undefined) w[jaw] = capped(w[jaw]);
    const l = index.get('mouthLowerDownLeft');
    const r = index.get('mouthLowerDownRight');
    const lip = Math.max(l !== undefined ? w[l] : 0, r !== undefined ? w[r] : 0);
    if (lip > 0) {
      const k = capped(lip) / lip;
      if (l !== undefined) w[l] *= k;
      if (r !== undefined) w[r] *= k;
    }
  };
}

// The archive's skin.glb, read here (the renderer reads its own copy): the lip budget, and the bones' rest pose.
export async function readSkin(url) {
  const reply = await fetch(url);
  if (!reply.ok) throw new Error(`The splat head could not be fetched (${reply.status}).`);
  const zip = await JSZip.loadAsync(await reply.arrayBuffer());
  const entry = Object.values(zip.files).find((f) => !f.dir && /(^|\/)skin\.glb$/.test(f.name));
  if (!entry) throw new Error('The splat head has no skin.glb.');
  const gltf = await new GLTFLoader().parseAsync(await entry.async('arraybuffer'), '');
  // The rest pose, as the file has it, before anything moves a bone: each frame starts from it.
  const rest = new Map();
  gltf.scene.traverse((node) => {
    if (node.isBone) rest.set(node.name, { position: node.position.clone(), quaternion: node.quaternion.clone(), scale: node.scale.clone() });
  });
  // Measure at unit scale (see the top of this file), or every gap reads 100 times small and the budget is null.
  gltf.scene.traverse((node) => node.scale.set(1, 1, 1));
  gltf.scene.updateMatrixWorld(true);
  return { budget: lipBudget(gltf.scene), rest };
}

// Down the neck to the head: how much of a head turn each bone takes (LAM's rig; FLAME's has "neck" alone).
const NECK_CHAIN = [['neckLower', 0.2], ['neckUpper', 0.3], ['head', 0.5]];
const _euler = new THREE.Euler();
const _share = new THREE.Quaternion();
const _local = new THREE.Quaternion();

const until = async (ready, what, ms = 20000) => {
  const start = performance.now();   // a timeout only: no pose depends on it
  while (!ready()) {
    if (performance.now() - start > ms) throw new Error(`The splat head did not ${what} in ${ms / 1000} s.`);
    await new Promise((resolve) => setTimeout(resolve, 1));
  }
};
const nextFrame = () => new Promise((resolve) => requestAnimationFrame(resolve));

export class SplatHead {
  // host: an element with a size (the canvas fills it). url: the archive, ending .zip (the renderer insists).
  // framing: the manifest's. background: any CSS colour. skin: readSkin(url)'s result. instant: no fade-in (record).
  static async create(host, { url, framing, background = '#101216', skin, instant = false }) {
    const head = new SplatHead();
    head.rest = skin.rest;
    head.expression = Object.fromEntries(ARKIT.map((n) => [n, 0]));
    head.turn = new THREE.Quaternion();
    head.bones = [];   // filled once the renderer has loaded the skin (its first frame, inside create(), has none)
    let colour = '0x101216';
    try { colour = `0x${new THREE.Color(background).getHexString()}`; } catch { /* keep the default */ }
    const r = await GaussianSplatRenderer.create(host, url, {
      backgroundColor: colour,
      // No getChatState: LAM's clip state machine never runs. Expressions and the head are mask's alone.
      getExpressionData: () => head.expression,
      getNeckPose: () => head.pose(),
    });
    head.r = r;
    const v = r.viewer;
    cancelAnimationFrame(v.requestFrameId);   // the page drives every frame (step, exact)
    if (v.resizeObserver) v.resizeObserver.disconnect();   // fit() sizes the canvas before each draw instead
    head.host = host;
    if (v.controls && v.controls.stopListenToKeyEvents) v.controls.stopListenToKeyEvents();
    // Hold LAM's clips still: no action plays, so the mixer moves no bone; pose() puts the rest pose back each frame.
    if (r.mixer) { r.mixer.stopAllAction(); r.mixer.timeScale = 0; }
    if (instant) v.sceneRevealMode = 2;   // SceneRevealMode.Instant: no fade-in that depends on the clock
    const found = new Map();
    v.boneRoot?.traverse((node) => {
      if (node.isBone && head.rest.has(node.name)) { head.bones.push([node, head.rest.get(node.name)]); found.set(node.name, node); }
    });
    // The bones that turn the head, each with its share, and its world rotation at rest (to turn about world axes).
    head.pose();
    v.avatarMesh?.updateMatrixWorld(true);
    const chain = [NECK_CHAIN, [['neck', 1]], [['head', 1]]].map((c) => c.filter(([n]) => found.has(n))).find((c) => c.length) || [];
    const total = chain.reduce((sum, [, share]) => sum + share, 0);
    head.turners = chain.map(([name, share]) => {
      const bone = found.get(name);
      const world = bone.getWorldQuaternion(new THREE.Quaternion());
      return { bone, share: share / total, world, inverse: world.clone().invert() };
    });
    const f = splatFraming(framing);
    const camera = v.camera;
    camera.fov = f.fov;
    camera.position.set(0, f.height + LIFT * f.distance, f.distance);
    camera.lookAt(0, f.height, 0);
    camera.updateProjectionMatrix();
    camera.updateMatrixWorld(true);
    head.framing = f;
    // create() resolves after the first section of a progressive load; the rest builds afterwards (each build
    // stops rendering until its sort is done). Wait for the last, or a recording's first frames race it.
    await until(() => !v.isLoadingOrUnloading() && v.splatRenderReady && v.splatMesh?.finalBuild && v.splatMesh.getSplatCount() > 0, 'load');
    return head;
  }

  get canvas() { return this.r.viewer.renderer.domElement; }

  // Called by the renderer once a frame (as getNeckPose), after its mixer and before skinning: the rest pose,
  // then mask's head turn. Each turning bone takes its share of the turn about world axes at its own pivot:
  // its local rotation becomes rest * (W^-1 D W), W its world rotation at rest, so its world rotation is D W.
  // Returns null: the bones are set here, nothing is left for the renderer to add.
  pose() {
    for (const [bone, rest] of this.bones) {
      bone.position.copy(rest.position);
      bone.quaternion.copy(rest.quaternion);
      bone.scale.copy(rest.scale);
    }
    for (const t of this.turners || []) {
      _share.identity().slerp(this.turn, t.share);
      _local.copy(t.inverse).multiply(_share).multiply(t.world);
      t.bone.quaternion.multiply(_local);
    }
    // The renderer reads the skeleton before it updates the bones' world matrices (runMorphUpdate), so a pose
    // set now would show a frame late: bring them up to date here.
    this.r?.viewer?.boneRoot?.updateMatrixWorld(true);
    return null;
  }

  // What to show: weights (any array) in *names* order, and the head turn in degrees ({ yaw, pitch, roll }).
  // Shapes the head lacks are ignored; names it has that the frame does not are 0.
  set(weights, names, head) {
    const e = this.expression;
    for (const n of ARKIT) e[n] = 0;
    for (let i = 0; i < names.length; i += 1) if (names[i] in e) e[names[i]] = weights[i];
    const rad = Math.PI / 180;
    // As the mesh viewer turns its head: Euler YXZ of (pitch, yaw, roll), about world axes.
    _euler.set(((head && head.pitch) || 0) * rad, ((head && head.yaw) || 0) * rad, ((head && head.roll) || 0) * rad, 'YXZ');
    this.turn.setFromEuler(_euler);
  }

  // The canvas at its host's size (CSS pixels, as the renderer counts them).
  fit() {
    const v = this.r.viewer;
    const size = v.renderer.getSize(new THREE.Vector2());
    const w = this.host.offsetWidth;
    const h = this.host.offsetHeight;
    if (w && h && (size.x !== w || size.y !== h)) v.renderer.setSize(w, h);
  }

  // One frame, now: pose, skin, start a depth sort, draw (with the last finished sort: fine while live).
  step() {
    this.fit();
    this.r.render();
    cancelAnimationFrame(this.r.viewer.requestFrameId);
  }

  // Record mode: resolves once exactly the pose last set() is on the canvas, drawn with the depth sort for it and
  // this camera. The sort runs in a worker: wait for any in flight, start this pose's, wait for it, then draw.
  async exact() {
    const v = this.r.viewer;
    await until(() => !v.sortRunning, 'finish sorting');
    this.step();
    await until(() => !v.sortRunning, 'finish sorting');
    await nextFrame();
    this.fit();
    v.render();
  }

  dispose() {
    try { this.r.dispose(); } catch { /* already gone */ }
  }
}
