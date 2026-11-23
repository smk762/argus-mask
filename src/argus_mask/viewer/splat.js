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
// - A backdrop (backdrop.js) is drawn with the renderer's own WebGL context and set as the background of its
//   three.js scene. The renderer draws that scene (background first) before the splats, but only when something
//   in it is visible, so an empty group stands in. One canvas: exact() covers the backdrop too.
// - The camera is fixed. The renderer builds no orbit controls (its setupControls is commented out); if a
//   later version does, they are told to stop listening to the window's keys (W A S D would be lost to text boxes).

import * as THREE from 'three';
import JSZip from 'jszip';
import { GLTFLoader } from './vendor/loaders/GLTFLoader.js';
import { GaussianSplatRenderer } from './vendor/gsplat-flame-avatar-renderer.esm.min.js';
import { Backdrop } from './backdrop.js';
import { ARKIT, lipBudget } from './motion.js';

// Where the camera sits when the manifest does not say (faces.py SPLAT_FRAMING): LAM's heads stand at about
// 1.65 m. The camera aims at `height` from a tenth of its distance above it, as a portrait is taken. A long lens, as
// the portrait was taken with: from 0.7 m at 50 degrees the same head reads narrower, longer and older (#151).
export const SPLAT_FRAMING = { distance: 1.6, height: 1.65, fov: 23 };
const SPLAT_FRAMING_OLD = { distance: 0.7, height: 1.65, fov: 50 };   // before #151, still in older manifests
const MESH_DEFAULT = { distance: 0.6, height: 0, fov: 30 };   // models.Framing's defaults: "not set" for a splat head
// Over a backdrop, closer: LAM's head fades out below the collarbone, which a plain dark background hides and a
// scene shows as a floating head. This shot ends just above the fade. Used only when the manifest sets no framing.
export const SPLAT_CLOSE = { distance: 1.03, height: 1.67, fov: 23 };
const LIFT = 0.1;
// Where a LAM head ends: its splats thin out into ragged wisps below the collarbone, where the portrait's shoulders
// ran out. Between these heights (metres, the head's own space) the picture fades to what is behind the head, so the
// bust ends in a soft edge of our making. LAM stands every head alike, so one pair serves them all.
export const FADE = { top: 1.5, bottom: 1.445 };
const FADE_ROWS = 12;   // strips across the fade: a smooth curve in vertex alpha
const FADE_COLS = 33;   // and down it: the shadow's curve across the picture
// The shadow under the bust over a backdrop: how dark at its deepest, and the bust's half width (metres) it covers
// fully before easing out over as much again.
export const SHADOW = { depth: 0.55, width: 0.2 };

export function splatFraming(framing) {
  const f = framing || {};
  return splatFraming.isDefault(framing) ? { ...SPLAT_FRAMING } : { distance: f.distance || SPLAT_FRAMING.distance, height: f.height || 0, fov: f.fov || SPLAT_FRAMING.fov };
}
// Whether a manifest's framing is "not set": absent, models.Framing's defaults (a mesh head's), or LAM's standard
// shot (now or before #151), which `inspect` writes into every splat manifest. Only a framing chosen for this face
// is its own.
splatFraming.isDefault = (framing) => {
  const f = framing || {};
  const same = (d) => f.distance === d.distance && (f.height || 0) === d.height && f.fov === d.fov;
  return !framing || same(MESH_DEFAULT) || same(SPLAT_FRAMING) || same(SPLAT_FRAMING_OLD);
};

export const JAW = { knee: 0.35, slope: 0.35 };

// How far a LAM head turns before the side its portrait never showed comes into view: past about 20 degrees of yaw
// the far temple smears, and tipped back past about 10 a pale gap opens at the hairline; tipped forward it holds
// longer. Degrees; a manifest's `limits` (models.Limits) replaces any of them for its face.
export const TURN = { yaw: 20, pitchUp: 8, pitchDown: 20 };
const TURN_KNEE = 0.6;   // the share of a limit that passes untouched

// x eased towards ±max: unchanged up to TURN_KNEE of it, then slower and slower, never reaching max. Smooth at
// the knee (slope 1 on both sides), so a turn that crosses it shows no kink.
export function eased(x, max) {
  const a = Math.abs(x);
  const knee = TURN_KNEE * max;
  if (!(max > 0) || a <= knee) return x;
  const span = max - knee;
  return Math.sign(x) * (knee + span * Math.tanh((a - knee) / span));
}

// A head turn ({ yaw, pitch, roll }, degrees; positive pitch tips forward) held inside a face's limits. The roll is
// left alone: a tilt shows no new side. `limits` is the manifest's (snake case) or null; null fields keep TURN's.
// off: no limit at all (?turn=0, to see where a head gives out).
export function turnLimit(limits, off = false) {
  const l = limits || {};
  const yaw = l.yaw ?? TURN.yaw;
  const up = l.pitch_up ?? TURN.pitchUp;
  const down = l.pitch_down ?? TURN.pitchDown;
  return (head) => {
    const h = head || {};
    if (off) return { yaw: h.yaw || 0, pitch: h.pitch || 0, roll: h.roll || 0 };
    const pitch = h.pitch || 0;
    return { yaw: eased(h.yaw || 0, yaw), pitch: eased(pitch, pitch < 0 ? up : down), roll: h.roll || 0 };
  };
}

// Where a face's jaw cap sits: what was asked (?jaw=), else the manifest's, else JAW.knee.
export function jawKnee(limits, asked = NaN) {
  if (Number.isFinite(asked)) return Math.max(0, asked);   // below 0, the jaw would never open
  return limits && Number.isFinite(limits.jaw) ? limits.jaw : JAW.knee;
}

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
const WHITE = new THREE.Color(1, 1, 1);

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
  // limits: the manifest's (turnLimit). unlimited: the head turns as far as it is told.
  static async create(host, { url, framing, background = '#101216', skin, instant = false, limits = null, unlimited = false }) {
    const head = new SplatHead();
    head.limit = turnLimit(limits, unlimited);
    head.rest = skin.rest;
    head.expression = Object.fromEntries(ARKIT.map((n) => [n, 0]));
    head.turn = new THREE.Quaternion();
    head.ground = new THREE.Color(0x101216);
    try { head.ground.set(background); } catch { /* keep the default */ }
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
    // From here the renderer holds a canvas in host, a WebGL context and a sort worker: a head that fails to come
    // up (the 60 s load, above all) is disposed before the error goes on, or each retry would leak all three.
    try {
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
      head.framing = splatFraming(framing);
      head.ownFraming = !splatFraming.isDefault(framing);
      head.aim(head.framing);
      // create() resolves after the first section of a progressive load; the rest builds afterwards (each build
      // stops rendering until its sort is done). Wait for the last, or a recording's first frames race it.
      // The build only advances in the renderer's frame loop, cancelled above, so draw a frame per animation frame
      // until it is done: waiting without drawing hangs whenever the load had not finished by create()'s return.
      // A minute: a phone or a software renderer builds slowly. A timeout only; no pose depends on it.
      // Known: a second renderer created in the same page after one was disposed can still stall here, in headless
      // Chromium, its build waiting on a sort that never answers. A page that shows a splat face more than once keeps
      // its SplatHead (demo/player.js) rather than make another.
      const loaded = () => !v.isLoadingOrUnloading() && v.splatRenderReady && v.splatMesh?.finalBuild && v.splatMesh.getSplatCount() > 0;
      const start = performance.now();
      while (!loaded()) {
        if (performance.now() - start > 60000) throw new Error('The splat head did not load in 60 s.');
        head.fit();
        r.render();
        cancelAnimationFrame(v.requestFrameId);
        await nextFrame();
      }
    } catch (error) {
      head.dispose();
      throw error;
    }
    return head;
  }

  get canvas() { return this.r.viewer.renderer.domElement; }

  // Point the camera: at `height` from a tenth of its distance above it, as a portrait is taken.
  aim(f) {
    const camera = this.r.viewer.camera;
    camera.fov = f.fov;
    camera.position.set(0, f.height + LIFT * f.distance, f.distance);
    camera.lookAt(0, f.height, 0);
    camera.updateProjectionMatrix();
    camera.updateMatrixWorld(true);
    this.shot = f;
    this.veiled = null;   // the fade sits at a height in the head's space: place it again for this camera
  }

  // The soft edge under the bust (FADE): a strip across the bottom of the picture, drawn over the splats with the
  // same WebGL context, showing what is behind the head (the backdrop's own picture, or the background colour) and
  // fading in from nothing at FADE.top to all of it at FADE.bottom. Over a backdrop, the picture is dimmed under the
  // bust (SHADOW), deepest below the head and gone towards the sides: a bust fading into a bright floor read as a
  // pale band (#151), a shadow under it does not. Nothing in it reads a clock: it is placed from the camera alone,
  // so a recording has it exactly.
  veil() {
    const v = this.r.viewer;
    const picture = this.backdrop ? this.backdrop.texture : null;
    const rows = FADE_ROWS + 2;   // the fade's strips, and one row at the bottom of the picture
    if (!this.fade) {
      const n = rows * FADE_COLS;
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(n * 3), 3));
      geometry.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(n * 2), 2));
      geometry.setAttribute('color', new THREE.BufferAttribute(new Float32Array(n * 4), 4));
      const index = [];
      for (let i = 0; i < rows - 1; i += 1) {
        for (let j = 0; j < FADE_COLS - 1; j += 1) {
          const a = i * FADE_COLS + j, b = a + 1, c = a + FADE_COLS, d = c + 1;
          index.push(a, b, c, b, d, c);
        }
      }
      geometry.setIndex(index);
      const material = new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, depthTest: false, depthWrite: false });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.frustumCulled = false;
      const scene = new THREE.Scene();
      scene.add(mesh);
      this.fade = { scene, mesh, camera: new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1) };
    }
    const { scene, mesh, camera } = this.fade;
    if (this.veiled !== this.shot || this.veiledShadow !== Boolean(picture)) {
      // The two heights on screen (y, -1 to 1), on the head's own axis, and the bust's half width at its foot.
      const at = (y) => new THREE.Vector3(0, y, 0).project(v.camera).y;
      const top = at(FADE.top);
      const bottom = Math.min(top, at(FADE.bottom));
      const half = Math.abs(new THREE.Vector3(SHADOW.width, FADE.bottom, 0).project(v.camera).x);
      const ys = [-1.01];   // past the bottom edge: the strip below the fade is all background
      for (let i = 0; i <= FADE_ROWS; i += 1) ys.push(bottom + (top - bottom) * (i / FADE_ROWS));
      const alphas = [1];
      for (let i = 0; i <= FADE_ROWS; i += 1) { const k = 1 - i / FADE_ROWS; alphas.push(k * k * (3 - 2 * k)); }   // smoothstep
      // Across: full shade within the bust's width, easing to none at twice it (smoothstep), on a backdrop only.
      const shade = (x) => {
        if (!picture) return 1;
        const k = Math.min(1, Math.max(0, (Math.abs(x) - half) / half));
        return 1 - SHADOW.depth * (1 - k * k * (3 - 2 * k));
      };
      const { position, uv, color } = mesh.geometry.attributes;
      ys.forEach((y, i) => {
        const yy = i === 0 ? Math.min(y, ys[1]) : y;   // the first row never rises above the fade's foot
        for (let j = 0; j < FADE_COLS; j += 1) {
          const x = -1 + (2 * j) / (FADE_COLS - 1);
          const at3 = i * FADE_COLS + j;
          const f = shade(x);
          position.setXYZ(at3, x, yy, 0);
          uv.setXY(at3, (x + 1) / 2, (yy + 1) / 2);
          color.setXYZW(at3, f, f, f, alphas[i]);
        }
      });
      position.needsUpdate = uv.needsUpdate = color.needsUpdate = true;
      this.veiled = this.shot;
      this.veiledShadow = Boolean(picture);
    }
    const material = mesh.material;
    if (material.map !== picture) { material.map = picture; material.needsUpdate = true; }
    material.color.copy(picture ? WHITE : this.ground);
    // Drawn as three.js draws a scene's background: a picture already in the output's colours is not tone mapped.
    material.toneMapped = Boolean(picture) && THREE.ColorManagement.getTransfer(picture.colorSpace) !== THREE.SRGBTransfer;
    const r = v.renderer;
    const auto = r.autoClear;
    r.autoClear = false;
    r.render(scene, camera);
    r.autoClear = auto;
  }

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
  // Shapes the head lacks are ignored; names it has that the frame does not are 0. The turn is held inside the
  // face's limits here, so every page that draws a splat head (viewer, editor, demo, record) shows the same one.
  set(weights, names, head) {
    const e = this.expression;
    for (const n of ARKIT) e[n] = 0;
    for (let i = 0; i < names.length; i += 1) if (names[i] in e) e[names[i]] = weights[i];
    const rad = Math.PI / 180;
    const turn = this.limit(head);
    this.shown = turn;
    // As the mesh viewer turns its head: Euler YXZ of (pitch, yaw, roll), about world axes.
    _euler.set(turn.pitch * rad, turn.yaw * rad, turn.roll * rad, 'YXZ');
    this.turn.setFromEuler(_euler);
  }

  // A backdrop behind the head (backdrop.js: a scene name), or none ('' or null). Changes may overlap: only the
  // latest one stays. Resolves to the Backdrop shown (null for none); throws if the scene cannot be loaded.
  async setScene(name, { quality = 'medium', options = {} } = {}) {
    const v = this.r.viewer;
    const gen = (this.sceneLoads = (this.sceneLoads || 0) + 1);
    const made = name ? await Backdrop.create(THREE, v.renderer, name, { quality, options }) : null;
    if (gen !== this.sceneLoads) { made?.dispose(); return null; }
    this.backdrop?.dispose();
    this.backdrop = made;
    if (!this.anchor) { this.anchor = new THREE.Group(); v.threeScene.add(this.anchor); }
    this.anchor.visible = Boolean(made);   // the renderer draws its scene, background and all, only if this is on
    v.threeScene.background = made ? made.texture : null;
    this.aim(this.shotFraming());
    this.sized = null;
    this.fit();
    return made;
  }

  // The shot (backgrounds.SHOTS): 'close', head and shoulders ending above the fade; 'wide', the floating head with
  // room around it; '' or null, whichever suits the background (close over a backdrop, wide on a plain one).
  setShot(shot) {
    this.shotName = ['close', 'wide'].includes(shot) ? shot : '';   // not this.shot: aim() keeps the camera there
    this.aim(this.shotFraming());
    this.sized = null;
    this.fit();
  }

  // A framing the face's manifest chose for itself wins over any shot.
  shotFraming() {
    if (this.ownFraming) return this.framing;
    const close = this.shotName ? this.shotName === 'close' : Boolean(this.backdrop);
    return close ? SPLAT_CLOSE : this.framing;
  }

  // The canvas at its host's size (CSS pixels, as the renderer counts them), and the backdrop at the canvas's.
  fit() {
    const v = this.r.viewer;
    const size = v.renderer.getSize(new THREE.Vector2());
    const w = this.host.offsetWidth;
    const h = this.host.offsetHeight;
    if (w && h && (size.x !== w || size.y !== h)) v.renderer.setSize(w, h);
    if (this.backdrop) {
      const buffer = v.renderer.getDrawingBufferSize(new THREE.Vector2());
      const key = `${buffer.x}x${buffer.y}`;
      if (key !== this.sized) { this.backdrop.setSize(buffer.x, buffer.y); this.sized = key; }
    }
  }

  // A named bone from the skin (LAM: neckLower, neckUpper, head), or null.
  bone(name) {
    for (const [bone] of this.bones) {
      if (bone.name === name) return bone;
    }
    return null;
  }

  // Code-only accessory overlay (#67): drawn after the splats so it is not buried under the Gaussians.
  // *overlay* is a BoneOverlay from accessories.js (or any { draw(renderer, camera), dispose() }).
  setAccessoryOverlay(overlay) {
    this.accessoryOverlay?.dispose?.();
    this.accessoryOverlay = overlay || null;
  }

  _drawAccessories() {
    const overlay = this.accessoryOverlay;
    if (!overlay) return;
    const v = this.r.viewer;
    // Bones were posed just before the splat draw; refresh world matrices for the follow holders.
    v.boneRoot?.updateMatrixWorld(true);
    overlay.draw(v.renderer, v.camera);
  }

  // One frame, now: pose, skin, start a depth sort, draw (with the last finished sort: fine while live).
  // t: seconds on the backdrop's clock (the page's own, live).
  step(t = 0) {
    this.fit();
    this.backdrop?.update(t);
    this.r.render();
    cancelAnimationFrame(this.r.viewer.requestFrameId);
    this.veil();
    this._drawAccessories();
  }

  // Record mode: resolves once exactly the pose last set() is on the canvas, drawn with the depth sort for it and
  // this camera. The sort runs in a worker: wait for any in flight, start this pose's, wait for it, then draw.
  // t: the posed frame's time on the take, in seconds, for the backdrop: never the wall clock.
  async exact(t = 0) {
    const v = this.r.viewer;
    await until(() => !v.sortRunning, 'finish sorting');
    this.step(t);
    await until(() => !v.sortRunning, 'finish sorting');
    await nextFrame();
    this.fit();
    this.backdrop?.update(t, { exact: true });
    v.render();
    this.veil();
    this._drawAccessories();
  }

  dispose() {
    this.accessoryOverlay?.dispose?.();
    this.accessoryOverlay = null;
    this.backdrop?.dispose();
    if (this.fade) { this.fade.mesh.geometry.dispose(); this.fade.mesh.material.dispose(); }
    // The renderer takes its canvas off the page, then starts its viewer's async dispose, which removes the same
    // canvas from its root again: that throws (removeChild: not a child) before the WebGL renderer is disposed, so
    // each splat head shown would leak a context. Put the canvas back (hidden) for the viewer to remove itself.
    const viewer = this.r?.viewer;
    const canvas = viewer?.renderer?.domElement;
    const root = viewer?.rootElement;
    // The viewer's dispose first waits for its last sort, which settles only in a frame the page no longer draws: it
    // would wait for ever, the old renderer would stay alive, and the next splat head would stall beside it. Ending
    // the sort worker now settles it, so the dispose runs through.
    try { viewer?.disposeSortWorker?.(); } catch { /* no worker */ }
    try { this.r.dispose(); } catch { /* already gone */ }
    if (canvas && root && !canvas.parentNode && !viewer.usingExternalRenderer) {
      canvas.hidden = true;   // back only for the viewer to remove: never seen, if its dispose is slow or fails
      root.appendChild(canvas);
    }
  }
}
