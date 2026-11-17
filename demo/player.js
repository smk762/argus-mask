// The hosted demo's player (#118): a recorded take and its audio, replayed in the browser with the viewer's own
// modules (look, motion, rig, backdrop), so a clip looks as it does on a screen at home. No server, no socket:
// the take is a file, and the audio element is the clock. Pause, scrub and captions come with that.
//
// Layout of the built site (make demo): index.html, player.js, viewer/ (the package's viewer, unchanged),
// faces/<id>/ (the stock heads, from their release), clips/<id>/{clip.json, take.ndjson, audio.m4a}.
//
// A LAM Gaussian-splat head (manifest.model.format "lam-splat", #126) is drawn by the viewer's splat.js, as on a
// screen: the same frames, motion and lip budget, its own renderer and canvas over the stage while it is shown.

import * as THREE from 'three';
import { GLTFLoader } from './viewer/vendor/loaders/GLTFLoader.js';
import { Look, lookOptions } from './viewer/look.js';
import { ARKIT, Motion, interpolate, lipBudget } from './viewer/motion.js';
import { Backdrop } from './viewer/backdrop.js';
import { ARMS_DOWN, BodyLife, Gestures, HeadTurn, STANDING, contactShadow, setPose } from './viewer/rig.js';

const BACKGROUND = '#101216';

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const stageEl = $('stage');
const audio = $('audio');
const playEl = $('play');
const scrubEl = $('scrub');
const timeEl = $('time');
const captionEl = $('caption');
const transcriptEl = $('transcript');
const madeEl = $('made');
const madeLineEl = $('made-line');
const statusEl = $('status');

// Each face idles its own way, as on a screen (viewer.js: faceSeed).
function faceSeed(id) {
  let seed = 7;
  for (const ch of id) seed = (seed * 31 + ch.charCodeAt(0)) >>> 0;
  return seed;
}

async function json(url) {
  const reply = await fetch(url);
  if (!reply.ok) throw new Error(`${url}: ${reply.status}`);
  return reply.json();
}

// A take file: a header line, a frame per line, an end line (take.py).
async function readTake(url) {
  const reply = await fetch(url);
  if (!reply.ok) throw new Error(`${url}: ${reply.status}`);
  const lines = (await reply.text()).split('\n').filter((l) => l.trim()).map((l) => JSON.parse(l));
  const header = lines.find((m) => m.type === 'header');
  const frames = lines.filter((m) => m.type === 'frame').map((f) => ({ ...f, weights: Float32Array.from(f.weights) }));
  return { header, frames };
}

// A clip's audio, whole, as a blob URL. A static host may answer a byte-range request with the whole file
// (Cloudflare Pages does), and then a browser will not seek in it: scrubbing would snap back to the start. A blob
// is always seekable, and a clip's audio is about 100 KB.
async function audioUrl(url) {
  const reply = await fetch(url);
  if (!reply.ok) throw new Error(`${url}: ${reply.status}`);
  return URL.createObjectURL(await reply.blob());
}

// ---- the stage: one renderer for the page, a face and backdrop per clip ----------------------------------

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.outputColorSpace = THREE.SRGBColorSpace;
stageEl.prepend(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(30, 1, 0.01, 100);
const look = new Look(renderer, scene, camera, lookOptions(params, 'medium'));

let shown = null;   // { clip, take, motion, names, current } and, for a mesh: root, head, body, gestures, backdrop,
                    // extras, targets; for a splat head: splat, host

function size() {
  const { width, height } = stageEl.getBoundingClientRect();
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(width, height, false);
  camera.aspect = width / Math.max(1, height);
  camera.updateProjectionMatrix();
  look.setSize(width, height);
  shown?.backdrop?.setSize(renderer.domElement.width, renderer.domElement.height);
}
new ResizeObserver(size).observe(stageEl);

function dispose(root) {
  root.traverse((node) => {
    if (!node.isMesh) return;
    node.geometry?.dispose();
    for (const m of [].concat(node.material || [])) {
      for (const v of Object.values(m)) if (v && v.isTexture && !v.userData.shared) v.dispose();
      m.dispose();
    }
  });
}

function clear() {
  if (!shown) return;
  if (shown.splat) {
    shown.host.hidden = true;   // kept for the next clip on this face (loadSplat)
    renderer.domElement.hidden = false;
  } else {
    for (const node of [shown.root, ...shown.extras]) { scene.remove(node); dispose(node); }
    shown.backdrop?.dispose();
    scene.background = null;
  }
  shown = null;
}

// Each load has a token; a click while one is in flight starts another, and only the latest may touch the stage.
// After every wait a load that is no longer the latest gives back what it made and stops.
let loadSeq = 0;
const stale = (token) => token !== loadSeq;

// Show this splat head's canvas (or, with none, the mesh renderer's) and hide every other one, including a head
// still being made for a load that was overtaken.
function showOnly(host) {
  for (const h of stageEl.querySelectorAll(':scope > .splat')) h.hidden = h !== host;
  renderer.domElement.hidden = Boolean(host);
}

const faces = new Map();   // id -> manifest
async function load(clipId, token) {
  statusEl.textContent = 'Loading…';
  const base = `clips/${clipId}/`;
  const [clip, take, sound] = await Promise.all([json(`${base}clip.json`), readTake(`${base}take.ndjson`), audioUrl(`${base}audio.m4a`)]);
  if (stale(token)) { URL.revokeObjectURL(sound); return; }
  if (!faces.has(clip.face)) faces.set(clip.face, await json(`faces/${clip.face}/manifest.json`));
  if (stale(token)) { URL.revokeObjectURL(sound); return; }
  const manifest = faces.get(clip.face);
  if (manifest.model?.format === 'lam-splat') return loadSplat(clip, take, sound, manifest, token);
  const framed = clip.framing !== 'head' && manifest.framings?.[clip.framing];
  const model = framed ? framed.model : manifest.model;
  const framing = (framed ? framed.framing : manifest.framing) || {};
  const gltf = await new GLTFLoader().loadAsync(`faces/${clip.face}/${model.file}`);
  const backdrop = clip.scene && !stale(token) ? await Backdrop.create(THREE, renderer, clip.scene, { quality: look.summary.quality }) : null;
  if (stale(token)) { dispose(gltf.scene); backdrop?.dispose(); URL.revokeObjectURL(sound); return; }

  clear();
  showOnly(null);
  const root = gltf.scene;
  scene.add(root);
  look.apply(root);
  const seed = faceSeed(manifest.id);
  const extras = [];
  let body = null;
  let gestures = null;
  if (framed) {
    setPose(THREE, root, clip.framing === 'full' ? STANDING : ARMS_DOWN);
    body = new BodyLife(THREE, root, seed);
    gestures = new Gestures(seed);
    for (const f of take.frames) gestures.add(f.t_ms, f.cue);
    gestures.end(take.frames.at(-1)?.t_ms ?? 0);
    if (clip.framing === 'full') { const card = contactShadow(THREE, root, scene); if (card) extras.push(card); }
  }
  const targets = new Map();
  root.traverse((node) => {
    if (!node.isMesh || !node.morphTargetDictionary) return;
    for (const [name, index] of Object.entries(node.morphTargetDictionary)) {
      if (!targets.has(name)) targets.set(name, []);
      targets.get(name).push({ influences: node.morphTargetInfluences, index });
    }
  });
  const motion = new Motion({ budget: lipBudget(root), seed, rest: clip.rest?.weights || {}, life: clip.rest?.life ?? 1 });
  camera.fov = framing.fov || 30;
  camera.position.set(0, framing.height || 0, framing.distance || 0.6);
  camera.lookAt(0, framing.height || 0, 0);
  if (backdrop) { scene.background = backdrop.texture; look.tintRims(backdrop.meta.tint); }
  else scene.background = new THREE.Color('#101216');
  shown = { clip, take, root, motion, head: new HeadTurn(THREE, root), body, gestures, backdrop, extras, targets,
    names: take.header.blendshapes, current: new Float32Array(take.header.blendshapes.length) };
  size();
  look.focus(root);
  ready(clip, manifest, sound);
}

function ready(clip, manifest, sound) {
  describe(clip, manifest);
  if (audio.src.startsWith('blob:')) URL.revokeObjectURL(audio.src);
  audio.src = sound;
  scrubEl.max = String(clip.duration_ms);
  scrubEl.value = '0';
  statusEl.textContent = '';
}

// A splat head: its own renderer, on a canvas over the stage (splat.js). Each splat face is made once and kept for
// the page's life, hidden while a mesh clip plays: a second renderer made in the same page after one was disposed can
// stall while it loads, and the demo shows the same face in more than one clip.
// The map holds the promise of the head, so two loads of the same face at once share one head, never make two.
const splats = new Map();   // face id -> Promise<{ splat, host, skin }>
async function makeSplat(face, manifest, token, { SplatHead, readSkin }) {
  const url = `faces/${face}/${manifest.model.file}`;   // the renderer wants a URL ending .zip
  const skin = await readSkin(url);
  if (!stale(token)) clear();
  const host = document.createElement('div');
  host.className = 'splat';
  host.hidden = stale(token);
  stageEl.prepend(host);
  try {
    return { splat: await SplatHead.create(host, { url, framing: manifest.framing, background: BACKGROUND, skin }), host, skin };
  } catch (error) {   // leave the stage as it was: no half-built head
    host.remove();
    throw error;
  }
}

async function loadSplat(clip, take, sound, manifest, token) {
  const mod = await import('./viewer/splat.js');
  const { jawCap, JAW } = mod;
  let making = splats.get(clip.face);
  if (!making) {
    making = makeSplat(clip.face, manifest, token, mod);
    splats.set(clip.face, making);
    making.catch(() => { if (splats.get(clip.face) === making) splats.delete(clip.face); });   // a later click retries
  }
  const kept = await making;
  // Overtaken: the head is kept for later (it is in the map); the latest load's showOnly decides what is seen.
  const leave = () => URL.revokeObjectURL(sound);
  if (stale(token)) return leave();
  await kept.splat.setScene(clip.scene || null, { quality: look.summary.quality });
  if (stale(token)) return leave();
  clear();
  showOnly(kept.host);
  await kept.splat.exact();   // a frame sorted for this canvas and camera before the clip shows: no stale depth order
  if (stale(token)) return leave();   // the latest load has the stage, or will clear it
  const motion = new Motion({ budget: kept.skin.budget, seed: faceSeed(manifest.id), rest: clip.rest?.weights || {}, life: clip.rest?.life ?? 1, cap: jawCap(JAW.knee) });
  shown = { clip, take, splat: kept.splat, host: kept.host, motion, names: take.header.blendshapes, current: new Float32Array(take.header.blendshapes.length) };
  ready(clip, manifest, sound);
}

// ---- words: captions, the transcript and what made the clip ------------------------------------------------

function describe(clip, manifest) {
  transcriptEl.replaceChildren(...clip.captions.map((c, i) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'line';
    b.textContent = c.text;
    b.dataset.index = String(i);
    b.addEventListener('click', () => { audio.currentTime = c.start_ms / 1000; audio.play().catch(() => {}); });
    return b;
  }));
  const m = clip.made_with;
  const driver = m.driver.name === 'a2f' ? `NVIDIA Audio2Face-3D${m.driver.model ? ` (${m.driver.model})` : ''}` : `${m.driver.name} (a stand-in, not the demo's driver)`;
  madeEl.replaceChildren(...[
    ['Face', manifest.model?.format === 'lam-splat'
      ? `${manifest.name}, a Gaussian-splat head made with LAM from a synthetic portrait`
      : `${manifest.name}, built with argus-mask's face editor`],
    ['Framing', clip.framing === 'head' ? 'head and shoulders' : clip.framing === 'waist' ? 'waist up' : 'full body'],
    ['Voice', `${m.voice} (XTTS-v2)`],
    ['Animation', `${driver}, ${m.fps} frames a second; blinks, glances and gestures from seed ${m.seed}`],
    ['Version', `argus-mask ${m.argus_mask}`],
    ['Backdrop', clip.scene ? `${clip.scene} (built in)` : 'none'],
  ].flatMap(([k, v]) => { const dt = document.createElement('dt'); dt.textContent = k; const dd = document.createElement('dd'); dd.textContent = v; return [dt, dd]; }));
  // "Made with", from the clip's own record: the driver, the clip and the argus-mask that made it (#123).
  const docs = document.createElement('a');
  docs.href = 'docs/#hosted-demo';
  docs.textContent = 'Docs';
  const driverName = m.driver.name === 'a2f' ? 'Audio2Face' : m.driver.name;
  madeLineEl.replaceChildren(`Made with ${driverName} · “${clip.title}” · argus-mask ${m.argus_mask} · `, docs);
}

function showCaption(tMs) {
  const caps = shown?.clip.captions || [];
  const i = caps.findIndex((c) => tMs >= c.start_ms && tMs < c.end_ms + 250);
  const text = i >= 0 ? caps[i].text : '';
  if (captionEl.textContent !== text) captionEl.textContent = text;
  for (const b of transcriptEl.children) b.classList.toggle('on', Number(b.dataset.index) === i);
}

// ---- drawing: the audio element is the clock ---------------------------------------------------------------

const headTurn = { yaw: 0, pitch: 0, roll: 0 };
function frameIndex(frames, tMs) {   // the last frame at or before tMs
  let lo = 0, hi = frames.length - 1;
  while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (frames[mid].t_ms <= tMs) lo = mid; else hi = mid - 1; }
  return lo;
}

let last = performance.now();
function draw() {
  requestAnimationFrame(draw);
  const wall = performance.now();
  const dt = Math.min(0.1, (wall - last) / 1000);
  last = wall;
  if (!shown) return;
  const s = shown;
  const tMs = audio.currentTime * 1000;
  const frames = s.take.frames;
  // Live while there is a frame for this moment: playing or paused part-way. Before and after, the face rests.
  const live = frames.length > 0 && tMs > 0 && tMs <= frames.at(-1).t_ms && !audio.ended;
  if (live) {
    const i = frameIndex(frames, tMs);
    const a = frames[i], b = frames[i + 1];
    if (b) {
      const k = Math.min(1, Math.max(0, (tMs - a.t_ms) / Math.max(1, b.t_ms - a.t_ms)));
      interpolate(frames[i - 1] || null, a, b, frames[i + 2] || null, k, s.current, headTurn);
    } else { s.current.set(a.weights); Object.assign(headTurn, a.head); }
  } else {
    const keep = Math.exp(-6 * dt);
    for (let j = 0; j < s.current.length; j += 1) s.current[j] *= keep;
  }
  const out = s.motion.present(s.current, s.names, { live, t: wall / 1000, dt });
  if (!live && out.head) { const k = 1 - Math.exp(-6 * dt); for (const ax of ['yaw', 'pitch', 'roll']) headTurn[ax] += (out.head[ax] - headTurn[ax]) * k; }
  if (s.splat) {
    s.splat.set(out.weights, s.names, headTurn);
    s.splat.step(wall / 1000);
  } else {
    for (let j = 0; j < s.names.length; j += 1) {
      for (const slot of s.targets.get(s.names[j]) || []) slot.influences[slot.index] = out.weights[j];
    }
    s.body?.apply(wall / 1000, headTurn.pitch, s.gestures?.pose(tMs));
    if (s.head.rigged) s.head.apply(headTurn.pitch, headTurn.yaw, headTurn.roll);
    s.backdrop?.update(wall / 1000);
    look.render();
  }
  if (!scrubbing) scrubEl.value = String(Math.round(tMs));
  timeEl.textContent = `${clock(tMs)} / ${clock(s.clip.duration_ms)}`;
  showCaption(tMs);
}
const clock = (ms) => { const sec = Math.max(0, Math.round(ms / 1000)); return `${Math.floor(sec / 60)}:${String(sec % 60).padStart(2, '0')}`; };

// ---- controls ----------------------------------------------------------------------------------------------

let scrubbing = false;
playEl.addEventListener('click', () => { if (audio.paused) audio.play().catch(() => {}); else audio.pause(); });
const playing = () => {
  playEl.setAttribute('aria-pressed', String(!audio.paused));
  playEl.querySelector('.label').textContent = audio.paused ? 'Play' : 'Pause';
};
audio.addEventListener('play', playing);
audio.addEventListener('pause', playing);
audio.addEventListener('ended', playing);
scrubEl.addEventListener('input', () => { scrubbing = true; audio.currentTime = Number(scrubEl.value) / 1000; });
scrubEl.addEventListener('change', () => { scrubbing = false; });
document.addEventListener('keydown', (e) => {
  if (e.key === ' ' && !(e.target instanceof HTMLButtonElement) && !(e.target instanceof HTMLInputElement)) {
    e.preventDefault();
    playEl.click();
  }
});

async function choose(clipId, button) {
  audio.pause();
  for (const b of document.querySelectorAll('#clips button')) b.setAttribute('aria-pressed', String(b === button));
  const token = ++loadSeq;
  try { await load(clipId, token); } catch (error) {
    if (!stale(token)) statusEl.textContent = `This clip did not load: ${error.message}`;
  }
  playing();
}

// The credits dialog opens and closes without script (command="show-modal", closedby="any", form method="dialog").
// Where a browser lacks those, the same behaviour from here: the page's CSP allows no polyfill from elsewhere.
function creditsFallbacks() {
  const dialog = $('credits');
  if (!dialog) return;
  if (!('commandForElement' in HTMLButtonElement.prototype)) {
    for (const b of document.querySelectorAll('button[commandfor="credits"][command="show-modal"]')) {
      b.addEventListener('click', () => dialog.showModal());
    }
  }
  if (!('closedBy' in HTMLDialogElement.prototype)) {
    // The pointer's position, not the event's target: the dialog is also the target for its own padding.
    const outside = (e) => {
      const r = dialog.getBoundingClientRect();
      return e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom;
    };
    let pressedOutside = false;
    dialog.addEventListener('pointerdown', (e) => { pressedOutside = outside(e); });
    dialog.addEventListener('pointerup', (e) => { if (pressedOutside && outside(e)) dialog.close(); });
  }
}
creditsFallbacks();

async function main() {
  const ids = await json('clips/index.json');
  const list = $('clips');
  const clips = await Promise.all(ids.map((id) => json(`clips/${id}/clip.json`)));
  for (const clip of clips) {
    const b = document.createElement('button');
    b.type = 'button';
    b.textContent = clip.title;
    b.setAttribute('aria-pressed', 'false');
    b.addEventListener('click', () => choose(clip.id, b));
    list.append(b);
  }
  const first = params.get('clip') && ids.includes(params.get('clip')) ? params.get('clip') : ids[0];
  await choose(first, [...list.children][ids.indexOf(first)]);
  draw();
}

main().catch((error) => { statusEl.textContent = `The demo could not start: ${error.message}`; });
