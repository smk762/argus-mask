// The argus-mask viewer: one rigged head, driven by frames from the server and kept in step
// with the audio by this page's own audio clock. A LAM Gaussian-splat head (manifest.model.format
// "lam-splat") is drawn by splat.js instead of three.js's mesh path, through the same frames and motion.
//
//   ?face=<id>       which face (default: the first the server lists)
//   ?display=flat    flat (default) | hld (1080x1920 portrait on black) | go (not built yet)
//   ?bg=<css colour> background for the flat tier, or "transparent"
//   ?scene=ftl       a backdrop with a little life in it, flat tier, mesh faces (backdrop.js: SCENES), over ?bg=
//                    ?scene.speed=1 ?scene.density=0.6 ?scene.tint=<css colour> tune it
//   ?hud=0           hide the status line
//   ?buffer=250      how far behind arrival the audio plays, in ms (more absorbs jitter, less feels snappier)
//   ?record=1        posed from outside (`argus-mask record`): no socket, no sound, no notice
//   ?kiosk=1         a dedicated screen: no HUD, no cursor, full screen on the first tap or key press
//                    (start Chromium with --kiosk --autoplay-policy=no-user-gesture-required for no tap at all)
//   ?audio=external  the sound plays elsewhere (house speakers, Snapcast): this page plays none, needs no tap
//                    for sound, and moves the face on the session's own timeline. ?mute=1 is the same.
//   ?delay=0         external sound only: show each frame this many ms later (0-5000), to meet speakers
//                    that play behind (Snapcast's buffer is 1000 ms by default)
//   ?quality=medium  low | medium (default) | high: the look's profile (see look.js); record uses high
//   ?ao= ?shadows= ?skin= ?cornea= ?ibl= ?smaa= ?dof= ?grain= ?bloom=   1 or 0, over the profile
//   ?hair=a2c        a2c (alpha to coverage) | hash | cutout: how hair cards meet the background
//   ?tone=agx        agx | aces | neutral | none, with ?exposure=1.0
//   ?rest=<name>     the rest set to show instead of the face's own (warm, calm, neutral, ...: GET /expressions)
//   ?life=1          how much the face moves at rest: 0 still, up to 2 (default: the rest set's own)
//   ?jaw=0.35        splat heads: where the jaw cap's knee sits (1 turns the cap off)
//
// Socket URLs come from the page's own origin, so the same page works behind a TLS proxy.

import * as THREE from 'three';
import { GLTFLoader } from './vendor/loaders/GLTFLoader.js';
import { Look, lookOptions } from './look.js';
import { ARKIT, AXES, Motion, interpolate, lipBudget } from './motion.js';
import { Backdrop } from './backdrop.js';

const params = new URLSearchParams(location.search);
const DISPLAY = params.get('display') || 'flat';
const SCENE = params.get('scene') || '';
const RECORD = params.get('record') === '1';
const KIOSK = params.get('kiosk') === '1';
const BUFFER_MS = params.has('buffer') ? Number(params.get('buffer')) : 250;
const BUFFER_S = Math.min(2, Math.max(0.05, (Number.isFinite(BUFFER_MS) ? BUFFER_MS : 250) / 1000));  // audio plays this far behind arrival; absorbs jitter
// Sound elsewhere: no AudioContext at all, so no gesture is needed and nothing reaches this page's speakers.
// The clock is then the page's own (performance.now()), anchored to the stream as the audio clock would be.
const EXTERNAL = params.get('audio') === 'external' || params.get('mute') === '1';
const DELAY_MS = Number(params.get('delay') || 0);
const DELAY_S = EXTERNAL ? Math.min(5, Math.max(0, (Number.isFinite(DELAY_MS) ? DELAY_MS : 0) / 1000)) : 0;
const RESYNC_S = 0.05;        // after an underrun, how soon the late audio starts
const RELAX_PER_S = 6;        // how fast the face settles when a stream ends
const HLD = { width: 1080, height: 1920 };

const stageEl = document.getElementById('stage');
const hudEl = document.getElementById('hud');
const noticeEl = document.getElementById('notice');
const noticeText = document.getElementById('notice-text');
const noticeButton = document.getElementById('notice-button');
if (params.get('hud') === '0' || KIOSK) hudEl.hidden = true;
if (KIOSK) document.body.classList.add('kiosk');
const startEl = document.getElementById('start');
const readyEl = document.getElementById('ready');

// What a test (or a curious operator in the console) can read.
const state = window.__mask = {
  face: null, ready: false, connected: false, session: null, frames: 0, audioChunks: 0,
  renderer: null,   // "mesh" (three.js) or "splat" (splat.js)
  streamsEnded: 0, underruns: 0, audioDropped: 0, lastSeq: -1, weights: null, error: null, audio: 'locked',
  look: null,   // the profile and effects in use: see look.js
  scene: null,  // the backdrop on show (backdrop.js), or null for a plain background
  screenReady: false, kiosk: KIOSK,   // connected and heard (or, sound external, connected); a dedicated screen
  sound: EXTERNAL ? 'external' : 'local', delayMs: DELAY_S * 1000,
};
if (EXTERNAL) state.audio = 'external';

function notice(text, button, onClick) {
  noticeText.textContent = text;
  noticeButton.hidden = !button;
  if (button) { noticeButton.textContent = button; noticeButton.onclick = onClick; }
  noticeEl.classList.toggle('show', Boolean(text));
}

function fail(text) { state.error = text; notice(text); }

// `argus-mask record` drives the face frame by frame: pose() shows one frame's weights and head and
// resolves once it has been drawn, so a screenshot taken then is exactly that frame.
let posed = null;
let posedResolve = null;
// tMs, the frame's own time on the take, sets the backdrop's clock: a recording never reads the wall clock.
state.pose = (weights, headTurn, blendshapes, tMs) => new Promise((resolve) => {
  if (blendshapes) names = blendshapes;   // the stream's order, from its header
  posed = { weights: Float32Array.from(weights), head: headTurn, t: Number.isFinite(tMs) ? tMs / 1000 : 0 };
  posedResolve = resolve;
});

// ---- scene ---------------------------------------------------------------------------------

// The mesh path's three.js scene, made once the face is known to be a mesh (a splat head draws its own).
let renderer = null;
let scene = null;
let camera = null;
let look = null;
let pivot = null;
let splat = null;          // the SplatHead (splat.js), for a splat face
let backdrop = null;       // the Backdrop (backdrop.js), for ?scene=
async function meshScene() {
  const transparent = DISPLAY === 'flat' && params.get('bg') === 'transparent';
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: transparent });
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  stageEl.appendChild(renderer.domElement);
  scene = new THREE.Scene();
  if (DISPLAY === 'hld') {
    scene.background = new THREE.Color(0x000000);   // the panel wants its subject on black
    document.body.style.background = '#000';
  } else if (transparent) {
    document.body.classList.add('transparent');
  } else {
    const colour = params.get('bg') || '#101216';
    try { scene.background = new THREE.Color(colour); document.body.style.background = colour; } catch { /* keep the default */ }
  }
  camera = new THREE.PerspectiveCamera(30, 1, 0.01, 100);
  // Lights, environment, tone mapping, materials and post: a recording renders offline, so it gets the most.
  look = new Look(renderer, scene, camera, lookOptions(params, RECORD ? 'high' : 'medium'));
  pivot = new THREE.Group();   // head rotation is applied here unless the model has a head bone
  scene.add(pivot);
  // HLD keeps its subject on black and a transparent page shows what is behind it: neither takes a scene.
  if (SCENE && DISPLAY === 'flat' && !transparent) {
    const options = {};
    for (const key of ['speed', 'density']) if (params.has(`scene.${key}`)) options[key] = Number.parseFloat(params.get(`scene.${key}`));
    if (params.has('scene.tint')) options.tint = params.get('scene.tint');
    backdrop = await Backdrop.create(THREE, renderer, SCENE, { quality: look.summary.quality, options });
    scene.background = backdrop.texture;
    look.tintRims(options.tint || backdrop.meta.tint);
  }
}

function resize() {
  if (!renderer) return;   // a splat head sizes itself to the page
  let width = window.innerWidth, height = window.innerHeight;
  if (DISPLAY === 'hld') {
    // Render at the panel's own size and scale the canvas to fit whatever window this is.
    renderer.setPixelRatio(1);
    renderer.setSize(HLD.width, HLD.height, false);
    const scale = Math.min(width / HLD.width, height / HLD.height);
    renderer.domElement.style.width = `${HLD.width * scale}px`;
    renderer.domElement.style.height = `${HLD.height * scale}px`;
    camera.aspect = HLD.width / HLD.height;
    look.setSize(HLD.width, HLD.height);
  } else {
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(width, height);
    camera.aspect = width / Math.max(1, height);
    look.setSize(width, height);
    backdrop?.setSize(renderer.domElement.width, renderer.domElement.height);
  }
  camera.updateProjectionMatrix();
  state.look = look.summary;
  state.scene = backdrop ? backdrop.summary : null;
}
window.addEventListener('resize', resize);

// ---- the face ------------------------------------------------------------------------------

let targets = new Map();   // blendshape name -> [{ influences, index }]
let motion = new Motion();  // the lip budget, the rest set and idle life (motion.js)
// ?life= over the rest set's own life; absent or not a number, the set's.
const LIFE = (() => { const v = Number.parseFloat(params.get('life')); return Number.isFinite(v) ? Math.min(2, Math.max(0, v)) : null; })();
let headBone = null;
let headRest = null;

async function loadFace() {
  let faceId = params.get('face');
  if (!faceId) {
    const faces = await (await fetch('/faces')).json();
    if (!faces.length) throw new Error('This server has no faces to show. Add one under its assets directory.');
    faceId = faces[0].id;
  }
  const reply = await fetch(`/face/${encodeURIComponent(faceId)}`);
  if (!reply.ok) throw new Error(`There is no face "${faceId}" on this server.`);
  const manifest = await reply.json();
  if (manifest.model && manifest.model.format === 'lam-splat') return loadSplat(faceId, manifest);
  state.renderer = 'mesh';
  await meshScene();
  resize();
  const gltf = await new GLTFLoader().loadAsync(`/face/${encodeURIComponent(faceId)}/model`);
  pivot.add(gltf.scene);
  look.apply(gltf.scene);
  gltf.scene.traverse((node) => {
    if (node.isMesh && node.morphTargetDictionary) {
      for (const [name, index] of Object.entries(node.morphTargetDictionary)) {
        if (!targets.has(name)) targets.set(name, []);
        targets.get(name).push({ influences: node.morphTargetInfluences, index });
      }
    }
    if (node.isBone && !headBone && /^head$/i.test(node.name)) headBone = node;
  });
  if (headBone) headRest = headBone.quaternion.clone();
  motion = new Motion({ budget: lipBudget(gltf.scene), seed: faceSeed(manifest.id), life: LIFE ?? 1 });
  state.lips = motion.budget;
  await loadRest(faceId);
  const framing = manifest.framing || {};
  camera.fov = framing.fov || 30;
  camera.position.set(0, framing.height || 0, framing.distance || 0.6);
  camera.lookAt(0, framing.height || 0, 0);
  resize();
  look.focus(gltf.scene);
  state.face = faceId;
  state.faceShown = manifest.id;   // the face itself: faceId may be an alias for it
  state.ready = true;
  return faceId;
}

// Each face idles its own way: the idle seed, from its id.
function faceSeed(id) {
  let seed = 7;
  for (const ch of id) seed = (seed * 31 + ch.charCodeAt(0)) >>> 0;
  return seed;
}

// A LAM Gaussian-splat head (splat.js): the same frames, rest set, idle life and lip budget, drawn by the
// splat renderer. The archive is fetched by its own name: the renderer wants a URL ending .zip.
async function loadSplat(faceId, manifest) {
  if (DISPLAY === 'hld') throw new Error('A splat face cannot be shown on display=hld yet. Use display=flat.');
  if (SCENE) throw new Error('A splat face cannot show a scene yet: it draws its own background. Leave out ?scene=.');
  state.renderer = 'splat';
  const { SplatHead, readSkin, jawCap, JAW } = await import('./splat.js');
  const url = `/face/${encodeURIComponent(faceId)}/model/${encodeURIComponent(manifest.model.file)}?v=${manifest.model.sha256}`;
  const skin = await readSkin(url);
  const host = document.createElement('div');
  host.style.cssText = 'position: fixed; inset: 0;';
  stageEl.appendChild(host);
  let colour = params.get('bg') || '#101216';
  if (colour === 'transparent') colour = '#101216';   // the splat renderer draws on an opaque canvas
  try { new THREE.Color(colour); document.body.style.background = colour; } catch { colour = '#101216'; }
  splat = await SplatHead.create(host, { url, framing: manifest.framing, background: colour, skin, instant: RECORD });
  const asked = Number.parseFloat(params.get('jaw'));
  const knee = Number.isFinite(asked) ? Math.max(0, asked) : JAW.knee;   // below 0, the jaw would never open
  motion = new Motion({ budget: skin.budget, seed: faceSeed(manifest.id), life: LIFE ?? 1, cap: jawCap(knee) });
  state.lips = motion.budget;
  state.jaw = knee;
  state.look = null;
  await loadRest(faceId);
  // The face at rest, drawn once before anyone is told it is ready.
  const order = ARKIT;
  const shown = motion.present(new Float32Array(order.length), order, { live: true, t: 0, dt: 0 });
  splat.set(shown.weights, order, head);
  await splat.exact();
  state.weights = shown.weights;
  if (RECORD) state.pose = splatPose;   // no draw loop: each pose is drawn when asked, and only then
  state.face = faceId;
  state.faceShown = manifest.id;
  state.ready = true;
  return faceId;
}

// Record mode on a splat head: the pose goes straight through motion.js to the splat renderer, and resolves
// only once exactly that pose, depth-sorted for this camera, is on the canvas (splat.js exact()).
function splatPose(weights, headTurn, blendshapes) {
  if (blendshapes) names = blendshapes;
  const order = names.length ? names : ARKIT;
  current = Float32Array.from(weights);
  Object.assign(head, headTurn);
  const shown = motion.present(current, order, { live: true, t: 0, dt: 0 });
  state.weights = shown.weights;
  splat.set(shown.weights, order, head);
  return splat.exact();
}

// The face's rest set: its own, or the page's ?rest=. A server without the route leaves the face neutral.
async function loadRest(faceId) {
  const q = params.get('rest') ? `?set=${encodeURIComponent(params.get('rest'))}` : '';
  try {
    const reply = await fetch(`/face/${encodeURIComponent(faceId)}/rest${q}`);
    if (!reply.ok) return;
    const rest = await reply.json();
    motion.setRest(rest.weights);
    if (LIFE === null && Number.isFinite(rest.life)) motion.idle.life = rest.life;   // the set's own, unless ?life=
    state.rest = rest.name;
  } catch { /* neutral */ }
}

// ---- audio clock ---------------------------------------------------------------------------

let audio = null;          // the AudioContext, once the browser allows one
let anchor = null;         // { clock, t }: clock time (s) at which timeline position t (ms) plays
let nextPlay = 0;          // where the next audio chunk goes on the audio clock
let sampleRate = 24000;

function clockNow() {
  return audio && audio.state === 'running' ? audio.currentTime : performance.now() / 1000;
}

function unlockAudio() {
  if (EXTERNAL) return;   // the sound plays elsewhere: nothing to unlock
  try {
    audio = audio || new (window.AudioContext || window.webkitAudioContext)();
  } catch { state.audio = 'unavailable'; return; }
  const settle = () => {
    const was = state.audio;
    state.audio = audio.state === 'running' ? 'running' : 'locked';
    if (audio.state === 'running' && was !== 'running') {
      startEl.classList.remove('show');
      anchor = null;   // re-anchor on the audio clock
      announce();
    }
    if (state.audio !== was) report();
  };
  audio.onstatechange = settle;   // suspended later (an interruption, a hidden tab): say so, not "ready"
  audio.resume().then(settle, settle);
  settle();
}

function playChunk(startMs, samples) {
  state.audioChunks += 1;
  const now = clockNow();
  if (!anchor) { anchor = { clock: now + BUFFER_S + DELAY_S, t: startMs }; nextPlay = anchor.clock; }
  let when = anchor.clock + (startMs - anchor.t) / 1000;
  if (when < now) {
    // The audio arrived too late to play on time: slide the whole timeline, mouth included.
    state.underruns += 1;
    anchor = { clock: now + RESYNC_S, t: startMs };
    when = anchor.clock;
  }
  if (audio && audio.state !== 'running' && samples.length) {
    // The browser paused the context (a tab switch, an interrupted output device). The chunk is
    // lost; count it so a clipped start can be told from a clipped reply, and ask for the clock back.
    state.audioDropped += 1;
    audio.resume().catch(() => {});
  }
  if (audio && audio.state === 'running' && samples.length) {
    const buffer = audio.createBuffer(1, samples.length, sampleRate);
    buffer.copyToChannel(samples, 0);
    const node = audio.createBufferSource();
    node.buffer = buffer;
    node.connect(audio.destination);
    node.start(when);
  }
}

function timelineNow() {
  return anchor ? anchor.t + (clockNow() - anchor.clock) * 1000 : null;
}

// ---- the stream ----------------------------------------------------------------------------

let names = [];            // blendshape order of the current stream
let queue = [];            // frames not yet passed, by t_ms
let passed = null;         // the frame before queue[0], for a curve through it
let ended = true;
let current = null;        // Float32Array of the weights on screen
const head = { yaw: 0, pitch: 0, roll: 0 };
state.head = head;        // the head turn on screen, degrees, updated in place: for checking motion from outside

// Reload once per stream: a page that joins a stream begun before a repoint is sent that stream's
// header (the old face) after loading the new one, and reloading again would loop until it ends.
function reloadOnce(session) {
  try {
    if (sessionStorage.getItem('mask.reloadedFor') === session) return false;
    sessionStorage.setItem('mask.reloadedFor', session);
  } catch { return false; }  // no storage: play the stream on the face shown rather than risk a loop
  location.reload();
  return true;
}

function onHeader(header) {
  // An alias (?face=holly) repointed to another face since this page loaded it: load the new one.
  // The stream goes on meanwhile, and the reloaded page joins it where it has got to.
  if (header.face_id && state.faceShown && header.face_id !== state.faceShown && reloadOnce(header.session_id)) return;
  names = header.blendshapes;
  sampleRate = header.audio.sample_rate;
  queue = [];
  passed = null;
  anchor = null;
  ended = false;
  current = current && current.length === names.length ? current : new Float32Array(names.length);
  state.session = header.session_id;
  state.lastSeq = -1;
}

function onMessage(event) {
  if (typeof event.data !== 'string') {
    const view = new DataView(event.data);
    playChunk(view.getFloat64(0, true), new Float32Array(event.data.slice(8)));
    return;
  }
  const message = JSON.parse(event.data);
  if (message.type === 'header') onHeader(message);
  else if (message.type === 'frame') {
    if (!anchor) anchor = { clock: clockNow() + BUFFER_S + DELAY_S, t: message.t_ms };   // frames before any audio
    queue.push(message);
    state.frames += 1;
    state.lastSeq = message.seq;
  } else if (message.type === 'end') { ended = true; state.streamsEnded += 1; }
  else if (message.type === 'error') fail(message.detail);
}

let socket = null;
let delay = 500;           // reconnect backoff: grows across attempts, back to 500 ms once connected
function connect(faceId) {
  const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
  socket = new WebSocket(`${scheme}://${location.host}/session/${encodeURIComponent(faceId)}/view`);
  socket.binaryType = 'arraybuffer';
  socket.onopen = () => { state.connected = true; delay = 500; report(); announce(); };
  socket.onmessage = onMessage;
  socket.onclose = () => {
    state.connected = false;
    ended = true;
    setTimeout(() => connect(faceId), delay);   // the server restarted, or the network blinked
    delay = Math.min(delay * 2, 8000);
  };
}

// Tell the server what this screen is and whether it can be heard: the editor shows it as
// "1 screen ready". State, not media (models.ViewerStatus).
function report() {
  if (socket && socket.readyState === WebSocket.OPEN) {
    // Only the tiers the server knows (models.ViewerStatus): any other ?display= draws flat, and a status
    // naming it would be dropped whole, leaving this screen uncounted.
    const display = ['flat', 'hld', 'go'].includes(DISPLAY) ? DISPLAY : 'flat';
    socket.send(JSON.stringify({ type: 'status', audio: state.audio === 'running', display, kiosk: KIOSK, sound: state.sound }));
  }
}

// Connected and heard: say so for a moment, then get out of the way.
let readyTimer = null;
function announce() {
  state.screenReady = state.connected && (EXTERNAL || state.audio === 'running');
  if (!state.screenReady) return;
  readyEl.classList.add('show');
  clearTimeout(readyTimer);
  readyTimer = setTimeout(() => readyEl.classList.remove('show'), 3000);
}

// The one gesture a browser needs: sound on, and full screen (a page cannot ask for either unprompted).
// Outside a kiosk, only the gesture that turns sound on goes full screen: a later tap must not pull
// back into full screen someone who left it.
let gestured = false;
function start() {
  const first = EXTERNAL ? !gestured : state.audio !== 'running';
  gestured = true;
  unlockAudio();
  const root = document.documentElement;
  if ((first || KIOSK) && !document.fullscreenElement && root.requestFullscreen) root.requestFullscreen().catch(() => {});
}

// ---- drawing -------------------------------------------------------------------------------

function applyWeights(weights, order) {
  for (let i = 0; i < order.length; i += 1) {
    const slots = targets.get(order[i]);
    if (!slots) continue;   // a shape this face does not have
    for (const slot of slots) slot.influences[slot.index] = weights[i];
  }
  state.weights = weights;
}

const euler = new THREE.Euler();
const turn = new THREE.Quaternion();
function applyHead() {
  const rad = Math.PI / 180;
  euler.set(head.pitch * rad, head.yaw * rad, head.roll * rad, 'YXZ');
  if (headBone) headBone.quaternion.copy(headRest).multiply(turn.setFromEuler(euler));
  else pivot.rotation.copy(euler);
}

let lastDraw = performance.now();
let drawn = 0;
let fps = 0;
let fpsSince = performance.now();
function draw() {
  requestAnimationFrame(draw);
  const wall = performance.now();
  const dt = Math.min(0.1, (wall - lastDraw) / 1000);
  lastDraw = wall;
  const t = timelineNow();
  state.timeline = t;   // the session time (ms) on screen now, or null: for checking sync from outside
  let live = false;
  if (posed) {
    live = true;
    if (!current || current.length !== posed.weights.length) current = new Float32Array(posed.weights.length);
    current.set(posed.weights);
    Object.assign(head, posed.head);
  } else if (current && t !== null && queue.length) {
    while (queue.length > 1 && queue[1].t_ms <= t) passed = queue.shift();   // frames already passed
    const a = queue[0];
    const b = queue[1];
    if (a.t_ms <= t) {   // until the first frame is due, the face stays idle
      live = true;
      if (b) {
        const k = b.t_ms > a.t_ms ? Math.min(1, Math.max(0, (t - a.t_ms) / (b.t_ms - a.t_ms))) : 0;
        const before = passed && passed.seq === a.seq - 1 ? passed : null;
        const after = queue[2] && queue[2].seq === b.seq + 1 ? queue[2] : null;
        interpolate(before, a, b, after, k, current, head);
      } else {
        current.set(a.weights);
        Object.assign(head, a.head);
      }
      if (!b && ended && t > a.t_ms + 100) { queue = []; passed = null; anchor = null; }   // the last frame has been shown
    }
  }
  const order = names.length ? names : ARKIT;   // before any stream, the ARKit order
  if (!current || current.length !== order.length) current = new Float32Array(order.length);
  if (!live) {
    const keep = Math.exp(-RELAX_PER_S * dt);
    for (let i = 0; i < current.length; i += 1) current[i] *= keep;   // settle into the rest set
  }
  // The rest set, idle life when nothing plays, and the lip budget, over the frame's own weights.
  const shown = motion.present(current, order, { live, t: wall / 1000, dt });
  if (!live) {
    const k = 1 - Math.exp(-RELAX_PER_S * dt);   // ease from the stream's last head to the idle's
    for (const axis of AXES) head[axis] += (shown.head[axis] - head[axis]) * k;
  }
  if (splat) {
    state.weights = shown.weights;
    splat.set(shown.weights, order, head);
    splat.step();
  } else {
    applyWeights(shown.weights, order);
    applyHead();
    if (backdrop) {
      if (posed) backdrop.update(posed.t, { exact: true });
      else backdrop.update(wall / 1000);
    }
    look.render();
  }
  if (posedResolve) { const done = posedResolve; posedResolve = null; done(); }

  drawn += 1;
  if (wall - fpsSince >= 1000) { fps = Math.round(drawn * 1000 / (wall - fpsSince)); drawn = 0; fpsSince = wall; }
  if (!hudEl.hidden) {
    hudEl.textContent = `${state.face || '…'}  ${DISPLAY}  ${state.connected ? (live ? 'live' : 'idle') : 'offline'}  ${fps} fps  audio ${state.audio}`
      + (state.underruns ? `  underruns ${state.underruns}` : '');
  }
}

// ---- start ---------------------------------------------------------------------------------

async function main() {
  if (DISPLAY === 'go') { fail('The Looking Glass Go tier is not built yet. Use display=flat or display=hld.'); return; }
  if (!['flat', 'hld'].includes(DISPLAY)) { fail(`Unknown display "${DISPLAY}". Use flat or hld.`); return; }
  let faceId;
  try { faceId = await loadFace(); } catch (error) { fail(error.message || String(error)); return; }
  if (RECORD) { if (!splat) draw(); return; }   // posed from outside; no stream, no sound, no notice
  connect(faceId);
  draw();
  // Browsers only start audio after a click or a key press. A kiosk can waive that
  // (Chromium: --autoplay-policy=no-user-gesture-required), so try first and ask only if refused.
  unlockAudio();
  setTimeout(() => {
    // Silent until asked: say so, plainly, rather than leave a face that will never be heard.
    if (!EXTERNAL && state.audio !== 'running' && state.audio !== 'unavailable') startEl.classList.add('show');
  }, 400);
  window.addEventListener('pointerdown', start);
  window.addEventListener('keydown', start);
}

main();
