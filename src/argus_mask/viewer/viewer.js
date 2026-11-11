// The argus-mask viewer: one rigged head, driven by frames from the server and kept in step
// with the audio by this page's own audio clock.
//
//   ?face=<id>       which face (default: the first the server lists)
//   ?display=flat    flat (default) | hld (1080x1920 portrait on black) | go (not built yet)
//   ?bg=<css colour> background for the flat tier, or "transparent"
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
//
// Socket URLs come from the page's own origin, so the same page works behind a TLS proxy.

import * as THREE from 'three';
import { GLTFLoader } from './vendor/loaders/GLTFLoader.js';
import { Look, lookOptions } from './look.js';

const params = new URLSearchParams(location.search);
const DISPLAY = params.get('display') || 'flat';
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
  streamsEnded: 0, underruns: 0, audioDropped: 0, lastSeq: -1, weights: null, error: null, audio: 'locked',
  look: null,   // the profile and effects in use: see look.js
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
state.pose = (weights, headTurn, blendshapes) => new Promise((resolve) => {
  if (blendshapes) names = blendshapes;   // the stream's order, from its header
  posed = { weights: Float32Array.from(weights), head: headTurn };
  posedResolve = resolve;
});

// ---- scene ---------------------------------------------------------------------------------

const transparent = DISPLAY === 'flat' && params.get('bg') === 'transparent';
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: transparent });
renderer.outputColorSpace = THREE.SRGBColorSpace;
stageEl.appendChild(renderer.domElement);
const scene = new THREE.Scene();
if (DISPLAY === 'hld') {
  scene.background = new THREE.Color(0x000000);   // the panel wants its subject on black
  document.body.style.background = '#000';
} else if (transparent) {
  document.body.classList.add('transparent');
} else {
  const colour = params.get('bg') || '#101216';
  try { scene.background = new THREE.Color(colour); document.body.style.background = colour; } catch { /* keep the default */ }
}
const camera = new THREE.PerspectiveCamera(30, 1, 0.01, 100);
// Lights, environment, tone mapping, materials and post: a recording renders offline, so it gets the most.
const look = new Look(renderer, scene, camera, lookOptions(params, RECORD ? 'high' : 'medium'));
const pivot = new THREE.Group();   // head rotation is applied here unless the model has a head bone
scene.add(pivot);

function resize() {
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
  }
  camera.updateProjectionMatrix();
  state.look = look.summary;
}
window.addEventListener('resize', resize);

// ---- the face ------------------------------------------------------------------------------

let targets = new Map();   // blendshape name -> [{ influences, index }]
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
let ended = true;
let current = null;        // Float32Array of the weights on screen
const head = { yaw: 0, pitch: 0, roll: 0 };

function onHeader(header) {
  // An alias (?face=holly) repointed to another face since this page loaded it: load the new one.
  // The stream goes on meanwhile, and the reloaded page joins it where it has got to.
  if (header.face_id && state.faceShown && header.face_id !== state.faceShown) { location.reload(); return; }
  names = header.blendshapes;
  sampleRate = header.audio.sample_rate;
  queue = [];
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
    socket.send(JSON.stringify({ type: 'status', audio: state.audio === 'running', display: DISPLAY, kiosk: KIOSK, sound: state.sound }));
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

function applyWeights(weights) {
  for (let i = 0; i < names.length; i += 1) {
    const slots = targets.get(names[i]);
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

// With no stream, a little life of the page's own. It is cosmetic and never recorded.
function localIdle(seconds) {
  const phase = seconds % 4.3;
  const blink = phase < 0.08 ? phase / 0.08 : phase < 0.2 ? 1 - (phase - 0.08) / 0.12 : 0;
  for (const name of ['eyeBlinkLeft', 'eyeBlinkRight']) {
    for (const slot of targets.get(name) || []) slot.influences[slot.index] = Math.max(slot.influences[slot.index], blink);
  }
  head.yaw = 1.5 * Math.sin(seconds / 7.3 * 2 * Math.PI);
  head.pitch = 0.9 * Math.sin(seconds / 5.9 * 2 * Math.PI);
  head.roll = 0.5 * Math.sin(seconds / 9.7 * 2 * Math.PI);
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
    while (queue.length > 1 && queue[1].t_ms <= t) queue.shift();   // frames already passed
    const a = queue[0];
    const b = queue[1];
    if (a.t_ms <= t) {   // until the first frame is due, the face stays idle
      live = true;
      const k = b && b.t_ms > a.t_ms ? Math.min(1, Math.max(0, (t - a.t_ms) / (b.t_ms - a.t_ms))) : 0;
      for (let i = 0; i < current.length; i += 1) {
        current[i] = b ? a.weights[i] + (b.weights[i] - a.weights[i]) * k : a.weights[i];
      }
      for (const axis of ['yaw', 'pitch', 'roll']) head[axis] = b ? a.head[axis] + (b.head[axis] - a.head[axis]) * k : a.head[axis];
      if (!b && ended && t > a.t_ms + 100) { queue = []; anchor = null; }   // the last frame has been shown
    }
  }
  if (current) {
    if (!live) {
      const keep = Math.exp(-RELAX_PER_S * dt);
      for (let i = 0; i < current.length; i += 1) current[i] *= keep;
    }
    applyWeights(current);
  }
  if (!live) localIdle(wall / 1000);
  applyHead();
  look.render();
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
  resize();
  let faceId;
  try { faceId = await loadFace(); } catch (error) { fail(error.message || String(error)); return; }
  if (RECORD) { draw(); return; }   // posed from outside; no stream, no sound, no notice
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
