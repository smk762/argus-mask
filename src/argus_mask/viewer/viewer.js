// The argus-mask viewer: one rigged head, driven by frames from the server and kept in step
// with the audio by this page's own audio clock.
//
//   ?face=<id>       which face (default: the first the server lists)
//   ?display=flat    flat (default) | hld (1080x1920 portrait on black) | go (not built yet)
//   ?bg=<css colour> background for the flat tier, or "transparent"
//   ?hud=0           hide the status line
//   ?buffer=250      how far behind arrival the audio plays, in ms (more absorbs jitter, less feels snappier)
//   ?record=1        posed from outside (`argus-mask record`): no socket, no sound, no notice
//
// Socket URLs come from the page's own origin, so the same page works behind a TLS proxy.

import * as THREE from 'three';
import { GLTFLoader } from './vendor/loaders/GLTFLoader.js';

const params = new URLSearchParams(location.search);
const DISPLAY = params.get('display') || 'flat';
const RECORD = params.get('record') === '1';
const BUFFER_MS = params.has('buffer') ? Number(params.get('buffer')) : 250;
const BUFFER_S = Math.min(2, Math.max(0.05, (Number.isFinite(BUFFER_MS) ? BUFFER_MS : 250) / 1000));  // audio plays this far behind arrival; absorbs jitter
const RESYNC_S = 0.05;        // after an underrun, how soon the late audio starts
const RELAX_PER_S = 6;        // how fast the face settles when a stream ends
const HLD = { width: 1080, height: 1920 };

const stageEl = document.getElementById('stage');
const hudEl = document.getElementById('hud');
const noticeEl = document.getElementById('notice');
const noticeText = document.getElementById('notice-text');
const noticeButton = document.getElementById('notice-button');
if (params.get('hud') === '0') hudEl.hidden = true;

// What a test (or a curious operator in the console) can read.
const state = window.__mask = {
  face: null, ready: false, connected: false, session: null, frames: 0, audioChunks: 0,
  streamsEnded: 0, underruns: 0, audioDropped: 0, lastSeq: -1, weights: null, error: null, audio: 'locked',
};

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
scene.add(new THREE.HemisphereLight(0xffffff, 0x3a3f4a, 1.6));
const key = new THREE.DirectionalLight(0xffffff, 2.2);
key.position.set(0.6, 0.8, 1.5);
scene.add(key);
const camera = new THREE.PerspectiveCamera(30, 1, 0.01, 100);
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
  } else {
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(width, height);
    camera.aspect = width / Math.max(1, height);
  }
  camera.updateProjectionMatrix();
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
  state.face = faceId;
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
  try {
    audio = audio || new (window.AudioContext || window.webkitAudioContext)();
  } catch { state.audio = 'unavailable'; return; }
  const settle = () => {
    state.audio = audio.state === 'running' ? 'running' : 'locked';
    if (audio.state === 'running') { notice(''); anchor = null; }   // re-anchor on the audio clock
  };
  audio.resume().then(settle, settle);
  settle();
}

function playChunk(startMs, samples) {
  state.audioChunks += 1;
  const now = clockNow();
  if (!anchor) { anchor = { clock: now + BUFFER_S, t: startMs }; nextPlay = anchor.clock; }
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
    if (!anchor) anchor = { clock: clockNow() + BUFFER_S, t: message.t_ms };   // frames before any audio
    queue.push(message);
    state.frames += 1;
    state.lastSeq = message.seq;
  } else if (message.type === 'end') { ended = true; state.streamsEnded += 1; }
  else if (message.type === 'error') fail(message.detail);
}

function connect(faceId) {
  const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
  const socket = new WebSocket(`${scheme}://${location.host}/session/${encodeURIComponent(faceId)}/view`);
  socket.binaryType = 'arraybuffer';
  let delay = 500;
  socket.onopen = () => { state.connected = true; delay = 500; };
  socket.onmessage = onMessage;
  socket.onclose = () => {
    state.connected = false;
    ended = true;
    setTimeout(() => connect(faceId), delay);   // the server restarted, or the network blinked
    delay = Math.min(delay * 2, 8000);
  };
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
  renderer.render(scene, camera);
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
    if (state.audio !== 'running' && state.audio !== 'unavailable') notice('The face is ready. Sound needs one click.', 'Turn sound on', unlockAudio);
  }, 400);
  window.addEventListener('pointerdown', unlockAudio);
  window.addEventListener('keydown', unlockAudio);
}

main();
