// The face editor: a form that writes a preset, POST /build, and a preview of the face it made.
// Everything the server sends is put on the page as text, never as HTML.

import * as THREE from 'three';
import { GLTFLoader } from './vendor/loaders/GLTFLoader.js';

const $ = (id) => document.getElementById(id);

// stock-02's recipe: a sensible starting point, and the mouth settings both stock heads use.
const DEFAULT = {
  id: 'my-face',
  name: 'My face',
  phenotype: {
    gender: 0.0, age: 0.5, muscle: 0.5, weight: 0.5, proportions: 0.5, height: 0.5, cupsize: 0.5, firmness: 0.5,
    race: { asian: 0.2, caucasian: 0.6, african: 0.2 },
  },
  assets: {
    skin: 'skins/young_caucasian_female/young_caucasian_female.mhmat',
    eyes: 'eyes/high-poly/high-poly.mhclo',
    eyebrows: 'eyebrows/eyebrow002/eyebrow002.mhclo',
    eyelashes: 'eyelashes/eyelashes02/eyelashes02.mhclo',
    teeth: 'teeth/teeth_base/teeth_base.mhclo',
    tongue: 'tongue/tongue01/tongue01.mhclo',
    hair: 'hair/short02/short02.mhclo',
    clothes: ['clothes/female_casualsuit01/female_casualsuit01.mhclo'],
  },
  bust: { height_m: 0.4, half_width_m: 0.23 },
  mouth: { tongue_flatten: 0.6, tongue_drop_mm: 0.0, upper_teeth_drop_mm: 2.5, cavity_shade: 0.02 },
  shape_scale: { mouthPucker: 1.5 },
  texture_gain: { teeth: 1.3 },
  blend_parts: ['eyebrows', 'eyelashes'],
  max_texture_px: 2048,
  max_alpha_texture_px: 1024,
  max_triangles: 30000,
};

const PARTS = [
  ['skin', 'Skin'], ['hair', 'Hair'], ['eyebrows', 'Eyebrows'], ['eyelashes', 'Eyelashes'],
  ['eyes', 'Eyes'], ['teeth', 'Teeth'], ['tongue', 'Tongue'],
];
const OPTIONAL = new Set(['hair']);

// [path in the preset, label, min, max, step, how to show the value]
const years = (a) => `${Math.round(25 + 130 * (a - 0.5))} y`;
const SLIDERS = [
  ['phenotype.gender', 'Female – male', 0, 1, 0.01],
  ['phenotype.age', 'Age', 0.5, 1, 0.005, years],
  ['phenotype.muscle', 'Muscle', 0, 1, 0.01],
  ['phenotype.weight', 'Weight', 0, 1, 0.01],
  ['phenotype.proportions', 'Proportions', 0, 1, 0.01],
  ['phenotype.race.african', 'African', 0, 1, 0.01],
  ['phenotype.race.asian', 'Asian', 0, 1, 0.01],
  ['phenotype.race.caucasian', 'Caucasian', 0, 1, 0.01],
];
const ADVANCED = [
  ['mouth.tongue_flatten', 'Tongue height', 0.2, 1, 0.05],
  ['mouth.upper_teeth_drop_mm', 'Upper teeth ↓ mm', 0, 6, 0.1],
  ['mouth.cavity_shade', 'Mouth shade', 0, 1, 0.01],
  ['shape_scale.mouthPucker', 'Pucker ×', 0, 3, 0.05],
  ['texture_gain.teeth', 'Teeth brightness', 0.2, 3, 0.05],
  ['bust.height_m', 'Bust height m', 0.2, 0.6, 0.01],
  ['max_triangles', 'Triangle budget', 1000, 30000, 500, (v) => String(v)],
];

let preset = structuredClone(DEFAULT);
const get = (obj, path) => path.split('.').reduce((o, k) => (o == null ? o : o[k]), obj);
function set(obj, path, value) {
  const keys = path.split('.');
  let o = obj;
  for (const k of keys.slice(0, -1)) o = o[k] ??= {};
  o[keys.at(-1)] = value;
}

function status(text, bad = false) {
  $('status').textContent = text;
  $('status').classList.toggle('bad', bad);
}

async function json(url, options) {
  const reply = await fetch(url, options);
  const body = await reply.json().catch(() => ({}));
  if (!reply.ok) {
    let detail = body.detail ?? reply.statusText;
    if (Array.isArray(detail)) detail = detail.map((d) => `${(d.loc || []).slice(1).join('.')}: ${d.msg}`).join('; ');
    throw new Error(String(detail));
  }
  return body;
}

// ---- the form ------------------------------------------------------------------------------

function slider(container, [path, label, min, max, step, show]) {
  const row = document.createElement('label');
  row.className = 'row';
  const name = document.createElement('span');
  name.textContent = label;
  const input = Object.assign(document.createElement('input'), { type: 'range', min, max, step });
  const out = document.createElement('output');
  const fmt = show || ((v) => Number(v).toFixed(2));
  input.addEventListener('input', () => {
    const v = step >= 1 ? parseInt(input.value, 10) : parseFloat(input.value);
    set(preset, path, v);
    out.textContent = fmt(v);
  });
  input.sync = () => {
    const v = get(preset, path) ?? min;
    input.value = v;
    out.textContent = fmt(v);
  };
  row.append(name, input, out);
  container.append(row);
  return input;
}

let controls = [];
const tiles = new Map(); // group key -> { group, list: [{ ref, el }] }

// A row of tiles. A body part is one field of preset.assets; "outfit" and "hat" share preset.assets.clothes,
// one item each.
function groups(library) {
  const body = PARTS.map(([part, label]) => ({
    key: part, label, part, optional: OPTIONAL.has(part), choices: library[part] || [],
    get: () => preset.assets[part] ?? null,
    set: (ref) => { preset.assets[part] = ref; },
  }));
  const clothes = library.clothes || [];
  const slotOf = new Map(clothes.map((a) => [a.ref, a.slot]));
  const worn = (slot) => ({
    key: slot, label: slot === 'hat' ? 'Hat' : 'Clothes', part: 'clothes', optional: true,
    choices: clothes.filter((a) => a.slot === slot),
    get: () => (preset.assets.clothes || []).find((r) => slotOf.get(r) === slot) ?? null,
    set: (ref) => {
      const rest = (preset.assets.clothes || []).filter((r) => slotOf.get(r) !== slot);
      preset.assets.clothes = ref ? [...rest, ref] : rest;
    },
  });
  // Clothes after skin and hair: they are the next thing a viewer sees.
  return [body[0], body[1], worn('outfit'), worn('hat'), ...body.slice(2)];
}

function buildParts(library) {
  const parts = $('parts');
  parts.replaceChildren();
  tiles.clear();
  for (const group of groups(library)) {
    const h = document.createElement('h2');
    h.textContent = group.label;
    const grid = document.createElement('div');
    grid.className = 'grid';
    const list = [];
    const choices = [...(group.optional ? [null] : []), ...group.choices];
    for (const asset of choices) {
      const el = document.createElement('div');
      el.className = 'tile';
      el.tabIndex = 0;
      if (asset) {
        const img = Object.assign(document.createElement('img'), { loading: 'lazy', alt: '' });
        img.src = `/library/${encodeURIComponent(group.part)}/${encodeURIComponent(asset.name)}/thumb`;
        img.onerror = () => img.replaceWith(Object.assign(document.createElement('div'), { className: 'none', textContent: '?' }));
        el.title = `${asset.name}\n${asset.license}${asset.author ? `, ${asset.author}` : ''}`;
        const name = document.createElement('span');
        name.textContent = asset.name.replaceAll('_', ' ');
        el.append(img, name);
      } else {
        el.append(Object.assign(document.createElement('div'), { className: 'none', textContent: 'none' }));
        el.title = `no ${group.label.toLowerCase()}`;
      }
      const ref = asset ? asset.ref : null;
      const choose = () => { group.set(ref); syncTiles(); };
      el.addEventListener('click', choose);
      el.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); choose(); } });
      grid.append(el);
      list.push({ ref, el });
    }
    if (!group.choices.length) {
      const none = document.createElement('div');
      none.className = 'tile';
      none.textContent = 'nothing installed';
      grid.append(none);
    }
    tiles.set(group.key, { group, list });
    parts.append(h, grid);
  }
}

function syncTiles() {
  for (const { group, list } of tiles.values()) {
    const chosen = group.get();
    for (const { ref, el } of list) el.classList.toggle('on', chosen === ref);
  }
}

function syncForm() {
  $('id').value = preset.id;
  $('name').value = preset.name;
  for (const c of controls) c.sync();
  syncTiles();
}

async function loadStarts() {
  const select = $('start');
  const presets = await json('/presets').catch(() => []);
  select.replaceChildren(new Option('New face (defaults)', ''));
  for (const p of presets) select.append(new Option(`${p.name} (${p.id})`, p.id));
  select.onchange = () => {
    const chosen = presets.find((p) => p.id === select.value);
    preset = structuredClone(chosen ? { ...DEFAULT, ...chosen } : DEFAULT);
    syncForm();
    if (chosen) showFace(chosen.id).catch(() => {});
  };
  return presets;
}

// ---- the preview -----------------------------------------------------------------------------

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.outputColorSpace = THREE.SRGBColorSpace;
$('preview').prepend(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color('#101216');
scene.add(new THREE.HemisphereLight(0xffffff, 0x3a3f4a, 1.6));
const key = new THREE.DirectionalLight(0xffffff, 2.2);
key.position.set(0.6, 0.8, 1.5);
scene.add(key);
const camera = new THREE.PerspectiveCamera(30, 1, 0.01, 100);
camera.position.set(0, -0.08, 0.9);
const pivot = new THREE.Group();
scene.add(pivot);
let targets = new Map();
let pose = 'rest';
let turn = 0;

function resize() {
  const box = $('preview').getBoundingClientRect();
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(box.width, box.height, false);
  camera.aspect = box.width / Math.max(1, box.height);
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);
resize();

function weigh(name, value) {
  for (const { influences, index } of targets.get(name) || []) influences[index] = value;
}

const POSES = {
  rest: () => ({}),
  open: () => ({ jawOpen: 0.6, mouthFunnel: 0.1 }),
  smile: () => ({ mouthSmileLeft: 0.7, mouthSmileRight: 0.7, jawOpen: 0.12 }),
  // A rough syllable rhythm: enough to judge the mouth in motion. The real motion comes from Audio2Face.
  talk: (t) => {
    const s = Math.max(0, Math.sin(t * 9.0)) * (0.55 + 0.45 * Math.sin(t * 2.3));
    return { jawOpen: 0.45 * s, mouthFunnel: 0.25 * Math.max(0, Math.sin(t * 4.1)), mouthClose: 0.1 * (1 - s),
             mouthPucker: 0.3 * Math.max(0, Math.sin(t * 3.3 + 1)), mouthStretchLeft: 0.2 * s, mouthStretchRight: 0.2 * s };
  },
};

function animate(now) {
  const t = now / 1000;
  const weights = POSES[pose](t);
  const blink = (t % 4.2) < 0.12 ? 1 : 0;
  for (const name of targets.keys()) weigh(name, weights[name] ?? 0);
  weigh('eyeBlinkLeft', blink);
  weigh('eyeBlinkRight', blink);
  pivot.rotation.y = turn;
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}
requestAnimationFrame(animate);

function poseButtons() {
  const bar = $('poses');
  bar.replaceChildren();
  for (const name of Object.keys(POSES)) {
    const b = Object.assign(document.createElement('button'), { type: 'button', textContent: name });
    b.onclick = () => { pose = name; for (const x of bar.children) x.classList.toggle('on', x === b); };
    b.classList.toggle('on', name === pose);
    bar.append(b);
  }
  const spin = Object.assign(document.createElement('button'), { type: 'button', textContent: 'turn' });
  spin.onclick = () => { turn = turn ? 0 : 0.6; };
  bar.append(spin);
  bar.hidden = false;
}

async function showFace(faceId) {
  const manifest = await json(`/face/${encodeURIComponent(faceId)}`);
  const gltf = await new GLTFLoader().loadAsync(`/face/${encodeURIComponent(faceId)}/model?v=${manifest.model.sha256}`);
  pivot.clear();
  targets = new Map();
  pivot.add(gltf.scene);
  gltf.scene.traverse((node) => {
    if (node.isMesh && node.morphTargetDictionary) {
      for (const [name, index] of Object.entries(node.morphTargetDictionary)) {
        if (!targets.has(name)) targets.set(name, []);
        targets.get(name).push({ influences: node.morphTargetInfluences, index });
      }
    }
  });
  const f = manifest.framing || {};
  camera.fov = f.fov || 30;
  camera.position.set(0, f.height || 0, f.distance || 0.6);
  camera.lookAt(0, f.height || 0, 0);
  resize();
  $('empty').hidden = true;
  $('credits').textContent = `${manifest.name} (${manifest.id}) · ${manifest.provenance.license} · hover for credits`;
  $('credits').title = manifest.provenance.notes;
  poseButtons();
  window.__editor = { face: faceId, shapes: targets.size };
}

// ---- build -----------------------------------------------------------------------------------

$('form').addEventListener('submit', async (event) => {
  event.preventDefault();
  preset.id = $('id').value.trim();
  preset.name = $('name').value.trim();
  const body = structuredClone(preset);
  if (!body.assets.hair) delete body.assets.hair;
  $('build').disabled = true;
  status('Building…');
  const started = performance.now();
  try {
    const done = await json('/build', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    status('Built. Loading the preview…');
    await showFace(done.manifest.id);
    status(`Built in ${done.seconds} s: ${done.triangles?.toLocaleString() ?? '?'} triangles, about ${done.age_years}. It is on the server's /faces now.`);
    const presets = await loadStarts();
    if (presets.some((p) => p.id === done.manifest.id)) $('start').value = done.manifest.id;
  } catch (err) {
    status(`${err.message} (${((performance.now() - started) / 1000).toFixed(1)} s)`, true);
  } finally {
    $('build').disabled = false;
  }
});

async function main() {
  const health = await json('/health');
  if (!health.ready || health.read_only) {
    $('banner').textContent = health.read_only ? 'This editor is read-only: it cannot build.' : `Cannot build: ${health.reason}`;
    $('banner').hidden = false;
    $('build').disabled = true;
  }
  controls = SLIDERS.map((s) => slider($('sliders'), s));
  controls.push(...ADVANCED.map((s) => slider($('advanced'), s)));
  buildParts(await json('/library'));
  await loadStarts();
  syncForm();
}

main().catch((err) => status(err.message, true));
