// The face editor: panels that write a preset, POST /build, and a preview of the face it made.
// Everything the server sends is put on the page as text, never as HTML.

import { Stage } from './stage.js';

const $ = (id) => document.getElementById(id);
const el = (tag, props = {}, ...children) => {
  const node = Object.assign(document.createElement(tag), props);
  node.append(...children);
  return node;
};
const store = {  // per-browser conveniences only; the page works without them
  get(key, fallback) { try { return JSON.parse(localStorage.getItem(`mask-editor.${key}`)) ?? fallback; } catch { return fallback; } },
  set(key, value) { try { localStorage.setItem(`mask-editor.${key}`, JSON.stringify(value)); } catch { /* private window */ } },
};

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
    clothes: ['clothes/thegreatengineer_galactic_warrior_uniform/thegreatengineer_galactic_warrior_uniform.mhclo'],
  },
  targets: {},
  bust: { height_m: 0.4, half_width_m: 0.23 },
  mouth: { tongue_flatten: 0.6, tongue_drop_mm: 0.0, upper_teeth_drop_mm: 2.5, cavity_shade: 0.02 },
  shape_scale: { mouthPucker: 1.5 },
  texture_gain: { teeth: 1.3 },
  blend_parts: ['eyebrows', 'eyelashes'],
  max_texture_px: 2048,
  max_alpha_texture_px: 1024,
  max_triangles: 30000,
};

// How each value reads: in the units it means, not MakeHuman's 0..1 or -1..+1.
const fixed = (v) => Number(v).toFixed(2);
const pct = (v) => `${Math.round(v * 100)}%`;
const years = (a) => `${Math.round(a < 0.5 ? 1 + 48 * a : 25 + 130 * (a - 0.5))} years`;  // MakeHuman's scale
const sex = (v) => (v <= 0.005 ? 'female' : v >= 0.995 ? 'male' : Math.abs(v - 0.5) < 0.005 ? 'even'
  : v < 0.5 ? `${Math.round((1 - v) * 100)}% female` : `${Math.round(v * 100)}% male`);
// MakeHuman normalises the three ancestry values to a mix: show each one's share of it.
const share = (key) => () => {
  const race = preset.phenotype.race || {};
  const total = (race.african || 0) + (race.asian || 0) + (race.caucasian || 0);
  return total > 0 ? `${Math.round(((race[key] || 0) / total) * 100)}% of mix` : 'even mix';
};
// A two-way shape: "45% wider", "20% down"; 0 is the phenotype's own shape.
const toward = ([low, high]) => (v) => (Math.abs(v) < 0.005 ? '—' : `${Math.round(Math.abs(v) * 100)}% ${v > 0 ? high : low}`);
const ADULT_AGE = 0.3542;  // 18 years; preset.py and build_head.py hold the same
// [path in the preset, label, min, max, step, how to show the value, the value it rests at]
const BODY = [
  ['phenotype.gender', 'Female – male', 0, 1, 0.01, sex, 0],
  ['phenotype.age', 'Age', ADULT_AGE, 1, 0.005, years, 0.5],
  ['phenotype.muscle', 'Muscle', 0, 1, 0.01, pct, 0.5],
  ['phenotype.weight', 'Weight', 0, 1, 0.01, pct, 0.5],
  ['phenotype.proportions', 'Proportions', 0, 1, 0.01, pct, 0.5],
  ['phenotype.race.african', 'African', 0, 1, 0.01, share('african'), 0.2],
  ['phenotype.race.asian', 'Asian', 0, 1, 0.01, share('asian'), 0.2],
  ['phenotype.race.caucasian', 'Caucasian', 0, 1, 0.01, share('caucasian'), 0.6],
];
const ADVANCED = [
  ['mouth.tongue_flatten', 'Tongue height', 0.2, 1, 0.05, (v) => `${pct(v)} of full`, 0.6],
  ['mouth.upper_teeth_drop_mm', 'Upper teeth lowered', 0, 6, 0.1, (v) => `${Number(v).toFixed(1)} mm`, 2.5],
  ['mouth.cavity_shade', 'Mouth darkness', 0, 1, 0.01, (v) => `${Math.round((1 - v) * 100)}% darker`, 0.02],
  ['shape_scale.mouthPucker', 'Pucker strength', 0, 3, 0.05, (v) => `× ${Number(v).toFixed(2)}`, 1.5],
  ['texture_gain.teeth', 'Teeth brightness', 0.2, 3, 0.05, (v) => `× ${Number(v).toFixed(2)}`, 1.3],
  ['bust.height_m', 'Bust height', 0.2, 0.6, 0.01, (v) => `${Math.round(v * 100)} cm`, 0.4],
  ['max_triangles', 'Triangle budget', 1000, 30000, 500, (v) => Number(v).toLocaleString(), 30000],
];
// Asset pickers: [key, title, library part, optional, clothes slot, enlarge on hover]
const PICKERS = [
  ['skin', 'Skin', 'skin', false, null, true], ['hair', 'Hair', 'hair', true, null, true],
  ['outfit', 'Clothes', 'clothes', false, 'outfit', true], ['hat', 'Hat', 'clothes', true, 'hat', true],
  ['eyebrows', 'Eyebrows', 'eyebrows', false], ['eyelashes', 'Eyelashes', 'eyelashes', false],
  ['eyes', 'Eyeballs', 'eyes', false], ['teeth', 'Teeth', 'teeth', false], ['tongue', 'Tongue', 'tongue', false],
];
const OPEN_BY_DEFAULT = new Set(['body', 'skin', 'hair', 'outfit', 'm:head']);

let preset = structuredClone(DEFAULT);
let built = null;   // the preset of the face on show, as JSON, to say when the form has moved on
let made = new Map(); // face id -> { exported, previewed }, from /presets
let busy = false;
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

// ---- panels ----------------------------------------------------------------------------------

const panels = [];   // { key, details, rows: [{ el, text }], changed() }
const controls = []; // everything with a sync()
let filterText = '';

function panel(key, title, { reset } = {}) {
  const open = store.get('open', {})[key] ?? OPEN_BY_DEFAULT.has(key);
  const count = el('span', { className: 'count' });
  const summary = el('summary', {}, el('span', { className: 'title', textContent: title }), count);
  if (reset) {
    const b = el('button', { type: 'button', className: 'small', textContent: 'reset', title: `Reset ${title.toLowerCase()}` });
    b.addEventListener('click', (e) => { e.preventDefault(); reset(); syncForm(); });
    summary.append(b);
  }
  const body = el('div', { className: 'body' });
  const details = el('details', { className: 'panel', open }, summary, body);
  details.addEventListener('toggle', () => {
    if (filterText) return;  // the filter opens panels; that is not the user's choice to remember
    const state = store.get('open', {});
    state[key] = details.open;
    store.set('open', state);
  });
  const entry = { key, title, details, body, countEl: count, rows: [], changed: () => 0 };
  panels.push(entry);
  $('panels').append(details);
  return entry;
}

function slider(entry, [path, label, min, max, step, show = fixed, rest = min < 0 ? 0 : min]) {
  const input = el('input', { type: 'range', min, max, step });
  const out = el('output');
  const row = el('label', { className: 'row', title: `${label} (double-click to reset)` }, el('span', { textContent: label }), input, out);
  const value = () => get(preset, path) ?? rest;
  const paint = () => {
    const v = value();
    out.textContent = show(v);
    row.classList.toggle('changed', Math.abs(v - rest) > 1e-9);
  };
  const sync = () => { input.value = value(); paint(); };
  input.addEventListener('input', () => {
    const v = step >= 1 ? parseInt(input.value, 10) : parseFloat(input.value);
    if (path.startsWith('targets.') && v === 0) delete preset.targets[path.slice(8)];
    else set(preset, path, v);
    paint();
    changed();
  });
  row.addEventListener('dblclick', () => {
    if (path.startsWith('targets.')) delete preset.targets[path.slice(8)];
    else set(preset, path, rest);
    sync();
    changed();
  });
  entry.body.append(row);
  entry.rows.push({ el: row, text: `${label} ${entry.title}`.toLowerCase() });
  controls.push({ sync, paint });  // paint again on any change: the ancestry shares depend on each other
  return { isChanged: () => Math.abs(value() - rest) > 1e-9 };
}

function sliderPanel(key, title, specs, { resetTo }) {
  const entry = panel(key, title, { reset: resetTo });
  const made = specs.map((s) => slider(entry, s));
  entry.changed = () => made.filter((m) => m.isChanged()).length;
  return entry;
}

// ---- the enlarged thumbnail on hover --------------------------------------------------------

const peek = $('peek');
function showPeek(tile, src, name, credit) {
  peek.querySelector('img').src = src;
  peek.querySelector('.cap').replaceChildren(el('b', { textContent: name }), credit);
  peek.hidden = false;
  // Beside the tile, on whichever side has room; never off the screen.
  const r = tile.getBoundingClientRect();
  const w = peek.offsetWidth, h = peek.offsetHeight, gap = 10;
  let x = r.right + gap;
  if (x + w > window.innerWidth - 8) x = r.left - gap - w;
  if (x < 8) x = Math.min(window.innerWidth - w - 8, Math.max(8, r.left));
  let y = r.top + r.height / 2 - h / 2;
  y = Math.max(8, Math.min(window.innerHeight - h - 8, y));
  peek.style.left = `${x}px`;
  peek.style.top = `${y}px`;
}
function hidePeek() { peek.hidden = true; }
window.addEventListener('scroll', hidePeek, { passive: true });

function picker(library, [key, title, part, optional, slot, enlarge]) {
  const all = library[part] || [];
  const choices = slot ? all.filter((a) => a.slot === slot) : all;
  const slotOf = new Map((library.clothes || []).map((a) => [a.ref, a.slot]));
  const current = slot
    ? () => (preset.assets.clothes || []).find((r) => slotOf.get(r) === slot) ?? null
    : () => preset.assets[part] ?? null;
  const choose = slot
    ? (ref) => {
      const rest = (preset.assets.clothes || []).filter((r) => slotOf.get(r) !== slot);
      preset.assets.clothes = ref ? [...rest, ref] : rest;
    }
    : (ref) => { preset.assets[part] = ref; };
  const entry = panel(key, title);
  const grid = el('div', { className: 'grid' });
  const tiles = [];
  for (const asset of [...(optional ? [null] : []), ...choices]) {
    const tile = el('div', { className: 'tile', tabIndex: 0 });
    if (asset) {
      const src = `/library/${encodeURIComponent(part)}/${encodeURIComponent(asset.name)}/thumb`;
      const name = asset.name.replaceAll('_', ' ');
      const credit = `${asset.license}${asset.author ? ` · ${asset.author}` : ''}`;
      const img = el('img', { loading: 'lazy', alt: '', src });
      img.onerror = () => img.replaceWith(el('div', { className: 'none', textContent: '?' }));
      tile.append(img, el('span', { textContent: name }));
      if (enlarge) {
        const show = () => showPeek(tile, src, name, credit);
        tile.addEventListener('mouseenter', show);
        tile.addEventListener('focus', show);
        tile.addEventListener('mouseleave', hidePeek);
        tile.addEventListener('blur', hidePeek);
      } else {
        tile.title = `${name}\n${credit}`;
      }
    } else {
      tile.append(el('div', { className: 'none', textContent: 'none' }));
      tile.title = `no ${title.toLowerCase()}`;
    }
    const ref = asset ? asset.ref : null;
    const pick = () => { choose(ref); syncTiles(); changed(); };
    tile.addEventListener('click', pick);
    tile.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pick(); } });
    grid.append(tile);
    tiles.push({ ref, tile });
    entry.rows.push({ el: tile, text: `${asset ? asset.name.replaceAll('_', ' ') : 'none'} ${title}`.toLowerCase() });
  }
  if (!choices.length) entry.body.append(el('div', { className: 'hint', textContent: 'Nothing installed. See scripts/README.md for the asset packs.' }));
  entry.body.append(grid);
  const syncTiles = () => { const c = current(); for (const t of tiles) t.tile.classList.toggle('on', c === t.ref); };
  controls.push({ sync: syncTiles });
}

// Within a group: the one-way shapes first, then by label, with moves and angles last.
function order(a, b) {
  const rank = (m) => (m.min === 0 ? 0 : /^(Move|Angle|Rotate)/.test(m.label) ? 2 : 1);
  return rank(a) - rank(b) || a.label.localeCompare(b.label, undefined, { numeric: true });
}

function buildPanels(library, catalogue) {
  sliderPanel('body', 'Body', BODY, { resetTo: () => { preset.phenotype = structuredClone(DEFAULT.phenotype); } });
  for (const spec of PICKERS) picker(library, spec);
  for (const [group, title] of Object.entries(catalogue.groups)) {
    const mods = catalogue.modifiers.filter((m) => m.group === group).sort(order);
    if (!mods.length) continue;
    const specs = mods.map((m) => [`targets.${m.name}`, m.label, m.min, 1, 0.05, m.min < 0 ? toward(m.ends) : pct, 0]);
    sliderPanel(`m:${group}`, title, specs, { resetTo: () => { for (const m of mods) delete preset.targets[m.name]; } });
  }
  const adv = sliderPanel('advanced', 'Advanced: mouth fixes and budgets', ADVANCED, {
    resetTo: () => { for (const k of ['mouth', 'shape_scale', 'texture_gain', 'bust', 'max_triangles']) preset[k] = structuredClone(DEFAULT[k]); },
  });
  adv.body.prepend(el('div', { className: 'hint', textContent: 'The recipe’s fixes for speech. The defaults suit every head so far.' }));
}

function syncForm() {
  $('id').value = preset.id;
  $('name').value = preset.name;
  for (const c of controls) c.sync();
  changed();
}

function applyFilter() {
  filterText = $('filter').value.trim().toLowerCase();
  const open = store.get('open', {});
  for (const p of panels) {
    let any = false;
    for (const r of p.rows) {
      const hit = !filterText || r.text.includes(filterText);
      r.el.hidden = !hit;
      any ||= hit;
    }
    p.details.hidden = !!filterText && !any;
    p.details.open = filterText ? any : (open[p.key] ?? OPEN_BY_DEFAULT.has(p.key));
  }
}

// ---- preview: one stage, shown in the card or in the large view -----------------------------

const stage = new Stage();
let shown = null;   // the manifest on show

function openModal() {
  if (!shown) return;
  hidePeek();
  $('modal').hidden = false;
  stage.attach($('stage'));
  $('close').focus();
}
function closeModal() {
  $('modal').hidden = true;
  stage.attach($('thumb'));
}
$('thumb').addEventListener('click', openModal);
$('close').addEventListener('click', closeModal);
$('modal').addEventListener('click', (e) => { if (e.target === $('modal')) closeModal(); });
window.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && !$('modal').hidden) closeModal();
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey) && !$('build').disabled) $('form').requestSubmit();
});

function meta() {
  const box = $('meta');
  const stale = !!shown && built !== null && JSON.stringify(current()) !== built;
  const state = shown ? made.get(shown.id) || {} : {};
  // Export the face on show, and only while it is what the form describes.
  $('export').disabled = busy || !shown || !state.previewed || stale;
  $('export').textContent = state.exported ? 'Export again' : 'Export';
  if (!shown) { box.hidden = true; return; }
  box.replaceChildren(
    el('span', { className: 'name', textContent: `${shown.name} (${shown.id})` }),
    el('span', { textContent: `${shown.provenance.license} · hover for credits`, title: shown.provenance.notes }),
    el('span', {
      className: stale ? 'stale' : '',
      // Live update does not build for a new name or id (that would make a preview per keystroke): say so.
      textContent: !stale ? 'Up to date with the form.'
        : contentOf(current()) === shownContent ? 'Name or id changed: Build to apply it.' : 'The form has changed since this build.',
    }),
    state.exported
      ? el('span', { className: 'live', textContent: `Exported: assets/${shown.id}, on the server's /faces.` })
      : el('span', { textContent: 'Preview only. Export puts it in assets/ and on /faces.' }),
  );
  box.hidden = false;
}

async function showFace(faceId) {
  const from = made.get(faceId)?.previewed === false ? 'export' : 'preview';
  const manifest = await json(`/face/${encodeURIComponent(faceId)}?from=${from}`);
  await stage.load(`/face/${encodeURIComponent(faceId)}/model?from=${from}&v=${manifest.model.sha256}`, manifest.framing || {});
  shown = manifest;
  if (!stage.host) stage.attach($('thumb'));
  $('empty').hidden = true;
  document.querySelector('#thumb .zoom').hidden = false;
  $('modal-name').textContent = `${manifest.name} (${manifest.id})`;
  $('modal-credits').textContent = manifest.provenance.license;
  $('modal-credits').title = manifest.provenance.notes;
  stage.poseButtons($('poses'));
  meta();
  window.__editor = { face: faceId, shapes: stage.targets.size };
}

// ---- the preset in and out -----------------------------------------------------------------

function current() {
  const body = structuredClone(preset);
  body.id = $('id').value.trim();
  body.name = $('name').value.trim();
  if (!body.assets.hair) delete body.assets.hair;
  return body;
}

function changed() {
  for (const c of controls) c.paint?.();
  for (const p of panels) {
    const n = p.changed();
    p.countEl.textContent = n ? `${n} changed` : '';
  }
  meta();
  store.set('draft', current());
  if (ready && live()) schedule();
}

// ---- live update: build by itself, a moment after the last change ---------------------------

const LIVE_DELAY_MS = 1000;
// What a build depends on: the preset without its id and name, so typing those never builds.
const contentOf = (body) => JSON.stringify({ ...body, id: undefined, name: undefined });
let shownContent = null;   // the content of the face on show (built here, or loaded from a saved one)
let failedContent = null;  // the content of the last build that failed: not retried until something changes
let timer = null;
let again = false;         // the form changed during a build: build once more when it ends
let ready = false;         // set once the page has loaded: restoring a draft is not a change to build

const live = () => $('live').checked;
function schedule(delay = LIVE_DELAY_MS) {
  clearTimeout(timer);
  timer = setTimeout(() => {
    const content = contentOf(current());
    if (content === shownContent || content === failedContent || !$('form').checkValidity()) return;
    if (busy) { again = true; return; }
    runBuild();
  }, delay);
}
$('live').checked = store.get('live', false);
$('live').addEventListener('change', () => {
  store.set('live', live());
  if (live()) schedule(0);
});

// ---- building: the preview fades and a spinner counts while Blender works ------------------

let ticker = null;
function building(on) {
  for (const host of [$('thumb'), $('stage')]) host.classList.toggle('building', on);
  for (const box of document.querySelectorAll('.busy')) box.hidden = !on;
  $('empty').hidden = on || !!shown;
  $('build').textContent = on ? 'Building…' : 'Build';
  clearInterval(ticker);
  if (!on) return;
  const started = performance.now();
  const expected = store.get('seconds', 9);  // the last build's time: a good guess at this one's
  const tick = () => {
    const t = (performance.now() - started) / 1000;
    for (const n of document.querySelectorAll('.busy .t')) n.textContent = `${t.toFixed(1)} s`;
    for (const n of document.querySelectorAll('.busy .bar i')) n.style.width = `${Math.min(95, (t / expected) * 100)}%`;
  };
  tick();
  ticker = setInterval(tick, 100);
}

async function loadStarts() {
  const select = $('start');
  const entries = await json('/presets').catch(() => []);
  made = new Map(entries.map((e) => [e.preset.id, { exported: e.exported, previewed: e.previewed }]));
  const presets = entries.map((e) => e.preset);
  const value = select.value;
  select.replaceChildren(el('option', { value: '', textContent: 'New face (defaults)' }));
  for (const p of presets) {
    const s = made.get(p.id);
    select.append(el('option', { value: p.id, textContent: `${p.name} (${p.id})${s.exported ? ' · exported' : ' · preview'}` }));
  }
  if (presets.some((p) => p.id === value)) select.value = value;
  meta();
  select.onchange = async () => {
    const chosen = presets.find((p) => p.id === select.value);
    if (chosen) adopt(chosen); else preset = structuredClone(DEFAULT);
    // A saved face is already built: live update has nothing to do until the form moves on from it.
    if (chosen) shownContent = contentOf(current());
    syncForm();
    if (chosen) {
      await showFace(chosen.id).catch(() => {});
      built = JSON.stringify(current());
      shownContent = contentOf(current());
      meta();
    }
  };
  return presets;
}

$('id').addEventListener('input', changed);
$('name').addEventListener('input', changed);
$('filter').addEventListener('input', applyFilter);

$('form').addEventListener('submit', (event) => {
  event.preventDefault();
  if (busy) { again = true; status('A build is running; this one follows it.'); return; }
  runBuild();
});

async function runBuild() {
  clearTimeout(timer);
  const body = current();
  const content = contentOf(body);
  again = false;
  busy = true;
  $('build').disabled = true;
  building(true);
  meta();
  status(live() ? 'Live update: building…' : 'Building…');
  const started = performance.now();
  try {
    const done = await json('/build', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    store.set('seconds', done.seconds);
    status('Built. Loading the preview…');
    built = JSON.stringify(body);
    shownContent = content;
    failedContent = null;
    await showFace(done.manifest.id);
    await loadStarts();
    $('start').value = done.manifest.id;
    status(`Built in ${done.seconds} s: ${done.triangles?.toLocaleString() ?? '?'} triangles, about ${done.age_years}. A preview: Export to put it on /faces.`);
  } catch (err) {
    failedContent = content;
    status(`${err.message} (${((performance.now() - started) / 1000).toFixed(1)} s)`, true);
  } finally {
    busy = false;
    $('build').disabled = false;
    building(false);
    meta();
  }
  // Changes made while Blender worked: one more build for the latest form, not one per change.
  const moved = contentOf(current()) !== content;
  if (again || (live() && moved)) {
    again = false;
    if (live() || moved) schedule(0);
  }
}

$('export').addEventListener('click', async () => {
  if (!shown) return;
  const id = shown.id;
  if (made.get(id)?.exported && !window.confirm(`Replace the exported ${id} in assets/ with this preview?`)) return;
  busy = true;
  meta();
  status(`Exporting ${id}…`);
  try {
    const done = await json(`/export/${encodeURIComponent(id)}`, { method: 'POST' });
    await loadStarts();
    status(`Exported to ${done.path}: on the server's /faces now.`);
  } catch (err) {
    status(err.message, true);
  } finally {
    busy = false;
    meta();
  }
});

$('download').addEventListener('click', () => {
  const body = current();
  const blob = new Blob([JSON.stringify(body, null, 2) + '\n'], { type: 'application/json' });
  const a = el('a', { href: URL.createObjectURL(blob), download: `${body.id || 'preset'}.json` });
  document.body.append(a);
  a.click();
  setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
});

async function main() {
  const [health, library, catalogue] = await Promise.all([json('/health'), json('/library'), json('/modifiers')]);
  if (!health.ready || health.read_only) {
    $('banner').textContent = health.read_only ? 'This editor is read-only: it cannot build.' : `Cannot build: ${health.reason}`;
    $('banner').hidden = false;
    $('build').disabled = true;
  }
  buildPanels(library, catalogue);
  await loadStarts();
  const params = new URLSearchParams(location.search);
  if (params.get('remix')) {
    await remix(params.get('remix'), params.get('from') === 'export' ? 'export' : 'preview');
    history.replaceState(null, '', location.pathname);
  } else {
    // An unsent draft survives a reload; anything the server no longer offers is refused at Build.
    const draft = store.get('draft', null);
    if (draft && draft.phenotype && draft.assets) adopt(draft);
  }
  syncForm();
  ready = true;
}

// Take a preset as the form, over the defaults (a preset may leave fields out).
function adopt(source, changes = {}) {
  preset = { ...structuredClone(DEFAULT), ...structuredClone(source), targets: source.targets || {}, ...changes };
}

// Start from a saved face's preset, under a new id, so building never touches the original.
async function remix(faceId, from) {
  try {
    const base = await json(`/saved/${encodeURIComponent(faceId)}/preset?from=${from}`);
    const taken = new Set((await json('/saved')).map((r) => r.id));
    const stem = `${base.id}-remix`.slice(0, 58);
    let id = stem;
    for (let n = 2; taken.has(id); n += 1) id = `${stem}${n}`;
    adopt(base, { id, name: `${base.name} (remix)`.slice(0, 80) });
    $('start').value = '';
    status(`Remixing ${base.name} as ${id}. Build to see it; the original is untouched.`);
  } catch (err) {
    status(`Could not remix ${faceId}: ${err.message}`, true);
  }
}

$('open').addEventListener('click', () => $('open-file').click());
$('open-file').addEventListener('change', async () => {
  const file = $('open-file').files[0];
  $('open-file').value = '';
  if (!file) return;
  try {
    const loaded = JSON.parse(await file.text());
    if (!loaded || typeof loaded !== 'object' || !loaded.phenotype || !loaded.assets) throw new Error('not a preset: it needs phenotype and assets');
    adopt(loaded);
    $('start').value = '';
    syncForm();
    status(`Opened ${file.name}. Build to see it; anything this machine does not have is refused then.`);
  } catch (err) {
    status(`Could not open ${file.name}: ${err.message}`, true);
  }
});

main().catch((err) => status(err.message, true));
