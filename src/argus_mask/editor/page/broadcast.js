// Broadcast: pick a face, say something through TTS or play a file, and watch it go out.
// The editor runs the CLI's own `say` and `push` against the server; this page fills in their options.
// Everything the server sends is put on the page as text, never as HTML.

import { openScreen, screensText } from './screen.js';
import { still } from './thumbs.js';

const $ = (id) => document.getElementById(id);
const el = (tag, props = {}, ...children) => {
  const node = Object.assign(document.createElement(tag), props);
  node.append(...children);
  return node;
};
const store = {
  get(key, fallback) { try { return JSON.parse(localStorage.getItem(`mask-broadcast.${key}`)) ?? fallback; } catch { return fallback; } },
  set(key, value) { try { localStorage.setItem(`mask-broadcast.${key}`, JSON.stringify(value)); } catch { /* private window */ } },
};
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
const shell = (s) => (/^[\w@%+=:,./-]+$/.test(s) ? s : `'${String(s).replace(/'/g, "'\\''")}'`);

let state = { faces: [], screens: {}, live: [] };
// Saved faces links here with ?face=<id>: that face is chosen, and remembered like a click on it. The
// parameter is dropped from the address so a reload keeps whatever is picked after.
const asked = new URLSearchParams(location.search).get('face');
if (asked) { store.set('face', asked); history.replaceState(null, '', location.pathname); }
let face = asked || store.get('face', '');
let job = null;   // the broadcast running from this page

// ---- faces -------------------------------------------------------------------------------------

function readiness(id) {
  if ((state.live || []).includes(id)) return { text: 'speaking', cls: 'live' };
  const screens = (state.screens || {})[id] || [];
  const s = screensText(screens);
  if (s.ready) return { text: `ready · ${screens.length} screen${screens.length === 1 ? '' : 's'}`, cls: 'ready' };
  return { text: s.text, cls: screens.length ? 'warn' : '' };
}

function renderFaces() {
  const box = $('faces');
  if (!state.server_ok) { box.replaceChildren(el('div', { className: 'empty', textContent: state.reason || 'The server is not answering.' })); return; }
  if (!state.faces.length) { box.replaceChildren(el('div', { className: 'empty', textContent: 'The server serves no faces. Export one from the editor.' })); return; }
  if (!state.faces.some((f) => f.id === face)) face = state.faces[0].id;
  box.replaceChildren(...state.faces.map((f) => {
    const r = readiness(f.id);
    const radio = el('input', { type: 'radio', name: 'face', value: f.id, checked: f.id === face });
    const onScreen = el('button', { type: 'button', className: 'small', textContent: 'On screen…' });
    onScreen.onclick = (e) => { e.stopPropagation(); openScreen(f.id, f.name); };
    const thumb = el('button', { type: 'button', className: 'thumb', textContent: '…', title: `Enlarge ${f.name}` });
    showStill(f, thumb);
    thumb.onclick = (e) => { e.preventDefault(); peek(f); };
    const row = el('label', { className: `face${f.id === face ? ' on' : ''}` },
      radio, thumb,
      el('span', { className: 'name', textContent: f.name }),
      el('span', { className: 'id', textContent: f.id }),
      el('span', { className: 'badges' },
        el('span', { className: `badge ${r.cls}`, textContent: r.text }),
        ...(f.cleared === false ? [el('span', { className: 'badge', textContent: 'personal', title: 'Your own screens only: not cleared to show or share' })] : [])),
      el('span', { className: 'tools' }, onScreen));
    radio.onchange = () => choose(f.id);
    return row;
  }));
  updateTarget();
}

function choose(id) {
  face = id; store.set('face', face); renderFaces(); updateTarget();
  if (voicesFor !== null) selectVoice(wantedVoice());   // the voice this face last spoke with
  updateCli();
}

// ---- stills: a face's picture beside its name, enlarged on a click --------------------------------
// The list is redrawn on every refresh, so each still is asked for once and kept by face id. An alias shows
// the face it names. Taken big enough to enlarge; the list shows it small.

const stills = new Map();   // face id -> { url, error, ready: Promise }
function stillOf(f) {
  const id = f.kind === 'alias' && f.face ? f.face : f.id;
  if (!stills.has(id)) {
    const s = {};
    s.ready = still(id, 'export', 480, 576).then((url) => { s.url = url; }, (err) => { s.error = err.message; });
    stills.set(id, s);
  }
  return stills.get(id);
}

function showStill(f, thumb) {
  const s = stillOf(f);
  const show = () => {
    if (s.url) thumb.replaceChildren(el('img', { src: s.url, alt: '' }));
    else if (s.error) { thumb.textContent = 'none'; thumb.title = `No picture: ${s.error}`; thumb.disabled = true; }
  };
  show();
  if (!s.url && !s.error) s.ready.then(show);
}

function peek(f) {
  const s = stillOf(f);
  if (!s.url) return;
  $('peek-img').src = s.url;
  $('peek-img').alt = f.name;
  $('peek-name').textContent = f.name;
  $('peek-pick').onclick = () => { $('peek').close(); choose(f.id); };
  $('peek').showModal();
}
$('peek-close').onclick = () => $('peek').close();
$('peek').onclick = (e) => { if (e.target === $('peek')) $('peek').close(); };   // a click outside the picture

function updateTarget() {
  const t = $('target');
  if (!face) { t.textContent = ''; return; }
  const name = state.faces.find((f) => f.id === face)?.name || face;
  const r = readiness(face);
  const warn = state.read_only ? 'The server is read-only: it will refuse to speak.'
    : state.driver && state.driver.available === false ? `The driver is down: ${state.driver.reason || 'no reason given'}.`
    : r.cls === 'live' ? `${name} is speaking now; a new broadcast waits its turn or is refused.`
    : r.cls === 'ready' ? '' : `${name}: ${r.text}. It will still speak, but no one may hear it.`;
  t.textContent = warn;
  t.className = 'warnline';
  $('send').textContent = `Speak as ${name}`;
  $('push-send').textContent = `Play as ${name}`;
}

async function refresh() {
  try { state = await json('/broadcast/state'); } catch (err) { state = { server_ok: false, reason: err.message, faces: [] }; }
  const s = $('server');
  s.className = state.server_ok ? 'ok' : 'bad';
  s.replaceChildren(el('b', { textContent: '● ' }), `Server ${state.server || ''}${state.server_ok ? '' : ': not answering'}${state.read_only ? ' (read-only)' : ''}`);
  if (!$('tts').value && state.tts_url) { $('tts').value = state.tts_url; loadVoices(); }
  const rec = state.record || { ok: false, reason: 'the editor did not say whether it can record' };
  $('record').disabled = !rec.ok;
  if (!rec.ok) $('record').checked = false;
  $('rec-hint').textContent = rec.ok ? `Saved in ${rec.dir}/, rendered once the broadcast ends.` : rec.reason;
  renderFaces();
  updateCli();
}

// ---- voices --------------------------------------------------------------------------------------

// The service's own voices when it lists them, then voices typed before (kept in this browser, so a
// service that cannot list its speakers, like rag-chat's XTTS, still gets a pick list). Each face
// remembers the voice it last spoke with: choosing it chooses that voice again.

const OTHER = '\u0001other';
const RECENT_MAX = 16;
const faceVoice = (id) => (id ? store.get(`voice:${id}`, null) : null);
function wantedVoice() {
  return faceVoice(face) ?? (store.get('voice', '') || state.tts_voice || '');
}
function rememberVoice(value, typed) {
  store.set('voice', value);
  if (face) store.set(`voice:${face}`, value);
  if (typed && value) store.set('voices', [value, ...store.get('voices', []).filter((v) => v !== value)].slice(0, RECENT_MAX));
}
function selectVoice(want) {
  const select = $('voice');
  const known = [...select.options].some((o) => o.value === want);
  if (want && !known) { select.value = OTHER; $('voice-other').value = want; } else select.value = want;
  $('voice-other-wrap').hidden = select.value !== OTHER;
}

let voicesFor = null;
let listed = [];   // the service's voices, from the last /voices
const VOICES_RETRY_MS = 15000;
async function loadVoices(again = false) {
  const tts = $('tts').value.trim();
  if (tts === voicesFor) return;
  voicesFor = tts;
  const select = $('voice');
  let found = { listed: false, voices: [] };
  try { found = await json(`/voices?tts=${encodeURIComponent(tts)}`); } catch { /* keep the free-text fallback */ }
  if (voicesFor !== tts) return;   // the TTS changed while this list was asked for: the latest call fills the select
  const keep = again ? (select.value === OTHER ? $('voice-other').value.trim() : select.value) : null;
  listed = found.voices || [];
  fillVoices();
  select.title = found.listed ? `${listed.length} voices from the service` : (found.reason || '');
  selectVoice(keep ?? wantedVoice());   // a retry keeps what was picked meanwhile
  updateCli();
  // A cold service (rag-chat while XTTS loads) answers its speaker list with an error, or none: ask again
  // until it has some, rather than leaving the list empty until a reload.
  if (!listed.length && tts) setTimeout(() => { if (voicesFor === tts) { voicesFor = null; loadVoices(true); } }, VOICES_RETRY_MS);
}

function fillVoices() {
  const options = [el('option', { value: '', textContent: state.tts_voice ? `default (${state.tts_voice})` : "the service's default" })];
  const seen = new Set(['']);
  if (listed.length) {
    const group = el('optgroup', { label: 'From the service' });
    for (const v of listed) {
      const langs = v.languages && v.languages.length ? ` · ${v.languages.join(', ')}` : '';
      const desc = v.description && v.description !== v.name ? ` (${v.description})` : '';
      group.append(el('option', { value: v.name, textContent: `${v.name}${desc}${langs}` }));
      seen.add(v.name);
      for (const sp of v.speakers || []) group.append(el('option', { value: `${v.name}\u0000${sp}`, textContent: `${v.name} · speaker ${sp}` }));
    }
    options.push(group);
  }
  // The last voice typed before this list existed counts as used before too.
  const recent = [...new Set([...store.get('voices', []), store.get('voice', '')])]
    .filter((v) => v && !v.includes('\u0000') && !seen.has(v));
  if (recent.length) options.push(el('optgroup', { label: 'Used before' }, ...recent.map((v) => el('option', { value: v, textContent: v }))));
  options.push(el('option', { value: OTHER, textContent: 'other: type a name or speaker…' }));
  $('voice').replaceChildren(...options);
}

// The voice as the CLI's --voice takes it, and any speaker as a -p field (Wyoming's voice.speaker).
function chosenVoice() {
  const v = $('voice').value;
  if (v === OTHER) return { voice: $('voice-other').value.trim(), speaker: '' };
  const [voice, speaker = ''] = v.split('\u0000');
  return { voice, speaker };
}

// ---- extra fields --------------------------------------------------------------------------------

function paramRow(key = '', value = '') {
  const k = el('input', { type: 'text', placeholder: 'key', value: key });
  const v = el('input', { type: 'text', placeholder: 'value', value });
  const del = el('button', { type: 'button', className: 'small', textContent: '×', title: 'Remove' });
  const row = el('div', { className: 'row' }, k, v, del);
  del.onclick = () => { row.remove(); updateCli(); };
  for (const i of [k, v]) i.addEventListener('input', updateCli);
  $('params').append(row);
}
function params() {
  const out = {};
  for (const row of $('params').children) {
    const [k, v] = row.querySelectorAll('input');
    const key = k.value.trim();
    if (!key) continue;
    let value = v.value;
    try { const parsed = JSON.parse(value); if (['number', 'boolean'].includes(typeof parsed)) value = parsed; } catch { /* a string */ }
    out[key] = value;
  }
  const { speaker } = chosenVoice();
  if (speaker && !('speaker' in out)) out.speaker = speaker;
  // The slider, unless left to the service (or a temperature typed as an extra field already says)
  if (!$('temp-default').checked && !('temperature' in out)) out.temperature = Number($('temp').value);
  return out;
}

function paintTemperature() {
  $('temp-out').textContent = $('temp-default').checked ? '—' : Number($('temp').value).toFixed(2);
  $('temp').disabled = $('temp-default').checked;
}

// ---- the command line, kept equal to the form --------------------------------------------------------

function updateCli() {
  const text = $('text').value;
  $('count').textContent = `${text.length} / 5000`;
  const { voice } = chosenVoice();
  const parts = ['argus-mask say', shell(text || '…'), '--face', shell(face || '<id>'), '--url', shell(state.server || 'http://127.0.0.1:8109')];
  const tts = $('tts').value.trim();
  if (tts && tts !== state.tts_url) parts.push('--tts-url', shell(tts));
  if (voice) parts.push('--voice', shell(voice));
  for (const [k, v] of Object.entries(params())) parts.push('-p', shell(`${k}=${typeof v === 'string' ? v : JSON.stringify(v)}`));
  if (Number($('pause').value) !== 300) parts.push('--pause', String(Number($('pause').value) || 0));
  if (Number($('seed').value)) parts.push('--seed', String(Number($('seed').value)));
  if ($('stream').checked) parts.push('--stream');
  if (!$('tidy').checked) parts.push('--as-is');
  if ($('context').checked) parts.push('--context');
  $('cli').value = parts.join(' ');
  const file = $('file').files[0];
  $('push-cli').value = ['argus-mask push', shell(file ? file.name : '<file.wav>'), '--face', shell(face || '<id>'),
    '--url', shell(state.server || 'http://127.0.0.1:8109'), ...(Number($('push-seed').value) ? ['--seed', String(Number($('push-seed').value))] : [])].join(' ');
  store.set('form', { tts, pause: $('pause').value, seed: $('seed').value, stream: $('stream').checked, tidy: $('tidy').checked, context: $('context').checked,
    temp: $('temp').value, tempDefault: $('temp-default').checked });
}

// ---- sending, and what happens -------------------------------------------------------------------

// Record to file: its options, or null when not asked for (or not possible here).
function recording() {
  if (!$('record').checked || $('record').disabled) return null;
  return { size: $('rec-size').value, quality: $('rec-quality').value };
}

async function loadRecordings() {
  let clips = [];
  try { clips = await json('/recordings'); } catch { /* none to show */ }
  const box = $('recordings');
  if (!clips.length) { box.replaceChildren(); return; }
  box.replaceChildren(el('h3', { textContent: 'Recordings' }), ...clips.slice(0, 8).map((c) => {
    const href = `/recordings/${encodeURIComponent(c.name)}`;
    return el('div', {}, el('a', { href, target: '_blank', rel: 'noopener', textContent: c.name }),
      el('span', { textContent: `${(c.bytes / 1048576).toFixed(1)} MB · ${new Date(c.modified * 1000).toLocaleString()}` }));
  }));
}

function renderJob() {
  const log = $('log');
  if (!job) { log.replaceChildren(); return; }
  log.replaceChildren(
    el('div', { className: 'head', textContent: `${job.kind === 'say' ? 'Saying' : 'Playing'} as ${job.face}: ${job.summary} · ${job.state}` }),
    ...job.events.map((e) => el('div', { className: `ev ${e.kind}` }, el('span', { className: 'at', textContent: `${e.at.toFixed(1)}s` }), el('span', { textContent: e.kind === 'file' ? `recorded ${e.text}` : e.text }))),
    // The clip itself, once written: to watch here, or to keep.
    ...(job.files || []).map((name) => {
      const href = `/recordings/${encodeURIComponent(name)}`;
      return el('div', { className: 'clip' }, el('video', { src: href, controls: true, preload: 'metadata' }),
        el('a', { href, download: name, textContent: `Download ${name}` }));
    }),
  );
  const running = job.state === 'running';
  for (const b of ['send', 'push-send']) $(b).disabled = running;
  for (const b of ['stop', 'push-stop']) $(b).disabled = !running;
}

async function follow(id, misses = 0) {
  try {
    job = await json(`/broadcast/${id}`);
  } catch (err) {
    // A poll that fails says nothing about the broadcast, which runs on in the editor: keep following
    // (and Stop working) for a while before giving up on it.
    if (misses < 10) { setTimeout(() => follow(id, misses + 1), 1000); return; }
    job = { ...(job || {}), state: 'lost', events: [...(job?.events || []), { at: 0, kind: 'error', text: err.message }] };
  }
  renderJob();
  if (job.state === 'running') setTimeout(() => follow(id), 500);
  else { refresh(); loadRecordings(); }
}

async function start(request) {
  for (const b of ['send', 'push-send']) $(b).disabled = true;
  try {
    job = await request();
    renderJob();
    follow(job.id);
    refresh();
  } catch (err) {
    job = { kind: '', face, summary: '', state: 'refused', events: [{ at: 0, kind: 'error', text: err.message }] };
    renderJob();
    for (const b of ['send', 'push-send']) $(b).disabled = false;
  }
}

$('say').addEventListener('submit', (e) => {
  e.preventDefault();
  const { voice } = chosenVoice();
  const typed = $('voice').value === OTHER;
  rememberVoice(typed ? voice : $('voice').value, typed);
  if (typed) { fillVoices(); selectVoice(voice); }
  start(() => json('/broadcast/say', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ face, text: $('text').value, tts_url: $('tts').value.trim(), voice, params: params(),
      seed: Number($('seed').value) || 0, stream: $('stream').checked, tidy: $('tidy').checked, context: $('context').checked, pause: Number($('pause').value) || 0,
      ...(recording() ? { record: recording() } : {}) }),
  }));
});
$('push').addEventListener('submit', (e) => {
  e.preventDefault();
  const file = $('file').files[0];
  if (!file) return;
  const rec = recording();
  const also = rec ? `&record=1&record_size=${rec.size}&record_quality=${rec.quality}` : '';
  start(() => json(`/broadcast/push?face=${encodeURIComponent(face)}&seed=${Number($('push-seed').value) || 0}&name=${encodeURIComponent(file.name)}${also}`,
    { method: 'POST', headers: { 'Content-Type': 'audio/wav' }, body: file }));
});
for (const b of ['stop', 'push-stop']) $(b).onclick = () => job && json(`/broadcast/${job.id}/stop`, { method: 'POST' }).catch(() => {});
for (const b of document.querySelectorAll('.tabs button')) {
  b.onclick = () => {
    for (const x of document.querySelectorAll('.tabs button')) x.classList.toggle('on', x === b);
    $('say').hidden = b.dataset.tab !== 'say';
    $('push').hidden = b.dataset.tab !== 'push';
  };
}
const copy = (input, button) => async () => {
  try { await navigator.clipboard.writeText($(input).value); } catch { $(input).select(); document.execCommand('copy'); }
  const was = $(button).textContent; $(button).textContent = 'Copied'; setTimeout(() => { $(button).textContent = was; }, 1200);
};
$('copy-cli').onclick = copy('cli', 'copy-cli');
$('copy-push-cli').onclick = copy('push-cli', 'copy-push-cli');
$('add-param').onclick = () => paramRow();
$('voice').addEventListener('change', () => {
  const v = $('voice').value;
  $('voice-other-wrap').hidden = v !== OTHER;
  if (v !== OTHER) rememberVoice(v, false);   // a pick from the list sticks to the face at once
  updateCli();
});
$('tts').addEventListener('change', loadVoices);
for (const id of ['text', 'voice-other', 'pause', 'seed', 'stream', 'tidy', 'context', 'tts', 'file', 'push-seed']) $(id).addEventListener('input', updateCli);
for (const id of ['temp', 'temp-default']) $(id).addEventListener('input', () => { paintTemperature(); updateCli(); });
$('text').addEventListener('keydown', (e) => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey) && !$('send').disabled) $('say').requestSubmit(); });

const savedRec = store.get('record', {});
if (savedRec.size) $('rec-size').value = savedRec.size;
if (savedRec.quality) $('rec-quality').value = savedRec.quality;
$('record').checked = Boolean(savedRec.on);
for (const id of ['record', 'rec-size', 'rec-quality']) {
  $(id).addEventListener('input', () => store.set('record', { on: $('record').checked, size: $('rec-size').value, quality: $('rec-quality').value }));
}
loadRecordings();
const saved = store.get('form', {});
if (saved.tts) $('tts').value = saved.tts;
if (saved.pause !== undefined) $('pause').value = saved.pause;
if (saved.seed !== undefined) $('seed').value = saved.seed;
$('stream').checked = Boolean(saved.stream);
$('tidy').checked = saved.tidy ?? true;
$('context').checked = Boolean(saved.context);
if (saved.temp !== undefined) $('temp').value = saved.temp;
if (saved.tempDefault !== undefined) $('temp-default').checked = saved.tempDefault;
paintTemperature();
await refresh();
await loadVoices();
// A broadcast already running from this editor (a reload mid-sentence): follow it again.
if (state.jobs && state.jobs.length) { job = state.jobs[0]; renderJob(); follow(job.id); }
setInterval(refresh, 3000);
