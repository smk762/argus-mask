// The Voices page: check reference clips, generate synthetic candidates, make a voice with its provenance, try one.
// Everything the server or the TTS service sends is put on the page as text, never as HTML.

const $ = (id) => document.getElementById(id);
const el = (tag, props = {}, ...children) => {
  const node = Object.assign(document.createElement(tag), props);
  node.append(...children.filter((c) => c !== null && c !== undefined));
  return node;
};
const store = {
  get(key, fallback) { try { return JSON.parse(localStorage.getItem(`mask-voices.${key}`)) ?? fallback; } catch { return fallback; } },
  set(key, value) { try { localStorage.setItem(`mask-voices.${key}`, JSON.stringify(value)); } catch { /* private window */ } },
};
async function json(url, options) {
  const reply = await fetch(url, options);
  const body = await reply.json().catch(() => ({}));
  if (!reply.ok) throw new Error(typeof body.detail === 'string' ? body.detail : (body.detail ? JSON.stringify(body.detail) : reply.statusText));
  return body;
}
const post = (url, body) => json(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });

// Scripts for generated candidates: plain words (no digits), complete sentences, a question, about 15 s read aloud.
const SCRIPTS = {
  neutral: 'Good morning. I keep an eye on the systems, the schedule and the weather, and I read out what matters. Today there are three things to look at, and none of them is urgent. Shall we start with the first one?',
  warm: 'Hello again, it is lovely to hear from you. Everything is running smoothly, so there is nothing to worry about this morning. I have put the three things that matter at the top for you. Would you like to take them one at a time?',
  bright: 'Good morning! Great news to start the day: every system is up and every test has passed. There are three new things waiting for you, and they all look interesting. Shall we dive in?',
  calm: 'Take your time. Everything is steady, and nothing needs your attention right now. When you are ready, there are three small things to look at. We can go through them slowly, one after another.',
  brisk: 'Right, here is where things stand. Systems are up, the backlog is moving, and three items need a decision today. I will keep it short. Ready for the first one?',
  serious: 'I need your attention for a moment. One of the systems reported a fault overnight, and it has not recovered on its own. I have the details ready, along with two possible fixes. Shall I go through them now?',
  playful: 'Well, look who it is! The systems behaved themselves overnight, which makes a nice change. There are three little puzzles waiting for you this morning. Want to guess which one I like best?',
  other: 'Good morning. I keep an eye on the systems, the schedule and the weather, and I read out what matters. Today there are three things to look at, and none of them is urgent. Shall we start with the first one?',
};

let state = { tts_url: '', can_make: false, voices: [], deliveries: [], sources: [] };
const clips = []; // { key, name, sha, transcript, generated, generator, use, report, url, error, checking }
let setReport = null;

// ---- small pieces ----------------------------------------------------------------------------------------

const verdictClass = (v) => (v === 'good' ? 'good' : v === 'usable' ? 'warn' : v === 'not suitable' ? 'bad' : v);
function checksTable(report, all = false) {
  const rows = report.checks.filter((c) => all || c.verdict !== 'good');
  if (!rows.length) return el('div', { className: 'hint', textContent: 'Every check passed.' });
  return el('table', { className: 'checks' }, ...rows.map((c) => el('tr', {},
    el('td', {}, el('span', { className: `dot ${c.verdict}`, title: c.verdict })),
    el('td', {}, el('b', { textContent: c.label }), ` ${c.value}`, c.advice ? el('div', { className: 'advice', textContent: c.advice }) : null))));
}
function drawWave(canvas, samples) {
  const w = canvas.width = canvas.clientWidth * devicePixelRatio || 600;
  const h = canvas.height = 48 * devicePixelRatio;
  const g = canvas.getContext('2d');
  g.fillStyle = '#6ea8fe';
  const step = Math.max(1, Math.floor(samples.length / w));
  for (let x = 0; x < w; x++) {
    let peak = 0;
    for (let i = x * step; i < Math.min(samples.length, (x + 1) * step); i++) peak = Math.max(peak, Math.abs(samples[i]));
    if (peak >= 0.999) g.fillStyle = '#f28b82'; else g.fillStyle = '#6ea8fe';
    const bar = Math.max(1, peak * h);
    g.fillRect(x, (h - bar) / 2, 1, bar);
  }
}
async function samplesOf(blob) {
  try {
    const ctx = new (window.OfflineAudioContext || window.webkitOfflineAudioContext)(1, 1, 48000);
    const buffer = await ctx.decodeAudioData(await blob.arrayBuffer());
    return buffer.getChannelData(0);
  } catch { return null; }
}
// Any audio the browser can decode, as a 16-bit WAV at 48 kHz: for files that are not WAV already.
function wavFrom(buffer) {
  const n = buffer.length; const ch = buffer.numberOfChannels; const rate = buffer.sampleRate;
  const out = new DataView(new ArrayBuffer(44 + n * 2));
  const str = (o, s) => [...s].forEach((c, i) => out.setUint8(o + i, c.charCodeAt(0)));
  str(0, 'RIFF'); out.setUint32(4, 36 + n * 2, true); str(8, 'WAVE'); str(12, 'fmt ');
  out.setUint32(16, 16, true); out.setUint16(20, 1, true); out.setUint16(22, 1, true); out.setUint32(24, rate, true);
  out.setUint32(28, rate * 2, true); out.setUint16(32, 2, true); out.setUint16(34, 16, true); str(36, 'data'); out.setUint32(40, n * 2, true);
  const chans = [...Array(ch).keys()].map((c) => buffer.getChannelData(c));
  for (let i = 0; i < n; i++) {
    let s = 0; for (const c of chans) s += c[i]; s = Math.max(-1, Math.min(1, s / ch));
    out.setInt16(44 + i * 2, Math.round(s * 32767), true);
  }
  return new Blob([out.buffer], { type: 'audio/wav' });
}

// ---- clips -----------------------------------------------------------------------------------------------

async function check(clip, blob) {
  clip.checking = true; clip.error = null; paintClips();
  try {
    if (!blob && !clip.url && !clip.sha) throw new Error('This clip could not be read: remove it and add the file again.');
    const body = blob ?? await (await fetch(clip.url || `/voice/clip/${clip.sha}`)).blob();
    const q = new URLSearchParams({ name: clip.name, transcript: clip.transcript });
    const done = await json(`/voice/check?${q}`, { method: 'POST', headers: { 'Content-Type': 'audio/wav' }, body });
    clip.sha = done.sha; clip.report = done.report;
  } catch (err) { clip.error = err.message; }
  clip.checking = false;
  paintClips();
  assessSet();
}

async function addFiles(files) {
  for (const file of files) {
    let blob = file;
    if (!/\.wav$/i.test(file.name) && !/wav/.test(file.type)) {
      try {
        const ctx = new OfflineAudioContext(1, 1, 48000);
        blob = wavFrom(await ctx.decodeAudioData(await file.arrayBuffer()));
      } catch { clips.push({ key: crypto.randomUUID?.() ?? String(Math.random()), name: file.name, transcript: '', use: false, error: 'The browser cannot decode this file: try WAV or FLAC.' }); paintClips(); continue; }
    }
    const clip = { key: String(Math.random()).slice(2), name: file.name.replace(/\.[^.]+$/, '') + '.wav', transcript: '', generated: false, use: true, url: URL.createObjectURL(blob) };
    clip.samples = await samplesOf(blob);
    clips.push(clip);
    await check(clip, blob);
  }
}

function paintClips() {
  $('clips').replaceChildren(...clips.map((clip, i) => {
    const transcript = el('textarea', { value: clip.transcript, placeholder: 'Exactly what is said, word for word, with capitals and punctuation. Numbers as words.', readOnly: Boolean(clip.generated) });
    transcript.addEventListener('change', () => { clip.transcript = transcript.value.trim(); check(clip); });
    const use = el('input', { type: 'checkbox', checked: clip.use, disabled: !clip.report });
    use.addEventListener('change', () => { clip.use = use.checked; paintClips(); assessSet(); });
    const remove = el('button', { type: 'button', className: 'small', textContent: 'Remove' });
    remove.onclick = () => { clips.splice(i, 1); paintClips(); assessSet(); };
    const up = el('button', { type: 'button', className: 'small', textContent: '↑', title: 'Earlier: Miso keeps the first 20 s', disabled: i === 0 });
    up.onclick = () => { [clips[i - 1], clips[i]] = [clips[i], clips[i - 1]]; paintClips(); assessSet(); };
    const canvas = el('canvas');
    const r = clip.report;
    const card = el('div', { className: `clip${clip.use ? '' : ' off'}` },
      el('div', { className: 'top' },
        el('label', { className: 'check', title: 'Use this clip for the voice' }, use),
        el('span', { className: 'title', textContent: clip.name }),
        clip.generated ? el('span', { className: 'badge', textContent: 'generated' }) : null,
        clip.checking ? el('span', { className: 'spin' }) : null,
        r ? el('span', { className: `badge ${verdictClass(r.verdict)}`, textContent: `${r.verdict} · ${r.score}` }) : null,
        r ? el('span', { className: 'hint', textContent: `${r.metrics.speech_seconds}s of speech` }) : null,
        el('span', { style: 'margin-left:auto' }), up, remove),
      clip.samples ? canvas : null,
      clip.url || clip.sha ? el('audio', { controls: true, src: clip.url || `/voice/clip/${clip.sha}`, preload: 'metadata' }) : null,
      el('label', {}, clip.generated ? 'Transcript (the script it read)' : 'Transcript', transcript),
      clip.error ? el('div', { className: 'err', textContent: clip.error }) : null,
      r ? checksTable(r) : null,
      r ? el('details', {}, el('summary', { className: 'hint', textContent: 'All checks and measurements' }), checksTable(r, true),
        el('pre', { className: 'hint', textContent: JSON.stringify(r.metrics, null, 1) })) : null);
    if (clip.samples) requestAnimationFrame(() => drawWave(canvas, clip.samples));
    return card;
  }));
  if (!clips.length) $('clips').append(el('div', { className: 'hint', textContent: 'No clips yet.' }));
}

async function assessSet() {
  const used = clips.filter((c) => c.use && c.report);
  setReport = used.length ? await post('/voice/assess', used.map((c) => c.report)).catch(() => null) : null;
  const box = $('set');
  if (!setReport) { box.replaceChildren(el('span', { className: 'hint', textContent: 'Add a clip to see whether the set is enough.' })); paintMake(); return; }
  box.replaceChildren(
    el('span', { className: `badge ${verdictClass(setReport.verdict)}`, textContent: setReport.verdict }),
    el('span', { textContent: `${used.length} clip${used.length === 1 ? '' : 's'}, ${setReport.metrics.speech_seconds}s of speech` }),
    el('div', { style: 'flex-basis:100%' }, checksTable(setReport)));
  paintMake();
}

// ---- generate --------------------------------------------------------------------------------------------

async function generate() {
  const script = $('gen-script').value.trim();
  const speaker = $('gen-speaker').value.trim(); const desc = $('gen-desc').value.trim();
  const temperature = Number($('gen-temp').value);
  $('gen-go').disabled = true; $('gen-note').replaceChildren(el('span', { className: 'spin' }), ' Generating…');
  try {
    let job = await post('/voice/generate', { script, speaker, description: desc, temperature });
    while (job.state === 'running') {
      await new Promise((r) => setTimeout(r, 1500));
      job = await json(`/voice/generate/${job.id}`);
      $('gen-note').replaceChildren(el('span', { className: 'spin' }), ` Generating… ${Math.round(job.seconds)}s`);
    }
    if (job.state !== 'done') throw new Error(job.error || 'it failed');
    const n = clips.filter((c) => c.generated).length + 1;
    const generator = desc ? `described voice "${desc}", temperature ${temperature}` : `speaker ${speaker || '0'}, temperature ${temperature}`;
    const clip = { key: job.id, name: `candidate-${n}.wav`, sha: job.result.sha, transcript: job.result.transcript, generated: true,
      generator: `${state.tts_url} ${generator}`, use: clips.every((c) => !c.use), report: job.result.report };
    clip.samples = await samplesOf(await (await fetch(`/voice/clip/${clip.sha}`)).blob());
    clips.push(clip);
    $('gen-note').textContent = `Candidate ${n} ready in ${Math.round(job.seconds)}s: listen to all of it before you use it.`;
    if (!$('v-generator').value) $('v-generator').value = clip.generator;
    paintClips(); assessSet();
  } catch (err) { $('gen-note').textContent = err.message; }
  $('gen-go').disabled = false;
}

// ---- make ------------------------------------------------------------------------------------------------

function source() {
  const kind = $('v-source').value;
  return { kind, confirmed: $('v-confirm').checked, generator: $('v-generator').value.trim(),
    speaker: $('v-speaker').value.trim(), release_sha256: $('v-release').value.trim() };
}
function paintMake() {
  const kind = $('v-source').value;
  $('v-generator-wrap').hidden = kind !== 'generated';
  $('v-consent-wrap').hidden = kind !== 'consented';
  const used = clips.filter((c) => c.use && c.report);
  const why = !state.can_make ? state.reason
    : !used.length ? 'Choose at least one checked clip.'
      : used.some((c) => !c.transcript) ? 'Every clip needs its transcript.'
        : used.some((c) => c.report.verdict === 'not suitable') ? 'A chosen clip is not suitable: fix or drop it.'
          : setReport?.verdict === 'not suitable' ? 'The set is not enough yet.'
            : !/^[a-z0-9][a-z0-9-]{0,62}$/.test($('v-id').value) ? 'An id: lower case letters, digits and dashes.'
              : !$('v-name').value.trim() ? 'A name.'
                : !$('v-confirm').checked ? 'Confirm where the clips come from.' : '';
  $('v-make').disabled = Boolean(why);
  $('v-note').textContent = why;
}
async function make() {
  const used = clips.filter((c) => c.use && c.report);
  $('v-make').disabled = true; $('v-note').replaceChildren(el('span', { className: 'spin' }), ' Making the voice on the TTS service…');
  try {
    const done = await post('/voice/create', {
      id: $('v-id').value, name: $('v-name').value.trim(), delivery: $('v-delivery').value, source: source(),
      clips: used.map((c) => ({ sha: c.sha, name: c.name, transcript: c.transcript, generated: Boolean(c.generated) })),
    });
    $('v-note').textContent = `Made ${done.id}: ${done.check.speech_seconds}s of speech, ${done.check.verdict}. Try it on the left, or pick it on the Broadcast page.`;
    await refresh();
    $('try-voice').value = done.id;
  } catch (err) { $('v-note').textContent = err.message; $('v-make').disabled = false; }
}

// ---- voices and trying them ------------------------------------------------------------------------------

async function refresh() {
  state = await json('/voice/state');
  $('tts').textContent = state.tts_url ? `TTS: ${state.tts_url}` : 'No TTS service set (ARGUS_MASK_TTS_URL)';
  let listed = [];
  try { const v = await json('/voices'); listed = v.listed ? v.voices : []; } catch { /* the service may be down */ }
  const records = new Map(state.voices.map((r) => [r.id, r]));
  const box = $('voices');
  const cards = state.voices.map((r) => {
    const del = el('button', { type: 'button', className: 'small', textContent: 'Delete' });
    del.onclick = async () => {
      if (!confirm(`Delete ${r.id} from the TTS service? Its record and clips go to the trash folder here.`)) return;
      try { await json(`/voice/${encodeURIComponent(r.id)}`, { method: 'DELETE' }); refresh(); } catch (err) { alert(err.message); }
    };
    const src = r.source.kind === 'generated' ? 'generated' : r.source.kind === 'own' ? 'own voice' : `consented: ${r.source.speaker}`;
    return el('div', { className: 'voice' },
      el('div', { className: 'top' }, el('span', { className: 'name', textContent: r.name }), el('span', { className: 'badge', textContent: r.delivery }),
        el('span', { className: `badge ${verdictClass(r.check.verdict)}`, textContent: r.check.verdict }), el('span', { style: 'margin-left:auto' }), del),
      el('span', { className: 'id', textContent: r.id }),
      el('span', { className: 'hint', textContent: `${src} · ${r.check.speech_seconds}s · ${new Date(r.created * 1000).toLocaleDateString()}` }));
  });
  const orphans = listed.filter((v) => v.name.startsWith('clone:') && !records.has(v.name.slice(6)));
  if (orphans.length) {
    cards.push(el('div', { className: 'hint', style: 'margin-top:8px', textContent: 'On the TTS service with no record here (made elsewhere: where they came from is not known):' }),
      ...orphans.map((v) => el('div', { className: 'voice' }, el('div', { className: 'top' }, el('span', { className: 'name', textContent: v.description || v.name.slice(6) }),
        el('span', { className: 'badge warn', textContent: 'no record' })), el('span', { className: 'id', textContent: v.name.slice(6) }))));
  }
  box.replaceChildren(...(cards.length ? cards : [el('div', { className: 'hint', textContent: 'No voices yet.' })]));
  const pick = $('try-voice'); const was = pick.value || store.get('try', '');
  pick.replaceChildren(...[...state.voices.map((r) => [r.id, `${r.name} (${r.delivery})`]),
    ...listed.filter((v) => !records.has(v.name.replace(/^clone:/, ''))).map((v) => [v.name, `${v.description || v.name}${v.name.startsWith('clone:') ? '' : ' (service speaker)'}`])]
    .map(([value, text]) => el('option', { value, textContent: text })));
  if ([...pick.options].some((o) => o.value === was)) pick.value = was;
  paintMake();
}
async function tryVoice() {
  $('try-go').disabled = true; $('try-note').replaceChildren(el('span', { className: 'spin' }), ' Speaking…');
  store.set('try', $('try-voice').value);
  try {
    const reply = await fetch('/voice/try', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ voice: $('try-voice').value, text: $('try-text').value, temperature: Number($('try-temp').value) }) });
    if (!reply.ok) throw new Error((await reply.json().catch(() => ({}))).detail || reply.statusText);
    const audio = $('try-audio'); audio.hidden = false; audio.src = URL.createObjectURL(await reply.blob()); audio.play().catch(() => {});
    $('try-note').textContent = '';
  } catch (err) { $('try-note').textContent = err.message; }
  $('try-go').disabled = false;
}

// ---- wiring ----------------------------------------------------------------------------------------------

for (const b of document.querySelectorAll('.tabs button')) {
  b.onclick = () => {
    for (const o of document.querySelectorAll('.tabs button')) o.classList.toggle('on', o === b);
    $('tab-upload').hidden = b.dataset.tab !== 'upload'; $('tab-generate').hidden = b.dataset.tab !== 'generate';
  };
}
$('files').addEventListener('change', (e) => { addFiles([...e.target.files]); e.target.value = ''; });
$('gen-go').onclick = generate;
$('v-make').onclick = make;
$('try-go').onclick = tryVoice;
for (const [range, out] of [['gen-temp', 'gen-temp-out'], ['try-temp', 'try-temp-out']]) {
  const paint = () => { $(out).textContent = Number($(range).value).toFixed(2); };
  $(range).addEventListener('input', paint); paint();
}
for (const id of ['v-id', 'v-name', 'v-confirm', 'v-source', 'v-generator', 'v-speaker', 'v-release']) $(id).addEventListener('input', paintMake);
$('v-name').addEventListener('input', () => {
  if ($('v-id').dataset.touched) return;
  const base = $('v-name').value.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
  $('v-id').value = base ? `${base}-${$('v-delivery').value}`.slice(0, 63) : '';
  paintMake();
});
$('v-id').addEventListener('input', () => { $('v-id').dataset.touched = '1'; });

await refresh().catch((err) => { $('tts').textContent = err.message; });
for (const sel of ['gen-delivery', 'v-delivery']) $(sel).replaceChildren(...state.deliveries.map((d) => el('option', { value: d, textContent: d })));
$('gen-delivery').addEventListener('change', () => {
  $('gen-script').value = SCRIPTS[$('gen-delivery').value] || SCRIPTS.neutral;
  $('v-delivery').value = $('gen-delivery').value;
  $('v-name').dispatchEvent(new Event('input'));
});
$('v-delivery').addEventListener('change', () => $('v-name').dispatchEvent(new Event('input')));
$('gen-script').value = SCRIPTS.neutral;
paintClips();
paintMake();
