// Guided head builder (#174): one page that reads the build record and runs each step.
// Everything shown (checks, scores, refusals, blockers) comes from GET /guided/{id}. The page never invents a verdict.

import { Stage, talkReady } from './stage.js';

const $ = (id) => document.getElementById(id);
const el = (tag, props = {}, ...children) => {
  const node = Object.assign(document.createElement(tag), props);
  node.append(...children);
  return node;
};
const enc = encodeURIComponent;

function status(text, bad = false) {
  const node = $('status');
  node.textContent = text || '';
  node.classList.toggle('bad', !!bad);
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

const CANDIDATE_STEPS = ['source', 'portrait', 'build', 'likeness', 'rest', 'turns', 'speech', 'age'];
const EVAL_STEPS = ['likeness', 'rest', 'turns', 'speech', 'age'];
const VERDICT_CLASS = {
  passed: 'good', overruled: 'good', advisory: 'warn', stale: 'stale',
  refused: 'bad', failed: 'bad', not_checked: 'bad', missing: '',
};

let catalog = null;   // GET /guided
let record = null;    // GET /guided/{id} view
let buildId = null;
let activeCandidate = null;
const stages = new Map();   // candidate -> Stage for pick previews

function stepMeta(name) {
  return (catalog?.steps || []).find((s) => s.name === name) || { name, about: name, needs: [] };
}

function resultOf(step, candidate) {
  if (!record) return null;
  const want = candidate ?? null;
  return record.results.find((r) => r.step === step && (r.candidate ?? null) === want) || null;
}

function fileUrl(file) {
  return `/guided/${enc(buildId)}/file/${file.split('/').map(enc).join('/')}`;
}

function badge(statusName) {
  return el('span', { className: `badge ${VERDICT_CLASS[statusName] || ''}`, textContent: statusName });
}

function checksList(result) {
  if (!result?.checks?.length) return el('p', { className: 'hint', textContent: 'No checks yet.' });
  const overruled = new Set((result.overruled || []).map((o) => o.check));
  const box = el('div', { className: 'checks' });
  for (const c of result.checks) {
    const row = el('div', { className: `check ${c.verdict}` });
    row.append(
      el('span', { className: 'name', textContent: c.name }),
      el('span', { className: 'reason', textContent: c.reason || c.verdict }),
    );
    if (c.verdict === 'advisory' && !overruled.has(c.name)) {
      const btn = el('button', { type: 'button', className: 'small', textContent: 'Overrule' });
      btn.onclick = () => overrule(result.step, result.candidate, c.name);
      row.append(btn);
    } else if (overruled.has(c.name)) {
      const why = (result.overruled || []).find((o) => o.check === c.name);
      row.append(el('span', { className: 'badge good', textContent: 'overruled', title: why?.reason || '' }));
    } else {
      row.append(el('span', { textContent: '' }));
    }
    box.append(row);
  }
  return box;
}

function scoresLine(result) {
  if (!result?.scores || !Object.keys(result.scores).length) return null;
  const line = el('div', { className: 'scores' });
  for (const [k, v] of Object.entries(result.scores)) {
    line.append(el('span', {}, `${k}: `, el('b', { textContent: typeof v === 'number' ? v.toFixed(3) : String(v) })));
  }
  return line;
}

function artThumbs(result) {
  const files = [...(result?.inputs || []), ...(result?.outputs || [])].filter((a) => a.file && /\.(png|jpe?g|webp)$/i.test(a.file));
  if (!files.length) return null;
  const row = el('div', { className: 'arts' });
  for (const a of files) {
    row.append(el('img', { src: fileUrl(a.file), alt: a.name, title: a.name, loading: 'lazy' }));
  }
  return row;
}

async function overrule(step, candidate, check) {
  const reason = window.prompt(`Why overrule ${step}/${check}?`);
  if (!reason || !reason.trim()) return;
  const path = candidate
    ? `/guided/${enc(buildId)}/candidates/${enc(candidate)}/overrule`
    : `/guided/${enc(buildId)}/overrule`;
  try {
    await json(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ step, check, reason: reason.trim() }),
    });
    status(`Overruled ${check} on ${step}.`);
    await reload();
  } catch (err) {
    status(err.message, true);
  }
}

async function pollJob(job) {
  let current = job;
  for (let i = 0; i < 600; i += 1) {
    if (!current.live && current.state !== 'queued' && current.state !== 'running') return current;
    await new Promise((r) => setTimeout(r, 400));
    current = await json(`/guided/${enc(buildId)}/jobs/${enc(current.id)}`);
    status(`${current.kind || 'job'}: ${current.state}${current.progress != null ? ` (${current.progress})` : ''}`);
  }
  throw new Error('job did not finish in time');
}

async function runJob(path, body) {
  try {
    const started = await json(path, {
      method: 'POST',
      headers: body ? { 'Content-Type': 'application/json' } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
    status(`Started ${started.kind || 'job'}…`);
    const done = await pollJob(started);
    if (done.state === 'done') status(`${done.kind || 'job'} done.`);
    else status(`${done.kind || 'job'}: ${done.state}${done.error ? ` — ${done.error}` : ''}`, done.state !== 'stopped');
    await reload();
    return done;
  } catch (err) {
    status(err.message, true);
    return null;
  }
}

function candidateState(candidate) {
  return record?.candidates?.[candidate] || { steps: {}, pick: [] };
}

function renderCandidateTabs() {
  const bar = $('candidates');
  bar.replaceChildren();
  const names = Object.keys(record?.candidates || {});
  if (!names.length) {
    activeCandidate = 'c1';
    bar.append(el('span', { className: 'hint', textContent: 'No candidates yet — add a source below as c1.' }));
    return;
  }
  if (!names.includes(activeCandidate)) activeCandidate = names[0];
  for (const name of names) {
    const btn = el('button', {
      type: 'button',
      role: 'tab',
      className: name === activeCandidate ? 'on' : '',
      textContent: name,
      'aria-selected': name === activeCandidate ? 'true' : 'false',
    });
    btn.onclick = () => { activeCandidate = name; renderWorkspace(); };
    bar.append(btn);
  }
  const add = el('button', { type: 'button', className: 'small', textContent: '+ candidate' });
  add.onclick = () => {
    const next = window.prompt('Candidate id', `c${names.length + 1}`);
    if (!next || !/^[a-z0-9][a-z0-9_\-]{0,63}$/.test(next)) return;
    activeCandidate = next;
    renderWorkspace();
  };
  bar.append(add);
}

function stepShell(name, statusName, body) {
  const meta = stepMeta(name);
  const section = el('section', { className: 'step', 'aria-labelledby': `step-${name}` });
  const title = el('h2', { id: `step-${name}` }, name, ' ', badge(statusName || 'missing'));
  section.append(title, el('p', { className: 'about', textContent: meta.about || '' }));
  for (const node of body) if (node) section.append(node);
  return section;
}

function sourcePanel(candidate) {
  const state = candidateState(candidate).steps.source || { status: 'missing', blockers: [] };
  const result = resultOf('source', candidate);
  const body = [];
  if (state.blockers?.length) body.push(el('p', { className: 'blockers', textContent: state.blockers.join('; ') }));
  if (result) {
    body.push(checksList(result), artThumbs(result));
    const kind = result.data?.kind;
    if (kind) body.push(el('p', { className: 'hint', textContent: `kind ${kind}${result.data.source ? ` · ${result.data.source}` : ''}` }));
  }

  const gen = el('form', { className: 'stack', 'aria-label': 'Generate source portraits' });
  gen.append(el('strong', { textContent: 'Generate' }));
  const fields = el('div', { className: 'fields' });
  const age = el('input', { id: 'src-age', type: 'number', min: 18, max: 120, value: 28, required: true });
  const style = el('select', { id: 'src-style' });
  for (const s of (catalog?.styles || ['character'])) style.append(new Option(s, s));
  const seeds = el('input', { id: 'src-seeds', value: '1 2 3 4', required: true });
  const hair = el('input', { id: 'src-hair', maxlength: 200 });
  const clothes = el('input', { id: 'src-clothes', maxlength: 200 });
  const eyes = el('input', { id: 'src-eyes', maxlength: 200 });
  const description = el('input', { id: 'src-desc', maxlength: 500 });
  fields.append(
    el('label', {}, 'Age (18+)', age),
    el('label', {}, 'Style', style),
    el('label', {}, 'Seeds', seeds),
    el('label', {}, 'Hair', hair),
    el('label', {}, 'Clothes', clothes),
    el('label', {}, 'Eyes', eyes),
  );
  gen.append(fields, el('label', {}, 'Description', description));
  const genBtn = el('button', { type: 'submit', className: 'primary', textContent: 'Generate batch' });
  gen.append(genBtn);
  gen.onsubmit = async (ev) => {
    ev.preventDefault();
    await runJob(`/guided/${enc(buildId)}/source/generate`, {
      age: Number(age.value), style: style.value, seeds: seeds.value.trim(),
      hair: hair.value, clothes: clothes.value, eyes: eyes.value, description: description.value,
    });
  };
  body.push(gen);

  const dark = el('details', {});
  dark.append(el('summary', { textContent: 'Pick a darkroom record' }));
  const recordsBox = el('div', { className: 'records' });
  const loadRec = el('button', { type: 'button', className: 'small', textContent: 'Load passed records' });
  loadRec.onclick = async () => {
    try {
      const listed = await json('/guided/darkroom/records?page_size=50');
      recordsBox.replaceChildren();
      const items = listed.records || listed.items || listed.results || [];
      if (!items.length) recordsBox.append(el('p', { className: 'hint', textContent: 'No passed records.' }));
      for (const rec of items) {
        const id = rec.id || rec.record_id;
        const btn = el('button', { type: 'button', textContent: `${id}${rec.style ? ` · ${rec.style}` : ''}` });
        btn.onclick = async () => {
          try {
            await json(`/guided/${enc(buildId)}/candidates/${enc(candidate)}/source/pick`, {
              method: 'POST', headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ record_id: id }),
            });
            status(`Picked darkroom record ${id}.`);
            await reload();
          } catch (err) { status(err.message, true); }
        };
        recordsBox.append(btn);
      }
    } catch (err) { status(err.message, true); }
  };
  dark.append(loadRec, recordsBox);
  body.push(dark);

  const up = el('form', { className: 'stack', 'aria-label': 'Upload a portrait' });
  up.append(el('strong', { textContent: 'Upload' }));
  const file = el('input', { type: 'file', accept: 'image/png,image/jpeg,image/webp', required: true });
  const kind = el('select');
  for (const k of ['personal', 'synthetic', 'consented']) kind.append(new Option(k, k));
  const sourceText = el('input', { maxlength: 500, placeholder: 'where it came from' });
  const license = el('input', { maxlength: 200 });
  up.append(
    el('label', {}, 'Portrait', file),
    el('label', {}, 'Kind', kind),
    el('label', {}, 'Source', sourceText),
    el('label', {}, 'Licence', license),
  );
  up.append(el('button', { type: 'submit', textContent: 'Upload' }));
  up.onsubmit = async (ev) => {
    ev.preventDefault();
    if (!file.files?.[0]) return;
    const data = new FormData();
    data.append('file', file.files[0]);
    data.append('kind', kind.value);
    data.append('source', sourceText.value);
    data.append('license', license.value);
    try {
      const reply = await fetch(`/guided/${enc(buildId)}/candidates/${enc(candidate)}/source/upload`, { method: 'POST', body: data });
      const bodyJson = await reply.json().catch(() => ({}));
      if (!reply.ok) throw new Error(String(bodyJson.detail || reply.statusText));
      status('Portrait uploaded.');
      await reload();
    } catch (err) { status(err.message, true); }
  };
  body.push(up);

  const restyle = el('form', { className: 'row', 'aria-label': 'Restyle source' });
  const rStyle = el('select');
  for (const s of (catalog?.styles || ['character'])) rStyle.append(new Option(s, s));
  restyle.append(el('label', {}, 'Restyle to', rStyle), el('button', { type: 'submit', textContent: 'Restyle' }));
  restyle.onsubmit = async (ev) => {
    ev.preventDefault();
    await runJob(`/guided/${enc(buildId)}/candidates/${enc(candidate)}/source/restyle`, { style: rStyle.value });
  };
  body.push(restyle);

  return stepShell('source', state.status, body);
}

function portraitPanel(candidate) {
  const state = candidateState(candidate).steps.portrait || { status: 'missing', blockers: [] };
  const result = resultOf('portrait', candidate);
  const body = [];
  if (state.blockers?.length) body.push(el('p', { className: 'blockers', textContent: state.blockers.join('; ') }));
  if (result) body.push(checksList(result), artThumbs(result), scoresLine(result));
  const actions = el('div', { className: 'actions' });
  const run = el('button', { type: 'button', className: 'primary', textContent: 'Run portrait checks' });
  run.disabled = state.blockers?.some((b) => b.includes('source')) && state.status === 'missing';
  run.onclick = () => runJob(`/guided/${enc(buildId)}/portrait`, { candidates: [candidate] });
  actions.append(run);
  if (catalog?.portrait && catalog.portrait.ready === false) {
    actions.append(el('span', { className: 'hint', textContent: catalog.portrait.reason || 'portrait checks not ready' }));
  }
  body.push(actions);
  return stepShell('portrait', state.status, body);
}

function buildPanel(candidate) {
  const state = candidateState(candidate).steps.build || { status: 'missing', blockers: [] };
  const result = resultOf('build', candidate);
  const body = [];
  if (state.blockers?.length) body.push(el('p', { className: 'blockers', textContent: state.blockers.join('; ') }));
  if (result) body.push(checksList(result), artThumbs(result), scoresLine(result));
  const actions = el('div', { className: 'actions' });
  const run = el('button', { type: 'button', className: 'primary', textContent: 'Build LAM head' });
  run.disabled = !!state.blockers?.length && state.status === 'missing';
  run.onclick = () => runJob(`/guided/${enc(buildId)}/candidates/${enc(candidate)}/build`, { fit_shape: false });
  const fit = el('button', { type: 'button', textContent: 'Build + fit shape' });
  fit.disabled = run.disabled;
  fit.onclick = () => runJob(`/guided/${enc(buildId)}/candidates/${enc(candidate)}/build`, { fit_shape: true });
  actions.append(run, fit);
  if (catalog?.lam && catalog.lam.ready === false) {
    actions.append(el('span', { className: 'hint', textContent: catalog.lam.reason || 'LAM not ready' }));
  }
  body.push(actions);
  return stepShell('build', state.status, body);
}

function evalPanel(step, candidate) {
  const state = candidateState(candidate).steps[step] || { status: 'missing', blockers: [] };
  const result = resultOf(step, candidate);
  const body = [];
  if (state.blockers?.length) body.push(el('p', { className: 'blockers', textContent: state.blockers.join('; ') }));
  if (result) {
    if (result.status === 'stale') body.push(el('p', { className: 'blockers', textContent: 'This result is stale — run it again.' }));
    body.push(checksList(result), artThumbs(result), scoresLine(result));
    if (step === 'turns' && result.data?.limits) {
      const L = result.data.limits;
      body.push(el('p', { className: 'hint', textContent: `limits yaw ${L.yaw}° / up ${L.pitch_up}° / down ${L.pitch_down}°` }));
    }
  }
  const actions = el('div', { className: 'actions' });
  const label = { likeness: 'Score likeness', rest: 'Fit rest set', turns: 'Measure turns', speech: 'Speech eval', age: 'Read age' }[step];
  const run = el('button', { type: 'button', className: 'primary', textContent: label });
  run.disabled = !!state.blockers?.length && state.status === 'missing';
  run.onclick = () => runJob(`/guided/${enc(buildId)}/candidates/${enc(candidate)}/${step}`);
  actions.append(run);
  const svc = catalog?.[step];
  if (svc && svc.ready === false) actions.append(el('span', { className: 'hint', textContent: svc.reason || `${step} not ready` }));
  body.push(actions);
  return stepShell(step, state.status, body);
}

function pickPanel() {
  const face = record?.face?.pick || { status: 'missing', candidate: null };
  const body = [];
  body.push(el('p', { className: 'hint', textContent: 'Compare settled candidates. Preview uses the screen camera (look / motion / splat).' }));
  if (face.candidate) body.push(el('p', { className: 'hint', textContent: `Picked: ${face.candidate}` }));

  const grid = el('div', { id: 'pick-grid' });
  const names = Object.keys(record?.candidates || {});
  for (const name of names) {
    const blockers = record.candidates[name].pick || [];
    const card = el('article', {
      className: `pick-card${blockers.length ? '' : ' ready'}${face.candidate === name ? ' picked' : ''}`,
      'aria-label': `Candidate ${name}`,
    });
    const top = el('div', { className: 'top' }, el('strong', { textContent: name }));
    if (!blockers.length) top.append(el('span', { className: 'badge good', textContent: 'ready' }));
    else top.append(el('span', { className: 'badge warn', textContent: 'blocked', title: blockers.join('; ') }));
    card.append(top);

    const portrait = resultOf('source', name)?.outputs?.find((a) => a.file);
    const render = resultOf('build', name)?.outputs?.find((a) => a.name === 'render' && a.file);
    const pics = el('div', { className: 'portrait-row' });
    if (portrait) pics.append(el('img', { src: fileUrl(portrait.file), alt: `${name} portrait` }));
    if (render) pics.append(el('img', { src: fileUrl(render.file), alt: `${name} head render` }));
    card.append(pics);

    const scoreBits = el('div', { className: 'scores' });
    for (const step of EVAL_STEPS) {
      const r = resultOf(step, name);
      const st = candidateState(name).steps[step]?.status || 'missing';
      scoreBits.append(el('span', {}, `${step}: `, el('b', { textContent: st })));
      if (r?.scores) {
        for (const [k, v] of Object.entries(r.scores).slice(0, 2)) {
          scoreBits.append(el('span', {}, `${k}=`, el('b', { textContent: typeof v === 'number' ? v.toFixed(2) : String(v) })));
        }
      }
    }
    card.append(scoreBits);
    if (blockers.length) card.append(el('p', { className: 'blockers', textContent: blockers.join('; ') }));

    const stageBox = el('div', { className: 'stage' });
    const poses = el('div', { className: 'poses' });
    card.append(stageBox, poses);

    const pickBtn = el('button', { type: 'button', className: 'primary', textContent: face.candidate === name ? 'Picked' : 'Pick this head' });
    pickBtn.disabled = blockers.length > 0 || face.candidate === name;
    pickBtn.onclick = async () => {
      try {
        await json(`/guided/${enc(buildId)}/pick`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ candidate: name }),
        });
        status(`Picked ${name}.`);
        await reload();
      } catch (err) { status(err.message, true); }
    };
    card.append(pickBtn);
    grid.append(card);

    // Preview after the card is in the DOM so the stage has a size.
    void previewCandidate(name, stageBox, poses, !blockers.length);
  }
  if (!names.length) grid.append(el('p', { className: 'hint', textContent: 'Add a candidate and settle its evals first.' }));
  body.push(grid);
  return stepShell('pick', face.status, body);
}

function scenePanel() {
  const face = record?.face?.scene || { status: 'missing', blockers: [] };
  const result = resultOf('scene', null);
  const body = [];
  body.push(el('p', {
    className: 'hint',
    textContent: 'Backdrop and shot as Saved faces uses them. A pale floor under the bust warns (garden’s path).',
  }));
  if (face.blockers?.length) body.push(el('p', { className: 'blockers', textContent: face.blockers.join('; ') }));
  if (result) {
    body.push(checksList(result), scoresLine(result));
    const chosen = result.data || {};
    body.push(el('p', {
      className: 'hint',
      textContent: `Recorded: ${chosen.scene || 'plain background'}${chosen.shot ? `, shot ${chosen.shot}` : ''}`,
    }));
  }

  const form = el('form', { className: 'stack', 'aria-label': 'Background and shot' });
  const sceneSel = el('select', { id: 'scene-name' });
  sceneSel.append(new Option('None: a plain background', ''));
  const shotSel = el('select', { id: 'scene-shot' });
  shotSel.append(
    new Option('Shot: as the background suits', ''),
    new Option('Shot: close, head and shoulders', 'close'),
    new Option('Shot: wide, a floating head', 'wide'),
  );
  if (result?.data) {
    sceneSel.value = result.data.scene || '';
    shotSel.value = result.data.shot || '';
  }
  form.append(
    el('label', {}, 'Backdrop', sceneSel),
    el('label', {}, 'Shot', shotSel),
  );
  const previewHost = el('div', { className: 'stage', style: 'max-width:320px;min-height:280px;aspect-ratio:5/6;background:#0a0c10;border-radius:8px;overflow:hidden' });
  const poses = el('div', { className: 'poses' });
  form.append(previewHost, poses);

  const save = el('button', { type: 'submit', className: 'primary', textContent: 'Keep background' });
  save.disabled = !!face.blockers?.length && face.status === 'missing';
  form.append(save);
  form.onsubmit = async (ev) => {
    ev.preventDefault();
    try {
      await json(`/guided/${enc(buildId)}/scene`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ scene: sceneSel.value || null, shot: shotSel.value || null }),
      });
      status(`Scene set to ${sceneSel.value || 'plain'}${shotSel.value ? ` / ${shotSel.value}` : ''}.`);
      await reload();
    } catch (err) { status(err.message, true); }
  };
  body.push(form);

  // Load scene list and preview the picked head over the chosen backdrop.
  void (async () => {
    try {
      const scenes = await json('/scenes');
      const keep = sceneSel.value;
      while (sceneSel.options.length > 1) sceneSel.remove(1);
      for (const s of scenes) {
        sceneSel.append(new Option(
          `${s.title} (${s.name}${s.built_in ? '' : ', yours'})${s.cleared ? '' : ': own screens only'}`,
          s.name,
        ));
      }
      sceneSel.value = keep;
    } catch (err) { status(err.message, true); }
    const picked = record?.picked;
    if (!picked) {
      previewHost.textContent = 'Pick a candidate first.';
      return;
    }
    const build = resultOf('build', picked);
    if (!build?.outputs?.some((a) => a.name === 'head')) {
      previewHost.textContent = 'No head for the picked candidate.';
      return;
    }
    let stage = stages.get(`scene:${picked}`);
    if (!stage) {
      stage = new Stage();
      stages.set(`scene:${picked}`, stage);
    }
    stage.attach(previewHost);
    const turns = resultOf('turns', picked);
    const rest = resultOf('rest', picked);
    if (rest?.data?.weights) stage.setRest({ weights: rest.data.weights, life: rest.data.life ?? 1 });
    try {
      await talkReady;
      await stage.load(
        `/guided/${enc(buildId)}/candidates/${enc(picked)}/head.zip`,
        {},
        'lam-splat',
        { limits: turns?.data?.limits || null },
      );
      stage.pose = 'speak';
      stage.poseButtons(poses);
      const apply = () => {
        stage.setScene(sceneSel.value || '').catch((err) => status(err.message, true));
        stage.setShot(shotSel.value || '');
      };
      sceneSel.onchange = apply;
      shotSel.onchange = apply;
      apply();
    } catch (err) {
      previewHost.textContent = `Preview failed: ${err.message}`;
    }
  })();

  return stepShell('scene', face.status, body);
}

async function previewCandidate(candidate, host, posesBar, ready) {
  if (!ready) {
    host.textContent = 'Settle likeness, rest, turns, speech and age to preview.';
    return;
  }
  const build = resultOf('build', candidate);
  if (!build?.outputs?.some((a) => a.name === 'head')) {
    host.textContent = 'No head yet.';
    return;
  }
  let stage = stages.get(candidate);
  if (!stage) {
    stage = new Stage();
    stages.set(candidate, stage);
  }
  stage.attach(host);
  const turns = resultOf('turns', candidate);
  const rest = resultOf('rest', candidate);
  const limits = turns?.data?.limits || null;
  if (rest?.data?.weights) stage.setRest({ weights: rest.data.weights, life: rest.data.life ?? 1 });
  try {
    await talkReady;
    await stage.load(
      `/guided/${enc(buildId)}/candidates/${enc(candidate)}/head.zip`,
      {},
      'lam-splat',
      { limits },
    );
    stage.pose = 'speak';
    stage.poseButtons(posesBar);
  } catch (err) {
    host.textContent = `Preview failed: ${err.message}`;
  }
}

function renderWorkspace() {
  if (!record) {
    $('workspace').hidden = true;
    return;
  }
  $('workspace').hidden = false;
  $('build-title').textContent = `${record.name || record.id} (${record.id})`;
  renderCandidateTabs();
  const steps = $('steps');
  steps.replaceChildren();
  const c = activeCandidate || 'c1';
  steps.append(
    sourcePanel(c),
    portraitPanel(c),
    buildPanel(c),
    ...EVAL_STEPS.map((s) => evalPanel(s, c)),
    pickPanel(),
    scenePanel(),
  );
}

function renderBuildList() {
  const list = $('build-list');
  list.replaceChildren();
  const builds = catalog?.builds || [];
  if (!builds.length) list.append(el('p', { className: 'hint', textContent: 'No builds yet.' }));
  for (const b of builds) {
    const chip = el('button', {
      type: 'button',
      className: `build-chip${b.id === buildId ? ' on' : ''}`,
      role: 'listitem',
    });
    chip.append(
      el('span', { className: 'id', textContent: b.id }),
      el('span', { textContent: b.name || '' }),
      el('span', { className: 'badge', textContent: b.picked ? `picked ${b.picked}` : `${(b.candidates || []).length} cand.` }),
    );
    chip.onclick = () => openBuild(b.id);
    list.append(chip);
  }
  if (catalog?.refused && Object.keys(catalog.refused).length) {
    list.append(el('p', { className: 'err', textContent: `Unreadable: ${Object.entries(catalog.refused).map(([k, v]) => `${k}: ${v}`).join('; ')}` }));
  }
}

async function openBuild(id) {
  buildId = id;
  history.replaceState(null, '', `?id=${enc(id)}`);
  try {
    record = await json(`/guided/${enc(id)}`);
    status(`Loaded ${id}.`);
    renderBuildList();
    renderWorkspace();
  } catch (err) {
    record = null;
    status(err.message, true);
    renderWorkspace();
  }
}

async function reload() {
  catalog = await json('/guided');
  renderBuildList();
  if (buildId) {
    record = await json(`/guided/${enc(buildId)}`);
    renderWorkspace();
  }
}

$('create').onsubmit = async (ev) => {
  ev.preventDefault();
  const id = $('new-id').value.trim();
  const name = $('new-name').value.trim();
  try {
    await json('/guided', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id, name }),
    });
    $('new-id').value = '';
    $('new-name').value = '';
    status(`Created ${id}.`);
    await reload();
    await openBuild(id);
  } catch (err) { status(err.message, true); }
};

$('refresh').onclick = () => reload().catch((err) => status(err.message, true));

(async () => {
  try {
    catalog = await json('/guided');
    renderBuildList();
    const want = new URLSearchParams(location.search).get('id');
    if (want) await openBuild(want);
    else status(`${(catalog.builds || []).length} build(s).`);
  } catch (err) {
    status(err.message, true);
  }
})();
