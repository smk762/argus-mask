// Saved faces: every face in the assets root and every preview, to review, remix, rename or delete.
// Rename and delete touch only faces the editor made; the server says so if not.
// Everything the server sends is put on the page as text, never as HTML.

import { Stage } from './stage.js';

const $ = (id) => document.getElementById(id);
const el = (tag, props = {}, ...children) => {
  const node = Object.assign(document.createElement(tag), props);
  node.append(...children);
  return node;
};

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

const enc = encodeURIComponent;
const when = (t) => new Date(t * 1000).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' });
const mb = (b) => `${(b / 1048576).toFixed(1)} MB`;
// The copy a card stands for: the export (what screens get) if there is one, else the preview.
const main = (row) => (row.export ? 'export' : 'preview');

// ---- thumbnails: one hidden stage takes a still of each face in turn -----------------------

const shots = new Stage({ animate: false });
const shotCache = new Map();  // `${place}/${id}/${sha}` -> data URL
let queue = Promise.resolve();
function thumbnail(row, img, wait) {
  const place = main(row);
  queue = queue.then(async () => {
    try {
      const manifest = await json(`/face/${enc(row.id)}?from=${place}`);
      const key = `${place}/${row.id}/${manifest.model.sha256}`;
      if (!shotCache.has(key)) {
        await shots.load(`/face/${enc(row.id)}/model?from=${place}&v=${manifest.model.sha256}`, manifest.framing || {});
        shotCache.set(key, shots.snapshot());
      }
      img.src = shotCache.get(key);
      wait.remove();
    } catch (err) {
      wait.textContent = 'no preview';
      wait.title = err.message;
    }
  });
}

// ---- the grid ----------------------------------------------------------------------------------

let rows = [];
function render() {
  const filter = $('filter').value.trim().toLowerCase();
  const grid = $('grid');
  grid.replaceChildren();
  const shown = rows.filter((r) => {
    const name = (r.export || r.preview || {}).name || '';
    return !filter || r.id.includes(filter) || name.toLowerCase().includes(filter);
  });
  $('empty').hidden = rows.length > 0;
  for (const row of shown) grid.append(card(row));
}

function card(row) {
  const copy = row[main(row)];
  const img = el('img', { alt: '' });
  const wait = el('div', { className: 'wait', textContent: row.refused ? 'not served' : 'rendering…' });
  const shot = el('div', { className: 'shot', title: 'Review' }, img, wait);
  const badges = el('div', { className: 'badges' });
  if (row.export) badges.append(el('span', { className: 'badge live', textContent: 'on /faces', title: `assets/${row.id}` }));
  // A preview the same as the export (just exported) is not worth a word; a different one is.
  if (row.preview && !row.export) badges.append(el('span', { className: 'badge preview', textContent: 'preview only', title: `assets/.preview/${row.id}` }));
  if (row.preview && row.export && row.preview.sha256 !== row.export.sha256) {
    badges.append(el('span', { className: 'badge preview', textContent: 'unexported changes', title: `assets/.preview/${row.id} differs from the export` }));
  }
  if (copy) badges.append(el('span', { className: 'badge', textContent: copy.kind }), el('span', { className: 'badge', textContent: copy.license }));
  const info = el('div', { className: 'info' },
    el('div', { className: 'name', textContent: copy ? copy.name : row.id }),
    el('div', { className: 'id', textContent: row.id }),
    badges,
  );
  if (copy) info.append(el('div', { className: 'meta', textContent: `${when(copy.updated)} · ${mb(copy.bytes)}` }));
  if (row.refused) info.append(el('div', { className: 'refused', textContent: `Not served: ${row.refused}` }));

  const editable = [row.export, row.preview].some((c) => c && c.editable);
  const notMine = 'Only faces the editor made can be changed here.';
  const review = el('button', { type: 'button', textContent: 'Review', disabled: !copy });
  const remix = el('button', { type: 'button', textContent: 'Remix', disabled: !editable, title: editable ? 'Open its preset in the editor, under a new id' : 'No preset: it was not made by the editor.' });
  // Rename moves every copy together, so it needs them all to be the editor's; delete can take one alone.
  const renamable = editable && [row.export, row.preview].every((c) => !c || c.editable);
  const rename = el('button', { type: 'button', textContent: 'Rename', disabled: !renamable, title: renamable ? '' : notMine });
  const del = el('button', { type: 'button', className: 'danger', textContent: 'Delete', disabled: !editable, title: editable ? 'Move to assets/.trash/' : notMine });
  review.onclick = () => openReview(row);
  shot.onclick = () => copy && openReview(row);
  remix.onclick = () => { location.href = `./?remix=${enc(row.id)}&from=${row.export?.editable ? 'export' : 'preview'}`; };
  rename.onclick = () => openRename(row);
  del.onclick = () => remove(row);

  if (copy) thumbnail(row, img, wait);
  return el('div', { className: 'card' }, shot, info, el('div', { className: 'actions' }, review, remix, rename, del));
}

async function load() {
  rows = await json('/saved');
  render();
  status(`${rows.length} face${rows.length === 1 ? '' : 's'}.`);
}

// ---- review --------------------------------------------------------------------------------

const stage = new Stage();
async function openReview(row, place = main(row)) {
  const manifest = await json(`/face/${enc(row.id)}?from=${place}`).catch((err) => { status(err.message, true); return null; });
  if (!manifest) return;
  $('modal').hidden = false;
  stage.attach($('stage'));
  await stage.load(`/face/${enc(row.id)}/model?from=${place}&v=${manifest.model.sha256}`, manifest.framing || {});
  stage.poseButtons($('m-poses'));
  $('m-name').textContent = `${manifest.name} (${manifest.id})`;
  const which = $('m-which');
  which.replaceChildren();
  if (row.export && row.preview) {
    for (const p of ['export', 'preview']) {
      const b = el('button', { type: 'button', textContent: p === 'export' ? 'exported' : 'preview', className: p === place ? 'on' : '' });
      b.onclick = () => openReview(row, p);
      which.append(b);
    }
  }
  const prov = manifest.provenance;
  $('m-side').replaceChildren(
    el('h2', { textContent: 'Where' }), el('p', { textContent: place === 'export' ? `assets/${row.id}: on the server's /faces` : `assets/.preview/${row.id}: the editor's preview` }),
    el('h2', { textContent: 'Licence' }), el('p', { textContent: prov.license }),
    el('h2', { textContent: 'Kind' }), el('p', { textContent: prov.kind }),
    el('h2', { textContent: 'Source' }), el('p', { textContent: prov.source }),
    el('h2', { textContent: 'Credits and notes' }), el('p', { textContent: prov.notes || '—' }),
    el('h2', { textContent: 'Model' }), el('p', { textContent: `${manifest.model.file}, ${manifest.blendshapes.length} of 52 shapes\nsha256 ${manifest.model.sha256}` }),
  );
  $('m-close').focus();
}
function closeReview() { $('modal').hidden = true; }
$('m-close').onclick = closeReview;
window.addEventListener('keydown', (e) => { if (e.key === 'Escape' && !$('modal').hidden) closeReview(); });

// ---- rename and delete -----------------------------------------------------------------------

let renaming = null;
function openRename(row) {
  renaming = row;
  const copy = row[main(row)];
  $('r-name').value = copy.name;
  $('r-id').value = row.id;
  $('r-err').textContent = '';
  $('rename').showModal();
  $('r-name').select();
}
$('r-cancel').onclick = () => $('rename').close();
$('rename-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const row = renaming;
  const copy = row[main(row)];
  const change = {};
  const name = $('r-name').value.trim();
  const id = $('r-id').value.trim();
  if (name && name !== copy.name) change.name = name;
  if (id && id !== row.id) change.id = id;
  if (!Object.keys(change).length) { $('rename').close(); return; }
  try {
    const done = await json(`/saved/${enc(row.id)}/rename`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(change) });
    $('rename').close();
    await load();
    status(change.id ? `Renamed ${row.id} to ${done.id}. Screens that asked for ${row.id} need the new id.` : `Renamed ${row.id}.`);
  } catch (err) {
    $('r-err').textContent = err.message;
  }
});

async function remove(row) {
  const what = [row.export?.editable && `assets/${row.id}`, row.preview?.editable && `the preview`].filter(Boolean).join(' and ');
  if (!window.confirm(`Delete ${row.id}? This moves ${what} to assets/.trash/; move it back to restore it.`)) return;
  try {
    // Only the copies the editor made: a preview over someone else's export goes alone.
    const only = row.export?.editable && row.preview?.editable ? '' : row.export?.editable ? '?from=export' : '?from=preview';
    const done = await json(`/saved/${enc(row.id)}${only}`, { method: 'DELETE' });
    await load();
    status(`Deleted ${row.id}: it is in ${done.trash}.`);
  } catch (err) {
    status(err.message, true);
  }
}

$('filter').addEventListener('input', render);
load().catch((err) => status(err.message, true));
