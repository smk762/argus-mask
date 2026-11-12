// The Asset packs page: browse the community packs in the editor's catalogue and install one into MPFB's data.
// Everything the server sends is put on the page as text, never as HTML.

const $ = (id) => document.getElementById(id);
const el = (tag, props = {}, ...children) => {
  const node = Object.assign(document.createElement(tag), props);
  node.append(...children.filter((c) => c !== null && c !== undefined && c !== false));
  return node;
};
const store = {
  get(key, fallback) { try { return JSON.parse(localStorage.getItem(`mask-packs.${key}`)) ?? fallback; } catch { return fallback; } },
  set(key, value) { try { localStorage.setItem(`mask-packs.${key}`, JSON.stringify(value)); } catch { /* private window */ } },
};
async function json(url, options) {
  const reply = await fetch(url, options);
  const body = await reply.json().catch(() => ({}));
  if (!reply.ok) throw new Error(typeof body.detail === 'string' ? body.detail : (body.detail ? JSON.stringify(body.detail) : reply.statusText));
  return body;
}

// The MPFB types the editor offers (library.py: PART_DIRS and clothes). Anything else installs, but no picker shows it.
const USED = new Set(['skins', 'eyes', 'eyebrows', 'eyelashes', 'teeth', 'tongue', 'hair', 'clothes']);
const TYPE_ORDER = ['hair', 'clothes', 'eyebrows', 'eyelashes', 'eyes', 'skins', 'teeth', 'tongue'];
const SHOWN = 12; // thumbnails per pack before "show all"

let data = { packs: [], jobs: [], ready: false, data_dir: null };
const jobs = new Map(); // pack id -> job
const open = new Set(); // packs showing every asset
let type = store.get('type', '');

const mb = (bytes) => (bytes ? `${bytes >= 2 ** 30 ? (bytes / 2 ** 30).toFixed(1) + ' GB' : Math.max(1, Math.round(bytes / 2 ** 20)) + ' MB'}` : '');
const licenceClass = (l) => (l === 'CC0' ? 'good' : l ? 'warn' : 'bad');

function matches(pack, asset, words) {
  if (type && asset.type !== type) return false;
  if ($('licence').value && asset.license !== $('licence').value) return false;
  if (!words.length) return true;
  const hay = `${pack.id} ${pack.description} ${asset.name} ${asset.author} ${asset.type}`.toLowerCase();
  return words.every((w) => hay.includes(w));
}

function packMatches(pack, words) {
  if ($('installed-only').checked && !pack.installed) return [];
  const assets = pack.assets.filter((a) => matches(pack, a, words));
  if (assets.length) return assets;
  // A pack without an asset table (faceunits01) still shows when nothing narrows by asset.
  const hay = `${pack.id} ${pack.description}`.toLowerCase();
  if (!pack.assets.length && !type && !$('licence').value && words.every((w) => hay.includes(w))) return [];
  return null;
}

function typeTabs() {
  const counts = {};
  for (const p of data.packs) for (const a of p.assets) counts[a.type] = (counts[a.type] || 0) + 1;
  const types = Object.keys(counts).sort((a, b) => {
    const ia = TYPE_ORDER.indexOf(a), ib = TYPE_ORDER.indexOf(b);
    return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib) || a.localeCompare(b);
  });
  if (type && !types.includes(type)) type = '';
  const tab = (value, label) => {
    const b = el('button', { type: 'button', className: `tab small${type === value ? ' on' : ''}`, textContent: label });
    b.setAttribute('aria-pressed', String(type === value));
    b.onclick = () => { type = value; store.set('type', type); typeTabs(); render(); };
    return b;
  };
  $('types').replaceChildren(tab('', 'All'), ...types.map((t) => tab(t, `${t} ${counts[t]}`)));
}

function assetCard(a) {
  const img = a.thumb
    ? el('img', { src: a.thumb, alt: '', loading: 'lazy', decoding: 'async', referrerPolicy: 'no-referrer' })
    : el('div', { className: 'noimg', textContent: 'no picture' });
  if (a.thumb) img.onerror = () => img.replaceWith(el('div', { className: 'noimg', textContent: 'no picture' }));
  const used = USED.has(a.type);
  return el('div', { className: `asset${used ? '' : ' unused'}`, title: `${a.name}\n${a.type} · ${a.author || 'author not stated'} · ${a.license}${used ? '' : '\nnot offered by the editor'}` },
    img,
    el('div', { className: 'an', textContent: a.name }),
    el('div', { className: 'meta', textContent: `${a.type} · ${a.author || '?'}` }));
}

function action(pack) {
  const job = jobs.get(pack.id);
  if (job && (job.state === 'downloading' || job.state === 'unpacking')) {
    const bar = el('progress', { max: job.total_bytes || 1, value: job.done_bytes || 0 });
    if (!job.total_bytes) bar.removeAttribute('value');
    return el('span', { className: 'act' },
      el('span', { className: 'hint', textContent: job.state === 'unpacking' ? 'Unpacking…' : `Downloading ${mb(job.done_bytes)} of ${mb(job.total_bytes)}` }),
      job.state === 'unpacking' ? el('span', { className: 'spin' }) : bar);
  }
  if (pack.installed) {
    const done = job && job.state === 'done';
    return el('span', { className: 'act' },
      done ? el('a', { href: './', textContent: 'Now in the editor →' }) : null,
      el('span', { className: 'badge good', textContent: 'installed' }));
  }
  const busy = [...jobs.values()].some((j) => j.state === 'downloading' || j.state === 'unpacking');
  const b = el('button', { type: 'button', className: 'primary small', textContent: `Install${pack.bytes ? ` (${mb(pack.bytes)})` : ''}` });
  b.disabled = !pack.installable || busy || !data.ready;
  b.title = !data.ready ? 'MPFB\'s data directory is not set' : pack.reason || (busy ? 'one install at a time' : `download, check the hash and unpack ${pack.id}`);
  b.onclick = () => install(pack.id);
  return el('span', { className: 'act' },
    job && job.state === 'failed' ? el('span', { className: 'err', textContent: job.error }) : null,
    pack.reason ? el('span', { className: 'hint', textContent: pack.reason }) : null,
    b);
}

// Thumbnails come from the community's site: shown once a type, a search or a pack narrows the list, so the
// whole catalogue (some 800 pictures) is never asked for at once.
function packCard(pack, assets, narrowed) {
  const all = open.has(pack.id);
  const pictures = narrowed || all;
  const shown = !pictures ? [] : all ? assets : assets.slice(0, SHOWN);
  const types = [...new Set(pack.assets.map((a) => a.type))];
  const usable = pack.assets.filter((a) => USED.has(a.type)).length;
  const card = el('section', { className: 'pack', id: `pack-${pack.id}` },
    el('div', { className: 'top' },
      el('a', { className: 'name', href: pack.page, target: '_blank', rel: 'noopener', textContent: pack.id }),
      el('span', { className: `badge ${licenceClass(pack.license)}`, textContent: pack.license || 'no licence' }),
      el('span', { className: 'badge', textContent: `${pack.assets.length} asset${pack.assets.length === 1 ? '' : 's'}${types.length ? ': ' + types.join(', ') : ''}` }),
      pack.assets.length && !usable ? el('span', { className: 'badge', textContent: 'nothing the editor offers' }) : null,
      action(pack)),
    pack.description ? el('div', { className: 'desc', textContent: pack.description }) : null);
  if (shown.length) card.append(el('div', { className: 'thumbs' }, ...shown.map(assetCard)));
  if (assets.length > (pictures ? SHOWN : 0)) {
    const label = all ? (narrowed ? 'Show fewer' : 'Hide assets') : narrowed ? `Show all ${assets.length}` : `Show ${assets.length} asset${assets.length === 1 ? '' : 's'}`;
    const more = el('button', { type: 'button', className: 'small', textContent: label });
    more.onclick = () => { all ? open.delete(pack.id) : open.add(pack.id); render(); };
    card.append(el('div', {}, more));
  }
  return card;
}

function render() {
  const words = $('filter').value.toLowerCase().split(/\s+/).filter(Boolean);
  const cards = [];
  let assetCount = 0;
  for (const pack of data.packs) {
    const assets = packMatches(pack, words);
    if (assets === null) continue;
    assetCount += assets.length;
    cards.push(packCard(pack, assets, Boolean(type || words.length || $('licence').value)));
  }
  $('packs').replaceChildren(...(cards.length ? cards : [el('div', { className: 'hint', textContent: data.packs.length ? 'No pack matches.' : 'The catalogue is empty.' })]));
  const installed = data.packs.filter((p) => p.installed).length;
  $('status').textContent = `${cards.length} of ${data.packs.length} packs, ${assetCount} assets shown · ${installed} installed`
    + (data.ready ? '' : ' · MPFB data directory not set');
}

async function load() {
  try {
    data = await json('./packs');
    for (const j of data.jobs) jobs.set(j.pack, j);
    $('error').hidden = true;
  } catch (err) {
    $('error').textContent = `Could not load the catalogue: ${err.message}`;
    $('error').hidden = false;
  }
  typeTabs();
  render();
  for (const j of data.jobs) follow(j);
}

async function install(id) {
  try {
    const job = await json(`./packs/${encodeURIComponent(id)}/install`, { method: 'POST' });
    jobs.set(id, job);
    render();
    follow(job);
  } catch (err) {
    jobs.set(id, { pack: id, state: 'failed', error: err.message });
    render();
  }
}

async function follow(job) {
  while (job.state === 'downloading' || job.state === 'unpacking') {
    await new Promise((r) => setTimeout(r, 700));
    try {
      job = await json(`./packs/jobs/${encodeURIComponent(job.id)}`);
    } catch (err) {
      job = { ...job, state: 'failed', error: err.message };
    }
    jobs.set(job.pack, job);
    render();
  }
  if (job.state === 'done') {
    const pack = data.packs.find((p) => p.id === job.pack);
    if (pack) { pack.installed = true; pack.installable = false; pack.reason = null; }
    render();
  }
}

$('filter').addEventListener('input', render);
$('licence').addEventListener('change', render);
$('installed-only').addEventListener('change', render);
load();
