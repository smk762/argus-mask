// "Open on screen": from an exported face to a viewer on a screen, ready to speak.
// Shared by the editor (index.html) and Saved faces (faces.html). The editor checks the server for
// the face and its screens (GET /screen/<id>), and hands out a link at the viewer's origin, never its own.
// Everything the server sends is put on the page as text, never as HTML.

const el = (tag, props = {}, ...children) => {
  const node = Object.assign(document.createElement(tag), props);
  node.append(...children);
  return node;
};
const store = {
  get(key, fallback) { try { return JSON.parse(localStorage.getItem(`mask-editor.${key}`)) ?? fallback; } catch { return fallback; } },
  set(key, value) { try { localStorage.setItem(`mask-editor.${key}`, JSON.stringify(value)); } catch { /* private window */ } },
};

async function json(url, options) {
  const reply = await fetch(url, options);
  const body = await reply.json().catch(() => ({}));
  if (!reply.ok) throw new Error(String(body.detail ?? reply.statusText));
  return body;
}

// What screens a face has, in a few words: for the dialog and for the cards.
export function screensText(screens) {
  if (!screens || !screens.length) return { text: 'No screen watching', ready: false };
  const heard = screens.filter((s) => s.audio_ready).length;
  const n = screens.length;
  const noun = n === 1 ? 'screen' : 'screens';
  if (heard === n) return { text: `${n} ${noun} ready`, ready: true };
  if (heard === 0) return { text: `${n} ${noun}, no sound yet: tap ${n === 1 ? 'it' : 'them'}`, ready: false };
  return { text: `${n} ${noun}, ${heard} with sound: tap the others`, ready: false };
}

const CSS = `
  dialog.screen { background: var(--panel, #171a20); color: var(--text, #e6e9ee); border: 1px solid var(--line, #2a303a);
    border-radius: 12px; padding: 0; width: min(560px, 94vw); }
  dialog.screen::backdrop { background: rgba(0, 0, 0, .6); }
  dialog.screen .wrap { padding: 16px 18px; display: grid; gap: 12px; }
  dialog.screen h2 { margin: 0; font-size: 16px; }
  dialog.screen .state { font-size: 13px; display: grid; gap: 4px; padding: 10px 12px; border-radius: 8px; background: var(--bg, #101216); }
  dialog.screen .state .ok { color: #8bd5a0; } dialog.screen .state .warn { color: var(--warn, #f6c177); }
  dialog.screen .state .bad { color: var(--bad, #f28b82); }
  dialog.screen .opts { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 8px; }
  dialog.screen label { display: grid; gap: 3px; font-size: 12px; color: var(--dim, #9aa4b2); }
  dialog.screen label.check { display: flex; gap: 6px; align-items: center; color: var(--text, #e6e9ee); font-size: 13px; }
  dialog.screen input[type=text], dialog.screen select { padding: 6px 8px; background: var(--bg, #101216); color: inherit;
    border: 1px solid var(--line, #2a303a); border-radius: 6px; font: inherit; width: 100%; box-sizing: border-box; }
  dialog.screen .row { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
  dialog.screen .row input { flex: 1 1 240px; font-family: ui-monospace, monospace; font-size: 12px; }
  dialog.screen .qr { display: grid; place-items: center; background: #fff; border-radius: 8px; padding: 10px; }
  dialog.screen .qr img { width: 220px; height: 220px; image-rendering: pixelated; }
  dialog.screen .note { font-size: 12px; color: var(--dim, #9aa4b2); }
  dialog.screen button { font: inherit; padding: 6px 12px; border-radius: 8px; border: 1px solid #5b6472; background: var(--raised, #1d2128); color: inherit; cursor: pointer; }
  dialog.screen button.primary { background: var(--accent, #6ea8fe); border-color: var(--accent, #6ea8fe); color: #0b1220; font-weight: 600; }
`;

const DEFAULTS = { display: 'flat', quality: 'medium', kiosk: true, hud: false, bg: '' };

export function openScreen(faceId, name = faceId) {
  if (!document.getElementById('screen-css')) document.head.append(el('style', { id: 'screen-css', textContent: CSS }));
  const opts = { ...DEFAULTS, ...store.get('screen', {}) };
  let viewer = '';
  let server = '';

  const stateBox = el('div', { className: 'state' }, el('span', { textContent: 'Asking the server…' }));
  const display = el('select', {}, ...['flat', 'hld'].map((v) => el('option', { value: v, textContent: v === 'hld' ? 'hld (portrait panel)' : 'flat' })));
  const quality = el('select', {}, ...['low', 'medium', 'high'].map((v) => el('option', { value: v, textContent: v })));
  const bg = el('input', { type: 'text', placeholder: 'default, #000, transparent' });
  const kiosk = el('input', { type: 'checkbox' });
  const hud = el('input', { type: 'checkbox' });
  display.value = opts.display; quality.value = opts.quality; bg.value = opts.bg; kiosk.checked = opts.kiosk; hud.checked = opts.hud;
  const link = el('input', { type: 'text', readOnly: true });
  const qrBox = el('div', { className: 'qr', hidden: true });
  const testNote = el('span', { className: 'note' });
  const say = el('input', { type: 'text', readOnly: true });

  const url = () => {
    const p = new URLSearchParams({ face: faceId, display: display.value, quality: quality.value });
    if (kiosk.checked) p.set('kiosk', '1');
    else if (!hud.checked) p.set('hud', '0');
    if (bg.value.trim()) p.set('bg', bg.value.trim());
    return `${viewer}/?${p}`;
  };
  const refresh = () => {
    link.value = viewer ? url() : '';
    store.set('screen', { display: display.value, quality: quality.value, bg: bg.value, kiosk: kiosk.checked, hud: hud.checked });
    if (!qrBox.hidden) drawQr();
    say.value = `argus-mask say "Hello from the other side of the room." --face ${faceId}${server ? ` --url ${server}` : ''}`;
  };
  const drawQr = () => {
    qrBox.replaceChildren();
    if (!window.qrcode || !link.value) { qrBox.append(el('span', { className: 'note', textContent: 'No QR code: the link is not ready.' })); return; }
    const qr = window.qrcode(0, 'M');
    qr.addData(link.value);
    qr.make();
    qrBox.append(el('img', { src: qr.createDataURL(6, 2), alt: `QR code for ${link.value}` }));
  };
  for (const input of [display, quality, bg, kiosk, hud]) input.addEventListener('input', refresh);

  const copy = (input, button) => async () => {
    try { await navigator.clipboard.writeText(input.value); } catch { input.select(); document.execCommand('copy'); }
    const was = button.textContent; button.textContent = 'Copied'; setTimeout(() => { button.textContent = was; }, 1200);
  };
  const openBtn = el('button', { type: 'button', className: 'primary', textContent: 'Open' });
  const copyBtn = el('button', { type: 'button', textContent: 'Copy link' });
  const qrBtn = el('button', { type: 'button', textContent: 'QR code' });
  const testBtn = el('button', { type: 'button', textContent: 'Test: play a chime' });
  const sayBtn = el('button', { type: 'button', textContent: 'Copy' });
  const closeBtn = el('button', { type: 'button', textContent: 'Close' });
  openBtn.onclick = () => window.open(link.value, '_blank', 'noopener');
  copyBtn.onclick = copy(link, copyBtn);
  sayBtn.onclick = copy(say, sayBtn);
  qrBtn.onclick = () => { qrBox.hidden = !qrBox.hidden; if (!qrBox.hidden) drawQr(); };
  testBtn.onclick = async () => {
    testBtn.disabled = true;
    testNote.textContent = 'Sending…';
    try {
      const done = await json(`/screen/${encodeURIComponent(faceId)}/test`, { method: 'POST' });
      testNote.textContent = done.screens ? `Chime sent to ${done.screens} screen${done.screens === 1 ? '' : 's'}: listen for it.` : 'Chime sent, but no screen is watching yet.';
    } catch (err) {
      testNote.textContent = err.message;
    } finally {
      testBtn.disabled = false;
    }
  };

  const dialog = el('dialog', { className: 'screen' }, el('div', { className: 'wrap' },
    el('h2', { textContent: `On screen: ${name} (${faceId})` }),
    stateBox,
    el('div', { className: 'opts' },
      el('label', {}, 'Display', display), el('label', {}, 'Quality', quality), el('label', {}, 'Background', bg),
      el('label', { className: 'check', title: 'No HUD, no cursor, full screen on the first tap' }, kiosk, 'Kiosk'),
      el('label', { className: 'check' }, hud, 'Status line')),
    el('div', { className: 'row' }, link, openBtn, copyBtn, qrBtn),
    qrBox,
    el('div', { className: 'note', textContent: 'On the screen, one tap turns the sound on and fills the screen. A kiosk browser started with --autoplay-policy=no-user-gesture-required needs none.' }),
    el('div', { className: 'row' }, testBtn, testNote),
    el('label', {}, 'Speak from a terminal', el('div', { className: 'row' }, say, sayBtn)),
    el('div', { className: 'row', style: 'justify-content:flex-end' }, closeBtn)));
  document.body.append(dialog);

  // The server's view of this face, every two seconds while the dialog is open.
  let timer = null;
  const poll = async () => {
    let s;
    try { s = await json(`/screen/${encodeURIComponent(faceId)}`); } catch (err) { s = { hint: err.message }; }
    viewer = s.viewer || viewer;
    server = s.server || server;
    const lines = [];
    if (s.server_ok) lines.push(el('span', { className: 'ok', textContent: `Server ${s.server}: up${s.read_only ? ', read-only (it will not speak)' : ''}` }));
    if (s.listed) {
      const t = screensText(s.screens);
      lines.push(el('span', { className: t.ready ? 'ok' : 'warn', textContent: t.text }));
    }
    if (s.hint) lines.push(el('span', { className: 'bad', textContent: s.hint }));
    stateBox.replaceChildren(...lines);
    for (const b of [openBtn, copyBtn, qrBtn]) b.disabled = !s.listed;
    testBtn.disabled = !s.listed || s.read_only;
    refresh();
  };
  const close = () => { clearTimeout(timer); timer = null; dialog.close(); dialog.remove(); };
  closeBtn.onclick = close;
  dialog.addEventListener('cancel', (e) => { e.preventDefault(); close(); });
  const loop = async () => { await poll(); if (dialog.isConnected) timer = setTimeout(loop, 2000); };
  dialog.showModal();
  loop();
}
