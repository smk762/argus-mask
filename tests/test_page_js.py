"""The editor pages' JavaScript, run in node where it can be held without a browser.

Each test takes the functions it needs out of the page's source and runs them against a small fake DOM and
promises the test resolves in the order it wants: the races are in which reply comes back first.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.test_viewer_js import _block

PAGE = Path(__file__).resolve().parent.parent / "src" / "argus_mask" / "editor" / "page"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="needs node")

# A fake DOM: enough of an element for el(), replaceChildren, listeners and focus.
DOM = r"""
class Node {
  constructor(tag) { this.tagName = tag.toUpperCase(); this.children = []; this.dataset = {}; this.listeners = {};
    this.style = {}; this.hidden = false; this.value = ''; this.textContent = ''; this.title = '';
    this.classList = { toggle() {}, add() {}, remove() {} }; }
  append(...kids) { for (const k of kids) this.children.push(typeof k === 'string' ? { text: k } : k); }
  replaceChildren(...kids) { this.children = []; this.append(...kids); }
  addEventListener(type, f) { (this.listeners[type] ||= []).push(f); }
  dispatch(type) { for (const f of this.listeners[type] || []) f({ target: this }); }
  focus() { document.activeElement = this; }
  setSelectionRange(start, end) { this.selectionStart = start; this.selectionEnd = end; }
  get options() { return this.all((c) => c.tagName === 'OPTION'); }   // optgroups' too, as a select's are
  all(pred, out = []) { for (const c of this.children) if (c instanceof Node) { if (pred(c)) out.push(c); c.all(pred, out); } return out; }
  querySelectorAll(sel) { return this.all((n) => n.tagName === sel.toUpperCase()); }
}
const ids = {};
const document = {
  activeElement: null,
  createElement: (tag) => new Node(tag),
  getElementById: (id) => (ids[id] ||= new Node('div')),
  querySelector: () => new Node('div'),
  querySelectorAll: () => [],
};
const requestAnimationFrame = () => 0;
const revoked = [];
let blobs = 0;
URL.createObjectURL = () => `blob:${++blobs}`;
URL.revokeObjectURL = (u) => revoked.push(u);
// A promise the test settles by hand, to have replies arrive in any order.
function later() { let resolve; const p = new Promise((r) => { resolve = r; }); p.resolve = resolve; return p; }
const tick = () => new Promise((r) => setTimeout(r, 0));
"""


def _function(source: str, head: str) -> str:
    """The function that starts with ``head``, to its closing brace (its parameters may hold braces)."""
    start = source.index(head)
    body = source.index(") {", start) + 2
    return source[start:body] + _block(source, body)


def _helpers(source: str, end: str) -> str:
    """The page's opening helpers ($, el, store, json): from ``const $`` to the line ``end``."""
    return source[source.index("const $ = ") : source.index(end)]


def _node(script: str) -> object:
    run = subprocess.run(
        ["node", "--input-type=module", "-e", DOM + script], capture_output=True, text=True, check=True, timeout=30
    )
    return json.loads(run.stdout.strip().splitlines()[-1])


def test_a_build_asked_for_during_an_export_runs_when_it_ends() -> None:
    """The export holds ``busy``: a build queued in that time (``again``) is scheduled once the export finishes."""
    source = (PAGE / "editor.js").read_text(encoding="utf-8")
    at = source.index("$('export').addEventListener('click', async () =>")
    handler = source[at:].split("async () =>", 1)[1]
    handler = "async () =>" + _block(handler, 0)
    out = _node(f"""
let shown = {{ id: 'f1', name: 'F' }};
const made = new Map();
const window = {{ confirm: () => true }};
let busy = false; let again = false;
const scheduled = [];
const schedule = (d) => scheduled.push(d);
const meta = () => {{}}; const status = () => {{}}; const openScreen = () => {{}}; const loadStarts = async () => {{}};
const reply = later();
const json = () => reply;
const exportClick = {handler};
const done = exportClick();
await tick();
again = true;   // a submit or the live timer, seeing busy
reply.resolve({{ path: 'assets/f1' }});
await done;
console.log(JSON.stringify({{ busy, again, scheduled }}));
""")
    assert out == {"busy": False, "again": False, "scheduled": [0]}


VOICES = (PAGE / "voices.js").read_text(encoding="utf-8")
VOICES_HELPERS = (
    _helpers(VOICES, "\n// Scripts for generated")
    + "\nconst verdictClass = (v) => v;\nconst checksTable = () => el('table');\n"
)


def test_only_the_latest_set_verdict_gates_make() -> None:
    """Two assessments in flight: the one for the set chosen last stands, however the replies arrive."""
    out = _node(f"""
const $ = (id) => document.getElementById(id);
const el = (tag, props = {{}}, ...children) => {{ const n = Object.assign(document.createElement(tag), props); n.append(...children); return n; }};
const verdictClass = (v) => v; const checksTable = () => el('table');
let setReport = null; let assessSeq = 0;
const clips = [{{ use: true, report: {{ id: 1 }} }}, {{ use: true, report: {{ id: 2 }} }}];
const replies = [];
const post = () => {{ const p = later(); replies.push(p); return p; }};
const gated = [];
const paintMake = () => gated.push(setReport?.verdict);
{_function(VOICES, "async function assessSet(")}
const first = assessSet();
clips[1].use = false;   // the user drops a clip: a second assessment, for one clip
const second = assessSet();
replies[1].resolve({{ verdict: 'not suitable', metrics: {{ speech_seconds: 4 }}, checks: [] }});
await second;
replies[0].resolve({{ verdict: 'good', metrics: {{ speech_seconds: 30 }}, checks: [] }});
await first;
console.log(JSON.stringify({{ verdict: setReport.verdict, gated }}));
""")
    assert out == {"verdict": "not suitable", "gated": ["not suitable"]}


def test_trying_a_voice_shows_a_structured_refusal_as_text_and_frees_the_last_sample() -> None:
    """A validation error's detail is a list: it reads as JSON, not ``[object Object]``. Each new sample frees the last."""
    out = _node(f"""
{VOICES_HELPERS}
globalThis.localStorage = {{ getItem: () => null, setItem: () => {{}} }};
const answers = [
  {{ ok: false, statusText: 'Unprocessable', json: async () => ({{ detail: [{{ loc: ['body', 'text'], msg: 'too long' }}] }}) }},
  {{ ok: true, blob: async () => 'a' }},
  {{ ok: true, blob: async () => 'b' }},
];
globalThis.fetch = async () => answers.shift();
const audio = $('try-audio'); audio.src = ''; audio.play = async () => {{}};
{_function(VOICES, "async function tryVoice(")}
await tryVoice();
const refused = $('try-note').textContent;
await tryVoice(); await tryVoice();
console.log(JSON.stringify({{ refused, src: audio.src, revoked }}));
""")
    assert out["refused"] == json.dumps([{"loc": ["body", "text"], "msg": "too long"}], separators=(",", ":"))
    assert out["src"] == "blob:2"
    assert out["revoked"] == ["blob:1"]


CLIPS = f"""
{VOICES_HELPERS}
let setReport = null;
const clips = [
  {{ key: 'a', name: 'a.wav', transcript: 'One.', use: true, url: 'blob:a', report: null }},
  {{ key: 'b', name: 'b.wav', transcript: '', use: true, url: 'blob:b', report: null }},
];
const assessSet = () => {{}}; const check = () => {{}}; const paintMake = () => {{}}; const drawWave = () => {{}};
{_function(VOICES, "function paintClips(")}
const areas = () => $('clips').querySelectorAll('textarea');
"""


def test_a_repaint_keeps_a_transcript_being_typed() -> None:
    """Another clip's check ending repaints every card: the text typed but not yet committed stays, with the focus."""
    out = _node(f"""
{CLIPS}
paintClips();
const typing = areas()[1];
typing.value = 'Hello the';
typing.dispatch('input');
typing.focus(); typing.setSelectionRange(5, 9);
paintClips();   // a background check ends
const now = areas()[1];
console.log(JSON.stringify({{ value: now.value, focused: document.activeElement === now, at: [now.selectionStart, now.selectionEnd],
  committed: clips[1].transcript }}));
""")
    assert out == {"value": "Hello the", "focused": True, "at": [5, 9], "committed": ""}


def test_removing_a_clip_frees_its_audio() -> None:
    out = _node(f"""
{CLIPS}
paintClips();
$('clips').all((n) => n.textContent === 'Remove')[0].onclick();
console.log(JSON.stringify({{ revoked, left: clips.map((c) => c.key) }}));
""")
    assert out == {"revoked": ["blob:a"], "left": ["b"]}


def test_a_late_voice_list_for_an_old_tts_is_dropped() -> None:
    """The TTS URL changes A then B: A's list arriving last does not fill the select for B."""
    source = (PAGE / "broadcast.js").read_text(encoding="utf-8")
    out = _node(f"""
const $ = (id) => document.getElementById(id);
const el = (tag, props = {{}}, ...children) => {{ const n = Object.assign(document.createElement(tag), props); n.append(...children); return n; }};
const store = {{ get: (k, f) => f, set() {{}} }};
let state = {{ tts_voice: '' }};
const updateCli = () => {{}};
const replies = {{}};
const json = (url) => (replies[url] = later());
let voicesFor = null; let listed = []; let face = '';
const OTHER = '\\u0001other';
const faceVoice = () => null;
{_function(source, "function wantedVoice(")}
{_function(source, "function selectVoice(")}
{_function(source, "function fillVoices(")}
{_function(source, "async function loadVoices(")}
$('tts').value = 'http://a'; const a = loadVoices();
$('tts').value = 'http://b'; const b = loadVoices();
replies['/voices?tts=' + encodeURIComponent('http://b')].resolve({{ listed: true, voices: [{{ name: 'from-b' }}] }}); await b;
replies['/voices?tts=' + encodeURIComponent('http://a')].resolve({{ listed: true, voices: [{{ name: 'from-a' }}] }}); await a;
console.log(JSON.stringify($('voice').options.map((o) => o.value)));
""")
    assert out == ["", "from-b", "\u0001other"]


def test_overlapping_reviews_show_one_face_throughout() -> None:
    """Face X's model loading after face Y's: the side panel and buttons are Y's, as the stage is."""
    source = (PAGE / "faces.js").read_text(encoding="utf-8")
    out = _node(f"""
const $ = (id) => document.getElementById(id);
const el = (tag, props = {{}}, ...children) => {{ const n = Object.assign(document.createElement(tag), props); n.append(...children); return n; }};
const enc = encodeURIComponent; const main = () => 'export'; const status = () => {{}}; const restSets = () => {{}};
const modelUrl = (id) => id;
const face = (id) => ({{ id, name: id.toUpperCase(), model: {{ sha256: id, file: 'f.glb', format: 'gltf' }}, framing: {{}}, blendshapes: [],
  provenance: {{ license: 'CC0', kind: 'synthetic', cleared: true, source: 's', notes: '' }} }});
const json = async (url) => (url.startsWith('/face/') ? face(url.split('/')[2].split('?')[0]) : url === '/scenes' ? [] : {{}});
const loads = []; const posed = []; const scenes = [];
const stage = {{
  attach() {{}},
  load: (url) => {{ const p = later(); loads.push(p); return p; }},
  poseButtons: (box, _, o) => posed.push(o.faceId),
  setScene: async (n) => scenes.push(n),
}};
let reviewSeq = 0;
{_function(source, "async function openReview(")}
{_function(source, "async function backdropPicker(")}
const x = openReview({{ id: 'x' }}); await tick();
const y = openReview({{ id: 'y' }}); await tick();
loads[1].resolve(); await y;
loads[0].resolve(); await x;
console.log(JSON.stringify({{ name: $('m-name').textContent, posed, scenes: scenes.length }}));
""")
    assert out == {"name": "Y (y)", "posed": ["y"], "scenes": 1}


def test_an_open_pack_with_few_assets_can_be_closed() -> None:
    """A pack of three assets opened with no filter keeps its ``Hide assets`` button."""
    source = (PAGE / "packs.js").read_text(encoding="utf-8")
    out = _node(f"""
const el = (tag, props = {{}}, ...children) => {{ const n = Object.assign(document.createElement(tag), props);
  n.append(...children.filter((c) => c !== null && c !== undefined && c !== false)); return n; }};
const SHOWN = 12; const USED = new Set(['hair']); const open = new Set();
const licenceClass = () => 'good'; const assetCard = () => el('div'); const action = () => el('span'); const render = () => {{}};
{_function(source, "function packCard(")}
const pack = {{ id: 'p', page: '#', license: 'CC0', assets: [1, 2, 3].map((i) => ({{ type: 'hair', name: `a${{i}}` }})) }};
const button = (card) => card.all((n) => n.tagName === 'BUTTON').map((b) => b.textContent);
const closed = button(packCard(pack, pack.assets, false));
open.add('p');
const opened = button(packCard(pack, pack.assets, false));
console.log(JSON.stringify({{ closed, opened }}));
""")
    assert out == {"closed": ["Show 3 assets"], "opened": ["Hide assets"]}


def test_a_qr_code_drawn_on_a_canvas_is_not_transposed() -> None:
    """``renderTo2dContext`` puts module (row, col) at x = col, y = row, as the image and table renderers do."""
    script = f"""
const qrcode = require({json.dumps(str(PAGE / "qr" / "qrcode.js"))});
const qr = qrcode(0, 'M');
qr.addData('http://203.0.113.10:8109/?face=stock-01&kiosk=1');
qr.make();
const dark = new Set();
const ctx = {{ fillStyle: '', fillRect(x, y) {{ if (this.fillStyle === 'black') dark.add(`${{x}},${{y}}`); }} }};
qr.renderTo2dContext(ctx, 1);
const n = qr.getModuleCount();
let wrong = 0, asymmetric = 0;
for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {{
  if (qr.isDark(r, c) !== dark.has(`${{c}},${{r}}`)) wrong++;
  if (qr.isDark(r, c) !== qr.isDark(c, r)) asymmetric++;
}}
console.log(JSON.stringify({{ wrong, asymmetric: asymmetric > 0 }}));
"""
    run = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True, timeout=30)
    assert json.loads(run.stdout) == {"wrong": 0, "asymmetric": True}
