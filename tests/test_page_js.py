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
let busy = false; let again = false; let submitted = false;
const scheduled = [];
const schedule = (d) => scheduled.push(d); const runBuild = () => {{}};
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


def test_a_missing_clip_is_not_posted_as_audio() -> None:
    """Re-checking a clip that is gone (404) says it could not be read, and does not send the error page on."""
    out = _node(f"""
const $ = (id) => document.getElementById(id);
let setReport = null;
const clips = [];
const paintClips = () => {{}}; const paintMake = () => {{}}; const assessSet = () => {{}};
let posted = 0;
const json = async () => {{ posted += 1; return {{ sha: 'abc', report: {{}} }}; }};
globalThis.fetch = async () => ({{ ok: false, status: 404, blob: async () => '<html>missing</html>' }});
{_function(VOICES, "async function check(")}
const clip = {{ name: 'a.wav', transcript: 'Hello.', sha: 'dead', url: '/voice/clip/dead' }};
await check(clip);
console.log(JSON.stringify({{ error: clip.error, posted, checking: clip.checking }}));
""")
    assert out["posted"] == 0 and out["checking"] is False
    assert "could not be read" in out["error"]


def test_make_stays_disabled_when_voices_cannot_be_made_and_no_reason_is_given() -> None:
    out = _node(f"""
const $ = (id) => document.getElementById(id);
let state = {{ can_make: false }};
let setReport = null;
const clips = [];
{_function(VOICES, "function paintMake(")}
paintMake();
const blocked = {{ disabled: $('v-make').disabled, note: $('v-note').textContent }};
state = {{ can_make: false, reason: 'The TTS service is down.' }};
paintMake();
console.log(JSON.stringify({{ blocked, reason: $('v-note').textContent, disabled: $('v-make').disabled }}));
""")
    assert out["blocked"] == {"disabled": True, "note": "Voices cannot be made right now."}
    assert out["disabled"] is True and out["reason"] == "The TTS service is down."


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
  setShot: () => {{}},
}};
let reviewClicks = 0; let reviewSeq = 0;
{_function(source, "async function openReview(")}
{_function(source, "async function backdropPicker(")}
const x = openReview({{ id: 'x' }}); await tick();
const y = openReview({{ id: 'y' }}); await tick();
loads[1].resolve(); await y;
loads[0].resolve(); await x;
console.log(JSON.stringify({{ name: $('m-name').textContent, posed, scenes: scenes.length }}));
""")
    assert out == {"name": "Y (y)", "posed": ["y"], "scenes": 1}


def _reviews(source: str, script: str) -> object:
    """openReview and its backdrop picker over a stage whose loads the test settles; ``face(id)`` is a manifest."""
    counters = source[source.index("let reviewClicks = 0;") : source.index("async function openReview(")]
    return _node(f"""
const $ = (id) => document.getElementById(id);
const el = (tag, props = {{}}, ...children) => {{ const n = Object.assign(document.createElement(tag), props); n.append(...children); return n; }};
const enc = encodeURIComponent; const main = () => 'export'; const restSets = () => {{}};
const said = []; const status = (text) => said.push(text);
const modelUrl = (id) => id;
const face = (id) => ({{ id, name: id.toUpperCase(), model: {{ sha256: id, file: 'f.glb', format: 'gltf' }}, framing: {{}}, blendshapes: [],
  provenance: {{ license: 'CC0', kind: 'synthetic', cleared: true, source: 's', notes: '' }} }});
const manifests = {{}};   // a face id here: its manifest waits for the test to settle it
const json = async (url) => {{
  if (!url.startsWith('/face/')) return url === '/scenes' ? [] : {{}};
  const id = url.split('/')[2].split('?')[0];
  return id in manifests ? manifests[id] : face(id);
}};
const loads = []; const posed = [];
const stage = {{
  attach() {{}},
  load: (url) => {{ const p = later(); p.url = url; loads.push(p); return p; }},
  poseButtons: (box, _, o) => posed.push(o.faceId),
  setScene: async () => {{}},
  setShot: () => {{}},
}};
const shown = () => ({{ name: $('m-name').textContent, posed, loads: loads.map((l) => l.url), side: $('m-side').children.length > 0, said }});
{counters}
{_function(source, "async function openReview(")}
{_function(source, "async function backdropPicker(")}
{script}
console.log(JSON.stringify(shown()));
""")


def test_a_review_that_cannot_load_leaves_the_one_in_progress_whole() -> None:
    """X's model still loading when a click on Y is refused (it has no manifest): X finishes as it would have.

    The stage is X's either way: Y never reached it. So the name, the pose buttons and the side panel (whose
    pickers save a rest set and a backdrop) must be X's too, not those of the review before.
    """
    source = (PAGE / "faces.js").read_text(encoding="utf-8")
    out = _reviews(
        source,
        """
manifests.y = Promise.reject(new Error('no manifest.json')); manifests.y.catch(() => {});
const x = openReview({ id: 'x' }); await tick();
await openReview({ id: 'y' });
loads[0].resolve(); await x;
""",
    )
    assert out == {"name": "X (x)", "posed": ["x"], "loads": ["x"], "side": True, "said": ["no manifest.json"]}


def test_a_manifest_arriving_late_for_an_earlier_click_never_takes_the_stage() -> None:
    """X clicked, then Y: Y's manifest comes first and Y has the stage; X's, arriving after, loads nothing."""
    source = (PAGE / "faces.js").read_text(encoding="utf-8")
    out = _reviews(
        source,
        """
manifests.x = later();
const x = openReview({ id: 'x' }); await tick();
const y = openReview({ id: 'y' }); await tick();
manifests.x.resolve(face('x')); await x;
loads[0].resolve(); await y;
""",
    )
    assert out == {"name": "Y (y)", "posed": ["y"], "loads": ["y"], "side": True, "said": []}


def test_an_open_pack_with_few_assets_can_be_closed() -> None:
    """A pack of three assets opened with no filter keeps its ``Hide assets`` button."""
    source = (PAGE / "packs.js").read_text(encoding="utf-8")
    out = _node(f"""
const el = (tag, props = {{}}, ...children) => {{ const n = Object.assign(document.createElement(tag), props);
  n.append(...children.filter((c) => c !== null && c !== undefined && c !== false)); return n; }};
const SHOWN = 12; const USED = new Set(['hair']); const open = new Set();
const licenceClass = () => 'good'; const assetCard = () => el('div'); const action = () => el('span'); const render = () => {{}};
const location = {{ href: 'http://203.0.113.10:8110/packs.html' }};
{_function(source, "function webPage(")}
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


EL = (
    "const el = (tag, props = {}, ...children) => { const n = Object.assign(document.createElement(tag), props);\n"
    "  n.append(...children.filter((c) => c !== null && c !== undefined && c !== false)); return n; };\n"
    "const $ = (id) => document.getElementById(id);\n"
)
THUMBS = f"""
const cache = new Map(); let queue = Promise.resolve(); let stage = null;
const snaps = [];
class Stage {{ async load(url, framing) {{ this.framing = framing; }} async snapshot() {{ snaps.push(this.framing); return `still:${{snaps.length}}`; }} }}
const modelUrl = (id) => id;
{_function((PAGE / "thumbs.js").read_text(encoding="utf-8"), "function still(")}
"""


def test_a_still_follows_a_framing_edit_that_keeps_the_model() -> None:
    """The same model at a new framing is a new still, not the cached one."""
    out = _node(f"""
{THUMBS}
const manifests = [{{ distance: 0.6 }}, {{ distance: 1.2 }}].map((framing) => ({{ model: {{ sha256: 'aa', format: 'gltf' }}, framing }}));
globalThis.fetch = async () => ({{ ok: true, json: async () => manifests.shift() }});
const first = await still('f1'); const second = await still('f1');
console.log(JSON.stringify({{ first, second, snaps }}));
""")
    assert out == {"first": "still:1", "second": "still:2", "snaps": [{"distance": 0.6}, {"distance": 1.2}]}


def test_a_still_of_a_face_with_no_model_says_so() -> None:
    """An ok reply that is not a manifest (no ``model``) rejects with the reason, not a TypeError."""
    out = _node(f"""
{THUMBS}
globalThis.fetch = async () => ({{ ok: true, json: async () => {{ throw new SyntaxError('not JSON'); }} }});
const why = await still('f1').then(() => 'loaded', (err) => `${{err.constructor.name}}: ${{err.message}}`);
console.log(JSON.stringify(why));
""")
    assert out == "Error: f1 has no model"


def test_a_pack_page_that_is_not_a_web_page_is_not_a_link() -> None:
    """A catalogue's ``javascript:`` or ``data:`` page is shown as text: only http and https become a link."""
    source = (PAGE / "packs.js").read_text(encoding="utf-8")
    out = _node(f"""
{EL}
const location = {{ href: 'http://203.0.113.10:8110/packs.html' }};
const SHOWN = 12; const USED = new Set(['hair']); const open = new Set();
const licenceClass = () => 'good'; const assetCard = () => el('div'); const action = () => el('span'); const render = () => {{}};
{_function(source, "function webPage(")}
{_function(source, "function packCard(")}
const links = (page) => packCard({{ id: 'p', page, license: 'CC0', assets: [] }}, [], false).all((n) => n.tagName === 'A').map((a) => a.href);
console.log(JSON.stringify(['javascript:alert(1)', 'data:text/html,x', 'https://example.org/p'].map(links)));
""")
    assert out == [[], [], ["https://example.org/p"]]


def test_only_the_latest_check_of_a_clip_stands() -> None:
    """Two quick transcript edits: the older check's reply never overwrites the newer, nor ungates Make early."""
    out = _node(f"""
let paints = 0;
const paintClips = () => {{}}; const paintMake = () => {{}}; const assessSet = () => {{}};
const replies = [];
const json = () => {{ const p = later(); replies.push(p); return p; }};
{_function(VOICES, "async function check(")}
const clip = {{ name: 'a.wav', transcript: 'old', url: 'blob:a' }};
const first = check(clip, 'wav'); await tick();
clip.transcript = 'new';
const second = check(clip, 'wav'); await tick();
replies[0].resolve({{ sha: 's', report: {{ transcript: 'old' }} }}); await first;
const checkingAfterOld = clip.checking;
replies[1].resolve({{ sha: 's', report: {{ transcript: 'new' }} }}); await second;
// and the other way round: the newer reply first, the older last
const third = check(clip, 'wav'); await tick();
const fourth = check(clip, 'wav'); await tick();
replies[3].resolve({{ sha: 's', report: {{ transcript: 'fourth' }} }}); await fourth;
replies[2].resolve({{ sha: 's', report: {{ transcript: 'third' }} }}); await third;
console.log(JSON.stringify({{ checkingAfterOld, report: clip.report.transcript, checking: clip.checking }}));
""")
    assert out == {"checkingAfterOld": True, "report": "fourth", "checking": False}


def test_an_alias_to_a_face_not_served_shows_that_face() -> None:
    """An alias whose face is refused or gone: its select shows that face, not the first served one."""
    source = (PAGE / "faces.js").read_text(encoding="utf-8")
    out = _node(f"""
{EL}
const rows = [{{ id: 'a', export: {{}} }}, {{ id: 'b', export: {{}} }}];
let aliases = {{ holly: 'gone' }};
const setAlias = () => {{}}; const openScreen = () => {{}}; const json = async () => {{}}; const load = async () => {{}};
const status = () => {{}}; const enc = encodeURIComponent; const window = {{ confirm: () => true }};
{_function(source, "function renderAliases(")}
renderAliases();
const pick = $('alias-list').all((n) => n.tagName === 'SELECT')[0];
console.log(JSON.stringify(pick.options.filter((o) => o.selected).map((o) => [o.value, Boolean(o.disabled)])));
""")
    assert out == [["gone", True]]


def test_closing_a_review_still_loading_keeps_it_closed() -> None:
    """The manifest arriving after the modal was closed does not open it again, nor load the stage."""
    source = (PAGE / "faces.js").read_text(encoding="utf-8")
    out = _node(f"""
{EL}
const window = {{ addEventListener() {{}} }};
const enc = encodeURIComponent; const main = () => 'export'; const status = () => {{}}; const modelUrl = (id) => id;
const reply = later();
const json = () => reply;
const loads = [];
const stage = {{ attach() {{}}, load: async (url) => loads.push(url), poseButtons() {{}}, setScene: async () => {{}} }};
let reviewClicks = 0; let reviewSeq = 0;
{_function(source, "async function openReview(")}
{_function(source, "function closeReview(")}
$('modal').hidden = true;
const review = openReview({{ id: 'x' }}); await tick();
closeReview();
reply.resolve({{ id: 'x', name: 'X', model: {{ sha256: 'x', format: 'gltf' }}, framing: {{}}, blendshapes: [], provenance: {{}} }});
await review;
console.log(JSON.stringify({{ hidden: $('modal').hidden, loads: loads.length }}));
""")
    assert out == {"hidden": True, "loads": 0}


def test_a_build_asked_for_during_an_export_runs_even_for_a_new_name() -> None:
    """Build pressed while an export runs, after only the name changed: it is built, not skipped as unchanged."""
    source = (PAGE / "editor.js").read_text(encoding="utf-8")
    at = source.index("$('export').addEventListener('click', async () =>")
    handler = "async () =>" + _block(source[at:].split("async () =>", 1)[1], 0)
    out = _node(f"""
let shown = {{ id: 'f1', name: 'F' }};
const made = new Map();
const window = {{ confirm: () => true }};
let busy = false; let again = false; let submitted = false;
const scheduled = []; const builds = [];
const schedule = (d) => scheduled.push(d); const runBuild = () => builds.push('build');
const meta = () => {{}}; const status = () => {{}}; const openScreen = () => {{}}; const loadStarts = async () => {{}};
const reply = later();
const json = () => reply;
const exportClick = {handler};
const done = exportClick();
await tick();
submitted = true;   // Build pressed during the export: only the name changed, so schedule() would skip it
reply.resolve({{ path: 'assets/f1' }});
await done;
console.log(JSON.stringify({{ busy, builds, scheduled }}));
""")
    assert out == {"busy": False, "builds": ["build"], "scheduled": []}


EDITOR = (PAGE / "editor.js").read_text(encoding="utf-8")
_SUBMIT = EDITOR.index("$('form').addEventListener('submit', (event) =>")
EDITOR_BUILD = f"""
{EL}
let busy = false; let again = false; let submitted = false; let formGen = 0; let built = null; let shownContent = null;
let failedContent = null; let timer = null; let canBuild = true; let detached = false; let made = new Map();
const DEFAULT = {{ id: 'new', name: 'New', hair: 0 }}; let preset = {{ ...DEFAULT }};
const contentOf = (body) => JSON.stringify({{ ...body, id: undefined, name: undefined }});
let form = {{ id: 'a', name: 'A', hair: 1 }};
const current = () => ({{ ...form }});
const adopt = (p) => {{ form = {{ ...p }}; }}; const syncForm = () => {{}};
const saved = {{ id: 'saved', name: 'Saved', hair: 2 }};
const builds = [];
const json = async (url, opts) => {{
  if (url === '/presets') return [{{ preset: saved, exported: true, previewed: true }}];
  const reply = later(); builds.push({{ body: JSON.parse(opts.body), reply }}); return reply;
}};
const building = () => {{}}; const meta = () => {{}}; const status = () => {{}}; const remember = () => {{}};
const showBefore = async () => {{}}; const renderHistory = () => {{}};
const store = {{ get: (k, f) => f, set() {{}} }}; const live = () => false; const schedule = () => {{}};
const onShow = []; const showFace = async (id) => {{ onShow.push(id); }};
const built_ = (id) => ({{ manifest: {{ id }}, key: id, seconds: 1, cached: true }});
const settle = async () => {{ for (let i = 0; i < 20; i++) await tick(); }};
{_function(EDITOR, "async function loadStarts(")}
{_function(EDITOR, "async function runBuild(")}
const submit = {_block(EDITOR, EDITOR.index("(event) =>", _SUBMIT))};
const press = () => submit({{ preventDefault() {{}} }});
"""


def test_a_build_asked_for_during_a_build_runs_even_for_a_new_name() -> None:
    """Build pressed while Blender works, after only the name changed: the promised build runs when it ends."""
    out = _node(f"""
{EDITOR_BUILD}
press(); await tick();
form.name = 'B';   // meta() says: Name or id changed: Build to apply it
press(); await tick();
builds[0].reply.resolve(built_('a')); await settle();
console.log(JSON.stringify(builds.map((b) => b.body.name)));
""")
    assert out == ["A", "B"]


def test_a_face_chosen_during_a_build_keeps_the_stage() -> None:
    """A saved face picked while a build runs stays on show, in the form and the selector, when the build ends."""
    out = _node(f"""
{EDITOR_BUILD}
await loadStarts();
press(); await tick();
$('start').value = 'saved'; await $('start').onchange();
builds[0].reply.resolve(built_('a')); await settle();
console.log(JSON.stringify({{ onShow, start: $('start').value, form: form.id, shown: shownContent === contentOf(saved) }}));
""")
    assert out == {"onShow": ["saved"], "start": "saved", "form": "saved", "shown": True}


def test_a_demo_clip_that_fails_to_load_frees_its_audio_and_model() -> None:
    """A clip whose backdrop or files fail: its audio blob is revoked and its loaded model disposed."""
    source = (Path(__file__).resolve().parent.parent / "demo" / "player.js").read_text(encoding="utf-8")
    loader = source[source.index("async function load(") : source.index("function ready(")]
    out = _node(f"""
const statusEl = {{}}; const faces = new Map(); const audio = {{ src: '' }};
let files = 'ok';
const json = async (u) => {{
  if (files === 'broken') throw new Error('no clip.json');
  return u.endsWith('clip.json') ? {{ face: 'f', framing: 'head', scene: 'garden' }} : {{ id: 'f', model: {{ file: 'f.glb', format: 'gltf' }}, framing: {{}} }};
}};
const readTake = async () => ({{ frames: [], header: {{ blendshapes: [] }} }});
const audioUrl = async () => URL.createObjectURL();
const stale = () => false; const loadSplat = async () => {{}};
const disposed = []; const dispose = (root) => disposed.push(root.name);
const scene = {{ remove() {{}} }};
class GLTFLoader {{ async loadAsync() {{ return {{ scene: {{ name: 'model' }} }}; }} }}
const Backdrop = {{ create: async () => {{ throw new Error('no plate'); }} }};
const THREE = {{}}; const renderer = {{}}; const look = {{ summary: {{ quality: 'high' }} }};
{loader}
const errors = [];
await load('c1', 1).catch((e) => errors.push(e.message));
files = 'broken';
await load('c2', 2).catch((e) => errors.push(e.message));
await tick();
console.log(JSON.stringify({{ errors, revoked, disposed }}));
""")
    assert out == {"errors": ["no plate", "no clip.json"], "revoked": ["blob:1", "blob:2"], "disposed": ["model"]}
