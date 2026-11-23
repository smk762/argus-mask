"""Checks for rules in CLAUDE.md's "Must never happen" table that no other test held (#130).

Each test names the rule it holds. They are written against the apps' route tables and the source, not one route
or one call, so a new route or a new engine module is caught too.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
from html.parser import HTMLParser
from pathlib import Path

import pytest
from starlette.routing import Route, WebSocketRoute
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from argus_mask.config import EditorSettings, Settings
from argus_mask.editor.app import create_editor_app
from argus_mask.models import CLEARED_KINDS, FaceManifest, Provenance
from argus_mask.server.app import POLICY, create_app
from tests.helpers import RELEASE, make_face, write_consent

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "src" / "argus_mask"
VIEWER = PACKAGE / "viewer"
CONSENTED = Provenance(
    kind="consented", source="a capture session", license="per the release", subject="A. Person", release_sha256=RELEASE
)
EVIL = {"origin": "https://evil.example", "sec-fetch-site": "cross-site"}


@pytest.fixture
def apps(tmp_path):
    """The server and the editor over one assets root holding a consented face, its record, and a plain face."""
    assets = tmp_path / "assets"
    make_face(assets, "agreed", provenance=CONSENTED)
    make_face(assets, "test-head")
    write_consent(assets)
    ed = EditorSettings(voices_dir=tmp_path / "voices", recordings_dir=tmp_path / "rec")
    return create_app(Settings(assets_dir=assets)), create_editor_app(Settings(assets_dir=assets), ed)


# Every path parameter the apps use, by name. A face is only ever ``face_id`` (and its model ``file``); the rest
# address other things. A new parameter must be added here, so a face route under another name cannot be filled
# with a placeholder and pass without the face ever being asked about.
NOT_A_FACE = {
    "name",
    "part",
    "pack_id",
    "job_id",
    "key",
    "sha",
    "voice_id",
    "build_id",
    "candidate",
    "record_id",  # darkroom generation id (#167), not a face
}


def _path(route: Route, face: str = "agreed") -> str:
    """The route's path with its parameters filled: the face, its model file, and a placeholder elsewhere."""

    def fill(m: re.Match) -> str:
        param = m.group(1)
        if param == "face_id":
            return face
        if param == "file":
            return "head.glb"  # make_face's model: a model route is asked for the face's own file
        assert param in NOT_A_FACE, f"{route.path}: add {{{param}}} to _path, as a face id or NOT_A_FACE"
        return "x"

    return re.sub(r"{(\w+)(:\w+)?}", fill, route.path)


# ── The consent record is never served, only verdicts ─────────────────────────────────────────────────────────


def test_no_route_serves_a_consent_record(apps) -> None:
    """Every GET route of both apps, asked about the consented face: none answers with the record's own fields.

    The subject and the release's hash are the face's provenance, served by design. What only the record holds
    (who collected it, the reference media's hashes, its terms) must never leave the machine.
    """
    record_only = ("B. Collector", "cd" * 32, "ef" * 32, "collected_by", "reference_images", "notice_days")
    # The face routes that may refuse the consented face, and why; every other one must serve it, so an error
    # reply (which cannot leak) never stands in for a route that was really asked.
    may_refuse = {"/saved/{face_id}/preset": "the editor did not make it, so it has no preset"}
    served = 0
    for app in apps:
        with TestClient(app) as client:
            for route in app.routes:
                if not isinstance(route, Route) or "GET" not in (route.methods or ()):
                    continue
                params = {"use": "video"} if route.path.endswith("/consent") else {}
                reply = client.get(_path(route), params=params)
                leaked = [f for f in record_only if f in reply.text]
                assert not leaked, f"GET {route.path} served {leaked}"
                if "{face_id}" in route.path and route.path not in may_refuse:
                    assert reply.status_code < 300, f"GET {route.path} answered {reply.status_code}: never asked"
                served += reply.status_code < 300
    assert served > 30  # the route tables were walked and answered, not an empty list or a wall of errors


# ── A socket checks the origin; a state-changing route is guarded ─────────────────────────────────────────────


def test_every_write_refuses_another_site(apps) -> None:
    """Every POST/PUT/PATCH/DELETE route of both apps: another site's page gets 403, whatever the route is."""
    checked = 0
    for app in apps:
        with TestClient(app) as client:
            for route in app.routes:
                if not isinstance(route, Route):
                    continue
                for method in sorted((route.methods or set()) & {"POST", "PUT", "PATCH", "DELETE"}):
                    reply = client.request(method, _path(route), headers=EVIL, content=b"{}")
                    assert reply.status_code == 403, f"{method} {route.path} answered {reply.status_code}"
                    checked += 1
    assert checked > 20


def test_every_socket_refuses_another_site(apps) -> None:
    """Every WebSocket route of both apps: CORS and the write guard do not cover sockets, so each checks itself."""
    sockets = [(app, r) for app in apps for r in app.routes if isinstance(r, WebSocketRoute)]
    assert sockets
    for app, route in sockets:
        with TestClient(app) as client:
            # A face that is not there: a socket that skips the origin check answers "no face" at once, and fails
            # here, where a real face would leave a viewer waiting for a stream that never comes.
            connecting = client.websocket_connect(_path(route, "nobody"), headers=EVIL)
            try:
                ws = connecting.__enter__()
            except WebSocketDisconnect as exc:  # refused in the handshake: only the policy code says it was this
                assert exc.code == POLICY, f"{route.path} closed the handshake with {exc.code}, not {POLICY}"
                continue
            try:
                # Accepted: the one acceptable answer is the cross-site refusal. A close for any other reason
                # (an unknown face, bad parameters) means the origin was never checked.
                reply = ws.receive_json()
            except WebSocketDisconnect as exc:
                pytest.fail(f"{route.path} accepted another site, then closed with {exc.code}")
            finally:
                connecting.__exit__(None, None, None)
            assert "cross-site" in reply.get("detail", ""), f"{route.path} let another site in: {reply}"


# ── The disclosure label shows whatever ?hud says ─────────────────────────────────────────────────────────────


def _function(source: str, name: str) -> str:
    start = source.index(f"function {name}(")
    depth, i = 0, source.index("{", start)
    for j in range(i, len(source)):
        depth += {"{": 1, "}": -1}.get(source[j], 0)
        if depth == 0:
            return source[start : j + 1]
    raise AssertionError(f"{name} has no end")


class _Ancestry(HTMLParser):
    """The ids of the elements open around the one with id *target*, as an HTML parser nests them."""

    VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}

    def __init__(self, target: str) -> None:
        super().__init__()
        self.target, self.open, self.found = target, [], None

    def handle_starttag(self, tag: str, attrs: list) -> None:
        ident = dict(attrs).get("id")
        if ident == self.target and self.found is None:
            self.found = [i for _, i in self.open if i]
        if tag not in self.VOID:
            self.open.append((tag, ident))

    def handle_endtag(self, tag: str) -> None:
        for i in range(len(self.open) - 1, -1, -1):  # close back to the matching tag, as a browser does
            if self.open[i][0] == tag:
                del self.open[i:]
                return


def _hides(declarations: str) -> bool:
    flat = re.sub(r"\s+", "", declarations.lower())
    return bool(re.search(r"(?:^|;)(?:display:none|visibility:hidden|opacity:0(?:\.0*)?)(?:;|!|$)", flat))


def _unwrap_at_rules(css: str) -> str:
    """Drop at-rule preludes so a selector nested in ``@media`` or ``@supports`` is matched on its own."""
    out: list[str] = []
    i, n = 0, len(css)
    while i < n:
        if css[i] != "@":
            nxt = css.find("@", i)
            if nxt < 0:
                out.append(css[i:])
                break
            out.append(css[i:nxt])
            i = nxt
            continue
        brace = css.find("{", i)
        if brace < 0:
            out.append(css[i:])
            break
        depth = 0
        for j in range(brace, n):
            if css[j] == "{":
                depth += 1
            elif css[j] == "}":
                depth -= 1
                if depth == 0:
                    out.append(_unwrap_at_rules(css[brace + 1 : j]))
                    i = j + 1
                    break
        else:
            out.append(css[i:])
            break
    return "".join(out)


def _label_rules(css: str) -> list[tuple[str, str]]:
    """Every CSS rule whose selector names the disclosure label: (selector, declarations)."""
    css = _unwrap_at_rules(re.sub(r"/\*.*?\*/", "", css, flags=re.S))
    return [(sel.strip(), body) for sel, body in re.findall(r"([^{}]+){([^}]*)}", css) if "disclosure" in sel]


@pytest.mark.parametrize(
    ("page", "ok"),
    [
        ('<div id="hud"></div><div id="disclosure"></div>', True),
        ('<div id="hud"><div>status</div><div id="disclosure"></div></div>', False),  # past a nested </div>
        ('<div id="hud"><span><br><div id="disclosure"></div></span></div>', False),
    ],
)
def test_the_label_check_sees_it_anywhere_inside_the_hud(page: str, ok: bool) -> None:
    found = _Ancestry("disclosure")
    found.feed(page)
    assert found.found is not None and ("hud" not in found.found) == ok


@pytest.mark.parametrize(
    ("css", "ok"),
    [
        ("#disclosure { position: fixed; opacity: 0.9 }", True),
        ("#disclosure { color: red } body.kiosk #disclosure { color: blue }", False),  # a later rule, on a switch
        (".hud-off #disclosure { top: 0 }", False),
        ("#disclosure { display:none }", False),
        ("#disclosure { display : none }", False),
        ("#disclosure { color: red } #disclosure { visibility:hidden }", False),
        ("#disclosure{opacity:0}", False),
        ("@media (max-width: 600px) { #disclosure { display: none } }", False),
        ("@supports (display: flex) { .hud-off #disclosure { top: 0 } }", False),
        ("@media (min-width: 1px) { #disclosure { position: fixed; opacity: 0.9 } }", True),
    ],
)
def test_the_label_check_reads_every_rule_and_spelling(css: str, ok: bool) -> None:
    rules = _label_rules(css)
    bad = [s for s, body in rules if _hides(body) or re.search(r"hud|kiosk", s, re.I)]
    assert rules and (not bad) == ok


def test_the_disclosure_label_never_depends_on_the_hud_or_kiosk_switches() -> None:
    """The viewer: the label is outside the status line, shown only by showDisclosure(), which reads neither switch.

    The viewer has no browser test in CI; this holds the shape that keeps the rule, and
    ``test_a_consented_face_is_labelled_with_the_hud_off`` checks it in a browser where one is installed.
    """
    page = (VIEWER / "index.html").read_text(encoding="utf-8")
    script = (VIEWER / "viewer.js").read_text(encoding="utf-8")
    ancestry = _Ancestry("disclosure")
    ancestry.feed(page)
    assert ancestry.found is not None and "hud" not in ancestry.found, "the label is inside the status line"
    show = _function(script, "showDisclosure")
    for switch in ("hud", "KIOSK", "params", "RECORD", "hidden = true"):
        assert switch not in show.replace("disclosureEl.hidden = !text", ""), f"showDisclosure reads {switch}"
    writes = re.findall(r"disclosureEl\.(?:hidden|textContent|style)\b[^=]*=", script)
    assert len(writes) == 2 and all(w in show for w in writes), "the label is set outside showDisclosure"
    styles = "".join(re.findall(r"<style[^>]*>(.*?)</style>", page, re.S))
    rules = _label_rules(styles)
    assert rules, "no rule places the label"
    for selector, body in rules:
        assert not _hides(body), f"{selector} hides the label"
        assert not re.search(r"hud|kiosk", selector, re.I), f"{selector} styles the label by a switch"


def _browser() -> bool:
    try:
        from playwright.sync_api import sync_playwright  # noqa: PLC0415

        with sync_playwright() as p:
            p.chromium.launch().close()
        return True
    except Exception:  # noqa: BLE001 - no playwright, or no browser for it
        return False


@pytest.mark.skipif(not _browser(), reason="needs playwright with a browser (the record extra)")
def test_a_consented_face_is_labelled_with_the_hud_off(tmp_path) -> None:
    from playwright.sync_api import sync_playwright  # noqa: PLC0415

    from tests.helpers import live_server  # noqa: PLC0415

    make_face(tmp_path, "agreed", provenance=CONSENTED)
    write_consent(tmp_path)
    with live_server(create_app(Settings(assets_dir=tmp_path))) as url, sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            for query in ("hud=0", "hud=0&kiosk=1"):
                page = browser.new_page()
                page.goto(f"{url}/?face=agreed&{query}")
                page.wait_for_function("window.__mask && window.__mask.disclosure", timeout=20_000)
                label = page.locator("#disclosure")
                assert label.is_visible() and "real person" in label.inner_text().lower(), query
                assert not page.locator("#hud").is_visible()
                page.close()
        finally:
            browser.close()


# ── Nothing in the engine reads the clock or an unseeded RNG ──────────────────────────────────────────────────


def _engine_modules() -> list[Path]:
    import demo.docs_page as docs  # noqa: PLC0415 - ENGINE: the code a take depends on, the drift check's list

    out: list[Path] = []
    for entry in docs.ENGINE:
        path = ROOT / entry
        out += sorted(path.rglob("*.py")) if path.is_dir() else [path]
    return out


CLOCKS = ("time", "datetime", "secrets", "uuid")
SEEDED = {"random.Random": ("x",), "numpy.random.default_rng": ("seed",)}  # each one's seed keyword


def _engine_problems(source: str, label: str = "<engine>") -> list[str]:
    """What in *source* could make a take differ: a clock, OS entropy, or an unseeded random number.

    Names are resolved through every import and alias first (``import random as rnd``, ``from numpy import random
    as npr``, ``from random import choice``), so a banned call cannot hide under another name.
    """
    tree = ast.parse(source, label)
    found: list[str] = []
    names: dict[str, str] = {}  # a local name -> what it is: "random", "numpy.random", "random.choice", ...

    def banned(full: str) -> bool:
        return full in ("random", "numpy.random", "os.urandom") or full.startswith(("random.", "numpy.random."))

    for node in ast.walk(tree):
        where = f"{label}:{getattr(node, 'lineno', 0)}"
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] in CLOCKS:
                    found.append(f"{where} imports {a.name}")
                names[a.asname or a.name.split(".")[0]] = a.name if a.asname else a.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] in CLOCKS:
                found.append(f"{where} imports {node.module}")
            for a in node.names:
                # ``from random import *`` binds nothing we can resolve: the import itself is the call site.
                if a.name == "*" and (banned(node.module) or node.module.split(".")[0] in CLOCKS):
                    found.append(f"{where} star-imports {node.module}")
                    continue
                names[a.asname or a.name] = f"{node.module}.{a.name}"

    def resolve(expr: ast.expr) -> str | None:
        if isinstance(expr, ast.Name):
            return names.get(expr.id)
        if isinstance(expr, ast.Attribute):
            owner = resolve(expr.value)
            return f"{owner}.{expr.attr}" if owner else None
        return None

    called = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        full = resolve(node.func)
        if full is None:
            continue
        called.add(id(node.func))
        if full in SEEDED:
            seed = node.args[0] if node.args else next((k.value for k in node.keywords if k.arg in SEEDED[full]), None)
            if seed is None or (isinstance(seed, ast.Constant) and seed.value is None):
                found.append(f"{label}:{node.lineno} calls {full} with no seed")
        elif banned(full):
            found.append(f"{label}:{node.lineno} calls {full}")
    # The module or a function of it taken without calling it (``rng = np.random``) is a call waiting elsewhere.
    parents = {id(c): p for p in ast.walk(tree) for c in ast.iter_child_nodes(p)}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Name | ast.Attribute) or isinstance(parents.get(id(node)), ast.Attribute):
            continue
        full = resolve(node)
        if full and id(node) not in called and full not in SEEDED and banned(full):
            found.append(f"{label}:{node.lineno} takes {full} without calling it")
    return found


@pytest.mark.parametrize(
    "source",
    [
        "from random import choice\nchoice([1])",
        "from numpy.random import rand\nrand()",
        "import random as rnd\nrnd.random()",
        "from numpy import random as npr\nnpr.normal()",
        "import numpy as np\nrng = np.random\nrng.rand()",
        "import numpy\nnumpy.random.rand()",
        "import random\nrandom.Random(None)",
        "import numpy as np\nnp.random.default_rng(None)",
        "import numpy as np\nnp.random.default_rng()",
        "from random import Random\nRandom()",
        "import os\nos.urandom(4)",
        "from time import monotonic",
        "import datetime as dt",
        "from random import *\nchoice([1])",
        "from numpy.random import *\nrand()",
    ],
)
def test_the_engine_check_sees_through_names_and_aliases(source: str) -> None:
    assert _engine_problems(source)


@pytest.mark.parametrize(
    "source",
    [
        "import random\nrng = random.Random(seed)\nrng.random()",
        "import random\nrandom.Random(x=seed)",
        "import numpy as np\nnp.random.default_rng(seed=s).normal()",
        "from random import Random\nRandom(7)",
        "import numpy as np\nnp.zeros(3)",
    ],
)
def test_the_engine_check_lets_a_seeded_generator_through(source: str) -> None:
    assert _engine_problems(source) == []


def test_the_engine_reads_no_clock_and_no_unseeded_random() -> None:
    """Takes are byte-identical for the same audio, driver and seed only if nothing the take depends on can differ.

    Banned in ``ENGINE``'s modules: importing time, datetime, secrets or uuid; anything of ``random`` but a seeded
    ``random.Random(seed)``; anything of ``numpy.random`` but ``default_rng(seed)``; ``os.urandom``. A seed of
    ``None`` is OS entropy, so it is no seed. The a2f driver's socket timeouts are asyncio's and touch no frame.
    """
    found = []
    for path in _engine_modules():
        found += _engine_problems(path.read_text(encoding="utf-8"), str(path.relative_to(ROOT)))
    assert not found, "\n".join(found)


# ── Only cleared material in the repo ─────────────────────────────────────────────────────────────────────────


def test_every_face_the_repo_ships_is_cleared() -> None:
    """The stock heads' locks (what ``fetch_stock_head`` writes) and the test fixtures' default provenance."""
    from tests.helpers import STOCK  # noqa: PLC0415

    locks = sorted((ROOT / "scripts" / "stock_heads").glob("*.lock.json"))
    assert locks
    for lock in locks:
        manifest = FaceManifest.model_validate(json.loads(lock.read_text(encoding="utf-8"))["manifest"])
        assert manifest.provenance.cleared, f"{lock.name}: {manifest.provenance.kind} is not cleared"
    assert STOCK.kind in CLEARED_KINDS


# ── Files Blender loads by path import nothing from the package ───────────────────────────────────────────────


@pytest.mark.parametrize("name", ["hair_colour.py", "build_head.py"])
def test_what_blender_loads_by_path_imports_nothing_from_the_package(name: str) -> None:
    """Blender's Python has numpy but not argus_mask: ``build_head.py`` runs there and loads ``hair_colour.py`` by path."""
    tree = ast.parse((PACKAGE / "editor" / name).read_text(encoding="utf-8"))
    imported = {
        (a.name if isinstance(node, ast.Import) else node.module or "").split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import | ast.ImportFrom) and not getattr(node, "level", 0)
        for a in (node.names if isinstance(node, ast.Import) else [node])
    }
    assert "argus_mask" not in imported and "argus_cortex" not in imported, imported
    assert not any(isinstance(n, ast.ImportFrom) and n.level for n in ast.walk(tree)), "a relative import"


# -- LAM and VHAP code is never copied into the repo (#136) ----------------------------------------------------------

# The headers that mark LAM's and VHAP's own files, as each source file carries them: Max-Planck's proprietary notice
# on FLAME's layer (in LAM and in VHAP), Toyota Motor Europe's on VHAP (CC BY-NC-SA), and LAM's own copyright line.
# A credit that names LAM (the demo's) is not a header and does not match.
COPIED_HEADERS = (
    "holder of all proprietary rights on this computer program",
    "Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual",
    "The Alibaba 3DAIGC Team Authors",
)
# LAM's and VHAP's packages: `lam/make_head.py` and `lam/overrides.py` import them inside LAM's container, where LAM
# is installed; nothing else may.
LAM_PACKAGES = {"lam", "vhap"}
RUNS_LAM = {"lam/make_head.py", "lam/overrides.py"}


def _tracked() -> list[str]:
    try:
        out = subprocess.run(  # noqa: S603 - fixed arguments
            ["git", "-C", str(ROOT), "ls-files", "-z"],  # noqa: S607
            capture_output=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout: the tracked files are what is checked")
    return [p for p in out.decode().split("\0") if p and (ROOT / p).is_file()]


def test_no_file_carries_a_lam_or_vhap_licence_header() -> None:
    this = Path(__file__).resolve().relative_to(ROOT).as_posix()  # it names the headers to find them
    found = []
    for rel in _tracked():
        if rel == this or (ROOT / rel).stat().st_size > 5_000_000:
            continue
        text = (ROOT / rel).read_bytes().decode("utf-8", errors="ignore")
        found += [f"{rel}: {header!r}" for header in COPIED_HEADERS if header in text]
    assert not found, "LAM's or VHAP's code is in the repo:\n" + "\n".join(found)


def test_no_lam_or_vhap_package_is_in_the_repo_or_imported_outside_the_head_builder() -> None:
    tracked = _tracked()
    assert not [p for p in tracked if "vhap" in Path(p).parts[:-1]], "a vhap/ directory"
    found = []
    for rel in tracked:
        if not rel.endswith(".py") or rel in RUNS_LAM:
            continue
        tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"), filename=rel)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level:
                names = [node.module or ""]
            else:
                continue
            found += [f"{rel}:{node.lineno} imports {n}" for n in names if n.split(".")[0] in LAM_PACKAGES]
    assert not found, "\n".join(found)
