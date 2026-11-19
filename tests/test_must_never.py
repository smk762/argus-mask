"""Checks for rules in CLAUDE.md's "Must never happen" table that no other test held (#130).

Each test names the rule it holds. They are written against the apps' route tables and the source, not one route
or one call, so a new route or a new engine module is caught too.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
from pathlib import Path

import pytest
from starlette.routing import Route, WebSocketRoute
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from argus_mask.config import EditorSettings, Settings
from argus_mask.editor.app import create_editor_app
from argus_mask.models import CLEARED_KINDS, FaceManifest, Provenance
from argus_mask.server.app import create_app
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


def _path(route: Route, face: str = "agreed") -> str:
    """The route's path with its parameters filled: a face id where it takes one, a placeholder elsewhere."""
    return re.sub(r"{(\w+)(:\w+)?}", lambda m: face if "face" in m.group(1) else "x", route.path)


# ── The consent record is never served, only verdicts ─────────────────────────────────────────────────────────


def test_no_route_serves_a_consent_record(apps) -> None:
    """Every GET route of both apps, asked about the consented face: none answers with the record's own fields.

    The subject and the release's hash are the face's provenance, served by design. What only the record holds
    (who collected it, the reference media's hashes, its terms) must never leave the machine.
    """
    record_only = ("B. Collector", "cd" * 32, "ef" * 32, "collected_by", "reference_images", "notice_days")
    asked = 0
    for app in apps:
        with TestClient(app) as client:
            for route in app.routes:
                if not isinstance(route, Route) or "GET" not in (route.methods or ()):
                    continue
                params = {"use": "video"} if route.path.endswith("/consent") else {}
                reply = client.get(_path(route), params=params)
                asked += 1
                leaked = [f for f in record_only if f in reply.text]
                assert not leaked, f"GET {route.path} served {leaked}"
    assert asked > 40  # the route tables were walked, not an empty list


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
            try:
                with client.websocket_connect(_path(route, "test-head"), headers=EVIL) as ws:
                    reply = ws.receive_json()
                    assert "cross-site" in reply.get("detail", ""), f"{route.path} let another site in: {reply}"
            except WebSocketDisconnect:
                pass  # refused before accepting: also fine


# ── The disclosure label shows whatever ?hud says ─────────────────────────────────────────────────────────────


def _function(source: str, name: str) -> str:
    start = source.index(f"function {name}(")
    depth, i = 0, source.index("{", start)
    for j in range(i, len(source)):
        depth += {"{": 1, "}": -1}.get(source[j], 0)
        if depth == 0:
            return source[start : j + 1]
    raise AssertionError(f"{name} has no end")


def test_the_disclosure_label_never_depends_on_the_hud_or_kiosk_switches() -> None:
    """The viewer: the label is outside the status line, shown only by showDisclosure(), which reads neither switch.

    The viewer has no browser test in CI; this holds the shape that keeps the rule, and
    ``test_a_consented_face_is_labelled_with_the_hud_off`` checks it in a browser where one is installed.
    """
    page = (VIEWER / "index.html").read_text(encoding="utf-8")
    script = (VIEWER / "viewer.js").read_text(encoding="utf-8")
    hud = re.search(r'<div id="hud"[^>]*>(.*?)</div>', page, re.S)
    assert '<div id="disclosure"' in page and (hud is None or "disclosure" not in hud.group(1))
    show = _function(script, "showDisclosure")
    for switch in ("hud", "KIOSK", "params", "RECORD", "hidden = true"):
        assert switch not in show.replace("disclosureEl.hidden = !text", ""), f"showDisclosure reads {switch}"
    writes = re.findall(r"disclosureEl\.(?:hidden|textContent|style)\b[^=]*=", script)
    assert len(writes) == 2 and all(w in show for w in writes), "the label is set outside showDisclosure"
    css = re.search(r"#disclosure\s*{[^}]*}", page)
    assert css and "display: none" not in css.group(0) and "hud" not in css.group(0)


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


def test_the_engine_reads_no_clock_and_no_unseeded_random() -> None:
    """Takes are byte-identical for the same audio, driver and seed only if nothing the take depends on can differ.

    Banned in ``ENGINE``'s modules: importing time, datetime, secrets or uuid; ``random``'s module functions
    (a seeded ``random.Random(seed)`` is fine); ``numpy.random`` apart from ``default_rng(seed)``; ``os.urandom``.
    The a2f driver's socket timeouts are asyncio's and touch no frame.
    """
    found = []
    for path in _engine_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        for node in ast.walk(tree):
            where = f"{path.relative_to(ROOT)}:{getattr(node, 'lineno', 0)}"
            if isinstance(node, ast.Import | ast.ImportFrom):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                found += [f"{where} imports {n}" for n in names if n.split(".")[0] in ("time", "datetime", "secrets")]
                found += [f"{where} imports {n}" for n in names if n == "uuid"]
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                owner = ast.unparse(node.func.value)
                name = node.func.attr
                if owner == "random" and not (name == "Random" and node.args):
                    found.append(f"{where} calls random.{name}")
                if owner in ("np.random", "numpy.random") and not (name == "default_rng" and node.args):
                    found.append(f"{where} calls {owner}.{name}")
                if owner == "os" and name == "urandom":
                    found.append(f"{where} calls os.urandom")
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
