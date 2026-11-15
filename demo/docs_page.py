"""The hosted demo's /docs/ (#123): argus-mask's README and docs/*.md, rendered to static pages at build time.

The docs are read from this checkout, so they are the docs of the commit being deployed (the deploy builds
``dev``'s HEAD). ``build(out, page_frame)`` writes ``docs/index.html`` (the README), ``docs/<name>/index.html`` per
doc, the images they show, and ``docs/sources.json``: the commit, the version at it, and a sha256 per file.

- **Opsec gate, fail closed.** Every imported doc and every published clip's ``clip.json`` is scanned for private
  or link-local addresses, home or server paths, email addresses and the terms in ``OPSEC_DENYLIST`` (a CI
  secret, never committed). A hit refuses the build and nothing is redacted: fix the doc.
- **Clip drift.** A clip records the commit that made it (``made_with.argus_mask``, ``git describe``). If the code
  that makes a take (``ENGINE``) has changed since, the clip no longer shows what HEAD would make, and the build
  refuses until ``demo/make_clips.py`` re-records it.
- **CSP-neutral.** Markdown is rendered with raw HTML off; no inline script or style (table alignment becomes a
  class); images are copied into the site, never hot-linked, and remote ones (badges) are dropped.

Needs markdown-it-py; the rest is the standard library.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import subprocess
import unicodedata
from pathlib import Path, PurePosixPath

from markdown_it import MarkdownIt
from markdown_it.token import Token

ROOT = Path(__file__).resolve().parent.parent
REPO = "smk762/argus-mask"
PYPI = "argus-mask"
# The allowlist: the public docs. Agent and planning files (CLAUDE.md, AGENTS.md, spec.md, TODO.md) never go here.
DOCS = ("README.md", *sorted(f"docs/{p.name}" for p in (ROOT / "docs").glob("*.md")))
# The code a take depends on: a change here since a clip was recorded means the clip is stale.
ENGINE = (
    "src/argus_mask/arkit.py",
    "src/argus_mask/audio.py",
    "src/argus_mask/drivers",
    "src/argus_mask/expressions.py",
    "src/argus_mask/idle.py",
    "src/argus_mask/models.py",
    "src/argus_mask/performance.py",
    "src/argus_mask/prosody.py",
    "src/argus_mask/speech.py",
    "src/argus_mask/take.py",
)
# Our own repo's files by absolute URL, at any branch: a doc is shown here, anything else at the imported commit.
BLOB = re.compile(rf"^https://github\.com/{re.escape(REPO)}/(?:blob|tree)/[^/]+/(.+)$")
RAW = re.compile(rf"^https://raw\.githubusercontent\.com/{re.escape(REPO)}/[^/]+/(.+)$")

OPSEC = (
    ("private IPv4 address", re.compile(r"\b(?:10\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])|192\.168)\.\d{1,3}\.\d{1,3}\b")),
    ("link-local IPv4 address", re.compile(r"\b169\.254\.\d{1,3}\.\d{1,3}\b")),
    (
        "link-local or unique-local IPv6 address",
        re.compile(r"\b(?:fe80|f[cd][0-9a-f]{2}):[0-9a-f:]*:[0-9a-f:]+\b", re.I),
    ),
    # Absolute paths only: ``docs/media/x.jpg`` is a path in the repo, not a server's ``/media``.
    ("home directory path", re.compile(r"(?<![\w./-])(?:/home/|/Users/|/root/|~/)[\w.-]+")),
    ("server path", re.compile(r"(?<![\w./-])/(?:srv|mnt|media|var/www)/[\w.-]+")),
    ("email address", re.compile(r"\b[\w.%+-]+@[\w-]+(?:\.[\w-]+)*\.[a-z]{2,}\b", re.I)),
)


class Refused(Exception):
    """The docs or the clips cannot be published as they are."""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---- the gate and the record -----------------------------------------------------------------------------------


def denylist(env: dict[str, str] | None = None) -> list[str]:
    """The private denylist: one term per line or comma, from ``OPSEC_DENYLIST``."""
    raw = (os.environ if env is None else env).get("OPSEC_DENYLIST", "")
    return [t.strip() for t in re.split(r"[\n,]", raw) if t.strip()]


def opsec_hits(text: str, deny: list[str] = ()) -> list[tuple[int, str, str]]:
    """Every hit in *text* as (line, kind, match). A denylisted term is never echoed, so a CI log can't leak it."""
    hits = []
    line = lambda i: text.count("\n", 0, i) + 1  # noqa: E731
    for kind, pattern in OPSEC:
        hits += [(line(m.start()), kind, m.group(0)) for m in pattern.finditer(text)]
    lower = text.lower()
    for term in deny:
        start = lower.find(term.lower())
        while start >= 0:
            hits.append((line(start), "denylisted term", "(withheld)"))
            start = lower.find(term.lower(), start + len(term))
    return sorted(hits)


def gate(label: str, text: str, deny: list[str]) -> None:
    if hits := opsec_hits(text, deny):
        lines = "\n".join(f"  {label}:{n}: {kind}: {match}" for n, kind, match in hits)
        raise Refused(f"opsec gate refused {label}; fix it at the source, nothing is redacted:\n{lines}")


def git(*args: str) -> str:
    return subprocess.run(  # noqa: S603 - fixed arguments
        ["git", "-C", str(ROOT), *args],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def version_from_describe(describe: str) -> str:
    """hatch-vcs's version for a commit, from ``git describe --long``: the tag, or the next patch's dev release."""
    found = re.fullmatch(r"v?(\d+)\.(\d+)\.(\d+)-(\d+)-g([0-9a-f]+)", describe.strip())
    if not found:
        raise Refused(f"cannot read a version from git describe: {describe!r}")
    major, minor, patch, distance, short = found.groups()
    if distance == "0":
        return f"{major}.{minor}.{patch}"
    return f"{major}.{minor}.{int(patch) + 1}.dev{distance}+g{short}"


def clip_commit(made_by: str) -> str:
    """The commit a clip's ``made_with.argus_mask`` names: ``git describe``'s ``-g<sha>``, or a bare sha. A clip made
    from uncommitted code (``-dirty``) names none that it matches."""
    if made_by.strip().endswith("-dirty"):
        raise Refused(f"a clip was made from uncommitted changes ({made_by!r}): re-record it from a clean checkout")
    found = re.search(r"(?:^|-g)([0-9a-f]{7,40})$", made_by.strip())
    if not found:
        raise Refused(f"a clip's made_with.argus_mask names no commit: {made_by!r}")
    return found.group(1)


def clip_drift(clips: list[dict]) -> dict[str, str]:
    """Each clip's recorded commit; refuses if the engine has changed since one was made."""
    stale, made = [], {}
    for clip in clips:
        commit = clip_commit(clip["made_with"]["argus_mask"])
        try:
            git("cat-file", "-e", f"{commit}^{{commit}}")
        except subprocess.CalledProcessError:
            raise Refused(
                f"clip {clip['id']!r} was made at {commit}, which this checkout doesn't have (a shallow clone? "
                "fetch the full history)"
            ) from None
        changed = git("diff", "--name-only", commit, "HEAD", "--", *ENGINE)
        if changed:
            stale.append(f"  {clip['id']} (made at {commit}): {', '.join(changed.splitlines())}")
        made[clip["id"]] = git("rev-parse", commit)
    if stale:
        raise Refused(
            "the code that makes a take has changed since these clips were recorded; re-record them with "
            "demo/make_clips.py and commit the result:\n" + "\n".join(stale)
        )
    return made


# ---- rendering ---------------------------------------------------------------------------------------------------


def slug(text: str) -> str:
    """GitHub's heading anchor: lower case, punctuation dropped, spaces to hyphens."""
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"[^\w\- ]", "", text).replace(" ", "-")


def page_path(doc: str) -> str:
    """The site path of an imported doc: the README is /docs/, docs/x.md is /docs/x/."""
    return "/docs/" if doc == "README.md" else f"/docs/{PurePosixPath(doc).stem}/"


def plain(token: Token) -> str:
    return "".join(c.content for c in token.children or () if c.type in ("text", "code_inline"))


class Doc:
    """One imported markdown file, rendered: its title, the headings on it, its HTML and the images it shows."""

    def __init__(self, path: str, text: str, commit: str):
        self.path = path
        self.images: dict[str, str] = {}  # site path -> repo path
        # HTML is parsed so that it can be dropped whole (the README's logo, comments), not shown escaped as text.
        md = MarkdownIt("commonmark", {"html": True}).enable(["table", "strikethrough"])
        tokens = [t for t in md.parse(text) if t.type != "html_block"]
        self.title = path
        self.headings: list[tuple[int, str, str]] = []
        seen: dict[str, int] = {}
        for i, tok in enumerate(tokens):
            if tok.type == "heading_open":
                text_ = plain(tokens[i + 1])
                base = slug(text_)
                n = seen.get(base, 0)
                seen[base] = n + 1
                tok.attrSet("id", base if n == 0 else f"{base}-{n}")
                level = int(tok.tag[1])
                if level == 1 and self.title == path:
                    self.title = text_
                elif level == 2:
                    self.headings.append((level, text_, tok.attrGet("id")))
            elif tok.type in ("th_open", "td_open") and (style := tok.attrGet("style")):
                tok.attrs.pop("style")  # an inline style attribute is blocked by style-src 'self'
                tok.attrSet("class", "align-" + style.split(":")[1].strip())
            elif tok.type == "inline":
                tok.children = self._inline(tok.children or [], commit)
        self.html = md.renderer.render(tokens, md.options, {})
        if self.title == path:  # no markdown title (the README's is a logo in HTML): the package's name
            self.title = PYPI
            self.html = f"<h1>{PYPI}</h1>\n{self.html}"

    def _inline(self, children: list[Token], commit: str) -> list[Token]:
        out: list[Token] = []
        external: list[bool] = []
        for tok in children:
            if tok.type == "html_inline":  # shown as text: a ``<placeholder>`` in prose parses as a tag
                tok.type = "text"
            if tok.type == "link_open":
                href, ext = self._href(tok.attrGet("href") or "", commit)
                tok.attrSet("href", href)
                if ext:
                    tok.attrSet("target", "_blank")
                    tok.attrSet("rel", "noopener noreferrer")
                external.append(ext)
            elif tok.type == "link_close" and external.pop():
                out.append(tok)
                cue = Token("html_inline", "", 0)
                cue.content = '<span class="sr-only"> (opens in new tab)</span>'
                out.append(cue)
                continue
            elif tok.type == "image":
                src = self._image(tok.attrGet("src") or "")
                if src is None:  # remote (a badge): dropped, never fetched
                    continue
                tok.attrSet("src", src)
                tok.attrSet("loading", "lazy")
            out.append(tok)
        return out

    def _repo_target(self, target: str) -> str | None:
        """A link or image *target* as a path in the repo, or None if it points elsewhere."""
        if found := RAW.match(target):
            return found.group(1)
        if re.match(r"^[a-z][a-z0-9+.-]*:|^//|^#", target, re.I):
            return None
        resolved = os.path.normpath(PurePosixPath(self.path).parent.joinpath(target).as_posix())
        return None if resolved.startswith("..") else resolved

    def _href(self, href: str, commit: str) -> tuple[str, bool]:
        if href.startswith("#"):
            return href, False
        target, _, anchor = href.partition("#")
        found = BLOB.match(target)
        path = found.group(1) if found else self._repo_target(target)
        if path is None:
            return href, bool(re.match(r"^https?://", href)) and not href.startswith("https://mask.dragonhound.dev")
        if path in DOCS:
            return page_path(path) + (f"#{anchor}" if anchor else ""), False
        kind = "tree" if (ROOT / path).is_dir() else "blob"
        return f"https://github.com/{REPO}/{kind}/{commit}/{path}" + (f"#{anchor}" if anchor else ""), True

    def _image(self, src: str) -> str | None:
        path = self._repo_target(src)
        if path is None:
            return None
        if not (ROOT / path).is_file():
            raise Refused(f"{self.path}: the image {src!r} is not in the repo")
        site = f"/docs/files/{path}"
        self.images[site] = path
        return site


def build(out: Path, frame, clips: list[dict], env: dict[str, str] | None = None) -> dict:
    """Write /docs/ into *out*. ``frame(title, body_html)`` wraps a page's body in the site's shell.

    Gates everything first, so a refusal writes nothing. Returns the sync record (also written as
    ``docs/sources.json``)."""
    deny = denylist(env)
    commit = git("rev-parse", "HEAD")
    described = git("describe", "--tags", "--long", "--match", "v[0-9]*", "HEAD")
    version = version_from_describe(described)
    release = described.rsplit("-", 2)[1] == "0"
    texts = {path: (ROOT / path).read_bytes() for path in DOCS}
    for path, data in texts.items():
        gate(path, data.decode("utf-8"), deny)
    for clip in clips:
        gate(f"demo/clips/{clip['id']}/clip.json", json.dumps(clip, indent=2), deny)
    made = clip_drift(clips)
    docs = [Doc(path, data.decode("utf-8"), commit) for path, data in texts.items()]

    short = commit[:7]
    pypi = f"https://pypi.org/project/{PYPI}/" + (f"{version}/" if release else "")
    header = (
        '<p class="docs-version">'
        f"<span>argus-mask {html.escape(version)} @ <code>{short}</code></span>"
        f' · <a href="{pypi}" target="_blank" rel="noopener noreferrer">PyPI<span class="sr-only">'
        " (opens in new tab)</span></a>"
        f' · <a href="https://github.com/{REPO}/tree/{commit}" target="_blank" rel="noopener noreferrer">GitHub'
        '<span class="sr-only"> (opens in new tab)</span></a></p>'
    )
    images: dict[str, str] = {}
    for doc in docs:
        nav = "".join(
            f'<li><a href="{page_path(d.path)}"'
            + (' aria-current="page"' if d is doc else "")
            + f">{html.escape(d.title if d.path != 'README.md' else 'Overview')}</a>"
            + (
                "<ul>" + "".join(f'<li><a href="#{a}">{html.escape(t)}</a></li>' for _, t, a in d.headings) + "</ul>"
                if d is doc and d.headings
                else ""
            )
            + "</li>"
            for d in docs
        )
        body = (
            '<div class="docs">'
            f'<nav class="docs-nav" aria-label="Docs pages"><ul>{nav}</ul></nav>'
            f'<article class="docs-body">{header}{doc.html}'
            f'<p class="docs-source">Source: <a href="https://github.com/{REPO}/blob/{commit}/{doc.path}" '
            f'target="_blank" rel="noopener noreferrer">{doc.path} at {short}<span class="sr-only"> (opens in new '
            "tab)</span></a></p></article></div>"
        )
        target = out / page_path(doc.path).strip("/") / "index.html"
        target.parent.mkdir(parents=True, exist_ok=True)
        title = "argus-mask docs" if doc.path == "README.md" else f"{doc.title} · argus-mask docs"
        target.write_text(frame(title, body), encoding="utf-8")
        images.update(doc.images)
    for site, path in images.items():
        dest = out / site.strip("/")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT / path).read_bytes())

    record = {
        "repo": REPO,
        "branch": "dev",
        "commit": commit,
        "version": version,
        "pypi": PYPI,
        "docs": {path: sha256(data) for path, data in texts.items()},
        "images": {path: sha256((ROOT / path).read_bytes()) for path in sorted(images.values())},
        "clips": {
            clip["id"]: {"made_with": clip["made_with"]["argus_mask"], "commit": made[clip["id"]]} for clip in clips
        },
    }
    (out / "docs" / "sources.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record
