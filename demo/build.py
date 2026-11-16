"""Build the hosted demo (#118) into demo/out/, ready for Cloudflare Pages.

    python3 demo/build.py [--out demo/out] [--faces-from assets]

The page, its styles and player, the shared design layer (vendor/dh, with the strip pasted in), the package's viewer
unchanged under viewer/, the committed clips, and each clip's stock head: fetched from the stock-heads release and
checked against its lock (scripts/fetch_stock_head.py), or copied from --faces-from if a checked copy is there.
Only stock heads: a clip names a face by id and only scripts/stock_heads/ is looked in. Writes _headers with a CSP
that allows the import map by its hash. /docs/ is the README and docs/*.md rendered at this commit (docs_page.py,
which needs markdown-it-py), and 404.html makes Pages answer unknown paths with a real 404, not the demo page.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
VIEWER = ROOT / "src" / "argus_mask" / "viewer"
LOCKS = ROOT / "scripts" / "stock_heads"
MARKER = "<!-- dh-strip: pasted from vendor/dh/dh-strip.html by build.py -->"
DH_FILES = ("dh-tokens.css", "dh-strip.css", "dh-badge.css", "dh-theme.js")

sys.path.insert(0, str(HERE))
import docs_page  # noqa: E402 - beside this script, found once HERE is on the path


def strip_html() -> str:
    """The strip's <nav>, as dh-strip.html has it (its leading comment dropped)."""
    text = (HERE / "vendor" / "dh" / "dh-strip.html").read_text(encoding="utf-8")
    start = text.index('<nav class="dh-strip"')
    return text[start:].strip()


def shell(title: str, body: str) -> str:
    """A page of the site other than the demo (/docs/, 404.html): the strip, the header and the footer, with
    root-relative paths so it works at any depth. No inline script or style."""
    esc = title.replace("&", "&amp;").replace("<", "&lt;")
    return f"""<!doctype html>
<html lang="en" data-dh-site="mask">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>{esc}</title>
<script src="/vendor/dh/dh-theme.js"></script>
<link rel="stylesheet" href="/vendor/dh/dh-tokens.css">
<link rel="stylesheet" href="/vendor/dh/dh-strip.css">
<link rel="stylesheet" href="/vendor/dh/dh-badge.css">
<link rel="stylesheet" href="/demo.css">
<link rel="stylesheet" href="/docs.css">
</head>
<body>
<a class="skip" href="#main">Skip to the content</a>
{strip_html()}
<header class="top">
  <div>
    <p class="site"><a href="/">Argus Mask</a></p>
  </div>
  <div class="tools">
    <nav class="site-nav" aria-label="Site"><a href="/">Demo</a><a href="/docs/">Docs</a></nav>
    <button type="button" class="theme" data-dh-theme-toggle aria-label="Dark theme" aria-pressed="false">
      <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>
    </button>
  </div>
</header>
<main id="main" class="page">
{body}
</main>
<footer class="foot">
  <p class="links">
    <a href="https://github.com/smk762/argus-mask">argus-mask on GitHub</a>
    <span aria-hidden="true">·</span>
    <a href="https://pypi.org/project/argus-mask/">PyPI</a>
    <span aria-hidden="true">·</span>
    <a href="https://docs.dragonhound.dev/">Argus docs</a>
  </p>
</footer>
</body>
</html>
"""


NOT_FOUND = """<div class="missing">
<h1>Not found</h1>
<p>There is no page here. Try <a href="/">the demo</a> or <a href="/docs/">the docs</a>.</p>
</div>"""


def csp(page: str) -> str:
    """The page's Content-Security-Policy: everything from this origin, the import map by its hash."""
    maps = re.findall(r'<script type="importmap">(.*?)</script>', page, re.S)
    hashes = " ".join(f"'sha256-{base64.b64encode(hashlib.sha256(m.encode()).digest()).decode()}'" for m in maps)
    return "; ".join(
        [
            "default-src 'self'",
            # 'wasm-unsafe-eval' lets the splat renderer's depth-sort worker compile its WebAssembly (#126); it allows
            # no eval() and no new Function().
            f"script-src 'self' 'wasm-unsafe-eval' {hashes}",
            "style-src 'self'",
            "img-src 'self' data: blob:",  # the glTF loader decodes textures from blobs
            "media-src 'self' blob:",  # the player plays each clip's audio from a blob, so it can seek
            "connect-src 'self' data: blob:",
            "worker-src 'self' blob:",
            "object-src 'none'",
            "base-uri 'none'",
            "frame-ancestors 'none'",
        ]
    )


def faces_for(clips: list[dict], out: Path, local: Path | None) -> None:
    for face in sorted({c["face"] for c in clips}):
        lock = LOCKS / f"{face}.lock.json"
        if not lock.is_file():
            sys.exit(f"clip face {face!r} is not a stock head (no {lock.relative_to(ROOT)})")
        manifest = json.loads(lock.read_text(encoding="utf-8"))["manifest"]
        framings = manifest.get("framings") or {}
        used = {c["framing"] for c in clips if c["face"] == face}
        if missing := sorted(used - {"head", *framings}):
            sys.exit(f"{face} has no {', '.join(missing)} framing")
        # Only the models a clip shows: each is about 10 MB.
        files = [manifest["model"] if f == "head" else framings[f]["model"] for f in sorted(used)]
        target = out / "faces" / face
        copied = False
        if local and (local / face).is_dir():
            source = local / face
            ok = all(
                (source / f["file"]).is_file()
                and hashlib.sha256((source / f["file"]).read_bytes()).hexdigest() == f["sha256"]
                for f in files
            )
            if ok:
                target.mkdir(parents=True, exist_ok=True)
                for f in files:
                    shutil.copyfile(source / f["file"], target / f["file"])
                copied = True
        if not copied:
            subprocess.run(  # noqa: S603 - our own script, fixed arguments
                [
                    sys.executable,
                    str(ROOT / "scripts" / "fetch_stock_head.py"),
                    "--id",
                    face,
                    "--assets-dir",
                    str(out / "faces"),
                ],
                check=True,
            )
        for extra in [*target.glob("*.glb"), *target.glob("*.zip")]:
            if extra.name not in {f["file"] for f in files}:
                extra.unlink()
        (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=HERE / "out")
    parser.add_argument("--faces-from", type=Path, help="an assets root with the stock heads already in it")
    args = parser.parse_args()
    out: Path = args.out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    page = (HERE / "index.html").read_text(encoding="utf-8")
    if MARKER not in page:
        sys.exit("index.html has lost the strip marker")
    page = page.replace(MARKER, strip_html())
    (out / "index.html").write_text(page, encoding="utf-8")
    for name in ("demo.css", "docs.css", "player.js"):
        shutil.copyfile(HERE / name, out / name)
    (out / "vendor" / "dh").mkdir(parents=True)
    for name in DH_FILES:
        shutil.copyfile(HERE / "vendor" / "dh" / name, out / "vendor" / "dh" / name)
    shutil.copytree(VIEWER, out / "viewer", ignore=shutil.ignore_patterns("index.html", "__pycache__"))

    index = json.loads((HERE / "clips" / "index.json").read_text(encoding="utf-8"))
    clips = []
    for clip_id in index:
        source = HERE / "clips" / clip_id
        clips.append(json.loads((source / "clip.json").read_text(encoding="utf-8")))
        shutil.copytree(source, out / "clips" / clip_id)
    shutil.copyfile(HERE / "clips" / "index.json", out / "clips" / "index.json")
    try:
        docs = docs_page.build(out, shell, clips)
    except docs_page.Refused as exc:
        sys.exit(str(exc))
    (out / "404.html").write_text(shell("Not found · Argus Mask", NOT_FOUND), encoding="utf-8")
    faces_for(clips, out, args.faces_from)

    (out / "_headers").write_text(
        "/*\n"
        f"  Content-Security-Policy: {csp(page)}\n"
        "  X-Content-Type-Options: nosniff\n"
        "  Referrer-Policy: strict-origin-when-cross-origin\n"
        "  Permissions-Policy: camera=(), microphone=(), geolocation=()\n"
        "/faces/*\n"
        "  Cache-Control: public, max-age=86400\n",
        encoding="utf-8",
    )
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"built {out} ({len(clips)} clips, docs at {docs['commit'][:7]} ({docs['version']}), {size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
