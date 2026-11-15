"""Build the hosted demo (#118) into demo/out/, ready for Cloudflare Pages.

    python3 demo/build.py [--out demo/out] [--faces-from assets]

The page, its styles and player, the shared design layer (vendor/dh, with the strip pasted in), the package's viewer
unchanged under viewer/, the committed clips, and each clip's stock head: fetched from the stock-heads release and
checked against its lock (scripts/fetch_stock_head.py), or copied from --faces-from if a checked copy is there.
Only stock heads: a clip names a face by id and only scripts/stock_heads/ is looked in. Writes _headers with a CSP
that allows the import map by its hash. Standard library only.
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


def strip_html() -> str:
    """The strip's <nav>, as dh-strip.html has it (its leading comment dropped)."""
    text = (HERE / "vendor" / "dh" / "dh-strip.html").read_text(encoding="utf-8")
    start = text.index('<nav class="dh-strip"')
    return text[start:].strip()


def csp(page: str) -> str:
    """The page's Content-Security-Policy: everything from this origin, the import map by its hash."""
    maps = re.findall(r'<script type="importmap">(.*?)</script>', page, re.S)
    hashes = " ".join(f"'sha256-{base64.b64encode(hashlib.sha256(m.encode()).digest()).decode()}'" for m in maps)
    return "; ".join(
        [
            "default-src 'self'",
            f"script-src 'self' {hashes}",
            "style-src 'self'",
            "img-src 'self' data: blob:",  # the glTF loader decodes textures from blobs
            "media-src 'self'",
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
        for extra in target.glob("*.glb"):
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
    for name in ("demo.css", "player.js"):
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
    print(f"built {out} ({len(clips)} clips, {size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
