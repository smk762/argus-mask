"""Builder.rename: the export and the preview move together or not at all."""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from pathlib import Path

import pytest

from argus_mask.editor.builder import Builder
from argus_mask.editor.library import AssetLibrary
from argus_mask.editor.preset import Preset
from tests.test_editor import preset


@pytest.fixture
def builder(tmp_path: Path, monkeypatch) -> Builder:
    """Faces in both places, marked by a file holding their id; the id rewrite is a plain copy."""
    made = Builder(blender="blender", blender_user=None, library=AssetLibrary(None), assets_dir=tmp_path / "assets")
    for root in (made.assets_dir, made.preview_dir):
        (root / "my-face").mkdir(parents=True)
        (root / "my-face" / "id").write_text("my-face")

    def rewrite(root: Path, face_id: str, new_id: str, name: str | None) -> Path:
        work = Path(tempfile.mkdtemp(prefix=".rename-", dir=root))
        shutil.copytree(root / face_id, work, dirs_exist_ok=True)
        (work / "id").write_text(f"{new_id}:{name}")
        return work

    monkeypatch.setattr(made, "editable", lambda face_id, place: True)
    monkeypatch.setattr(made, "_rewrite", rewrite)
    return made


def fail_moving_into(monkeypatch, root: Path) -> None:
    """Moving a staged copy into *root* fails, as a full or read-only disk would."""
    real = Path.rename

    def rename(self: Path, target):
        if self.name.startswith(".rename-") and Path(target).parent == root:
            raise OSError("disk full")
        return real(self, target)

    monkeypatch.setattr(Path, "rename", rename)


def contents(root: Path) -> dict[str, str]:
    return {p.name: (p / "id").read_text() for p in root.iterdir() if p.is_dir() and p.name != ".preview"}


@pytest.mark.parametrize("new_id", ["renamed", "my-face"])
def test_a_move_that_fails_part_way_puts_back_what_was_moved(builder, monkeypatch, new_id) -> None:
    fail_moving_into(monkeypatch, builder.preview_dir)  # the export moves first, then the preview fails
    with pytest.raises(OSError, match="disk full"):
        asyncio.run(builder.rename("my-face", new_id=new_id, name="New"))
    for root in (builder.assets_dir, builder.preview_dir):
        assert contents(root) == {"my-face": "my-face"}, "both copies as they were, and nothing left over"


def test_a_rename_moves_both(builder) -> None:
    assert asyncio.run(builder.rename("my-face", new_id="renamed", name="New")) == "renamed"
    for root in (builder.assets_dir, builder.preview_dir):
        assert contents(root) == {"renamed": "renamed:New"}


def test_the_cache_key_changes_when_mpfb_or_its_assets_change(tmp_path: Path) -> None:
    """The key covers the add-on's files and the asset packs, so an in-place upgrade is not served from cache."""
    user = tmp_path / "user"
    manifest = user / "extensions" / "user_default" / "mpfb" / "blender_manifest.toml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text('version = "1.0.0"\n')
    index = tmp_path / "data" / "packs" / "core.json"
    index.parent.mkdir(parents=True)
    index.write_text("{}\n")
    assets = tmp_path / "assets"
    assets.mkdir()
    face = Preset.model_validate(preset())

    def key() -> str:
        return Builder(
            blender="blender", blender_user=user, library=AssetLibrary(tmp_path / "data"), assets_dir=assets
        ).key(face)

    def touch(path: Path, text: str) -> None:
        path.write_text(text)
        st = path.stat()
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))

    first = key()
    touch(manifest, 'version = "2.0.0"\n# upgraded\n')
    second = key()
    assert second != first
    touch(index, '{"pack": 1}\n')
    assert key() != second
