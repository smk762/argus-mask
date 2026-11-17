"""The fixtures build what they say they build."""

from __future__ import annotations

import json
from pathlib import Path

from argus_mask.faces import MANIFEST_NAME
from tests.helpers import make_face, make_splat_face


def test_provenance_none_builds_a_face_without_one(tmp_path: Path) -> None:
    for make in (make_face, make_splat_face):
        directory = make(tmp_path / make.__name__, provenance=None)
        assert "provenance" not in json.loads((directory / MANIFEST_NAME).read_text(encoding="utf-8"))
        assert "provenance" in json.loads((make(tmp_path / f"{make.__name__}-2") / MANIFEST_NAME).read_text())
