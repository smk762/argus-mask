"""bake/mesh.py reads a head's .glb with numpy. It is not part of the package, so it is loaded by path."""

from __future__ import annotations

import importlib.util
import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location("bake_mesh", Path(__file__).parent.parent / "bake" / "mesh.py")
assert _SPEC is not None and _SPEC.loader is not None
mesh = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = mesh  # its dataclass looks itself up there
_SPEC.loader.exec_module(mesh)

PNG_A, PNG_B = b"texture-a", b"texture-b"


def glb(doc: dict, binary: bytes | None, *, bin_kind: int = 0x004E4942) -> bytes:
    text = json.dumps(doc, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 4)
    chunks = [struct.pack("<II", len(text), 0x4E4F534A), text]
    if binary is not None:
        binary += b"\0" * (-len(binary) % 4)
        chunks += [struct.pack("<II", len(binary), bin_kind), binary]
    body = b"".join(chunks)
    return struct.pack("<III", 0x46546C67, 2, 12 + len(body)) + body


def head(parts: list[str] = ["a", "b"], *, mode: int | None = None) -> tuple[dict, bytes]:  # noqa: B006
    """One triangle per part, each with its own base colour texture."""
    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], "<f4").tobytes()
    uv = np.array([[0, 0], [1, 0], [0, 1]], "<f4").tobytes()
    idx = np.array([0, 1, 2, 0], "<u2").tobytes()  # padded to 4 bytes
    blob = pos + uv + idx + PNG_A + b"\0\0\0" + PNG_B + b"\0\0\0"
    views = [
        {"buffer": 0, "byteOffset": 0, "byteLength": 36},
        {"buffer": 0, "byteOffset": 36, "byteLength": 24},
        {"buffer": 0, "byteOffset": 60, "byteLength": 6},
        {"buffer": 0, "byteOffset": 68, "byteLength": len(PNG_A)},
        {"buffer": 0, "byteOffset": 80, "byteLength": len(PNG_B)},
    ]
    accessors = [
        {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
        {"bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC2"},
        {"bufferView": 2, "componentType": 5123, "count": 3, "type": "SCALAR"},
    ]
    prims = []
    for i in range(len(parts)):
        prim = {"attributes": {"POSITION": 0, "TEXCOORD_0": 1}, "indices": 2, "material": i % 2}
        if mode is not None:
            prim["mode"] = mode
        prims.append(prim)
    doc = {
        "asset": {"version": "2.0"},
        "nodes": [{"mesh": i, "name": name} for i, name in enumerate(parts)],
        "meshes": [{"name": f"m{i}", "primitives": [p]} for i, p in enumerate(prims)],
        "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": i}}} for i in range(2)],
        "textures": [{"source": 0}, {"source": 1}],
        "images": [{"bufferView": 3, "mimeType": "image/png"}, {"bufferView": 4, "mimeType": "image/png"}],
        "buffers": [{"byteLength": len(blob)}],
        "bufferViews": views,
        "accessors": accessors,
    }
    return doc, blob


def write(tmp_path: Path, data: bytes) -> Path:
    path = tmp_path / "head.glb"
    path.write_bytes(data)
    return path


def test_a_head_loads_its_parts_and_textures(tmp_path: Path) -> None:
    doc, blob = head()
    h = mesh.load(write(tmp_path, glb(doc, blob)), ["jawOpen"])
    assert h.names == ["a", "b"]
    assert h.tris.tolist() == [[0, 1, 2], [3, 4, 5]]
    assert h.textures == {"a": PNG_A, "b": PNG_B}


def test_a_glb_without_a_bin_chunk_reads_as_an_empty_buffer(tmp_path: Path) -> None:
    gltf, blob = mesh._glb(write(tmp_path, glb({"asset": {"version": "2.0"}}, None)))
    assert gltf["asset"]["version"] == "2.0"
    assert blob == b""


def test_a_second_chunk_that_is_not_bin_is_refused(tmp_path: Path) -> None:
    doc, blob = head()
    with pytest.raises(ValueError, match="not BIN"):
        mesh._glb(write(tmp_path, glb(doc, blob, bin_kind=0x12345678)))


def test_a_truncated_bin_chunk_is_refused(tmp_path: Path) -> None:
    doc, blob = head()
    with pytest.raises(ValueError, match="truncated"):
        mesh._glb(write(tmp_path, glb(doc, blob)[:-8]))


@pytest.mark.parametrize("mode", [5, 6])
def test_a_strip_or_fan_is_refused_not_read_as_triangles(tmp_path: Path, mode: int) -> None:
    doc, blob = head(["a"], mode=mode)
    with pytest.raises(ValueError, match="mode"):
        mesh.load(write(tmp_path, glb(doc, blob)), ["jawOpen"])


def test_explicit_triangles_mode_loads(tmp_path: Path) -> None:
    doc, blob = head(["a"], mode=4)
    assert mesh.load(write(tmp_path, glb(doc, blob)), ["jawOpen"]).tris.tolist() == [[0, 1, 2]]


def test_parts_with_the_same_name_are_refused(tmp_path: Path) -> None:
    doc, blob = head(["skin", "skin"])
    with pytest.raises(ValueError, match="repeat"):
        mesh.load(write(tmp_path, glb(doc, blob)), ["jawOpen"])
