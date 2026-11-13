from __future__ import annotations

import json

import pytest

from argus_mask.arkit import ARKIT_52
from argus_mask.faces import MANIFEST_NAME, FaceError, FaceStore, inspect_glb, read_glb
from argus_mask.models import Provenance
from tests.helpers import make_face, make_glb


def test_a_model_with_all_52_shapes(tmp_path) -> None:
    path = tmp_path / "head.glb"
    path.write_bytes(make_glb())
    report = inspect_glb(path)
    assert report.present == list(ARKIT_52) and not report.missing and not report.extra
    assert report.triangles == 1 and report.warnings == []


def test_missing_and_extra_shapes_are_reported_not_refused(tmp_path) -> None:
    path = tmp_path / "head.glb"
    path.write_bytes(make_glb(["jawOpen", "eyeBlinkLeft", "myCustomShape"], extra_static_mesh=True))
    report = inspect_glb(path)
    assert report.present == ["eyeBlinkLeft", "jawOpen"]  # in ARKit order
    assert len(report.missing) == 50 and report.extra == ["myCustomShape"]
    assert report.static_meshes == ["hair"]
    assert any("50 of the 52" in w for w in report.warnings) and any("hair" in w for w in report.warnings)
    make_face(tmp_path / "assets", "partial", targets=["jawOpen"])
    manifest, _ = FaceStore(tmp_path / "assets").load("partial")
    assert manifest.blendshapes == ["jawOpen"]


def test_not_a_glb_is_an_error(tmp_path) -> None:
    (tmp_path / "x.glb").write_bytes(b"this is not a model, it is a sentence")
    with pytest.raises(FaceError, match="not a .glb"):
        read_glb(tmp_path / "x.glb")


def test_a_face_without_provenance_is_refused(tmp_path) -> None:
    make_face(tmp_path, "good")
    make_face(tmp_path, "anonymous", provenance=None)
    accepted, refused = FaceStore(tmp_path).scan()
    assert [m.id for m in accepted] == ["good"]
    assert refused == {"anonymous": "the manifest has no provenance"}


def test_a_curated_face_is_refused(tmp_path) -> None:
    make_face(tmp_path, "someone", provenance=Provenance(kind="curated", source="a curator export", license="unknown"))
    with pytest.raises(FaceError, match="curated faces are not accepted yet"):
        FaceStore(tmp_path).load("someone")


RELEASE = "a" * 64


def test_a_personal_face_needs_only_its_source_and_is_never_cleared(tmp_path) -> None:
    make_face(
        tmp_path, "cartoon", provenance=Provenance(kind="personal", source="a cartoon character, for my own screens")
    )
    manifest, _ = FaceStore(tmp_path).load("cartoon")
    assert manifest.provenance.kind == "personal" and manifest.provenance.cleared is False
    assert json.loads(manifest.model_dump_json())["provenance"]["cleared"] is False
    stock, _ = FaceStore(tmp_path).load(make_face(tmp_path, "stock-head").name)
    assert stock.provenance.cleared is True


@pytest.mark.parametrize(
    ("fields", "why"),
    [
        ({"kind": "stock", "source": "a pack"}, "a stock face states its licence"),
        ({"kind": "synthetic", "source": "SDXL", "license": "  "}, "a synthetic face states its licence"),
        ({"kind": "consented", "source": "a capture", "license": "release", "subject": "A. Person"}, "sha256"),
        ({"kind": "consented", "source": "a capture", "license": "release", "release_sha256": RELEASE}, "subject"),
    ],
)
def test_a_cleared_kind_proves_what_it_claims(fields, why) -> None:
    with pytest.raises(ValueError, match=why):
        Provenance(**fields)


def test_a_provenance_refuses_what_it_does_not_know_and_takes_licence_for_license() -> None:
    with pytest.raises(ValueError, match="liscense"):
        Provenance(kind="personal", source="x", liscense="CC0-1.0")
    assert Provenance(kind="stock", source="a pack", licence="CC0").license == "CC0"
    with pytest.raises(ValueError, match="disagree"):
        Provenance(kind="stock", source="a pack", licence="CC0", license="CC-BY")
    # A dumped record carries the derived ``cleared``; it reads back, recomputed rather than believed.
    dumped = Provenance(kind="personal", source="a cartoon").model_dump()
    assert dumped["cleared"] is False
    assert Provenance.model_validate({**dumped, "cleared": True}).cleared is False


@pytest.mark.parametrize("source", ["", "   ", "\t\n"])
def test_a_provenance_names_a_source(source) -> None:
    with pytest.raises(ValueError, match="source"):
        Provenance(kind="personal", source=source)


def test_a_consented_face_is_served_and_cleared(tmp_path) -> None:
    provenance = Provenance(
        kind="consented", source="a capture", license="per the release", subject="A. Person", release_sha256=RELEASE
    )
    make_face(tmp_path, "agreed", provenance=provenance)
    manifest, _ = FaceStore(tmp_path).load("agreed")
    assert manifest.provenance.cleared is True and manifest.provenance.subject == "A. Person"


def test_a_manifest_must_match_its_directory_and_its_model(tmp_path) -> None:
    make_face(tmp_path, "one").rename(tmp_path / "two")
    with pytest.raises(FaceError, match="the directory is 'two'"):
        FaceStore(tmp_path).load("two")
    directory = make_face(tmp_path, "swapped")
    (directory / "head.glb").write_bytes(make_glb(["jawOpen"]))
    with pytest.raises(FaceError, match="sha256 does not match"):
        FaceStore(tmp_path).load("swapped")


def test_a_model_outside_the_face_directory_is_refused(tmp_path) -> None:
    directory = make_face(tmp_path / "assets", "linked")
    outside = tmp_path / "elsewhere.glb"
    (directory / "head.glb").rename(outside)
    (directory / "head.glb").symlink_to(outside)
    with pytest.raises(FaceError, match="not in the face's directory"):
        FaceStore(tmp_path / "assets").load("linked")


@pytest.mark.parametrize("face_id", ["../etc", "UPPER", "", "a/b", ".hidden"])
def test_a_face_id_is_a_plain_name(tmp_path, face_id: str) -> None:
    with pytest.raises(FaceError, match="not a face id"):
        FaceStore(tmp_path).load(face_id)


def test_a_manifest_from_a_newer_major_is_refused(tmp_path) -> None:
    directory = make_face(tmp_path, "future")
    data = json.loads((directory / MANIFEST_NAME).read_text())
    data["mask_version"] = "2.0"
    (directory / MANIFEST_NAME).write_text(json.dumps(data))
    assert "future" in FaceStore(tmp_path).scan()[1]
