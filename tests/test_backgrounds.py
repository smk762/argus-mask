"""Plates: making one through darkroom, checking one, and the plates the viewer ships (#55)."""

from __future__ import annotations

import hashlib
import json
import struct
import zlib
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import ValidationError

from argus_mask import backgrounds
from argus_mask.backgrounds import Darkroom, PlateError, PlateRequest, add_depth, check_plate, make_plate
from argus_mask.models import PlateManifest
from tests.helpers import live_server

SCENES = Path(backgrounds.__file__).parent / "viewer" / "scenes"


def png(width: int = 16, height: int = 9, grey: bool = False, shade: int = 90) -> bytes:
    """A plain PNG, built with the standard library."""
    channels, colour_type = (1, 0) if grey else (3, 2)
    row = b"\x00" + bytes([shade]) * (width * channels)
    body = zlib.compress(row * height)

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, colour_type, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", body) + chunk(b"IEND", b"")


@contextmanager
def fake_darkroom(
    *,
    outcome: str = "passed",
    busy: int = 0,
    image: bytes | None = None,
    image_sha: str | None = None,
    depth: bytes | None = None,
    depth_sha: str | None = None,
    expand: bool = False,
) -> Iterator[tuple[str, list[str]]]:
    """argus-darkroom's generation, image and depth routes, as mask uses them. Yields (url, calls)."""
    picture = image if image is not None else png()
    depth_png = depth if depth is not None else png(grey=True, shade=200)
    calls: list[str] = []
    state = {"busy": busy, "record": None}
    app = FastAPI()

    @app.post("/generate/stream")
    async def generate(request: Request):
        calls.append("generate")
        if state["busy"]:
            state["busy"] -= 1
            return JSONResponse({"detail": "a generation is already running"}, status_code=409)
        body = await request.json()
        record = {
            "id": "rec1",
            "darkroom_version": "0.3.0",
            "outcome": outcome,
            "outcome_reason": None if outcome == "passed" else "a reason",
            "request": {
                "final_prompt": body["user_prompt"] + (", and more" if expand else ""),
                "expanded_prompt": "more" if expand else None,
                "negative_prompt": body["negative_prompt"],
                "seed": body["seed"],
                "checkpoint": {"name": body["checkpoint"]},
                "sampling": body["sampling"],
                "workflow": "flux",
            },
            "guard": {"enforcing_policy": body["enforcing_policy"]},
            "image": {"sha256": image_sha or hashlib.sha256(picture).hexdigest()},
        }
        state["record"] = record

        def events():
            yield 'event: progress\ndata: {"step": 1, "total": 4}\n\n'
            yield f"event: complete\ndata: {json.dumps({'record': record})}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.get("/image/{rid}")
    def get_image(rid: str):
        calls.append(f"image {rid}")
        return Response(picture, media_type="image/png")

    @app.post("/depth/{rid}")
    def make_depth(rid: str):
        calls.append(f"make_depth {rid}")
        return {
            "model": "depth-anything-v2-small",
            "model_license": "Apache-2.0",
            "sha256": depth_sha or hashlib.sha256(depth_png).hexdigest(),
            "workflow": "depth-anything-v2-small",
        }

    @app.get("/depth/{rid}")
    def get_depth(rid: str):
        calls.append(f"depth {rid}")
        return Response(depth_png, media_type="image/png")

    with live_server(app) as url:
        yield url, calls


def request(**over) -> PlateRequest:
    return PlateRequest(prompt="a quiet conservatory", seed=7, **over)


def make(url: str, directory: Path, **over) -> PlateManifest:
    return make_plate(
        Darkroom(url, sleep=lambda _s: None), directory, over.pop("req", request()), title="A garden", **over
    )


def test_a_plate_is_made_from_a_darkroom_record(tmp_path: Path) -> None:
    with fake_darkroom() as (url, calls):
        manifest = make(url, tmp_path / "garden", tint="#f3e6c8")
    assert calls == ["generate", "image rec1", "make_depth rec1", "depth rec1"]
    assert sorted(p.name for p in (tmp_path / "garden").iterdir()) == ["background.json", "depth.png", "plate.png"]
    assert manifest == check_plate(tmp_path / "garden")
    assert (manifest.name, manifest.title, manifest.tint) == ("garden", "A garden", "#f3e6c8")
    assert (manifest.plate.width, manifest.plate.height) == (16, 9) and manifest.depth is not None
    p = manifest.provenance
    assert p.kind == "synthetic" and p.cleared and p.license == "Apache-2.0"
    assert p.source == "argus-darkroom record rec1"
    assert (
        p.generator is not None
        and p.generator.seed == 7
        and p.generator.model.endswith("flux1-schnell-fp8-e4m3fn.safetensors")
    )
    d = manifest.darkroom
    assert d is not None and d.record_id == "rec1" and d.enforcing_policy == "default" and d.steps == 4
    assert d.prompt == "a quiet conservatory, no people, no text, no logos"
    assert d.depth is not None and d.depth.sha256 == manifest.depth.sha256


def test_the_request_asks_for_a_safe_plate_with_no_people() -> None:
    body = request().body(7)
    assert body["enforcing_policy"] == "default" and body["expand"] is False and body["loras"] == []
    assert body["user_prompt"].endswith("no people, no text, no logos")
    assert request(as_is=True).body(7)["user_prompt"] == "a quiet conservatory"


@pytest.mark.parametrize("outcome", ["review", "deleted", "omitted", "refused", "failed"])
def test_only_a_released_image_becomes_a_plate(tmp_path: Path, outcome: str) -> None:
    with fake_darkroom(outcome=outcome) as (url, calls), pytest.raises(PlateError, match=outcome):
        make(url, tmp_path / "garden")
    assert calls == ["generate"] and not (tmp_path / "garden").exists()


@pytest.mark.parametrize(
    ("over", "match"),
    [
        ({"image_sha": "0" * 64}, "image does not match"),
        ({"depth_sha": "0" * 64}, "depth map does not match"),
        ({"expand": True}, "expanded"),
    ],
)
def test_a_record_that_does_not_hold_up_writes_nothing(tmp_path: Path, over: dict, match: str) -> None:
    with fake_darkroom(**over) as (url, _calls), pytest.raises(PlateError, match=match):
        make(url, tmp_path / "garden")
    assert not (tmp_path / "garden").exists()


def test_an_unknown_licence_is_refused_before_darkroom_is_asked(tmp_path: Path) -> None:
    with fake_darkroom() as (url, calls):
        with pytest.raises(PlateError, match="--license"):
            make(url, tmp_path / "garden", req=request(checkpoint="sdxl/someone.safetensors"))
        assert calls == []
        stated = make(url, tmp_path / "garden", req=request(checkpoint="sdxl/someone.safetensors"), license="CC0-1.0")
    assert stated.provenance.license == "CC0-1.0"


def test_a_plate_is_never_written_over_another(tmp_path: Path) -> None:
    (tmp_path / "garden").mkdir()
    (tmp_path / "garden" / "plate.png").write_bytes(b"mine")
    with fake_darkroom() as (url, calls), pytest.raises(PlateError, match="not empty"):
        make(url, tmp_path / "garden")
    assert calls == [] and (tmp_path / "garden" / "plate.png").read_bytes() == b"mine"


def test_a_busy_darkroom_is_waited_for(tmp_path: Path) -> None:
    with fake_darkroom(busy=2) as (url, calls):
        make(url, tmp_path / "garden")
    assert calls[:3] == ["generate"] * 3


def test_depth_can_be_added_later(tmp_path: Path) -> None:
    with fake_darkroom() as (url, calls):
        first = make(url, tmp_path / "garden", depth=False)
        assert first.depth is None and calls == ["generate", "image rec1"]
        added = add_depth(Darkroom(url), tmp_path / "garden")
        assert calls[2:] == ["make_depth rec1", "depth rec1"]
        with pytest.raises(PlateError, match="already"):
            add_depth(Darkroom(url), tmp_path / "garden")
    assert added.depth is not None and added.darkroom is not None and added.darkroom.depth is not None
    assert added.plate == first.plate and added.provenance == first.provenance


def test_an_unreachable_darkroom_says_where_it_looked(tmp_path: Path) -> None:
    with pytest.raises(PlateError, match="unreachable at http://127.0.0.1:9"):
        make("http://127.0.0.1:9", tmp_path / "garden")


# ---- checking a plate ----------------------------------------------------------------------------


@pytest.fixture
def plate(tmp_path: Path) -> Path:
    with fake_darkroom() as (url, _calls):
        make(url, tmp_path / "garden")
    return tmp_path / "garden"


def test_a_changed_image_is_caught(plate: Path) -> None:
    (plate / "plate.png").write_bytes(png(shade=10))
    with pytest.raises(PlateError, match="plate: plate.png does not match its sha256"):
        check_plate(plate)


def test_a_link_is_not_a_plate_file(plate: Path, tmp_path: Path) -> None:
    (tmp_path / "elsewhere.png").write_bytes((plate / "depth.png").read_bytes())
    (plate / "depth.png").unlink()
    (plate / "depth.png").symlink_to(tmp_path / "elsewhere.png")
    with pytest.raises(PlateError, match="depth.png is missing or not a plain file"):
        check_plate(plate)


def test_a_plate_lives_under_its_own_name(plate: Path) -> None:
    moved = plate.rename(plate.parent / "orchard")
    with pytest.raises(PlateError, match="named 'garden' but lives in 'orchard'"):
        check_plate(moved)


def test_an_oversized_image_is_caught(plate: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backgrounds, "MAX_IMAGE_BYTES", 10)
    with pytest.raises(PlateError, match="over"):
        check_plate(plate)


def _manifest(plate: Path) -> dict:
    return json.loads((plate / "background.json").read_text())


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (lambda m: m.pop("provenance"), "provenance"),
        (lambda m: m["provenance"].update(kind="curated"), "curated"),
        (
            lambda m: (
                m.pop("darkroom"),
                m["provenance"].pop("generator"),
                m["provenance"].update(kind="consented", subject="x", release_sha256="0" * 64),
            ),
            "shows no one",
        ),
        (lambda m: m["depth"].update(width=8), "depth is 8x9"),
        (lambda m: m["depth"].update(file="plate.png"), "its own file"),
        (lambda m: m["plate"].update(file="../plate.png"), "pattern"),
        (lambda m: m["darkroom"].update(seed=8), "checkpoint and seed"),
        (lambda m: m["darkroom"]["depth"].update(sha256="0" * 64), "not the one darkroom made"),
    ],
)
def test_the_manifest_refuses_what_does_not_hold_up(plate: Path, change, match: str) -> None:  # noqa: ANN001
    manifest = _manifest(plate)
    change(manifest)
    with pytest.raises(ValidationError, match=match):
        PlateManifest.model_validate(manifest)


# ---- the plates the viewer ships -----------------------------------------------------------------


def built_in_plates() -> list[Path]:
    return sorted(p for p in SCENES.iterdir() if (p / "background.json").is_file())


def test_the_viewer_ships_garden() -> None:
    assert "garden" in [p.name for p in built_in_plates()]


@pytest.mark.parametrize("directory", built_in_plates(), ids=lambda p: p.name)
def test_a_shipped_plate_holds_up_and_is_cleared(directory: Path) -> None:
    manifest = check_plate(directory)
    assert manifest.provenance.cleared, "only cleared material ships"
    assert manifest.darkroom is not None, "a shipped plate names the darkroom record it came from"
    assert sorted(p.name for p in directory.iterdir()) == sorted(
        ["background.json"]
        + [i.file for i in (manifest.plate, manifest.depth, manifest.sway_mask, manifest.light_mask) if i]
    ), "nothing but the plate's own files"
