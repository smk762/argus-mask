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
    depth_busy: int = 0,
    busy_detail: str = "a generation is already running",
    checkpoint: dict | None = None,
    negative: str | None = None,
) -> Iterator[tuple[str, list[str]]]:
    """argus-darkroom's generation, image and depth routes, as mask uses them. Yields (url, calls)."""
    picture = image if image is not None else png()
    depth_png = depth if depth is not None else png(grey=True, shade=200)
    calls: list[str] = []
    state = {"busy": busy, "depth_busy": depth_busy, "record": None}
    app = FastAPI()

    @app.post("/generate/stream")
    async def generate(request: Request):
        calls.append("generate")
        if state["busy"]:
            state["busy"] -= 1
            return JSONResponse({"detail": busy_detail}, status_code=409)
        body = await request.json()
        record = {
            "id": "rec1",
            "darkroom_version": "0.3.0",
            "outcome": outcome,
            "outcome_reason": None if outcome == "passed" else "a reason",
            "request": {
                "final_prompt": body["user_prompt"] + (", and more" if expand else ""),
                "expanded_prompt": "more" if expand else None,
                "negative_prompt": body["negative_prompt"] if negative is None else negative,
                "seed": body["seed"],
                "checkpoint": checkpoint if checkpoint is not None else {"name": body["checkpoint"]},
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
        if state["depth_busy"]:
            state["depth_busy"] -= 1
            return JSONResponse({"detail": busy_detail}, status_code=409)
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
        ({"checkpoint": {"name": "flux/flux1-dev.safetensors"}}, "checkpoint 'flux/flux1-dev"),
        ({"checkpoint": {}}, "checkpoint None"),
        ({"negative": "people"}, "different negative prompt"),
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


@pytest.mark.parametrize(
    "detail",
    [
        "a generation is already running",  # darkroom one at a time
        "8 generations are already queued or running",  # darkroom behind the argus-iris GPU broker, queue full
    ],
)
def test_a_busy_darkroom_is_waited_for_on_both_routes(tmp_path: Path, detail: str) -> None:
    with fake_darkroom(busy=1, depth_busy=1, busy_detail=detail) as (url, calls):
        plate = make(url, tmp_path / "garden")
    assert plate.depth is not None
    assert calls == ["generate", "generate", "image rec1", "make_depth rec1", "make_depth rec1", "depth rec1"]


def test_a_conflict_that_is_not_busy_is_an_answer_at_once(tmp_path: Path) -> None:
    with (
        fake_darkroom(busy=1, busy_detail="no such checkpoint") as (url, calls),
        pytest.raises(PlateError, match="no such checkpoint"),
    ):
        make(url, tmp_path / "garden")
    assert calls == ["generate"] and not (tmp_path / "garden").exists()


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


def test_a_failed_depth_write_leaves_the_plate_as_it_was(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with fake_darkroom() as (url, _calls):
        make(url, tmp_path / "garden", depth=False)
        before = sorted(p.name for p in (tmp_path / "garden").iterdir())
        with monkeypatch.context() as m:

            def full(self: Path, *args, **kwargs) -> int:  # noqa: ANN002, ANN003
                raise OSError(28, "No space left on device")

            m.setattr(Path, "write_text", full)
            with pytest.raises(OSError, match="No space"):
                add_depth(Darkroom(url), tmp_path / "garden")
        assert sorted(p.name for p in (tmp_path / "garden").iterdir()) == before
        assert add_depth(Darkroom(url), tmp_path / "garden").depth is not None, "a retry is not refused"


def test_a_plate_that_fails_part_way_leaves_nothing_to_block_a_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(directory: Path) -> PlateManifest:
        raise PlateError("the disk went away")

    with fake_darkroom() as (url, _calls):
        with monkeypatch.context() as m:
            m.setattr(backgrounds, "check_plate", broken)
            with pytest.raises(PlateError, match="went away"):
                make(url, tmp_path / "garden")
        assert not (tmp_path / "garden").exists()
        assert make(url, tmp_path / "garden").name == "garden"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["garden"], "no staging directory is left behind"


def test_a_plate_made_meanwhile_under_the_same_name_is_not_written_over(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = backgrounds.check_plate

    def meanwhile(directory: Path) -> PlateManifest:
        found = real(directory)
        if directory != tmp_path / "garden" and not (tmp_path / "garden").exists():
            (tmp_path / "garden").mkdir()
            (tmp_path / "garden" / "plate.png").write_bytes(b"theirs")
        return found

    monkeypatch.setattr(backgrounds, "check_plate", meanwhile)
    with fake_darkroom() as (url, _calls), pytest.raises(PlateError, match="not empty"):
        make(url, tmp_path / "garden")
    assert sorted(p.name for p in (tmp_path / "garden").iterdir()) == ["plate.png"]
    assert (tmp_path / "garden" / "plate.png").read_bytes() == b"theirs", "the other writer's plate is untouched"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["garden"]


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


@pytest.mark.parametrize("limit", ["MAX_IMAGE_BYTES", "MAX_PLATE_BYTES"])
def test_an_oversized_image_is_refused_before_it_is_read(
    plate: Path, monkeypatch: pytest.MonkeyPatch, limit: str
) -> None:
    read: list[str] = []
    real = Path.read_bytes

    def reading(self: Path) -> bytes:
        read.append(self.name)
        return real(self)

    monkeypatch.setattr(backgrounds, limit, 10)
    monkeypatch.setattr(Path, "read_bytes", reading)
    with pytest.raises(PlateError, match="over"):
        check_plate(plate)
    assert read == ["background.json"], "only the manifest is read"


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


# ---- plates of one's own, served from the assets ---------------------------------------------------


@pytest.fixture
def assets(tmp_path: Path) -> Path:
    """An assets root with one face and one plate (``backgrounds/rain``)."""
    from tests.helpers import make_face  # noqa: PLC0415

    root = tmp_path / "assets"
    make_face(root, "test-head")
    with fake_darkroom() as (url, _calls):
        make(url, root / "backgrounds" / "rain")
    return root


def client(assets: Path):  # noqa: ANN201 - a TestClient
    from starlette.testclient import TestClient  # noqa: PLC0415

    from argus_mask.config import Settings  # noqa: PLC0415
    from argus_mask.server.app import create_app  # noqa: PLC0415

    return TestClient(create_app(Settings(assets_dir=assets)))


def test_the_server_lists_built_in_scenes_then_plates(assets: Path) -> None:
    with client(assets) as c:
        scenes = c.get("/scenes").json()
    assert [(s["name"], s["kind"], s["built_in"]) for s in scenes] == [
        ("ftl", "procedural", True),
        ("garden", "plate", True),
        ("lake", "plate", True),
        ("rain", "plate", False),
    ]
    assert all(s["cleared"] for s in scenes)


def test_the_server_serves_a_checked_plate_and_only_its_files(assets: Path) -> None:
    plate = assets / "backgrounds" / "rain"
    with client(assets) as c:
        manifest = c.get("/scene/rain/background.json").json()
        assert manifest["name"] == "rain" and manifest["provenance"]["kind"] == "synthetic"
        image = c.get("/scene/rain/plate.png")
        assert image.status_code == 200 and image.content == (plate / "plate.png").read_bytes()
        assert (
            image.headers["content-type"] == "image/png" and image.headers["etag"] == f'"{manifest["plate"]["sha256"]}"'
        )
        assert c.get("/scene/rain/depth.png").content == (plate / "depth.png").read_bytes()
        (plate / "notes.png").write_bytes(png())
        assert c.get("/scene/rain/notes.png").status_code == 404, "a file the manifest does not name"
        # Each probe names a file that is really there, so a 404 means the server refused the path.
        secret = png(shade=3)
        (assets / "backgrounds" / "secret.png").write_bytes(secret)
        for probe in (
            "/scene/rain/..%2Fsecret.png",
            "/scene/rain/..%2F..%2Fbackgrounds%2Fsecret.png",
            "/scene/rain/%2E%2E%2Fsecret.png",
            "/scene/rain/..%2Frain%2Fplate.png",
        ):
            reply = c.get(probe)
            assert reply.status_code == 404, probe
            assert reply.content not in (secret, (plate / "plate.png").read_bytes()), probe
        assert c.get("/scene/nope/background.json").status_code == 404


def test_a_plate_that_does_not_hold_up_is_not_served(assets: Path) -> None:
    plate = assets / "backgrounds" / "rain"
    with client(assets) as c:
        assert c.get("/scene/rain/background.json").status_code == 200
        manifest = json.loads((plate / "background.json").read_text())
        del manifest["provenance"]
        (plate / "background.json").write_text(json.dumps(manifest))
        assert c.get("/scene/rain/background.json").status_code == 404, "re-checked once a file changes"
        assert c.get("/scene/rain/plate.png").status_code == 404
        assert "rain" not in [s["name"] for s in c.get("/scenes").json()]


def test_a_dangling_link_in_a_plate_refuses_only_that_plate(assets: Path, tmp_path: Path) -> None:
    rain = assets / "backgrounds" / "rain"
    (rain / "notes.png").symlink_to(tmp_path / "nowhere.png")  # not named by the manifest: no part of the plate
    store = backgrounds.PlateStore(assets)
    assert [m.name for m in store.scan()[0]] == ["rain"]
    (rain / "depth.png").unlink()
    (rain / "depth.png").symlink_to(tmp_path / "nowhere.png")
    accepted, refused = store.scan()
    assert accepted == [] and "depth.png is missing or not a plain file" in refused["rain"]
    assert [s.name for s in store.scenes()] == list(backgrounds.BUILT_IN_SCENES)


def test_a_plate_cannot_take_a_built_in_name(assets: Path) -> None:
    (assets / "backgrounds" / "rain").rename(assets / "backgrounds" / "garden")
    store = backgrounds.PlateStore(assets)
    assert store.scan()[0] == []
    with pytest.raises(PlateError, match="built-in"):
        store.load("garden")


def test_backgrounds_is_not_a_face(assets: Path) -> None:
    from argus_mask.editor.preset import Preset  # noqa: PLC0415
    from argus_mask.faces import FaceError, FaceStore  # noqa: PLC0415

    faces, refused = FaceStore(assets).scan()
    assert [f.id for f in faces] == ["test-head"] and "backgrounds" not in refused
    with pytest.raises(FaceError, match="reserved"):
        FaceStore(assets).load("backgrounds")
    with pytest.raises(ValidationError, match="reserved"):
        Preset.model_validate({"id": "backgrounds"})


def test_record_accepts_a_plate_the_server_serves(assets: Path) -> None:
    from argus_mask import record  # noqa: PLC0415
    from argus_mask.config import Settings  # noqa: PLC0415
    from argus_mask.server.app import create_app  # noqa: PLC0415

    with live_server(create_app(Settings(assets_dir=assets))) as url:
        assert record.scene_refusal(url, "rain") is None
        assert "choose from ftl, garden, lake, rain" in (record.scene_refusal(url, "warp") or "")
    assert "could not list" in (record.scene_refusal("http://127.0.0.1:9", "rain") or "")


# ---- each face's backdrop ------------------------------------------------------------------------


def test_a_reload_keeps_the_previous_choices_until_the_new_file_is_parsed(tmp_path, monkeypatch) -> None:
    """Readers share the cache unlocked. A reload must not publish an empty one while the new file is parsed."""
    path = tmp_path / "scenes.json"
    path.write_text(json.dumps({"faces": {"test-head": "garden"}, "shots": {"test-head": "wide"}}))
    choices = backgrounds.SceneChoices(tmp_path, backgrounds.PlateStore(tmp_path))
    assert choices.raw() == {"test-head": "garden"}
    path.write_text(json.dumps({"faces": {"test-head": "lake"}, "shots": {"test-head": "close"}}))
    seen: dict[str, dict[str, str]] = {}
    real = backgrounds.json.loads

    def spy(text: str, *args, **kwargs):
        seen["faces"] = dict(choices._faces)
        seen["shots"] = dict(choices._shots)
        return real(text, *args, **kwargs)

    monkeypatch.setattr(backgrounds.json, "loads", spy)
    assert choices.raw() == {"test-head": "lake"} and choices.raw_shots() == {"test-head": "close"}
    assert seen == {"faces": {"test-head": "garden"}, "shots": {"test-head": "wide"}}


def test_a_face_shows_its_chosen_backdrop_until_a_screen_asks_for_another(assets: Path) -> None:
    choices = backgrounds.SceneChoices(assets, backgrounds.PlateStore(assets))
    with client(assets) as c:
        assert c.get("/face/test-head/scene").json() == {"scene": None, "shot": None}
        choices.choose("test-head", "rain")
        assert c.get("/face/test-head/scene").json() == {"scene": "rain", "shot": None}
        assert c.get("/face/nobody/scene").status_code == 404
        # The plate goes: the choice stays in the file but is not offered, so the face shows a plain background.
        (assets / "backgrounds" / "rain" / "plate.png").unlink()
        assert c.get("/face/test-head/scene").json() == {"scene": None, "shot": None}
    assert choices.raw() == {"test-head": "rain"}


def test_a_face_whose_chosen_plate_was_changed_shows_a_plain_background(assets: Path) -> None:
    choices = backgrounds.SceneChoices(assets, backgrounds.PlateStore(assets))
    with client(assets) as c:
        choices.choose("test-head", "rain")
        assert c.get("/face/test-head/scene").json() == {"scene": "rain", "shot": None}
        # A replaced picture no longer matches its sha256: the plate is refused, as a missing one is.
        (assets / "backgrounds" / "rain" / "plate.png").write_bytes(png(shade=1))
        assert c.get("/face/test-head/scene").json() == {"scene": None, "shot": None}
    assert choices.raw() == {"test-head": "rain"}


def test_a_choice_must_be_a_scene_and_follows_its_face(assets: Path) -> None:
    choices = backgrounds.SceneChoices(assets, backgrounds.PlateStore(assets))
    with pytest.raises(PlateError, match="no scene 'warp'"):
        choices.choose("test-head", "warp")
    choices.choose("test-head", "ftl")
    choices.repoint("test-head", "renamed")
    assert choices.choices() == {"renamed": "ftl"}
    choices.choose("renamed", None)
    assert choices.choices() == {} and json.loads((assets / "scenes.json").read_text()) == {"faces": {}}


def test_a_splat_head_is_framed_in_the_shot_chosen_with_its_backdrop(assets: Path) -> None:
    """Close or wide, chosen with the backdrop (#151): a floating head on a plain background, or close over a scene."""
    choices = backgrounds.SceneChoices(assets, backgrounds.PlateStore(assets))
    with client(assets) as c:
        choices.choose("test-head", None, "close")  # a plain background, framed close
        assert c.get("/face/test-head/scene").json() == {"scene": None, "shot": "close"}
        choices.choose("test-head", "rain", "wide")  # a floating head over a scene
        assert c.get("/face/test-head/scene").json() == {"scene": "rain", "shot": "wide"}
        choices.choose("test-head", "rain")  # no shot: the one the background suits
        assert c.get("/face/test-head/scene").json() == {"scene": "rain", "shot": None}
    with pytest.raises(PlateError, match="no shot 'medium'"):
        choices.choose("test-head", "rain", "medium")
    choices.choose("test-head", None, "wide")
    choices.repoint("test-head", "renamed")
    assert choices.shots() == {"renamed": "wide"} and choices.choices() == {}
    choices.repoint("renamed", None)
    assert choices.shots() == {} and json.loads((assets / "scenes.json").read_text()) == {"faces": {}}
    # A shot written by hand that is not one there is: not offered.
    (assets / "scenes.json").write_text(json.dumps({"faces": {}, "shots": {"test-head": "fisheye"}}))
    assert choices.shot_for("test-head") is None


def test_none_is_never_a_plate(assets: Path) -> None:
    (assets / "backgrounds" / "rain").rename(assets / "backgrounds" / "none")
    with pytest.raises(PlateError, match="built-in"):
        backgrounds.PlateStore(assets).load("none")
