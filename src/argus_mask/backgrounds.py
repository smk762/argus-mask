"""Plates: backdrops made from a still image, generated through argus-darkroom (#55).

A plate is a directory: ``background.json`` (:class:`~argus_mask.models.PlateManifest`), the picture
(``plate.png``), and optionally a depth map and masks. The viewer animates the still with built-in effects
(``viewer/scenes/plate.js``); nothing here runs in the server or the engine.

:func:`make_plate` is the paved path: it asks darkroom for the picture (its prompt check, its output check, its
record), then for a depth map of that record (``POST /depth/{id}``), checks every byte against the record's
sha256 and writes the directory with its provenance. :func:`check_plate` is what a reader of a plate runs: the
manifest parses, every file is there, a plain PNG of the stated size, and matches its sha256.

Stdlib only (urllib, not an HTTP library), like :mod:`argus_mask.client`.
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import struct
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from argus_mask.models import (
    SCENE_NAME_PATTERN,
    DarkroomDepth,
    DarkroomRender,
    Generator,
    PlateEffects,
    PlateImage,
    PlateManifest,
    Provenance,
    SceneInfo,
)

MANIFEST = "background.json"
# The viewer's built-in scenes, in backdrop.js's order (a test holds the two together), with their titles: a
# procedural scene is a module (viewer/scenes/<name>.js), a plate a directory (viewer/scenes/<name>/).
BUILT_IN_SCENES: dict[str, str] = {
    "ftl": "FTL starstreaks",
    "garden": "Conservatory garden",
    "lake": "Alpine lake at dusk",
}
VIEWER_SCENES = Path(__file__).parent / "viewer" / "scenes"
# Plates of one's own live in <assets>/backgrounds/<name>/; the name is reserved, so no face can take it.
PLATES_DIR = "backgrounds"
NO_SCENE = "none"  # ?scene=none: a plain background, whatever the face's default; never a plate's name
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
# A plate ships in the viewer or is fetched by every screen: one image at most this big, all of them together
# at most the total. A 1344x768 PNG from darkroom is about 2 MB; a depth map a few hundred KB.
MAX_IMAGE_BYTES = 4 * 1024 * 1024
MAX_PLATE_BYTES = 6 * 1024 * 1024

# Checkpoints whose licence lets their output be a cleared plate, by a fragment of the file name darkroom uses.
# Any other model needs its licence stated (``--license``), and stating it is the author's claim.
KNOWN_LICENSES: dict[str, str] = {
    "flux1-schnell": "Apache-2.0",
}
# Every plate is safe for work: darkroom enforces its SFW policy, whatever its own default is.
POLICY = "default"
# What every plate prompt asks for, unless the author takes it as given: the face is the only person.
PROMPT_SUFFIX = "no people, no text, no logos"


class PlateError(Exception):
    """A plate could not be made, or a plate directory does not hold up."""


# ---- reading a plate -----------------------------------------------------------------------------


def png_size(data: bytes) -> tuple[int, int]:
    """Width and height from a PNG's IHDR chunk; raises :class:`PlateError` for anything that is not a PNG."""
    if len(data) < 24 or not data.startswith(PNG_SIGNATURE) or data[12:16] != b"IHDR":
        raise PlateError("not a PNG")
    width, height = struct.unpack(">II", data[16:24])
    if not (width and height):
        raise PlateError("a PNG with no pixels")
    return width, height


def image_of(directory: Path, file: str, data: bytes | None = None) -> PlateImage:
    """A :class:`PlateImage` for a file in *directory* (or for *data*, about to be written there as *file*)."""
    if data is None:
        data = (directory / file).read_bytes()
    width, height = png_size(data)
    return PlateImage(file=file, sha256=hashlib.sha256(data).hexdigest(), width=width, height=height)


def check_plate(directory: Path) -> PlateManifest:
    """The plate in *directory*, checked; raises :class:`PlateError` listing every problem found."""
    path = directory / MANIFEST
    if not path.is_file():
        raise PlateError(f"{directory}: no {MANIFEST}")
    try:
        manifest = PlateManifest.model_validate_json(path.read_bytes())
    except ValidationError as exc:
        raise PlateError(f"{path}: {exc}") from exc
    if manifest.name != directory.name:
        raise PlateError(f"{path}: the plate is named {manifest.name!r} but lives in {directory.name!r}")
    problems: list[str] = []
    total = 0
    for role in ("plate", "depth", "sway_mask", "light_mask"):
        image: PlateImage | None = getattr(manifest, role)
        if image is None:
            continue
        file = directory / image.file
        if file.is_symlink() or not file.is_file():  # a link could point anywhere: a plate holds its own files
            problems.append(f"{role}: {image.file} is missing or not a plain file")
            continue
        data = file.read_bytes()
        total += len(data)
        if len(data) > MAX_IMAGE_BYTES:
            problems.append(f"{role}: {image.file} is {len(data) // 1024} KB, over {MAX_IMAGE_BYTES // 1024} KB")
        try:
            found = image_of(directory, image.file, data)
        except PlateError as exc:
            problems.append(f"{role}: {image.file}: {exc}")
            continue
        if found.sha256 != image.sha256:
            problems.append(f"{role}: {image.file} does not match its sha256")
        if (found.width, found.height) != (image.width, image.height):
            problems.append(f"{role}: {image.file} is {found.width}x{found.height}, not {image.width}x{image.height}")
    if total > MAX_PLATE_BYTES:
        problems.append(f"the plate's images come to {total // 1024} KB, over {MAX_PLATE_BYTES // 1024} KB")
    if problems:
        raise PlateError(f"{directory}: " + "; ".join(problems))
    return manifest


# ---- darkroom ------------------------------------------------------------------------------------


def _sse(lines: Iterator[bytes]) -> Iterator[tuple[str, Any]]:
    """(event, data) from a text/event-stream, data parsed as JSON."""
    event, data = "message", []
    for raw in lines:
        line = raw.decode("utf-8").rstrip("\r\n")
        if line == "":
            if data:
                yield event, json.loads("\n".join(data))
            event, data = "message", []
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip())
    if data:
        yield event, json.loads("\n".join(data))


# How darkroom says it is busy, in a 409's detail: one generation at a time, or (behind the argus-iris GPU broker) its
# queue full. Any other 409 (no stored image, say) is an answer, not a reason to wait.
_BUSY = (b"already running", b"already queued")


def _busy(detail: bytes) -> bool:
    return any(marker in detail for marker in _BUSY)


@dataclass
class Darkroom:
    """The few argus-darkroom routes a plate needs. Waits while darkroom is busy with another generation."""

    url: str
    timeout: float = 900.0
    busy_wait: float = 5.0
    busy_timeout: float = 600.0
    sleep: Callable[[float], None] = field(default=time.sleep, repr=False)

    def _open(self, method: str, path: str, body: dict | None = None):  # noqa: ANN202 - an http.client response
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(f"{self.url.rstrip('/')}{path}", data=data, method=method)
        if data is not None:
            request.add_header("Content-Type", "application/json")
        return urllib.request.urlopen(request, timeout=self.timeout)  # noqa: S310 - the URL is the operator's

    def _call(self, method: str, path: str, body: dict | None = None) -> bytes:
        deadline = time.monotonic() + self.busy_timeout
        while True:
            try:
                with self._open(method, path, body) as reply:
                    return reply.read()
            except urllib.error.HTTPError as exc:
                detail = exc.read()
                if exc.code == 409 and _busy(detail) and time.monotonic() < deadline:
                    self.sleep(self.busy_wait)
                    continue
                raise PlateError(f"darkroom {method} {path}: {exc.code} {_detail(detail)}") from exc
            except OSError as exc:
                raise PlateError(f"darkroom is unreachable at {self.url}: {exc}") from exc

    def generate(self, body: dict, on_event: Callable[[str, Any], None] | None = None) -> dict:
        """One generation; the record darkroom kept for it."""
        deadline = time.monotonic() + self.busy_timeout
        while True:
            try:
                with self._open("POST", "/generate/stream", body) as reply:
                    for event, data in _sse(reply):
                        if on_event:
                            on_event(event, data)
                        if event == "complete":
                            return data["record"]
                        if event == "error":
                            raise PlateError(f"darkroom: {data.get('detail', 'the generation failed')}")
                raise PlateError("darkroom's stream ended without a record")
            except urllib.error.HTTPError as exc:
                detail = exc.read()
                if exc.code == 409 and time.monotonic() < deadline:
                    self.sleep(self.busy_wait)
                    continue
                raise PlateError(f"darkroom refused the generation: {exc.code} {_detail(detail)}") from exc
            except OSError as exc:
                raise PlateError(f"darkroom is unreachable at {self.url}: {exc}") from exc

    def image(self, record_id: str) -> bytes:
        return self._call("GET", f"/image/{record_id}")

    def make_depth(self, record_id: str) -> dict:
        return json.loads(self._call("POST", f"/depth/{record_id}"))

    def depth(self, record_id: str) -> bytes:
        return self._call("GET", f"/depth/{record_id}")


def _detail(body: bytes) -> str:
    try:
        return str(json.loads(body).get("detail", body.decode(errors="replace")))
    except (ValueError, AttributeError):
        return body.decode(errors="replace")[:300]


# ---- making a plate ------------------------------------------------------------------------------


@dataclass
class PlateRequest:
    """What to ask darkroom for. The defaults suit FLUX.1 schnell, the one checkpoint known to be cleared."""

    prompt: str
    checkpoint: str = "flux/flux1-schnell-fp8-e4m3fn.safetensors"
    negative: str = ""  # FLUX workflows take none; the prompt itself asks for no people
    seed: int | None = None
    width: int = 1344
    height: int = 768
    steps: int = 4
    cfg: float = 1.0
    guidance: float | None = 3.5
    sampler: str = "euler"
    scheduler: str = "simple"
    as_is: bool = False  # take the prompt as given: no PROMPT_SUFFIX

    def final_prompt(self) -> str:
        prompt = self.prompt.strip()
        if self.as_is or PROMPT_SUFFIX in prompt:
            return prompt
        return f"{prompt.rstrip(',. ')}, {PROMPT_SUFFIX}"

    def body(self, seed: int) -> dict:
        sampling: dict[str, Any] = {
            "sampler": self.sampler,
            "scheduler": self.scheduler,
            "steps": self.steps,
            "cfg": self.cfg,
            "width": self.width,
            "height": self.height,
        }
        if self.guidance is not None:
            sampling["guidance"] = self.guidance
        return {
            "user_prompt": self.final_prompt(),
            "negative_prompt": self.negative,
            "expand": False,  # darkroom's expansion would replace the prompt the plate records
            "seed": seed,
            "checkpoint": self.checkpoint,
            "sampling": sampling,
            "loras": [],
            "enforcing_policy": POLICY,
            "advisory_policies": [],
        }


def license_for(checkpoint: str, stated: str = "") -> str:
    """The licence a plate from *checkpoint* is under: the one stated, else a known one; refuses neither."""
    if stated.strip():
        return stated.strip()
    name = checkpoint.rsplit("/", 1)[-1].lower()
    for fragment, licence in KNOWN_LICENSES.items():
        if fragment in name:
            return licence
    known = ", ".join(f"{k} ({v})" for k, v in KNOWN_LICENSES.items())
    raise PlateError(
        f"{checkpoint}: its output licence is not known here; state it with --license after checking the model's"
        f" terms, or use a known one: {known}"
    )


def _check_record(record: dict, body: dict) -> DarkroomRender:
    """The render a released record describes, refusing one that is not what was asked for."""
    rid = str(record.get("id") or "")
    outcome = record.get("outcome")
    if outcome != "passed":
        reason = record.get("outcome_reason") or ""
        raise PlateError(f"darkroom record {rid}: {outcome}{': ' + reason if reason else ''}; no plate made")
    asked = record.get("request") or {}
    if asked.get("expanded_prompt"):
        raise PlateError(f"darkroom record {rid}: the prompt was expanded")
    if asked.get("final_prompt") != body["user_prompt"]:
        raise PlateError(f"darkroom record {rid}: darkroom rendered a different prompt")
    if asked.get("seed") != body["seed"]:
        raise PlateError(f"darkroom record {rid}: seed {asked.get('seed')}, not {body['seed']}")
    policy = (record.get("guard") or {}).get("enforcing_policy")
    if policy != POLICY:
        raise PlateError(f"darkroom record {rid}: enforced {policy!r}, not {POLICY!r}")
    checkpoint = (asked.get("checkpoint") or {}).get("name") or body["checkpoint"]
    sampling = asked.get("sampling") or {}
    return DarkroomRender(
        record_id=rid,
        darkroom_version=str(record.get("darkroom_version") or ""),
        enforcing_policy=policy,
        checkpoint=checkpoint,
        seed=body["seed"],
        prompt=body["user_prompt"],
        negative_prompt=asked.get("negative_prompt", body["negative_prompt"]),
        workflow=asked.get("workflow"),
        sampler=sampling.get("sampler"),
        scheduler=sampling.get("scheduler"),
        steps=sampling.get("steps"),
        cfg=sampling.get("cfg"),
        guidance=sampling.get("guidance"),
    )


def make_plate(
    darkroom: Darkroom,
    directory: Path,
    request: PlateRequest,
    *,
    title: str,
    license: str = "",  # noqa: A002 - the provenance field's own name
    tint: str = "#ffffff",
    effects: PlateEffects | None = None,
    depth: bool = True,
    on_event: Callable[[str, Any], None] | None = None,
) -> PlateManifest:
    """Generate a plate through darkroom into *directory* (its name is the plate's), and check it."""
    licence = license_for(request.checkpoint, license)  # before the GPU is asked for anything
    if directory.exists() and any(directory.iterdir()):
        raise PlateError(f"{directory} is not empty: a plate is never written over another")
    seed = request.seed if request.seed is not None else secrets.randbelow(2**31)
    body = request.body(seed)
    record = darkroom.generate(body, on_event)
    render = _check_record(record, body)
    picture = darkroom.image(render.record_id)
    expected = (record.get("image") or {}).get("sha256")
    if not expected or hashlib.sha256(picture).hexdigest() != expected:
        raise PlateError(f"darkroom record {render.record_id}: the image does not match the record's sha256")
    files = {"plate.png": picture}
    depth_image = None
    if depth:
        render.depth, files["depth.png"] = _fetch_depth(darkroom, render.record_id)
        depth_image = image_of(directory, "depth.png", files["depth.png"])
    manifest = PlateManifest(
        name=directory.name,
        title=title,
        tint=tint,
        plate=image_of(directory, "plate.png", picture),
        depth=depth_image,
        effects=effects or PlateEffects(),
        provenance=Provenance(
            kind="synthetic",
            source=f"argus-darkroom record {render.record_id}",
            license=licence,
            notes="A still generated through argus-darkroom under its SFW policy, with no person in the prompt.",
            generator=Generator(model=render.checkpoint, seed=seed, prompt=render.prompt),
        ),
        darkroom=render,
    )
    directory.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (directory / name).write_bytes(data)
    (directory / MANIFEST).write_text(manifest.model_dump_json(indent=2, exclude_none=True) + "\n")
    return check_plate(directory)


def _fetch_depth(darkroom: Darkroom, record_id: str) -> tuple[DarkroomDepth, bytes]:
    """Darkroom's depth map of a record, checked against the sha256 darkroom gives for it."""
    made = darkroom.make_depth(record_id)
    png = darkroom.depth(record_id)
    if hashlib.sha256(png).hexdigest() != made.get("sha256"):
        raise PlateError(f"darkroom record {record_id}: the depth map does not match its sha256")
    try:
        found = DarkroomDepth(
            model=made["model"],
            model_license=made["model_license"],
            sha256=made["sha256"],
            workflow=made.get("workflow"),
        )
    except (KeyError, ValidationError) as exc:
        raise PlateError(f"darkroom record {record_id}: the depth record is incomplete: {exc}") from exc
    return found, png


def add_depth(darkroom: Darkroom, directory: Path) -> PlateManifest:
    """Give a darkroom plate made without one (``--no-depth``) darkroom's depth map of its record."""
    manifest = check_plate(directory)
    if manifest.darkroom is None:
        raise PlateError(f"{directory}: not a darkroom plate, so darkroom has no record to make a depth map from")
    if manifest.depth is not None:
        raise PlateError(f"{directory}: the plate has a depth map already")
    if (directory / "depth.png").exists():
        raise PlateError(f"{directory}: depth.png is there but not in the manifest; move it aside first")
    found, png = _fetch_depth(darkroom, manifest.darkroom.record_id)
    darkroom_block = manifest.darkroom.model_copy(update={"depth": found})
    updated = PlateManifest.model_validate(
        manifest.model_dump()
        | {"depth": image_of(directory, "depth.png", png).model_dump(), "darkroom": darkroom_block.model_dump()}
    )
    (directory / "depth.png").write_bytes(png)
    (directory / MANIFEST).write_text(updated.model_dump_json(indent=2, exclude_none=True) + "\n")
    return check_plate(directory)


# ---- serving plates ------------------------------------------------------------------------------


class PlateStore:
    """The plates in ``<assets>/backgrounds/``, each checked (:func:`check_plate`) before it is served.

    A plate that does not hold up (no provenance, a changed file, a link) is refused with the reason, as a face
    is. The verdict is kept until a file in the plate's directory changes, so a screen asking for a plate does not
    hash its images every time. A built-in scene's name is never taken by a plate here.
    """

    def __init__(self, assets_root: Path) -> None:
        self.root = assets_root / PLATES_DIR
        self._verdicts: dict[str, tuple[tuple, PlateManifest | str]] = {}

    @staticmethod
    def _version(directory: Path) -> tuple:
        return tuple(sorted((p.name, p.stat().st_mtime_ns, p.stat().st_size) for p in directory.iterdir()))

    def load(self, name: str) -> tuple[PlateManifest, Path]:
        """The plate *name* and its directory; raises :class:`PlateError` saying why it is not served."""
        if not re.fullmatch(SCENE_NAME_PATTERN, name):
            raise PlateError(f"{name!r} is not a scene name")
        if name in BUILT_IN_SCENES or name == NO_SCENE:
            raise PlateError(f"{name} is a built-in scene: a plate of one's own cannot take its name")
        directory = self.root / name
        if directory.is_symlink() or not directory.is_dir():
            raise PlateError(f"no plate {name!r}")
        version = self._version(directory)
        cached = self._verdicts.get(name)
        if cached is None or cached[0] != version:
            try:
                verdict: PlateManifest | str = check_plate(directory)
            except PlateError as exc:
                verdict = str(exc)
            self._verdicts[name] = (version, verdict)
        verdict = self._verdicts[name][1]
        if isinstance(verdict, str):
            raise PlateError(verdict)
        return verdict, directory

    def scan(self) -> tuple[list[PlateManifest], dict[str, str]]:
        """Every plate that may be served, and for each directory that may not, the reason."""
        accepted: list[PlateManifest] = []
        refused: dict[str, str] = {}
        if not self.root.is_dir():
            return accepted, refused
        for directory in sorted(p for p in self.root.iterdir() if p.is_dir() and not p.name.startswith(".")):
            try:
                accepted.append(self.load(directory.name)[0])
            except PlateError as exc:
                refused[directory.name] = str(exc)
        return accepted, refused

    def scenes(self) -> list[SceneInfo]:
        """What a screen can show: the built-in scenes, then the plates here."""
        found = []
        for name, title in BUILT_IN_SCENES.items():
            plate = VIEWER_SCENES / name / MANIFEST
            cleared = (
                PlateManifest.model_validate_json(plate.read_bytes()).provenance.cleared if plate.is_file() else True
            )
            found.append(
                SceneInfo(
                    name=name,
                    title=title,
                    kind="plate" if plate.is_file() else "procedural",
                    built_in=True,
                    cleared=cleared,
                )
            )
        for m in self.scan()[0]:
            found.append(
                SceneInfo(name=m.name, title=m.title, kind="plate", built_in=False, cleared=m.provenance.cleared)
            )
        return found


# ---- each face's backdrop ------------------------------------------------------------------------

SCENES_NAME = "scenes.json"


class SceneChoices:
    """Which backdrop each face shows by default, in ``<assets>/scenes.json``: ``{"faces": {"ginger-3": "garden"}}``.

    Written by the editor (Saved faces) and read by the server, like ``expressions.json``. A screen's own
    ``?scene=`` wins; a choice whose scene is gone (a plate removed or refused) is ignored, so the face shows a
    plain background rather than none at all.
    """

    def __init__(self, assets_root: Path, plates: PlateStore) -> None:
        self.path = assets_root / SCENES_NAME
        self.plates = plates
        self._seen: tuple[int, int, int] | None = None
        self._faces: dict[str, str] = {}
        self._lock = threading.Lock()  # the editor's routes run in threads: one read-modify-write at a time

    def raw(self) -> dict[str, str]:
        """Face id -> scene name, as written: valid or not."""
        try:
            stat = self.path.stat()
        except OSError:
            self._seen, self._faces = None, {}
            return {}
        key = (stat.st_ino, stat.st_mtime_ns, stat.st_size)  # every write replaces the file: a new inode
        if key != self._seen:
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                faces = data.get("faces") if isinstance(data, dict) else None
                self._faces = {str(k): str(v) for k, v in faces.items()} if isinstance(faces, dict) else {}
            except (OSError, ValueError):
                self._faces = {}
            self._seen = key
        return dict(self._faces)

    def available(self) -> set[str]:
        return {s.name for s in self.plates.scenes()}

    def choices(self) -> dict[str, str]:
        """Face id -> its scene, for the faces whose scene can be shown."""
        available = self.available()
        return {f: s for f, s in sorted(self.raw().items()) if s in available}

    def for_face(self, face_id: str) -> str | None:
        return self.choices().get(face_id)

    def _write(self, faces: dict[str, str]) -> None:
        tmp = self.path.with_name(SCENES_NAME + ".part")
        tmp.write_text(json.dumps({"faces": dict(sorted(faces.items()))}, indent=2) + "\n", encoding="utf-8")
        tmp.replace(self.path)

    def choose(self, face_id: str, scene: str | None) -> None:
        """*face_id* shows *scene* by default; None clears its choice (a plain background)."""
        with self._lock:
            if scene is not None and scene not in self.available():
                raise PlateError(f"no scene {scene!r}: there are {', '.join(sorted(self.available()))}")
            faces = {k: v for k, v in self.raw().items() if k != face_id}
            if scene is not None:
                faces[face_id] = scene
            self._write(faces)

    def repoint(self, old: str, new: str | None) -> None:
        """A face was renamed (*new*) or deleted (None): its choice follows it, or goes."""
        with self._lock:
            faces = self.raw()
            if old not in faces:
                return
            scene = faces.pop(old)
            if new is not None:
                faces[new] = scene
            self._write(faces)
