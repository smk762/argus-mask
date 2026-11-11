"""The face editor's HTTP app.

``GET  /health``                        whether a build can run, and why not; whether one is running
``GET  /library``                       the installed, licensed assets, by part (and ``clothes``, with a ``slot``)
``GET  /library/{part}/{name}/thumb``   an asset's thumbnail (PNG)
``GET  /modifiers``                     the shape sliders a preset may set (MakeHuman targets), by group
``GET  /presets``                       the faces the editor made: each preset, and whether it is previewed, exported
``POST /build``                         build a preset into a preview; answers when it is done (about 10 s)
``POST /export/{id}``                   copy a preview into the assets root, where the server serves it
``GET  /face/{id}``, ``/face/{id}/model``  a face, for the page: ``?from=preview`` or ``export``; by default
                                        the editor's preview, else its export
``GET  /cache/{key}``, ``/cache/{key}/model``  an earlier build, by the key /build answered with
``GET  /saved``                         every face in the assets root and every preview, newest first
``GET  /saved/{id}/preset``             the preset behind a face the editor made (``?from=preview|export``)
``POST /saved/{id}/rename``             ``{"id": new id, "name": new name}``: the export and the preview together
``DELETE /saved/{id}``                  move a face the editor made to ``.trash/`` (``?from=preview|export``, default both)
``GET  /faces.html``                    the saved-faces page
``GET  /broadcast/state``               the server's faces, who watches them, the TTS defaults, running broadcasts
``GET  /voices``                        the voices a (Wyoming) TTS service lists: ``?tts=`` its URL
``POST /broadcast/say``, ``/broadcast/push``  say text through TTS, or play a WAV (the body), into a face
``GET  /broadcast/{job}``, ``POST /broadcast/{job}/stop``  a broadcast's progress; stop it
``GET  /broadcast.html``                the Broadcast page
``GET  /``                              the editor page

``POST /build`` starts Blender, so it is behind the same origin and read-only
guards as the server's writes. A preset names assets by reference, and every
reference is looked up in the installed library: no path from the request
reaches the file system or Blender's command line except through that lookup.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Literal

import numpy as np
import structlog
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from argus_mask import __version__, client
from argus_mask.audio import AudioError, read_wav
from argus_mask.config import EditorSettings, Settings
from argus_mask.editor.broadcast import Broadcasts, RecordOptions, recording_ready
from argus_mask.editor.broadcast import Busy as Broadcasting
from argus_mask.editor.builder import PLACES, Builder, BuildError, Busy, NoPreview, NotOwned, Refused, Taken
from argus_mask.editor.library import CLOTHES, AssetLibrary, LibraryError
from argus_mask.editor.preset import MODIFIER_GROUPS, MODIFIERS, PART_DIRS, Preset, one_way
from argus_mask.editor.voices import (
    DELIVERIES,
    SOURCES,
    VOICE_ID,
    Generations,
    VoiceError,
    VoiceStore,
    provenance,
    wav_bytes,
)
from argus_mask.faces import FaceError, FaceStore
from argus_mask.models import FACE_ID_PATTERN, FaceManifest
from argus_mask.server.app import VIEWER_DIR, install_guards
from argus_mask.voice import analyse, assess

log = structlog.get_logger(__name__)

SERVER_TIMEOUT_S = 3.0
_FACE_ID = re.compile(FACE_ID_PATTERN)


def chime(rate: int = 24000) -> np.ndarray:
    """A short two-note chime, for hearing that a screen plays sound: no voice, nothing to mistake for one."""
    out = []
    for freq, seconds in ((660.0, 0.22), (880.0, 0.34)):
        t = np.arange(int(rate * seconds)) / rate
        env = np.minimum(1, t / 0.01) * np.exp(-t * 6)
        out.append((0.3 * env * np.sin(2 * np.pi * freq * t)).astype(np.float32))
    return np.concatenate([*out, np.zeros(int(rate * 0.15), dtype=np.float32)])


def server_get(base: str, path: str) -> tuple[int | None, dict | list | None, str | None]:
    """GET *path* on the argus-mask server: (status, JSON, why not), never raising."""
    try:
        with urllib.request.urlopen(f"{base}{path}", timeout=SERVER_TIMEOUT_S) as reply:  # noqa: S310 - http(s) only, checked in config
            return reply.status, json.loads(reply.read()), None
    except urllib.error.HTTPError as exc:
        return exc.code, None, f"{exc.code} {exc.reason}"
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return None, None, f"no argus-mask server answers at {base} ({getattr(exc, 'reason', exc)})"


MAX_TEXT = 5000  # characters a broadcast may say: a long paragraph or two
MAX_AUDIO_BYTES = 25 * 1024 * 1024  # a pushed file: minutes of 48 kHz 16-bit audio


def wav_from_bytes(data: bytes, name: str) -> tuple[np.ndarray, int]:
    """:func:`read_wav` on an uploaded file's bytes; errors name the upload, not the temporary file."""
    with tempfile.NamedTemporaryFile(suffix=".wav") as handle:
        handle.write(data)
        handle.flush()
        try:
            return read_wav(handle.name)
        except AudioError as exc:
            raise AudioError(str(exc).replace(handle.name, name)) from exc


RecordSize = Literal["1920x1080", "1280x720", "854x480", "1080x1920"]  # 1080p, 720p, 480p; 1080x1920: the HLD panel
RecordQuality = Literal["low", "medium", "high"]
_RECORDING = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.(mp4|webm|wav|ndjson)$")


class Record(BaseModel):
    """Record the broadcast to an MP4, as `argus-mask record` does."""

    model_config = ConfigDict(extra="forbid")
    size: RecordSize = "1280x720"
    quality: RecordQuality = "high"

    def options(self) -> RecordOptions:
        width, height = (int(n) for n in self.size.split("x"))
        return RecordOptions(width=width, height=height, quality=self.quality)


class Say(BaseModel):
    """Everything `argus-mask say` takes, as the Broadcast page sends it."""

    model_config = ConfigDict(extra="forbid")
    face: str = Field(..., pattern=FACE_ID_PATTERN)
    text: str = Field(..., min_length=1, max_length=MAX_TEXT)
    tts_url: str = Field("", max_length=500, description="default: ARGUS_MASK_TTS_URL")
    voice: str = Field("", max_length=200, description="default: ARGUS_MASK_TTS_VOICE")
    params: dict[str, str | int | float | bool] = Field(default_factory=dict, description="-p key=value fields")
    seed: int = Field(0, ge=0, le=2**31 - 1)
    stream: bool = False
    pause: int = Field(300, ge=0, le=10_000, description="ms between sentences; 0 sends the text whole")
    tidy: bool = Field(True, description="numbers as words, short sentences joined, soft edges, even loudness")
    context: bool = Field(False, description="each sentence sent with the one before it as context (experimental)")
    record: Record | None = Field(None, description="also record the broadcast to a file")


class VoiceGenerate(BaseModel):
    """A synthetic candidate: the TTS service's own speaker, or a described voice, reading a script."""

    model_config = ConfigDict(extra="forbid")
    script: str = Field(..., min_length=20, max_length=1500)
    speaker: str = Field("", max_length=40, description="one of the service's own speakers (Miso: 0 or 1)")
    description: str = Field("", max_length=300, description="a described voice, on a service that designs them")
    instruction: str = Field("", max_length=300, description="how to say it, on a service that takes one")
    temperature: float | None = Field(None, ge=0.1, le=1.5)
    tts_url: str = Field("", max_length=500)


class VoiceClip(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sha: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    name: str = Field("clip.wav", max_length=200)
    transcript: str = Field(..., min_length=1, max_length=3000)
    generated: bool = False


class VoiceCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(..., pattern=VOICE_ID)
    name: str = Field(..., min_length=1, max_length=80)
    delivery: Literal[DELIVERIES] = "neutral"  # type: ignore[valid-type]
    clips: list[VoiceClip] = Field(..., min_length=1, max_length=5)
    source: dict = Field(..., description="kind (generated, own, consented), and what made it or the release")
    tts_url: str = Field("", max_length=500)


class VoiceTry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    voice: str = Field(..., min_length=1, max_length=200)
    text: str = Field(..., min_length=1, max_length=600)
    temperature: float | None = Field(None, ge=0.1, le=1.5)
    tidy: bool = True
    tts_url: str = Field("", max_length=500)


class Rename(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str | None = Field(None, pattern=FACE_ID_PATTERN)
    name: str | None = Field(None, min_length=1, max_length=80)


PAGE_DIR = Path(__file__).resolve().parent / "page"


def create_editor_app(
    settings: Settings | None = None, editor: EditorSettings | None = None, *, builder: Builder | None = None
) -> FastAPI:
    cfg = settings or Settings.from_env()
    ed = editor or EditorSettings.from_env()
    library = AssetLibrary(ed.data_dir)
    builder = builder or Builder(
        blender=ed.blender,
        blender_user=ed.blender_user,
        library=library,
        assets_dir=cfg.assets_dir,
        timeout_s=ed.build_timeout_s,
        cache_size=ed.build_cache,
    )
    store = FaceStore(cfg.assets_dir)
    previews = FaceStore(builder.preview_dir)
    broadcasts = Broadcasts(ed.server_url, ed.recordings_dir)
    voice_store = VoiceStore(ed.voices_dir)
    generations = Generations(voice_store)
    making = asyncio.Lock()  # one voice made at a time: two of the same id must not both pass the checks

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        await broadcasts.stop_all()
        await generations.stop_all()

    app = FastAPI(title="Argus Mask face editor", version=__version__, lifespan=lifespan)
    app.state.builder = builder
    origin_refusal = install_guards(app, cfg)

    @app.get("/health")
    def health() -> dict:
        reason = builder.check()
        return {
            "version": __version__,
            "ready": reason is None,
            "reason": reason,
            "building": builder.busy,
            "read_only": cfg.read_only,
        }

    @app.get("/library")
    def assets() -> dict[str, list[dict]]:
        return {part: [a.public() for a in found] for part, found in builder.library.scan().items()}

    @app.get("/library/{part}/{name}/thumb")
    def thumb(part: str, name: str) -> FileResponse:
        asset = builder.library.find(part, name) if part in PART_DIRS or part == CLOTHES else None
        if asset is None or asset.thumb is None:
            raise HTTPException(404, "no thumbnail")
        return FileResponse(asset.thumb, media_type="image/png")

    @app.get("/modifiers")
    def modifiers() -> dict:
        return {
            "groups": MODIFIER_GROUPS,
            "modifiers": [
                {
                    "name": m["name"],
                    "group": m["group"],
                    "label": m["label"],
                    "min": 0.0 if one_way(m["name"]) else -1.0,
                    "ends": m.get("ends") or [],
                }
                for m in MODIFIERS.values()
            ],
        }

    @app.get("/presets")
    def presets() -> list[dict]:
        return [{**e, "preset": e["preset"].model_dump(mode="json", exclude_none=True)} for e in builder.presets()]

    def find_face(face_id: str, where: str | None) -> tuple[FaceManifest, Path]:
        """A preview (the editor's own), or a face in the assets root: the same ones the server serves."""
        if where not in (None, "preview", "export"):
            raise HTTPException(422, "from is preview or export")
        preview = where == "preview" or (where is None and builder.previewed(face_id))
        faces = previews if preview else store
        try:
            return faces.load(face_id)
        except FaceError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/face/{face_id}", response_model=FaceManifest)
    def face(face_id: str, where: str | None = Query(None, alias="from")) -> FaceManifest:
        return find_face(face_id, where)[0]

    @app.get("/face/{face_id}/model")
    def face_model(face_id: str, where: str | None = Query(None, alias="from")) -> FileResponse:
        manifest, path = find_face(face_id, where)
        return FileResponse(path, media_type="model/gltf-binary", headers={"ETag": f'"{manifest.model.sha256}"'})

    @app.post("/build")
    async def build(preset: Preset) -> dict:
        try:
            result = await builder.build(preset)
        except Busy as exc:
            raise HTTPException(409, str(exc)) from exc
        except NotOwned as exc:
            raise HTTPException(409, str(exc)) from exc
        except (LibraryError, Refused) as exc:
            raise HTTPException(422, str(exc)) from exc
        except BuildError as exc:
            log.warning("build_failed", face=preset.id, reason=str(exc))
            raise HTTPException(500, str(exc)) from exc
        recipe = result.recipe
        return {
            "manifest": result.manifest.model_dump(mode="json"),
            "seconds": result.seconds,
            "triangles": recipe.get("triangles_total"),
            "bytes": recipe.get("bytes"),
            "age_years": preset.age_years(),
            "key": result.key,
            "cached": result.cached,
        }

    @app.post("/export/{face_id}")
    async def export(face_id: str) -> dict:
        try:
            manifest = await builder.export(face_id)
        except NoPreview as exc:
            raise HTTPException(404, str(exc)) from exc
        except (Busy, NotOwned) as exc:
            raise HTTPException(409, str(exc)) from exc
        except (BuildError, OSError) as exc:
            log.warning("export_failed", face=face_id, reason=str(exc))
            raise HTTPException(500, f"export failed: {exc}") from exc
        return {"manifest": manifest.model_dump(mode="json"), "path": f"{cfg.assets_dir / face_id}"}

    def from_cache(key: str) -> tuple[FaceManifest, Path]:
        found = builder.cached_face(key)
        if found is None:
            raise HTTPException(404, "that build is no longer in the cache")
        return found

    @app.get("/cache/{key}", response_model=FaceManifest)
    def cache_manifest(key: str) -> FaceManifest:
        return from_cache(key)[0]

    @app.get("/cache/{key}/model")
    def cache_model(key: str) -> FileResponse:
        manifest, path = from_cache(key)
        # The key names what the build was made from, so its model never changes: cache it hard.
        return FileResponse(
            path, media_type="model/gltf-binary", headers={"Cache-Control": "private, max-age=86400, immutable"}
        )

    @app.get("/saved")
    def saved() -> list[dict]:
        return builder.saved()

    @app.get("/saved/{face_id}/preset")
    def saved_preset(face_id: str, where: str = Query("preview", alias="from")) -> dict:
        if where not in PLACES:
            raise HTTPException(422, "from is preview or export")
        preset = builder.preset_in(face_id, where)
        if preset is None:
            raise HTTPException(404, f"no preset for {face_id!r} in the {where}")
        return preset.model_dump(mode="json", exclude_none=True)

    @app.post("/saved/{face_id}/rename")
    async def rename(face_id: str, change: Rename) -> dict:
        if change.id is None and change.name is None:
            raise HTTPException(422, "give a new id, a new name, or both")
        try:
            new_id = await builder.rename(face_id, new_id=change.id, name=change.name)
        except NoPreview as exc:
            raise HTTPException(404, str(exc)) from exc
        except (Busy, NotOwned, Taken) as exc:
            raise HTTPException(409, str(exc)) from exc
        except (BuildError, OSError, ValueError) as exc:
            raise HTTPException(500, f"rename failed: {exc}") from exc
        return {"id": new_id}

    @app.delete("/saved/{face_id}")
    async def delete(face_id: str, where: str | None = Query(None, alias="from")) -> dict:
        if where not in (None, *PLACES):
            raise HTTPException(422, "from is preview or export")
        try:
            trash = await builder.delete(face_id, (where,) if where else PLACES)
        except NoPreview as exc:
            raise HTTPException(404, str(exc)) from exc
        except (Busy, NotOwned) as exc:
            raise HTTPException(409, str(exc)) from exc
        except OSError as exc:
            raise HTTPException(500, f"delete failed: {exc}") from exc
        return {"trash": str(trash)}

    # ---- from the editor to a screen ------------------------------------------------------------

    async def on_server(face_id: str) -> dict:
        """What the server says about a face: is it up, does it list the face, which screens watch it."""
        status, health, why = await asyncio.to_thread(server_get, ed.server_url, "/health")
        up = status == 200 and isinstance(health, dict)
        out = {
            "server": ed.server_url,
            "viewer": ed.viewer,
            "server_ok": up,
            "reason": why if why or up else f"{ed.server_url}/health is not an argus-mask server's",
            "read_only": bool(up and health.get("read_only")),
            "listed": False,
            "screens": [],
        }
        if up:
            out["screens"] = (health.get("screens") or {}).get(face_id, [])
            listed, _, _ = await asyncio.to_thread(server_get, ed.server_url, f"/face/{face_id}")
            out["listed"] = listed == 200
        return out

    @app.get("/screen/{face_id}")
    async def screen(face_id: str) -> dict:
        if not _FACE_ID.match(face_id):
            raise HTTPException(404, f"{face_id!r} is not a face id")
        try:
            store.load(face_id)
            exported = True
        except FaceError:
            exported = False
        out = await on_server(face_id) | {"face": face_id, "exported": exported}
        if not exported:
            out["hint"] = "Only an exported face can go on a screen: Export it first. A preview is never served."
        elif not out["server_ok"]:
            out["hint"] = f"{out['reason']}. Start it with: argus-mask serve --assets-dir {cfg.assets_dir}"
        elif not out["listed"]:
            # The usual cause: the editor and the server were started on different assets directories.
            out["hint"] = (
                f"The server at {ed.server_url} does not list {face_id}. Is it serving the same assets directory "
                f"as the editor ({cfg.assets_dir})? Start it with: argus-mask serve --assets-dir {cfg.assets_dir}"
            )
        return out

    @app.get("/screens")
    async def screens() -> dict:
        """The server's screens per face, for the saved-faces cards."""
        status, health, why = await asyncio.to_thread(server_get, ed.server_url, "/health")
        if status != 200 or not isinstance(health, dict):
            return {"server_ok": False, "reason": why, "screens": {}}
        return {"server_ok": True, "reason": None, "screens": health.get("screens") or {}}

    @app.post("/screen/{face_id}/test")
    async def screen_test(face_id: str) -> dict:
        """A chime through the server's own source socket, as `push` sends one: its read-only and guards apply."""

        state = await screen(face_id)
        if not state["listed"]:
            raise HTTPException(404, state.get("hint") or f"the server does not list {face_id}")
        try:
            done = await client.push(ed.server_url, face_id, chime(), 24000)
        except client.ClientError as exc:
            raise HTTPException(409, str(exc)) from exc  # the server's own reason: read-only, a live source ...
        return {"ok": True, "screens": len(state["screens"]), "frames": done.get("frames")}

    # ---- broadcast: say and push from the page, as from a terminal --------------------------------

    app.state.broadcasts = broadcasts

    async def served(face_id: str) -> None:
        status, _, why = await asyncio.to_thread(server_get, ed.server_url, f"/face/{face_id}")
        if status != 200:
            raise HTTPException(
                404, why if status is None else f"the server at {ed.server_url} does not serve {face_id}"
            )

    @app.get("/broadcast/state")
    async def broadcast_state() -> dict:
        """The server's faces and who is watching them, the TTS defaults, and what is broadcasting from here."""
        status, health, why = await asyncio.to_thread(server_get, ed.server_url, "/health")
        out = {
            "server": ed.server_url,
            "server_ok": status == 200,
            "reason": why,
            "tts_url": cfg.tts_url,
            "tts_voice": cfg.tts_voice,
            "faces": [],
            "jobs": [j.public() for j in broadcasts.jobs.values() if j.state == "running"],
            "record": {"ok": (why_not := recording_ready()) is None, "reason": why_not, "dir": str(ed.recordings_dir)},
        }
        if status == 200 and isinstance(health, dict):
            _, faces, _ = await asyncio.to_thread(server_get, ed.server_url, "/faces")
            out |= {
                "read_only": bool(health.get("read_only")),
                "driver": health.get("driver") or {},
                "live": health.get("live") or [],
                "screens": health.get("screens") or {},
                "faces": [
                    {"id": f["id"], "name": f.get("name") or f["id"], "kind": (f.get("provenance") or {}).get("kind")}
                    for f in (faces or [])
                    if isinstance(f, dict) and f.get("id")
                ],
            }
        return out

    def chosen_tts(request: Request, given: str) -> str:
        """The TTS service a request names, or the configured one. A host the caller names is reached from here,
        so that takes the origin check the writes have: no other site may use the editor to probe the LAN."""
        given = given.strip()
        if given and given != cfg.tts_url:
            why = origin_refusal(request.scope, request.headers)
            if why is None and request.headers.get("sec-fetch-site") == "cross-site":
                why = "a cross-site request may not name a TTS service"
            if why is not None:
                raise HTTPException(403, why)
        return given or cfg.tts_url

    # -- the Voices page --------------------------------------------------------------------------------

    @app.get("/voice/state")
    def voice_state() -> dict:
        base = client.http_base(cfg.tts_url)
        return {
            "tts_url": cfg.tts_url,
            "can_make": base is not None,
            "reason": None
            if base is not None
            else "making a voice needs an HTTP TTS service with the /synthesise contract (Miso); set ARGUS_MASK_TTS_URL",
            "voices": voice_store.records(),
            "sources": list(SOURCES),
            "deliveries": list(DELIVERIES),
        }

    @app.post("/voice/check")
    async def voice_check(
        request: Request,
        name: str = Query("clip.wav", max_length=200),
        transcript: str = Query("", max_length=3000),
    ) -> dict:
        """A clip (the body, a WAV file) checked as a reference for a voice, and kept by its hash for making one."""
        too_big = HTTPException(413, f"send a WAV file of at most {MAX_AUDIO_BYTES // 2**20} MB")
        declared = request.headers.get("content-length") or "0"
        if not declared.isdigit() or int(declared) > MAX_AUDIO_BYTES:
            raise too_big
        buffer = bytearray()  # read up to the cap, never the whole of whatever is sent
        async for chunk in request.stream():
            buffer += chunk
            if len(buffer) > MAX_AUDIO_BYTES:
                raise too_big
        data = bytes(buffer)
        try:
            samples, rate = await asyncio.to_thread(wav_from_bytes, data, name)
        except AudioError as exc:
            raise HTTPException(422, str(exc)) from exc
        report = await asyncio.to_thread(analyse, samples, rate, transcript)
        sha = await asyncio.to_thread(voice_store.stage, data)
        return {"sha": sha, "name": name, "report": report.public()}

    @app.get("/voice/clip/{sha}")
    def voice_clip(sha: str) -> FileResponse:
        try:
            return FileResponse(voice_store.clip(sha), media_type="audio/wav")
        except VoiceError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.post("/voice/assess")
    def voice_assess(clips: list[dict]) -> dict:
        """The set of checked clips (their reports, as /voice/check gave them): enough speech, and one voice."""
        try:
            return assess(clips[:5]).public()
        except (KeyError, TypeError, AttributeError, ValueError) as exc:
            raise HTTPException(422, "send the reports /voice/check gave") from exc

    @app.post("/voice/generate")
    async def voice_generate(req: VoiceGenerate, request: Request) -> dict:
        tts = chosen_tts(request, req.tts_url)
        if not tts.startswith(("http://", "https://", "wyoming://")):
            raise HTTPException(422, "a TTS service is needed: set ARGUS_MASK_TTS_URL")
        body: dict = {"text": req.script}
        if req.description.strip():
            body["voice_description"] = req.description.strip()
        else:
            body = client.voiced(body, req.speaker.strip() or "0", tts)
        if req.instruction.strip():
            body["instruction"] = req.instruction.strip()
        if req.temperature is not None:
            body["temperature"] = req.temperature
        try:
            job = generations.start(tts, body)
        except VoiceError as exc:
            raise HTTPException(409, str(exc)) from exc
        log.info("voice_generate", chars=len(req.script))  # never the script
        return job.public()

    @app.get("/voice/generate/{job_id}")
    def voice_generation(job_id: str) -> dict:
        job = generations.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "no such candidate")
        return job.public()

    @app.post("/voice/create")
    async def voice_create(req: VoiceCreate, request: Request) -> dict:
        """Make a voice on the TTS service from checked clips, and keep its record: where it came from, and the clips."""
        async with making:
            return await make_voice(req, request)

    async def make_voice(req: VoiceCreate, request: Request) -> dict:
        tts = chosen_tts(request, req.tts_url)
        if client.http_base(tts) is None:
            raise HTTPException(422, "making a voice needs an HTTP TTS service with the /synthesise contract (Miso)")
        try:
            source = provenance(req.source)
            paths = [voice_store.clip(c.sha) for c in req.clips]
        except VoiceError as exc:
            raise HTTPException(422, str(exc)) from exc
        if voice_store.record(req.id) is not None:
            raise HTTPException(409, f"there is already a voice called {req.id}: delete it first, or pick another name")
        existing = await asyncio.to_thread(client.tts_voices, tts)
        if existing and any(v["name"] == f"clone:{req.id}" for v in existing):
            raise HTTPException(409, f"the TTS service already has a voice called {req.id}: pick another name")
        reports = []
        for clip, path in zip(req.clips, paths, strict=True):
            samples, rate = await asyncio.to_thread(read_wav, path)
            reports.append(
                (await asyncio.to_thread(analyse, samples, rate, clip.transcript, generated=clip.generated)).public()
            )
        for clip, report in zip(req.clips, reports, strict=True):
            if report["verdict"] == "not suitable":
                bad = "; ".join(c["label"] for c in report["checks"] if c["verdict"] == "bad")
                raise HTTPException(422, f"{clip.name} is not suitable ({bad}): see its checks")
        overall = assess(reports).public()
        if overall["verdict"] == "not suitable":
            raise HTTPException(422, "; ".join(c["advice"] for c in overall["checks"] if c["verdict"] == "bad"))
        transcript = " ".join(c.transcript.strip() for c in req.clips)
        clips = [(c.name, await asyncio.to_thread(path.read_bytes)) for c, path in zip(req.clips, paths, strict=True)]
        try:
            made = await asyncio.to_thread(client.clone_voice, tts, req.id, req.name, clips, transcript)
        except client.ClientError as exc:
            raise HTTPException(502, str(exc)) from exc
        record = {
            "id": req.id,
            "name": req.name,
            "delivery": req.delivery,
            "tts": client.http_base(tts),
            "created": time.time(),
            "source": source,
            "clips": [
                {
                    "file": f"clip-{i + 1}.wav",
                    "name": c.name,
                    "sha256": c.sha,
                    "transcript": c.transcript,
                    "seconds": r["metrics"]["seconds"],
                    "verdict": r["verdict"],
                }  # fmt: skip
                for i, (c, r) in enumerate(zip(req.clips, reports, strict=True))
            ],
            "check": {
                "verdict": overall["verdict"],
                "score": overall["score"],
                "speech_seconds": overall["metrics"]["speech_seconds"],
            },  # noqa: E501
            "service": {k: made.get(k) for k in ("voice_clone_id", "total_seconds", "engine") if k in made},
        }
        await asyncio.to_thread(voice_store.write, record, paths)
        log.info("voice_made", voice=req.id, source=source["kind"], delivery=req.delivery)
        return record

    @app.post("/voice/try")
    async def voice_try(req: VoiceTry, request: Request) -> Response:
        """A line in a voice, as a WAV for the browser to play: no face involved."""
        tts = chosen_tts(request, req.tts_url)
        if not tts.startswith(("http://", "https://", "wyoming://")):
            raise HTTPException(422, "a TTS service is needed: set ARGUS_MASK_TTS_URL")
        body = client.voiced(
            {"text": req.text, **({"temperature": req.temperature} if req.temperature else {})}, req.voice, tts
        )
        try:
            pcm, fmt, _ = await client.synthesise(tts, body, tidy=req.tidy)
        except client.ClientError as exc:
            raise HTTPException(502, str(exc)) from exc
        from argus_mask.editor.voices import pcm_to_float  # noqa: PLC0415

        return Response(wav_bytes(pcm_to_float(pcm, fmt), fmt.sample_rate), media_type="audio/wav")

    @app.delete("/voice/{voice_id}")
    async def voice_delete(voice_id: str, request: Request, tts_url: str = Query("", max_length=500)) -> dict:
        """Remove a voice made here: from the TTS service, and its record to the trash. Only voices with a record."""
        record = voice_store.record(voice_id)
        if record is None:
            raise HTTPException(404, "only a voice made on this page (one with a record here) can be deleted here")
        # The service the voice was made on, unless the caller names one.
        made_on = record.get("tts")
        if tts_url.strip() or not isinstance(made_on, str) or not made_on:
            tts = chosen_tts(request, tts_url)
        else:
            tts = f"{made_on}/synthesise"
        removed = False
        if client.http_base(tts) is not None:
            try:
                await asyncio.to_thread(client.delete_voice, tts, voice_id)
                removed = True
            except client.ServiceRefused as exc:
                if exc.status != 404:  # gone already: the record still goes
                    raise HTTPException(502, str(exc)) from exc
            except client.ClientError as exc:
                raise HTTPException(502, str(exc)) from exc
        voice_store.trash(voice_id)
        return {"id": voice_id, "removed_from_tts": removed}

    @app.get("/voices")
    async def voices(request: Request, tts: str = Query("", max_length=500)) -> dict:
        """The voices the TTS service offers, if it says (a Wyoming server does; the HTTP contract cannot).

        A host the caller names is reached from here, so a read though it is, that takes the origin check
        the writes have: no other site may use the editor to probe the LAN.
        """
        tts = chosen_tts(request, tts)
        if not tts:
            return {"tts_url": tts, "listed": False, "voices": [], "reason": "no TTS service set"}
        try:
            found = await asyncio.to_thread(client.tts_voices, tts)
        except client.ClientError as exc:
            return {"tts_url": tts, "listed": False, "voices": [], "reason": str(exc)}
        if found is None:
            return {"tts_url": tts, "listed": False, "voices": [], "reason": "this service does not list its voices"}
        return {"tts_url": tts, "listed": True, "voices": found}

    @app.post("/broadcast/say")
    async def broadcast_say(req: Say) -> dict:
        tts_url = (req.tts_url or cfg.tts_url).strip()
        if not tts_url.startswith(("http://", "https://", "wyoming://")):
            raise HTTPException(422, "a TTS service is needed: an http(s):// or wyoming:// URL (or ARGUS_MASK_TTS_URL)")
        await served(req.face)
        body = client.voiced({**req.params, "text": req.text}, req.voice or cfg.tts_voice, tts_url)
        if req.record is not None and (why_not := recording_ready()) is not None:
            raise HTTPException(422, why_not)
        try:
            job = broadcasts.say(
                req.face,
                tts_url,
                body,
                seed=req.seed,
                stream=req.stream,
                pause_ms=req.pause,
                tidy=req.tidy,
                context=req.context,
                record=req.record.options() if req.record else None,
            )
        except Broadcasting as exc:
            raise HTTPException(409, str(exc)) from exc
        log.info("broadcast", face=req.face, kind="say", chars=len(req.text))  # never the text itself
        return job.public()

    @app.post("/broadcast/push")
    async def broadcast_push(
        request: Request,
        face: str = Query(..., pattern=FACE_ID_PATTERN),
        seed: int = Query(0, ge=0, le=2**31 - 1),
        name: str = Query("audio.wav", max_length=200),
        record: bool = Query(False, description="also record the broadcast to a file"),
        record_size: RecordSize = Query("1280x720"),
        record_quality: RecordQuality = Query("high"),
    ) -> dict:
        """An audio file (WAV) for a face, as `argus-mask push` plays one: the request's body is the file."""
        if record and (why_not := recording_ready()) is not None:
            raise HTTPException(422, why_not)
        too_big = HTTPException(413, f"send a WAV file of at most {MAX_AUDIO_BYTES // 2**20} MB")
        declared = request.headers.get("content-length") or "0"
        if not declared.isdigit() or int(declared) > MAX_AUDIO_BYTES:
            raise too_big
        data = bytearray()  # read up to the cap, never the whole of whatever is sent
        async for chunk in request.stream():
            data += chunk
            if len(data) > MAX_AUDIO_BYTES:
                raise too_big
        if not data:
            raise HTTPException(422, f"send a WAV file of at most {MAX_AUDIO_BYTES // 2**20} MB")
        await served(face)
        try:
            mono, rate = await asyncio.to_thread(wav_from_bytes, bytes(data), name)
        except AudioError as exc:
            raise HTTPException(422, str(exc)) from exc
        try:
            job = broadcasts.push(
                face,
                mono,
                rate,
                seed=seed,
                name=name,
                record=Record(size=record_size, quality=record_quality).options() if record else None,
            )
        except Broadcasting as exc:
            raise HTTPException(409, str(exc)) from exc
        log.info("broadcast", face=face, kind="push", seconds=round(len(mono) / rate, 2))
        return job.public()

    @app.get("/recordings")
    def recordings() -> list[dict]:
        """The clips recorded here, newest first."""
        root = ed.recordings_dir
        if not root.is_dir():
            return []
        clips = []
        for p in root.iterdir():
            if p.suffix in (".mp4", ".webm") and _RECORDING.match(p.name):
                with contextlib.suppress(OSError):  # gone, or not a file, between the listing and the look
                    if p.is_file():
                        clips.append((p.name, p.stat()))
        clips.sort(key=lambda c: c[1].st_mtime, reverse=True)
        return [{"name": name, "bytes": st.st_size, "modified": st.st_mtime} for name, st in clips]

    @app.get("/recordings/{name}")
    def recording(name: str) -> FileResponse:
        """One recorded file: a plain name in the recordings directory, nothing else."""
        path = ed.recordings_dir / name
        if not _RECORDING.match(name) or not path.is_file() or path.resolve().parent != ed.recordings_dir.resolve():
            raise HTTPException(404, "no such recording")
        return FileResponse(path, filename=name)

    @app.get("/broadcast/{job_id}")
    def broadcast_job(job_id: str) -> dict:
        job = broadcasts.get(job_id)
        if job is None:
            raise HTTPException(404, "no such broadcast")
        return job.public()

    @app.post("/broadcast/{job_id}/stop")
    def broadcast_stop(job_id: str) -> dict:
        if not broadcasts.stop(job_id):
            raise HTTPException(404, "no such broadcast running")
        return {"stopping": job_id}

    @app.get("/look.js")
    def look() -> FileResponse:
        # The viewer's lighting and materials, so the editor's previews look as the face will on a screen.
        return FileResponse(VIEWER_DIR / "look.js", media_type="text/javascript")

    # three.js is the viewer's vendored copy: one copy in the package.
    app.mount("/vendor", StaticFiles(directory=VIEWER_DIR / "vendor"), name="vendor")
    app.mount("/", StaticFiles(directory=PAGE_DIR, html=True), name="page")
    return app
