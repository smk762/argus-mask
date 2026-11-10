"""The face editor's HTTP app.

``GET  /health``                        whether a build can run, and why not; whether one is running
``GET  /library``                       the installed, licensed assets, by part (and ``clothes``, with a ``slot``)
``GET  /library/{part}/{name}/thumb``   an asset's thumbnail (PNG)
``GET  /modifiers``                     the shape sliders a preset may set (MakeHuman targets), by group
``GET  /presets``                       the faces the editor made: each preset, and whether it is previewed, exported
``POST /build``                         build a preset into a preview; answers when it is done (about 10 s)
``POST /export/{id}``                   copy a preview into the assets root, where the server serves it
``GET  /face/{id}``, ``/face/{id}/model``  a face the editor made (its preview, else its export), for the page
``GET  /``                              the editor page

``POST /build`` starts Blender, so it is behind the same origin and read-only
guards as the server's writes. A preset names assets by reference, and every
reference is looked up in the installed library: no path from the request
reaches the file system or Blender's command line except through that lookup.
"""

from __future__ import annotations

from pathlib import Path

import structlog
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from argus_mask import __version__
from argus_mask.config import EditorSettings, Settings
from argus_mask.editor.builder import Builder, BuildError, Busy, NoPreview, NotOwned, Refused
from argus_mask.editor.library import CLOTHES, AssetLibrary, LibraryError
from argus_mask.editor.preset import MODIFIER_GROUPS, MODIFIERS, PART_DIRS, Preset, one_way
from argus_mask.faces import FaceError, FaceStore
from argus_mask.models import FaceManifest
from argus_mask.server.app import VIEWER_DIR, install_guards

log = structlog.get_logger(__name__)

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
    )
    store = FaceStore(cfg.assets_dir)
    previews = FaceStore(builder.preview_dir)
    app = FastAPI(title="Argus Mask face editor", version=__version__)
    app.state.builder = builder
    install_guards(app, cfg)

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
                }
                for m in MODIFIERS.values()
            ],
        }

    @app.get("/presets")
    def presets() -> list[dict]:
        return [{**e, "preset": e["preset"].model_dump(mode="json", exclude_none=True)} for e in builder.presets()]

    def owned_face(face_id: str) -> tuple[FaceManifest, Path]:
        """The editor's preview of a face, else its exported copy. Never a face it did not make."""
        for made, faces in ((builder.previewed, previews), (builder.owned, store)):
            if made(face_id):
                try:
                    return faces.load(face_id)
                except FaceError as exc:
                    raise HTTPException(404, str(exc)) from exc
        raise HTTPException(404, f"the editor made no face {face_id!r}")

    @app.get("/face/{face_id}", response_model=FaceManifest)
    def face(face_id: str) -> FaceManifest:
        return owned_face(face_id)[0]

    @app.get("/face/{face_id}/model")
    def face_model(face_id: str) -> FileResponse:
        manifest, path = owned_face(face_id)
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

    # three.js is the viewer's vendored copy: one copy in the package.
    app.mount("/vendor", StaticFiles(directory=VIEWER_DIR / "vendor"), name="vendor")
    app.mount("/", StaticFiles(directory=PAGE_DIR, html=True), name="page")
    return app
