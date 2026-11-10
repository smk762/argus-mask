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
``GET  /saved``                         every face in the assets root and every preview, newest first
``GET  /saved/{id}/preset``             the preset behind a face the editor made (``?from=preview|export``)
``POST /saved/{id}/rename``             ``{"id": new id, "name": new name}``: the export and the preview together
``DELETE /saved/{id}``                  move a face the editor made to ``.trash/`` (``?from=preview|export``, default both)
``GET  /faces.html``                    the saved-faces page
``GET  /``                              the editor page

``POST /build`` starts Blender, so it is behind the same origin and read-only
guards as the server's writes. A preset names assets by reference, and every
reference is looked up in the installed library: no path from the request
reaches the file system or Blender's command line except through that lookup.
"""

from __future__ import annotations

from pathlib import Path

import structlog
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from argus_mask import __version__
from argus_mask.config import EditorSettings, Settings
from argus_mask.editor.builder import PLACES, Builder, BuildError, Busy, NoPreview, NotOwned, Refused, Taken
from argus_mask.editor.library import CLOTHES, AssetLibrary, LibraryError
from argus_mask.editor.preset import MODIFIER_GROUPS, MODIFIERS, PART_DIRS, Preset, one_way
from argus_mask.faces import FaceError, FaceStore
from argus_mask.models import FACE_ID_PATTERN, FaceManifest
from argus_mask.server.app import VIEWER_DIR, install_guards

log = structlog.get_logger(__name__)


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

    # three.js is the viewer's vendored copy: one copy in the package.
    app.mount("/vendor", StaticFiles(directory=VIEWER_DIR / "vendor"), name="vendor")
    app.mount("/", StaticFiles(directory=PAGE_DIR, html=True), name="page")
    return app
