"""The save step of a guided face (#176; step 7 of #163): write the talking face the server serves.

Copies the picked LAM head into ``<assets>/<id>/``, with a manifest (limits, framing, provenance),
the fitted rest set, the scene and shot, and the build record beside the face as the editor's ownership
mark (``build.json``, in place of ``preset.json`` for guided heads). Refuses a record that is not
settled. A look-alike check is optional on save; without one a cleared face is not ``demo_ready``.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

import structlog
from pydantic import ValidationError

from argus_mask.backgrounds import SceneChoices
from argus_mask.editor.builder import PRESET_NAME, NotOwned, Taken
from argus_mask.editor.guided import (
    RECORD_NAME,
    Artifact,
    Check,
    GuidedError,
    GuidedStore,
    NotReady,
    StepResult,
)
from argus_mask.expressions import Expressions
from argus_mask.faces import MANIFEST_NAME, SPLAT_FRAMING, FaceError, FaceStore, build_manifest, sha256_file
from argus_mask.models import Generator, Limits, LookAlike, Provenance, RestSet, is_face_id

log = structlog.get_logger(__name__)

STEP = "save"
BUILD_MARK = RECORD_NAME  # ownership mark for a guided face (#176)


class SaveError(GuidedError):
    """Save cannot run now, with the reason."""


def owned_guided(assets_dir: Path, face_id: str) -> bool:
    """Whether ``<assets>/<id>`` is a guided face the editor saved (``build.json`` beside the model)."""
    return is_face_id(face_id) and (assets_dir / face_id / BUILD_MARK).is_file()


def editor_owned(assets_dir: Path, face_id: str) -> bool:
    """Whether the editor may replace ``<assets>/<id>``: a mesh export (``preset.json``) or a guided save."""
    if not is_face_id(face_id):
        return False
    return (assets_dir / face_id / PRESET_NAME).is_file() or owned_guided(assets_dir, face_id)


def _provenance(source: StepResult, build: StepResult, lookalike: LookAlike | None, notes: str) -> Provenance:
    """Provenance for the person: the portrait's record, and every part the head depends on."""
    data = source.data if isinstance(source.data, dict) else {}
    kind = str(data.get("kind") or "personal")
    if kind not in ("personal", "stock", "synthetic", "consented"):
        kind = "personal"
    source_text = str(data.get("source") or "").strip() or "guided build: portrait source not stated"
    license_ = str(data.get("license") or "").strip()
    if kind != "personal" and not license_:
        license_ = "see build record"
    generator = None
    raw_gen = data.get("generator")
    if isinstance(raw_gen, dict) and kind in ("synthetic", "personal"):
        with contextlib.suppress(ValidationError):
            generator = Generator.model_validate(raw_gen)
    parts = [f"{tool.name} ({tool.license})" if tool.license else tool.name for tool in build.tools]
    body = notes.strip()
    if parts:
        credit = "; ".join(parts)
        body = f"{body} Parts: {credit}.".strip() if body else f"Parts: {credit}."
    release = data.get("release_sha256")
    return Provenance(
        kind=kind,  # type: ignore[arg-type]
        source=source_text,
        license=license_,
        notes=body,
        subject=str(data.get("subject") or ""),
        release_sha256=release if isinstance(release, str) else None,
        generator=generator,
        lookalike=lookalike,
    )


def _limits(turns: StepResult | None) -> Limits | None:
    if turns is None or not isinstance(turns.data, dict):
        return None
    raw = turns.data.get("limits")
    if not isinstance(raw, dict):
        return None
    with contextlib.suppress(ValidationError):
        return Limits.model_validate(raw)
    return None


def _rest(rest: StepResult | None, face_id: str) -> RestSet | None:
    if rest is None or not isinstance(rest.data, dict):
        return None
    weights = rest.data.get("weights")
    if not isinstance(weights, dict) or not weights:
        return None
    name = str(rest.data.get("name") or f"fit-{face_id}")
    if not is_face_id(name):
        name = f"fit-{face_id}"
    life = rest.data.get("life", 1.0)
    with contextlib.suppress(ValidationError, TypeError, ValueError):
        return RestSet(name=name, weights={str(k): float(v) for k, v in weights.items()}, life=float(life))
    return None


def save(
    store: GuidedStore,
    *,
    assets_dir: Path,
    expressions: Expressions,
    scenes: SceneChoices,
    build_id: str,
    name: str = "",
    lookalike: LookAlike | None = None,
    notes: str = "",
) -> StepResult:
    """Write the picked head into the assets root. Needs settled pick and scene."""
    record = store.load(build_id)
    if blocked := record.blockers("save"):
        raise NotReady("save cannot run: " + "; ".join(blocked))
    candidate = record.picked
    if not candidate:
        raise SaveError("nothing is picked")
    face_id = build_id
    if not is_face_id(face_id):
        raise SaveError(f"{face_id!r} is not a face id")

    build = record.get("build", candidate)
    source = record.get("source", candidate)
    if build is None or source is None:
        raise SaveError(f"{candidate} has no build or source")
    head = next((a for a in build.outputs if a.name == "head" and a.file), None)
    if head is None:
        raise SaveError(f"{candidate} has no head")
    head_path = store.path(build_id, head.file)
    if not head_path.is_file():
        raise SaveError("the head is missing on disk")
    if sha256_file(head_path) != head.sha256:
        raise SaveError("the head changed since the build: run the build again")

    target = assets_dir / face_id
    if target.exists() and not editor_owned(assets_dir, face_id):
        raise NotOwned(f"the editor did not make {face_id}: it has no preset.json or build.json")
    if target.exists() and (target / PRESET_NAME).is_file() and not owned_guided(assets_dir, face_id):
        raise Taken(f"{face_id} is a mesh face the editor made: save a guided head under another id")

    prov = _provenance(source, build, lookalike, notes)
    limits = _limits(record.get("turns", candidate))
    rest = _rest(record.get("rest", candidate), face_id)
    scene_result = record.get("scene")
    scene_data = scene_result.data if scene_result and isinstance(scene_result.data, dict) else {}
    scene_name = scene_data.get("scene") if isinstance(scene_data.get("scene"), str) else None
    shot = scene_data.get("shot") if isinstance(scene_data.get("shot"), str) else None

    work = Path(tempfile.mkdtemp(prefix=f".save-{face_id}-", dir=assets_dir))
    installed = False
    try:
        model_name = f"{face_id}.zip"
        shutil.copy2(head_path, work / model_name)
        portrait = next((a for a in source.outputs if a.name == "portrait" and a.file), None)
        if portrait is not None:
            src_dir = work / "source"
            src_dir.mkdir()
            src_path = store.path(build_id, portrait.file)
            if src_path.is_file():
                shutil.copy2(src_path, src_dir / "portrait.png")
        try:
            manifest = build_manifest(
                work / model_name,
                face_id=face_id,
                name=name.strip() or record.name or face_id,
                provenance=prov,
                framing=SPLAT_FRAMING,
            )
        except (FaceError, ValidationError) as exc:
            raise SaveError(str(exc)) from exc
        if limits is not None:
            manifest = manifest.model_copy(update={"limits": limits})
        (work / MANIFEST_NAME).write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
        shutil.copy2(store.directory(build_id) / RECORD_NAME, work / BUILD_MARK)

        assets_dir.mkdir(parents=True, exist_ok=True)
        staged = assets_dir / f".install-{face_id}-{uuid.uuid4().hex}"
        work.rename(staged)
        work = staged
        if target.exists():
            trash = assets_dir / ".trash" / f"{face_id}-{uuid.uuid4().hex}"
            trash.parent.mkdir(parents=True, exist_ok=True)
            target.rename(trash)
        os.replace(staged, target)
        installed = True
        work = Path()
        FaceStore(assets_dir).load(face_id)
    except Exception:
        if not installed and work.exists():
            shutil.rmtree(work, ignore_errors=True)
        raise

    if rest is not None:
        expressions.define(rest)
        expressions.choose(face_id, rest.name)
    scenes.choose(
        face_id,
        scene_name if isinstance(scene_name, str) else None,
        shot if isinstance(shot, str) else None,
    )

    face_art = Artifact(
        name="face", sha256=sha256_file(target / f"{face_id}.zip"), bytes=(target / f"{face_id}.zip").stat().st_size
    )
    checks = [Check(name="inspect", verdict="passed")]
    if prov.lookalike is not None:
        checks.append(Check(name="lookalike", verdict="passed"))
    elif prov.cleared:
        checks.append(
            Check(
                name="lookalike",
                verdict="advisory",
                reason="no look-alike check: the face is cleared but not demo_ready",
            )
        )
    else:
        checks.append(Check(name="lookalike", verdict="passed", reason="not required for a personal face"))

    data: dict[str, Any] = {
        "face": face_id,
        "scene": scene_name,
        "shot": shot,
        "demo_ready": prov.demo_ready,
    }
    kept = store.put(
        build_id,
        StepResult(step=STEP, checks=checks, outputs=[face_art], data=data),
    )
    log.info("guided_save", build=build_id, face=face_id, demo_ready=prov.demo_ready)
    return kept
