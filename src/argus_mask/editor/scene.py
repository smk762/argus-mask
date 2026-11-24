"""The scene step of a guided face (#175; step 6 of #163): background and shot, with a bright-floor warning.

Reuses the editor's scenes list and shot values (close / wide). The choice is kept in the build record;
save (#176) is what writes it into ``scenes.json`` for screens. A plate whose floor band behind the bust
is very bright gets an advisory (garden's pale path); a procedural scene or a plain background has nothing
to measure.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import structlog

from argus_mask.backgrounds import BUILT_IN_SCENES, SHOTS, VIEWER_SCENES, PlateError, PlateStore
from argus_mask.bright_floor import BRIGHT_SHARE, BrightFloor, measure_path
from argus_mask.editor.guided import Check, GuidedError, GuidedStore, NotReady, StepResult
from argus_mask.models import SCENE_NAME_PATTERN

log = structlog.get_logger(__name__)

STEP = "scene"
CHECK = "bright-floor"


class SceneError(GuidedError):
    """The scene step cannot run now, with the reason."""


def plate_png(name: str, plates: PlateStore) -> Path | None:
    """Where a named scene's plate PNG lives, or None if it has none (procedural / unknown)."""
    if not re.fullmatch(SCENE_NAME_PATTERN, name):  # before it is part of any path
        return None
    built = VIEWER_SCENES / name / "plate.png"
    if built.is_file():
        return built
    try:
        _manifest, directory = plates.load(name)
    except PlateError:
        return None
    path = directory / "plate.png"
    return path if path.is_file() else None


def bright_floor_check(report: BrightFloor | None) -> Check:
    """The check written into the scene step: advisory when the floor is pale, else passed (or skipped)."""
    if report is None:
        return Check(name=CHECK, verdict="passed", reason="no plate to measure")
    measure = {
        "floor_mean": report.mean,
        "floor_p95": report.p95,
        "floor_bright_share": report.bright_share,
        "threshold": report.threshold,
    }
    if report.bright:
        return Check(
            name=CHECK,
            verdict="advisory",
            reason=(
                f"the floor behind the bust is bright "
                f"(pale share {report.bright_share:.2%} ≥ {report.threshold:.0%}): "
                "a dark bust fading into it can read as a grey band"
            ),
            measure=measure,
        )
    return Check(name=CHECK, verdict="passed", measure=measure)


def choose(
    store: GuidedStore,
    plates: PlateStore,
    build_id: str,
    *,
    scene: str | None,
    shot: Literal["close", "wide"] | None,
    threshold: float = BRIGHT_SHARE,
) -> StepResult:
    """Record the background and shot for this build. Needs a settled pick."""
    known = {s.name for s in plates.scenes()}
    if scene is not None and scene not in known:
        raise SceneError(f"{scene!r} is not a scene: choose from {', '.join(sorted(known))} or none")
    if shot is not None and shot not in SHOTS:
        raise SceneError(f"{shot!r} is not a shot: choose from {', '.join(SHOTS)} or none")

    report: BrightFloor | None = None
    if scene:
        path = plate_png(scene, plates)
        if path is not None:
            try:
                report = measure_path(path, threshold=threshold)
            except (OSError, ValueError) as exc:
                raise SceneError(f"{scene}: could not score its plate: {exc}") from exc
        elif scene not in BUILT_IN_SCENES:
            raise SceneError(f"{scene}: has no plate.png to score")

    checks = [Check(name="choice", verdict="passed"), bright_floor_check(report)]
    data: dict = {"scene": scene, "shot": shot}
    scores = report.as_scores() if report else {}
    try:
        kept = store.put(
            build_id,
            StepResult(step=STEP, checks=checks, scores=scores, data=data),
        )
    except NotReady:
        raise
    except GuidedError:
        raise
    log.info(
        "guided_scene",
        build=build_id,
        scene=scene,
        shot=shot,
        bright=bool(report and report.bright),
    )
    return kept
