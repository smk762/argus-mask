"""Rest sets: named baseline expressions a face holds at rest and speaks from.

A face at rest with every weight at zero looks stern on most heads (the mouth's corners sit low), so a
face shows a rest set instead: ``warm`` unless chosen otherwise. Built-in sets are below; more, and each
face's choice, are kept in ``<assets>/expressions.json``, written by the editor and read by the server
(whose assets mount may be read-only), like ``aliases.json``::

    {"sets": {"cheery": {"weights": {"mouthSmileLeft": 0.5, "mouthSmileRight": 0.5}, "life": 1.2}},
     "faces": {"ginger-3": "calm"}}

A set in the file may replace a built-in one of the same name. Entries that are not sound (an unknown
shape, a weight outside 0 to 1, a face choosing a set that does not exist) are ignored, and ``problems``
says why.
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path

from argus_mask.arkit import ARKIT_52
from argus_mask.models import FACE_ID_PATTERN, RestSet

EXPRESSIONS_NAME = "expressions.json"
DEFAULT_SET = "warm"
_NAME = re.compile(FACE_ID_PATTERN)
_SHAPES = frozenset(ARKIT_52)

# Tuned by eye on ginger-3 (a smile under about 0.3 hardly reads on an MPFB head). A little lid closure
# takes the stare off; the brows' inner ends lifted a touch read as attentive rather than cross.
BUILTIN: dict[str, RestSet] = {
    s.name: s
    for s in (
        RestSet(name="neutral", weights={}, life=1.0),
        RestSet(
            name="calm",
            weights={
                "mouthSmileLeft": 0.18,
                "mouthSmileRight": 0.18,
                "eyeBlinkLeft": 0.12,
                "eyeBlinkRight": 0.12,
                "browInnerUp": 0.04,
            },  # fmt: skip
            life=0.7,
        ),
        RestSet(
            name="warm",
            weights={
                "mouthSmileLeft": 0.32,
                "mouthSmileRight": 0.32,
                "cheekSquintLeft": 0.12,
                "cheekSquintRight": 0.12,
                "eyeSquintLeft": 0.05,
                "eyeSquintRight": 0.05,
                "eyeBlinkLeft": 0.10,
                "eyeBlinkRight": 0.10,
                "browInnerUp": 0.06,
            },  # fmt: skip
            life=1.0,
        ),
        RestSet(
            name="bright",
            weights={
                "mouthSmileLeft": 0.48,
                "mouthSmileRight": 0.48,
                "cheekSquintLeft": 0.20,
                "cheekSquintRight": 0.20,
                "eyeSquintLeft": 0.08,
                "eyeSquintRight": 0.08,
                "browInnerUp": 0.08,
                "browOuterUpLeft": 0.10,
                "browOuterUpRight": 0.10,
            },  # fmt: skip
            life=1.3,
        ),
        RestSet(
            name="attentive",
            weights={
                "mouthSmileLeft": 0.12,
                "mouthSmileRight": 0.12,
                "browInnerUp": 0.12,
                "browOuterUpLeft": 0.06,
                "browOuterUpRight": 0.06,
                "eyeWideLeft": 0.05,
                "eyeWideRight": 0.05,
            },  # fmt: skip
            life=1.1,
        ),
    )
}


class ExpressionError(ValueError):
    """A rest set or a choice that may not be made, with the reason."""


def check_set(name: str, weights: object, life: object = 1.0) -> RestSet:
    """A rest set from untrusted input, or `ExpressionError` saying what is wrong with it."""
    if not _NAME.match(name):
        raise ExpressionError(f"{name!r} is not a usable name (lowercase letters, digits, - and _)")
    if not isinstance(weights, dict):
        raise ExpressionError("weights must be an object: shape name -> weight")
    clean: dict[str, float] = {}
    for shape, value in weights.items():
        if shape not in _SHAPES:
            raise ExpressionError(f"{shape!r} is not an ARKit blendshape")
        if isinstance(value, bool) or not isinstance(value, int | float) or not 0.0 <= float(value) <= 1.0:
            raise ExpressionError(f"{shape}: a weight is a number from 0 to 1")
        if float(value):
            clean[str(shape)] = round(float(value), 4)
    if isinstance(life, bool) or not isinstance(life, int | float) or not 0.0 <= float(life) <= 2.0:
        raise ExpressionError("life is a number from 0 (still) to 2")
    return RestSet(name=name, weights=dict(sorted(clean.items())), life=float(life))


class Expressions:
    """The rest sets under one assets root, and which one each face shows."""

    def __init__(self, root: str | Path) -> None:
        self.path = Path(root) / EXPRESSIONS_NAME
        self._seen: tuple[int, int, int] | None = None
        self._raw: dict = {}
        self._lock = threading.Lock()  # the editor's routes run in threads: one read-modify-write at a time

    def raw(self) -> dict:
        """The file as written: {"sets": {...}, "faces": {...}}, valid or not."""
        try:
            stat = self.path.stat()
        except OSError:
            self._seen, self._raw = None, {}
            return {"sets": {}, "faces": {}}
        key = (stat.st_ino, stat.st_mtime_ns, stat.st_size)  # every write replaces the file: a new inode
        if key != self._seen:
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                self._raw = data if isinstance(data, dict) else {}
            except (OSError, ValueError):
                self._raw = {}
            self._seen = key
        sets = self._raw.get("sets")
        faces = self._raw.get("faces")
        return {
            "sets": dict(sets) if isinstance(sets, dict) else {},
            "faces": {str(k): str(v) for k, v in faces.items()} if isinstance(faces, dict) else {},
        }

    def _own(self, raw: dict) -> tuple[dict[str, RestSet], dict[str, str]]:
        good, problems = {}, {}
        for name, entry in raw["sets"].items():
            try:
                if not isinstance(entry, dict):
                    raise ExpressionError("a set is an object with weights and life")
                good[str(name)] = check_set(str(name), entry.get("weights", {}), entry.get("life", 1.0))
            except ExpressionError as exc:
                problems[f"set {name}"] = str(exc)
        return good, problems

    def sets(self, raw: dict | None = None) -> dict[str, RestSet]:
        """Every usable set: the built-in ones, then the file's (which may replace them), by name."""
        own, _ = self._own(self.raw() if raw is None else raw)
        return dict(sorted({**BUILTIN, **own}.items()))

    def choices(self, raw: dict | None = None, sets: dict[str, RestSet] | None = None) -> dict[str, str]:
        """Face id -> the set it shows, for the faces that chose a usable one."""
        raw = self.raw() if raw is None else raw
        sets = sets if sets is not None else self.sets(raw)
        return {f: s for f, s in sorted(raw["faces"].items()) if s in sets}

    def problems(self) -> dict[str, str]:
        raw = self.raw()
        own, problems = self._own(raw)
        sets = {**BUILTIN, **own}
        for face, name in raw["faces"].items():
            if name not in sets:
                problems[f"face {face}"] = f"no rest set {name!r}"
        return problems

    def for_face(self, face_id: str, override: str | None = None) -> RestSet:
        """The set *face_id* shows: *override* if given (and it exists), else its choice, else the default."""
        raw = self.raw()
        sets = self.sets(raw)
        if override:
            if override not in sets:
                raise ExpressionError(f"no rest set {override!r}: there are {', '.join(sets)}")
            return sets[override]
        return sets.get(self.choices(raw, sets).get(face_id, DEFAULT_SET), sets.get(DEFAULT_SET, BUILTIN["neutral"]))

    def _write(self, raw: dict) -> None:
        tmp = self.path.with_name(EXPRESSIONS_NAME + ".part")
        body = {"sets": dict(sorted(raw["sets"].items())), "faces": dict(sorted(raw["faces"].items()))}
        tmp.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
        tmp.replace(self.path)

    def choose(self, face_id: str, name: str | None) -> None:
        """*face_id* shows the set *name*; None (or the default) clears its choice."""
        with self._lock:
            raw = self.raw()
            if name is not None and name not in self.sets(raw):
                raise ExpressionError(f"no rest set {name!r}")
            faces = {k: v for k, v in raw["faces"].items() if k != face_id}
            if name is not None and name != DEFAULT_SET:
                faces[face_id] = name
            self._write({"sets": raw["sets"], "faces": faces})

    def define(self, rest: RestSet) -> None:
        """Add or replace a set (a built-in one too: the file's wins)."""
        with self._lock:
            raw = self.raw()
            sets = dict(raw["sets"])
            sets[rest.name] = {"weights": rest.weights, "life": rest.life}
            self._write({"sets": sets, "faces": raw["faces"]})

    def remove(self, name: str) -> bool:
        """Remove a set of the file's own (a built-in one comes back); faces that chose it go back to the default."""
        with self._lock:
            raw = self.raw()
            if name not in raw["sets"]:
                return False
            # Faces that chose a built-in set's name keep it (the built-in one is back); others go to the default.
            faces = raw["faces"] if name in BUILTIN else {k: v for k, v in raw["faces"].items() if v != name}
            self._write({"sets": {k: v for k, v in raw["sets"].items() if k != name}, "faces": faces})
            return True

    def repoint(self, old: str, new: str | None) -> None:
        """A face was renamed (*new*) or deleted (None): its choice follows it, or goes."""
        with self._lock:
            raw = self.raw()
            if old not in raw["faces"]:
                return
            faces = {k: v for k, v in raw["faces"].items() if k != old}
            if new is not None:
                faces[new] = raw["faces"][old]
            self._write({"sets": raw["sets"], "faces": faces})
