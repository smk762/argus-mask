"""A head preset: the whole recipe ``build_head.py`` builds from.

The same JSON as ``scripts/stock_heads/<id>.json``. Blender's Python has no
pydantic, so ``build_head.py`` keeps its own checks; this model is the stricter
one, applied before a preset reaches Blender, and the tests hold both to the
committed stock presets.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from argus_mask.arkit import ARKIT_52
from argus_mask.models import FACE_ID_PATTERN

# The youngest a head may be: 18 years on MakeHuman's scale. Below 0.5 the age macro maps to years as
# 1 + 48 * age, so 18 years is 17/48 (0.35417); this is that, rounded up. build_head.py holds the same.
ADULT_AGE = 0.3542

# The parts a preset names, and MPFB's data directory for each.
PART_DIRS: dict[str, str] = {
    "skin": "skins",
    "eyes": "eyes",
    "eyebrows": "eyebrows",
    "eyelashes": "eyelashes",
    "teeth": "teeth",
    "tongue": "tongue",
    "hair": "hair",
}
OPTIONAL_PARTS = ("hair",)
CLOTHES_DIR = "clothes"
MAX_CLOTHES = 3
# MakeHuman modelling targets a preset may set (CC0, bundled with MPFB): the allow-list in
# modifiers.json, generated from MPFB's own catalogue by scripts/make_modifiers.py. build_head.py reads
# the same file. A preset cannot name any other target, and so no other file.
_CATALOGUE = json.loads(Path(__file__).with_name("modifiers.json").read_text(encoding="utf-8"))
MODIFIERS: dict[str, dict] = {m["name"]: m for m in _CATALOGUE["modifiers"]}
MODIFIER_GROUPS: dict[str, str] = _CATALOGUE["groups"]


def one_way(name: str) -> bool:
    """A target with no opposite: 0 to 1, not -1 to +1."""
    return not MODIFIERS[name]["negative"]


# An asset reference: <part directory>/<asset>/<asset>.<mhmat|mhclo>. No dot-leading names, so no "..".
ASSET_REF = r"^[a-z]+/[A-Za-z0-9][A-Za-z0-9_-]*/[A-Za-z0-9][A-Za-z0-9_-]*\.(mhmat|mhclo)$"

Unit = Annotated[float, Field(ge=0.0, le=1.0)]
AssetRef = Annotated[str, Field(pattern=ASSET_REF)]
ShapeName = Literal[ARKIT_52]  # type: ignore[valid-type]
ModifierName = Literal[tuple(MODIFIERS)]  # type: ignore[valid-type]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Race(_Strict):
    asian: Unit
    caucasian: Unit
    african: Unit


class Phenotype(_Strict):
    """MakeHuman's macro settings, each 0 to 1."""

    gender: Unit  # 0 female, 1 male
    age: Annotated[float, Field(ge=ADULT_AGE, le=1.0)]
    muscle: Unit
    weight: Unit
    proportions: Unit
    height: Unit
    cupsize: Unit
    firmness: Unit
    race: Race


class Assets(_Strict):
    skin: AssetRef
    eyes: AssetRef
    eyebrows: AssetRef
    eyelashes: AssetRef
    teeth: AssetRef
    tongue: AssetRef
    hair: AssetRef | None = None
    # Worn on top: an outfit, a hat. Fitted like the body parts; MPFB hides the skin under them. Every
    # head is dressed: at least one garment here, and the library checks that one is an outfit.
    clothes: list[AssetRef] = Field(..., max_length=MAX_CLOTHES)

    @field_validator("clothes")
    @classmethod
    def _each_once(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("every head is dressed: choose an outfit")
        if len(set(value)) != len(value):
            raise ValueError("the same clothes are listed twice")
        return value

    def refs(self) -> dict[str, str]:
        """Part to reference, for the body parts this preset uses (not the clothes)."""
        return {part: ref for part in PART_DIRS if (ref := getattr(self, part))}


class Bust(_Strict):
    height_m: float = Field(0.40, ge=0.2, le=0.6)
    half_width_m: float = Field(0.23, ge=0.1, le=0.4)


class Mouth(_Strict):
    tongue_flatten: float = Field(1.0, ge=0.2, le=1.0)
    tongue_drop_mm: float = Field(0.0, ge=0.0, le=10.0)
    upper_teeth_drop_mm: float = Field(0.0, ge=0.0, le=6.0)
    cavity_shade: float | None = Field(None, ge=0.0, le=1.0)


class Preset(_Strict):
    id: str = Field(..., pattern=FACE_ID_PATTERN)
    name: str = Field(..., min_length=1, max_length=80)
    phenotype: Phenotype
    assets: Assets
    # Shape the face beyond the phenotype: face shape, chin, nose, lips. 0 leaves it as the phenotype made it.
    targets: dict[ModifierName, Annotated[float, Field(ge=-1.0, le=1.0)]] = Field(default_factory=dict)
    bust: Bust = Field(default_factory=Bust)
    mouth: Mouth = Field(default_factory=Mouth)
    shape_scale: dict[ShapeName, Annotated[float, Field(ge=0.0, le=3.0)]] = Field(default_factory=dict)
    texture_gain: dict[
        Literal["eyes", "eyebrows", "eyelashes", "teeth", "tongue", "hair"], Annotated[float, Field(ge=0.2, le=3.0)]
    ] = Field(default_factory=dict)
    blend_parts: list[Literal["eyebrows", "eyelashes"]] = Field(default_factory=list)
    max_texture_px: int = Field(2048, ge=256, le=2048)
    max_alpha_texture_px: int = Field(1024, ge=256, le=2048)
    max_triangles: int = Field(30_000, ge=1000, le=30_000)

    @field_validator("targets")
    @classmethod
    def _one_way_targets_are_not_negative(cls, value: dict[str, float]) -> dict[str, float]:
        for name, weight in value.items():
            if one_way(name) and weight < 0:
                raise ValueError(f"{name} goes from 0 to 1")
        return value

    def age_years(self) -> int:
        """MakeHuman's own mapping of the age macro to years."""
        age = self.phenotype.age
        return round(1 + 48 * age if age < 0.5 else 25 + 130 * (age - 0.5))
