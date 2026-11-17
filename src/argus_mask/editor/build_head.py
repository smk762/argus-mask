"""Build a stock head for argus-mask with MPFB (the MakeHuman plugin for Blender).

Run it headless, inside Blender, with MPFB enabled and its asset packs installed
(see scripts/README.md):

    blender -b -P src/argus_mask/editor/build_head.py -- \
        --preset scripts/stock_heads/stock-01.json --out assets/stock-01/head.glb

The preset is the recipe: the MakeHuman phenotype, which skin, hair, eyes and
so on, what the bust wears, where it is cut, and the size budget. The same preset, Blender,
MPFB and asset packs build the same head. A second file, ``<out>.recipe.json``,
records what was actually used, for the face manifest's provenance.

What it does, in order:

1. refuses a preset under 18 or with no clothes;
2. creates the human and adds the skin, the body parts and the clothes (the
   body under a garment is removed, as MakeHuman does);
3. bakes the modelling shape keys, loads MPFB's ARKit face units
   (``faceunits01``) and interpolates them onto the eyes, lashes, brows, teeth
   and tongue, because glTF cannot drive one mesh's morph targets from another;
4. removes MakeHuman's helper geometry, cuts everything below the shoulders
   and centres the head on the origin;
5. shrinks textures to the budget, converts the opaque ones to JPEG, and
   exports a ``.glb`` with one morph target per face unit, named as ARKit
   names them.

This file imports ``bpy``: it is a Blender script, not part of the package.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import shutil
import struct
import sys
import tempfile

import bmesh
import bpy
import numpy as np  # bundled with Blender
from mathutils import Matrix, Vector, kdtree
from mathutils.bvhtree import BVHTree

# MakeHuman's age macro: 0.0 is 1 year, 0.5 about 25, 1.0 about 90; below 0.5 it maps to years as
# 1 + 48 * age. The youngest a head may be is 18 years, 17/48 (0.35417), rounded up here. The editor's
# preset model holds the same value.
ADULT_AGE = 0.3542

BODY_PARTS = ("eyes", "eyebrows", "eyelashes", "teeth", "tongue", "hair")
# ``assets.clothes`` is a list of garments. Each is named in the model after its asset (the mhclo's
# base name), so ``texture_gain`` and ``blend_parts`` can refer to it.
CLOTHES = "clothes"
# Parts that must carry the face shapes. Hair may stay static.
MOVING_PARTS = ("eyes", "eyebrows", "eyelashes", "teeth", "tongue")
PHENOTYPE_KEYS = ("gender", "age", "muscle", "weight", "proportions", "height", "cupsize", "firmness")
RACE_KEYS = ("asian", "caucasian", "african")
# MPFB's rig for every head: 53 bones, Unreal-style names (pelvis, spine_01..03, neck_01, head, ...).
RIG = "game_engine"
# How much of the body a model keeps: head (the bust, cut by the preset's bust settings), waist (down to the hips,
# arms and all) or full (head to shoes, nothing cut). Each framing is its own model, built from the same preset (#62).
FRAMINGS = ("head", "waist", "full")
WAIST_BELOW_HIPS_M = 0.04
# More body, more triangles: a framing's share of the preset's budget, within MAX_TRIANGLES.
BUDGET_FACTOR = {"head": 1.0, "waist": 1.5, "full": 2.0}
# The most any model may have (faces.py: MAX_TRIANGLES; a test holds the two together).
MAX_TRIANGLES = 120_000
# Skin under a garment is deleted (hide_covered_skin): cloth this far out along the skin's normal covers it, and so
# does cloth this far in (skin that has pushed through).
COVER_OUT_M = 0.03
COVER_IN_M = 0.012
# A garment over the budget is thinned (Decimate), but never below this share of its triangles.
GARMENT_MIN_RATIO = 0.4

# Modelling targets a preset may set under ``targets``: the allow-list in modifiers.json beside this
# script (generated from MPFB's catalogue by scripts/make_modifiers.py; the editor reads the same file).
# Each names its negative and positive files under MPFB's data/targets; one with no negative files is
# one-way, 0 to 1. Two-way ones go from -1 to +1.
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "modifiers.json"), encoding="utf-8") as _fh:
    MODIFIERS = {m["name"]: m for m in json.load(_fh)["modifiers"]}
# Hair colour: the same file the package's tests import, loaded by path (Blender's Python has no argus_mask).
_spec = importlib.util.spec_from_file_location(
    "argus_mask_hair_colour", os.path.join(os.path.dirname(os.path.abspath(__file__)), "hair_colour.py")
)
hair_colour = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hair_colour)


class RecipeError(Exception):
    """The preset cannot be built, or the result is outside the budget."""


def age_years(age: float) -> float:
    """MakeHuman's own mapping of the age macro to years."""
    return 1 + 48 * age if age < 0.5 else 25 + 130 * (age - 0.5)


def load_preset(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        preset = json.load(handle)
    phenotype = preset.get("phenotype") or {}
    missing = [k for k in PHENOTYPE_KEYS if not isinstance(phenotype.get(k), (int, float))]
    race = phenotype.get("race") or {}
    missing += [f"race.{k}" for k in RACE_KEYS if not isinstance(race.get(k), (int, float))]
    if missing:
        raise RecipeError(f"{path}: phenotype is missing {', '.join(missing)}")
    for key in PHENOTYPE_KEYS:
        if not 0.0 <= phenotype[key] <= 1.0:
            raise RecipeError(f"{path}: phenotype.{key} must be between 0 and 1")
    for key in RACE_KEYS:
        if not 0.0 <= race[key] <= 1.0:
            raise RecipeError(f"{path}: phenotype.race.{key} must be between 0 and 1")
    age = float(phenotype["age"])
    if age < ADULT_AGE:
        raise RecipeError(
            f"{path}: phenotype.age {age} is about {age_years(age):.0f} years on MakeHuman's scale. "
            f"Heads are 18 or older: use {ADULT_AGE} or more."
        )
    assets = preset.get("assets") or {}
    for key in ("skin", *MOVING_PARTS):
        if not assets.get(key):
            raise RecipeError(f"{path}: assets.{key} is required")
    clothes = assets.get(CLOTHES) or []
    if isinstance(clothes, str):
        clothes = assets[CLOTHES] = [clothes]
    if not isinstance(clothes, list) or not all(isinstance(c, str) and c for c in clothes):
        raise RecipeError(f"{path}: assets.clothes must be a list of clothes assets")
    if not clothes:
        raise RecipeError(f"{path}: assets.clothes is required: every head is dressed")
    names = [asset_name(c) for c in clothes]
    if clash := [n for n in names if n in (*BODY_PARTS, "head")]:
        raise RecipeError(f"{path}: a garment cannot be called {clash[0]!r}")
    if len(set(names)) != len(names):
        raise RecipeError(f"{path}: assets.clothes lists the same garment twice")
    shaping = preset.get("targets") or {}
    if not isinstance(shaping, dict):
        raise RecipeError(f"{path}: targets must map a target name to a value")
    for name, value in shaping.items():
        if name not in MODIFIERS:
            raise RecipeError(f"{path}: targets.{name} is not one of {', '.join(MODIFIERS)}")
        low = -1.0 if MODIFIERS[name]["negative"] else 0.0
        if not isinstance(value, (int, float)) or not low <= value <= 1.0:
            raise RecipeError(f"{path}: targets.{name} must be between {low:g} and 1")
    check_hair_colour(path, preset.get("hair_colour"))
    return preset


def check_hair_colour(path: str, spec) -> None:
    """The same rules as preset.py's HairColour, for a preset handed to Blender directly."""
    if spec is None:
        return
    if not isinstance(spec, dict):
        raise RecipeError(f"{path}: hair_colour must be an object")
    known = {"eumelanin", "pheomelanin", "dye", "dye_strength", "parts"}
    if extra := sorted(set(spec) - known):
        raise RecipeError(f"{path}: hair_colour has unknown fields: {', '.join(extra)}")
    for key in ("eumelanin", "pheomelanin", "dye_strength"):
        value = spec.get(key, 0.0)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
            raise RecipeError(f"{path}: hair_colour.{key} must be between 0 and 1")
    dye = spec.get("dye")
    if dye is not None:
        try:
            if not isinstance(dye, str) or not dye.startswith("#"):
                raise ValueError(dye)
            hair_colour.hex_to_linear(dye)
        except (ValueError, TypeError) as exc:
            raise RecipeError(f"{path}: hair_colour.dye must be #rrggbb") from exc
    elif spec.get("dye_strength", 0.0) > 0:
        raise RecipeError(f"{path}: hair_colour.dye_strength needs a dye colour")
    parts = spec.get("parts", ["hair"])
    if not isinstance(parts, list) or not parts or any(p not in hair_colour.PARTS for p in parts):
        raise RecipeError(f"{path}: hair_colour.parts must list some of {', '.join(hair_colour.PARTS)}")
    if len(set(parts)) != len(parts):
        raise RecipeError(f"{path}: hair_colour.parts lists a part twice")


def find_mpfb():
    """The MPFB extension's package, whichever repository it was installed into."""
    for name in ("bl_ext.user_default.mpfb", "bl_ext.blender_org.mpfb"):
        try:
            return importlib.import_module(name)
        except ImportError:
            continue
    for name in list(sys.modules):
        if name.endswith(".mpfb"):
            return sys.modules[name]
    raise RecipeError("MPFB is not installed or not enabled in this Blender (see scripts/README.md)")


def resolve_asset(services, fragment: str) -> str:
    """``hair/short02/short02.mhclo`` to a file under one of MPFB's data roots, and never outside them."""
    subdir, _, rest = fragment.partition("/")
    parts = rest.split("/")
    if "\\" in fragment or os.path.isabs(rest) or any(p in ("", ".", "..") for p in [subdir, *parts]):
        raise RecipeError(f"asset {fragment!r} is not a path inside MPFB's data directory")
    for root in services.AssetService.get_asset_roots(subdir):
        # Contained by its text, not by resolving links: an asset pack may be installed as a symlink.
        root = os.path.normpath(root)
        candidate = os.path.normpath(os.path.join(root, *parts))
        if os.path.commonpath([candidate, root]) == root and os.path.isfile(candidate):
            return candidate
    raise RecipeError(f"asset {fragment!r} is not installed: is the asset pack that holds it in MPFB's data directory?")


def asset_name(fragment: str) -> str:
    """``clothes/suit01/suit01.mhclo`` to ``suit01``."""
    return os.path.splitext(os.path.basename(fragment))[0]


def apply_modifiers(mpfb, targets, basemesh, values: dict) -> dict[str, float]:
    """Load the preset's modelling targets onto the new human, before anything is fitted to it.

    Each value picks the negative or the positive file and sets its weight. They are baked with the
    phenotype, so the face shapes, the teeth and the clothes are fitted to the result.
    """
    root = os.path.join(os.path.dirname(mpfb.__file__), "data", "targets")
    applied = {}
    for name, value in sorted(values.items()):
        value = float(value)
        if value == 0.0:
            continue
        files = MODIFIERS[name]["positive" if value > 0 else "negative"]
        for fragment in files:
            path = os.path.join(root, *fragment.split("/")) + ".target.gz"
            if not os.path.isfile(path):
                raise RecipeError(f"MPFB has no target {fragment!r} (looked for {path})")
            targets.load_target(basemesh, path, weight=abs(value), name=os.path.basename(fragment))
        applied[name] = value
    return applied


def mesh_children(basemesh) -> dict[str, object]:
    """The body-part and garment objects hanging off the basemesh, or off its rig (MPFB parents them to the
    armature when there is one): body parts by part name, garments by asset name (tagged when the garment is
    added, not read back from MPFB's object name)."""
    found = {}
    owners = {basemesh, basemesh.parent} - {None}
    for obj in bpy.data.objects:
        if obj.type == "MESH" and obj is not basemesh and obj.parent in owners:
            kind = str(obj.get("argus_garment") or obj.get("MPFB_GEN_object_type") or "").lower() or obj.name
            if kind in found:
                raise RecipeError(f"two parts are both called {kind!r}")
            found[kind] = obj
    return found


def triangles(obj) -> int:
    return sum(len(p.vertices) - 2 for p in obj.data.polygons)


def cut_to_bust(objects, z_cut: float, half_width: float | None) -> None:
    """Cut away everything below *z_cut* and further than *half_width* from the centre line (None: no side cuts,
    for a framing that keeps the arms).

    Three plane cuts, so the bust ends in clean edges. bmesh carries the shape keys through the
    cut: the new vertices on the cut line get every face shape, interpolated.
    Objects must have their transforms applied (local coordinates are world coordinates).
    """
    planes = [((0, 0, z_cut), (0, 0, 1))]
    if half_width is not None:
        planes += [((half_width, 0, 0), (-1, 0, 0)), ((-half_width, 0, 0), (1, 0, 0))]
    for obj in objects:
        mesh = bmesh.new()
        mesh.from_mesh(obj.data)
        for point, normal in planes:
            geometry = mesh.verts[:] + mesh.edges[:] + mesh.faces[:]
            bmesh.ops.bisect_plane(mesh, geom=geometry, dist=1e-6, plane_co=point, plane_no=normal, clear_inner=True)
        mesh.to_mesh(obj.data)
        mesh.free()
        obj.data.update()


def key_coords(obj, key: str | None = None) -> np.ndarray:
    """Vertex positions of *obj*: its mesh, or one of its shape keys (absolute, not a delta)."""
    data = obj.data.shape_keys.key_blocks[key].data if key else obj.data.vertices
    flat = np.empty(len(obj.data.vertices) * 3, dtype=np.float32)
    data.foreach_get("co", flat)
    return flat.reshape(-1, 3)


def rewrite_keys(obj, basis: np.ndarray, deltas: dict[str, np.ndarray]) -> None:
    """Give *obj* a new rest shape and, for each shape key, a new offset from it."""
    blocks = obj.data.shape_keys.key_blocks
    obj.data.vertices.foreach_set("co", basis.astype(np.float32).ravel())
    blocks[0].data.foreach_set("co", basis.astype(np.float32).ravel())
    for name, delta in deltas.items():
        blocks[name].data.foreach_set("co", (basis + delta).astype(np.float32).ravel())
    obj.data.update()


def seat_tongue(tongue, *, flatten: float, drop_m: float, front_m: float = 0.015) -> dict:
    """Lay the tongue on the floor of the mouth, and make it drop with the chin.

    MakeHuman's tongue is a dome as tall as the palate, and MPFB carries the jaw shapes to it as
    the jaw's own rotation about its hinge. The back half of the tongue is near that hinge and
    hardly moves, so in an open mouth the dome still reaches the upper lip and, seen from the
    front, fills the whole opening. Two changes:

    * the rest shape is flattened toward its underside and lowered, so it lies below the crowns
      of the lower teeth;
    * under ``jawOpen`` the whole tongue moves as its tip does (the tip sits behind the lower
      incisors and has their motion), so the back of it drops out of the opening too.

    ``tongueOut`` keeps the shape MPFB gave it: a full, raised tongue pushed out between the teeth.
    The sideways and forward jaw shapes are already plain translations and are left alone.
    Coordinates are Blender's: Z up, the face looking down -Y.
    """
    names = [k.name for k in tongue.data.shape_keys.key_blocks[1:]] if tongue.data.shape_keys else []
    old = key_coords(tongue)
    absolute = {name: key_coords(tongue, name) for name in names}
    floor = old[:, 2].min()
    new = old.copy()
    new[:, 2] = floor + (old[:, 2] - floor) * flatten - drop_m
    deltas = {name: absolute[name] - old for name in names}
    report = {
        "top_before_mm": round(float(old[:, 2].max()) * 1000, 1),
        "top_after_mm": round(float(new[:, 2].max()) * 1000, 1),
    }
    if "jawOpen" in deltas:
        tip = old[:, 1] <= old[:, 1].min() + front_m
        fall = deltas["jawOpen"][tip].mean(axis=0)
        deltas["jawOpen"] = np.tile(fall, (len(old), 1))
        report["jaw_open_move_mm"] = [round(float(v) * 1000, 1) for v in fall]
    if "tongueOut" in deltas:
        deltas["tongueOut"] = absolute["tongueOut"] - new
    rewrite_keys(tongue, new, deltas)
    return report


def lower_upper_teeth(teeth, drop_m: float) -> dict:
    """Move the upper teeth down, so their edge shows below the upper lip when the mouth opens.

    As fitted, the upper incisors end level with the lip's edge and are never seen, while the
    lower ones stand well clear of the lower lip: an open mouth shows only lower teeth. The upper
    teeth are the vertices ``jawOpen`` does not move.
    """
    old = key_coords(teeth)
    names = [k.name for k in teeth.data.shape_keys.key_blocks[1:]]
    deltas = {name: key_coords(teeth, name) - old for name in names}
    upper = np.linalg.norm(deltas["jawOpen"], axis=1) < 0.0005
    new = old.copy()
    new[upper, 2] -= drop_m
    rewrite_keys(teeth, new, deltas)
    return {
        "upper_vertices": int(upper.sum()),
        "edge_before_mm": round(float(old[upper, 2].min()) * 1000, 1),
        "edge_after_mm": round(float(new[upper, 2].min()) * 1000, 1),
    }


def scale_shapes(objects, factors: dict[str, float]) -> None:
    """Exaggerate (or soften) named face shapes on every mesh that carries them."""
    for obj in objects:
        if obj.type != "MESH" or not obj.data.shape_keys:
            continue
        basis = key_coords(obj)
        for name, factor in factors.items():
            if name in obj.data.shape_keys.key_blocks:
                scaled = basis + (key_coords(obj, name) - basis) * float(factor)
                obj.data.shape_keys.key_blocks[name].data.foreach_set("co", scaled.astype(np.float32).ravel())
        obj.data.update()


def smoothstep(lo: float, hi: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def shade_mouth(head, teeth, tongue, floor: float) -> dict:
    """Darken the inside of the mouth with vertex colours, deeper is darker.

    The viewer has no shadows, so without this the back of the mouth is lit like a cheek and an
    open mouth is a bright pink wall. A vertex colour multiplies the texture: 1 leaves it alone,
    *floor* is the darkest. Three meshes get one:

    * the head, for the vertices inside the mouth: those in a box around it that are hidden from
      the front by the lips or the face (a ray cast forward from them hits the head);
    * the teeth, so the incisors stay bright and the molars fall into the dark;
    * the tongue, bright at the tip only.

    Depth is measured back from the front of the lips. Coordinates are Blender's: Z up, the face
    looking down -Y; the origin is at eye height.
    """
    base = key_coords(head)
    box = (np.abs(base[:, 0]) < 0.034) & (base[:, 2] < -0.040) & (base[:, 2] > -0.110) & (base[:, 1] < 0.03)
    # The lips themselves: a narrow band on the centre line, below the nose and above the chin's bulge.
    lips = (np.abs(base[:, 0]) < 0.012) & (base[:, 2] < -0.060) & (base[:, 2] > -0.085)
    if not lips.any():
        raise RecipeError(
            "mouth.cavity_shade: no head vertex where the lips should be (|x| < 12 mm, z from -85 to -60 mm "
            "below the eyes): is the head centred and at MakeHuman's scale?"
        )
    lip_front = float(base[lips][:, 1].min())
    tree = BVHTree.FromPolygons([tuple(v) for v in base.tolist()], [tuple(p.vertices) for p in head.data.polygons])
    # Straight ahead, and a little up, down and to each side: a vertex seen through the gap
    # between slightly parted lips on one ray is still inside if the others are blocked.
    rays = [Vector(d).normalized() for d in ((0, -1, 0), (0, -1, 0.25), (0, -1, -0.25), (0.25, -1, 0), (-0.25, -1, 0))]
    shade = np.ones(len(base), dtype=np.float32)
    inside = 0
    for index in np.flatnonzero(box):
        origin = Vector(base[index].tolist())
        blocked = sum(1 for ray in rays if tree.ray_cast(origin + ray * 0.0005, ray, 0.15)[0] is not None)
        if blocked >= 3:
            depth = base[index, 1] - lip_front
            shade[index] = 1.0 - (1.0 - floor) * float(smoothstep(0.007, 0.020, np.array(depth)))
            inside += 1
    parts = [(head, shade)]
    if teeth is not None:
        depth = key_coords(teeth)[:, 1] - lip_front
        parts.append((teeth, 1.0 - (1.0 - max(floor, 0.2)) * smoothstep(0.018, 0.048, depth)))
    if tongue is not None:
        depth = key_coords(tongue)[:, 1] - lip_front
        parts.append((tongue, 0.85 - (0.85 - max(floor, 0.2)) * smoothstep(0.022, 0.045, depth)))
    for obj, values in parts:
        attribute = obj.data.color_attributes.new("Shade", "FLOAT_COLOR", "POINT")
        rgba = np.ones((len(values), 4), dtype=np.float32)
        rgba[:, :3] = np.asarray(values, dtype=np.float32)[:, None]
        attribute.data.foreach_set("color", rgba.ravel())
        obj.data.color_attributes.active_color = attribute
        # The new attribute's own index: a mesh that already has one (an import's, or an earlier Shade, which makes
        # this one Shade.001) would otherwise render and export the old one.
        names = [a.name for a in obj.data.color_attributes]
        obj.data.color_attributes.render_color_index = names.index(attribute.name)
    return {"lip_front_mm": round(lip_front * 1000, 1), "head_vertices_shaded": inside, "floor": floor}


WET_LINE = "wetline"
SKIN_DETAIL = "argus_skin_detail"  # the baked normal map's node, which drop_normal_maps keeps


def bake_skin_maps(head, workdir: str, *, strength: float = 0.18, radius: int = 3) -> dict:
    """Normal and roughness maps for the skin, from its own colour map.

    MakeHuman's skins ship a colour map only. Its fine detail (the colour minus a blur of it: creases,
    lip lines, blotches, the larger pores) is read as a height field, and becomes a tangent-space normal
    map (OpenGL convention, as glTF has it) and a roughness map, a touch rougher in the dark creases.
    Saved here, raw, so nothing colour-manages the data on the way into the .glb.
    """
    material = head.active_material
    if material is None or not material.node_tree:
        return {"made": False, "why": "the head has no node material"}
    nodes, links = material.node_tree.nodes, material.node_tree.links
    bsdf = next((n for n in nodes if n.type == "BSDF_PRINCIPLED"), None)
    textures = [n for n in nodes if n.type == "TEX_IMAGE" and n.image is not None and n.image.size[0]]
    if bsdf is None or not textures:
        return {"made": False, "why": "no principled BSDF or colour map on the head"}
    colour = max(textures, key=lambda n: n.image.size[0] * n.image.size[1]).image
    w, h = colour.size
    px = np.empty(w * h * 4, dtype=np.float32)
    colour.pixels.foreach_get(px)
    rgb = px.reshape(h, w, 4)[:, :, :3]
    lum = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

    def blur(a: np.ndarray, r: int) -> np.ndarray:  # a box blur twice over, about a Gaussian of r
        for _ in range(2):
            for axis in (0, 1):
                c = np.cumsum(
                    np.pad(a, [(r + 1, r) if ax == axis else (0, 0) for ax in (0, 1)], mode="edge"), axis=axis
                )
                a = (
                    np.take(c, range(2 * r + 1, c.shape[axis]), axis=axis)
                    - np.take(c, range(0, c.shape[axis] - 2 * r - 1), axis=axis)
                ) / (2 * r + 1)
        return a

    detail = lum - blur(lum, radius)
    height = np.clip(detail / (detail.std() + 1e-6), -3, 3)
    gy, gx = np.gradient(height)  # rows run up the image in Blender's pixel order: gy is +V, as OpenGL's +Y
    n = np.dstack([-gx * strength, -gy * strength, np.ones_like(height)])
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    rough = np.clip(0.62 - 0.05 * height, 0.4, 0.85)
    report = {"made": True, "source": colour.name, "size": [w, h]}
    made = {}  # the images made here: a name already taken gets a suffix, so never look them up by name
    for name, data in (
        ("skin_detail_normal", n * 0.5 + 0.5),
        ("skin_detail_roughness", np.dstack([rough, rough, rough])),
    ):
        image = bpy.data.images.new(name, w, h, alpha=False)
        image.colorspace_settings.name = "Non-Color"
        image.pixels.foreach_set(np.dstack([data, np.ones((h, w, 1))]).astype(np.float32).ravel())
        image["argus_baked"] = True  # fit_textures leaves it: already sized, and not colour
        path = os.path.join(workdir, f"{name}.jpg")
        image.filepath_raw = path
        image.file_format = "JPEG"
        image.save(filepath=path, quality=92)
        image.source = "FILE"
        image.reload()
        made[name] = image
        report[name] = path
    normal_tex = nodes.new("ShaderNodeTexImage")
    normal_tex.image = made["skin_detail_normal"]
    normal_map = nodes.new("ShaderNodeNormalMap")
    normal_map.name = SKIN_DETAIL
    links.new(normal_tex.outputs["Color"], normal_map.inputs["Color"])
    links.new(normal_map.outputs["Normal"], bsdf.inputs["Normal"])
    rough_tex = nodes.new("ShaderNodeTexImage")
    rough_tex.image = made["skin_detail_roughness"]
    split = nodes.new("ShaderNodeSeparateColor")
    links.new(rough_tex.outputs["Color"], split.inputs["Color"])
    links.new(split.outputs["Green"], bsdf.inputs["Roughness"])  # glTF reads roughness from green
    return report


def wet_line(head, eyes, *, columns: int = 14, width_m: float = 0.0012, lift_m: float = 0.00015) -> dict:
    """A thin strip along each lower lid where it meets the eye: the tear line that makes an eye look wet.

    MakeHuman's lids are not an opening in the mesh (the skin folds over the margin into the socket),
    and its eye is not one sphere to measure from. So the margin is found as the camera sees it: in
    columns across each eye, rays cast straight at the face from the front, stepping down from the
    pupil; the first that meets skin before eye marks the lower lid. The strip lies on the eye from
    there a little way up, just off its surface, and each strip vertex moves under every shape key
    with the skin triangle at the margin, so it blinks and squints with the lid. It is its own object,
    named for the viewer to give it a wet, fading material.
    """
    from mathutils.geometry import barycentric_transform  # noqa: PLC0415 - Blender's own

    head_co = key_coords(head)
    eye_co = key_coords(eyes)
    keys = [k.name for k in head.data.shape_keys.key_blocks[1:]] if head.data.shape_keys else []
    key_delta = {k: key_coords(head, k) - head_co for k in keys}
    head.data.calc_loop_triangles()
    tris = [tuple(t.vertices) for t in head.data.loop_triangles]
    skin = BVHTree.FromPolygons([tuple(v) for v in head_co.tolist()], tris)
    eye_tris = []
    eyes.data.calc_loop_triangles()
    eye_tris = [tuple(t.vertices) for t in eyes.data.loop_triangles]
    eye = BVHTree.FromPolygons([tuple(v) for v in eye_co.tolist()], eye_tris)
    ahead = Vector((0.0, 1.0, 0.0))  # from the camera into the face: the face looks down -Y
    front = float(head_co[:, 1].min()) - 0.05

    def first_hit(x: float, z: float):
        """What a ray from the front meets first at (x, z): 'eye', 'skin' or None, and where."""
        o = Vector((x, front, z))
        e = eye.ray_cast(o, ahead, 1.0)
        k = skin.ray_cast(o, ahead, 1.0)
        if e[0] is None and k[0] is None:
            return None, None
        if k[0] is None or (e[0] is not None and e[3] < k[3]):
            return "eye", e
        return "skin", k

    verts, faces, uvs, anchors = [], [], [], []  # anchors: per strip vertex, (triangle, barycentric weights)
    report = {}
    for side in (1, -1):
        mine = eye_co[np.sign(eye_co[:, 0]) == side]
        if len(mine) < 64:
            continue
        pupil = mine[np.argmin(mine[:, 1])]  # the eye's frontmost point
        half = (mine[:, 0].max() - mine[:, 0].min()) / 2
        row = []
        for x in np.linspace(pupil[0] - half * 0.85, pupil[0] + half * 0.85, columns):
            # down from the pupil's height while the eye shows; the first skin is the lower lid
            z, seen, last_eye = pupil[2], False, None
            while z > pupil[2] - 0.02:
                what, hit = first_hit(x, z)
                if what == "eye":
                    seen, last_eye = True, hit
                elif what == "skin" and seen:
                    break
                z -= 0.0002
            else:
                continue  # never saw the eye here, or never reached the lid
            if not seen or what != "skin":
                continue
            row.append((x, z, hit, last_eye))
        if len(row) < 4:
            report[side] = "lid margin not found"
            continue
        start = len(verts)
        for n, (x, z, lid, above) in enumerate(row):
            # the margin, pushed back onto the eye's surface depth there; and the eye a little way up
            low = Vector((x, above[0].y, z)) - ahead * lift_m
            up_hit = eye.ray_cast(Vector((x, front, z + width_m)), ahead, 1.0)
            high = (up_hit[0] if up_hit[0] is not None else Vector((x, above[0].y, z + width_m))) - ahead * lift_m
            tri = tris[lid[2]]
            a, b, c = (Vector(head_co[i]) for i in tri)
            weights = barycentric_transform(lid[0], a, b, c, Vector((1, 0, 0)), Vector((0, 1, 0)), Vector((0, 0, 1)))
            for p in (low, high):
                verts.append(np.array(p))
                anchors.append((tri, (weights.x, weights.y, weights.z)))
            u = n / (len(row) - 1)
            uvs += [(u, 0.0), (u, 1.0)]
        for n in range(len(row) - 1):
            a = start + 2 * n
            faces.append((a, a + 2, a + 3, a + 1))
        report[side] = len(row)
    if not faces:
        return {"made": False, "sides": report}
    mesh = bpy.data.meshes.new(WET_LINE)
    mesh.from_pydata([tuple(v) for v in verts], [], faces)
    layer = mesh.uv_layers.new(name="UVMap")
    for poly in mesh.polygons:
        for li in poly.loop_indices:
            layer.data[li].uv = uvs[mesh.loops[li].vertex_index]
    obj = bpy.data.objects.new(WET_LINE, mesh)
    bpy.context.scene.collection.objects.link(obj)
    basis = np.array(verts)
    obj.shape_key_add(name="Basis", from_mix=False)
    corner = np.array([a[0] for a in anchors])  # (n, 3) vertex indices
    weight = np.array([a[1] for a in anchors])[:, :, None]  # (n, 3, 1)
    for k in keys:
        delta = (key_delta[k][corner] * weight).sum(axis=1)
        if np.abs(delta).max() < 1e-7:
            continue  # this shape does not move the lids
        block = obj.shape_key_add(name=k, from_mix=False)
        block.data.foreach_set("co", (basis + delta).astype(np.float32).ravel())
    material = bpy.data.materials.new(WET_LINE)
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:  # for any glTF viewer: a faint, glossy film; the argus viewer fades it across
        bsdf.inputs["Base Color"].default_value = (0.92, 0.78, 0.76, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.05
        bsdf.inputs["Alpha"].default_value = 0.35
    if hasattr(material, "surface_render_method"):
        material.surface_render_method = "BLENDED"
    elif hasattr(material, "blend_method"):
        material.blend_method = "BLEND"
    obj.data.materials.append(material)
    return {"made": True, "sides": report, "object": obj}


def base_colour_images(obj) -> set:
    """The images feeding *obj*'s materials' Base Color: the ones a colourway may change (never a normal or roughness
    map)."""
    found = set()
    for slot in obj.material_slots:
        tree = slot.material.node_tree if slot.material else None
        if tree is None:
            continue
        for link in tree.links:
            if link.to_socket.name == "Base Color" and link.from_node.type == "TEX_IMAGE" and link.from_node.image:
                found.add(link.from_node.image)
    return found


def part_images(obj) -> set:
    """The images *obj*'s materials read."""
    found = set()
    for slot in obj.material_slots:
        if slot.material and slot.material.node_tree:
            for node in slot.material.node_tree.nodes:
                if node.type == "TEX_IMAGE" and node.image:
                    found.add(node.image)
    return found


def recolour(image, target) -> None:
    """Bake colour *target* into an image's colour, keeping its alpha.

    *target* is linear RGB (hair: the strands' variation kept), or a garment colourway's ``(mode, linear RGB)``:
    ``tint`` keeps the fabric's variation the same way, ``solid`` paints it flat.
    """
    colourway = isinstance(target, tuple | list) and len(target) == 2 and isinstance(target[0], str)
    mode, rgb = target if colourway else ("tint", target)
    pixels = np.empty(image.size[0] * image.size[1] * image.channels, dtype=np.float32)
    image.pixels.foreach_get(pixels)
    paint = hair_colour.solid if mode == "solid" else hair_colour.recolour
    pixels = paint(pixels.reshape(-1, image.channels), rgb).astype(np.float32)
    image.pixels.foreach_set(pixels.ravel())
    image.update()


def brighten(image, gain: float) -> None:
    """Multiply an image's colour (not its alpha) by *gain*."""
    pixels = np.empty(image.size[0] * image.size[1] * image.channels, dtype=np.float32)
    image.pixels.foreach_get(pixels)
    pixels = pixels.reshape(-1, image.channels)
    pixels[:, :3] = np.clip(pixels[:, :3] * gain, 0.0, 1.0)
    image.pixels.foreach_set(pixels.ravel())
    image.update()


def is_opaque(image) -> bool:
    """Whether every pixel's alpha is (near enough) 1."""
    if image.channels < 4:
        return True
    if not image.size[0] * image.size[1]:
        return True  # no pixels (an image Blender could not load): nothing to keep the alpha of
    pixels = np.empty(image.size[0] * image.size[1] * image.channels, dtype=np.float32)
    image.pixels.foreach_get(pixels)
    return bool(pixels[3 :: image.channels].min() > 0.995)


def unlink_alpha(image) -> None:
    """Stop materials reading *image*'s alpha, so the exporter writes them as opaque."""
    for material in bpy.data.materials:
        if not material.node_tree:
            continue
        for link in list(material.node_tree.links):
            if (
                link.from_node.type == "TEX_IMAGE"
                and link.from_node.image is image
                and link.from_socket.name == "Alpha"
            ):
                material.node_tree.links.remove(link)


def drop_normal_maps() -> None:
    """Unlink normal maps: on a face this size they cost more bytes than they show."""
    for material in bpy.data.materials:
        if not material.node_tree:
            continue
        for node in list(material.node_tree.nodes):
            if node.type == "NORMAL_MAP" and node.name != SKIN_DETAIL:
                material.node_tree.nodes.remove(node)


def used_images() -> set:
    used = set()
    for material in bpy.data.materials:
        if material.users and material.node_tree:
            for node in material.node_tree.nodes:
                if node.type == "TEX_IMAGE" and node.image and node.outputs and any(o.is_linked for o in node.outputs):
                    used.add(node.image)
    return used


def fit_textures(
    max_px: int, max_alpha_px: int, workdir: str, gains: dict | None = None, colours: dict | None = None
) -> list[dict]:
    """Shrink every used image, recolour and brighten those asked for, and store the opaque ones as JPEG.

    Opaque images are capped at *max_px*. Images with real transparency stay PNG, which is far
    larger per pixel, so they get their own, usually smaller, cap *max_alpha_px*.
    """
    scene = bpy.context.scene
    # save_render writes through the scene's colour management, which must not grade the texture: no look,
    # exposure or gamma (a user's startup file may set them), and a data image (Non-Color) written raw below.
    scene.display_settings.display_device = "sRGB"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    report = []
    for image in sorted(used_images(), key=lambda i: i.name):
        if image.get("argus_baked"):
            report.append({"source": image.name, "size": list(image.size), "format": "JPEG", "baked": True})
            continue
        width, height = image.size
        if not width or not height:
            continue
        source = os.path.basename(bpy.path.abspath(image.filepath)) or image.name
        # MPFB's game-engine material links every texture's alpha, used or not. Only an image that
        # really has transparent pixels (hair, lashes, brows) keeps it; the rest become opaque JPEGs.
        keep_alpha = not is_opaque(image)
        if not keep_alpha:
            unlink_alpha(image)
        scale = min(1.0, (max_alpha_px if keep_alpha else max_px) / max(width, height))
        if scale < 1.0:
            image.scale(max(1, round(width * scale)), max(1, round(height * scale)))
        if colours and image in colours:
            recolour(image, colours[image])
        if gains and image in gains:
            brighten(image, gains[image])
        scene.render.image_settings.file_format = "PNG" if keep_alpha else "JPEG"
        scene.render.image_settings.color_mode = "RGBA" if keep_alpha else "RGB"
        scene.render.image_settings.quality = 90
        path = os.path.join(workdir, f"{len(report):02d}.{'png' if keep_alpha else 'jpg'}")
        # Standard undoes a colour image's sRGB exactly; a data image's values are scene linear, so the display's
        # sRGB curve would land on them: Raw writes them unconverted.
        scene.view_settings.view_transform = "Raw" if image.colorspace_settings.is_data else "Standard"
        image.save_render(path, scene=scene)
        image.source = "FILE"
        image.filepath = path
        image.reload()
        report.append({"source": source, "size": list(image.size), "format": "PNG" if keep_alpha else "JPEG"})
    return report


def patch_glb(path: str, blended: set[str]) -> dict:
    """Set cut-out transparency on the alpha materials, and return the glTF JSON.

    Blended hair sorts wrongly against itself and the face in a real-time renderer; a mask with
    a cutoff does not. Blender's exporter only writes MASK for one particular node pattern, so
    the exported JSON is edited instead. Materials named in *blended* stay blended: brows and
    lashes lie flat on the skin, never overlap themselves, and as cut-outs they are hard black
    shapes.
    """
    with open(path, "rb") as handle:
        data = handle.read()
    json_len = struct.unpack_from("<I", data, 12)[0]
    doc = json.loads(data[20 : 20 + json_len])
    for material in doc.get("materials", []):
        if material.get("alphaMode") == "BLEND" and material.get("name") not in blended:
            material["alphaMode"] = "MASK"
            material["alphaCutoff"] = 0.35
            material["doubleSided"] = True
    text = json.dumps(doc, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 4)
    rest = data[20 + json_len :]
    with open(path, "wb") as handle:
        handle.write(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(text) + len(rest)))
        handle.write(struct.pack("<II", len(text), 0x4E4F534A))
        handle.write(text)
        handle.write(rest)
    return doc


def clear_scene() -> None:
    """Remove every object, and the data and materials only they used: a mesh or material left in ``bpy.data``
    keeps its name, so a later one given the same name (``head``, ``rig``) would get a ``.001`` suffix."""
    data = {obj.data for obj in bpy.data.objects if obj.data is not None}
    materials = {m for block in data for m in (getattr(block, "materials", None) or ()) if m is not None}
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    # Data first, so the materials it held lose those users; anything something else still uses is kept.
    for blocks in (data, materials):
        bpy.data.batch_remove([b for b in blocks if b.users == 0])


def build(preset: dict, out: str, framing: str = "head") -> dict:
    mpfb = find_mpfb()
    services = importlib.import_module(mpfb.__name__ + ".services")
    human, targets, faces, export = (
        services.HumanService,
        services.TargetService,
        services.FaceService,
        services.ExportService,
    )
    if not faces.is_faceunits01_installed(force_recheck=True):
        raise RecipeError("MPFB's faceunits01 asset pack is not installed: it holds the ARKit face shapes")

    clear_scene()

    phenotype = {k: float(preset["phenotype"][k]) for k in PHENOTYPE_KEYS}
    phenotype["race"] = {k: float(preset["phenotype"]["race"][k]) for k in RACE_KEYS}
    basemesh = human.create_human(
        mask_helpers=True,
        detailed_helpers=True,
        extra_vertex_groups=True,
        feet_on_ground=True,
        scale=0.1,
        macro_detail_dict=phenotype,
    )

    shaped = apply_modifiers(mpfb, targets, basemesh, preset.get("targets") or {})

    # The rig, before anything is fitted to the body: MPFB weights the skin, body parts and clothes to it
    # as they are added, so the whole bust is skinned. The viewer turns the head through neck_01 and head,
    # and the shoulders stay put; a wider framing (#62) poses the rest.
    rig = human.add_builtin_rig(basemesh, RIG, import_weights=True)
    if rig is None:
        raise RecipeError(f"MPFB has no {RIG!r} rig")

    assets = preset["assets"]
    human.set_character_skin(
        resolve_asset(services, assets["skin"]), basemesh, skin_type="GAMEENGINE", material_instances=False
    )
    for part in BODY_PARTS:
        if assets.get(part):
            human.add_mhclo_asset(
                resolve_asset(services, assets[part]),
                basemesh,
                asset_type=part,
                subdiv_levels=0,
                material_type="GAMEENGINE",
            )
    # Clothes are fitted like the body parts. MPFB also hides the skin under them (the delete group,
    # baked below with the masks), which saves the triangles the clothes add.
    for garment in assets.get(CLOTHES) or []:
        worn = human.add_mhclo_asset(
            resolve_asset(services, garment),
            basemesh,
            asset_type="Clothes",
            subdiv_levels=0,
            material_type="GAMEENGINE",
        )
        worn["argus_garment"] = asset_name(garment)

    # The order MPFB's own "export copy" uses: bake the modelling keys, then load the face units onto
    # the finished shape, carry them to the child meshes, and only then drop the helper geometry the
    # children were fitted to.
    targets.bake_targets(basemesh)
    faces.load_targets(basemesh, load_microsoft_visemes=False, load_meta_visemes=False, load_arkit_faceunits=True)
    faces.interpolate_targets(basemesh)
    export.bake_modifiers_remove_helpers(
        basemesh, bake_masks=True, bake_subdiv=False, remove_helpers=True, also_proxy=True
    )
    keep_only_armature(basemesh)

    parts = mesh_children(basemesh)
    objects = [basemesh, *parts.values()]
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
        keep_only_armature(obj)
    bpy.context.view_layer.objects.active = basemesh
    bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
    bpy.context.view_layer.update()

    # The armature's transform is applied with the meshes', so its bones and the skinned vertices stay in
    # one frame (the rig is in its rest pose: the meshes are not deformed by it while this happens).
    rig.select_set(True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    neck_base = rig.data.bones["neck_01"].head_local.z
    covered = hide_covered_skin(basemesh, [o for o in parts.values() if o.get("argus_garment")], neck_base)

    # Landmarks, in Blender's frame: Z up, the face looking down -Y.
    top = max((basemesh.matrix_world @ v.co).z for v in basemesh.data.vertices)
    eyes = parts.get("eyes")
    if eyes is None:
        raise RecipeError("the eyes did not load, so the head cannot be centred")
    eye_points = [eyes.matrix_world @ v.co for v in eyes.data.vertices]
    eye_z = sum(p.z for p in eye_points) / len(eye_points)
    if framing == "head":
        bust = preset.get("bust") or {}
        cut_to_bust(objects, top - float(bust.get("height_m", 0.40)), float(bust.get("half_width_m", 0.23)))
    elif framing == "waist":
        # Waist up: cut a little below the hip joints (the thigh bones' heads), arms and all.
        hips = min(rig.data.bones[name].head_local.z for name in ("thigh_l", "thigh_r"))
        cut_to_bust(objects, hips - WAIST_BELOW_HIPS_M, None)
    elif framing == "full":
        pass  # head to shoes: nothing is cut
    else:
        raise RecipeError(f"framing {framing!r}: choose from {', '.join(FRAMINGS)}")

    # Origin: on the centre line, at eye height, halfway through the skull. The viewer turns the head
    # about the origin, and looks at it from +Z (glTF) with the camera the manifest's framing describes.
    skull = [
        basemesh.matrix_world @ v.co for v in basemesh.data.vertices if (basemesh.matrix_world @ v.co).z > eye_z - 0.03
    ]
    centre_y = (min(p.y for p in skull) + max(p.y for p in skull)) / 2
    for obj in (*objects, rig):
        obj.location.y -= centre_y
        obj.location.z -= eye_z
    bpy.ops.object.select_all(action="DESELECT")
    for obj in (*objects, rig):
        obj.select_set(True)
    bpy.context.view_layer.objects.active = basemesh
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    basemesh.name = basemesh.data.name = "head"
    for kind, obj in parts.items():
        obj.name = obj.data.name = kind

    mouth = preset.get("mouth") or {}
    mouth_report = {}
    if "tongue" in parts and ("tongue_flatten" in mouth or "tongue_drop_mm" in mouth):
        mouth_report["tongue"] = seat_tongue(
            parts["tongue"],
            flatten=float(mouth.get("tongue_flatten", 1.0)),
            drop_m=float(mouth.get("tongue_drop_mm", 0.0)) / 1000,
        )
    if "teeth" in parts and mouth.get("upper_teeth_drop_mm"):
        mouth_report["upper_teeth"] = lower_upper_teeth(parts["teeth"], float(mouth["upper_teeth_drop_mm"]) / 1000)
    shape_scale = {str(k): float(v) for k, v in (preset.get("shape_scale") or {}).items()}
    scale_shapes(objects, shape_scale)
    if mouth.get("cavity_shade"):
        mouth_report["shade"] = shade_mouth(
            basemesh, parts.get("teeth"), parts.get("tongue"), float(mouth["cavity_shade"])
        )
    wet = wet_line(basemesh, parts["eyes"]) if "eyes" in parts else {"made": False}
    if wet.get("made"):
        strip = wet.pop("object")
        strip.vertex_groups.new(name="head").add(range(len(strip.data.vertices)), 1.0, "REPLACE")
        objects.append(strip)
    placed = add_accessories(preset.get("accessories") or [], basemesh, rig)
    objects.extend(obj for obj, _ in placed)
    # Every mesh is the armature's child, skinned by its Armature modifier: what the glTF exporter wants
    # to write a skin. Both are at the origin with no transform, so no parent inverse is needed.
    rig.name = rig.data.name = "rig"
    for obj in objects:
        keep_only_armature(obj, rig)
        obj.parent = rig
        obj.matrix_parent_inverse.identity()
    for obj in objects:
        for polygon in obj.data.polygons:
            polygon.use_smooth = True

    counts = {obj.name: triangles(obj) for obj in objects}
    total = sum(counts.values())
    budget = int(preset.get("max_triangles", 45000))
    budget = min(int(budget * BUDGET_FACTOR[framing]), MAX_TRIANGLES)  # arms, hands and legs are dense
    decimated = {}
    if total > budget:
        decimated = thin_garments([o for o in objects if o.get("argus_garment")], total - budget)
        counts = {obj.name: triangles(obj) for obj in objects}
        total = sum(counts.values())
    if total > budget:
        raise RecipeError(f"{total} triangles is over the budget of {budget}: {counts}")

    shapes = {
        obj.name: [k.name for k in obj.data.shape_keys.key_blocks[1:]] if obj.data.shape_keys else [] for obj in objects
    }
    still = [kind for kind in MOVING_PARTS if kind in parts and not shapes[kind]]

    # The shrunk textures only live until the exporter has packed them into the .glb: the editor runs
    # this for every build, so they must not pile up in the temp directory.
    workdir = tempfile.mkdtemp(prefix="argus-mask-head-")
    try:
        skin_maps = bake_skin_maps(basemesh, workdir)
        drop_normal_maps()
        max_px = int(preset.get("max_texture_px", 2048))
        gains = {
            image: float(gain)
            for kind, gain in (preset.get("texture_gain") or {}).items()
            if kind in parts
            for image in part_images(parts[kind])
        }
        dyed = preset.get("hair_colour")
        target = hair_colour.colour(dyed) if dyed else None
        colours = {
            image: target
            for kind in (dyed or {}).get("parts", ["hair"])
            if dyed and kind in parts
            for image in part_images(parts[kind])
        }
        # Colourways (#61): a garment's colour textures, tinted (its shading kept) or painted solid (print and all).
        for ref, spec in (preset.get("garment_colours") or {}).items():
            garment = parts.get(asset_name(ref))
            if garment is None:
                raise RecipeError(f"garment_colours: {ref} is not worn")
            for image in base_colour_images(garment):
                colours[image] = (spec.get("mode", "tint"), hair_colour.hex_to_linear(spec["colour"]))
        textures = fit_textures(max_px, int(preset.get("max_alpha_texture_px", max_px)), workdir, gains, colours)
        blended = {
            slot.material.name
            for kind in preset.get("blend_parts") or []
            if kind in parts
            for slot in parts[kind].material_slots
            if slot.material
        } | {WET_LINE}  # a film, not a cut-out

        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        bpy.ops.export_scene.gltf(
            filepath=out,
            export_format="GLB",
            use_selection=False,
            export_apply=False,
            export_yup=True,
            export_morph=True,
            export_morph_normal=False,
            export_morph_tangent=False,
            export_try_sparse_sk=True,
            export_skins=True,
            export_def_bones=False,
            export_rest_position_armature=True,
            export_animations=False,
            export_cameras=False,
            export_lights=False,
            export_tangents=False,
            export_materials="EXPORT",
            export_vertex_color="ACTIVE",
            export_all_vertex_colors=False,
            export_image_format="AUTO",
            export_draco_mesh_compression_enable=False,
        )
        doc = patch_glb(out, blended)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    lowest = min((obj.matrix_world @ v.co).z for obj in objects for v in obj.data.vertices)
    highest = max((obj.matrix_world @ v.co).z for obj in objects for v in obj.data.vertices)
    return {
        "preset": preset.get("id"),
        "blender": bpy.app.version_string,
        "mpfb": ".".join(str(n) for n in getattr(mpfb, "bl_info", {}).get("version", ())) or read_mpfb_version(mpfb),
        "phenotype": phenotype,
        "targets": shaped,
        "age_years": round(age_years(phenotype["age"])),
        "assets": assets,
        "triangles": counts,
        "framing": framing,
        "triangles_total": total,
        "rig": {"name": RIG, "bones": len(rig.data.bones)},
        "decimated": decimated,
        "skin_covered_faces": covered,
        "accessories": [report for _, report in placed],
        "garment_colours": preset.get("garment_colours") or {},
        "shapes": {name: len(names) for name, names in shapes.items()},
        "parts_without_shapes": still,
        "mouth": mouth_report,
        "wet_line": {str(k): v for k, v in (wet.get("sides") or {}).items()},
        "skin_maps": {k: v for k, v in skin_maps.items() if k in ("made", "why", "source", "size")},
        "shape_scale": shape_scale,
        "textures": textures,
        "hair_colour": (
            {
                "colour": hair_colour.to_hex(target),
                "parts": sorted({k for k in dyed.get("parts", ["hair"]) if k in parts}),
            }
            if dyed
            else None
        ),
        "materials": [
            {"name": m.get("name"), "alphaMode": m.get("alphaMode", "OPAQUE")} for m in doc.get("materials", [])
        ],
        "extent_m": {"below_eyes": round(-lowest, 3), "above_eyes": round(highest, 3)},
        "bytes": os.path.getsize(out),
    }


# Accessories (#61): small rigid pieces made here from primitives (our own, so CC0), placed on the dressed body by
# casting rays at it, and skinned whole to one bone so they follow it. Sizes in metres. The wearer faces -Y, so their
# left is +X. Each kind: the bone it rides on, and how it is found.
ACCESSORY_BONES = {"badge": "spine_03", "pins": "spine_03", "earpiece": "head"}


def add_accessories(specs: list, basemesh, rig) -> list:
    """Make and place each accessory; returns ``[(object, report)]``. One the rays cannot place is reported, not made."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    neck = rig.data.bones["neck_01"].head_local  # the base of the neck: the collar, and the chest below it
    head_tree = BVHTree.FromObject(basemesh, depsgraph)
    made = []
    for spec in specs:
        kind, side = spec["kind"], spec.get("side", "left")
        sign = 1.0 if side == "left" else -1.0
        colour = spec.get("colour", "#c9a227")
        report = {"kind": kind, "side": side, "colour": colour}
        if kind == "badge":  # on the chest, over the heart's side: the first surface a ray from the front meets
            hit = cast(depsgraph, Vector((sign * 0.085, -1.0, neck.z - 0.13)), Vector((0, 1, 0)))
            mesh = disc(radius=0.017, depth=0.004) if hit else None
        elif kind == "pins":  # on the collar's outer face, beside the throat: two small studs
            # Collars differ in height: try a band of heights and keep the hit nearest the front (smallest y): that
            # is the collar itself, where a ray past its edge would reach the neck behind it.
            hits = [
                cast(depsgraph, Vector((sign * 0.05, -1.0, neck.z + dz)), Vector((0, 1, 0)))
                for dz in (-0.04, -0.03, -0.02, -0.01, 0.0, 0.01)
            ]
            hits = [h for h in hits if h]
            hit = min(hits, key=lambda h: h[0].y) if hits else None
            mesh = studs(count=2, radius=0.0032, spacing=0.010) if hit else None
        elif kind == "earpiece":  # on the side of the head at the ear (the head alone: hair would stop the ray)
            found = head_tree.ray_cast(Vector((sign * 0.5, 0.0, -0.022)), Vector((-sign, 0, 0)))
            # The tree is the basemesh's own (local) space; the other accessories are cast in the world's.
            world = basemesh.matrix_world
            hit = (
                (world @ found[0], (world.to_3x3().inverted_safe().transposed() @ found[1]).normalized())
                if found[0] is not None
                else None
            )
            mesh = bud(radius=0.0075) if hit else None
        else:
            raise RecipeError(f"no accessory {kind!r}: there are {', '.join(ACCESSORY_BONES)}")
        if not hit:
            report["placed"] = False
            made.append((None, report))
            continue
        location, normal = hit
        obj = bpy.data.objects.new(f"accessory_{kind}_{side}", mesh)
        bpy.context.scene.collection.objects.link(obj)
        turn = Vector((0, 0, 1)).rotation_difference(normal).to_matrix().to_4x4()
        mesh.transform(Matrix.Translation(location + normal * 0.0015) @ turn)
        mesh.materials.append(metal(f"accessory-{kind}", colour))
        obj.vertex_groups.new(name=ACCESSORY_BONES[kind]).add(range(len(mesh.vertices)), 1.0, "REPLACE")
        for polygon in mesh.polygons:
            polygon.use_smooth = True
        report |= {"placed": True, "at": [round(c, 3) for c in location]}
        made.append((obj, report))
    return [(o, r) for o, r in made if o is not None] + [(None, r) for o, r in made if o is None]


def cast(depsgraph, origin, direction):
    """The first surface a ray meets, as (location, normal facing the ray), or None."""
    hit, location, normal, _, _, _ = bpy.context.scene.ray_cast(depsgraph, origin, direction, distance=2.0)
    if not hit:
        return None
    return location, (normal if normal.dot(direction) < 0 else -normal)


def _mesh(name: str, build) -> object:
    bm = bmesh.new()
    build(bm)
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    return mesh


def disc(radius: float, depth: float):
    """A flat round badge, its face along +Z."""
    return _mesh(
        "badge",
        lambda bm: bmesh.ops.create_cone(
            bm, cap_ends=True, segments=28, radius1=radius, radius2=radius * 0.92, depth=depth
        ),
    )


def studs(count: int, radius: float, spacing: float):
    """Small domed studs in a row along X."""

    def build(bm):
        for i in range(count):
            out = bmesh.ops.create_uvsphere(bm, u_segments=10, v_segments=6, radius=radius)
            offset = (i - (count - 1) / 2) * spacing
            bmesh.ops.scale(bm, vec=(1.0, 1.0, 0.55), verts=out["verts"])
            bmesh.ops.translate(bm, vec=(offset, 0.0, 0.0), verts=out["verts"])

    return _mesh("pins", build)


def bud(radius: float):
    """An earpiece: a small rounded bud standing off the ear."""

    def build(bm):
        out = bmesh.ops.create_uvsphere(bm, u_segments=14, v_segments=8, radius=radius)
        bmesh.ops.scale(bm, vec=(1.0, 1.25, 0.85), verts=out["verts"])

    return _mesh("earpiece", build)


def metal(name: str, colour: str):
    """A metallic material of *colour* (#rrggbb), for the glTF exporter's Principled BSDF."""
    material = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    rgb = hair_colour.hex_to_linear(colour)
    bsdf.inputs["Base Color"].default_value = (float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0)
    bsdf.inputs["Metallic"].default_value = 0.85
    bsdf.inputs["Roughness"].default_value = 0.32
    return material


def hide_covered_skin(basemesh, garments, below_z: float) -> int:
    """Delete the skin a garment covers, and so the skin that pokes through it; return the faces deleted.

    MPFB hides skin under clothes by each garment's delete group, which is drawn loosely: under a coverall the
    shins showed through at the knees. Here a skin vertex is covered when a garment is within COVER_OUT_M along
    its normal (the cloth over it) or within COVER_IN_M against it (cloth the skin has pushed through). A face goes
    when all its vertices are covered. At a garment's edge (a cuff, a collar) the rays miss, so the skin there stays.
    Only skin below *below_z* is considered (the base of the neck): under the chin a ray could reach a high collar,
    and a head that tilts up would show the hole. Objects must have their transforms applied.
    """
    if not garments:
        return 0
    trees = []
    for garment in garments:
        cloth = bmesh.new()
        cloth.from_mesh(garment.data)
        trees.append(BVHTree.FromBMesh(cloth))
        cloth.free()

    def hit(origin, direction, reach) -> bool:
        return any(tree.ray_cast(origin, direction, reach)[0] is not None for tree in trees)

    skin = bmesh.new()
    skin.from_mesh(basemesh.data)
    covered = set()
    for vert in skin.verts:
        normal = vert.normal
        if vert.co.z >= below_z or normal.length < 0.5:
            continue
        if hit(vert.co, normal, COVER_OUT_M) or hit(vert.co, -normal, COVER_IN_M):
            covered.add(vert.index)
    gone = [face for face in skin.faces if all(v.index in covered for v in face.verts)]
    # The faces, and the edges and vertices only they used: no wire edges left at the hole, and loose geometry
    # that was there before is not this function's to remove.
    bmesh.ops.delete(skin, geom=gone, context="FACES")
    skin.to_mesh(basemesh.data)
    skin.free()
    basemesh.data.update()
    return len(gone)


def thin_garments(garments, excess: int) -> dict[str, float]:
    """Decimate the garments, all by one ratio, to shed *excess* triangles (never below GARMENT_MIN_RATIO).

    Only clothes: the face, hands and hair keep every triangle. Vertex groups, so the skin weights, survive the
    Decimate; shape keys cannot (Blender refuses to apply it to a mesh that has them), and a garment fitted near the
    face carries some (a collar's jawOpen). So each key is kept as an offset per vertex, cleared, and given back to
    the thinned mesh from the nearest vertex of the original: these shapes are smooth across a garment. Returns the
    ratio applied, by garment; the caller checks the budget again.
    """
    total = sum(triangles(g) for g in garments)
    if not total:
        return {}
    ratio = max(GARMENT_MIN_RATIO, (total - excess) / total)
    done = {}
    for garment in garments:
        keys = garment.data.shape_keys
        offsets = {}
        if keys:
            basis = np.array([v.co[:] for v in keys.key_blocks[0].data])
            offsets = {k.name: np.array([v.co[:] for v in k.data]) - basis for k in keys.key_blocks[1:]}
            garment.shape_key_clear()
        modifier = garment.modifiers.new("thin", "DECIMATE")
        modifier.ratio = ratio
        modifier.use_collapse_triangulate = True
        with bpy.context.temp_override(object=garment, active_object=garment):
            bpy.ops.object.modifier_move_to_index(modifier=modifier.name, index=0)
            bpy.ops.object.modifier_apply(modifier=modifier.name)
        if offsets:
            tree = kdtree.KDTree(len(basis))
            for i, co in enumerate(basis):
                tree.insert(co, i)
            tree.balance()
            nearest = [tree.find(v.co)[1] for v in garment.data.vertices]
            garment.shape_key_add(name="Basis", from_mix=False)
            for name, offset in offsets.items():
                key = garment.shape_key_add(name=name, from_mix=False)
                for vertex, i in zip(key.data, nearest, strict=True):
                    vertex.co = vertex.co + Vector(offset[i])
        done[garment.name] = round(ratio, 3)
    return done


def keep_only_armature(obj, rig=None) -> None:
    """Remove every modifier but the Armature, which skins the mesh; with *rig*, make sure there is one, on it."""
    for modifier in list(obj.modifiers):
        if modifier.type != "ARMATURE":
            obj.modifiers.remove(modifier)
    if rig is None:
        return
    armature = next((m for m in obj.modifiers if m.type == "ARMATURE"), None) or obj.modifiers.new("rig", "ARMATURE")
    armature.object = rig


def read_mpfb_version(mpfb) -> str:
    manifest = os.path.join(os.path.dirname(mpfb.__file__), "blender_manifest.toml")
    try:
        with open(manifest, encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("version"):
                    return line.split("=", 1)[1].strip().strip('"')
    except OSError:
        pass
    return "unknown"


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(prog="build_head.py", description=__doc__.split("\n\n")[0])
    parser.add_argument("--preset", required=True, help="a recipe under scripts/stock_heads/")
    parser.add_argument("--out", required=True, help="the .glb to write")
    parser.add_argument("--framing", default="head", choices=FRAMINGS, help="how much of the body to keep")
    args = parser.parse_args(argv)
    try:
        recipe = build(load_preset(args.preset), args.out, args.framing)
    except RecipeError as exc:
        print(f"build_head: {exc}", file=sys.stderr)
        sys.exit(2)
    with open(os.path.splitext(args.out)[0] + ".recipe.json", "w", encoding="utf-8") as handle:
        json.dump(recipe, handle, indent=2)
        handle.write("\n")
    print("build_head: wrote", args.out)
    print(json.dumps(recipe, indent=2))


main()
