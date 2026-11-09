"""Build a stock head for argus-mask with MPFB (the MakeHuman plugin for Blender).

Run it headless, inside Blender, with MPFB enabled and its asset packs installed
(see scripts/README.md):

    blender -b -P scripts/build_stock_head.py -- \
        --preset scripts/stock_heads/stock-01.json --out assets/stock-01/head.glb

The preset is the recipe: the MakeHuman phenotype, which skin, hair, eyes and
so on, where the bust is cut, and the size budget. The same preset, Blender,
MPFB and asset packs build the same head. A second file, ``<out>.recipe.json``,
records what was actually used, for the face manifest's provenance.

What it does, in order:

1. refuses a preset that is not an adult;
2. creates the human and adds the skin and the body parts;
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
import json
import os
import struct
import sys
import tempfile

import bmesh
import bpy

# MakeHuman's age macro: 0.0 is a baby, 0.1875 a child of about 10, 0.5 a young adult of about 25,
# 1.0 about 90. Below 0.5 the mesh blends the child target in, so "about 18" (0.35) is a head that
# is partly a child's. This recipe therefore starts at the young-adult target and goes no lower.
ADULT_AGE = 0.5

BODY_PARTS = ("eyes", "eyebrows", "eyelashes", "teeth", "tongue", "hair")
# Parts that must carry the face shapes. Hair may stay static.
MOVING_PARTS = ("eyes", "eyebrows", "eyelashes", "teeth", "tongue")
PHENOTYPE_KEYS = ("gender", "age", "muscle", "weight", "proportions", "height", "cupsize", "firmness")
RACE_KEYS = ("asian", "caucasian", "african")


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
    age = float(phenotype["age"])
    if age < ADULT_AGE:
        raise RecipeError(
            f"{path}: phenotype.age {age} is about {age_years(age):.0f} years on MakeHuman's scale, and below "
            f"{ADULT_AGE} the mesh blends in the child target. Stock heads are adults only: use {ADULT_AGE} "
            "(about 25 years) or more."
        )
    assets = preset.get("assets") or {}
    for key in ("skin", *MOVING_PARTS):
        if not assets.get(key):
            raise RecipeError(f"{path}: assets.{key} is required")
    return preset


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
    """``hair/short02/short02.mhclo`` to a file under one of MPFB's data roots."""
    subdir, _, rest = fragment.partition("/")
    for root in services.AssetService.get_asset_roots(subdir):
        candidate = os.path.join(root, rest)
        if os.path.isfile(candidate):
            return candidate
    raise RecipeError(f"asset {fragment!r} is not installed: is the asset pack that holds it in MPFB's data directory?")


def mesh_children(basemesh) -> dict[str, object]:
    """The body-part objects hanging off the basemesh, by part name."""
    found = {}
    for obj in bpy.data.objects:
        if obj.type == "MESH" and obj is not basemesh and obj.parent is basemesh:
            kind = str(obj.get("MPFB_GEN_object_type") or "").lower()
            found[kind or obj.name] = obj
    return found


def triangles(obj) -> int:
    return sum(len(p.vertices) - 2 for p in obj.data.polygons)


def cut_to_bust(objects, z_cut: float, half_width: float) -> None:
    """Cut away everything below *z_cut* and further than *half_width* from the centre line.

    Three plane cuts, so the bust ends in clean edges. bmesh carries the shape keys through the
    cut: the new vertices on the cut line get every face shape, interpolated.
    Objects must have their transforms applied (local coordinates are world coordinates).
    """
    planes = (((0, 0, z_cut), (0, 0, 1)), ((half_width, 0, 0), (-1, 0, 0)), ((-half_width, 0, 0), (1, 0, 0)))
    for obj in objects:
        mesh = bmesh.new()
        mesh.from_mesh(obj.data)
        for point, normal in planes:
            geometry = mesh.verts[:] + mesh.edges[:] + mesh.faces[:]
            bmesh.ops.bisect_plane(mesh, geom=geometry, dist=1e-6, plane_co=point, plane_no=normal, clear_inner=True)
        mesh.to_mesh(obj.data)
        mesh.free()
        obj.data.update()


def is_opaque(image) -> bool:
    """Whether every pixel's alpha is (near enough) 1."""
    import numpy as np  # Blender bundles numpy

    if image.channels < 4:
        return True
    pixels = np.empty(image.size[0] * image.size[1] * image.channels, dtype=np.float32)
    image.pixels.foreach_get(pixels)
    return bool(pixels[3 :: image.channels].min() > 0.995)


def unlink_alpha(image) -> None:
    """Stop materials reading *image*'s alpha, so the exporter writes them as opaque."""
    for material in bpy.data.materials:
        if not material.node_tree:
            continue
        for link in list(material.node_tree.links):
            if link.from_node.type == "TEX_IMAGE" and link.from_node.image is image and link.from_socket.name == "Alpha":
                material.node_tree.links.remove(link)


def drop_normal_maps() -> None:
    """Unlink normal maps: on a face this size they cost more bytes than they show."""
    for material in bpy.data.materials:
        if not material.node_tree:
            continue
        for node in list(material.node_tree.nodes):
            if node.type == "NORMAL_MAP":
                material.node_tree.nodes.remove(node)


def used_images() -> set:
    used = set()
    for material in bpy.data.materials:
        if material.users and material.node_tree:
            for node in material.node_tree.nodes:
                if node.type == "TEX_IMAGE" and node.image and node.outputs and any(o.is_linked for o in node.outputs):
                    used.add(node.image)
    return used


def fit_textures(max_px: int, max_alpha_px: int, workdir: str) -> list[dict]:
    """Shrink every used image and store the opaque ones as JPEG. Returns what it did.

    Opaque images are capped at *max_px*. Images with real transparency stay PNG, which is far
    larger per pixel, so they get their own, usually smaller, cap *max_alpha_px*.
    """
    scene = bpy.context.scene
    scene.view_settings.view_transform = "Standard"  # save_render must not grade the texture
    report = []
    for image in sorted(used_images(), key=lambda i: i.name):
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
        scene.render.image_settings.file_format = "PNG" if keep_alpha else "JPEG"
        scene.render.image_settings.color_mode = "RGBA" if keep_alpha else "RGB"
        scene.render.image_settings.quality = 90
        path = os.path.join(workdir, f"{len(report):02d}.{'png' if keep_alpha else 'jpg'}")
        image.save_render(path, scene=scene)
        image.source = "FILE"
        image.filepath = path
        image.reload()
        report.append({"source": source, "size": list(image.size), "format": "PNG" if keep_alpha else "JPEG"})
    return report


def patch_glb(path: str) -> dict:
    """Set cut-out transparency on the alpha materials, and return the glTF JSON.

    Blended hair, lashes and brows sort wrongly against the face in a real-time
    renderer; a mask with a cutoff does not. Blender's exporter only writes MASK
    for one particular node pattern, so the exported JSON is edited instead.
    """
    with open(path, "rb") as handle:
        data = handle.read()
    json_len = struct.unpack_from("<I", data, 12)[0]
    doc = json.loads(data[20 : 20 + json_len])
    for material in doc.get("materials", []):
        if material.get("alphaMode") == "BLEND":
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


def build(preset: dict, out: str) -> dict:
    mpfb = find_mpfb()
    services = importlib.import_module(mpfb.__name__ + ".services")
    human, targets, faces, export = services.HumanService, services.TargetService, services.FaceService, services.ExportService
    if not faces.is_faceunits01_installed(force_recheck=True):
        raise RecipeError("MPFB's faceunits01 asset pack is not installed: it holds the ARKit face shapes")

    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    phenotype = {k: float(preset["phenotype"][k]) for k in PHENOTYPE_KEYS}
    phenotype["race"] = {k: float(preset["phenotype"]["race"][k]) for k in RACE_KEYS}
    basemesh = human.create_human(mask_helpers=True, detailed_helpers=True, extra_vertex_groups=True, feet_on_ground=True, scale=0.1, macro_detail_dict=phenotype)

    assets = preset["assets"]
    human.set_character_skin(resolve_asset(services, assets["skin"]), basemesh, skin_type="GAMEENGINE", material_instances=False)
    for part in BODY_PARTS:
        if assets.get(part):
            human.add_mhclo_asset(resolve_asset(services, assets[part]), basemesh, asset_type=part, subdiv_levels=0, material_type="GAMEENGINE")

    # The order MPFB's own "export copy" uses: bake the modelling keys, then load the face units onto
    # the finished shape, carry them to the child meshes, and only then drop the helper geometry the
    # children were fitted to.
    targets.bake_targets(basemesh)
    faces.load_targets(basemesh, load_microsoft_visemes=False, load_meta_visemes=False, load_arkit_faceunits=True)
    faces.interpolate_targets(basemesh)
    export.bake_modifiers_remove_helpers(basemesh, bake_masks=True, bake_subdiv=False, remove_helpers=True, also_proxy=True)
    for modifier in list(basemesh.modifiers):
        basemesh.modifiers.remove(modifier)

    parts = mesh_children(basemesh)
    objects = [basemesh, *parts.values()]
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
        for modifier in list(obj.modifiers):
            obj.modifiers.remove(modifier)
    bpy.context.view_layer.objects.active = basemesh
    bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
    bpy.context.view_layer.update()

    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    # Landmarks, in Blender's frame: Z up, the face looking down -Y.
    top = max((basemesh.matrix_world @ v.co).z for v in basemesh.data.vertices)
    eyes = parts.get("eyes")
    if eyes is None:
        raise RecipeError("the eyes did not load, so the head cannot be centred")
    eye_points = [eyes.matrix_world @ v.co for v in eyes.data.vertices]
    eye_z = sum(p.z for p in eye_points) / len(eye_points)
    bust = preset.get("bust") or {}
    cut_to_bust(objects, top - float(bust.get("height_m", 0.40)), float(bust.get("half_width_m", 0.23)))

    # Origin: on the centre line, at eye height, halfway through the skull. The viewer turns the head
    # about the origin, and looks at it from +Z (glTF) with the camera the manifest's framing describes.
    skull = [basemesh.matrix_world @ v.co for v in basemesh.data.vertices if (basemesh.matrix_world @ v.co).z > eye_z - 0.03]
    centre_y = (min(p.y for p in skull) + max(p.y for p in skull)) / 2
    for obj in objects:
        obj.location.y -= centre_y
        obj.location.z -= eye_z
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = basemesh
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    basemesh.name = basemesh.data.name = "head"
    for kind, obj in parts.items():
        obj.name = obj.data.name = kind
    for obj in objects:
        for polygon in obj.data.polygons:
            polygon.use_smooth = True

    counts = {obj.name: triangles(obj) for obj in objects}
    total = sum(counts.values())
    budget = int(preset.get("max_triangles", 30000))
    if total > budget:
        raise RecipeError(f"{total} triangles is over the budget of {budget}: {counts}")

    shapes = {obj.name: [k.name for k in obj.data.shape_keys.key_blocks[1:]] if obj.data.shape_keys else [] for obj in objects}
    still = [kind for kind in MOVING_PARTS if kind in parts and not shapes[kind]]

    drop_normal_maps()
    workdir = tempfile.mkdtemp(prefix="argus-mask-head-")
    max_px = int(preset.get("max_texture_px", 2048))
    textures = fit_textures(max_px, int(preset.get("max_alpha_texture_px", max_px)), workdir)

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
        export_skins=False,
        export_animations=False,
        export_cameras=False,
        export_lights=False,
        export_tangents=False,
        export_materials="EXPORT",
        export_image_format="AUTO",
        export_draco_mesh_compression_enable=False,
    )
    doc = patch_glb(out)

    lowest = min((obj.matrix_world @ v.co).z for obj in objects for v in obj.data.vertices)
    highest = max((obj.matrix_world @ v.co).z for obj in objects for v in obj.data.vertices)
    return {
        "preset": preset.get("id"),
        "blender": bpy.app.version_string,
        "mpfb": ".".join(str(n) for n in getattr(mpfb, "bl_info", {}).get("version", ())) or read_mpfb_version(mpfb),
        "phenotype": phenotype,
        "age_years": round(age_years(phenotype["age"])),
        "assets": assets,
        "triangles": counts,
        "triangles_total": total,
        "shapes": {name: len(names) for name, names in shapes.items()},
        "parts_without_shapes": still,
        "textures": textures,
        "materials": [{"name": m.get("name"), "alphaMode": m.get("alphaMode", "OPAQUE")} for m in doc.get("materials", [])],
        "extent_m": {"below_eyes": round(-lowest, 3), "above_eyes": round(highest, 3)},
        "bytes": os.path.getsize(out),
    }


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
    parser = argparse.ArgumentParser(prog="build_stock_head.py", description=__doc__.split("\n\n")[0])
    parser.add_argument("--preset", required=True, help="a recipe under scripts/stock_heads/")
    parser.add_argument("--out", required=True, help="the .glb to write")
    args = parser.parse_args(argv)
    try:
        recipe = build(load_preset(args.preset), args.out)
    except RecipeError as exc:
        print(f"build_stock_head: {exc}", file=sys.stderr)
        sys.exit(2)
    with open(os.path.splitext(args.out)[0] + ".recipe.json", "w", encoding="utf-8") as handle:
        json.dump(recipe, handle, indent=2)
        handle.write("\n")
    print("build_stock_head: wrote", args.out)
    print(json.dumps(recipe, indent=2))


main()
