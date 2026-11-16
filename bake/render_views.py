"""Blender script: render a head in Cycles from many views at many expressions, for training splats (#126 route A).

    blender -b -P bake/render_views.py -- --glb head.glb --expressions expressions.json --out renders \
        [--views 48] [--holdout-views 8] [--size 768] [--samples 64] [--only train|holdout] [--limit N]

Writes ``<out>/<set>/<e>_<v>.png`` (RGBA, sRGB, Khronos PBR Neutral as the viewer's look: the pixels are the colours splats store) and
``<out>/cameras.json``: per image its set, expression index, camera-to-world in the glTF scene's frame (OpenGL
convention: the camera looks down -Z, +Y up), vertical field of view and size. The cameras are computed in glTF
space and converted for Blender, so the trainer reads them as the viewer would.

Light: the viewer's three-point rig (look.js), as area lights, and Blender's bundled interior HDRI for fill and
reflections. Skin gets subsurface scattering. Deterministic: views come from a seeded generator per expression.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

GLTF_TO_BLENDER = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)  # (x, y, z) glTF -> (x, -z, y) Blender
TARGET = np.array([0.0, -0.06, 0.0])  # between the eyes (origin) and the collar, glTF space
FOV_Y = 30.0


def args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glb", type=Path, required=True)
    ap.add_argument("--expressions", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--views", type=int, default=48)
    ap.add_argument("--holdout-views", type=int, default=8)
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--only", choices=["train", "holdout"])
    ap.add_argument("--limit", type=int, help="render only the first N expressions of each set (a test)")
    return ap.parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])


def look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Camera-to-world, OpenGL convention, glTF frame."""
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, [0.0, 1.0, 0.0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = r, u, -f, eye
    return m


def views(rng: np.random.Generator, n: int) -> list[np.ndarray]:
    """Mostly in front (where the viewer looks from), some to the sides and a few behind the ear line."""
    out = []
    for _ in range(n):
        az = rng.normal(0, 40) if rng.random() < 0.8 else rng.uniform(-110, 110)
        az = float(np.clip(az, -110, 110))
        el = float(np.clip(rng.normal(5, 12), -25, 35))
        dist = float(rng.uniform(0.5, 0.8))
        a, e = math.radians(az), math.radians(el)
        target = TARGET + rng.uniform(-0.02, 0.02, 3)
        eye = target + dist * np.array([math.sin(a) * math.cos(e), math.sin(e), math.cos(a) * math.cos(e)])
        out.append(look_at(eye, target))
    return out


def to_blender(m: np.ndarray) -> Matrix:
    b = np.eye(4)
    b[:3, :3] = GLTF_TO_BLENDER @ m[:3, :3]
    b[:3, 3] = GLTF_TO_BLENDER @ m[:3, 3]
    return Matrix(b.tolist())


def scene(size: int, samples: int) -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    s = bpy.context.scene
    s.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for kind in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = kind
            prefs.get_devices()
            if any(d.type == kind for d in prefs.devices):
                break
        except TypeError:
            continue
    for d in prefs.devices:
        d.use = d.type != "CPU"
    s.cycles.device = "GPU"
    s.cycles.samples = samples
    s.cycles.use_denoising = True
    s.cycles.seed = 0
    s.render.use_persistent_data = True  # keep the synced scene between views: only the camera moves
    s.render.resolution_x = s.render.resolution_y = size
    s.render.film_transparent = True
    s.render.image_settings.file_format = "PNG"
    s.render.image_settings.color_mode = "RGBA"
    s.view_settings.view_transform = "Khronos PBR Neutral"  # look.js's tone mapping
    s.view_settings.look = "None"
    world = bpy.data.worlds.new("world")
    s.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    env = nodes.new("ShaderNodeTexEnvironment")
    hdri = Path(bpy.utils.system_resource("DATAFILES")) / "studiolights" / "world" / "interior.exr"
    env.image = bpy.data.images.load(str(hdri))
    world.node_tree.links.new(env.outputs["Color"], nodes["Background"].inputs["Color"])
    nodes["Background"].inputs["Strength"].default_value = 0.25


def light(name: str, colour: tuple, direction: tuple, energy: float, size: float) -> None:
    d = np.array(direction, float)
    d /= np.linalg.norm(d)
    pos = GLTF_TO_BLENDER @ (TARGET + d * 1.5)
    data = bpy.data.lights.new(name, "AREA")
    data.energy, data.size, data.color = energy, size, colour
    obj = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = Vector(pos)
    obj.rotation_euler = (Vector(GLTF_TO_BLENDER @ TARGET) - obj.location).to_track_quat("-Z", "Y").to_euler()


def rig() -> None:  # look.js's: warm key high to one side, cool fill, two cool rims from behind
    light("key", (1.0, 0.93, 0.84), (0.55, 0.45, 1.0), 40, 0.6)
    light("fill", (0.86, 0.91, 1.0), (-0.8, 0.05, 0.75), 10, 1.0)
    light("rim_l", (0.75, 0.85, 1.0), (-0.55, 0.5, -0.9), 30, 0.4)
    light("rim_r", (0.75, 0.85, 1.0), (0.6, 0.35, -0.9), 18, 0.4)


def materials(meshes: list) -> None:
    for obj in meshes:
        for slot in obj.material_slots:
            m = slot.material
            if not m or not m.use_nodes:
                continue
            bsdf = next((n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
            if bsdf is None:
                continue
            if obj.name.startswith("head"):  # skin: light scatters under it, reddened
                bsdf.inputs["Subsurface Weight"].default_value = 0.35
                bsdf.inputs["Subsurface Radius"].default_value = (1.0, 0.35, 0.2)
                bsdf.inputs["Subsurface Scale"].default_value = 0.004


def main() -> None:
    a = args()
    ex = json.loads(a.expressions.read_text(encoding="utf-8"))
    names = ex["blendshapes"]
    scene(a.size, a.samples)
    rig()
    bpy.ops.import_scene.gltf(filepath=str(a.glb))
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    materials(meshes)
    cam = bpy.data.objects.new("camera", bpy.data.cameras.new("camera"))
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    cam.data.sensor_fit = "VERTICAL"
    cam.data.angle_y = math.radians(FOV_Y)
    cam.data.clip_start, cam.data.clip_end = 0.05, 20.0
    index_path = a.out / "cameras.json"
    cameras = json.loads(index_path.read_text()) if index_path.exists() else []
    done = {c["file"] for c in cameras}
    sets = [("train", a.views), ("holdout", a.holdout_views)]
    for label, n_views in sets:
        if a.only and label != a.only:
            continue
        weights_list = ex[label][: a.limit] if a.limit else ex[label]
        for e, weights in enumerate(weights_list):
            for obj in meshes:
                keys = obj.data.shape_keys
                if not keys:
                    continue
                for kb in keys.key_blocks[1:]:
                    kb.value = weights[names.index(kb.name)] if kb.name in names else 0.0
            rng = np.random.default_rng([0 if label == "train" else 1, e])
            for v, c2w in enumerate(views(rng, n_views)):
                file = f"{label}/{e:02d}_{v:02d}.png"
                if file not in done:
                    cam.matrix_world = to_blender(c2w)
                    bpy.context.scene.render.filepath = str(a.out / file)
                    bpy.ops.render.render(write_still=True)
                    cameras.append(
                        {
                            "file": file,
                            "set": label,
                            "expression": e,
                            "c2w": c2w.tolist(),
                            "fov_y": FOV_Y,
                            "width": a.size,
                            "height": a.size,
                        }
                    )
                    index_path.write_text(json.dumps(cameras) + "\n")
            print(f"{label} expression {e + 1}/{len(weights_list)} done", flush=True)


if __name__ == "__main__":
    main()
