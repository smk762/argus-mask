"""Blender script: render a mask head from several views, for measuring where face landmarks sit on it (#126).

    blender --background --python lam/landmarks/render_head.py -- <head.glb> <out dir>

Writes `<view>.png` per view, and `scene.npz`: each view's camera (world-to-camera matrix, vertical field of view,
resolution) and the head's surface (the `head` and `eyes` meshes at rest, triangulated, in Blender's world space,
metres, Z up, the face toward -Y). Never imported: Blender runs it by path, like `editor/build_head.py`.
"""

import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Euler, Vector

RES = 1024
FOV_DEG = 25.0
DISTANCE = 0.75  # metres from the face's centre
YAWS = (-30, -15, 0, 15, 30)
PITCHES = (-10, 0, 10)
SURFACE = ("head", "eyes")  # what a landmark ray may land on
HIDDEN = ("Icosphere",)  # the editor's environment helper, not part of the face


def main(glb: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(glb))
    scene = bpy.context.scene
    for name in HIDDEN:
        if name in scene.objects:
            scene.objects[name].hide_render = True

    meshes = {o.name: o for o in scene.objects if o.type == "MESH"}
    missing = [n for n in SURFACE if n not in meshes]
    if missing:
        raise SystemExit(f"{glb}: no {', '.join(missing)} mesh")
    verts, faces, parts = surface([meshes[n] for n in SURFACE])
    eyes = meshes["eyes"]
    centre = mean_world(eyes)  # aim between the eyes

    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = scene.render.resolution_y = RES
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Standard"
    world = bpy.data.worlds.new("grey")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.5, 0.5, 0.5, 1)
    world.node_tree.nodes["Background"].inputs[1].default_value = 0.6
    scene.world = world
    for name, rot, energy in (("key", (50, 0, -30), 3.0), ("fill", (60, 0, 40), 1.5), ("top", (0, 0, 0), 1.0)):
        light = bpy.data.objects.new(name, bpy.data.lights.new(name, "SUN"))
        light.data.energy = energy
        light.rotation_euler = Euler([math.radians(a) for a in rot])
        scene.collection.objects.link(light)

    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    cam.data.sensor_fit = "VERTICAL"
    cam.data.angle = math.radians(FOV_DEG)
    scene.collection.objects.link(cam)
    scene.camera = cam

    names, world_to_cam = [], []
    for pitch in PITCHES:
        for yaw in YAWS:
            # Orbit the face's centre: yaw about Z, pitch about X; the camera looks back along its offset.
            offset = Euler((math.radians(-pitch), 0, math.radians(yaw))).to_matrix() @ Vector((0, -DISTANCE, 0))
            cam.location = centre + offset
            cam.rotation_euler = (-offset).to_track_quat("-Z", "Y").to_euler()
            bpy.context.view_layer.update()
            name = f"y{yaw:+03d}_p{pitch:+03d}"
            scene.render.filepath = str(out / f"{name}.png")
            bpy.ops.render.render(write_still=True)
            names.append(name)
            world_to_cam.append(np.array(cam.matrix_world.inverted()))
    np.savez(
        out / "scene.npz",
        views=np.array(names),
        world_to_cam=np.stack(world_to_cam),
        fov_y=math.radians(FOV_DEG),
        resolution=RES,
        verts=verts,
        faces=faces,
        parts=np.array(SURFACE),
        part_verts=parts,
    )
    print("RENDERED", len(names), "views,", len(faces), "triangles")


def surface(objects):
    """The objects' evaluated meshes, triangulated, in world space, and each one's vertex count."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    all_v, all_f, counts, base = [], [], [], 0
    for obj in objects:
        ev = obj.evaluated_get(depsgraph)
        mesh = ev.to_mesh()
        try:
            mesh.calc_loop_triangles()
            m = np.array(ev.matrix_world)
            # reshape: an empty mesh still gives (0, 3), so it concatenates with the rest.
            v = np.array([tuple(m[:3, :3] @ np.array(p.co) + m[:3, 3]) for p in mesh.vertices]).reshape(-1, 3)
            f = np.array([tuple(t.vertices) for t in mesh.loop_triangles], dtype=np.int64).reshape(-1, 3)
        finally:
            ev.to_mesh_clear()
        all_v.append(v)
        all_f.append(f + base)
        base += len(v)
        counts.append(len(v))
    return np.concatenate(all_v), np.concatenate(all_f), np.array(counts)


def mean_world(obj) -> Vector:
    total = Vector()
    for v in obj.data.vertices:
        total += obj.matrix_world @ v.co
    return total / len(obj.data.vertices)


if __name__ == "__main__":
    args = sys.argv[sys.argv.index("--") + 1 :]
    main(Path(args[0]), Path(args[1]))
