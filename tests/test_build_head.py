"""build_head.py's parts that run without Blender: Blender's modules stood in for, the recipe's own logic checked."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "src" / "argus_mask" / "editor" / "build_head.py"


@pytest.fixture
def recipe(monkeypatch):
    """``build_head.py`` loaded outside Blender, without running ``main()``."""
    for name in ("bpy", "bmesh", "mathutils", "mathutils.bvhtree"):
        monkeypatch.setitem(sys.modules, name, mock.MagicMock(name=name))
    source = SCRIPT.read_text(encoding="utf-8")
    assert source.endswith("\nmain()\n"), "the recipe runs main() as its last line"
    module = types.ModuleType("build_head_outside_blender")
    module.__file__ = str(SCRIPT)
    exec(compile(source.removesuffix("main()\n"), str(SCRIPT), "exec"), module.__dict__)  # noqa: S102
    return module


def phenotype(**race) -> dict:
    keys = ("gender", "age", "muscle", "weight", "proportions", "height", "cupsize", "firmness")
    return {**dict.fromkeys(keys, 0.5), "race": {"asian": 0.2, "caucasian": 0.6, "african": 0.2, **race}}


@pytest.mark.parametrize(("key", "value"), [("african", 5), ("asian", -0.3)])
def test_a_race_weight_outside_0_to_1_is_refused(recipe, tmp_path, key, value) -> None:
    path = tmp_path / "preset.json"
    path.write_text(json.dumps({"phenotype": phenotype(**{key: value})}))
    with pytest.raises(recipe.RecipeError, match=f"race.{key} must be between 0 and 1"):
        recipe.load_preset(str(path))


def test_an_image_with_no_pixels_is_opaque(recipe) -> None:
    empty = SimpleNamespace(size=(0, 0), channels=4, pixels=SimpleNamespace(foreach_get=lambda out: None))
    assert recipe.is_opaque(empty)


def test_shapes_are_scaled_only_on_meshes(recipe) -> None:
    rig = SimpleNamespace(type="ARMATURE", data=SimpleNamespace())  # an armature's data has no shape keys
    empty = SimpleNamespace(type="EMPTY", data=None)
    recipe.scale_shapes([rig, empty], {"jawOpen": 1.2})


def test_a_head_with_nothing_where_the_lips_are_is_refused_by_name(recipe, monkeypatch) -> None:
    monkeypatch.setattr(recipe, "key_coords", lambda obj, key=None: np.array([[1.0, 1.0, 1.0]], dtype=np.float32))
    with pytest.raises(recipe.RecipeError, match="where the lips should be"):
        recipe.shade_mouth(SimpleNamespace(), None, None, 0.2)


class Co:
    def __init__(self, z: float) -> None:
        self.z, self.length = z, 1.0

    def __neg__(self) -> Co:
        return self


def test_covered_skin_goes_with_its_edges_and_nothing_else(recipe, monkeypatch) -> None:
    verts = [SimpleNamespace(index=i, co=Co(0.0), normal=Co(0.0), link_faces=[]) for i in range(4)]
    loose = SimpleNamespace(index=4, co=Co(2.0), normal=Co(0.0), link_faces=[])  # above the neck, no faces
    covered = SimpleNamespace(verts=verts[:3])
    deleted = []
    skin = SimpleNamespace(verts=[*verts, loose], faces=[covered], from_mesh=lambda data: None)
    skin.to_mesh = skin.free = lambda *a: None
    cloth = SimpleNamespace(from_mesh=lambda data: None, free=lambda: None)
    made = iter([cloth, skin])  # the garment's mesh, then the skin's
    delete = SimpleNamespace(delete=lambda mesh, geom, context: deleted.append((context, list(geom))))
    monkeypatch.setattr(recipe, "bmesh", SimpleNamespace(new=lambda: next(made), ops=delete))
    tree = SimpleNamespace(ray_cast=lambda origin, direction, reach: (object(),))
    monkeypatch.setattr(recipe, "BVHTree", SimpleNamespace(FromBMesh=lambda mesh: tree))
    basemesh = SimpleNamespace(data=SimpleNamespace(update=lambda: None))

    assert recipe.hide_covered_skin(basemesh, [SimpleNamespace(data=None)], below_z=1.0) == 1
    # One delete, of the faces: Blender drops the edges and vertices only they used, and leaves loose ones alone.
    assert deleted == [("FACES", [covered])]


class Block:
    def __init__(self, name: str, *, users_elsewhere: int = 0, materials=()) -> None:
        self.name, self.extra, self.materials = name, users_elsewhere, list(materials)
        self.holders: list = []

    @property
    def users(self) -> int:
        return self.extra + len(self.holders)

    def __hash__(self) -> int:
        return hash(self.name)


def test_a_cleared_scene_leaves_no_orphaned_data(recipe, monkeypatch) -> None:
    material, kept_material = Block("Material"), Block("Shared")
    cube = Block("Cube", materials=[material])
    library = Block("Library", users_elsewhere=1, materials=[kept_material])  # a fake user: not ours to remove
    objects = [SimpleNamespace(data=cube), SimpleNamespace(data=library), SimpleNamespace(data=None)]
    for obj in objects:
        if obj.data is not None:
            obj.data.holders.append(obj)
            for m in obj.data.materials:
                m.holders.append(obj.data)
    removed = []

    class Objects(list):
        def remove(self, obj, do_unlink) -> None:
            super().remove(obj)
            if obj.data is not None:
                obj.data.holders.remove(obj)

    def batch_remove(blocks) -> None:
        for block in blocks:
            removed.append(block.name)
            for m in block.materials:
                m.holders.remove(block)

    objects = Objects(objects)
    monkeypatch.setattr(
        recipe, "bpy", SimpleNamespace(data=SimpleNamespace(objects=objects, batch_remove=batch_remove))
    )

    recipe.clear_scene()
    assert objects == []
    assert sorted(removed) == ["Cube", "Material"]
