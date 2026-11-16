"""Run LAM with argus-mask's FLAME core in place of the Max-Planck-noticed code it ships (#126).

LAM vendors a FLAME layer whose files carry Max-Planck's proprietary notice: the skinning module (`lbs.py`, in LAM
and in its vendored VHAP tracker), and copies of the same maths inside `FlameHeadSubdivided`. `install()` must run
before anything from LAM is imported. It registers `flame_core` under both skinning modules' names, so LAM's own
`from ...lbs import ...` binds ours. `patch()` runs after LAM is imported. It points the FLAME modules' helpers and
`FlameHeadSubdivided`'s posing and skinning methods at ours, and gives the tracker a grey texture in place of FLAME's
non-commercial painted one. Nothing in the LAM checkout is edited.

`verify()` says what is still LAM's: every name that `install`/`patch` should have replaced and has not.
"""

from __future__ import annotations

import importlib
import sys

import flame_core

LBS_MODULES = ("lam.models.rendering.flame_model.lbs", "vhap.model.lbs")
FLAME_MODULES = (
    "lam.models.rendering.flame_model.flame",
    "lam.models.rendering.flame_model.flame_arkit",
    "vhap.model.flame",
)
HELPERS = ("to_tensor", "to_np", "Struct")
LBS_NAMES = ("lbs", "vertices2landmarks", "blend_shapes", "vertices2joints", "batch_rigid_transform", "batch_rodrigues")


def install() -> None:
    clash = [name for name in LBS_MODULES if name in sys.modules and sys.modules[name] is not flame_core]
    if clash:
        raise RuntimeError(f"LAM's skinning was imported before overrides.install(): {', '.join(clash)}")
    for name in LBS_MODULES:
        sys.modules[name] = flame_core


# FlameHeadSubdivided's own posing and skinning, with the signatures its callers use.


def _get_transformed_mat(self, pose, v_shaped, posedirs, parents, J_regressor, pose2rot, dtype):
    return flame_core.transforms_for_pose(pose, v_shaped, parents, J_regressor, pose2rot=pose2rot, dtype=dtype)


def _skinning(self, v_posed, A, lbs_weights, batch_size, num_joints, dtype, device):
    return flame_core.skinning(v_posed, A, lbs_weights)


def _inverse_skinning(self, v_posed, A, lbs_weights, batch_size, num_joints, dtype, device):
    return flame_core.inverse_skinning(v_posed, A, lbs_weights)


METHODS = {"get_transformed_mat": _get_transformed_mat, "skinning": _skinning, "inverse_skinning": _inverse_skinning}

# The tracker's albedo starts from FLAME's painted mean texture, from the non-commercial FLAME texture space. With
# photometric fitting off, as LAM runs it, the texture is loaded but never fitted to: a flat grey gives the same head,
# byte for byte (checked on #126). This stands in for it without reading any file.
GREY = 128 / 255
TEXTURE_USERS = ("vhap.model.flame", "vhap.model.tracker")


def _grey_texture():
    import torch

    class GreyTexture(torch.nn.Module):
        """FlameTexPainted's interface, with a flat grey in place of the painted texture."""

        def __init__(self, tex_size=512, painted_tex_path=None):
            super().__init__()
            self.tex_size = tex_size
            self.register_buffer("tex_painted", torch.full((1, 3, tex_size, tex_size), GREY, dtype=torch.float32))

        def forward(self):
            return self.tex_painted

    return GreyTexture


# FLAME's vertex masks (`FLAME_masks.pkl`, an MPI research-only download) name 14 regions. LAM's own regions (lip
# rings, teeth) are its hard-coded vertex lists, and the MPI regions feed only photometric tracking, which LAM leaves
# off: with every region a single vertex the head is byte-identical (#126). So no file is read; each region is a
# placeholder, and `check_photometric_off` refuses a configuration where they would matter.
MASK_REGIONS = (
    "face",
    "neck",
    "scalp",
    "boundary",
    "right_eyeball",
    "left_eyeball",
    "right_ear",
    "left_ear",
    "forehead",
    "eye_region",
    "nose",
    "lips",
    "right_eye_region",
    "left_eye_region",
)
TRACKING_CONFIG = "configs/vhap_tracking/base_tracking_config.yaml"


def _placeholder_regions(self, flame_parts_path=None):
    import torch

    self.v = sys.modules[type(self).__module__].BufferContainer()
    for region in MASK_REGIONS:
        self.v.register_buffer(region, torch.zeros(1, dtype=torch.long))
    self.create_custom_mask()


def check_photometric_off(config_path: str = TRACKING_CONFIG) -> None:
    """Refuse photometric tracking: it is what reads the painted texture and the MPI regions we replace."""
    from pathlib import Path

    text = Path(config_path).read_text()
    if "photometric: false" not in text or "photometric: true" in text:
        raise RuntimeError(f"{config_path}: photometric tracking must be off (the texture and masks are placeholders)")


def patch() -> None:
    install()
    check_photometric_off()
    grey = _grey_texture()
    for name in TEXTURE_USERS:
        importlib.import_module(name).FlameTexPainted = grey
    for name in FLAME_MODULES:
        module = importlib.import_module(name)
        module.FlameMask.process_vertex_mask = _placeholder_regions
        for helper in HELPERS:
            setattr(module, helper, getattr(flame_core, helper))
        for fn in LBS_NAMES:
            if hasattr(module, fn):
                setattr(module, fn, getattr(flame_core, fn))
        subdivided = getattr(module, "FlameHeadSubdivided", None)
        if subdivided is not None:
            for method, fn in METHODS.items():
                if method in vars(subdivided):
                    setattr(subdivided, method, fn)


def verify() -> list[str]:
    """Every replaced name still bound to LAM's code: empty when the core is fully in place."""
    left = [name for name in LBS_MODULES if sys.modules.get(name) is not flame_core]
    left += [
        f"{name}.FlameTexPainted"
        for name in TEXTURE_USERS
        if name in sys.modules and sys.modules[name].FlameTexPainted.__name__ != "GreyTexture"
    ]
    for name in FLAME_MODULES:
        module = sys.modules.get(name)
        if module is None:
            continue
        if module.FlameMask.process_vertex_mask is not _placeholder_regions:
            left.append(f"{name}.FlameMask.process_vertex_mask")
        left += [f"{name}.{h}" for h in HELPERS if getattr(module, h) is not getattr(flame_core, h)]
        left += [
            f"{name}.{f}" for f in LBS_NAMES if hasattr(module, f) and getattr(module, f) is not getattr(flame_core, f)
        ]
        subdivided = getattr(module, "FlameHeadSubdivided", None)
        if subdivided is not None:
            left += [
                f"{name}.FlameHeadSubdivided.{m}" for m, fn in METHODS.items() if vars(subdivided).get(m, fn) is not fn
            ]
    return left
