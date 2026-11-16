"""FLAME's forward kinematics and skinning, written for argus-mask from the published model (#126).

FLAME (Li, Bolkart, Black, Li, Romero: "Learning a model of facial shape and expression from 4D scans",
SIGGRAPH Asia 2017) is a template mesh deformed in three steps: blendshapes (identity and expression), pose
correctives driven by the joint rotations, then linear blend skinning over a five-joint chain (global, neck, jaw,
two eyes). This module is that maths, nothing more. It replaces the FLAME layer LAM ships, whose files carry
Max-Planck's proprietary notice: `overrides.py` puts it in their place before LAM imports them, so the head builder
runs no MPG code. The model data itself is FLAME 2023 Open (CC BY 4.0), loaded by LAM from `flame2023.pkl`.

The functions keep the names and call signatures LAM and VHAP use, so they drop in. Shapes: B batch, V vertices,
J joints, L landmarks, K blendshape coefficients.
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = [
    "Struct",
    "batch_rigid_transform",
    "batch_rodrigues",
    "blend_shapes",
    "inverse_skinning",
    "lbs",
    "skinning",
    "to_np",
    "to_tensor",
    "transform_mat",
    "transforms_for_pose",
    "vertices2joints",
    "vertices2landmarks",
]


# ── the model file ──────────────────────────────────────────────────────────────────────────────────────────────


class Struct:
    """The model file's dictionary, as attributes."""

    def __init__(self, **fields):
        self.__dict__.update(fields)


def to_np(array, dtype=np.float32) -> np.ndarray:
    """A model array (numpy, a scipy sparse matrix or a chumpy array) as a dense numpy array."""
    if hasattr(array, "toarray"):  # scipy.sparse
        array = array.toarray()
    return np.asarray(array, dtype=dtype)


def to_tensor(array, dtype=torch.float32) -> torch.Tensor:
    if torch.is_tensor(array):
        return array.to(dtype)
    return torch.as_tensor(np.asarray(array), dtype=dtype)


# ── rotations and rigid transforms ──────────────────────────────────────────────────────────────────────────────


def batch_rodrigues(rot_vecs: torch.Tensor, epsilon: float = 1e-8, dtype=torch.float32) -> torch.Tensor:
    """Axis-angle vectors (N, 3) to rotation matrices (N, 3, 3): R = I + sin(θ) K + (1 − cos(θ)) K², K = [k]×.

    `epsilon` keeps the axis defined (and the gradient finite) at θ = 0, where R is the identity whatever the axis.
    """
    rot_vecs = rot_vecs.to(dtype)
    theta = torch.linalg.vector_norm(rot_vecs + epsilon, dim=1, keepdim=True)  # (N, 1)
    axis = rot_vecs / theta
    sin, cos = torch.sin(theta)[:, :, None], torch.cos(theta)[:, :, None]
    x, y, z = axis.unbind(dim=1)
    zero = torch.zeros_like(x)
    k = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], dim=1).view(-1, 3, 3)
    eye = torch.eye(3, dtype=dtype, device=rot_vecs.device).expand_as(k)
    return eye + sin * k + (1 - cos) * (k @ k)


def transform_mat(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Rotations (N, 3, 3) and translations (N, 3, 1) as homogeneous transforms (N, 4, 4)."""
    top = torch.cat([R, t], dim=2)
    bottom = torch.zeros_like(top[:, :1])
    bottom[..., 3] = 1
    return torch.cat([top, bottom], dim=1)


def batch_rigid_transform(rot_mats: torch.Tensor, joints: torch.Tensor, parents: torch.Tensor, dtype=torch.float32):
    """Pose a kinematic chain.

    rot_mats (B, J, 3, 3) are each joint's rotation relative to its parent; joints (B, J, 3) the rest positions;
    parents (J,) each joint's parent, -1 for the root, every parent listed before its children.
    Returns the posed joint positions (B, J, 3) and, per joint, the transform (B, J, 4, 4) taking a rest-pose point to
    its posed position were it bound to that joint alone: the world transform with the joint's rest position removed.
    """
    B, J = joints.shape[:2]
    joints = joints.to(dtype)
    offsets = joints.clone()  # each joint relative to its parent; the root keeps its own position
    offsets[:, 1:] -= joints[:, parents[1:]]
    local = transform_mat(rot_mats.reshape(-1, 3, 3), offsets.reshape(-1, 3, 1)).view(B, J, 4, 4)
    world = [local[:, 0]]
    for j in range(1, J):
        world.append(world[int(parents[j])] @ local[:, j])
    world = torch.stack(world, dim=1)  # (B, J, 4, 4)
    posed_joints = world[:, :, :3, 3]
    # world · [x; 1] − world · [j_rest; 0] = world · [x − j_rest; 1]: subtract the rotated rest position.
    rel = world.clone()
    rel[:, :, :3, 3] -= (world[:, :, :3, :3] @ joints[..., None])[..., 0]
    return posed_joints, rel


# ── blendshapes, joints, landmarks ──────────────────────────────────────────────────────────────────────────────


def blend_shapes(betas: torch.Tensor, shape_disps: torch.Tensor) -> torch.Tensor:
    """Coefficients (B, K) times displacement bases (V, 3, K): per-vertex offsets (B, V, 3)."""
    return torch.einsum("bk,vck->bvc", betas, shape_disps)


def vertices2joints(J_regressor: torch.Tensor, vertices: torch.Tensor) -> torch.Tensor:
    """Joint positions (B, J, 3) from vertices (B, V, 3), each a fixed weighting (J, V) of the vertices."""
    return torch.einsum("jv,bvc->bjc", J_regressor, vertices)


def vertices2landmarks(vertices, faces, lmk_faces_idx, lmk_bary_coords) -> torch.Tensor:
    """Landmarks (B, L, 3) as barycentric points on mesh faces.

    lmk_faces_idx is (L,) or (B, L) face indices into faces (F, 3); lmk_bary_coords (L, 3) or (B, L, 3).
    """
    B = vertices.shape[0]
    idx = lmk_faces_idx.expand(B, -1) if lmk_faces_idx.dim() == 1 else lmk_faces_idx
    bary = lmk_bary_coords.expand(B, -1, -1) if lmk_bary_coords.dim() == 2 else lmk_bary_coords
    corner_ids = faces[idx]  # (B, L, 3) vertex ids
    batch = torch.arange(B, device=vertices.device)[:, None, None]
    corners = vertices[batch, corner_ids]  # (B, L, 3 corners, 3 coords)
    return (bary[..., None] * corners).sum(dim=2)


# ── posing and skinning ─────────────────────────────────────────────────────────────────────────────────────────


def _rotations(pose: torch.Tensor, pose2rot: bool, dtype) -> torch.Tensor:
    """Pose (B, J·3 axis-angle, or anything holding B·J rotation matrices) as (B, J, 3, 3)."""
    B = pose.shape[0]
    if pose2rot:
        return batch_rodrigues(pose.reshape(-1, 3), dtype=dtype).view(B, -1, 3, 3)
    return pose.reshape(B, -1, 3, 3).to(dtype)


def transforms_for_pose(pose, v_shaped, parents, J_regressor, pose2rot=True, dtype=torch.float32):
    """The joints of a shaped mesh, posed: (per-joint skinning transforms (B, J, 4, 4), posed joints (B, J, 3))."""
    joints = vertices2joints(J_regressor, v_shaped)
    posed_joints, transforms = batch_rigid_transform(_rotations(pose, pose2rot, dtype), joints, parents, dtype=dtype)
    return transforms, posed_joints


def _blended(transforms: torch.Tensor, lbs_weights: torch.Tensor) -> torch.Tensor:
    """Each vertex's transform (B, V, 4, 4): its skinning weights (V, J) over the joints' transforms (B, J, 4, 4)."""
    return torch.einsum("vj,bjrc->bvrc", lbs_weights.to(transforms.dtype), transforms)


def skinning(v_posed: torch.Tensor, transforms: torch.Tensor, lbs_weights: torch.Tensor) -> torch.Tensor:
    """Linear blend skinning: rest-pose vertices (B, V, 3) to posed ones."""
    T = _blended(transforms, lbs_weights)
    return (T[..., :3, :3] @ v_posed[..., None])[..., 0] + T[..., :3, 3]


def inverse_skinning(v_posed: torch.Tensor, transforms: torch.Tensor, lbs_weights: torch.Tensor) -> torch.Tensor:
    """Undo `skinning`: posed vertices (B, V, 3) back to the rest pose, through each vertex's inverted transform."""
    T = torch.linalg.inv(_blended(transforms, lbs_weights))
    return (T[..., :3, :3] @ v_posed[..., None])[..., 0] + T[..., :3, 3]


def lbs(pose, v_shaped, posedirs, J_regressor, parents, lbs_weights, pose2rot=True, dtype=torch.float32):
    """A shaped mesh (B, V, 3), posed: pose correctives, then skinning.

    posedirs (P, V·3) map the flattened (R − I) of every joint but the root (P = (J − 1)·9) to vertex offsets.
    Returns (vertices (B, V, 3), posed joints (B, J, 3), the neck's skinning transform (B, 4, 4)): the third is
    what LAM's callers take from this function.
    """
    B = pose.shape[0]
    rot = _rotations(pose, pose2rot, dtype)
    eye = torch.eye(3, dtype=dtype, device=rot.device)
    pose_feature = (rot[:, 1:] - eye).reshape(B, -1)
    v_posed = v_shaped + (pose_feature @ posedirs).view(B, -1, 3)
    joints = vertices2joints(J_regressor, v_shaped)
    posed_joints, transforms = batch_rigid_transform(rot, joints, parents, dtype=dtype)
    return skinning(v_posed, transforms, lbs_weights), posed_joints, transforms[:, 1]
