"""TRELLIS.2 geometry with SAM 3D's pose (the third arm of the P3.3 bake-off).

TRELLIS.2 places nothing in the image; SAM 3D does. The hybrid draws both models on the same
keyframe and seed, finds the rotation that turns TRELLIS.2's model frame onto SAM 3D's, and
returns the TRELLIS.2 mesh with SAM 3D's camera pose carried across that rotation. Registration
(P3.2) then starts from SAM 3D's guess instead of the canonical one in ``models/trellis2.py``.

The frames are matched on shape alone: both clouds are centred and scaled by their RMS radius,
then rotations are tried (the 24 turns of the cube, each followed by a yaw about model z, since
both models decode z-up) and the one with the smallest symmetric chamfer distance wins. A
symmetric object has several equally good answers; any of them gives a valid start, because
registration only needs the rotation to be near and tries yaw starts itself.
"""

from __future__ import annotations

import dataclasses
import itertools

import numpy as np

from video2sim.models.shape_model import F64, ShapeModel, ShapeSample, ShapeView
from video2sim.schemas import SE3
from video2sim.stages.registration import sample_surface

MODEL_NAME = "trellis2"  # the geometry's source, as ``RegisteredObject.geometry_source``


def cube_rotations() -> list[F64]:
    """The 24 proper rotations that permute and flip the coordinate axes."""
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            r = np.zeros((3, 3))
            for row, (col, sign) in enumerate(zip(perm, signs, strict=True)):
                r[row, col] = sign
            if np.linalg.det(r) > 0:
                out.append(r)
    return out


def _yaw(angle: float) -> F64:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _cloud(vertices: F64, faces: np.ndarray, count: int) -> F64:
    points = sample_surface(vertices, faces, count, np.random.default_rng(0))
    points = points - points.mean(axis=0)  # not the bounding box centre: that turns with the mesh
    out: F64 = points / np.sqrt((points**2).sum(axis=1).mean())
    return out


def frame_rotation(
    sam_vertices: F64,
    sam_faces: np.ndarray,
    trellis_vertices: F64,
    trellis_faces: np.ndarray,
    points: int = 600,
    yaw_step_deg: float = 15.0,
) -> tuple[F64, float]:
    """``R`` with ``x_sam ~ R @ x_trellis`` for matching points, and its chamfer distance in
    units of the RMS radius."""
    from scipy.spatial import cKDTree  # type: ignore[import-untyped]  # heavy import

    sam = _cloud(sam_vertices, sam_faces, points)
    tre = _cloud(trellis_vertices, trellis_faces, points)
    sam_tree, tre_tree = cKDTree(sam), cKDTree(tre)
    yaws = [_yaw(np.deg2rad(a)) for a in np.arange(0.0, 360.0, yaw_step_deg)]
    best_r, best = np.eye(3), np.inf
    for cube in cube_rotations():
        for yaw in yaws:
            r = yaw @ cube
            to_sam = sam_tree.query(tre @ r.T)[0].mean()
            to_tre = tre_tree.query(sam @ r)[0].mean()  # sam ~ r @ tre  <=>  tre ~ r.T @ sam
            cost = float(to_sam + to_tre)
            if cost < best:
                best_r, best = r, cost
    return best_r, best / 2.0


class HybridShape:
    """``ShapeModel`` that returns ``trellis``'s mesh posed by ``sam3d``'s layout."""

    name = MODEL_NAME

    def __init__(self, sam3d: ShapeModel, trellis: ShapeModel) -> None:
        self._sam3d = sam3d
        self._trellis = trellis

    def sample(self, object_id: str, view: ShapeView, seed: int) -> ShapeSample:
        pose = self._sam3d.sample(object_id, view, seed)
        mesh = self._trellis.sample(object_id, view, seed)
        r_rel, _ = frame_rotation(pose.vertices_model, pose.faces, mesh.vertices_model, mesh.faces)
        matrix = pose.T_camera_from_model.to_numpy().copy()
        matrix[:3, :3] = matrix[:3, :3] @ r_rel
        return dataclasses.replace(
            mesh,
            T_camera_from_model=SE3.from_numpy(matrix),
            checkpoint=f"{mesh.checkpoint}; pose {pose.checkpoint}",
        )
