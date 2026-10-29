from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from video2sim.schemas import SE3, CameraFrame, Intrinsics, Plane, WorldModel
from video2sim.stages.lift import LiftConfig
from video2sim.stages.primitive_fit import Kind, PrimitiveConfig, _outline, fit_clusters
from video2sim.stages.segmentation import BoolMask
from video2sim.stages.tracking import Track

K = Intrinsics(fx=500, fy=500, cx=320, cy=240, width_px=640, height_px=480)
N_VIEWS = 12

LIFT = LiftConfig(
    voxel_m=0.005,
    max_height_m=0.2,
    margin_m=0.05,
    max_extent_m=1.0,
    max_views=N_VIEWS,
    mask_downscale=2,
    min_views=4,
    min_agreement=0.9,
    min_voxels=20,
    min_points=8,
    table_margin_m=0.003,
    trim_points_per_side=0,
)
CFG = PrimitiveConfig(
    mask_downscale=2,
    min_views=5,
    min_mask_px=20,
    iterations=300,
    shrink_every=100,
    shrink=0.6,
    step_m=0.005,
    step_deg=5.0,
    min_size_m=0.005,
    max_cost=0.10,
    seed=0,
)


def _look_at(position: np.ndarray[Any, Any], target: np.ndarray[Any, Any]) -> SE3:
    z = target - position
    z /= np.linalg.norm(z)
    right = np.cross(z, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(z, right)
    pose = np.eye(4)
    pose[:3, :3] = np.stack([right, down, z], axis=1)
    pose[:3, 3] = position
    return SE3.from_numpy(pose)


def _world(arc_deg: float = 360.0) -> WorldModel:
    frames = []
    for i in range(N_VIEWS):
        a = np.radians(arc_deg) * i / N_VIEWS
        position = np.array([0.6 * np.cos(a), 0.6 * np.sin(a), 0.5])
        frames.append(
            CameraFrame(
                frame_index=i,
                image_path=f"frames/{i:04d}.jpg",
                T_world_from_camera=_look_at(position, np.array([0.0, 0.0, 0.03])),
            )
        )
    return WorldModel(
        intrinsics=K,
        frames=frames,
        scale_factor=1.0,
        scale_source="marker",
        table_plane=Plane(normal=(0.0, 0.0, 1.0), offset_m=0.0),
        point_cloud_path="reconstruction/points.ply",
        mean_reprojection_error_px=0.5,
        registered_fraction=1.0,
    )


def _masks(world: WorldModel, solid: np.ndarray[Any, Any]) -> dict[int, BoolMask]:
    """Masks of a convex solid given by dense samples (N, 3), one per view."""
    import cv2

    masks: dict[int, BoolMask] = {}
    for frame in world.frames:
        pose = frame.T_world_from_camera.to_numpy()
        cam = (solid - pose[:3, 3]) @ pose[:3, :3]
        uv: Any = np.round(
            np.c_[K.fx * cam[:, 0] / cam[:, 2] + K.cx, K.fy * cam[:, 1] / cam[:, 2] + K.cy]
        ).astype(np.int32)
        canvas = np.zeros((K.height_px, K.width_px), dtype=np.uint8)
        cv2.fillConvexPoly(canvas, cv2.convexHull(uv), 1)
        masks[frame.frame_index] = canvas.astype(bool)
    return masks


def _solid(kind: Kind, size: tuple[float, float, float], yaw_deg: float = 0.0) -> Any:
    """Dense outline samples of a primitive at the origin, made independently of the fit."""
    if kind == "sphere":
        r = size[0] / 2
        t, p = np.meshgrid(np.linspace(0, np.pi, 40), np.linspace(0, 2 * np.pi, 80))
        return np.c_[
            (r * np.sin(t) * np.cos(p)).ravel(),
            (r * np.sin(t) * np.sin(p)).ravel(),
            (r + r * np.cos(t)).ravel(),
        ]
    params = np.array([0.0, 0.0, np.radians(yaw_deg), *size])
    return _outline(kind, params, 0.0)


def _surface_sample(solid: Any, n: int = 300) -> Any:
    """Points inside the convex solid, shrunk towards its centre: the sparse points of a
    textured object never reach its edges."""
    rng = np.random.default_rng(1)
    centre = solid.mean(axis=0)
    picks = solid[rng.integers(0, len(solid), n)]
    return centre + 0.8 * (picks - centre)


@pytest.mark.parametrize(
    ("kind", "size", "yaw"),
    [
        ("box", (0.10, 0.06, 0.05), 20.0),
        ("cylinder", (0.08, 0.08, 0.12), 0.0),
        ("sphere", (0.10, 0.10, 0.10), 0.0),
    ],
)
def test_a_primitive_is_recovered_from_its_masks(
    kind: Kind, size: tuple[float, float, float], yaw: float
) -> None:
    world = _world()
    solid = _solid(kind, size, yaw)
    track = Track(0, _masks(world, solid))
    points = _surface_sample(solid)
    fit = fit_clusters([track], {0: points}, world, LIFT, CFG)[0]
    assert fit.source == "primitive" and fit.primitive is not None
    assert fit.primitive.kind == kind
    got = [*sorted(fit.primitive.size_m[:2], reverse=True), fit.primitive.size_m[2]]
    want = [*sorted(size[:2], reverse=True), size[2]]
    # The start (shrunk points) is 20% small; the masks bring it to within two mask pixels
    # (a pixel of the downscaled mask is 2.4 mm at 0.6 m here).
    assert np.allclose(got, want, atol=0.005)
    assert np.min(fit.cluster[:, 2]) >= -1e-9


def test_the_fit_is_repeatable() -> None:
    world = _world()
    solid = _solid("box", (0.08, 0.05, 0.04), 10.0)
    track = Track(2, _masks(world, solid))
    points = _surface_sample(solid)
    a = fit_clusters([track], {2: points}, world, LIFT, CFG)[2]
    b = fit_clusters([track], {2: points}, world, LIFT, CFG)[2]
    assert a.primitive == b.primitive


def test_a_poor_fit_keeps_the_points_in_a_union() -> None:
    world = _world()
    solid = _solid("box", (0.08, 0.05, 0.04))
    points = _surface_sample(solid)
    strict = replace(CFG, max_cost=1e-6)
    fit = fit_clusters([Track(0, _masks(world, solid))], {0: points}, world, LIFT, strict)[0]
    assert fit.source == "union" and fit.primitive is not None
    assert len(fit.cluster) == len(fit.primitive.outline()) + len(points)
    assert "union" in fit.reason


def test_too_few_whole_masks_leave_the_points() -> None:
    world = _world()
    solid = _solid("box", (0.08, 0.05, 0.04))
    masks = _masks(world, solid)
    few = {f: m for f, m in masks.items() if f < 3}
    points = _surface_sample(solid)
    fit = fit_clusters([Track(0, few)], {0: points}, world, LIFT, CFG)[0]
    assert fit.source == "points" and fit.primitive is None
    assert np.array_equal(fit.cluster, points)


def test_another_track_in_front_does_not_shrink_the_fit() -> None:
    """Outline pixels on another track's mask cost nothing: the object may be hidden there."""
    world = _world()
    size = (0.10, 0.06, 0.05)
    solid = _solid("box", size)
    full = _masks(world, solid)
    occluder: dict[int, BoolMask] = {}
    cut: dict[int, BoolMask] = {}
    for f, m in full.items():
        cols = np.nonzero(m.any(axis=0))[0]
        right = np.zeros_like(m)
        right[:, (int(np.min(cols)) + int(np.max(cols))) // 2 + 1 :] = True
        occluder[f] = m & right
        cut[f] = m & ~right
    tracks = [Track(0, cut), Track(1, occluder)]
    fit = fit_clusters(tracks, {0: _surface_sample(solid)}, world, LIFT, CFG)[0]
    assert fit.primitive is not None
    got = sorted(fit.primitive.size_m[:2], reverse=True)
    assert got[0] >= 0.09  # not cut to the visible half


def test_config_rejects_bad_values() -> None:
    base = {k: getattr(CFG, k) for k in CFG.__dataclass_fields__}
    assert PrimitiveConfig.from_config({"segmentation": {"primitive_fit": base}}) == CFG
    for key, value in (("iterations", 0), ("shrink", 1.0), ("max_cost", 0.0)):
        with pytest.raises(ValueError):
            PrimitiveConfig.from_config({"segmentation": {"primitive_fit": {**base, key: value}}})
