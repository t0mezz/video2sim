from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from video2sim.evaluation.geometry import box_vertices
from video2sim.schemas import SE3, CameraFrame, Intrinsics, Plane, WorldModel
from video2sim.stages.lift import LiftConfig, _largest_component, lift_points, lift_tracks
from video2sim.stages.segmentation import BoolMask
from video2sim.stages.tracking import Track

K = Intrinsics(fx=500, fy=500, cx=320, cy=240, width_px=640, height_px=480)
N_VIEWS = 12


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


def _world() -> WorldModel:
    frames = []
    for i in range(N_VIEWS):
        a = 2 * np.pi * i / N_VIEWS
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


def _box_masks(
    world: WorldModel, size: tuple[float, float, float], xy: tuple[float, float]
) -> dict[int, BoolMask]:
    import cv2

    corners = box_vertices(size, (xy[0], xy[1], size[2] / 2), 0.0)
    masks = {}
    for frame in world.frames:
        cam = (np.linalg.inv(frame.T_world_from_camera.to_numpy()) @ np.c_[corners, np.ones(8)].T).T
        u = K.fx * cam[:, 0] / cam[:, 2] + K.cx
        v = K.fy * cam[:, 1] / cam[:, 2] + K.cy
        canvas = np.zeros((K.height_px, K.width_px), dtype=np.uint8)
        points: Any = np.round(np.c_[u, v]).astype(np.int32)
        hull: Any = cv2.convexHull(points)
        cv2.fillConvexPoly(canvas, hull, 1)
        masks[frame.frame_index] = canvas.astype(bool)
    return masks


CFG = LiftConfig(
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
)


def _extent(points: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return np.asarray(points.max(axis=0) - points.min(axis=0))


def test_a_box_comes_back_at_its_size() -> None:
    world = _world()
    size = (0.08, 0.06, 0.05)
    clusters, report = lift_tracks([Track(0, _box_masks(world, size, (0.0, 0.0)))], world, CFG)
    assert report.dropped == {}
    # A visual hull is a little larger than the object; it must not be smaller or much larger.
    extent = _extent(clusters[0])
    assert np.all(extent >= np.array(size) - CFG.voxel_m)
    assert np.all(extent <= np.array(size) + 0.02)
    assert np.min(clusters[0][:, 2]) >= 0.0


def test_two_objects_give_two_separate_clusters() -> None:
    world = _world()
    a = Track(3, _box_masks(world, (0.05, 0.05, 0.05), (-0.08, 0.0)))
    b = Track(7, _box_masks(world, (0.05, 0.05, 0.08), (0.08, 0.0)))
    clusters, report = lift_tracks([a, b], world, CFG)
    assert sorted(clusters) == [3, 7] and report.dropped == {}
    assert clusters[3][:, 0].mean() < 0 < clusters[7][:, 0].mean()


def test_a_mask_seen_in_one_view_adds_nothing() -> None:
    world = _world()
    masks = _box_masks(world, (0.05, 0.05, 0.05), (0.0, 0.0))
    stray = {0: masks[0]}  # one view cannot reach min_views, let alone agree
    clusters, report = lift_tracks([Track(0, masks), Track(1, stray)], world, CFG)
    assert 0 in clusters and 1 not in clusters
    assert "no voxel" in report.dropped[1] or "voxels agree" in report.dropped[1]


def test_largest_component_drops_a_detached_voxel() -> None:
    cells = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 1], [9, 9, 9]], dtype=np.int64)
    assert _largest_component(cells).tolist() == [True, True, True, False]


def test_config_rejects_bad_values() -> None:
    base = {
        "voxel_m": 0.005, "max_height_m": 0.3, "margin_m": 0.1, "max_extent_m": 1.5,
        "max_views": 10, "mask_downscale": 4, "min_views": 5, "min_agreement": 0.85,
        "min_voxels": 20, "min_points": 8, "table_margin_m": 0.003,
    }  # fmt: skip
    assert LiftConfig.from_config({"segmentation": {"lift": base}}).max_views == 10
    for key, value in (("voxel_m", 0.0), ("min_views", 0), ("min_agreement", 1.5)):
        with pytest.raises(ValueError):
            LiftConfig.from_config({"segmentation": {"lift": {**base, key: value}}})


def _surface_points(size: tuple[float, float, float], xy: tuple[float, float]) -> Any:
    """Points on the six faces of a box resting on the table, 1 cm apart."""
    rng = np.random.default_rng(0)
    pts = []
    for axis in range(3):
        for side in (-0.5, 0.5):
            p = (rng.random((200, 3)) - 0.5) * np.array(size)
            p[:, axis] = side * size[axis]
            pts.append(p)
    return np.concatenate(pts) + np.array([xy[0], xy[1], size[2] / 2])


def test_voted_points_follow_the_surface_and_leave_out_the_table() -> None:
    world = _world()
    size = (0.08, 0.06, 0.05)
    box = _surface_points(size, (0.0, 0.0))
    table = np.array([[0.02, 0.0, 0.001], [0.0, 0.02, 0.0]])  # under table_margin_m
    far = np.array([[0.15, 0.15, 0.03]])  # on the table, outside every mask
    cloud = np.concatenate([box, table, far])
    clusters, report = lift_points(
        [Track(4, _box_masks(world, size, (0.0, 0.0)))], world, cloud, CFG
    )
    assert report.dropped == {}
    kept = clusters[4]
    # Points on the bottom face are the table. Edge points can fall just outside a mask.
    above = int(np.sum(box[:, 2] > CFG.table_margin_m))
    assert 0.8 * above <= len(kept) <= above
    assert np.all(kept[:, 2] > CFG.table_margin_m)
    assert not np.any(np.all(np.isclose(kept, far[0]), axis=1))
    # Surface points only reach the object's own extent, never past it.
    assert np.all(_extent(kept) <= np.array(size) + 1e-9)


def test_points_of_a_track_seen_in_one_view_are_dropped() -> None:
    world = _world()
    size = (0.05, 0.05, 0.05)
    masks = _box_masks(world, size, (0.0, 0.0))
    cloud = _surface_points(size, (0.0, 0.0))
    clusters, report = lift_points([Track(0, masks), Track(1, {0: masks[0]})], world, cloud, CFG)
    assert 0 in clusters and 1 not in clusters
    assert "sparse points" in report.dropped[1]
