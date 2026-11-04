from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from video2sim.evaluation.geometry import box_vertices
from video2sim.schemas import SE3, CameraFrame, Intrinsics, Plane, WorldModel
from video2sim.stages.lift import (
    LiftConfig,
    _cap_reason,
    _duplicate_tracks,
    _largest_component,
    footprint_axes,
    lift_points,
    lift_tracks,
    trim_extremes,
)
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


def _world(radius: float = 0.6, height: float = 0.5, tz: float = 0.03) -> WorldModel:
    frames = []
    for i in range(N_VIEWS):
        a = 2 * np.pi * i / N_VIEWS
        position = np.array([radius * np.cos(a), radius * np.sin(a), height])
        frames.append(
            CameraFrame(
                frame_index=i,
                image_path=f"frames/{i:04d}.jpg",
                T_world_from_camera=_look_at(position, np.array([0.0, 0.0, tz])),
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
    masks: dict[int, BoolMask] = {}
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
    trim_points_per_side=0,
    occlusion_aware=False,
    small_mask_px=15000,
    small_min_agreement=0.9,
    small_min_voxels=20,
    dup_dist_m=0.02,
    tall_height_m=0.35,
    cap_top_m=0.02,
    cap_frac=0.5,
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


def test_a_tall_box_is_kept_when_it_caps_below_the_tall_grid() -> None:
    world = _world(radius=1.2, height=1.0, tz=0.3)  # high enough to see the box's top
    tall = replace(CFG, max_height_m=0.2, tall_height_m=0.6)
    size = (0.08, 0.06, 0.5)
    clusters, report = lift_tracks([Track(0, _box_masks(world, size, (0.0, 0.0)))], world, tall)
    assert 0 in clusters, report.dropped
    assert np.min(clusters[0][:, 2]) >= 0.0
    assert np.max(clusters[0][:, 2]) >= 0.45  # not cut off at the 0.2 m first pass
    assert np.max(clusters[0][:, 2]) <= 0.5 + 0.02


def test_the_tall_pass_only_votes_above_the_first_pass(monkeypatch: Any) -> None:
    from video2sim.stages import lift as lift_module

    world = _world(radius=1.2, height=1.0, tz=0.3)
    tall = replace(CFG, max_height_m=0.2, tall_height_m=0.6)
    voted: list[np.ndarray[Any, Any]] = []
    real_vote = lift_module._vote

    def spy(grid: Any, tracks: Any, w: Any, cfg: Any, views: Any, **kw: Any) -> Any:
        voted.append(grid.centers(np.arange(grid.size)))
        return real_vote(grid, tracks, w, cfg, views, **kw)

    monkeypatch.setattr(lift_module, "_vote", spy)
    size = (0.08, 0.06, 0.5)
    clusters, report = lift_tracks([Track(0, _box_masks(world, size, (0.0, 0.0)))], world, tall)
    assert 0 in clusters, report.dropped
    assert len(voted) == 2
    assert float(voted[1][:, 2].min()) >= tall.max_height_m


def test_a_column_through_the_tall_grid_is_dropped_as_background() -> None:
    world = _world(radius=1.2, height=1.0, tz=0.3)
    tall = replace(CFG, max_height_m=0.2, tall_height_m=0.6)
    clusters, report = lift_tracks(
        [Track(0, _box_masks(world, (0.08, 0.06, 0.9), (0.0, 0.0)))], world, tall
    )
    assert 0 not in clusters
    assert "not a tabletop object" in report.dropped[0]


def _column(nx: int, ny: int, nz: int, layers: int, inset: int) -> Any:
    cells = [
        [x, y, z]
        for z in range(layers)
        for x in range(inset, nx - inset)
        for y in range(inset, ny - inset)
    ]
    return np.array(cells, dtype=np.int64)


def test_cap_reason_tells_a_capped_object_from_background() -> None:
    shape = (20, 20, 60)
    assert _cap_reason(_column(20, 20, 60, 60, 4), shape, CFG) == "fills the grid to its top"
    assert _cap_reason(_column(20, 20, 60, 40, 4), shape, CFG) is None
    assert _cap_reason(_column(20, 20, 60, 40, 0), shape, CFG) == "fills the grid to its edge"


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


def test_occluded_views_are_excused_with_occlusion_aware() -> None:
    world = _world()
    back = _box_masks(world, (0.04, 0.04, 0.04), (0.0, 0.0))
    front = _box_masks(world, (0.08, 0.08, 0.16), (0.09, 0.0))
    # SAM sees only the visible part: the back box minus the blocker's silhouette.
    visible = {i: m & ~front[i] for i, m in back.items()}
    tracks = [Track(0, visible), Track(1, front)]
    clusters, report = lift_tracks(tracks, world, CFG)
    assert 0 not in clusters and 1 in clusters
    assert report.dropped[0] == "no voxel is inside the mask in enough views"
    aware = replace(CFG, occlusion_aware=True)
    clusters, report = lift_tracks(tracks, world, aware)
    assert sorted(clusters) == [0, 1] and report.dropped == {}


def test_coinciding_clusters_drop_the_smaller_as_a_duplicate() -> None:
    rng = np.random.default_rng(0)
    cloud = rng.random((50, 3)) * 0.05
    shifted = cloud + np.array([0.005, 0.0, 0.0])  # 5 mm over: the same object twice
    far = cloud + np.array([0.16, 0.0, 0.0])  # a genuinely separate object
    assert _duplicate_tracks({0: cloud, 1: shifted}, 0.02) == {1: "duplicate of track 0"}
    assert _duplicate_tracks({0: cloud, 2: far}, 0.02) == {}
    assert _duplicate_tracks({0: cloud, 1: shifted}, 0.0) == {}


def test_small_track_uses_the_relaxed_agreement() -> None:
    world = _world()
    masks = _box_masks(world, (0.04, 0.04, 0.04), (0.0, 0.0))
    assert float(np.median([m.sum() for m in masks.values()])) < CFG.small_mask_px
    jittered = dict(masks)
    for i in (3, 7):
        jittered[i] = np.roll(masks[i], 30, axis=(0, 1))  # 10 of 12 views agree: 0.83
    clusters, report = lift_tracks([Track(0, jittered)], world, CFG)
    assert 0 not in clusters
    relaxed = replace(CFG, small_min_agreement=0.75)
    clusters, report = lift_tracks([Track(0, jittered)], world, relaxed)
    assert 0 in clusters and report.dropped == {}


def test_largest_component_drops_a_detached_voxel() -> None:
    cells = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 1], [9, 9, 9]], dtype=np.int64)
    assert _largest_component(cells).tolist() == [True, True, True, False]


def test_config_rejects_bad_values() -> None:
    base = {
        "voxel_m": 0.005, "max_height_m": 0.3, "margin_m": 0.1, "max_extent_m": 1.5,
        "max_views": 10, "mask_downscale": 4, "min_views": 5, "min_agreement": 0.85,
        "min_voxels": 20, "min_points": 8, "table_margin_m": 0.003, "trim_points_per_side": 2,
        "occlusion_aware": False, "small_mask_px": 15000, "small_min_agreement": 0.85,
        "small_min_voxels": 20, "dup_dist_m": 0.02, "tall_height_m": 1.0,
        "cap_top_m": 0.02, "cap_frac": 0.5, "edge_frac": 0.05,
    }  # fmt: skip
    assert LiftConfig.from_config({"segmentation": {"lift": base}}).max_views == 10
    for key, value in (
        ("voxel_m", 0.0),
        ("min_views", 0),
        ("min_agreement", 1.5),
        ("small_min_agreement", 0.0),
        ("small_min_voxels", 0),
        ("small_mask_px", 0.0),
        ("dup_dist_m", -0.01),
        ("trim_points_per_side", -1),
        ("tall_height_m", 0.1),
        ("cap_top_m", 0.0),
        ("cap_frac", 0.0),
        ("edge_frac", 0.0),
    ):
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
    clusters, report, _vote = lift_points(
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


def test_point_vote_names_the_explaining_track() -> None:
    world = _world()
    size = (0.08, 0.06, 0.05)
    box = _surface_points(size, (0.0, 0.0))
    far = np.array([[0.15, 0.15, 0.03]])
    cloud = np.concatenate([box, far])
    _, _, vote = lift_points([Track(4, _box_masks(world, size, (0.0, 0.0)))], world, cloud, CFG)
    inside = vote.best_ratio[: len(box)]
    # Surface points are explained by track 4; the far table point by nothing.
    assert float(np.median(inside)) >= 0.5
    assert np.all(vote.best_track[: len(box)][inside >= 0.5] == 4)
    assert vote.best_ratio[-1] == 0.0
    assert vote.best_seen[-1] >= CFG.min_views  # seen, just never inside a mask


def test_points_of_a_track_seen_in_one_view_are_dropped() -> None:
    world = _world()
    size = (0.05, 0.05, 0.05)
    masks = _box_masks(world, size, (0.0, 0.0))
    cloud = _surface_points(size, (0.0, 0.0))
    clusters, report, _vote = lift_points(
        [Track(0, masks), Track(1, {0: masks[0]})], world, cloud, CFG
    )
    assert 0 in clusters and 1 not in clusters
    assert "sparse points" in report.dropped[1]


def test_trimming_drops_a_stray_point_and_keeps_the_box() -> None:
    size = (0.08, 0.06, 0.05)
    box = _surface_points(size, (0.0, 0.0))
    stray = np.array([[0.0, 0.0, 0.08]])  # 3 cm above the top, like a short SfM track
    trimmed = trim_extremes(np.concatenate([box, stray]), 2, 8)
    assert not np.any(np.all(np.isclose(trimmed, stray[0]), axis=1))
    assert len(trimmed) >= len(box) + 1 - 12
    # Two points per side cost the box under 5 mm on dense faces.
    assert np.all(_extent(trimmed) >= np.array(size) - 0.005)
    assert np.all(_extent(trimmed) <= np.array(size) + 1e-9)


def test_trimming_leaves_a_sparse_cluster_whole() -> None:
    few = _surface_points((0.05, 0.05, 0.05), (0.0, 0.0))[:15]
    assert len(trim_extremes(few, 2, 8)) == 15  # 15 <= 6 * 2 + 8
    assert len(trim_extremes(few, 0, 8)) == 15


def test_trimming_follows_a_rotated_footprint() -> None:
    a = np.radians(30.0)
    rot = np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])
    box = _surface_points((0.12, 0.04, 0.03), (0.0, 0.0)) @ rot.T
    u, n = footprint_axes(box)
    assert abs(abs(float(u @ [np.cos(a), np.sin(a)])) - 1.0) < 1e-3
    assert abs(float(u @ n)) < 1e-9
    # The trimmed box along the long side loses only a few millimetres.
    trimmed = trim_extremes(box, 2, 8)
    long_side = np.ptp(trimmed[:, :2] @ u)
    assert 0.115 <= long_side <= 0.12 + 1e-9


def test_lift_points_trims_each_cluster() -> None:
    world = _world()
    size = (0.08, 0.06, 0.05)
    cloud = np.concatenate([_surface_points(size, (0.0, 0.0)), [[0.0, 0.0, 0.07]]])
    masks = _box_masks(world, size, (0.0, 0.0))
    untrimmed, _, _v1 = lift_points([Track(4, masks)], world, cloud, CFG)
    trimmed, _, _v2 = lift_points(
        [Track(4, masks)], world, cloud, LiftConfig(**{**CFG.__dict__, "trim_points_per_side": 2})
    )
    assert len(trimmed[4]) == len(untrimmed[4]) - 12


def _sheet(x: int, ys: range, zs: range) -> Any:
    return np.array([[x, y, z] for z in zs for y in ys], dtype=np.int64)


def test_cap_reason_ignores_a_thin_incidental_sheet() -> None:
    shape = (20, 20, 60)
    core = _column(20, 20, 60, 40, 4)  # capped, interior: 12 by 12 by 40
    sheet = _sheet(0, range(5, 10), range(10, 30))  # 100 voxels on the x face
    cells = np.concatenate([core, sheet])
    assert len(sheet) / len(cells) < 0.05
    assert _cap_reason(cells, shape, CFG) is None


def test_cap_reason_drops_a_wall_sharing_the_edge() -> None:
    shape = (20, 20, 60)
    wall = np.array(
        [[x, y, z] for z in range(40) for y in range(20) for x in (0, 1)], dtype=np.int64
    )
    assert _cap_reason(wall, shape, CFG) == "fills the grid to its edge"


def test_cap_reason_edge_bar_is_configurable() -> None:
    shape = (20, 20, 60)
    core = _column(20, 20, 60, 40, 4)
    sheet = _sheet(0, range(5, 10), range(10, 30))
    cells = np.concatenate([core, sheet])
    strict = LiftConfig(**{**CFG.__dict__, "edge_frac": 0.01})
    assert _cap_reason(cells, shape, strict) == "fills the grid to its edge"
