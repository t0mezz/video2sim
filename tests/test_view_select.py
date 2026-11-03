"""Coverage-based view selection (S12.9, task 9.3)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from numpy.typing import NDArray

from video2sim.schemas import SE3, CameraFrame, Intrinsics, Plane, WorldModel
from video2sim.stages.view_select import ViewConfig, compute_visibility, select_views

ROOT = Path(__file__).resolve().parent.parent


def _down_look(x: float, y: float, h: float) -> SE3:
    """Camera at (x, y, h) looking straight down (OpenCV +z forward)."""
    a = np.diag([1.0, -1.0, -1.0, 1.0])
    a[:3, 3] = [x, y, h]
    return SE3.from_numpy(a)


def _world(cameras: list[tuple[float, float, float]]) -> WorldModel:
    return WorldModel(
        intrinsics=Intrinsics(
            fx=500.0, fy=500.0, cx=960.0, cy=540.0, width_px=1920, height_px=1080
        ),
        frames=[
            CameraFrame(
                frame_index=i, image_path=f"frames/{i}.png", T_world_from_camera=_down_look(x, y, h)
            )
            for i, (x, y, h) in enumerate(cameras)
        ],
        scale_factor=1.0,
        scale_source="marker",
        table_plane=Plane(normal=(0.0, 0.0, 1.0), offset_m=0.0),
        point_cloud_path="reconstruction/points.ply",
        mean_reprojection_error_px=0.5,
        registered_fraction=1.0,
    )


def _cfg(count: int = 6, min_views_per_cell: int = 1) -> ViewConfig:
    return ViewConfig(
        mode="coverage",
        count=count,
        cell_m=0.02,
        probe_heights_m=(0.03, 0.10),
        min_views_per_cell=min_views_per_cell,
        workspace_margin_m=0.10,
        max_image_px=1024,
        table_margin_m=0.003,
        max_height_m=0.35,
    )


def _wall_scene() -> tuple[WorldModel, NDArray[np.float64]]:
    """One camera at height 1, a 0.3 m wall at x = 0.4, floor points out to x = 1."""
    floor_x = np.linspace(0.1, 1.0, 46)
    floor = np.stack([floor_x, np.zeros_like(floor_x), np.full_like(floor_x, 0.01)], axis=1)
    wall_y = np.linspace(-0.08, 0.08, 5)
    wall = np.stack([np.full_like(wall_y, 0.4), wall_y, np.full_like(wall_y, 0.3)], axis=1)
    return _world([(0.0, 0.0, 1.0)]), np.concatenate([floor, wall])


def _gap_scene() -> tuple[WorldModel, NDArray[np.float64]]:
    """A 0.3 m wall at x = 0 with cameras at x = -0.8 and x = +0.8, height 1."""
    y = np.linspace(-0.1, 0.1, 5)
    wall = np.stack([np.zeros_like(y), y, np.full_like(y, 0.3)], axis=1)
    xx = np.linspace(-0.9, 0.9, 37)
    floor = np.stack([xx, np.zeros_like(xx), np.full_like(xx, 0.01)], axis=1)
    cloud = np.concatenate([floor, wall])
    return _world([(-0.8, 0.0, 1.0), (0.8, 0.0, 1.0)]), cloud


def test_shadow_length_matches_the_closed_form() -> None:
    # Camera at height H = 1 above the table, occluder of height h = 0.3 whose far edge
    # (with the one-cell dilation) is at d = 0.44: shadow length on the table L = d h/(H-h),
    # so cells past x = 0.63 are seen and the lower probe binds first at x = 0.61.
    world, cloud = _wall_scene()
    visibility, lo, shape = compute_visibility(world, cloud, _cfg())
    assert visibility.shape == (1, int(np.prod(shape)))
    centers_x = lo[0] + (np.arange(shape[0]) + 0.5) * 0.02
    # Center row only: rows off to the side see past the wall, as they should.
    seen = visibility[0].reshape(shape[1], shape[0])[shape[1] // 2]
    shadow = (centers_x > 0.45) & (centers_x <= 0.57)
    assert not seen[shadow].any()
    assert seen[centers_x >= 0.67].all()


def test_two_opposite_cameras_cover_what_one_alone_does_not() -> None:
    world, cloud = _gap_scene()
    visibility, _, _ = compute_visibility(world, cloud, _cfg())
    left_only = visibility[0] & ~visibility[1]
    right_only = visibility[1] & ~visibility[0]
    assert left_only.any() and right_only.any()
    got = select_views(world, cloud, _cfg(count=2, min_views_per_cell=1))
    assert got.frames == [0, 1]
    # Wall-interior cells are unseeable from anywhere; everything else is covered.
    assert got.coverage >= 0.99


def test_small_cap_reports_partial_coverage() -> None:
    world, cloud = _gap_scene()
    got = select_views(world, cloud, _cfg(count=1, min_views_per_cell=1))
    assert len(got.frames) == 1
    visibility, _, _ = compute_visibility(world, cloud, _cfg())
    assert got.coverage == max(visibility[0].mean(), visibility[1].mean()) < 1.0
    assert (got.cell_view_counts >= 1).mean() == got.coverage


def test_selection_is_deterministic() -> None:
    _, cloud = _wall_scene()
    world2 = _world([(0.0, 0.0, 1.0), (0.5, 0.2, 1.0), (-0.4, 0.1, 1.2)])
    first = select_views(world2, cloud, _cfg(count=2, min_views_per_cell=1))
    second = select_views(world2, cloud, _cfg(count=2, min_views_per_cell=1))
    assert first.frames == second.frames
    assert first.coverage == second.coverage


def test_empty_workspace_falls_back_to_even_frames() -> None:
    world = _world([(float(i), 0.0, 1.0) for i in range(5)])
    cloud = np.zeros((10, 3))  # all on the plane: nothing above the table margin
    got = select_views(world, cloud, _cfg(count=2))
    assert got.frames == [0, 4]
    assert got.coverage == 0.0
    assert got.reason != ""


def test_view_config_loads_from_defaults_and_rejects_bad_mode() -> None:
    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    views = ViewConfig.from_config(cfg)
    assert views.mode == "even"
    assert views.count == 6
    bad = yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    bad["segmentation"]["router"]["views"]["mode"] = "orbit"
    try:
        ViewConfig.from_config(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("bad views.mode was accepted")
