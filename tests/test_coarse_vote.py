"""The coarse pass of the voxel vote (TODO.md N12) skips only voxels that cannot be members.

The pruned vote must give every track the members of the full vote, bit for bit, whatever the
lens model, occlusion rule, thresholds or grid offset.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from test_exact_speedups import CONFIG, _box_masks, _world
from video2sim.schemas import Intrinsics
from video2sim.stages import lift
from video2sim.stages.lift import LiftConfig, _grid, _pixel_bounds, _pixels, _views
from video2sim.stages.tracking import Track

DISTORTED = Intrinsics(
    fx=500, fy=480, cx=318, cy=243, width_px=640, height_px=480,
    distortion=[-0.21, 0.09, 0.0015, -0.0012, -0.02],
)  # fmt: skip


def _members(ratio: Any, best: Any, agree: Any) -> list[Any]:
    return [np.flatnonzero((ratio >= a) & (best == r)) for r, a in enumerate(agree)]


def _scene(distorted: bool) -> tuple[Any, list[Track]]:
    world = _world(12)
    if distorted:
        world = world.model_copy(update={"intrinsics": DISTORTED})
    return world, [
        Track(0, _box_masks(world, (0.08, 0.06, 0.10), (0.0, 0.0))),
        Track(3, _box_masks(world, (0.05, 0.05, 0.05), (0.07, 0.02), skip=3)),
        Track(7, _box_masks(world, (0.30, 0.10, 0.04), (-0.05, -0.12))),
        Track(9, _box_masks(world, (0.02, 0.02, 0.02), (0.12, -0.10), skip=2)),
    ]


@pytest.mark.parametrize("distorted", [False, True])
@pytest.mark.parametrize("occlusion_aware", [True, False])
@pytest.mark.parametrize("agreement", [0.9, 0.6, 0.3])
@pytest.mark.parametrize("z0", [0, 7])
def test_pruning_keeps_every_member(
    distorted: bool, occlusion_aware: bool, agreement: float, z0: int
) -> None:
    world, tracks = _scene(distorted)
    cfg = replace(
        LiftConfig.from_config(CONFIG),
        occlusion_aware=occlusion_aware,
        min_views=3,
        margin_m=0.06,
        max_height_m=0.15,
    )
    feet = np.array([[-0.2, -0.2], [0.2, 0.1]])
    grid = _grid(feet, world.table_plane, cfg, cfg.max_height_m)
    if z0:
        nx, ny, nz = grid.shape
        grid = replace(grid, shape=(nx, ny, nz - z0), z0=z0)
    views = _views(world, cfg)
    agree = np.array([agreement, agreement, 0.95, 0.5])  # thresholds differ per track
    dense_ratio, dense_best = lift._vote(grid, tracks, world, cfg, views)
    ratio, best = lift._vote(grid, tracks, world, cfg, views, chunk=7919, agree=agree)
    dense = _members(dense_ratio, dense_best, agree)
    assert sum(len(m) for m in dense) > 0  # the scene has members to keep
    for a, b in zip(dense, _members(ratio, best, agree), strict=True):
        assert np.array_equal(a, b)
    voted = ratio > 0
    assert np.array_equal(ratio[voted], dense_ratio[voted])  # a voted voxel is voted as before
    prepared = lift._vote_views(tracks, world, cfg, views)
    kept = lift._candidates(grid, len(tracks), prepared, world, cfg, agree)
    assert len(kept) < grid.size // 2  # and most of the grid is skipped


def test_pixel_bounds_enclose_the_lens_model() -> None:
    rng = np.random.default_rng(4)
    for _ in range(300):
        x0, y0 = rng.uniform(-0.9, 0.9, 2)
        x1, y1 = x0 + rng.uniform(0, 0.3), y0 + rng.uniform(0, 0.3)
        (u0, u1), (v0, v1) = _pixel_bounds(
            (np.array([x0]), np.array([x1])), (np.array([y0]), np.array([y1])), DISTORTED
        )
        x = rng.uniform(x0, x1, 400)
        y = rng.uniform(y0, y1, 400)
        corners_x = np.array([x0, x1, x0, x1])
        corners_y = np.array([y0, y0, y1, y1])
        u, v = _pixels(np.r_[x, corners_x], np.r_[y, corners_y], DISTORTED)
        assert float(np.min(u)) >= u0[0] - 1e-9 and float(np.max(u)) <= u1[0] + 1e-9
        assert float(np.min(v)) >= v0[0] - 1e-9 and float(np.max(v)) <= v1[0] + 1e-9


def test_the_cells_do_not_depend_on_the_chunk() -> None:
    world, tracks = _scene(True)
    cfg = replace(LiftConfig.from_config(CONFIG), min_views=3, margin_m=0.06, max_height_m=0.15)
    grid = _grid(np.array([[-0.2, -0.2], [0.2, 0.1]]), world.table_plane, cfg, cfg.max_height_m)
    prepared = lift._vote_views(tracks, world, cfg, _views(world, cfg))
    agree = np.full(len(tracks), 0.9)
    whole = lift._candidates(grid, len(tracks), prepared, world, cfg, agree)
    assert np.array_equal(
        whole, lift._candidates(grid, len(tracks), prepared, world, cfg, agree, 37)
    )
