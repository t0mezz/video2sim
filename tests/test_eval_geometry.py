from __future__ import annotations

import numpy as np
import pytest

from video2sim.evaluation.geometry import box_vertices, convex_hull_2d, resting_box


def test_hull_drops_interior_and_collinear_points() -> None:
    pts = np.array([[0, 0], [2, 0], [2, 2], [0, 2], [1, 1], [1, 0], [0.5, 1.5]], dtype=float)
    hull = convex_hull_2d(pts)
    assert {tuple(p) for p in hull} == {(0, 0), (2, 0), (2, 2), (0, 2)}


def test_recovers_rotated_box_exactly() -> None:
    # 10 x 4 x 5 cm at 35 degrees: the axis-aligned footprint would be 10.5 x 9.0 cm.
    verts = box_vertices((0.10, 0.04, 0.05), (0.30, 0.20, 0.025), 35.0)
    box = resting_box(verts)
    assert box.size_m == pytest.approx((0.10, 0.04, 0.05), abs=1e-9)
    assert box.center_m == pytest.approx((0.30, 0.20, 0.025), abs=1e-9)
    assert box.yaw_deg == pytest.approx(35.0, abs=1e-6)


@pytest.mark.parametrize("yaw", [0.0, 10.0, 90.0, 125.0, 179.0])
def test_yaw_round_trip_and_length_is_the_long_side(yaw: float) -> None:
    box = resting_box(box_vertices((0.12, 0.05, 0.08), (0.0, 0.0, 0.04), yaw))
    assert box.size_m[0] >= box.size_m[1]
    assert box.yaw_deg == pytest.approx(yaw % 180.0, abs=1e-6)


def test_interior_vertices_do_not_change_the_box() -> None:
    corners = box_vertices((0.10, 0.04, 0.05), (0.0, 0.0, 0.025), 20.0)
    rng = np.random.default_rng(0)
    inside = np.column_stack(
        [rng.uniform(-0.01, 0.01, 50), rng.uniform(-0.01, 0.01, 50), rng.uniform(0, 0.05, 50)]
    )
    a = resting_box(corners)
    b = resting_box(np.vstack([corners, inside]))
    assert b.size_m == pytest.approx(a.size_m, abs=1e-9)
    assert b.yaw_deg == pytest.approx(a.yaw_deg, abs=1e-6)


def test_degenerate_inputs() -> None:
    flat_line = resting_box(np.array([[0, 0, 0], [0.1, 0, 0], [0.05, 0, 0.02]], dtype=float))
    assert flat_line.size_m == pytest.approx((0.1, 0.0, 0.02))
    point = resting_box(np.array([[1.0, 2.0, 3.0]]))
    assert point.size_m == pytest.approx((0.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        resting_box(np.empty((0, 3)))
    with pytest.raises(ValueError):
        resting_box(np.array([[0.0, np.nan, 0.0]]))
