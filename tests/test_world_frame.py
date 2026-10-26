from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from video2sim.stages.world_frame import (
    BoardLayout,
    PlaneFit,
    angle_between_deg,
    apply_scale,
    fit_board,
    fit_plane_ransac,
    fit_similarity,
    world_from_recon,
)

F64 = NDArray[np.float64]


def _rot(axis: tuple[float, float, float], deg: float) -> F64:
    k = np.asarray(axis, dtype=np.float64)
    k /= np.linalg.norm(k)
    a = np.radians(deg)
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.asarray(np.eye(3) + np.sin(a) * kx + (1 - np.cos(a)) * kx @ kx, dtype=np.float64)


def _layout(tmp_path: Path) -> BoardLayout:
    # A2-like board: tags 70 mm, centres at the corners and mid-edges of a 560 x 380 mm grid.
    centers = {0: (50, 50), 1: (510, 50), 2: (50, 330), 3: (510, 330), 4: (50, 190), 5: (510, 190)}
    data = {
        "tag_edge_mm_nominal": 70.0,
        "tags": {str(i): {"center_mm": list(c)} for i, c in centers.items()},
    }
    p = tmp_path / "layout.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return BoardLayout.from_make_board_json(p)


def test_layout_corner_geometry(tmp_path: Path) -> None:
    corners = _layout(tmp_path).tag_corners_m[0]
    assert corners.shape == (4, 3)
    assert np.allclose(np.linalg.norm(corners[1] - corners[0]), 0.07)
    assert np.allclose(corners[:, 2], 0.0)


def test_measured_tag_edge_overrides_nominal(tmp_path: Path) -> None:
    p = tmp_path / "layout.json"
    p.write_text(
        json.dumps({"tag_edge_mm_nominal": 70.0, "tags": {"0": {"center_mm": [50, 50]}}}),
        encoding="utf-8",
    )
    c = BoardLayout.from_make_board_json(p, tag_edge_m=0.0695).tag_corners_m[0]
    assert np.isclose(np.linalg.norm(c[1] - c[0]), 0.0695)


def test_fit_similarity_recovers_known_transform() -> None:
    rng = np.random.default_rng(1)
    src = rng.normal(size=(30, 3))
    rot, t, s = _rot((1, 2, 3), 40), np.array([1.0, -2.0, 0.5]), 0.37
    dst = s * (rot @ src.T).T + t
    fit = fit_similarity(src, dst)
    assert np.isclose(fit.scale, s)
    assert np.allclose(fit.R_board_from_recon, rot)
    assert np.allclose(fit.t_board_from_recon, t)
    assert fit.rms_m < 1e-9


def test_fit_similarity_never_returns_a_mirror() -> None:
    src = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0.0]])
    dst = src * np.array([1, 1, -1])  # planar points: a mirror and a rotation both fit
    assert np.linalg.det(fit_similarity(src, dst).R_board_from_recon) > 0


def test_scale_from_tags_within_marker_tolerance(tmp_path: Path) -> None:
    """Gate: distance between two known marker points correct within 2% after scaling."""
    layout = _layout(tmp_path)
    rng = np.random.default_rng(2)
    rot, t, s = _rot((0, 1, 1), 25), np.array([3.0, 1.0, 2.0]), 7.3  # recon units, s: m->units
    detected = {
        i: (c @ rot.T * s + t) + rng.normal(scale=0.002 * s, size=(4, 3))
        for i, c in layout.tag_corners_m.items()
    }
    fit = fit_board(layout, detected)
    measured = np.linalg.norm(apply_scale(detected[1][0] - detected[0][0], fit.scale))
    true = np.linalg.norm(layout.tag_corners_m[1][0] - layout.tag_corners_m[0][0])
    assert abs(measured - true) / true < 0.02


def test_fit_board_rejects_unknown_id_and_single_tag(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    c = layout.tag_corners_m[0]
    with pytest.raises(ValueError, match="not on the board"):
        fit_board(layout, {0: c, 99: c})
    with pytest.raises(ValueError, match="at least 2"):
        fit_board(layout, {0: c})


def _table_scene(tilt_deg: float, n: int = 2000) -> tuple[F64, F64]:
    """Table points plus clutter in a tilted recon frame. Returns (points, true up in recon)."""
    rng = np.random.default_rng(3)
    table = np.column_stack([rng.uniform(-0.5, 0.5, (n, 2)), rng.normal(0, 0.001, n)])
    clutter = np.column_stack([rng.uniform(-0.3, 0.3, (300, 2)), rng.uniform(0.01, 0.15, 300)])
    rot = _rot((1, 0.3, 0), tilt_deg)
    return (np.vstack([table, clutter]) @ rot.T), rot @ np.array([0.0, 0.0, 1.0])


def test_plane_ransac_up_axis_within_gate() -> None:
    """Gate: up axis within 2 degrees of the measured table normal (here: known truth)."""
    pts, up = _table_scene(35.0)
    fit = fit_plane_ransac(pts, inlier_threshold_m=0.004, iterations=300, seed=0, up_hint=up)
    assert angle_between_deg(np.asarray(fit.plane.normal), up) < 0.5
    assert fit.rms_m < 0.003
    assert fit.inlier_fraction > 0.8


def test_plane_ransac_is_seeded_and_flips_to_hint() -> None:
    pts, up = _table_scene(10.0)

    def fit(hint: F64) -> PlaneFit:
        return fit_plane_ransac(pts, inlier_threshold_m=0.004, iterations=100, seed=7, up_hint=hint)

    a, b = fit(up), fit(up)
    assert a.plane == b.plane
    flipped = fit(-up)
    assert np.allclose(flipped.plane.normal, -np.asarray(a.plane.normal))


def test_world_frame_round_trip(tmp_path: Path) -> None:
    """Tags lying on a tilted, scaled table land on z = 0 with correct board x/y in world."""
    layout = _layout(tmp_path)
    rot, t, s = _rot((1, 2, 0.5), 55), np.array([2.0, -1.0, 4.0]), 5.0
    detected = {i: c @ rot.T * s + t for i, c in layout.tag_corners_m.items()}
    fit = fit_board(layout, detected)

    # Table points: the board plane extended, in recon units, then scaled to meters.
    rng = np.random.default_rng(4)
    xy = np.column_stack([rng.uniform(-0.2, 0.8, (1500, 2)), np.zeros(1500)])
    table_scaled = apply_scale(xy @ rot.T * s + t, fit.scale)
    up = rot @ np.array([0.0, 0.0, 1.0])
    plane = fit_plane_ransac(
        table_scaled, inlier_threshold_m=0.004, iterations=200, seed=0, up_hint=up
    ).plane

    T = world_from_recon(fit, plane)
    assert np.allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3), atol=1e-9)  # rigid, no scale
    assert np.isclose(np.linalg.det(T[:3, :3]), 1.0)

    def to_world(p: F64) -> F64:
        return np.asarray((T[:3, :3] @ apply_scale(p, fit.scale).T).T + T[:3, 3], dtype=np.float64)

    for i, board in layout.tag_corners_m.items():
        w = to_world(detected[i])
        assert np.allclose(w, board, atol=1e-6), i
    assert np.allclose(to_world(table_scaled / fit.scale)[:, 2], 0.0, atol=1e-6)
