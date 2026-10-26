from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from video2sim.stages.marker_detection import TagView
from video2sim.stages.reconstruction import (
    SfmImage,
    SfmModel,
    WorldFrameConfig,
    solve_world_frame,
)

F64 = NDArray[np.float64]

K = np.array([[1500.0, 0, 960], [0, 1500.0, 540], [0, 0, 1]])
TAG_M = 0.1
# Tag lower-left corners on a table, in metres; two arbitrary tags, nothing about their spacing
# is used by the code under test.
TAG_ORIGINS = {3: (0.10, 0.20), 7: (0.65, 0.35)}


def _rot(axis: tuple[float, float, float], deg: float) -> F64:
    k = np.asarray(axis, dtype=np.float64)
    k /= np.linalg.norm(k)
    a = np.radians(deg)
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.asarray(np.eye(3) + np.sin(a) * kx + (1 - np.cos(a)) * kx @ kx, dtype=np.float64)


def _look_at(eye: F64, target: F64) -> F64:
    z = (target - eye) / np.linalg.norm(target - eye)
    x = np.cross(z, np.array([0.0, 0.0, 1.0]))
    x /= np.linalg.norm(x)
    t = np.eye(4)
    t[:3, :3] = np.column_stack([x, np.cross(z, x), z])
    t[:3, 3] = eye
    return t


def _scene(unit: float, seed: int = 0) -> tuple[list[TagView], F64]:
    """A table scene seen in a reconstruction frame that is rotated, shifted and ``unit`` x too big.

    Returns the views (ideal tag detections plus 0.2 px noise) and sparse points (table plus
    clutter). Metric truth: tags lie on z = 0 with lower-left corners at ``TAG_ORIGINS``.
    """
    rng = np.random.default_rng(seed)
    rot, shift = _rot((1, 2, 0.5), 50), np.array([2.0, -1.0, 4.0])

    def to_recon(p: F64) -> F64:
        return np.asarray((p @ rot.T) * unit + shift, dtype=np.float64)

    corners_m = {
        i: np.array([[x, y, 0], [x + TAG_M, y, 0], [x + TAG_M, y + TAG_M, 0], [x, y + TAG_M, 0.0]])
        for i, (x, y) in TAG_ORIGINS.items()
    }
    views = []
    for eye in ([0.3, -0.5, 0.8], [0.9, 0.7, 0.7], [-0.1, 0.5, 0.9], [0.6, 0.1, 1.0]):
        t_wc = _look_at(to_recon(np.array(eye)), to_recon(np.array([0.35, 0.3, 0.0])))
        t_cw = np.linalg.inv(t_wc)
        det: dict[int, F64] = {}
        for i, c in corners_m.items():
            cam = (t_cw[:3, :3] @ to_recon(c).T).T + t_cw[:3, 3]
            px = (K @ cam.T).T
            det[i] = px[:, :2] / px[:, 2:] + rng.normal(scale=0.2, size=(4, 2))
        views.append(TagView(K, np.zeros(0), t_wc, det))
    table = np.column_stack([rng.uniform(-0.2, 1.0, (2000, 2)), rng.normal(0, 0.001, 2000)])
    clutter = np.column_stack([rng.uniform(0, 0.8, (300, 2)), rng.uniform(0.02, 0.2, 300)])
    far = rng.uniform(-5, 5, (200, 3))  # the room: must not influence the table fit
    return views, to_recon(np.vstack([table, clutter, far]))


CFG = WorldFrameConfig(
    tag_edge_m=TAG_M,
    max_tag_reprojection_px=3.0,
    max_scale_spread=0.03,
    plane_inlier_threshold_m=0.004,
    plane_ransac_iterations=300,
    plane_radius_m=1.0,
    seed=0,
)


def test_scale_and_world_frame_from_two_arbitrary_tags() -> None:
    unit = 3.7
    views, points = _scene(unit)
    res = solve_world_frame(views, points, CFG)

    assert res.scale.scale == pytest.approx(1 / unit, rel=0.01)
    assert not res.flags
    assert res.reference_tag == 3  # both seen in 4 frames: the lowest id wins
    # Reference tag centre is the origin; the other tag sits where the layout says, on z = 0.
    expected = np.array([0.65 - 0.10, 0.35 - 0.20, 0.0])
    assert np.allclose(res.tag_centers_world_m[3], 0.0, atol=2e-3)
    assert np.allclose(res.tag_centers_world_m[7], expected, atol=5e-3)
    t = res.T_world_from_scaled_recon
    assert np.allclose(t[:3, :3] @ t[:3, :3].T, np.eye(3), atol=1e-9)
    assert res.table.inlier_fraction > 0.6


def test_single_tag_is_flagged_but_still_works() -> None:
    views, points = _scene(2.0)
    only = [TagView(v.K, v.distortion, v.T_world_from_camera, {3: v.corners_px[3]}) for v in views]
    res = solve_world_frame(only, points, CFG)
    assert res.scale.scale == pytest.approx(0.5, rel=0.02)
    assert any(f.startswith("single_tag") for f in res.flags)


def test_no_tag_is_a_user_facing_error() -> None:
    views, points = _scene(2.0)
    blind = [TagView(v.K, v.distortion, v.T_world_from_camera, {}) for v in views]
    with pytest.raises(ValueError, match="marker not seen"):
        solve_world_frame(blind, points, CFG)


def test_points_far_from_the_marker_are_an_error() -> None:
    views, _ = _scene(2.0)
    with pytest.raises(ValueError, match="within"):
        solve_world_frame(views, np.full((10, 3), 100.0), CFG)


def test_sfm_model_reports_fragmentation_and_registered_fraction() -> None:
    img = SfmImage("0.png", K, np.zeros(5), np.eye(4))
    whole = SfmModel([img] * 95, np.zeros((0, 3)), 0.5, 100, [95])
    split = SfmModel([img] * 95, np.zeros((0, 3)), 0.5, 100, [3, 95])
    assert whole.registered_fraction == 0.95 and not whole.fragmented
    assert split.fragmented
