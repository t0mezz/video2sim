from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from video2sim.stages.marker_detection import TagView
from video2sim.stages.reconstruction import (
    ReconstructionLimits,
    SfmImage,
    SfmModel,
    SfmOptions,
    WorldFrameConfig,
    build_world_model,
    camera_coverage,
    check_reconstruction,
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


def _scene(
    unit: float, seed: int = 0, n_table: int = 2000, n_tops: int = 0, tag_7_z: float = 0.0
) -> tuple[list[TagView], F64]:
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
    corners_m[7] = corners_m[7] + [0, 0, tag_7_z]
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
    table = np.column_stack([rng.uniform(-0.2, 1.0, (n_table, 2)), rng.normal(0, 0.001, n_table)])
    tops = np.column_stack([rng.uniform(0, 0.8, (n_tops, 2)), rng.normal(0.05, 0.001, n_tops)])
    clutter = np.column_stack([rng.uniform(0, 0.8, (300, 2)), rng.uniform(0.02, 0.2, 300)])
    far = rng.uniform(-5, 5, (200, 3))  # the room: must not influence the table fit
    return views, to_recon(np.vstack([table, tops, clutter, far]))


CFG = WorldFrameConfig(
    tag_edge_m=TAG_M,
    max_tag_reprojection_px=3.0,
    max_scale_spread=0.03,
    plane_inlier_threshold_m=0.004,
    plane_ransac_iterations=300,
    plane_radius_m=1.0,
    tags_on_table=True,
    max_tag_plane_distance_m=0.01,
    tag_refit_band_m=0.02,
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


def test_plane_on_object_tops_is_refitted_through_the_tags() -> None:
    views, points = _scene(2.0, n_table=150, n_tops=600)
    res = solve_world_frame(views, points, CFG)
    assert res.tag_plane_distance_m < 0.005
    assert any(f.startswith("table_refit_from_tags") for f in res.flags)
    assert np.allclose(res.tag_centers_world_m[7][2], 0.0, atol=5e-3)


def test_tags_at_different_heights_keep_the_first_plane() -> None:
    views, points = _scene(2.0, tag_7_z=0.2)
    res = solve_world_frame(views, points, CFG)
    assert any(f.startswith("tags_off_table") for f in res.flags)
    assert not any(f.startswith("table_refit") for f in res.flags)
    assert np.allclose(res.tag_centers_world_m[3], 0.0, atol=2e-3)
    assert res.tag_centers_world_m[7][2] == pytest.approx(0.2, abs=5e-3)


def test_tag_check_can_be_disabled_for_tags_off_the_table() -> None:
    views, points = _scene(2.0, n_table=150, n_tops=600)
    res = solve_world_frame(views, points, replace(CFG, tags_on_table=False))
    assert res.tag_plane_distance_m == pytest.approx(0.05, abs=0.01)  # still reported
    assert not any(f.startswith("table_refit") for f in res.flags)


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


def _model_from(
    views: list[TagView], n_input: int, sizes: list[int], reproj: float = 0.5
) -> SfmModel:
    images = [
        SfmImage(f"{i}.png", v.K, v.distortion, v.T_world_from_camera) for i, v in enumerate(views)
    ]
    return SfmModel(images, np.zeros((0, 3)), reproj, n_input, sizes)


LIMITS = ReconstructionLimits(
    registered_fraction_min=0.95, mean_reprojection_error_max_px=1.0, min_azimuth_coverage_deg=60
)


def test_good_reconstruction_passes_the_gate() -> None:
    views, points = _scene(3.0)
    world = solve_world_frame(views, points, CFG)
    model = _model_from(views, len(views), [len(views)])
    cov = camera_coverage(model, world)
    assert cov.azimuth_deg > 60
    assert check_reconstruction(model, world, LIMITS) == []


def test_low_registration_message_names_fragmentation() -> None:
    views, points = _scene(3.0)
    world = solve_world_frame(views, points, CFG)
    model = _model_from(views, 20, [3, len(views)])
    reasons = check_reconstruction(model, world, LIMITS)
    assert len(reasons) == 1 and "20%" in reasons[0] and "2 disconnected pieces" in reasons[0]


def test_reprojection_and_missing_marker_messages() -> None:
    views, _ = _scene(3.0)
    model = _model_from(views, len(views), [len(views)], reproj=2.4)
    reasons = check_reconstruction(model, None, LIMITS)
    assert any("2.40 px" in r for r in reasons)
    assert any("marker could not be located" in r for r in reasons)


def test_narrow_camera_path_is_reported() -> None:
    views, points = _scene(3.0)
    world = solve_world_frame(views, points, CFG)
    model = _model_from(views, len(views), [len(views)])
    strict = ReconstructionLimits(0.95, 1.0, min_azimuth_coverage_deg=359)
    reasons = check_reconstruction(model, world, strict)
    assert len(reasons) == 1 and "degrees around the table" in reasons[0]


def test_world_model_is_metric_and_gravity_aligned(tmp_path: Path) -> None:
    unit = 3.0
    views, points = _scene(unit)
    world = solve_world_frame(views, points, CFG)
    model = _model_from(views, len(views), [len(views)])
    model = SfmModel(model.images, points, 0.4, len(views), [len(views)])
    paths = [f"capture/frames/{i}.png" for i in range(len(views))][::-1]  # not in sfm order
    wm = build_world_model(
        model, world, [p for p in paths], (1920, 1080), tmp_path, "reconstruction/points.ply"
    )

    assert wm.scale_source == "marker" and wm.scale_factor == pytest.approx(1 / unit, rel=0.01)
    assert wm.registered_fraction == 1.0 and wm.mean_reprojection_error_px == 0.4
    assert wm.intrinsics.fx == 1500.0 and wm.intrinsics.width_px == 1920
    assert wm.table_plane.normal == (0.0, 0.0, 1.0) and wm.table_plane.offset_m == 0.0
    assert [f.frame_index for f in wm.frames] == sorted(f.frame_index for f in wm.frames)

    # Cameras sit above the table, at the heights the scene used (0.7 to 1.0 m), in meters.
    heights = sorted(f.T_world_from_camera.to_numpy()[2, 3] for f in wm.frames)
    assert heights[0] > 0.65 and heights[-1] < 1.05

    ply = (tmp_path / "points.ply").read_bytes()
    body = ply.split(b"end_header\n", 1)[1]
    xyz = np.frombuffer(body, dtype="<f4").reshape(-1, 3)
    assert len(xyz) == len(points)
    near_table = xyz[np.abs(xyz[:, 2]) < 0.02]
    assert len(near_table) > 1500  # the table points land on z = 0


def test_world_model_rejects_a_frame_capture_does_not_know(tmp_path: Path) -> None:
    views, points = _scene(3.0)
    world = solve_world_frame(views, points, CFG)
    model = _model_from(views, len(views), [len(views)])
    with pytest.raises(ValueError, match="not one of the capture frames"):
        build_world_model(model, world, ["capture/frames/other.png"], (10, 10), tmp_path, "r/p.ply")


def test_sfm_options_from_config() -> None:
    assert SfmOptions.from_config(None) == SfmOptions()
    assert SfmOptions.from_config({"matcher": "sequential", "loop_stride": 8}).loop_stride == 8
    with pytest.raises(ValueError, match="unknown"):
        SfmOptions.from_config({"matchr": "sequential"})
    with pytest.raises(ValueError, match="matcher"):
        SfmOptions.from_config({"matcher": "vocab"})
    with pytest.raises(ValueError, match="mapper"):
        SfmOptions.from_config({"mapper": "glomap"})
