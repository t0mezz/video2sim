from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import trimesh
import yaml
from numpy.typing import NDArray

from video2sim.schemas import SE3, Intrinsics
from video2sim.stages.registration import (
    MaskView,
    Registration,
    RegistrationConfig,
    _build_problem,
    level,
    refine_stretch,
    register,
    rotation_from_vector,
)

F64 = NDArray[np.float64]
CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
K = Intrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width_px=640, height_px=480)

# A box in the model's own units and frame, 10x larger than the real object, off-centre.
MODEL_EXTENTS = np.array([0.8, 0.5, 1.2])
MODEL = trimesh.creation.box(extents=MODEL_EXTENTS)
MODEL.apply_translation([0.3, -0.2, 0.1])
TRUE_SCALE = 0.1
TRUE_R = rotation_from_vector(np.array([0.0, 0.0, np.radians(30.0)])) @ rotation_from_vector(
    np.array([np.radians(90.0), 0.0, 0.0])
)
TRUE_T = np.array([0.1, -0.05, 0.0])


def _place(points: F64, scale: float, r: F64, t: F64) -> F64:
    out: F64 = (scale * points) @ r.T + t
    return out


def _true_vertices() -> F64:
    return _place(np.asarray(MODEL.vertices), TRUE_SCALE, TRUE_R, TRUE_T)


def _true_surface(count: int, seed: int = 0) -> F64:
    points, _ = trimesh.sample.sample_surface(MODEL, count, seed=seed)
    return _place(np.asarray(points), TRUE_SCALE, TRUE_R, TRUE_T)


def _look_at(eye: F64, target: F64) -> SE3:
    z = target - eye
    z /= np.linalg.norm(z)
    down = np.array([0.0, 0.0, -1.0])
    y = down - z * float(down @ z)
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    m = np.eye(4)
    m[:3, :3] = np.column_stack([x, y, z])
    m[:3, 3] = eye
    return SE3.from_numpy(m)


def _silhouette(vertices: F64, cam: SE3) -> NDArray[np.bool_]:
    """Exact mask of a convex mesh: the convex hull of its projected corners."""
    a = cam.inverse().to_numpy()
    c = vertices @ a[:3, :3].T + a[:3, 3]
    uv = np.column_stack([K.fx * c[:, 0] / c[:, 2] + K.cx, K.fy * c[:, 1] / c[:, 2] + K.cy])
    hull = cv2.convexHull(np.round(uv).astype(np.int32))
    mask = np.zeros((K.height_px, K.width_px), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 1)
    return mask.astype(bool)


def _cameras(count: int = 6) -> list[SE3]:
    center = _true_vertices().mean(axis=0)
    return [
        _look_at(center + np.array([0.45 * np.cos(a), 0.45 * np.sin(a), 0.3]), center)
        for a in np.linspace(-1.2, 1.2, count)  # a handheld arc, not a full orbit
    ]


def _views(occlude_first: bool = False) -> list[MaskView]:
    views = []
    for i, cam in enumerate(_cameras()):
        mask = _silhouette(_true_vertices(), cam)
        occluders = None
        if occlude_first and i == 0:
            cols = np.nonzero(mask.any(axis=0))[0]
            cut = (int(np.min(cols)) + int(np.max(cols))) // 2
            occluders = np.zeros_like(mask)
            occluders[:, :cut] = mask[:, :cut]
            mask = mask.copy()
            mask[:, :cut] = False
        views.append(MaskView(cam, K, mask, occluders))
    return views


def _config(**overrides: object) -> RegistrationConfig:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    cfg["shape"]["registration"].update(overrides)
    return RegistrationConfig.from_config(cfg)


def _corner_error_m(result_vertices: F64) -> float:
    """Largest distance from a true corner to the nearest registered corner; a box's own
    symmetries give the same corner set, so they do not count as errors."""
    truth = _true_vertices()
    d = np.linalg.norm(truth[:, None, :] - result_vertices[None, :, :], axis=2)
    return float(np.max(np.min(d, axis=1)))


def _registered_vertices(t: SE3, scale: float) -> F64:
    a = t.to_numpy()
    return _place(np.asarray(MODEL.vertices), scale, a[:3, :3], a[:3, 3])


def _perturbed_start(angle_deg: float) -> F64:
    out: F64 = rotation_from_vector(np.radians(angle_deg) * np.array([0.6, -0.3, 0.74])) @ TRUE_R
    return out


def test_rotation_from_vector_is_a_proper_rotation_about_the_axis() -> None:
    r = rotation_from_vector(np.array([0.0, 0.0, np.pi / 2]))
    np.testing.assert_allclose(r @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(rotation_from_vector(np.zeros(3)), np.eye(3))


def test_recovers_pose_and_scale_from_a_perturbed_start() -> None:
    result = register(
        np.asarray(MODEL.vertices), np.asarray(MODEL.faces), _perturbed_start(12.0),
        _true_surface(1500), _views(), _config(), seed=0,
    )  # fmt: skip
    assert result.scale == pytest.approx(TRUE_SCALE, rel=0.01)
    assert _corner_error_m(_registered_vertices(result.T_world_from_model, result.scale)) < 0.003
    assert result.fit_score > 0.9


def test_stray_points_do_not_pull_the_fit_off() -> None:
    points = _true_surface(1500)
    rng = np.random.default_rng(3)
    strays = points[:75] + rng.normal(0.0, 1.0, (75, 3)) * 0.02
    result = register(
        np.asarray(MODEL.vertices), np.asarray(MODEL.faces), _perturbed_start(8.0),
        np.vstack([points, strays]), _views(), _config(), seed=0,
    )  # fmt: skip
    assert result.scale == pytest.approx(TRUE_SCALE, rel=0.02)
    assert _corner_error_m(_registered_vertices(result.T_world_from_model, result.scale)) < 0.004


def test_yaw_hypotheses_recover_a_start_that_is_a_quarter_turn_off() -> None:
    start = rotation_from_vector(np.array([0.0, 0.0, np.pi / 2])) @ TRUE_R
    result = register(
        np.asarray(MODEL.vertices), np.asarray(MODEL.faces), start,
        _true_surface(1500), _views(), _config(yaw_hypotheses=4), seed=0,
    )  # fmt: skip
    assert _corner_error_m(_registered_vertices(result.T_world_from_model, result.scale)) < 0.003


def test_masks_fix_the_scale_when_only_the_top_is_observed() -> None:
    surface = _true_surface(4000)
    top = surface[surface[:, 2] > float(np.max(surface[:, 2])) - 1e-6]
    assert 100 < len(top) < 2000
    result = register(
        np.asarray(MODEL.vertices), np.asarray(MODEL.faces), _perturbed_start(5.0),
        top, _views(), _config(), seed=0,
    )  # fmt: skip
    assert result.scale == pytest.approx(TRUE_SCALE, rel=0.015)
    # without the masks nothing holds the unseen sides, and the scale comes out short
    points_only = register(
        np.asarray(MODEL.vertices), np.asarray(MODEL.faces), _perturbed_start(5.0),
        top, _views(), _config(mask_weight=0.0), seed=0,
    )  # fmt: skip
    assert abs(points_only.scale / TRUE_SCALE - 1) > 0.02


def test_level_stands_the_nearest_model_axis_straight_up() -> None:
    r, up_model = level(_perturbed_start(12.0))
    np.testing.assert_array_equal(up_model, [0.0, 1.0, 0.0])  # TRUE_R turns model +y up
    np.testing.assert_allclose(r @ up_model, [0.0, 0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-12)
    # least turn: a start that is level already stays as it is
    level_start = rotation_from_vector(np.array([0.0, 0.0, 0.4])) @ TRUE_R
    np.testing.assert_allclose(level(level_start)[0], level_start, atol=1e-12)


def test_registered_model_axis_stays_vertical() -> None:
    result = register(
        np.asarray(MODEL.vertices), np.asarray(MODEL.faces), _perturbed_start(12.0),
        _true_surface(1500), _views(), _config(), seed=0,
    )  # fmt: skip
    up = result.T_world_from_model.to_numpy()[:3, :3] @ [0.0, 1.0, 0.0]
    assert np.degrees(np.arccos(up[2])) < 0.5


def test_contact_fixes_the_scale_when_only_the_top_is_observed_and_masks_are_off() -> None:
    surface = _true_surface(4000)
    top = surface[surface[:, 2] > float(np.max(surface[:, 2])) - 1e-6]
    result = register(
        np.asarray(MODEL.vertices), np.asarray(MODEL.faces), _perturbed_start(5.0),
        top, _views(), _config(mask_weight=0.0), seed=0,
        support_z=float(np.min(_true_vertices()[:, 2])),
    )  # fmt: skip
    assert result.scale == pytest.approx(TRUE_SCALE, rel=0.01)


def _stretched_model(factors: F64) -> trimesh.Trimesh:
    return trimesh.Trimesh(np.asarray(MODEL.vertices) * factors, MODEL.faces, process=False)


def _placed_extents(result: Registration, mesh: trimesh.Trimesh) -> F64:
    out: F64 = np.ptp(result.scale * np.asarray(result.stretch) * np.asarray(mesh.vertices), axis=0)
    return out


def test_stretch_corrects_a_mesh_with_wrong_proportions() -> None:
    wrong = _stretched_model(np.array([1.25, 0.9, 1.0]))
    truth = TRUE_SCALE * MODEL_EXTENTS
    start = TRUE_R @ rotation_from_vector(np.array([0.05, -0.04, 0.06]))
    target = _true_surface(1500)
    off = register(
        wrong.vertices, wrong.faces, start, target, _views(), _config(stretch_sigma=0), 3
    )
    on = refine_stretch(wrong.vertices, wrong.faces, off, target, _views(), _config(), 3)
    assert off.stretch == (1.0, 1.0, 1.0)
    error_off = np.max(np.abs(_placed_extents(off, wrong) / truth - 1))
    error_on = np.max(np.abs(_placed_extents(on, wrong) / truth - 1))
    assert error_off > 0.1
    assert error_on < 0.04
    placed = on.scale * np.asarray(on.stretch) * np.asarray(wrong.vertices)
    world = placed @ on.T_world_from_model.to_numpy()[:3, :3].T
    world = world + on.T_world_from_model.to_numpy()[:3, 3]
    assert _corner_error_m(world) < 0.005


def test_stretch_stays_at_one_when_the_proportions_are_right() -> None:
    start = TRUE_R @ rotation_from_vector(np.array([0.05, -0.04, 0.06]))
    fit = register(MODEL.vertices, MODEL.faces, start, _true_surface(1500), _views(), _config(), 3)
    result = refine_stretch(
        MODEL.vertices, MODEL.faces, fit, _true_surface(1500), _views(), _config(), 3
    )
    # only the product with the scale is meaningful, so compare the sizes
    np.testing.assert_allclose(
        _placed_extents(result, MODEL), TRUE_SCALE * MODEL_EXTENTS, rtol=0.015
    )


def test_jacobian_matches_central_differences() -> None:
    distorted = K.model_copy(update={"distortion": [0.05, -0.02, 0.001, -0.002, 0.01]})
    views = [MaskView(v.T_world_from_camera, distorted, v.mask, v.occluders) for v in _views()]
    up = level(TRUE_R)[1]
    problem, _ = _build_problem(
        np.asarray(MODEL.vertices), np.asarray(MODEL.faces), _true_surface(500), views,
        _config(), 0, up, 0.0,
    )  # fmt: skip
    base = (TRUE_SCALE, TRUE_R, TRUE_T + np.array([0.002, -0.001, 0.003]))
    p = np.array([0.02, 0.03, -0.02, 0.05, 0.001, 0.002, -0.001, 0.03, -0.02, 0.01])
    index, _, weight = problem.match(p, base)
    exact = problem.jacobian(p, base, index, weight)
    step = 1e-7
    numeric = np.empty_like(exact)
    for i in range(len(p)):
        d = np.zeros(len(p))
        d[i] = step
        high = problem.residuals(p + d, base, index, weight)
        low = problem.residuals(p - d, base, index, weight)
        numeric[:, i] = (high - low) / (2 * step)
    # bilinear pixel costs have kinks at cell borders, so a few rows may differ
    agree = np.abs(exact - numeric).max(axis=1) <= 1e-3 * (1 + np.abs(numeric).max(axis=1))
    assert agree.mean() > 0.995


def test_config_rejects_a_negative_stretch_sigma() -> None:
    with pytest.raises(ValueError, match="stretch_sigma"):
        _config(stretch_sigma=-0.1)


def test_config_rejects_an_upright_sigma_of_ninety_degrees() -> None:
    with pytest.raises(ValueError, match="upright"):
        _config(upright_sigma_deg=90.0)


def test_an_occluded_part_of_the_mask_is_not_charged() -> None:
    result = register(
        np.asarray(MODEL.vertices), np.asarray(MODEL.faces), _perturbed_start(8.0),
        _true_surface(1500), _views(occlude_first=True), _config(), seed=0,
    )  # fmt: skip
    assert _corner_error_m(_registered_vertices(result.T_world_from_model, result.scale)) < 0.003
    assert result.mask_inside_fraction > 0.95


def test_a_wrong_shape_scores_lower_than_the_right_one() -> None:
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=0.6)
    args = (_perturbed_start(5.0), _true_surface(1500), _views(), _config())
    right = register(np.asarray(MODEL.vertices), np.asarray(MODEL.faces), *args, seed=0)
    wrong = register(np.asarray(sphere.vertices), np.asarray(sphere.faces), *args, seed=0)
    assert right.fit_score > wrong.fit_score + 0.2


def test_same_seed_gives_the_same_result() -> None:
    args = (
        np.asarray(MODEL.vertices), np.asarray(MODEL.faces), _perturbed_start(8.0),
        _true_surface(800), _views(), _config(),
    )  # fmt: skip
    a, b = register(*args, seed=4), register(*args, seed=4)
    np.testing.assert_array_equal(a.T_world_from_model.to_numpy(), b.T_world_from_model.to_numpy())
    assert a.scale == b.scale


def test_mask_view_rejects_a_mask_of_the_wrong_size() -> None:
    with pytest.raises(ValueError, match="shape"):
        MaskView(_cameras()[0], K, np.zeros((10, 10), dtype=bool))


def test_register_rejects_empty_inputs() -> None:
    with pytest.raises(ValueError):
        register(
            np.asarray(MODEL.vertices), np.asarray(MODEL.faces), TRUE_R,
            np.zeros((0, 3)), _views(), _config(), seed=0,
        )  # fmt: skip
