from __future__ import annotations

import numpy as np
import pytest

from video2sim.models.hybrid import HybridShape, cube_rotations, frame_rotation
from video2sim.models.shape_model import ShapeSample, ShapeView
from video2sim.schemas import SE3

# a box with a notch and a spike, so no turn of it looks like another
V = np.array(
    [
        [0, 0, 0], [4, 0, 0], [4, 2, 0], [0, 2, 0],
        [0, 0, 1], [4, 0, 1], [4, 2, 1], [0, 2, 1],
        [1, 1, 3],
    ],
    dtype=float,
)  # fmt: skip
F = np.array(
    [
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
        [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
        [4, 6, 8], [6, 7, 8], [7, 4, 8], [4, 5, 8], [5, 6, 8],
    ]
)  # fmt: skip


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    cos = (np.trace(a.T @ b) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def test_there_are_24_proper_cube_rotations() -> None:
    rotations = cube_rotations()
    assert len(rotations) == 24
    for r in rotations:
        np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(r) == pytest.approx(1.0)


def test_frame_rotation_recovers_a_cube_turn_with_yaw_and_a_different_scale() -> None:
    rng = np.random.default_rng(3)
    for k in (0, 5, 11, 17):
        cube = cube_rotations()[k]
        c, s = np.cos(np.deg2rad(30)), np.sin(np.deg2rad(30))
        yaw = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        truth = yaw @ cube  # x_sam = truth @ x_trellis
        trellis = (V - V.mean(0)) / 7.0 + rng.normal(0, 1e-4, V.shape)
        sam = 0.12 * (trellis @ truth.T)
        got, cost = frame_rotation(sam, F, trellis, F)
        assert _angle_deg(got, truth) < 3.0
        assert cost < 0.05


def test_hybrid_returns_trellis_geometry_with_the_sam3d_pose_carried_across() -> None:
    cube = cube_rotations()[7]
    trellis_v = (V - V.mean(0)) / 7.0
    sam_v = 0.12 * (trellis_v @ cube.T)  # x_sam = cube @ x_trellis
    T_sam = np.eye(4)
    T_sam[:3, :3] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    T_sam[:3, 3] = [0.1, 0.2, 0.9]

    def make(model: str, vertices: np.ndarray, T: np.ndarray, checkpoint: str) -> ShapeSample:
        return ShapeSample("o", 4, 1, vertices, F, SE3.from_numpy(T), 1.0, model, checkpoint)

    class Fixed:
        def __init__(self, sample: ShapeSample) -> None:
            self.name = sample.model
            self._sample = sample

        def sample(self, object_id: str, view: ShapeView, seed: int) -> ShapeSample:
            return self._sample

    pose = make("sam3d", sam_v, T_sam, "sam@1")
    mesh = make("trellis2", trellis_v, np.eye(4), "tre@2")
    out = HybridShape(Fixed(pose), Fixed(mesh)).sample("o", None, 1)  # type: ignore[arg-type]

    assert out.model == "trellis2"
    assert out.checkpoint == "tre@2; pose sam@1"
    np.testing.assert_array_equal(out.vertices_model, trellis_v)
    # a TRELLIS.2 point lands where SAM 3D's own copy of it would
    got = out.T_camera_from_model.to_numpy()
    want = T_sam[:3, :3] @ cube
    assert _angle_deg(got[:3, :3], want) < 3.0
    np.testing.assert_allclose(got[:3, 3], T_sam[:3, 3])
