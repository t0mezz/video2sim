from __future__ import annotations

import io
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import trimesh
import yaml
from numpy.typing import NDArray

from video2sim.models.sam3d import (
    GLB_FROM_MODEL,
    Sam3d,
    Sam3dAnswer,
    Sam3dCall,
    Sam3dConfig,
    camera_from_model,
    model_mesh_from_glb,
    quaternion_wxyz_to_matrix,
)
from video2sim.models.shape_model import ShapeView, draw_samples, sample_seed

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
P3D_FROM_CV = np.diag([-1.0, -1.0, 1.0])


def _quat_about(axis: tuple[float, float, float], angle: float) -> list[float]:
    a = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    return [float(np.cos(angle / 2)), *(np.sin(angle / 2) * a).tolist()]


def _glb(vertices_model: NDArray[np.float64], faces: NDArray[np.int64]) -> bytes:
    """A GLB as SAM 3D's ``to_glb`` writes it: model vertices turned from z-up to y-up."""
    mesh = trimesh.Trimesh(vertices_model @ GLB_FROM_MODEL.T, faces, process=False)
    data = mesh.export(file_type="glb")
    assert isinstance(data, bytes)
    return data


TETRA_V = np.array([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.0, 0.3, 0.0], [0.0, 0.0, 0.2]])
TETRA_F = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])


def _config(**overrides: object) -> Sam3dConfig:
    base: dict[str, object] = {
        "checkpoint": "facebook/sam-3d-objects",
        "stage1_inference_steps": None,
        "stage2_inference_steps": None,
        "use_pointmap": False,
    }
    base.update(overrides)
    return Sam3dConfig.from_config({"shape": {"sam3d": base}})


class FakeTransport:
    """Answers every call with the same pose and mesh, and records what was sent."""

    def __init__(self, pose: dict[str, object] | None = None) -> None:
        self.calls: list[Sam3dCall] = []
        self.pose = pose or {
            "rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
            "translation": [0.0, 0.0, 2.0],
            "scale": [0.5, 0.5, 0.5],
            "checkpoint": "facebook/sam-3d-objects@abc123",
        }

    def __call__(self, call: Sam3dCall) -> Sam3dAnswer:
        self.calls.append(call)
        return Sam3dAnswer(glb=_glb(TETRA_V, TETRA_F), pose_json=json.dumps(self.pose))


def _image(tmp_path: Path, height: int = 48, width: int = 64) -> Path:
    path = tmp_path / "frame_000010.jpg"
    rng = np.random.default_rng(0)
    cv2.imwrite(str(path), rng.integers(0, 255, (height, width, 3), dtype=np.uint8))
    return path


def _mask(height: int = 48, width: int = 64) -> NDArray[np.bool_]:
    mask = np.zeros((height, width), dtype=bool)
    mask[10:30, 20:40] = True
    return mask


def test_quaternion_matches_closed_form_rotation_about_z() -> None:
    np.testing.assert_allclose(quaternion_wxyz_to_matrix([1.0, 0.0, 0.0, 0.0]), np.eye(3))
    r = quaternion_wxyz_to_matrix(_quat_about((0, 0, 1), np.pi / 2))
    np.testing.assert_allclose(r @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], atol=1e-12)
    # an unnormalised quaternion describes the same rotation
    np.testing.assert_allclose(
        quaternion_wxyz_to_matrix([2.0, 0.0, 0.0, 0.0]), np.eye(3), atol=1e-12
    )
    with pytest.raises(ValueError):
        quaternion_wxyz_to_matrix([0.0, 0.0, 0.0, 0.0])


def test_camera_from_model_flips_pytorch3d_axes_to_opencv() -> None:
    t, s = camera_from_model([1.0, 0.0, 0.0, 0.0], [0.1, 0.2, 2.0], [0.5, 0.5, 0.5])
    assert s == pytest.approx(0.5)
    a = t.to_numpy()
    point_cv = a[:3, :3] @ (s * np.array([1.0, 0.0, 0.0])) + a[:3, 3]
    # PyTorch3D (0.5 + 0.1, 0.2, 2.0) is x left, y up; OpenCV is x right, y down
    np.testing.assert_allclose(point_cv, [-0.6, -0.2, 2.0], atol=1e-12)


def test_camera_from_model_reproduces_sam3d_make_scene() -> None:
    """Same result as SAM 3D's ``make_scene``: row vectors, ``(p * s) @ R + t`` in PyTorch3D
    camera axes (``compose_transform`` with ``quaternion_to_matrix``), then axes to OpenCV."""
    q = _quat_about((0.3, -1.0, 0.5), 0.9)
    t_p3d, scale = np.array([0.05, -0.02, 1.4]), 0.27
    points = np.random.default_rng(1).normal(size=(20, 3))
    p3d = (points * scale) @ quaternion_wxyz_to_matrix(q) + t_p3d
    expected_cv = p3d @ P3D_FROM_CV.T

    t, s = camera_from_model(q, t_p3d.tolist(), [scale] * 3)
    a = t.to_numpy()
    got = (s * points) @ a[:3, :3].T + a[:3, 3]
    np.testing.assert_allclose(got, expected_cv, atol=1e-12)


def test_camera_from_model_rejects_anisotropic_scale() -> None:
    with pytest.raises(ValueError, match="scale"):
        camera_from_model([1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.5, 0.5, 0.6])


def test_glb_is_turned_back_into_the_model_frame() -> None:
    vertices, faces = model_mesh_from_glb(_glb(TETRA_V, TETRA_F))
    np.testing.assert_allclose(vertices, TETRA_V, atol=1e-6)
    np.testing.assert_array_equal(faces, TETRA_F)


def test_sample_decodes_answer_and_records_checkpoint_and_seed(tmp_path: Path) -> None:
    transport = FakeTransport()
    model = Sam3d(_config(), transport, cache_dir=tmp_path / "cache")
    view = ShapeView(frame_index=10, image_path=_image(tmp_path), mask=_mask())
    sample = model.sample("obj_001", view, seed=7)

    assert (sample.object_id, sample.frame_index, sample.seed) == ("obj_001", 10, 7)
    assert sample.model == "sam3d"
    assert sample.checkpoint == "facebook/sam-3d-objects@abc123"
    assert sample.scale_guess == pytest.approx(0.5)
    np.testing.assert_allclose(sample.vertices_model, TETRA_V, atol=1e-6)
    np.testing.assert_allclose(sample.T_camera_from_model.to_numpy()[:3, 3], [0.0, 0.0, 2.0])

    call = transport.calls[0]
    assert call.seed == 7
    assert call.pointmap_npy is None
    mask_back = cv2.imdecode(np.frombuffer(call.mask_png, np.uint8), cv2.IMREAD_GRAYSCALE)
    assert mask_back is not None
    np.testing.assert_array_equal(mask_back > 0, _mask())


def test_answers_are_cached_by_what_was_sent(tmp_path: Path) -> None:
    transport = FakeTransport()
    view = ShapeView(frame_index=10, image_path=_image(tmp_path), mask=_mask())
    first = Sam3d(_config(), transport, cache_dir=tmp_path / "cache").sample("obj_001", view, 7)
    again = Sam3d(_config(), transport, cache_dir=tmp_path / "cache").sample("obj_001", view, 7)
    assert len(transport.calls) == 1
    np.testing.assert_array_equal(first.vertices_model, again.vertices_model)

    Sam3d(_config(), transport, cache_dir=tmp_path / "cache").sample("obj_001", view, 8)
    Sam3d(_config(stage1_inference_steps=4), transport, cache_dir=tmp_path / "cache").sample(
        "obj_001", view, 7
    )
    assert len(transport.calls) == 3


def test_mask_must_match_the_image_and_cover_pixels(tmp_path: Path) -> None:
    model = Sam3d(_config(), FakeTransport(), cache_dir=tmp_path / "cache")
    image = _image(tmp_path)
    with pytest.raises(ValueError, match="shape"):
        model.sample("o", ShapeView(0, image, _mask(height=40)), seed=0)
    with pytest.raises(ValueError, match="empty"):
        model.sample("o", ShapeView(0, image, np.zeros((48, 64), dtype=bool)), seed=0)


def test_pointmap_is_sent_in_pytorch3d_axes_only_when_enabled(tmp_path: Path) -> None:
    pointmap = np.zeros((48, 64, 3))
    pointmap[..., 0], pointmap[..., 1], pointmap[..., 2] = 0.1, 0.2, 1.5
    view = ShapeView(0, _image(tmp_path), _mask(), pointmap=pointmap)

    off = FakeTransport()
    Sam3d(_config(), off, cache_dir=tmp_path / "c1").sample("o", view, seed=0)
    assert off.calls[0].pointmap_npy is None

    on = FakeTransport()
    Sam3d(_config(use_pointmap=True), on, cache_dir=tmp_path / "c2").sample("o", view, seed=0)
    payload = on.calls[0].pointmap_npy
    assert payload is not None
    sent = np.load(io.BytesIO(payload))
    assert sent.dtype == np.float32 and sent.shape == (48, 64, 3)
    np.testing.assert_allclose(sent[0, 0], [-0.1, -0.2, 1.5], rtol=1e-6)


def test_draw_samples_covers_every_keyframe_with_distinct_reproducible_seeds(
    tmp_path: Path,
) -> None:
    transport = FakeTransport()
    model = Sam3d(_config(), transport, cache_dir=tmp_path / "cache")
    image = _image(tmp_path)
    views = [ShapeView(12, image, _mask()), ShapeView(3, image, _mask())]
    samples = draw_samples(model, "obj_002", views, samples_per_keyframe=3, base_seed=5)

    assert [s.frame_index for s in samples] == [12, 12, 12, 3, 3, 3]
    seeds = [s.seed for s in samples]
    assert len(set(seeds)) == 6
    assert seeds[0] == sample_seed(5, "obj_002", 12, 0)
    assert sample_seed(5, "obj_002", 12, 0) != sample_seed(5, "obj_003", 12, 0)
    with pytest.raises(ValueError):
        draw_samples(model, "obj_002", views, samples_per_keyframe=0, base_seed=5)


def test_default_config_has_a_valid_sam3d_section() -> None:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    sam3d = Sam3dConfig.from_config(cfg)
    assert sam3d.checkpoint
    assert int(cfg["shape"]["samples_per_keyframe"]) >= 1
