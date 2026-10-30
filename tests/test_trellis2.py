from __future__ import annotations

import io
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from video2sim.models.shape_model import SampleFailed, ShapeView, draw_samples
from video2sim.models.trellis2 import (
    CAMERA_FROM_MODEL_GUESS,
    ModalTransport,
    Trellis2,
    Trellis2Answer,
    Trellis2Call,
    Trellis2Config,
    mesh_from_npz,
)
from video2sim.models.trellis2_modal import REPO_ID, DisabledRembg, mesh_npz, vertex_colors
from video2sim.stages.shape import default_model

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
TETRA_V = np.array([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.0, 0.3, 0.0], [0.0, 0.0, 0.2]])
TETRA_F = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])


def _config(**overrides: object) -> Trellis2Config:
    base: dict[str, object] = {
        "checkpoint": REPO_ID,
        "pipeline_type": "1024_cascade",
        "max_faces": 200000,
        "colors": True,
    }
    base.update(overrides)
    return Trellis2Config.from_config({"shape": {"trellis2": base}})


class FakeTransport:
    def __init__(self) -> None:
        self.calls: list[Trellis2Call] = []

    def __call__(self, call: Trellis2Call) -> Trellis2Answer:
        self.calls.append(call)
        return Trellis2Answer(
            mesh_npz(TETRA_V, TETRA_F), json.dumps({"checkpoint": f"{REPO_ID}@abc123"})
        )


def _view(tmp_path: Path, mask_rows: slice = slice(10, 30)) -> ShapeView:
    path = tmp_path / "frame_000010.jpg"
    rng = np.random.default_rng(0)
    cv2.imwrite(str(path), rng.integers(0, 255, (48, 64, 3), dtype=np.uint8))
    mask = np.zeros((48, 64), dtype=bool)
    mask[mask_rows, 20:40] = True
    return ShapeView(frame_index=10, image_path=path, mask=mask)


def test_default_config_loads_and_names_a_known_pipeline() -> None:
    cfg = yaml.safe_load(CONFIG.read_text())
    config = Trellis2Config.from_config(cfg)
    assert config.checkpoint == REPO_ID
    with pytest.raises(ValueError, match="pipeline_type"):
        _config(pipeline_type="2048")
    with pytest.raises(ValueError, match="max_faces"):
        _config(max_faces=1)


def test_start_pose_guess_is_a_proper_rotation_with_model_up_along_camera_up() -> None:
    r = CAMERA_FROM_MODEL_GUESS
    np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(r) == pytest.approx(1.0)
    np.testing.assert_allclose(r @ [0.0, 0.0, 1.0], [0.0, -1.0, 0.0])  # OpenCV -y is up
    np.testing.assert_allclose(r @ [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])  # forward is depth


def test_sample_decodes_answer_and_records_checkpoint_and_seed(tmp_path: Path) -> None:
    transport = FakeTransport()
    model = Trellis2(_config(), transport, cache_dir=tmp_path / "cache")
    sample = model.sample("obj_001", _view(tmp_path), seed=7)

    assert (sample.object_id, sample.frame_index, sample.seed) == ("obj_001", 10, 7)
    assert sample.model == "trellis2"
    assert sample.checkpoint == f"{REPO_ID}@abc123"
    assert sample.scale_guess == 1.0
    np.testing.assert_allclose(sample.vertices_model, TETRA_V, atol=1e-6)
    np.testing.assert_array_equal(sample.faces, TETRA_F)
    np.testing.assert_allclose(
        sample.T_camera_from_model.to_numpy()[:3, :3], CAMERA_FROM_MODEL_GUESS
    )

    call = transport.calls[0]
    assert call.seed == 7
    mask_back = cv2.imdecode(np.frombuffer(call.mask_png, np.uint8), cv2.IMREAD_GRAYSCALE)
    assert mask_back is not None
    np.testing.assert_array_equal(mask_back > 0, _view(tmp_path).mask)


def test_answers_are_cached_by_what_was_sent(tmp_path: Path) -> None:
    transport = FakeTransport()
    view = _view(tmp_path)
    Trellis2(_config(), transport, tmp_path / "cache").sample("o", view, 7)
    Trellis2(_config(), transport, tmp_path / "cache").sample("o", view, 7)
    assert len(transport.calls) == 1
    Trellis2(_config(), transport, tmp_path / "cache").sample("o", view, 8)
    Trellis2(_config(pipeline_type="512"), transport, tmp_path / "cache").sample("o", view, 7)
    assert len(transport.calls) == 3


def test_mask_must_match_the_image_and_cover_pixels(tmp_path: Path) -> None:
    model = Trellis2(_config(), FakeTransport(), tmp_path / "cache")
    view = _view(tmp_path)
    with pytest.raises(ValueError, match="shape"):
        model.sample("o", ShapeView(0, view.image_path, view.mask[:40]), seed=0)
    with pytest.raises(ValueError, match="empty"):
        model.sample("o", ShapeView(0, view.image_path, np.zeros((48, 64), bool)), seed=0)


def test_bad_meshes_are_rejected() -> None:
    def npz(vertices: np.ndarray, faces: np.ndarray) -> bytes:
        buffer = io.BytesIO()
        np.savez(buffer, vertices=vertices, faces=faces)
        return buffer.getvalue()

    with pytest.raises(ValueError, match="vertices"):
        mesh_from_npz(npz(vertices=np.zeros((3, 2)), faces=np.zeros((1, 3), int)))
    with pytest.raises(ValueError, match="index"):
        mesh_from_npz(npz(vertices=TETRA_V, faces=np.array([[0, 1, 9]])))
    with pytest.raises(ValueError, match="finite"):
        mesh_from_npz(npz(vertices=TETRA_V * np.nan, faces=TETRA_F))


def test_modal_transport_passes_the_call_to_the_worker() -> None:
    seen: list[tuple[object, ...]] = []

    class Method:
        def remote(self, *args: object) -> dict[str, object]:
            seen.append(args)
            return {"mesh_npz": mesh_npz(TETRA_V, TETRA_F), "meta_json": "{}"}

    class Worker:
        sample = Method()

    transport = ModalTransport(lookup=lambda: Worker())
    answer = transport(Trellis2Call(b"img", b"mask", 3, {"pipeline_type": "512"}))
    assert seen == [(b"img", b"mask", 3, {"pipeline_type": "512"})]
    assert mesh_from_npz(answer.mesh_npz)[0].shape == (4, 3)
    assert mesh_from_npz(answer.mesh_npz)[2] is None


def test_shape_backend_selects_the_trellis2_adapter(tmp_path: Path) -> None:
    cfg = yaml.safe_load(CONFIG.read_text())
    cfg["shape"]["backend"] = "trellis2"
    assert default_model(cfg, tmp_path).name == "trellis2"
    cfg["shape"]["backend"] = "nope"
    with pytest.raises(ValueError, match="no adapter"):
        default_model(cfg, tmp_path)


def test_disabled_background_remover_holds_nothing_and_refuses_to_run() -> None:
    remover = DisabledRembg(model_name="briaai/RMBG-2.0")
    assert remover.to("cuda") is remover and remover.cpu() is remover
    with pytest.raises(RuntimeError, match="alpha"):
        remover(object())


def test_a_failed_draw_is_skipped_reported_and_cached(tmp_path: Path) -> None:
    calls: list[Trellis2Call] = []

    def transport(call: Trellis2Call) -> Trellis2Answer:
        calls.append(call)
        if call.seed % 2:
            return Trellis2Answer(b"", json.dumps({"failed": "max(): numel() == 0"}))
        return FakeTransport()(call)

    model = Trellis2(_config(), transport, tmp_path / "cache")
    view = _view(tmp_path)
    odd = next(seed for seed in range(10) if seed % 2)
    with pytest.raises(SampleFailed, match="numel"):
        model.sample("o", view, odd)
    with pytest.raises(SampleFailed):
        model.sample("o", view, odd)
    assert len(calls) == 1  # the failure is cached like a mesh

    failures: list[str] = []
    samples = draw_samples(model, "o", [view], 6, base_seed=0, failures=failures)
    assert len(samples) + len(failures) == 6
    assert all("frame 10 seed" in f for f in failures)


def test_vertex_colors_come_from_the_voxel_a_vertex_lies_in() -> None:
    size = 1.0 / 8
    coords = np.array([[0, 0, 0], [4, 4, 4], [7, 7, 7]])
    base = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    # centres of voxel (0,0,0), (4,4,4) and (7,7,7), then a vertex one voxel off the second
    verts = np.array([[-0.5 + 0.5 * size] * 3, [0.0 + 0.5 * size] * 3, [0.5 - 0.5 * size] * 3])
    verts = np.vstack([verts, verts[1] + [size, 0.0, 0.0]])
    colors, missing = vertex_colors(verts, coords, base, size)
    assert missing == 0
    np.testing.assert_array_equal(colors[:3], [[255, 0, 0], [0, 255, 0], [0, 0, 255]])
    np.testing.assert_array_equal(colors[3], [0, 255, 0])  # nearest occupied voxel


def test_vertices_far_from_every_voxel_get_grey_and_are_counted() -> None:
    size = 1.0 / 8
    colors, missing = vertex_colors(
        np.array([[0.4, 0.4, 0.4]]), np.array([[0, 0, 0]]), np.array([[1.0, 0.0, 0.0]]), size
    )
    assert missing == 1
    np.testing.assert_array_equal(colors[0], [128, 128, 128])


def test_colors_travel_with_the_mesh_and_are_checked() -> None:
    colors = (np.arange(12).reshape(4, 3) * 20).astype(np.uint8)
    _, _, got = mesh_from_npz(mesh_npz(TETRA_V, TETRA_F, colors))
    np.testing.assert_array_equal(got, colors)
    with pytest.raises(ValueError, match="colour"):
        mesh_from_npz(mesh_npz(TETRA_V, TETRA_F, colors[:3]))
