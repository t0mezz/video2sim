from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
import trimesh
import yaml
from numpy.typing import NDArray

from video2sim.models.shape_model import ShapeSample, ShapeView
from video2sim.runner import OUTPUT_FILE, StageContext
from video2sim.schemas import (
    SE3,
    CameraFrame,
    Flag,
    Instances,
    Intrinsics,
    ObjectInstance,
    Plane,
    RegisteredObjects,
    WorldModel,
)
from video2sim.stages.registration import rotation_from_vector
from video2sim.stages.shape import ShapeStage, ShapeStageConfig, spread

F64 = NDArray[np.float64]
CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
K = Intrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width_px=640, height_px=480)

# A box in the model's own frame and units, 10x the real object; model +y is its up axis.
MODEL = trimesh.creation.box(extents=[0.8, 0.5, 1.2])
TRUE_SCALE = 0.1
TRUE_R = rotation_from_vector(np.array([0.0, 0.0, np.radians(30.0)])) @ rotation_from_vector(
    np.array([np.radians(90.0), 0.0, 0.0])
)
REST_T = np.array([0.1, -0.05, 0.025])  # bottom face on the table (half of 0.5 * 0.1)


def _world_vertices(points: F64, lift_m: float = 0.0) -> F64:
    out: F64 = (TRUE_SCALE * points) @ TRUE_R.T + REST_T + np.array([0.0, 0.0, lift_m])
    return out


def _look_at(eye: F64, target: F64) -> SE3:
    z = (target - eye) / np.linalg.norm(target - eye)
    y = np.array([0.0, 0.0, -1.0]) - z * z[2] * -1.0
    y /= np.linalg.norm(y)
    m = np.eye(4)
    m[:3, :3] = np.column_stack([np.cross(y, z), y, z])
    m[:3, 3] = eye
    return SE3.from_numpy(m)


def _silhouette(vertices: F64, cam: SE3) -> NDArray[np.bool_]:
    a = cam.inverse().to_numpy()
    c = vertices @ a[:3, :3].T + a[:3, 3]
    uv = np.column_stack([K.fx * c[:, 0] / c[:, 2] + K.cx, K.fy * c[:, 1] / c[:, 2] + K.cy])
    mask = np.zeros((K.height_px, K.width_px), dtype=np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(np.round(uv).astype(np.int32)), 1)
    return mask.astype(bool)


class FakeModel:
    """Returns ``mesh`` with the true pose, turned a little, in the keyframe camera."""

    name = "sam3d"

    def __init__(self, cameras: dict[int, SE3], mesh: trimesh.Trimesh, lift_m: float) -> None:
        self.cameras, self.mesh, self.lift_m = cameras, mesh, lift_m
        self.calls: list[tuple[int, int]] = []

    def sample(self, object_id: str, view: ShapeView, seed: int) -> ShapeSample:
        self.calls.append((view.frame_index, seed))
        turn = rotation_from_vector(np.radians(6.0) * np.array([0.3, -0.5, 0.8]))
        T = np.eye(4)
        T[:3, :3] = turn @ TRUE_R
        T[:3, 3] = REST_T + np.array([0.0, 0.0, self.lift_m])
        camera_from_model = self.cameras[view.frame_index].inverse().to_numpy() @ T
        return ShapeSample(
            object_id=object_id,
            frame_index=view.frame_index,
            seed=seed,
            vertices_model=np.asarray(self.mesh.vertices, dtype=float),
            faces=np.asarray(self.mesh.faces, dtype=np.int64),
            T_camera_from_model=SE3.from_numpy(camera_from_model),
            scale_guess=0.37,  # a wrong guess, which must not be used
            model="sam3d",
            checkpoint="fake@1",
        )


def _make_run(tmp: Path, lift_m: float = 0.0) -> tuple[Path, dict[int, SE3]]:
    corners = _world_vertices(np.asarray(MODEL.vertices), lift_m)
    centre = corners.mean(axis=0)
    cameras = {
        i: _look_at(centre + np.array([0.45 * np.cos(a), 0.45 * np.sin(a), 0.3]), centre)
        for i, a in enumerate(np.linspace(-1.2, 1.2, 6))
    }
    run = tmp / "run"
    (run / "reconstruction").mkdir(parents=True)
    world = WorldModel(
        intrinsics=K,
        frames=[
            CameraFrame(
                frame_index=i, image_path=f"capture/frames/{i:05d}.png", T_world_from_camera=c
            )
            for i, c in cameras.items()
        ],
        scale_factor=1.0,
        scale_source="marker",
        table_plane=Plane(normal=(0.0, 0.0, 1.0), offset_m=0.0),
        point_cloud_path="reconstruction/points.ply",
        mean_reprojection_error_px=0.5,
        registered_fraction=1.0,
    )
    (run / "reconstruction" / OUTPUT_FILE).write_text(world.model_dump_json())
    seg = run / "segmentation"
    (seg / "masks" / "obj_000").mkdir(parents=True)
    (seg / "clusters").mkdir()
    for i, cam in cameras.items():
        mask = _silhouette(corners, cam)
        cv2.imwrite(str(seg / "masks" / "obj_000" / f"{i:05d}.png"), mask.astype(np.uint8) * 255)
    surface, _ = trimesh.sample.sample_surface(MODEL, 1500, seed=0)
    np.save(seg / "clusters" / "obj_000.npy", _world_vertices(np.asarray(surface), lift_m))
    instances = Instances(
        objects=[
            ObjectInstance(object_id="obj_000", label="box", mask_dir="segmentation/masks/obj_000",
                           point_cluster_path="segmentation/clusters/obj_000.npy",
                           keyframes=[2, 3, 0])
        ]
    )  # fmt: skip
    (seg / OUTPUT_FILE).write_text(instances.model_dump_json())
    return run, cameras


def _shape_config(**overrides: object) -> dict[str, Any]:
    cfg: dict[str, Any] = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["shape"]
    cfg.update({"keyframes_per_object": 2, "samples_per_keyframe": 1})
    cfg.update(overrides)
    return cfg


def _run_stage(run: Path, model: FakeModel, cfg: dict[str, Any]) -> tuple[RegisteredObjects, Any]:
    out = run / "shape"
    out.mkdir()
    ctx = StageContext(
        stage="shape",
        run_dir=run,
        video=run / "video.mp4",
        out_dir=out,
        config=cfg,
        seed=7,
        dependencies=("reconstruction", "segmentation"),
        output_types={"reconstruction": WorldModel, "segmentation": Instances},
    )
    result = ShapeStage(lambda _cfg, _work: model).run(ctx)
    return result, ctx


def _placed(run: Path, result: RegisteredObjects) -> F64:
    obj = result.objects[0]
    mesh: Any = trimesh.load(run / obj.mesh_path, force="mesh")
    T = obj.T_world_from_object.to_numpy()
    out: F64 = np.asarray(mesh.vertices) @ T[:3, :3].T + T[:3, 3]
    return out


def _corner_error_m(placed: F64, truth: F64) -> float:
    d = np.linalg.norm(truth[:, None, :] - placed[None, :, :], axis=2)
    return float(np.max(np.min(d, axis=1)))


def test_stage_registers_the_sample_and_sets_it_on_the_table(tmp_path: Path) -> None:
    run, cameras = _make_run(tmp_path)
    model = FakeModel(cameras, MODEL, 0.0)
    result, ctx = _run_stage(run, model, _shape_config())
    obj = result.objects[0]
    assert obj.geometry_source == "sam3d"
    assert obj.flags == []
    assert obj.fit_score is not None and obj.fit_score > 0.9
    placed = _placed(run, result)
    assert _corner_error_m(placed, _world_vertices(np.asarray(MODEL.vertices))) < 0.003
    assert float(np.min(placed[:, 2])) == pytest.approx(0.0, abs=1e-9)  # exactly on the table
    # the object frame's origin is the volume centroid, and the scale guess was not used
    mesh: Any = trimesh.load(run / obj.mesh_path, force="mesh")
    np.testing.assert_allclose(mesh.center_mass, 0.0, atol=1e-6)
    assert [f for f, _ in model.calls] == [2, 3]  # the first two keyframes
    assert ctx.checkpoints == {"sam3d": "fake@1"}
    report = (run / "shape" / "report.json").read_text()
    assert '"kept": 0' in report or '"kept": 1' in report


def test_a_stacked_object_keeps_its_height(tmp_path: Path) -> None:
    run, cameras = _make_run(tmp_path, lift_m=0.1)
    result, _ = _run_stage(run, FakeModel(cameras, MODEL, 0.1), _shape_config())
    placed = _placed(run, result)
    assert float(np.min(placed[:, 2])) == pytest.approx(0.1, abs=0.003)


def test_a_sample_below_the_fit_threshold_falls_back_to_the_cluster_hull(tmp_path: Path) -> None:
    run, cameras = _make_run(tmp_path)
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.6)
    result, _ = _run_stage(run, FakeModel(cameras, sphere, 0.0), _shape_config(min_fit_score=0.9))
    obj = result.objects[0]
    assert obj.geometry_source == "proxy_hull"
    assert obj.fit_score is None
    assert set(obj.flags) == {Flag.PROXY_GEOMETRY, Flag.LOW_FIT}
    cluster = np.load(run / "segmentation" / "clusters" / "obj_000.npy")
    placed = _placed(run, result)
    # the hull's vertices are cluster points (PLY stores float32)
    assert np.min(np.linalg.norm(placed[:, None] - cluster[None], axis=2), axis=1).max() < 1e-6


def test_the_stage_is_deterministic(tmp_path: Path) -> None:
    outputs = []
    for name in ("a", "b"):
        run, cameras = _make_run(tmp_path / name)
        result, _ = _run_stage(run, FakeModel(cameras, MODEL, 0.0), _shape_config())
        outputs.append((result.model_dump_json(), (run / "shape/meshes/obj_000.ply").read_bytes()))
    assert outputs[0] == outputs[1]


def test_spread_keeps_the_ends_and_at_most_count_frames() -> None:
    assert spread([5, 1, 3], 10) == [1, 3, 5]
    picks = spread(list(range(100)), 12)
    assert len(picks) == 12 and picks[0] == 0 and picks[-1] == 99


def test_config_rejects_a_fit_threshold_above_one() -> None:
    with pytest.raises(ValueError, match="min_fit_score"):
        ShapeStageConfig.from_config({"shape": _shape_config(min_fit_score=1.5)})
