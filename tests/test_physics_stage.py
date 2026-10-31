"""The physics stage end to end, on a small synthetic scene with no hosted call."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import pytest
import trimesh
import yaml

from video2sim.runner import OUTPUT_FILE, StageContext
from video2sim.schemas import (
    SE3,
    CameraFrame,
    Flag,
    Instances,
    Intrinsics,
    ObjectInstance,
    Plane,
    RegisteredObject,
    RegisteredObjects,
    SimAssets,
    WorldModel,
)
from video2sim.stages.physics_stage import CASES, PhysicsStage, physics_stage_config

ROOT = Path(__file__).resolve().parents[1]
K = Intrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width_px=640, height_px=480)
FLAT = Plane(normal=(0.0, 0.0, 1.0), offset_m=0.0)


def wedge() -> trimesh.Trimesh:
    """A solid with its volume centroid away from the origin, so the frame has to move."""
    v = [[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.05], [0.1, 0, 0.05], [0, 0.1, 0.05]]
    f = [[0, 2, 1], [3, 4, 5], [0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4], [2, 0, 3], [2, 3, 5]]
    return trimesh.Trimesh(np.array(v, dtype=float), np.array(f), process=True)


def _write(run_dir: Path, name: str, text: str) -> None:
    folder = run_dir / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / OUTPUT_FILE).write_text(text)


class FakeClient:
    """Answers in grams; ``None`` gives a reply that cannot be read."""

    def __init__(self, grams: tuple[float, float, float] | None) -> None:
        self.grams = grams

    def ask(self, prompt: str, images: Sequence[Path]) -> str:
        assert "block" in prompt and len(images) == 1
        if self.grams is None:
            return "no idea"
        low, nominal, high = self.grams
        return f'{{"mass_low_g": {low}, "mass_nominal_g": {nominal}, "mass_high_g": {high}}}'


WIDE = FakeClient((1.0, 100.0, 1e5))  # brackets any geometry mass, so the geometry mass stands


def run_stage(
    tmp_path: Path,
    mesh: trimesh.Trimesh,
    pose: SE3,
    client: FakeClient | None = WIDE,
    vlm_mass: bool = True,
) -> tuple[SimAssets, Path]:
    run_dir = tmp_path / "run"
    (run_dir / "frames").mkdir(parents=True)
    (run_dir / "segmentation" / "masks" / "obj_000").mkdir(parents=True)
    cv2.imwrite(str(run_dir / "frames" / "0.png"), np.full((48, 64, 3), 128, np.uint8))
    mask = np.zeros((48, 64), np.uint8)
    mask[10:30, 20:40] = 255
    cv2.imwrite(str(run_dir / "segmentation" / "masks" / "obj_000" / "00000.png"), mask)
    (run_dir / "shape" / "meshes").mkdir(parents=True)
    mesh.export(run_dir / "shape" / "meshes" / "obj_000.ply")
    world = WorldModel(
        intrinsics=K,
        frames=[
            CameraFrame(
                frame_index=0, image_path="frames/0.png", T_world_from_camera=SE3.identity()
            )
        ],
        scale_factor=1.0,
        scale_source="marker",
        table_plane=FLAT,
        point_cloud_path="reconstruction/points.ply",
        mean_reprojection_error_px=0.5,
        registered_fraction=1.0,
    )
    instance = ObjectInstance(
        object_id="obj_000",
        label="block",
        mask_dir="segmentation/masks/obj_000",
        point_cluster_path="segmentation/clusters/obj_000.ply",
        keyframes=[0],
        flags=[Flag.THIN],
        material_class="wood",
        hollow="solid",
        material_confidence=0.9,
    )
    registered = RegisteredObject(
        object_id="obj_000",
        mesh_path="shape/meshes/obj_000.ply",
        T_world_from_object=pose,
        geometry_source="sam3d",
        fit_score=0.9,
        flags=[Flag.THIN],
    )
    _write(run_dir, "reconstruction", world.model_dump_json())
    _write(run_dir, "segmentation", Instances(objects=[instance]).model_dump_json())
    _write(run_dir, "shape", RegisteredObjects(objects=[registered]).model_dump_json())

    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    cfg["physics"]["vlm_mass"]["enabled"] = vlm_mass
    targets = yaml.safe_load((ROOT / "configs/targets.yaml").read_text(encoding="utf-8"))
    out_dir = run_dir / "physics.tmp"
    out_dir.mkdir()
    stage = PhysicsStage(ROOT / "configs/material_table.yaml", client)
    ctx = StageContext(
        stage="physics",
        run_dir=run_dir,
        video=tmp_path / "video.mp4",
        out_dir=out_dir,
        config=physics_stage_config(cfg, targets),
        seed=0,
        dependencies=stage.dependencies,
        output_types={
            "reconstruction": WorldModel,
            "segmentation": Instances,
            "shape": RegisteredObjects,
        },
    )
    assets = stage.run(ctx)
    out_dir.rename(run_dir / "physics")
    return assets, run_dir


@pytest.fixture
def block(tmp_path: Path) -> tuple[SimAssets, Path, SE3, trimesh.Trimesh]:
    mesh = trimesh.creation.box(extents=(0.1, 0.08, 0.04))
    mesh.visual.vertex_colors = np.full((len(mesh.vertices), 4), [200, 50, 10, 255], np.uint8)
    mesh.apply_translation([0.01, -0.02, 0.005])  # away from the centroid, as a mesh can be
    pose = np.eye(4)
    pose[:3, 3] = [0.2, 0.1, 0.02 - 0.005]
    t = SE3.from_numpy(pose)
    assets, run_dir = run_stage(tmp_path, mesh, t)
    return assets, run_dir, t, mesh


def test_a_block_becomes_a_valid_asset_with_ranges(
    block: tuple[SimAssets, Path, SE3, trimesh.Trimesh],
) -> None:
    assets, _, _, _ = block
    asset = assets.assets[0]
    p = asset.properties
    assert asset.is_valid and {"watertight", "manifold", "collision_loads"} <= set(asset.checks)
    assert p.material_class == "wood" and not p.is_hollow
    for e in (p.mass_kg, p.density_kg_m3, p.friction_coefficient, p.restitution):
        assert e.low <= e.nominal <= e.high
    assert asset.flags == [Flag.THIN]


def test_the_origin_moves_and_the_world_surface_does_not(
    block: tuple[SimAssets, Path, SE3, trimesh.Trimesh],
) -> None:
    assets, run_dir, pose, mesh = block
    asset = assets.assets[0]
    visual = cast(trimesh.Trimesh, trimesh.load(run_dir / asset.visual_mesh_path, force="mesh"))
    assert np.allclose(visual.center_mass, 0.0, atol=1.5e-3)
    moved = asset.T_world_from_object.to_numpy()
    before = trimesh.transform_points(mesh.bounds, pose.to_numpy())
    after = trimesh.transform_points(visual.bounds, moved)
    assert np.allclose(before, after, atol=1e-6)
    part = cast(
        trimesh.Trimesh, trimesh.load(run_dir / asset.collision_mesh_paths[0], force="mesh")
    )
    assert part.is_watertight
    assert np.allclose(trimesh.transform_points(part.bounds, moved), before, atol=2e-3)


def test_settling_covers_every_case_and_does_not_touch_the_checks(
    block: tuple[SimAssets, Path, SE3, trimesh.Trimesh],
) -> None:
    assets, run_dir, _, _ = block
    report = json.loads((run_dir / "physics" / "settling.json").read_text())
    assert set(report) == set(CASES)
    assert all(report[c]["stable"] for c in CASES)
    assert "stable" not in assets.assets[0].checks


def test_an_object_in_the_air_is_reported_and_stays_valid(tmp_path: Path) -> None:
    mesh = wedge()
    pose = np.eye(4)
    pose[2, 3] = 0.3
    assets, run_dir = run_stage(tmp_path, mesh, SE3.from_numpy(pose))
    report = json.loads((run_dir / "physics" / "settling.json").read_text())
    assert not report["nominal"]["stable"]
    assert report["nominal"]["bodies"][0]["object_id"] == "obj_000"
    assert assets.assets[0].is_valid


def test_the_visual_mesh_keeps_its_colors(
    block: tuple[SimAssets, Path, SE3, trimesh.Trimesh],
) -> None:
    assets, run_dir, _, _ = block
    path = run_dir / assets.assets[0].visual_mesh_path
    visual = cast(trimesh.Trimesh, trimesh.load(path, force="mesh"))
    assert isinstance(visual.visual, trimesh.visual.ColorVisuals)
    assert visual.visual.kind == "vertex"
    assert np.all(np.asarray(visual.visual.vertex_colors)[:, :3] == [200, 50, 10])


def box_mesh() -> trimesh.Trimesh:
    return cast(trimesh.Trimesh, trimesh.creation.box(extents=(0.1, 0.08, 0.04)))


def resting() -> SE3:
    pose = np.eye(4)
    pose[2, 3] = 0.02
    return SE3.from_numpy(pose)


def test_a_mass_outside_the_model_range_is_replaced_and_inertia_follows(tmp_path: Path) -> None:
    base, _ = run_stage(tmp_path / "a", box_mesh(), resting())
    (tmp_path / "b").mkdir()
    grams = (9000.0, 10000.0, 11000.0)
    assets, run_dir = run_stage(tmp_path / "b", box_mesh(), resting(), FakeClient(grams))
    before, after = base.assets[0].properties, assets.assets[0].properties
    assert after.mass_kg.nominal == pytest.approx(10.0)
    assert after.mass_kg.low == pytest.approx(9.0) and after.mass_kg.high == pytest.approx(11.0)
    ratio = after.mass_kg.nominal / before.mass_kg.nominal
    assert np.allclose(
        np.asarray(after.inertia_kg_m2), ratio * np.asarray(before.inertia_kg_m2), rtol=1e-6
    )
    report = json.loads((run_dir / "physics" / "report.json").read_text())
    assert report["obj_000"]["mass_source"] == "model"


def test_a_mass_inside_the_model_range_stands(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    _, run_dir = run_stage(tmp_path / "a", box_mesh(), resting())
    report = json.loads((run_dir / "physics" / "report.json").read_text())
    assert report["obj_000"]["mass_source"] == "geometry"


def test_an_unreadable_answer_keeps_the_geometry_mass_and_flags_it(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    assets, run_dir = run_stage(tmp_path / "a", box_mesh(), resting(), FakeClient(None))
    asset = assets.assets[0]
    assert Flag.LOW_CONFIDENCE_MATERIAL in asset.flags and asset.is_valid
    report = json.loads((run_dir / "physics" / "report.json").read_text())
    assert report["obj_000"]["mass_source"] == "geometry_no_answer"
    assert report["obj_000"]["reasons"]


def test_disabled_vlm_mass_needs_no_client_or_key_and_raises_no_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("BASETEN_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)  # no .env to read a key from
    (tmp_path / "a").mkdir()
    assets, run_dir = run_stage(tmp_path / "a", box_mesh(), resting(), None, vlm_mass=False)
    assert Flag.LOW_CONFIDENCE_MATERIAL not in assets.assets[0].flags
    report = json.loads((run_dir / "physics" / "report.json").read_text())
    assert report["obj_000"]["mass_source"] == "geometry_only"
    assert not (run_dir / "physics.work" / "vlm_mass_cache").exists()
