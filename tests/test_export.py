"""USD and URDF export, the QA report and the stage, on a small synthetic scene."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
import trimesh
import yaml
from pxr import Usd, UsdGeom

from video2sim.runner import OUTPUT_FILE, StageContext
from video2sim.schemas import (
    SE3,
    CameraFrame,
    Estimate,
    Flag,
    Instances,
    Intrinsics,
    ObjectInstance,
    PhysicalProperties,
    Plane,
    RegisteredObject,
    RegisteredObjects,
    SceneExport,
    SimAsset,
    SimAssets,
    WorldModel,
)
from video2sim.stages.export import ExportCheckFailed, ExportStage, export_stage_config
from video2sim.stages.physics_stage import CASES
from video2sim.stages.scene_render import RenderObject, render, side_by_side
from video2sim.stages.settling import SettlingConfig
from video2sim.stages.urdf_writer import (
    check_model,
    settle_urdfs,
    urdf_scene_model,
    write_urdf,
)
from video2sim.stages.usd_writer import (
    check_usd,
    linear_from_srgb,
    principal_inertia,
    read_usd,
    write_usd,
)

ROOT = Path(__file__).resolve().parents[1]
FLAT = Plane(normal=(0.0, 0.0, 1.0), offset_m=0.0)
K = Intrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width_px=640, height_px=480)
SIDES = (0.10, 0.06, 0.04)
MASS = 0.2
GRAVITY = 9.80665
TOLERANCE, TOLERANCE_M = 1e-4, 1e-4


def box_inertia() -> list[list[float]]:
    a, b, c = SIDES
    moments = [MASS / 12 * (b**2 + c**2), MASS / 12 * (a**2 + c**2), MASS / 12 * (a**2 + b**2)]
    return [[moments[i] if i == j else 0.0 for j in range(3)] for i in range(3)]


def yawed(yaw: float, x: float, y: float, z: float = SIDES[2] / 2) -> SE3:
    t = np.eye(4)
    t[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
    t[:3, 3] = [x, y, z]
    return SE3.from_numpy(t)


def make_asset(
    run_dir: Path, name: str, pose: SE3, checks: dict[str, bool] | None = None
) -> SimAsset:
    for folder in ("meshes", "collision"):
        (run_dir / "physics" / folder).mkdir(parents=True, exist_ok=True)
    mesh = trimesh.creation.box(SIDES)
    mesh.visual.vertex_colors = np.tile([200, 40, 10, 255], (len(mesh.vertices), 1))
    mesh.export(run_dir / "physics" / "meshes" / f"{name}.ply")
    trimesh.creation.box(SIDES).export(run_dir / "physics" / "collision" / f"{name}_0.obj")
    range_ = Estimate(low=0.3, nominal=0.5, high=0.7)
    return SimAsset(
        object_id=name,
        T_world_from_object=pose,
        visual_mesh_path=f"physics/meshes/{name}.ply",
        collision_mesh_paths=[f"physics/collision/{name}_0.obj"],
        properties=PhysicalProperties(
            material_class="wood",
            is_hollow=False,
            mass_kg=Estimate(low=0.1, nominal=MASS, high=0.4),
            density_kg_m3=Estimate(low=400, nominal=600, high=800),
            friction_coefficient=range_,
            restitution=Estimate(low=0.1, nominal=0.2, high=0.3),
            inertia_kg_m2=box_inertia(),
            confidence=0.9,
        ),
        checks=checks or {"watertight": True},
    )


@pytest.fixture
def settling() -> SettlingConfig:
    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text())
    targets = yaml.safe_load((ROOT / "configs/targets.yaml").read_text())
    return SettlingConfig.from_config(cfg, targets)


@pytest.fixture
def scene(tmp_path: Path) -> tuple[Path, list[SimAsset]]:
    return tmp_path, [
        make_asset(tmp_path, "obj_000", yawed(0.5, 0.1, 0.0)),
        make_asset(tmp_path, "obj_001", yawed(-1.1, -0.2, 0.15)),
    ]


def written_usd(run_dir: Path, assets: list[SimAsset]) -> Path:
    path = run_dir / "export" / "scene.usd"
    write_usd(path, assets, run_dir, FLAT, GRAVITY, 2.0)
    return path


def problems(path: Path, assets: list[SimAsset]) -> list[str]:
    return check_usd(path, assets, GRAVITY, TOLERANCE, TOLERANCE_M)


def test_usd_stage_is_z_up_in_meters_and_reads_back(scene: tuple[Path, list[SimAsset]]) -> None:
    run_dir, assets = scene
    path = written_usd(run_dir, assets)
    read = read_usd(path)
    assert (read.up_axis, read.meters_per_unit, read.kilograms_per_unit) == ("Z", 1.0, 1.0)
    assert problems(path, assets) == []
    first = read.objects["obj_000"]
    assert first.mass_kg == pytest.approx(MASS)
    assert first.ranges["mass_kg"] == pytest.approx((0.1, MASS, 0.4))
    assert first.friction == pytest.approx(0.5)
    assert first.has_color


def test_usd_inertia_with_cross_terms_survives_the_principal_axes(tmp_path: Path) -> None:
    asset = make_asset(tmp_path, "obj_000", yawed(0.3, 0.0, 0.0))
    i = np.array(box_inertia())
    a = 0.6
    r = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    tilted = r @ i @ r.T
    p = asset.properties.model_copy(update={"inertia_kg_m2": tilted.tolist()})
    asset = asset.model_copy(update={"properties": p})
    read = read_usd(written_usd(tmp_path, [asset])).objects["obj_000"]
    assert read.inertia_kg_m2 == pytest.approx(tilted, rel=1e-5, abs=1e-9)


def test_principal_axes_are_a_proper_rotation() -> None:
    moments, q = principal_inertia(np.array(box_inertia()))
    assert np.linalg.norm(q) == pytest.approx(1.0)
    assert moments[0] <= moments[1] <= moments[2]


def test_usd_check_catches_wrong_units_and_a_changed_mass(
    scene: tuple[Path, list[SimAsset]],
) -> None:
    run_dir, assets = scene
    path = written_usd(run_dir, assets)
    stage = Usd.Stage.Open(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    stage.GetRootLayer().Save()
    p = assets[0].properties
    changed = p.model_copy(update={"mass_kg": Estimate(low=0.1, nominal=0.3, high=0.4)})
    wrong = [assets[0].model_copy(update={"properties": changed}), assets[1]]
    found = " ".join(problems(path, wrong))
    assert "per unit" in found
    assert "mass" in found


def test_display_color_is_linear() -> None:
    bytes_ = np.array([[0, 255, 188, 255]], dtype=np.uint8)
    assert linear_from_srgb(bytes_)[0] == pytest.approx([0.0, 1.0, 0.503], abs=1e-3)


def test_urdf_inertia_uses_the_matrix_entries(tmp_path: Path) -> None:
    asset = make_asset(tmp_path, "obj_000", yawed(0.0, 0.0, 0.0))
    i = np.array(box_inertia())
    i[0, 1] = i[1, 0] = -1e-5
    p = asset.properties.model_copy(update={"inertia_kg_m2": i.tolist()})
    asset = asset.model_copy(update={"properties": p})
    root = ET.parse(write_urdf(tmp_path / "urdf", asset, tmp_path)).getroot()
    inertia = root.find("link/inertial/inertia")
    assert inertia is not None
    assert float(inertia.get("ixy", "0")) == pytest.approx(-1e-5)
    assert float(inertia.get("ixx", "0")) == pytest.approx(i[0, 0])


def urdf_model(
    run_dir: Path, assets: list[SimAsset], settling: SettlingConfig
) -> tuple[Any, dict[str, Path]]:
    urdfs = {a.object_id: write_urdf(run_dir / "urdf", a, run_dir) for a in assets}
    return urdf_scene_model(urdfs, assets, FLAT, settling), urdfs


def test_urdf_scene_matches_the_assets_and_rests(
    scene: tuple[Path, list[SimAsset]], settling: SettlingConfig
) -> None:
    run_dir, assets = scene
    model, _ = urdf_model(run_dir, assets, settling)
    assert check_model(model, assets, run_dir, TOLERANCE, TOLERANCE_M) == []
    result = settle_urdfs(model, assets, FLAT, settling)
    assert all(b.stable for b in result.bodies)
    assert result.final_poses["obj_000"][2, 3] == pytest.approx(SIDES[2] / 2, abs=1e-3)


def test_urdf_check_catches_millimeter_meshes(
    scene: tuple[Path, list[SimAsset]], settling: SettlingConfig
) -> None:
    run_dir, assets = scene
    urdfs = {a.object_id: write_urdf(run_dir / "urdf", a, run_dir) for a in assets}
    big = trimesh.creation.box(np.array(SIDES) * 1000)
    big.export(run_dir / "urdf" / "meshes" / "obj_000_collision_0.obj")
    model = urdf_scene_model(urdfs, assets, FLAT, settling)
    found = check_model(model, assets, run_dir, TOLERANCE, TOLERANCE_M)
    assert any("obj_000" in p and "bounds" in p for p in found)


def test_render_puts_the_object_where_the_camera_sees_it() -> None:
    mesh = trimesh.creation.box((0.1, 0.1, 0.1))
    camera = np.eye(4)
    camera[:3, 3] = [0.0, 0.0, -1.0]  # OpenCV camera looks along +z, at the origin
    image = render([RenderObject(mesh, np.eye(4))], camera, K)
    changed = np.argwhere(np.any(image != image[0, 0], axis=2))
    assert changed[:, 1].mean() == pytest.approx(K.cx, abs=2)
    assert changed[:, 0].mean() == pytest.approx(K.cy, abs=2)
    # half width 0.05 m at the front face, 0.95 m away
    assert np.max(changed[:, 1]) - K.cx == pytest.approx(500 * 0.05 / 0.95, abs=2)


def test_render_with_an_object_behind_the_camera_is_empty() -> None:
    image = render([RenderObject(trimesh.creation.box((0.1, 0.1, 0.1)), np.eye(4))], np.eye(4), K)
    assert np.all(image == image[0, 0])


def test_side_by_side_resizes_the_render() -> None:
    joined = side_by_side(np.zeros((48, 64, 3), np.uint8), np.zeros((480, 640, 3), np.uint8))
    assert joined.shape == (48, 128, 3)


def settling_case(names: list[str]) -> dict[str, Any]:
    body = {"max_drift_m": 0.001, "stable": True, "ignored": False}
    return {"stable": True, "bodies": [{"object_id": n, **body} for n in names]}


def run_stage(run_dir: Path, assets: list[SimAsset]) -> SceneExport:
    world = WorldModel(
        intrinsics=K,
        frames=[
            CameraFrame(
                frame_index=0,
                image_path="frames/0.png",
                T_world_from_camera=SE3.from_numpy(
                    np.array(
                        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 1.0], [0, 0, 0, 1]], dtype=float
                    )
                ),
            )
        ],
        scale_factor=1.0,
        scale_source="marker",
        table_plane=FLAT,
        point_cloud_path="reconstruction/points.ply",
        mean_reprojection_error_px=0.5,
        registered_fraction=1.0,
        flags=["a scene warning"],
    )
    (run_dir / "frames").mkdir()
    cv2.imwrite(str(run_dir / "frames" / "0.png"), np.full((48, 64, 3), 128, np.uint8))
    names = [a.object_id for a in assets]
    instances = Instances(
        objects=[
            ObjectInstance(
                object_id=n,
                label="block",
                mask_dir=f"segmentation/masks/{n}",
                point_cluster_path=f"segmentation/clusters/{n}.ply",
                keyframes=[0],
            )
            for n in names
        ]
    )
    registered = RegisteredObjects(
        objects=[
            RegisteredObject(
                object_id=a.object_id,
                mesh_path=f"shape/meshes/{a.object_id}.ply",
                T_world_from_object=a.T_world_from_object,
                geometry_source="sam3d",
                fit_score=0.9,
            )
            for a in assets
        ]
    )
    outputs = {
        "reconstruction": world,
        "segmentation": instances,
        "shape": registered,
        "physics": SimAssets(assets=assets),
    }
    for folder, model in outputs.items():
        (run_dir / folder).mkdir(exist_ok=True)
        (run_dir / folder / OUTPUT_FILE).write_text(model.model_dump_json())
    entry = {
        "collision_kind": "box",
        "collision_parts": 1,
        "mass_source": "model",
        "reasons": ["a physics note"],
    }
    (run_dir / "physics" / "report.json").write_text(json.dumps({n: entry for n in names}))
    (run_dir / "physics" / "settling.json").write_text(
        json.dumps({c: settling_case(names) for c in CASES})
    )
    out_dir = run_dir / "export"
    out_dir.mkdir(exist_ok=True)
    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text())
    targets = yaml.safe_load((ROOT / "configs/targets.yaml").read_text())
    ctx = StageContext(
        stage="export",
        run_dir=run_dir,
        video=run_dir / "v.mp4",
        out_dir=out_dir,
        config=export_stage_config(cfg, targets),
        seed=0,
        dependencies=ExportStage.dependencies,
        output_types={
            "reconstruction": WorldModel,
            "segmentation": Instances,
            "shape": RegisteredObjects,
            "physics": SimAssets,
        },
    )
    return ExportStage().run(ctx)


def test_stage_writes_the_package_and_a_clean_report(
    scene: tuple[Path, list[SimAsset]],
) -> None:
    run_dir, assets = scene
    out = run_stage(run_dir, assets)
    assert (run_dir / out.usd_path).exists()
    assert set(out.urdf_paths) == {"obj_000", "obj_001"}
    report = json.loads((run_dir / "export" / "qa_report.json").read_text())
    assert report["objects_exported"] == 2
    assert report["urdf_check"]["agrees"]
    assert report["objects"][0]["mass_source"] == "model"
    assert report["objects"][0]["mass_kg"]["high"] == pytest.approx(0.4)
    assert report["scene_flags"] == ["a scene warning"]
    assert "not verified" in report["validated_in"]
    assert [(o["vertices"], o["faces"]) for o in report["objects"]] == [(8, 12), (8, 12)]
    assert report["discovery"] is None
    assert [o["found_by"] for o in report["objects"]] == ["generic", "generic"]
    html = (run_dir / out.qa_report_path).read_text()
    assert "qa/view_00000.jpg" in html
    assert "polygons" in html and ">12<" in html
    assert "Discovery" not in html
    assert (run_dir / "export" / "qa" / "view_00000.jpg").exists()


def test_stage_blocks_a_failed_asset_and_reports_it(tmp_path: Path) -> None:
    good = make_asset(tmp_path, "obj_000", yawed(0.0, 0.1, 0.0))
    bad = make_asset(tmp_path, "obj_001", yawed(0.0, -0.2, 0.0), {"watertight": False})
    bad = bad.model_copy(update={"flags": [Flag.FAILED_VALIDATION]})
    out = run_stage(tmp_path, [good, bad])
    assert set(out.urdf_paths) == {"obj_000"}
    assert set(read_usd(tmp_path / out.usd_path).objects) == {"obj_000"}
    assert [(s.object_id, s.reasons) for s in out.skipped] == [("obj_001", ["watertight"])]
    report = json.loads((tmp_path / "export" / "qa_report.json").read_text())
    assert report["status"] == "review"
    assert report["objects"][1]["status"] == "skipped"


def test_stage_fails_when_a_file_does_not_read_back(
    scene: tuple[Path, list[SimAsset]], monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, assets = scene
    monkeypatch.setattr("video2sim.stages.export.check_usd", lambda *a: ["wrong"])
    with pytest.raises(ExportCheckFailed, match="wrong"):
        run_stage(run_dir, assets)
