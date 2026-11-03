"""Discovery block of the QA report (S12.9.5)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from video2sim.schemas import (
    SE3,
    CameraFrame,
    DiscoveryCheck,
    Estimate,
    Instances,
    Intrinsics,
    ObjectInstance,
    PhysicalProperties,
    Plane,
    RegisteredObject,
    RegisteredObjects,
    SimAsset,
    SimAssets,
    WorldModel,
)
from video2sim.stages.qa_report import build_report, write_report


def _estimate() -> Estimate:
    return Estimate(low=0.1, nominal=0.2, high=0.3)


def _world() -> WorldModel:
    return WorldModel(
        intrinsics=Intrinsics(fx=500.0, fy=500.0, cx=32.0, cy=24.0, width_px=64, height_px=48),
        frames=[
            CameraFrame(
                frame_index=0,
                image_path="frames/0.png",
                T_world_from_camera=SE3.identity(),
            )
        ],
        scale_factor=1.0,
        scale_source="marker",
        table_plane=Plane(normal=(0.0, 0.0, 1.0), offset_m=0.0),
        point_cloud_path="reconstruction/points.ply",
        mean_reprojection_error_px=0.5,
        registered_fraction=1.0,
    )


def _report(tmp_path: Path, instances: Instances) -> tuple[dict[str, Any], str]:
    registered = RegisteredObjects(
        objects=[
            RegisteredObject(
                object_id="obj_000",
                mesh_path="shape/meshes/obj_000.glb",
                T_world_from_object=SE3.identity(),
                geometry_source="sam3d",
                fit_score=0.9,
            )
        ]
    )
    assets = SimAssets(
        assets=[
            SimAsset(
                object_id="obj_000",
                T_world_from_object=SE3.identity(),
                visual_mesh_path="meshes/obj_000.glb",
                collision_mesh_paths=["collision/obj_000.obj"],
                properties=PhysicalProperties(
                    material_class="steel",
                    is_hollow=False,
                    mass_kg=_estimate(),
                    density_kg_m3=_estimate(),
                    friction_coefficient=_estimate(),
                    restitution=_estimate(),
                    inertia_kg_m2=[[2e-4, 0, 0], [0, 2e-4, 0], [0, 0, 1e-4]],
                    confidence=0.8,
                ),
                checks={"watertight": True},
            )
        ]
    )
    physics = {
        "obj_000": {
            "collision_kind": "box",
            "collision_parts": 1,
            "mass_source": "geometry",
            "reasons": [],
        }
    }
    report = build_report(
        run_id="r1",
        run_dir=tmp_path,
        world=_world(),
        instances=instances,
        registered=registered,
        assets=assets,
        physics=physics,
        settling={},
        skipped=[],
        urdf_check={"agrees": True, "problems": []},
        views=[],
    )
    folder = tmp_path / "export"
    folder.mkdir(exist_ok=True)
    html_path = write_report(report, folder)
    return report, html_path.read_text()


def _instance(
    found_by: Literal["generic", "noun", "cluster_box"] = "generic",
    flag_reasons: list[str] | None = None,
) -> ObjectInstance:
    return ObjectInstance(
        object_id="obj_000",
        label="pliers",
        mask_dir="segmentation/masks/obj_000",
        point_cluster_path="segmentation/clusters/obj_000.npy",
        keyframes=[0],
        category="pliers",
        found_by=found_by,
        flag_reasons=[] if flag_reasons is None else flag_reasons,
    )


def test_report_without_discovery_has_no_discovery_block(tmp_path: Path) -> None:
    report, html = _report(tmp_path, Instances(objects=[_instance()]))

    assert report["discovery"] is None
    assert report["objects"][0]["found_by"] == "generic"
    assert "Discovery" not in html
    assert "found by gap fill" not in html


def test_report_with_discovery_shows_gaps_and_provenance(tmp_path: Path) -> None:
    instances = Instances(
        objects=[
            _instance(
                found_by="noun",
                flag_reasons=["found by gap fill: noun 'pliers'"],
            )
        ],
        discovery=DiscoveryCheck(
            view_frames=[0, 10],
            view_coverage=0.75,
            listed=2,
            tracked=1,
            missing_suspected=True,
            unmatched=["pliers"],
            unexplained_clusters=1,
            recovered=1,
            video_requests=2,
        ),
    )
    report, html = _report(tmp_path, instances)

    assert report["discovery"]["recovered"] == 1
    assert report["objects"][0]["found_by"] == "noun"
    assert "1 tracked of 2 listed" in html
    assert "unmatched inventory: pliers" in html
    assert "1 unexplained point clusters" in html
    assert "found by gap fill: noun" in html
