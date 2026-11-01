"""Stage 6: export and validation (P5.1 to P5.3).

Input:  ``SimAssets``, ``WorldModel``, ``RegisteredObjects``, ``Instances``, and the physics
        stage's ``report.json`` and ``settling.json``.
Output: ``SceneExport``, a scene package: USD scene (Z-up, meters, kilograms), one URDF per
        object, and a QA report with per-object status, confidence, fallbacks used and stability
        results.

Export is blocked for any asset whose validators did not all pass: it stays out of the scene and
the QA report says why. A USD or URDF file that does not read back as ``SimAssets`` fails the
stage, because it would load fine and behave wrongly. Settling the URDF files gives the poses of
the QA renders, and a different stable result from the physics stage is reported, not blocking
(D-023).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import trimesh

from video2sim.runner import StageContext
from video2sim.schemas import (
    Instances,
    RegisteredObjects,
    SceneExport,
    SimAsset,
    SimAssets,
    SkippedAsset,
    StageName,
    WorldModel,
)
from video2sim.stages.qa_report import build_report, write_report
from video2sim.stages.scene_render import RenderObject, render, side_by_side
from video2sim.stages.settling import SceneResult, SettlingConfig
from video2sim.stages.urdf_writer import (
    check_model,
    settle_urdfs,
    urdf_scene_model,
    write_urdf,
)
from video2sim.stages.usd_writer import check_usd, write_usd


class ExportCheckFailed(Exception):
    """A written file does not match ``SimAssets``."""


def export_stage_config(cfg: Mapping[str, Any], targets: Mapping[str, Any]) -> dict[str, Any]:
    """The export stage's config section, with the settling numbers it shares with physics."""
    t = targets["targets"]
    return {
        "export": cfg["export"],
        "settling": cfg["physics"]["settling"],
        "stability": {k: t[k] for k in ("stability_window_s", "stability_max_drift_m")},
    }


def _urdf_check(
    result: SceneResult | None, settling: dict[str, dict[str, Any]], problems: list[str]
) -> dict[str, Any]:
    """How the URDF settling compares with the physics stage's nominal case."""
    if result is None:
        return {"agrees": True, "problems": problems, "differences": []}
    nominal = {b["object_id"]: b for b in settling.get("nominal", {}).get("bodies", [])}
    differences = [
        body.object_id
        for body in result.bodies
        if body.object_id not in nominal or body.stable != nominal[body.object_id]["stable"]
    ]
    return {
        "agrees": not differences,
        "problems": problems,
        "differences": differences,
        "largest_drift_m": max((b.max_drift_m for b in result.bodies), default=0.0),
    }


def _write_views(
    out_dir: Path,
    run_dir: Path,
    world: WorldModel,
    assets: list[SimAsset],
    poses: dict[str, Any],
    count: int,
) -> list[str]:
    """Side-by-side images from ``count`` frames evenly spaced over the video."""
    objects = [
        RenderObject(
            cast(
                trimesh.Trimesh,
                trimesh.load(run_dir / a.visual_mesh_path, force="mesh", process=False),
            ),
            poses[a.object_id],
        )
        for a in assets
    ]
    folder = out_dir / "qa"
    folder.mkdir(exist_ok=True)
    views = []
    picks = np.linspace(0, len(world.frames) - 1, min(count, len(world.frames))).round()
    for k in picks.astype(int):
        frame = world.frames[int(k)]
        image = cv2.imread(str(run_dir / frame.image_path))
        if image is None:
            raise FileNotFoundError(run_dir / frame.image_path)
        rendered = render(objects, frame.T_world_from_camera.to_numpy(), world.intrinsics)
        name = f"view_{frame.frame_index:05d}.jpg"
        cv2.imwrite(str(folder / name), side_by_side(image.astype(np.uint8), rendered))
        views.append(f"qa/{name}")
    return views


class ExportStage:
    """Runner stage. Its config comes from ``export_stage_config``."""

    name: StageName = "export"
    output_type = SceneExport
    dependencies: tuple[StageName, ...] = ("reconstruction", "segmentation", "shape", "physics")
    extra_inputs: tuple[Path, ...] = ()

    def run(self, ctx: StageContext) -> SceneExport:
        world = ctx.load("reconstruction", WorldModel)
        instances = ctx.load("segmentation", Instances)
        registered = ctx.load("shape", RegisteredObjects)
        sim_assets = ctx.load("physics", SimAssets)
        export = ctx.config["export"]
        settling_cfg = SettlingConfig.from_config(
            {"physics": {"settling": ctx.config["settling"]}}, {"targets": ctx.config["stability"]}
        )
        physics_dir = ctx.run_dir / "physics"
        physics = json.loads((physics_dir / "report.json").read_text(encoding="utf-8"))
        settling = json.loads((physics_dir / "settling.json").read_text(encoding="utf-8"))

        assets = [a for a in sim_assets.assets if a.is_valid]
        skipped = [
            SkippedAsset(
                object_id=a.object_id,
                reasons=[k for k, v in a.checks.items() if not v] or ["no checks recorded"],
            )
            for a in sim_assets.assets
            if not a.is_valid
        ]

        tolerance = float(export["check_tolerance"])
        tolerance_m = float(export["check_tolerance_m"])
        usd_path = ctx.out_dir / "scene.usd"
        write_usd(
            usd_path,
            assets,
            ctx.run_dir,
            world.table_plane,
            settling_cfg.gravity_m_s2,
            float(export["table_size_m"]),
        )
        problems = check_usd(usd_path, assets, settling_cfg.gravity_m_s2, tolerance, tolerance_m)

        urdfs = {a.object_id: write_urdf(ctx.out_dir / "urdf", a, ctx.run_dir) for a in assets}
        result = None
        poses = {a.object_id: a.T_world_from_object.to_numpy() for a in assets}
        if assets:
            model = urdf_scene_model(urdfs, assets, world.table_plane, settling_cfg)
            problems += check_model(model, assets, ctx.run_dir, tolerance, tolerance_m)
            if not problems:
                result = settle_urdfs(model, assets, world.table_plane, settling_cfg)
                poses = result.final_poses
        if problems:
            raise ExportCheckFailed("; ".join(problems))

        views = _write_views(
            ctx.out_dir, ctx.run_dir, world, assets, poses, int(export["qa_views"])
        )
        report = build_report(
            ctx.run_dir.name,
            world,
            instances,
            registered,
            sim_assets,
            physics,
            settling,
            skipped,
            _urdf_check(result, settling, problems),
            views,
        )
        write_report(report, ctx.out_dir)
        return SceneExport(
            usd_path="export/scene.usd",
            urdf_paths={k: f"export/urdf/{v.name}" for k, v in urdfs.items()},
            qa_report_path="export/qa_report.html",
            skipped=skipped,
        )


__all__ = ["ExportCheckFailed", "ExportStage", "export_stage_config"]
