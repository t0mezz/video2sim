"""Run the physics stage (P4.8) on runs that already have ``RegisteredObjects``.

    uv run python benchmark/run_physics.py --out benchmark/results/p4_8_physics.json

Takes the finished stages as they are, the way ``run_shape.py`` does, and writes ``physics/`` in
each run of the rigid dev scenes, then sums up the checks and ``physics/settling.json``. No hosted
call and no GPU. Run a scene at a time because CoACD takes up to 99 s per object.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import yaml
from run_shape import accept_finished_stages

from video2sim.evaluation.scenes import discover_scenes
from video2sim.runner import OUTPUT_FILE, Runner
from video2sim.schemas import SimAssets
from video2sim.stages.capture import CaptureStage
from video2sim.stages.physics_stage import CASES, PhysicsStage, physics_stage_config
from video2sim.stages.reconstruction import ReconstructionStage, reconstruction_stage_config
from video2sim.stages.segmentation_stage import SegmentationStage
from video2sim.stages.shape import ShapeStage

ROOT = Path(__file__).resolve().parent.parent


def run_one(job: tuple[Path, str]) -> tuple[str, str]:
    runs, run_id = job
    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    targets = yaml.safe_load((ROOT / "configs/targets.yaml").read_text(encoding="utf-8"))
    table = ROOT / cfg["physics"]["material_table"]
    runner = Runner(
        runs,
        [
            CaptureStage(),
            ReconstructionStage(),
            SegmentationStage(),
            ShapeStage(),
            PhysicsStage(table),
        ],
        {
            "capture": cfg["capture"],
            "reconstruction": reconstruction_stage_config(cfg, targets),
            "segmentation": cfg["segmentation"],
            "shape": cfg["shape"],
            "physics": physics_stage_config(cfg, targets),
        },
    )
    scene, _, which = run_id.rpartition("_")
    video = ROOT / "benchmark/scenes" / scene / f"video_{which}.mp4"
    accept_finished_stages(runner, runs / run_id, video, until="physics")
    manifest = runner.run(video, run_id=run_id)
    return run_id, manifest.stages["physics"].status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ROOT / "runs_instances")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--scene", action="append", help="limit to these scene ids")
    args = parser.parse_args()
    targets = yaml.safe_load((ROOT / "configs/targets.yaml").read_text(encoding="utf-8"))

    scenes, _ = discover_scenes(ROOT / "benchmark/scenes", ("dev",), None)
    run_ids = []
    for scene in scenes:
        if scene.has_deformable or (args.scene and scene.scene_id not in args.scene):
            continue
        run_ids += [
            r.name
            for r in sorted(args.runs.glob(f"{scene.scene_id}_*"))
            if (r / "shape" / OUTPUT_FILE).exists()
        ][:1]
    with ProcessPoolExecutor(args.workers) as pool:
        statuses = dict(pool.map(run_one, [(args.runs, r) for r in run_ids]))

    objects: list[dict[str, Any]] = []
    settling: dict[str, dict[str, Any]] = {}
    for run_id in run_ids:
        folder = args.runs / run_id / "physics"
        assets = SimAssets.model_validate_json((folder / OUTPUT_FILE).read_text())
        settling[run_id] = json.loads((folder / "settling.json").read_text())
        report = json.loads((folder / "report.json").read_text())
        for a in assets.assets:
            p = a.properties
            objects.append(
                {
                    "run": run_id,
                    "object": a.object_id,
                    "valid": a.is_valid,
                    "failed_checks": [k for k, v in a.checks.items() if not v],
                    "flags": [f.value for f in a.flags],
                    "kind": report[a.object_id]["collision_kind"],
                    "parts": report[a.object_id]["collision_parts"],
                    "mass_kg": p.mass_kg.model_dump(),
                    "friction": p.friction_coefficient.model_dump(),
                    "restitution": p.restitution.model_dump(),
                }
            )
    fraction = {
        c: sum(settling[r][c]["stable"] for r in run_ids) / max(len(run_ids), 1) for c in CASES
    }
    summary = {
        "simulator": "mujoco (D-021)",
        "runs": len(run_ids),
        "statuses": statuses,
        "objects": len(objects),
        "objects_valid": sum(o["valid"] for o in objects),
        "settling_gate": targets["targets"]["stability_pass_fraction_min"],
        "settling_pass_fraction": fraction,
        "stable_in_every_case": [
            r for r in run_ids if all(settling[r][c]["stable"] for c in CASES)
        ],
    }
    args.out.write_text(
        json.dumps({"summary": summary, "objects": objects, "settling": settling}, indent=1) + "\n"
    )
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
