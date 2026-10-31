"""Scripted pick-and-place in MuJoCo on the exported dev scenes (P5.4).

    uv run python benchmark/run_pick_place.py --out benchmark/results/p5_4_pick_place.json

Needs ``scripts/fetch_panda.sh`` and finished ``export/`` folders. Writes ``demo/pick_place.mp4``
beside them for the nominal case. It makes no hosted call.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml
from run_physics import ROOT

from video2sim.evaluation.scenes import discover_scenes
from video2sim.pick_place import DemoConfig, run_demo
from video2sim.runner import OUTPUT_FILE
from video2sim.schemas import SceneExport, SimAssets, WorldModel
from video2sim.stages.settling import SettlingConfig


def run_one(run_dir: Path) -> dict[str, Any]:
    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    targets = yaml.safe_load((ROOT / "configs/targets.yaml").read_text(encoding="utf-8"))
    export = SceneExport.model_validate_json((run_dir / "export" / OUTPUT_FILE).read_text())
    assets = [
        a
        for a in SimAssets.model_validate_json(
            (run_dir / "physics" / OUTPUT_FILE).read_text()
        ).assets
        if a.object_id in export.urdf_paths
    ]
    world = WorldModel.model_validate_json((run_dir / "reconstruction" / OUTPUT_FILE).read_text())
    plan, cases = run_demo(
        {k: run_dir / v for k, v in export.urdf_paths.items()},
        assets,
        world.table_plane,
        SettlingConfig.from_config(cfg, targets),
        run_dir,
        DemoConfig.from_config(cfg, targets, ROOT),
        run_dir / "demo" / "pick_place.mp4",
    )
    return {
        "run": run_dir.name,
        "object": plan.object_id if plan else None,
        "clearance_m": plan.clearance_m if plan else None,
        "success_every_case": bool(cases) and all(c.success for c in cases),
        "cases": [asdict(c) for c in cases],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ROOT / "runs_instances")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--scene", action="append", help="limit to these scene ids")
    parser.add_argument("--split", default="dev", help="dev, or test at a phase gate only")
    args = parser.parse_args()
    scenes, _ = discover_scenes(ROOT / "benchmark/scenes", (args.split,), None)
    folders = []
    for scene in scenes:
        if scene.has_deformable or (args.scene and scene.scene_id not in args.scene):
            continue
        folders += [
            r
            for r in sorted(args.runs.glob(f"{scene.scene_id}_*"))
            if (r / "export" / OUTPUT_FILE).exists()
        ][:1]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(run_one, folders))
    summary = {
        "simulator": "mujoco (D-027)",
        "scenes": len(rows),
        "scenes_with_a_graspable_object": sum(r["object"] is not None for r in rows),
        "success_every_case": [r["run"] for r in rows if r["success_every_case"]],
    }
    args.out.write_text(json.dumps({"summary": summary, "scenes": rows}, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    for row in rows:
        print(row["run"], row["object"], [
            (c["case"], round(c["lifted_m"], 3), round(c["place_error_m"], 3), c["success"])
            for c in row["cases"]
        ])  # fmt: skip


if __name__ == "__main__":
    main()
