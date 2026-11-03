"""Run the export stage (P5.1 to P5.3) on runs that already have ``SimAssets``.

    uv run python benchmark/run_export.py --out benchmark/results/p5_export.json

Takes the finished stages as they are, the way ``run_physics.py`` does, and writes ``export/`` in
each run of the rigid scenes of a split, then sums up the QA reports. It makes no hosted call.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import yaml
from run_physics import ROOT
from run_shape import accept_finished_stages

from video2sim.evaluation.scenes import discover_scenes
from video2sim.pipeline import build_runner
from video2sim.runner import OUTPUT_FILE, RunRejected, StageFailed


def run_one(job: tuple[Path, str]) -> dict[str, Any]:
    runs, run_id = job
    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    targets = yaml.safe_load((ROOT / "configs/targets.yaml").read_text(encoding="utf-8"))
    runner = build_runner(runs, cfg, targets, ROOT)
    scene, _, which = run_id.rpartition("_")
    video = ROOT / "benchmark/scenes" / scene / f"video_{which}.mp4"
    accept_finished_stages(runner, runs / run_id, video, until="export")
    try:
        manifest = runner.run(video, run_id=run_id)
    except (StageFailed, RunRejected) as e:
        if isinstance(e, RunRejected):
            return {"run": run_id, "failed": e.reasons}
        return {"run": run_id, "failed": str(e.manifest.stages["export"].message)}
    report = json.loads((runs / run_id / "export/qa_report.json").read_text())
    return {
        "run": run_id,
        "seconds": manifest.stages["export"].seconds,
        **{k: v for k, v in report.items() if k not in ("objects", "views", "validated_in")},
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
    run_ids = []
    for scene in scenes:
        if scene.has_deformable or (args.scene and scene.scene_id not in args.scene):
            continue
        run_ids += [
            r.name
            for r in sorted(args.runs.glob(f"{scene.scene_id}_*"))
            if (r / "physics" / OUTPUT_FILE).exists()
        ][:1]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(run_one, [(args.runs, r) for r in run_ids]))
    done = [r for r in rows if "failed" not in r]
    summary = {
        "simulator": "mujoco (D-027)",
        "runs": len(rows),
        "failed": [r for r in rows if "failed" in r],
        "objects_exported": sum(r["objects_exported"] for r in done),
        "objects_skipped": sum(r["objects_skipped"] for r in done),
        "urdf_settling_agrees": sum(r["urdf_check"]["agrees"] for r in done),
        "urdf_differences": {
            r["run"]: r["urdf_check"]["differences"] for r in done if r["urdf_check"]["differences"]
        },
        "stable_every_case": sum(r["stable_every_case"] for r in done),
    }
    args.out.write_text(json.dumps({"summary": summary, "scenes": rows}, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
