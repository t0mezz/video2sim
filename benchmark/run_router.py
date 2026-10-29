"""Run the scene router on finished runs and compare its object list with tracks (P2.5).

    uv run python benchmark/run_router.py s01_b s17_b --out benchmark/results/p2_5_router.json

One hosted VLM request per run (Baseten, same model as discovery method A), cached in
``runs/<id>/segmentation/router_cache`` so a rerun makes no call. Prints and writes, per run, the
chosen surface and prompt, how many objects the model listed, how many tracks exist in
``runs/<id>/segmentation/tracks`` (the ones from the current default prompt), how many rigid
ground-truth objects the scene has, and whether the count check suspects a missing object.
Nothing is written into the run folders except the cache. The key comes from the environment or
``.env`` and is never printed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from video2sim.evaluation.scenes import load_scene
from video2sim.runner import OUTPUT_FILE
from video2sim.schemas import WorldModel
from video2sim.stages.discovery_vlm_sam import ResponseCache, read_secret
from video2sim.stages.material_pass import BasetenMaterialClient
from video2sim.stages.scene_router import RouterConfig, compare_counts, route_scene
from video2sim.stages.tracking import load_tracks

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_ids", nargs="+")
    parser.add_argument("--runs", type=Path, default=ROOT / "runs")
    parser.add_argument("--out", type=Path, default=ROOT / "benchmark/results/p2_5_router.json")
    args = parser.parse_args()
    cfg: dict[str, Any] = yaml.safe_load(
        (ROOT / "configs/default.yaml").read_text(encoding="utf-8")
    )
    rcfg = RouterConfig.from_config(cfg)
    client = BasetenMaterialClient(read_secret("BASETEN_API_KEY"), rcfg.timeout_s, rcfg.retries)

    rows: list[dict[str, Any]] = []
    for run_id in args.run_ids:
        scene_id = run_id.rsplit("_", 1)[0]
        run_dir = args.runs / run_id
        seg = run_dir / "segmentation"
        world = WorldModel.model_validate_json(
            (run_dir / "reconstruction" / OUTPUT_FILE).read_text(encoding="utf-8")
        )
        route = route_scene(world, run_dir, client, ResponseCache(seg / "router_cache"), rcfg)
        tracks = len(load_tracks(seg / "tracks")) if (seg / "tracks").is_dir() else 0
        gt = load_scene(ROOT / "benchmark/scenes" / scene_id / "ground_truth.json")
        check = compare_counts(route, tracks)
        row = {
            "run": run_id,
            "surface": route.surface,
            "prompt": route.prompt,
            "fell_back": route.fell_back,
            "reason": route.reason,
            "objects": list(route.objects),
            "listed": check.listed,
            "tracks": tracks,
            "gt_rigid": sum(1 for o in gt.objects if not o.deformable),
            "gt_deformable": sum(1 for o in gt.objects if o.deformable),
            "missing_suspected": check.missing_suspected,
        }
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "objects"}), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
