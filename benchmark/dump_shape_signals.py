"""Print size, thinness and occlusion numbers per track of finished runs (P2.5).

    uv run python benchmark/dump_shape_signals.py s01_b s17_b

Reads ``segmentation/clusters`` (visual hull, P2.3) and ``segmentation/tracks`` and calls no
hosted API. Used to choose the ``segmentation.shape_flags`` thresholds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from video2sim.runner import OUTPUT_FILE
from video2sim.schemas import WorldModel
from video2sim.stages.shape_flags import ShapeConfig, shape_signals
from video2sim.stages.tracking import load_tracks

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_ids", nargs="+")
    parser.add_argument("--runs", type=Path, default=ROOT / "runs")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    cfg = ShapeConfig.from_config(
        yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    )
    rows = []
    for run_id in args.run_ids:
        run_dir = args.runs / run_id
        seg = run_dir / "segmentation"
        if not (seg / "tracks").is_dir():
            continue
        world = WorldModel.model_validate_json(
            (run_dir / "reconstruction" / OUTPUT_FILE).read_text(encoding="utf-8")
        )
        tracks = load_tracks(seg / "tracks")
        for sig in shape_signals(tracks, world, run_dir, seg / "clusters", cfg):
            row = {"run": run_id, **sig.as_dict()}
            rows.append(row)
            print(json.dumps(row), flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
