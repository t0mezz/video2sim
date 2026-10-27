"""Run the capture and reconstruction stages through the Runner on one video (P1.5).

    uv run python benchmark/run_stages.py benchmark/scenes/s01/video_b.mp4 --run-id s01_b

Writes ``runs/<run-id>/`` (ignored by git) and prints the manifest status per stage and the
``WorldModel`` summary. Config comes from ``configs/default.yaml`` and the gate numbers from
``configs/targets.yaml``, both read only. A rejected video prints the reasons and exits 1.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from video2sim.runner import OUTPUT_FILE, Runner, StageFailed
from video2sim.schemas import WorldModel
from video2sim.stages.capture import CaptureStage
from video2sim.stages.reconstruction import (
    ReconstructionStage,
    reconstruction_stage_config,
)

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs", type=Path, default=ROOT / "runs")
    parser.add_argument("--force", action="store_true", help="ignore the stage cache")
    args = parser.parse_args()

    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    targets = yaml.safe_load((ROOT / "configs/targets.yaml").read_text(encoding="utf-8"))
    runner = Runner(
        args.runs,
        [CaptureStage(), ReconstructionStage()],
        {"capture": cfg["capture"], "reconstruction": reconstruction_stage_config(cfg, targets)},
    )
    try:
        manifest = runner.run(
            args.video, run_id=args.run_id, force_from="capture" if args.force else None
        )
    except StageFailed as e:
        for name, rec in e.manifest.stages.items():
            if rec.status != "pending":
                print(f"{name}: {rec.status} {rec.message or ''}")
        sys.exit(1)

    for name, rec in manifest.stages.items():
        if rec.status != "pending":
            print(f"{name}: {rec.status} in {(rec.seconds or 0):.1f}s")
    out = args.runs / args.run_id / "reconstruction" / OUTPUT_FILE
    wm = WorldModel.model_validate_json(out.read_text(encoding="utf-8"))
    print(json.dumps(wm.model_dump(exclude={"frames"}), indent=2))
    print(f"frames: {len(wm.frames)}")


if __name__ == "__main__":
    main()
