"""Per-object failure flags of finished runs, next to what the ground truth expects (P2.5).

    uv run python benchmark/report_flags.py s17_b s18_b --out benchmark/results/p2_5_flags.json

Combines ``stages/shape_flags.py`` (small, thin, occluded from the hull and masks) with the
material pass written by ``run_material_pass.py --vlm`` to ``segmentation/material_pass.json``
(reflective, non-rigid), and makes no hosted call. Tracks are not matched to ground-truth objects
(that needs the board alignment), so the comparison is per scene: the ground truth gives counts
of rigid objects that are small (largest side at most ``--gt-small-m``), thin (smallest side at
most ``--gt-thin-m`` and at most ``--gt-thin-ratio`` of the largest) and deformable, to set
against the number of tracks carrying each flag. The listing names the model's category for each
flagged track so the flags can be judged by eye.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from video2sim.evaluation.scenes import load_scene
from video2sim.runner import OUTPUT_FILE
from video2sim.schemas import Flag, WorldModel
from video2sim.stages.shape_flags import ShapeConfig, judge_all, shape_signals
from video2sim.stages.tracking import load_tracks

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_ids", nargs="+")
    parser.add_argument("--runs", type=Path, default=ROOT / "runs")
    parser.add_argument("--out", type=Path, default=ROOT / "benchmark/results/p2_5_flags.json")
    parser.add_argument("--gt-small-m", type=float, default=0.045)
    parser.add_argument("--gt-thin-m", type=float, default=0.012)
    parser.add_argument("--gt-thin-ratio", type=float, default=0.2)
    args = parser.parse_args()
    cfg = ShapeConfig.from_config(
        yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    )
    scenes: dict[str, Any] = {}
    for run_id in args.run_ids:
        run_dir = args.runs / run_id
        seg = run_dir / "segmentation"
        world = WorldModel.model_validate_json(
            (run_dir / "reconstruction" / OUTPUT_FILE).read_text(encoding="utf-8")
        )
        tracks = load_tracks(seg / "tracks") if (seg / "tracks").is_dir() else []
        shape = judge_all(shape_signals(tracks, world, run_dir, seg / "clusters", cfg), cfg)
        mat_path = seg / "material_pass.json"
        material = (
            {int(r["track"]): r for r in json.loads(mat_path.read_text())}
            if mat_path.is_file()
            else {}
        )
        gt = load_scene(ROOT / "benchmark/scenes" / run_id.rsplit("_", 1)[0] / "ground_truth.json")
        sizes = [sorted(o.size_m) for o in gt.objects if not o.deformable]
        expected = {
            "rigid": len(sizes),
            "deformable": sum(1 for o in gt.objects if o.deformable),
            "small": sum(1 for s in sizes if s[2] <= args.gt_small_m),
            "thin": sum(
                1 for s in sizes if s[0] <= args.gt_thin_m and s[0] / s[2] <= args.gt_thin_ratio
            ),
        }
        rows = []
        for t in tracks:
            m = material.get(t.track_id)
            flags = [f.value for f in shape[t.track_id].flags]
            if m and m["flag"]:
                flags.append(Flag.TRANSPARENT_OR_REFLECTIVE.value)
            if m and m.get("non_rigid"):
                flags.append(Flag.NON_RIGID.value)
            rows.append(
                {
                    "track": t.track_id,
                    "category": (m["answer"] or {}).get("category") if m else None,
                    "finish": (m["answer"] or {}).get("finish") if m else None,
                    "rigidity": (m["answer"] or {}).get("rigidity") if m else None,
                    "flags": flags,
                    "reasons": list(shape[t.track_id].reasons)
                    + ([m["reason"]] if m and m["flag"] else [])
                    + ([m["non_rigid"]] if m and m.get("non_rigid") else []),
                }
            )
        got = {
            name: sum(1 for r in rows if name in r["flags"])
            for name in (
                "small", "thin", "occluded", "transparent_or_reflective", "non_rigid"
            )
        }  # fmt: skip
        scenes[run_id] = {
            "tracks": len(tracks),
            "expected": expected,
            "flagged": got,
            "objects": rows,
        }
        print(run_id, json.dumps({"tracks": len(tracks), "expected": expected, "flagged": got}))
        for r in rows:
            if r["flags"]:
                print("   ", r["track"], r["category"], r["finish"], r["rigidity"], r["flags"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(scenes, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
