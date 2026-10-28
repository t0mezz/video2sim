"""Track objects over the whole video and score identity (P2.2).

    uv run python benchmark/run_tracking.py s01_b s02_b --score --out benchmark/results/p2_2.json

One SAM 3.1 video request per run (billed per frame, at least 50), cached in
``runs/<id>/segmentation/tracking_cache`` so a rerun makes no call. Tracks are repaired with
3D consistency, and ``--score`` counts identity switches and fragments before and after the
repair against ground truth (grid scenes and the aligned clean-board scenes, as in
``run_discovery.py``). Tracks go to ``runs/<id>/segmentation/tracks/``. The key comes from the
environment or ``.env`` and is never printed.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import yaml
from run_discovery import CLEAN_ALIGNED, board_layout

from video2sim.evaluation.alignment import align_clean, align_run
from video2sim.evaluation.scenes import load_scene
from video2sim.evaluation.segmentation import score_tracks
from video2sim.runner import OUTPUT_FILE
from video2sim.schemas import WorldModel
from video2sim.stages.discovery_vlm_sam import ResponseCache, read_secret
from video2sim.stages.track_repair import RepairConfig, repair_tracks
from video2sim.stages.tracking import (
    MetaVideoSamClient,
    Track,
    TrackingConfig,
    save_tracks,
    track_objects,
)

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_ids", nargs="+")
    parser.add_argument("--runs", type=Path, default=ROOT / "runs")
    parser.add_argument("--out", type=Path, default=ROOT / "benchmark/results/p2_2_tracking.json")
    parser.add_argument("--score", action="store_true", help="score identity (needs alignment)")
    args = parser.parse_args()

    cfg: dict[str, Any] = yaml.safe_load(
        (ROOT / "configs/default.yaml").read_text(encoding="utf-8")
    )
    tcfg, rcfg = TrackingConfig.from_config(cfg), RepairConfig.from_config(cfg)
    client = MetaVideoSamClient(read_secret("META_MODEL_API_KEY"), tcfg.timeout_s)
    layout = board_layout(float(cfg["capture"]["marker"]["tag_size_m"]))
    family = str(cfg["capture"]["marker"]["family"])

    rows: list[dict[str, Any]] = []
    for run_id in args.run_ids:
        scene_id = run_id.rsplit("_", 1)[0]
        run_dir = args.runs / run_id
        seg_dir = run_dir / "segmentation"
        world = WorldModel.model_validate_json(
            (run_dir / "reconstruction" / OUTPUT_FILE).read_text(encoding="utf-8")
        )
        raw = track_objects(
            world,
            run_dir,
            client,
            ResponseCache(seg_dir / "tracking_cache"),
            tcfg,
            seg_dir / "tracking_work",
        )
        poses = {f.frame_index: f.T_world_from_camera for f in world.frames}
        repaired, report = repair_tracks(raw, poses, world.intrinsics, world.table_plane, rcfg)
        out_dir = seg_dir / "tracks"
        save_tracks(repaired, out_dir)
        row: dict[str, Any] = {
            "run": run_id,
            "scene": scene_id,
            "frames": len(world.frames),
            "tracks_raw": len(raw),
            "tracks_repaired": len(repaired),
            "splits": [[a, list(b)] for a, b in report.splits],
            "merges": [list(m) for m in report.merges],
        }
        if args.score:
            gt = load_scene(ROOT / "benchmark/scenes" / scene_id / "ground_truth.json")
            alignment = None
            if gt.board == "grid":
                alignment = align_run(world, run_dir, layout, family)
            elif scene_id in CLEAN_ALIGNED:
                alignment = align_clean(world, run_dir, family)
            if alignment is None:
                row["unscored"] = "clean board: origin of the ground truth is not established"
            else:
                board_poses = {i: alignment.pose_in_board(p) for i, p in poses.items()}
                for name, tracks in (("raw", raw), ("repaired", repaired)):
                    row[name] = score(gt.objects, world, board_poses, tracks)
        rows.append(row)
        print(json.dumps(row), flush=True)

    scored = [r for r in rows if "raw" in r]
    summary: dict[str, Any] = {"scenes_scored": [r["scene"] for r in scored]}
    for name in ("raw", "repaired"):
        summary[name] = {
            "switches_per_scene": statistics.fmean(r[name]["switches"] for r in scored)
            if scored
            else None,
            "fragmented_total": sum(r[name]["fragmented"] for r in scored),
            "unassigned_tracks_total": sum(r[name]["unassigned_tracks"] for r in scored),
        }
    print(json.dumps(summary, indent=2))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "scenes": rows}, indent=2) + "\n")


def score(objects: Any, world: WorldModel, board_poses: Any, tracks: list[Track]) -> dict[str, Any]:
    result = score_tracks(
        objects, world.intrinsics, board_poses, {t.track_id: t.masks for t in tracks}
    )
    return {
        "switches": result.switches,
        "fragmented": result.fragmented,
        "tracks_per_object": dict(result.tracks_per_object),
        "unassigned_tracks": result.unassigned_tracks,
    }


if __name__ == "__main__":
    main()
