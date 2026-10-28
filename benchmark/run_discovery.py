"""Run discovery method A on finished runs and score mask recall (P2.1).

    uv run python benchmark/run_discovery.py s01_b s02_b --out benchmark/results/p2_1_a.json

Scoring against ground truth is off by default (``--score``). The ground truth is in the board
frame and the world frame sits on the reference tag, so grid scenes are first aligned by fitting
the known board layout to the re-detected tags (``evaluation/alignment.py``). Clean-board scenes
have no layout and are listed as unscored.

Each run id is ``<scene>_<a|b>`` under ``runs/`` with a reconstruction output. Frames are sent to
Baseten (Muse Spark) and the Meta Model API (SAM 3.1); responses are cached in
``runs/<id>/segmentation/discovery_cache`` so a rerun makes no calls. Keys come from the
environment or ``.env`` and are never printed. Prints and writes recall per scene and overall.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from video2sim.evaluation.alignment import align_clean, align_run
from video2sim.evaluation.scenes import load_scene
from video2sim.evaluation.segmentation import mask_recall
from video2sim.runner import OUTPUT_FILE
from video2sim.schemas import WorldModel
from video2sim.stages.discovery_sam_generic import GenericSamBackend
from video2sim.stages.discovery_vlm_sam import (
    BasetenLister,
    MetaSamClient,
    ResponseCache,
    VlmSamBackend,
    read_secret,
)
from video2sim.stages.segmentation import DiscoveryBackend, DiscoveryConfig, discover_objects
from video2sim.stages.world_frame import BoardLayout

ROOT = Path(__file__).resolve().parent.parent
# Clean-board scenes whose ground truth lines up with the tag-0 corner frame (human note,
# 2026-10-28), checked by overlaying the ground-truth boxes on the frames. s05 does not line up
# under that rule or its three 90-degree rotations, so it stays unscored until its origin is known.
CLEAN_ALIGNED = ("s14", "s24")


def board_layout(tag_edge_m: float) -> BoardLayout:
    """The A1 grid board, as ``benchmark/eval_world_frame.py`` builds it."""
    spec = importlib.util.spec_from_file_location("make_board", ROOT / "benchmark/make_board.py")
    assert spec is not None and spec.loader is not None
    make_board = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(make_board)
    layout = make_board.build("A1", tag_edge_m * 1000.0, False)[1]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "layout.json"
        path.write_text(json.dumps(layout), encoding="utf-8")
        return BoardLayout.from_make_board_json(path, tag_edge_m=tag_edge_m)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_ids", nargs="+")
    parser.add_argument("--runs", type=Path, default=ROOT / "runs")
    parser.add_argument(
        "--method",
        choices=("vlm_sam", "sam_generic"),
        help="discovery method; default is segmentation.discovery.method in default.yaml",
    )
    parser.add_argument("--out", type=Path, help="default: benchmark/results/p2_1_<method>.json")
    parser.add_argument("--score", action="store_true", help="score recall (needs frame alignment)")
    args = parser.parse_args()

    cfg: dict[str, Any] = yaml.safe_load(
        (ROOT / "configs/default.yaml").read_text(encoding="utf-8")
    )
    dcfg = DiscoveryConfig.from_config(cfg)
    method = args.method or dcfg.method
    if method not in ("vlm_sam", "sam_generic"):
        raise SystemExit(f"method {method!r} has no backend yet")
    out_path = args.out or ROOT / f"benchmark/results/p2_1_{method}.json"
    # Masks of method A stay in ``discovery`` (the preview scripts read it); others get a folder.
    discovery_dir = "discovery" if method == "vlm_sam" else f"discovery_{method}"
    sam = MetaSamClient(read_secret("META_MODEL_API_KEY"))
    lister = BasetenLister(read_secret("B10_API_KEY")) if method == "vlm_sam" else None
    layout = board_layout(float(cfg["capture"]["marker"]["tag_size_m"]))

    rows: list[dict[str, Any]] = []
    for run_id in args.run_ids:
        scene_id = run_id.rsplit("_", 1)[0]
        run_dir = args.runs / run_id
        world = WorldModel.model_validate_json(
            (run_dir / "reconstruction" / OUTPUT_FILE).read_text(encoding="utf-8")
        )
        gt = load_scene(ROOT / "benchmark/scenes" / scene_id / "ground_truth.json")
        cache = ResponseCache(run_dir / "segmentation" / "discovery_cache")
        backend: DiscoveryBackend
        if lister is not None:
            backend = VlmSamBackend(lister, sam, cache, max_workers=dcfg.max_workers)
        else:
            backend = GenericSamBackend(
                sam, cache, dcfg.generic_prompt, max_workers=dcfg.max_workers
            )
        found = discover_objects(world, run_dir, backend, dcfg)
        score = None
        alignment = None
        family = str(cfg["capture"]["marker"]["family"])
        if args.score and gt.board == "grid":
            alignment = align_run(world, run_dir, layout, family)
        elif args.score and scene_id in CLEAN_ALIGNED:
            alignment = align_clean(world, run_dir, family)
        if alignment is not None:
            pose = {f.frame_index: f.T_world_from_camera for f in world.frames}
            score = mask_recall(
                gt.objects,
                world.intrinsics,
                [
                    (alignment.pose_in_board(pose[r.frame_index]), [m.mask for m in r.masks])
                    for r in found
                ],
            )
        out_dir = run_dir / "segmentation" / discovery_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        for r in found:
            np.savez_compressed(
                out_dir / f"{r.frame_index:05d}.npz",
                labels=np.array([m.label or "" for m in r.masks]),
                masks=np.array([m.mask for m in r.masks], dtype=bool),
            )
        row: dict[str, Any] = {
            "run": run_id,
            "scene": scene_id,
            "tier": gt.tier,
            "frames": [r.frame_index for r in found],
            "labels": sorted({m.label or "" for r in found for m in r.masks}),
            "masks_per_frame": [len(r.masks) for r in found],
            "gt_objects": sum(1 for o in gt.objects if not o.deformable),
        }
        if args.score and alignment is None:
            row["unscored"] = "clean board: origin of the ground truth is not established"
        if alignment is not None:
            row["alignment"] = {
                "scale": round(alignment.scale, 4),
                "rms_mm": round(1000 * alignment.rms_m, 2),
                "tags": list(alignment.tag_ids),
            }
        if score is not None:
            row.update(
                visible=list(score.visible),
                found=list(score.found),
                missed=[o for o in score.visible if o not in score.found],
                false_positives=score.false_positives,
                recall=score.recall,
            )
        rows.append(row)
        print(row, flush=True)

    if not args.score:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"scenes": rows}, indent=2) + "\n")
        return

    scored = [r for r in rows if "found" in r]
    n_vis = sum(len(r["visible"]) for r in scored)
    n_found = sum(len(r["found"]) for r in scored)
    by_tier: dict[str, float] = {}
    for tier in sorted({r["tier"] for r in scored}):
        sel = [r for r in scored if r["tier"] == tier]
        v, f = sum(len(r["visible"]) for r in sel), sum(len(r["found"]) for r in sel)
        by_tier[tier] = f / v if v else float("nan")
    summary = {
        "method": method,
        "n_sample_frames": dcfg.n_sample_frames,
        "objects_visible": n_vis,
        "objects_found": n_found,
        "recall_micro": n_found / n_vis if n_vis else None,
        "recall_macro": statistics.fmean(r["recall"] for r in scored if r["recall"] is not None),
        "recall_by_tier": by_tier,
        "false_positives": sum(r["false_positives"] for r in scored),
        "scenes_scored": [r["scene"] for r in scored],
        "scenes_unscored": [r["scene"] for r in rows if "found" not in r],
    }
    print(json.dumps(summary, indent=2))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "scenes": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
