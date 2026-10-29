"""Lift the tracks of finished runs to point clusters, two ways, and score their boxes (P2.3).

    uv run python benchmark/run_lift.py s01_b s02_b --score --out benchmark/results/p2_3.json

Needs ``runs/<id>/segmentation/tracks/`` from ``benchmark/run_tracking.py``; makes no hosted
call. Two clusters per track, kept separately so Phase 3 can compare them as registration
targets (see ``stages/lift.py``):

* ``hull``: a visual hull by voting, in ``segmentation/clusters/track_NNN.npy``;
* ``points``: the sparse SfM points the masks agree on, minus the outermost strays, in
  ``segmentation/points/track_NNN.npy``;
* ``fitted``: a box, cylinder or sphere fitted to the masks from those points
  (``stages/primitive_fit.py``), in ``segmentation/fitted/track_NNN.npy``, with ``fit.json``.

Both are in the world frame, metres. ``segmentation/lift.json`` lists what each kept and
dropped. ``--score`` moves the clusters to the ground-truth frame (as in ``run_discovery.py``)
and reports the bounding-box size error per object: the Phase 2 exit criterion is a median of at
most ``gates.segmentation.cluster_bbox_error_max`` on the easy tier.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray
from run_discovery import CLEAN_ALIGNED, board_layout

from video2sim.evaluation.alignment import BoardAlignment, align_clean, align_run
from video2sim.evaluation.metrics import evaluate_scene
from video2sim.evaluation.scenes import GtScene, Prediction, PredObject, load_scene
from video2sim.runner import OUTPUT_FILE
from video2sim.schemas import WorldModel
from video2sim.stages.lift import LiftReport, lift_points, lift_tracks, load_points
from video2sim.stages.tracking import load_tracks

ROOT = Path(__file__).resolve().parent.parent
METHODS = ("hull", "points", "fitted")
FOLDERS = {"hull": "clusters", "points": "points", "fitted": "fitted"}


def save_clusters(directory: Path, clusters: Mapping[int, NDArray[np.float64]]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for old in directory.glob("track_*.npy"):
        old.unlink()
    for track_id, points in clusters.items():
        np.save(directory / f"track_{track_id:03d}.npy", points.astype(np.float32))


def report_json(report: LiftReport) -> dict[str, Any]:
    return {
        "views_used": report.views_used,
        "counts": {str(k): v for k, v in report.voxels.items()},
        "dropped": {str(k): v for k, v in report.dropped.items()},
    }


def score_clusters(
    gt: GtScene, alignment: BoardAlignment, clusters: Mapping[int, NDArray[np.float64]]
) -> dict[str, Any]:
    t = alignment.T_board_from_world
    objects = [
        PredObject(f"track_{i:03d}", p @ t[:3, :3].T + t[:3, 3]) for i, p in clusters.items()
    ]
    result = evaluate_scene(gt, Prediction(objects))
    return {
        "gt_objects": result.n_evaluable,
        "matched": result.n_matched,
        "false_positives": result.n_false_positive,
        "size_errors": {
            o.object_id: None if o.size_error is None else round(o.size_error, 3)
            for o in result.objects
        },
        "position_errors_mm": {
            o.object_id: round(1000 * o.position_error_m)
            for o in result.objects
            if o.position_error_m is not None
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_ids", nargs="+")
    parser.add_argument("--runs", type=Path, default=ROOT / "runs")
    parser.add_argument("--out", type=Path, default=ROOT / "benchmark/results/p2_3_lift.json")
    parser.add_argument("--score", action="store_true", help="score boxes (needs alignment)")
    args = parser.parse_args()

    from video2sim.stages.lift import LiftConfig
    from video2sim.stages.primitive_fit import PrimitiveConfig, fit_clusters
    from video2sim.stages.segmentation_stage import fit_report

    cfg: dict[str, Any] = yaml.safe_load(
        (ROOT / "configs/default.yaml").read_text(encoding="utf-8")
    )
    lcfg = LiftConfig.from_config(cfg)
    pcfg = PrimitiveConfig.from_config(cfg)
    gate = float(
        yaml.safe_load((ROOT / "configs/targets.yaml").read_text(encoding="utf-8"))["gates"][
            "segmentation"
        ]["cluster_bbox_error_max"]
    )
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
        tracks = load_tracks(seg_dir / "tracks")
        started = time.perf_counter()
        hull, hull_report = lift_tracks(tracks, world, lcfg)
        hull_s = time.perf_counter() - started
        started = time.perf_counter()
        cloud = load_points(run_dir / world.point_cloud_path)
        points, points_report = lift_points(tracks, world, cloud, lcfg)
        points_s = time.perf_counter() - started
        started = time.perf_counter()
        fits = fit_clusters(tracks, points, world, lcfg, pcfg)
        fitted_s = time.perf_counter() - started
        (seg_dir / "fit.json").write_text(json.dumps(fit_report(fits), indent=2) + "\n")
        found = {"hull": hull, "points": points, "fitted": {i: f.cluster for i, f in fits.items()}}
        for method in METHODS:
            save_clusters(seg_dir / FOLDERS[method], found[method])
        (seg_dir / "lift.json").write_text(
            json.dumps(
                {"hull": report_json(hull_report), "points": report_json(points_report)}, indent=2
            )
            + "\n"
        )
        row: dict[str, Any] = {
            "run": run_id,
            "scene": scene_id,
            "tracks": len(tracks),
            "hull": {"seconds": round(hull_s, 1), "clusters": len(hull)},
            "points": {"seconds": round(points_s, 1), "clusters": len(points)},
            "fitted": {
                "seconds": round(fitted_s, 1),
                "clusters": len(fits),
                "sources": {str(i): f.source for i, f in fits.items()},
            },
        }
        row["hull"]["dropped"] = report_json(hull_report)["dropped"]
        row["points"]["dropped"] = report_json(points_report)["dropped"]
        row["fitted"]["dropped"] = row["points"]["dropped"]
        if args.score:
            gt = load_scene(ROOT / "benchmark/scenes" / scene_id / "ground_truth.json")
            alignment = None
            if gt.board == "grid":
                alignment = align_run(world, run_dir, layout, family)
            elif scene_id in CLEAN_ALIGNED:
                alignment = align_clean(world, run_dir, family)
            row["tier"] = gt.tier
            if alignment is None:
                row["unscored"] = "clean board: origin of the ground truth is not established"
            else:
                for method in METHODS:
                    row[method].update(score_clusters(gt, alignment, found[method]))
        rows.append(row)
        print(json.dumps(row), flush=True)

    scored = [r for r in rows if "matched" in r["hull"]]
    summary: dict[str, Any] = {
        "scenes_scored": [r["scene"] for r in scored],
        "gate": gate,
    }
    for method in METHODS:
        easy = [
            e
            for r in scored
            if r["tier"] == "easy"
            for e in r[method]["size_errors"].values()
            if e is not None
        ]
        every = [e for r in scored for e in r[method]["size_errors"].values() if e is not None]
        summary[method] = {
            "objects": sum(r[method]["gt_objects"] for r in scored),
            "matched": sum(r[method]["matched"] for r in scored),
            "size_error_median_easy": statistics.median(easy) if easy else None,
            "size_error_median_all": statistics.median(every) if every else None,
            "within_gate": sum(e <= gate for e in every),
            "measured": len(every),
        }
    print(json.dumps(summary, indent=2))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "scenes": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
