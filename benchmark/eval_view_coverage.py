"""Offline view-coverage experiment (S12.9, task 9.2): is anything hidden from all router views?

    uv run python benchmark/eval_view_coverage.py --out benchmark/results/s12_9_coverage.json

On the dev scenes with finished runs, using aligned ground-truth boxes as in
``evaluation/segmentation.py``: per object and frame, the visible fraction of its projected
box after removing pixels covered by nearer ground-truth boxes. Depth order is by box center
distance to the camera; the projection is the full 3D box hull, so self-occlusion is ignored
and boxes behind the camera project empty. This is an approximation, stated here and in the
result file.

Per scene and view policy (``even-3`` as today, ``even-6``, ``coverage-6`` once 9.3 exists):
the share of objects whose visible fraction is at least 0.3 in at least one chosen view, and
the minimum over objects of the number of such views.

Decision rule, also written into the result file: if ``even-3`` already shows every
ground-truth object in at least one view on every scored dev scene, multi-view routing (9.3,
9.4) is not justified by occlusion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from video2sim.evaluation.alignment import CLEAN_ALIGNED, align_clean, align_run, grid_board_layout
from video2sim.evaluation.scenes import GtObject, load_scene
from video2sim.evaluation.segmentation import project_box_mask
from video2sim.runner import OUTPUT_FILE
from video2sim.schemas import WorldModel
from video2sim.stages.segmentation import sample_frame_indices

ROOT = Path(__file__).resolve().parent.parent
VISIBLE_FRAC = 0.3  # an object counts as seen when this share of its box is uncovered
DEV_RUNS = [
    "s01_b",
    "s02_b",
    "s03_b",
    "s04_b",
    "s05_b",
    "s06_b",
    "s11_b",
    "s12_b",
    "s13_b",
    "s14_b",
    "s17_b",
    "s18_b",
    "s24_b",
]  # dev scenes with finished runs, minus deformable s21/s22 (D-014)


def _box_center_distance(obj: GtObject, T_camera_from_world: NDArray[np.float64]) -> float:
    """Distance of the ground-truth box center along the camera forward axis."""
    center = np.array(obj.position_m, dtype=float)
    cam = T_camera_from_world @ np.append(center, 1.0)
    return float(cam[2])


def _visible_fractions(
    objects: list[GtObject], world: WorldModel, poses: dict[int, Any]
) -> dict[str, list[float]]:
    """Visible fraction of each object's projected box per clip position (nearer boxes win)."""
    order = [f.frame_index for f in world.frames]
    fracs: dict[str, list[float]] = {o.object_id: [] for o in objects}
    for i in order:
        pose = poses[i]
        T_cam = pose.inverse().to_numpy()
        masks = {o.object_id: project_box_mask(o, world.intrinsics, pose) for o in objects}
        near_first = sorted(objects, key=lambda o: _box_center_distance(o, T_cam))
        covered = np.zeros((world.intrinsics.height_px, world.intrinsics.width_px), dtype=bool)
        for o in near_first:
            mask = masks[o.object_id]
            total = int(mask.sum())
            fracs[o.object_id].append(
                float(np.logical_and(mask, ~covered).sum()) / total if total else 0.0
            )
            covered = np.logical_or(covered, mask)
    return fracs


def _policy_views(
    policy: str,
    order: list[int],
    run_dir: Path,
    world: WorldModel,
    cfg: dict[str, Any],
) -> list[int]:
    """Clip positions chosen by a view policy."""
    if policy.startswith("even-"):
        return [int(i) for i in sample_frame_indices(len(order), int(policy.removeprefix("even-")))]
    if policy == "coverage-6":
        from video2sim.stages.lift import load_points
        from video2sim.stages.view_select import ViewConfig, select_views

        vcfg = ViewConfig.from_config(cfg)
        cloud = load_points(run_dir / world.point_cloud_path)
        views = select_views(world, cloud, vcfg, run_dir)
        return [order.index(f) for f in views.frames]
    raise ValueError(f"unknown view policy {policy!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ROOT / "runs_instances")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["even-3", "even-6"],
        choices=["even-3", "even-6", "coverage-6"],
    )
    args = parser.parse_args()

    cfg: dict[str, Any] = yaml.safe_load(
        (ROOT / "configs/default.yaml").read_text(encoding="utf-8")
    )
    layout = grid_board_layout(float(cfg["capture"]["marker"]["tag_size_m"]))
    family = str(cfg["capture"]["marker"]["family"])

    scenes = []
    for run_id in DEV_RUNS:
        scene_id = run_id.rsplit("_", 1)[0]
        run_dir = args.runs / run_id
        row: dict[str, Any] = {"run": run_id, "scene": scene_id}
        gt = load_scene(ROOT / "benchmark/scenes" / scene_id / "ground_truth.json")
        world = WorldModel.model_validate_json(
            (run_dir / "reconstruction" / OUTPUT_FILE).read_text(encoding="utf-8")
        )
        alignment = None
        if gt.board == "grid":
            alignment = align_run(world, run_dir, layout, family)
        elif scene_id in CLEAN_ALIGNED:
            alignment = align_clean(world, run_dir, family)
        if alignment is None:
            row["unscored"] = "no board alignment"
            scenes.append(row)
            continue
        poses = {
            f.frame_index: alignment.pose_in_board(f.T_world_from_camera) for f in world.frames
        }
        rigid = [o for o in gt.objects if not o.deformable]
        fracs = _visible_fractions(rigid, world, poses)
        row["objects"] = len(rigid)
        row["policies"] = {}
        for policy in args.policies:
            order = [f.frame_index for f in world.frames]
            views = _policy_views(policy, order, run_dir, world, cfg)
            seen_counts = [
                sum(1 for v in views if fracs[o.object_id][v] >= VISIBLE_FRAC) for o in rigid
            ]
            row["policies"][policy] = {
                "views": views,
                "share_seen": sum(1 for c in seen_counts if c >= 1) / len(rigid),
                "min_views_seen": min(seen_counts),
                "worst_objects": sorted(
                    o.object_id
                    for o, c in zip(rigid, seen_counts, strict=True)
                    if c == min(seen_counts)
                ),
            }
        scenes.append(row)

    scored = [s for s in scenes if "policies" in s]
    even3_covers_all = all(s["policies"]["even-3"]["share_seen"] == 1.0 for s in scored)
    result = {
        "visible_frac": VISIBLE_FRAC,
        "depth_order": (
            "box center distance to the camera; full-box hull projection, "
            "self-occlusion ignored, boxes behind the camera project empty"
        ),
        "scenes": scenes,
        "decision": (
            "even-3 already shows every ground-truth object in at least one view on every "
            "scored dev scene: multi-view routing (9.3, 9.4) is NOT justified by occlusion; "
            "build only 9.6 and 9.8 with views.mode even"
            if even3_covers_all and "even-3" in args.policies
            else "some object is hidden from every even-3 view on some dev scene: "
            "multi-view routing stays justified (9.3 and 9.4 proceed)"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"scenes": len(scenes), "scored": len(scored)}, indent=1))
    for s in scenes:
        if "policies" in s:
            print(
                s["run"],
                {p: (v["share_seen"], v["min_views_seen"]) for p, v in s["policies"].items()},
            )
        else:
            print(s["run"], s["unscored"])


if __name__ == "__main__":
    main()
