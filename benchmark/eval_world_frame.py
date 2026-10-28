"""Check metric scale and the world frame on benchmark scenes (P1.2, P1.3).

    uv run python benchmark/eval_world_frame.py s01/video_b s02/video_a --out results.json

For each ``scene/video``: extract frames, run pycolmap, detect tags, then ``solve_world_frame``
(scale from the tag edge only). The printed grid board is used as an independent check, never as
input: its known tag spacing gives a second scale, the pairwise tag distances in the world frame
are compared with the layout, and the table normal is compared with the normal of the board
fitted to the tag corners. Results are compared with ``gates.reconstruction`` in
``configs/targets.yaml``, which is read and never changed.

The board comparison runs only for scenes whose ``ground_truth.json`` says ``board: grid``; the
board is then assumed to be A1 with the tag edge from ``capture.marker.tag_size_m``. For
``board: clean`` scenes the tag placement is not known, so the board fields and the gates that
need them (marker distance, up axis) are left out of the row. Work files go to ``--work``
(default ``data/world_frame_work``, ignored by git). Use dev-split scenes while tuning.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from video2sim.evaluation.scenes import load_scene
from video2sim.stages.reconstruction import (
    ReconstructionLimits,
    SfmOptions,
    WorldFrameConfig,
    camera_coverage,
    check_reconstruction,
    extract_frames,
    run_sfm,
    solve_world_frame,
    tag_views,
)
from video2sim.stages.world_frame import BoardLayout, angle_between_deg, fit_board

log = logging.getLogger("eval_world_frame")
ROOT = Path(__file__).resolve().parent.parent


def _board_layout(tag_edge_m: float) -> BoardLayout:
    spec = importlib.util.spec_from_file_location("make_board", ROOT / "benchmark/make_board.py")
    assert spec is not None and spec.loader is not None
    make_board = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(make_board)
    layout = make_board.build("A1", tag_edge_m * 1000.0, False)[1]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "layout.json"
        path.write_text(json.dumps(layout), encoding="utf-8")
        return BoardLayout.from_make_board_json(path, tag_edge_m=tag_edge_m)


def evaluate(
    scene: str,
    video: str,
    work: Path,
    cfg: dict[str, Any],
    layout: BoardLayout,
    limits: ReconstructionLimits,
    options: SfmOptions | None = None,
    fps: float = 3.0,
    timings: dict[str, float] | None = None,
) -> dict[str, Any]:
    board_kind = load_scene(ROOT / f"benchmark/scenes/{scene}/ground_truth.json").board
    if board_kind not in ("grid", "clean"):
        raise ValueError(
            f"{scene}: ground_truth.json has board {board_kind!r}, expected grid or clean"
        )
    wf = WorldFrameConfig.from_config(cfg)
    root = work / f"{scene}_{video.replace('/', '_')}"
    frames_dir = root / "frames"
    if not frames_dir.exists():
        extract_frames(ROOT / f"benchmark/scenes/{scene}/{video}.mp4", frames_dir, fps)
    sfm = run_sfm(frames_dir, root / "sfm", timings, options)
    views = tag_views(sfm, frames_dir, cfg["capture"]["marker"]["family"])
    try:
        res = solve_world_frame(views, sfm.points_xyz, wf)
    except ValueError as e:  # no marker in 3D: the gate says so, there is nothing to compare
        return {
            "scene": scene,
            "video": video,
            "frames": sfm.n_input_frames,
            "registered": len(sfm.images),
            "registered_fraction": round(sfm.registered_fraction, 4),
            "model_sizes": sfm.model_sizes,
            "sfm_reprojection_px": round(sfm.mean_reprojection_px, 3),
            "world_frame_error": str(e),
            "gate_reasons": check_reconstruction(sfm, None, limits),
        }

    row = {
        "scene": scene,
        "video": video,
        "board": board_kind,
        "frames": sfm.n_input_frames,
        "registered": len(sfm.images),
        "registered_fraction": round(sfm.registered_fraction, 4),
        "model_sizes": sfm.model_sizes,
        "sfm_reprojection_px": round(sfm.mean_reprojection_px, 3),
        "tags": sorted(res.tag_views),
        "tag_views": {str(k): v for k, v in sorted(res.tag_views.items())},
        "tag_reprojection_px": round(res.tag_reprojection_px, 3),
        "scale_m_per_unit": res.scale.scale,
        "scale_spread_pct": round(100 * res.scale.spread, 2),
        "reference_tag": res.reference_tag,
        "plane_inlier_fraction": round(res.table.inlier_fraction, 3),
        "plane_rms_mm": round(1000 * res.table.rms_m, 2),
        "flags": res.flags + (["sfm_fragmented"] if sfm.fragmented else []),
        "azimuth_coverage_deg": round(camera_coverage(sfm, res).azimuth_deg, 1),
        "gate_reasons": check_reconstruction(sfm, res, limits),
    }
    if board_kind == "clean":
        return row

    # Independent check from the known board spacing; the pipeline never sees it.
    board = fit_board(layout, res.tag_corners_recon)
    centers = {i: layout.tag_corners_m[i].mean(axis=0) for i in res.tag_centers_world_m}
    ids = sorted(centers)
    pair_err = [
        100.0
        * (
            np.linalg.norm(res.tag_centers_world_m[a] - res.tag_centers_world_m[b])
            / np.linalg.norm(centers[a] - centers[b])
            - 1.0
        )
        for k, a in enumerate(ids)
        for b in ids[k + 1 :]
    ]
    n_board = board.R_board_from_recon.T @ np.array([0.0, 0.0, 1.0])
    up = np.asarray(res.table.plane.normal)
    up_err = min(angle_between_deg(up, n_board), angle_between_deg(up, -n_board))
    row.update(
        {
            "scale_vs_board_pct": round(100 * (res.scale.scale / board.scale - 1.0), 2),
            "board_fit_rms_mm": round(1000 * board.rms_m, 2),
            "up_axis_error_deg": round(up_err, 3),
            "max_pair_distance_error_pct": round(float(np.max(np.abs(pair_err))), 2),
        }
    )
    return row


def _git_commit() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT
    )
    return out.stdout.strip() or "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("videos", nargs="+", help="scene/video, for example s01/video_b")
    parser.add_argument("--out", type=Path, required=True, help="results JSON to write")
    parser.add_argument("--work", type=Path, default=ROOT / "data/world_frame_work")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/default.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    gates = yaml.safe_load((ROOT / "configs/targets.yaml").read_text(encoding="utf-8"))["gates"][
        "reconstruction"
    ]
    layout = _board_layout(float(cfg["capture"]["marker"]["tag_size_m"]))
    limits = ReconstructionLimits.from_configs(cfg, {"gates": {"reconstruction": gates}})

    rows = []
    for item in args.videos:
        scene, video = item.split("/", 1)  # video may be a path such as alt_lighting/s01B
        try:
            row = evaluate(scene, video, args.work, cfg, layout, limits)
        except ValueError as e:  # a user-facing reconstruction failure, kept in the results
            row = {"scene": scene, "video": video, "error": str(e)}
        if "error" not in row and "registered_fraction" in row and "tags" in row:
            row["passes_gates"] = {
                "registered_fraction": row["registered_fraction"]
                >= gates["registered_frame_fraction_min"],
                "sfm_reprojection_px": row["sfm_reprojection_px"]
                <= gates["mean_reprojection_error_max_px"],
            }
            if row["board"] == "grid":  # these two need the known board layout
                row["passes_gates"]["marker_distance"] = (
                    row["max_pair_distance_error_pct"] / 100.0 <= gates["marker_distance_error_max"]
                )
                row["passes_gates"]["up_axis"] = (
                    row["up_axis_error_deg"] <= gates["up_axis_error_max_deg"]
                )
        log.info(json.dumps(row))
        rows.append(row)

    import pycolmap

    result = {
        "date": date.today().isoformat(),
        "git_commit": _git_commit(),
        "pycolmap": pycolmap.__version__,
        "board_assumed": "grid scenes: A1 grid, tag edge from capture.marker.tag_size_m",
        "gates": gates,
        "world_frame_config": cfg["reconstruction"]["world_frame"],
        "quality_config": cfg["reconstruction"]["quality"],
        "videos": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=float) + "\n", encoding="utf-8")
    sys.exit(0)


if __name__ == "__main__":
    main()
