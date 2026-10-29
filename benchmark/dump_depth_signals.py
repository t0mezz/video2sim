"""Dump raw unstable-depth signals per track for finished runs (P2.5 calibration).

    uv run python benchmark/dump_depth_signals.py s17_b s18_b --out results/p2_5_signals.json
    uv run python benchmark/dump_depth_signals.py s17_b --preview /tmp/sig

Needs ``runs/<id>/segmentation/tracks/`` and ``clusters/`` (the hull lift) from
``run_lift.py``, and the SfM model in ``runs/<id>/reconstruction.work/sparse/`` for per-point
reprojection error and track length. Makes no hosted call and decides nothing: it only writes the
numbers, so thresholds can be chosen after looking at them. Signals per track:

* ``sfm_*``: SfM points whose projection lies in the mask in at least half the views that see
  them (looser than the P2.3 vote): count, density per bbox surface, mean error and track length
  against the scene's table points, and the share lying outside the hull box (points seen
  through the object).
* ``hull_iou_*``: IoU between the projected hull cluster and the mask, over a spread of views.
* ``sat_frac`` and ``grad_med``: specular highlight share and median gradient inside the mask.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from video2sim.runner import OUTPUT_FILE
from video2sim.schemas import WorldModel
from video2sim.stages.depth_flags import depth_signals
from video2sim.stages.lift import LiftConfig
from video2sim.stages.tracking import load_tracks

ROOT = Path(__file__).resolve().parent.parent
F64 = NDArray[np.float64]
PHOTO_VIEWS = 12


def photometric(run_dir: Path, world: WorldModel, track: Any) -> dict[str, float | None]:
    import cv2

    frames = [f for f in world.frames if f.frame_index in track.masks]
    sizes = [(int(track.masks[f.frame_index].sum()), f) for f in frames]
    sizes = [s for s in sizes if s[0] > 400]
    if not sizes:
        return {"sat_frac": None, "grad_med": None}
    step = max(1, len(sizes) // PHOTO_VIEWS)
    sat, grad = [], []
    for _, f in sizes[::step][:PHOTO_VIEWS]:
        gray = cv2.imread(str(run_dir / f.image_path), cv2.IMREAD_GRAYSCALE)
        inner = cv2.erode(track.masks[f.frame_index].astype(np.uint8), np.ones((5, 5), np.uint8))
        if inner.sum() < 100:
            continue
        sel = inner > 0
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        sat.append(float((gray[sel] >= 245).mean()))
        grad.append(float(np.median(np.hypot(gx, gy)[sel])))
    if not sat:
        return {"sat_frac": None, "grad_med": None}
    return {"sat_frac": statistics.median(sat), "grad_med": statistics.median(grad)}


def scene_signals(run_dir: Path, cfg: LiftConfig) -> list[dict[str, Any]]:
    seg = run_dir / "segmentation"
    world = WorldModel.model_validate_json(
        (run_dir / "reconstruction" / OUTPUT_FILE).read_text(encoding="utf-8")
    )
    tracks = load_tracks(seg / "tracks")
    hulls = {
        t.track_id: np.load(seg / "clusters" / f"track_{t.track_id:03d}.npy").astype(np.float64)
        for t in tracks
        if (seg / "clusters" / f"track_{t.track_id:03d}.npy").exists()
    }
    by_id = {t.track_id: t for t in tracks}
    rows = []
    for sig in depth_signals(tracks, world, run_dir, hulls, cfg):
        row = {k: round(v, 3) if isinstance(v, float) else v for k, v in asdict(sig).items()}
        row.update(
            {
                k: None if v is None else round(v, 3)
                for k, v in photometric(run_dir, world, by_id[sig.track_id]).items()
            }
        )
        rows.append(row)
    return rows


def sheet(path: Path, run_dir: Path, run_id: str) -> None:
    """One crop per track (largest mask), labelled with the track id, for labelling by eye."""
    import cv2

    world = WorldModel.model_validate_json(
        (run_dir / "reconstruction" / OUTPUT_FILE).read_text(encoding="utf-8")
    )
    paths = {f.frame_index: run_dir / f.image_path for f in world.frames}
    tiles = []
    for tr in load_tracks(run_dir / "segmentation" / "tracks"):
        index = max(tr.masks, key=lambda i: int(tr.masks[i].sum()))
        mask = tr.masks[index]
        ys, xs = np.nonzero(mask)
        if not len(ys):
            continue
        pad = 40
        image = cv2.imread(str(paths[index]))
        y0, y1 = max(0, ys.min() - pad), min(mask.shape[0], ys.max() + pad)
        x0, x1 = max(0, xs.min() - pad), min(mask.shape[1], xs.max() + pad)
        crop = cv2.resize(image[y0:y1, x0:x1], (240, 240))
        cv2.putText(
            crop,
            f"{run_id} t{tr.track_id}",
            (4, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )
        tiles.append(crop)
    if tiles:
        rows = [
            np.hstack(tiles[i : i + 6] + [np.zeros_like(tiles[0])] * (6 - len(tiles[i : i + 6])))
            for i in range(0, len(tiles), 6)
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), np.vstack(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_ids", nargs="+")
    parser.add_argument("--runs", type=Path, default=ROOT / "runs")
    parser.add_argument("--out", type=Path, default=ROOT / "benchmark/results/p2_5_signals.json")
    parser.add_argument("--preview", type=Path, default=None)
    args = parser.parse_args()
    cfg = LiftConfig.from_config(
        yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    )
    scenes = {}
    for run_id in args.run_ids:
        scenes[run_id] = scene_signals(args.runs / run_id, cfg)
        print(run_id, len(scenes[run_id]), "tracks", flush=True)
        if args.preview is not None:
            sheet(args.preview / f"{run_id}.png", args.runs / run_id, run_id)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(scenes, indent=2) + "\n")


if __name__ == "__main__":
    main()
