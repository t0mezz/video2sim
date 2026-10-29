"""Pick keyframes per object for finished runs and compare with a no-diversity pick (P2.4).

    uv run python benchmark/run_keyframes.py s01_b s02_b --out benchmark/results/p2_4.json
    uv run python benchmark/run_keyframes.py s01_b --preview /tmp/kf

Needs ``runs/<id>/segmentation/tracks/`` and a lift (``points/`` or ``clusters/``) from
``run_lift.py``; makes no hosted call. The object centre for the viewing angle is the median of
the sparse-point cluster, else of the hull cluster. For each run the pick with the diversity
penalty is written to ``segmentation/keyframes.json`` and compared with the same scores and
``diversity_weight`` 0, on mean quality and on the smallest and mean angle between the viewing
directions of the chosen frames. ``--preview DIR`` writes one contact sheet of crops per track.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
from collections.abc import Mapping, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from video2sim.runner import OUTPUT_FILE
from video2sim.schemas import WorldModel
from video2sim.stages.keyframes import (
    KeyframeChoice,
    KeyframeConfig,
    keyframes_for_tracks,
    view_directions,
)
from video2sim.stages.tracking import Track, load_tracks

ROOT = Path(__file__).resolve().parent.parent


def load_centers(seg_dir: Path) -> dict[int, NDArray[np.float64]]:
    centers: dict[int, NDArray[np.float64]] = {}
    for folder in ("clusters", "points"):  # points last, so they win
        for path in sorted((seg_dir / folder).glob("track_*.npy")):
            pts = np.load(path).astype(np.float64)
            if len(pts):
                centers[int(path.stem.split("_")[1])] = np.median(pts, axis=0)
    return centers


def angles_deg(
    world: WorldModel, center: NDArray[np.float64], picks: Sequence[KeyframeChoice]
) -> list[float]:
    d = view_directions(world, center, [p.frame_index for p in picks])
    return [
        float(np.degrees(np.arccos(np.clip(d[a.frame_index] @ d[b.frame_index], -1.0, 1.0))))
        for a, b in combinations(picks, 2)
    ]


def summarize(
    world: WorldModel,
    centers: Mapping[int, NDArray[np.float64]],
    picks: Mapping[int, list[KeyframeChoice]],
) -> dict[str, Any]:
    qualities = [p.quality for ps in picks.values() for p in ps]
    pair = [a for t, ps in picks.items() for a in angles_deg(world, centers[t], ps)]
    mins = [min(a) for t, ps in picks.items() if (a := angles_deg(world, centers[t], ps))]
    return {
        "frames": sum(len(ps) for ps in picks.values()),
        "mean_quality": round(statistics.fmean(qualities), 3) if qualities else None,
        "mean_pair_angle_deg": round(statistics.fmean(pair), 1) if pair else None,
        "mean_min_pair_angle_deg": round(statistics.fmean(mins), 1) if mins else None,
    }


def preview(
    path: Path, run_dir: Path, world: WorldModel, track: Track, picks: Sequence[KeyframeChoice]
) -> None:
    import cv2

    paths = {f.frame_index: run_dir / f.image_path for f in world.frames}
    tiles = []
    for p in picks:
        image = cv2.imread(str(paths[p.frame_index]))
        mask = track.masks[p.frame_index]
        ys, xs = np.nonzero(mask)
        pad = 30
        y0, y1 = max(0, ys.min() - pad), min(mask.shape[0], ys.max() + pad)
        x0, x1 = max(0, xs.min() - pad), min(mask.shape[1], xs.max() + pad)
        crop = image[y0:y1, x0:x1].copy()
        edge = mask[y0:y1, x0:x1].astype(np.uint8) * 255
        contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(crop, contours, -1, (0, 255, 0), 1)
        crop = cv2.resize(crop, (240, 240))
        label = f"f{p.frame_index} q{p.quality:.2f} s{p.score:.2f}"
        cv2.putText(crop, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        tiles.append(crop)
    if tiles:
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), np.hstack(tiles))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_ids", nargs="+")
    parser.add_argument("--runs", type=Path, default=ROOT / "runs")
    parser.add_argument("--out", type=Path, default=ROOT / "benchmark/results/p2_4_keyframes.json")
    parser.add_argument("--preview", type=Path, default=None)
    args = parser.parse_args()

    cfg_all: dict[str, Any] = yaml.safe_load(
        (ROOT / "configs/default.yaml").read_text(encoding="utf-8")
    )
    cfg = KeyframeConfig.from_config(cfg_all)
    plain_cfg = dataclasses.replace(cfg, diversity_weight=0.0)

    rows: list[dict[str, Any]] = []
    for run_id in args.run_ids:
        run_dir = args.runs / run_id
        seg_dir = run_dir / "segmentation"
        world = WorldModel.model_validate_json(
            (run_dir / "reconstruction" / OUTPUT_FILE).read_text(encoding="utf-8")
        )
        tracks = load_tracks(seg_dir / "tracks")
        centers = load_centers(seg_dir)
        picks, report = keyframes_for_tracks(tracks, world, run_dir, centers, cfg)
        plain, _ = keyframes_for_tracks(tracks, world, run_dir, centers, plain_cfg)
        (seg_dir / "keyframes.json").write_text(
            json.dumps(
                {
                    "keyframes": {
                        str(t): [dataclasses.asdict(c) for c in cs] for t, cs in picks.items()
                    },
                    "notes": {str(t): n for t, n in report.notes.items()},
                },
                indent=2,
            )
            + "\n"
        )
        row = {
            "run": run_id,
            "tracks": len(tracks),
            "with_tracks_centered": len(picks),
            "diverse": summarize(world, centers, picks),
            "quality_only": summarize(world, centers, plain),
            "notes": {str(t): n for t, n in report.notes.items()},
            "frames": {str(t): [c.frame_index for c in cs] for t, cs in picks.items()},
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        if args.preview is not None:
            by_id = {t.track_id: t for t in tracks}
            for t, cs in picks.items():
                preview(args.preview / f"{run_id}_track{t:03d}.png", run_dir, world, by_id[t], cs)

    def avg(key: str, field: str) -> float:
        vals = [r[key][field] for r in rows if r[key][field] is not None]
        return round(statistics.fmean(vals), 2) if vals else float("nan")

    summary = {
        key: {
            f: avg(key, f)
            for f in ("mean_quality", "mean_pair_angle_deg", "mean_min_pair_angle_deg")
        }
        for key in ("diverse", "quality_only")
    }
    print(json.dumps(summary, indent=2))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "scenes": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
