"""Time pycolmap per stage on benchmark videos, from scratch each run (PLAN.md item 6b, step 1).

    uv run python benchmark/profile_sfm.py s01/video_b s16/video_a --out benchmark/results/x.json

Frames are sampled at 3 fps like ``eval_world_frame.py``. The work directory is wiped first so
nothing is cached. Records frame decode, feature extraction, matching and mapping seconds, the
frames registered, the mean reprojection error, and the host (CPU count, pycolmap version).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", nargs="+", help="scene/video, for example s01/video_b")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--work", type=Path, default=ROOT / "data/profile_sfm_work")
    parser.add_argument("--fps", type=float, default=3.0)
    args = parser.parse_args()

    import pycolmap

    from video2sim.stages.reconstruction import extract_frames, run_sfm

    rows = []
    for name in args.videos:
        scene, video = name.split("/")
        root = args.work / f"{scene}_{video}"
        shutil.rmtree(root, ignore_errors=True)
        t0 = time.perf_counter()
        mp4 = ROOT / f"benchmark/scenes/{scene}/{video}.mp4"
        n = extract_frames(mp4, root / "frames", args.fps)
        decode_s = time.perf_counter() - t0
        timings: dict[str, float] = {}
        sfm = run_sfm(root / "frames", root / "sfm", timings)
        total = decode_s + sum(timings.values())
        row = {
            "video": name,
            "fps": args.fps,
            "frames": n,
            "registered": len(sfm.images),
            "model_sizes": sfm.model_sizes,
            "points": len(sfm.points_xyz),
            "mean_reprojection_px": round(sfm.mean_reprojection_px, 3),
            "decode_s": round(decode_s, 1),
            **{f"{k}_s": round(v, 1) for k, v in timings.items()},
            "total_s": round(total, 1),
        }
        print(json.dumps(row), flush=True)
        rows.append(row)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "host": {
                        "machine": platform.machine(),
                        "platform": platform.platform(),
                        "cpu_count": os.cpu_count(),
                        "pycolmap": pycolmap.__version__,
                        "has_cuda": pycolmap.has_cuda,
                    },
                    "runs": rows,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
