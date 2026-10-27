"""Experimental: time pycolmap's SIFT extraction and matching on CPU vs CUDA.

Not part of the pipeline. One-off script for the pycolmap-gpu-benchmark experiment
branch, to get real numbers on this machine before deciding whether to revisit
D-016 (pycolmap on CPU, PLAN.md). See benchmark/README.md for the result.

`pycolmap` (CPU wheel) and `pycolmap-cuda12` (GPU wheel) both import as `pycolmap` and
cannot be installed together, so this script must be run once per environment:

    .venv/bin/python     benchmark/gpu_bench.py --device cpu  --out benchmark/results/gpu_benchmark_cpu.json
    .venv-gpu/bin/python  benchmark/gpu_bench.py --device cuda --out benchmark/results/gpu_benchmark_cuda.json

`pycolmap.incremental_mapping` (bundle adjustment) has no `device` argument in either
build, so the "map" phase is CPU-bound regardless and is only timed for context.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import cv2
import pycolmap

DEFAULT_SCENES = ["s01", "s02", "s05", "s10", "s19", "s23"]
TARGET_FPS = 5.0


def extract_frames(video: Path, out_dir: Path, target_fps: float) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"cannot open video {video}")
    step = max(1, round(cap.get(cv2.CAP_PROP_FPS) / target_fps))
    index = written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % step == 0:
            cv2.imwrite(str(out_dir / f"{written:05d}.png"), frame)
            written += 1
        index += 1
    cap.release()
    return written


def time_phase(fn) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def run_once(frames_dir: Path, work_dir: Path, device: "pycolmap.Device") -> dict:
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)
    db = work_dir / "database.db"
    sparse = work_dir / "sparse"
    sparse.mkdir()

    t_extract = time_phase(
        lambda: pycolmap.extract_features(
            db, frames_dir, camera_mode=pycolmap.CameraMode.SINGLE, device=device
        )
    )
    t_match = time_phase(lambda: pycolmap.match_exhaustive(db, device=device))
    t_map = time_phase(lambda: pycolmap.incremental_mapping(db, frames_dir, sparse))

    models = [pycolmap.Reconstruction(str(m)) for m in sorted(sparse.iterdir())]
    best = max(models, key=lambda r: r.num_reg_images()) if models else None
    n_input = len(list(frames_dir.glob("*.png")))
    result = {
        "extract_s": t_extract,
        "match_s": t_match,
        "map_s": t_map,
        "total_s": t_extract + t_match + t_map,
        "n_frames": n_input,
        "n_registered": best.num_reg_images() if best else 0,
        "mean_reproj_px": best.compute_mean_reprojection_error() if best else None,
    }
    shutil.rmtree(work_dir)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "cuda"], required=True)
    ap.add_argument("--scenes", nargs="*", default=DEFAULT_SCENES)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1, help="untimed runs before recording")
    ap.add_argument("--work-root", default="benchmark/gpu_bench_work")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = {"cpu": pycolmap.Device.cpu, "cuda": pycolmap.Device.cuda}[args.device]
    if args.device == "cuda" and not pycolmap.has_cuda:
        raise SystemExit("this pycolmap build has no CUDA support (need pycolmap-cuda12)")

    work_root = Path(args.work_root)
    results: dict[str, list[dict]] = {}
    for scene in args.scenes:
        video = Path(f"benchmark/scenes/{scene}/video_a.mp4")
        frames_dir = work_root / scene / "frames"
        if not frames_dir.exists() or not any(frames_dir.iterdir()):
            n = extract_frames(video, frames_dir, TARGET_FPS)
            print(f"{scene}: extracted {n} frames")

        for i in range(args.warmup):
            run_once(frames_dir, work_root / scene / f"warmup_{args.device}_{i}", device)

        runs = []
        for i in range(args.repeats):
            r = run_once(frames_dir, work_root / scene / f"work_{args.device}_{i}", device)
            print(f"{scene} run {i}: {r}")
            runs.append(r)
        results[scene] = runs

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
