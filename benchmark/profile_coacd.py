"""How much of the machine does CoACD use, and does it gain from more threads or more objects?

    uv run python benchmark/profile_coacd.py s02_b --out FILE.json

CoACD's meshes are the run's repaired visual meshes that took a decomposition. Each measurement is
a fresh process, because OpenMP reads ``OMP_NUM_THREADS`` once. No hosted call.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
import yaml

ROOT = Path(__file__).resolve().parent.parent


def one(path: Path, threshold: float, parts: int, seed: int) -> tuple[float, str]:
    import coacd

    mesh: Any = trimesh.load(path, force="mesh", process=False)
    start = time.monotonic()
    result = coacd.run_coacd(
        coacd.Mesh(np.asarray(mesh.vertices, float), np.asarray(mesh.faces, np.int64)),
        threshold=threshold,
        max_convex_hull=parts,
        seed=seed,
    )
    digest = hashlib.sha256(b"".join(np.asarray(v).tobytes() for v, _ in result)).hexdigest()
    return time.monotonic() - start, digest[:12]


def child(meshes: list[str], workers: int) -> None:
    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text())["physics"]
    threshold = float(cfg["collision"]["coacd_threshold"])
    parts, seed = int(cfg["max_convex_parts"]), int(cfg["collision"]["seed"])
    import coacd

    coacd.set_log_level("off")  # creates CoACD's logger, which two threads must not do at once
    start = time.monotonic()
    with ThreadPoolExecutor(workers) as pool:
        rows = list(pool.map(lambda m: one(Path(m), threshold, parts, seed), meshes))
    wall = time.monotonic() - start
    use = resource.getrusage(resource.RUSAGE_SELF)
    cpu = use.ru_utime + use.ru_stime
    peak_gb = use.ru_maxrss / 1024**3  # bytes on macOS
    print(
        json.dumps(
            {
                "wall_s": wall,
                "cpu_s": cpu,
                "mean_cores": cpu / wall,
                "peak_rss_gb": peak_gb,
                "each_s": [r[0] for r in rows],
                "digests": [r[1] for r in rows],
            }
        )
    )


def measure(meshes: list[Path], workers: int, threads: int | None) -> dict[str, Any]:
    env = dict(os.environ)
    if threads is not None:
        env["OMP_NUM_THREADS"] = str(threads)
    command = [sys.executable, __file__, "--child", "--workers", str(workers)]
    for mesh in meshes:
        command += ["--mesh", str(mesh)]
    done = subprocess.run(command, env=env, capture_output=True, text=True, check=True)
    row: dict[str, Any] = json.loads(done.stdout.strip().splitlines()[-1])
    row.update(workers=workers, omp_threads=threads, meshes=len(meshes))
    print(
        f"{len(meshes)} meshes, {workers} at once, OMP {threads}: {row['wall_s']:.1f} s wall, "
        f"{row['mean_cores']:.2f} cores, {row['peak_rss_gb']:.2f} GB",
        flush=True,
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", nargs="?")
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--runs", type=Path, default=ROOT / "runs_instances")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--mesh", action="append", default=[])
    args = parser.parse_args()
    if args.child:
        child(args.mesh, args.workers)
        return
    run_dir = args.runs / args.run_id / "physics"
    report = json.loads((run_dir / "report.json").read_text())
    meshes = [
        run_dir / "meshes" / f"{k}.ply"
        for k, v in sorted(report.items())
        if v["collision_kind"] == "convex_decomposition"
    ]
    rows = []
    for _ in range(2):
        for threads in (1, 4, None):
            rows.append(measure(meshes, 1, threads))
        for workers in (2, 3, 4):
            rows.append(measure(meshes, workers, None))
    identical = len({tuple(r["digests"]) for r in rows}) == 1
    args.out.write_text(json.dumps({"same_result_everywhere": identical, "rows": rows}, indent=1))
    print("same result everywhere:", identical)


if __name__ == "__main__":
    main()
