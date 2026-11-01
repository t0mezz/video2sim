"""Time the shape and physics stages and track CPU use, on one dev run.

    uv run python benchmark/profile_stages.py s02_b --arm pipelined --out FILE.json
    uv run python benchmark/profile_stages.py s02_b --arm sequential --out FILE.json

Each call starts from an empty cache in ``runs_timing/<tag>/``, linked to the run's earlier stages,
so **every SAM 3D draw and mass request is a real hosted call**, unless ``--cached`` points the
cache at the run's own. ``sequential`` is the earlier behavior: one object at a time. Core use is
process CPU time over wall time, read four times a second, and each task's wall time is recorded.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path
from typing import Any

import yaml

from video2sim.models.sam3d import warm_containers
from video2sim.runner import OUTPUT_FILE, StageContext, seed_for
from video2sim.schemas import Instances, RegisteredObjects, RunManifest, SimAssets, WorldModel
from video2sim.stages import collision
from video2sim.stages import physics_stage as physics
from video2sim.stages import shape as shape_module
from video2sim.stages.physics_stage import PhysicsStage, physics_stage_config
from video2sim.stages.shape import ShapeStage, ShapeStageConfig

ROOT = Path(__file__).resolve().parent.parent
LINKED = ("capture", "reconstruction", "segmentation")
PERIOD_S = 0.25


class Sampler(threading.Thread):
    """Process CPU time and memory over time."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.rows: list[tuple[float, float, int]] = []
        self.done = threading.Event()

    def run(self) -> None:
        pid = str(os.getpid())
        while not self.done.wait(PERIOD_S):
            t = os.times()
            rss = subprocess.run(["ps", "-o", "rss=", "-p", pid], capture_output=True, text=True)
            self.rows.append((time.monotonic(), t.user + t.system, int(rss.stdout.strip() or 0)))


class Tasks:
    """Wall time of each wrapped call, by name."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self.lock = threading.Lock()

    def wrap(self, module: Any, name: str) -> None:
        inner: Callable[..., Any] = getattr(module, name)

        def timed(*args: Any, **kwargs: Any) -> Any:
            start = time.monotonic()
            try:
                return inner(*args, **kwargs)
            finally:
                row = {"task": name, "start": start, "end": time.monotonic()}
                with self.lock:
                    self.rows.append(row)

        setattr(module, name, timed)


def summarize(rows: list[tuple[float, float, int]], start: float, end: float) -> dict[str, float]:
    window = [r for r in rows if start <= r[0] <= end]
    if len(window) < 2:
        return {"wall_s": end - start, "cpu_s": 0.0, "mean_cores": 0.0, "peak_cores": 0.0}
    rates = [(b[1] - a[1]) / (b[0] - a[0]) for a, b in pairwise(window)]
    cpu = window[-1][1] - window[0][1]
    return {
        "wall_s": end - start,
        "cpu_s": cpu,
        "mean_cores": cpu / max(window[-1][0] - window[0][0], 1e-9),
        "peak_cores": max(rates),
        "peak_rss_gb": max(r[2] for r in window) / 1024**2,
    }


def stage_context(
    stage: Any, run_dir: Path, config: Any, seed: int, types: dict[str, Any]
) -> StageContext:
    out = run_dir / stage.name
    if out.exists():
        shutil.rmtree(out)
    out.mkdir()
    return StageContext(
        stage=stage.name,
        run_dir=run_dir,
        video=run_dir / "video.mp4",
        out_dir=out,
        config=config,
        seed=seed,
        dependencies=stage.dependencies,
        output_types=types,
    )


def sequential_shape(stage: ShapeStage) -> Callable[[StageContext], RegisteredObjects]:
    """The stage as it was: every object through ``shape_object`` in turn."""

    def run(ctx: StageContext) -> RegisteredObjects:
        world = ctx.load("reconstruction", WorldModel)
        instances = ctx.load("segmentation", Instances)
        cfg: dict[str, Any] = {"shape": dict(ctx.config)}
        stage_cfg = ShapeStageConfig.from_config(cfg)
        work = ctx.run_dir / "shape.work"
        work.mkdir(exist_ok=True)
        model = stage._model(cfg, work)
        masks = {o.object_id: shape_module.load_masks(ctx.run_dir, o) for o in instances.objects}
        results = [
            shape_module.shape_object(
                o, world, ctx.run_dir, ctx.out_dir, masks, model, stage_cfg, ctx.seed
            )
            for o in instances.objects
        ]
        reports = {o.object_id: r[1] for o, r in zip(instances.objects, results, strict=True)}
        (ctx.out_dir / "report.json").write_text(json.dumps(reports, indent=2, sort_keys=True))
        return RegisteredObjects(objects=[r[0] for r in results])

    return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--arm", choices=["sequential", "pipelined"], required=True)
    parser.add_argument("--tag", help="run folder name; default: the arm")
    parser.add_argument("--runs", type=Path, default=ROOT / "runs_instances")
    parser.add_argument("--cached", action="store_true", help="reuse the run's hosted answers")
    parser.add_argument("--parallel-objects", type=int, help="override physics.parallel_objects")
    parser.add_argument("--stages", default="shape,physics")
    parser.add_argument("--warm", type=int, default=0, help="containers to warm before shape")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    source = args.runs / args.run_id
    run_dir = ROOT / "runs_timing" / (args.tag or args.arm) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in LINKED:
        if not (run_dir / name).exists():
            (run_dir / name).symlink_to((source / name).resolve())
    if args.cached:
        for name in ("shape.work", "physics.work"):
            if not (run_dir / name).exists():
                (run_dir / name).symlink_to((source / name).resolve())

    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    targets = yaml.safe_load((ROOT / "configs/targets.yaml").read_text(encoding="utf-8"))
    sequential = args.arm == "sequential"
    if sequential:
        cfg["physics"].update(parallel_mass_calls=1, parallel_objects=1)
    if args.parallel_objects:
        cfg["physics"]["parallel_objects"] = args.parallel_objects
    seed = RunManifest.model_validate_json((source / "manifest.json").read_text()).seed
    types = {
        "reconstruction": WorldModel,
        "segmentation": Instances,
        "shape": RegisteredObjects,
        "physics": SimAssets,
    }

    tasks = Tasks()
    for name in ("register_sample", "refine_stretch"):
        tasks.wrap(shape_module, name)
    for name in ("model_mass", "repair_mesh", "collision_geometry", "estimate_mass"):
        tasks.wrap(physics, name)
    tasks.wrap(physics, "settling_report")
    tasks.wrap(collision, "_decompose")

    hold_s = float(cfg["shape"]["sam3d"]["warm_hold_s"])
    warmed = warm_containers(args.warm, hold_s) if args.warm else []
    sampler = Sampler()
    sampler.start()
    spans: dict[str, tuple[float, float]] = {}
    stages: dict[str, tuple[Any, Any]] = {
        "shape": (ShapeStage(), cfg["shape"]),
        "physics": (
            PhysicsStage(ROOT / cfg["physics"]["material_table"]),
            physics_stage_config(cfg, targets),
        ),
    }
    for name in args.stages.split(","):
        stage, config = stages[name]
        ctx = stage_context(stage, run_dir, config, seed_for(seed, name), types)
        run = sequential_shape(stage) if name == "shape" and sequential else stage.run
        start = time.monotonic()
        output = run(ctx)
        spans[name] = (start, time.monotonic())
        (ctx.out_dir / OUTPUT_FILE).write_text(output.model_dump_json(indent=2))
    sampler.done.set()
    sampler.join()

    origin = min(s for s, _ in spans.values())
    grouped: dict[str, dict[str, float]] = {}
    for row in tasks.rows:
        entry = grouped.setdefault(row["task"], {"calls": 0, "busy_s": 0.0, "longest_s": 0.0})
        length = row["end"] - row["start"]
        entry["calls"] += 1
        entry["busy_s"] += length
        entry["longest_s"] = max(entry["longest_s"], length)
    result = {
        "run": args.run_id,
        "arm": args.arm,
        "cached": args.cached,
        "parallel_objects": cfg["physics"]["parallel_objects"],
        "warmed": warmed,
        "stages": {k: summarize(sampler.rows, *v) for k, v in spans.items()},
        "tasks": grouped,
        "timeline": [
            {"t_s": round(b[0] - origin, 2), "cores": round((b[1] - a[1]) / (b[0] - a[0]), 2)}
            for a, b in pairwise(sampler.rows)
        ],
        "task_spans": [
            {**r, "start": round(r["start"] - origin, 2), "end": round(r["end"] - origin, 2)}
            for r in tasks.rows
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: result[k] for k in ("arm", "stages", "tasks")}, indent=1))


if __name__ == "__main__":
    main()
