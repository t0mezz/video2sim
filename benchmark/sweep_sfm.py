"""Try pycolmap speed settings on one video and score each against the reconstruction gates.

    uv run python benchmark/sweep_sfm.py s16/video_a --out benchmark/results/x.json \
        --variant seq10=fps=3,matcher=sequential,sequential_overlap=10

PLAN.md item 6b. A variant is ``name=key=value,key=value``; ``fps`` is the frame sampling rate
(default 3) and every other key is a field of ``SfmOptions``. ``baseline`` is always run first
unless ``--no-baseline`` is given. Each variant starts from scratch in its own work folder, and
is scored with ``eval_world_frame.evaluate`` (tag distance against the printed board, up-axis,
tag and SfM reprojection) next to its per-stage seconds.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import time
from pathlib import Path
from typing import Any

import yaml

from video2sim.stages.reconstruction import ReconstructionLimits, SfmOptions

ROOT = Path(__file__).resolve().parents[1]


def _load_eval() -> Any:
    spec = importlib.util.spec_from_file_location(
        "eval_world_frame", ROOT / "benchmark/eval_world_frame.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_variant(text: str) -> tuple[str, float, dict[str, Any]]:
    name, _, rest = text.partition("=")
    fps = 3.0
    opts: dict[str, Any] = {}
    for item in filter(None, rest.split(",")):
        key, _, value = item.partition("=")
        if key == "fps":
            fps = float(value)
        else:
            opts[key] = yaml.safe_load(value)
    return name, fps, opts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="scene/video, for example s16/video_a")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--variant", action="append", default=[])
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("--work", type=Path, default=ROOT / "data/sweep_sfm_work")
    args = parser.parse_args()

    ev = _load_eval()
    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    gates = yaml.safe_load((ROOT / "configs/targets.yaml").read_text(encoding="utf-8"))["gates"][
        "reconstruction"
    ]
    layout = ev._board_layout(float(cfg["capture"]["marker"]["tag_size_m"]))
    limits = ReconstructionLimits.from_configs(cfg, {"gates": {"reconstruction": gates}})
    scene, video = args.video.split("/", 1)

    variants = [parse_variant(v) for v in args.variant]
    if not args.no_baseline:
        variants.insert(0, ("baseline", 3.0, {}))
    rows = []
    for name, fps, opts in variants:
        work = args.work / f"{scene}_{video}_{name}"
        shutil.rmtree(work, ignore_errors=True)
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        row = ev.evaluate(
            scene, video, work, cfg, layout, limits, SfmOptions.from_config(opts), fps, timings
        )
        row.update(
            variant=name,
            fps=fps,
            options=opts,
            **{f"{k}_s": round(v, 1) for k, v in timings.items()},
            sfm_total_s=round(sum(timings.values()), 1),
            wall_s=round(time.perf_counter() - t0, 1),
        )
        print(json.dumps(row), flush=True)
        rows.append(row)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps({"gates": gates, "runs": rows}, indent=2, default=float) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
