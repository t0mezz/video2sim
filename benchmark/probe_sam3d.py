"""Cold and warm SAM 3D draw times and GPU memory, from the worker's own stats.

    uv run python benchmark/probe_sam3d.py s02_b --out FILE.json

One cold call on one keyframe, three warm calls, a gap for the containers to scale down, then one
burst of six calls at once on different objects. Ten hosted calls, none cached. Seeds are new, so
these meshes are thrown away.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import yaml

from video2sim.models.sam3d import Sam3d, Sam3dConfig, Sam3dTransport, _deployed_worker
from video2sim.models.shape_model import ShapeView
from video2sim.runner import OUTPUT_FILE
from video2sim.schemas import Instances, WorldModel
from video2sim.stages.shape import load_masks

ROOT = Path(__file__).resolve().parent.parent
SCALE_DOWN_WAIT_S = 360  # scaledown_window is 300 in models/sam3d_modal.py


def no_upload(call: Any) -> Any:
    raise RuntimeError("the probe calls the worker itself")


def views(run_dir: Path) -> list[ShapeView]:
    world = WorldModel.model_validate_json((run_dir / "reconstruction" / OUTPUT_FILE).read_text())
    images = {f.frame_index: run_dir / f.image_path for f in world.frames}
    found = []
    instances = Instances.model_validate_json((run_dir / "segmentation" / OUTPUT_FILE).read_text())
    for obj in instances.objects:
        masks = load_masks(run_dir, obj)
        frame = next(k for k in obj.keyframes if k in masks and k in images)
        found.append(ShapeView(frame, images[frame], masks[frame].full()))
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--runs", type=Path, default=ROOT / "runs_instances")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    transport: Sam3dTransport = no_upload
    sam3d = Sam3d(Sam3dConfig.from_config(cfg), transport, ROOT / "runs_timing" / "probe_cache")
    worker = _deployed_worker()
    shown = views(args.runs / args.run_id)

    def call(label: str, view: ShapeView, seed: int) -> dict[str, Any]:
        request = sam3d._build_call(view, seed)
        sent = time.time()
        out = worker.sample.remote(
            request.image, request.mask_png, request.seed, request.pointmap_npy, request.options
        )
        wall = time.time() - sent
        stats = dict(out["stats"])
        # The worker's clock and ours differ slightly, so queueing is read from durations.
        row = {"label": label, "seed": seed, "wall_s": wall, "glb_mb": len(out["glb"]) / 1e6}
        row.update(stats)
        row["outside_sample_s"] = wall - stats["sample_s"]
        print(
            json.dumps({k: row[k] for k in ("label", "wall_s", "sample_s", "load_s")}), flush=True
        )
        return row

    rows = [call("cold", shown[0], 900001)]
    rows += [call("warm", shown[0], 900002 + k) for k in range(3)]
    time.sleep(SCALE_DOWN_WAIT_S)
    jobs = [(shown[k % len(shown)], 900010 + k) for k in range(6)]
    with ThreadPoolExecutor(6) as pool:
        rows += list(pool.map(lambda j: call("burst", j[0], j[1]), jobs))
    args.out.write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
