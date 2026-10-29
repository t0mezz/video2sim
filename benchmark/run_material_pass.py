"""Material pass and reflective flag per track for finished runs (P2.5, P4.4 first half).

    uv run python benchmark/run_material_pass.py s17_b s18_b --out results/p2_5_material.json
    uv run python benchmark/run_material_pass.py s17_b --vlm      # makes hosted calls

Needs the inputs of ``dump_depth_signals.py`` plus ``segmentation/keyframes.json`` (a track
without keyframes falls back to its largest masks). By default no hosted call is made: the crops
are written to ``segmentation/material_crops/`` so they can be looked at, and every track is
flagged as unverified. ``--vlm`` sends ``segmentation.material_pass.keyframes`` crops per track in
one request to the hosted model (needs ``BASETEN_API_KEY``); answers are cached under
``segmentation/material_vlm_cache``, so a rerun with the same crops and prompt costs nothing.
Writes ``segmentation/material_pass.json`` per run.
Each hosted request is timed; the summary and the output JSON carry per-scene and per-request
seconds. Requests run ``segmentation.material_pass.max_workers`` at a time.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from video2sim.runner import OUTPUT_FILE
from video2sim.schemas import WorldModel
from video2sim.stages.depth_flags import DepthConfig, depth_signals, judge
from video2sim.stages.discovery_vlm_sam import ResponseCache, read_secret
from video2sim.stages.lift import LiftConfig
from video2sim.stages.material_pass import (
    BasetenMaterialClient,
    MaterialClient,
    MaterialConfig,
    crop_for_vlm,
    decide_all,
    pick_frames,
)
from video2sim.stages.tracking import load_tracks

ROOT = Path(__file__).resolve().parent.parent
FLAGGED = "flagged_reflective_or_transparent"
NOT_FLAGGED = "not_flagged_reflective_or_transparent"


def run_one(
    run_dir: Path,
    lift: LiftConfig,
    depth: DepthConfig,
    cfg: MaterialConfig,
    client: MaterialClient | None,
) -> list[dict[str, Any]]:
    seg = run_dir / "segmentation"
    world = WorldModel.model_validate_json(
        (run_dir / "reconstruction" / OUTPUT_FILE).read_text(encoding="utf-8")
    )
    tracks = load_tracks(seg / "tracks")
    by_id = {t.track_id: t for t in tracks}
    hulls = {
        t.track_id: np.load(path).astype(np.float64)
        for t in tracks
        if (path := seg / "clusters" / f"track_{t.track_id:03d}.npy").exists()
    }
    keyframe_file = seg / "keyframes.json"
    keyframes = (
        json.loads(keyframe_file.read_text(encoding="utf-8"))["keyframes"]
        if keyframe_file.exists()
        else {}
    )

    def crops(track_id: int) -> list[Path]:
        picks = [c["frame_index"] for c in keyframes.get(str(track_id), [])]
        frames = pick_frames(by_id[track_id], picks, cfg.keyframes)
        return [
            crop_for_vlm(
                run_dir,
                world,
                by_id[track_id],
                f,
                cfg.crop_pad_px,
                seg / "material_crops" / f"track_{track_id:03d}_f{f:05d}.png",
            )
            for f in frames
        ]

    signals = depth_signals(tracks, world, run_dir, hulls, lift)
    verdicts = [judge(s, depth) for s in signals]
    if client is None:  # dry run: still write the crops so they can be inspected
        for s in signals:
            crops(s.track_id)
    cache = ResponseCache(seg / "material_vlm_cache") if client else None
    decisions = decide_all(signals, verdicts, cfg, crops, client, cache)
    rows = [
        {
            "track": d.track_id,
            "flag": d.flag.value if d.flag else None,
            "non_rigid": d.non_rigid,
            "reason": d.reason,
            "geometry": d.geometry,
            "geometry_reason": d.geometry_reason,
            "answer": asdict(d.answer) if d.answer else None,
            "raw_answer": d.raw_answer,
            "seconds": None if d.seconds is None else round(d.seconds, 2),
            "cached": d.cached,
            "signals": asdict(s),
        }
        for d, s in zip(decisions, signals, strict=True)
    ]
    (seg / "material_pass.json").write_text(json.dumps(rows, indent=2) + "\n")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_ids", nargs="+")
    parser.add_argument("--runs", type=Path, default=ROOT / "runs")
    parser.add_argument("--out", type=Path, default=ROOT / "benchmark/results/p2_5_material.json")
    parser.add_argument("--vlm", action="store_true", help="make hosted calls, one per track")
    args = parser.parse_args()
    all_cfg: dict[str, Any] = yaml.safe_load(
        (ROOT / "configs/default.yaml").read_text(encoding="utf-8")
    )
    lift, depth = LiftConfig.from_config(all_cfg), DepthConfig.from_config(all_cfg)
    cfg = MaterialConfig.from_config(all_cfg)
    client = (
        BasetenMaterialClient(read_secret("BASETEN_API_KEY"), cfg.timeout_s, cfg.retries)
        if args.vlm
        else None
    )
    scenes: dict[str, Any] = {}
    seconds: dict[str, float] = {}
    totals: Counter[str] = Counter()
    started = time.monotonic()
    for run_id in args.run_ids:
        t0 = time.monotonic()
        rows = run_one(args.runs / run_id, lift, depth, cfg, client)
        seconds[run_id] = round(time.monotonic() - t0, 1)
        scenes[run_id] = rows
        labels = [FLAGGED if r["flag"] else NOT_FLAGGED for r in rows]
        totals.update(labels)
        flags = dict(Counter(labels))
        print(run_id, flags, f"{seconds[run_id]} s", flush=True)
    wall = round(time.monotonic() - started, 1)
    asked = [r["seconds"] for rows in scenes.values() for r in rows if r["seconds"] is not None]
    live = [
        r["seconds"] for rows in scenes.values() for r in rows if r["seconds"] and not r["cached"]
    ]
    timing = {
        "wall_s": wall,
        "scene_s": seconds,
        "requests": len(asked),
        "cached": len(asked) - len(live),
        "request_s_mean": round(statistics.fmean(live), 2) if live else None,
        "request_s_max": round(max(live), 2) if live else None,
    }
    print("totals:", dict(totals))
    print("timing:", {k: v for k, v in timing.items() if k != "scene_s"})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"totals": dict(totals), "timing": timing, "scenes": scenes}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
