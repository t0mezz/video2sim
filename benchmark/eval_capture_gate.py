"""Run the capture quality gate on benchmark videos and on deliberately degraded copies (P1.4).

    uv run python benchmark/eval_capture_gate.py --out benchmark/results/p1_4_capture_gate.json

Real videos: every ``*.mp4`` under the dev-split scenes in ``benchmark/scene_list.yaml``, the
alternative captures included. Scenes with a deformable object are left out (D-014). The test
split is not touched. Degraded copies are made from one good video (``--base``) with a known
defect each and written to ``--work`` (ignored by git); the gate must reject every one and
name the defect. Nothing here changes the gate: thresholds are in ``configs/default.yaml``.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from video2sim.stages.capture import CaptureQualityConfig, sample_and_check
from video2sim.stages.marker_detection import detect_tags

log = logging.getLogger("eval_capture_gate")
ROOT = Path(__file__).resolve().parent.parent


def _dev_videos() -> list[Path]:
    scenes = yaml.safe_load((ROOT / "benchmark/scene_list.yaml").read_text(encoding="utf-8"))
    out: list[Path] = []
    for s in scenes:
        if s["split"] != "dev" or "deformable" in s["scene"]:
            continue
        out += sorted((ROOT / "benchmark/scenes" / s["scene"]).rglob("*.mp4"))
    return out


def _hide_marker(frame: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for corners in detect_tags(gray).values():
        pad = np.asarray(corners)
        c = pad.mean(axis=0)
        cv2.fillPoly(frame, [((pad - c) * 1.6 + c).astype(np.int32)], (200, 200, 200))
    return frame


def _degraded(base: Path, out_dir: Path) -> dict[str, Path]:
    """One defect per copy: whole-video blur, a blurred burst, dark, overexposed, no marker."""
    cap = cv2.VideoCapture(str(base))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    variants: dict[str, Callable[[int, np.ndarray[Any, Any]], np.ndarray[Any, Any]]] = {
        "blur_whole": lambda i, f: cv2.GaussianBlur(f, (0, 0), 9),
        "blur_burst_50pct": lambda i, f: (
            cv2.GaussianBlur(f, (0, 0), 9) if n * 0.25 <= i < n * 0.75 else f
        ),
        "dark": lambda i, f: (f * 0.12).astype(np.uint8),
        "overexposed": lambda i, f: cv2.add(f, np.full_like(f, 120)),
        "marker_hidden": lambda i, f: _hide_marker(f.copy()),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(3)), int(cap.get(4)))
    paths = {name: out_dir / f"{name}.avi" for name in variants}
    writers = {
        name: cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)  # type: ignore[attr-defined]  # stubs lack fourcc
        for name, p in paths.items()
    }
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for name, fn in variants.items():
            writers[name].write(fn(i, frame))
        i += 1
    for w in writers.values():
        w.release()
    return paths


def _row(video: Path, work: Path, cfg: CaptureQualityConfig, label: str) -> dict[str, Any]:
    run_dir = work / "runs" / label.replace("/", "_")
    shutil.rmtree(run_dir, ignore_errors=True)
    result, report = sample_and_check(video, run_dir, "frames", cfg)
    shutil.rmtree(run_dir, ignore_errors=True)  # frames are not part of the results
    return {
        "video": label,
        "accepted": result.accepted,
        "reasons": result.reasons,
        "frames": len(result.frame_paths),
        "metrics": {k: round(v, 3) for k, v in report.metrics.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--work", type=Path, default=ROOT / "data/capture_gate_work")
    parser.add_argument("--base", default="s01/video_a", help="good video to degrade")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/default.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    cfg = CaptureQualityConfig.from_config(raw)
    real = []
    for v in _dev_videos():
        row = _row(v, args.work, cfg, str(v.relative_to(ROOT / "benchmark/scenes")))
        log.info(json.dumps(row))
        real.append(row)

    base = ROOT / "benchmark/scenes" / f"{args.base}.mp4"
    degraded = []
    for name, path in _degraded(base, args.work / "degraded").items():
        row = _row(path, args.work, cfg, f"degraded/{name}")
        row["expected_rejection"] = True
        log.info(json.dumps(row))
        degraded.append(row)
    shutil.rmtree(args.work / "degraded", ignore_errors=True)

    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT
    ).stdout.strip()
    result = {
        "date": date.today().isoformat(),
        "git_commit": commit or "unknown",
        "capture_config": raw["capture"],
        "degraded_from": args.base,
        "real_videos": real,
        "degraded_videos": degraded,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
