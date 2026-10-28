"""Command line entry point: ``video2sim eval`` (P0.6). ``run`` arrives with P5.5."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from video2sim.evaluation import baselines
from video2sim.evaluation.loader import load_run
from video2sim.evaluation.metrics import SceneResult, evaluate_scene
from video2sim.evaluation.report import build_report, write_html, write_json
from video2sim.evaluation.scenes import GtScene, Prediction, discover_scenes
from video2sim.evaluation.segmentation import load_instances, summarize_segmentation
from video2sim.runner import _code_version, sha256_file

SPLIT_CHOICES = {"dev": ["dev"], "test": ["test"], "all": ["dev", "test"]}


def _predict(
    args: argparse.Namespace, scene: GtScene, segmentation_flags: dict[str, dict[str, Any]]
) -> Prediction:
    if args.baseline == "oracle":
        return baselines.oracle(
            scene,
            seed=args.seed,
            size_sigma=args.size_sigma,
            position_sigma_m=args.position_sigma_m,
            yaw_sigma_deg=args.yaw_sigma_deg,
            mass_sigma=args.mass_sigma,
        )
    if args.baseline == "constant":
        return baselines.constant_box()
    if args.stage == "segmentation":
        run = load_instances(args.runs / scene.scene_id)
        segmentation_flags[scene.scene_id] = dict(run.flags)
        return run.prediction
    return load_run(args.runs / scene.scene_id)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="video2sim")
    sub = parser.add_subparsers(dest="command", required=True)
    ev = sub.add_parser("eval", help="score predictions against the benchmark ground truth")
    ev.add_argument("--benchmark", type=Path, default=Path("benchmark/scenes"))
    ev.add_argument(
        "--split",
        choices=sorted(SPLIT_CHOICES),
        default="dev",
        help="test is for phase gates only; never tune on it (default: dev)",
    )
    ev.add_argument("--scene", help="evaluate only this scene id, whatever its split")
    ev.add_argument(
        "--baseline",
        choices=["oracle", "constant"],
        help="score a baseline instead of pipeline runs",
    )
    ev.add_argument(
        "--runs",
        type=Path,
        default=Path("runs"),
        help="folder holding one run folder per scene id (default: runs)",
    )
    ev.add_argument(
        "--stage",
        choices=["shape", "segmentation"],
        default="shape",
        help="which stage output to score for --runs: shape meshes or segmentation clusters",
    )
    ev.add_argument("--targets", type=Path, default=Path("configs/targets.yaml"))
    ev.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ev.add_argument("--out", type=Path, default=Path("eval_out"))
    ev.add_argument(
        "--fail-on-target",
        action="store_true",
        help="exit 1 when a measured metric misses its target",
    )
    ev.add_argument("--seed", type=int, default=0, help="oracle baseline noise seed")
    ev.add_argument("--size-sigma", type=float, default=0.0)
    ev.add_argument("--position-sigma-m", type=float, default=0.0)
    ev.add_argument("--yaw-sigma-deg", type=float, default=0.0)
    ev.add_argument("--mass-sigma", type=float, default=0.0)
    return parser


def run_eval(args: argparse.Namespace) -> int:
    targets: dict[str, Any] = yaml.safe_load(args.targets.read_text(encoding="utf-8"))
    tolerance = float(targets["targets"]["mass_tolerance"])
    scenes, missing = discover_scenes(args.benchmark, SPLIT_CHOICES[args.split], args.scene)

    results: list[SceneResult] = []
    segmentation_flags: dict[str, dict[str, Any]] = {}
    unpredicted: list[str] = []
    deformable = [s.scene_id for s in scenes if s.has_deformable and args.scene is None]
    for scene in scenes:
        if scene.scene_id in deformable:
            continue
        try:
            prediction = _predict(args, scene, segmentation_flags)
        except (OSError, ValueError) as error:
            print(f"{scene.scene_id}: no usable prediction ({error})", file=sys.stderr)
            unpredicted.append(scene.scene_id)
            continue
        results.append(evaluate_scene(scene, prediction, tolerance))

    header = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split": (args.scene and f"scene {args.scene}") or args.split,
        "source": args.baseline or f"runs in {args.runs}",
        "code_version": _code_version(),
        "config_sha256": sha256_file(args.config) if args.config.exists() else None,
        "scenes_scored": len(results),
        "scenes_without_prediction": ", ".join(unpredicted) or None,
        "scenes_excluded_deformable": ", ".join(deformable) or None,
    }
    report = build_report(results, targets, header, missing)
    if args.stage == "segmentation" or (args.baseline and args.baseline != "constant"):
        # Identity switches need track masks, which no loader reads yet (P2.2): no_data.
        report["segmentation"] = summarize_segmentation(
            results,
            {k: {o: tuple(f) for o, f in v.items()} for k, v in segmentation_flags.items()},
            {},
            targets.get("gates", {}).get("segmentation", {}),
            targets["targets"].get("object_recall_min"),
        )
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(report, args.out / "report.json")
    write_html(report, args.out / "report.html")
    print(f"wrote {args.out / 'report.json'} and {args.out / 'report.html'}")
    if missing:
        print(f"{len(missing)} scene folder(s) without ground truth: {', '.join(missing)}")

    if args.fail_on_target:
        failed = [m["name"] for m in report["summary"]["pooled"] if m["passed"] is False]
        if failed:
            print(f"targets missed: {', '.join(failed)}", file=sys.stderr)
            return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "eval":
        return run_eval(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
