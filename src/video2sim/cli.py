"""Command line entry point: ``eval`` (P0.6), ``run`` (P5.5) and ``doctor``."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from video2sim.evaluation import baselines
from video2sim.evaluation.alignment import BoardAlignment, align_scene
from video2sim.evaluation.loader import load_run
from video2sim.evaluation.metrics import SceneResult, evaluate_scene
from video2sim.evaluation.report import build_report, write_html, write_json
from video2sim.evaluation.scenes import GtScene, Prediction, discover_scenes
from video2sim.evaluation.segmentation import (
    load_instances,
    load_tracks,
    score_tracks,
    summarize_segmentation,
)
from video2sim.progress import ProgressReporter
from video2sim.runner import (
    MANIFEST_FILE,
    OUTPUT_FILE,
    VERBOSE_ENV,
    RunRejected,
    StageFailed,
    _code_version,
    sha256_file,
)
from video2sim.schemas import SE3, WorldModel

SPLIT_CHOICES = {"dev": ["dev"], "test": ["test"], "all": ["dev", "test"]}


def _predict(
    args: argparse.Namespace,
    scene: GtScene,
    segmentation_flags: dict[str, dict[str, Any]],
    track_switches: dict[str, int] | None = None,
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
        run_dir = args.runs / scene.scene_id
        alignment = _board_from_world(args, scene, run_dir)
        run = load_instances(run_dir, alignment.T_board_from_world)
        segmentation_flags[scene.scene_id] = dict(run.flags)
        if track_switches is not None:
            try:
                track_switches[scene.scene_id] = _count_switches(scene, run_dir, alignment)
            except (OSError, ValueError) as error:
                print(f"{scene.scene_id}: no track data ({error})", file=sys.stderr)
        return run.prediction
    run_dir = args.runs / scene.scene_id
    return load_run(run_dir, _board_from_world(args, scene, run_dir).T_board_from_world)


def _count_switches(scene: GtScene, run_dir: Path, alignment: BoardAlignment) -> int:
    """Identity switches of the repaired tracks against the scene ground truth.

    Raises OSError when files are missing and ValueError when the data does not fit.
    """
    world = WorldModel.model_validate_json(
        (run_dir / "reconstruction" / OUTPUT_FILE).read_text(encoding="utf-8")
    )
    tracks = load_tracks(run_dir)
    if not tracks:
        raise ValueError("no track masks")
    sample = next(iter(next(iter(tracks.values())).values()))
    if sample.shape != (world.intrinsics.height_px, world.intrinsics.width_px):
        raise ValueError("track masks do not match the reconstruction resolution")
    frames = {f.frame_index: f for f in world.frames}
    poses: dict[int, SE3] = {}
    for masks in tracks.values():
        for frame in masks:
            if frame not in frames:
                raise ValueError(f"track frame {frame} has no camera pose")
            poses.setdefault(frame, alignment.pose_in_board(frames[frame].T_world_from_camera))
    return score_tracks(scene.objects, world.intrinsics, poses, tracks).switches


def _board_from_world(args: argparse.Namespace, scene: GtScene, run_dir: Path) -> BoardAlignment:
    """Board alignment of a run, so its clusters meet the board-frame ground truth.

    Raises ``ValueError`` for a clean-board scene whose origin is not established, which the
    caller reports as a scene without a usable prediction."""
    marker = yaml.safe_load(args.config.read_text(encoding="utf-8"))["capture"]["marker"]
    world = WorldModel.model_validate_json(
        (run_dir / "reconstruction" / OUTPUT_FILE).read_text(encoding="utf-8")
    )
    alignment = align_scene(
        world, run_dir, scene.scene_id, scene.board, float(marker["tag_size_m"]), marker["family"]
    )
    if alignment is None:
        raise ValueError("clean board: origin of the ground truth is not established")
    return alignment


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
    run = sub.add_parser("run", help="turn one video into a scene package")
    run.add_argument("video", type=Path, nargs="?", help="the video (not needed with --dry-run)")
    run.add_argument(
        "--out", type=Path, help="folder for the scene package (not needed with --dry-run)"
    )
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="show fake stage progress; read no video, call nothing, write nothing",
    )
    run.add_argument("--runs", type=Path, default=Path("runs"), help="run folders (default: runs)")
    run.add_argument("--run-id", help="resume this run; default: a new id from time and video")
    run.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    run.add_argument("--targets", type=Path, default=Path("configs/targets.yaml"))
    run.add_argument(
        "--gateway",
        metavar="URL",
        help="send every model call to the judge gateway at URL with the token in V2S_JUDGE_TOKEN",
    )
    noise = run.add_mutually_exclusive_group()
    noise.add_argument("--quiet", action="store_true", help="no progress on stderr")
    noise.add_argument(
        "--verbose",
        action="store_true",
        help="plain progress with run id and cache keys, and the logs of COLMAP and CoACD",
    )
    rp = sub.add_parser(
        "replay",
        help="show a recorded run of a benchmark scene: no models, keys or GPU needed",
    )
    rp.add_argument("scene", nargs="?", help="scene id; leave out to list the scenes")
    rp.add_argument("--out", type=Path, help="folder for the scene package")
    rp.add_argument("--video", type=Path, help="pick the scene by the SHA-256 of this video")
    rp.add_argument("--archive", type=Path, help="use this snapshot zip instead of downloading")
    rp.add_argument("--index", type=Path, default=None, help="lock file (default: the one in git)")
    rp.add_argument("--cache", type=Path, default=None, help="where archives and runs are kept")
    rp.add_argument("--instant", action="store_true", help="do not wait between stages")
    rp.add_argument("--quiet", action="store_true", help="no progress on stderr")
    doctor = sub.add_parser("doctor", help="check that the configured model endpoints work")
    doctor.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    doctor.add_argument(
        "--offline", action="store_true", help="check the config and keys only; call nothing"
    )
    doctor.add_argument(
        "--gateway",
        metavar="URL",
        help="check the judge gateway at URL with the token in V2S_JUDGE_TOKEN",
    )
    return parser


def _load_config(path: Path, gateway: str | None) -> dict[str, Any] | None:
    """The config at ``path``, pointed at the judge gateway when one is given. None, with the
    reason on stderr, when the config cannot use the gateway."""
    cfg: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    if gateway is None:
        return cfg
    from video2sim.gateway_profile import apply_gateway

    try:
        return apply_gateway(cfg, gateway)
    except ValueError as error:
        print(f"--gateway: {error}", file=sys.stderr)
        return None


def run_doctor_command(args: argparse.Namespace) -> int:
    from video2sim.doctor import run_doctor

    cfg = _load_config(args.config, args.gateway)
    if cfg is None:
        return 2
    checks = run_doctor(cfg, offline=args.offline, gateway=args.gateway)
    for check in checks:
        print(f"{'ok  ' if check.ok else 'FAIL'} {check.name}: {check.detail}")
    return 0 if all(c.ok for c in checks) else 1


def run_eval(args: argparse.Namespace) -> int:
    targets: dict[str, Any] = yaml.safe_load(args.targets.read_text(encoding="utf-8"))
    tolerance = float(targets["targets"]["mass_tolerance"])
    scenes, missing = discover_scenes(args.benchmark, SPLIT_CHOICES[args.split], args.scene)

    results: list[SceneResult] = []
    segmentation_flags: dict[str, dict[str, Any]] = {}
    track_switches: dict[str, int] = {}
    unpredicted: list[str] = []
    deformable = [s.scene_id for s in scenes if s.has_deformable and args.scene is None]
    for scene in scenes:
        if scene.scene_id in deformable:
            continue
        try:
            prediction = _predict(args, scene, segmentation_flags, track_switches)
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
        # Baselines have no tracks, so their switches row stays no_data.
        report["segmentation"] = summarize_segmentation(
            results,
            {k: {o: tuple(f) for o, f in v.items()} for k, v in segmentation_flags.items()},
            track_switches,
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


def run_pipeline(args: argparse.Namespace) -> int:
    if args.dry_run:
        from video2sim.dry_run import simulate

        fake = ProgressReporter(sys.stderr, verbose=args.verbose)
        try:
            simulate(lambda event: None if args.quiet else fake(event))
        finally:
            fake.close()
        print("dry run: nothing was run or written", file=sys.stderr)
        return 0
    if args.video is None or args.out is None:
        print("run needs a VIDEO and --out FOLDER (or --dry-run)", file=sys.stderr)
        return 2
    from video2sim.pipeline import build_runner  # heavy imports, not needed for eval

    cfg = _load_config(args.config, args.gateway)
    if cfg is None:
        return 2
    targets: dict[str, Any] = yaml.safe_load(args.targets.read_text(encoding="utf-8"))
    if args.verbose:
        os.environ[VERBOSE_ENV] = "1"
    reporter = (
        None
        if args.quiet
        else ProgressReporter(sys.stderr, verbose=args.verbose, runs_root=args.runs)
    )
    runner = build_runner(args.runs, cfg, targets, Path.cwd(), on_event=reporter)
    try:
        manifest = runner.run(args.video, run_id=args.run_id)
    except (StageFailed, RunRejected) as error:
        print(str(error), file=sys.stderr)
        return 1
    finally:
        if reporter is not None:
            reporter.close()
    run_dir = args.runs / manifest.run_id
    shutil.copytree(run_dir / "export", args.out, dirs_exist_ok=True)
    shutil.copy2(run_dir / MANIFEST_FILE, args.out / MANIFEST_FILE)
    print(f"wrote the scene package to {args.out}; run folder {run_dir}")
    return 0


def run_replay(args: argparse.Namespace) -> int:
    from video2sim import replay as rp

    index = args.index or rp.DEFAULT_INDEX
    cache = args.cache or rp.DEFAULT_CACHE
    try:
        entries = rp.load_index(index)
        if args.scene is None and args.video is None:
            for entry in entries.values():
                print(f"{entry.scene_id}  {entry.note}".rstrip())
            return 0
        if args.out is None:
            print("replay needs --out FOLDER", file=sys.stderr)
            return 2
        entry = rp.pick_scene(entries, args.scene, args.video)
        archive = args.archive or rp.fetch_archive(entry, cache, rp.download_url)
        run_dir = rp.unpack(archive, entry, cache / "runs")
        reporter = None if args.quiet else ProgressReporter(sys.stderr, runs_root=cache / "runs")
        if reporter is not None:
            print(
                "replay of a recorded run: nothing is computed, no model is called", file=sys.stderr
            )
        try:
            rp.replay(run_dir, reporter or (lambda event: None), instant=args.instant)
        finally:
            if reporter is not None:
                reporter.close()
        rp.copy_package(run_dir, args.out)
    except rp.SnapshotError as error:
        print(str(error), file=sys.stderr)
        return 1
    print(f"wrote the recorded scene package of {entry.scene_id} to {args.out}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "eval":
        return run_eval(args)
    if args.command == "run":
        return run_pipeline(args)
    if args.command == "replay":
        return run_replay(args)
    if args.command == "doctor":
        return run_doctor_command(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
