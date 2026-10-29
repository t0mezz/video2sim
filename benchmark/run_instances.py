"""Run the segmentation stage (P2.6) on copies of finished runs and write ``Instances``.

    uv run python benchmark/run_instances.py s17_b s24_b --out runs_instances

The runner replaces a stage folder when it runs it, and the benchmark scripts keep their caches
and intermediate files in ``runs/<id>/segmentation/``. So the stage never runs in ``runs/``: for
each run id this script builds ``<out>/<id>/`` with links to the capture and reconstruction
folders, a copy of the manifest, and the hosted-answer caches moved into ``segmentation.work``,
so a run with unchanged inputs makes no hosted call. Anything missing from the cache is a hosted
call (router, SAM video, material pass); the keys come from the environment or ``.env``.

The scene id is the run id up to the last underscore and the video is
``benchmark/scenes/<scene>/video_<a|b>.mp4`` from the suffix. Prints the stage status and, per
object, the flags and the material answer; writes a summary to ``--summary``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

from video2sim.runner import OUTPUT_FILE, Runner, StageFailed, sha256_file
from video2sim.schemas import Instances, RunManifest
from video2sim.stages.capture import CaptureStage
from video2sim.stages.reconstruction import ReconstructionStage, reconstruction_stage_config
from video2sim.stages.segmentation_stage import SegmentationStage

ROOT = Path(__file__).resolve().parent.parent
CACHES = ("tracking_cache", "router_cache", "material_vlm_cache", "material_crops")


def prepare(source: Path, target: Path) -> None:
    """Link the finished stages of ``source`` into ``target`` and seed the hosted caches."""
    target.mkdir(parents=True, exist_ok=True)
    for name in ("capture", "reconstruction", "reconstruction.work"):
        link = target / name
        if not link.exists() and (source / name).exists():
            link.symlink_to((source / name).resolve())
    if not (target / "manifest.json").exists():
        shutil.copy(source / "manifest.json", target / "manifest.json")
    work = target / "segmentation.work"
    work.mkdir(exist_ok=True)
    for name in CACHES:
        if (source / "segmentation" / name).is_dir() and not (work / name).exists():
            shutil.copytree(source / "segmentation" / name, work / name)


def accept_finished_stages(runner: Runner, run_dir: Path, video: Path) -> None:
    """Record the cache keys of today's code for the capture and reconstruction stages.

    Those runs were made with earlier config and source files, so their recorded keys no longer
    match and the runner would redo reconstruction (minutes per run) and replace the linked
    folders. The outputs are taken as they are. This only edits the copy of the manifest in
    ``--out``, and only a harness does it: a real run executes every stage.
    """
    manifest = RunManifest.model_validate_json((run_dir / "manifest.json").read_text())
    video_sha = sha256_file(video)
    for stage in runner.stages:
        if stage.name == "segmentation":
            break
        key = runner._cache_key(stage, manifest, video_sha)
        record = manifest.stages[stage.name].model_copy(update={"input_sha256": key})
        manifest = manifest.model_copy(update={"stages": {**manifest.stages, stage.name: record}})
    (run_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_ids", nargs="+")
    parser.add_argument("--runs", type=Path, default=ROOT / "runs")
    parser.add_argument("--out", type=Path, default=ROOT / "runs_instances")
    parser.add_argument(
        "--summary", type=Path, default=ROOT / "benchmark/results/p2_6_instances.json"
    )
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    targets = yaml.safe_load((ROOT / "configs/targets.yaml").read_text(encoding="utf-8"))
    runner = Runner(
        args.out,
        [CaptureStage(), ReconstructionStage(), SegmentationStage()],
        {
            "capture": cfg["capture"],
            "reconstruction": reconstruction_stage_config(cfg, targets),
            "segmentation": cfg["segmentation"],
        },
    )
    summary: dict[str, Any] = {}
    for run_id in args.run_ids:
        scene, _, which = run_id.rpartition("_")
        prepare(args.runs / run_id, args.out / run_id)
        video = ROOT / "benchmark/scenes" / scene / f"video_{which}.mp4"
        accept_finished_stages(runner, args.out / run_id, video)
        try:
            manifest = runner.run(video, run_id=run_id)
        except StageFailed as e:
            print(run_id, "FAILED", e)
            sys.exit(1)
        status = {n: (r.status, round(r.seconds or 0, 1)) for n, r in manifest.stages.items()}
        inst = Instances.model_validate_json(
            (args.out / run_id / "segmentation" / OUTPUT_FILE).read_text(encoding="utf-8")
        )
        print(run_id, {k: v for k, v in status.items() if k in ("reconstruction", "segmentation")})
        for w in inst.flags:
            print("   scene:", w)
        for o in inst.objects:
            print("   ", o.object_id, o.label, o.material_class, o.rigidity,
                  [f.value for f in o.flags], o.keyframes)  # fmt: skip
        summary[run_id] = {
            "status": status,
            "scene_flags": inst.flags,
            "objects": [o.model_dump(mode="json") for o in inst.objects],
        }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
