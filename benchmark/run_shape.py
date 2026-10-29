"""Run the shape stage (P3.4 to P3.6) on runs that already have ``Instances``.

    uv run python benchmark/run_shape.py s01_b --runs runs_instances

Each run id must be a folder in ``--runs`` made by ``benchmark/run_instances.py``. The finished
capture, reconstruction and segmentation stages are taken as they are (their recorded cache keys
are brought up to today's code, in that folder's manifest only), then the shape stage runs and
writes ``shape/output.json``, ``shape/meshes/`` and ``shape/report.json``. Shape model answers
are cached in ``shape.work/``; anything missing is a hosted call on Modal. Prints one line per
object and writes ``scene.json`` for ``scripts/view_scene.py`` beside the stage folder.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from video2sim.runner import OUTPUT_FILE, Runner, StageFailed, sha256_file
from video2sim.schemas import RegisteredObjects, RunManifest
from video2sim.stages.capture import CaptureStage
from video2sim.stages.reconstruction import ReconstructionStage, reconstruction_stage_config
from video2sim.stages.segmentation_stage import SegmentationStage
from video2sim.stages.shape import ShapeStage

ROOT = Path(__file__).resolve().parent.parent


def accept_finished_stages(runner: Runner, run_dir: Path, video: Path) -> None:
    """Record today's cache keys for every stage before ``shape``, as ``run_instances.py`` does
    for the stages before ``segmentation``. A harness shortcut: a real run executes them."""
    manifest = RunManifest.model_validate_json((run_dir / "manifest.json").read_text())
    video_sha = sha256_file(video)
    for stage in runner.stages:
        if stage.name == "shape":
            break
        key = runner._cache_key(stage, manifest, video_sha)
        record = manifest.stages[stage.name].model_copy(update={"input_sha256": key})
        manifest = manifest.model_copy(update={"stages": {**manifest.stages, stage.name: record}})
    (run_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2))


def viewer_scene(run_dir: Path, registered: RegisteredObjects) -> dict[str, Any]:
    """``scripts/view_scene.py`` input: meshes already in meters in the object frame."""
    report = json.loads((run_dir / "shape" / "report.json").read_text())
    objects = []
    for obj in registered.objects:
        entry = report[obj.object_id]
        info = {"source": obj.geometry_source, "samples": str(len(entry["samples"]))}
        if obj.fit_score is not None:
            info["fit score"] = f"{obj.fit_score:.2f}"
        if obj.flags:
            info["flags"] = ", ".join(f.value for f in obj.flags)
        objects.append(
            {
                "name": obj.object_id,
                "glb": str(run_dir / obj.mesh_path),
                "T_world_from_model": obj.T_world_from_object.matrix,
                "scale": 1.0,
                "info": info,
            }
        )
    return {"run": str(run_dir), "objects": objects}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_ids", nargs="+")
    parser.add_argument("--runs", type=Path, default=ROOT / "runs_instances")
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    targets = yaml.safe_load((ROOT / "configs/targets.yaml").read_text(encoding="utf-8"))
    runner = Runner(
        args.runs,
        [CaptureStage(), ReconstructionStage(), SegmentationStage(), ShapeStage()],
        {
            "capture": cfg["capture"],
            "reconstruction": reconstruction_stage_config(cfg, targets),
            "segmentation": cfg["segmentation"],
            "shape": cfg["shape"],
        },
    )
    for run_id in args.run_ids:
        scene, _, which = run_id.rpartition("_")
        run_dir = args.runs / run_id
        video = ROOT / "benchmark/scenes" / scene / f"video_{which}.mp4"
        accept_finished_stages(runner, run_dir, video)
        try:
            manifest = runner.run(video, run_id=run_id)
        except StageFailed as e:
            print(run_id, "FAILED", e)
            sys.exit(1)
        record = manifest.stages["shape"]
        print(run_id, record.status, f"{record.seconds or 0:.0f} s", manifest.checkpoints)
        registered = RegisteredObjects.model_validate_json(
            (run_dir / "shape" / OUTPUT_FILE).read_text(encoding="utf-8")
        )
        for obj in registered.objects:
            fit = "-" if obj.fit_score is None else f"{obj.fit_score:.2f}"
            print("   ", obj.object_id, obj.geometry_source, fit, [f.value for f in obj.flags])
        (run_dir / "scene.json").write_text(json.dumps(viewer_scene(run_dir, registered), indent=1))


if __name__ == "__main__":
    main()
