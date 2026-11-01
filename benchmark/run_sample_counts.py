"""How few SAM 3D draws can the shape stage take? Replay cached draws with n = 1, 2 or 6.

    uv run python benchmark/run_sample_counts.py --out runs_samples

n = 1 takes the best keyframe (P2.4 order), and n = 2 the best two, one draw each, against today's
6 (2 keyframes, 3 seeds). Draws are random, so n = 1 and n = 2 repeat over the three cached seeds
(``r0`` to ``r2``). Each arm goes through the real ``shape_object`` and writes
``<out>/<arm>/<run>/`` with links to the earlier stages, for ``benchmark/compare_shape_arms.py``.

It makes no hosted call: a draw missing from ``shape.work`` is an error. To stay small on disk, the
kept mesh is decimated to ``--max-faces`` in every arm, including n = 6.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

import fast_simplification  # type: ignore[import-untyped]
import numpy as np
import trimesh
import yaml

from video2sim.evaluation.scenes import discover_scenes
from video2sim.models.sam3d import Sam3d, Sam3dCall, Sam3dConfig
from video2sim.models.shape_model import ShapeModel, ShapeSample, ShapeView, sample_seed
from video2sim.runner import OUTPUT_FILE, seed_for
from video2sim.schemas import Instances, RegisteredObjects, RunManifest, WorldModel
from video2sim.stages.shape import ShapeStageConfig, load_masks, shape_object

ROOT = Path(__file__).resolve().parent.parent
LINKED = ("capture", "reconstruction", "segmentation")


@dataclass(frozen=True)
class Arm:
    name: str
    keyframes: int
    samples: int
    replicate: int | None  # which cached seed; None draws them all


ARMS = (
    Arm("n6", 2, 3, None),
    *(Arm(f"n1_r{r}", 1, 1, r) for r in range(3)),
    *(Arm(f"n2_r{r}", 2, 1, r) for r in range(3)),
)


def no_upload(call: Sam3dCall) -> NoReturn:
    raise RuntimeError("draw missing from the cache; this script makes no hosted call")


class Replicate:
    """Answers with cached draw ``replicate`` whatever seed ``draw_samples`` asks for."""

    def __init__(self, inner: ShapeModel, base_seed: int, replicate: int) -> None:
        self.name = inner.name
        self._inner = inner
        self._base_seed = base_seed
        self._replicate = replicate

    def sample(self, object_id: str, view: ShapeView, seed: int) -> ShapeSample:
        chosen = sample_seed(self._base_seed, object_id, view.frame_index, self._replicate)
        return self._inner.sample(object_id, view, chosen)


def decimate(path: Path, max_faces: int) -> None:
    mesh: Any = trimesh.load(path, force="mesh", process=False)
    if len(mesh.faces) <= max_faces:
        return
    vertices, faces = fast_simplification.simplify(
        np.asarray(mesh.vertices, dtype=np.float64),
        np.asarray(mesh.faces),
        target_count=max_faces,
    )
    trimesh.Trimesh(vertices, faces, process=False).export(path, encoding="binary")


def run_arm(job: tuple[Path, Path, Arm, int]) -> tuple[str, str]:
    source, root, arm, max_faces = job
    run_dir = root / arm.name / source.name
    if (run_dir / "shape" / OUTPUT_FILE).exists():
        return arm.name, source.name
    out_dir = run_dir / "shape"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in LINKED:
        link = run_dir / name
        if not link.exists():
            link.symlink_to((source / name).resolve())

    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    shape = dict(cfg["shape"], keyframes_per_object=arm.keyframes)
    shape["samples_per_keyframe"] = arm.samples
    stage_cfg = ShapeStageConfig.from_config({"shape": shape})
    manifest = RunManifest.model_validate_json((source / "manifest.json").read_text())
    seed = seed_for(manifest.seed, "shape")
    model: ShapeModel = Sam3d(
        Sam3dConfig.from_config({"shape": shape}), no_upload, source / "shape.work" / "sam3d_cache"
    )
    if arm.replicate is not None:
        model = Replicate(model, seed, arm.replicate)

    world = WorldModel.model_validate_json((run_dir / "reconstruction" / OUTPUT_FILE).read_text())
    instances = Instances.model_validate_json((run_dir / "segmentation" / OUTPUT_FILE).read_text())
    masks = {o.object_id: load_masks(run_dir, o) for o in instances.objects}
    objects, report = [], {}
    for obj in instances.objects:
        registered, entry, _ = shape_object(
            obj, world, run_dir, out_dir, masks, model, stage_cfg, seed
        )
        decimate(run_dir / registered.mesh_path, max_faces)
        objects.append(registered)
        report[obj.object_id] = entry
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (out_dir / OUTPUT_FILE).write_text(RegisteredObjects(objects=objects).model_dump_json(indent=2))
    return arm.name, source.name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ROOT / "runs_instances")
    parser.add_argument("--out", type=Path, default=ROOT / "runs_samples")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--max-faces", type=int, default=100_000)
    parser.add_argument("--scene", action="append", help="limit to these scene ids")
    args = parser.parse_args()
    scenes, _ = discover_scenes(ROOT / "benchmark/scenes", ("dev",), None)
    runs = []
    for scene in scenes:
        if scene.has_deformable or (args.scene and scene.scene_id not in args.scene):
            continue
        found = [
            r
            for r in sorted(args.runs.glob(f"{scene.scene_id}_b"))
            if (r / "shape" / OUTPUT_FILE).exists()
        ]
        runs += found[:1]
    jobs = [(run, args.out, arm, args.max_faces) for arm in ARMS for run in runs]
    with ProcessPoolExecutor(args.workers) as pool:
        for arm_name, run_name in pool.map(run_arm, jobs):
            print(arm_name, run_name, flush=True)


if __name__ == "__main__":
    main()
