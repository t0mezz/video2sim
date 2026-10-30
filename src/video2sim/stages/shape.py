"""Stage 4: shape and pose (P3.4 to P3.6). ``WorldModel`` and ``Instances`` in,
``RegisteredObjects`` out.

Per object:

1. Samples (P3.1). The shape model is run on the first ``keyframes_per_object`` keyframes,
   ``samples_per_keyframe`` seeds each (seeds from ``models/shape_model.sample_seed``). Answers
   are cached in ``shape.work/`` outside the hashed stage folder, as ``segmentation.work/`` is,
   so a rerun with the same inputs makes no hosted call.
2. Registration (P3.2). Each sample is registered to ``point_cluster_path`` and the object's
   masks in up to ``max_views`` frames spread over the video, other objects' masks counting as
   occluders. The start rotation is the model's guess in its keyframe. On s01 the cluster (the
   P2.4 primitive outline) and the voted SfM points gave sizes within 2 points of each other
   and the same mask IoU, so the typed cluster is used as it is.
3. Selection (P3.4). The ``stretch_samples`` samples with the highest ``fit_score`` get the
   per-axis refinement (``registration.refine_stretch``), and the highest of those after it is
   kept. When none reaches
   ``min_fit_score``, the object gets the convex hull of its cluster instead, with
   ``Flag.PROXY_GEOMETRY`` and ``Flag.LOW_FIT``, and ``geometry_source`` ``proxy_hull``.
4. Resting (P3.5). An object whose cluster comes within ``support_gap_m`` of the table rests on
   it: registration holds its lowest vertex there (``support_z``), and the small gap or
   overlap left is then closed exactly by a vertical shift. The stable face is the model axis
   that the start rotation turns closest to up (``registration.level``); a stacked object
   keeps its fitted height. The world frame's +Z is the table normal (stage 2), so the table
   height is ``offset_m``.
5. Output (P3.6). The mesh is written in the object frame, in meters, already scaled, as
   ``shape/meshes/<object_id>.ply``. The frame is the registered model frame moved to the
   volume centroid; for a mesh that is not closed the centroid of its convex hull stands in
   (P4.1 repairs meshes). ``shape/report.json`` lists every sample's fit for the P3.3 bake-off.

A failed hosted call fails the stage rather than leaving an object out quietly (AGENTS.md).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast, get_args

import cv2
import numpy as np
from numpy.typing import NDArray

from video2sim.models.shape_model import ShapeModel, ShapeSample, ShapeView, draw_samples
from video2sim.runner import StageContext
from video2sim.schemas import (
    SE3,
    Flag,
    GeometrySource,
    Instances,
    ObjectInstance,
    RegisteredObject,
    RegisteredObjects,
    StageName,
    WorldModel,
)
from video2sim.stages.registration import (
    MaskView,
    Registration,
    RegistrationConfig,
    refine_stretch,
    register,
)

F64 = NDArray[np.float64]
Masks = dict[int, NDArray[np.bool_]]
MESH_DIR = "meshes"


@dataclass(frozen=True)
class ShapeStageConfig:
    """Values from ``shape`` in ``configs/default.yaml``."""

    backend: str
    keyframes_per_object: int
    samples_per_keyframe: int
    max_views: int
    min_fit_score: float
    support_gap_m: float
    stretch_samples: int
    parallel_samples: int
    registration: RegistrationConfig

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> ShapeStageConfig:
        c = cfg["shape"]
        out = cls(
            str(c["backend"]),
            int(c["keyframes_per_object"]),
            int(c["samples_per_keyframe"]),
            int(c["max_views"]),
            float(c["min_fit_score"]),
            float(c["support_gap_m"]),
            int(c["stretch_samples"]),
            int(c["parallel_samples"]),
            RegistrationConfig.from_config(cfg),
        )
        counts = (
            out.keyframes_per_object,
            out.samples_per_keyframe,
            out.max_views,
            out.stretch_samples,
            out.parallel_samples,
        )
        if min(counts) < 1:
            raise ValueError(
                "keyframes_per_object, samples_per_keyframe, max_views, stretch_samples and "
                "parallel_samples must be at least 1"
            )
        if not 0 <= out.min_fit_score <= 1:
            raise ValueError("min_fit_score must be in [0, 1]")
        if out.support_gap_m < 0:
            raise ValueError("support_gap_m must not be negative")
        return out


ModelFactory = Callable[[dict[str, Any], Path], ShapeModel]


def default_model(cfg: dict[str, Any], work: Path) -> ShapeModel:
    """The hosted model named by ``shape.backend``."""
    backend = cfg["shape"]["backend"]
    if backend == "sam3d":
        from video2sim.models.sam3d import ModalTransport, Sam3d, Sam3dConfig

        return Sam3d(Sam3dConfig.from_config(cfg), ModalTransport(), work / "sam3d_cache")
    if backend == "trellis2":
        from video2sim.models.trellis2 import ModalTransport as TrellisTransport
        from video2sim.models.trellis2 import Trellis2, Trellis2Config

        return Trellis2(
            Trellis2Config.from_config(cfg), TrellisTransport(), work / "trellis2_cache"
        )
    if backend == "hybrid":
        # TRELLIS.2 geometry, SAM 3D pose (P3.3): both caches live in this run's work folder
        from video2sim.models.hybrid import HybridShape
        from video2sim.models.sam3d import ModalTransport as Sam3dTransport
        from video2sim.models.sam3d import Sam3d, Sam3dConfig
        from video2sim.models.trellis2 import ModalTransport as TrellisTransport
        from video2sim.models.trellis2 import Trellis2, Trellis2Config

        return HybridShape(
            Sam3d(Sam3dConfig.from_config(cfg), Sam3dTransport(), work / "sam3d_cache"),
            Trellis2(Trellis2Config.from_config(cfg), TrellisTransport(), work / "trellis2_cache"),
        )
    raise ValueError(f"shape.backend {backend!r} has no adapter (sam3d, trellis2 or hybrid)")


def load_masks(run_dir: Path, obj: ObjectInstance) -> Masks:
    """The object's masks by frame index, from the PNG folder that stage 3 wrote."""
    masks: Masks = {}
    for path in sorted((run_dir / obj.mask_dir).glob("*.png")):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"cannot read mask {path}")
        if image.any():
            masks[int(path.stem)] = image > 0
    return masks


def spread(frames: Sequence[int], count: int) -> list[int]:
    """Up to ``count`` frames evenly spaced over ``frames`` (sorted), first and last included."""
    ordered = sorted(frames)
    if len(ordered) <= count:
        return ordered
    picks = np.linspace(0, len(ordered) - 1, count).round().astype(int)
    return [ordered[i] for i in sorted(set(picks.tolist()))]


def mask_views(
    world: WorldModel, own: Masks, others: Sequence[Masks], frames: Sequence[int]
) -> list[MaskView]:
    poses = {f.frame_index: f.T_world_from_camera for f in world.frames}
    views = []
    for frame in frames:
        hidden = [m[frame] for m in others if frame in m]
        occluders = np.any(hidden, axis=0) if hidden else None
        views.append(MaskView(poses[frame], world.intrinsics, own[frame], occluders))
    return views


@dataclass(frozen=True)
class Candidate:
    sample: ShapeSample
    registration: Registration


def register_sample(
    sample: ShapeSample,
    world: WorldModel,
    target: F64,
    views: Sequence[MaskView],
    cfg: RegistrationConfig,
    seed: int,
    support_z: float | None,
) -> Registration:
    poses = {f.frame_index: f.T_world_from_camera.to_numpy() for f in world.frames}
    R_start = poses[sample.frame_index][:3, :3] @ sample.T_camera_from_model.to_numpy()[:3, :3]
    return register(
        sample.vertices_model, sample.faces, R_start, target, views, cfg, seed, support_z
    )


def place(candidate: Candidate, support_z: float | None) -> tuple[F64, F64]:
    """World vertices of the kept sample, set down exactly on the support, and its rotation."""
    reg = candidate.registration
    T = reg.T_world_from_model.to_numpy()
    scaled = reg.scale * np.asarray(reg.stretch) * candidate.sample.vertices_model
    world = scaled @ T[:3, :3].T + T[:3, 3]
    if support_z is not None:
        world[:, 2] += support_z - float(np.min(world[:, 2]))
    return world, T[:3, :3]


def centroid(vertices: F64, faces: NDArray[np.int64]) -> F64:
    """Volume centroid of a closed mesh, else that of its convex hull."""
    import trimesh  # kept out of module import time

    mesh = trimesh.Trimesh(vertices, faces, process=False)
    solid = mesh if mesh.is_watertight and mesh.volume > 0 else mesh.convex_hull
    return np.asarray(solid.center_mass, dtype=float)


def write_mesh(
    path: Path, vertices: F64, faces: NDArray[np.int64], colors: NDArray[np.uint8] | None
) -> None:
    import trimesh

    mesh = trimesh.Trimesh(vertices, faces, vertex_colors=colors, process=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = mesh.export(file_type="ply", encoding="binary")
    if not isinstance(data, bytes):
        raise TypeError("trimesh returned no PLY bytes")
    path.write_bytes(data)


def proxy_hull(target: F64) -> tuple[F64, NDArray[np.int64]]:
    """Convex hull of the cluster points, in the world frame."""
    import trimesh

    hull = trimesh.convex.convex_hull(target)
    return np.asarray(hull.vertices, dtype=float), np.asarray(hull.faces, dtype=np.int64)


def support_height(world: WorldModel, target: F64, gap_m: float) -> float | None:
    """The table height when the cluster reaches down to it, else None (a stacked object)."""
    table_z = world.table_plane.offset_m
    return table_z if float(np.min(target[:, 2])) - table_z <= gap_m else None


def geometry_source(model: str) -> GeometrySource:
    if model not in get_args(GeometrySource):
        raise ValueError(f"shape model {model!r} is not a geometry source in schemas.py")
    return cast(GeometrySource, model)


def sample_record(c: Candidate) -> dict[str, Any]:
    r = c.registration
    return {
        "frame_index": c.sample.frame_index,
        "seed": c.sample.seed,
        "fit_score": r.fit_score,
        "inlier_fraction": r.inlier_fraction,
        "mask_inside_fraction": r.mask_inside_fraction,
        "scale": r.scale,
        "stretch": list(r.stretch),
        "cost": r.cost,
        "yaw_index": r.yaw_index,
    }


def shape_object(
    obj: ObjectInstance,
    world: WorldModel,
    run_dir: Path,
    out_dir: Path,
    masks: Mapping[str, Masks],
    model: ShapeModel,
    cfg: ShapeStageConfig,
    seed: int,
) -> tuple[RegisteredObject, dict[str, Any], str | None]:
    """One object's ``RegisteredObject``, its report entry and the checkpoint used, if any."""
    target = np.asarray(np.load(run_dir / obj.point_cluster_path), dtype=float).reshape(-1, 3)
    own = masks[obj.object_id]
    others = [m for k, m in masks.items() if k != obj.object_id]
    posed = {f.frame_index for f in world.frames}
    view_frames = spread([f for f in own if f in posed], cfg.max_views)
    views = mask_views(world, own, others, view_frames)
    support_z = support_height(world, target, cfg.support_gap_m)
    images = {f.frame_index: run_dir / f.image_path for f in world.frames}
    keyframes = [k for k in obj.keyframes if k in own and k in images][: cfg.keyframes_per_object]

    candidates: list[Candidate] = []
    failures: list[str] = []
    checkpoint = None
    if views and len(target):
        shape_views = [ShapeView(k, images[k], own[k]) for k in keyframes]
        samples = draw_samples(
            model,
            obj.object_id,
            shape_views,
            cfg.samples_per_keyframe,
            seed,
            failures,
            cfg.parallel_samples,
        )
        for sample in samples:
            reg = register_sample(sample, world, target, views, cfg.registration, seed, support_z)
            candidates.append(Candidate(sample, reg))
            checkpoint = sample.checkpoint
    ranked = sorted(candidates, key=lambda c: -c.registration.fit_score)
    finalists = [
        Candidate(
            c.sample,
            refine_stretch(
                c.sample.vertices_model, c.sample.faces, c.registration, target, views,
                cfg.registration, seed, support_z,
            ),
        )
        for c in ranked[: cfg.stretch_samples]
    ]  # fmt: skip
    for old, new in zip(ranked, finalists, strict=False):
        candidates[candidates.index(old)] = new
    best = max(finalists, key=lambda c: c.registration.fit_score, default=None)
    report: dict[str, Any] = {
        "views": view_frames,
        "support_z": support_z,
        "samples": [sample_record(c) for c in candidates],
        "failed_samples": failures,
    }
    mesh_rel = f"shape/{MESH_DIR}/{obj.object_id}.ply"

    if best is None or best.registration.fit_score < cfg.min_fit_score:
        if len(target) < 4:
            raise ValueError(f"{obj.object_id}: no sample fits and the cluster is too small")
        vertices, faces = proxy_hull(target)
        centre = centroid(vertices, faces)
        write_mesh(out_dir / MESH_DIR / f"{obj.object_id}.ply", vertices - centre, faces, None)
        T = np.eye(4)
        T[:3, 3] = centre
        report["kept"] = None
        flags = [Flag.PROXY_GEOMETRY, Flag.LOW_FIT]
        return (
            RegisteredObject(
                object_id=obj.object_id,
                mesh_path=mesh_rel,
                T_world_from_object=SE3.from_numpy(T),
                geometry_source="proxy_hull",
                fit_score=None,
                flags=flags,
            ),
            report,
            checkpoint,
        )

    world_vertices, rotation = place(best, support_z)
    centre = centroid(world_vertices, best.sample.faces)
    object_vertices = (world_vertices - centre) @ rotation  # rows: R.T @ (x - c)
    write_mesh(
        out_dir / MESH_DIR / f"{obj.object_id}.ply",
        object_vertices,
        best.sample.faces,
        best.sample.vertex_colors,
    )
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = centre
    report["kept"] = candidates.index(best)
    return (
        RegisteredObject(
            object_id=obj.object_id,
            mesh_path=mesh_rel,
            T_world_from_object=SE3.from_numpy(T),
            geometry_source=geometry_source(best.sample.model),
            fit_score=best.registration.fit_score,
        ),
        report,
        checkpoint,
    )


class ShapeStage:
    """Runner stage: ``WorldModel`` and ``Instances`` in, ``RegisteredObjects`` out. Its config
    is ``shape`` of ``configs/default.yaml``."""

    name: StageName = "shape"
    output_type = RegisteredObjects
    dependencies: tuple[StageName, ...] = ("reconstruction", "segmentation")
    extra_inputs: tuple[Path, ...] = ()

    def __init__(self, model: ModelFactory = default_model):
        self._model = model

    def run(self, ctx: StageContext) -> RegisteredObjects:
        world = ctx.load("reconstruction", WorldModel)
        instances = ctx.load("segmentation", Instances)
        cfg: dict[str, Any] = {"shape": dict(ctx.config)}
        stage_cfg = ShapeStageConfig.from_config(cfg)
        work = ctx.run_dir / "shape.work"
        work.mkdir(exist_ok=True)
        model = self._model(cfg, work)
        masks = {o.object_id: load_masks(ctx.run_dir, o) for o in instances.objects}

        objects, report = [], {}
        for obj in instances.objects:
            registered, entry, checkpoint = shape_object(
                obj, world, ctx.run_dir, ctx.out_dir, masks, model, stage_cfg, ctx.seed
            )
            if checkpoint is not None:
                ctx.record_checkpoint(model.name, checkpoint)
            objects.append(registered)
            report[obj.object_id] = entry
        (ctx.out_dir / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return RegisteredObjects(objects=objects)
