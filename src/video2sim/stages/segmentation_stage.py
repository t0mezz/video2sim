"""Stage 3 as a runner stage (P2.6): ``WorldModel`` in, ``Instances`` out.

The pieces are the ones built in P2.1 to P2.5, in this order:

1. ``scene_router``: one VLM request picks the SAM prompt from the pool and lists the objects.
2. ``tracking`` and ``track_repair``: one SAM 3.1 video request, then identity repair in 3D.
3. ``lift`` and ``primitive_fit``: the SfM points that the masks agree on, without the outermost
   strays, and a box, cylinder or sphere fitted to the masks from those points. The fitted
   cluster is written to ``point_cluster_path`` (4.3% median box error on the easy tier, human
   decision 2026-10-29): the primitive's outline points, or their union with the voted points
   when no primitive fits the masks well (with a reason), or the points alone when too few
   masks are whole (with a reason). ``fit.json`` records the kind, size and mask cost. The voted
   points (``obj_NNN_points.npy``) and the visual hull (``obj_NNN_hull.npy``) are written beside
   it, outside the typed output. A track with no sparse points falls back to its hull, with a
   reason. The shape flags use the hull, which has a volume.
4. ``keyframes``: the best frames per object.
5. ``depth_flags``, ``material_pass`` and ``shape_flags``: the failure flags and the material
   answer per object.

A track with no hull is dropped and named in ``Instances.flags``, so the objects in the
output all have geometry; an object the router listed but no track covers is named there too.
Hosted answers are cached in ``segmentation.work/`` (outside the hashed stage folder, like
``reconstruction.work/``), so a rerun with the same inputs makes no call. Hosted clients are
built by ``clients`` and can be replaced in tests.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from video2sim.models.hosted import ResponseCache, read_secret
from video2sim.runner import StageContext
from video2sim.schemas import Flag, Instances, ObjectInstance, StageName, WorldModel
from video2sim.stages.depth_flags import DepthConfig, depth_signals, judge
from video2sim.stages.keyframes import KeyframeChoice, KeyframeConfig, keyframes_for_tracks
from video2sim.stages.lift import LiftConfig, lift_points, lift_tracks, load_points
from video2sim.stages.material_pass import (
    BasetenMaterialClient,
    MaterialClient,
    MaterialConfig,
    MaterialDecision,
    crop_for_vlm,
    decide_all,
    pick_frames,
)
from video2sim.stages.primitive_fit import FitResult, PrimitiveConfig, fit_clusters
from video2sim.stages.scene_router import (
    RouterClient,
    RouterConfig,
    compare_counts,
    route_scene,
    tracking_prompt,
)
from video2sim.stages.shape_flags import ShapeConfig, judge_all, shape_signals
from video2sim.stages.track_repair import RepairConfig, RepairReport, repair_tracks
from video2sim.stages.tracking import (
    MetaVideoSamClient,
    Track,
    TrackingConfig,
    VideoSamClient,
    track_objects,
)

F64 = NDArray[np.float64]


@dataclass(frozen=True)
class SegmentationClients:
    """The hosted services of the stage. ``router`` and ``material`` may be None in a dry run: the
    router then falls back to the default prompt and every object is flagged as unverified."""

    router: RouterClient | None
    tracking: VideoSamClient
    material: MaterialClient | None


def default_clients(cfg: dict[str, Any]) -> SegmentationClients:
    """Clients for the hosted APIs. Keys come from the environment or ``.env``."""
    router, material = RouterConfig.from_config(cfg), MaterialConfig.from_config(cfg)
    tracking = TrackingConfig.from_config(cfg)
    baseten = read_secret("BASETEN_API_KEY")
    return SegmentationClients(
        BasetenMaterialClient(baseten, router.timeout_s, router.retries),
        MetaVideoSamClient(read_secret("META_MODEL_API_KEY"), tracking.timeout_s),
        BasetenMaterialClient(baseten, material.timeout_s, material.retries),
    )


def repaired_ids(report: RepairReport) -> set[int]:
    """Ids of tracks that were produced by a split or a merge."""
    ids = {i for _, parts in report.splits for i in parts}
    ids |= {min(group) for group in report.merges}
    return ids


def object_flags(
    shape_flags: tuple[Flag, ...],
    decision: MaterialDecision,
    repaired: bool,
) -> list[Flag]:
    """The flags of one object, each at most once, in a fixed order."""
    flags = list(shape_flags)
    if decision.flag is not None:
        flags.append(decision.flag)
    if decision.non_rigid is not None:
        flags.append(Flag.NON_RIGID)
    if repaired:
        flags.append(Flag.ID_SWITCH_REPAIRED)
    return list(dict.fromkeys(flags))


def write_object_files(
    out_dir: Path,
    object_id: str,
    track: Track,
    cluster: F64,
    extra: Mapping[str, F64] | None = None,
) -> tuple[str, str]:
    """Masks as one PNG per frame and the cluster as ``.npy``; relative paths of both. Each entry
    of ``extra`` is written beside the cluster as ``<object_id>_<name>.npy``, outside the typed
    output."""
    import cv2

    mask_dir = out_dir / "masks" / object_id
    mask_dir.mkdir(parents=True, exist_ok=True)
    for frame, mask in sorted(track.masks.items()):
        cv2.imwrite(str(mask_dir / f"{frame:05d}.png"), mask.astype(np.uint8) * 255)
    cluster_dir = out_dir / "clusters"
    cluster_dir.mkdir(parents=True, exist_ok=True)
    np.save(cluster_dir / f"{object_id}.npy", cluster)
    for name, points in (extra or {}).items():
        np.save(cluster_dir / f"{object_id}_{name}.npy", points)
    return f"segmentation/masks/{object_id}", f"segmentation/clusters/{object_id}.npy"


def fit_report(fits: Mapping[int, FitResult]) -> dict[str, Any]:
    """``fit.json``: how each fitted cluster was made, by object id."""
    out: dict[str, Any] = {}
    for track_id, fit in sorted(fits.items()):
        p = fit.primitive
        out[f"obj_{track_id:03d}"] = {
            "source": fit.source,
            "reason": fit.reason,
            "kind": p.kind if p else None,
            "size_m": list(p.size_m) if p else None,
            "center_xy_m": list(p.center_xy_m) if p else None,
            "yaw_rad": p.yaw_rad if p else None,
            "mask_cost": p.cost if p else None,
            "views": p.views if p else None,
        }
    return out


def choose_cluster(
    fit: FitResult | None, points: F64 | None, hull: F64
) -> tuple[F64, dict[str, F64], str | None]:
    """The cluster for ``point_cluster_path``, the files written beside it, and a note for the
    QA report when the cluster is not a clean primitive fit."""
    if fit is None or points is None:
        return hull, {}, "no sparse points agree with the masks: the hull is the cluster"
    note = None if fit.source == "primitive" else f"cluster: {fit.reason}"
    return fit.cluster, {"points": points, "hull": hull}, note


def build_object(
    object_id: str,
    mask_dir: str,
    cluster_path: str,
    keyframes: list[KeyframeChoice],
    flags: list[Flag],
    reasons: list[str],
    decision: MaterialDecision,
) -> ObjectInstance:
    answer = decision.answer
    return ObjectInstance(
        object_id=object_id,
        label=answer.category if answer is not None else "object",
        mask_dir=mask_dir,
        point_cluster_path=cluster_path,
        keyframes=[k.frame_index for k in keyframes],
        flags=flags,
        flag_reasons=reasons,
        category=answer.category if answer is not None else None,
        material_class=answer.material_class if answer is not None else None,
        finish=answer.finish if answer is not None else None,
        hollow=answer.hollow if answer is not None else None,
        rigidity=answer.rigidity if answer is not None else None,
        material_confidence=answer.confidence if answer is not None else None,
    )


class SegmentationStage:
    """Runner stage: ``WorldModel`` in, ``Instances`` out. Its config is ``segmentation`` of
    ``configs/default.yaml``."""

    name: StageName = "segmentation"
    output_type = Instances
    dependencies: tuple[StageName, ...] = ("reconstruction",)
    extra_inputs: tuple[Path, ...] = ()

    def __init__(self, clients: Callable[[dict[str, Any]], SegmentationClients] = default_clients):
        self._clients = clients

    def run(self, ctx: StageContext) -> Instances:
        world = ctx.load("reconstruction", WorldModel)
        cfg: dict[str, Any] = {"segmentation": dict(ctx.config)}
        work = ctx.run_dir / "segmentation.work"
        work.mkdir(exist_ok=True)
        clients = self._clients(cfg)
        rcfg = RouterConfig.from_config(cfg)
        lift_cfg, depth_cfg = LiftConfig.from_config(cfg), DepthConfig.from_config(cfg)
        material_cfg, shape_cfg = MaterialConfig.from_config(cfg), ShapeConfig.from_config(cfg)
        scene_flags: list[str] = []

        route = route_scene(
            world, ctx.run_dir, clients.router, ResponseCache(work / "router_cache"), rcfg
        )
        if route.fell_back:
            scene_flags.append(f"router fell back to the {route.surface} prompt: {route.reason}")
        tcfg = replace(TrackingConfig.from_config(cfg), prompt=tracking_prompt(route, rcfg))
        raw = track_objects(
            world,
            ctx.run_dir,
            clients.tracking,
            ResponseCache(work / "tracking_cache"),
            tcfg,
            work / "tracking_work",
        )
        poses = {f.frame_index: f.T_world_from_camera for f in world.frames}
        repaired, report = repair_tracks(
            raw, poses, world.intrinsics, world.table_plane, RepairConfig.from_config(cfg)
        )

        hulls, hull_report = lift_tracks(repaired, world, lift_cfg)
        kept = [t for t in repaired if t.track_id in hulls]
        for t in repaired:
            if t.track_id not in hulls:
                reason = hull_report.dropped.get(t.track_id, "no point cluster")
                scene_flags.append(f"track {t.track_id} dropped: {reason}")
        check = compare_counts(route, len(kept))
        if check.missing_suspected:
            scene_flags.append(f"possible missing object: {check.reason}")
        if not kept:
            scene_flags.append("no objects tracked")
            return Instances(objects=[], flags=scene_flags)

        cloud = load_points(ctx.run_dir / world.point_cloud_path)
        sparse, _ = lift_points(kept, world, cloud, lift_cfg)
        fits = fit_clusters(kept, sparse, world, lift_cfg, PrimitiveConfig.from_config(cfg))
        ctx.out_dir.mkdir(parents=True, exist_ok=True)
        (ctx.out_dir / "fit.json").write_text(
            json.dumps(fit_report(fits), indent=2) + "\n", encoding="utf-8"
        )
        centers = {i: np.median(p, axis=0) for i, p in hulls.items() if len(p)}
        keyframes, kf_report = keyframes_for_tracks(
            kept, world, ctx.run_dir, centers, KeyframeConfig.from_config(cfg)
        )

        by_id = {t.track_id: t for t in kept}
        signals = depth_signals(kept, world, ctx.run_dir, hulls, lift_cfg)
        verdicts = [judge(s, depth_cfg) for s in signals]

        def crops(track_id: int) -> list[Path]:
            frames = pick_frames(
                by_id[track_id],
                [k.frame_index for k in keyframes.get(track_id, [])],
                material_cfg.keyframes,
            )
            return [
                crop_for_vlm(
                    ctx.run_dir,
                    world,
                    by_id[track_id],
                    f,
                    material_cfg.crop_pad_px,
                    work / "material_crops" / f"track_{track_id:03d}_f{f:05d}.png",
                )
                for f in frames
            ]

        decisions = {
            d.track_id: d
            for d in decide_all(
                signals,
                verdicts,
                material_cfg,
                crops,
                clients.material,
                ResponseCache(work / "material_vlm_cache") if clients.material else None,
            )
        }
        shape = judge_all(shape_signals(kept, hulls, shape_cfg), shape_cfg)
        repaired_tracks = repaired_ids(report)

        objects: list[ObjectInstance] = []
        for t in kept:
            object_id = f"obj_{t.track_id:03d}"
            decision, geometry = decisions[t.track_id], shape[t.track_id]
            flags = object_flags(geometry.flags, decision, t.track_id in repaired_tracks)
            reasons = list(geometry.reasons)
            if decision.flag is not None:
                reasons.append(decision.reason)
            if decision.non_rigid is not None:
                reasons.append(decision.non_rigid)
            if t.track_id in repaired_tracks:
                reasons.append("track produced by an identity split or merge")
            if t.track_id in kf_report.notes:
                reasons.append(f"keyframes: {kf_report.notes[t.track_id]}")
            cluster, extra, note = choose_cluster(
                fits.get(t.track_id), sparse.get(t.track_id), hulls[t.track_id]
            )
            if note is not None:
                reasons.append(note)
            mask_dir, cluster_path = write_object_files(ctx.out_dir, object_id, t, cluster, extra)
            objects.append(
                build_object(
                    object_id,
                    mask_dir,
                    cluster_path,
                    keyframes.get(t.track_id, []),
                    flags,
                    reasons,
                    decision,
                )
            )
        return Instances(objects=objects, flags=scene_flags)
