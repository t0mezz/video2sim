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
``route.json`` records the router's path (surface, prompt, listed objects, reason) and the raw
and repaired track counts with splits and merges.
Hosted answers are cached in ``segmentation.work/`` (outside the hashed stage folder, like
``reconstruction.work/``), so a rerun with the same inputs makes no call. Hosted clients are
built by ``clients`` and can be replaced in tests.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from video2sim.models.hosted import ResponseCache
from video2sim.models.vlm import VlmClient, make_vlm_client
from video2sim.runner import StageContext
from video2sim.schemas import (
    DiscoveryCheck,
    Flag,
    Instances,
    ObjectInstance,
    StageName,
    WorldModel,
)
from video2sim.stages.cluster_check import check_clusters
from video2sim.stages.depth_flags import DepthConfig, depth_signals, judge
from video2sim.stages.gap_fill import (
    FitPass,
    FoundBy,
    GapFillConfig,
    board_hull,
    execute_gap_fill,
    fit_pass,
    offset_tracks,
    residual_clusters,
    table_view_coverage,
)
from video2sim.stages.keyframes import KeyframeChoice, KeyframeConfig, keyframes_for_tracks
from video2sim.stages.lift import LiftConfig, load_points
from video2sim.stages.marker_filter import (
    MarkerFilterConfig,
    cached_tag_detector,
    filter_marker_tracks,
)
from video2sim.stages.material_pass import (
    MaterialClient,
    MaterialConfig,
    MaterialDecision,
    crop_for_vlm,
    decide_all,
    pick_frames,
)
from video2sim.stages.primitive_fit import (
    FitResult,
    PrimitiveConfig,
)
from video2sim.stages.scene_router import (
    Route,
    RouterClient,
    RouterConfig,
    compare_counts,
    route_scene,
    tracking_prompt,
)
from video2sim.stages.segmentation import BoolMask, sample_frame_indices
from video2sim.stages.shape_flags import ShapeConfig, judge_all, shape_signals
from video2sim.stages.track_repair import RepairConfig, RepairReport, repair_tracks
from video2sim.stages.tracking import (
    Track,
    TrackingConfig,
    VideoSamClient,
    encode_clip,
    make_tracking_client,
    track_objects,
    tracking_is_cached,
)
from video2sim.stages.verify_coverage import VerifyConfig, execute_verify

F64 = NDArray[np.float64]


@dataclass(frozen=True)
class SegmentationClients:
    """The hosted services of the stage. ``router`` and ``material`` may be None in a dry run: the
    router then falls back to the default prompt and every object is flagged as unverified."""

    router: RouterClient | None
    tracking: VideoSamClient
    material: MaterialClient | None
    verify: VlmClient | None = None


def default_clients(cfg: dict[str, Any]) -> SegmentationClients:
    """Clients for the hosted APIs. Keys come from the environment or ``.env``."""
    router, material = RouterConfig.from_config(cfg), MaterialConfig.from_config(cfg)
    verify = VerifyConfig.from_config(cfg)
    return SegmentationClients(
        make_vlm_client(cfg, router.timeout_s, router.retries, "router"),
        make_tracking_client(cfg),
        make_vlm_client(cfg, material.timeout_s, material.retries, "material"),
        make_vlm_client(cfg, verify.timeout_s, verify.retries, "verify"),
    )


def route_report(
    route: Route, raw: list[Track], repaired: list[Track], report: RepairReport
) -> dict[str, Any]:
    """JSON-serializable tracking summary for ``route.json``: the router's path and what
    tracking and 3D repair made of it. Counts and names only, so it is deterministic. The
    inventory keys appear only in coverage mode, so even mode stays byte-identical."""
    summary: dict[str, Any] = {
        "surface": route.surface,
        "prompt": route.prompt,
        "objects": list(route.objects),
        "reason": route.reason,
        "fell_back": route.fell_back,
        "tracks_raw": len(raw),
        "tracks_repaired": len(repaired),
        "splits": [[a, list(b)] for a, b in report.splits],
        "merges": [list(m) for m in report.merges],
    }
    if route.inventory:
        summary["inventory"] = [
            {"noun": i.noun, "count": i.count, "max_hidden_seen": i.max_hidden_seen}
            for i in route.inventory
        ]
        summary["view_frames"] = list(route.view_frames)
    return summary


def tracking_live_line(summary: dict[str, Any]) -> str:
    """The route line for the live log, from a ``route_report`` dict."""
    listed = len(summary["objects"]) if isinstance(summary["objects"], list) else 0
    splits = len(summary["splits"]) if isinstance(summary["splits"], list) else 0
    merges = len(summary["merges"]) if isinstance(summary["merges"], list) else 0
    reason = str(summary["reason"] or summary["surface"])
    repaired = int(summary["tracks_repaired"])
    return f"route {reason}: {listed} listed, {repaired} tracks ({merges} merges, {splits} splits)"


def flag_live_line(object_id: str, label: str, flags: list[str]) -> str:
    """One flagged object for the live log."""
    return f"{object_id} ({label}): {', '.join(flags)}"


def repaired_ids(report: RepairReport) -> set[int]:
    """Ids of tracks that were produced by a split or a merge."""
    ids = {i for _, parts in report.splits for i in parts}
    ids |= {min(group) for group in report.merges}
    return ids


def drop_reasons(first: FitPass, final: FitPass, kept_ids: set[int]) -> dict[int, str]:
    """Every dropped track's reason, first pass and gap-fill pass, by track id.

    A track the first pass dropped is not in the second pass's drops, and gap
    fill only receives the first pass's kept tracks, so reporting only the
    final pass would drop those tracks quietly. A track that ends up kept
    (recovered under the same id) is not reported as dropped.
    """
    merged = {**first.drops, **final.drops}
    return {tid: merged[tid] for tid in sorted(merged) if tid not in kept_ids}


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
        cv2.imwrite(str(mask_dir / f"{frame:05d}.png"), mask.full().astype(np.uint8) * 255)
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
    found_by: Literal["generic", "noun", "cluster_box"] = "generic",
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
        found_by=found_by,
        category=answer.category if answer is not None else None,
        material_class=answer.material_class if answer is not None else None,
        finish=answer.finish if answer is not None else None,
        hollow=answer.hollow if answer is not None else None,
        rigidity=answer.rigidity if answer is not None else None,
        material_confidence=answer.confidence if answer is not None else None,
    )


PALETTE: tuple[tuple[int, int, int], ...] = (  # BGR, distinct on print and screen
    (0, 0, 255),
    (0, 165, 255),
    (0, 255, 0),
    (255, 0, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
    (128, 0, 255),
    (255, 128, 0),
    (0, 128, 255),
    (128, 255, 0),
    (255, 0, 128),
)


def _draw_masks(
    overlay: NDArray[np.uint8],
    masks: Mapping[int, BoolMask],
    colors: Mapping[int, tuple[int, int, int]],
) -> list[int]:
    """Track masks tinted, contoured and numbered onto ``overlay``; ids drawn, sorted.

    Masks at another resolution than the frame are resampled nearest-neighbour,
    explicitly, like the tracking masks. Colours come from ``track_colors`` so one
    track keeps its colour across views and frames.
    """
    import cv2  # heavy import, kept out of module import time

    drawn: list[int] = []
    for tid in sorted(masks):
        mask = masks[tid]
        if mask.shape != overlay.shape[:2]:
            small = cv2.resize(
                mask.astype("uint8"),
                (overlay.shape[1], overlay.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        else:
            small = mask
        if not small.any():
            continue
        color = colors[tid]
        tint = overlay.copy()
        tint[small] = color
        cv2.addWeighted(tint, 0.22, overlay, 0.78, 0, overlay)
        contours, _ = cv2.findContours(
            small.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, color, 2)
        # The label sits inside the object, at the mask's median point, sized
        # with the image so it stays legible at the size the model receives.
        ys, xs = small.nonzero()
        label = str(tid)
        height, width = overlay.shape[:2]
        scale = min(max(max(height, width) / 1024.0, 0.5), 2.0)
        font_scale = 0.9 * scale
        thickness = max(1, round(2 * scale))
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        pad = max(3, round(3 * scale))
        cx, cy = int(np.median(xs)), int(np.median(ys))
        x = min(max(cx - tw // 2, 0), max(width - tw - 2 * pad, 0))
        baseline = min(max(cy + th // 2, th + pad), height - pad)
        cv2.rectangle(
            overlay, (x, baseline - th - pad), (x + tw + 2 * pad, baseline + pad), color, -1
        )
        blue, green, red = (float(c) for c in color)
        ink = (0, 0, 0) if 0.114 * blue + 0.587 * green + 0.299 * red > 140 else (255, 255, 255)
        cv2.putText(
            overlay,
            label,
            (x + pad, baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            ink,
            thickness,
        )
        drawn.append(tid)
    return drawn


def track_colors(ids: Sequence[int]) -> dict[int, tuple[int, int, int]]:
    """Legend colour per track id, in sorted-id order."""
    ordered = sorted(ids)
    return {tid: PALETTE[pos % len(PALETTE)] for pos, tid in enumerate(ordered)}


def render_tracks_video(
    world: WorldModel,
    run_dir: Path,
    tracks: Sequence[Track],
    kept: set[int],
    notes: Mapping[int, str],
    out_path: Path,
    fps: float = 5.0,
) -> int:
    """Every track over every frame, tinted and numbered, with a fate legend.

    ``notes`` names each track's fate (``kept …`` or the drop reason); kept ids
    show green, the rest red. Returns the frames written.
    """
    import cv2  # heavy import, kept out of module import time

    first = cv2.imread(str(run_dir / world.frames[0].image_path))
    if first is None:
        raise ValueError(f"cannot read {world.frames[0].image_path}")
    height, width = first.shape[:2]
    colors = track_colors([t.track_id for t in tracks])
    by_frame: dict[int, list[Track]] = {}
    for track in tracks:
        for frame_index in track.masks:
            by_frame.setdefault(frame_index, []).append(track)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # VideoWriter_fourcc is missing from the cv2 stubs.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    written = 0
    try:
        for frame in world.frames:
            image = cv2.imread(str(run_dir / frame.image_path))
            if image is None:
                raise ValueError(f"cannot read {frame.image_path}")
            overlay: NDArray[np.uint8] = np.asarray(image.copy(), dtype=np.uint8)
            present = {
                t.track_id: t.masks[frame.frame_index].full()
                for t in by_frame.get(frame.frame_index, [])
            }
            _draw_masks(overlay, present, colors)
            y = 24
            for tid in sorted(colors):
                color = (0, 200, 0) if tid in kept else (60, 60, 220)
                line = f"T{tid}: {notes.get(tid, '?')}"
                (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(overlay, (8, y - th - 6), (8 + tw + 8, y + 2), (0, 0, 0), -1)
                cv2.putText(overlay, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
                y += 22
            writer.write(overlay)
            written += 1
    finally:
        writer.release()
    return written


class SegmentationStage:
    """Runner stage: ``WorldModel`` in, ``Instances`` out. Its config is ``segmentation`` of
    ``configs/default.yaml``."""

    name: StageName = "segmentation"
    output_type = Instances
    dependencies: tuple[StageName, ...] = ("reconstruction",)
    extra_inputs: tuple[Path, ...] = ()

    shared_config = ("vlm", "capture")  # the model endpoints; the tag family of the marker filter

    def __init__(
        self,
        clients: Callable[[dict[str, Any]], SegmentationClients] = default_clients,
        on_start: Callable[[], None] | None = None,
    ):
        """``on_start`` runs first, for work that should overlap this stage (D-032)."""
        self._clients = clients
        self._on_start = on_start

    def run(self, ctx: StageContext) -> Instances:
        if self._on_start is not None:
            self._on_start()
        world = ctx.load("reconstruction", WorldModel)
        cfg: dict[str, Any] = {"segmentation": dict(ctx.config), **ctx.shared}
        work = ctx.run_dir / "segmentation.work"
        work.mkdir(exist_ok=True)
        clients = self._clients(cfg)
        rcfg = RouterConfig.from_config(cfg)
        lift_cfg, depth_cfg = LiftConfig.from_config(cfg), DepthConfig.from_config(cfg)
        prim_cfg = PrimitiveConfig.from_config(cfg)
        material_cfg, shape_cfg = MaterialConfig.from_config(cfg), ShapeConfig.from_config(cfg)
        scene_flags: list[str] = []

        cloud = load_points(ctx.run_dir / world.point_cloud_path)
        route = route_scene(
            world,
            ctx.run_dir,
            clients.router,
            ResponseCache(work / "router_cache"),
            rcfg,
            cloud,
        )
        if route.fell_back:
            scene_flags.append(f"router fell back to the {route.surface} prompt: {route.reason}")
        tcfg = replace(TrackingConfig.from_config(cfg), prompt=tracking_prompt(route, rcfg))
        marker_cfg = MarkerFilterConfig.from_config(cfg)
        raw = track_objects(
            world,
            ctx.run_dir,
            clients.tracking,
            ResponseCache(work / "tracking_cache"),
            tcfg,
            work / "tracking_work",
        )
        all_tracks = list(raw)
        tag_detect = cached_tag_detector(marker_cfg.family)
        raw, marker_report = filter_marker_tracks(raw, world, ctx.run_dir, marker_cfg, tag_detect)
        for track_id, reason in marker_report.dropped.items():
            scene_flags.append(f"track {track_id} dropped as a marker: {reason}")
            ctx.report(f"dropped track {track_id}: marker ({reason})")
        poses = {f.frame_index: f.T_world_from_camera for f in world.frames}
        repaired, report = repair_tracks(
            raw, poses, world.intrinsics, world.table_plane, RepairConfig.from_config(cfg)
        )
        summary = route_report(route, raw, repaired, report)
        ctx.report(tracking_live_line(summary))

        first = fit_pass(repaired, world, lift_cfg, prim_cfg, cloud)
        gcfg = GapFillConfig.from_config(cfg)
        listed = sum(i.count for i in route.inventory) if route.inventory else len(route.objects)
        final = first
        discovery: DiscoveryCheck | None = None
        prov: dict[int, FoundBy] = {t.track_id: "generic" for t in first.kept}
        gap_flag_reasons: dict[int, str] = {}
        if gcfg.enabled:
            # Residual clusters -> VLM check -> one box request (steps 2-7).
            # No cluster: no extra VLM call and no extra video request.
            hull = board_hull(world, ctx.run_dir, marker_cfg.family, tag_detect)
            if hull is None:
                scene_flags.append(
                    "board footprint unavailable: no tag could be triangulated; "
                    "board exclusion skipped"
                )
            assert first.vote is not None
            clusters = residual_clusters(
                world,
                cloud,
                first.vote,
                gcfg.residual_params(lift_cfg.min_views),
                lift_cfg.table_margin_m,
                lift_cfg.max_height_m,
                hull,
            )
            order = [f.frame_index for f in world.frames]
            view_frames = list(route.view_frames) or [
                order[i] for i in sample_frame_indices(len(order), rcfg.n_frames)
            ]
            coverage = table_view_coverage(
                world,
                cloud,
                view_frames,
                order,
                rcfg.views,
                rcfg.views.cell_m,
                rcfg.views.min_views_per_cell,
                gcfg.residual_params(lift_cfg.min_views),
                lift_cfg.table_margin_m,
            )
            if not clusters:
                discovery = DiscoveryCheck(
                    view_frames=view_frames,
                    view_coverage=coverage,
                    listed=listed,
                    tracked=len(first.kept),
                    missing_suspected=listed > len(first.kept),
                    unmatched=[],
                    unexplained_clusters=0,
                    recovered=0,
                    video_requests=1,
                )
                summary["gap_fill"] = {
                    "listed": listed,
                    "unexplained_clusters": 0,
                    "clusters": [],
                    "video_requests": 1,
                    "board_flag": hull is None,
                }
            else:
                checks = check_clusters(
                    world,
                    ctx.run_dir,
                    clusters,
                    first.kept,
                    clients.router,
                    ResponseCache(work / "cluster_check_cache"),
                    work / "cluster_check",
                    gcfg.check,
                    rcfg.noun_denylist,
                )
                for cluster_check in checks:
                    if cluster_check.verdict.status == "unchecked":
                        scene_flags.append(
                            f"cluster {cluster_check.index} unchecked: "
                            f"{cluster_check.verdict.reason}"
                        )
                tracking_cache = ResponseCache(work / "tracking_cache")
                # One clip for every gap-fill request: the generic request
                # already encoded it, unless its answer was cached.
                clip_path = work / "tracking_work" / "clip.mp4"
                if not clip_path.is_file():
                    clip_path.parent.mkdir(parents=True, exist_ok=True)
                    encode_clip(
                        [ctx.run_dir / f.image_path for f in world.frames],
                        tcfg.fps,
                        clip_path,
                    )

                def track_fn(prompt: str, offset: int) -> tuple[list[Track], bool]:
                    rtcfg = replace(tcfg, prompt=prompt)
                    cached = tracking_is_cached(world, ctx.run_dir, tracking_cache, rtcfg)
                    tracks = track_objects(
                        world,
                        ctx.run_dir,
                        clients.tracking,
                        tracking_cache,
                        rtcfg,
                        work / "tracking_work",
                        clip_path=clip_path,
                    )
                    return offset_tracks(tracks, offset), cached

                final, discovery, gap_notes, prov = execute_gap_fill(
                    world,
                    ctx.run_dir,
                    checks,
                    gcfg,
                    lift_cfg,
                    prim_cfg,
                    RepairConfig.from_config(cfg),
                    marker_cfg,
                    cloud,
                    poses,
                    first.kept,
                    listed,
                    view_frames,
                    coverage,
                    track_fn,
                    first=first,
                    detect=tag_detect,
                )
                gap_notes["board_flag"] = hull is None
                summary["gap_fill"] = gap_notes
                gap_flag_reasons = {int(k): v for k, v in gap_notes["flag_reasons"].items()}
                for note in gap_notes["report_lines"]:
                    ctx.report(note)
        vcfg = VerifyConfig.from_config(cfg)
        if vcfg.enabled:
            # Coverage verification: one VLM look over whole-scene views with
            # the kept tracks outlined, then noun prompts for listed nouns the
            # VLM still sees untracked. Survivors join the kept tracks below.
            order = [f.frame_index for f in world.frames]
            verify_positions = sample_frame_indices(len(order), vcfg.views)
            verify_cache = ResponseCache(work / "tracking_cache")
            verify_clip = work / "tracking_work" / "clip.mp4"
            if not verify_clip.is_file():
                verify_clip.parent.mkdir(parents=True, exist_ok=True)
                encode_clip(
                    [ctx.run_dir / f.image_path for f in world.frames],
                    tcfg.fps,
                    verify_clip,
                )

            def verify_track_fn(prompt: str, offset: int) -> tuple[list[Track], bool]:
                rtcfg = replace(tcfg, prompt=prompt)
                cached = tracking_is_cached(world, ctx.run_dir, verify_cache, rtcfg)
                tracks = track_objects(
                    world,
                    ctx.run_dir,
                    clients.tracking,
                    verify_cache,
                    rtcfg,
                    work / "tracking_work",
                    clip_path=verify_clip,
                )
                return offset_tracks(tracks, offset), cached

            verify_listed = (
                [i.noun for i in route.inventory] if route.inventory else list(route.objects)
            )
            final, discovery, verify_notes, prov, vreasons, verify_new = execute_verify(
                world,
                ctx.run_dir,
                list(final.kept),
                list(dict.fromkeys(verify_listed)),
                vcfg,
                gcfg,
                lift_cfg,
                prim_cfg,
                RepairConfig.from_config(cfg),
                marker_cfg,
                cloud,
                poses,
                verify_track_fn,
                final,
                discovery,
                prov,
                verify_positions,
                clients.verify,
                ResponseCache(work / "verify_cache"),
                work / "verify_views",
                rcfg.noun_denylist,
                tag_detect,
            )
            all_tracks.extend(verify_new)
            gap_flag_reasons.update({int(k): v for k, v in vreasons.items()})
            summary["verify"] = verify_notes
            for note in verify_notes["report_lines"]:
                ctx.report(note)
            for flag in verify_notes["scene_flags"]:
                scene_flags.append(flag)
        kept = final.kept
        drops = drop_reasons(first, final, {t.track_id for t in kept})
        for track_id, reason in drops.items():
            scene_flags.append(f"track {track_id} dropped: {reason}")
            ctx.report(f"dropped track {track_id}: {reason}")
        if gcfg.enabled and gcfg.save_video:
            by_id = {t.track_id: t for t in all_tracks}
            for track in final.kept:
                by_id[track.track_id] = track
            notes = {}
            for tid in sorted(by_id):
                if tid in marker_report.dropped:
                    notes[tid] = f"marker: {marker_report.dropped[tid]}"
                elif tid in drops:
                    notes[tid] = drops[tid]
                else:
                    notes[tid] = "kept"
            preview = ctx.out_dir / "tracks_preview.mp4"
            frames = render_tracks_video(
                world,
                ctx.run_dir,
                [by_id[i] for i in sorted(by_id)],
                {t.track_id for t in kept},
                notes,
                preview,
                tcfg.fps,
            )
            ctx.report(f"tracks preview: {frames} frames in {preview.name}")
        check = compare_counts(route, len(kept))
        if check.missing_suspected:
            scene_flags.append(f"possible missing object: {check.reason}")
        (ctx.out_dir / "route.json").write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
        if not kept:
            scene_flags.append("no objects tracked")
            return Instances(objects=[], flags=scene_flags, discovery=discovery)
        hulls, sparse, fits = final.hulls, final.sparse, final.fits
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
            found_by = prov.get(t.track_id, "generic")
            gap_reason = gap_flag_reasons.get(t.track_id)
            if gap_reason is not None:
                reasons.append(gap_reason)
            objects.append(
                build_object(
                    object_id,
                    mask_dir,
                    cluster_path,
                    keyframes.get(t.track_id, []),
                    flags,
                    reasons,
                    decision,
                    found_by,
                )
            )
            if flags:
                label = decision.answer.category if decision.answer is not None else "object"
                ctx.report(flag_live_line(object_id, label, [f.value for f in flags]))
        return Instances(objects=objects, flags=scene_flags, discovery=discovery)
