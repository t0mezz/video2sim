"""Residual gap fill after tracking (S12.9 redesign of 9.6/9.7).

Residual clusters -> VLM check -> one box request. The sparse cloud holds
point clusters no kept track explains (the per-point vote of ``lift``);
each cluster is checked by the VLM on cropped views, and only accepted
clusters earn a video request: a single box request for all of them, with
a noun request in parallel only for an accepted cluster whose box track
fails the geometric gate. Everything that survives goes through the same
repair, lift, fit and duplicate rules as generic tracks.

A scene with no residual cluster makes no extra VLM call and no extra
video request. With ``enabled: false`` nothing here runs and the stage
output is byte-identical.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from video2sim.models.sam_prompts import ObjectPrompt, build_object_prompt
from video2sim.schemas import SE3, DiscoveryCheck, WorldModel
from video2sim.stages.cluster_check import ClusterCheck, ClusterCheckConfig
from video2sim.stages.lift import (
    LiftConfig,
    PointVote,
    lift_points,
    lift_tracks,
    point_vote,
)
from video2sim.stages.marker_filter import (
    MarkerFilterConfig,
    TagDetector,
    cached_tag_detector,
    filter_marker_tracks,
)
from video2sim.stages.primitive_fit import (
    FitResult,
    PrimitiveConfig,
    drop_duplicate_fits,
    fit_clusters,
)
from video2sim.stages.track_repair import RepairConfig, RepairReport, _groups, repair_tracks
from video2sim.stages.tracking import Track
from video2sim.stages.view_select import ViewConfig, compute_visibility

F64 = NDArray[np.float64]
Box = tuple[int, int, int, int]  # inclusive x1, y1, x2, y2 in media pixels
FoundBy = Literal["generic", "noun", "cluster_box"]

CLUSTER_ID_BASE = 2000  # box-prompt ids: 2000 + cluster index (at most 16 clusters)
NOUN_ID_BASE = 10000  # noun requests: NOUN_ID_BASE + NOUN_ID_STEP * k; clear of the box ids
NOUN_ID_STEP = 1000  # per-request stride; a request returns far fewer tracks than this


@dataclass(frozen=True)
class ClusterParams:
    """Residual grouping: KD-tree link distance and the smallest group that counts."""

    radius_m: float
    min_points: int


@dataclass(frozen=True)
class GapFillConfig:
    """Values from ``segmentation.tracking.gap_fill`` in ``configs/default.yaml``."""

    enabled: bool
    max_rounds: int
    max_noun_requests: int
    max_video_requests: int
    max_workers: int
    cluster: ClusterParams
    check: ClusterCheckConfig
    merge_iou: float
    merge_new_iou: float
    part_inclusion: float
    save_video: bool = True  # tracks_preview.mp4 in the stage folder

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> GapFillConfig:
        g = cfg["segmentation"]["tracking"]["gap_fill"]
        c = g["cluster"]
        out = cls(
            enabled=bool(g["enabled"]),
            max_rounds=int(g["max_rounds"]),
            max_noun_requests=int(g["max_noun_requests"]),
            max_video_requests=int(g["max_video_requests"]),
            max_workers=int(g.get("max_workers", 4)),
            cluster=ClusterParams(
                radius_m=float(c["radius_m"]),
                min_points=int(c["min_points"]),
            ),
            check=ClusterCheckConfig.from_config(cfg),
            merge_iou=float(g["merge_iou"]),
            merge_new_iou=float(g["merge_new_iou"]),
            part_inclusion=float(g["part_inclusion"]),
            save_video=bool(g.get("save_video", True)),
        )
        if out.max_rounds != 1:
            raise ValueError("only one gap-fill round is implemented")
        if out.max_noun_requests < 0 or out.max_video_requests < 1 or out.max_workers < 1:
            raise ValueError("bad gap_fill request caps")
        if out.cluster.radius_m <= 0 or out.cluster.min_points < 1:
            raise ValueError("bad gap_fill cluster grouping")
        for v in (out.merge_iou, out.merge_new_iou, out.part_inclusion):
            if not 0.0 < v <= 1.0:
                raise ValueError("gap_fill IoU gates must be in (0, 1]")
        if out.merge_new_iou > out.merge_iou:
            raise ValueError("gap_fill merge_new_iou must not clear merge_iou")
        return out

    def residual_params(self, min_views: int) -> ResidualParams:
        """The residual-cluster thresholds, from the cluster and check config."""
        return ResidualParams(
            explain_ratio=self.check.explain_ratio,
            min_views=min_views,
            radius_m=self.cluster.radius_m,
            min_points=self.cluster.min_points,
            table_cell_m=self.check.table_cell_m,
            table_dilate_m=self.check.table_dilate_m,
            board_margin_m=self.check.board_margin_m,
        )


@dataclass(frozen=True)
class GapRequest:
    """One gap-fill video request's tracks, ready to merge."""

    tracks: tuple[Track, ...]
    found_by: FoundBy
    note: str  # provenance recorded on surviving tracks, e.g. the noun


@dataclass(frozen=True)
class MergeResult:
    tracks: list[Track]
    notes: dict[int, tuple[str, ...]]  # surviving track id -> provenance notes
    dropped: dict[int, str]  # dropped new track id -> reason
    remapped: tuple[tuple[int, int], ...] = ()  # (old, new) ids when a collision remapped


def offset_tracks(tracks: Sequence[Track], offset: int) -> list[Track]:
    """Tracks with ids shifted so requests never collide (see the id bases above)."""
    return [Track(t.track_id + offset, dict(t.masks)) for t in tracks]


@dataclass(frozen=True)
class ResidualParams:
    """Thresholds of the residual-cluster step (steps 1-3); guesses until tuned on dev.

    The stage reads these from ``gap_fill.cluster_check`` (see
    ``ClusterCheckConfig``); the defaults here only serve tests and callers
    that build the params directly.
    """

    explain_ratio: float = 0.5  # tau_explain: best vote at or above this explains a point
    min_views: int = 5  # the winning track must have seen the point this often
    radius_m: float = 0.03  # connected-component link distance
    min_points: int = 12  # smaller groups are noise
    table_cell_m: float = 0.05  # grid cell of the table-region flood fill
    table_dilate_m: float = 0.10  # the table region grows by this much
    board_margin_m: float = 0.02  # the board footprint grows by this much


def table_region(
    points: F64, world: WorldModel, table_margin_m: float, cell_m: float, dilate_m: float
) -> NDArray[np.bool_]:
    """Per-point membership in the table region.

    Points within ``table_margin_m`` of the table plane are gridded in XY;
    the connected occupied piece that contains the world origin (which sits
    on the reference tag) is the table, widened by ``dilate_m``. A wall or a
    far desk at table height forms its own piece and never joins, so its
    points are outside. With no near-plane point at all, nothing is inside.
    """
    from scipy import ndimage  # type: ignore[import-untyped]  # heavy import

    inside = np.zeros(len(points), dtype=bool)
    if not len(points) or cell_m <= 0:
        return inside
    plane_n = np.array(world.table_plane.normal, dtype=np.float64)
    heights = points @ plane_n - float(world.table_plane.offset_m)
    near = np.nonzero(np.abs(heights) <= table_margin_m)[0]
    if not len(near):
        return inside
    xy = points[near, :2]
    lo = np.minimum(xy.min(axis=0), np.array([0.0, 0.0]))
    cells = ((xy - lo) / cell_m).astype(np.int64)
    origin = ((np.array([0.0, 0.0]) - lo) / cell_m).astype(np.int64)
    shape = tuple((cells.max(axis=0) + 2).tolist())
    grid = np.zeros(shape, dtype=bool)
    grid[cells[:, 0], cells[:, 1]] = True
    labels, _ = ndimage.label(grid, structure=np.ones((3, 3), dtype=int))
    home = int(labels[origin[0], origin[1]])
    if home == 0:  # no table point under the origin: take the nearest piece
        occupied = np.argwhere(grid)
        dist = np.sum((occupied - origin) ** 2, axis=1)
        home = int(labels[tuple(occupied[int(np.argmin(dist))])])
    piece = labels == home
    steps = int(np.ceil(dilate_m / cell_m)) if dilate_m > 0 else 0
    if steps:
        piece = ndimage.binary_dilation(piece, iterations=steps)
    all_cells = ((points[:, :2] - lo) / cell_m).astype(np.int64)
    ok = (
        (all_cells[:, 0] >= 0)
        & (all_cells[:, 0] < piece.shape[0])
        & (all_cells[:, 1] >= 0)
        & (all_cells[:, 1] < piece.shape[1])
    )
    inside[ok] = piece[all_cells[ok, 0], all_cells[ok, 1]]
    return inside


def board_hull(
    world: WorldModel,
    run_dir: Path,
    family: str,
    detect: TagDetector | None = None,
) -> F64 | None:
    """Convex hull (XY, ordered) of the triangulated tag corners, or None.

    Each detected corner is back-projected (pinhole, distortion ignored: a
    corner is a few pixels of error at most) and intersected with the table
    plane, so the footprint follows the printed board even on a clean board
    whose layout is unknown. ``detect`` is the shared once-per-frame detector
    of fix 3. None when no tag corner could be triangulated; the caller then
    skips the board test and records a flag.
    """
    detect_fn = detect or cached_tag_detector(family)
    k = world.intrinsics
    plane_n = np.array(world.table_plane.normal, dtype=np.float64)
    plane_d = float(world.table_plane.offset_m)
    corners_world: list[F64] = []
    for frame in world.frames:
        corners, (height, width) = detect_fn(run_dir / frame.image_path)
        if not corners:
            continue
        pose = frame.T_world_from_camera.to_numpy()
        rot, center = pose[:3, :3], pose[:3, 3]
        for quad in corners.values():
            scaled = np.asarray(quad, dtype=np.float64) * np.array(
                [k.width_px / width, k.height_px / height]
            )
            for u, v in scaled:
                ray_cam = np.array([(u - k.cx) / k.fx, (v - k.cy) / k.fy, 1.0])
                ray = rot @ ray_cam
                denom = float(plane_n @ ray)
                if abs(denom) < 1e-9:
                    continue
                s = (plane_d - float(plane_n @ center)) / denom
                if s <= 0:
                    continue
                corners_world.append((center + s * ray)[None, :])
    if len(corners_world) < 3:
        return None
    pts = np.concatenate(corners_world)[:, :2]
    if float(np.ptp(pts[:, 0])) < 1e-6 or float(np.ptp(pts[:, 1])) < 1e-6:
        return None
    from scipy.spatial import ConvexHull  # type: ignore[import-untyped]  # heavy import

    hull = ConvexHull(pts)
    return np.asarray(pts[hull.vertices], dtype=np.float64)


def outside_board(points: F64, hull: F64 | None, margin_m: float) -> NDArray[np.bool_]:
    """True for points further than ``margin_m`` outside the board hull (all True if None)."""
    if hull is None or not len(points):
        return np.ones(len(points), dtype=bool)
    import cv2  # heavy import, kept out of module import time

    out = np.empty(len(points), dtype=bool)
    for i, p in enumerate(points):
        dist = cv2.pointPolygonTest(
            np.asarray(hull, dtype=np.float32), (float(p[0]), float(p[1])), True
        )
        out[i] = dist < -margin_m
    return out


def residual_clusters(
    world: WorldModel,
    cloud: F64,
    vote: PointVote,
    params: ResidualParams,
    table_margin_m: float,
    max_height_m: float,
    hull: F64 | None,
) -> list[F64]:
    """Point groups no kept track explains (step 3), largest first.

    A point is residual when its best vote is under ``explain_ratio`` (or its
    winning track saw it in fewer than ``min_views`` views), it sits between
    ``table_margin_m`` and ``max_height_m`` above the plane, over the table
    region and outside the board footprint. Groups come from the KD-tree
    grouping of ``track_repair._groups``; ties break by centroid x then y, so
    the order is deterministic. This replaces ``unexplained_clusters``,
    ``_near_any`` and ``is_board_cluster`` in the wired stage.
    """
    if not len(cloud):
        return []
    plane_n = np.array(world.table_plane.normal, dtype=np.float64)
    heights = cloud @ plane_n - float(world.table_plane.offset_m)
    explained = (vote.best_ratio >= params.explain_ratio) & (vote.best_seen >= params.min_views)
    cand = np.nonzero(
        ~explained
        & (heights >= table_margin_m)
        & (heights <= max_height_m)
        & table_region(cloud, world, table_margin_m, params.table_cell_m, params.table_dilate_m)
        & outside_board(cloud, hull, params.board_margin_m)
    )[0]
    if not len(cand):
        return []
    pts = np.asarray(cloud[cand], dtype=np.float64)
    groups = [g for g in _groups(pts, params.radius_m) if len(g) >= params.min_points]
    groups.sort(
        key=lambda g: (-len(g), float(pts[g].mean(axis=0)[0]), float(pts[g].mean(axis=0)[1]))
    )
    return [pts[g] for g in groups]


def _overlaps(
    a: Mapping[int, NDArray[np.bool_]], b: Mapping[int, NDArray[np.bool_]]
) -> list[tuple[int, float, float]]:
    """(frame, IoU, containment of a in b) over shared frames, 0.0 when incomparable."""
    out = []
    for frame in sorted(set(a) & set(b)):
        ma, mb = a[frame], b[frame]
        if ma.shape != mb.shape:
            out.append((frame, 0.0, 0.0))
            continue
        inter = float(np.logical_and(ma, mb).sum())
        union = float(np.logical_or(ma, mb).sum())
        area = float(ma.sum())
        out.append((frame, inter / union if union else 1.0, inter / area if area else 1.0))
    return out


def cluster_gate(
    track: Track, points: F64, world: WorldModel, lift_cfg: LiftConfig, accept_ratio: float
) -> str | None:
    """Rejection reason for a box-found track, or None when it passes (step 7).

    The track's own masks must contain its cluster's points: the median
    per-point vote of the cluster under this one track must clear
    ``accept_ratio``. A table patch (masks elsewhere) and a neighbouring
    object (masks covering a different cluster) both fail, however well
    their prompted-frame mask matched the prompted box.
    """
    if not len(points):
        return "no cluster points to gate on"
    # The gate's ratio is h/s over the views that see each point; the
    # lift's min_views floor would zero every point of a short clip.
    vote = point_vote(
        [track], world, np.asarray(points, dtype=np.float64), replace(lift_cfg, min_views=1)
    )
    median = float(np.median(vote.best_ratio))
    if median < accept_ratio:
        return f"cluster vote median {median:.2f} below {accept_ratio}"
    return None


def box_prompt_for_checks(
    checks: Sequence[Any],
    size: tuple[int, int],
    first_id: int = CLUSTER_ID_BASE,
) -> tuple[str | None, tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """One object-prompt string for the accepted clusters (step 6).

    Prompt frames are the views chosen in step 4; each id encodes its
    cluster (``first_id`` + cluster index). At most 16 clusters are prompted
    (the API limit); the rest come back as ``over_cap`` and must be
    reported. Accepted clusters with no chosen view are ``unprompted``.
    Returns (prompt, prompted track ids, unprompted indices, over-cap indices).
    """
    from video2sim.stages.cluster_check import ClusterCheck

    accepted = [c for c in checks if isinstance(c, ClusterCheck) and c.verdict.status == "accepted"]
    prompted, over_cap = accepted[:16], accepted[16:]
    prompts: dict[int, list[ObjectPrompt]] = {}
    unprompted: list[int] = []
    ids: list[int] = []
    for check in prompted:
        track_id = first_id + check.index
        if not check.views:
            unprompted.append(check.index)
            continue
        ids.append(track_id)
        for view in check.views:
            prompt = ObjectPrompt(
                object_id=track_id,
                box=tuple(float(v) for v in view.box),  # type: ignore[arg-type]
                positive_points=(view.point,) if view.point is not None else (),
            )
            prompts.setdefault(view.pos, []).append(prompt)
    if not prompts:
        return None, (), tuple(unprompted), tuple(c.index for c in over_cap)
    return (
        build_object_prompt(prompts, size),
        tuple(ids),
        tuple(unprompted),
        tuple(c.index for c in over_cap),
    )


def merge_requests(
    generic: Sequence[Track], requests: Sequence[GapRequest], cfg: GapFillConfig
) -> MergeResult:
    """New tracks merged into the generic set. For a new track T against each existing
    track E over shared frames S: no shared frames (new, the repair and duplicate rules
    cover it after the fact); 1 or 2 shared frames (same object only if IoU >= the bar
    on every frame, a part only if containment holds on every frame); 3 or more (median
    IoU, median containment). The bar is ``merge_iou`` against a generic track and the
    lower ``merge_new_iou`` when E is gap-fill new itself: two prompts of one object
    (a noun and a box) draw different extents, so their masks agree less than two
    views of one prompt. Box-found tracks have already passed the geometric
    cluster gate (step 7) in the caller. Ids that still collide are remapped past
    the maximum in use."""
    existing = list(generic)
    generic_ids = {t.track_id for t in existing}
    used = set(generic_ids)
    notes: dict[int, list[str]] = {t.track_id: [] for t in existing}
    dropped: dict[int, str] = {}
    remapped: list[tuple[int, int]] = []
    for request in requests:
        for track in sorted(request.tracks, key=lambda t: t.track_id):
            orig = track.track_id
            tid = orig
            if tid in used:
                tid = max(used) + 1
                track = Track(tid, dict(track.masks))
                remapped.append((orig, tid))
            same: Track | None = None
            part: Track | None = None
            for other in sorted(existing, key=lambda t: t.track_id):
                pairs = _overlaps(track.masks, other.masks)
                if not pairs:
                    continue
                bar = cfg.merge_iou if other.track_id in generic_ids else cfg.merge_new_iou
                ious = [iou for _, iou, _ in pairs]
                conts = [c for _, _, c in pairs]
                if len(pairs) >= 3:
                    if float(np.median(ious)) >= bar:
                        same = other
                        break
                    if float(np.median(conts)) >= cfg.part_inclusion:
                        part = part or other
                elif all(i >= bar for i in ious):
                    same = other
                    break
                elif all(c >= cfg.part_inclusion for c in conts):
                    part = part or other
            if same is not None:
                notes[same.track_id].append(request.note)
            elif part is not None:
                dropped[tid] = f"part of obj_{part.track_id:03d}: {request.note}"
            else:
                existing.append(track)
                used.add(tid)
                notes[tid] = [request.note]
    return MergeResult(existing, {k: tuple(v) for k, v in notes.items()}, dropped, tuple(remapped))


def inherit_provenance(prov: Mapping[int, FoundBy], report: RepairReport) -> dict[int, FoundBy]:
    """Found-by per track id after repair: splits inherit their parent, and a merge group
    is generic when any member is (else the first member's wins)."""
    out = dict(prov)
    for orig, parts in report.splits:
        for part in parts:
            out.setdefault(part, out.get(orig, "generic"))
    for group in report.merges:
        members = sorted(group)
        merged = (
            "generic"
            if any(out.get(m, "generic") == "generic" for m in members)
            else out.get(members[0], "generic")
        )
        out[members[0]] = merged
    return out


@dataclass(frozen=True)
class FitPass:
    """One run of the lift-and-fit block: the kept tracks and their geometry."""

    kept: list[Track]
    hulls: Mapping[int, F64]
    sparse: Mapping[int, F64]
    fits: Mapping[int, FitResult]
    drops: dict[int, str]  # track id -> bare reason; the stage adds "track {id} dropped:"
    vote: PointVote | None = None  # per-point vote over the cloud, for residual clusters


def fit_pass(
    tracks: Sequence[Track],
    world: WorldModel,
    lift_cfg: LiftConfig,
    prim_cfg: PrimitiveConfig,
    cloud: F64,
) -> FitPass:
    """Repair-then-lift-and-fit block, run once on the generic tracks and, after gap fill,
    again on the union."""
    hulls, hull_report = lift_tracks(tracks, world, lift_cfg)
    kept = [t for t in tracks if t.track_id in hulls]
    drops = {
        t.track_id: hull_report.dropped.get(t.track_id, "no point cluster")
        for t in tracks
        if t.track_id not in hulls
    }
    sparse, _, vote = lift_points(kept, world, cloud, lift_cfg)
    fits = fit_clusters(kept, sparse, world, lift_cfg, prim_cfg)
    dups = drop_duplicate_fits(fits, {t.track_id: t for t in kept}, prim_cfg)
    for i in dups:
        fits.pop(i, None)
    kept = [t for t in kept if t.track_id not in dups]
    drops.update(dups)
    return FitPass(kept, hulls, sparse, fits, drops, vote)


def table_view_coverage(
    world: WorldModel,
    cloud: F64,
    view_frames: Sequence[int],
    order: Sequence[int],
    view_cfg: ViewConfig,
    cell_m: float,
    min_views_per_cell: int,
    params: ResidualParams,
    table_margin_m: float,
) -> float:
    """Share of table-region cells the router frames see (step 8).

    Visibility is computed only over the points of the table region, not
    the whole cloud, so this is cheap; it fills ``DiscoveryCheck``.
    """
    if not len(cloud):
        return 0.0
    region = table_region(cloud, world, table_margin_m, params.table_cell_m, params.table_dilate_m)
    points = cloud[region] if region.any() else cloud
    visibility, _lo, _shape = compute_visibility(world, points, view_cfg)
    positions = [order.index(f) for f in view_frames if f in order]
    if not visibility.shape[1] or not positions:
        return 0.0
    counts = visibility[positions].sum(axis=0)
    return float((counts >= min_views_per_cell).mean())


def execute_gap_fill(
    world: WorldModel,
    run_dir: Path,
    checks: Sequence[ClusterCheck],
    gcfg: GapFillConfig,
    lift_cfg: LiftConfig,
    prim_cfg: PrimitiveConfig,
    repair_cfg: RepairConfig,
    marker_cfg: MarkerFilterConfig,
    cloud: F64,
    poses: Mapping[int, SE3],
    generic_kept: Sequence[Track],
    listed: int,
    view_frames: Sequence[int],
    view_coverage: float,
    track_fn: Callable[[str, int], tuple[list[Track], bool]],
    first: FitPass | None = None,
    detect: TagDetector | None = None,
) -> tuple[FitPass, DiscoveryCheck, dict[str, Any], dict[int, FoundBy]]:
    """Send the one box request, gate it, fall back to nouns, merge (steps 6-8).

    ``track_fn(prompt, offset)`` sends one video request and returns its
    tracks with ids shifted plus whether the answer came from the cache.
    Box tracks are gated geometrically against their own cluster; an
    accepted cluster whose box track fails earns one noun request (its
    verdict noun), in parallel, within ``max_video_requests``. Returns the
    second fit pass (the first, as-is, when no new track survives), the
    discovery audit, the ``route.json`` gap section and found-by per track.
    """
    detect_once = cached_tag_detector(marker_cfg.family, detect)
    size = (world.intrinsics.width_px, world.intrinsics.height_px)
    prompt, _prompted_ids, unprompted, over_cap = box_prompt_for_checks(checks, size)
    accepted = {c.index: c for c in checks if c.verdict.status == "accepted"}
    by_id = {CLUSTER_ID_BASE + c.index: c for c in accepted.values()}

    requests: list[GapRequest] = []
    records: list[dict[str, Any]] = []
    gate_results: dict[int, str] = {}  # cluster index -> gate outcome
    fallback: list[ClusterCheck] = []
    marker_dropped = 0
    video_requests = 1  # the generic request

    def run_tracks(prompt_text: str, offset: int) -> tuple[list[Track], bool, int]:
        tracks, cached = track_fn(prompt_text, offset)
        kept_tracks, mrep = filter_marker_tracks(tracks, world, run_dir, marker_cfg, detect_once)
        return kept_tracks, cached, len(mrep.dropped)

    if prompt is not None:
        video_requests += 1
        box_tracks, cached, dropped_n = run_tracks(prompt, 0)
        marker_dropped += dropped_n
        records.append({"kind": "cluster_box", "cached": cached, "tracks": len(box_tracks)})
        passed: list[Track] = []
        for track in sorted(box_tracks, key=lambda t: t.track_id):
            check = by_id.get(track.track_id)
            if check is None:
                gate_results[track.track_id] = "box track without a prompted cluster"
                continue
            reason = cluster_gate(track, check.points, world, lift_cfg, gcfg.check.accept_ratio)
            if reason is None:
                gate_results[check.index] = "passed"
                passed.append(track)
            else:
                gate_results[check.index] = reason
                fallback.append(check)
        if passed:
            requests.append(GapRequest(tuple(passed), "cluster_box", "unexplained point cluster"))
    else:
        records.append({"kind": "cluster_box", "cached": False, "tracks": 0})

    # Noun fallback: only accepted clusters whose box track failed the gate.
    noun_budget = max(gcfg.max_video_requests - video_requests, 0)
    noun_checks = fallback[: min(len(fallback), gcfg.max_noun_requests, noun_budget)]
    noun_dropped = [c.verdict.noun for c in fallback[len(noun_checks) :]]

    def run_noun(k: int, check: ClusterCheck) -> tuple[GapRequest, dict[str, Any], int]:
        noun = check.verdict.noun or ""
        tracks, cached, dropped_n = run_tracks(noun, NOUN_ID_BASE + NOUN_ID_STEP * k)
        return (
            GapRequest(tuple(tracks), "noun", f"noun '{noun}'"),
            {"kind": "noun", "noun": noun, "cached": cached, "tracks": len(tracks)},
            dropped_n,
        )

    if len(noun_checks) > 1 and gcfg.max_workers > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=gcfg.max_workers) as pool:
            noun_results = list(pool.map(run_noun, range(len(noun_checks)), noun_checks))
    else:
        noun_results = [run_noun(k, c) for k, c in enumerate(noun_checks)]
    video_requests += len(noun_results)
    for req, rec, dropped_n in noun_results:
        requests.append(req)
        records.append(rec)
        marker_dropped += dropped_n

    merged = merge_requests(list(generic_kept), requests, gcfg)
    prov: dict[int, FoundBy] = {t.track_id: "generic" for t in generic_kept}
    for request in requests:
        for track in request.tracks:
            prov.setdefault(track.track_id, request.found_by)
    for old, new in merged.remapped:
        prov[new] = prov.get(old, "generic")
    repaired, report = repair_tracks(
        merged.tracks, poses, world.intrinsics, world.table_plane, repair_cfg
    )
    prov = inherit_provenance(prov, report)
    generic_ids = {t.track_id for t in generic_kept}
    if first is not None and not ({t.track_id for t in merged.tracks} - generic_ids):
        second = first  # no new track survived: the union lift-and-fit would repeat this
    else:
        second = fit_pass(repaired, world, lift_cfg, prim_cfg, cloud)

    survived = {note for notes in merged.notes.values() for note in notes}
    matched_nouns = {
        note[6:-1]
        for notes in merged.notes.values()
        for note in notes
        if note.startswith("noun '") and note.endswith("'")
    }
    kept_ids = {t.track_id for t in second.kept}
    unmatched: list[str] = []
    for check in accepted.values():
        noun = check.verdict.noun
        if noun is None:
            continue
        box_alive = (
            gate_results.get(check.index) == "passed"
            and "unexplained point cluster" in survived
            and (
                CLUSTER_ID_BASE + check.index in kept_ids
                or any(prov.get(tid) == "cluster_box" for tid in kept_ids)
            )
        )
        if not box_alive and noun not in matched_nouns:
            unmatched.append(noun)
    recovered = sum(1 for t in second.kept if prov.get(t.track_id, "generic") != "generic")
    discovery = DiscoveryCheck(
        view_frames=list(view_frames),
        view_coverage=view_coverage,
        listed=listed,
        tracked=len(second.kept),
        missing_suspected=listed > len(second.kept),
        unmatched=unmatched,
        unexplained_clusters=len(checks),
        recovered=recovered,
        video_requests=video_requests,
    )
    report_lines = [f"gap fill: {recovered} object(s) recovered in {len(records)} extra request(s)"]
    flag_reasons: dict[str, str] = {}
    for t in second.kept:
        note = _origin_note(t.track_id, merged.notes, report)
        if note is None:
            continue
        if prov.get(t.track_id, "generic") != "generic":
            flag_reasons[str(t.track_id)] = f"found by gap fill: {note}"
        else:
            flag_reasons[str(t.track_id)] = f"also found by gap fill: {note}"
    for t in second.kept:
        if str(t.track_id) in flag_reasons:
            report_lines.append(f"obj_{t.track_id:03d}: {flag_reasons[str(t.track_id)]}")
    for index in unprompted:
        report_lines.append(f"gap fill: cluster {index} has no placeable box")
    for index in over_cap:
        report_lines.append(f"gap fill: cluster {index} past the 16-object limit, not prompted")
    cluster_records = [
        {
            "index": c.index,
            "points": len(c.points),
            "crops": [str(p) for p in c.crops],
            "views": [v.frame_index for v in c.views],
            "verdict": c.verdict.status,
            "noun": c.verdict.noun,
            "confidence": c.verdict.confidence,
            "reason": c.verdict.reason,
            "track_id": (CLUSTER_ID_BASE + c.index) if c.index in accepted else None,
            "gate": gate_results.get(c.index),
        }
        for c in checks
    ]
    gap_notes: dict[str, Any] = {
        "listed": listed,
        "unexplained_clusters": len(checks),
        "clusters": cluster_records,
        "requests": records,
        "marker_dropped": marker_dropped,
        "merged_notes": {str(k): list(v) for k, v in merged.notes.items() if v},
        "dropped": {str(k): v for k, v in merged.dropped.items()},
        "gate_dropped": {str(k): v for k, v in gate_results.items() if v != "passed"},
        "unprompted_clusters": list(unprompted),
        "over_cap_clusters": list(over_cap),
        "noun_dropped": noun_dropped,
        "recovered": recovered,
        "video_requests": video_requests,
        "report_lines": report_lines,
        "flag_reasons": flag_reasons,
    }
    return second, discovery, gap_notes, prov


def _origin_note(
    tid: int, notes: Mapping[int, tuple[str, ...]], report: RepairReport
) -> str | None:
    """Provenance note for a post-repair track: its own, else its split parent's."""
    if notes.get(tid):
        return notes[tid][0]
    for orig, parts in report.splits:
        if tid in parts and notes.get(orig):
            return notes[orig][0]
    return None


__all__ = [
    "CLUSTER_ID_BASE",
    "NOUN_ID_BASE",
    "NOUN_ID_STEP",
    "Box",
    "ClusterParams",
    "FitPass",
    "FoundBy",
    "GapFillConfig",
    "GapRequest",
    "MergeResult",
    "ResidualParams",
    "board_hull",
    "box_prompt_for_checks",
    "cluster_gate",
    "execute_gap_fill",
    "fit_pass",
    "inherit_provenance",
    "merge_requests",
    "offset_tracks",
    "outside_board",
    "residual_clusters",
    "table_region",
    "table_view_coverage",
]
