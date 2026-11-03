"""One gap-fill round after tracking (S12.9, task 9.6).

Tracking runs one generic SAM concept, so objects it never fires on leave two traces: the
router's inventory lists more than the tracks cover, and the sparse cloud holds point
clusters no track explains. Gap fill sends at most one round of extra SAM video requests —
one per distinct inventory noun on a shortfall, one box-prompt request for the unexplained
clusters (the 9.0 probe confirmed the hosted model accepts box prompts and tracks them) —
then merges the new tracks into the generic set. Everything that survives goes through the
same repair, lift, fit and duplicate rules as generic tracks.

With ``enabled: false`` nothing here runs and the stage output is byte-identical.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from video2sim.models.sam_prompts import ObjectPrompt, build_object_prompt
from video2sim.schemas import SE3, DiscoveryCheck, WorldModel
from video2sim.stages.lift import (
    MIN_DEPTH_M,
    LiftConfig,
    PointVote,
    _pixels,
    lift_points,
    lift_tracks,
)
from video2sim.stages.marker_filter import (
    MarkerFilterConfig,
    TagDetector,
    cached_tag_detector,
    filter_marker_tracks,
    tag_mask,
)
from video2sim.stages.overlay_audit import OverlayAuditConfig
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
BATCH_OFFSET = 90000  # the one batched audit-noun request; clear of box and noun ids


@dataclass(frozen=True)
class ClusterParams:
    radius_m: float
    min_points: int
    explained_dist_m: float
    box_pad_px: int
    frames_per_cluster: int
    min_azimuth_gap_deg: float
    area_margin_m: float = 0.5  # XY distance from a kept track that bounds candidates


@dataclass(frozen=True)
class GapFillConfig:
    """Values from ``segmentation.tracking.gap_fill`` in ``configs/default.yaml``."""

    enabled: bool
    max_rounds: int
    max_noun_requests: int
    max_video_requests: int
    max_workers: int
    cluster: ClusterParams
    audit: OverlayAuditConfig
    merge_iou: float
    merge_new_iou: float
    part_inclusion: float
    box_mask_min_iou: float
    box_mask_min_fill: float
    box_mask_max_overfill: float

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
                explained_dist_m=float(c["explained_dist_m"]),
                box_pad_px=int(c["box_pad_px"]),
                frames_per_cluster=int(c["frames_per_cluster"]),
                min_azimuth_gap_deg=float(c["min_azimuth_gap_deg"]),
                area_margin_m=float(c.get("area_margin_m", 0.5)),
            ),
            audit=OverlayAuditConfig.from_config(cfg),
            merge_iou=float(g["merge_iou"]),
            merge_new_iou=float(g["merge_new_iou"]),
            part_inclusion=float(g["part_inclusion"]),
            box_mask_min_iou=float(g["box_mask_min_iou"]),
            box_mask_min_fill=float(g["box_mask_min_fill"]),
            box_mask_max_overfill=float(g["box_mask_max_overfill"]),
        )
        if out.max_rounds != 1:
            raise ValueError("only one gap-fill round is implemented")
        if out.max_noun_requests < 0 or out.max_video_requests < 1 or out.max_workers < 1:
            raise ValueError("bad gap_fill request caps")
        cc = out.cluster
        if cc.radius_m <= 0 or cc.min_points < 1 or cc.explained_dist_m < 0 or cc.area_margin_m < 0:
            raise ValueError("bad gap_fill cluster grouping")
        if cc.box_pad_px < 0 or cc.frames_per_cluster < 1 or cc.min_azimuth_gap_deg < 0:
            raise ValueError("bad gap_fill box placement")
        for v in (
            out.merge_iou,
            out.merge_new_iou,
            out.part_inclusion,
            out.box_mask_min_iou,
            out.box_mask_min_fill,
        ):
            if not 0.0 < v <= 1.0:
                raise ValueError("gap_fill IoU and fill gates must be in (0, 1]")
        if out.merge_new_iou > out.merge_iou:
            raise ValueError("gap_fill merge_new_iou must not clear merge_iou")
        if out.box_mask_max_overfill < out.box_mask_min_fill:
            raise ValueError("gap_fill overfill cap must clear the fill floor")
        return out


@dataclass(frozen=True)
class GapRequest:
    """One gap-fill video request's tracks, ready to merge."""

    tracks: tuple[Track, ...]
    found_by: FoundBy
    note: str  # provenance recorded on surviving tracks, e.g. the noun
    boxes: tuple[tuple[int, int, Box], ...] = ()  # (track_id, frame, prompted box)


@dataclass(frozen=True)
class MergeResult:
    tracks: list[Track]
    notes: dict[int, tuple[str, ...]]  # surviving track id -> provenance notes
    dropped: dict[int, str]  # dropped new track id -> reason
    remapped: tuple[tuple[int, int], ...] = ()  # (old, new) ids when a collision remapped


def offset_tracks(tracks: Sequence[Track], offset: int) -> list[Track]:
    """Tracks with ids shifted so requests never collide (see the id bases above)."""
    return [Track(t.track_id + offset, dict(t.masks)) for t in tracks]


def _near_any(points: F64, kept: Sequence[F64], dist_m: float) -> NDArray[np.bool_]:
    """Points within ``dist_m`` of any kept-cluster point (KD-tree per cluster)."""
    from scipy.spatial import cKDTree  # type: ignore[import-untyped]  # heavy import

    near = np.zeros(len(points), dtype=bool)
    if not len(points):
        return near
    for cluster in kept:
        if not len(cluster):
            continue
        dist, _ = cKDTree(np.asarray(cluster, dtype=np.float64)).query(
            np.asarray(points, dtype=np.float64), k=1, distance_upper_bound=dist_m
        )
        near |= dist <= dist_m
        if near.all():
            break
    return near


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
    from scipy.spatial import ConvexHull  # heavy import, scipy is untyped here

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


def _project(points: F64, pose: F64, k: Any) -> tuple[F64, F64, NDArray[np.bool_]]:
    """Pixel coordinates and the in-front mask of world points in one frame."""
    rot, center = pose[:3, :3], pose[:3, 3]
    cam = (points - center) @ rot
    z = cam[:, 2]
    ok = z > MIN_DEPTH_M
    u = np.full(len(points), -1.0)
    v = np.full(len(points), -1.0)
    if ok.any():
        pu, pv = _pixels(cam[ok, 0] / z[ok], cam[ok, 1] / z[ok], k)
        u[ok], v[ok] = pu, pv
    return u, v, ok


def is_board_cluster(
    points: F64,
    world: WorldModel,
    run_dir: Path,
    family: str,
    detect: TagDetector | None = None,
    min_tag_share: float = 0.5,
) -> bool:
    """True when most projected points land on detected tags in some frame: the cluster is
    the printed board or a tag, not an object. ``detect`` defaults to OpenCV; tests inject."""
    if not len(points):
        return False
    detect_fn = detect or cached_tag_detector(family)
    k = world.intrinsics
    for frame in world.frames:
        corners, (height, width) = detect_fn(run_dir / frame.image_path)
        if not corners:
            continue
        pose = frame.T_world_from_camera.to_numpy()
        u, v, ok = _project(points, pose, k)
        inside_image = ok & (u >= 0) & (u < k.width_px) & (v >= 0) & (v < k.height_px)
        if not inside_image.any():
            continue
        sy, sx = height / k.height_px, width / k.width_px
        quads = [c * np.array([sx, sy]) for c in corners.values()]
        tags = tag_mask(quads, (k.height_px, k.width_px), 0)
        if (
            float(tags[v[inside_image].astype(int), u[inside_image].astype(int)].mean())
            >= min_tag_share
        ):
            return True
    return False


def unexplained_clusters(
    world: WorldModel,
    cloud: F64,
    kept: Mapping[int, F64],
    run_dir: Path,
    family: str,
    cluster_cfg: ClusterParams,
    table_margin_m: float,
    max_height_m: float,
    detect: TagDetector | None = None,
) -> list[F64]:
    """Above-table point groups no kept track explains, largest first (ties by centroid x
    then y). Points within ``explained_dist_m`` of a kept cluster, further than
    ``area_margin_m`` (XY) from every kept track, inside the marker board
    footprint, or in groups below ``min_points`` are not unexplained. With no
    kept track there is no area to bound to, so every group counts."""
    plane_n = np.array(world.table_plane.normal, dtype=np.float64)
    plane_d = float(world.table_plane.offset_m)
    heights = cloud @ plane_n - plane_d
    kept_list = [p for p in kept.values() if len(p)]
    in_area = (
        _near_any(cloud[:, :2], [p[:, :2] for p in kept_list], cluster_cfg.area_margin_m)
        if kept_list
        else np.ones(len(cloud), dtype=bool)
    )
    cand = np.nonzero(
        (heights >= table_margin_m)
        & (heights <= max_height_m)
        & in_area
        & ~_near_any(cloud, kept_list, cluster_cfg.explained_dist_m)
    )[0]
    if not len(cand):
        return []
    groups = [
        g
        for g in _groups(np.asarray(cloud[cand], dtype=np.float64), cluster_cfg.radius_m)
        if len(g) >= cluster_cfg.min_points
    ]
    # Largest first, ties by centroid x then y, so the order is deterministic.
    groups.sort(
        key=lambda g: (
            -len(g),
            float(cloud[cand[g]].mean(axis=0)[0]),
            float(cloud[cand[g]].mean(axis=0)[1]),
        )
    )
    detect_once = cached_tag_detector(family, detect)
    out = []
    for group in groups:
        points = np.asarray(cloud[cand[group]], dtype=np.float64)
        if not is_board_cluster(points, world, run_dir, family, detect_once):
            out.append(points)
    return out


def _frame_box(
    world: WorldModel, pos: int, points: F64, pad_px: int
) -> tuple[Box, tuple[float, float] | None] | None:
    """Prompt box and median point on one clip position, or None when unusable.

    The box is the 2D bounding box of the projected eight AABB corners plus their drops to
    the table plane, padded and clamped; None when a corner is behind the camera or the
    padded box is cut by the image border. The median point is omitted when behind.
    """
    frame = world.frames[pos]
    k = world.intrinsics
    pose = frame.T_world_from_camera.to_numpy()
    plane_n = np.array(world.table_plane.normal, dtype=np.float64)
    plane_d = float(world.table_plane.offset_m)
    lo, hi = points.min(axis=0), points.max(axis=0)
    corners = np.array(np.meshgrid(*zip(lo, hi, strict=True))).reshape(3, -1).T
    drops = corners - plane_n[None, :] * ((corners @ plane_n - plane_d)[:, None])
    u, v, ok = _project(np.concatenate([corners, drops]), pose, k)
    if not ok.all():
        return None
    x1, y1 = int(np.min(u)), int(np.min(v))
    x2, y2 = int(np.max(u)), int(np.max(v))
    if x2 <= x1 or y2 <= y1:
        return None
    padded = (x1 - pad_px, y1 - pad_px, x2 + pad_px, y2 + pad_px)
    clamped = (
        max(padded[0], 0),
        max(padded[1], 0),
        min(padded[2], k.width_px - 1),
        min(padded[3], k.height_px - 1),
    )
    if clamped != padded or clamped[2] <= clamped[0] or clamped[3] <= clamped[1]:
        return None
    median = np.median(points, axis=0)
    mu, mv, mok = _project(median[None, :], pose, k)
    point = (float(mu[0]), float(mv[0])) if mok[0] else None
    return clamped, point


def _azimuths(world: WorldModel, centroid: F64) -> NDArray[np.float64]:
    """Camera azimuth around a point, per clip position."""
    return np.array(
        [
            np.arctan2(
                f.T_world_from_camera.to_numpy()[1, 3] - centroid[1],
                f.T_world_from_camera.to_numpy()[0, 3] - centroid[0],
            )
            for f in world.frames
        ]
    )


def _far_enough(
    pos: int, placed: Sequence[int], azimuth: NDArray[np.float64], gap_deg: float
) -> bool:
    return all(
        abs((azimuth[pos] - azimuth[q] + np.pi) % (2 * np.pi) - np.pi) >= np.deg2rad(gap_deg)
        for q in placed
    )


def _ranked_positions(
    world: WorldModel,
    visibility: NDArray[np.bool_],
    grid: tuple[F64, tuple[int, int]],
    cell_m: float,
    points: F64,
) -> list[int]:
    """Clip positions by visible share of the cluster's cells, best first."""
    lo, shape = grid
    idx = ((points[:, :2] - lo) / cell_m).astype(int)
    ok = (idx[:, 0] >= 0) & (idx[:, 0] < shape[0]) & (idx[:, 1] >= 0) & (idx[:, 1] < shape[1])
    cells = np.unique(idx[ok, 1] * shape[0] + idx[ok, 0])
    if not len(cells):
        return []
    share = visibility[:, cells].mean(axis=1)
    return sorted(range(len(world.frames)), key=lambda p: (-share[p], p))


def pick_prompt_frames(
    world: WorldModel,
    visibility: NDArray[np.bool_],
    grid: tuple[F64, tuple[int, int]],
    cell_m: float,
    points: F64,
    n_frames: int,
    min_gap_deg: float,
) -> list[int]:
    """Clip positions prompting one cluster: largest visible share of its cells first, each
    at least ``min_gap_deg`` of azimuth apart around the cluster centroid."""
    azimuth = _azimuths(world, np.median(points, axis=0))
    picked: list[int] = []
    for pos in _ranked_positions(world, visibility, grid, cell_m, points):
        if _far_enough(pos, picked, azimuth, min_gap_deg):
            picked.append(pos)
        if len(picked) >= n_frames:
            break
    return picked


def box_request(
    world: WorldModel,
    visibility: NDArray[np.bool_],
    grid: tuple[F64, tuple[int, int]],
    cell_m: float,
    clusters: Sequence[F64],
    size: tuple[int, int],
    cluster_cfg: ClusterParams,
    first_id: int = CLUSTER_ID_BASE,
) -> tuple[str, dict[int, tuple[int, Box]], list[int]]:
    """One object-prompt string for the unexplained clusters plus, per prompted id, the
    prompted frame and box for the box gates. The same id is used on all its frames, and
    the id encodes the cluster (``first_id`` + cluster index). Clusters with no placeable
    box come back in ``unprompted`` and must be reported, never dropped quietly."""
    prompts: dict[int, list[ObjectPrompt]] = {}
    gated: dict[int, tuple[int, Box]] = {}
    unprompted: list[int] = []
    for index, points in enumerate(clusters):
        track_id = first_id + index
        azimuth = _azimuths(world, np.median(points, axis=0))
        placed_positions: list[int] = []
        # Try frames in visibility order until frames_per_cluster boxes fit; a
        # frame whose box is cut by the image border does not end the search.
        for pos in _ranked_positions(world, visibility, grid, cell_m, points):
            if len(placed_positions) >= cluster_cfg.frames_per_cluster:
                break
            if not _far_enough(pos, placed_positions, azimuth, cluster_cfg.min_azimuth_gap_deg):
                continue
            placed = _frame_box(world, pos, points, cluster_cfg.box_pad_px)
            if placed is None:
                continue
            box, point = placed
            prompt = ObjectPrompt(
                object_id=track_id,
                box=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                positive_points=(point,) if point is not None else (),
            )
            prompts.setdefault(pos, []).append(prompt)
            gated.setdefault(track_id, (world.frames[pos].frame_index, box))
            placed_positions.append(pos)
        if track_id not in gated:
            unprompted.append(index)
    if not prompts:
        raise ValueError("no cluster box could be placed on any frame")
    return build_object_prompt(prompts, size), gated, unprompted


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


def _box_canvas(shape: tuple[int, int], box: Box) -> NDArray[np.bool_]:
    canvas = np.zeros(shape, dtype=bool)
    canvas[box[1] : box[3] + 1, box[0] : box[2] + 1] = True
    return canvas


def _box_gates(track: Track, frame: int, box: Box, cfg: GapFillConfig) -> str | None:
    """Rejection reason for a box-found track, or None when it passes. The prompted-frame
    mask must overlap the prompted box (IoU), fill enough of it, and not sprawl past it —
    a box over bare table returns a table patch (9.0 negative control). IoU never exceeds
    fill, so the fill gate only binds when ``box_mask_min_iou`` is at most
    ``box_mask_min_fill``."""
    mask = track.masks.get(frame)
    if mask is None:
        return f"no mask on the prompted frame {frame}"
    canvas = _box_canvas(mask.shape, box)
    inter = float(np.logical_and(mask, canvas).sum())
    union = float(np.logical_or(mask, canvas).sum())
    box_area = float(canvas.sum())
    mask_area = float(mask.sum())
    iou = inter / union if union else 0.0
    fill = inter / box_area if box_area else 0.0
    over = mask_area / box_area if box_area else float("inf")
    if iou < cfg.box_mask_min_iou:
        return f"prompted-box IoU {iou:.2f} below {cfg.box_mask_min_iou}"
    if fill < cfg.box_mask_min_fill:
        return f"box fill {fill:.2f} below {cfg.box_mask_min_fill}"
    if over > cfg.box_mask_max_overfill:
        return f"box overfill {over:.2f} past {cfg.box_mask_max_overfill}"
    return None


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
    views of one prompt. Box-found tracks also pass the prompted-box gates. Ids that
    still collide are remapped past the maximum in use."""
    existing = list(generic)
    generic_ids = {t.track_id for t in existing}
    used = set(generic_ids)
    notes: dict[int, list[str]] = {t.track_id: [] for t in existing}
    dropped: dict[int, str] = {}
    remapped: list[tuple[int, int]] = []
    for request in requests:
        gated_boxes: dict[int, tuple[int, Box]] = {
            tid: (frame, box) for tid, frame, box in request.boxes
        }
        for track in sorted(request.tracks, key=lambda t: t.track_id):
            orig = track.track_id
            tid = orig
            if tid in used:
                tid = max(used) + 1
                track = Track(tid, dict(track.masks))
                remapped.append((orig, tid))
            if request.found_by == "cluster_box":
                gated = gated_boxes.get(orig)
                if gated is None:
                    dropped[tid] = "cluster track without a prompted box"
                    continue
                frame, box = gated
                reason = _box_gates(track, frame, box, cfg)
                if reason is not None:
                    dropped[tid] = reason
                    continue
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


@dataclass(frozen=True)
class GapPlan:
    """One gap-fill round, planned before any hosted call."""

    nouns: tuple[str, ...]  # noun prompts in send order (box request goes first)
    nouns_dropped: tuple[str, ...]  # over-cap nouns, never attempted
    send_box: bool
    box_prompt: str | None
    box_gated: tuple[tuple[int, int, Box], ...]  # (track_id, frame, prompted box)
    box_unprompted: tuple[int, ...]  # cluster indices with no placeable box
    clusters_total: int
    view_frames: list[int]  # frame_index values of the router views
    view_coverage: float  # share of cells the router views see unoccluded
    video_requests: int  # hosted video requests, generic one included
    audit_nouns: tuple[str, ...] = ()  # batched into one request, after the box request
    audit_prompt: str | None = None  # the batched concepts, None when not sent


def batch_prompt(nouns: Sequence[str]) -> str:
    """One video prompt for the audit nouns: SAM answers per-object ids anyway, so one
    request covers them all. Multi-concept prompts are new (9.7); the dev eval (9.8)
    confirms the hosted model honours every noun."""
    return ", ".join(nouns)


def plan_gap_fill(
    world: WorldModel,
    cloud: F64,
    nouns: Sequence[str],
    clusters: Sequence[F64],
    view_frames: Sequence[int],
    order: Sequence[int],
    view_cfg: ViewConfig,
    cell_m: float,
    min_views_per_cell: int,
    gcfg: GapFillConfig,
    audit_nouns: Sequence[str] = (),
    shortfall: int | None = None,
) -> GapPlan:
    """Which requests a gap-fill round sends. Box prompts go first (probe-verified,
    geometry-placed); the audit nouns follow batched into one request; inventory
    nouns follow one per request, and only on a shortfall (``shortfall`` > 0;
    None means the caller did not count, so they are sent). Past
    ``max_video_requests`` the trailing nouns are dropped. Clusters past the
    16-object API limit are not prompted. Visibility is computed over the
    cluster points only when there are clusters, not the whole cloud."""
    cluster_points = [np.asarray(c, dtype=np.float64) for c in clusters if len(c)]
    vis_points = np.concatenate(cluster_points) if cluster_points else cloud
    visibility, lo, shape = compute_visibility(world, vis_points, view_cfg)
    if shortfall is not None and shortfall <= 0:
        nouns = ()
    positions = [order.index(f) for f in view_frames if f in order]
    if visibility.shape[1] and positions:
        counts = visibility[positions].sum(axis=0)
        coverage = float((counts >= min_views_per_cell).mean())
    else:
        coverage = 0.0
    box_prompt: str | None = None
    box_gated: tuple[tuple[int, int, Box], ...] = ()
    box_unprompted: tuple[int, ...] = ()
    use_box = bool(clusters) and gcfg.max_video_requests >= 2
    if use_box:
        size = (world.intrinsics.width_px, world.intrinsics.height_px)
        prompt, gated, unprompted = box_request(
            world, visibility, (lo, shape), cell_m, clusters[:16], size, gcfg.cluster
        )
        box_prompt, box_unprompted = prompt, tuple(unprompted)
        box_gated = tuple((tid, frame, box) for tid, (frame, box) in gated.items())
    use_batch = bool(audit_nouns) and gcfg.max_video_requests >= 2 + (1 if use_box else 0)
    budget = gcfg.max_video_requests - 1 - (1 if use_box else 0) - (1 if use_batch else 0)
    keep = min(len(nouns), gcfg.max_noun_requests, max(budget, 0))
    dropped = list(nouns[keep:])
    batch: tuple[str, ...] = tuple(audit_nouns) if use_batch else ()
    if audit_nouns and not use_batch:
        dropped = [*audit_nouns, *dropped]
    return GapPlan(
        nouns=tuple(nouns[:keep]),
        nouns_dropped=tuple(dropped),
        send_box=use_box,
        box_prompt=box_prompt,
        box_gated=box_gated,
        box_unprompted=box_unprompted,
        clusters_total=len(clusters),
        view_frames=list(view_frames),
        view_coverage=coverage,
        video_requests=1 + keep + (1 if use_box else 0) + (1 if use_batch else 0),
        audit_nouns=batch,
        audit_prompt=batch_prompt(batch) if use_batch else None,
    )


def execute_gap_fill(
    world: WorldModel,
    run_dir: Path,
    plan: GapPlan,
    gcfg: GapFillConfig,
    lift_cfg: LiftConfig,
    prim_cfg: PrimitiveConfig,
    repair_cfg: RepairConfig,
    marker_cfg: MarkerFilterConfig,
    cloud: F64,
    poses: Mapping[int, SE3],
    generic_kept: Sequence[Track],
    shortfall: int,
    listed: int,
    track_fn: Callable[[str, int], tuple[list[Track], bool]],
    first: FitPass | None = None,
    detect: TagDetector | None = None,
) -> tuple[FitPass, DiscoveryCheck, dict[str, Any], dict[int, FoundBy]]:
    """Send the planned requests, merge, and run the lift-and-fit block on the union.

    ``track_fn(prompt, offset)`` sends one video request and returns its tracks with ids
    shifted plus whether the answer came from the response cache. Returns the second fit
    pass, the discovery audit, the ``route.json`` gap section (counts and names only, no
    arrays) and found-by per union track id. When no new track survives the merge and
    ``first`` (the generic tracks' fit pass) is given, it is returned as-is instead of
    paying for the same lift-and-fit again. Tag detection is shared across every
    marker filter here, one detection per frame.
    """
    detect_once = cached_tag_detector(marker_cfg.family, detect)
    # (prompt, offset, found_by, note, boxes, record) per request, in send order.
    specs: list[
        tuple[str, int, FoundBy, str, tuple[tuple[int, int, Box], ...], dict[str, Any]]
    ] = []
    if plan.send_box:
        assert plan.box_prompt is not None
        specs.append(
            (
                plan.box_prompt,
                0,
                "cluster_box",
                "unexplained point cluster",
                plan.box_gated,
                {"kind": "cluster_box"},
            )
        )
    batch_note = ""
    if plan.audit_nouns:
        assert plan.audit_prompt is not None
        batch_note = "audit nouns " + ", ".join(f"'{n}'" for n in plan.audit_nouns)
        specs.append(
            (
                plan.audit_prompt,
                BATCH_OFFSET,
                "noun",
                batch_note,
                (),
                {"kind": "audit_batch", "nouns": list(plan.audit_nouns)},
            )
        )
    for k, noun in enumerate(plan.nouns):
        specs.append(
            (
                noun,
                NOUN_ID_BASE + NOUN_ID_STEP * k,
                "noun",
                f"noun '{noun}'",
                (),
                {"kind": "noun", "noun": noun},
            )
        )

    def run_one(
        spec: tuple[str, int, FoundBy, str, tuple[tuple[int, int, Box], ...], dict[str, Any]],
    ) -> tuple[GapRequest, dict[str, Any], int]:
        prompt, offset, found_by, note, boxes, record = spec
        tracks, cached = track_fn(prompt, offset)
        kept_tracks, mrep = filter_marker_tracks(tracks, world, run_dir, marker_cfg, detect_once)
        return (
            GapRequest(tuple(kept_tracks), found_by, note, boxes),
            {**record, "cached": cached, "tracks": len(kept_tracks)},
            len(mrep.dropped),
        )

    # Requests are independent hosted calls; results are merged in plan order.
    if len(specs) > 1 and gcfg.max_workers > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=gcfg.max_workers) as pool:
            results = list(pool.map(run_one, specs))
    else:
        results = [run_one(spec) for spec in specs]
    requests = [r for r, _, _ in results]
    records = [rec for _, rec, _ in results]
    marker_dropped = sum(n for _, _, n in results)
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
    requested = [r["noun"] for r in records if r["kind"] == "noun"]
    survived = {note for notes in merged.notes.values() for note in notes}
    unmatched = [n for n in requested if f"noun '{n}'" not in survived]
    if batch_note and batch_note not in survived:
        unmatched += [n for n in plan.audit_nouns if n not in requested]
    unmatched += [n for n in plan.nouns_dropped if n not in requested]
    recovered = sum(1 for t in second.kept if prov.get(t.track_id, "generic") != "generic")
    discovery = DiscoveryCheck(
        view_frames=plan.view_frames,
        view_coverage=plan.view_coverage,
        listed=listed,
        tracked=len(second.kept),
        missing_suspected=listed > len(second.kept),
        unmatched=unmatched,
        unexplained_clusters=plan.clusters_total,
        recovered=recovered,
        video_requests=plan.video_requests,
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
    for index in plan.box_unprompted:
        report_lines.append(f"gap fill: cluster {index} has no placeable box")
    gap_notes: dict[str, Any] = {
        "listed": listed,
        "shortfall_trigger": shortfall,
        "unexplained_clusters": plan.clusters_total,
        "requests": records,
        "marker_dropped": marker_dropped,
        "merged_notes": {str(k): list(v) for k, v in merged.notes.items() if v},
        "dropped": {str(k): v for k, v in merged.dropped.items()},
        "unprompted_clusters": list(plan.box_unprompted),
        "recovered": recovered,
        "video_requests": plan.video_requests,
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
    "BATCH_OFFSET",
    "CLUSTER_ID_BASE",
    "NOUN_ID_BASE",
    "NOUN_ID_STEP",
    "Box",
    "ClusterParams",
    "FitPass",
    "FoundBy",
    "GapFillConfig",
    "GapPlan",
    "GapRequest",
    "MergeResult",
    "ResidualParams",
    "batch_prompt",
    "board_hull",
    "box_request",
    "execute_gap_fill",
    "fit_pass",
    "inherit_provenance",
    "is_board_cluster",
    "merge_requests",
    "offset_tracks",
    "outside_board",
    "pick_prompt_frames",
    "plan_gap_fill",
    "residual_clusters",
    "table_region",
    "unexplained_clusters",
]
