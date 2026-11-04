"""Fit a box, cylinder or sphere standing on the table to an object's masks (Phase 2 box gate).

The voted sparse points (``lift.lift_points``) lie only on visible, textured surfaces, so the box
of a cluster stops short of plain edges and hidden faces. The masks hold the missing evidence:
the outline of the object in every view. A primitive standing on the table is posed and sized so
that its projected outline matches the masks, starting from the box of the trimmed points.

* Three kinds are tried: an upright box, an upright elliptic cylinder and a sphere. The one whose
  outline disagrees least with the masks wins.
* The disagreement in one view is the outline area outside the mask plus the mask area outside
  the outline, over the mask area. Outline pixels on another track's mask do not count: the
  object may be hidden there. Views where the mask touches the image edge are skipped, because
  the mask is cut there.
* The search is a seeded random walk with a step that shrinks, so a rerun gives the same answer.
* An object that is none of the three (a mug with its handle, a toy car) fits badly. When the
  best cost is above ``max_cost``, the cluster is the union of the primitive outline and the
  trimmed points, so parts outside the primitive still count. The result says which.
* A stacked object is the exception to standing on the table. When most of its footprint lies
  over another object, and its lowest points start near that object's top, the cluster is moved
  to span from there to its highest points. The masks cannot tell, because they do not see depth.

On the dev scenes this took the easy-tier median box error from 8.9% (trimmed points) to 4.3%,
15 of 19 objects within 10% (``docs/research/cluster_box_accuracy.md``). The result is the
cluster of ``ObjectInstance.point_cluster_path`` (human decision 2026-10-29).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from video2sim.schemas import WorldModel
from video2sim.stages.lift import MIN_DEPTH_M, LiftConfig, _pixels, _views, footprint_axes
from video2sim.stages.tracking import Track

F64 = NDArray[np.float64]
BoolMask = NDArray[np.bool_]
Kind = Literal["box", "cylinder", "sphere"]
KINDS: tuple[Kind, ...] = ("box", "cylinder", "sphere")

# Outline samples: a cylinder ring and a sphere grid are convex, so the convex hull of these
# projected samples is the silhouette to within a pixel at the mask scale used here.
# The samples double as the shape stage's registration target, so the walls carry samples
# too: rims alone make the point term brittle to rim wobble on generative meshes (s15
# video_b basket: a good hamper scored inlier 0 on two rim rings).
_RING = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
_WALL_LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)  # rings up the wall, share of the height
_FACE_STEPS = (-1.0, 0.0, 1.0)  # 3x3 grids on each box face, share of the half sizes
_POLAR, _AZIMUTH = (
    a.ravel() for a in np.meshgrid(np.linspace(0.0, np.pi, 12), _RING[::2], indexing="ij")
)


@dataclass(frozen=True)
class PrimitiveConfig:
    """Values from ``segmentation.primitive_fit`` in ``configs/default.yaml``."""

    mask_downscale: int
    min_views: int
    min_mask_px: int
    iterations: int
    shrink_every: int
    shrink: float
    step_m: float
    step_deg: float
    min_size_m: float
    max_cost: float
    seed: int
    stack_min_overlap: float
    stack_gap_m: float
    stack_min_points: int
    stack_percentile: float
    duplicate_dist_m: float
    duplicate_size_frac: float

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> PrimitiveConfig:
        c = cfg["segmentation"]["primitive_fit"]
        out = cls(
            int(c["mask_downscale"]),
            int(c["min_views"]),
            int(c["min_mask_px"]),
            int(c["iterations"]),
            int(c["shrink_every"]),
            float(c["shrink"]),
            float(c["step_m"]),
            float(c["step_deg"]),
            float(c["min_size_m"]),
            float(c["max_cost"]),
            int(c["seed"]),
            float(c["stack_min_overlap"]),
            float(c["stack_gap_m"]),
            int(c["stack_min_points"]),
            float(c["stack_percentile"]),
            float(c["duplicate_dist_m"]),
            float(c["duplicate_size_frac"]),
        )
        if min(out.mask_downscale, out.min_views, out.iterations, out.shrink_every) < 1:
            raise ValueError("primitive_fit counts must be at least 1")
        if not 0.0 < out.shrink < 1.0:
            raise ValueError("shrink must be in (0, 1)")
        if min(out.step_m, out.step_deg, out.min_size_m, out.max_cost) <= 0:
            raise ValueError("primitive_fit steps, sizes and max_cost must be positive")
        if out.duplicate_dist_m <= 0 or not 0.0 < out.duplicate_size_frac < 1.0:
            raise ValueError("duplicate_dist_m must be positive and duplicate_size_frac in (0, 1)")
        return out


@dataclass(frozen=True)
class Primitive:
    """A primitive standing on the table plane, world frame, metres and radians.

    ``size_m`` is the extent along the yaw direction, across it, and the height. A sphere has
    three equal sizes. ``cost`` is the mean mask disagreement over ``views`` views."""

    kind: Kind
    center_xy_m: tuple[float, float]
    yaw_rad: float
    size_m: tuple[float, float, float]
    table_z_m: float
    cost: float
    views: int

    def outline(self) -> F64:
        """Points (N, 3) whose convex hull is the primitive: box corners, two cylinder rings or a
        sphere grid."""
        return _outline(self.kind, _params(self), self.table_z_m)


@dataclass(frozen=True)
class FitResult:
    """The cluster for one track and how it was made."""

    cluster: F64
    primitive: Primitive | None
    source: Literal["primitive", "union", "points"]
    reason: str


def _params(p: Primitive) -> F64:
    return np.array([*p.center_xy_m, p.yaw_rad, *p.size_m], dtype=np.float64)


def _outline(kind: Kind, params: F64, table_z: float) -> F64:
    cx, cy, yaw, length, width, height = (float(v) for v in params)
    c, s = np.cos(yaw), np.sin(yaw)
    turn = np.array([[c, s], [-s, c]])
    if kind == "box":
        half = np.array([length, width, height]) / 2
        grids = []
        for axis in range(3):
            others = [a for a in range(3) if a != axis]
            for side in (-1.0, 1.0):
                a, b = np.meshgrid(_FACE_STEPS, _FACE_STEPS, indexing="ij")
                face = np.zeros((9, 3))
                face[:, axis] = side
                face[:, others[0]] = a.ravel()
                face[:, others[1]] = b.ravel()
                grids.append(face * half)
        local = np.vstack(grids)
        local[:, 2] += half[2]  # faces span z in [0, height], like the corners did
    elif kind == "cylinder":
        ring = np.stack([length / 2 * np.cos(_RING), width / 2 * np.sin(_RING)], axis=1)
        local = np.vstack([ring for _ in _WALL_LEVELS])
    else:
        r = (length + width + height) / 6
        x = r * np.sin(_POLAR) * np.cos(_AZIMUTH)
        y = r * np.sin(_POLAR) * np.sin(_AZIMUTH)
        z = table_z + r + r * np.cos(_POLAR)
        return np.stack([cx + x, cy + y, z], axis=1)
    xy = local[:, :2] @ turn + np.array([cx, cy])
    if kind == "cylinder":
        levels = np.array(_WALL_LEVELS)
        z = table_z + np.repeat(levels, len(ring)) * height
    else:
        z = table_z + local[:, 2]
    out: F64 = np.c_[xy, z]
    return out


@dataclass(frozen=True)
class _View:
    pose: F64  # T_world_from_camera, 4x4
    mask: BoolMask  # this track, downscaled
    free: BoolMask  # pixels the object may not cover without cost: not on another track
    area: int


def _track_views(
    track: Track,
    tracks: Sequence[Track],
    world: WorldModel,
    lift: LiftConfig,
    cfg: PrimitiveConfig,
    smalls: dict[tuple[int, int], BoolMask] | None = None,
) -> list[_View]:
    """The usable views of ``track``. ``smalls`` caches downscaled masks by (track id, frame
    index), so a caller fitting every track of a scene shrinks each mask once, not once per
    track that sees it as an occluder."""
    import cv2  # heavy import, kept out of module import time

    scale = 1.0 / cfg.mask_downscale
    cache: dict[tuple[int, int], BoolMask] = {} if smalls is None else smalls

    def small_of(t: Track, frame_index: int) -> BoolMask:
        key = (t.track_id, frame_index)
        if key not in cache:
            out = cv2.resize(
                t.masks[frame_index].astype(np.uint8),
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_NEAREST,
            )
            cache[key] = np.asarray(out, dtype=bool)
        return cache[key]

    views: list[_View] = []
    for position in _views(world, lift):
        frame = world.frames[position]
        if frame.frame_index not in track.masks:
            continue
        m = small_of(track, frame.frame_index)
        area = int(m.sum())
        if area < cfg.min_mask_px:
            continue
        if m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any():
            continue  # cut off by the image edge: the outline there is not the object's
        others = np.zeros_like(m)
        for t in tracks:
            if t.track_id != track.track_id and frame.frame_index in t.masks:
                others |= small_of(t, frame.frame_index)
        views.append(_View(frame.T_world_from_camera.to_numpy(), m, ~m & ~others, area))
    return views


def _cost(
    kind: Kind, params: F64, views: Sequence[_View], world: WorldModel, cfg: PrimitiveConfig
) -> float:
    import cv2  # heavy import, kept out of module import time

    if float(np.min(params[3:])) < cfg.min_size_m:
        return float("inf")
    outline = _outline(kind, params, world.table_plane.offset_m)
    total = 0.0
    for view in views:
        cam = (outline - view.pose[:3, 3]) @ view.pose[:3, :3]
        if bool(np.any(cam[:, 2] <= MIN_DEPTH_M)):
            return float("inf")
        u, v = _pixels(cam[:, 0] / cam[:, 2], cam[:, 1] / cam[:, 2], world.intrinsics)
        uv = (np.stack([u, v], axis=1) / cfg.mask_downscale).astype(np.float32)
        hull: Any = cv2.convexHull(uv).astype(np.int32)
        # The silhouette lies inside the hull's box, so it is drawn only there: outside it
        # nothing is extra, and every mask pixel is missed. Same counts as a full canvas.
        corners = hull.reshape(-1, 2)
        height, width = view.mask.shape
        x0, y0 = max(int(corners[:, 0].min()), 0), max(int(corners[:, 1].min()), 0)
        x1 = min(int(corners[:, 0].max()) + 1, width)
        y1 = min(int(corners[:, 1].max()) + 1, height)
        if x1 <= x0 or y1 <= y0:
            extra, missed = 0, view.area
        else:
            sil = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
            cv2.fillConvexPoly(sil, hull - np.array([x0, y0], dtype=np.int32), 1)
            inside = sil.astype(bool)
            extra = int(np.count_nonzero(inside & view.free[y0:y1, x0:x1]))
            missed = view.area - int(np.count_nonzero(inside & view.mask[y0:y1, x0:x1]))
        total += (extra + missed) / view.area
    return total / len(views)


def initial_params(points: F64, table_z: float) -> F64:
    """Centre, yaw and sizes of the resting box of ``points``: the start of the search."""
    u, n = footprint_axes(points)
    a: F64 = points[:, :2] @ u
    b: F64 = points[:, :2] @ n
    a_lo, a_hi, b_lo, b_hi = float(np.min(a)), float(np.max(a)), float(np.min(b)), float(np.max(b))
    center = u * (a_lo + a_hi) / 2 + n * (b_lo + b_hi) / 2
    yaw = float(np.arctan2(u[1], u[0]))
    height = float(np.max(points[:, 2])) - table_z
    return np.array([center[0], center[1], yaw, a_hi - a_lo, b_hi - b_lo, height], dtype=np.float64)


def fit_primitive(
    track_id: int,
    views: Sequence[_View],
    start: F64,
    world: WorldModel,
    cfg: PrimitiveConfig,
) -> Primitive:
    """The best of the three kinds, each found by a seeded random walk from ``start``."""
    best: Primitive | None = None
    for k, kind in enumerate(KINDS):
        rng = np.random.default_rng([cfg.seed, track_id, k])
        params = start.copy()
        cost = _cost(kind, params, views, world, cfg)
        step = np.array(
            [cfg.step_m, cfg.step_m, np.radians(cfg.step_deg), cfg.step_m, cfg.step_m, cfg.step_m]
        )
        for it in range(cfg.iterations):
            trial = params + rng.normal(0.0, 1.0, 6) * step
            trial_cost = _cost(kind, trial, views, world, cfg)
            if trial_cost < cost:
                params, cost = trial, trial_cost
            if (it + 1) % cfg.shrink_every == 0:
                step = step * cfg.shrink
        size = (float(params[3]), float(params[4]), float(params[5]))
        if kind == "sphere":
            d = sum(size) / 3
            size = (d, d, d)
        candidate = Primitive(
            kind,
            (float(params[0]), float(params[1])),
            float(params[2]),
            size,
            world.table_plane.offset_m,
            cost,
            len(views),
        )
        if best is None or candidate.cost < best.cost:
            best = candidate
    assert best is not None
    return best


def _footprint_overlap(a: F64, b: F64) -> float:
    """The share of the box of ``a`` seen from above that lies inside the box of ``b``."""
    low, high = a[:, :2].min(axis=0), a[:, :2].max(axis=0)
    sides = np.minimum(high, b[:, :2].max(axis=0)) - np.maximum(low, b[:, :2].min(axis=0))
    return float(np.prod(np.clip(sides, 0.0, None)) / np.prod(high - low))


def seat_stacked(
    fits: Mapping[int, FitResult],
    points: Mapping[int, F64],
    table_z: float,
    cfg: PrimitiveConfig,
) -> dict[int, FitResult]:
    """Move clusters that stand on another object off the table; see the module docstring."""
    out = dict(fits)
    for track_id, fit in sorted(fits.items(), key=lambda kv: float(np.max(kv[1].cluster[:, 2]))):
        cloud = points.get(track_id)
        if cloud is None or len(cloud) < cfg.stack_min_points:
            continue
        cluster = out[track_id].cluster
        base, top = float(np.min(cluster[:, 2])), float(np.max(cluster[:, 2]))
        first = float(np.percentile(cloud[:, 2], cfg.stack_percentile))
        supports = [
            float(np.max(other.cluster[:, 2]))
            for k, other in out.items()
            if k != track_id
            and _footprint_overlap(cluster, other.cluster) >= cfg.stack_min_overlap
            and abs(float(np.max(other.cluster[:, 2])) - first) <= cfg.stack_gap_m
        ]
        if not supports or base - table_z > cfg.stack_gap_m:
            continue
        bottom = max(supports)
        last = max(top, float(np.percentile(cloud[:, 2], 100.0 - cfg.stack_percentile)))
        if last <= bottom:
            continue
        moved = cluster.copy()
        moved[:, 2] = bottom + (cluster[:, 2] - base) * (last - bottom) / (top - base)
        out[track_id] = replace(fit, cluster=moved, reason=f"{fit.reason}; stacked at {bottom:.3f}")
    return out


def fit_clusters(
    tracks: Sequence[Track],
    points: Mapping[int, F64],
    world: WorldModel,
    lift: LiftConfig,
    cfg: PrimitiveConfig,
) -> dict[int, FitResult]:
    """The fitted cluster per track id that has voted points; see the module docstring."""
    out: dict[int, FitResult] = {}
    table_z = world.table_plane.offset_m
    smalls: dict[tuple[int, int], BoolMask] = {}
    for track in tracks:
        cloud = points.get(track.track_id)
        if cloud is None or len(cloud) == 0:
            continue
        views = _track_views(track, tracks, world, lift, cfg, smalls)
        if len(views) < cfg.min_views:
            out[track.track_id] = FitResult(
                cloud, None, "points", f"only {len(views)} whole masks: the points are the cluster"
            )
            continue
        prim = fit_primitive(track.track_id, views, initial_params(cloud, table_z), world, cfg)
        if not np.isfinite(prim.cost):
            out[track.track_id] = FitResult(
                cloud, None, "points", "no primitive projects into every view"
            )
        elif prim.cost <= cfg.max_cost:
            out[track.track_id] = FitResult(
                prim.outline(), prim, "primitive", f"{prim.kind}, mask cost {prim.cost:.3f}"
            )
        else:
            out[track.track_id] = FitResult(
                np.vstack([prim.outline(), cloud]),
                prim,
                "union",
                f"best fit {prim.kind} has mask cost {prim.cost:.3f} above {cfg.max_cost}: "
                "union with the points",
            )
    return seat_stacked(out, points, table_z, cfg)


def drop_duplicate_fits(
    fits: Mapping[int, FitResult],
    tracks: Mapping[int, Track],
    cfg: PrimitiveConfig,
) -> dict[int, str]:
    """Ids to drop: two fitted primitives of one object tracked twice.

    A fragmented track gives frame-disjoint pieces whose voted points and table
    footprints depend on the covered views, so neither the lift duplicate check nor the
    repair merge fires on them. The fitted primitive does not depend on the views that
    way: two tracks of one box fit the same box. Both s12 video_b breakfast tracks
    agreed to 5 mm and 1% with no shared frame, while the closest distinct pair anywhere
    (s13 stacked boxes) differs by 48% in size. Shared frames veto the drop: stacked
    identical boxes co-occur, and only a mask-IoU check could tell those apart from
    concurrent double tracks (not implemented).
    """
    prims = {i: f.primitive for i, f in fits.items() if f.primitive is not None}
    dropped: dict[int, str] = {}
    ids = sorted(prims)
    for x in range(len(ids)):
        for y in range(x + 1, len(ids)):
            a, b = ids[x], ids[y]
            if a in dropped or b in dropped:
                continue
            pa, pb = prims[a], prims[b]
            if pa.kind != pb.kind:
                continue
            center = np.linalg.norm(np.array(pa.center_xy_m) - np.array(pb.center_xy_m))
            if float(center) >= cfg.duplicate_dist_m:
                continue
            sizes = (np.array(pa.size_m), np.array(pb.size_m))
            rel = np.abs(sizes[0] - sizes[1]).max() / max(sizes[0].max(), sizes[1].max())
            if float(rel) >= cfg.duplicate_size_frac:
                continue
            if set(tracks[a].masks) & set(tracks[b].masks):
                continue
            better = (pa.views, -pa.cost) >= (pb.views, -pb.cost)
            keep, drop = (a, b) if better else (b, a)
            dropped[drop] = f"duplicate of track {keep}"
    return dropped
