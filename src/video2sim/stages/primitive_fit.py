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

On the dev scenes this took the easy-tier median box error from 8.9% (trimmed points) to 4.3%,
15 of 19 objects within 10% (``docs/research/cluster_box_accuracy.md``). The result is the
cluster of ``ObjectInstance.point_cluster_path`` (human decision 2026-10-29).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
_RING = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
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
        )
        if min(out.mask_downscale, out.min_views, out.iterations, out.shrink_every) < 1:
            raise ValueError("primitive_fit counts must be at least 1")
        if not 0.0 < out.shrink < 1.0:
            raise ValueError("shrink must be in (0, 1)")
        if min(out.step_m, out.step_deg, out.min_size_m, out.max_cost) <= 0:
            raise ValueError("primitive_fit steps, sizes and max_cost must be positive")
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
    if kind == "box":
        half = np.array([length, width]) / 2
        local = np.array([[sx, sy] for sx in (-1, 1) for sy in (-1, 1)]) * half
    elif kind == "cylinder":
        local = np.stack([length / 2 * np.cos(_RING), width / 2 * np.sin(_RING)], axis=1)
    else:
        r = (length + width + height) / 6
        x = r * np.sin(_POLAR) * np.cos(_AZIMUTH)
        y = r * np.sin(_POLAR) * np.sin(_AZIMUTH)
        z = table_z + r + r * np.cos(_POLAR)
        return np.stack([cx + x, cy + y, z], axis=1)
    xy = local @ np.array([[c, s], [-s, c]]) + np.array([cx, cy])
    bottom = np.c_[xy, np.full(len(xy), table_z)]
    top = np.c_[xy, np.full(len(xy), table_z + height)]
    return np.vstack([bottom, top])


@dataclass(frozen=True)
class _View:
    pose: F64  # T_world_from_camera, 4x4
    mask: BoolMask  # this track, downscaled
    free: BoolMask  # pixels the object may not cover without cost: not on another track
    area: int


def _track_views(
    track: Track, tracks: Sequence[Track], world: WorldModel, lift: LiftConfig, cfg: PrimitiveConfig
) -> list[_View]:
    import cv2  # heavy import, kept out of module import time

    scale = 1.0 / cfg.mask_downscale

    def small(mask: BoolMask) -> BoolMask:
        out = cv2.resize(
            mask.astype(np.uint8), None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST
        )
        return np.asarray(out, dtype=bool)

    views: list[_View] = []
    for position in _views(world, lift):
        frame = world.frames[position]
        mask = track.masks.get(frame.frame_index)
        if mask is None:
            continue
        m = small(mask)
        area = int(m.sum())
        if area < cfg.min_mask_px:
            continue
        if m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any():
            continue  # cut off by the image edge: the outline there is not the object's
        others = np.zeros_like(m)
        for t in tracks:
            other = t.masks.get(frame.frame_index)
            if t.track_id != track.track_id and other is not None:
                others |= small(other)
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
        sil = np.zeros(view.mask.shape, dtype=np.uint8)
        cv2.fillConvexPoly(sil, hull, 1)
        inside = sil.astype(bool)
        extra = int(np.count_nonzero(inside & view.free))
        missed = int(np.count_nonzero(view.mask & ~inside))
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
    for track in tracks:
        cloud = points.get(track.track_id)
        if cloud is None or len(cloud) == 0:
            continue
        views = _track_views(track, tracks, world, lift, cfg)
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
    return out
