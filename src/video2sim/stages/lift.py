"""Lift tracked masks to point clusters in the world frame (P2.3).

The pipeline has no depth maps, so a cluster is a visual hull found by voting. A voxel grid
sits on the table around the tracked objects. Every voxel is projected into a subset of the
registered frames. A voxel belongs to a track when, among the views that see it, nearly all
have it inside that track's mask. That is the "several views agree" rule: a stray mask
or a single bad frame cannot add a point, and a voxel that most views place on the table or
on another object is dropped.

* A view where a track has no mask is skipped for that track: SAM found nothing there, and a
  lost object must not veto its own voxels (s03's ball is missing from the first 17 frames).
* Without ``occlusion_aware``, the slack in ``min_agreement`` is the only occlusion
  budget. A voxel hidden behind another object
  projects onto the occluder's mask in those views and still counts as in view.
* A voxel claimed by two tracks goes to the one with the higher agreement.
* Cells of 4 x 4 x 4 voxels are bounded first (``_candidates``): where no track can reach its
  threshold anywhere in a cell, its voxels are not voted. The bound is conservative, so the
  members are those of voting every voxel. On the local runs it votes 0.6 to 12% of them.
* With ``occlusion_aware`` a voxel that lands on another track's mask but not its own
  excuses that view (like a missing mask), instead of counting seen-but-not-hit.
  Without it, an object hidden in more than ``1 - min_agreement`` of views can never pass.
* Tracks below ``small_mask_px`` median mask pixels vote with ``small_min_agreement``
  and ``small_min_voxels``: fist-sized thresholds kill small objects, whose few voxels
  die first to mask jitter at the edges.
* Coinciding clusters are duplicates: when two cluster centers are closer than
  ``dup_dist_m`` the smaller cluster is dropped as a duplicate of the larger, so a
  fractured track can never export the same object twice.
* Only the largest connected group of each track is kept, which removes outlier voxels.
* A track with too few voxels is dropped and listed in the report. A track that reaches the
  top of the first-pass grid is re-voted on a taller grid and kept only when its column
  shows a cap: occupancy that falls off near the top and only incidental contact with
  the grid's lateral edges (below ``edge_frac`` of its voxels). A column that stays
  flat through the tall grid, or substantially touches its sides, is
  background (a wall or person) and is dropped. Hands and arms are not detected as such:
  they move between views, so their
  masks agree on almost no voxels and the track falls out through ``min_voxels``.

``lift_points`` is the second, independent lift of the same tracks: the sparse SfM points that
the masks agree on. P2.3 keeps both so Phase 3 can compare them as registration targets (the fit
in P3.2 compares surface points, which favours points; the hull gives a filled volume).

The visual hull is slightly larger than the object (silhouettes are loose), so cluster boxes
read a little high. A concave or hidden surface is never carved out from the inside.

The voted points keep a few strays: short SfM tracks triangulated with a small baseline can
project into the masks of most views of a short camera arc and float 1 to 2 cm off the object.
A box is set by its outermost points, so ``trim_extremes`` drops the ``trim_points_per_side``
outermost points on each side of the resting box (two per side took the easy-tier median box
error from 18.7% to 8.9%, ``docs/research/cluster_box_accuracy.md``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from video2sim.schemas import SE3, Intrinsics, Plane, WorldModel
from video2sim.stages.segmentation import CropMask
from video2sim.stages.track_repair import mask_footprint
from video2sim.stages.tracking import Track

F64 = NDArray[np.float64]
I64 = NDArray[np.int64]

# Camera depth below which a voxel is not projected: it is behind or touching the lens.
MIN_DEPTH_M = 0.05


@dataclass(frozen=True)
class LiftConfig:
    """Values from ``segmentation.lift`` in ``configs/default.yaml``."""

    voxel_m: float
    max_height_m: float
    margin_m: float
    max_extent_m: float
    max_views: int
    mask_downscale: int
    min_views: int
    min_agreement: float
    min_voxels: int
    min_points: int
    table_margin_m: float
    trim_points_per_side: int
    occlusion_aware: bool
    small_mask_px: float
    small_min_agreement: float
    small_min_voxels: int
    dup_dist_m: float
    tall_height_m: float
    cap_top_m: float
    cap_frac: float
    edge_frac: float = 0.05  # lateral-edge voxels below this share of the component pass

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> LiftConfig:
        c = cfg["segmentation"]["lift"]
        out = cls(
            float(c["voxel_m"]),
            float(c["max_height_m"]),
            float(c["margin_m"]),
            float(c["max_extent_m"]),
            int(c["max_views"]),
            int(c["mask_downscale"]),
            int(c["min_views"]),
            float(c["min_agreement"]),
            int(c["min_voxels"]),
            int(c["min_points"]),
            float(c["table_margin_m"]),
            int(c["trim_points_per_side"]),
            bool(c["occlusion_aware"]),
            float(c["small_mask_px"]),
            float(c["small_min_agreement"]),
            int(c["small_min_voxels"]),
            float(c["dup_dist_m"]),
            float(c["tall_height_m"]),
            float(c["cap_top_m"]),
            float(c["cap_frac"]),
            float(c["edge_frac"]),
        )
        if (
            min(out.voxel_m, out.max_height_m, out.margin_m, out.max_extent_m) <= 0
            or out.table_margin_m < 0
        ):
            raise ValueError("lift lengths must be positive")
        if out.tall_height_m < out.max_height_m:
            raise ValueError("tall_height_m must clear max_height_m")
        if out.cap_top_m <= 0 or out.cap_top_m > out.tall_height_m:
            raise ValueError("cap_top_m must be within (0, tall_height_m]")
        if not 0.0 < out.cap_frac <= 1.0:
            raise ValueError("cap_frac must be in (0, 1]")
        if not 0.0 < out.edge_frac <= 1.0:
            raise ValueError("edge_frac must be in (0, 1]")
        if (
            min(
                out.max_views,
                out.mask_downscale,
                out.min_views,
                out.min_voxels,
                out.min_points,
                out.small_min_voxels,
            )
            < 1
        ):
            raise ValueError("lift counts must be at least 1")
        if out.trim_points_per_side < 0:
            raise ValueError("trim_points_per_side must not be negative")
        if not 0.0 < out.min_agreement <= 1.0:
            raise ValueError("min_agreement must be in (0, 1]")
        if not 0.0 < out.small_min_agreement <= 1.0:
            raise ValueError("small_min_agreement must be in (0, 1]")
        if out.small_mask_px <= 0:
            raise ValueError("small_mask_px must be positive")
        if out.dup_dist_m < 0:
            raise ValueError("dup_dist_m must not be negative")
        return out


@dataclass(frozen=True)
class LiftReport:
    """What was kept and what was dropped, per track id. Nothing is dropped quietly."""

    views_used: int
    voxels: Mapping[int, int]
    dropped: Mapping[int, str]


@dataclass(frozen=True)
class PointVote:
    """Per-point best mask agreement over the tracks (step 1 of the residual design).

    ``best_ratio[i]`` is ``max_t h_t(p_i) / s_t(p_i)`` over tracks that saw point
    ``i`` in at least ``min_views`` views (0 when no track qualifies),
    ``best_track[i]`` is the winning track id (-1 for none) and ``best_seen[i]``
    is that track's view count. A point is explained when ``best_ratio`` clears
    the explain threshold and ``best_seen`` clears ``min_views``; the vote is a
    by-product of the lift, so it costs nothing extra.
    """

    best_ratio: F64
    best_track: I64
    best_seen: I64


# Voxels voted at a time. The (tracks x voxels) counts and the projection temporaries exist only
# for one chunk, so memory no longer grows with the grid: on the 155-frame run
# 20261103T021624Z the tall pass (13 M voxels) took the stage from 1.6 GB of masks to a 5.7 GB
# peak. The result does not depend on it.
_VOTE_CHUNK = 1 << 18


@dataclass(frozen=True)
class _VoxelGrid:
    """A voxel grid on the table, described instead of stored.

    ``shape`` cells of ``voxel_m`` from ``origin``, in C order; ``z0`` skips that many bottom
    layers (the upper block that the tall pass votes). Voxel ``i`` is cell
    ``unravel_index(i, shape)`` raised by ``z0``, its centre ``origin + (cell + 0.5) * voxel_m``.
    """

    origin: F64
    shape: tuple[int, int, int]
    voxel_m: float
    z0: int = 0

    @property
    def size(self) -> int:
        return self.shape[0] * self.shape[1] * self.shape[2]

    def cells(self, flat: I64) -> I64:
        """Integer cells (N, 3) of voxels ``flat``, with ``z0`` added."""
        x, y, z = np.unravel_index(flat, self.shape)
        return np.stack((x, y, z + self.z0), axis=1).astype(np.int64)

    def centers(self, flat: I64) -> F64:
        """World centres (N, 3) of voxels ``flat``."""
        out: F64 = self.origin + (self.cells(flat) + 0.5) * self.voxel_m
        return out


def _grid(centers_xy: F64, plane: Plane, cfg: LiftConfig, height_m: float) -> _VoxelGrid:
    """The voxel grid over the footprints. World +Z is up, the grid starts at the table."""
    lo = centers_xy.min(axis=0) - cfg.margin_m
    hi = centers_xy.max(axis=0) + cfg.margin_m
    if float(np.max(hi - lo)) > cfg.max_extent_m:
        raise ValueError(
            f"tracks span {float(np.max(hi - lo)):.2f} m, more than max_extent_m; "
            "one track is probably far from the objects"
        )
    shape = (
        int(np.ceil((hi[0] - lo[0]) / cfg.voxel_m)),
        int(np.ceil((hi[1] - lo[1]) / cfg.voxel_m)),
        int(np.ceil(height_m / cfg.voxel_m)),
    )
    origin = np.array([lo[0], lo[1], plane.offset_m], dtype=np.float64)
    return _VoxelGrid(origin, shape, cfg.voxel_m)


def _pixels(x: F64, y: F64, k: Intrinsics) -> tuple[F64, F64]:
    """Pixels of normalized camera coordinates, with the OpenCV lens model (k1 k2 p1 p2 k3)."""
    k1, k2, p1, p2, k3 = (list(k.distortion) + [0.0] * 5)[:5]
    r2 = x * x + y * y
    radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
    xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return k.fx * xd + k.cx, k.fy * yd + k.cy


def _views(world: WorldModel, cfg: LiftConfig) -> list[int]:
    """Positions in ``world.frames`` of the frames used, spread evenly."""
    n = len(world.frames)
    if n <= cfg.max_views:
        return list(range(n))
    return sorted({round(i * (n - 1) / (cfg.max_views - 1)) for i in range(cfg.max_views)})


def _largest_component(cells: I64) -> NDArray[np.bool_]:
    """Membership in the largest 26-connected group of integer cells (N, 3)."""
    n = len(cells)
    if n == 0:
        return np.zeros(0, dtype=bool)
    shifted = cells - cells.min(axis=0)
    dims = shifted.max(axis=0) + 3
    key = ((shifted[:, 0] + 1) * dims[1] + (shifted[:, 1] + 1)) * dims[2] + (shifted[:, 2] + 1)
    order = np.argsort(key)
    sorted_keys = key[order]
    offsets = [
        (dx * dims[1] + dy) * dims[2] + dz
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if (dx, dy, dz) != (0, 0, 0)
    ]
    neighbours = []
    for off in offsets:
        want = key + off
        pos = np.clip(np.searchsorted(sorted_keys, want), 0, n - 1)
        hit = sorted_keys[pos] == want
        neighbours.append(np.where(hit, order[pos], -1))
    nbr = np.stack(neighbours, axis=1)
    label = np.arange(n)
    while True:  # propagate the smallest label until nothing changes
        lowest = np.where(nbr >= 0, label[np.maximum(nbr, 0)], n).min(axis=1)
        new = np.minimum(label, lowest)
        if np.array_equal(new, label):
            break
        label = new
    counts = np.bincount(label, minlength=n)
    return np.asarray(label == int(np.argmax(counts)), dtype=bool)


def _count_dtype(views: int) -> type[np.unsignedinteger[Any]]:
    """The smallest unsigned type that holds a count of ``views``."""
    if views <= np.iinfo(np.uint8).max:
        return np.uint8
    if views <= np.iinfo(np.uint16).max:
        return np.uint16
    return np.uint32


@dataclass(frozen=True)
class _VoteView:
    """One frame of the vote: its pose, the tracks with a mask there and those masks shrunk."""

    R: F64  # R_world_from_camera
    t: F64  # camera centre in the world
    rows: list[int]
    smalls: dict[int, CropMask]  # by row: the downscaled mask
    free: NDArray[np.bool_] | None  # flattened: on no track's mask, when occlusion-aware


def _vote_views(
    tracks: Sequence[Track], world: WorldModel, cfg: LiftConfig, views: list[int]
) -> list[_VoteView]:
    """The downscaled masks of every view, made once for all voxel chunks."""
    import cv2  # heavy import, kept out of module import time

    out = []
    for position in views:
        frame = world.frames[position]
        rows = [r for r, t in enumerate(tracks) if frame.frame_index in t.masks]
        if not rows:
            continue  # SAM found nothing here: no evidence either way for any track
        smalls = {
            r: tracks[r].masks[frame.frame_index].downscaled(cfg.mask_downscale, cv2.INTER_AREA)
            for r in rows
        }
        free: NDArray[np.bool_] | None = None
        if cfg.occlusion_aware:
            if len({m.shape for m in smalls.values()}) > 1:
                raise ValueError(f"track masks of frame {frame.frame_index} differ in size")
            combined = np.zeros(smalls[rows[0]].shape, dtype=bool)
            for small in smalls.values():
                h, w = small.data.shape
                combined[small.top : small.top + h, small.left : small.left + w] |= small.data
            # On another track's mask only: hidden, not evidence against this track.
            free = ~combined.ravel()
        pose = frame.T_world_from_camera.to_numpy()
        out.append(_VoteView(pose[:3, :3], pose[:3, 3], rows, smalls, free))
    return out


# Coarse pass of the vote (N12): fine voxels per side of a cell, and the slack in pixels kept
# around a cell's projected bounds, far above float rounding. Neither changes a result.
_CELL = 4
_CELL_CHUNK = 1 << 14
_PIXEL_SLACK = 1e-3
_DEPTH_SLACK_M = 1e-9

Interval = tuple[F64, F64]


def _imul(a: Interval, b: Interval) -> Interval:
    p = (a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1])
    lo: F64 = np.minimum(np.minimum(p[0], p[1]), np.minimum(p[2], p[3]))
    hi: F64 = np.maximum(np.maximum(p[0], p[1]), np.maximum(p[2], p[3]))
    return lo, hi


def _iscale(c: float, a: Interval) -> Interval:
    return (c * a[0], c * a[1]) if c >= 0 else (c * a[1], c * a[0])


def _iadd(a: Interval, b: Interval) -> Interval:
    return a[0] + b[0], a[1] + b[1]


def _ishift(a: Interval, c: float) -> Interval:
    return a[0] + c, a[1] + c


def _isquare(a: Interval) -> Interval:
    lo2, hi2 = a[0] * a[0], a[1] * a[1]
    lo: F64 = np.where((a[0] <= 0) & (a[1] >= 0), 0.0, np.minimum(lo2, hi2))
    return lo, np.maximum(lo2, hi2)


def _pixel_bounds(x: Interval, y: Interval, k: Intrinsics) -> tuple[Interval, Interval]:
    """Enclosures of ``_pixels`` (u and v) over boxes of normalized coordinates, by interval
    arithmetic on the same lens model. They may be loose, never too tight (up to rounding)."""
    k1, k2, p1, p2, k3 = (float(d) for d in (list(k.distortion) + [0.0] * 5)[:5])
    x2, y2, xy = _isquare(x), _isquare(y), _imul(x, y)
    r2 = _iadd(x2, y2)
    radial = _ishift(_imul(r2, _ishift(_imul(r2, _ishift(_iscale(k3, r2), k2)), k1)), 1.0)
    xd = _iadd(
        _iadd(_imul(x, radial), _iscale(2.0 * p1, xy)),
        _iscale(p2, _iadd(r2, _iscale(2.0, x2))),
    )
    yd = _iadd(
        _iadd(_imul(y, radial), _iscale(p1, _iadd(r2, _iscale(2.0, y2)))),
        _iscale(2.0 * p2, xy),
    )
    return _ishift(_iscale(k.fx, xd), k.cx), _ishift(_iscale(k.fy, yd), k.cy)


def _window_any(
    table: NDArray[np.int64], small: CropMask, r0: I64, r1: I64, c0: I64, c1: I64
) -> NDArray[np.bool_]:
    """Whether ``small`` has a pixel in each window of rows ``r0..r1`` and columns ``c0..c1``
    (inclusive, frame coordinates); ``table`` is the summed-area table of ``small.data``."""
    h, w = small.data.shape
    if h == 0 or w == 0:
        return np.zeros(len(r0), dtype=bool)
    a0 = np.maximum(r0, small.top) - small.top
    a1 = np.minimum(r1, small.top + h - 1) - small.top
    b0 = np.maximum(c0, small.left) - small.left
    b1 = np.minimum(c1, small.left + w - 1) - small.left
    empty = (a0 > a1) | (b0 > b1)
    a0, a1 = np.clip(a0, 0, h - 1), np.clip(a1, 0, h - 1)
    b0, b1 = np.clip(b0, 0, w - 1), np.clip(b1, 0, w - 1)
    total = table[a1 + 1, b1 + 1] - table[a0, b1 + 1] - table[a1 + 1, b0] + table[a0, b0]
    return np.asarray(~empty & (total > 0), dtype=bool)


def _cell_ends(grid: _VoxelGrid, axis: int) -> tuple[F64, F64]:
    """World coordinate of the first and last fine voxel centre of each cell along ``axis``,
    computed as ``_VoxelGrid.centers`` computes them."""
    n = grid.shape[axis]
    first = np.arange(0, n, _CELL, dtype=np.int64)
    last = np.minimum(first + _CELL, n) - 1
    offset = grid.z0 if axis == 2 else 0
    lo: F64 = grid.origin[axis] + (first + offset + 0.5) * grid.voxel_m
    hi: F64 = grid.origin[axis] + (last + offset + 0.5) * grid.voxel_m
    return lo, hi


def _candidates(
    grid: _VoxelGrid,
    n_tracks: int,
    prepared: Sequence[_VoteView],
    world: WorldModel,
    cfg: LiftConfig,
    agree: F64,
    chunk: int = _CELL_CHUNK,
) -> I64:
    """The fine voxels that can belong to a track, found on cells of ``_CELL`` voxels a side.

    For a cell and a view, the centres of the cell's voxels lie in the box spanned by its
    corner centres; their camera depth and normalized image coordinates take their extremes
    at the corners, and interval arithmetic bounds the lens model, so every voxel lands in a
    known pixel window. Per track that gives an upper bound on its hits (views where its mask
    reaches the window, ``A``) and a lower bound on the views that see a voxel and miss
    (``B``: every voxel is in the image and, occlusion-aware, no mask at all reaches the
    window). A voxel's ratio is at most ``A / (A + B)`` and it needs ``min_views`` views, so
    a cell where no track can clear its threshold ``agree`` holds no member, and is skipped.
    Cells are bounded ``chunk`` at a time.
    """
    k = world.intrinsics
    scale = 1.0 / cfg.mask_downscale
    ends = [_cell_ends(grid, a) for a in range(3)]
    cells = (len(ends[0][0]), len(ends[1][0]), len(ends[2][0]))
    n_cells = cells[0] * cells[1] * cells[2]
    tables = [{r: _summed_area(view.smalls[r]) for r in view.rows} for view in prepared]
    slack = _PIXEL_SLACK
    keep = np.zeros(n_cells, dtype=bool)
    for start in range(0, n_cells, chunk):
        stop = min(start + chunk, n_cells)
        index = np.unravel_index(np.arange(start, stop, dtype=np.int64), cells)
        pick = [np.stack([lo[i], hi[i]], axis=1) for (lo, hi), i in zip(ends, index, strict=True)]
        corners = np.stack(
            np.broadcast_arrays(
                pick[0][:, :, None, None], pick[1][:, None, :, None], pick[2][:, None, None, :]
            ),
            axis=-1,
        ).reshape(-1, 3)
        n = stop - start
        hit_views = np.zeros((n_tracks, n), dtype=np.int32)  # A
        miss_views = np.zeros((n_tracks, n), dtype=np.int32)  # B
        seen_views = np.zeros((n_tracks, n), dtype=np.int32)  # most views that can see one
        for view, table in zip(prepared, tables, strict=True):
            cam = ((corners - view.t) @ view.R).reshape(n, 8, 3)
            z = cam[:, :, 2]
            bounded = z.min(axis=1) > MIN_DEPTH_M + _DEPTH_SLACK_M
            safe = np.where(bounded[:, None], z, 1.0)
            nx_, ny_ = cam[:, :, 0] / safe, cam[:, :, 1] / safe
            (u0, u1), (v0, v1) = _pixel_bounds(
                (nx_.min(axis=1), nx_.max(axis=1)), (ny_.min(axis=1), ny_.max(axis=1)), k
            )
            outside = (u1 < -slack) | (u0 >= k.width_px + slack) | (v1 < -slack)
            outside |= v0 >= k.height_px + slack
            reach = ~((z.max(axis=1) <= MIN_DEPTH_M - _DEPTH_SLACK_M) | (bounded & outside))
            all_in = bounded & (u0 >= slack) & (u1 < k.width_px - slack)
            all_in &= (v0 >= slack) & (v1 < k.height_px - slack)
            whole = np.iinfo(np.int32).max  # an unbounded cell may land anywhere
            c0 = np.where(bounded, np.floor(np.maximum(u0, 0.0) * scale) - 1, 0).astype(np.int64)
            c1 = np.where(bounded, np.floor(np.minimum(u1, k.width_px) * scale) + 1, whole)
            r0 = np.where(bounded, np.floor(np.maximum(v0, 0.0) * scale) - 1, 0).astype(np.int64)
            r1 = np.where(bounded, np.floor(np.minimum(v1, k.height_px) * scale) + 1, whole)
            c1, r1 = c1.astype(np.int64), r1.astype(np.int64)
            any_mask = np.zeros(n, dtype=bool)
            own: dict[int, NDArray[np.bool_]] = {}
            for r in view.rows:
                own[r] = _window_any(table[r], view.smalls[r], r0, r1, c0, c1)
                any_mask |= own[r]
                hit_views[r] += own[r] & reach
                seen_views[r] += reach
            for r in view.rows:
                # A sure miss: every voxel is in view, off this track's mask and, occlusion-
                # aware, off every mask (on another one it would be excused, not missed).
                clear = any_mask if view.free is not None else own[r]
                miss_views[r] += all_in & ~clear
        total = (hit_views + miss_views).astype(np.float64)
        bound = np.where(hit_views > 0, hit_views / np.maximum(total, 1.0), 0.0)
        possible = (seen_views >= cfg.min_views) & (hit_views > 0) & (bound >= agree[:, None])
        keep[start:stop] = possible.any(axis=0)
    fine = keep.reshape(cells).repeat(_CELL, 0).repeat(_CELL, 1).repeat(_CELL, 2)
    fine = fine[: grid.shape[0], : grid.shape[1], : grid.shape[2]]
    return np.flatnonzero(fine).astype(np.int64)


def _summed_area(small: CropMask) -> NDArray[np.int64]:
    """Summed-area table of a crop's pixels, one row and column of zeros in front."""
    table = np.zeros((small.data.shape[0] + 1, small.data.shape[1] + 1), dtype=np.int64)
    table[1:, 1:] = np.cumsum(np.cumsum(small.data, axis=0), axis=1)
    return table


def _vote(
    grid: _VoxelGrid,
    tracks: Sequence[Track],
    world: WorldModel,
    cfg: LiftConfig,
    views: list[int],
    chunk: int = _VOTE_CHUNK,
    agree: F64 | None = None,
) -> tuple[F64, NDArray[np.int32]]:
    """The best agreement ratio per voxel of ``grid`` and the track (row) that has it.

    Given the tracks' thresholds ``agree``, only the voxels of ``_candidates`` are voted and the
    rest keep ratio 0: none of them could clear its track's threshold, so the members
    (``ratio >= agree[best]``) are those of the full vote.

    A track's ratio is hits over seen views, 0 below ``min_views`` seen views. A voxel belongs
    to its best track once that ratio clears the track's agreement threshold; ties go to the
    lower row. The caller turns ``(ratio, best)`` into members.

    Only voxels in front of the camera and inside the image (``ok``) take part in a view.
    Occlusion-aware, a view counts for a track where the voxel is on its own mask or on no
    mask (``ok & (~combined | own)``), and a hit where it is on its own mask (``ok & own``).
    Because a track's own mask is part of ``combined``, the shared term ``~combined`` is
    looked up once per view and the own mask once per track. Voxels are voted ``chunk`` at a
    time, so the counts of all tracks exist only for one chunk.
    """
    k = world.intrinsics
    scale = 1.0 / cfg.mask_downscale
    prepared = _vote_views(tracks, world, cfg, views)
    counts = _count_dtype(len(views))
    best_ratio = np.zeros(grid.size, dtype=np.float64)
    best = np.zeros(grid.size, dtype=np.int32)
    voted = None if agree is None else _candidates(grid, len(tracks), prepared, world, cfg, agree)
    n_voted = grid.size if voted is None else len(voted)
    for start in range(0, n_voted, chunk):
        stop = min(start + chunk, n_voted)
        index = np.arange(start, stop, dtype=np.int64) if voted is None else voted[start:stop]
        centers = grid.centers(index)
        seen = np.zeros((len(tracks), stop - start), dtype=counts)
        hits = np.zeros((len(tracks), stop - start), dtype=counts)
        for view in prepared:
            cam = (centers - view.t) @ view.R  # R^T (x - t), camera frame
            z = cam[:, 2]
            ok = z > MIN_DEPTH_M
            safe = np.where(ok, z, 1.0)
            u, v = _pixels(cam[:, 0] / safe, cam[:, 1] / safe, k)
            u, v = np.where(ok, u, -1.0), np.where(ok, v, -1.0)
            ok &= (u >= 0) & (u < k.width_px) & (v >= 0) & (v < k.height_px)
            inside = np.flatnonzero(ok)
            ui = np.clip((u[inside] * scale).astype(np.int64), 0, None)
            vi = np.clip((v[inside] * scale).astype(np.int64), 0, None)
            pixels: dict[tuple[int, int], tuple[I64, I64]] = {}  # (row, col) per mask shape
            for small in view.smalls.values():
                h, w = small.shape
                if (h, w) not in pixels:
                    pixels[(h, w)] = (np.minimum(vi, h - 1), np.minimum(ui, w - 1))
            free = None
            if view.free is not None:
                (h, w), (pr, pc) = next(iter(pixels.items()))
                free = view.free[pr * w + pc]
            for r in view.rows:
                small = view.smalls[r]
                own = small.at(*pixels[small.shape])
                hits[r, inside] += own
                seen[r, inside] += (own | free) if free is not None else 1
        # The first maximum wins, as argmax over the tracks would pick it.
        top: F64 = np.zeros(stop - start, dtype=np.float64)
        winner = np.zeros(stop - start, dtype=np.int32)
        for r in range(len(tracks)):
            s = seen[r]
            ratio = np.where(
                s >= cfg.min_views,
                hits[r].astype(np.float64) / np.maximum(s, 1).astype(np.float64),
                0.0,
            )
            if r == 0:
                top = ratio
                continue
            better = ratio > top
            top = np.where(better, ratio, top)
            winner[better] = r
        best_ratio[index] = top
        best[index] = winner
    return best_ratio, best


def _cap_reason(cells: I64, shape: tuple[int, int, int], cfg: LiftConfig) -> str | None:
    """Why a column filling the tall grid is background, or None when it caps.

    A tabletop object is an island: its layer occupancy falls off near its top and only
    incidentally touches the grid's lateral edges. A wall or person stays flat through
    the top layer, or spans past the tracked feet to the grid's sides. The edge test is
    a share, not any touch: sparse objects (flower gaps) agree on background margin
    voxels that connect to the core, and whether dying fragments trim that bridge must
    not decide the object.
    """
    top_layers = max(1, round(cfg.cap_top_m / cfg.voxel_m))
    counts = np.bincount(cells[:, 2], minlength=shape[2])
    peak = int(np.max(counts))
    if peak == 0 or float(np.sum(counts[shape[2] - top_layers :])) > cfg.cap_frac * peak:
        return "fills the grid to its top"
    nx, ny = shape[0], shape[1]
    on_edge = (
        (cells[:, 0] <= 0) | (cells[:, 0] >= nx - 1) | (cells[:, 1] <= 0) | (cells[:, 1] >= ny - 1)
    )
    if float(np.sum(on_edge)) >= cfg.edge_frac * len(cells):
        return "fills the grid to its edge"
    return None


def lift_tracks(
    tracks: Sequence[Track], world: WorldModel, cfg: LiftConfig
) -> tuple[dict[int, F64], LiftReport]:
    """Point clusters (world frame, metres) by track id, and the report of what was dropped."""
    poses: dict[int, SE3] = {f.frame_index: f.T_world_from_camera for f in world.frames}
    feet = [
        p
        for t in tracks
        for f, m in t.masks.items()
        if (p := mask_footprint(m, poses[f], world.intrinsics, world.table_plane)) is not None
    ]
    if not tracks or not feet:
        return {}, LiftReport(0, {}, {t.track_id: "no usable mask" for t in tracks})
    xy = np.array(feet)
    grid = _grid(xy, world.table_plane, cfg, cfg.max_height_m)
    shape = grid.shape

    views = _views(world, cfg)
    areas = np.array(
        [float(np.median([m.area for m in t.masks.values()])) if t.masks else 0.0 for t in tracks]
    )
    small_track = areas < cfg.small_mask_px
    agree = np.where(small_track, cfg.small_min_agreement, cfg.min_agreement)
    minvox = np.where(small_track, cfg.small_min_voxels, cfg.min_voxels)
    ratio, best = _vote(grid, tracks, world, cfg, views, agree=agree)  # best ratio, its track

    clusters: dict[int, F64] = {}
    dropped: dict[int, str] = {}
    voxels: dict[int, int] = {}
    top_layer = shape[2] - 1
    tall_rows: list[int] = []
    tall_low_cells: dict[int, I64] = {}
    for row, track in enumerate(tracks):
        member = np.nonzero((ratio >= agree[row]) & (best == row))[0]
        if member.size == 0:
            dropped[track.track_id] = "no voxel is inside the mask in enough views"
            continue
        cells = np.stack(np.unravel_index(member, shape), axis=1).astype(np.int64)
        raw_cells = cells
        keep = _largest_component(cells)
        member, cells = member[keep], cells[keep]
        if len(member) < minvox[row]:
            dropped[track.track_id] = f"only {len(member)} voxels agree"
        elif int(np.max(cells[:, 2])) >= top_layer:
            tall_rows.append(row)  # decided on the tall grid below
            tall_low_cells[row] = raw_cells  # reused below; the tall vote covers the rest
        else:
            clusters[track.track_id] = grid.centers(member)
            voxels[track.track_id] = len(member)
    if tall_rows:
        # Second pass, paid only when something reaches the first pass. The layers below
        # the first-pass top reuse that vote; only the layers above it are re-voted, on
        # the same footprint, so a tall object can show its cap.
        tall_grid = _grid(xy, world.table_plane, cfg, cfg.tall_height_m)
        tall_shape = tall_grid.shape
        nx, ny, nz = tall_shape
        low_layers = shape[2]
        high = replace(tall_grid, shape=(nx, ny, nz - low_layers), z0=low_layers)
        tall_ratio, tall_best = _vote(high, tracks, world, cfg, views, agree=agree)
        tall_top = tall_shape[2] - 1
        for row in tall_rows:
            track = tracks[row]
            pos = np.nonzero((tall_ratio >= agree[row]) & (tall_best == row))[0]
            hx, hy, hz = np.unravel_index(pos, (nx, ny, nz - low_layers))
            high_cells = np.stack((hx, hy, hz + low_layers), axis=1).astype(np.int64)
            cells = np.concatenate((tall_low_cells[row], high_cells))
            member = (cells[:, 0] * ny + cells[:, 1]) * nz + cells[:, 2]
            if member.size == 0:
                dropped[track.track_id] = "no voxel is inside the mask in enough views"
                continue
            keep = _largest_component(cells)
            member, cells = member[keep], cells[keep]
            if len(member) < minvox[row]:
                dropped[track.track_id] = f"only {len(member)} voxels agree"
            elif int(np.max(cells[:, 2])) >= tall_top:
                dropped[track.track_id] = "reaches the top of the tall grid (not a tabletop object)"
            elif (reason := _cap_reason(cells, tall_shape, cfg)) is not None:
                dropped[track.track_id] = f"{reason} (not a tabletop object)"
            else:
                clusters[track.track_id] = tall_grid.centers(member)
                voxels[track.track_id] = len(member)
    if cfg.dup_dist_m > 0.0:
        dups = _duplicate_tracks(clusters, cfg.dup_dist_m)
        for i, reason in dups.items():
            del clusters[i]
            voxels.pop(i, None)
            dropped[i] = reason
    return clusters, LiftReport(len(views), voxels, dropped)


def _duplicate_tracks(clusters: Mapping[int, F64], dup_dist_m: float) -> dict[int, str]:
    """Ids to drop: clusters coinciding within ``dup_dist_m`` keep the larger one.

    A fractured track lifts twice to the same object; the pieces agree voxel for voxel,
    so their centers coincide far below any distance two real objects can have.
    """
    means = {i: np.asarray(c).mean(axis=0) for i, c in clusters.items()}
    dups: dict[int, str] = {}
    for a in sorted(clusters):
        for b in sorted(clusters):
            if b <= a or a in dups or b in dups:
                continue
            if float(np.linalg.norm(means[a] - means[b])) < dup_dist_m:
                keep, drop = (a, b) if len(clusters[a]) >= len(clusters[b]) else (b, a)
                dups[drop] = f"duplicate of track {keep}"
    return dups


def footprint_axes(points: F64) -> tuple[F64, F64]:
    """Unit directions (2,) of the long and short side of the minimum-area rectangle around the
    XY footprint, the box convention of the ground truth."""
    import cv2  # heavy import, kept out of module import time

    if len(points) < 3:
        return np.array([1.0, 0.0]), np.array([0.0, 1.0])
    (_, _), (w, h), angle = cv2.minAreaRect(points[:, :2].astype(np.float32))
    a = np.radians(angle if w >= h else angle + 90.0)
    u = np.array([np.cos(a), np.sin(a)])
    return u, np.array([-u[1], u[0]])


def trim_extremes(points: F64, per_side: int, min_left: int) -> F64:
    """The points without the ``per_side`` outermost ones on each of the six sides of the resting
    box (long side, short side, height). A cluster that would keep fewer than ``min_left``
    points is returned whole: a sparse cluster has no strays to spare."""
    if per_side == 0 or len(points) <= 6 * per_side + min_left:
        return points
    u, n = footprint_axes(points)
    keep = np.ones(len(points), dtype=bool)
    for coord in (points[:, :2] @ u, points[:, :2] @ n, points[:, 2]):
        order = np.argsort(coord, kind="stable")
        keep[order[:per_side]] = False
        keep[order[-per_side:]] = False
    out: F64 = points[keep]
    return out


def load_points(path: Path) -> F64:
    """Vertices of the reconstruction's sparse cloud (world frame, metres)."""
    import trimesh  # heavy import, kept out of module import time

    cloud: Any = trimesh.load(path)
    return np.asarray(cloud.vertices, dtype=np.float64)


def point_vote(
    tracks: Sequence[Track], world: WorldModel, points: F64, cfg: LiftConfig
) -> PointVote:
    """Best per-point agreement ratio over ``tracks`` and the winning track id.

    For each point and track, ``s`` counts the views where the track has a mask
    and the point projects inside the frame, ``h`` those where the mask also
    covers it; the ratio is ``h / s`` and is zeroed when ``s < min_views``.
    """
    k = world.intrinsics
    views = _views(world, cfg)
    seen = np.zeros((len(tracks), len(points)), dtype=np.int32)
    hits = np.zeros((len(tracks), len(points)), dtype=np.int32)
    for position in views:
        frame = world.frames[position]
        pose = frame.T_world_from_camera.to_numpy()
        cam = (points - pose[:3, 3]) @ pose[:3, :3]
        z = cam[:, 2]
        ok = z > MIN_DEPTH_M
        safe = np.where(ok, z, 1.0)
        u, v = _pixels(cam[:, 0] / safe, cam[:, 1] / safe, k)
        ok &= (u >= 0) & (u < k.width_px) & (v >= 0) & (v < k.height_px)
        ui = np.clip(u.astype(np.int64), 0, k.width_px - 1)
        vi = np.clip(v.astype(np.int64), 0, k.height_px - 1)
        for row, track in enumerate(tracks):
            mask = track.masks.get(frame.frame_index)
            if mask is None:
                continue
            seen[row] += ok
            hits[row] += ok & mask.at(vi, ui)
    ratio = np.where(seen >= cfg.min_views, hits / np.maximum(seen, 1), 0.0)
    if len(tracks) == 0 or len(points) == 0:
        return PointVote(
            np.zeros(len(points)),
            np.full(len(points), -1, dtype=np.int64),
            np.zeros(len(points), dtype=np.int64),
        )
    best = ratio.argmax(axis=0)
    rows = np.arange(len(points))
    return PointVote(
        best_ratio=np.asarray(ratio[best, rows], dtype=np.float64),
        best_track=np.array([tracks[int(b)].track_id for b in best], dtype=np.int64),
        best_seen=np.asarray(seen[best, rows], dtype=np.int64),
    )


def lift_points(
    tracks: Sequence[Track], world: WorldModel, points: F64, cfg: LiftConfig
) -> tuple[dict[int, F64], LiftReport, PointVote]:
    """The second lift: sparse SfM points that the masks agree on, per track id.

    Each point was triangulated from image features, so it lies on a visible surface. It is
    kept for the track whose mask holds it in at least ``min_agreement`` of the views that
    see it, with the same view rules as ``lift_tracks``. There is no volume, so the box of a
    cluster follows the visible faces, but a plain or glossy object may have few points or
    none, and a point in front of the object's edge can pass the vote (there is no occlusion
    test). Each cluster goes through ``trim_extremes`` to drop stray points.

    Also returns the per-point vote (best ratio and winning track) that the
    residual-cluster step uses to decide which points are already explained.
    """
    vote = point_vote(tracks, world, points, cfg)
    views = _views(world, cfg)
    if len(tracks) == 0:
        return {}, LiftReport(len(views), {}, {}), vote
    height = points[:, 2] - world.table_plane.offset_m
    usable = (vote.best_ratio >= cfg.min_agreement) & (height > cfg.table_margin_m)
    usable &= height < cfg.tall_height_m
    clusters: dict[int, F64] = {}
    dropped: dict[int, str] = {}
    counts: dict[int, int] = {}
    for track in tracks:
        voted = points[usable & (vote.best_track == track.track_id)]
        chosen = trim_extremes(voted, cfg.trim_points_per_side, cfg.min_points)
        if len(chosen) < cfg.min_points:
            dropped[track.track_id] = f"only {len(chosen)} sparse points agree"
        else:
            clusters[track.track_id] = chosen
            counts[track.track_id] = len(chosen)
    return clusters, LiftReport(len(views), counts, dropped), vote
