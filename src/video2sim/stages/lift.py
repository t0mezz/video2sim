"""Lift tracked masks to point clusters in the world frame (P2.3).

The pipeline has no depth maps, so a cluster is a visual hull found by voting. A voxel grid
sits on the table around the tracked objects. Every voxel is projected into a subset of the
registered frames. A voxel belongs to a track when, among the views that see it, nearly all
have it inside that track's mask. That is the "several views agree" rule: a stray mask
or a single bad frame cannot add a point, and a voxel that most views place on the table or
on another object is dropped.

* A view where a track has no mask is skipped for that track: SAM found nothing there, and a
  lost object must not veto its own voxels (s03's ball is missing from the first 17 frames).
* The slack in ``min_agreement`` is for occlusion. A voxel hidden behind another object
  projects onto the occluder's mask in those views and still counts as in view.
* A voxel claimed by two tracks goes to the one with the higher agreement.
* Only the largest connected group of each track is kept, which removes outlier voxels.
* A track with too few voxels, or one that reaches the top of the grid, is dropped and listed
  in the report. Hands and arms are not detected as such: they move between views, so their
  masks agree on almost no voxels and the track falls out through ``min_voxels``.

``lift_points`` is the second, independent lift of the same tracks: the sparse SfM points that
the masks agree on. P2.3 keeps both so Phase 3 can compare them as registration targets (the fit
in P3.2 compares surface points, which favours points; the hull gives a filled volume).

The visual hull is slightly larger than the object (silhouettes are loose), so cluster boxes
read a little high. A concave or hidden surface is never carved out from the inside.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from video2sim.schemas import SE3, Intrinsics, Plane, WorldModel
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
        )
        if (
            min(out.voxel_m, out.max_height_m, out.margin_m, out.max_extent_m) <= 0
            or out.table_margin_m < 0
        ):
            raise ValueError("lift lengths must be positive")
        if (
            min(out.max_views, out.mask_downscale, out.min_views, out.min_voxels, out.min_points)
            < 1
        ):
            raise ValueError("lift counts must be at least 1")
        if not 0.0 < out.min_agreement <= 1.0:
            raise ValueError("min_agreement must be in (0, 1]")
        return out


@dataclass(frozen=True)
class LiftReport:
    """What was kept and what was dropped, per track id. Nothing is dropped quietly."""

    views_used: int
    voxels: Mapping[int, int]
    dropped: Mapping[int, str]


def _grid(centers_xy: F64, plane: Plane, cfg: LiftConfig) -> tuple[F64, tuple[int, int, int]]:
    """Voxel centres (N, 3) and the grid shape. World +Z is up, the grid starts at the table."""
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
        int(np.ceil(cfg.max_height_m / cfg.voxel_m)),
    )
    origin = np.array([lo[0], lo[1], plane.offset_m], dtype=np.float64)
    idx = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), axis=-1)
    centers = origin + (idx.reshape(-1, 3) + 0.5) * cfg.voxel_m
    return centers.astype(np.float64), shape


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


def lift_tracks(
    tracks: Sequence[Track], world: WorldModel, cfg: LiftConfig
) -> tuple[dict[int, F64], LiftReport]:
    """Point clusters (world frame, metres) by track id, and the report of what was dropped."""
    import cv2  # heavy import, kept out of module import time

    poses: dict[int, SE3] = {f.frame_index: f.T_world_from_camera for f in world.frames}
    feet = [
        p
        for t in tracks
        for f, m in t.masks.items()
        if (p := mask_footprint(m, poses[f], world.intrinsics, world.table_plane)) is not None
    ]
    if not tracks or not feet:
        return {}, LiftReport(0, {}, {t.track_id: "no usable mask" for t in tracks})
    centers, shape = _grid(np.array(feet), world.table_plane, cfg)

    views = _views(world, cfg)
    k = world.intrinsics
    scale = 1.0 / cfg.mask_downscale
    seen = np.zeros((len(tracks), len(centers)), dtype=np.int32)
    hits = np.zeros((len(tracks), len(centers)), dtype=np.int32)
    for position in views:
        frame = world.frames[position]
        pose = frame.T_world_from_camera.to_numpy()
        cam = (centers - pose[:3, 3]) @ pose[:3, :3]  # R^T (x - t), camera frame
        z = cam[:, 2]
        ok = z > MIN_DEPTH_M
        safe = np.where(ok, z, 1.0)
        u, v = _pixels(cam[:, 0] / safe, cam[:, 1] / safe, k)
        u, v = np.where(ok, u, -1.0), np.where(ok, v, -1.0)
        ok &= (u >= 0) & (u < k.width_px) & (v >= 0) & (v < k.height_px)
        ui = np.clip((u * scale).astype(np.int64), 0, None)
        vi = np.clip((v * scale).astype(np.int64), 0, None)
        for row, track in enumerate(tracks):
            mask = track.masks.get(frame.frame_index)
            if mask is None:
                continue  # SAM found nothing here: no evidence either way for this track
            small = cv2.resize(
                mask.astype(np.uint8),
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA,
            ).astype(bool)
            inside = (
                ok & small[np.minimum(vi, small.shape[0] - 1), np.minimum(ui, small.shape[1] - 1)]
            )
            seen[row] += ok
            hits[row] += inside

    ratio = np.where(seen >= cfg.min_views, hits / np.maximum(seen, 1), 0.0)
    best = ratio.argmax(axis=0)
    claimed = ratio.max(axis=0) >= cfg.min_agreement

    clusters: dict[int, F64] = {}
    dropped: dict[int, str] = {}
    voxels: dict[int, int] = {}
    top_layer = shape[2] - 1
    for row, track in enumerate(tracks):
        member = np.nonzero(claimed & (best == row))[0]
        if member.size == 0:
            dropped[track.track_id] = "no voxel is inside the mask in enough views"
            continue
        cells = np.stack(np.unravel_index(member, shape), axis=1).astype(np.int64)
        keep = _largest_component(cells)
        member, cells = member[keep], cells[keep]
        if len(member) < cfg.min_voxels:
            dropped[track.track_id] = f"only {len(member)} voxels agree"
        elif int(np.max(cells[:, 2])) >= top_layer:
            dropped[track.track_id] = "reaches the top of the grid (not a tabletop object)"
        else:
            clusters[track.track_id] = centers[member]
            voxels[track.track_id] = len(member)
    return clusters, LiftReport(len(views), voxels, dropped)


def load_points(path: Path) -> F64:
    """Vertices of the reconstruction's sparse cloud (world frame, metres)."""
    import trimesh  # heavy import, kept out of module import time

    cloud: Any = trimesh.load(path)
    return np.asarray(cloud.vertices, dtype=np.float64)


def lift_points(
    tracks: Sequence[Track], world: WorldModel, points: F64, cfg: LiftConfig
) -> tuple[dict[int, F64], LiftReport]:
    """The second lift: sparse SfM points that the masks agree on, per track id.

    Each point was triangulated from image features, so it lies on a visible surface. It is
    kept for the track whose mask holds it in at least ``min_agreement`` of the views that
    see it, with the same view rules as ``lift_tracks``. There is no volume, so the box of a
    cluster follows the visible faces and is never larger than the object, but a plain or
    glossy object may have few points or none, and a point in front of the object's edge can
    pass the vote (there is no occlusion test).
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
            hits[row] += ok & mask[vi, ui]
    ratio = np.where(seen >= cfg.min_views, hits / np.maximum(seen, 1), 0.0)
    if len(tracks) == 0:
        return {}, LiftReport(len(views), {}, {})
    best = ratio.argmax(axis=0)
    height = points[:, 2] - world.table_plane.offset_m
    usable = (ratio.max(axis=0) >= cfg.min_agreement) & (height > cfg.table_margin_m)
    usable &= height < cfg.max_height_m
    clusters: dict[int, F64] = {}
    dropped: dict[int, str] = {}
    counts: dict[int, int] = {}
    for row, track in enumerate(tracks):
        chosen = points[usable & (best == row)]
        if len(chosen) < cfg.min_points:
            dropped[track.track_id] = f"only {len(chosen)} sparse points agree"
        else:
            clusters[track.track_id] = chosen
            counts[track.track_id] = len(chosen)
    return clusters, LiftReport(len(views), counts, dropped)
