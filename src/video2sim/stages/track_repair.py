"""Identity repair of SAM tracks with 3D consistency (P2.2).

SAM keeps an id only while an object stays in view, so a track can end and a new one start on
the same object (a fragmented object), and one id can jump to a neighbour (a switch).

Each mask gets a footprint: the point where the ray through its lowest pixels meets the table
plane, in world XY. Objects on the table do not move, so the footprints of one object stay
together over all views, whatever the camera does.

* Split: a track whose footprints fall into two or more groups far apart covered two objects.
  Feet from occlusion slivers (masks far below the track's median size, ``sliver_frac``)
  are left out of the grouping: a nearly hidden object must not found a new group. Those
  frames rejoin the temporally nearest group afterwards.
* Merge: two tracks that never appear in the same frame and share a footprint are one object.
  A shared frame means two objects were visible at once, so such tracks are never merged;
  this also keeps a stack apart, because the upper object's base projects to a different spot.
  Median footprints over different view arcs are not comparable, so a handoff also merges:
  the closest frames of the two tracks are adjacent and their masks overlap there, which is
  what an id that switched away and back looks like.

Footprints are the near edge of an object, not its center, so they wobble by up to an object
radius between views. The thresholds in ``configs/default.yaml`` are guesses to tune on dev.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from video2sim.schemas import SE3, Intrinsics, Plane
from video2sim.stages.segmentation import BoolMask
from video2sim.stages.tracking import Track

F64 = NDArray[np.float64]

# Share of the mask height, counted up from its lowest row, that stands for the contact edge.
BASE_FRACTION = 0.1


@dataclass(frozen=True)
class RepairConfig:
    """Values from ``segmentation.tracking.repair`` in ``configs/default.yaml``."""

    merge_radius_m: float
    split_gap_m: float
    min_frames: int
    sliver_frac: float
    merge_boundary_gap: int
    merge_boundary_iou: float

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> RepairConfig:
        r = cfg["segmentation"]["tracking"]["repair"]
        out = cls(
            float(r["merge_radius_m"]),
            float(r["split_gap_m"]),
            int(r["min_frames"]),
            float(r["sliver_frac"]),
            int(r["merge_boundary_gap"]),
            float(r["merge_boundary_iou"]),
        )
        if out.merge_radius_m <= 0 or out.split_gap_m <= 0 or out.min_frames < 1:
            raise ValueError("repair thresholds must be positive")
        if not 0.0 <= out.sliver_frac < 1.0:
            raise ValueError("sliver_frac must be in [0, 1)")
        if out.merge_boundary_gap < 1:
            raise ValueError("merge_boundary_gap must be at least 1")
        if not 0.0 < out.merge_boundary_iou < 1.0:
            raise ValueError("merge_boundary_iou must be in (0, 1)")
        return out


@dataclass(frozen=True)
class RepairReport:
    splits: tuple[tuple[int, tuple[int, ...]], ...]  # original id -> ids it became
    merges: tuple[tuple[int, ...], ...]  # original ids that became one track


def mask_footprint(
    mask: BoolMask, T_world_from_camera: SE3, intrinsics: Intrinsics, plane: Plane
) -> F64 | None:
    """World XY where the ray through the mask's lowest pixels meets the table, or None."""
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return None
    bottom = int(np.max(rows))
    near = rows >= bottom - BASE_FRACTION * (bottom - int(np.min(rows)) + 1)
    u, v = float(cols[near].mean()), float(bottom)
    pose = T_world_from_camera.to_numpy()
    ray = pose[:3, :3] @ np.array(
        [(u - intrinsics.cx) / intrinsics.fx, (v - intrinsics.cy) / intrinsics.fy, 1.0]
    )
    n = np.asarray(plane.normal, dtype=np.float64)
    denom = float(n @ ray)
    if abs(denom) < 1e-9:
        return None
    s = (plane.offset_m - float(n @ pose[:3, 3])) / denom
    if s <= 0:
        return None
    return np.asarray((pose[:3, 3] + s * ray)[:2], dtype=np.float64)


def _footprints(
    track: Track,
    poses: Mapping[int, SE3],
    intrinsics: Intrinsics,
    plane: Plane,
) -> dict[int, F64]:
    out: dict[int, F64] = {}
    for frame, mask in track.masks.items():
        point = mask_footprint(mask, poses[frame], intrinsics, plane)
        if point is not None:
            out[frame] = point
    return out


def _groups(points: NDArray[np.float64], link_m: float) -> list[list[int]]:
    """Connected groups of points where any two within ``link_m`` are linked.

    KD-tree pairs plus a sparse connected-components pass: the pairwise
    distance matrix this replaces needs about 32 bytes per point squared,
    which is gigabytes on a room-sized cloud.
    """
    from scipy.sparse import coo_matrix  # type: ignore[import-untyped]  # heavy import
    from scipy.sparse.csgraph import connected_components  # type: ignore[import-untyped]
    from scipy.spatial import cKDTree  # type: ignore[import-untyped]  # heavy import

    n = len(points)
    if n == 0:
        return []
    if n == 1 or link_m < 0:
        return [[i] for i in range(n)]
    pairs = cKDTree(np.asarray(points, dtype=np.float64)).query_pairs(link_m, output_type="ndarray")
    if not len(pairs):
        return [[i] for i in range(n)]
    rows = np.concatenate((pairs[:, 0], pairs[:, 1]))
    cols = np.concatenate((pairs[:, 1], pairs[:, 0]))
    graph = coo_matrix((np.ones(len(rows), dtype=np.int32), (rows, cols)), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(int(labels[i]), []).append(i)
    return sorted(groups.values(), key=lambda g: g[0])


def _split(track: Track, feet: Mapping[int, F64], cfg: RepairConfig, next_id: int) -> list[Track]:
    """The track itself, or one track per far-apart group of footprints."""
    if cfg.sliver_frac > 0.0 and track.masks:
        typical = float(np.median([m.sum() for m in track.masks.values()]))
        feet = {
            f: p for f, p in feet.items() if int(track.masks[f].sum()) >= cfg.sliver_frac * typical
        }
    frames = sorted(feet)
    if len(frames) < 2 * cfg.min_frames:
        return [track]
    groups = _groups(np.array([feet[f] for f in frames]), cfg.split_gap_m)
    big = [g for g in groups if len(g) >= cfg.min_frames]
    if len(big) < 2:
        return [track]
    centers = [np.median([feet[frames[i]] for i in g], axis=0) for g in big]
    members: list[list[int]] = [[frames[i] for i in g] for g in big]
    for g in groups:
        if len(g) >= cfg.min_frames:
            continue
        for i in g:  # too few frames to be an object: join the nearest group, never drop
            nearest = int(np.argmin([np.linalg.norm(feet[frames[i]] - c) for c in centers]))
            members[nearest].append(frames[i])
    # Frames without a footprint stay with the group that is nearest in time.
    for frame in sorted(set(track.masks) - set(frames)):
        nearest = int(np.argmin([min(abs(frame - m) for m in group) for group in members]))
        members[nearest].append(frame)
    pieces = []
    for k, group in enumerate(members):
        piece_id = track.track_id if k == 0 else next_id + k - 1
        pieces.append(Track(piece_id, {f: track.masks[f] for f in sorted(group)}))
    return pieces


def _mask_iou(a: BoolMask, b: BoolMask) -> float | None:
    """Mask overlap, or None when the masks are not comparable."""
    if a.shape != b.shape:
        return None
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return None
    return float(np.logical_and(a, b).sum() / union)


def _handover(a: Sequence[Track], b: Sequence[Track], cfg: RepairConfig) -> bool:
    """Whether two frame-disjoint track groups meet at a handoff.

    Their closest frames are within ``merge_boundary_gap`` and the masks there overlap
    past ``merge_boundary_iou``: the new id's first mask repeats the old id's last mask
    in an adjacent frame. Only the closest frame pair is scored, so the cost stays flat.
    """
    fa = sorted({f for t in a for f in t.masks})
    fb = sorted({f for t in b for f in t.masks})
    i = 0
    j = 0
    gap: int | None = None
    while i < len(fa) and j < len(fb):
        close = abs(fa[i] - fb[j])
        if gap is None or close < gap:
            gap = close
        if fa[i] < fb[j]:
            i += 1
        else:
            j += 1
    if gap is None or gap > cfg.merge_boundary_gap:
        return False
    best = 0.0
    for ta in a:
        for frame_a, mask_a in ta.masks.items():
            for tb in b:
                for frame_b, mask_b in tb.masks.items():
                    if abs(frame_a - frame_b) == gap:
                        iou = _mask_iou(mask_a, mask_b)
                        if iou is not None and iou > best:
                            best = iou
    return best >= cfg.merge_boundary_iou


def _merge(
    tracks: Sequence[Track], feet: Mapping[int, Mapping[int, F64]], cfg: RepairConfig
) -> tuple[list[Track], list[tuple[int, ...]]]:
    """Greedily merge frame-disjoint tracks that share a footprint or a handoff."""
    centers = {
        t.track_id: np.median(list(feet[t.track_id].values()), axis=0)
        for t in tracks
        if feet[t.track_id]
    }
    groups: dict[int, list[Track]] = {t.track_id: [t] for t in tracks}
    pairs = sorted(
        (float(np.linalg.norm(centers[a] - centers[b])), a, b)
        for a in centers
        for b in centers
        if a < b
    )
    owner = {t.track_id: t.track_id for t in tracks}
    for distance, a, b in pairs:
        ga, gb = owner[a], owner[b]
        if ga == gb:
            continue
        frames_a = {f for t in groups[ga] for f in t.masks}
        frames_b = {f for t in groups[gb] for f in t.masks}
        if frames_a & frames_b:
            continue
        if distance > cfg.merge_radius_m and not _handover(groups[ga], groups[gb], cfg):
            continue
        keep, drop = min(ga, gb), max(ga, gb)
        groups[keep].extend(groups.pop(drop))
        for k, v in owner.items():
            if v == drop:
                owner[k] = keep
    merged: list[Track] = []
    merges: list[tuple[int, ...]] = []
    for keep in sorted(groups):
        parts = groups[keep]
        masks = {f: m for t in parts for f, m in t.masks.items()}
        merged.append(Track(keep, dict(sorted(masks.items()))))
        if len(parts) > 1:
            merges.append(tuple(sorted(t.track_id for t in parts)))
    return merged, merges


def repair_tracks(
    tracks: Sequence[Track],
    poses: Mapping[int, SE3],
    intrinsics: Intrinsics,
    plane: Plane,
    cfg: RepairConfig,
) -> tuple[list[Track], RepairReport]:
    """Split tracks that cover two objects, then merge fragments of one object."""
    next_id = max((t.track_id for t in tracks), default=-1) + 1
    pieces: list[Track] = []
    splits: list[tuple[int, tuple[int, ...]]] = []
    for track in tracks:
        parts = _split(track, _footprints(track, poses, intrinsics, plane), cfg, next_id)
        next_id += len(parts) - 1
        if len(parts) > 1:
            splits.append((track.track_id, tuple(p.track_id for p in parts)))
        pieces.extend(parts)
    feet = {t.track_id: _footprints(t, poses, intrinsics, plane) for t in pieces}
    merged, merges = _merge(pieces, feet, cfg)
    return merged, RepairReport(tuple(splits), tuple(merges))
