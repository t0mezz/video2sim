"""Geometric failure flags (P2.5): small, thin and heavily occluded objects.

These are measured, not asked of a model. Size and thinness come from the visual hull of P2.3
(``segmentation/clusters``): the extents along the cluster's principal axes, each widened by one
voxel because the hull is a little larger than the object. ``Flag.SMALL`` means the largest extent
is at most ``small_max_extent_m``; ``Flag.THIN`` means the smallest extent is at most
``thin_max_extent_m`` and at most ``thin_max_ratio`` of the largest. Occlusion is the share of an
object's frames where more than ``occluded_frame_frac`` of the ring around its mask lies on other
tracks' masks (the same measure as keyframe selection); ``Flag.OCCLUDED`` means that share is at
least ``occluded_share``. Hands and untracked items in front of an object are not seen, as in P2.4.

A track without a cluster gets no size or thinness verdict and a reason saying so, so a missing
measurement shows in the QA report. Thresholds are in ``segmentation.shape_flags``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from video2sim.schemas import Flag
from video2sim.stages.keyframes import KeyframeConfig, frame_stats
from video2sim.stages.tracking import Track

F64 = NDArray[np.float64]


@dataclass(frozen=True)
class ShapeConfig:
    """Values from ``segmentation.shape_flags`` in ``configs/default.yaml``."""

    voxel_m: float
    small_max_extent_m: float
    thin_max_extent_m: float
    thin_max_ratio: float
    occluded_frame_frac: float
    occluded_share: float
    min_frames: int
    keyframes: KeyframeConfig

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> ShapeConfig:
        c = cfg["segmentation"]["shape_flags"]
        out = cls(
            float(cfg["segmentation"]["lift"]["voxel_m"]),
            float(c["small_max_extent_m"]),
            float(c["thin_max_extent_m"]),
            float(c["thin_max_ratio"]),
            float(c["occluded_frame_frac"]),
            float(c["occluded_share"]),
            int(c["min_frames"]),
            KeyframeConfig.from_config(cfg),
        )
        if min(out.small_max_extent_m, out.thin_max_extent_m) <= 0 or out.voxel_m <= 0:
            raise ValueError("extent thresholds and voxel_m must be positive")
        if not 0 < out.thin_max_ratio <= 1:
            raise ValueError("thin_max_ratio must be in (0, 1]")
        if not 0 <= out.occluded_frame_frac <= 1 or not 0 <= out.occluded_share <= 1:
            raise ValueError("occlusion fractions must be in [0, 1]")
        if out.min_frames < 1:
            raise ValueError("min_frames must be at least 1")
        return out


@dataclass(frozen=True)
class ShapeSignals:
    track_id: int
    extents_m: tuple[float, float, float] | None  # sorted smallest to largest; None: no cluster
    frames: int
    occluded_share: float | None  # None: fewer than ``min_frames`` usable frames

    def as_dict(self) -> dict[str, Any]:
        return {
            "track": self.track_id,
            "extents_m": None if self.extents_m is None else [round(e, 4) for e in self.extents_m],
            "frames": self.frames,
            "occluded_share": None
            if self.occluded_share is None
            else round(self.occluded_share, 3),
        }


@dataclass(frozen=True)
class ShapeFlags:
    track_id: int
    flags: tuple[Flag, ...]
    reasons: tuple[str, ...]  # one per flag, plus notes about missing measurements


def principal_extents(points: F64, voxel_m: float) -> tuple[float, float, float]:
    """Extents along the principal axes, smallest first, each widened by one voxel."""
    centred = points - points.mean(axis=0)
    _, _, axes = np.linalg.svd(centred, full_matrices=False)
    spans = np.ptp(centred @ axes.T, axis=0) + voxel_m
    a, b, c = sorted(float(s) for s in spans)
    return a, b, c


def shape_signals(
    tracks: Sequence[Track],
    hulls: Mapping[int, F64],
    cfg: ShapeConfig,
) -> list[ShapeSignals]:
    """Size, thinness and occlusion numbers for every track."""
    frames = sorted({f for t in tracks for f in t.masks})
    seen: dict[int, list[float]] = {t.track_id: [] for t in tracks}
    for frame in frames:
        masks = {t.track_id: t.masks[frame] for t in tracks if frame in t.masks}
        for track_id, mask in masks.items():
            rest = [m for i, m in masks.items() if i != track_id]
            others = np.any(np.stack(rest), axis=0) if rest else None
            s = frame_stats(frame, mask, others, None, cfg.keyframes)
            if s is not None:
                seen[track_id].append(s.occlusion_frac)
    out = []
    for t in tracks:
        hull = hulls.get(t.track_id)
        extents = principal_extents(hull, cfg.voxel_m) if hull is not None and len(hull) else None
        fracs = seen[t.track_id]
        share = (
            float(np.mean([f > cfg.occluded_frame_frac for f in fracs]))
            if len(fracs) >= cfg.min_frames
            else None
        )
        out.append(ShapeSignals(t.track_id, extents, len(fracs), share))
    return out


def load_hulls(directory: Path) -> dict[int, F64]:
    """Hull clusters written as ``track_NNN.npy``, by track id."""
    return {
        int(path.stem.split("_")[1]): np.load(path).astype(np.float64)
        for path in sorted(directory.glob("track_*.npy"))
    }


def judge_shape(sig: ShapeSignals, cfg: ShapeConfig) -> ShapeFlags:
    """Flags and reasons for one track from its signals."""
    flags: list[Flag] = []
    reasons: list[str] = []
    if sig.extents_m is None:
        reasons.append("no point cluster: size and thinness not measured")
    else:
        smallest, _, largest = sig.extents_m
        if largest <= cfg.small_max_extent_m:
            flags.append(Flag.SMALL)
            reasons.append(f"largest extent {largest * 1000:.0f} mm")
        if smallest <= cfg.thin_max_extent_m and smallest / largest <= cfg.thin_max_ratio:
            flags.append(Flag.THIN)
            reasons.append(
                f"smallest extent {smallest * 1000:.0f} mm, {smallest / largest:.2f} of the largest"
            )
    if sig.occluded_share is None:
        reasons.append(f"only {sig.frames} usable frames: occlusion not measured")
    elif sig.occluded_share >= cfg.occluded_share:
        flags.append(Flag.OCCLUDED)
        reasons.append(
            f"{sig.occluded_share:.0%} of frames have over {cfg.occluded_frame_frac:.0%} "
            "of the mask outline on other objects"
        )
    return ShapeFlags(sig.track_id, tuple(flags), tuple(reasons))


def judge_all(signals: Sequence[ShapeSignals], cfg: ShapeConfig) -> Mapping[int, ShapeFlags]:
    return {s.track_id: judge_shape(s, cfg) for s in signals}
