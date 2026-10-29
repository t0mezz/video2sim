"""Keyframe selection per object (P2.4).

The shape stage predicts a mesh from a few crops of each object, so the crops should show it
well and from different sides. For every tracked object this picks the best frames in two parts.

Quality of one frame is a product of four scores in [0, 1], each raised to a weight:

* ``area``: mask area against the 90th percentile over the object's frames. A frame where the
  object is small or partly hidden scores low.
* ``border``: one minus the share of mask pixels in a band at the image edge. A cut-off object
  gives the shape model a truncated view.
* ``occlusion``: one minus the share of a thin ring around the mask that lies inside another
  track's mask. The neighbour in front of the object is what hides it.
* ``sharpness``: variance of the Laplacian inside the mask against the 90th percentile.

Selection is greedy. The best frame goes first. Every later frame is scored as its quality
times ``1 - diversity_weight * similarity``, where similarity is the largest Gaussian
(width ``diversity_sigma_deg``) of the angle between its viewing direction and that of a frame
already chosen. The viewing direction is the unit vector from the object's centre to the camera
centre, so two cameras on the same side of the object count as the same view whatever their
roll. At 3 to 5 fps neighbouring frames are almost identical and would otherwise fill all the
slots.

The penalty only reorders. A frame is eligible when its own quality reaches ``min_quality``, so
a camera path over a narrow arc still fills the slots, widest-spread frames first. Frames closer
than ``min_separation_deg`` to a chosen one are never chosen, because they add nothing. An
object with too few good or distinct frames gets fewer than ``count`` keyframes rather than poor
or duplicate ones.

Occlusion is judged from the other tracks' masks only. Something in front of the object that is
not tracked (a hand, an untracked item) is not seen. The angle uses the object's centre, which
the caller passes in; the point cluster centre is the intended source.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from video2sim.schemas import WorldModel
from video2sim.stages.tracking import Track

F64 = NDArray[np.float64]
BoolMask = NDArray[np.bool_]


@dataclass(frozen=True)
class KeyframeConfig:
    """Values from ``segmentation.keyframes_per_object`` and ``segmentation.keyframes``."""

    count: int
    min_mask_px: int
    border_band_frac: float
    occluder_ring_px: int
    weight_area: float
    weight_border: float
    weight_occlusion: float
    weight_sharpness: float
    diversity_weight: float
    diversity_sigma_deg: float
    min_quality: float
    min_separation_deg: float

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> KeyframeConfig:
        seg = cfg["segmentation"]
        k = seg["keyframes"]
        out = cls(
            int(seg["keyframes_per_object"]),
            int(k["min_mask_px"]),
            float(k["border_band_frac"]),
            int(k["occluder_ring_px"]),
            float(k["weight_area"]),
            float(k["weight_border"]),
            float(k["weight_occlusion"]),
            float(k["weight_sharpness"]),
            float(k["diversity_weight"]),
            float(k["diversity_sigma_deg"]),
            float(k["min_quality"]),
            float(k["min_separation_deg"]),
        )
        if min(out.count, out.min_mask_px, out.occluder_ring_px) < 1:
            raise ValueError("keyframe counts must be at least 1")
        if not 0.0 <= out.border_band_frac < 0.5:
            raise ValueError("border_band_frac must be in [0, 0.5)")
        if min(out.weight_area, out.weight_border, out.weight_occlusion, out.weight_sharpness) < 0:
            raise ValueError("keyframe weights must not be negative")
        if not 0.0 <= out.diversity_weight <= 1.0:
            raise ValueError("diversity_weight must be in [0, 1]")
        if out.min_separation_deg < 0:
            raise ValueError("min_separation_deg must not be negative")
        if out.diversity_sigma_deg <= 0:
            raise ValueError("diversity_sigma_deg must be positive")
        return out


@dataclass(frozen=True)
class FrameStats:
    """Raw measurements of one track in one frame, before they are scored."""

    frame_index: int
    area_px: int
    border_frac: float
    occlusion_frac: float
    sharpness: float


@dataclass(frozen=True)
class KeyframeChoice:
    """One chosen frame. ``quality`` ignores the other choices; ``score`` includes the penalty."""

    frame_index: int
    quality: float
    score: float
    similarity: float  # to the closest frame chosen before it; 0 for the first
    area: float
    border: float
    occlusion: float
    sharpness: float


@dataclass(frozen=True)
class KeyframeReport:
    """Tracks that got fewer keyframes than asked, with the reason. Nothing is dropped quietly."""

    notes: Mapping[int, str]


def frame_stats(
    frame_index: int,
    mask: BoolMask,
    others: BoolMask | None,
    gray: NDArray[np.uint8] | None,
    cfg: KeyframeConfig,
) -> FrameStats | None:
    """Measure one mask. ``None`` when the mask is smaller than ``min_mask_px``."""
    import cv2

    area = int(mask.sum())
    if area < cfg.min_mask_px:
        return None
    height, width = mask.shape
    band = max(1, round(cfg.border_band_frac * min(height, width)))
    edge = np.zeros_like(mask)
    edge[:band], edge[-band:], edge[:, :band], edge[:, -band:] = True, True, True, True
    border = float((mask & edge).sum()) / area

    rows, cols = np.any(mask, axis=1), np.any(mask, axis=0)
    y_lo, y_hi = int(np.argmax(rows)), int(height - 1 - np.argmax(rows[::-1]))
    x_lo, x_hi = int(np.argmax(cols)), int(width - 1 - np.argmax(cols[::-1]))
    r = cfg.occluder_ring_px
    y0, y1 = max(0, y_lo - r - 1), min(height, y_hi + r + 2)
    x0, x1 = max(0, x_lo - r - 1), min(width, x_hi + r + 2)
    crop = mask[y0:y1, x0:x1].astype(np.uint8)

    occlusion = 0.0
    if others is not None:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        ring = (cv2.dilate(crop, kernel) > 0) & (crop == 0)
        ring &= ~edge[y0:y1, x0:x1]  # outside the image there is nothing to occlude with
        ring_px = int(ring.sum())
        if ring_px:
            occlusion = float((ring & others[y0:y1, x0:x1]).sum()) / ring_px

    sharpness = 0.0
    if gray is not None:
        inner = cv2.erode(crop, np.ones((3, 3), np.uint8), iterations=2) > 0
        inner = inner if inner.any() else crop > 0
        lap = cv2.Laplacian(gray[y0:y1, x0:x1], cv2.CV_64F)
        sharpness = float(lap[inner].var())
    return FrameStats(frame_index, area, border, occlusion, sharpness)


def _relative(values: F64) -> F64:
    """Values over their 90th percentile, capped at 1; all 1 when there is no spread to use."""
    ref = float(np.percentile(values, 90))
    if ref <= 0:
        return np.ones_like(values)
    return np.minimum(1.0, values / ref)


def select_keyframes(
    stats: Sequence[FrameStats],
    directions: Mapping[int, F64],
    cfg: KeyframeConfig,
    use_sharpness: bool = True,
) -> list[KeyframeChoice]:
    """Greedy pick of up to ``cfg.count`` frames, best first, with the view-diversity penalty.

    ``directions`` maps frame index to the unit vector from the object to the camera. Frames
    without a direction are not eligible. With ``use_sharpness`` False the sharpness score is 1
    (no images were available).
    """
    rows = [s for s in stats if s.frame_index in directions]
    if not rows:
        return []
    area = _relative(np.array([s.area_px for s in rows], dtype=np.float64))
    sharp = (
        _relative(np.array([s.sharpness for s in rows], dtype=np.float64))
        if use_sharpness
        else np.ones(len(rows))
    )
    border = np.array([1.0 - s.border_frac for s in rows])
    occ = np.array([1.0 - s.occlusion_frac for s in rows])
    quality = (
        area**cfg.weight_area
        * border**cfg.weight_border
        * occ**cfg.weight_occlusion
        * sharp**cfg.weight_sharpness
    )
    dirs = np.stack([directions[s.frame_index] for s in rows])

    chosen: list[int] = []  # positions in rows
    out: list[KeyframeChoice] = []
    sigma = np.radians(cfg.diversity_sigma_deg)
    floor = np.radians(cfg.min_separation_deg)
    eligible = quality >= cfg.min_quality
    while len(chosen) < cfg.count:
        if chosen:
            angle = np.arccos(np.clip(dirs @ dirs[chosen].T, -1.0, 1.0))
            similarity = np.exp(-0.5 * (angle / sigma) ** 2).max(axis=1)
            allowed = eligible & (angle.min(axis=1) >= floor)
            allowed[chosen] = False
        else:
            similarity = np.zeros(len(rows))
            allowed = eligible
        score = np.where(allowed, quality * (1.0 - cfg.diversity_weight * similarity), -1.0)
        best = int(np.argmax(score))
        if score[best] < 0:
            break
        chosen.append(best)
        out.append(
            KeyframeChoice(
                rows[best].frame_index,
                float(quality[best]),
                float(score[best]),
                float(similarity[best]),
                float(area[best]),
                float(border[best]),
                float(occ[best]),
                float(sharp[best]),
            )
        )
    return out


def view_directions(world: WorldModel, center: F64, frame_indices: Sequence[int]) -> dict[int, F64]:
    """Unit vectors from ``center`` to each camera centre, for the given frame indices."""
    wanted = set(frame_indices)
    out: dict[int, F64] = {}
    for f in world.frames:
        if f.frame_index not in wanted:
            continue
        v = f.T_world_from_camera.to_numpy()[:3, 3] - center
        norm = float(np.linalg.norm(v))
        if norm > 1e-9:
            out[f.frame_index] = v / norm
    return out


def keyframes_for_tracks(
    tracks: Sequence[Track],
    world: WorldModel,
    run_dir: Path,
    centers: Mapping[int, F64],
    cfg: KeyframeConfig,
) -> tuple[dict[int, list[KeyframeChoice]], KeyframeReport]:
    """Keyframes by track id for every track that has a centre in ``centers``.

    ``run_dir`` is what ``CameraFrame.image_path`` is relative to. Frame images are read once
    for sharpness; if the first one cannot be read, sharpness is left out for all.
    """
    import cv2

    wanted = [t for t in tracks if t.track_id in centers]
    notes: dict[int, str] = {
        t.track_id: "no point cluster, so no viewing direction" for t in tracks if t not in wanted
    }
    paths = {f.frame_index: run_dir / f.image_path for f in world.frames}
    use_sharpness = bool(paths) and cv2.imread(str(next(iter(paths.values())))) is not None

    stats: dict[int, list[FrameStats]] = {t.track_id: [] for t in wanted}
    frames = sorted({f for t in wanted for f in t.masks})
    for frame in frames:
        gray = None
        if use_sharpness and frame in paths:
            image = cv2.imread(str(paths[frame]), cv2.IMREAD_GRAYSCALE)
            gray = None if image is None else np.asarray(image, dtype=np.uint8)
        masks = {t.track_id: t.masks[frame] for t in tracks if frame in t.masks}
        for t in wanted:
            mask = masks.get(t.track_id)
            if mask is None:
                continue
            rest = [m for i, m in masks.items() if i != t.track_id]
            others = np.any(np.stack(rest), axis=0) if rest else None
            s = frame_stats(frame, mask, others, gray, cfg)
            if s is not None:
                stats[t.track_id].append(s)

    result: dict[int, list[KeyframeChoice]] = {}
    for t in wanted:
        dirs = view_directions(
            world, centers[t.track_id], [s.frame_index for s in stats[t.track_id]]
        )
        picked = select_keyframes(stats[t.track_id], dirs, cfg, use_sharpness)
        result[t.track_id] = picked
        if not picked:
            notes[t.track_id] = "no frame with a usable mask"
        elif len(picked) < cfg.count:
            notes[t.track_id] = (
                f"{len(picked)} of {cfg.count} keyframes: the rest are weak or repeat a view"
            )
    return result, KeyframeReport(notes)
