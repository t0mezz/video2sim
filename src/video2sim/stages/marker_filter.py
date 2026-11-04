"""Drop tracks that are the ArUco tags themselves (or swallowed them), before identity repair.

When OpenCV decodes a tag in a frame, nothing covers it there: an object that partly hides the
tag's black border or bits makes the detection fail. So a mask that overlaps a detected tag in
such a frame is not an object standing on or in front of the tag. It is the tag, or a region
(paper, board, table) that absorbed it. A track with enough of those frames is dropped whole.

Tags that are never detected in a frame cannot be checked there; the filter says nothing about
them. Detection runs on the registered frames again, which is cheap next to the hosted calls.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from video2sim.schemas import WorldModel
from video2sim.stages.marker_detection import TAG_FILE, detect_file, read_tag_file
from video2sim.stages.segmentation import BoolMask, MaskLike, as_crop
from video2sim.stages.tracking import Track

F64 = NDArray[np.float64]
Detections = tuple[dict[int, F64], tuple[int, int]]  # tag id -> (4, 2) pixel corners; (rows, cols)
TagDetector = Callable[[Path], Detections]


@dataclass(frozen=True)
class MarkerFilterConfig:
    """Values from ``segmentation.tracking.marker_filter`` in ``configs/default.yaml``."""

    enabled: bool
    family: str
    erode_px: int
    min_mask_share: float
    min_tag_share: float
    min_hit_frames: int
    min_hit_fraction: float

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> MarkerFilterConfig:
        m = cfg["segmentation"]["tracking"]["marker_filter"]
        out = cls(
            bool(m["enabled"]),
            str(cfg["capture"]["marker"]["family"]),
            int(m["erode_px"]),
            float(m["min_mask_share"]),
            float(m["min_tag_share"]),
            int(m["min_hit_frames"]),
            float(m["min_hit_fraction"]),
        )
        shares = (out.min_mask_share, out.min_tag_share, out.min_hit_fraction)
        if out.erode_px < 0 or out.min_hit_frames < 1 or not all(0 < s <= 1 for s in shares):
            raise ValueError(f"bad marker_filter config: {m}")
        return out


@dataclass(frozen=True)
class MarkerFilterReport:
    dropped: Mapping[int, str]  # track id -> reason


def cached_tag_detector(family: str, detect: TagDetector | None = None) -> TagDetector:
    """A tag detector that reads each frame file at most once.

    One stage run checks the same frames in the marker filter, in every
    gap-fill request's marker filter and in the board check; the frames do
    not change during a run, so callers share one of these instead of
    re-detecting (and re-decoding) every frame for every check.
    """
    base = detect or (lambda p: _detect_file(p, family))
    cache: dict[Path, Detections] = {}
    lock = threading.Lock()

    def detect_once(path: Path) -> Detections:
        with lock:
            if path not in cache:
                cache[path] = base(path)
            return cache[path]

    return detect_once


def tag_mask(corners: Sequence[F64], shape: tuple[int, int], erode_px: int) -> BoolMask:
    """Filled tag quads on a ``shape`` canvas, each shrunk by ``erode_px`` pixels."""
    import cv2

    canvas: NDArray[np.uint8] = np.zeros(shape, dtype=np.uint8)
    for quad in corners:
        one: Any = np.zeros(shape, dtype=np.uint8)
        cv2.fillPoly(one, [np.round(quad).astype(np.int32)], 1)
        if erode_px > 0:
            one = cv2.erode(one, np.ones((2 * erode_px + 1, 2 * erode_px + 1), np.uint8))
        canvas |= one
    return np.asarray(canvas, dtype=bool)


def overlaps_tag(mask: MaskLike, tags: BoolMask, cfg: MarkerFilterConfig) -> bool:
    """True when the mask covers enough of its own area, or of the tags' area, in the tags."""
    crop = as_crop(mask)
    h, w = crop.data.shape
    under = tags[crop.top : crop.top + h, crop.left : crop.left + w]
    overlap = int(np.count_nonzero(crop.data & under))
    if overlap == 0:
        return False
    return bool(
        overlap >= cfg.min_mask_share * crop.area
        or overlap >= cfg.min_tag_share * np.count_nonzero(tags)
    )


def filter_marker_tracks(
    tracks: Sequence[Track],
    world: WorldModel,
    run_dir: Path,
    cfg: MarkerFilterConfig,
    detect: TagDetector | None = None,
) -> tuple[list[Track], MarkerFilterReport]:
    """The tracks that are not markers, and why the others were dropped.

    ``detect`` defaults to OpenCV; tests replace it. Corners are scaled to the
    mask resolution when the masks and the image differ in size.
    """
    if not cfg.enabled or not tracks:
        return list(tracks), MarkerFilterReport({})
    detect = detect or (lambda p: _detect_file(p, cfg.family))

    detected: dict[int, int] = {t.track_id: 0 for t in tracks}  # frames with a tag and a mask
    hits: dict[int, int] = {t.track_id: 0 for t in tracks}
    for frame in world.frames:
        users = [t for t in tracks if frame.frame_index in t.masks]
        if not users:
            continue
        path = run_dir / frame.image_path
        corners, (height, width) = detect(path)
        if not corners:
            continue
        by_shape: dict[tuple[int, int], BoolMask] = {}  # the tag mask is the same for every track
        for track in users:
            mask = track.masks[frame.frame_index]
            shape = (int(mask.shape[0]), int(mask.shape[1]))
            if shape not in by_shape:
                sy, sx = shape[0] / height, shape[1] / width
                quads = [c * np.array([sx, sy]) for c in corners.values()]
                by_shape[shape] = tag_mask(quads, shape, cfg.erode_px)
            tags = by_shape[shape]
            detected[track.track_id] += 1
            hits[track.track_id] += int(overlaps_tag(mask, tags, cfg))

    kept: list[Track] = []
    dropped: dict[int, str] = {}
    for track in tracks:
        n, h = detected[track.track_id], hits[track.track_id]
        if h >= cfg.min_hit_frames and h >= cfg.min_hit_fraction * n:
            dropped[track.track_id] = (
                f"overlaps a detected marker tag in {h} of {n} frames where one was visible"
            )
        else:
            kept.append(track)
    return kept, MarkerFilterReport(dropped)


def _detect_file(path: Path, family: str) -> Detections:
    return detect_file(path, family)


def handed_tag_detector(run_dir: Path, family: str) -> TagDetector | None:
    """A detector that answers from the reconstruction stage's ``tags.json`` (D-039), or None
    when that file is missing or for another family. A frame the file does not list is
    detected from its image, as without the file."""
    table = read_tag_file(run_dir / "reconstruction" / TAG_FILE, family)
    if table is None:
        return None

    def detect(path: Path) -> Detections:
        # Callers pass ``run_dir / frame.image_path``; the key is that image path.
        key = path.relative_to(run_dir).as_posix() if path.is_relative_to(run_dir) else ""
        found = table.get(key)
        return found if found is not None else _detect_file(path, family)

    return detect
