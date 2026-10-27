"""ArUco tag detection and multi-view triangulation of tag corners (P1.2).

Feeds ``world_frame.fit_board``: detect the board tags in each frame, then triangulate each
corner from every frame that saw it, using camera poses from the reconstruction back end. The
poses may be in any scale; the triangulated corners then live in that same arbitrary scale.

Corner order everywhere here matches ``world_frame.BoardLayout``: lower-left, lower-right,
upper-right, upper-left of the tag as printed. ArUco reports top-left, top-right, bottom-right,
bottom-left, so ``detect_tags`` reverses it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

F64 = NDArray[np.float64]

# Names used in configs/default.yaml (capture.marker.family) -> cv2.aruco constant names.
_FAMILIES = {"aruco_4x4_50": "DICT_4X4_50"}


def detect_tags(gray: NDArray[np.uint8], family: str = "aruco_4x4_50") -> dict[int, F64]:
    """Detect tags in one grayscale image. Returns id -> (4, 2) pixel corners (column, row).

    Corners are refined to sub-pixel, because the scale error scales with their error.
    """
    import cv2

    if family not in _FAMILIES:
        raise ValueError(f"unknown marker family {family!r}; known: {sorted(_FAMILIES)}")
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, _FAMILIES[family]))
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    corners, ids, _ = cv2.aruco.ArucoDetector(dictionary, params).detectMarkers(gray)
    found: dict[int, F64] = {}
    if ids is None:
        return found
    for tag_corners, tag_id in zip(corners, ids.ravel(), strict=True):
        found[int(tag_id)] = np.asarray(tag_corners, dtype=np.float64).reshape(4, 2)[::-1].copy()
    return found


@dataclass(frozen=True)
class TagView:
    """One frame's tag detections plus that frame's camera. Same convention as ``CameraFrame``."""

    K: F64  # (3, 3) intrinsics
    distortion: F64  # OpenCV order k1 k2 p1 p2 k3; may be empty
    T_world_from_camera: F64  # (4, 4), OpenCV camera axes
    corners_px: dict[int, F64]  # tag id -> (4, 2)


@dataclass(frozen=True)
class Triangulation:
    corners: dict[int, F64]  # tag id -> (4, 3), in the frame of the camera poses
    mean_reprojection_px: float
    max_reprojection_px: float
    n_views: dict[int, int]  # tag id -> number of frames used


def triangulate_tag_corners(
    views: list[TagView], *, min_views: int = 2, max_reprojection_px: float = 2.0
) -> Triangulation:
    """Triangulate every tag corner seen in at least ``min_views`` frames (linear DLT).

    Each corner drops its worst views until all remaining ones reproject within
    ``max_reprojection_px``. A tag whose corner cannot reach that with ``min_views`` views is
    left out of the scale. Raises if nothing survives, which the caller turns into a
    user-facing "marker not seen" message.
    """
    import cv2

    # Per view: undistorted normalized coordinates, and the camera_from_world projection.
    norm: list[dict[int, F64]] = []
    proj: list[F64] = []
    for v in views:
        t_cw = np.linalg.inv(v.T_world_from_camera)
        proj.append(np.asarray(t_cw[:3, :4], dtype=np.float64))
        per_tag: dict[int, F64] = {}
        for tag_id, px in v.corners_px.items():
            pts = cv2.undistortPoints(
                px.reshape(-1, 1, 2), v.K, v.distortion if v.distortion.size else None
            )
            per_tag[tag_id] = np.asarray(pts, dtype=np.float64).reshape(4, 2)
        norm.append(per_tag)

    tag_ids = sorted({i for per_tag in norm for i in per_tag})
    corners: dict[int, F64] = {}
    n_views: dict[int, int] = {}
    errors: list[float] = []
    for tag_id in tag_ids:
        seen = [k for k, per_tag in enumerate(norm) if tag_id in per_tag]
        if len(seen) < min_views:
            continue
        pts3 = np.zeros((4, 3))
        tag_err: list[float] = []
        used: set[int] = set()
        for j in range(4):
            point, err, kept = _robust_corner(
                [(proj[k], norm[k][tag_id][j], float(views[k].K[0, 0]), k) for k in seen],
                min_views,
                max_reprojection_px,
            )
            if point is None:
                break
            pts3[j] = point
            tag_err.extend(err)
            used.update(kept)
        else:
            corners[tag_id] = pts3
            n_views[tag_id] = len(used)
            errors.extend(tag_err)
    if not corners:
        raise ValueError("no tag could be triangulated: marker not seen in enough frames")
    return Triangulation(corners, float(np.mean(errors)), float(np.max(errors)), n_views)


def _dlt(obs: list[tuple[F64, F64, float, int]]) -> tuple[F64, list[float]]:
    """Linear triangulation of one point; returns it and its per-view reprojection error (px)."""
    rows = []
    for p, (x, y), _, _ in obs:
        rows.append(x * p[2] - p[0])
        rows.append(y * p[2] - p[1])
    h = np.linalg.svd(np.asarray(rows))[2][-1]
    point = h[:3] / h[3]
    err = []
    for p, (x, y), f, _ in obs:
        cam = p @ np.append(point, 1.0)
        err.append(float(np.hypot(cam[0] / cam[2] - x, cam[1] / cam[2] - y) * f))
    return point, err


# Above this many views one outlier barely moves the fit, so the worst view is the bad one.
# Below it, an outlier drags the point toward itself, so each view is tried in turn instead.
_LEAVE_ONE_OUT_MAX_VIEWS = 12


def _robust_corner(
    obs: list[tuple[F64, F64, float, int]], min_views: int, max_reprojection_px: float
) -> tuple[F64 | None, list[float], list[int]]:
    """DLT for one corner; drop the worst view until all fit or ``min_views`` remain.

    Detections in blurred frames, or in frames where the pose is poor, can be several pixels
    off. Dropping one tag for one bad view would throw away good data, so only the view goes.
    Returns ``(None, [], [])`` if the corner still does not fit with ``min_views`` left.
    """
    obs = list(obs)
    point, err = _dlt(obs)
    while max(err) > max_reprojection_px:
        if len(obs) <= min_views:
            return None, [], []
        if len(obs) <= _LEAVE_ONE_OUT_MAX_VIEWS:
            trials = [_dlt(obs[:i] + obs[i + 1 :]) for i in range(len(obs))]
            drop = int(np.argmin([max(e) for _, e in trials]))
            point, err = trials[drop]
        else:
            drop = int(np.argmax(err))
            point, err = _dlt(obs[:drop] + obs[drop + 1 :])
        del obs[drop]
    return point, err, [k for _, _, _, k in obs]
