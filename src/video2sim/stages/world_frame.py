"""Metric scale and world frame from marker tags (P1.2, P1.3).

Pure geometry on NumPy arrays; it does not detect tags or run a reconstruction. The caller
supplies the 3D corners of the detected tags in the reconstruction frame, which has an arbitrary
scale and orientation, plus the table points.

Production use is one or two tags placed anywhere, so only the printed tag edge is known:
``estimate_scale`` reads scale from the tag edges and ``world_from_tag`` puts the world origin
on a reference tag. The benchmark grid board additionally has known tag positions;
``fit_board`` uses them as an independent check and is not part of the pipeline.

Frames: ``recon`` is the reconstruction frame, ``board`` is the printed grid board (X along the
long edge, Y up the short edge, Z out of the sheet, origin at the grid corner), ``world`` is +Z
up with the origin on the reference tag or board. Scale is returned as a number and applied to
geometry once, by ``apply_scale``; no transform returned here contains it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from video2sim.schemas import Plane

F64 = NDArray[np.float64]

# Corner order of one tag in its own frame, counter-clockwise seen from +Z: lower-left,
# lower-right, upper-right, upper-left. Matches the ArUco corner order seen on the board.
_CORNER_SIGNS = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)


@dataclass(frozen=True)
class BoardLayout:
    """Tag corners in the board frame, in meters. Shape of each entry: (4, 3), z = 0."""

    tag_corners_m: dict[int, F64]

    @classmethod
    def from_make_board_json(cls, path: Path, tag_edge_m: float | None = None) -> BoardLayout:
        """Read the layout written by ``benchmark/make_board.py``.

        ``tag_edge_m`` should be the caliper-measured edge (``capture.marker.tag_size_m``). The
        layout file only holds the nominal value and says to use measured numbers.
        """
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        edge_mm = (
            tag_edge_m * 1000.0 if tag_edge_m is not None else float(data["tag_edge_mm_nominal"])
        )
        centers = {
            int(tag_id): (float(entry["center_mm"][0]), float(entry["center_mm"][1]))
            for tag_id, entry in data["tags"].items()
        }
        return cls.from_tag_centers(centers, edge_mm / 1000.0)

    @classmethod
    def from_tag_centers(
        cls, centers_mm: dict[int, tuple[float, float]], tag_edge_m: float
    ) -> BoardLayout:
        """Tag corners from tag centres in board millimetres and the tag edge in meters."""
        edge_mm = tag_edge_m * 1000.0
        corners: dict[int, F64] = {}
        for tag_id, (cx, cy) in centers_mm.items():
            lower_left = np.array([cx - edge_mm / 2.0, cy - edge_mm / 2.0])
            xy = lower_left + _CORNER_SIGNS * edge_mm
            corners[tag_id] = np.column_stack([xy, np.zeros(4)]) / 1000.0
        return cls(tag_corners_m=corners)


@dataclass(frozen=True)
class SimilarityFit:
    """``x_board = scale * R @ x_recon + t``; ``rms_m`` is the residual in meters."""

    scale: float
    R_board_from_recon: F64
    t_board_from_recon: F64
    rms_m: float
    n_points: int


def fit_similarity(src: F64, dst: F64) -> SimilarityFit:
    """Least-squares similarity transform src -> dst (Umeyama 1991), reflections excluded."""
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"need matching (N, 3) arrays, got {src.shape} and {dst.shape}")
    n = src.shape[0]
    if n < 3:
        raise ValueError("need at least 3 point pairs")
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    s0, d0 = src - mu_s, dst - mu_d
    var_s = float((s0**2).sum() / n)
    if var_s <= 0.0:
        raise ValueError("source points are all identical")
    u, sing, vt = np.linalg.svd(d0.T @ s0 / n)
    # Flip the last axis if needed so R is a proper rotation, not a mirror.
    d = np.diag([1.0, 1.0, 1.0 if np.linalg.det(u) * np.linalg.det(vt) > 0 else -1.0])
    rot = u @ d @ vt
    scale = float(np.trace(np.diag(sing) @ d) / var_s)
    t = mu_d - scale * rot @ mu_s
    resid = dst - (scale * (rot @ src.T).T + t)
    rms = float(np.sqrt((resid**2).sum(axis=1).mean()))
    return SimilarityFit(scale, rot, t, rms, n)


def fit_board(layout: BoardLayout, detected_corners_recon: dict[int, F64]) -> SimilarityFit:
    """Fit the grid board to tag corners triangulated in the reconstruction frame.

    Grid benchmark scenes only: it needs the known tag positions, so it is a check on
    ``estimate_scale`` there and not part of the pipeline.

    ``detected_corners_recon`` maps tag id to a (4, 3) array in the layout's corner order.
    Ids that are not on the board are an error rather than being ignored, because a wrong id
    means a wrong tag was matched.
    """
    unknown = set(detected_corners_recon) - set(layout.tag_corners_m)
    if unknown:
        raise ValueError(f"tag ids not on the board: {sorted(unknown)}")
    ids = sorted(detected_corners_recon)
    if len(ids) < 2:
        raise ValueError("need at least 2 tags to fix scale and orientation")
    src = np.concatenate([detected_corners_recon[i] for i in ids])
    dst = np.concatenate([layout.tag_corners_m[i] for i in ids])
    return fit_similarity(src, dst)


@dataclass(frozen=True)
class ScaleEstimate:
    """Metric scale from tag edges. ``scale`` multiplies reconstruction units to give meters."""

    scale: float
    per_tag: dict[int, float]
    spread: float  # (max - min) / median over tags; 0 with a single tag
    disagree: bool  # tags disagree by more than the limit; surfaced in the QA report


def estimate_scale(
    corners_recon: dict[int, F64], tag_edge_m: float, max_spread: float
) -> ScaleEstimate:
    """Scale from the printed tag edge (P1.2), one estimate per tag, median over tags.

    Works with a single tag. With two or more, a spread above ``max_spread`` sets ``disagree``
    instead of dropping a tag, because the pipeline must not quietly pick one.
    """
    if not corners_recon:
        raise ValueError("need at least one tag")
    per_tag: dict[int, float] = {}
    for tag_id, c in corners_recon.items():
        edges = np.linalg.norm(c - np.roll(c, -1, axis=0), axis=1)
        mean_edge = float(edges.mean())
        if mean_edge <= 0.0:
            raise ValueError(f"tag {tag_id} has zero size in the reconstruction")
        per_tag[tag_id] = tag_edge_m / mean_edge
    values = np.array(list(per_tag.values()))
    median = float(np.median(values))
    spread = float((values.max() - values.min()) / median) if len(values) > 1 else 0.0
    return ScaleEstimate(median, per_tag, spread, spread > max_spread)


def reference_tag_id(n_views: dict[int, int]) -> int:
    """The tag that carries the world origin: seen in the most frames, lowest id on a tie."""
    if not n_views:
        raise ValueError("need at least one tag")
    return min(n_views, key=lambda i: (-n_views[i], i))


def world_from_tag(tag_corners_scaled: F64, table: Plane) -> F64:
    """World frame on a reference tag (P1.3). Returns ``T_world_from_scaled_recon``.

    Up is the table normal. The origin is the tag centre dropped onto the table plane, and
    world X is the tag's lower-left to lower-right edge projected into the plane. Corners must
    already be scaled with ``apply_scale``, in the layout's corner order, and ``table`` must be
    fitted on points in the same scaled frame.
    """
    up = np.asarray(table.normal, dtype=np.float64)
    x_edge = tag_corners_scaled[1] - tag_corners_scaled[0]
    x_axis = x_edge - (x_edge @ up) * up
    norm = float(np.linalg.norm(x_axis))
    if norm < 1e-9:
        raise ValueError("tag edge is parallel to the table normal; tag is not on the table")
    x_axis /= norm
    rot_recon_from_world = np.column_stack([x_axis, np.cross(up, x_axis), up])
    centre = tag_corners_scaled.mean(axis=0)
    origin = centre - (up @ centre - table.offset_m) * up
    t = np.eye(4)
    t[:3, :3] = rot_recon_from_world.T
    t[:3, 3] = -rot_recon_from_world.T @ origin
    return t


def apply_scale(points: F64, scale: float) -> F64:
    """The one place the metric scale is applied to reconstruction geometry."""
    return points * scale


@dataclass(frozen=True)
class PlaneFit:
    plane: Plane
    inlier_mask: NDArray[np.bool_]
    inlier_fraction: float
    rms_m: float


def fit_plane_ransac(
    points: F64,
    *,
    inlier_threshold_m: float,
    iterations: int,
    seed: int,
    up_hint: F64,
) -> PlaneFit:
    """RANSAC plane through metric points, refined on the inliers by SVD (P1.3).

    ``up_hint`` picks the sign of the normal (for example the mean direction from the table
    to the cameras). The seed makes the result repeatable.
    """
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        raise ValueError("need at least 3 points of shape (N, 3)")
    rng = np.random.default_rng(seed)
    best_mask: NDArray[np.bool_] | None = None
    best_count = -1
    for _ in range(iterations):
        a, b, c = points[rng.choice(len(points), size=3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = float(np.linalg.norm(n))
        if norm < 1e-12:
            continue
        n /= norm
        mask: NDArray[np.bool_] = np.abs((points - a) @ n) < inlier_threshold_m
        count = int(mask.sum())
        if count > best_count:
            best_mask, best_count = mask, count
    if best_mask is None or best_count < 3:
        raise ValueError("no plane found: points are degenerate")

    inl = points[best_mask]
    centroid = inl.mean(axis=0)
    # Thin SVD: the full U would be (N, N), tens of GiB for dense point clouds.
    normal = np.linalg.svd(inl - centroid, full_matrices=False)[2][-1]
    if float(normal @ up_hint) < 0:
        normal = -normal
    dist = (points - centroid) @ normal
    mask = np.abs(dist) < inlier_threshold_m
    rms = float(np.sqrt((dist[mask] ** 2).mean()))
    plane = Plane(
        normal=(float(normal[0]), float(normal[1]), float(normal[2])),
        offset_m=float(normal @ centroid),
    )
    return PlaneFit(plane, mask, float(mask.mean()), rms)


def world_from_recon(fit: SimilarityFit, table: Plane) -> F64:
    """Build the gravity-aligned world frame (P1.3). Returns ``T_world_from_scaled_recon``.

    Tilt comes from the table normal, which uses thousands of points and so beats the tilt of a
    fit to six tags. Yaw and origin come from the board fit: the board X axis, projected into
    the table plane, becomes world X, and the board origin projected onto the plane becomes the
    world origin. The returned 4x4 is rigid and acts on points that ``apply_scale`` already
    scaled by ``s = fit.scale``; ``table`` must be fitted on those scaled points.
    """
    up = np.asarray(table.normal, dtype=np.float64)
    r_board = fit.R_board_from_recon
    x_board = r_board.T @ np.array([1.0, 0.0, 0.0])  # board X axis in recon directions
    x_axis = x_board - (x_board @ up) * up
    norm = float(np.linalg.norm(x_axis))
    if norm < 1e-9:
        raise ValueError("board X axis is parallel to the table normal; board is not on the table")
    x_axis /= norm
    y_axis = np.cross(up, x_axis)
    rot_recon_from_world = np.column_stack([x_axis, y_axis, up])

    # Board origin in the scaled recon frame, then dropped onto the table plane.
    origin_scaled = r_board.T @ -fit.t_board_from_recon
    origin = origin_scaled - (up @ origin_scaled - table.offset_m) * up

    t = np.eye(4)
    t[:3, :3] = rot_recon_from_world.T
    t[:3, 3] = -rot_recon_from_world.T @ origin
    return t


def angle_between_deg(a: F64, b: F64) -> float:
    """Angle between two directions, for the up-axis check against the table normal."""
    c = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))
