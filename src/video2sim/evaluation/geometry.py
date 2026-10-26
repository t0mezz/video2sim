"""Resting-pose box of a point set, in the convention of the ground truth (P0.6).

Ground truth is ``[length, width, height]`` with ``length >= width``, a yaw about +Z of the
long side, and the bounding-box center. ``resting_box`` recovers the same quantities from the
world-frame vertices of a predicted mesh, so the two sides are compared like for like. An
axis-aligned box would inflate the footprint of any object that is not aligned with X and Y.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Points = NDArray[np.float64]


@dataclass(frozen=True)
class RestingBox:
    size_m: tuple[float, float, float]  # length >= width, height is the Z extent
    center_m: tuple[float, float, float]
    yaw_deg: float  # direction of the long side about +Z, in [0, 180)


def convex_hull_2d(points: Points) -> Points:
    """Andrew's monotone chain. Counter-clockwise hull without repeated points."""
    unique = np.unique(np.round(points, 12), axis=0)
    if len(unique) <= 2:
        return unique
    pts = [tuple(p) for p in unique]  # np.unique sorts by x, then y

    def cross(o: tuple[float, ...], a: tuple[float, ...], b: tuple[float, ...]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    chains: list[list[tuple[float, ...]]] = []
    for sequence in (pts, pts[::-1]):
        chain: list[tuple[float, ...]] = []
        for p in sequence:
            while len(chain) >= 2 and cross(chain[-2], chain[-1], p) <= 0:
                chain.pop()
            chain.append(p)
        chains.append(chain[:-1])
    return np.asarray(chains[0] + chains[1], dtype=float)


def resting_box(vertices: Points) -> RestingBox:
    """Minimum-area rectangle of the XY footprint, plus the Z extent.

    The optimal rectangle has a side flush with a hull edge, so one candidate per hull edge
    is enough. Each is evaluated directly with dot products: hulls have tens of vertices, so
    rotating calipers (O(n) in total) would add code and no measurable speed.
    """
    v = np.asarray(vertices, dtype=float)
    if v.ndim != 2 or v.shape[1] != 3 or len(v) == 0:
        raise ValueError(f"expected an (N, 3) array of vertices, got shape {v.shape}")
    if not np.isfinite(v).all():
        raise ValueError("vertices contain NaN or infinity")

    hull = convex_hull_2d(v[:, :2])
    edges = np.roll(hull, -1, axis=0) - hull
    lengths = np.linalg.norm(edges, axis=1)
    edges = edges[lengths > 1e-12]
    if len(edges) == 0:  # all points coincide in XY: no direction, take the X axis
        directions = np.array([[1.0, 0.0]])
    else:
        directions = edges / np.linalg.norm(edges, axis=1, keepdims=True)

    best: tuple[float, float, float, float, float, float] | None = None
    for u in directions:
        n = np.array([-u[1], u[0]])
        a, b = hull @ u, hull @ n
        extent_u, extent_n = float(a.max() - a.min()), float(b.max() - b.min())
        area = extent_u * extent_n
        if best is None or area < best[0] - 1e-15:
            best = (
                area,
                extent_u,
                extent_n,
                float(a.min() + a.max()) / 2,
                float(b.min() + b.max()) / 2,
                math.atan2(u[1], u[0]),
            )
    assert best is not None
    _, extent_u, extent_n, mid_u, mid_n, angle = best
    u = np.array([math.cos(angle), math.sin(angle)])
    n = np.array([-u[1], u[0]])
    center_xy = u * mid_u + n * mid_n
    if extent_u >= extent_n:
        length, width, long_angle = extent_u, extent_n, angle
    else:
        length, width, long_angle = extent_n, extent_u, angle + math.pi / 2
    z_min, z_max = float(v[:, 2].min()), float(v[:, 2].max())
    return RestingBox(
        size_m=(length, width, z_max - z_min),
        center_m=(float(center_xy[0]), float(center_xy[1]), (z_min + z_max) / 2),
        yaw_deg=math.degrees(long_angle) % 180.0,
    )


def box_vertices(
    size_m: tuple[float, float, float], center_m: tuple[float, float, float], yaw_deg: float
) -> Points:
    """The 8 corners of a box with the given resting-pose size, center and yaw."""
    half = np.asarray(size_m, dtype=float) / 2
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    local = signs * half
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return np.asarray(local @ rotation.T + np.asarray(center_m), dtype=float)
