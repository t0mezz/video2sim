"""Register a generated mesh to the observed object (P3.2).

A shape model returns a mesh in its own frame and units with a guessed pose. This finds the
similarity transform (scale, rotation, translation) that places it on the object in the metric
world frame, by minimising the PLAN.md cost

    mean_y min_x |y - (s R x + t)|^2  +  lambda * sum_f L_mask(projection in frame f, mask f)

* The point term is one-sided: every observed point should lie on the mesh, but a mesh point
  with no observation near it costs nothing, so the unseen back of the object is free. Mesh
  points are a dense area-weighted sample of the surface; nearest neighbours are found again
  in every outer round (ICP style) and weighted with a Huber loss, so the 1 to 2 cm strays of
  short SfM tracks (``stages/lift.py``) do not drag the fit.
* The mask term stands in for rendering, in two parts. Outside: surface samples are projected
  into each view and each is charged its pixel distance to the nearest allowed pixel (a
  distance transform). Allowed is the object's own mask plus the masks of other objects,
  because a neighbour in front may hide the object there; a sample outside the image costs
  nothing. Reach: along ``mask_directions`` image directions, the projected mesh must reach as
  far as the mask, no farther and no shorter. A mesh's reach along a direction is set by a
  vertex, so this is exact without rendering. A direction is skipped where the mask's farthest
  pixel touches another object or the image edge, since the object may go on there. The
  silhouette is the projection of the whole object, so neither part needs a visibility test.
* Points alone cannot hold the unseen sides: with only the top of a box observed, the scale
  came out 3.5% short without the masks and 0.7% off with them (``tests/test_registration.py``).
* Resting prior, for objects that stand on something. Upright: the model axis (one of +-x,
  +-y, +-z) that the start rotation turns closest to world up is levelled at the start and held
  near vertical by a soft term, so a box lying on any face keeps that face down. Without it,
  points and masks from a short camera arc let s01's phone box tilt 8 degrees and read 32% tall.
  Contact: when the caller gives a support height, the lowest mesh vertex is held on it. It is
  the caller's to give, because a stacked object rests on another object, not the table.

* Stretch. A generated mesh can be right in shape but wrong in proportion, and one scale cannot
  fix that: on the dev scenes the worst of the best sample's axes read 6% off where a primitive
  read 2 to 3%. After the similarity has converged, the fit is refined once more with a scale
  along each model axis, held to 1 by ``stretch_sigma``, a prior on the log of each. The fit
  stays with the masks and points, not with a fixed box, so a poor cluster does not reshape a
  good mesh by itself. This is ``refine_stretch``, a separate step, so a caller can spend it
  on its best samples only.

Scale comes from this fit against the metric cloud and masks, never from the model (AGENTS.md).
The start takes only the rotation of the model's guess. Scale starts from the ratio of box
diagonals and the centre from the box centre of the observed points, because the model's
translation and scale are in its own depth units. Rotation is also tried at
``yaw_hypotheses`` turns about world up, since a model's guess can be a quarter turn off for a
box-like object. Each start gets ``yaw_screen_iterations`` rounds and only the cheapest is
refined to convergence.

Solver: Levenberg-Marquardt over seven parameters (log scale, rotation vector, translation),
ten with the log stretches, with exact Jacobians (``_Problem.jacobian``). The rotation is about
the centroid of the surface samples, so rotation and translation updates do not couple.
Nearest neighbours come from a KD-tree.

The weights in ``shape.registration`` are first guesses, to be tuned on the dev scenes in P3.3.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from video2sim.schemas import SE3, Intrinsics
from video2sim.stages.lift import MIN_DEPTH_M, _pixels

F64 = NDArray[np.float64]
I64 = NDArray[np.int64]

_INSIDE_PX = 1.0  # a projected sample this close to an allowed pixel counts as inside
_CONVERGED = 1e-5  # stop the outer rounds when the cost falls by less than this share


@dataclass(frozen=True)
class RegistrationConfig:
    """Values from ``shape.registration`` in ``configs/default.yaml``."""

    surface_samples: int
    mask_samples: int
    mask_directions: int
    max_vertices: int
    max_target_points: int
    sigma_points_m: float
    sigma_mask_px: float
    mask_weight: float
    huber_m: float
    inlier_m: float
    yaw_hypotheses: int
    yaw_screen_iterations: int
    outer_iterations: int
    inner_iterations: int
    upright_sigma_deg: float  # 0 turns the upright term off
    contact_sigma_m: float
    stretch_sigma: float  # log scale along each model axis; 0 keeps the mesh's proportions

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> RegistrationConfig:
        c = cfg["shape"]["registration"]
        out = cls(
            int(c["surface_samples"]),
            int(c["mask_samples"]),
            int(c["mask_directions"]),
            int(c["max_vertices"]),
            int(c["max_target_points"]),
            float(c["sigma_points_m"]),
            float(c["sigma_mask_px"]),
            float(c["mask_weight"]),
            float(c["huber_m"]),
            float(c["inlier_m"]),
            int(c["yaw_hypotheses"]),
            int(c["yaw_screen_iterations"]),
            int(c["outer_iterations"]),
            int(c["inner_iterations"]),
            float(c["upright_sigma_deg"]),
            float(c["contact_sigma_m"]),
            float(c["stretch_sigma"]),
        )
        counts = (
            out.surface_samples,
            out.mask_samples,
            out.mask_directions,
            out.max_vertices,
            out.max_target_points,
            out.yaw_hypotheses,
            out.yaw_screen_iterations,
            out.outer_iterations,
            out.inner_iterations,
        )
        if min(counts) < 1:
            raise ValueError("sample counts, yaw_hypotheses and iterations must be at least 1")
        if min(out.sigma_points_m, out.sigma_mask_px, out.huber_m, out.inlier_m) <= 0:
            raise ValueError("sigmas, huber_m and inlier_m must be positive")
        if out.mask_weight < 0:
            raise ValueError("mask_weight must not be negative")
        if not 0 <= out.upright_sigma_deg < 90:
            raise ValueError("upright_sigma_deg must be in [0, 90)")
        if out.contact_sigma_m <= 0:
            raise ValueError("contact_sigma_m must be positive")
        if out.stretch_sigma < 0:
            raise ValueError("stretch_sigma must not be negative")
        return out


@dataclass(frozen=True)
class MaskView:
    """One registered frame with the object's mask, and optionally where others hide it."""

    T_world_from_camera: SE3
    intrinsics: Intrinsics
    mask: NDArray[np.bool_]  # [row, column]
    occluders: NDArray[np.bool_] | None = None  # union of the other objects' masks

    def __post_init__(self) -> None:
        shape = (self.intrinsics.height_px, self.intrinsics.width_px)
        if self.mask.shape != shape:
            raise ValueError(f"mask shape {self.mask.shape} does not match the image {shape}")
        if self.occluders is not None and self.occluders.shape != shape:
            raise ValueError(f"occluder shape {self.occluders.shape} does not match {shape}")
        if not self.mask.any():
            raise ValueError("a view for registration needs a non-empty mask")


@dataclass(frozen=True)
class Registration:
    """The mesh goes to the world as ``R @ (scale * stretch * x) + t`` with
    ``T_world_from_model``; ``stretch`` multiplies each model axis."""

    T_world_from_model: SE3
    scale: float
    fit_score: float  # inlier_fraction * mask_inside_fraction, in [0, 1]
    inlier_fraction: float  # observed points within inlier_m of the mesh
    mask_inside_fraction: float  # projected mesh samples on an allowed pixel
    cost: float
    yaw_index: int
    stretch: tuple[float, float, float] = (1.0, 1.0, 1.0)


def rotation_from_vector(v: F64) -> F64:
    """Rotation matrix of a rotation vector (axis times angle in radians), Rodrigues."""
    angle = float(np.linalg.norm(v))
    if angle < 1e-12:
        return np.eye(3)
    k = np.asarray(v, dtype=float) / angle
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    out: F64 = np.eye(3) + np.sin(angle) * kx + (1.0 - np.cos(angle)) * (kx @ kx)
    return out


def _left_jacobian(v: F64) -> F64:
    """J with ``d(exp(v) y) = -[exp(v) y]x J dv``, the left Jacobian of SO(3)."""
    angle = float(np.linalg.norm(v))
    kx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    if angle < 1e-6:
        return np.eye(3) + 0.5 * kx
    a = (1.0 - np.cos(angle)) / angle**2
    b = (angle - np.sin(angle)) / angle**3
    out: F64 = np.eye(3) + a * kx + b * (kx @ kx)
    return out


def sample_surface(vertices: F64, faces: I64, count: int, rng: np.random.Generator) -> F64:
    """Area-weighted uniform samples on a triangle mesh."""
    tri = vertices[faces]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    if area.sum() <= 0:
        raise ValueError("the mesh has no area")
    chosen = rng.choice(len(faces), size=count, p=area / area.sum())
    r1, r2 = rng.random(count), rng.random(count)
    s = np.sqrt(r1)
    a, b, c = (1.0 - s), s * (1.0 - r2), s * r2
    t = tri[chosen]
    out: F64 = a[:, None] * t[:, 0] + b[:, None] * t[:, 1] + c[:, None] * t[:, 2]
    return out


def _nearest(points: F64, candidates: F64) -> tuple[I64, F64]:
    """Index of and distance to the nearest candidate of every point."""
    from scipy.spatial import cKDTree  # type: ignore[import-untyped]  # heavy import

    dist, index = cKDTree(candidates).query(points)
    return np.asarray(index, dtype=np.int64), np.asarray(dist, dtype=float)


@dataclass(frozen=True)
class _View:
    R_camera_from_world: F64
    t_camera_from_world: F64
    intrinsics: Intrinsics
    cost_px: NDArray[np.float32]  # distance to the nearest allowed pixel
    directions: F64  # (D, 2) unit image directions where the mask shows its own edge
    extent_px: F64  # (D,) how far the mask reaches along each of them


def _prepare(view: MaskView, directions: int) -> _View:
    import cv2  # kept out of module import time

    allowed = view.mask if view.occluders is None else view.mask | view.occluders
    # distanceTransform measures to the nearest zero pixel, so allowed pixels are the zeros
    cost = cv2.distanceTransform((~allowed).astype(np.uint8), cv2.DIST_L2, 5)
    # A direction counts only where the mask's farthest pixel is the object's own edge: not
    # next to a neighbour that may hide more of it, not at the image edge.
    blocked = np.zeros_like(view.mask)
    if view.occluders is not None:
        kernel = np.ones((3, 3), np.uint8)
        blocked = cv2.dilate(view.occluders.astype(np.uint8), kernel, iterations=2).astype(bool)
    h, w = view.mask.shape
    rows, cols = np.nonzero(view.mask)
    kept_dirs, kept_ext = [], []
    for angle in np.arange(directions) * 2 * np.pi / directions:
        d = np.array([np.cos(angle), np.sin(angle)])
        reach = cols * d[0] + rows * d[1]
        i = int(np.argmax(reach))
        r, c = int(rows[i]), int(cols[i])
        if blocked[r, c] or r < 2 or c < 2 or r > h - 3 or c > w - 3:
            continue
        kept_dirs.append(d)
        kept_ext.append(float(reach[i]) + 0.5)  # pixel centre to the pixel's outer side
    a = view.T_world_from_camera.inverse().to_numpy()
    return _View(
        a[:3, :3], a[:3, 3], view.intrinsics, np.asarray(cost, dtype=np.float32),
        np.asarray(kept_dirs, dtype=float).reshape(-1, 2), np.asarray(kept_ext, dtype=float),
    )  # fmt: skip


def _project(points: F64, view: _View) -> tuple[F64, F64, NDArray[np.bool_]]:
    """Pixel coordinates of world points, and which of them are in the image."""
    cam = points @ view.R_camera_from_world.T + view.t_camera_from_world
    front = cam[:, 2] > MIN_DEPTH_M
    safe = np.where(front, cam[:, 2], 1.0)
    u, v = _pixels(cam[:, 0] / safe, cam[:, 1] / safe, view.intrinsics)
    h, w = view.cost_px.shape
    seen = front & (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)
    return u, v, seen


def _mask_costs(u: F64, v: F64, seen: NDArray[np.bool_], view: _View) -> F64:
    """Bilinear pixel cost of projected points; zero for points outside the image."""
    h, w = view.cost_px.shape
    u, v = np.clip(u, 0, w - 1), np.clip(v, 0, h - 1)
    u0 = np.minimum(np.floor(u).astype(np.int64), w - 2)
    v0 = np.minimum(np.floor(v).astype(np.int64), h - 2)
    fu, fv = u - u0, v - v0
    c = view.cost_px
    value = (
        (1 - fu) * (1 - fv) * c[v0, u0]
        + fu * (1 - fv) * c[v0, u0 + 1]
        + (1 - fu) * fv * c[v0 + 1, u0]
        + fu * fv * c[v0 + 1, u0 + 1]
    )
    out: F64 = np.where(seen, value, 0.0)
    return out


def _mask_gradients(u: F64, v: F64, seen: NDArray[np.bool_], view: _View) -> tuple[F64, F64]:
    """Derivatives of ``_mask_costs`` along u and v."""
    h, w = view.cost_px.shape
    u, v = np.clip(u, 0, w - 1), np.clip(v, 0, h - 1)
    u0 = np.minimum(np.floor(u).astype(np.int64), w - 2)
    v0 = np.minimum(np.floor(v).astype(np.int64), h - 2)
    fu, fv = u - u0, v - v0
    c = view.cost_px
    c00, c01, c10, c11 = c[v0, u0], c[v0, u0 + 1], c[v0 + 1, u0], c[v0 + 1, u0 + 1]
    du = (1 - fv) * (c01 - c00) + fv * (c11 - c10)
    dv = (1 - fu) * (c10 - c00) + fu * (c11 - c01)
    return np.where(seen, du, 0.0), np.where(seen, dv, 0.0)


def _project_jacobian(points: F64, jac: F64, view: _View) -> tuple[F64, F64]:
    """Derivatives of the pixel coordinates; ``jac`` is (N, 3, P), the two results (N, P)."""
    cam = points @ view.R_camera_from_world.T + view.t_camera_from_world
    dcam = view.R_camera_from_world @ jac
    z = np.where(cam[:, 2] > MIN_DEPTH_M, cam[:, 2], 1.0)
    x, y = cam[:, 0] / z, cam[:, 1] / z
    dx = (dcam[:, 0] - x[:, None] * dcam[:, 2]) / z[:, None]
    dy = (dcam[:, 1] - y[:, None] * dcam[:, 2]) / z[:, None]
    k = view.intrinsics
    k1, k2, p1, p2, k3 = (list(k.distortion) + [0.0] * 5)[:5]
    r2 = x * x + y * y
    radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
    slope = k1 + r2 * (2.0 * k2 + 3.0 * k3 * r2)  # d radial / d r2
    cross = 2.0 * x * y * slope + 2.0 * p1 * x + 2.0 * p2 * y
    xx = radial + 2.0 * x * x * slope + 2.0 * p1 * y + 6.0 * p2 * x
    yy = radial + 2.0 * y * y * slope + 6.0 * p1 * y + 2.0 * p2 * x
    du = k.fx * (xx[:, None] * dx + cross[:, None] * dy)
    dv = k.fy * (cross[:, None] * dx + yy[:, None] * dy)
    return du, dv


def _extent_gaps(points: F64, view: _View) -> F64:
    """Reach of the projected points along each kept direction minus the mask's reach.

    The reach of a mesh along a direction is set by a vertex, so projecting all vertices gives
    it exactly, with no sampling bias. Both signs cost: too far is outside the mask, too short
    leaves part of it uncovered."""
    if len(view.directions) == 0:
        return np.zeros(0)
    u, v, _ = _project(points, view)
    reach = np.column_stack([u, v]) @ view.directions.T
    out: F64 = reach.max(axis=0) - view.extent_px
    return out


@dataclass(frozen=True)
class _Problem:
    samples: F64  # centred surface samples, model units
    mask_samples: F64  # a subset of them for the mask term
    vertices: F64  # centred mesh vertices, for the mask extents
    target: F64
    views: list[_View]
    cfg: RegistrationConfig
    up_model: F64 | None  # the model axis held vertical, or None without the upright term
    support_z: float | None  # height the lowest vertex rests on, or None

    def place(self, p: F64, base: tuple[float, F64, F64], points: F64) -> F64:
        scale0, r0, t0 = base
        r = rotation_from_vector(p[1:4]) @ r0
        stretch = np.exp(p[7:10]) if len(p) > 7 else 1.0
        out: F64 = (scale0 * np.exp(p[0]) * stretch * points) @ r.T + t0 + p[4:7]
        return out

    def tangent(self, p: F64, base: tuple[float, F64, F64], points: F64) -> F64:
        """Derivative of ``place`` by every parameter, (N, 3, P)."""
        scale0, r0, _ = base
        stretch = np.exp(p[7:10]) if len(p) > 7 else np.ones(3)
        scaled = scale0 * np.exp(p[0]) * stretch * points
        turned = rotation_from_vector(p[1:4]) @ r0
        moved = scaled @ turned.T
        out: F64 = np.zeros((len(points), 3, len(p)))
        out[:, :, 0] = moved
        left = _left_jacobian(p[1:4])
        for k in range(3):
            out[:, :, 1 + k] = np.cross(left[:, k], moved)
            out[:, k, 4 + k] = 1.0
        if len(p) > 7:
            for j in range(3):
                out[:, :, 7 + j] = scaled[:, j, None] * turned[:, j]
        return out

    def residuals(self, p: F64, base: tuple[float, F64, F64], index: I64, weight: F64) -> F64:
        cfg = self.cfg
        moved = self.place(p, base, self.samples[index])
        r_points = (np.sqrt(weight)[:, None] * (self.target - moved)).ravel()
        r_points /= cfg.sigma_points_m * np.sqrt(len(self.target))
        parts = [r_points]
        if cfg.mask_weight > 0 and self.views:
            placed = self.place(p, base, self.mask_samples)
            outside = np.sqrt(cfg.mask_weight / (len(self.views) * len(placed))) / cfg.sigma_mask_px
            corners = self.place(p, base, self.vertices)
            for view in self.views:
                u, v, seen = _project(placed, view)
                parts.append(_mask_costs(u, v, seen, view) * outside)
                if len(view.directions):
                    reach = np.sqrt(cfg.mask_weight / (len(self.views) * len(view.directions)))
                    parts.append(_extent_gaps(corners, view) * reach / cfg.sigma_mask_px)
        if self.up_model is not None:
            # the horizontal part of the turned axis is the sine of the tilt
            up = rotation_from_vector(p[1:4]) @ base[1] @ self.up_model
            parts.append(up[:2] / np.sin(np.radians(cfg.upright_sigma_deg)))
        if self.support_z is not None:
            lowest = float(np.min(self.place(p, base, self.vertices)[:, 2]))
            parts.append(np.array([(lowest - self.support_z) / cfg.contact_sigma_m]))
        if len(p) > 7:
            parts.append(p[7:10] / cfg.stretch_sigma)
        return np.concatenate(parts)

    def jacobian(self, p: F64, base: tuple[float, F64, F64], index: I64, weight: F64) -> F64:
        """Derivative of ``residuals``, row for row. Index, weights, the vertex that sets a
        reach and the lowest vertex stay fixed, as they do within a small step."""
        cfg = self.cfg
        chosen = self.samples[index]
        scale = -np.sqrt(weight)[:, None, None] / (cfg.sigma_points_m * np.sqrt(len(self.target)))
        parts = [(scale * self.tangent(p, base, chosen)).reshape(-1, len(p))]
        if cfg.mask_weight > 0 and self.views:
            placed = self.place(p, base, self.mask_samples)
            jac = self.tangent(p, base, self.mask_samples)
            outside = np.sqrt(cfg.mask_weight / (len(self.views) * len(placed))) / cfg.sigma_mask_px
            corners = self.place(p, base, self.vertices)
            for view in self.views:
                u, v, seen = _project(placed, view)
                du, dv = _project_jacobian(placed, jac, view)
                gu, gv = _mask_gradients(u, v, seen, view)
                parts.append((gu[:, None] * du + gv[:, None] * dv) * outside)
                if len(view.directions):
                    uc, vc, _ = _project(corners, view)
                    far = np.argmax(np.column_stack([uc, vc]) @ view.directions.T, axis=0)
                    du, dv = _project_jacobian(
                        corners[far], self.tangent(p, base, self.vertices[far]), view
                    )
                    d = view.directions
                    reach = np.sqrt(cfg.mask_weight / (len(self.views) * len(d)))
                    parts.append(
                        (d[:, 0, None] * du + d[:, 1, None] * dv) * reach / cfg.sigma_mask_px
                    )
        if self.up_model is not None:
            up = rotation_from_vector(p[1:4]) @ base[1] @ self.up_model
            row: F64 = np.zeros((2, len(p)))
            left = _left_jacobian(p[1:4])
            for k in range(3):
                row[:, 1 + k] = np.cross(left[:, k], up)[:2]
            parts.append(row / np.sin(np.radians(cfg.upright_sigma_deg)))
        if self.support_z is not None:
            low = int(np.argmin(self.place(p, base, self.vertices)[:, 2]))
            jac = self.tangent(p, base, self.vertices[low : low + 1])
            parts.append(jac[:, 2] / cfg.contact_sigma_m)
        if len(p) > 7:
            prior = np.zeros((3, len(p)))
            prior[:, 7:] = np.eye(3) / cfg.stretch_sigma
            parts.append(prior)
        return np.concatenate(parts)

    def match(self, p: F64, base: tuple[float, F64, F64]) -> tuple[I64, F64, F64]:
        index, dist = _nearest(self.target, self.place(p, base, self.samples))
        weight = np.where(dist <= self.cfg.huber_m, 1.0, self.cfg.huber_m / np.maximum(dist, 1e-12))
        return index, dist, weight


def _levenberg_marquardt(
    f: Callable[[F64], F64], jacobian: Callable[[F64], F64], p: F64, iterations: int
) -> F64:
    r = f(p)
    cost = float(r @ r)
    damping = 1e-3
    for _ in range(iterations):
        jac = jacobian(p)
        h, g = jac.T @ jac, jac.T @ r
        improved = False
        for _ in range(10):
            delta = np.linalg.solve(h + damping * np.diag(np.diag(h) + 1e-12), -g)
            r_new = f(p + delta)
            cost_new = float(r_new @ r_new)
            if cost_new < cost:
                p, r, cost, improved = p + delta, r_new, cost_new, True
                damping = max(damping / 3, 1e-9)
                break
            damping *= 4
        if not improved:
            break
    return p


def _refine(
    problem: _Problem, base: tuple[float, F64, F64], p: F64, rounds: int
) -> tuple[F64, float]:
    """Nearest-neighbour rounds, each solved by Levenberg-Marquardt; the final cost is taken
    with fresh neighbours, so starts compare fairly."""
    previous = np.inf
    for _ in range(rounds):
        index, _, weight = problem.match(p, base)

        def f(q: F64, index: I64 = index, weight: F64 = weight) -> F64:
            return problem.residuals(q, base, index, weight)

        def jac(q: F64, index: I64 = index, weight: F64 = weight) -> F64:
            return problem.jacobian(q, base, index, weight)

        p = _levenberg_marquardt(f, jac, p, problem.cfg.inner_iterations)
        r = f(p)
        cost = float(r @ r)
        if np.isfinite(previous) and previous - cost <= _CONVERGED * previous:
            break
        previous = cost
    index, _, weight = problem.match(p, base)
    r = problem.residuals(p, base, index, weight)
    return p, float(r @ r)


def level(R_world_from_model: F64) -> tuple[F64, F64]:
    """The model axis (one of +-x, +-y, +-z) that the rotation turns closest to world up, and
    the rotation turned by the least angle so that this axis points straight up."""
    column = int(np.argmax(np.abs(R_world_from_model[2])))
    up_model = np.zeros(3)
    up_model[column] = np.sign(R_world_from_model[2, column])
    up = R_world_from_model @ up_model
    axis = np.cross(up, [0.0, 0.0, 1.0])
    angle = float(np.arctan2(np.linalg.norm(axis), up[2]))
    if np.linalg.norm(axis) > 1e-12:
        axis = axis / np.linalg.norm(axis) * angle
    return rotation_from_vector(axis) @ R_world_from_model, up_model


def _build_problem(
    vertices_model: F64,
    faces: I64,
    target_points: F64,
    views: Sequence[MaskView],
    cfg: RegistrationConfig,
    seed: int,
    up_model: F64 | None,
    support_z: float | None,
) -> tuple[_Problem, F64]:
    """The problem and the surface centroid its points are centred on."""
    rng = np.random.default_rng(seed)
    samples = sample_surface(
        np.asarray(vertices_model, float), np.asarray(faces, np.int64), cfg.surface_samples, rng
    )
    centroid = samples.mean(axis=0)
    samples = samples - centroid
    target = np.asarray(target_points, dtype=float)
    if len(target) > cfg.max_target_points:
        target = target[np.sort(rng.choice(len(target), cfg.max_target_points, replace=False))]
    pick = rng.choice(len(samples), min(cfg.mask_samples, len(samples)), replace=False)
    vertices = np.asarray(vertices_model, dtype=float) - centroid
    if len(vertices) > cfg.max_vertices:
        vertices = vertices[np.sort(rng.choice(len(vertices), cfg.max_vertices, replace=False))]
    prepared = [_prepare(v, cfg.mask_directions) for v in views]
    problem = _Problem(
        samples, samples[np.sort(pick)], vertices, target, prepared, cfg, up_model, support_z
    )
    return problem, centroid


def _result(
    problem: _Problem,
    base: tuple[float, F64, F64],
    p: F64,
    centroid: F64,
    cost: float,
    yaw_index: int,
) -> Registration:
    cfg = problem.cfg
    scale0, r0, t0 = base
    rotation = rotation_from_vector(p[1:4]) @ r0
    scale = scale0 * float(np.exp(p[0]))
    t = t0 + p[4:7]
    stretch = np.exp(p[7:10]) if len(p) > 7 else np.ones(3)

    _, dist, _ = problem.match(p, base)
    inlier = float(np.mean(dist <= cfg.inlier_m))
    inside_total = seen_total = 0
    placed = problem.place(p, base, problem.mask_samples)
    for view in problem.views:
        u, v, seen = _project(placed, view)
        costs = _mask_costs(u, v, seen, view)
        inside_total += int(np.count_nonzero(seen & (costs <= _INSIDE_PX)))
        seen_total += int(np.count_nonzero(seen))
    inside = inside_total / seen_total if seen_total else 0.0

    # The fit placed (x - centroid); fold the centroid into the translation.
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = t - scale * rotation @ (stretch * centroid)
    return Registration(
        T_world_from_model=SE3.from_numpy(matrix),
        scale=scale,
        fit_score=inlier * inside,
        inlier_fraction=inlier,
        mask_inside_fraction=inside,
        cost=cost,
        yaw_index=yaw_index,
        stretch=(float(stretch[0]), float(stretch[1]), float(stretch[2])),
    )


def register(
    vertices_model: F64,
    faces: I64,
    R_world_from_model_start: F64,
    target_points: F64,
    views: Sequence[MaskView],
    cfg: RegistrationConfig,
    seed: int,
    support_z: float | None = None,
) -> Registration:
    """Similarity transform that puts the mesh on the observed points and inside the masks.
    ``support_z`` is the world height the object rests on (the table plane's), if any."""
    if len(target_points) == 0 or len(faces) == 0:
        raise ValueError("registration needs observed points and a mesh with faces")
    R_start = np.asarray(R_world_from_model_start, dtype=float)
    up_model = None
    if cfg.upright_sigma_deg > 0:
        R_start, up_model = level(R_start)
    problem, centroid = _build_problem(
        vertices_model, faces, target_points, views, cfg, seed, up_model, support_z
    )
    samples, target = problem.samples, problem.target

    # Every yaw start gets a few rounds; only the cheapest is refined to convergence.
    lo, hi = target.min(axis=0), target.max(axis=0)
    starts = []
    for k in range(cfg.yaw_hypotheses):
        yaw = rotation_from_vector(np.array([0.0, 0.0, 2 * np.pi * k / cfg.yaw_hypotheses]))
        r0 = yaw @ R_start
        turned = samples @ r0.T
        scale0 = float(np.linalg.norm(hi - lo) / np.linalg.norm(turned.max(0) - turned.min(0)))
        t0 = (lo + hi) / 2 - scale0 * (turned.max(0) + turned.min(0)) / 2
        base = (scale0, r0, t0)
        p, cost = _refine(problem, base, np.zeros(7), cfg.yaw_screen_iterations)
        starts.append((cost, k, base, p))
    _, k, base, p = min(starts, key=lambda s: s[0])
    p, cost = _refine(problem, base, p, cfg.outer_iterations)
    return _result(problem, base, p, centroid, cost, k)


def refine_stretch(
    vertices_model: F64,
    faces: I64,
    start: Registration,
    target_points: F64,
    views: Sequence[MaskView],
    cfg: RegistrationConfig,
    seed: int,
    support_z: float | None = None,
) -> Registration:
    """``start``, a result of ``register`` with the same inputs, refined with a scale along
    each model axis. With ``stretch_sigma`` 0 it is returned as it is."""
    if cfg.stretch_sigma == 0:
        return start
    T = start.T_world_from_model.to_numpy()
    up_model = level(T[:3, :3])[1] if cfg.upright_sigma_deg > 0 else None
    problem, centroid = _build_problem(
        vertices_model, faces, target_points, views, cfg, seed, up_model, support_z
    )
    base = (start.scale, T[:3, :3], T[:3, 3] + start.scale * T[:3, :3] @ centroid)
    p, cost = _refine(problem, base, np.zeros(10), cfg.outer_iterations)
    return _result(problem, base, p, centroid, cost, start.yaw_index)
