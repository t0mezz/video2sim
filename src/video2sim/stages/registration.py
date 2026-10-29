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

Scale comes from this fit against the metric cloud and masks, never from the model (AGENTS.md).
The start takes only the rotation of the model's guess. Scale starts from the ratio of box
diagonals and the centre from the box centre of the observed points, because the model's
translation and scale are in its own depth units. Rotation is also tried at
``yaw_hypotheses`` turns about world up, since a model's guess can be a quarter turn off for a
box-like object. Each start gets ``yaw_screen_iterations`` rounds and only the cheapest is
refined to convergence.

Solver: Levenberg-Marquardt over seven parameters (log scale, rotation vector, translation),
with central-difference Jacobians, in NumPy only. The rotation is about the centroid of the
surface samples, so rotation and translation updates do not couple.

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

_STEP = 1e-6  # central-difference step for every parameter (log scale, radians, meters)
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
    """The mesh goes to the world as ``R @ (scale * x) + t`` with ``T_world_from_model``."""

    T_world_from_model: SE3
    scale: float
    fit_score: float  # inlier_fraction * mask_inside_fraction, in [0, 1]
    inlier_fraction: float  # observed points within inlier_m of the mesh
    mask_inside_fraction: float  # projected mesh samples on an allowed pixel
    cost: float
    yaw_index: int


def rotation_from_vector(v: F64) -> F64:
    """Rotation matrix of a rotation vector (axis times angle in radians), Rodrigues."""
    angle = float(np.linalg.norm(v))
    if angle < 1e-12:
        return np.eye(3)
    k = np.asarray(v, dtype=float) / angle
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    out: F64 = np.eye(3) + np.sin(angle) * kx + (1.0 - np.cos(angle)) * (kx @ kx)
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


def _nearest(points: F64, candidates: F64, chunk: int = 2048) -> tuple[I64, F64]:
    """Index of and distance to the nearest candidate of every point, by brute force."""
    index = np.empty(len(points), dtype=np.int64)
    dist = np.empty(len(points))
    cand_sq = (candidates**2).sum(axis=1)
    for start in range(0, len(points), chunk):
        p = points[start : start + chunk]
        d2 = (p**2).sum(axis=1)[:, None] - 2.0 * p @ candidates.T + cand_sq[None, :]
        j = np.argmin(d2, axis=1)
        index[start : start + chunk] = j
        dist[start : start + chunk] = np.sqrt(np.maximum(d2[np.arange(len(p)), j], 0.0))
    return index, dist


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

    def place(self, p: F64, base: tuple[float, F64, F64], points: F64) -> F64:
        scale0, r0, t0 = base
        r = rotation_from_vector(p[1:4]) @ r0
        out: F64 = (scale0 * np.exp(p[0]) * points) @ r.T + t0 + p[4:7]
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
        return np.concatenate(parts)

    def match(self, p: F64, base: tuple[float, F64, F64]) -> tuple[I64, F64, F64]:
        index, dist = _nearest(self.target, self.place(p, base, self.samples))
        weight = np.where(dist <= self.cfg.huber_m, 1.0, self.cfg.huber_m / np.maximum(dist, 1e-12))
        return index, dist, weight


def _levenberg_marquardt(f: Callable[[F64], F64], p: F64, iterations: int) -> F64:
    r = f(p)
    cost = float(r @ r)
    damping = 1e-3
    for _ in range(iterations):
        jac = np.empty((len(r), len(p)))
        for i in range(len(p)):
            step = np.zeros(len(p))
            step[i] = _STEP
            jac[:, i] = (f(p + step) - f(p - step)) / (2 * _STEP)
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

        p = _levenberg_marquardt(f, p, problem.cfg.inner_iterations)
        r = f(p)
        cost = float(r @ r)
        if np.isfinite(previous) and previous - cost <= _CONVERGED * previous:
            break
        previous = cost
    index, _, weight = problem.match(p, base)
    r = problem.residuals(p, base, index, weight)
    return p, float(r @ r)


def register(
    vertices_model: F64,
    faces: I64,
    R_world_from_model_start: F64,
    target_points: F64,
    views: Sequence[MaskView],
    cfg: RegistrationConfig,
    seed: int,
) -> Registration:
    """Similarity transform that puts the mesh on the observed points and inside the masks."""
    if len(target_points) == 0 or len(faces) == 0:
        raise ValueError("registration needs observed points and a mesh with faces")
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
    problem = _Problem(samples, samples[np.sort(pick)], vertices, target, prepared, cfg)

    # Every yaw start gets a few rounds; only the cheapest is refined to convergence.
    lo, hi = target.min(axis=0), target.max(axis=0)
    starts = []
    for k in range(cfg.yaw_hypotheses):
        yaw = rotation_from_vector(np.array([0.0, 0.0, 2 * np.pi * k / cfg.yaw_hypotheses]))
        r0 = yaw @ R_world_from_model_start
        turned = samples @ r0.T
        scale0 = float(np.linalg.norm(hi - lo) / np.linalg.norm(turned.max(0) - turned.min(0)))
        t0 = (lo + hi) / 2 - scale0 * (turned.max(0) + turned.min(0)) / 2
        base = (scale0, r0, t0)
        p, cost = _refine(problem, base, np.zeros(7), cfg.yaw_screen_iterations)
        starts.append((cost, k, base, p))
    _, k, base, p = min(starts, key=lambda s: s[0])
    p, cost = _refine(problem, base, p, cfg.outer_iterations)
    scale0, r0, t0 = base
    rotation = rotation_from_vector(p[1:4]) @ r0
    scale = scale0 * float(np.exp(p[0]))
    t = t0 + p[4:7]

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
    matrix[:3, 3] = t - scale * rotation @ centroid
    return Registration(
        T_world_from_model=SE3.from_numpy(matrix),
        scale=scale,
        fit_score=inlier * inside,
        inlier_fraction=inlier,
        mask_inside_fraction=inside,
        cost=cost,
        yaw_index=k,
    )
