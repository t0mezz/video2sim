"""Collision geometry for the physics stage (P4.2).

Input is a repaired, closed mesh in the object frame (``stages/mesh_repair.py``); output is a list
of convex parts in the same frame, the way a simulator wants them.

* A primitive is used when the object is close to one. An oriented box, a cylinder along each of
  the three box axes and a sphere are fitted to the mesh and the best is kept if its outline is
  close (95th percentile of the surface distances in both directions, as a share of the mesh's
  bounding-box diagonal, at most ``primitive_tolerance``) and its volume is within
  ``primitive_volume_ratio`` of the sealed body's (``stages/mass.py``). The outline test uses
  surfaces, not inside-outside points, because that needs ``rtree``. A primitive that then fails
  a check is dropped for the decomposition, with ``steps["primitive_rejected"]``.
* Otherwise the mesh is split into at most ``physics.max_convex_parts`` convex parts by CoACD.
  Without ``coacd`` the single convex hull stands in and ``steps["coacd_unavailable"]`` says so;
  the ``collision_fit`` check then decides whether the hull is too fat to use.
* The checks go into ``SimAsset.checks``: ``collision_loads`` (every part is finite, closed,
  convex, has positive volume and at most ``max_part_vertices`` vertices, and there are no more
  than ``max_convex_parts`` of them), and ``collision_fit`` (the parts follow the visual surface,
  measured on surfaces and not on volumes, see below). Loading in the simulator is checked in
  P4.7.

The fit is two numbers, both as a share of the visual mesh's bounding-box diagonal. Coverage is
the share of visual surface points inside a part or within ``fit_cover_tolerance`` of one; below
``fit_min_coverage`` the object would sink into what it touches. Tightness is the 95th percentile
of the distance from the parts' surfaces to the visual surface, counting only what lies outside
the sealed body. The cuts between parts lie inside it; above ``fit_tolerance`` the parts fill in
concavities, so a convex hull of a cup or a case fails. A volume ratio was tried first and does
not work: the meshes SAM 3D gives are often thin shells (walls under 2% of the object's size),
whose enclosed volume says nothing about their shape. Asking which side of the nearest visual
surface a point is on does not work either: a cut through the cavity of a shell faces an inner
wall and read as outside, which failed 11 of 61 dev objects whose plain hulls were tight
(D-022). The body seals mouths under about 8% of the longest side, so parts that fill such a
narrow neck are not caught.

Convex hulls with more than ``max_part_vertices`` vertices are thinned by edge collapse and
re-hulled, which shrinks them slightly. Edge collapse stalls on long smooth hulls, and farthest
point selection then takes over.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import trimesh
from numpy.typing import NDArray

from video2sim.stages.mass import MassConfig, SealedBody, sealed_body

F64 = NDArray[np.float64]
Kind = Literal["box", "cylinder", "sphere", "convex_decomposition", "convex_hull"]
PRIMITIVES: tuple[Kind, ...] = ("box", "cylinder", "sphere")
SURFACE_SAMPLES = 4000


@dataclass(frozen=True)
class CollisionConfig:
    """Values from ``physics`` in ``configs/default.yaml``."""

    max_convex_parts: int
    primitive_tolerance: float
    primitive_volume_ratio: tuple[float, float]
    fit_tolerance: float
    fit_min_coverage: float
    fit_cover_tolerance: float
    max_part_vertices: int
    coacd_threshold: float
    seed: int
    sealing: MassConfig

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> CollisionConfig:
        p = cfg["physics"]
        c = p["collision"]
        out = cls(
            max_convex_parts=int(p["max_convex_parts"]),
            primitive_tolerance=float(c["primitive_tolerance"]),
            primitive_volume_ratio=(
                float(c["primitive_volume_ratio"][0]),
                float(c["primitive_volume_ratio"][1]),
            ),
            fit_tolerance=float(c["fit_tolerance"]),
            fit_min_coverage=float(c["fit_min_coverage"]),
            fit_cover_tolerance=float(c["fit_cover_tolerance"]),
            max_part_vertices=int(c["max_part_vertices"]),
            coacd_threshold=float(c["coacd_threshold"]),
            seed=int(c["seed"]),
            sealing=MassConfig.from_config(cfg),
        )
        if out.max_convex_parts < 1:
            raise ValueError("physics.max_convex_parts must be at least 1")
        if out.primitive_tolerance <= 0:
            raise ValueError("physics.collision.primitive_tolerance must be positive")
        lo, hi = out.primitive_volume_ratio
        if not 0 < lo <= 1 <= hi:
            raise ValueError(
                "physics.collision.primitive_volume_ratio must satisfy 0 < low <= 1 <= high"
            )
        if out.fit_tolerance <= 0 or out.fit_cover_tolerance <= 0:
            raise ValueError("physics.collision fit tolerances must be positive")
        if not 0 < out.fit_min_coverage <= 1:
            raise ValueError("physics.collision.fit_min_coverage must be in (0, 1]")
        if out.max_part_vertices < 4:
            raise ValueError("physics.collision.max_part_vertices must be at least 4")
        return out


@dataclass(frozen=True)
class CollisionResult:
    parts: list[trimesh.Trimesh]
    kind: Kind
    checks: dict[str, bool]
    primitive_score: float | None = None
    steps: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return all(self.checks.values())


def _farthest_vertices(hull: trimesh.Trimesh, max_vertices: int) -> trimesh.Trimesh:
    """The hull of ``max_vertices`` vertices picked by farthest point selection."""
    v = np.asarray(hull.vertices, dtype=np.float64)
    picked = int(np.argmax(np.linalg.norm(v - v.mean(axis=0), axis=1)))
    chosen = [picked]
    distance = np.linalg.norm(v - v[picked], axis=1)
    for _ in range(max_vertices - 1):
        picked = int(np.argmax(distance))
        chosen.append(picked)
        distance = np.minimum(distance, np.linalg.norm(v - v[picked], axis=1))
    return cast(trimesh.Trimesh, trimesh.convex.convex_hull(v[chosen]))


def _limit_vertices(hull: trimesh.Trimesh, max_vertices: int) -> trimesh.Trimesh:
    """Thin a convex hull to at most ``max_vertices`` vertices; it stays convex."""
    if len(hull.vertices) <= max_vertices:
        return hull
    try:
        import fast_simplification  # type: ignore[import-untyped]  # heavy import
    except ImportError:
        return _farthest_vertices(hull, max_vertices)
    target = 2 * max_vertices - 4  # faces of a triangulated convex polyhedron
    for _ in range(6):
        v, _f = fast_simplification.simplify(
            np.asarray(hull.vertices, np.float64),
            np.asarray(hull.faces, np.int32),
            target_count=target,
            agg=7,
        )
        out = cast(trimesh.Trimesh, trimesh.convex.convex_hull(v))
        if len(out.vertices) <= max_vertices:
            return out
        target = int(target * 0.8)
    return _farthest_vertices(hull, max_vertices)


def _part_loads(p: trimesh.Trimesh, cfg: CollisionConfig, min_volume_m3: float) -> bool:
    if len(p.faces) < 4 or not np.isfinite(p.vertices).all() or not p.is_watertight:
        return False
    if len(p.vertices) > cfg.max_part_vertices or float(p.volume) <= min_volume_m3:
        return False
    return bool(trimesh.convex.is_convex(p))  # type: ignore[no-untyped-call]


FIT_VISUAL_SAMPLES = 20000
FIT_PART_SAMPLES = 1000


def _plane_distance(part: trimesh.Trimesh, points: F64) -> F64:
    """How far each point is outside a convex ``part``, as the largest face-plane distance.

    Zero or negative inside. Exact when the nearest feature is a face and a lower bound near
    edges and corners, which is enough for a coverage test at a tolerance of a percent.
    """
    tri = part.triangles[:, 0, :]
    d = np.einsum("pfk,fk->pf", points[:, None, :] - tri[None, :, :], part.face_normals)
    return cast(F64, d.max(axis=1))


def surface_fit(
    parts: list[trimesh.Trimesh],
    visual: trimesh.Trimesh,
    cfg: CollisionConfig,
    body: SealedBody | None = None,
) -> tuple[float, float]:
    """``(coverage, tightness)`` of ``parts`` on ``visual``; see the module docstring.

    ``body`` is the sealed body of ``visual``, for a caller that already has it.
    """
    from scipy.spatial import cKDTree  # type: ignore[import-untyped]  # heavy import

    diag = float(np.linalg.norm(visual.extents))
    body = sealed_body(visual, cfg.sealing) if body is None else body
    vis = trimesh.sample.sample_surface(visual, FIT_VISUAL_SAMPLES, seed=cfg.seed)[0]
    outside = np.min([_plane_distance(p, vis) for p in parts], axis=0)
    covered = outside <= cfg.fit_cover_tolerance * diag
    surf = np.concatenate(
        [trimesh.sample.sample_surface(p, FIT_PART_SAMPLES, seed=cfg.seed)[0] for p in parts]
    )
    dist = cKDTree(vis).query(surf)[0]
    tightness = float(np.percentile(np.where(body.contains(surf), 0.0, dist), 95)) / diag
    return float(covered.mean()), tightness


def check_parts(
    parts: list[trimesh.Trimesh],
    visual: trimesh.Trimesh,
    cfg: CollisionConfig,
    min_volume_m3: float = 1e-12,
    body: SealedBody | None = None,
) -> dict[str, bool]:
    """The ``collision_loads`` and ``collision_fit`` checks for ``parts`` against ``visual``."""
    loads = 1 <= len(parts) <= cfg.max_convex_parts and all(
        _part_loads(p, cfg, min_volume_m3) for p in parts
    )
    fit = False
    if loads and len(visual.faces) > 0:
        coverage, tightness = surface_fit(parts, visual, cfg, body)
        fit = coverage >= cfg.fit_min_coverage and tightness <= cfg.fit_tolerance
    return {"collision_loads": bool(loads), "collision_fit": bool(fit)}


def _axis_to_z(axis: int) -> F64:
    """Rotation taking the cylinder's own z axis onto box axis ``axis``."""
    t = np.eye(4)
    if axis == 0:
        t[:3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
    elif axis == 1:
        t[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    return t


def fit_primitives(mesh: trimesh.Trimesh) -> dict[Kind, trimesh.Trimesh]:
    """Candidate primitives as convex meshes in the mesh's frame: the best cylinder of three."""
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)  # type: ignore[no-untyped-call]
    frame = np.linalg.inv(to_origin)
    box = trimesh.creation.box(extents=extents, transform=frame)
    cylinders: list[trimesh.Trimesh] = []
    for k in range(3):
        a, b = (i for i in range(3) if i != k)
        radius = 0.25 * (extents[a] + extents[b])
        cylinders.append(
            trimesh.creation.cylinder(
                radius=radius, height=extents[k], sections=32, transform=frame @ _axis_to_z(k)
            )
        )
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=float(np.mean(extents)) / 2)
    sphere.apply_translation(frame[:3, 3])
    return {"box": box, "cylinder": _best_cylinder(mesh, cylinders), "sphere": sphere}


def _surface_error(mesh: trimesh.Trimesh, other: trimesh.Trimesh, seed: int) -> float:
    """95th percentile of the surface distances in both directions, over the mesh's diagonal."""
    from scipy.spatial import cKDTree  # heavy import

    a = trimesh.sample.sample_surface(mesh, SURFACE_SAMPLES, seed=seed)[0]
    b = trimesh.sample.sample_surface(other, SURFACE_SAMPLES, seed=seed)[0]
    d = np.concatenate([cKDTree(b).query(a)[0], cKDTree(a).query(b)[0]])
    diag = float(np.linalg.norm(mesh.extents))
    return float(np.percentile(d, 95)) / diag


def _best_cylinder(mesh: trimesh.Trimesh, cylinders: list[trimesh.Trimesh]) -> trimesh.Trimesh:
    return min(cylinders, key=lambda c: _surface_error(mesh, c, 0))


def primitive_for(
    mesh: trimesh.Trimesh, cfg: CollisionConfig, body: SealedBody
) -> tuple[Kind, trimesh.Trimesh, float] | None:
    """The best close primitive for ``mesh``, or None when the object is none of them."""
    if not mesh.is_watertight or body.volume_m3 <= 0:
        return None
    best: tuple[Kind, trimesh.Trimesh, float] | None = None
    for kind, prim in fit_primitives(mesh).items():
        score = _surface_error(mesh, prim, cfg.seed)
        ratio = float(prim.volume) / body.volume_m3
        lo, hi = cfg.primitive_volume_ratio
        close = score <= cfg.primitive_tolerance and lo <= ratio <= hi
        if close and (best is None or score < best[2]):
            best = (kind, prim, score)
    return best


def _decompose(
    mesh: trimesh.Trimesh, cfg: CollisionConfig, steps: dict[str, int]
) -> list[trimesh.Trimesh]:
    try:
        import coacd  # type: ignore[import-untyped,import-not-found,unused-ignore]  # heavy import
    except ImportError:
        steps["coacd_unavailable"] = 1
        return [_limit_vertices(cast(trimesh.Trimesh, mesh.convex_hull), cfg.max_part_vertices)]
    result = coacd.run_coacd(
        coacd.Mesh(np.asarray(mesh.vertices, float), np.asarray(mesh.faces, np.int64)),
        threshold=cfg.coacd_threshold,
        max_convex_hull=cfg.max_convex_parts,
        seed=cfg.seed,
    )
    parts = [trimesh.Trimesh(np.asarray(v), np.asarray(f), process=True) for v, f in result]
    return [_limit_vertices(p, cfg.max_part_vertices) for p in parts]


def collision_geometry(mesh: trimesh.Trimesh, cfg: CollisionConfig) -> CollisionResult:
    """Convex collision parts for the repaired ``mesh``, with their checks."""
    steps: dict[str, int] = {}
    body = sealed_body(mesh, cfg.sealing)
    prim = primitive_for(mesh, cfg, body)
    if prim is not None:
        kind, part, score = prim
        checks = check_parts([part], mesh, cfg, body=body)
        if all(checks.values()):
            return CollisionResult([part], kind, checks, score, steps)
        steps["primitive_rejected"] = 1
    parts = _decompose(mesh, cfg, steps)
    kind = (
        "convex_hull"
        if steps.get("coacd_unavailable") or len(parts) == 1
        else "convex_decomposition"
    )
    return CollisionResult(parts, kind, check_parts(parts, mesh, cfg, body=body), None, steps)


def write_collision(result: CollisionResult, out_dir: Path, stem: str) -> list[Path]:
    """Write each part as ``<stem>_collision_<i>.obj`` and return the paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, part in enumerate(result.parts):
        path = out_dir / f"{stem}_collision_{i}.obj"
        part.export(path)
        paths.append(path)
    return paths
