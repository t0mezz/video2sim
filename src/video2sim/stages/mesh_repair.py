"""Mesh repair and validators for the physics stage (P4.1).

A reconstructed mesh is not yet a rigid body: it may carry floaters, duplicate or degenerate
faces, flipped faces and holes. ``repair_mesh`` cleans one mesh in a fixed order and reports what
it did; ``validate_mesh`` returns the named checks that end up in ``SimAsset.checks``. Export is
blocked unless every check passes, so a mesh that cannot be repaired is reported, not patched
silently.

* Floaters are connected components whose surface area is under ``min_component_area_frac`` of
  the largest component. Components are found on shared vertices after merging.
* Holes are closed by a fan from each boundary loop's centroid. A boundary that is not a closed
  loop leaves the mesh open and the ``watertight`` check false. ``trimesh.repair.fill_holes`` is
  not used because it needs ``networkx``.
* The checks are ``finite`` (no NaN or infinity), ``watertight`` (every edge in two faces),
  ``manifold`` (also consistent winding, so the faces form one oriented surface) and
  ``positive_volume`` (outward normals; the signed volume is above ``min_volume_m3``).

The repaired mesh is not decimated: mass, inertia and collision geometry are computed from it at
full detail (D-033). ``visual_mesh`` makes the light copy that is rendered:

* A mesh above ``physics.visual.max_triangles`` is decimated by quadric edge collapse
  (``fast_simplification``). Collapse can leave edges shared by three or more faces; those faces
  are dropped and the hole closed again. A copy is kept if it is still closed and consistently
  wound and its surface stays within ``max_surface_error``: the 95th percentile of the surface
  distances in both directions, as a share of the bounding box diagonal. Volume is not the test,
  because the visual mesh does not carry the mass; collapsing a ridged surface (a Rubik's cube in
  s12) moves the volume by 6% while moving the surface by about half a millimeter.
* If no copy at the budget keeps the shape, the larger budgets of ``fallback_triangles`` are
  tried in turn. A copy over the budget is shipped and reported (``within_triangle_budget`` in
  ``VisualResult.advisory``), not blocked, because a heavy object in the scene is better than a
  missing one. The last budget is the ceiling: if no copy passes there either, the full mesh
  stands and the ``within_triangle_ceiling`` check is false, so export is blocked. Without
  ``fast_simplification`` an over-budget mesh is handled the same way, with
  ``steps["decimation_unavailable"]`` set.

``repair_file`` is the same on a mesh file: read, repair, and write the result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from numpy.typing import NDArray

# Fractions of each budget that decimation aims for, in turn, until one result passes the checks.
DECIMATION_TARGET_MARGINS = (0.98, 0.95, 0.9)
# Surface points drawn on each mesh for the surface error, and candidate faces per point.
SURFACE_SAMPLES = 20000
NEAREST_FACES = 16


@dataclass(frozen=True)
class RepairConfig:
    """Values from ``physics.repair`` in ``configs/default.yaml``."""

    min_component_area_frac: float
    min_volume_m3: float

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> RepairConfig:
        c = cfg["physics"]["repair"]
        out = cls(
            min_component_area_frac=float(c["min_component_area_frac"]),
            min_volume_m3=float(c["min_volume_m3"]),
        )
        if not 0 <= out.min_component_area_frac < 1:
            raise ValueError("physics.repair.min_component_area_frac must be in [0, 1)")
        if out.min_volume_m3 <= 0:
            raise ValueError("physics.repair.min_volume_m3 must be positive")
        return out


@dataclass(frozen=True)
class VisualConfig:
    """Values from ``physics.visual`` in ``configs/default.yaml``."""

    max_triangles: int
    fallback_triangles: tuple[int, ...]
    max_surface_error: float
    decimation_aggression: int
    seed: int = 0

    @property
    def budgets(self) -> tuple[int, ...]:
        """The budgets tried in turn; the last is the ceiling."""
        return (self.max_triangles, *self.fallback_triangles)

    @property
    def ceiling(self) -> int:
        return self.budgets[-1]

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> VisualConfig:
        c = cfg["physics"]["visual"]
        out = cls(
            max_triangles=int(c["max_triangles"]),
            fallback_triangles=tuple(int(n) for n in c.get("fallback_triangles") or ()),
            max_surface_error=float(c["max_surface_error"]),
            decimation_aggression=int(c["decimation_aggression"]),
            seed=int(c.get("seed", 0)),
        )
        if out.max_triangles < 4:
            raise ValueError("physics.visual.max_triangles must be at least 4")
        if any(b <= a for a, b in zip(out.budgets[:-1], out.budgets[1:], strict=True)):
            raise ValueError("physics.visual.fallback_triangles must rise above max_triangles")
        if not 0 < out.max_surface_error < 1:
            raise ValueError("physics.visual.max_surface_error must be in (0, 1)")
        if not 3 <= out.decimation_aggression <= 10:
            raise ValueError("physics.visual.decimation_aggression must be in [3, 10]")
        return out


@dataclass(frozen=True)
class RepairResult:
    mesh: trimesh.Trimesh
    checks: dict[str, bool]
    steps: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return all(self.checks.values())


@dataclass(frozen=True)
class VisualResult:
    """The visual copy. ``checks`` block export; ``advisory`` is reported and flagged only."""

    mesh: trimesh.Trimesh
    checks: dict[str, bool]
    advisory: dict[str, bool]
    surface_error: float | None  # against the repaired mesh; None if it was not decimated
    steps: dict[str, int] = field(default_factory=dict)


def validate_mesh(mesh: trimesh.Trimesh, cfg: RepairConfig) -> dict[str, bool]:
    """Named pass or fail checks for one mesh."""
    finite = bool(len(mesh.vertices)) and bool(np.isfinite(mesh.vertices).all())
    if not finite or len(mesh.faces) == 0:
        return {
            "finite": finite,
            "watertight": False,
            "manifold": False,
            "positive_volume": False,
        }
    watertight = bool(mesh.is_watertight)
    return {
        "finite": True,
        "watertight": watertight,
        "manifold": watertight and bool(mesh.is_winding_consistent),
        "positive_volume": watertight and float(mesh.volume) > cfg.min_volume_m3,
    }


def _drop_floaters(m: trimesh.Trimesh, min_frac: float) -> tuple[trimesh.Trimesh, int]:
    """Keep faces of the components with at least ``min_frac`` of the largest one's area.

    Done with scipy: ``Trimesh.split`` fills holes through ``networkx``, which is not installed.
    """
    from scipy.sparse import coo_matrix  # type: ignore[import-untyped]  # heavy import
    from scipy.sparse.csgraph import connected_components  # type: ignore[import-untyped]

    n = len(m.vertices)
    f = m.faces
    graph = coo_matrix(
        (np.ones(3 * len(f)), (f.reshape(-1), np.roll(f, -1, axis=1).reshape(-1))), shape=(n, n)
    )
    count, label = connected_components(graph, directed=False)
    face_label = label[f[:, 0]]
    area = np.bincount(face_label, weights=m.area_faces, minlength=count)
    keep = area >= min_frac * area.max()
    if keep.all():
        return m, 0
    kept = trimesh.Trimesh(m.vertices, f[keep[face_label]], process=False)
    kept.remove_unreferenced_vertices()
    return kept, int((~keep & (area > 0)).sum())


def _orient(m: trimesh.Trimesh) -> None:
    """Make winding consistent and outward. ``trimesh`` needs ``networkx`` when it has work to do;
    without it the mesh is left as is and the ``manifold`` check reports it."""
    try:
        trimesh.repair.fix_winding(m)  # type: ignore[no-untyped-call]
        trimesh.repair.fix_normals(m)  # type: ignore[no-untyped-call]
    except ModuleNotFoundError:
        pass


def _close_holes(m: trimesh.Trimesh, steps: dict[str, int]) -> trimesh.Trimesh:
    """Orient the faces consistently, close every boundary loop with a fan, orient outward.

    A vertex may have several boundary edges (where faces were removed around it); the walk takes
    them in turn, so each loop it finds is closed.
    """
    _orient(m)
    _, inverse, counts = np.unique(m.edges_sorted, axis=0, return_inverse=True, return_counts=True)
    boundary_edges = np.flatnonzero(counts[inverse.reshape(-1)] == 1)
    nxt: dict[int, list[int]] = {}
    for e in boundary_edges:
        f, k = divmod(int(e), 3)
        nxt.setdefault(int(m.faces[f][k]), []).append(int(m.faces[f][(k + 1) % 3]))
    new_faces: list[list[int]] = []
    verts = [np.asarray(v) for v in m.vertices]

    def fan(loop: list[int]) -> None:
        c = len(verts)
        verts.append(np.mean([verts[i] for i in loop], axis=0))
        # the boundary runs a->b in an existing face, so the patch runs b->a
        new_faces.extend([loop[(i + 1) % len(loop)], loop[i], c] for i in range(len(loop)))

    for start in list(nxt):
        while nxt[start]:
            # a vertex visited twice splits the walk into simple loops: a fan over a loop that
            # touches itself would put an edge in four faces
            path = [start]
            where = {start: 0}
            cur = nxt[start].pop()
            while cur != start:
                if cur in where:
                    cycle = path[where[cur] :]
                    del path[where[cur] :]
                    for v in cycle:
                        where.pop(v)
                    if len(cycle) >= 3:
                        fan(cycle)
                    # cur stays on the path: the walk continues from it
                where[cur] = len(path)
                path.append(cur)
                if not nxt.get(cur):
                    path = []  # an open chain: not a loop, leave it
                    break
                cur = nxt[cur].pop()
            if len(path) >= 3:
                fan(path)
    steps["faces_added"] = len(new_faces)
    if new_faces:
        m = trimesh.Trimesh(np.asarray(verts), np.vstack([m.faces, new_faces]), process=False)
        _orient(m)
    return m


def _drop_nonmanifold_faces(m: trimesh.Trimesh) -> trimesh.Trimesh:
    """Remove every face on an edge shared by more than two faces; the hole is closed later."""
    _, inverse, counts = np.unique(m.edges_sorted, axis=0, return_inverse=True, return_counts=True)
    bad_edge = counts[inverse.reshape(-1)] > 2
    if not bad_edge.any():
        return m
    bad_face = np.zeros(len(m.faces), dtype=bool)
    bad_face[np.flatnonzero(bad_edge) // 3] = True
    out = trimesh.Trimesh(m.vertices, m.faces[~bad_face], process=False)
    out.remove_unreferenced_vertices()
    return out


def _distance_to_surface(points: NDArray[np.float64], mesh: trimesh.Trimesh) -> NDArray[np.float64]:
    """Distance from each point to the nearest of ``mesh``'s faces near it.

    The candidates are the faces with the ``NEAREST_FACES`` nearest centroids, so no ``rtree`` is
    needed. On a sliver far longer than its neighbors this can overestimate, never underestimate.
    """
    from scipy.spatial import cKDTree  # type: ignore[import-untyped]  # heavy import

    k = min(NEAREST_FACES, len(mesh.faces))
    idx = cKDTree(np.asarray(mesh.triangles_center)).query(points, k=k)[1].reshape(len(points), k)
    tri = np.asarray(mesh.triangles)[idx.reshape(-1)]
    closest = trimesh.triangles.closest_point(  # type: ignore[no-untyped-call]  # trimesh is untyped
        tri, np.repeat(points, k, axis=0)
    )
    d = np.linalg.norm(closest - np.repeat(points, k, axis=0), axis=1).reshape(len(points), k)
    return np.asarray(d.min(axis=1), dtype=np.float64)


def surface_error(mesh: trimesh.Trimesh, other: trimesh.Trimesh, seed: int = 0) -> float:
    """95th percentile of the point to surface distances in both directions, over ``mesh``'s
    bounding box diagonal."""
    a = np.asarray(trimesh.sample.sample_surface(mesh, SURFACE_SAMPLES, seed=seed)[0])
    b = np.asarray(trimesh.sample.sample_surface(other, SURFACE_SAMPLES, seed=seed)[0])
    d = np.concatenate([_distance_to_surface(a, other), _distance_to_surface(b, mesh)])
    return float(np.percentile(d, 95)) / float(np.linalg.norm(mesh.extents))


def _collapse(m: trimesh.Trimesh, target: int, aggression: int) -> trimesh.Trimesh:
    import fast_simplification  # type: ignore[import-untyped,unused-ignore]  # heavy import

    v, f = fast_simplification.simplify(
        np.asarray(m.vertices, dtype=np.float64),
        np.asarray(m.faces, dtype=np.int32),
        target_count=target,
        agg=aggression,
    )
    small = trimesh.Trimesh(v, f, process=False)
    small.merge_vertices()
    small.update_faces(small.nondegenerate_faces())
    small.remove_unreferenced_vertices()
    return _close_holes(_drop_nonmanifold_faces(small), {})


def visual_mesh(mesh: trimesh.Trimesh, cfg: VisualConfig) -> VisualResult:
    """The copy of the repaired ``mesh`` that is rendered, decimated as the module says.

    ``mesh`` is not modified; it is itself the result when it is within the budget, open, or
    cannot be decimated within ``cfg.max_surface_error`` up to the ceiling.
    """
    steps: dict[str, int] = {}
    chosen, error = mesh, None
    if len(mesh.faces) > cfg.max_triangles and mesh.is_watertight:
        try:
            chosen, error = _decimate(mesh, cfg, steps)
        except ImportError:
            steps["decimation_unavailable"] = 1
    faces = len(chosen.faces)
    return VisualResult(
        chosen,
        {"within_triangle_ceiling": faces <= cfg.ceiling},
        {"within_triangle_budget": faces <= cfg.max_triangles},
        error,
        steps,
    )


def _decimate(
    m: trimesh.Trimesh, cfg: VisualConfig, steps: dict[str, int]
) -> tuple[trimesh.Trimesh, float | None]:
    """The first copy that passes, trying each budget and each margin in turn; or ``m``.

    Raises ``ImportError`` without ``fast_simplification``.
    """
    for rung, budget in enumerate(cfg.budgets):
        if len(m.faces) <= budget:
            steps["budget_rung"] = rung
            return m, None
        for margin in DECIMATION_TARGET_MARGINS:
            # aim under the budget: closing the holes left by dropped faces adds a few back
            small = _collapse(m, int(budget * margin), cfg.decimation_aggression)
            if not (
                small.is_watertight
                and small.is_winding_consistent
                and len(small.faces) <= budget
                and float(small.volume) > 0
            ):
                continue
            error = surface_error(m, small, cfg.seed)
            if error <= cfg.max_surface_error:
                steps["budget_rung"] = rung
                steps["faces_removed_by_decimation"] = len(m.faces) - len(small.faces)
                return small, error
        steps[f"decimation_rejected_at_{budget}"] = 1
    steps["decimation_rejected"] = 1
    return m, None


def repair_mesh(mesh: trimesh.Trimesh, cfg: RepairConfig) -> RepairResult:
    """Clean ``mesh`` and return the result with its checks. The input is not modified.

    The result is at full detail; ``visual_mesh`` makes the light copy.
    """
    m = mesh.copy()
    steps: dict[str, int] = {}
    if len(m.faces) == 0 or not np.isfinite(m.vertices).all():
        return RepairResult(m, validate_mesh(m, cfg), steps)

    m.merge_vertices()
    before = len(m.faces)
    m.update_faces(m.nondegenerate_faces())
    m.update_faces(m.unique_faces())
    m.remove_unreferenced_vertices()
    steps["faces_dropped"] = before - len(m.faces)

    m, steps["floaters_removed"] = _drop_floaters(m, cfg.min_component_area_frac)
    m = _close_holes(m, steps)
    if m.is_watertight and m.volume < 0:
        m.invert()
    return RepairResult(m, validate_mesh(m, cfg), steps)


def repair_file(src: Path, dst: Path, cfg: RepairConfig) -> RepairResult:
    """Read the mesh at ``src``, repair it, and write the result to ``dst``.

    The file is written even when a check fails, so the failing mesh can be looked at; the caller
    decides on ``RepairResult.ok``. A file that is not a single mesh is an error.
    """
    loaded = trimesh.load(src, force="mesh", process=False)
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(f"{src} does not hold one triangle mesh")
    res = repair_mesh(loaded, cfg)
    dst.parent.mkdir(parents=True, exist_ok=True)
    res.mesh.export(dst)
    return res
