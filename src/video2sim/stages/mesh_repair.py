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

* A mesh above ``max_triangles`` is decimated by quadric edge collapse (``fast_simplification``)
  to the budget. Collapse can leave edges shared by three or more faces; those faces are dropped
  and the hole closed again. The result must pass the checks above and keep its volume within
  ``max_volume_change_frac``; otherwise the repaired, undecimated mesh stands and
  the ``within_triangle_budget`` check is false, so export is blocked rather than shipping a
  mesh that is too heavy or distorted. Without ``fast_simplification`` an over-budget mesh gets
  the same verdict, with ``steps["decimation_unavailable"]`` set.

``repair_file`` is the same on a mesh file: read, repair, and write the result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

# Fractions of the budget that decimation aims for, in turn, until one result passes the checks.
DECIMATION_TARGET_MARGINS = (0.98, 0.95, 0.9)


@dataclass(frozen=True)
class RepairConfig:
    """Values from ``physics.repair`` in ``configs/default.yaml``."""

    min_component_area_frac: float
    min_volume_m3: float
    max_triangles: int
    max_volume_change_frac: float
    decimation_aggression: int

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> RepairConfig:
        c = cfg["physics"]["repair"]
        out = cls(
            min_component_area_frac=float(c["min_component_area_frac"]),
            min_volume_m3=float(c["min_volume_m3"]),
            max_triangles=int(c["max_triangles"]),
            max_volume_change_frac=float(c["max_volume_change_frac"]),
            decimation_aggression=int(c["decimation_aggression"]),
        )
        if not 0 <= out.min_component_area_frac < 1:
            raise ValueError("physics.repair.min_component_area_frac must be in [0, 1)")
        if out.min_volume_m3 <= 0:
            raise ValueError("physics.repair.min_volume_m3 must be positive")
        if out.max_triangles < 4:
            raise ValueError("physics.repair.max_triangles must be at least 4")
        if not 0 < out.max_volume_change_frac < 1:
            raise ValueError("physics.repair.max_volume_change_frac must be in (0, 1)")
        if not 3 <= out.decimation_aggression <= 10:
            raise ValueError("physics.repair.decimation_aggression must be in [3, 10]")
        return out


@dataclass(frozen=True)
class RepairResult:
    mesh: trimesh.Trimesh
    checks: dict[str, bool]
    steps: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return all(self.checks.values())


def validate_mesh(mesh: trimesh.Trimesh, cfg: RepairConfig) -> dict[str, bool]:
    """Named pass or fail checks for one mesh."""
    finite = bool(len(mesh.vertices)) and bool(np.isfinite(mesh.vertices).all())
    if not finite or len(mesh.faces) == 0:
        return {
            "finite": finite,
            "watertight": False,
            "manifold": False,
            "positive_volume": False,
            "within_triangle_budget": len(mesh.faces) <= cfg.max_triangles,
        }
    watertight = bool(mesh.is_watertight)
    return {
        "finite": True,
        "watertight": watertight,
        "manifold": watertight and bool(mesh.is_winding_consistent),
        "positive_volume": watertight and float(mesh.volume) > cfg.min_volume_m3,
        "within_triangle_budget": len(mesh.faces) <= cfg.max_triangles,
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


def _decimate(m: trimesh.Trimesh, cfg: RepairConfig, steps: dict[str, int]) -> trimesh.Trimesh:
    """Collapse edges down to the triangle budget, or return ``m`` if that would not be safe.

    Edge collapse can leave a few edges shared by three or more faces. The faces on them are
    dropped and the hole is closed again; the result is kept only if it passes the checks.
    """
    if len(m.faces) <= cfg.max_triangles or not m.is_watertight:
        return m
    try:
        import fast_simplification  # type: ignore[import-untyped]  # heavy import
    except ImportError:
        steps["decimation_unavailable"] = 1
        return m
    before = float(m.volume)
    for margin in DECIMATION_TARGET_MARGINS:
        # aim under the budget: closing the holes left by dropped faces adds a few back
        v, f = fast_simplification.simplify(
            np.asarray(m.vertices, dtype=np.float64),
            np.asarray(m.faces, dtype=np.int32),
            target_count=int(cfg.max_triangles * margin),
            agg=cfg.decimation_aggression,
        )
        small = trimesh.Trimesh(v, f, process=False)
        small.merge_vertices()
        small.update_faces(small.nondegenerate_faces())
        small.remove_unreferenced_vertices()
        small = _close_holes(_drop_nonmanifold_faces(small), {})
        if (
            small.is_watertight
            and small.is_winding_consistent
            and len(small.faces) <= cfg.max_triangles
            and abs(float(small.volume) - before) <= cfg.max_volume_change_frac * before
        ):
            steps["faces_removed_by_decimation"] = len(m.faces) - len(small.faces)
            return small
    steps["decimation_rejected"] = 1
    return m


def repair_mesh(mesh: trimesh.Trimesh, cfg: RepairConfig) -> RepairResult:
    """Clean ``mesh`` and return the result with its checks. The input is not modified."""
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
    m = _decimate(m, cfg, steps)
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
