"""Mesh repair and validators for the physics stage (P4.1).

A reconstructed mesh is not yet a rigid body: it may carry floaters, duplicate or degenerate
faces, flipped faces and holes. ``repair_mesh`` cleans one mesh in a fixed order and reports what
it did; ``validate_mesh`` returns the named checks that end up in ``SimAsset.checks``. Export is
blocked unless every check passes, so a mesh that cannot be repaired is reported, not patched
silently.

* Floaters are connected components whose surface area is under ``min_component_area_frac`` of
  the largest component. Components are found on shared vertices after merging.
* Holes are closed by a fan from the loop's centroid. Only simple loops (each boundary vertex has
  one outgoing boundary edge) are closed; anything else leaves the mesh open and the
  ``watertight`` check false. ``trimesh.repair.fill_holes`` is not used because it needs
  ``networkx``.
* The checks are ``finite`` (no NaN or infinity), ``watertight`` (every edge in two faces),
  ``manifold`` (also consistent winding, so the faces form one oriented surface) and
  ``positive_volume`` (outward normals; the signed volume is above ``min_volume_m3``).

Decimation to a triangle budget (the last part of P4.1) is not here: it needs a quadric
simplifier that is not a dependency yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import trimesh


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


def _close_holes(m: trimesh.Trimesh, steps: dict[str, int]) -> trimesh.Trimesh:
    """Orient the faces consistently, close simple boundary loops, orient outward."""
    trimesh.repair.fix_winding(m)  # type: ignore[no-untyped-call]
    edges = m.edges_sorted
    _uniq, inverse, counts = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    boundary_edges = np.flatnonzero(counts[inverse] == 1)
    nxt: dict[int, int] = {}
    simple = True
    for e in boundary_edges:
        f, k = divmod(int(e), 3)
        a, b = int(m.faces[f][k]), int(m.faces[f][(k + 1) % 3])
        if a in nxt:
            simple = False
            break
        nxt[a] = b
    new_faces: list[list[int]] = []
    verts = [np.asarray(v) for v in m.vertices]
    if simple and nxt:
        seen: set[int] = set()
        for start in list(nxt):
            if start in seen:
                continue
            loop = [start]
            seen.add(start)
            cur = nxt.get(start)
            while cur is not None and cur != start and cur not in seen:
                loop.append(cur)
                seen.add(cur)
                cur = nxt.get(cur)
            if cur != start or len(loop) < 3:
                continue
            c = len(verts)
            verts.append(np.mean([verts[i] for i in loop], axis=0))
            # the boundary runs a->b in an existing face, so the patch runs b->a
            new_faces += [[loop[(i + 1) % len(loop)], loop[i], c] for i in range(len(loop))]
    steps["faces_added"] = len(new_faces)
    if new_faces:
        m = trimesh.Trimesh(np.asarray(verts), np.vstack([m.faces, new_faces]), process=False)
    trimesh.repair.fix_normals(m)  # type: ignore[no-untyped-call]
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
    if m.is_watertight and m.volume < 0:
        m.invert()
    return RepairResult(m, validate_mesh(m, cfg), steps)
