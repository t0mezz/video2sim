"""Mass and inertia for the physics stage (P4.5).

Input is a repaired, metric mesh in the object frame (``stages/mesh_repair.py``) and the material
of P4.4. The signed volume of that mesh cannot be used: most SAM 3D meshes are thin shells, a
median 0.13 of the true volume (``docs/research/mesh_volume.md``). So the body is a voxel fill:

* Surface samples are voxelized, gaps up to ``seal_radius`` of the longest side are closed by
  a dilation, the outside is flooded, and the same distance is taken back. A crack in a box seals
  and the box becomes solid. The mouth of a cup stays open until the radius reaches about half its
  width, so the cup keeps its cavity. A vessel with a mouth under about 8% of the longest side is
  sealed as a solid.
* A hollow answer scales the body: ``V_eff = f * V_sealed``, where ``f`` is
  ``hollow_fill_fraction`` of the material table. It applies to closed objects only. A body under
  ``open_vessel_hull_ratio`` of its convex hull has already lost its cavity, and a second
  correction would count it twice. Concave solids fall under the same test and keep ``f = 1``.
* A solid answer keeps ``f = 1``.

Low and high multiply the density range by the range of ``f``, and leave out the scale error.
``f`` dilutes the body evenly rather than moving mass to a wall, so the inertia of an empty box
reads low. Inertia is at nominal mass, about the center of mass of the body. That center is not
the object-frame origin, which is the mesh centroid: ``center_of_mass_m`` carries the offset for
P4.8, which moves the origin there so that ``RegisteredObject``'s convention holds in the export.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import trimesh
from numpy.typing import NDArray

from video2sim.schemas import Estimate
from video2sim.stages.physics import ObjectMaterial

F64 = NDArray[np.float64]


@dataclass(frozen=True)
class MassConfig:
    """Values from ``physics.mass`` in ``configs/default.yaml``."""

    seal_radius: float
    open_vessel_hull_ratio: float
    grid: int
    samples_per_voxel_area: float
    max_surface_samples: int
    seed: int

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> MassConfig:
        c = cfg["physics"]["mass"]
        out = cls(
            seal_radius=float(c["seal_radius"]),
            open_vessel_hull_ratio=float(c["open_vessel_hull_ratio"]),
            grid=int(c["grid"]),
            samples_per_voxel_area=float(c["samples_per_voxel_area"]),
            max_surface_samples=int(c["max_surface_samples"]),
            seed=int(c["seed"]),
        )
        if not 0 < out.seal_radius < 0.5:
            raise ValueError("physics.mass.seal_radius must be in (0, 0.5)")
        if not 0 < out.open_vessel_hull_ratio < 1:
            raise ValueError("physics.mass.open_vessel_hull_ratio must be in (0, 1)")
        if out.grid < 8:
            raise ValueError("physics.mass.grid must be at least 8")
        if out.samples_per_voxel_area <= 0 or out.max_surface_samples <= 0:
            raise ValueError("physics.mass surface sampling must be positive")
        return out


@dataclass(frozen=True)
class SealedBody:
    """The voxels of the sealed fill. Cell ``i`` spans ``origin_m + i * pitch_m`` by one pitch."""

    voxels: NDArray[np.bool_]
    origin_m: F64
    pitch_m: float

    @property
    def volume_m3(self) -> float:
        return float(self.voxels.sum()) * self.pitch_m**3


@dataclass(frozen=True)
class MassEstimate:
    """Mass and inertia of one object. ``PhysicalProperties`` takes the first two."""

    mass_kg: Estimate
    inertia_kg_m2: F64
    center_of_mass_m: F64  # in the object frame; P4.8 moves the origin here
    volume_sealed_m3: float
    volume_eff_m3: float
    fill_fraction: Estimate
    is_open: bool
    reasons: tuple[str, ...]


def sealed_body(mesh: trimesh.Trimesh, cfg: MassConfig) -> SealedBody:
    """Fill the outer surface of ``mesh`` after closing gaps up to ``cfg.seal_radius``."""
    from scipy import ndimage  # type: ignore[import-untyped]  # heavy import

    pitch = float(mesh.extents.max()) / cfg.grid
    radius = max(cfg.seal_radius * cfg.grid, 1.0)  # voxels
    pad = int(radius) + 4
    count = int(mesh.area / pitch**2 * cfg.samples_per_voxel_area)
    points = trimesh.sample.sample_surface(
        mesh, min(count, cfg.max_surface_samples), seed=cfg.seed
    )[0]
    origin = np.asarray(mesh.bounds[0], dtype=float) - pad * pitch
    index = np.floor((points - origin) / pitch).astype(int)
    outside_surface: NDArray[np.bool_] = np.ones(tuple(index.max(axis=0) + pad + 1), dtype=bool)
    outside_surface[index[:, 0], index[:, 1], index[:, 2]] = False
    filled = ndimage.binary_fill_holes(ndimage.distance_transform_edt(outside_surface) <= radius)
    solid: NDArray[np.bool_] = ndimage.distance_transform_edt(filled) > radius
    return SealedBody(solid, origin, pitch)


def body_inertia(body: SealedBody, density_kg_m3: float) -> tuple[F64, F64]:
    """Center of mass and the inertia tensor about it, in the frame of the body.

    Each voxel adds its point mass term and its own inertia as a cube of side ``p``, which is
    ``m p^2 / 6`` about each axis.
    """
    centres = (np.argwhere(body.voxels) + 0.5) * body.pitch_m + body.origin_m
    com = centres.mean(axis=0)
    r = centres - com
    voxel_mass = density_kg_m3 * body.pitch_m**3
    second_moment = r.T @ r
    inertia = np.trace(second_moment) * np.eye(3) - second_moment
    inertia += len(r) * body.pitch_m**2 / 6 * np.eye(3)
    return com, voxel_mass * inertia


def estimate_mass(
    mesh: trimesh.Trimesh,
    material: ObjectMaterial,
    hollow_fill_fraction: Estimate,
    cfg: MassConfig,
) -> MassEstimate:
    """``m = rho * V_eff`` and the inertia of the sealed body at that density (AGENTS.md)."""
    body = sealed_body(mesh, cfg)
    volume = body.volume_m3
    hull_ratio = volume / float(mesh.convex_hull.volume)
    is_open = hull_ratio < cfg.open_vessel_hull_ratio
    reasons = []
    fill = Estimate(low=1.0, nominal=1.0, high=1.0)
    if material.is_hollow and is_open:
        reasons.append(f"sealed body is {hull_ratio:.2f} of the hull: read as open, no correction")
    elif material.is_hollow:
        fill = hollow_fill_fraction
    rho = material.numbers.density_kg_m3
    mass = Estimate(
        low=rho.low * fill.low * volume,
        nominal=rho.nominal * fill.nominal * volume,
        high=rho.high * fill.high * volume,
    )
    com, inertia = body_inertia(body, rho.nominal * fill.nominal)
    return MassEstimate(
        mass_kg=mass,
        inertia_kg_m2=inertia,
        center_of_mass_m=com,
        volume_sealed_m3=volume,
        volume_eff_m3=fill.nominal * volume,
        fill_fraction=fill,
        is_open=is_open,
        reasons=tuple(reasons),
    )


__all__ = [
    "MassConfig",
    "MassEstimate",
    "SealedBody",
    "body_inertia",
    "estimate_mass",
    "sealed_body",
]
