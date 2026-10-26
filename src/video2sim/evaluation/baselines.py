"""Baselines for the evaluation harness (P0.6).

* ``oracle`` copies the ground truth and adds seeded noise. Zero noise must score perfectly
  and known noise must give a known error, so it tests the metrics. It is not a floor.
* ``constant_box`` knows nothing about the scene: one fixed box on the board. It is the
  floor any real method has to beat. The mask-based baseline replaces it in a later
  iteration (see PLAN.md).
"""

from __future__ import annotations

import zlib

import numpy as np

from video2sim.evaluation.geometry import box_vertices
from video2sim.evaluation.scenes import GtScene, Prediction, PredObject

# Used only to give the constant baseline a mass: a light, roughly wood-like guess.
CONSTANT_DENSITY_KG_M3 = 500.0


def oracle(
    gt: GtScene,
    seed: int = 0,
    size_sigma: float = 0.0,
    position_sigma_m: float = 0.0,
    yaw_sigma_deg: float = 0.0,
    mass_sigma: float = 0.0,
    drop_fraction: float = 0.0,
) -> Prediction:
    """Ground truth with Gaussian noise: relative on size and mass, absolute on pose."""
    rng = np.random.default_rng((seed, zlib.crc32(gt.scene_id.encode())))
    out: list[PredObject] = []
    for obj in gt.objects:
        if obj.size_m is None or obj.position_m is None:
            continue
        if rng.random() < drop_fraction:
            continue
        size = tuple(float(s * (1 + rng.normal(0, size_sigma))) for s in obj.size_m)
        center = tuple(float(c + rng.normal(0, position_sigma_m)) for c in obj.position_m)
        yaw = (obj.yaw_deg or 0.0) + float(rng.normal(0, yaw_sigma_deg))
        mass = None if obj.mass_kg is None else obj.mass_kg * (1 + float(rng.normal(0, mass_sigma)))
        verts = box_vertices((size[0], size[1], size[2]), (center[0], center[1], center[2]), yaw)
        out.append(PredObject(obj.object_id, verts, mass))
    return Prediction(out, notes=["oracle baseline: ground truth plus noise"])


def constant_box(
    center_xy_m: tuple[float, float] = (0.25, 0.25),
    size_m: tuple[float, float, float] = (0.08, 0.06, 0.06),
) -> Prediction:
    """One fixed box in the middle of the board, for every scene."""
    center = (center_xy_m[0], center_xy_m[1], size_m[2] / 2)
    mass = CONSTANT_DENSITY_KG_M3 * size_m[0] * size_m[1] * size_m[2]
    return Prediction(
        [PredObject("const_0", box_vertices(size_m, center, 0.0), mass)],
        notes=["constant-box floor: no information about the scene"],
    )
