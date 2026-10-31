from __future__ import annotations

import math
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import trimesh
import yaml

from video2sim.schemas import Estimate, PhysicalProperties
from video2sim.stages.mass import MassConfig, SealedBody, body_inertia, estimate_mass, sealed_body
from video2sim.stages.material_table import MaterialNumbers, load_material_table
from video2sim.stages.physics import ObjectMaterial

ROOT = Path(__file__).resolve().parent.parent
FILL = Estimate(low=0.1, nominal=0.2, high=0.5)


@pytest.fixture(scope="module")
def cfg() -> MassConfig:
    raw = yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    return MassConfig.from_config(raw)


def _material(is_hollow: bool = False) -> ObjectMaterial:
    rho = Estimate(low=800, nominal=1000, high=1200)
    span = Estimate(low=0.1, nominal=0.5, high=0.9)
    numbers = MaterialNumbers("plastic_hard", rho, span, span, (), False)
    return ObjectMaterial(numbers, is_hollow, 0.9, (), ())


def _shell(outer: trimesh.Trimesh, wall_m: float) -> trimesh.Trimesh:
    """A closed thin shell: the outer box and an inward-facing inner box, as SAM 3D gives."""
    inner = trimesh.creation.box(outer.extents - 2 * wall_m)
    inner.invert()
    return cast(trimesh.Trimesh, trimesh.util.concatenate([outer, inner]))


def _cup(radius_m: float, height_m: float, wall_m: float) -> trimesh.Trimesh:
    """An open-topped cup, closed as a surface: outer and inner wall joined at the rim."""
    profile = np.array(
        [
            [0.0, 0.0],
            [radius_m, 0.0],
            [radius_m, height_m],
            [radius_m - wall_m, height_m],
            [radius_m - wall_m, wall_m],
            [0.0, wall_m],
        ]
    )
    return trimesh.creation.revolve(profile, sections=64)


def test_solid_box_matches_closed_forms(cfg: MassConfig) -> None:
    a, b, c = 0.2, 0.1, 0.06
    est = estimate_mass(trimesh.creation.box((a, b, c)), _material(), FILL, cfg)
    m = 1000 * a * b * c
    assert est.fill_fraction.nominal == 1.0
    assert est.mass_kg.nominal == pytest.approx(m, rel=0.08)  # the fill reads high, see the note
    i = est.inertia_kg_m2 / est.mass_kg.nominal
    assert i[0, 0] == pytest.approx((b**2 + c**2) / 12, rel=0.06)
    assert i[1, 1] == pytest.approx((a**2 + c**2) / 12, rel=0.06)
    assert i[2, 2] == pytest.approx((a**2 + b**2) / 12, rel=0.06)
    assert np.abs(est.center_of_mass_m).max() < 0.002


def test_solid_cylinder_and_sphere_match_closed_forms(cfg: MassConfig) -> None:
    r, h = 0.04, 0.12
    cyl = estimate_mass(trimesh.creation.cylinder(r, h, sections=64), _material(), FILL, cfg)
    assert cyl.volume_sealed_m3 == pytest.approx(math.pi * r**2 * h, rel=0.08)
    i = cyl.inertia_kg_m2 / cyl.mass_kg.nominal
    assert i[2, 2] == pytest.approx(r**2 / 2, rel=0.06)
    assert i[0, 0] == pytest.approx((3 * r**2 + h**2) / 12, rel=0.06)

    ball = estimate_mass(trimesh.creation.icosphere(4, 0.05), _material(), FILL, cfg)
    assert ball.volume_sealed_m3 == pytest.approx(4 / 3 * math.pi * 0.05**3, rel=0.08)
    assert ball.inertia_kg_m2[0, 0] / ball.mass_kg.nominal == pytest.approx(
        2 / 5 * 0.05**2, rel=0.06
    )


def test_inertia_of_a_voxel_block_is_exact() -> None:
    voxels = np.zeros((6, 5, 4), dtype=bool)
    voxels[1:5, 1:4, 1:3] = True  # 4 x 3 x 2 cells of 1 cm
    p = 0.01
    com, inertia = body_inertia(SealedBody(voxels, np.zeros(3), p), 1000.0)
    a, b, c = 4 * p, 3 * p, 2 * p
    m = 1000 * a * b * c
    assert com == pytest.approx([3 * p, 2.5 * p, 2 * p])
    assert np.diag(inertia) == pytest.approx(
        [m * (b**2 + c**2) / 12, m * (a**2 + c**2) / 12, m * (a**2 + b**2) / 12]
    )


def test_inertia_scales_with_the_fifth_power(cfg: MassConfig) -> None:
    box = trimesh.creation.box((0.2, 0.1, 0.06))
    small = estimate_mass(box, _material(), FILL, cfg)
    big_mesh = box.copy()
    big_mesh.apply_scale(2.0)
    big = estimate_mass(big_mesh, _material(), FILL, cfg)
    assert big.mass_kg.nominal / small.mass_kg.nominal == pytest.approx(8, rel=1e-6)
    assert big.inertia_kg_m2[0, 0] / small.inertia_kg_m2[0, 0] == pytest.approx(32, rel=1e-6)


def test_thin_shell_counts_as_its_outer_volume(cfg: MassConfig) -> None:
    outer = trimesh.creation.box((0.2, 0.1, 0.06))
    shell = _shell(outer, 0.001)
    assert abs(shell.volume) < 0.1 * outer.volume
    est = estimate_mass(shell, _material(), FILL, cfg)
    assert est.volume_sealed_m3 == pytest.approx(outer.volume, rel=0.08)


def test_hollow_closed_object_scales_mass_and_inertia_by_fill(cfg: MassConfig) -> None:
    box = trimesh.creation.box((0.2, 0.1, 0.06))
    solid = estimate_mass(box, _material(), FILL, cfg)
    hollow = estimate_mass(box, _material(is_hollow=True), FILL, cfg)
    assert not hollow.is_open and hollow.fill_fraction == FILL
    assert hollow.mass_kg.nominal == pytest.approx(0.2 * solid.mass_kg.nominal)
    assert hollow.mass_kg.low == pytest.approx(800 * 0.1 * hollow.volume_sealed_m3)
    assert hollow.mass_kg.high == pytest.approx(1200 * 0.5 * hollow.volume_sealed_m3)
    assert hollow.inertia_kg_m2 == pytest.approx(0.2 * solid.inertia_kg_m2)
    assert hollow.volume_eff_m3 == pytest.approx(0.2 * hollow.volume_sealed_m3)


def test_open_cup_keeps_its_cavity_and_gets_no_second_correction(cfg: MassConfig) -> None:
    r, h, wall = 0.04, 0.10, 0.003
    cup = _cup(r, h, wall)
    assert cup.is_watertight
    est = estimate_mass(cup, _material(is_hollow=True), FILL, cfg)
    assert est.is_open
    assert est.fill_fraction.nominal == 1.0
    assert est.reasons and "open" in est.reasons[0]
    assert est.volume_sealed_m3 < 0.5 * math.pi * r**2 * h
    # The material sits low because of the floor, so the center of mass is below the mesh middle.
    assert est.center_of_mass_m[2] < h / 2


def test_crack_in_a_shell_is_sealed(cfg: MassConfig) -> None:
    outer = trimesh.creation.box((0.2, 0.2, 0.2))
    fine = outer.subdivide_to_size(0.005)
    centres = fine.triangles_center
    # A slot across the top, about 7% of the side wide, under the 8% that a 4% radius closes.
    slot = (np.abs(centres[:, 0]) < 0.005) & (centres[:, 2] > 0.09)
    assert slot.any()
    fine.update_faces(~slot)
    assert not fine.is_watertight
    est = estimate_mass(fine, _material(), FILL, cfg)
    assert est.volume_sealed_m3 == pytest.approx(0.008, rel=0.1)


def test_output_is_accepted_as_physical_properties(cfg: MassConfig) -> None:
    est = estimate_mass(trimesh.creation.box((0.2, 0.1, 0.002)), _material(), FILL, cfg)
    span = Estimate(low=0.1, nominal=0.5, high=0.9)
    PhysicalProperties(
        material_class="plastic_hard",
        is_hollow=False,
        mass_kg=est.mass_kg,
        density_kg_m3=_material().numbers.density_kg_m3,
        friction_coefficient=span,
        restitution=span,
        inertia_kg_m2=est.inertia_kg_m2.tolist(),
        confidence=0.9,
    )


def test_same_seed_gives_the_same_body(cfg: MassConfig) -> None:
    box = trimesh.creation.box((0.2, 0.1, 0.06))
    assert np.array_equal(sealed_body(box, cfg).voxels, sealed_body(box, cfg).voxels)


@pytest.mark.parametrize(
    "bad",
    [
        {"seal_radius": 0.0},
        {"open_vessel_hull_ratio": 1.0},
        {"grid": 4},
        {"max_surface_samples": 0},
    ],
)
def test_bad_config_is_refused(bad: dict[str, float]) -> None:
    raw = yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    raw["physics"]["mass"] |= bad
    with pytest.raises(ValueError, match=r"physics\.mass"):
        MassConfig.from_config(raw)


def test_table_fill_fraction_is_a_share() -> None:
    raw = yaml.safe_load((ROOT / "configs/material_table.yaml").read_text(encoding="utf-8"))
    fill = load_material_table(ROOT / "configs/material_table.yaml").hollow_fill_fraction
    assert 0 < fill.low <= fill.nominal <= fill.high <= 1
    assert raw["hollow_fill_fraction"]["nominal"] == fill.nominal
