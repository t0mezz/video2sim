"""Settling against closed forms: rest, sliding on a slope, tipping, overlaps and units."""

from __future__ import annotations

from dataclasses import replace
from typing import cast

import numpy as np
import pytest
import trimesh
from numpy.typing import NDArray

from video2sim.schemas import Plane
from video2sim.stages.settling import SettleBody, SettlingConfig, is_round, settle_scene

F64 = NDArray[np.float64]

CFG = SettlingConfig(
    round_tolerance=0.04,
    timestep_s=0.001,
    gravity_m_s2=9.80665,
    penetration_tolerance_m=0.0005,
    max_separation_iterations=200,
    noslip_iterations=5,
    drift_sample_s=0.02,
    window_s=1.0,
    max_drift_m=0.02,
)
FLAT = Plane(normal=(0.0, 0.0, 1.0), offset_m=0.0)


def pose(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> F64:
    t = np.eye(4)
    t[:3, 3] = [x, y, z]
    return t


def box(
    name: str,
    t: F64,
    extents: tuple[float, float, float] = (0.1, 0.1, 0.1),
    mass_kg: float = 0.2,
    friction: float = 0.5,
) -> SettleBody:
    a, b, c = extents
    inertia = mass_kg / 12 * np.diag([b**2 + c**2, a**2 + c**2, a**2 + b**2])
    return SettleBody(
        name, [trimesh.creation.box(extents=extents)], t, mass_kg, inertia, np.zeros(3), friction
    )


def slope(angle_deg: float) -> Plane:
    a = np.radians(angle_deg)
    return Plane(normal=(float(np.sin(a)), 0.0, float(np.cos(a))), offset_m=0.0)


def on_slope(angle_deg: float, half_height_m: float = 0.05) -> F64:
    a = np.radians(angle_deg)
    t = np.eye(4)
    t[:3, :3] = [[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]
    t[:3, 3] = half_height_m * np.array([np.sin(a), 0, np.cos(a)])
    return t


def test_a_box_resting_on_the_table_does_not_move() -> None:
    result = settle_scene([box("a", pose(z=0.05))], FLAT, CFG)
    assert result.stable
    assert result.bodies[0].max_drift_m < 1e-3


def test_a_light_box_on_a_heavy_one_stays() -> None:
    bodies = [
        box("low", pose(z=0.05), mass_kg=1.0),
        box("top", pose(z=0.15), (0.06, 0.06, 0.1), mass_kg=0.05),
    ]
    result = settle_scene(bodies, FLAT, CFG)
    assert result.stable
    assert max(b.max_drift_m for b in result.bodies) < 2e-3


def test_free_fall_matches_the_closed_form() -> None:
    drop_m = 0.5 * CFG.gravity_m_s2 * 0.1**2
    cfg = SettlingConfig(**{**CFG.__dict__, "window_s": 0.1})
    result = settle_scene([box("a", pose(z=0.05 + 2 * drop_m))], FLAT, cfg)
    assert result.bodies[0].final_drift_m == pytest.approx(drop_m, rel=0.02)


def test_friction_decides_a_slope_at_the_angle_of_repose() -> None:
    angle = 20.0
    critical = np.tan(np.radians(angle))
    table = slope(angle)
    assert settle_scene([box("a", on_slope(angle), friction=critical * 1.2)], table, CFG).stable
    assert not settle_scene([box("a", on_slope(angle), friction=critical * 0.8)], table, CFG).stable


def test_slip_on_a_slope_matches_the_closed_form() -> None:
    angle, friction = 20.0, 0.1
    a = np.radians(angle)
    acceleration = CFG.gravity_m_s2 * (np.sin(a) - friction * np.cos(a))
    result = settle_scene([box("a", on_slope(angle), friction=friction)], slope(angle), CFG)
    assert result.bodies[0].final_drift_m == pytest.approx(0.5 * acceleration, rel=0.05)


def test_a_box_tips_only_when_its_center_of_mass_clears_the_edge() -> None:
    low = box("low", pose(z=0.05), (0.1, 0.1, 0.1), mass_kg=1.0)
    # the edge of the lower box is at x = 0.05
    held = box("top", pose(x=0.04, z=0.11), (0.1, 0.1, 0.02), mass_kg=0.1)
    over = box("top", pose(x=0.06, z=0.11), (0.1, 0.1, 0.02), mass_kg=0.1)
    assert settle_scene([low, held], FLAT, CFG).stable
    assert not settle_scene([low, over], FLAT, CFG).stable


def test_the_center_of_mass_is_used() -> None:
    """The same overhang rests when the mass sits over the support."""
    low = box("low", pose(z=0.05), (0.1, 0.1, 0.1), mass_kg=1.0)
    over = box("top", pose(x=0.06, z=0.11), (0.1, 0.1, 0.02), mass_kg=0.1)
    shifted = SettleBody(
        over.object_id,
        over.parts,
        over.T_world_from_object,
        over.mass_kg,
        over.inertia_kg_m2,
        np.array([-0.03, 0.0, 0.0]),
        over.friction,
    )
    assert settle_scene([low, shifted], FLAT, CFG).stable


def test_an_object_sunk_in_the_table_is_lifted_out() -> None:
    result = settle_scene([box("a", pose(z=0.04))], FLAT, CFG)
    body = result.bodies[0]
    assert body.separation_m == pytest.approx(0.01, abs=1e-3)
    assert body.max_drift_m == pytest.approx(0.01, abs=2e-3)
    assert result.stable


def test_overlapping_objects_are_separated_about_equally() -> None:
    bodies = [box("a", pose(x=-0.045, z=0.05)), box("b", pose(x=0.045, z=0.05))]
    result = settle_scene(bodies, FLAT, CFG)
    assert result.residual_penetration_m <= CFG.penetration_tolerance_m
    assert result.bodies[0].separation_m == pytest.approx(0.005, abs=1e-3)
    assert result.bodies[1].separation_m == pytest.approx(0.005, abs=1e-3)


def test_a_part_that_cannot_load_raises() -> None:
    flat = trimesh.Trimesh(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0.0]]), np.array([[0, 1, 2]]))
    good = box("a", pose(z=0.05))
    bad = SettleBody(
        "a", [flat], good.T_world_from_object, 0.2, good.inertia_kg_m2, np.zeros(3), 0.5
    )
    with pytest.raises(ValueError):
        settle_scene([bad], FLAT, CFG)


def test_no_bodies_is_stable() -> None:
    assert settle_scene([], FLAT, CFG).stable


def egg() -> trimesh.Trimesh:
    mesh = trimesh.creation.icosphere(subdivisions=4, radius=0.024)
    mesh.apply_scale([1.0, 1.0, 1.3])
    return cast(trimesh.Trimesh, mesh)


def test_balls_and_eggs_are_round_and_boxes_and_cans_are_not() -> None:
    assert is_round(trimesh.creation.icosphere(subdivisions=4, radius=0.07), 0.04)
    assert is_round(egg(), 0.04)
    assert not is_round(trimesh.creation.box(extents=(0.1, 0.1, 0.1)), 0.04)
    assert not is_round(trimesh.creation.box(extents=(0.24, 0.14, 0.04)), 0.04)
    assert not is_round(trimesh.creation.cylinder(radius=0.03, height=0.2, sections=48), 0.04)


def test_an_ignored_object_does_not_fail_the_scene_and_is_still_reported() -> None:
    falling = box("a", pose(z=0.3))
    assert not settle_scene([falling], FLAT, CFG).stable
    ignored = replace(falling, ignored=True)
    result = settle_scene([ignored], FLAT, CFG)
    assert result.stable
    assert result.bodies[0].ignored and not result.bodies[0].stable
    assert result.bodies[0].max_drift_m > CFG.max_drift_m
