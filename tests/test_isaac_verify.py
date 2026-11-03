"""The USD side of the Isaac Sim verifier, on a small hand-written scene (no Kit needed)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from video2sim.isaac_verify import (
    BodyDrift,
    apply_case,
    ignored_names,
    judge,
    object_names,
    positions,
    scene_stable,
)

INERTIA = (0.001, 0.002, 0.003)


@pytest.fixture
def stage(tmp_path: Path) -> Usd.Stage:
    stage = Usd.Stage.CreateNew(str(tmp_path / "scene.usd"))
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/Materials")
    UsdGeom.Xform.Define(stage, "/World/Objects")
    for name, flags in (("obj_000", []), ("obj_001", ["ignored_round"])):
        material = UsdPhysics.MaterialAPI.Apply(
            stage.DefinePrim(f"/World/Materials/{name}", "Material")
        )
        material.CreateStaticFrictionAttr(0.5)
        material.CreateDynamicFrictionAttr(0.5)
        prim = stage.DefinePrim(f"/World/Objects/{name}", "Xform")
        xform = UsdGeom.Xformable(prim)
        xform.AddTranslateOp().Set(Gf.Vec3d(1.0, 2.0, 0.5))
        mass = UsdPhysics.MassAPI.Apply(prim)
        mass.CreateMassAttr(0.2)
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*INERTIA))
        for key, nominal, low, high in (
            ("mass_kg", 0.2, 0.1, 0.4),
            ("friction_coefficient", 0.5, 0.2, 0.9),
        ):
            for level, value in (("nominal", nominal), ("low", low), ("high", high)):
                prim.CreateAttribute(f"video2sim:{key}:{level}", Sdf.ValueTypeNames.Double).Set(
                    value
                )
        prim.CreateAttribute("video2sim:flags", Sdf.ValueTypeNames.StringArray).Set(flags)
    return stage


def test_friction_case_sets_static_and_dynamic_friction(stage: Usd.Stage) -> None:
    apply_case(stage, "friction_high")
    material = UsdPhysics.MaterialAPI(stage.GetPrimAtPath("/World/Materials/obj_000"))
    assert material.GetStaticFrictionAttr().Get() == pytest.approx(0.9)
    assert material.GetDynamicFrictionAttr().Get() == pytest.approx(0.9)
    # mass untouched
    assert UsdPhysics.MassAPI(
        stage.GetPrimAtPath("/World/Objects/obj_000")
    ).GetMassAttr().Get() == (pytest.approx(0.2))


def test_mass_case_scales_inertia_with_mass(stage: Usd.Stage) -> None:
    apply_case(stage, "mass_high")
    api = UsdPhysics.MassAPI(stage.GetPrimAtPath("/World/Objects/obj_001"))
    assert api.GetMassAttr().Get() == pytest.approx(0.4)
    assert tuple(api.GetDiagonalInertiaAttr().Get()) == pytest.approx(tuple(2 * i for i in INERTIA))


def test_nominal_changes_nothing_and_unknown_case_is_refused(stage: Usd.Stage) -> None:
    apply_case(stage, "nominal")
    material = UsdPhysics.MaterialAPI(stage.GetPrimAtPath("/World/Materials/obj_000"))
    assert material.GetStaticFrictionAttr().Get() == pytest.approx(0.5)
    with pytest.raises(ValueError):
        apply_case(stage, "mass_medium")


def test_names_positions_and_ignored(stage: Usd.Stage) -> None:
    names = object_names(stage)
    assert names == ["obj_000", "obj_001"]
    assert positions(stage, names) == pytest.approx(np.tile([1.0, 2.0, 0.5], (2, 1)))
    assert ignored_names(stage) == {"obj_001"}


def test_judge_uses_the_largest_drift_not_the_last() -> None:
    start = np.zeros((2, 3))
    rolled = np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]])  # second object rolls away ...
    final = np.array([[0.001, 0.0, 0.0], [0.0, 0.0, 0.0]])  # ... and comes back
    bodies = judge(["a", "b"], start, [rolled], final, 0.02, set())
    assert [b.stable for b in bodies] == [True, False]
    assert bodies[1].max_drift_m == pytest.approx(0.05)
    assert bodies[1].final_drift_m == 0.0
    assert not scene_stable(bodies)


def test_ignored_round_object_does_not_gate() -> None:
    far = np.array([[0.5, 0.0, 0.0]])
    bodies = judge(["ball"], np.zeros((1, 3)), [], far, 0.02, {"ball"})
    assert bodies == [BodyDrift("ball", 0.5, 0.5, False, True)]
    assert scene_stable(bodies)
