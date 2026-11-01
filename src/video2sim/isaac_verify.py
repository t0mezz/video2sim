"""What the Isaac Sim settling verifier does to the exported ``scene.usd`` (D-027, item 8).

Isaac Sim ships its own interpreter, so this module imports only the standard library, NumPy and
``pxr``, and talks to the rest of the pipeline through the file. It holds everything that does not
need Kit running: choosing a case, reading object positions and judging drift. The Kit side
(opening the stage, stepping PhysX) is ``benchmark/run_isaac_settling.py``.

The cases and the gate are those of ``stages/settling.py``. Differences, so a PhysX number is read
against the right MuJoCo number:

* No overlap separation before the drop: the USD is simulated as written, so a pose that puts two
  meshes through each other shows up as drift (MuJoCo moves them apart first and reports that
  separately).
* Drift is the largest distance of an object-frame origin from its pose in the file, sampled
  during the window, as in MuJoCo.
* An object flagged ``ignored_round`` stays in the report and does not gate (D-025).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

F64 = NDArray[np.float64]
CASES = ("nominal", "friction_low", "friction_high", "mass_low", "mass_high")
OBJECTS = "/World/Objects"
MATERIALS = "/World/Materials"
IGNORED_ROUND = "ignored_round"  # Flag.IGNORED_ROUND, spelled out to keep schemas out of Kit


@dataclass(frozen=True)
class BodyDrift:
    object_id: str
    max_drift_m: float
    final_drift_m: float
    stable: bool
    ignored: bool = False


def object_names(stage: Any) -> list[str]:
    return [p.GetName() for p in stage.GetPrimAtPath(OBJECTS).GetChildren()]


def ignored_names(stage: Any) -> set[str]:
    """Objects the export flagged as round, which do not count towards a scene being stable."""
    out = set()
    for name in object_names(stage):
        flags = stage.GetPrimAtPath(f"{OBJECTS}/{name}").GetAttribute("video2sim:flags").Get()
        if flags and IGNORED_ROUND in flags:
            out.add(name)
    return out


def apply_case(stage: Any, case: str) -> None:
    """Set friction or mass of every object to ``case``, from its ``video2sim:*`` estimates.

    Mass cases scale the inertia with the mass, because the body is the same and only denser, as
    in ``benchmark/run_settling.py``.
    """
    from pxr import Gf, UsdPhysics

    if case not in CASES:
        raise ValueError(f"unknown case {case!r}, expected one of {CASES}")
    kind, _, level = case.partition("_")
    for name in object_names(stage):
        prim = stage.GetPrimAtPath(f"{OBJECTS}/{name}")
        if kind == "friction":
            value = float(prim.GetAttribute(f"video2sim:friction_coefficient:{level}").Get())
            material = UsdPhysics.MaterialAPI(stage.GetPrimAtPath(f"{MATERIALS}/{name}"))
            material.GetStaticFrictionAttr().Set(value)
            material.GetDynamicFrictionAttr().Set(value)
        elif kind == "mass":
            nominal = float(prim.GetAttribute("video2sim:mass_kg:nominal").Get())
            mass = float(prim.GetAttribute(f"video2sim:mass_kg:{level}").Get())
            api = UsdPhysics.MassAPI(prim)
            inertia = Gf.Vec3f(*prim.GetAttribute("physics:diagonalInertia").Get())
            api.GetMassAttr().Set(mass)
            api.GetDiagonalInertiaAttr().Set(inertia * (mass / nominal))


def positions(stage: Any, names: list[str]) -> F64:
    """World position of each object-frame origin, ``[n, 3]``, at the current time."""
    from pxr import Usd, UsdGeom

    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    return np.array(
        [
            cache.GetLocalToWorldTransform(
                stage.GetPrimAtPath(f"{OBJECTS}/{n}")
            ).ExtractTranslation()
            for n in names
        ],
        dtype=np.float64,
    ).reshape(len(names), 3)


def judge(
    names: list[str],
    start: F64,
    samples: list[F64],
    final: F64,
    max_drift_m: float,
    ignored: set[str],
) -> list[BodyDrift]:
    """Drift of each object: the largest over ``samples`` and ``final``, and the final one.

    The stable test takes the largest drift and not the last one, so an object that rolls away and
    stops is not excused.
    """
    largest = np.linalg.norm(final - start, axis=1)
    for sample in samples:
        largest = np.maximum(largest, np.linalg.norm(sample - start, axis=1))
    last = np.linalg.norm(final - start, axis=1)
    return [
        BodyDrift(
            n, float(largest[i]), float(last[i]), bool(largest[i] <= max_drift_m), n in ignored
        )
        for i, n in enumerate(names)
    ]


def scene_stable(bodies: list[BodyDrift]) -> bool:
    return all(b.stable or b.ignored for b in bodies)
