"""Scene settling for the physics stage (P4.7).

Input is the scene as the earlier physics steps leave it: convex parts in the object frame
(P4.2), mass, inertia and center of mass (P4.5), friction (P4.3) and the observed pose of each
object, with the table plane of ``WorldModel``. Output is how far each object moved.

MuJoCo stands in for the target simulator until P5.1 (D-021), so a pass here is provisional. It
also checks what ``stages/collision.py`` leaves to the simulator: a part it cannot load raises
``ValueError`` from ``settle_scene``.

* Each object is a free body whose frame is the object frame, with the mass and inertia given and
  not computed from the parts. The table is an infinite plane with zero friction, because
  MuJoCo takes the larger coefficient of a pair: an object on the table then has its own value,
  which the material table states "against a typical table surface".
* Overlaps are resolved first, because a pose that fits the video can still put two meshes
  through each other, and the contact solver would fire them apart. Every body in a contact
  deeper than ``penetration_tolerance_m`` moves along the contact normal, half each, or all of it
  against the table, until none is left or ``max_separation_iterations`` runs out.
* Drift is the distance of the object-frame origin from its observed pose, so it includes the
  separation. The stable test takes the largest drift in the window and not the last one, so an
  object that rolls away and stops is not excused. Rotation is reported and does not gate:
  ``stability_max_drift_m`` is a length.
* A round object stays in the scene and its drift is reported, but it does not count towards a
  scene being stable (D-025): a ball rolls and an egg falls over on the real table too.

Restitution is not simulated. MuJoCo has no restitution coefficient, only a contact spring, and
these objects fall a few millimeters. It stays in the export for P5.1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import trimesh
from numpy.typing import NDArray

from video2sim.schemas import Plane

F64 = NDArray[np.float64]
ROUND_SAMPLES = 4000


@dataclass(frozen=True)
class SettlingConfig:
    """``physics.settling`` in ``configs/default.yaml``, and the gate from ``targets.yaml``."""

    round_tolerance: float
    timestep_s: float
    gravity_m_s2: float
    penetration_tolerance_m: float
    max_separation_iterations: int
    noslip_iterations: int
    drift_sample_s: float
    window_s: float
    max_drift_m: float

    @classmethod
    def from_config(cls, cfg: dict[str, Any], targets: dict[str, Any]) -> SettlingConfig:
        c = cfg["physics"]["settling"]
        t = targets["targets"]
        out = cls(
            round_tolerance=float(c["round_tolerance"]),
            timestep_s=float(c["timestep_s"]),
            gravity_m_s2=float(c["gravity_m_s2"]),
            penetration_tolerance_m=float(c["penetration_tolerance_m"]),
            max_separation_iterations=int(c["max_separation_iterations"]),
            noslip_iterations=int(c["noslip_iterations"]),
            drift_sample_s=float(c["drift_sample_s"]),
            window_s=float(t["stability_window_s"]),
            max_drift_m=float(t["stability_max_drift_m"]),
        )
        if out.timestep_s <= 0 or out.gravity_m_s2 <= 0 or out.drift_sample_s < out.timestep_s:
            raise ValueError("physics.settling needs positive steps, and samples of at least one")
        if out.penetration_tolerance_m <= 0 or out.max_separation_iterations < 0:
            raise ValueError("physics.settling separation settings are out of range")
        return out


@dataclass(frozen=True)
class SettleBody:
    """One object in the world. Mass and friction are the case being run, not an ``Estimate``."""

    object_id: str
    parts: list[trimesh.Trimesh]
    T_world_from_object: F64
    mass_kg: float
    inertia_kg_m2: F64  # about the center of mass, in the object frame
    center_of_mass_m: F64  # in the object frame
    friction: float
    ignored: bool = False  # round, so it does not count towards the scene (D-025)


@dataclass(frozen=True)
class BodyResult:
    object_id: str
    separation_m: float  # moved to clear overlaps, before the drop
    max_drift_m: float
    final_drift_m: float
    rotation_deg: float
    stable: bool
    ignored: bool = False


@dataclass(frozen=True)
class SceneResult:
    bodies: list[BodyResult]
    residual_penetration_m: float  # the deepest overlap the separation left

    @property
    def stable(self) -> bool:
        return all(b.stable or b.ignored for b in self.bodies)


def is_round(mesh: trimesh.Trimesh, tolerance: float, seed: int = 0) -> bool:
    """Whether ``mesh`` is close to the ellipsoid in its oriented bounding box.

    The 95th percentile of the surface distances in both directions, as a share of the bounding
    box diagonal, as in ``collision.py``.
    """
    from scipy.spatial import cKDTree  # type: ignore[import-untyped]  # heavy import

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)  # type: ignore[no-untyped-call]
    ellipsoid = trimesh.creation.icosphere(subdivisions=4)
    ellipsoid.apply_scale(np.asarray(extents) / 2)
    ellipsoid.apply_transform(np.linalg.inv(to_origin))
    a = trimesh.sample.sample_surface(mesh, ROUND_SAMPLES, seed=seed)[0]
    b = trimesh.sample.sample_surface(ellipsoid, ROUND_SAMPLES, seed=seed)[0]
    distance = np.concatenate([cKDTree(b).query(a)[0], cKDTree(a).query(b)[0]])
    return bool(np.percentile(distance, 95) / np.linalg.norm(mesh.extents) <= tolerance)


def _floats(values: F64) -> str:
    return " ".join(repr(float(x)) for x in np.asarray(values).reshape(-1))


def _body_xml(index: int, body: SettleBody) -> tuple[str, str]:
    """The ``<asset>`` and ``<worldbody>`` entries of one object."""
    meshes = []
    geoms = []
    for k, part in enumerate(body.parts):
        name = f"m{index}_{k}"
        meshes.append(f'<mesh name="{name}" vertex="{_floats(np.asarray(part.vertices))}"/>')
        geoms.append(
            f'<geom type="mesh" mesh="{name}" friction="{float(body.friction)!r} 0 0" condim="3"/>'
        )
    i = np.asarray(body.inertia_kg_m2, dtype=float)
    # MuJoCo's order for fullinertia is xx yy zz xy xz yz
    full = np.array([i[0, 0], i[1, 1], i[2, 2], i[0, 1], i[0, 2], i[1, 2]])
    xml = (
        f'<body name="b{index}"><freejoint/>'
        f'<inertial pos="{_floats(body.center_of_mass_m)}" mass="{float(body.mass_kg)!r}" '
        f'fullinertia="{_floats(full)}"/>{"".join(geoms)}</body>'
    )
    return "".join(meshes), xml


def scene_xml(bodies: list[SettleBody], table: Plane, cfg: SettlingConfig) -> str:
    """The MJCF of the scene: an infinite table, and the bodies at the origin."""
    entries = [_body_xml(i, b) for i, b in enumerate(bodies)]
    normal = np.asarray(table.normal, dtype=float)
    # multiccd: without it a pair of convex parts gets one contact point and a box rocks
    return (
        '<mujoco><compiler inertiafromgeom="false"/>'
        f'<option timestep="{cfg.timestep_s!r}" gravity="0 0 {-cfg.gravity_m_s2!r}" '
        f'cone="elliptic" noslip_iterations="{cfg.noslip_iterations}">'
        '<flag multiccd="enable"/></option>'
        f"<asset>{''.join(e[0] for e in entries)}</asset>"
        f'<worldbody><geom name="table" type="plane" size="0 0 1" friction="0 0 0" '
        f'condim="3" pos="{_floats(normal * table.offset_m)}" zaxis="{_floats(normal)}"/>'
        f"{''.join(e[1] for e in entries)}</worldbody></mujoco>"
    )


def _place(mujoco: Any, model: Any, data: Any, bodies: list[SettleBody]) -> F64:
    """Write the observed poses into ``qpos`` and return them as ``[n, 7]``."""
    pose = np.zeros((len(bodies), 7))
    for i, body in enumerate(bodies):
        t = np.asarray(body.T_world_from_object, dtype=float)
        pose[i, :3] = t[:3, 3]
        mujoco.mju_mat2Quat(pose[i, 3:], t[:3, :3].reshape(-1))
    data.qpos[: pose.size] = pose.reshape(-1)
    mujoco.mj_forward(model, data)
    return pose


def _deepest_contacts(model: Any, data: Any) -> dict[tuple[int, int], tuple[float, F64]]:
    """The deepest contact of each pair of bodies: depth, and the normal from first to second."""
    deepest: dict[tuple[int, int], tuple[float, F64]] = {}
    for contact in data.contact[: data.ncon]:
        first = int(model.geom_bodyid[contact.geom1])
        second = int(model.geom_bodyid[contact.geom2])
        depth = -float(contact.dist)
        key = (first, second)
        if key not in deepest or depth > deepest[key][0]:
            deepest[key] = (depth, np.array(contact.frame[:3], dtype=float))
    return deepest


def _separate(
    mujoco: Any, model: Any, data: Any, count: int, cfg: SettlingConfig
) -> tuple[F64, float]:
    """Push overlapping bodies apart in ``data.qpos``; returns each move and the deepest left."""
    start = data.qpos[: 7 * count].reshape(count, 7)[:, :3].copy()
    deepest = 0.0
    for iteration in range(cfg.max_separation_iterations + 1):
        mujoco.mj_forward(model, data)
        contacts = _deepest_contacts(model, data)
        deepest = max((d for d, _ in contacts.values()), default=0.0)
        if deepest <= cfg.penetration_tolerance_m or iteration == cfg.max_separation_iterations:
            break
        shift = np.zeros((count + 1, 3))  # by body id; 0 is the world and stays where it is
        for (first, second), (depth, normal) in contacts.items():
            if depth <= cfg.penetration_tolerance_m:
                continue
            first_share = 0.0 if first == 0 else (1.0 if second == 0 else 0.5)
            second_share = 0.0 if second == 0 else (1.0 if first == 0 else 0.5)
            shift[first] -= first_share * depth * normal
            shift[second] += second_share * depth * normal
        for body_id in range(1, count + 1):
            data.qpos[7 * (body_id - 1) : 7 * (body_id - 1) + 3] += shift[body_id]
    moved = data.qpos[: 7 * count].reshape(count, 7)[:, :3] - start
    return np.linalg.norm(moved, axis=1), max(deepest, 0.0)


def _rotation_deg(a: F64, b: F64) -> float:
    """Angle between two unit quaternions."""
    return float(np.degrees(2 * np.arccos(np.clip(abs(float(a @ b)), 0.0, 1.0))))


def settle_scene(bodies: list[SettleBody], table: Plane, cfg: SettlingConfig) -> SceneResult:
    """Separate the bodies, run ``cfg.window_s`` of gravity and measure drift from the poses."""
    import mujoco  # type: ignore[import-untyped]  # heavy import

    if not bodies:
        return SceneResult([], 0.0)
    model = mujoco.MjModel.from_xml_string(scene_xml(bodies, table, cfg))
    data = mujoco.MjData(model)
    count = len(bodies)
    observed = _place(mujoco, model, data, bodies)
    separation, residual = _separate(mujoco, model, data, count, cfg)
    mujoco.mj_forward(model, data)

    largest = separation.copy()
    every = max(round(cfg.drift_sample_s / cfg.timestep_s), 1)
    steps = round(cfg.window_s / cfg.timestep_s)
    for step in range(1, steps + 1):
        mujoco.mj_step(model, data)
        if step % every == 0 or step == steps:
            position = data.qpos[: 7 * count].reshape(count, 7)[:, :3]
            largest = np.maximum(largest, np.linalg.norm(position - observed[:, :3], axis=1))
    final_pose = data.qpos[: 7 * count].reshape(count, 7)
    final = np.linalg.norm(final_pose[:, :3] - observed[:, :3], axis=1)
    results = [
        BodyResult(
            object_id=body.object_id,
            separation_m=float(separation[i]),
            max_drift_m=float(largest[i]),
            final_drift_m=float(final[i]),
            rotation_deg=_rotation_deg(observed[i, 3:], final_pose[i, 3:]),
            stable=bool(largest[i] <= cfg.max_drift_m),
            ignored=body.ignored,
        )
        for i, body in enumerate(bodies)
    ]
    return SceneResult(results, residual)


__all__ = [
    "BodyResult",
    "SceneResult",
    "SettleBody",
    "SettlingConfig",
    "is_round",
    "scene_xml",
    "settle_scene",
]
