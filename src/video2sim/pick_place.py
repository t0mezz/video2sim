"""Scripted pick-and-place with a Panda arm on a scene package (P5.4, D-027, D-029).

The scene comes from the exported URDF files, and is settled first, so the demo starts from what
a simulator would really hold. The arm stands beside the object on the side with the most free
room and picks from above, with the fingers closing across the narrow side of the footprint.

Friction: the Panda's geoms get zero friction, because MuJoCo takes the larger coefficient of a
pair (``stages/settling.py``). The grip then rests on the object's own value, which is what the
friction cases test; the model's own finger friction would hide them.

The control is open loop: joint targets from inverse kinematics on a straight tool path, with
nothing fed back from the object. A pass means that the twin's geometry, mass and friction allow
a plain scripted grasp, not that a real robot would succeed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import trimesh
from numpy.typing import NDArray

from video2sim.schemas import Flag, Plane, SimAsset
from video2sim.stages.physics_stage import CASES
from video2sim.stages.settling import SettlingConfig
from video2sim.stages.urdf_writer import settle_urdfs, urdf_scene_spec

F64 = NDArray[np.float64]
ARM_JOINTS = 7
GRIPPER_OPEN, GRIPPER_CLOSED = 255.0, 0.0  # the model's actuator range
HOME = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])  # the model's home keyframe
PAD_OFFSET_M = 0.1034  # pad center from the hand frame, along the hand's z (panda.xml)
IK_TOLERANCE = 1e-5  # meters and radians together
ARM_RADIUS_M = 0.09  # the arm's width, kept clear of other objects
# Flags that mean the object in the scene may not be the real one.
UNSUITABLE = {
    Flag.PROXY_GEOMETRY,
    Flag.LOW_FIT,
    Flag.IGNORED_ROUND,
    Flag.FAILED_VALIDATION,
    Flag.NON_RIGID,
}
# name, seconds, gripper control at the end of the phase
PHASES = (
    ("approach", 2.5, GRIPPER_OPEN),
    ("descend", 1.0, GRIPPER_OPEN),
    ("close", 0.8, GRIPPER_CLOSED),
    ("lift", 1.2, GRIPPER_CLOSED),
    ("transfer", 1.5, GRIPPER_CLOSED),
    ("lower", 1.2, GRIPPER_CLOSED),
    ("open", 0.8, GRIPPER_OPEN),
    ("retreat", 1.0, GRIPPER_OPEN),
    ("rest", 1.0, GRIPPER_OPEN),
)


@dataclass(frozen=True)
class DemoConfig:
    panda_model: Path
    base_distance_m: float
    place_distance_m: float
    grasp_depth_m: float
    min_grasp_height_m: float
    grip_force_n: float
    lift_clearance_m: float
    max_width_m: float
    min_height_m: float
    min_clearance_m: float
    base_angles: int
    control_dt_s: float
    video_fps: int
    video_size_px: tuple[int, int]
    lift_min_m: float
    place_error_max_m: float

    @classmethod
    def from_config(
        cls, cfg: dict[str, Any], targets: dict[str, Any], base_dir: Path
    ) -> DemoConfig:
        d = cfg["demo"]
        gate = targets["gates"]["pick_place"]
        return cls(
            panda_model=base_dir / d["panda_model"],
            base_distance_m=float(d["base_distance_m"]),
            place_distance_m=float(d["place_distance_m"]),
            grasp_depth_m=float(d["grasp_depth_m"]),
            min_grasp_height_m=float(d["min_grasp_height_m"]),
            grip_force_n=float(d["grip_force_n"]),
            lift_clearance_m=float(d["lift_clearance_m"]),
            max_width_m=float(d["max_width_m"]),
            min_height_m=float(d["min_height_m"]),
            min_clearance_m=float(d["min_clearance_m"]),
            base_angles=int(d["base_angles"]),
            control_dt_s=float(d["control_dt_s"]),
            video_fps=int(d["video_fps"]),
            video_size_px=(int(d["video_size_px"][0]), int(d["video_size_px"][1])),
            lift_min_m=float(gate["lift_min_m"]),
            place_error_max_m=float(gate["place_error_max_m"]),
        )


@dataclass(frozen=True)
class Footprint:
    """An object's outline at its settled pose."""

    object_id: str
    center: F64  # xy
    long_axis: F64  # unit xy
    width_m: float  # across the narrow side
    length_m: float
    bottom_m: float
    top_m: float


@dataclass(frozen=True)
class Plan:
    object_id: str
    base_xy: F64
    base_yaw: float
    grasp_m: F64  # pad center
    close_axis: F64  # unit xy across the narrow side
    width_m: float
    place_offset_m: F64  # xy
    clearance_m: float


@dataclass(frozen=True)
class CaseResult:
    case: str
    lifted_m: float
    place_error_m: float
    success: bool


def footprint(asset: SimAsset, pose: F64, run_dir: Path) -> Footprint:
    points = []
    for relative in asset.collision_mesh_paths:
        part: Any = trimesh.load(run_dir / relative, force="mesh")
        points.append(np.asarray(part.vertices) @ pose[:3, :3].T + pose[:3, 3])
    all_points = np.concatenate(points)
    (cx, cy), (w, h), angle = cv2.minAreaRect(all_points[:, :2].astype(np.float32))
    a = np.radians(angle)
    first = np.array([np.cos(a), np.sin(a)])
    second = np.array([-first[1], first[0]])
    long_axis, width, length = (first, h, w) if w >= h else (second, w, h)
    return Footprint(
        asset.object_id,
        np.array([cx, cy]),
        long_axis,
        float(width),
        float(length),
        float(all_points[:, 2].min()),
        float(all_points[:, 2].max()),
    )


def _radius(f: Footprint) -> float:
    return float(np.hypot(f.width_m, f.length_m)) / 2


def _gap(a: Footprint, b: Footprint) -> float:
    """Free distance between two footprints, counting each as its enclosing circle."""
    return float(np.linalg.norm(a.center - b.center)) - _radius(a) - _radius(b)


def _segment_distance(point: F64, start: F64, end: F64) -> float:
    d = end - start
    s = np.clip((point - start) @ d / max(float(d @ d), 1e-12), 0.0, 1.0)
    return float(np.linalg.norm(point - (start + s * d)))


def plan_pick(
    target: Footprint, others: list[Footprint], table_z: float, cfg: DemoConfig
) -> Plan | None:
    """Where the arm stands and where the object goes, or None if no side leaves it room."""
    best: tuple[float, F64, F64] | None = None
    for k in range(cfg.base_angles):
        angle = 2 * np.pi * k / cfg.base_angles
        outward = np.array([np.cos(angle), np.sin(angle)])
        base = target.center + cfg.base_distance_m * outward
        arm = min(
            (
                _segment_distance(o.center, base, target.center) - _radius(o) - ARM_RADIUS_M
                for o in others
            ),
            default=np.inf,
        )
        for side in (1.0, -1.0):
            offset = side * cfg.place_distance_m * np.array([-outward[1], outward[0]])
            goal = target.center + offset
            room = min(
                (
                    float(np.linalg.norm(o.center - goal)) - _radius(o) - _radius(target)
                    for o in others
                ),
                default=np.inf,
            )
            score = min(float(arm), room)
            if best is None or score > best[0]:
                best = (score, base, offset)
    if best is None or best[0] < 0:
        return None
    score, base, offset = best
    towards = target.center - base
    z = max(target.top_m - cfg.grasp_depth_m, table_z + cfg.min_grasp_height_m)
    return Plan(
        target.object_id,
        base,
        float(np.arctan2(towards[1], towards[0])),
        np.array([*target.center, z]),
        np.array([-target.long_axis[1], target.long_axis[0]]),
        target.width_m,
        offset,
        score,
    )


def choose_object(
    assets: list[SimAsset], prints: dict[str, Footprint], table_z: float, cfg: DemoConfig
) -> Plan | None:
    """The object that fits the gripper with the most free room around it."""
    best: Plan | None = None
    for asset in assets:
        f = prints[asset.object_id]
        others = [p for name, p in prints.items() if name != asset.object_id]
        if UNSUITABLE & set(asset.flags) or f.width_m > cfg.max_width_m:
            continue
        if f.top_m - f.bottom_m < cfg.min_height_m or f.bottom_m > table_z + 0.01:
            continue  # too low to pinch, or standing on something
        if any(_gap(f, o) < cfg.min_clearance_m for o in others):
            continue
        plan = plan_pick(f, others, table_z, cfg)
        if plan is not None and (best is None or plan.clearance_m > best.clearance_m):
            best = plan
    return best


def _panda_spec(cfg: DemoConfig) -> Any:
    import mujoco  # type: ignore[import-untyped]  # heavy import

    panda = mujoco.MjSpec.from_file(str(cfg.panda_model))
    hand = panda.body("hand")
    site = hand.add_site()
    site.name = "pad"
    site.pos = [0.0, 0.0, PAD_OFFSET_M]
    for body in panda.bodies:
        body.gravcomp = 1.0  # the real controller cancels the arm's weight
    for geom in panda.geoms:
        geom.friction = [0.0, 0.0, 0.0]
    return panda


def build_model(
    urdfs: dict[str, Path],
    assets: list[SimAsset],
    table: Plane,
    settling: SettlingConfig,
    plan: Plan,
    colors: dict[str, F64],
    cfg: DemoConfig,
) -> Any:
    import mujoco  # heavy import

    spec = urdf_scene_spec(urdfs, assets, table, settling)
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.visual.global_.offwidth, spec.visual.global_.offheight = cfg.video_size_px
    light = spec.worldbody.add_light()
    light.pos = [*plan.grasp_m[:2].tolist(), 2.0]
    for asset in assets:
        for geom in spec.body(f"{asset.object_id}/{asset.object_id}").geoms:
            geom.rgba = [*colors[asset.object_id].tolist(), 1.0]
    panda = _panda_spec(cfg)
    frame = spec.worldbody.add_frame()
    frame.pos = [*plan.base_xy.tolist(), table.offset_m]
    half = plan.base_yaw / 2
    frame.quat = [np.cos(half), 0.0, 0.0, np.sin(half)]
    frame.attach_body(panda.body("link0"), "panda/", "")
    model = spec.compile()
    set_grip(model, plan.width_m, cfg.grip_force_n)
    return model


def set_grip(model: Any, width_m: float, force_n: float) -> None:
    """Make the gripper squeeze an object of ``width_m`` with ``force_n``.

    The model's actuator is a soft position servo, which at 5 cm pushes with 2.5 N in all. A
    position error stands for the force here: ``ctrl`` 0 sits ``width_m / 2`` from the closed
    tendon, so the stiffness follows from it. The tendon carries half of each finger's motion, so
    each finger gets half its force, and a pinch of ``force_n`` needs twice that.
    """
    tendon_force = 2 * force_n
    id_ = model.actuator("panda/actuator8").id
    stiffness = tendon_force / (width_m / 2)
    ratio = stiffness / -model.actuator_biasprm[id_, 1]
    model.actuator_gainprm[id_, 0] *= ratio
    model.actuator_biasprm[id_, 1] = -stiffness
    model.actuator_forcerange[id_] = [-tendon_force, tendon_force]


def apply_case(model: Any, assets: list[SimAsset], case: str) -> None:
    """Set ``case`` of ``CASES`` in a freshly compiled ``model``."""
    for i, asset in enumerate(assets):
        p = asset.properties
        body = i + 1
        if case.startswith("mass_"):
            mass = float(getattr(p.mass_kg, case.removeprefix("mass_")))
            ratio = mass / p.mass_kg.nominal
            model.body_mass[body] *= ratio
            model.body_inertia[body] *= ratio
        if case.startswith("friction_"):
            friction = float(getattr(p.friction_coefficient, case.removeprefix("friction_")))
            model.geom_friction[model.geom_bodyid == body, 0] = friction


class _Arm:
    """Addresses of the arm in the model, and inverse kinematics on a separate ``MjData``."""

    def __init__(self, mujoco: Any, model: Any) -> None:
        self.mujoco, self.model = mujoco, model
        self.data = mujoco.MjData(model)
        joints = [model.joint(f"panda/joint{i + 1}") for i in range(ARM_JOINTS)]
        self.qpos = np.array([j.qposadr[0] for j in joints])
        self.dof = np.array([j.dofadr[0] for j in joints])
        self.fingers = [model.joint(f"panda/finger_joint{i}").qposadr[0] for i in (1, 2)]
        self.actuators = [model.actuator(f"panda/actuator{i + 1}").id for i in range(ARM_JOINTS)]
        self.gripper = model.actuator("panda/actuator8").id
        self.site = model.site("panda/pad").id

    def solve(self, q: F64, position: F64, rotation: F64) -> F64:
        m = self.mujoco
        target = np.zeros(4)
        m.mju_mat2Quat(target, rotation.reshape(-1))
        q = q.copy()
        for _ in range(100):
            self.data.qpos[self.qpos] = q
            m.mj_kinematics(self.model, self.data)
            m.mj_comPos(self.model, self.data)
            now = np.zeros(4)
            m.mju_mat2Quat(now, self.data.site_xmat[self.site])
            turn = np.zeros(3)
            m.mju_subQuat(turn, target, now)  # in the tool frame; the Jacobian is in the world's
            turn = self.data.site_xmat[self.site].reshape(3, 3) @ turn
            error = np.r_[position - self.data.site_xpos[self.site], turn]
            if np.linalg.norm(error) < IK_TOLERANCE:
                return q
            jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
            m.mj_jacSite(self.model, self.data, jp, jr, self.site)
            j = np.vstack([jp, jr])[:, self.dof]
            step = j.T @ np.linalg.solve(j @ j.T + 1e-4 * np.eye(6), error)
            step += (np.eye(ARM_JOINTS) - np.linalg.pinv(j) @ j) @ (HOME - q) * 0.05
            q += np.clip(step, -0.2, 0.2)
        raise ValueError(f"no joint angles reach the tool at {np.round(position, 3).tolist()}")


def _trajectory(arm: _Arm, plan: Plan, cfg: DemoConfig) -> list[tuple[F64, float]]:
    """Joint targets and gripper control for each control tick."""
    y = np.array([*plan.close_axis, 0.0])
    z = np.array([0.0, 0.0, -1.0])
    rotation = np.column_stack([np.cross(y, z), y, z])
    up = np.array([0.0, 0.0, cfg.lift_min_m + cfg.lift_clearance_m])
    goal = plan.grasp_m + np.array([*plan.place_offset_m, 0.002])  # a hair above, not pressed down
    tool = {
        "descend": (plan.grasp_m + up, plan.grasp_m),
        "close": (plan.grasp_m, plan.grasp_m),
        "lift": (plan.grasp_m, plan.grasp_m + up),
        "transfer": (plan.grasp_m + up, goal + up),
        "lower": (goal + up, goal),
        "open": (goal, goal),
        "retreat": (goal, goal + up),
        "rest": (goal + up, goal + up),
    }
    start = arm.solve(HOME, plan.grasp_m + up, rotation)
    ticks: list[tuple[F64, float]] = []
    q = start
    gripper = GRIPPER_OPEN
    for name, seconds, end_gripper in PHASES:
        count = max(round(seconds / cfg.control_dt_s), 1)
        for k in range(1, count + 1):
            s = k / count
            smooth = s * s * s * (10 + s * (-15 + 6 * s))  # minimum jerk
            if name == "approach":
                target = HOME + smooth * (start - HOME)
            else:
                a, b = tool[name]
                q = arm.solve(q, a + smooth * (b - a), rotation)
                target = q
            ticks.append((target.copy(), gripper + s * (end_gripper - gripper)))
        gripper = end_gripper
    return ticks


def run_case(
    model: Any,
    plan: Plan,
    index: int,
    poses: dict[str, F64],
    assets: list[SimAsset],
    cfg: DemoConfig,
    video: Path | None = None,
) -> tuple[float, float]:
    """Run the plan; returns how far the object rose and its distance from the goal at the end."""
    import mujoco  # heavy import

    arm = _Arm(mujoco, model)
    data = mujoco.MjData(model)
    for i, asset in enumerate(assets):
        t = poses[asset.object_id]
        data.qpos[7 * i : 7 * i + 3] = t[:3, 3]
        mujoco.mju_mat2Quat(data.qpos[7 * i + 3 : 7 * i + 7], t[:3, :3].reshape(-1))
    data.qpos[arm.qpos] = HOME
    data.qpos[arm.fingers] = 0.04
    data.ctrl[arm.actuators] = HOME
    data.ctrl[arm.gripper] = GRIPPER_OPEN
    mujoco.mj_forward(model, data)

    start = data.qpos[7 * index : 7 * index + 3].copy()
    goal = start + np.array([*plan.place_offset_m, 0.0])
    highest = start[2]
    every = max(round(cfg.control_dt_s / model.opt.timestep), 1)
    width, height = cfg.video_size_px
    renderer = camera = writer = None
    frame_every = max(round(1.0 / (cfg.video_fps * cfg.control_dt_s)), 1)
    if video is not None:
        renderer = mujoco.Renderer(model, height, width)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [*(plan.grasp_m[:2] + plan.base_xy) / 2, 0.15]
        camera.distance, camera.elevation = 1.3, -30.0
        camera.azimuth = float(np.degrees(plan.base_yaw)) + 90.0
        video.parent.mkdir(parents=True, exist_ok=True)
        fourcc: Any = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]  # OpenCV stub gap
        writer = cv2.VideoWriter(str(video), fourcc, cfg.video_fps, (width, height))

    for tick, (target, gripper) in enumerate(_trajectory(arm, plan, cfg)):
        data.ctrl[arm.actuators] = target
        data.ctrl[arm.gripper] = gripper
        for _ in range(every):
            mujoco.mj_step(model, data)
        highest = max(highest, float(data.qpos[7 * index + 2]))
        if renderer is not None and writer is not None and tick % frame_every == 0:
            renderer.update_scene(data, camera)
            writer.write(renderer.render()[:, :, ::-1])
    if writer is not None and renderer is not None:
        writer.release()
        renderer.close()
    final = data.qpos[7 * index : 7 * index + 3]
    return highest - float(start[2]), float(np.linalg.norm(final - goal))


def mean_color(asset: SimAsset, run_dir: Path) -> F64:
    mesh: Any = trimesh.load(run_dir / asset.visual_mesh_path, force="mesh", process=False)
    if not isinstance(mesh.visual, trimesh.visual.ColorVisuals) or not mesh.visual.kind:
        return np.array([0.6, 0.6, 0.6])
    rgb: F64 = np.asarray(mesh.visual.vertex_colors, dtype=float)[:, :3]
    return np.asarray(rgb.mean(axis=0) / 255.0, dtype=float)


def run_demo(
    urdfs: dict[str, Path],
    assets: list[SimAsset],
    table: Plane,
    settling: SettlingConfig,
    run_dir: Path,
    cfg: DemoConfig,
    video: Path | None = None,
) -> tuple[Plan | None, list[CaseResult]]:
    """Choose an object and run every case of ``CASES``; the nominal one is recorded."""
    empty = urdf_scene_spec(urdfs, assets, table, settling).compile()
    poses = settle_urdfs(empty, assets, table, settling).final_poses
    prints = {a.object_id: footprint(a, poses[a.object_id], run_dir) for a in assets}
    plan = choose_object(assets, prints, table.offset_m, cfg)
    if plan is None:
        return None, []
    index = [a.object_id for a in assets].index(plan.object_id)
    colors = {a.object_id: mean_color(a, run_dir) for a in assets}
    results = []
    for case in CASES:
        model = build_model(urdfs, assets, table, settling, plan, colors, cfg)
        apply_case(model, assets, case)
        lifted, error = run_case(
            model, plan, index, poses, assets, cfg, video if case == "nominal" else None
        )
        results.append(
            CaseResult(
                case,
                lifted,
                error,
                bool(lifted >= cfg.lift_min_m and error <= cfg.place_error_max_m),
            )
        )
    return plan, results


__all__ = [
    "CaseResult",
    "DemoConfig",
    "Plan",
    "choose_object",
    "footprint",
    "plan_pick",
    "run_demo",
    "set_grip",
]
