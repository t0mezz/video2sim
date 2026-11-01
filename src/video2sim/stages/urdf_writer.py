"""One URDF per object (P5.2), and the MuJoCo cross-check of those files.

A URDF has no place in the world and no friction, so the scene pose and friction stay in the USD
scene; the cross-check takes them from ``SimAssets``. The inertia entries are those of the matrix,
which are the negatives of the products of inertia (``AGENTS.md``), about the origin, which is the
center of mass.

The check loads every file into one MuJoCo model, compares what MuJoCo compiled with ``SimAssets``,
and settles that model. So a wrong unit, axis or inertia in a URDF shows as a difference from the
settling of the physics stage, which never read these files.
"""

from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, cast

import numpy as np
import trimesh
from numpy.typing import NDArray

from video2sim.schemas import Plane, SimAsset
from video2sim.stages.settling import SceneResult, SettlingConfig, settle_scene
from video2sim.stages.settling import SettleBody as Body

F64 = NDArray[np.float64]


def _inertial(link: ET.Element, asset: SimAsset) -> None:
    i = asset.properties.inertia_kg_m2
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", xyz="0 0 0", rpy="0 0 0")
    ET.SubElement(inertial, "mass", value=repr(float(asset.properties.mass_kg.nominal)))
    ET.SubElement(
        inertial,
        "inertia",
        ixx=repr(float(i[0][0])),
        ixy=repr(float(i[0][1])),
        ixz=repr(float(i[0][2])),
        iyy=repr(float(i[1][1])),
        iyz=repr(float(i[1][2])),
        izz=repr(float(i[2][2])),
    )


def _geometry(parent: ET.Element, tag: str, filename: str) -> None:
    element = ET.SubElement(parent, tag)
    ET.SubElement(element, "origin", xyz="0 0 0", rpy="0 0 0")
    ET.SubElement(ET.SubElement(element, "geometry"), "mesh", filename=filename)


def write_urdf(folder: Path, asset: SimAsset, run_dir: Path) -> Path:
    """Write ``<id>.urdf`` and its meshes in ``folder``, so the folder can move as a whole."""
    meshes = folder / "meshes"
    meshes.mkdir(parents=True, exist_ok=True)
    name = asset.object_id
    # MuJoCo reads OBJ and not PLY, and this keeps the vertex colors for other readers.
    visual = cast(
        trimesh.Trimesh, trimesh.load(run_dir / asset.visual_mesh_path, force="mesh", process=False)
    )
    visual_file = meshes / f"{name}_visual.obj"
    visual.export(visual_file, file_type="obj", include_normals=False)

    robot = ET.Element("robot", name=name)
    # The defaults would drop a body with no joint, and move the mesh files' folder.
    ET.SubElement(
        ET.SubElement(robot, "mujoco"),
        "compiler",
        strippath="false",
        fusestatic="false",
        balanceinertia="false",
        discardvisual="true",
    )
    link = ET.SubElement(robot, "link", name=name)
    _inertial(link, asset)
    _geometry(link, "visual", f"meshes/{visual_file.name}")
    for k, relative in enumerate(asset.collision_mesh_paths):
        part = meshes / f"{name}_collision_{k}.obj"
        shutil.copyfile(run_dir / relative, part)
        _geometry(link, "collision", f"meshes/{part.name}")
    path = folder / f"{name}.urdf"
    ET.indent(robot)
    ET.ElementTree(robot).write(path, encoding="utf-8", xml_declaration=True)
    return path


def urdf_scene_spec(
    urdfs: dict[str, Path], assets: list[SimAsset], table: Plane, cfg: SettlingConfig
) -> Any:
    """An ``MjSpec`` of the scene built from the URDF files, one free body per asset, in order."""
    import mujoco  # type: ignore[import-untyped]  # heavy import

    spec = mujoco.MjSpec()
    spec.option.timestep = cfg.timestep_s
    spec.option.gravity = [0.0, 0.0, -cfg.gravity_m_s2]
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.noslip_iterations = cfg.noslip_iterations
    normal = np.asarray(table.normal, dtype=float)
    z = np.zeros(4)
    mujoco.mju_quatZ2Vec(z, normal)
    plane = spec.worldbody.add_geom()
    plane.type = mujoco.mjtGeom.mjGEOM_PLANE
    plane.size = [0.0, 0.0, 1.0]
    plane.friction = [0.0, 0.0, 0.0]
    plane.pos = (normal * table.offset_m).tolist()
    plane.quat = z.tolist()
    for asset in assets:
        name = asset.object_id
        child = mujoco.MjSpec.from_file(str(urdfs[name]))
        frame = spec.worldbody.add_frame()
        body = frame.attach_body(child.body(name), f"{name}/", "")
        body.add_freejoint()
        for geom in body.geoms:
            geom.friction = [float(asset.properties.friction_coefficient.nominal), 0.0, 0.0]
            geom.condim = 3
    return spec


def urdf_scene_model(
    urdfs: dict[str, Path], assets: list[SimAsset], table: Plane, cfg: SettlingConfig
) -> Any:
    return urdf_scene_spec(urdfs, assets, table, cfg).compile()


def settle_bodies(assets: list[SimAsset]) -> list[Body]:
    """The bodies ``settle_scene`` measures. Their shape comes from the model it is given."""
    return [
        Body(
            a.object_id,
            [],
            a.T_world_from_object.to_numpy(),
            a.properties.mass_kg.nominal,
            np.asarray(a.properties.inertia_kg_m2),
            np.zeros(3),
            a.properties.friction_coefficient.nominal,
        )
        for a in assets
    ]


def _bounds(model: Any, data: Any, body_id: int) -> F64:
    """World bounds ``[min, max]`` of a body's collision meshes in the compiled model."""
    corners = []
    for geom in np.flatnonzero(model.geom_bodyid == body_id):
        mesh = model.geom_dataid[geom]
        start = model.mesh_vertadr[mesh]
        vertices = model.mesh_vert[start : start + model.mesh_vertnum[mesh]]
        rotation = data.geom_xmat[geom].reshape(3, 3)
        corners.append(vertices @ rotation.T + data.geom_xpos[geom])
    points = np.concatenate(corners)
    return np.stack([points.min(axis=0), points.max(axis=0)])


def check_model(
    model: Any, assets: list[SimAsset], run_dir: Path, tolerance: float, tolerance_m: float
) -> list[str]:
    """What differs between the compiled model and ``assets``, at the observed poses."""
    import mujoco  # heavy import

    data = mujoco.MjData(model)
    for i, asset in enumerate(assets):
        t = asset.T_world_from_object.to_numpy()
        data.qpos[7 * i : 7 * i + 3] = t[:3, 3]
        mujoco.mju_mat2Quat(data.qpos[7 * i + 3 : 7 * i + 7], t[:3, :3].reshape(-1))
    mujoco.mj_forward(model, data)

    problems = []
    for i, asset in enumerate(assets):
        name, body_id = asset.object_id, i + 1
        p = asset.properties
        if not np.isclose(model.body_mass[body_id], p.mass_kg.nominal, rtol=tolerance):
            problems.append(f"{name}: mass in the URDF differs")
        moments = np.linalg.eigvalsh(np.asarray(p.inertia_kg_m2))
        compiled = np.sort(model.body_inertia[body_id])
        if not np.allclose(compiled, moments, rtol=tolerance, atol=0):
            problems.append(f"{name}: principal inertia in the URDF differs")
        t = asset.T_world_from_object.to_numpy()
        points = []
        for relative in asset.collision_mesh_paths:
            part = cast(trimesh.Trimesh, trimesh.load(run_dir / relative, force="mesh"))
            points.append(np.asarray(part.vertices) @ t[:3, :3].T + t[:3, 3])
        all_points = np.concatenate(points)
        expected = np.stack([all_points.min(axis=0), all_points.max(axis=0)])
        if not np.allclose(_bounds(model, data, body_id), expected, rtol=0, atol=tolerance_m):
            problems.append(f"{name}: collision bounds in the URDF differ")
    return problems


def settle_urdfs(
    model: Any, assets: list[SimAsset], table: Plane, cfg: SettlingConfig
) -> SceneResult:
    return settle_scene(settle_bodies(assets), table, cfg, model=model)


__all__ = [
    "check_model",
    "settle_bodies",
    "settle_urdfs",
    "urdf_scene_model",
    "urdf_scene_spec",
    "write_urdf",
]
