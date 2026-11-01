"""The USD scene of stage 6 (P5.1), and the read-back that checks it.

Z-up, meters and kilograms are set on the stage, because USD's fallbacks are Y-up and
centimeters. Each object is a rigid body at its pose, with the origin at the center of mass
(D-023), a visual mesh and one convex collision mesh per part. Poses use translate and orient ops
and not a matrix, since USD matrices are the transpose of ``SE3`` and that mistake still loads.

The table is one infinite plane, which also serves as the ground: nothing in the scene says how
far below it a floor would be. It has zero friction and restitution and the combine mode is max,
the rule MuJoCo applies in ``stages/settling.py``, so an object on the table has its own values.
The combine mode is a PhysX attribute that ``usd-core`` has no schema for. Nothing here can run it,
so it stays unverified until the Isaac Sim verifier (D-027).

USD holds one nominal value per property, so the ranges go in ``video2sim:`` attributes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import trimesh
from numpy.typing import NDArray

from video2sim.schemas import Estimate, Plane, SimAsset

F64 = NDArray[np.float64]
ESTIMATES = ("mass_kg", "density_kg_m3", "friction_coefficient", "restitution")


def principal_inertia(inertia: F64) -> tuple[F64, F64]:
    """The principal moments and the wxyz quaternion of the axes, a proper rotation."""
    from scipy.spatial.transform import Rotation  # type: ignore[import-untyped]  # heavy import

    moments, axes = np.linalg.eigh(np.asarray(inertia, dtype=float))
    if np.linalg.det(axes) < 0:
        axes[:, 2] = -axes[:, 2]
    x, y, z, w = Rotation.from_matrix(axes).as_quat()
    return moments, np.array([w, x, y, z])


def linear_from_srgb(colors: NDArray[np.uint8]) -> F64:
    """RGB bytes as linear values: mesh colors are sRGB, and ``displayColor`` is linear."""
    c = np.asarray(colors, dtype=float)[:, :3] / 255.0
    return cast(F64, np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4))


def _load(path: Path) -> trimesh.Trimesh:
    return cast(trimesh.Trimesh, trimesh.load(path, force="mesh", process=False))


def _mesh_prim(stage: Any, path: str, mesh: trimesh.Trimesh) -> Any:
    from pxr import UsdGeom, Vt  # heavy import

    prim = UsdGeom.Mesh.Define(stage, path)
    prim.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    prim.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(mesh.vertices, dtype=np.float32)))
    prim.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(mesh.faces), 3, np.int32)))
    prim.CreateFaceVertexIndicesAttr(
        Vt.IntArray.FromNumpy(np.asarray(mesh.faces, dtype=np.int32).reshape(-1))
    )
    return prim


def _set_estimate(prim: Any, name: str, value: Estimate) -> None:
    from pxr import Sdf  # heavy import

    for end in ("low", "nominal", "high"):
        attr = prim.CreateAttribute(f"video2sim:{name}:{end}", Sdf.ValueTypeNames.Double)
        attr.Set(float(getattr(value, end)))


def _material(stage: Any, path: str, friction: float, restitution: float) -> Any:
    from pxr import Sdf, UsdPhysics, UsdShade  # heavy import

    material = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr(float(friction))
    api.CreateDynamicFrictionAttr(float(friction))
    api.CreateRestitutionAttr(float(restitution))
    prim = material.GetPrim()
    prim.AddAppliedSchema("PhysxMaterialAPI")
    for name in ("friction", "restitution"):
        attr = prim.CreateAttribute(f"physxMaterial:{name}CombineMode", Sdf.ValueTypeNames.Token)
        attr.Set("max")
    return material


def _bind(prim: Any, material: Any) -> None:
    from pxr import UsdShade  # heavy import

    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        material, UsdShade.Tokens.weakerThanDescendants, "physics"
    )


def _write_table(stage: Any, table: Plane, size_m: float) -> None:
    from pxr import Gf, UsdGeom, UsdPhysics  # heavy import
    from scipy.spatial.transform import Rotation  # heavy import

    normal = np.asarray(table.normal, dtype=float)
    plane = UsdGeom.Plane.Define(stage, "/World/Table")
    plane.CreateAxisAttr(UsdGeom.Tokens.z)
    plane.CreateWidthAttr(float(size_m))
    plane.CreateLengthAttr(float(size_m))
    x, y, z, w = Rotation.align_vectors([normal], [[0.0, 0.0, 1.0]])[0].as_quat()
    plane.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(*(normal * table.offset_m).tolist())
    )
    plane.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(w, x, y, z))
    UsdPhysics.CollisionAPI.Apply(plane.GetPrim())
    _bind(plane.GetPrim(), _material(stage, "/World/Materials/table", 0.0, 0.0))


def _write_object(stage: Any, asset: SimAsset, run_dir: Path) -> None:
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, Vt  # heavy import
    from scipy.spatial.transform import Rotation  # heavy import

    path = f"/World/Objects/{asset.object_id}"
    p = asset.properties
    t = asset.T_world_from_object.to_numpy()
    xform = UsdGeom.Xform.Define(stage, path)
    xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*t[:3, 3].tolist()))
    x, y, z, w = Rotation.from_matrix(t[:3, :3]).as_quat()
    xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(w, x, y, z))
    prim = xform.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)

    mass = UsdPhysics.MassAPI.Apply(prim)
    moments, axes = principal_inertia(np.asarray(p.inertia_kg_m2))
    mass.CreateMassAttr(float(p.mass_kg.nominal))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*moments.tolist()))
    mass.CreatePrincipalAxesAttr(Gf.Quatf(float(axes[0]), *axes[1:].tolist()))

    for name in ESTIMATES:
        _set_estimate(prim, name, getattr(p, name))
    prim.CreateAttribute("video2sim:material_class", Sdf.ValueTypeNames.String).Set(
        p.material_class
    )
    prim.CreateAttribute("video2sim:is_hollow", Sdf.ValueTypeNames.Bool).Set(p.is_hollow)
    prim.CreateAttribute("video2sim:confidence", Sdf.ValueTypeNames.Double).Set(p.confidence)
    prim.CreateAttribute("video2sim:flags", Sdf.ValueTypeNames.StringArray).Set(
        [f.value for f in asset.flags]
    )

    visual_mesh = _load(run_dir / asset.visual_mesh_path)
    visual = _mesh_prim(stage, f"{path}/visual", visual_mesh)
    if isinstance(visual_mesh.visual, trimesh.visual.ColorVisuals) and visual_mesh.visual.kind:
        colors = linear_from_srgb(np.asarray(visual_mesh.visual.vertex_colors))
        primvar = visual.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex)
        primvar.Set(Vt.Vec3fArray.FromNumpy(colors.astype(np.float32)))

    material = _material(
        stage,
        f"/World/Materials/{asset.object_id}",
        p.friction_coefficient.nominal,
        p.restitution.nominal,
    )
    UsdGeom.Scope.Define(stage, f"{path}/collision")
    for k, relative in enumerate(asset.collision_mesh_paths):
        part = _mesh_prim(stage, f"{path}/collision/part_{k}", _load(run_dir / relative))
        part.CreatePurposeAttr(UsdGeom.Tokens.guide)
        UsdPhysics.CollisionAPI.Apply(part.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(part.GetPrim()).CreateApproximationAttr("convexHull")
        _bind(part.GetPrim(), material)


def write_usd(
    path: Path,
    assets: list[SimAsset],
    run_dir: Path,
    table: Plane,
    gravity_m_s2: float,
    table_size_m: float,
) -> None:
    """Write the scene. ``assets`` must already exclude any asset that failed a check."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics  # heavy import

    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr(float(gravity_m_s2))
    _write_table(stage, table, table_size_m)
    for asset in assets:
        _write_object(stage, asset, run_dir)
    stage.GetRootLayer().Save()


@dataclass(frozen=True)
class UsdObject:
    """One object as read back from the file."""

    T_world_from_object: F64
    mass_kg: float
    inertia_kg_m2: F64
    center_of_mass_m: F64
    friction: float
    restitution: float
    ranges: dict[str, tuple[float, float, float]]
    collision_parts: int
    visual_vertices: F64
    has_color: bool


@dataclass(frozen=True)
class UsdScene:
    up_axis: str
    meters_per_unit: float
    kilograms_per_unit: float
    gravity_direction: F64
    gravity_m_s2: float
    objects: dict[str, UsdObject]


def _read_object(prim: Any) -> UsdObject:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade  # heavy import

    t = np.array(
        UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    ).T  # USD matrices are row-major with the translation in the last row
    mass = UsdPhysics.MassAPI(prim)
    q = mass.GetPrincipalAxesAttr().Get()
    axes = np.array(Gf.Matrix3d(Gf.Quatd(q.GetReal(), Gf.Vec3d(q.GetImaginary())))).T
    moments = np.array(mass.GetDiagonalInertiaAttr().Get(), dtype=float)
    parts = list(prim.GetChild("collision").GetChildren())
    binding = UsdShade.MaterialBindingAPI(parts[0]).GetDirectBinding("physics")
    material = UsdPhysics.MaterialAPI(binding.GetMaterial().GetPrim())
    visual = UsdGeom.Mesh(prim.GetChild("visual"))
    colors = visual.GetDisplayColorPrimvar().Get()
    return UsdObject(
        T_world_from_object=t,
        mass_kg=float(mass.GetMassAttr().Get()),
        inertia_kg_m2=axes @ np.diag(moments) @ axes.T,
        center_of_mass_m=np.array(mass.GetCenterOfMassAttr().Get(), dtype=float),
        friction=float(material.GetDynamicFrictionAttr().Get()),
        restitution=float(material.GetRestitutionAttr().Get()),
        ranges={
            name: cast(
                "tuple[float, float, float]",
                tuple(
                    float(prim.GetAttribute(f"video2sim:{name}:{end}").Get())
                    for end in ("low", "nominal", "high")
                ),
            )
            for name in ESTIMATES
        },
        collision_parts=len(parts),
        visual_vertices=np.array(visual.GetPointsAttr().Get(), dtype=float),
        has_color=colors is not None and len(colors) > 0,
    )


def read_usd(path: Path) -> UsdScene:
    from pxr import Usd, UsdGeom, UsdPhysics  # heavy import

    stage = Usd.Stage.Open(str(path))
    scene = UsdPhysics.Scene(stage.GetPrimAtPath("/World/PhysicsScene"))
    objects = stage.GetPrimAtPath("/World/Objects")
    children = objects.GetChildren() if objects else []
    return UsdScene(
        up_axis=str(UsdGeom.GetStageUpAxis(stage)),
        meters_per_unit=float(UsdGeom.GetStageMetersPerUnit(stage)),
        kilograms_per_unit=float(UsdPhysics.GetStageKilogramsPerUnit(stage)),
        gravity_direction=np.array(scene.GetGravityDirectionAttr().Get(), dtype=float),
        gravity_m_s2=float(scene.GetGravityMagnitudeAttr().Get()),
        objects={p.GetName(): _read_object(p) for p in children},
    )


def check_usd(
    path: Path, assets: list[SimAsset], gravity_m_s2: float, tolerance: float, tolerance_m: float
) -> list[str]:
    """What differs between the file and ``assets``; empty when the scene reads back correctly."""
    scene = read_usd(path)
    problems = []
    if scene.up_axis != "Z" or scene.meters_per_unit != 1.0 or scene.kilograms_per_unit != 1.0:
        problems.append(
            f"stage is {scene.up_axis}-up, {scene.meters_per_unit} m, "
            f"{scene.kilograms_per_unit} kg per unit"
        )
    if not np.allclose(scene.gravity_direction, [0, 0, -1]) or not np.isclose(
        scene.gravity_m_s2, gravity_m_s2, rtol=tolerance
    ):
        problems.append("gravity differs")
    if set(scene.objects) != {a.object_id for a in assets}:
        problems.append("objects in the file differ from the exported assets")
        return problems
    for asset in assets:
        read = scene.objects[asset.object_id]
        p = asset.properties
        name = asset.object_id
        size = np.abs(np.linalg.eigvalsh(np.asarray(p.inertia_kg_m2))).max()
        expected = {
            "mass": (read.mass_kg, p.mass_kg.nominal),
            "friction": (read.friction, p.friction_coefficient.nominal),
            "restitution": (read.restitution, p.restitution.nominal),
        }
        problems += [
            f"{name}: {key} reads {got}, expected {want}"
            for key, (got, want) in expected.items()
            if not np.isclose(got, want, rtol=tolerance, atol=1e-7)
        ]
        t = asset.T_world_from_object.to_numpy()
        if not np.allclose(read.T_world_from_object, t, rtol=0, atol=tolerance_m):
            problems.append(f"{name}: pose differs")
        if not np.allclose(
            read.inertia_kg_m2, np.asarray(p.inertia_kg_m2), rtol=0, atol=tolerance * size
        ):
            problems.append(f"{name}: inertia differs")
        if np.abs(read.center_of_mass_m).max() > tolerance_m:
            problems.append(f"{name}: center of mass is not at the origin")
        if read.collision_parts != len(asset.collision_mesh_paths):
            problems.append(f"{name}: collision parts differ")
    return problems


__all__ = [
    "ESTIMATES",
    "UsdObject",
    "UsdScene",
    "check_usd",
    "linear_from_srgb",
    "principal_inertia",
    "read_usd",
    "write_usd",
]
