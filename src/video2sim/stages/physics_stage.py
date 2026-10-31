"""Stage 5 as a runner stage (P4.8): ``RegisteredObjects`` in, ``SimAssets`` out.

Each object goes through repair (P4.1), collision geometry (P4.2), its material (P4.4) and mass
(P4.5, P4.6), which the vision-language model's mass may replace (D-024,
``stages/vlm_mass.py``); the body stays, so inertia scales with the mass. The body gives the
center of mass, and the asset's frame moves there: meshes shift by it and
``T_world_from_object`` takes it up, so the world surface stays where it was and ``AGENTS.md``'s
object frame holds. Inertia is about that point already.

The scene is then settled (P4.7, D-021) for the nominal values and each low and high friction and
mass. The result goes to ``physics/settling.json`` for the QA report and does not change
``SimAsset.checks``: an object that drifts stays in the scene, and the report says so (D-023). An
object that fails a check is written with ``Flag.FAILED_VALIDATION``, and export blocks it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import trimesh

from video2sim.models.hosted import ResponseCache, read_secret
from video2sim.runner import StageContext
from video2sim.schemas import (
    SE3,
    Estimate,
    Flag,
    Instances,
    ObjectInstance,
    PhysicalProperties,
    Plane,
    RegisteredObject,
    RegisteredObjects,
    SimAsset,
    SimAssets,
    StageName,
    WorldModel,
)
from video2sim.stages.collision import CollisionConfig, collision_geometry, write_collision
from video2sim.stages.mass import MassConfig, estimate_mass
from video2sim.stages.material_pass import (
    BasetenMaterialClient,
    MaterialClient,
    crop_for_vlm,
    pick_frames,
)
from video2sim.stages.material_table import MaterialTable, load_material_table
from video2sim.stages.mesh_repair import RepairConfig, repair_mesh
from video2sim.stages.physics import object_material
from video2sim.stages.settling import SettleBody, SettlingConfig, is_round, settle_scene
from video2sim.stages.shape import load_masks
from video2sim.stages.tracking import Track
from video2sim.stages.vlm_mass import VlmMassConfig, ask_mass, combine

CASES = ("nominal", "friction_low", "friction_high", "mass_low", "mass_high")


@dataclass(frozen=True)
class PhysicsConfig:
    repair: RepairConfig
    collision: CollisionConfig
    mass: MassConfig
    settling: SettlingConfig
    vlm_mass: VlmMassConfig
    min_confidence: float

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> PhysicsConfig:
        """``cfg`` is what ``physics_stage_config`` builds."""
        full: dict[str, Any] = {"physics": cfg["physics"]}
        return cls(
            repair=RepairConfig.from_config(full),
            collision=CollisionConfig.from_config(full),
            mass=MassConfig.from_config(full),
            settling=SettlingConfig.from_config(full, {"targets": cfg["stability"]}),
            vlm_mass=VlmMassConfig.from_config(full),
            min_confidence=float(cfg["material_min_confidence"]),
        )


def physics_stage_config(cfg: Mapping[str, Any], targets: Mapping[str, Any]) -> dict[str, Any]:
    """The physics stage's config section, with the numbers it needs from elsewhere."""
    t = targets["targets"]
    return {
        "physics": cfg["physics"],
        "material_min_confidence": cfg["segmentation"]["material_pass"]["min_confidence"],
        "stability": {k: t[k] for k in ("stability_window_s", "stability_max_drift_m")},
    }


@dataclass(frozen=True)
class PreparedObject:
    asset: SimAsset
    parts: list[trimesh.Trimesh]  # in the asset's frame


def _flags(*groups: tuple[Flag, ...] | list[Flag]) -> list[Flag]:
    return list(dict.fromkeys(flag for group in groups for flag in group))


def _carry_colors(mesh: trimesh.Trimesh, source: trimesh.Trimesh) -> None:
    """Give ``mesh`` the color of the nearest ``source`` vertex; repair does not keep colors."""
    from scipy.spatial import cKDTree  # type: ignore[import-untyped]  # heavy import

    if not isinstance(source.visual, trimesh.visual.ColorVisuals) or not source.visual.kind:
        return
    nearest = cKDTree(np.asarray(source.vertices)).query(np.asarray(mesh.vertices))[1]
    colors = np.asarray(source.visual.vertex_colors)[nearest]
    mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=colors)


def prepare_object(
    registered: RegisteredObject,
    instance: ObjectInstance,
    table: MaterialTable,
    model_mass: Estimate | None,
    run_dir: Path,
    out_dir: Path,
    cfg: PhysicsConfig,
) -> tuple[PreparedObject, dict[str, Any]]:
    """One object's files and asset, and its entry in ``physics/report.json``."""
    mesh = cast(
        trimesh.Trimesh, trimesh.load(run_dir / registered.mesh_path, force="mesh", process=False)
    )
    repaired = repair_mesh(mesh, cfg.repair)
    collision = collision_geometry(repaired.mesh, cfg.collision)
    material = object_material(instance, table, cfg.min_confidence)
    mass = estimate_mass(repaired.mesh, material, cfg.mass)
    final_mass, mass_source = combine(mass.mass_kg, model_mass, cfg.vlm_mass.enabled)
    inertia = mass.inertia_kg_m2 * final_mass.nominal / mass.mass_kg.nominal  # same body

    offset = mass.center_of_mass_m
    visual = repaired.mesh.copy()
    _carry_colors(visual, mesh)
    visual.apply_translation(-offset)
    parts = [p.copy() for p in collision.parts]
    for part in parts:
        part.apply_translation(-offset)
    T_object_from_asset = np.eye(4)
    T_object_from_asset[:3, 3] = offset
    pose = registered.T_world_from_object.to_numpy() @ T_object_from_asset

    meshes = out_dir / "meshes"
    meshes.mkdir(exist_ok=True)
    visual_path = meshes / f"{registered.object_id}.ply"
    visual.export(visual_path, file_type="ply", encoding="binary")
    written = write_collision(
        replace(collision, parts=parts), out_dir / "collision", registered.object_id
    )

    checks = {**repaired.checks, **collision.checks}
    failed = (Flag.FAILED_VALIDATION,) if not all(checks.values()) else ()
    no_answer = (Flag.LOW_CONFIDENCE_MATERIAL,) if mass_source == "geometry_no_answer" else ()
    round_flag = (Flag.IGNORED_ROUND,) if is_round(visual, cfg.settling.round_tolerance) else ()
    numbers = material.numbers
    asset = SimAsset(
        object_id=registered.object_id,
        T_world_from_object=SE3.from_numpy(pose),
        visual_mesh_path=f"physics/meshes/{visual_path.name}",
        collision_mesh_paths=[f"physics/collision/{p.name}" for p in written],
        properties=PhysicalProperties(
            material_class=numbers.material_class,
            is_hollow=material.is_hollow,
            mass_kg=final_mass,
            density_kg_m3=numbers.density_kg_m3,
            friction_coefficient=numbers.friction_coefficient,
            restitution=numbers.restitution,
            inertia_kg_m2=inertia.tolist(),
            confidence=material.confidence,
        ),
        checks=checks,
        flags=_flags(registered.flags, material.flags, no_answer, round_flag, failed),
    )
    entry = {
        "collision_kind": collision.kind,
        "collision_parts": len(parts),
        "repair_steps": repaired.steps,
        "collision_steps": collision.steps,
        "fill_fraction": mass.fill_fraction.nominal,
        "mass_source": mass_source,
        "mass_geometry_kg": mass.mass_kg.model_dump(),
        "mass_model_kg": None if model_mass is None else model_mass.model_dump(),
        "center_of_mass_offset_m": float(np.linalg.norm(offset)),
        "reasons": [
            *material.reasons,
            *mass.reasons,
            *(["no mass from the model: kept the geometry mass"] if no_answer else []),
        ],
    }
    return PreparedObject(asset, parts), entry


def model_mass(
    instance: ObjectInstance,
    world: WorldModel,
    client: MaterialClient,
    run_dir: Path,
    work: Path,
    cfg: VlmMassConfig,
) -> Estimate | None:
    """The model's mass for one object; a failed call counts as no answer and is flagged."""
    track = Track(int(instance.object_id.removeprefix("obj_")), load_masks(run_dir, instance))
    crops = [
        crop_for_vlm(
            run_dir,
            world,
            track,
            frame,
            cfg.crop_pad_px,
            work / "vlm_mass_crops" / f"{instance.object_id}_f{frame:05d}.png",
        )
        for frame in pick_frames(track, instance.keyframes, cfg.keyframes)
    ]
    cache = ResponseCache(work / "vlm_mass_cache")
    try:
        return ask_mass(client, cache, instance.category or instance.label, crops)
    except (OSError, ValueError, KeyError, RuntimeError):
        return None


def settle_body(item: PreparedObject, case: str) -> SettleBody:
    """``item`` with the friction or mass of ``case``, one of ``CASES``."""
    p = item.asset.properties
    mass = p.mass_kg.nominal
    friction = p.friction_coefficient.nominal
    if case.startswith("mass_"):
        mass = getattr(p.mass_kg, case.removeprefix("mass_"))
    elif case.startswith("friction_"):
        friction = getattr(p.friction_coefficient, case.removeprefix("friction_"))
    return SettleBody(
        item.asset.object_id,
        item.parts,
        item.asset.T_world_from_object.to_numpy(),
        mass,
        np.asarray(p.inertia_kg_m2) * mass / p.mass_kg.nominal,  # same body, so linear in mass
        np.zeros(3),  # the asset's origin is the center of mass
        friction,
        Flag.IGNORED_ROUND in item.asset.flags,
    )


def settling_report(
    items: list[PreparedObject], table: Plane, cfg: SettlingConfig
) -> dict[str, dict[str, Any]]:
    """Each case of ``CASES``: stable, and the bodies, or the error if MuJoCo refused the scene."""
    report: dict[str, dict[str, Any]] = {}
    for case in CASES:
        try:
            result = settle_scene([settle_body(i, case) for i in items], table, cfg)
        except ValueError as exc:
            report[case] = {"stable": False, "load_error": str(exc)}
            continue
        report[case] = {
            "stable": result.stable,
            "residual_penetration_m": result.residual_penetration_m,
            "bodies": [asdict(b) for b in result.bodies],
        }
    return report


class PhysicsStage:
    """Runner stage. Its config comes from ``physics_stage_config``."""

    name: StageName = "physics"
    output_type = SimAssets
    dependencies: tuple[StageName, ...] = ("reconstruction", "segmentation", "shape")

    def __init__(self, material_table: Path, client: MaterialClient | None = None):
        """``client`` is the hosted model; by default it reads ``BASETEN_API_KEY``."""
        self._table = material_table
        self._client = client
        self.extra_inputs: tuple[Path, ...] = (material_table,)

    def run(self, ctx: StageContext) -> SimAssets:
        world = ctx.load("reconstruction", WorldModel)
        instances = {o.object_id: o for o in ctx.load("segmentation", Instances).objects}
        registered = ctx.load("shape", RegisteredObjects)
        cfg = PhysicsConfig.from_config(ctx.config)
        table = load_material_table(self._table)
        client = self._client
        if client is None and cfg.vlm_mass.enabled:
            client = BasetenMaterialClient(
                read_secret("BASETEN_API_KEY"), cfg.vlm_mass.timeout_s, cfg.vlm_mass.retries
            )
        work = ctx.run_dir / "physics.work"
        work.mkdir(exist_ok=True)

        items, report = [], {}
        for obj in registered.objects:
            instance = instances[obj.object_id]
            answer = None
            if client is not None and cfg.vlm_mass.enabled:
                answer = model_mass(instance, world, client, ctx.run_dir, work, cfg.vlm_mass)
            item, entry = prepare_object(
                obj, instance, table, answer, ctx.run_dir, ctx.out_dir, cfg
            )
            items.append(item)
            report[obj.object_id] = entry
        (ctx.out_dir / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (ctx.out_dir / "settling.json").write_text(
            json.dumps(settling_report(items, world.table_plane, cfg.settling), indent=1) + "\n",
            encoding="utf-8",
        )
        return SimAssets(assets=[i.asset for i in items])


__all__ = [
    "CASES",
    "PhysicsConfig",
    "PhysicsStage",
    "PreparedObject",
    "model_mass",
    "physics_stage_config",
    "prepare_object",
    "settle_body",
    "settling_report",
]
