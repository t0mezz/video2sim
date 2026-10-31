"""Settle the dev scenes under gravity in MuJoCo (P4.7, D-021).

    uv run python benchmark/run_settling.py --runs runs_instances \
        --out benchmark/results/p4_7_settling.json

Repairs every registered mesh, builds its collision parts and its mass, and caches them in
``<run>/physics.work/``, then runs ``settle_scene`` once for the nominal values and once for each
low and high friction and mass. A scene with an object that could not be prepared does not pass:
the object is listed and never dropped. Dev scenes only, no hosted call and no GPU.
"""

from __future__ import annotations

import argparse
import json
import pickle
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
import yaml

from video2sim.evaluation.scenes import discover_scenes
from video2sim.runner import OUTPUT_FILE
from video2sim.schemas import Instances, RegisteredObjects, WorldModel
from video2sim.stages.collision import CollisionConfig, collision_geometry
from video2sim.stages.mass import MassConfig, estimate_mass
from video2sim.stages.material_table import load_material_table
from video2sim.stages.mesh_repair import RepairConfig, repair_mesh
from video2sim.stages.physics import object_material
from video2sim.stages.settling import SettleBody, SettlingConfig, settle_scene

ROOT = Path(__file__).resolve().parent.parent
CASES = ("nominal", "friction_low", "friction_high", "mass_low", "mass_high")


def load_configs() -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    targets = yaml.safe_load((ROOT / "configs/targets.yaml").read_text(encoding="utf-8"))
    return cfg, targets


def prepare(run_dir: Path, object_id: str) -> dict[str, Any]:
    """Repair, collision and mass of one object, from the cache when it is there."""
    path = run_dir / "physics.work" / f"{object_id}.pkl"
    if path.exists():
        cached: dict[str, Any] = pickle.loads(path.read_bytes())
        return cached
    cfg, _ = load_configs()
    table = load_material_table(ROOT / cfg["physics"]["material_table"])
    min_conf = float(cfg["segmentation"]["material_pass"]["min_confidence"])
    instance = next(
        o
        for o in Instances.model_validate_json(
            (run_dir / "segmentation" / OUTPUT_FILE).read_text()
        ).objects
        if o.object_id == object_id
    )
    registered = next(
        o
        for o in RegisteredObjects.model_validate_json(
            (run_dir / "shape" / OUTPUT_FILE).read_text()
        ).objects
        if o.object_id == object_id
    )
    mesh = trimesh.load(run_dir / registered.mesh_path, force="mesh", process=False)
    repaired = repair_mesh(mesh, RepairConfig.from_config(cfg))
    collision = collision_geometry(repaired.mesh, CollisionConfig.from_config(cfg))
    material = object_material(instance, table, min_conf)
    mass = estimate_mass(repaired.mesh, material, MassConfig.from_config(cfg))
    out = {
        "object_id": object_id,
        "pose": registered.T_world_from_object.to_numpy(),
        "parts": [(np.asarray(p.vertices), np.asarray(p.faces)) for p in collision.parts],
        "kind": collision.kind,
        "checks": {**repaired.checks, **collision.checks},
        "mass_kg": mass.mass_kg.model_dump(),
        "inertia_kg_m2": mass.inertia_kg_m2,
        "center_of_mass_m": mass.center_of_mass_m,
        "friction": material.numbers.friction_coefficient.model_dump(),
    }
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(pickle.dumps(out))
    return out


def prepare_safely(job: tuple[Path, str]) -> dict[str, Any]:
    run_dir, object_id = job
    try:
        return prepare(run_dir, object_id)
    except Exception as exc:  # one bad mesh must be reported, not stop the other scenes
        return {"object_id": object_id, "error": f"{type(exc).__name__}: {exc}"}


def body(obj: dict[str, Any], case: str) -> SettleBody:
    mass = float(obj["mass_kg"]["nominal"])
    case_mass = float(obj["mass_kg"][case.removeprefix("mass_")]) if "mass_" in case else mass
    friction = obj["friction"][case.removeprefix("friction_") if "friction_" in case else "nominal"]
    return SettleBody(
        obj["object_id"],
        [trimesh.Trimesh(v, f, process=False) for v, f in obj["parts"]],
        obj["pose"],
        case_mass,
        obj["inertia_kg_m2"] * case_mass / mass,  # the body is the same, only denser
        obj["center_of_mass_m"],
        float(friction),
    )


def settle_case(job: tuple[str, list[dict[str, Any]], Any, SettlingConfig, str]) -> dict[str, Any]:
    scene_id, objects, table, cfg, case = job
    try:
        result = settle_scene([body(o, case) for o in objects], table, cfg)
    except ValueError as exc:
        return {"scene": scene_id, "case": case, "stable": False, "load_error": str(exc)}
    return {
        "scene": scene_id,
        "case": case,
        "stable": result.stable,
        "residual_penetration_m": result.residual_penetration_m,
        "bodies": [asdict(b) for b in result.bodies],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ROOT / "runs_instances")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--scene", action="append", help="limit to these scene ids")
    args = parser.parse_args()
    cfg, targets = load_configs()
    settling = SettlingConfig.from_config(cfg, targets)

    scenes, _ = discover_scenes(ROOT / "benchmark/scenes", ("dev",), None)
    prepared: dict[str, tuple[Path, list[str]]] = {}
    for scene in scenes:
        if args.scene and scene.scene_id not in args.scene:
            continue
        if scene.has_deformable:  # post-MVP, D-014
            continue
        run_dir = next(
            (
                r
                for r in sorted(args.runs.glob(f"{scene.scene_id}_*"))
                if (r / "shape" / OUTPUT_FILE).exists()
            ),
            None,
        )
        if run_dir is None:
            print(scene.scene_id, "has no shape run", flush=True)
            continue
        ids = [
            o.object_id
            for o in RegisteredObjects.model_validate_json(
                (run_dir / "shape" / OUTPUT_FILE).read_text()
            ).objects
        ]
        prepared[scene.scene_id] = (run_dir, ids)

    jobs = [(run_dir, i) for run_dir, ids in prepared.values() for i in ids]
    with ProcessPoolExecutor(args.workers) as pool:
        done = list(pool.map(prepare_safely, jobs))
    print("prepared", len(done), "objects", flush=True)
    by_key = {(j[0], d["object_id"]): d for j, d in zip(jobs, done, strict=True)}

    sims = []
    failed: dict[str, list[dict[str, str]]] = {}
    for scene_id, (run_dir, ids) in prepared.items():
        objects = [by_key[(run_dir, i)] for i in ids]
        failed[scene_id] = [
            {"object_id": o["object_id"], "error": o["error"]} for o in objects if "error" in o
        ]
        good = [o for o in objects if "error" not in o]
        table = WorldModel.model_validate_json(
            (run_dir / "reconstruction" / OUTPUT_FILE).read_text()
        ).table_plane
        sims += [(scene_id, good, table, settling, case) for case in CASES]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(settle_case, sims))
    for row in rows:
        row["stable"] = row["stable"] and not failed[row["scene"]]

    gate = float(targets["targets"]["stability_pass_fraction_min"])
    scene_ids = list(prepared)
    per_case = {
        case: sum(r["stable"] for r in rows if r["case"] == case) / max(len(scene_ids), 1)
        for case in CASES
    }
    every_case = [s for s in scene_ids if all(r["stable"] for r in rows if r["scene"] == s)]
    objects = [o for o in done if "error" not in o]
    summary = {
        "simulator": "mujoco (D-021)",
        "scenes": len(scene_ids),
        "objects": len(done),
        "objects_failed_preparation": sum("error" in o for o in done),
        "objects_all_checks_pass": sum(all(o["checks"].values()) for o in objects),
        "collision_kinds": {
            k: sum(o["kind"] == k for o in objects) for k in sorted({o["kind"] for o in objects})
        },
        "gate": gate,
        "pass_fraction": per_case,
        "scenes_stable_in_every_case": every_case,
    }
    checks = {
        f"{run_dir.name}/{d['object_id']}": {k: v for k, v in d["checks"].items() if not v}
        for (run_dir, _), d in by_key.items()
        if "error" not in d and not all(d["checks"].values())
    }
    payload = {"summary": summary, "failed_checks": checks, "failed": failed, "rows": rows}
    args.out.write_text(json.dumps(payload, indent=1, default=float) + "\n")
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
