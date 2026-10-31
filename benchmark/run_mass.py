"""Score the P4.5 mass estimate against the weighed inventory on finished runs.

    uv run python benchmark/run_mass.py --runs runs_instances \
        --out benchmark/results/p4_5_mass.json

Repairs each matched dev mesh, runs ``estimate_mass`` and compares it with the weighed mass, with
and without the hollow correction. ``fill_needed`` is the ``f`` that would make each closed
hollow object exact at nominal density; its percentiles are the source of
``hollow_fill_fraction`` in ``configs/material_table.yaml``. No hosted call and no GPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
import yaml

from video2sim.evaluation.alignment import align_scene
from video2sim.evaluation.loader import load_run
from video2sim.evaluation.metrics import evaluate_scene
from video2sim.evaluation.scenes import discover_scenes
from video2sim.runner import OUTPUT_FILE
from video2sim.schemas import Instances, RegisteredObjects, WorldModel
from video2sim.stages.mass import MassConfig, estimate_mass
from video2sim.stages.material_table import load_material_table
from video2sim.stages.mesh_repair import RepairConfig, repair_mesh
from video2sim.stages.physics import object_material

ROOT = Path(__file__).resolve().parent.parent


def score(rows: list[dict[str, Any]], tolerance: float) -> dict[str, Any]:
    ratios = [r["mass_nominal_kg"] / r["mass_true_kg"] for r in rows]
    uncorrected = [r["mass_uncorrected_kg"] / r["mass_true_kg"] for r in rows]
    return {
        "objects": len(rows),
        "median_ratio": float(np.median(ratios)) if rows else None,
        "within_tolerance": sum(abs(x - 1) <= tolerance for x in ratios),
        "within_tolerance_uncorrected": sum(abs(x - 1) <= tolerance for x in uncorrected),
        "truth_inside_low_high": sum(
            r["mass_low_kg"] <= r["mass_true_kg"] <= r["mass_high_kg"] for r in rows
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ROOT / "runs_instances")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    targets = yaml.safe_load((ROOT / "configs/targets.yaml").read_text(encoding="utf-8"))
    tolerance = float(targets["targets"]["mass_tolerance"])
    repair_cfg = RepairConfig.from_config(cfg)
    mass_cfg = MassConfig.from_config(cfg)
    table = load_material_table(ROOT / cfg["physics"]["material_table"])
    min_conf = float(cfg["segmentation"]["material_pass"]["min_confidence"])
    marker = cfg["capture"]["marker"]

    rows: list[dict[str, Any]] = []
    scenes, _ = discover_scenes(ROOT / "benchmark/scenes", ("dev",), None)
    for scene in scenes:
        run_dir = next(
            (
                r
                for r in sorted(args.runs.glob(f"{scene.scene_id}_*"))
                if (r / "shape" / OUTPUT_FILE).exists()
            ),
            None,
        )
        if run_dir is None:
            continue
        world = WorldModel.model_validate_json(
            (run_dir / "reconstruction" / OUTPUT_FILE).read_text()
        )
        alignment = align_scene(
            world,
            run_dir,
            scene.scene_id,
            scene.board,
            float(marker["tag_size_m"]),
            marker["family"],
        )
        if alignment is None:
            continue
        result = evaluate_scene(scene, load_run(run_dir, alignment.T_board_from_world))
        raw = json.loads(
            (ROOT / "benchmark/scenes" / scene.scene_id / "ground_truth.json").read_text()
        )
        truth = {o["object_id"]: o for o in raw["objects"]}
        instances = {
            o.object_id: o
            for o in Instances.model_validate_json(
                (run_dir / "segmentation" / OUTPUT_FILE).read_text()
            ).objects
        }
        registered = {
            o.object_id: o
            for o in RegisteredObjects.model_validate_json(
                (run_dir / "shape" / OUTPUT_FILE).read_text()
            ).objects
        }
        for r in result.objects:
            if r.pred_id is None:
                continue
            gt, reg = truth[r.object_id], registered[r.pred_id]
            material = object_material(instances[r.pred_id], table, min_conf)
            mesh = trimesh.load(run_dir / reg.mesh_path, force="mesh", process=False)
            est = estimate_mass(
                repair_mesh(mesh, repair_cfg).mesh, material, table.hollow_fill_fraction, mass_cfg
            )
            rho = material.numbers.density_kg_m3.nominal
            mass_true = float(gt["mass_kg"])
            rows.append(
                {
                    "scene": scene.scene_id,
                    "object": r.object_id,
                    "label": r.label,
                    "material_class": material.numbers.material_class,
                    "model_hollow": material.is_hollow,
                    "inventory_hollow": gt["hollow"],
                    "is_open": est.is_open,
                    "size_error": r.size_error,
                    "mass_true_kg": mass_true,
                    "mass_low_kg": est.mass_kg.low,
                    "mass_nominal_kg": est.mass_kg.nominal,
                    "mass_high_kg": est.mass_kg.high,
                    "mass_uncorrected_kg": rho * est.volume_sealed_m3,
                    "fill_fraction": est.fill_fraction.nominal,
                    "fill_needed": mass_true / (rho * est.volume_sealed_m3),
                    "volume_sealed_m3": est.volume_sealed_m3,
                    "center_of_mass_offset_m": float(np.linalg.norm(est.center_of_mass_m)),
                    "inertia_principal_kg_m2": np.linalg.eigvalsh(est.inertia_kg_m2).tolist(),
                    "reasons": list(est.reasons),
                }
            )
            print(
                scene.scene_id,
                r.label[:30],
                f"m {est.mass_kg.nominal:.3f} true {mass_true:.3f}",
                f"f {est.fill_fraction.nominal:.2f}",
                flush=True,
            )

    corrected = [r for r in rows if r["model_hollow"] and not r["is_open"]]
    needed = [r["fill_needed"] for r in corrected]
    p10, p50, p90 = np.percentile(needed, (10, 50, 90)) if needed else (None, None, None)
    summary = {
        "all": score(rows, tolerance),
        "hollow_corrected": score(corrected, tolerance),
        "hollow_open": score([r for r in rows if r["model_hollow"] and r["is_open"]], tolerance),
        "solid": score([r for r in rows if not r["model_hollow"]], tolerance),
        "fill_needed_percentiles": {"p10": p10, "p50": p50, "p90": p90},
        "center_of_mass_offset_median_m": float(
            np.median([r["center_of_mass_offset_m"] for r in rows])
        ),
    }
    payload = {"tolerance": tolerance, "summary": summary, "objects": rows}
    args.out.write_text(json.dumps(payload, indent=1, default=float) + "\n")
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
