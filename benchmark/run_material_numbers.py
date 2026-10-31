"""Check the P4.4 material mapping against the weighed inventory on finished runs.

    uv run python benchmark/run_material_numbers.py --runs runs_instances \
        --out benchmark/results/p4_4_material_numbers.json

Reads ``<runs>/<run id>/`` per dev scene, matches the meshes to the ground truth the way ``eval``
does, and compares the model's class and hollow answer with the inventory. Clean-board scenes
without an established origin (s05) cannot be matched and are listed as skipped. No hosted call.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from video2sim.evaluation.alignment import align_scene
from video2sim.evaluation.loader import load_run
from video2sim.evaluation.metrics import evaluate_scene
from video2sim.evaluation.scenes import discover_scenes
from video2sim.runner import OUTPUT_FILE
from video2sim.schemas import Instances, WorldModel
from video2sim.stages.material_table import load_material_table
from video2sim.stages.physics import object_material

ROOT = Path(__file__).resolve().parent.parent

# Inventory wording to the table keys an answer may use. Inventory "metal" does not say which.
INVENTORY_CLASSES = {
    "hard plastic": {"plastic_hard"},
    "PLA": {"plastic_hard"},
    "plastic": {"plastic_hard"},
    "plastic mesh": {"plastic_hard"},
    "metal": {"metal_steel", "metal_aluminium"},
    "cardboard": {"cardboard"},
    "fabric": {"textile"},
    "basket fabric": {"textile"},
    "toy faux fur and wool": {"textile"},
    "aluminium foil": {"foil_wrapped_food", "metal_aluminium"},
    "glass": {"glass"},
    "ceramic": {"ceramic"},
    "wood": {"wood"},
    "foam": {"foam"},
    "paper": {"paper"},
    "rubber": {"rubber"},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ROOT / "runs_instances")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    table = load_material_table(ROOT / cfg["physics"]["material_table"])
    min_confidence = float(cfg["segmentation"]["material_pass"]["min_confidence"])
    marker = cfg["capture"]["marker"]

    rows: list[dict[str, Any]] = []
    skipped: dict[str, str] = {}
    flags: Counter[str] = Counter()
    scenes, _ = discover_scenes(ROOT / "benchmark/scenes", ("dev",), None)
    for scene in scenes:
        runs = sorted(args.runs.glob(f"{scene.scene_id}_*"))
        run_dir = next((r for r in runs if (r / "shape" / OUTPUT_FILE).exists()), None)
        if run_dir is None:
            skipped[scene.scene_id] = "no shape output"
            continue
        world = WorldModel.model_validate_json(
            (run_dir / "reconstruction" / OUTPUT_FILE).read_text()
        )
        instances = Instances.model_validate_json(
            (run_dir / "segmentation" / OUTPUT_FILE).read_text()
        )
        materials = {
            o.object_id: (o, object_material(o, table, min_confidence)) for o in instances.objects
        }
        flags.update("low_confidence" if m.flags else "ok" for _, m in materials.values())
        alignment = align_scene(
            world,
            run_dir,
            scene.scene_id,
            scene.board,
            float(marker["tag_size_m"]),
            marker["family"],
        )
        if alignment is None:
            skipped[scene.scene_id] = "clean board: origin of the ground truth is not established"
            continue
        result = evaluate_scene(scene, load_run(run_dir, alignment.T_board_from_world))
        raw = json.loads(
            (ROOT / "benchmark/scenes" / scene.scene_id / "ground_truth.json").read_text()
        )
        truth = {o["object_id"]: o for o in raw["objects"]}
        for r in result.objects:
            if r.pred_id is None or r.pred_id not in materials:
                continue
            obj, material = materials[r.pred_id]
            gt = truth[r.object_id]
            allowed = INVENTORY_CLASSES[gt["material_class"]]
            rows.append(
                {
                    "scene": scene.scene_id,
                    "object": r.object_id,
                    "label": r.label,
                    "inventory_class": gt["material_class"],
                    "predicted_class": obj.material_class,
                    "class_ok": obj.material_class in allowed,
                    "inventory_hollow": gt["hollow"],
                    "predicted_hollow": obj.hollow,
                    "hollow_ok": material.is_hollow == gt["hollow"],
                    "confidence": obj.material_confidence,
                    "reasons": list(material.reasons),
                }
            )
    n = len(rows)
    summary = {
        "objects": n,
        "class_ok": sum(r["class_ok"] for r in rows),
        "hollow_ok": sum(r["hollow_ok"] for r in rows),
        "flags_all_run_objects": dict(flags),
        "skipped": skipped,
    }
    args.out.write_text(json.dumps({"summary": summary, "objects": rows}, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    for r in rows:
        if not r["class_ok"] or not r["hollow_ok"]:
            print(
                r["scene"],
                r["label"],
                r["inventory_class"],
                r["inventory_hollow"],
                "->",
                r["predicted_class"],
                r["predicted_hollow"],
                r["confidence"],
            )


if __name__ == "__main__":
    main()
