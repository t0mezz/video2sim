"""Ask the vision-language model for each object's mass and score it on the dev objects.

    uv run python benchmark/run_vlm_mass.py --out benchmark/results/p4_vlm_mass.json

The experiment behind D-024 (``stages/vlm_mass.py``); the pipeline uses the ``crops`` arm with the
rule. Two arms on the cached crops of the material pass: ``crops`` sends
the crops and the category, ``measured`` adds the measured size and sealed volume. ``blend`` needs
no call: the geometric mean of ``measured`` and the mass of P4.6, in sample like that mass.
``rule_held_out`` combines ``crops`` with the leave-one-object-out mass of P4.6.
Answers are cached in ``<run>/vlm_mass_cache``. One hosted request per object and arm, which needs
``BASETEN_API_KEY``. Needs ``benchmark/results/p4_6_mass.json`` and finished ``physics/`` runs.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
import yaml
from fit_fill_fractions import ratios_held_out

from video2sim.evaluation.alignment import align_scene
from video2sim.evaluation.loader import load_run
from video2sim.evaluation.metrics import evaluate_scene
from video2sim.evaluation.scenes import discover_scenes
from video2sim.models.hosted import ResponseCache, read_secret
from video2sim.runner import OUTPUT_FILE
from video2sim.schemas import Instances, SimAssets, WorldModel
from video2sim.stages.material_pass import BasetenMaterialClient
from video2sim.stages.vlm_mass import PROMPT, parse_mass

ROOT = Path(__file__).resolve().parent.parent
ARMS = ("crops", "measured")
MEASURED = """Measured from the video: size {x:.1f} x {y:.1f} x {z:.1f} cm, \
enclosed volume {volume:.0f} cm3.
"""


def crops_of(run_dir: Path, object_id: str, keyframes: list[int]) -> list[Path]:
    """The crops the material pass used. ``obj_N`` is track ``N`` when these files exist."""
    track = int(object_id.removeprefix("obj_"))
    work = run_dir / "segmentation.work" / "material_crops"
    paths = [work / f"track_{track:03d}_f{f:05d}.png" for f in keyframes[:2]]
    return [p for p in paths if p.exists()]


def rows_to_ask() -> list[dict[str, Any]]:
    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    marker = cfg["capture"]["marker"]
    base = json.loads((ROOT / "benchmark/results/p4_6_mass.json").read_text())["objects"]
    physics = {(r["scene"], r["object"]): r for r in base}
    rows: list[dict[str, Any]] = []
    scenes, _ = discover_scenes(ROOT / "benchmark/scenes", ("dev",), None)
    for scene in scenes:
        run_dir = next(iter(sorted((ROOT / "runs_instances").glob(f"{scene.scene_id}_*"))), None)
        if run_dir is None or not (run_dir / "physics" / OUTPUT_FILE).exists():
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
        instances = {
            o.object_id: o
            for o in Instances.model_validate_json(
                (run_dir / "segmentation" / OUTPUT_FILE).read_text()
            ).objects
        }
        assets = {
            a.object_id: a
            for a in SimAssets.model_validate_json(
                (run_dir / "physics" / OUTPUT_FILE).read_text()
            ).assets
        }
        for r in result.objects:
            if r.pred_id is None or (scene.scene_id, r.object_id) not in physics:
                continue
            base_row = physics[(scene.scene_id, r.object_id)]
            obj = instances[r.pred_id]
            mesh = trimesh.load(run_dir / assets[r.pred_id].visual_mesh_path, force="mesh")
            rows.append(
                {
                    "scene": scene.scene_id,
                    "object": r.object_id,
                    "label": r.label,
                    "pred_id": r.pred_id,
                    "run_dir": str(run_dir),
                    "category": obj.category or obj.label,
                    "crop_paths": [str(p) for p in crops_of(run_dir, r.pred_id, obj.keyframes)],
                    "extents_cm": sorted((100 * np.asarray(mesh.extents)).tolist(), reverse=True),
                    "volume_cm3": base_row["volume_sealed_m3"] * 1e6,
                    "mass_true_kg": base_row["mass_true_kg"],
                    "mass_physics_kg": base_row["mass_nominal_kg"],
                }
            )
    return rows


def ask(row: dict[str, Any], arm: str, client: BasetenMaterialClient) -> dict[str, Any]:
    x, y, z = row["extents_cm"]
    measured = MEASURED.format(x=x, y=y, z=z, volume=row["volume_cm3"]) if arm == "measured" else ""
    prompt = PROMPT.format(category=row["category"], measured=measured)
    images = [Path(p) for p in row["crop_paths"]]
    cache = ResponseCache(Path(row["run_dir"]) / "vlm_mass_cache")
    try:
        text = cache.get_or_call(
            [prompt, *[p.name for p in images]], lambda: client.ask(prompt, images)
        )
    except Exception as exc:  # one failed call is counted as a miss and listed
        return {"arm": arm, "error": f"{type(exc).__name__}: {exc}"}
    masses = parse_mass(text)
    if masses is None:
        return {"arm": arm, "error": f"unreadable answer: {text[:200]}"}
    return {"arm": arm, "low_kg": masses.low, "nominal_kg": masses.nominal, "high_kg": masses.high}


def score(pairs: list[tuple[float | None, float]], tolerance: float) -> dict[str, Any]:
    good = [(p, t) for p, t in pairs if p is not None]
    ratios = np.array([p / t for p, t in good])
    return {
        "objects": len(pairs),
        "answered": len(good),
        "within_tolerance": int((np.abs(ratios - 1) <= tolerance).sum()),
        "fraction": float((np.abs(ratios - 1) <= tolerance).sum()) / max(len(pairs), 1),
        "median_ratio": float(np.median(ratios)) if len(good) else None,
        "mean_mnre": float(np.minimum(ratios, 1 / ratios).mean()) if len(good) else None,
    }


def held_out_rule(rows: list[dict[str, Any]], tolerance: float) -> dict[str, Any]:
    """Score the combination with the leave-one-object-out physics mass of P4.6.

    The rule keeps the physics mass when it falls inside the model's range, else the model's
    nominal. The interval resamples distinct objects, because a repeated object is one sample.
    """
    base = json.loads((ROOT / "benchmark/results/p4_6_mass.json").read_text())["objects"]
    ratios = ratios_held_out(base, by_class=True)
    physics = {
        (r["scene"], r["object"]): q * r["mass_true_kg"] for r, q in zip(base, ratios, strict=True)
    }
    hits = []
    for r in rows:
        mass = physics[(r["scene"], r["object"])]
        a = r["crops"]
        chosen = mass if a["low_kg"] <= mass <= a["high_kg"] else a["nominal_kg"]
        r["mass_physics_held_out_kg"] = mass
        r["mass_rule_held_out_kg"] = chosen
        hits.append(abs(chosen / r["mass_true_kg"] - 1) <= tolerance)
    by_object: dict[str, list[bool]] = defaultdict(list)
    for r, hit in zip(rows, hits, strict=True):
        by_object[r["object"]].append(hit)
    groups = list(by_object.values())
    generator = np.random.default_rng(0)
    shares = []
    for _ in range(10000):
        picked = generator.integers(len(groups), size=len(groups))
        sample = [h for i in picked for h in groups[i]]
        shares.append(sum(sample) / len(sample))
    return {
        "physics_held_out": score(
            [(physics[(r["scene"], r["object"])], r["mass_true_kg"]) for r in rows], tolerance
        ),
        "rule_held_out": score(
            [(r["mass_rule_held_out_kg"], r["mass_true_kg"]) for r in rows], tolerance
        ),
        "rule_used_physics": sum(
            r["mass_rule_held_out_kg"] == r["mass_physics_held_out_kg"] for r in rows
        ),
        "rule_held_out_interval_90": [float(x) for x in np.percentile(shares, (5, 95))],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true", help="list the objects, call nothing")
    args = parser.parse_args()
    targets = yaml.safe_load((ROOT / "configs/targets.yaml").read_text(encoding="utf-8"))
    tolerance = float(targets["targets"]["mass_tolerance"])
    rows = rows_to_ask()
    print(
        len(rows),
        "objects;",
        sum(len(r["crop_paths"]) == 2 for r in rows),
        "with 2 crops",
        flush=True,
    )
    if args.dry_run:
        for r in rows[:6]:
            print(r["scene"], r["label"], r["pred_id"], r["category"], len(r["crop_paths"]))
        return
    client = BasetenMaterialClient(read_secret("BASETEN_API_KEY"), 120.0, 4)
    with ThreadPoolExecutor(args.workers) as pool:
        for arm in ARMS:
            for row, answer in zip(
                rows, pool.map(lambda r, a=arm: ask(r, a, client), rows), strict=True
            ):
                row[arm] = answer

    truth = [r["mass_true_kg"] for r in rows]
    summary: dict[str, Any] = {"tolerance": tolerance}
    summary["physics_in_sample"] = score(
        [(r["mass_physics_kg"], r["mass_true_kg"]) for r in rows], tolerance
    )
    for arm in ARMS:
        summary[arm] = score(
            [(r[arm].get("nominal_kg"), t) for r, t in zip(rows, truth, strict=True)], tolerance
        )
        inside = [r for r in rows if "nominal_kg" in r[arm]]
        summary[arm]["truth_inside_low_high"] = sum(
            r[arm]["low_kg"] <= r["mass_true_kg"] <= r[arm]["high_kg"] for r in inside
        )
    summary["blend_in_sample"] = score(
        [
            (
                float(np.sqrt(r["measured"]["nominal_kg"] * r["mass_physics_kg"]))
                if "nominal_kg" in r["measured"]
                else None,
                r["mass_true_kg"],
            )
            for r in rows
        ],
        tolerance,
    )
    summary.update(held_out_rule(rows, tolerance))
    args.out.write_text(json.dumps({"summary": summary, "objects": rows}, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
