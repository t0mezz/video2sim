"""Is a registered mesh a solid or a shell? Compare its volumes with the weighed truth (P4.5).

    uv run python benchmark/measure_mesh_volume.py --runs runs_instances \
        --out benchmark/results/p4_5_mesh_volume.json

For each matched dev object: the signed volume of the repaired mesh, the volume enclosed by its
outer surface (surface voxels flood-filled from outside), the convex hull, and the volume of the
measured primitive. Nominal table density times each volume is compared with the weighed mass.
Runs an analytic self-check first. No hosted call and no GPU.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
import yaml
from numpy.typing import NDArray
from scipy import ndimage

from video2sim.evaluation.alignment import align_scene
from video2sim.evaluation.loader import load_run
from video2sim.evaluation.metrics import evaluate_scene
from video2sim.evaluation.scenes import discover_scenes
from video2sim.runner import OUTPUT_FILE
from video2sim.schemas import Instances, RegisteredObjects, WorldModel
from video2sim.stages.material_table import load_material_table
from video2sim.stages.mesh_repair import RepairConfig, repair_mesh
from video2sim.stages.physics import object_material

ROOT = Path(__file__).resolve().parent.parent
GRID = 128  # voxels along the longest side
SAMPLES_PER_VOXEL_AREA = 8.0


def enclosed_volume(mesh: trimesh.Trimesh) -> tuple[float, float]:
    """Volume inside the outer surface, and the surface-voxel share of the bounding grid.

    Gaps up to two voxels are closed by a dilation, which the erosion after the fill takes back.
    """
    pitch = float(mesh.extents.max()) / GRID
    count = min(int(mesh.area / pitch**2 * SAMPLES_PER_VOXEL_AREA), 3_000_000)
    points = trimesh.sample.sample_surface(mesh, count, seed=0)[0]
    lo = mesh.bounds[0] - 3.5 * pitch
    index = np.floor((points - lo) / pitch).astype(int)
    shape = tuple(index.max(axis=0) + 4)
    grid: NDArray[np.bool_] = np.zeros(shape, dtype=bool)
    grid[index[:, 0], index[:, 1], index[:, 2]] = True
    closed = ndimage.binary_dilation(grid, iterations=1)
    filled = ndimage.binary_fill_holes(closed)
    solid = ndimage.binary_erosion(filled, iterations=1)
    return float(solid.sum()) * pitch**3, float(closed.sum()) / float(filled.sum())


def self_check() -> dict[str, float]:
    """Relative error of ``enclosed_volume`` on shapes with a known answer."""
    box = trimesh.creation.box((0.2, 0.1, 0.05))
    inner = trimesh.creation.box((0.19, 0.09, 0.04))
    inner.invert()
    shell = trimesh.util.concatenate([box, inner])
    sphere = trimesh.creation.icosphere(subdivisions=4, radius=0.05)
    return {
        "box": enclosed_volume(box)[0] / box.volume - 1,
        "thin_box_shell": enclosed_volume(shell)[0] / box.volume - 1,
        "sphere": enclosed_volume(sphere)[0] / (4 / 3 * math.pi * 0.05**3) - 1,
    }


def primitive_volume(gt: dict[str, Any]) -> tuple[float, str]:
    a, b, c = gt["size_m"]
    kind = gt.get("shape_primitive") or ""
    if kind in ("cylinder",):
        return math.pi / 4 * ((a + b) / 2) ** 2 * c, "cylinder"
    if kind in ("sphere",):
        return math.pi / 6 * a * b * c, "sphere"
    if kind in ("cuboid", "cube"):
        return a * b * c, "box"
    return a * b * c, "bounding box only"


def measure(path: Path, cfg: RepairConfig) -> dict[str, Any]:
    mesh = trimesh.load(path, force="mesh", process=False)
    repaired = repair_mesh(mesh, cfg)
    fill, surface_share = enclosed_volume(repaired.mesh)
    return {
        "faces": len(mesh.faces),
        "repair_ok": repaired.ok,
        "repair_steps": repaired.steps,
        "volume_mesh_m3": abs(float(repaired.mesh.volume)),
        "volume_fill_m3": fill,
        "volume_hull_m3": float(repaired.mesh.convex_hull.volume),
        "surface_share": surface_share,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ROOT / "runs_instances")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    repair_cfg = RepairConfig.from_config(cfg)
    table = load_material_table(ROOT / cfg["physics"]["material_table"])
    min_conf = float(cfg["segmentation"]["material_pass"]["min_confidence"])
    marker = cfg["capture"]["marker"]
    check = self_check()
    print("self check", {k: round(v, 3) for k, v in check.items()}, flush=True)

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
            gt, obj, reg = truth[r.object_id], instances[r.pred_id], registered[r.pred_id]
            row = measure(run_dir / reg.mesh_path, repair_cfg)
            volume_true, kind = primitive_volume(gt)
            density = object_material(obj, table, min_conf).numbers.density_kg_m3.nominal
            row |= {
                "scene": scene.scene_id,
                "object": r.object_id,
                "label": r.label,
                "geometry_source": reg.geometry_source,
                "size_error": r.size_error,
                "volume_true_m3": volume_true,
                "true_shape": kind,
                "mass_true_kg": gt["mass_kg"],
                "hollow": gt["hollow"],
                "density_kg_m3": density,
                "material_class": obj.material_class,
                "mass_from_mesh_kg": density * row["volume_mesh_m3"],
                "mass_from_fill_kg": density * row["volume_fill_m3"],
            }
            rows.append(row)
            print(
                scene.scene_id,
                r.label[:30],
                reg.geometry_source,
                f"mesh/true {row['volume_mesh_m3'] / volume_true:.2f}",
                f"fill/true {row['volume_fill_m3'] / volume_true:.2f}",
                f"hull/true {row['volume_hull_m3'] / volume_true:.2f}",
                flush=True,
            )

    def median(key: str, subset: list[dict[str, Any]]) -> float:
        return statistics.median(r[key] / r["volume_true_m3"] for r in subset)

    def within(key: str, subset: list[dict[str, Any]]) -> int:
        return sum(abs(r[key] / r["mass_true_kg"] - 1) <= 0.30 for r in subset)

    primitive = [r for r in rows if r["true_shape"] != "bounding box only"]
    solid = [r for r in rows if not r["hollow"]]
    summary = {
        "objects": len(rows),
        "repair_ok": sum(r["repair_ok"] for r in rows),
        "with_primitive_truth": len(primitive),
        "median_ratio_to_true_volume": {
            k: median(f"volume_{k}_m3", primitive) for k in ("mesh", "fill", "hull")
        },
        "mass_within_30pct_all": {
            "mesh": within("mass_from_mesh_kg", rows),
            "fill": within("mass_from_fill_kg", rows),
        },
        "mass_within_30pct_not_hollow": {
            "objects": len(solid),
            "mesh": within("mass_from_mesh_kg", solid),
            "fill": within("mass_from_fill_kg", solid),
        },
    }
    payload = {"self_check": check, "grid": GRID, "summary": summary, "objects": rows}
    args.out.write_text(json.dumps(payload, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
