"""Compare shape backends on the same scenes (P3.3 bake-off).

    uv run python benchmark/compare_shape_arms.py --out benchmark/results/p3_3_bakeoff_dev.json \
        sam3d=runs_instances trellis2=runs_trellis2 hybrid=runs_hybrid

Each ``name=folder`` holds ``<scene>_b`` run folders with a finished shape stage. For every arm
the script runs ``video2sim eval --stage shape`` (via symlinks named by scene id), then reads the
reports and adds one measure the harness lacks: the distance from the observed cluster
(``point_cluster_path``, metric points) to the registered mesh surface, which is what P3.2
fits. Objects are compared only where every arm matched the same ground-truth object, so the
medians are paired. A proxy hull counts as that arm's answer, since that is what the stage emits.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from video2sim.schemas import Instances, RegisteredObjects
from video2sim.stages.registration import sample_surface

ROOT = Path(__file__).resolve().parent.parent
SURFACE_SAMPLES = 20000
NEAR_M = 0.005


def run_eval(arm: str, folder: Path, split: str, scenes: list[str], tmp: Path) -> dict[str, Any]:
    links = tmp / f"runs_{arm}"
    links.mkdir()
    for scene in scenes:
        (links / scene).symlink_to((folder / f"{scene}_b").resolve())
    out = tmp / f"eval_{arm}"
    cmd = ["uv", "run", "video2sim", "eval", "--runs", str(links), "--split", split]
    cmd += ["--stage", "shape", "--out", str(out)]
    subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True)
    report: dict[str, Any] = json.loads((out / "report.json").read_text())
    return report


def cluster_to_mesh(run_dir: Path, pred_id: str) -> tuple[float, float]:
    """Median distance (m) from the observed cluster to the mesh, and the share within NEAR_M."""
    registered = RegisteredObjects.model_validate_json(
        (run_dir / "shape" / "output.json").read_text(encoding="utf-8")
    )
    instances = Instances.model_validate_json(
        (run_dir / "segmentation" / "output.json").read_text(encoding="utf-8")
    )
    obj = next(o for o in registered.objects if o.object_id == pred_id)
    inst = next(o for o in instances.objects if o.object_id == pred_id)
    mesh = trimesh.load(run_dir / obj.mesh_path, process=False)
    T = np.asarray(obj.T_world_from_object.matrix, dtype=float)
    vertices = np.asarray(mesh.vertices) @ T[:3, :3].T + T[:3, 3]
    surface = sample_surface(
        vertices, np.asarray(mesh.faces), SURFACE_SAMPLES, np.random.default_rng(0)
    )
    cluster = np.asarray(np.load(run_dir / inst.point_cluster_path), dtype=float).reshape(-1, 3)
    d = cKDTree(surface).query(cluster)[0]
    return float(np.median(d)), float((d <= NEAR_M).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arms", nargs="+", help="name=folder")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    arms = {a.split("=", 1)[0]: ROOT / a.split("=", 1)[1] for a in args.arms}

    rows: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    summaries: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        # scenes that every arm has finished
        scenes = sorted(
            set.intersection(
                *(
                    {p.parent.parent.name[: -len("_b")] for p in f.glob("s*_b/shape/output.json")}
                    for f in arms.values()
                )
            )
        )
        for arm, folder in arms.items():
            report = run_eval(arm, folder, args.split, scenes, tmp)
            summaries[arm] = report["summary"]
            rows[arm] = {}
            for scene in report["scenes"]:
                for o in scene["objects"]:
                    if not o["matched"]:
                        continue
                    run_dir = folder / f"{scene['scene_id']}_b"
                    med, near = cluster_to_mesh(run_dir, o["pred_id"])
                    registered = RegisteredObjects.model_validate_json(
                        (run_dir / "shape/output.json").read_text(encoding="utf-8")
                    )
                    source = next(r for r in registered.objects if r.object_id == o["pred_id"])
                    rows[arm][(scene["scene_id"], o["object_id"])] = {
                        "tier": scene["tier"],
                        "label": o["label"],
                        "size_error": o["size_error"],
                        "position_error_m": o["position_error_m"],
                        "orientation_error_deg": o["orientation_error_deg"],
                        "cluster_to_mesh_median_m": med,
                        "cluster_within_5mm": near,
                        "proxy": source.geometry_source.startswith("proxy"),
                        "fit_score": source.fit_score,
                    }

    common = sorted(set.intersection(*(set(r) for r in rows.values())))

    def med(arm: str, key: str, keys: list[tuple[str, str]]) -> float | None:
        v = [rows[arm][k][key] for k in keys if rows[arm][k][key] is not None]
        return statistics.median(v) if v else None

    tiers = sorted({rows[next(iter(arms))][k]["tier"] for k in common})
    result: dict[str, Any] = {"scenes": scenes, "n_common_objects": len(common), "arms": {}}
    for arm in arms:
        entry: dict[str, Any] = {"harness_summary": summaries[arm], "paired": {}}
        for name, keys in [("all", common)] + [
            (t, [k for k in common if rows[arm][k]["tier"] == t]) for t in tiers
        ]:
            entry["paired"][name] = {
                "n": len(keys),
                "size_error_median": med(arm, "size_error", keys),
                "position_error_median_m": med(arm, "position_error_m", keys),
                "orientation_error_median_deg": med(arm, "orientation_error_deg", keys),
                "cluster_to_mesh_median_m": med(arm, "cluster_to_mesh_median_m", keys),
                "cluster_within_5mm_median": med(arm, "cluster_within_5mm", keys),
                "size_error_over_15pct": sum(rows[arm][k]["size_error"] > 0.15 for k in keys),
                "proxies": sum(rows[arm][k]["proxy"] for k in keys),
            }
        result["arms"][arm] = entry
    names = list(arms)
    wins = {n: 0 for n in names}
    for k in common:
        wins[min(names, key=lambda n: rows[n][k]["size_error"])] += 1
    result["lowest_size_error_count"] = wins
    result["objects"] = [
        {"scene": k[0], "object": k[1], "label": rows[names[0]][k]["label"],
         **{n: rows[n][k] for n in names}}
        for k in common
    ]  # fmt: skip
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")

    header = f"{'':8}" + "".join(f"{n:>14}" for n in names)
    print(f"{len(common)} objects matched by every arm, scenes {', '.join(scenes)}")
    print(header)
    for key, fmt in [("size_error_median", "{:.1%}"), ("position_error_median_m", "{:.4f}"),
                     ("orientation_error_median_deg", "{:.2f}"),
                     ("cluster_to_mesh_median_m", "{:.4f}"), ("size_error_over_15pct", "{}"),
                     ("proxies", "{}")]:  # fmt: skip
        cells = []
        for n in names:
            v = result["arms"][n]["paired"]["all"][key]
            cells.append("-" if v is None else fmt.format(v))
        print(f"{key[:30]:32}" + "".join(f"{c:>14}" for c in cells))
    print("lowest size error:", wins)


if __name__ == "__main__":
    sys.exit(main())
