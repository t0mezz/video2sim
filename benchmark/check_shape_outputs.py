"""Check the shape stage's outputs against two Phase 3 exit criteria.

    uv run python benchmark/check_shape_outputs.py s01_b s02_b --runs runs_instances \
        --out benchmark/results/p3_shape_checks_dev.json

1. Every object has a registered mesh or an explicit low-confidence fallback: each instance from
   segmentation has a ``RegisteredObject`` whose mesh file exists, and a proxy hull carries
   ``PROXY_GEOMETRY`` and ``LOW_FIT``.
2. Registered meshes resting on the table show no visible gap or penetration: for every object
   the stage held on the table (``support_z`` in ``shape/report.json``), the lowest world vertex
   against that height. The tolerance is ``--tolerance-mm`` (2 mm by default, a choice made here,
   not in ``targets.yaml``). Proxy hulls are listed apart, since they come from the cluster.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from video2sim.schemas import Flag, Instances, RegisteredObjects

ROOT = Path(__file__).resolve().parent.parent


def check_run(run_dir: Path, tolerance_m: float) -> dict[str, Any]:
    instances = Instances.model_validate_json(
        (run_dir / "segmentation" / "output.json").read_text(encoding="utf-8")
    )
    registered = RegisteredObjects.model_validate_json(
        (run_dir / "shape" / "output.json").read_text(encoding="utf-8")
    )
    report = json.loads((run_dir / "shape" / "report.json").read_text(encoding="utf-8"))
    by_id = {o.object_id: o for o in registered.objects}
    missing = [o.object_id for o in instances.objects if o.object_id not in by_id]
    bad_fallback, no_file, contact = [], [], []
    for obj in registered.objects:
        mesh_file = run_dir / obj.mesh_path
        if not mesh_file.is_file():
            no_file.append(obj.object_id)
            continue
        proxy = obj.geometry_source.startswith("proxy")
        if proxy and not {Flag.PROXY_GEOMETRY, Flag.LOW_FIT} <= set(obj.flags):
            bad_fallback.append(obj.object_id)
        support_z = report.get(obj.object_id, {}).get("support_z")
        if support_z is None:
            continue
        mesh = trimesh.load(mesh_file, process=False)
        T = np.asarray(obj.T_world_from_object.matrix, dtype=float)
        z = (np.asarray(mesh.vertices) @ T[:3, :3].T + T[:3, 3])[:, 2]
        contact.append(
            {"object": obj.object_id, "proxy": proxy, "gap_m": float(z.min() - support_z)}
        )
    return {
        "instances": len(instances.objects),
        "registered": len(registered.objects),
        "without_registered_object": missing,
        "mesh_file_missing": no_file,
        "proxy_without_flags": bad_fallback,
        "proxies": sum(o.geometry_source.startswith("proxy") for o in registered.objects),
        "contact": contact,
        "contact_over_tolerance": [c["object"] for c in contact if abs(c["gap_m"]) > tolerance_m],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_ids", nargs="+")
    parser.add_argument("--runs", type=Path, default=ROOT / "runs_instances")
    parser.add_argument("--tolerance-mm", type=float, default=2.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    tol = args.tolerance_mm / 1000.0
    runs = {r: check_run(args.runs / r, tol) for r in args.run_ids}
    gaps = [c["gap_m"] for r in runs.values() for c in r["contact"] if not c["proxy"]]
    proxy_gaps = [c["gap_m"] for r in runs.values() for c in r["contact"] if c["proxy"]]
    summary = {
        "tolerance_mm": args.tolerance_mm,
        "instances": sum(r["instances"] for r in runs.values()),
        "registered": sum(r["registered"] for r in runs.values()),
        "proxies": sum(r["proxies"] for r in runs.values()),
        "objects_without_registered_object": sum(
            len(r["without_registered_object"]) for r in runs.values()
        ),
        "mesh_files_missing": sum(len(r["mesh_file_missing"]) for r in runs.values()),
        "proxies_without_flags": sum(len(r["proxy_without_flags"]) for r in runs.values()),
        "resting_meshes": len(gaps),
        "resting_max_abs_gap_mm": max((abs(g) for g in gaps), default=0.0) * 1000,
        "resting_over_tolerance": sum(abs(g) > tol for g in gaps),
        "resting_proxies": len(proxy_gaps),
        "resting_proxy_max_abs_gap_mm": max((abs(g) for g in proxy_gaps), default=0.0) * 1000,
        "resting_proxy_over_tolerance": sum(abs(g) > tol for g in proxy_gaps),
    }  # fmt: skip
    args.out.write_text(json.dumps({"summary": summary, "runs": runs}, indent=1) + "\n")
    for k, v in summary.items():
        print(f"{k:38} {v if not isinstance(v, float) else round(v, 3)}")


if __name__ == "__main__":
    main()
