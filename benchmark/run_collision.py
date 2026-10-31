"""Run the P4.2 collision geometry on the raw SAM 3D meshes of finished shape runs.

    uv run python benchmark/run_collision.py --out benchmark/results/p4_2_collision.json

Reads every ``runs/*/shape.work/sam3d_cache/*.glb``, repairs it (P4.1), then builds its collision
geometry (``stages/collision.py``) and reports the kind, part count, checks, surface fit
(coverage and tightness) and seconds per mesh. Needs ``coacd``. No hosted call and no GPU.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml
from run_mesh_repair import load_mesh

from video2sim.stages.collision import CollisionConfig, collision_geometry, surface_fit
from video2sim.stages.mesh_repair import RepairConfig, repair_mesh

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, default=ROOT / "runs")
    ap.add_argument("--out", type=Path, default=ROOT / "benchmark/results/p4_2_collision.json")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    repair_cfg = RepairConfig.from_config(cfg)
    coll_cfg = CollisionConfig.from_config(cfg)
    rows: list[dict[str, Any]] = []
    for path in sorted(args.runs.glob("*/shape.work/sam3d_cache/*.glb")):
        mesh = repair_mesh(load_mesh(path), repair_cfg).mesh
        t0 = time.time()
        res = collision_geometry(mesh, coll_cfg)
        coverage, tightness = surface_fit(res.parts, mesh, coll_cfg)
        rows.append(
            {
                "mesh": str(path.relative_to(args.runs)),
                "faces": len(mesh.faces),
                "kind": res.kind,
                "parts": len(res.parts),
                "primitive_score": res.primitive_score,
                "checks": res.checks,
                "steps": res.steps,
                "coverage": coverage,
                "tightness": tightness,
                "seconds": round(time.time() - t0, 2),
            }
        )
    summary = {
        "meshes": len(rows),
        "pass": sum(all(r["checks"].values()) for r in rows),
        "by_kind": {k: sum(r["kind"] == k for r in rows) for k in {r["kind"] for r in rows}},
        "config": asdict(coll_cfg),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "meshes": rows}, indent=2), "utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
