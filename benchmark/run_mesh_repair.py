"""Run the P4.1 mesh repair on the raw SAM 3D meshes of finished shape runs.

    uv run python benchmark/run_mesh_repair.py --out benchmark/results/p4_1_mesh_repair.json

Reads every ``runs/*/shape.work/sam3d_cache/*.glb`` (raw samples in the object frame, before
registration), repairs each with ``stages/mesh_repair.py`` and reports the checks, the triangle
counts and the volume change from decimation. No hosted call and no GPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import trimesh
import yaml

from video2sim.stages.mesh_repair import RepairConfig, repair_mesh, validate_mesh

ROOT = Path(__file__).resolve().parent.parent


def load_mesh(path: Path) -> trimesh.Trimesh:
    scene = trimesh.load(path, process=False)
    parts = [
        g
        for g in (scene.geometry.values() if hasattr(scene, "geometry") else [scene])
        if isinstance(g, trimesh.Trimesh)
    ]
    if not parts:
        raise ValueError(f"{path} holds no triangle mesh")
    return trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, default=ROOT / "runs")
    ap.add_argument("--out", type=Path, default=ROOT / "benchmark/results/p4_1_mesh_repair.json")
    args = ap.parse_args()
    cfg = RepairConfig.from_config(
        yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    )
    rows: list[dict[str, Any]] = []
    for path in sorted(args.runs.glob("*/shape.work/sam3d_cache/*.glb")):
        mesh = load_mesh(path)
        res = repair_mesh(mesh, cfg)
        raw_vol = float(mesh.volume) if mesh.is_watertight else None
        rows.append(
            {
                "mesh": str(path.relative_to(args.runs)),
                "faces_in": len(mesh.faces),
                "faces_out": len(res.mesh.faces),
                "checks_before": validate_mesh(mesh, cfg),
                "checks_after": res.checks,
                "steps": res.steps,
                "ok": res.ok,
                "volume_change_frac": (
                    (float(res.mesh.volume) - raw_vol) / raw_vol
                    if raw_vol and res.mesh.is_watertight
                    else None
                ),
            }
        )
    summary = {
        "meshes": len(rows),
        "pass_before": sum(all(r["checks_before"].values()) for r in rows),
        "pass_after": sum(r["ok"] for r in rows),
        "config": cfg.__dict__,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "meshes": rows}, indent=2), "utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
