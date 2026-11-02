"""Run the P4.1 mesh repair on the raw SAM 3D meshes of finished shape runs.

    uv run python benchmark/run_mesh_repair.py --out benchmark/results/p4_1_mesh_repair.json

Reads every ``runs/*/shape.work/sam3d_cache/*.glb`` (raw samples in the object frame, before
registration), repairs each with ``stages/mesh_repair.py`` and reports the checks, the triangle
counts, the volume change from repair, and the visual copy: its triangle count, the budget it
reached and its surface error (D-033). No hosted call and no GPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import trimesh
import yaml

from video2sim.stages.mesh_repair import (
    RepairConfig,
    VisualConfig,
    repair_mesh,
    validate_mesh,
    visual_mesh,
)

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
    raw = yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    cfg, vcfg = RepairConfig.from_config(raw), VisualConfig.from_config(raw)
    rows: list[dict[str, Any]] = []
    for path in sorted(args.runs.glob("*/shape.work/sam3d_cache/*.glb")):
        mesh = load_mesh(path)
        res = repair_mesh(mesh, cfg)
        vis = visual_mesh(res.mesh, vcfg)
        raw_vol = float(mesh.volume) if mesh.is_watertight else None
        rows.append(
            {
                "mesh": str(path.relative_to(args.runs)),
                "faces_in": len(mesh.faces),
                "faces_out": len(res.mesh.faces),
                "checks_before": validate_mesh(mesh, cfg),
                "checks_after": res.checks,
                "steps": res.steps,
                "visual_faces": len(vis.mesh.faces),
                "visual_checks": vis.checks,
                "visual_advisory": vis.advisory,
                "visual_surface_error": vis.surface_error,
                "visual_steps": vis.steps,
                "ok": res.ok and all(vis.checks.values()),
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
        "within_budget": sum(r["visual_advisory"]["within_triangle_budget"] for r in rows),
        "config": {"repair": cfg.__dict__, "visual": vcfg.__dict__},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "meshes": rows}, indent=2), "utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
