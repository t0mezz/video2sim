"""The QA report of a scene (P5.3): ``qa_report.json`` for tools, ``qa_report.html`` for people.

It shows every object, exported or not, with what could be wrong with it. The mass source and
range are always shown, since masses may be off by more than 30% on about half of the objects
(D-026). Each row also carries the visual mesh polygon count, which tells a light proxy hull
from a dense generative mesh at a glance. Stability comes from the physics stage and the URDF
check in MuJoCo only: Isaac Sim is not run (D-027), and the report says so.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, cast

from video2sim.schemas import (
    Instances,
    RegisteredObjects,
    SimAssets,
    SkippedAsset,
    WorldModel,
)

VALIDATED_IN = "MuJoCo (D-027); Isaac Sim and PhysX not verified"


def _mesh_counts(run_dir: Path, relpath: str) -> tuple[int | None, int | None]:
    """(vertices, faces) of the visual mesh, or Nones when it cannot be read.

    Export reads every visual mesh before the report is built, so a failure here means
    the file changed under us; the row shows a dash instead of failing the stage.
    """
    import trimesh  # heavy import, kept out of module import time

    try:
        mesh = cast("trimesh.Trimesh", trimesh.load(run_dir / relpath, force="mesh", process=False))
        return len(mesh.vertices), len(mesh.faces)
    except Exception:
        return None, None


def _object_row(
    asset: Any,
    instance: Any,
    registered: Any,
    physics: dict[str, Any],
    settling: dict[str, dict[str, Any]],
    skipped: dict[str, list[str]],
    mesh_counts: dict[str, tuple[int | None, int | None]],
) -> dict[str, Any]:
    name = asset.object_id
    p = asset.properties
    vertices, faces = mesh_counts.get(name, (None, None))
    drifts: dict[str, float] = {}
    stable = True
    for case, result in settling.items():
        body = next((b for b in result.get("bodies", []) if b["object_id"] == name), None)
        if body is None:
            stable = False
            continue
        drifts[case] = body["max_drift_m"]
        stable = stable and (body["stable"] or body["ignored"])
    return {
        "object_id": name,
        "label": instance.category or instance.label,
        "status": "skipped" if name in skipped else "exported",
        "skip_reasons": skipped.get(name, []),
        "geometry_source": registered.geometry_source,
        "fit_score": registered.fit_score,
        "vertices": vertices,
        "faces": faces,
        "collision_kind": physics["collision_kind"],
        "collision_parts": physics["collision_parts"],
        "material_class": p.material_class,
        "is_hollow": p.is_hollow,
        "confidence": p.confidence,
        "mass_source": physics["mass_source"],
        "mass_kg": p.mass_kg.model_dump(),
        "friction": p.friction_coefficient.model_dump(),
        "restitution": p.restitution.model_dump(),
        "flags": [f.value for f in asset.flags],
        "notes": [*instance.flag_reasons, *physics["reasons"]],
        "found_by": getattr(instance, "found_by", "generic"),
        "stable_every_case": stable,
        "largest_drift_m": max(drifts.values(), default=None),
    }


def build_report(
    run_id: str,
    run_dir: Path,
    world: WorldModel,
    instances: Instances,
    registered: RegisteredObjects,
    assets: SimAssets,
    physics: dict[str, dict[str, Any]],
    settling: dict[str, dict[str, Any]],
    skipped: list[SkippedAsset],
    urdf_check: dict[str, Any],
    views: list[str],
) -> dict[str, Any]:
    by_instance = {o.object_id: o for o in instances.objects}
    by_shape = {o.object_id: o for o in registered.objects}
    reasons = {s.object_id: s.reasons for s in skipped}
    mesh_counts = {a.object_id: _mesh_counts(run_dir, a.visual_mesh_path) for a in assets.assets}
    rows = [
        _object_row(
            a,
            by_instance[a.object_id],
            by_shape[a.object_id],
            physics[a.object_id],
            settling,
            reasons,
            mesh_counts,
        )
        for a in assets.assets
    ]
    missing = sorted(set(by_instance) - {r["object_id"] for r in rows})
    review = [
        r["object_id"]
        for r in rows
        if r["status"] == "skipped" or r["flags"] or not r["stable_every_case"]
    ]
    scene_stable = all(v.get("stable", False) for v in settling.values())
    good = not skipped and not missing and scene_stable and not urdf_check["problems"]
    discovery = instances.discovery.model_dump() if instances.discovery is not None else None
    return {
        "run_id": run_id,
        "status": "ok" if good and not review else "review",
        "validated_in": VALIDATED_IN,
        "discovery": discovery,
        "objects_found": len(by_instance),
        "objects_exported": len(rows) - len(skipped),
        "objects_skipped": len(skipped),
        "objects_without_asset": missing,
        "objects_to_review": review,
        "scene_flags": [*world.flags, *instances.flags],
        "stable_every_case": scene_stable,
        "stability_cases": {k: bool(v.get("stable", False)) for k, v in settling.items()},
        "urdf_check": urdf_check,
        "views": views,
        "objects": rows,
    }


def _triple(value: dict[str, float], digits: str = ".3g") -> str:
    return " / ".join(format(value[k], digits) for k in ("low", "nominal", "high"))


def _cell(value: object) -> str:
    return f"<td>{html.escape(str(value))}</td>"


def _html(report: dict[str, Any]) -> str:
    columns = (
        "object", "label", "status", "geometry", "fit", "polygons", "collision", "material",
        "confidence", "mass source", "mass kg (low / nominal / high)", "friction", "restitution",
        "drift cm", "stable", "flags and notes",
    )  # fmt: skip
    rows = []
    for o in report["objects"]:
        drift = o["largest_drift_m"]
        notes = [*o["flags"], *o["skip_reasons"], *o["notes"]]
        if o.get("found_by", "generic") != "generic":
            notes.append(f"found by gap fill: {o['found_by']}")
        cells = [
            o["object_id"],
            o["label"],
            o["status"],
            o["geometry_source"],
            "-" if o["fit_score"] is None else f"{o['fit_score']:.2f}",
            "-" if o["faces"] is None else str(o["faces"]),
            f"{o['collision_kind']} ({o['collision_parts']})",
            o["material_class"] + (" hollow" if o["is_hollow"] else ""),
            f"{o['confidence']:.2f}",
            o["mass_source"],
            _triple(o["mass_kg"]),
            _triple(o["friction"]),
            _triple(o["restitution"]),
            "-" if drift is None else f"{drift * 100:.2f}",
            "yes" if o["stable_every_case"] else "NO",
            "; ".join(notes),
        ]
        marked = o["object_id"] in report["objects_to_review"]
        rows.append(
            f"<tr{' class=review' if marked else ''}>{''.join(_cell(c) for c in cells)}</tr>"
        )
    facts = {
        "Status": report["status"],
        "Validated in": report["validated_in"],
        "Objects": (
            f"{report['objects_exported']} exported, {report['objects_skipped']} skipped, "
            f"{len(report['objects_without_asset'])} without an asset, of "
            f"{report['objects_found']} found"
        ),
        "Without an asset": ", ".join(report["objects_without_asset"]) or "none",
        "Settling": ", ".join(
            f"{k} {'ok' if v else 'FAILED'}" for k, v in report["stability_cases"].items()
        ),
        "URDF settling": "agrees" if report["urdf_check"]["agrees"] else "DISAGREES",
        "Scene flags": "; ".join(report["scene_flags"]) or "none",
    }
    discovery = report.get("discovery")
    if discovery is not None:
        warnings = []
        if discovery["unmatched"]:
            warnings.append(f"unmatched inventory: {', '.join(discovery['unmatched'])}")
        if discovery["unexplained_clusters"]:
            warnings.append(f"{discovery['unexplained_clusters']} unexplained point clusters")
        facts["Discovery"] = (
            f"{discovery['tracked']} tracked of {discovery['listed']} listed, "
            f"{discovery['recovered']} recovered, "
            f"coverage {discovery['view_coverage']:.2f} over frames "
            f"{', '.join(str(f) for f in discovery['view_frames'])}; "
            + ("WARNING: " + "; ".join(warnings) if warnings else "no gaps")
        )
    summary = "".join(f"<tr><th>{html.escape(k)}</th>{_cell(v)}</tr>" for k, v in facts.items())
    images = "".join(
        f'<img src="{html.escape(v)}" alt="real frame beside the simulated scene">'
        for v in report["views"]
    )
    header = "".join(f"<th>{html.escape(c)}</th>" for c in columns)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>QA report {html.escape(report["run_id"])}</title>
<style>
body {{ font: 14px system-ui, sans-serif; margin: 24px; background: #fff; color: #222; }}
table {{ border-collapse: collapse; margin-bottom: 24px; }}
th, td {{ border: 1px solid #ccc; padding: 4px 8px; text-align: left; vertical-align: top; }}
tr.review {{ background: #fff4d6; }}
img {{ display: block; max-width: 100%; margin-bottom: 12px; }}
</style></head><body>
<h1>QA report {html.escape(report["run_id"])}</h1>
<table>{summary}</table>
<h2>Objects</h2>
<table><tr>{header}</tr>{"".join(rows)}</table>
<h2>Real frame and simulated scene, after settling</h2>
{images}
</body></html>
"""


def write_report(report: dict[str, Any], folder: Path) -> Path:
    """Write both files and return the HTML path."""
    (folder / "qa_report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    path = folder / "qa_report.html"
    path.write_text(_html(report), encoding="utf-8")
    return path


__all__ = ["VALIDATED_IN", "build_report", "write_report"]
