"""The section 2 metrics for one scene and over many scenes (P0.6).

Definitions (agreed in the P0.6 design review):

* A ground-truth object is *evaluable* when it has a size and a position. Others are
  counted as excluded and reported, not scored. Scenes with a deformable object are left
  out before this point (D-014), see ``GtScene.has_deformable``.
* Matching: Hungarian assignment on box-center distance inside a per-object gate
  (``matching.gate_m``). Unmatched ground truth is a miss, unmatched predictions are false
  positives that do not enter recall.
* Size error: largest relative error over length, width and height.
* Position error: 3D distance between the predicted and the ground-truth box centers.
* Orientation error: yaw difference modulo 180 degrees, skipped when the ground-truth
  footprint is symmetric (``yaw_is_defined``).
* Mass: relative error of the nominal mass. A matched object without a predicted mass counts
  as outside the tolerance.
* Error medians cover matched objects only; misses show up in recall.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from video2sim.evaluation.geometry import resting_box
from video2sim.evaluation.ground_truth import yaw_is_defined
from video2sim.evaluation.matching import gate_m, match
from video2sim.evaluation.scenes import GtScene, Prediction

NOT_MEASURED = {
    "stability_pass_fraction": "needs the simulator (Phase 5)",
    "asset_validity_fraction": "needs the export validators (Phase 5)",
    "manual_steps": "needs run logs of the full pipeline (Phase 5)",
}


@dataclass(frozen=True)
class ObjectResult:
    scene_id: str
    tier: str
    object_id: str
    label: str
    matched: bool
    pred_id: str | None = None
    size_error: float | None = None
    position_error_m: float | None = None
    orientation_error_deg: float | None = None
    mass_error: float | None = None  # relative; None when the mass is unknown or not predicted
    mass_within_tolerance: bool | None = None


@dataclass(frozen=True)
class SceneResult:
    scene_id: str
    tier: str
    split: str
    n_evaluable: int
    n_excluded: int
    n_matched: int
    n_false_positive: int
    runtime_s: float | None
    objects: tuple[ObjectResult, ...]


def yaw_difference_deg(a: float, b: float) -> float:
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def evaluate_scene(
    gt: GtScene, prediction: Prediction, mass_tolerance: float = 0.30
) -> SceneResult:
    evaluable = [o for o in gt.objects if o.size_m is not None and o.position_m is not None]
    boxes = [resting_box(p.vertices) for p in prediction.objects]

    gt_centers = np.array([o.position_m for o in evaluable], dtype=float).reshape(-1, 3)
    gates = np.array([gate_m(o.size_m) for o in evaluable if o.size_m], dtype=float)
    pred_centers = np.array([b.center_m for b in boxes], dtype=float).reshape(-1, 3)
    pairs = dict(match(gt_centers, gates, pred_centers))

    results: list[ObjectResult] = []
    for gi, obj in enumerate(evaluable):
        pi = pairs.get(gi)
        if pi is None:
            results.append(ObjectResult(gt.scene_id, gt.tier, obj.object_id, obj.label, False))
            continue
        box, pred = boxes[pi], prediction.objects[pi]
        assert obj.size_m is not None and obj.position_m is not None
        size_error = max(abs(p - g) / g for p, g in zip(box.size_m, obj.size_m, strict=True))
        position_error = float(np.linalg.norm(np.subtract(box.center_m, obj.position_m)))
        orientation_error = None
        if obj.yaw_deg is not None and yaw_is_defined(obj.size_m):
            orientation_error = yaw_difference_deg(box.yaw_deg, obj.yaw_deg)
        mass_error: float | None = None
        within: bool | None = None
        if obj.mass_kg is not None:
            if pred.mass_kg is None:
                within = False
            else:
                mass_error = abs(pred.mass_kg - obj.mass_kg) / obj.mass_kg
                within = mass_error <= mass_tolerance
        results.append(
            ObjectResult(
                gt.scene_id,
                gt.tier,
                obj.object_id,
                obj.label,
                True,
                pred.object_id,
                size_error,
                position_error,
                orientation_error,
                mass_error,
                within,
            )
        )
    return SceneResult(
        scene_id=gt.scene_id,
        tier=gt.tier,
        split=gt.split,
        n_evaluable=len(evaluable),
        n_excluded=len(gt.objects) - len(evaluable),
        n_matched=len(pairs),
        n_false_positive=len(prediction.objects) - len(pairs),
        runtime_s=prediction.runtime_s,
        objects=tuple(results),
    )


def _metric(
    name: str, value: float | None, n: int, target: float | None, direction: str | None
) -> dict[str, Any]:
    """One metric row. ``direction`` is "min" (value must reach target) or "max"."""
    entry: dict[str, Any] = {
        "name": name,
        "status": "measured" if value is not None else "no_data",
        "value": value,
        "n": n,
        "target": target,
        "passed": None,
    }
    if value is not None and target is not None and direction is not None:
        entry["passed"] = value >= target if direction == "min" else value <= target
    return entry


def _median(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


def summarize(scenes: Iterable[SceneResult], targets: Mapping[str, float]) -> list[dict[str, Any]]:
    """Metric rows over the given scenes, each with its target and pass/fail."""
    scene_list = list(scenes)
    objects = [o for s in scene_list for o in s.objects]
    matched = [o for o in objects if o.matched]

    def present(values: Iterable[float | None]) -> list[float]:
        return [v for v in values if v is not None]

    n_eval = sum(s.n_evaluable for s in scene_list)
    recall = (len(matched) / n_eval) if n_eval else None
    sizes = present(o.size_error for o in matched)
    positions = present(o.position_error_m for o in matched)
    orientations = present(o.orientation_error_deg for o in matched)
    mass_flags = [o.mass_within_tolerance for o in matched if o.mass_within_tolerance is not None]
    mass_fraction = (sum(mass_flags) / len(mass_flags)) if mass_flags else None
    runtimes = present(s.runtime_s for s in scene_list)

    t = targets
    rows = [
        _metric("object_recall", recall, n_eval, t.get("object_recall_min"), "min"),
        _metric(
            "size_error_median", _median(sizes), len(sizes), t.get("size_error_median_max"), "max"
        ),
        _metric(
            "position_error_median_m",
            _median(positions),
            len(positions),
            t.get("position_error_median_max_m"),
            "max",
        ),
        _metric(
            "orientation_error_median_deg",
            _median(orientations),
            len(orientations),
            t.get("orientation_error_median_max_deg"),
            "max",
        ),
        _metric(
            "mass_within_tolerance_fraction",
            mass_fraction,
            len(mass_flags),
            t.get("mass_within_30pct_fraction_min"),
            "min",
        ),
        _metric(
            "runtime_max_minutes",
            (max(runtimes) / 60.0) if runtimes else None,
            len(runtimes),
            t.get("runtime_max_minutes"),
            "max",
        ),
    ]
    rows += [
        {
            "name": name,
            "status": "not_measured",
            "value": None,
            "n": 0,
            "target": None,
            "passed": None,
            "reason": reason,
        }
        for name, reason in NOT_MEASURED.items()
    ]
    return rows


def aggregate(scenes: Sequence[SceneResult], targets: Mapping[str, float]) -> dict[str, Any]:
    """Pooled metrics, plus the same metrics per tier (values only, no pass/fail)."""
    per_tier = {}
    for tier in dict.fromkeys(s.tier for s in scenes):
        rows = summarize([s for s in scenes if s.tier == tier], targets)
        per_tier[tier] = [
            {**r, "passed": None, "target": None} for r in rows if r["status"] != "not_measured"
        ]
    return {
        "pooled": summarize(scenes, targets),
        "per_tier": per_tier,
        "n_scenes": len(scenes),
        "n_evaluable_objects": sum(s.n_evaluable for s in scenes),
        "n_excluded_objects": sum(s.n_excluded for s in scenes),
        "n_false_positives": sum(s.n_false_positive for s in scenes),
    }


def scene_to_dict(result: SceneResult) -> dict[str, Any]:
    data = asdict(result)
    # NaN is not valid JSON; none of the fields should hold one, so fail loudly if one does.
    for obj in data["objects"]:
        for key, value in obj.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{result.scene_id}/{obj['object_id']}: {key} is {value}")
    return data
