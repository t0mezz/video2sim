"""Fit fill fractions on the weighed dev objects and report mass error before and after (P4.6).

    uv run python benchmark/fit_fill_fractions.py --mass benchmark/results/p4_5_mass.json \
        --out benchmark/results/p4_6_fill_fit.json

Reads the per-object rows of ``benchmark/run_mass.py``; no mesh is touched. An object seen in
several scenes counts once in a fit, as the geometric mean of its runs, so a repeated object does
not outvote the others. A class and solid-or-hollow answer gets its own value from
``MIN_OBJECTS`` distinct objects, otherwise the global value of that answer. ``f`` stays in
(0, 1]: a need above 1 says the density is wrong, not the fill.

Scores are leave-one-object-out: every run of an object is predicted from a fit without that
object, because an object that is held out only per scene would still be in its own fit.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
MIN_OBJECTS = 3  # at 3 a median has a middle; below it one object would set the value
MIN_FILL = 0.01  # floor of a fitted share, so that a range stays positive

Row = dict[str, Any]
Fill = dict[str, float]


def answer(row: Row) -> str:
    return "hollow" if row["model_hollow"] else "solid"


def corrected(row: Row) -> bool:
    """An open vessel has already lost its cavity and keeps ``f = 1``."""
    return not (row["model_hollow"] and row["is_open"])


def fill_of(needs: list[float]) -> Fill:
    low, nominal, high = np.clip(np.percentile(needs, (10, 50, 90)), MIN_FILL, 1.0)
    return {"low": float(low), "nominal": float(nominal), "high": float(high)}


def per_object(rows: list[Row]) -> list[float]:
    by_object: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_object[r["object"]].append(float(r["fill_needed"]))
    return [float(np.exp(np.mean(np.log(v)))) for v in by_object.values()]


def fit(rows: list[Row], by_class: bool) -> dict[str, Any]:
    rows = [r for r in rows if corrected(r)]
    out: dict[str, Any] = {"global": {}, "classes": {}}
    for a in ("hollow", "solid"):
        subset = [r for r in rows if answer(r) == a]
        out["global"][a] = fill_of(per_object(subset))
        if not by_class:
            continue
        for name in sorted({r["material_class"] for r in subset}):
            needs = per_object([r for r in subset if r["material_class"] == name])
            if len(needs) >= MIN_OBJECTS:
                out["classes"].setdefault(name, {})[a] = fill_of(needs)
    return out


def fill_for(row: Row, model: dict[str, Any]) -> float:
    if not corrected(row):
        return 1.0
    entry = model["classes"].get(row["material_class"], {})
    return float(entry.get(answer(row), model["global"][answer(row)])["nominal"])


def ratios_held_out(rows: list[Row], by_class: bool) -> list[float]:
    out = []
    for r in rows:
        model = fit([x for x in rows if x["object"] != r["object"]], by_class)
        out.append(r["mass_uncorrected_kg"] * fill_for(r, model) / r["mass_true_kg"])
    return out


def score(ratios: list[float], tolerance: float) -> dict[str, Any]:
    a = np.asarray(ratios)
    return {
        "within_tolerance": int((np.abs(a - 1) <= tolerance).sum()),
        "share": float((np.abs(a - 1) <= tolerance).mean()),
        "median_ratio": float(np.median(a)),
        "median_abs_log2_ratio": float(np.median(np.abs(np.log2(a)))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mass", type=Path, default=ROOT / "benchmark/results/p4_5_mass.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows: list[Row] = json.loads(args.mass.read_text())["objects"]
    targets = yaml.safe_load((ROOT / "configs/targets.yaml").read_text(encoding="utf-8"))
    tolerance = float(targets["targets"]["mass_tolerance"])

    final = fit(rows, by_class=True)
    uncorrected = [r["mass_uncorrected_kg"] / r["mass_true_kg"] for r in rows]
    scores = {
        "uncorrected": score(uncorrected, tolerance),
        "global_held_out": score(ratios_held_out(rows, by_class=False), tolerance),
        "per_class_held_out": score(ratios_held_out(rows, by_class=True), tolerance),
        "per_class_in_sample": score(
            [r["mass_uncorrected_kg"] * fill_for(r, final) / r["mass_true_kg"] for r in rows],
            tolerance,
        ),
    }
    payload = {
        "objects": len(rows),
        "distinct_objects": len({r["object"] for r in rows}),
        "tolerance": tolerance,
        "min_objects": MIN_OBJECTS,
        "fit": final,
        "scores": scores,
    }
    args.out.write_text(json.dumps(payload, indent=1) + "\n")
    print(json.dumps(payload, indent=1))


if __name__ == "__main__":
    main()
