"""Settle exported scenes under gravity in Isaac Sim / PhysX (D-027, Phase 5 item 8).

    $ISAACLAB/.venv/bin/python benchmark/run_isaac_settling.py RUN_DIR ... \
        --out benchmark/results/isaac_settling.json

Runs the five cases of ``run_settling.py`` (nominal, low and high friction and mass) on
``export/scene.usd`` as the export stage wrote it, with the gate of ``targets.yaml``: drift of each
object over ``stability_window_s`` below ``stability_max_drift_m``. The case and drift logic is in
``video2sim/isaac_verify.py``; this file only drives Kit. Needs Isaac Sim's interpreter (the one in
the IsaacLab ``.venv``) and ``OMNI_KIT_ACCEPT_EULA=YES``. A GPU is used by PhysX; no hosted call.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from video2sim.isaac_verify import (
    CASES,
    apply_case,
    ignored_names,
    judge,
    object_names,
    positions,
    scene_stable,
)

ROOT = Path(__file__).resolve().parent.parent
LOG = logging.getLogger("isaac_settling")
SAMPLE_EVERY_STEPS = 20  # drift_sample_s over timestep_s in default.yaml


def run_case(
    app: Any, usd: Path, case: str, window_s: float, dt: float, max_drift_m: float
) -> dict[str, Any]:
    import omni.timeline
    import omni.usd
    from pxr import PhysxSchema

    omni.usd.get_context().open_stage(str(usd))
    for _ in range(5):
        app.update()
    stage = omni.usd.get_context().get_stage()
    names = object_names(stage)
    apply_case(stage, case)
    rate = round(1 / dt)
    PhysxSchema.PhysxSceneAPI.Apply(
        stage.GetPrimAtPath("/World/PhysicsScene")
    ).CreateTimeStepsPerSecondAttr(rate)
    stage.SetTimeCodesPerSecond(rate)
    start = positions(stage, names)

    timeline = omni.timeline.get_timeline_interface()
    timeline.set_target_framerate(rate)
    timeline.play()
    steps = round(window_s / dt)
    samples = []
    for step in range(1, steps + 1):  # one app update is one physics step at this rate
        app.update()
        if step % SAMPLE_EVERY_STEPS == 0:
            samples.append(positions(stage, names))
    final = positions(stage, names)
    timeline.stop()
    bodies = judge(names, start, samples, final, max_drift_m, ignored_names(stage))
    return {
        "case": case,
        "stable": scene_stable(bodies),
        "bodies": [asdict(b) for b in bodies],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="run folders with an export/scene.usd")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--window-s", type=float, help="shorter than the gate, for a smoke test")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    targets = yaml.safe_load((ROOT / "configs/targets.yaml").read_text(encoding="utf-8"))["targets"]
    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    window = args.window_s or float(targets["stability_window_s"])
    max_drift = float(targets["stability_max_drift_m"])
    dt = float(cfg["physics"]["settling"]["timestep_s"])

    from isaacsim import SimulationApp  # starts Kit; must come before any omni import

    app = SimulationApp({"headless": True})
    rows = []
    for run in args.runs:
        usd = run.resolve() / "export" / "scene.usd"
        results = [run_case(app, usd, c, window, dt, max_drift) for c in CASES]
        rows.append(
            {
                "run": run.name,
                "stable_every_case": all(r["stable"] for r in results),
                "cases": results,
            }
        )
        LOG.info("%s stable_every_case=%s", run.name, rows[-1]["stable_every_case"])
    summary = {
        "simulator": "isaac sim / physx (D-027)",
        "window_s": window,
        "max_drift_m": max_drift,
        "runs": len(rows),
        "stable_every_case": [r["run"] for r in rows if r["stable_every_case"]],
    }
    args.out.write_text(json.dumps({"summary": summary, "runs": rows}, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    for row in rows:
        for r in row["cases"]:
            worst = max(b["max_drift_m"] for b in r["bodies"])
            print(row["run"], r["case"], "stable" if r["stable"] else "UNSTABLE", f"{worst:.4f} m")
    app.close()


if __name__ == "__main__":
    main()
