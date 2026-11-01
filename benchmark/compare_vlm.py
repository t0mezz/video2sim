"""Score two vision-language models on the same dev inputs: router, material pass and mass.

    uv run python benchmark/compare_vlm.py --out benchmark/results/vlm_compare.json \
        --local-url http://localhost:9931/v1

``spark`` is the hosted Muse Spark Open (``BASETEN_API_KEY``), ``local`` is the OpenAI-compatible
server of ``vlm.openai_compatible`` (``--local-url`` overrides its ``base_url``). Both get the
pipeline's own prompts, parsers and images. Inputs are the finished runs in ``runs/`` (router: the
frames of each scene) and ``runs_instances/`` (material and mass: the crops of each dev object),
as in ``run_router.py`` and ``run_vlm_mass.py``. Scoring against ground truth:

- router: listed objects against the number of rigid ground-truth objects, and the label overlap;
- material: ``hollow`` against the ground truth, share of readable answers, and how often the
  two models name the same material class and finish;
- mass: the ``crops`` arm of D-024, within ``mass_tolerance`` of the weighed mass.

Answers are cached per model under ``<out dir>/vlm_compare_cache/<model>/``, so a rerun costs
nothing. Each request is timed.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml
from run_vlm_mass import rows_to_ask, score

from video2sim.evaluation.scenes import load_scene
from video2sim.models.hosted import ResponseCache, image_data_url, read_secret
from video2sim.models.vlm import BasetenMaterialClient, ChatVlmClient, VlmConfig, make_vlm_client
from video2sim.runner import OUTPUT_FILE
from video2sim.schemas import WorldModel
from video2sim.stages import material_pass, scene_router
from video2sim.stages.vlm_mass import PROMPT as MASS_PROMPT
from video2sim.stages.vlm_mass import parse_mass

ROOT = Path(__file__).resolve().parent.parent


class Timed:
    """A client that records the seconds of every real request."""

    def __init__(self, client: ChatVlmClient, cache_dir: Path) -> None:
        self.client, self.model, self.cache_dir = client, client.model, cache_dir
        self.seconds: list[float] = []

    def ask(self, prompt: str, images: Sequence[Path]) -> str:
        key = [prompt, self.model, *(image_data_url(p) for p in images)]

        def call() -> str:
            t0 = time.monotonic()
            text = self.client.ask(prompt, images)
            self.seconds.append(time.monotonic() - t0)
            return text

        return ResponseCache(self.cache_dir).get_or_call(key, call)


def timing(clients: Sequence[Timed]) -> dict[str, Any]:
    s = [x for c in clients for x in c.seconds]
    return {
        "requests": len(s),
        "mean_s": round(statistics.fmean(s), 2) if s else None,
        "max_s": round(max(s), 2) if s else None,
    }


def safe(call: Callable[[], str]) -> tuple[str | None, str | None]:
    try:
        return call(), None
    except Exception as exc:  # a failed request counts as a miss and is listed
        return None, f"{type(exc).__name__}: {exc}"


def run_router(client: Timed, cfg: dict[str, Any], runs: Path) -> list[dict[str, Any]]:
    rcfg = scene_router.RouterConfig.from_config(cfg)
    rows = []
    for run_dir in sorted(runs.glob("s*_*")):
        scene_id = run_dir.name.rsplit("_", 1)[0]
        gt_file = ROOT / "benchmark/scenes" / scene_id / "ground_truth.json"
        recon = run_dir / "reconstruction" / OUTPUT_FILE
        if not gt_file.exists() or not recon.exists():
            continue
        world = WorldModel.model_validate_json(recon.read_text(encoding="utf-8"))
        picks = [
            world.frames[i]
            for i in scene_router.sample_frame_indices(len(world.frames), rcfg.n_frames)
        ]
        images = [run_dir / f.image_path for f in picks]
        raw, error = safe(lambda imgs=images: client.ask(scene_router.build_prompt(rcfg), imgs))
        route = scene_router.decide_route(raw, rcfg, failure=error)
        gt = [o for o in load_scene(gt_file).objects if not o.deformable]
        rows.append(
            {
                "scene": run_dir.name,
                "fell_back": route.fell_back,
                "reason": route.reason,
                "surface": route.surface,
                "listed": len(route.objects),
                "objects": list(route.objects),
                "gt_rigid": len(gt),
                "gt_labels": [o.label for o in gt],
            }
        )
        print("router", run_dir.name, client.model, len(route.objects), "of", len(gt), flush=True)
    return rows


def run_material(
    client: Timed, cfg: dict[str, Any], rows: list[dict[str, Any]], workers: int
) -> list[dict[str, Any]]:
    mcfg = material_pass.MaterialConfig.from_config(cfg)
    prompt = material_pass.build_prompt(mcfg)
    gt = {}
    for scene in {r["scene"] for r in rows}:
        raw_gt = json.loads((ROOT / "benchmark/scenes" / scene / "ground_truth.json").read_text())
        gt.update({(scene, o["object_id"]): o for o in raw_gt["objects"]})

    def one(r: dict[str, Any]) -> dict[str, Any]:
        images = [Path(p) for p in r["crop_paths"]]
        raw, error = safe(lambda: client.ask(prompt, images))
        answer = material_pass.parse_answer(raw, mcfg) if raw else None
        truth = gt[(r["scene"], r["object"])]
        return {
            "scene": r["scene"],
            "object": r["object"],
            "label": r["label"],
            "gt_hollow": truth.get("hollow"),
            "gt_material": truth.get("material_class"),
            "error": error,
            "answer": asdict(answer) if answer else None,
        }

    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(one, rows))


def run_mass(client: Timed, rows: list[dict[str, Any]], workers: int) -> list[dict[str, Any]]:
    def one(r: dict[str, Any]) -> dict[str, Any]:
        prompt = MASS_PROMPT.format(category=r["category"], measured="")
        raw, error = safe(lambda: client.ask(prompt, [Path(p) for p in r["crop_paths"]]))
        mass = parse_mass(raw) if raw else None
        return {
            "scene": r["scene"],
            "object": r["object"],
            "label": r["label"],
            "mass_true_kg": r["mass_true_kg"],
            "nominal_kg": mass.nominal if mass else None,
            "low_kg": mass.low if mass else None,
            "high_kg": mass.high if mass else None,
            "error": error or (None if mass else f"unreadable: {(raw or '')[:120]}"),
        }

    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(one, rows))


def summarize(
    router: dict[str, list[dict[str, Any]]],
    material: dict[str, list[dict[str, Any]]],
    mass: dict[str, list[dict[str, Any]]],
    tolerance: float,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for tag, rows in router.items():
        ok = [r for r in rows if not r["fell_back"]]
        out.setdefault("router", {})[tag] = {
            "scenes": len(rows),
            "answered": len(ok),
            "count_exact": sum(r["listed"] == r["gt_rigid"] for r in ok),
            "count_within_1": sum(abs(r["listed"] - r["gt_rigid"]) <= 1 for r in ok),
            "mean_abs_count_error": round(
                statistics.fmean(abs(r["listed"] - r["gt_rigid"]) for r in ok), 3
            )
            if ok
            else None,
        }
    for tag, rows in material.items():
        read = [r for r in rows if r["answer"]]
        hollow = [
            r for r in read if r["answer"]["hollow"] != "unknown" and r["gt_hollow"] is not None
        ]
        out.setdefault("material", {})[tag] = {
            "objects": len(rows),
            "readable": len(read),
            "hollow_answered": len(hollow),
            "hollow_correct": sum(
                (r["answer"]["hollow"] == "hollow") == bool(r["gt_hollow"]) for r in hollow
            ),
            "classes": dict(Counter(r["answer"]["material_class"] for r in read)),
        }
    if len(material) == 2:
        a, b = (dict(((r["scene"], r["object"]), r) for r in rows) for rows in material.values())
        both = [k for k in a if a[k]["answer"] and b.get(k, {}).get("answer")]
        out["material_agreement"] = {
            "both_readable": len(both),
            "same_class": sum(
                a[k]["answer"]["material_class"] == b[k]["answer"]["material_class"] for k in both
            ),
            "same_finish": sum(a[k]["answer"]["finish"] == b[k]["answer"]["finish"] for k in both),
        }
    for tag, rows in mass.items():
        s = score([(r["nominal_kg"], r["mass_true_kg"]) for r in rows], tolerance)
        s["truth_inside_low_high"] = sum(
            r["low_kg"] <= r["mass_true_kg"] <= r["high_kg"] for r in rows if r["nominal_kg"]
        )
        out.setdefault("mass", {})[tag] = s
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runs", type=Path, default=ROOT / "runs")
    parser.add_argument("--local-url", help="overrides vlm.openai_compatible.base_url")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--tasks", default="router,material,mass")
    parser.add_argument("--limit", type=int, help="only the first N dev objects, for a trial")
    args = parser.parse_args()
    cfg: dict[str, Any] = yaml.safe_load(
        (ROOT / "configs/default.yaml").read_text(encoding="utf-8")
    )
    cfg["vlm"]["backend"] = "openai_compatible"
    if args.local_url:
        cfg["vlm"]["openai_compatible"]["base_url"] = args.local_url
    tolerance = float(
        yaml.safe_load((ROOT / "configs/targets.yaml").read_text(encoding="utf-8"))["targets"][
            "mass_tolerance"
        ]
    )
    cache = args.out.parent / "vlm_compare_cache"
    local = make_vlm_client(cfg, 300.0, 2)
    spark = BasetenMaterialClient(read_secret("BASETEN_API_KEY"), 300.0, 4)
    clients = {"spark": Timed(spark, cache / "spark"), "local": Timed(local, cache / "local")}
    tasks = args.tasks.split(",")
    objects = rows_to_ask()[: args.limit]
    router: dict[str, Any] = {}
    material: dict[str, Any] = {}
    mass: dict[str, Any] = {}
    for tag, client in clients.items():
        if "router" in tasks:
            router[tag] = run_router(client, cfg, args.runs)
        if "material" in tasks:
            material[tag] = run_material(client, cfg, objects, args.workers)
        if "mass" in tasks:
            mass[tag] = run_mass(client, objects, args.workers)
    result = {
        "models": {t: c.model for t, c in clients.items()},
        "local_server": VlmConfig.from_config(cfg).model,
        "summary": summarize(router, material, mass, tolerance),
        "timing": {t: timing([c]) for t, c in clients.items()},
        "router": router,
        "material": material,
        "mass": mass,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("models", "summary", "timing")}, indent=1))


if __name__ == "__main__":
    main()
