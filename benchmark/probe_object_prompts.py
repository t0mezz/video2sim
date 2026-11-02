"""Live probe of SAM 3.1 object prompts for video (S12.9.5, task 9.0).

    uv run python benchmark/probe_object_prompts.py s02_b --out benchmark/results/s12_9_probe.json

At most 4 hosted video requests (about 50 frames each, under $0.10), cached in the run
folder so a rerun makes no call:

1. control: today's text request, to confirm the setup;
2. one object boxed on one frame, direction ``both``;
3. two objects, each boxed on a different frame, one of them prompted on two frames;
4. a box over empty table (negative control: what does SAM return?).

Prompt boxes come from the projected ground-truth boxes
(``evaluation/segmentation.project_box_mask``), so no model guess is involved. The plan names
``s02_a``; the finished dev run is ``s02_b`` (same scene, other video), which is used instead.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from video2sim.evaluation.alignment import align_run, grid_board_layout
from video2sim.evaluation.scenes import GtObject, load_scene
from video2sim.evaluation.segmentation import project_box_mask
from video2sim.models.hosted import (
    SAM_MODEL,
    ResponseCache,
    decode_sam_mask,
    file_digest,
    parse_sam_text,
    read_secret,
)
from video2sim.models.sam_prompts import ObjectPrompt, build_object_prompt
from video2sim.runner import OUTPUT_FILE
from video2sim.schemas import WorldModel
from video2sim.stages.tracking import MetaVideoSamClient, TrackingConfig, encode_clip

ROOT = Path(__file__).resolve().parent.parent
MIN_BILLED_FRAMES = 50  # dev.meta.ai pricing bills at least 50 frames per request
EMPTY_BOX_PX = 160  # side of the empty-table box of the negative control


def _mask_box(mask: NDArray[np.bool_]) -> tuple[int, int, int, int]:
    """Inclusive (x1, y1, x2, y2) bounding box of a non-empty mask, in media pixels."""
    rows, cols = np.nonzero(mask)
    return int(np.min(cols)), int(np.min(rows)), int(np.max(cols)), int(np.max(rows))


def _iou(a: NDArray[np.bool_], b: NDArray[np.bool_]) -> float:
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return inter / union if union else 1.0


def _spread_from(pos: int, n: int) -> list[int]:
    """The prompted frame plus three others at fixed offsets, clipped into the clip."""
    out = [pos]
    for off in (5, 10, -5):
        q = min(max(pos + off, 0), n - 1)
        if q not in out:
            out.append(q)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--runs", type=Path, default=ROOT / "runs_instances")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="print prompts, send nothing")
    args = parser.parse_args()

    cfg: dict[str, Any] = yaml.safe_load(
        (ROOT / "configs/default.yaml").read_text(encoding="utf-8")
    )
    tcfg = TrackingConfig.from_config(cfg)
    run_dir = args.runs / args.run_id
    scene_id = args.run_id.rsplit("_", 1)[0]
    world = WorldModel.model_validate_json(
        (run_dir / "reconstruction" / OUTPUT_FILE).read_text(encoding="utf-8")
    )
    gt = load_scene(ROOT / "benchmark/scenes" / scene_id / "ground_truth.json")
    if gt.board != "grid":
        raise ValueError(f"probe needs a grid-board scene, got board {gt.board!r}")
    layout = grid_board_layout(float(cfg["capture"]["marker"]["tag_size_m"]))
    family = str(cfg["capture"]["marker"]["family"])
    alignment = align_run(world, run_dir, layout, family)
    if alignment is None:
        raise ValueError(f"could not align {args.run_id} to its board")
    poses = {f.frame_index: alignment.pose_in_board(f.T_world_from_camera) for f in world.frames}
    order = [f.frame_index for f in world.frames]
    n = len(order)
    width, height = world.intrinsics.width_px, world.intrinsics.height_px

    rigid: list[GtObject] = [o for o in gt.objects if not o.deformable]
    masks: dict[str, list[NDArray[np.bool_]]] = {
        o.object_id: [project_box_mask(o, world.intrinsics, poses[i]) for i in order] for o in rigid
    }
    areas = {oid: [int(m.sum()) for m in ms] for oid, ms in masks.items()}
    ranked = sorted(areas, key=lambda oid: float(np.median(areas[oid])), reverse=True)
    if len(ranked) < 2 or max(areas[ranked[0]]) == 0:
        raise ValueError("need two ground-truth objects visible from some frame")
    obj_a, obj_b = ranked[0], ranked[1]
    fa = int(np.argmax(areas[obj_a]))
    fb = max(
        (i for i in range(n) if i != fa and areas[obj_b][i] > 0),
        key=lambda i: (areas[obj_b][i], -abs(i - fa)),
        default=-1,
    )
    far = [i for i in range(n) if abs(i - fa) >= 10 and areas[obj_a][i] > 0]
    if not far:
        raise ValueError("no second frame far from the first for object A")
    fa2 = max(far, key=lambda i: areas[obj_a][i])
    if len({fa, fb, fa2}) < 3 or min(fa, fb, fa2) < 0:
        raise ValueError("could not spread the prompted frames apart")

    def prompt_box(oid: str, pos: int, prompt_id: int) -> ObjectPrompt:
        x1, y1, x2, y2 = _mask_box(masks[oid][pos])
        return ObjectPrompt(object_id=prompt_id, box=(float(x1), float(y1), float(x2), float(y2)))

    union = [functools.reduce(np.logical_or, [masks[oid][i] for oid in masks]) for i in range(n)]
    candidates = []
    for i in range(n):
        for y in range(0, height - EMPTY_BOX_PX + 1, 80):
            for x in range(0, width - EMPTY_BOX_PX + 1, 80):
                if not union[i][y : y + EMPTY_BOX_PX, x : x + EMPTY_BOX_PX].any():
                    dist = abs(x + EMPTY_BOX_PX / 2 - width / 2) + abs(
                        y + EMPTY_BOX_PX / 2 - height / 2
                    )
                    candidates.append((dist, i, x, y))
    if not candidates:
        raise ValueError("no empty table region for the negative control")
    # Closest to the image centre: bare table, not a background corner.
    _, ei, ex, ey = min(candidates)
    empty = (ei, ex, ey, ex + EMPTY_BOX_PX - 1, ey + EMPTY_BOX_PX - 1)

    pa = prompt_box(obj_a, fa, 1)
    pb = prompt_box(obj_b, fb, 2)
    pa2 = prompt_box(obj_a, fa2, 1)
    empty_prompt = ObjectPrompt(
        object_id=1,
        box=(float(empty[1]), float(empty[2]), float(empty[3]), float(empty[4])),
    )
    size = (width, height)
    requests: list[dict[str, Any]] = [
        {"label": "control", "prompt": tcfg.prompt, "eval": {}},
        {
            "label": "single_box",
            "prompt": build_object_prompt({fa: [pa]}, size),
            "eval": {1: (obj_a, _spread_from(fa, n))},
        },
        {
            "label": "two_objects",
            "prompt": build_object_prompt({fa: [pa], fa2: [pa2], fb: [pb]}, size),
            "eval": {1: (obj_a, _spread_from(fa, n)), 2: (obj_b, _spread_from(fb, n))},
        },
        {
            "label": "empty_table",
            "prompt": build_object_prompt({empty[0]: [empty_prompt]}, size),
            "eval": {},
        },
    ]
    plan = {
        "run": args.run_id,
        "media": {"w": width, "h": height},
        "clip_frames": n,
        "objects": {"A": obj_a, "B": obj_b},
        "prompted_clip_positions": {"fa": fa, "fa2": fa2, "fb": fb, "empty": empty[0]},
        "prompts": [{"label": r["label"], "prompt": r["prompt"]} for r in requests],
    }
    print(json.dumps(plan, indent=1), flush=True)
    if args.dry_run:
        return

    work_dir = run_dir / "segmentation.work" / "probe_object_prompts"
    work_dir.mkdir(parents=True, exist_ok=True)
    paths = [run_dir / f.image_path for f in world.frames]
    encode_clip(paths, tcfg.fps, work_dir / "clip.mp4")
    # Key on the frames, not the clip bytes, so a new encoder build does not void the cache.
    digest = hashlib.sha256(json.dumps([file_digest(p) for p in paths]).encode()).hexdigest()
    client = MetaVideoSamClient(read_secret("META_MODEL_API_KEY"), tcfg.timeout_s)
    cache = ResponseCache(work_dir / "cache")
    clip = work_dir / "clip.mp4"

    rows = []
    for request in requests:
        # A rejection carries its error text as the result, so catch everything here.
        label, prompt = str(request["label"]), str(request["prompt"])
        row: dict[str, Any] = {"label": label, "prompt": prompt}
        try:
            text = cache.get_or_call(
                ["sam_video", SAM_MODEL, prompt, "both", f"{tcfg.fps}", digest],
                functools.partial(client.track, clip, prompt, "both"),
            )
            row["accepted"] = True
        except Exception as e:
            row["accepted"] = False
            row["error"] = f"{type(e).__name__}: {e}"[:500]
            rows.append(row)
            print(json.dumps({k: row[k] for k in ("label", "accepted", "error")}), flush=True)
            continue
        try:
            records = parse_sam_text(text)
            row["parse_ok"] = True
        except Exception as e:
            row["parse_ok"] = False
            row["error"] = f"{type(e).__name__}: {e}"[:500]
            rows.append(row)
            continue
        by_id: dict[int, list[int]] = {}
        for record in records:
            by_id.setdefault(record.object_id, []).append(record.frame)
        row["ids_returned"] = sorted(by_id)
        row["frames_returned"] = {str(k): sorted(v) for k, v in sorted(by_id.items())}
        row["ids_preserved"] = sorted(by_id) == sorted(request["eval"]) if request["eval"] else None
        ious: dict[str, dict[str, float | None]] = {}
        for oid, (gt_oid, positions) in request["eval"].items():
            obj_ious: dict[str, float | None] = {}
            for pos in positions:
                hit = next((r for r in records if r.object_id == oid and r.frame == pos), None)
                if hit is None:
                    obj_ious[str(order[pos])] = None
                else:
                    obj_ious[str(order[pos])] = _iou(decode_sam_mask(hit), masks[gt_oid][pos])
            ious[str(oid)] = obj_ious
        row["mask_iou_by_frame_index"] = ious
        row["billed_frames"] = max(n, MIN_BILLED_FRAMES)
        rows.append(row)
        print(
            json.dumps(
                {k: row[k] for k in ("label", "accepted", "parse_ok", "ids_preserved")},
            ),
            flush=True,
        )

    result = {
        **plan,
        "prompt_frames_are_clip_positions": True,
        "prompt_in_input_text_part": True,
        "requests": rows,
        "gate": {
            "single_box_accepted": next(
                (r.get("accepted", False) for r in rows if r["label"] == "single_box"), False
            ),
            "ids_preserved_everywhere": all(
                r["ids_preserved"] for r in rows if r["ids_preserved"] is not None
            ),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
