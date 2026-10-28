"""Draw the masks of fixed SAM prompts onto the sampled frames of finished runs (P2.1).

    uv run python benchmark/preview_prompts.py s01_b s02_b --prompt "objects on a table"

Reads the SAM responses cached by the prompt experiment in
``runs/<id>/segmentation/prompt_exp_cache`` (no API call is made; a missing response is an
error). Writes one image per frame and a ``montage.png`` to
``runs/<id>/segmentation/preview_prompts/<prompt>/``, next to the method A previews and not in
them. Each mask is drawn like in ``preview_discovery.py`` and numbered, because a fixed prompt
carries no per-object label.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import yaml
from preview_discovery import render, write_montage

from video2sim.runner import OUTPUT_FILE
from video2sim.schemas import WorldModel
from video2sim.stages import discovery_vlm_sam as d
from video2sim.stages.segmentation import sample_frame_indices

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPTS = ("thing", "object on table", "objects on a table")


def slug(prompt: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", prompt.lower()).strip("_")


def _cached_only() -> str:
    raise RuntimeError("response not cached; run the prompt experiment first")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_ids", nargs="+")
    parser.add_argument("--runs", type=Path, default=ROOT / "runs")
    parser.add_argument(
        "--prompt", action="append", help="repeatable; default: the three vague ones"
    )
    args = parser.parse_args()
    prompts = tuple(args.prompt) if args.prompt else DEFAULT_PROMPTS
    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    n_frames = int(cfg["segmentation"]["discovery"]["n_sample_frames"])

    for run_id in args.run_ids:
        run_dir = args.runs / run_id
        world = WorldModel.model_validate_json(
            (run_dir / "reconstruction" / OUTPUT_FILE).read_text(encoding="utf-8")
        )
        cache = d.ResponseCache(run_dir / "segmentation" / "prompt_exp_cache")
        picks = [world.frames[i] for i in sample_frame_indices(len(world.frames), n_frames)]
        for prompt in prompts:
            out_dir = run_dir / "segmentation" / "preview_prompts" / slug(prompt)
            out_dir.mkdir(parents=True, exist_ok=True)
            tiles = []
            for frame in picks:
                image = cv2.imread(str(run_dir / frame.image_path))
                if image is None:
                    raise FileNotFoundError(f"frame {frame.image_path} of {run_id}")
                text = cache.get_or_call(
                    ["sam", d.SAM_MODEL, prompt, d.file_digest(run_dir / frame.image_path)],
                    _cached_only,
                )
                masks = [d.decode_sam_mask(r) for r in d.parse_sam_text(text)]
                labels = np.array([f"{i + 1}" for i in range(len(masks))])
                stack = np.array(masks, dtype=bool).reshape(len(masks), *image.shape[:2])
                tile = render(image, labels, stack)
                caption = f'"{prompt}": {len(masks)} masks'
                cv2.putText(tile, caption, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
                cv2.putText(
                    tile, caption, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2
                )
                cv2.imwrite(str(out_dir / f"{frame.frame_index:05d}.png"), tile)
                tiles.append(tile)
            write_montage(tiles, out_dir)
            print(f"{run_id} {prompt!r}: {len(tiles)} frames -> {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
