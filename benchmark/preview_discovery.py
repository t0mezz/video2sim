"""Draw the discovery masks onto the sampled frames of finished runs (P2.1).

    uv run python benchmark/preview_discovery.py s01_b s02_b

Reads ``runs/<id>/segmentation/discovery/*.npz`` (written by ``run_discovery.py``) and the
frames in ``runs/<id>/capture/frames``. Writes one image per frame to
``runs/<id>/segmentation/preview/`` plus ``montage.png`` with all frames of the run. Each mask
gets its own colour, an outline and its label; masks that share a label share nothing else, so
duplicates of one object are easy to spot.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
WIDTH = 960
PALETTE = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
    (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212),
    (0, 128, 128), (170, 110, 40),
]  # fmt: skip


def render(image: np.ndarray, labels: np.ndarray, masks: np.ndarray) -> np.ndarray:  # type: ignore[type-arg]
    scale = WIDTH / image.shape[1]
    out = cv2.resize(image, (WIDTH, round(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    overlay = out.copy()
    drawn: list[tuple[str, np.ndarray, tuple[int, int, int]]] = []  # type: ignore[type-arg]
    for i, (label, mask) in enumerate(zip(labels, masks, strict=True)):
        small = cv2.resize(mask.astype(np.uint8), (out.shape[1], out.shape[0]), cv2.INTER_NEAREST)
        rgb = PALETTE[i % len(PALETTE)]
        overlay[small.astype(bool)] = rgb[::-1]
        drawn.append((str(label), small, rgb))
    out = cv2.addWeighted(overlay, 0.45, out, 0.55, 0)
    for label, small, rgb in drawn:
        contours, _ = cv2.findContours(small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, rgb[::-1], 2)
        ys, xs = np.nonzero(small)
        if len(xs):
            org = (int(xs.min()), max(14, int(ys.min()) - 4))
            cv2.putText(out, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, rgb[::-1], 1, cv2.LINE_AA)
    return out


def write_montage(tiles: list[np.ndarray], out_dir: Path) -> None:  # type: ignore[type-arg]
    """All frames of a run in a grid of two columns, saved as ``montage.png``."""
    if not tiles:
        return
    small = [cv2.resize(t, (WIDTH // 2, tiles[0].shape[0] // 2)) for t in tiles]
    rows = [np.hstack(small[i : i + 2]) for i in range(0, len(small), 2)]
    if len(rows) > 1 and rows[-1].shape != rows[0].shape:
        pad = np.zeros_like(rows[0])
        pad[:, : rows[-1].shape[1]] = rows[-1]
        rows[-1] = pad
    cv2.imwrite(str(out_dir / "montage.png"), np.vstack(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_ids", nargs="+")
    parser.add_argument("--runs", type=Path, default=ROOT / "runs")
    args = parser.parse_args()

    for run_id in args.run_ids:
        run_dir = args.runs / run_id
        out_dir = run_dir / "segmentation" / "preview"
        out_dir.mkdir(parents=True, exist_ok=True)
        tiles = []
        for npz in sorted((run_dir / "segmentation" / "discovery").glob("*.npz")):
            data = np.load(npz)
            frame = cv2.imread(str(run_dir / "capture" / "frames" / f"{npz.stem}.png"))
            if frame is None:
                raise FileNotFoundError(f"frame {npz.stem} of {run_id}")
            tile = render(frame, data["labels"], data["masks"])
            cv2.imwrite(str(out_dir / f"{npz.stem}.png"), tile)
            tiles.append(tile)
        write_montage(tiles, out_dir)
        print(f"{run_id}: {len(tiles)} frames -> {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
