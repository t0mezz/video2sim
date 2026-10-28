from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from video2sim.stages.discovery_sam_generic import GenericSamBackend
from video2sim.stages.discovery_vlm_sam import ResponseCache, SamRecord
from video2sim.stages.segmentation import BoolMask

ONE = "<0f>0<|box;x1=1;y1=1;x2=2;y2=2;w=8;h=8|><|mask;x=0;y=0;data=2,2,~a|>"
TWO = ONE + ",1<|box;x1=4;y1=4;x2=5;y2=5;w=8;h=8|><|mask;x=0;y=0;data=2,2,~b|>"


class _Sam:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def segment(self, image: Path, noun: str) -> str:
        self.calls.append((image.name, noun))
        return TWO if image.name == "b.jpg" else ONE


def _decode(record: SamRecord) -> BoolMask:
    mask = np.zeros((8, 8), dtype=bool)
    mask[record.object_id, record.object_id] = True
    return mask


def _frames(tmp_path: Path) -> list[tuple[int, Path]]:
    frames = []
    for i, name in enumerate(("a.jpg", "b.jpg", "c.jpg")):
        (tmp_path / name).write_bytes(bytes([i]) * 8)
        frames.append((i * 10, tmp_path / name))
    return frames


def test_one_prompt_per_frame_and_masks_are_unlabeled(tmp_path: Path) -> None:
    sam = _Sam()
    backend = GenericSamBackend(sam, ResponseCache(tmp_path / "c"), "objects on a table", _decode)
    out = backend.discover(_frames(tmp_path))
    assert [f.frame_index for f in out] == [0, 10, 20]
    assert [len(f.masks) for f in out] == [1, 2, 1]
    assert all(m.label is None for f in out for m in f.masks)
    assert sorted(sam.calls) == [(n, "objects on a table") for n in ("a.jpg", "b.jpg", "c.jpg")]


def test_reruns_are_served_from_the_cache(tmp_path: Path) -> None:
    sam = _Sam()
    backend = GenericSamBackend(sam, ResponseCache(tmp_path / "c"), "things", _decode)
    frames = _frames(tmp_path)
    backend.discover(frames)
    backend.discover(frames)
    assert len(sam.calls) == 3


def test_parallel_matches_serial(tmp_path: Path) -> None:
    frames = _frames(tmp_path)

    def run(workers: int, cache: str) -> list[list[tuple[int, ...]]]:
        backend = GenericSamBackend(
            _Sam(), ResponseCache(tmp_path / cache), "things", _decode, max_workers=workers
        )
        return [[tuple(np.argwhere(m.mask)[0]) for m in f.masks] for f in backend.discover(frames)]

    assert run(4, "p") == run(1, "s")


def test_empty_prompt_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="prompt"):
        GenericSamBackend(_Sam(), ResponseCache(tmp_path), " ")
