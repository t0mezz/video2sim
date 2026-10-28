from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from video2sim.stages.discovery_vlm_sam import (
    ResponseCache,
    SamRecord,
    VlmSamBackend,
    _sse_deltas,
    decode_sam_mask,
    parse_nouns,
    parse_sam_text,
    read_secret,
)
from video2sim.stages.segmentation import BoolMask

SAM_TEXT = (
    "<0f>0<|box;x1=10;y1=20;x2=50;y2=80;w=640;h=480|><|mask;x=10;y=20;data=60,40,~abc|>,"
    "2<|box;x1=1;y1=2;x2=3;y2=4;w=640;h=480;c=0.9|><|mask;x=0;y=0;c=0.9;data=2,2,!z|>\n"
    "<3f>0<|box;x1=11;y1=21;x2=51;y2=81;w=640;h=480|><|mask;x=11;y=21;data=60,40,~def|>"
)


def test_nouns_are_deduplicated_lowercased_and_limited() -> None:
    answer = 'Sure:\n```json\n["Mug", "mug", "Red Cube", "", 3, "pen"]\n```'
    assert parse_nouns(answer, 2) == ["mug", "red cube"]
    assert parse_nouns("no list here", 5) == []
    assert parse_nouns("[not json]", 5) == []


def test_sam_text_parses_frames_ids_boxes_and_mask_headers() -> None:
    records = parse_sam_text(SAM_TEXT)
    assert [(r.frame, r.object_id) for r in records] == [(0, 0), (0, 2), (3, 0)]
    first = records[0]
    assert first.box == (10.0, 20.0, 50.0, 80.0)
    assert first.frame_size == (640, 480)
    assert first.mask_offset == (10, 20)
    assert first.mask_shape == (60, 40)  # rows first, as the docs stress
    assert first.mask_payload == "~abc"
    assert records[1].mask_payload == "!z"
    assert parse_sam_text("") == []


def test_sse_deltas_keep_only_text_delta_events() -> None:
    raw = (
        'event: x\ndata: {"type": "response.created"}\n\n'
        'data: {"type": "response.output_text.delta", "delta": "<0f>"}\n'
        "data: [DONE]\n"
        'data: {"type": "response.output_text.delta", "delta": "rest"}\n'
    )
    assert _sse_deltas(raw) == ["<0f>", "rest"]


def test_secret_comes_from_env_then_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = tmp_path / ".env"
    env.write_text("A_KEY='from-file'  # a comment\nEMPTY=\n", encoding="utf-8")
    monkeypatch.delenv("A_KEY", raising=False)
    assert read_secret("A_KEY", env) == "from-file"
    monkeypatch.setenv("A_KEY", "from-env")
    assert read_secret("A_KEY", env) == "from-env"
    with pytest.raises(RuntimeError, match="EMPTY"):
        read_secret("EMPTY", env)


class _Lister:
    def __init__(self) -> None:
        self.calls = 0

    def list_objects(self, image: Path) -> str:
        self.calls += 1
        return '["mug", "cube"]' if image.name == "a.jpg" else '["cube", "pen"]'


class _Sam:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def segment(self, image: Path, noun: str) -> str:
        self.calls.append(noun)
        return SAM_TEXT if noun == "mug" else ""


def _decode(record: SamRecord) -> BoolMask:
    return np.ones((4, 4), dtype=bool)


def test_backend_unions_nouns_prompts_sam_and_caches(tmp_path: Path) -> None:
    frames = []
    for i, name in enumerate(("a.jpg", "b.jpg")):
        (tmp_path / name).write_bytes(bytes([i]) * 8)
        frames.append((i * 10, tmp_path / name))
    lister, sam = _Lister(), _Sam()
    backend = VlmSamBackend(lister, sam, ResponseCache(tmp_path / "cache"), _decode, max_nouns=5)

    out = backend.discover(frames)
    assert [f.frame_index for f in out] == [0, 10]
    assert all(m.label == "mug" for f in out for m in f.masks)
    assert [len(f.masks) for f in out] == [3, 3]
    assert sorted(set(sam.calls)) == ["cube", "mug", "pen"]

    backend.discover(frames)  # second run is served from the cache
    assert lister.calls == 2
    assert len(sam.calls) == 6


def test_parallel_backend_matches_the_serial_result(tmp_path: Path) -> None:
    frames = []
    for i, name in enumerate(("a.jpg", "b.jpg", "c.jpg")):
        (tmp_path / name).write_bytes(bytes([i]) * 8)
        frames.append((i * 10, tmp_path / name))

    def run(workers: int, cache: str) -> list[tuple[int, list[str | None]]]:
        backend = VlmSamBackend(
            _Lister(), _Sam(), ResponseCache(tmp_path / cache), _decode, max_workers=workers
        )
        return [(f.frame_index, [m.label for m in f.masks]) for f in backend.discover(frames)]

    assert run(4, "parallel") == run(1, "serial")


def _record(payload: str, rows: int, cols: int) -> SamRecord:
    return SamRecord(0, 0, (4.0, 2.0, 9.0, 5.0), (16, 8), (0, 0), (rows, cols), payload)


def test_decoder_places_a_box_local_mask_in_the_frame() -> None:
    from meta_sam_parser._mask_codec import _encode_raster

    local = [1, 0, 0, 1, 1, 1]  # 2 rows x 3 cols, stretched to the 6 x 4 inclusive box
    record = _record(_encode_raster(local, 3, 2), rows=2, cols=3)
    mask = decode_sam_mask(record)
    assert mask.shape == (8, 16) and mask.dtype == bool
    assert mask[2:6, 4:10].sum() == mask.sum()  # nothing outside the box
    assert mask[2, 4] and not mask[2, 6] and mask[5, 9]
    assert mask.sum() == 16  # four set cells, each stretched to 2 x 2


def test_decoder_rejects_a_corrupt_payload() -> None:
    from meta_sam_parser import InvalidSegmentationMaskError

    with pytest.raises(InvalidSegmentationMaskError):
        decode_sam_mask(_record("!not a payload", rows=2, cols=3))
