from __future__ import annotations

from pathlib import Path

import pytest

from video2sim.models.hosted import (
    ResponseCache,
    SamRecord,
    decode_sam_mask,
    parse_sam_text,
    read_secret,
    sse_deltas,
)

SAM_TEXT = (
    "<0f>0<|box;x1=10;y1=20;x2=50;y2=80;w=640;h=480|><|mask;x=10;y=20;data=60,40,~abc|>,"
    "2<|box;x1=1;y1=2;x2=3;y2=4;w=640;h=480;c=0.9|><|mask;x=0;y=0;c=0.9;data=2,2,!z|>\n"
    "<3f>0<|box;x1=11;y1=21;x2=51;y2=81;w=640;h=480|><|mask;x=11;y=21;data=60,40,~def|>"
)


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
    assert sse_deltas(raw) == ["<0f>", "rest"]


def test_secret_comes_from_env_then_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = tmp_path / ".env"
    env.write_text("A_KEY='from-file'  # a comment\nEMPTY=\n", encoding="utf-8")
    monkeypatch.delenv("A_KEY", raising=False)
    assert read_secret("A_KEY", env) == "from-file"
    monkeypatch.setenv("A_KEY", "from-env")
    assert read_secret("A_KEY", env) == "from-env"
    with pytest.raises(RuntimeError, match="EMPTY"):
        read_secret("EMPTY", env)


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


def test_cache_serves_the_second_call_from_disk(tmp_path: Path) -> None:
    cache, calls = ResponseCache(tmp_path / "cache"), []

    def call() -> str:
        calls.append(1)
        return "answer"

    assert cache.get_or_call(["sam", "mug"], call) == "answer"
    assert cache.get_or_call(["sam", "mug"], call) == "answer"
    assert cache.get_or_call(["sam", "cube"], call) == "answer"
    assert len(calls) == 2
