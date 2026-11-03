"""Tests for the two-tag A4 marker sheet in benchmark/make_board.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PAIR_IDS = (0, 1)
TAG_MM = 100.0
PAGE_MM = (297.0, 210.0)  # A4 landscape


def _make_board() -> Any:
    path = Path(__file__).parents[1] / "benchmark" / "make_board.py"
    spec = importlib.util.spec_from_file_location("make_board", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pattern_image(board: Any, tag_id: int, side_px: int) -> Any:
    """Render one tag from the committed bit pattern, one pixel per unit cell."""
    pattern = board.TAG_PATTERNS[tag_id]
    img = np.full((6, 6), 255, np.uint8)
    for row, bits in enumerate(pattern):
        for col, bit in enumerate(bits):
            if bit == "1":
                img[row, col] = 0
    return np.repeat(np.repeat(img, side_px // 6, axis=0), side_px // 6, axis=1)


def test_pair_patterns_match_opencv_dictionary() -> None:
    board = _make_board()
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    for tag_id in PAIR_IDS:
        expected = cv2.aruco.generateImageMarker(dictionary, tag_id, 60)
        assert _pattern_image(board, tag_id, 60).tolist() == expected.tolist()


def test_pair_tags_decode_to_their_ids() -> None:
    # Render the committed patterns (not OpenCV's own drawing) and decode them.
    board = _make_board()
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    sheet = np.full((700, 1400), 255, np.uint8)
    for i, tag_id in enumerate(PAIR_IDS):
        sheet[50:650, 50 + i * 680 : 650 + i * 680] = _pattern_image(board, tag_id, 600)
    _, ids, _ = detector.detectMarkers(sheet)
    assert ids is not None
    assert sorted(int(v) for v in ids.flatten()) == [0, 1]


def test_pair_layout_fits_a4_landscape() -> None:
    board = _make_board()
    _, layout = board.build_pair(PAIR_IDS, TAG_MM)
    assert layout["page_mm"] == [PAGE_MM[0], PAGE_MM[1]]
    assert layout["tag_edge_mm_nominal"] == TAG_MM
    assert sorted(layout["tags"]) == ["0", "1"]
    page_w, page_h = PAGE_MM
    boxes = []
    for tag_id in PAIR_IDS:
        entry = layout["tags"][str(tag_id)]
        x, y = entry["lower_left_mm"]
        cx, cy = entry["center_mm"]
        assert (cx, cy) == (x + TAG_MM / 2.0, y + TAG_MM / 2.0)
        assert x >= 0.0 and x + TAG_MM <= page_w
        assert y >= 0.0 and y + TAG_MM <= page_h
        boxes.append((x, y))
    (x0, _), (x1, _) = boxes
    assert x1 - x0 >= TAG_MM  # side by side, not overlapping


def test_pair_pdf_has_a4_landscape_mediabox(tmp_path: Path) -> None:
    board = _make_board()
    out = tmp_path / "pair.pdf"
    canvas, _ = board.build_pair(PAIR_IDS, TAG_MM)
    board.write_pdf(out, PAGE_MM[0], PAGE_MM[1], canvas)
    raw = out.read_bytes()
    assert raw.startswith(b"%PDF-1.4")
    pt_per_mm = 72.0 / 25.4
    # Same %.3f-then-strip formatting as make_board._num.
    nums = [f"{v * pt_per_mm:.3f}".rstrip("0").rstrip(".") for v in PAGE_MM]
    assert f"[0 0 {nums[0]} {nums[1]}]".encode() in raw
