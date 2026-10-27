from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from video2sim.stages.capture import CaptureQualityConfig, sample_and_check

U8 = NDArray[np.uint8]
W, H, FPS, SECONDS = 640, 480, 10, 6

CFG = CaptureQualityConfig(
    target_fps=5,
    marker_family="aruco_4x4_50",
    blur_relative_floor=0.3,
    blur_reference_percentile=90,
    min_median_sharpness=50,
    max_blurred_fraction=0.3,
    min_median_brightness=35,
    max_median_brightness=235,
    max_bad_exposure_fraction=0.25,
    min_marker_fraction=0.5,
    min_usable_frames=10,
)


def _base_frame(with_tag: bool = True) -> U8:
    rng = np.random.default_rng(0)
    tex = cv2.GaussianBlur(rng.integers(60, 200, (H, W)).astype(np.uint8), (0, 0), 1.0)
    if with_tag:
        tag = cv2.aruco.generateImageMarker(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 0, 180
        )
        tex[150:330, 230:410] = tag
        tex[140:150, 220:420] = tex[330:340, 220:420] = 255  # quiet zone
        tex[140:340, 220:230] = tex[140:340, 410:420] = 255
    return np.asarray(tex, dtype=np.uint8)


def _write(path: Path, frame_fn: Callable[[int], U8], n: int = FPS * SECONDS) -> Path:
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))  # type: ignore[attr-defined]  # stubs lack the fourcc helper
    for i in range(n):
        out.write(cv2.cvtColor(frame_fn(i), cv2.COLOR_GRAY2BGR))
    out.release()
    return path


def _blur(f: U8) -> U8:
    return np.asarray(cv2.GaussianBlur(f, (0, 0), 6), dtype=np.uint8)


def _run(tmp_path: Path, frame_fn: Callable[[int], U8], n: int = FPS * SECONDS):  # type: ignore[no-untyped-def]  # tuple of two result types, only used in this file
    video = _write(tmp_path / "v.avi", frame_fn, n)
    return sample_and_check(video, tmp_path / "run", "frames", CFG)


def test_good_video_is_accepted_with_sampled_frames(tmp_path: Path) -> None:
    base = _base_frame()
    result, report = _run(tmp_path, lambda i: base)
    assert result.accepted and result.reasons == []
    assert result.fps == 5
    assert len(result.frame_paths) == len(report.usable) == SECONDS * 5
    assert all((tmp_path / "run" / p).exists() for p in result.frame_paths)
    assert not Path(result.frame_paths[0]).is_absolute()
    assert report.metrics["marker_fraction"] == 1.0


def test_sampling_keeps_the_sharpest_frame_of_each_window(tmp_path: Path) -> None:
    base = _base_frame()
    # In every 2-frame window (10 fps / 5 fps), only the odd frame is sharp.
    _, report = _run(tmp_path, lambda i: base if i % 2 == 1 else _blur(base))
    assert report.scores and all(s.index % 2 == 1 for s in report.scores)


def test_whole_video_out_of_focus_has_its_own_message(tmp_path: Path) -> None:
    result, _ = _run(tmp_path, lambda i: _blur(_base_frame()))
    assert not result.accepted
    assert any("out of focus" in r for r in result.reasons)


def test_motion_blur_segment_is_reported_with_its_share(tmp_path: Path) -> None:
    base = _base_frame()
    n = FPS * SECONDS
    result, _ = _run(tmp_path, lambda i: _blur(base) if n * 0.3 <= i < n * 0.8 else base)
    assert not result.accepted
    msg = next(r for r in result.reasons if "motion-blurred" in r)
    assert "Move more slowly" in msg and "%" in msg


@pytest.mark.parametrize(("gain", "word"), [(0.1, "too dark"), (1.9, "overexposed")])
def test_exposure_problems(tmp_path: Path, gain: float, word: str) -> None:
    base = _base_frame().astype(np.float64)
    shifted = np.clip(base * gain if gain < 1 else base + 110, 0, 255).astype(np.uint8)
    result, _ = _run(tmp_path, lambda i: shifted)
    assert not result.accepted
    assert any(word in r for r in result.reasons)


def test_missing_marker_is_reported(tmp_path: Path) -> None:
    base = _base_frame(with_tag=False)
    result, _ = _run(tmp_path, lambda i: base)
    assert not result.accepted
    assert any("scale marker" in r and "0%" in r for r in result.reasons)


def test_dark_video_is_not_also_called_out_of_focus(tmp_path: Path) -> None:
    dark = (_base_frame().astype(np.float64) * 0.1).astype(np.uint8)
    result, _ = _run(tmp_path, lambda i: dark)
    assert [("too dark" in r) for r in result.reasons] == [True]


def test_too_short_video(tmp_path: Path) -> None:
    base = _base_frame()
    result, _ = _run(tmp_path, lambda i: base, n=FPS * 1)  # 5 windows, below the 10 required
    assert not result.accepted
    assert any("usable frames" in r for r in result.reasons)


def test_independent_problems_are_reported_together_and_no_frames_are_left(
    tmp_path: Path,
) -> None:
    n = FPS * SECONDS
    base = _base_frame(with_tag=False)
    # Blur burst plus no marker at all: two separate faults, two messages.
    result, _ = _run(tmp_path, lambda i: _blur(base) if n * 0.2 <= i < n * 0.8 else base)
    assert not result.accepted
    assert any("motion-blurred" in r for r in result.reasons)
    assert any("scale marker" in r for r in result.reasons)
    frames = tmp_path / "run" / "frames"
    assert not frames.exists() or not list(frames.glob("*.png"))


def test_unreadable_video_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot open"):
        sample_and_check(tmp_path / "missing.mp4", tmp_path / "run", "frames", CFG)
