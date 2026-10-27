"""Stage 1: capture and quality gate (P1.4).

Input:  a video file.
Output: ``CaptureResult``: accepted frames, or reasons the video must be re-filmed.

Frames are chosen blur-aware: the video is cut into windows of ``1 / target_fps`` seconds and
the sharpest frame of each window is kept, so a short shake costs one window instead of a
blurred frame. The gate then looks at the kept frames and reports every problem it finds in
one message each, so the user can fix them all in one retake: blur, exposure, marker not
visible, too few frames. Reconstruction-side checks (registration, reprojection, camera
coverage) are in ``reconstruction.check_reconstruction``.

Sharpness is the variance of the Laplacian at a fixed width. It depends on how textured the
scene is, so blur is judged against the video's own sharp level (a high percentile) and only
a very low median is judged absolutely. OpenCV applies the rotation metadata of phone video
when it decodes, so the frames match the intrinsics estimated later. Time is frame index over
the nominal frame rate, so a variable frame rate shifts the windows slightly but never the
pixels.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from video2sim.schemas import CaptureResult

# Fixed width for the sharpness measure, so the value does not depend on the phone's resolution.
_SHARPNESS_WIDTH_PX = 960


@dataclass(frozen=True)
class CaptureQualityConfig:
    """Values from ``capture`` in ``configs/default.yaml``."""

    target_fps: float
    marker_family: str
    blur_relative_floor: float
    blur_reference_percentile: float
    min_median_sharpness: float
    max_blurred_fraction: float
    min_median_brightness: float
    max_median_brightness: float
    max_bad_exposure_fraction: float
    min_marker_fraction: float
    min_usable_frames: int

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> CaptureQualityConfig:
        cap, q = cfg["capture"], cfg["capture"]["quality"]
        return cls(
            target_fps=float(cap["target_fps"]),
            marker_family=str(cap["marker"]["family"]),
            blur_relative_floor=float(q["blur_relative_floor"]),
            blur_reference_percentile=float(q["blur_reference_percentile"]),
            min_median_sharpness=float(q["min_median_sharpness"]),
            max_blurred_fraction=float(q["max_blurred_fraction"]),
            min_median_brightness=float(q["min_median_brightness"]),
            max_median_brightness=float(q["max_median_brightness"]),
            max_bad_exposure_fraction=float(q["max_bad_exposure_fraction"]),
            min_marker_fraction=float(q["min_marker_fraction"]),
            min_usable_frames=int(q["min_usable_frames"]),
        )


@dataclass(frozen=True)
class FrameScore:
    index: int  # decoded frame number
    time_s: float
    sharpness: float
    brightness: float  # median grey level, 0 to 255


@dataclass(frozen=True)
class CaptureReport:
    """What the gate measured, for the QA report and for tuning. Not a pipeline contract."""

    scores: list[FrameScore]  # the sharpest frame of every window
    usable: list[FrameScore]  # those that are neither blurred nor badly exposed
    metrics: dict[str, float]


def score_frame(gray: NDArray[np.uint8]) -> tuple[float, float]:
    """Return ``(sharpness, median brightness)`` of one grey frame."""
    import cv2

    w = _SHARPNESS_WIDTH_PX
    small = (
        gray
        if gray.shape[1] <= w
        else cv2.resize(
            gray, (w, round(gray.shape[0] * w / gray.shape[1])), interpolation=cv2.INTER_AREA
        )
    )
    return float(cv2.Laplacian(small, cv2.CV_64F).var()), float(np.median(small))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _window_winners(video: Path, target_fps: float) -> tuple[list[FrameScore], float, int]:
    """Pass 1: score every frame, keep the sharpest of each window. Returns winners, fps, count."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"cannot open video {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        raise ValueError(f"video {video} reports no frame rate")
    best: dict[int, FrameScore] = {}
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        window = int((index / fps) * target_fps)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharp, bright = score_frame(np.asarray(gray, dtype=np.uint8))
        if window not in best or sharp > best[window].sharpness:
            best[window] = FrameScore(index, index / fps, sharp, bright)
        index += 1
    cap.release()
    return [best[w] for w in sorted(best)], fps, index


def sample_and_check(
    video: Path, run_dir: Path, frames_subdir: str, cfg: CaptureQualityConfig
) -> tuple[CaptureResult, CaptureReport]:
    """Pick frames, run the gate, and write the accepted frames under ``run_dir/frames_subdir``.

    A rejected video leaves no frames behind. Frame paths in the result are relative to ``run_dir``.
    """
    import cv2

    from video2sim.stages.marker_detection import detect_tags

    winners, fps, n_decoded = _window_winners(video, cfg.target_fps)
    digest = _sha256(video)
    if not winners:
        return _reject(digest, ["The video has no readable frames."]), CaptureReport([], [], {})

    sharp = np.array([s.sharpness for s in winners])
    median_sharp = float(np.median(sharp))
    # The reference is a high percentile, not the median: if most of the video is blurred the
    # median is itself a blurred value and nothing would look blurred against it.
    reference_sharp = float(np.percentile(sharp, cfg.blur_reference_percentile))
    blurred = sharp < cfg.blur_relative_floor * reference_sharp
    bright = np.array([s.brightness for s in winners])
    bad_exposure = (bright < cfg.min_median_brightness) | (bright > cfg.max_median_brightness)
    keep = ~blurred & ~bad_exposure
    usable = [s for s, k in zip(winners, keep, strict=True) if k]

    metrics: dict[str, float] = {
        "duration_s": n_decoded / fps,
        "windows": float(len(winners)),
        "median_sharpness": median_sharp,
        "reference_sharpness": reference_sharp,
        "blurred_fraction": float(blurred.mean()),
        "median_brightness": float(np.median(bright)),
        "bad_exposure_fraction": float(bad_exposure.mean()),
        "usable_frames": float(len(usable)),
    }

    # Pass 2: read the usable frames back (by index) and look for the marker in each.
    frames_dir = run_dir / frames_subdir
    tag_hits = 0
    written: list[str] = []
    if usable:
        wanted = {s.index: n for n, s in enumerate(usable)}
        cap = cv2.VideoCapture(str(video))
        index = 0
        while len(written) < len(usable):
            ok, frame = cap.read()
            if not ok:
                break
            if index in wanted:
                gray = np.asarray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), dtype=np.uint8)
                tag_hits += bool(detect_tags(gray, cfg.marker_family))
                rel = Path(frames_subdir) / f"{wanted[index]:05d}.png"
                written.append(str(rel))
                frames_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(run_dir / rel), frame)
            index += 1
        cap.release()
    marker_fraction = tag_hits / len(usable) if usable else 0.0
    metrics["marker_fraction"] = marker_fraction

    reasons = _reasons(metrics, cfg)
    report = CaptureReport(winners, usable, metrics)
    if reasons:
        for rel_path in written:  # drop frames written before a later check failed
            (run_dir / rel_path).unlink(missing_ok=True)
        return _reject(digest, reasons), report
    return (
        CaptureResult(
            input_video_sha256=digest, accepted=True, frame_paths=written, fps=cfg.target_fps
        ),
        report,
    )


def _reject(digest: str, reasons: list[str]) -> CaptureResult:
    return CaptureResult(input_video_sha256=digest, accepted=False, reasons=reasons)


def _reasons(m: dict[str, float], cfg: CaptureQualityConfig) -> list[str]:
    """One specific, actionable message per failed check.

    Checks that are a consequence of another are left out: sharpness falls with brightness, so
    a dark video is not also called out of focus, and a video whose frames were thrown away for
    blur or exposure is not also told to film longer.
    """
    out = []
    exposure_bad = m["bad_exposure_fraction"] > cfg.max_bad_exposure_fraction
    blur_bad = m["blurred_fraction"] > cfg.max_blurred_fraction
    soft = m["median_sharpness"] < cfg.min_median_sharpness
    if soft and not exposure_bad:
        out.append(
            "The whole video is out of focus or too soft. Tap and hold on an object to lock "
            "focus before recording, and keep the phone 30 to 80 cm from the nearest objects."
        )
    if blur_bad and not exposure_bad:
        out.append(
            f"{100 * m['blurred_fraction']:.0f}% of the video is motion-blurred (limit "
            f"{100 * cfg.max_blurred_fraction:.0f}%). Move more slowly, with your body and not "
            "your wrists, especially at corners and when changing height."
        )
    if exposure_bad:
        dark = m["median_brightness"] < 128
        out.append(
            f"The video is {'too dark' if dark else 'overexposed'}: "
            f"{100 * m['bad_exposure_fraction']:.0f}% of frames are outside the usable "
            f"brightness range. {'Turn on more lights' if dark else 'Reduce glare and light'}"
            " and keep exposure steady."
        )
    if m["usable_frames"] < cfg.min_usable_frames and not (soft or blur_bad or exposure_bad):
        out.append(
            f"Only {m['usable_frames']:.0f} usable frames (need {cfg.min_usable_frames}). "
            "Film a longer orbit around the table."
        )
    if m["marker_fraction"] < cfg.min_marker_fraction and not exposure_bad:
        out.append(
            f"The scale marker is visible in only {100 * m['marker_fraction']:.0f}% of the usable "
            f"frames (need {100 * cfg.min_marker_fraction:.0f}%). Keep the printed marker in view "
            "for most of the video."
        )
    return out
