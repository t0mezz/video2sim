"""Tracking (P2.2): one SAM 3.1 video request per scene, then identity repair in 3D.

The registered frames are encoded as a clip and sent with the concept prompt of discovery.
The API answers with one line per frame and a stable integer id per object (16 objects per
frame at most). Ids hold only while an object stays in view (dev.meta.ai/docs/sam), so the
tracks are repaired afterwards in ``track_repair``.

Cost: video is billed per frame, at least 50 per request. s01 with 46 frames was billed 50,
and a typical scene of 84 frames costs about $0.017 (docs/licenses.md).
"""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from video2sim.schemas import WorldModel
from video2sim.stages.discovery_vlm_sam import (
    META_URL,
    SAM_MODEL,
    MaskDecoder,
    ResponseCache,
    _post_json,
    _sse_deltas,
    decode_sam_mask,
    file_digest,
    parse_sam_text,
)
from video2sim.stages.segmentation import BoolMask

Direction = Literal["both", "forward", "backward"]


@dataclass(frozen=True)
class TrackingConfig:
    """Values from ``segmentation.tracking`` in ``configs/default.yaml``."""

    prompt: str
    fps: float
    direction: Direction
    timeout_s: float

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> TrackingConfig:
        t = cfg["segmentation"]["tracking"]
        prompt, fps = str(t["prompt"]).strip(), float(t["fps"])
        direction = str(t["direction"])
        if not prompt:
            raise ValueError("tracking prompt must not be empty")
        if fps <= 0:
            raise ValueError("tracking fps must be positive")
        if direction == "both":
            return cls(prompt, fps, "both", float(t["timeout_s"]))
        if direction == "forward":
            return cls(prompt, fps, "forward", float(t["timeout_s"]))
        if direction == "backward":
            return cls(prompt, fps, "backward", float(t["timeout_s"]))
        raise ValueError(f"unknown propagation direction {direction!r}")


@dataclass(frozen=True)
class Track:
    """One tracked object: a full-frame mask per world-model frame index where it is visible."""

    track_id: int
    masks: Mapping[int, BoolMask]


class VideoSamClient(Protocol):
    def track(self, video: Path, prompt: str, direction: Direction) -> str:
        """SAM 3.1 special-token text for a clip, one line per frame."""
        ...


class MetaVideoSamClient:
    def __init__(self, key: str, timeout_s: float = 600.0) -> None:
        self._key, self._timeout_s = key, timeout_s

    def track(self, video: Path, prompt: str, direction: Direction) -> str:
        # The data URL form is not shown in the docs; it was checked on s01 (2026-10-28).
        url = "data:video/mp4;base64," + base64.b64encode(video.read_bytes()).decode()
        body = {
            "model": SAM_MODEL,
            "stream": True,
            "metadata": {"propagation_direction": direction},
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_video", "video_url": url},
                    ],
                }
            ],
        }
        raw = _post_json(META_URL, self._key, body, self._timeout_s).decode()
        return "\n".join(_sse_deltas(raw))


class FakeVideoSamClient:
    """Replays one recorded answer, for tests that must not touch a network."""

    def __init__(self, answer: str) -> None:
        self._answer = answer
        self.calls = 0

    def track(self, video: Path, prompt: str, direction: Direction) -> str:
        self.calls += 1
        return self._answer


def encode_clip(frames: list[Path], fps: float, out: Path) -> None:
    """Write the frames as an MP4 clip (MPEG-4 part 2, an accepted codec) at ``fps``."""
    import cv2  # heavy import, kept out of module import time

    first = cv2.imread(str(frames[0]))
    if first is None:
        raise ValueError(f"cannot read {frames[0]}")
    height, width = first.shape[:2]
    # VideoWriter_fourcc is missing from the cv2 stubs.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(str(out), fourcc, fps, (width, height))
    try:
        for path in frames:
            image = cv2.imread(str(path))
            if image is None or image.shape[:2] != (height, width):
                raise ValueError(f"{path} is unreadable or differs in size from the first frame")
            writer.write(image)
    finally:
        writer.release()


def tracks_from_text(
    text: str, frame_indices: list[int], decode: MaskDecoder = decode_sam_mask
) -> list[Track]:
    """Tracks of an answer. ``<Nf>`` is the position in the clip, mapped to a world frame index."""
    masks: dict[int, dict[int, BoolMask]] = {}
    for record in parse_sam_text(text):
        if not 0 <= record.frame < len(frame_indices):
            raise ValueError(f"answer names frame {record.frame}, clip has {len(frame_indices)}")
        masks.setdefault(record.object_id, {})[frame_indices[record.frame]] = decode(record)
    return [Track(i, masks[i]) for i in sorted(masks)]


def track_objects(
    world: WorldModel,
    run_dir: Path,
    client: VideoSamClient,
    cache: ResponseCache,
    cfg: TrackingConfig,
    work_dir: Path,
    decode: MaskDecoder = decode_sam_mask,
) -> list[Track]:
    """Track every object over all registered frames with one hosted request (cached)."""
    paths = [run_dir / f.image_path for f in world.frames]
    work_dir.mkdir(parents=True, exist_ok=True)
    clip = work_dir / "clip.mp4"
    encode_clip(paths, cfg.fps, clip)
    # Key on the frames, not the clip bytes, so a new encoder build does not void the cache.
    digest = hashlib.sha256(json.dumps([file_digest(p) for p in paths]).encode()).hexdigest()
    call: Callable[[], str] = lambda: client.track(clip, cfg.prompt, cfg.direction)  # noqa: E731
    text = cache.get_or_call(
        ["sam_video", SAM_MODEL, cfg.prompt, cfg.direction, f"{cfg.fps}", digest], call
    )
    return tracks_from_text(text, [f.frame_index for f in world.frames], decode)
