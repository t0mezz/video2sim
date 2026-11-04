"""Tracking (P2.2): one SAM 3.1 video request per scene, then identity repair in 3D.

The registered frames are encoded as a clip and sent with one concept prompt.
The API answers with one line per frame and a stable integer id per object (16 objects per
frame at most). Ids hold only while an object stays in view (dev.meta.ai/docs/sam), so the
tracks are repaired afterwards in ``track_repair``.

Cost: video is billed per frame, at least 50 per request. s01 with 46 frames was billed 50,
and a typical scene of 84 frames costs about $0.017 (docs/licenses.md).
"""

from __future__ import annotations

import base64
import hashlib
import importlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from video2sim.models.hosted import (
    META_URL,
    SAM_MODEL,
    MaskDecoder,
    ResponseCache,
    SamRecord,
    cached_sam_rasters,
    decode_sam_mask,
    file_digest,
    parse_sam_text,
    place_sam_raster,
    post_json,
    read_secret,
    sse_deltas,
)
from video2sim.schemas import WorldModel
from video2sim.stages.segmentation import CropMask, MaskLike, as_crop

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


@dataclass(frozen=True, init=False)
class Track:
    """One tracked object: its mask per world-model frame index where it is visible.

    Masks are kept as ``CropMask`` (N4); full-frame arrays given here are cut to their box.
    """

    track_id: int
    masks: Mapping[int, CropMask]

    def __init__(self, track_id: int, masks: Mapping[int, MaskLike]) -> None:
        object.__setattr__(self, "track_id", track_id)
        object.__setattr__(self, "masks", {f: as_crop(m) for f, m in masks.items()})


class VideoSamClient(Protocol):
    def track(self, video: Path, prompt: str, direction: Direction) -> str:
        """SAM 3.1 special-token text for a clip, one line per frame."""
        ...


@dataclass(frozen=True)
class TrackingEndpoint:
    """Where the SAM 3.1 video request goes: ``segmentation.tracking.endpoint``. ``meta_responses``
    is the Meta Model API's Responses protocol, which a self-hosted server can imitate;
    ``python:package.module:factory`` loads a factory ``(endpoint, timeout_s) -> VideoSamClient``
    for a server with another protocol, as long as the client answers in SAM 3.1 text."""

    protocol: str = "meta_responses"
    url: str = META_URL
    model: str = SAM_MODEL
    api_key_env: str | None = "META_MODEL_API_KEY"

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> TrackingEndpoint:
        e = cfg["segmentation"]["tracking"].get("endpoint") or {}
        protocol = str(e.get("protocol", "meta_responses"))
        if protocol != "meta_responses" and not protocol.startswith("python:"):
            raise ValueError(
                "tracking endpoint protocol must be meta_responses or python:module:factory, "
                f"got {protocol!r}"
            )
        env = e.get("api_key_env", "META_MODEL_API_KEY")
        return cls(
            protocol,
            str(e.get("url", META_URL)),
            str(e.get("model", SAM_MODEL)),
            str(env) if env else None,
        )


def make_tracking_client(cfg: Mapping[str, Any]) -> VideoSamClient:
    """The client of ``segmentation.tracking.endpoint``."""
    endpoint = TrackingEndpoint.from_config(cfg)
    timeout_s = TrackingConfig.from_config(dict(cfg)).timeout_s
    if endpoint.protocol.startswith("python:"):
        module, _, name = endpoint.protocol.removeprefix("python:").partition(":")
        if not (module and name):
            raise ValueError("tracking protocol python: needs python:package.module:factory")
        client: VideoSamClient = getattr(importlib.import_module(module), name)(endpoint, timeout_s)
        return client
    key = read_secret(endpoint.api_key_env) if endpoint.api_key_env else None
    return MetaVideoSamClient(key, timeout_s, endpoint.url, endpoint.model)


class MetaVideoSamClient:
    def __init__(
        self,
        key: str | None,
        timeout_s: float = 600.0,
        url: str = META_URL,
        model: str = SAM_MODEL,
    ) -> None:
        self._key, self._timeout_s, self._url, self._model = key, timeout_s, url, model

    def track(self, video: Path, prompt: str, direction: Direction) -> str:
        # The data URL form is not shown in the docs; it was checked on s01 (2026-10-28).
        url = "data:video/mp4;base64," + base64.b64encode(video.read_bytes()).decode()
        body = {
            "model": self._model,
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
        raw = post_json(self._url, self._key, body, self._timeout_s).decode()
        return "\n".join(sse_deltas(raw))


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
    text: str,
    frame_indices: list[int],
    decode: MaskDecoder = decode_sam_mask,
    raster_cache: Path | None = None,
) -> list[Track]:
    """Tracks of an answer. ``<Nf>`` is the position in the clip, mapped to a world frame index.

    With the default decoder, large answers are decoded in worker processes
    (``decode_sam_rasters``) and, given ``raster_cache``, the decoded rasters are stored there
    by answer, so a rerun reads them back; the tracks are the same either way.
    """
    records = parse_sam_text(text)
    for record in records:
        if not 0 <= record.frame < len(frame_indices):
            raise ValueError(f"answer names frame {record.frame}, clip has {len(frame_indices)}")
    decoded: list[MaskLike]
    if decode is decode_sam_mask:
        rasters = cached_sam_rasters(text, records, raster_cache)
        decoded = [sam_crop(r, raster) for r, raster in zip(records, rasters, strict=True)]
    else:
        decoded = [decode(r) for r in records]
    masks: dict[int, dict[int, MaskLike]] = {}
    for record, mask in zip(records, decoded, strict=True):
        masks.setdefault(record.object_id, {})[frame_indices[record.frame]] = mask
    return [Track(i, masks[i]) for i in sorted(masks)]


def sam_crop(record: SamRecord, raster: NDArray[np.uint8]) -> CropMask:
    """``place_sam_raster`` as a crop: the raster stretched to its box, without the frame.

    A box that leaves the frame goes through ``place_sam_raster`` itself, so it fails or
    clips exactly as the full-frame mask does.
    """
    import cv2  # heavy import, kept out of module import time

    left, top = int(record.box[0]), int(record.box[1])
    right, bottom = int(record.box[2]) + 1, int(record.box[3]) + 1
    width, height = record.frame_size
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        return CropMask.from_full(place_sam_raster(record, raster))
    box = cv2.resize(raster, (right - left, bottom - top), interpolation=cv2.INTER_NEAREST)
    return CropMask(top, left, np.asarray(box).astype(bool), (height, width))


def _cache_parts(paths: list[Path], cfg: TrackingConfig) -> list[str]:
    """Cache key parts of one video request: model, prompt, direction, frame digests."""
    digest = hashlib.sha256(json.dumps([file_digest(p) for p in paths]).encode()).hexdigest()
    return ["sam_video", SAM_MODEL, cfg.prompt, cfg.direction, f"{cfg.fps}", digest]


def tracking_is_cached(
    world: WorldModel, run_dir: Path, cache: ResponseCache, cfg: TrackingConfig
) -> bool:
    """Whether this exact request already has a cached answer."""
    paths = [run_dir / f.image_path for f in world.frames]
    return cache.get(_cache_parts(paths, cfg)) is not None


def track_objects(
    world: WorldModel,
    run_dir: Path,
    client: VideoSamClient,
    cache: ResponseCache,
    cfg: TrackingConfig,
    work_dir: Path,
    decode: MaskDecoder = decode_sam_mask,
    clip_path: Path | None = None,
) -> list[Track]:
    """Track every object over all registered frames with one hosted request (cached).

    A cache hit returns without touching the clip. On a miss the clip is
    encoded once to ``clip_path`` (default ``work_dir / "clip.mp4"``); a
    caller sending several prompts for the same frames passes one shared
    ``clip_path`` and the first miss encodes it, the rest only read it.
    """
    paths = [run_dir / f.image_path for f in world.frames]
    # Key on the frames, not the clip bytes, so a new encoder build does not void the cache.
    parts = _cache_parts(paths, cfg)
    frame_indices = [f.frame_index for f in world.frames]
    rasters = work_dir / "raster_cache"
    cached = cache.get(parts)
    if cached is not None:
        return tracks_from_text(cached, frame_indices, decode, rasters)
    work_dir.mkdir(parents=True, exist_ok=True)
    clip = clip_path if clip_path is not None else work_dir / "clip.mp4"
    if clip_path is None or not clip.is_file():
        encode_clip(paths, cfg.fps, clip)
    call: Callable[[], str] = lambda: client.track(clip, cfg.prompt, cfg.direction)  # noqa: E731
    text = cache.get_or_call(parts, call)
    return tracks_from_text(text, frame_indices, decode, rasters)


def save_tracks(tracks: list[Track], out_dir: Path) -> None:
    """One ``track_NNN.npz`` per track with its frame indices and stacked full-frame masks."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for track in tracks:
        frames = sorted(track.masks)
        np.savez_compressed(
            out_dir / f"track_{track.track_id:03d}.npz",
            frames=np.array(frames),
            masks=np.array([track.masks[f].full() for f in frames], dtype=bool),
        )


def load_tracks(directory: Path) -> list[Track]:
    """Tracks written by ``save_tracks``, in id order."""
    tracks = []
    for path in sorted(directory.glob("track_*.npz")):
        with np.load(path) as data:
            frames, masks = data["frames"], data["masks"]
        tracks.append(
            Track(
                int(path.stem.split("_")[1]),
                {int(f): m for f, m in zip(frames, masks, strict=True)},
            )
        )
    return tracks
