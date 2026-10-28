"""Stage 3: segmentation, tracking, lifting to 3D. Implemented in Phase 2 (P2.1 to P2.6).

Input:  ``WorldModel`` and the frames.
Output: ``Instances``: per object, masks per frame, a point cluster in the world frame, the
        best keyframes for shape prediction, and failure flags.

SAM 3.1 provides masks and tracks. Identity errors are repaired with 3D consistency across
views. Objects that look transparent, reflective, thin or tiny are flagged here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from video2sim.schemas import WorldModel

BoolMask = NDArray[np.bool_]
DiscoveryMethod = Literal["vlm_sam", "sam_generic", "auto_proposals"]


@dataclass(frozen=True)
class DiscoveryConfig:
    """Values from ``segmentation.discovery`` in ``configs/default.yaml`` (P2.1)."""

    method: DiscoveryMethod
    n_sample_frames: int
    max_workers: int
    generic_prompt: str

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> DiscoveryConfig:
        d = cfg["segmentation"]["discovery"]
        method = str(d["method"])
        n = int(d["n_sample_frames"])
        workers = int(d["max_workers"])
        prompt = str(d["generic_prompt"]).strip()
        if n < 1:
            raise ValueError("n_sample_frames must be at least 1")
        if workers < 1:
            raise ValueError("max_workers must be at least 1")
        if not prompt:
            raise ValueError("generic_prompt must not be empty")
        if method == "vlm_sam":
            return cls("vlm_sam", n, workers, prompt)
        if method == "sam_generic":
            return cls("sam_generic", n, workers, prompt)
        if method == "auto_proposals":
            return cls("auto_proposals", n, workers, prompt)
        raise ValueError(f"unknown discovery method {method!r}")


@dataclass(frozen=True)
class DiscoveredMask:
    """One candidate object in one frame. ``label`` is None for unlabeled proposals."""

    label: str | None
    mask: BoolMask


@dataclass(frozen=True)
class FrameDiscovery:
    frame_index: int
    masks: tuple[DiscoveredMask, ...]


class DiscoveryBackend(Protocol):
    """A discovery method: images in, candidate masks out.

    ``frames`` pairs each frame index with its image path (absolute). Implementations return
    one ``FrameDiscovery`` per input frame, in the same order. Hosted calls and model imports
    belong inside the implementation, never at module import.
    """

    def discover(self, frames: Sequence[tuple[int, Path]]) -> list[FrameDiscovery]: ...


class FakeDiscoveryBackend:
    """Replays recorded discoveries by frame index, for tests that must not touch a network."""

    def __init__(self, recorded: Mapping[int, FrameDiscovery]) -> None:
        self._recorded = dict(recorded)

    def discover(self, frames: Sequence[tuple[int, Path]]) -> list[FrameDiscovery]:
        return [self._recorded.get(i, FrameDiscovery(i, ())) for i, _ in frames]


def sample_frame_indices(n_available: int, n_wanted: int) -> list[int]:
    """Positions of ``n_wanted`` frames spread evenly over ``n_available``, ends included."""
    if n_available < 1:
        return []
    if n_wanted >= n_available:
        return list(range(n_available))
    if n_wanted == 1:
        return [n_available // 2]
    return sorted({round(i * (n_available - 1) / (n_wanted - 1)) for i in range(n_wanted)})


def discover_objects(
    world: WorldModel,
    run_dir: Path,
    backend: DiscoveryBackend,
    cfg: DiscoveryConfig,
) -> list[FrameDiscovery]:
    """Run discovery on evenly spaced registered frames of the world model."""
    picks = sample_frame_indices(len(world.frames), cfg.n_sample_frames)
    frames = [(world.frames[i].frame_index, run_dir / world.frames[i].image_path) for i in picks]
    results = backend.discover(frames)
    if [r.frame_index for r in results] != [i for i, _ in frames]:
        raise ValueError("backend must return one result per frame, in input order")
    return results
