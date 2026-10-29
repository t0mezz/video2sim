"""One interface for the generative shape models of stage 4 (P3.1): SAM 3D Objects now,
TRELLIS.2 later.

A model takes one keyframe image, the object's mask in it and a seed, and returns one sample:
a mesh in the model's own frame and units, plus the model's guess of where that frame sits in
the keyframe camera. The guess is only a starting pose for registration (P3.2). Its scale is a
guess too, and the mesh is never scaled by it as is (AGENTS.md, "Scale").

Both models are generative, so the stage draws several samples per keyframe and P3.4 keeps the
one that fits the point cloud best. Each sample has its own seed, derived from the run seed, the
object and the keyframe, so a rerun draws the same samples and adding an object does not change
the seeds of the others.

These types stay inside the shape stage; what leaves it is ``RegisteredObjects``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from video2sim.schemas import SE3

F64 = NDArray[np.float64]


@dataclass(frozen=True)
class ShapeView:
    """One keyframe of an object, as the shape model sees it."""

    frame_index: int
    image_path: Path
    mask: NDArray[np.bool_]  # [row, column], same size as the image
    # Optional dense geometry for this frame: HxWx3 points in the OpenCV camera frame, meters,
    # NaN where unknown. Used only if the model's config asks for it.
    pointmap: F64 | None = None


@dataclass(frozen=True)
class ShapeSample:
    """One draw of a shape model for one object in one keyframe."""

    object_id: str
    frame_index: int
    seed: int
    vertices_model: F64  # (N, 3) in the model frame and its units, not meters
    faces: NDArray[np.int64]  # (M, 3)
    # Rigid pose of the model frame in the keyframe's OpenCV camera frame. A point goes to the
    # camera as ``R @ (scale_guess * x) + t``; scale is kept apart from the transform.
    T_camera_from_model: SE3
    scale_guess: float
    model: str
    checkpoint: str  # as reported by the host, recorded in the run manifest
    vertex_colors: NDArray[np.uint8] | None = None  # (N, 3) RGB, when the model gives them


class ShapeModel(Protocol):
    name: str

    def sample(self, object_id: str, view: ShapeView, seed: int) -> ShapeSample: ...


def sample_seed(base_seed: int, object_id: str, frame_index: int, draw: int) -> int:
    """Seed of one draw, stable under reordering or adding objects."""
    key = f"{base_seed}:{object_id}:{frame_index}:{draw}".encode()
    return int(hashlib.sha256(key).hexdigest()[:8], 16)


def draw_samples(
    model: ShapeModel,
    object_id: str,
    views: Sequence[ShapeView],
    samples_per_keyframe: int,
    base_seed: int,
) -> list[ShapeSample]:
    """``samples_per_keyframe`` samples for each view, in view order (best keyframe first)."""
    if samples_per_keyframe < 1:
        raise ValueError("samples_per_keyframe must be at least 1")
    return [
        model.sample(object_id, view, sample_seed(base_seed, object_id, view.frame_index, draw))
        for view in views
        for draw in range(samples_per_keyframe)
    ]
