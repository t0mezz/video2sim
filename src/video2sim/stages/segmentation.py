"""Stage 3: segmentation, tracking, lifting to 3D (Phase 2). Shared types and helpers.

Input:  ``WorldModel`` and the frames.
Output: ``Instances``: per object, masks per frame, a point cluster in the world frame, the
        best keyframes for shape prediction, and failure flags.

The stage itself is ``segmentation_stage.SegmentationStage``. Objects are found by the SAM 3.1
video tracker (``tracking``) with a prompt the scene router picks; the per-frame discovery of
P2.1 was removed after it stopped feeding the pipeline (its last commit is 289bfe2).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np
from numpy.typing import NDArray

BoolMask = NDArray[np.bool_]


def sample_frame_indices(n_available: int, n_wanted: int) -> list[int]:
    """Positions of ``n_wanted`` frames spread evenly over ``n_available``, ends included."""
    if n_available < 1:
        return []
    if n_wanted >= n_available:
        return list(range(n_available))
    if n_wanted == 1:
        return [n_available // 2]
    return sorted({round(i * (n_available - 1) / (n_wanted - 1)) for i in range(n_wanted)})


# Downscale factors whose reciprocal is exact in binary, so that ``cv2.resize`` with
# ``fx = 1 / factor`` scales by exactly ``factor`` and works block by block.
_BLOCK_FACTORS = (1, 2, 4, 8, 16)


@dataclass(frozen=True, eq=False)
class CropMask:
    """A frame mask kept as the box that holds its pixels (N4).

    ``data`` sits at row ``top`` and column ``left`` of a frame of ``shape`` (rows, cols); every
    pixel outside it is False. A track holds one per frame, so a 1080p mask costs the size of
    its box (3 to 12% of the frame on the benchmark) instead of 2 MB. Each method gives what
    the same operation on the full frame (``full()``) gives, bit for bit.
    """

    top: int
    left: int
    data: BoolMask
    shape: tuple[int, int]

    def __post_init__(self) -> None:
        if self.data.ndim != 2 or self.data.dtype != np.bool_:
            raise ValueError("a crop mask holds a 2-D boolean array")
        rows, cols = self.data.shape
        height, width = self.shape
        if self.top < 0 or self.left < 0 or self.top + rows > height or self.left + cols > width:
            raise ValueError(
                f"crop of {rows}x{cols} at ({self.top}, {self.left}) leaves the {self.shape} frame"
            )

    @classmethod
    def from_full(cls, mask: NDArray[Any]) -> CropMask:
        """The crop of a full-frame mask, cut to the box of its pixels."""
        full = np.asarray(mask, dtype=bool)
        if full.ndim != 2:
            raise ValueError(f"a mask is 2-D, got shape {full.shape}")
        shape = (int(full.shape[0]), int(full.shape[1]))
        rows = np.flatnonzero(np.any(full, axis=1))
        if rows.size == 0:
            return cls(0, 0, np.zeros((0, 0), dtype=bool), shape)
        top, bottom = int(rows[0]), int(rows[-1]) + 1
        cols = np.flatnonzero(np.any(full[top:bottom], axis=0))
        left, right = int(cols[0]), int(cols[-1]) + 1
        return cls(top, left, full[top:bottom, left:right].copy(), shape)

    @cached_property
    def area(self) -> int:
        """Number of mask pixels."""
        return int(np.count_nonzero(self.data))

    def any(self) -> bool:
        return self.area > 0

    @cached_property
    def bounds(self) -> tuple[int, int, int, int] | None:
        """``(top, bottom, left, right)`` rows and columns of the mask pixels, inclusive, in
        frame coordinates; None for an empty mask."""
        rows = np.flatnonzero(np.any(self.data, axis=1))
        if rows.size == 0:
            return None
        cols = np.flatnonzero(np.any(self.data, axis=0))
        return (
            self.top + int(rows[0]),
            self.top + int(rows[-1]),
            self.left + int(cols[0]),
            self.left + int(cols[-1]),
        )

    def full(self) -> BoolMask:
        """The full-frame mask."""
        out = np.zeros(self.shape, dtype=bool)
        rows, cols = self.data.shape
        out[self.top : self.top + rows, self.left : self.left + cols] = self.data
        return out

    def window(self, rows: slice, cols: slice) -> BoolMask:
        """``full()[rows, cols]`` for slices with step 1, with NumPy's rules for negative and
        out-of-range bounds."""
        r0, r1, rstep = rows.indices(self.shape[0])
        c0, c1, cstep = cols.indices(self.shape[1])
        if rstep != 1 or cstep != 1:
            raise ValueError("a crop window has step 1")
        out = np.zeros((max(0, r1 - r0), max(0, c1 - c0)), dtype=bool)
        h, w = self.data.shape
        a0, a1 = max(r0, self.top), min(r1, self.top + h)
        b0, b1 = max(c0, self.left), min(c1, self.left + w)
        if a0 < a1 and b0 < b1:
            out[a0 - r0 : a1 - r0, b0 - c0 : b1 - c0] = self.data[
                a0 - self.top : a1 - self.top, b0 - self.left : b1 - self.left
            ]
        return out

    def at(self, rows: NDArray[np.integer[Any]], cols: NDArray[np.integer[Any]]) -> BoolMask:
        """``full()[rows, cols]`` for pixel indices inside the frame."""
        r = rows - self.top
        c = cols - self.left
        h, w = self.data.shape
        inside = (r >= 0) & (r < h) & (c >= 0) & (c < w)
        out = np.zeros(np.shape(r), dtype=bool)
        out[inside] = self.data[r[inside], c[inside]]
        return out

    def overlap(self, other: CropMask) -> int:
        """Pixels on both masks; 0 for masks of different frame sizes."""
        if self.shape != other.shape:
            return 0
        h, w = self.data.shape
        oh, ow = other.data.shape
        a0, a1 = max(self.top, other.top), min(self.top + h, other.top + oh)
        b0, b1 = max(self.left, other.left), min(self.left + w, other.left + ow)
        if a0 >= a1 or b0 >= b1:
            return 0
        mine = self.data[a0 - self.top : a1 - self.top, b0 - self.left : b1 - self.left]
        theirs = other.data[a0 - other.top : a1 - other.top, b0 - other.left : b1 - other.left]
        return int(np.count_nonzero(mine & theirs))

    def downscaled(
        self, factor: int, interpolation: int, dsize: tuple[int, int] | None = None
    ) -> CropMask:
        """The mask shrunk the way ``cv2.resize`` shrinks the full frame as uint8, read back as
        nonzero: with ``fx = fy = 1 / factor``, or to ``dsize`` (width, height) when given.

        For ``INTER_AREA`` and ``INTER_NEAREST`` at a power-of-two factor that divides the
        frame, each small pixel depends only on its own block of the frame, so only a
        block-aligned window around the box is resized. Any other case resizes the full frame.
        """
        import cv2  # heavy import, kept out of module import time

        height, width = self.shape
        small_shape = (height // factor, width // factor) if factor >= 1 else (0, 0)
        blockwise = (
            factor in _BLOCK_FACTORS
            and interpolation in (cv2.INTER_AREA, cv2.INTER_NEAREST)
            and height % factor == 0
            and width % factor == 0
            and (dsize is None or dsize == (small_shape[1], small_shape[0]))
        )
        if not blockwise:
            full = self.full().astype(np.uint8)
            if dsize is None:
                out = cv2.resize(full, None, fx=1.0 / factor, fy=1.0 / factor,
                                 interpolation=interpolation)  # fmt: skip
            else:
                out = cv2.resize(full, dsize, interpolation=interpolation)
            return CropMask.from_full(np.asarray(out) != 0)
        if factor == 1:
            return self
        h, w = self.data.shape
        if h == 0 or w == 0:
            return CropMask(0, 0, np.zeros((0, 0), dtype=bool), small_shape)
        t0, l0 = self.top // factor * factor, self.left // factor * factor
        t1 = -(-(self.top + h) // factor) * factor
        l1 = -(-(self.left + w) // factor) * factor
        window = self.window(slice(t0, t1), slice(l0, l1)).astype(np.uint8)
        if dsize is None:
            small = cv2.resize(window, None, fx=1.0 / factor, fy=1.0 / factor,
                               interpolation=interpolation)  # fmt: skip
        else:
            small = cv2.resize(
                window, ((l1 - l0) // factor, (t1 - t0) // factor), interpolation=interpolation
            )
        return CropMask(t0 // factor, l0 // factor, np.asarray(small) != 0, small_shape)


MaskLike = BoolMask | CropMask


def as_crop(mask: MaskLike) -> CropMask:
    """``mask`` as a ``CropMask``; a full-frame array is cut to its box."""
    return mask if isinstance(mask, CropMask) else CropMask.from_full(mask)
