"""Checks on benchmark ground-truth entries (P0.2). Format: ``benchmark/README.md``."""

from __future__ import annotations

import math
from collections.abc import Sequence


def size_convention_errors(size_m: Sequence[float]) -> list[str]:
    """Return the ways ``size_m`` breaks the ``[length, width, height]`` convention.

    ``length >= width`` is the only ordering rule. ``height`` is the vertical extent in the
    resting pose, so it may be larger or smaller than either horizontal side.
    """
    if len(size_m) != 3:
        return [f"size_m needs 3 values, got {len(size_m)}"]
    length, width, height = size_m
    errors = [
        f"{name} must be a positive number, got {v}"
        for name, v in (("length", length), ("width", width), ("height", height))
        if not (math.isfinite(v) and v > 0)
    ]
    if not errors and length < width:
        errors.append(f"length {length} is smaller than width {width}")
    return errors


def yaw_is_defined(size_m: Sequence[float], rel_tol: float = 0.03) -> bool:
    """False when length and width are equal within ``rel_tol``.

    The horizontal axes are then interchangeable, so ``yaw_deg`` is arbitrary and the
    orientation metric should skip the object. The tolerance covers the 1 to 3 mm spread of
    repeated caliper readings on a 50 to 150 mm object.
    """
    length, width = size_m[0], size_m[1]
    return not math.isclose(length, width, rel_tol=rel_tol)
