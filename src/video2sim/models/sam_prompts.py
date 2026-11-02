"""Box and point prompts for SAM 3.1 video (`docs/research/occlusion_robust_discovery_plan.md`).

The object-prompt grammar comes from the language-neutral protocol
(`protocol/sam3.md` in `meta-models/meta-sam`, PR 17): one block per source frame opened by
`<Nf>` (for example `<30f>`), comma-separated segments of a bare object id followed by
concatenated tokens. A box token carries inclusive corners
(`<|box;x1=..;y1=..;x2=..;y2=..;w=..;h=..|>`, at most one per object per frame); a point
token is positive by default and a `-` directly before it makes it negative
(`1<|box;...|>-<|point;...|>`). `w` and `h` are the media's displayed size in pixels, and
coordinates are rounded down to whole pixels and clamped to the media.

Whether the hosted `sam-3.1` model accepts these prompts is decided by the live probe (task
9.0), not by this module. One reading taken here still needs that probe's confirmation: the
block frame is the 0-based position in the clip (as `tracks_from_text` reads the answers).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ObjectPrompt:
    """Where to look for one object on one frame, in media pixels (inclusive box)."""

    object_id: int
    box: tuple[float, float, float, float]  # x1, y1, x2, y2, x2 and y2 inclusive
    positive_points: tuple[tuple[float, float], ...] = ()
    negative_points: tuple[tuple[float, float], ...] = ()


def _pixel(value: float, limit: int) -> int:
    """Floor to a whole pixel and clamp into the media (inclusive bound `limit - 1`)."""
    return min(max(math.floor(value), 0), limit - 1)


def build_object_prompt(frames: Mapping[int, Sequence[ObjectPrompt]], size: tuple[int, int]) -> str:
    """Render object prompts as one SAM 3.1 prompt string.

    `frames` maps a source frame number to that frame's prompts, `size` is the `(width,
    height)` media size in pixels. Frame blocks come in ascending order, segments in ascending
    object id; every token carries `w` and `h`. Raises `ValueError` on an empty mapping, an
    empty frame, a duplicated object id within one frame, or a box with `x2 < x1` or `y2
    < y1`.
    """
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError(f"media size must be positive, got {size!r}")
    if not frames:
        raise ValueError("no frames to prompt")
    blocks = []
    for frame in sorted(frames):
        prompts = list(frames[frame])
        if not prompts:
            raise ValueError(f"frame {frame} has no prompts")
        seen = set()
        for prompt in prompts:
            if prompt.object_id in seen:
                raise ValueError(f"frame {frame} prompts object {prompt.object_id} twice")
            seen.add(prompt.object_id)
        segments = []
        for prompt in sorted(prompts, key=lambda p: p.object_id):
            x1, y1, x2, y2 = prompt.box
            if x2 < x1 or y2 < y1:
                raise ValueError(f"object {prompt.object_id} has an inverted box {prompt.box!r}")
            tokens = [
                f"<|box;x1={_pixel(x1, width)};y1={_pixel(y1, height)};"
                f"x2={_pixel(x2, width)};y2={_pixel(y2, height)};w={width};h={height}|>"
            ]
            for x, y in prompt.positive_points:
                tokens.append(
                    f"<|point;x={_pixel(x, width)};y={_pixel(y, height)};w={width};h={height}|>"
                )
            for x, y in prompt.negative_points:
                tokens.append(
                    f"-<|point;x={_pixel(x, width)};y={_pixel(y, height)};w={width};h={height}|>"
                )
            segments.append(f"{prompt.object_id}{''.join(tokens)}")
        blocks.append(f"<{frame}f>{','.join(segments)}")
    return "".join(blocks)


__all__ = ["ObjectPrompt", "build_object_prompt"]
