"""Discovery method C (P2.1): one generic SAM 3.1 prompt per sampled frame.

``objects on a table`` (``segmentation.discovery.generic_prompt``) asks SAM 3.1 for every
object at once. There is no vision-language model and one hosted call per frame, so a scene
costs 6 calls where method A costs dozens. Masks carry no label, because the prompt does not
name the objects, and a frame can miss objects that other frames find (see PLAN.md P2.1).

Responses are cached under the same key as method A's SAM calls (model, prompt text, image
hash), so a prompt that was already sent for a frame is not sent again.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

from video2sim.stages.discovery_vlm_sam import (
    SAM_MODEL,
    MaskDecoder,
    ResponseCache,
    SamClient,
    decode_sam_mask,
    file_digest,
    parse_sam_text,
)
from video2sim.stages.segmentation import DiscoveredMask, FrameDiscovery


class GenericSamBackend:
    """The same prompt on every sampled frame; every match becomes an unlabeled mask."""

    def __init__(
        self,
        sam: SamClient,
        cache: ResponseCache,
        prompt: str,
        decode: MaskDecoder = decode_sam_mask,
        max_workers: int = 1,
    ) -> None:
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        self._sam, self._cache, self._prompt = sam, cache, prompt
        self._decode, self._max_workers = decode, max_workers

    def discover(self, frames: Sequence[tuple[int, Path]]) -> list[FrameDiscovery]:
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            texts = list(pool.map(self._segment, [path for _, path in frames]))
        return [
            FrameDiscovery(
                index, tuple(DiscoveredMask(None, self._decode(r)) for r in parse_sam_text(text))
            )
            for (index, _), text in zip(frames, texts, strict=True)
        ]

    def _segment(self, path: Path) -> str:
        return self._cache.get_or_call(
            ["sam", SAM_MODEL, self._prompt, file_digest(path)],
            partial(self._sam.segment, path, self._prompt),
        )
