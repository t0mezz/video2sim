"""Discovery method A (P2.1): a vision-language model names the objects, SAM 3.1 segments each.

Muse Spark Open on Baseten lists object nouns per sampled frame (OpenAI-compatible chat
completions). SAM 3.1 on the Meta Model API is then prompted once per noun and frame. Every
hosted response is cached on disk under the key of what was sent, so reruns cost nothing.

Standard library only: HTTP goes through ``urllib``. Both clients sit behind small protocols so
tests replace them. Keys come from the environment or ``.env``, never from code or config.

Not yet confirmed against the live APIs (see PLAN.md P2.1): how a local frame is passed to
SAM 3.1. The docs only show a public URL; a data URL is tried here. Masks are decoded with
``meta_sam_parser`` (SAM License, see docs/licenses.md).
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Protocol

from video2sim.stages.segmentation import BoolMask, DiscoveredMask, FrameDiscovery

BASETEN_URL = "https://inference.baseten.co/v1/chat/completions"
BASETEN_MODEL = "meta-models/muse-spark-open-1026"
META_URL = "https://api.meta.ai/v1/responses"
SAM_MODEL = "sam-3.1"

VLM_PROMPT = (
    "List every distinct physical object lying on the table in this photo, as short singular "
    "nouns a segmentation model can find (for example 'mug', 'red cube'). Leave out the table, "
    "the printed board or sheet, printed markers, shadows, hands and arms. Answer with a JSON "
    "array of strings and nothing else."
)


def read_secret(name: str, env_file: Path = Path(".env")) -> str:
    """Value of ``name`` from the environment, else from ``env_file``. Never logged."""
    value = os.environ.get(name)
    if value:
        return value
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            key, sep, rest = line.partition("=")
            if sep and key.strip() == name and rest.strip():
                return re.split(r"\s+#", rest.strip(), maxsplit=1)[0].strip().strip("'\"")
    raise RuntimeError(f"{name} is not set in the environment or {env_file}")


class ResponseCache:
    """Text responses on disk, keyed by a hash of everything that was sent."""

    def __init__(self, directory: Path) -> None:
        self._dir = directory

    def get_or_call(self, parts: Sequence[str], call: Callable[[], str]) -> str:
        key = hashlib.sha256("\x1f".join(parts).encode()).hexdigest()
        path = self._dir / f"{key}.txt"
        if path.is_file():
            return path.read_text(encoding="utf-8")
        text = call()
        self._dir.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return text


def _image_data_url(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


_RETRIES = 4


def _post_json(url: str, key: str, body: dict[str, Any], timeout_s: float = 120.0) -> bytes:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    for attempt in range(_RETRIES):
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                data: bytes = response.read()
            return data
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == _RETRIES - 1:
                raise
        time.sleep(2.0 * 2**attempt)
    raise AssertionError("unreachable")


class ObjectLister(Protocol):
    def list_objects(self, image: Path) -> str:
        """Raw model answer for one frame: a JSON array of nouns."""
        ...


class SamClient(Protocol):
    def segment(self, image: Path, noun: str) -> str:
        """SAM 3.1 special-token text for one frame and one noun."""
        ...


class BasetenLister:
    def __init__(self, key: str) -> None:
        self._key = key

    def list_objects(self, image: Path) -> str:
        body = {
            "model": BASETEN_MODEL,
            "temperature": 0,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": VLM_PROMPT},
                        {"type": "image_url", "image_url": {"url": _image_data_url(image)}},
                    ],
                }
            ],
        }
        reply = json.loads(_post_json(BASETEN_URL, self._key, body))
        return str(reply["choices"][0]["message"]["content"])


class MetaSamClient:
    def __init__(self, key: str) -> None:
        self._key = key

    def segment(self, image: Path, noun: str) -> str:
        body = {
            "model": SAM_MODEL,
            "stream": True,
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": noun},
                        {"type": "input_image", "image_url": _image_data_url(image)},
                    ],
                }
            ],
        }
        raw = _post_json(META_URL, self._key, body).decode()
        return "\n".join(_sse_deltas(raw))


def _sse_deltas(raw: str) -> list[str]:
    """Delta strings of ``response.output_text.delta`` events in a server-sent-event stream."""
    deltas: list[str] = []
    for line in raw.splitlines():
        if not line.startswith("data:"):
            continue
        try:
            event = json.loads(line[5:].strip())
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and event.get("type") == "response.output_text.delta":
            deltas.append(str(event.get("delta", "")))
    return deltas


def parse_nouns(answer: str, limit: int) -> list[str]:
    """Nouns from a model answer, lower-cased and deduplicated, in order; junk gives []."""
    match = re.search(r"\[.*\]", answer, flags=re.DOTALL)
    if match is None:
        return []
    try:
        items = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    seen: dict[str, None] = {}
    for item in items if isinstance(items, list) else []:
        if isinstance(item, str) and item.strip():
            seen.setdefault(item.strip().lower(), None)
    return list(seen)[:limit]


@dataclass(frozen=True)
class SamRecord:
    """One match from a SAM 3.1 line: box in source pixels plus the still-encoded mask."""

    frame: int
    object_id: int
    box: tuple[float, float, float, float]
    frame_size: tuple[int, int]  # (width, height) of the source frame
    mask_offset: tuple[int, int]  # (x, y)
    mask_shape: tuple[int, int]  # (rows, cols)
    mask_payload: str  # encoding marker followed by the payload


_LINE = re.compile(r"^<(\d+)f>(.*)$")
_RECORD = re.compile(r"(\d+)<\|box;([^|]*)\|><\|mask;([^|]*)\|>")


def _fields(text: str) -> dict[str, str]:
    return dict(part.split("=", 1) for part in text.split(";") if "=" in part)


def parse_sam_text(text: str) -> list[SamRecord]:
    """Records of a SAM 3.1 answer, following the grammar on dev.meta.ai/docs/sam."""
    records: list[SamRecord] = []
    for line in text.splitlines():
        head = _LINE.match(line.strip())
        if head is None:
            continue
        for obj, box_text, mask_text in _RECORD.findall(head.group(2)):
            box, mask = _fields(box_text), _fields(mask_text)
            rows, cols, payload = mask["data"].split(",", 2)
            records.append(
                SamRecord(
                    frame=int(head.group(1)),
                    object_id=int(obj),
                    box=(float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"])),
                    frame_size=(int(box["w"]), int(box["h"])),
                    mask_offset=(int(mask.get("x", 0)), int(mask.get("y", 0))),
                    mask_shape=(int(rows), int(cols)),
                    mask_payload=payload,
                )
            )
    return records


MaskDecoder = Callable[[SamRecord], BoolMask]


def decode_sam_mask(record: SamRecord) -> BoolMask:
    """Full-frame boolean mask of a record.

    The wire raster is box-local and smaller than the box: it covers the half-open bounds
    ``[x1, x2 + 1) x [y1, y2 + 1)`` (the box corners are inclusive). It is decoded by
    ``meta_sam_parser`` and stretched to the box with nearest-neighbour sampling.
    """
    import cv2  # heavy import, kept out of module import time
    import numpy as np
    from meta_sam_parser import SegmentationMask, decode_mask_to_raster

    rows, cols = record.mask_shape
    payload = record.mask_payload
    mask = SegmentationMask(
        "lossless" if payload.startswith("~") else "one_bit", payload, cols, rows
    )
    raster = decode_mask_to_raster(mask)
    local = np.frombuffer(raster, dtype=np.uint8).reshape(rows, cols)
    left, top = int(record.box[0]), int(record.box[1])
    right, bottom = int(record.box[2]) + 1, int(record.box[3]) + 1
    width, height = record.frame_size
    box_mask = cv2.resize(local, (right - left, bottom - top), interpolation=cv2.INTER_NEAREST)
    full = np.zeros((height, width), dtype=bool)
    full[top:bottom, left:right] = box_mask.astype(bool)
    return full


class VlmSamBackend:
    """Nouns from the VLM over all sampled frames, then SAM 3.1 per noun and frame."""

    def __init__(
        self,
        lister: ObjectLister,
        sam: SamClient,
        cache: ResponseCache,
        decode: MaskDecoder = decode_sam_mask,
        max_nouns: int = 12,
        max_workers: int = 1,
    ) -> None:
        self._lister, self._sam, self._cache = lister, sam, cache
        self._decode, self._max_nouns, self._max_workers = decode, max_nouns, max_workers

    def discover(self, frames: Sequence[tuple[int, Path]]) -> list[FrameDiscovery]:
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            answers = list(pool.map(self._ask_nouns, [path for _, path in frames]))
            nouns: dict[str, None] = {}
            for answer in answers:
                for noun in parse_nouns(answer, self._max_nouns):
                    nouns.setdefault(noun, None)
            wanted = list(nouns)[: self._max_nouns]
            jobs = [(path, noun) for _, path in frames for noun in wanted]
            texts = iter(pool.map(lambda job: self._segment(*job), jobs))
        results: list[FrameDiscovery] = []
        for index, _ in frames:
            masks: list[DiscoveredMask] = []
            for noun in wanted:
                masks.extend(
                    DiscoveredMask(noun, self._decode(r)) for r in parse_sam_text(next(texts))
                )
            results.append(FrameDiscovery(index, tuple(masks)))
        return results

    def _ask_nouns(self, path: Path) -> str:
        return self._cache.get_or_call(
            ["vlm", BASETEN_MODEL, VLM_PROMPT, file_digest(path)],
            partial(self._lister.list_objects, path),
        )

    def _segment(self, path: Path, noun: str) -> str:
        return self._cache.get_or_call(
            ["sam", SAM_MODEL, noun, file_digest(path)], partial(self._sam.segment, path, noun)
        )


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
