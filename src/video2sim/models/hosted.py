"""Hosted model APIs shared by the segmentation stage: Baseten (Muse Spark) and the Meta Model API
(SAM 3.1).

Standard library only for HTTP (``urllib``), with retries on rate limits and server errors.
Every hosted response is cached on disk under the key of what was sent, so reruns cost nothing.
Keys come from the environment or ``.env``, never from code or config. SAM 3.1 answers are
special-token text; masks in them are decoded with ``meta_sam_parser`` (SAM License, see
docs/licenses.md).
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import io
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

BASETEN_URL = "https://inference.baseten.co/v1/chat/completions"
BASETEN_MODEL = "meta-models/muse-spark-open-1026"
META_URL = "https://api.meta.ai/v1/responses"
SAM_MODEL = "sam-3.1"


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

    def get(self, parts: Sequence[str]) -> str | None:
        """The cached answer for ``parts``, or None. Lets a caller skip work
        (such as encoding a clip) that only a cache miss needs."""
        key = hashlib.sha256("\x1f".join(parts).encode()).hexdigest()
        path = self._dir / f"{key}.txt"
        return path.read_text(encoding="utf-8") if path.is_file() else None

    def get_or_call(self, parts: Sequence[str], call: Callable[[], str]) -> str:
        cached = self.get(parts)
        if cached is not None:
            return cached
        key = hashlib.sha256("\x1f".join(parts).encode()).hexdigest()
        path = self._dir / f"{key}.txt"
        text = call()
        self._dir.mkdir(parents=True, exist_ok=True)
        # Atomic publish: parallel callers on the same key (parallel router
        # views, cluster checks) must never read a half-written answer.
        tmp = self._dir / f"{key}.{threading.get_ident()}.tmp"
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
        return text


def image_data_url(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


_RETRIES = 4
GATEWAY_HEADER = "X-V2S-Gateway"


def explain_http_error(error: urllib.error.HTTPError) -> urllib.error.HTTPError:
    """``error``, with the judge gateway's reason as its text when the gateway refused the call
    (``workers/gateway``: quota used up, token revoked, gateway switched off). Other errors are
    returned as they are, so a provider's error body is never put into a message."""
    if error.headers is None or error.headers.get(GATEWAY_HEADER) != "1":
        return error
    try:
        body = error.read(4096)
        message = str(json.loads(body)["message"])
    except Exception:  # an unreadable refusal keeps the plain status text
        return error
    return urllib.error.HTTPError(error.url, error.code, message, error.headers, io.BytesIO(body))


def post_json(
    url: str,
    key: str | None,
    body: dict[str, Any],
    timeout_s: float = 120.0,
    retries: int = _RETRIES,
    extra_headers: Mapping[str, str] | None = None,
) -> bytes:
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    headers.update(extra_headers or {})
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                data: bytes = response.read()
            return data
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == retries - 1:
                raise explain_http_error(e) from e
        time.sleep(2.0 * 2**attempt)
    raise AssertionError("unreachable")


def sse_deltas(raw: str) -> list[str]:
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


MaskDecoder = Callable[[SamRecord], NDArray[np.bool_]]

# The raster decoder of meta_sam_parser is pure Python: 72 ms a mask on the s05-sized answer of
# run 20261103T200206Z and 329 ms on the 241-frame one of 20261103T090231Z (Apple M4), most of
# a segmentation replay. Answers with at least this many masks are decoded in worker processes,
# which decoded those answers 4.2x and 4.6x faster with 8 workers and identical masks. Fewer
# masks do not repay starting the workers. Neither number changes a result.
_PARALLEL_DECODE_MIN_MASKS = 16
_PARALLEL_DECODE_MAX_WORKERS = 8


def decode_sam_raster(record: SamRecord) -> NDArray[np.uint8]:
    """The record's wire raster, 0 or 1 per pixel, (rows, cols), before it is fitted to the box."""
    from meta_sam_parser import SegmentationMask, decode_mask_to_raster

    rows, cols = record.mask_shape
    payload = record.mask_payload
    mask = SegmentationMask(
        "lossless" if payload.startswith("~") else "one_bit", payload, cols, rows
    )
    raster = decode_mask_to_raster(mask)
    return np.frombuffer(raster, dtype=np.uint8).reshape(rows, cols)


def place_sam_raster(record: SamRecord, local: NDArray[np.uint8]) -> NDArray[np.bool_]:
    """Full-frame boolean mask of a record from its decoded wire raster.

    The wire raster is box-local and smaller than the box: it covers the half-open bounds
    ``[x1, x2 + 1) x [y1, y2 + 1)`` (the box corners are inclusive). It is stretched to the box
    with nearest-neighbour sampling.
    """
    import cv2  # heavy import, kept out of module import time

    left, top = int(record.box[0]), int(record.box[1])
    right, bottom = int(record.box[2]) + 1, int(record.box[3]) + 1
    width, height = record.frame_size
    box_mask = cv2.resize(local, (right - left, bottom - top), interpolation=cv2.INTER_NEAREST)
    full = np.zeros((height, width), dtype=bool)
    full[top:bottom, left:right] = box_mask.astype(bool)
    return full


def decode_sam_mask(record: SamRecord) -> NDArray[np.bool_]:
    """Full-frame boolean mask of a record, decoded by ``meta_sam_parser``."""
    return place_sam_raster(record, decode_sam_raster(record))


def _raster_or_none(record: SamRecord) -> NDArray[np.uint8] | None:
    """Worker side of ``decode_sam_masks``. A failure comes back as None, because the parser's
    exceptions take keyword arguments and do not survive the trip between processes."""
    try:
        return decode_sam_raster(record)
    except Exception:
        return None


def decode_sam_rasters(
    records: Sequence[SamRecord], max_workers: int | None = None
) -> list[NDArray[np.uint8]]:
    """``decode_sam_raster`` of every record, in order, decoded in parallel.

    The result is the same as decoding one by one. Small answers and ``max_workers`` 1 are
    decoded here; a record that fails in a worker is decoded again here, so it raises the
    parser's own error.
    """
    workers = (
        min(_PARALLEL_DECODE_MAX_WORKERS, os.cpu_count() or 1)
        if max_workers is None
        else max_workers
    )
    if workers <= 1 or len(records) < _PARALLEL_DECODE_MIN_MASKS:
        return [decode_sam_raster(r) for r in records]
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    # spawn on every OS: the stage may call this from worker threads (gap fill), and forking a
    # process that runs threads can deadlock.
    context = multiprocessing.get_context("spawn")
    chunk = max(1, len(records) // (8 * workers))
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        rasters = list(pool.map(_raster_or_none, records, chunksize=chunk))
    return [
        decode_sam_raster(r) if raster is None else raster
        for r, raster in zip(records, rasters, strict=True)
    ]


# Bump when the stored layout changes; the parser version is part of the key as well.
_RASTER_CACHE_FORMAT = 1


def _raster_cache_key(text: str) -> str:
    from importlib.metadata import version

    parser = version("meta_sam_parser")
    return hashlib.sha256(f"{_RASTER_CACHE_FORMAT}\x1f{parser}\x1f{text}".encode()).hexdigest()


def _read_rasters(path: Path, records: Sequence[SamRecord]) -> list[NDArray[np.uint8]] | None:
    """Rasters stored for ``records``, or None when the file is missing or does not fit them."""
    try:
        with np.load(path) as data:
            bits, starts, shapes = data["bits"], data["starts"], data["shapes"]
    except (OSError, KeyError, ValueError):
        return None
    if len(shapes) != len(records) or len(starts) != len(records) + 1:
        return None
    out = []
    for i, record in enumerate(records):
        rows, cols = (int(v) for v in shapes[i])
        if (rows, cols) != record.mask_shape:
            return None
        packed = bits[int(starts[i]) : int(starts[i + 1])]
        flat = np.unpackbits(packed, count=rows * cols)
        out.append(np.ascontiguousarray(flat.reshape(rows, cols), dtype=np.uint8))
    return out


def _write_rasters(path: Path, rasters: Sequence[NDArray[np.uint8]]) -> None:
    """Store ``rasters`` bit-packed; atomic, so a parallel reader never sees half a file."""
    packed = [np.packbits(r.ravel() != 0) for r in rasters]
    starts = np.zeros(len(packed) + 1, dtype=np.int64)
    starts[1:] = np.cumsum([len(p) for p in packed])
    shapes = np.array([r.shape for r in rasters], dtype=np.int64).reshape(-1, 2)
    bits = np.concatenate(packed) if packed else np.zeros(0, dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f"{path.stem}.{os.getpid()}.{threading.get_ident()}.tmp"
    with open(tmp, "wb") as f:
        np.savez(f, bits=bits, starts=starts, shapes=shapes)
    tmp.replace(path)


def cached_sam_rasters(
    text: str, records: Sequence[SamRecord], cache_dir: Path | None
) -> list[NDArray[np.uint8]]:
    """``decode_sam_rasters`` of an answer's records, stored under ``cache_dir`` by the answer
    text (N7). Decoding is pure Python and was most of a segmentation replay (212 s on one core
    for the 241-frame answer of run 20261103T090231Z); a stored raster is the same bits."""
    if cache_dir is None:
        return decode_sam_rasters(records)
    path = cache_dir / f"{_raster_cache_key(text)}.npz"
    stored = _read_rasters(path, records)
    if stored is not None:
        return stored
    rasters = decode_sam_rasters(records)
    with contextlib.suppress(OSError):  # an unwritten cache only costs the next run the decode
        _write_rasters(path, rasters)
    return rasters


def decode_sam_masks(
    records: Sequence[SamRecord], max_workers: int | None = None
) -> list[NDArray[np.bool_]]:
    """``decode_sam_mask`` of every record, in order, with the rasters decoded in parallel."""
    rasters = decode_sam_rasters(records, max_workers)
    return [place_sam_raster(r, raster) for r, raster in zip(records, rasters, strict=True)]


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
