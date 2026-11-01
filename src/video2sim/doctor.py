"""``video2sim doctor``: check that the configured model endpoints are usable before a run.

Each model role is built from the config, which resolves its key and imports any
``python:`` factory. Unless ``offline`` is set, the VLM roles then get a one-pixel image and the
HTTP shape server a ``GET /health``. The tracking endpoint is only built, never called, since a
request is billed per frame. A failure names the role and the reason; nothing is retried.
"""

from __future__ import annotations

import struct
import tempfile
import zlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from video2sim.models.vlm import ROLES, VlmConfig, make_vlm_client


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


def _one_pixel_png(path: Path) -> None:
    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    header = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    raw = zlib.compress(b"\x00\xff\xff\xff")
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", raw) + chunk(b"IEND", b"")
    )


def _run(name: str, probe: Callable[[], str]) -> Check:
    try:
        return Check(name, True, probe())
    except Exception as error:  # a doctor reports every failure kind, so it catches broadly
        return Check(name, False, f"{type(error).__name__}: {error}")


def check_vlm(cfg: Mapping[str, Any], offline: bool) -> list[Check]:
    def probe(role: str) -> str:
        settings = VlmConfig.from_config(cfg, role)
        client = make_vlm_client(cfg, 60.0, 1, role)
        if offline:
            return f"{settings.protocol} {settings.model} at {settings.base_url}"
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "pixel.png"
            _one_pixel_png(image)
            answer = client.ask("Reply with the word OK.", [image])
        return f"{settings.model} answered {answer.strip()[:40]!r}"

    return [_run(f"vlm.{role}", lambda role=role: probe(role)) for role in ROLES]  # type: ignore[misc]


def check_shape(cfg: dict[str, Any], offline: bool) -> list[Check]:
    from video2sim.models.sam3d import ModalTransport, Sam3dConfig, make_transport

    def probe() -> str:
        backend = str(cfg["shape"]["backend"])
        if backend != "sam3d":
            from video2sim.stages.shape import default_model

            with tempfile.TemporaryDirectory() as tmp:
                default_model(cfg, Path(tmp))
            return f"{backend} built"
        Sam3dConfig.from_config(cfg)
        transport = make_transport(cfg)
        name = str(cfg["shape"].get("transport", "modal"))
        if offline:
            return f"sam3d over {name}"
        health = getattr(transport, "health", None)
        if callable(health):
            health()
            return f"sam3d over {name}, /health answered"
        if isinstance(transport, ModalTransport):
            transport._lookup()  # raises when the deployment is missing
            return "sam3d over modal, deployment found"
        return f"sam3d over {name}"

    return [_run("shape", probe)]


def check_tracking(cfg: dict[str, Any]) -> list[Check]:
    from video2sim.stages.tracking import TrackingEndpoint, make_tracking_client

    def probe() -> str:
        make_tracking_client(cfg)
        endpoint = TrackingEndpoint.from_config(cfg)
        return f"{endpoint.protocol} {endpoint.model} at {endpoint.url} (built, not called)"

    return [_run("tracking", probe)]


def run_doctor(cfg: dict[str, Any], offline: bool = False) -> list[Check]:
    return [*check_vlm(cfg, offline), *check_shape(cfg, offline), *check_tracking(cfg)]
