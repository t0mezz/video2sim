"""The three upstreams. Their URLs are fixed here, so a request cannot name a destination.

An upstream failure raises ``UpstreamFailed`` with the status only: the gateway never passes an
upstream's error body or headers on, since either could echo a key or an account detail.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from v2s_gateway.checks import Sam3dRequest

BASETEN_CHAT_URL = "https://inference.baseten.co/v1/chat/completions"
META_RESPONSES_URL = "https://api.meta.ai/v1/responses"
# Statuses after which the same request may succeed later; the call is then not charged.
RETRYABLE = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


class UpstreamFailed(Exception):
    """The upstream did not serve the request. ``status`` is None for a transport error."""

    def __init__(self, status: int | None, retryable: bool) -> None:
        super().__init__(f"upstream status {status}")
        self.status, self.retryable = status, retryable


class ChatUpstream(Protocol):
    async def chat(self, body: dict[str, Any]) -> bytes:
        """The chat-completions answer, as the upstream sent it."""
        ...


class TrackingUpstream(Protocol):
    async def responses(self, body: dict[str, Any]) -> AsyncIterator[bytes]:
        """The server-sent-event stream, once the upstream answered 200."""
        ...


@dataclass(frozen=True)
class GpuTiming:
    """When one draw ran, on the worker's clock: ``start_s`` is when its container started
    working for it (boot and weight loading included when the draw started the container)."""

    container: str
    start_s: float
    end_s: float


@dataclass(frozen=True)
class Sam3dResult:
    glb: bytes
    pose_json: str
    timing: GpuTiming | None  # None when the worker reported no timings


class Sam3dUpstream(Protocol):
    async def sample(self, request: Sam3dRequest) -> Sam3dResult:
        """The GLB, the pose JSON and the GPU timing of one draw."""
        ...


@dataclass(frozen=True)
class Upstreams:
    chat: ChatUpstream
    tracking: TrackingUpstream
    sam3d: Sam3dUpstream


def _failed(status: int) -> UpstreamFailed:
    return UpstreamFailed(status, status in RETRYABLE)


class HttpUpstreams:
    """Baseten chat completions and the Meta Responses API over one ``httpx.AsyncClient``."""

    def __init__(self, client: httpx.AsyncClient, baseten_key: str, meta_key: str) -> None:
        self._client = client
        self._baseten = {"Authorization": f"Bearer {baseten_key}"}
        self._meta = {"Authorization": f"Bearer {meta_key}"}

    async def chat(self, body: dict[str, Any]) -> bytes:
        try:
            response = await self._client.post(BASETEN_CHAT_URL, json=body, headers=self._baseten)
        except httpx.HTTPError:
            raise UpstreamFailed(None, True) from None
        if response.status_code != 200:
            raise _failed(response.status_code)
        return response.content

    async def responses(self, body: dict[str, Any]) -> AsyncIterator[bytes]:
        request = self._client.build_request(
            "POST", META_RESPONSES_URL, json=body, headers=self._meta
        )
        try:
            response = await self._client.send(request, stream=True)
        except httpx.HTTPError:
            raise UpstreamFailed(None, True) from None
        if response.status_code != 200:
            await response.aclose()
            raise _failed(response.status_code)
        return _chunks(response)


async def _chunks(response: httpx.Response) -> AsyncIterator[bytes]:
    try:
        async for chunk in response.aiter_bytes():
            yield chunk
    finally:
        await response.aclose()


class ModalSam3d:
    """The deployed ``Sam3dWorker`` of ``models/sam3d_modal.py``, called as the pipeline's own
    ``ModalTransport`` calls it. The gateway runs in the same Modal workspace, so it needs no
    Modal token and the worker gets no public endpoint."""

    def __init__(self, app: str, cls: str, max_idle_s: float) -> None:
        self._app, self._cls, self._max_idle_s = app, cls, max_idle_s
        self._worker: Any = None

    async def sample(self, request: Sam3dRequest) -> Sam3dResult:
        import modal
        import modal.exception

        if self._worker is None:
            self._worker = modal.Cls.from_name(self._app, self._cls)()
        try:
            out = await self._worker.sample.remote.aio(
                request.image,
                request.mask_png,
                request.seed,
                request.pointmap_npy,
                request.options,
            )
        except (modal.exception.Error, OSError, TimeoutError):
            raise UpstreamFailed(None, True) from None
        except Exception:
            # Raised by the worker's own code, such as "SAM 3D returned no mesh": the draw is
            # seeded, so the same request would fail again.
            raise UpstreamFailed(None, False) from None
        timing = gpu_timing(out.get("stats"), self._max_idle_s)
        return Sam3dResult(bytes(out["glb"]), str(out["pose_json"]), timing)


def gpu_timing(stats: Any, max_idle_s: float) -> GpuTiming | None:
    """The timing of a draw from the ``stats`` that ``Sam3dWorker.sample`` returns.

    A container's start is its ``boot_started_utc``. A container already running for someone
    else (the operator's own runs) would bring its whole history along, so the start is at most
    weight loading, one idle window (``max_idle_s``, the worker's scaledown) and the draw before
    the draw's end. A container idle for longer would have stopped."""
    try:
        container = str(stats["container"])
        boot = float(stats["boot_started_utc"])
        end = float(stats["sample_started_utc"]) + float(stats["sample_s"])
        busy = float(stats["load_s"]) + max_idle_s + float(stats["sample_s"])
    except (KeyError, TypeError, ValueError):
        return None
    if not container:
        return None
    return GpuTiming(container, max(boot, end - busy), end)
