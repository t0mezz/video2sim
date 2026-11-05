"""The judge gateway (``workers/gateway``) against fake upstreams; no network.

The VLM, tracking and SAM 3D clients are the pipeline's own, built from ``configs/default.yaml``
with ``apply_gateway``; their ``urllib`` calls are answered by the gateway app in a TestClient,
and the gateway's upstream calls by fake Baseten, Meta and SAM 3D services.
"""

from __future__ import annotations

import asyncio
import email.message
import io
import json
import logging
import re
import threading
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import anyio
import httpx
import pytest
import yaml
from fastapi.testclient import TestClient
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

import admin
import modal_app
from v2s_gateway.app import GATEWAY_HEADER, create_app
from v2s_gateway.checks import Sam3dRequest
from v2s_gateway.ledger import KILL_KEY, Busy, ConcurrencyGate, Ledger, MemoryKv, MintRefused
from v2s_gateway.policy import POLICY_FILE, Policy, load_policy
from v2s_gateway.upstreams import HttpUpstreams, UpstreamFailed, Upstreams
from video2sim.doctor import _one_pixel_png
from video2sim.gateway_profile import JUDGE_TOKEN_ENV, apply_gateway
from video2sim.models.sam3d import Sam3dCall, Sam3dConfig, make_transport
from video2sim.models.vlm import make_vlm_client
from video2sim.stages.tracking import make_tracking_client

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 11, 20, 12, 0, tzinfo=timezone.utc)
GATEWAY = "https://gw.test"
BASETEN_KEY = "bt-real-key-0123456789"
META_KEY = "meta-real-key-9876543210"
ADMIN_KEY = "admin-key-for-the-request-page"
UPSTREAM_HEADER = "X-Upstream-Account"
DELTAS = ["<0f>1<|box;x1=1;y1=2;x2=3;y2=4;w=8;h=8|>", "\n<1f>"]


def default_cfg() -> dict[str, Any]:
    cfg: dict[str, Any] = yaml.safe_load((ROOT / "configs/default.yaml").read_text("utf-8"))
    return cfg


def gateway_cfg() -> dict[str, Any]:
    return apply_gateway(default_cfg(), GATEWAY)


def make_policy(**changes: dict[str, Any] | float) -> Policy:
    data = yaml.safe_load(POLICY_FILE.read_text("utf-8"))
    for key, value in changes.items():
        if isinstance(value, dict):
            data[key].update(value)
        else:
            data[key] = value
    return Policy.model_validate(data)


@dataclass
class FakeProvider:
    """Baseten and Meta in one Starlette app. Answers carry an account header and, on errors,
    the key, which the gateway must never pass on."""

    chat_status: int = 200
    stream_status: int = 200
    calls: list[tuple[str, str | None, str, Any]] = field(default_factory=list)

    def app(self) -> Starlette:
        async def chat(request: Request) -> Response:
            self.calls.append(
                (
                    "chat",
                    request.url.hostname,
                    request.headers["authorization"],
                    await request.json(),
                )
            )
            headers = {UPSTREAM_HEADER: "acct-42"}
            if self.chat_status != 200:
                body = {"error": f"key {BASETEN_KEY} rejected for acct-42"}
                return JSONResponse(body, status_code=self.chat_status, headers=headers)
            answer = {"choices": [{"message": {"role": "assistant", "content": "OK"}}]}
            return JSONResponse(answer, headers=headers)

        async def responses(request: Request) -> Response:
            self.calls.append(
                (
                    "responses",
                    request.url.hostname,
                    request.headers["authorization"],
                    await request.json(),
                )
            )
            if self.stream_status != 200:
                body = {"error": f"key {META_KEY} rejected"}
                return JSONResponse(body, status_code=self.stream_status)

            async def events() -> Any:
                yield b'data: {"type": "response.created"}\n\n'
                for delta in DELTAS:
                    event = {"type": "response.output_text.delta", "delta": delta}
                    yield f"data: {json.dumps(event)}\n\n".encode()
                yield b'data: {"type": "response.completed"}\n\n'

            return StreamingResponse(
                events(), media_type="text/event-stream", headers={UPSTREAM_HEADER: "acct-42"}
            )

        return Starlette(
            routes=[
                Route("/v1/chat/completions", chat, methods=["POST"]),
                Route("/v1/responses", responses, methods=["POST"]),
            ]
        )


@dataclass
class FakeSam3d:
    calls: list[Sam3dRequest] = field(default_factory=list)
    fail: UpstreamFailed | None = None
    block: threading.Event | None = None
    entered: threading.Event = field(default_factory=threading.Event)

    async def sample(self, request: Sam3dRequest) -> tuple[bytes, str]:
        self.calls.append(request)
        self.entered.set()
        if self.block is not None:
            await anyio.to_thread.run_sync(self.block.wait)
        if self.fail is not None:
            raise self.fail
        pose = {"rotation_wxyz": [1, 0, 0, 0], "translation": [0, 0, 1], "scale": [1, 1, 1]}
        return b"glTF-fake-mesh", json.dumps({**pose, "checkpoint": "facebook/sam-3d-objects@x"})


def _message(headers: Any) -> email.message.Message:
    message = email.message.Message()
    for key, value in headers.items():
        message[key] = value
    return message


class _Answer:
    def __init__(self, response: httpx.Response) -> None:
        self._body = response.content
        self.headers = _message(response.headers)

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _Answer:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


@dataclass
class Bridge:
    """``urllib.request.urlopen`` answered by the gateway app, for the pipeline's own clients."""

    client: TestClient
    urls: list[str] = field(default_factory=list)

    def __call__(self, request: urllib.request.Request | str, timeout: Any = None) -> _Answer:
        if isinstance(request, str):
            request = urllib.request.Request(request)
        self.urls.append(request.full_url)
        parts = urllib.parse.urlsplit(request.full_url)
        assert f"{parts.scheme}://{parts.netloc}" == GATEWAY, "a client bypassed the gateway"
        data = request.data if isinstance(request.data, bytes) else None
        response = self.client.request(
            request.get_method(), parts.path, content=data, headers=dict(request.header_items())
        )
        if response.status_code >= 400:
            raise urllib.error.HTTPError(
                request.full_url,
                response.status_code,
                response.reason_phrase,
                _message(response.headers),
                io.BytesIO(response.content),
            )
        return _Answer(response)


def mint_token(data: dict[str, Any], label: str, days: float = 7, scenes: int | None = None) -> str:
    """A token minted the way ``admin.py`` mints, into the gateway's store."""
    ledger = Ledger(MemoryKv(data), clock=lambda: NOW)
    return asyncio.run(admin.mint(ledger, [label], days, scenes))[0][1]


@dataclass
class Harness:
    client: TestClient
    data: dict[str, Any]
    provider: FakeProvider
    sam3d: FakeSam3d
    token: str
    policy: Policy
    clock: list[datetime]

    def headers(self, token: str | None = None) -> dict[str, str]:
        return {"Authorization": f"Bearer {token or self.token}"}

    def usage(self) -> dict[str, Any]:
        out: dict[str, Any] = self.client.get("/v1/usage", headers=self.headers()).json()
        return out


@pytest.fixture
def make_harness() -> Iterator[Any]:
    clients: list[TestClient] = []

    def build(policy: Policy | None = None) -> Harness:
        policy = policy or load_policy()
        data: dict[str, Any] = {}
        clock = [NOW]
        token = mint_token(data, "judge-a")
        provider, sam3d = FakeProvider(), FakeSam3d()
        http = HttpUpstreams(
            httpx.AsyncClient(transport=httpx.ASGITransport(app=provider.app())),
            BASETEN_KEY,
            META_KEY,
        )
        ledger = Ledger(MemoryKv(data), clock=lambda: clock[0])
        app = create_app(policy, ledger, Upstreams(http, http, sam3d), ADMIN_KEY)
        client = TestClient(app, base_url=GATEWAY)
        clients.append(client)
        return Harness(client, data, provider, sam3d, token, policy, clock)

    yield build
    for client in clients:
        client.close()


@pytest.fixture
def harness(make_harness: Any) -> Harness:
    h: Harness = make_harness()
    return h


@pytest.fixture
def bridge(harness: Harness, monkeypatch: pytest.MonkeyPatch) -> Bridge:
    monkeypatch.setenv(JUDGE_TOKEN_ENV, harness.token)
    b = Bridge(harness.client)
    monkeypatch.setattr(urllib.request, "urlopen", b)
    return b


def png(tmp_path: Path, name: str = "pixel.png") -> Path:
    path = tmp_path / name
    _one_pixel_png(path)
    return path


def chat_body(**changes: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": "meta-models/muse-spark-open-1026",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        "temperature": 0,
    }
    body.update(changes)
    return body


def tracking_body(**changes: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": "sam-3.1",
        "stream": True,
        "metadata": {"propagation_direction": "both"},
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "objects on a table"},
                    {"type": "input_video", "video_url": "data:video/mp4;base64,AAAA"},
                ],
            }
        ],
    }
    body.update(changes)
    return body


def sam3d_form(
    tmp_path: Path, options: dict[str, Any] | None = None, image: bytes | None = None
) -> dict[str, Any]:
    pixel = png(tmp_path).read_bytes()
    opts = options or Sam3dConfig.from_config(default_cfg()).options()
    return {
        "files": {
            "image": ("image", image or pixel, "application/octet-stream"),
            "mask": ("mask.png", pixel, "image/png"),
        },
        "data": {"seed": "7", "options": json.dumps(opts)},
    }


def refusal(response: httpx.Response, status: int, code: str) -> str:
    assert response.status_code == status, response.text
    assert response.headers[GATEWAY_HEADER] == "1"
    body = response.json()
    assert body["error"] == code
    message: str = body["message"]
    return message


# --- policy and deployment ---------------------------------------------------------------


def test_shipped_policy_parses_and_sizes_quotas_by_the_formula() -> None:
    policy = load_policy()
    k = policy.vlm.calls_per_scene
    assert policy.quota("vlm") == round(policy.safety_factor * policy.scenes_per_token * k)
    assert policy.quota("vlm", scenes=1) == round(policy.safety_factor * k)
    assert policy.daily_cap("sam3d") >= policy.quota("sam3d")


def test_policy_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="extra"):
        make_policy(vlm={"model_allowlist": ["x"]})


def test_modal_image_pins_are_locked_versions() -> None:
    lock = (ROOT / "uv.lock").read_text("utf-8")
    locked = re.findall(r'^name = "([^"]+)"\nversion = "([^"]+)"', lock, flags=re.MULTILINE)
    for name, version in modal_app.PINS.items():
        assert (name, version) in locked, f"{name}=={version} is not in uv.lock"


# --- auth ----------------------------------------------------------------------------------


def test_missing_unknown_expired_and_revoked_tokens_get_401(harness: Harness) -> None:
    body = chat_body()
    message = refusal(harness.client.post("/v1/chat/completions", json=body), 401, "no_token")
    assert JUDGE_TOKEN_ENV in message
    response = harness.client.post(
        "/v1/chat/completions", json=body, headers=harness.headers("v2s_not-a-token")
    )
    assert "replay" in refusal(response, 401, "unknown_token")

    harness.clock[0] = NOW + timedelta(days=8)
    response = harness.client.post("/v1/chat/completions", json=body, headers=harness.headers())
    assert "expired on 2026-11-27" in refusal(response, 401, "expired_token")

    harness.clock[0] = NOW
    revoke = Ledger(MemoryKv(harness.data)).revoke("judge-a")
    assert asyncio.run(revoke) == 1
    response = harness.client.post("/v1/chat/completions", json=body, headers=harness.headers())
    refusal(response, 401, "unknown_token")
    assert harness.provider.calls == []


def test_tokens_are_stored_only_as_hashes(harness: Harness) -> None:
    assert harness.token.startswith("v2s_")
    assert harness.token not in json.dumps(harness.data)


def test_mint_refuses_a_label_in_use() -> None:
    data: dict[str, Any] = {}
    mint_token(data, "judge-a")
    with pytest.raises(MintRefused, match="judge-a"):
        mint_token(data, "judge-a")
    with pytest.raises(MintRefused, match="a-z"):
        mint_token(data, "judge@example.com")  # labels reach logs, so never an email
    with pytest.raises(ValueError, match="distinct"):
        asyncio.run(admin.mint(Ledger(MemoryKv(data)), ["judge-b", "judge-b"], 1))


# --- VLM route -----------------------------------------------------------------------------


def test_pipeline_vlm_client_works_through_the_gateway(
    harness: Harness, bridge: Bridge, tmp_path: Path
) -> None:
    client = make_vlm_client(gateway_cfg(), 60.0, 1, role="material")
    assert client.ask("What is it?", [png(tmp_path)]) == "OK"
    assert bridge.urls == [f"{GATEWAY}/v1/chat/completions"]
    [(kind, host, auth, body)] = harness.provider.calls
    assert (kind, host, auth) == ("chat", "inference.baseten.co", f"Bearer {BASETEN_KEY}")
    assert body["model"] == default_cfg()["vlm"]["model"]
    assert body["max_tokens"] == 4096  # the client sent none
    assert body["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/png")
    assert harness.usage()["routes"]["vlm"]["used"] == 1


def test_vlm_text_only_request_and_max_tokens_clamp(harness: Harness) -> None:
    body = chat_body(messages=[{"role": "user", "content": "merge these nouns"}], max_tokens=10**6)
    response = harness.client.post("/v1/chat/completions", json=body, headers=harness.headers())
    assert response.status_code == 200
    assert harness.provider.calls[0][3]["max_tokens"] == 4096
    small = chat_body(max_tokens=300)
    harness.client.post("/v1/chat/completions", json=small, headers=harness.headers())
    assert harness.provider.calls[1][3]["max_tokens"] == 300


@pytest.mark.parametrize(
    ("body", "fragment"),
    [
        (chat_body(tools=[{"type": "function"}]), "tools"),
        (chat_body(stream=True), "stream"),
        (chat_body(model="gpt-big"), "model must be one of"),
        (
            chat_body(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": "https://x.test/a.png"}}
                        ],
                    }
                ]
            ),
            "data: URLs",
        ),
        (chat_body(messages=[{"role": "assistant", "content": "hi"}]), "roles"),
        (chat_body(temperature=7), "temperature"),
        (chat_body(max_tokens=True), "max_tokens"),
    ],
)
def test_vlm_requests_outside_the_allowlist_get_400(
    harness: Harness, body: dict[str, Any], fragment: str
) -> None:
    response = harness.client.post("/v1/chat/completions", json=body, headers=harness.headers())
    assert fragment in refusal(response, 400, "invalid_request")
    assert harness.provider.calls == []
    assert harness.usage()["routes"]["vlm"]["used"] == 0


def test_vlm_image_count_and_body_size_caps(make_harness: Any) -> None:
    h: Harness = make_harness(make_policy(vlm={"max_images": 1, "max_body_mb": 0.001}))
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    body = chat_body(messages=[{"role": "user", "content": [image, image]}])
    response = h.client.post("/v1/chat/completions", json=body, headers=h.headers())
    assert "at most 1 images" in refusal(response, 400, "invalid_request")
    big = chat_body(messages=[{"role": "user", "content": "x" * 5000}])
    response = h.client.post("/v1/chat/completions", json=big, headers=h.headers())
    refusal(response, 413, "too_large")
    assert h.provider.calls == []


def test_upstream_errors_are_scrubbed_and_refunded_when_retryable(harness: Harness) -> None:
    harness.provider.chat_status = 401  # e.g. the gateway's own Baseten key was rotated
    response = harness.client.post(
        "/v1/chat/completions", json=chat_body(), headers=harness.headers()
    )
    refusal(response, 422, "upstream_rejected")
    assert BASETEN_KEY not in response.text and "acct-42" not in response.text
    assert UPSTREAM_HEADER not in response.headers
    assert harness.usage()["routes"]["vlm"]["used"] == 1  # a rejected request is charged

    harness.provider.chat_status = 503
    response = harness.client.post(
        "/v1/chat/completions", json=chat_body(), headers=harness.headers()
    )
    refusal(response, 503, "upstream_unavailable")
    assert BASETEN_KEY not in response.text
    assert harness.usage()["routes"]["vlm"]["used"] == 1  # an unserved one is not


def test_success_headers_from_upstream_are_not_passed_on(harness: Harness) -> None:
    response = harness.client.post(
        "/v1/chat/completions", json=chat_body(), headers=harness.headers()
    )
    assert response.status_code == 200
    assert UPSTREAM_HEADER not in response.headers
    assert response.json()["choices"][0]["message"]["content"] == "OK"


# --- tracking route ------------------------------------------------------------------------


def test_pipeline_tracking_client_gets_the_stream_through(
    harness: Harness, bridge: Bridge, tmp_path: Path
) -> None:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"\x00\x00\x00\x18ftypmp42fake")
    text = make_tracking_client(gateway_cfg()).track(clip, "objects on a table", "both")
    assert text == "\n".join(DELTAS)
    [(kind, host, auth, body)] = harness.provider.calls
    assert (kind, host, auth) == ("responses", "api.meta.ai", f"Bearer {META_KEY}")
    assert body["model"] == "sam-3.1" and body["stream"] is True
    assert harness.usage()["routes"]["tracking"]["used"] == 1


def test_stream_is_passed_through_as_server_sent_events(harness: Harness) -> None:
    response = harness.client.post("/v1/responses", json=tracking_body(), headers=harness.headers())
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert UPSTREAM_HEADER not in response.headers
    assert response.text.startswith('data: {"type": "response.created"}')
    assert response.text.count("response.output_text.delta") == len(DELTAS)


def test_failed_stream_start_is_scrubbed_and_refunded(harness: Harness) -> None:
    harness.provider.stream_status = 500
    response = harness.client.post("/v1/responses", json=tracking_body(), headers=harness.headers())
    refusal(response, 503, "upstream_unavailable")
    assert META_KEY not in response.text
    assert harness.usage()["routes"]["tracking"]["used"] == 0


@pytest.mark.parametrize(
    ("body", "fragment"),
    [
        (tracking_body(stream=False), "stream"),
        (tracking_body(tools=[]), "tools"),
        (tracking_body(model="sam-4"), "model"),
        (tracking_body(metadata={"propagation_direction": "sideways"}), "propagation"),
        (
            tracking_body(
                input=[
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "objects"},
                            {"type": "input_video", "video_url": "https://x.test/v.mp4"},
                        ],
                    }
                ]
            ),
            "MP4 data: URL",
        ),
    ],
)
def test_tracking_requests_outside_the_allowlist_get_400(
    harness: Harness, body: dict[str, Any], fragment: str
) -> None:
    response = harness.client.post("/v1/responses", json=body, headers=harness.headers())
    assert fragment in refusal(response, 400, "invalid_request")
    assert harness.provider.calls == []


# --- SAM 3D route --------------------------------------------------------------------------


def test_pipeline_sam3d_transport_works_through_the_gateway(
    harness: Harness, bridge: Bridge, tmp_path: Path
) -> None:
    cfg = gateway_cfg()
    transport = make_transport(cfg)
    health = getattr(transport, "health")  # noqa: B009 - HttpTransport only
    health()
    pixel = png(tmp_path).read_bytes()
    options = Sam3dConfig.from_config(cfg).options()
    answer = transport(Sam3dCall(pixel, pixel, 3, None, options))
    assert answer.glb == b"glTF-fake-mesh"
    assert json.loads(answer.pose_json)["checkpoint"].startswith("facebook/sam-3d-objects")
    [request] = harness.sam3d.calls
    assert (request.seed, request.options, request.pointmap_npy) == (3, options, None)
    assert bridge.urls == [f"{GATEWAY}/sam3d/health", f"{GATEWAY}/sam3d/sample"]
    assert harness.usage()["routes"]["sam3d"]["used"] == 1


@pytest.mark.parametrize(
    ("options", "image", "fragment"),
    [
        ({"checkpoint": "facebook/sam-3d-objects", "extra": 1}, None, "exactly the keys"),
        (
            {
                "checkpoint": "someone/else",
                "stage1_inference_steps": None,
                "stage2_inference_steps": None,
            },
            None,
            "checkpoint",
        ),
        (
            {
                "checkpoint": "facebook/sam-3d-objects",
                "stage1_inference_steps": 500,
                "stage2_inference_steps": None,
            },
            None,
            "must be null",
        ),
        (None, b"GIF89a-not-allowed", "PNG or JPEG"),
    ],
)
def test_sam3d_requests_outside_the_allowlist_get_400(
    harness: Harness,
    tmp_path: Path,
    options: dict[str, Any] | None,
    image: bytes | None,
    fragment: str,
) -> None:
    form = sam3d_form(tmp_path, options, image)
    response = harness.client.post("/sam3d/sample", headers=harness.headers(), **form)
    assert fragment in refusal(response, 400, "invalid_request")
    assert harness.sam3d.calls == []


def test_sam3d_size_caps(make_harness: Any, tmp_path: Path) -> None:
    h: Harness = make_harness(make_policy(sam3d={"max_image_mb": 0.0001}))
    big = png(tmp_path).read_bytes() + b"\x00" * 2000
    response = h.client.post(
        "/sam3d/sample", headers=h.headers(), **sam3d_form(tmp_path, None, big)
    )
    assert "larger than" in refusal(response, 400, "invalid_request")
    huge = b"\x89PNG\r\n\x1a\n" + b"\x00" * (70 * 1024 * 1024)
    response = h.client.post("/sam3d/sample", headers=h.headers(), content=huge)
    refusal(response, 413, "too_large")
    assert h.sam3d.calls == []


def test_sam3d_worker_failures_map_to_retry_codes(harness: Harness, tmp_path: Path) -> None:
    harness.sam3d.fail = UpstreamFailed(None, retryable=False)
    response = harness.client.post(
        "/sam3d/sample", headers=harness.headers(), **sam3d_form(tmp_path)
    )
    refusal(response, 422, "upstream_rejected")
    harness.sam3d.fail = UpstreamFailed(None, retryable=True)
    response = harness.client.post(
        "/sam3d/sample", headers=harness.headers(), **sam3d_form(tmp_path)
    )
    refusal(response, 503, "upstream_unavailable")
    assert harness.usage()["routes"]["sam3d"]["used"] == 1


# --- quotas, budget, concurrency, kill switch ----------------------------------------------


def test_quota_exhaustion_is_403_with_a_replay_hint_through_the_client(
    make_harness: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    h: Harness = make_harness(
        make_policy(scenes_per_token=1, safety_factor=1, vlm={"calls_per_scene": 1})
    )
    monkeypatch.setenv(JUDGE_TOKEN_ENV, h.token)
    monkeypatch.setattr(urllib.request, "urlopen", Bridge(h.client))
    client = make_vlm_client(gateway_cfg(), 60.0, 4, role="router")
    assert client.ask("first", [png(tmp_path)]) == "OK"
    with pytest.raises(urllib.error.HTTPError) as caught:
        client.ask("second", [png(tmp_path)])
    assert caught.value.code == 403
    assert "used its vlm quota of 1 calls" in str(caught.value)
    assert "video2sim replay" in str(caught.value)
    assert len(h.provider.calls) == 1  # 403 is not retried


def test_daily_budget_covers_all_tokens(make_harness: Any) -> None:
    h: Harness = make_harness(
        make_policy(daily_scenes=1, safety_factor=1, vlm={"calls_per_scene": 1})
    )
    second = mint_token(h.data, "judge-b")
    ok = h.client.post("/v1/chat/completions", json=chat_body(), headers=h.headers())
    assert ok.status_code == 200
    response = h.client.post("/v1/chat/completions", json=chat_body(), headers=h.headers(second))
    assert "00:00 UTC" in refusal(response, 403, "daily_budget")
    h.clock[0] = NOW + timedelta(days=1)
    again = h.client.post("/v1/chat/completions", json=chat_body(), headers=h.headers(second))
    assert again.status_code == 200


def test_requests_over_the_concurrency_limit_get_429(make_harness: Any, tmp_path: Path) -> None:
    h: Harness = make_harness(make_policy(queue_wait_s=0, sam3d={"concurrency": 1}))
    h.sam3d.block = threading.Event()
    form = sam3d_form(tmp_path)
    first: list[int] = []
    worker = threading.Thread(
        target=lambda: first.append(
            h.client.post("/sam3d/sample", headers=h.headers(), **form).status_code
        )
    )
    worker.start()
    assert h.sam3d.entered.wait(5)
    try:
        response = h.client.post("/sam3d/sample", headers=h.headers(), **form)
        refusal(response, 429, "busy")
        other = h.client.post("/v1/chat/completions", json=chat_body(), headers=h.headers())
        assert other.status_code == 200  # the limit is per route
    finally:
        h.sam3d.block.set()
        worker.join(5)
    assert first == [200]
    assert h.client.post("/sam3d/sample", headers=h.headers(), **form).status_code == 200
    assert h.usage()["routes"]["sam3d"]["used"] == 2  # the refused request was not charged


def test_concurrency_gate_queues_until_a_slot_frees() -> None:
    async def scenario() -> list[str]:
        gate, order = ConcurrencyGate(), []
        key = ("token", "sam3d")
        await gate.acquire(key, 1, 0)

        async def waiter() -> None:
            await gate.acquire(key, 1, 5)
            order.append("second in")
            await gate.release(key)

        task = asyncio.create_task(waiter())
        await asyncio.sleep(0.01)
        order.append("first out")
        await gate.release(key)
        await task
        with pytest.raises(Busy):
            await gate.acquire(key, 0, 0.01)
        assert gate.active(key) == 0
        return order

    assert asyncio.run(scenario()) == ["first out", "second in"]


def test_kill_switch_stops_every_route_but_usage(harness: Harness, tmp_path: Path) -> None:
    harness.data[KILL_KEY] = True
    h = harness.headers()
    assert "switched off" in refusal(harness.client.get("/health"), 503, "switched_off")
    refusal(harness.client.get("/sam3d/health"), 503, "switched_off")
    refusal(
        harness.client.post("/v1/chat/completions", json=chat_body(), headers=h),
        503,
        "switched_off",
    )
    refusal(
        harness.client.post("/v1/responses", json=tracking_body(), headers=h), 503, "switched_off"
    )
    response = harness.client.post("/sam3d/sample", headers=h, **sam3d_form(tmp_path))
    refusal(response, 503, "switched_off")
    assert harness.usage()["switched_off"] is True
    assert harness.provider.calls == [] and harness.sam3d.calls == []
    harness.data[KILL_KEY] = False
    assert harness.client.get("/health").status_code == 200


# --- logs ----------------------------------------------------------------------------------


def test_logs_hold_metadata_only(
    harness: Harness, caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    marker = "PROMPT-MARKER-5f3a"
    caplog.set_level(logging.DEBUG)
    h = harness.headers()
    harness.client.post(
        "/v1/chat/completions",
        json=chat_body(messages=[{"role": "user", "content": marker}]),
        headers=h,
    )
    harness.client.post("/v1/responses", json=tracking_body(), headers=h)
    harness.provider.chat_status = 500
    harness.client.post("/v1/chat/completions", json=chat_body(), headers=h)
    harness.client.post("/v1/chat/completions", json={"model": marker}, headers=h)
    text = "\n".join(record.getMessage() for record in caplog.records)
    for secret in (marker, harness.token, BASETEN_KEY, META_KEY, "acct-42", "base64"):
        assert secret not in text
    access = [r.getMessage() for r in caplog.records if r.name == "v2s_gateway"]
    assert any(
        line.startswith("path=/v1/chat/completions label=judge-a status=200") for line in access
    )
    assert any("path=/v1/responses label=judge-a status=200" in line for line in access)


def test_admin_report_lists_usage_against_quota(harness: Harness) -> None:
    harness.client.post("/v1/chat/completions", json=chat_body(), headers=harness.headers())
    out = io.StringIO()
    asyncio.run(admin.report(Ledger(MemoryKv(harness.data)), harness.policy, out))
    text = out.getvalue()
    assert text.startswith("gateway on")
    assert f"judge-a: expires 2026-11-27 12:00 UTC; vlm 1/{harness.policy.quota('vlm')}" in text


# --- admin endpoint for the token request page ----------------------------------------------


def admin_headers(key: str = ADMIN_KEY) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def test_admin_endpoint_mints_a_working_token_once(harness: Harness) -> None:
    response = harness.client.post(
        "/admin/tokens", json={"label": "j-7f3a", "days": 14, "scenes": 2}, headers=admin_headers()
    )
    assert response.status_code == 201, response.text
    minted = response.json()
    assert minted["label"] == "j-7f3a" and minted["token"].startswith("v2s_")
    assert minted["expires_utc"].startswith("2026-12-04T12:00")
    assert minted["quota"]["vlm"] == harness.policy.quota("vlm", scenes=2)
    assert minted["token"] not in json.dumps(harness.data)
    ok = harness.client.post(
        "/v1/chat/completions", json=chat_body(), headers=harness.headers(minted["token"])
    )
    assert ok.status_code == 200
    again = harness.client.post(
        "/admin/tokens", json={"label": "j-7f3a", "days": 14}, headers=admin_headers()
    )
    refusal(again, 409, "not_minted")


def test_admin_endpoint_needs_the_admin_key(harness: Harness) -> None:
    body = {"label": "j-1", "days": 1}
    for headers in ({}, admin_headers("wrong"), harness.headers()):
        refusal(harness.client.post("/admin/tokens", json=body, headers=headers), 401, "not_admin")
        refusal(harness.client.get("/admin/tokens", headers=headers), 401, "not_admin")
        response = harness.client.delete("/admin/tokens/judge-a", headers=headers)
        refusal(response, 401, "not_admin")
    assert harness.usage()["label"] == "judge-a"  # nothing was revoked


def test_admin_routes_are_absent_without_an_admin_key() -> None:
    policy = load_policy()
    http = HttpUpstreams(httpx.AsyncClient(), BASETEN_KEY, META_KEY)
    app = create_app(policy, Ledger(MemoryKv()), Upstreams(http, http, FakeSam3d()))
    with TestClient(app, base_url=GATEWAY) as client:
        response = client.post("/admin/tokens", json={"label": "j", "days": 1})
        refusal(response, 404, "not_found")


@pytest.mark.parametrize(
    ("body", "fragment"),
    [
        ({"label": "j-1", "days": 400}, "at most 2"),
        ({"label": "j-1", "days": 0}, "above 0"),
        ({"label": "j-1", "days": 1, "scenes": 99}, "scenes"),
        ({"label": "j-1", "days": 1, "quota": 10**6}, "{label, days, scenes}"),
        ({"label": "Judge Name", "days": 1}, "a-z"),
    ],
)
def test_admin_mint_bounds(make_harness: Any, body: dict[str, Any], fragment: str) -> None:
    h: Harness = make_harness(make_policy(admin={"max_days": 21, "max_scenes": 3, "max_tokens": 5}))
    response = h.client.post("/admin/tokens", json=body, headers=admin_headers())
    assert response.status_code == 400, response.text
    assert fragment in response.json()["message"]


def test_admin_mint_stops_at_the_token_cap_and_expired_tokens_free_room(
    make_harness: Any,
) -> None:
    h: Harness = make_harness(make_policy(admin={"max_tokens": 2}))  # judge-a is one
    first = h.client.post(
        "/admin/tokens", json={"label": "j-1", "days": 1}, headers=admin_headers()
    )
    assert first.status_code == 201
    full = h.client.post("/admin/tokens", json={"label": "j-2", "days": 1}, headers=admin_headers())
    assert "revoke some first" in refusal(full, 403, "not_minted")
    h.clock[0] = NOW + timedelta(days=2)  # j-1 has expired
    later = h.client.post(
        "/admin/tokens", json={"label": "j-2", "days": 1}, headers=admin_headers()
    )
    assert later.status_code == 201


def test_admin_list_and_revoke(harness: Harness) -> None:
    harness.client.post("/v1/chat/completions", json=chat_body(), headers=harness.headers())
    listing = harness.client.get("/admin/tokens", headers=admin_headers()).json()
    [row] = listing["tokens"]
    assert row["label"] == "judge-a" and row["used"]["vlm"] == 1
    assert "token" not in row
    response = harness.client.delete("/admin/tokens/judge-a", headers=admin_headers())
    assert response.json() == {"revoked": 1}
    refusal(harness.client.get("/v1/usage", headers=harness.headers()), 401, "unknown_token")
    gone = harness.client.delete("/admin/tokens/judge-a", headers=admin_headers())
    refusal(gone, 404, "not_found")
