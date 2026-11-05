"""The gateway's HTTP app: the wire formats the pipeline's clients already speak.

=========================  ==============================  ===========================
Route                      Client                          Upstream
=========================  ==============================  ===========================
POST /v1/chat/completions  ``models/vlm.py`` (base_url)    Baseten chat completions
POST /v1/responses         ``stages/tracking.py``          Meta Model API, streamed
POST /sam3d/sample         ``models/sam3d_http.py``        the Modal SAM 3D worker
GET  /sam3d/health         ``video2sim doctor``            none
GET  /v1/usage             ``video2sim doctor``            none
=========================  ==============================  ===========================

Status codes follow the clients' retry rule (``post_json`` and ``HttpTransport`` retry 429 and
5xx): 401 bad token, 403 quota or daily budget used up (not retried), 413 too large, 400 or 422
not forwardable, 429 too many requests in flight (retried), 503 switched off or upstream down.
Every refusal is JSON with ``error`` and ``message`` and the ``X-V2S-Gateway`` header, which the
client uses to show the message. Logs carry metadata only: path, token label, status, bytes and
milliseconds, never a body, a header or a token.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, NoReturn

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.datastructures import UploadFile
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from v2s_gateway.checks import (
    Invalid,
    Sam3dRequest,
    check_chat,
    check_responses,
    check_sam3d,
    parse_json,
)
from v2s_gateway.ledger import Busy, ConcurrencyGate, Ledger, Refused, Reservation
from v2s_gateway.policy import MB, ROUTES, Policy, Route, Sam3dPolicy
from v2s_gateway.upstreams import UpstreamFailed, Upstreams

LOG = logging.getLogger("v2s_gateway")
GATEWAY_HEADER = "X-V2S-Gateway"
POSE_HEADER = "X-Pose-Json"
REPLAY_HINT = "`uv run video2sim replay` shows recorded runs without any model or key."
_FORM_FIELD_BYTES = 64 * 1024


class GatewayError(Exception):
    """A refusal. ``message`` is shown to the judge, so it never holds upstream text."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


@dataclass(frozen=True)
class Caller:
    token_hash: str
    label: str
    expires_utc: str
    scenes: int | None


class AccessLog:
    """One metadata line per request, written when the response (or its stream) ends."""

    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return
        started = time.monotonic()
        counts = {"status": 0, "in": 0, "out": 0}

        async def counting_receive() -> Message:
            message = await receive()
            if message["type"] == "http.request":
                counts["in"] += len(message.get("body", b""))
            return message

        async def counting_send(message: Message) -> None:
            if message["type"] == "http.response.start":
                counts["status"] = int(message["status"])
            elif message["type"] == "http.response.body":
                counts["out"] += len(message.get("body", b""))
            await send(message)

        try:
            await self._app(scope, counting_receive, counting_send)
        finally:
            LOG.info(
                "path=%s label=%s status=%d in=%d out=%d ms=%d",
                scope.get("path", "?"),
                scope.get("state", {}).get("label", "-"),
                counts["status"],
                counts["in"],
                counts["out"],
                (time.monotonic() - started) * 1000,
            )


def _refusal(error: GatewayError) -> JSONResponse:
    return JSONResponse(
        {"error": error.code, "message": error.message},
        status_code=error.status,
        headers={GATEWAY_HEADER: "1"},
    )


def _switched_off() -> GatewayError:
    return GatewayError(503, "switched_off", f"The judge gateway is switched off. {REPLAY_HINT}")


def _refused(refused: Refused) -> GatewayError:
    if refused.reason == "quota":
        return GatewayError(
            403,
            "quota_exhausted",
            f"This judge token has used its {refused.route} quota of {refused.limit} calls. "
            f"{REPLAY_HINT}",
        )
    return GatewayError(
        403,
        "daily_budget",
        f"The gateway's {refused.route} budget for today ({refused.limit} calls) is used up; "
        f"it resets at 00:00 UTC. {REPLAY_HINT}",
    )


async def read_body(request: Request, limit: int) -> bytes:
    """The request body, or 413 as soon as it passes ``limit`` bytes."""
    too_large = GatewayError(413, "too_large", f"The body is larger than {limit // MB} MB.")
    length = request.headers.get("content-length", "")
    if length.isdigit() and int(length) > limit:
        raise too_large
    data = bytearray()
    async for chunk in request.stream():
        data += chunk
        if len(data) > limit:
            raise too_large
    return bytes(data)


async def read_sam3d_form(scope: Scope, raw: bytes, policy: Sam3dPolicy) -> Sam3dRequest:
    """The SAM 3D form of ``raw``, the already size-checked body."""

    async def replay() -> Message:
        return {"type": "http.request", "body": raw, "more_body": False}

    async def file(form: Any, name: str) -> bytes | None:
        value = form.get(name)
        if value is None:
            return None
        if not isinstance(value, UploadFile):
            raise Invalid(f"{name} must be a file")
        data: bytes = await value.read()
        return data

    allowed = {"image", "mask", "pointmap", "seed", "options"}
    try:
        async with Request(scope, replay).form(
            max_files=3, max_fields=2, max_part_size=_FORM_FIELD_BYTES
        ) as form:
            extra = sorted(set(form.keys()) - allowed)
            if extra:
                raise Invalid(f"the form has fields the gateway does not forward: {extra}")
            image, mask = await file(form, "image"), await file(form, "mask")
            pointmap = await file(form, "pointmap")
            seed, options = form.get("seed"), form.get("options")
    except Invalid:
        raise
    except Exception:
        raise Invalid("the body is not a readable multipart form") from None
    if image is None or mask is None or not isinstance(seed, str) or not isinstance(options, str):
        raise Invalid("the form needs image, mask, seed and options")
    return check_sam3d(image, mask, seed, options, pointmap, policy)


def create_app(policy: Policy, ledger: Ledger, upstreams: Upstreams) -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(AccessLog)
    gate = ConcurrencyGate()

    async def gateway_error(request: Request, error: Exception) -> Response:
        assert isinstance(error, GatewayError)
        return _refusal(error)

    app.add_exception_handler(GatewayError, gateway_error)

    async def authenticate(request: Request, check_kill: bool = True) -> Caller:
        if check_kill and await ledger.killed():
            raise _switched_off()
        scheme, _, token = request.headers.get("authorization", "").partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            raise GatewayError(
                401,
                "no_token",
                "Send your judge token as `Authorization: Bearer ...`; video2sim reads it "
                "from V2S_JUDGE_TOKEN.",
            )
        found = await ledger.token(token.strip())
        if found is None:
            raise GatewayError(
                401, "unknown_token", f"This judge token is unknown or was revoked. {REPLAY_HINT}"
            )
        token_hash, record = found
        if record.expires_utc <= ledger.now():
            raise GatewayError(
                401,
                "expired_token",
                f"This judge token expired on {record.expires_utc:%Y-%m-%d %H:%M} UTC. "
                f"{REPLAY_HINT}",
            )
        request.state.label = record.label
        return Caller(token_hash, record.label, record.expires_utc.isoformat(), record.scenes)

    async def admit(caller: Caller, route: Route) -> tuple[tuple[str, str], Reservation]:
        """A concurrency slot and one counted call, or a refusal."""
        key = (caller.token_hash, route)
        limit = policy.limits(route).concurrency
        try:
            await gate.acquire(key, limit, policy.queue_wait_s)
        except Busy:
            raise GatewayError(
                429,
                "busy",
                f"This token already has {limit} {route} requests in flight; retry shortly.",
            ) from None
        try:
            reservation = await ledger.reserve(
                caller.token_hash,
                route,
                policy.quota(route, caller.scenes),
                policy.daily_cap(route),
            )
        except Refused as refused:
            await gate.release(key)
            raise _refused(refused) from None
        except BaseException:
            await gate.release(key)
            raise
        return key, reservation

    async def upstream_failed(reservation: Reservation, failure: UpstreamFailed) -> NoReturn:
        LOG.warning(
            "upstream route=%s status=%s retryable=%s",
            reservation.route,
            failure.status,
            failure.retryable,
        )
        if failure.retryable:
            await ledger.refund(reservation)
            raise GatewayError(
                503, "upstream_unavailable", "The model provider did not answer; retry later."
            )
        raise GatewayError(422, "upstream_rejected", "The model provider rejected this request.")

    def invalid(error: Invalid) -> GatewayError:
        return GatewayError(400, "invalid_request", str(error))

    @app.get("/health")
    @app.get("/sam3d/health")
    async def health() -> JSONResponse:
        if await ledger.killed():
            raise _switched_off()
        return JSONResponse({"status": "ok"})

    @app.get("/v1/usage")
    async def usage(request: Request) -> JSONResponse:
        caller = await authenticate(request, check_kill=False)
        used = await ledger.used(caller.token_hash)
        return JSONResponse(
            {
                "label": caller.label,
                "expires_utc": caller.expires_utc,
                "switched_off": await ledger.killed(),
                "routes": {
                    r: {"used": used[r], "quota": policy.quota(r, caller.scenes)} for r in ROUTES
                },
            }
        )

    @app.post("/v1/chat/completions")
    async def chat(request: Request) -> Response:
        caller = await authenticate(request)
        raw = await read_body(request, int(policy.vlm.max_body_mb * MB))
        try:
            body = check_chat(parse_json(raw), policy.vlm)
        except Invalid as error:
            raise invalid(error) from None
        key, reservation = await admit(caller, "vlm")
        try:
            answer = await upstreams.chat.chat(body)
        except UpstreamFailed as failure:
            await upstream_failed(reservation, failure)
        finally:
            await gate.release(key)
        return Response(answer, media_type="application/json")

    @app.post("/v1/responses")
    async def responses(request: Request) -> Response:
        caller = await authenticate(request)
        raw = await read_body(request, int(policy.tracking.max_body_mb * MB))
        try:
            body = check_responses(parse_json(raw), policy.tracking)
        except Invalid as error:
            raise invalid(error) from None
        key, reservation = await admit(caller, "tracking")
        try:
            chunks = await upstreams.tracking.responses(body)
        except UpstreamFailed as failure:
            await gate.release(key)
            await upstream_failed(reservation, failure)
        except BaseException:
            await gate.release(key)
            raise

        async def stream() -> AsyncIterator[bytes]:
            # A broken upstream stream propagates, so the client sees a cut connection rather
            # than a short answer it would read as fewer tracks.
            try:
                async for chunk in chunks:
                    yield chunk
            finally:
                close = getattr(chunks, "aclose", None)
                if close is not None:
                    await close()
                await gate.release(key)

        return StreamingResponse(
            stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
        )

    @app.post("/sam3d/sample")
    async def sam3d_sample(request: Request) -> Response:
        caller = await authenticate(request)
        p = policy.sam3d
        limit = int((p.max_image_mb + p.max_mask_mb + p.max_pointmap_mb) * MB) + _FORM_FIELD_BYTES
        raw = await read_body(request, limit)
        try:
            draw = await read_sam3d_form(request.scope, raw, p)
        except Invalid as error:
            raise invalid(error) from None
        key, reservation = await admit(caller, "sam3d")
        try:
            glb, pose = await upstreams.sam3d.sample(draw)
        except UpstreamFailed as failure:
            await upstream_failed(reservation, failure)
        finally:
            await gate.release(key)
        return Response(glb, media_type="model/gltf-binary", headers={POSE_HEADER: pose})

    return app
