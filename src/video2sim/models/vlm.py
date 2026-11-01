"""Choice of the vision-language model behind the router, the material pass and the mass request.

The ``vlm`` section of ``configs/default.yaml`` names one endpoint: a protocol, a base URL, a model
and the environment variable that holds its key. ``openai_chat`` covers the hosted Baseten Muse
Spark, llama.cpp, vLLM, Ollama and any other server with OpenAI chat completions and image parts.
``anthropic_messages`` covers the Anthropic Messages API. ``python:package.module:factory`` loads
a factory ``(settings, timeout_s, retries) -> client`` for anything else. The router, the material
pass and the mass request may each override fields under ``vlm.roles.<role>``.

Standard library only. Every client has one method, ``ask(prompt, images) -> str``, and a ``model``
attribute that goes into the response-cache key, so switching model never serves an old answer.
See docs/model_interfaces.md.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from video2sim.models.hosted import (
    BASETEN_MODEL,
    BASETEN_URL,
    image_data_url,
    post_json,
    read_secret,
)

PROTOCOLS = ("openai_chat", "anthropic_messages")
ROLES = ("router", "material", "mass")
ANTHROPIC_VERSION = "2023-06-01"
BASETEN_BASE_URL = BASETEN_URL.removesuffix("/chat/completions")


class VlmClient(Protocol):
    """What the stages need from a VLM. ``model`` is part of the response-cache key."""

    model: str

    def ask(self, prompt: str, images: Sequence[Path]) -> str: ...


class ChatVlmClient:
    """One user message with text and images to a chat-completions endpoint."""

    def __init__(
        self,
        url: str,
        model: str,
        key: str | None,
        timeout_s: float,
        retries: int,
        max_tokens: int | None = None,
        temperature: float | None = 0,
        extra_body: Mapping[str, Any] | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        self._url, self.model, self._key = url, model, key
        self._timeout_s, self._retries, self._max_tokens = timeout_s, retries, max_tokens
        self._temperature = temperature
        self._extra_body = dict(extra_body or {})
        self._extra_headers = dict(extra_headers or {})

    def ask(self, prompt: str, images: Sequence[Path]) -> str:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        content += [
            {"type": "image_url", "image_url": {"url": image_data_url(path)}} for path in images
        ]
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
        }
        if self._temperature is not None:
            body["temperature"] = self._temperature
        if self._max_tokens is not None:
            body["max_tokens"] = self._max_tokens
        body.update(self._extra_body)
        raw = post_json(
            self._url, self._key, body, self._timeout_s, self._retries, self._extra_headers
        )
        return str(json.loads(raw)["choices"][0]["message"]["content"])


class AnthropicVlmClient:
    """One user message to the Anthropic Messages API (``POST {base_url}/v1/messages``)."""

    def __init__(
        self,
        url: str,
        model: str,
        key: str | None,
        timeout_s: float,
        retries: int,
        max_tokens: int | None = None,
        temperature: float | None = 0,
        extra_body: Mapping[str, Any] | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        self._url, self.model, self._key = url, model, key
        self._timeout_s, self._retries = timeout_s, retries
        self._max_tokens = 4096 if max_tokens is None else max_tokens  # the API requires one
        self._temperature = temperature
        self._extra_body = dict(extra_body or {})
        self._extra_headers = dict(extra_headers or {})

    def ask(self, prompt: str, images: Sequence[Path]) -> str:
        content: list[dict[str, Any]] = []
        for path in images:
            mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
            data = image_data_url(path).split(",", 1)[1]
            content.append(
                {"type": "image", "source": {"type": "base64", "media_type": mime, "data": data}}
            )
        content.append({"type": "text", "text": prompt})
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self._max_tokens,
            "messages": [{"role": "user", "content": content}],
        }
        if self._temperature is not None:
            body["temperature"] = self._temperature
        body.update(self._extra_body)
        headers = {"anthropic-version": ANTHROPIC_VERSION, **self._extra_headers}
        if self._key:
            headers["x-api-key"] = self._key
        raw = post_json(self._url, None, body, self._timeout_s, self._retries, headers)
        blocks = json.loads(raw)["content"]
        return "".join(str(b["text"]) for b in blocks if b.get("type") == "text")


class BasetenMaterialClient(ChatVlmClient):
    """The hosted Muse Spark regardless of ``vlm.backend``, for the benchmark experiments that
    were run on it."""

    def __init__(self, key: str, timeout_s: float, retries: int) -> None:
        super().__init__(BASETEN_URL, BASETEN_MODEL, key, timeout_s, retries)


@dataclass(frozen=True)
class VlmConfig:
    """The ``vlm`` section of ``configs/default.yaml``, with the overrides of one role applied."""

    protocol: str
    base_url: str
    model: str
    api_key_env: str | None
    max_tokens: int | None
    temperature: float | None = 0
    extra_body: Mapping[str, Any] = field(default_factory=dict)
    extra_headers: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any], role: str | None = None) -> VlmConfig:
        c: dict[str, Any] = dict(cfg.get("vlm") or {})
        if role is not None:
            if role not in ROLES:
                raise ValueError(f"vlm role must be one of {ROLES}, got {role!r}")
            c.update((c.get("roles") or {}).get(role) or {})
        protocol = str(c.get("protocol", "openai_chat"))
        if protocol not in PROTOCOLS and not protocol.startswith("python:"):
            raise ValueError(
                f"vlm.protocol must be one of {PROTOCOLS} or python:module:factory, "
                f"got {protocol!r}"
            )
        base_url = str(c.get("base_url", BASETEN_BASE_URL) or "").rstrip("/")
        model = str(c.get("model", BASETEN_MODEL) or "")
        if not (base_url and model):
            raise ValueError("vlm needs base_url and model")
        env = c.get("api_key_env", "BASETEN_API_KEY")
        max_tokens = c.get("max_tokens")
        if max_tokens is not None and int(max_tokens) < 1:
            raise ValueError("vlm.max_tokens must be at least 1")
        temperature = c.get("temperature", 0)
        return cls(
            protocol,
            base_url,
            model,
            str(env) if env else None,
            None if max_tokens is None else int(max_tokens),
            None if temperature is None else float(temperature),
            dict(c.get("extra_body") or {}),
            {str(k): str(v) for k, v in (c.get("extra_headers") or {}).items()},
        )


def make_vlm_client(
    cfg: Mapping[str, Any], timeout_s: float, retries: int, role: str | None = None
) -> VlmClient:
    """The client of the ``vlm`` section, with the timeout and retries of the calling stage."""
    vlm = VlmConfig.from_config(cfg, role)
    if vlm.protocol.startswith("python:"):
        module, _, name = vlm.protocol.removeprefix("python:").partition(":")
        if not (module and name):
            raise ValueError("vlm.protocol python: needs the form python:package.module:factory")
        factory = getattr(importlib.import_module(module), name)
        client: VlmClient = factory(vlm, timeout_s, retries)
        return client
    key = read_secret(vlm.api_key_env) if vlm.api_key_env else None
    if vlm.protocol == "anthropic_messages":
        return AnthropicVlmClient(
            f"{vlm.base_url}/v1/messages",
            vlm.model,
            key,
            timeout_s,
            retries,
            vlm.max_tokens,
            vlm.temperature,
            vlm.extra_body,
            vlm.extra_headers,
        )
    return ChatVlmClient(
        f"{vlm.base_url}/chat/completions",
        vlm.model,
        key,
        timeout_s,
        retries,
        vlm.max_tokens,
        vlm.temperature,
        vlm.extra_body,
        vlm.extra_headers,
    )


def cache_model(client: object) -> str:
    """Model name that goes into a response-cache key. Clients without one are the Baseten model."""
    return str(getattr(client, "model", BASETEN_MODEL))


def cache_key_part(client: object) -> str | None:
    """What a cache key adds for this client's model. None for the model whose runs predate the
    configurable endpoint, so their cached answers stay valid."""
    model = cache_model(client)
    return None if model == BASETEN_MODEL else model


__all__ = [
    "PROTOCOLS",
    "AnthropicVlmClient",
    "BasetenMaterialClient",
    "ChatVlmClient",
    "VlmClient",
    "VlmConfig",
    "cache_key_part",
    "cache_model",
    "make_vlm_client",
]
