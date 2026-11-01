"""Choice of the vision-language model behind the router, the material pass and the mass request.

``vlm.backend`` in ``configs/default.yaml`` is ``baseten`` (the hosted Muse Spark, key from
``BASETEN_API_KEY``) or ``openai_compatible``: any server that speaks the OpenAI chat-completions
protocol with image parts, such as llama.cpp's ``llama-server``, vLLM or Ollama. The second one
takes its URL and model name from config; a key is read from the environment variable named by
``api_key_env`` only when one is set, since a server on a private network usually needs none.

Standard library only. Every client has one method, ``ask(prompt, images) -> str``, and a ``model``
attribute that goes into the response-cache key, so switching model never serves an old answer.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from video2sim.models.hosted import (
    BASETEN_MODEL,
    BASETEN_URL,
    image_data_url,
    post_json,
    read_secret,
)

BACKENDS = ("baseten", "openai_compatible")


class ChatVlmClient:
    """One user message with text and images to a chat-completions endpoint, temperature 0."""

    def __init__(
        self,
        url: str,
        model: str,
        key: str | None,
        timeout_s: float,
        retries: int,
        max_tokens: int | None = None,
    ) -> None:
        self._url, self.model, self._key = url, model, key
        self._timeout_s, self._retries, self._max_tokens = timeout_s, retries, max_tokens

    def ask(self, prompt: str, images: Sequence[Path]) -> str:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        content += [
            {"type": "image_url", "image_url": {"url": image_data_url(path)}} for path in images
        ]
        body: dict[str, Any] = {
            "model": self.model,
            "temperature": 0,
            "messages": [{"role": "user", "content": content}],
        }
        if self._max_tokens is not None:
            body["max_tokens"] = self._max_tokens
        reply = json.loads(post_json(self._url, self._key, body, self._timeout_s, self._retries))
        return str(reply["choices"][0]["message"]["content"])


@dataclass(frozen=True)
class VlmConfig:
    """Values from ``vlm`` in ``configs/default.yaml``."""

    backend: str
    base_url: str
    model: str
    api_key_env: str | None
    max_tokens: int | None

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> VlmConfig:
        c = cfg.get("vlm", {})
        backend = str(c.get("backend", "baseten"))
        if backend not in BACKENDS:
            raise ValueError(f"vlm.backend must be one of {BACKENDS}, got {backend!r}")
        oc = c.get("openai_compatible", {})
        base_url = str(oc.get("base_url") or "").rstrip("/")
        model = str(oc.get("model") or "")
        if backend == "openai_compatible" and not (base_url and model):
            raise ValueError("vlm.openai_compatible needs base_url and model")
        env = oc.get("api_key_env")
        max_tokens = oc.get("max_tokens")
        if max_tokens is not None and int(max_tokens) < 1:
            raise ValueError("vlm.openai_compatible.max_tokens must be at least 1")
        return cls(
            backend,
            base_url,
            model,
            str(env) if env else None,
            None if max_tokens is None else int(max_tokens),
        )


def make_vlm_client(cfg: Mapping[str, Any], timeout_s: float, retries: int) -> ChatVlmClient:
    """The client of ``vlm.backend``, with the timeout and retries of the calling stage."""
    vlm = VlmConfig.from_config(cfg)
    if vlm.backend == "baseten":
        return ChatVlmClient(
            BASETEN_URL, BASETEN_MODEL, read_secret("BASETEN_API_KEY"), timeout_s, retries
        )
    key = read_secret(vlm.api_key_env) if vlm.api_key_env else None
    return ChatVlmClient(
        f"{vlm.base_url}/chat/completions", vlm.model, key, timeout_s, retries, vlm.max_tokens
    )


def cache_model(client: object) -> str:
    """Model name that goes into a response-cache key. Clients without one are the Baseten model."""
    return str(getattr(client, "model", BASETEN_MODEL))


__all__ = ["BACKENDS", "ChatVlmClient", "VlmConfig", "cache_model", "make_vlm_client"]
