"""Request checks: only the shapes the pipeline's own clients send get through.

Each check returns the body to forward, rebuilt from the fields it allows, so nothing a caller
adds reaches an upstream. Images and videos must be ``data:`` URLs, so no upstream is ever asked
to fetch a URL.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from v2s_gateway.policy import Sam3dPolicy, TrackingPolicy, VlmPolicy

IMAGE_URL_PREFIXES = ("data:image/png;base64,", "data:image/jpeg;base64,")
VIDEO_URL_PREFIX = "data:video/mp4;base64,"
DIRECTIONS = ("both", "forward", "backward")
SAM3D_OPTION_KEYS = frozenset({"checkpoint", "stage1_inference_steps", "stage2_inference_steps"})
MAX_SEED = 2**32 - 1
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8\xff"
NPY_MAGIC = b"\x93NUMPY"


class Invalid(Exception):
    """The request does not have a shape the gateway forwards. The message is safe to show."""


def _object(value: Any, what: str, allowed: frozenset[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise Invalid(f"{what} must be a JSON object")
    extra = sorted(set(value) - allowed)
    if extra:
        raise Invalid(f"{what} has fields the gateway does not forward: {', '.join(extra)}")
    return value


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def parse_json(raw: bytes) -> Any:
    try:
        return json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise Invalid("the body is not JSON") from None


def _chat_part(part: Any) -> tuple[dict[str, Any], bool]:
    """A content part to forward, and whether it is an image."""
    if not isinstance(part, dict):
        raise Invalid("a content part must be an object")
    kind = part.get("type")
    if kind == "text":
        _object(part, "a text part", frozenset({"type", "text"}))
        if not isinstance(part.get("text"), str):
            raise Invalid("a text part needs text")
        return {"type": "text", "text": part["text"]}, False
    if kind == "image_url":
        _object(part, "an image part", frozenset({"type", "image_url"}))
        image = _object(part.get("image_url"), "image_url", frozenset({"url"}))
        url = image.get("url")
        if not isinstance(url, str) or not url.startswith(IMAGE_URL_PREFIXES):
            raise Invalid("images must be PNG or JPEG data: URLs")
        return {"type": "image_url", "image_url": {"url": url}}, True
    raise Invalid("content parts must be text or image_url")


def check_chat(body: Any, policy: VlmPolicy) -> dict[str, Any]:
    """An OpenAI chat-completions body limited to model, messages, temperature and max_tokens.
    ``max_tokens`` is clamped to the cap, and set to it when missing."""
    body = _object(body, "the body", frozenset({"model", "messages", "temperature", "max_tokens"}))
    model = body.get("model")
    if model not in policy.models:
        raise Invalid(f"model must be one of {', '.join(policy.models)}")
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise Invalid("messages must be a non-empty list")
    clean: list[dict[str, Any]] = []
    images = 0
    for message in messages:
        message = _object(message, "a message", frozenset({"role", "content"}))
        if message.get("role") not in ("system", "user"):
            raise Invalid("message roles must be system or user")
        content = message.get("content")
        if isinstance(content, str):
            clean.append({"role": message["role"], "content": content})
            continue
        if not isinstance(content, list) or not content:
            raise Invalid("message content must be text or a non-empty list of parts")
        parts = []
        for part in content:
            forwarded, is_image = _chat_part(part)
            parts.append(forwarded)
            images += is_image
        clean.append({"role": message["role"], "content": parts})
    if images > policy.max_images:
        raise Invalid(f"at most {policy.max_images} images per request")
    out: dict[str, Any] = {"model": model, "messages": clean}
    if "temperature" in body and body["temperature"] is not None:
        temperature = body["temperature"]
        if not _number(temperature) or not 0 <= temperature <= 2:
            raise Invalid("temperature must be a number from 0 to 2")
        out["temperature"] = temperature
    max_tokens = body.get("max_tokens")
    if max_tokens is not None and (
        not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1
    ):
        raise Invalid("max_tokens must be a positive integer")
    out["max_tokens"] = min(max_tokens or policy.max_tokens_cap, policy.max_tokens_cap)
    return out


def check_responses(body: Any, policy: TrackingPolicy) -> dict[str, Any]:
    """A Meta Responses body with one user message: one text prompt and one MP4 clip."""
    body = _object(body, "the body", frozenset({"model", "stream", "metadata", "input"}))
    model = body.get("model")
    if model not in policy.models:
        raise Invalid(f"model must be one of {', '.join(policy.models)}")
    if body.get("stream") is not True:
        raise Invalid("stream must be true")
    out: dict[str, Any] = {"model": model, "stream": True}
    if body.get("metadata") is not None:
        metadata = _object(body["metadata"], "metadata", frozenset({"propagation_direction"}))
        if "propagation_direction" in metadata:
            if metadata["propagation_direction"] not in DIRECTIONS:
                raise Invalid(f"propagation_direction must be one of {', '.join(DIRECTIONS)}")
            out["metadata"] = {"propagation_direction": metadata["propagation_direction"]}
    items = body.get("input")
    if not isinstance(items, list) or len(items) != 1:
        raise Invalid("input must hold exactly one message")
    message = _object(items[0], "the message", frozenset({"type", "role", "content"}))
    if message.get("type") != "message" or message.get("role") != "user":
        raise Invalid("the message must be a user message")
    content = message.get("content")
    if not isinstance(content, list) or len(content) != 2:
        raise Invalid("the message must hold one input_text and one input_video")
    text = next((p for p in content if isinstance(p, dict) and p.get("type") == "input_text"), None)
    video = next(
        (p for p in content if isinstance(p, dict) and p.get("type") == "input_video"), None
    )
    if text is None or video is None:
        raise Invalid("the message must hold one input_text and one input_video")
    _object(text, "input_text", frozenset({"type", "text"}))
    _object(video, "input_video", frozenset({"type", "video_url"}))
    prompt, url = text.get("text"), video.get("video_url")
    if not isinstance(prompt, str) or not prompt.strip():
        raise Invalid("the prompt must be non-empty text")
    if len(prompt) > policy.max_prompt_chars:
        raise Invalid(f"the prompt is longer than {policy.max_prompt_chars} characters")
    if not isinstance(url, str) or not url.startswith(VIDEO_URL_PREFIX):
        raise Invalid("the clip must be an MP4 data: URL")
    out["input"] = [
        {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_video", "video_url": url},
            ],
        }
    ]
    return out


@dataclass(frozen=True)
class Sam3dRequest:
    """One SAM 3D draw, in the arguments of ``Sam3dWorker.sample``."""

    image: bytes
    mask_png: bytes
    seed: int
    pointmap_npy: bytes | None
    options: dict[str, Any]


def check_sam3d(
    image: bytes,
    mask_png: bytes,
    seed_text: str,
    options_text: str,
    pointmap_npy: bytes | None,
    policy: Sam3dPolicy,
) -> Sam3dRequest:
    """The form of ``models/sam3d_http.py``, with the options of ``Sam3dConfig.options()``."""
    mb = 1024 * 1024
    if not image.startswith((PNG_MAGIC, JPEG_MAGIC)):
        raise Invalid("image must be a PNG or JPEG file")
    if len(image) > policy.max_image_mb * mb:
        raise Invalid(f"image is larger than {policy.max_image_mb} MB")
    if not mask_png.startswith(PNG_MAGIC):
        raise Invalid("mask must be a PNG file")
    if len(mask_png) > policy.max_mask_mb * mb:
        raise Invalid(f"mask is larger than {policy.max_mask_mb} MB")
    if pointmap_npy is not None:
        if not pointmap_npy.startswith(NPY_MAGIC):
            raise Invalid("pointmap must be a .npy file")
        if len(pointmap_npy) > policy.max_pointmap_mb * mb:
            raise Invalid(f"pointmap is larger than {policy.max_pointmap_mb} MB")
    if not re.fullmatch(r"[0-9]{1,10}", seed_text.strip()) or int(seed_text) > MAX_SEED:
        raise Invalid(f"seed must be an integer from 0 to {MAX_SEED}")
    options = parse_json(options_text.encode())
    if not isinstance(options, dict) or set(options) != SAM3D_OPTION_KEYS:
        raise Invalid(f"options must have exactly the keys {', '.join(sorted(SAM3D_OPTION_KEYS))}")
    if options["checkpoint"] not in policy.checkpoints:
        raise Invalid(f"checkpoint must be one of {', '.join(policy.checkpoints)}")
    for key in ("stage1_inference_steps", "stage2_inference_steps"):
        steps = options[key]
        if steps is None:
            continue
        cap = policy.max_inference_steps
        if cap is None or not isinstance(steps, int) or isinstance(steps, bool):
            raise Invalid(f"{key} must be null" if cap is None else f"{key} must be an integer")
        if not 1 <= steps <= cap:
            raise Invalid(f"{key} must be from 1 to {cap}")
    return Sam3dRequest(image, mask_png, int(seed_text), pointmap_npy, dict(options))
