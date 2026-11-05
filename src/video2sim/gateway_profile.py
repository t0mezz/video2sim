"""``--gateway URL``: point all three model endpoints at the judge gateway (``workers/gateway``).

The overlay is applied in code, not kept as a copied config, so it cannot drift from
``configs/default.yaml``. It sets the VLM base URL, the SAM 3.1 tracking URL and the SAM 3D HTTP
transport to the gateway and reads one key for all of them, the judge token in
``V2S_JUDGE_TOKEN``. Model names stay as they are, so response caches stay valid.

Standard library only.
"""

from __future__ import annotations

import copy
import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from typing import Any

from video2sim.models.hosted import explain_http_error, read_secret

JUDGE_TOKEN_ENV = "V2S_JUDGE_TOKEN"
_LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")


def gateway_base(url: str) -> str:
    """``url`` without a trailing slash. HTTPS is required except for a gateway on this machine."""
    parts = urllib.parse.urlsplit(url.strip())
    local = parts.hostname in _LOCAL_HOSTS
    if parts.scheme != "https" and not (parts.scheme == "http" and local):
        raise ValueError(f"the gateway URL must start with https:// (got {url!r})")
    if not parts.hostname or parts.query or parts.fragment:
        raise ValueError(f"the gateway URL must be a plain https://host[/path] (got {url!r})")
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path.rstrip("/"), "", ""))


def apply_gateway(cfg: Mapping[str, Any], url: str) -> dict[str, Any]:
    """A copy of ``cfg`` whose VLM, tracking and SAM 3D endpoints are the gateway at ``url``."""
    base = gateway_base(url)
    out: dict[str, Any] = copy.deepcopy(dict(cfg))

    vlm = out.setdefault("vlm", {})
    roles = vlm.get("roles") or {}
    for name, section in [("vlm", vlm), *((f"vlm.roles.{r}", s) for r, s in roles.items())]:
        if not isinstance(section, dict):
            continue
        if section.get("extra_body"):
            raise ValueError(
                f"{name}.extra_body is set, but the gateway forwards only model, messages, "
                "temperature and max_tokens"
            )
        if section is vlm or any(k in section for k in ("protocol", "base_url", "api_key_env")):
            section.update(
                protocol="openai_chat", base_url=f"{base}/v1", api_key_env=JUDGE_TOKEN_ENV
            )

    tracking = out.setdefault("segmentation", {}).setdefault("tracking", {})
    endpoint = dict(tracking.get("endpoint") or {})
    endpoint.update(
        protocol="meta_responses", url=f"{base}/v1/responses", api_key_env=JUDGE_TOKEN_ENV
    )
    tracking["endpoint"] = endpoint

    shape = out.setdefault("shape", {})
    if str(shape.get("backend", "sam3d")) != "sam3d":
        raise ValueError(f"the gateway serves the sam3d shape backend, not {shape['backend']!r}")
    http = dict(shape.get("http") or {})
    http.update(url=f"{base}/sam3d", api_key_env=JUDGE_TOKEN_ENV)
    shape["transport"] = "http"
    shape["http"] = http
    return out


def gateway_usage(url: str, timeout_s: float = 30.0) -> dict[str, Any]:
    """The judge token's usage and quotas (``GET /v1/usage``); calls no model."""
    request = urllib.request.Request(
        f"{gateway_base(url)}/v1/usage",
        headers={"Authorization": f"Bearer {read_secret(JUDGE_TOKEN_ENV)}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            usage: dict[str, Any] = json.loads(response.read())
    except urllib.error.HTTPError as error:
        raise explain_http_error(error) from error
    return usage


def describe_usage(usage: Mapping[str, Any]) -> str:
    routes = ", ".join(
        f"{name} {r['used']}/{r['quota']}" for name, r in (usage.get("routes") or {}).items()
    )
    state = "; switched off" if usage.get("switched_off") else ""
    return f"token {usage.get('label')}, expires {usage.get('expires_utc')}: {routes}{state}"
