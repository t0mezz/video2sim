"""HTTP transport for SAM 3D: any server that implements the wire format below can stand in for
the Modal worker (``workers/sam3d_http`` is a reference server).

``POST {url}/sample`` with ``multipart/form-data``:

* ``image``: the keyframe, an encoded file;
* ``mask``: PNG, non-zero where the object is;
* ``pointmap``: optional float32 ``.npy``, HxWx3 in PyTorch3D axes;
* ``seed``: integer text;
* ``options``: JSON text, the dict of ``Sam3dConfig.options()``.

The answer is the GLB from SAM 3D's ``to_glb`` as the binary body, and the pose JSON of
``models/sam3d.py`` (``rotation_wxyz``, ``translation``, ``scale``, ``checkpoint``) in the
``X-Pose-Json`` header. ``GET {url}/health`` answers 200 once the weights are loaded.

Standard library only. A key, if configured, goes out as ``Authorization: Bearer``.
"""

from __future__ import annotations

import urllib.error
import urllib.request
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from video2sim.models.hosted import explain_http_error, read_secret
from video2sim.models.sam3d import Sam3dAnswer, Sam3dCall

POSE_HEADER = "X-Pose-Json"
_RETRY_CODES = (429, 500, 502, 503, 504)


@dataclass(frozen=True)
class HttpConfig:
    """Values from ``shape.http`` in ``configs/default.yaml``."""

    url: str
    api_key_env: str | None
    timeout_s: float
    retries: int

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> HttpConfig:
        c = cfg["shape"].get("http") or {}
        url = str(c.get("url") or "").rstrip("/")
        if not url:
            raise ValueError("shape.http.url must be set for shape.transport: http")
        retries = int(c.get("retries", 3))
        if retries < 1:
            raise ValueError("shape.http.retries must be at least 1")
        env = c.get("api_key_env")
        return cls(url, str(env) if env else None, float(c.get("timeout_s", 600)), retries)


def encode_multipart(
    fields: Mapping[str, str], files: Mapping[str, tuple[str, bytes, str]]
) -> tuple[bytes, str]:
    """Body and content type of a multipart form. ``files`` maps a field to (name, data, type)."""
    boundary = uuid.uuid4().hex
    out = bytearray()
    for name, value in fields.items():
        out += f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
        out += value.encode() + b"\r\n"
    for name, (filename, data, ctype) in files.items():
        out += (
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; '
            f'filename="{filename}"\r\nContent-Type: {ctype}\r\n\r\n'
        ).encode()
        out += data + b"\r\n"
    out += f"--{boundary}--\r\n".encode()
    return bytes(out), f"multipart/form-data; boundary={boundary}"


class HttpTransport:
    """``Sam3dTransport`` that posts each call to a SAM 3D server."""

    def __init__(self, config: HttpConfig) -> None:
        self._config = config
        self._key = read_secret(config.api_key_env) if config.api_key_env else None

    def __call__(self, call: Sam3dCall) -> Sam3dAnswer:
        import json
        import time

        files = {
            "image": ("image", call.image, "application/octet-stream"),
            "mask": ("mask.png", call.mask_png, "image/png"),
        }
        if call.pointmap_npy is not None:
            files["pointmap"] = ("pointmap.npy", call.pointmap_npy, "application/octet-stream")
        body, ctype = encode_multipart(
            {"seed": str(call.seed), "options": json.dumps(call.options)}, files
        )
        headers = {"Content-Type": ctype}
        if self._key:
            headers["Authorization"] = f"Bearer {self._key}"
        request = urllib.request.Request(
            f"{self._config.url}/sample", data=body, headers=headers, method="POST"
        )
        for attempt in range(self._config.retries):
            try:
                with urllib.request.urlopen(request, timeout=self._config.timeout_s) as response:
                    glb: bytes = response.read()
                    pose = response.headers.get(POSE_HEADER)
                if not pose:
                    raise ValueError(f"the server answer has no {POSE_HEADER} header")
                return Sam3dAnswer(glb=glb, pose_json=pose)
            except urllib.error.HTTPError as e:
                if e.code not in _RETRY_CODES or attempt == self._config.retries - 1:
                    raise explain_http_error(e) from e
            time.sleep(2.0 * 2**attempt)
        raise AssertionError("unreachable")

    def health(self) -> None:
        """Raises if the server does not answer ``GET /health`` with 200."""
        request = urllib.request.Request(f"{self._config.url}/health")
        try:
            with urllib.request.urlopen(request, timeout=min(self._config.timeout_s, 30)) as resp:
                resp.read()
        except urllib.error.HTTPError as e:
            raise explain_http_error(e) from e
