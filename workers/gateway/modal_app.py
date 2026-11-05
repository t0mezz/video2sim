"""Modal deployment of the judge gateway (``v2s_gateway``).

One container (``max_containers=1``), so the in-process lock and concurrency counts hold for
every request; usage, tokens and the kill switch live in the ``modal.Dict`` named below, so they
survive restarts. The upstream keys come from the Modal secret ``video2sim-gateway``
(``BASETEN_API_KEY``, ``META_MODEL_API_KEY``, and ``V2S_GATEWAY_ADMIN_KEY`` for the token
request page), created by a human. The SAM 3D worker is called
inside the workspace, so no Modal token leaves it.

Commands (each needs a human go-ahead, see workers/gateway/README.md):

    uv run modal deploy workers/gateway/modal_app.py
    uv run modal app stop video2sim-gateway
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import modal

from v2s_gateway.ledger import STATE_DICT

APP_NAME = "video2sim-gateway"
SECRET_NAME = "video2sim-gateway"
POLICY_PATH = "/root/policy.yaml"
# Kept equal to uv.lock by tests/test_gateway.py, so the deployed gateway runs what was tested.
PINS = {
    "fastapi": "0.162.0",
    "starlette": "1.10.0",
    "pydantic": "2.14.1",
    "httpx": "0.28.1",
    "python-multipart": "0.0.33",
    "pyyaml": "6.0.3",
}

app = modal.App(APP_NAME)
state = modal.Dict.from_name(STATE_DICT, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(*(f"{name}=={version}" for name, version in PINS.items()))
    .add_local_file(Path(__file__).parent / "policy.yaml", POLICY_PATH)
    .add_local_python_source("v2s_gateway")
)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name(SECRET_NAME)],
    max_containers=1,
    timeout=1800,  # one tracking answer can take about 10 minutes
    scaledown_window=900,
)
@modal.concurrent(max_inputs=64)
@modal.asgi_app()
def web() -> Any:
    import logging
    import os

    import httpx

    from v2s_gateway.app import create_app
    from v2s_gateway.ledger import Ledger, ModalDictKv
    from v2s_gateway.policy import load_policy
    from v2s_gateway.upstreams import HttpUpstreams, ModalSam3d, Upstreams

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # its lines name upstream URLs only
    policy = load_policy(Path(POLICY_PATH))
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(
            policy.upstream_read_timeout_s, connect=policy.upstream_connect_timeout_s
        )
    )
    http = HttpUpstreams(client, os.environ["BASETEN_API_KEY"], os.environ["META_MODEL_API_KEY"])
    sam3d = ModalSam3d(policy.sam3d.app, policy.sam3d.cls, policy.sam3d.scaledown_window_s)
    upstreams = Upstreams(http, http, sam3d)
    # Optional: enables /admin/tokens for the token request page on the home server.
    admin_key = os.environ.get("V2S_GATEWAY_ADMIN_KEY")
    return create_app(policy, Ledger(ModalDictKv(state)), upstreams, admin_key)
