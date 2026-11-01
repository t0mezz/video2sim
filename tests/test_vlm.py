from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

from video2sim.models.hosted import BASETEN_MODEL, BASETEN_URL
from video2sim.models.vlm import VlmConfig, cache_model, make_vlm_client

CFG: dict[str, Any] = {
    "vlm": {
        "backend": "openai_compatible",
        "openai_compatible": {
            "base_url": "http://127.0.0.1:1/v1/",
            "model": "local-model",
            "api_key_env": None,
            "max_tokens": 64,
        },
    }
}


def test_openai_compatible_client_uses_configured_url_and_model() -> None:
    client = make_vlm_client(CFG, 5.0, 1)
    assert client.model == "local-model"
    assert cache_model(client) == "local-model"
    assert VlmConfig.from_config(CFG).base_url == "http://127.0.0.1:1/v1"


def test_baseten_is_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BASETEN_API_KEY", "k")
    client = make_vlm_client({}, 5.0, 1)
    assert client.model == BASETEN_MODEL
    assert client._url == BASETEN_URL


def test_unknown_backend_and_missing_url_are_rejected() -> None:
    with pytest.raises(ValueError, match="backend"):
        VlmConfig.from_config({"vlm": {"backend": "nope"}})
    with pytest.raises(ValueError, match="base_url"):
        VlmConfig.from_config({"vlm": {"backend": "openai_compatible"}})


def test_client_without_model_attribute_caches_as_baseten() -> None:
    assert cache_model(object()) == BASETEN_MODEL


def test_ask_posts_images_and_reads_the_reply(tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            seen["path"] = self.path
            seen["auth"] = self.headers.get("Authorization")
            seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            data = json.dumps({"choices": [{"message": {"content": "hello"}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args: object) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        cfg = json.loads(json.dumps(CFG))
        cfg["vlm"]["openai_compatible"]["base_url"] = f"http://127.0.0.1:{server.server_port}/v1"
        image = tmp_path / "a.png"
        image.write_bytes(b"\x89PNG")
        assert make_vlm_client(cfg, 5.0, 1).ask("what?", [image]) == "hello"
    finally:
        server.shutdown()
    assert seen["path"] == "/v1/chat/completions"
    assert seen["auth"] is None
    assert seen["body"]["model"] == "local-model"
    assert seen["body"]["max_tokens"] == 64
    parts = seen["body"]["messages"][0]["content"]
    assert parts[0] == {"type": "text", "text": "what?"}
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")
