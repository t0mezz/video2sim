from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

from video2sim.models.hosted import BASETEN_MODEL, BASETEN_URL
from video2sim.models.vlm import VlmConfig, cache_key_part, cache_model, make_vlm_client

CFG: dict[str, Any] = {
    "vlm": {
        "protocol": "openai_chat",
        "base_url": "http://127.0.0.1:1/v1/",
        "model": "local-model",
        "api_key_env": None,
        "max_tokens": 64,
    }
}


def test_openai_chat_client_uses_configured_url_and_model() -> None:
    client = make_vlm_client(CFG, 5.0, 1)
    assert client.model == "local-model"
    assert cache_model(client) == "local-model"
    assert cache_key_part(client) == "local-model"
    assert VlmConfig.from_config(CFG).base_url == "http://127.0.0.1:1/v1"


def test_baseten_is_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BASETEN_API_KEY", "k")
    client = make_vlm_client({}, 5.0, 1)
    assert client.model == BASETEN_MODEL
    assert client._url == BASETEN_URL  # type: ignore[attr-defined]
    assert cache_key_part(client) is None


def test_unknown_protocol_is_rejected() -> None:
    with pytest.raises(ValueError, match="protocol"):
        VlmConfig.from_config({"vlm": {"protocol": "nope"}})
    with pytest.raises(ValueError, match="role"):
        VlmConfig.from_config(CFG, "nobody")


def test_roles_override_the_base_endpoint() -> None:
    cfg = {"vlm": {**CFG["vlm"], "roles": {"mass": {"model": "big", "max_tokens": 9}}}}
    assert VlmConfig.from_config(cfg, "mass").model == "big"
    assert VlmConfig.from_config(cfg, "mass").max_tokens == 9
    assert VlmConfig.from_config(cfg, "router").model == "local-model"
    assert make_vlm_client(cfg, 5.0, 1, "mass").model == "big"


def test_client_without_model_attribute_caches_as_baseten() -> None:
    assert cache_model(object()) == BASETEN_MODEL


def test_python_factory_protocol() -> None:
    cfg = {"vlm": {**CFG["vlm"], "protocol": "python:test_vlm:_factory"}}
    client = make_vlm_client(cfg, 7.0, 2)
    assert client.ask("p", []) == "local-model 7.0 2"


def _factory(settings: VlmConfig, timeout_s: float, retries: int) -> Any:
    class Fake:
        model = settings.model

        def ask(self, prompt: str, images: Any) -> str:
            return f"{settings.model} {timeout_s} {retries}"

    return Fake()


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
        cfg["vlm"]["base_url"] = f"http://127.0.0.1:{server.server_port}/v1"
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


def test_anthropic_messages_protocol(tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            seen["path"] = self.path
            seen["key"] = self.headers.get("x-api-key")
            seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            data = json.dumps({"content": [{"type": "text", "text": "hi"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args: object) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        cfg = {
            "vlm": {
                "protocol": "anthropic_messages",
                "base_url": f"http://127.0.0.1:{server.server_port}",
                "model": "m",
                "api_key_env": "TEST_ANTHROPIC_KEY",
            }
        }
        image = tmp_path / "a.png"
        image.write_bytes(b"\x89PNG")
        import os

        os.environ["TEST_ANTHROPIC_KEY"] = "secret"
        try:
            assert make_vlm_client(cfg, 5.0, 1).ask("q", [image]) == "hi"
        finally:
            del os.environ["TEST_ANTHROPIC_KEY"]
    finally:
        server.shutdown()
    assert seen["path"] == "/v1/messages"
    assert seen["key"] == "secret"
    assert seen["body"]["max_tokens"] == 4096
    assert seen["body"]["messages"][0]["content"][0]["source"]["media_type"] == "image/png"
