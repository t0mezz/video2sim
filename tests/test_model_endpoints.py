"""Pluggable model endpoints: shape transports and backends, tracking, doctor, shared config."""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest
import yaml

from video2sim.doctor import run_doctor
from video2sim.models.sam3d import (
    ModalTransport,
    Sam3dAnswer,
    Sam3dCall,
    make_transport,
    start_warm_up,
)
from video2sim.models.sam3d_http import HttpConfig, HttpTransport, encode_multipart
from video2sim.runner import Runner
from video2sim.stages.shape import default_model
from video2sim.stages.tracking import MetaVideoSamClient, make_tracking_client

CONFIG = Path(__file__).parent.parent / "configs" / "default.yaml"


def _cfg() -> dict[str, Any]:
    cfg: dict[str, Any] = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    return cfg


def test_default_transport_is_modal_and_warm_is_optional() -> None:
    transport = make_transport(_cfg())
    assert isinstance(transport, ModalTransport)
    start_warm_up(object(), 2, 30)  # no warm method: nothing happens


def test_unknown_transport_and_backend_are_rejected(tmp_path: Path) -> None:
    cfg = _cfg()
    cfg["shape"]["transport"] = "carrier-pigeon"
    with pytest.raises(ValueError, match="transport"):
        make_transport(cfg)
    cfg = _cfg()
    cfg["shape"]["backend"] = "nope"
    with pytest.raises(ValueError, match="no adapter"):
        default_model(cfg, tmp_path)


def test_python_shape_backend_and_transport() -> None:
    cfg = _cfg()
    cfg["shape"]["backend"] = "python:test_model_endpoints:_shape_factory"
    assert default_model(cfg, Path("."))._marker == "custom"  # type: ignore[attr-defined]
    cfg["shape"]["transport"] = "python:test_model_endpoints:_transport_factory"
    assert make_transport(cfg)(_call()).pose_json == "{}"


def _shape_factory(cfg: dict[str, Any], work: Path) -> Any:
    class Custom:
        name = "custom"
        _marker = "custom"

    return Custom()


def _transport_factory(cfg: dict[str, Any]) -> Any:
    return lambda call: Sam3dAnswer(b"glb", "{}")


def _call() -> Sam3dCall:
    return Sam3dCall(b"img", b"mask", 7, None, {"checkpoint": "m"})


def test_http_transport_posts_multipart_and_reads_glb_and_pose() -> None:
    seen: dict[str, Any] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            seen["path"] = self.path
            seen["auth"] = self.headers.get("Authorization")
            seen["type"] = self.headers["Content-Type"]
            seen["body"] = self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("X-Pose-Json", '{"checkpoint": "m@1"}')
            self.send_header("Content-Length", "3")
            self.end_headers()
            self.wfile.write(b"glb")

        def do_GET(self) -> None:
            seen["health"] = self.path
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *args: object) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        cfg = _cfg()
        cfg["shape"]["transport"] = "http"
        cfg["shape"]["http"]["url"] = f"http://127.0.0.1:{server.server_port}/"
        transport = make_transport(cfg)
        assert isinstance(transport, HttpTransport)
        answer = transport(_call())
        transport.health()
    finally:
        server.shutdown()
    assert answer == Sam3dAnswer(b"glb", '{"checkpoint": "m@1"}')
    assert seen["path"] == "/sample" and seen["health"] == "/health" and seen["auth"] is None
    assert seen["type"].startswith("multipart/form-data; boundary=")
    for part in (b'name="seed"', b'name="options"', b'name="image"', b'name="mask"', b"img"):
        assert part in seen["body"]
    assert b'name="pointmap"' not in seen["body"]


def test_http_config_needs_a_url() -> None:
    with pytest.raises(ValueError, match="url"):
        HttpConfig.from_config(_cfg())


def test_multipart_encoding_round_trips_fields_and_files() -> None:
    body, ctype = encode_multipart({"a": "1"}, {"f": ("n.bin", b"\x00\x01", "application/x")})
    boundary = ctype.split("boundary=")[1].encode()
    assert body.count(b"--" + boundary) == 3
    assert b"\x00\x01" in body


def test_tracking_endpoint_from_config() -> None:
    cfg = _cfg()
    cfg["segmentation"]["tracking"]["endpoint"].update(
        url="http://gpu:9000/v1/responses", model="my-sam", api_key_env=None
    )
    client = make_tracking_client(cfg)
    assert isinstance(client, MetaVideoSamClient)
    assert (client._url, client._model, client._key) == (
        "http://gpu:9000/v1/responses",
        "my-sam",
        None,
    )
    cfg["segmentation"]["tracking"]["endpoint"]["protocol"] = "grpc"
    with pytest.raises(ValueError, match="protocol"):
        make_tracking_client(cfg)


def test_vlm_section_reaches_the_stages_that_share_it_and_enters_their_cache_key(
    tmp_path: Path,
) -> None:
    from video2sim.stages.physics_stage import PhysicsStage
    from video2sim.stages.segmentation_stage import SegmentationStage
    from video2sim.stages.shape import ShapeStage

    runner = Runner(tmp_path, [], {"vlm": {"model": "a"}}, tmp_path)
    assert runner._shared(SegmentationStage()) == {"vlm": {"model": "a"}, "capture": {}}
    assert runner._shared(PhysicsStage(tmp_path / "t.yaml")) == {"vlm": {"model": "a"}}
    assert runner._shared(ShapeStage()) == {}


def test_doctor_offline_passes_on_the_default_config_and_reports_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The default config names keys that live in .env locally and are absent in CI.
    for name in ("BASETEN_API_KEY", "META_MODEL_API_KEY"):
        monkeypatch.setenv(name, "test-key")
    assert all(c.ok for c in run_doctor(_cfg(), offline=True))
    cfg = _cfg()
    cfg["vlm"]["protocol"] = "nope"
    cfg["shape"]["transport"] = "http"  # no url
    failed = {c.name for c in run_doctor(cfg, offline=True) if not c.ok}
    assert {"vlm.router", "vlm.material", "vlm.mass", "shape"} <= failed
