"""``--gateway``: the config overlay, the CLI flags and the gateway's refusal messages."""

from __future__ import annotations

import copy
import email.message
import io
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest
import yaml

from video2sim import cli
from video2sim.doctor import run_doctor
from video2sim.gateway_profile import (
    JUDGE_TOKEN_ENV,
    apply_gateway,
    describe_usage,
    gateway_base,
    gateway_usage,
)
from video2sim.models.hosted import GATEWAY_HEADER, explain_http_error, post_json
from video2sim.models.vlm import VlmConfig
from video2sim.stages.tracking import TrackingEndpoint

ROOT = Path(__file__).resolve().parents[1]
URL = "https://judges.example.test"


def _cfg() -> dict[str, Any]:
    cfg: dict[str, Any] = yaml.safe_load((ROOT / "configs/default.yaml").read_text("utf-8"))
    return cfg


def test_overlay_points_all_three_endpoints_at_the_gateway_and_keeps_models() -> None:
    cfg = _cfg()
    before = copy.deepcopy(cfg)
    out = apply_gateway(cfg, URL + "/")
    assert cfg == before  # the input is not modified
    for role in ("router", "material", "mass", "verify"):
        vlm = VlmConfig.from_config(out, role)
        assert (vlm.protocol, vlm.base_url) == ("openai_chat", f"{URL}/v1")
        assert vlm.api_key_env == JUDGE_TOKEN_ENV
        assert vlm.model == VlmConfig.from_config(cfg, role).model
    endpoint = TrackingEndpoint.from_config(out)
    assert (endpoint.url, endpoint.api_key_env) == (f"{URL}/v1/responses", JUDGE_TOKEN_ENV)
    assert endpoint.model == TrackingEndpoint.from_config(cfg).model
    assert out["shape"]["transport"] == "http"
    assert out["shape"]["http"]["url"] == f"{URL}/sam3d"
    assert out["shape"]["http"]["api_key_env"] == JUDGE_TOKEN_ENV
    assert out["shape"]["http"]["timeout_s"] == cfg["shape"]["http"]["timeout_s"]
    assert out["shape"]["sam3d"] == cfg["shape"]["sam3d"]  # same options, same cached draws


def test_role_overrides_that_name_an_endpoint_are_redirected_too() -> None:
    cfg = _cfg()
    cfg["vlm"]["roles"] = {
        "mass": {"base_url": "https://elsewhere.test/v1", "api_key_env": "OTHER_KEY"},
        "router": {"model": "meta-models/muse-spark-open-1026"},
    }
    out = apply_gateway(cfg, URL)
    mass = VlmConfig.from_config(out, "mass")
    assert (mass.base_url, mass.api_key_env) == (f"{URL}/v1", JUDGE_TOKEN_ENV)
    assert out["vlm"]["roles"]["router"] == {"model": "meta-models/muse-spark-open-1026"}


@pytest.mark.parametrize(
    ("change", "fragment"),
    [
        (lambda c: c["vlm"].update(extra_body={"top_p": 0.5}), "extra_body"),
        (lambda c: c["vlm"].update(roles={"mass": {"extra_body": {"x": 1}}}), "roles.mass"),
        (lambda c: c["shape"].update(backend="python:pkg.mod:make"), "sam3d"),
    ],
)
def test_configs_the_gateway_cannot_serve_are_refused(change: Any, fragment: str) -> None:
    cfg = _cfg()
    change(cfg)
    with pytest.raises(ValueError, match=fragment):
        apply_gateway(cfg, URL)


@pytest.mark.parametrize(
    ("url", "ok"),
    [
        ("https://gw.example.test", True),
        ("https://gw.example.test/base/", True),
        ("http://localhost:8000", True),
        ("http://127.0.0.1:8000", True),
        ("http://gw.example.test", False),
        ("https://gw.example.test/?token=x", False),
        ("gw.example.test", False),
    ],
)
def test_gateway_url_must_be_https_unless_local(url: str, ok: bool) -> None:
    if ok:
        assert not gateway_base(url).endswith("/")
    else:
        with pytest.raises(ValueError):
            gateway_base(url)


def _refusal(code: int, body: dict[str, Any], gateway: bool) -> urllib.error.HTTPError:
    headers = email.message.Message()
    headers["Content-Type"] = "application/json"
    if gateway:
        headers[GATEWAY_HEADER] = "1"
    data = json.dumps(body).encode()
    return urllib.error.HTTPError("https://x.test", code, "Forbidden", headers, io.BytesIO(data))


def test_gateway_refusals_carry_their_message_and_provider_errors_do_not() -> None:
    message = "This judge token has used its vlm quota of 240 calls."
    explained = explain_http_error(_refusal(403, {"message": message}, gateway=True))
    assert explained.code == 403
    assert str(explained) == f"HTTP Error 403: {message}"
    provider = _refusal(401, {"error": "key sk-123 is wrong"}, gateway=False)
    assert explain_http_error(provider) is provider
    assert "sk-123" not in str(provider)


def test_post_json_raises_the_gateway_message_without_retrying_403(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def urlopen(request: urllib.request.Request, timeout: float) -> Any:
        calls.append(request.full_url)
        raise _refusal(403, {"message": "quota used up; try video2sim replay"}, gateway=True)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    with pytest.raises(urllib.error.HTTPError, match="try video2sim replay"):
        post_json("https://gw.test/v1/chat/completions", "tok", {}, retries=4)
    assert len(calls) == 1


def _usage_answer(body: dict[str, Any]) -> Any:
    class Answer(io.BytesIO):
        def __enter__(self) -> Answer:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    return Answer(json.dumps(body).encode())


USAGE = {
    "label": "judge-a",
    "expires_utc": "2026-11-27T12:00:00+00:00",
    "switched_off": False,
    "routes": {"vlm": {"used": 4, "quota": 240}, "tracking": {"used": 0, "quota": 30}},
}


def test_usage_sends_the_token_and_reads_quotas(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[urllib.request.Request] = []

    def urlopen(request: urllib.request.Request, timeout: float) -> Any:
        seen.append(request)
        return _usage_answer(USAGE)

    monkeypatch.setenv(JUDGE_TOKEN_ENV, "v2s_tok")
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    usage = gateway_usage(URL)
    assert seen[0].full_url == f"{URL}/v1/usage"
    assert seen[0].get_header("Authorization") == "Bearer v2s_tok"
    assert describe_usage(usage) == (
        "token judge-a, expires 2026-11-27T12:00:00+00:00: vlm 4/240, tracking 0/30"
    )


def test_doctor_with_gateway_offline_needs_only_the_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("BASETEN_API_KEY", "META_MODEL_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(ROOT / "tests")  # no .env here
    monkeypatch.setenv(JUDGE_TOKEN_ENV, "v2s_tok")
    checks = run_doctor(apply_gateway(_cfg(), URL), offline=True, gateway=URL)
    assert checks[0].name == "gateway"
    assert all(c.ok for c in checks), [c for c in checks if not c.ok]
    monkeypatch.delenv(JUDGE_TOKEN_ENV)
    failed = {c.name for c in run_doctor(apply_gateway(_cfg(), URL), True, URL) if not c.ok}
    assert {"gateway", "vlm.router", "tracking"} <= failed


def test_cli_doctor_gateway_prints_usage(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(JUDGE_TOKEN_ENV, "v2s_tok")
    monkeypatch.setattr(urllib.request, "urlopen", lambda request, timeout: _usage_answer(USAGE))
    monkeypatch.setattr("video2sim.doctor.check_vlm", lambda cfg, offline: [])
    monkeypatch.setattr("video2sim.doctor.check_shape", lambda cfg, offline: [])
    config = str(ROOT / "configs/default.yaml")
    assert cli.main(["doctor", "--config", config, "--gateway", URL]) == 0
    assert "ok   gateway: token judge-a" in capsys.readouterr().out


def test_cli_run_refuses_a_plain_http_gateway(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    monkeypatch.setattr("video2sim.pipeline.build_runner", lambda *a, **k: pytest.fail("ran"))
    args = ["run", "v.mp4", "--out", str(tmp_path / "o"), "--gateway", "http://gw.example.test"]
    assert cli.main([*args, "--config", str(ROOT / "configs/default.yaml")]) == 2
    assert "https://" in capsys.readouterr().err
