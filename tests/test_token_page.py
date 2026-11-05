"""The judge token request page (``workers/token_page``) against the real gateway app; offline.

The page's ``HttpGateway`` makes real ``urllib`` calls, answered here by the gateway app in a
TestClient, so minting, the admin key and the gateway's bounds are exercised end to end.
"""

from __future__ import annotations

import email.message
import io
import re
import sqlite3
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from fastapi.testclient import TestClient

import manage
import token_page
from token_page import HttpGateway, Settings, Store, client_ip, hash_code
from v2s_gateway.app import create_app as create_gateway
from v2s_gateway.ledger import Ledger, MemoryKv
from v2s_gateway.policy import POLICY_FILE, Policy
from v2s_gateway.upstreams import HttpUpstreams, Upstreams

GATEWAY = "https://gw.test"
ADMIN_KEY = "admin-key-of-the-page"
CODE = "orange-kettle-river-42"
NOW = datetime(2026, 11, 20, 12, 0, tzinfo=timezone.utc)


class _NoSam3d:
    async def sample(self, request: Any) -> tuple[bytes, str]:
        raise AssertionError("not used")


def gateway_policy(**admin: int) -> Policy:
    data = yaml.safe_load(POLICY_FILE.read_text("utf-8"))
    data["admin"].update(admin)
    return Policy.model_validate(data)


@dataclass
class World:
    page: TestClient
    gateway: TestClient
    store: Store
    settings: Settings
    clock: list[datetime]
    gateway_data: dict[str, Any]
    calls: list[str] = field(default_factory=list)

    def ask(self, **changes: str) -> httpx.Response:
        form = {
            "name": "Ada Judge",
            "email": "ada@example.org",
            "affiliation": "Hackathon",
            "code": CODE,
            "consent": "yes",
        }
        form.update(changes)
        response: httpx.Response = self.page.post("/request", data=form, follow_redirects=False)
        return response


def _message(headers: Any) -> email.message.Message:
    message = email.message.Message()
    for key, value in headers.items():
        message[key] = value
    return message


class _Answer(io.BytesIO):
    def __enter__(self) -> _Answer:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


@pytest.fixture
def make_world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    clients: list[TestClient] = []

    def build(settings_changes: dict[str, Any] | None = None, **admin: int) -> World:
        clock = [NOW]
        data: dict[str, Any] = {}
        http = HttpUpstreams(httpx.AsyncClient(), "bt", "meta")
        gateway_app = create_gateway(
            gateway_policy(**admin),
            Ledger(MemoryKv(data), clock=lambda: clock[0]),
            Upstreams(http, http, _NoSam3d()),
            ADMIN_KEY,
        )
        gateway = TestClient(gateway_app, base_url=GATEWAY)
        world_calls: list[str] = []

        def urlopen(request: urllib.request.Request, timeout: float) -> _Answer:
            parts = urllib.parse.urlsplit(request.full_url)
            assert f"{parts.scheme}://{parts.netloc}" == GATEWAY
            world_calls.append(f"{request.get_method()} {parts.path}")
            data_bytes = request.data if isinstance(request.data, bytes) else None
            response = gateway.request(
                request.get_method(),
                parts.path,
                content=data_bytes,
                headers=dict(request.header_items()),
            )
            if response.status_code >= 400:
                raise urllib.error.HTTPError(
                    request.full_url,
                    response.status_code,
                    response.reason_phrase,
                    _message(response.headers),
                    io.BytesIO(response.content),
                )
            return _Answer(response.content)

        monkeypatch.setattr(urllib.request, "urlopen", urlopen)
        settings = Settings(
            gateway_url=GATEWAY,
            admin_key=ADMIN_KEY,
            invite_hashes=(hash_code(CODE),),
            db_path=tmp_path / f"page{len(clients)}.sqlite3",
            contact="organisers@example.org",
        )
        settings = replace(settings, **(settings_changes or {}))
        store = Store(settings.db_path, clock=lambda: clock[0])
        app = token_page.create_app(settings, store, HttpGateway(GATEWAY, ADMIN_KEY))
        page = TestClient(app, base_url="https://vts.kortec.me")
        clients.extend([gateway, page])
        return World(page, gateway, store, settings, clock, data, world_calls)

    yield build
    for client in clients:
        client.close()


@pytest.fixture
def world(make_world: Any) -> World:
    w: World = make_world()
    return w


def token_in(response: httpx.Response) -> str:
    match = re.search(r'<pre class="token">(v2s_[A-Za-z0-9_-]+)</pre>', response.text)
    assert match, response.text
    return match.group(1)


def test_valid_request_shows_a_working_token_exactly_once(world: World) -> None:
    form = world.page.get("/")
    assert form.status_code == 200 and 'name="code"' in form.text
    response = world.ask()
    assert response.status_code == 303
    link = response.headers["location"]
    assert re.fullmatch(r"/token/[A-Za-z0-9_-]{32}", link)

    shown = world.page.get(link)
    assert shown.status_code == 200
    token = token_in(shown)
    assert "export V2S_JUDGE_TOKEN=" + token in shown.text
    assert f"--gateway {GATEWAY}" in shown.text
    usage = world.gateway.get("/v1/usage", headers={"Authorization": f"Bearer {token}"}).json()
    assert re.fullmatch(r"j-[0-9a-f]{8}", usage["label"])  # the gateway sees a pseudonym only
    assert "ada" not in str(world.gateway_data).lower()

    again = world.page.get(link)
    assert again.status_code == 410 and token not in again.text
    assert "organisers@example.org" in again.text
    [row] = world.store.rows()
    assert row["email"] == "ada@example.org" and row["shown_utc"]
    with sqlite3.connect(world.settings.db_path) as db:
        assert db.execute("SELECT token FROM requests").fetchall() == [(None,)]


def test_wrong_code_mints_nothing_and_is_not_echoed(world: World) -> None:
    response = world.ask(code="guess-1234567", name="<script>alert(1)</script>")
    assert response.status_code == 400
    assert "invite code is not valid" in response.text
    assert "guess-1234567" not in response.text
    assert "<script>alert" not in response.text and "&lt;script&gt;" in response.text
    assert world.calls == [] and world.store.rows() == []


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"consent": ""}, "Please confirm"),
        ({"email": "not-an-email"}, "valid email"),
        ({"name": ""}, "Enter your name"),
        ({"affiliation": "x" * 201}, "longer than 200"),
    ],
)
def test_incomplete_requests_get_the_form_back(
    world: World, changes: dict[str, str], message: str
) -> None:
    response = world.ask(**changes)
    assert response.status_code == 400 and message in response.text
    assert world.calls == []


def test_one_token_per_email(world: World) -> None:
    assert world.ask().status_code == 303
    response = world.ask(email="ADA@example.org")
    assert "already issued for this email" in response.text
    assert "organisers@example.org" in response.text
    assert world.calls == ["POST /admin/tokens"]


def test_each_code_is_good_for_a_limited_number_of_tokens(make_world: Any) -> None:
    w: World = make_world({"max_per_code": 1})
    assert w.ask().status_code == 303
    assert "used up" in w.ask(email="bob@example.org").text


def test_form_posts_are_rate_limited_per_address(make_world: Any) -> None:
    w: World = make_world({"max_attempts_per_hour": 2})
    w.ask(code="wrong-guess-0001")
    w.ask(code="wrong-guess-0002")
    assert "Too many attempts" in w.ask().text  # even the right code, until the hour is over
    w.clock[0] = NOW + timedelta(hours=1, minutes=1)
    assert w.ask().status_code == 303


def test_gateway_token_cap_is_reported_and_the_email_can_retry(make_world: Any) -> None:
    w: World = make_world(max_tokens=1)
    assert w.ask().status_code == 303
    response = w.ask(email="bob@example.org")
    assert "No more tokens can be issued" in response.text
    assert [r["email"] for r in w.store.rows()] == ["ada@example.org"]  # bob's row was dropped


def test_unreachable_gateway_is_reported(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    def down(request: urllib.request.Request, timeout: float) -> Any:
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", down)
    response = world.ask()
    assert "cannot be reached" in response.text
    assert world.store.rows() == []


def test_a_wrong_admin_key_mints_nothing(make_world: Any) -> None:
    w: World = make_world()
    app = token_page.create_app(w.settings, w.store, HttpGateway(GATEWAY, "not-the-admin-key"))
    with TestClient(app, base_url="https://vts.kortec.me") as page:
        form = {"name": "Ada", "email": "ada@example.org", "code": CODE, "consent": "yes"}
        response = page.post("/request", data=form, follow_redirects=False)
    assert "gateway refused" in response.text
    assert w.gateway_data == {}


def test_unopened_tokens_are_wiped_after_a_day(world: World) -> None:
    link = world.ask().headers["location"]
    world.clock[0] = NOW + timedelta(hours=25)
    response = world.page.get(link)
    assert response.status_code == 410
    assert world.page.get("/token/unknown").status_code == 404


def test_links_follow_the_proxy_prefix(make_world: Any) -> None:
    w: World = make_world()
    app = token_page.create_app(w.settings, w.store, HttpGateway(GATEWAY, ADMIN_KEY))
    with TestClient(app, base_url="https://vts.kortec.me", root_path="/judges") as page:
        assert 'action="/judges/request"' in page.get("/").text
        form = {"name": "Ada", "email": "ada@example.org", "code": CODE, "consent": "yes"}
        response = page.post("/request", data=form, follow_redirects=False)
    assert response.headers["location"].startswith("/judges/token/")


def test_pages_send_strict_headers(world: World) -> None:
    headers = world.page.get("/").headers
    assert headers["cache-control"] == "no-store"
    assert headers["referrer-policy"] == "no-referrer"
    assert "frame-ancestors 'none'" in headers["content-security-policy"]


def test_settings_from_env() -> None:
    env = {
        "V2S_GATEWAY_URL": GATEWAY + "/",
        "V2S_GATEWAY_ADMIN_KEY": ADMIN_KEY,
        "TOKEN_PAGE_INVITE_SHA256": f"{hash_code('a' * 12)}, {hash_code(CODE)}",
        "TOKEN_PAGE_TRUST_PROXY": "1",
        "TOKEN_PAGE_SCENES": "2",
    }
    settings = Settings.from_env(env)
    assert settings.gateway_url == GATEWAY and settings.trust_proxy and settings.scenes == 2
    assert settings.invite_hash(f"  {CODE} ") == hash_code(CODE)
    assert settings.invite_hash("nope") is None
    with pytest.raises(RuntimeError, match="V2S_GATEWAY_ADMIN_KEY"):
        Settings.from_env({k: v for k, v in env.items() if k != "V2S_GATEWAY_ADMIN_KEY"})
    with pytest.raises(RuntimeError, match="SHA-256"):
        Settings.from_env({**env, "TOKEN_PAGE_INVITE_SHA256": CODE})


def test_client_ip_trusts_forwarded_for_only_behind_a_proxy() -> None:
    from starlette.requests import Request

    scope = {
        "type": "http",
        "headers": [(b"x-forwarded-for", b"6.6.6.6, 203.0.113.7")],
        "client": ("127.0.0.1", 5000),
    }
    request = Request(scope)
    assert client_ip(request, trust_proxy=True) == "203.0.113.7"
    assert client_ip(request, trust_proxy=False) == "127.0.0.1"


def test_manage_hash_list_and_revoke(
    world: World, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("manage.getpass.getpass", lambda prompt: CODE)
    assert manage.main(["hash-code"]) == 0
    assert capsys.readouterr().out.strip() == hash_code(CODE)
    monkeypatch.setattr("manage.getpass.getpass", lambda prompt: "short")
    assert manage.main(["hash-code"]) == 1

    token = token_in(world.page.get(world.ask().headers["location"]))
    monkeypatch.setenv("TOKEN_PAGE_DB", str(world.settings.db_path))
    monkeypatch.setenv("V2S_GATEWAY_URL", GATEWAY)
    monkeypatch.setenv("V2S_GATEWAY_ADMIN_KEY", ADMIN_KEY)
    monkeypatch.setenv("TOKEN_PAGE_INVITE_SHA256", hash_code(CODE))
    capsys.readouterr()
    assert manage.main(["list"]) == 0
    assert "ada@example.org  Ada Judge  Hackathon  shown" in capsys.readouterr().out
    assert manage.main(["revoke", "ada@example.org"]) == 0
    assert "1 token(s) on the gateway" in capsys.readouterr().out
    usage = world.gateway.get("/v1/usage", headers={"Authorization": f"Bearer {token}"})
    assert usage.status_code == 401
    assert manage.main(["revoke", "nobody@example.org"]) == 1


def test_privacy_is_its_own_page_linked_from_the_form(make_world: Any) -> None:
    w: World = make_world()
    app = token_page.create_app(w.settings, w.store, HttpGateway(GATEWAY, ADMIN_KEY))
    with TestClient(app, base_url="https://vts.kortec.me", root_path="/judges") as page:
        form = page.get("/").text
        privacy = page.get("/privacy")
    assert 'href="/judges/privacy"' in form
    assert "retention terms" not in form
    assert privacy.status_code == 200 and "retention terms" in privacy.text
    assert 'href="/judges/"' in privacy.text


def test_form_and_token_page_show_the_setup_steps(world: World) -> None:
    form = world.page.get("/").text
    assert "export V2S_JUDGE_TOKEN=&lt;your token&gt;" in form
    assert f"uv run video2sim doctor --gateway {GATEWAY}" in form
    shown = world.page.get(world.ask().headers["location"])
    token_html = shown.text
    assert f"export V2S_JUDGE_TOKEN={token_in(shown)}" in token_html
    assert f"uv run video2sim run my_table.mp4 --out scene/ --gateway {GATEWAY}" in token_html
    assert "&lt;your token&gt;" not in token_html
