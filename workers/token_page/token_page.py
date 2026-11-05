"""Judge token request page for the home server (D-042).

A judge enters a name, an email, an optional affiliation and the invite code from the judging
instructions. A correct code mints a gateway token at once through ``POST /admin/tokens``, and
the judge is sent to a private link that shows the token exactly once. The page holds the
gateway's admin key and nothing else: no provider key, no Modal token.

What it stores, in one SQLite file: the judge's details, the pseudonymous label their token has
on the gateway, and the token itself only until it is shown. Labels are ``j-`` plus random hex,
so gateway logs never carry a name or an email.

Run it behind the site's reverse proxy (TLS there), for example under ``/judges``:

    uvicorn --factory token_page:app_from_env --host 127.0.0.1 --port 8090 --root-path /judges

Settings come from the environment (see ``Settings.from_env`` and ``README.md``).
"""

from __future__ import annotations

import hashlib
import hmac
import html
import json
import os
import re
import secrets
import sqlite3
import threading
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol

import anyio
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response

EMAIL = re.compile(r"[^@\s]{1,64}@[^@\s]{1,190}\.[^@\s.]{2,}")
UNSHOWN_TOKEN_HOURS = 24  # a token nobody looked at within this time is wiped from the file
ATTEMPT_WINDOW = timedelta(hours=1)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def hash_code(code: str) -> str:
    """How an invite code is configured: its SHA-256, so the code itself is not in the env."""
    return hashlib.sha256(code.strip().encode()).hexdigest()


@dataclass(frozen=True)
class Settings:
    gateway_url: str
    admin_key: str
    invite_hashes: tuple[str, ...]
    db_path: Path
    days: float = 21.0
    scenes: int | None = None
    max_per_code: int = 25
    max_attempts_per_hour: int = 10
    trust_proxy: bool = False
    contact: str | None = None

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> Settings:
        """``V2S_GATEWAY_URL``, ``V2S_GATEWAY_ADMIN_KEY`` and ``TOKEN_PAGE_INVITE_SHA256`` (comma
        separated) are required; ``TOKEN_PAGE_DB``, ``TOKEN_PAGE_DAYS``, ``TOKEN_PAGE_SCENES``,
        ``TOKEN_PAGE_MAX_PER_CODE``, ``TOKEN_PAGE_MAX_ATTEMPTS_PER_HOUR``,
        ``TOKEN_PAGE_TRUST_PROXY`` (1 behind a reverse proxy) and ``TOKEN_PAGE_CONTACT`` are not."""
        missing = [
            name
            for name in ("V2S_GATEWAY_URL", "V2S_GATEWAY_ADMIN_KEY", "TOKEN_PAGE_INVITE_SHA256")
            if not env.get(name)
        ]
        if missing:
            raise RuntimeError(f"set {', '.join(missing)}")
        hashes = tuple(h.strip().lower() for h in env["TOKEN_PAGE_INVITE_SHA256"].split(","))
        if not all(re.fullmatch(r"[0-9a-f]{64}", h) for h in hashes):
            raise RuntimeError("TOKEN_PAGE_INVITE_SHA256 holds SHA-256 hex digests (manage.py)")
        scenes = env.get("TOKEN_PAGE_SCENES")
        return cls(
            gateway_url=env["V2S_GATEWAY_URL"].rstrip("/"),
            admin_key=env["V2S_GATEWAY_ADMIN_KEY"],
            invite_hashes=hashes,
            db_path=Path(env.get("TOKEN_PAGE_DB", "token_page.sqlite3")),
            days=float(env.get("TOKEN_PAGE_DAYS", "21")),
            scenes=int(scenes) if scenes else None,
            max_per_code=int(env.get("TOKEN_PAGE_MAX_PER_CODE", "25")),
            max_attempts_per_hour=int(env.get("TOKEN_PAGE_MAX_ATTEMPTS_PER_HOUR", "10")),
            trust_proxy=env.get("TOKEN_PAGE_TRUST_PROXY") == "1",
            contact=env.get("TOKEN_PAGE_CONTACT") or None,
        )

    def invite_hash(self, code: str) -> str | None:
        """The configured hash ``code`` matches, compared in constant time, or None."""
        given = hash_code(code)
        match = None
        for expected in self.invite_hashes:
            if hmac.compare_digest(given, expected):
                match = expected
        return match


class GatewayRefused(Exception):
    """The gateway did not mint. The message is ours, safe to show."""


class Gateway(Protocol):
    def mint(self, label: str, days: float, scenes: int | None) -> dict[str, Any]: ...

    def revoke(self, label: str) -> int: ...


class HttpGateway:
    """``/admin/tokens`` of the judge gateway, with the admin key."""

    def __init__(self, url: str, admin_key: str, timeout_s: float = 30.0) -> None:
        self._url, self._key, self._timeout_s = url.rstrip("/"), admin_key, timeout_s

    def _call(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        request = urllib.request.Request(
            f"{self._url}{path}",
            data=None if body is None else json.dumps(body).encode(),
            method=method,
            headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self._timeout_s) as response:
            return json.loads(response.read())

    def mint(self, label: str, days: float, scenes: int | None) -> dict[str, Any]:
        try:
            out: dict[str, Any] = self._call(
                "POST", "/admin/tokens", {"label": label, "days": days, "scenes": scenes}
            )
        except urllib.error.HTTPError as error:
            if error.code == 403:
                raise GatewayRefused("No more tokens can be issued right now.") from None
            raise GatewayRefused("The gateway refused the request.") from None
        except (OSError, ValueError):
            raise GatewayRefused("The gateway cannot be reached right now.") from None
        return out

    def revoke(self, label: str) -> int:
        try:
            return int(self._call("DELETE", f"/admin/tokens/{label}")["revoked"])
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return 0
            raise


SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id TEXT PRIMARY KEY,           -- the private link
    created_utc TEXT NOT NULL,
    name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,    -- lower case; one token per email
    affiliation TEXT NOT NULL,
    code_hash TEXT NOT NULL,       -- which invite code was used
    label TEXT NOT NULL UNIQUE,    -- the token's pseudonym on the gateway
    expires_utc TEXT,
    token TEXT,                    -- only until shown
    shown_utc TEXT,
    revoked_utc TEXT
);
CREATE TABLE IF NOT EXISTS attempts (ip TEXT NOT NULL, at_utc TEXT NOT NULL);
"""


@dataclass(frozen=True)
class Shown:
    token: str | None  # None when it was shown before
    expires_utc: str | None
    label: str


class Store:
    """The SQLite file. Every call opens its own connection, so worker threads can share it."""

    def __init__(self, path: Path, clock: Callable[[], datetime] = utc_now) -> None:
        self.path, self.clock = path, clock
        with self._db() as db:
            db.executescript(SCHEMA)

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=10)
        try:
            with db:
                yield db
        finally:
            db.close()

    def note_attempt(self, ip: str, limit: int) -> bool:
        """Count one form post from ``ip``; False when it is over ``limit`` in the last hour."""
        now = self.clock()
        with self._db() as db:
            db.execute(
                "DELETE FROM attempts WHERE at_utc < ?", ((now - ATTEMPT_WINDOW).isoformat(),)
            )
            (count,) = db.execute("SELECT COUNT(*) FROM attempts WHERE ip = ?", (ip,)).fetchone()
            db.execute("INSERT INTO attempts VALUES (?, ?)", (ip, now.isoformat()))
        return int(count) < limit

    def issued_for_code(self, code_hash: str) -> int:
        with self._db() as db:
            (count,) = db.execute(
                "SELECT COUNT(*) FROM requests WHERE code_hash = ?", (code_hash,)
            ).fetchone()
        return int(count)

    def reserve(self, name: str, email: str, affiliation: str, code_hash: str) -> tuple[str, str]:
        """A new request row; (id, label). Raises ``sqlite3.IntegrityError`` for a known email."""
        request_id, label = secrets.token_urlsafe(24), f"j-{secrets.token_hex(4)}"
        with self._db() as db:
            db.execute(
                "INSERT INTO requests (id, created_utc, name, email, affiliation, code_hash, label)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (request_id, self.clock().isoformat(), name, email, affiliation, code_hash, label),
            )
        return request_id, label

    def fill(self, request_id: str, token: str, expires_utc: str) -> None:
        with self._db() as db:
            db.execute(
                "UPDATE requests SET token = ?, expires_utc = ? WHERE id = ?",
                (token, expires_utc, request_id),
            )

    def drop(self, request_id: str) -> None:
        with self._db() as db:
            db.execute("DELETE FROM requests WHERE id = ?", (request_id,))

    def take(self, request_id: str) -> Shown | None:
        """The token of ``request_id``, wiped from the file as it is read. None if unknown."""
        now = self.clock()
        stale = (now - timedelta(hours=UNSHOWN_TOKEN_HOURS)).isoformat()
        with self._db() as db:
            db.execute(
                "UPDATE requests SET token = NULL WHERE token IS NOT NULL AND created_utc < ?",
                (stale,),
            )
            row = db.execute(
                "SELECT token, expires_utc, label FROM requests WHERE id = ?", (request_id,)
            ).fetchone()
            if row is None:
                return None
            if row[0] is not None:
                db.execute(
                    "UPDATE requests SET token = NULL, shown_utc = ? WHERE id = ?",
                    (now.isoformat(), request_id),
                )
        return Shown(row[0], row[1], row[2])

    def rows(self) -> list[dict[str, Any]]:
        with self._db() as db:
            db.row_factory = sqlite3.Row
            found = db.execute(
                "SELECT created_utc, name, email, affiliation, label, expires_utc, shown_utc,"
                " revoked_utc FROM requests ORDER BY created_utc"
            ).fetchall()
        return [dict(row) for row in found]

    def mark_revoked(self, email: str) -> str | None:
        """The label of ``email``'s token, marked revoked here; None if there is none."""
        with self._db() as db:
            row = db.execute(
                "SELECT label FROM requests WHERE email = ?", (email.lower(),)
            ).fetchone()
            if row is None:
                return None
            db.execute(
                "UPDATE requests SET revoked_utc = ?, token = NULL WHERE email = ?",
                (self.clock().isoformat(), email.lower()),
            )
        return str(row[0])


class Rejected(Exception):
    """A request that gets the form back with this message."""


class Service:
    """The request flow, without HTTP: check, reserve the email, mint, store until shown."""

    def __init__(self, settings: Settings, store: Store, gateway: Gateway) -> None:
        self.settings, self.store, self.gateway = settings, store, gateway
        self._lock = threading.Lock()  # one issue at a time, so the per-code count holds

    def request_token(self, form: Mapping[str, str], ip: str) -> str:
        """The private link id for a valid request; raises ``Rejected`` otherwise."""
        s = self.settings
        if not self.store.note_attempt(ip, s.max_attempts_per_hour):
            raise Rejected("Too many attempts from your network. Try again in an hour.")
        name = form.get("name", "").strip()
        email = form.get("email", "").strip().lower()
        affiliation = form.get("affiliation", "").strip()
        if not 1 <= len(name) <= 100:
            raise Rejected("Enter your name (up to 100 characters).")
        if len(email) > 200 or not EMAIL.fullmatch(email):
            raise Rejected("Enter a valid email address.")
        if len(affiliation) > 200:
            raise Rejected("The affiliation is longer than 200 characters.")
        if form.get("consent") != "yes":
            raise Rejected("Please confirm that you have read what is sent to the model providers.")
        code_hash = s.invite_hash(form.get("code", "")[:200])
        if code_hash is None:
            raise Rejected("That invite code is not valid.")
        with self._lock:
            if self.store.issued_for_code(code_hash) >= s.max_per_code:
                raise Rejected("This invite code has been used up.")
            try:
                request_id, label = self.store.reserve(name, email, affiliation, code_hash)
            except sqlite3.IntegrityError:
                raise Rejected(
                    "A token was already issued for this email. If you lost it, "
                    + (f"write to {s.contact}." if s.contact else "contact the organisers.")
                ) from None
        try:
            minted = self.gateway.mint(label, s.days, s.scenes)
        except GatewayRefused as refused:
            self.store.drop(request_id)
            raise Rejected(str(refused)) from None
        except BaseException:
            self.store.drop(request_id)
            raise
        self.store.fill(request_id, str(minted["token"]), str(minted["expires_utc"]))
        return request_id


CSS = """
:root { --bg: #fafaf8; --fg: #1d1d1b; --muted: #5f5f5a; --card: #fff; --line: #d9d9d3;
  --accent: #2456c6; --on-accent: #fff; --bad: #a3261b; --code: #f0f0ec; }
@media (prefers-color-scheme: dark) { :root { --bg: #151514; --fg: #ececea; --muted: #a3a39d;
  --card: #1f1f1d; --line: #3a3a37; --accent: #7aa2ff; --on-accent: #0d1424; --bad: #ff8a7a;
  --code: #2a2a27; } }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg);
  font: 16px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 640px; margin: 0 auto; padding: 32px 16px 64px; }
h1 { font-size: 1.6rem; margin: 0 0 4px; } h2 { font-size: 1.1rem; margin: 28px 0 8px; }
p, li { color: var(--fg); } .muted { color: var(--muted); font-size: .92rem; }
form, .card { background: var(--card); border: 1px solid var(--line); border-radius: 10px;
  padding: 20px; margin-top: 20px; }
label { display: block; font-weight: 600; margin: 14px 0 4px; } label:first-child { margin-top: 0; }
input[type=text], input[type=email], input[type=password] { width: 100%; padding: 9px 10px;
  border: 1px solid var(--line); border-radius: 6px; background: var(--bg); color: var(--fg);
  font: inherit; }
.check { display: flex; gap: 8px; align-items: flex-start; font-weight: 400; margin-top: 16px; }
.check input { margin-top: 5px; }
button { margin-top: 18px; padding: 10px 18px; border: 0; border-radius: 6px;
  background: var(--accent); color: var(--on-accent); font: inherit; font-weight: 600;
  cursor: pointer; }
.error { color: var(--bad); font-weight: 600; margin: 0 0 8px; }
code, pre { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: .9rem; }
pre { background: var(--code); padding: 12px; border-radius: 6px; overflow-x: auto;
  white-space: pre-wrap; word-break: break-all; }
.token { user-select: all; font-size: 1rem; }
a { color: var(--accent); }
ol.steps { padding-left: 1.3rem; } ol.steps li { margin: 0 0 12px; } ol.steps pre { margin: 6px 0; }
"""

HEADERS = {
    "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; "
    "base-uri 'none'; frame-ancestors 'none'",
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
    "Cache-Control": "no-store",
}


def page(title: str, body: str, status: int = 200) -> HTMLResponse:
    document = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f'<meta name="robots" content="noindex"><title>{html.escape(title)}</title>'
        f"<style>{CSS}</style></head><body><main>{body}</main></body></html>"
    )
    return HTMLResponse(document, status_code=status, headers=HEADERS)


REPO = "https://github.com/t0mezz/video2sim"
UV_INSTALL = "https://docs.astral.sh/uv/getting-started/installation/"


def privacy_page(root: str) -> HTMLResponse:
    body = f"""
<p class="muted"><a href="{html.escape(root)}/">&larr; Back</a></p>
<h1>What happens to your video</h1>
<ul>
<li>Frames, crops and a short clip of your video go through the gateway to Baseten, the Meta
Model API and Modal. The gateway does not store them.</li>
<li>The providers' retention terms are not checked yet, so please film nothing private: no
people, screens or documents.</li>
<li>Your name, email and affiliation stay on the organiser's server and are deleted after
judging.</li>
</ul>
"""
    return page("What happens to your video · video2sim", body)


def setup_guide(settings: Settings, token: str | None) -> str:
    """The few commands a judge runs; ``token`` fills them in on the token page."""
    gateway = html.escape(settings.gateway_url)
    value = html.escape(token) if token else "&lt;your token&gt;"
    return f"""
<h2>How to use it</h2>
<ol class="steps">
<li>Install <a href="{UV_INSTALL}">uv</a>, then get the code:
<pre>git clone {REPO}
cd video2sim
uv sync</pre></li>
<li>Set your token (PowerShell: <code>$env:V2S_JUDGE_TOKEN="..."</code>):
<pre>export V2S_JUDGE_TOKEN={value}</pre></li>
<li>Check the connection; it also shows your quota:
<pre>uv run video2sim doctor --gateway {gateway}</pre></li>
<li>Film your table with <a href="{REPO}/blob/main/markers_a4.pdf">markers_a4.pdf</a> printed
at actual size and in view (<a href="{REPO}/blob/main/docs/capture_protocol.md">how to film</a>),
then:
<pre>uv run video2sim run my_table.mp4 --out scene/ --gateway {gateway}</pre></li>
</ol>
<p class="muted">Quota used up? <code>uv run video2sim replay</code> shows recorded runs
offline.</p>
"""


def form_page(
    settings: Settings, root: str, values: Mapping[str, str], error: str | None = None
) -> HTMLResponse:
    def value(name: str) -> str:
        return html.escape(values.get(name, ""), quote=True)

    problem = f'<p class="error" role="alert">{html.escape(error)}</p>' if error else ""
    checked = " checked" if values.get("consent") == "yes" else ""
    privacy = f"{html.escape(root)}/privacy"
    body = f"""
<h1>Judge access to video2sim</h1>
<p class="muted">Get a personal token to run the pipeline on your own video. It has a small quota
and expires after {settings.days:g} days.</p>
<form method="post" action="{html.escape(root)}/request" autocomplete="on">
{problem}
<label for="name">Name</label>
<input type="text" id="name" name="name" maxlength="100" required value="{value("name")}">
<label for="email">Email</label>
<input type="email" id="email" name="email" maxlength="200" required value="{value("email")}">
<label for="affiliation">Affiliation <span class="muted">(optional)</span></label>
<input type="text" id="affiliation" name="affiliation" maxlength="200"
 value="{value("affiliation")}">
<label for="code">Invite code</label>
<input type="password" id="code" name="code" maxlength="200" required autocomplete="off">
<label class="check"><input type="checkbox" name="consent" value="yes" required{checked}>
<span>I have read <a href="{privacy}" target="_blank" rel="noopener">what happens to my
video</a>.</span></label>
<button type="submit">Get my token</button>
</form>
{setup_guide(settings, None)}
"""
    return page("Judge access · video2sim", body, 400 if error else 200)


def token_page(settings: Settings, shown: Shown) -> HTMLResponse:
    if shown.token is None:
        help_line = f" write to {html.escape(settings.contact)}" if settings.contact else ""
        body = f"""
<h1>This token was already shown</h1>
<p>For safety a token is shown only once. If you did not save it, ask for a new one:{help_line}
(your request is <code>{html.escape(shown.label)}</code>).</p>
"""
        return page("Token already shown · video2sim", body, 410)
    token = html.escape(shown.token)
    expires = html.escape((shown.expires_utc or "")[:16].replace("T", " "))
    body = f"""
<h1>Your judge token</h1>
<p><strong>Copy it now.</strong> This page shows it only once; reloading will not show it again.</p>
<div class="card"><pre class="token">{token}</pre>
<p class="muted">Valid until {expires} UTC. Keep it to yourself: whoever has it uses your quota.</p>
</div>
{setup_guide(settings, shown.token)}
"""
    return page("Your token · video2sim", body)


def client_ip(request: Request, trust_proxy: bool) -> str:
    """The caller's address. Behind a trusted reverse proxy, the last X-Forwarded-For entry,
    which that proxy appended."""
    if trust_proxy:
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded.strip():
            return forwarded.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"


def create_app(settings: Settings, store: Store, gateway: Gateway) -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    service = Service(settings, store, gateway)

    def root(request: Request) -> str:
        """The path prefix the proxy serves this app under (``--root-path``)."""
        return str(request.scope.get("root_path", "")).rstrip("/")

    @app.get("/")
    async def index(request: Request) -> Response:
        return form_page(settings, root(request), {})

    @app.get("/privacy")
    async def privacy(request: Request) -> Response:
        return privacy_page(root(request))

    @app.post("/request")
    async def request_token(request: Request) -> Response:
        try:
            async with request.form(max_files=0, max_fields=8, max_part_size=4096) as form:
                values = {k: v for k, v in form.items() if isinstance(v, str)}
        except Exception:
            return form_page(settings, root(request), {}, "The form could not be read.")
        ip = client_ip(request, settings.trust_proxy)
        try:
            request_id = await anyio.to_thread.run_sync(service.request_token, values, ip)
        except Rejected as rejected:
            safe = {k: v for k, v in values.items() if k != "code"}
            return form_page(settings, root(request), safe, str(rejected))
        target = f"{root(request)}/token/{request_id}"
        return RedirectResponse(target, status_code=303, headers=HEADERS)

    @app.get("/token/{request_id}")
    async def show(request_id: str) -> Response:
        shown = await anyio.to_thread.run_sync(store.take, request_id)
        if shown is None:
            return page("Not found · video2sim", "<h1>Not found</h1>", 404)
        return token_page(settings, shown)

    return app


def app_from_env() -> FastAPI:
    """For ``uvicorn --factory token_page:app_from_env``."""
    settings = Settings.from_env()
    return create_app(
        settings,
        Store(settings.db_path),
        HttpGateway(settings.gateway_url, settings.admin_key),
    )
