"""Tokens, usage counters and the kill switch, in a key-value store.

On Modal the store is a ``modal.Dict`` and the gateway runs in one container
(``max_containers=1`` in ``modal_app.py``), so the lock here is enough to make a reservation
atomic. Tokens are stored only as their SHA-256; revoking one deletes its row.

Keys: ``token:<sha256>`` holds a ``TokenRecord`` as a dict, ``used:<sha256>:<route>`` the calls a
token has made on a route, ``day:<YYYY-MM-DD>:<route>`` the calls of all tokens on a UTC day, and
``kill`` is true while the gateway is switched off.
"""

from __future__ import annotations

import asyncio
import hashlib
import secrets
from collections.abc import Callable, Iterator, MutableMapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, Protocol

from v2s_gateway.policy import ROUTES, Route

STATE_DICT = "video2sim-gateway-state"  # the modal.Dict that holds all of this on Modal
TOKEN_PREFIX = "v2s_"
KILL_KEY = "kill"


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def new_token() -> str:
    return TOKEN_PREFIX + secrets.token_urlsafe(32)


def token_key(token_hash: str) -> str:
    return f"token:{token_hash}"


def used_key(token_hash: str, route: Route) -> str:
    return f"used:{token_hash}:{route}"


def day_key(day: str, route: Route) -> str:
    return f"day:{day}:{route}"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class TokenRecord:
    """One judge token. ``scenes`` overrides the policy's ``scenes_per_token``."""

    label: str
    expires_utc: datetime
    scenes: int | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "expires_utc": self.expires_utc.isoformat(),
            "scenes": self.scenes,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> TokenRecord:
        expires = datetime.fromisoformat(str(data["expires_utc"]))
        if expires.tzinfo is None:
            raise ValueError("token expiry must carry a time zone")
        scenes = data.get("scenes")
        return cls(str(data["label"]), expires, None if scenes is None else int(scenes))


class Kv(Protocol):
    """Async key-value store. ``get`` answers None for a missing key."""

    async def get(self, key: str) -> Any: ...

    async def put(self, key: str, value: Any) -> None: ...

    async def delete(self, key: str) -> None: ...


class MemoryKv:
    """``Kv`` over a plain dict, for tests and local runs."""

    def __init__(self, data: MutableMapping[str, Any] | None = None) -> None:
        self.data: MutableMapping[str, Any] = {} if data is None else data

    async def get(self, key: str) -> Any:
        return self.data.get(key)

    async def put(self, key: str, value: Any) -> None:
        self.data[key] = value

    async def delete(self, key: str) -> None:
        self.data.pop(key, None)


class ModalDictKv:
    """``Kv`` over a ``modal.Dict`` through its async interface."""

    def __init__(self, store: Any) -> None:
        self._store = store

    async def get(self, key: str) -> Any:
        return await self._store.get.aio(key)

    async def put(self, key: str, value: Any) -> None:
        await self._store.put.aio(key, value)

    async def delete(self, key: str) -> None:
        if await self._store.contains.aio(key):
            await self._store.pop.aio(key)


class Refused(Exception):
    """A reservation was refused: the token's quota or the gateway's daily cap is used up."""

    def __init__(self, reason: Literal["quota", "daily"], route: Route, limit: int) -> None:
        super().__init__(f"{reason} limit {limit} reached on {route}")
        self.reason, self.route, self.limit = reason, route, limit


@dataclass(frozen=True)
class Reservation:
    token_hash: str
    route: Route
    day: str


class Ledger:
    def __init__(self, kv: Kv, clock: Callable[[], datetime] = utc_now) -> None:
        self._kv = kv
        self._clock = clock
        self._lock = asyncio.Lock()

    def now(self) -> datetime:
        return self._clock()

    async def token(self, token: str) -> tuple[str, TokenRecord] | None:
        """The hash and record of ``token``, or None when it is unknown or revoked."""
        token_hash = hash_token(token)
        data = await self._kv.get(token_key(token_hash))
        return None if data is None else (token_hash, TokenRecord.from_json(data))

    async def killed(self) -> bool:
        return bool(await self._kv.get(KILL_KEY))

    async def reserve(
        self, token_hash: str, route: Route, quota: int, daily_cap: int
    ) -> Reservation:
        """Count one call, or raise ``Refused`` without counting it."""
        day = self.now().date().isoformat()
        async with self._lock:
            used = int(await self._kv.get(used_key(token_hash, route)) or 0)
            if used >= quota:
                raise Refused("quota", route, quota)
            today = int(await self._kv.get(day_key(day, route)) or 0)
            if today >= daily_cap:
                raise Refused("daily", route, daily_cap)
            await self._kv.put(used_key(token_hash, route), used + 1)
            await self._kv.put(day_key(day, route), today + 1)
        return Reservation(token_hash, route, day)

    async def refund(self, reservation: Reservation) -> None:
        """Give back a call the upstream did not serve."""
        r = reservation
        async with self._lock:
            for key in (used_key(r.token_hash, r.route), day_key(r.day, r.route)):
                value = int(await self._kv.get(key) or 0)
                await self._kv.put(key, max(0, value - 1))

    async def used(self, token_hash: str) -> dict[Route, int]:
        return {r: int(await self._kv.get(used_key(token_hash, r)) or 0) for r in ROUTES}


class Busy(Exception):
    """No concurrency slot became free in time."""


class ConcurrencyGate:
    """At most ``limit`` requests in flight per key (a token and a route)."""

    def __init__(self) -> None:
        self._active: dict[tuple[str, str], int] = {}
        self._changed = asyncio.Condition()

    def active(self, key: tuple[str, str]) -> int:
        return self._active.get(key, 0)

    async def acquire(self, key: tuple[str, str], limit: int, wait_s: float) -> None:
        async with self._changed:
            if self.active(key) >= limit:
                try:
                    await asyncio.wait_for(
                        self._changed.wait_for(lambda: self.active(key) < limit), wait_s
                    )
                except asyncio.TimeoutError:
                    raise Busy(f"{limit} requests already in flight") from None
            self._active[key] = self.active(key) + 1

    async def release(self, key: tuple[str, str]) -> None:
        async with self._changed:
            left = self.active(key) - 1
            if left > 0:
                self._active[key] = left
            else:
                self._active.pop(key, None)
            self._changed.notify_all()


def token_rows(data: MutableMapping[str, Any]) -> Iterator[tuple[str, TokenRecord]]:
    """Hash and record of every token in a synchronous mapping (for ``admin.py``)."""
    for key, value in list(data.items()):
        if str(key).startswith("token:"):
            yield str(key).removeprefix("token:"), TokenRecord.from_json(value)
