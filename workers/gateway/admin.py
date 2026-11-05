"""Operator commands for the judge gateway: tokens, usage and the kill switch.

    uv run python workers/gateway/admin.py mint judge-a judge-b --days 21
    uv run python workers/gateway/admin.py list
    uv run python workers/gateway/admin.py revoke judge-a
    uv run python workers/gateway/admin.py kill        # every request answers 503 at once
    uv run python workers/gateway/admin.py revive

Each command reads or writes the gateway's ``modal.Dict`` with the operator's own Modal login.
``mint`` prints each token once; only its SHA-256 is stored, so a lost token is revoked and
minted again. Send tokens through the submission form, never through the repository.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator, MutableMapping, Sequence
from datetime import datetime, timedelta, timezone
from typing import Any, TextIO

from v2s_gateway.ledger import (
    KILL_KEY,
    STATE_DICT,
    TokenRecord,
    hash_token,
    new_token,
    token_key,
    token_rows,
    used_key,
)
from v2s_gateway.policy import ROUTES, Policy, load_policy


def mint(
    data: MutableMapping[str, Any],
    labels: Sequence[str],
    days: float,
    scenes: int | None = None,
    now: datetime | None = None,
) -> list[tuple[str, str]]:
    """New tokens for ``labels`` as (label, token). A label already in use is refused."""
    taken = {record.label for _, record in token_rows(data)}
    clash = sorted(set(labels) & taken)
    if clash or len(set(labels)) != len(labels):
        raise ValueError(f"labels must be new and distinct: {', '.join(clash) or 'duplicates'}")
    expires = (now or datetime.now(timezone.utc)) + timedelta(days=days)
    out = []
    for label in labels:
        token = new_token()
        data[token_key(hash_token(token))] = TokenRecord(label, expires, scenes).to_json()
        out.append((label, token))
    return out


def revoke(data: MutableMapping[str, Any], label: str) -> int:
    """Delete the tokens with ``label``; their usage counts stay for the record."""
    hashes = [h for h, record in token_rows(data) if record.label == label]
    for token_hash in hashes:
        del data[token_key(token_hash)]
    return len(hashes)


def report(data: MutableMapping[str, Any], policy: Policy, out: TextIO) -> None:
    killed = bool(data.get(KILL_KEY))
    print(f"gateway {'SWITCHED OFF' if killed else 'on'}", file=out)
    for token_hash, record in sorted(token_rows(data), key=lambda row: row[1].label):
        usage = ", ".join(
            f"{r} {int(data.get(used_key(token_hash, r)) or 0)}/{policy.quota(r, record.scenes)}"
            for r in ROUTES
        )
        print(f"{record.label}: expires {record.expires_utc:%Y-%m-%d %H:%M} UTC; {usage}", file=out)


class ModalDictMapping(MutableMapping[str, Any]):
    """The gateway's ``modal.Dict`` as a mapping, for the commands above."""

    def __init__(self, store: Any) -> None:
        self._store = store

    def __getitem__(self, key: str) -> Any:
        if not self._store.contains(key):
            raise KeyError(key)
        return self._store.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        self._store.put(key, value)

    def __delitem__(self, key: str) -> None:
        self._store.pop(key)

    def __iter__(self) -> Iterator[str]:
        return iter(list(self._store.keys()))

    def __len__(self) -> int:
        return int(self._store.len())

    def items(self) -> Any:
        return list(self._store.items())  # one round trip instead of one per key


def _modal_state() -> MutableMapping[str, Any]:
    import modal

    return ModalDictMapping(modal.Dict.from_name(STATE_DICT, create_if_missing=True))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="admin.py", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("mint", help="new tokens, printed once")
    m.add_argument("labels", nargs="+", help="one label per judge, shown in logs and usage")
    m.add_argument("--days", type=float, default=21.0, help="days until expiry (default 21)")
    m.add_argument("--scenes", type=int, help="scenes per token instead of the policy's")
    sub.add_parser("list", help="tokens, expiry and usage")
    r = sub.add_parser("revoke", help="delete the tokens with this label")
    r.add_argument("label")
    sub.add_parser("kill", help="switch the gateway off: every request answers 503")
    sub.add_parser("revive", help="switch the gateway on again")
    args = parser.parse_args(argv)

    data = _modal_state()
    if args.command == "mint":
        for label, token in mint(data, args.labels, args.days, args.scenes):
            print(f"{label}\t{token}")
    elif args.command == "list":
        report(data, load_policy(), sys.stdout)
    elif args.command == "revoke":
        count = revoke(data, args.label)
        print(f"revoked {count} token(s) labelled {args.label}")
        return 0 if count else 1
    elif args.command == "kill":
        data[KILL_KEY] = True
        print("gateway switched off")
    elif args.command == "revive":
        data[KILL_KEY] = False
        print("gateway switched on")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
