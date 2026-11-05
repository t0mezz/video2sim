"""Operator commands for the judge gateway: tokens, usage and the kill switch.

    uv run python workers/gateway/admin.py mint judge-a judge-b --days 21
    uv run python workers/gateway/admin.py list
    uv run python workers/gateway/admin.py revoke judge-a
    uv run python workers/gateway/admin.py kill        # every model request answers 503 at once
    uv run python workers/gateway/admin.py revive

Each command reads or writes the gateway's ``modal.Dict`` with the operator's own Modal login.
``mint`` prints each token once; only its SHA-256 is stored, so a lost token is revoked and
minted again. Judges normally get their token from the request page on the home server, which
mints through ``/admin/tokens``; this script is for the operator.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from typing import TextIO

from v2s_gateway.ledger import STATE_DICT, Ledger, ModalDictKv
from v2s_gateway.policy import ROUTES, Policy, load_policy


async def mint(
    ledger: Ledger, labels: Sequence[str], days: float, scenes: int | None = None
) -> list[tuple[str, str]]:
    """New tokens for ``labels`` as (label, token). The operator's own command has no token cap;
    the admin endpoint for the request page has one (``policy.yaml``)."""
    if len(set(labels)) != len(labels):
        raise ValueError("labels must be distinct")
    return [(label, (await ledger.mint(label, days, scenes))[0]) for label in labels]


async def report(ledger: Ledger, policy: Policy, out: TextIO) -> None:
    print(f"gateway {'SWITCHED OFF' if await ledger.killed() else 'on'}", file=out)
    for token_hash, record in await ledger.tokens():
        used = await ledger.used(token_hash)
        usage = ", ".join(f"{r} {used[r]}/{policy.quota(r, record.scenes)}" for r in ROUTES)
        print(f"{record.label}: expires {record.expires_utc:%Y-%m-%d %H:%M} UTC; {usage}", file=out)


def _modal_ledger() -> Ledger:
    import modal

    return Ledger(ModalDictKv(modal.Dict.from_name(STATE_DICT, create_if_missing=True)))


async def run(args: argparse.Namespace, ledger: Ledger) -> int:
    if args.command == "mint":
        for label, token in await mint(ledger, args.labels, args.days, args.scenes):
            print(f"{label}\t{token}")
    elif args.command == "list":
        await report(ledger, load_policy(), sys.stdout)
    elif args.command == "revoke":
        count = await ledger.revoke(args.label)
        print(f"revoked {count} token(s) labelled {args.label}")
        return 0 if count else 1
    else:
        await ledger.set_killed(args.command == "kill")
        print(f"gateway switched {'off' if args.command == 'kill' else 'on'}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="admin.py", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("mint", help="new tokens, printed once")
    m.add_argument("labels", nargs="+", help="one pseudonym per judge (a-z, 0-9, . _ -)")
    m.add_argument("--days", type=float, default=21.0, help="days until expiry (default 21)")
    m.add_argument("--scenes", type=int, help="scenes per token instead of the policy's")
    sub.add_parser("list", help="tokens, expiry and usage")
    r = sub.add_parser("revoke", help="delete the tokens with this label")
    r.add_argument("label")
    sub.add_parser("kill", help="switch the gateway off: every model request answers 503")
    sub.add_parser("revive", help="switch the gateway on again")
    return asyncio.run(run(parser.parse_args(argv), _modal_ledger()))


if __name__ == "__main__":
    raise SystemExit(main())
