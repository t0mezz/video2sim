"""Operator commands for the token request page.

    python manage.py status           # budgets used, and every token: who has it and its use
    python manage.py hash-code        # asks for an invite code, prints its SHA-256 for the env
    python manage.py list             # requests on this server only: who has which label
    python manage.py revoke EMAIL     # revokes that judge's token on the gateway

``list`` and ``revoke`` read ``TOKEN_PAGE_DB``; ``status`` and ``revoke`` also need
``V2S_GATEWAY_URL`` and ``V2S_GATEWAY_ADMIN_KEY``. After a revoke the judge can request again
only once their row is deleted from the file, which is deliberate: a lost token is a
conversation, not a button.
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from token_page import HttpGateway, Settings, Store, hash_code

ROUTES = ("vlm", "tracking", "sam3d")


def _uses(used: Mapping[str, Any], caps: Mapping[str, Any]) -> str:
    return " ".join(f"{r} {used.get(r, 0)}/{caps.get(r, '?')}" for r in ROUTES)


def status_lines(rows: Sequence[Mapping[str, Any]], listing: Mapping[str, Any]) -> list[str]:
    """The report of ``status``: the gateway's budgets, then each token with its holder.

    ``rows`` are this page's requests, ``listing`` is ``GET /admin/tokens``. They meet on the
    label; a token minted with ``admin.py`` has no holder here."""
    budget = listing.get("budget") or {}
    total, today = budget.get("total") or {}, budget.get("today") or {}
    gpu = budget.get("gpu_hours") or {}
    gpu_cap = f"{float(gpu['cap']):g}" if "cap" in gpu else "?"
    lines = [
        f"gateway {'SWITCHED OFF' if listing.get('switched_off') else 'on'}",
        "judging budget: "
        + ", ".join(f"{r} {total[r]['used']}/{total[r]['cap']}" for r in ROUTES if r in total)
        + f", GPU {float(gpu.get('used', 0)):.2f}/{gpu_cap} h",
        "today: "
        + ", ".join(f"{r} {today[r]['used']}/{today[r]['cap']}" for r in ROUTES if r in today),
    ]
    on_gateway = {str(t["label"]): t for t in listing.get("tokens") or []}
    holders = {str(row["label"]): row for row in rows}
    shown = sum(bool(row.get("shown_utc")) for row in rows)
    revoked = sum(bool(row.get("revoked_utc")) for row in rows)
    others = sorted(set(on_gateway) - set(holders))
    lines.append(
        f"tokens: {len(rows)} issued by this page ({shown} shown, {revoked} revoked), "
        f"{len(others)} minted by hand, {len(on_gateway)} on the gateway now"
    )
    for label, row in holders.items():
        who = f"{row['name']} <{row['email']}>"
        if row.get("affiliation"):
            who += f" ({row['affiliation']})"
        token = on_gateway.get(label)
        if row.get("revoked_utc"):
            state = "revoked"
        elif token is None:
            state = "not on the gateway"
        else:
            state = (
                ("shown" if row.get("shown_utc") else "not yet shown")
                + ", "
                + _uses(token.get("used") or {}, token.get("quota") or {})
            )
        issued = str(row.get("created_utc") or "")[:10]
        expires = str(row.get("expires_utc") or "-")[:10]
        lines.append(f"  {label}  {who}  issued {issued}  expires {expires}  {state}")
    for label in others:
        token = on_gateway[label]
        uses = _uses(token.get("used") or {}, token.get("quota") or {})
        expires = str(token.get("expires_utc") or "-")[:10]
        lines.append(f"  {label}  (minted with admin.py)  expires {expires}  {uses}")
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="manage.py", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="budgets used and every token with its holder and use")
    sub.add_parser("hash-code", help="print the SHA-256 of an invite code (asked, not echoed)")
    sub.add_parser("list", help="every request with its gateway label")
    r = sub.add_parser("revoke", help="revoke the token issued for EMAIL")
    r.add_argument("email")
    args = parser.parse_args(argv)

    if args.command == "hash-code":
        code = getpass.getpass("invite code: ")
        if len(code.strip()) < 12:
            print("use at least 12 characters, so the code cannot be guessed", file=sys.stderr)
            return 1
        print(hash_code(code))
        return 0
    store = Store(Path(os.environ.get("TOKEN_PAGE_DB", "token_page.sqlite3")))
    if args.command == "list":
        for row in store.rows():
            state = "revoked" if row["revoked_utc"] else "shown" if row["shown_utc"] else "issued"
            print(
                f"{row['label']}  {row['email']}  {row['name']}  {row['affiliation'] or '-'}  "
                f"{state}  expires {row['expires_utc'] or '-'}"
            )
        return 0
    settings = Settings.from_env()
    gateway = HttpGateway(settings.gateway_url, settings.admin_key)
    if args.command == "status":
        print("\n".join(status_lines(store.rows(), gateway.listing())))
        return 0
    label = store.mark_revoked(args.email)
    if label is None:
        print(f"no request for {args.email}", file=sys.stderr)
        return 1
    count = gateway.revoke(label)
    print(f"revoked {label} ({count} token(s) on the gateway)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
