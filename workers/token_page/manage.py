"""Operator commands for the token request page.

    python manage.py hash-code        # asks for an invite code, prints its SHA-256 for the env
    python manage.py list             # requests: who has which label, shown, revoked
    python manage.py revoke EMAIL     # revokes that judge's token on the gateway

``list`` and ``revoke`` read ``TOKEN_PAGE_DB``; ``revoke`` also needs ``V2S_GATEWAY_URL`` and
``V2S_GATEWAY_ADMIN_KEY``. After a revoke the judge can request again only once their row is
deleted from the file, which is deliberate: a lost token is a conversation, not a button.
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from token_page import HttpGateway, Settings, Store, hash_code


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="manage.py", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
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
    label = store.mark_revoked(args.email)
    if label is None:
        print(f"no request for {args.email}", file=sys.stderr)
        return 1
    count = HttpGateway(settings.gateway_url, settings.admin_key).revoke(label)
    print(f"revoked {label} ({count} token(s) on the gateway)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
