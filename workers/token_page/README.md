# Judge token request page

A small standalone service for the home server (`vts.kortec.me`, D-042). A judge enters a name,
an email, an optional affiliation and the invite code from the judging instructions; a correct
code mints a gateway token at once, and a private link shows it exactly once. Your existing site
only links to it, for example at `https://vts.kortec.me/judges`.

Pages: `/` (the form, then four setup steps: clone and sync, set the token, `doctor`, `run`),
`/privacy` (what is sent where; linked from the consent box) and `/token/<id>` (the token once,
with the same steps filled in).

It talks to the gateway's `/admin/tokens` (see `workers/gateway/README.md`) with the admin key.
It holds no provider key and no Modal token. Tests: `tests/test_token_page.py`, which run it
against the real gateway app, offline.

## What it guards

* **Invite codes**, stored as SHA-256 in the environment and compared in constant time. Several
  codes are allowed (comma-separated), each good for `TOKEN_PAGE_MAX_PER_CODE` tokens.
* **One token per email.** A second request names the contact instead.
* **Ten form posts per IP address per hour**, failed or not, so the code cannot be guessed by trying.
* **Labels are pseudonyms** (`j-` plus 8 hex digits). Names and emails stay in the SQLite file
  here; the gateway and its logs see only the label.
* **The token is shown once.** It sits in the file only until the private link is opened, and at
  most 24 hours; then it is wiped. The page sends `no-store`, `no-referrer` and a strict CSP.

The gateway adds its own bounds on top: at most 30 unexpired tokens, 30 days and 5 scenes each.

## Set up

1. Copy `token_page.py`, `manage.py` and `requirements.txt` to the home server and install:

   ```bash
   python -m venv .venv && .venv/bin/pip install -r requirements.txt
   ```

2. Pick an invite code (at least 12 characters, for example four random words) and hash it:

   ```bash
   .venv/bin/python manage.py hash-code
   ```

3. Set the environment, for example in a systemd unit or an env file readable only by the
   service user:

   | Variable | Value |
   | --- | --- |
   | `V2S_GATEWAY_URL` | the gateway URL Modal printed |
   | `V2S_GATEWAY_ADMIN_KEY` | the same admin key as in the Modal secret `video2sim-gateway` |
   | `TOKEN_PAGE_INVITE_SHA256` | the hash from step 2 (comma-separated for several codes) |
   | `TOKEN_PAGE_DB` | path of the SQLite file, outside the web root |
   | `TOKEN_PAGE_TRUST_PROXY` | `1` behind the reverse proxy, so the rate limit sees the real IP |
   | `TOKEN_PAGE_CONTACT` | optional: an address judges can write to about a lost token |
   | `TOKEN_PAGE_DAYS`, `TOKEN_PAGE_SCENES` | optional: token validity (default 21 days) and scenes |
   | `TOKEN_PAGE_MAX_PER_CODE`, `TOKEN_PAGE_MAX_ATTEMPTS_PER_HOUR` | optional: 25 and 10 by default |

4. Run it on localhost under the path the proxy forwards:

   ```bash
   .venv/bin/uvicorn --factory token_page:app_from_env --host 127.0.0.1 --port 8090 --root-path /judges
   ```

5. Forward `/judges` to it over HTTPS. With nginx:

   ```nginx
   location /judges/ {
       proxy_pass http://127.0.0.1:8090/;
       proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
   }
   ```

   Link to `https://vts.kortec.me/judges/` from the site and put the invite code into the judging
   instructions, not on the page.

## Operate

```bash
.venv/bin/python manage.py list                    # label, email, name, shown or revoked
.venv/bin/python manage.py revoke judge@example.com
```

`list` is how a gateway label (in `admin.py list` or the gateway logs) maps back to a person.
To stop new tokens, stop the service or remove the invite hash; to stop all use, use the
gateway's kill switch. After judging, delete the SQLite file: it is the only place that holds the
judges' details.
