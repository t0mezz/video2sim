# Judge gateway

A small FastAPI service that lets a judge run `video2sim run` on their own video with a personal
token instead of model keys. The pipeline needs no code change to use it: `--gateway URL` points
the three model endpoints that `configs/default.yaml` already makes configurable at the gateway
(`src/video2sim/gateway_profile.py`).

| Route | Client | Forwarded to |
| --- | --- | --- |
| `POST /v1/chat/completions` | `models/vlm.py` (router, material, mass, verify) | Baseten chat completions |
| `POST /v1/responses` | `stages/tracking.py`, streamed (SSE) | `https://api.meta.ai/v1/responses` |
| `POST /sam3d/sample`, `GET /sam3d/health` | `models/sam3d_http.py` | the Modal worker of `models/sam3d_modal.py` |
| `GET /v1/usage`, `GET /health` | `video2sim doctor --gateway` | nothing |

Upstream URLs are fixed in `v2s_gateway/upstreams.py`, so a request cannot name a destination.
The real keys live only in the Modal secret. What a token may send, how much and how many at once
is in `policy.yaml`. The tests are `tests/test_gateway.py` (fake upstreams, no network).

## Controls

| Risk | Control |
| --- | --- |
| Key leak | Keys only in the Modal secret. Upstream error bodies and headers are never passed on: an upstream failure becomes our own 503 (retryable, refunded) or 422 (rejected). Logs hold path, token label, status, bytes and milliseconds only. |
| Free LLM for others | Model allowlist per route. The VLM body is rebuilt from `model`, `messages`, `temperature` and `max_tokens`; `max_tokens` is clamped to 4096 and set to 4096 when missing. Images and the clip must be `data:` URLs, so no upstream fetches a URL. |
| GPU burn | SAM 3D options must have exactly the keys of `Sam3dConfig.options()`, an allowed checkpoint and null steps. Size caps on image, mask, pointmap and every body. |
| Cost runaway | Per-token quota per route, a daily cap per route over all tokens, and a kill switch read on every request. Provider spend limits are the real backstop. |
| Token leak | Tokens are stored only as SHA-256, with an expiry. Revoking deletes the row. |
| Overload | Per-token concurrency per route (8 VLM, 4 tracking, 6 SAM 3D, matching the pipeline's own parallelism); a request over it waits up to `queue_wait_s` for a slot, then gets 429. |

The status codes follow the clients' retry rule (`post_json` and `HttpTransport` retry 429 and
5xx). A quota or daily budget that is used up answers 403, which is not retried. Each refusal is
JSON with a `message`, which the client puts into its error text (`explain_http_error` in
`models/hosted.py`), so a judge sees for example "This judge token has used its vlm quota of 240
calls. `uv run video2sim replay` shows recorded runs without any model or key."

Quota sizing: for route $r$ a token gets $Q_r = \lceil c \cdot n_{\text{scenes}} \cdot k_r \rceil$
calls, with $c$ = `safety_factor`, $n_{\text{scenes}}$ = `scenes_per_token` and $k_r$ =
`calls_per_scene`. The daily cap uses `daily_scenes` in place of $n_{\text{scenes}}$. The $k_r$ in
`policy.yaml` are provisional: the largest per-scene counts of cached answers in the local dev
runs. Task G4 replaces them with a measured run.

## Set up (human, each step needs a go-ahead)

1. Create dedicated hackathon keys at Baseten and Meta, with spend limits set in each dashboard,
   and set a workspace spend limit at Modal. Rotate or revoke them after judging.
2. Store them in the Modal secret the gateway reads:

   ```bash
   uv run modal secret create video2sim-gateway BASETEN_API_KEY=... META_MODEL_API_KEY=...
   ```

3. The SAM 3D worker must be deployed in the same Modal workspace
   (`uv run modal deploy src/video2sim/models/sam3d_modal.py`, see `AGENTS.md`).
4. Deploy the gateway. Modal prints its URL (`https://<workspace>--video2sim-gateway-web.modal.run`):

   ```bash
   uv run modal deploy workers/gateway/modal_app.py
   ```

5. Mint one token per judge. Each is printed once; only its hash is stored. Send tokens through
   the submission form, never through the repository or a public channel.

   ```bash
   uv run python workers/gateway/admin.py mint judge-a judge-b judge-c --days 21
   ```

## Operate

```bash
uv run python workers/gateway/admin.py list            # tokens, expiry, usage against quota
uv run python workers/gateway/admin.py revoke judge-a  # that token answers 401 from now on
uv run python workers/gateway/admin.py kill            # every model route answers 503 at once
uv run python workers/gateway/admin.py revive
```

`--scenes N` on `mint` gives one token a different scene allowance. Quotas follow `policy.yaml`
at the time of the request, so a changed policy takes effect on the next deploy.

**Kill switch.** `admin.py kill` sets a flag the gateway reads on every request. If Modal itself
is the problem, `uv run modal app stop video2sim-gateway` stops the app; a new deploy starts it
again with the same tokens and usage.

**Key rotation.** Create the new provider key, update the secret, redeploy, then revoke the old
key at the provider:

```bash
uv run modal secret create video2sim-gateway BASETEN_API_KEY=... META_MODEL_API_KEY=... --force
```

```bash
uv run modal deploy workers/gateway/modal_app.py
```

**Teardown after judging.** Stop the app, delete the state and the secret, and revoke the
provider keys in their dashboards:

```bash
uv run modal app stop video2sim-gateway
```

```bash
uv run modal dict delete video2sim-gateway-state
```

```bash
uv run modal secret delete video2sim-gateway
```

## Not verified from here (task G3)

These depend on Modal's web-endpoint limits and need a deploy with a tiny cap and a real clip:

* the largest tracking clip passes Modal's request-size limit (bodies up to 64 MB here);
* a tracking request of about 10 minutes streams through without a timeout (the app's function
  timeout is 1800 s);
* a cold SAM 3D draw completes through the gateway. Over HTTP the pipeline does not warm the GPU
  containers the way `ModalTransport` does (`shape.sam3d.warm_containers`), so a judge's first
  draws start cold.

If Modal's web timeout turns out too short, the fallback is a long-lived container host for the
gateway plus a Modal token scoped to the SAM 3D app.

The gateway runs in one container (`max_containers=1`), because the quota lock and the
concurrency counts are in memory. While Modal rolls a new deploy, two containers may overlap for a
moment, and a token could then go a call or two over its quota.
