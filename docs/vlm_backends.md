# VLM backends

Three stages ask a vision-language model (VLM) a question about images: the scene router, the
material pass (both in `segmentation`) and the mass request (`physics.vlm_mass`). They all go
through `video2sim.models.vlm.make_vlm_client`, which reads the `vlm` section of
`configs/default.yaml`. SAM 3.1 video tracking and SAM 3D are separate services and are not
affected.

## Switching the backend

```yaml
vlm:
  backend: openai_compatible      # or baseten (the default)
  openai_compatible:
    base_url: http://localhost:9931/v1   # "/chat/completions" is appended
    model: muse-glimmer-30B
    api_key_env: null             # env var / .env entry holding a bearer token; null sends none
    max_tokens: 1024              # null leaves it to the server
```

- `baseten` is the hosted Muse Spark and needs `BASETEN_API_KEY`.
- `openai_compatible` works with any server that implements OpenAI chat completions with
  `image_url` content parts (llama.cpp `llama-server`, vLLM, Ollama's `/v1`, LM Studio, ...).
  Each request is one `user` message with the prompt and the images as base64 data URLs,
  `temperature: 0`. The answer is `choices[0].message.content`. Timeouts and retries
  (HTTP 429 and 5xx, 2/4/8 s backoff) come from each stage's own `timeout_s` and `retries`.

To keep the committed config untouched, copy it and pass `--config my.yaml` to `video2sim run`
or `video2sim eval`. The config is hashed into the run manifest, so a run records which
backend and model produced it.

## Caching

Answers are cached on disk by a hash of the prompt, the images and the model name, so a different
model never reads another model's cached answer. Existing Baseten caches stay valid.

## A local llama.cpp instance (Muse Glimmer 30B, port 9931)

On the development setup (reached at localhost:9931, e.g. through an SSH tunnel) `GET /v1/models` reports `muse-glimmer-30B`, llama.cpp, Q4_K_M, inputs text, image and video.

Limits to know about:

- **Context is 8960 tokens** (`n_ctx`; the model trained on 131072). The router sends 3 full
  frames in one request, which may not fit. If requests fail, lower `segmentation.router.n_frames`
  or restart `llama-server` with a larger `-c`.
- **Send small images.** A 3840x2160, 2 MB PNG made the server drop the connection and refuse
  connections for about a minute on 2026-11-01. 512 px (62 KB) and 1024 px (166 KB) JPEGs
  answered in 7 and 9 s. The pipeline's crops are small, but the router sends full-resolution
  frames; check their size before a first run.
- It is a local model with no retention terms to read, but unlike Baseten it has a single
  slot: `segmentation.material_pass.max_workers` and `physics.parallel_mass_calls` above 1 just
  queue at the server.

## Smoke test

```bash
curl -s http://localhost:9931/v1/models
```

```bash
uv run pytest tests/test_vlm.py
```
