# Model interfaces

The project builds on three models, each with a fixed role: a **vision-language model** (VLM) for
the scene router, the material pass and the mass request, **SAM 3D Objects** for object shape, and
**SAM 3.1** for video tracking. The role is fixed; the **endpoint** is yours. Point each one at
your own instance, or at a different model that does the same job, through the config. Copy
`configs/default.yaml`, edit, and pass `--config my.yaml` to `video2sim run` or `eval`. The config
is hashed into the run manifest, so a run records which endpoints produced it.

```bash
uv run video2sim doctor --config my.yaml
```

checks that each endpoint builds and answers (`--offline` checks config and keys only; the tracking
endpoint is never called, since it is billed per frame).

Every role also accepts `python:package.module:factory`: the named factory is imported and its
object used. That is the escape hatch for a private API; the contracts it must meet are below.
The module must be importable (installed, or on `PYTHONPATH`).

## VLM

Contract: `client.ask(prompt: str, images: Sequence[Path]) -> str` and a `client.model: str` that
goes into the response-cache key. Temperature 0, one user message with the prompt and the images.

```yaml
vlm:
  protocol: openai_chat         # openai_chat | anthropic_messages | python:package.module:factory
  base_url: https://inference.baseten.co/v1   # "/chat/completions" or "/v1/messages" is appended
  model: meta-models/muse-spark-open-1026
  api_key_env: BASETEN_API_KEY  # env var or .env entry; null sends no key
  max_tokens: null              # thinking plus answer; a local reasoning model needs ~4096
  temperature: 0                # null leaves it to the server
  extra_body: {}                # merged into every request body (provider options)
  extra_headers: {}
  roles: {}                     # per-role overrides: router, material or mass
```

- `openai_chat` works with any server that implements OpenAI chat completions with `image_url`
  parts: Baseten, llama.cpp `llama-server`, vLLM, Ollama's `/v1`, LM Studio, OpenRouter, ...
- `anthropic_messages` speaks the Anthropic Messages API (`x-api-key`, base64 image blocks).
- `python:` factories are called as `factory(settings: VlmConfig, timeout_s, retries)`.
- Roles: the three questions can use different models, for example a small local model for the
  router and a stronger one for mass:

```yaml
vlm:
  roles:
    mass: {base_url: http://localhost:8001/v1, model: big-model, api_key_env: null}
```

Timeouts and retries (HTTP 429 and 5xx, 2/4/8 s backoff) come from each stage's own `timeout_s`
and `retries`.

## SAM 3D Objects (shape)

Contract: given one keyframe image, the object's mask and a seed, return a GLB mesh and the pose
of the model frame in the camera as `rotation_wxyz`, `translation`, `scale` (PyTorch3D
conventions) plus the `checkpoint` that ran. `models/sam3d.py` documents the conventions.

```yaml
shape:
  backend: sam3d                # or python:package.module:factory(cfg, work_dir) -> ShapeModel
  transport: modal              # modal | http | python:package.module:factory(cfg) -> transport
  modal: {app: video2sim-sam3d, cls: Sam3dWorker}
  http: {url: http://localhost:8000, api_key_env: null, timeout_s: 600, retries: 3}
```

- `transport: modal` (default) uses the worker of `models/sam3d_modal.py`, deployed under the app
  and class names above.
- `transport: http` posts to any server with the wire format in `models/sam3d_http.py`: multipart
  `POST /sample` in, GLB bytes out with the pose JSON in an `X-Pose-Json` header, `GET /health`.
  `workers/sam3d_http/` is a reference server and Dockerfile for it.
- A different image-to-3D model goes in through `backend: python:...`, returning an object with
  `name` and `sample(object_id, view, seed) -> ShapeSample` (`models/shape_model.py`). Its adapter
  converts the model's own frame and units; scale is never trusted as is (AGENTS.md).
- Warm-up (`shape.sam3d.warm_containers`) applies only to transports that can warm, which today
  is Modal.

## SAM 3.1 (tracking)

Contract: given a clip, a concept prompt and a direction, return SAM 3.1 special-token text, one
line per frame (`stages/tracking.py`, `models/hosted.py::parse_sam_text`).

```yaml
segmentation:
  tracking:
    endpoint:
      protocol: meta_responses  # or python:package.module:factory(endpoint, timeout_s)
      url: https://api.meta.ai/v1/responses
      model: sam-3.1
      api_key_env: META_MODEL_API_KEY
```

A self-hosted server that imitates the Meta Responses protocol needs only a different `url`. One
with another protocol uses a `python:` factory returning a `VideoSamClient`.

## Caching

Answers are cached on disk by a hash of the prompt, the images and the model name, so a different
model never reads another model's cached answer. Existing Baseten caches stay valid.

## A local llama.cpp instance (Muse Glimmer 30B, port 9931)

```yaml
vlm:
  protocol: openai_chat
  base_url: http://localhost:9931/v1
  model: muse-glimmer-30B
  api_key_env: null
  max_tokens: 4096
```

The development instance is `llama-server` serving `muse-glimmer-30B` (Q4_K_M, inputs text, image
and video), reached at `localhost:9931`, for example through an SSH tunnel. Its launch flags,
without host and file paths:

| Flag | Meaning |
| --- | --- |
| `-a muse-glimmer-30B` | model name clients send as `model` |
| `--mmproj <mmproj GGUF>` | vision projector, needed for image input |
| `-ngl 30` | 30 layers on the GPU |
| `-c 66560 -np 4` | 66560 tokens of context split over 4 slots: **16640 tokens per request** |
| `-fa on -ctk q4_0 -ctv q4_0` | flash attention, KV cache quantized to 4 bit |
| `-md <draft GGUF> -ngld 99` | draft model for speculative decoding, fully on the GPU |
| `--jinja` | the model's own chat template |
| `--temp 1.0 --top-p 0.95 --top-k 64` | server sampling defaults |

What this means for the pipeline:

- **Sampling:** the client sends `temperature: 0`, which overrides the server's `--temp 1.0`.
  `top_p` and `top_k` stay at the server defaults.
- **Context:** `GET /v1/models` shows `n_ctx` 16640, the per-slot share. On 2026-11-01 one
  3840x2160 PNG (2.2 MB) was answered in 26 s, and the same image three times in one request, as
  the router sends frames, in 47 s. Larger requests or more frames than that may not fit.
- **Concurrency:** 4 slots, which matches `segmentation.material_pass.max_workers: 4` and
  `physics.parallel_mass_calls`. A fifth request waits at the server.
- **History:** an earlier instance had a 8960-token context. A 3840x2160 PNG made it drop the
  connection and refuse connections for about a minute; 512 px and 1024 px JPEGs answered in 7
  and 9 s.

## Quality against Muse Spark Open

`benchmark/compare_vlm.py` runs the pipeline's own prompts and parsers on the same dev inputs
with both models (result: `benchmark/results/vlm_compare.json`, answers cached beside it). Muse
Spark Open is `meta-models/muse-spark-open-1026` on Baseten; local is `muse-glimmer-30B` on the
instance above, one run each on 2026-11-01, `max_tokens: 4096`.

| Task (dev inputs) | Spark | Local |
| --- | --- | --- |
| Router, 24 scenes: object count equals ground truth | 19 | 19 |
| Router: count within 1, surface chosen | 24, table | 24, table |
| Mass, 56 objects: within 30% of weighed mass | 23 (41%) | 23 (41%) |
| Mass: median ratio to truth, truth inside low-high | 1.02, 28 | 1.10, 30 |
| Material, 56 objects: answer readable | 56 | 56 |
| Material: `hollow` answered, correct among answered | 56, 38 | 38, 23 |
| Same material class / same finish as the other model | 50 / 56, 47 / 56 | |
| Mean seconds per request (136 each) | 13 | 86 (max 290) |

Router misses are the same five scenes in both models. Hollow is compared with the ground-truth
`hollow` flag only; material class and finish have no ground truth here, only the agreement
between the models. The local model answers `unknown` for `hollow` on 18 objects where Spark
commits. It is a reasoning model, so a request spends about 800 tokens thinking: that is why
`max_tokens` is 4096 and why it takes about 6.6 times as long. With `max_tokens: 1024` some answers
came back empty.

## Smoke test

```bash
curl -s http://localhost:9931/v1/models
```

```bash
uv run pytest tests/test_vlm.py
```
