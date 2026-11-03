# Model interfaces

Three roles with fixed contracts; endpoints are configured per deployment. Copy `configs/default.yaml`, edit, pass `--config my.yaml` to `video2sim run` or `eval`. The config is hashed into the run manifest. `uv run video2sim doctor --config my.yaml` checks that each endpoint builds and answers (`--offline` checks config/keys only; tracking is never called because it bills per frame).

Each role also accepts `python:package.module:factory` (importable directly or via `PYTHONPATH`) for private APIs.

## VLM (scene router, material pass, mass request)

Contract: `client.ask(prompt: str, images: Sequence[Path]) -> str`, plus `client.model: str` in the response-cache key. Temperature 0, one user message with prompt and images.

```yaml
vlm:
  protocol: openai_chat         # openai_chat | anthropic_messages | python:package.module:factory
  base_url: https://inference.baseten.co/v1
  model: meta-models/muse-spark-open-1026
  api_key_env: BASETEN_API_KEY  # null sends no key
  max_tokens: null
  temperature: 0
  extra_body: {}
  extra_headers: {}
  roles: {}                     # per-role overrides: router, material, mass, verify
```

`openai_chat` fits any OpenAI-chat-compatible server with `image_url` parts (Baseten, llama.cpp `llama-server`, vLLM, Ollama `/v1`, LM Studio, OpenRouter). `anthropic_messages` speaks the Anthropic Messages API. Per-role models: e.g. `vlm.roles.mass` points at a stronger endpoint. Timeouts/retries (HTTP 429/5xx, 2/4/8 s backoff) come from each stage's `timeout_s`/`retries`.

## SAM 3D Objects (shape)

Contract: one keyframe image + object mask + seed returns a GLB mesh and the model-frame-in-camera pose (`rotation_wxyz`, `translation`, `scale`, PyTorch3D conventions) plus the `checkpoint`. Conventions: `models/sam3d.py`. Model-predicted scale is never used as metric scale.

```yaml
shape:
  backend: sam3d                # or python:package.module:factory(cfg, work_dir) -> ShapeModel
  transport: modal              # modal | http | python:package.module:factory(cfg) -> transport
  modal: {app: video2sim-sam3d, cls: Sam3dWorker}
  http: {url: http://localhost:8000, api_key_env: null, timeout_s: 600, retries: 3}
```

`modal` uses the `models/sam3d_modal.py` worker. `http` posts multipart `POST /sample` (GLB bytes + `X-Pose-Json` pose header, `GET /health`); wire format: `models/sam3d_http.py`, reference server: `workers/sam3d_http/`. Other image-to-3D models enter via `backend: python:...` returning `name` and `sample(object_id, view, seed) -> ShapeSample` (`models/shape_model.py`). `shape.sam3d.warm_containers` applies only to warmable transports (Modal).

## SAM 3.1 (tracking)

Contract: clip + concept prompt + direction returns SAM 3.1 special-token text, one line per frame (`stages/tracking.py`, `models/hosted.py::parse_sam_text`).

```yaml
segmentation:
  tracking:
    endpoint:
      protocol: meta_responses  # or python:package.module:factory(endpoint, timeout_s)
      url: https://api.meta.ai/v1/responses
      model: sam-3.1
      api_key_env: META_MODEL_API_KEY
```

Self-hosted Meta-Responses-compatible servers need only a different `url`; other protocols use a `python:` factory returning `VideoSamClient`. A self-hosted tracking server receives the object-prompt grammar from `docs/research/occlusion_robust_discovery_plan.md` section 2 and answers in the existing grammar with the sent ids.

## Caching

Answers cache on disk by hash of prompt, images, and model name; different models never share cache entries. Existing Baseten caches stay valid.

## Local llama.cpp instance (Muse Glimmer 30B, port 9931)

`llama-server` serving `muse-glimmer-30B` (Q4_K_M, text/image/video) at `localhost:9931` (e.g. via SSH tunnel). Launch: model alias `muse-glimmer-30B`, vision projector (`--mmproj`), 30 GPU layers, 66560-token context over 4 slots (16640 per request), flash attention with 4-bit KV cache, speculative-decoding draft model, model chat template, server sampling defaults `--temp 1.0 --top-p 0.95 --top-k 64`.

```yaml
vlm:
  protocol: openai_chat
  base_url: http://localhost:9931/v1
  model: muse-glimmer-30B
  api_key_env: null
  max_tokens: 4096
```

Client `temperature: 0` overrides the server default; `top_p`/`top_k` stay server-side. `GET /v1/models` reports per-slot `n_ctx` 16640. Measured 2026-11-01: one 3840x2160 PNG answered in 26 s; three copies in one request (as the router sends) in 47 s; larger requests may not fit. Four slots match `segmentation.material_pass.max_workers: 4` and `physics.parallel_mass_calls`; a fifth request queues. An earlier 8960-token instance dropped connections on full-resolution PNGs (7-9 s at 512-1024 px JPEG).

## Local vs hosted quality

`benchmark/compare_vlm.py` runs pipeline prompts/parsers with both models on identical dev inputs (`benchmark/results/vlm_compare.json`, 2026-11-01, one run each, `max_tokens: 4096`). Spark Open is `meta-models/muse-spark-open-1026` on Baseten; local is `muse-glimmer-30B` above.

| Task (dev inputs) | Spark | Local |
| --- | --- | --- |
| Router, 24 scenes: count equals truth | 19 | 19 |
| Router: count within 1, surface chosen | 24, table | 24, table |
| Mass, 56 objects: within 30% | 23 (41%) | 23 (41%) |
| Mass: median ratio, truth in low-high | 1.02, 28 | 1.10, 30 |
| Material, 56 objects: readable | 56 | 56 |
| Material: `hollow` answered / correct | 56 / 38 | 38 / 23 |
| Same class / finish as other model | 50 / 56, 47 / 56 | |
| Mean s/request (136 each) | 13 | 86 (max 290) |

Router misses are the same five scenes. Hollow is scored against the ground-truth flag; class/finish have no ground truth here, only cross-model agreement. Local answers `unknown` for `hollow` on 18 objects where Spark commits. Local is a reasoning model (~800 thinking tokens/request), hence `max_tokens: 4096` and ~6.6x latency; `max_tokens: 1024` returned some empty answers.

Smoke test: `curl -s http://localhost:9931/v1/models`; `uv run pytest tests/test_vlm.py`.
