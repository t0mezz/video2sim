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
    max_tokens: 4096              # thinking plus answer; null leaves it to the server
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
