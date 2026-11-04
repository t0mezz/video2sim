<p align="center">
  <img src="assets/v2s_logo.svg" height="108" alt="Video2Sim logo">
</p>

<h1 align="center">Video2Sim</h1>

<p align="center">
  <b>Film a tabletop. Get a physics-ready USD/URDF scene.</b>
</p>

<p align="center">
  <img alt="python" src="https://img.shields.io/badge/python-3.10%2B-3776ab">
  <img alt="sim" src="https://img.shields.io/badge/validated%20in-MuJoCo-4c8bf5">
  <img alt="export" src="https://img.shields.io/badge/export-USD%20%2B%20URDF-76b900">
  <a href="LICENSE"><img alt="license" src="https://img.shields.io/badge/license-MIT-green"></a>
  <a href="https://github.com/t0mezz/video2sim/actions/workflows/ci.yml"><img alt="ci" src="https://github.com/t0mezz/video2sim/actions/workflows/ci.yml/badge.svg"></a>
  <a href=".github/workflows/ci.yml"><img alt="platform" src="https://img.shields.io/badge/platform-linux%20%7C%20macOS%20%7C%20windows-E95420"></a>
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> ·
  <a href="#make-a-scene">Make a scene</a> ·
  <a href="#where-to-read-next">Docs</a>
</p>

---
### This section is primarily intended for the judges, 
### so you can dry run the project of of replays, without using the API's.
Thank you for taking the time to review this project! <3


---
A smartphone video goes in. SAM 3.1 segments and tracks the objects, SAM 3D completes their
shapes, a reconstruction stage gives them a metric position in the world, and a physics stage adds
collision geometry, mass, inertia and friction ranges. The result loads in Isaac Sim and is meant
as a starting point for reinforcement learning and robot training.

```text
capture -> reconstruction -> segmentation -> shape -> physics -> export
```

## Try it without any model or key

`replay` shows a recorded run of a benchmark scene. It computes nothing, calls no model and needs
no `.env`, GPU or Modal account; it only downloads the stored result and checks its hash.

```bash
uv sync
uv run video2sim replay                       # list the scenes
uv run video2sim replay s01 --out scene/      # same progress and package as `video2sim run`
```

Details and the limits of a replay: [docs/judge_snapshot.md](docs/judge_snapshot.md).

## Quick start

```bash
uv sync
cp .env.example .env     # fill in META_MODEL_API_KEY and BASETEN_API_KEY
uv run video2sim doctor  # check the model endpoints; --offline checks keys only
uv run video2sim run my_table.mp4 --out scene/
```

There are three model endpoints required for this project:
- a VLM: Muse Spark Open or Muse Glimmer 30b both work well
- a perception model, used for detection, segmentation and tracking (SAM3.1 default)
- a capable image to 3D model like SAM 3D

All three model endpoints are configurable: [docs/model_interfaces.md](docs/model_interfaces.md).

## Make a scene

1. Film the table per [docs/capture_protocol.md](docs/capture_protocol.md), marker board visible.
2. Set keys in `.env` and deploy the SAM 3D worker on Modal ([AGENTS.md](AGENTS.md) has the commands). Frames and crops are sent to these services.
3. Run `uv run video2sim run my_table.mp4 --out scene/` (`--run-id` resumes; `--quiet` / `--verbose` control progress on stderr).

`scene/` holds `scene.usd`, one URDF per object in `urdf/`, and `qa_report.html` (confidence, fallbacks, mass range, settling result). For the pick-and-place demo, run [scripts/fetch_panda.sh](scripts/fetch_panda.sh) and see [benchmark/run_pick_place.py](benchmark/run_pick_place.py).

## Dataset

The benchmark scene captures and the object inventory are public on Hugging Face:
[TomKrb/video2sim-captures](https://huggingface.co/datasets/TomKrb/video2sim-captures). Use them
to reproduce the evaluation or to try the pipeline without filming your own table. How the scenes
and ground truth are laid out is in [benchmark/README.md](benchmark/README.md).

## Acknowledgements

Inference for the project research was supported with credits from
- [Meta](https://dev.meta.ai) (SAM 3.1 through Meta Model API)
- [Modal](https://modal.com) (GPU hosting for the SAM 3D worker)
- [Baseten](https://www.baseten.co) (Muse Spark Open VLM inference)

Thank you for making this project possible!

---

## Where to read next

| File | What it tells you |
| --- | --- |
| [TODO.md](TODO.md) | Open work: unmet exit criteria, missed targets, known gaps, and the backlog after the MVP. |
| [docs/README.md](docs/README.md) | Index of the docs: [overview](docs/overview.md), [stages](docs/stages.md), [lessons](docs/lessons.md). |
| [docs/decisions.md](docs/decisions.md) | Every decision (D-001 to D-033) with its evidence. |
| [docs/history/](docs/history/) | Full task and evidence records of the finished phases, the old changelog, and the retired `PLAN.md`. |
| [benchmark/README.md](benchmark/README.md) | How benchmark scenes and ground truth are laid out. |
| [Hugging Face dataset](https://huggingface.co/datasets/TomKrb/video2sim-captures) | The public benchmark captures and object inventory. |
| [docs/capture_protocol.md](docs/capture_protocol.md) | How to film a scene. |
| [docs/model_interfaces.md](docs/model_interfaces.md) | Hooking up your own VLM, SAM 3D and SAM 3.1 instances, or swapping the models. |
| [docs/licenses.md](docs/licenses.md) | License status of every model and tool we depend on. |
| [docs/final_benchmark.md](docs/final_benchmark.md) | End-to-end results on the test split. |
