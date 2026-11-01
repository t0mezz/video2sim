<h1 align="center">Video2Sim</h1>

<p align="center">
  <b>Film a tabletop. Get a physics-ready USD/URDF scene.</b>
</p>

<p align="center">
  <img alt="python" src="https://img.shields.io/badge/python-3.10%2B-3776ab">
  <img alt="sim" src="https://img.shields.io/badge/validated%20in-MuJoCo-4c8bf5">
  <img alt="export" src="https://img.shields.io/badge/export-USD%20%2B%20URDF-76b900">
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> ·
  <a href="#make-a-scene">Make a scene</a> ·
  <a href="#where-to-read-next">Docs</a>
</p>

---

A smartphone video goes in. SAM 3.1 segments and tracks the objects, SAM 3D completes their
shapes, a reconstruction stage gives them a metric position in the world, and a physics stage adds
collision geometry, mass, inertia and friction ranges. The result loads in Isaac Sim and is meant
as a starting point for reinforcement learning and robot training.

```text
capture -> reconstruction -> segmentation -> shape -> physics -> export
```

## Quick start

```bash
uv sync                  # install runtime and dev dependencies
cp .env.example .env     # then fill in META_MODEL_API_KEY and BASETEN_API_KEY
uv run pytest -m "not gpu and not slow and not network"   # fast tests, no GPU or weights
uv run video2sim run my_table.mp4 --out scene/            # film first, see below
```

Every model endpoint (VLM, SAM 3D, SAM 3.1) is configurable, so you can point the pipeline at your
own instance or swap in an equivalent model: [docs/model_interfaces.md](docs/model_interfaces.md).
`uv run video2sim doctor` checks the endpoints before a run.

## Make a scene

1. Film the table as [docs/capture_protocol.md](docs/capture_protocol.md) says, with the printed
   marker board in view.
2. Set up the hosted services: the keys in `.env`, and a Modal account with the SAM 3D worker
   deployed (`AGENTS.md` has the commands). Video frames and object crops are sent to them.
3. Run `uv run video2sim run my_table.mp4 --out scene/`.

Progress goes to stderr, one line per stage (a spinner on a terminal); `--quiet` turns it off, and
`--verbose` adds the run id, cache keys and the COLMAP and CoACD logs. Stdout keeps only the final
line.

`scene/` holds `scene.usd`, one URDF per object in `urdf/`, and `qa_report.html` with each
object's confidence, fallbacks, mass range and settling result. Run again with `--run-id` to
resume from the last finished stage. For the pick-and-place demo, run `scripts/fetch_panda.sh`
and see `benchmark/run_pick_place.py`.

## Where to read next

| File | What it tells you |
| --- | --- |
| [PLAN.md](PLAN.md) | Scope, targets, phases, task list, risks, decisions. |
| [AGENTS.md](AGENTS.md) | Conventions and working rules, for people and coding agents. |
| [benchmark/README.md](benchmark/README.md) | How benchmark scenes and ground truth are laid out. |
| [docs/capture_protocol.md](docs/capture_protocol.md) | How to film a scene. |
| [docs/model_interfaces.md](docs/model_interfaces.md) | Hooking up your own VLM, SAM 3D and SAM 3.1 instances, or swapping the models. |
| [docs/licenses.md](docs/licenses.md) | License status of every model and tool we depend on. |
