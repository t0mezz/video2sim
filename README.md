# Video2Sim

Turn a smartphone video of a tabletop workspace into a physics-ready USD/URDF scene.

SAM 3.1 segments and tracks the objects, SAM 3D completes their shapes, a reconstruction
stage gives them a metric position in the world, and a physics stage adds collision geometry,
mass, inertia and friction ranges. The result loads in Isaac Sim and is meant as a starting
point for reinforcement learning and robot training.

**Status:** pre-MVP, in Phase 5. All six stages run. Scenes are validated in MuJoCo; loading the
USD in Isaac Sim is not verified yet (D-027). See [PLAN.md](PLAN.md).

## Quick start

```bash
uv sync                 # install runtime and dev dependencies
uv run pytest           # fast tests, no GPU or model weights needed
uv run ruff check .     # lint
uv run mypy             # type check
```

## Make a scene

Film the table as [docs/capture_protocol.md](docs/capture_protocol.md) says, with the printed
marker board in view. The pipeline calls hosted models, so it needs a Meta Model API key and a
Baseten key in `.env` (see `.env.example`), and a Modal account with the SAM 3D worker deployed
(`AGENTS.md` has the commands). It sends video frames and object crops to those services.

```bash
uv run video2sim run my_table.mp4 --out scene/
```

`scene/` then holds `scene.usd`, one URDF per object in `urdf/`, and `qa_report.html`, which lists
every object with its confidence, fallbacks, mass range and settling result. Run folders go to
`runs/`, and running again with `--run-id` resumes from the last finished stage.

The pick-and-place demo needs the Panda model: run `scripts/fetch_panda.sh`, then see
`benchmark/run_pick_place.py`.

## Where to read next

| File | What it tells you |
| --- | --- |
| [PLAN.md](PLAN.md) | Scope, targets, phases, task list, risks, decisions. |
| [AGENTS.md](AGENTS.md) | Conventions and working rules, for people and coding agents. |
| [benchmark/README.md](benchmark/README.md) | How benchmark scenes and ground truth are laid out. |
| [docs/capture_protocol.md](docs/capture_protocol.md) | How to film a scene. |
| [docs/licenses.md](docs/licenses.md) | License status of every model and tool we depend on. |
| [docs/vlm_backends.md](docs/vlm_backends.md) | Pointing the VLM calls at Baseten or any OpenAI-compatible server. |
