# Video2Sim

Turn a smartphone video of a tabletop workspace into a physics-ready USD/URDF scene.

SAM 3.1 segments and tracks the objects, SAM 3D completes their shapes, a reconstruction
stage gives them a metric position in the world, and a physics stage adds collision geometry,
mass, inertia and friction ranges. The result loads in Isaac Sim and is meant as a starting
point for reinforcement learning and robot training.

**Status:** pre-MVP. The repository holds the contracts between stages and the plan. The
stages themselves are implemented phase by phase; see [PLAN.md](PLAN.md).

## Quick start

```bash
uv sync                 # install runtime and dev dependencies
uv run pytest           # fast tests, no GPU or model weights needed
uv run ruff check .     # lint
uv run mypy             # type check
```

## Where to read next

| File | What it tells you |
| --- | --- |
| [PLAN.md](PLAN.md) | Scope, targets, phases, task list, risks, decisions. |
| [AGENTS.md](AGENTS.md) | Conventions and working rules, for people and coding agents. |
| [benchmark/README.md](benchmark/README.md) | How benchmark scenes and ground truth are laid out. |
| [docs/capture_protocol.md](docs/capture_protocol.md) | How to film a scene. |
| [docs/licenses.md](docs/licenses.md) | License status of every model and tool we depend on. |
