# AGENTS.md

Rules for anyone, human or AI agent, who changes this repository. Read it fully before your
first change. It is short on purpose, and every rule has a reason next to it.

## Orientation

Video2Sim turns one smartphone video of a tabletop into a USD/URDF scene that a physics
simulator can run. Six stages, in order: capture, reconstruction, segmentation, shape,
physics, export. Each stage hands the next one a typed object and some files.

**Order of authority** when two sources disagree:

1. The human's current instruction
2. `configs/targets.yaml` for every number that gates a decision
3. `PLAN.md` for scope, schedule and status
4. This file for how to work
5. Code comments

If two of them conflict, stop and ask. Do not pick one silently.

**Where we are:** the status table at the top of `PLAN.md`. Work on tasks of the current
phase. Tasks have ids such as `P3.2`; use them in branch names and commit messages.

## Repo map

```
AGENTS.md  PLAN.md  README.md
configs/
  default.yaml           pipeline settings; hashed into every run manifest
  targets.yaml           MVP success targets and phase gates (human-edited only)
  material_table.yaml    sourced material properties; the only home of material numbers
src/video2sim/
  schemas.py             the contracts between stages; start here
  stages/                one module per stage; each documents its contract
  models/                adapters around external models (SAM 3.1, SAM 3D, TRELLIS.2, VLM)
  evaluation/            metrics, baselines and reports against benchmark ground truth
  cli.py                 the `video2sim` command (`eval` today)
tests/                   fast tests; no GPU, weights or network needed
benchmark/               ground-truth format; scene data itself is not in git
docs/                    capture protocol, license tracker
data/  runs/             large artifacts; ignored by git
```

## Commands

These exist today. Run them before you say a change is done.

```bash
uv sync                      # install runtime and dev dependencies
uv run pytest                # tests
uv run ruff check .          # lint
uv run ruff format .         # format (use --check in CI)
uv run mypy                  # type check, strict
uv run video2sim eval --baseline constant   # metric report on the dev scenes, see below
```

`video2sim eval` scores a baseline (`--baseline oracle|constant`) or pipeline runs
(`--runs DIR`, one run folder per scene id) against `benchmark/scenes/*/ground_truth.json`.
It defaults to the dev split; `--split test` is for phase gates only, and `--scene ID`
scores one scene. Scenes with a deformable object are left out (D-014) unless named with `--scene`. It writes `report.json` and `report.html` and exits 0 unless
`--fail-on-target` is given. The metric definitions are decision D-012 in `PLAN.md`.

CI (`.github/workflows/ci.yml`) runs ruff check, `ruff format --check`, mypy and the fast
pytest set on every push, on Python 3.10 and 3.12, with `uv sync --locked`. A change that
touches dependencies commits the updated `uv.lock`. To reproduce the 3.10 run locally:
`uv run --python 3.10 --isolated mypy`.

Commands that arrive with their task: `video2sim run` (P5.5).
Never document a command that does not exist. When you add one, add it here in the same
change.

## Fundamentals

These conventions are the part of the project that is easy to get subtly wrong and hard to
notice. Breaking one gives a scene that loads fine and behaves wrongly.

### Units and frames

* **Units.** Meters, kilograms, seconds, radians. A name carries its unit: `mass_kg`,
  `size_m`, `angle_deg`. Convert only at the edges, such as reports and user messages.
* **World frame.** Right-handed, +Z up, gravity along -Z, origin on the scale-marker board.
* **Transforms.** `T_a_from_b` maps points expressed in frame `b` into frame `a`. Compose by
  matching inner names: `T_a_from_c = T_a_from_b @ T_b_from_c`. Transforms are rigid (SE(3)).
  Scale is never hidden inside one; it is applied to geometry once, at a named step.
* **Camera.** OpenCV convention: +x right, +y down, +z forward. Pixel `(x, y)` is
  `(column, row)`, while a NumPy array is indexed `[row, column]`.
* **Object frame.** Origin at the volume centroid. Meshes are stored in this frame, in
  meters, already scaled.
* **Scale.** Metric scale comes from the printed marker. A scale predicted by a generative
  model (SAM 3D, TRELLIS.2) is a guess and is never used as is.

### Physics

* Scale the mesh first, then compute mass and inertia. Under uniform scaling by $s$ at fixed
  density, $m \to s^{3} m$ and $\mathbf{I} \to s^{5} \mathbf{I}$. Never rescale mass or inertia
  after the fact.
* Mass and inertia follow from the repaired mesh under uniform density:

  $$m = \rho\,V_{\text{eff}}, \qquad \mathbf{I} = \rho \int_V \left(\lVert\mathbf{r}\rVert^2 \mathbf{1} - \mathbf{r}\mathbf{r}^{\top}\right) dV$$

  where $\mathbf{r}$ is measured from the center of mass. Hollow objects use an effective
  volume, never the filled one.
* Size errors are amplified in mass, which is why scale accuracy matters:

  $$\frac{\delta m}{m} \approx 3\,\frac{\delta s}{s} + \frac{\delta \rho}{\rho}$$

* Every physical property is an `Estimate` (low, nominal, high), never a bare float.
* The vision-language model picks a **material class**. Numbers come from
  `configs/material_table.yaml`, each with a source. A model must never write a number
  that ends up in an asset, because then results cannot be audited or repeated.

### Pipeline contract

* A stage is a function of a typed input and files, returning a typed output and files.
  Types are in `schemas.py`. Nothing else crosses a stage boundary.
* One run, one folder: `runs/<run_id>/manifest.json` plus a subfolder per stage. Paths in
  manifests are relative to the run folder; the schema rejects absolute paths.
* Stages are deterministic for a given seed, idempotent, and cached by a hash of input,
  config and code version. They never modify their inputs or write outside the run folder.
  Anything that samples (generative models especially) takes a seed and records it.
* **No silent failures.** If a stage cannot do something well, it sets a `Flag`, uses a
  fallback (for example a convex-hull proxy) and the QA report shows it. It never drops an
  object quietly, and never accepts a bad result quietly.
* Validators gate export. An asset with a failed check is not written to the scene.

### Data, models and secrets

* No videos, meshes, point clouds, model weights or secrets in git. The `.gitignore` covers
  the usual extensions; do not add exceptions. Keys live in `.env`, listed in `.env.example`.
* Videos of workspaces can show people, screens and documents. Treat benchmark videos as
  private. Decision D-004 in `PLAN.md` allows sending benchmark videos to hosted APIs (Meta Model
  API; Modal or Baseten under D-015). Never use the Muse "contributor" modes, which let
  Meta train on the content. Retention terms are not yet read, so ask before the first
  upload.
* Do not download large model weights unless asked. Record each checkpoint id in the run
  manifest.
* Before depending on a new model, tool or dataset, check and update `docs/licenses.md`.
* The benchmark has dev and test scenes. Tune on dev. Test is run at phase gates only.

## Code standards

* Python 3.10 or newer, type hints everywhere, `mypy --strict` clean, `ruff` clean, line length
  100. Use `pathlib.Path`, `logging` instead of `print`, and no module-level mutable state.
* Anything that crosses a stage boundary is a pydantic model from `schemas.py`. Arrays are
  `NDArray[np.float64]` (or an explicit dtype) and live in files, not in JSON.
* Import heavy libraries (torch, mesh and reconstruction libraries) inside the function that
  needs them, so the package imports and the tests run on a machine without a GPU.
* Isaac Sim ships its own Python interpreter. Code that runs inside it lives in a separate
  module that imports only the standard library, NumPy and `schemas.py`, and talks to the
  rest of the pipeline through files.
* Anything that changes a result goes in `configs/`, not in a constant. The config hash in
  the manifest has to identify the run.
* Comments say why, not what. A physical constant or a threshold cites its source or the
  experiment id that justified it.

## Testing

* Every change ships with a test. Tests must not need a GPU, model weights or network.
* Mark exceptions with `gpu`, `slow` or `network`. The fast set is
  `uv run pytest -m "not gpu and not slow and not network"`.
* Test physics against closed forms. A solid box with sides $a, b, c$ and mass $m$ has
  $I_{xx} = \tfrac{1}{12}\, m\,(b^{2} + c^{2})$. Build synthetic fixtures (box, cylinder,
  sphere) with known volume, mass and inertia, and check transforms by round trip.
* Do not judge quality by looking at one scene. A change that affects quality reports metric
  deltas from `video2sim eval` on the dev scenes.

## Working agreements for agents

**Before you start.** Find the task id in `PLAN.md`. Say what you understood the task to be
and what done means. For a change that touches a contract (`schemas.py`) or more than three
files, write a short plan and get agreement before editing.

**Ask a human first** before you:

* change a field in `schemas.py`, `configs/targets.yaml`, benchmark ground truth, or a
  phase's exit criteria;
* add a dependency, especially a heavy one, or download weights;
* call a paid or external API, or upload any data anywhere;
* delete files or rewrite git history.

**Never:**

* edit ground truth, or a target, to make a number pass;
* tune on the test split;
* invent a physical constant, material number or license fact. Cite it, or leave it empty
  and say so;
* tick a checkbox in `PLAN.md` without linking the evidence (a report, a PR, a run id);
* silence a linter, type checker or validator without a comment that says why.

**Definition of done.** Tests, `ruff` and `mypy` pass. The task's checkbox in `PLAN.md` is
ticked with evidence, or the remaining work is written there. If you added a command or
changed a convention, this file says so.

**Git.** One branch named like `p3-2-registration`. Small commits, imperative
mood, task id first: `P3.2: add one-sided chamfer registration`. A pull request says what
changed, why, the metric evidence, and what could go wrong.

## Known pitfalls

* **COLMAP stores `camera_from_world`**, the inverse of what most of this code calls a camera
  pose. It also writes quaternions as `w, x, y, z`; SciPy's `Rotation.from_quat` expects
  `x, y, z, w`. Convert at the boundary, and store rotations as matrices.
* **Camera axes differ between tools.** OpenCV looks along +z with +y down. USD and OpenGL
  cameras look along -z with +y up. A wrong flip gives mirrored or upside-down results
  that still look plausible.
* **USD fallbacks are Y-up and centimeters.** Set `upAxis = "Z"` and `metersPerUnit = 1.0`
  explicitly on every stage you write.
* **URDF inertia is about the center of mass**, in the inertial frame, in $\text{kg}\,\text{m}^2$.
  Its off-diagonal entries are the entries of the inertia matrix, which are the negatives of
  the products of inertia, for example $I_{xy} = -\int xy\,dm$. Mixing the two conventions is a
  classic bug, and simulators silently accept non-physical tensors.
* **Phone video carries rotation metadata and often a variable frame rate.** Decode with the
  rotation applied and by timestamp, or the frames will not match the intrinsics.
* **Generated shapes come in their own frame**, with arbitrary orientation and scale. A mesh
  that looks complete may still not be watertight. Validate before decomposing or measuring
  volume.
* **Convex decomposition of a non-manifold mesh is unreliable.** Repair and validate first.
* **If a model returns masks at a different resolution than the frames**, resize explicitly
  and test it. Do not rely on libraries to guess.
* **Keep coordinates small.** Far from the origin, float32 loses millimeters. Keep the world
  origin near the table.
