# PLAN.md

Scope, schedule and status of the Video2Sim MVP. This file is the working copy of the
roadmap: update it as work happens. Numbers that gate a decision live in
`configs/targets.yaml`; if this file and that file disagree, the config wins.

## Status

Last updated: 2026-10-22. Start date: not set (decision D-002).

| Phase | Weeks | Focus | Status | Gate |
| --- | --- | --- | --- | --- |
| 0 | 1 to 2 | Foundations, benchmark, evaluation harness | Not started | Open |
| 1 | 3 to 4 | Reconstruction, metric scale, world frame | Not started | Open |
| 2 | 5 to 6 | Segmentation, tracking, 3D lifting | Not started | Open |
| 3 | 7 to 8 | Shape completion and pose fusion | Not started | Open |
| 4 | 9 to 10 | Physics readiness, material properties | Not started | Open |
| 5 | 11 to 12 | Export, simulator validation, demo | Not started | Open |

Status values: Not started, In progress, Done. Gate values: Open, Passed, Failed.
**Current focus:** Phase 0.

## How to use this file

* Work on the current phase. Pick a task by id and say which one in the branch and commit.
* Tick a task only when it is done, and add its evidence on the same line: a report path, a
  run id or a PR link. A tick without evidence does not count.
* A phase gate passes only when every exit criterion is ticked with evidence. A phase starts
  when the previous gate has passed, except where Phase 0 notes say otherwise.
* Humans change scope, targets, exit criteria and decisions. Agents propose changes in a pull
  request and do not edit them directly.
* All durations are planning estimates, not measurements. Revise them when real numbers exist
  and record the change in the changelog.

## 1. MVP definition

The MVP turns one 10 to 60 second phone video of a static tabletop into a metric-scale USD
scene that loads in Isaac Sim, stays stable under gravity, and needs no manual steps on the
happy path.

**In scope**

* Input: one handheld video of 5 to 15 rigid objects on a table, with a printed scale marker
  in view.
* Output per object: a closed mesh, convex collision geometry, mass, inertia, and friction and
  restitution ranges. The table is a plane in a gravity-aligned world frame.
* Formats: USD for the scene, URDF per object, one target simulator.
* An automatic QA report per scene that flags every object it could not reconstruct well.

**Out of scope** (deferred on purpose, see section 12)

* Articulated objects. They are treated as rigid and flagged as likely candidates.
* Deformables and liquids: cloth, cables, bags, fluids.
* Transparent, mirror-like and very thin objects. They are flagged and replaced by a proxy or
  left out.
* Dynamic scenes. Objects must not move during capture; hands and arms are masked out.
* Rooms and large environments. Tabletop plus floor plane only.
* Measured friction. Friction and restitution come from material classes, as ranges.
* Training policies. The MVP proves the twin works in Isaac Lab with a scripted task.
* A product surface. Command line and config files only.

## 2. Success targets

Provisional. Re-set them once the Phase 0 baseline exists, then freeze them. The machine
readable copy is `configs/targets.yaml`.

| Metric | Target | Measured by |
| --- | --- | --- |
| Object recall | 90% or more of visible objects become assets | Count against labeled benchmark scenes |
| Object size error | Median 5% or less | Caliper measurements |
| Position error | Median 1.5 cm or less | Objects placed on a printed grid with known coordinates |
| Orientation error | Median 10 degrees or less | Same grid |
| Mass error | Within 30% for 70% of objects | Kitchen-scale weights |
| Simulation stability | 95% of scenes settle, max drift 2 cm in 5 s | Automated drop-under-gravity test |
| Asset validity | 100% of exported meshes pass every validator | Automated validators |
| Runtime | 60 minutes or less per scene on one GPU | Pipeline logs |
| Manual steps | None on the happy path | Run logs |

Mass error compounds from size error, which is why the size target is tight:

$$\frac{\delta m}{m} \approx 3\,\frac{\delta s}{s} + \frac{\delta \rho}{\rho}$$

A 5% scale error alone costs about 15% of mass, leaving the rest of the 30% budget for the
density guess.

## 3. Architecture

```
video
  |
  v
1 capture          quality gate, frame selection                -> CaptureResult
  v
2 reconstruction   poses, point cloud, metric scale, table      -> WorldModel
  v
3 segmentation     SAM 3.1 masks and tracks, 3D clusters        -> Instances
  v
4 shape            SAM 3D meshes registered to the cloud        -> RegisteredObjects
  v
5 physics          repair, collision, material, mass, inertia   -> SimAssets
  v
6 export           USD scene, URDF files, QA report             -> scene package
```

| Stage | Module | Reads | Writes | Phase |
| --- | --- | --- | --- | --- |
| 1 | `stages/capture.py` | video | `CaptureResult` | 1 |
| 2 | `stages/reconstruction.py` | `CaptureResult` | `WorldModel` | 1 |
| 3 | `stages/segmentation.py` | `WorldModel`, frames | `Instances` | 2 |
| 4 | `stages/shape.py` | `Instances`, `WorldModel` | `RegisteredObjects` | 3 |
| 5 | `stages/physics.py` | `RegisteredObjects`, crops | `SimAssets` | 4 |
| 6 | `stages/export.py` | `SimAssets`, `WorldModel`, `RegisteredObjects` | scene package | 5 |

Stage 4 carries the most uncertainty, which is why Phase 3 includes a time-boxed bake-off.
The types are in `src/video2sim/schemas.py`. Conventions are in `AGENTS.md`.

## 4. Phase 0: Foundations and evaluation set (weeks 1 to 2)

Goal: be able to measure every later change, and settle the access questions that could
block the plan.

**Do first, in parallel with code.** These need physical time or other people, so start them
on day one: print the marker board and measure it, collect and weigh the objects, get GPU
access, and start reading the model licenses (P0.7).

Tasks

- [ ] **P0.1** Capture the benchmark: 24 scenes, two camera paths each.
      Tiers: 10 easy, 6 medium, 8 hard (D-013). Prefer objects with published meshes.
      Four scenes use a clean board (production tags only, no printed grid) with object
      positions measured by tape, to check that the grid texture does not flatter
      registration. The rest use the benchmark grid board. Same tag IDs and size on both.
      Where: `benchmark/scenes/` (not in git), record in `benchmark/scene_list.yaml`.
      Status 2026-10-26: all 24 scenes captured per the scene list; s01 to s23 are on disk,
      s24 and the ground truth files are still to be imported. Tick when all are in place.
- [x] **P0.2** Record ground truth per scene: weights, sizes, grid positions, orientations.
      Format: `benchmark/README.md`. Evidence: commits e2aa49c, 322b27c, 1e808e7
- [x] **P0.3** Revise the capture protocol after the first scenes. Where: `docs/capture_protocol.md`.
      Evidence: commit 417cacb
- [x] **P0.4** Print the scale marker board, measure the tag size with calipers, set the
      marker config. Where: `configs/default.yaml`. Evidence: tag 100 mm verified by calipers
      (reported by the user), `capture.marker` set in `configs/default.yaml`
- [x] **P0.5** Stage runner: run folder, manifest, hash-based caching. Where: `src/video2sim/runner.py`.
      Evidence: commit 98a46f4
- [ ] **P0.6** Evaluation harness that computes the section 2 metrics and writes a JSON and
      an HTML report. Run it on a trivial baseline (boxes from masks) to get a floor.
      Where: `src/video2sim/evaluation/`
- [ ] **P0.7** Access and limits. SAM 3.1 through the Meta Model API versus self-hosting the
      released checkpoint: video length, rate limits, cost. SAM 3D license and GPU memory.
      TRELLIS.2 GPU memory. Record decisions D-004 and D-005. Where: `docs/licenses.md`
- [ ] **P0.8** One GPU environment that runs every model, as a container image. Where: `Dockerfile`
- [ ] **P0.9** Continuous integration: ruff, mypy and pytest on every push.

Exit criteria

- [ ] 24 scenes captured, ground truth stored next to each
- [ ] Evaluation harness produces the full metric report on the baseline
- [ ] Written decisions on SAM 3.1 hosting (D-004) and on model licensing (D-005)
- [ ] Container image runs all models

Evidence:

## 5. Phase 1: Reconstruction, metric scale, world frame (weeks 3 to 4)

Goal: camera poses, a dense point cloud, true metric scale and a gravity-aligned world
frame. Everything later depends on this being right.

Tasks

- [ ] **P1.1** Compare two or three back ends on the benchmark, for example classical
      COLMAP-style SfM with MVS against a feed-forward model such as VGGT or MASt3R. Judge on
      pose accuracy, runtime and failures on low-texture tables. Keep one and a fallback.
- [ ] **P1.2** Metric scale from the marker board: detect tags, compare their measured spacing
      with the known printed size, scale the reconstruction. Accept phone depth or LiDAR as an
      optional cross-check.
- [ ] **P1.3** Fit the table plane (RANSAC), take its normal as the up axis, put the origin on
      the marker board.
- [ ] **P1.4** Quality gate and frame sampling: blur and exposure filter, camera coverage,
      reprojection error. A failing video gets a specific message.
- [ ] **P1.5** Emit `WorldModel`.
- [ ] **P1.6** Evaluate the gate on the benchmark (`gates.reconstruction` in the config).

Exit criteria

- [ ] 95% of frames registered on easy and medium scenes
- [ ] Mean reprojection error 1 px or less
- [ ] Distance between two known marker points correct within 2% after scaling
- [ ] Up axis within 2 degrees of the measured table normal
- [ ] Quality gate rejects the deliberately bad test videos with a useful message

Evidence:

## 6. Phase 2: Segmentation, tracking, 3D lifting (weeks 5 to 6)

Goal: every physical object is exactly one instance, with a mask in each frame and a clean
point cluster in the world frame.

Tasks

- [ ] **P2.1** Object discovery. A vision-language model lists the objects on a few sampled
      frames, then SAM 3.1 is prompted per noun. Compare with automatic mask proposals on
      recall and keep the better one.
- [ ] **P2.2** Tracking with SAM 3.1 tracks. Repair identity errors with 3D consistency:
      merge tracks that project onto the same cluster, split tracks that cover two.
- [ ] **P2.3** Lift masks to 3D. A point is kept only when several views agree. Remove
      outliers, hands and arms.
- [ ] **P2.4** Keyframe selection per object: score mask area, occlusion, border contact and
      sharpness. Store the top 3 to 5.
- [ ] **P2.5** Failure flags: unstable depth (likely transparent or reflective), thin, tiny,
      heavily occluded.
- [ ] **P2.6** Emit `Instances`.

Exit criteria

- [ ] Object recall 90% or more on the benchmark
- [ ] At most one identity switch per scene on average
- [ ] Point clusters on the easy scenes have bounding boxes within 10% of measured size
- [ ] Failure flags raised on all four hard scenes

Evidence:

## 7. Phase 3: Shape completion and pose fusion (weeks 7 to 8)

Goal: a closed, textured mesh per object, registered to the measured point cloud in the
metric world frame. This is the highest-risk phase.

Tasks

- [ ] **P3.1** Adapters for SAM 3D Objects and TRELLIS.2 behind one interface. Draw several
      samples per keyframe, because both are generative. Where: `src/video2sim/models/`
- [ ] **P3.2** Registration. SAM 3D's predicted layout is the starting pose. Refine a
      similarity transform against the fused point cloud and the masks in all views, with a
      one-sided term so the unseen back of an object is not penalized. Scale is constrained by
      the metric cloud, not trusted from the model.

  $$\min_{s,R,t}\; \frac{1}{|Y|}\sum_{y\in Y}\min_{x\in\mathcal{M}}\lVert y-(sRx+t)\rVert^{2} + \lambda\sum_f \mathcal{L}_{\text{mask}}\big(\pi_f(s,R,t;\mathcal{M}),\,M_f\big)$$

  Here $Y$ is the observed cluster, $\mathcal{M}$ the generated mesh, $\pi_f$ the rendering
  into frame $f$ and $M_f$ the SAM mask in that frame.
- [ ] **P3.3** Bake-off, time-boxed to one week: SAM 3D, TRELLIS.2, and SAM 3D pose with
      TRELLIS.2 geometry, scored on the full benchmark by pose error, point-to-mesh distance
      and the Phase 4 stability pass rate. Keep one path and record it as a decision.
      Where: `configs/default.yaml` (`shape.backend`)
- [ ] **P3.4** Pick the best sample per object by fit score. Below the threshold, fall back to
      a convex hull or oriented box of the observed points and set `PROXY_GEOMETRY`.
- [ ] **P3.5** Snap resting objects to the table: choose the stable face from the convex hull,
      align the up axis, remove the small gap or penetration.
- [ ] **P3.6** Emit `RegisteredObjects`.

Exit criteria

- [ ] Median position error 1.5 cm or less and orientation error 10 degrees or less on easy
      and medium scenes
- [ ] Bake-off result written up and one geometry path chosen
- [ ] Every object has a registered mesh or an explicit low-confidence fallback
- [ ] Registered meshes resting on the table show no visible gap or penetration

Evidence:

## 8. Phase 4: Physics readiness and material properties (weeks 9 to 10)

Goal: every asset is simulation-ready and carries mass, inertia and friction values with
stated uncertainty.

Tasks

- [ ] **P4.1** Mesh repair: remove floaters, fill holes, make each mesh watertight and
      manifold, decimate to a triangle budget. Validators block export on failure.
- [ ] **P4.2** Collision geometry by convex decomposition (CoACD or V-HACD) with a cap on parts
      per object. Use a box, cylinder or sphere primitive where the object is close to one.
- [ ] **P4.3** Build `configs/material_table.yaml`: density, friction and restitution ranges per
      material class, every value with a source.
- [ ] **P4.4** Material inference. A vision-language model sees keyframe crops and returns
      structured fields only: category, material class, solid or hollow, confidence. Code
      maps the class to numbers through the table.
- [ ] **P4.5** Mass and inertia from the repaired mesh under uniform density, with a hollow
      correction. Store low, nominal and high.

  $$m = \rho\,V_{\text{eff}}, \qquad \mathbf{I} = \rho \int_V \left(\lVert\mathbf{r}\rVert^2 \mathbf{1} - \mathbf{r}\mathbf{r}^{\top}\right) dV$$
- [ ] **P4.6** Calibrate on the weighed benchmark objects: fit per-material corrections (for
      example the wall fraction of hollow containers) and report mass error before and after.
- [ ] **P4.7** Scene settling: resolve interpenetrations, drop the scene under gravity in the
      simulator, compare final poses with the observed ones.
- [ ] **P4.8** Emit `SimAssets`.

Exit criteria

- [ ] 100% of exported meshes pass the watertight, manifold and collision-load validators
- [ ] Mass within 30% for 70% of benchmark objects
- [ ] 95% of scenes settle with maximum drift of 2 cm in 5 s
- [ ] Every property in the export carries a low, nominal and high value

Evidence:

## 9. Phase 5: Export, simulator validation, demo (weeks 11 to 12)

Goal: one command takes a video to a USD scene that runs in Isaac Sim, with a QA report and a
basic manipulation demo.

Tasks

- [ ] **P5.1** USD writer: Z-up, meters, kilograms; per object a transform with rigid body and
      mass properties, visual mesh with textures, convex collision meshes, physics materials
      from the friction and restitution values; table and ground plane; physics scene.
- [ ] **P5.2** URDF writer per object (inertial, visual, collision), loaded in a second
      simulator such as PyBullet or MuJoCo to cross-check units, axes and inertia.
- [ ] **P5.3** QA report per scene: status, confidence and fallbacks per object, stability
      result, side-by-side renders of the real frame and the simulated scene from the same
      camera pose.
- [ ] **P5.4** Isaac Lab check: scripted pick-and-place with a manipulator on one object from
      the twin, repeated across the stored friction and mass ranges. Stretch: a short RL run.
- [ ] **P5.5** Command line (`video2sim run video.mp4 --out scene/`), container image, README
      with the capture guide.
- [ ] **P5.6** Final benchmark run on the test split and a write-up of every target, met or
      missed, with causes.

Exit criteria

- [ ] Final benchmark run meets every target, or the misses are documented with causes
- [ ] USD scenes open in Isaac Sim without edits; URDF loads in the second simulator
- [ ] Scripted pick-and-place demo recorded on at least three scenes
- [ ] A new person can reproduce a scene from the README alone

Evidence:

## 10. Risks and kill criteria

The first three can sink the plan. The trigger says when to change course instead of pushing
on.

| Risk | Mitigation | Trigger to change course |
| --- | --- | --- |
| SAM 3D pose or scale unreliable on tabletop crops | Refine against the metric point cloud; compare with TRELLIS.2 plus registration in P3.3 | Median orientation error above 20 degrees on easy scenes after refinement |
| Reconstruction fails on textureless tables or fast motion | Capture protocol, textured mat or printed board, feed-forward back end as fallback | More than 20% of benchmark videos fail to register |
| Model access or license blocks the intended use | Decide hosted versus self-hosted in P0.7; read every license | Any license that forbids the intended use, decided by end of week 2 |
| Generated geometry differs from the real object | Fit score gate, convex hull fallback, low-confidence flag | More than 25% of objects fall back to proxies |
| Mass and friction guesses are poor | Ranges instead of points, calibration on the weighed set, randomization in simulation | Mass target missed by more than 15 points after calibration |
| Bad meshes break the simulator | Validators block export; primitive collision fallbacks | Validator failure rate above 10% of objects after repair |
| Tracking identity errors create duplicate or missing assets | 3D-consistency merge and split, report flags | More than 2 identity switches per scene on average |
| Several large models exceed GPU memory or runtime | Load models one at a time, cache each stage, call hosted models where allowed | Runtime above 60 minutes per scene after caching |
| Scope creep toward articulation or rooms | Section 1 out-of-scope list is the contract; new asks go to section 12 | Any task that adds more than one week |

## 11. Decision log

Status values: Decided, Assumed (needs confirming), Proposed, Open. Add a row for every
decision that changes scope, a contract or a dependency. Do not rewrite old rows; add a new
one that supersedes.

| Id | Decision | Status | Due | Notes |
| --- | --- | --- | --- | --- |
| D-001 | Target simulator: Isaac Sim and Isaac Lab | Assumed | Week 1 | Changing it changes Phase 5 |
| D-002 | 12 weeks, two engineers, start date not set | Assumed | Week 1 | All durations depend on it |
| D-003 | Compute: one GPU workstation | Open | Week 1 | GPU memory needs of SAM 3D, SAM 3.1 and TRELLIS.2 not yet checked |
| D-004 | SAM 3.1: Meta Model API or self-hosted checkpoint | Open | Week 2 | Also decides whether videos may leave the machine |
| D-005 | Intended use (research or commercial) and the license consequences | Open | Week 2 | See `docs/licenses.md` |
| D-006 | Where benchmark videos and meshes are stored | Open | Week 1 | Not in git; they may show people or private spaces |
| D-007 | Scale source: printed marker, phone depth as optional cross-check | Assumed | Week 1 | |
| D-008 | Model for material-class inference | Open | Week 8 | The original brief names the Meta Model API |
| D-009 | Benchmark split: 12 dev scenes, 8 test scenes, stratified by tier: easy 6/4, medium 4/2, hard 2/2 (dev/test). Of the three clean-board easy scenes, two are dev and one is test | Decided | Week 1 | Test is run only at phase gates |
| D-010 | Object frame origin at the volume centroid; meshes in meters | Decided | | Encoded in `schemas.py` |
| D-011 | Python 3.10+, uv, pydantic for contracts, ruff and mypy strict | Decided | | Encoded in `pyproject.toml` |
| D-013 | Benchmark grown from 20 to 24 scenes, supersedes D-009. Four hard scenes were added: s21 to s23 contain a deformable object and s24 is a glossy kitchen stove. Split is now 15 dev and 9 test: easy 6/4, medium 4/2, hard 5/3. Four clean-board scenes: s05, s14, s24 dev and s10 test. Deformable objects are for testing after the MVP: the harness excludes objects whose ground truth has `deformable: true` (counted as excluded, not scored) and scores the rigid objects in the same scenes | Decided | Week 2 | `benchmark/scene_list.yaml`, `evaluation/metrics.py` |

## 12. After the MVP

In order:

1. **Articulation.** Capture a short interaction clip, fit joint axes and limits from the
   moving part, export joints to USD and URDF.
2. **Training loop.** Domain randomization over the stored property ranges, full RL runs in
   Isaac Lab, then a sim-to-real comparison on a physical robot.
3. **Harder objects.** Transparent, reflective and thin items; stacked objects and containers
   with contents.
4. **Larger scenes.** Shelves, cabinets and rooms, with structure priors for walls and
   furniture.
5. **Measured properties.** Push and tilt tests recorded on video to estimate friction,
   replacing the material-class guess.

## 13. Changelog

| Date | Change |
| --- | --- |
| 2026-10-22 | First version, from the roadmap document |
| 2026-10-22 | P0.1: three of the ten easy scenes use a clean board without the printed grid |
| 2026-10-22 | D-009: split decided, stratified by tier, clean-board scenes two dev and one test |
| 2026-10-26 | D-013: benchmark is 24 scenes (15 dev, 9 test) with four clean-board scenes; `benchmark/scene_list.yaml` added |
