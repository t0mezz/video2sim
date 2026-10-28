# PLAN.md

Scope, schedule and status of the Video2Sim MVP. This file is the working copy of the
roadmap: update it as work happens. Numbers that gate a decision live in
`configs/targets.yaml`; if this file and that file disagree, the config wins.

## Status

Last updated: 2026-10-26. Start date: not set (decision D-002).

| Phase | Weeks | Focus | Status | Gate |
| --- | --- | --- | --- | --- |
| 0 | 1 to 2 | Foundations, benchmark, evaluation harness | Done | Passed |
| 1 | 3 to 4 | Reconstruction, metric scale, world frame | Done | Passed |
| 2 | 5 to 6 | Segmentation, tracking, 3D lifting | Open | Open |
| 3 | 7 to 8 | Shape completion and pose fusion | Not started | Open |
| 4 | 9 to 10 | Physics readiness, material properties | Not started | Open |
| 5 | 11 to 12 | Export, simulator validation, demo | Not started | Open |

Status values: Not started, In progress, Done. Gate values: Open, Passed, Failed.
**Current focus:** Phase 1.

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

- [x] **P0.1** Capture the benchmark: 24 scenes, two camera paths each.

      Tiers: 10 easy, 6 medium, 8 hard (D-013). Prefer objects with published meshes.
      Four scenes use a clean board (production tags only, no printed grid) with object
      positions measured by tape, to check that the grid texture does not flatter
      registration. The rest use the benchmark grid board. Same tag IDs and size on both.
      Where: `benchmark/scenes/` (not in git), record in `benchmark/scene_list.yaml`.
      Done 2026-10-26: 24 scenes captured and 24 `ground_truth.json` files generated from
      `benchmark/scenes.csv` and `benchmark/inventory.csv`, 103 objects, none incomplete.
      s06, s14 and s24 have only `video_b.mp4`: no orbit was possible there (kitchen surface
      against the wall, accepted). Evidence: `benchmark/scene_list.yaml`, the two sheets in `benchmark/`.

- [x] **P0.2** Record ground truth per scene: weights, sizes, grid positions, orientations.

      Format: `benchmark/README.md`. Evidence: commits deeade1, d7af32d, acd2d78

- [x] **P0.3** Revise the capture protocol after the first scenes. Where: `docs/capture_protocol.md`.

      Evidence: commit 4768c4e

- [x] **P0.4** Print the scale marker board, measure the tag size with calipers, set the

      marker config. Where: `configs/default.yaml`. Evidence: tag 100 mm verified by calipers
      (reported by the user), `capture.marker` set in `configs/default.yaml`

- [x] **P0.5** Stage runner: run folder, manifest, hash-based caching. Where: `src/video2sim/runner.py`.

      Evidence: commit 3ba4d80

- [x] **P0.6** Evaluation harness that computes the section 2 metrics and writes a JSON and

      an HTML report. Run it on a trivial baseline to get a floor. Where:
      `src/video2sim/evaluation/`, command `video2sim eval`. Conventions: D-012, D-014.
      Evidence: commit 9bdcce7. On the 13 scored dev scenes (63 objects; s21 and s22 left
      out as deformable) the oracle scores 100% recall with zero error, and the constant-box
      floor scores 0% recall, so it has no error medians. The floor is a constant box plus a
      noise-perturbed oracle that tests the metrics. The boxes-from-masks baseline (segment,
      then extrude a box on the table plane from the marker pose) moves to the next
      iteration, with its costs: it needs a segmentation backend (D-004), the marker pose and
      the table plane from Phase 1.

- [x] **P0.7** Access and limits. SAM 3.1 through the Meta Model API versus self-hosting the

      released checkpoint: video length, rate limits, cost. SAM 3D license and GPU memory.
      TRELLIS.2 GPU memory. Record decisions D-004 and D-005. Where: `docs/licenses.md`
      Evidence: `docs/licenses.md`, D-004 and D-005 rows in the decision table, commits
      2a3bbe0 and 49730c9. Open: Meta Model API retention and length limits, Modal and
      Baseten terms.

- [ ] **P0.8** One GPU environment that runs every model, as a container image. Where: `Dockerfile`

      Deferred past the MVP (2026-10-26): no Docker image for now. Not done, so unticked.

- [x] **P0.9** Continuous integration: ruff, mypy and pytest on every push.

      Where: `.github/workflows/ci.yml` (Python 3.10 and 3.12). Evidence: commit 70c531d; the
      same four commands pass locally on 2026-10-26 (ruff check, ruff format --check, mypy,
      102 tests). A green run on GitHub has not been checked.

Exit criteria

- [x] 24 scenes captured, ground truth stored next to each
- [x] Evaluation harness produces the full metric report on the baseline
- [x] Written decisions on SAM 3.1 hosting (D-004) and on model licensing (D-005)
- [ ] Container image runs all models (deferred past the MVP, see P0.8)

Evidence: gate passed 2026-10-26 by human decision with the container criterion deferred
(P0.8). CI is green on GitHub.

## 5. Phase 1: Reconstruction, metric scale, world frame (weeks 3 to 4)

Goal: camera poses, a dense point cloud, true metric scale and a gravity-aligned world
frame. Everything later depends on this being right.

Tasks

- [x] **P1.1** Compare two or three back ends on the benchmark, for example classical

      COLMAP-style SfM with MVS against a feed-forward model such as VGGT or MASt3R. Judge on
      pose accuracy, runtime and failures on low-texture tables. Keep one and a fallback.

      Decision D-016 (2026-10-26): pycolmap on CPU is the back end for the MVP, with no fallback
      yet. The VGGT and MASt3R comparison and the GPU build are deferred, see section 12.
      Ticked 2026-10-27 by human decision with the comparison deferred: it did not happen, and
      the evidence is D-016 (six scenes, reprojection 0.46 to 0.72 px, all frames registered), not
      a back-end comparison.

- [x] **P1.2** Metric scale from the marker board: detect tags, compare their measured spacing

      with the known printed size, scale the reconstruction. Accept phone depth or LiDAR as an
      optional cross-check after the MVP.

      Done with the printed tag edge instead of the spacing, so one or two arbitrarily placed
      tags are enough; the grid board's known spacing is only a check on grid scenes. Evidence:
      `benchmark/results/p1_world_frame_dev.json` (9 dev grid videos: scale within 0.5% of the
      board scale, tag pair distances within 0.75%); code in commits b65dfb4, 72215b7, 540c39e,
      92e70ec. Tag edge since measured with calipers: 100.26 mm (5 readings, 100.1 to 100.4,
      s.e. 0.06), `tag_size_m: 0.1003`; re-run in `benchmark/results/p1_world_frame_dev_tag_1003.json`. Not measured:
      clean-board scenes and the board layout (still the nominal A1 layout).

- [x] **P1.3** Fit the table plane (RANSAC), take its normal as the up axis, put the origin on

      the marker board.

      The origin is on a reference tag (the one seen in most frames). Evidence: same file; up
      axis within 0.35 degrees of the board normal on the 9 dev grid videos. The board normal
      comes from the tag corners, so it is not fully independent of the plane fit.

- [x] **P1.4** Quality gate and frame sampling: blur and exposure filter, camera coverage,

      reprojection error. A failing video gets a specific message.

      Two gates, one message per failed check, all failures in one report. Before
      reconstruction (`stages/capture.py`): blur-aware sampling (sharpest frame of each
      1/5 s window), blur, exposure, marker visibility, frame count. After it
      (`reconstruction.check_reconstruction`): registered fraction, fragmentation,
      reprojection error, marker in 3D, camera coverage. Evidence:
      `benchmark/results/p1_4_capture_gate.json` (36 dev videos, 31 accepted, 5 deliberately degraded copies
      of s01 video_a: all 5 rejected with the matching message) and
      `benchmark/results/p1_world_frame_dev.json` (20 dev videos with reconstruction gate
      reasons). Findings: s12, the scene noted as blurred, passes both gates and reconstructs
      (0.4 to 0.6% scale error); s01 alt_lighting B is rejected by both (37% blurred, 71% of
      frames registered); s12 alt_path is rejected for blur but reconstructs (1.0% scale error),
      so the blur limit is stricter than reconstruction needs. Thresholds are guesses tuned on
      dev videos (`capture.quality`, `reconstruction.quality`); the coverage limit (60 degrees)
      is not backed by data and should be revisited in Phase 3. Not done: a duration limit (the
      10 to 60 s rule would reject benchmark videos that reconstruct fine), wiring the gates
      into the stage runner (P1.5), and a rolling-shutter or autofocus-hunting check.

- [x] **P1.5** Emit `WorldModel`.

      `CaptureStage` and `ReconstructionStage` run through the `Runner`; gate failures raise
      with the specific messages. Added `WorldModel.flags` for scene-level warnings. Evidence:
      run `s01_b` (`benchmark/run_stages.py`, local, `runs/` is not in git), summary in
      `benchmark/results/p1_5_worldmodel_s01_b.json`: 46 of 46 frames registered, 0.54 px, scale
      from the marker, cameras 0.58 to 0.75 m above the table. Not done: a rejected video run
      through the real stages (only a synthetic test covers it), more than one video, and
      cropping the point cloud (stray points lie below the table).

- [x] **P1.6** Evaluate the gate on the benchmark (`gates.reconstruction` in the config).

      Evidence: `benchmark/results/p1_world_frame_dev_tag_1003.json` (20 dev videos, tag edge 100.3 mm, pycolmap 4.2.1,
      code at commit 30b3530). 19 videos accepted and all pass the four gates; s01 alt_lighting B is
      rejected with specific messages. Caveats: (1) dev split only, the test split has not been
      run, so this is not a phase-gate result; (2) frames sampled at a fixed 3 fps, not through
      the capture gate's blur-aware sampling; (3) the board layout is the nominal A1 grid, not
      measured, so the marker-distance and up-axis errors inherit that assumption (the board
      normal comes from the tag corners, so up axis is not fully independent of the plane fit);
      (4) grid scenes only, no clean-board scene; (5) flags remain on s02 video_a
      (`sfm_fragmented`) and s12 video_a, video_b, alt_path (`tag_scale_disagree`, spread 4.2 to
      16.5%), which the gates do not reject.

Exit criteria

- [x] 95% of frames registered on easy and medium scenes: 100% on all 19 accepted dev videos (s01B: 57%, rejected)
- [x] Mean reprojection error 1 px or less: max 0.77 px (s12 alt_path)
- [x] Distance between two known marker points correct within 2% after scaling. Only applies to the scenes that are shot on the known grid surface. Max 1.56% (s12 alt_path), grid scenes only
- [x] Up axis within 2 degrees of the measured table normal: max 0.25 degrees (s06 video_b); the normal is the board normal from tag corners
- [x] Quality gate rejects the deliberately bad test videos with a useful message: s01 alt_lighting B rejected (57% registered vs 95%, 49 degrees coverage vs 60); the 5 degraded copies in `benchmark/results/p1_4_capture_gate.json`

Evidence: `benchmark/results/p1_world_frame_dev_tag_1003.json` (20 dev videos, 19 accepted, 1 rejected). Dev split only, fixed 3 fps, nominal board layout; see the P1.6 entry for caveats.

Test split (phase-gate run): `benchmark/results/p1_world_frame_test.json` (16 videos from 8 grid scenes, commit 1413958). Registered frames 100% on all 16, max reprojection error 0.72 px (s08 video_a), max up-axis error 1.28 degrees (s15 video_b). Marker-pair distance passes on 15 of 16; s16 video_b fails at 2.53% (limit 2%) and s16 video_a passes at 1.95%. Human decision: the s16 miss is expected, because the marker is badly warped in that footage; both s16 videos carry the `tag_scale_disagree` flag (5.9% and 3.3%). It is an input defect, not a pipeline pass. Without s16 the worst error is 1.54% (s07 video_b). Not run: s10, the only clean-board test scene, because the evaluation script handles grid scenes only; clean-board reconstruction is unevaluated in both splits. Same caveats as P1.6: fixed 3 fps sampling and a nominal board layout.

## 6. Phase 2: Segmentation, tracking, 3D lifting (weeks 5 to 6)

Goal: every physical object is exactly one instance, with a mask in each frame and a clean
point cluster in the world frame.

Tasks

- [x] **P2.1** Object discovery. A vision-language model lists the objects on a few sampled

      frames, then SAM 3.1 is prompted per noun. Compare with automatic mask proposals on
      recall and keep the better one.

- [x] **P2.2** Tracking with SAM 3.1 tracks. Repair identity errors with 3D consistency:

      merge tracks that project onto the same cluster, split tracks that cover two.

- [ ] **P2.3** Lift masks to 3D. A point is kept only when several views agree. Remove

      outliers, hands and arms.

- [ ] **P2.4** Keyframe selection per object: score mask area, occlusion, border contact and

      sharpness. Store the top 3 to 5.

- [ ] **P2.5** Failure flags: unstable depth (likely transparent or reflective), thin, tiny,

      heavily occluded.

- [ ] **P2.6** Emit `Instances`.

Exit criteria

- [x] Object recall 90% or more on the benchmark
- [x] At most one identity switch per scene on average
- [ ] Point clusters on the easy scenes have bounding boxes within 10% of measured size
- [ ] Failure flags raised on all four hard scenes

Evidence:

- P2.1 discovery comparison: `benchmark/results/p2_1_sam_generic.json` and `benchmark/results/p2_1_vlm_sam.json`. 13 scenes (12 scored dev scenes plus s16 from the test split), 63 visible objects, 6 sampled frames per scene, ground truth aligned to the world frame (`evaluation/alignment.py`), recall counted over objects found in any of the 6 frames, rerun from cached responses with no new API calls. s05 is unscored (origin unexplained).

  | method | objects found | recall (micro) | recall (macro) | easy / medium / hard | false positives |
  |---|---|---|---|---|---|
  | `sam_generic` | 61 of 63 | 96.8% | 91.3% | 100% / 100% / 85.7% | 20 |
  | `vlm_sam` | 62 of 63 | 98.4% | 99.0% | 100% / 100% / 92.9% | 112 |

  Both clear the 90% recall exit criterion. `vlm_sam` finds one more object (the glossy s24 stove cube) but sends about 10 times as many hosted calls and returns 5.6 times as many false positives, so `sam_generic` stays the default in `configs/default.yaml`. Both miss s18 obj_045 (pliers). The hybrid (generic first, noun prompts to fill gaps) is deferred. The criterion was measured on discovery masks only, not on tracked instances; the dev-split scenes were also used to choose the method, and s16 is the only test-split scene.

- P2.2 tracking: `benchmark/results/p2_2_dev.json` (12 dev scenes) and `benchmark/results/p2_2_test.json` (s16_a, the only test-split scene). One SAM 3.1 video request per scene with "objects on a table", direction `both`, the registered frames sent as a 5 fps clip (`stages/tracking.py`), then 3D repair (`stages/track_repair.py`). Identity is scored against ground truth by projecting the aligned boxes into each frame and assigning each track mask to the box it overlaps most (`evaluation/segmentation.py:score_tracks`); occlusion and lens distortion are ignored, so the count is approximate.
  - Identity switches: 0 in every one of the 13 scored scenes (mean 0.0 against a limit of 1.0), both before and after repair.
  - Fragments: one scene (s12) had an object split over two tracks; the repair merged them. No track was split. The thresholds (merge 5 cm, split 10 cm, 3 frames) were guesses and were not tuned.
  - Objects tracked: 60 of 63 rigid objects have a track. Missing: one object of the s13 stack (2 tracks for 3 objects), s18 obj_045 (pliers), and the s24 glossy cube (0 tracks). These are the same misses as discovery, so the generic prompt limits tracking recall too.
  - Stray tracks (no ground-truth object under them): s14 1, s18 1, s16_a 1.
  - Cost: video is billed per frame, at least 50 per request. The 13 scenes have 1,144 frames in total, about $0.23 at $0.20 per 1,000 frames; the billed count was read from the response only for s01 (46 frames billed as 50). s05 was not run.
  - Caveat: the API alone already gave 0 switches, so the benchmark does not exercise the repair; it merged one pair and split none. Whether it fixes real switches in harder footage is untested. The criterion says nothing about missed objects, which is why the recall figure above is reported next to it.

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
| D-003 | Compute: no local GPU inference. SAM 3.1 and the language model through the Meta Model API; SAM 3D and TRELLIS.2 checkpoints on a hosted GPU service (Modal or Baseten). Supersedes the single GPU workstation | Assumed | Week 1 | Not yet confirmed that either service can run the checkpoints; see D-015. Not checked: SAM 3D and TRELLIS.2 GPU memory (TRELLIS.2 needs 24 GB or more) |
| D-004 | SAM 3.1 through the Meta Model API, not self-hosted. Benchmark videos may be sent to hosted APIs (Meta Model API; Modal or Baseten under D-015). Never use the Muse "contributor" modes, which let Meta train on the content | Decided | Week 2 | Retention terms and video length limits still to be read (`docs/licenses.md`). `AGENTS.md` still says to ask before uploading |
| D-005 | Intended use: research. The SAM License has no research-only or commercial wording and asks for acknowledgment in publications. `nvdiffrast` and `nvdiffrec`, which TRELLIS.2 needs, allow research or evaluation only, so commercial use is blocked until they are replaced or relicensed | Decided | Week 2 | Texts read 2026-10-26, see `docs/licenses.md`. Still open: official Meta Model API terms and the Modal or Baseten terms |
| D-006 | Where benchmark videos and meshes are stored | Open | Week 1 | Not in git; they may show people or private spaces |
| D-007 | Scale source: printed marker, phone depth as optional cross-check | Assumed | Week 1 | |
| D-008 | Model for material-class inference | Open | Week 8 | The original brief names the Meta Model API |
| D-009 | Benchmark split: 12 dev scenes, 8 test scenes, stratified by tier: easy 6/4, medium 4/2, hard 2/2 (dev/test). Of the three clean-board easy scenes, two are dev and one is test | Decided | Week 1 | Test is run only at phase gates |
| D-010 | Object frame origin at the volume centroid; meshes in meters | Decided | | Encoded in `schemas.py` |
| D-011 | Python 3.10+, uv, pydantic for contracts, ruff and mypy strict | Decided | | Encoded in `pyproject.toml` |
| D-012 | Evaluation conventions (P0.6). Matching: Hungarian on box-center distance, gate = max(5 cm, half the smaller horizontal side). Predicted box: minimum-area rectangle of the XY footprint (hull edges) plus Z extent, so size, yaw and center follow the ground-truth convention. Size error: largest relative error over the three axes. Position error: 3D center distance. Orientation: yaw modulo 180 degrees, skipped for symmetric footprints. Mass: nominal within `mass_tolerance`; a matched object with no predicted mass fails. Medians cover matched objects only; misses count in recall. Ground truth and prediction share the board frame, with no alignment. Dev split by default; `--split test` and `--scene` are explicit. Stability, asset validity and manual steps report `not_measured` until Phase 5. Adds the `trimesh` dependency | Decided | Week 2 | `evaluation/metrics.py`, `evaluation/geometry.py` |
| D-013 | Benchmark grown from 20 to 24 scenes, supersedes D-009. Four hard scenes were added: s21 to s23 contain a deformable object and s24 is a glossy kitchen stove. Split is now 15 dev and 9 test: easy 6/4, medium 4/2, hard 5/3. Four clean-board scenes: s05, s14, s24 dev and s10 test. Deformable objects are for testing after the MVP: the harness excludes objects whose ground truth has `deformable: true` (counted as excluded, not scored) and scores the rigid objects in the same scenes | Decided | Week 2 | `benchmark/scene_list.yaml`, `evaluation/metrics.py` |
| D-014 | Deformables, refining D-013: `video2sim eval` leaves out every scene that has an object with `deformable: true` in its ground truth (s21 to s23), not just the object. They are listed in the report header as `scenes_excluded_deformable` and do not enter any metric or count. `--scene ID` still evaluates one on request. The scenes stay in the benchmark for testing after the MVP | Decided | Week 2 | `evaluation/scenes.py`, `cli.py` |
| D-015 | Shape stage inference: run the SAM 3D Objects and TRELLIS.2 checkpoints on Modal or Baseten GPUs, because the Meta Model API does not offer them. Hope for support, not confirmed | Open | Week 2 | Terms of Modal and Baseten read 2026-10-26: neither trains on customer data, neither forbids it, both leave the model licenses in force. SAM 3D access granted 2026-10-26. Needs: GPU credits on Modal or Baseten, a test run of a checkpoint on one host, and a fallback of boxes from masks if neither host works |
| D-016 | Reconstruction back end for the MVP: pycolmap on CPU (`reconstruction.backend: pycolmap`). Evidence 2026-10-26, mean reprojection error and frames registered: s01_b 0.50 px 28/28, s02_a 0.72 px 108/108, s05_b 0.54 px 20/20, s10_a 0.46 px 187/187, s19_a 0.58 px 193/193, s23_a 0.47 px 68/68, at 3 fps and 1280 px on an Apple M4 CPU, 3 s to 281 s per video. Pose and scale accuracy not yet measured. VGGT, MASt3R and `pycolmap-cuda12` are deferred past the MVP | Decided | Week 3 | Run was in a scratchpad script, not committed. `pycolmap-cuda12` is a separate Linux-only package with the same module name, so it would be a mutually exclusive extra |

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

6. **Reconstruction back ends (from P1.1).** Compare VGGT (and MASt3R) with pycolmap on the

   benchmark for tag-scale and up-axis error and on failures on low-texture tables, and check
   that `pycolmap-cuda12` runs and how much faster it is on a GPU. Both need a GPU host,
   weight downloads and license decisions (`docs/licenses.md`).

   If VGGT passes the accuracy gates, add it as an opt-in `reconstruction.backend: vggt` (a
   `run_sfm_vggt` returning `SfmModel`, torch behind an optional extra); expected gain is mainly
   on long (~190 frame) videos, unmeasured.

## 13. Changelog

| Date | Change |
| --- | --- |
| 2026-10-22 | First version, from the roadmap document |
| 2026-10-22 | P0.1: three of the ten easy scenes use a clean board without the printed grid |
| 2026-10-22 | D-009: split decided, stratified by tier, clean-board scenes two dev and one test |
| 2026-10-26 | D-013: benchmark is 24 scenes (15 dev, 9 test) with four clean-board scenes; `benchmark/scene_list.yaml` added |
| 2026-10-26 | D-014: the harness leaves out whole scenes with a deformable object |
| 2026-10-26 | P0.6: harness implemented on synthetic fixtures; floor is a constant-box baseline, boxes-from-masks deferred; D-012 records the metric conventions |
| 2026-10-26 | P0.8: Docker image skipped for the MVP and deferred; the matching exit criterion is deferred with it |
| 2026-10-26 | D-003 changed to hosted inference; D-005 decided as research; D-015 added for shape-stage hosting; D-004 leans to the Meta Model API |
| 2026-10-26 | D-004 decided: Meta Model API, and videos may be sent to hosted APIs |
| 2026-10-26 | D-005: license texts read; research use, TRELLIS.2 blocks commercial use through the NVIDIA renderer licenses |
| 2026-10-26 | P0.9: green CI run confirmed on GitHub; Phase 0 gate passed with the container criterion deferred; Phase 1 is the current focus |
| 2026-10-26 | D-016: pycolmap on CPU is the MVP reconstruction back end; VGGT, MASt3R and the CUDA build moved to section 12 |
| 2026-10-27 | P1.2, P1.3: scale from tag edge and world frame on a reference tag pass on 9 dev grid videos; results in `benchmark/results/p1_world_frame_dev.json` |
| 2026-10-27 | P1.5: `WorldModel` emitted by `ReconstructionStage` after `CaptureStage`; one-video run in `benchmark/results/p1_5_worldmodel_s01_b.json` |
| 2026-10-27 | P1.1 ticked by human decision with the VGGT and MASt3R comparison deferred (section 12, D-016) |
| 2026-10-27 | P1.4: capture and reconstruction quality gates, saved results in `benchmark/results/` |
| 2026-10-27 | P1.6: gate evaluated on 20 dev videos with the caliper tag edge (100.26 mm); P1.2 caliper note updated; Phase 1 exit criteria ticked with caveats in the P1.6 entry |
| 2026-10-27 | P1.6: test-split run on 16 videos; s16 marker-distance miss accepted as a warped-marker input defect by human decision; s10 clean-board not run; results in `benchmark/results/p1_world_frame_test.json` |