# PLAN.md

Scope, schedule and status of the Video2Sim MVP. This file is the working copy of the
roadmap: update it as work happens. Numbers that gate a decision live in
`configs/targets.yaml`; if this file and that file disagree, the config wins.

## Status

Last updated: 2026-10-31. Start date: not set (decision D-002).

| Phase | Weeks | Focus | Status | Gate |
| --- | --- | --- | --- | --- |
| 0 | 1 to 2 | Foundations, benchmark, evaluation harness | Done | Passed |
| 1 | 3 to 4 | Reconstruction, metric scale, world frame | Done | Passed |
| 2 | 5 to 6 | Segmentation, tracking, 3D lifting | Done | Passed |
| 3 | 7 to 8 | Shape completion and pose fusion | Done | Passed |
| 4 | 9 to 10 | Physics readiness, material properties | In progress | Open |
| 5 | 11 to 12 | Export, simulator validation, demo | Not started | Open |

Status values: Not started, In progress, Done. Gate values: Open, Passed, Failed.
**Current focus:** Phase 4.

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
block the plan. Gate passed 2026-10-26. Full record: `docs/history/phases_0_2.md`.

- [x] **P0.1** Benchmark: 24 scenes (10 easy, 6 medium, 8 hard, D-013), 103 objects, two camera paths each; s06, s14 and s24 have only `video_b` (no orbit possible). Evidence: `benchmark/scene_list.yaml`, `benchmark/scenes.csv`, `benchmark/inventory.csv`
- [x] **P0.2** Ground truth per scene, format in `benchmark/README.md`. Evidence: commits deeade1, d7af32d, acd2d78
- [x] **P0.3** Capture protocol revised, `docs/capture_protocol.md`. Evidence: commit 4768c4e
- [x] **P0.4** Marker board printed, tag edge measured with calipers, `capture.marker` in `configs/default.yaml`
- [x] **P0.5** Stage runner with run folder, manifest and hash caching, `src/video2sim/runner.py`. Evidence: commit 3ba4d80
- [x] **P0.6** Evaluation harness, `video2sim eval` (D-012, D-014): oracle 100% recall, constant-box floor 0%. Evidence: commit 9bdcce7
- [x] **P0.7** Access and limits, `docs/licenses.md`, D-004 and D-005. Evidence: commits 2a3bbe0, 49730c9. Open: Meta Model API retention and length limits, Modal and Baseten terms
- [ ] **P0.8** GPU container image. Deferred past the MVP (2026-10-26)
- [x] **P0.9** CI with ruff, mypy and pytest, `.github/workflows/ci.yml`. Evidence: commit 70c531d, green on GitHub

Exit criteria

- [x] 24 scenes captured, ground truth stored next to each
- [x] Evaluation harness produces the full metric report on the baseline
- [x] Written decisions on SAM 3.1 hosting (D-004) and on model licensing (D-005)
- [ ] Container image runs all models (deferred past the MVP, see P0.8)

## 5. Phase 1: Reconstruction, metric scale, world frame (weeks 3 to 4)

Goal: camera poses, a dense point cloud, true metric scale and a gravity-aligned world
frame. Gate passed 2026-10-27. Full record: `docs/history/phases_0_2.md`.

- [x] **P1.1** Back end: pycolmap on CPU, sequential matching (D-016, section 12 item 6b); the VGGT and MASt3R comparison is a post-MVP item (section 12 item 6). Ticked by human decision with the comparison deferred
- [x] **P1.2** Metric scale from the printed tag edge (100.26 mm, calipers): scale within 0.5% of the board, tag pairs within 0.75% on 9 dev grid videos. Evidence: `benchmark/results/p1_world_frame_dev_tag_1003.json`
- [x] **P1.3** Table plane and up axis, origin on the reference tag: up axis within 0.35 deg of the board normal. Evidence: same file
- [x] **P1.4** Quality gates before and after reconstruction with one message per failed check: 5 of 5 degraded videos rejected. Evidence: `benchmark/results/p1_4_capture_gate.json`. Open: a duration limit, a rolling-shutter or autofocus check; the 60 deg coverage limit is a guess to revisit
- [x] **P1.5** `WorldModel` from `CaptureStage` and `ReconstructionStage`. Evidence: `benchmark/results/p1_5_worldmodel_s01_b.json`. Open: point cloud cropping (stray points below the table)
- [x] **P1.6** Gate evaluated on 20 dev videos. Evidence: `benchmark/results/p1_world_frame_dev_tag_1003.json`

Exit criteria (dev: 19 accepted videos; test: `benchmark/results/p1_world_frame_test.json`, 16 grid videos)

- [x] 95% of frames registered on easy and medium scenes: 100% on dev and test
- [x] Mean reprojection error 1 px or less: max 0.77 px dev, 0.72 px test
- [x] Marker-point distance within 2% after scaling (grid scenes only): max 1.56% dev; test 15 of 16, s16 video_b 2.53% accepted by human decision (warped marker, flagged `tag_scale_disagree`)
- [x] Up axis within 2 deg of the table normal: max 0.25 deg dev, 1.28 deg test
- [x] Quality gate rejects the deliberately bad test videos with a useful message

Caveats: fixed 3 fps sampling, nominal board layout, clean-board reconstruction not evaluated.

## 6. Phase 2: Segmentation, tracking, 3D lifting (weeks 5 to 6)

Goal: every physical object is exactly one instance, with a mask in each frame and a clean
point cluster in the world frame. Gate passed 2026-10-29. Full record: `docs/history/phases_0_2.md`; cluster box
work: `docs/research/cluster_box_accuracy.md`.

- [x] **P2.1** Object discovery: generic SAM prompt 96.8% recall, VLM plus SAM 98.4% with 5.6x the false positives; generic kept. Evidence: `benchmark/results/p2_1_sam_generic.json`, `p2_1_vlm_sam.json`. The discovery code was removed in eef52b6: tracking finds the objects
- [x] **P2.2** One SAM 3.1 video request per scene plus 3D identity repair: 0 switches on 13 scenes, 60 of 63 rigid objects tracked. Evidence: `benchmark/results/p2_2_dev.json`, `p2_2_test.json`
- [x] **P2.3** Lift to 3D: voted sparse points without the outermost strays, then a box, cylinder or sphere fitted to the masks; the visual hull is kept for the shape flags. Evidence: `benchmark/results/p2_3b_dev.json` (earlier lifts: `p2_3_dev.json`, `p2_3_test.json`)
- [x] **P2.4** Keyframes per object with a camera-pose diversity penalty. Evidence: `benchmark/results/p2_4_dev.json`
- [x] **P2.5** Failure flags (small, thin, occluded, reflective, non-rigid), scene router, material pass (D-017). Evidence: `benchmark/results/p2_5_flags.json`, `p2_5_material.json`, `p2_5_router.json`, `p2_5_router_hard.json`
- [x] **P2.6** `Instances` from `SegmentationStage`, `point_cluster_path` is the fitted cluster (human decision 2026-10-29). Evidence: `benchmark/results/p2_6_instances.json`, `p2_6b_instances.json`, `p2_test_instances.json`

Exit criteria

- [x] Object recall 90% or more on the benchmark: 96.8% (P2.1); 92.9% on the test split through the stage
- [x] At most one identity switch per scene on average: 0.0 (P2.2)
- [x] Point clusters on the easy scenes have bounding boxes within 10% of measured size: test 6.0% median over 13 objects in s07 to s09 (`benchmark/results/p2_eval_segmentation_test.json`), dev 4.3%. Ticked 2026-10-29 by human decision with s10 unscored (clean-board origin not established)
- [x] Failure flags raised on all eight hard scenes (s17 to s24), with the per-scene rules in `docs/history/phases_0_2.md`. Evidence: `benchmark/results/p2_5_flags.json`

Open from Phase 2 (none blocks Phase 3):

- `eval` does not score identity switches (no track loader) and its flag rule is the old one-flag-per-scene check; the real checks are `benchmark/run_tracking.py` and `benchmark/report_flags.py`.
- No end-to-end stage test with fake hosted clients. The router's missing-object count has no `schemas.py` field.
- Easy-tier box failures are white-on-white parts the masks miss (in-ear box tab, mug handles).
- Revisit the cleared reflective objects (s17 mirror cube, s18 foil figure) after the first Phase 3 shape runs.
- The s05 and s10 clean-board origins are not established, so both scenes are unscored.

## 7. Phase 3: Shape completion and pose fusion (weeks 7 to 8)

Goal: a closed, textured mesh per object, registered to the measured point cloud in the
metric world frame. This is the highest-risk phase.

Tasks

- [x] **P3.1** Adapters for SAM 3D Objects and TRELLIS.2 behind one interface (TRELLIS.2 was
      dropped on 2026-10-30, D-018; its notes below are history). Draw several

      samples per keyframe, because both are generative. Where: `src/video2sim/models/`
      Status 2026-10-29: interface (`shape_model.py`) and the SAM 3D client with pose and mesh
      decoding (`sam3d.py`, `shape.sam3d` config) done and tested offline. The Modal worker
      (`sam3d_modal.py`, `modal` dependency added 2026-10-29 by human approval) is built and
      deployed (human go-ahead 2026-10-29), checkpoint `facebook/sam-3d-objects@2e73555`.
      First hosted run, s01 `track_003` keyframe 37, seed 0: 112 s with a cold start, 702k
      vertices; the placed mesh projected with our intrinsics overlaps the mask at IoU 0.72.
      The same mesh with the GLB turn kept scores 0.67 and without the PyTorch3D axis flip
      0.00, so the decoded conventions hold, though one flat box does not separate the GLB
      turn sharply. Open: more objects. TRELLIS.2 adapter written 2026-10-30
      (`models/trellis2.py`, `trellis2_modal.py`, `shape.trellis2` config, `shape.backend:
      trellis2`), tested offline only: the worker is deployed (2026-10-30, app `video2sim-trellis2`, weights on
      its volume, background remover stubbed out) and made its first mesh the same day (s01 `obj_003`, the earbud box, keyframe 37, seed
      423485915: 372 s with a cold start; `transformers` had to be pinned to 4.56.2). The raw
      mesh is 0.98 x 1.00 x 0.11 in the unit cube with z thin, so z-up holds, but it is a flat
      slab against a box of 116 x 86 x 45 mm: registration scored fit 0 and the stage fell back
      to the proxy hull. One sample on one hard object (white box on a white board), not a
      verdict. Full s01 run into `runs_trellis2/s01_b` (`benchmark/run_shape.py --backend
      trellis2`, 30 calls, 6 samples per object, one draw failed on an empty sparse structure
      and was skipped, so the worker and `draw_samples` now report that as `SampleFailed`),
      `benchmark/results/p3_1_trellis2_s01.json`: recall 1.0, median size error 6.4% against
      4.0% for SAM 3D on the same five objects, position 3.7 mm against 3.5 mm. Per object,
      TRELLIS.2 against SAM 3D: cube 5.1 / 3.9%, small cube 6.4 / 5.5%, phone box 7.1 / 3.6%,
      coffee can 5.1 / 4.0%, earbud box 14.8 / 12.6% (the earbud box fell back to the proxy
      hull, best fit 0.24, where SAM 3D reached 0.996). One scene, one seed set: not a
      bake-off result. Vertex colours added 2026-10-30 (the raw mesh had none; the worker now
      reads `base_color` from the nearest voxel, `shape.trellis2.colors`) and s01 rerun: the
      hue follows the photo for good draws, but some draws are far too dark and fit score
      cannot see it (phone box kept sample: mean RGB 1, 0, 0 against 65, 49, 31 in the image).
      A colour check in sample selection is open. It returns the
      raw mesh (no pose, unitless, up axis assumed z, UNVERIFIED), so the start rotation is a
      canonical guess and registration's yaw starts do the rest. Single view first,
      MV-SAM3D only as a P3.3 arm (`docs/research/sam3d_multiview.md`)
      Evidence: SAM 3D adapter and Modal worker, `benchmark/results/p3_6_dev_stretch.json` and `benchmark/results/p3_eval_shape_test.json`, commits 5401dec, 69b9d04; the TRELLIS.2 half was built, measured and dropped (D-018, `benchmark/results/p3_3_bakeoff_dev.json`, commit 509cf09).

- [x] **P3.2** Registration. SAM 3D's predicted layout is the starting pose. Refine a

      similarity transform against the fused point cloud and the masks in all views, with a
      one-sided term so the unseen back of an object is not penalized. Scale is constrained by
      the metric cloud, not trusted from the model.

  $$\min_{s,R,t}\; \frac{1}{|Y|}\sum_{y\in Y}\min_{x\in\mathcal{M}}\lVert y-(sRx+t)\rVert^{2} + \lambda\sum_f \mathcal{L}_{\text{mask}}\big(\pi_f(s,R,t;\mathcal{M}),\,M_f\big)$$

  Here $Y$ is the observed cluster, $\mathcal{M}$ the generated mesh, $\pi_f$ the rendering
  into frame $f$ and $M_f$ the SAM mask in that frame.

  Status 2026-10-29: `stages/registration.py` (`shape.registration` config) fits the cost
  with Levenberg-Marquardt in NumPy. $\mathcal{L}_{\text{mask}}$ is done without rendering:
  mesh samples outside the mask (other objects' masks allowed, for occlusion) plus the mesh's
  reach along 8 image directions against the mask's. Only the rotation of SAM 3D's layout is
  used as the start, with four yaw starts; scale and centre start from the cluster, since
  SAM 3D's translation and scale are in its own depth units. Target $Y$ is
  `point_cluster_path` (human decision 2026-10-29; the voted SfM points would need a schema
  field). Synthetic box, six views on a handheld arc: under 1 mm corner error and under 0.75%
  scale error, also with 5% strays, one occluded view or only the top face observed
  (`tests/test_registration.py`). First real sample, s01 `track_003` (SAM 3D, keyframe 37,
  12 views, 5 s): median mask IoU 0.95 (raw SAM 3D pose 0.72), model up within 1.2° of world
  up, size 102 x 84 x 42 mm against ground truth 116 x 86 x 45 (-12%, -3%, -6%). The P2.4
  primitive fit reads the same object 98 x 83 x 43 mm, while the other s01 objects read
  within 6%. Re-measured by hand 2026-10-29 at 118.8 x 86.4 x 45.9 mm (three readings each),
  so ground truth holds and both methods read the long side 13 to 17% short. They share the
  masks: likely the white box against the white board, a P2 segmentation issue to check.
  All five s01 objects (one seed each, points target), size error against ground truth:
  coffee can +6/+6/-4%, small cube +7/+4/+5%, cube +10/+6/-5%, earbud box -11/-3/-1%,
  phone box -1/+12/+32% (tilted 8°, so its height reads high). Median mask IoU 0.92 to 0.96
  for all. Registration only scales uniformly, so SAM 3D's proportions carry through: the can
  is 12% too wide for its height, the phone box 13% too wide for its length. Voted SfM points
  and the hull cluster give the same poses; the hull's filled interior drops the inlier
  fraction (0.29 to 0.75, points 0.84 to 0.98), so `fit_score` is not comparable across them.
  Resting prior added the same day: the model axis nearest to up is levelled and held within
  `upright_sigma_deg`, and an optional support height holds the lowest vertex
  (`contact_sigma_m`). Rerun, points target, with contact on the table plane: coffee can
  +7/+7/-4%, small cube +5/+3/+3%, cube +9/+6/-6%, earbud box -11/-3/-6%, phone box
  -3/+7/+2% (height was +32%), median mask IoU 0.92 to 0.96, lowest vertex within 1 mm of
  the table. Left: sizes run 3 to 9% large on all but the earbud box, from SAM 3D's
  proportions and loose masks. Open: weights are untuned first guesses, one sample per
  object, not yet wired into a stage.
  Evidence: commits abf530b, acc2a36; `benchmark/results/p3_6_dev_stretch.json` and `benchmark/results/p3_eval_shape_test.json`. Still open as noted above: untuned weights, one sample per object.

- [x] **P3.3** Bake-off, time-boxed to one week: SAM 3D, TRELLIS.2, and SAM 3D pose with

      TRELLIS.2 geometry, scored on the full benchmark by pose error, point-to-mesh distance
      and the Phase 4 stability pass rate. Keep one path and record it as a decision.
      Where: `configs/default.yaml` (`shape.backend`)

  Status 2026-10-30 (dev split only; the test split is for the gate): three arms on the nine
  dev scenes (s01 to s04, s06, s11 to s14), 42 ground-truth objects matched by all three,
  `benchmark/results/p3_3_bakeoff_dev.json` (`benchmark/compare_shape_arms.py`). Arms: SAM 3D
  (`shape.backend: sam3d`, the existing runs), TRELLIS.2 (`trellis2`, 6 draws per object, about
  230 hosted calls for the eight scenes beyond s01) and TRELLIS.2 geometry started from SAM 3D's
  layout (`hybrid`, no new calls: `models/hybrid.py` matches the two model frames by chamfer
  search over the 24 cube turns and a yaw, then carries SAM 3D's rotation across). Same
  registration, stretch and settings for all.

  | Paired, 42 objects | SAM 3D | TRELLIS.2 | Hybrid |
  | --- | --- | --- | --- |
  | Median size error | 4.5% | 8.7% | 8.4% |
  | Easy (19) / medium (23) | 4.6% / 4.3% | 6.0% / 20.5% | 4.9% / 16.2% |
  | Objects over 15% | 5 | 15 | 15 |
  | Proxy hulls | 1 | 20 | 11 |
  | Median position error | 6.1 mm | 6.0 mm | 6.9 mm |
  | Median orientation error | 0.99° | 1.01° | 1.13° |
  | Observed cluster to mesh, median | 1.2 mm | 1.3 mm | 1.5 mm |
  | Lowest size error (objects) | 26 | 12 | 4 |

  On the 22 objects where every arm kept a real mesh: size error 4.1% (SAM 3D), 8.7%
  (TRELLIS.2), 7.1% (hybrid); over 15%: 1, 7, 8; SAM 3D lowest on 17 of 22. So the gap is not the
  proxy hulls. TRELLIS.2 draws vary far more from seed to seed (flat slabs, near-empty meshes,
  failed draws), and its fit score finds a good draw for about half the objects: 20 of 42 fall
  back to the proxy hull, against 1 for SAM 3D. Starting from SAM 3D's pose rescues 9 of those
  20 from the hull: median size error 10.4% on them (hull 13.2%, SAM 3D 9.9%), with one at 97%. The point-to-mesh distance is observed cluster to registered mesh, since
  ground truth holds boxes and no surfaces; it is within 1 mm of equal for all arms and so does
  not separate them. The Phase 4 stability pass rate was not measured (needs Phase 4).
  Recommendation, for the human to decide and record: keep `sam3d` as the geometry path;
  TRELLIS.2 is behind on every size measure and not ahead on any other. Sample selection stayed
  on fit score alone (human decision 2026-10-30, no colour check). Not done: the test split,
  MV-SAM3D, and the Phase 4 stability measure. Fixes made on the way: `shape.parallel_samples`
  (hosted draws in flight at once), the worker now reports an empty sparse structure as a
  failed draw whether torch raises `RuntimeError` or `IndexError`.
  Evidence: `benchmark/results/p3_3_bakeoff_dev.json`, D-018, commits 3e531bd, 509cf09. Dev split only (the test split ran SAM 3D alone), no stability measure.

- [x] **P3.4** Pick the best sample per object by fit score. Below the threshold, fall back to

      a convex hull or oriented box of the observed points and set `PROXY_GEOMETRY`.
      Evidence: `benchmark/results/p3_6_dev_stretch.json` (1 proxy among the 43 objects on dev), `benchmark/results/p3_shape_checks_test.json` (5 proxies of 32 on test), commit abf530b.

- [x] **P3.5** Snap resting objects to the table: choose the stable face from the convex hull,

      align the up axis, remove the small gap or penetration.
      Evidence: `benchmark/results/p3_shape_checks_dev.json` and `p3_shape_checks_test.json`: lowest vertex of every resting mesh at the support height (68 meshes, max gap 0.0 mm), commit abf530b.

- [x] **P3.6** Emit `RegisteredObjects`.

  Status 2026-10-29 (P3.4 to P3.6): `stages/shape.py` (`ShapeStage`, config `shape`) runs the
  model on the first `keyframes_per_object` keyframes with `samples_per_keyframe` seeds,
  registers each sample to `point_cluster_path` and the masks of up to `max_views` frames,
  keeps the highest `fit_score`, and below `min_fit_score` writes the convex hull of the
  cluster with `PROXY_GEOMETRY` and `LOW_FIT`. Resting objects are held on the table during
  registration and then set down exactly; the stable face is the model axis nearest to up
  (`registration.level`), not a convex-hull face as written above. Meshes go to
  `shape/meshes/<id>.ply` in the object frame (volume centroid, or the hull's for an open
  mesh), every sample's fit to `shape/report.json`. Tests: `tests/test_shape_stage.py`.
  First run, s01 (`benchmark/run_shape.py s01_b`, 30 SAM 3D calls, 12 min): recall 1.0,
  median size error 3.8%, median position error 3.6 mm (after `video2sim eval` learned to
  align shape meshes to the board frame, as it already did for clusters), median orientation
  error 0.6° (phone box 0.8°, earbud box 0.4°). That is after a ground-truth fix confirmed by
  the human on 2026-10-29: the earbud box in s01 stands at yaw 90°, not 0°
  (`benchmark/scenes.csv`; it had scored 89.6°). Its long side still reads 13% short. Fit scores against the 8-point box outline sit at 0.97 to 1.00
  for most samples and separate them poorly.
  Dev split, easy and medium scenes (s01 to s04, s06, s11 to s14; 8 more runs, 240 SAM 3D
  calls, 2 to 17 min per scene), `benchmark/results/p3_6_dev.json`: recall 0.93 (42 of 45,
  the same misses as P2), median position error 6.1 mm, orientation 1.0°, size 6.4%
  (target 5%). Easy: size 5.5%, position 4.5 mm, orientation 0.9°. Medium: 8.1%, 7.4 mm,
  2.2°. Against the P2.4 primitives on the same 42 objects (`p2_eval_segmentation_dev.json`),
  median size error goes from 4.3% to 5.5% on easy and from 14.7% to 8.1% on medium. Meshes
  win where no primitive fits (s12 electronics box 93% to 8%, a mug 32% to 3%, the s11 shapes)
  and lose where the box or cylinder was already right (water bottle 7% to 33%, the s13 breakfast
  bag box 21% to 37%, s02 electronics box 3% to 18%, coffee cans 2-3% to 6-10%): registration
  scales uniformly, so SAM 3D's proportions carry through. One proxy (s13 electronics box,
  inlier fraction 0 for all six samples); its P2 box stands on the table through the breakfast
  box it sits on, so the stack is lost before shape (both methods 66%). Fit scores against
  the 8-point outline step in eighths, and the kept sample is often not the lowest cost.
  Every one of the 6 samples scored against ground truth (cached samples re-registered, same
  fits): median size error 5.0% if the best sample were always kept, 6.4% as picked, 8.6% for
  a typical sample, 15.8% for the worst. Fit score picks the best sample for 17 of 42 objects
  (rank correlation with size error -0.38), so better selection gains at most 1.4 points;
  proportions remain the larger error.
  Per-axis stretch added 2026-10-30 (`registration.refine_stretch`, `shape.registration.
  stretch_sigma`, `shape.stretch_samples`): after the similarity fit, the kept sample is
  refined once more with a scale along each model axis, against the same points and masks,
  with a prior on each log scale. Same cached samples, no hosted calls,
  `benchmark/results/p3_6_dev_stretch.json`: median size error 4.5% (easy 4.6%, medium 4.3%,
  target 5%), position 6.1 mm, orientation 1.0°, recall 0.93. 25 objects improve, 12 worsen.
  Prior width swept on the dev scenes only, median size error with stretch on the single best
  sample: 0.1 gives 5.1%, 0.3 gives 4.7%, 0.5 gives 4.5%, 1.0 gives 4.7%. Stretching the top
  2, 3 or 6 samples and reselecting is no better (4.6 to 4.7% at 0.5), so one sample it is.
  Registration also got exact Jacobians and a KD-tree: fits unchanged on s03, about 2 s a
  sample instead of 4. Still over 15%: s14 breadboard 75%, s13 electronics box 66% (proxy),
  s04 water bottle 29%, the white cuboid in s11 (29%) and s14 (21%). The size target is
  provisional and the settings are tuned on dev, so the test split may read higher.
  Open: a fit score that ranks samples, stacked clusters in P2, and the bake-off.
  Evidence: `benchmark/results/p3_eval_shape_test.json` (test split, run once 2026-10-30) and `p3_6_dev_stretch.json`, commit abf530b.

Exit criteria

- [x] Median position error 1.5 cm or less and orientation error 10 degrees or less on easy
      and medium scenes. Evidence: dev 6.1 mm and 1.0° (`benchmark/results/p3_6_dev_stretch.json`);
      test, run once: easy 3.9 mm and 0.71° (13 and 7 objects), medium 4.6 mm and 0.38° (2 and 1
      objects, s16 only) (`benchmark/results/p3_eval_shape_test.json`)
- [x] Bake-off result written up and one geometry path chosen. Evidence: P3.3 status above,
      `benchmark/results/p3_3_bakeoff_dev.json`, D-018 (SAM 3D)
- [x] Every object has a registered mesh or an explicit low-confidence fallback. Evidence:
      `benchmark/results/p3_shape_checks_dev.json` (43 of 43, 1 proxy) and
      `benchmark/results/p3_shape_checks_test.json` (32 of 32, 5 proxies), every proxy flagged
      `PROXY_GEOMETRY` and `LOW_FIT`
- [x] Registered meshes resting on the table show no visible gap or penetration. Evidence: same
      two files, 68 resting meshes, largest gap or penetration 0.0 mm (tolerance 2 mm). This
      checks the meshes against the stage's own support height, which the position error above
      (4.4 mm median on test, z included) tests against the ground truth

Evidence:

* Gate run 2026-10-30, test split, once, nothing tuned on it: SAM 3D through `benchmark/run_shape.py`
  on s07 to s10, s16, s19, s20 and s23 (about 190 hosted calls), scored with `video2sim eval
  --split test --stage shape`: recall 0.93 (26 of 28), median size error 5.4% (13 easy objects
  5.1%, 11 hard 14.4%; the provisional target is 5%), position 4.4 mm, orientation 0.71°.
* Not covered, so read the gate with these in mind: **s15 (medium) was never run** through
  Phase 2, so it has no shape result; s10 is unscored (clean board, origin not established); the
  medium tier on test is therefore only s16 (4 objects, 2 matched, one at 64% size error). The
  size target is missed on test by 0.4 points and is not a Phase 3 exit criterion. The Phase 4
  stability pass rate was not measured for the bake-off (D-018).
* Largest test misses: s23 (one object, proxy hull), s19 small wooden toy 114%, s16 coffee can
  64%, s20 scissors 44% and alarm clock 39%.

## 8. Phase 4: Physics readiness and material properties (weeks 9 to 10)

Goal: every asset is simulation-ready and carries mass, inertia and friction values with
stated uncertainty.

Tasks

- [x] **P4.1** Mesh repair: remove floaters, fill holes, make each mesh watertight and

      manifold, decimate to a triangle budget. Validators block export on failure.
      Evidence: `src/video2sim/stages/mesh_repair.py`, `tests/test_mesh_repair.py`; on the 5 raw
      SAM 3D meshes of `runs/s01_b` (410k to 1.4M faces) all 5 pass the checks after repair and
      decimation to 20000 faces, volume within 0.7% (`benchmark/results/p4_1_mesh_repair.json`,
      `benchmark/run_mesh_repair.py`). Only s01 has raw meshes on disk; the exit criterion over all
      exported meshes is checked in P4.8.

- [x] **P4.2** Collision geometry by convex decomposition (CoACD or V-HACD) with a cap on parts

      per object. Use a box, cylinder or sphere primitive where the object is close to one.
      Evidence: `src/video2sim/stages/collision.py`, `tests/test_collision.py`; on the 5 raw SAM 3D
      meshes of `runs/s01_b` (thin-walled shells, walls 0.4 to 1.2% of the object size) 4 pass
      `collision_loads` and `collision_fit` (1 box, 3 CoACD with 16 parts) and 1 fails fit
      (8 parts, tightness 0.12 against a limit of 0.05), so export is blocked for it
      (`benchmark/results/p4_2_collision.json`, `benchmark/run_collision.py`). CoACD takes 12 to
      78 s per object. The fit check is surface-based because a volume ratio fails on shells.
      Not yet run on registered, metric meshes (none on disk).

- [x] **P4.3** Build `configs/material_table.yaml`: density, friction and restitution ranges per

      material class, every value with a source.
      Evidence: `configs/material_table.yaml`, `docs/research/material_sources.md`,
      `tests/test_material_table.py`; 14 classes. Amended by D-019: values with no source are
      listed in `unsourced` as wide placeholders (density 6 of 14 classes, friction 13, restitution
      12), and only an unsourced density raises `Flag.LOW_CONFIDENCE_MATERIAL`.

- [x] **P4.4** Material inference. A vision-language model sees keyframe crops and returns

      structured fields only: category, material class, solid or hollow, confidence. Code
      maps the class to numbers through the table. The classification half already runs in
      Phase 2 (D-017, `stages/material_pass.py`, plus a surface-finish field); this task is
      done when the numbers half exists and the class list is final (P4.3).
      Evidence: `object_material` in `src/video2sim/stages/physics.py`, `tests/test_physics.py`;
      the 14 classes of P4.3 are the final list. On 56 matched dev objects (12 scenes, s05 has no
      ground-truth origin) the class agrees with the inventory on 40 and hollow on 38
      (`benchmark/results/p4_4_material_numbers.json`, `benchmark/run_material_numbers.py`), after
      the four cardboard packages obj_002, 003, 014 and 029 were marked hollow in the inventory
      by human decision (2026-10-31, ground truth regenerated, hollow fields only). Coffee can
      obj_004 is open: inventory cardboard and not hollow, model steel and hollow.

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
| D-017 | Material pass pulled forward (2026-10-28, human decision). The classification half of P4.4 (category, material class, surface finish, solid or hollow, confidence) runs in Phase 2 as one hosted VLM request per object on its best 2 keyframe crops (`segmentation.material_pass.keyframes`), and it also raises `Flag.TRANSPARENT_OR_REFLECTIVE` (P2.5). The class list in `configs/default.yaml` is provisional, made from the materials seen on the dev scenes, with no numbers; P4.3 completes the list and the values with sources, and the mapping to numbers stays in P4.4. Reason: the geometry-only gate cleared a mirror cube and a foil figure, and asking every object is cheaper than two separate calls | Decided | Week 5 | `src/video2sim/stages/material_pass.py`, `benchmark/run_material_pass.py` |
| D-018 | Shape stage geometry path: SAM 3D Objects only (`shape.backend: sam3d`). TRELLIS.2 is dropped from the project (human decision 2026-10-30, after the P3.3 dev-split bake-off). Evidence: `benchmark/results/p3_3_bakeoff_dev.json`, 42 objects matched by all three arms: median size error 4.5% for SAM 3D, 8.7% for TRELLIS.2, 8.4% for TRELLIS.2 geometry on SAM 3D's pose; objects over 15%: 5, 15, 15; proxy hulls: 1, 20, 11. Removed: `models/trellis2.py`, `models/trellis2_modal.py`, `models/hybrid.py`, `shape.trellis2`, their tests. Kept: the results, `benchmark/compare_shape_arms.py`, `shape.parallel_samples`, `SampleFailed`. Consequences: `nvdiffrast` and `nvdiffrec` (research use only, D-005) are no longer in the pipeline; the DINOv3 and RMBG-2.0 licenses no longer apply; the SAM License terms remain. Not done: the test split and the Phase 4 stability measure were not run for the bake-off. `GeometrySource` in `schemas.py` still lists `trellis2` (schema change left for a human). The Modal app `video2sim-trellis2` and its weights volume are still deployed | Decided | Week 8 | Commands to remove them are in the 2026-10-30 changelog row |
| D-019 | Unsourced friction and restitution are accepted as wide educated placeholders (human decision 2026-10-31). Every one of the 14 classes has at least one unsourced value, so flagging on any of them would flag every object and say nothing. `Flag.LOW_CONFIDENCE_MATERIAL` is raised only for an unsourced density or a low model confidence. The benchmark records mass only, so P4.6 cannot narrow friction or restitution | Decided | Week 9 | `configs/material_table.yaml`, `src/video2sim/stages/material_table.py` |
| D-020 | Optional tags-on-table layer on the table plane fit (human decision 2026-10-31, `reconstruction.world_frame.tags_on_table`, on by default). On s05 RANSAC settled on the object tops, 3 to 7 cm above a bare white table, which cut the objects off in lifting and sent two of them to proxy hulls, with no flag. A plane more than 15 mm from a tag corner is refitted on the points near the plane through the tags; a refit that cannot hold all the tags is dropped and `tags_off_table` is raised (s24, 23 cm). Disable it for scenes whose markers are not on the table plane. Dev scenes: good ones up to 8.2 mm, s05 64 mm before and 1.1 mm after; the test split was not measured | Decided | Week 9 | `src/video2sim/stages/world_frame.py`, `src/video2sim/stages/reconstruction.py` |

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

6. **Reconstruction back ends (from P1.1).** On s16 video_a, pycolmap is the most accurate
   back end; VGGT is about 20x faster on a GPU with twice the tag distance error. Verdict
   (2026-10-28): VGGT becomes an opt-in fast mode, `reconstruction.backend: vggt`, after the
   MVP, with an automatic pycolmap fallback when a gate fails; measure the fallback rate on the
   full dev set first; non-commercial checkpoint. D-016 stands. Results and conditions:
   `docs/history/reconstruction_experiments.md`; branches `experiment/vggt-modal` and
   `experiment/colmap-cuda-modal` stay unmerged.

6b. **pycolmap optimisation.** Sequential matching is the default (1.3x to 4.2x faster on five
   videos, quality inside the proposed limits); fewer frames or a smaller image lose the tag
   distance gate; the global mapper stays off. Open: rerun the full dev set and test split
   through the P1.6 evaluation, and put the proposed quality limits in `configs/targets.yaml`
   (human). Tables and steps: `docs/history/reconstruction_experiments.md`.

7. **VGGT geometry for SAM 3D (from P2.3 and the shape stage).** Experiment: improve SAM 3D
   output by giving it a pointmap and by registering its mesh against dense geometry, instead
   of the sparse pycolmap cloud (10k to 26k points, weak on glossy surfaces).

   Idea: run VGGT on the registered frames for per-frame dense depth and pointmaps with a
   confidence per pixel. Pass the pointmap of each object keyframe to SAM 3D (its optional
   pointmap input, format to be checked in its docs). Keep the pycolmap poses and the marker
   scale as the reference: fit the VGGT geometry to them so the metric scale and the world
   frame do not change. Dense surface points would also replace the visual hull of P2.3,
   whose boxes read about 35% too large from a 127 degree handheld arc even with perfect
   silhouettes.

   Measure against the MVP, with the same benchmark scenes: SAM 3D mesh size and pose error
   with and without the pointmap, cluster box error with dense points, VGGT pose and scale
   error against pycolmap, and runtime and memory per video. Pass only if the metrics gain
   without breaking the reconstruction gates. Needs a GPU host, weights and a license check
   (`docs/licenses.md`); `benchmark/modal_vggt.py` is an untracked first attempt.

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
| 2026-10-30 | P3 gate: test split run once (SAM 3D, 8 of 9 scenes, s15 never ran through Phase 2); all four exit criteria ticked with evidence; Phase 3 Done and the gate Passed, Phase 4 is the current focus |
| 2026-10-30 | D-018: TRELLIS.2 dropped after the P3.3 dev bake-off; SAM 3D is the only shape model. Cleanup of the hosted side, not done by an agent: `uv run modal app stop video2sim-trellis2` and `uv run modal volume delete video2sim-trellis2-weights` |
| 2026-10-30 | D-005: DINOv3 License (TRELLIS.2's image encoder) read; no commercial block, acknowledgment in publications, hosting on Modal not addressed. RMBG-2.0 (BiRefNet, CC BY-NC 4.0, gated) is not loaded: the TRELLIS.2 worker stubs the background remover because inputs carry the mask as alpha |
| 2026-10-26 | P0.9: green CI run confirmed on GitHub; Phase 0 gate passed with the container criterion deferred; Phase 1 is the current focus |
| 2026-10-26 | D-016: pycolmap on CPU is the MVP reconstruction back end; VGGT, MASt3R and the CUDA build moved to section 12 |
| 2026-10-27 | P1.2, P1.3: scale from tag edge and world frame on a reference tag pass on 9 dev grid videos; results in `benchmark/results/p1_world_frame_dev.json` |
| 2026-10-27 | P1.5: `WorldModel` emitted by `ReconstructionStage` after `CaptureStage`; one-video run in `benchmark/results/p1_5_worldmodel_s01_b.json` |
| 2026-10-27 | P1.1 ticked by human decision with the VGGT and MASt3R comparison deferred (section 12, D-016) |
| 2026-10-27 | P1.4: capture and reconstruction quality gates, saved results in `benchmark/results/` |
| 2026-10-27 | P1.6: gate evaluated on 20 dev videos with the caliper tag edge (100.26 mm); P1.2 caliper note updated; Phase 1 exit criteria ticked with caveats in the P1.6 entry |
| 2026-10-27 | P1.6: test-split run on 16 videos; s16 marker-distance miss accepted as a warped-marker input defect by human decision; s10 clean-board not run; results in `benchmark/results/p1_world_frame_test.json` |
| 2026-10-28 | Section 12 item 6: s16 video_a table of pycolmap (CPU, GPU), VGGT and VGGT with tag refinement, accuracy and process times, gathered from the two experiment branches |
| 2026-10-28 | Section 12 item 6: P1.1 experiment closed; VGGT added as a post-MVP opt-in fast mode with automatic pycolmap fallback; item 6b added for pycolmap optimisation |
| 2026-10-28 | Section 12 item 6b step 1: local pycolmap profile on five videos; exhaustive matching is 59 to 80% of the time above 28 frames |
| 2026-10-28 | Section 12 item 6b steps 2 to 5: sequential matching is 1.3 to 4.2x faster on five videos with quality inside the proposed limits and is the new default (`reconstruction.sfm.matcher`); global mapper, fewer frames, smaller images, fewer features and lighter BA tried and not adopted |
| 2026-10-28 | P2.4: keyframe selection per object with a camera-pose diversity penalty, `benchmark/results/p2_4_dev.json` |
| 2026-10-28 | P2.5 and P4.4: D-017 pulls the VLM material classification forward as a single per-object pass that also raises the reflective flag; geometry kept as a cross-check, `segmentation.material_pass` config added, no hosted run yet |
| 2026-10-29 | Phase 2 cluster box criterion ticked by human decision (test split 6.0% median on s07 to s09, s10 unscored); Phase 2 gate passed; Phase 3 is the current focus |
| 2026-10-29 | Cleanup before Phase 3: evidence narratives of Phases 0 to 2 and the section 12 reconstruction experiments moved verbatim to `docs/history/`; P2.1 discovery code removed (eef52b6) |
| 2026-10-31 | P4.3 ticked under D-019: friction and restitution placeholders accepted, only an unsourced density flags an object; Phase 4 set to In progress |
| 2026-10-31 | P4.4 ticked: class to numbers mapping in `stages/physics.py`, checked against the inventory on 56 dev objects |
| 2026-10-31 | Inventory: obj_002, obj_003, obj_014 and obj_029 marked hollow by human decision; ground truth regenerated, only the 23 hollow fields changed |
| 2026-10-31 | D-020: tags-on-table layer on the plane fit; s05 rerun, all 4 objects now SAM 3D with fit 0.95 to 0.98 and `lift.max_extent_m` back at 1.5 |
