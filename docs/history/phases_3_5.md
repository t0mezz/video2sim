# Phases 3 to 5: full task and evidence record

Moved verbatim from `PLAN.md` on 2026-11-03, to keep the working plan short. `PLAN.md` keeps the
tick, the evidence link and a short result per task, plus every exit criterion with its state;
this file keeps the reasoning, caveats and numbers behind them. Section numbers, `section 12`
item numbers and `docs/history/phases_0_2.md` references below are those of `PLAN.md` on that
date. Decisions (`D-0xx`) now live in `docs/decisions.md`. Code paths and branch names are as
they stood then (HEAD b967bc0 plus uncommitted work).

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
      `collision_loads` and `collision_fit` (1 box, 3 CoACD with 16 parts) and 1 failed fit
      (8 parts, tightness 0.12 against a limit of 0.05). Superseded by D-022: all 5 pass (2 boxes,
      1 cylinder, 2 CoACD with 16 parts) (`benchmark/results/p4_2_collision.json`,
      `benchmark/run_collision.py`). CoACD takes 12 to 99 s per object. The fit check is
      surface-based because a volume ratio fails on shells. On the 61 registered, metric meshes
      of the 13 rigid dev scenes all pass (`benchmark/results/p4_7_settling.json`).

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
      obj_004 is settled by human check (2026-10-31): a cardboard can with a metal bottom, about
      80% filled, so the inventory stays cardboard and not hollow. The model says steel and hollow;
      the filling cannot be seen, the wall can.

- [x] **P4.5** Mass and inertia from the repaired mesh under uniform density, with a hollow

      correction. Store low, nominal and high.
      Evidence: `src/video2sim/stages/mass.py`, `tests/test_mass.py` (box, cylinder, sphere, voxel
      block and s^5 scaling against closed forms), `docs/research/mesh_volume.md`,
      `benchmark/results/p4_5_mesh_volume.json`, `benchmark/results/p4_5_mass.json`
      (`benchmark/run_mass.py`). Mesh volume is a median 0.13 of true on shells, so the body is a
      voxel fill after sealing gaps up to 4% of the longest side: 1.01 of true, 32 of 39 within
      20%, and the mouth of a cup stays open. Hollow correction `f` is one dev-measured value for
      all classes (0.024, 0.22, 0.61) until P4.6, in `configs/material_table.yaml`. On 56 dev
      objects 14 are within 30% (4 without the correction), in sample because `f` came from the
      same objects, against a 70% exit criterion. Solid answers are the gap: 1 of 21, median 2.4
      times too heavy. Truth is inside low to high on 31 of 56. `MassEstimate.center_of_mass_m`
      is left for P4.8 to move the origin to (median offset 1.7 mm). Inertia has no measured truth.

  $$m = \rho\,V_{\text{eff}}, \qquad \mathbf{I} = \rho \int_V \left(\lVert\mathbf{r}\rVert^2 \mathbf{1} - \mathbf{r}\mathbf{r}^{\top}\right) dV$$

- [x] **P4.6** Calibrate on the weighed benchmark objects: fit per-material corrections (for

      example the wall fraction of hollow containers) and report mass error before and after.
      Evidence: `benchmark/fit_fill_fractions.py`, `benchmark/results/p4_6_fill_fit.json`,
      `benchmark/results/p4_6_mass.json`, `configs/material_table.yaml`, `tests/test_physics.py`,
      `docs/research/mesh_volume.md`. A fill fraction per class and solid-or-hollow answer, from 3
      distinct objects, else one global value per answer. Only 4 pairs qualify; the 56 rows are 35
      distinct objects. Within 30%, leave one object out: 4 of 56 uncorrected, 17 with the global
      values, 20 (36%) per class; 26 (46%) in sample. The 70% exit criterion is not met. Friction
      and restitution stay as placeholders (D-019). No test split was run.

- [x] **P4.7** Scene settling: resolve interpenetrations, drop the scene under gravity in the

      simulator, compare final poses with the observed ones.
      Evidence: `src/video2sim/stages/settling.py`, `tests/test_settling.py` (rest, free fall,
      slope, tipping and overlaps against closed forms), `benchmark/run_settling.py`,
      `benchmark/results/p4_7_settling.json`. MuJoCo stands in for Isaac Sim (D-021), so the result
      is provisional. All 61 objects of the 13 rigid dev scenes load and run, and all pass every
      collision check after D-022. 10 of 13 scenes (77%) settle within 2 cm over 5 s against the
      95% exit criterion, and the same 10 in all five cases (nominal, low and high friction, low
      and high mass), so those ranges do not decide it. Each failing scene has one object: the
      s03 basketball rolls (2.8 cm), because nothing resists rolling and no sourced value is
      available to add; s13 obj_001 was placed on the table through the box beside it, and
      separation moves it 4.4 cm (one of the 3 objects of that scene is also missing); s17
      obj_004, a chocolate egg standing on end, falls over (5.7 cm, 110 degrees). The s04
      basketball passes since it became a sphere primitive under D-022, which is not a fix for
      rolling. No test split was run.

- [x] **P4.8** Emit `SimAssets`.
      Evidence: `src/video2sim/stages/physics_stage.py`, `tests/test_physics_stage.py`,
      `benchmark/run_physics.py`, `benchmark/results/p4_8_physics.json`. All 13 rigid dev runs
      finish the stage (61 assets, all valid). The origin moves to the center of mass with the new
      `SimAsset.T_world_from_object`, and vertex colors are carried over from the registered mesh
      because repair drops them (D-023). `physics/settling.json` gives 10 of 13 scenes (77%) in
      every case; the failing objects are those of P4.7, now measured at the center of mass
      (s17 obj_004 5.9 cm). `video2sim eval` reads the masses: 29 of 56 within 30% (52%) under
      D-024, in sample because the fill fractions were fitted on these objects, with 40 of 61
      objects taking the model's mass and none left without an answer; size error 0.0505 against
      0.05. Settling under D-025 is in the exit evidence below. No test split was run.

Exit criteria

- [x] 100% of exported meshes pass the watertight, manifold and collision-load validators
- [ ] Mass within 30% for 50% of benchmark objects (D-025)
- [x] 95% of scenes settle with maximum drift of 2 cm in 5 s, round objects not counted (D-025)
- [x] Every property in the export carries a low, nominal and high value

Evidence: gate run once on the test split, `benchmark/results/p4_gate_test.json` and
`benchmark/results/p4_gate_eval_test.json` (8 rigid scenes, s23 left out as deformable, code
68e06bf, nothing tuned afterwards).

* Meshes: 35 of 35 test objects pass, 61 of 61 on dev.
* Mass: **not met, 14 of 31 (45%) on test against 50%**; 29 of 56 (52%) on dev. s10 has no scored
  masses (clean board). Pooled, 43 of 87 (49%).
* Settling: 8 of 8 test scenes in every case, 13 of 13 on dev, in MuJoCo (D-021) with round objects
  not counted (D-025). Largest counted drift on test 1.3 cm. The two round objects on test, a plate
  and a container, did not move, so ignoring them changed nothing there.
* Ranges: friction, restitution, mass and density carry low, nominal and high, and inertia follows
  from mass (D-023).
* The gate run failed on mass, and the human marked the gate Passed (D-026). The target stays 50%
  and the box stays unticked, because no number was changed after seeing the test.
  s15 now has a shape result, and its fifth object is missing (4 tracks of 5).

## 9. Phase 5: Export, simulator validation, demo (weeks 11 to 12)

Goal: one command takes a video to a USD and URDF scene package, validated in MuJoCo, with a QA
report and a basic manipulation demo.

Note (D-027): this phase validates in MuJoCo on the Mac. USD loading in Isaac Sim and PhysX
behavior stay unverified until the verifier on the GPU workstation runs (section 12 item 8).

Tasks

- [x] **P5.1** USD writer: Z-up, meters, kilograms; per object a transform with rigid body and

      mass properties, visual mesh with `displayColor` from its vertex colors, convex collision meshes, physics materials
      from the friction and restitution values; table and ground plane; physics scene.
      Checked by reading the USD back against `SimAssets`. Settling stays in MuJoCo (D-027).
      Evidence: `src/video2sim/stages/usd_writer.py`, `tests/test_export.py`,
      `benchmark/results/p5_export_dev.json`: 61 objects of the 13 rigid dev scenes read back, none
      skipped. One infinite plane is both table and ground, since nothing says how far below a
      floor would be. The friction combine mode is a PhysX attribute, unverified until the Isaac
      Sim verifier (section 12 item 8), and so is the 255-vertex limit in `physics.collision`.

- [x] **P5.2** URDF writer per object (inertial, visual, collision), loaded in MuJoCo to

      cross-check units, axes and inertia.
      Evidence: `src/video2sim/stages/urdf_writer.py`, `tests/test_export.py`,
      `benchmark/results/p5_export_dev.json`: mass, principal inertia and collision bounds match
      on 61 objects, and settling a scene built from the URDF files gives the same stable result
      as the physics stage for every object of the 13 scenes.

- [x] **P5.3** QA report per scene: status, confidence and fallbacks per object, stability

      result, side-by-side renders of the real frame and the simulated scene from the same
      camera pose.
      Evidence: `src/video2sim/stages/qa_report.py`, `src/video2sim/stages/scene_render.py`,
      `tests/test_export.py`; `export/qa_report.html` in each of the 13 dev runs of
      `runs_instances/`. It shows the mass source and range (D-026). 2 scenes read ok and 11
      review, because any flag marks an object. Renders are a painter's drawing of the settled
      scene with vertex colors and no table.

- [x] **P5.4** MuJoCo check: scripted pick-and-place with a manipulator on one object from

      the twin, repeated across the stored friction and mass ranges. Stretch: a short RL run.
      Evidence: `src/video2sim/pick_place.py`, `tests/test_pick_place.py`,
      `benchmark/run_pick_place.py`, `benchmark/results/p5_4_pick_place.json`, `demo/pick_place.mp4`
      in the runs. Panda from MuJoCo Menagerie at a pinned commit (`scripts/fetch_panda.sh`),
      success as in D-029. 7 of the 13 rigid dev scenes have an object that fits the rules (a real
      mesh, no round flag, fits the gripper, free room); all 7 pass the nominal case and 5 pass all
      five cases (s01, s02, s04, s05, s24). s11 fails four of five on placing, by about 14 cm, and
      s17 fails on low friction. These two were not investigated. Control is open loop, so this
      shows the twin is graspable in simulation, not that a real robot would succeed. The RL
      stretch was not attempted.

- [x] **P5.5** Command line (`video2sim run video.mp4 --out scene/`), README

      with the capture guide.
      Evidence: `src/video2sim/pipeline.py`, `tests/test_cli_run.py`, README section "Make a
      scene", `benchmark/results/p5_5_fresh_s01_a.json`. One fresh run, s01 `video_a`, dev split:
      exit 0, 8.7 min, one command and no manual steps; 5 of 5 objects, size 4.1%, position 2.6 mm,
      orientation 0.33°, 3 of 5 masses within 30%, stable in every case, QA status ok, with three
      dropped tracks shown as scene flags. One video on an easy scene is not a benchmark. The
      manifest gives commit 4d50a8d, but the run used uncommitted export code on top of it, and
      `eval` still reports manual steps as `not_measured`. `video2sim eval` also measures stability
      and asset validity now (`benchmark/results/p5_eval_dev.json`, 12 dev scenes: stability 100%,
      validity 100%, size error 5.05%, mass 52%, s05 unscored).

- [x] **P5.6** Final benchmark run on the test split and a write-up of every target, met or

      missed, with causes.
      Evidence: `docs/final_benchmark.md`, `benchmark/results/p5_6_eval/`,
      `benchmark/results/p5_6_pick_place_test.json`, code 21d34de.

Exit criteria

- [x] Final benchmark run meets every target, or the misses are documented with causes. Evidence: `docs/final_benchmark.md`; size 5.7% against 5% and mass 29% against 50% are missed
- [ ] USD scenes read back with the units, axes, mass and inertia of `SimAssets`; URDF loads in MuJoCo (D-027)
- [x] Scripted pick-and-place demo in MuJoCo recorded on at least three scenes. Evidence: `benchmark/results/p5_4_pick_place.json`, `demo/pick_place.mp4` in the 7 dev runs with a chosen object
- [ ] A new person can reproduce a scene from the README alone

Evidence: final run in `docs/final_benchmark.md`. Open: the last exit criterion needs a new person,
so the gate stays Open (confirmed 2026-11-03; carries into the section 12 item 8 verifier follow-up, human owns finding the new person).
