# PLAN.md

Scope, schedule and status of the Video2Sim MVP. This file is the working copy of the
roadmap: update it as work happens. Numbers that gate a decision live in
`configs/targets.yaml`; if this file and that file disagree, the config wins.

## Status

Last updated: 2026-10-29. Start date: not set (decision D-002).

| Phase | Weeks | Focus | Status | Gate |
| --- | --- | --- | --- | --- |
| 0 | 1 to 2 | Foundations, benchmark, evaluation harness | Done | Passed |
| 1 | 3 to 4 | Reconstruction, metric scale, world frame | Done | Passed |
| 2 | 5 to 6 | Segmentation, tracking, 3D lifting | Done | Passed |
| 3 | 7 to 8 | Shape completion and pose fusion | Not started | Open |
| 4 | 9 to 10 | Physics readiness, material properties | Not started | Open |
| 5 | 11 to 12 | Export, simulator validation, demo | Not started | Open |

Status values: Not started, In progress, Done. Gate values: Open, Passed, Failed.
**Current focus:** Phase 3.

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

- [x] **P2.3** Lift masks to 3D. A point is kept only when several views agree. Remove

      outliers, hands and arms.

- [x] **P2.4** Keyframe selection per object: score mask area, occlusion, border contact and

      sharpness. Store the top 3 to 5. Evidence: `benchmark/results/p2_4_dev.json`, see the
      P2.4 entry below. Frames from similar camera poses are penalised.

- [x] **P2.5** Failure flags: unstable depth (likely transparent or reflective), thin, tiny,

      heavily occluded, non-rigid. The reflective flag comes from the material pass (D-017) with the SfM
      geometry as a cross-check. Evidence: `benchmark/results/p2_5_flags.json`, see the P2.5
      entry below (final run, 18 runs).

- [x] **P2.6** Emit `Instances`. Drops tracks that have no point cluster (decision 2026-10-29) and carries the P2.5 flags, the router count check and the P2.4 keyframes. Evidence: `benchmark/results/p2_6_instances.json`, see the P2.6 entry below.

Exit criteria

- [x] Object recall 90% or more on the benchmark
- [x] At most one identity switch per scene on average
- [x] Point clusters on the easy scenes have bounding boxes within 10% of measured size. Evidence: `benchmark/results/p2_eval_segmentation_test.json` (test split, 6.0% median over 13 objects in s07 to s09; s10 unscored, clean-board origin not established), dev `benchmark/results/p2_eval_segmentation_dev.json` (4.3%), see the cluster box gate entry below. Ticked 2026-10-29 by human decision with s10 unscored
- [x] Failure flags raised on all eight hard scenes (s17 to s24). Reflective: every object labelled reflective in s17 to s20 is flagged (s18: the chocolate santa only; s19: the alarm is reflective on its screen only, so either result passes). Deformable: every tracked object with a point cluster in s21 to s23 is flagged (those scenes hold deformable objects only; tracks without a cluster are dropped when `Instances` is emitted, P2.6), except the s21 basketball, which is borderline (inflated shell) and counts as a pass either way. s24 additionally needs at least one track. Test scenes s19, s20 and s23 are reported separately and not used for tuning (changed 2026-10-28, human decision) Evidence: `benchmark/results/p2_5_flags.json`, P2.5 final run entry below (decision 2026-10-29: clustered tracks only).

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
- P2.3 lifting: `benchmark/results/p2_3_dev.json` (12 dev scenes) and `benchmark/results/p2_3_test.json` (s16_a), made by `benchmark/run_lift.py`. No depth maps exist, so there are two independent lifts of the same tracks, kept apart on purpose (decision for Phase 3 below). Viewer: `benchmark/view_instances.py RUN_ID --gt` (hull, points or both).
  - **hull** (`stages/lift.py:lift_tracks`): a visual hull by voting. A 5 mm voxel grid on the table is projected into 40 frames; a voxel goes to the track whose mask holds it in at least 90% of the views that see it (at least 5). A view where a track has no mask is skipped for that track. Overlaps go to the higher agreement, only the largest connected group is kept, and a track with under 20 voxels or one reaching the 35 cm grid top is dropped and listed in `segmentation/lift.json`.
  - **points** (`lift_points`): the sparse SfM points (10k to 26k per scene) with the same vote and the same view rules, minus points within 3 mm of the table plane; at least 8 points to keep a track.

  | 12 dev scenes, 59 objects | hull | points |
  |---|---|---|
  | clusters matched to ground truth | 53 | 54 |
  | median box size error, easy tier | 36.4% | 18.7% |
  | median box size error, all matched | 34.7% | 20.2% |
  | objects within the 10% gate | 5 of 53 | 16 of 54 |
  | time per scene | 0.4 to 7 s | 0.1 s or less |

  s16_a (test split, run once, not tuned on): both match 3 of 4 objects. Box errors hull 6%, 46%, 37%; points 2%, 23%, 20%.
  - **Exit criterion (cluster box within 10%): not met by either.** Hull 36% and points 19% median on the easy tier.
  - Why the hull is loose, checked on s01: perfect ground-truth silhouettes through the same code still give a 0.074 m hull for a 0.055 m cube. The `_b` camera paths cover only 45 to 130 degrees of azimuth (s16_a covers 353), and a silhouette-only hull is loose in directions no camera viewed. The SAM masks are not the cause (IoU 0.96 with the projected ground-truth box). Lens distortion is modelled; erosion and stricter agreement gave 27 to 43% on five dev scenes and were not kept.
  - Why points are tighter but not tight: they lie on visible surfaces, so a box never grows past the object, but plain or glossy objects have few (two s16_a tracks have no agreeing point) and there is no occlusion test, so a point in front of an object's edge can pass the vote. s12 loses two tracks (the hull loses one), and s13 matches 2 of 3 objects (the hull 1).
  - Tried in a scratch script, not adopted: keeping only the points inside the hull changed nothing (both come from the same mask votes). Clipping the hull to the box of its points, with the hull alone under 30 points, gave a 26% median and 10 objects within 10%, with 54 matched.
  - Hands and arms are not detected as such. They move between views, so their masks agree on almost no voxels and the track is dropped with a reason. None of the 13 scenes has a hand track, so this is untested. `video_a` orbits were not run except s16_a; the `_b` paths of the other scenes are short arcs, and whether a fuller `video_a` orbit tightens the hull is untested.
  - Dev scenes were used to choose `min_agreement` (0.9); s16_a was not.
  - **Decision for Phase 3 (human, 2026-10-28):** keep both clusters and compare them as registration targets in P3.2. The fit compares observed surface points with the mesh surface, which favours the points; the hull gives a filled volume and seeds position. Mass scales with size cubed (AGENTS.md), so a 35% size bias matters. The 10% criterion stays unticked until that comparison or a human change to it.
- P2.4 keyframes: `benchmark/results/p2_4_dev.json`, made by `benchmark/run_keyframes.py` from the existing dev runs (13 runs, 11 with tracks, 58 tracks); code in `stages/keyframes.py`, config `segmentation.keyframes`. Contact sheets of the picks were checked by eye for s01_b and s13_b (`--preview`). Not yet wired into `Instances` (P2.6).
  - **Score.** Per frame, a product of four scores in [0, 1]: mask area against the object's 90th percentile, one minus the mask share in a 2% border band, one minus the share of a 6 px ring around the mask that lies in another track's mask, and Laplacian sharpness inside the mask against its 90th percentile (exponent 0.5). A frame below quality 0.15 is never chosen.
  - **Pose penalty.** Greedy: after the best frame, each candidate keeps `1 - 0.8 * s` of its quality, where `s` is the largest Gaussian (sigma 25 deg) of the angle between its viewing direction and a chosen one. The direction is object centre (median of the point cluster, else the hull) to camera centre, so roll does not count. Frames within 5 deg of a chosen one are skipped. The penalty only reorders, so a short arc still fills the slots.
  - **Effect** (mean over the 11 runs, all picks up to 5 per track; the baseline is the same scores and 5 deg floor with no penalty):

    | | with penalty | quality only |
    |---|---|---|
    | mean frame quality | 0.53 | 0.58 |
    | mean angle between chosen views | 41 deg | 27 deg |
    | mean smallest angle between chosen views | 14 deg | 6 deg |

    The penalty costs 0.05 of mean quality for about 2.3 times the smallest view spacing. Without the 5 deg floor the quality-only pick took frames at about 1 deg from each other (s01_b: 1.0 deg mean smallest angle).
  - **Not covered.** Occlusion counts only other tracks, so a hand or an untracked item in front of an object is not seen. Quality weights, sigma and the penalty weight are guesses, not tuned, and nothing yet measures whether the chosen frames give better meshes; that is a Phase 3 question. s13_b scores low (0.22) because its two tracks are a stack that occlude each other. s12_b track 3 has no cluster (dropped by the lift), so it gets no keyframes, and s14_b track 5 gets 1 of 5 frames. s05_b has no tracks and s24_b has none on disk.
- P2.5 depth flags (in progress, not ticked): geometry evidence in `stages/depth_flags.py` (config `segmentation.depth_flags`, signals in `benchmark/results/p2_5_signals.json`, `benchmark/dump_depth_signals.py`) and the decision in `stages/material_pass.py` (config `segmentation.material_pass`, `benchmark/run_material_pass.py`, D-017). `Flag.TRANSPARENT_OR_REFLECTIVE` means reflectiveness is likely to break a later stage, not that the object is shiny (decision of 2026-10-28). Every object gets one VLM request with its best 2 keyframe crops; the object is flagged when the model names a metallic, mirror, transparent or translucent finish with confidence of at least 0.6, when the geometry says the lift is unstable (even if the model says matte), or when the answer is unreadable, low-confidence or missing. The entries below were made with the first design (geometry decides, VLM only for uncertain tracks, one yes/no crop) and `benchmark/run_depth_flags.py`, which the material pass replaced; the first run of the new pass is below.
  - **Dev result, 63 tracks in 12 runs.** 26 clear, 1 flagged, 36 uncertain (would go to the VLM) at `clear_err_ratio` 1.5; 31, 1 and 31 at 1.7, which was set after the VLM run (it clears 5 tracks that the VLM had answered no, loses no labelled reflective object). Of the five reflective objects labelled by eye in s17 and s18, the glass vase is flagged, the spray can and the foil egg are uncertain, and the mirror cube and the foil figure are cleared. Thresholds are guesses fitted on those five. s24 produced no tracks and s19 and s20 have no runs (test split), so the four hard scenes are not yet checked.
  - **Accepted miss, to revisit.** The cleared mirror cube and foil figure have a clean mask and lift, so they are left unflagged on the view that reflectiveness only matters if it breaks a later stage. That is untested for shape prediction. After the first Phase 3 runs of SAM 3D and TRELLIS.2 on the s17 and s18 reflective objects, check whether the cleared ones fail; if they do, add a detector for them (likely the VLM on every object) and re-tune the gate.
  - **VLM check on the dev runs** (`benchmark/results/p2_5_vlm.json`, 36 crops sent, one per uncertain track, model `meta-models/muse-spark-open-1026`). 32 no, 3 yes (spray can and foil egg in s17, pliers in s18) and 1 unreadable answer (the model wrote "None" for the s18 hole punch), which the fail-safe flags. With the geometry flag on the glass vase, 3 of the 5 labelled reflective objects are flagged; the mirror cube and the foil figure stay cleared by geometry as accepted. The hole punch flag is a false positive from the fail-safe; the pliers flag is arguable (metal jaws). Not run on s24, s19, s20.
  - **Material pass on the dev runs** (`benchmark/results/p2_5_material.json`, 63 requests with 2 crops each, 42 answered from cache, wall time 143 s with 4 workers; live requests averaged 15 s, the slowest 62 s). 10 flagged, 53 not. All five labelled reflective objects are flagged (glass vase, spray can, foil egg, mirror cube, foil figure), plus the s17 paint can and the s18 pliers (metallic finish). Three flags are low confidence, not reflectiveness: s11 t0 and t3 (dark featureless cubes, confidence 0.52 and 0.55) and s12 t3 (blurred crop, 0.31). Geometry never overruled a matte answer; it only agreed on the vase. Finish counts: 52 matte, 4 glossy, 4 metallic, 2 transparent, 1 mirror. Known gaps: metal tins and cans are labelled matte (s01 t0, s04 t3 flask with a polished cap, s06 t2, s12 t1, s12 t5, s14 t3, s16 t3), glossy Rubik's cubes are mostly matte, and a few classes are wrong (s18 t3 hole punch as metal_steel, s06 t1 toy car as wood). Class accuracy is not scored.
  - **Scene router (wired into `benchmark/run_tracking.py`, `--no-router` restores the config prompt)** (`stages/scene_router.py`, `segmentation.router`, `benchmark/run_router.py`, `benchmark/results/p2_5_router.json`). One VLM request per scene on 3 frames picks a surface from a pool (SAM prompt per key) and lists the objects; fewer tracks than listed objects raises a missing-object suspicion for the QA report. Dev run, 14 runs: 13 answered, s03 timed out twice (300 s) and fell back to the table prompt. The object list matched the ground-truth rigid count in all 13. The check fired on s05 (0 tracks), s13 (2 of 3) and s24 (0 of 1), and did not fire on a clean scene. It missed s18: a stray extra track hides the missing pliers, because only totals are compared. Prompt comparison on tracking (scratchpad, not in the repo): "object on a surface" gives 5 to 21 unassigned tracks per scene and 6 identity switches in s02, but tracks the s24 cube (with 8 stray tracks) and the third s13 stack object; "objects on a table" gives 0 or 1 unassigned. The first pool description (any other surface) sent s06 and s14 to the surface prompt, where tracking is noisier. After narrowing it to glossy or reflective surfaces, a re-run on s01, s03, s06, s14, s17 and s24 routes everything to the table prompt except s24 (the other eight dev runs were not re-run, and the committed JSON is from the first description). Off the default surface, a scene with one listed object is tracked by that noun (`tracking_prompt`, `noun_prompt_max_objects`): s24 with "cube" gives 1 track, 0 unassigned, 0 switches (the surface prompt gave 8 stray tracks). Scenes with several listed objects off the default surface would still get the noisy surface prompt; no benchmark scene exercises that. Discovery is not routed: `vlm_sam` makes its own nouns and `sam_generic` keeps `generic_prompt`, and nothing downstream reads discovery output. The count suspicion has no field in `schemas.py` yet (needs a human decision).
  - **Final P2.5 run** (`benchmark/report_flags.py`, `benchmark/results/p2_5_flags.json`, `p2_5_material.json`, `p2_5_router_hard.json`; code `stages/shape_flags.py`, `stages/material_pass.py`, `stages/scene_router.py`). 18 runs: the 13 dev runs plus s19_b, s20_b, s21_b, s22_a, s23_a. s22_b and s23_b were rejected by the capture gate for motion blur (46% and 33%, limit 30%), so s22 and s23 use `video_a`. Flags are `small`, `thin`, `occluded` (geometry, `segmentation.shape_flags`, provisional thresholds), `transparent_or_reflective` and the new `non_rigid` (VLM, with geometry cross-check). Ground truth enters per scene as counts only; tracks are not matched to objects.
    * **Reflective, hard scenes.** s17: 5 of 5 flagged (glass bottle, mirror cube, paint can, second bottle, foil egg). s18: the chocolate figure is flagged; the pliers are flagged too (metallic jaws, counted as a false positive on the labels). s19: the alarm is not flagged (passes either way). s20: scissors, bottle and container flagged. s24: the cube is tracked and not flagged (the object is matte).
    * **Deformable, hard scenes.** s21: the basketball is flagged `non_rigid` (borderline, passes either way); the folded towel has no track, which the router count check reports (2 listed, 1 tracked). s22: the pouch is flagged; two further tracks of 5 and 3 frames have no point cluster and no flag. s23: the sponge is flagged.
    * **Dev scenes, false and arguable flags.** `non_rigid` fires on a few rigid objects (earplug and basketball as semi-rigid, a block, a paper sheet as deformable); reflective false positives are mostly low-confidence answers (s11 cubes, s06 model car, s12 clip, s16 shoe). The VLM is not repeatable at temperature 0: the same crops of the s17 mirror cube answered matte, glossy, metallic or mirror across prompt variants and reruns, and the final prompt adds a line on silver, chrome and foil surfaces that was chosen on that s17 object (a dev scene). Treat single-run flag sets as noisy.
    * **Small, thin, occluded.** Small: 5 of 6 expected in s11, 1 of 1 in s02, plus a few extras; misses the 38 mm reflective puzzle toys in s17 and s20 (the hull reads larger than the object). Occluded: raised on both s13 stack boxes and on three s16 objects, no check against labels. Thin: raised on the s14 breadboard and the s18 pliers (2 of 2 expected on dev) and on two extras (s14 cube, s12 box); it misses all four thin objects on the test scenes (s19 breadboard, s20 scissors and flat container, plus s19 is borderline), because the visual hull overstates thickness by 3 to 4 times for flat objects (scissors 5 mm read 39 mm). The thresholds were fitted by eye on dev tracks and were not changed after looking at the test scenes. A follow-up is to measure thickness from the SfM-point cluster or the height above the table instead of the hull.
    * **Decided 2026-10-29.** The hard-scene flag criterion counts only tracks with a point cluster (s22's two stray tracks have none); P2.6 must drop tracks without a cluster. Still open: `FLAG_SCENES` in `evaluation/segmentation.py` still lists the four old scenes and its rule is one flagged object per scene; it needs updating when `Instances` carries flags (P2.6).
- P2.6 `Instances` (`stages/segmentation_stage.py`, `SegmentationStage`; harness `benchmark/run_instances.py`; `benchmark/results/p2_6_instances.json`). The stage runs router, tracking and repair, both lifts, keyframes, depth and shape flags and the material pass, and writes `segmentation/output.json` with per object: PNG masks per frame in `masks/obj_NNN/`, the sparse-point cluster in `clusters/obj_NNN.npy` (human decision 2026-10-29; the hull beside it as `obj_NNN_hull.npy`, outside the typed output, and used for the shape flags), keyframes, flags with reasons and the material answer. `schemas.py` changed with human approval (2026-10-29): optional material fields and `flag_reasons` on `ObjectInstance`, `flags: list[str]` on `Instances`. Hosted caches live in `segmentation.work/`, outside the stage folder.
  - **Result.** All 18 runs (13 dev, s16_a, s19_b, s20_b, s21_b, s22_a, s23_a) emit valid `Instances`, 4 to 105 s each. Most hosted answers came from the copied caches; the first pass still made about 12 new calls: 6 router requests (the pool description was narrowed after those scenes were last routed) and 6 material requests (the stage picks keyframes from hull centres, the benchmark scripts from sparse-point centres, so a few crops differed). Scene flags: s12 (track 3 dropped, 10 listed and 9 tracked), s13 (3 listed, 2 tracked), s14 (the paper sheet dropped: reaches the grid top), s16_a (two tracks dropped, 4 listed and 3 tracked), s21 (the towel is not tracked), s22 (two stray tracks dropped). s05 has no tracks and is not in the list.
  - **Harness caveat.** The runner replaces a stage folder, so the stage runs on copies in `runs_instances/` (ignored by git) with links to capture and reconstruction. The copied manifest gets today's cache keys for those two stages, because the old ones no longer match (config and source edits) and reconstruction would otherwise rerun. A real pipeline run executes every stage; `video2sim run` (P5.5) does not exist.
  - **Not done.** No end-to-end stage test with fake hosted clients (only unit tests of the assembly helpers). `video2sim eval` expects `runs/<scene>/shape/output.json` and does not read `Instances` yet, so the harness flag criterion (`FLAG_SCENES`) is unchanged.
  - **Cluster choice.** `point_cluster_path` is the sparse-point lift (18.7% median box error on the easy tier in P2.3, against 36.4% for the hull); a track with no sparse points falls back to its hull with a reason (1 of 75 objects: s12 obj_006, a thin box). Neither lift meets the 10% gate, so that exit criterion stays open. 75 objects over 18 runs, every cluster and hull file present, sparse points per object: min 12, median 550. Whether Phase 3 should register against points or hull is still P3.2's comparison.
  - **Reruns.** A rerun of all 18 after the cluster switch made no hosted calls. Compared with the P2.5 report, one object changed: the s24 cube is now flagged reflective because the stage's fresh answer had confidence 0.48 (matte), where the P2.5 answer was confident; the flag on that matte cube is VLM noise, not a measurement.
- Cluster box gate follow-up (2026-10-29, ticked). Write-up: `docs/research/cluster_box_accuracy.md`; evidence `benchmark/results/p2_3b_dev.json` (`benchmark/run_lift.py` on the 13 dev runs, s05 unscored). Two changes: `lift.trim_extremes` drops the 2 outermost sparse points per side of the resting box (`segmentation.lift.trim_points_per_side`), which removes the stray short-track points that set most boxes; `stages/primitive_fit.py` fits an upright box, cylinder or sphere to the masks from those points, and takes the union with the points when the best fit is poor (`segmentation.primitive_fit`).

  | dev scenes | hull | points, trimmed | fitted |
  |---|---|---|---|
  | median box error, easy tier (19 objects) | 36.4% | 8.9% | 4.3% |
  | easy objects within 10% | 3 | 10 | 15 |
  | median box error, all matched | 35.3% (54) | 13.2% (55) | 7.3% (55) |

  - **Decision (human, 2026-10-29): the criterion reads the fitted cluster.** `point_cluster_path` now points to it (superseding the P2.6 cluster choice above; no `schemas.py` field changed); the trimmed points are beside it as `clusters/obj_NNN_points.npy`, with the hull and `segmentation/fit.json` (kind, size, mask cost, source). A union or points-only cluster carries a note in `flag_reasons`. Checked offline on s01_b, s03_b and s06_b through `SegmentationStage` with clients that refuse hosted calls: every object and scene flag equals the P2.6 output, except the s03 ball's material answer, which is a cache the harness did not seed (it lives only in `runs_instances/`).
  - **Through the stage** (`benchmark/results/p2_6b_instances.json`): `SegmentationStage` rerun on all 18 runs in `runs_instances/` with clients that refuse hosted calls; no call was needed. Against P2.6, the only change is a note in `flag_reasons` on the 27 union clusters. Scoring `point_cluster_path` of the 12 scored dev runs with the benchmark alignment gives 4.3% easy-tier median (15 of 19 within 10%) and 7.5% over all 56 matched objects, the same as `run_lift.py`.
  - **`video2sim eval --stage segmentation` now aligns.** Before, it scored world-frame clusters against board-frame ground truth (recall 3%, 2 of 59 matched, on the same runs). It now moves each run's clusters into the board frame with `evaluation/alignment.py:align_scene`: the board fit on grid scenes, tag 0 on the clean-board scenes s14 and s24 (`CLEAN_ALIGNED`, moved from `benchmark/run_discovery.py`), and any other clean board (s05) is listed as without a usable prediction. On the 12 scored dev runs in `runs_instances/` (`benchmark/results/p2_eval_segmentation_dev.json`): recall 94.9% (56 of 59), cluster box error 4.3% median on the easy tier (n 19, gate 10%), the same as the benchmark scripts. Switches stay `no_data` and the flag row `partial` (s19 and s20 are test scenes), as before.
  - **Test split (2026-10-29, run once, nothing tuned on it).** s07_b, s08_b, s09_b and s10_b run through capture, reconstruction (`benchmark/run_stages.py`; every frame registered, 0.29 to 0.55 px) and `SegmentationStage` with the hosted clients (`benchmark/run_instances.py`, `benchmark/results/p2_test_instances.json`; first upload of these four videos, human approval 2026-10-29: 4 SAM video requests, 4 router requests, 16 material requests). `video2sim eval --split test --stage segmentation` over those plus the earlier test runs s16_a, s19_b, s20_b (`benchmark/results/p2_eval_segmentation_test.json`): **cluster box error 6.0% median on the easy tier, 13 objects in s07 to s09, 9 within 10% (gate 10%, passed)**; recall 92.9% (26 of 28). s10 is a clean board whose ground-truth origin is not established, so it is unscored, like s05. The easy failures are the same kinds as on dev: in-ear box 18%, mugs 19% and 31%, action camera 20%.
  - **Ticked 2026-10-29 by human decision**, with s10 unscored (3 of the 4 easy test scenes measured). On dev, both values were chosen on the same 19 objects; the test split was run once and not tuned on. Remaining easy failures after the fit are white-on-white parts that the masks miss (s01 and s04 in-ear box tab, s02 mug handles).

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
      maps the class to numbers through the table. The classification half already runs in
      Phase 2 (D-017, `stages/material_pass.py`, plus a surface-finish field); this task is
      done when the numbers half exists and the class list is final (P4.3).

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

   Status 2026-10-28, s16 video_a (160 frames, 3 fps, tag edge 100.26 mm) on three back ends.
   Results are on branches `experiment/colmap-cuda-modal` (`benchmark/results/p1_colmap_cuda_s16_*.json`)
   and `experiment/vggt-modal` (`p1_vggt_s16.json`, `p1_vggt_refined_s16.json`); neither is merged
   here. One run per cell. D-016 stays.

   | | pycolmap (Modal CPU) | pycolmap (Modal A10G) | VGGT (A10G) | VGGT + tag refine (A10G) |
   | --- | --- | --- | --- | --- |
   | Frames registered | 160/160 | 160/160 | 160/160 | 160/160 |
   | SfM reprojection | 0.40 px | 0.42 px | n/a | n/a |
   | Tag reprojection | 0.92 px | 0.92 px | 1.72 px | 0.61 px |
   | Tag distance error (gate 2%) | 0.76% pass | 0.76% pass | 1.71% pass | 2.20% **miss** |
   | Up-axis error (gate 2 deg) | 0.09 deg | 0.08 deg | 0.30 deg | 0.38 deg |
   | Tag scale spread | 7.3% | 7.3% | 6.7% | 5.9% |
   | Board fit RMS | 4.05 mm | 4.05 mm | 4.02 mm | 4.52 mm |
   | Extract / feature step | 75 s | 187 s | n/a | n/a |
   | Match | 215 s | 198 s | n/a | n/a |
   | Map / forward pass | 421 s | 349 s | 33 s | 33 s |
   | Refine | n/a | n/a | n/a | 1.3 s |
   | Model load | n/a | n/a | 13 s | 13 s |
   | Total | 711 s | 734 s | 37 s (71 s wall incl. Modal overhead) | 38 s (96 s wall) |
   | Peak GPU memory | n/a | n/a | 15.4 GB | 15.4 GB |

   Findings:

   * pycolmap is the most accurate on s16 video_a on every metric (tag distance, up-axis, tag
     reprojection) and is the only one with a low SfM reprojection error. VGGT is about 20x
     faster on the GPU (37 s against about 12 min), but its tag distance error is 2x and its
     up-axis error 3 to 4x pycolmap's, though both pass the gates.
   * Tag refinement cuts tag reprojection from 1.7 to 0.6 px, but this is the fitted quantity,
     not an independent check. It made tag distance worse (1.71% to 2.20%, now over the gate)
     and up-axis slightly worse, so it does not help on this video.
   * The earlier note that the s16 video_a miss is specific to VGGT is only true for the refined
     variant: plain VGGT passes the tag distance gate on video_a (1.71%). video_b: pycolmap
     2.66% and plain VGGT 2.35% both miss, consistent with the warped marker accepted in P1.6.
   * Timings are not like for like: the pycolmap rows are CPU-bound per-stage times on a
     Modal host, the VGGT rows a single GPU forward pass; the 71 s and 96 s wall times differ
     only by Modal overhead for the same 37 s of server time. Local M4 CPU took about 30 min
     (human-reported).
   * Recommendation unchanged: keep pycolmap (D-016). VGGT is a viable fast option for long
     videos if its accuracy margin is acceptable, but its license is non-commercial and it was
     measured on one video here.

   **Verdict (P1.1 experiment closed, 2026-10-28).** Add VGGT after the MVP as an opt-in,
   approximate fast mode, `reconstruction.backend: vggt`, for users with an eligible NVIDIA GPU
   (about 16 GB for 160 frames, fewer with a frame cap). It is not the default, not for the
   MVP, and D-016 stands. Conditions before it ships:

   * Run its output through the P1.4 and P1.6 gates and fall back to pycolmap automatically
     when the tag distance or up-axis gate fails. The cost of a failure is one VGGT pass,
     about 40 s on an A10G.
   * Measure the fallback rate on the full dev set first. Plain VGGT passed the tag distance
     gate on only 2 of 8 videos, so if most scenes fall back the mode is slower, not faster.
   * No tag refinement (`benchmark/tag_refine.py`); it was worse on s16 video_a.
   * Non-commercial checkpoint (`facebook/VGGT-1B`): flag it in the CLI and in
     `docs/licenses.md`, and keep it out of any commercial path.
   * Torch behind an optional extra, imported inside the function. `run_sfm_vggt` returns
     `SfmModel`.

   The experiment branches stay unmerged: `experiment/vggt-modal` and
   `experiment/colmap-cuda-modal`.

6b. **pycolmap optimisation (from the s16 timings; do before the VGGT mode).** Reconstruction
   is the slowest stage seen so far: s16 video_a (160 frames) takes about 12 min on Modal CPU
   and about 30 min on the local M4, split roughly 75 s features, 215 s exhaustive matching,
   420 s mapping on Modal (local split in step 1). A GPU does not help (mapping is CPU-bound), so the work is to do less.
   Whole-pipeline timing is not measured; take it first so we know the share.

   Goal: at least 2x faster on the long videos with the same gate results. Proposed limits,
   for a human to put in `configs/targets.yaml` before the work starts: gates pass on every
   video that passes today; mean reprojection error no more than 0.1 px worse; tag distance
   error no more than 0.3 percentage points worse; up-axis error no more than 0.2 deg worse.

   Steps, in order, one lever at a time so each effect is attributable:

   1. **Profile locally. Done 2026-10-28**, `benchmark/profile_sfm.py`, results in
      `benchmark/results/p1_6b_profile_local_baseline.json` (Apple M4, 10 cores, pycolmap
      4.2.1 CPU, 3 fps, from scratch, one run per video):

      | video | frames | decode | extract | match | map | total | match share |
      | --- | --- | --- | --- | --- | --- | --- | --- |
      | s01 video_b | 28 | 1 s | 7 s | 7 s | 4 s | 20 s | 37% |
      | s02 video_a | 108 | 5 s | 24 s | 83 s | 18 s | 131 s | 64% |
      | s16 video_a | 160 | 7 s | 37 s | 488 s | 136 s | 669 s | 73% |
      | s10 video_a | 187 | 9 s | 61 s | 777 s | 466 s | 1313 s | 59% |
      | s19 video_a | 193 | 8 s | 55 s | 479 s | 55 s | 597 s | 80% |

      Exhaustive matching is the largest stage on every video above 28 frames (59 to 80% of
      the total), so step 2 is the first lever to pull. Mapping is next and varies a lot with
      scene (55 s on s19 against 466 s on s10 at similar frame counts; the sparse cloud is
      35k points on s19 and 173k on s10), so frame count alone does not predict it. Local
      s16 video_a took 11 min, not the 30 min reported by hand earlier, and the Modal CPU
      run (12 min) split differently (match 215 s, map 421 s), so the Modal numbers do not
      transfer to the M4. Decoding and feature extraction together are under 10% of the
      total. All frames registered on every video. Reprojection error 0.38 to 0.73 px.
   2. **Matcher.** Replace `match_exhaustive` (O(n^2) pairs) with sequential matching with an
      overlap window, plus loop closure because the capture orbits the table (352 deg
      azimuth on s16). Loop detection needs a vocabulary tree file, which is a download:
      ask first. Fallback without it: sequential plus exhaustive matching on every k-th frame.
   3. **Frame count.** Try 3 fps down to 2 and 1.5 fps, and keyframe selection by baseline or
      blur from P1.4, instead of a fixed rate. Mapping time grows faster than linearly with
      frames, so this is likely the biggest lever.
   4. **Resolution and features.** Extraction at 1280 px down to 960 px, and a cap on SIFT
      features per image.
   5. **Mapper settings.** Fewer local and global bundle adjustment iterations, a larger
      global BA growth ratio, explicit thread count. Check whether pycolmap 4.2.1 exposes a
      global mapper (`global_mapping`) as an alternative to incremental; a new dependency is
      not needed for this, but a new weights or data download is.
   6. **Combine** the levers that held the limits, rerun the five-video table, then the full
      dev set and the test split through the P1.6 evaluation.

   **Results, steps 2 to 6 (2026-10-28).** Code: `SfmOptions` in `reconstruction.py`,
   `reconstruction.sfm` in `configs/default.yaml`, `benchmark/sweep_sfm.py`. Same M4, 3 fps, from
   scratch, one run each. Scored with the P1.6 evaluation (tag distance against the printed
   board, up-axis, tag and SfM reprojection).

   Single levers on s16 video_a (`benchmark/results/p1_6b_sweep_s16_video_a.json`; the
   baseline was rerun and took 709 s, against 669 s in step 1, so run-to-run noise is about
   6%, more in the matching stage):

   | variant | total | match | tag dist (gate 2%) | up-axis | SfM reproj |
   | --- | --- | --- | --- | --- | --- |
   | baseline (exhaustive) | 709 s | 523 s | 0.75% | 0.09 deg | 0.40 px |
   | sequential, overlap 10 | 205 s | 65 s | 0.81% | 0.08 deg | 0.33 px |
   | sequential + loop pairs every 8th frame | 208 s | 70 s | 0.81% | 0.08 deg | 0.33 px |
   | sequential + global mapper | 194 s | 71 s | 0.79% | 0.10 deg | 0.33 px |
   | 2 fps (exhaustive) | 374 s | 271 s | 2.30% **miss** | 0.17 deg | 0.40 px |
   | 1.5 fps (exhaustive) | 220 s | 155 s | 2.21% **miss** | 0.17 deg | 0.42 px |
   | features 4096 (exhaustive) | 665 s | 492 s | 0.76% | 0.09 deg | 0.43 px |
   | lighter bundle adjustment (exhaustive) | 796 s | 613 s | 0.76% | 0.08 deg | 0.40 px |
   | max image size 960 (exhaustive) | 180 s | 140 s | 2.14% **miss** | 0.22 deg | 0.74 px |

   * Sequential matching alone gives 3.5x with no measurable quality loss. Loop pairs added
     nothing on this video: the orbit stays consistent without them (they did add matches,
     1146 pairs against 1025). Keep `loop_stride` at 0 until a scene shows drift.
   * Fewer frames and a smaller image save time but lose the tag distance gate, so they are
     not used. Fewer features and lighter bundle adjustment save nothing worth having
     (differences are inside the noise).

   Sequential matching (overlap 10, incremental mapper) on four more videos, against the
   step 1 baseline for time (`p1_6b_sweep_<scene>_<video>.json`):

   | video | frames | baseline | sequential | speed-up | tag dist | up-axis | SfM reproj |
   | --- | --- | --- | --- | --- | --- | --- | --- |
   | s01 video_b | 28 | 20 s | 16 s | 1.3x | 1.12% (was 1.07%) | 0.19 (0.16) deg | 0.52 (0.54) px |
   | s02 video_a | 108 | 131 s | 61 s | 2.1x | 1.05% (1.06%) | 0.02 (0.08) deg | 0.61 (0.73) px |
   | s16 video_a | 160 | 709 s | 205 s | 3.5x | 0.81% (0.75%) | 0.08 (0.09) deg | 0.33 (0.40) px |
   | s19 video_a | 193 | 597 s | 142 s | 4.2x | 1.34% (1.13%) | 0.26 (0.14) deg | 0.57 (0.66) px |
   | s10 video_a | 187 | 1313 s | 440 s | 3.0x | not measured | not measured | 0.31 (0.38) px |

   All frames registered, one model, no gate reasons on every video. Quality stays inside the
   proposed limits (tag distance within 0.3 points, up-axis within 0.2 deg, reprojection no
   worse); the worst case is s19 at +0.21 points and +0.12 deg. Caveats: the s01 and s02
   baselines come from the dev results file and the s19 baseline from the test results file,
   both older runs with the same sampling; s10 is a clean board, so it has no tag distance or
   up-axis, and its tag reprojection and scale spread were not measured with exhaustive
   matching (0.88 px and 0.31% with sequential). The proposed limits are not yet in
   `targets.yaml`.

   The global mapper (GLOMAP in pycolmap 4.2.1, `global_mapping`, no new dependency) is
   another 1.0 to 1.9x on the mapping stage (s10: 148 s against 280 s; s19: 22 s against
   32 s) with similar quality (s10 316 s total), but SfM reprojection rose on s19 (0.78 px
   against 0.57 px) and up-axis was slightly worse on s01 and s02. Left off: `mapper:
   incremental` stays the default; revisit if mapping becomes the largest stage.

   Decision taken in this change: `reconstruction.sfm.matcher` is `sequential` in
   `configs/default.yaml`. Revert it to `exhaustive` to restore the old behaviour. Not done:
   the full dev set and test split rerun through the P1.6 evaluation (step 6), and a rerun
   of the s16 hand-timed numbers on the Modal CPU.

   Every knob goes in `configs/default.yaml` under `reconstruction`, since it changes results
   and is hashed into the run manifest. No change to `schemas.py`. Evidence: a results file in
   `benchmark/results/` per step. Ask first for: the vocabulary tree download, any new
   dependency, and the `targets.yaml` limits above.

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
