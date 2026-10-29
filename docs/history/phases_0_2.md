# Phases 0 to 2: full task and evidence record

Moved verbatim from `PLAN.md` on 2026-10-29, when all three phase gates had passed, to keep
the working plan short. `PLAN.md` keeps the tick, the evidence link and a one-line result per
task; this file keeps the reasoning, caveats and numbers behind them. Code paths are as of
commit 289bfe2 (the per-frame discovery code of P2.1 was removed after it, in eef52b6).

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
  - **Not done.** No end-to-end stage test with fake hosted clients (only unit tests of the assembly helpers). `video2sim eval --stage segmentation` reads `segmentation/output.json` and the point clusters (`load_instances`), but the harness flag criterion is unchanged: `FLAG_SCENES` still lists s17 to s20 with one flagged object per scene, and the harness skips deformable scenes (s21 to s23, D-014). The criterion as written in the exit criteria is checked by `benchmark/report_flags.py`, not by `eval`.
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

