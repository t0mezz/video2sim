# TODO

Open work only. Finished work lives in `docs/history/` (task and evidence records) and
`docs/decisions.md`; scope, targets and the pipeline are in `docs/overview.md`. The old `PLAN.md` is
kept as `docs/history/plan_2026-11-03.md`. Last updated 2026-11-04. Tick a box only with the
evidence linked on the same line (a result file, a commit, a run id); ticked items are removed at
the next cleanup, and their evidence stays in the doc or decision row it belongs to.

**[human]** marks work only a human can do or decide. **[agent]** is work an agent can do once the
human has approved what `AGENTS.md` says to ask about first (hosted calls, dependencies, schema or
target changes, test split runs). The backlog items keep the numbers they had in section 12 of
the old `PLAN.md`, because decisions, docs and code comments cite "section 12 item N"; it is item N
here.

## Open exit criteria

The Phase 5 gate stays Open (confirmed 2026-11-03).

- [ ] **[human] A new person can reproduce a scene from the README alone.** Needs someone who has
  not seen the repo: they follow "Make a scene" on a fresh video, and every place they get stuck
  becomes a README fix.
- [ ] **[human] USD scenes read back with the units, axes, mass and inertia of `SimAssets`; URDF
  loads in MuJoCo (D-027).** P5.1 and P5.2 report both checks passing on 61 objects of the 13 rigid
  dev scenes (`benchmark/results/p5_export_dev.json`), yet the box was never ticked and the old plan
  does not say why. Probably because PhysX behavior is unverified (item 8). Decide: tick it on that
  evidence, or hold it for item 8.
- [ ] **[human] Container image that runs all models (P0.8).** Deferred past the MVP on 2026-10-26;
  the matching Phase 0 exit criterion stays unticked with it.

## Missed and waived targets

Kept visible so they are not read as passes. These are not tasks unless a human reopens them.
Targets are final (`configs/targets.yaml`, frozen 2026-11-03). Details: `docs/final_benchmark.md`.

* **Mass within 30% for 50% of objects.** The Phase 4 gate run read 45% on test and was waived by
  human decision (D-026). The final benchmark reads 29% (9 of 31) on test; dev reads 52%, in sample.
  Cause: material class and fill fraction are not visible in video, and hollow objects deviate from
  uniform density (D-024). The range of the method is about 29 to 45% per draw. A new mass method is
  built on dev only, because the test split has been seen.
* **Median size error 5% or less.** 5.7% on test; the easy tier alone is 3.3%. The misses are thin,
  small or occluded objects, because registration keeps SAM 3D's proportions: s19 obj_010 122%,
  s20 obj_044 75%, s15 obj_038 33%, s15 obj_040 30%, s16 obj_004 27%.
* **Medium-tier recall is 78%**, although overall recall (93.9%) meets the target. Misses: s15
  obj_021 and s16 obj_003.

## Human-owned

- [ ] **Run the TRELLIS.2 teardown (D-018).** Approved 2026-11-03; an agent does not run it.
  ```bash
  uv run modal app stop video2sim-trellis2
  ```
  ```bash
  uv run modal volume delete video2sim-trellis2-weights
  ```
- [ ] **Settle the terms still unread (D-004, D-005, `docs/licenses.md`).** The Meta Model API
  retention terms and video length limits are unread. D-005 lists the official Meta Model API terms
  and the Modal and Baseten terms as open, while D-015 records the Modal and Baseten terms as read on
  2026-10-26: reconcile the two rows. `docs/licenses.md` also marks OpenCV, pycolmap (COLMAP, SIFT)
  and SciPy as only partly checked, because their bundled licenses are unread.
- [ ] **Keep or relax the upload rule.** `AGENTS.md` still says to ask before uploading any data,
  although D-004 allows sending benchmark videos to hosted APIs.
- [ ] **Confirm or retire D-002** (12 weeks, two engineers, no start date; status Assumed).

## Known gaps in finished work

None of these blocks anything. All are [agent] work unless marked.

**Capture and reconstruction**

* The capture gate (P1.4) has no duration limit and no rolling-shutter or autofocus check.
* The point cloud is not cropped: stray points remain below the table (P1.5).
* Frame sampling is a fixed 3 fps, the board layout is nominal, and clean-board reconstruction was
  not evaluated. The tags-on-table layer (D-020) was not measured on the test split.
* s05 and s10 have no established clean-board origin, so both scenes are unscored.

**Segmentation**

* `video2sim eval` does not score identity switches (no track loader), and its flag rule is the old
  one-flag-per-scene check. The real checks are `benchmark/run_tracking.py` and
  `benchmark/report_flags.py`.
* No end-to-end stage test with fake hosted clients.
* Easy-tier box failures are white-on-white parts the masks miss (in-ear box tab, mug handles).
* The reflective flag is unchecked against shape results. Compare the VLM finish labels with the
  objects that fail in shape prediction (s17 mirror cube, s18 foil figure, which the geometry-only
  design cleared unseen), then re-tune `min_confidence` and `reflective_finishes` (D-017). Needs a
  hosted run, so ask first.
* Frame-level misses are not handled (parked 2026-10-28): where the 3D lift expects a mask and a
  frame has none, reproject and re-prompt SAM with a box or point. Item 9 covers whole-object
  misses only.

**Shape**

* Registration weights are untuned first guesses. The fit score ranks samples poorly: it picks the
  best of 6 for 17 of 42 objects, and perfect selection would gain at most 1.4 points of size
  error. Proportions are the larger error.

**Physics**

* Friction and restitution are mostly unsourced placeholders (D-019): friction is unsourced in 13 of
  14 classes, restitution in 12, density in 6. Sourcing them (`docs/research/material_sources.md`,
  `docs/sources/`) would let the low-confidence flag cover them.
* Restitution is never simulated. Rolling resistance has no sourced value, so round objects (the
  basketball, the egg) roll and are flagged `IGNORED_ROUND` instead (D-025).
* Collision fit (D-022): mouths under about 8% of the longest side are sealed, so parts filling a
  narrow neck are not caught.
* Inertia has no measured ground truth.

**Export and demo**

* Pick-and-place failures were not investigated: dev s11 (four of five cases fail on placing, about
  14 cm off) and s17 (low friction); test s08 (high friction ends 13.5 cm from the goal, and at 4.5
  times nominal mass the object diverged to 11.9 km, read as a simulator failure). The RL stretch of
  P5.4 was not attempted, and the control is open loop.
* `video2sim eval` still reports manual steps as `not_measured`; the final benchmark counted them
  from run logs.
* Three full videos run in parallel crashed the Mac after 6.5 min (memory pressure).

**Not measured yet**

* D-032: the warm-up inside a full `video2sim run`, and its cost.
* PhysX: scenes other than the one settled so far, the friction combine mode and the 255-vertex
  collision limit (item 8).

## Backlog after the MVP

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
   (human; left open 2026-11-03, targets frozen as final). Tables and steps: `docs/history/reconstruction_experiments.md`.

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

   Note 2026-11-03: `benchmark/modal_vggt.py` is not in the main tree. It exists on the unmerged
   branch `experiment/vggt-modal` (commit a4dc796).

8. **Isaac Sim verifier (D-027).** After Phase 5, on the NVIDIA GPU workstation, build a verifier
   that runs alongside the MuJoCo checks: load the exported USD scenes in Isaac Sim and compare
   their behavior under PhysX with the MuJoCo results.

   Status 2026-11-04: the settling part is merged (commit a6e1ed3, from the branch
   `origin/item8-isaac-settling-verifier` of 2026-11-01). `benchmark/run_isaac_settling.py` runs
   the five MuJoCo cases on `export/scene.usd` in Isaac Sim 6.2 under PhysX (5 s window, 0.02 m
   limit); `src/video2sim/isaac_verify.py` holds the USD side, and `tests/test_isaac_verify.py`
   tests it without Kit. Two runs of one scene (`20261101T112852Z_faf46bdc`,
   `20261101T120448Z_faf46bdc`) are stable in all five cases, largest drift 0.44 mm
   (`benchmark/results/isaac_settling.json`). Open: the comparison with the MuJoCo numbers, the
   other dev scenes (no other exported run existed locally on 2026-11-01), and the pick-and-place
   port. These need the GPU workstation.

9. **Occlusion-robust object discovery (from P2.1, P2.5).** Optimisation of stage 3. The router
   sees 3 evenly spaced frames and tracking uses one generic SAM concept, so 3 of 59 dev objects
   get no track (s13 stack member, s18 pliers, s24 glossy cube). Build: coverage-based view
   selection for the router, a multi-view router with a merged object inventory, and a gap-fill
   round after tracking (per-noun SAM requests and box prompts placed on unexplained point
   clusters), with a `DiscoveryCheck` in `Instances` (schema change approved by the human). Human
   decision 2026-11-02: item added, schema change approved, hosted calls on dev approved. Plan,
   research on SAM 3.1 box and point prompts, tasks 9.0 to 9.10 and risks:
   `docs/research/occlusion_robust_discovery_plan.md`. The test split is not run before a gate.

   Status 2026-11-03: 9.0 to 9.6 are built and every branch named below is merged into main (checked
   with `git merge-base --is-ancestor`, so "uncommitted chain" in the 9.3 line is stale). The 9.x
   features and `segmentation.verify` (below) are off by default (`router.views.mode: even`,
   `gap_fill.enabled: false`, `verify.enabled: false`), which keeps the stage output byte-identical
   to before. The detailed plan was compacted out of
   `docs/research/occlusion_robust_discovery_plan.md`; the full 500-line version, with the task
   specs and the risk table, is `git show 2fc00e5:docs/research/occlusion_robust_discovery_plan.md`.

   Open:

   - [ ] **9.8** Dev evaluation, hosted, dev split only. Planned arms: A even-3 views with gap fill
     off (the existing `runs_instances/`), B coverage-6 with it off, C even-3 with noun gap fill,
     D even-3 with cluster boxes, E coverage-6 with noun and cluster boxes. Score each with
     `video2sim eval --stage segmentation`, plus recall per tier, recovered misses, false-positive
     tracks per scene, identity switches, requests and dollars per scene, wall time and
     `view_coverage`; under $2 of video per arm. `benchmark/run_instances.py` has no `--views`,
     `--gap-fill`, `--out-runs` or `--max-requests` yet (it has `--runs` and `--out`). A subset
     already ran (`benchmark/results/s12_9_98_eval.json`, commit dfc436d): arm A against a new arm F
     (the redesigned residual clusters, VLM check, one box request) on s13, s18 and s24. Recall is
     1.0 in both, which proves nothing: in those fresh runs generic tracking already found all three
     known misses. Arm F added one false positive on s24 (a match bundle, since rejected by the
     raised 0.7 confidence) and was slower on s13 (+66.6 s) and s24 (+31.5 s). Criteria not met;
     defaults stay off. **[human] decide first whether the full run is still worth it**, since the
     misses that motivated it did not reproduce; if so, an agent extends the script and runs it. Do
     not run the test split before a gate.
   - [ ] **9.9** Defaults, docs, tick. Blocked on 9.8. Decide the default config from its table
     (the human confirms), then `configs/default.yaml`, `AGENTS.md` (new commands, same change),
     `docs/model_interfaces.md` (a self-hosted tracking server or `python:` factory now also
     receives the object-prompt grammar as the prompt and must answer with the ids sent), and
     `docs/licenses.md` only for facts the probe established.
   - [ ] **9.10** Capture-side coverage check (optional). A check in `stages/capture.py` and
     `benchmark/eval_capture_gate.py`: azimuth span of the registered cameras around the workspace
     centre and minimum elevation, reported as a warning in the capture result. It is the only guard
     against an object hidden from every view. Propose thresholds from the dev videos
     (`configs/targets.yaml` stays human-edited).

   Not in the 9.x list: `stages/verify_coverage.py` (commits 6ce514e and ac99902, 2026-11-03;
   `tests/test_verify_coverage.py`) adds an end-of-segmentation check. After gap fill, a VLM look
   at whole-scene views with the kept tracks outlined names router-listed objects that are visible
   but untracked; each earns one SAM noun request, and the check loops for up to `max_turns` rounds
   (`segmentation.verify`). The committed default is `enabled: false` (human decision 2026-11-03; it
   had been `true`, against its own comment). It has no row in `docs/decisions.md` and has not been
   evaluated: evaluate it together with gap fill in 9.8 before turning any default on.

   Closed, not open: **9.7** overlay audit, superseded by the redesigned 9.6 (residual clusters, then
   a per-cluster VLM check in `stages/cluster_check.py`). `stages/overlay_audit.py` and its test were
   deleted on 2026-11-03; branch `s12-9-7-overlay-audit` keeps the history.

   Built, with evidence (kept here because 9.8 and 9.9 build on it):

   - [x] **9.5** Schema and QA report. Evidence: branch `s12-9-5-schema-qa`, `DiscoveryCheck`/`found_by`/`discovery` in `schemas.py`, Discovery block in `qa_report.py`, `tests/test_schemas.py::TestDiscovery`, new `tests/test_qa_report.py`, doc clarifications in `docs/research/occlusion_robust_discovery_plan.md`
   - [x] **9.1** Object-prompt builder. Evidence: branch `s12-9-1-prompt-builder`, `models/sam_prompts.py`, `tests/test_sam_prompts.py` (6 tests)
   - [x] **9.0** Live probe of object prompts (gate for the cluster-box path). Evidence: `benchmark/probe_object_prompts.py`, `benchmark/results/s12_9_probe.json`, caches in `runs_instances/s02_b/segmentation.work/probe_object_prompts/`. Gate PASSED: 4/4 requests accepted on s02_b (s02_a has no finished run), ids preserved, box tracks cover all 50 frames (prompted-frame IoU vs GT box 0.56 to 0.64), empty-table box returns a table patch on 11/50 frames with fill decaying 0.61 to 0.02. Cost $0.04. Cluster-box path confirmed; section 2 open points settled (same `input_text` part, `<Nf>` is the 0-based clip position)
   - [x] **9.2** Offline view-coverage experiment (gate for 9.3 and 9.4). Evidence: `benchmark/eval_view_coverage.py`, `benchmark/results/s12_9_coverage.json`. Outcome: even-3 already shows every GT object at >=0.3 visible fraction in >=1 view on all 12 scored dev scenes (min 2 to 3 views), coverage-6 adds nothing; multi-view routing NOT justified by occlusion (misses look like SAM misses). 9.3 built regardless per human decision; 9.4 kept opt-in behind `views.mode` by human call 2026-11-03, 9.8 deferred
   - [x] **9.3** View selection. Evidence: branch `s12-9-0-probe` (uncommitted chain), `stages/view_select.py` (`compute_visibility`, `select_views`, `ViewConfig`), `configs/default.yaml` `router.views` block, `tests/test_view_select.py` (6 tests incl. closed-form shadow L = d h/(H-h))
   - [x] **9.4** Multi-view router. Evidence: branch `s12-9-4-router`, map-reduce in `stages/scene_router.py` behind `views.mode` (even keeps today's single call and cache key), `merge_synonyms`/`noun_denylist`/`max_workers` config, `tests/test_scene_router.py` (22 tests incl. the even-mode pin)
   - [x] **9.6** Gap fill, redesigned 2026-11-03 on `lift-tall-cap`: residual clusters (per-point vote, table region, triangulated board, KD-tree grouping) -> per-cluster VLM check -> one box request with geometric acceptance, noun fallback only for a failed accepted cluster. `stages/gap_fill.py`, `stages/cluster_check.py`, stage wiring with `DiscoveryCheck`/`found_by`/`route.json` gap section, `tests/test_gap_fill.py`, `tests/test_cluster_check.py`. Defaults off = byte-identical until 9.8/9.9 decide.
