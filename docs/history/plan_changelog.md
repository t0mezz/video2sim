# PLAN.md changelog

Moved verbatim from `PLAN.md` section 13 on 2026-11-03. Rows are in the order they had there,
which is not strictly by date. Git history of `PLAN.md` and `docs/decisions.md` carries the rest.

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
| 2026-10-31 | P4.5: volume test written up in `docs/research/mesh_volume.md`; meshes are shells, the outer volume is right and density is the main mass error |
| 2026-10-31 | P4.5 ticked: `stages/mass.py`, sealed voxel body, one dev-measured hollow fill fraction and the center of mass left to P4.8 (human decisions); 14 of 56 dev objects within 30%, exit criterion not met |
| 2026-10-31 | P4.6 ticked: fill fractions per class and answer; 20 of 56 dev objects within 30% with objects held out (36%), exit criterion not met |
| 2026-10-31 | P4.7 ticked under D-021: settling in MuJoCo, 10 of 13 rigid dev scenes pass (77%), exit criterion not met; `mujoco` added by human approval |
| 2026-10-31 | P4.8 ticked under D-023: physics stage emits `SimAssets` on 13 rigid dev runs, 61 of 61 valid; all Phase 4 tasks done, the mass and settling exit criteria are not met and the gate stays Open |
| 2026-10-31 | P4 gate run once on the test split: meshes 35 of 35, settling 8 of 8, mass 45% against 50%; gate Failed on mass |
| 2026-10-31 | D-026: mass miss waived and the Phase 4 gate marked Passed by human decision; Phase 5 is the current focus |
| 2026-10-31 | D-025: mass target 50%, round objects ignored in settling, stacked objects seated; 13 of 13 rigid dev scenes settle and mass is 52% on dev, gate still Open until the test split runs |
| 2026-10-31 | D-024: mass combines geometry with the vision-language model; 52% of dev objects within 30%, exit criterion not met, `AGENTS.md` rule amended by human decision |
| 2026-10-31 | D-022: collision fit ignores surfaces inside the sealed body, primitive fallback and vertex thinning fixed; 61 of 61 dev objects pass the collision checks |
| 2026-10-31 | D-020: tags-on-table layer on the plane fit; s05 rerun, all 4 objects now SAM 3D with fit 0.95 to 0.98 and `lift.max_extent_m` back at 1.5 |
| 2026-10-31 | D-027: Phase 5 validates in MuJoCo on the Mac; a parallel Isaac Sim verifier follows on the GPU workstation (section 12 item 8) |
| 2026-10-31 | Phase 5 goal, P5.1, P5.2, P5.4 and exit criteria reworded for MuJoCo under D-027 by human approval |
| 2026-11-01 | D-032: SAM 3D on A100-40GB with two warm containers; s02 shape 62 s warm against 151 s cold |
| 2026-10-31 | SAM 3D worker redeployed with timing and GPU memory stats, no change to meshes; probe: cold 110 s, warm 17 s, about 18.9 GB peak allocated (`docs/research/sam3d_cold_start.md`) |
| 2026-10-31 | D-031: shape and physics run across the scene; s02 from 315 to 202 s in those two stages, output unchanged on cached draws |
| 2026-10-31 | D-030: shape stage down to 2 SAM 3D draws per object after the dev replay; size error 5.7% against 5.0%, pose unchanged; the final benchmark stays at 6 draws |
| 2026-10-31 | P5.6 ticked: final test run on 8 rigid scenes, 7 scored; size 5.7% and mass 29% missed, every other target met; mass fell from 45% with no code change, see `docs/final_benchmark.md` |
| 2026-10-31 | P5.5 ticked: fresh run of s01 `video_a` through `video2sim run`, 8.7 min, no manual steps |
| 2026-10-31 | P5.4 ticked under D-029: 7 of 7 chosen dev objects pass nominal, 5 pass every case; demo exit criterion ticked |
| 2026-10-31 | P5.1 to P5.3 ticked: export stage writes USD, URDF and the QA report on 13 rigid dev scenes, 61 objects; `video2sim run` added, P5.5 stays open until a run from a fresh video |
| 2026-10-31 | D-028: `SceneExport` added to `schemas.py`, `usd-core` 26.8 added, P5.1 uses `displayColor`, container image removed from P5.5 |
| 2026-11-01 | D-001 confirmed by the human; Isaac Sim settling verifier added (section 12 item 8, first part) |
| 2026-11-02 | Section 12 item 9 added: occlusion-robust object discovery (multi-view router, gap fill, `DiscoveryCheck`); plan and SAM 3.1 object-prompt research in `docs/research/occlusion_robust_discovery_plan.md`; nothing built yet |
| 2026-11-03 | Repo cleanup, human decisions: working-tree scratch outputs deleted (`*_out/`, `eval_out/`, `MUJOCO_LOG.TXT`, `trajectory-*.json`, `runs_instances_98/`, `board_a4_pair.json`, three `benchmark/results/` leftovers) and `.gitignore` extended; D-006, D-008, D-015 decided, D-001/D-003/D-007 confirmed, D-002 stays Assumed; license/terms items stay Open (human owner); D-018 schema half verified done, Modal teardown approved for human to run; `configs/targets.yaml` frozen as final; P5 gate stays Open into the item 8 verifier follow-up; item 9 9.4 stays opt-in, 9.8 deferred; item 6b left open; P1.6 config guesses accepted as-is untuned |
| 2026-11-04 | Cutover by human decision: `PLAN.md` retired into `docs/history/plan_2026-11-03.md`, open work in `TODO.md`; D-030 dropped by D-033; `segmentation.verify` off in the committed config; the Isaac Sim settling verifier merged (commit a6e1ed3); the `.gitignore` extension of 2026-11-03 was dropped before committing |
