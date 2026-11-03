# Final benchmark (P5.6)

One `video2sim run` per scene, no manual steps, code 21d34de, test split, 2026-10-31. Seven scenes on one Mac sequentially; s20 on the GPU workstation (Linux, same commit, config, and video hash). Scores: `benchmark/results/p5_6_eval/`. Pick-and-place: `benchmark/results/p5_6_pick_place_test.json`. A parallel three-video attempt crashed the Mac after 6.5 min (memory pressure); its partial output was discarded.

Coverage: 8 rigid test scenes run, 7 scored. s10 has no clean-board origin; s23 is deformable (D-014). The medium tier has 2 scenes, 9 objects. Validation is MuJoCo only (D-027); Isaac Sim / PhysX behavior is unverified.

| Target | Result | Limit | Met |
| --- | --- | --- | --- |
| Object recall | 93.9% (31 of 33) | 90% or more | Yes |
| Size error, median | 5.7% | 5% or less | No |
| Position error, median | 4.3 mm | 15 mm or less | Yes |
| Orientation error, median | 1.5 deg (17 objects with defined yaw) | 10 deg or less | Yes |
| Mass within 30% | 29% (9 of 31) | 50% or more | No |
| Stability | 7 of 7 scored, 8 of 8 run | 95% or more | Yes |
| Asset validity | 33 of 33 scored, 37 of 37 run | 100% | Yes |
| Runtime, worst scene | 14.9 min | 60 min or less | Yes |
| Manual steps | 0 on 8 of 8 runs | None | Yes |

Recall misses: s15 obj_021, s16 obj_003; medium tier 78%. Size misses concentrate on hard/medium objects: s19 obj_010 122%, s20 obj_044 75%, s15 obj_038 33%, s15 obj_040 30%, s16 obj_004 27%; easy tier alone is 3.3%. Registration preserves most of SAM 3D's proportions, and thin, small, or occluded objects have few observed points. Phase 3/4 gates on the same videos read 5.4%/6.0%, so the miss is stable.

Mass 29% compares with 45% at the Phase 4 gate on the same videos with unchanged mass code, table, and config. Both runs draw fresh hosted-model answers and SAM 3D samples; 21 objects took the model mass in both. Six objects left tolerance and one entered; median mass error moved 33% to 41%. The method's range on these objects is therefore about 29 to 45% per draw. Underlying cause (D-024, D-026): material class and fill fraction are not visible in video, and hollow objects deviate strongly from uniform density. No cheap fix remains on these inputs.

Pick-and-place is not a target. 2 of 8 test scenes had a graspable object under the selection rules (dev: 7 of 13). Both pass nominal; s20 passes all five friction/mass cases. s08 fails two: high friction ends 13.5 cm from goal; high mass (4.5x nominal) diverged (object at 11.9 km), read as simulator failure, uninvestigated.

Limits: test scenes were seen in Phases 1 to 4, so this is not a clean first look (D-026). Size and mass targets are provisional; mass was lowered from 70% (D-025). Friction/restitution are mostly unsourced placeholders (D-019); restitution was never simulated. Pick-and-place control is open loop and skips difficult objects.

Note: this run used 6 SAM 3D draws per object. D-030 later set the default to 2 based on dev-only measurement (`docs/research/sample_counts.md`).
