# Final benchmark (P5.6)

Run on 2026-10-31, code 21d34de, test split, once, with nothing tuned afterwards. Every scene went
from video to package with one `video2sim run` and no manual steps. Seven scenes ran on the Mac one
at a time; s20 ran on the GPU workstation (Linux, same commit, same config, same video hash). Scores:
`benchmark/results/p5_6_eval/`. Pick-and-place: `benchmark/results/p5_6_pick_place_test.json`.

An earlier attempt with three videos at once crashed the Mac after 6.5 minutes, most likely from
memory pressure, so its partial output was discarded and no result comes from it.

**Coverage.** 8 rigid test scenes ran. 7 are scored: s10 has no established clean-board origin, and
s23 is left out as deformable (D-014). The medium tier has 2 scenes and 9 objects.

**Validation.** MuJoCo only (D-027). USD loading in Isaac Sim and all PhysX behavior are not verified.

| Target | Result | Limit | Met | Cause of a miss, or note |
| --- | --- | --- | --- | --- |
| Object recall | 93.9% (31 of 33) | 90% or more | Yes | Missed: s15 obj_021, s16 obj_003. Medium tier 78% |
| Size error, median | 5.7% | 5% or less | **No** | Easy 3.3%, medium 7.2%, hard 12.2%. See below |
| Position error, median | 4.3 mm | 15 mm or less | Yes | |
| Orientation error, median | 1.5 deg | 10 deg or less | Yes | 17 objects with a defined yaw |
| Mass within 30% | 29% (9 of 31) | 50% or more | **No** | See below |
| Stability | 7 of 7 scored, 8 of 8 run | 95% or more | Yes | MuJoCo, round objects not counted (D-025) |
| Asset validity | 33 of 33 scored, 37 of 37 run | 100% | Yes | No object skipped |
| Runtime, worst scene | 14.9 min | 60 min or less | Yes | SAM 3D on hosted GPUs, the rest on a laptop CPU |
| Manual steps | 0 on 8 of 8 runs | None | Yes | From the run records; `eval` still reports `not_measured` |

## Size error

The target is missed on the pooled median, and only there: the easy tier is at 3.3%. The misses are
hard and medium objects, largest first: s19 obj_010 122%, s20 obj_044 75%, s15 obj_038 33%, s15
obj_040 30%, s16 obj_004 27%. These are the known limits of Phase 3: registration keeps most of SAM
3D's proportions, and thin, small or occluded objects have few observed points. This was not
diagnosed object by object here. The Phase 3 and 4 gates read 5.4% and 6.0% on these videos, so the
miss is stable.

## Mass

**29% is lower than the 45% of the Phase 4 gate on the same videos, with no change to the mass
code, material table or config since.** So 6 of 31 objects moved out of tolerance and 1 moved in,
with median mass error going from 33% to 41%. Both runs drew fresh answers from the hosted models
and fresh SAM 3D samples, and 21 objects took the model's mass in both runs. The two runs are one
draw each, so about 29 to 45% is the honest range for this method on these objects, and D-026
should be read with that in mind. Two of the objects that flipped were already at the limit (29.7%
to 30.1%, 30.0% to 35.2%). Which source moved the other flips was not separated. The underlying
cause stays as in D-024 and D-026: material class and fill cannot be seen in a video, and hollow
objects are far from uniform density. No cheap fix remains on these inputs.

## Pick-and-place on the test scenes

This is not a target. Only 2 of 8 test scenes had an object within the selection rules (real mesh,
fits the gripper, free room), against 7 of 13 on dev. Both pass the nominal case. s20 passes all
five cases. s08 fails two: on high friction the object ends 13.5 cm from the goal, and on high mass
the simulation diverged (the object reached 11.9 km). The high mass there is 4.5 times the nominal,
and the divergence was not investigated, so read that case as a simulator failure, not a measured
outcome.

## Limits of this result

* Numbers are from a benchmark whose test scenes were seen at Phases 1 to 4, so they are not a
  clean first look (D-026).
* Size and mass targets are provisional, and the mass target was lowered from 70% (D-025).
* Friction and restitution are mostly unsourced placeholders (D-019), and restitution has never
  been simulated.
* The pick-and-place control is open loop and object selection skips difficult objects.
