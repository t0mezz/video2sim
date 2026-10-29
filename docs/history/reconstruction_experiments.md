# Reconstruction experiments (PLAN.md section 12, items 6 and 6b)

Moved verbatim from `PLAN.md` on 2026-10-29. `PLAN.md` keeps the verdicts and the open steps.

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

