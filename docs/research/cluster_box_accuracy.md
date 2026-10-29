# Cluster box accuracy: what limits it and what moved it (Phase 2 box gate)

Written 2026-10-29. The Phase 2 exit criterion "point clusters on the easy scenes have bounding
boxes within 10% of measured size" is the median, over matched easy-tier objects, of the
per-object size error, which is the **worst** of the three axes of the resting box
(`evaluation/metrics.py`, `gates.segmentation.cluster_bbox_error_max`). The dev easy tier is
s01, s02, s03, s04 and s06 (s05 is unscored: the clean-board origin is not established), so the
gate is the 10th of 19 objects. All numbers are on the `_b` runs in `runs/`, dev split only.
Evidence file: `benchmark/results/p2_3b_dev.json` (`benchmark/run_lift.py ... --score`).

## Result

| cluster | easy median | easy within 10% | all dev median | all within 10% |
| --- | --- | --- | --- | --- |
| hull (P2.3) | 36.4% | 3/19 | 35.3% | 5/54 |
| sparse points (P2.3, before; scratch rescore) | 18.7% | 7/19 | 18.7% | 16/55 |
| **points, trimmed** (`trim_points_per_side: 2`) | **8.9%** | 10/19 | 13.2% | 21/55 |
| **fitted** (`primitive_fit`, from the trimmed points) | **4.3%** | 15/19 | 7.3% | 33/55 |

The fitted cluster is what `ObjectInstance.point_cluster_path` holds (human decision,
2026-10-29), so `video2sim eval --stage segmentation` scores it. The trimmed points are written
beside it as `segmentation/clusters/obj_NNN_points.npy`, the hull as `obj_NNN_hull.npy`, and
`segmentation/fit.json` records per object the kind, size, mask cost and whether the cluster is
the primitive, its union with the points, or the points alone.

Through `SegmentationStage` (all 18 runs in `runs_instances/`, no hosted call), the clusters at
`point_cluster_path` score the same: 4.3% easy-tier median, 15 of 19 within 10%
(`benchmark/results/p2_6b_instances.json`). `video2sim eval --stage segmentation`
reports the same on the dev split once it aligns the clusters to the board frame (fixed
2026-10-29; before, it matched 2 of 59 objects): recall 94.9%, easy-tier median 4.3% over 19
objects (`benchmark/results/p2_eval_segmentation_dev.json`).

## Diagnosis

Per-axis signed errors of the P2.3 sparse-point clusters showed two separate faults.

1. **Stray points set the box.** The sparse clusters were mostly *too large*, not too small.
   Flat boxes read 15 to 22 mm too tall (phone box 62 to 78 mm, in-ear box 45 to 64 mm,
   electronics box 55 to 77 mm) because 1 to 4% of their points float above the top face, and
   the s06 phone box read 72% too wide because of a single isolated point. These are short SfM
   tracks triangulated from nearby frames: their depth is poor, and on the short `_b` camera
   arcs they still project into the object's mask in at least 90% of the 40 views, so the vote
   keeps them. A box is set by its outermost points, so a handful is enough.
2. **Without the strays the boxes are too small.** Points lie only on visible, textured
   surfaces: plain edges, white parts and hidden faces have none. After trimming, 6 of the 9
   failing easy objects were *under* (mugs -12% and -13% along the handle, action camera -11%,
   deodorant cap -33% with 20 points, car body height -14%, in-ear box -12%).

The masks carry the missing evidence: SAM's masks match the projected ground-truth boxes with
IoU 0.96 (P2.3), and the outline of an object in many views pins down its extent even where no
feature was triangulated. The visual hull uses that evidence but is loose along viewing
directions the short arc never saw, which is why it reads about 35% large.

## Changes, ranked by effect

**1. Trim the outermost points (cheap, big).** `lift.trim_extremes` drops the 2 outermost points
on each of the six sides of the resting box (long side, short side, height; axes from
`cv2.minAreaRect`, the same convention as the ground truth). A cluster with at most
`6 * k + min_points` points is left whole. Effect: 18.7% to 8.9% on the easy tier. It passes by
one object, so it is fragile on its own. Trimming 4 per side gave 7.5% in a scratch run, but
undershoot grows with it, so 2 was kept.

**2. Fit a primitive to the masks (bigger, slower).** `stages/primitive_fit.py` fits an upright
box, an upright elliptic cylinder and a sphere, all standing on the table, by a seeded random
walk from the box of the trimmed points; the kind whose projected outline disagrees least with
the masks wins. Outline pixels on another track's mask cost nothing (occlusion), and views where
the mask touches the image edge are skipped. When the best cost is above `max_cost` (0.10), the
cluster is the union of the primitive outline and the points, so a handle or wheels still count.
Effect: 8.9% to 4.3% on the easy tier, 13.2% to 7.3% on all dev scenes.

- Box-shaped objects and round ones come out within 0 to 7% (cubes, packages, cans, balls, the
  metal bottle that the points read 23% wide, the s06 phone box that one stray point made 59%
  wide).
- The kind choice matters: a box alone shrinks onto a can or a ball (box-only median 21%),
  because a box cannot match a round outline.
- Cost: 1 to 12 s per scene on the M4 (300 steps per kind, masks at a quarter of full size).

**3. Mask quality on white-on-white parts (not done; segmentation, not lift).** After the fit,
the easy failures are the in-ear box (white tab against the white board: -18% with every method)
and the two mugs (white handles; the s02 obj_041 union is 32% off, its points 12%). Both the
masks and the SfM features miss those parts, so no lift change can recover them.

## Tried and not kept

All on the same dev runs, easy-tier median:

| variant | easy median | note |
| --- | --- | --- |
| extend the cluster down to the table plane | 18.9% | heights were mostly too large, so it made things worse; after trimming it still did not help |
| SfM track length >= 4 or 5 | 10.1%, 9.6% | removes strays and good edge points alike; worse when combined with trimming |
| reprojection error < 0.75 to 1 px | about 10.3% | same |
| triangulation angle >= 3 deg | 12.2% | |
| `min_agreement` 1.0 | 10.1% | drops 5 matched objects on the other tiers |
| statistical or radius outlier removal | 12 to 23% | strays sit near dense faces; thinning hurts edges |
| largest connected group (7 to 10 mm) | 29 to 73% | sparse clusters break into pieces |
| mask erosion before voting (3 px) | 14% | loses edge points |
| all frames instead of 40 | 18.7% | no change |
| vote only in the frames where SfM observed the point | 75% | lets the mis-triangulated short tracks in: they are inside the mask in the 3 frames that see them |
| clip the hull to the box of its points (P2.3 scratch) | 26% | |

## What could go wrong

- **Tuned on the scored set.** `trim_points_per_side` and `max_cost` were chosen on the same
  19 dev objects. The easy test scenes (s07 to s10) have no runs yet, so this is not a
  validated pass.
- **The gate sits on a few objects.** The trimmed points pass by one object; the fitted cluster
  by five.
- **The cluster is a shape, not observed points.** For a primitive fit, the cluster file holds
  the primitive outline (8 box corners, two rings of 48 cylinder points, or a sphere grid). A
  later stage that wants observed surface samples (registration in P3.2) reads
  `obj_NNN_points.npy`, and `fit.json` for the primitive; both sit outside the typed output, so
  P3.2 may want them in `ObjectInstance` (a `schemas.py` change).
- **Short arcs.** Everything was measured on `_b` paths (45 to 130 degrees of azimuth). The full
  orbits (`video_a`, s01 to s04) were not tracked: that needs a first upload of those videos to
  the hosted SAM service.
- **Ground truth.** Sizes were taken with a folding rule, three readings each (`inventory.csv`);
  positions are about 1 cm off in places (visible in top views of s01 and s06), which does not
  affect the size metric.

## Decision (human, 2026-10-29)

The Phase 2 box criterion reads the fitted cluster: `point_cluster_path` points to it. No
`schemas.py` field changed; the field's meaning did. This supersedes the P2.6 choice of the
sparse-point lift.
