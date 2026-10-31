# How many SAM 3D draws does the shape stage need?

Replay of cached draws on the dev scenes (2026-10-31), no hosted call, test split not used:
`benchmark/run_sample_counts.py`, `benchmark/compare_shape_arms.py`,
`benchmark/results/sample_counts_dev.json`. 13 scenes, 56 objects matched by every arm. Keyframes
in their P2.4 quality order: n = 1 is the best keyframe, n = 2 the best two, one draw each. Each
repeats over the three cached seeds. Fit score still picks the kept draw, and kept meshes are
decimated to 100k faces in every arm; n = 6 still gives the 5.05% of the earlier dev runs.

| Paired, 56 objects | n = 6 | n = 2 (three seeds) | n = 1 (three seeds) |
| --- | --- | --- | --- |
| Size error, median | 5.0% | 6.2, 5.3, 5.5% (mean 5.7) | 5.7, 6.1, 6.9% (mean 6.2) |
| Objects over 15% | 10 | 11, 8, 11 | 12, 9, 14 |
| Proxy hulls | 2 | 2, 3, 3 | 2, 5, 3 |
| Position error, median | 5.5 mm | 5.5, 5.9, 5.8 mm | 5.8, 5.8, 5.6 mm |
| Orientation error, median | 1.06 deg | 1.04, 1.07, 1.54 deg | 1.16, 1.10, 1.38 deg |
| Cluster to mesh, median | 1.2 mm | 1.3, 1.3, 1.3 mm | 1.2, 1.3, 1.4 mm |

Pose barely moves. Mesh size gets worse by about 0.7 points at n = 2 and 1.2 points at n = 1, and
the spread between seeds is as large as the gap between arms. Mass and settling were not measured,
and one dev set cannot separate n = 1 from n = 2 cleanly.
