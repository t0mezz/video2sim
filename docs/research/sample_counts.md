# SAM 3D draw count

Cached-draw replay on dev (2026-10-31), no hosted calls, test untouched: `benchmark/run_sample_counts.py`, `benchmark/compare_shape_arms.py`, `benchmark/results/sample_counts_dev.json`. 13 scenes, 56 objects matched by all arms. Keyframes in P2.4 quality order (n=1 best keyframe, n=2 best two, one draw each), three cached seeds each. Fit score picks the kept draw; all kept meshes decimated to 100k faces; n=6 holds the earlier 5.05%.

| Paired, 56 objects | n=6 | n=2 (3 seeds) | n=1 (3 seeds) |
| --- | --- | --- | --- |
| Size error, median | 5.0% | 6.2 / 5.3 / 5.5% (mean 5.7) | 5.7 / 6.1 / 6.9% (mean 6.2) |
| Objects >15% | 10 | 11 / 8 / 11 | 12 / 9 / 14 |
| Proxy hulls | 2 | 2 / 3 / 3 | 2 / 5 / 3 |
| Position median | 5.5 mm | 5.5 / 5.9 / 5.8 mm | 5.8 / 5.8 / 5.6 mm |
| Orientation median | 1.06° | 1.04 / 1.07 / 1.54° | 1.16 / 1.10 / 1.38° |
| Cluster-to-mesh median | 1.2 mm | 1.3 / 1.3 / 1.3 mm | 1.2 / 1.3 / 1.4 mm |

Pose is flat across arms. Size degrades ~0.7 pts at n=2 and ~1.2 pts at n=1; seed spread matches arm gaps, so one dev set does not cleanly separate n=1 from n=2. Mass and settling unmeasured.
