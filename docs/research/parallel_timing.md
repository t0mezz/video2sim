# Parallel shape and physics (D-031)

Scene s02 (dev, 5 objects, 4 needing CoACD, 2 draws/object), M4 MacBook Air 10 cores. Commands: `benchmark/profile_stages.py`, `benchmark/profile_coacd.py`. One hosted run per arm, each from empty cache, 6 min apart for cold GPUs.

| Stage | Sequential | Pipelined | Change | Mean cores | Peak memory |
| --- | --- | --- | --- | --- | --- |
| Shape | 170 s | 126 s | −26% | 0.18-0.25 | 3.2-4.3 GB |
| Physics | 144 s | 77 s | −47% | 1.5-3.1 | 4.0-4.6 GB |
| Both | 315 s | 202 s | −36% | | |

Mass requests (43 s of work) overlap in the pipelined arm. Registration is 12 s of an otherwise waiting stage — pipelining gains little (shared Python lock; 3× thread time for 2 s wall gain on cached draws). CoACD is 78 s busy alone, 88 s with two at once. Repair 12 s, mass 1.5 s, settling 1.2 s. Cached-draw outputs are byte-identical across arms; hosted runs are not (same seed gives fit-score diffs ≤0.017; mass answers redrawn).

CoACD capacity (4 meshes × 2 rounds, separate processes, identical results every row): 1-at-once 81-94 s wall (2.5-2.7 cores, 1.3 GB); 2-at-once 67 s (4.6 cores, 2.4 GB); 3-at-once 59-60 s (5.5 cores, 2.7 GB); 4-at-once 60 s (5.5 cores, 2.2 GB). CoACD is already threaded (~2.7 cores avg, bursts near 10; `OMP_NUM_THREADS` 1 vs 4 changes nothing) — 3-at-once is the ceiling here. Floor is the slowest mesh alone (44-60 s). Single-scene, single-machine figures; larger scenes may gain more from 3-4.
