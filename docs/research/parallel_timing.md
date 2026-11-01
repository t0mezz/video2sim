# Parallel shape and physics (D-031)

Scene s02 (dev, 5 objects, 4 needing CoACD, 2 draws per object), M4 MacBook Air with 10 cores.
Commands: `benchmark/profile_stages.py`, `benchmark/profile_coacd.py`. One hosted run per arm, each
from an empty cache, with a 6 minute gap so that GPUs went cold.

| Stage | Sequential | Pipelined | Change | Mean cores | Peak memory |
| --- | --- | --- | --- | --- | --- |
| Shape | 170 s | 126 s | -26% | 0.18 to 0.25 | 3.2 to 4.3 GB |
| Physics | 144 s | 77 s | -47% | 1.5 to 3.1 | 4.0 to 4.6 GB |
| Both | 315 s | 202 s | -36% | | |

Inside these runs: mass requests took 43 s of work in both arms, and now overlap. Registration is
12 s of a stage that is otherwise waiting, so pipelining it gains little: threads share Python's
lock, and on cached draws its work took 3 times as long in thread time for a 2 s gain in wall time.
CoACD was 78 s of busy time alone and 88 s with two at once. Repair 12 s, mass 1.5 s, settling 1.2 s.

On cached draws, sequential and pipelined outputs are byte-identical. Hosted runs are not comparable
that way: the same seed gave slightly different SAM 3D meshes (fit scores differ by up to 0.017), and
mass answers are drawn fresh.

## CoACD capacity

Four meshes, two rounds each, in separate processes. Results were identical in every row.

| At once | Wall | Mean cores | Peak memory |
| --- | --- | --- | --- |
| 1 | 81 to 94 s | 2.5 to 2.7 | 1.3 GB |
| 2 | 67 s | 4.6 | 2.4 GB |
| 3 | 59 to 60 s | 5.5 | 2.7 GB |
| 4 | 60 s | 5.5 | 2.2 GB |

CoACD is already multi-threaded, at about 2.7 cores on average with bursts near all 10, and
`OMP_NUM_THREADS` of 1 or 4 changes nothing. So 3 at once is the ceiling here. The floor is the
slowest mesh, about 44 to 60 s alone, which parallel objects cannot shorten. These are four meshes
of one scene on one machine, so a scene with more meshes may gain more from 3 or 4.
