# SAM 3D cold start and GPU memory

Probe of the deployed worker on A100-80GB, 2026-10-31: `benchmark/probe_sam3d.py`,
`benchmark/results/sam3d_probe_s02.json`. Ten hosted calls on s02, one run, no retries.

| Call | Wall | Generation | Elsewhere |
| --- | --- | --- | --- |
| Cold, one call | 110 s | 24 s | 64 s loading weights, 22 s container start and transfer |
| Warm, three calls | 16 to 18 s | 10 s | 7 to 8 s transfer of the image and the 14 to 16 MB GLB |

The first draw on a container takes about 24 to 30 s, and later ones about 10 s.

**Burst.** Six calls at once on cold containers started only 2 containers. One served 5 calls in turn,
about 11 s apart, finishing at 97 to 143 s; the other served 1 at 148 s. So `parallel_samples: 6`
does not give six GPUs: Modal scaled slowly, and each container runs one call at a time.

**GPU memory.** Peak allocated 18.1 to 18.9 GB, peak reserved by PyTorch 24.2 to 27.3 GB, of 79 GB.
These are PyTorch's numbers, so they leave out memory outside its allocator. Upstream's setup notes
ask for at least 32 GB.

## A100-40GB with two warm containers (D-032)

Same probe on A100-40GB: peak allocated 18.1 to 18.9 GB, peak reserved 24.2 to 27.3 GB of 39.5 GB,
as on 80 GB. No out-of-memory error in 42 calls. Draws take the same 10 s. Cold call 92 s, warm calls
13 to 14 s. The burst split 3 and 3 across the two containers and ended at 117 s,
against 148 s on 80 GB, where one container took 5 of the 6 calls.

Shape stage on s02, 10 draws, all from cold containers before the warm-up:

| Arm | Shape wall time |
| --- | --- |
| A100-80GB, cold | 126 s |
| A100-40GB, cold | 151 s |
| A100-40GB, warm-up without a hold (one container warmed) | 122 s |
| A100-40GB, two containers warmed | **62 s** |

One run each, so the 126 and 151 s cold figures differ by more than the GPU can explain. Warm calls
must overlap, or both land on one container, which is why each stays busy for `warm_hold_s`. The
warm-up itself takes about 110 s and is meant to overlap segmentation; that timing inside a full
`video2sim run` was not measured. The warm-up's GPU time was not priced.
