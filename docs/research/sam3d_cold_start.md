# SAM 3D cold start and GPU memory

Probe of the deployed worker (`benchmark/probe_sam3d.py`, `benchmark/results/sam3d_probe_s02.json`): 10 hosted calls on s02, one run, no retries.

| Call | Wall | Generation | Elsewhere |
| --- | --- | --- | --- |
| Cold, one call | 110 s | 24 s | 64 s weights load, 22 s container start + transfer |
| Warm, three calls | 16-18 s | 10 s | 7-8 s image + 14-16 MB GLB transfer |

First draw per container takes ~24-30 s; later draws ~10 s. Six simultaneous cold calls started only 2 containers: one served 5 in turn (~11 s apart, done at 97-143 s), the other 1 (148 s). `parallel_samples: 6` does not yield six GPUs — Modal scales slowly and each container runs one call at a time. Peak PyTorch allocation 18.1-18.9 GB, reserved 24.2-27.3 GB of 79 GB on A100-80GB (allocator figures only; upstream setup notes require ≥32 GB).

A100-40GB with two warm containers (D-032): same 18.1-18.9 GB allocated / 24.2-27.3 GB reserved of 39.5 GB; no OOM in 42 calls; same 10 s draws; cold 92 s, warm 13-14 s. Burst split 3+3 and finished at 117 s vs 148 s on 80 GB (where one container took 5/6). Shape stage on s02 (10 draws, cold start before warm-up): 80GB-cold 126 s; 40GB-cold 151 s; 40GB one-container warm-up 122 s; 40GB two-container warm-up 62 s — one run each, so cold figures differ beyond GPU type. Warm calls must overlap (each container stays busy `warm_hold_s`). Warm-up costs ~110 s, intended to overlap segmentation (unmeasured inside full `video2sim run`); its GPU time is unpriced.
