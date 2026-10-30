# SAM 3D Objects, single view against multi-view variants: quality, cost, fit for Phase 3

Read 2026-10-29 from primary sources (list at the end). This builds on `sam3d.md`
(2026-10-28) and does not repeat it. Items marked UNVERIFIED were not confirmed in a primary
source. Paper text was read through a fetch tool that summarises; repository source files and
GitHub issue and API data were fetched as plain text and read directly. Nothing was cloned,
installed, downloaded as weights or run.

## Verdict

| Question | Answer |
| --- | --- |
| Primary path for P3.1 and P3.3 | Official single-view SAM 3D Objects. Several samples from the best keyframes, then P3.2 registration. |
| Role of MV-SAM3D | A bake-off arm for shape only in P3.3, behind the same adapter interface. Not a dependency, and not a source of pose. |
| Does multi-view fusion improve pose or scale? | No evidence that it does. In MV-SAM3D the pose comes from view 0 only. Its layout gains come from a post-hoc fit to metric depth, which P3.2 already does. |
| Does it improve shape? | Yes on clean renders (GSO-30 Chamfer down 59% at 5 views). An independent paper finds only about 1% to 5% over the best single view on occluded real scenes. Choosing the view mattered more. |
| Cost difference | Sampling cost grows about linearly with the number of views. Either way the benchmark costs tens of dollars on Modal, not hundreds. |

Comparison (evidence; sources in section 7):

| Row | SAM 3D (official, single view) | MV-SAM3D (devinli123) |
| --- | --- | --- |
| Quality: shape | SA-3DAO F1@0.01 0.234, Chamfer 0.040 (best of 7 in the paper). GSO-30 Chamfer 42.0 x1e-3 (from MV-SAM3D's Table 1) | GSO-30 Chamfer 20.2 (2 views), 17.3 (5 views) x1e-3. Not reproduced by users (issue #18). No eval script |
| Quality: pose/scale | ADT ICP-Rot 15.3 deg, ADD-S@0.1 0.77; SA-3DAO 20.8 deg, 0.72. Scale is in pointmap units, so metric if the pointmap is metric (contributor, issue #57) | Pose taken from view 0 only (code, author in issue #1). Scene Acc@5cm 55.7 -> 60.2 from fusion, -> 91.7 with physics-aware refinement. The collision and contact terms are not in the released code (issue #19) |
| VRAM | 32 GB stated minimum (V100, `doc/setup.md`, contributor in issue #6). 24 GB works if mesh decoding is skipped (users, UNVERIFIED) | 19.9 GiB (3 views), 21.9 GiB (8 views); 12.2 and 14.0 GiB with `--low_vram` (wyim-pgl fork, RTX 4090). OOM on 48 GB with visibility weighting reported (issue #7, unanswered). Paper used an A100 80 GB |
| Runtime per object | 8 to 12 s per sample on H100 (user, issue #64); 3.5 s on H20 with 10 and 1 steps, Gaussians only (user, issue #65). Paper gives only relative numbers: distilled 4-step is about 10x faster | About 50 s per reconstruction plus about 40 s weight loading per process on RTX 4090 (wyim-pgl fork; view count not stated). Sampling grows linearly with views: one denoiser call per view per step (code). Plus a DA3 run per scene |
| Hosted $ (103 objects x 4 samples, Modal) | About $4 to $14 | About $14 to $41 at 5 views, plus about $3 to $5 for DA3 and weight loading |
| Inputs | One RGB image, one binary mask, optional dense HxWx3 pointmap (MoGe by default) | Images, one RGBA mask per object per view, `da3_output.npz` and DA3's `scene.glb` (for visibility weighting, merge and pose fit) |
| Maturity | Meta, 7,606 stars, 113 open issues, last commit 2026-06-02 | One author, 11 commits, 12 open issues, last commit 2026-05-30. Active community fork wyim-pgl (last commit 2026-10-23, 0 stars) |
| License | SAM License (code and weights) | `LICENSE` is the SAM License text (GitHub API reports NOASSERTION). DA3NESTED-GIANT-LARGE weights are CC BY-NC 4.0 |

## 1. Quality evidence

### 1.1 MV-SAM3D paper (arXiv 2603.11633v2, 9 April 2026)

- Shape, GSO-30, Chamfer x1e-3 (Table 1): SAM 3D at 1 view scores 42.0, TRELLIS 44.5,
  MV-SAM3D 20.2 at 2 views and 17.3 at 5. EscherNet 21.5 and 17.5. GSO views are renders at
  known poses of clean, unoccluded single objects.
- Novel views, GSO-30 (Table 2): PSNR 19.82 (SAM 3D) against 23.45 (2 views) and 25.22 (5 views).
- Scenes, MV-SAM3D-Scenes (Table 3): own capture, 2 to 8 objects, 15 viewpoints each. SAM 3D
  against MV-SAM3D without optimisation against full: PSNR 19.52 / 19.95 / 21.83; depth MAE
  0.079 / 0.068 / 0.025; Acc@5cm 55.66 / 60.23 / 91.66; RelAcc@5% 44.07 / 49.52 / 91.13.
  Metrics are not defined in the paper, units are not stated, view count not stated.
- Physics ablation (Table 6): post-refinement alone lifts Acc@5cm from 60.23 to 87.82 and the
  collision rate from 15.32% to 0. Layout injection alone gives 71.45. Both give 91.66.
- Views (Table 5): Acc@5cm 85.20 (1 view), 90.45 (2), 91.18 (3), 91.66 (5). The 1-view row
  does not match Table 3's 55.66; the paper does not explain this. It reads as if refinement
  was on, so views add about 6 points once the pose is fitted to depth (inference, UNVERIFIED).
- The refinement (Eq. 9) optimises only a similarity transform per object, mesh fixed, with
  Chamfer distance to a point cloud from metric depth, plus collision and contact terms.
- No limitations section. One failure mode named: symmetric or repetitive objects confuse
  front and back in the entropy weighting. Thin, small, tabletop objects are not discussed.

### 1.2 What the code does with pose

- `sam3d_objects/pipeline/multi_view_utils.py`: `POSE_KEYS` (translation, rotation, scale and
  variants) are "View 0 only"; shape latents are averaged or entropy-weighted across views.
- Author, issue #1: "Our multi-view extension does NOT affect the pose prediction accuracy";
  pose is "in View 0's camera coordinate system". Issue #8: the reference view is view 0.
- `sam3d_objects/pose_align/pose_optimization.py` (`--run_pose_optimization`) is a Chamfer fit
  of scale, rotation and translation to DA3 points picked by mask back-projection with a 10%
  relative depth tolerance. No collision or contact term or layout injection is in the
  repository (grep of the source, and issues #16 and #19). Issue #19 reports that the released
  pose optimisation makes a 4-object ScanNet++ layout worse.
- Users in issue #9 and #18 found the simple average better than the weighted version and
  could not reproduce the GSO numbers. No author answer on #18.

### 1.3 Independent check: Scene-SAM3D (arXiv 2607.16805v1, 18 July 2026)

A second training-free multi-view SAM 3D method for calibrated views (known camera-to-world
poses and intrinsics, estimated depth). Indoor scenes, furniture-sized objects. Table 1,
instance Chamfer (Replica / ScanNet++): SAM 3D one random view 61.96 / 79.25; SAM 3D one
view chosen by coverage 41.60 / 57.71; plain multi-diffusion with 4 views 40.98 / 58.51;
Scene-SAM3D with 4 views 39.37 / 57.36. So choosing the best view cut Chamfer by about a third;
fusing 4 views added about 1% to 5% on top, and plain averaging was worse than one good view
on ScanNet++. Scene level (Table 2): Gaussian layout refinement against multi-view masks, RGB
and depth cut Chamfer 43.8% (Replica) and 30.9% (ScanNet++) against ICP. Code repository
`xibi777/Scene-SAM3D` is empty as of today.

### 1.4 SAM 3D paper (arXiv 2511.16624)

- Shape, SA-3DAO (Table 2): F1@0.01 0.2344, vIoU 0.2311, Chamfer 0.0400, EMD 0.1211, best
  of seven methods.
- Layout (Table 3), joint prediction: SA-3DAO 3D IoU 0.425, ICP-Rot 20.77 deg, ADD-S 0.266,
  ADD-S@0.1 0.723; ADT 0.497, 15.25 deg, 0.077, 0.767. SAM 3D shape plus FoundationPose is
  worse (ADT 18.91 deg). ADD-S is normalised by object diameter; no translation or scale error
  in cm is reported.
- Appendix E.5: with and without pointmap, shape preference is 48% each (LVIS). No layout
  ablation table on pointmaps was found.
- Limitations (appendix F): 64^3 shape grid loses thin detail; one object at a time with no
  contact, stability or ground plane reasoning; texture is pose-unaware, so symmetric objects
  can look rotated; mask errors compound.
- Runtime: none in absolute terms. Distilled geometry model about 10x faster at 4 steps and
  about 38x at 1 step than 25-step flow matching with guidance.

### 1.5 Small objects and metric scale in practice (wyim-pgl fork, non-author, one capture)

`docs/wiki/Metric-Scale.md` and `docs/Dimensional-Accuracy.md`: an 8 mm bearing at 18 to
28 px came out as a sphere sliced flat (volume 12% of its hull). A cup came out 105 to 109 mm
high against 110 mm; rim and base estimates were off by -4% to +7%. The fork states
MV-SAM3D outputs are "not in physical units" for their captures and scales them with a
reference sphere. This conflicts with the DA3 README, which says DA3NESTED output "is
already in meters". Development evidence on one scene, not a benchmark.

## 2. Cost

### 2.1 Facts

- Single view, official: 32 GB minimum (`doc/setup.md`). Users report 8 to 12 s per sample on
  H100 (issue #64) and 3.5 s on H20 with fewer steps and no mesh (issue #65). Memory grows
  with the stage 1 voxel count; above about 10k voxels OOM is likely (user, issue #6).
- MV-SAM3D: the denoiser is called once per view per step in a Python loop
  (`multi_view_weighted.py`, lines 440 to 465 and 640 to 656), plus a one-step warm-up pass
  to collect attention. So sampling FLOPs grow linearly with views; decoding is once per
  object. Defaults: `--stage1_steps 50`, `--stage2_steps 25`, decode `gaussian,mesh`.
- Scene-SAM3D Table 4 (H200): plain multi-diffusion SAM 3D with 10 views, 3,120 TFLOPs and
  43.18 s of flow sampling (per object or per scene not stated). About 4.3 s per view.
- wyim-pgl fork: about 40 s to load 13 GB of weights per process, about 50 s per
  reconstruction, RTX 4090.
- Modal, per second, read today: H100 $0.001097, A100 80 GB $0.000694, L40S $0.000542,
  H200 $0.001261. Baseten lists H100 at $0.10833 and A100 at $0.06667 (per minute, as read).

### 2.2 Estimate for the benchmark (opinion, assumptions stated)

Assumptions: 103 objects, 4 samples each (412 calls). Single view takes 10 to 30 s per call on
H100 (the user figure, widened for mesh decoding and postprocessing). MV-SAM3D at 5 views takes
the single-view time plus about 4.3 s per extra view, then widened 3x: 30 to 90 s. A100 is
1.5x slower than H100 (UNVERIFIED). One container start per scene and video (48), 40 s each.
DA3 at about 60 s per video (UNVERIFIED).

| Arm | H100 | A100 80 GB |
| --- | --- | --- |
| Single view, 412 calls | 4,120 to 12,360 s: $4.52 to $13.56 | 6,180 to 18,540 s: $4.29 to $12.87 |
| MV-SAM3D, 5 views, 412 calls | 12,360 to 37,080 s: $13.56 to $40.68 | 18,540 to 55,620 s: $12.87 to $38.60 |
| Weight loading, 48 starts | 1,920 s: $2.11 | $1.33 |
| DA3, 48 videos (MV only) | 2,880 s: $3.16 | $2.00 |

A full P3.3 bake-off with three arms and two reruns stays near $100. Cost does not decide
between the two; engineering time and the license of DA3 do. Note that 4 single-view samples
from 4 keyframes cost about as much as one MV-SAM3D sample over the same 4 views.

## 3. Inputs, outputs and fit

- MV-SAM3D layout: `scene/images/0.png ...` and `scene/<object>/0.png ...` RGBA masks (RGB
  images are fine, author in issue #3). `--image_names` picks views; view 0 is the reference.
- `da3_output.npz` (from `scripts/run_da3.py`): `depth` (N,H,W), `pointmaps` (N,H,W,3, camera
  frame of view i), `pointmaps_sam3d` (N,3,H,W, required key), `extrinsics` (world-to-camera),
  `intrinsics`, `image_files`, `process_res` (504 default). Views are matched by file stem.
  Merging, visibility weighting and the pose fit also read DA3's `scene.glb` and its
  `hf_alignment` matrix. `--da3_output` is optional for entropy weighting only (help text).
- Known poses: no flag. Two routes, both untested: write our own npz (pycolmap world-to-camera
  extrinsics, intrinsics, dense depth per keyframe) and a `scene.glb` with an identity
  `hf_alignment`; or call DA3 with our poses, since `DepthAnything3.inference` takes
  `extrinsics`, `intrinsics` and `align_to_input_ext_scale=True` (DA3 `api.py`). Sparse pycolmap
  points are not enough; a dense depth per keyframe is needed (same conclusion as `sam3d.md`).
- Output: `result.glb` (mesh in canonical space), `result.ply` (Gaussians), `params.npz`
  (rotation, translation, scale to view 0's camera, pointmap scale and shift). Merged and
  optimised GLBs are in DA3's aligned frame. So a mesh is available.
- Fit to P3.2: our registration fits a similarity transform against the metric pycolmap
  cloud and the masks in all views. That is the same job as MV-SAM3D's post-refinement, with
  a better target (marker-scaled cloud, one-sided term, mask term). The multi-view pose and
  scale gains in the paper are therefore mostly ones we already plan to get.
- Fit to the Phase 3 exit criteria (1.5 cm, 10 deg): SAM 3D's own layout (15 to 21 deg mean
  ICP-Rot) does not meet 10 deg unaided, with or without fusion. The criteria will be met or
  missed by P3.2 and P3.4. Shape matters for P3.2 through the point-to-mesh fit and for the
  Phase 4 stability pass, which is where multi-view could help on occluded objects.

## 4. Maturity and license

- MV-SAM3D: created 2025-11-27, 11 commits, last commit 2026-05-30 ("Fix punctuation in
  acknowledgments"), 561 stars, 51 forks, 12 open of 19 issues. Missing visualisation folder
  at release (issue #9). Upstream PR #37 (the earlier simple-average version) is still open,
  not merged.
- Forks: `wyim-pgl/MV-SAM3D` adds an installer, `--low_vram`, batch runs, coordinate and
  optimiser fixes, grounding, 193 tests (its docs), last commit 2026-10-23.
  `Roworks-ai/sam-3d-objects-multi` packages PR #37 for Replicate (Cog) on branch
  `multi-view-support`, last commit 2026-02-09. `devinli123/multi-view-sam-3d-objects` is the
  superseded first version, last commit 2025-11-28.
- Official multi-view: none. README news lists only 2025-11-19 (launch), 2026-06-01 (encoder
  weights) and 2026-06-02 (SA-3DAO set and leaderboard). Contributors said in issue #21 it is
  single view only, with multi-view "in the future".
- License: MV-SAM3D `LICENSE` is the SAM License, last updated 19 Nov 2025, same as upstream,
  so D-005 is unchanged for SAM 3D itself. Its DA3 dependency is not: DA3NESTED-GIANT-LARGE
  (and -1.1, GIANT, LARGE) weights are CC BY-NC 4.0, non-commercial. DA3 code is Apache-2.0;
  DA3-BASE, DA3-SMALL, DA3METRIC-LARGE and DA3MONO-LARGE weights are Apache 2.0. Using the
  giant model adds a second non-commercial component next to TRELLIS.2's NVIDIA renderers.
  Feeding our own npz avoids DA3 entirely. Not legal advice.

## 5. Recommendation (opinion)

1. P3.1: build the adapter on the official single-view `Inference` (image, mask, seed,
   optional pointmap). Draw samples over the top P2.4 keyframes rather than many seeds on one
   frame: Scene-SAM3D's single-view coverage result says view choice is the larger lever.
2. P3.2: keep registration as planned; it is the part that decides the 1.5 cm and 10 deg gate.
   Do not use MV-SAM3D's pose optimiser.
3. P3.3: add MV-SAM3D as a fourth arm only after the single-view arm runs end to end on Modal,
   for shape only: the same keyframes and masks, our poses in a hand-written npz, one sample.
   Score it on point-to-mesh distance and the Phase 4 stability pass, split by easy, medium and
   hard and by occlusion. Keep it if it beats the best of the single-view samples on medium and
   hard scenes; drop it otherwise. Pin the commit; prefer the wyim-pgl fixes if they hold up.
4. Do not adopt DA3 giant weights for this; use our own depth (VGGT or rasterised from
   pycolmap under PLAN section 12 item 7), which also keeps metric scale on the marker.

## 6. Open items

Closed from `sam3d.md`:
- Depth model: MoGe v1 (`moge.model.v1.MoGeModel.from_pretrained` in the traceback of issue
  #96; MV-SAM3D ships `depth_models/moge.py`).
- Metric pointmap: a contributor says the output transform maps to pointmap space, so a metric
  pointmap gives metric scale (issue #57); one user reports errors below 1 cm (anecdote).
- Per-call runtime and memory: only user figures (section 2.1). No figure from Meta.
- MV-SAM3D license (SAM License), last commit (2026-05-30), pose input (section 3).

Still open:
- `make_scene` internals.
- Single-view runtime with mesh decoding on A100 and H100: measure in P3.1.
- MV-SAM3D peak VRAM with visibility weighting (OOM at 48 GB reported) and wall time per view.
- Whether MV-SAM3D runs with a hand-written npz and an identity `hf_alignment`.
- Definitions of Acc@5cm and RelAcc@5% in the MV-SAM3D paper, and the Table 5 mismatch.
- Scene-SAM3D code (repository empty) and its depth source.
- Baseten price units (read as per minute).

## 7. Sources (all read 2026-10-29)

- https://arxiv.org/html/2603.11633 (v2): Tables 1 to 6, method, implementation details
- https://raw.githubusercontent.com/devinli123/MV-SAM3D/main/README.md, `LICENSE`,
  `run_inference_weighted.py`, `scripts/run_da3.py`, `sam3d_objects/pipeline/multi_view_utils.py`,
  `sam3d_objects/pipeline/multi_view_weighted.py`, `sam3d_objects/pose_align/pose_optimization.py`
- https://api.github.com/repos/devinli123/MV-SAM3D (metadata, commits, file tree, issues
  #1, #3, #7, #8, #9, #16, #18, #19 and comments)
- https://github.com/wyim-pgl/MV-SAM3D: `README.md`, `examples/README.md`,
  `docs/Dimensional-Accuracy.md`, `docs/wiki/Metric-Scale.md`; API metadata and commits
- https://api.github.com/repos/Roworks-ai/sam-3d-objects-multi (branches, commits);
  https://api.github.com/repos/devinli123/multi-view-sam-3d-objects (metadata)
- https://arxiv.org/html/2511.16624 (full text, three parts): Tables 2 to 4, appendix C.4,
  E.5, F
- https://raw.githubusercontent.com/facebookresearch/sam-3d-objects/main/README.md; API
  metadata; issues #6, #20, #21, #30, #37 (PR), #57, #64, #65, #96, #110, #167
- https://arxiv.org/abs/2607.16805 and https://arxiv.org/html/2607.16805 (Scene-SAM3D v1);
  https://api.github.com/repos/xibi777/Scene-SAM3D (empty)
- https://raw.githubusercontent.com/ByteDance-Seed/Depth-Anything-3/main/README.md and
  `src/depth_anything_3/api.py`; https://huggingface.co/depth-anything/DA3NESTED-GIANT-LARGE
- https://modal.com/pricing; https://www.baseten.co/pricing/
- Search only, not used for claims: Meta SAM 3D blog and press coverage (no multi-view release
  found)
