# SAM 3D Objects: inference API, multi-object use, hardware, license, multi-view variants

Read 2026-10-28 from primary sources (list at the end). Items marked UNVERIFIED were not
confirmed in a primary source. Page text was read through a fetch tool that summarises, so
code quotes below are as reported by that tool, not copied from a local clone. Nothing was
downloaded or run.

## Verdict on the five beliefs

| Belief | Verdict |
| --- | --- |
| (a) Single image + mask, not multiple views | Correct for the official model. |
| (b) Cheaper to extract several objects from one frame | Not supported. The official notebook runs one full model call per mask. The only shared cost is the image and (optionally) the pointmap. |
| (c) Keyframes should serve all objects at once | Not supported by cost. Per-object frame quality matters more. Opinion only, see section 5. |
| (d) Multi-view only via a community project "SAM 3D MV" | Correct that official has none. It is not one project: MV-SAM3D (devinli123, with a paper) plus several forks and an upstream PR #37. |
| (e) SAM 3D accepts a pointmap | Correct, but it must be a dense per-pixel HxWx3 map in the image grid. Sparse COLMAP points are not accepted as such. Metric status is unclear. |

## 1. Inference API of SAM 3D Objects

- Entry point: `notebook/inference.py`, class `Inference`, wrapping `InferencePipelinePointMap`
  (`sam3d_objects/pipeline/inference_pipeline_pointmap.py`).
- `Inference.__call__(self, image, mask, seed=None, pointmap=None) -> dict`. The mask is
  annotated `Optional` but is merged unconditionally: `image = self.merge_mask_to_rgba(image, mask)`,
  which does `mask.astype(np.uint8) * 255` and concatenates it as the alpha channel
  (`np.concatenate([image[..., :3], mask], axis=-1)`). So the mask is required, one binary mask
  per call, and inputs must be NumPy arrays in practice.
- The pipeline call is `self._pipeline.run(image, None, seed, stage1_only=False, ...,
  with_mesh_postprocess=False, with_texture_baking=False, with_layout_postprocess=False,
  use_vertex_color=True, pointmap=pointmap)`.
- `run()` signature (pipeline file): `run(self, image, mask=None, seed=None, stage1_only=False,
  with_mesh_postprocess=True, with_texture_baking=True, with_layout_postprocess=False,
  use_vertex_color=False, stage1_inference_steps=None, stage2_inference_steps=None,
  use_stage1_distillation=False, use_stage2_distillation=False, pointmap=None,
  decode_formats=None, estimate_plane=False)`.
- Pointmap:
  - Optional. If `None`, `compute_pointmap` runs `self.depth_model(loaded_image)` and reads
    `output["pointmaps"]`. The source does not name the depth model in the part read
    (it is not stated to be MoGe: UNVERIFIED which model the shipped config instantiates).
  - Format: HxWx3, x, y, z per pixel (permuted to CHW). If its size differs from the image it is
    resized with nearest-neighbour. Points are converted from "R3 camera space" to PyTorch3D
    camera space via `camera_to_pytorch3d_camera` (look-at eye `[0,0,-1]`, up `[0,-1,0]`).
  - Intrinsics are not supplied by the caller; when a pointmap is passed they are inferred with
    `infer_intrinsics_from_pointmap`.
  - Metric: the code does not say. It normalises the pointmap (`pointmap_scale`, `pointmap_shift`)
    and keeps an unnormalised copy (`rgb_pointmap_unnorm`) for post-optimisation. UNVERIFIED
    whether a metric map is required or expected.
  - Paper section 2.2: pointmaps are optional; the model "can be conditioned on a coarse scene
    point map ... obtained via hardware sensors (e.g., LiDAR on an iPhone)" or monocular depth.
    Section 4: for data without pointmaps, they estimate them. Appendix E.5 (as reported):
    shape and texture quality reportedly do not depend on pointmap conditioning. Pointmap mainly
    affects layout (pose/scale).
  - Sparse COLMAP points: no code path for them. We would have to rasterise/densify to a
    per-pixel map (NaN where missing is plausible, since `_clip_pointmap` uses NaN, but whether
    the model tolerates large NaN regions is UNVERIFIED). Opinion: a dense map from a depth model
    is the supported path.
- Outputs: `run` returns `**ss_return_dict` (pose decoder results: `rotation`, `translation`,
  `scale`), `**outputs` (may contain `"glb"`, `"gaussian"`), plus `"pointmap"` and
  `"pointmap_colors"`. The repo quick start exports `output["gs"].save_ply(...)` (README).
  `make_scene` reads `output["gaussian"]`, `["rotation"]`, `["translation"]`, `["scale"]`.
  Mesh needs `with_mesh_postprocess`/texture baking enabled (defaults in `run`, disabled by
  `Inference`).
- Scale: paper section 3.1.2: "translation and scale in normalized camera coordinates";
  section 2.1/2.2: R, t, s in camera coordinates, scale in R^3. In code,
  `ss_return_dict["scale"] *= ss_return_dict["downsample_factor"]` and post-optimisation
  `refine_scale` averages the three scale channels. So scale is relative to the (normalised)
  pointmap frame, not guaranteed metric. We must register to our metric cloud anyway (P3.2).
- Samples/seeds: one sample per call; `seed` only does `torch.manual_seed`. No sample-count
  parameter. Several seeds means several calls (this matches PLAN P3.1 "draw several").
- Speed knobs: `stage1_inference_steps`, `stage2_inference_steps`, `use_stage*_distillation`.
  Paper 3.2.2: distillation cuts NFE from 25 to 4, "sub-second shape and layout" (no timing
  table seen; hardware for that claim UNVERIFIED).

## 2. Multi-object handling

- `notebook/demo_multi_object.ipynb` loads one image and all `.png` masks in a folder, then:
  `outputs = [inference(image, mask, seed=42) for mask in masks]` and
  `scene_gs = make_scene(*outputs)`.
  So: one full model call per mask, no batching (the pipeline notes "no batch dimension as of
  now"), and the scene is composed afterwards from each object's own predicted pose and scale.
- The layout is not solved jointly. Each object gets pose independently in the same camera frame;
  `make_scene` merges them (its internals were not read: UNVERIFIED whether it does any
  collision handling). MV-SAM3D's paper says independent poses can interpenetrate or float,
  which is consistent with this.
- Cost: no runtime or memory comparison of many-per-frame versus one-per-call is given in the
  repo or the paper section read. By the code, runtime scales linearly with the number of
  masks; the only amortisable work is image loading and, if reused, the depth model/pointmap
  (we could compute it once per frame and pass `pointmap=`). The paper (section 2.2, 3.2.2) does
  not claim a many-object saving. Paper section 4.1 notes that most SA-3DAO scenes contain one object.
- The paper's remaining text (offset 100000 onward of the arXiv HTML) was not read; a runtime
  table may exist there: UNVERIFIED.

## 3. Hardware, license, hosting

- GPU: `doc/setup.md`: "NVIDIA GPU with at least 32 GB of VRAM", 64-bit Linux. PyTorch3D may
  need building on a GPU node. Weights about 20 GB on disk (third-party page, UNVERIFIED in
  primary sources). Note: `docs/licenses.md` says no figure was found on the model/repo page;
  `doc/setup.md` does give 32 GB.
- Access: Hugging Face `facebook/sam-3d-objects` is gated (legal name, date of birth,
  organisation; form not editable). Setup doc: not available in comprehensively sanctioned
  jurisdictions. HF page says no Inference Provider deploys the model.
- License: SAM License (repo `LICENSE`, HF license tag "other"), covering code and checkpoints.
  Reported terms: non-exclusive, worldwide, non-transferable, royalty-free; comply with laws and
  trade controls; no reverse engineering (1(b)(iv)); no ITAR/military/nuclear/espionage uses
  (1(b)(v)); acknowledge use in published research (1(b)(ii)); pass the license on when
  distributing (1(b)(i)); outputs "AS IS", risk on the user (section 3); you own your derivative
  works (5.a); Meta may terminate on breach and may modify the terms (6, 8). No clause restricting
  outputs or commercial use was reported. Not legal advice; see `docs/licenses.md`.
- Modal/Baseten: no documented SAM 3D recipe found in primary sources or search results
  (UNVERIFIED that either works; PLAN D-015 already flags this). Roboflow lists a self-host
  option (third party). Our own hosting terms are already tracked in `docs/licenses.md`.

## 4. Community multi-view variants

There is no single "SAM 3D MV". Found:

- **MV-SAM3D**, `github.com/devinli123/MV-SAM3D` (561 stars when read; 11 commits on main;
  last commit date not visible: UNVERIFIED; license file exists but name not read: UNVERIFIED).
  Paper: arXiv 2603.11633, "MV-SAM3D: Adaptive Multi-View Fusion for Layout-Aware 3D
  Generation" (Baicheng Li, Dong Wu, Jun Li, Shunkai Zhou, Zecui Zeng, Lusong Li, Hongbin Zha;
  v2 April 2026). Training-free. Fusion is a multi-diffusion process in the 3D latent space with
  attention-entropy weighting and visibility weighting per view; for multi-object scenes,
  physics-aware optimisation adds collision/contact constraints. Evaluation: abstract claims
  improvements on "standard benchmarks and real-world multi-object scenes"; a search snippet
  quotes PSNR 19.95 versus 19.52 for the single-view baseline (not checked in the paper).
  Repo has only qualitative comparisons.
- Inputs per README: multiple view images in `scene/images/`, a per-object, per-view RGBA mask
  PNG, and `da3_output.npz` from `scripts/run_da3.py` (Depth Anything 3 output). Whether known
  camera poses can be used instead of DA3's estimates, and what the npz holds, is UNVERIFIED.
  The `run_inference_weighted.py` flags include `stage1_entropy_alpha`,
  `stage2_weight_source=entropy`, `stage2_visibility_alpha`.
- Forks: `wyim-pgl/MV-SAM3D` (reports about 19.9 GiB VRAM for 3 views, 21.9 GiB for 8, with a
  `--low_vram` option, per search snippet, UNVERIFIED), `ushashi149/MV-SAM3D-AFM`,
  `devinli123/multi-view-sam-3d-objects` (earlier multidiffusion version),
  `Roworks-ai/sam-3d-objects-multi` (merges upstream PR #37 for multi-view).
- Fit for us (opinion, based on the above): training-free and takes 3 to 8 views with per-object
  masks, which matches our 3 to 5 keyframes per object and our SAM 3.1 tracks. Mismatches: it
  expects DA3 depth/pose output rather than our COLMAP poses (integration unverified), it is
  community code of unknown maturity and license, and it still has to be registered to our
  metric cloud. It would be a P3.3 bake-off candidate, not a dependency.

## 5. Implications for P2.2, P2.3, P2.4

Evidence-based:
- P2.4 as written (top 3 to 5 frames per object by mask area, occlusion, border contact,
  sharpness) matches how the model is called: one image and one mask per call, one call per
  object per seed. Nothing in the code gives a discount for objects sharing a frame, so a
  "frames that serve all objects" rule has no cost basis.
- With the official model, only one frame per object is used per call (P3.1 "draw several"
  means several frames and/or seeds, each a separate call). With a multi-view variant, the 3 to
  5 stored frames and their masks would be used together.
- The pointmap is per frame (dense HxWx3 in that image's camera frame). If we want to pass one,
  P2.4 should store, per keyframe, the camera pose and intrinsics so a dense map can be
  rasterised from the reconstruction, or use the model's default depth estimate. Sparse points
  alone are not enough.
- P2.2/P2.3 need no change from this research. Masks must be clean binary masks per frame
  (P2.2 tracks supply them); P2.3's lifted 3D points are useful for registration (P3.2), not as
  SAM 3D input.

Opinion:
- Prefer per-object best frames; sharing a frame across objects is a tiebreaker, because the one
  shared cost is a pointmap or image load we can cache anyway.
- Pass a pointmap only after a P3.3 A/B test: the paper reports shape/texture largely
  independent of it, so the gain would be pose and scale, which we refine against the metric
  cloud anyway.

## Open items

- Which depth model the shipped `pipeline.yaml` uses; whether pointmap must be metric.
- Per-call runtime and memory (paper appendix and repo, not found).
- `make_scene` internals.
- MV-SAM3D license, last commit date, pose input.

## Sources (all read 2026-10-28)

- https://github.com/facebookresearch/sam-3d-objects (README)
- https://raw.githubusercontent.com/facebookresearch/sam-3d-objects/main/notebook/inference.py
- https://raw.githubusercontent.com/facebookresearch/sam-3d-objects/main/sam3d_objects/pipeline/inference_pipeline_pointmap.py
- https://raw.githubusercontent.com/facebookresearch/sam-3d-objects/main/notebook/demo_multi_object.ipynb
- https://raw.githubusercontent.com/facebookresearch/sam-3d-objects/main/doc/setup.md
- https://github.com/facebookresearch/sam-3d-objects/blob/main/LICENSE
- https://huggingface.co/facebook/sam-3d-objects
- https://arxiv.org/abs/2511.16624 and https://arxiv.org/html/2511.16624 (first 100000 characters)
- https://github.com/devinli123/MV-SAM3D
- https://arxiv.org/abs/2603.11633
- Search results only (not opened): https://github.com/wyim-pgl/MV-SAM3D,
  https://github.com/Roworks-ai/sam-3d-objects-multi,
  https://github.com/devinli123/multi-view-sam-3d-objects
- Not read: Meta's SAM 3D web page and model card (only the HF card was read).
