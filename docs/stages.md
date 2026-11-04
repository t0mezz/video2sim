# Stages

How each stage works, with the numbers that justify the design. Settings live in
`configs/default.yaml`; decision ids point into [decisions.md](decisions.md). Evidence files are in
`benchmark/results/`. "Dev" and "test" are the benchmark splits ([overview.md](overview.md)).

## 1. Capture (`stages/capture.py`)

Decides whether the video is usable and picks the frames. The gate runs before reconstruction and
reports every failed check at once, one message each: blur (sharpest frame of each 1/3 s window, `capture.target_fps: 3`, D-035),
exposure, marker visibility, frame count. How to film: [capture_protocol.md](capture_protocol.md).

* 5 of 5 deliberately degraded copies of s01 `video_a` are rejected with the matching message, and
  31 of 36 dev videos pass (`p1_4_capture_gate.json`).
* The thresholds are guesses tuned on dev. The blur limit is stricter than reconstruction needs: s12
  `alt_path` is rejected for blur but reconstructs with 1.0% scale error.
* There is no duration limit (the 10 to 60 s rule would reject videos that reconstruct fine) and no
  rolling-shutter or autofocus-hunting check.

## 2. Reconstruction (`stages/reconstruction.py`, `world_frame.py`)

Camera poses, a sparse point cloud, metric scale and a gravity-aligned world frame. Everything later
depends on it.

* **Back end.** pycolmap on CPU (D-016), at 3 fps on whole frames (1920 px wide for the benchmark
  videos; `max_image_size: -1` keeps pycolmap's 3200 px cap, D-038), with sequential matching and
  the incremental mapper. Exhaustive matching was 59 to 80% of the time on every video over 28
  frames. Sequential matching gives 3.5x on s16 `video_a` (709 s to 205 s) and 1.3x to 4.2x on five
  videos, with quality inside the limits. Fewer frames (2 fps) or a smaller image (960 px) lost the
  tag-distance gate, and the global mapper was left off. Evidence:
  [history/reconstruction_experiments.md](history/reconstruction_experiments.md).
* **Tag detections** of every registered frame, made on the frame PNG read back as grey, are written
  to `reconstruction/tags.json`. Segmentation's marker filter and board check read them instead of
  detecting again (D-036, D-039). Capture keeps its own detection: its grey comes from the decoded
  video frame and differs by one level in about 45% of pixels, which changed whether a tag was seen
  in 10 of 2,702 local frames.
* **Metric scale** comes from the printed tag edge, 100.26 mm (five caliper readings), so one or two
  tags anywhere in view are enough. The known grid spacing is only a check on grid scenes. Scale is
  within 0.5% of the board on dev. Model-predicted scale is never used as metric scale.
* **World frame.** A RANSAC table plane gives the up axis, and the origin sits on the reference tag
  seen in most frames. The tags-on-table layer (D-020) refits the plane through the tags when it
  lies more than 15 mm from a tag corner. It exists because on s05 RANSAC settled on the object tops,
  3 to 7 cm above a bare white table. A refit that cannot hold all tags is dropped and raises
  `tags_off_table` (s24).
* **Gates** (`configs/targets.yaml`): 95% of frames registered, reprojection 1 px, marker distance 2%,
  up axis 2 degrees. Dev: 100% registered, 0.77 px at worst, 1.56% marker distance, up axis 0.25
  degrees. Test, 16 grid videos: 100%, 0.72 px, 15 of 16 within 2% (s16 `video_b` reads 2.53%
  because its marker is warped; accepted by human decision), up axis 1.28 degrees.
* **Limits.** The sparse cloud has 10k to 26k points and is weak on glossy surfaces. Stray points
  lie below the table. Clean-board reconstruction is not evaluated on the test split.
* **VGGT** is about 20x faster on a GPU (37 s against about 12 min on s16 `video_a`) but reads twice
  the tag-distance error and its checkpoint is non-commercial. It is a post-MVP opt-in
  ([../TODO.md](../TODO.md), item 6).

## 3. Segmentation (`stages/segmentation_stage.py` and helpers)

Turns the video into one instance per physical object: masks per frame, a clean point cluster in the
world frame, keyframes, flags and a material answer.

1. **Router** (`scene_router.py`). One VLM request on 3 frames picks the SAM prompt and lists the
   objects. Fewer tracks than listed objects raises a missing-object suspicion for the QA report.
2. **Tracking** (`tracking.py`). One SAM 3.1 video request per scene with the generic concept
   "objects on a table", both directions, the registered frames sent as a 5 fps clip. Video is billed
   per frame at $0.20 per 1,000 with at least 50 per request: 13 dev scenes cost about $0.23.
3. **Identity repair in 3D** (`track_repair.py`). Tracks with no shared frame whose table footprints
   lie within 5 cm merge, and a track whose footprints fall into groups 10 cm apart splits. A sliver
   guard stops an almost fully occluded mask from seeding a new object, and a post-lift pass drops
   duplicates within 2 cm.
4. **Lift to 3D** (`lift.py`, `primitive_fit.py`). Sparse SfM points vote: a point belongs to a
   track when its mask holds it in 90% of the views that see it (40 views), and views where another
   track's mask excuses the occlusion do not count against it. The two outermost points per side of
   the resting box are trimmed, then an upright box, cylinder or sphere is fitted to the masks. That
   fitted cluster is `point_cluster_path`. The visual hull is kept only for the shape flags. Its
   vote first bounds cells of 4 x 4 x 4 voxels (projected windows with the lens model bounded by
   interval arithmetic) and votes only cells where some track can reach its threshold, which gives
   the members of the full vote (N12).
5. **Keyframes** (`keyframes.py`). Per object, up to 5 frames scored by mask area, border contact,
   overlap with other tracks and sharpness, with a penalty for frames from similar camera poses.
6. **Flags and material pass** (`shape_flags.py`, `material_pass.py`, D-017). Geometry flags `small`,
   `thin` and `occluded`. One VLM request per object on its best 2 crops returns category, material
   class, finish, hollow or solid, and confidence, and it raises `transparent_or_reflective` and
   `non_rigid`. The flag means reflectiveness is likely to break a later stage, not that the object
   is shiny.

Masks are held as `CropMask` (`segmentation.py`): the box that holds the mask pixels, placed in the
frame, instead of a 2 MB full frame at 1080p. SAM answers are decoded straight into crops, and every
step that reads a mask (lookups, windows, overlaps, downscaling for the vote and the primitive fit)
gives what the full frame gives, bit for bit. Only the PNG writer and the overlays expand a mask.
Decoded SAM rasters are stored per answer in `segmentation.work/tracking_work/raster_cache/`, so a
rerun skips the pure-Python decoder.

Numbers:

| Cluster box error, dev easy tier (19 objects) | Median |
| --- | --- |
| Visual hull | 36.4% |
| Sparse SfM points | 18.7% |
| Points with the 2 outermost per side trimmed | 8.9% |
| Fitted primitive (in use) | 4.3% |

Recall is 94.9% on dev (56 of 59) and 92.9% on test; identity switches are 0 in all 13 scored scenes,
though the benchmark never exercises the repair. The cluster box on the test easy tier is 6.0%.
Discovery by the generic prompt reaches 96.8% recall against 98.4% for VLM nouns plus SAM, but the
latter needs about 10 times the hosted calls and gives 5.6 times the false positives, so the generic
prompt stays. Detail: [research/cluster_box_accuracy.md](research/cluster_box_accuracy.md),
[research/lift_tracking_recall.md](research/lift_tracking_recall.md),
[history/phases_0_2.md](history/phases_0_2.md).

Known misses: white-on-white parts the masks miss, objects the generic concept never fires on (s13
stack member, s18 pliers, s24 glossy cube on earlier runs), and thin objects, because the hull reads
flat objects 3 to 4 times too thick. Optional discovery add-ons (multi-view router, gap fill) and an
end-of-segmentation coverage check are built and off by default; their status is in
[../TODO.md](../TODO.md), item 9.
Design notes: [research/occlusion_robust_discovery_plan.md](research/occlusion_robust_discovery_plan.md).

## 4. Shape (`stages/shape.py`, `registration.py`)

Gives each object a closed mesh in the metric world frame.

* **Model.** SAM 3D Objects only (D-018), served by a Modal worker on A100-40GB with at most 4
  containers, all 4 warmed when segmentation starts (D-034, raising D-032's 2; the shape wall time is
  about draws x 13 to 14 s / containers, and 4 containers are not yet measured). In the probes a call took 110 s cold
  and 16 to 18 s warm on 80 GB, and 92 s and 13 to 14 s on the 40 GB card in use
  ([research/sam3d_cold_start.md](research/sam3d_cold_start.md)). Contract and conventions: [model_interfaces.md](model_interfaces.md),
  [research/sam3d.md](research/sam3d.md).
* **Draws.** The model runs on an object's best keyframes, one seed each: the default is one draw on
  each of up to 5 keyframes (D-033, which dropped D-030's 2 draws; a dev replay put 2 draws at 5.7%
  size error against 5.0% with 6). The kept draw is the one with the best fit score.
  `shape.early_stop` (off by default, D-040) draws `min_draws` per object first and skips the rest
  when each of those fits at or above `min_fit_score` and their resting boxes agree within
  `size_tolerance` per side. The report records draws made, draws skipped and the reason. Not yet
  evaluated on dev.
* **Masks** are read from the segmentation PNGs as crops (N5); only the registration views and the
  keyframes sent to SAM 3D are expanded to full frames, so the PNG bytes SAM 3D receives and its
  cache keys are unchanged.
* **Registration.** SAM 3D's rotation is the start pose, with four yaw starts. Its translation and
  scale are in its own depth units, so scale and center start from the observed cluster. A
  Levenberg-Marquardt fit in NumPy minimises, over scale `s`, rotation `R` and translation `t`:

$$\min_{s,R,t}\; \frac{1}{|Y|}\sum_{y\in Y}\min_{x\in\mathcal{M}}\lVert y-(sRx+t)\rVert^{2} + \lambda\sum_f \mathcal{L}_{\text{mask}}\big(\pi_f(s,R,t;\mathcal{M}),\,M_f\big)$$

  Here `Y` is the observed cluster, `M` the generated mesh, `pi_f` the projection into frame `f`
  and `M_f` the SAM mask in that frame. The first term is one-sided in effect, so the unseen back of an
  object is not penalised. The mask term is computed without rendering: mesh samples outside the mask
  (other objects' masks allowed, for occlusion) plus the mesh's reach along 8 image directions. A
  resting prior levels the model axis nearest to up and holds the lowest vertex on the table plane.
  A last per-axis stretch with a prior of width 0.5 refines the kept draw.
* **Fallback.** Below `min_fit_score` (0.5, a guess) the convex hull of the cluster stands in, flagged
  `PROXY_GEOMETRY` and `LOW_FIT`. Resting meshes sit on the table with 0.0 mm gap on 68 meshes.
* **Numbers.** Dev: size 4.5%, position 6.1 mm, orientation 1.0 degrees. Test, run once at the
  gate: 5.4%, 4.4 mm, 0.71 degrees. Proxy hulls: 1 of 43 on dev, 5 of 32 on test.
* **Bake-off (D-018).** On 42 dev objects matched by all arms, median size error was 4.5% for SAM 3D,
  8.7% for TRELLIS.2 and 8.4% for TRELLIS.2 geometry on SAM 3D's pose, with 1, 20 and 11 proxy hulls.
  TRELLIS.2 was dropped, which also removed the research-only NVIDIA renderers from the pipeline.
* **Limits.** Registration scales uniformly, so SAM 3D's proportions carry through (a can 12% too
  wide for its height). The fit score ranks draws poorly. Thin, small and occluded objects miss most:
  s14 breadboard 75%, s19 small wooden toy 114% on test. Meshes are thin shells, which matters for
  physics.

## 5. Physics (`stages/physics_stage.py`)

Makes each object simulation-ready, with mass, inertia and friction as low, nominal and high.

* **Mesh repair** (`mesh_repair.py`). Floaters removed, holes filled, watertight and manifold,
  decimated to 20,000 triangles with volume change under 2%. Validators block export.
* **Collision** (`collision.py`). A box, cylinder or sphere where the object is close to one,
  otherwise convex decomposition with CoACD under a cap on parts. The fit check compares the parts'
  surfaces with the visual surface and ignores surfaces inside the sealed body of the mass module
  (D-022). All 61 registered dev meshes pass.
* **Material** (`physics.py`, `material_table.py`). The VLM picks one of 14 material classes and
  whether the object is hollow; code maps the class to numbers in `configs/material_table.yaml`, where
  every value is sourced or listed as an unsourced placeholder (D-019, sources in
  [research/material_sources.md](research/material_sources.md)). The VLM writes no number except
  mass (D-024).
* **Mass and inertia** (`mass.py`, `vlm_mass.py`). The body is a voxel fill of the mesh after sealing
  gaps up to 4% of the longest side, because the signed mesh volume is a median 0.13 of the true
  volume. Closed hollow objects are scaled by a fill fraction `f` fitted per class and answer:

$$m = \rho\,V_{\text{eff}}, \qquad \mathbf{I} = \rho \int_V \left(\lVert\mathbf{r}\rVert^2 \mathbf{1} - \mathbf{r}\mathbf{r}^{\top}\right) dV$$

  One VLM request per object returns a low, nominal and high mass. The geometry mass stands when it
  falls inside that range, otherwise the model's range is used (D-024). On 56 weighed dev objects this
  holds 52% within 30%, against 36% for geometry alone. Detail:
  [research/mesh_volume.md](research/mesh_volume.md).
* **Settling** (`settling.py`, D-021). The scene is dropped under gravity in MuJoCo and each object's
  drift from its observed pose must stay under 2 cm over 5 s. Round objects are not counted (D-025):
  the basketball and the egg really roll and fall over. Result: 13 of 13 dev scenes and 8 of 8 test
  scenes settle, the test scenes in all five friction and mass cases.
* **Output.** `SimAssets` puts the origin at the center of mass (`T_world_from_object`, D-023).
  Inertia is one matrix at nominal mass, scaled with mass for low and high. An object that fails
  settling is reported and not blocked.

Mass within 30%: 52% on dev (in sample), 45% on test at the Phase 4 gate, 29% on the final run. The
cause is density, not geometry: [lessons.md](lessons.md), lesson 1.

## 6. Export (`stages/export.py`)

* **USD** (`usd_writer.py`). Z up, meters and kilograms set explicitly. Per object a transform with a
  rigid body and mass properties, a visual mesh with `displayColor` from its vertex colors (no
  textures, D-028), convex collision meshes and physics materials. One infinite plane is both table
  and ground. Read back against `SimAssets` on 61 objects of the 13 rigid dev scenes.
* **URDF** (`urdf_writer.py`). One per object, with inertia about the center of mass. Mass,
  principal inertia and collision bounds match the physics stage on all 61 objects.
* **QA report** (`qa_report.py`, `scene_render.py`). `qa_report.html` shows status, confidence and
  fallbacks per object, the stability result, the mass source and range, and the real frame beside
  the simulated scene from the same camera. Any flag marks an object, so 11 of 13 dev scenes read
  `review` and 2 read `ok`.
* **Pick-and-place demo** (`pick_place.py`, D-029). A Panda from MuJoCo Menagerie lifts an object at
  least 10 cm and places it within 2 cm of a goal, across the stored friction and mass ranges. 7 of
  the 13 dev scenes have an object that fits the rules; all 7 pass the nominal case and 5 pass all
  five. The control is open loop, so this shows the twin is graspable in simulation only.
* **Isaac Sim.** One exported scene settles in Isaac Sim under PhysX in all five friction and mass
  cases, with 0.44 mm largest drift (`isaac_verify.py`, `benchmark/run_isaac_settling.py`). Other
  scenes, the comparison with MuJoCo, the friction combine mode and the 255-vertex collision limit
  are not verified ([../TODO.md](../TODO.md), item 8).
