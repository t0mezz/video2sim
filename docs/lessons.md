# Lessons

What the work taught us, with the evidence. Each lesson names where the numbers come from. The
decisions that follow from them are in [decisions.md](decisions.md); what is still open is in
[../TODO.md](../TODO.md).

## What we learned

1. **Mass is a density problem, not a geometry problem.** SAM 3D meshes are thin shells: signed mesh
   volume is a median 0.13 of the true volume (0 of 39 within 20%), while a voxel fill after sealing
   gaps up to 4% of the longest side reads 1.01 (32 of 39). With the volume fixed, table density times
   volume still misses the weighed mass by factors of 2 to 40, because the objects are mostly air:
   boxes, infilled prints and sheet-metal cans. Uncorrected mass is within 30% for 4 of 56 dev
   objects; a fill fraction per class and answer raises it to 36% held out, and combining geometry with
   a VLM mass range reaches 52% on dev. On test it was 45% at the gate and 29% in the final run, from
   the same code: the VLM answers and SAM 3D draws are redrawn each run, so the method sits between
   about 29% and 45% per draw. Material class and fill fraction are not visible in video, and no cheap
   fix remains on these inputs. [research/mesh_volume.md](research/mesh_volume.md), D-024, D-026.

2. **A generative model supplies shape and orientation, never scale.** SAM 3D's translation and scale
   are in its own depth units. Scale comes from the printed tag edge (within 0.5% of the board), and
   registration fits the mesh to the metric point cloud and the masks. What survives is the model's
   proportions, because registration scales uniformly: a can came out 12% too wide for its height.
   That is the remaining size error (median 4.5% on dev, 5.7% in the final run), and mass goes with
   its cube.

3. **Cleaning beat adding data for object boxes.** The sparse cloud gave 18.7% median box error on
   the easy tier. Stray short SfM tracks set the box, so trimming the two outermost points per side
   gave 8.9%, and fitting a box, cylinder or sphere to the masks gave 4.3%. The visual hull stayed at
   36% even with perfect silhouettes (0.074 m for a 0.055 m cube), because the short handheld arcs
   cover only 45 to 130 degrees of azimuth. [research/cluster_box_accuracy.md](research/cluster_box_accuracy.md).

4. **Doing less held up better than expected, until it did not.** A generic SAM prompt found 96.8% of
   objects against 98.4% for VLM nouns plus SAM at a tenth of the calls. Sequential matching made
   reconstruction 3.5x faster with no measurable loss. Two SAM 3D draws per object cost about 0.7
   points of size error against six. But 2 fps or 960 px images lost the tag-distance gate. Change one
   lever at a time and measure it against the gate that matters.
   [history/reconstruction_experiments.md](history/reconstruction_experiments.md),
   [research/sample_counts.md](research/sample_counts.md).

5. **A bake-off settles what priors do not, and the fit score cannot rank draws.** TRELLIS.2 looked
   like a peer to SAM 3D; on 42 dev objects it read 8.7% size error against 4.5% and fell back to the
   proxy hull for 20 objects against 1 (D-018). Dropping it also removed research-only licenses from
   the pipeline. The registration fit score picks the best of six draws for only 17 of 42 objects
   (rank correlation -0.38 with size error), and perfect selection would gain at most 1.4 points, so
   proportions are the larger error. The score also cannot see colour.

6. **Round things roll, and MuJoCo is not PhysX.** The basketball rolls 2.8 cm because nothing resists
   rolling and no sourced value exists; the chocolate egg stands only through a dent made by hand.
   The settling gate therefore excludes objects within 0.04 of an ellipsoid (D-025). Every stability
   number but one comes from MuJoCo's contact solver (D-027); the one PhysX run, on a single scene,
   settled with 0.44 mm drift.

7. **No published numbers exist for tabletop friction and restitution.** The handbook friction pairs
   are metal, wood and polymer on each other, none object on tabletop, and restitution values exist only
   for spheres on steel plates at 2 to 3 m/s. So friction is unsourced for 13 of 14 material classes
   and restitution for 12, and both are carried as wide ranges (D-019). Restitution is never simulated.
   [research/material_sources.md](research/material_sources.md).

8. **Hosted models are noisy; replay from cache whenever you can.** At temperature 0 the same crops
   of the s17 mirror cube were answered matte, glossy, metallic or mirror across reruns and prompt
   variants, so a single-run flag set is noisy. Shape fit scores differ by up to 0.017 for the same
   seed. Most decisions here (draw count, box fixes, view coverage, the lift settings) were made by
   replaying cached answers with no hosted call, which also makes them repeatable.

9. **Hosting costs time in unglamorous places.** Six simultaneous cold SAM 3D calls started only two
   containers, and a cold call takes about 110 s, so two containers are warmed while segmentation runs
   (shape on s02: 151 s cold, 62 s warm). Pipelining shape and physics cut s02 from 315 s to 202 s.
   CoACD is already threaded, so three at once is the ceiling, and it aborts the process if its logger
   is first created in a thread. Three full videos in parallel crashed the Mac after 6.5 minutes.
   [research/sam3d_cold_start.md](research/sam3d_cold_start.md),
   [research/parallel_timing.md](research/parallel_timing.md).

10. **Capture sets the ceiling.** Blur and focus errors cannot be fixed afterwards, and an object
    hidden from every frame cannot be recovered downstream. The known discovery misses looked like
    SAM prompt misses, not occlusion: three evenly spaced frames already show every ground-truth object
    in at least one view on all 12 scored dev scenes, and a later fresh run of the three scenes with
    known misses found all of them. Treat single-run recall differences with the same caution as
    lesson 8. The s16 marker is warped, which gave a 2.53% marker miss that is an input defect, not
    something to tune away.

11. **Ground truth needs a human check before it judges a model.** The s01 earbud box stands at yaw
    90 degrees, not 0 (it scored 89.6 degrees until corrected); the VLM called the coffee can steel when
    it is cardboard with a metal bottom, about 80% filled; four cardboard packages were marked hollow by
    hand. Fix the data, never the target: no ground truth or target is edited to make a number pass.

12. **Keep the test split for gates.** Thresholds were tuned on dev, some on as few as 19 objects (the
    trim passes by 1 object, the fit by 5). The test split ran once per gate. Because it was then seen,
    the final benchmark is not a clean first look and a future mass method is built on dev only (D-026).

13. **Ship new behavior off, and pin the off path.** The discovery features of backlog item 9 leave the
    stage output byte-identical when off, and a test pins it. The end-of-segmentation coverage check
    follows the same rule, and all of them are off in the committed config
    ([../TODO.md](../TODO.md), item 9).

## Risks and how they played out

The risk register of the plan, with what happened. The trigger is the point at which the plan said to
change course.

| Risk | Trigger | What happened |
| --- | --- | --- |
| SAM 3D pose or scale unreliable | Median orientation error above 20 degrees on easy scenes | Not reached: 1.0 degrees on dev, 0.71 on test after refinement |
| Reconstruction fails on textureless tables or fast motion | More than 20% of videos fail to register | Not reached: 100% of frames registered on every accepted dev and test video. The gate rejected only deliberately degraded footage, and s22 and s23 `video_b` for blur |
| Model access or license blocks the intended use | Any license that forbids it, by week 2 | Resolved as research use (D-005). TRELLIS.2's renderers allowed research only, so TRELLIS.2 was dropped (D-018). Retention terms are still unread |
| Generated geometry differs from the real object | More than 25% of objects fall back to proxies | Not reached: 1 of 43 on dev, 5 of 32 (16%) on test |
| Mass and friction guesses are poor | Mass target missed by more than 15 points after calibration | Reached: 29% against 50% in the final run, 21 points short. 45% at the Phase 4 gate was 5 short and was waived (D-026). Friction is placeholders (D-019) |
| Bad meshes break the simulator | Validator failures above 10% after repair | Not reached: 33 of 33 scored and 61 of 61 dev meshes pass |
| Tracking identity errors | More than 2 switches per scene on average | Not reached: 0.0, though the benchmark does not exercise the repair step |
| Large models exceed GPU memory or runtime | Runtime above 60 minutes per scene | Not reached: 14.9 minutes at worst, with all heavy models hosted |
| Scope creep toward articulation or rooms | A task that adds more than one week | Held: out-of-scope asks went to the backlog. One unplanned feature, the coverage check, was built outside the plan |
