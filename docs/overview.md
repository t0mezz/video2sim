# Overview

Video2Sim turns one smartphone video of a tabletop into a metric-scale USD scene, with one URDF per
object, that a physics simulator loads and that stays stable under gravity. This page gives the
scope, the pipeline and the results. How each stage works is in [stages.md](stages.md), what went
wrong and what it taught us is in [lessons.md](lessons.md), and every decision is in
[decisions.md](decisions.md).

Status on 2026-11-03: Phases 0 to 4 are done (Phase 4 passed by human decision with the mass miss
waived, D-026). Phase 5 is built and its gate is still open. Open work is in [../TODO.md](../TODO.md).

## Scope

**In scope**

* Input: one handheld video of 5 to 15 rigid objects on a table, 10 to 60 s, with a printed scale
  marker in view. How to film it: [capture_protocol.md](capture_protocol.md).
* Output per object: a closed mesh, convex collision geometry, mass, inertia, and friction and
  restitution ranges. The table is a plane in a gravity-aligned world frame.
* Formats: USD for the scene, URDF per object. Settling and the pick-and-place demo run in MuJoCo
  (D-027). One scene has also been settled in Isaac Sim under PhysX (`TODO.md` item 8); beyond
  that, Isaac Sim loading and PhysX behavior are unverified.
* A QA report per scene that flags every object it could not reconstruct well.

**Out of scope**, deferred on purpose (the backlog is in [../TODO.md](../TODO.md))

* Articulated objects (treated as rigid and flagged), deformables and liquids, transparent,
  mirror-like and very thin objects (flagged, replaced by a proxy or left out).
* Dynamic scenes: objects must not move during capture, and hands and arms are masked out.
* Rooms and large environments: tabletop plus a floor plane only.
* Measured friction: friction and restitution come from material classes, as ranges.
* Training policies and any product surface: command line and config files only.

## Pipeline

```text
video -> capture -> reconstruction -> segmentation -> shape -> physics -> export -> scene package
```

| # | Stage | Main modules in `src/video2sim/stages/` | Output | Hosted models |
| --- | --- | --- | --- | --- |
| 1 | Capture | `capture.py` | `CaptureResult` | none |
| 2 | Reconstruction | `reconstruction.py`, `world_frame.py`, `marker_detection.py` | `WorldModel` | none (pycolmap on CPU) |
| 3 | Segmentation | `segmentation_stage.py`, `scene_router.py`, `tracking.py`, `lift.py`, `primitive_fit.py`, `material_pass.py` | `Instances` | VLM, SAM 3.1 |
| 4 | Shape | `shape.py`, `registration.py` | `RegisteredObjects` | SAM 3D on Modal |
| 5 | Physics | `physics_stage.py`, `mesh_repair.py`, `collision.py`, `mass.py`, `settling.py` | `SimAssets` | VLM (material, mass) |
| 6 | Export | `export.py`, `usd_writer.py`, `urdf_writer.py`, `qa_report.py` | `SceneExport` | none |

Each stage takes a typed pydantic input plus files and returns a typed output plus files. The types
live in `src/video2sim/schemas.py`, and nothing else crosses a stage boundary. A run writes
`runs/<run_id>/` with a manifest that holds the config hash, and a stage is skipped when the hash of
its own config section, its source file and its dependencies' outputs is unchanged. Commands: `video2sim run`, `video2sim eval`, `video2sim doctor`.

Compute (D-003, D-004, D-015): nothing runs on a local GPU. SAM 3.1 goes through the Meta Model API,
the VLM through Baseten (`meta-models/muse-spark-open-1026`, D-008), and SAM 3D Objects on Modal
GPUs. Reconstruction, collision decomposition and settling run on the CPU of the machine that
calls `video2sim run`. Endpoints are configurable: [model_interfaces.md](model_interfaces.md).

## Targets and results

The numbers that gate a decision live in `configs/targets.yaml`, final since 2026-11-03; if this
table and that file disagree, the file wins. Results are the final benchmark on the test split, one
`video2sim run` per scene with no manual steps, code 21d34de, 2026-10-31
([final_benchmark.md](final_benchmark.md)).

| Metric | Target | Result | Met |
| --- | --- | --- | --- |
| Object recall | 90% or more | 93.9% (31 of 33) | Yes |
| Object size error, median | 5% or less | 5.7% | No |
| Position error, median | 1.5 cm or less | 4.3 mm | Yes |
| Orientation error, median | 10 degrees or less | 1.5 degrees | Yes |
| Mass within 30% | 50% of objects or more (D-025) | 29% (9 of 31) | No |
| Simulation stability | 95% of scenes settle, drift 2 cm in 5 s | 7 of 7 scored, 8 of 8 run | Yes |
| Asset validity | 100% | 33 of 33 scored, 37 of 37 run | Yes |
| Runtime per scene | 60 minutes or less | 14.9 min, worst scene | Yes |
| Manual steps | none | 0 on 8 of 8 runs | Yes |

Mass error compounds from size error, which is why the size target is tight:

$$\frac{\delta m}{m} \approx 3\,\frac{\delta s}{s} + \frac{\delta \rho}{\rho}$$

A 5% scale error alone costs about 15% of mass, leaving the rest of the 30% budget for the density
guess. In practice the density term dominates: [lessons.md](lessons.md), lesson 1.

Read the results with these limits. The test scenes were seen in Phases 1 to 4, so this is not a
clean first look (D-026). The size and mass targets are provisional, and mass was lowered from 70%
(D-025). Friction and restitution are mostly unsourced placeholders (D-019). Pick-and-place is not a
target, and its control is open loop.

## How it is evaluated

* **Benchmark.** 24 tabletop scenes, 103 objects, two camera paths each (s06, s14 and s24 have only
  `video_b`), in three tiers: 10 easy, 6 medium, 8 hard (D-013). Videos and meshes are not in git
  (D-006). Layout and ground-truth format: [../benchmark/README.md](../benchmark/README.md).
* **Split.** 15 dev scenes and 9 test scenes, stratified by tier (D-013). Thresholds are tuned on dev
  only, and the test split is run once per phase gate.
* **Deformables.** Scenes s21 to s23 hold deformable objects and are left out of every metric
  (D-014). They stay in the benchmark for testing after the MVP.
* **Clean boards.** Four scenes (s05, s14, s24 on dev, s10 on test) use a board with production tags
  and no printed grid, to check that the grid texture does not flatter registration. s05 and s10
  have no established ground-truth origin, so both are unscored.
* **Conventions (D-012).** Predictions are matched to ground truth by the Hungarian method on box
  center distance, with a gate of max(5 cm, half the smaller horizontal side). The predicted box is
  the minimum-area rectangle of the XY footprint plus the Z extent. Size error is the largest
  relative error over the three axes, position error the 3D center distance, orientation error the
  yaw modulo 180 degrees (skipped for symmetric footprints). Medians cover matched objects only;
  misses count in recall.
* **Command.** `uv run video2sim eval --baseline oracle` scores a baseline, `--runs runs` scores
  pipeline runs, `--split test` is for phase gates. It writes `report.json` and `report.html`.

## How the project went

| Phase | Gate | Outcome |
| --- | --- | --- |
| 0 Foundations | 2026-10-26 | Benchmark, ground truth, evaluation harness, CI, license decisions. Container image deferred. |
| 1 Reconstruction | 2026-10-27 | pycolmap on CPU, scale from the printed tag edge. 100% of frames registered on dev and test. |
| 2 Segmentation | 2026-10-29 | Recall 94.9% on dev, 92.9% on test. Cluster boxes 4.3% (dev) and 6.0% (test) on the easy tier. |
| 3 Shape | 2026-10-30 | SAM 3D Objects only after a bake-off (D-018). Size 5.4%, position 4.4 mm on test. |
| 4 Physics | 2026-10-31 | Passed by human decision with the mass miss waived (D-026): 45% against 50% on test. |
| 5 Export | open | Built and measured in MuJoCo. Two exit criteria open: a new person reproduces a scene, USD read-back. |

The full task and evidence records are in [history/](history/).
