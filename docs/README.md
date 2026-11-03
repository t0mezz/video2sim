# Documentation

Where things are. For the project in one paragraph, start with the [root README](../README.md).

**Read in this order to understand the project:** [overview.md](overview.md), then
[capture_protocol.md](capture_protocol.md), [stages.md](stages.md), [final_benchmark.md](final_benchmark.md)
and [lessons.md](lessons.md).

**Read these before changing code:** [../AGENTS.md](../AGENTS.md) for the rules,
[../TODO.md](../TODO.md) for what is open, and [decisions.md](decisions.md) for why things are the way
they are.

## The project

| File | What it tells you |
| --- | --- |
| [overview.md](overview.md) | Scope, pipeline, targets against results, how it is evaluated, the phase timeline. |
| [stages.md](stages.md) | How each of the six stages works, with the numbers behind the design. |
| [lessons.md](lessons.md) | What the work taught us, and how each planned risk played out. |
| [decisions.md](decisions.md) | Every decision (D-001 to D-033) with its evidence. Ids are cited from code and configs. |
| [final_benchmark.md](final_benchmark.md) | End-to-end results on the test split, target by target. |

## Using it

| File | What it tells you |
| --- | --- |
| [capture_protocol.md](capture_protocol.md) | How to film a scene. |
| [model_interfaces.md](model_interfaces.md) | The three model roles, how to point them at your own endpoints, hosted against local quality. |
| [licenses.md](licenses.md) | License and terms status of every model and tool we depend on. |

## Evidence

| Where | What it holds |
| --- | --- |
| [research/](research/) | One note per experiment, each with the command and result file that produced it: [mesh_volume](research/mesh_volume.md), [cluster_box_accuracy](research/cluster_box_accuracy.md), [lift_tracking_recall](research/lift_tracking_recall.md), [material_sources](research/material_sources.md), [sample_counts](research/sample_counts.md), [parallel_timing](research/parallel_timing.md), [sam3d](research/sam3d.md), [sam3d_multiview](research/sam3d_multiview.md), [sam3d_cold_start](research/sam3d_cold_start.md), [occlusion_robust_discovery_plan](research/occlusion_robust_discovery_plan.md). |
| [history/](history/) | Full task and evidence records of the finished phases: [phases_0_2](history/phases_0_2.md), [phases_3_5](history/phases_3_5.md), [reconstruction_experiments](history/reconstruction_experiments.md), the old [plan_changelog](history/plan_changelog.md), and the final [PLAN.md](history/plan_2026-11-03.md), retired on 2026-11-03. Section numbers inside refer to that plan. |
| `sources/` | The source PDFs behind `configs/material_table.yaml` (git-ignored). |
| `../benchmark/results/` | The JSON files that every number in these docs cites. |

The benchmark's own layout and ground-truth format are in [../benchmark/README.md](../benchmark/README.md).
