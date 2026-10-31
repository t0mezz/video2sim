"""The full pipeline as one ``Runner``: capture through export."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from video2sim.runner import Runner
from video2sim.stages.capture import CaptureStage
from video2sim.stages.export import ExportStage, export_stage_config
from video2sim.stages.physics_stage import PhysicsStage, physics_stage_config
from video2sim.stages.reconstruction import ReconstructionStage, reconstruction_stage_config
from video2sim.stages.segmentation_stage import SegmentationStage
from video2sim.stages.shape import ShapeStage


def build_runner(
    runs_root: Path, cfg: dict[str, Any], targets: dict[str, Any], base_dir: Path
) -> Runner:
    """``cfg`` is ``configs/default.yaml`` and ``targets`` is ``configs/targets.yaml``."""
    return Runner(
        runs_root,
        [
            CaptureStage(),
            ReconstructionStage(),
            SegmentationStage(),
            ShapeStage(),
            PhysicsStage(base_dir / cfg["physics"]["material_table"]),
            ExportStage(),
        ],
        {
            "capture": cfg["capture"],
            "reconstruction": reconstruction_stage_config(cfg, targets),
            "segmentation": cfg["segmentation"],
            "shape": cfg["shape"],
            "physics": physics_stage_config(cfg, targets),
            "export": export_stage_config(cfg, targets),
        },
        base_dir,
    )


__all__ = ["build_runner"]
