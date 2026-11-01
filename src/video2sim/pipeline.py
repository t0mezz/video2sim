"""The full pipeline as one ``Runner``: capture through export."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from video2sim.models.sam3d import Sam3dConfig, make_transport, start_warm_up
from video2sim.runner import EventHandler, Runner
from video2sim.stages.capture import CaptureStage
from video2sim.stages.export import ExportStage, export_stage_config
from video2sim.stages.physics_stage import PhysicsStage, physics_stage_config
from video2sim.stages.reconstruction import ReconstructionStage, reconstruction_stage_config
from video2sim.stages.segmentation_stage import SegmentationStage
from video2sim.stages.shape import ShapeStage


def build_runner(
    runs_root: Path,
    cfg: dict[str, Any],
    targets: dict[str, Any],
    base_dir: Path,
    on_event: EventHandler | None = None,
) -> Runner:
    """``cfg`` is ``configs/default.yaml`` and ``targets`` is ``configs/targets.yaml``."""

    # Segmentation takes about as long as a cold start, and the 5 minute scale-down window
    # would close on containers started at the top of a long reconstruction.
    def warm_shape_model() -> None:
        if cfg["shape"]["backend"] != "sam3d":
            return
        sam3d = Sam3dConfig.from_config(cfg)
        start_warm_up(make_transport(cfg), sam3d.warm_containers, sam3d.warm_hold_s)

    return Runner(
        runs_root,
        [
            CaptureStage(),
            ReconstructionStage(),
            SegmentationStage(on_start=warm_shape_model),
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
            "vlm": cfg["vlm"],  # read by the stages that list it in shared_config
        },
        base_dir,
        on_event,
    )


__all__ = ["build_runner"]
