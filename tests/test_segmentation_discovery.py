from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from video2sim.evaluation.scenes import GtObject
from video2sim.evaluation.segmentation import mask_recall, project_box_mask
from video2sim.schemas import SE3, CameraFrame, Intrinsics, Plane, WorldModel
from video2sim.stages.segmentation import (
    DiscoveredMask,
    DiscoveryConfig,
    FakeDiscoveryBackend,
    FrameDiscovery,
    discover_objects,
    sample_frame_indices,
)


def _cfg(**override: object) -> dict[str, object]:
    d = {
        "method": "vlm_sam",
        "n_sample_frames": 3,
        "max_workers": 2,
        "generic_prompt": "things",
        **override,
    }
    return {"segmentation": {"discovery": d}}


CFG = _cfg()


def _pose() -> SE3:
    # Camera 0.5 m above the board looking straight down.
    return SE3.from_numpy(
        np.array([[1.0, 0, 0, 0.1], [0, -1.0, 0, 0], [0, 0, -1.0, 0.5], [0, 0, 0, 1.0]])
    )


def _world(n_frames: int) -> WorldModel:
    k = Intrinsics(fx=800, fy=800, cx=320, cy=240, width_px=640, height_px=480)
    frames = [
        CameraFrame(
            frame_index=10 * i, image_path=f"frames/{i:04d}.jpg", T_world_from_camera=_pose()
        )
        for i in range(n_frames)
    ]
    return WorldModel(
        intrinsics=k,
        frames=frames,
        scale_factor=1.0,
        scale_source="marker",
        table_plane=Plane(normal=(0.0, 0.0, 1.0), offset_m=0.0),
        point_cloud_path="reconstruction/points.ply",
        mean_reprojection_error_px=0.5,
        registered_fraction=1.0,
    )


def test_sampling_spreads_evenly_and_keeps_the_ends() -> None:
    assert sample_frame_indices(10, 4) == [0, 3, 6, 9]
    assert sample_frame_indices(3, 6) == [0, 1, 2]
    assert sample_frame_indices(9, 1) == [4]
    assert sample_frame_indices(0, 3) == []


def test_config_rejects_unknown_method_and_zero_frames() -> None:
    assert DiscoveryConfig.from_config(CFG).n_sample_frames == 3
    with pytest.raises(ValueError, match="unknown"):
        DiscoveryConfig.from_config(_cfg(method="x"))
    with pytest.raises(ValueError, match="at least 1"):
        DiscoveryConfig.from_config(_cfg(n_sample_frames=0))
    with pytest.raises(ValueError, match="max_workers"):
        DiscoveryConfig.from_config(_cfg(max_workers=0))
    with pytest.raises(ValueError, match="generic_prompt"):
        DiscoveryConfig.from_config(_cfg(generic_prompt="  "))
    assert DiscoveryConfig.from_config(_cfg(method="sam_generic")).method == "sam_generic"


def test_discovery_samples_frames_and_feeds_mask_recall(tmp_path: Path) -> None:
    world = _world(7)
    k = world.intrinsics
    objs = (
        GtObject("a", "box", (0.06, 0.06, 0.06), (0.0, 0.0, 0.03), 0.0, 0.2),
        GtObject("b", "box", (0.06, 0.06, 0.06), (0.2, 0.0, 0.03), 0.0, 0.2),
    )
    gt = {o.object_id: project_box_mask(o, k, _pose()) for o in objs}
    recorded = {
        0: FrameDiscovery(0, (DiscoveredMask("box", gt["a"]),)),
        60: FrameDiscovery(60, (DiscoveredMask(None, gt["b"]),)),
    }
    results = discover_objects(
        world, tmp_path, FakeDiscoveryBackend(recorded), DiscoveryConfig.from_config(CFG)
    )
    assert [r.frame_index for r in results] == [0, 30, 60]
    frames = [
        (world.frames[i].T_world_from_camera, [m.mask for m in r.masks])
        for i, r in zip((0, 3, 6), results, strict=True)
    ]
    assert mask_recall(objs, k, frames).recall == 1.0


def test_backend_that_drops_frames_is_rejected(tmp_path: Path) -> None:
    class Dropping:
        def discover(self, frames: object) -> list[FrameDiscovery]:
            return []

    with pytest.raises(ValueError, match="one result per frame"):
        discover_objects(_world(4), tmp_path, Dropping(), DiscoveryConfig.from_config(CFG))
