from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from video2sim.schemas import SE3, CameraFrame, Intrinsics, Plane, WorldModel
from video2sim.stages.marker_filter import (
    MarkerFilterConfig,
    filter_marker_tracks,
    overlaps_tag,
    tag_mask,
)
from video2sim.stages.tracking import Track

SHAPE = (100, 100)
TAG = np.array([[10.0, 10.0], [40.0, 10.0], [40.0, 40.0], [10.0, 40.0]])


def _cfg(**over: Any) -> MarkerFilterConfig:
    base: dict[str, Any] = {
        "enabled": True,
        "family": "aruco_4x4_50",
        "erode_px": 2,
        "min_mask_share": 0.2,
        "min_tag_share": 0.2,
        "min_hit_frames": 2,
        "min_hit_fraction": 0.2,
    }
    return MarkerFilterConfig(**{**base, **over})


def _world(n: int) -> WorldModel:
    return WorldModel(
        intrinsics=Intrinsics(fx=100.0, fy=100.0, cx=50.0, cy=50.0, width_px=100, height_px=100),
        frames=[
            CameraFrame(
                frame_index=i, image_path=f"f{i}.png", T_world_from_camera=SE3.from_numpy(np.eye(4))
            )
            for i in range(n)
        ],
        scale_factor=1.0,
        scale_source="marker",
        table_plane=Plane(normal=(0.0, 0.0, 1.0), offset_m=0.0),
        point_cloud_path="p.ply",
        mean_reprojection_error_px=0.5,
        registered_fraction=1.0,
    )


def _box(r0: int, r1: int, c0: int, c1: int) -> np.ndarray[Any, Any]:
    m = np.zeros(SHAPE, dtype=bool)
    m[r0:r1, c0:c1] = True
    return m


def _detect(visible: set[str]) -> Any:
    def detect(path: Path) -> Any:
        return ({0: TAG} if path.name in visible else {}), SHAPE

    return detect


def test_tag_mask_is_shrunk_by_the_erosion() -> None:
    assert tag_mask([TAG], SHAPE, 0).sum() > tag_mask([TAG], SHAPE, 2).sum() > 0


def test_a_mask_on_the_tag_hits_and_one_beside_it_does_not() -> None:
    tags = tag_mask([TAG], SHAPE, 2)
    assert overlaps_tag(_box(12, 38, 12, 38), tags, _cfg())
    assert not overlaps_tag(_box(60, 80, 60, 80), tags, _cfg())
    assert not overlaps_tag(_box(10, 90, 39, 45), tags, _cfg(min_tag_share=0.9))  # edge bleed


def test_marker_track_is_dropped_and_object_kept(tmp_path: Path) -> None:
    frames = range(4)
    marker = Track(1, {i: _box(12, 38, 12, 38) for i in frames})
    thing = Track(2, {i: _box(60, 80, 60, 80) for i in frames})
    kept, report = filter_marker_tracks(
        [marker, thing], _world(4), tmp_path, _cfg(), _detect({f"f{i}.png" for i in frames})
    )
    assert [t.track_id for t in kept] == [2]
    assert set(report.dropped) == {1}


def test_one_bad_frame_does_not_kill_a_track(tmp_path: Path) -> None:
    masks = {i: _box(60, 80, 60, 80) for i in range(5)}
    masks[0] = _box(12, 38, 12, 38)
    kept, report = filter_marker_tracks(
        [Track(1, masks)], _world(5), tmp_path, _cfg(), _detect({f"f{i}.png" for i in range(5)})
    )
    assert len(kept) == 1 and not report.dropped


def test_frames_without_a_detected_tag_are_not_judged(tmp_path: Path) -> None:
    marker = Track(1, {i: _box(12, 38, 12, 38) for i in range(4)})
    kept, _ = filter_marker_tracks([marker], _world(4), tmp_path, _cfg(), _detect(set()))
    assert len(kept) == 1


def test_disabled_filter_keeps_everything(tmp_path: Path) -> None:
    marker = Track(1, {i: _box(12, 38, 12, 38) for i in range(4)})
    kept, report = filter_marker_tracks(
        [marker], _world(4), tmp_path, _cfg(enabled=False), _detect({f"f{i}.png" for i in range(4)})
    )
    assert len(kept) == 1 and not report.dropped


def test_default_config_loads() -> None:
    import yaml

    cfg = yaml.safe_load(Path("configs/default.yaml").read_text())
    assert MarkerFilterConfig.from_config(cfg).enabled
