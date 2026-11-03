from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from video2sim.evaluation.scenes import GtObject
from video2sim.evaluation.segmentation import project_box_mask, score_tracks
from video2sim.models.hosted import ResponseCache, SamRecord
from video2sim.schemas import SE3, CameraFrame, Intrinsics, Plane, WorldModel
from video2sim.stages.segmentation import BoolMask
from video2sim.stages.track_repair import RepairConfig, _groups, mask_footprint, repair_tracks
from video2sim.stages.tracking import (
    FakeVideoSamClient,
    Track,
    TrackingConfig,
    track_objects,
    tracks_from_text,
)

K = Intrinsics(fx=100, fy=100, cx=50, cy=50, width_px=100, height_px=100)
# Camera 1 m above the table looking straight down: pixel (u, v) is world ((u-50)/100, -(v-50)/100).
POSE = SE3.from_numpy(
    np.array([[1.0, 0, 0, 0], [0, -1.0, 0, 0], [0, 0, -1.0, 1.0], [0, 0, 0, 1.0]])
)
TABLE = Plane(normal=(0.0, 0.0, 1.0), offset_m=0.0)
CFG = RepairConfig(
    merge_radius_m=0.05,
    split_gap_m=0.10,
    min_frames=3,
    sliver_frac=0.2,
    merge_boundary_gap=2,
    merge_boundary_iou=0.5,
)


def _blob(u: int, v: int) -> BoolMask:
    mask = np.zeros((100, 100), dtype=bool)
    mask[v - 3 : v + 1, u - 3 : u + 4] = True
    return mask


def _track(track_id: int, frames: range, u: int, v: int) -> Track:
    return Track(track_id, {f: _blob(u, v) for f in frames})


def _repair(tracks: list[Track], n: int = 30):  # type: ignore[no-untyped-def]
    return repair_tracks(tracks, {f: POSE for f in range(n)}, K, TABLE, CFG)


def test_footprint_is_where_the_lowest_pixels_meet_the_table() -> None:
    point = mask_footprint(_blob(70, 60), POSE, K, TABLE)
    assert point is not None
    assert point == pytest.approx([0.2, -0.1], abs=1e-9)
    assert mask_footprint(np.zeros((100, 100), dtype=bool), POSE, K, TABLE) is None


def test_fragments_of_one_object_merge() -> None:
    tracks, report = _repair([_track(0, range(0, 10), 30, 60), _track(5, range(12, 20), 31, 60)])
    assert [t.track_id for t in tracks] == [0]
    assert sorted(tracks[0].masks) == list(range(0, 10)) + list(range(12, 20))
    assert report.merges == ((0, 5),)


def test_tracks_that_share_a_frame_are_never_merged() -> None:
    tracks, report = _repair([_track(0, range(0, 10), 30, 60), _track(1, range(5, 15), 31, 60)])
    assert len(tracks) == 2 and report.merges == ()


def test_fragments_with_distant_medians_merge_on_boundary_overlap() -> None:
    # Same object, id switched away and back: track 1's first mask repeats track 0's
    # last mask (adjacent frames, IoU 1), but both tracks' medians sit far apart
    # because their other views cover different arcs, so the footprint rule cannot fire.
    first = {f: _blob(30, 60) for f in range(10)}
    second = {10: _blob(30, 60), **{f: _blob(70, 60) for f in range(11, 20)}}
    tracks, report = _repair([Track(0, first), Track(1, second)])
    assert [t.track_id for t in tracks] == [0]
    assert sorted(tracks[0].masks) == list(range(20))
    assert report.merges == ((0, 1),)


def test_adjacent_tracks_of_different_objects_stay_apart() -> None:
    tracks, report = _repair([_track(0, range(0, 10), 30, 60), _track(1, range(10, 20), 70, 60)])
    assert len(tracks) == 2 and report.merges == ()


def test_distant_tracks_stay_apart() -> None:
    tracks, _ = _repair([_track(0, range(0, 10), 30, 60), _track(1, range(12, 20), 70, 60)])
    assert len(tracks) == 2


def test_a_track_that_jumps_between_two_objects_is_split() -> None:
    jumpy = Track(0, {f: _blob(30, 60) if f < 6 else _blob(70, 60) for f in range(12)})
    tracks, report = _repair([jumpy])
    assert len(tracks) == 2
    assert report.splits == ((0, (0, 1)),)
    assert sorted(len(t.masks) for t in tracks) == [6, 6]
    assert sorted(f for t in tracks for f in t.masks) == list(range(12))


def test_a_short_stray_group_joins_the_nearest_object_instead_of_splitting() -> None:
    masks = {f: _blob(30, 60) for f in range(10)}
    masks[10] = _blob(70, 60)
    tracks, report = _repair([Track(0, masks)])
    assert len(tracks) == 1 and report.splits == ()


def _sliver() -> BoolMask:
    sliver = np.zeros((100, 100), dtype=bool)
    sliver[58:60, 69:71] = True  # 4 px far away, like a nearly hidden object
    return sliver


def test_an_occlusion_sliver_window_splits_without_the_guard() -> None:
    from dataclasses import replace

    masks = {f: _blob(30, 60) for f in range(12)}
    for f in range(5, 9):
        masks[f] = _sliver()
    tracks, report = repair_tracks(
        [Track(0, masks)], {f: POSE for f in range(12)}, K, TABLE, replace(CFG, sliver_frac=0.0)
    )
    assert len(tracks) == 2 and report.splits == ((0, (0, 1)),)


def test_an_occlusion_sliver_does_not_found_a_new_group() -> None:
    masks = {f: _blob(30, 60) for f in range(12)}
    for f in range(5, 9):
        masks[f] = _sliver()
    tracks, report = _repair([Track(0, masks)])
    assert len(tracks) == 1 and report.splits == ()
    assert sorted(tracks[0].masks) == list(range(12))


def _repair_section(**overrides: object) -> dict[str, Any]:
    repair: dict[str, Any] = {
        "merge_radius_m": 0.05,
        "split_gap_m": 0.10,
        "min_frames": 3,
        "sliver_frac": 0.2,
        "merge_boundary_gap": 2,
        "merge_boundary_iou": 0.5,
    }
    repair.update(overrides)
    return {"segmentation": {"tracking": {"repair": repair}}}


def test_repair_config_reads_handoff_thresholds() -> None:
    cfg = RepairConfig.from_config(_repair_section())
    assert (cfg.merge_boundary_gap, cfg.merge_boundary_iou) == (2, 0.5)
    for key, value in (("merge_boundary_gap", 0), ("merge_boundary_iou", 0.0)):
        with pytest.raises(ValueError):
            RepairConfig.from_config(_repair_section(**{key: value}))


def test_config_rejects_bad_values() -> None:
    cfg = {
        "segmentation": {
            "tracking": {"prompt": "x", "fps": 5, "direction": "both", "timeout_s": 60}
        }
    }
    assert TrackingConfig.from_config(cfg).direction == "both"
    for key, value in (("prompt", " "), ("fps", 0), ("direction", "up")):
        bad = {"segmentation": {"tracking": {**cfg["segmentation"]["tracking"], key: value}}}
        with pytest.raises(ValueError):
            TrackingConfig.from_config(bad)


def test_groups_links_only_within_the_link_distance() -> None:
    points = np.array([[0.0, 0.0], [0.01, 0.0], [0.02, 0.0], [1.0, 1.0], [1.01, 1.0], [5.0, 5.0]])
    assert _groups(points, 0.03) == [[0, 1, 2], [3, 4], [5]]
    assert _groups(np.zeros((0, 2)), 0.03) == []


def test_the_clip_is_encoded_once_for_several_prompts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cv2

    from video2sim.stages import tracking as tracking_module

    frame = tmp_path / "00000.png"
    cv2.imwrite(str(frame), np.full((8, 8, 3), 128, dtype=np.uint8))
    world = WorldModel(
        intrinsics=K,
        frames=[CameraFrame(frame_index=0, image_path=frame.name, T_world_from_camera=POSE)],
        scale_factor=1.0,
        scale_source="marker",
        table_plane=TABLE,
        point_cloud_path="points.ply",
        mean_reprojection_error_px=0.5,
        registered_fraction=1.0,
    )
    encodes: list[Path] = []

    def fake_encode(frames: list[Path], fps: float, out: Path) -> None:
        encodes.append(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"clip")

    monkeypatch.setattr(tracking_module, "encode_clip", fake_encode)
    client = FakeVideoSamClient("")
    cache = ResponseCache(tmp_path / "cache")
    clip = tmp_path / "work" / "clip.mp4"
    for prompt in ("objects on a table", "mug", "cup"):
        cfg = TrackingConfig(prompt, 5.0, "both", 60.0)
        tracks = track_objects(
            world, tmp_path, client, cache, cfg, tmp_path / "work", clip_path=clip
        )
        assert tracks == []
    assert encodes == [clip]  # one encode; the shared clip serves every prompt
    assert client.calls == 3


def test_answer_frames_map_to_world_frame_indices() -> None:
    text = (
        "<0f>3<|box;x1=1;y1=1;x2=2;y2=2;w=8;h=8|><|mask;x=0;y=0;data=2,2,~a|>\n"
        "<1f>3<|box;x1=1;y1=1;x2=2;y2=2;w=8;h=8|><|mask;x=0;y=0;data=2,2,~a|>"
        ",5<|box;x1=4;y1=4;x2=5;y2=5;w=8;h=8|><|mask;x=0;y=0;data=2,2,~a|>"
    )

    def decode(record: SamRecord) -> BoolMask:
        mask = np.zeros((8, 8), dtype=bool)
        mask[record.object_id, 0] = True
        return mask

    tracks = tracks_from_text(text, [10, 20], decode)
    assert [(t.track_id, sorted(t.masks)) for t in tracks] == [(3, [10, 20]), (5, [20])]
    with pytest.raises(ValueError, match="frame"):
        tracks_from_text(text, [10], decode)


def test_scoring_counts_switches_and_fragments(tmp_path: Path) -> None:
    objs = [
        GtObject("a", "box", (0.06, 0.06, 0.06), (0.0, 0.0, 0.03), 0.0, 0.2),
        GtObject("b", "box", (0.06, 0.06, 0.06), (0.2, 0.0, 0.03), 0.0, 0.2),
    ]
    k = Intrinsics(fx=800, fy=800, cx=320, cy=240, width_px=640, height_px=480)
    pose = SE3.from_numpy(
        np.array([[1.0, 0, 0, 0.1], [0, -1.0, 0, 0], [0, 0, -1.0, 0.5], [0, 0, 0, 1.0]])
    )
    a, b = (project_box_mask(o, k, pose) for o in objs)
    tracks = {0: {0: a, 1: a, 2: a, 3: b}, 1: {4: a}}  # track 0 jumps from a to b
    score = score_tracks(objs, k, {i: pose for i in range(5)}, tracks)
    assert score.switches == 1
    assert score.tracks_per_object == {"a": 2}
    assert score.fragmented == 1 and score.unassigned_tracks == 0
