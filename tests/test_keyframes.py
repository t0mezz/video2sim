from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from video2sim.stages.keyframes import (
    FrameStats,
    KeyframeConfig,
    frame_stats,
    select_keyframes,
)


def _cfg(**over: Any) -> KeyframeConfig:
    base: dict[str, Any] = {
        "count": 3,
        "min_mask_px": 100,
        "border_band_frac": 0.02,
        "occluder_ring_px": 4,
        "weight_area": 1.0,
        "weight_border": 1.0,
        "weight_occlusion": 1.0,
        "weight_sharpness": 0.5,
        "diversity_weight": 0.8,
        "diversity_sigma_deg": 25.0,
        "min_quality": 0.15,
        "min_separation_deg": 5.0,
    }
    base.update(over)
    return KeyframeConfig.from_config(
        {
            "segmentation": {
                "keyframes_per_object": base.pop("count"),
                "keyframes": base,
            }
        }
    )


def _dir(azimuth_deg: float, elevation_deg: float = 40.0) -> Any:
    a, e = np.radians(azimuth_deg), np.radians(elevation_deg)
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


def _stat(
    i: int, area: int = 1000, border: float = 0.0, occ: float = 0.0, sharp: float = 10.0
) -> FrameStats:
    return FrameStats(i, area, border, occ, sharp)


def test_similar_views_are_penalised() -> None:
    # Frames 0 to 2 look from almost the same side, frame 3 from the opposite side and is
    # slightly worse. Without the penalty the top 2 would be 0 and 1.
    stats = [_stat(0), _stat(1), _stat(2), _stat(3, area=900)]
    dirs = {0: _dir(0), 1: _dir(2), 2: _dir(4), 3: _dir(180)}
    plain = select_keyframes(
        stats, dirs, _cfg(count=2, diversity_weight=0.0, min_separation_deg=0.0)
    )
    diverse = select_keyframes(stats, dirs, _cfg(count=2))
    assert [c.frame_index for c in plain] == [0, 1]
    assert [c.frame_index for c in diverse] == [0, 3]
    assert diverse[1].similarity < 0.01
    assert diverse[1].score < diverse[1].quality + 1e-12


def test_first_choice_is_best_quality_and_unpenalised() -> None:
    stats = [_stat(0, area=500), _stat(1, area=1000), _stat(2, area=800)]
    dirs = {0: _dir(0), 1: _dir(90), 2: _dir(180)}
    out = select_keyframes(stats, dirs, _cfg())
    assert out[0].frame_index == 1
    assert out[0].similarity == 0.0
    assert out[0].score == out[0].quality


def test_border_contact_and_occlusion_lower_quality() -> None:
    stats = [_stat(0), _stat(1, border=0.5), _stat(2, occ=0.6)]
    dirs = {0: _dir(0), 1: _dir(120), 2: _dir(240)}
    out = select_keyframes(stats, dirs, _cfg(diversity_weight=0.0))
    assert [c.frame_index for c in out] == [0, 1, 2]


def test_fewer_than_count_when_quality_is_low() -> None:
    stats = [_stat(0), _stat(1, border=0.95), _stat(2, occ=0.95)]
    dirs = {0: _dir(0), 1: _dir(120), 2: _dir(240)}
    out = select_keyframes(stats, dirs, _cfg())
    assert [c.frame_index for c in out] == [0]


def test_narrow_arc_still_fills_slots_but_skips_duplicates() -> None:
    # Views 10 degrees apart are heavily penalised yet still chosen; 2 degrees apart is a repeat.
    stats = [_stat(i) for i in range(4)]
    dirs = {0: _dir(0), 1: _dir(2), 2: _dir(12), 3: _dir(22)}
    out = select_keyframes(stats, dirs, _cfg(count=3))
    assert [c.frame_index for c in out] == [0, 3, 2]


def test_frames_without_direction_are_skipped() -> None:
    assert select_keyframes([_stat(0)], {}, _cfg()) == []


def test_sharpness_ignored_without_images() -> None:
    stats = [_stat(0, sharp=0.0), _stat(1, sharp=0.0)]
    dirs = {0: _dir(0), 1: _dir(180)}
    out = select_keyframes(stats, dirs, _cfg(), use_sharpness=False)
    assert len(out) == 2 and out[0].sharpness == 1.0


def test_frame_stats_border_occlusion_sharpness() -> None:
    cfg = _cfg()
    mask = np.zeros((100, 100), dtype=bool)
    mask[40:60, 40:60] = True
    others = np.zeros_like(mask)
    others[:, 60:] = True  # a neighbour touching the right side of the mask
    rng = np.random.default_rng(0)
    textured = rng.integers(0, 255, (100, 100)).astype(np.uint8)
    flat = np.full((100, 100), 128, dtype=np.uint8)

    s = frame_stats(3, mask, others, textured, cfg)
    assert s is not None and s.frame_index == 3 and s.area_px == 400
    assert s.border_frac == 0.0
    assert 0.2 < s.occlusion_frac < 0.4  # about a quarter of the ring
    flat_s = frame_stats(3, mask, None, flat, cfg)
    assert flat_s is not None and flat_s.occlusion_frac == 0.0
    assert flat_s.sharpness == 0.0 < s.sharpness

    edge = np.zeros((100, 100), dtype=bool)
    edge[40:60, 0:20] = True
    e = frame_stats(0, edge, None, None, cfg)
    assert e is not None and e.border_frac == pytest.approx(0.1)  # 2 of 20 columns in the band

    tiny = np.zeros((100, 100), dtype=bool)
    tiny[:5, :5] = True
    assert frame_stats(0, tiny, None, None, cfg) is None


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="diversity_weight"):
        _cfg(diversity_weight=1.5)
    with pytest.raises(ValueError, match="sigma"):
        _cfg(diversity_sigma_deg=0)
