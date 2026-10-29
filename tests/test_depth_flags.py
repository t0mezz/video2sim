from __future__ import annotations

from dataclasses import replace

import pytest

from video2sim.stages.depth_flags import DepthConfig, DepthSignals, judge

CFG = DepthConfig(
    min_points=100,
    flag_err_ratio=2.0,
    flag_short_frac=0.45,
    clear_err_ratio=1.5,
    clear_short_frac=0.35,
    clear_hull_iou=0.8,
    clear_outside_hull_frac=0.05,
)

GOOD = DepthSignals(
    track_id=0,
    sfm_points=500,
    err_ratio=1.0,
    short_frac=0.2,
    outside_hull_frac=0.0,
    hull_iou=0.9,
)


def test_stable_signals_are_cleared() -> None:
    assert judge(GOOD, CFG)[0] == "clear"


def test_noisy_short_lived_points_are_flagged() -> None:
    glass = replace(GOOD, err_ratio=2.35, short_frac=0.51)
    assert judge(glass, CFG)[0] == "flag"


def test_one_bad_signal_alone_is_uncertain() -> None:
    assert judge(replace(GOOD, err_ratio=2.5), CFG)[0] == "uncertain"
    assert judge(replace(GOOD, hull_iou=0.5), CFG)[0] == "uncertain"


def test_too_little_data_is_never_cleared() -> None:
    assert judge(replace(GOOD, sfm_points=20), CFG)[0] == "uncertain"
    assert judge(replace(GOOD, hull_iou=None), CFG)[0] == "uncertain"
    assert judge(replace(GOOD, err_ratio=None), CFG)[0] == "uncertain"


def test_config_rejects_overlapping_limits() -> None:
    base = {
        "min_points": 100, "flag_err_ratio": 2.0, "flag_short_frac": 0.45,
        "clear_err_ratio": 1.5, "clear_short_frac": 0.35, "clear_hull_iou": 0.8,
        "clear_outside_hull_frac": 0.05,
    }  # fmt: skip
    assert DepthConfig.from_config({"segmentation": {"depth_flags": base}}).min_points == 100
    for key, value in (("clear_err_ratio", 2.5), ("clear_short_frac", 0.5), ("min_points", 0)):
        with pytest.raises(ValueError):
            DepthConfig.from_config({"segmentation": {"depth_flags": {**base, key: value}}})
