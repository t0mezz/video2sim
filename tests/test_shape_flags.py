from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from video2sim.schemas import Flag
from video2sim.stages.shape_flags import (
    ShapeConfig,
    ShapeSignals,
    judge_shape,
    principal_extents,
)

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def cfg() -> ShapeConfig:
    return ShapeConfig.from_config(
        yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    )


def sig(extents: tuple[float, float, float] | None, share: float | None = 0.0) -> ShapeSignals:
    return ShapeSignals(0, extents, 50, share)


def test_extents_follow_the_principal_axes_not_the_world_axes() -> None:
    t = np.linspace(0, 1, 50)
    line = np.stack([t, t, np.zeros_like(t)], axis=1) * 0.1  # a diagonal rod, 0.1 m per axis
    box = np.concatenate([line, line + np.array([0.0, 0.0, 0.02])])
    smallest, _, largest = principal_extents(box, 0.005)
    assert largest == pytest.approx(0.1414 + 0.005, abs=0.01)
    assert smallest < 0.03


def test_small_and_thin_are_independent(cfg: ShapeConfig) -> None:
    assert judge_shape(sig((0.03, 0.04, 0.05)), cfg).flags == (Flag.SMALL,)
    assert judge_shape(sig((0.02, 0.12, 0.15)), cfg).flags == (Flag.THIN,)
    assert judge_shape(sig((0.1, 0.1, 0.2)), cfg).flags == ()


def test_a_small_cube_is_not_thin(cfg: ShapeConfig) -> None:
    assert Flag.THIN not in judge_shape(sig((0.03, 0.035, 0.04)), cfg).flags


def test_occlusion_needs_the_share_and_enough_frames(cfg: ShapeConfig) -> None:
    assert judge_shape(sig((0.1, 0.1, 0.1), 0.6), cfg).flags == (Flag.OCCLUDED,)
    assert judge_shape(sig((0.1, 0.1, 0.1), 0.2), cfg).flags == ()
    short = judge_shape(sig((0.1, 0.1, 0.1), None), cfg)
    assert short.flags == () and "occlusion not measured" in short.reasons[-1]


def test_missing_cluster_is_reported_not_silently_cleared(cfg: ShapeConfig) -> None:
    out = judge_shape(sig(None), cfg)
    assert out.flags == () and "not measured" in out.reasons[0]


def test_config_rejects_bad_values() -> None:
    base = yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    for key, value in (("thin_max_ratio", 0), ("occluded_share", 1.5), ("min_frames", 0)):
        bad = yaml.safe_load(yaml.safe_dump(base))
        bad["segmentation"]["shape_flags"][key] = value
        with pytest.raises(ValueError):
            ShapeConfig.from_config(bad)
