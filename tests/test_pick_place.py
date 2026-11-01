"""Choosing an object and placing the arm, with no simulation, plus one run if the Panda is here."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from test_export import make_asset, yawed
from video2sim.pick_place import (
    DemoConfig,
    Footprint,
    choose_object,
    footprint,
    plan_pick,
    run_demo,
)
from video2sim.schemas import Flag, Plane
from video2sim.stages.settling import SettlingConfig
from video2sim.stages.urdf_writer import write_urdf

ROOT = Path(__file__).resolve().parents[1]
FLAT = Plane(normal=(0.0, 0.0, 1.0), offset_m=0.0)


@pytest.fixture
def cfg() -> DemoConfig:
    config = yaml.safe_load((ROOT / "configs/default.yaml").read_text())
    targets = yaml.safe_load((ROOT / "configs/targets.yaml").read_text())
    return DemoConfig.from_config(config, targets, ROOT)


def block(name: str, x: float, y: float, width: float = 0.05, top: float = 0.08) -> Footprint:
    return Footprint(name, np.array([x, y]), np.array([1.0, 0.0]), width, 0.08, 0.0, top)


def test_footprint_finds_the_narrow_side_of_a_yawed_box(tmp_path: Path) -> None:
    asset = make_asset(tmp_path, "obj_000", yawed(0.5, 0.1, 0.2))
    f = footprint(asset, asset.T_world_from_object.to_numpy(), tmp_path)
    assert f.width_m == pytest.approx(0.06, abs=1e-3)
    assert f.length_m == pytest.approx(0.10, abs=1e-3)
    assert f.center == pytest.approx([0.1, 0.2], abs=1e-3)
    assert abs(f.long_axis @ [np.cos(0.5), np.sin(0.5)]) == pytest.approx(1.0, abs=1e-3)


def test_arm_and_goal_avoid_a_neighbour(cfg: DemoConfig) -> None:
    target = block("a", 0.0, 0.0)
    neighbour = block("b", 0.25, 0.0)
    plan = plan_pick(target, [neighbour], 0.0, cfg)
    assert plan is not None
    assert plan.base_xy[0] < 0.2
    assert np.linalg.norm(plan.place_offset_m) == pytest.approx(cfg.place_distance_m)
    assert np.linalg.norm(target.center + plan.place_offset_m - neighbour.center) > 0.1


def test_grasp_stays_above_the_table_on_a_low_object(cfg: DemoConfig) -> None:
    plan = plan_pick(block("a", 0.0, 0.0, top=0.03), [], 0.0, cfg)
    assert plan is not None
    assert plan.grasp_m[2] == pytest.approx(cfg.min_grasp_height_m)


def test_choice_leaves_out_wide_flagged_and_stacked_objects(
    tmp_path: Path, cfg: DemoConfig
) -> None:
    names = ["wide", "proxy", "stacked", "good"]
    assets = [make_asset(tmp_path, n, yawed(0.0, 0.5 * i, 0.0)) for i, n in enumerate(names)]
    assets[1] = assets[1].model_copy(update={"flags": [Flag.PROXY_GEOMETRY]})
    prints = {
        "wide": block("wide", 0.0, 0.0, width=0.075),
        "proxy": block("proxy", 0.5, 0.0),
        "stacked": Footprint(
            "stacked", np.array([1.0, 0.0]), np.array([1.0, 0.0]), 0.05, 0.08, 0.05, 0.1
        ),
        "good": block("good", 1.5, 0.0),
    }
    plan = choose_object(assets, prints, 0.0, cfg)
    assert plan is not None
    assert plan.object_id == "good"
    assert plan.width_m == pytest.approx(0.05)


def test_a_crowded_object_is_not_chosen(tmp_path: Path, cfg: DemoConfig) -> None:
    assets = [make_asset(tmp_path, n, yawed(0.0, 0.0, 0.0)) for n in ("a", "b")]
    prints = {"a": block("a", 0.0, 0.0), "b": block("b", 0.1, 0.0)}
    assert choose_object(assets, prints, 0.0, cfg) is None


@pytest.mark.slow
def test_panda_picks_a_box_and_places_it(tmp_path: Path, cfg: DemoConfig) -> None:
    if not cfg.panda_model.exists():
        pytest.skip("run scripts/fetch_panda.sh")
    config = yaml.safe_load((ROOT / "configs/default.yaml").read_text())
    targets = yaml.safe_load((ROOT / "configs/targets.yaml").read_text())
    asset = make_asset(tmp_path, "obj_000", yawed(0.4, 0.0, 0.0))
    urdfs = {"obj_000": write_urdf(tmp_path / "urdf", asset, tmp_path)}
    settling = SettlingConfig.from_config(config, targets)
    plan, cases = run_demo(urdfs, [asset], FLAT, settling, tmp_path, cfg)
    assert plan is not None
    assert [c.case for c in cases if not c.success] == []
