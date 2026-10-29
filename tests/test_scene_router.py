import json
from pathlib import Path

import pytest
import yaml

from video2sim.stages.discovery_vlm_sam import ResponseCache
from video2sim.stages.scene_router import (
    RouterConfig,
    compare_counts,
    decide_route,
    parse_answer,
    route_scene,
)

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def cfg() -> RouterConfig:
    return RouterConfig.from_config(
        yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    )


def answer(surface: str = "table", objects: list[str] | None = None, conf: float = 0.9) -> str:
    return json.dumps(
        {
            "surface": surface,
            "objects": objects if objects is not None else ["mug"],
            "confidence": conf,
        }
    )


def test_parse_reads_json_wrapped_in_prose(cfg: RouterConfig) -> None:
    parsed = parse_answer("Sure: " + answer("surface", ["Cube", " "]) + " done", cfg)
    assert parsed is not None
    assert (parsed.surface, parsed.objects) == ("surface", ("cube",))


@pytest.mark.parametrize(
    "text", ["None", "{}", answer("moon"), answer(conf=1.5), '{"surface": "table"}']
)
def test_parse_rejects_unusable_answers(cfg: RouterConfig, text: str) -> None:
    assert parse_answer(text, cfg) is None


def test_route_uses_the_pool_prompt_of_the_chosen_surface(cfg: RouterConfig) -> None:
    route = decide_route(answer("surface", ["cube"]), cfg)
    assert (route.prompt, route.fell_back) == (cfg.pool["surface"][1], False)


def test_low_confidence_falls_back_but_keeps_the_object_list(cfg: RouterConfig) -> None:
    route = decide_route(answer("surface", ["cube"], conf=0.2), cfg)
    assert route.fell_back and route.surface == cfg.default_surface
    assert route.objects == ("cube",)


def test_failure_and_unreadable_answers_fall_back(cfg: RouterConfig) -> None:
    for route in (decide_route(None, cfg, failure="Timeout"), decide_route("None", cfg)):
        assert route.fell_back and route.prompt == cfg.pool[cfg.default_surface][1]
        assert route.objects == ()


def test_shortfall_of_tracks_is_suspected_extra_tracks_are_not(cfg: RouterConfig) -> None:
    route = decide_route(answer(objects=["a", "b", "c"]), cfg)
    assert compare_counts(route, 2).missing_suspected
    assert not compare_counts(route, 3).missing_suspected
    assert not compare_counts(route, 5).missing_suspected


def test_no_list_means_no_suspicion(cfg: RouterConfig) -> None:
    assert not compare_counts(decide_route(None, cfg, failure="x"), 0).missing_suspected


def test_route_scene_without_client_makes_no_call(cfg: RouterConfig, tmp_path: Path) -> None:
    route = route_scene(None, tmp_path, None, ResponseCache(tmp_path), cfg)  # type: ignore[arg-type]
    assert route.fell_back


def test_tracking_prompt_uses_the_noun_only_off_the_default_surface(cfg: RouterConfig) -> None:
    from video2sim.stages.scene_router import tracking_prompt

    one = decide_route(answer("surface", ["cube"]), cfg)
    assert tracking_prompt(one, cfg) == "cube"
    many = decide_route(answer("surface", ["cube", "ball"]), cfg)
    assert tracking_prompt(many, cfg) == cfg.pool["surface"][1]
    table = decide_route(answer("table", ["cube"]), cfg)
    assert tracking_prompt(table, cfg) == cfg.pool["table"][1]
    failed = decide_route(None, cfg, failure="Timeout")
    assert tracking_prompt(failed, cfg) == cfg.pool[cfg.default_surface][1]
