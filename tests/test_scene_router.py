import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from video2sim.models.hosted import ResponseCache
from video2sim.schemas import SE3, CameraFrame, Intrinsics, Plane, WorldModel
from video2sim.stages.scene_router import (
    RouterConfig,
    build_prompt,
    compare_counts,
    decide_route,
    merge_inventory,
    parse_answer,
    parse_view_answer,
    reduce_views,
    route_scene,
    sanitise_noun,
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


def test_stack_surface_routes_multi_object_scene_to_the_singular_prompt(
    cfg: RouterConfig,
) -> None:
    from video2sim.stages.scene_router import tracking_prompt

    route = decide_route(answer("stack", ["box", "box", "cube"]), cfg)
    assert not route.fell_back
    assert route.prompt == "object on a table"
    assert tracking_prompt(route, cfg) == "object on a table"


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


def view_answer(
    surface: str = "table",
    items: Sequence[tuple[str, int, bool]] = (),
    conf: float = 0.9,
) -> str:
    return json.dumps(
        {
            "surface": surface,
            "confidence": conf,
            "objects": [
                {"noun": noun, "count": count, "partly_hidden": hidden}
                for noun, count, hidden in items
            ],
        }
    )


class MapClient:
    """RouterClient answering per image name, text-only calls under ``""``."""

    def __init__(
        self,
        answers: dict[str, str],
        fail: frozenset[str] = frozenset(),
        default: str = "",
    ) -> None:
        self.answers = answers
        self.fail = fail
        self.default = default
        self.calls: list[tuple[str, list[str]]] = []

    def ask(self, prompt: str, images: Sequence[Path]) -> str:
        names = [Path(i).name for i in images]
        self.calls.append((prompt, names))
        if not names:
            if "" in self.answers:
                return self.answers[""]
            raise TimeoutError("no text-only answer")
        if names[0] in self.fail:
            raise TimeoutError("boom")
        return self.answers.get(names[0], self.default)


def _coverage_world(run_dir: Path, n: int = 4) -> WorldModel:
    for i in range(n):
        # Distinct gray per frame: byte-identical frames share one
        # byte-keyed cache entry, so parallel views would read each other's
        # answer (the intermittent CI failure in the two coverage tests).
        cv2.imwrite(str(run_dir / f"{i:05d}.png"), np.full((48, 64, 3), 100 + 20 * i, np.uint8))
    (run_dir / "points.ply").write_text(
        "ply\nformat ascii 1.0\nelement vertex 4\nproperty float x\nproperty float y\n"
        "property float z\nend_header\n" + "".join(f"{x} 0 0.01\n" for x in (-0.2, -0.1, 0.1, 0.2)),
        encoding="utf-8",
    )
    poses = [(-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0)]
    frames = []
    for i in range(n):
        a = np.diag([1.0, -1.0, -1.0, 1.0])
        a[:3, 3] = [*poses[i], 1.0]
        frames.append(
            CameraFrame(
                frame_index=i,
                image_path=f"{i:05d}.png",
                T_world_from_camera=SE3.from_numpy(a),
            )
        )
    return WorldModel(
        intrinsics=Intrinsics(fx=50.0, fy=50.0, cx=32.0, cy=24.0, width_px=64, height_px=48),
        frames=frames,
        scale_factor=1.0,
        scale_source="marker",
        table_plane=Plane(normal=(0.0, 0.0, 1.0), offset_m=0.0),
        point_cloud_path="points.ply",
        mean_reprojection_error_px=0.5,
        registered_fraction=1.0,
    )


def _coverage_cfg(cfg: RouterConfig) -> RouterConfig:
    return replace(cfg, views=replace(cfg.views, mode="coverage"))


def _view_name(i: int) -> str:
    return f"view_{i:05d}_1024px.png"


def test_even_mode_reproduces_todays_single_call_and_cache_key(
    cfg: RouterConfig, tmp_path: Path
) -> None:
    from video2sim.stages.segmentation import sample_frame_indices

    world = _coverage_world(tmp_path)
    raw = answer("surface", ["cube"])
    client = MapClient({}, default=raw)
    route = route_scene(world, tmp_path, client, ResponseCache(tmp_path / "cache"), cfg)
    assert route == decide_route(raw, cfg)
    assert route.inventory == () and route.view_frames == ()
    assert len(client.calls) == 1
    prompt, names = client.calls[0]
    assert prompt == build_prompt(cfg)
    want = sample_frame_indices(len(world.frames), cfg.n_frames)
    assert names == [world.frames[i].image_path for i in want]


def test_coverage_merges_max_counts_and_keeps_identical_objects(
    cfg: RouterConfig, tmp_path: Path
) -> None:
    world = _coverage_world(tmp_path, n=2)
    ccfg = replace(_coverage_cfg(cfg), merge_synonyms=False)
    client = MapClient(
        {
            _view_name(0): view_answer(items=[("mug", 1, False), ("red cube", 2, False)]),
            _view_name(1): view_answer(items=[("mug", 2, True), ("red cube", 1, False)]),
        }
    )
    route = route_scene(world, tmp_path, client, ResponseCache(tmp_path / "cache"), ccfg)
    assert not route.fell_back and route.surface == "table"
    assert [(i.noun, i.count) for i in route.inventory] == [("mug", 2), ("red cube", 2)]
    assert route.inventory[0].max_hidden_seen
    assert not route.inventory[1].max_hidden_seen
    assert route.objects == ("mug", "mug", "red cube", "red cube")
    assert route.view_frames == (0, 1)


def test_synonym_merge_on_and_off(cfg: RouterConfig, tmp_path: Path) -> None:
    world = _coverage_world(tmp_path, n=2)
    views = {
        _view_name(0): view_answer(items=[("mug", 2, False)]),
        _view_name(1): view_answer(items=[("cup", 1, False)]),
    }
    groups = json.dumps({"groups": [["mug", "cup"]]})
    on = route_scene(
        world,
        tmp_path,
        MapClient({**views, "": groups}),
        ResponseCache(tmp_path / "c1"),
        replace(_coverage_cfg(cfg), merge_synonyms=True),
    )
    assert [(i.noun, i.count) for i in on.inventory] == [("mug", 2)]
    assert "synonyms merged" in on.reason
    off = route_scene(
        world,
        tmp_path,
        MapClient(views),
        ResponseCache(tmp_path / "c2"),
        replace(_coverage_cfg(cfg), merge_synonyms=False),
    )
    assert [(i.noun, i.count) for i in off.inventory] == [("mug", 2), ("cup", 1)]
    assert "synonym merge off" in off.reason


def test_failed_synonym_call_falls_back_to_exact_strings(cfg: RouterConfig, tmp_path: Path) -> None:
    world = _coverage_world(tmp_path, n=2)
    views = {
        _view_name(0): view_answer(items=[("mug", 1, False)]),
        _view_name(1): view_answer(items=[("cup", 1, False)]),
    }
    client = MapClient(views, fail=frozenset({""}))
    route = route_scene(
        world,
        tmp_path,
        client,
        ResponseCache(tmp_path / "cache"),
        replace(_coverage_cfg(cfg), merge_synonyms=True),
    )
    assert [(i.noun, i.count) for i in route.inventory] == [("cup", 1), ("mug", 1)]
    assert "synonym merge failed" in route.reason


def test_one_failing_view_does_not_fail_the_scene(cfg: RouterConfig, tmp_path: Path) -> None:
    world = _coverage_world(tmp_path, n=2)
    ccfg = replace(_coverage_cfg(cfg), merge_synonyms=False)
    client = MapClient(
        {_view_name(1): view_answer(items=[("mug", 1, False)])},
        fail=frozenset({_view_name(0)}),
    )
    route = route_scene(world, tmp_path, client, ResponseCache(tmp_path / "cache"), ccfg)
    assert not route.fell_back
    assert [(i.noun, i.count) for i in route.inventory] == [("mug", 1)]
    assert "1 view(s) failed" in route.reason


def test_all_failing_views_fall_back(cfg: RouterConfig, tmp_path: Path) -> None:
    world = _coverage_world(tmp_path, n=2)
    client = MapClient({}, fail=frozenset({_view_name(0), _view_name(1)}))
    route = route_scene(
        world, tmp_path, client, ResponseCache(tmp_path / "cache"), _coverage_cfg(cfg)
    )
    assert route.fell_back and route.objects == () and route.inventory == ()


def test_nouns_are_sanitised_before_sam_prompts(cfg: RouterConfig) -> None:
    assert sanitise_noun("Red Mug", cfg.noun_denylist) == "red mug"
    assert sanitise_noun("table", cfg.noun_denylist) is None
    assert sanitise_noun("shadow", cfg.noun_denylist) is None
    assert sanitise_noun("one two three four", cfg.noun_denylist) is None
    assert sanitise_noun("x" * 41, cfg.noun_denylist) is None
    assert sanitise_noun("  ", cfg.noun_denylist) is None


def test_reduce_vote_and_max_reduce_are_deterministic(cfg: RouterConfig) -> None:
    from video2sim.stages.scene_router import ViewAnswer, ViewItem

    answers = (
        ViewAnswer("table", 0.9, (ViewItem("mug", 2, False),)),
        ViewAnswer("surface", 0.9, (ViewItem("cup", 1, True),)),
        ViewAnswer("table", 0.6, (ViewItem("mug", 1, False),)),
    )
    inventory = merge_inventory(answers, [["mug"], ["cup"]])
    route = reduce_views(answers, inventory, (0, 1, 2), cfg, "test")
    assert route.surface == "table" and not route.fell_back
    assert [(i.noun, i.count, i.max_hidden_seen) for i in inventory] == [
        ("mug", 2, False),
        ("cup", 1, True),
    ]
    assert parse_view_answer("no json here", cfg) is None
    assert parse_view_answer(view_answer(surface="moon"), cfg) is None
