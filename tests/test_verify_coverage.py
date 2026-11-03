"""End-of-segmentation coverage verification (P-verify)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from numpy.typing import NDArray

from video2sim.models.hosted import ResponseCache
from video2sim.schemas import SE3, CameraFrame, DiscoveryCheck, Intrinsics, Plane, WorldModel
from video2sim.stages.gap_fill import FitPass, FoundBy, GapFillConfig
from video2sim.stages.lift import LiftConfig
from video2sim.stages.marker_filter import MarkerFilterConfig
from video2sim.stages.primitive_fit import PrimitiveConfig
from video2sim.stages.track_repair import RepairConfig
from video2sim.stages.tracking import Track
from video2sim.stages.verify_coverage import (
    VERIFY_ID_BASE,
    VERIFY_ID_STEP,
    VerifyConfig,
    check_coverage,
    execute_verify,
    make_overlays,
    parse_verify_answer,
)

ROOT = Path(__file__).resolve().parent.parent
SHAPE = (48, 64)  # (height, width) of the synthetic masks


def _mask(rows: slice, cols: slice) -> NDArray[np.bool_]:
    canvas = np.zeros(SHAPE, dtype=bool)
    canvas[rows, cols] = True
    return canvas


def _world(n: int = 4) -> WorldModel:
    frames = []
    for i in range(n):
        a = np.diag([1.0, -1.0, -1.0, 1.0])
        a[:3, 3] = [0.2 * i, 0.0, 1.0]
        frames.append(
            CameraFrame(
                frame_index=i, image_path=f"{i:05d}.png", T_world_from_camera=SE3.from_numpy(a)
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


def _write_frames(world: WorldModel, run_dir: Path) -> None:
    for frame in world.frames:
        path = run_dir / frame.image_path
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), np.zeros((48, 64, 3), dtype=np.uint8))


def _cfg() -> dict[str, Any]:
    cfg: dict[str, Any] = yaml.safe_load(
        (ROOT / "configs/default.yaml").read_text(encoding="utf-8")
    )
    return cfg


class _ScriptedVlm:
    """Fake VLM client: replays one answer per call and records its calls."""

    model = "test-verify"

    def __init__(self, answers: Sequence[str]) -> None:
        self.answers = list(answers)
        self.calls: list[tuple[str, list[Path]]] = []

    def ask(self, prompt: str, images: Sequence[Path]) -> str:
        self.calls.append((prompt, list(images)))
        return self.answers[min(len(self.calls) - 1, len(self.answers) - 1)]


def _cloud() -> NDArray[np.float64]:
    rng = np.random.default_rng(3)
    table = rng.uniform([-0.15, -0.15, 0.0], [0.15, -0.15, 0.001], (60, 3))
    blob_a = rng.normal([0.03, 0.0, 0.05], 0.01, (60, 3))
    blob_b = rng.normal([0.03, -0.12, 0.05], 0.01, (60, 3))
    return np.concatenate([table, blob_a, blob_b])


MASKS = {
    # Two bands that share no rows, so neither merges into the other; each
    # holds its own blob's projection (v 24 and v 30) in frames 0 to 2, and
    # neither spans the frame, so lift keeps both clusters.
    "mug": _mask(slice(20, 28), slice(8, 40)),
    "cup": _mask(slice(28, 36), slice(8, 40)),
}


def _lift() -> LiftConfig:
    return replace(LiftConfig.from_config(_cfg()), min_views=2)


def _track_fn(
    masks: dict[str, NDArray[np.bool_]], calls: list[tuple[str, int]]
) -> Callable[[str, int], tuple[list[Track], bool]]:
    def track_fn(prompt: str, offset: int) -> tuple[list[Track], bool]:
        calls.append((prompt, offset))
        mask = masks[prompt]
        return [Track(offset, {i: mask.copy() for i in range(4)})], False

    return track_fn


def _run(
    tmp_path: Path,
    client: _ScriptedVlm,
    vcfg: VerifyConfig,
    masks: dict[str, NDArray[np.bool_]],
    calls: list[tuple[str, int]],
    discovery: DiscoveryCheck | None = None,
) -> tuple[
    tuple[
        FitPass,
        DiscoveryCheck | None,
        dict[str, Any],
        dict[int, FoundBy],
        dict[str, str],
        list[Track],
    ],
    list[tuple[str, int]],
    _ScriptedVlm,
]:
    cfg = _cfg()
    world = _world()
    _write_frames(world, tmp_path)
    return (
        execute_verify(
            world,
            tmp_path,
            [],
            ["mug", "cup", "bottle"],
            vcfg,
            GapFillConfig.from_config(cfg),
            _lift(),
            PrimitiveConfig.from_config(cfg),
            RepairConfig.from_config(cfg),
            MarkerFilterConfig.from_config(cfg),
            _cloud(),
            {f.frame_index: f.T_world_from_camera for f in world.frames},
            _track_fn(masks, calls),
            FitPass([], {}, {}, {}, {}, None),
            discovery or _discovery(4),
            {},
            [0, 1, 2],
            client,
            ResponseCache(tmp_path / "cache"),
            tmp_path / "views",
            ("table",),
            lambda p: ({}, (48, 64)),
        ),
        calls,
        client,
    )


def _discovery(n_frames: int) -> DiscoveryCheck:
    return DiscoveryCheck(
        view_frames=list(range(n_frames)),
        view_coverage=0.5,
        listed=3,
        tracked=0,
        missing_suspected=True,
        unmatched=[],
        unexplained_clusters=0,
        recovered=0,
        video_requests=1,
    )


def test_parse_accepts_listed_missing_and_skips_denied_and_unknown() -> None:
    verdict = parse_verify_answer(
        '{"missing": ["mug", "cup", "table", "spaceship"], "confidence": 0.9}',
        ["mug", "cup", "bottle"],
        ("table",),
        0.5,
    )
    assert verdict.status == "ready"
    assert verdict.missing == ("mug", "cup")
    assert verdict.ignored == ("table", "spaceship")


def test_parse_unreadable_and_low_confidence_verify_nothing() -> None:
    assert parse_verify_answer("no json here", ["mug"], (), 0.5).status == "unchecked"
    low = parse_verify_answer('{"missing": ["mug"], "confidence": 0.1}', ["mug"], (), 0.5)
    assert low.status == "unchecked" and low.reason == "low confidence"


def test_check_coverage_without_client_verifies_nothing(tmp_path: Path) -> None:
    verdict = check_coverage(
        [], ["mug"], 0, None, ResponseCache(tmp_path), VerifyConfig.from_config({}), ()
    )
    assert verdict.status == "unchecked" and verdict.reason == "no VLM client"


def test_check_coverage_caches_the_vlm_answer(tmp_path: Path) -> None:
    world = _world(n=2)
    _write_frames(world, tmp_path)
    overlays = make_overlays(world, tmp_path, [], [0, 1], tmp_path / "views", 1024)
    assert len(overlays) == 2 and all(p.is_file() for p in overlays)
    client = _ScriptedVlm(['{"missing": ["mug"], "confidence": 0.9}'])
    cfg = VerifyConfig.from_config({})
    cache = ResponseCache(tmp_path / "cache")
    first = check_coverage(overlays, ["mug"], 0, client, cache, cfg, ())
    second = check_coverage(overlays, ["mug"], 0, client, cache, cfg, ())
    assert first.missing == ("mug",) and second.missing == ("mug",)
    assert len(client.calls) == 1


def test_no_missing_nouns_make_no_video_request(tmp_path: Path) -> None:
    cfg = _cfg()
    world = _world()
    _write_frames(world, tmp_path)
    client = _ScriptedVlm(['{"missing": [], "confidence": 0.9}'])
    empty = FitPass([], {}, {}, {}, {}, None)
    discovery = _discovery(4)

    def track_fn(prompt: str, offset: int) -> tuple[list[Track], bool]:
        raise AssertionError("no prompt should be sent when nothing is missing")

    second, discovery2, notes, _prov, reasons, new = execute_verify(
        world,
        tmp_path,
        [],
        ["mug"],
        VerifyConfig.from_config(cfg),
        GapFillConfig.from_config(cfg),
        _lift(),
        PrimitiveConfig.from_config(cfg),
        RepairConfig.from_config(cfg),
        MarkerFilterConfig.from_config(cfg),
        _cloud(),
        {f.frame_index: f.T_world_from_camera for f in world.frames},
        track_fn,
        empty,
        discovery,
        {},
        [0, 2],
        client,
        ResponseCache(tmp_path / "cache"),
        tmp_path / "views",
        (),
        lambda p: ({}, (48, 64)),
    )
    assert second.kept == [] and new == [] and reasons == {}
    assert discovery2 is not None and discovery2.tracked == 0
    assert notes["verdict"] == "ready" and notes["requests"] == []
    assert notes["report_lines"] == [
        "verify round 1: no missing nouns in the overlaid views",
        "verify: 0 object(s) recovered in 0 extra request(s) over 1 round(s)",
    ]
    assert notes["stop_reason"] == "nothing missing"


def test_loop_recovers_across_rounds_until_nothing_is_missing(tmp_path: Path) -> None:
    vcfg = replace(VerifyConfig.from_config(_cfg()), max_noun_requests=1)
    client = _ScriptedVlm(
        [
            '{"missing": ["mug", "cup", "table"], "confidence": 0.9}',
            '{"missing": ["cup"], "confidence": 0.9}',
            '{"missing": [], "confidence": 0.9}',
        ]
    )
    calls: list[tuple[str, int]] = []
    (second, discovery2, notes, prov, reasons, new), _, _ = _run(
        tmp_path, client, vcfg, MASKS, calls
    )
    # One prompt per round; the denylisted table never earns one.
    assert [c[0] for c in calls] == ["mug", "cup"]
    assert [c[1] for c in calls] == [VERIFY_ID_BASE, VERIFY_ID_BASE + VERIFY_ID_STEP]
    assert notes["prompted"] == ["mug", "cup"] and notes["capped"] == []
    assert notes["stop_reason"] == "nothing missing" and len(notes["rounds"]) == 3
    assert notes["recovered"] == 2 and len(second.kept) == 2
    assert discovery2 is not None and discovery2.recovered == 2
    assert discovery2.video_requests == 1 + 2
    for tid in [t.track_id for t in second.kept]:
        assert prov[tid] == "noun"
        assert reasons[str(tid)] == f"found by verify: verify '{'mug' if tid == 30000 else 'cup'}'"
    assert len(new) == 2
    assert len(client.calls) == 3  # one VLM look per round


def test_max_turns_stops_early_and_reports_capped(tmp_path: Path) -> None:
    vcfg = replace(VerifyConfig.from_config(_cfg()), max_noun_requests=1, max_turns=1)
    client = _ScriptedVlm(['{"missing": ["mug", "cup"], "confidence": 0.9}'])
    calls: list[tuple[str, int]] = []
    (second, _d2, notes, _prov, _reasons, _new), _, _ = _run(tmp_path, client, vcfg, MASKS, calls)
    assert [c[0] for c in calls] == ["mug"]
    assert notes["capped"] == ["cup"]
    assert notes["stop_reason"] == "max turns (1) reached"
    assert len(client.calls) == 1 and len(second.kept) == 1


def test_minus_one_loops_until_exhausted(tmp_path: Path) -> None:
    vcfg = replace(VerifyConfig.from_config(_cfg()), max_noun_requests=1, max_turns=-1)
    client = _ScriptedVlm(
        [
            '{"missing": ["mug", "cup", "bottle"], "confidence": 0.9}',
            '{"missing": ["mug", "cup", "bottle"], "confidence": 0.9}',
            '{"missing": ["mug", "cup", "bottle"], "confidence": 0.9}',
        ]
    )
    calls: list[tuple[str, int]] = []
    (_second, _d2, notes, _prov, _reasons, _new), _, _ = _run(
        tmp_path, client, vcfg, {**MASKS, "bottle": _mask(slice(20, 28), slice(None))}, calls
    )
    # A noun is prompted at most once even without a turn cap; the loop ends.
    assert sorted(c[0] for c in calls) == ["bottle", "cup", "mug"]
    assert notes["stop_reason"] == "nothing missing"
    assert notes["capped"] == []


def test_unreadable_answer_stops_the_loop_and_keeps_recovery(tmp_path: Path) -> None:
    vcfg = replace(VerifyConfig.from_config(_cfg()), max_noun_requests=1)
    client = _ScriptedVlm(
        [
            '{"missing": ["mug"], "confidence": 0.9}',
            "not json at all",
        ]
    )
    calls: list[tuple[str, int]] = []
    (second, _d2, notes, _prov, _reasons, _new), _, _ = _run(tmp_path, client, vcfg, MASKS, calls)
    assert [c[0] for c in calls] == ["mug"] and len(second.kept) == 1
    assert notes["scene_flags"] == ["verify unchecked: unreadable answer"]
    assert notes["stop_reason"] == "VLM answer unreadable"


def test_verify_config_defaults_and_validation() -> None:
    cfg = VerifyConfig.from_config({})
    assert cfg.enabled is False and cfg.max_turns == 3 and cfg.max_noun_requests == 4
    assert cfg.views == 3
    file_cfg = _cfg()
    assert "verify" in file_cfg["segmentation"]  # the documented wiring path
    file_vcfg = VerifyConfig.from_config(file_cfg)
    assert file_vcfg.enabled is False and file_vcfg.max_turns == 3  # off in the committed config
    for bad in ({"max_turns": 0}, {"max_turns": -2}, {"max_noun_requests": 0}):
        try:
            VerifyConfig.from_config({"segmentation": {"verify": bad}})
        except ValueError:
            continue
        raise AssertionError(f"{bad} should not validate")
