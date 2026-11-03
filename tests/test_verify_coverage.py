"""End-of-segmentation coverage verification (P-verify)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from numpy.typing import NDArray

from video2sim.models.hosted import ResponseCache
from video2sim.schemas import SE3, CameraFrame, DiscoveryCheck, Intrinsics, Plane, WorldModel
from video2sim.stages.gap_fill import FitPass, GapFillConfig
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


class _Vlm:
    """Fake VLM client: replays one answer and records its calls."""

    model = "test-verify"

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.calls: list[tuple[str, list[Path]]] = []

    def ask(self, prompt: str, images: Sequence[Path]) -> str:
        self.calls.append((prompt, list(images)))
        return self.answer


def _cloud() -> NDArray[np.float64]:
    rng = np.random.default_rng(3)
    table = rng.uniform([-0.15, -0.15, 0.0], [0.15, 0.15, 0.001], (60, 3))
    blob = rng.normal([0.03, 0.0, 0.05], 0.01, (60, 3))
    return np.concatenate([table, blob])


def _discovery(n_frames: int) -> DiscoveryCheck:
    return DiscoveryCheck(
        view_frames=list(range(n_frames)),
        view_coverage=0.5,
        listed=1,
        tracked=0,
        missing_suspected=True,
        unmatched=[],
        unexplained_clusters=0,
        recovered=0,
        video_requests=1,
    )


def test_parse_accepts_listed_missing_and_skips_denied_and_unknown() -> None:
    verdict = parse_verify_answer(
        '{"missing": ["mug", "table", "spaceship", "cup"], "confidence": 0.9}',
        ["mug", "cup", "bottle"],
        ("table",),
        0.5,
    )
    assert verdict.status == "ready"
    assert verdict.missing == ("mug", "cup")
    assert "table" in verdict.ignored and "spaceship" in verdict.ignored


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
    client = _Vlm('{"missing": ["mug"], "confidence": 0.9}')
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
    client = _Vlm('{"missing": [], "confidence": 0.9}')
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
        replace(LiftConfig.from_config(cfg), min_views=2),
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
    )
    assert second is empty and discovery2 is discovery and new == [] and reasons == {}
    assert notes["verdict"] == "ready" and notes["requests"] == []
    assert notes["report_lines"] == ["verify: no missing nouns in the overlaid views"]


def test_missing_noun_is_prompted_capped_and_recovered(tmp_path: Path) -> None:
    cfg = _cfg()
    world = _world()
    _write_frames(world, tmp_path)
    full = np.ones(SHAPE, dtype=bool)
    calls: list[tuple[str, int]] = []

    def track_fn(prompt: str, offset: int) -> tuple[list[Track], bool]:
        calls.append((prompt, offset))
        return [Track(offset, {i: full.copy() for i in range(4)})], False

    client = _Vlm('{"missing": ["mug", "table", "mat", "cup", "bottle", "pen"], "confidence": 0.9}')
    vcfg = replace(VerifyConfig.from_config(cfg), max_noun_requests=2)
    discovery = _discovery(4)
    final, discovery2, notes, prov, reasons, new = execute_verify(
        world,
        tmp_path,
        [],
        ["mug", "cup", "bottle"],
        vcfg,
        GapFillConfig.from_config(cfg),
        replace(LiftConfig.from_config(cfg), min_views=2),
        PrimitiveConfig.from_config(cfg),
        RepairConfig.from_config(cfg),
        MarkerFilterConfig.from_config(cfg),
        _cloud(),
        {f.frame_index: f.T_world_from_camera for f in world.frames},
        track_fn,
        FitPass([], {}, {}, {}, {}, None),
        discovery,
        {},
        [0, 1, 2],
        client,
        ResponseCache(tmp_path / "cache"),
        tmp_path / "views",
        ("table",),
        lambda p: ({}, (48, 64)),
    )
    # The denylisted table and the unlisted mat/pen earn no request; the cap takes two.
    assert [c[0] for c in calls] == ["mug", "cup"]
    assert [c[1] for c in calls] == [VERIFY_ID_BASE, VERIFY_ID_BASE + VERIFY_ID_STEP]
    assert notes["prompted"] == ["mug", "cup"] and notes["capped"] == ["bottle"]
    assert set(notes["ignored"]) >= {"table", "mat", "pen"}
    assert notes["recovered"] == 1 and len(final.kept) == 1
    assert discovery2 is not None and discovery2.recovered == 1
    assert discovery2.video_requests == discovery.video_requests + 2
    kept_id = final.kept[0].track_id
    assert prov[kept_id] == "noun"
    assert reasons[str(kept_id)] == "found by verify: verify 'mug'"
    assert notes["report_lines"][0] == "verify: VLM names 3 missing noun(s)"
    assert f"obj_{kept_id:03d}: found by verify: verify 'mug'" in notes["report_lines"]
    assert len(new) == 2


def test_verify_config_defaults_and_validation() -> None:
    cfg = VerifyConfig.from_config({})
    assert cfg.enabled is False and cfg.max_noun_requests == 4 and cfg.views == 3
    for bad in ({"max_noun_requests": 9}, {"views": 0}, {"min_confidence": 2.0}):
        try:
            VerifyConfig.from_config({"segmentation": {"verify": bad}})
        except ValueError:
            continue
        raise AssertionError(f"{bad} should not validate")
