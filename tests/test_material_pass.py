from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from video2sim.models.hosted import ResponseCache
from video2sim.schemas import Flag
from video2sim.stages.depth_flags import DepthSignals, Verdict
from video2sim.stages.material_pass import (
    MaterialConfig,
    decide,
    decide_all,
    parse_answer,
    pick_frames,
)
from video2sim.stages.tracking import Track

ROOT = Path(__file__).resolve().parent.parent
CFG = MaterialConfig(
    keyframes=2,
    crop_pad_px=40,
    min_confidence=0.6,
    max_workers=2,
    timeout_s=30.0,
    retries=2,
    finishes=("matte", "glossy", "metallic", "mirror", "transparent", "translucent"),
    reflective_finishes=("metallic", "mirror", "transparent", "translucent"),
    classes=("plastic_hard", "glass", "metal_steel", "other"),
)
SIGNALS = DepthSignals(
    track_id=3, sfm_points=500, err_ratio=1.0, short_frac=0.2, outside_hull_frac=0.0, hull_iou=0.9
)


def _answer(
    finish: str = "matte",
    confidence: float = 0.9,
    cls: str = "plastic_hard",
    rigidity: str = "rigid",
) -> str:
    return json.dumps(
        {
            "category": "box",
            "material_class": cls,
            "finish": finish,
            "hollow": "solid",
            "rigidity": rigidity,
            "confidence": confidence,
        }
    )


def test_default_config_is_valid() -> None:
    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    out = MaterialConfig.from_config(cfg)
    assert out.keyframes == 2 and set(out.reflective_finishes) <= set(out.finishes)


def test_config_rejects_bad_values() -> None:
    base = {
        "keyframes": 2, "crop_pad_px": 40, "min_confidence": 0.6,
        "max_workers": 4, "timeout_s": 120, "retries": 4,
        "finishes": ["matte", "mirror"], "reflective_finishes": ["mirror"],
        "classes": ["glass", "other"],
    }  # fmt: skip
    assert MaterialConfig.from_config({"segmentation": {"material_pass": base}}).keyframes == 2
    for key, value in (
        ("keyframes", 0),
        ("min_confidence", 1.5),
        ("max_workers", 0),
        ("timeout_s", 0),
        ("retries", 0),
        ("classes", ["glass", "glass"]),
        ("reflective_finishes", ["transparent"]),
    ):
        with pytest.raises(ValueError):
            MaterialConfig.from_config({"segmentation": {"material_pass": {**base, key: value}}})


def test_answer_parses_with_text_around_the_json() -> None:
    got = parse_answer("Here you go:\n" + _answer("Mirror", 0.8, "glass") + "\nThanks", CFG)
    assert got is not None and got.finish == "mirror" and got.material_class == "glass"


def test_invalid_answers_parse_to_none() -> None:
    bad = [
        "None",
        "",
        "{not json}",
        _answer(cls="unobtainium"),
        _answer(finish="sparkly"),
        _answer(confidence=1.5),
        json.dumps({"category": "x"}),
    ]
    assert all(parse_answer(text, CFG) is None for text in bad)


def test_reflective_finish_flags_even_when_geometry_is_clear() -> None:
    d = decide(SIGNALS, "clear", "stable points and hull", _answer("mirror"), CFG)
    assert d.flag is Flag.TRANSPARENT_OR_REFLECTIVE and "mirror" in d.reason


def test_confident_matte_with_stable_geometry_is_unflagged() -> None:
    d = decide(SIGNALS, "clear", "stable", _answer("glossy"), CFG)
    assert d.flag is None and d.answer is not None


def test_geometry_flag_beats_a_matte_answer() -> None:
    d = decide(SIGNALS, "flag", "point error 2.35x", _answer("matte"), CFG)
    assert d.flag is Flag.TRANSPARENT_OR_REFLECTIVE and "geometry" in d.reason


def test_missing_unreadable_or_low_confidence_answers_stay_flagged() -> None:
    for raw, failure in ((None, "RuntimeError"), ("None", None), (_answer(confidence=0.3), None)):
        d = decide(SIGNALS, "clear", "stable", raw, CFG, failure=failure)
        assert d.flag is Flag.TRANSPARENT_OR_REFLECTIVE


def test_pick_frames_prefers_keyframes_then_largest_masks() -> None:
    def mask(n: int) -> np.ndarray:
        m = np.zeros((10, 10), dtype=bool)
        m.flat[:n] = True
        return m

    track = Track(1, {0: mask(5), 1: mask(50), 2: mask(20), 3: mask(30)})
    assert pick_frames(track, [2, 9], 2) == [2, 1]  # 9 is not in the track
    assert pick_frames(track, [], 2) == [1, 3]
    assert pick_frames(track, [3, 0, 1], 2) == [3, 0]


class FakeClient:
    def __init__(self, answer: str | Exception) -> None:
        self.answer, self.calls = answer, 0

    def ask(self, prompt: str, images: Sequence[Path]) -> str:
        self.calls += 1
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def _crops(tmp_path: Path):  # type: ignore[no-untyped-def]
    def crops(track_id: int) -> list[Path]:
        paths = [tmp_path / f"{track_id}_{i}.png" for i in range(2)]
        for p in paths:
            p.write_bytes(b"png")
        return paths

    return crops


def test_decide_all_asks_once_and_caches(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path / "cache")
    verdicts: list[tuple[Verdict, str]] = [("clear", "stable")]
    first, second = FakeClient(_answer("mirror")), FakeClient(_answer("matte"))
    a = decide_all([SIGNALS], verdicts, CFG, _crops(tmp_path), first, cache)[0]
    b = decide_all([SIGNALS], verdicts, CFG, _crops(tmp_path), second, cache)[0]
    assert first.calls == 1 and second.calls == 0  # the rerun is answered from the cache
    assert a.flag is Flag.TRANSPARENT_OR_REFLECTIVE and b.flag is Flag.TRANSPARENT_OR_REFLECTIVE


def test_decide_all_flags_on_failed_call_and_without_client(tmp_path: Path) -> None:
    verdicts: list[tuple[Verdict, str]] = [("clear", "stable")]
    failed = decide_all([SIGNALS], verdicts, CFG, _crops(tmp_path), FakeClient(OSError()))[0]
    none = decide_all([SIGNALS], verdicts, CFG, _crops(tmp_path), None)[0]
    assert failed.flag is Flag.TRANSPARENT_OR_REFLECTIVE and "OSError" in failed.reason
    assert none.flag is Flag.TRANSPARENT_OR_REFLECTIVE


def test_decisions_carry_timing_and_cache_flag(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path / "cache")
    verdicts: list[tuple[Verdict, str]] = [("clear", "stable")]
    live = decide_all([SIGNALS], verdicts, CFG, _crops(tmp_path), FakeClient(_answer()), cache)[0]
    again = decide_all([SIGNALS], verdicts, CFG, _crops(tmp_path), FakeClient(_answer()), cache)[0]
    assert live.seconds is not None and not live.cached
    assert again.seconds is not None and again.cached
    assert decide_all([SIGNALS], verdicts, CFG, _crops(tmp_path), None)[0].seconds is None


def test_parallel_requests_keep_track_order(tmp_path: Path) -> None:
    signals = [replace(SIGNALS, track_id=i) for i in range(6)]
    verdicts: list[tuple[Verdict, str]] = [("clear", "stable")] * 6
    got = decide_all(signals, verdicts, CFG, _crops(tmp_path), FakeClient(_answer()))
    assert [d.track_id for d in got] == list(range(6))


def test_non_rigid_answer_raises_the_non_rigid_reason() -> None:
    for rigidity in ("deformable", "semi_rigid"):
        d = decide(SIGNALS, "clear", "ok", _answer(rigidity=rigidity), CFG)
        assert d.non_rigid is not None and rigidity in d.non_rigid
        assert d.flag is None  # matte, so not reflective


def test_rigid_unknown_and_missing_rigidity_do_not_raise_it() -> None:
    assert decide(SIGNALS, "clear", "ok", _answer(rigidity="rigid"), CFG).non_rigid is None
    assert decide(SIGNALS, "clear", "ok", _answer(rigidity="unknown"), CFG).non_rigid is None
    old = json.loads(_answer())
    del old["rigidity"]
    parsed = parse_answer(json.dumps(old), CFG)
    assert parsed is not None and parsed.rigidity == "unknown"


def test_low_confidence_rigidity_is_not_trusted() -> None:
    d = decide(SIGNALS, "clear", "ok", _answer(confidence=0.3, rigidity="deformable"), CFG)
    assert d.non_rigid is None


def test_invalid_rigidity_makes_the_answer_unreadable() -> None:
    assert parse_answer(_answer(rigidity="squishy"), CFG) is None


def test_prompt_lists_the_rigidity_values() -> None:
    from video2sim.stages.material_pass import RIGIDITY_VALUES, build_prompt

    prompt = build_prompt(CFG)
    assert all(v in prompt for v in RIGIDITY_VALUES) and "{" not in prompt.replace('{"', "")
