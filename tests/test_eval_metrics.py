from __future__ import annotations

import pytest

from video2sim.evaluation.baselines import constant_box, oracle
from video2sim.evaluation.metrics import (
    ObjectResult,
    SceneResult,
    aggregate,
    evaluate_scene,
    summarize,
)
from video2sim.evaluation.scenes import GtObject, GtScene, discover_scenes, parse_scene

TARGETS = {
    "object_recall_min": 0.9,
    "size_error_median_max": 0.05,
    "position_error_median_max_m": 0.015,
    "orientation_error_median_max_deg": 10.0,
    "mass_within_30pct_fraction_min": 0.7,
    "runtime_max_minutes": 60,
}


def _scene(tier: str = "easy") -> GtScene:
    objs = (
        GtObject("a", "box", (0.10, 0.04, 0.05), (0.10, 0.10, 0.025), 35.0, 0.30),
        GtObject("b", "cube", (0.05, 0.05, 0.05), (0.30, 0.10, 0.025), 0.0, 0.10),
        GtObject("c", "tall", (0.12, 0.06, 0.20), (0.20, 0.30, 0.10), 120.0, 0.50),
        GtObject("d", "gap", None, None, None, None, ("size_m",)),
    )
    return GtScene("s_test", tier, "dev", "grid", objs)


def _by_id(result: SceneResult) -> dict[str, ObjectResult]:
    return {o.object_id: o for o in result.objects}


def test_perfect_oracle_scores_perfectly() -> None:
    result = evaluate_scene(_scene(), oracle(_scene()))
    assert (result.n_evaluable, result.n_excluded, result.n_matched) == (3, 1, 3)
    assert result.n_false_positive == 0
    for o in result.objects:
        assert o.size_error == pytest.approx(0, abs=1e-9)
        assert o.position_error_m == pytest.approx(0, abs=1e-9)
        assert o.mass_within_tolerance is True
    # The cube has a symmetric footprint, so its yaw is not scored.
    objs = _by_id(result)
    assert objs["b"].orientation_error_deg is None
    assert objs["a"].orientation_error_deg == pytest.approx(0, abs=1e-6)


def test_known_noise_gives_known_error() -> None:
    gt = _scene()
    noisy = oracle(gt, size_sigma=0.0, position_sigma_m=0.0, yaw_sigma_deg=0.0)
    # Shift every box by 1 cm in x and scale mass by 1.5 by editing the prediction directly.
    shifted = []
    for p in noisy.objects:
        v = p.vertices.copy()
        v[:, 0] += 0.01
        shifted.append(type(p)(p.object_id, v, None if p.mass_kg is None else p.mass_kg * 1.5))
    result = evaluate_scene(gt, type(noisy)(shifted))
    for o in result.objects:
        assert o.position_error_m == pytest.approx(0.01, abs=1e-9)
        assert o.mass_error == pytest.approx(0.5)
        assert o.mass_within_tolerance is False


def test_yaw_error_wraps_at_180() -> None:
    gt = GtScene(
        "s",
        "easy",
        "dev",
        "grid",
        (GtObject("a", "box", (0.10, 0.04, 0.05), (0.1, 0.1, 0.025), 170.0, 0.1),),
    )
    pred = oracle(
        GtScene(
            "s",
            "easy",
            "dev",
            "grid",
            (GtObject("a", "box", (0.10, 0.04, 0.05), (0.1, 0.1, 0.025), 5.0, 0.1),),
        )
    )
    (o,) = evaluate_scene(gt, pred).objects
    assert o.orientation_error_deg == pytest.approx(15.0, abs=1e-6)


def test_missing_and_extra_objects() -> None:
    gt = _scene()
    pred = oracle(gt, drop_fraction=1.0)
    result = evaluate_scene(gt, pred)
    assert result.n_matched == 0 and all(not o.matched for o in result.objects)
    extra = constant_box((0.9, 0.9))
    assert evaluate_scene(gt, extra).n_false_positive == 1


def test_matched_object_without_mass_fails_the_mass_metric() -> None:
    gt = _scene()
    pred = oracle(gt)
    bare = type(pred)([type(p)(p.object_id, p.vertices, None) for p in pred.objects])
    result = evaluate_scene(gt, bare)
    assert all(o.mass_within_tolerance is False for o in result.objects)


def test_summary_rows_targets_and_not_measured() -> None:
    result = evaluate_scene(_scene(), oracle(_scene()))
    rows = {r["name"]: r for r in summarize([result], TARGETS)}
    assert rows["object_recall"]["value"] == 1.0 and rows["object_recall"]["passed"] is True
    assert rows["size_error_median"]["passed"] is True
    assert rows["orientation_error_median_deg"]["n"] == 2  # the cube is skipped
    assert rows["runtime_max_minutes"]["status"] == "no_data"
    assert rows["stability_pass_fraction"]["status"] == "not_measured"
    floor = evaluate_scene(_scene(), constant_box())
    assert {r["name"]: r for r in summarize([floor], TARGETS)}["object_recall"]["passed"] is False


def test_aggregate_splits_by_tier() -> None:
    results = [
        evaluate_scene(_scene("easy"), oracle(_scene("easy"))),
        evaluate_scene(_scene("hard"), constant_box()),
    ]
    agg = aggregate(results, TARGETS)
    assert set(agg["per_tier"]) == {"easy", "hard"}
    assert agg["n_scenes"] == 2 and agg["n_excluded_objects"] == 2


def test_oracle_is_reproducible_across_calls() -> None:
    a = oracle(_scene(), seed=3, position_sigma_m=0.01)
    b = oracle(_scene(), seed=3, position_sigma_m=0.01)
    assert all((x.vertices == y.vertices).all() for x, y in zip(a.objects, b.objects, strict=True))


def test_discover_scenes_lists_missing_ground_truth(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import json

    ok = tmp_path / "s01"
    ok.mkdir()
    (ok / "ground_truth.json").write_text(
        json.dumps(
            {"scene_id": "s01", "tier": "easy", "split": "dev", "board": "grid", "objects": []}
        )
    )
    (tmp_path / "s02").mkdir()
    test = tmp_path / "s03"
    test.mkdir()
    (test / "ground_truth.json").write_text(
        json.dumps(
            {"scene_id": "s03", "tier": "easy", "split": "test", "board": "grid", "objects": []}
        )
    )
    scenes, missing = discover_scenes(tmp_path, ["dev"])
    assert [s.scene_id for s in scenes] == ["s01"] and missing == ["s02"]
    scenes, _ = discover_scenes(tmp_path, ["dev"], scene_id="s03")
    assert [s.scene_id for s in scenes] == ["s03"]
    _, missing = discover_scenes(tmp_path, ["dev"], scene_id="nope")
    assert missing == ["nope"]
    assert parse_scene({"scene_id": "x", "tier": "easy", "split": "dev", "objects": []}).board == ""


def test_scene_with_a_deformable_object_is_flagged() -> None:
    rigid = _scene()
    assert not rigid.has_deformable
    cloth = GtObject("c", "cloth", (0.2, 0.2, 0.01), (0.3, 0.3, 0.005), 0.0, 0.05, (), True)
    assert GtScene("s_def", "hard", "dev", "grid", (*rigid.objects, cloth)).has_deformable
