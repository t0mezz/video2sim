"""Per-cluster VLM check (S12.9 redesign, steps 4-5)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from video2sim.models.hosted import ResponseCache
from video2sim.schemas import SE3, CameraFrame, Intrinsics, Plane, WorldModel
from video2sim.stages.cluster_check import (
    ClusterCheckConfig,
    check_one,
    choose_views,
    make_crop,
    parse_check_answer,
)
from video2sim.stages.tracking import Track

SHAPE = (48, 64)


def _world(tmp_path: Path, n: int = 4) -> WorldModel:
    import cv2

    frames = []
    for i in range(n):
        image = np.full((SHAPE[0], SHAPE[1], 3), 200, dtype=np.uint8)
        cv2.imwrite(str(tmp_path / f"{i:05d}.png"), image)
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


def _cube() -> NDArray[np.float64]:
    return np.array(
        [[x, y, z] for x in (-0.02, 0.02) for y in (-0.02, 0.02) for z in (0.03, 0.07)],
        dtype=np.float64,
    )


class FakeClient:
    model = "fake-vlm"

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.calls = 0

    def ask(self, prompt: str, images: Sequence[Path]) -> str:
        self.calls += 1
        assert images and all(p.is_file() for p in images)
        return self.answer


CFG = ClusterCheckConfig(views_per_cluster=2, min_azimuth_gap_deg=1.0, box_pad_px=0)


def test_chosen_view_contains_the_box(tmp_path: Path) -> None:
    world = _world(tmp_path)
    views = choose_views(world, _cube(), [], CFG)
    assert views, "a centred cube must have a placeable view"
    for view in views:
        x1, y1, x2, y2 = view.box
        assert 0 <= x1 < x2 < 64 and 0 <= y1 < y2 < 48


def test_crop_size_and_dimming(tmp_path: Path) -> None:
    import cv2

    world = _world(tmp_path)
    views = choose_views(world, _cube(), [], CFG)
    view = views[0]
    kept_mask = np.zeros(SHAPE, dtype=bool)
    kept_mask[view.box[1] : view.box[3] + 1, view.box[0] : view.box[2] + 1] = True
    kept = [Track(0, {view.frame_index: kept_mask})]
    out = make_crop(
        tmp_path / f"{view.frame_index:05d}.png", view, kept, tmp_path / "crop.png", CFG
    )
    crop = cv2.imread(str(out))
    assert crop is not None
    # Margin is 0.5 * box side (2 px) but at least 16 px, clamped by the frame.
    assert crop.shape[1] <= 64 and crop.shape[0] <= 48
    # The kept-mask region was dimmed from 200 toward 200 * 0.35 = 70 (box border aside).
    margin = max(CFG.crop_min_margin_px, round(CFG.crop_margin_frac * 5))
    cx = (view.box[0] + view.box[2]) // 2 - max(view.box[0] - margin, 0)
    cy = (view.box[1] + view.box[3]) // 2 - max(view.box[1] - margin, 0)
    assert int(crop[cy, cx][0]) < 150


def test_parse_accept_reject_unreadable() -> None:
    cfg = ClusterCheckConfig(min_confidence=0.5)
    ok = parse_check_answer(
        '{"is_object": true, "noun": "Mug", "count": 1, "confidence": 0.9}', (), cfg.min_confidence
    )
    assert ok.status == "accepted" and ok.noun == "mug" and ok.count == 1
    low = parse_check_answer(
        '{"is_object": true, "noun": "mug", "count": 1, "confidence": 0.2}', (), cfg.min_confidence
    )
    assert low.status == "rejected"
    deny = parse_check_answer(
        '{"is_object": true, "noun": "table", "count": 1, "confidence": 0.9}',
        ("table",),
        cfg.min_confidence,
    )
    assert deny.status == "rejected"
    not_obj = parse_check_answer(
        '{"is_object": false, "noun": null, "count": 0, "confidence": 0.9}', (), cfg.min_confidence
    )
    assert not_obj.status == "rejected"
    assert parse_check_answer("no json here", (), cfg.min_confidence).status == "unchecked"


def test_check_one_accept_reject_unreadable_exception(tmp_path: Path) -> None:
    crop = tmp_path / "crop.png"
    crop.write_bytes(b"png-bytes")
    cfg = ClusterCheckConfig()
    cache = ResponseCache(tmp_path / "cache")

    accept = FakeClient('{"is_object": true, "noun": "mug", "count": 1, "confidence": 0.9}')
    assert check_one([crop], accept, cache, cfg, ()).status == "accepted"
    # Cached on the crop bytes: a second call with the same crop makes no new call.
    assert check_one([crop], accept, cache, cfg, ()).status == "accepted"
    assert accept.calls == 1

    reject = FakeClient('{"is_object": false, "noun": null, "count": 0, "confidence": 0.9}')
    assert check_one([crop], reject, ResponseCache(tmp_path / "c2"), cfg, ()).status == "rejected"

    unreadable = FakeClient("I cannot tell")
    verdict = check_one([crop], unreadable, ResponseCache(tmp_path / "c3"), cfg, ())
    assert verdict.status == "unchecked"

    class Boom:
        model = "fake-vlm"

        def ask(self, prompt: str, images: Sequence[Path]) -> str:
            raise RuntimeError("down")

    verdict = check_one([crop], Boom(), ResponseCache(tmp_path / "c4"), cfg, ())
    assert verdict.status == "unchecked"
    assert check_one([crop], None, cache, cfg, ()).status == "unchecked"


def test_check_config_from_default_yaml() -> None:
    import yaml

    root = Path(__file__).resolve().parent.parent
    cfg = yaml.safe_load((root / "configs/default.yaml").read_text(encoding="utf-8"))
    out = ClusterCheckConfig.from_config(cfg)
    assert out.views_per_cluster >= 1 and out.max_workers >= 1
    # 0.7, not 0.5: in the 9.8 subset the s24 match bundle was accepted at 0.68.
    assert out.min_confidence == 0.7


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
