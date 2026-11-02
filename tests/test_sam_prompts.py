"""Object-prompt builder for SAM 3.1 video (S12.9.1)."""

from __future__ import annotations

import re

import pytest

from video2sim.models.sam_prompts import ObjectPrompt, build_object_prompt


def _box_prompt(
    object_id: int = 1, box: tuple[float, float, float, float] = (10.0, 20.0, 30.0, 40.0)
) -> ObjectPrompt:
    return ObjectPrompt(object_id=object_id, box=box)


def test_worked_example_scales_and_renders_a_point() -> None:
    # A detection at (100, 50) in a 640x360 image lands at (300, 150) in 1920x1080 media.
    x, y = 100 * 1920 / 640, 50 * 1080 / 360
    prompt = ObjectPrompt(object_id=1, box=(x, y, x, y), positive_points=((x, y),))
    assert build_object_prompt({0: [prompt]}, (1920, 1080)) == (
        "<0f>1<|box;x1=300;y1=150;x2=300;y2=150;w=1920;h=1080|><|point;x=300;y=150;w=1920;h=1080|>"
    )


def test_coordinates_are_floored_and_clamped() -> None:
    prompt = _box_prompt(box=(10.9, 20.9, 30.9, 40.9))
    assert "<|box;x1=10;y1=20;x2=30;y2=40;w=64;h=48|>" in build_object_prompt(
        {0: [prompt]}, (64, 48)
    )
    outside = _box_prompt(box=(-5.0, -5.0, 1000.0, 1000.0))
    assert "<|box;x1=0;y1=0;x2=63;y2=47;w=64;h=48|>" in build_object_prompt(
        {0: [outside]}, (64, 48)
    )


def test_two_objects_on_two_frames_come_in_order() -> None:
    frames = {
        120: [_box_prompt(2), _box_prompt(1)],
        30: [ObjectPrompt(object_id=2, box=(0.0, 0.0, 5.0, 5.0))],
    }
    assert build_object_prompt(frames, (64, 48)) == (
        "<30f>2<|box;x1=0;y1=0;x2=5;y2=5;w=64;h=48|>"
        "<120f>1<|box;x1=10;y1=20;x2=30;y2=40;w=64;h=48|>,"
        "2<|box;x1=10;y1=20;x2=30;y2=40;w=64;h=48|>"
    )


def test_negative_points_carry_a_leading_minus() -> None:
    prompt = ObjectPrompt(
        object_id=1,
        box=(10.0, 20.0, 30.0, 40.0),
        positive_points=((12.0, 22.0),),
        negative_points=((50.0, 40.0),),
    )
    assert build_object_prompt({0: [prompt]}, (64, 48)) == (
        "<0f>1<|box;x1=10;y1=20;x2=30;y2=40;w=64;h=48|>"
        "<|point;x=12;y=22;w=64;h=48|>"
        "-<|point;x=50;y=40;w=64;h=48|>"
    )


def test_round_trip_through_a_small_parser() -> None:
    text = build_object_prompt(
        {
            30: [_box_prompt(2), _box_prompt(1)],
            120: [
                ObjectPrompt(
                    object_id=2,
                    box=(0.0, 0.0, 5.0, 5.0),
                    negative_points=((1.0, 1.0),),
                )
            ],
        },
        (64, 48),
    )
    parts = re.split(r"<(\d+)f>", text)
    frames_out = [int(n) for n in parts[1::2]]
    blocks = list(zip(frames_out, parts[2::2], strict=True))
    assert [frame for frame, _ in blocks] == [30, 120]
    ids = [[int(m) for m in re.findall(r"(?:^|,)(\d+)<\|", body)] for _, body in blocks]
    assert ids == [[1, 2], [2]]
    assert text.count("-<|point;") == 1


def test_rejects_duplicate_id_empty_frames_and_inverted_boxes() -> None:
    with pytest.raises(ValueError, match="twice"):
        build_object_prompt({0: [_box_prompt(1), _box_prompt(1)]}, (64, 48))
    with pytest.raises(ValueError, match="no prompts"):
        build_object_prompt({0: []}, (64, 48))
    with pytest.raises(ValueError, match="no frames"):
        build_object_prompt({}, (64, 48))
    with pytest.raises(ValueError, match="inverted box"):
        build_object_prompt({0: [_box_prompt(box=(30.0, 20.0, 10.0, 40.0))]}, (64, 48))
    with pytest.raises(ValueError, match="inverted box"):
        build_object_prompt({0: [_box_prompt(box=(10.0, 40.0, 30.0, 20.0))]}, (64, 48))
