"""Overlay audit after tracking (S12.9, task 9.7): VLM check for untracked objects."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import yaml
from numpy.typing import NDArray

from video2sim.models.hosted import ResponseCache
from video2sim.schemas import SE3, CameraFrame, Intrinsics, Plane, WorldModel
from video2sim.stages.overlay_audit import (
    OverlayAuditConfig,
    audit_missing_nouns,
    build_audit_prompt,
    parse_audit_answer,
    render_audit_view,
    render_tracks_video,
    track_colors,
)
from video2sim.stages.tracking import Track

ROOT = Path(__file__).resolve().parent.parent
SHAPE = (48, 64)  # (height, width) of the synthetic frames and masks


def _mask(rows: slice, cols: slice) -> NDArray[np.bool_]:
    canvas = np.zeros(SHAPE, dtype=bool)
    canvas[rows, cols] = True
    return canvas


def _frame(path: Path) -> None:
    cv2.imwrite(str(path), np.full((*SHAPE, 3), 128, dtype=np.uint8))


def _world(run_dir: Path, n: int = 3) -> WorldModel:
    frames = []
    for i in range(n):
        name = f"{i:05d}.png"
        _frame(run_dir / name)
        frames.append(
            CameraFrame(
                frame_index=i, image_path=name, T_world_from_camera=SE3.from_numpy(np.eye(4))
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


class FakeAuditClient:
    """Canned VLM answer; records calls."""

    model = "fake-audit"

    def __init__(self, answer: str) -> None:
        self._answer = answer
        self.calls = 0

    def ask(self, prompt: str, images: Sequence[Path]) -> str:
        self.calls += 1
        return self._answer


def _acfg(**over: object) -> OverlayAuditConfig:
    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    g = cfg["segmentation"]["tracking"]["gap_fill"]
    g["enabled"] = True
    audit = dict(g.get("overlay_audit", {}))
    audit["enabled"] = True
    for key, value in over.items():
        if key == "audit":
            assert isinstance(value, dict)
            audit.update(value)
        else:
            g[key] = value
    g["overlay_audit"] = audit
    return OverlayAuditConfig.from_config(cfg)


def test_prompt_names_the_tracked_count() -> None:
    prompt = build_audit_prompt(4)
    assert "4 tracked" in prompt
    assert "nothing else" in prompt


def test_parse_missing_and_uncertain() -> None:
    text = '{"missing": ["Laptop", "bead"], "uncertain": ["shadow"]}'
    missing, uncertain = parse_audit_answer(text, ("table",))
    assert missing == ["laptop", "bead"]
    assert uncertain == ["shadow"]


def test_parse_drops_denied_duplicates_and_garbage() -> None:
    text = (
        '{"missing": ["table", "red shiny laptop computer pro max", '
        '"bead", "bead", ""], "uncertain": []}'
    )
    missing, _ = parse_audit_answer(text, ("table",))
    assert missing == ["bead"]


def test_parse_unreadable_answer_gives_no_nouns() -> None:
    assert parse_audit_answer("no json here", ()) == ([], [])
    assert parse_audit_answer('{"missing": "laptop"}', ()) == ([], [])


def test_render_draws_ids_and_returns_them(tmp_path: Path) -> None:
    frame = tmp_path / "frame.png"
    _frame(frame)
    out = tmp_path / "audit.png"
    drawn = render_audit_view(
        frame, {2: _mask(slice(4, 12), slice(4, 12)), 5: _mask(slice(20, 30), slice(40, 50))}, out
    )
    assert drawn == [2, 5]
    assert out.is_file()
    rendered = cv2.imread(str(out))
    assert rendered is not None and rendered.shape == (*SHAPE, 3)


def test_render_resamples_foreign_masks(tmp_path: Path) -> None:
    frame = tmp_path / "frame.png"
    _frame(frame)
    small = np.zeros((12, 16), dtype=bool)
    small[2:6, 2:6] = True
    out = tmp_path / "audit.png"
    assert render_audit_view(frame, {0: small}, out) == [0]


def test_audit_names_the_uncovered_noun(tmp_path: Path) -> None:
    world = _world(tmp_path)
    kept = [
        Track(2, {0: _mask(slice(4, 12), slice(4, 12)), 1: _mask(slice(4, 12), slice(4, 12))}),
        Track(5, {0: _mask(slice(20, 30), slice(40, 50))}),
    ]
    client = FakeAuditClient('{"missing": ["laptop"], "uncertain": []}')
    cache = ResponseCache(tmp_path / "cache")
    missing, uncertain, images = audit_missing_nouns(
        world, tmp_path, kept, (0, 1, 2), _acfg(), (), client, cache, tmp_path / "work"
    )
    assert missing == ["laptop"]
    assert uncertain == []
    assert len(images) == 2
    assert client.calls == 1


def test_audit_skips_frames_without_masks(tmp_path: Path) -> None:
    world = _world(tmp_path)
    kept = [Track(2, {2: _mask(slice(4, 12), slice(4, 12))})]
    client = FakeAuditClient('{"missing": [], "uncertain": []}')
    cache = ResponseCache(tmp_path / "cache")
    missing, _, images = audit_missing_nouns(
        world, tmp_path, kept, (0, 1, 2), _acfg(), (), client, cache, tmp_path / "work"
    )
    assert missing == []
    assert [p.name for p in images] == ["audit_00002.png"]


def test_audit_without_client_names_nothing(tmp_path: Path) -> None:
    world = _world(tmp_path)
    kept = [Track(2, {0: _mask(slice(4, 12), slice(4, 12))})]
    cache = ResponseCache(tmp_path / "cache")
    assert audit_missing_nouns(
        world, tmp_path, kept, (0,), _acfg(), (), None, cache, tmp_path / "work"
    ) == ([], [], [])


def test_audit_answer_is_cached(tmp_path: Path) -> None:
    world = _world(tmp_path)
    kept = [Track(2, {0: _mask(slice(4, 12), slice(4, 12))})]
    client = FakeAuditClient('{"missing": ["laptop"], "uncertain": []}')
    cache = ResponseCache(tmp_path / "cache")
    cfg = _acfg()
    first, _, _ = audit_missing_nouns(
        world, tmp_path, kept, (0,), cfg, (), client, cache, tmp_path / "w"
    )
    second, _, _ = audit_missing_nouns(
        world, tmp_path, kept, (0,), cfg, (), client, cache, tmp_path / "w"
    )
    assert (first, second) == (["laptop"], ["laptop"])
    assert client.calls == 1


def test_audit_config_defaults_off() -> None:
    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    audit = cfg["segmentation"]["tracking"]["gap_fill"]["overlay_audit"]
    assert set(audit) >= {"enabled", "n_views", "save_video"}


def test_tracks_video_writes_every_frame(tmp_path: Path) -> None:
    world = _world(tmp_path, n=2)
    kept = [Track(0, {0: _mask(slice(4, 12), slice(4, 12))})]
    dropped = [Track(1, {1: _mask(slice(20, 30), slice(40, 50))})]
    out = tmp_path / "tracks_preview.mp4"
    frames = render_tracks_video(
        world, tmp_path, [*kept, *dropped], {0}, {0: "kept", 1: "no voxel"}, out, fps=5.0
    )
    assert frames == 2
    assert out.is_file() and out.stat().st_size > 0


def test_track_colors_are_stable() -> None:
    assert track_colors([3, 1, 2]) == track_colors([1, 2, 3])
    assert len(set(track_colors([1, 2, 3]).values())) == 3
