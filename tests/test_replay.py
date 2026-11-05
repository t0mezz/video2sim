"""Judge snapshot replay on a tiny synthetic run: pack, lock, unpack, replay, tamper checks."""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from video2sim import cli, replay
from video2sim.runner import MANIFEST_FILE, OUTPUT_FILE, StageEvent, sha256_file
from video2sim.schemas import STAGE_ORDER, RunManifest, StageRecord


def make_run(root: Path, scene: str = "s99", video_sha: str = "ab" * 32) -> Path:
    run = root / scene
    stages = {
        s: StageRecord(status="done", seconds=2.0, input_sha256="k", output_sha256=f"{i:064x}")
        for i, s in enumerate(STAGE_ORDER)
    }
    manifest = RunManifest(
        run_id=scene,
        created_utc="2026-11-04T00:00:00Z",
        input_video_sha256=video_sha,
        config_sha256="c",
        code_version=None,
        seed=0,
        stages=stages,
    )
    run.mkdir(parents=True)
    (run / MANIFEST_FILE).write_text(manifest.model_dump_json())
    for s in STAGE_ORDER:
        (run / s).mkdir()
        (run / s / OUTPUT_FILE).write_text("{}")
    (run / "export" / "urdf").mkdir(parents=True)
    (run / "export" / "scene.usd").write_text("usd")
    (run / "export" / "urdf" / "obj_000.urdf").write_text("urdf")
    (run / "demo").mkdir()
    (run / "demo" / "pick_place.mp4").write_text("mp4")
    return run


@pytest.fixture
def snapshot(tmp_path: Path) -> tuple[Path, replay.SceneEntry, Path]:
    run = make_run(tmp_path / "runs")
    archive = tmp_path / "archives" / "s99.zip"
    archive_sha, export_sha = replay.pack(run, archive)
    entry = replay.SceneEntry("s99", "ab" * 32, "s99.zip", archive_sha, export_sha, note="demo")
    index = tmp_path / "index.json"
    replay.write_index(index, {"s99": entry})
    return index, entry, archive


def test_index_round_trip(snapshot: tuple[Path, replay.SceneEntry, Path]) -> None:
    index, entry, _ = snapshot
    assert replay.load_index(index) == {"s99": entry}


def test_pack_is_reproducible(tmp_path: Path) -> None:
    run = make_run(tmp_path / "runs")
    first = replay.pack(run, tmp_path / "a.zip")
    second = replay.pack(run, tmp_path / "b.zip")
    assert first == second


def test_pack_rejects_incomplete_run(tmp_path: Path) -> None:
    run = make_run(tmp_path / "runs")
    (run / "shape" / OUTPUT_FILE).unlink()
    with pytest.raises(replay.SnapshotError, match="shape"):
        replay.pack(run, tmp_path / "a.zip")


def test_pick_scene_by_id_and_video(
    snapshot: tuple[Path, replay.SceneEntry, Path], tmp_path: Path
) -> None:
    index, entry, _ = snapshot
    entries = replay.load_index(index)
    assert replay.pick_scene(entries, "s99", None) == entry
    with pytest.raises(replay.SnapshotError, match="available: s99"):
        replay.pick_scene(entries, "nope", None)
    video = tmp_path / "other.mp4"
    video.write_bytes(b"not a snapshot video")
    with pytest.raises(replay.SnapshotError, match="not one of the snapshot videos"):
        replay.pick_scene(entries, None, video)


def test_replay_reports_every_stage_without_waiting(
    snapshot: tuple[Path, replay.SceneEntry, Path], tmp_path: Path
) -> None:
    _, entry, archive = snapshot
    run_dir = replay.unpack(archive, entry, tmp_path / "cache")
    events: list[StageEvent] = []
    waits: list[float] = []
    replay.replay(run_dir, events.append, sleep=waits.append)
    assert [e.stage for e in events if e.kind == "done"] == list(STAGE_ORDER)
    assert waits == [pytest.approx(0.1)] * len(STAGE_ORDER)  # 2 s recorded at 5%
    waits.clear()
    replay.replay(run_dir, lambda e: None, sleep=waits.append, instant=True)
    assert set(waits) == {0.0}


def test_unpack_detects_tampering(
    snapshot: tuple[Path, replay.SceneEntry, Path], tmp_path: Path
) -> None:
    _, entry, archive = snapshot
    archive.write_bytes(archive.read_bytes() + b"x")
    with pytest.raises(replay.SnapshotError, match="archive hash"):
        replay.unpack(archive, entry, tmp_path / "cache")


def test_unpack_detects_wrong_export_hash(
    snapshot: tuple[Path, replay.SceneEntry, Path], tmp_path: Path
) -> None:
    _, entry, archive = snapshot
    wrong = replay.SceneEntry(**{**entry.__dict__, "export_sha256": "0" * 64})
    with pytest.raises(replay.SnapshotError, match="export/"):
        replay.unpack(archive, wrong, tmp_path / "cache")


def test_unpack_rejects_path_escape(tmp_path: Path) -> None:
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../escape.txt", "x")
    entry = replay.SceneEntry("s99", "a", "evil.zip", sha256_file(archive), "e")
    with pytest.raises(replay.SnapshotError, match="outside"):
        replay.unpack(archive, entry, tmp_path / "cache")
    assert not (tmp_path / "escape.txt").exists()


def test_fetch_downloads_once_and_needs_a_url(
    snapshot: tuple[Path, replay.SceneEntry, Path], tmp_path: Path
) -> None:
    _, entry, archive = snapshot
    calls: list[str] = []

    def fake_download(url: str, target: Path) -> None:
        calls.append(url)
        target.write_bytes(archive.read_bytes())

    with_url = replay.SceneEntry(**{**entry.__dict__, "url": "https://example.invalid/s99.zip"})
    cache = tmp_path / "cache"
    first = replay.fetch_archive(with_url, cache, fake_download)
    replay.fetch_archive(with_url, cache, fake_download)
    assert calls == ["https://example.invalid/s99.zip"]  # second call hit the verified cache
    assert first.read_bytes() == archive.read_bytes()
    with pytest.raises(replay.SnapshotError, match="no url"):
        replay.fetch_archive(entry, tmp_path / "other", fake_download)


def test_cli_replay_end_to_end(
    snapshot: tuple[Path, replay.SceneEntry, Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    index, _, archive = snapshot
    base = ["replay", "--index", str(index), "--cache", str(tmp_path / "cache")]
    assert cli.main(base) == 0
    assert "s99  demo" in capsys.readouterr().out
    out = tmp_path / "out"
    code = cli.main([*base, "s99", "--archive", str(archive), "--out", str(out), "--instant"])
    assert code == 0
    assert (out / "scene.usd").read_text() == "usd"
    assert (out / MANIFEST_FILE).exists()
    assert (out / "demo" / "pick_place.mp4").exists()
    assert cli.main([*base, "nope", "--out", str(out)]) == 1


def test_replay_never_imports_a_model_adapter() -> None:
    code = (
        "import sys, video2sim.replay, video2sim.cli;"
        "bad=[m for m in sys.modules if m.startswith('video2sim.models')];"
        "sys.exit(1 if bad else 0)"
    )
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
