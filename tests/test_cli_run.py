"""``video2sim run`` with the pipeline replaced, so no stage or hosted model runs."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from video2sim import cli
from video2sim.runner import StageFailed
from video2sim.schemas import RunManifest


class FakeRunner:
    def __init__(self, manifest: Any, runs: Path, fail: bool) -> None:
        self.manifest, self.runs, self.fail = manifest, runs, fail

    def run(self, video: Path, run_id: str | None = None) -> Any:
        if self.fail:
            raise StageFailed("capture", self.manifest)
        folder = self.runs / self.manifest.run_id / "export"
        folder.mkdir(parents=True)
        (folder / "scene.usd").write_text("usd")
        (folder.parent / "manifest.json").write_text("{}")
        return SimpleNamespace(run_id=self.manifest.run_id)


def run_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail: bool) -> int:
    manifest = RunManifest(
        run_id="r1",
        created_utc="2026-10-31T00:00:00Z",
        input_video_sha256="a",
        config_sha256="b",
        code_version=None,
        seed=0,
    )
    runs = tmp_path / "runs"
    monkeypatch.setattr(
        "video2sim.pipeline.build_runner", lambda *a, **k: FakeRunner(manifest, runs, fail)
    )
    return cli.main(["run", "v.mp4", "--out", str(tmp_path / "scene"), "--runs", str(runs)])


def test_run_copies_the_package_to_out(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert run_cli(tmp_path, monkeypatch, fail=False) == 0
    assert (tmp_path / "scene" / "scene.usd").read_text() == "usd"
    assert (tmp_path / "scene" / "manifest.json").exists()


def test_a_failed_stage_exits_1_and_writes_no_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert run_cli(tmp_path, monkeypatch, fail=True) == 1
    assert not (tmp_path / "scene").exists()


def test_progress_goes_to_a_handler_unless_quiet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[Any] = []

    def build(*a: Any, on_event: Any = None) -> FakeRunner:
        seen.append(on_event)
        return FakeRunner(manifest, tmp_path / "runs", False)

    manifest = RunManifest(
        run_id="r1",
        created_utc="2026-10-31T00:00:00Z",
        input_video_sha256="a",
        config_sha256="b",
        code_version=None,
        seed=0,
    )
    monkeypatch.setattr("video2sim.pipeline.build_runner", build)
    base = ["run", "v.mp4", "--runs", str(tmp_path / "runs")]
    assert cli.main([*base, "--out", str(tmp_path / "a")]) == 0
    (tmp_path / "runs" / "r1").rename(tmp_path / "runs" / "r0")
    assert cli.main([*base, "--out", str(tmp_path / "b"), "--quiet"]) == 0
    assert callable(seen[0]) and seen[1] is None


def test_verbose_sets_the_native_log_switch_and_conflicts_with_quiet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("VIDEO2SIM_VERBOSE", raising=False)
    monkeypatch.setattr(
        "video2sim.pipeline.build_runner",
        lambda *a, **k: FakeRunner(
            RunManifest(
                run_id="r1",
                created_utc="2026-10-31T00:00:00Z",
                input_video_sha256="a",
                config_sha256="b",
                code_version=None,
                seed=0,
            ),
            tmp_path / "runs",
            False,
        ),
    )
    args = ["run", "v.mp4", "--out", str(tmp_path / "s"), "--runs", str(tmp_path / "runs")]
    assert cli.main([*args, "--verbose"]) == 0
    assert os.environ["VIDEO2SIM_VERBOSE"] == "1"
    monkeypatch.delenv("VIDEO2SIM_VERBOSE")
    with pytest.raises(SystemExit):
        cli.main([*args, "--verbose", "--quiet"])
