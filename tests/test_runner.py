from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from video2sim.runner import (
    OUTPUT_FILE,
    Runner,
    RunnerError,
    StageContext,
    StageFailed,
    seed_for,
)
from video2sim.schemas import Model, RelPath, RunManifest, StageName


class Out(Model):
    value: str
    files: list[RelPath]


class FakeStage:
    """Output depends on its config 'value', its seed, its dependencies and extra inputs."""

    output_type = Out

    def __init__(
        self,
        name: StageName,
        dependencies: tuple[StageName, ...] = (),
        extra_inputs: tuple[Path, ...] = (),
    ) -> None:
        self.name = name
        self.dependencies = dependencies
        self.extra_inputs = extra_inputs
        self.calls = 0
        self.fail = False
        self.crash = False
        self.symlink_to: Path | None = None

    def run(self, ctx: StageContext) -> Out:
        self.calls += 1
        parts = [str(ctx.config.get("value", "")), str(ctx.seed)]
        parts += [ctx.load(dep, Out).value for dep in self.dependencies]
        parts += [path.read_text() for path in self.extra_inputs]
        value = "|".join(parts)
        (ctx.out_dir / "data.txt").write_text(value)
        if self.crash:
            raise KeyboardInterrupt  # not an Exception: simulates the process being killed
        if self.fail:
            raise ValueError("boom")
        if self.symlink_to is not None:
            (ctx.out_dir / "link").symlink_to(self.symlink_to)
        return Out(value=value, files=[f"{self.name}/data.txt"])


@pytest.fixture
def video(tmp_path: Path) -> Path:
    path = tmp_path / "video.mp4"
    path.write_bytes(b"not really a video")
    return path


@pytest.fixture
def stages() -> tuple[FakeStage, FakeStage, FakeStage]:
    return (
        FakeStage("capture"),
        FakeStage("reconstruction", dependencies=("capture",)),
        FakeStage("segmentation", dependencies=("reconstruction",)),
    )


def make_runner(
    tmp_path: Path, stages: tuple[FakeStage, ...], config: dict[str, Any] | None = None
) -> Runner:
    return Runner(tmp_path / "runs", stages, config or {"seed": 0}, base_dir=tmp_path)


def calls(stages: tuple[FakeStage, ...]) -> list[int]:
    return [stage.calls for stage in stages]


def read_manifest(tmp_path: Path, run_id: str) -> RunManifest:
    return RunManifest.model_validate_json(
        (tmp_path / "runs" / run_id / "manifest.json").read_text()
    )


def test_first_run_writes_manifest_and_stage_folders(
    tmp_path: Path, video: Path, stages: tuple[FakeStage, ...]
) -> None:
    manifest = make_runner(tmp_path, stages).run(video, run_id="r1")
    run_dir = tmp_path / "runs" / "r1"
    for stage in stages:
        assert (run_dir / stage.name / OUTPUT_FILE).is_file()
        record = manifest.stages[stage.name]
        assert record.status == "done"
        assert record.input_sha256 and record.output_sha256
    assert manifest.stages["shape"].status == "pending"
    assert read_manifest(tmp_path, "r1") == manifest
    assert not list(run_dir.glob("*.tmp"))


def test_second_run_is_a_full_cache_hit(
    tmp_path: Path, video: Path, stages: tuple[FakeStage, ...]
) -> None:
    runner = make_runner(tmp_path, stages)
    first = runner.run(video, run_id="r1")
    second = runner.run(video, run_id="r1")
    assert calls(stages) == [1, 1, 1]
    assert second.stages == first.stages


def test_default_run_id_has_timestamp_and_video_hash(
    tmp_path: Path, video: Path, stages: tuple[FakeStage, ...]
) -> None:
    manifest = make_runner(tmp_path, stages).run(video)
    assert manifest.run_id.endswith(manifest.input_video_sha256[:8])
    assert (tmp_path / "runs" / manifest.run_id).is_dir()


def test_config_change_reruns_that_stage_and_dependents_whose_input_changed(
    tmp_path: Path, video: Path, stages: tuple[FakeStage, ...]
) -> None:
    config: dict[str, Any] = {"seed": 0, "reconstruction": {"value": "a"}}
    make_runner(tmp_path, stages, config).run(video, run_id="r1")
    config["reconstruction"] = {"value": "b"}
    make_runner(tmp_path, stages, config).run(video, run_id="r1")
    assert calls(stages) == [1, 2, 2]  # capture is upstream and untouched


def test_unchanged_output_stops_the_cascade(
    tmp_path: Path, video: Path, stages: tuple[FakeStage, ...]
) -> None:
    config: dict[str, Any] = {"seed": 0, "reconstruction": {"value": "a", "note": 1}}
    make_runner(tmp_path, stages, config).run(video, run_id="r1")
    config["reconstruction"] = {"value": "a", "note": 2}  # in the key, not in the output
    make_runner(tmp_path, stages, config).run(video, run_id="r1")
    assert calls(stages) == [1, 2, 1]


def test_other_stages_config_does_not_invalidate(
    tmp_path: Path, video: Path, stages: tuple[FakeStage, ...]
) -> None:
    make_runner(tmp_path, stages, {"seed": 0, "shape": {"value": "a"}}).run(video, run_id="r1")
    make_runner(tmp_path, stages, {"seed": 0, "shape": {"value": "b"}}).run(video, run_id="r1")
    assert calls(stages) == [1, 1, 1]


def test_extra_input_change_reruns_only_its_stage(tmp_path: Path, video: Path) -> None:
    table = tmp_path / "table.yaml"
    table.write_text("wood: 700")
    stages = (FakeStage("capture"), FakeStage("reconstruction", extra_inputs=(table,)))
    runner = make_runner(tmp_path, stages)
    runner.run(video, run_id="r1")
    table.write_text("wood: 800")
    runner.run(video, run_id="r1")
    assert calls(stages) == [1, 2]


def test_a_new_seed_reruns_everything(
    tmp_path: Path, video: Path, stages: tuple[FakeStage, ...]
) -> None:
    make_runner(tmp_path, stages, {"seed": 0}).run(video, run_id="r1")
    make_runner(tmp_path, stages, {"seed": 1}).run(video, run_id="r1")
    assert calls(stages) == [2, 2, 2]


def test_failure_stops_the_run_and_the_next_run_resumes(
    tmp_path: Path, video: Path, stages: tuple[FakeStage, ...]
) -> None:
    stages[1].fail = True
    runner = make_runner(tmp_path, stages)
    with pytest.raises(StageFailed) as caught:
        runner.run(video, run_id="r1")
    assert caught.value.stage == "reconstruction"
    on_disk = read_manifest(tmp_path, "r1")
    assert on_disk.stages["capture"].status == "done"
    assert on_disk.stages["reconstruction"].status == "failed"
    assert "boom" in (on_disk.stages["reconstruction"].message or "")
    assert on_disk.stages["segmentation"].status == "pending"
    assert calls(stages) == [1, 1, 0]
    assert not (tmp_path / "runs" / "r1" / "reconstruction.tmp").exists()

    stages[1].fail = False
    runner.run(video, run_id="r1")
    assert calls(stages) == [1, 2, 1]


def test_a_killed_stage_leaves_no_finished_folder(
    tmp_path: Path, video: Path, stages: tuple[FakeStage, ...]
) -> None:
    stages[1].crash = True
    runner = make_runner(tmp_path, stages)
    with pytest.raises(KeyboardInterrupt):
        runner.run(video, run_id="r1")
    run_dir = tmp_path / "runs" / "r1"
    assert not (run_dir / "reconstruction").exists()
    assert (run_dir / "reconstruction.tmp").exists()
    assert read_manifest(tmp_path, "r1").stages["reconstruction"].status == "pending"

    stages[1].crash = False
    runner.run(video, run_id="r1")
    assert (run_dir / "reconstruction" / OUTPUT_FILE).is_file()
    assert not (run_dir / "reconstruction.tmp").exists()


def test_force_from_reruns_that_stage_and_everything_after(
    tmp_path: Path, video: Path, stages: tuple[FakeStage, ...]
) -> None:
    runner = make_runner(tmp_path, stages)
    runner.run(video, run_id="r1")
    runner.run(video, run_id="r1", force_from="reconstruction")
    assert calls(stages) == [1, 2, 2]


def test_a_symlink_out_of_the_run_folder_is_rejected(
    tmp_path: Path, video: Path, stages: tuple[FakeStage, ...]
) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    stages[0].symlink_to = outside
    with pytest.raises(StageFailed, match="outside the run folder"):
        make_runner(tmp_path, stages).run(video, run_id="r1")
    assert not (tmp_path / "runs" / "r1" / "capture").exists()


def test_same_inputs_give_the_same_output_hash_in_two_runs(
    tmp_path: Path, video: Path, stages: tuple[FakeStage, ...]
) -> None:
    runner = make_runner(tmp_path, stages)
    a = runner.run(video, run_id="a")
    b = runner.run(video, run_id="b")
    assert [r.output_sha256 for r in a.stages.values()] == [
        r.output_sha256 for r in b.stages.values()
    ]
    assert calls(stages) == [2, 2, 2]  # separate runs do not share a cache


def test_resuming_with_a_different_video_is_an_error(
    tmp_path: Path, video: Path, stages: tuple[FakeStage, ...]
) -> None:
    runner = make_runner(tmp_path, stages)
    runner.run(video, run_id="r1")
    other = tmp_path / "other.mp4"
    other.write_bytes(b"another video")
    with pytest.raises(RunnerError, match="different video"):
        runner.run(other, run_id="r1")


def test_loading_an_undeclared_dependency_fails_the_stage(tmp_path: Path, video: Path) -> None:
    class Loader(FakeStage):
        def run(self, ctx: StageContext) -> Out:
            ctx.load("capture", Out)
            return super().run(ctx)

    # Loader declares no dependencies but loads capture
    stages = (FakeStage("capture"), Loader("reconstruction"))
    with pytest.raises(StageFailed, match="did not declare"):
        make_runner(tmp_path, stages).run(video, run_id="r1")


def test_checkpoints_are_recorded_in_the_manifest(tmp_path: Path, video: Path) -> None:
    class Model(FakeStage):
        def run(self, ctx: StageContext) -> Out:
            ctx.record_checkpoint("sam", "sam-3.1-abc")
            return super().run(ctx)

    manifest = make_runner(tmp_path, (Model("capture"),)).run(video, run_id="r1")
    assert manifest.checkpoints == {"sam": "sam-3.1-abc"}


def test_stages_must_be_unique_and_in_order(tmp_path: Path) -> None:
    with pytest.raises(RunnerError, match="pipeline order"):
        make_runner(tmp_path, (FakeStage("shape"), FakeStage("capture")))
    with pytest.raises(RunnerError, match="does not run before"):
        make_runner(tmp_path, (FakeStage("capture", dependencies=("shape",)),))


@pytest.mark.parametrize("bad", ["", "..", "a/b"])
def test_run_id_must_be_a_plain_folder_name(
    tmp_path: Path, video: Path, stages: tuple[FakeStage, ...], bad: str
) -> None:
    with pytest.raises(RunnerError, match="run_id"):
        make_runner(tmp_path, stages).run(video, run_id=bad)


def test_seeds_differ_per_stage_and_are_stable() -> None:
    assert seed_for(0, "capture") != seed_for(0, "shape")
    assert seed_for(0, "capture") == seed_for(0, "capture")
    assert seed_for(0, "capture") != seed_for(1, "capture")
