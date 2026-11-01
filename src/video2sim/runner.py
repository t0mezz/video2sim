"""Stage runner: run folder, manifest and hash-based caching (task P0.5).

One run is one folder, ``runs/<run_id>/``, holding ``manifest.json`` and one subfolder per
stage. A stage is skipped when its cache key matches the key recorded in the manifest.

The cache key is the SHA-256 of the stage's own config section, the base seed, the hash of
the stage's source file, the ``output_sha256`` of every declared dependency (the video hash
for a stage without dependencies) and the bytes of its ``extra_inputs``.

Stages write into ``<stage>.tmp/`` and the runner renames it to ``<stage>/`` only after the
stage succeeded, so a crash never leaves a folder that looks complete. A stage's output must
not contain wall-clock times, absolute paths or random ids, or its ``output_sha256`` changes
on every run and the stages after it never hit the cache.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import shutil
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol, TypeVar

from pydantic import BaseModel

from video2sim.schemas import STAGE_ORDER, RunManifest, StageName, StageRecord

VERBOSE_ENV = "VIDEO2SIM_VERBOSE"
OUTPUT_FILE = "output.json"
MANIFEST_FILE = "manifest.json"

_T = TypeVar("_T", bound=BaseModel)


class RunnerError(Exception):
    """The runner was set up or called wrongly."""


class StageContractError(Exception):
    """A stage broke the pipeline contract, for example by returning the wrong type."""


class StageFailed(Exception):
    """A stage raised. The manifest on this exception records the failure."""

    def __init__(self, stage: StageName, manifest: RunManifest) -> None:
        record = manifest.stages[stage]
        super().__init__(f"stage {stage!r} failed: {record.message}")
        self.stage = stage
        self.manifest = manifest


@dataclass(frozen=True)
class StageEvent:
    """What the runner tells a listener: a stage is cached, starts, finishes or fails.

    ``index`` counts from 1 over the stages this runner has. ``seconds`` is set on ``done`` and
    ``failed``, ``message`` on ``failed``. ``run_id`` and ``detail`` are for verbose output:
    ``detail`` is the short cache key, and on ``done`` the short hash of the stage's output.
    """

    kind: Literal["cached", "started", "done", "failed"]
    stage: StageName
    index: int
    total: int
    seconds: float | None = None
    message: str | None = None
    run_id: str | None = None
    detail: str | None = None


EventHandler = Callable[[StageEvent], None]


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def canonical_sha256(value: object) -> str:
    """SHA-256 of the canonical JSON of ``value`` (sorted keys, no whitespace)."""
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_dir(path: Path) -> str:
    """Hash of the sorted relative paths and the streamed bytes of every file below ``path``."""
    digest = hashlib.sha256()
    for file in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(file.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        with file.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def native_logs_wanted() -> bool:
    """True under ``video2sim run --verbose``: stages then leave the logs of native libraries
    (COLMAP, CoACD) on instead of silencing them."""
    return bool(os.environ.get(VERBOSE_ENV))


def seed_for(base_seed: int, stage: StageName) -> int:
    """Per-stage seed, so that stages do not share a random stream."""
    return int(hashlib.sha256(f"{base_seed}:{stage}".encode()).hexdigest()[:8], 16)


def _code_version() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


# ---------------------------------------------------------------------------
# Stage interface
# ---------------------------------------------------------------------------


class Stage(Protocol):
    """What the runner needs to know about a stage."""

    @property
    def name(self) -> StageName: ...

    @property
    def output_type(self) -> type[BaseModel]: ...

    @property
    def dependencies(self) -> tuple[StageName, ...]:
        """Earlier stages whose outputs this stage loads."""
        ...

    @property
    def extra_inputs(self) -> tuple[Path, ...]:
        """Files outside the config that change the result, for example the material table."""
        ...

    def run(self, ctx: StageContext) -> BaseModel: ...


@dataclass
class StageContext:
    """What a stage may touch: its own output folder and its declared dependencies."""

    stage: StageName
    run_dir: Path
    video: Path
    out_dir: Path
    config: Mapping[str, Any]
    seed: int
    dependencies: tuple[StageName, ...]
    output_types: Mapping[StageName, type[BaseModel]]
    checkpoints: dict[str, str] = field(default_factory=dict)
    # Top-level config sections the stage declared in ``shared_config``, such as ``vlm``
    shared: Mapping[str, Any] = field(default_factory=dict)

    def load(self, stage: StageName, model: type[_T]) -> _T:
        """Typed output of a declared dependency. Its files are under ``run_dir / stage``."""
        if stage not in self.dependencies:
            raise StageContractError(f"{self.stage} loaded {stage}, which it did not declare")
        registered = self.output_types[stage]
        if not issubclass(registered, model):
            raise StageContractError(
                f"{stage} produces {registered.__name__}, not {model.__name__}"
            )
        text = (self.run_dir / stage / OUTPUT_FILE).read_text()
        return model.model_validate_json(text)

    def record_checkpoint(self, model: str, checkpoint_id: str) -> None:
        self.checkpoints[model] = checkpoint_id


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class Runner:
    def __init__(
        self,
        runs_root: Path,
        stages: Sequence[Stage],
        config: Mapping[str, Any],
        base_dir: Path | None = None,
        on_event: EventHandler | None = None,
    ) -> None:
        """``base_dir`` is where relative ``extra_inputs`` are resolved (default: cwd).
        ``on_event`` hears about every stage; it is how a command line shows progress."""
        names = [stage.name for stage in stages]
        order = [name for name in STAGE_ORDER if name in names]
        if names != order or len(set(names)) != len(names):
            raise RunnerError(f"stages must be unique and in pipeline order, got {names}")
        for stage in stages:
            for dep in stage.dependencies:
                if dep not in names[: names.index(stage.name)]:
                    raise RunnerError(
                        f"{stage.name} depends on {dep}, which does not run before it"
                    )
        self.runs_root = runs_root
        self.stages = list(stages)
        self.config = config
        self.base_dir = base_dir or Path.cwd()
        self._on_event = on_event
        self._run_id: str | None = None
        self._output_types: dict[StageName, type[BaseModel]] = {
            stage.name: stage.output_type for stage in stages
        }

    # -- public --------------------------------------------------------------------------

    def run(
        self,
        video: Path,
        *,
        run_id: str | None = None,
        force_from: StageName | None = None,
    ) -> RunManifest:
        """Run every stage that is not cached. Passing an existing ``run_id`` resumes it.

        ``force_from`` re-runs that stage and every stage after it.
        Raises ``StageFailed`` (carrying the manifest) when a stage raises.
        """
        video_sha = sha256_file(video)
        if run_id is None:
            run_id = self._new_run_id(video_sha)
        if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
            raise RunnerError(f"run_id must be a plain folder name, got {run_id!r}")
        run_dir = self.runs_root / run_id
        self._run_id = run_id
        manifest = self._open(run_dir, run_id, video_sha)
        force_index = STAGE_ORDER.index(force_from) if force_from else None

        for stage in self.stages:
            for dep in stage.dependencies:
                if manifest.stages[dep].status != "done":
                    raise RunnerError(f"{stage.name} needs {dep}, which is not done")
            key = self._cache_key(stage, manifest, video_sha)
            record = manifest.stages[stage.name]
            forced = force_index is not None and STAGE_ORDER.index(stage.name) >= force_index
            if (
                not forced
                and record.status == "done"
                and record.input_sha256 == key
                and (run_dir / stage.name).is_dir()
            ):
                self._emit("cached", stage.name, detail=f"key {key[:8]}")
                continue
            manifest = self._execute(stage, run_dir, video, manifest, key)
        return manifest

    # -- internals -----------------------------------------------------------------------

    def _emit(
        self,
        kind: Literal["cached", "started", "done", "failed"],
        stage: StageName,
        seconds: float | None = None,
        message: str | None = None,
        detail: str | None = None,
    ) -> None:
        if self._on_event is None:
            return
        names = [s.name for s in self.stages]
        self._on_event(
            StageEvent(
                kind,
                stage,
                names.index(stage) + 1,
                len(names),
                seconds,
                message,
                self._run_id,
                detail,
            )
        )

    @staticmethod
    def _new_run_id(video_sha: str) -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"{stamp}_{video_sha[:8]}"

    def _open(self, run_dir: Path, run_id: str, video_sha: str) -> RunManifest:
        config_sha = canonical_sha256(self.config)
        seed = int(self.config.get("seed", 0))
        path = run_dir / MANIFEST_FILE
        if path.exists():
            manifest = RunManifest.model_validate_json(path.read_text())
            if manifest.input_video_sha256 != video_sha:
                raise RunnerError(f"run {run_id} was made from a different video")
            manifest = manifest.model_copy(update={"config_sha256": config_sha, "seed": seed})
        else:
            run_dir.mkdir(parents=True, exist_ok=True)
            manifest = RunManifest(
                run_id=run_id,
                created_utc=datetime.now(timezone.utc),
                input_video_sha256=video_sha,
                config_sha256=config_sha,
                code_version=_code_version(),
                seed=seed,
            )
        self._write_manifest(run_dir, manifest)
        return manifest

    @staticmethod
    def _write_manifest(run_dir: Path, manifest: RunManifest) -> None:
        tmp = run_dir / f"{MANIFEST_FILE}.tmp"
        tmp.write_text(manifest.model_dump_json(indent=2))
        os.replace(tmp, run_dir / MANIFEST_FILE)

    def _shared(self, stage: Stage) -> dict[str, Any]:
        """The top-level sections a stage lists in an optional ``shared_config`` attribute."""
        return {k: self.config.get(k, {}) for k in getattr(stage, "shared_config", ())}

    def _cache_key(self, stage: Stage, manifest: RunManifest, video_sha: str) -> str:
        deps = {dep: manifest.stages[dep].output_sha256 for dep in stage.dependencies}
        extra: dict[str, str] = {}
        for path in stage.extra_inputs:
            resolved = path if path.is_absolute() else self.base_dir / path
            extra[path.as_posix()] = sha256_file(resolved)
        source = inspect.getsourcefile(type(stage))
        payload = {
            "stage": stage.name,
            "config": self.config.get(stage.name, {}),
            "seed": manifest.seed,
            "source": sha256_file(Path(source)) if source else None,
            "deps": deps,
            "video": None if stage.dependencies else video_sha,
            "extra": extra,
        }
        if shared := self._shared(stage):  # absent when empty, so earlier keys stay valid
            payload["shared"] = shared
        return canonical_sha256(payload)

    def _execute(
        self, stage: Stage, run_dir: Path, video: Path, manifest: RunManifest, key: str
    ) -> RunManifest:
        final = run_dir / stage.name
        tmp = run_dir / f"{stage.name}.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir()
        ctx = StageContext(
            stage=stage.name,
            run_dir=run_dir,
            video=video,
            out_dir=tmp,
            config=self.config.get(stage.name, {}),
            shared=self._shared(stage),
            seed=seed_for(manifest.seed, stage.name),
            dependencies=stage.dependencies,
            output_types=self._output_types,
        )
        start = time.perf_counter()
        self._emit("started", stage.name, detail=f"key {key[:8]}")
        try:
            output = stage.run(ctx)
            if not isinstance(output, stage.output_type):
                raise StageContractError(
                    f"{stage.name} returned {type(output).__name__}, "
                    f"expected {stage.output_type.__name__}"
                )
            (tmp / OUTPUT_FILE).write_text(output.model_dump_json(indent=2))
            _check_stays_inside(tmp, run_dir)
            output_sha = sha256_dir(tmp)
        except Exception as error:
            shutil.rmtree(tmp, ignore_errors=True)
            failed = StageRecord(
                status="failed",
                seconds=time.perf_counter() - start,
                input_sha256=key,
                message=f"{type(error).__name__}: {error}",
            )
            manifest = self._with_record(manifest, stage.name, failed)
            self._write_manifest(run_dir, manifest)
            self._emit("failed", stage.name, failed.seconds, failed.message)
            raise StageFailed(stage.name, manifest) from error

        shutil.rmtree(final, ignore_errors=True)
        tmp.rename(final)
        done = StageRecord(
            status="done",
            seconds=time.perf_counter() - start,
            input_sha256=key,
            output_sha256=output_sha,
        )
        manifest = self._with_record(manifest, stage.name, done, ctx.checkpoints)
        self._write_manifest(run_dir, manifest)
        self._emit("done", stage.name, done.seconds, detail=f"output {output_sha[:8]}")
        return manifest

    @staticmethod
    def _with_record(
        manifest: RunManifest,
        stage: StageName,
        record: StageRecord,
        checkpoints: Mapping[str, str] | None = None,
    ) -> RunManifest:
        update: dict[str, Any] = {"stages": {**manifest.stages, stage: record}}
        if checkpoints:
            update["checkpoints"] = {**manifest.checkpoints, **checkpoints}
        return manifest.model_copy(update=update)


def _check_stays_inside(folder: Path, run_dir: Path) -> None:
    """Reject symlinks that point out of the run folder."""
    root = run_dir.resolve()
    for path in folder.rglob("*"):
        if path.is_symlink() and not path.resolve().is_relative_to(root):
            raise StageContractError(f"{path.name} points outside the run folder")
