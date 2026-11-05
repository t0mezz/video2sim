"""``video2sim replay``: show a recorded run of a scene, with no model, key or GPU.

A judge snapshot holds, per scene, a zip of the run's ``manifest.json``, the six stage
``output.json`` files and ``export/`` (plus ``demo/`` when the pick-and-place was recorded).
``docs/judge_snapshot/index.json`` is the lock file: archive URL, archive SHA-256 and the hash
of ``export/``. Replay checks those hashes, unpacks the archive and copies ``export/`` to the
output folder. It never runs a stage and never recomputes anything (D1 in ``PLAN.md``), so the
cache keys of the runner play no part and editing a stage file cannot break it.

Standard library and ``video2sim.runner`` / ``schemas`` only: it must not import a model
adapter, which a test checks.
"""

from __future__ import annotations

import json
import shutil
import time
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from video2sim.runner import MANIFEST_FILE, OUTPUT_FILE, StageEvent, sha256_dir, sha256_file
from video2sim.schemas import STAGE_ORDER, RunManifest

INDEX_VERSION = 1
DEFAULT_INDEX = Path("docs/judge_snapshot/index.json")
DEFAULT_CACHE = Path("data/judge_snapshot")
# Recorded stage times are shown compressed: a demo should not take the 90 s of the original.
# The factor and the cap only change how long the display waits, never a result.
TIME_SCALE = 0.05
MAX_STAGE_WAIT_S = 1.5


class SnapshotError(Exception):
    """The snapshot is missing, damaged or does not match its lock file."""


@dataclass(frozen=True)
class SceneEntry:
    scene_id: str
    video_sha256: str
    archive: str  # file name inside a release, or a full URL
    archive_sha256: str
    export_sha256: str
    url: str | None = None
    note: str = ""


def load_index(path: Path) -> dict[str, SceneEntry]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise SnapshotError(f"cannot read the lock file {path}: {error}") from error
    if raw.get("version") != INDEX_VERSION:
        raise SnapshotError(f"{path} has version {raw.get('version')!r}, want {INDEX_VERSION}")
    try:
        return {
            scene_id: SceneEntry(scene_id=scene_id, **fields)
            for scene_id, fields in sorted(raw["scenes"].items())
        }
    except (KeyError, TypeError) as error:
        raise SnapshotError(f"{path} is malformed: {error}") from error


def write_index(path: Path, entries: dict[str, SceneEntry]) -> None:
    scenes = {
        e.scene_id: {
            "video_sha256": e.video_sha256,
            "archive": e.archive,
            "archive_sha256": e.archive_sha256,
            "export_sha256": e.export_sha256,
            "url": e.url,
            "note": e.note,
        }
        for e in sorted(entries.values(), key=lambda e: e.scene_id)
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"version": INDEX_VERSION, "scenes": scenes}, indent=2) + "\n", encoding="utf-8"
    )


def pick_scene(entries: dict[str, SceneEntry], scene: str | None, video: Path | None) -> SceneEntry:
    """The scene named by id, or the one whose video hash matches ``video``."""
    if scene is not None:
        if scene not in entries:
            raise SnapshotError(f"no scene {scene!r}; available: {', '.join(entries) or 'none'}")
        return entries[scene]
    if video is not None:
        digest = sha256_file(video)
        for entry in entries.values():
            if entry.video_sha256 == digest:
                return entry
        raise SnapshotError(
            f"{video} is not one of the snapshot videos; replay shows recorded runs only. "
            f"Available scenes: {', '.join(entries) or 'none'}"
        )
    raise SnapshotError("name a scene or pass --video")


def fetch_archive(entry: SceneEntry, cache: Path, download: Callable[[str, Path], None]) -> Path:
    """The archive in ``cache``, downloaded first if it is not there. Checked against the lock."""
    target = cache / f"{entry.scene_id}.zip"
    if not (target.exists() and sha256_file(target) == entry.archive_sha256):
        if not entry.url:
            raise SnapshotError(f"{entry.scene_id} has no url in the lock file; pass --archive")
        cache.mkdir(parents=True, exist_ok=True)
        part = target.with_suffix(".part")
        download(entry.url, part)
        part.replace(target)
    return target


def download_url(url: str, target: Path) -> None:
    try:
        with urllib.request.urlopen(url, timeout=60) as response, target.open("wb") as out:
            shutil.copyfileobj(response, out)
    except OSError as error:
        raise SnapshotError(f"download of {url} failed: {error}") from error


def unpack(archive: Path, entry: SceneEntry, runs_root: Path) -> Path:
    """Check the archive hash, unpack to ``runs_root/<scene>`` and check the export hash."""
    if sha256_file(archive) != entry.archive_sha256:
        raise SnapshotError(f"{archive} does not match the archive hash in the lock file")
    run_dir = runs_root / entry.scene_id
    shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True)
    root = run_dir.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for name in bundle.namelist():  # zip-slip guard; the archive is ours but not trusted
            if not (run_dir / name).resolve().is_relative_to(root):
                raise SnapshotError(f"{archive} holds a path outside the run folder: {name}")
        bundle.extractall(run_dir)
    export = run_dir / "export"
    if not export.is_dir():
        raise SnapshotError(f"{archive} has no export/ folder")
    if sha256_dir(export) != entry.export_sha256:
        raise SnapshotError(f"export/ of {entry.scene_id} does not match the lock file")
    return run_dir


def replay(
    run_dir: Path,
    on_event: Callable[[StageEvent], None],
    sleep: Callable[[float], None] = time.sleep,
    instant: bool = False,
) -> RunManifest:
    """Report the recorded stages as if they ran. Nothing is computed."""
    manifest = RunManifest.model_validate_json((run_dir / MANIFEST_FILE).read_text("utf-8"))
    total = len(STAGE_ORDER)
    for index, stage in enumerate(STAGE_ORDER, start=1):
        record = manifest.stages[stage]
        if record.status != "done" or not (run_dir / stage / OUTPUT_FILE).is_file():
            raise SnapshotError(f"the recorded run has no finished {stage} stage")
        key = f"replay {(record.output_sha256 or '')[:8]}"
        on_event(StageEvent("started", stage, index, total, run_id=manifest.run_id, detail=key))
        recorded = record.seconds or 0.0
        wait = 0.0 if instant else min(recorded * TIME_SCALE, MAX_STAGE_WAIT_S)
        sleep(wait)
        on_event(
            StageEvent("done", stage, index, total, recorded, run_id=manifest.run_id, detail=key)
        )
    return manifest


def copy_package(run_dir: Path, out: Path) -> None:
    """The same layout ``video2sim run`` writes: ``export/`` plus the manifest."""
    shutil.copytree(run_dir / "export", out, dirs_exist_ok=True)
    shutil.copy2(run_dir / MANIFEST_FILE, out / MANIFEST_FILE)
    if (run_dir / "demo").is_dir():
        shutil.copytree(run_dir / "demo", out / "demo", dirs_exist_ok=True)


def pack(run_dir: Path, archive: Path) -> tuple[str, str]:
    """Write the snapshot zip of a finished run. Returns (archive SHA-256, export SHA-256)."""
    wanted: list[Path] = [run_dir / MANIFEST_FILE]
    wanted += [run_dir / stage / OUTPUT_FILE for stage in STAGE_ORDER]
    for folder in ("export", "demo"):
        if (run_dir / folder).is_dir():
            wanted += sorted(p for p in (run_dir / folder).rglob("*") if p.is_file())
    wanted = list(dict.fromkeys(wanted))  # export/output.json is both a stage output and export
    missing = [p for p in wanted if not p.is_file()]
    if missing:
        raise SnapshotError(f"{run_dir} is incomplete: {missing[0].relative_to(run_dir)}")
    archive.parent.mkdir(parents=True, exist_ok=True)
    # Fixed timestamps and order, so packing the same run twice gives the same archive hash.
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in wanted:
            info = zipfile.ZipInfo(path.relative_to(run_dir).as_posix(), (2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(info, path.read_bytes())
    return sha256_file(archive), sha256_dir(run_dir / "export")
