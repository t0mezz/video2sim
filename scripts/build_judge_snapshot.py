"""Pack finished runs into judge snapshot archives and write the lock file.

    python scripts/build_judge_snapshot.py RUNS_DIR SCENE_ID ... \\
        [--out data/judge_snapshot/archives] [--release-url https://github.com/OWNER/REPO/releases/download/TAG]

Reads ``RUNS_DIR/<scene>/`` (manifest, stage outputs, ``export/``), writes ``<scene>.zip`` and
merges the scene into ``docs/judge_snapshot/index.json``. It makes no hosted call and uploads
nothing: attaching the zips to a release is a human step (``docs/judge_snapshot.md``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from video2sim.replay import (
    DEFAULT_INDEX,
    SceneEntry,
    SnapshotError,
    load_index,
    pack,
    write_index,
)
from video2sim.runner import MANIFEST_FILE
from video2sim.schemas import RunManifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path)
    parser.add_argument("scenes", nargs="+")
    parser.add_argument("--out", type=Path, default=Path("data/judge_snapshot/archives"))
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--release-url", help="base URL the zips will be attached to")
    args = parser.parse_args()

    failed: list[str] = []
    entries = load_index(args.index) if args.index.exists() else {}
    for scene in args.scenes:
        run_dir = args.runs / scene
        try:
            manifest = RunManifest.model_validate_json(
                (run_dir / MANIFEST_FILE).read_text(encoding="utf-8")
            )
            archive = args.out / f"{scene}.zip"
            archive_sha, export_sha = pack(run_dir, archive)
        except (OSError, ValueError, SnapshotError) as error:
            print(f"{scene}: skipped, {error}", file=sys.stderr)
            failed.append(scene)
            continue
        url = f"{args.release_url.rstrip('/')}/{archive.name}" if args.release_url else None
        entries[scene] = SceneEntry(
            scene_id=scene,
            video_sha256=manifest.input_video_sha256,
            archive=archive.name,
            archive_sha256=archive_sha,
            export_sha256=export_sha,
            url=url,
            note=entries[scene].note if scene in entries else "",
        )
        print(f"{scene}: {archive} ({archive.stat().st_size / 1e6:.1f} MB)")
    write_index(args.index, entries)
    print(f"wrote {args.index}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
