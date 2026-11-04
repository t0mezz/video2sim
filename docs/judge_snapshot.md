# Judge snapshot and `video2sim replay`

A way to see Video2Sim work with no API key, model, GPU or Modal account.

## What a replay is, and is not

`video2sim replay SCENE --out DIR` shows a **recorded** run of a benchmark scene. It checks the
stored archive against `docs/judge_snapshot/index.json`, reports the six stages with the times
they took when they really ran (the wait on screen is compressed to 5%, at most 1.5 s per stage),
and copies the scene package (`scene.usd`, `urdf/`, `qa_report.html`, `qa_report.json`, the
manifest, and `demo/pick_place.mp4` when recorded) to `DIR`.

It does **not** recompute anything, so it shows the pipeline's output, not that the code can
reproduce it from the video. To run the pipeline itself, use `video2sim run` with keys
(README, "Make a scene"). Replay builds no model client and reads no key, and a test checks that
it does not import the model adapters.

The input videos are not in the snapshot (benchmark videos are private, D-004). `--video FILE`
selects a scene by the SHA-256 of a video you hold; the hashes are in the lock file.

## Use

```bash
uv run video2sim replay                      # list scenes
uv run video2sim replay s01 --out scene/     # downloads once to data/judge_snapshot/
uv run video2sim replay s01 --out scene/ --archive path/to/s01.zip   # no network at all
```

Open `scene/qa_report.html` for the report and `scene/scene.usd` in Isaac Sim or a USD viewer.

## Integrity

The lock file holds, per scene, the video hash, the archive URL and SHA-256, and the SHA-256 of
`export/`. Replay refuses an archive or an unpacked `export/` that does not match, and an archive
with a path outside its folder.

## Maintainers: build and publish

```bash
python scripts/build_judge_snapshot.py runs_dev s01 s02 ... \
    --release-url https://github.com/OWNER/REPO/releases/download/judge-snapshot-v1
```

This packs each finished run (manifest, six stage `output.json`, `export/`, `demo/`) into
`data/judge_snapshot/archives/<scene>.zip` and rewrites the lock file; packing the same run twice
gives the same hash. It uploads nothing. Attach the zips to the release by hand, commit the lock
file, then check a clean checkout with `video2sim replay SCENE --out X`. Rebuild and re-publish
when a run changes; the lock file is the only thing in git.
