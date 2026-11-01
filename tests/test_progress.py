from __future__ import annotations

import io
import json
import time
from pathlib import Path

from video2sim.progress import (
    ProgressReporter,
    banner,
    format_clock,
    format_seconds,
    summarize_stage,
)
from video2sim.runner import StageEvent


class Tty(io.StringIO):
    def isatty(self) -> bool:
        return True


def event(kind: str, seconds: float | None = None, message: str | None = None) -> StageEvent:
    return StageEvent(kind, "reconstruction", 2, 6, seconds, message)  # type: ignore[arg-type]


def test_plain_stream_gets_one_line_per_event_and_no_control_characters() -> None:
    out = io.StringIO()
    reporter = ProgressReporter(out)
    for e in (
        StageEvent("cached", "capture", 1, 6),
        event("started"),
        event("done", 94.2),
        StageEvent("failed", "shape", 4, 6, 3.0, "RuntimeError: boom"),
    ):
        reporter(e)
    reporter.close()
    assert out.getvalue().splitlines() == [
        "[1/6] capture  cached",
        "[2/6] reconstruction  running",
        "[2/6] reconstruction  done in 94 s",
        "[4/6] shape  failed after 3 s: RuntimeError: boom",
    ]
    assert "\r" not in out.getvalue() and "\033" not in out.getvalue()


def test_terminal_shows_a_spinner_that_the_result_line_replaces() -> None:
    out = Tty()
    reporter = ProgressReporter(out)
    reporter(event("started"))
    time.sleep(0.25)
    reporter(event("done", 2.0))
    reporter.close()
    text = out.getvalue()
    assert "[2/6] reconstruction  0:00" in text  # at least one spinner frame was drawn
    assert text.count("done in 2 s") == 1
    assert text.index("0:00") < text.index("done in")
    assert text.count("\n") == 1  # only the result line ends in a newline
    frame = text.split("\033[2K")[1]
    assert frame[1:].startswith(" [2/6]")  # a spinner frame, a space, then the label
    assert "\033[2K  [2/6] reconstruction  done in 2 s\n" in text  # label in the same column


def test_a_second_started_event_stops_the_first_spinner() -> None:
    reporter = ProgressReporter(Tty())
    reporter(event("started"))
    reporter(event("started"))
    reporter.close()
    assert reporter._thread is None


def test_time_formats() -> None:
    assert format_seconds(94.2) == "94 s"
    assert format_seconds(192) == "3 min 12 s"
    assert format_clock(75.9) == "1:15"


def test_verbose_is_plain_even_on_a_terminal_and_adds_run_id_and_keys() -> None:
    out = Tty()
    reporter = ProgressReporter(out, verbose=True)
    reporter(StageEvent("cached", "capture", 1, 6, run_id="r1", detail="key 3fa9c1d2"))
    reporter(StageEvent("started", "reconstruction", 2, 6, run_id="r1", detail="key 77ab01ce"))
    reporter(StageEvent("done", "reconstruction", 2, 6, 1.0, run_id="r1", detail="output 9b1e77aa"))
    reporter.close()
    assert out.getvalue().splitlines() == [
        "run r1",
        "[1/6] capture  cached  (key 3fa9c1d2)",
        "[2/6] reconstruction  running  (key 77ab01ce)",
        "[2/6] reconstruction  done in 1 s  (output 9b1e77aa)",
    ]


def test_without_verbose_the_details_are_left_out() -> None:
    out = io.StringIO()
    ProgressReporter(out)(StageEvent("cached", "capture", 1, 6, run_id="r1", detail="key 3fa9c1d2"))
    assert out.getvalue() == "[1/6] capture  cached\n"


class _Utf8(Tty):
    encoding = "utf-8"


class _Ascii(Tty):
    encoding = "ascii"


def test_banner_is_only_for_terminals() -> None:
    assert banner(io.StringIO()) == ""
    assert "\033[48;2;" in banner(_Utf8(), {"COLORTERM": "truecolor"})
    assert "\033[48;5;" in banner(_Utf8(), {})
    assert "\033" not in banner(_Utf8(), {"NO_COLOR": "1"})
    assert banner(_Ascii(), {}) == "video2sim\n\n"


def _write_stage(
    root: Path, run: str, stage: str, payload: object, sidecars: dict[str, object] | None = None
) -> None:
    stage_dir = root / run / stage
    stage_dir.mkdir(parents=True)
    (stage_dir / "output.json").write_text(json.dumps(payload), encoding="utf-8")
    for name, data in (sidecars or {}).items():
        (stage_dir / name).write_text(json.dumps(data), encoding="utf-8")


def test_done_event_reports_stage_summary_by_default(tmp_path: Path) -> None:
    _write_stage(
        tmp_path,
        "r1",
        "capture",
        {"accepted": True, "frame_paths": ["a", "b", "c"], "fps": 5.0},
    )
    out = io.StringIO()
    reporter = ProgressReporter(out, runs_root=tmp_path)
    reporter(StageEvent("done", "capture", 1, 6, 2.0, run_id="r1"))
    reporter.close()
    assert out.getvalue().splitlines() == [
        "[1/6] capture  done in 2 s",
        "3 frames at 5.0 fps",
    ]


def test_cached_event_reports_summary_and_missing_output_stays_quiet(tmp_path: Path) -> None:
    _write_stage(
        tmp_path,
        "r1",
        "reconstruction",
        {
            "frames": [{}, {}],
            "registered_fraction": 1.0,
            "mean_reprojection_error_px": 0.5,
            "scale_source": "marker",
            "flags": [],
        },
    )
    out = io.StringIO()
    reporter = ProgressReporter(out, runs_root=tmp_path)
    reporter(StageEvent("cached", "reconstruction", 2, 6, run_id="r1"))
    reporter(StageEvent("done", "shape", 4, 6, 3.0, run_id="r1"))  # no shape folder
    reporter.close()
    assert out.getvalue().splitlines() == [
        "[2/6] reconstruction  cached",
        "2 frames, 100% registered, reprojection 0.50 px, scale from marker",
        "[4/6] shape  done in 3 s",
    ]


def test_segmentation_reports_router_path_and_flags(tmp_path: Path) -> None:
    _write_stage(
        tmp_path,
        "r1",
        "segmentation",
        {
            "objects": [
                {"object_id": "obj_000", "flags": ["occluded"]},
                {"object_id": "obj_001", "flags": ["occluded", "small"]},
            ],
            "flags": ["possible missing object: 3 objects listed but 2 tracks"],
        },
        {
            "route.json": {
                "surface": "table",
                "prompt": "objects on a table",
                "objects": ["box", "box", "cube"],
                "reason": "VLM surface table (0.99)",
                "fell_back": False,
                "tracks_raw": 2,
                "tracks_repaired": 2,
                "splits": [],
                "merges": [[0, 1]],
            }
        },
    )
    assert summarize_stage(tmp_path, "r1", "segmentation") == [
        "route VLM surface table (0.99): 3 listed, 2 tracks (1 merges, 0 splits)",
        "flags: possible missing object: 3 objects listed but 2 tracks; occluded (2), small (1)",
    ]


def test_physics_reports_mass_sources_and_worst_drift(tmp_path: Path) -> None:
    _write_stage(
        tmp_path,
        "r1",
        "physics",
        {
            "assets": [
                {"object_id": "obj_000", "checks": {"watertight": True}},
                {"object_id": "obj_001", "checks": {}},
            ]
        },
        {
            "report.json": {
                "obj_000": {"mass_source": "geometry"},
                "obj_001": {"mass_source": "model"},
            },
            "settling.json": {
                "nominal": {
                    "stable": True,
                    "bodies": [
                        {"object_id": "obj_000", "max_drift_m": 0.0044},
                        {"object_id": "obj_001", "max_drift_m": 0.001},
                    ],
                }
            },
        },
    )
    assert summarize_stage(tmp_path, "r1", "physics") == [
        "2 assets, 1 valid; mass from geometry x1, model x1",
        "worst drift 4.4 mm obj_000",
    ]


def test_shape_export_and_rejected_capture_summaries(tmp_path: Path) -> None:
    _write_stage(
        tmp_path,
        "r1",
        "shape",
        {
            "objects": [
                {"object_id": "obj_000", "geometry_source": "sam3d", "fit_score": 0.97},
                {"object_id": "obj_001", "geometry_source": "proxy_hull", "fit_score": None},
                {"object_id": "obj_002", "geometry_source": "sam3d", "fit_score": 0.62},
            ]
        },
    )
    assert summarize_stage(tmp_path, "r1", "shape") == [
        "2 modeled, 1 proxy; worst fit 0.62 obj_002"
    ]
    _write_stage(
        tmp_path,
        "r1",
        "export",
        {
            "urdf_paths": {"obj_000": "urdf/obj_000.urdf"},
            "skipped": [{"object_id": "obj_001", "reasons": ["collision_loads failed"]}],
        },
    )
    assert summarize_stage(tmp_path, "r1", "export") == [
        "1 objects exported; 1 skipped: obj_001 (collision_loads failed)"
    ]
    _write_stage(
        tmp_path, "r2", "capture", {"accepted": False, "reasons": ["too blurred: 42% over 30%"]}
    )
    assert summarize_stage(tmp_path, "r2", "capture") == ["rejected: too blurred: 42% over 30%"]


def test_unreadable_or_foreign_outputs_give_no_lines(tmp_path: Path) -> None:
    (tmp_path / "r1" / "shape").mkdir(parents=True)
    (tmp_path / "r1" / "shape" / "output.json").write_text("not json", encoding="utf-8")
    assert summarize_stage(tmp_path, "r1", "shape") == []
    _write_stage(tmp_path, "r1", "capture", [1, 2])
    assert summarize_stage(tmp_path, "r1", "capture") == []
    assert summarize_stage(tmp_path, "r1", "unknown_stage") == []
    assert summarize_stage(tmp_path, "nope", "capture") == []


def test_dry_run_reports_every_stage_without_waiting() -> None:
    from video2sim.dry_run import simulate
    from video2sim.schemas import STAGE_ORDER

    out = io.StringIO()
    simulate(ProgressReporter(out, interactive=False), stage_s=0.0, sleep=lambda _: None)
    lines = out.getvalue().splitlines()
    assert len(lines) == 2 * len(STAGE_ORDER) - 1 and "cached" in lines[0]
    assert sum("done" in line for line in lines) == len(STAGE_ORDER) - 1
