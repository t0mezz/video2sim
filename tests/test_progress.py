from __future__ import annotations

import io
import time

from video2sim.progress import ProgressReporter, format_clock, format_seconds
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
