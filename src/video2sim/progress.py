"""Progress lines for ``video2sim run``, written to stderr so stdout stays the final result.

``ProgressReporter`` is a runner ``on_event`` handler. On a terminal it draws a spinner with the
running stage and its elapsed time on one line and replaces it with a result line when the stage
ends. Result lines are indented by the width of the spinner and its space, so the stage labels
of past and running stages line up in one column. Anywhere else (a pipe, a log file, CI) it
writes one plain line per event and never uses carriage returns.

With ``verbose`` it always writes plain lines, so the logs of native libraries that are left on
do not tangle with a spinner, and adds the run id and the cache keys. Standard library only.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Mapping
from typing import TextIO

from video2sim.logo import render
from video2sim.runner import StageEvent

_BRAILLE = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_ASCII = "|/-\\"
_ERASE = "\r\033[2K"
_INDENT = "  "  # one spinner frame and a space
_TICK_S = 0.1


def banner(stream: TextIO, env: Mapping[str, str] | None = None) -> str:
    """The logo for the top of a run, or "" when the stream is not a terminal.

    A pipe or log file gets nothing, so logs keep one plain line per event. A terminal that
    cannot show block characters gets the name in plain text; ``NO_COLOR`` and ``TERM=dumb``
    give a one-color logo, and a terminal without ``COLORTERM=truecolor`` gets 256 colors.
    """
    if not stream.isatty():
        return ""
    env = os.environ if env is None else env
    encoding = (getattr(stream, "encoding", None) or "").lower()
    if not encoding.startswith("utf"):
        return "video2sim\n\n"
    color = "NO_COLOR" not in env and env.get("TERM") != "dumb"
    truecolor = env.get("COLORTERM", "").lower() in ("truecolor", "24bit")
    return render(color, truecolor) + "\n\n"


def format_seconds(seconds: float) -> str:
    """``94 s`` below two minutes, ``3 min 12 s`` above."""
    whole = round(seconds)
    if whole < 120:
        return f"{whole} s"
    return f"{whole // 60} min {whole % 60:02d} s"


def format_clock(seconds: float) -> str:
    whole = int(seconds)
    return f"{whole // 60}:{whole % 60:02d}"


class ProgressReporter:
    def __init__(
        self, stream: TextIO, interactive: bool | None = None, verbose: bool = False
    ) -> None:
        self._stream = stream
        self._verbose = verbose
        self._announced = False
        tty = stream.isatty() if interactive is None else interactive
        self._interactive = tty and not verbose
        encoding = (getattr(stream, "encoding", None) or "").lower()
        self._frames = _BRAILLE if encoding.startswith("utf") else _ASCII
        self._lock = threading.Lock()
        self._label = ""
        self._started = 0.0
        self._stop: threading.Event | None = None
        self._thread: threading.Thread | None = None

    def __call__(self, event: StageEvent) -> None:
        label = f"[{event.index}/{event.total}] {event.stage}"
        if self._verbose:
            self._announce(event)
        tail = f"  ({event.detail})" if self._verbose and event.detail else ""
        if event.kind == "started":
            self._begin(label, tail)
        elif event.kind == "cached":
            self._line(f"{label}  cached{tail}")
        elif event.kind == "done":
            self._end(f"{label}  done in {format_seconds(event.seconds or 0.0)}{tail}")
        else:
            reason = f": {event.message}" if event.message else ""
            seconds = format_seconds(event.seconds or 0.0)
            self._end(f"{label}  failed after {seconds}{reason}{tail}")

    def _announce(self, event: StageEvent) -> None:
        if not self._announced and event.run_id:
            self._announced = True
            self._line(f"run {event.run_id}")

    def close(self) -> None:
        """Stop the spinner and clear its line, for example after Ctrl-C."""
        self._halt()
        if self._interactive:
            with self._lock:
                self._stream.write(_ERASE)
                self._stream.flush()

    def _begin(self, label: str, tail: str) -> None:
        self._halt()
        self._label, self._started = label, time.monotonic()
        if not self._interactive:
            self._line(f"{label}  running{tail}")
            return
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._spin, args=(self._stop,), daemon=True)
        self._thread.start()

    def _end(self, text: str) -> None:
        self._halt()
        self._line(text)

    def _halt(self) -> None:
        if self._stop is not None and self._thread is not None:
            self._stop.set()
            self._thread.join()
        self._stop = self._thread = None

    def _line(self, text: str) -> None:
        with self._lock:
            prefix = _ERASE + _INDENT if self._interactive else ""
            self._stream.write(prefix + text + "\n")
            self._stream.flush()

    def _spin(self, stop: threading.Event) -> None:
        tick = 0
        while True:
            frame = self._frames[tick % len(self._frames)]
            elapsed = format_clock(time.monotonic() - self._started)
            with self._lock:
                self._stream.write(f"{_ERASE}{frame} {self._label}  {elapsed}")
                self._stream.flush()
            tick += 1
            if stop.wait(_TICK_S):
                return
