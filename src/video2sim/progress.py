"""Progress lines for ``video2sim run``, written to stderr so stdout stays the final result.

``ProgressReporter`` is a runner ``on_event`` handler. On a terminal it draws a spinner with the
running stage and its elapsed time on one line and replaces it with a result line when the stage
ends. Anywhere else (a pipe, a log file, CI) it writes one plain line per event and never uses
carriage returns. Standard library only.
"""

from __future__ import annotations

import threading
import time
from typing import TextIO

from video2sim.runner import StageEvent

_BRAILLE = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_ASCII = "|/-\\"
_ERASE = "\r\033[2K"
_TICK_S = 0.1


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
    def __init__(self, stream: TextIO, interactive: bool | None = None) -> None:
        self._stream = stream
        self._interactive = stream.isatty() if interactive is None else interactive
        encoding = (getattr(stream, "encoding", None) or "").lower()
        self._frames = _BRAILLE if encoding.startswith("utf") else _ASCII
        self._lock = threading.Lock()
        self._label = ""
        self._started = 0.0
        self._stop: threading.Event | None = None
        self._thread: threading.Thread | None = None

    def __call__(self, event: StageEvent) -> None:
        label = f"[{event.index}/{event.total}] {event.stage}"
        if event.kind == "started":
            self._begin(label)
        elif event.kind == "cached":
            self._line(f"{label}  cached")
        elif event.kind == "done":
            self._end(f"{label}  done in {format_seconds(event.seconds or 0.0)}")
        else:
            reason = f": {event.message}" if event.message else ""
            self._end(f"{label}  failed after {format_seconds(event.seconds or 0.0)}{reason}")

    def close(self) -> None:
        """Stop the spinner and clear its line, for example after Ctrl-C."""
        self._halt()
        if self._interactive:
            with self._lock:
                self._stream.write(_ERASE)
                self._stream.flush()

    def _begin(self, label: str) -> None:
        self._halt()
        self._label, self._started = label, time.monotonic()
        if not self._interactive:
            self._line(f"{label}  running")
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
            self._stream.write((_ERASE if self._interactive else "") + text + "\n")
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
