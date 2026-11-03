"""Progress lines for ``video2sim run``, written to stderr so stdout stays the final result.

``ProgressReporter`` is a runner ``on_event`` handler. On a terminal it draws a spinner with the
running stage and its elapsed time on one line and replaces it with a result line when the stage
ends. A finished stage gets a green check in the spinner's column; other result lines keep the
indented column, so the stage labels of past and running stages line up. Anywhere else (a pipe,
a log file, CI) it writes one plain line per event and never uses carriage returns.

With ``verbose`` it always writes plain lines, so the logs of native libraries that are left on
do not tangle with a spinner, and adds the run id and the cache keys. Standard library only.

With ``runs_root`` a finished stage (``done`` or ``cached``) gains up to two summary lines read
from that stage's own ``output.json`` and its sidecars: frame counts, the router's path, proxy
fallbacks, mass sources. Each summary line starts with an arrow (``↳`` on UTF-8 streams,
``->`` elsewhere) in the spinner/checkmark column. A missing or unreadable file means
no summary lines, never an error,
so dry runs and foreign folders stay quiet.

An ``info`` event is a live line a stage reports while it runs (``StageContext.report``): one
line per object per milestone, indented two spaces past the stage label, gray on color
terminals and plain but still indented anywhere else. On a terminal the spinner pauses
for the line and resumes with its elapsed time kept.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TextIO

from video2sim.logo import render
from video2sim.runner import StageEvent

_MAX_SUMMARY_LINES = 2
_MAX_SUMMARY_CHARS = 200

_BRAILLE = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_ASCII = "|/-\\"
_ERASE = "\r\033[2K"
_INDENT = "  "  # one spinner frame and a space
_GREEN = "\033[32m"
_GRAY = "\033[90m"
_RESET = "\033[0m"
_TICK_S = 0.1


def _done_mark(interactive: bool, encoding: str) -> str:
    """Green check for finished stages, in the spinner's column.

    Terminals only: pipes, log files and CI keep the plain indented line, and
    ``NO_COLOR``/dumb terminals and non-UTF streams fall back to the indent.
    """
    if not interactive:
        return ""
    if "NO_COLOR" in os.environ or os.environ.get("TERM") == "dumb":
        return ""
    if not encoding.startswith("utf"):
        return ""
    return f"{_GREEN}✓{_RESET} "


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


# ---------------------------------------------------------------------------
# Stage summaries: short lines read from the stage outputs, shown by default.
# ---------------------------------------------------------------------------


def _clip(text: str, limit: int = _MAX_SUMMARY_CHARS) -> str:
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= limit else collapsed[: limit - 3] + "..."


def _as_list(value: object) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _as_dict(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_str(value: object) -> str:
    return value if isinstance(value, str) else ""


def _as_float(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _sidecar(stage_dir: Path, name: str) -> dict[str, Any]:
    """A JSON sidecar next to ``output.json`` (``route.json``, ``report.json``), or {}."""
    try:
        data = json.loads((stage_dir / name).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _summarize_capture(output: dict[str, Any], _stage_dir: Path) -> list[str]:
    frames = _as_list(output.get("frame_paths"))
    if not output.get("accepted", False):
        reasons = "; ".join(r for r in (_as_str(r) for r in _as_list(output.get("reasons"))) if r)
        return ["rejected: " + _clip(reasons or "no reason given")]
    fps = _as_float(output.get("fps"))
    return [f"{len(frames)} frames" + (f" at {fps:.1f} fps" if fps is not None else "")]


def _summarize_reconstruction(output: dict[str, Any], _stage_dir: Path) -> list[str]:
    parts = [f"{len(_as_list(output.get('frames')))} frames"]
    fraction = _as_float(output.get("registered_fraction"))
    if fraction is not None:
        parts.append(f"{100 * fraction:.0f}% registered")
    error = _as_float(output.get("mean_reprojection_error_px"))
    if error is not None:
        parts.append(f"reprojection {error:.2f} px")
    source = _as_str(output.get("scale_source"))
    if source:
        parts.append(f"scale from {source}")
    lines = [", ".join(parts)]
    flags = [f for f in (_as_str(f) for f in _as_list(output.get("flags"))) if f]
    if flags:
        lines.append("flags: " + _clip(", ".join(flags)))
    return lines


def _summarize_segmentation(output: dict[str, Any], stage_dir: Path) -> list[str]:
    objects = _as_list(output.get("objects"))
    route = _sidecar(stage_dir, "route.json")
    if route:
        listed = len(_as_list(route.get("objects")))
        repaired = route.get("tracks_repaired", len(objects))
        splits = len(_as_list(route.get("splits")))
        merges = len(_as_list(route.get("merges")))
        reason = _as_str(route.get("reason")) or _as_str(route.get("surface"))
        lines = [
            f"route {_clip(reason, 80)}: "
            f"{listed} listed, {repaired} tracks ({merges} merges, {splits} splits)"
        ]
    else:
        lines = [f"{len(objects)} objects"]
    flag_counts: dict[str, int] = {}
    for obj in objects:
        for flag in _as_list(_as_dict(obj).get("flags")):
            name = _as_str(flag)
            if name:
                flag_counts[name] = flag_counts.get(name, 0) + 1
    notes = [f for f in (_as_str(f) for f in _as_list(output.get("flags"))) if f]
    counts = ", ".join(f"{name} ({n})" for name, n in sorted(flag_counts.items()))
    tail = "; ".join(part for part in ["; ".join(notes[:2]), counts] if part)
    if len(notes) > 2:
        tail += f" (+{len(notes) - 2} more)"
    if tail:
        lines.append(_clip("flags: " + tail))
    return lines


def _checks_ok(checks: object) -> bool:
    table = _as_dict(checks)
    return bool(table) and all(table.values())


def _summarize_shape(output: dict[str, Any], _stage_dir: Path) -> list[str]:
    objects = [_as_dict(o) for o in _as_list(output.get("objects"))]
    proxies = sum(1 for o in objects if _as_str(o.get("geometry_source")) != "sam3d")
    modeled = len(objects) - proxies
    parts = f"{modeled} modeled, {proxies} proxy" if proxies else f"{len(objects)} modeled meshes"
    scored: list[tuple[float, str]] = []
    for obj in objects:
        score = _as_float(obj.get("fit_score"))
        if score is not None:
            scored.append((score, _as_str(obj.get("object_id"))))
    if scored:
        worst = min(scored, key=lambda pair: pair[0])
        parts += f"; worst fit {worst[0]:.2f} {worst[1]}"
    return [parts]


def _summarize_physics(output: dict[str, Any], stage_dir: Path) -> list[str]:
    assets = [_as_dict(a) for a in _as_list(output.get("assets"))]
    valid = sum(1 for a in assets if _checks_ok(a.get("checks")))
    line = f"{len(assets)} assets, {valid} valid"
    report = _sidecar(stage_dir, "report.json")
    if report:
        entries = [_as_dict(e) for e in report.values()]
        geometry = sum(1 for e in entries if _as_str(e.get("mass_source")).startswith("geometry"))
        model = sum(1 for e in entries if _as_str(e.get("mass_source")) == "model")
        line += f"; mass from geometry x{geometry}, model x{model}"
    lines = [line]
    settling = _sidecar(stage_dir, "settling.json")
    cases = [_as_dict(v) for v in settling.values() if isinstance(v, dict)]
    nominal = _as_dict(settling.get("nominal")) or next(iter(cases), {})
    drifts: list[tuple[float, str]] = []
    for body in _as_list(nominal.get("bodies")):
        drift = _as_float(_as_dict(body).get("max_drift_m"))
        if drift is not None:
            drifts.append((drift, _as_str(_as_dict(body).get("object_id"))))
    if drifts:
        worst = max(drifts, key=lambda pair: pair[0])
        lines.append(f"worst drift {worst[0] * 1000:.1f} mm {worst[1]}")
    return lines


def _summarize_export(output: dict[str, Any], _stage_dir: Path) -> list[str]:
    exported = len(_as_dict(output.get("urdf_paths")))
    skipped = [_as_dict(s) for s in _as_list(output.get("skipped"))]
    line = f"{exported} objects exported"
    if skipped:
        notes = "; ".join(_skipped_note(s) for s in skipped)
        line += f"; {len(skipped)} skipped: " + _clip(notes)
    return [line]


def _skipped_note(skipped: dict[str, Any]) -> str:
    reasons = "; ".join(_as_str(r) for r in _as_list(skipped.get("reasons")))
    return f"{_as_str(skipped.get('object_id'))} ({_clip(reasons, 80)})"


SummaryFn = Callable[[dict[str, Any], Path], list[str]]

_SUMMARIES: dict[str, SummaryFn] = {
    "capture": _summarize_capture,
    "reconstruction": _summarize_reconstruction,
    "segmentation": _summarize_segmentation,
    "shape": _summarize_shape,
    "physics": _summarize_physics,
    "export": _summarize_export,
}


def summarize_stage(runs_root: Path, run_id: str, stage: str) -> list[str]:
    """Up to two summary lines for a finished stage, read from its own outputs.

    Only display data: nothing here affects the run. A missing stage, a missing or
    unreadable file, or an unknown stage means no lines, never an error.
    """
    try:
        raw = (runs_root / run_id / stage / "output.json").read_text(encoding="utf-8")
        output = json.loads(raw)
    except (OSError, ValueError):
        return []
    if not isinstance(output, dict):
        return []
    summarize = _SUMMARIES.get(stage)
    if summarize is None:
        return []
    try:
        lines = [line for line in summarize(output, runs_root / run_id / stage) if line]
    except Exception:
        return []
    return lines[:_MAX_SUMMARY_LINES]


class ProgressReporter:
    def __init__(
        self,
        stream: TextIO,
        interactive: bool | None = None,
        verbose: bool = False,
        runs_root: str | Path | None = None,
    ) -> None:
        self._stream = stream
        self._verbose = verbose
        self._announced = False
        self._runs_root = Path(runs_root) if runs_root is not None else None
        tty = stream.isatty() if interactive is None else interactive
        self._interactive = tty and not verbose
        encoding = (getattr(stream, "encoding", None) or "").lower()
        self._mark = _done_mark(self._interactive, encoding)
        # The mark is non-empty exactly on color-capable terminals, so it doubles
        # as the gray switch for live lines below.
        self._color = bool(self._mark)
        self._frames = _BRAILLE if encoding.startswith("utf") else _ASCII
        self._arrow = "↳ " if encoding.startswith("utf") else "-> "
        self._lock = threading.Lock()
        # Serializes spinner transitions (halt/resume) across the runner thread and
        # stage worker threads reporting live lines. Reentrant: _halt/_resume below
        # assume the caller holds it. Always taken before _lock, never the reverse.
        self._transition = threading.RLock()
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
        elif event.kind == "info":
            self._info(event.message or "")
        elif event.kind == "cached":
            self._line(f"{label}  cached{tail}")
            self._report_summaries(event)
        elif event.kind == "done":
            self._end(
                f"{label}  done in {format_seconds(event.seconds or 0.0)}{tail}",
                ok=True,
            )
            self._report_summaries(event)
        else:
            reason = f": {event.message}" if event.message else ""
            seconds = format_seconds(event.seconds or 0.0)
            self._end(f"{label}  failed after {seconds}{reason}{tail}")

    def _report_summaries(self, event: StageEvent) -> None:
        """Summary lines for a finished stage, read from its own outputs when known.

        Each line gets an arrow marker (``↳`` on UTF-8 streams, ``->`` elsewhere)
        in the spinner/checkmark column, so summaries stand out from live lines
        and result lines.
        """
        if self._runs_root is None or not event.run_id:
            return
        try:
            lines = summarize_stage(self._runs_root, event.run_id, event.stage)
        except Exception:
            return
        for line in lines:
            self._summary(line)

    def _announce(self, event: StageEvent) -> None:
        if not self._announced and event.run_id:
            self._announced = True
            self._line(f"run {event.run_id}")

    def close(self) -> None:
        """Stop the spinner and clear its line, for example after Ctrl-C."""
        with self._transition:
            self._halt()
            if self._interactive:
                with self._lock:
                    self._stream.write(_ERASE)
                    self._stream.flush()

    def _info(self, text: str) -> None:
        """A live line from a running stage. Never blank, never longer than a summary.

        Gray and indented two spaces past the stage label on color terminals; plain
        but still indented anywhere else, so pipes and logs stay free of escapes.
        """
        line = _clip(text)
        if not line:
            return
        if self._color:
            line = f"{_GRAY}{line}{_RESET}"
        line = f"  {line}"
        with self._transition:
            if self._interactive and self._thread is not None:
                label, started = self._label, self._started
                self._halt()
                self._line(line)
                self._label, self._started = label, started
                self._resume()
            else:
                self._line(line)

    def _begin(self, label: str, tail: str) -> None:
        with self._transition:
            self._halt()
            self._label, self._started = label, time.monotonic()
            if not self._interactive:
                self._line(f"{label}  running{tail}")
                return
            self._resume()

    def _resume(self) -> None:
        # Call with _transition held. The thread starts before it is published: a
        # concurrent _halt may then only ever see started threads, so join() cannot
        # hit a thread that was created but not started. That race crashed long
        # stages reporting live lines from worker threads (RuntimeError: cannot
        # join thread before it is started).
        stop = threading.Event()
        thread = threading.Thread(target=self._spin, args=(stop,), daemon=True)
        thread.start()
        self._stop = stop
        self._thread = thread

    def _end(self, text: str, ok: bool = False) -> None:
        with self._transition:
            self._halt()
            if ok and self._mark:
                with self._lock:
                    self._stream.write(f"{_ERASE}{self._mark}{text}\n")
                    self._stream.flush()
            else:
                self._line(text)

    def _halt(self) -> None:
        # Call with _transition held. The grabbed thread was started before it was
        # published (see _resume), so join() below always sees a started thread.
        stop, thread = self._stop, self._thread
        self._stop = self._thread = None
        if stop is not None and thread is not None:
            stop.set()
            thread.join()

    def _summary(self, text: str) -> None:
        """A stage summary line, with the arrow in the spinner/checkmark column."""
        with self._lock:
            if self._interactive:
                self._stream.write(f"{_ERASE}{self._arrow}{text}\n")
            else:
                self._stream.write(f"{self._arrow}{text}\n")
            self._stream.flush()

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
