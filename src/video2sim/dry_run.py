"""``video2sim run --dry-run``: the progress output of a run, without any work.

Sends the runner's own ``StageEvent`` stream to a listener, so the spinner and the
result lines can be looked at or tested without a video, a config, hosted calls or a run folder.
The first stage is reported as cached, to show that line too. Standard library only.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from video2sim.runner import StageEvent
from video2sim.schemas import STAGE_ORDER

RUN_ID = "dry-run"


def simulate(
    on_event: Callable[[StageEvent], None],
    stage_s: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Report every stage as started and done after ``stage_s`` seconds; the first as cached."""
    total = len(STAGE_ORDER)
    for index, stage in enumerate(STAGE_ORDER, start=1):
        detail = f"{index:08x}"
        if index == 1:
            on_event(StageEvent("cached", stage, index, total, run_id=RUN_ID, detail=detail))
            continue
        on_event(StageEvent("started", stage, index, total, run_id=RUN_ID, detail=detail))
        start = time.monotonic()
        sleep(stage_s)
        seconds = time.monotonic() - start
        on_event(StageEvent("done", stage, index, total, seconds, run_id=RUN_ID, detail=detail))
