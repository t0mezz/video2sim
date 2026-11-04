"""End-of-segmentation coverage verification.

After gap fill, a VLM look over whole-scene views with the kept tracks'
outlines drawn asks which router-listed nouns are visible but untracked,
then each missing noun earns one SAM noun video request and survivors go
through the same merge, repair, lift and fit block as gap-fill tracks.
The check loops: a new round re-verifies with the recovered tracks
outlined, so nouns the VLM names again (or names for the first time) get
prompted until nothing missing remains, ``max_turns`` rounds are done, or
the VLM answer cannot be read. ``max_turns: -1`` loops until a round
prompts nothing new, which always ends because a noun is prompted at most
once per scene (a repeated prompt would serve the cached answer anyway).
Tracks found this way keep ``found_by`` ``"noun"`` (they are noun-prompted
tracks), so no contract changes; the verification provenance lives in the
per-object reasons and in the ``verify`` section of ``route.json``.

A scene with nothing missing makes no extra video request, and an
unreadable VLM answer verifies nothing: both keep the gap-fill result as
is. With ``enabled: false`` nothing here runs and the stage output is
byte-identical.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from video2sim.models.hosted import ResponseCache, file_digest
from video2sim.models.vlm import VlmClient, cache_model
from video2sim.schemas import SE3, DiscoveryCheck, WorldModel
from video2sim.stages.gap_fill import (
    FitPass,
    FoundBy,
    GapFillConfig,
    GapRequest,
    fit_pass,
    inherit_provenance,
    merge_requests,
)
from video2sim.stages.lift import LiftConfig
from video2sim.stages.marker_filter import (
    MarkerFilterConfig,
    TagDetector,
    filter_marker_tracks,
)
from video2sim.stages.primitive_fit import PrimitiveConfig
from video2sim.stages.scene_router import sanitise_noun
from video2sim.stages.track_repair import RepairConfig, RepairReport, repair_tracks
from video2sim.stages.tracking import Track

logger = logging.getLogger(__name__)

F64 = NDArray[np.float64]
VERIFY_ID_BASE = 30000  # noun-request ids: clear of the gap-fill noun range
VERIFY_ID_STEP = 1000  # per-request stride; a request returns far fewer tracks
VerdictStatus = Literal["ready", "unchecked"]


@dataclass(frozen=True)
class VerifyConfig:
    """Values from ``segmentation.verify`` in ``configs/default.yaml``."""

    enabled: bool
    max_turns: int  # verification rounds at most; -1 loops until a round prompts nothing new
    max_noun_requests: int  # targeted SAM video requests per round at most
    views: int  # whole-scene frames the VLM checks
    max_image_px: int  # overlays are downscaled to this long side
    min_confidence: float  # below this the VLM answer is ignored
    timeout_s: float
    retries: int

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> VerifyConfig:
        v = cfg.get("segmentation", {}).get("verify", {}) or {}
        out = cls(
            enabled=bool(v.get("enabled", False)),
            max_turns=int(v.get("max_turns", 3)),
            max_noun_requests=int(v.get("max_noun_requests", 4)),
            views=int(v.get("views", 3)),
            max_image_px=int(v.get("max_image_px", 1024)),
            min_confidence=float(v.get("min_confidence", 0.5)),
            timeout_s=float(v.get("timeout_s", 300)),
            retries=int(v.get("retries", 3)),
        )
        if out.max_turns == 0 or out.max_turns < -1:
            raise ValueError("verify max_turns must be -1 or at least 1")
        if not 1 <= out.max_noun_requests <= 8:
            raise ValueError("verify max_noun_requests must be in [1, 8]")
        if out.views < 1 or out.max_image_px < 1 or out.retries < 1 or out.timeout_s <= 0:
            raise ValueError("verify views, max_image_px and retries need at least 1")
        if not 0.0 <= out.min_confidence <= 1.0:
            raise ValueError("verify min_confidence must be in [0, 1]")
        return out


@dataclass(frozen=True)
class VerifyVerdict:
    """The VLM's coverage answer; ``unchecked`` verifies nothing."""

    status: VerdictStatus
    missing: tuple[str, ...] = ()
    ignored: tuple[str, ...] = ()  # named nouns outside the listed inventory
    confidence: float | None = None
    reason: str = ""


def build_verify_prompt(tracked: int, listed: Sequence[str]) -> str:
    """Ask which listed nouns are visible, trackable and have no outline, as JSON."""
    names = ", ".join(listed)
    outline = "one outlined object" if tracked == 1 else f"{tracked} outlined objects"
    return (
        "The images show one tabletop scene from several viewpoints. Colored "
        f"outlines mark objects this pipeline already tracks ({outline}). The "
        f"earlier inventory of this scene listed: {names}. Answer with a single "
        "JSON object and nothing else, with two keys: "
        '"missing" (the listed nouns that are clearly visible in these images, '
        "distinct enough to follow across frames, and have no outline around "
        "them, in the order listed above) and "
        '"confidence" (a number from 0 to 1). A noun that is not visible, is '
        "too small or indistinct to track, or already has an outline, "
        "is not missing."
    )


def parse_verify_answer(
    text: str,
    candidates: Sequence[str],
    denylist: tuple[str, ...],
    min_confidence: float,
) -> VerifyVerdict:
    """Strict parsing of one coverage answer; anything unreadable is unchecked."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match is None:
        return VerifyVerdict("unchecked", reason="unreadable answer")
    try:
        raw = json.loads(match.group(0))
    except ValueError:
        return VerifyVerdict("unchecked", reason="unreadable answer")
    if not isinstance(raw, dict) or not isinstance(raw.get("missing"), list):
        return VerifyVerdict("unchecked", reason="unreadable answer")
    confidence = raw.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return VerifyVerdict("unchecked", reason="unreadable answer")
    conf = float(confidence)
    if conf < min_confidence:
        return VerifyVerdict("unchecked", confidence=conf, reason="low confidence")
    wanted = {c.lower(): c for c in candidates}
    missing: list[str] = []
    ignored: list[str] = []
    for entry in raw["missing"]:
        noun = sanitise_noun(str(entry).strip().lower(), denylist)
        if noun is None or noun not in wanted:
            ignored.append(str(entry))
            continue
        if wanted[noun] not in missing:
            missing.append(wanted[noun])
    return VerifyVerdict("ready", tuple(missing), tuple(ignored), conf)


def make_overlays(
    world: WorldModel,
    run_dir: Path,
    kept: Sequence[Track],
    positions: Sequence[int],
    out_dir: Path,
    max_image_px: int,
) -> list[Path]:
    """One whole-scene view per position with the kept tracks outlined and labelled."""
    import cv2  # heavy import, kept out of module import time

    palette = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for pos in positions:
        frame = world.frames[pos]
        image = cv2.imread(str(run_dir / frame.image_path))
        if image is None:
            raise ValueError(f"cannot read {frame.image_path}")
        overlay = image.copy()
        for track in sorted(kept, key=lambda t: t.track_id):
            crop = track.masks.get(frame.frame_index)
            if crop is None or not crop.any():
                continue
            mask = crop.full()
            if mask.shape != overlay.shape[:2]:
                mask = cv2.resize(
                    mask.astype("uint8"),
                    (overlay.shape[1], overlay.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            color = palette[track.track_id % len(palette)]
            contours, _ = cv2.findContours(
                mask.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(overlay, contours, -1, color, 2)
            ys, xs = np.nonzero(mask)
            cv2.putText(
                overlay,
                f"T{track.track_id}",
                (int(np.min(xs)), max(int(np.min(ys)) - 4, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )
        long_side = max(overlay.shape[:2])
        if long_side > max_image_px:
            scale = max_image_px / long_side
            overlay = cv2.resize(overlay, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        path = out_dir / f"verify_view{pos:02d}.png"
        cv2.imwrite(str(path), overlay)
        paths.append(path)
    return paths


def check_coverage(
    overlays: Sequence[Path],
    listed: Sequence[str],
    tracked: int,
    client: VlmClient | None,
    cache: ResponseCache,
    cfg: VerifyConfig,
    denylist: tuple[str, ...],
) -> VerifyVerdict:
    """One VLM call over the overlaid views; never raises, so a failure verifies nothing."""
    if client is None:
        return VerifyVerdict("unchecked", reason="no VLM client")
    if not listed:
        return VerifyVerdict("ready", reason="nothing listed to verify")
    if not overlays:
        return VerifyVerdict("unchecked", reason="no view to check")
    prompt = build_verify_prompt(tracked, listed)
    parts = [prompt, cache_model(client), *listed, *[file_digest(p) for p in overlays]]
    try:
        text = cache.get_or_call(parts, lambda: client.ask(prompt, list(overlays)))
    except Exception:  # the check must not abort the stage
        logger.warning("verification call failed; nothing verified", exc_info=True)
        return VerifyVerdict("unchecked", reason="VLM call failed")
    return parse_verify_answer(text, listed, denylist, cfg.min_confidence)


def execute_verify(
    world: WorldModel,
    run_dir: Path,
    kept: Sequence[Track],
    listed: Sequence[str],
    vcfg: VerifyConfig,
    gcfg: GapFillConfig,
    lift_cfg: LiftConfig,
    prim_cfg: PrimitiveConfig,
    repair_cfg: RepairConfig,
    marker_cfg: MarkerFilterConfig,
    cloud: F64,
    poses: Mapping[int, SE3],
    track_fn: Callable[[str, int], tuple[list[Track], bool]],
    final: FitPass,
    discovery: DiscoveryCheck | None,
    prov: Mapping[int, FoundBy],
    positions: Sequence[int],
    client: VlmClient | None,
    cache: ResponseCache,
    work_dir: Path,
    denylist: tuple[str, ...],
    detect: TagDetector | None = None,
) -> tuple[
    FitPass, DiscoveryCheck | None, dict[str, Any], dict[int, FoundBy], dict[str, str], list[Track]
]:
    """Verification rounds: VLM coverage check, then noun prompts for missing nouns.

    Every round re-verifies with the recovered tracks outlined, so nouns the
    VLM names again (or for the first time) get prompted until no unprompted
    noun remains, ``max_turns`` rounds are done, or the VLM answer cannot be
    read. A noun is prompted at most once: a repeat would serve the cached
    answer. Returns the fit pass to continue with (the input ``final`` when
    nothing was missing or nothing survived), the discovery audit, the
    ``route.json`` records, the provenance per track id, the per-object
    reasons, and every new track for the tracks preview.
    """
    report_lines: list[str] = []
    scene_flags: list[str] = []
    rounds: list[dict[str, Any]] = []
    round_states: list[tuple[Mapping[int, tuple[str, ...]], Any]] = []
    records: list[dict[str, Any]] = []
    new_tracks: list[Track] = []
    prompted: list[str] = []
    pending: list[str] = []
    marker_dropped = 0
    second = final
    out_prov: dict[int, FoundBy] = dict(prov)
    accum_drops = dict(final.drops)
    turn = 0
    stop_reason = ""
    last_verdict = VerifyVerdict("ready", reason="no round ran")
    while True:
        if vcfg.max_turns >= 0 and turn >= vcfg.max_turns:
            stop_reason = f"max turns ({vcfg.max_turns}) reached"
            break
        turn += 1
        overlays = make_overlays(
            world,
            run_dir,
            list(second.kept),
            positions,
            work_dir / f"round_{turn:02d}",
            vcfg.max_image_px,
        )
        verdict = check_coverage(overlays, listed, len(second.kept), client, cache, vcfg, denylist)
        last_verdict = verdict
        if verdict.status == "unchecked":
            scene_flags.append(f"verify unchecked: {verdict.reason}")
            rounds.append({"round": turn, "verdict": "unchecked", "reason": verdict.reason})
            stop_reason = "VLM answer unreadable"
            break
        for noun in verdict.missing:
            if noun not in prompted and noun not in pending:
                pending.append(noun)
        if not pending:
            report_lines.append(f"verify round {turn}: no missing nouns in the overlaid views")
            rounds.append({"round": turn, "verdict": "ready", "missing": [], "prompted": []})
            stop_reason = "nothing missing"
            break
        todo = pending[: vcfg.max_noun_requests]
        pending = pending[vcfg.max_noun_requests :]
        report_lines.append(
            f"verify round {turn}: VLM names {len(verdict.missing)} missing noun(s), "
            f"{len(todo)} prompted"
        )
        requests: list[GapRequest] = []
        for noun in todo:
            tracks, cached = track_fn(noun, VERIFY_ID_BASE + VERIFY_ID_STEP * len(prompted))
            kept_noun, mrep = filter_marker_tracks(tracks, world, run_dir, marker_cfg, detect)
            marker_dropped += len(mrep.dropped)
            prompted.append(noun)
            report_lines.append(
                f"verify round {turn}: noun '{noun}' prompt: {len(kept_noun)} track(s)"
            )
            records.append(
                {
                    "kind": "noun",
                    "noun": noun,
                    "cached": cached,
                    "tracks": len(kept_noun),
                    "round": turn,
                }
            )
            if kept_noun:
                requests.append(GapRequest(tuple(kept_noun), "noun", f"verify '{noun}'"))
                new_tracks.extend(kept_noun)
        kept_ids = {t.track_id for t in second.kept}
        non_generic_before = sum(1 for t in second.kept if out_prov.get(t.track_id) != "generic")
        if requests:
            merged = merge_requests(list(second.kept), requests, gcfg)
            for request in requests:
                for track in request.tracks:
                    out_prov.setdefault(track.track_id, "noun")
            for old, new in merged.remapped:
                out_prov[new] = out_prov.get(old, "generic")
            repaired, repair_report = repair_tracks(
                merged.tracks, poses, world.intrinsics, world.table_plane, repair_cfg
            )
            out_prov = inherit_provenance(out_prov, repair_report)
            round_states.append((merged.notes, repair_report))
            if {t.track_id for t in merged.tracks} - kept_ids:
                lifted = fit_pass(repaired, world, lift_cfg, prim_cfg, cloud)
                accum_drops.update(lifted.drops)
                second = replace(lifted, drops=dict(accum_drops))
            round_recovered = (
                sum(1 for t in second.kept if out_prov.get(t.track_id) != "generic")
                - non_generic_before
            )
        else:
            round_states.append(({}, RepairReport(splits=(), merges=())))
            round_recovered = 0
        rounds.append(
            {
                "round": turn,
                "verdict": "ready",
                "missing": list(verdict.missing),
                "prompted": list(todo),
                "recovered": round_recovered,
            }
        )
        report_lines.append(f"verify round {turn}: {round_recovered} object(s) recovered")

    matched = {
        note.split("verify '", 1)[1][:-1]
        for notes, _ in round_states
        for note_list in notes.values()
        for note in note_list
        if note.startswith("verify '") and note.endswith("'")
    }
    still_missing = [n for n in prompted if n not in matched]
    capped = list(pending)
    for noun in capped:
        report_lines.append(f"verify: noun '{noun}' never prompted ({stop_reason})")
    recovered = sum(
        1
        for t in second.kept
        if out_prov.get(t.track_id, "generic") != "generic"
        and any(
            _verify_note(t.track_id, notes, report) is not None for notes, report in round_states
        )
    )
    flag_reasons: dict[str, str] = {}
    for t in second.kept:
        note = next(
            (
                _verify_note(t.track_id, notes, report)
                for notes, report in round_states
                if _verify_note(t.track_id, notes, report) is not None
            ),
            None,
        )
        if note is None:
            continue
        if out_prov.get(t.track_id, "generic") != "generic":
            flag_reasons[str(t.track_id)] = f"found by verify: {note}"
        else:
            flag_reasons[str(t.track_id)] = f"also found by verify: {note}"
    video_requests = 1 + len(records)
    report_lines.append(
        f"verify: {recovered} object(s) recovered in {len(records)} extra request(s) "
        f"over {turn} round(s)"
    )
    for t in second.kept:
        if str(t.track_id) in flag_reasons:
            report_lines.append(f"obj_{t.track_id:03d}: {flag_reasons[str(t.track_id)]}")

    discovery2 = None
    if discovery is not None:
        discovery2 = discovery.model_copy(
            update={
                "tracked": len(second.kept),
                "missing_suspected": discovery.listed > len(second.kept),
                "unmatched": [
                    *discovery.unmatched,
                    *[n for n in [*still_missing, *capped] if n not in discovery.unmatched],
                ],
                "recovered": discovery.recovered + recovered,
                "video_requests": discovery.video_requests + len(records),
            }
        )
    notes = _notes(
        listed,
        last_verdict,
        prompted,
        capped,
        records,
        marker_dropped,
        video_requests,
        report_lines,
        flag_reasons,
        scene_flags,
        None,
        still_missing,
        recovered,
        rounds,
        stop_reason,
    )
    return second, discovery2, notes, out_prov, flag_reasons, new_tracks


def _verify_note(tid: int, notes: Mapping[int, tuple[str, ...]], report: Any) -> str | None:
    """Provenance note for a post-repair track: its own, else its split parent's."""
    if notes.get(tid):
        return notes[tid][0]
    for orig, parts in report.splits:
        if tid in parts and notes.get(orig):
            return notes[orig][0]
    return None


def _notes(
    listed: Sequence[str],
    verdict: VerifyVerdict,
    wanted: Sequence[str],
    capped: Sequence[str],
    records: Sequence[dict[str, Any]],
    marker_dropped: int,
    video_requests: int,
    report_lines: Sequence[str],
    flag_reasons: Mapping[str, str],
    scene_flags: Sequence[str] = (),
    merged: Any = None,
    still_missing: Sequence[str] = (),
    recovered: int = 0,
    rounds: Sequence[dict[str, Any]] = (),
    stop_reason: str = "",
) -> dict[str, Any]:
    return {
        "listed": list(listed),
        "verdict": verdict.status,
        "missing": list(verdict.missing),
        "ignored": list(verdict.ignored),
        "confidence": verdict.confidence,
        "reason": verdict.reason,
        "prompted": list(wanted),
        "capped": list(capped),
        "requests": list(records),
        "rounds": list(rounds),
        "stop_reason": stop_reason,
        "turns": len(rounds),
        "marker_dropped": marker_dropped,
        "merged_notes": (
            {str(k): list(v) for k, v in merged.notes.items() if v} if merged is not None else {}
        ),
        "dropped": ({str(k): v for k, v in merged.dropped.items()} if merged is not None else {}),
        "still_missing": list(still_missing),
        "recovered": recovered,
        "video_requests": video_requests,
        "report_lines": list(report_lines),
        "flag_reasons": dict(flag_reasons),
        "scene_flags": list(scene_flags),
    }


__all__ = [
    "VERIFY_ID_BASE",
    "VERIFY_ID_STEP",
    "VerifyConfig",
    "VerifyVerdict",
    "build_verify_prompt",
    "check_coverage",
    "execute_verify",
    "make_overlays",
    "parse_verify_answer",
]
