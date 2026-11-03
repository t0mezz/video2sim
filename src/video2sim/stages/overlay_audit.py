"""Overlay audit after tracking (S12.9, task 9.7).

The generic SAM concept can miss an object entirely — s25's laptop never became a
track — and a count comparison cannot say what is missing. The audit renders the
kept tracks as numbered coloured overlays on the coverage-selected router views
and asks the vision-language model which distinct physical objects have no
overlay. The returned nouns feed the gap-fill noun requests of 9.6, so every
extra video request is spent on an object known to be missing.

With ``enabled: false`` nothing here runs. All parsing is strict: an unreadable
answer contributes no nouns, never a guess.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from video2sim.models.hosted import ResponseCache
from video2sim.models.vlm import cache_model
from video2sim.schemas import WorldModel
from video2sim.stages.scene_router import RouterClient, sanitise_noun
from video2sim.stages.segmentation import BoolMask
from video2sim.stages.tracking import Track

PALETTE: tuple[tuple[int, int, int], ...] = (  # BGR, distinct on print and screen
    (0, 0, 255),
    (0, 165, 255),
    (0, 255, 0),
    (255, 0, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
    (128, 0, 255),
    (255, 128, 0),
    (0, 128, 255),
    (128, 255, 0),
    (255, 0, 128),
)


@dataclass(frozen=True)
class OverlayAuditConfig:
    """Values from ``segmentation.tracking.gap_fill.overlay_audit``."""

    enabled: bool
    n_views: int  # labeled views sent to the model, router-view order
    save_video: bool  # every track over every frame, as tracks_preview.mp4 in the stage folder

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> OverlayAuditConfig:
        g = cfg["segmentation"]["tracking"]["gap_fill"].get("overlay_audit", {})
        out = cls(
            enabled=bool(g.get("enabled", False)),
            n_views=int(g.get("n_views", 2)),
            save_video=bool(g.get("save_video", True)),
        )
        if out.n_views < 1:
            raise ValueError("overlay_audit n_views must be at least 1")
        return out


def build_audit_prompt(n_tracked: int) -> str:
    """Ask for the physical objects with no overlay, as JSON and nothing else."""
    return (
        "The photo shows physical objects resting on a surface. Each coloured region "
        f"with a white number marks one already-tracked object ({n_tracked} tracked). "
        "Answer with a single JSON object and nothing else, with two keys: "
        '"missing" (a JSON array with one entry per distinct physical object with '
        'NO coloured number on it — each entry has "noun" (one short singular noun, '
        'count identical objects once each) and "near" (the white number closest to '
        "it, or null when it sits far from every number); leave out the surface, "
        "printed boards, sheets and markers, shadows, hands and arms; an empty array "
        'when everything visible is marked), "uncertain" (nouns you cannot decide '
        "on; they are never re-tracked)."
    )


@dataclass(frozen=True)
class AuditHit:
    """One missing noun with the overlaid label the model saw closest to it."""

    noun: str
    near: int | None  # nearest white label number, null when far from every label


def parse_audit_answer(text: str, denylist: tuple[str, ...]) -> tuple[list[AuditHit], list[str]]:
    """Sanitised missing hits and uncertain nouns, or empties when unreadable.

    Only well-formed entries survive: a short sanitised noun, each once and in
    order, with ``near`` kept only when it is an integer label number.
    """
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match is None:
        return [], []
    try:
        raw = json.loads(match.group(0))
        missing, uncertain = raw.get("missing", []), raw.get("uncertain", [])
        if not isinstance(missing, list) or not isinstance(uncertain, list):
            return [], []
    except (ValueError, AttributeError, TypeError):
        return [], []
    hits: list[AuditHit] = []
    seen: set[str] = set()
    for entry in missing:
        noun, near = _hit(entry, denylist)
        if noun is not None and noun not in seen:
            seen.add(noun)
            hits.append(AuditHit(noun, near))
    return hits, _clean([str(o).strip().lower() for o in uncertain], denylist)


def _hit(entry: object, denylist: tuple[str, ...]) -> tuple[str | None, int | None]:
    """Sanitised noun and label number of one missing entry, or (None, None)."""
    if not isinstance(entry, dict):
        return None, None
    clean = sanitise_noun(str(entry.get("noun", "")).strip().lower(), denylist)
    if clean is None:
        return None, None
    near = entry.get("near", None)
    if isinstance(near, bool) or not isinstance(near, int):
        near = None
    return clean, near


def _clean(nouns: Sequence[str], denylist: tuple[str, ...]) -> list[str]:
    out: list[str] = []
    for noun in nouns:
        clean = sanitise_noun(noun, denylist)
        if clean is not None and clean not in out:
            out.append(clean)
    return out


def _draw_masks(
    overlay: NDArray[np.uint8],
    masks: Mapping[int, BoolMask],
    colors: Mapping[int, tuple[int, int, int]],
) -> list[int]:
    """Track masks tinted, contoured and numbered onto ``overlay``; ids drawn, sorted.

    Masks at another resolution than the frame are resampled nearest-neighbour,
    explicitly, like the tracking masks. Colours come from ``track_colors`` so one
    track keeps its colour across views and frames.
    """
    import cv2  # heavy import, kept out of module import time

    drawn: list[int] = []
    for tid in sorted(masks):
        mask = masks[tid]
        if mask.shape != overlay.shape[:2]:
            small = cv2.resize(
                mask.astype("uint8"),
                (overlay.shape[1], overlay.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        else:
            small = mask
        if not small.any():
            continue
        color = colors[tid]
        tint = overlay.copy()
        tint[small] = color
        cv2.addWeighted(tint, 0.22, overlay, 0.78, 0, overlay)
        contours, _ = cv2.findContours(
            small.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, color, 2)
        ys, xs = small.nonzero()
        label = str(tid)
        org = (int(np.min(xs)), int(max(int(np.min(ys)) - 6, 0)))
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(overlay, (org[0], org[1] - th - 6), (org[0] + tw + 6, org[1]), color, -1)
        cv2.putText(
            overlay,
            label,
            (org[0] + 3, org[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
        )
        drawn.append(tid)
    return drawn


def render_audit_view(
    frame_path: Path,
    masks: Mapping[int, BoolMask],
    out_path: Path,
) -> list[int]:
    """Frame with each track mask tinted, contoured and numbered; ids drawn, sorted.

    Returns the drawn ids.
    """
    import cv2  # heavy import, kept out of module import time

    frame = cv2.imread(str(frame_path))
    if frame is None:
        raise ValueError(f"cannot read {frame_path}")
    overlay = frame.copy()
    drawn = _draw_masks(overlay, masks, track_colors(list(masks)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)
    return drawn


def track_colors(ids: Sequence[int]) -> dict[int, tuple[int, int, int]]:
    """Legend colour per track id, in sorted-id order."""
    ordered = sorted(ids)
    return {tid: PALETTE[pos % len(PALETTE)] for pos, tid in enumerate(ordered)}


def render_tracks_video(
    world: WorldModel,
    run_dir: Path,
    tracks: Sequence[Track],
    kept: set[int],
    notes: Mapping[int, str],
    out_path: Path,
    fps: float = 5.0,
) -> int:
    """Every track over every frame, tinted and numbered, with a fate legend.

    ``notes`` names each track's fate (``kept …`` or the drop reason); kept ids
    show green, the rest red. Returns the frames written.
    """
    import cv2  # heavy import, kept out of module import time

    first = cv2.imread(str(run_dir / world.frames[0].image_path))
    if first is None:
        raise ValueError(f"cannot read {world.frames[0].image_path}")
    height, width = first.shape[:2]
    colors = track_colors([t.track_id for t in tracks])
    by_frame: dict[int, list[Track]] = {}
    for track in tracks:
        for frame_index in track.masks:
            by_frame.setdefault(frame_index, []).append(track)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # VideoWriter_fourcc is missing from the cv2 stubs.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    written = 0
    try:
        for frame in world.frames:
            image = cv2.imread(str(run_dir / frame.image_path))
            if image is None:
                raise ValueError(f"cannot read {frame.image_path}")
            overlay = image.copy()
            present = {
                t.track_id: t.masks[frame.frame_index] for t in by_frame.get(frame.frame_index, [])
            }
            _draw_masks(overlay, present, colors)
            y = 24
            for tid in sorted(colors):
                color = (0, 200, 0) if tid in kept else (60, 60, 220)
                line = f"T{tid}: {notes.get(tid, '?')}"
                (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(overlay, (8, y - th - 6), (8 + tw + 8, y + 2), (0, 0, 0), -1)
                cv2.putText(overlay, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
                y += 22
            writer.write(overlay)
            written += 1
    finally:
        writer.release()
    return written


def audit_missing_nouns(
    world: WorldModel,
    run_dir: Path,
    kept: Sequence[Track],
    view_frames: Sequence[int],
    cfg: OverlayAuditConfig,
    denylist: tuple[str, ...],
    client: RouterClient | None,
    cache: ResponseCache,
    work_dir: Path,
) -> tuple[list[str], list[str], list[Path]]:
    """Nouns the model sees without an overlay, plus uncertain ones and the views.

    No client (dry run) or an unreadable answer contributes no nouns. Views come
    first from the coverage-selected router frames; frames with none of the kept
    masks are skipped, so a view always shows something to judge against. A noun
    the model cites next to a kept label is already tracked and never re-tracked.
    """
    if not cfg.enabled or client is None or not kept or not view_frames:
        return [], [], []
    by_frame = {f.frame_index: f for f in world.frames}
    images: list[Path] = []
    for frame_index in view_frames:
        if len(images) >= cfg.n_views:
            break
        frame = by_frame.get(frame_index)
        if frame is None:
            continue
        masks = {t.track_id: t.masks[frame_index] for t in kept if frame_index in t.masks}
        if not masks:
            continue
        out = work_dir / f"audit_{frame_index:05d}.png"
        render_audit_view(run_dir / frame.image_path, masks, out)
        images.append(out)
    if not images:
        return [], [], []
    prompt = build_audit_prompt(len(kept))
    parts = [prompt, cache_model(client), *[p.name for p in images]]
    text = cache.get_or_call(parts, lambda: client.ask(prompt, images))
    hits, uncertain = parse_audit_answer(text, denylist)
    kept_ids = {t.track_id for t in kept}
    missing = [h.noun for h in hits if h.near not in kept_ids]
    return missing, uncertain, images
