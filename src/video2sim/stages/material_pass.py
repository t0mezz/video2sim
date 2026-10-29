"""Material pass (P2.5 flag, and the classification half of P4.4 pulled forward, D-017).

One hosted request per object: a vision-language model sees the object's best keyframe crops and
returns structured fields only: a short category, a material class from the closed list in
``segmentation.material_pass.classes``, a surface finish, solid or hollow, and a confidence. The
model never writes a number; mapping a class to density and friction is P4.3 and P4.4.

``Flag.TRANSPARENT_OR_REFLECTIVE`` means "reflectiveness is likely to break a downstream stage".
It is raised when any of these holds, and the reason says which:

* the geometry evidence (``depth_flags.judge``) says the lift is unstable, even if the model
  calls the surface matte, so a disagreement is surfaced;
* the model's finish is one of ``reflective_finishes`` with confidence at least
  ``min_confidence``;
* the model's answer cannot be read, its confidence is low, or the call failed, so a missing check
  shows up in the QA report instead of passing silently.

Otherwise the object is left unflagged. Whether the reflective objects that end up unflagged hurt
shape prediction is checked after the first Phase 3 runs (PLAN.md, P2.5 entry).
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from video2sim.schemas import Flag, WorldModel
from video2sim.stages.depth_flags import DepthSignals, Verdict
from video2sim.stages.discovery_vlm_sam import (
    BASETEN_MODEL,
    BASETEN_URL,
    ResponseCache,
    _image_data_url,
    _post_json,
)
from video2sim.stages.tracking import Track

HOLLOW_VALUES = ("solid", "hollow", "unknown")
RIGIDITY_VALUES = ("rigid", "semi_rigid", "deformable", "unknown")
NON_RIGID_VALUES = ("semi_rigid", "deformable")


@dataclass(frozen=True)
class MaterialConfig:
    """Values from ``segmentation.material_pass`` in ``configs/default.yaml``."""

    keyframes: int
    crop_pad_px: int
    min_confidence: float
    max_workers: int
    timeout_s: float
    retries: int
    finishes: tuple[str, ...]
    reflective_finishes: tuple[str, ...]
    classes: tuple[str, ...]

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> MaterialConfig:
        c = cfg["segmentation"]["material_pass"]
        out = cls(
            int(c["keyframes"]),
            int(c["crop_pad_px"]),
            float(c["min_confidence"]),
            int(c["max_workers"]),
            float(c["timeout_s"]),
            int(c["retries"]),
            tuple(str(x) for x in c["finishes"]),
            tuple(str(x) for x in c["reflective_finishes"]),
            tuple(str(x) for x in c["classes"]),
        )
        if out.keyframes < 1 or out.crop_pad_px < 0:
            raise ValueError("keyframes must be at least 1 and crop_pad_px not negative")
        if out.max_workers < 1 or out.retries < 1 or out.timeout_s <= 0:
            raise ValueError("max_workers and retries must be at least 1 and timeout_s positive")
        if not 0.0 <= out.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        if not out.classes or len(set(out.classes)) != len(out.classes):
            raise ValueError("classes must be non-empty and unique")
        if not out.finishes or len(set(out.finishes)) != len(out.finishes):
            raise ValueError("finishes must be non-empty and unique")
        if not set(out.reflective_finishes) <= set(out.finishes):
            raise ValueError("reflective_finishes must be a subset of finishes")
        return out


@dataclass(frozen=True)
class MaterialAnswer:
    category: str
    material_class: str
    finish: str
    hollow: str
    confidence: float
    rigidity: str = "unknown"


def build_prompt(cfg: MaterialConfig) -> str:
    return (
        "The photos show one physical object, the same object from different "
        "views. Describe the object in the middle. Answer with a single JSON object and nothing "
        'else, with these keys: "category" (a short noun such as "mug"), "material_class" (one '
        f'of {", ".join(cfg.classes)}), "finish" (the surface, one of {", ".join(cfg.finishes)}), '
        '"hollow" (one of solid, hollow, unknown), '
        f'"rigidity" (one of {", ".join(RIGIDITY_VALUES)}), '
        '"confidence" (a number from 0 to 1 for your whole answer). Use rigidity rigid when the '
        "object keeps its shape when pressed, semi_rigid when it holds its shape under its own "
        "weight but visibly yields when pressed (an inflated ball, a stuffed toy), and deformable "
        "when it changes shape under its own weight (cloth, a bag, a cable, a sponge, clay). "
        "Use finish metallic or mirror when the surface reflects its surroundings, "
        "transparent or translucent when light passes through it, glossy for shiny but opaque "
        "plastic, paint or ceramic, and matte otherwise. Look closely: a surface that shows a "
        "clear reflection of its surroundings, such as silver or chrome stickers or foil, is "
        "mirror or metallic even when the object is small."
    )


def parse_answer(text: str, cfg: MaterialConfig) -> MaterialAnswer | None:
    """The answer, or None when it is not a JSON object with every field valid."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match is None:
        return None
    try:
        raw = json.loads(match.group(0))
        category = str(raw["category"]).strip()
        material_class = str(raw["material_class"]).strip().lower()
        finish = str(raw["finish"]).strip().lower()
        hollow = str(raw["hollow"]).strip().lower()
        rigidity = str(raw.get("rigidity", "unknown")).strip().lower()
        confidence = float(raw["confidence"])
    except (ValueError, KeyError, TypeError):
        return None
    if (
        not category
        or material_class not in cfg.classes
        or finish not in cfg.finishes
        or hollow not in HOLLOW_VALUES
        or rigidity not in RIGIDITY_VALUES
        or not 0.0 <= confidence <= 1.0
    ):
        return None
    return MaterialAnswer(category, material_class, finish, hollow, confidence, rigidity)


class MaterialClient(Protocol):
    def ask(self, prompt: str, images: Sequence[Path]) -> str:
        """Raw model answer for the crops of one object."""
        ...


class BasetenMaterialClient:
    """Same hosted model and endpoint as discovery method A."""

    def __init__(self, key: str, timeout_s: float, retries: int) -> None:
        self._key, self._timeout_s, self._retries = key, timeout_s, retries

    def ask(self, prompt: str, images: Sequence[Path]) -> str:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        content += [
            {"type": "image_url", "image_url": {"url": _image_data_url(path)}} for path in images
        ]
        body = {
            "model": BASETEN_MODEL,
            "temperature": 0,
            "messages": [{"role": "user", "content": content}],
        }
        reply = json.loads(_post_json(BASETEN_URL, self._key, body, self._timeout_s, self._retries))
        return str(reply["choices"][0]["message"]["content"])


def crop_for_vlm(
    run_dir: Path, world: WorldModel, track: Track, frame_index: int, pad_px: int, out: Path
) -> Path:
    """Write the crop around the track's mask in one frame and return its path."""
    import cv2

    frame = next(f for f in world.frames if f.frame_index == frame_index)
    image: Any = cv2.imread(str(run_dir / frame.image_path))
    mask = track.masks[frame_index]
    ys, xs = np.nonzero(mask)
    y0, y1 = max(0, int(np.min(ys)) - pad_px), min(mask.shape[0], int(np.max(ys)) + pad_px)
    x0, x1 = max(0, int(np.min(xs)) - pad_px), min(mask.shape[1], int(np.max(xs)) + pad_px)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), image[y0:y1, x0:x1])
    return out


def pick_frames(track: Track, keyframes: Sequence[int], count: int) -> list[int]:
    """Up to ``count`` frames: the track's keyframes in order, else its largest masks."""
    chosen = [f for f in keyframes if f in track.masks][:count]
    if len(chosen) < count:
        rest = sorted(
            (f for f in track.masks if f not in chosen),
            key=lambda f: int(track.masks[f].sum()),
            reverse=True,
        )
        chosen += rest[: count - len(chosen)]
    return chosen


@dataclass(frozen=True)
class MaterialDecision:
    track_id: int
    flag: Flag | None
    reason: str
    geometry: Verdict
    geometry_reason: str
    answer: MaterialAnswer | None
    raw_answer: str | None
    non_rigid: str | None = None  # reason for ``Flag.NON_RIGID``, None when not raised
    seconds: float | None = None  # wall time of the request, None when none was made
    cached: bool = False  # the answer came from the response cache


def decide(
    signals: DepthSignals,
    geometry: Verdict,
    geometry_reason: str,
    raw_answer: str | None,
    cfg: MaterialConfig,
    failure: str | None = None,
) -> MaterialDecision:
    """The flag decision for one track from its geometry verdict and the model's answer."""
    answer = parse_answer(raw_answer, cfg) if raw_answer is not None else None
    reasons: list[str] = []
    if failure is not None:
        reasons.append(f"VLM call failed: {failure}")
    elif answer is None:
        reasons.append("VLM answer unreadable")
    elif answer.confidence < cfg.min_confidence:
        reasons.append(f"VLM confidence {answer.confidence:.2f} is low")
    elif answer.finish in cfg.reflective_finishes:
        reasons.append(f"VLM finish {answer.finish} ({answer.material_class})")
    if geometry == "flag":
        reasons.append(f"geometry: {geometry_reason}")
    flagged = bool(reasons)
    non_rigid = None
    if (
        answer is not None
        and answer.confidence >= cfg.min_confidence
        and answer.rigidity in NON_RIGID_VALUES
    ):
        non_rigid = f"VLM rigidity {answer.rigidity} ({answer.category})"
    return MaterialDecision(
        signals.track_id,
        Flag.TRANSPARENT_OR_REFLECTIVE if flagged else None,
        "; ".join(reasons) if flagged else "VLM and geometry see no reflectiveness",
        geometry,
        geometry_reason,
        answer,
        raw_answer,
        non_rigid,
    )


def ask_for_track(
    client: MaterialClient,
    prompt: str,
    images: Sequence[Path],
    cache: ResponseCache | None,
) -> tuple[str, bool]:
    """The raw answer and whether it came from the cache."""
    if cache is None:
        return client.ask(prompt, images), False
    called = False

    def call() -> str:
        nonlocal called
        called = True
        return client.ask(prompt, images)

    key = [prompt, BASETEN_MODEL, *(_image_data_url(p) for p in images)]
    return cache.get_or_call(key, call), not called


def decide_all(
    signals: Sequence[DepthSignals],
    verdicts: Sequence[tuple[Verdict, str]],
    cfg: MaterialConfig,
    crops: Callable[[int], list[Path]],
    client: MaterialClient | None,
    cache: ResponseCache | None = None,
) -> list[MaterialDecision]:
    """Decisions for every track, in order. Requests run ``cfg.max_workers`` at a time. Without a
    ``client`` no hosted call is made and every track is flagged as unverified."""
    prompt = build_prompt(cfg)

    def one(item: tuple[DepthSignals, tuple[Verdict, str]]) -> MaterialDecision:
        s, (verdict, reason) = item
        if client is None:
            return decide(s, verdict, reason, None, cfg, failure="no client, not verified")
        start = time.monotonic()
        try:
            raw, cached = ask_for_track(client, prompt, crops(s.track_id), cache)
            result = decide(s, verdict, reason, raw, cfg)
            return replace(result, seconds=time.monotonic() - start, cached=cached)
        except Exception as e:  # a failed call must not drop the track unflagged
            failed = decide(s, verdict, reason, None, cfg, failure=type(e).__name__)
            return replace(failed, seconds=time.monotonic() - start)

    with ThreadPoolExecutor(max_workers=cfg.max_workers) as pool:
        return list(pool.map(one, zip(signals, verdicts, strict=True)))
