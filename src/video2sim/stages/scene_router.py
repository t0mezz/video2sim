"""Scene router (P2.1 and P2.2 prompt choice, P2.5 missing-object signal).

One hosted VLM request per scene, on a few frames spread over the capture. The model returns the
kind of surface the objects lie on, picked from the closed pool ``segmentation.router.pool``, and
a list of the distinct objects it sees. The surface decides the one concept prompt that SAM 3.1
gets for tracking (``tracking_prompt``), so a scene costs a fixed number of calls and nothing
is rerun on a miss. The list is a light reference, not ground truth: after tracking,
``compare_counts`` reports a scene where there are fewer tracks than listed objects, so a
missing object shows in the QA report instead of passing silently.

The model never writes a SAM prompt; it only picks a pool key. An unreadable answer, a key outside
the pool or a failed call falls back to ``default_surface`` and says so in ``reason``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from video2sim.schemas import WorldModel
from video2sim.stages.discovery_vlm_sam import BASETEN_MODEL, ResponseCache, _image_data_url
from video2sim.stages.segmentation import sample_frame_indices


@dataclass(frozen=True)
class RouterConfig:
    """Values from ``segmentation.router`` in ``configs/default.yaml``."""

    n_frames: int
    max_objects: int
    min_confidence: float
    noun_prompt_max_objects: int
    default_surface: str
    timeout_s: float
    retries: int
    pool: dict[str, tuple[str, str]]  # key -> (description for the model, SAM concept prompt)

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> RouterConfig:
        c = cfg["segmentation"]["router"]
        pool = {
            str(k): (str(v["description"]).strip(), str(v["prompt"]).strip())
            for k, v in c["pool"].items()
        }
        out = cls(
            int(c["n_frames"]),
            int(c["max_objects"]),
            float(c["min_confidence"]),
            int(c["noun_prompt_max_objects"]),
            str(c["default_surface"]),
            float(c["timeout_s"]),
            int(c["retries"]),
            pool,
        )
        if out.n_frames < 1 or out.max_objects < 1 or out.retries < 1 or out.timeout_s <= 0:
            raise ValueError("n_frames, max_objects and retries must be at least 1, timeout_s > 0")
        if not 0.0 <= out.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        if out.noun_prompt_max_objects < 0:
            raise ValueError("noun_prompt_max_objects must not be negative")
        if out.default_surface not in out.pool:
            raise ValueError("default_surface must be a key of pool")
        if any(not d or not p for d, p in out.pool.values()):
            raise ValueError("every pool entry needs a description and a prompt")
        return out


@dataclass(frozen=True)
class RouterAnswer:
    surface: str
    objects: tuple[str, ...]
    confidence: float


@dataclass(frozen=True)
class Route:
    surface: str  # pool key that decided the prompt
    prompt: str  # SAM concept prompt for tracking
    objects: tuple[str, ...]  # what the model saw; empty when the answer was unusable
    reason: str  # why this surface: the model's pick, or the fallback and its cause
    fell_back: bool
    raw_answer: str | None


def build_prompt(cfg: RouterConfig) -> str:
    options = "; ".join(f'"{k}" ({d})' for k, (d, _) in cfg.pool.items())
    return (
        "The photos show the same scene from different views: physical objects resting on a "
        "surface. Answer with a single JSON object and nothing else, with these keys: "
        f'"surface" (the surface the objects rest on, one of {options}), "objects" (a JSON array '
        "with one short singular noun per distinct physical object, counted once across all "
        "photos, so two identical cubes appear twice; leave out the surface, printed boards, "
        'sheets and markers, shadows, hands and arms), "confidence" (a number from 0 to 1 for the '
        "surface choice)."
    )


def parse_answer(text: str, cfg: RouterConfig) -> RouterAnswer | None:
    """The answer, or None when it is not a JSON object with a pool key and a list of nouns."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match is None:
        return None
    try:
        raw = json.loads(match.group(0))
        surface = str(raw["surface"]).strip().lower()
        objects = [str(o).strip().lower() for o in raw["objects"]]
        confidence = float(raw["confidence"])
    except (ValueError, KeyError, TypeError):
        return None
    if surface not in cfg.pool or not 0.0 <= confidence <= 1.0:
        return None
    return RouterAnswer(surface, tuple(o for o in objects if o)[: cfg.max_objects], confidence)


class RouterClient(Protocol):
    def ask(self, prompt: str, images: Sequence[Path]) -> str:
        """Raw model answer for the sampled frames of one scene."""
        ...


def decide_route(raw_answer: str | None, cfg: RouterConfig, failure: str | None = None) -> Route:
    """The route from the model's raw answer; the default surface when it cannot be used."""
    fallback = cfg.default_surface

    def fell_back(reason: str, objects: tuple[str, ...] = ()) -> Route:
        return Route(fallback, cfg.pool[fallback][1], objects, reason, True, raw_answer)

    if failure is not None:
        return fell_back(f"VLM call failed: {failure}")
    answer = parse_answer(raw_answer, cfg) if raw_answer is not None else None
    if answer is None:
        return fell_back("VLM answer unreadable")
    if answer.confidence < cfg.min_confidence:
        return fell_back(
            f"VLM confidence {answer.confidence:.2f} for {answer.surface} is low", answer.objects
        )
    return Route(
        answer.surface,
        cfg.pool[answer.surface][1],
        answer.objects,
        f"VLM surface {answer.surface} ({answer.confidence:.2f})",
        False,
        raw_answer,
    )


def route_scene(
    world: WorldModel,
    run_dir: Path,
    client: RouterClient | None,
    cache: ResponseCache,
    cfg: RouterConfig,
) -> Route:
    """One hosted request (cached) on ``cfg.n_frames`` frames spread over the registered frames.
    Without a ``client`` no call is made and the default surface is used."""
    if client is None:
        return decide_route(None, cfg, failure="no client")
    picks = [world.frames[i] for i in sample_frame_indices(len(world.frames), cfg.n_frames)]
    images = [run_dir / f.image_path for f in picks]
    prompt = build_prompt(cfg)
    key = [prompt, BASETEN_MODEL, *(_image_data_url(p) for p in images)]
    try:
        raw = cache.get_or_call(key, lambda: client.ask(prompt, images))
    except Exception as e:  # a failed call must not stop the scene
        return decide_route(None, cfg, failure=type(e).__name__)
    return decide_route(raw, cfg)


def tracking_prompt(route: Route, cfg: RouterConfig) -> str:
    """The one concept prompt for the tracking request. On a surface other than the default, a
    scene of at most ``noun_prompt_max_objects`` listed objects is tracked by the model's own noun
    (SAM finds "cube" on a glass stove where "objects on a table" finds nothing); anything else
    gets the pool prompt, because the video request takes a single concept."""
    few = 0 < len(route.objects) <= cfg.noun_prompt_max_objects
    if route.surface != cfg.default_surface and not route.fell_back and few:
        return route.objects[0]
    return route.prompt


@dataclass(frozen=True)
class CountCheck:
    listed: int
    tracked: int
    missing_suspected: bool  # fewer tracks than listed objects
    reason: str


def compare_counts(route: Route, n_tracks: int) -> CountCheck:
    """Compare the model's object list with the tracks. Only a shortfall of tracks raises the
    suspicion; extra tracks (a split object, a marker) are reported but not suspected."""
    listed = len(route.objects)
    if route.fell_back and not route.objects:
        return CountCheck(listed, n_tracks, False, "no object list to compare with")
    if n_tracks < listed:
        return CountCheck(listed, n_tracks, True, f"{listed} objects listed but {n_tracks} tracks")
    return CountCheck(listed, n_tracks, False, f"{listed} objects listed, {n_tracks} tracks")
