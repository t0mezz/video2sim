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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from video2sim.models.hosted import ResponseCache, image_data_url
from video2sim.models.vlm import cache_model
from video2sim.schemas import WorldModel
from video2sim.stages.segmentation import sample_frame_indices
from video2sim.stages.view_select import ViewConfig

BASE_NOUNS = ("table", "board", "marker", "sheet", "shadow", "hand", "arm")


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
    views: ViewConfig
    merge_synonyms: bool
    noun_denylist: tuple[str, ...]
    max_workers: int  # parallel per-view calls, same pattern as material_pass

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> RouterConfig:
        c = cfg["segmentation"]["router"]
        pool = {
            str(k): (str(v["description"]).strip(), str(v["prompt"]).strip())
            for k, v in c["pool"].items()
        }
        views = ViewConfig.from_config(cfg)
        extra = [str(x).lower() for x in c.get("noun_denylist", ())]
        denylist = tuple(dict.fromkeys([*BASE_NOUNS, *extra]))
        out = cls(
            int(c["n_frames"]),
            int(c["max_objects"]),
            float(c["min_confidence"]),
            int(c["noun_prompt_max_objects"]),
            str(c["default_surface"]),
            float(c["timeout_s"]),
            int(c["retries"]),
            pool,
            views,
            bool(c.get("merge_synonyms", False)),
            denylist,
            int(c["views"]["max_workers"]),
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
        if out.max_workers < 1:
            raise ValueError("views.max_workers must be at least 1")
        return out


@dataclass(frozen=True)
class RouterAnswer:
    surface: str
    objects: tuple[str, ...]
    confidence: float


@dataclass(frozen=True)
class InventoryItem:
    """One canonical noun of the merged router inventory, for gap fill (9.6)."""

    noun: str
    count: int  # the maximum over views, a lower bound on the true count
    max_hidden_seen: bool  # any view saw it partly hidden


@dataclass(frozen=True)
class ViewItem:
    noun: str
    count: int
    partly_hidden: bool


@dataclass(frozen=True)
class ViewAnswer:
    surface: str
    confidence: float
    items: tuple[ViewItem, ...]


@dataclass(frozen=True)
class Route:
    surface: str  # pool key that decided the prompt
    prompt: str  # SAM concept prompt for tracking
    objects: tuple[str, ...]  # what the model saw; empty when the answer was unusable
    reason: str  # why this surface: the model's pick, or the fallback and its cause
    fell_back: bool
    raw_answer: str | None
    inventory: tuple[InventoryItem, ...] = ()  # merged per-noun counts, coverage mode only
    view_frames: tuple[int, ...] = ()  # frame_index values sent to the router


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


def build_view_prompt(cfg: RouterConfig) -> str:
    options = "; ".join(f'"{k}" ({d})' for k, (d, _) in cfg.pool.items())
    return (
        "The photo shows physical objects resting on a surface. Answer with a single JSON "
        "object and nothing else, with these keys: "
        f'"surface" (the surface the objects rest on, one of {options}), "objects" (a JSON '
        "array with one entry per distinct physical object you see, each with "
        '"noun" (a short singular noun, with a colour word when identical objects must stay '
        'separate), "count" (how many of them are in this photo) and "partly_hidden" (true '
        "when any of them is partly hidden behind another object); include objects that are "
        "partly hidden; leave out the surface, printed boards, sheets and markers, shadows, "
        'hands and arms), "confidence" (a number from 0 to 1 for the surface choice).'
    )


def parse_view_answer(text: str, cfg: RouterConfig) -> ViewAnswer | None:
    """One view's answer, or None when it is not a JSON object with a pool key."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match is None:
        return None
    try:
        raw = json.loads(match.group(0))
        surface = str(raw["surface"]).strip().lower()
        confidence = float(raw["confidence"])
        items = []
        for entry in raw["objects"]:
            noun = str(entry["noun"]).strip().lower()
            try:
                count = int(entry.get("count", 1))
            except (TypeError, ValueError):
                count = 1
            hidden = bool(entry.get("partly_hidden", False))
            if noun:
                items.append(ViewItem(noun, max(count, 1), hidden))
    except (ValueError, KeyError, TypeError, AttributeError):
        return None
    if surface not in cfg.pool or not 0.0 <= confidence <= 1.0:
        return None
    return ViewAnswer(surface, confidence, tuple(items[: cfg.max_objects]))


def sanitise_noun(noun: str, denylist: tuple[str, ...]) -> str | None:
    """A noun clean enough for a SAM prompt, or None to drop it. Lowercase, at most 3
    words and 40 characters, never a denylisted word. The model never writes a number."""
    words = noun.lower().split()
    if not words or len(words) > 3 or len(noun) > 40:
        return None
    clean = " ".join(words)
    if clean in denylist:
        return None
    return clean


def build_synonym_prompt(nouns: Sequence[str]) -> str:
    listed = ", ".join(f'"{n}"' for n in nouns)
    return (
        "These nouns name objects seen in photos of one tabletop scene; some may be synonyms "
        "for the same kind of object. Answer with a single JSON object and nothing else, with "
        'one key "groups": a JSON array of groups, each a JSON array of nouns from the list '
        "below that name the same kind of object, each noun in exactly one group, nouns with "
        f"nothing to merge alone in their group. Nouns: {listed}."
    )


def parse_synonym_groups(text: str, nouns: Sequence[str]) -> list[list[str]] | None:
    """Noun groups from the synonym answer, or None when it is unusable. Only nouns from the
    list are kept, each once; unmentioned nouns stay alone."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match is None:
        return None
    try:
        groups = json.loads(match.group(0))["groups"]
        seen: set[str] = set()
        out = []
        for group in groups:
            members = [str(m).strip().lower() for m in group]
            members = [m for m in members if m in nouns and m not in seen]
            seen.update(members)
            if members:
                out.append(members)
        for noun in nouns:
            if noun not in seen:
                out.append([noun])
    except (ValueError, KeyError, TypeError, AttributeError):
        return None
    return out


def merge_inventory(
    answers: Sequence[ViewAnswer], groups: Sequence[Sequence[str]]
) -> tuple[InventoryItem, ...]:
    """Per canonical noun the maximum count over views (a valid lower bound: each view shows
    only a subset) and whether any view saw it partly hidden. Canonical is the highest-count
    noun, then the shortest, then alphabetical."""
    counts: dict[str, int] = {}
    hidden: dict[str, bool] = {}
    for answer in answers:
        for item in answer.items:
            clean = " ".join(item.noun.lower().split())
            if not clean:
                continue
            counts[clean] = max(counts.get(clean, 0), item.count)
            hidden[clean] = hidden.get(clean, False) or item.partly_hidden
    merged = []
    for group in groups:
        members = [m for m in group if m in counts]
        if not members:
            continue
        canonical = sorted(members, key=lambda m: (-counts[m], len(m), m))[0]
        merged.append(
            InventoryItem(
                canonical, max(counts[m] for m in members), any(hidden[m] for m in members)
            )
        )
    merged.sort(key=lambda i: (-i.count, i.noun))
    return tuple(merged)


def _downscaled(src: Path, dest: Path, max_px: int) -> Path:
    """Copy of ``src`` at ``dest`` with the long side at most ``max_px`` (the local model
    dropped connections on a 3840x2160 PNG). Written once; the bytes are deterministic."""
    import cv2  # heavy import, kept out of module import time

    if not dest.is_file():
        image = cv2.imread(str(src))
        if image is None:
            raise ValueError(f"cannot read {src}")
        height, width = image.shape[:2]
        scale = min(1.0, max_px / max(height, width))
        if scale < 1.0:
            image = cv2.resize(image, (round(width * scale), round(height * scale)))
        dest.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dest), image)
    return dest


def _ask_view(
    client: RouterClient,
    cache: ResponseCache,
    prompt: str,
    image: Path,
) -> str | None:
    """One view's raw answer, or None when the call failed (must not stop the scene)."""
    key = [prompt, cache_model(client), image_data_url(image)]
    try:
        return cache.get_or_call(key, lambda: client.ask(prompt, [image]))
    except Exception:
        return None


def sanitise_all(answer: ViewAnswer, cfg: RouterConfig) -> list[str]:
    """The view's nouns clean enough for a SAM prompt, in answer order."""
    out = []
    for item in answer.items:
        clean = sanitise_noun(item.noun, cfg.noun_denylist)
        if clean is not None and clean not in out:
            out.append(clean)
    return out


def synonym_groups(
    client: RouterClient, cache: ResponseCache, nouns: Sequence[str]
) -> list[list[str]] | None:
    """Noun groups from the text-only synonym call, or None on any failure (exact-string
    merge then applies)."""
    prompt = build_synonym_prompt(nouns)
    try:
        raw = cache.get_or_call([prompt, cache_model(client)], lambda: client.ask(prompt, []))
    except Exception:
        return None
    return parse_synonym_groups(raw, nouns)


def reduce_views(
    answers: Sequence[ViewAnswer],
    inventory: tuple[InventoryItem, ...],
    view_frames: Sequence[int],
    cfg: RouterConfig,
    synonym_note: str,
) -> Route:
    """One route from the usable per-view answers: confidence-weighted surface vote and the
    merged inventory, flattened so ``tracking_prompt`` and ``compare_counts`` are
    unchanged."""
    weights: dict[str, float] = {}
    votes: dict[str, int] = {}
    for answer in answers:
        weights[answer.surface] = weights.get(answer.surface, 0.0) + answer.confidence
        votes[answer.surface] = votes.get(answer.surface, 0) + 1
    order = list(cfg.pool)
    surface = max(weights, key=lambda k: (weights[k], -order.index(k)))
    mean_conf = weights[surface] / votes[surface]
    objects: tuple[str, ...] = ()
    for item in inventory:
        objects += (item.noun,) * item.count
    objects = objects[: cfg.max_objects]
    surfaces = ", ".join(f"{k}: {votes[k]}" for k in sorted(votes))
    reason = (
        f"multi-view router on frames {list(view_frames)}: surfaces {surfaces}; "
        f"{len(inventory)} nouns merged over {len(answers)} views ({synonym_note})"
    )
    if mean_conf < cfg.min_confidence:
        return Route(
            cfg.default_surface,
            cfg.pool[cfg.default_surface][1],
            objects,
            f"{reason}; vote confidence {mean_conf:.2f} is low",
            True,
            None,
            inventory,
            tuple(view_frames),
        )
    return Route(
        surface,
        cfg.pool[surface][1],
        objects,
        f"{reason}; VLM surface {surface} ({mean_conf:.2f})",
        False,
        None,
        inventory,
        tuple(view_frames),
    )


def route_scene(
    world: WorldModel,
    run_dir: Path,
    client: RouterClient | None,
    cache: ResponseCache,
    cfg: RouterConfig,
) -> Route:
    """One hosted request (cached) on ``cfg.n_frames`` frames spread over the registered frames.
    Without a ``client`` no call is made and the default surface is used. In coverage mode one
    cached call runs per selected view (``views.count`` at most) plus one synonym call, in
    parallel; a failing view is skipped, and when all fail the scene falls back as today."""
    if client is None:
        return decide_route(None, cfg, failure="no client")
    if cfg.views.mode == "even":
        picks = [world.frames[i] for i in sample_frame_indices(len(world.frames), cfg.n_frames)]
        images = [run_dir / f.image_path for f in picks]
        prompt = build_prompt(cfg)
        key = [prompt, cache_model(client), *(image_data_url(p) for p in images)]
        try:
            raw = cache.get_or_call(key, lambda: client.ask(prompt, images))
        except Exception as e:  # a failed call must not stop the scene
            return decide_route(None, cfg, failure=type(e).__name__)
        return decide_route(raw, cfg)
    from video2sim.stages.lift import load_points
    from video2sim.stages.view_select import select_views

    cloud = load_points(run_dir / world.point_cloud_path)
    selection = select_views(world, cloud, cfg.views, run_dir)
    positions = [order for order, f in enumerate(world.frames) if f.frame_index in selection.frames]
    view_dir = run_dir / "segmentation.work" / "router_views"
    prompt = build_view_prompt(cfg)
    images = [
        _downscaled(
            run_dir / world.frames[p].image_path,
            view_dir / f"view_{world.frames[p].frame_index:05d}_{cfg.views.max_image_px}px.png",
            cfg.views.max_image_px,
        )
        for p in positions
    ]
    with ThreadPoolExecutor(max_workers=cfg.max_workers) as pool:
        raws = list(pool.map(lambda image: _ask_view(client, cache, prompt, image), images))
    answers: list[ViewAnswer] = []
    failed = 0
    for maybe_raw in raws:
        if maybe_raw is None:
            failed += 1
            continue
        answer = parse_view_answer(maybe_raw, cfg)
        if answer is None:
            failed += 1
        else:
            answers.append(answer)
    if not answers:
        return decide_route(None, cfg, failure=f"all {len(positions)} views failed")
    nouns = sorted({w for a in answers for w in sanitise_all(a, cfg)})
    groups: Sequence[Sequence[str]] = [[n] for n in nouns]
    synonym_note = "synonym merge off"
    if cfg.merge_synonyms and len(nouns) > 1:
        merged = synonym_groups(client, cache, nouns)
        if merged is None:
            synonym_note = "synonym merge failed, exact strings kept"
        else:
            groups, synonym_note = merged, "synonyms merged"
    inventory = merge_inventory(answers, groups)
    route = reduce_views(
        answers, inventory, [world.frames[p].frame_index for p in positions], cfg, synonym_note
    )
    if failed:
        route = Route(
            route.surface,
            route.prompt,
            route.objects,
            f"{route.reason}; {failed} view(s) failed",
            route.fell_back,
            route.raw_answer,
            inventory,
            route.view_frames,
        )
    return route


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
