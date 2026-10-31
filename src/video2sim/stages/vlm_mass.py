"""Mass from the vision-language model, combined with the mass from geometry (D-024).

The one exception to the rule that a model never writes a number: geometry and a fill fraction
alone reach 36% of the weighed dev objects within 30%, because a class cannot say how full or how
heavy one object is. The model knows what a kind of product weighs, and fails on other objects:
alone it reaches 46%, and the rule below 52%, both with objects held out
(``benchmark/results/p4_vlm_mass.json``). Measured size is not sent, because it made the answers
worse there (41%).

The rule keeps the geometry mass when it falls inside the model's range, and otherwise takes the
model's mass with its range. Without a readable answer the geometry mass stands, and the object is
flagged by the caller. With ``physics.vlm_mass.enabled`` false no request is made, no key is
needed, and every object has the geometry mass with no flag, because that was asked for.
``physics/report.json`` says which mass each object got.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from video2sim.models.hosted import ResponseCache
from video2sim.runner import sha256_file
from video2sim.schemas import Estimate
from video2sim.stages.material_pass import MaterialClient

MassSource = Literal["geometry", "model", "geometry_no_answer", "geometry_only"]
PROMPT = """The images show one real object from a tabletop scene, from different views.
It was identified as: {category}.
{measured}Estimate its real mass as it sits there, including any contents you can infer.
Think about what this exact kind of product typically weighs.
Answer with JSON only, no other text:
{{"contents": "<a few words>", "mass_low_g": <number>, "mass_nominal_g": <number>,
"mass_high_g": <number>}}
Low and high should bracket the true mass with about 80% probability."""


@dataclass(frozen=True)
class VlmMassConfig:
    """Values from ``physics.vlm_mass`` in ``configs/default.yaml``."""

    enabled: bool
    keyframes: int
    crop_pad_px: int
    timeout_s: float
    retries: int

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> VlmMassConfig:
        c = cfg["physics"]["vlm_mass"]
        out = cls(
            enabled=bool(c["enabled"]),
            keyframes=int(c["keyframes"]),
            crop_pad_px=int(c["crop_pad_px"]),
            timeout_s=float(c["timeout_s"]),
            retries=int(c["retries"]),
        )
        if out.keyframes < 1 or out.crop_pad_px < 0:
            raise ValueError("physics.vlm_mass needs a keyframe and a margin of at least 0")
        return out


def parse_mass(text: str) -> Estimate | None:
    """The model's answer in kilograms, or None if it cannot be read or is not positive."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match is None:
        return None
    try:
        d = json.loads(match.group(0))
        low, nominal, high = (
            float(d[k]) / 1000 for k in ("mass_low_g", "mass_nominal_g", "mass_high_g")
        )
    except (ValueError, KeyError, TypeError):
        return None
    if not 0 < low <= nominal <= high:
        return None
    return Estimate(low=low, nominal=nominal, high=high)


def ask_mass(
    client: MaterialClient, cache: ResponseCache, category: str, crops: Sequence[Path]
) -> Estimate | None:
    """One request for one object, cached by the prompt and the crops' bytes."""
    prompt = PROMPT.format(category=category, measured="")
    parts = [prompt, *[sha256_file(p) for p in crops]]
    return parse_mass(cache.get_or_call(parts, lambda: client.ask(prompt, crops)))


def combine(
    geometry: Estimate, model: Estimate | None, enabled: bool = True
) -> tuple[Estimate, MassSource]:
    """The module docstring's rule."""
    if not enabled:
        return geometry, "geometry_only"
    if model is None:
        return geometry, "geometry_no_answer"
    if model.low <= geometry.nominal <= model.high:
        return geometry, "geometry"
    return model, "model"


__all__ = ["PROMPT", "MassSource", "VlmMassConfig", "ask_mass", "combine", "parse_mass"]
