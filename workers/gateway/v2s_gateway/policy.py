"""The gateway policy: allowlists, size caps, concurrency and quotas per route (``policy.yaml``)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

Route = Literal["vlm", "tracking", "sam3d"]
ROUTES: tuple[Route, ...] = ("vlm", "tracking", "sam3d")
MB = 1024 * 1024

POLICY_FILE = Path(__file__).resolve().parent.parent / "policy.yaml"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RouteLimits(_Strict):
    calls_per_scene: int = Field(ge=1)
    concurrency: int = Field(ge=1)


class VlmPolicy(RouteLimits):
    models: tuple[str, ...] = Field(min_length=1)
    max_tokens_cap: int = Field(ge=1)
    max_images: int = Field(ge=0)
    max_body_mb: float = Field(gt=0)


class TrackingPolicy(RouteLimits):
    models: tuple[str, ...] = Field(min_length=1)
    max_prompt_chars: int = Field(ge=1)
    max_body_mb: float = Field(gt=0)


class Sam3dPolicy(RouteLimits):
    app: str
    cls: str
    checkpoints: tuple[str, ...] = Field(min_length=1)
    max_inference_steps: int | None = Field(default=None, ge=1)
    max_image_mb: float = Field(gt=0)
    max_mask_mb: float = Field(gt=0)
    max_pointmap_mb: float = Field(gt=0)


class AdminPolicy(_Strict):
    """Bounds on ``POST /admin/tokens``, so a leaked admin key can do limited harm."""

    max_tokens: int = Field(ge=1)
    max_days: float = Field(gt=0)
    max_scenes: int = Field(ge=1)


class Policy(_Strict):
    scenes_per_token: int = Field(ge=1)
    safety_factor: float = Field(gt=0)
    daily_scenes: int = Field(ge=1)
    queue_wait_s: float = Field(ge=0)
    upstream_connect_timeout_s: float = Field(gt=0)
    upstream_read_timeout_s: float = Field(gt=0)
    vlm: VlmPolicy
    tracking: TrackingPolicy
    sam3d: Sam3dPolicy
    admin: AdminPolicy

    def limits(self, route: Route) -> RouteLimits:
        limits: RouteLimits = getattr(self, route)
        return limits

    def quota(self, route: Route, scenes: int | None = None) -> int:
        """Calls one token may make on ``route``: ``ceil(c * n_scenes * k_r)``."""
        n = self.scenes_per_token if scenes is None else scenes
        return math.ceil(self.safety_factor * n * self.limits(route).calls_per_scene)

    def daily_cap(self, route: Route) -> int:
        """Calls all tokens together may make on ``route`` in one UTC day."""
        return math.ceil(
            self.safety_factor * self.daily_scenes * self.limits(route).calls_per_scene
        )


def load_policy(path: Path = POLICY_FILE) -> Policy:
    return Policy.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
