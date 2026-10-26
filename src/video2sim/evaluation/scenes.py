"""Benchmark scenes and predictions as the harness sees them (P0.6).

Ground truth is read from ``benchmark/scenes/<scene_id>/ground_truth.json`` (format:
``benchmark/README.md``). A prediction is the world-frame vertices of each predicted mesh and,
if the pipeline got that far, its nominal mass. The world frame and the ground-truth frame are
the same one: origin on the marker board, +Z up (AGENTS.md).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

GT_FILE = "ground_truth.json"


@dataclass(frozen=True)
class GtObject:
    object_id: str
    label: str
    size_m: tuple[float, float, float] | None
    position_m: tuple[float, float, float] | None
    yaw_deg: float | None
    mass_kg: float | None
    incomplete: tuple[str, ...] = ()
    deformable: bool = False


@dataclass(frozen=True)
class GtScene:
    scene_id: str
    tier: str
    split: str
    board: str
    objects: tuple[GtObject, ...]

    @property
    def has_deformable(self) -> bool:
        """Deformables are post-MVP (D-014): the harness leaves the whole scene out."""
        return any(o.deformable for o in self.objects)


@dataclass(frozen=True)
class PredObject:
    object_id: str
    vertices: NDArray[np.float64]  # (N, 3), world frame, meters
    mass_kg: float | None = None


@dataclass(frozen=True)
class Prediction:
    objects: Sequence[PredObject]
    runtime_s: float | None = None
    notes: list[str] = field(default_factory=list)


def _triple(value: Any) -> tuple[float, float, float] | None:
    if value is None:
        return None
    a, b, c = (float(x) for x in value)
    return (a, b, c)


def parse_scene(data: dict[str, Any]) -> GtScene:
    objects = tuple(
        GtObject(
            object_id=str(o["object_id"]),
            label=str(o.get("label", "")),
            size_m=_triple(o.get("size_m")),
            position_m=_triple(o.get("position_m")),
            yaw_deg=None if o.get("yaw_deg") is None else float(o["yaw_deg"]),
            mass_kg=None if o.get("mass_kg") is None else float(o["mass_kg"]),
            incomplete=tuple(o.get("incomplete", ())),
            deformable=bool(o.get("deformable")),
        )
        for o in data["objects"]
    )
    return GtScene(
        scene_id=str(data["scene_id"]),
        tier=str(data["tier"]),
        split=str(data["split"]),
        board=str(data.get("board", "")),
        objects=objects,
    )


def load_scene(path: Path) -> GtScene:
    return parse_scene(json.loads(path.read_text(encoding="utf-8")))


def discover_scenes(
    root: Path, splits: Iterable[str], scene_id: str | None = None
) -> tuple[list[GtScene], list[str]]:
    """Scenes under ``root`` in the given splits, and the folders that have no ground truth.

    A scene folder without a readable ``ground_truth.json`` cannot be scored. It is listed
    instead of dropped, so a partial benchmark is never mistaken for a full one. With
    ``scene_id`` only that scene is considered, whatever its split.
    """
    wanted = set(splits)
    scenes: list[GtScene] = []
    missing: list[str] = []
    folders = sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
    for folder in folders:
        if scene_id is not None and folder.name != scene_id:
            continue
        path = folder / GT_FILE
        try:
            scene = load_scene(path)
        except (OSError, ValueError, KeyError, TypeError):
            missing.append(folder.name)
            continue
        if scene_id is not None or scene.split in wanted:
            scenes.append(scene)
    found = any(s.scene_id == scene_id for s in scenes)
    if scene_id is not None and not found and scene_id not in missing:
        missing.append(scene_id)
    return scenes, missing
