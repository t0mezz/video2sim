"""Material table loader and class-to-numbers mapping (P4.3, numbers half of P4.4).

``configs/material_table.yaml`` is the only source of material numbers. The vision-language model
picks a class (``stages/material_pass.py``); ``lookup`` turns it into density, friction and
restitution ``Estimate`` s. A value with no source is listed in the entry's ``unsourced`` field.
Friction and restitution are accepted as wide placeholders (D-019), so only an unsourced density
tells the caller to raise ``Flag.LOW_CONFIDENCE_MATERIAL``.

A class that is not in the table, or none at all, maps to the ``other`` entry: a wide range and an
``unsourced`` verdict, never a guess.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, model_validator

from video2sim.schemas import Estimate, Model

FALLBACK_CLASS = "other"
QUANTITIES = ("density_kg_m3", "friction_coefficient", "restitution")


class MaterialEntry(Model):
    density_kg_m3: Estimate
    friction_coefficient: Estimate
    restitution: Estimate
    sources: list[str] = Field(min_length=1)
    unsourced: list[str] = Field(default_factory=list)
    note: str = ""

    @model_validator(mode="after")
    def _checked(self) -> MaterialEntry:
        unknown = set(self.unsourced) - set(QUANTITIES)
        if unknown:
            raise ValueError(f"unsourced names unknown quantities: {sorted(unknown)}")
        if self.density_kg_m3.low <= 0:
            raise ValueError("density must be positive")
        if self.friction_coefficient.low < 0:
            raise ValueError("friction must not be negative")
        if not self.restitution.low >= 0 or self.restitution.high > 1:
            raise ValueError("restitution must lie in [0, 1]")
        if set(self.unsourced) != set(QUANTITIES) and self.sources == ["none"]:
            raise ValueError("an entry that names a source for no quantity must say 'none'")
        return self


@dataclass(frozen=True)
class MaterialNumbers:
    material_class: str
    density_kg_m3: Estimate
    friction_coefficient: Estimate
    restitution: Estimate
    unsourced: tuple[str, ...]
    fell_back: bool

    @property
    def density_unsourced(self) -> bool:
        return "density_kg_m3" in self.unsourced


class MaterialTable(Model):
    hollow_fill_fraction: Estimate
    materials: dict[str, MaterialEntry]

    @model_validator(mode="after")
    def _has_fallback(self) -> MaterialTable:
        if FALLBACK_CLASS not in self.materials:
            raise ValueError(f"the table needs a '{FALLBACK_CLASS}' entry")
        if self.hollow_fill_fraction.low <= 0 or self.hollow_fill_fraction.high > 1:
            raise ValueError("hollow_fill_fraction must lie in (0, 1]")
        return self

    def lookup(self, material_class: str | None) -> MaterialNumbers:
        fell_back = material_class not in self.materials
        key = FALLBACK_CLASS if fell_back or material_class is None else material_class
        e = self.materials[key]
        return MaterialNumbers(
            material_class=key,
            density_kg_m3=e.density_kg_m3,
            friction_coefficient=e.friction_coefficient,
            restitution=e.restitution,
            unsourced=tuple(e.unsourced) if not fell_back else QUANTITIES,
            fell_back=fell_back,
        )


def load_material_table(path: Path) -> MaterialTable:
    raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    return MaterialTable.model_validate(raw)


__all__ = ["MaterialEntry", "MaterialNumbers", "MaterialTable", "load_material_table"]
