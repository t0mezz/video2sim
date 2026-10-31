from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from video2sim.stages.material_table import (
    QUANTITIES,
    MaterialTable,
    load_material_table,
)

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def table() -> MaterialTable:
    return load_material_table(ROOT / "configs/material_table.yaml")


def test_keys_equal_the_material_pass_classes(table: MaterialTable) -> None:
    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    assert set(table.materials) == set(cfg["segmentation"]["material_pass"]["classes"])


def test_every_value_has_a_source_or_is_marked_unsourced(table: MaterialTable) -> None:
    for name, e in table.materials.items():
        if e.sources == ["none"]:
            assert set(e.unsourced) == set(QUANTITIES), name
        else:
            assert set(e.unsourced) < set(QUANTITIES) or name == "other", name


def test_known_class_maps_to_its_numbers(table: MaterialTable) -> None:
    n = table.lookup("wood")
    assert (n.density_kg_m3.low, n.density_kg_m3.nominal, n.density_kg_m3.high) == (336, 560, 784)
    assert n.unsourced == ("restitution",) and not n.fell_back


def test_unknown_or_missing_class_falls_back_wide_and_unsourced(table: MaterialTable) -> None:
    for c in ("unobtainium", None):
        n = table.lookup(c)
        assert n.fell_back and n.material_class == "other"
        assert set(n.unsourced) == set(QUANTITIES)


def test_only_an_unsourced_density_counts_against_a_class(table: MaterialTable) -> None:
    assert not table.lookup("plastic_hard").density_unsourced  # friction, restitution placeholders
    assert table.lookup("foam").density_unsourced
    assert table.lookup(None).density_unsourced


def test_a_table_without_the_fallback_is_rejected() -> None:
    entry = {
        "density_kg_m3": {"low": 1, "nominal": 2, "high": 3},
        "friction_coefficient": {"low": 0.1, "nominal": 0.2, "high": 0.3},
        "restitution": {"low": 0.1, "nominal": 0.2, "high": 0.3},
        "sources": ["x"],
    }
    with pytest.raises(ValidationError):
        MaterialTable.model_validate({"materials": {"wood": entry}})


def test_bad_ranges_are_rejected() -> None:
    base = {
        "density_kg_m3": {"low": 1, "nominal": 2, "high": 3},
        "friction_coefficient": {"low": 0.1, "nominal": 0.2, "high": 0.3},
        "restitution": {"low": 0.1, "nominal": 0.2, "high": 0.3},
        "sources": ["x"],
    }
    for patch in (
        {"restitution": {"low": 0.5, "nominal": 0.9, "high": 1.2}},
        {"density_kg_m3": {"low": 3, "nominal": 2, "high": 1}},
        {"unsourced": ["mass"]},
    ):
        with pytest.raises(ValidationError):
            MaterialTable.model_validate({"materials": {"other": {**base, **patch}}})
