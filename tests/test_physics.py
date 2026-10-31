from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from video2sim.schemas import Flag, ObjectInstance
from video2sim.stages.material_table import MaterialTable, load_material_table
from video2sim.stages.physics import object_material

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def table() -> MaterialTable:
    return load_material_table(ROOT / "configs/material_table.yaml")


def _obj(**fields: Any) -> ObjectInstance:
    base: dict[str, Any] = {
        "object_id": "obj_000",
        "label": "mug",
        "mask_dir": "segmentation/masks/obj_000",
        "point_cluster_path": "segmentation/clusters/obj_000.npy",
        "keyframes": [0],
        "material_class": "plastic_hard",
        "hollow": "solid",
        "material_confidence": 0.9,
    }
    return ObjectInstance(**{**base, **fields})


def test_sourced_density_keeps_the_class_and_raises_no_flag(table: MaterialTable) -> None:
    m = object_material(_obj(), table, 0.6)
    assert m.numbers.material_class == "plastic_hard"
    assert m.numbers.density_kg_m3.nominal == 1100
    assert not m.is_hollow and m.confidence == 0.9
    assert m.flags == () and m.reasons == ()


def test_unsourced_density_is_flagged(table: MaterialTable) -> None:
    m = object_material(_obj(material_class="foam"), table, 0.6)
    assert m.numbers.material_class == "foam"
    assert m.flags == (Flag.LOW_CONFIDENCE_MATERIAL,)
    assert "no source" in m.reasons[0]


def test_low_confidence_keeps_the_class_and_is_flagged(table: MaterialTable) -> None:
    m = object_material(_obj(material_confidence=0.58), table, 0.6)
    assert m.numbers.material_class == "plastic_hard"
    assert m.flags == (Flag.LOW_CONFIDENCE_MATERIAL,)


@pytest.mark.parametrize("name", [None, "unobtainium"])
def test_missing_class_falls_back_to_other(table: MaterialTable, name: str | None) -> None:
    m = object_material(_obj(material_class=name, material_confidence=None), table, 0.6)
    assert m.numbers.material_class == "other"
    assert m.confidence == 0.0
    assert m.flags == (Flag.LOW_CONFIDENCE_MATERIAL,)
    assert len(m.reasons) == 1


@pytest.mark.parametrize(
    ("answer", "hollow", "noted"),
    [
        ("hollow", True, False),
        ("solid", False, False),
        ("unknown", False, True),
        (None, False, True),
    ],
)
def test_hollow_answers(
    table: MaterialTable, answer: str | None, hollow: bool, noted: bool
) -> None:
    m = object_material(_obj(hollow=answer), table, 0.6)
    assert m.is_hollow is hollow
    assert bool(m.reasons) is noted
    assert m.flags == ()  # hollow alone says nothing about the material numbers


def test_fill_fraction_is_the_class_value_when_fitted(table: MaterialTable) -> None:
    assert (
        object_material(_obj(hollow="hollow"), table, 0.6).fill_fraction
        == table.materials["plastic_hard"].hollow_fill_fraction
    )
    assert (
        object_material(_obj(), table, 0.6).fill_fraction
        == table.materials["plastic_hard"].solid_fill_fraction
    )


@pytest.mark.parametrize(("answer", "hollow"), [("hollow", True), ("solid", False)])
def test_fill_fraction_falls_back_to_the_answer(
    table: MaterialTable, answer: str, hollow: bool
) -> None:
    fill = object_material(_obj(material_class="wood", hollow=answer), table, 0.6).fill_fraction
    assert fill == (table.hollow_fill_fraction if hollow else table.solid_fill_fraction)


def test_fill_fraction_above_one_is_refused(table: MaterialTable) -> None:
    raw = table.model_dump()
    raw["solid_fill_fraction"] = {"low": 0.5, "nominal": 1.0, "high": 1.5}
    with pytest.raises(ValueError, match="fill fraction"):
        MaterialTable.model_validate(raw)
