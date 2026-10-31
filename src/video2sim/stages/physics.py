"""Stage 5: physics preparation. Implemented in Phase 4 (P4.1 to P4.8).

Input:  ``RegisteredObjects`` and keyframe crops.
Output: ``SimAssets``: repaired visual mesh, collision geometry, mass, inertia, friction and
        restitution, each with low, nominal and high values.

The vision-language model picks a material class. Numbers come from
``configs/material_table.yaml``, never from the model.
"""

from __future__ import annotations

from dataclasses import dataclass

from video2sim.schemas import Estimate, Flag, ObjectInstance
from video2sim.stages.material_table import MaterialNumbers, MaterialTable


@dataclass(frozen=True)
class ObjectMaterial:
    """What the material pass said about one object, turned into numbers (P4.4)."""

    numbers: MaterialNumbers
    fill_fraction: Estimate
    is_hollow: bool
    confidence: float
    flags: tuple[Flag, ...]
    reasons: tuple[str, ...]


def object_material(
    obj: ObjectInstance, table: MaterialTable, min_confidence: float
) -> ObjectMaterial:
    """Map one object's class to table numbers and say how far to trust them (D-019).

    An unanswered, unknown or unconfident class, or a class whose density has no source, raises
    ``Flag.LOW_CONFIDENCE_MATERIAL``. Unsourced friction and restitution do not. ``hollow:
    unknown`` counts as solid with a reason, because ``PhysicalProperties.is_hollow`` is a bool.
    """
    numbers = table.lookup(obj.material_class)
    confidence = obj.material_confidence if obj.material_confidence is not None else 0.0
    reasons = []
    if numbers.fell_back:
        reasons.append(f"material class {obj.material_class} not in the table: used other")
    elif confidence < min_confidence:
        reasons.append(f"material confidence {confidence:.2f} below {min_confidence:.2f}")
    if numbers.density_unsourced and not numbers.fell_back:
        reasons.append(f"density of {numbers.material_class} has no source")
    flags = (Flag.LOW_CONFIDENCE_MATERIAL,) if reasons else ()
    if obj.hollow != "hollow" and obj.hollow != "solid":
        reasons.append(f"hollow is {obj.hollow}: treated as solid")
    is_hollow = obj.hollow == "hollow"
    fill = table.fill_fraction(numbers.material_class, is_hollow)
    return ObjectMaterial(numbers, fill, is_hollow, confidence, flags, tuple(reasons))


__all__ = ["ObjectMaterial", "object_material"]
