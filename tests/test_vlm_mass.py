from __future__ import annotations

import pytest

from video2sim.schemas import Estimate
from video2sim.stages.vlm_mass import combine, parse_mass


def test_parse_reads_grams_as_kilograms_around_other_text() -> None:
    text = (
        'Sure: {"contents": "full", "mass_low_g": 100, "mass_nominal_g": 200, "mass_high_g": 300}'
    )
    assert parse_mass(text) == Estimate(low=0.1, nominal=0.2, high=0.3)


@pytest.mark.parametrize(
    "text",
    [
        "None",
        "{}",
        '{"mass_low_g": 0, "mass_nominal_g": 1, "mass_high_g": 2}',
        '{"mass_low_g": 3, "mass_nominal_g": 1, "mass_high_g": 2}',
        '{"mass_low_g": "a", "mass_nominal_g": 1, "mass_high_g": 2}',
    ],
)
def test_parse_refuses_what_is_not_a_range(text: str) -> None:
    assert parse_mass(text) is None


def test_combine_follows_the_rule() -> None:
    geometry = Estimate(low=0.1, nominal=0.2, high=0.4)
    inside = Estimate(low=0.15, nominal=0.3, high=0.5)
    outside = Estimate(low=0.5, nominal=0.6, high=0.7)
    assert combine(geometry, inside) == (geometry, "geometry")
    assert combine(geometry, outside) == (outside, "model")
    assert combine(geometry, None) == (geometry, "geometry_no_answer")
    assert combine(geometry, outside, enabled=False) == (geometry, "geometry_only")
