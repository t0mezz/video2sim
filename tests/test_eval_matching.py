from __future__ import annotations

import itertools

import numpy as np
import pytest
from numpy.typing import NDArray

from video2sim.evaluation.matching import gate_m, linear_sum_assignment, match


def _brute_force_cost(cost: NDArray[np.float64]) -> float:
    n, m = cost.shape
    k = min(n, m)
    best = np.inf
    for rows in itertools.combinations(range(n), k):
        for cols in itertools.permutations(range(m), k):
            best = min(best, sum(cost[r, c] for r, c in zip(rows, cols, strict=True)))
    return float(best)


@pytest.mark.parametrize("shape", [(1, 1), (3, 3), (2, 5), (5, 2), (4, 6), (6, 4)])
def test_assignment_is_optimal(shape: tuple[int, int]) -> None:
    rng = np.random.default_rng(sum(shape))
    for _ in range(20):
        cost = rng.uniform(0, 10, shape)
        pairs = linear_sum_assignment(cost)
        assert len(pairs) == min(shape)
        assert len({r for r, _ in pairs}) == len(pairs)
        assert len({c for _, c in pairs}) == len(pairs)
        assert sum(cost[r, c] for r, c in pairs) == pytest.approx(_brute_force_cost(cost))


def test_empty_inputs() -> None:
    assert linear_sum_assignment(np.empty((0, 3))) == []
    assert match(np.empty((0, 3)), np.empty(0), np.zeros((2, 3))) == []


def test_gate_scales_with_footprint_but_has_a_floor() -> None:
    assert gate_m((0.04, 0.03, 0.1)) == 0.05
    assert gate_m((0.30, 0.20, 0.1)) == pytest.approx(0.10)


def test_match_respects_the_gate_and_prefers_global_optimum() -> None:
    gt = np.array([[0.0, 0.0, 0.0], [0.10, 0.0, 0.0]])
    gates = np.array([0.06, 0.06])
    # The first prediction is nearest to both; the second is only in gate of the first.
    pred = np.array([[0.05, 0.0, 0.0], [-0.04, 0.0, 0.0]])
    assert match(gt, gates, pred) == [(0, 1), (1, 0)]
    far = np.array([[1.0, 1.0, 0.0]])
    assert match(gt, gates, far) == []
