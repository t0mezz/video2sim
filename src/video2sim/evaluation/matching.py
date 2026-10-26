"""Assign predicted objects to ground-truth objects (P0.6).

Hungarian assignment on center distance, with a per-ground-truth gate: a pair farther apart
than the gate can never match. Scenes hold a handful of objects, so a plain O(n^3) solver is
enough and avoids a SciPy dependency.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

GATE_MIN_M = 0.05
GATE_FRACTION = 0.5
_FORBIDDEN = 1e9


def gate_m(size_m: tuple[float, float, float]) -> float:
    """Half the smaller horizontal side of the ground-truth object, but at least 5 cm."""
    return max(GATE_MIN_M, GATE_FRACTION * min(size_m[0], size_m[1]))


def linear_sum_assignment(cost: NDArray[np.float64]) -> list[tuple[int, int]]:
    """Minimum-cost assignment of rows to columns (shortest augmenting paths)."""
    n_rows, n_cols = cost.shape
    if n_rows == 0 or n_cols == 0:
        return []
    transposed = n_rows > n_cols
    a = cost.T if transposed else cost
    n, m = a.shape  # n <= m
    u, v = np.zeros(n + 1), np.zeros(m + 1)
    owner = np.zeros(m + 1, dtype=int)  # owner[j] = 1-based row assigned to column j
    way = np.zeros(m + 1, dtype=int)
    for i in range(1, n + 1):
        owner[0] = i
        j0 = 0
        min_v = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = owner[j0]
            delta, j1 = np.inf, 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                reduced = a[i0 - 1, j - 1] - u[i0] - v[j]
                if reduced < min_v[j]:
                    min_v[j], way[j] = reduced, j0
                if min_v[j] < delta:
                    delta, j1 = min_v[j], j
            for j in range(m + 1):
                if used[j]:
                    u[owner[j]] += delta
                    v[j] -= delta
                else:
                    min_v[j] -= delta
            j0 = j1
            if owner[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            owner[j0] = owner[j1]
            j0 = j1
    pairs = [(int(owner[j]) - 1, j - 1) for j in range(1, m + 1) if owner[j]]
    pairs = [(c, r) for r, c in pairs] if transposed else pairs
    return sorted(pairs)


def match(
    gt_centers: NDArray[np.float64],
    gt_gates: NDArray[np.float64],
    pred_centers: NDArray[np.float64],
) -> list[tuple[int, int]]:
    """Pairs ``(gt_index, pred_index)`` within each ground-truth gate, minimum total distance."""
    if len(gt_centers) == 0 or len(pred_centers) == 0:
        return []
    dist = np.linalg.norm(gt_centers[:, None, :] - pred_centers[None, :, :], axis=2)
    cost = np.where(dist <= gt_gates[:, None], dist, _FORBIDDEN)
    return [(g, p) for g, p in linear_sum_assignment(cost) if cost[g, p] < _FORBIDDEN]
