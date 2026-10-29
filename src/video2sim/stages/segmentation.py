"""Stage 3: segmentation, tracking, lifting to 3D (Phase 2). Shared types and helpers.

Input:  ``WorldModel`` and the frames.
Output: ``Instances``: per object, masks per frame, a point cluster in the world frame, the
        best keyframes for shape prediction, and failure flags.

The stage itself is ``segmentation_stage.SegmentationStage``. Objects are found by the SAM 3.1
video tracker (``tracking``) with a prompt the scene router picks; the per-frame discovery of
P2.1 was removed after it stopped feeding the pipeline (its last commit is 289bfe2).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

BoolMask = NDArray[np.bool_]


def sample_frame_indices(n_available: int, n_wanted: int) -> list[int]:
    """Positions of ``n_wanted`` frames spread evenly over ``n_available``, ends included."""
    if n_available < 1:
        return []
    if n_wanted >= n_available:
        return list(range(n_available))
    if n_wanted == 1:
        return [n_available // 2]
    return sorted({round(i * (n_available - 1) / (n_wanted - 1)) for i in range(n_wanted)})
