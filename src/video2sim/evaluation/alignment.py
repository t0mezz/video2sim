"""Board frame from world frame, for grid-board scenes (P2.1 scoring).

Ground truth is in the board frame (origin at the grid corner). The pipeline's world frame has
its origin on a reference tag, and ``WorldModel`` does not record which one. Instead of
guessing the tag, the printed tags are detected again in the registered frames, triangulated
with the world-frame camera poses, and fitted to the known board layout. The fit does not
depend on which tag carried the origin, and the pipeline never sees the layout.

Clean-board scenes have no known tag layout. Their ground truth is measured from the bottom-left
corner of the reference tag as seen in the first frame where it is visible (human note,
2026-10-28), with X to the right along the tag's lower edge and Y up the tag, so ``align_clean``
rebuilds that frame from the tag alone.

``align_scene`` picks between the two for a benchmark scene, and returns None for a clean-board
scene whose origin is not established (s05), so that scene is reported as unscored.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from video2sim.schemas import SE3, WorldModel
from video2sim.stages.marker_detection import TagView, detect_tags, triangulate_tag_corners
from video2sim.stages.world_frame import BoardLayout, fit_board

F64 = NDArray[np.float64]

# World units are meters, so a fit far from scale 1 means a wrong tag match, not a real scale.
MAX_SCALE_DEVIATION = 0.05
# Clean-board scenes whose ground truth lines up with the tag-0 corner frame (human note,
# 2026-10-28), checked by overlaying the ground-truth boxes on the frames. s05 does not line up
# under that rule or its three 90-degree rotations, so it stays unscored until its origin is known.
CLEAN_ALIGNED = ("s14", "s24")
# The A1 benchmark grid board as ``benchmark/make_board.py`` draws it (page, printer margin and
# grid pitch in mm); ``tests/test_eval_alignment.py`` checks the tag positions against that script.
_A1_PAGE_MM = (841.0, 594.0)
_PAGE_MARGIN_MM = 10.0
_GRID_MM = 50.0


@dataclass(frozen=True)
class BoardAlignment:
    T_board_from_world: F64  # (4, 4), rigid
    scale: float  # fitted scale, expected close to 1
    rms_m: float  # residual of the tag corners against the layout
    tag_ids: tuple[int, ...]

    def pose_in_board(self, T_world_from_camera: SE3) -> SE3:
        return SE3.from_numpy(self.T_board_from_world @ T_world_from_camera.to_numpy())


def grid_board_layout(tag_edge_m: float) -> BoardLayout:
    """Tag corners of the A1 grid board, ids 0 to 5, in the board frame (origin at the grid corner).

    The positions follow ``make_board.build("A1", tag_edge_m * 1000, False)``, as the benchmark
    scripts have always built them: the measured edge sets the quiet zone, so the positions move
    by under 0.1 mm against the 100 mm print, well under the board fit's millimetre residual.
    """
    tag_mm = tag_edge_m * 1000.0
    quiet = max(tag_mm / 6.0, 5.0)
    grid_w = math.floor((_A1_PAGE_MM[0] - 2 * _PAGE_MARGIN_MM) / _GRID_MM) * _GRID_MM
    grid_h = math.floor((_A1_PAGE_MM[1] - 2 * _PAGE_MARGIN_MM) / _GRID_MM) * _GRID_MM
    left, right = quiet, grid_w - quiet - tag_mm
    bottom, top = quiet, grid_h - quiet - tag_mm
    mid = (grid_h - tag_mm) / 2.0
    lower_left = {0: (left, bottom), 1: (right, bottom), 2: (left, top), 3: (right, top),
                  4: (left, mid), 5: (right, mid)}  # fmt: skip
    centers = {i: (x + tag_mm / 2.0, y + tag_mm / 2.0) for i, (x, y) in lower_left.items()}
    return BoardLayout.from_tag_centers(centers, tag_edge_m)


def align_views(views: Sequence[TagView], layout: BoardLayout) -> BoardAlignment:
    """Fit the board layout to tags triangulated from views whose poses are in the world frame."""
    tri = triangulate_tag_corners(list(views))
    fit = fit_board(layout, tri.corners)
    if abs(fit.scale - 1.0) > MAX_SCALE_DEVIATION:
        raise ValueError(f"board fit has scale {fit.scale:.3f}; tag ids or poses are wrong")
    t = np.eye(4)
    t[:3, :3] = fit.R_board_from_recon
    t[:3, 3] = fit.t_board_from_recon
    return BoardAlignment(t, fit.scale, fit.rms_m, tuple(sorted(tri.corners)))


def align_run(
    world: WorldModel, run_dir: Path, layout: BoardLayout, family: str = "aruco_4x4_50"
) -> BoardAlignment:
    """Detect the tags in every registered frame of a run and align to the board layout."""
    import cv2  # heavy import, kept out of module import time

    k = world.intrinsics
    camera = np.array([[k.fx, 0.0, k.cx], [0.0, k.fy, k.cy], [0.0, 0.0, 1.0]])
    distortion = np.array(k.distortion, dtype=np.float64)
    views = []
    for frame in world.frames:
        gray = cv2.imread(str(run_dir / frame.image_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f"cannot read frame {run_dir / frame.image_path}")
        views.append(
            TagView(
                camera,
                distortion,
                frame.T_world_from_camera.to_numpy(),
                detect_tags(np.asarray(gray, dtype=np.uint8), family),
            )
        )
    return align_views(views, layout)


def _visual_corners(corners_px: F64) -> tuple[int, int, int]:
    """Indices of the tag corners that look bottom-left, bottom-right and top-left in an image."""
    by_height = np.argsort(corners_px[:, 1])  # image y grows downward
    bottom = sorted(by_height[2:], key=lambda i: corners_px[i, 0])
    top = sorted(by_height[:2], key=lambda i: corners_px[i, 0])
    return int(bottom[0]), int(bottom[1]), int(top[0])


def board_from_tag_view(corners_world: F64, corners_px: F64) -> F64:
    """``T_board_from_world`` for a board whose origin is the tag's visual bottom-left corner.

    ``corners_world`` is (4, 3), the tag corners in the world frame; ``corners_px`` is (4, 2),
    the same corners in one image, in the same order. X runs along the visual bottom edge to
    the right, projected into the table plane (world +Z is up), Y is Z cross X, and the origin
    sits on the table plane under the corner. The image must look at the tag from above, as the
    capture protocol asks, or "bottom-left" is ambiguous.
    """
    ll, lr, ul = _visual_corners(corners_px)
    x_edge = corners_world[lr] - corners_world[ll]
    x_axis = np.array([x_edge[0], x_edge[1], 0.0])
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross([0.0, 0.0, 1.0], x_axis)
    up_edge = corners_world[ul] - corners_world[ll]
    if float(up_edge[:2] @ y_axis[:2]) <= 0.0:
        raise ValueError("tag looks mirrored: its left edge does not run up the board")
    t_world_from_board = np.eye(4)
    t_world_from_board[:3, :3] = np.column_stack([x_axis, y_axis, [0.0, 0.0, 1.0]])
    t_world_from_board[:3, 3] = [corners_world[ll][0], corners_world[ll][1], 0.0]
    return np.asarray(np.linalg.inv(t_world_from_board), dtype=np.float64)


def align_clean(
    world: WorldModel, run_dir: Path, family: str = "aruco_4x4_50", origin_tag: int = 0
) -> BoardAlignment:
    """Align a clean-board scene on ``origin_tag`` (marker 0, the bottom-left tag of the board).

    The pipeline's reference tag can differ from it (s14 puts the world origin on tag 1), so the
    tag is named here and not read from the world frame.
    """
    import cv2  # heavy import, kept out of module import time

    k = world.intrinsics
    camera = np.array([[k.fx, 0.0, k.cx], [0.0, k.fy, k.cy], [0.0, 0.0, 1.0]])
    distortion = np.array(k.distortion, dtype=np.float64)
    views = []
    for frame in world.frames:
        gray = cv2.imread(str(run_dir / frame.image_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f"cannot read frame {run_dir / frame.image_path}")
        views.append(
            TagView(
                camera,
                distortion,
                frame.T_world_from_camera.to_numpy(),
                detect_tags(np.asarray(gray, dtype=np.uint8), family),
            )
        )
    tri = triangulate_tag_corners(views)
    ref = origin_tag
    if ref not in tri.corners:
        raise ValueError(f"tag {ref} was not triangulated; seen: {sorted(tri.corners)}")
    first = next(v for v in views if ref in v.corners_px)
    t = board_from_tag_view(tri.corners[ref], first.corners_px[ref])
    return BoardAlignment(t, 1.0, 0.0, (ref,))


def align_scene(
    world: WorldModel,
    run_dir: Path,
    scene_id: str,
    board: str,
    tag_edge_m: float,
    family: str = "aruco_4x4_50",
) -> BoardAlignment | None:
    """``T_board_from_world`` for a benchmark scene: the board fit on a grid board, the tag-0
    frame on a clean board listed in ``CLEAN_ALIGNED``, None for any other clean board."""
    if board == "grid":
        return align_run(world, run_dir, grid_board_layout(tag_edge_m), family)
    if scene_id in CLEAN_ALIGNED:
        return align_clean(world, run_dir, family)
    return None
