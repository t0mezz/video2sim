from __future__ import annotations

import numpy as np
import pytest

from video2sim.evaluation.alignment import align_views, board_from_tag_view
from video2sim.stages.marker_detection import TagView
from video2sim.stages.world_frame import BoardLayout

K = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
FLIP = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])  # camera looking straight down


def _tag(x: float, y: float, edge: float = 0.1) -> np.ndarray:
    return np.array(
        [[x, y, 0.0], [x + edge, y, 0.0], [x + edge, y + edge, 0.0], [x, y + edge, 0.0]]
    )


LAYOUT = BoardLayout({0: _tag(0.0, 0.0), 1: _tag(0.5, 0.0), 2: _tag(0.0, 0.4)})


def _yaw(deg: float, shift: tuple[float, float]) -> np.ndarray:
    a = np.radians(deg)
    t = np.eye(4)
    t[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
    t[:2, 3] = shift
    return t


def _views(t_world_from_board: np.ndarray, scale: float = 1.0) -> list[TagView]:
    views = []
    for cx, cy in [(0.1, 0.1), (0.4, 0.1), (0.1, 0.3), (0.4, 0.3), (0.25, 0.2)]:
        cam_board = np.eye(4)
        cam_board[:3, :3] = FLIP
        cam_board[:3, 3] = [cx, cy, 0.6]
        cam_world = t_world_from_board @ cam_board
        inv = np.linalg.inv(cam_world)
        px = {}
        for tag_id, corners in LAYOUT.tag_corners_m.items():
            world = (t_world_from_board @ np.c_[corners * scale, np.ones(4)].T).T[:, :3]
            cam = (inv[:3, :3] @ world.T).T + inv[:3, 3]
            uv = (K @ cam.T).T
            px[tag_id] = uv[:, :2] / uv[:, 2:3]
        views.append(TagView(K, np.zeros(0), cam_world, px))
    return views


def test_alignment_recovers_the_board_frame() -> None:
    t_world_from_board = _yaw(30.0, (0.3, -0.2))
    result = align_views(_views(t_world_from_board), LAYOUT)
    assert result.tag_ids == (0, 1, 2)
    assert result.rms_m < 1e-6
    assert np.allclose(result.T_board_from_world, np.linalg.inv(t_world_from_board), atol=1e-6)


def test_alignment_rejects_a_wrong_scale() -> None:
    with pytest.raises(ValueError, match="scale"):
        align_views(_views(_yaw(0.0, (0.0, 0.0)), scale=1.3), LAYOUT)


def _project_down(points: np.ndarray) -> np.ndarray:
    """Pixels of world points seen by a camera 0.6 m above the origin looking straight down."""
    cam = (FLIP @ (points - [0.0, 0.0, 0.6]).T).T
    uv = (K @ cam.T).T
    return np.asarray(uv[:, :2] / uv[:, 2:3], dtype=np.float64)


def test_clean_board_origin_is_the_visual_bottom_left_corner() -> None:
    # Tag corners in its own order (LL, LR, UR, UL), turned on the table so that its own
    # lower-left corner (0.2, 0.3) looks top-left. The camera looks straight down with image
    # up equal to world +y, so the visual bottom-left corner is the second one, (0.2, 0.2).
    corners = np.array([[0.2, 0.3, 0.0], [0.2, 0.2, 0.0], [0.3, 0.2, 0.0], [0.3, 0.3, 0.0]])
    t = board_from_tag_view(corners, _project_down(corners))
    board = [(t @ np.append(c, 1.0))[:3] for c in corners]
    assert np.allclose(board[1], [0.0, 0.0, 0.0], atol=1e-9)  # origin
    assert np.allclose(board[2], [0.1, 0.0, 0.0], atol=1e-9)  # +X along the visual bottom edge
    assert np.allclose(board[0], [0.0, 0.1, 0.0], atol=1e-9)  # +Y up the visual left edge


def test_clean_board_alignment_rejects_a_mirrored_view() -> None:
    corners = np.array([[0.2, 0.3, 0.0], [0.2, 0.2, 0.0], [0.3, 0.2, 0.0], [0.3, 0.3, 0.0]])
    mirrored = _project_down(corners)
    mirrored[:, 0] = 640.0 - mirrored[:, 0]
    with pytest.raises(ValueError, match="mirrored"):
        board_from_tag_view(corners, mirrored)
