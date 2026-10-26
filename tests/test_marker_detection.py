from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from video2sim.stages.marker_detection import (
    TagView,
    detect_tags,
    triangulate_tag_corners,
)
from video2sim.stages.world_frame import BoardLayout, apply_scale, estimate_scale, fit_board

F64 = NDArray[np.float64]

PX_PER_MM = 6  # texture resolution; 70 mm * 6 splits into the 6 tag cells exactly
BOARD_MM = (600, 400)
CENTERS_MM = {0: (60, 60), 1: (540, 60), 2: (60, 340), 3: (540, 340), 4: (60, 200), 5: (540, 200)}
TAG_MM = 70


@pytest.fixture(scope="module")
def layout(tmp_path_factory: pytest.TempPathFactory) -> BoardLayout:
    p: Path = tmp_path_factory.mktemp("board") / "layout.json"
    data = {
        "tag_edge_mm_nominal": TAG_MM,
        "tags": {str(i): {"center_mm": list(c)} for i, c in CENTERS_MM.items()},
    }
    p.write_text(json.dumps(data), encoding="utf-8")
    return BoardLayout.from_make_board_json(p)


def _board_texture() -> NDArray[np.uint8]:
    w, h = BOARD_MM[0] * PX_PER_MM, BOARD_MM[1] * PX_PER_MM
    img = np.full((h, w), 255, np.uint8)
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    side = TAG_MM * PX_PER_MM // 6 * 6  # whole number of cells
    for tag_id, (cx, cy) in CENTERS_MM.items():
        tag = cv2.aruco.generateImageMarker(d, tag_id, side)
        x0 = int(cx * PX_PER_MM - side / 2)
        y0 = int((BOARD_MM[1] - cy) * PX_PER_MM - side / 2)  # board y is up, image rows go down
        img[y0 : y0 + side, x0 : x0 + side] = tag
    return img


def _look_at(eye: F64, target: F64) -> F64:
    """T_world_from_camera, OpenCV axes (+z forward, +y down), world +Z roughly up."""
    z = (target - eye) / np.linalg.norm(target - eye)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    t = np.eye(4)
    t[:3, :3] = np.column_stack([x, y, z])
    t[:3, 3] = eye
    return t


K = np.array([[1200.0, 0, 640], [0, 1200.0, 480], [0, 0, 1]])
SIZE = (1280, 960)


def _render(tex: NDArray[np.uint8], t_wc: F64) -> NDArray[np.uint8]:
    """Photograph the z = 0 board (metres) from a camera pose by warping the texture."""
    t_cw = np.linalg.inv(t_wc)
    h_img_from_board_m = K @ np.column_stack([t_cw[:3, 0], t_cw[:3, 1], t_cw[:3, 3]])
    tex_from_m = np.array(
        [[1000 * PX_PER_MM, 0, 0], [0, -1000 * PX_PER_MM, BOARD_MM[1] * PX_PER_MM], [0, 0, 1.0]]
    )
    h = h_img_from_board_m @ np.linalg.inv(tex_from_m)
    out = cv2.warpPerspective(tex, h, SIZE, flags=cv2.INTER_AREA, borderValue=128)
    return np.asarray(out, dtype=np.uint8)


POSES = [
    _look_at(np.array([0.3, -0.5, 0.9]), np.array([0.3, 0.2, 0.0])),
    _look_at(np.array([0.9, 0.7, 0.8]), np.array([0.3, 0.2, 0.0])),
    _look_at(np.array([-0.2, 0.6, 0.7]), np.array([0.3, 0.2, 0.0])),
    _look_at(np.array([0.6, 0.0, 1.1]), np.array([0.3, 0.2, 0.0])),
]


def _views(tex: NDArray[np.uint8], poses: list[F64]) -> list[TagView]:
    return [TagView(K, np.zeros(0), p, detect_tags(_render(tex, p))) for p in poses]


def test_detects_all_tags_in_layout_corner_order(layout: BoardLayout) -> None:
    tex = _board_texture()
    det = detect_tags(_render(tex, POSES[3]))
    assert sorted(det) == [0, 1, 2, 3, 4, 5]
    # Project the known board corners; detection must agree to sub-pixel, in the same order.
    t_cw = np.linalg.inv(POSES[3])
    for tag_id, board_corners in layout.tag_corners_m.items():
        cam = (t_cw[:3, :3] @ board_corners.T).T + t_cw[:3, 3]
        px = (K @ cam.T).T
        px = px[:, :2] / px[:, 2:]
        assert np.abs(px - det[tag_id]).max() < 1.0, tag_id


def test_unknown_family_is_an_error() -> None:
    with pytest.raises(ValueError, match="unknown marker family"):
        detect_tags(np.zeros((10, 10), np.uint8), family="april_36h11")


def test_blank_image_has_no_tags() -> None:
    assert detect_tags(np.full((200, 200), 255, np.uint8)) == {}


def test_end_to_end_scale_from_rendered_views(layout: BoardLayout) -> None:
    """Rendered photos -> detect -> triangulate -> fit: scale recovers a deliberately wrong one."""
    wrong_scale = 3.7  # the "reconstruction" uses units that are 3.7x too large
    poses = []
    for p in POSES:
        q = p.copy()
        q[:3, 3] *= wrong_scale
        poses.append(q)
    views = [
        TagView(K, np.zeros(0), q, v.corners_px)
        for q, v in zip(poses, _views(_board_texture(), POSES), strict=True)
    ]
    tri = triangulate_tag_corners(views)
    assert sorted(tri.corners) == [0, 1, 2, 3, 4, 5]
    assert tri.mean_reprojection_px < 0.5

    fit = fit_board(layout, tri.corners)
    assert abs(fit.scale * wrong_scale - 1.0) < 0.01  # within the 2% gate with margin
    d = apply_scale(tri.corners[1][0] - tri.corners[0][0], fit.scale)
    true = layout.tag_corners_m[1][0] - layout.tag_corners_m[0][0]
    assert abs(np.linalg.norm(d) / np.linalg.norm(true) - 1.0) < 0.02


def test_triangulation_needs_two_views(layout: BoardLayout) -> None:
    with pytest.raises(ValueError, match="marker not seen"):
        triangulate_tag_corners(_views(_board_texture(), POSES[:1]))


def test_outlier_detection_drops_that_tag(layout: BoardLayout) -> None:
    views = _views(_board_texture(), POSES)
    bad = dict(views[0].corners_px)
    bad[2] = bad[2] + np.array([40.0, -30.0])
    views[0] = TagView(K, np.zeros(0), views[0].T_world_from_camera, bad)
    tri = triangulate_tag_corners(views, max_reprojection_px=1.0)
    assert 2 not in tri.corners and 0 in tri.corners


def test_single_tag_gives_scale_without_any_board_layout() -> None:
    """Production case: one tag, only its printed edge known. Within the 2% gate."""
    wrong_scale = 3.7
    views = []
    for p, v in zip(POSES, _views(_board_texture(), POSES), strict=True):
        q = p.copy()
        q[:3, 3] *= wrong_scale
        views.append(TagView(K, np.zeros(0), q, {3: v.corners_px[3]}))
    tri = triangulate_tag_corners(views)
    assert list(tri.corners) == [3]
    est = estimate_scale(tri.corners, tag_edge_m=TAG_MM / 1000, max_spread=0.03)
    assert abs(est.scale * wrong_scale - 1.0) < 0.02
