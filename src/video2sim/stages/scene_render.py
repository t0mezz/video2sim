"""Draw the scene from a real camera pose for the QA report (P5.3).

A small painter's renderer on OpenCV. It keeps the vertex colors, which MuJoCo's meshes cannot
show, and it projects with the pipeline's own OpenCV camera, so no axis flip stands between the
render and the frame it is compared with. Triangles are drawn far to near, which is good enough
to see that an object is in the right place; it is not a depth-correct image.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import cv2
import numpy as np
import trimesh
from numpy.typing import NDArray

from video2sim.schemas import Intrinsics

F64 = NDArray[np.float64]
UINT8 = NDArray[np.uint8]
BACKGROUND = (70, 70, 70)
NEAR_M = 0.01  # triangles closer to the camera than this are left out


@dataclass(frozen=True)
class RenderObject:
    mesh: trimesh.Trimesh  # in the object frame
    T_world_from_object: F64


def _colors(mesh: trimesh.Trimesh) -> UINT8:
    """RGB per face, gray for a mesh without colors."""
    if not isinstance(mesh.visual, trimesh.visual.ColorVisuals) or not mesh.visual.kind:
        return np.full((len(mesh.faces), 3), 160, dtype=np.uint8)
    rgb = np.asarray(mesh.visual.vertex_colors, dtype=float)[:, :3]
    return cast(UINT8, rgb[np.asarray(mesh.faces)].mean(axis=1).astype(np.uint8))


def render(objects: list[RenderObject], T_world_from_camera: F64, intrinsics: Intrinsics) -> UINT8:
    """A BGR image of ``intrinsics``' size."""
    image = np.full((intrinsics.height_px, intrinsics.width_px, 3), BACKGROUND, dtype=np.uint8)
    T_camera_from_world = np.linalg.inv(T_world_from_camera)
    matrix = np.array(
        [[intrinsics.fx, 0, intrinsics.cx], [0, intrinsics.fy, intrinsics.cy], [0, 0, 1.0]]
    )
    distortion = np.array((intrinsics.distortion + [0.0] * 5)[:5])
    depths, triangles, shades = [], [], []
    for item in objects:
        vertices = np.asarray(item.mesh.vertices, dtype=float)
        t = T_camera_from_world @ item.T_world_from_object
        camera = vertices @ t[:3, :3].T + t[:3, 3]
        faces = np.asarray(item.mesh.faces)
        corners = camera[faces]
        normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        lengths = np.maximum(np.linalg.norm(normals, axis=1), 1e-12)
        centers = corners.mean(axis=1)
        towards = -np.sum(normals * centers, axis=1)  # above zero when the face looks at the camera
        keep = (towards > 0) & (corners[:, :, 2].min(axis=1) > NEAR_M)
        if not keep.any():
            continue
        pixels = cv2.projectPoints(camera, np.zeros(3), np.zeros(3), matrix, distortion)[0]
        pixels = pixels.reshape(-1, 2)
        light = towards / (lengths * np.maximum(np.linalg.norm(centers, axis=1), 1e-12))
        shade = _colors(item.mesh)[keep] * (0.4 + 0.6 * light[keep, None])
        depths.append(centers[keep, 2])
        triangles.append(pixels[faces[keep]])
        shades.append(shade)
    if not depths:
        return image
    depth = np.concatenate(depths)
    corners_px = np.concatenate(triangles).round().astype(np.int32)
    bgr = np.concatenate(shades)[:, ::-1].astype(np.uint8)
    for k in np.argsort(-depth):
        color = (int(bgr[k, 0]), int(bgr[k, 1]), int(bgr[k, 2]))
        cv2.fillConvexPoly(image, corners_px[k], color, cv2.LINE_8)
    return image


def side_by_side(frame: UINT8, rendered: UINT8) -> UINT8:
    """The real frame on the left and the render, resized to it, on the right."""
    height, width = frame.shape[:2]
    resized = cv2.resize(rendered, (width, height), interpolation=cv2.INTER_AREA)
    return cast(UINT8, np.hstack([frame, resized]))


__all__ = ["RenderObject", "render", "side_by_side"]
