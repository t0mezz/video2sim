"""Coverage-based router view selection (S12.9, task 9.3).

Every cell of the table workspace should be seen unoccluded by at least ``k`` of the frames
the router receives. The workspace is a grid over the XY footprint of the sparse cloud, the
height map carries the tallest point per cell (dilated by one cell so sparse points still
make a wall), and a cell counts as seen by a frame when the ray from the camera to the cell
centre at each probe height projects inside the image and stays above the height map in
every cell it crosses. Frames are picked by greedy set cover, which is within ``1 - 1/e``
of optimal for this monotone submodular objective. Ties go to the frame with the largest
azimuth gap from the chosen ones, then to the sharper frame, then to the lower frame
index, so the result is deterministic.

``compute_visibility`` builds the ``(n_frames, n_cells)`` matrix over all registered
frames; ``select_views`` picks router frames from it in coverage mode only. Gap fill calls
``compute_visibility`` itself, lazily and only with unexplained clusters, so in even mode
with gap fill off the matrix is never computed. ``run_dir`` is optional and only feeds the
sharpness tie-break; without it all frames tie on sharpness and the lower index wins.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from video2sim.schemas import WorldModel
from video2sim.stages.lift import MIN_DEPTH_M, _pixels
from video2sim.stages.segmentation import sample_frame_indices

F64 = NDArray[np.float64]


@dataclass(frozen=True)
class ViewConfig:
    """Values from ``segmentation.router.views`` in ``configs/default.yaml``."""

    mode: str  # even keeps today's behaviour; coverage is the set cover below
    count: int  # router views picked (K); the even fallback also uses it
    cell_m: float
    probe_heights_m: tuple[float, ...]
    min_views_per_cell: int  # k: views that must see a cell unoccluded
    workspace_margin_m: float
    max_image_px: int  # downscale of router images; used by the router (9.4), not here
    table_margin_m: float  # points below this above the plane are the table
    max_height_m: float  # points above this are not workspace

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> ViewConfig:
        v = cfg["segmentation"]["router"]["views"]
        out = cls(
            mode=str(v["mode"]),
            count=int(v["count"]),
            cell_m=float(v["cell_m"]),
            probe_heights_m=tuple(float(h) for h in v["probe_heights_m"]),
            min_views_per_cell=int(v["min_views_per_cell"]),
            workspace_margin_m=float(v["workspace_margin_m"]),
            max_image_px=int(v["max_image_px"]),
            table_margin_m=float(v["table_margin_m"]),
            max_height_m=float(v["max_height_m"]),
        )
        if out.mode not in ("even", "coverage"):
            raise ValueError("views.mode must be even or coverage")
        if out.count < 1 or out.min_views_per_cell < 1 or out.max_image_px < 1:
            raise ValueError("views.count, min_views_per_cell and max_image_px need >= 1")
        if out.cell_m <= 0 or out.workspace_margin_m < 0 or out.table_margin_m < 0:
            raise ValueError("views cell and margin sizes must not be negative")
        if out.max_height_m <= out.table_margin_m:
            raise ValueError("views.max_height_m must clear table_margin_m")
        if any(h <= 0 for h in out.probe_heights_m):
            raise ValueError("views.probe_heights_m must be positive")
        return out


@dataclass(frozen=True)
class ViewSelection:
    frames: list[int]  # frame_index values, in capture order
    coverage: float  # share of cells with >= min_views_per_cell chosen views
    cell_view_counts: NDArray[np.int32]
    reason: str = ""  # why coverage is 0, e.g. an empty workspace; empty when fine


def _workspace(
    cloud: F64, plane_n: F64, plane_d: float, cfg: ViewConfig
) -> tuple[F64, tuple[int, int], F64] | None:
    """Grid origin, (nx, ny) shape and height map, or None when no point qualifies."""
    heights = cloud @ plane_n - plane_d
    keep = (heights >= cfg.table_margin_m) & (heights <= cfg.max_height_m)
    if not keep.any():
        return None
    pts, hts = cloud[keep][:, :2], heights[keep]
    lo = pts.min(axis=0) - cfg.workspace_margin_m
    hi = pts.max(axis=0) + cfg.workspace_margin_m
    nx = max(1, int(np.ceil((hi[0] - lo[0]) / cfg.cell_m)))
    ny = max(1, int(np.ceil((hi[1] - lo[1]) / cfg.cell_m)))
    shape = (nx, ny)
    idx = np.clip(((pts - lo) / cfg.cell_m).astype(int), 0, np.array(shape) - 1)
    flat = idx[:, 1] * shape[0] + idx[:, 0]
    cells = np.full(int(np.prod(shape)), -np.inf)
    np.maximum.at(cells, flat, hts)
    height = cells.reshape(shape[1], shape[0])
    height[~np.isfinite(height)] = -np.inf
    # Dilate by one cell so sparse points still make a wall.
    padded = np.pad(height, 1, constant_values=-np.inf)
    dilated: F64 = np.full_like(height, -np.inf)
    for dy in range(3):
        for dx in range(3):
            dilated = np.maximum(dilated, padded[dy : dy + shape[1], dx : dx + shape[0]])
    return lo, shape, dilated


def _targets(
    lo: F64, shape: tuple[int, int], plane_n: F64, plane_d: float, cfg: ViewConfig
) -> tuple[F64, NDArray[np.int64]]:
    """Probe points (n_cells * n_probes, 3) above the plane and their flat cell indices."""
    nx, ny = shape
    cx = lo[0] + (np.arange(nx) + 0.5) * cfg.cell_m
    cy = lo[1] + (np.arange(ny) + 0.5) * cfg.cell_m
    gx, gy = np.meshgrid(cx, cy)
    cells = np.stack([gx.ravel(), gy.ravel()], axis=1)
    # Point on the plane under each centre, raised along the normal per probe height.
    pz = (plane_d - plane_n[0] * cells[:, 0] - plane_n[1] * cells[:, 1]) / plane_n[2]
    base = np.stack([cells[:, 0], cells[:, 1], pz], axis=1)
    probes = np.array(cfg.probe_heights_m, dtype=np.float64)
    points = base[:, None, :] + probes[None, :, None] * plane_n[None, None, :]
    flat = np.arange(cells.shape[0], dtype=np.int64)
    return points.reshape(-1, 3), np.repeat(flat, len(probes))


def compute_visibility(
    world: WorldModel, cloud: NDArray[np.float64], cfg: ViewConfig
) -> tuple[NDArray[np.bool_], F64, tuple[int, int]]:
    """``(n_frames, n_cells)`` visibility over all registered frames, plus grid origin/shape.

    A cell counts as seen when, at every probe height, its centre projects inside the image
    and the ray from the camera centre stays above the height map in every crossed cell but
    its own. Frames whose camera is not above the plane see nothing. An empty workspace
    gives an all-False matrix with an empty grid.
    """
    plane_n = np.array(world.table_plane.normal, dtype=np.float64)
    plane_d = float(world.table_plane.offset_m)
    grid = _workspace(np.asarray(cloud, dtype=np.float64), plane_n, plane_d, cfg)
    n = len(world.frames)
    if grid is None:
        return np.zeros((n, 0), dtype=bool), np.zeros(2), (0, 0)
    lo, shape, height = grid
    targets, target_cell = _targets(lo, shape, plane_n, plane_d, cfg)
    n_cells = shape[0] * shape[1]
    flat_height = height.ravel()
    out = np.zeros((n, n_cells), dtype=bool)
    for f, frame in enumerate(world.frames):
        pose = frame.T_world_from_camera.to_numpy()
        center = pose[:3, 3]
        if center @ plane_n - plane_d <= 0:
            continue
        rot = pose[:3, :3]
        rel = (targets - center) @ rot
        z = rel[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u, v = _pixels(rel[:, 0] / z, rel[:, 1] / z, world.intrinsics)
        inside = (
            (z > MIN_DEPTH_M)
            & (u >= 0)
            & (u < world.intrinsics.width_px)
            & (v >= 0)
            & (v < world.intrinsics.height_px)
        )
        rays = targets - center
        maxd = max(float(np.max(np.linalg.norm(rays, axis=1))), 0.0)
        steps = max(8, min(512, int(np.ceil(maxd / (cfg.cell_m / 2)))))
        t = np.linspace(0.0, 1.0, steps, endpoint=False)[1:]
        samples = center[None, None, :] + rays[:, None, :] * t[None, :, None]
        sample_h = samples @ plane_n - plane_d
        sample_cell = np.clip(
            ((samples[:, :, :2] - lo) / cfg.cell_m).astype(int), 0, np.array(shape) - 1
        )
        sample_flat = sample_cell[:, :, 1] * shape[0] + sample_cell[:, :, 0]
        blocked = (
            (sample_h < flat_height[sample_flat] - 1e-9) & (sample_flat != target_cell[:, None])
        ).any(axis=1)
        seen = (inside & ~blocked).reshape(n_cells, len(cfg.probe_heights_m)).all(axis=1)
        out[f] = seen
    return out, lo, shape


def _sharpness(run_dir: Path | None, world: WorldModel) -> list[float]:
    """Laplacian variance per clip position, as in ``keyframes.py``; zeros without images."""
    if run_dir is None:
        return [0.0] * len(world.frames)
    import cv2  # heavy import, kept out of module import time

    out = []
    for frame in world.frames:
        image = cv2.imread(str(run_dir / frame.image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            out.append(0.0)
            continue
        out.append(float(cv2.Laplacian(image, cv2.CV_64F).var()))
    return out


def select_views(
    world: WorldModel,
    cloud: NDArray[np.float64],
    cfg: ViewConfig,
    run_dir: Path | None = None,
) -> ViewSelection:
    """Router frames by greedy set cover, in coverage mode only (see module docstring)."""
    n = len(world.frames)
    visibility, lo, shape = compute_visibility(world, cloud, cfg)
    if visibility.shape[1] == 0:
        frames = [world.frames[i].frame_index for i in sample_frame_indices(n, cfg.count)]
        return ViewSelection(frames, 0.0, np.zeros(0, dtype=np.int32), "empty workspace")
    center_xy = lo + np.array(shape, dtype=np.float64) * cfg.cell_m / 2
    if cfg.count >= n:
        positions = list(range(n))
    else:
        azimuth = np.array(
            [
                np.arctan2(
                    f.T_world_from_camera.to_numpy()[1, 3] - center_xy[1],
                    f.T_world_from_camera.to_numpy()[0, 3] - center_xy[0],
                )
                for f in world.frames
            ]
        )
        sharp = _sharpness(run_dir, world)
        counts = np.zeros(visibility.shape[1], dtype=np.int32)
        chosen: list[int] = []
        for _ in range(cfg.count):
            best: tuple[float, float, float, int] | None = None
            best_pos = -1
            for pos in range(n):
                if pos in chosen:
                    continue
                gain = float(
                    np.minimum(
                        visibility[pos],
                        cfg.min_views_per_cell - counts.clip(max=cfg.min_views_per_cell),
                    ).sum()
                )
                gap = (
                    min(
                        abs((azimuth[pos] - azimuth[q] + np.pi) % (2 * np.pi) - np.pi)
                        for q in chosen
                    )
                    if chosen
                    else 0.0
                )
                key = (gain, float(gap), sharp[pos], -pos)
                if best is None or key > best:
                    best, best_pos = key, pos
            assert best_pos >= 0
            chosen.append(best_pos)
            counts += visibility[best_pos]
        positions = sorted(chosen)
    counts = visibility[positions].sum(axis=0).astype(np.int32)
    coverage = float((counts >= cfg.min_views_per_cell).mean())
    return ViewSelection([world.frames[i].frame_index for i in positions], coverage, counts)


__all__ = ["ViewConfig", "ViewSelection", "compute_visibility", "select_views"]
