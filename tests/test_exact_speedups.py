"""The speedups of branch ``perf-exact-speedups`` against the code they replaced.

Each ``_old_*`` function is the implementation as of commit c6ee24d, copied here so that the
faster version can be compared with it on synthetic inputs. Every comparison is exact: the
rewrites change how the work is done, not a single number of the result.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import trimesh
import yaml
from numpy.typing import NDArray

from video2sim.models.hosted import SamRecord, decode_sam_mask, decode_sam_masks
from video2sim.schemas import SE3, CameraFrame, Intrinsics, Plane, WorldModel
from video2sim.stages import lift, primitive_fit
from video2sim.stages.keyframes import FrameStats, KeyframeConfig, frame_stats
from video2sim.stages.lift import MIN_DEPTH_M, LiftConfig, _grid, _pixels, _views
from video2sim.stages.mesh_repair import _edge_use
from video2sim.stages.primitive_fit import PrimitiveConfig, _cost, _track_views, initial_params
from video2sim.stages.track_repair import BASE_FRACTION, mask_footprint
from video2sim.stages.tracking import Track

ROOT = Path(__file__).resolve().parent.parent
CONFIG = yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
K = Intrinsics(fx=500, fy=500, cx=320, cy=240, width_px=640, height_px=480)
BoolMask = NDArray[np.bool_]


def _look_at(position: NDArray[np.float64], target: NDArray[np.float64]) -> SE3:
    z = target - position
    z /= np.linalg.norm(z)
    right = np.cross(z, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    pose = np.eye(4)
    pose[:3, :3] = np.stack([right, np.cross(z, right), z], axis=1)
    pose[:3, 3] = position
    return SE3.from_numpy(pose)


def _world(views: int = 10) -> WorldModel:
    frames = []
    for i in range(views):
        a = 2 * np.pi * i / views
        position = np.array([0.55 * np.cos(a), 0.55 * np.sin(a), 0.45])
        pose = _look_at(position, np.array([0.0, 0.0, 0.03]))
        frames.append(CameraFrame(frame_index=i, image_path=f"f/{i}.png", T_world_from_camera=pose))
    return WorldModel(
        intrinsics=K,
        frames=frames,
        scale_factor=1.0,
        scale_source="marker",
        table_plane=Plane(normal=(0.0, 0.0, 1.0), offset_m=0.0),
        point_cloud_path="reconstruction/points.ply",
        mean_reprojection_error_px=0.5,
        registered_fraction=1.0,
    )


def _box_masks(
    world: WorldModel, size: tuple[float, float, float], xy: tuple[float, float], skip: int = 0
) -> dict[int, BoolMask]:
    """Masks of an upright box in every frame but every ``skip``-th (0 keeps all)."""
    import cv2

    sx, sy, sz = size
    offset = np.array([xy[0], xy[1], 0.0])
    corners = offset + np.array(
        [[x, y, z] for x in (-sx / 2, sx / 2) for y in (-sy / 2, sy / 2) for z in (0.0, sz)]
    )
    out: dict[int, BoolMask] = {}
    for f in world.frames:
        if skip and f.frame_index % skip == 0:
            continue
        pose = f.T_world_from_camera.to_numpy()
        cam = (corners - pose[:3, 3]) @ pose[:3, :3]
        u = K.fx * cam[:, 0] / cam[:, 2] + K.cx
        v = K.fy * cam[:, 1] / cam[:, 2] + K.cy
        canvas = np.zeros((K.height_px, K.width_px), dtype=np.uint8)
        points: Any = np.round(np.c_[u, v]).astype(np.int32)
        hull: Any = cv2.convexHull(points)
        cv2.fillConvexPoly(canvas, hull, 1)
        out[f.frame_index] = canvas.astype(bool)
    return out


def _scene() -> tuple[WorldModel, list[Track]]:
    """Three boxes that hide each other in some views, one tracked in only some frames, and one
    large enough to leave the image in some views."""
    world = _world()
    return world, [
        Track(0, _box_masks(world, (0.08, 0.06, 0.10), (0.0, 0.0))),
        Track(3, _box_masks(world, (0.05, 0.05, 0.05), (0.07, 0.02), skip=3)),
        Track(7, _box_masks(world, (0.30, 0.10, 0.04), (-0.05, -0.12))),
    ]


# --- SAM mask decoding ---------------------------------------------------------------------


def _records(count: int, corrupt: int | None = None) -> list[SamRecord]:
    from meta_sam_parser._mask_codec import _encode_raster  # test-only use of the encoder

    rng = np.random.default_rng(3)
    out = []
    for i in range(count):
        rows, cols = (int(n) for n in rng.integers(2, 9, 2))
        raster = (rng.random(rows * cols) < 0.5).astype(int).tolist()
        payload = "!not a payload" if i == corrupt else _encode_raster(raster, cols, rows)
        x1, y1 = (float(n) for n in rng.integers(0, 20, 2))
        w, h = (float(n) for n in rng.integers(1, 12, 2))
        box = (x1, y1, x1 + w, y1 + h)
        out.append(SamRecord(i, i % 3, box, (48, 40), (0, 0), (rows, cols), payload))
    return out


def test_parallel_decoding_gives_the_serial_masks() -> None:
    records = _records(24)
    serial = [decode_sam_mask(r) for r in records]
    parallel = decode_sam_masks(records, max_workers=2)
    assert len(parallel) == len(serial)
    assert all(
        a.dtype == b.dtype and np.array_equal(a, b) for a, b in zip(serial, parallel, strict=True)
    )


def test_a_corrupt_mask_raises_the_parsers_error_from_the_parallel_path() -> None:
    from meta_sam_parser import InvalidSegmentationMaskError

    with pytest.raises(InvalidSegmentationMaskError):
        decode_sam_masks(_records(20, corrupt=11), max_workers=2)


# --- lift: the voxel vote ------------------------------------------------------------------


def _old_vote(
    centers: NDArray[np.float64],
    tracks: list[Track],
    world: WorldModel,
    cfg: LiftConfig,
    views: list[int],
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    import cv2

    k = world.intrinsics
    scale = 1.0 / cfg.mask_downscale
    seen = np.zeros((len(tracks), len(centers)), dtype=np.int32)
    hits = np.zeros((len(tracks), len(centers)), dtype=np.int32)
    for position in views:
        frame = world.frames[position]
        pose = frame.T_world_from_camera.to_numpy()
        cam = (centers - pose[:3, 3]) @ pose[:3, :3]
        z = cam[:, 2]
        ok = z > MIN_DEPTH_M
        safe = np.where(ok, z, 1.0)
        u, v = _pixels(cam[:, 0] / safe, cam[:, 1] / safe, k)
        u, v = np.where(ok, u, -1.0), np.where(ok, v, -1.0)
        ok &= (u >= 0) & (u < k.width_px) & (v >= 0) & (v < k.height_px)
        ui = np.clip((u * scale).astype(np.int64), 0, None)
        vi = np.clip((v * scale).astype(np.int64), 0, None)
        smalls: list[BoolMask | None] = []
        for track in tracks:
            mask = track.masks.get(frame.frame_index)
            if mask is None:
                smalls.append(None)
                continue
            smalls.append(
                cv2.resize(
                    mask.astype(np.uint8), None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
                ).astype(bool)
            )
        combined: BoolMask | None = None
        if cfg.occlusion_aware:
            present = [s for s in smalls if s is not None]
            if present:
                combined = np.asarray(np.stack(present).any(axis=0), dtype=bool)
        for row, small in enumerate(smalls):
            if small is None:
                continue
            own = small[np.minimum(vi, small.shape[0] - 1), np.minimum(ui, small.shape[1] - 1)]
            use = ok
            if combined is not None:
                occluded = (
                    combined[
                        np.minimum(vi, combined.shape[0] - 1), np.minimum(ui, combined.shape[1] - 1)
                    ]
                    & ~own
                )
                use = ok & ~occluded
            seen[row] += use
            hits[row] += use & own
    ratio = np.where(seen >= cfg.min_views, hits / np.maximum(seen, 1), 0.0)
    return ratio, ratio.argmax(axis=0)


@pytest.mark.parametrize("occlusion_aware", [True, False])
def test_the_voxel_vote_is_unchanged(occlusion_aware: bool) -> None:
    world, tracks = _scene()
    cfg = replace(
        LiftConfig.from_config(CONFIG), occlusion_aware=occlusion_aware, min_views=3, margin_m=0.05
    )
    feet = np.array([[-0.2, -0.2], [0.2, 0.1]])
    centers, _ = _grid(feet, world.table_plane, cfg, cfg.max_height_m)
    views = _views(world, cfg)
    old_ratio, old_best = _old_vote(centers, tracks, world, cfg, views)
    new_ratio, new_best = lift._vote(centers, tracks, world, cfg, views)
    assert old_ratio.any()  # the scene exercises the vote
    assert np.array_equal(old_ratio, new_ratio) and np.array_equal(old_best, new_best)


# --- keyframes and shape flags: per-frame mask statistics -----------------------------------


def _old_frame_stats(
    frame_index: int,
    mask: BoolMask,
    others: BoolMask | None,
    gray: NDArray[np.uint8] | None,
    cfg: KeyframeConfig,
) -> FrameStats | None:
    import cv2

    area = int(mask.sum())
    if area < cfg.min_mask_px:
        return None
    height, width = mask.shape
    band = max(1, round(cfg.border_band_frac * min(height, width)))
    edge = np.zeros_like(mask)
    edge[:band], edge[-band:], edge[:, :band], edge[:, -band:] = True, True, True, True
    border = float((mask & edge).sum()) / area
    rows, cols = np.any(mask, axis=1), np.any(mask, axis=0)
    y_lo, y_hi = int(np.argmax(rows)), int(height - 1 - np.argmax(rows[::-1]))
    x_lo, x_hi = int(np.argmax(cols)), int(width - 1 - np.argmax(cols[::-1]))
    r = cfg.occluder_ring_px
    y0, y1 = max(0, y_lo - r - 1), min(height, y_hi + r + 2)
    x0, x1 = max(0, x_lo - r - 1), min(width, x_hi + r + 2)
    crop = mask[y0:y1, x0:x1].astype(np.uint8)
    occlusion = 0.0
    if others is not None:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        ring = (cv2.dilate(crop, kernel) > 0) & (crop == 0)
        ring &= ~edge[y0:y1, x0:x1]
        ring_px = int(ring.sum())
        if ring_px:
            occlusion = float((ring & others[y0:y1, x0:x1]).sum()) / ring_px
    sharpness = 0.0
    if gray is not None:
        inner = cv2.erode(crop, np.ones((3, 3), np.uint8), iterations=2) > 0
        inner = inner if inner.any() else crop > 0
        lap = cv2.Laplacian(gray[y0:y1, x0:x1], cv2.CV_64F)
        sharpness = float(lap[inner].var())
    return FrameStats(frame_index, area, border, occlusion, sharpness)


def test_frame_statistics_are_unchanged() -> None:
    from video2sim.stages.keyframes import KeyframeConfig

    cfg = KeyframeConfig.from_config(CONFIG)
    world, tracks = _scene()
    gray = np.random.default_rng(5).integers(0, 255, (K.height_px, K.width_px)).astype(np.uint8)
    edge_cut = np.zeros((K.height_px, K.width_px), dtype=bool)
    edge_cut[:40, 600:] = True  # in the corner band, so border and ring meet the image edge
    compared = 0
    for frame in world.frames:
        masks = [t.masks[frame.frame_index] for t in tracks if frame.frame_index in t.masks]
        masks.append(edge_cut)
        for i, mask in enumerate(masks):
            rest = masks[:i] + masks[i + 1 :]
            union = np.any(np.stack(rest), axis=0)
            for g in (gray, None):
                old = _old_frame_stats(frame.frame_index, mask, union, g, cfg)
                assert frame_stats(frame.frame_index, mask, rest, g, cfg) == old
                assert frame_stats(frame.frame_index, mask, union, g, cfg) == old
                none = _old_frame_stats(frame.frame_index, mask, None, g, cfg)
                assert frame_stats(frame.frame_index, mask, [], g, cfg) == none
                compared += old is not None
    empty = np.zeros_like(edge_cut)
    assert frame_stats(0, empty, [], None, cfg) is None
    assert compared > 20


# --- track repair: mask footprints ---------------------------------------------------------


def _old_footprint_u(mask: BoolMask) -> tuple[float, float] | None:
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return None
    bottom = int(np.max(rows))
    near = rows >= bottom - BASE_FRACTION * (bottom - int(np.min(rows)) + 1)
    return float(cols[near].mean()), float(bottom)


def test_mask_footprints_are_unchanged() -> None:
    world, tracks = _scene()
    rng = np.random.default_rng(8)
    masks = [m for t in tracks for m in t.masks.values()]
    masks += [rng.random((K.height_px, K.width_px)) < p for p in (0.001, 0.02)]
    for k, mask in enumerate(masks):
        pose = world.frames[k % len(world.frames)].T_world_from_camera
        expected = _old_footprint_u(mask)
        assert expected is not None
        u, v = expected
        n = np.asarray(world.table_plane.normal, dtype=np.float64)
        p = pose.to_numpy()
        ray = p[:3, :3] @ np.array([(u - K.cx) / K.fx, (v - K.cy) / K.fy, 1.0])
        s = (world.table_plane.offset_m - float(n @ p[:3, 3])) / float(n @ ray)
        got = mask_footprint(mask, pose, K, world.table_plane)
        if s <= 0:
            assert got is None
        else:
            assert got is not None and np.array_equal(got, (p[:3, 3] + s * ray)[:2])


# --- primitive fit: the mask cost ----------------------------------------------------------


def _old_cost(
    kind: Any, params: NDArray[np.float64], views: Any, world: WorldModel, cfg: PrimitiveConfig
) -> float:
    import cv2

    if float(np.min(params[3:])) < cfg.min_size_m:
        return float("inf")
    outline = primitive_fit._outline(kind, params, world.table_plane.offset_m)
    total = 0.0
    for view in views:
        cam = (outline - view.pose[:3, 3]) @ view.pose[:3, :3]
        if bool(np.any(cam[:, 2] <= MIN_DEPTH_M)):
            return float("inf")
        u, v = _pixels(cam[:, 0] / cam[:, 2], cam[:, 1] / cam[:, 2], world.intrinsics)
        uv = (np.stack([u, v], axis=1) / cfg.mask_downscale).astype(np.float32)
        hull: Any = cv2.convexHull(uv).astype(np.int32)
        sil = np.zeros(view.mask.shape, dtype=np.uint8)
        cv2.fillConvexPoly(sil, hull, 1)
        inside = sil.astype(bool)
        extra = int(np.count_nonzero(inside & view.free))
        missed = int(np.count_nonzero(view.mask & ~inside))
        total += (extra + missed) / view.area
    return total / len(views)


def test_the_primitive_mask_cost_is_unchanged() -> None:
    world, tracks = _scene()
    lift_cfg = LiftConfig.from_config(CONFIG)
    cfg = replace(PrimitiveConfig.from_config(CONFIG), min_mask_px=1)
    rng = np.random.default_rng(2)
    shared: dict[tuple[int, int], BoolMask] = {}
    for track in tracks:
        views = _track_views(track, tracks, world, lift_cfg, cfg, shared)
        alone = _track_views(track, tracks, world, lift_cfg, cfg)
        assert len(views) == len(alone) > 0
        for a, b in zip(views, alone, strict=True):
            assert np.array_equal(a.mask, b.mask) and np.array_equal(a.free, b.free)
        start = initial_params(np.array([[-0.03, -0.02, 0.0], [0.03, 0.02, 0.08]]), 0.0)
        for kind in ("box", "cylinder", "sphere"):
            for _ in range(15):
                params = start + rng.normal(0.0, 1.0, 6) * [0.05, 0.05, 1.0, 0.05, 0.05, 0.05]
                params[3:] = np.abs(params[3:]) + 0.006
                assert _cost(kind, params, views, world, cfg) == _old_cost(
                    kind, params, views, world, cfg
                )
        far = start.copy()
        far[:2] = [3.0, 3.0]  # off every image: every mask pixel is missed
        assert _cost("box", far, views, world, cfg) == _old_cost("box", far, views, world, cfg)


# --- mesh repair: edge use counts ----------------------------------------------------------


def test_edge_use_counts_match_the_row_wise_unique() -> None:
    sphere = trimesh.creation.icosphere(subdivisions=3)
    torn = trimesh.Trimesh(sphere.vertices, sphere.faces[40:], process=False)  # boundary edges
    doubled = trimesh.Trimesh(  # a face repeated: edges shared by more than two faces
        sphere.vertices, np.vstack([sphere.faces, sphere.faces[:5]]), process=False
    )
    for mesh in (sphere, torn, doubled):
        _, inverse, counts = np.unique(
            mesh.edges_sorted, axis=0, return_inverse=True, return_counts=True
        )
        assert np.array_equal(_edge_use(mesh), counts[inverse.reshape(-1)])
