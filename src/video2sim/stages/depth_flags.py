"""Geometry evidence for unstable depth (P2.5): SfM and mask-lift statistics per track.

Reflective, glass and foil surfaces give noisy, short-lived SfM points and a hull that disagrees
with the masks. ``depth_signals`` measures that and ``judge`` turns it into ``flag`` (clearly
unstable), ``clear`` (clearly well conditioned) or ``uncertain``. It is one input of the flag
decision in ``material_pass``, which also asks a vision-language model about the surface finish
and cross-checks the two. Geometry alone misses a mirror cube and a foil figure, whose points
triangulate normally, which is why it does not decide alone.

The signals need the SfM model kept in ``reconstruction.work/sparse`` for the per-point error and
track length. Thresholds are guesses fitted on five reflective objects of two dev scenes
(``benchmark/results/p2_5_signals.json``).
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from video2sim.schemas import WorldModel
from video2sim.stages.lift import MIN_DEPTH_M, LiftConfig, _pixels, _views, load_points
from video2sim.stages.tracking import Track

F64 = NDArray[np.float64]
Verdict = Literal["flag", "clear", "uncertain"]

LOOSE_AGREEMENT = 0.5  # share of views a point may lie in the mask; looser than the P2.3 vote
HULL_MARGIN_M = 0.02
IOU_DOWNSCALE = 4
IOU_VIEWS = 20
SHORT_TRACK = 3  # an SfM point seen in this many images or fewer is short-lived
MIN_SEEN_VIEWS = 5


@dataclass(frozen=True)
class DepthConfig:
    """Values from ``segmentation.depth_flags`` in ``configs/default.yaml``."""

    min_points: int
    flag_err_ratio: float
    flag_short_frac: float
    clear_err_ratio: float
    clear_short_frac: float
    clear_hull_iou: float
    clear_outside_hull_frac: float

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> DepthConfig:
        c = cfg["segmentation"]["depth_flags"]
        out = cls(
            int(c["min_points"]),
            float(c["flag_err_ratio"]),
            float(c["flag_short_frac"]),
            float(c["clear_err_ratio"]),
            float(c["clear_short_frac"]),
            float(c["clear_hull_iou"]),
            float(c["clear_outside_hull_frac"]),
        )
        if out.min_points < 1:
            raise ValueError("min_points must be at least 1")
        if out.clear_err_ratio >= out.flag_err_ratio:
            raise ValueError("clear_err_ratio must be below flag_err_ratio")
        if out.clear_short_frac >= out.flag_short_frac:
            raise ValueError("clear_short_frac must be below flag_short_frac")
        return out


@dataclass(frozen=True)
class DepthSignals:
    """Raw measurements of one track. ``None`` means there was nothing to measure."""

    track_id: int
    sfm_points: int  # points the masks agree on, by the P2.3 vote
    err_ratio: float | None  # median reprojection error of the track's points over the table's
    short_frac: float | None  # share of the track's points seen in few images
    outside_hull_frac: float | None  # points in the mask but away from the hull: seen through it
    hull_iou: float | None  # median overlap of the projected hull and the mask
    err_med: float | None = None
    len_med: float | None = None
    table_err_med: float | None = None
    hull_surface_cm2: float | None = None
    points_per_100cm2: float | None = None


def sfm_attributes(run_dir: Path, n_points: int) -> tuple[F64, F64]:
    """Reprojection error and track length per point, in the order of the written PLY."""
    import pycolmap

    for model_dir in sorted((run_dir / "reconstruction.work" / "sparse").iterdir()):
        rec = pycolmap.Reconstruction(str(model_dir))
        if len(rec.points3D) == n_points:
            err = np.array([p.error for p in rec.points3D.values()], dtype=np.float64)
            length = np.array([p.track.length() for p in rec.points3D.values()], dtype=np.float64)
            return err, length
    raise ValueError("no SfM model matches the point cloud")


def _project(world: WorldModel, position: int, pts: F64) -> tuple[F64, F64, NDArray[np.bool_]]:
    k = world.intrinsics
    pose = world.frames[position].T_world_from_camera.to_numpy()
    cam = (pts - pose[:3, 3]) @ pose[:3, :3]
    z = cam[:, 2]
    ok = z > MIN_DEPTH_M
    safe = np.where(ok, z, 1.0)
    u, v = _pixels(cam[:, 0] / safe, cam[:, 1] / safe, k)
    ok &= (u >= 0) & (u < k.width_px) & (v >= 0) & (v < k.height_px)
    return u, v, ok


def _vote(
    world: WorldModel, tracks: Sequence[Track], points: F64, views: Sequence[int]
) -> NDArray[np.float64]:
    """Per track and point, the share of seeing views whose mask holds the point."""
    seen = np.zeros((len(tracks), len(points)))
    hits = np.zeros((len(tracks), len(points)))
    for pos in views:
        index = world.frames[pos].frame_index
        u, v, ok = _project(world, pos, points)
        for row, track in enumerate(tracks):
            mask = track.masks.get(index)
            if mask is None:
                continue
            h, w = mask.shape
            ui, vi = np.clip(u.astype(int), 0, w - 1), np.clip(v.astype(int), 0, h - 1)
            seen[row] += ok
            hits[row] += ok & mask[vi, ui]
    return np.where(seen >= MIN_SEEN_VIEWS, hits / np.maximum(seen, 1), 0.0)


def _hull_iou(world: WorldModel, track: Track, hull: F64, views: Sequence[int]) -> float | None:
    import cv2

    k = world.intrinsics
    w, h = k.width_px // IOU_DOWNSCALE, k.height_px // IOU_DOWNSCALE
    scores: list[float] = []
    for pos in views:
        mask = track.masks.get(world.frames[pos].frame_index)
        if mask is None or not len(hull):
            continue
        u, v, ok = _project(world, pos, hull)
        if ok.sum() < 20:
            continue
        raw = np.zeros((h, w), np.uint8)
        raw[(v[ok] / IOU_DOWNSCALE).astype(int), (u[ok] / IOU_DOWNSCALE).astype(int)] = 1
        img: Any = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        small = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        union = int(((img > 0) | (small > 0)).sum())
        if union:
            scores.append(float(((img > 0) & (small > 0)).sum() / union))
    return statistics.median(scores) if scores else None


def depth_signals(
    tracks: Sequence[Track],
    world: WorldModel,
    run_dir: Path,
    hulls: Mapping[int, F64],
    lift: LiftConfig,
) -> list[DepthSignals]:
    """Signals for every track. ``hulls`` are the visual-hull clusters of ``lift_tracks``."""
    if not tracks:
        return []
    points = load_points(run_dir / world.point_cloud_path)
    err, length = sfm_attributes(run_dir, len(points))
    views = _views(world, lift)
    ratio = _vote(world, tracks, points, views)
    height = points[:, 2] - world.table_plane.offset_m
    inside_band = (height > lift.table_margin_m) & (height < lift.max_height_m)
    on_table = np.abs(height) < lift.table_margin_m
    table_err = float(np.median(err[on_table])) if on_table.any() else None
    best, top = ratio.argmax(axis=0), ratio.max(axis=0)
    loose = (top >= LOOSE_AGREEMENT) & inside_band
    strict = (top >= lift.min_agreement) & inside_band
    iou_views = list(views[:: max(1, len(views) // IOU_VIEWS)])
    out: list[DepthSignals] = []
    for row, track in enumerate(tracks):
        hull = hulls.get(track.track_id, np.zeros((0, 3)))
        mine = loose & (best == row)
        n_strict = int((strict & (best == row)).sum())
        fields: dict[str, Any] = {"table_err_med": table_err}
        if len(hull):
            lo, hi = hull.min(axis=0), hull.max(axis=0)
            a, b, c = hi - lo
            surface_cm2 = float(2 * (a * b + b * c + a * c)) * 1e4
            fields["hull_surface_cm2"] = surface_cm2
            fields["points_per_100cm2"] = 100 * n_strict / max(surface_cm2, 1.0)
            if mine.any():
                p = points[mine]
                near = np.all((p >= lo - HULL_MARGIN_M) & (p <= hi + HULL_MARGIN_M), axis=1)
                fields["outside_hull_frac"] = float(1 - near.mean())
        if mine.any():
            fields["err_med"] = float(np.median(err[mine]))
            fields["len_med"] = float(np.median(length[mine]))
            fields["short_frac"] = float((length[mine] <= SHORT_TRACK).mean())
            if table_err:
                fields["err_ratio"] = fields["err_med"] / table_err
        out.append(
            DepthSignals(
                track_id=track.track_id,
                sfm_points=n_strict,
                err_ratio=fields.get("err_ratio"),
                short_frac=fields.get("short_frac"),
                outside_hull_frac=fields.get("outside_hull_frac"),
                hull_iou=_hull_iou(world, track, hull, iou_views),
                err_med=fields.get("err_med"),
                len_med=fields.get("len_med"),
                table_err_med=table_err,
                hull_surface_cm2=fields.get("hull_surface_cm2"),
                points_per_100cm2=fields.get("points_per_100cm2"),
            )
        )
    return out


def judge(s: DepthSignals, cfg: DepthConfig) -> tuple[Verdict, str]:
    """Geometry's verdict on one track, with the reason. Too little data is never ``clear``."""
    if (
        s.sfm_points < cfg.min_points
        or s.err_ratio is None
        or s.short_frac is None
        or s.outside_hull_frac is None
        or s.hull_iou is None
    ):
        return "uncertain", f"only {s.sfm_points} usable sparse points or no hull to compare"
    if s.err_ratio >= cfg.flag_err_ratio and s.short_frac >= cfg.flag_short_frac:
        return "flag", (
            f"point error {s.err_ratio:.2f}x the table's and {s.short_frac:.0%} short-lived points"
        )
    if (
        s.err_ratio <= cfg.clear_err_ratio
        and s.short_frac <= cfg.clear_short_frac
        and s.hull_iou >= cfg.clear_hull_iou
        and s.outside_hull_frac <= cfg.clear_outside_hull_frac
    ):
        return "clear", "stable points and hull"
    return "uncertain", "signals between the clear and flag limits"
