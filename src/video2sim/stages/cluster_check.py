"""Per-cluster VLM check (S12.9 redesign, steps 4-5).

A residual cluster is not yet an object: it can be the board, a sheet, a
shadow or a piece of wall. Before any video request is spent on it, two
cropped views of the cluster go to the vision-language model, one call per
cluster, in parallel. Only a cluster the model accepts (a real object, with
confidence and a noun that survives the denylist) earns the single box
request of step 6. A failed or unreadable call is ``unchecked``, never a
silent drop: the stage records it and flags the scene.

The crop helpers also own the projected prompt box (``frame_box``), which
the box request reuses, so a view the check approved is the view the box is
prompted on.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from video2sim.models.hosted import ResponseCache, file_digest
from video2sim.models.vlm import cache_model
from video2sim.schemas import WorldModel
from video2sim.stages.lift import MIN_DEPTH_M, _pixels
from video2sim.stages.scene_router import RouterClient, sanitise_noun
from video2sim.stages.tracking import Track

logger = logging.getLogger(__name__)

F64 = NDArray[np.float64]
Box = tuple[int, int, int, int]  # inclusive x1, y1, x2, y2 in media pixels
VerdictStatus = Literal["accepted", "rejected", "unchecked"]


@dataclass(frozen=True)
class ClusterCheckConfig:
    """Values from ``segmentation.tracking.gap_fill.cluster_check``; guesses until 9.8."""

    explain_ratio: float = 0.5  # tau_explain of step 1
    accept_ratio: float = 0.5  # tau_accept of step 7 (median cluster vote)
    table_cell_m: float = 0.05
    table_dilate_m: float = 0.10
    board_margin_m: float = 0.02
    views_per_cluster: int = 2
    min_azimuth_gap_deg: float = 25.0
    box_pad_px: int = 8
    crop_margin_frac: float = 0.5  # crop margin, times the box's long side
    crop_min_margin_px: int = 16
    crop_max_px: int = 512  # crops are resized to at most this long side
    crop_dim: float = 0.35  # kept-mask pixels are multiplied by this in the crop
    min_confidence: float = 0.7
    max_workers: int = 4

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> ClusterCheckConfig:
        g = cfg["segmentation"]["tracking"]["gap_fill"].get("cluster_check", {})
        region = g.get("table_region", {})
        crop = g.get("crop", {})
        out = cls(
            explain_ratio=float(g.get("explain_ratio", 0.5)),
            accept_ratio=float(g.get("accept_ratio", 0.5)),
            table_cell_m=float(region.get("cell_m", 0.05)),
            table_dilate_m=float(region.get("dilate_m", 0.10)),
            board_margin_m=float(g.get("board_margin_m", 0.02)),
            views_per_cluster=int(g.get("views_per_cluster", 2)),
            min_azimuth_gap_deg=float(g.get("min_azimuth_gap_deg", 25.0)),
            box_pad_px=int(g.get("box_pad_px", 8)),
            crop_margin_frac=float(crop.get("margin_frac", 0.5)),
            crop_min_margin_px=int(crop.get("min_margin_px", 16)),
            crop_max_px=int(crop.get("max_px", 512)),
            crop_dim=float(crop.get("dim", 0.35)),
            min_confidence=float(g.get("min_confidence", 0.7)),
            max_workers=int(g.get("max_workers", 4)),
        )
        if not 0.0 < out.explain_ratio <= 1.0 or not 0.0 < out.accept_ratio <= 1.0:
            raise ValueError("cluster_check ratios must be in (0, 1]")
        if not 0.0 <= out.min_confidence <= 1.0:
            raise ValueError("cluster_check min_confidence must be in [0, 1]")
        if out.views_per_cluster < 1 or out.max_workers < 1 or out.crop_max_px < 1:
            raise ValueError("cluster_check counts must be at least 1")
        if min(out.table_cell_m, out.table_dilate_m, out.board_margin_m) < 0:
            raise ValueError("cluster_check lengths must not be negative")
        if not 0.0 <= out.crop_dim <= 1.0 or out.crop_margin_frac < 0:
            raise ValueError("cluster_check crop values are out of range")
        return out


def project_points(points: F64, pose: F64, k: Any) -> tuple[F64, F64, NDArray[np.bool_]]:
    """Pixel coordinates and the in-front mask of world points in one frame."""
    rot, center = pose[:3, :3], pose[:3, 3]
    cam = (points - center) @ rot
    z = cam[:, 2]
    ok = z > MIN_DEPTH_M
    u = np.full(len(points), -1.0)
    v = np.full(len(points), -1.0)
    if ok.any():
        pu, pv = _pixels(cam[ok, 0] / z[ok], cam[ok, 1] / z[ok], k)
        u[ok], v[ok] = pu, pv
    return u, v, ok


def frame_box(
    world: WorldModel, pos: int, points: F64, pad_px: int
) -> tuple[Box, tuple[float, float] | None] | None:
    """Prompt box and median point on one clip position, or None when unusable.

    The box is the 2D bounding box of the projected eight AABB corners plus their drops to
    the table plane, padded and clamped; None when a corner is behind the camera or the
    padded box is cut by the image border. The median point is omitted when behind.
    """
    frame = world.frames[pos]
    k = world.intrinsics
    pose = frame.T_world_from_camera.to_numpy()
    plane_n = np.array(world.table_plane.normal, dtype=np.float64)
    plane_d = float(world.table_plane.offset_m)
    lo, hi = points.min(axis=0), points.max(axis=0)
    corners = np.array(np.meshgrid(*zip(lo, hi, strict=True))).reshape(3, -1).T
    drops = corners - plane_n[None, :] * ((corners @ plane_n - plane_d)[:, None])
    u, v, ok = project_points(np.concatenate([corners, drops]), pose, k)
    if not ok.all():
        return None
    x1, y1 = int(np.min(u)), int(np.min(v))
    x2, y2 = int(np.max(u)), int(np.max(v))
    if x2 <= x1 or y2 <= y1:
        return None
    padded = (x1 - pad_px, y1 - pad_px, x2 + pad_px, y2 + pad_px)
    clamped = (
        max(padded[0], 0),
        max(padded[1], 0),
        min(padded[2], k.width_px - 1),
        min(padded[3], k.height_px - 1),
    )
    if clamped != padded or clamped[2] <= clamped[0] or clamped[3] <= clamped[1]:
        return None
    median = np.median(points, axis=0)
    mu, mv, mok = project_points(median[None, :], pose, k)
    point = (float(mu[0]), float(mv[0])) if mok[0] else None
    return clamped, point


def azimuths(world: WorldModel, centroid: F64) -> NDArray[np.float64]:
    """Camera azimuth around a point, per clip position."""
    return np.array(
        [
            np.arctan2(
                f.T_world_from_camera.to_numpy()[1, 3] - centroid[1],
                f.T_world_from_camera.to_numpy()[0, 3] - centroid[0],
            )
            for f in world.frames
        ]
    )


def _far_enough(
    pos: int, placed: Sequence[int], azimuth: NDArray[np.float64], gap_deg: float
) -> bool:
    return all(
        abs((azimuth[pos] - azimuth[q] + np.pi) % (2 * np.pi) - np.pi) >= np.deg2rad(gap_deg)
        for q in placed
    )


@dataclass(frozen=True)
class ClusterView:
    """One chosen view of a cluster: clip position, frame index and prompt box."""

    pos: int
    frame_index: int
    box: Box
    point: tuple[float, float] | None = None  # projected cluster median, a positive point


def choose_views(
    world: WorldModel,
    points: F64,
    kept: Sequence[Track],
    cfg: ClusterCheckConfig,
) -> list[ClusterView]:
    """Views for one cluster (step 4), best first.

    Frames rank by whether the box fits fully in the frame, then by how
    little of the box the kept masks cover, then by box area. Picks are at
    least ``min_azimuth_gap_deg`` apart around the cluster centroid, falling
    through to the next frame until ``views_per_cluster`` boxes fit; a frame
    whose box is cut by the border never qualifies.
    """
    scored: list[tuple[float, int, Box, tuple[float, float] | None]] = []
    for pos, frame in enumerate(world.frames):
        placed = frame_box(world, pos, points, cfg.box_pad_px)
        if placed is None:
            continue
        box, point = placed
        x1, y1, x2, y2 = box
        area = float((x2 - x1 + 1) * (y2 - y1 + 1))
        covered = 0.0
        for track in kept:
            mask = track.masks.get(frame.frame_index)
            if mask is None:
                continue
            covered = max(covered, float(mask[y1 : y2 + 1, x1 : x2 + 1].mean()) if area else 0.0)
        scored.append((covered, pos, box, point))
    # Rank: least covered first, then largest box; pos breaks ties deterministically.
    ranked = sorted(
        scored,
        key=lambda s: (
            s[0],
            -float((s[2][2] - s[2][0] + 1) * (s[2][3] - s[2][1] + 1)),
            s[1],
        ),
    )
    azimuth = azimuths(world, np.median(points, axis=0))
    picked: list[ClusterView] = []
    for _covered, pos, box, point in ranked:
        if len(picked) >= cfg.views_per_cluster:
            break
        if not _far_enough(pos, [v.pos for v in picked], azimuth, cfg.min_azimuth_gap_deg):
            continue
        picked.append(ClusterView(pos, world.frames[pos].frame_index, box, point))
    return picked


def make_crop(
    frame_path: Path,
    view: ClusterView,
    kept: Sequence[Track],
    out_path: Path,
    cfg: ClusterCheckConfig,
) -> Path:
    """The VLM crop of one cluster view (step 4).

    Margin is ``margin_frac`` times the box's long side, at least
    ``min_margin_px``. Pixels under kept masks are dimmed by ``crop_dim`` so
    the unchecked cluster stands out, the cluster box is drawn, and the crop
    is resized to at most ``crop_max_px`` on its long side.
    """
    import cv2  # heavy import, kept out of module import time

    frame = cv2.imread(str(frame_path))
    if frame is None:
        raise ValueError(f"cannot read {frame_path}")
    x1, y1, x2, y2 = view.box
    margin = max(cfg.crop_min_margin_px, round(cfg.crop_margin_frac * max(x2 - x1, y2 - y1)))
    cx1, cy1 = max(x1 - margin, 0), max(y1 - margin, 0)
    cx2, cy2 = min(x2 + margin, frame.shape[1] - 1), min(y2 + margin, frame.shape[0] - 1)
    crop = frame[cy1 : cy2 + 1, cx1 : cx2 + 1].copy()
    for track in kept:
        mask = track.masks.get(view.frame_index)
        if mask is None:
            continue
        if mask.shape != frame.shape[:2]:
            mask = cv2.resize(
                mask.astype("uint8"),
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        region = mask[cy1 : cy2 + 1, cx1 : cx2 + 1]
        crop[region] = (crop[region].astype(np.float64) * cfg.crop_dim).astype(np.uint8)
    cv2.rectangle(crop, (x1 - cx1, y1 - cy1), (x2 - cx1, y2 - cy1), (0, 255, 255), 2)
    long_side = max(crop.shape[:2])
    if long_side > cfg.crop_max_px:
        scale = cfg.crop_max_px / long_side
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), crop)
    return out_path


def build_check_prompt() -> str:
    """Ask whether the boxed cluster is a real object, as JSON and nothing else."""
    return (
        "The images are crops of one tabletop scene. A yellow box marks a cluster of "
        "3D points that no tracked object explains; dimmed regions are objects that "
        "are already tracked. Answer with a single JSON object and nothing else, "
        'with four keys: "is_object" (true only when the box holds one distinct '
        'physical object resting on the surface), "noun" (one short singular noun '
        'for it, or null), "count" (how many such objects the box holds), and '
        '"confidence" (a number from 0 to 1). The board, a sheet of paper, a '
        "shadow, a reflection, a hand or arm, table texture and background are "
        "not objects: answer is_object false for them."
    )


@dataclass(frozen=True)
class ClusterVerdict:
    """The VLM's answer for one cluster; ``unchecked`` when no answer could be read."""

    status: VerdictStatus
    noun: str | None = None
    count: int | None = None
    confidence: float | None = None
    reason: str = ""


def parse_check_answer(
    text: str, denylist: tuple[str, ...], min_confidence: float
) -> ClusterVerdict:
    """Strict parsing of one cluster answer; anything unreadable is unchecked."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match is None:
        return ClusterVerdict("unchecked", reason="unreadable answer")
    try:
        raw = json.loads(match.group(0))
    except ValueError:
        return ClusterVerdict("unchecked", reason="unreadable answer")
    if not isinstance(raw, dict) or not isinstance(raw.get("is_object"), bool):
        return ClusterVerdict("unchecked", reason="unreadable answer")
    confidence = raw.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return ClusterVerdict("unchecked", reason="unreadable answer")
    conf = float(confidence)
    if not raw["is_object"]:
        return ClusterVerdict("rejected", confidence=conf, reason="not an object")
    noun = sanitise_noun(str(raw.get("noun") or "").strip().lower(), denylist)
    if noun is None:
        return ClusterVerdict("rejected", confidence=conf, reason="noun rejected or empty")
    count = raw.get("count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        count = 1
    if conf < min_confidence:
        return ClusterVerdict(
            "rejected", noun=noun, count=count, confidence=conf, reason="low confidence"
        )
    return ClusterVerdict("accepted", noun=noun, count=count, confidence=conf)


def check_one(
    crops: Sequence[Path],
    client: RouterClient | None,
    cache: ResponseCache,
    cfg: ClusterCheckConfig,
    denylist: tuple[str, ...],
) -> ClusterVerdict:
    """One VLM call for one cluster, cached on the crop bytes.

    No client, a failed call or an unreadable answer gives ``unchecked``;
    this never raises, so a cluster is never dropped without a record.
    """
    if client is None:
        return ClusterVerdict("unchecked", reason="no VLM client")
    if not crops:
        return ClusterVerdict("unchecked", reason="no crop to check")
    prompt = build_check_prompt()
    parts = [prompt, cache_model(client), *[file_digest(p) for p in crops]]
    try:
        text = cache.get_or_call(parts, lambda: client.ask(prompt, list(crops)))
    except Exception:  # the check must not abort the stage
        logger.warning("cluster check call failed; cluster unchecked", exc_info=True)
        return ClusterVerdict("unchecked", reason="VLM call failed")
    return parse_check_answer(text, denylist, cfg.min_confidence)


@dataclass(frozen=True)
class ClusterCheck:
    """One residual cluster with its views, crop files and verdict."""

    index: int
    points: F64
    views: tuple[ClusterView, ...]
    crops: tuple[Path, ...]
    verdict: ClusterVerdict


def check_clusters(
    world: WorldModel,
    run_dir: Path,
    clusters: Sequence[F64],
    kept: Sequence[Track],
    client: RouterClient | None,
    cache: ResponseCache,
    work_dir: Path,
    cfg: ClusterCheckConfig,
    denylist: tuple[str, ...],
) -> list[ClusterCheck]:
    """Views, crops and verdicts for every cluster, in cluster order.

    One VLM call per cluster, in a thread pool of ``max_workers``; a cluster
    with no placeable view is unchecked with no call.
    """

    def run_one(index: int, points: F64) -> ClusterCheck:
        views = choose_views(world, points, kept, cfg)
        if not views:
            return ClusterCheck(
                index, points, (), (), ClusterVerdict("unchecked", reason="no placeable view")
            )
        crops = tuple(
            make_crop(
                run_dir / world.frames[v.pos].image_path,
                v,
                kept,
                work_dir / f"cluster_{index:02d}_view{v.pos:02d}.png",
                cfg,
            )
            for v in views
        )
        verdict = check_one(crops, client, cache, cfg, denylist)
        return ClusterCheck(index, points, tuple(views), crops, verdict)

    if len(clusters) > 1 and cfg.max_workers > 1:
        with ThreadPoolExecutor(max_workers=cfg.max_workers) as pool:
            return list(pool.map(run_one, range(len(clusters)), clusters))
    return [run_one(i, c) for i, c in enumerate(clusters)]


__all__ = [
    "Box",
    "ClusterCheck",
    "ClusterCheckConfig",
    "ClusterVerdict",
    "ClusterView",
    "azimuths",
    "build_check_prompt",
    "check_clusters",
    "check_one",
    "choose_views",
    "frame_box",
    "make_crop",
    "parse_check_answer",
    "project_points",
]
