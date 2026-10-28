"""Phase 2 exit-criteria metrics: clusters, identity switches, failure flags (P2).

Definitions (agreed 2026-10-27):

* Recall and cluster bounding-box error reuse the shape-stage matching (D-012), applied to the
  point clusters in ``Instances``. The box gate is the median size error over matched objects
  on the easy tier, compared with ``gates.segmentation.cluster_bbox_error_max``.
* Ground truth holds no track identities, so they are derived. Each ground-truth box is
  projected into a frame with the registered camera pose, and a track mask is assigned to the
  ground-truth object whose projected box it overlaps most (IoU above ``MIN_ASSIGN_IOU``).
  A switch is a change of assigned object between consecutive assigned frames of one track.
  Occlusion and lens distortion are ignored, so the count is approximate on cluttered scenes.
* Flags: criterion (a). A listed hard scene passes when at least one object carries a flag.
  Whether the flag is the right one is judged by eye from the per-object listing.
"""

from __future__ import annotations

import itertools
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from video2sim.evaluation.geometry import box_vertices
from video2sim.evaluation.metrics import SceneResult
from video2sim.evaluation.scenes import GtObject, Prediction, PredObject
from video2sim.runner import OUTPUT_FILE
from video2sim.schemas import SE3, Instances, Intrinsics

# Below this a mask only grazes a projected box, which is as likely a neighbour as the object.
MIN_ASSIGN_IOU = 0.3
# The four hard scenes of the Phase 2 flag criterion (human decision, 2026-10-27).
FLAG_SCENES = ("s17", "s18", "s19", "s20")
EASY_TIER = "easy"

BoolMask = NDArray[np.bool_]


@dataclass(frozen=True)
class SegmentationRun:
    """What ``Instances`` adds to a ``Prediction``: flags per object, by object id."""

    prediction: Prediction
    flags: Mapping[str, tuple[str, ...]]


def _load_points(path: Path) -> NDArray[np.float64]:
    if path.suffix == ".npy":
        return np.asarray(np.load(path), dtype=np.float64)
    import trimesh  # heavy import, kept out of module import time

    cloud: Any = trimesh.load(path)
    return np.asarray(cloud.vertices, dtype=np.float64)


def load_instances(run_dir: Path) -> SegmentationRun:
    """Read ``segmentation/output.json`` and the point clusters of a run folder."""
    instances = Instances.model_validate_json(
        (run_dir / "segmentation" / OUTPUT_FILE).read_text(encoding="utf-8")
    )
    objects = [
        PredObject(o.object_id, _load_points(run_dir / o.point_cluster_path))
        for o in instances.objects
    ]
    flags = {o.object_id: tuple(f.value for f in o.flags) for o in instances.objects}
    return SegmentationRun(Prediction(objects), flags)


def project_box_mask(
    obj: GtObject,
    intrinsics: Intrinsics,
    T_world_from_camera: SE3,
    min_depth_m: float = 0.05,
) -> BoolMask:
    """Image mask of a ground-truth box seen by a camera; empty if any corner is behind it."""
    import cv2  # heavy import, kept out of module import time

    shape = (intrinsics.height_px, intrinsics.width_px)
    if obj.size_m is None or obj.position_m is None:
        return np.zeros(shape, dtype=bool)
    corners = box_vertices(obj.size_m, obj.position_m, obj.yaw_deg or 0.0)
    T_camera_from_world = T_world_from_camera.inverse().to_numpy()
    cam = (T_camera_from_world @ np.c_[corners, np.ones(len(corners))].T).T[:, :3]
    if (cam[:, 2] < min_depth_m).any():
        return np.zeros(shape, dtype=bool)
    # OpenCV pixel (x, y) is (column, row); distortion is ignored (see module docstring).
    u = intrinsics.fx * cam[:, 0] / cam[:, 2] + intrinsics.cx
    v = intrinsics.fy * cam[:, 1] / cam[:, 2] + intrinsics.cy
    points: Any = np.round(np.c_[u, v]).astype(np.int32)
    hull: Any = cv2.convexHull(points)
    canvas = np.zeros(shape, dtype=np.uint8)
    cv2.fillConvexPoly(canvas, hull, 1)
    return canvas.astype(bool)


def assign_mask(mask: BoolMask, gt_masks: Mapping[str, BoolMask]) -> str | None:
    """Id of the ground-truth object a mask overlaps most, or None below ``MIN_ASSIGN_IOU``."""
    best_id: str | None = None
    best_iou = MIN_ASSIGN_IOU
    for object_id, gt_mask in gt_masks.items():
        union = np.logical_or(mask, gt_mask).sum()
        if union == 0:
            continue
        iou = float(np.logical_and(mask, gt_mask).sum()) / float(union)
        if iou > best_iou:
            best_id, best_iou = object_id, iou
    return best_id


@dataclass(frozen=True)
class MaskRecall:
    """Mask-level recall of object discovery (P2.1), before any point cluster exists."""

    visible: tuple[str, ...]
    found: tuple[str, ...]
    false_positives: int

    @property
    def recall(self) -> float | None:
        return len(self.found) / len(self.visible) if self.visible else None


def mask_recall(
    objects: Sequence[GtObject],
    intrinsics: Intrinsics,
    frames: Sequence[tuple[SE3, Sequence[BoolMask]]],
) -> MaskRecall:
    """Which ground-truth objects discovery found, judged on the sampled frames.

    ``frames`` pairs each frame's camera pose with the discovered masks of that frame. An
    object is visible when its projected box is non-empty in at least one frame, and found
    when some mask is assigned to it (``assign_mask``) in any frame. Masks assigned to no
    object count as false positives; occlusion is ignored, as in the module docstring.
    Deformable objects are left out, as in the other metrics (D-014).
    """
    rigid = [o for o in objects if not o.deformable]
    visible: set[str] = set()
    found: set[str] = set()
    false_positives = 0
    for pose, masks in frames:
        gt_masks = {o.object_id: project_box_mask(o, intrinsics, pose) for o in rigid}
        gt_masks = {oid: m for oid, m in gt_masks.items() if m.any()}
        visible.update(gt_masks)
        for mask in masks:
            assigned = assign_mask(mask, gt_masks)
            if assigned is None:
                false_positives += 1
            else:
                found.add(assigned)
    order = [o.object_id for o in rigid]
    return MaskRecall(
        tuple(i for i in order if i in visible),
        tuple(i for i in order if i in found),
        false_positives,
    )


def count_id_switches(assignments: Mapping[str, Sequence[str | None]]) -> int:
    """Switches over tracks. Each sequence lists the assigned object per frame, in time order.

    Frames without an assignment are skipped, so occlusion alone is not a switch.
    """
    total = 0
    for sequence in assignments.values():
        assigned = [a for a in sequence if a is not None]
        total += sum(1 for a, b in itertools.pairwise(assigned) if a != b)
    return total


@dataclass(frozen=True)
class TrackScore:
    """Identity quality of tracks against ground truth (P2.2)."""

    switches: int
    tracks_per_object: Mapping[str, int]  # tracks that mostly follow each ground-truth object
    unassigned_tracks: int  # tracks that never overlap a ground-truth object (false positives)

    @property
    def fragmented(self) -> int:
        """Extra tracks beyond one per object: a perfect tracking gives 0."""
        return sum(n - 1 for n in self.tracks_per_object.values())


def score_tracks(
    objects: Sequence[GtObject],
    intrinsics: Intrinsics,
    poses_in_board: Mapping[int, SE3],
    tracks: Mapping[int, Mapping[int, BoolMask]],
) -> TrackScore:
    """Assign each track mask to a ground-truth box (``assign_mask``) and count switches.

    ``tracks`` maps a track id to its masks by frame index; ``poses_in_board`` gives the camera
    pose of every frame in the ground-truth frame.
    """
    rigid = [o for o in objects if not o.deformable]
    gt_masks: dict[int, dict[str, BoolMask]] = {}

    def gt_in(frame: int) -> dict[str, BoolMask]:
        if frame not in gt_masks:
            gt_masks[frame] = {
                o.object_id: project_box_mask(o, intrinsics, poses_in_board[frame]) for o in rigid
            }
        return gt_masks[frame]

    assignments: dict[str, list[str | None]] = {}
    owner: dict[str, str] = {}
    for track_id, masks in tracks.items():
        sequence = [assign_mask(masks[f], gt_in(f)) for f in sorted(masks)]
        assignments[str(track_id)] = sequence
        votes = [a for a in sequence if a is not None]
        if votes:
            owner[str(track_id)] = max(set(votes), key=votes.count)
    per_object: dict[str, int] = {}
    for object_id in owner.values():
        per_object[object_id] = per_object.get(object_id, 0) + 1
    return TrackScore(count_id_switches(assignments), per_object, len(assignments) - len(owner))


def _row(name: str, value: float | None, n: int, target: float | None, kind: str) -> dict[str, Any]:
    passed = None
    if value is not None and target is not None:
        passed = value >= target if kind == "min" else value <= target
    return {
        "name": name,
        "status": "measured" if value is not None else "no_data",
        "value": value,
        "n": n,
        "target": target,
        "passed": passed,
    }


def summarize_segmentation(
    scenes: Sequence[SceneResult],
    flags: Mapping[str, Mapping[str, tuple[str, ...]]],
    id_switches: Mapping[str, int],
    gates: Mapping[str, float],
    recall_target: float | None,
    flag_scenes: Sequence[str] = FLAG_SCENES,
) -> dict[str, Any]:
    """Rows for the four Phase 2 exit criteria.

    ``flags`` is object flags by scene id then object id; ``id_switches`` is a count per scene
    for the scenes whose tracks were available. A criterion with nothing to measure reports
    ``no_data`` instead of passing.
    """
    n_eval = sum(s.n_evaluable for s in scenes)
    matched = sum(s.n_matched for s in scenes)
    easy = [o.size_error for s in scenes if s.tier == EASY_TIER for o in s.objects if o.matched]
    easy_errors = [e for e in easy if e is not None]
    switches = list(id_switches.values())
    listed = [s for s in flag_scenes if s in flags]
    flagged = [s for s in listed if any(flags[s].values())]

    rows = [
        _row("object_recall", matched / n_eval if n_eval else None, n_eval, recall_target, "min"),
        _row(
            "cluster_bbox_error_median_easy",
            statistics.median(easy_errors) if easy_errors else None,
            len(easy_errors),
            gates.get("cluster_bbox_error_max"),
            "max",
        ),
        _row(
            "id_switches_per_scene",
            statistics.fmean(switches) if switches else None,
            len(switches),
            gates.get("id_switches_per_scene_max"),
            "max",
        ),
        # Every listed scene must be flagged, so the target is the full fraction.
        _row(
            "hard_scenes_flagged_fraction",
            len(flagged) / len(listed) if listed else None,
            len(listed),
            1.0,
            "min",
        ),
    ]
    if listed and len(listed) < len(flag_scenes):
        # A dev-only run sees two of the four scenes; passing on those would mislead.
        rows[-1]["passed"] = None
        rows[-1]["status"] = "partial"
    return {
        "rows": rows,
        "flag_scenes": list(flag_scenes),
        "flag_scenes_missing": [s for s in flag_scenes if s not in flags],
        "flags_by_object": {s: {o: list(f) for o, f in flags[s].items()} for s in listed},
    }
