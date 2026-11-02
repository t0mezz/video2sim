from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from video2sim import cli
from video2sim.cli import main
from video2sim.evaluation.alignment import BoardAlignment
from video2sim.evaluation.baselines import oracle
from video2sim.evaluation.metrics import evaluate_scene
from video2sim.evaluation.scenes import GtObject, GtScene
from video2sim.evaluation.segmentation import (
    BoolMask,
    assign_mask,
    count_id_switches,
    load_instances,
    load_tracks,
    mask_recall,
    project_box_mask,
    score_tracks,
    summarize_segmentation,
)
from video2sim.runner import OUTPUT_FILE
from video2sim.schemas import SE3, CameraFrame, Intrinsics, Plane, WorldModel

GATES = {"id_switches_per_scene_max": 1.0, "cluster_bbox_error_max": 0.10}


def _obj(oid: str, x: float) -> GtObject:
    return GtObject(oid, "box", (0.06, 0.06, 0.06), (x, 0.0, 0.03), 0.0, 0.2)


def _scene(sid: str, tier: str = "easy") -> GtScene:
    return GtScene(sid, tier, "dev", "grid", (_obj("a", 0.0), _obj("b", 0.2)))


def _camera() -> tuple[Intrinsics, SE3]:
    # Camera 0.5 m above the board looking straight down: +z forward is world -Z.
    pose = np.array(
        [[1.0, 0.0, 0.0, 0.1], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.5], [0, 0, 0, 1.0]]
    )
    k = Intrinsics(fx=800, fy=800, cx=320, cy=240, width_px=640, height_px=480)
    return k, SE3.from_numpy(pose)


def test_switches_ignore_gaps_and_count_changes() -> None:
    tracks = {"t0": ["a", None, "a", "b", "b"], "t1": [None, "b", "b"]}
    assert count_id_switches(tracks) == 1


def test_projected_box_assigns_to_the_right_object() -> None:
    k, pose = _camera()
    masks = {o.object_id: project_box_mask(o, k, pose) for o in _scene("s01").objects}
    assert masks["a"].any() and masks["b"].any()
    assert assign_mask(masks["a"], masks) == "a"
    assert assign_mask(np.zeros_like(masks["a"]), masks) is None


def test_mask_recall_counts_found_missed_and_false_positives() -> None:
    k, pose = _camera()
    objs = _scene("s01").objects
    gt = {o.object_id: project_box_mask(o, k, pose) for o in objs}
    stray = np.zeros_like(gt["a"])
    stray[0:10, 0:10] = True
    result = mask_recall(objs, k, [(pose, [gt["a"], stray])])
    assert result.visible == ("a", "b")
    assert result.found == ("a",)
    assert result.false_positives == 1
    assert result.recall == 0.5


def test_mask_recall_accumulates_over_frames_and_skips_deformables() -> None:
    k, pose = _camera()
    soft = GtObject("c", "cloth", (0.06, 0.06, 0.06), (0.1, 0.1, 0.03), 0.0, None, deformable=True)
    objs = (*_scene("s01").objects, soft)
    gt = {o.object_id: project_box_mask(o, k, pose) for o in objs}
    result = mask_recall(objs, k, [(pose, [gt["a"]]), (pose, [gt["b"]])])
    assert result.found == ("a", "b")
    assert result.recall == 1.0


def test_mask_recall_without_visible_objects_has_no_value() -> None:
    k, pose = _camera()
    behind = GtObject("c", "box", (0.06, 0.06, 0.06), (0.0, 0.0, 0.9), 0.0, None)
    assert mask_recall([behind], k, [(pose, [])]).recall is None


def test_box_behind_the_camera_has_empty_mask() -> None:
    k, pose = _camera()
    behind = GtObject("c", "box", (0.06, 0.06, 0.06), (0.0, 0.0, 0.9), 0.0, None)
    assert not project_box_mask(behind, k, pose).any()


def test_oracle_clusters_score_zero_bbox_error() -> None:
    scene = _scene("s01")
    result = evaluate_scene(scene, oracle(scene))
    rows = {r["name"]: r for r in summarize_segmentation([result], {}, {}, GATES, 0.9)["rows"]}
    assert rows["object_recall"]["value"] == 1.0
    assert rows["cluster_bbox_error_median_easy"]["value"] == 0.0
    assert rows["cluster_bbox_error_median_easy"]["passed"] is True
    assert rows["id_switches_per_scene"]["status"] == "no_data"
    assert rows["hard_scenes_flagged_fraction"]["status"] == "no_data"


def test_bbox_error_only_counts_easy_scenes() -> None:
    easy, hard = _scene("s01"), _scene("s17", "hard")
    results = [
        evaluate_scene(easy, oracle(easy)),
        evaluate_scene(hard, oracle(hard, size_sigma=0.5, seed=1)),
    ]
    summary = summarize_segmentation(results, {}, {}, GATES, 0.9)
    assert summary["rows"][1]["value"] == 0.0
    assert summary["rows"][1]["n"] == 2


def test_flag_criterion_is_partial_without_all_four_scenes() -> None:
    flags: dict[str, dict[str, tuple[str, ...]]] = {"s17": {"o": ("thin",)}, "s18": {"o": ()}}
    summary = summarize_segmentation([], flags, {}, GATES, 0.9)
    row = summary["rows"][3]
    assert (row["value"], row["n"], row["status"], row["passed"]) == (0.5, 2, "partial", None)
    assert summary["flag_scenes_missing"] == ["s19", "s20"]


def test_flag_criterion_passes_when_every_scene_is_flagged() -> None:
    flags = {s: {"o": ("small",)} for s in ("s17", "s18", "s19", "s20")}
    row = summarize_segmentation([], flags, {}, GATES, 0.9)["rows"][3]
    assert row["passed"] is True


def test_switch_gate_uses_the_mean_over_scenes() -> None:
    row = summarize_segmentation([], {}, {"s01": 0, "s02": 3}, GATES, 0.9)["rows"][2]
    assert row["value"] == 1.5 and row["passed"] is False


def _write_run(root: Path) -> Path:
    run = root / "s01"
    (run / "segmentation").mkdir(parents=True)
    np.save(run / "segmentation" / "obj_0.npy", np.array([[0, 0, 0], [0.1, 0.1, 0.1]]))
    (run / "segmentation" / OUTPUT_FILE).write_text(
        json.dumps(
            {
                "objects": [
                    {
                        "object_id": "obj_0",
                        "label": "box",
                        "mask_dir": "segmentation/masks/obj_0",
                        "point_cluster_path": "segmentation/obj_0.npy",
                        "keyframes": [0],
                        "flags": ["thin"],
                    }
                ]
            }
        )
    )
    return run


def test_load_instances_reads_clusters_and_flags(tmp_path: Path) -> None:
    run = load_instances(_write_run(tmp_path))
    assert run.prediction.objects[0].vertices.shape == (2, 3)
    assert run.flags == {"obj_0": ("thin",)}


def _write_masks(run: Path, masks: dict[int, dict[int, BoolMask]]) -> None:
    import cv2

    for track_id, frames in masks.items():
        track_dir = run / "segmentation" / "masks" / f"obj_{track_id:03d}"
        track_dir.mkdir(parents=True, exist_ok=True)
        for frame, mask in frames.items():
            cv2.imwrite(str(track_dir / f"{frame:05d}.png"), mask.astype(np.uint8) * 255)


def test_load_tracks_reads_masks_by_track_and_frame(tmp_path: Path) -> None:
    run = tmp_path / "s01"
    a = np.zeros((8, 8), dtype=bool)
    a[1:3, 1:3] = True
    b = np.zeros((8, 8), dtype=bool)
    b[4:6, 4:6] = True
    _write_masks(run, {0: {0: a, 1: b}, 1: {1: b}})
    masks = run / "segmentation" / "masks"
    (masks / "notes").mkdir()  # junk without an id is skipped
    (masks / "obj_002.png").write_bytes(b"not a directory")  # files are skipped
    (masks / "obj_000" / "xx.png").write_bytes(b"not a frame")  # bad names are skipped
    tracks = load_tracks(run)
    assert sorted(tracks) == [0, 1]
    assert sorted(tracks[0]) == [0, 1] and sorted(tracks[1]) == [1]
    assert tracks[0][0].dtype == bool and tracks[0][0].sum() == 4
    assert load_tracks(tmp_path / "missing") == {}


def test_score_tracks_counts_a_switch_between_objects() -> None:
    k, pose = _camera()
    objs = _scene("s01").objects
    gt = {o.object_id: project_box_mask(o, k, pose) for o in objs}
    tracks = {0: {0: gt["a"], 1: gt["a"], 2: gt["b"]}, 1: {0: gt["b"], 1: gt["b"]}}
    poses = {0: pose, 1: pose, 2: pose}
    result = score_tracks(objs, k, poses, tracks)
    assert result.switches == 1
    assert result.tracks_per_object == {"a": 1, "b": 1}
    assert result.unassigned_tracks == 0


def test_cli_reports_phase_2_section_for_the_oracle(tmp_path: Path) -> None:
    root = tmp_path / "scenes"
    (root / "s01").mkdir(parents=True)
    (root / "s01" / "ground_truth.json").write_text(
        json.dumps(
            {
                "scene_id": "s01",
                "tier": "easy",
                "split": "dev",
                "board": "grid",
                "objects": [
                    {
                        "object_id": "o1",
                        "label": "cube",
                        "size_m": [0.05, 0.05, 0.05],
                        "position_m": [0.1, 0.1, 0.025],
                        "yaw_deg": 0.0,
                        "mass_kg": 0.1,
                        "incomplete": [],
                    }
                ],
            }
        )
    )
    out = tmp_path / "out"
    code = main(["eval", "--benchmark", str(root), "--baseline", "oracle", "--out", str(out)])
    report = json.loads((out / "report.json").read_text())
    assert code == 0
    assert report["segmentation"]["rows"][0]["value"] == 1.0
    assert "Phase 2 exit criteria" in (out / "report.html").read_text()


def test_load_instances_moves_clusters_into_the_board_frame(tmp_path: Path) -> None:
    t = np.eye(4)
    t[:3, :3] = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]  # 90 degrees about +Z
    t[:3, 3] = [0.3, 0.2, 0.0]
    run = load_instances(_write_run(tmp_path), t)
    assert np.allclose(run.prediction.objects[0].vertices, [[0.3, 0.2, 0.0], [0.2, 0.3, 0.1]])


def _gt_scene(root: Path, scene_id: str, board: str) -> None:
    (root / scene_id).mkdir(parents=True)
    (root / scene_id / "ground_truth.json").write_text(
        json.dumps(
            {
                "scene_id": scene_id,
                "tier": "easy",
                "split": "dev",
                "board": board,
                "objects": [
                    {
                        "object_id": "o1",
                        "label": "cube",
                        "size_m": [0.1, 0.1, 0.1],
                        "position_m": [0.35, 0.25, 0.05],
                        "yaw_deg": 0.0,
                        "mass_kg": 0.1,
                        "incomplete": [],
                    }
                ],
            }
        )
    )


def _world_json(run: Path) -> None:
    (run / "reconstruction").mkdir(parents=True)
    world = WorldModel(
        intrinsics=Intrinsics(fx=500, fy=500, cx=320, cy=240, width_px=640, height_px=480),
        frames=[],
        scale_factor=1.0,
        scale_source="marker",
        table_plane=Plane(normal=(0.0, 0.0, 1.0), offset_m=0.0),
        point_cloud_path="reconstruction/points.ply",
        mean_reprojection_error_px=0.5,
        registered_fraction=1.0,
    )
    (run / "reconstruction" / OUTPUT_FILE).write_text(world.model_dump_json())


def test_cli_scores_segmentation_clusters_in_the_board_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, runs = tmp_path / "scenes", tmp_path / "runs"
    _gt_scene(root, "s01", "grid")
    _gt_scene(root, "s05", "clean")
    _write_run(runs)
    cube = np.array([[x, y, z] for x in (0, 0.1) for y in (0, 0.1) for z in (0, 0.1)])
    np.save(runs / "s01" / "segmentation" / "obj_0.npy", cube)  # a 0.1 m cube at the origin
    _world_json(runs / "s01")
    (runs / "s05").symlink_to(runs / "s01", target_is_directory=True)
    shift = np.eye(4)
    shift[:3, 3] = [0.3, 0.2, 0.0]  # the board corner is 0.3 m and 0.2 m from the world origin
    seen: list[tuple[str, str]] = []

    def fake_align(
        world: WorldModel, run_dir: Path, scene_id: str, board: str, edge: float, family: str
    ) -> BoardAlignment | None:
        seen.append((scene_id, board))
        return BoardAlignment(shift, 1.0, 0.0, (0,)) if board == "grid" else None

    monkeypatch.setattr(cli, "align_scene", fake_align)
    out = tmp_path / "out"
    code = main(
        ["eval", "--benchmark", str(root), "--runs", str(runs), "--stage", "segmentation",
         "--out", str(out)]
    )  # fmt: skip
    report = json.loads((out / "report.json").read_text())
    assert code == 0 and sorted(seen) == [("s01", "grid"), ("s05", "clean")]
    recall, bbox = report["segmentation"]["rows"][:2]
    # Unshifted, the cluster centre (0.05, 0.05) is 0.36 m from the cube and would not match.
    assert recall["value"] == 1.0 and recall["n"] == 1  # s05 cannot be aligned: not scored
    assert bbox["value"] < 1e-9
    assert "s05" in report["header"]["scenes_without_prediction"]


def test_cli_scores_track_switches_from_masks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, runs = tmp_path / "scenes", tmp_path / "runs"
    (root / "s01").mkdir(parents=True)
    (root / "s01" / "ground_truth.json").write_text(
        json.dumps(
            {
                "scene_id": "s01",
                "tier": "easy",
                "split": "dev",
                "board": "grid",
                "objects": [
                    {
                        "object_id": "o1",
                        "label": "cube",
                        "size_m": [0.1, 0.1, 0.1],
                        "position_m": [0.2, 0.25, 0.05],
                        "yaw_deg": 0.0,
                        "mass_kg": 0.1,
                        "incomplete": [],
                    },
                    {
                        "object_id": "o2",
                        "label": "cube",
                        "size_m": [0.1, 0.1, 0.1],
                        "position_m": [0.5, 0.25, 0.05],
                        "yaw_deg": 0.0,
                        "mass_kg": 0.1,
                        "incomplete": [],
                    },
                ],
            }
        )
    )
    run = _write_run(runs)  # prediction cluster and flags; track masks below are separate
    k = Intrinsics(fx=800, fy=800, cx=320, cy=240, width_px=640, height_px=480)
    pose = SE3.from_numpy(
        np.array(
            [[1.0, 0.0, 0.0, 0.35], [0.0, -1.0, 0.0, 0.25], [0.0, 0.0, -1.0, 0.6], [0, 0, 0, 1.0]]
        )
    )
    gt = json.loads((root / "s01" / "ground_truth.json").read_text())["objects"]
    boxes = {
        o["object_id"]: project_box_mask(
            GtObject(
                o["object_id"],
                o["label"],
                tuple(o["size_m"]),
                tuple(o["position_m"]),
                o["yaw_deg"],
                o["mass_kg"],
            ),
            k,
            pose,
        )
        for o in gt
    }
    assert all(m.any() for m in boxes.values())
    _write_masks(run, {0: {0: boxes["o1"], 1: boxes["o2"]}})  # one track jumping cubes
    world = WorldModel(
        intrinsics=k,
        frames=[
            CameraFrame(frame_index=f, image_path=f"frames/{f:05d}.png", T_world_from_camera=pose)
            for f in (0, 1)
        ],
        scale_factor=1.0,
        scale_source="marker",
        table_plane=Plane(normal=(0.0, 0.0, 1.0), offset_m=0.0),
        point_cloud_path="reconstruction/points.ply",
        mean_reprojection_error_px=0.5,
        registered_fraction=1.0,
    )
    (run / "reconstruction").mkdir(parents=True)
    (run / "reconstruction" / OUTPUT_FILE).write_text(world.model_dump_json())

    def fake_align(
        world: WorldModel, run_dir: Path, scene_id: str, board: str, edge: float, family: str
    ) -> BoardAlignment | None:
        return BoardAlignment(np.eye(4), 1.0, 0.0, (0,))

    monkeypatch.setattr(cli, "align_scene", fake_align)
    out = tmp_path / "out"
    code = main(
        ["eval", "--benchmark", str(root), "--runs", str(runs), "--stage", "segmentation",
         "--out", str(out)]
    )  # fmt: skip
    report = json.loads((out / "report.json").read_text())
    assert code == 0
    rows = report["segmentation"]["rows"]
    (switches,) = [r for r in rows if r["name"] == "id_switches_per_scene"]
    assert switches["status"] == "measured"
    assert switches["value"] == 1.0 and switches["n"] == 1
