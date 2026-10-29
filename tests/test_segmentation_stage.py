from __future__ import annotations

from pathlib import Path

import numpy as np

from video2sim.schemas import Flag, Instances
from video2sim.stages.depth_flags import Verdict
from video2sim.stages.keyframes import KeyframeChoice
from video2sim.stages.material_pass import MaterialAnswer, MaterialDecision
from video2sim.stages.primitive_fit import FitResult, Primitive
from video2sim.stages.segmentation_stage import (
    build_object,
    choose_cluster,
    fit_report,
    object_flags,
    repaired_ids,
    write_object_files,
)
from video2sim.stages.track_repair import RepairReport
from video2sim.stages.tracking import Track


def decision(
    flag: Flag | None = None,
    non_rigid: str | None = None,
    answer: MaterialAnswer | None = None,
) -> MaterialDecision:
    verdict: Verdict = "clear"
    return MaterialDecision(3, flag, "reason", verdict, "ok", answer, None, non_rigid)


def test_repaired_ids_cover_split_parts_and_the_kept_id_of_a_merge() -> None:
    report = RepairReport(splits=((2, (2, 7)),), merges=((4, 5, 6),))
    assert repaired_ids(report) == {2, 7, 4}


def test_flags_are_combined_once_in_a_fixed_order() -> None:
    flags = object_flags(
        (Flag.SMALL, Flag.THIN),
        decision(Flag.TRANSPARENT_OR_REFLECTIVE, "VLM rigidity deformable (bag)"),
        repaired=True,
    )
    assert flags == [
        Flag.SMALL,
        Flag.THIN,
        Flag.TRANSPARENT_OR_REFLECTIVE,
        Flag.NON_RIGID,
        Flag.ID_SWITCH_REPAIRED,
    ]
    assert object_flags((), decision(), repaired=False) == []


def test_files_are_written_and_the_paths_are_relative(tmp_path: Path) -> None:
    mask = np.zeros((6, 8), dtype=bool)
    mask[2:4, 3:6] = True
    track = Track(5, {10: mask, 12: mask})
    cluster = np.array([[0.0, 0.0, 0.01], [0.1, 0.0, 0.02]])
    hull = cluster + 1.0
    mask_dir, cluster_path = write_object_files(tmp_path, "obj_005", track, cluster, {"hull": hull})
    assert mask_dir == "segmentation/masks/obj_005"
    assert cluster_path == "segmentation/clusters/obj_005.npy"
    assert sorted(p.name for p in (tmp_path / "masks" / "obj_005").iterdir()) == [
        "00010.png",
        "00012.png",
    ]
    assert np.array_equal(np.load(tmp_path / "clusters" / "obj_005.npy"), cluster)
    assert np.array_equal(np.load(tmp_path / "clusters" / "obj_005_hull.npy"), hull)


def test_object_without_a_material_answer_gets_a_generic_label() -> None:
    keyframe = KeyframeChoice(4, 0.5, 0.5, 0.0, 1.0, 1.0, 1.0, 1.0)
    obj = build_object("obj_001", "segmentation/masks/obj_001", "segmentation/clusters/obj_001.npy",
                       [keyframe], [], [], decision())  # fmt: skip
    assert (obj.label, obj.category, obj.keyframes) == ("object", None, [4])


def test_material_answer_fills_the_instance_fields() -> None:
    answer = MaterialAnswer("mug", "ceramic", "glossy", "hollow", 0.9, "rigid")
    obj = build_object("obj_001", "segmentation/masks/obj_001", "segmentation/clusters/obj_001.npy",
                       [], [], [], decision(answer=answer))  # fmt: skip
    assert (obj.label, obj.material_class, obj.rigidity, obj.material_confidence) == (
        "mug",
        "ceramic",
        "rigid",
        0.9,
    )
    Instances(objects=[obj], flags=["note"])  # validates, scene flags are plain strings


def test_fit_report_names_the_source_and_the_primitive() -> None:
    prim = Primitive("box", (0.1, 0.2), 0.5, (0.08, 0.05, 0.04), 0.0, 0.03, 12)
    fits = {
        3: FitResult(prim.outline(), prim, "primitive", "box, mask cost 0.030"),
        7: FitResult(np.zeros((9, 3)), None, "points", "only 2 whole masks"),
    }
    report = fit_report(fits)
    assert list(report) == ["obj_003", "obj_007"]
    assert report["obj_003"]["kind"] == "box" and report["obj_003"]["size_m"] == [0.08, 0.05, 0.04]
    assert report["obj_007"] == {
        "source": "points",
        "reason": "only 2 whole masks",
        "kind": None,
        "size_m": None,
        "center_xy_m": None,
        "yaw_rad": None,
        "mask_cost": None,
        "views": None,
    }


def test_the_fitted_cluster_is_the_cluster_and_the_points_go_beside_it() -> None:
    prim = Primitive("cylinder", (0.0, 0.0), 0.0, (0.08, 0.08, 0.1), 0.0, 0.02, 30)
    points, hull = np.ones((20, 3)), np.zeros((50, 3))
    fit = FitResult(prim.outline(), prim, "primitive", "cylinder, mask cost 0.020")
    cluster, extra, note = choose_cluster(fit, points, hull)
    assert np.array_equal(cluster, prim.outline()) and note is None
    assert set(extra) == {"points", "hull"} and extra["points"] is points


def test_a_union_or_points_cluster_carries_a_note() -> None:
    prim = Primitive("box", (0.0, 0.0), 0.0, (0.1, 0.05, 0.05), 0.0, 0.3, 30)
    points, hull = np.ones((20, 3)), np.zeros((50, 3))
    union = FitResult(np.vstack([prim.outline(), points]), prim, "union", "best fit box ... union")
    _, _, note = choose_cluster(union, points, hull)
    assert note == "cluster: best fit box ... union"
    cluster, extra, note = choose_cluster(None, None, hull)
    assert cluster is hull and extra == {} and note is not None and "hull" in note
