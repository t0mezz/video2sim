from __future__ import annotations

from pathlib import Path

import numpy as np

from video2sim.schemas import Flag, Instances
from video2sim.stages.depth_flags import Verdict
from video2sim.stages.keyframes import KeyframeChoice
from video2sim.stages.material_pass import MaterialAnswer, MaterialDecision
from video2sim.stages.segmentation_stage import (
    build_object,
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
    mask_dir, cluster_path = write_object_files(tmp_path, "obj_005", track, cluster)
    assert mask_dir == "segmentation/masks/obj_005"
    assert cluster_path == "segmentation/clusters/obj_005.npy"
    assert sorted(p.name for p in (tmp_path / "masks" / "obj_005").iterdir()) == [
        "00010.png",
        "00012.png",
    ]
    assert np.array_equal(np.load(tmp_path / "clusters" / "obj_005.npy"), cluster)


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
