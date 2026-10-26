from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import trimesh

from video2sim.cli import main
from video2sim.evaluation.geometry import box_vertices, resting_box
from video2sim.evaluation.loader import load_run
from video2sim.runner import OUTPUT_FILE
from video2sim.schemas import SE3

OBJECTS = [
    {
        "object_id": "o1",
        "label": "box",
        "size_m": [0.10, 0.04, 0.05],
        "position_m": [0.10, 0.10, 0.025],
        "yaw_deg": 35.0,
        "mass_kg": 0.3,
        "incomplete": [],
    },
    {
        "object_id": "o2",
        "label": "cube",
        "size_m": [0.05, 0.05, 0.05],
        "position_m": [0.30, 0.10, 0.025],
        "yaw_deg": 0.0,
        "mass_kg": 0.1,
        "incomplete": [],
    },
]


def _write_scene(root: Path, sid: str, split: str) -> None:
    folder = root / sid
    folder.mkdir(parents=True)
    (folder / "ground_truth.json").write_text(
        json.dumps(
            {"scene_id": sid, "tier": "easy", "split": split, "board": "grid", "objects": OBJECTS}
        )
    )


@pytest.fixture
def benchmark(tmp_path: Path) -> Path:
    root = tmp_path / "scenes"
    _write_scene(root, "s01", "dev")
    _write_scene(root, "s02", "test")
    (root / "s03").mkdir()  # no ground truth yet
    return root


def _targets(tmp_path: Path) -> Path:
    path = tmp_path / "targets.yaml"
    path.write_text(
        "status: provisional\ntargets:\n  object_recall_min: 0.9\n  mass_tolerance: 0.3\n"
    )
    return path


def _args(benchmark: Path, tmp_path: Path, *extra: str) -> list[str]:
    return [
        "eval",
        "--benchmark",
        str(benchmark),
        "--targets",
        str(_targets(tmp_path)),
        "--out",
        str(tmp_path / "out"),
        "--config",
        str(tmp_path / "none.yaml"),
        *extra,
    ]


def test_default_split_is_dev_and_missing_ground_truth_is_reported(
    benchmark: Path, tmp_path: Path
) -> None:
    assert main(_args(benchmark, tmp_path, "--baseline", "oracle")) == 0
    report = json.loads((tmp_path / "out" / "report.json").read_text())
    assert [s["scene_id"] for s in report["scenes"]] == ["s01"]
    assert report["header"]["scenes_without_ground_truth"] == ["s03"]
    assert report["header"]["targets_status"] == "provisional"
    html = (tmp_path / "out" / "report.html").read_text()
    assert "no usable ground truth" in html and "provisional" in html


def test_split_and_single_scene_selection(benchmark: Path, tmp_path: Path) -> None:
    main(_args(benchmark, tmp_path, "--baseline", "oracle", "--split", "all"))
    report = json.loads((tmp_path / "out" / "report.json").read_text())
    assert [s["scene_id"] for s in report["scenes"]] == ["s01", "s02"]
    main(_args(benchmark, tmp_path, "--baseline", "oracle", "--scene", "s02"))
    report = json.loads((tmp_path / "out" / "report.json").read_text())
    assert [s["scene_id"] for s in report["scenes"]] == ["s02"]


def test_exit_code_depends_on_fail_on_target(benchmark: Path, tmp_path: Path) -> None:
    assert main(_args(benchmark, tmp_path, "--baseline", "constant")) == 0
    assert main(_args(benchmark, tmp_path, "--baseline", "constant", "--fail-on-target")) == 1
    assert main(_args(benchmark, tmp_path, "--baseline", "oracle", "--fail-on-target")) == 0


def test_load_run_applies_the_object_pose(tmp_path: Path) -> None:
    run = tmp_path / "run"
    (run / "shape").mkdir(parents=True)
    # A 10 x 4 x 5 cm box, centered on its own origin, placed at yaw 35 degrees.
    mesh = trimesh.creation.box(extents=(0.10, 0.04, 0.05))
    mesh.export(run / "shape" / "o1.ply")
    pose = np.eye(4)
    angle = np.radians(35.0)
    pose[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    pose[:3, 3] = [0.10, 0.10, 0.025]
    output: dict[str, Any] = {
        "objects": [
            {
                "object_id": "o1",
                "mesh_path": "shape/o1.ply",
                "T_world_from_object": SE3.from_numpy(pose).model_dump(),
                "geometry_source": "proxy_box",
                "fit_score": None,
                "flags": [],
            }
        ]
    }
    (run / "shape" / OUTPUT_FILE).write_text(json.dumps(output))
    prediction = load_run(run)
    box = resting_box(prediction.objects[0].vertices)
    assert box.size_m == pytest.approx((0.10, 0.04, 0.05), abs=1e-6)
    assert box.yaw_deg == pytest.approx(35.0, abs=1e-4)
    assert prediction.objects[0].mass_kg is None and prediction.runtime_s is None
    assert box_vertices((0.1, 0.04, 0.05), (0, 0, 0), 0).shape == (8, 3)


def test_deformable_scenes_are_left_out_unless_named(benchmark: Path, tmp_path: Path) -> None:
    path = benchmark / "s01" / "ground_truth.json"
    data = json.loads(path.read_text())
    data["objects"][0]["deformable"] = True
    path.write_text(json.dumps(data))
    _write_scene(benchmark, "s04", "dev")
    main(_args(benchmark, tmp_path, "--baseline", "oracle"))
    report = json.loads((tmp_path / "out" / "report.json").read_text())
    assert [s["scene_id"] for s in report["scenes"]] == ["s04"]
    assert report["header"]["scenes_excluded_deformable"] == "s01"
    main(_args(benchmark, tmp_path, "--baseline", "oracle", "--scene", "s01"))
    report = json.loads((tmp_path / "out" / "report.json").read_text())
    assert [s["scene_id"] for s in report["scenes"]] == ["s01"]
