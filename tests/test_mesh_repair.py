from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pytest
import trimesh
import yaml

from video2sim.stages.mesh_repair import RepairConfig, repair_mesh, validate_mesh

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def cfg() -> RepairConfig:
    return RepairConfig.from_config(
        yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    )


def cube(size: float = 0.1) -> trimesh.Trimesh:
    return cast(trimesh.Trimesh, trimesh.creation.box(extents=(size, size, size)))


def test_a_clean_cube_passes_untouched(cfg: RepairConfig) -> None:
    res = repair_mesh(cube(), cfg)
    assert res.ok
    assert res.mesh.volume == pytest.approx(1e-3)


def test_a_hole_is_filled_and_volume_recovered(cfg: RepairConfig) -> None:
    m = cube()
    m.update_faces(np.arange(len(m.faces)) != 0)  # drop one triangle
    assert not validate_mesh(m, cfg)["watertight"]
    res = repair_mesh(m, cfg)
    assert res.ok and res.steps["faces_added"] > 0
    assert res.mesh.volume == pytest.approx(1e-3, rel=0.01)


def test_a_floater_is_removed_but_a_real_second_part_stays(cfg: RepairConfig) -> None:
    speck = trimesh.creation.icosphere(radius=0.002)
    speck.apply_translation([0.3, 0, 0])
    res = repair_mesh(cast(trimesh.Trimesh, trimesh.util.concatenate([cube(), speck])), cfg)
    assert res.steps["floaters_removed"] == 1 and len(res.mesh.faces) == 12

    handle = cube(0.05)
    handle.apply_translation([0.3, 0, 0])
    res = repair_mesh(cast(trimesh.Trimesh, trimesh.util.concatenate([cube(), handle])), cfg)
    assert res.steps["floaters_removed"] == 0


def test_inside_out_mesh_is_turned_outward(cfg: RepairConfig) -> None:
    m = cube()
    m.invert()
    res = repair_mesh(m, cfg)
    assert res.ok and res.mesh.volume > 0


def test_duplicate_faces_are_dropped(cfg: RepairConfig) -> None:
    m = cube()
    m = trimesh.Trimesh(m.vertices, np.vstack([m.faces, m.faces[:3]]), process=False)
    res = repair_mesh(m, cfg)
    assert res.ok and len(res.mesh.faces) == 12


def test_an_unfillable_or_degenerate_mesh_fails_loudly(cfg: RepairConfig) -> None:
    plane = trimesh.Trimesh([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]])
    res = repair_mesh(plane, cfg)
    assert not res.checks["positive_volume"] or not res.checks["watertight"]
    empty = trimesh.Trimesh()
    assert not repair_mesh(empty, cfg).ok
    bad = cube()
    bad.vertices[0, 0] = np.nan
    assert not repair_mesh(bad, cfg).checks["finite"]


def test_input_is_not_modified(cfg: RepairConfig) -> None:
    m = cube()
    m.update_faces(np.arange(len(m.faces)) != 0)
    n = len(m.faces)
    repair_mesh(m, cfg)
    assert len(m.faces) == n
