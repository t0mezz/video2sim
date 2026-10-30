from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import trimesh
import yaml

from video2sim.stages.mesh_repair import RepairConfig, repair_file, repair_mesh, validate_mesh

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


def sphere(subdivisions: int) -> trimesh.Trimesh:
    return cast(trimesh.Trimesh, trimesh.creation.icosphere(subdivisions=subdivisions))


def small_budget(cfg: RepairConfig, n: int) -> RepairConfig:
    return RepairConfig(
        cfg.min_component_area_frac,
        cfg.min_volume_m3,
        n,
        cfg.max_volume_change_frac,
        cfg.decimation_aggression,
    )


def test_a_mesh_within_budget_is_not_decimated(cfg: RepairConfig) -> None:
    res = repair_mesh(sphere(2), cfg)
    assert res.ok and "faces_removed_by_decimation" not in res.steps


def test_over_budget_mesh_is_decimated_and_stays_a_solid(cfg: RepairConfig) -> None:
    pytest.importorskip("fast_simplification")
    m = sphere(4)  # 5120 faces
    res = repair_mesh(m, small_budget(cfg, 1000))
    assert res.ok and len(res.mesh.faces) <= 1000
    assert res.mesh.volume == pytest.approx(m.volume, rel=0.02)


def test_decimation_that_distorts_the_volume_is_rejected(cfg: RepairConfig) -> None:
    pytest.importorskip("fast_simplification")
    tight = RepairConfig(cfg.min_component_area_frac, cfg.min_volume_m3, 100, 1e-6, 5)
    res = repair_mesh(sphere(4), tight)
    assert res.steps.get("decimation_rejected") == 1
    assert not res.checks["within_triangle_budget"] and not res.ok


def test_over_budget_without_a_simplifier_blocks_export(
    cfg: RepairConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "fast_simplification", None)  # makes the import fail
    res = repair_mesh(sphere(3), small_budget(cfg, 100))
    assert res.steps["decimation_unavailable"] == 1
    assert res.checks["watertight"] and not res.checks["within_triangle_budget"]


def test_repair_file_round_trips_and_writes_a_failing_mesh_too(
    cfg: RepairConfig, tmp_path: Path
) -> None:
    src = tmp_path / "in.ply"
    cube().export(src)
    res = repair_file(src, tmp_path / "out" / "cube.ply", cfg)
    assert res.ok
    back = cast(trimesh.Trimesh, trimesh.load(tmp_path / "out" / "cube.ply", force="mesh"))
    assert back.is_watertight and back.volume == pytest.approx(1e-3)

    bad = tmp_path / "bad.ply"
    trimesh.Trimesh([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]]).export(bad)
    res = repair_file(bad, tmp_path / "bad_out.ply", cfg)
    assert not res.ok and (tmp_path / "bad_out.ply").exists()


def test_holes_that_share_a_vertex_are_closed_without_a_non_manifold_edge(
    cfg: RepairConfig,
) -> None:
    m = sphere(2)
    faces = m.faces
    pair = next(
        (i, j)
        for i in range(len(faces))
        for j in range(i + 1, len(faces))
        if len(set(faces[i]) & set(faces[j])) == 1
    )
    keep = np.ones(len(faces), dtype=bool)
    keep[list(pair)] = False
    cut = trimesh.Trimesh(m.vertices, faces[keep], process=False)
    assert not cut.is_watertight
    res = repair_mesh(cut, cfg)
    assert res.ok and res.steps["faces_added"] > 0
    assert res.mesh.volume == pytest.approx(m.volume, rel=0.05)
