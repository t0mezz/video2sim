from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import trimesh
import yaml
from numpy.typing import NDArray

from video2sim.stages.collision import (
    CollisionConfig,
    check_parts,
    collision_geometry,
    prepare_coacd,
    surface_fit,
    write_collision,
)

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def cfg() -> CollisionConfig:
    return CollisionConfig.from_config(
        yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    )


def rotation(angle: float, axis: list[float]) -> NDArray[np.float64]:
    return np.asarray(trimesh.transformations.rotation_matrix(angle, axis))  # type: ignore[no-untyped-call]


def lshape() -> trimesh.Trimesh:
    """An L-shaped prism, 3/4 of its bounding box, so no primitive fits."""
    u = 0.05
    poly = np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]], dtype=float) * u
    h = u
    v = np.vstack([np.c_[poly, np.zeros(6)], np.c_[poly, np.full(6, h)]])
    # the L as a fan from its inner corner, counter-clockwise seen from +z
    tri = [[0, 1, 3], [1, 2, 3], [0, 3, 4], [0, 4, 5]]
    faces = [[a, c, b] for a, b, c in tri]  # bottom faces down
    faces += [[a + 6, b + 6, c + 6] for a, b, c in tri]  # top faces up
    for i in range(6):
        j = (i + 1) % 6
        faces += [[i, j, j + 6], [i, j + 6, i + 6]]
    m = trimesh.Trimesh(v, faces, process=True)
    return m


def test_a_box_gets_a_box(cfg: CollisionConfig) -> None:
    m = trimesh.creation.box(extents=(0.1, 0.06, 0.03))
    m.apply_transform(rotation(0.6, [0, 0, 1]))
    res = collision_geometry(m, cfg)
    assert res.kind == "box" and res.ok and len(res.parts) == 1
    assert res.parts[0].volume == pytest.approx(m.volume, rel=0.02)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_a_cylinder_gets_a_cylinder_along_its_own_axis(cfg: CollisionConfig, axis: int) -> None:
    m = trimesh.creation.cylinder(radius=0.03, height=0.12, sections=48)
    rot = {
        0: rotation(np.pi / 2, [0, 1, 0]),
        1: rotation(np.pi / 2, [1, 0, 0]),
        2: np.eye(4),
    }[axis]
    m.apply_transform(rot)
    res = collision_geometry(m, cfg)
    assert res.kind == "cylinder" and res.ok
    assert res.parts[0].volume == pytest.approx(m.volume, rel=0.05)


def test_a_sphere_gets_a_sphere(cfg: CollisionConfig) -> None:
    res = collision_geometry(trimesh.creation.icosphere(subdivisions=3, radius=0.04), cfg)
    assert res.kind == "sphere" and res.ok


def test_a_primitive_keeps_the_object_frame(cfg: CollisionConfig) -> None:
    m = trimesh.creation.box(extents=(0.1, 0.06, 0.03))
    m.apply_translation([0.2, -0.1, 0.05])
    part = collision_geometry(m, cfg).parts[0]
    assert np.allclose(part.bounds, m.bounds, atol=1e-6)


def test_a_non_primitive_is_not_given_a_primitive(
    cfg: CollisionConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "coacd", None)
    res = collision_geometry(lshape(), cfg)
    assert res.kind == "convex_hull" and res.steps["coacd_unavailable"] == 1
    assert res.checks["collision_loads"]


def test_coacd_splits_an_l_shape_within_the_cap(cfg: CollisionConfig) -> None:
    pytest.importorskip("coacd")
    res = collision_geometry(lshape(), cfg)
    assert res.kind == "convex_decomposition" and 2 <= len(res.parts) <= cfg.max_convex_parts
    assert res.ok


def test_checks_fail_for_too_many_parts_a_nonconvex_part_and_a_bad_volume(
    cfg: CollisionConfig,
) -> None:
    visual = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    ok = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    assert all(check_parts([ok], visual, cfg).values())
    many = [ok.copy() for _ in range(cfg.max_convex_parts + 1)]
    assert not check_parts(many, visual, cfg)["collision_loads"]
    assert not check_parts([], visual, cfg)["collision_loads"]
    small = trimesh.creation.box(extents=(0.05, 0.05, 0.05))
    r = check_parts([small], visual, cfg)
    assert r["collision_loads"] and not r["collision_fit"]
    ring = trimesh.creation.annulus(r_min=0.02, r_max=0.05, height=0.03)
    assert not check_parts([ring], visual, cfg)["collision_loads"]


def test_an_open_mesh_gets_no_primitive(
    cfg: CollisionConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "coacd", None)
    m = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    m.update_faces(np.arange(len(m.faces)) != 0)
    assert collision_geometry(m, cfg).kind == "convex_hull"


def test_write_collision_writes_one_obj_per_part(cfg: CollisionConfig, tmp_path: Path) -> None:
    res = collision_geometry(trimesh.creation.box(extents=(0.1, 0.1, 0.1)), cfg)
    paths = write_collision(res, tmp_path / "physics", "obj_001")
    assert [p.name for p in paths] == ["obj_001_collision_0.obj"]
    back = cast(trimesh.Trimesh, trimesh.load(paths[0], force="mesh"))
    assert back.is_watertight and back.volume == pytest.approx(1e-3, rel=1e-3)


def test_config_ranges_are_validated() -> None:
    base = yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    base["physics"]["collision"]["fit_min_coverage"] = 1.5
    with pytest.raises(ValueError):
        CollisionConfig.from_config(base)


def hollow_box(wall_m: float = 0.005) -> trimesh.Trimesh:
    outer = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    inner = trimesh.creation.box(extents=(0.1 - 2 * wall_m,) * 3)
    inner.invert()
    return cast(trimesh.Trimesh, trimesh.util.concatenate([outer, inner]))


def test_a_hollow_shell_is_judged_by_its_outer_volume(cfg: CollisionConfig) -> None:
    res = collision_geometry(hollow_box(), cfg)
    assert res.kind == "box" and res.ok


def test_cuts_through_the_cavity_of_a_shell_do_not_count_as_loose(cfg: CollisionConfig) -> None:
    left = trimesh.creation.box(extents=(0.05, 0.1, 0.1))
    left.apply_translation([-0.025, 0, 0])
    right = trimesh.creation.box(extents=(0.05, 0.1, 0.1))
    right.apply_translation([0.025, 0, 0])
    assert surface_fit([left, right], hollow_box(), cfg)[1] <= cfg.fit_tolerance


def test_the_hull_of_a_cup_still_fails_the_fit(cfg: CollisionConfig) -> None:
    profile = [[0, -0.05], [0.04, -0.05], [0.04, 0.05], [0.035, 0.05], [0.035, -0.04], [0, -0.04]]
    cup = trimesh.creation.revolve(np.array(profile), sections=48)
    assert cup.is_watertight
    assert not check_parts([cast(trimesh.Trimesh, cup.convex_hull)], cup, cfg)["collision_fit"]


def test_a_primitive_that_fails_a_check_gives_way_to_the_decomposition(
    cfg: CollisionConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "coacd", None)
    m = trimesh.creation.box(extents=(0.1, 0.06, 0.03))
    monkeypatch.setattr(
        "video2sim.stages.collision.check_parts",
        lambda parts, visual, c, min_volume_m3=1e-12, body=None: {"collision_fit": len(parts) > 1},
    )
    res = collision_geometry(m, cfg)
    assert res.steps["primitive_rejected"] == 1 and res.kind == "convex_hull"


def test_a_hull_over_the_vertex_cap_is_thinned_and_still_convex(cfg: CollisionConfig) -> None:
    pytest.importorskip("fast_simplification")
    from video2sim.stages.collision import _limit_vertices

    ball = trimesh.creation.icosphere(subdivisions=4)  # 2562 vertices
    out = _limit_vertices(ball, cfg.max_part_vertices)
    assert len(out.vertices) <= cfg.max_part_vertices
    assert trimesh.convex.is_convex(out)  # type: ignore[no-untyped-call]
    assert out.volume == pytest.approx(ball.volume, rel=0.1)


def test_a_smooth_hull_edge_collapse_cannot_thin_still_meets_the_cap(
    cfg: CollisionConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    from video2sim.stages.collision import _limit_vertices

    monkeypatch.setitem(sys.modules, "fast_simplification", None)
    can = trimesh.creation.cylinder(radius=0.03, height=0.2, sections=800)
    out = _limit_vertices(can, cfg.max_part_vertices)
    assert len(out.vertices) <= cfg.max_part_vertices
    assert trimesh.convex.is_convex(out)  # type: ignore[no-untyped-call]
    assert out.volume == pytest.approx(can.volume, rel=0.02)


class _StubMesh:
    def __init__(self, vertices: Any, faces: Any) -> None:
        self.vertices = vertices
        self.faces = faces


class _StubCoacd:
    """A fake ``coacd`` module that returns canned boxes and records its inputs."""

    Mesh = _StubMesh

    def __init__(self, mode: str = "halves") -> None:
        self.mode = mode
        self.calls: list[tuple[int, float, int, int, int]] = []

    def run_coacd(self, mesh: Any, **kw: Any) -> list[tuple[Any, Any]]:
        self.calls.append(
            (
                len(np.asarray(mesh.vertices)),
                float(kw["threshold"]),
                int(kw["max_convex_hull"]),
                int(kw["resolution"]),
                int(kw["mcts_iterations"]),
            )
        )
        u = 0.05
        if self.mode == "halves" and int(kw["max_convex_hull"]) > 4:
            bottom = trimesh.creation.box(extents=(2 * u, u, u))
            bottom.apply_translation([u, u / 2, u / 2])
            left = trimesh.creation.box(extents=(u, u, u))
            left.apply_translation([u / 2, 3 * u / 2, u / 2])
            return [
                (np.asarray(bottom.vertices), np.asarray(bottom.faces)),
                (np.asarray(left.vertices), np.asarray(left.faces)),
            ]
        strip = trimesh.creation.box(extents=(2 * u, u, u))
        strip.apply_translation([u, u / 2, u / 2])
        return [(np.asarray(strip.vertices), np.asarray(strip.faces))]


def _stubbed(
    cfg: CollisionConfig, monkeypatch: pytest.MonkeyPatch, **kw: Any
) -> tuple[_StubCoacd, CollisionConfig]:
    import dataclasses

    stub = _StubCoacd(str(kw.pop("mode", "halves")))
    monkeypatch.setitem(sys.modules, "coacd", stub)
    return stub, dataclasses.replace(cfg, **kw)


def test_decomposition_shortcuts_are_on_by_default(cfg: CollisionConfig) -> None:
    assert cfg.decomp_decimate and cfg.adaptive_escalation and cfg.decomp_cache


def test_cheap_rung_wins_and_full_rung_backs_it_up(
    cfg: CollisionConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub, cfg = _stubbed(cfg, monkeypatch)
    res = collision_geometry(lshape(), cfg)
    assert [c[1:3] for c in stub.calls] == [
        (cfg.cheap_threshold, cfg.cheap_max_parts),
        (cfg.coacd_threshold, cfg.max_convex_parts),
    ]
    assert res.ok and len(res.parts) == 2
    assert res.steps["decomp_rung_cheap_failed"] == 1
    assert res.steps["decomp_rung_full"] == 1


def test_search_knobs_reach_coacd(cfg: CollisionConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    stub, cfg = _stubbed(
        cfg, monkeypatch, coacd_resolution=500, coacd_mcts_iterations=20, adaptive_escalation=False
    )
    collision_geometry(lshape(), cfg)
    assert [(c[3], c[4]) for c in stub.calls] == [(500, 20)]


def test_hull_rung_wins_for_a_near_convex_mesh_without_calling_coacd(
    cfg: CollisionConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub, cfg = _stubbed(cfg, monkeypatch, primitive_tolerance=1e-6)
    m = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    v = np.asarray(m.vertices)
    v[0] -= [0.002, 0.002, 0.002]  # a dent no primitive may claim and no hull hides badly
    m.vertices = v
    res = collision_geometry(m, cfg)
    assert stub.calls == []
    assert res.kind == "convex_hull" and res.ok and len(res.parts) == 1
    assert res.steps["decomp_rung_hull"] == 1


def test_decimated_copy_feeds_coacd_and_the_toggle_restores_the_full_mesh(
    cfg: CollisionConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("fast_simplification")
    stub, cfg = _stubbed(cfg, monkeypatch)
    big = lshape()
    for _ in range(5):  # midpoint splits: more faces, the same surface
        big = big.subdivide()
    assert big.is_watertight and len(big.faces) > cfg.decomp_max_faces
    res = collision_geometry(big, cfg)
    assert res.steps["decomp_decimated"] == 1
    assert res.steps["decomp_input_faces"] <= cfg.decomp_max_faces
    # Decimated to decomp_max_faces faces, so far below the full mesh's vertices.
    assert all(c[0] <= cfg.decomp_max_faces for c in stub.calls)
    assert all(c[0] < len(big.vertices) for c in stub.calls)
    assert res.ok and res.steps["decomp_rung_full"] == 1
    full_stub, full_cfg = _stubbed(cfg, monkeypatch, decomp_decimate=False)
    collision_geometry(big, full_cfg)
    assert full_stub.calls and all(c[0] > 5000 for c in full_stub.calls)


def test_cache_reuses_parts_and_ignores_corruption(
    cfg: CollisionConfig, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stub, cfg = _stubbed(cfg, monkeypatch)
    first = collision_geometry(lshape(), cfg, cache_dir=tmp_path)
    assert len(stub.calls) == 2 and first.ok  # cheap rung fails, full rung wins
    assert first.steps["decomp_rung_hull_failed"] == 1
    second = collision_geometry(lshape(), cfg, cache_dir=tmp_path)
    assert len(stub.calls) == 2  # no new CoACD call
    assert second.steps["decomp_cache_hit"] == 1 and second.ok
    assert [p.volume for p in second.parts] == pytest.approx([p.volume for p in first.parts])
    for p in tmp_path.glob("*.npz"):
        p.write_bytes(b"not a zip file")
    third = collision_geometry(lshape(), cfg, cache_dir=tmp_path)
    assert len(stub.calls) == 4  # corrupt entry recomputed
    assert third.ok


def test_no_cache_without_a_cache_dir(
    cfg: CollisionConfig, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stub, cfg = _stubbed(cfg, monkeypatch)
    collision_geometry(lshape(), cfg)
    collision_geometry(lshape(), cfg, cache_dir=tmp_path)
    collision_geometry(lshape(), cfg)
    assert len(stub.calls) == 6  # two rungs each, nothing cached without a dir


def test_without_escalation_only_the_full_rung_runs(
    cfg: CollisionConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub, cfg = _stubbed(cfg, monkeypatch, adaptive_escalation=False)
    res = collision_geometry(lshape(), cfg)
    assert [c[1:3] for c in stub.calls] == [(cfg.coacd_threshold, cfg.max_convex_parts)]
    assert res.ok and res.steps["decomp_rung_full"] == 1


def test_a_thin_walled_cup_survives_the_decimated_path(cfg: CollisionConfig) -> None:
    pytest.importorskip("coacd")
    pytest.importorskip("fast_simplification")
    profile = [[0, -0.05], [0.04, -0.05], [0.04, 0.05], [0.035, 0.05], [0.035, -0.04], [0, -0.04]]
    cup = trimesh.creation.revolve(np.array(profile), sections=2000)
    assert cup.is_watertight and len(cup.faces) > cfg.decomp_max_faces
    res = collision_geometry(cup, cfg)
    assert res.steps["decomp_decimated"] == 1
    assert res.kind == "convex_decomposition" and res.ok


def test_decomposition_ranges_are_validated() -> None:
    for key, bad in [
        ("decomp_max_faces", 3),
        ("decomp_volume_change_frac", 1.5),
        ("coacd_resolution", 0),
        ("coacd_mcts_iterations", 0),
        ("cheap_threshold", 0.0),
        ("cheap_max_parts", 0),
    ]:
        bad_cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
        bad_cfg["physics"]["collision"][key] = bad
        with pytest.raises(ValueError):
            CollisionConfig.from_config(bad_cfg)


def test_coacd_runs_in_two_threads_after_setup() -> None:
    coacd = pytest.importorskip("coacd")
    prepare_coacd()
    mesh = trimesh.creation.annulus(0.05, 0.1, height=0.05)

    def run(_: int) -> int:
        solid = coacd.Mesh(np.asarray(mesh.vertices, float), np.asarray(mesh.faces, np.int64))
        return len(coacd.run_coacd(solid, threshold=0.2, max_convex_hull=4, seed=0))

    with ThreadPoolExecutor(2) as pool:
        counts = list(pool.map(run, range(2)))
    assert counts[0] == counts[1] >= 1


def test_the_ladder_checks_each_rung_once_and_returns_its_body(
    cfg: CollisionConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    from video2sim.stages import collision as collision_module
    from video2sim.stages.mass import sealed_body

    _stub, cfg = _stubbed(cfg, monkeypatch)
    real = collision_module.check_parts
    calls: list[int] = []

    def counted(*args: Any, **kwargs: Any) -> dict[str, bool]:
        calls.append(len(args[0]))
        return real(*args, **kwargs)

    monkeypatch.setattr(collision_module, "check_parts", counted)
    mesh = lshape()
    res = collision_geometry(mesh, cfg)
    assert res.steps["decomp_rung_full"] == 1
    # the hull, cheap and full rungs, and no fourth check of the full rung's parts
    assert calls == [1, 1, 2]
    assert res.checks == real(res.parts, mesh, cfg, body=res.body)
    assert res.body is not None
    assert np.array_equal(res.body.voxels, sealed_body(mesh, cfg.sealing).voxels)


def test_a_cached_decomposition_is_still_checked(
    cfg: CollisionConfig, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stub, cfg = _stubbed(cfg, monkeypatch)
    first = collision_geometry(lshape(), cfg, cache_dir=tmp_path)
    second = collision_geometry(lshape(), cfg, cache_dir=tmp_path)
    assert second.steps.get("decomp_cache_hit") == 1
    assert second.checks == first.checks and second.ok
