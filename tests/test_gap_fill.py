"""Gap fill after tracking (S12.9, task 9.6)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from video2sim.models.hosted import parse_sam_text
from video2sim.schemas import SE3, CameraFrame, Intrinsics, Plane, WorldModel
from video2sim.stages.gap_fill import (
    CLUSTER_ID_BASE,
    NOUN_ID_BASE,
    NOUN_ID_STEP,
    FitPass,
    GapFillConfig,
    GapPlan,
    GapRequest,
    ResidualParams,
    board_hull,
    box_request,
    execute_gap_fill,
    inherit_provenance,
    merge_requests,
    offset_tracks,
    plan_gap_fill,
    residual_clusters,
    table_region,
    unexplained_clusters,
)
from video2sim.stages.lift import LiftConfig, PointVote
from video2sim.stages.marker_filter import MarkerFilterConfig
from video2sim.stages.primitive_fit import PrimitiveConfig
from video2sim.stages.track_repair import RepairConfig
from video2sim.stages.tracking import Track
from video2sim.stages.view_select import ViewConfig

ROOT = Path(__file__).resolve().parent.parent
SHAPE = (48, 64)  # (height, width) of the synthetic masks


def _mask(rows: slice, cols: slice) -> NDArray[np.bool_]:
    canvas = np.zeros(SHAPE, dtype=bool)
    canvas[rows, cols] = True
    return canvas


def _track(tid: int, frames: dict[int, NDArray[np.bool_]]) -> Track:
    return Track(tid, dict(frames))


def _gcfg(**over: object) -> GapFillConfig:
    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    g = cfg["segmentation"]["tracking"]["gap_fill"]
    g["enabled"] = True
    for key, value in over.items():
        g[key] = value
    return GapFillConfig.from_config(cfg)


def _vcfg() -> ViewConfig:
    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    return ViewConfig.from_config(cfg)


def _world(n: int = 4) -> WorldModel:
    frames = []
    for i in range(n):
        a = np.diag([1.0, -1.0, -1.0, 1.0])
        a[:3, 3] = [0.2 * i, 0.0, 1.0]
        frames.append(
            CameraFrame(
                frame_index=i, image_path=f"{i:05d}.png", T_world_from_camera=SE3.from_numpy(a)
            )
        )
    return WorldModel(
        intrinsics=Intrinsics(fx=50.0, fy=50.0, cx=32.0, cy=24.0, width_px=64, height_px=48),
        frames=frames,
        scale_factor=1.0,
        scale_source="marker",
        table_plane=Plane(normal=(0.0, 0.0, 1.0), offset_m=0.0),
        point_cloud_path="points.ply",
        mean_reprojection_error_px=0.5,
        registered_fraction=1.0,
    )


def test_merge_duplicate_part_and_new() -> None:
    cfg = _gcfg()
    box = _mask(slice(10, 20), slice(10, 20))
    generic = [_track(0, {0: box, 1: box, 2: box})]
    # Same masks on the shared frames: a duplicate, whatever its id.
    dup = _track(1000, {0: box, 1: box, 2: box})
    # A quarter of the generic mask on most shared frames: median IoU low, a part.
    quarter = _mask(slice(10, 15), slice(10, 15))
    part = _track(1001, {0: quarter, 1: quarter, 2: box})
    # No shared frame with anything: new (the repair rules cover it after the fact).
    new = _track(1002, {5: box})
    out = merge_requests(
        generic,
        [GapRequest((dup, part, new), "noun", "noun 'mug'")],
        cfg,
    )
    assert [t.track_id for t in out.tracks] == [0, 1002]
    assert out.notes[0] == ("noun 'mug'",)
    assert out.dropped[1001].startswith("part of obj_000")
    assert out.notes[1002] == ("noun 'mug'",)


def test_merge_one_shared_frame_needs_every_frame() -> None:
    cfg = _gcfg()
    box = _mask(slice(10, 20), slice(10, 20))
    generic = [_track(0, {0: box, 1: box})]
    # IoU 1.0 on frame 0 but 0.25 on frame 1: not the same object on every frame.
    other = _mask(slice(10, 20), slice(30, 40))
    tricky = _track(1000, {0: box, 1: other})
    out = merge_requests(generic, [GapRequest((tricky,), "noun", "noun 'box'")], cfg)
    assert [t.track_id for t in out.tracks] == [0, 1000]


def test_merge_remaps_colliding_ids() -> None:
    cfg = _gcfg()
    box = _mask(slice(10, 20), slice(10, 20))
    generic = [_track(0, {0: box}), _track(1000, {3: box})]
    clash = _track(1000, {5: box})
    out = merge_requests(generic, [GapRequest((clash,), "noun", "noun 'box'")], cfg)
    assert [t.track_id for t in out.tracks] == [0, 1000, 1001]
    assert out.remapped == ((1000, 1001),)


def _box_tuple(
    gated: dict[int, tuple[int, tuple[int, int, int, int]]],
) -> tuple[tuple[int, int, tuple[int, int, int, int]], ...]:
    return tuple((tid, frame, box) for tid, (frame, box) in gated.items())


def test_box_gates_reject_table_patches() -> None:
    cfg = _gcfg()
    box = _mask(slice(10, 20), slice(10, 20))
    generic = [_track(0, {0: box})]
    # Covers the box but sprawls past it: overfill (IoU 0.625 passes first).
    sprawl = _track(2000, {0: _mask(slice(10, 26), slice(10, 20))})
    # Barely inside the box: the IoU gate binds before the fill gate.
    speck = _track(2001, {0: _mask(slice(12, 13), slice(12, 13))})
    # No mask on the prompted frame at all.
    missing = _track(2002, {1: box})
    out = merge_requests(
        generic,
        [
            GapRequest(
                (sprawl, speck, missing),
                "cluster_box",
                "cluster",
                _box_tuple({i: (0, (10, 10, 19, 19)) for i in (2000, 2001, 2002)}),
            )
        ],
        cfg,
    )
    assert [t.track_id for t in out.tracks] == [0]
    assert "overfill" in out.dropped[2000]
    assert "IoU" in out.dropped[2001]
    assert "no mask" in out.dropped[2002]


def test_box_fill_gate_binds_when_iou_is_loose() -> None:
    # IoU <= fill always, so the fill gate only binds when min_iou <= min_fill.
    cfg = _gcfg(box_mask_min_iou=0.01)
    box = _mask(slice(10, 20), slice(10, 20))
    speck = _track(2000, {0: _mask(slice(12, 13), slice(12, 13))})
    out = merge_requests(
        [_track(0, {0: box})],
        [GapRequest((speck,), "cluster_box", "cluster", _box_tuple({2000: (0, (10, 10, 19, 19))}))],
        cfg,
    )
    assert [t.track_id for t in out.tracks] == [0]
    assert "fill" in out.dropped[2000]


def test_box_passing_track_is_new() -> None:
    cfg = _gcfg()
    box = _mask(slice(10, 20), slice(10, 20))
    generic = [_track(0, {0: _mask(slice(30, 40), slice(30, 40))})]
    good = _track(2000, {0: box})
    out = merge_requests(
        generic,
        [GapRequest((good,), "cluster_box", "cluster", _box_tuple({2000: (0, (8, 8, 21, 21))}))],
        cfg,
    )
    assert [t.track_id for t in out.tracks] == [0, 2000]


def test_inherit_provenance_prefers_generic() -> None:
    from video2sim.stages.track_repair import RepairReport

    prov = inherit_provenance({0: "generic", 5: "noun"}, RepairReport(splits=(), merges=((0, 5),)))
    assert prov[0] == "generic"
    prov = inherit_provenance({5: "noun"}, RepairReport(splits=((5, (5, 9)),), merges=()))
    assert prov[9] == "noun"


def test_offset_tracks_shifts_ids() -> None:
    tracks = offset_tracks([_track(3, {0: _mask(slice(0, 2), slice(0, 2))})], 1000)
    assert tracks[0].track_id == 1003


def test_unexplained_clusters_split_explained_and_noise() -> None:
    world = _world()
    rng = np.random.default_rng(0)
    kept_blob = rng.normal([0.0, 0.0, 0.05], 0.005, (30, 3))
    far_blob = rng.normal([0.5, 0.0, 0.05], 0.005, (40, 3))
    near_kept = rng.normal([0.0, 0.0, 0.05], 0.005, (20, 3))
    noise = rng.uniform([-1.0, -1.0, 0.05], [1.0, 1.0, 0.06], (5, 3))
    cloud = np.concatenate([kept_blob, far_blob, near_kept, noise])
    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    gcfg = GapFillConfig.from_config(cfg)
    out = unexplained_clusters(
        world,
        cloud,
        {0: kept_blob},
        Path("/none"),
        "aruco",
        gcfg.cluster,
        0.003,
        0.35,
        detect=lambda p: ({}, (48, 64)),
    )
    assert len(out) == 1
    assert abs(out[0][:, 0].mean() - 0.5) < 0.05


def _vote(n: int, explained: slice) -> PointVote:
    ratio = np.zeros(n)
    ratio[explained] = 0.9
    seen = np.full(n, 10, dtype=np.int64)
    track = np.where(ratio >= 0.5, 0, -1).astype(np.int64)
    return PointVote(ratio, track, seen)


def test_table_region_keeps_only_the_origin_table() -> None:
    world = _world()
    rng = np.random.default_rng(7)
    table = rng.uniform([-0.1, -0.1, 0.0], [0.1, 0.1, 0.001], (40, 3))
    desk = rng.uniform([1.9, -0.1, 0.0], [2.1, 0.1, 0.001], (40, 3))  # far desk at table height
    wall = rng.uniform([2.0, 0.5, 0.0], [2.0, 0.6, 0.3], (20, 3))
    cloud = np.concatenate([table, desk, wall])
    inside = table_region(cloud, world, 0.003, 0.05, 0.10)
    assert inside[:40].all()
    assert not inside[40:].any()


def test_residual_clusters_two_blobs_explained_wall_board() -> None:
    world = _world()
    rng = np.random.default_rng(8)
    # Table surface around the origin so the table region forms.
    table = rng.uniform([-0.15, -0.15, 0.0], [0.15, 0.15, 0.001], (60, 3))
    blob_a = rng.normal([0.10, 0.0, 0.05], 0.004, (30, 3))
    blob_b = rng.normal([-0.10, 0.05, 0.05], 0.004, (25, 3))
    explained_blob = rng.normal([0.0, -0.10, 0.05], 0.004, (30, 3))
    wall = rng.uniform([2.0, 0.0, 0.05], [2.0, 0.2, 0.06], (30, 3))
    board_blob = rng.normal([0.0, 0.0, 0.01], 0.002, (20, 3))  # inside the tag hull below
    cloud = np.concatenate([table, blob_a, blob_b, explained_blob, wall, board_blob])
    n_table = len(table)
    vote = _vote(len(cloud), slice(n_table + 55, n_table + 85))  # the explained blob
    hull = np.array([[-0.05, -0.05], [0.05, -0.05], [0.05, 0.05], [-0.05, 0.05]])
    out = residual_clusters(world, cloud, vote, ResidualParams(min_points=12), 0.003, 0.35, hull)
    assert len(out) == 2
    centers = sorted(float(c[:, 0].mean()) for c in out)
    assert abs(centers[0] - -0.10) < 0.03 and abs(centers[1] - 0.10) < 0.03


def test_board_hull_triangulates_tag_corners() -> None:
    world = _world(n=1)

    def tags(_: Path) -> tuple[dict[int, NDArray[np.float64]], tuple[int, int]]:
        quad = np.array([[10.0, 10.0], [54.0, 10.0], [54.0, 38.0], [10.0, 38.0]])
        return {0: quad}, (48, 64)

    hull = board_hull(world, Path("/none"), "aruco", detect=tags)
    assert hull is not None
    # u=10..54, v=10..38 at fx=50, cx=32, cy=24 from z=1: x in [-0.44, 0.44], y in [-0.28, 0.28].
    assert abs(float(hull[:, 0].min()) - -0.44) < 0.02
    assert abs(float(hull[:, 0].max()) - 0.44) < 0.02
    assert board_hull(world, Path("/none"), "aruco", detect=lambda p: ({}, (48, 64))) is None


def test_board_clusters_are_not_unexplained() -> None:
    world = _world()
    rng = np.random.default_rng(1)
    blob = rng.normal([0.3, 0.0, 0.05], 0.005, (30, 3))
    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    gcfg = GapFillConfig.from_config(cfg)

    def tags(_: Path) -> tuple[dict[int, NDArray[np.float64]], tuple[int, int]]:
        quad = np.array([[0.0, 0.0], [64.0, 0.0], [64.0, 48.0], [0.0, 48.0]])
        return {0: quad}, (48, 64)

    out = unexplained_clusters(
        world, blob, {}, Path("/none"), "aruco", gcfg.cluster, 0.003, 0.35, detect=tags
    )
    assert out == []


def test_box_projection_matches_hand_math() -> None:
    world = _world(n=1)
    # A 4 cm cube centred at (0, 0, 0.05): projects to u = 50 x + 32, v = -50 y + 24.
    cube = np.array([[x, y, z] for x in (-0.02, 0.02) for y in (-0.02, 0.02) for z in (0.03, 0.07)])
    from video2sim.stages.gap_fill import _frame_box

    placed = _frame_box(world, 0, cube, 0)
    assert placed is not None
    (x1, y1, x2, y2), point = placed
    assert (x1, y1, x2, y2) == (30, 22, 33, 25)
    assert point == (32.0, 24.0)


def test_prompt_round_trip_through_a_fake_client() -> None:
    world = _world(n=2)
    rng = np.random.default_rng(2)
    blob = rng.normal([0.05, 0.0, 0.05], 0.004, (30, 3))
    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    gcfg = GapFillConfig.from_config(cfg)
    vcfg = _vcfg()
    from video2sim.stages.view_select import compute_visibility

    visibility, lo, shape = compute_visibility(world, blob, vcfg)
    prompt, gated, unprompted = box_request(
        world, visibility, (lo, shape), vcfg.cell_m, [blob], (64, 48), gcfg.cluster
    )
    assert unprompted == []
    assert list(gated) == [2000]
    # The fake client replays a recorded-style answer for the prompted id and frames.
    answer = "\n".join(
        f"<{f}f>2000<|box;x1=30;y1=20;x2=34;y2=26;w=64;h=48|><|mask;x=0;y=0;data=1,1,~x|>"
        for f in (0, 1)
    )
    records = parse_sam_text(answer)
    assert [(r.object_id, r.frame) for r in records] == [(2000, 0), (2000, 1)]
    assert prompt.startswith("<") and "2000<|box;" in prompt


def test_plan_caps_requests_and_reports_coverage() -> None:
    world = _world()
    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    cfg["segmentation"]["tracking"]["gap_fill"]["max_video_requests"] = 2
    gcfg = GapFillConfig.from_config(cfg)
    vcfg = _vcfg()
    rng = np.random.default_rng(3)
    blob = rng.normal([0.05, 0.0, 0.05], 0.004, (30, 3))
    order = [f.frame_index for f in world.frames]
    plan = plan_gap_fill(
        world,
        blob,
        ["mug", "cup", "ball"],
        [blob],
        [0, 1],
        order,
        vcfg,
        vcfg.cell_m,
        1,
        gcfg,
    )
    assert plan.send_box
    assert plan.nouns == ()  # 1 generic + 1 box fills the cap of 2
    assert plan.nouns_dropped == ("mug", "cup", "ball")
    assert plan.video_requests == 2
    assert 0.0 <= plan.view_coverage <= 1.0


def test_gap_fill_defaults_off_until_dev_evaluation() -> None:
    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    gcfg = GapFillConfig.from_config(cfg)
    # PLAN.md 9.6/9.9: defaults off until 9.8 evaluates and 9.9 decides them.
    assert gcfg.enabled is False  # off keeps the stage byte-identical
    assert gcfg.audit.enabled is False and gcfg.audit.n_views >= 1
    assert gcfg.merge_new_iou <= gcfg.merge_iou
    assert cfg["segmentation"]["router"]["views"]["mode"] == "even"
    from video2sim.stages.scene_router import Route
    from video2sim.stages.segmentation_stage import route_report
    from video2sim.stages.track_repair import RepairReport

    route = Route("table", "objects on a table", ("mug",), "test", False, None)
    summary = route_report(route, [], [], RepairReport(splits=(), merges=()))
    assert "inventory" not in summary and "view_frames" not in summary and "gap_fill" not in summary


def test_plan_defaults_send_nouns_then_box() -> None:
    world = _world()
    gcfg = _gcfg()
    vcfg = _vcfg()
    order = [f.frame_index for f in world.frames]
    plan = plan_gap_fill(
        world,
        np.zeros((10, 3)),
        ["mug", "cup"],
        [],
        [0, 1],
        order,
        vcfg,
        vcfg.cell_m,
        1,
        gcfg,
    )
    assert not plan.send_box and plan.box_prompt is None
    assert plan.nouns == ("mug", "cup")
    assert plan.video_requests == 3  # generic + two nouns, under the cap of 8


def _plan(
    world: WorldModel,
    blob: NDArray[np.float64],
    nouns: Sequence[str],
    clusters: Sequence[NDArray[np.float64]],
    frames: Sequence[int],
    order: Sequence[int],
    vcfg: ViewConfig,
    gcfg: GapFillConfig,
    audit: Sequence[str] = (),
) -> GapPlan:
    return plan_gap_fill(
        world,
        blob,
        nouns,
        clusters,
        frames,
        order,
        vcfg,
        vcfg.cell_m,
        1,
        gcfg,
        audit_nouns=audit,
    )


def test_plan_batches_audit_nouns_into_one_request() -> None:
    world = _world()
    gcfg, vcfg = _gcfg(), _vcfg()
    order = [f.frame_index for f in world.frames]
    plan = _plan(
        world,
        np.zeros((10, 3)),
        ["mug", "cup"],
        [],
        [0, 1],
        order,
        vcfg,
        gcfg,
        audit=["laptop", "tape dispenser"],
    )
    assert plan.audit_nouns == ("laptop", "tape dispenser")
    assert plan.audit_prompt == "laptop, tape dispenser"
    assert plan.nouns == ("mug", "cup")
    assert plan.video_requests == 4  # generic + batch + two nouns, under the cap of 8


def test_plan_batch_takes_reserved_budget_first() -> None:
    world = _world()
    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    cfg["segmentation"]["tracking"]["gap_fill"]["max_video_requests"] = 2
    gcfg, vcfg = GapFillConfig.from_config(cfg), _vcfg()
    order = [f.frame_index for f in world.frames]
    plan = _plan(
        world,
        np.zeros((10, 3)),
        ["mug"],
        [],
        [0, 1],
        order,
        vcfg,
        gcfg,
        audit=["laptop"],
    )
    assert plan.audit_nouns == ("laptop",)
    assert plan.nouns == () and plan.nouns_dropped == ("mug",)
    assert plan.video_requests == 2


def test_merge_new_tracks_use_the_lower_bar() -> None:
    cfg = _gcfg()
    first = _mask(slice(10, 20), slice(10, 20))
    shifted = _mask(slice(10, 20), slice(14, 24))  # IoU 60/140 against first
    generic = [_track(0, {0: first, 1: first, 2: first})]
    noun = _track(1000, {0: shifted, 1: shifted, 2: shifted})
    box = _track(2000, {0: shifted, 1: shifted, 2: shifted})
    out = merge_requests(
        generic,
        [
            GapRequest((noun,), "noun", "noun 'laptop'"),
            GapRequest(
                (box,),
                "cluster_box",
                "unexplained point cluster",
                ((2000, 0, (14, 10, 23, 19)),),
            ),
        ],
        cfg,
    )
    # 0.43 clears merge_new_iou (0.35) but not merge_iou (0.5): the box twin joins
    # the noun track while the noun track stays separate from the generic one.
    assert [t.track_id for t in out.tracks] == [0, 1000]
    assert out.notes[1000] == ("noun 'laptop'", "unexplained point cluster")


def test_plan_sends_nouns_only_on_a_shortfall() -> None:
    world = _world()
    gcfg, vcfg = _gcfg(), _vcfg()
    order = [f.frame_index for f in world.frames]

    def plan(shortfall: int) -> GapPlan:
        return plan_gap_fill(
            world,
            np.zeros((10, 3)),
            ["mug", "cup"],
            [],
            [0, 1],
            order,
            vcfg,
            vcfg.cell_m,
            1,
            gcfg,
            shortfall=shortfall,
        )

    no_shortfall = plan(0)
    assert no_shortfall.nouns == () and no_shortfall.nouns_dropped == ()
    assert no_shortfall.video_requests == 1
    short = plan(1)
    assert short.nouns == ("mug", "cup") and short.video_requests == 3


def test_clusters_far_from_every_kept_track_are_not_unexplained() -> None:
    world = _world()
    rng = np.random.default_rng(5)
    kept_blob = rng.normal([0.0, 0.0, 0.05], 0.005, (30, 3))
    room_blob = rng.normal([2.0, 0.0, 0.05], 0.005, (40, 3))
    cloud = np.concatenate([kept_blob, room_blob])
    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    gcfg = GapFillConfig.from_config(cfg)
    out = unexplained_clusters(
        world,
        cloud,
        {0: kept_blob},
        Path("/none"),
        "aruco",
        gcfg.cluster,
        0.003,
        0.35,
        detect=lambda p: ({}, (48, 64)),
    )
    assert out == []


def test_box_request_tries_more_frames_until_a_box_fits(monkeypatch: Any) -> None:
    world = _world(n=4)
    rng = np.random.default_rng(6)
    blob = rng.normal([0.05, 0.0, 0.05], 0.004, (30, 3))
    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    gcfg = GapFillConfig.from_config(cfg)
    vcfg = _vcfg()
    from video2sim.stages import gap_fill as gap_module
    from video2sim.stages.view_select import compute_visibility

    visibility, lo, shape = compute_visibility(world, blob, vcfg)
    real_frame_box = gap_module._frame_box
    calls = {"n": 0}

    def flaky(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] <= 2:
            return None  # the best-ranked frames cannot place the box
        return real_frame_box(*args, **kwargs)

    monkeypatch.setattr(gap_module, "_frame_box", flaky)
    _, gated, unprompted = box_request(
        world, visibility, (lo, shape), vcfg.cell_m, [blob], (64, 48), gcfg.cluster
    )
    assert calls["n"] > 2
    assert list(gated) == [2000] and unprompted == []


def test_board_check_detects_tags_once_per_frame_for_all_clusters() -> None:
    world = _world()
    rng = np.random.default_rng(4)
    blob_a = rng.normal([0.5, 0.0, 0.05], 0.005, (40, 3))
    blob_b = rng.normal([-0.5, 0.0, 0.05], 0.005, (40, 3))
    cloud = np.concatenate([blob_a, blob_b])
    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    gcfg = GapFillConfig.from_config(cfg)
    calls: list[str] = []

    def detect(path: Path) -> tuple[dict[int, NDArray[np.float64]], tuple[int, int]]:
        calls.append(path.name)
        return {}, (48, 64)

    out = unexplained_clusters(
        world, cloud, {}, Path("/none"), "aruco", gcfg.cluster, 0.003, 0.35, detect=detect
    )
    assert len(out) == 2
    assert sorted(calls) == [f"{i:05d}.png" for i in range(4)]


def test_execute_reuses_first_pass_when_no_new_track_survives(tmp_path: Path) -> None:
    world = _world()
    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    generic = _track(0, {0: _mask(slice(10, 20), slice(10, 20))})
    first = FitPass([generic], {0: np.zeros((3, 3))}, {}, {}, {7: "only 3 voxels agree"})
    plan = GapPlan(
        nouns=("mug", "cup"),
        nouns_dropped=(),
        send_box=False,
        box_prompt=None,
        box_gated=(),
        box_unprompted=(),
        clusters_total=0,
        view_frames=[0, 1],
        view_coverage=0.0,
        video_requests=3,
    )
    offsets: list[int] = []

    def track_fn(prompt: str, offset: int) -> tuple[list[Track], bool]:
        offsets.append(offset)
        return [], False

    second, discovery, _, prov = execute_gap_fill(
        world,
        tmp_path,
        plan,
        _gcfg(),
        LiftConfig.from_config(cfg),
        PrimitiveConfig.from_config(cfg),
        RepairConfig.from_config(cfg),
        MarkerFilterConfig.from_config(cfg),
        np.zeros((3, 3)),
        {f.frame_index: f.T_world_from_camera for f in world.frames},
        [generic],
        shortfall=2,
        listed=3,
        track_fn=track_fn,
        first=first,
    )
    assert second is first
    assert discovery.tracked == 1 and discovery.recovered == 0
    assert prov == {0: "generic"}
    # Requests run in parallel, so offsets arrive in any order; the noun ids
    # live in their own range, clear of the box-prompt ids.
    assert sorted(offsets) == [NOUN_ID_BASE, NOUN_ID_BASE + NOUN_ID_STEP]
    assert not (CLUSTER_ID_BASE <= max(offsets) < CLUSTER_ID_BASE + 1000)
