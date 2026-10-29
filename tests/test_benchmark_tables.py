import importlib.util
import json
from pathlib import Path
from typing import Any

from video2sim.evaluation.benchmark_tables import (
    build_scenes,
    load_table,
    main,
    station_to_mm,
    write_scenes,
)

INVENTORY = """,,,
,,,
object_id,label,length_mm_1,length_mm_2,width_mm_1,width_mm_2,height_mm_1,height_mm_2,mass_g,material_class,hollow,deformable,person
obj_000,box,100,102,50,50,20,20,250,cardboard,FALSE,FALSE,T. K.
obj_001,can,67,68,67,68,240,240,,metal,TRUE,FALSE,T. K.
obj_002,wide,40,40,60,60,10,10,5,plastic,FALSE,FALSE,T. K.
obj_003,unused,10,10,10,10,10,10,1,wood,FALSE,FALSE,T. K.
obj_004,,,,,,,,,,,,T. K.
"""

SCENES = """scene_id,tier,split,board,marker_family,tag_size_mm,object_id,x_mm,y_mm,yaw_deg
s1,easy,dev,grid,aruco_4x4_50,60,obj_000,200,100,30
s1,easy,dev,grid,aruco_4x4_50,60,obj_001,300,100,0
s1,easy,dev,grid,aruco_4x4_50,60,obj_099,100,100,0
s2,medium,test,grid,aruco_4x4_50,60,obj_000,200,100,
s2,medium,test,grid,aruco_4x4_50,60,obj_002,,,10
"""


def _load(tmp_path: Path) -> tuple[dict[str, Any], list[str]]:
    inv_path, scn_path = tmp_path / "inv.csv", tmp_path / "scn.csv"
    inv_path.write_text(INVENTORY, encoding="utf-8")
    scn_path.write_text(SCENES, encoding="utf-8")
    inv, w1 = load_table(inv_path, ["object_id"])
    scn, w2 = load_table(scn_path, ["scene_id", "object_id"])
    scenes, w3 = build_scenes(inv, scn)
    return scenes, [*w1, *w2, *w3]


def test_complete_object_is_converted_to_meters_and_kg(tmp_path: Path) -> None:
    scenes, _ = _load(tmp_path)
    obj = scenes["s1"]["objects"][0]
    assert obj["size_m"] == [0.101, 0.05, 0.02]  # mean of the repeated readings
    assert obj["mass_kg"] == 0.25
    assert obj["deformable"] is False
    assert obj["position_m"] == [0.2, 0.1, 0.01]  # z is half the height
    assert obj["incomplete"] == []
    assert scenes["s1"]["marker"]["tag_size_m"] == 0.06


def test_blank_rows_above_header_are_skipped_with_warning(tmp_path: Path) -> None:
    _, warnings = _load(tmp_path)
    assert any("2 blank rows" in w for w in warnings)


def test_missing_mass_is_flagged_and_still_written(tmp_path: Path) -> None:
    scenes, warnings = _load(tmp_path)
    obj = scenes["s1"]["objects"][1]
    assert obj["mass_kg"] is None
    assert obj["incomplete"] == ["mass_g"]
    assert any("obj_001" in w and "mass_g" in w for w in warnings)


def test_object_missing_from_inventory_is_flagged(tmp_path: Path) -> None:
    scenes, warnings = _load(tmp_path)
    assert scenes["s1"]["objects"][2]["incomplete"] == ["inventory_row"]
    assert any("obj_099 is not in the inventory" in w for w in warnings)


def test_missing_position_and_yaw_are_flagged(tmp_path: Path) -> None:
    scenes, _ = _load(tmp_path)
    assert scenes["s2"]["objects"][0]["incomplete"] == ["yaw_deg"]
    assert "position_m" in scenes["s2"]["objects"][1]["incomplete"]


def test_length_below_width_is_flagged(tmp_path: Path) -> None:
    scenes, warnings = _load(tmp_path)
    assert "size_m" in scenes["s2"]["objects"][1]["incomplete"]
    assert any("obj_002" in w for w in warnings)


def test_reused_and_unused_objects_are_reported(tmp_path: Path) -> None:
    _, warnings = _load(tmp_path)
    assert any("obj_000: placed in several scenes" in w for w in warnings)
    assert any("obj_003: in the inventory but in no scene" in w for w in warnings)
    assert not any("obj_004: in the inventory but in no scene" in w for w in warnings)


def test_conflicting_scene_fields_are_reported() -> None:
    rows = [
        {"scene_id": "s", "object_id": "a", "tier": "easy"},
        {"scene_id": "s", "object_id": "b", "tier": "hard"},
    ]
    _, warnings = build_scenes([], rows)
    assert any("tier differs" in w for w in warnings)


def test_unsafe_scene_id_is_refused() -> None:
    scenes, warnings = build_scenes([], [{"scene_id": "../x", "object_id": "a"}])
    assert scenes == {}
    assert any("invalid scene_id" in w for w in warnings)


def test_cli_writes_json_and_strict_sets_exit_code(tmp_path: Path) -> None:
    (tmp_path / "inv.csv").write_text(INVENTORY, encoding="utf-8")
    (tmp_path / "scn.csv").write_text(SCENES, encoding="utf-8")
    args = [
        str(tmp_path / "scn.csv"),
        "--inventory",
        str(tmp_path / "inv.csv"),
        "--out",
        str(tmp_path / "out"),
    ]
    assert main(args) == 0
    assert main([*args, "--strict"]) == 1
    data = json.loads((tmp_path / "out" / "s1" / "ground_truth.json").read_text("utf-8"))
    assert [o["object_id"] for o in data["objects"]] == ["obj_000", "obj_001", "obj_099"]
    assert main([str(tmp_path / "scn.csv"), "--inventory", str(tmp_path / "missing.csv")]) == 2


def test_write_scenes_creates_one_file_per_scene(tmp_path: Path) -> None:
    scenes, _ = _load(tmp_path)
    assert len(write_scenes(scenes, tmp_path / "o")) == 2


def test_legacy_obj_id_column_is_accepted(tmp_path: Path) -> None:
    path = tmp_path / "inv.csv"
    path.write_text("obj_id,label\nobj_000,box\n", encoding="utf-8")
    rows, warnings = load_table(path, ["object_id"])
    assert rows == [{"object_id": "obj_000", "label": "box"}]
    assert warnings == []


def _make_board() -> Any:
    path = Path(__file__).parents[1] / "benchmark" / "make_board.py"
    spec = importlib.util.spec_from_file_location("make_board", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_station_names_match_the_printed_board() -> None:
    # Round trip against the generator for all three sheet sizes, large and small stations.
    board = _make_board()
    for size in board.PAGES_MM:
        _, layout = board.build(size, board.DEFAULT_TAG_MM[size], clean=False)
        for station in layout["stations"]:
            assert station_to_mm(station["name"]) == (station["x_mm"], station["y_mm"])


def test_station_examples() -> None:
    assert station_to_mm("B2") == (200.0, 200.0)
    assert station_to_mm("b2") == (150.0, 150.0)
    assert station_to_mm("c3") == (250.0, 250.0)
    assert station_to_mm("A1") == (100.0, 100.0)
    for bad in ("", "B", "2B", "B0", "BB2", "B-2", "B2x"):
        assert station_to_mm(bad) is None


def _one_object(position: dict[str, str], board: str = "grid", **kwargs: Any) -> tuple[Any, Any]:
    inventory = [{"object_id": "o", "length_mm_1": "100", "width_mm_1": "50", "height_mm_1": "20"}]
    row = {"scene_id": "s", "object_id": "o", "board": board, "yaw_deg": "0", **position}
    scenes, warnings = build_scenes(inventory, [row], **kwargs)
    return scenes["s"]["objects"][0], warnings


def test_station_sets_position_in_meters() -> None:
    obj, _ = _one_object({"station": "B2"})
    assert obj["position_m"] == [0.2, 0.2, 0.01]
    assert "position_m" not in obj["incomplete"]


def test_millimeters_still_work_on_a_grid_board() -> None:
    obj, _ = _one_object({"x_mm": "123.5", "y_mm": "40"})
    assert obj["position_m"] == [0.1235, 0.04, 0.01]


def _stack_rows(notes: list[str]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    heights = {"a": "20", "b": "50", "c": "40"}
    inventory = [
        {"object_id": k, "length_mm_1": "100", "width_mm_1": "50", "height_mm_1": h}
        for k, h in heights.items()
    ]
    rows = [
        {
            "scene_id": "s",
            "object_id": k,
            "board": "grid",
            "station": "B2",
            "yaw_deg": "0",
            "notes": n,
        }
        for k, n in zip(heights, notes, strict=True)
    ]
    return inventory, rows


def test_stacked_objects_sit_on_the_ones_below() -> None:
    inventory, rows = _stack_rows(["stack 0 (lowest)", "stack 1 (ontop of a)", "stack 2"])
    scenes, warnings = build_scenes(inventory, rows)
    z = [o["position_m"][2] for o in scenes["s"]["objects"]]
    assert z == [0.01, 0.045, 0.09]  # 20 / 2, 20 + 50 / 2, 20 + 50 + 40 / 2 (mm)
    assert not [w for w in warnings if "stack" in w]


def test_stack_order_follows_the_numbers_not_the_row_order() -> None:
    inventory, rows = _stack_rows(["stack 2", "stack 0", "stack 1"])
    scenes, _ = build_scenes(inventory, rows)
    z = {o["object_id"]: o["position_m"][2] for o in scenes["s"]["objects"]}
    assert z == {"a": 0.1, "b": 0.025, "c": 0.07}  # b 0-50 mm, c 50-90, a 90-110


def test_a_stack_with_a_gap_is_flagged_and_left_on_the_board() -> None:
    inventory, rows = _stack_rows(["stack 0", "stack 2", "stack 3"])
    scenes, warnings = build_scenes(inventory, rows)
    assert [o["position_m"][2] for o in scenes["s"]["objects"]] == [0.01, 0.025, 0.02]
    assert any("not 0, 1, 2" in w for w in warnings)


def test_station_and_millimeters_together_is_a_conflict() -> None:
    obj, warnings = _one_object({"station": "B2", "x_mm": "200", "y_mm": "200"})
    assert "position_m" in obj["incomplete"]
    assert any("both station" in w for w in warnings)


def test_station_on_clean_board_and_bad_name_are_flagged() -> None:
    obj, warnings = _one_object({"station": "B2"}, board="clean")
    assert "position_m" in obj["incomplete"]
    assert any("clean board" in w for w in warnings)
    obj, warnings = _one_object({"station": "22"})
    assert "position_m" in obj["incomplete"]
    assert any("not a name like" in w for w in warnings)


def test_no_position_is_flagged() -> None:
    obj, warnings = _one_object({})
    assert "position_m" in obj["incomplete"]
    assert any("no position" in w for w in warnings)


def test_station_missing_from_the_printed_board_is_flagged() -> None:
    obj, warnings = _one_object({"station": "A1"}, known_stations={"B2"})
    assert obj["position_m"] == [0.1, 0.1, 0.01]
    assert any("not on the printed board" in w for w in warnings)
    _, warnings = _one_object({"station": "B2"}, known_stations={"B2"})
    assert not any("printed board" in w for w in warnings)


def test_cli_board_layout_option(tmp_path: Path) -> None:
    (tmp_path / "inv.csv").write_text(INVENTORY, encoding="utf-8")
    (tmp_path / "scn.csv").write_text(
        "scene_id,board,object_id,station\ns,grid,obj_000,A1\n", encoding="utf-8"
    )
    layout = tmp_path / "layout.json"
    layout.write_text('{"stations": [{"name": "B2"}]}', encoding="utf-8")
    args = [
        str(tmp_path / "scn.csv"),
        "--inventory",
        str(tmp_path / "inv.csv"),
        "--out",
        str(tmp_path / "o"),
    ]
    assert main([*args, "--board-layout", str(layout), "--strict"]) == 1
    assert main([*args, "--board-layout", str(tmp_path / "nope.json")]) == 2


def test_missing_deformable_flag_is_a_gap() -> None:
    obj, _ = _one_object({"station": "B2"})  # the helper's inventory row has no flags
    assert {"hollow", "deformable"} <= set(obj["incomplete"])


def test_numeric_object_ids_and_repeated_mass_readings(tmp_path: Path) -> None:
    inv = tmp_path / "inv.csv"
    scn = tmp_path / "scn.csv"
    inv.write_text(
        "obj_id,label,length_mm_1,width_mm_1,height_mm_1,mass_g_1,mass_g_2,mass_g_3,"
        "material_class,hollow,deformable\n"
        "obj_007,box,100,50,20,100,102,101,cardboard,FALSE,FALSE\n",
        encoding="utf-8",
    )
    scn.write_text(
        "scene_id,tier,split,board,marker_family,tag_size_mm,object_id,x_mm,y_mm,yaw_deg\n"
        "s1,easy,dev,clean,aruco_4x4_50,100,7,200,100,30\n",
        encoding="utf-8",
    )
    rows, _ = load_table(inv, ["object_id"])
    placed, _ = load_table(scn, ["scene_id", "object_id"])
    scenes, warnings = build_scenes(rows, placed)
    obj = scenes["s1"]["objects"][0]
    assert obj["object_id"] == "obj_007"
    assert obj["mass_kg"] == 0.101
    assert obj["incomplete"] == []
    assert warnings == []
