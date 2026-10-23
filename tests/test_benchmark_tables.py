import json
from pathlib import Path
from typing import Any

from video2sim.evaluation.benchmark_tables import build_scenes, load_table, main, write_scenes

INVENTORY = """,,,
,,,
object_id,label,length_mm_1,length_mm_2,width_mm_1,width_mm_2,height_mm_1,height_mm_2,mass_g,material_class,hollow,person
obj_000,box,100,102,50,50,20,20,250,cardboard,FALSE,T. K.
obj_001,can,67,68,67,68,240,240,,metal,TRUE,T. K.
obj_002,wide,40,40,60,60,10,10,5,plastic,FALSE,T. K.
obj_003,unused,10,10,10,10,10,10,1,wood,FALSE,T. K.
obj_004,,,,,,,,,,,T. K.
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
    args = [str(tmp_path / "inv.csv"), str(tmp_path / "scn.csv"), "--out", str(tmp_path / "out")]
    assert main(args) == 0
    assert main([*args, "--strict"]) == 1
    data = json.loads((tmp_path / "out" / "s1" / "ground_truth.json").read_text("utf-8"))
    assert [o["object_id"] for o in data["objects"]] == ["obj_000", "obj_001", "obj_099"]
    assert main([str(tmp_path / "missing.csv"), str(tmp_path / "scn.csv")]) == 2


def test_write_scenes_creates_one_file_per_scene(tmp_path: Path) -> None:
    scenes, _ = _load(tmp_path)
    assert len(write_scenes(scenes, tmp_path / "o")) == 2


def test_legacy_obj_id_column_is_accepted(tmp_path: Path) -> None:
    path = tmp_path / "inv.csv"
    path.write_text("obj_id,label\nobj_000,box\n", encoding="utf-8")
    rows, warnings = load_table(path, ["object_id"])
    assert rows == [{"object_id": "obj_000", "label": "box"}]
    assert warnings == []
