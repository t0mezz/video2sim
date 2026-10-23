"""Build per-scene ``ground_truth.json`` files from the two benchmark sheets (P0.2).

The sheets are exported from Google Sheets as CSV:

* inventory: one row per physical object (size, mass, material).
* scenes: one row per object placed in a scene (position, yaw) plus the scene-level fields,
  repeated on every row of that scene. The position is a grid station such as B2 or b2, or
  x_mm and y_mm for scenes measured by tape.

Nothing is dropped quietly. Every missing or inconsistent value becomes a warning, and an
object with missing values is still written, with the names of the gaps in ``incomplete`` so
the evaluation can skip it. Column layout: ``benchmark/README.md``.

Run: ``python -m video2sim.evaluation.benchmark_tables SCENES.csv``
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from video2sim.evaluation.ground_truth import size_convention_errors

logger = logging.getLogger(__name__)

TIERS = ("easy", "medium", "hard")
SPLITS = ("dev", "test")
BOARDS = ("grid", "clean")
SCENE_FIELDS = ("tier", "split", "board", "marker_family", "tag_size_mm")
SIZE_AXES = ("length", "width", "height")
SAFE_ID = re.compile(r"[A-Za-z0-9_-]+")
STATION = re.compile(r"([A-Za-z])([0-9]+)")
# Lattice of the printed board, `benchmark/make_board.py` (STATION_MM). Large stations
# sit on the lattice, small ones in the middle of each cell.
STATION_MM = 100.0
# The sheet pre-fills these on rows that are still placeholders for objects not yet entered.
PREFILLED = ("object_id", "person")
# Older column names still used by the inventory sheet, mapped to the current ones.
ALIASES = {"obj_id": "object_id"}

Row = dict[str, str]
Scene = dict[str, Any]


def _number(text: str) -> float | None:
    try:
        return float(text.strip().replace(",", "."))
    except ValueError:
        return None


def load_table(path: Path, required: Sequence[str]) -> tuple[list[Row], list[str]]:
    """Read a CSV exported from Sheets. Leading blank rows are skipped with a warning."""
    warnings: list[str] = []
    # utf-8-sig: Excel adds a BOM, which would otherwise end up in the first column name.
    with path.open(encoding="utf-8-sig", newline="") as f:
        raw = list(csv.reader(f))
    skipped = 0
    while raw and not any(cell.strip() for cell in raw[0]):
        raw.pop(0)
        skipped += 1
    if skipped:
        warnings.append(f"{path.name}: {skipped} blank rows above the header were skipped")
    if not raw:
        return [], [*warnings, f"{path.name}: no header row"]
    header = [ALIASES.get(h.strip(), h.strip()) for h in raw[0]]
    missing = [c for c in required if c not in header]
    if missing:
        return [], [*warnings, f"{path.name}: missing columns {missing}"]
    rows: list[Row] = []
    for cells in raw[1:]:
        if not any(cell.strip() for cell in cells):
            continue
        padded = [*cells, *[""] * (len(header) - len(cells))]
        rows.append({h: cell.strip() for h, cell in zip(header, padded, strict=False) if h})
    return rows, warnings


def station_to_mm(name: str) -> tuple[float, float] | None:
    """Board position in mm of a station name, or None if the name is malformed.

    "B2" (capital) is a large station at x = 200, y = 200. "b2" (lower case) is a small
    station in the middle of cell b2, at x = 150, y = 150. Same scheme as `make_board.py`.
    """
    match = STATION.fullmatch(name)
    if match is None or int(match.group(2)) < 1:
        return None
    letter, number = match.group(1), int(match.group(2))
    if letter.isupper():
        return (ord(letter) - ord("A") + 1) * STATION_MM, number * STATION_MM
    half = STATION_MM / 2.0
    return (ord(letter) - ord("a")) * STATION_MM + half, (number - 1) * STATION_MM + half


def _position_mm(
    where: str, row: Row, board: str | None, known: set[str] | None
) -> tuple[tuple[float, float] | None, list[str]]:
    """Resolve the position of one placed object from `station` or from `x_mm` and `y_mm`."""
    station, x_text, y_text = row.get("station", ""), row.get("x_mm", ""), row.get("y_mm", "")
    if station:
        if x_text or y_text:
            return None, [f"{where}: both station {station!r} and x_mm / y_mm are set"]
        if board == "clean":
            return None, [f"{where}: station {station!r} on a clean board, which has no grid"]
        position = station_to_mm(station)
        if position is None:
            return None, [f"{where}: station {station!r} is not a name like B2 or b2"]
        if known is not None and station not in known:
            # Stations near the tags or the footer are not printed, so no object stood there.
            return position, [f"{where}: station {station!r} is not on the printed board"]
        return position, []
    x, y = _number(x_text), _number(y_text)
    if x is None or y is None:
        return None, [f"{where}: no position, give station or x_mm and y_mm"]
    return (x, y), []


def _is_placeholder(row: Row) -> bool:
    return not any(v for k, v in row.items() if k not in PREFILLED)


def _object_from_inventory(row: Row) -> tuple[Scene, list[str]]:
    """Convert one inventory row to a ground-truth object. Returns it and its gaps."""
    gaps: list[str] = []
    size_mm: list[float] = []
    for axis in SIZE_AXES:
        values = [_number(row.get(f"{axis}_mm_{i}", "")) for i in (1, 2, 3)]
        readings = [v for v in values if v is not None]
        if readings:
            size_mm.append(sum(readings) / len(readings))
        else:
            gaps.append(f"{axis}_mm")
    size_m = [round(v / 1000.0, 6) for v in size_mm] if len(size_mm) == 3 else None
    if size_m is not None and size_convention_errors(size_m):
        gaps.append("size_m")

    mass_g = _number(row.get("mass_g", ""))
    if mass_g is None or mass_g <= 0:
        mass_g = None
        gaps.append("mass_g")
    flags: dict[str, bool | None] = {}
    for name in ("hollow", "deformable"):
        flags[name] = {"true": True, "false": False}.get(row.get(name, "").lower())
        if flags[name] is None:
            gaps.append(name)
    if not row.get("material_class"):
        gaps.append("material_class")

    obj: Scene = {
        "object_id": row["object_id"],
        "label": row.get("label", ""),
        "size_m": size_m,
        "mass_kg": None if mass_g is None else round(mass_g / 1000.0, 6),
        "hollow": flags["hollow"],
        "deformable": flags["deformable"],
        "material_class": row.get("material_class") or None,
        # The sheet's older name for this column is reference_mesh, but it holds primitives.
        "shape_primitive": row.get("shape_primitive") or row.get("reference_mesh") or None,
    }
    return obj, gaps


def _check_inventory(inventory: list[Row]) -> tuple[dict[str, Row], list[str]]:
    warnings: list[str] = []
    by_id: dict[str, Row] = {}
    for row in inventory:
        oid = row.get("object_id", "")
        if not oid:
            warnings.append("inventory: a row has no object_id")
        elif oid in by_id:
            warnings.append(f"inventory: duplicate object_id {oid}, the first row is used")
        else:
            by_id[oid] = row
    return by_id, warnings


def build_scenes(
    inventory: list[Row], scene_rows: list[Row], known_stations: set[str] | None = None
) -> tuple[dict[str, Scene], list[str]]:
    """Join the scenes sheet with the inventory. Returns scenes by id and the warnings.

    With `known_stations` (names from the board layout file), a station that is not on the
    printed board gets a warning.
    """
    by_id, warnings = _check_inventory(inventory)

    grouped: dict[str, list[Row]] = defaultdict(list)
    for row in scene_rows:
        sid = row.get("scene_id", "")
        if not SAFE_ID.fullmatch(sid):
            # The id becomes a folder name, so anything with path characters is refused.
            warnings.append(f"scenes: row with invalid scene_id {sid!r} skipped")
            continue
        grouped[sid].append(row)

    scenes: dict[str, Scene] = {}
    placed: dict[str, list[str]] = defaultdict(list)
    for sid, rows in grouped.items():
        scenes[sid], scene_warnings = _build_scene(sid, rows, by_id, placed, known_stations)
        warnings.extend(scene_warnings)

    for oid, row in by_id.items():
        if len(placed[oid]) > 1:
            warnings.append(f"{oid}: placed in several scenes {placed[oid]}")
        elif not placed[oid] and not _is_placeholder(row):
            # Placeholder rows at the end of the sheet are not expected to be in a scene.
            warnings.append(f"{oid}: in the inventory but in no scene")
    return scenes, warnings


def _scene_fields(sid: str, rows: list[Row]) -> tuple[dict[str, str | None], list[str]]:
    warnings: list[str] = []
    fields: dict[str, str | None] = {}
    for name in SCENE_FIELDS:
        distinct = list(dict.fromkeys(r.get(name, "") for r in rows if r.get(name, "")))
        if not distinct:
            warnings.append(f"{sid}: {name} is empty on every row")
        elif len(distinct) > 1:
            warnings.append(f"{sid}: {name} differs between rows {distinct}, using {distinct[0]}")
        fields[name] = distinct[0] if distinct else None
    for name, allowed in (("tier", TIERS), ("split", SPLITS), ("board", BOARDS)):
        if fields[name] is not None and fields[name] not in allowed:
            warnings.append(f"{sid}: {name} {fields[name]!r} is not one of {list(allowed)}")
    return fields, warnings


def _build_scene(
    sid: str,
    rows: list[Row],
    inventory: dict[str, Row],
    placed: dict[str, list[str]],
    known_stations: set[str] | None,
) -> tuple[Scene, list[str]]:
    fields, warnings = _scene_fields(sid, rows)
    tag_mm = _number(fields["tag_size_mm"] or "")
    if fields["tag_size_mm"] is not None and (tag_mm is None or tag_mm <= 0):
        warnings.append(f"{sid}: tag_size_mm {fields['tag_size_mm']!r} is not a positive number")
        tag_mm = None

    objects: list[Scene] = []
    seen: set[str] = set()
    for row in rows:
        oid = row.get("object_id", "")
        if not oid:
            warnings.append(f"{sid}: a row has no object_id")
            continue
        if oid in seen:
            warnings.append(f"{sid}: {oid} appears twice, the second row is skipped")
            continue
        seen.add(oid)
        placed[oid].append(sid)

        inv = inventory.get(oid)
        if inv is None:
            warnings.append(f"{sid}: {oid} is not in the inventory")
            obj: Scene = {"object_id": oid, "size_m": None}
            gaps = ["inventory_row"]
        else:
            obj, gaps = _object_from_inventory(inv)
            if _is_placeholder(inv):
                warnings.append(f"{sid}: {oid} has an empty inventory row")

        position, position_warnings = _position_mm(
            f"{sid}: {oid}", row, fields["board"], known_stations
        )
        warnings.extend(position_warnings)
        yaw = _number(row.get("yaw_deg", ""))
        size_m = obj["size_m"]
        if position is None:
            gaps.append("position_m")
        if yaw is None:
            gaps.append("yaw_deg")
        if position is None or size_m is None:
            obj["position_m"] = None
        else:
            x, y = position
            # The object rests on the board, so the bounding-box center is half the height
            # up. It differs from the volume centroid for cones and mugs; see the README.
            obj["position_m"] = [
                round(x / 1000.0, 6),
                round(y / 1000.0, 6),
                round(size_m[2] / 2.0, 6),
            ]
        obj["yaw_deg"] = yaw
        obj["incomplete"] = gaps
        if gaps and inv is not None:
            warnings.append(f"{sid}: {oid} ({obj.get('label', '')}) incomplete: {gaps}")
        objects.append(obj)

    scene: Scene = {
        "scene_id": sid,
        "tier": fields["tier"],
        "split": fields["split"],
        "board": fields["board"],
        "marker": {
            "family": fields["marker_family"],
            "tag_size_m": None if tag_mm is None else tag_mm / 1000.0,
        },
        "objects": objects,
    }
    return scene, warnings


def write_scenes(scenes: dict[str, Scene], out_dir: Path) -> list[Path]:
    """Write ``<out_dir>/<scene_id>/ground_truth.json`` for every scene."""
    paths = []
    for sid, scene in scenes.items():
        path = out_dir / sid / "ground_truth.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(scene, indent=2) + "\n", encoding="utf-8")
        paths.append(path)
    return paths


def _known_stations(path: Path) -> set[str]:
    layout = json.loads(path.read_text(encoding="utf-8"))
    return {str(station["name"]) for station in layout["stations"]}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build ground_truth.json files from the sheets.")
    parser.add_argument("scenes", type=Path, help="scenes sheet exported as CSV")
    parser.add_argument(
        "--inventory",
        type=Path,
        default=Path("benchmark/inventory.csv"),
        help="inventory sheet exported as CSV (default: %(default)s)",
    )
    parser.add_argument("--out", type=Path, default=Path("benchmark/scenes"))
    parser.add_argument(
        "--board-layout",
        type=Path,
        help="layout JSON written by make_board.py; warns about stations that are not printed",
    )
    parser.add_argument("--strict", action="store_true", help="exit 1 if there is any warning")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        inventory, w_inv = load_table(args.inventory, ["object_id"])
        scene_rows, w_scn = load_table(args.scenes, ["scene_id", "object_id"])
        known = _known_stations(args.board_layout) if args.board_layout else None
    except (OSError, ValueError, KeyError) as e:
        logger.error("cannot read input: %r", e)
        return 2
    scenes, w_join = build_scenes(inventory, scene_rows, known)
    warnings = [*w_inv, *w_scn, *w_join]
    for message in warnings:
        logger.warning("%s", message)
    paths = write_scenes(scenes, args.out)
    n_objects = sum(len(s["objects"]) for s in scenes.values())
    n_gaps = sum(1 for s in scenes.values() for o in s["objects"] if o["incomplete"])
    logger.info(
        "%d scenes, %d objects (%d incomplete), %d warnings, wrote %d files to %s",
        len(scenes),
        n_objects,
        n_gaps,
        len(warnings),
        len(paths),
        args.out,
    )
    return 1 if (warnings and args.strict) else 0


if __name__ == "__main__":
    raise SystemExit(main())
