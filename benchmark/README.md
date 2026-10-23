# Benchmark

20 tabletop scenes, about 60 objects, with ground truth. This is how we know whether a change
helped. Videos and meshes are large and may show people or private spaces, so they are
**not in git**; `benchmark/scenes/` is ignored. Where they live is decision D-006 in PLAN.md.

## Tiers

| Tier | Scenes | What it contains |
| --- | --- | --- |
| easy | 10 | Matte, textured, well separated objects |
| medium | 6 | Clutter and partial occlusion |
| hard | 4 | Reflective, thin or dark objects |

Prefer objects with published meshes (the YCB set, for example) because their shape ground
truth is free. Measure the others with calipers and a kitchen scale.

## Splits

12 **dev** scenes and 8 **test** scenes (D-009), stratified by tier:

| Tier | Dev | Test |
| --- | --- | --- |
| easy | 6 | 4 |
| medium | 4 | 2 |
| hard | 2 | 2 |

Three of the easy scenes use a clean board (production tags only, no printed grid): two in dev,
one in test. Boards are generated with `benchmark/make_board.py`.

Tune and debug on dev. Run test only at phase gates, and never tune on it. If a result only
holds on dev, it does not count.

## Layout of one scene

```
benchmark/scenes/<scene_id>/
  video_a.mp4          first camera path
  video_b.mp4          second path, different from the first
  ground_truth.json    see below
  notes.md             anything odd about the capture
```

Capture each scene on the printed grid with the scale marker in view (see
`docs/capture_protocol.md`). The grid exists only to produce position and orientation ground
truth. The pipeline itself needs only the scale marker.

## `ground_truth.json`

```json
{
  "scene_id": "s01_easy_mugs",
  "tier": "easy",
  "split": "dev",
  "marker": {"family": "aruco_4x4_50", "tag_size_m": 0.060, "origin": "board corner"},
  "objects": [
    {
      "object_id": "obj_000",
      "label": "mug",
      "size_m": [0.082, 0.082, 0.095],
      "mass_kg": 0.31,
      "position_m": [0.20, 0.15, 0.0475],
      "yaw_deg": 35.0,
      "hollow": true,
      "material_class": "ceramic",
      "reference_mesh": null
    }
  ]
}
```

The values above are placeholders showing the shape of the file, not measurements.

Rules:

* `position_m` is the object's volume centroid in the world frame (+Z up, origin on the
  board), measured from the grid. `yaw_deg` is rotation about +Z.
* `size_m` is the axis-aligned bounding box in the object's resting pose: `[length, width,
  height]`, with `length >= width`. `height` is the vertical extent in the resting pose, so it
  can be larger or smaller than either horizontal side. When `length` and `width` are equal
  (cylinders, cubes), the horizontal axes are interchangeable, `yaw_deg` is arbitrary, and the
  orientation metric skips the object. Checks: `video2sim.evaluation.ground_truth`.
* Weigh every object. Mass ground truth is what calibrates and scores the physics stage.
* Never edit ground truth to make a number pass. If a measurement was wrong, fix it, note why
  in `notes.md`, and say so in the PR.

## Sheets to `ground_truth.json`

Ground truth is kept in two Google Sheets, exported as CSV (File, Download, CSV). Header in
row 1, one table per sheet, `.` as decimal separator.

**Inventory**, one row per physical object: `object_id` (the older name `obj_id` is accepted),
`label`, `length_mm_1..3`, `width_mm_1..3`, `height_mm_1..3` (repeated readings, averaged),
`mass_g`, `material_class`, `hollow` and `deformable` (TRUE or FALSE), `shape_primitive` (the older name
`reference_mesh` is accepted). Other columns are ignored.

**Scenes**, one row per object placed in a scene. Scene-level columns repeat on every row of
the scene. See `scenes_example.csv`; its positions are made up.

| Column | Meaning |
| --- | --- |
| `scene_id` | Becomes the folder name; letters, digits, `_` and `-` only |
| `tier`, `split`, `board` | `easy` / `medium` / `hard`, `dev` / `test`, `grid` / `clean` |
| `marker_family`, `tag_size_mm` | Scale marker, for example `aruco_4x4_50` and `60` |
| `object_id` | Row in the inventory |
| `station` | Grid scenes: `B2` (capital, large station on the 100 mm lattice, x = 200, y = 200) or `c3` (lower case, small station in the middle of the cell, x = 250, y = 250). Same names as printed by `make_board.py` |
| `x_mm`, `y_mm` | Tape-measured scenes: position from the board origin (bottom-left of the grid area, +X along the long edge, +Y up). Leave empty when `station` is set |
| `yaw_deg` | Rotation about +Z, see the size convention above |

A position is either a `station` or `x_mm` and `y_mm`, never both. The position is where the
object's bounding-box center sits, so put the object's center on the cross of the station. A
station only fixes the position to a few millimetres; read the yaw from the dial.

```bash
uv run python -m video2sim.evaluation.benchmark_tables scenes.csv     --board-layout board_a2.json
```

The inventory defaults to `benchmark/inventory.csv`; `--inventory PATH`
overrides it. `--board-layout` takes the JSON that `make_board.py` writes next to the PDF. It adds a warning
for a station that is not printed on that board, because stations near the tags are left out.

This writes `benchmark/scenes/<scene_id>/ground_truth.json` and logs one warning for every
gap: an object missing from the inventory, a missing size, mass, material, hollow flag,
position or yaw, a station name that is malformed or used on a clean board, a scene field that differs between rows, an object in no scene or in several.
An object with gaps is still written, with the gap names in `incomplete`, so the evaluation
can skip it. `--strict` exits with 1 on any warning. The z of `position_m` is half the
height, the bounding-box center, which differs from the volume centroid for cones and mugs.
The JSON has `shape_primitive` where the sketch above shows `reference_mesh`; a mesh path is
not recorded yet.
