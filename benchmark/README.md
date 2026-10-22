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

Proposed: 12 **dev** scenes and 8 **test** scenes, mixed across tiers. Tune and debug on dev.
Run test only at phase gates, and never tune on it. If a result only holds on dev, it does not
count.

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
* `size_m` is the axis-aligned bounding box in the object's resting pose, length by width by
  height.
* Weigh every object. Mass ground truth is what calibrates and scores the physics stage.
* Never edit ground truth to make a number pass. If a measurement was wrong, fix it, note why
  in `notes.md`, and say so in the PR.
