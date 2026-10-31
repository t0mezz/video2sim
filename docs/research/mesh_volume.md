# Mesh volume: solid or shell? (P4.5)

Written 2026-10-31. Question: can mass be computed from the volume of a registered SAM 3D mesh?
Evidence file: `benchmark/results/p4_5_mesh_volume.json` (`benchmark/measure_mesh_volume.py`).
Data: the 56 matched dev objects on 12 scenes in `runs_instances/` (54 SAM 3D, 2 proxy hulls), all
passing repair. s05 is left out because its ground-truth origin is not established. True volume
comes from the measured primitive, so the volume ratios cover the 39 box and cylinder objects.
The 17 others have only a bounding box. Dev split only; the test split was not touched.

## Result

**The mesh volume cannot be used. Almost all meshes are thin shells.** The volume inside the
outer surface is right, but table density times the right volume is still far from the weighed
mass, so the remaining error is density, not geometry.

| Volume | Median ratio to true volume | Within 20% (of 39) |
| --- | --- | --- |
| Mesh, signed | 0.13 | 0 |
| Voxel fill of the outer surface | 0.96 | 28 |
| Convex hull | 1.03 | 33 |
| Fill, or hull when fill / hull < 0.5 | 1.01 | 34 |

43 of 56 meshes have mesh / fill below 0.5, and only 2 are above 0.85.

## The fill leaks on open shells

The fill floods the inside when the shell has a gap wider than it closes, and then returns little
more than the wall. The Rubik's cube (smaller), the deodorant cap and two packages read 0.14 to
0.37 of true volume by fill and about 1.0 by hull. Below fill / hull of 0.7, 15 of 56 objects:

* Leaks, where the hull is right: those above.
* Really concave objects, where neither is right: the round hair brush (hull 0.36 of its measured
  cylinder, which is itself a poor primitive for a brush) and probably the plier, mugs and gamepad,
  which have no primitive truth.

The rule above gives the same result for any threshold from 0.4 to 0.7, because no object sits
between 0.59 and 0.7 and those between 0.4 and 0.59 have no primitive truth or are the brush.
That is a small sample, so 0.5 is a starting value and not a measured limit.

The hull is wrong in one known way: it overstates concave objects, and when it replaces a leaked
fill on one it gives too much mass. A mesh with a leak cannot tell which case it is.

## Density is now the main error

Mass here is nominal table density times volume, as a solid. **No hollow correction is applied.**

| Objects | Volume used | Median mass ratio | Within 30% |
| --- | --- | --- | --- |
| Inventory hollow (27) | Mesh | 0.65 | 7 |
| Inventory hollow (27) | Fill or hull | 4.55 | 0 |
| Inventory not hollow (29) | Mesh | 0.39 | 4 |
| Inventory not hollow (29) | Fill or hull | 2.49 | 2 |

The mesh line scores better only by accident: a wall a tenth of the volume cancels a density far
too high. It also fails for the solid objects. It is not a usable result.

True density (weighed mass over true volume) against the table's nominal:

| Class from the model | Inventory | Objects | True density, median | Table nominal |
| --- | --- | --- | --- | --- |
| `cardboard` | hollow | 13 | 218 | 929 |
| `plastic_hard` | not hollow | 13 | 489 (95 to 1080) | 1100 |
| `plastic_hard` | hollow | 2 | 171 | 1100 |
| `foam` | hollow | 3 | 223 | 60 |
| `metal_steel` | not hollow | 5 | 182 | 7850 |

Why they differ:

* The table holds the density of the material, while a real object is mostly air: a box, a 3D print
  with infill, a can. The hollow correction has to close this gap, and it is a factor of 2 to 40.
* s11 printed shapes come out 2.3 to 5.8 times too heavy. The inventory lists them as not hollow,
  but a print with infill is not solid resin, which `material_table.yaml` already warns about.
* The coffee can comes out about 43 times too heavy. The model says steel, and the object is a
  mixed can (inventory cardboard) with `obj_004` still undecided.
* The small paint can (7.7 times) and the hole punch (6.7 times) show the same for metal: thin
  sheet over air, listed as not hollow.

## What to do in P4.5 and P4.6

1. Use the outer volume: voxel fill, or the hull when fill / hull is below a config threshold,
   starting at 0.5. Inertia needs the same solid, not the shell.
2. Do not take the wall from the mesh. The SAM 3D wall is an artefact, not the real one.
3. Treat `is_hollow` as the correction: `V_eff = f * V_outer`, with `f` a fill fraction per material
   class fitted in P4.6 on dev objects. Mass is then `rho * f * V_outer`. Low and high must come
   from the spread of `f`, not only from the table range.
4. Non-hollow is not enough to mean solid. The inventory `hollow` column still marks prints and
   metal cans as not hollow, so check those rows before fitting.

## Limits

* The voxel fill reads high by design: +3 to +6% on analytic shapes at 128 voxels along the longest
  side (`self_check` in the script), worst on thin boxes. Bias is not removed from the numbers above.
* True volume is a primitive from caliper sizes, so a real rounded box is slightly less.
* The 17 objects without primitive truth, including the concave ones where the hull is weakest,
  have no volume score. They are in the mass rows.
* 7 of the 56 objects have a size error above 20%, and that error is inside these volumes.
* Density ratios use the model's class, which agrees with the inventory on 40 of 56.
