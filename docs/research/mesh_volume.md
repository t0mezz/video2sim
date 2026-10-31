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
| Fill after sealing gaps up to 4% of the longest side | 1.01 | 32 |

43 of 56 meshes have mesh / fill below 0.5, and only 2 are above 0.85.

## Cracks and cups look the same to a plain fill

A plain fill floods the inside of any mesh with a gap, and then returns little more than the wall.
That is wrong for a cracked box and right for a cup, whose cavity really is open. The first version
of this note used the hull whenever the fill was below half of it. **That rule is wrong for cups:**
it scored 34 of 39 only because the cups have no primitive truth, and it gave the mugs 635 and
602 cm3 against about 92 and 104 cm3 of ceramic.

Sealing separates the two. Close gaps up to a radius, fill, take the radius back, and watch the
volume as the radius grows (`sealed_volumes`). A crack seals early and the volume jumps to the
solid. A mouth stays open until the radius reaches about half its width.

Sealed volume as a share of the hull, radius as a share of the longest side:

| Object | 1% | 2% | 4% | 8% | 16% | Reading |
| --- | --- | --- | --- | --- | --- | --- |
| Rubik's cube smaller, 3 runs | 0.20 to 0.23 | 0.22 to 1.01 | 1.0 | 1.0 | 1.0 | Crack |
| Package for electronics, s02 | 0.19 | 1.02 | 1.02 | 1.02 | 1.02 | Crack |
| Package for breakfast bags, s13 | 0.31 | 1.01 | 1.01 | 1.01 | 1.01 | Crack |
| Mug, 2 objects | 0.20 | 0.20 | 0.20 | 0.21 | 0.24 | Open vessel |
| Glass water bottle | 0.28 | 0.30 | 0.31 | 0.96 | 0.97 | Open, narrow neck |
| Deodorant cap, s02 | 0.13 | 0.13 | 0.13 | 0.13 | 0.14 | Open vessel |
| Gamepad, plier | 0.25 to 0.59 | 0.27 to 0.59 | 0.27 to 0.59 | 0.28 to 0.60 | 0.29 to 0.61 | Concave solid |

4% is the working value. It seals every crack above and keeps every mug, cap and bottle open. 8%
already closes the bottle neck, so the safe band is narrow: 2% misses two Rubik's cubes and 8% loses
the bottle. A vessel with a mouth under 8% of its longest side will be sealed as a solid.

A mesh wall is real here. The mug section shows separate inner and outer walls with an open mouth,
so these meshes hold a real cavity, unlike the closed thin shells of boxes.

| Object | Sealed at 4%, cm3 | True material, cm3 | Mass ratio |
| --- | --- | --- | --- |
| Mug obj_041 | 126 | 92 | 1.38 |
| Mug obj_046 | 118 | 104 | 1.13 |
| Glass water bottle | 330 | 236 | 1.40 |

True material is weighed mass over table density, so it assumes the table is right. The excess is
at least partly the voxel: one voxel is about 1 mm on a mug, and a thin wall gains that on each
side. A finer grid should reduce it; that was not tested.

Remaining misses at 4% against a primitive (7 of 39):

* The s02 deodorant cap (0.13) and s12 cap (0.71): a cap is open, so its cylinder is not its truth.
  The s02 value is right, 3 cm3 against 3 g of plastic. The two runs disagree with each other.
* Round hair brush (0.25): the cylinder is not its truth either.
* Package for electronics, s13 (1.59): a proxy hull with a 66% size error.
* White 3D print cuboid (0.75), in-ear package (0.79) and breadboard (1.25): shape or size errors.

## Density is now the main error

Mass here is nominal table density times volume, as a solid. **No hollow correction is applied.**

| Objects | Volume used | Median mass ratio | Within 30% |
| --- | --- | --- | --- |
| Inventory hollow (27) | Mesh | 0.65 | 7 |
| Inventory hollow (27) | Sealed at 4% | 3.76 | 2 |
| Inventory not hollow (29) | Mesh | 0.39 | 4 |
| Inventory not hollow (29) | Sealed at 4% | 2.43 | 2 |

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

## What P4.5 does, and what is left for P4.6

Implemented in `stages/mass.py`, scored in `benchmark/results/p4_5_mass.json`
(`benchmark/run_mass.py`) on the same 56 objects, using the model's hollow answer:

1. The body is the sealed fill at 4% of the longest side. There is no hull fallback, because it
   fills cups. Inertia is integrated over the same voxels.
2. An open vessel keeps its cavity and `f = 1`. A body under 0.5 of its hull counts as open.
3. A closed hollow object gets `V_eff = f * V_sealed`. `f` is one value for all classes, 0.024,
   0.22 and 0.61 (10th, 50th and 90th percentile of the `f` each of the 31 such objects needs).
4. A solid answer keeps `f = 1`.

| Objects | Median mass ratio | Within 30% | Within 30% with no correction | Truth inside low to high |
| --- | --- | --- | --- | --- |
| Closed, model says hollow (31) | 1.00 | 11 | 1 | 26 |
| Open, model says hollow (4) | 1.25 | 2 | 2 | 2 |
| Model says solid (21) | 2.40 | 1 | 1 | 3 |
| All (56) | 1.25 | 14 | 4 | 31 |

**14 of 56 is 25%, against the 70% exit criterion.** The closed row is in sample, since `f` came
from those objects, so a median of 1.00 is by construction and says nothing about accuracy.

Left for P4.6:

* The largest gap is the solid row. Non-hollow is not enough to mean solid: prints with infill and
  metal cans are in it, and the inventory `hollow` column marks them the same way, so check those
  rows before fitting.
* Fit `f` per class. The spread of 25 times in the global value is mostly different classes.
  Keep the fit apart from the objects it is scored on.
* `f` divides by nominal density, so it absorbs table errors too.

## Limits of the mass estimate

* `f` dilutes the body evenly. An empty box has its mass in the walls, so its inertia per mass is
  higher than this gives, by up to 5/3 for a thin spherical shell. There is no measured inertia on
  the benchmark to size this; only the closed forms are checked.
* Low and high carry the density and `f` ranges, not the scale error, which counts three times in
  mass. On solid objects, where `f = 1`, the truth is outside the range on 18 of 21.
* The sealed body's center of mass is a median 1.7 mm from the mesh centroid, the object-frame
  origin. P4.8 has to move the origin to it.
* The 0.5 hull test also catches concave solids. A hollow answer on one, such as the chocolate
  santa, gets no correction.

## Limits of the volume test

* The voxel fill reads high by design: +3 to +6% on analytic shapes at 128 voxels along the longest
  side (`self_check` in the script), worst on thin boxes, and more on thin walls. Bias is not
  removed from the numbers above.
* Sealing was tuned on few objects: 2 mugs, 2 caps and 1 bottle are the only open vessels in the
  matched dev set, and the white round basket and other wide containers were not among them. The
  4% value has not been run on the test split.
* `sealed_volumes` has no analytic check on a cup, because the mesh boolean backend is not
  installed; the mugs and bottle against the weighed masses are the check.
* True volume is a primitive from caliper sizes, so a real rounded box is slightly less.
* The 17 objects without primitive truth, including the concave ones where the hull is weakest,
  have no volume score. They are in the mass rows.
* 7 of the 56 objects have a size error above 20%, and that error is inside these volumes.
* Density ratios use the model's class, which agrees with the inventory on 40 of 56.
