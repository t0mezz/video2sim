# Mesh volume and mass from meshes (P4.5-P4.6)

56 matched dev objects on 12 scenes in `runs_instances/` (54 SAM 3D, 2 proxy hulls), all passing repair; s05 excluded (no ground-truth origin). True volume comes from measured primitives, so volume ratios cover the 39 box/cylinder objects; 17 box-less objects enter only the mass rows. Dev only; test untouched. Evidence: `benchmark/results/p4_5_mesh_volume.json` (`benchmark/measure_mesh_volume.py`), `benchmark/results/p4_5_mass.json` (`benchmark/run_mass.py`), `benchmark/results/p4_6_fill_fit.json` + `p4_6_mass.json` (`benchmark/fit_fill_fractions.py`).

Signed mesh volume is unusable: almost all meshes are thin shells. Outer-surface volume is correct but table-density × correct volume still misses weighed mass — the residual is density, not geometry.

| Volume | Median ratio to true | Within 20% (of 39) |
| --- | --- | --- |
| Mesh, signed | 0.13 | 0 |
| Voxel fill of outer surface | 0.96 | 28 |
| Convex hull | 1.03 | 33 |
| Fill after sealing gaps ≤4% of longest side | 1.01 | 32 |

43 of 56 meshes have mesh/fill below 0.5; 2 exceed 0.85.

Cracks vs cups: plain fill floods any gapped mesh, which is wrong for cracked boxes and right for cups with genuinely open cavities. Hull-fallback-below-half-fill scored 34/39 only because cups lack primitive truth, while assigning mugs 635/602 cm³ against ~92/104 cm³ ceramic. Sealing separates them: close gaps up to a radius, fill, erode back, track volume vs radius. Cracks seal early and jump to solid; mouths stay open until radius ≈ half mouth width. Sealed volume as hull share (radius as longest-side share): Rubik's cubes 0.20-0.23 → 0.22-1.01 → 1.0 at 1%/2%/4% (crack); electronics/breakfast-bag packages 0.19/0.31 → ~1.0 by 2% (crack); mugs 0.20 flat to 16% (open); bottle 0.28/0.30/0.31 → 0.96 at 8% (narrow neck); s02 cap 0.13 flat (open); gamepad/plier 0.25-0.61 flat (concave solid). Working radius is 4%: seals all observed cracks, keeps mugs/caps/bottle open; 2% misses two Rubik's cubes, 8% closes the bottle neck. Mouths under 8% of longest side seal as solid. Mug cross-sections show real inner+outer walls with open mouths, unlike closed thin box shells. Sealed-vs-true-material volumes (weighed mass ÷ table density): mugs 126 vs 92 cm³ (1.38×), 118 vs 104 (1.13×); bottle 330 vs 236 (1.40×) — excess at least partly 1 mm-voxel wall thickening on both sides; finer grids untested. Residual 4% misses vs primitives (7/39): caps/brush shapes whose cylinder is not their truth (s02 cap 0.13 is correct: 3 cm³ vs 3 g plastic), s13 proxy hull with 66% size error (1.59×), and shape/size errors (0.75-1.25×).

Density dominates once volume is fixed (mass = table-density × volume, solid, no hollow correction):

| Objects | Volume | Median mass ratio | ≤30% |
| --- | --- | --- | --- |
| Inventory hollow (27) | mesh / sealed-4% | 0.65 / 3.76 | 7 / 2 |
| Inventory solid (29) | mesh / sealed-4% | 0.39 / 2.43 | 4 / 2 |

The mesh row scores better only by accident (0.1× wall volume cancels excess density) and fails solids. True-vs-nominal densities (model class): `cardboard` hollow 218 vs 929 (13 objs); `plastic_hard` solid 489 (95-1080) vs 1100 (13); `plastic_hard` hollow 171 vs 1100 (2); `foam` hollow 223 vs 60 (3); `metal_steel` solid 182 vs 7850 (5). Gaps: table holds material density while objects are mostly air (boxes, infilled prints, cans) — factors of 2-40; s11 prints 2.3-5.8× heavy (listed solid but infilled, not solid resin); coffee can ~43× heavy (model says steel; actually cardboard + metal bottom, ~80% coffee-filled); paint can 7.7× / hole punch 6.7× (sheet metal over air, listed solid).

P4.5 (`stages/mass.py`, model hollow answer): body is the 4%-sealed fill (no hull fallback — it fills cups); inertia integrates the same voxels. Open vessels (body < 0.5 hull) keep cavity, f = 1. Closed hollow: V_eff = f·V_sealed with global f ∈ {0.024, 0.22, 0.61} (10th/50th/90th percentile need of 31 such objects); solid kept f = 1.

| Objects | Median ratio | ≤30% | ≤30% uncorrected | Truth in low-high |
| --- | --- | --- | --- | --- |
| Closed, model hollow (31) | 1.00 | 11 | 1 | 26 |
| Open, model hollow (4) | 1.25 | 2 | 2 | 2 |
| Model solid (21) | 2.40 | 1 | 1 | 3 |
| All (56) | 1.25 | 14 | 4 | 31 |

14/56 = 25% vs the 70% exit criterion; the closed-row median 1.00 is by construction (in-sample f).

P4.6 fits f per class (≥3 distinct objects) else per-answer global, including solids; open vessels keep f = 1. 56 rows are 35 distinct objects, so scores hold out all runs of each object:

| Fill | ≤30% of 56 | Median ratio |
| --- | --- | --- |
| None | 4 (7%) | 2.85 |
| Global per answer, held out | 17 (30%) | 1.11 |
| Per class, held out | 20 (36%) | 0.97 |
| Per class, in-sample | 26 (46%) | 0.99 |

Honest figure is 36% vs 70%: density + fill alone do not reach it. Truth in range rises 31 → 42/56 in-sample. Fitted (low/nominal/high): `cardboard` hollow (4 objs) 0.21/0.248/0.302; `metal_steel` hollow (4) 0.028/0.073/0.122; `plastic_hard` hollow (5) 0.415/0.601/0.947; `plastic_hard` solid (12) 0.18/0.357/0.535; global hollow 0.039/0.278/0.681; global solid 0.317/0.441/1.0. In-sample misses (30/56): `plastic_hard` spans solid parts, thin housings, sparse prints (3× range wider than tolerance, 9 misses); coffee can 3× heavy all 4 runs (wrong steel class + invisible filling, pulls the steel fit); model-called-`foam` prints/cap 10-30× light (need >1 unfittable, f ∈ (0,1]); toy basketball 7.5× heavy (single object → global value; inflated ball needs ~0.04); 7 runs carry >20% size error. Real levers are separating sparse from dense objects (class list or model answer) and shape-stage size errors — both outside P4.6.

Limits: f dilutes evenly, understating thin-shell inertia/mass (up to 5/3 for thin spherical shells; no benchmark inertia truth, only closed-form checks). Low-high carry density+f ranges, not scale error (3× in mass); range holds truth on 42/56 in-sample (16/21 solids). f is dev-fitted, test-unrun. Sealed-body COM sits median 1.7 mm from mesh centroid (object-frame origin); P4.8 moves the origin there. The 0.5-hull test also catches concave solids (e.g. chocolate santa gets no correction on a hollow answer). Volume-test biases: +3-6% voxelfill bias at 128 voxels/side (thin boxes worst) unremoved; sealing tuned on 2 mugs/2 caps/1 bottle (wide containers absent); no analytic cup check (boolean backend missing — mugs/bottle vs weights are the check); primitive truth slightly overstates rounded boxes; 17 box-less objects (incl. concave hull-weak cases) unscored in volume; 7/56 carry >20% size error; density ratios use model classes (40/56 agree with inventory).
