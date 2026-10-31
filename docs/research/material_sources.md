# Material table sources (P4.3)

Values read from the source files by hand, with where they are. The sourced values are in
`configs/material_table.yaml`; what is still missing there is listed per class in `unsourced`
and stays a wide placeholder (D-019). Source PDFs are in `docs/sources/` (git-ignored):
`wood_handbook_fpl_gtr_190.pdf`, `natureworks_ingeo_4043d_tds.pdf`, and Marks' Standard Handbook
for Mechanical Engineers, 10th ed. ("Marks" below; page numbers are PDF pages).

## Verified

| Class | Quantity | Value | Source |
| --- | --- | --- | --- |
| PLA, solid resin | density | 1.24 g/cc = 1240 kg/m3 (ASTM D1505) | NatureWorks, Ingeo 4043D Technical Data Sheet (NW4043D_051815V1), p. 1, film properties table |
| wood | density at 12% moisture | 336 to 784 kg/m3 for specific gravity 0.30 to 0.70; 560 at 0.50 | USDA Wood Handbook, FPL-GTR-190, chapter 4, Table 4-6a (12% row) |
| wood | ovendry density | most species 320 to 720 kg/m3; extremes 160 (balsa) to over 1040 | same, chapter 4, section on density and specific gravity |
| wood | friction | kinetic, smooth dry wood against hard smooth surfaces 0.3 to 0.5; static is generally greater, no static figure given | same, chapter 4, Friction Properties, p. 4-17 |
| plywood | specific gravity | 0.4 to 0.6 | same, chapter 12, Table 12-1 |
| particleboard | specific gravity | 0.6 to 0.8 | same, Table 12-1 |
| MDF | specific gravity | 0.7 to 0.9 (mills 0.71 to 0.95) | same, Table 12-1 and Table 12-6 |
| hardboard | specific gravity | 0.9 to 1.0 | same, Table 12-1 |

### From Marks' (density: Table 6.1.5, PDF pp. 348-350; plastics: Table 6.12.1, specific gravity row, ASTM D792)

| Class | Density, kg/m3 | Note |
| --- | --- | --- |
| steel | 7703 to 7832 (specific gravity 7.70 to 7.83) | cold-drawn, machine, tool steel |
| aluminium | 2643 average (specific gravity 2.55 to 2.80) | cast and hammered |
| glass | 2580 to 2595 average (specific gravity 2.40 to 2.80, common) | common, plate or crown; crystal 2.90 to 3.00 |
| cork | 240 (0.22 to 0.26) | |
| paper | 929 average (0.70 to 1.15) | solid sheet, not a board |
| leather | 945 average (0.86 to 1.02) | |
| rubber | 946 average caoutchouc (0.92 to 0.96); "rubber goods" 1.0 to 2.0 | filled goods vary widely |
| cotton fibre | 1520 (specific gravity 1.52); wool 1.32, nylon 1.14, silk 1.35 | Table 6.8.6; fibre, not fabric |
| ABS | specific gravity 1.02 to 1.08 | |
| acrylic | 1.17 to 1.20 | |
| nylon 6 | 1.12 to 1.14 | |
| polycarbonate | 1.20 | |
| polyethylene | 0.918 to 0.965 | LD to HD |
| polystyrene | 1.04 to 1.05 | |

Friction (Table 3.2.4, PDF pp. 130-131; Table 3.2.6, steel on polymers): all pairs are
metal, wood on wood or polymer against steel. Wood: oak on oak parallel to grain 0.62 static,
0.48 sliding; perpendicular 0.54 and 0.32. Steel on polymers, dry: nylon 0.4, Plexiglas 0.5, PVC
0.5, polystyrene 0.5, LD polyethylene 0.4, HD polyethylene 0.15. Natural cork on glass 0.52, on
pine 0.49. These are against steel or the same material, not against a tabletop.

Restitution: Marks gives only the definition (e = sqrt(h/H) for a ball dropped from H) and no
values.

### Restitution papers

Marinack, Musgrave and Higgs, "Experimental Investigations on the Coefficient of Restitution
of Single Particles", Tribology Transactions 56(4), 2013, 572-580
(`Experimental Investigations on the Coefficient of.pdf`). Spheres of 4.76 mm and 6.35 mm
dropped onto thin plates (mostly S7 tool steel, 6.35 mm thick), impact speed about 2.1 to 3.1
m/s, 6 trials at each of 6 heights, rebound height from high-speed video. Fig. 3, p. 577, read
off the plot by eye (about 0.02 uncertainty), 4.76 mm spheres:

| Sphere on plate | e |
| --- | --- |
| glass on glass | 0.93 to 0.94 |
| glass on S7 steel | 0.88 to 0.93 (stated in the text for 4.76 mm; 0.84 to 0.86 for 6.35 mm) |
| polybutadiene (rubber) on S7 steel | 0.74 to 0.78 |
| brass, chrome steel, S2 tool steel, low-carbon steel on S7 steel | about 0.64 to 0.76 |
| tungsten carbide on S7 steel | about 0.52 to 0.55 |

The paper reports metals falling with impact speed and e falling as the sphere-to-plate-thickness
ratio rises. These are small spheres onto thin steel plates at 2 to 3 m/s: no tabletop, no
plastic, wood or ceramic, and the objects here are much larger and land slower.

Skórski, Obszański and Zawisza, Materials 14(19), 5638, 2021 (`materials-14-05638-v2.pdf`):
epoxy laminates with woven carbon, Kevlar-carbon, glass fabric and wood veneer, 149 g steel ram
dropped 100 to 500 mm onto clamped 120 mm plates. Fitted e at zero drop height 0.84 to 0.88
(measured, Table 3), falling about 0.0006 to 0.0008 per mm of drop height. Composite laminates,
so not a class in our table.

## Not covered yet

* Restitution for plastic (PLA, ABS), wood, cardboard, ceramic, paper, foam and fabric, and
  for any object against a tabletop. Only glass, rubber and metal spheres have values, from
  small spheres on steel plates at 2 to 3 m/s.
* Friction against a tabletop for any class: the sources give wood on wood, and plastics,
  cork or metals against steel.
* Density for printed PLA with infill (the resin density is an upper bound), foam, ceramic,
  aluminium foil as a wrapped object, cardboard (as a board) and fabric (as a finished item).
