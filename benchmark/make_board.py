"""Generate the printable benchmark board as a PDF (A1, A2 or A3, landscape).

    python benchmark/make_board.py --size A2 --out board_a2.pdf
    python benchmark/make_board.py --size A2 --clean --out board_a2_clean.pdf

The grid board is the production scale marker (six ArUco tags, DICT_4X4_50, ids 0 to 5) plus
a 50 mm grid, large stations with heading dials and small stations for small objects. The
pipeline reads only the tags. The grid, stations and dials exist so that object positions and
yaw are known when a benchmark scene is captured. `--clean` prints the tags only, for the
clean-board scenes.

Board frame, in millimetres: origin at the bottom-left corner of the grid area, +X to the
right along the long edge, +Y up the short edge, +Z out of the sheet. Heading is measured
counter-clockwise from +X. Tags: 0 bottom-left, 1 bottom-right, 2 top-left, 3 top-right,
4 middle of the left edge, 5 middle of the right edge.
Large stations (capital letter, number) sit on the 100 mm lattice: "B2" is x = 200, y = 200.
Small stations (lower case) sit in the middle of each 100 mm cell: "a1" is x = 50, y = 50 and
"b2" is x = 150, y = 150.

Print at 100 % ("actual size", no fit to page), let the sheet settle, then measure the tag edge
and the distances between tags with calipers. Put the measured values in the marker config.

Standard library only. The tag bit patterns were generated once with
cv2.aruco.generateImageMarker(DICT_4X4_50, id, 6); '1' is a black cell, row 0 is the top.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

PT_PER_MM = 72.0 / 25.4

PAGES_MM = {"A1": (841.0, 594.0), "A2": (594.0, 420.0), "A3": (420.0, 297.0)}
DEFAULT_TAG_MM = {"A1": 100.0, "A2": 70.0, "A3": 50.0}

TAG_PATTERNS = {
    0: ["111111", "101001", "110101", "111001", "111011", "111111"],
    1: ["111111", "111111", "100001", "101101", "101011", "111111"],
    2: ["111111", "111001", "111001", "111011", "100101", "111111"],
    3: ["111111", "101101", "101101", "110111", "110011", "111111"],
    4: ["111111", "110101", "110111", "101101", "100011", "111111"],
    5: ["111111", "110001", "101101", "100111", "100101", "111111"],
}

PAGE_MARGIN_MM = 10.0  # printers cannot reach the sheet edge
GRID_MM = 50.0
STATION_MM = 100.0
DIAL_RADIUS_MM = 16.0
SMALL_DIAL_RADIUS_MM = 7.0
FOOTER_HEIGHT_MM = 12.0

GRID_GREY = 0.82
GRID_MAJOR_GREY = 0.68
STATION_GREY = 0.35


def _num(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


class Canvas:
    """Collects PDF drawing operators in millimetres, origin at the page bottom-left."""

    def __init__(self) -> None:
        self.ops: list[str] = []

    def line_width(self, width: float) -> None:
        self.ops.append(f"{_num(width)} w")

    def stroke_grey(self, grey: float) -> None:
        self.ops.append(f"{_num(grey)} G")

    def fill_grey(self, grey: float) -> None:
        self.ops.append(f"{_num(grey)} g")

    def line(self, x0: float, y0: float, x1: float, y1: float) -> None:
        self.ops.append(f"{_num(x0)} {_num(y0)} m {_num(x1)} {_num(y1)} l S")

    def rect_fill(self, x: float, y: float, w: float, h: float) -> None:
        self.ops.append(f"{_num(x)} {_num(y)} {_num(w)} {_num(h)} re f")

    def circle(self, cx: float, cy: float, r: float) -> None:
        k = 0.5522847498 * r
        pts = [
            (cx + r, cy, cx + r, cy + k, cx + k, cy + r, cx, cy + r),
            (cx, cy + r, cx - k, cy + r, cx - r, cy + k, cx - r, cy),
            (cx - r, cy, cx - r, cy - k, cx - k, cy - r, cx, cy - r),
            (cx, cy - r, cx + k, cy - r, cx + r, cy - k, cx + r, cy),
        ]
        path = f"{_num(cx + r)} {_num(cy)} m "
        for seg in pts:
            path += " ".join(_num(v) for v in seg[2:]) + " c "
        self.ops.append(path + "S")

    def text(self, x: float, y: float, size: float, content: str, grey: float = 0.0) -> None:
        safe = content.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        self.ops.append(
            f"BT {_num(grey)} g /F1 {_num(size)} Tf {_num(x)} {_num(y)} Td ({safe}) Tj ET"
        )

    def stream(self) -> bytes:
        scale = _num(PT_PER_MM)
        body = f"{scale} 0 0 {scale} 0 0 cm\n" + "\n".join(self.ops) + "\n"
        return body.encode("latin-1")


def write_pdf(path: Path, width_mm: float, height_mm: float, canvas: Canvas) -> None:
    content = canvas.stream()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {_num(width_mm * PT_PER_MM)} "
            f"{_num(height_mm * PT_PER_MM)}] /Resources << /Font << /F1 5 0 R >> >> "
            f"/Contents 4 0 R >>"
        ).encode(),
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"endstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    ).encode()
    path.write_bytes(bytes(out))


def draw_tag(canvas: Canvas, tag_id: int, x: float, y: float, size: float) -> None:
    """Draw one tag with its lower-left corner at (x, y), side `size`."""
    pattern = TAG_PATTERNS[tag_id]
    cell = size / 6.0
    canvas.fill_grey(0.0)
    for row, bits in enumerate(pattern):
        col = 0
        while col < 6:
            if bits[col] != "1":
                col += 1
                continue
            end = col
            while end < 6 and bits[end] == "1":
                end += 1
            # 0.03 mm overlap hides hairline seams between neighbouring rectangles.
            canvas.rect_fill(
                x + col * cell - 0.015,
                y + (5 - row) * cell - 0.015,
                (end - col) * cell + 0.03,
                cell + 0.03,
            )
            col = end


def station_name(i: int, j: int) -> str:
    return f"{chr(ord('A') + i - 1)}{j}"


def build(size: str, tag_mm: float, clean: bool) -> tuple[Canvas, dict[str, object]]:
    page_w, page_h = PAGES_MM[size]
    quiet = max(tag_mm / 6.0, 5.0)  # white margin around each tag
    keepout = tag_mm + 2 * quiet

    # Grid area: a multiple of GRID_MM, centred inside the page margins.
    grid_w = math.floor((page_w - 2 * PAGE_MARGIN_MM) / GRID_MM) * GRID_MM
    grid_h = math.floor((page_h - 2 * PAGE_MARGIN_MM) / GRID_MM) * GRID_MM
    ox = (page_w - grid_w) / 2.0
    oy = (page_h - grid_h) / 2.0

    # Tag lower-left corners in board coordinates: four corners and the middle of the short edges.
    inset = quiet
    left, right = inset, grid_w - inset - tag_mm
    bottom, top = inset, grid_h - inset - tag_mm
    mid = (grid_h - tag_mm) / 2.0
    tags = {
        0: (left, bottom),
        1: (right, bottom),
        2: (left, top),
        3: (right, top),
        4: (left, mid),
        5: (right, mid),
    }

    canvas = Canvas()

    if not clean:
        canvas.stroke_grey(GRID_GREY)
        canvas.line_width(0.15)
        for k in range(int(grid_w / GRID_MM) + 1):
            if (k * GRID_MM) % STATION_MM:
                canvas.line(ox + k * GRID_MM, oy, ox + k * GRID_MM, oy + grid_h)
        for k in range(int(grid_h / GRID_MM) + 1):
            if (k * GRID_MM) % STATION_MM:
                canvas.line(ox, oy + k * GRID_MM, ox + grid_w, oy + k * GRID_MM)
        canvas.stroke_grey(GRID_MAJOR_GREY)
        canvas.line_width(0.25)
        for k in range(int(grid_w / STATION_MM) + 1):
            canvas.line(ox + k * STATION_MM, oy, ox + k * STATION_MM, oy + grid_h)
        for k in range(int(grid_h / STATION_MM) + 1):
            canvas.line(ox, oy + k * STATION_MM, ox + grid_w, oy + k * STATION_MM)

    # Tags, drawn after the grid so that nothing prints inside the quiet zone.
    canvas.fill_grey(1.0)
    for x, y in tags.values():
        canvas.rect_fill(ox + x - quiet, oy + y - quiet, keepout, keepout)
    for tag_id, (x, y) in tags.items():
        draw_tag(canvas, tag_id, ox + x, oy + y, tag_mm)

    stations: list[dict[str, object]] = []
    footer_top = FOOTER_HEIGHT_MM + quiet
    if not clean:
        pad = DIAL_RADIUS_MM + 5.0
        for i in range(1, int(grid_w / STATION_MM)):
            for j in range(1, int(grid_h / STATION_MM)):
                sx, sy = i * STATION_MM, j * STATION_MM
                blocked = any(
                    tx - quiet - pad < sx < tx + tag_mm + quiet + pad
                    and ty - quiet - pad < sy < ty + tag_mm + quiet + pad
                    for tx, ty in tags.values()
                )
                if blocked or sy < footer_top + pad:
                    continue
                draw_station(canvas, ox + sx, oy + sy, station_name(i, j))
                stations.append({"name": station_name(i, j), "x_mm": sx, "y_mm": sy})

        # Small stations in the middle of each 100 mm cell, 70 mm from the nearest large one.
        pad = SMALL_DIAL_RADIUS_MM + 3.0
        for i in range(int(grid_w / STATION_MM)):
            for j in range(int(grid_h / STATION_MM)):
                sx, sy = i * STATION_MM + GRID_MM, j * STATION_MM + GRID_MM
                blocked = any(
                    tx - quiet - pad < sx < tx + tag_mm + quiet + pad
                    and ty - quiet - pad < sy < ty + tag_mm + quiet + pad
                    for tx, ty in tags.values()
                )
                if blocked or sy < footer_top + pad:
                    continue
                name = f"{chr(ord('a') + i)}{j + 1}"
                draw_small_station(canvas, ox + sx, oy + sy, name)
                stations.append({"name": name, "x_mm": sx, "y_mm": sy, "small": True})

    # Axis arrows at the origin and a 100 mm check bar.
    canvas.stroke_grey(0.0)
    canvas.line_width(0.5)
    canvas.line(ox, oy, ox + 30, oy)
    canvas.line(ox, oy, ox, oy + 30)
    canvas.text(ox + 31, oy - 1.0, 3.0, "X")
    canvas.text(ox - 1.2, oy + 31, 3.0, "Y")
    fx = ox + grid_w / 2.0 - 50.0
    fy = oy + quiet
    canvas.line_width(0.6)
    canvas.line(fx, fy + 5.5, fx + 100, fy + 5.5)
    canvas.line(fx, fy + 4.0, fx, fy + 7.0)
    canvas.line(fx + 100, fy + 4.0, fx + 100, fy + 7.0)
    kind = "clean board (tags only)" if clean else "benchmark grid"
    canvas.text(fx, fy + 8.5, 2.6, "This bar must measure 100 mm. Print at 100 %.")
    canvas.text(
        fx,
        fy + 1.8,
        2.4,
        f"video2sim {kind} | {size} | ArUco DICT_4X4_50 ids 0-5 | tag {_num(tag_mm)} mm (nominal)",
    )
    if not clean:
        canvas.text(
            fx,
            fy - 1.5,
            2.4,
            "B2 = x 200, y 200; small b2 = x 150, y 150 (mm). Heading 0 deg = +X, CCW.",
        )

    layout: dict[str, object] = {
        "size": size,
        "page_mm": [page_w, page_h],
        "grid_area_mm": [grid_w, grid_h],
        "origin": "bottom-left corner of the grid area; +X right (long edge), +Y up (short edge)",
        "tag_dictionary": "DICT_4X4_50",
        "tag_edge_mm_nominal": tag_mm,
        "tags": {
            str(tag_id): {
                "lower_left_mm": [x, y],
                "center_mm": [x + tag_mm / 2.0, y + tag_mm / 2.0],
            }
            for tag_id, (x, y) in tags.items()
        },
        "clean": clean,
        "stations": stations,
        "note": "Nominal values. Measure the printed sheet and use the measured numbers.",
    }
    return canvas, layout


def draw_station(canvas: Canvas, cx: float, cy: float, name: str) -> None:
    canvas.stroke_grey(STATION_GREY)
    canvas.line_width(0.25)
    canvas.line(cx - 5, cy, cx + 5, cy)
    canvas.line(cx, cy - 5, cx, cy + 5)
    canvas.line_width(0.15)
    canvas.circle(cx, cy, DIAL_RADIUS_MM)
    for deg in range(0, 360, 15):
        major = deg % 45 == 0
        r_out = DIAL_RADIUS_MM + (4.0 if major else 2.0)
        a = math.radians(deg)
        canvas.line(
            cx + DIAL_RADIUS_MM * math.cos(a),
            cy + DIAL_RADIUS_MM * math.sin(a),
            cx + r_out * math.cos(a),
            cy + r_out * math.sin(a),
        )
    canvas.text(cx + DIAL_RADIUS_MM + 4.5, cy - 0.8, 2.0, "0", STATION_GREY)
    canvas.text(cx - 1.2, cy + DIAL_RADIUS_MM + 4.8, 2.0, "90", STATION_GREY)
    canvas.text(cx - DIAL_RADIUS_MM - 8.0, cy - 0.8, 2.0, "180", STATION_GREY)
    canvas.text(cx - 2.0, cy - DIAL_RADIUS_MM - 6.5, 2.0, "270", STATION_GREY)
    canvas.text(cx + 3.0, cy + 3.0, 3.0, name, 0.15)


def draw_small_station(canvas: Canvas, cx: float, cy: float, name: str) -> None:
    r = SMALL_DIAL_RADIUS_MM
    canvas.stroke_grey(STATION_GREY)
    canvas.line_width(0.2)
    canvas.line(cx - 3, cy, cx + 3, cy)
    canvas.line(cx, cy - 3, cx, cy + 3)
    canvas.line_width(0.12)
    canvas.circle(cx, cy, r)
    for deg in range(0, 360, 15):
        r_out = r + (2.5 if deg % 90 == 0 else 1.2)
        a = math.radians(deg)
        canvas.line(
            cx + r * math.cos(a),
            cy + r * math.sin(a),
            cx + r_out * math.cos(a),
            cy + r_out * math.sin(a),
        )
    canvas.text(cx + r + 3.0, cy - 0.6, 1.6, "0", STATION_GREY)
    canvas.text(cx + 1.2, cy + 1.2, 1.8, name, 0.15)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--size", choices=sorted(PAGES_MM), default="A2")
    parser.add_argument("--tag-mm", type=float, help="tag edge in mm (default depends on size)")
    parser.add_argument("--clean", action="store_true", help="tags only, no grid or stations")
    parser.add_argument("--out", type=Path, help="output PDF (default board_<size>[_clean].pdf)")
    args = parser.parse_args()

    tag_mm = args.tag_mm or DEFAULT_TAG_MM[args.size]
    out = args.out or Path(f"board_{args.size.lower()}{'_clean' if args.clean else ''}.pdf")
    canvas, layout = build(args.size, tag_mm, args.clean)
    page_w, page_h = PAGES_MM[args.size]
    write_pdf(out, page_w, page_h, canvas)
    out.with_suffix(".json").write_text(json.dumps(layout, indent=2) + "\n")
    print(f"wrote {out} and {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
