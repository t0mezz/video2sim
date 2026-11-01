"""The V logo (``data/logos/v.svg``) as terminal art, for the top of ``video2sim run``.

``_V_PIXELS`` is the three polygons of the SVG rasterized to 20 x 14 pixels with 8 x 8
supersampling and a 50 % coverage threshold (``b`` is the logo blue, ``.`` is transparent).
``_FULL_PIXELS`` is ``data/logos/v2s_logo_med.png`` (V ``b``, 2 ``t``, S ``d``; the white ring
counts as transparent) reduced to 54 x 14 by the majority color of each cell. A terminal cell is
about twice as tall as wide, so one pixel is one cell and the logos keep their proportions.

Every pixel is a space on a background color, never a half-block glyph: a glyph covers only part
of its cell, and terminals with extra line spacing leave the rest of the cell, and the gap to the
next row, in the terminal background, which shows as dark seams along the edges. Standard library
only.
"""

from __future__ import annotations

_V_PIXELS = (
    "bbbbb..........bbbbb",
    ".bbbb.b........bbbb.",
    ".bbbbb.b......bbbbb.",
    "..bbbb.bbb....bbbb..",
    "..bbbbb.bbbb.bbbbb..",
    "...bbbb.bbbb.bbbb...",
    "...bbbbb.bb.bbbbb...",
    "....bbbb....bbbb....",
    "....bbbbb..bbbb.....",
    ".....bbbb.bbbbb.....",
    ".....bbbbbbbbb......",
    "......bbbbbbbb......",
    ".......bbbbbb.......",
    ".......bbbbbb.......",
)
_FULL_PIXELS = (
    ".bbbb...........bbb.....tttttttt......ddddddddddd.ddd.",
    "..bbbb.........bbbb...ttt......ttt....ddddddddddd.dddd",
    "..bbbb.bb.....bbbbb..tt...tttt...tt...ddddddddddd.dddd",
    "...bbbb.bb....bbbb..tt..ttt..ttt..tt..dddd........dddd",
    "...bbbb.bbbb.bbbbbttt...tt....tt..tt..dddd............",
    "....bbbb.bbb.bbbb............tt...tt..ddd.............",
    "....bbbb.bb.bbbbb..........ttt...tt...dddd.dddddddddd.",
    ".....bbbb...bbbb........ttt....ttt....dddd.ddddddddddd",
    ".....bbbb..bbbb.......tt.....ttt.................ddddd",
    "......bbbb.bbbb.....ttt...ttt....................ddddd",
    "......bbbbbbbb.....ttt..ttt...........dddd.......ddddd",
    ".......bbbbbbb.....tt..tttttttttttt...dddd.......ddddd",
    ".......bbbbbb......tt.................dddddddddddddddd",
    "........bbbbb......tttttttttttttttttttdddd.dddddddddd.",
)
_COLORS = {"b": (0x29, 0xAC, 0xF0), "t": (0x3E, 0x89, 0xA8), "d": (0x36, 0x46, 0x56)}
_VARIANTS = {"v": _V_PIXELS, "full": _FULL_PIXELS}
_RESET = "\033[0m"


def _cube(value: int) -> int:
    return 0 if value < 48 else 1 if value < 115 else (value - 35) // 40


def _xterm256(rgb: tuple[int, int, int]) -> int:
    """Nearest color of the 6x6x6 cube, for terminals without 24-bit color."""
    r, g, b = (_cube(c) for c in rgb)
    return 16 + 36 * r + 6 * g + b


def _background(rgb: tuple[int, int, int], truecolor: bool) -> str:
    if truecolor:
        return f"\033[48;2;{rgb[0]};{rgb[1]};{rgb[2]}m"
    return f"\033[48;5;{_xterm256(rgb)}m"


def render(color: bool = True, truecolor: bool = True, variant: str = "v") -> str:
    """The logo as text lines: ``variant`` ``"v"`` is the V alone, ``"full"`` is V2S.

    Without ``color`` it is a one-color silhouette.

    A color code is written only where it changes along a line.
    """
    lines = []
    for row in _VARIANTS[variant]:
        if not color:
            lines.append("".join("█" if pixel != "." else " " for pixel in row).rstrip())
            continue
        out: list[str] = []
        current = "."
        for pixel in row:
            if pixel != current:
                out.append("\033[49m" if pixel == "." else _background(_COLORS[pixel], truecolor))
                current = pixel
            out.append(" ")
        lines.append("".join(out) + _RESET)
    return "\n".join(lines)
