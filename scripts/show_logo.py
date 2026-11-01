"""Dev only: print the terminal logo in every rendering mode, to check it by eye.

    uv run python scripts/show_logo.py [--mode MODE ...] [--size]

``--logo`` picks ``v``, ``full`` or ``both`` (default).
Modes: ``truecolor``, ``256``, ``mono`` (one color, no escape codes), ``pixels`` (the raw bitmap)
and ``banner`` (what ``progress.banner`` returns for this terminal). Without ``--mode`` all are
shown, the colored ones on the terminal background and on a white and a black block.
Look for: a solid V with no dark seams between the rows, an intact notch at the top of the V.
Standard library only.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys

from video2sim import logo
from video2sim.progress import banner

MODES = ("truecolor", "256", "mono", "pixels", "banner")
VARIANTS = ("v", "full")
_RESET = "\033[0m"


def _on_background(text: str, rgb: tuple[int, int, int]) -> str:
    """The logo over a solid background, with the same padding on every line."""
    width = max(len(_strip(line)) for line in text.split("\n"))
    code = f"\033[48;2;{rgb[0]};{rgb[1]};{rgb[2]}m"
    lines = [f"{code}  {_pad(line, width)}  {_RESET}" for line in text.split("\n")]
    blank = f"{code}{' ' * (width + 4)}{_RESET}"
    return "\n".join([blank, *lines, blank])


def _strip(line: str) -> str:
    return re.sub(r"\033\[[0-9;]*m", "", line)


def _pad(line: str, width: int) -> str:
    """Pad to ``width`` visible cells; escape sequences take no space."""
    visible, in_escape = 0, False
    for char in line:
        if char == "\033":
            in_escape = True
        elif in_escape:
            in_escape = char != "m"
        else:
            visible += 1
    return line + " " * (width - visible)


def _show(title: str, body: str) -> None:
    print(f"\n== {title} ==")
    print(body)


def _show_variant(variant: str, modes: list[str], everything: bool, size: bool) -> None:
    print(f"\n######## logo: {variant} ########")
    if "pixels" in modes:
        pixels = logo._VARIANTS[variant]
        _show("pixels", "\n".join(row.replace(".", " ").replace("b", "█") for row in pixels))
    if "truecolor" in modes:
        text = logo.render(color=True, truecolor=True, variant=variant)
        _show("truecolor", text)
        if everything:
            _show("truecolor on white", _on_background(text, (255, 255, 255)))
            _show("truecolor on black", _on_background(text, (0, 0, 0)))
    if "256" in modes:
        _show("256 colors", logo.render(color=True, truecolor=False, variant=variant))
    if "mono" in modes:
        _show("mono", logo.render(color=False, variant=variant))
    if size:
        rows = logo.render(color=False, variant=variant).split("\n")
        columns, lines = shutil.get_terminal_size()
        print(
            f"\n{variant}: {max(map(len, rows))} x {len(rows)} cells; terminal {columns} x {lines}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--mode", action="append", choices=MODES, help="default: all")
    parser.add_argument("--logo", choices=(*VARIANTS, "both"), default="both")
    parser.add_argument("--size", action="store_true", help="also print the size in cells")
    args = parser.parse_args()
    modes = args.mode or list(MODES)
    for variant in VARIANTS if args.logo == "both" else (args.logo,):
        _show_variant(variant, modes, args.mode is None, args.size)
    if "banner" in modes:
        text = banner(sys.stdout)
        _show("banner() (used by the CLI)", text if text else "(empty: stdout is not a terminal)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
