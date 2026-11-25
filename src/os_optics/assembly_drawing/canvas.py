"""Backend-independent drawing canvas: recorded operations in paper millimetres (y down), rendered by ``svg`` or ``pdf``.

Operations: ``group`` / ``end`` (a named SVG group with ``data-*`` attributes; the PDF backend ignores it), ``path``
(sub-paths of points, open or closed, one ``Style``), ``circle`` and ``text`` (start / middle / end anchor, regular or
bold, rotated by one of the fixed ``ROTATIONS`` only, so neither backend needs a libm call).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

V2 = tuple[float, float]
Rect = tuple[float, float, float, float]  # x0, y0, x1, y1 on paper (mm, y down)

PAPER_W = 420.0  # the A3 landscape default sheet (layout.SHEETS lists every sheet)
PAPER_H = 297.0

INK = "#000000"
GREY = "#555555"
LIGHT_GREY = "#8a8a8a"
ACCENT = "#c8102e"  # the design beam axis
PREVIEW = "#b45309"  # nominal set-up value markers
DIAGNOSTIC = "#b91c1c"
PAPER = "#ffffff"
BOARD_TINT = "#efefef"
PANEL_TINT = "#f3f3f3"
WATERMARK = "#f2b8b8"

#: Text rotations the backends support (degrees -> (cos, sin), exact or correctly rounded: no libm).
SQRT3_HALF = 0.8660254037844386
ROTATIONS: dict[float, tuple[float, float]] = {0.0: (1.0, 0.0), -90.0: (0.0, -1.0), -30.0: (SQRT3_HALF, -0.5)}


@dataclass(frozen=True)
class Style:
    stroke: str | None = INK
    width: float = 0.25
    dash: tuple[float, ...] | None = None
    fill: str | None = None
    evenodd: bool = False
    cap: str = "round"


S_OUTLINE = Style(width=0.5)
S_OUTLINE_THIN = Style(width=0.35)
S_FEATURE = Style(width=0.25)
S_HIDDEN = Style(width=0.18, dash=(1.6, 0.8), cap="butt")
S_CENTER = Style(width=0.18, dash=(6.0, 1.0, 0.8, 1.0), cap="butt")
S_THIN = Style(width=0.25)
S_HAIR = Style(width=0.18)
S_DIM = Style(width=0.18)
S_DIM_PREVIEW = Style(stroke=PREVIEW, width=0.25, dash=(2.0, 0.8), cap="butt")
S_BEAM = Style(stroke=ACCENT, width=0.35, dash=(8.0, 1.2, 1.2, 1.2), cap="butt")
#: A stretch of a beam axis that only weak light reaches (a SECONDARY stretch): thinner, short dashes.
S_BEAM_SECONDARY = Style(stroke=ACCENT, width=0.18, dash=(1.2, 1.2), cap="butt")
S_FRAME = Style(width=0.7, cap="square")
S_TABLE = Style(width=0.35, cap="square")
S_TABLE_THIN = Style(width=0.18, cap="square")
S_FILL_INK = Style(stroke=None, fill=INK)
S_FILL_ACCENT = Style(stroke=None, fill=ACCENT)
S_FILL_PREVIEW = Style(stroke=None, fill=PREVIEW)
S_FILL_BOARD = Style(stroke=None, fill=BOARD_TINT)
S_FILL_PAPER = Style(stroke=None, fill=PAPER)
S_EXTERNAL = Style(stroke=GREY, width=0.3, dash=(1.2, 0.8), cap="butt")
S_DIAGNOSTIC_BOX = Style(stroke=DIAGNOSTIC, width=0.7, fill=PAPER)


class Canvas:
    """Recorded drawing operations in paper millimetres (y down) on one sheet of ``width`` x ``height`` mm."""

    def __init__(self, width: float = PAPER_W, height: float = PAPER_H) -> None:
        self.width = width
        self.height = height
        self.ops: list[tuple[Any, ...]] = []

    def group(self, attrs: dict[str, str]) -> None:
        self.ops.append(("g", tuple(attrs.items())))

    def end(self) -> None:
        self.ops.append(("/g",))

    def path(self, subpaths: Sequence[Sequence[V2]], style: Style, closed: bool = False) -> None:
        subs = tuple((tuple(pts), closed) for pts in subpaths if len(pts) >= 2)
        if subs:
            self.ops.append(("path", subs, style))

    def line(self, a: V2, b: V2, style: Style) -> None:
        self.path([[a, b]], style)

    def polygon(self, pts: Sequence[V2], style: Style) -> None:
        self.path([pts], style, closed=True)

    def rect(self, r: Rect, style: Style) -> None:
        x0, y0, x1, y1 = r
        self.polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], style)

    def circle(self, c: V2, r: float, style: Style) -> None:
        self.ops.append(("circle", c, r, style))

    def text(self, p: V2, s: str, size: float, *, anchor: str = "start", bold: bool = False, rotate: float = 0.0,
             color: str = INK) -> None:
        if rotate not in ROTATIONS:
            raise ValueError(f"unsupported text rotation {rotate!r}")
        self.ops.append(("text", p, s, size, anchor, bold, rotate, color))

    def texts(self) -> list[str]:
        """Every string the sheet prints (for the output self-scan and tests)."""

        return [op[2] for op in self.ops if op[0] == "text"]

    def attribute_values(self) -> list[str]:
        return [value for op in self.ops if op[0] == "g" for _, value in op[1]]
