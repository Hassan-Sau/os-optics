"""Named layout constants of the drawing sheets (paper millimetres, y down).

The sheet is laid out for legibility of one resolved assembly: every value below is a paper
size, spacing or area, never a part dimension. Part geometry and every printed number come from the packages.

``SHEETS`` are the landscape ISO A-series sheets a drawing may use, smallest first. The A3 constants below (``FRAME``,
``LEFT``, ``RIGHT``, ``BANNER`` and the four fixed view areas) are the default A3 sheet; ``SheetGeometry`` derives the same
frame, banner, view region and text columns for any sheet and any number of text columns (on A3 with one column it
reproduces the constants exactly), and ``view_areas`` splits the view region of a larger sheet by the content.
"""

from __future__ import annotations

from dataclasses import dataclass

Rect = tuple[float, float, float, float]

BORDER = 5.0
FRAME: Rect = (10.0, 10.0, 410.0, 287.0)
FRAME_ZONES = (8, 6)  # columns 1..8, rows A..F
FRAME_ZONE_TEXT_INSET = 2.5
FRAME_CENTRE_MARK = 5.0

LEFT: Rect = (14.0, 14.0, 276.0, 283.0)
RIGHT: Rect = (280.0, 14.0, 406.0, 283.0)
BANNER: Rect = (14.0, 14.0, 276.0, 25.0)
FRONT_AREA: Rect = (14.0, 27.0, 200.0, 142.0)
SIDE_AREA: Rect = (202.0, 27.0, 276.0, 142.0)
PLAN_AREA: Rect = (14.0, 143.0, 150.0, 283.0)
PICTORIAL_AREA: Rect = (152.0, 143.0, 276.0, 283.0)

STANDARD_SCALES = (1.0, 0.5, 0.4, 1.0 / 3.0, 0.25, 0.2, 0.1, 0.05, 0.02)
#: Legible scales: a sheet is first chosen so that the elevations and the plan reach at least these; only if no sheet
#: allows that, the smallest sheet that fits at any supported scale is used.
MIN_ELEVATION_SCALE = 0.2
MIN_PLAN_SCALE = 0.1
VIEW_LABEL_GAP = 5.5  # below the lowest drawn content
VIEW_LABEL_HEIGHT = 7.0
VIEW_PAD_MM = 6.0  # world margin around the cropped content of an elevation
AREA_INSET = 2.0

# dimensions
DIM_FIRST = 6.0  # first dimension line from the content it measures
DIM_STEP = 8.0  # further dimension lines
DIM_EXT_GAP = 1.0
DIM_EXT_OVER = 1.5
DIM_ARROW = 2.2
DIM_ARROW_HALF = 0.42
DIM_TEXT = 2.5
DIM_TEXT_OFFSET = 0.9
DIM_TEXT_CLEAR = 1.5  # free length kept at each arrow
ORDINATE_MIN_LINES = 4  # a side stacking more dimension lines than this draws its common-datum group as ordinates
ORD_LEADER = 2.0  # the jog of an ordinate extension line beyond its base line
ORD_TEXT_GAP = 0.8
ORD_LABEL_GAP = 1.0  # between neighbouring (rotated) ordinate labels

# symbols
HOLE_SYMBOL_R = 0.9  # breadboard hole symbol radius on paper (holes are symbols: not to scale)
FIXATION_SYMBOL_R = 2.0
FIXATION_TAG_GAP = 0.6
EXTERNAL_SYMBOL = 3.2  # half size of the customer-supplied source symbol
CALLOUT_INSET = 3.0
CALLOUT_PAD = 1.2
ANCHOR_TICK = 2.5
BEAM_ARROW = 3.0
BEAM_ARROW_HALF = 0.7
BEAM_END_ON_R = 1.6
BREAK_ZIG = 1.4
BREAK_OVER = 2.0

# balloons
BALLOON_R = 3.6
BALLOON_TEXT = 2.8
BALLOON_RINGS = (8.0, 14.0, 20.0, 26.0)
BALLOON_ONE_BY_ONE_MAX = 12  # a multi-chain assembly with more parts gets one stacked balloon group per chain
STACK_RINGS = (6.0, 10.0, 14.0, 18.0, 24.0, 30.0)
BALLOON_CANDIDATE_STEP = 1.0
BALLOON_ANCHOR_STEP = 0.6
BALLOON_MIN_LEADER = 3.0
BALLOON_GAP = 1.2
ANCHOR_DOT = 0.5
TAN_6_DEG = 0.10510423526567646  # leaders within 6 deg of horizontal / vertical are avoided

# text blocks
TEXT_TITLE = 3.5
TEXT_HEAD = 2.5
TEXT_BODY = 2.0
TEXT_SMALL = 1.8
TEXT_TINY = 1.6
LINE_FACTOR = 1.32
BLOCK_GAP = 2.2
TABLE_ROW = 3.4
BANNER_TEXT = 4.6
WATERMARK_TEXT = 9.0
TITLE_BLOCK_H = 44.0


# ------------------------------------------------------------------------------------------------ sheets

MARGIN = 14.0  # frame inset + 4 mm: the drawable region of every sheet starts here
FRAME_INSET = 10.0
TEXT_COLUMN_W = 126.0  # one right-hand text column (notes, tables, steps, legend, title block)
TEXT_COLUMN_GAP = 4.0
BANNER_BOTTOM = 25.0
VIEWS_TOP = 27.0
VIEW_GAP = 2.0  # between neighbouring view areas
BALLOON_ROOM = 14.0  # paper room kept around a view's content for balloons (computed view areas only)
PICTORIAL_MIN = (90.0, 70.0)  # smallest pictorial area of a computed layout (w, h)


@dataclass(frozen=True)
class SheetSpec:
    """One landscape ISO A-series sheet: the contract value, the size printed in the title block, paper size, zones."""

    key: str
    size: str
    width: float
    height: float
    zones: tuple[int, int]  # ISO 5457 frame zones (columns 1..n, rows A..)
    text_columns: tuple[int, ...]  # the numbers of text columns tried on this sheet, fewest first


SHEETS = (
    SheetSpec("A3_LANDSCAPE_420x297_MM", "A3", 420.0, 297.0, (8, 6), (1,)),
    SheetSpec("A2_LANDSCAPE_594x420_MM", "A2", 594.0, 420.0, (12, 8), (1, 2)),
    SheetSpec("A1_LANDSCAPE_841x594_MM", "A1", 841.0, 594.0, (16, 12), (1, 2, 3)),
    SheetSpec("A0_LANDSCAPE_1189x841_MM", "A0", 1189.0, 841.0, (24, 16), (1, 2, 3, 4)),
)


@dataclass(frozen=True)
class SheetGeometry:
    """Frame, banner, view region and text columns of one sheet with ``columns`` right-hand text columns."""

    spec: SheetSpec
    columns: int

    @property
    def frame(self) -> Rect:
        return (FRAME_INSET, FRAME_INSET, self.spec.width - FRAME_INSET, self.spec.height - FRAME_INSET)

    @property
    def right(self) -> Rect:
        x0 = self.spec.width - MARGIN - self.columns * TEXT_COLUMN_W - (self.columns - 1) * TEXT_COLUMN_GAP
        return (x0, MARGIN, self.spec.width - MARGIN, self.spec.height - MARGIN)

    @property
    def left(self) -> Rect:
        return (MARGIN, MARGIN, self.right[0] - TEXT_COLUMN_GAP, self.spec.height - MARGIN)

    @property
    def banner(self) -> Rect:
        return (MARGIN, MARGIN, self.left[2], BANNER_BOTTOM)

    @property
    def view_region(self) -> Rect:
        return (MARGIN, VIEWS_TOP, self.left[2], self.spec.height - MARGIN)

    def text_columns(self) -> list[Rect]:
        x0, y0, _, y1 = self.right
        return [(x0 + k * (TEXT_COLUMN_W + TEXT_COLUMN_GAP), y0, x0 + k * (TEXT_COLUMN_W + TEXT_COLUMN_GAP) + TEXT_COLUMN_W, y1)
                for k in range(self.columns)]

    @property
    def is_v012_a3(self) -> bool:
        """The default A3 sheet with its fixed view areas."""

        return self.spec.key == SHEETS[0].key and self.columns == 1


def sheet_geometries() -> list[SheetGeometry]:
    """Every (sheet, text columns) arrangement, in the order they are tried: smallest sheet, then fewest columns."""

    return [SheetGeometry(spec, n) for spec in SHEETS for n in spec.text_columns]


@dataclass(frozen=True)
class Columns:
    """Relative column widths of a table (fractions summing to one)."""

    widths: tuple[float, ...]

    def edges(self, x0: float, x1: float) -> list[float]:
        out = [x0]
        for w in self.widths:
            out.append(out[-1] + w * (x1 - x0))
        return out


LEGEND_COLUMNS = Columns((0.12, 0.4, 0.48))  # line ref, OpenSci SKU, display name (wrapped)
DIMENSION_COLUMNS = Columns((0.09, 0.36, 0.1, 0.17, 0.28))  # ref, kind / subject, view, value, basis
