"""Render an Optical Layout Package into the Optical Layout Drawing (SVG, PDF and its manifest), on this device.

The sheet of an optics-only purchase: the relative optical layout the customer builds their own mechanics
around. Everything drawn and printed is read from the package (``OpticalLayoutPackageV1``, verified first: schema,
references and ``layout_content_hash``):

* PLAN (design X-Y, viewed along -Z) and an ELEVATION (design X-Z viewed along +Y, or Y-Z viewed along -X when the
  layout extends further along Y than along X), one scale: each element's outline (a catalog product's public outline,
  or the geometry the Project declares for the customer's own optic) projected at its design pose (the silhouette of its
  round or rectangular body; a symbol, "outline unknown", without one), a short arrow along
  its axis direction, the nominal beam paths in their wavelength's display colour (labelled without overlap), the
  observation planes and the datum marker; a small axis cross names the view's axes;
* the element table: id; the OpenSci product name, SKU and BOM line (or "your own <role>" with "modelled as <product>"
  / "custom component <key>"); x, y, z relative to the datum (0.1 mm); the axis direction as azimuth / elevation
  (0.1 deg, reference stated in the notes); the retained DOFs the design requests;
* the spacing table: from -> to and the design distance along each beam path (the package's segment lengths);
* notes: positions relative to the datum in the design frame, no height above a table or breadboard implied (relative
  heights kept); OpenSci supplies the optics only, the mounting is the customer's; customer-supplied components are not
  delivered; the axis-direction convention;
* the title block: "Optical Layout Drawing", the Order Number and the customer once ordered (else the commercial
  resolution), short references and the sheet facts.

The sheet is the smallest landscape A-series sheet (A3 .. A0) on which the views reach a legible scale and the text fits
its columns (then any supported scale). Texts come from the message catalog (``messages``) in ``en`` or ``zh-CN``; the
PDF writes Chinese with the predefined CID font (``assembly_drawing.pdf``). Only ``+ - * /`` and ``sqrt`` touch the
geometry (``angles`` for azimuth / elevation), so the bytes are the same on every runtime, Pyodide included. Every output
token is a package field, a generator literal or a number (a property checked at build time over every layout the tests
generate).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .._vendor.shared_contracts.assembly_preview import json_file_bytes
from .._vendor.shared_contracts.optical_layout import (
    OPTICAL_LAYOUT_SCHEMA_ID,
    OPTICAL_LAYOUT_SCHEMA_VERSION,
    OpticalLayoutElementV1,
    OpticalLayoutPackageV1,
)
from .._vendor.shared_contracts.optical_layout.drawing import (
    LAYOUT_DRAWING_FILES,
    LAYOUT_MANIFEST_PATH,
    LAYOUT_PDF_PATH,
    LAYOUT_SVG_PATH,
    OPTICAL_LAYOUT_GENERATOR_ID,
    OpticalLayoutDrawingManifestV1,
    layout_drawing_manifest_for,
    verify_layout_drawing_outputs,
)
from .._version import DISTRIBUTION_NAME, ENGINE_VERSION
from ..assembly_drawing import layout as L
from ..assembly_drawing.canvas import (
    ACCENT,
    GREY,
    INK,
    PAPER,
    Canvas,
    Rect,
    Style,
    S_EXTERNAL,
    S_FILL_INK,
    S_FRAME,
    S_HAIR,
    S_OUTLINE_THIN,
    S_TABLE,
    S_TABLE_THIN,
)
from ..assembly_drawing.geometry import UNIT_CIRCLE
from ..assembly_drawing.pdf import render_pdf
from ..assembly_drawing.sheet import _overlap, _SegGrid
from ..assembly_drawing.svg import render_svg
from ..assembly_drawing.text import deg, mm, text_width, value_text, wrap
from ..manifest.build import engine_build_id
from ..optical_view.colors import wavelength_display_rgb
from .angles import azimuth_elevation
from .messages import CATALOG

#: Version of the layout drawing's semantics (views, tables, templates). Bump when a drawing of the same package could change.
GENERATOR_VERSION = "1.1.0"  # 1.1.0: title block without references and generator; "Not ordered" before an Order
LOCALES = ("en", "zh-CN")
SVG_METADATA_SCHEMA = "opensci.optical_layout.svg_metadata"
SVG_METADATA_ID = "opensci-optical-layout-metadata"
MAX_PACKAGE_BYTES = 4 * 1024 * 1024

V2 = tuple[float, float]
Vec3 = tuple[float, float, float]

# ---- paper layout (mm)
MARGIN = 14.0
FRAME_INSET = 10.0
BORDER = 5.0
BANNER_BOTTOM = 25.0
VIEWS_TOP = 28.0
COLUMN_W = 150.0
COLUMN_GAP = 4.0
VIEW_GAP = 4.0
VIEW_PAD = 10.0  # paper room around a view's content (labels)
VIEW_LABEL_H = 11.0
TITLE_BLOCK_H = 36.0
TEXT_SIZES = (2.2, 2.0, 1.8, 1.6)
COLUMNS = {"A3": (1,), "A2": (1, 2), "A1": (1, 2, 3), "A0": (1, 2, 3, 4)}
SCALES = (10.0, 5.0, 2.0, 1.0, 0.5, 0.4, 1.0 / 3.0, 0.25, 0.2, 0.1, 0.05, 0.02, 0.01)
MIN_SCALE = 0.2
LINE = 1.32
TITLE_TEXT = 3.5
HEAD_TEXT = 2.5
SMALL = 1.8
TINY = 1.5
SYMBOL_R = 1.8
AXIS_ARROW = 7.0
ARROW_HEAD = 1.4
ARROW_HALF = 0.45
BEAM_WIDTH = 0.45
LABEL_GAP = 1.2
DEFAULT_BEAM_RGB = (0.55, 0.58, 0.62)
CHANNEL = 255.0
ELEMENT_COLUMNS = (0.09, 0.41, 0.09, 0.09, 0.09, 0.12, 0.11)
TB_ROWS = (12.0, 11.0)
TB_BOTTOM_COLUMNS = (0.16, 0.2, 0.18, 0.46)  # units, size, sheet, frame
TB_LOGO_W = 26.0
TB_LOGO_TEXT = 5.0
TB_ORDER_SPLIT = 0.4
TB_ORDER_TEXT = 3.4
TB_PAD = 1.0
TB_LABEL_DROP = 0.3
TB_SUB_RAISE = 1.2
CHANNEL_MAX = 255
ROUNDING = 0.5
ASCENT = 0.75  # of the font size above the baseline
DESCENT = 0.25
LINE_BASELINE = 0.32  # baseline inside a line box, as a fraction of the font size above its bottom
OBSERVATION_TICK = 1.5
DATUM_TIP = 0.6
DATUM_HALF = 1.6
DATUM_BASE = 3.4
MIN_HULL_PAPER = 0.6  # a projected outline smaller than this (paper mm, width + height) is drawn as the symbol
THIN = 0.2
HAIRLINE = 0.25
SYMBOL_WIDTH = 0.3
SYMBOL_DASH = (1.0, 0.6)
SYMBOL_CROSS = 0.7
AXIS_SHOWN = 0.2  # the axis arrow is drawn when its projection is at least this fraction of its length
AXIS_DASH = (0.8, 0.5)
CENTRE_DOT = 0.35
MIN_RUN = 0.05  # paper mm: a shorter beam run is seen end-on
LABEL_ALONG = (0.5, 0.25, 0.75, 0.1, 0.9)
LABEL_STACK = 3
AXIS_CROSS_INSET = 4.0
AXIS_CROSS_RAISE = 3.0
AXIS_CROSS_LEN = 8.0
AXIS_CROSS_TEXT = 9.0
AXIS_CROSS_NUDGE = 0.6
UNDERLINE = 0.9
SUBTITLE_GAP = 1.2
SMALL_FACTOR = 0.9
CELL_PAD = 0.8
CELL_DROP = 0.3
CELL_RISE = 0.2
CELL_WRAP = 1.2
BLOCK_GAP = 2.0
NOTE_INDENT = 5.0
NOTE_GAP = 0.6
SPACING_INDENT = 4.0
FIT_SLACK = 0.4
BANNER_TITLE = 4.6
BANNER_TITLE_DROP = 5.6
BANNER_SUB = 1.8
BANNER_SUB_DROP = 8.4
BANNER_SUB_LINES = 2
BANNER_PAD = 4.0
TITLE_BLOCK_GAP = 3.0
HEADING_ROOM = 4.0  # text lines a heading needs below it in the same column


RUN_SCENE_FILE = "optical_scene.json"  # what makes a directory a run directory
#: Documents people pass by mistake, and what to call them in the refusal.
NOT_A_PACKAGE_SCHEMAS = {"opensci.project": "a Project document", "opensci.optical_simulation_scene": "a run's optical scene"}


class OpticalLayoutErrorCode(StrEnum):
    #: No Optical Layout Package was given: no file at the path, a run directory, a Project or an optical scene.
    PACKAGE_REQUIRED = "OPTICAL_LAYOUT_PACKAGE_REQUIRED"
    #: Not an Optical Layout Package (not JSON, not the contract's shape, references or content hash).
    PACKAGE_INVALID = "PACKAGE_INVALID"
    #: Another schema id or version than this generator renders.
    UNSUPPORTED_VERSION = "UNSUPPORTED_VERSION"
    #: An argument of an unaccepted type or value (e.g. an unknown locale).
    INPUT_INVALID = "INPUT_INVALID"
    #: The layout does not fit any supported sheet.
    LAYOUT_DOES_NOT_FIT = "LAYOUT_DOES_NOT_FIT"


class OpticalLayoutError(ValueError):
    """The renderer refused; ``code`` is an ``OpticalLayoutErrorCode``, ``problems`` lists why."""

    def __init__(self, code: OpticalLayoutErrorCode, problems: Sequence[str]):
        self.code = code
        self.problems = tuple(problems) or (code.value,)
        super().__init__(f"{code.value}: " + "; ".join(self.problems))

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code.value, "problems": [{"code": self.code.value, "message": p} for p in self.problems]}


@dataclass(frozen=True)
class OpticalLayoutDrawing:
    """One rendered drawing: the three output files, the manifest and the package it renders."""

    files: Mapping[str, bytes]
    manifest: OpticalLayoutDrawingManifestV1
    package: OpticalLayoutPackageV1

    @property
    def locale(self) -> str:
        return self.manifest.locale

    def write(self, out_dir: str | Path, extra: Mapping[str, bytes] | None = None) -> Path:
        """Write exactly the three files (and ``extra`` files of the same set, e.g. the package itself) into ``out_dir``
        (created if needed), replacing an earlier layout's files there; other files are left alone
        (``os_optics.output_files``)."""

        from ..output_files import replace_files

        return replace_files(out_dir, {**{name: self.files[name] for name in LAYOUT_DRAWING_FILES}, **(extra or {})})


# ============================================================================== input


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [k for k, _ in pairs]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate JSON object key")
    return dict(pairs)


def package_required(what: str) -> OpticalLayoutError:
    """The refusal when something other than an Optical Layout Package was given: what it is, what produces a package,
    and the 2D drawing a run directory has."""

    return OpticalLayoutError(OpticalLayoutErrorCode.PACKAGE_REQUIRED, (
        f"{what}: the Optical Layout Drawing is rendered from an Optical Layout Package (optical-layout.json)",
        "'os-optics order layout ORDER_ID --out DIR' or 'os-optics resolution layout RESOLUTION_ID --out DIR' writes that "
        "file for an optics-only purchase",
        "for the 2D layout of a run directory use 'os-optics view RUN_DIR --section ALL --snapshot DIR'",
    ))


def load_layout_package(source: Any) -> OpticalLayoutPackageV1:
    """Verify an Optical Layout Package: a verified model, its JSON document (``dict``), its JSON bytes / text, or a
    path to its file. Raises ``OpticalLayoutError`` (``OPTICAL_LAYOUT_PACKAGE_REQUIRED`` / ``PACKAGE_INVALID`` /
    ``UNSUPPORTED_VERSION`` / ``INPUT_INVALID``)."""

    if isinstance(source, OpticalLayoutPackageV1):
        return source
    if isinstance(source, Path):
        if source.is_dir():
            kind = "a run directory" if (source / RUN_SCENE_FILE).is_file() else "a directory"
            raise package_required(f"{source} is {kind}")
        try:
            source = source.read_bytes()
        except FileNotFoundError:
            raise package_required(f"{source} does not exist") from None
        except OSError as exc:
            raise OpticalLayoutError(OpticalLayoutErrorCode.INPUT_INVALID, (f"cannot read the package {source}: {exc.strerror}",)) from None
    if isinstance(source, memoryview):
        source = source.tobytes()
    if isinstance(source, (bytes, bytearray, str)):
        if len(source) > MAX_PACKAGE_BYTES:
            raise OpticalLayoutError(OpticalLayoutErrorCode.PACKAGE_INVALID, (f"the package exceeds {MAX_PACKAGE_BYTES} bytes",))
        try:
            source = json.loads(source, object_pairs_hook=_no_duplicates)
        except RecursionError:
            raise OpticalLayoutError(OpticalLayoutErrorCode.PACKAGE_INVALID, ("the package is nested too deeply",)) from None
        except ValueError as exc:
            raise OpticalLayoutError(OpticalLayoutErrorCode.PACKAGE_INVALID, (f"the package is not JSON: {type(exc).__name__}",)) from None
    if not isinstance(source, dict):
        raise OpticalLayoutError(OpticalLayoutErrorCode.INPUT_INVALID, ("a package is its JSON document, bytes or file path",))
    if source.get("schema") in NOT_A_PACKAGE_SCHEMAS:
        raise package_required(f"this is {NOT_A_PACKAGE_SCHEMAS[source['schema']]}")
    if source.get("schema") != OPTICAL_LAYOUT_SCHEMA_ID or source.get("schema_version") != OPTICAL_LAYOUT_SCHEMA_VERSION:
        raise OpticalLayoutError(OpticalLayoutErrorCode.UNSUPPORTED_VERSION, (
            f"supported: {OPTICAL_LAYOUT_SCHEMA_ID} {OPTICAL_LAYOUT_SCHEMA_VERSION}",))
    try:
        return OpticalLayoutPackageV1.model_validate(source)
    except ValidationError as exc:
        errors = exc.errors(include_input=False, include_url=False)
        where = ".".join(str(p) for p in errors[0].get("loc", ())) if errors else ""
        message = str(errors[0].get("msg", "")) if errors else ""
        detail = "layout_content_hash does not match" if "layout_content_hash" in message else (where or "document")
        raise OpticalLayoutError(OpticalLayoutErrorCode.PACKAGE_INVALID, (f"not a valid Optical Layout Package ({detail})",)) from None
    except (RecursionError, ValueError, TypeError) as exc:
        raise OpticalLayoutError(OpticalLayoutErrorCode.PACKAGE_INVALID, (f"not a valid Optical Layout Package: {type(exc).__name__}",)) from None


# ============================================================================== geometry


def _rotate(q: Sequence[float], v: Vec3) -> Vec3:
    norm = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    w, x, y, z = (c / norm for c in q)
    cx, cy, cz = y * v[2] - z * v[1], z * v[0] - x * v[2], x * v[1] - y * v[0]
    dx, dy, dz = y * cz - z * cy, z * cx - x * cz, x * cy - y * cx
    return (v[0] + 2.0 * (w * cx + dx), v[1] + 2.0 * (w * cy + dy), v[2] + 2.0 * (w * cz + dz))


def outline_points(element: OpticalLayoutElementV1) -> list[Vec3]:
    """The outline's corner / rim points in the design frame (relative to the datum), placed by the element's pose."""

    outline = element.outline
    if outline is None:
        return []
    if outline.shape == "ROUND":
        r = outline.diameter_mm / 2.0  # type: ignore[operator]
        local = [(c * r, s * r, z) for z in (outline.z_min_mm, outline.z_max_mm) for c, s in UNIT_CIRCLE]
    else:
        hw, hh = outline.width_mm / 2.0, outline.height_mm / 2.0  # type: ignore[operator]
        local = [(sx * hw, sy * hh, z) for z in (outline.z_min_mm, outline.z_max_mm) for sx in (-1.0, 1.0) for sy in (-1.0, 1.0)]
    p = element.position_mm
    return [tuple(a + b for a, b in zip(p, _rotate(element.quaternion_wxyz, v), strict=True)) for v in local]  # type: ignore[misc]


def convex_hull(points: Sequence[V2]) -> list[V2]:
    """Andrew's monotone chain (deterministic; collinear points dropped)."""

    pts = sorted(set((round(p[0], 6), round(p[1], 6)) for p in points))
    if len(pts) <= 2:
        return pts

    def cross(o: V2, a: V2, b: V2) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[V2] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0.0:
            lower.pop()
        lower.append(p)
    upper: list[V2] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0.0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


@dataclass(frozen=True)
class ViewSpec:
    key: str  # PLAN / ELEVATION_X / ELEVATION_Y
    u: int  # design axis index shown horizontally (increasing to the right)
    v: int  # design axis index shown vertically (increasing upwards)

    def uv(self, p: Sequence[float]) -> V2:
        return (p[self.u], p[self.v])


PLAN = ViewSpec("PLAN", 0, 1)
ELEVATION_X = ViewSpec("ELEVATION_X", 0, 2)
ELEVATION_Y = ViewSpec("ELEVATION_Y", 1, 2)
AXIS_NAMES = "XYZ"


@dataclass
class _Placed:
    spec: ViewSpec
    scale: float
    origin: V2
    area: Rect

    def P(self, p: Sequence[float]) -> V2:
        u, v = self.spec.uv(p)
        return (self.origin[0] + self.scale * u, self.origin[1] - self.scale * v)


def _scale_label(s: float) -> str:
    return f"{value_text(s)}:1" if s >= 1.0 else f"1:{value_text(1.0 / s)}"


def _hex(rgb: Sequence[float]) -> str:
    return "#" + "".join(f"{min(CHANNEL_MAX, max(0, int(c * CHANNEL + ROUNDING))):02x}" for c in rgb)


# ============================================================================== the sheet


class LayoutSheet:
    def __init__(self, package: OpticalLayoutPackageV1, locale: str) -> None:
        self.pkg = package
        self.locale = locale
        self.elements = {e.instance_id: e for e in package.elements}
        self.cv = Canvas()
        self.rects: list[Rect] = []
        self.segs: list[tuple[V2, V2]] = []
        self.spec = L.SHEETS[0]
        self.columns: list[Rect] = []
        extents = self._points()
        span_x = max(p[0] for p in extents) - min(p[0] for p in extents)
        span_y = max(p[1] for p in extents) - min(p[1] for p in extents)
        self.views = (PLAN, ELEVATION_X if span_x >= span_y else ELEVATION_Y)

    def t(self, key: str, **values: object) -> str:
        return CATALOG.text(self.locale, key, **values)

    # ------------------------------------------------------------------ content

    def _points(self) -> list[Vec3]:
        pts: list[Vec3] = []
        for e in self.pkg.elements:
            pts.append(tuple(e.position_mm))  # type: ignore[arg-type]
            pts.extend(outline_points(e))
        for path in self.pkg.beam_paths:
            pts.extend(tuple(p) for p in path.points_mm)  # type: ignore[misc]
        pts.extend(tuple(o.position_mm) for o in self.pkg.observations)  # type: ignore[misc]
        return pts

    def _range(self, spec: ViewSpec) -> tuple[float, float, float, float]:
        uv = [spec.uv(p) for p in self._points()]
        return min(p[0] for p in uv), min(p[1] for p in uv), max(p[0] for p in uv), max(p[1] for p in uv)

    def _view_region(self, ncols: int) -> Rect:
        right = self.spec.width - MARGIN - ncols * COLUMN_W - (ncols - 1) * COLUMN_GAP
        return (MARGIN, VIEWS_TOP, right - COLUMN_GAP, self.spec.height - MARGIN)

    def _plan_views(self, ncols: int, strict: bool) -> list[_Placed] | None:
        x0, y0, x1, y1 = self._view_region(ncols)
        ranges = [self._range(spec) for spec in self.views]
        widths = [max(r[2] - r[0], 1.0) for r in ranges]
        heights = [max(r[3] - r[1], 1.0) for r in ranges]
        for s in SCALES:
            if strict and s < MIN_SCALE:
                break
            need_h = sum(h * s + 2.0 * VIEW_PAD + VIEW_LABEL_H for h in heights) + VIEW_GAP
            if max(w * s + 2.0 * VIEW_PAD for w in widths) > x1 - x0 or need_h > y1 - y0:
                continue
            spare = (y1 - y0 - need_h) / 2.0
            placed = []
            top = y0
            for spec, r, h in zip(self.views, ranges, heights):
                area_h = h * s + 2.0 * VIEW_PAD + VIEW_LABEL_H + spare
                area = (x0, top, x1, top + area_h)
                cx = (x0 + x1) / 2.0 - (r[0] + r[2]) / 2.0 * s
                cy = top + (area_h - VIEW_LABEL_H) / 2.0 + (r[1] + r[3]) / 2.0 * s
                placed.append(_Placed(spec, s, (cx, cy), area))
                top += area_h + VIEW_GAP
            return placed
        return None

    # ------------------------------------------------------------------ primitives

    def _text(self, p: V2, s: str, size: float, *, anchor: str = "start", bold: bool = False, color: str = INK,
              track: bool = True) -> Rect:
        self.cv.text(p, s, size, anchor=anchor, bold=bold, color=color)
        w = text_width(s, size, bold)
        x0 = {"start": 0.0, "middle": -w / 2.0, "end": -w}[anchor]
        r = (p[0] + x0, p[1] - size * ASCENT, p[0] + x0 + w, p[1] + size * DESCENT)
        if track:
            self.rects.append(r)
        return r

    def _path(self, pts: Sequence[V2], style: Style, *, closed: bool = False) -> None:
        self.cv.path([pts], style, closed)
        seq = list(pts) + ([pts[0]] if closed and len(pts) > 2 else [])
        self.segs.extend(zip(seq, seq[1:]))

    def _arrow(self, a: V2, b: V2, style: Style, fill: Style) -> None:
        length = math.sqrt((b[0] - a[0]) * (b[0] - a[0]) + (b[1] - a[1]) * (b[1] - a[1]))
        if length < ARROW_HEAD:
            return
        ux, uy = (b[0] - a[0]) / length, (b[1] - a[1]) / length
        self._path([a, b], style)
        bx, by = b[0] - ux * ARROW_HEAD, b[1] - uy * ARROW_HEAD
        self.cv.polygon([b, (bx - uy * ARROW_HALF, by + ux * ARROW_HALF), (bx + uy * ARROW_HALF, by - ux * ARROW_HALF)], fill)

    def _free(self, vp: _Placed, rect: Rect, grid: _SegGrid) -> bool:
        area = vp.area
        if rect[0] < area[0] or rect[2] > area[2] or rect[1] < area[1] or rect[3] > area[3] - VIEW_LABEL_H:
            return False
        if any(_overlap(rect, r) for r in self.rects):
            return False
        return not grid.hits_rect(rect)

    def _label(self, vp: _Placed, around: Rect, lines: Sequence[tuple[str, float, bool, str]], attrs: dict[str, str]) -> None:
        """Lines of text beside a paper box, at the first free place (right, left, above, below, then further out)."""

        grid = _SegGrid(self.segs)
        width = max(text_width(t, size, bold) for t, size, bold, _ in lines)
        height = sum(size * LINE for _, size, _, _ in lines)
        cx, cy = (around[0] + around[2]) / 2.0, (around[1] + around[3]) / 2.0
        spots = []
        for k in range(4):
            g = LABEL_GAP + k * 3.0
            spots += [(around[2] + g, cy - height / 2.0), (around[0] - g - width, cy - height / 2.0),
                      (cx - width / 2.0, around[1] - g - height), (cx - width / 2.0, around[3] + g),
                      (around[2] + g, around[1] - g - height), (around[0] - g - width, around[3] + g)]
        chosen = next((s for s in spots if self._free(vp, (s[0], s[1], s[0] + width, s[1] + height), grid)), spots[0])
        self.cv.group(attrs)
        y = chosen[1]
        for text, size, bold, color in lines:
            y += size * LINE
            self._text((chosen[0], y - size * LINE_BASELINE), text, size, bold=bold, color=color)
        self.cv.end()

    # ------------------------------------------------------------------ views

    def _draw_view(self, vp: _Placed) -> None:
        key = vp.spec.key
        self.cv.group({"class": "view", "data-view": key, "data-scale": _scale_label(vp.scale)})
        boxes: dict[str, Rect] = {}
        for e in self.pkg.elements:  # outlines first, so every label avoids every outline
            boxes[e.instance_id] = self._element(vp, e)
        for path in self.pkg.beam_paths:
            self._beam(vp, path)
        for obs in self.pkg.observations:
            c = vp.P(obs.position_mm)
            self.cv.group({"class": "observation", "data-view": key, "data-observation-id": obs.observation_id})
            t = OBSERVATION_TICK
            self._path([(c[0] - t, c[1]), (c[0] + t, c[1])], Style(stroke=GREY, width=HAIRLINE))
            self._path([(c[0], c[1] - t), (c[0], c[1] + t)], Style(stroke=GREY, width=HAIRLINE))
            self.cv.end()
            boxes[obs.observation_id] = (c[0] - t, c[1] - t, c[0] + t, c[1] + t)
        datum = self.elements[self.pkg.datum]
        d = vp.P(datum.position_mm)
        self.cv.group({"class": "datum", "data-view": key, "data-instance-id": datum.instance_id})
        self.cv.polygon([(d[0], d[1] + DATUM_TIP), (d[0] - DATUM_HALF, d[1] + DATUM_BASE), (d[0] + DATUM_HALF, d[1] + DATUM_BASE)],
                        S_FILL_INK)
        self.cv.end()
        self._axis_cross(vp)
        self._beam_labels(vp)
        for e in self.pkg.elements:
            lines = [(e.instance_id, SMALL, True, INK)]
            if e.outline is None:
                lines.append((self.t("outline.unknown"), TINY, False, GREY))
            if e.instance_id == self.pkg.datum:
                lines.append((self.t("datum", id=e.instance_id), TINY, True, INK))
            self._label(vp, boxes[e.instance_id], lines, {"class": "element-label", "data-view": key, "data-instance-id": e.instance_id})
        for obs in self.pkg.observations:
            self._label(vp, boxes[obs.observation_id], [(obs.observation_id, TINY, False, GREY)],
                        {"class": "observation-label", "data-view": key, "data-observation-id": obs.observation_id})
        self._view_title(vp)
        self.cv.end()

    def _element(self, vp: _Placed, e: OpticalLayoutElementV1) -> Rect:
        c = vp.P(e.position_mm)
        customer = e.supplied_by == "CUSTOMER"
        attrs = {"class": "element", "data-view": vp.spec.key, "data-instance-id": e.instance_id,
                 "data-supplied-by": e.supplied_by, "data-outline": "OUTLINE" if e.outline is not None else "UNKNOWN"}
        if e.opensci_sku:
            attrs["data-sku"] = e.opensci_sku
        self.cv.group(attrs)
        hull = convex_hull([vp.P(p) for p in outline_points(e)])
        style = S_EXTERNAL if customer else S_OUTLINE_THIN
        if len(hull) >= 3 and max(p[0] for p in hull) - min(p[0] for p in hull) + max(p[1] for p in hull) - min(p[1] for p in hull) > MIN_HULL_PAPER:
            self._path(hull, style, closed=True)
            box = (min(p[0] for p in hull), min(p[1] for p in hull), max(p[0] for p in hull), max(p[1] for p in hull))
        else:  # no outline (or one too small to draw): the element symbol
            r = SYMBOL_R
            k = r * SYMBOL_CROSS
            self.cv.circle(c, r, Style(stroke=GREY if customer else INK, width=SYMBOL_WIDTH, dash=SYMBOL_DASH if customer else None))
            self._path([(c[0] - k, c[1] - k), (c[0] + k, c[1] + k)], Style(stroke=GREY, width=THIN))
            self._path([(c[0] - k, c[1] + k), (c[0] + k, c[1] - k)], Style(stroke=GREY, width=THIN))
            box = (c[0] - r, c[1] - r, c[0] + r, c[1] + r)
        a = e.axis_direction
        du, dv = vp.spec.uv(a)
        projected = math.sqrt(du * du + dv * dv)
        if projected > AXIS_SHOWN:  # the axis direction, where the view shows it
            tip = (c[0] + du * AXIS_ARROW, c[1] - dv * AXIS_ARROW)
            self._arrow(c, tip, Style(stroke=GREY, width=THIN, dash=AXIS_DASH, cap="butt"), Style(stroke=None, fill=GREY))
        self.cv.circle(c, CENTRE_DOT, S_FILL_INK)
        self.cv.end()
        return box

    def _beam(self, vp: _Placed, path) -> None:  # noqa: ANN001 - OpticalLayoutBeamPathV1
        color = _hex(wavelength_display_rgb(path.wavelength_nm) if path.wavelength_nm else DEFAULT_BEAM_RGB)
        pts = [vp.P(p) for p in path.points_mm]
        self.cv.group({"class": "beam-path", "data-view": vp.spec.key, "data-path-key": path.path_key,
                       **({"data-wavelength-nm": value_text(path.wavelength_nm)} if path.wavelength_nm else {})})
        runs = [(a, b) for a, b in zip(pts, pts[1:]) if abs(b[0] - a[0]) + abs(b[1] - a[1]) > MIN_RUN]
        for a, b in runs:
            self.cv.path([[a, b]], Style(stroke=color, width=BEAM_WIDTH, cap="butt"))
            self.segs.append((a, b))
        if runs:
            a, b = runs[0]
            mid = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
            length = math.sqrt((b[0] - a[0]) * (b[0] - a[0]) + (b[1] - a[1]) * (b[1] - a[1]))
            ux, uy = (b[0] - a[0]) / length, (b[1] - a[1]) / length
            tip = (mid[0] + ux * ARROW_HEAD / 2.0, mid[1] + uy * ARROW_HEAD / 2.0)
            bx, by = tip[0] - ux * ARROW_HEAD, tip[1] - uy * ARROW_HEAD
            self.cv.polygon([tip, (bx - uy * ARROW_HALF, by + ux * ARROW_HALF), (bx + uy * ARROW_HALF, by - ux * ARROW_HALF)],
                            Style(stroke=None, fill=color))
        self.cv.end()

    def _beam_labels(self, vp: _Placed) -> None:
        """One label per set of beam paths that coincide in this view (their keys and wavelengths), placed beside the
        path where it overlaps nothing."""

        groups: dict[tuple, list] = {}
        for path in self.pkg.beam_paths:
            pts = tuple((round(p[0], 1), round(p[1], 1)) for p in (vp.P(q) for q in path.points_mm))
            groups.setdefault(pts, []).append(path)
        for pts, paths in groups.items():
            runs = [(a, b) for a, b in zip(pts, pts[1:]) if abs(b[0] - a[0]) + abs(b[1] - a[1]) > MIN_RUN]
            if not runs:
                continue
            nms = []
            for p in paths:
                if p.wavelength_nm and value_text(p.wavelength_nm) not in nms:
                    nms.append(value_text(p.wavelength_nm))
            keys = ", ".join(p.path_key for p in paths)
            text = self.t("beam.label_nm", path=keys, nm=", ".join(nms)) if nms else self.t("beam.label", path=keys)
            grid = _SegGrid(self.segs)
            width = text_width(text, TINY)
            spots = []
            for k in range(LABEL_STACK):
                off = LABEL_GAP + k * TINY * LINE
                for a, b in runs:
                    for f in LABEL_ALONG:
                        p = (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)
                        spots += [(p[0] - width / 2.0, p[1] - off), (p[0] - width / 2.0, p[1] + off + TINY * ASCENT),
                                  (p[0] + off, p[1] + TINY * DESCENT), (p[0] - off - width, p[1] + TINY * DESCENT)]
            chosen = next((s for s in spots if self._free(vp, (s[0], s[1] - TINY * ASCENT, s[0] + width, s[1] + TINY * DESCENT),
                                                          grid)), spots[0])
            color = _hex(wavelength_display_rgb(paths[0].wavelength_nm) if paths[0].wavelength_nm else DEFAULT_BEAM_RGB)
            self.cv.group({"class": "beam-label", "data-view": vp.spec.key, "data-path-keys": " ".join(p.path_key for p in paths)})
            self._text(chosen, text, TINY, color=color if color != "#ffffff" else INK)
            self.cv.end()

    def _axis_cross(self, vp: _Placed) -> None:
        x0, y0, _, y1 = vp.area
        o = (x0 + AXIS_CROSS_INSET, y1 - VIEW_LABEL_H - AXIS_CROSS_RAISE)
        style, fill = Style(stroke=INK, width=HAIRLINE), Style(stroke=None, fill=INK)
        self.cv.group({"class": "axis-cross", "data-view": vp.spec.key})
        self._arrow(o, (o[0] + AXIS_CROSS_LEN, o[1]), style, fill)
        self._arrow(o, (o[0], o[1] - AXIS_CROSS_LEN), style, fill)
        self._text((o[0] + AXIS_CROSS_TEXT, o[1] + AXIS_CROSS_NUDGE), AXIS_NAMES[vp.spec.u], SMALL, bold=True)
        self._text((o[0] - AXIS_CROSS_NUDGE, o[1] - AXIS_CROSS_TEXT), AXIS_NAMES[vp.spec.v], SMALL, bold=True)
        self.cv.end()

    def _view_title(self, vp: _Placed) -> None:
        x = (vp.area[0] + vp.area[2]) / 2.0
        y = vp.area[3] - VIEW_LABEL_H + TITLE_TEXT + 1.0
        title = self.t(f"view.{vp.spec.key}")
        self._text((x, y), title, TITLE_TEXT, anchor="middle", bold=True)
        w = text_width(title, TITLE_TEXT, True)
        self.cv.line((x - w / 2.0, y + UNDERLINE), (x + w / 2.0, y + UNDERLINE), S_OUTLINE_THIN)
        head = self.t("view.scale", scale=_scale_label(vp.scale))
        self._text((x, y + TITLE_TEXT + SUBTITLE_GAP), self.t("view.label", head=head, extra=self.t(f"view_dir.{vp.spec.key}")), SMALL,
                   anchor="middle")

    # ------------------------------------------------------------------ text column

    def _item_text(self, e: OpticalLayoutElementV1) -> str:
        role = CATALOG.enum(self.locale, "role", e.optical_role)
        name = (e.display_name_zh_cn if self.locale == "zh-CN" and e.display_name_zh_cn else e.display_name)
        if e.supplied_by == "OPENSCI":
            return self.t("item.opensci", name=name, sku=e.opensci_sku, line=e.line_key)
        if e.opensci_sku and name:
            what = self.t("item.modelled", role=role, name=name, sku=e.opensci_sku)
        elif e.opensci_sku:
            what = self.t("item.modelled_sku", role=role, sku=e.opensci_sku)
        elif e.custom_component_key:
            what = self.t("item.custom", role=role, component=e.custom_component_key)
        else:
            what = self.t("item.own", role=role)
        return what + self.t("join") + self.t("item.not_delivered")

    def _element_rows(self) -> list[list[str]]:
        rows = []
        for e in self.pkg.elements:
            az, el = azimuth_elevation(tuple(e.axis_direction))  # type: ignore[arg-type]
            dofs = [self.t("dof.item", axis=d.axis, control=self.t(f"control.{d.control}")) for d in e.retained_dofs]
            rows.append([e.instance_id, self._item_text(e), mm(e.position_mm[0]), mm(e.position_mm[1]), mm(e.position_mm[2]),
                         self.t("axis.value", az=deg(az), el=deg(el)), ", ".join(dofs) or self.t("dof.none")])
        return rows

    def _notes(self) -> list[str]:
        notes = [self.t("note.relative", datum=self.pkg.datum), self.t("note.axis"), self.t("note.supply")]
        customer = [e.instance_id for e in self.pkg.elements if e.supplied_by == "CUSTOMER"]
        if customer:
            notes.append(self.t("note.customer", ids=", ".join(customer)))
        if any(e.retained_dofs for e in self.pkg.elements):
            notes.append(self.t("note.dofs"))
        notes.append(self.t("note.outlines"))
        if self.pkg.beam_paths:
            notes.append(self.t("note.spacing"))
        return notes

    def _text_column(self, size: float, columns: list[Rect], last_bottom: float) -> bool:
        """Lay out the element table, the spacing table and the notes through ``columns``; False if they overflow."""

        index = 0
        bottoms = [c[3] for c in columns[:-1]] + [last_bottom]
        y = columns[0][1]

        def fit(height: float) -> bool:
            nonlocal index, y
            if y + height <= bottoms[index]:
                return True
            if index + 1 < len(columns):
                index += 1
                y = columns[index][1]
                return y + height <= bottoms[index]
            return False

        def heading(text: str) -> bool:
            nonlocal y
            lines = wrap(text, COLUMN_W, HEAD_TEXT, True)
            if not fit(len(lines) * HEAD_TEXT * LINE + HEADING_ROOM * size * LINE):
                return False
            for line in lines:
                self._text((columns[index][0], y + HEAD_TEXT), line, HEAD_TEXT, bold=True, track=False)
                y += HEAD_TEXT * LINE
            y += CELL_PAD
            return True

        # element table
        self.cv.group({"class": "element-table"})
        if not heading(self.t("elements.heading")):
            return False
        heads = [self.t(f"elements.head.{k}") for k in ("id", "item", "x", "y", "z", "axis", "dofs")]
        small = size * SMALL_FACTOR

        def edges() -> list[float]:
            x = columns[index][0]
            out = [x]
            for w in ELEMENT_COLUMNS:
                out.append(out[-1] + w * COLUMN_W)
            return out

        def header() -> None:
            nonlocal y
            e = edges()
            self.cv.line((e[0], y), (e[-1], y), S_TABLE)
            for text, x in zip(heads, e):
                self._text((x + CELL_PAD, y + small + CELL_DROP), text, small, bold=True, track=False)
            y += small * LINE + CELL_PAD
            self.cv.line((e[0], y), (e[-1], y), S_TABLE_THIN)

        header()
        for e_obj, row in zip(self.pkg.elements, self._element_rows()):
            column = index
            e = edges()
            cells = [wrap(text, e[k + 1] - e[k] - CELL_WRAP, small, k == 0) for k, text in enumerate(row)]
            height = max(len(c) for c in cells) * small * LINE + CELL_PAD
            if not fit(height):
                return False
            if index != column:
                header()
                e = edges()
            self.cv.group({"class": "element-row", "data-instance-id": e_obj.instance_id, "data-supplied-by": e_obj.supplied_by})
            for k, lines in enumerate(cells):
                for n, line in enumerate(lines):
                    self._text((e[k] + CELL_PAD, y + small + CELL_RISE + n * small * LINE), line, small, bold=k == 0, track=False,
                               color=GREY if e_obj.supplied_by == "CUSTOMER" and k == 1 else INK)
            self.cv.end()
            y += height
            self.cv.line((e[0], y), (e[-1], y), S_TABLE_THIN)
        self.cv.end()
        y += BLOCK_GAP
        # spacing table
        if self.pkg.beam_paths:
            self.cv.group({"class": "spacing-table"})
            if not heading(self.t("spacing.heading")):
                return False
            for path in self.pkg.beam_paths:
                nm = self.t("spacing.path_nm", nm=value_text(path.wavelength_nm)) if path.wavelength_nm else ""
                if not fit(size * LINE * 2.0):
                    return False
                self._text((columns[index][0], y + size), self.t("spacing.path", path=path.path_key, nm=nm), size, bold=True, track=False)
                y += size * LINE
                for k, d in enumerate(path.segment_lengths_mm):
                    text = self.t("spacing.row", a=path.through[k], b=path.through[k + 1], d=mm(d))
                    if not fit(size * LINE):
                        return False
                    self.cv.group({"class": "spacing-row", "data-path-key": path.path_key, "data-from": path.through[k],
                                   "data-to": path.through[k + 1], "data-distance-mm": mm(d)})
                    self._text((columns[index][0] + SPACING_INDENT, y + size), text, size, track=False)
                    self.cv.end()
                    y += size * LINE
            self.cv.end()
            y += BLOCK_GAP
        # notes
        self.cv.group({"class": "notes"})
        if not heading(self.t("notes.heading")):
            return False
        for k, note in enumerate(self._notes(), start=1):
            lines = wrap(note, COLUMN_W - NOTE_INDENT, size)
            if not fit(len(lines) * size * LINE + NOTE_GAP):
                return False
            for n, line in enumerate(lines):
                if n == 0:
                    self._text((columns[index][0], y + size), f"{k}.", size, bold=True, track=False)
                self._text((columns[index][0] + NOTE_INDENT, y + size), line, size, track=False)
                y += size * LINE
            y += NOTE_GAP
        self.cv.end()
        return True

    def _title_block(self, r: Rect) -> None:
        b = self.pkg.binding
        x0, y0, x1, y1 = r
        g = self.cv
        g.group({"class": "title-block"})
        g.rect(r, S_TABLE)

        def cell(cx0: float, cy0: float, cx1: float, cy1: float, label: str, value: str, sizes: Sequence[float],
                 attrs: dict[str, str] | None = None, bold: bool = True) -> None:
            g.rect((cx0, cy0, cx1, cy1), S_TABLE_THIN)
            if attrs:
                g.group(attrs)
            self._text((cx0 + TB_PAD, cy0 + TINY + TB_LABEL_DROP), label, TINY, color=GREY, track=False)
            top = cy0 + TINY + TB_PAD
            for size in sizes:
                lines = wrap(value, cx1 - cx0 - 2.0 * TB_PAD, size, bold)
                if len(lines) * size * LINE <= cy1 - top - FIT_SLACK or size == sizes[-1]:
                    break
            yy = top + size * ASCENT + TB_LABEL_DROP
            for line in lines:
                self._text((cx0 + TB_PAD, yy), line, size, bold=bold, track=False)
                yy += size * LINE
            if attrs:
                g.end()

        ya = y0 + TB_ROWS[0]
        g.rect((x0, y0, x0 + TB_LOGO_W, ya), S_TABLE_THIN)
        self._text((x0 + TB_LOGO_W / 2.0, y0 + TB_ROWS[0] / 2.0 + DATUM_HALF), "OpenSci", TB_LOGO_TEXT, anchor="middle", bold=True,
                   track=False)
        g.rect((x0 + TB_LOGO_W, y0, x1, ya), S_TABLE_THIN)
        tx = x0 + TB_LOGO_W + TB_PAD
        self._text((tx, y0 + TINY + TB_LABEL_DROP), self.t("tb.label.title"), TINY, color=GREY, track=False)
        self._text((tx, y0 + TINY + TB_PAD + HEAD_TEXT + FIT_SLACK), self.t("tb.title"), HEAD_TEXT + ROUNDING, bold=True, track=False)
        sub = self.t("tb.title_sub" if b.purchase_scope == "OPTICS_ONLY" else "tb.title_sub_complete")
        self._text((tx, ya - TB_SUB_RAISE), sub, SMALL, track=False)
        yb = ya + TB_ROWS[1]
        sizes = (TB_ORDER_TEXT, HEAD_TEXT, 2.0, SMALL, TINY)
        if b.order_number is not None:
            split = x0 + (x1 - x0) * TB_ORDER_SPLIT
            cell(x0, ya, split, yb, self.t("tb.label.order"), b.order_number, sizes, {"class": "order-number"})
            cell(split, ya, x1, yb, self.t("tb.label.customer"), b.customer_display_name or "-", sizes, {"class": "customer"})
        else:  # not ordered yet: the package identities are in the SVG metadata and the drawing manifest
            cell(x0, ya, x1, yb, self.t("tb.label.order"), self.t("tb.not_ordered"), sizes[1:], {"class": "order-state"})
        cells = [("units", "mm"), ("size", self.spec.size), ("sheet", "1 / 1"), ("frame", self.t("tb.frame"))]
        x = x0
        for (key, value), w in zip(cells, TB_BOTTOM_COLUMNS):
            cell(x, yb, x + w * (x1 - x0), y1, self.t(f"tb.label.{key}"), value, (2.0, SMALL, TINY), bold=key == "sheet")
            x += w * (x1 - x0)
        g.end()

    def _frame(self) -> None:
        w, h = self.spec.width, self.spec.height
        self.cv.group({"class": "frame"})
        self.cv.rect((BORDER, BORDER, w - BORDER, h - BORDER), S_HAIR)
        self.cv.rect((FRAME_INSET, FRAME_INSET, w - FRAME_INSET, h - FRAME_INSET), S_FRAME)
        self.cv.end()

    def _banner(self, right: float) -> None:
        x0, x1 = MARGIN, right
        self.cv.group({"class": "notice-banner"})
        self.cv.rect((x0, MARGIN, x1, BANNER_BOTTOM), S_TABLE)
        self._text(((x0 + x1) / 2.0, MARGIN + BANNER_TITLE_DROP), self.t("banner.title"), BANNER_TITLE, anchor="middle", bold=True)
        for k, line in enumerate(wrap(self.t("banner.subtitle"), x1 - x0 - BANNER_PAD, BANNER_SUB)[:BANNER_SUB_LINES]):
            self._text(((x0 + x1) / 2.0, MARGIN + BANNER_SUB_DROP + k * BANNER_SUB * LINE), line, BANNER_SUB, anchor="middle")
        self.cv.end()

    # ------------------------------------------------------------------ build

    def _attempt(self, spec: L.SheetSpec, ncols: int, size: float, views: list[_Placed]) -> Canvas | None:
        self.spec = spec
        self.cv = Canvas(spec.width, spec.height)
        self.rects, self.segs = [], []
        x_right = spec.width - MARGIN - ncols * COLUMN_W - (ncols - 1) * COLUMN_GAP
        self.columns = [(x_right + k * (COLUMN_W + COLUMN_GAP), MARGIN, x_right + k * (COLUMN_W + COLUMN_GAP) + COLUMN_W,
                         spec.height - MARGIN) for k in range(ncols)]
        tb = (self.columns[-1][0], spec.height - MARGIN - TITLE_BLOCK_H, self.columns[-1][2], spec.height - MARGIN)
        self._frame()
        self._banner(x_right - COLUMN_GAP)
        for vp in views:
            self._draw_view(vp)
        if not self._text_column(size, self.columns, tb[1] - TITLE_BLOCK_GAP):
            return None
        self._title_block(tb)
        return self.cv

    def build(self) -> Canvas:
        for strict in (True, False):
            for spec in L.SHEETS:
                for ncols in COLUMNS[spec.size]:
                    self.spec = spec
                    views = self._plan_views(ncols, strict)
                    if views is None:
                        continue
                    for size in TEXT_SIZES:
                        canvas = self._attempt(spec, ncols, size, views)
                        if canvas is not None:
                            return canvas
        raise OpticalLayoutError(OpticalLayoutErrorCode.LAYOUT_DOES_NOT_FIT, ("the layout does not fit any supported sheet",))


# ============================================================================== generation


def generate_optical_layout(source: Any, *, locale: str = "en") -> OpticalLayoutDrawing:
    """Render an Optical Layout Package (see ``load_layout_package`` for the accepted sources) in ``locale``."""

    if locale not in LOCALES:
        raise OpticalLayoutError(OpticalLayoutErrorCode.INPUT_INVALID, (f"locale must be one of {', '.join(LOCALES)}",))
    package = load_layout_package(source)
    sheet = LayoutSheet(package, locale)
    canvas = sheet.build()
    generator = {"generator_id": OPTICAL_LAYOUT_GENERATOR_ID, "generator_version": GENERATOR_VERSION,
                 "engine_package": DISTRIBUTION_NAME, "engine_version": ENGINE_VERSION, "engine_build_id": engine_build_id()}
    binding = package.binding
    title = CATALOG.text(locale, "title")
    description = CATALOG.text(locale, "svg.desc", title=title, layout=package.layout_content_hash,
                               resolution=binding.commercial_resolution_id, project=binding.project_hash,
                               description=CATALOG.text(locale, "description"))
    metadata = {"schema": SVG_METADATA_SCHEMA, "generator_id": OPTICAL_LAYOUT_GENERATOR_ID, "generator_version": GENERATOR_VERSION,
                "locale": locale, "layout_content_hash": package.layout_content_hash, "project_hash": binding.project_hash,
                "local_validation_hash": binding.local_validation_hash, "public_catalog_version": binding.public_catalog_version,
                "commercial_resolution_id": binding.commercial_resolution_id, "purchase_scope": binding.purchase_scope,
                "order_number": binding.order_number, "customer_display_name": binding.customer_display_name,
                "datum": package.datum, "sheet": sheet.spec.key, "elements": [e.instance_id for e in package.elements]}
    files: dict[str, bytes] = {
        LAYOUT_SVG_PATH: render_svg(canvas, title, description, metadata, locale=locale, metadata_id=SVG_METADATA_ID),
        LAYOUT_PDF_PATH: render_pdf(canvas, title),
    }
    manifest = layout_drawing_manifest_for(generator=generator, locale=locale, layout_content_hash=package.layout_content_hash,
                                           binding=binding, sheet=sheet.spec.key, files=files)
    files[LAYOUT_MANIFEST_PATH] = json_file_bytes(manifest.payload())
    problems = verify_layout_drawing_outputs(files)
    if problems:  # pragma: no cover - the generator builds exactly what the verifier checks
        raise OpticalLayoutError(OpticalLayoutErrorCode.PACKAGE_INVALID, problems)
    return OpticalLayoutDrawing(files=files, manifest=manifest, package=package)


def verify_optical_layout_drawing(source: str | Path | Mapping[str, bytes]) -> tuple[str, ...]:
    """Every problem of a written output set (a directory or a ``{name: bytes}`` map); empty = valid. Never raises."""

    try:
        if isinstance(source, Mapping):
            files = {str(k): bytes(v) for k, v in source.items()}
        else:
            directory = Path(source)
            files = {p.name: p.read_bytes() for p in sorted(directory.iterdir()) if p.is_file() and p.name in LAYOUT_DRAWING_FILES}
    except (OSError, TypeError) as exc:
        return (f"cannot read the layout drawing outputs: {type(exc).__name__}",)
    return verify_layout_drawing_outputs(files)
