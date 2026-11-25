"""PNG image of an exact optical section (``os_optics.optical_view.section``): the same drawing as ``section_svg``,
rasterized with NumPy (no browser): glass filled, clear apertures dark, rims grey, clear-aperture ticks, observation
planes, rays projected onto the plane, labels, a scale bar. One scale on both axes. Deterministic."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..optical_view.section import (
    AXIS_STROKE,
    BAR_TICK_PX,
    BAR_WIDTH_PX,
    CHANNEL_MAX,
    CLEAR_STROKE,
    CLEAR_WIDTH_PX,
    EDGE_WIDTH_PX,
    EMITTER_RGB,
    GLASS_FILL,
    KEY_FROM_BOTTOM_PX,
    LABEL_CHAR_PX,
    LABEL_LIFT_PX,
    NOTE_FROM_BOTTOM_PX,
    PLANE_STROKE,
    PLANE_WIDTH_PX,
    RIM_STROKE,
    RIM_WIDTH_PX,
    SCALE_BAR_FROM_BOTTOM_PX,
    SECOND_NOTE_FROM_BOTTOM_PX,
    SUBSTRATE_FILL,
    SVG_BOTTOM_PX,
    SVG_HEIGHT_PX,
    SVG_MARGIN_PX,
    SVG_TOP_PX,
    SVG_WIDTH_PX,
    TICK_STROKE,
    TICK_WIDTH_PX,
    label_places,
    ray_key,
    ray_paints,
    section_layout,
    section_notes,
    section_polygons,
    section_subject,
)
from .png import png_bytes
from .raster import SUPERSAMPLE, Canvas, _disc, _samples, clip_long_segments, draw_text, text_size

TITLE_RGB = (0.12, 0.14, 0.16)
SUB_RGB = (0.3, 0.33, 0.38)
GLASS_ALPHA = 0.75
SUBSTRATE_ALPHA = 0.8
HEX_BASE = 16
AXIS_WIDTH_PX = 0.8
REFERENCE_WIDTH_PX = 1.0
EMITTER_WIDTH_PX = 3.0
TICK_HALF_PX = 5
TEXT_LEFT_PX = 12
TITLE_TOP_PX = 8
SUBTITLE_TOP_PX = 30
KEY_TOP_PX = 50
NOTE_RISE_PX = 8  # a note's text box sits this far above its baseline row


def _rgb(hex_colour: str) -> tuple[float, float, float]:
    return tuple(int(hex_colour[i:i + 2], HEX_BASE) / CHANNEL_MAX for i in (1, 3, 5))  # type: ignore[return-value]


def _fill_polygon(canvas: Canvas, points: list[tuple[float, float]], rgb, alpha: float) -> None:
    """Even-odd scanline fill of a polygon given in output pixels."""

    if len(points) < 3:
        return
    pts = np.asarray(points, dtype=float) * SUPERSAMPLE
    x0, y0 = pts[:, 0], pts[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    top = max(int(np.floor(pts[:, 1].min())), 0)
    bottom = min(int(np.ceil(pts[:, 1].max())), canvas.h - 1)
    colour = np.asarray(rgb, dtype=float)
    for row in range(top, bottom + 1):
        y = row + 0.5
        crossing = ((y0 <= y) & (y1 > y)) | ((y1 <= y) & (y0 > y))
        if not crossing.any():
            continue
        xs = np.sort(x0[crossing] + (y - y0[crossing]) * (x1[crossing] - x0[crossing]) / (y1[crossing] - y0[crossing]))
        for a, b in zip(xs[0::2], xs[1::2]):
            c0, c1 = max(int(np.ceil(a - 0.5)), 0), min(int(np.floor(b - 0.5)), canvas.w - 1)
            if c0 <= c1:
                canvas.rgb[row, c0:c1 + 1] = canvas.rgb[row, c0:c1 + 1] * (1.0 - alpha) + colour * alpha


def _polyline(canvas: Canvas, points: list[tuple[float, float]], rgb, width_px: float, alpha: float = 1.0, dash: bool = False) -> None:
    if len(points) < 2:
        return
    pts = np.asarray([(x * SUPERSAMPLE, y * SUPERSAMPLE, 0.0) for x, y in points])
    samples = _samples(*clip_long_segments(pts[:-1], pts[1:], canvas.w, canvas.h), dash)
    dy, dx = _disc(width_px * SUPERSAMPLE / 2.0)
    ys = (np.floor(samples[:, 1])[:, None] + dy[None, :]).astype(np.int64).reshape(-1)
    xs = (np.floor(samples[:, 0])[:, None] + dx[None, :]).astype(np.int64).reshape(-1)
    ok = (ys >= 0) & (ys < canvas.h) & (xs >= 0) & (xs < canvas.w)
    keep = np.unique(ys[ok] * canvas.w + xs[ok])
    canvas.blend(keep // canvas.w, keep % canvas.w, rgb, alpha)


def section_png(section: dict[str, Any], *, width: int = SVG_WIDTH_PX, height: int = SVG_HEIGHT_PX, frame: Any = "optics") -> bytes:
    """The section drawing as PNG bytes; ``frame`` (``section_box``) sets what it shows, and the image takes that aspect
    within ``width`` x ``height`` output pixels."""

    layout = section_layout(section, width, height, frame)
    width, height, scale, to_px = layout.width, layout.height, layout.scale, layout.to_px
    canvas = Canvas(width, height)
    canvas.rgb[:] = (1.0, 1.0, 1.0)

    def px(points) -> list[tuple[float, float]]:
        return [to_px(z, t) for t, z in points]

    _polyline(canvas, [to_px(layout.box[0], 0.0), to_px(layout.box[1], 0.0)], _rgb(AXIS_STROKE), AXIS_WIDTH_PX, dash=True)
    for e in section["elements"]:
        for key in ("substrate", "plate"):
            for polygon in section_polygons(e, key):
                _fill_polygon(canvas, px(polygon), _rgb(SUBSTRATE_FILL), SUBSTRATE_ALPHA)
        for m in e["media"]:
            outline = px(m["polygon"])
            _fill_polygon(canvas, outline, _rgb(GLASS_FILL), GLASS_ALPHA)
            _polyline(canvas, [*outline, outline[0]], _rgb(RIM_STROKE), EDGE_WIDTH_PX)
    for r, paint in zip(section["rays"], ray_paints(section), strict=True):
        lit = r["points"][:r["reference_from"] + 1] if "reference_from" in r else r["points"]
        _polyline(canvas, px(lit), paint.rgb, paint.width_px, paint.opacity)
        if "reference_from" in r:  # behind a central obstruction: the beam's axis, not light
            _polyline(canvas, px(r["points"][r["reference_from"]:]), _rgb(AXIS_STROKE), REFERENCE_WIDTH_PX, dash=True)
    for e in section["elements"]:
        for s in e["surfaces"]:
            for seg in s["segments"]:
                clear = seg["zone"] == "CLEAR_APERTURE"
                _polyline(canvas, px(seg["points"]), _rgb(CLEAR_STROKE if clear else RIM_STROKE), CLEAR_WIDTH_PX if clear else RIM_WIDTH_PX)
            for t, z in s.get("clear_aperture_ticks", []):
                x, y = to_px(z, t)
                _polyline(canvas, [(x - TICK_HALF_PX, y), (x + TICK_HALF_PX, y)], _rgb(TICK_STROKE), TICK_WIDTH_PX)
    for emitter in section.get("emitters", ()):
        _polyline(canvas, px([*emitter["outline"], emitter["outline"][0]]), EMITTER_RGB, EMITTER_WIDTH_PX)
        for line in e.get("symbol", ()):
            _polyline(canvas, px(line), _rgb(CLEAR_STROKE), CLEAR_WIDTH_PX)
    for plane in section.get("observation_planes", ()):
        _polyline(canvas, [to_px(plane["z_mm"], plane["t_mm"][0]), to_px(plane["z_mm"], plane["t_mm"][1])], _rgb(PLANE_STROKE), PLANE_WIDTH_PX)
    canvas.rgb[: SVG_TOP_PX * SUPERSAMPLE] = (1.0, 1.0, 1.0)  # rays beyond the frame are cut at the drawing's edge
    canvas.rgb[(height - SVG_BOTTOM_PX) * SUPERSAMPLE:] = (1.0, 1.0, 1.0)
    for text, x, y in label_places(section, layout):
        w, h = text_size(text)
        draw_text(canvas, text, x - w / 2, y - h, TITLE_RGB)
    draw_text(canvas, f'Section {section["plane"]} of {section_subject(section)}: exact profiles, true scale', TEXT_LEFT_PX, TITLE_TOP_PX, TITLE_RGB)
    draw_text(canvas, f'z: {section["axes"]["z"]} (mm)   t: {section["axes"]["t"]} (mm)', TEXT_LEFT_PX, SUBTITLE_TOP_PX, SUB_RGB)
    draw_text(canvas, f'profiles within {section["chord_tolerance_mm"]:g} mm   rays: projected onto the plane, pale farther from it',
              TEXT_LEFT_PX, KEY_TOP_PX, SUB_RGB)
    bar = layout.bar_mm
    x1 = width - SVG_MARGIN_PX
    x0 = x1 - bar * scale
    y = height - SCALE_BAR_FROM_BOTTOM_PX
    _polyline(canvas, [(x0, y), (x1, y)], TITLE_RGB, BAR_WIDTH_PX)
    _polyline(canvas, [(x0, y - BAR_TICK_PX), (x0, y + BAR_TICK_PX)], TITLE_RGB, BAR_WIDTH_PX)
    _polyline(canvas, [(x1, y - BAR_TICK_PX), (x1, y + BAR_TICK_PX)], TITLE_RGB, BAR_WIDTH_PX)
    label = f"{bar:g} mm"
    w, h = text_size(label)
    draw_text(canvas, label, (x0 + x1) / 2 - w / 2, y - LABEL_LIFT_PX - h, TITLE_RGB)
    kx = float(TEXT_LEFT_PX)
    for source_id, rgb in ray_key(section):
        draw_text(canvas, source_id, kx, height - KEY_FROM_BOTTOM_PX - NOTE_RISE_PX, rgb)
        kx += (len(source_id) + 2) * LABEL_CHAR_PX
    for note, from_bottom in zip(section_notes(section, layout), (NOTE_FROM_BOTTOM_PX, SECOND_NOTE_FROM_BOTTOM_PX)):
        draw_text(canvas, note, TEXT_LEFT_PX, height - from_bottom - NOTE_RISE_PX, SUB_RGB)
    return png_bytes(canvas.image())
