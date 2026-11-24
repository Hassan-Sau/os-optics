"""Headless software renderer of a ``ViewerScene``: orthographic z-buffer rasteriser, NumPy only.

Per view: opaque triangles (matte ones flat shaded by a head light, metal lit per pixel by the studio of ``shading.py``; no
shadows), then the beams (SECONDARY stretches first and thin/dashed, PRIMARY stretches over them; stretches hidden behind
opaque parts are drawn faint so the path stays readable), then the see-through optics / planes, per pixel and far to near:
glass by the material model of ``shading.py`` with its normals interpolated, a plane flat. Then part outlines (crease,
boundary and silhouette edges, depth tested), markers, part-id labels and the overlays (title, legend, axes triad, scale
bar). The image is rendered at ``SUPERSAMPLE`` x the output size and box-filtered down. Deterministic: the same scene and
size always give the same pixels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .font import GLYPH_HEIGHT, text_mask
from .outline import face_normals as _face_normals
from .outline import outline_groups
from .model import ViewerScene, ViewMesh, camera_angles, camera_presets, fit_distance, frame_points, scene_points
from .palette import AXIS_RGB, BACKGROUND_RGB, BEAM_STYLES, EDGE_DARKEN, HALO_RGB, TEXT_RGB, class_table
from .shading import BEAM_CASING_ALPHA, BEAM_CASING_EXTRA_PX, BEAM_CASING_RGB, FAINT_EDGE_SHARE, OPTICAL_VIEW, backdrop, beam_needs_casing, glass_edge, luminance, mesh_normals, shade

SUPERSAMPLE = 2
MARGIN_PX = 36
TOP_RESERVED_PX = 50  # title lines
AMBIENT = 0.58
DIFFUSE = 0.42
LIGHT_FROM_UP = 0.55  # the head light sits a little above and left of the eye
LIGHT_FROM_LEFT = 0.3
EDGE_WIDTH_PX = 1.0
EDGE_DEPTH_BIAS = 0.004  # of the scene depth range
SAMPLE_STEP_PX = 0.5
HIDDEN_BEAM_ALPHA = 0.28
DASH_PX = 7.0
GAP_PX = 5.0
DOT_PX = 2.0  # hidden stretches: DOT_PX on, DOT_PX off
HIDDEN_OPTIC_ALPHA = 0.55
NESTED_BEAM_EXTRA_PX = 2.0  # main beams of different colours on one line nest: each earlier colour this much wider
MAX_NESTED_BEAMS = 4
TEXT_SCALE = 2
LABEL_GAP_PX = 6
LABEL_TRIES = 12
LEADER_DOT_PX = 1.5
LABEL_BEAM_COST = 40.0  # per beam pixel a label box would cover
LABEL_PART_COST = 1.0  # per part pixel
LABEL_DISTANCE_COST = 30.0  # per step down the candidate list
MARKER_RADIUS_PX = 7.0
TRIAD_ORIGIN_PX = (70, 70)  # from the bottom-left corner
TRIAD_LENGTH_PX = 44.0
TRIAD_MIN_PX = 6.0
SCALE_BAR_TARGET_PX = 120.0
FRAMED_OVERLAY_PX = 120  # an explicit frame keeps this band free below the fit for the axes triad and the scale bar
FRAMED_MIN_WIDTH_PX = 1040  # the title and legend need this much; a narrower box leaves white beside it
FRAMED_MIN_HEIGHT_PX = 360
MIN_FRAME_SPAN_MM = 1e-3
LONG_SEGMENT_CANVASES = 4.0  # a segment longer than this many canvas sizes is cut to the canvas before it is sampled
SCALE_STEPS = (1.0, 2.0, 5.0)
LEGEND_SWATCH_PX = 12
LEGEND_ROW_PX = 18
LEGEND_PAD_PX = 8
LEGEND_ALPHA = 0.88
LEGEND_ITEM_GAP_PX = 22
NEAR_ZERO = 1e-12
CHANNEL_MAX = 255.0
DECADE = 10.0
SUBTITLE_RGB = (0.3, 0.33, 0.38)
RULE_RGB = (0.8, 0.82, 0.85)
TRIAD_LABEL_REACH = 1.25  # axis letters sit this far out along their arrow
INK_ON_DARK = (0.9, 0.92, 0.95)  # lines and text drawn directly on a dark backdrop (leaders, scale bar)
AXIS_ON_DARK = 0.45  # an axis colour mixed this much with white on a dark backdrop
#: ``background``: the studio backdrop of the view (the default), a transparent one (an RGBA image, the alpha channel the
#: coverage of what is drawn) or one flat colour ``(r, g, b)`` in 0..1.
STUDIO_BACKGROUND = "studio"
TRANSPARENT_BACKGROUND = "transparent"
#: ``size_mode``: with a frame, ``fit`` gives the image the frame's aspect within the requested size; ``exact`` keeps the
#: requested size and centres the fitted frame on it.
SIZE_MODES = ("fit", "exact")
DARK_LUMINANCE = 0.18  # a flat backdrop darker than this takes the light ink
CONTENT_DECIMALS = 2  # the framed content's box in pixels
TARGET_DECIMALS = 6  # recorded camera vectors and points (mm)
VIEW_TITLES = {"iso": "ISO", "path_horizontal": "PATH HORIZONTAL (the main beam runs left to right)",
               "top": "TOP (looking down -Z)", "front": "FRONT (looking along +Y)", "side": "SIDE (looking along -X)",
               "beam_axis": "BEAM AXIS (looking along the first main beam)", "plane_xy": "XY PLANE (X right, Y up)",
               "plane_zx": "ZX PLANE (Z right, X up)", "plane_yz": "YZ PLANE (Y right, Z up)"}


@dataclass
class Camera:
    right: np.ndarray
    up: np.ndarray
    forward: np.ndarray  # into the screen
    target: np.ndarray
    scale: float  # supersampled pixels per mm
    cx: float
    cy: float

    def project(self, points: np.ndarray) -> np.ndarray:
        d = np.asarray(points, dtype=float) - self.target
        return np.stack([self.cx + (d @ self.right) * self.scale, self.cy - (d @ self.up) * self.scale, d @ self.forward], axis=-1)


class Canvas:
    """The supersampled drawing. With a transparent ``background`` the colours are kept premultiplied by the coverage
    ``alpha`` (every write is ``out = src + dst * (1 - a)`` on both), and ``image()`` returns RGBA."""

    def __init__(self, width: int, height: int, view_kind: str | None = None, background=None):
        self.w, self.h = width * SUPERSAMPLE, height * SUPERSAMPLE
        self.alpha: np.ndarray | None = None
        if background == TRANSPARENT_BACKGROUND:
            self.rgb = np.zeros((self.h, self.w, 3), dtype=np.float64)
            self.alpha = np.zeros((self.h, self.w), dtype=np.float64)
        elif background not in (None, STUDIO_BACKGROUND):
            self.rgb = np.empty((self.h, self.w, 3), dtype=np.float64)
            self.rgb[:] = background
        elif view_kind is None:
            self.rgb = np.empty((self.h, self.w, 3), dtype=np.float64)
            self.rgb[:] = BACKGROUND_RGB
        else:
            self.rgb = backdrop(view_kind, self.h, self.w)
        dark = view_kind == OPTICAL_VIEW if background in (None, STUDIO_BACKGROUND, TRANSPARENT_BACKGROUND) else luminance(background) < DARK_LUMINANCE
        self.ink = INK_ON_DARK if dark else TEXT_RGB  # lines over the backdrop
        self.z = np.full((self.h, self.w), np.inf)  # opaque depth
        self.zbeam = np.full((self.h, self.w), np.inf)
        self.ids = np.full((self.h, self.w), -1, dtype=np.int32)  # which opaque part is visible at each pixel

    def blend(self, ys: np.ndarray, xs: np.ndarray, rgb, alpha: float | np.ndarray) -> None:
        a = np.asarray(alpha, dtype=float)[..., None] if np.ndim(alpha) else float(alpha)
        self.rgb[ys, xs] = self.rgb[ys, xs] * (1.0 - a) + np.asarray(rgb, dtype=float) * a
        self.cover(ys, xs, alpha)

    def cover(self, ys, xs, alpha) -> None:
        """Coverage of a write of opacity ``alpha`` at these pixels (a transparent canvas only)."""

        if self.alpha is not None:
            a = np.asarray(alpha, dtype=float)
            self.alpha[ys, xs] = a + self.alpha[ys, xs] * (1.0 - a)

    def image(self) -> np.ndarray:
        """(height, width, 3) uint8; a transparent canvas: (height, width, 4), straight (not premultiplied) colours."""

        s = SUPERSAMPLE
        small = self.rgb.reshape(self.h // s, s, self.w // s, s, 3).mean(axis=(1, 3))
        if self.alpha is None:
            return np.clip(np.round(small * CHANNEL_MAX), 0, CHANNEL_MAX).astype(np.uint8)
        alpha = self.alpha.reshape(self.h // s, s, self.w // s, s).mean(axis=(1, 3))
        straight = np.where(alpha[..., None] > NEAR_ZERO, small / np.maximum(alpha, NEAR_ZERO)[..., None], 0.0)
        out = np.concatenate([straight, alpha[..., None]], axis=-1)
        return np.clip(np.round(out * CHANNEL_MAX), 0, CHANNEL_MAX).astype(np.uint8)


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > NEAR_ZERO else v


def view_name(view) -> str:
    """A view's name: a camera preset's, or an explicit pose's ``name``."""

    return view if isinstance(view, str) else str(view.get("name", "camera"))


def _pose(scene: ViewerScene, view) -> dict:
    """A named camera preset (``model.camera_presets``), or an explicit pose: a dict with ``view_direction`` (from the
    target to the eye), ``up`` and optionally ``target_mm``, ``name``, ``title`` and ``distance`` (mm, or ``"<k>x"``: k
    times the size of what the view frames)."""

    if isinstance(view, str):
        return camera_presets(scene)[view]
    pose = dict(view)
    if "target_mm" not in pose:
        pose["target_mm"] = camera_presets(scene)["iso"]["target_mm"]
    return pose


def _frame_spans(scene: ViewerScene, view, frame) -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    preset = _pose(scene, view)
    direction = _unit(np.asarray(preset["view_direction"], dtype=float))
    up = np.asarray(preset["up"], dtype=float)
    forward = -direction
    right = _unit(np.cross(forward, up))
    up = _unit(np.cross(right, forward))
    target = np.asarray(preset["target_mm"], dtype=float)
    d = frame_points(scene, frame) - target
    x, y = d @ right, d @ up
    span_x, span_y = float(x.max() - x.min()), float(y.max() - y.min())
    if frame is not None and (span_x < MIN_FRAME_SPAN_MM or span_y < MIN_FRAME_SPAN_MM):
        raise ValueError(f"the frame has no extent in the {view_name(view)} view")
    return x, y, max(span_x, NEAR_ZERO), max(span_y, NEAR_ZERO), right, up, forward, target


def make_camera(scene: ViewerScene, view, width: int, height: int, reserve_bottom_px: int = 0, frame=None) -> Camera:
    x, y, span_x, span_y, right, up, forward, target = _frame_spans(scene, view, frame)
    usable_w = (width - 2 * MARGIN_PX) * SUPERSAMPLE
    usable_h = (height - 2 * MARGIN_PX - TOP_RESERVED_PX - reserve_bottom_px) * SUPERSAMPLE
    scale = min(usable_w / span_x, usable_h / span_y)
    mid_x, mid_y = (x.max() + x.min()) / 2.0, (y.max() + y.min()) / 2.0
    cx = width * SUPERSAMPLE / 2.0 - mid_x * scale
    cy = (height + TOP_RESERVED_PX - reserve_bottom_px) * SUPERSAMPLE / 2.0 + mid_y * scale
    return Camera(right=right, up=up, forward=forward, target=target, scale=scale, cx=cx, cy=cy)


def _framed_size(scene: ViewerScene, view, frame, width: int, height: int, reserve_bottom_px: int) -> tuple[int, int]:
    """The image size that follows a frame's aspect within ``width`` x ``height``: the box fills it at one scale. Never
    larger than the requested size (the side the box fills is that side's length, not one rounded pixel more)."""

    _, _, span_x, span_y, _, _, _, _ = _frame_spans(scene, view, frame)
    usable_w, usable_h = width - 2 * MARGIN_PX, height - 2 * MARGIN_PX - TOP_RESERVED_PX - reserve_bottom_px
    scale = min(usable_w / span_x, usable_h / span_y)
    return (min(width, max(min(FRAMED_MIN_WIDTH_PX, width), math.ceil(span_x * scale) + 2 * MARGIN_PX)),
            min(height, max(min(FRAMED_MIN_HEIGHT_PX, height), math.ceil(span_y * scale) + 2 * MARGIN_PX + TOP_RESERVED_PX + reserve_bottom_px)))


# ------------------------------------------------------------------------------------------------------ triangles


def _shade(normals: np.ndarray, cam: Camera) -> np.ndarray:
    light = _unit(-cam.forward + cam.up * LIGHT_FROM_UP - cam.right * LIGHT_FROM_LEFT)
    return AMBIENT + DIFFUSE * np.abs(normals @ light)


def _camera_normals(normals: np.ndarray, cam: Camera) -> np.ndarray:
    return np.stack([normals @ cam.right, normals @ cam.up, -(normals @ cam.forward)], axis=-1)


def _raster_triangle(canvas: Canvas, tri: np.ndarray):
    """Pixel rows / columns, depth and barycentric weights (k, 3) covered by one projected triangle (``tri``: 3 x (x, y, depth))."""

    x0 = max(int(math.floor(tri[:, 0].min())), 0)
    x1 = min(int(math.ceil(tri[:, 0].max())), canvas.w - 1)
    y0 = max(int(math.floor(tri[:, 1].min())), 0)
    y1 = min(int(math.ceil(tri[:, 1].max())), canvas.h - 1)
    if x0 > x1 or y0 > y1:
        return None
    (ax, ay, az), (bx, by, bz), (qx, qy, qz) = tri
    area = (bx - ax) * (qy - ay) - (qx - ax) * (by - ay)
    if abs(area) < NEAR_ZERO:
        return None
    px = np.arange(x0, x1 + 1) + 0.5
    py = (np.arange(y0, y1 + 1) + 0.5)[:, None]
    w0 = ((bx - px) * (qy - py) - (qx - px) * (by - py)) / area
    w1 = ((qx - px) * (ay - py) - (ax - px) * (qy - py)) / area
    w2 = 1.0 - w0 - w1
    inside = (w0 >= 0.0) & (w1 >= 0.0) & (w2 >= 0.0)
    if not inside.any():
        return None
    ys, xs = np.nonzero(inside)
    weights = np.stack([w0[ys, xs], w1[ys, xs], w2[ys, xs]], axis=-1)
    return ys + y0, xs + x0, weights @ tri[:, 2], weights


def draw_opaque(canvas: Canvas, cam: Camera, positions: np.ndarray, triangles: np.ndarray, rgb, part_index: int = -1) -> None:
    screen = cam.project(positions)
    colours = np.asarray(rgb, dtype=float)[None, :] * _shade(_face_normals(positions, triangles), cam)[:, None]
    for k, t in enumerate(triangles):
        hit = _raster_triangle(canvas, screen[t])
        if hit is None:
            continue
        ys, xs, depth, _ = hit
        closer = depth < canvas.z[ys, xs]
        ys, xs = ys[closer], xs[closer]
        canvas.z[ys, xs] = depth[closer]
        canvas.rgb[ys, xs] = colours[k]
        canvas.cover(ys, xs, 1.0)
        canvas.ids[ys, xs] = part_index


def _studio_shade(mesh: ViewMesh, normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return shade(normals, mesh.rgb, alpha=mesh.alpha, metallic=mesh.metallic, roughness=mesh.roughness, transmission=mesh.transmission,
                 ior=mesh.ior, thickness_mm=mesh.thickness_mm, tint=mesh.tint)


def draw_opaque_studio(canvas: Canvas, cam: Camera, mesh: ViewMesh, part_index: int) -> None:
    """An opaque metal surface, its normals interpolated over each triangle."""

    screen = cam.project(mesh.positions)
    normals = _camera_normals(mesh_normals(mesh), cam)
    pixels, shaded = [], []
    for t in mesh.triangles:
        hit = _raster_triangle(canvas, screen[t])
        if hit is None:
            continue
        ys, xs, depth, weights = hit
        closer = depth < canvas.z[ys, xs]
        ys, xs = ys[closer], xs[closer]
        canvas.z[ys, xs] = depth[closer]
        canvas.ids[ys, xs] = part_index
        pixels.append(ys * canvas.w + xs)
        shaded.append(weights[closer] @ normals[t])
    if not pixels:
        return
    keys = np.concatenate(pixels)
    # a later triangle may have covered an earlier one's pixel: its normal, last written, wins
    keys, last = np.unique(keys[::-1], return_index=True)
    ns = np.concatenate(shaded)[::-1][last]
    ns /= np.maximum(np.linalg.norm(ns, axis=1, keepdims=True), NEAR_ZERO)
    src, a = _studio_shade(mesh, ns)
    ys, xs = keys // canvas.w, keys % canvas.w
    canvas.rgb[ys, xs] = src + canvas.rgb[ys, xs] * (1.0 - a)[:, None]
    canvas.cover(ys, xs, a)


def _see_through_fragments(canvas: Canvas, cam: Camera, mesh: ViewMesh) -> tuple[np.ndarray, ...] | None:
    """Every pixel of a see-through mesh in front of what is drawn so far: ``(pixel key, depth, premultiplied colour, coverage)``."""

    screen = cam.project(mesh.positions)
    if mesh.studio_lit:
        corners = _camera_normals(mesh_normals(mesh), cam)
    else:
        flat = np.asarray(mesh.rgb, dtype=float)[None, :] * _shade(_face_normals(mesh.positions, mesh.triangles), cam)[:, None] * mesh.alpha
    keys, depths, parts = [], [], []
    for k, t in enumerate(mesh.triangles):
        hit = _raster_triangle(canvas, screen[t])
        if hit is None:
            continue
        ys, xs, depth, weights = hit
        front = (depth < canvas.z[ys, xs]) & (depth < canvas.zbeam[ys, xs])
        if not front.any():
            continue
        keys.append(ys[front] * canvas.w + xs[front])
        depths.append(depth[front])
        parts.append(weights[front] @ corners[t] if mesh.studio_lit else np.broadcast_to(flat[k], (int(front.sum()), 3)))
    if not keys:
        return None
    if mesh.studio_lit:
        ns = np.concatenate(parts)
        ns /= np.maximum(np.linalg.norm(ns, axis=1, keepdims=True), NEAR_ZERO)
        src, a = _studio_shade(mesh, ns)
    else:
        src = np.concatenate(parts)
        a = np.full(len(src), mesh.alpha)
    return np.concatenate(keys), np.concatenate(depths), src.astype(np.float32), a.astype(np.float32)


def draw_see_through(canvas: Canvas, cam: Camera, meshes: list[ViewMesh]) -> None:
    """Every see-through mesh, layered per pixel, far to near: ``out = src + dst * (1 - a)``."""

    fragments = [f for f in (_see_through_fragments(canvas, cam, m) for m in meshes) if f is not None]
    if not fragments:
        return
    keys, depth, src, a = (np.concatenate(column) for column in zip(*fragments, strict=True))
    order = np.lexsort((-depth, keys))
    keys, src, a = keys[order], src[order], a[order]
    starts = np.concatenate([[True], keys[1:] != keys[:-1]])
    first = np.maximum.accumulate(np.where(starts, np.arange(len(keys)), 0))
    layer = np.arange(len(keys)) - first
    for k in range(int(layer.max()) + 1):
        here = layer == k
        ys, xs = keys[here] // canvas.w, keys[here] % canvas.w
        canvas.rgb[ys, xs] = src[here] + canvas.rgb[ys, xs] * (1.0 - a[here])[:, None]
        canvas.cover(ys, xs, a[here])


# ----------------------------------------------------------------------------------------------------------- lines


def _disc(radius: float) -> tuple[np.ndarray, np.ndarray]:
    r = max(int(math.ceil(radius)), 0)
    dy, dx = np.mgrid[-r:r + 1, -r:r + 1]
    keep = dx * dx + dy * dy <= max(radius, SAMPLE_STEP_PX) ** 2
    return dy[keep], dx[keep]


def clip_long_segments(p0: np.ndarray, p1: np.ndarray, width: float, height: float) -> tuple[np.ndarray, np.ndarray]:
    """Projected segments with those much longer than the canvas cut to it (Liang-Barsky), so that a tight frame around
    part of a long beam samples only what shows; a segment that misses the canvas collapses to its start."""

    d = p1 - p0
    long = np.hypot(d[:, 0], d[:, 1]) > LONG_SEGMENT_CANVASES * (width + height)
    if not long.any():
        return p0, p1
    p0, p1 = p0.copy(), p1.copy()
    for k in np.nonzero(long)[0]:
        lo, hi = 0.0, 1.0
        for axis, size in ((0, width), (1, height)):
            if d[k, axis] == 0.0:
                if not 0.0 <= p0[k, axis] <= size:
                    lo, hi = 1.0, 0.0
                continue
            a, b = (0.0 - p0[k, axis]) / d[k, axis], (size - p0[k, axis]) / d[k, axis]
            lo, hi = max(lo, min(a, b)), min(hi, max(a, b))
        start = p0[k].copy()
        p0[k] = start + d[k] * (lo if lo <= hi else 0.0)
        p1[k] = start + d[k] * (hi if lo <= hi else 0.0)
    return p0, p1


def _samples(p0: np.ndarray, p1: np.ndarray, dash: bool) -> np.ndarray:
    """Points (x, y, depth) every SAMPLE_STEP_PX along projected segments; dashed lines keep the dash parts only."""

    out = []
    for a, b in zip(p0, p1, strict=True):
        length = float(np.hypot(b[0] - a[0], b[1] - a[1]))
        n = max(int(math.ceil(length / SAMPLE_STEP_PX)), 1)
        t = np.linspace(0.0, 1.0, n + 1)
        pts = a[None, :] + (b - a)[None, :] * t[:, None]
        if dash:
            pts = pts[(t * length) % ((DASH_PX + GAP_PX) * SUPERSAMPLE) < DASH_PX * SUPERSAMPLE]
        out.append(pts)
    return np.concatenate(out) if out else np.zeros((0, 3))


def draw_lines(canvas: Canvas, cam: Camera, p0: np.ndarray, p1: np.ndarray, rgb, *, width_px: float, alpha: float = 1.0,
               dash: bool = False, depth_bias: float = 0.0, hidden_alpha: float = 0.0, beam: bool = False, depth_test: bool = True) -> None:
    """Depth-tested lines. Stretches hidden behind opaque parts are drawn dotted at ``hidden_alpha`` (0: not drawn)."""

    if len(p0) == 0:
        return
    pts = _samples(*clip_long_segments(cam.project(p0), cam.project(p1), canvas.w, canvas.h), dash)
    if len(pts) == 0:
        return
    along = np.arange(len(pts)) * SAMPLE_STEP_PX
    dotted = (along % (DOT_PX * 2.0 * SUPERSAMPLE)) < DOT_PX * SUPERSAMPLE
    dy, dx = _disc(width_px * SUPERSAMPLE / 2.0)
    ys = (np.floor(pts[:, 1])[:, None] + dy[None, :]).astype(np.int64).reshape(-1)
    xs = (np.floor(pts[:, 0])[:, None] + dx[None, :]).astype(np.int64).reshape(-1)
    depth = np.repeat(pts[:, 2], len(dy))
    dotted = np.repeat(dotted, len(dy))
    ok = (ys >= 0) & (ys < canvas.h) & (xs >= 0) & (xs < canvas.w)
    ys, xs, depth, dotted = ys[ok], xs[ok], depth[ok], dotted[ok]
    if not depth_test:
        canvas.blend(ys, xs, rgb, alpha)
        return
    limit = canvas.z[ys, xs] if beam else np.minimum(canvas.z[ys, xs], canvas.zbeam[ys, xs])
    visible = depth <= limit + depth_bias
    if hidden_alpha > 0.0:
        hidden = ~visible & dotted
        canvas.blend(ys[hidden], xs[hidden], rgb, hidden_alpha)
    canvas.blend(ys[visible], xs[visible], rgb, alpha)
    if beam:
        canvas.zbeam[ys[visible], xs[visible]] = np.minimum(canvas.zbeam[ys[visible], xs[visible]], depth[visible])


# ------------------------------------------------------------------------------------------------------------- text


def draw_text(canvas: Canvas, text: str, x: float, y: float, rgb=TEXT_RGB, *, halo: bool = True, scale: int = TEXT_SCALE) -> tuple[int, int, int, int]:
    """Draw one line with its top-left at output pixel (x, y); returns its output-pixel box (x0, y0, x1, y1)."""

    mask = text_mask(text, scale * SUPERSAMPLE)
    x0, y0 = int(round(x * SUPERSAMPLE)), int(round(y * SUPERSAMPLE))
    h, w = mask.shape
    if halo:
        pad = SUPERSAMPLE * 2
        grown = np.zeros((h + 2 * pad, w + 2 * pad), dtype=bool)
        for oy in range(-pad, pad + 1, SUPERSAMPLE):
            for ox in range(-pad, pad + 1, SUPERSAMPLE):
                grown[pad + oy:pad + oy + h, pad + ox:pad + ox + w] |= mask
        _stamp(canvas, grown, x0 - pad, y0 - pad, HALO_RGB, LEGEND_ALPHA)
    _stamp(canvas, mask, x0, y0, rgb, 1.0)
    return int(x), int(y), int(x + w / SUPERSAMPLE), int(y + h / SUPERSAMPLE)


def text_size(text: str, scale: int = TEXT_SCALE) -> tuple[int, int]:
    mask = text_mask(text, scale)
    return mask.shape[1], mask.shape[0]


def _stamp(canvas: Canvas, mask: np.ndarray, x0: int, y0: int, rgb, alpha: float) -> None:
    ys, xs = np.nonzero(mask)
    ys, xs = ys + y0, xs + x0
    ok = (ys >= 0) & (ys < canvas.h) & (xs >= 0) & (xs < canvas.w)
    canvas.blend(ys[ok], xs[ok], rgb, alpha)


def fill_rect(canvas: Canvas, x0: float, y0: float, x1: float, y1: float, rgb, alpha: float = 1.0) -> None:
    a, b = max(int(x0 * SUPERSAMPLE), 0), min(int(x1 * SUPERSAMPLE), canvas.w)
    c, d = max(int(y0 * SUPERSAMPLE), 0), min(int(y1 * SUPERSAMPLE), canvas.h)
    if a < b and c < d:
        canvas.rgb[c:d, a:b] = canvas.rgb[c:d, a:b] * (1.0 - alpha) + np.asarray(rgb, dtype=float) * alpha
        if canvas.alpha is not None:
            canvas.alpha[c:d, a:b] = alpha + canvas.alpha[c:d, a:b] * (1.0 - alpha)


def screen_line(canvas: Canvas, p0, p1, rgb, width_px: float, alpha: float = 1.0) -> None:
    """A 2D line between output-pixel points (overlays)."""

    a = np.array([[p0[0] * SUPERSAMPLE, p0[1] * SUPERSAMPLE, 0.0]])
    b = np.array([[p1[0] * SUPERSAMPLE, p1[1] * SUPERSAMPLE, 0.0]])
    pts = _samples(a, b, False)
    dy, dx = _disc(width_px * SUPERSAMPLE / 2.0)
    ys = (np.floor(pts[:, 1])[:, None] + dy[None, :]).astype(np.int64).reshape(-1)
    xs = (np.floor(pts[:, 0])[:, None] + dx[None, :]).astype(np.int64).reshape(-1)
    ok = (ys >= 0) & (ys < canvas.h) & (xs >= 0) & (xs < canvas.w)
    canvas.blend(ys[ok], xs[ok], rgb, alpha)


# ----------------------------------------------------------------------------------------------------------- render


def _nice_length(mm: float) -> float:
    if mm <= 0.0:
        return 1.0
    power = DECADE ** math.floor(math.log10(mm))
    best = power
    for step in SCALE_STEPS:
        if step * power <= mm:
            best = step * power
    return best


def _fmt(mm: float) -> str:
    return f"{mm:g}"


def parts_silhouette(scene: ViewerScene, view: str, width: int, height: int) -> tuple[np.ndarray, Camera]:
    """Where one named view draws its parts: a (height, width) bool image, true where at least half of a pixel is covered, without
    beams, outlines or overlays, and the view's camera. Anything drawing the same GLB with this camera covers the same pixels."""

    band = _legend_band_height(_legend_layout(_legend_rows(scene), width))
    cam = make_camera(scene, view, width, height, reserve_bottom_px=band)
    canvas = Canvas(width, height)
    covered = np.zeros((canvas.h, canvas.w), dtype=bool)
    for part in scene.parts:
        for mesh in part.meshes:
            screen = cam.project(mesh.positions)
            for t in mesh.triangles:
                hit = _raster_triangle(canvas, screen[t])
                if hit is not None:
                    covered[hit[0], hit[1]] = True
    s = SUPERSAMPLE
    return covered.reshape(height, s, width, s).mean(axis=(1, 3)) >= 0.5, cam


def render_view(scene: ViewerScene, view, width: int, height: int, *, labels: bool = True, frame=None, background=None,
                size_mode: str = "fit") -> tuple[np.ndarray, dict]:
    """Render one view (a camera preset's name, or an explicit pose: ``_pose``); returns the (height, width, 3) uint8 image
    (RGBA with a transparent ``background``) and what was drawn (for the legend and ``scene_summary.json``). ``frame``
    (``model.frame_points``) sets what the view fits: with ``size_mode`` ``fit`` the image then takes that aspect within
    ``width`` x ``height`` (never larger), with ``exact`` it keeps ``width`` x ``height`` and the fit is centred on it; the
    overlays have a band of their own. ``background``: ``STUDIO_BACKGROUND`` (default), ``TRANSPARENT_BACKGROUND`` or an
    ``(r, g, b)`` colour."""

    if size_mode not in SIZE_MODES:
        raise ValueError(f"size mode must be one of {', '.join(SIZE_MODES)}")
    requested = [width, height]
    rows = _legend_rows(scene)
    layout = _legend_layout(rows, width)
    band = _legend_band_height(layout)
    if frame is not None and size_mode == "fit":
        width, height = _framed_size(scene, view, frame, width, height, band + FRAMED_OVERLAY_PX)
        layout = _legend_layout(rows, width)
        band = _legend_band_height(layout)
    cam = make_camera(scene, view, width, height, reserve_bottom_px=band + (FRAMED_OVERLAY_PX if frame is not None else 0), frame=frame)
    canvas = Canvas(width, height, scene.view_kind, background)
    classes = class_table(scene.view_kind)
    depth_range = max(float(np.ptp(cam.project(scene_points(scene, focus_only=False))[:, 2])), 1.0)
    bias = EDGE_DEPTH_BIAS * depth_range
    see_through = []
    for index, part in enumerate(scene.parts):
        for mesh in part.meshes:
            if mesh.alpha < 1.0 or mesh.transmission > 0.0:
                see_through.append(mesh)
            elif mesh.studio_lit:
                draw_opaque_studio(canvas, cam, mesh, index)
            else:
                draw_opaque(canvas, cam, mesh.positions, mesh.triangles, mesh.rgb, index)
    _beams(canvas, cam, scene, bias)
    draw_see_through(canvas, cam, see_through)
    edge_rgb, edge_alpha = glass_edge(scene.view_kind)
    # outlines: features always, silhouettes for this view; a see-through optic hidden inside its mount is drawn dotted
    for part in scene.parts:
        for group in outline_groups(part, merge_surfaces=scene.view_kind == OPTICAL_VIEW):
            facing = np.einsum("ijk,k->ij", group.sides, cam.forward)
            silhouette = np.sign(facing[:, 0]) != np.sign(facing[:, 1])
            keep = group.feature | silhouette
            if not keep.any():
                continue
            clear = group.see_through
            colour = edge_rgb if clear else tuple(c * EDGE_DARKEN for c in group.rgb)
            draw_lines(canvas, cam, group.points[keep, 0], group.points[keep, 1], colour, width_px=EDGE_WIDTH_PX,
                       alpha=(edge_alpha * (FAINT_EDGE_SHARE if group.faint else 1.0)) if clear else 1.0, depth_bias=bias,
                       hidden_alpha=(HIDDEN_OPTIC_ALPHA * (FAINT_EDGE_SHARE if group.faint else 1.0)) if clear else 0.0)
    for marker in scene.markers:
        cls = classes[marker.marker_class]
        x, y, _ = cam.project(marker.position.reshape(1, 3))[0]
        _marker(canvas, x / SUPERSAMPLE, y / SUPERSAMPLE, cls.rgb, square=marker.marker_class != "CUSTOMER_SUPPLIED")
    if labels:
        _labels(canvas, cam, scene, height - band, width, skip_outside=frame is not None)
    _title(canvas, scene, view)
    legend = _legend(canvas, layout, height - band)
    _triad(canvas, cam, height - band)
    bar = _scale_bar(canvas, cam, width, height - band)
    drawn = {"legend": legend, "scale_bar_mm": bar, "pixels_per_mm": cam.scale / SUPERSAMPLE,
             "width_px": width, "height_px": height, "image_px": [width, height], "requested_px": requested,
             "size_mode": "fit" if frame is None else size_mode,
             "view_direction": [float(-c) for c in cam.forward], "screen_right": [float(c) for c in cam.right],
             "screen_up": [float(c) for c in cam.up], "camera": _camera_record(scene, view, cam, frame),
             "background": _background_name(background)}
    if frame is not None:
        drawn["content_box_px"] = _content_box(scene, cam, frame)
    return canvas.image(), drawn


def _background_name(background) -> str:
    if background in (None, STUDIO_BACKGROUND, TRANSPARENT_BACKGROUND):
        return background or STUDIO_BACKGROUND
    return "#" + "".join(f"{round(float(c) * CHANNEL_MAX):02x}" for c in background)


def _content_box(scene: ViewerScene, cam: Camera, frame) -> list[float]:
    """``[x0, y0, x1, y1]``: where the framed content lies in the image, output pixels from the top-left corner."""

    screen = cam.project(frame_points(scene, frame)) / SUPERSAMPLE
    return [round(float(v), CONTENT_DECIMALS) + 0.0 for v in (screen[:, 0].min(), screen[:, 1].min(), screen[:, 0].max(), screen[:, 1].max())]


def _camera_record(scene: ViewerScene, view, cam: Camera, frame) -> dict:
    """The pose a view was drawn from: the view direction (target to eye) and screen up, the centre of what the view
    fits (``target_mm``: the middle of its extent across and up the screen, at the depth of the camera's target), the eye
    distance a perspective camera would take (``distance_mm``; an orthographic image does not depend on it) and the
    angles of ``model.camera_from_angles`` measured back from those vectors."""

    pose = _pose(scene, view)
    direction = -cam.forward
    points = frame_points(scene, frame)
    across, up = (points - cam.target) @ cam.right, (points - cam.target) @ cam.up
    target = cam.target + cam.right * ((across.max() + across.min()) / 2.0) + cam.up * ((up.max() + up.min()) / 2.0)
    _, fitted = fit_distance(points)
    distance = pose.get("distance")
    if isinstance(distance, str):  # "<k>x": k times the size (box diagonal) of what the view frames
        distance = float(distance[:-1]) * float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    distance = fitted if distance is None else float(distance)
    return {"name": view_name(view), "projection": "orthographic", "view_direction": _rounded(direction), "up": _rounded(cam.up),
            "target_mm": _rounded(target), "distance_mm": round(distance, TARGET_DECIMALS) + 0.0,
            "eye_mm": _rounded(target + direction * distance), **camera_angles(scene, direction, cam.up)}


def _rounded(values) -> list[float]:
    return [round(float(v), TARGET_DECIMALS) + 0.0 for v in values]


def _beams(canvas: Canvas, cam: Camera, scene: ViewerScene, bias: float) -> None:
    """Weak stretches first, main stretches over them. Main beams of different colours that run on the same line nest
    (each earlier colour a little wider), so every colour stays visible; display extensions are dashed and faint."""

    colours: list = []
    for beam in scene.beams:
        if beam.rgb not in colours:
            colours.append(beam.rgb)
    nested = min(len(colours), MAX_NESTED_BEAMS)
    for beam in scene.beams:  # under every main beam, so that it never covers another colour nested on the same line
        if beam_needs_casing(beam.rgb, scene.view_kind):
            for role, pts in beam.runs:
                if role in ("PRIMARY", "DESIGN_AXIS") and len(pts) >= 2:
                    width = BEAM_STYLES[role]["width_px"] + BEAM_CASING_EXTRA_PX + NESTED_BEAM_EXTRA_PX * (nested - 1 - min(colours.index(beam.rgb), nested - 1))
                    draw_lines(canvas, cam, pts[:-1], pts[1:], BEAM_CASING_RGB, width_px=width, alpha=BEAM_CASING_ALPHA, depth_bias=bias, beam=True)
    for wanted in (("SECONDARY_PUPIL", "PUPIL"), ("SECONDARY", "REFERENCE"), ("PRIMARY", "DESIGN_AXIS")):
        for beam in sorted(scene.beams, key=lambda b: colours.index(b.rgb)):
            rank = min(colours.index(beam.rgb), nested - 1)
            for extension, (role, pts) in [(False, run) for run in beam.runs] + [(True, run) for run in beam.extensions]:
                if role not in wanted or len(pts) < 2:
                    continue
                style = BEAM_STYLES["PUPIL" if role == "SECONDARY_PUPIL" else role]
                main = role in ("PRIMARY", "DESIGN_AXIS")
                width = style["width_px"] + (NESTED_BEAM_EXTRA_PX * (nested - 1 - rank) if main else 0.0)
                alpha = style["alpha"] * (HIDDEN_BEAM_ALPHA * 2.0 if role == "SECONDARY_PUPIL" or extension else 1.0)
                draw_lines(canvas, cam, pts[:-1], pts[1:], style.get("rgb", beam.rgb), width_px=width, alpha=alpha,
                           dash=style["dash"] or extension, depth_bias=bias, hidden_alpha=HIDDEN_BEAM_ALPHA * 2.0 * alpha if main else 0.0,
                           beam=True)


def _marker(canvas: Canvas, x: float, y: float, rgb, *, square: bool, radius: float = MARKER_RADIUS_PX) -> None:
    r = radius * SUPERSAMPLE
    dy, dx = np.mgrid[-int(r) - SUPERSAMPLE:int(r) + SUPERSAMPLE + 1, -int(r) - SUPERSAMPLE:int(r) + SUPERSAMPLE + 1]
    dist = np.maximum(np.abs(dx), np.abs(dy)) if square else np.hypot(dx, dy)
    cx, cy = int(x * SUPERSAMPLE), int(y * SUPERSAMPLE)
    for limit, colour in ((r + SUPERSAMPLE, TEXT_RGB), (r, rgb)):
        yy, xx = np.nonzero(dist <= limit)
        yy, xx = yy + cy - int(r) - SUPERSAMPLE, xx + cx - int(r) - SUPERSAMPLE
        ok = (yy >= 0) & (yy < canvas.h) & (xx >= 0) & (xx < canvas.w)
        canvas.blend(yy[ok], xx[ok], colour, 1.0)


def _overlaps(box, placed) -> bool:
    return any(not (box[2] < p[0] or box[0] > p[2] or box[3] < p[1] or box[1] > p[3]) for p in placed)


def _anchors(canvas: Canvas, cam: Camera, scene: ViewerScene) -> list[tuple]:
    """(text, anchor x, anchor y, box x0, y0, x1, y1) per labelled part / marker, output pixels. The anchor is the middle
    of the part's visible pixels (its projected centre when it is see-through or hidden)."""

    items = []
    visible = canvas.ids >= 0
    ys, xs = np.nonzero(visible)
    owner = canvas.ids[ys, xs]
    for index, part in enumerate(scene.parts):
        if part.part_class == "BREADBOARD" or not part.meshes:
            continue
        screen = cam.project(np.concatenate([m.positions for m in part.meshes])) / SUPERSAMPLE
        box = (float(screen[:, 0].min()), float(screen[:, 1].min()), float(screen[:, 0].max()), float(screen[:, 1].max()))
        mine = owner == index
        if mine.any():
            ax, ay = float(np.median(xs[mine])) / SUPERSAMPLE, float(np.median(ys[mine])) / SUPERSAMPLE
        else:
            ax, ay = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        items.append((part.part_id, ax, ay, *box))
    for marker in scene.markers:
        x, y, _ = cam.project(marker.position.reshape(1, 3))[0] / SUPERSAMPLE
        items.append((marker.marker_id, float(x), float(y), x - MARKER_RADIUS_PX, y - MARKER_RADIUS_PX, x + MARKER_RADIUS_PX,
                      y + MARKER_RADIUS_PX))
    return items


def _labels(canvas: Canvas, cam: Camera, scene: ViewerScene, height: int, right_limit: int, *, skip_outside: bool = False) -> None:
    """Part ids in small boxes beside their parts, with a leader line to the part; boxes never overlap each other.
    ``skip_outside``: no label for what lies wholly outside the image (a frame that leaves it out)."""

    placed: list[tuple[float, float, float, float]] = []
    anchors = _anchors(canvas, cam, scene)
    if skip_outside:
        anchors = [a for a in anchors if a[5] >= 0 and a[3] <= right_limit and a[6] >= TOP_RESERVED_PX and a[4] <= height]
    items = sorted(anchors, key=lambda item: (item[2], item[1]))
    s = SUPERSAMPLE
    small = (canvas.h // s, canvas.w // s)
    beams = np.isfinite(canvas.zbeam).reshape(small[0], s, small[1], s).any(axis=(1, 3))
    solid = (canvas.ids >= 0).reshape(small[0], s, small[1], s).any(axis=(1, 3))
    cost_map = np.pad((beams * LABEL_BEAM_COST + solid * LABEL_PART_COST).cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)))

    def cost(box) -> float:
        x0, y0 = int(min(max(box[0], 0), small[1])), int(min(max(box[1], 0), small[0]))
        x1, y1 = int(min(max(box[2], 0), small[1])), int(min(max(box[3], 0), small[0]))
        return float(cost_map[y1, x1] - cost_map[y0, x1] - cost_map[y1, x0] + cost_map[y0, x0])

    for text, ax, ay, bx0, by0, bx1, by1 in items:
        w, h = text_size(text)
        candidates = []
        for k in range(LABEL_TRIES):
            for dy in ((0,) if k == 0 else (-k * (h + LABEL_GAP_PX), k * (h + LABEL_GAP_PX))):
                candidates += [(bx1 + LABEL_GAP_PX, ay - h / 2 + dy), (bx0 - LABEL_GAP_PX - w, ay - h / 2 + dy),
                               (ax - w / 2 + dy, by0 - LABEL_GAP_PX - h), (ax - w / 2 + dy, by1 + LABEL_GAP_PX)]
        box, best = None, math.inf
        for rank, (x0, y0) in enumerate(candidates):
            x0 = min(max(x0, 2), right_limit - w - 4)
            y0 = min(max(y0, TOP_RESERVED_PX), height - h - 2)
            candidate = (x0 - 3, y0 - 2, x0 + w + 3, y0 + h + 2)
            if _overlaps(candidate, placed):
                continue
            score = cost(candidate) + rank * LABEL_DISTANCE_COST  # keep beams and parts uncovered, then stay close
            if score < best:
                box, best = candidate, score
        if box is None:
            continue
        placed.append(box)
        near_x = min(max(ax, box[0]), box[2])
        near_y = min(max(ay, box[1]), box[3])
        if math.hypot(near_x - ax, near_y - ay) > LABEL_GAP_PX:
            screen_line(canvas, (near_x, near_y), (ax, ay), canvas.ink, EDGE_WIDTH_PX, LEGEND_ALPHA)
            _marker(canvas, ax, ay, canvas.ink, square=False, radius=LEADER_DOT_PX)
        fill_rect(canvas, *box, (1.0, 1.0, 1.0), LEGEND_ALPHA)
        draw_text(canvas, text, box[0] + 3, box[1] + 2, halo=False)


#: The coordinate frame of a view in words (the snapshot header).
FRAME_WORDS = {"PROJECT_DESIGN_FRAME": "design", "ASSEMBLY_WORLD": "assembly"}


def view_title(scene: ViewerScene, view) -> str:
    """The words a view is titled with: a preset's ``VIEW_TITLES``, an explicit pose's ``title`` (else its name)."""

    if isinstance(view, str):
        return VIEW_TITLES.get(view, view)
    return str(view.get("title") or view_name(view))


ELLIPSIS = "..."


def _fit_line(candidates: tuple[str, ...], width_px: float) -> str:
    """The first of ``candidates`` (fullest first) that is within ``width_px`` output pixels; when none is, the last one cut
    with an ellipsis, so a title is never cut off mid-word by the image edge."""

    for line in candidates:
        if text_size(line)[0] <= width_px:
            return line
    text = candidates[-1]
    while len(text) > 1 and text_size(text + ELLIPSIS)[0] > width_px:
        text = text[:-1]
    return text.rstrip() + ELLIPSIS


def _title(canvas: Canvas, scene: ViewerScene, view) -> None:
    fill_rect(canvas, 0, 0, canvas.w / SUPERSAMPLE, TOP_RESERVED_PX - 4, (1.0, 1.0, 1.0))
    room = canvas.w / SUPERSAMPLE - MARGIN_PX / 3 * 2
    words = view_title(scene, view)  # a narrow image: without the scene's name, then without "orthographic", then cut
    title = _fit_line((f"{scene.title} - {words}, orthographic", f"{scene.title} - {words}", words), room)
    draw_text(canvas, title, MARGIN_PX / 3, 5, halo=False)
    subtitle = f"units: mm   design up: +Z   frame: {FRAME_WORDS.get(scene.frame, scene.frame.lower())}   exact geometry"
    draw_text(canvas, _fit_line((subtitle,), room), MARGIN_PX / 3, 5 + GLYPH_HEIGHT * TEXT_SCALE + 4, halo=False, rgb=SUBTITLE_RGB)


def _legend_rows(scene: ViewerScene) -> list[tuple[str, tuple, str]]:
    classes = class_table(scene.view_kind)
    rows: list[tuple[str, tuple, str]] = [(classes[k].en, tuple(scene.class_swatches(k)), "boxes") for k in scene.classes]
    roles: list[str] = []
    for beam in scene.beams:
        for role, _ in beam.runs:
            if role in ("PRIMARY", "SECONDARY", "DESIGN_AXIS", "REFERENCE") and role not in roles:
                roles.append(role)
    colours: list = []
    for beam in scene.beams:
        if (beam.rgb, beam.wavelength_nm) not in colours:
            colours.append((beam.rgb, beam.wavelength_nm))
    for role in roles:
        for rgb, wavelength in colours:
            band = f" {wavelength:g} nm" if wavelength is not None and len(colours) > 1 and role != "SECONDARY" else ""
            rows.append((BEAM_STYLES[role]["en"] + band, BEAM_STYLES[role].get("rgb", rgb), role))
            if role in ("SECONDARY", "REFERENCE"):
                break  # one row names the weak-branch / reference-axis style
    return rows


def _legend_layout(rows, width: int) -> list[tuple[int, int, tuple]]:
    """Legend entries flowed left to right into lines across the bottom band: ``(line, x, row)``."""

    out, line, x = [], 0, LEGEND_PAD_PX
    for row in rows:
        item_w = LEGEND_SWATCH_PX * 2 + LEGEND_PAD_PX + text_size(row[0])[0] + LEGEND_ITEM_GAP_PX
        if x + item_w > width - LEGEND_PAD_PX and x > LEGEND_PAD_PX:
            line, x = line + 1, LEGEND_PAD_PX
        out.append((line, x, row))
        x += item_w
    return out


def _legend_band_height(layout) -> int:
    if not layout:
        return 0
    return LEGEND_PAD_PX * 2 + LEGEND_ROW_PX * (layout[-1][0] + 1)


def _legend(canvas: Canvas, layout, band_top: int) -> list[dict]:
    if not layout:
        return []
    fill_rect(canvas, 0, band_top, canvas.w / SUPERSAMPLE, canvas.h / SUPERSAMPLE, (1.0, 1.0, 1.0))
    screen_line(canvas, (0, band_top), (canvas.w / SUPERSAMPLE, band_top), RULE_RGB, 1.0)
    out = []
    for line, sx, (label, rgb, kind) in layout:
        y = band_top + LEGEND_PAD_PX + line * LEGEND_ROW_PX
        if kind == "boxes":  # one box split among the colours the class is drawn in (a compound unit: its members')
            colours, rgb = rgb, rgb[0]
            fill_rect(canvas, sx, y + 1, sx + LEGEND_SWATCH_PX * 2, y + 1 + LEGEND_SWATCH_PX, tuple(c * EDGE_DARKEN for c in rgb))
            step = (LEGEND_SWATCH_PX * 2 - 2) / len(colours)
            for k, colour in enumerate(colours):
                fill_rect(canvas, sx + 1 + k * step, y + 2, sx + 1 + (k + 1) * step, y + LEGEND_SWATCH_PX, colour)
        else:
            style = BEAM_STYLES[kind]
            mid = y + 1 + LEGEND_SWATCH_PX / 2
            if style["dash"]:
                for d in range(0, LEGEND_SWATCH_PX * 2, int(DASH_PX / 2 + GAP_PX / 2) + 1):
                    screen_line(canvas, (sx + d, mid), (sx + min(d + DASH_PX / 2, LEGEND_SWATCH_PX * 2), mid), rgb, style["width_px"],
                                style["alpha"])
            else:
                screen_line(canvas, (sx, mid), (sx + LEGEND_SWATCH_PX * 2, mid), rgb, style["width_px"], style["alpha"])
        draw_text(canvas, label, sx + LEGEND_SWATCH_PX * 2 + LEGEND_PAD_PX, y + 1, halo=False)
        out.append({"label": label, "rgb": [round(float(c), 4) for c in rgb], "kind": "part_class" if kind == "boxes" else "beam"})
    return out


def _triad(canvas: Canvas, cam: Camera, height: int) -> None:
    ox, oy = TRIAD_ORIGIN_PX[0], height - TRIAD_ORIGIN_PX[1]
    for name, axis in (("X", np.array([1.0, 0.0, 0.0])), ("Y", np.array([0.0, 1.0, 0.0])), ("Z", np.array([0.0, 0.0, 1.0]))):
        dx, dy = float(axis @ cam.right) * TRIAD_LENGTH_PX, -float(axis @ cam.up) * TRIAD_LENGTH_PX
        rgb = AXIS_RGB[name]
        if canvas.ink is INK_ON_DARK:
            rgb = tuple(c + (1.0 - c) * AXIS_ON_DARK for c in rgb)
        if math.hypot(dx, dy) < TRIAD_MIN_PX:  # pointing at or away from the viewer
            toward = float(axis @ cam.forward) < 0.0
            _marker(canvas, ox, oy, rgb, square=False)
            _marker(canvas, ox, oy, (1.0, 1.0, 1.0) if toward else rgb, square=False, radius=MARKER_RADIUS_PX / 2)
            draw_text(canvas, f"{name} {'toward you' if toward else 'away from you'}", ox - MARKER_RADIUS_PX, oy + MARKER_RADIUS_PX + 4, rgb,
                      halo=canvas.ink is not INK_ON_DARK)
            continue
        screen_line(canvas, (ox, oy), (ox + dx, oy + dy), rgb, 2.0)
        draw_text(canvas, name, ox + dx * TRIAD_LABEL_REACH - 5, oy + dy * TRIAD_LABEL_REACH - GLYPH_HEIGHT, rgb, halo=canvas.ink is not INK_ON_DARK)


def _scale_bar(canvas: Canvas, cam: Camera, width: int, height: int) -> float:
    per_mm = cam.scale / SUPERSAMPLE
    length = _nice_length(SCALE_BAR_TARGET_PX / per_mm)
    px = length * per_mm
    x1 = width - MARGIN_PX
    x0 = x1 - px
    y = height - MARGIN_PX
    screen_line(canvas, (x0, y), (x1, y), canvas.ink, 2.0)
    screen_line(canvas, (x0, y - 5), (x0, y + 5), canvas.ink, 2.0)
    screen_line(canvas, (x1, y - 5), (x1, y + 5), canvas.ink, 2.0)
    label = f"{_fmt(length)} mm"
    w, _ = text_size(label)
    draw_text(canvas, label, (x0 + x1) / 2 - w / 2, y - GLYPH_HEIGHT * TEXT_SCALE - 8, canvas.ink, halo=canvas.ink is not INK_ON_DARK)
    return length
