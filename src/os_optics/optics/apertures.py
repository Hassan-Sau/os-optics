"""Aperture regions in a surface's in-plane coordinates ``(u, v)``: one place for every question asked of an aperture.

An ``ApertureG`` (``scene/model.py``) is one of

* ``none``: unbounded;
* ``circle``: radius ``p1`` about ``(offset_u, offset_v)``, with an optional opaque central disc of radius ``obstruction``
  (an annulus; the obstruction blocks light, it is part of the element);
* ``rectangle``: half sides ``p1`` (along u) and ``p2`` (along v) about the offset;
* ``ellipse``: semi-axes ``p1`` along the unit in-plane direction ``axis`` and ``p2`` across it, about the offset;
* ``d_shape``: the disc of radius ``p1`` about the offset cut by a flat: the points whose coordinate along ``axis`` is at
  most ``p2`` (``-p1 < p2 < p1``; ``axis`` is the flat's outward normal);
* ``polygon``: a simple polygon (convex or not, triangles included), corners ``vertices`` relative to the offset,
  counter-clockwise;

and ``holes``: openings through the element (each a circle, rectangle, ellipse, d_shape or polygon with its own offset in
the same ``(u, v)`` frame, inside the region and apart from each other). A hole is not part of the element: there is no
surface there, light passes straight through it. A region with holes is the outer shape less its holes.

Every function keeps the exact operations (and so the exact bits) the circle and rectangle had before the other shapes
existed; the other shapes are answered exactly where the answer is exact and otherwise by a stated sound bound (a lower
bound of a distance, a range that holds the true one). Only ``+ - * /``, ``sqrt``, ``abs``, ``min`` / ``max`` and
``math.hypot`` where the circle and rectangle used it: the same bits in every runtime.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from ..math.exact_trig import atan2, cos_sin
from ..math.vectors import correctly_rounded_sum
from ..scene.model import ApertureG

Array = NDArray[np.float64]

NONE, CIRCLE, RECTANGLE, ELLIPSE, D_SHAPE, POLYGON = "none", "circle", "rectangle", "ellipse", "d_shape", "polygon"
#: The shapes an aperture region (and a hole) may have.
SHAPES = (NONE, CIRCLE, RECTANGLE, ELLIPSE, D_SHAPE, POLYGON)

#: Points per full turn of a round boundary drawn as a closed ring (``boundary``); a ring holds every corner of a polygon.
RING_POINTS = 64


# --------------------------------------------------------------------------------------------- frame helpers


def _local(ap: ApertureG, du: Array | float, dv: Array | float) -> tuple[Array | float, Array | float]:
    """Coordinates along ``axis`` and across it (``axis`` turned +90 degrees) of an offset from the centre."""
    c, s = ap.axis
    return c * du + s * dv, c * dv - s * du


def _polygon_arrays(ap: ApertureG) -> tuple[Array, Array]:
    corners = np.asarray(ap.vertices, dtype=np.float64).reshape(-1, 2)
    return corners[:, 0], corners[:, 1]


# --------------------------------------------------------------------------------------------- membership


def bounded(ap: ApertureG | None) -> bool:
    """Has the region a boundary (any shape but ``none``)?"""
    return ap is not None and ap.shape != NONE


def _square(x: float) -> float:
    """``x * x`` (a product, not libm ``pow``); a radius whose square overflows is refused like ``pow`` refuses it."""
    square = float(x) * float(x)
    if not math.isfinite(square):
        raise OverflowError(f"aperture radius {x!r} mm is too large: its square overflows")
    return square


def _outer_inside(ap: ApertureG, u: Array, v: Array, tol: float) -> NDArray[np.bool_]:
    du, dv = u - ap.offset_u, v - ap.offset_v
    if ap.shape == CIRCLE:
        r2 = du * du + dv * dv
        inside = r2 <= _square(ap.p1 + tol)
        if ap.obstruction > 0.0:
            inside &= r2 >= _square(ap.obstruction - tol)
        return inside
    if ap.shape == RECTANGLE:
        return (np.abs(du) <= ap.p1 + tol) & (np.abs(dv) <= ap.p2 + tol)
    if ap.shape == ELLIPSE:
        x, y = _local(ap, du, dv)
        a, b = ap.p1 + tol, ap.p2 + tol
        return (x * x) / (a * a) + (y * y) / (b * b) <= 1.0
    if ap.shape == D_SHAPE:
        x, _y = _local(ap, du, dv)
        return (du * du + dv * dv <= _square(ap.p1 + tol)) & (x <= ap.p2 + tol)
    return _polygon_inside(ap, np.asarray(du, dtype=np.float64), np.asarray(dv, dtype=np.float64), tol)


def _polygon_inside(ap: ApertureG, du: Array, dv: Array, tol: float) -> NDArray[np.bool_]:
    """Even-odd membership of a simple polygon, the boundary within ``tol`` included."""
    xs, ys = _polygon_arrays(ap)
    inside = np.zeros(np.shape(du), dtype=bool)
    near = np.zeros(np.shape(du), dtype=bool)
    n = xs.size
    for i in range(n):
        x1, y1, x2, y2 = float(xs[i]), float(ys[i]), float(xs[(i + 1) % n]), float(ys[(i + 1) % n])
        straddles = (y1 > dv) != (y2 > dv)
        if y2 != y1:
            crossing = x1 + (x2 - x1) * (dv - y1) / (y2 - y1)
            inside ^= straddles & (du < crossing)
        near |= _segment_distance_sq(du, dv, x1, y1, x2, y2) <= tol * tol
    # a positive tolerance takes in the boundary band, a negative one leaves it out
    return inside | near if tol >= 0.0 else inside & ~near


def _segment_distance_sq(px: Array, py: Array, x1: float, y1: float, x2: float, y2: float) -> Array:
    """Squared distance of the points ``(px, py)`` from the segment ``(x1, y1)-(x2, y2)``."""
    ex, ey = x2 - x1, y2 - y1
    length_sq = ex * ex + ey * ey
    rx, ry = px - x1, py - y1
    if length_sq == 0.0:
        return rx * rx + ry * ry
    t = np.clip((rx * ex + ry * ey) / length_sq, 0.0, 1.0)
    qx, qy = rx - t * ex, ry - t * ey
    return qx * qx + qy * qy


def inside(ap: ApertureG, u: Array, v: Array, tol: float) -> NDArray[np.bool_]:
    """Inclusive (within ``tol``) membership of the in-surface points ``(u, v)``: in the region and not in a hole (a hole's
    own boundary, within ``tol``, belongs to the element)."""
    if ap.shape == NONE:
        return np.ones_like(u, dtype=bool)
    result = _outer_inside(ap, u, v, tol)
    for hole in ap.holes:
        result = result & ~_outer_inside(hole, u, v, -tol)
    return result


# --------------------------------------------------------------------------------------------- distances


def _outer_distance(ap: ApertureG, u: float, v: float) -> float:
    """Signed distance from the outer boundary (positive outside): exact for a circle and a polygon; a rectangle's outside
    the larger of the two axis distances, an ellipse's ``b (|A d| - 1)`` (``b`` the smaller semi-axis, ``A`` the map to the
    unit disc), a d_shape's outside the larger of the disc's and the flat's: each a lower bound of the distance outside and
    of the distance to the boundary inside."""
    du, dv = u - ap.offset_u, v - ap.offset_v
    if ap.shape == CIRCLE:
        return float(np.sqrt(du * du + dv * dv)) - ap.p1
    if ap.shape == RECTANGLE:
        return max(abs(du) - ap.p1, abs(dv) - ap.p2)
    if ap.shape == ELLIPSE:
        x, y = _local(ap, du, dv)
        scaled = math.sqrt((x / ap.p1) * (x / ap.p1) + (y / ap.p2) * (y / ap.p2))
        return min(ap.p1, ap.p2) * (scaled - 1.0)
    if ap.shape == D_SHAPE:
        x, _y = _local(ap, du, dv)
        return max(math.sqrt(du * du + dv * dv) - ap.p1, x - ap.p2)
    xs, ys = _polygon_arrays(ap)
    n = xs.size
    nearest = math.inf
    for i in range(n):
        nearest = min(nearest, float(_segment_distance_sq(np.float64(du), np.float64(dv), float(xs[i]), float(ys[i]),
                                                          float(xs[(i + 1) % n]), float(ys[(i + 1) % n]))))
    distance = math.sqrt(nearest)
    held = bool(_polygon_inside(ap, np.array([du]), np.array([dv]), 0.0)[0])
    return -distance if held else distance


def miss_distance(ap: ApertureG, u: float, v: float) -> float:
    """Distance (mm) from the point ``(u, v)`` to the edge of an aperture (positive outside, negative inside); a point in
    a hole is outside, by its distance to the hole's edge. A lower bound of the true distance (``_outer_distance``)."""
    outer = _outer_distance(ap, u, v)
    if not ap.holes or outer > 0.0:
        return outer
    for hole in ap.holes:
        into = -_outer_distance(hole, u, v)
        if into >= 0.0:
            return into
        outer = max(outer, into)
    return outer


def room(ap: ApertureG, u: float, v: float) -> float:
    """How far the point ``(u, v)`` lies inside the region (negative outside): ``-miss_distance`` for a circle and a
    rectangle, and a lower bound of the radius of the largest disc about the point within the region for every shape."""
    du, dv = u - ap.offset_u, v - ap.offset_v
    if ap.shape == CIRCLE and not ap.holes:
        return ap.p1 - math.sqrt(du * du + dv * dv)
    if ap.shape == RECTANGLE and not ap.holes:
        return min(ap.p1 - abs(du), ap.p2 - abs(dv))
    return -miss_distance(ap, u, v)


# --------------------------------------------------------------------------------------------- extents


def dilate(ap: ApertureG, margin: float) -> ApertureG:
    """The region grown by ``margin`` (a mount ring around the clear aperture; any central obstruction is dropped, holes are
    kept): a circle's radius, a rectangle's half sides, an ellipse's semi-axes, a d_shape's radius and flat, and a polygon's
    edges each moved out by ``margin`` (corners where the moved edges meet: a convex polygon's corners move out along their
    bisectors)."""
    if margin == 0.0:
        return replace(ap, obstruction=0.0)
    if ap.shape == RECTANGLE:
        return replace(ap, p1=ap.p1 + margin, p2=ap.p2 + margin, obstruction=0.0)
    if ap.shape in (ELLIPSE, D_SHAPE):
        return replace(ap, p1=ap.p1 + margin, p2=ap.p2 + margin, obstruction=0.0)
    if ap.shape == POLYGON:
        return replace(ap, vertices=offset_polygon(ap.vertices, margin), obstruction=0.0)
    return replace(ap, p1=ap.p1 + margin, obstruction=0.0)


def offset_polygon(vertices: tuple[tuple[float, float], ...], margin: float) -> tuple[tuple[float, float], ...]:
    """A counter-clockwise polygon with every edge moved out by ``margin`` along its outward normal; each corner is where its
    two moved edges meet."""
    n = len(vertices)
    normals = []
    for i in range(n):
        (x1, y1), (x2, y2) = vertices[i], vertices[(i + 1) % n]
        ex, ey = x2 - x1, y2 - y1
        length = math.sqrt(ex * ex + ey * ey)
        normals.append((ey / length, -ex / length))  # outward for a counter-clockwise polygon
    out = []
    for i in range(n):
        (ax, ay), (bx, by) = normals[i - 1], normals[i]
        x, y = vertices[i]
        # the corner moves by m (na + nb) / (1 + na . nb): on both moved lines
        scale = margin / (1.0 + ax * bx + ay * by)
        out.append((x + scale * (ax + bx), y + scale * (ay + by)))
    return tuple(out)


def bounds(ap: ApertureG) -> tuple[float, float, float, float]:
    """``(u_lo, u_hi, v_lo, v_hi)``: the region's bounding box in the surface's ``(u, v)`` frame (a circle and a rectangle:
    the offset less / plus the half sides)."""
    half_u, half_v = ap.p1, (ap.p1 if ap.shape == CIRCLE else ap.p2)
    if ap.shape == ELLIPSE:
        c, s = ap.axis
        half_u = math.sqrt((ap.p1 * c) * (ap.p1 * c) + (ap.p2 * s) * (ap.p2 * s))
        half_v = math.sqrt((ap.p1 * s) * (ap.p1 * s) + (ap.p2 * c) * (ap.p2 * c))
    elif ap.shape == D_SHAPE:
        lo_u, hi_u = -spread(ap, -1.0, 0.0), spread(ap, 1.0, 0.0)
        lo_v, hi_v = -spread(ap, 0.0, -1.0), spread(ap, 0.0, 1.0)
        return ap.offset_u + lo_u, ap.offset_u + hi_u, ap.offset_v + lo_v, ap.offset_v + hi_v
    elif ap.shape == POLYGON:
        xs, ys = _polygon_arrays(ap)
        return (ap.offset_u + float(np.min(xs)), ap.offset_u + float(np.max(xs)),
                ap.offset_v + float(np.min(ys)), ap.offset_v + float(np.max(ys)))
    return ap.offset_u - half_u, ap.offset_u + half_u, ap.offset_v - half_v, ap.offset_v + half_v


def spread(ap: ApertureG, du: float, dv: float) -> float:
    """``max (x - offset) . (du, dv)`` over the region (its support function about the offset): exact for every shape."""
    if ap.shape == CIRCLE:
        return ap.p1 * math.sqrt(du * du + dv * dv)
    if ap.shape == RECTANGLE:
        return ap.p1 * abs(du) + ap.p2 * abs(dv)
    if ap.shape == ELLIPSE:
        x, y = _local(ap, du, dv)
        return math.sqrt((ap.p1 * x) * (ap.p1 * x) + (ap.p2 * y) * (ap.p2 * y))
    if ap.shape == D_SHAPE:
        x, y = _local(ap, du, dv)
        length = math.sqrt(du * du + dv * dv)
        if length == 0.0 or ap.p1 * x <= ap.p2 * length:  # the disc's farthest point lies on the kept side
            return ap.p1 * length
        chord = math.sqrt(max(ap.p1 * ap.p1 - ap.p2 * ap.p2, 0.0))
        return ap.p2 * x + chord * abs(y)
    xs, ys = _polygon_arrays(ap)
    return float(np.max(xs * du + ys * dv))


def reach(ap: ApertureG) -> float:
    """The largest distance of a point of the region from its offset (a circle's radius, a rectangle's half diagonal)."""
    if ap.shape == CIRCLE:
        return ap.p1
    if ap.shape == RECTANGLE:
        return math.sqrt(ap.p1 * ap.p1 + ap.p2 * ap.p2)
    if ap.shape in (ELLIPSE, D_SHAPE):
        return max(ap.p1, ap.p2) if ap.shape == ELLIPSE else ap.p1
    xs, ys = _polygon_arrays(ap)
    return float(np.sqrt(np.max(xs * xs + ys * ys)))


def farthest(ap: ApertureG, du: float, dv: float) -> float:
    """An upper bound of the distance from the point ``(-du, -dv)`` (relative to the offset) to any point of the region:
    exact for a circle and a rectangle."""
    if ap.shape == CIRCLE:
        return math.hypot(du, dv) + ap.p1
    if ap.shape == RECTANGLE:
        return math.hypot(abs(du) + ap.p1, abs(dv) + ap.p2)
    return math.hypot(du, dv) + reach(ap)


def rho_range(ap: ApertureG) -> tuple[float, float]:
    """``(near, far)``: a range that holds the distance from the frame's origin ``(0, 0)`` of every point of the region (a
    circle's and a rectangle's are exact)."""
    ou, ov = ap.offset_u, ap.offset_v
    if ap.shape == CIRCLE:
        centre = math.sqrt(ou * ou + ov * ov)
        return max(centre - ap.p1, 0.0), centre + ap.p1
    if ap.shape == RECTANGLE:
        p1, p2 = ap.p1, ap.p2
        near_u, near_v = max(abs(ou) - p1, 0.0), max(abs(ov) - p2, 0.0)
        far_u, far_v = abs(ou) + p1, abs(ov) + p2
        return math.sqrt(near_u * near_u + near_v * near_v), math.sqrt(far_u * far_u + far_v * far_v)
    centre = math.sqrt(ou * ou + ov * ov)
    radius = reach(ap)
    near = max(centre - radius, 0.0)
    if ap.shape == POLYGON and room(ap, 0.0, 0.0) >= 0.0:
        near = 0.0
    u_lo, u_hi, v_lo, v_hi = bounds(ap)
    far_u, far_v = max(abs(u_lo), abs(u_hi)), max(abs(v_lo), abs(v_hi))
    return near, min(centre + radius, math.sqrt(far_u * far_u + far_v * far_v))


def far_u(ap: ApertureG) -> float:
    """The largest ``|u|`` over the region (a cylinder's profile coordinate along u)."""
    u_lo, u_hi, _v_lo, _v_hi = bounds(ap)
    return max(abs(u_lo), abs(u_hi))


def index_area(ap: ApertureG) -> tuple[int, float, float]:
    """A bounded active area as the scene index's pre-filters read it: ``(1, radius, p2)`` a disc, ``(2, half_u, half_v)``
    a rectangle about the offset; any other shape as its enclosing disc about the offset (a pre-filter that keeps more
    candidates, never fewer)."""
    if ap.shape == CIRCLE:
        return 1, ap.p1, ap.p2
    if ap.shape == RECTANGLE:
        return 2, ap.p1, ap.p2
    return 1, reach(ap), 0.0


def convex(ap: ApertureG) -> bool:
    """Is the region convex (no central obstruction, no hole, not a polygon with a reflex corner)?"""
    if ap.obstruction > 0.0 or ap.holes:
        return False
    if ap.shape != POLYGON:
        return True
    n = len(ap.vertices)
    for i in range(n):
        (x0, y0), (x1, y1), (x2, y2) = ap.vertices[i - 1], ap.vertices[i], ap.vertices[(i + 1) % n]
        if (x1 - x0) * (y2 - y1) - (y1 - y0) * (x2 - x1) < 0.0:
            return False
    return True


def area(ap: ApertureG) -> float:
    """The region's area (mm^2), holes and obstruction removed; ``inf`` unbounded."""
    if ap.shape == NONE:
        return math.inf
    total = _outer_area(ap) - correctly_rounded_sum(_outer_area(hole) for hole in ap.holes)
    if ap.shape == CIRCLE and ap.obstruction > 0.0:
        total -= math.pi * ap.obstruction * ap.obstruction
    return total


def _outer_area(ap: ApertureG) -> float:
    if ap.shape == CIRCLE:
        return math.pi * ap.p1 * ap.p1
    if ap.shape == RECTANGLE:
        return 4.0 * ap.p1 * ap.p2
    if ap.shape == ELLIPSE:
        return math.pi * ap.p1 * ap.p2
    if ap.shape == D_SHAPE:
        half_angle = flat_half_angle(ap)
        return ap.p1 * ap.p1 * (math.pi - half_angle) + ap.p2 * math.sqrt(ap.p1 * ap.p1 - ap.p2 * ap.p2)
    xs, ys = _polygon_arrays(ap)
    return 0.5 * float(np.sum(xs * np.roll(ys, -1) - np.roll(xs, -1) * ys))


# --------------------------------------------------------------------------------------------- outlines


def ring(ap: ApertureG, points: int = RING_POINTS) -> list[tuple[float, float]]:
    """The outer boundary as a closed counter-clockwise ring of ``(u, v)`` points (not repeating the first): a polygon's and a
    rectangle's corners exactly, a round boundary at ``points`` per turn (a d_shape's flat between its two corners)."""
    ou, ov = ap.offset_u, ap.offset_v
    if ap.shape == RECTANGLE:
        return [(ou - ap.p1, ov - ap.p2), (ou + ap.p1, ov - ap.p2), (ou + ap.p1, ov + ap.p2), (ou - ap.p1, ov + ap.p2)]
    if ap.shape == POLYGON:
        return [(ou + x, ov + y) for x, y in ap.vertices]
    cosines, sines = cos_sin(np.arange(points) * (2.0 * math.pi / points))
    turn = [(float(c), float(s)) for c, s in zip(cosines, sines)]
    if ap.shape == CIRCLE:
        return [(ou + ap.p1 * c, ov + ap.p1 * s) for c, s in turn]
    c0, s0 = ap.axis
    if ap.shape == ELLIPSE:
        return [(ou + ap.p1 * c * c0 - ap.p2 * s * s0, ov + ap.p1 * c * s0 + ap.p2 * s * c0) for c, s in turn]
    # d_shape: the arc beyond the flat's corners, then the flat
    half = flat_half_angle(ap)
    steps = max(2, int(math.ceil((2.0 * math.pi - 2.0 * half) / (2.0 * math.pi / points))))
    cosines, sines = cos_sin(half + (2.0 * math.pi - 2.0 * half) * (np.arange(steps + 1) / steps))
    out = []
    for c, s in zip(cosines, sines):
        x, y = ap.p1 * float(c), ap.p1 * float(s)
        out.append((ou + x * c0 - y * s0, ov + x * s0 + y * c0))
    return out


def flat_half_angle(ap: ApertureG) -> float:
    """A d_shape's half angle (rad) between its axis and a corner of its flat, ``arccos(p2 / p1)`` in ``(0, pi)``."""
    return float(atan2(math.sqrt(max(ap.p1 * ap.p1 - ap.p2 * ap.p2, 0.0)), ap.p2))


def rings(ap: ApertureG, points: int = RING_POINTS) -> list[list[tuple[float, float]]]:
    """The outer ring, then one ring per hole (each counter-clockwise; a hole's ring bounds an opening)."""
    if ap.shape == NONE:
        return []
    return [ring(ap, points), *(ring(hole, points) for hole in ap.holes)]


def describe(ap: ApertureG) -> dict:
    """The region as plain numbers for a report (its shape, sizes, offset and holes)."""
    out: dict = {"shape": ap.shape}
    if ap.shape == CIRCLE:
        out.update(radius_mm=ap.p1, obstruction_radius_mm=ap.obstruction)
    elif ap.shape == RECTANGLE:
        out.update(half_width_mm=ap.p1, half_height_mm=ap.p2)
    elif ap.shape == ELLIPSE:
        out.update(semi_axes_mm=[ap.p1, ap.p2], axis_uv=[ap.axis[0], ap.axis[1]])
    elif ap.shape == D_SHAPE:
        out.update(radius_mm=ap.p1, flat_distance_mm=ap.p2, flat_normal_uv=[ap.axis[0], ap.axis[1]])
    elif ap.shape == POLYGON:
        out.update(vertices_mm=[[x, y] for x, y in ap.vertices])
    if ap.shape != NONE:
        out["center_offset_mm"] = [ap.offset_u, ap.offset_v]
    if ap.holes:
        out["holes"] = [describe(hole) for hole in ap.holes]
    return out


def clip_context(ap: ApertureG, u0: float, v0: float, chief_cos: float) -> tuple[dict, float]:
    """What a clip report states about the aperture the chief ray missed at ``(u0, v0)``, and the distance the aperture would
    have to reach (``suggested_min_aperture_radius_mm`` adds the beam radius)."""
    if ap.shape == CIRCLE:
        dist = math.hypot(u0 - ap.offset_u, v0 - ap.offset_v)
        return {"aperture_radius_mm": ap.p1, "projected_aperture_tangential_mm": ap.p1 * chief_cos, "center_to_edge_mm": ap.p1 - dist}, dist
    if ap.shape == RECTANGLE:
        dist = max(abs(u0 - ap.offset_u) - ap.p1, abs(v0 - ap.offset_v) - ap.p2)
        return {"aperture_half_width_mm": ap.p1, "aperture_half_height_mm": ap.p2, "center_to_edge_mm": -dist}, dist
    if ap.shape == NONE:  # pragma: no cover - unbounded apertures cannot clip
        return {}, 0.0
    outside = miss_distance(ap, u0, v0)
    return {"aperture_reach_mm": reach(ap), "center_to_edge_mm": -outside}, math.hypot(u0 - ap.offset_u, v0 - ap.offset_v)


# --------------------------------------------------------------------------------------------- affine images


def _norm2(a: float, b: float, c: float, d: float) -> float:
    """The spectral norm (largest singular value) of the 2 x 2 matrix ``[[a, b], [c, d]]``."""
    trace = a * a + b * b + c * c + d * d
    det = a * d - b * c
    return math.sqrt(0.5 * (trace + math.sqrt(max(trace * trace - 4.0 * det * det, 0.0))))


def _image_shape(ap: ApertureG, matrix: tuple[float, float, float, float], offset: tuple[float, float]
                 ) -> tuple[list[tuple[float, float]], tuple[float, float, tuple[float, float, float, float]] | None]:
    """A set that holds the image of the region ``ap`` (its outer boundary; holes and an obstruction only remove light) under
    ``p -> matrix p + offset`` (``matrix`` row-major): the images of a polygon's / rectangle's corners (the region lies in
    their convex hull), or the image ``{c + L x: |x| <= 1}`` of the ellipse that holds a round region (a circle, an ellipse,
    a d_shape's disc) as ``(c_u, c_v, L)``."""
    m00, m01, m10, m11 = matrix
    ou, ov = ap.offset_u, ap.offset_v
    cu, cv = m00 * ou + m01 * ov + offset[0], m10 * ou + m11 * ov + offset[1]
    if ap.shape in (RECTANGLE, POLYGON):
        corners = ring(ap)
        return [(m00 * u + m01 * v + offset[0], m10 * u + m11 * v + offset[1]) for u, v in corners], None
    if ap.shape == ELLIPSE:
        c0, s0 = ap.axis
        e00, e01, e10, e11 = ap.p1 * c0, -ap.p2 * s0, ap.p1 * s0, ap.p2 * c0
    else:  # (a circle; a d_shape lies in its disc)
        e00, e01, e10, e11 = ap.p1, 0.0, 0.0, ap.p1
    return [], (cu, cv, (m00 * e00 + m01 * e10, m00 * e01 + m01 * e11, m10 * e00 + m11 * e10, m10 * e01 + m11 * e11))


def holds_affine_image(outer: ApertureG, inner: ApertureG, matrix: tuple[float, float, float, float],
                       offset: tuple[float, float], margin: float, tol: float) -> bool:
    """Does the region ``outer`` hold the image of the region ``inner`` under ``p -> matrix p + offset`` (``matrix``
    row-major), grown by ``margin``, within ``tol`` of its edge? A sufficient test, exact for corners and for round regions
    on round ones about the same centre: ``outer`` less its holes must be convex (no obstruction); the image is held by the
    image of the corners or of the enclosing ellipse (``_image_shape``), and each of ``outer``'s constraints -- its edges'
    half planes (a rectangle's sides, a polygon's edges, a d_shape's flat) and its ellipse (a circle, an ellipse, a d_shape's
    disc) -- must hold that set's support grown by ``margin``. A hole of ``outer`` must lie, grown by ``margin``, inside the
    image of a convex hole of ``inner`` (the inverse map takes it into that hole): no light that passed ``inner`` reaches it."""
    if not bounded(outer) or not bounded(inner):
        return False
    if outer.holes:
        shell = replace(outer, holes=())
        if not holds_affine_image(shell, replace(inner, holes=()), matrix, offset, margin, tol):
            return False
        m00, m01, m10, m11 = matrix
        det = m00 * m11 - m01 * m10
        if det == 0.0:
            return False
        inverse = (m11 / det, -m01 / det, -m10 / det, m00 / det)
        back = (-(inverse[0] * offset[0] + inverse[1] * offset[1]), -(inverse[2] * offset[0] + inverse[3] * offset[1]))
        grown = _norm2(*inverse) * margin
        return all(any(convex(source) and holds_affine_image(source, hole, inverse, back, grown, tol) for source in inner.holes)
                   for hole in outer.holes)
    if not convex(outer):
        return False
    points, ellipse = _image_shape(inner, matrix, offset)
    ou, ov = outer.offset_u, outer.offset_v
    half_planes: list[tuple[float, float, float]] = []  # (n_u, n_v, h): n . x <= h, n a unit vector
    if outer.shape == RECTANGLE:
        half_planes = [(1.0, 0.0, ou + outer.p1), (-1.0, 0.0, outer.p1 - ou), (0.0, 1.0, ov + outer.p2), (0.0, -1.0, outer.p2 - ov)]
    elif outer.shape == POLYGON:
        corners = ring(outer)
        for i, (x0, y0) in enumerate(corners):
            x1, y1 = corners[(i + 1) % len(corners)]
            length = math.sqrt((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0))
            if length <= 0.0:
                continue
            nu, nv = (y1 - y0) / length, (x0 - x1) / length  # (outward: the corners turn counter-clockwise)
            half_planes.append((nu, nv, nu * x0 + nv * y0))
    elif outer.shape == D_SHAPE:
        c0, s0 = outer.axis
        half_planes.append((c0, s0, c0 * ou + s0 * ov + outer.p2))
    for nu, nv, h in half_planes:
        if ellipse is not None:
            cu, cv, (l00, l01, l10, l11) = ellipse
            tu, tv = l00 * nu + l10 * nv, l01 * nu + l11 * nv  # (L^T n)
            support = nu * cu + nv * cv + math.sqrt(tu * tu + tv * tv)
        else:
            support = max(nu * u + nv * v for u, v in points)
        if support + margin > h + tol:
            return False
    if outer.shape in (CIRCLE, ELLIPSE, D_SHAPE):
        if outer.shape == ELLIPSE:
            c0, s0 = outer.axis
            t00, t01, t10, t11 = c0 / outer.p1, s0 / outer.p1, -s0 / outer.p2, c0 / outer.p2  # (to the unit disc)
            widest = max(outer.p1, outer.p2)
        else:
            t00, t01, t10, t11 = 1.0 / outer.p1, 0.0, 0.0, 1.0 / outer.p1
            widest = outer.p1
        grow = _norm2(t00, t01, t10, t11) * margin
        slack = 1.0 + tol / widest
        if ellipse is not None:
            cu, cv, (l00, l01, l10, l11) = ellipse
            du, dv = cu - ou, cv - ov
            x, y = t00 * du + t01 * dv, t10 * du + t11 * dv
            spread = _norm2(t00 * l00 + t01 * l10, t00 * l01 + t01 * l11, t10 * l00 + t11 * l10, t10 * l01 + t11 * l11)
            if math.sqrt(x * x + y * y) + spread + grow > slack:
                return False
        else:
            for u, v in points:
                du, dv = u - ou, v - ov
                x, y = t00 * du + t01 * dv, t10 * du + t11 * dv
                if math.sqrt(x * x + y * y) + grow > slack:
                    return False
    return True
