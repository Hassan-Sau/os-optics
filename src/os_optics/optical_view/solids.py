"""Convex solids of ``SURFACE_SET`` models in float arithmetic (element frame, mm).

A published ``ConvexBodyV1`` is the solid an element's light travels in: the intersection of its half-spaces
``n . x <= d`` (a cube, a prism, a rectangular or wedged plate) and, for a round plate, the cylinder of ``side_radius``
about local Z. This module gives the lowering, the validity checks and the optical view the same geometry:

* ``solid_from_planes``: the solid of given half-spaces and side radius, its corners found by intersecting the faces
  three at a time (the local runtime builds it from the contract's exact corners instead, ``local.bodies.solid_of``);
* ``face_polygon``: the corners of one face, counter-clockwise seen from outside (a round plate's faces as a polygon of
  ``segments`` points on the side cylinder, for display);
* ``section_polygon``: the polygon where a plane cuts the solid (an internal coated surface, e.g. a cube's diagonal);
* ``clip_segment``: the exact parameter interval of a segment inside the solid (half-space slabs and, for a round
  plate, the side cylinder's quadratic).

Pure Python (``math``, no platform trigonometry): the optical view, which must not import NumPy, and the local runtime
both use it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .geometry import pseudo_angle, unit_circle

Vec3 = tuple[float, float, float]
# Points this close to a plane (mm) lie on it: published corners are exact decimals, so their float images agree to ~1e-12.
ON_PLANE_TOLERANCE_MM = 1e-9
# Tessellation of a round plate's outline when it is drawn (display only).
ROUND_SEGMENTS = 48
# Three faces whose normals' triple product is this small do not meet in one corner.
SINGULAR_DETERMINANT = 1e-12


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale(a: Vec3, s: float) -> Vec3:
    return (a[0] * s, a[1] * s, a[2] * s)


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


@dataclass(frozen=True)
class Solid:
    """A convex body in floats: ``planes`` ``(outward unit normal, offset)`` with ``normal . x <= offset`` inside,
    ``side_radius`` (a round plate) or ``None``, and the polyhedron's ``vertices`` (empty for a round plate)."""

    medium: str
    planes: tuple[tuple[Vec3, float], ...]
    side_radius: float | None
    vertices: tuple[Vec3, ...]

    def contains(self, point: Vec3, tolerance: float) -> bool:
        if any(_dot(n, point) - d > tolerance for n, d in self.planes):
            return False
        return self.side_radius is None or math.hypot(point[0], point[1]) <= self.side_radius + tolerance


def solid_from_planes(medium: str, planes: tuple[tuple[Vec3, float], ...], side_radius: float | None) -> Solid:
    """The solid of ``planes`` (outward unit normal, offset) and ``side_radius``; a polyhedron's corners are the feasible
    intersections of three faces (deduplicated)."""

    vertices: list[Vec3] = []
    if side_radius is None:
        count = len(planes)
        for i in range(count):
            for j in range(i + 1, count):
                for k in range(j + 1, count):
                    (ni, di), (nj, dj), (nk, dk) = planes[i], planes[j], planes[k]
                    jk, ki, ij = _cross(nj, nk), _cross(nk, ni), _cross(ni, nj)
                    det = _dot(ni, jk)
                    if abs(det) <= SINGULAR_DETERMINANT:
                        continue
                    point = _scale(_add(_add(_scale(jk, di), _scale(ki, dj)), _scale(ij, dk)), 1.0 / det)
                    if all(_dot(n, point) - offset <= ON_PLANE_TOLERANCE_MM for n, offset in planes):
                        vertices.append(point)
    return Solid(medium, planes, side_radius, tuple(sorted(_unique(vertices))))


def _ordered(points: list[Vec3], normal: Vec3) -> list[Vec3]:
    """Coplanar points counter-clockwise about ``normal`` (seen from its tip)."""
    if len(points) < 3:
        return points
    count = float(len(points))
    centre = (sum(p[0] for p in points) / count, sum(p[1] for p in points) / count, sum(p[2] for p in points) / count)
    first = _sub(points[0], centre)
    second = _cross(normal, first)
    return sorted(points, key=lambda p: pseudo_angle(_dot(_sub(p, centre), first), _dot(_sub(p, centre), second)))


def _unique(points: list[Vec3]) -> list[Vec3]:
    out: list[Vec3] = []
    for p in points:
        if not any(max(abs(p[k] - q[k]) for k in range(3)) <= ON_PLANE_TOLERANCE_MM for q in out):
            out.append(p)
    return out


def _round_outline(solid: Solid, normal: Vec3, offset: float, segments: int) -> list[Vec3]:
    """Points of a plane ``normal . x = offset`` on the side cylinder of a round plate (the plane must not be parallel to Z)."""
    radius = solid.side_radius or 0.0
    points = []
    for c, s in unit_circle(segments):
        x, y = radius * c, radius * s
        points.append((x, y, (offset - normal[0] * x - normal[1] * y) / normal[2]))
    return points


def face_polygon(solid: Solid, index: int, *, segments: int = ROUND_SEGMENTS) -> list[Vec3]:
    """The corners of face ``index`` counter-clockwise seen from outside (a round plate's face: its outline on the side
    cylinder, ``segments`` points)."""
    normal, offset = solid.planes[index]
    if solid.side_radius is not None:
        return _ordered(_round_outline(solid, normal, offset, segments), normal)
    return _ordered([v for v in solid.vertices if abs(_dot(normal, v) - offset) <= ON_PLANE_TOLERANCE_MM], normal)


def _edges(solid: Solid) -> list[tuple[Vec3, Vec3]]:
    """Polyhedron edges: corner pairs that share at least two faces."""
    on = [frozenset(i for i, (n, d) in enumerate(solid.planes) if abs(_dot(n, v) - d) <= ON_PLANE_TOLERANCE_MM) for v in solid.vertices]
    count = len(solid.vertices)
    return [(solid.vertices[i], solid.vertices[j]) for i in range(count) for j in range(i + 1, count) if len(on[i] & on[j]) >= 2]


def section_polygon(solid: Solid, point: Vec3, normal: Vec3, *, segments: int = ROUND_SEGMENTS) -> list[Vec3]:
    """The polygon where the plane through ``point`` with unit ``normal`` cuts the solid, counter-clockwise about
    ``normal`` (empty when the plane misses it)."""
    offset = _dot(normal, point)
    if solid.side_radius is not None:
        if abs(normal[2]) <= ON_PLANE_TOLERANCE_MM:
            return []
        candidates = [p for p in _round_outline(solid, normal, offset, segments) if solid.contains(p, ON_PLANE_TOLERANCE_MM)]
        return _ordered(candidates, normal)
    cuts = []
    for a, b in _edges(solid):
        fa, fb = _dot(normal, a) - offset, _dot(normal, b) - offset
        if abs(fa) <= ON_PLANE_TOLERANCE_MM:
            cuts.append(a)
        if abs(fb) <= ON_PLANE_TOLERANCE_MM:
            cuts.append(b)
        if fa * fb < 0.0:
            cuts.append(_add(a, _scale(_sub(b, a), fa / (fa - fb))))
    return _ordered(_unique(cuts), normal)


def clip_segment(solid: Solid, start: Vec3, end: Vec3) -> tuple[float, float] | None:
    """Parameter interval ``[t0, t1]`` (``0 <= t0 < t1 <= 1``) of ``start + t (end - start)`` inside the solid, or None."""
    d = _sub(end, start)
    t0, t1 = 0.0, 1.0
    for normal, offset in solid.planes:
        rate = _dot(normal, d)
        gap = offset - _dot(normal, start)
        if rate == 0.0:
            if gap < 0.0:
                return None
            continue
        t = gap / rate
        if rate > 0.0:
            t1 = min(t1, t)
        else:
            t0 = max(t0, t)
        if t0 >= t1:
            return None
    if solid.side_radius is not None:
        a = d[0] * d[0] + d[1] * d[1]
        b = start[0] * d[0] + start[1] * d[1]
        c = start[0] * start[0] + start[1] * start[1] - solid.side_radius * solid.side_radius
        if a == 0.0:
            if c > 0.0:
                return None
        else:
            disc = b * b - a * c
            if disc < 0.0:
                return None
            root = math.sqrt(disc)
            t0, t1 = max(t0, (-b - root) / a), min(t1, (-b + root) / a)
            if t0 >= t1:
                return None
    return t0, t1
