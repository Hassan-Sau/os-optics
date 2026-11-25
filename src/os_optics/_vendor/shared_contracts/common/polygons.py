"""Simple-polygon checks shared by the aperture contracts (pure arithmetic: exact on ``Decimal``, on floats within rounding).

A polygon is a closed ring of corners ``(x, y)`` (the last corner joins the first, which is not repeated). It is *simple*
when no corner repeats, no edge has zero length and no two edges meet except neighbours at their shared corner; its corners
are *counter-clockwise* when its signed area (shoelace) is positive.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

#: The most corners a polygon aperture or outline may have.
MAX_POLYGON_VERTICES = 64


def signed_area_twice(points: Sequence[tuple[Any, Any]]) -> Any:
    """Twice the signed area (positive: counter-clockwise)."""

    n = len(points)
    total = points[0][0] * 0
    for i in range(n):
        (x1, y1), (x2, y2) = points[i], points[(i + 1) % n]
        total += x1 * y2 - x2 * y1
    return total


def _orientation(a: tuple[Any, Any], b: tuple[Any, Any], c: tuple[Any, Any]) -> int:
    value = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    return (value > 0) - (value < 0)


def _on_segment(a: tuple[Any, Any], b: tuple[Any, Any], p: tuple[Any, Any]) -> bool:
    return min(a[0], b[0]) <= p[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= p[1] <= max(a[1], b[1])


def _segments_meet(a: tuple[Any, Any], b: tuple[Any, Any], c: tuple[Any, Any], d: tuple[Any, Any]) -> bool:
    o1, o2, o3, o4 = _orientation(a, b, c), _orientation(a, b, d), _orientation(c, d, a), _orientation(c, d, b)
    if o1 != o2 and o3 != o4:
        return True
    return ((o1 == 0 and _on_segment(a, b, c)) or (o2 == 0 and _on_segment(a, b, d))
            or (o3 == 0 and _on_segment(c, d, a)) or (o4 == 0 and _on_segment(c, d, b)))


def polygon_problem(points: Sequence[tuple[Any, Any]]) -> str | None:
    """Why ``points`` is not a simple counter-clockwise polygon of at most ``MAX_POLYGON_VERTICES`` corners (``None``: it
    is one)."""

    n = len(points)
    if n < 3:
        return "a polygon needs at least 3 corners"
    if n > MAX_POLYGON_VERTICES:
        return f"a polygon has at most {MAX_POLYGON_VERTICES} corners, got {n}"
    if len(set(points)) != n:
        return "a polygon's corners are distinct (do not repeat the first corner at the end)"
    for i in range(n):
        for j in range(i + 1, n):
            if j == i + 1 or (i == 0 and j == n - 1):
                continue  # neighbours share a corner
            if _segments_meet(points[i], points[(i + 1) % n], points[j], points[(j + 1) % n]):
                return f"a polygon's edges do not cross or touch: edges {i} and {j} meet"
        a, b, c = points[i - 1], points[i], points[(i + 1) % n]
        if _orientation(a, b, c) == 0 and not (min(a[0], c[0]) <= b[0] <= max(a[0], c[0]) and min(a[1], c[1]) <= b[1] <= max(a[1], c[1])):
            return f"a polygon's edges do not fold back on themselves (at corner {i})"
    if not signed_area_twice(points) > 0:
        return "a polygon's corners run counter-clockwise (positive area): reverse their order"
    return None


__all__ = ["MAX_POLYGON_VERTICES", "polygon_problem", "signed_area_twice"]
