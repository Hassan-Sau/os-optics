"""Triangle-mesh checks shared by the facet contracts (pure float arithmetic: correctly rounded IEEE operations, the same
answer on every runtime).

A mesh is a list of corners ``(x, y, z)`` and of triangles (corner index triples, counter-clockwise seen from the side the
mesh faces). It is a valid facet surface when every index names a corner, no triangle is degenerate (its doubled area is
more than ``DEGENERATE_AREA_FRACTION`` of the square of its longest edge), every edge belongs to at most two triangles that
run it in opposite directions (a manifold sheet of one orientation) and, for a closed mesh (one bounding a solid), every
edge belongs to exactly two triangles and the enclosed volume is positive (the triangles face outward).
"""

from __future__ import annotations

from collections.abc import Sequence

#: The most corners and triangles of one facet mesh.
MAX_MESH_VERTICES = 20000
MAX_MESH_TRIANGLES = 40000
#: A triangle whose doubled area is at most this fraction of the square of its longest edge is degenerate (a sliver, a point).
DEGENERATE_AREA_FRACTION = 1e-12


def mesh_problem(vertices: Sequence[Sequence[float]], triangles: Sequence[Sequence[int]], closed: bool) -> str | None:
    """Why ``(vertices, triangles)`` is not a valid facet mesh (``None``: it is one; module docstring)."""

    count = len(vertices)
    directed: dict[tuple[int, int], int] = {}
    volume = 0.0
    for k, (a, b, c) in enumerate(triangles):
        if not (0 <= a < count and 0 <= b < count and 0 <= c < count):
            return f"triangle {k} names a vertex outside 0 .. {count - 1}"
        if a == b or b == c or a == c:
            return f"triangle {k} repeats a vertex"
        pa, pb, pc = vertices[a], vertices[b], vertices[c]
        u = (pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2])
        v = (pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2])
        w = (pc[0] - pb[0], pc[1] - pb[1], pc[2] - pb[2])
        cross = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
        area2 = cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]
        longest = max(u[0] * u[0] + u[1] * u[1] + u[2] * u[2], v[0] * v[0] + v[1] * v[1] + v[2] * v[2], w[0] * w[0] + w[1] * w[1] + w[2] * w[2])
        if area2 <= (DEGENERATE_AREA_FRACTION * longest) ** 2:
            return f"triangle {k} is degenerate (no area)"
        for edge in ((a, b), (b, c), (c, a)):
            if edge in directed:
                return (f"edge {edge[0]}-{edge[1]} is run in the same direction by triangles {directed[edge]} and {k}: the facets are not "
                        "oriented consistently, or more than two facets share the edge")
            directed[edge] = k
        volume += pa[0] * cross[0] + pa[1] * cross[1] + pa[2] * cross[2]
    for a, b in directed:
        if closed and (b, a) not in directed:
            return f"edge {a}-{b} belongs to one facet only: a closed facet surface has no open edge"
    if closed and not volume > 0.0:
        return "the facets of a closed surface face inward (the enclosed volume is not positive): list each triangle counter-clockwise seen from outside"
    return None


__all__ = ["DEGENERATE_AREA_FRACTION", "MAX_MESH_TRIANGLES", "MAX_MESH_VERTICES", "mesh_problem"]
