"""Display meshes of a surface whose drawn regions are not star-shaped about its centre: a clear aperture with holes through
the element, or a non-convex clear outline (presentation only; element frame, mm).

The regions are polygons in the surface's local X / Y (the outline rings the scene states, round parts already chords within
the view's drawing tolerance): the clear aperture (its outline minus its holes), drawn in the surface's material, and the
opaque rim between the clear aperture and the outline of the glass (``optic_rim``), whose outer ring is the outline row the
edge walls meet. Each region is triangulated by ear clipping, its holes first joined to its outer ring by bridges
(D. Eberly, *Triangulation by Ear Clipping*, Geometric Tools, 2002), then a triangle farther than the chord tolerance from
a curved profile (``surface_mesh.max_deviation``'s rule) is bisected at its longest edge, with its neighbour across that edge
(no crack), until every triangle lies within the tolerance or the vertex budget is spent. The holes are open: nothing is drawn inside them. Pure Python.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence

from . import geometry as g
from . import surface_mesh as sm

Vec2 = tuple[float, float]
#: Two ring points closer than this are one point (mm).
SAME_POINT_MM = 1e-12


def _area2(ring: Sequence[Vec2]) -> float:
    return sum(ring[i - 1][0] * ring[i][1] - ring[i][0] * ring[i - 1][1] for i in range(len(ring)))


def _oriented(ring: Sequence[Vec2], ccw: bool) -> list[Vec2]:
    out: list[Vec2] = []
    for p in ring:
        if not out or abs(p[0] - out[-1][0]) > SAME_POINT_MM or abs(p[1] - out[-1][1]) > SAME_POINT_MM:
            out.append((float(p[0]), float(p[1])))
    if len(out) > 1 and abs(out[0][0] - out[-1][0]) <= SAME_POINT_MM and abs(out[0][1] - out[-1][1]) <= SAME_POINT_MM:
        out.pop()
    return out if (_area2(out) > 0.0) == ccw else out[::-1]


def _cross(o: Vec2, a: Vec2, b: Vec2) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _inside_triangle(p: Vec2, a: Vec2, b: Vec2, c: Vec2) -> bool:
    return _cross(a, b, p) >= 0.0 and _cross(b, c, p) >= 0.0 and _cross(c, a, p) >= 0.0


def _bridge(outer: list[Vec2], hole: list[Vec2]) -> list[Vec2]:
    """``outer`` (counter-clockwise) with ``hole`` (clockwise, inside it) joined by a bridge from the hole's rightmost point to a
    point of ``outer`` it sees: the ray toward +X from the hole point, its nearest crossing of ``outer``, and the visible
    reflex point of least angle inside the triangle it spans."""

    m_index = max(range(len(hole)), key=lambda k: (hole[k][0], -hole[k][1]))
    m = hole[m_index]
    best_t, best_edge = float("inf"), -1
    for i in range(len(outer)):
        a, b = outer[i], outer[(i + 1) % len(outer)]
        if (a[1] - m[1]) * (b[1] - m[1]) > 0.0 or a[1] == b[1]:
            continue
        t = a[0] + (m[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1]) - m[0]
        if 0.0 <= t < best_t:
            best_t, best_edge = t, i
    if best_edge < 0:  # pragma: no cover - a hole inside the outer ring always has a crossing to its right
        raise ValueError("a hole lies outside its outline")
    hit = (m[0] + best_t, m[1])
    a, b = outer[best_edge], outer[(best_edge + 1) % len(outer)]
    p_index = best_edge if a[0] > b[0] else (best_edge + 1) % len(outer)
    p = outer[p_index]
    tri = (m, hit, p) if _cross(m, hit, p) >= 0.0 else (m, p, hit)
    best_angle = None
    for k, q in enumerate(outer):
        if k == p_index:
            continue
        prev, nxt = outer[k - 1], outer[(k + 1) % len(outer)]
        reflex = _cross(prev, q, nxt) < 0.0
        if reflex and _inside_triangle(q, *tri):
            dx, dy = q[0] - m[0], q[1] - m[1]
            angle = abs(dy) / max(abs(dx) + abs(dy), SAME_POINT_MM)
            if best_angle is None or angle < best_angle:
                best_angle, p_index = angle, k
    joined = outer[:p_index + 1] + hole[m_index:] + hole[:m_index + 1] + outer[p_index:]
    return joined


def triangulate(outer: Sequence[Vec2], holes: Sequence[Sequence[Vec2]] = ()) -> tuple[list[Vec2], list[tuple[int, int, int]]]:
    """Triangles (counter-clockwise, indices into the returned points) of the polygon ``outer`` minus ``holes``."""

    ring = _oriented(outer, True)
    for hole in sorted((_oriented(h, False) for h in holes), key=lambda h: -max(p[0] for p in h)):
        ring = _bridge(ring, hole)
    points = list(ring)
    index = list(range(len(points)))
    triangles: list[tuple[int, int, int]] = []
    guard = 0
    while len(index) > 3 and guard < 4 * len(points) * len(points):
        guard += 1
        n = len(index)
        clipped = False
        for k in range(n):
            i0, i1, i2 = index[k - 1], index[k], index[(k + 1) % n]
            a, b, c = points[i0], points[i1], points[i2]
            if _cross(a, b, c) <= 0.0:
                continue
            if any(_inside_triangle(points[j], a, b, c) for j in index if j not in (i0, i1, i2) and points[j] not in (a, b, c)):
                continue
            triangles.append((i0, i1, i2))
            del index[k]
            clipped = True
            break
        if not clipped:  # (a degenerate remainder: collinear points; drop the flattest corner)
            k = min(range(n), key=lambda k: abs(_cross(points[index[k - 1]], points[index[k]], points[index[(k + 1) % n]])))
            del index[k]
    if len(index) == 3 and _cross(points[index[0]], points[index[1]], points[index[2]]) > 0.0:
        triangles.append((index[0], index[1], index[2]))
    return points, triangles


def subdivided(points: list[Vec2], triangles: list[tuple[int, int, int]]) -> tuple[list[Vec2], list[tuple[int, int, int]]]:
    """Every triangle split into four at its edge midpoints (shared edges share their midpoint: no cracks)."""

    points = list(points)
    middle: dict[tuple[int, int], int] = {}

    def mid(i: int, j: int) -> int:
        key = (i, j) if i < j else (j, i)
        k = middle.get(key)
        if k is None:
            k = middle[key] = len(points)
            points.append(((points[i][0] + points[j][0]) / 2.0, (points[i][1] + points[j][1]) / 2.0))
        return k

    out: list[tuple[int, int, int]] = []
    for a, b, c in triangles:
        ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
        out.extend(((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)))
    return points, out


def _deviation(profile: sm.Profile, points: list[Vec2], tri: tuple[int, int, int]) -> float:
    """The largest first-order normal distance between the flat triangle and the profile at its centroid and edge midpoints
    (``surface_mesh.max_deviation``'s rule); ``inf`` where the profile is undefined there."""

    corners = [points[k] for k in tri]
    sags = []
    for a, b in corners:
        hit = profile.sag_normal(a, b)
        if hit is None:
            return float("inf")
        sags.append(hit[0])
    worst = 0.0
    for (wa, wb, wc) in ((1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0), (0.5, 0.5, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5)):
        x = wa * corners[0][0] + wb * corners[1][0] + wc * corners[2][0]
        y = wa * corners[0][1] + wb * corners[1][1] + wc * corners[2][1]
        hit = profile.sag_normal(x, y)
        if hit is None:
            return float("inf")
        worst = max(worst, abs(wa * sags[0] + wb * sags[1] + wc * sags[2] - hit[0]) * hit[3])
    return worst


def refined(profile: sm.Profile, points: list[Vec2], triangles: list[tuple[int, int, int]], tolerance: float, max_points: int
            ) -> tuple[list[Vec2], list[tuple[int, int, int]]]:
    """Triangles refined by longest-edge bisection along the longest-edge propagation path (M.-C. Rivara, Int. J. Numer.
    Meth. Eng. 40 (1997) 3313): a triangle beyond ``tolerance`` of the profile is bisected with its neighbour across their
    common longest edge (walking to the pair whose longest edges agree first), so no vertex lies inside another triangle's
    edge and the triangles keep their shape, until each lies within ``tolerance`` or ``max_points`` points are used."""

    points = list(points)
    tris: dict[int, tuple[int, int, int]] = {}
    by_edge: dict[tuple[int, int], set[int]] = {}

    def edge_key(i: int, j: int) -> tuple[int, int]:
        return (i, j) if i < j else (j, i)

    def add(key: int, tri: tuple[int, int, int]) -> None:
        tris[key] = tri
        for i, j in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            by_edge.setdefault(edge_key(i, j), set()).add(key)

    def drop(key: int) -> tuple[int, int, int]:
        tri = tris.pop(key)
        for i, j in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            by_edge[edge_key(i, j)].discard(key)
        return tri

    def longest(tri: tuple[int, int, int]) -> tuple[int, int]:
        edges = ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0]))
        i, j = max(edges, key=lambda e: ((points[e[0]][0] - points[e[1]][0]) ** 2 + (points[e[0]][1] - points[e[1]][1]) ** 2,
                                         -min(e), -max(e)))
        return edge_key(i, j)

    def neighbour(key: int, edge: tuple[int, int]) -> int | None:
        others = by_edge.get(edge, set()) - {key}
        return next(iter(others)) if others else None

    counter = 0

    def bisect(edge: tuple[int, int]) -> list[int]:
        nonlocal counter
        i, j = edge
        m = len(points)
        points.append(((points[i][0] + points[j][0]) / 2.0, (points[i][1] + points[j][1]) / 2.0))
        made = []
        for other in list(by_edge.get(edge, ())):
            a, b, c = drop(other)
            first, second, apex = (a, b, c) if edge_key(a, b) == edge else (b, c, a) if edge_key(b, c) == edge else (c, a, b)
            for half in ((first, m, apex), (m, second, apex)):
                counter += 1
                add(counter, half)
                made.append(counter)
        return made

    for tri in triangles:
        counter += 1
        add(counter, tri)
    pending = deque(k for k, tri in tris.items() if _deviation(profile, points, tri) > tolerance)
    while pending and len(points) < max_points:
        key = pending.popleft()
        while key in tris and len(points) < max_points:
            walk = key
            for _ in range(len(tris)):
                edge = longest(tris[walk])
                other = neighbour(walk, edge)
                if other is None or longest(tris[other]) == edge:
                    break
                walk = other
            for made in bisect(edge):
                if _deviation(profile, points, tris[made]) > tolerance:
                    pending.append(made)
    return points, list(tris.values())


def lifted(profile: sm.Profile, placement: sm.Placement, points: list[Vec2], triangles: list[tuple[int, int, int]], facing: float
           ) -> g.Mesh | None:
    """The triangles on the profile (element frame), wound counter-clockwise seen from ``(0, 0, facing)``; ``None`` where the
    profile is undefined at a vertex (the caller draws the surface its other way)."""

    mesh = g.Mesh()
    for a, b in points:
        hit = profile.sag_normal(a, b)
        if hit is None:
            return None
        z, nx, ny, nz = hit
        mesh.positions.append(placement.point(a, b, z))
        mesh.normals.append(placement.vector(nx * facing, ny * facing, nz * facing))
    for tri in triangles:
        mesh.indices.extend(tri if facing > 0 else (tri[0], tri[2], tri[1]))
    return mesh


def region_meshes(profile: sm.Profile, placement: sm.Placement, regions: Sequence[tuple[str, Sequence[Vec2], Sequence[Sequence[Vec2]]]],
                  facing: float, tolerance: float, max_vertices: int) -> tuple[list[tuple[str, g.Mesh]], float] | None:
    """Each region ``(material, outer ring, hole rings)`` triangulated, split until within ``tolerance`` of the profile (a flat
    profile is exact as triangulated) or until the next split would pass ``max_vertices``, and lifted. Returns the meshes
    and the largest deviation reached, or ``None`` where the profile is undefined."""

    flat = [(material, *triangulate(outer, holes)) for material, outer, holes in regions]
    if not profile.is_flat:
        share = max_vertices // max(1, len(flat))
        flat = [(material, *refined(profile, points, triangles, tolerance, share)) for material, points, triangles in flat]
    meshes = []
    for material, points, triangles in flat:
        mesh = lifted(profile, placement, points, triangles, facing)
        if mesh is None:
            return None
        meshes.append((material, mesh))
    deviation = 0.0 if profile.is_flat else sm.max_deviation(meshes, profile, placement)
    return meshes, deviation


__all__ = ["refined", "region_meshes", "subdivided", "triangulate"]
