"""Whether a closed facet surface bounds a solid the trace can follow (``solid_problem``).

``contracts.optics.mesh_problem`` checks a closed facet surface edge by edge: every edge on two facets that run it in
opposite directions, a positive enclosed volume. That admits surfaces whose sides do not hold: two shells one inside the
other that both face outward (inside the inner one the facets name the solid's medium on both sides of a face), a cavity
whose facets face away from it, facets that cross each other (two overlapping solids written as one surface, a face pushed
through another), a surface that touches itself at a corner. Light in such a solid meets a face whose sides do not match
the medium it is in.

``solid_problem`` finds these from the geometry: the surface is split into shells (facets connected through their
edges); no two facets cross or overlap (facets sharing an edge are neighbours; facets sharing one corner may meet there
only; others may touch at a point); the facets around every corner form one fan (the surface passes each corner once);
and each shell faces outward (positive enclosed volume) when an even number of other shells contain it -- a solid, or a
solid inside a cavity -- and inward when an odd number do -- a cavity. Several separate solids, hollow solids and solids
inside cavities pass.

Pure Python floats; distances are judged against ``CONTACT_FRACTION`` of the surface's extent.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from .facets import mesh_extent

#: Distances below this fraction of the surface's extent (the diagonal of its vertices' bounding box) count as contact.
CONTACT_FRACTION = 1e-9
#: Ray directions tried in turn for the inside test, until one passes clear of every edge and corner it meets.
_PROBE_DIRECTIONS = ((0.5224, 0.3361, 0.7838), (-0.4119, 0.7547, 0.5107), (0.6893, -0.5531, 0.4679), (0.2917, 0.8846, -0.3638),
                     (-0.7312, -0.2271, 0.6432))
#: Barycentric margin within which a probe ray counts as meeting an edge or corner of a facet (the next direction is tried).
_PROBE_EDGE = 1e-9
#: Below this length of the cross product of two facets' unit normals their planes are taken as parallel (compared in one plane).
_PARALLEL_PLANES = 1e-12
#: The most cells of the broad-phase grid along each axis (about the cube root of the facet count).
_GRID_CELLS = 64
#: A volume is six times the sum of the facets' signed tetrahedra about the origin.
_TETRAHEDRA_PER_PARALLELEPIPED = 6.0

Vec = tuple[float, float, float]


def _sub(a: Sequence[float], b: Sequence[float]) -> Vec:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Sequence[float], b: Sequence[float]) -> Vec:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


class _Union:
    def __init__(self) -> None:
        self.parent: dict[int, int] = {}

    def find(self, a: int) -> int:
        self.parent.setdefault(a, a)
        root = a
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[a] != root:
            self.parent[a], a = root, self.parent[a]
        return root

    def join(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def shells(triangles: Sequence[Sequence[int]]) -> list[list[int]]:
    """The facets in groups connected through shared edges, each ascending, ordered by their first facet."""
    union = _Union()
    by_edge: dict[tuple[int, int], int] = {}
    for k, t in enumerate(triangles):
        union.find(k)
        for i in range(3):
            a, b = t[i], t[(i + 1) % 3]
            edge = (min(a, b), max(a, b))
            if edge in by_edge:
                union.join(k, by_edge[edge])
            else:
                by_edge[edge] = k
    groups: dict[int, list[int]] = {}
    for k in range(len(triangles)):
        groups.setdefault(union.find(k), []).append(k)
    return sorted(groups.values(), key=lambda g: g[0])


def _pinched_corner(triangles: Sequence[Sequence[int]]) -> tuple[int, int] | None:
    """``(vertex, fans)`` of the first corner whose facets form more than one fan (connected through edges at it)."""
    links: dict[int, list[tuple[int, int]]] = {}
    for a, b, c in triangles:
        links.setdefault(a, []).append((b, c))
        links.setdefault(b, []).append((c, a))
        links.setdefault(c, []).append((a, b))
    for vertex in sorted(links):
        union = _Union()
        for x, y in links[vertex]:
            union.join(x, y)
        fans = len({union.find(x) for edge in links[vertex] for x in edge})
        if fans > 1:
            return vertex, fans
    return None


def _planes_apart(d: Sequence[float]) -> bool:
    return all(x > 0.0 for x in d) or all(x < 0.0 for x in d)


def _interval(points: Sequence[Vec], d: Sequence[float], line: Vec) -> tuple[float, float]:
    """The extent along ``line`` of the part of a triangle (corners ``points``, signed distances ``d`` to the other plane,
    zero within the contact distance) that lies in the other triangle's plane."""
    values = [_dot(points[i], line) for i in range(3) if d[i] == 0.0]
    for i in range(3):
        j = (i + 1) % 3
        if d[i] * d[j] < 0.0:
            s = d[i] / (d[i] - d[j])
            values.append(_dot(points[i], line) + s * (_dot(points[j], line) - _dot(points[i], line)))
    return min(values), max(values)


def _overlap_in_plane(p: Sequence[Vec], q: Sequence[Vec], normal: Vec, contact: float) -> bool:
    """Whether two coplanar triangles overlap by more than ``contact`` (no separating edge direction)."""
    for tri in (p, q):
        for i in range(3):
            edge = _sub(tri[(i + 1) % 3], tri[i])
            axis = _cross(normal, edge)
            length = math.sqrt(_dot(axis, axis))
            if length == 0.0:
                continue
            axis = (axis[0] / length, axis[1] / length, axis[2] / length)
            a = [_dot(v, axis) for v in p]
            b = [_dot(v, axis) for v in q]
            if min(max(a), max(b)) - max(min(a), min(b)) <= contact:
                return False
    return True


def _unit_normal(t: Sequence[Vec]) -> Vec:
    n = _cross(_sub(t[1], t[0]), _sub(t[2], t[0]))
    length = math.sqrt(_dot(n, n))
    return (n[0] / length, n[1] / length, n[2] / length)


def _sectors_overlap(p: Sequence[Vec], q: Sequence[Vec], corner: Vec, normal: Vec, contact: float) -> bool:
    """Whether two coplanar triangles with the one common corner ``corner`` overlap: an edge from the corner of one runs
    strictly inside the other's angle there (or the angles coincide)."""
    def arms(t: Sequence[Vec]) -> tuple[Vec, Vec]:
        others = [v for v in t if v != corner]
        a, b = _sub(others[0], corner), _sub(others[1], corner)
        return (a, b) if _dot(_cross(a, b), normal) > 0.0 else (b, a)  # (counter-clockwise about the normal)

    def length(x: Vec) -> float:
        return math.sqrt(_dot(x, x))

    def inside(x: Vec, arm: tuple[Vec, Vec]) -> bool:
        # (x's tip more than the contact distance on the inner side of both arms' lines)
        return (_dot(_cross(arm[0], x), normal) > contact * length(arm[0]) and _dot(_cross(x, arm[1]), normal) > contact * length(arm[1]))

    pa, qa = arms(p), arms(q)
    if any(inside(x, qa) for x in pa) or any(inside(x, pa) for x in qa):
        return True
    # (the same angle: each arm along the other's)
    return all(length(_cross(x, y)) <= contact * max(length(x), length(y)) and _dot(x, y) > 0.0 for x, y in zip(pa, qa))


def facets_cross(p: Sequence[Vec], q: Sequence[Vec], contact: float, normals: tuple[Vec, Vec] | None = None,
                 corner: Vec | None = None) -> bool:
    """Whether two triangles (corners ``p``, ``q``; ``normals``: their unit normals if known; ``corner``: their one common
    corner, if they share one) cross or overlap: their common part is more than a point (longer than ``contact``)."""
    n1, n2 = normals if normals is not None else (_unit_normal(p), _unit_normal(q))
    o1 = _dot(n1, p[0])
    dq = [x if abs(x) > contact else 0.0 for x in (_dot(n1, v) - o1 for v in q)]
    if _planes_apart(dq):
        return False
    o2 = _dot(n2, q[0])
    dp = [x if abs(x) > contact else 0.0 for x in (_dot(n2, v) - o2 for v in p)]
    if _planes_apart(dp):
        return False
    line = _cross(n1, n2)
    length = math.sqrt(_dot(line, line))
    if all(x == 0.0 for x in dq) or length < _PARALLEL_PLANES:
        if corner is not None:
            return _sectors_overlap(p, q, corner, n1, contact)
        return _overlap_in_plane(p, q, n1, contact)
    line = (line[0] / length, line[1] / length, line[2] / length)
    a_lo, a_hi = _interval(p, dp, line)
    b_lo, b_hi = _interval(q, dq, line)
    return min(a_hi, b_hi) - max(a_lo, b_lo) > contact


def _candidate_pairs(boxes: Sequence[tuple[Vec, Vec]], low: Vec, high: Vec):
    """Pairs of facets whose boxes share a cell of a uniform grid over the surface (each pair once)."""
    count = len(boxes)
    cells_per_axis = max(1, min(_GRID_CELLS, round(count ** (1.0 / 3.0))))
    size = [(high[k] - low[k]) / cells_per_axis for k in range(3)]
    grid: dict[tuple[int, int, int], list[int]] = {}
    for k, (lo, hi) in enumerate(boxes):
        span = [(0, 0) if size[a] == 0.0 else (min(cells_per_axis - 1, int((lo[a] - low[a]) / size[a])),
                                                min(cells_per_axis - 1, int((hi[a] - low[a]) / size[a]))) for a in range(3)]
        for i in range(span[0][0], span[0][1] + 1):
            for j in range(span[1][0], span[1][1] + 1):
                for m in range(span[2][0], span[2][1] + 1):
                    grid.setdefault((i, j, m), []).append(k)
    seen: set[tuple[int, int]] = set()
    for members in grid.values():
        for x in range(len(members)):
            a = members[x]
            for y in range(x + 1, len(members)):
                b = members[y]
                pair = (a, b) if a < b else (b, a)
                if pair in seen:
                    continue
                seen.add(pair)
                (alo, ahi), (blo, bhi) = boxes[a], boxes[b]
                if all(alo[r] <= bhi[r] and blo[r] <= ahi[r] for r in range(3)):
                    yield pair


def _crossing(vertices: Sequence[Vec], triangles: Sequence[Sequence[int]], contact: float, face_of: dict[int, int]
              ) -> tuple[int, int] | None:
    """The first pair of facets (in ascending order) that cross or overlap (``facets_cross``); facets of one face of
    ``face_of`` tile it and are not compared."""
    corners = [[vertices[i] for i in t] for t in triangles]
    normals = [_unit_normal(c) for c in corners]
    boxes = []
    for pts in corners:
        boxes.append((tuple(min(p[a] for p in pts) - contact for a in range(3)), tuple(max(p[a] for p in pts) + contact for a in range(3))))
    low = tuple(min(b[0][a] for b in boxes) for a in range(3))
    high = tuple(max(b[1][a] for b in boxes) for a in range(3))
    found: list[tuple[int, int]] = []
    for a, b in _candidate_pairs(boxes, low, high):  # type: ignore[arg-type]
        shared = set(triangles[a]) & set(triangles[b])
        if len(shared) >= 2:
            continue  # neighbours through an edge
        if a in face_of and face_of.get(b) == face_of[a]:
            continue
        corner = vertices[next(iter(shared))] if shared else None
        if facets_cross(corners[a], corners[b], contact, (normals[a], normals[b]), corner):
            found.append((a, b))
    return min(found) if found else None


def _inside(point: Vec, vertices: Sequence[Vec], triangles: Sequence[Sequence[int]], members: Sequence[int]) -> bool | None:
    """Whether ``point`` lies inside the closed shell ``members`` (crossings of a ray from it, counted; ``None``: every probe
    direction met an edge or corner)."""
    for direction in _PROBE_DIRECTIONS:
        crossings, clear = 0, True
        for k in members:
            a, b, c = (vertices[i] for i in triangles[k])
            e1, e2 = _sub(b, a), _sub(c, a)
            h = _cross(direction, e2)
            det = _dot(e1, h)
            if det == 0.0:
                continue
            s = _sub(point, a)
            u = _dot(s, h) / det
            qv = _cross(s, e1)
            v = _dot(direction, qv) / det
            t = _dot(e2, qv) / det
            if t <= 0.0 or u < -_PROBE_EDGE or v < -_PROBE_EDGE or u + v > 1.0 + _PROBE_EDGE:
                continue
            if u < _PROBE_EDGE or v < _PROBE_EDGE or u + v > 1.0 - _PROBE_EDGE:
                clear = False
                break
            crossings += 1
        if clear:
            return crossings % 2 == 1
    return None  # pragma: no cover - five generic directions all grazing an edge


def _volume(vertices: Sequence[Vec], triangles: Sequence[Sequence[int]], members: Sequence[int]) -> float:
    total = 0.0
    for k in members:
        a, b, c = (vertices[i] for i in triangles[k])
        total += _dot(a, _cross(_sub(b, a), _sub(c, a)))
    return total / _TETRAHEDRA_PER_PARALLELEPIPED


def _named(members: Sequence[int]) -> str:
    return f"facets {members[0]} .. {members[-1]} ({len(members)} facets)" if len(members) > 1 else f"facet {members[0]}"


def solid_problem(vertices: Sequence[Sequence[float]], triangles: Sequence[Sequence[int]],
                  faces: Sequence[Sequence[int]] = ()) -> str | None:
    """Why a closed facet surface (``mesh_problem`` holds for it) does not bound a solid the trace can follow, or ``None``
    (module docstring). ``faces``: groups of facets known to tile one simple polygon, facing one way (the planar patches of
    more than one facet with an outline, ``facets.planar_patches``): the facets of one such face do not overlap (each
    point inside the outline is covered as often as the outline winds around it: once), so they are not compared."""
    points = [(float(v[0]), float(v[1]), float(v[2])) for v in vertices]
    contact = CONTACT_FRACTION * mesh_extent(points)
    pinch = _pinched_corner(triangles)
    if pinch is not None:
        vertex, fans = pinch
        return (f"the facets touch themselves at vertex {vertex} (its facets form {fans} separate fans): a solid's surface passes "
                "each corner once")
    face_of = {k: f for f, members in enumerate(faces) if len(members) > 1 for k in members}
    crossing = _crossing(points, triangles, contact, face_of)
    if crossing is not None:
        a, b = crossing
        return (f"facets {a} and {b} cross or overlap each other: the facets of a solid do not intersect (overlapping solids are "
                "separate parts, or one surface around their union)")
    groups = shells(triangles)
    if len(groups) == 1:
        return None
    for g, members in enumerate(groups):
        a, b, c = (points[i] for i in triangles[members[0]])
        probe = ((a[0] + b[0] + c[0]) / 3.0, (a[1] + b[1] + c[1]) / 3.0, (a[2] + b[2] + c[2]) / 3.0)
        around = [h for h, other in enumerate(groups) if h != g and _inside(probe, points, triangles, other)]
        outward = _volume(points, triangles, members) > 0.0
        if outward != (len(around) % 2 == 0):
            where = ("inside no other shell" if not around
                     else f"inside {len(around)} other shell{'s' if len(around) > 1 else ''} (the first: {_named(groups[around[0]])})")
            return (f"the shell of {_named(members)} faces {'outward' if outward else 'inward'} and lies {where}: the outer surface "
                    "of a solid faces outward, the surface of a cavity in it faces into the cavity (list its triangles "
                    "counter-clockwise seen from the cavity), a solid inside a cavity faces outward again")
    return None


__all__ = ["CONTACT_FRACTION", "facets_cross", "shells", "solid_problem"]
