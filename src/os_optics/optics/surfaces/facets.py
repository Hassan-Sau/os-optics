"""Triangle-facet surfaces as planar patches (``contracts.optics.MeshSurface``).

A facet surface is traced as the planes it is made of: its triangles are grouped into *patches*, each a set of
edge-connected triangles of one facet group that lie in one plane, and each patch becomes one plane surface whose clear
aperture is the patch's outline (a simple counter-clockwise polygon in the plane's ``(u, v)`` frame, seen from the
positive side).

Which triangles lie in one plane is judged against the precision the coordinates are given to (``coplanar_tolerance``):
coordinates rounded to ``D`` decimals, or to single precision (float32, as binary STL stores them), place a vertex within
``q`` of its true position along each axis (``q``: half a unit of the last decimal, or half a float32 spacing of the
largest coordinate), so the corners of one flat face lie within ``sqrt(3) q`` of its true plane. A triangle joins a patch
when its corners lie within ``COPLANAR_QUANTA * q`` of the patch's plane (at least ``COPLANAR_FLOOR_MM``, at most
``COPLANAR_MAX_FRACTION`` of the mesh's extent). The plane of a patch of several triangles is fitted to its corners
(normal: the sum of its triangles' area vectors; offset: midway between the farthest corners on either side), and every
corner lies within the patch's ``deviation_mm`` of it (at most the tolerance): the geometric error of tracing the patch as
that plane. A patch whose corners all lie within ``COPLANAR_FLOOR_MM`` of its first triangle's plane keeps that plane
(exact data: the plane through its corners).

Where fitted planes meet, the outline corners are moved onto them (each mesh vertex of an outline to the point nearest it
on the planes of every patch whose outline it is on, in the least-squares sense): three planes meet at one point and two
along one line, so neighbouring patches share their edges exactly and no ray passes between them; at a vertex where four
or more fitted planes meet, the outlines meet within the planes' deviations.

A patch whose outline is not one simple polygon (a patch around a hole, a patch touching itself at a corner), or has more
than ``MAX_FACET_OUTLINE_CORNERS`` corners, is traced triangle by triangle. Every outline corner is a mesh vertex (moved
onto the planes it lies on).

Pure Python floats (correctly rounded IEEE operations, no fused multiply-add): the same patches, planes and outlines in
every runtime.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass

from ..._vendor.shared_contracts.common.polygons import MAX_POLYGON_VERTICES, signed_area_twice

Vec = tuple[float, float, float]

#: The least coplanarity tolerance (mm): coordinates given to full double precision (the arithmetic's own rounding on
#: part-sized coordinates is far below it, and it is below the aperture edge tolerance of 1e-9 mm).
COPLANAR_FLOOR_MM = 1e-10
#: Kept for callers of the former fixed tolerance: the floor.
COPLANAR_TOL_MM = COPLANAR_FLOOR_MM
#: The coplanarity tolerance in units of the coordinates' rounding ``q``: rounding puts the corners of one flat face within
#: ``sqrt(3) q`` of its plane, and a plane fitted to them within twice that (``4 sqrt(3) ~ 6.93``).
COPLANAR_QUANTA = 4.0 * math.sqrt(3.0)
#: The coplanarity tolerance is at most this fraction of the mesh's extent (the diagonal of its bounding box): coarsely
#: rounded coordinates never merge faces that differ by more.
COPLANAR_MAX_FRACTION = 1e-4
#: A face fitted to facets whose corners lie farther from its plane than this fraction of the mesh's extent is stated in a
#: notice (``FACET_FACE_FITTED``): a bend of the part that small is traced flat.
FITTED_FACE_NOTICE_FRACTION = 1e-5
#: The most corners a patch's outline keeps (more: the patch is traced triangle by triangle).
MAX_FACET_OUTLINE_CORNERS = 1024
#: The base of the decimals coordinates are written in, and the significand bits of a single-precision (float32) number.
_DECIMAL_BASE = 10.0
_FLOAT32_BITS = 24
#: Below this weight a direction of the planes meeting at a vertex is taken as undetermined (two nearly parallel planes):
#: the vertex stays where it is along it.
_SNAP_RIDGE = 1e-9


@dataclass(frozen=True)
class Patch:
    """One planar patch: its triangles (mesh indices, ascending), its plane (``origin``: its outline's first corner, on the
    plane; ``normal``: unit, toward the positive side, ``u_axis``: unit, in the plane; ``v = normal x u``), its outline:
    the mesh vertices ``loop`` in order and their ``(u, v)`` coordinates ``polygon`` (counter-clockwise), and
    ``deviation_mm``: the farthest any of its triangles' corners lies from the plane (0 for one triangle)."""

    triangles: tuple[int, ...]
    group: int
    origin: Vec
    normal: Vec
    u_axis: Vec
    loop: tuple[int, ...]
    polygon: tuple[tuple[float, float], ...]
    deviation_mm: float = 0.0

    @property
    def first(self) -> int:
        return self.triangles[0]


def _sub(a: Sequence[float], b: Sequence[float]) -> Vec:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Sequence[float], b: Sequence[float]) -> Vec:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _unit(a: Vec) -> Vec:
    length = math.sqrt(_dot(a, a))
    return (a[0] / length, a[1] / length, a[2] / length)


def triangle_normal(vertices: Sequence[Sequence[float]], triangle: Sequence[int]) -> Vec:
    """Unit normal of a triangle, toward the side its corners run counter-clockwise from."""
    a, b, c = (vertices[i] for i in triangle)
    return _unit(_cross(_sub(b, a), _sub(c, a)))


# ------------------------------------------------------------------------------------------------ tolerance


def _decimals(x: float) -> int:
    """The decimals of the shortest representation of ``x`` (``0.25`` 2, ``10.0`` 0, ``1.5e-07`` 8)."""
    text = repr(float(x)).lower()
    mantissa, _, exponent = text.partition("e")
    digits = mantissa.partition(".")[2].rstrip("0")
    return max(0, len(digits) - (int(exponent) if exponent else 0))


def _float32_exact(x: float) -> bool:
    try:
        return struct.unpack("<f", struct.pack("<f", x))[0] == x
    except (OverflowError, struct.error):  # pragma: no cover - mesh coordinates are part-sized
        return False


def coordinate_quantum(vertices: Sequence[Sequence[float]]) -> float:
    """``q``, the rounding of the coordinates as given (mm): half a unit of the last decimal any coordinate uses, or, where
    every coordinate is a single-precision (float32) number, half the float32 spacing at the largest coordinate when that
    is coarser. Coordinates given to full double precision: below 1e-15 mm."""
    values = [float(x) for v in vertices for x in v]
    quantum = 0.5 * _DECIMAL_BASE ** (-max(_decimals(x) for x in values))
    if all(_float32_exact(x) for x in values):
        largest = max(abs(x) for x in values)
        if largest > 0.0:
            # (half the float32 spacing 2**(e - 24) of a number of exponent e)
            quantum = max(quantum, math.ldexp(0.5, math.frexp(largest)[1] - _FLOAT32_BITS))
    return quantum


def mesh_extent(vertices: Sequence[Sequence[float]]) -> float:
    """The diagonal (mm) of the vertices' bounding box."""
    lows = [min(float(v[k]) for v in vertices) for k in range(3)]
    highs = [max(float(v[k]) for v in vertices) for k in range(3)]
    span = [h - lo for lo, h in zip(lows, highs)]
    return math.sqrt(span[0] * span[0] + span[1] * span[1] + span[2] * span[2])


def coplanar_tolerance(vertices: Sequence[Sequence[float]]) -> float:
    """The distance (mm) within which the corners of triangles count as lying in one plane (module docstring):
    ``COPLANAR_QUANTA * coordinate_quantum``, at least ``COPLANAR_FLOOR_MM`` and at most ``COPLANAR_MAX_FRACTION`` of the
    diagonal of the vertices' bounding box."""
    extent = mesh_extent(vertices)
    return max(COPLANAR_FLOOR_MM, min(COPLANAR_QUANTA * coordinate_quantum(vertices), COPLANAR_MAX_FRACTION * extent))


# ------------------------------------------------------------------------------------------------ outlines


def _frame(points: Sequence[Sequence[float]], normal: Vec) -> tuple[Vec, Vec, tuple[tuple[float, float], ...]]:
    """``(origin, u_axis, polygon)`` of an outline's corners ``points`` in the plane of ``normal`` through the first: origin
    the first corner, u along the first edge (in the plane)."""
    origin = tuple(float(x) for x in points[0])
    edge = _sub(points[1], origin)
    along = _dot(edge, normal)
    u_axis = _unit((edge[0] - along * normal[0], edge[1] - along * normal[1], edge[2] - along * normal[2]))
    v_axis = _cross(normal, u_axis)
    polygon = []
    for point in points:
        rel = _sub(point, origin)
        polygon.append((_dot(rel, u_axis), _dot(rel, v_axis)))
    return origin, u_axis, tuple(polygon)  # type: ignore[return-value]


def _outline(members: Sequence[int], triangles: Sequence[Sequence[int]]) -> list[int] | None:
    """The boundary of a set of triangles as one loop of mesh vertices, in the triangles' sense, or ``None`` when it is not
    one loop (a hole, two pieces, a corner the boundary passes twice)."""
    directed = {(t[i], t[(i + 1) % 3]) for k in members for t in (triangles[k],) for i in range(3)}
    successor: dict[int, int] = {}
    for a, b in directed:
        if (b, a) in directed:
            continue  # an inner edge of the patch
        if a in successor:
            return None
        successor[a] = b
    if not successor:  # pragma: no cover - a finite set of triangles has a boundary
        return None
    start = min(successor)
    loop = [start]
    while True:
        nxt = successor[loop[-1]]
        if nxt == start:
            break
        if len(loop) > len(successor):  # pragma: no cover - a vertex with one successor each closes its loop
            return None
        loop.append(nxt)
    return loop if len(loop) == len(successor) else None


def _straight(points: Sequence[tuple[float, float]]) -> list[int]:
    """The indices of the outline's corners that are not straight (a straight corner lies exactly on the line of its
    neighbours, between them)."""
    keep = []
    n = len(points)
    for i in range(n):
        (ax, ay), (bx, by), (cx, cy) = points[i - 1], points[i], points[(i + 1) % n]
        turn = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        if turn == 0.0 and min(ax, cx) <= bx <= max(ax, cx) and min(ay, cy) <= by <= max(ay, cy):
            continue
        keep.append(i)
    return keep


def _orientation(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> int:
    value = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    return (value > 0) - (value < 0)


def _on_segment(a: tuple[float, float], b: tuple[float, float], p: tuple[float, float]) -> bool:
    return min(a[0], b[0]) <= p[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= p[1] <= max(a[1], b[1])


def _segments_meet(a, b, c, d) -> bool:
    o1, o2, o3, o4 = _orientation(a, b, c), _orientation(a, b, d), _orientation(c, d, a), _orientation(c, d, b)
    if o1 != o2 and o3 != o4:
        return True
    return ((o1 == 0 and _on_segment(a, b, c)) or (o2 == 0 and _on_segment(a, b, d))
            or (o3 == 0 and _on_segment(c, d, a)) or (o4 == 0 and _on_segment(c, d, b)))


def outline_problem(points: Sequence[tuple[float, float]]) -> str | None:
    """Why ``points`` is not a simple counter-clockwise polygon of at most ``MAX_FACET_OUTLINE_CORNERS`` corners (``None``:
    it is one): the checks of ``shared_contracts.common.polygons.polygon_problem`` without its limit of
    ``MAX_POLYGON_VERTICES`` corners, the edge pairs found by a sweep along ``u`` (only edges whose ``u`` and ``v`` ranges
    overlap are compared)."""
    n = len(points)
    if n < 3:
        return "a polygon needs at least 3 corners"
    if n > MAX_FACET_OUTLINE_CORNERS:
        return f"a facet outline has at most {MAX_FACET_OUTLINE_CORNERS} corners, got {n}"
    if len(set(points)) != n:
        return "a polygon's corners are distinct"
    boxes = []
    for i in range(n):
        (ax, ay), (bx, by) = points[i], points[(i + 1) % n]
        boxes.append((min(ax, bx), max(ax, bx), min(ay, by), max(ay, by), i))
    boxes.sort()
    for k, (u_lo, u_hi, v_lo, v_hi, i) in enumerate(boxes):
        for other_lo, _other_hi, other_v_lo, other_v_hi, j in boxes[k + 1:]:
            if other_lo > u_hi:
                break
            if other_v_lo > v_hi or other_v_hi < v_lo:
                continue
            a, b = (i, j) if i < j else (j, i)
            if b == a + 1 or (a == 0 and b == n - 1):
                continue  # neighbours share a corner
            if _segments_meet(points[a], points[(a + 1) % n], points[b], points[(b + 1) % n]):
                return f"a polygon's edges do not cross or touch: edges {a} and {b} meet"
    for i in range(n):
        a, b, c = points[i - 1], points[i], points[(i + 1) % n]
        if _orientation(a, b, c) == 0 and not _on_segment(a, c, b):
            return f"a polygon's edges do not fold back on themselves (at corner {i})"
    if not signed_area_twice(points) > 0:
        return "a polygon's corners run counter-clockwise (positive area): reverse their order"
    return None


# ------------------------------------------------------------------------------------------------ planes


def _solve3(m: list[list[float]], rhs: list[float]) -> Vec | None:
    """``m x = rhs`` for a symmetric positive (semi)definite 3 x 3 ``m`` by Gaussian elimination with partial pivoting
    (``None``: singular)."""
    a = [row[:] + [r] for row, r in zip(m, rhs)]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda r: abs(a[r][col]))
        if a[pivot][col] == 0.0:
            return None
        a[col], a[pivot] = a[pivot], a[col]
        for r in range(col + 1, 3):
            f = a[r][col] / a[col][col]
            for c in range(col, 4):
                a[r][c] -= f * a[col][c]
    x = [0.0, 0.0, 0.0]
    for r in (2, 1, 0):
        rest = a[r][3]
        for c in range(r + 1, 3):
            rest -= a[r][c] * x[c]
        x[r] = rest / a[r][r]
    return (x[0], x[1], x[2])


def _snap(point: Sequence[float], planes: list[tuple[Vec, float]]) -> Vec:
    """The point nearest ``point`` on the planes ``(normal, offset)`` (``normal . x = offset``) in the least-squares sense:
    on all of them where they meet in one point (three or more independent planes) or a line (two), the nearest point of
    that line; directions the planes do not fix (weight below ``_SNAP_RIDGE``) keep ``point``'s position."""
    m = [[_SNAP_RIDGE if r == c else 0.0 for c in range(3)] for r in range(3)]
    rhs = [0.0, 0.0, 0.0]
    for normal, offset in planes:
        miss = _dot(normal, point) - offset
        for r in range(3):
            rhs[r] -= normal[r] * miss
            for c in range(3):
                m[r][c] += normal[r] * normal[c]
    step = _solve3(m, rhs)
    if step is None:  # pragma: no cover - the ridge keeps the matrix positive definite
        return (float(point[0]), float(point[1]), float(point[2]))
    return (point[0] + step[0], point[1] + step[1], point[2] + step[2])


def _project(point: Sequence[float], normal: Vec, offset: float) -> Vec:
    miss = _dot(normal, point) - offset
    return (point[0] - miss * normal[0], point[1] - miss * normal[1], point[2] - miss * normal[2])


@dataclass
class _Plane:
    """A patch in the making: its triangles, its plane (``normal . x = offset``), whether the plane is fitted (otherwise the
    plane of its first triangle, through its corners) and the farthest its corners lie from it."""

    members: list[int]
    group: int
    normal: Vec
    offset: float
    fitted: bool
    deviation: float


def _grow(seed: int, vertices, triangles, normals, across, group_of, owner, tol: float,
          plane: tuple[Vec, float] | None) -> list[int]:
    """The triangles edge-connected to ``seed`` (in no patch so far, of its group, facing its way) whose corners lie within
    ``tol`` of the plane ``plane`` -- or, for ``None``, of the plane fitted to the triangles found so far (the sum of their
    area vectors through their area-weighted centroid)."""
    area = [0.0, 0.0, 0.0]
    centre = [0.0, 0.0, 0.0]

    def add(k: int) -> None:
        a, b, c = (vertices[i] for i in triangles[k])
        cross = _cross(_sub(b, a), _sub(c, a))
        weight = math.sqrt(_dot(cross, cross))
        for r in range(3):
            area[r] += cross[r]
            centre[r] += weight * (a[r] + b[r] + c[r]) / 3.0

    add(seed)
    members = [seed]
    taken = {seed}
    queue = [seed]
    while queue:
        k = queue.pop()
        t = triangles[k]
        for i in range(3):
            j = across.get((t[(i + 1) % 3], t[i]))
            if j is None or j in taken or owner[j] >= 0 or group_of[j] != group_of[seed] or _dot(normals[j], normals[seed]) <= 0.0:
                continue
            if plane is None:
                normal = _unit((area[0], area[1], area[2]))
                total = math.sqrt(_dot(area, area))
                offset = _dot(normal, centre) / total
            else:
                normal, offset = plane
            if _dot(normals[j], normal) <= 0.0:
                continue
            if all(abs(_dot(normal, vertices[c]) - offset) <= tol for c in triangles[j]):
                taken.add(j)
                members.append(j)
                queue.append(j)
                add(j)
    return members


def _fit(members: Sequence[int], vertices, triangles) -> tuple[Vec, float, float]:
    """``(normal, offset, deviation)``: the plane of a set of triangles (normal the sum of their area vectors, offset midway
    between the farthest corners on either side) and the farthest a corner lies from it."""
    area = [0.0, 0.0, 0.0]
    for k in members:
        a, b, c = (vertices[i] for i in triangles[k])
        cross = _cross(_sub(b, a), _sub(c, a))
        for r in range(3):
            area[r] += cross[r]
    normal = _unit((area[0], area[1], area[2]))
    heights = [_dot(normal, vertices[i]) for k in members for i in triangles[k]]
    low, high = min(heights), max(heights)
    return normal, 0.5 * (low + high), 0.5 * (high - low)


def planar_patches(vertices: Sequence[Sequence[float]], triangles: Sequence[Sequence[int]],
                   groups: Sequence[int] | None = None, tolerance: float | None = None) -> list[Patch]:
    """The planar patches of a valid facet surface (``contracts.optics.mesh_problem`` holds), ordered by their first
    triangle (module docstring). ``groups``: the facet group of each triangle (triangles of different groups never share a
    patch); ``tolerance``: the coplanarity tolerance (mm; default ``coplanar_tolerance(vertices)``)."""
    count = len(triangles)
    tol = coplanar_tolerance(vertices) if tolerance is None else tolerance
    group_of = list(groups) if groups is not None else [0] * count
    across: dict[tuple[int, int], int] = {}
    for k, t in enumerate(triangles):
        for i in range(3):
            across[(t[i], t[(i + 1) % 3])] = k
    normals = [triangle_normal(vertices, t) for t in triangles]
    owner = [-1] * count
    planes: list[_Plane] = []
    for seed in range(count):
        if owner[seed] >= 0:
            continue
        members = _grow(seed, vertices, triangles, normals, across, group_of, owner, tol, None)
        normal, offset, deviation = _fit(members, vertices, triangles)
        if deviation > tol:
            # (the plane moved as it grew: keep the triangles within the tolerance of the plane it ended at)
            members = _grow(seed, vertices, triangles, normals, across, group_of, owner, tol, (normal, offset))
            heights = [_dot(normal, vertices[i]) for k in members for i in triangles[k]]
            deviation = max(abs(h - offset) for h in heights)
        for k in members:
            owner[k] = seed
        members.sort()
        seed_normal = normals[seed]
        anchor = vertices[triangles[seed][0]]
        off = max(abs(_dot(_sub(vertices[i], anchor), seed_normal)) for k in members for i in triangles[k])
        if off <= COPLANAR_FLOOR_MM:
            # exact data: the plane of the first triangle, through the corners
            planes.append(_Plane(members, group_of[seed], seed_normal, _dot(seed_normal, anchor), False, off))
        else:
            planes.append(_Plane(members, group_of[seed], normal, offset, True, deviation))

    def single(k: int) -> _Plane:
        n = normals[k]
        return _Plane([k], group_of[k], n, _dot(n, vertices[triangles[k][0]]), False, 0.0)

    # outlines; a patch with no single outline loop is traced triangle by triangle
    pieces: list[tuple[_Plane, list[int]]] = []
    for plane in planes:
        loop = _outline(plane.members, triangles) if len(plane.members) > 1 else list(triangles[plane.members[0]])
        if loop is None or (plane.fitted and _patch(plane, loop, vertices, {}) is None):
            # (decided before the corners move, so that the triangles' own planes take part in moving them)
            pieces.extend((single(k), list(triangles[k])) for k in plane.members)
        else:
            pieces.append((plane, loop))
    corners = _snapped(vertices, pieces) if any(p.fitted for p, _ in pieces) else {}

    patches: list[Patch] = []
    for plane, loop in pieces:
        patch = _patch(plane, loop, vertices, corners)
        if patch is not None:
            patches.append(patch)
            continue
        # (not one simple polygon: the patch's triangles one by one)
        for k in plane.members:
            one = _patch(single(k), list(triangles[k]), vertices, corners)
            if one is None:  # pragma: no cover - a valid (non-degenerate) facet is a simple triangle
                raise ValueError(f"facet {k} has no simple outline")
            patches.append(one)
    patches.sort(key=lambda p: p.first)
    return patches


def _snapped(vertices, pieces: list[tuple[_Plane, list[int]]]) -> dict[int, Vec]:
    """Every outline vertex of a fitted patch moved onto the planes of the patches whose outlines it is on (``_snap``),
    where some of them miss it by more than ``COPLANAR_FLOOR_MM``."""
    on: dict[int, list[tuple[Vec, float]]] = {}
    fitted: set[int] = set()
    for plane, loop in pieces:
        for i in loop:
            on.setdefault(i, []).append((plane.normal, plane.offset))
            if plane.fitted:
                fitted.add(i)
    out: dict[int, Vec] = {}
    for i in sorted(fitted):
        point = vertices[i]
        if any(abs(_dot(n, point) - d) > COPLANAR_FLOOR_MM for n, d in on[i]):
            out[i] = _snap(point, on[i])
    return out


def _patch(plane: _Plane, loop: list[int], vertices, corners: dict[int, Vec]) -> Patch | None:
    """The patch of ``plane`` with its outline ``loop`` (``None``: the outline is not a simple polygon of at most
    ``MAX_FACET_OUTLINE_CORNERS`` corners)."""
    normal = plane.normal
    if not plane.fitted and not any(i in corners for i in loop):
        points = [vertices[i] for i in loop]
    else:
        points = [_project(corners.get(i, vertices[i]), normal, plane.offset) for i in loop]
    if len(loop) > 3:
        keep = _straight(_frame(points, normal)[2])
        if len(keep) < 3:
            return None
        points, loop = [points[k] for k in keep], [loop[k] for k in keep]
    origin, u_axis, polygon = _frame(points, normal)
    if outline_problem(polygon) is not None:
        return None
    return Patch(tuple(plane.members), plane.group, origin, normal, u_axis, tuple(loop), polygon, plane.deviation)


__all__ = ["COPLANAR_FLOOR_MM", "COPLANAR_MAX_FRACTION", "COPLANAR_QUANTA", "COPLANAR_TOL_MM", "FITTED_FACE_NOTICE_FRACTION",
           "MAX_FACET_OUTLINE_CORNERS", "MAX_POLYGON_VERTICES", "Patch", "coordinate_quantum", "coplanar_tolerance", "mesh_extent",
           "outline_problem", "planar_patches", "triangle_normal"]
