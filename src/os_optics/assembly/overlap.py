"""Overlap of two closed part meshes, measured by exact ray columns (pure Python, deterministic).

The two meshes are expressed in one evaluation frame. For each of the frame's three axes, a grid of parallel
"columns" (rays along that axis) is laid over the intersection of the two meshes' bounding boxes. Along each column
the solid interval set of a mesh is exact: every triangle the column crosses contributes +1 (entering, its normal
opposes the ray) or -1 (leaving), and the column is inside the part where the running winding number is positive.
So overlapping shells of one part unite, and a tube's bore is empty. The two parts' interval sets are intersected
exactly along the column; only the column spacing is sampled.

Columns whose overlap is thinner than ``noise`` are surface contact (coplanar faces, float32 rounding of the
meshes, thin markers) and are ignored. Over the remaining "penetrating" columns each axis gives an overlap volume
(sum of interval length x column cell area) and the median column thickness. ``penetration`` is the smallest median
thickness over the three axes: for two overlapping boxes it is the true separation depth, a thin shell of
interference (a slightly undersized bore) gives about the shell thickness, and faces in contact give 0.

A median can be dominated by a broad shallow overlap that hides a small deep one (a pin driven through a plate
that also rests slightly sunk on a face). ``deep_volume`` catches that: the three axis grids share one lattice of
sample points, and a lattice point of the residual overlap counts as *deep* when the overlap interval through it is
at least ``deep`` thick along all three axes. A thin shell (thick along one axis only) has no deep points; a pin, a
stud or any solid chunk of interpenetration has.

Point-in-triangle tests use edge functions computed from the lexicographically smaller endpoint (so the two
triangles sharing an edge compute exact negatives) and a fixed tie rule on the edge direction (the rasterisation
"top-left" rule), so a column through a shared edge or vertex is counted exactly once. Column centres are also
offset by a fixed irrational-looking fraction of the spacing, making exact edge hits unlikely in the first place.

An optional ``EngagementZone`` (a cylinder about the frame's +Z axis) splits the overlap into the part inside the
zone -- the engagement an interface allows by design -- and the residual outside it, which is what is judged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .transforms import Vec3

COLUMN_JITTER = 0.0137  # fraction of a column cell by which column centres are offset (never on a cell centre)
NEGATIVE_STRETCH_TOLERANCE_MM = 1e-6  # a winding dip below zero shorter than this is rounding at a shared silhouette
AXES = (0, 1, 2)
ENTER, LEAVE = 1, -1

Interval = tuple[float, float]
Triangle = tuple[Vec3, Vec3, Vec3]


@dataclass(frozen=True)
class EngagementZone:
    """Cylinder about the evaluation frame's +Z axis: radius, and axial range ``z_min..z_max``."""

    radius: float
    z_min: float
    z_max: float

    def interval(self, axis: int, a: float, b: float) -> Interval | None:
        """The zone's extent along a column of ``axis`` at the other two coordinates ``(a, b)`` (in axis order)."""

        r2 = self.radius * self.radius
        if axis == 2:  # column along z at (x, y) = (a, b)
            return (self.z_min, self.z_max) if a * a + b * b <= r2 else None
        # column along x at (y, z) = (a, b), or along y at (x, z) = (a, b): the other radial coordinate is a
        if not (self.z_min <= b <= self.z_max) or a * a > r2:
            return None
        half = math.sqrt(r2 - a * a)
        return (-half, half)


@dataclass(frozen=True)
class OverlapMeasure:
    """Result of one pair evaluation. Volumes in mm^3, lengths in mm."""

    volume: float  # penetrating residual overlap volume (outside the zone), mean of the three axis integrations
    penetration: float  # smallest median residual thickness over the three axes (0 when there is no residual overlap)
    engaged_volume: float  # overlap inside the zone (0 without a zone), mean of the three integrations
    deep_volume: float  # residual overlap at least ``deep`` thick along all three axes (lattice estimate)
    inconsistent_columns: int  # columns whose winding numbers did not close (the measure is then unreliable)
    columns: int  # total columns cast


EMPTY = OverlapMeasure(0.0, 0.0, 0.0, 0, 0, 0.0)


def bounds_of(points: tuple[Vec3, ...]) -> tuple[Vec3, Vec3]:
    lo = (min(p[0] for p in points), min(p[1] for p in points), min(p[2] for p in points))
    hi = (max(p[0] for p in points), max(p[1] for p in points), max(p[2] for p in points))
    return lo, hi


def box_intersection(a: tuple[Vec3, Vec3], b: tuple[Vec3, Vec3]) -> tuple[Vec3, Vec3] | None:
    lo = tuple(max(a[0][k], b[0][k]) for k in AXES)
    hi = tuple(min(a[1][k], b[1][k]) for k in AXES)
    if any(lo[k] >= hi[k] for k in AXES):
        return None
    return lo, hi  # type: ignore[return-value]


def box_gap(a: tuple[Vec3, Vec3], b: tuple[Vec3, Vec3]) -> float:
    """Euclidean distance between two axis-aligned boxes (0 if they touch or overlap)."""

    total = 0.0
    for k in AXES:
        d = max(0.0, a[0][k] - b[1][k], b[0][k] - a[1][k])
        total += d * d
    return math.sqrt(total)


def _edge(pu: float, pv: float, qu: float, qv: float, x: float, y: float) -> float:
    """Edge function of directed edge P->Q at (x, y), bit-antisymmetric under P <-> Q."""

    if (pu, pv) <= (qu, qv):
        return (qu - pu) * (y - pv) - (qv - pv) * (x - pu)
    return -((pu - qu) * (y - qv) - (pv - qv) * (x - qu))


def _owns_tie(pu: float, pv: float, qu: float, qv: float) -> bool:
    """Tie rule: a point exactly on edge P->Q belongs to this triangle iff the edge points into a fixed half-plane."""

    du, dv = qu - pu, qv - pv
    return dv < 0.0 or (dv == 0.0 and du > 0.0)


@dataclass(frozen=True)
class _ProjectedTriangle:
    corners: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]  # counter-clockwise in (u, v)
    origin: Vec3  # (u0, v0, w0) of the first original corner
    gu: float
    gv: float
    delta: int  # ENTER or LEAVE along +w
    bbox: tuple[float, float, float, float]  # u_min, u_max, v_min, v_max

    def contains(self, x: float, y: float) -> bool:
        (au, av), (bu, bv), (cu, cv) = self.corners
        for pu, pv, qu, qv in ((au, av, bu, bv), (bu, bv, cu, cv), (cu, cv, au, av)):
            e = _edge(pu, pv, qu, qv, x, y)
            if e < 0.0 or (e == 0.0 and not _owns_tie(pu, pv, qu, qv)):
                return False
        return True

    def w_at(self, x: float, y: float) -> float:
        return self.origin[2] + self.gu * (x - self.origin[0]) + self.gv * (y - self.origin[1])


def _project(triangles: list[Triangle], axis: int) -> list[_ProjectedTriangle]:
    ui, vi = [(1, 2), (2, 0), (0, 1)][axis]  # right-handed (u, v, w) so +w = u x v
    out = []
    for p, q, r in triangles:
        a, b, c = (p[ui], p[vi], p[axis]), (q[ui], q[vi], q[axis]), (r[ui], r[vi], r[axis])
        d1u, d1v, d1w = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        d2u, d2v, d2w = c[0] - a[0], c[1] - a[1], c[2] - a[2]
        area = d1u * d2v - d1v * d2u  # = the triangle normal's w component (x2)
        if area == 0.0:
            continue  # edge-on to the column direction: never crossed
        gu = (d1w * d2v - d2w * d1v) / area
        gv = (d2w * d1u - d1w * d2u) / area
        corners = ((a[0], a[1]), (b[0], b[1]), (c[0], c[1])) if area > 0.0 else ((a[0], a[1]), (c[0], c[1]), (b[0], b[1]))
        bbox = (min(a[0], b[0], c[0]), max(a[0], b[0], c[0]), min(a[1], b[1], c[1]), max(a[1], b[1], c[1]))
        out.append(_ProjectedTriangle(corners, a, gu, gv, LEAVE if area > 0.0 else ENTER, bbox))
    return out


def _grid(lo: float, hi: float, spacing: float, max_columns: int) -> tuple[list[float], float]:
    extent = hi - lo
    n = min(max_columns, max(1, math.ceil(extent / spacing)))
    step = extent / n
    return [lo + (i + 0.5 + COLUMN_JITTER) * step for i in range(n)], step


def _buckets(tris: list[_ProjectedTriangle], us: list[float], vs: list[float]) -> dict[tuple[int, int], list[_ProjectedTriangle]]:
    buckets: dict[tuple[int, int], list[_ProjectedTriangle]] = {}
    for tri in tris:
        u_min, u_max, v_min, v_max = tri.bbox
        iu = [i for i, u in enumerate(us) if u_min <= u <= u_max]
        iv = [j for j, v in enumerate(vs) if v_min <= v <= v_max]
        for i in iu:
            for j in iv:
                buckets.setdefault((i, j), []).append(tri)
    return buckets


def _solid(tris: list[_ProjectedTriangle], x: float, y: float) -> tuple[list[Interval], bool]:
    """Solid intervals along the column at (x, y), and whether the winding closed consistently."""

    crossings = sorted((t.w_at(x, y), -t.delta) for t in tris if t.contains(x, y))  # at equal w, enter before leave
    intervals: list[Interval] = []
    winding, start, dip_start, ok = 0, 0.0, 0.0, True
    for w, neg_delta in crossings:
        before = winding
        winding -= neg_delta
        if before <= 0 < winding:
            start = w
        elif before > 0 >= winding:
            intervals.append((start, w))
        if before >= 0 > winding:
            dip_start = w
        elif before < 0 <= winding and w - dip_start > NEGATIVE_STRETCH_TOLERANCE_MM:
            ok = False
    return intervals, ok and winding == 0


def _intersect(a: list[Interval], b: list[Interval]) -> list[Interval]:
    out, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if lo < hi:
            out.append((lo, hi))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def _outside(intervals: list[Interval], zone: Interval | None) -> list[Interval]:
    """The parts of ``intervals`` outside the zone interval."""

    if zone is None:
        return intervals
    out = []
    for lo, hi in intervals:
        if lo < zone[0]:
            out.append((lo, min(hi, zone[0])))
        if hi > zone[1]:
            out.append((max(lo, zone[1]), hi))
    return [(lo, hi) for lo, hi in out if lo < hi]


def _length(intervals: list[Interval]) -> float:
    return sum(hi - lo for lo, hi in intervals)


def _thick_at(intervals: list[Interval] | None, w: float, deep: float) -> bool:
    """Whether ``w`` lies in an interval at least ``deep`` long."""

    return any(lo <= w <= hi and hi - lo >= deep for lo, hi in intervals or ())


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def measure_overlap(tris_a: list[Triangle], tris_b: list[Triangle], region: tuple[Vec3, Vec3], *, spacing: float,
                    max_columns: int, noise: float, deep: float, zone: EngagementZone | None = None) -> OverlapMeasure:
    """Overlap of two closed meshes (already in the evaluation frame) inside ``region`` (their bbox intersection)."""

    volumes, engaged, medians = [], [], []
    inconsistent = columns = 0
    residual: list[dict[tuple[int, int], list[Interval]]] = [{}, {}, {}]  # per axis: column -> residual intervals
    for axis in AXES:
        ui, vi = [(1, 2), (2, 0), (0, 1)][axis]
        us, du = _grid(region[0][ui], region[1][ui], spacing, max_columns)
        vs, dv = _grid(region[0][vi], region[1][vi], spacing, max_columns)
        bucket_a = _buckets(_project(tris_a, axis), us, vs)
        bucket_b = _buckets(_project(tris_b, axis), us, vs)
        cell = du * dv
        residual_sum = engaged_sum = 0.0
        thickness: list[float] = []
        for i, u in enumerate(us):
            for j, v in enumerate(vs):
                columns += 1
                ta, tb = bucket_a.get((i, j)), bucket_b.get((i, j))
                if not ta or not tb:
                    continue
                sa, ok_a = _solid(ta, u, v)
                sb, ok_b = _solid(tb, u, v)
                if not (ok_a and ok_b):
                    inconsistent += 1
                    continue
                both = _intersect(sa, sb)
                if not both:
                    continue
                # zone coordinates: for axis 2 the column is at (x, y) = (u, v); for axes 0/1 the radial one is the
                # coordinate that is not z, and the axial one is z.
                if zone is None:
                    zone_interval = None
                elif axis == 2:
                    zone_interval = zone.interval(2, u, v)
                elif axis == 0:  # (u, v) = (y, z)
                    zone_interval = zone.interval(0, u, v)
                else:  # axis 1: (u, v) = (z, x)
                    zone_interval = zone.interval(1, v, u)
                outside = _outside(both, zone_interval)
                out = _length(outside)
                engaged_sum += (_length(both) - out) * cell
                residual[axis][(i, j)] = outside
                if out > noise:
                    residual_sum += out * cell
                    thickness.append(out)
        volumes.append(residual_sum)
        engaged.append(engaged_sum)
        medians.append(_median(thickness) if thickness else 0.0)
    # The three grids share one lattice: coordinate c always uses _grid(region c). Axis 2 columns are keyed (x, y),
    # axis 0 columns (y, z), axis 1 columns (z, x).
    grids = [_grid(region[0][c], region[1][c], spacing, max_columns) for c in AXES]
    xs, ys, zs = (g[0] for g in grids)
    lattice_cell = grids[0][1] * grids[1][1] * grids[2][1]
    deep_points = 0
    for (i, j), intervals in residual[2].items():
        for k, z in enumerate(zs):
            if (_thick_at(intervals, z, deep) and _thick_at(residual[0].get((j, k)), xs[i], deep)
                    and _thick_at(residual[1].get((k, i)), ys[j], deep)):
                deep_points += 1
    return OverlapMeasure(volume=sum(volumes) / len(volumes), penetration=min(medians),
                          engaged_volume=sum(engaged) / len(engaged), inconsistent_columns=inconsistent, columns=columns,
                          deep_volume=deep_points * lattice_cell)


def solid_intervals_along(tris: list[Triangle]) -> tuple[list[Interval], bool]:
    """Solid intervals of a closed mesh along the evaluation frame's +Z axis through the origin (one column)."""

    return _solid(_project(tris, 2), 0.0, 0.0)
