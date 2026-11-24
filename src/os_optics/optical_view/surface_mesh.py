"""Display meshes of optical surfaces within a stated chord tolerance (presentation only, pure Python, deterministic).

Every optical surface is drawn as a triangle mesh whose vertices lie exactly on the published profile and whose
triangles stay within ``CHORD_TOLERANCE_MM`` of it, measured along the surface normal (to first order: the axial gap
between a triangle and the profile times the cosine of the surface slope). Round outlines and apertures are regular
polygons whose sides stay within the same tolerance of the circle. Curvature is never scaled: the mesh is the profile.

Construction (a "star" mesh around a centre that lies inside every boundary):

* directions: ``n`` evenly spaced unit vectors (``geometry.turn_cos_sin``) plus the directions of every polygon corner
  (a rectangular outline or aperture keeps its exact straight sides);
* zones along every direction: the clear aperture from the centre to the aperture boundary, then (when the outline is
  larger) a narrow band that marks the clear-aperture outline, then the opaque rim out to the outline; every zone has its
  own stations (fractions of its length) placed by ``profile_stations`` so that the chord of the profile between two
  stations stays within half the tolerance (the other half is the budget of the directions);
* a rotationally symmetric profile in concentric round boundaries is exact by construction (the meridional chords by the
  station sampler, the azimuthal ones by ``n``); every other surface is checked (``max_deviation``: the profile at every
  triangle's centroid and edge midpoints) and refined (twice the directions, every station interval halved) until it is
  within the tolerance.

Determinism: ``+ - * /`` and ``math.sqrt`` only (no libm trig, no ``math.hypot``), so CPython and Pyodide take the same
decisions and write the same bits. Normals are analytic (the gradient of the sag), not finite differences.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from .._vendor.shared_contracts.common.sag_shapes import SagForm, sag_form, sag_slopes, sag_value
from . import geometry as g

#: Every drawn optical surface lies within this distance (mm) of its exact profile, and every drawn round outline or
#: aperture within this distance of its circle.
CHORD_TOLERANCE_MM = 0.002
#: Share of the tolerance spent along the profile (the stations); the rest is the budget between directions.
MERIDIONAL_SHARE = 0.5
#: Interior points per station interval at which the chord deviation is measured.
CHORD_PROBES = 8
MAX_STATION_DEPTH = 40
MAX_REFINEMENTS = 5
#: Directions of a star mesh with no circular boundary and a flat profile (a rectangle's corners are added).
MIN_DIRECTIONS = 8
#: Points closer than this (in pseudo-angle) are one direction.
DIRECTION_EPS = 1e-12
#: A profile is evaluated at most this close to its domain edge.
DOMAIN_MARGIN = 1e-12
#: Halvings that pull a point back inside a two-variable profile's domain along its direction from the centre.
DOMAIN_BISECTIONS = 48
#: A boundary point this close to another one is the same point (mm).
SAME_POINT_MM = 1e-9
#: Station fractions closer than this are one station.
SAME_FRACTION = 1e-12
#: Within this distance (mm) of an axicon's apex the profile's normal is its axis (the apex has no tangent plane; the
#: Engine's ``AXICON_APEX_RADIUS_MM``).
APEX_RADIUS_MM = 1e-9
#: Direction counts are c * 2^k with c in BASE_MIN..BASE_MAX (rows thin out by halving towards the centre).
BASE_MIN = 8
BASE_MAX = 15

Vec2 = tuple[float, float]


# ------------------------------------------------------------------------------------------------------ profiles


@dataclass(frozen=True)
class Profile:
    """The sag of a published surface profile in its local (x, y) frame (the formula of ``OpticSurfaceV1.profile_sag``):

    ``z = c rho^2 / (1 + sqrt(1 - (1 + k) c^2 rho^2)) + A4 rho^4 + A6 rho^6 + ... + a rho``, with ``rho^2 = x^2 + y^2``
    or, for a cylinder straight along X (Y), ``y^2`` (``x^2``); ``a = linear`` is the axicon term (a cone, or a roof on a
    cylinder), whose apex is a kink: a mesh puts a vertex there (a star mesh's centre, a station at 0). ``form`` is a
    two-variable sag shape (``common.sag_shapes``): a Zernike or XY polynomial shape adds to these terms, a biconic or a toroid
    replaces them; such a profile is not a surface of revolution and is always checked against its mesh."""

    curvature: float = 0.0
    conic: float = 0.0
    coefficients: tuple[float, ...] = ()
    cylinder_axis: str | None = None
    linear: float = 0.0
    form: SagForm | None = None

    @classmethod
    def of(cls, surface) -> Profile:
        """The profile of an ``OpticSurfaceV1`` (or any object with ``radius_mm``, ``conic``, ``aspheric_coefficients``,
        ``cylinder_axis`` as floats)."""

        radius = surface.radius_mm
        shape = getattr(surface, "sag", None)
        return cls(curvature=0.0 if radius is None else 1.0 / radius, conic=surface.conic or 0.0,
                   coefficients=tuple(surface.aspheric_coefficients or ()), cylinder_axis=surface.cylinder_axis,
                   linear=getattr(surface, "linear_sag_coefficient", None) or 0.0, form=None if shape is None else sag_form(shape))

    @property
    def is_flat(self) -> bool:
        return self.curvature == 0.0 and not self.coefficients and self.linear == 0.0 and self.form is None

    def _cone_slope(self, rho_sq: float, root: float) -> float:
        """The axicon term's part of the gradient factor: ``root a / rho`` (0 at the apex and without the term)."""

        if self.linear == 0.0:
            return 0.0
        rho = math.sqrt(rho_sq)
        return root * self.linear / rho if rho > APEX_RADIUS_MM else 0.0

    @property
    def rotational(self) -> bool:
        return self.cylinder_axis is None and self.form is None

    def rho_sq(self, x: float, y: float) -> float:
        if self.cylinder_axis == "X":
            return y * y
        if self.cylinder_axis == "Y":
            return x * x
        return x * x + y * y

    @property
    def domain_rho(self) -> float:
        """The largest profile coordinate at which the profile is defined (``inf`` when it is defined everywhere); a biconic or a
        toroid: the largest radius of a centred disc inside its domain."""

        if self.form is not None and not self.form.additive:
            return self._form_domain_rho()
        k1c2 = (1.0 + self.conic) * self.curvature * self.curvature
        return 1.0 / math.sqrt(k1c2) if k1c2 > 0.0 else math.inf

    def _form_domain_rho(self) -> float:
        form = self.form
        k1c2 = max(form.conic_x or 0.0, form.conic_y or 0.0)  # type: ignore[union-attr]
        limit = 1.0 / math.sqrt(k1c2) if k1c2 > 0.0 else math.inf
        if form.rotation is not None:  # type: ignore[union-attr]
            limit = min(limit, abs(form.rotation))  # type: ignore[union-attr]
        return limit

    def sag_rho_sq(self, rho_sq: float) -> float | None:
        sag = 0.0
        if self.curvature != 0.0:
            arg = 1.0 - (1.0 + self.conic) * self.curvature * self.curvature * rho_sq
            if arg < 0.0:
                return None
            sag = self.curvature * rho_sq / (1.0 + math.sqrt(arg))
        power = rho_sq * rho_sq
        for coefficient in self.coefficients:
            sag += coefficient * power
            power *= rho_sq
        if self.linear != 0.0:
            sag += self.linear * math.sqrt(rho_sq)
        return sag

    def sag(self, x: float, y: float) -> float | None:
        if self.form is None:
            return self.sag_rho_sq(self.rho_sq(x, y))
        z, ok = sag_value(self.form, x, y)
        if not ok:
            return None
        if not self.form.additive:
            return z
        base = self.sag_rho_sq(self.rho_sq(x, y))
        return None if base is None else base + z

    def _form_sag_normal(self, x: float, y: float) -> tuple[float, float, float, float] | None:
        """``(sag, nx, ny, nz)`` of a two-variable profile: its slopes in the kernel's form (``(-h_x, -h_y, w)``), the rotational
        terms added with their root multiplied through as in ``sag_normal``."""

        z, h_x, h_y, w, ok = sag_slopes(self.form, x, y)  # type: ignore[arg-type]
        if not ok:
            return None
        if self.form.additive:  # type: ignore[union-attr]
            rho_sq = x * x + y * y
            root = 1.0
            if self.curvature != 0.0:
                arg = 1.0 - (1.0 + self.conic) * self.curvature * self.curvature * rho_sq
                if arg < 0.0:
                    return None
                root = math.sqrt(arg)
                z += self.curvature * rho_sq / (1.0 + root)
            poly, p2, p4 = 0.0, rho_sq, rho_sq * rho_sq
            for i, coefficient in enumerate(self.coefficients):
                z += coefficient * p4
                poly += (2 * i + 4) * coefficient * p2
                p2 *= rho_sq
                p4 *= rho_sq
            slope = self.curvature + root * poly
            h_x, h_y, w = slope * x + root * h_x, slope * y + root * h_y, root
        length = math.sqrt(h_x * h_x + h_y * h_y + w * w)
        if length == 0.0:
            return (z, 0.0, 0.0, 1.0)
        return (z, -h_x / length, -h_y / length, w / length)

    def normal(self, x: float, y: float) -> g.Vec3 | None:
        """Unit normal at ``(x, y)`` on the +z side of the profile (``(0, 0, 1)`` at the vertex), from the analytic
        gradient written so it stays finite at the domain edge: ``(-(c + sqrt(D) g) x', ..., sqrt(D))`` with
        ``D = 1 - (1 + k) c^2 rho^2`` and ``g = sum (2i + 4) A rho^(2i + 2)``."""

        if self.form is not None:
            hit = self._form_sag_normal(x, y)
            return None if hit is None else hit[1:]
        rho_sq = self.rho_sq(x, y)
        root = 1.0
        if self.curvature != 0.0:
            arg = 1.0 - (1.0 + self.conic) * self.curvature * self.curvature * rho_sq
            if arg < 0.0:
                return None
            root = math.sqrt(arg)
        poly, power = 0.0, rho_sq
        for i, coefficient in enumerate(self.coefficients):
            poly += (2 * i + 4) * coefficient * power
            power *= rho_sq
        slope = self.curvature + root * poly + self._cone_slope(rho_sq, root)
        gx = 0.0 if self.cylinder_axis == "X" else slope * x
        gy = 0.0 if self.cylinder_axis == "Y" else slope * y
        length = math.sqrt(gx * gx + gy * gy + root * root)
        if length == 0.0:  # exactly at the rim of a hemisphere-like domain edge with no slope term: radial
            return (0.0, 0.0, 1.0)
        return (-gx / length, -gy / length, root / length)

    def sag_normal(self, x: float, y: float) -> tuple[float, float, float, float] | None:
        """``(sag, nx, ny, nz)`` at ``(x, y)`` in one evaluation (``sag`` and ``normal`` share their square root)."""

        if self.form is not None:
            return self._form_sag_normal(x, y)
        rho_sq = x * x + y * y if self.cylinder_axis is None else (y * y if self.cylinder_axis == "X" else x * x)
        c = self.curvature
        sag, root = 0.0, 1.0
        if c != 0.0:
            arg = 1.0 - (1.0 + self.conic) * c * c * rho_sq
            if arg < 0.0:
                return None
            root = math.sqrt(arg)
            sag = c * rho_sq / (1.0 + root)
        slope = c
        if self.coefficients:
            poly, p2, p4 = 0.0, rho_sq, rho_sq * rho_sq
            for i, coefficient in enumerate(self.coefficients):
                sag += coefficient * p4
                poly += (2 * i + 4) * coefficient * p2
                p2 *= rho_sq
                p4 *= rho_sq
            slope = c + root * poly
        if self.linear != 0.0:
            sag += self.linear * math.sqrt(rho_sq)
            slope = slope + self._cone_slope(rho_sq, root)
        gx = 0.0 if self.cylinder_axis == "X" else slope * x
        gy = 0.0 if self.cylinder_axis == "Y" else slope * y
        length = math.sqrt(gx * gx + gy * gy + root * root)
        if length == 0.0:
            return (sag, 0.0, 0.0, 1.0)
        return (sag, -gx / length, -gy / length, root / length)

    def domain_point(self, x: float, y: float) -> Vec2:
        """``(x, y)`` moved inside the profile's domain (along the curved coordinate) when it lies beyond it."""

        if self.form is not None and not self.form.additive:
            if sag_value(self.form, x, y)[1]:
                return (x, y)
            low, high = 0.0, 1.0  # along the direction from the axis: the point itself is outside, the axis inside
            for _ in range(DOMAIN_BISECTIONS):
                middle = 0.5 * (low + high)
                if sag_value(self.form, x * middle, y * middle)[1]:
                    low = middle
                else:
                    high = middle
            return (x * low, y * low)
        limit = self.domain_rho * (1.0 - DOMAIN_MARGIN)
        if self.cylinder_axis == "X":
            return (x, max(-limit, min(limit, y)))
        if self.cylinder_axis == "Y":
            return (max(-limit, min(limit, x)), y)
        rho_sq = x * x + y * y
        if rho_sq <= limit * limit:
            return (x, y)
        f = limit / math.sqrt(rho_sq)
        return (x * f, y * f)


def held_point(x: float, y: float, edge: tuple[float, float] | None, round_edge: bool) -> Vec2:
    """The point whose sag a surface with its own mechanical edge has at ``(x, y)``: the point itself inside the edge (a
    centred circle of radius ``edge[0]``, or a centred rectangle of half widths ``edge``), beyond it the edge point on the
    ray from the axis through it -- the surface held flat at its rim height (``SurfaceSequenceModelV1.held_point_mm``).
    ``edge`` None: the surface reaches the outline (the point itself)."""

    if edge is None:
        return x, y
    if round_edge:
        rho_sq = x * x + y * y
        if rho_sq <= edge[0] * edge[0]:
            return x, y
        f = edge[0] / math.sqrt(rho_sq)
    else:
        reach = max(abs(x) / edge[0], abs(y) / edge[1])
        if reach <= 1.0:
            return x, y
        f = 1.0 / reach
    return x * f, y * f


def profile_stations(f: Callable[[float], float | None], t0: float, t1: float, tolerance: float) -> list[float]:
    """Stations ``t0 < ... < t1`` along a curve ``(t, f(t))`` such that every chord between consecutive stations stays
    within ``tolerance`` of the curve (perpendicular distance, measured at ``CHORD_PROBES - 1`` interior points of every
    interval; an interval where ``f`` is undefined at one end is split down to ``MAX_STATION_DEPTH`` to find where it ends, one
    undefined at both ends and at its probes lies beyond the profile and stays whole)."""

    out = [t0]

    def undefined_throughout(a: float, b: float) -> bool:
        return all(f(a + (b - a) * k / CHORD_PROBES) is None for k in range(1, CHORD_PROBES))

    def deviation(a: float, fa: float, b: float, fb: float) -> float:
        dt, dz = b - a, fb - fa
        length = math.sqrt(dt * dt + dz * dz)
        worst = 0.0
        for k in range(1, CHORD_PROBES):
            t = a + (b - a) * k / CHORD_PROBES
            z = f(t)
            if z is None:
                return math.inf
            d = abs((t - a) * dz - (z - fa) * dt) / length
            if d > worst:
                worst = d
        return worst

    def split(a: float, fa: float | None, b: float, fb: float | None, depth: int) -> None:
        if fa is None and fb is None and undefined_throughout(a, b):
            out.append(b)
        elif depth < MAX_STATION_DEPTH and (fa is None or fb is None or deviation(a, fa, b, fb) > tolerance):
            m = 0.5 * (a + b)
            fm = f(m)
            split(a, fa, m, fm, depth + 1)
            split(m, fm, b, fb, depth + 1)
        else:
            out.append(b)

    if t1 > t0:
        split(t0, f(t0), t1, f(t1), 0)
    return out


def sphere_stations(radius: float, rho0: float, rho1: float, budget: float) -> list[float]:
    """Stations ``rho0 < ... < rho1`` (transverse radii) on a sphere of signed ``radius`` at equal arc angles whose chords
    have a sagitta of exactly ``budget`` (the last interval is shorter): ``cos(h) = 1 - budget / |R|`` per half step,
    stepped by an exact rotation recurrence (no trigonometry)."""

    r = abs(radius)
    if budget >= r or rho1 <= rho0:
        return [rho0, rho1] if rho1 > rho0 else [rho0]
    q = budget / r
    ch, sh = 1.0 - q, math.sqrt(q * (2.0 - q))
    cd, sd = ch * ch - sh * sh, 2.0 * sh * ch
    s = rho0 / r
    c = math.sqrt(max(0.0, (1.0 - s) * (1.0 + s)))
    out = [rho0]
    while True:
        c, s = c * cd - s * sd, s * cd + c * sd
        if c <= 0.0 or s * r >= rho1:
            break
        out.append(s * r)
    out.append(rho1)
    return out


def fractions(stations: Sequence[float]) -> list[float]:
    """Stations ``t0 .. t1`` as fractions in ``(0, 1]`` of their span (``t0`` dropped)."""

    t0, t1 = stations[0], stations[-1]
    span = t1 - t0
    return [(t - t0) / span for t in stations[1:-1]] + [1.0]


def merged(*lists: Sequence[float]) -> list[float]:
    """Sorted union of station fractions (ending with exactly 1.0)."""

    values = sorted({v for lst in lists for v in lst if 0.0 < v < 1.0})
    out: list[float] = []
    for v in values:
        if not out or v - out[-1] > SAME_FRACTION:
            out.append(v)
    return out + [1.0]


def bisected(stations: Sequence[float]) -> list[float]:
    out: list[float] = []
    previous = 0.0
    for s in stations:
        out += [0.5 * (previous + s), s]
        previous = s
    return out


def bisected_values(values: Sequence[float]) -> list[float]:
    """Ascending values with the midpoint of every interval inserted."""

    out = [values[0]]
    for a, b in zip(values, values[1:]):
        out += [0.5 * (a + b), b]
    return out


# ---------------------------------------------------------------------------------------------------- boundaries


@dataclass(frozen=True)
class Circle:
    cx: float
    cy: float
    r: float

    def exit(self, ox: float, oy: float, dx: float, dy: float) -> float:
        """Distance along the unit direction ``(dx, dy)`` from ``(ox, oy)`` (inside) to the circle."""

        px, py = ox - self.cx, oy - self.cy
        b = dx * px + dy * py
        c = px * px + py * py - self.r * self.r
        return -b + math.sqrt(max(b * b - c, 0.0))

    def contains(self, x: float, y: float, margin: float = 0.0) -> bool:
        px, py = x - self.cx, y - self.cy
        return px * px + py * py < (self.r - margin) * (self.r - margin)

    @property
    def corners(self) -> tuple[Vec2, ...]:
        return ()

    @property
    def reach(self) -> float:
        return math.sqrt(self.cx * self.cx + self.cy * self.cy) + self.r


@dataclass(frozen=True)
class Polygon:
    """A convex polygon, corners counter-clockwise."""

    points: tuple[Vec2, ...]

    @classmethod
    def rectangle(cls, cx: float, cy: float, hx: float, hy: float) -> Polygon:
        return cls(((cx - hx, cy - hy), (cx + hx, cy - hy), (cx + hx, cy + hy), (cx - hx, cy + hy)))

    def _edges(self):
        count = len(self.points)
        for i in range(count):
            (ax, ay), (bx, by) = self.points[i], self.points[(i + 1) % count]
            nx, ny = by - ay, ax - bx  # outward for a counter-clockwise polygon
            yield nx, ny, nx * ax + ny * ay

    def exit(self, ox: float, oy: float, dx: float, dy: float) -> float:
        best = math.inf
        for nx, ny, h in self._edges():
            rate = nx * dx + ny * dy
            if rate > 0.0:
                t = (h - (nx * ox + ny * oy)) / rate
                if t < best:
                    best = t
        return max(best, 0.0)

    def contains(self, x: float, y: float, margin: float = 0.0) -> bool:
        for nx, ny, h in self._edges():
            length = math.sqrt(nx * nx + ny * ny)
            if nx * x + ny * y > h - margin * length:
                return False
        return True

    @property
    def corners(self) -> tuple[Vec2, ...]:
        return self.points

    @property
    def reach(self) -> float:
        return max(math.sqrt(x * x + y * y) for x, y in self.points)


Boundary = Circle | Polygon


def directions(n: int, centre: Vec2, corner_sets: Sequence[Sequence[Vec2]]) -> list[Vec2]:
    """``n`` evenly spaced unit directions plus the directions from ``centre`` to every corner, counter-clockwise."""

    found: list[tuple[float, Vec2]] = [(g.pseudo_angle(c, s), (c, s)) for c, s in g.unit_circle(n)]
    for corners in corner_sets:
        for x, y in corners:
            dx, dy = x - centre[0], y - centre[1]
            length = math.sqrt(dx * dx + dy * dy)
            if length > SAME_POINT_MM:
                found.append((g.pseudo_angle(dx, dy), (dx / length, dy / length)))
    found.sort(key=lambda item: item[0])
    out: list[tuple[float, Vec2]] = []
    for angle, d in found:
        if not out or angle - out[-1][0] > DIRECTION_EPS:
            out.append((angle, d))
    if len(out) > 1 and out[0][0] + 4.0 - out[-1][0] <= DIRECTION_EPS:
        out.pop()
    return [d for _, d in out]


# ----------------------------------------------------------------------------------------------------- the mesh


@dataclass(frozen=True)
class Placement:
    """Where a surface's local frame sits in the element frame: its vertex ``origin`` and axes ``u``, ``v``, ``w``
    (``w``: the surface axis; a coaxial surface: the element axes, origin ``(0, 0, vertex_z)``)."""

    origin: g.Vec3
    u: g.Vec3 = (1.0, 0.0, 0.0)
    v: g.Vec3 = (0.0, 1.0, 0.0)
    w: g.Vec3 = (0.0, 0.0, 1.0)

    @classmethod
    def coaxial(cls, vertex_z: float) -> Placement:
        return cls((0.0, 0.0, vertex_z))

    @property
    def is_coaxial(self) -> bool:
        return self.u == (1.0, 0.0, 0.0) and self.v == (0.0, 1.0, 0.0) and self.w == (0.0, 0.0, 1.0) and self.origin[:2] == (0.0, 0.0)

    def point(self, a: float, b: float, z: float) -> g.Vec3:
        if self.is_coaxial:
            return (a, b, self.origin[2] + z)
        o, u, v, w = self.origin, self.u, self.v, self.w
        return (o[0] + a * u[0] + b * v[0] + z * w[0], o[1] + a * u[1] + b * v[1] + z * w[1], o[2] + a * u[2] + b * v[2] + z * w[2])

    def vector(self, a: float, b: float, c: float) -> g.Vec3:
        if self.is_coaxial:
            return (a, b, c)
        u, v, w = self.u, self.v, self.w
        return (a * u[0] + b * v[0] + c * w[0], a * u[1] + b * v[1] + c * w[1], a * u[2] + b * v[2] + c * w[2])


@dataclass(frozen=True)
class Zone:
    """One zone of a star mesh: from ``inner`` (per direction, its distance from the centre; ``None``: the centre itself)
    to ``outer`` along every direction, at ``stations`` (fractions in ``(0, 1]``), drawn in ``material``.

    ``strides``: per station, the row uses every ``stride``-th direction (1: all of them); ``inner_stride`` is the stride
    of the inner boundary row (the previous zone's last row). Consecutive rows differ by at most a factor of 2 and a row
    is never coarser than a row inside it (``row_strides``).

    ``hold``: per direction, the distance at which the profile is held (a surface's own mechanical edge): a zone beyond
    it is the surface's flat flange, every vertex at the sag of the edge point along its direction (``held_point``), its
    normal along the surface axis."""

    material: str
    inner: tuple[float, ...] | None
    outer: tuple[float, ...]
    stations: tuple[float, ...]
    strides: tuple[int, ...] | None = None
    inner_stride: int = 1
    hold: tuple[float, ...] | None = None


@dataclass
class SurfaceMesh:
    """A surface's meshes by material, its outline row (element-frame points along the last zone's outer boundary, one per
    direction), every zone's outer row (``rows``, in zone order; empty when that row does not use every direction) and
    whether its profile is undefined somewhere it is drawn."""

    meshes: list[tuple[str, g.Mesh]] = field(default_factory=list)
    outline: list[g.Vec3] = field(default_factory=list)
    undefined: bool = False
    rows: list[list[g.Vec3]] = field(default_factory=list)


UNDEFINED_MATERIAL = "profile_undefined"
#: A surface's flat flange beyond its own mechanical edge (a ground face, as the glass's edge wall): not on its profile.
FLANGE_MATERIAL = "optic_edge"


def _stitch(triangles: list[tuple[int, int, int]], inner: int, inner_stride: int, outer: int, outer_stride: int, count: int) -> None:
    """Triangles between two rows (first vertex indices ``inner`` / ``outer``) over ``count`` directions, counter-clockwise
    seen from +Z; the inner row has ``inner_stride`` = ``outer_stride`` or twice it."""

    n_in, n_out = count // inner_stride, count // outer_stride
    if n_in == n_out:
        for j in range(n_in):
            k = (j + 1) % n_in
            a, b, c, d = inner + j, outer + j, outer + k, inner + k
            triangles.extend(((a, b, c), (a, c, d)))
        return
    for j in range(n_in):  # one inner segment, two outer ones
        i0, i1 = inner + j, inner + (j + 1) % n_in
        o0, o1, o2 = outer + 2 * j, outer + 2 * j + 1, outer + (2 * j + 2) % n_out
        triangles.extend(((i0, o0, o1), (i0, o1, i1), (i1, o1, o2)))


def star_mesh(profile: Profile, placement: Placement, centre: Vec2, dirs: Sequence[Vec2], zones: Sequence[Zone], facing: float
              ) -> SurfaceMesh:
    """Triangles of ``zones`` around ``centre`` (local x, y of the surface), on the profile, in the element frame.

    Winding is counter-clockwise seen from ``(0, 0, facing)``; normals are the analytic profile normals turned to
    ``facing``. A vertex where the profile is undefined is held at the domain edge along its direction and every
    triangle touching one is drawn in ``UNDEFINED_MATERIAL`` instead of its zone's material."""

    out = SurfaceMesh()
    count = len(dirs)
    cx, cy = centre
    coaxial = placement.is_coaxial
    z0 = placement.origin[2]
    for zone in zones:
        positions: list[g.Vec3] = []
        normals: list[g.Vec3] = []
        bad: list[int] = []

        def row(fraction: float | None, stride: int, zone: Zone = zone, positions: list = positions, normals: list = normals,
                bad: list = bad) -> int:
            """Append one row (``fraction`` None: the inner boundary); returns its first vertex index."""

            first = len(positions)
            inner, outer = zone.inner, zone.outer
            for j in range(0, count, stride):
                dx, dy = dirs[j]
                lo = 0.0 if inner is None else inner[j]
                t = lo if fraction is None else lo + (outer[j] - lo) * fraction
                a, b = cx + dx * t, cy + dy * t
                if zone.hold is not None:  # the flat flange: the sag of the edge point along this direction
                    held = min(t, zone.hold[j])
                    hit = profile.sag_normal(cx + dx * held, cy + dy * held)
                    if hit is None:
                        bad.append(len(positions))
                        hit = profile.sag_normal(*profile.domain_point(cx + dx * held, cy + dy * held)) or (0.0, 0.0, 0.0, 1.0)
                    z = hit[0]
                    if coaxial:
                        positions.append((a, b, z0 + z))
                        normals.append((0.0, 0.0, facing))
                    else:
                        positions.append(placement.point(a, b, z))
                        normals.append(placement.vector(0.0, 0.0, facing))
                    continue
                hit = profile.sag_normal(a, b)
                if hit is None:
                    bad.append(len(positions))
                    a, b = profile.domain_point(a, b)
                    hit = profile.sag_normal(a, b) or (0.0, 0.0, 0.0, 1.0)
                z, nx, ny, nz = hit
                if coaxial:
                    positions.append((a, b, z0 + z))
                    normals.append((nx * facing, ny * facing, nz * facing))
                else:
                    positions.append(placement.point(a, b, z))
                    normals.append(placement.vector(nx * facing, ny * facing, nz * facing))
            return first

        strides = zone.strides or (1,) * len(zone.stations)
        triangles: list[tuple[int, int, int]] = []
        if zone.inner is None:
            hit = profile.sag_normal(cx, cy) or (0.0, 0.0, 0.0, 1.0)
            positions.append((cx, cy, z0 + hit[0]) if coaxial else placement.point(cx, cy, hit[0]))
            normals.append(placement.vector(hit[1] * facing, hit[2] * facing, hit[3] * facing))
            previous, previous_stride = -1, strides[0]
        else:
            previous, previous_stride = row(None, zone.inner_stride), zone.inner_stride
        for fraction, stride in zip(zone.stations, strides, strict=True):
            current = row(fraction, stride)
            if previous < 0:
                n = count // stride
                triangles.extend((0, current + j, current + (j + 1) % n) for j in range(n))
            else:
                _stitch(triangles, previous, previous_stride, current, stride, count)
            previous, previous_stride = current, stride
        out.outline = positions[previous:] if previous_stride == 1 else []
        out.rows.append(out.outline)
        bad_set = set(bad)
        good: list[int] = []
        poor: list[int] = []
        for tri in triangles:
            pa, pb, pc = positions[tri[0]], positions[tri[1]], positions[tri[2]]
            if pa == pb or pb == pc or pc == pa:  # a zone of zero width along a direction (an aperture flush with the outline)
                continue
            target = poor if bad_set and (tri[0] in bad_set or tri[1] in bad_set or tri[2] in bad_set) else good
            target.extend(tri if facing > 0 else (tri[0], tri[2], tri[1]))
        if not poor and len(good) == 3 * len(triangles):
            out.meshes.append((zone.material, g.Mesh(positions=positions, normals=normals, indices=good)))
        else:
            for material, indices in ((zone.material, good), (UNDEFINED_MATERIAL, poor)):
                if indices:
                    out.meshes.append((material, _compact(positions, normals, indices)))
        out.undefined = out.undefined or bool(bad)
    return out


def _compact(positions: list[g.Vec3], normals: list[g.Vec3], indices: list[int]) -> g.Mesh:
    """The sub-mesh of ``indices``, with only the vertices it uses (in first-use order)."""

    remap: dict[int, int] = {}
    m = g.Mesh()
    for i in indices:
        j = remap.get(i)
        if j is None:
            j = remap[i] = len(m.positions)
            m.positions.append(positions[i])
            m.normals.append(normals[i])
        m.indices.append(j)
    return m


def row_strides(required: Sequence[int], count: int) -> list[int]:
    """Per row (inside out), the stride over ``count`` directions: the coarsest power of two that still gives the row its
    ``required`` directions (and at least ``MIN_DIRECTIONS``), then evened out so a row is never coarser than a row
    inside it and the stride at most doubles from one row to the next inward."""

    strides = []
    for need in required:
        stride = 1
        while count % (2 * stride) == 0 and count // (2 * stride) >= max(need, MIN_DIRECTIONS):
            stride *= 2
        strides.append(stride)
    changed = True
    while changed:
        changed = False
        for r in range(len(strides) - 1, 0, -1):  # inward: at most twice the stride outside
            if strides[r - 1] > 2 * strides[r]:
                strides[r - 1] = 2 * strides[r]
                changed = True
        for r in range(1, len(strides)):  # outward: never coarser than inside
            if strides[r] > strides[r - 1]:
                strides[r] = strides[r - 1]
                changed = True
    return strides


def base_count(required: int) -> int:
    """The number of directions of a mesh that needs ``required``: the smallest ``c * 2^k >= required`` with ``c`` in 8..15,
    so its rows can thin out by halving towards the centre."""

    k = 0
    while BASE_MAX * (1 << k) < required:
        k += 1
    c = max(BASE_MIN, -(-required // (1 << k)))
    return c * (1 << k)


def max_deviation(meshes: Sequence[tuple[str, g.Mesh]], profile: Profile, placement: Placement) -> float:
    """The largest first-order normal distance between the drawn triangles and the exact profile: at every triangle's
    centroid and edge midpoints, ``|z_drawn - z_profile| * cos(slope)`` in the surface frame (undefined triangles are
    skipped)."""

    worst = 0.0
    o, u, v, w = placement.origin, placement.u, placement.v, placement.w
    coaxial = placement.is_coaxial
    for material, mesh in meshes:
        if material in (UNDEFINED_MATERIAL, FLANGE_MATERIAL):
            continue
        if coaxial:
            local = [(p[0], p[1], p[2] - o[2]) for p in mesh.positions]
        else:
            local = []
            for p in mesh.positions:
                d = (p[0] - o[0], p[1] - o[1], p[2] - o[2])
                local.append((d[0] * u[0] + d[1] * u[1] + d[2] * u[2], d[0] * v[0] + d[1] * v[1] + d[2] * v[2],
                              d[0] * w[0] + d[1] * w[1] + d[2] * w[2]))
        idx = mesh.indices
        for t in range(0, len(idx), 3):
            ax, ay, az = local[idx[t]]
            bx, by, bz = local[idx[t + 1]]
            cx, cy, cz = local[idx[t + 2]]
            for x, y, z in (((ax + bx + cx) / 3.0, (ay + by + cy) / 3.0, (az + bz + cz) / 3.0), ((ax + bx) / 2.0, (ay + by) / 2.0, (az + bz) / 2.0),
                            ((bx + cx) / 2.0, (by + cy) / 2.0, (bz + cz) / 2.0), ((cx + ax) / 2.0, (cy + ay) / 2.0, (cz + az) / 2.0)):
                hit = profile.sag_normal(x, y)
                if hit is None:
                    continue
                d = abs(z - hit[0]) * hit[3]
                if d > worst:
                    worst = d
    return worst


def wall(bottom: Sequence[g.Vec3], top: Sequence[g.Vec3], dirs: Sequence[Vec2], round_outline: bool) -> g.Mesh:
    """The edge wall between two outline rows (one point per direction, closed loop): a round outline with smooth radial
    normals, a polygon with each side's own flat normal."""

    m = g.Mesh()
    count = len(bottom)
    for j in range(count):
        k = (j + 1) % count
        if round_outline:
            nj, nk = (dirs[j][0], dirs[j][1], 0.0), (dirs[k][0], dirs[k][1], 0.0)
        else:
            ex, ey = bottom[k][0] - bottom[j][0], bottom[k][1] - bottom[j][1]
            length = math.sqrt(ex * ex + ey * ey)
            if length <= SAME_POINT_MM:
                continue
            nj = nk = (ey / length, -ex / length, 0.0)
        base = len(m.positions)
        m.positions.extend((bottom[j], top[j], bottom[k], top[k]))
        m.normals.extend((nj, nj, nk, nk))
        m.indices.extend((base, base + 2, base + 3, base, base + 3, base + 1))
    return m


def grid_mesh(profile: Profile, placement: Placement, xs: Sequence[float], ys: Sequence[float], material_of: Callable[[float, float], str],
              facing: float) -> SurfaceMesh:
    """A surface over the rectangular grid ``xs`` x ``ys`` (local x, y; both ascending), each cell's two triangles drawn in
    ``material_of(cell centre)``; ``outline`` is the grid's boundary loop, counter-clockwise from the corner
    ``(xs[0], ys[0])``. Undefined vertices are handled as in ``star_mesh``."""

    out = SurfaceMesh()
    nx, ny = len(xs), len(ys)
    positions: list[g.Vec3] = []
    normals: list[g.Vec3] = []
    bad: set[int] = set()
    coaxial = placement.is_coaxial
    z0 = placement.origin[2]
    for y in ys:
        for x in xs:
            a, b = x, y
            hit = profile.sag_normal(a, b)
            if hit is None:
                bad.add(len(positions))
                a, b = profile.domain_point(a, b)
                hit = profile.sag_normal(a, b) or (0.0, 0.0, 0.0, 1.0)
            z, gx, gy, gz = hit
            if coaxial:
                positions.append((a, b, z0 + z))
                normals.append((gx * facing, gy * facing, gz * facing))
            else:
                positions.append(placement.point(a, b, z))
                normals.append(placement.vector(gx * facing, gy * facing, gz * facing))
    by_material: dict[str, list[int]] = {}
    for j in range(ny - 1):
        for i in range(nx - 1):
            a, b, c, d = j * nx + i, j * nx + i + 1, (j + 1) * nx + i + 1, (j + 1) * nx + i
            name = material_of(0.5 * (xs[i] + xs[i + 1]), 0.5 * (ys[j] + ys[j + 1]))
            if bad and (a in bad or b in bad or c in bad or d in bad):
                name = UNDEFINED_MATERIAL
            tris = (a, b, c, a, c, d) if facing > 0 else (a, c, b, a, d, c)
            by_material.setdefault(name, []).extend(tris)
    for name, indices in by_material.items():
        out.meshes.append((name, _compact(positions, normals, indices)))
    loop = [i for i in range(nx - 1)] + [j * nx + nx - 1 for j in range(ny - 1)]
    loop += [(ny - 1) * nx + i for i in range(nx - 1, 0, -1)] + [j * nx for j in range(ny - 1, 0, -1)]
    out.outline = [positions[k] for k in loop]
    out.undefined = bool(bad)
    return out


def substrate_face_z(vertex_z: float, z_min: float, z_max: float, reflects: bool) -> float:
    """The flat far face of the unmodelled substrate of a single coated surface whose design body spans ``[z_min, z_max]``
    (element frame). Behind a reflecting surface the substrate lies on its ``-Z`` side (light meets the coated face from
    ``+Z``, as the face-normal frame places it), so its far face is the body's ``-Z`` face and the air in front of a
    concave mirror is never glass; behind any other coated surface it is the body face farther from the vertex."""

    if reflects:
        return z_min
    return z_min if vertex_z - z_min >= z_max - vertex_z else z_max
