"""Stdlib pose math and display meshes of the Optical Simulation View (no NumPy: presentation only, Pyodide-suitable).

Pose convention (the same as every OpenSci contract): Hamilton ``(w, x, y, z)`` unit quaternion, active rotation,
``x_global = R(q) x_local + position_mm``. Meshes are plain triangle lists in a local frame; they carry no optics.

Determinism: every vertex is computed with ``+ - * /`` and ``math.sqrt`` only (both correctly rounded under IEEE 754),
so CPython and Pyodide write the same bits. Points on a circle use ``turn_cos_sin`` (an octant reduction in integers and
a fixed Taylor polynomial) instead of the platform's ``math.cos`` / ``math.sin``; ``pseudo_angle`` orders directions
without ``math.atan2``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]

# Display tessellation (view settings, not physics).
CIRCLE_SEGMENTS = 48
PROFILE_RINGS = 12  # rings from the vertex to the edge of a curved optical surface
TUBE_SIDES = 6
# perpendicular_basis switches its helper axis when the axis is this close to +-X (a deterministic, well-conditioned choice).
HELPER_AXIS_SWITCH = 0.9
FULL_TURN = 2 * math.pi


# Taylor coefficients of sin / cos about 0, enough for full double precision on [0, pi/4].
_SIN_TERMS = tuple((1.0 if k % 2 == 0 else -1.0) / math.factorial(2 * k + 1) for k in range(11))
_COS_TERMS = tuple((1.0 if k % 2 == 0 else -1.0) / math.factorial(2 * k) for k in range(11))
_EIGHTH_TURN = math.pi / 4.0
# (cos, sin) of k quarter turns, exact.
_QUARTER_TURNS = ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0))


def _taylor(terms: tuple[float, ...], p: float) -> float:
    value = 0.0
    for c in reversed(terms):
        value = value * p + c
    return value


def turn_cos_sin(k: int, n: int) -> tuple[float, float]:
    """``(cos, sin)`` of the angle ``2 pi k / n`` from ``+ - * /`` only: the angle is reduced to an octant in integer
    arithmetic, the remainder (at most pi/4) goes through fixed Taylor polynomials and the octant is restored by an exact
    quarter-turn rotation. Identical bits on every IEEE 754 runtime (no platform libm), and exactly symmetric."""

    k %= n
    octant, rem = divmod(8 * k, n)  # angle = (octant + rem / n) * pi / 4
    if octant % 2 == 0:
        phi, quarter, sign = _EIGHTH_TURN * rem / n, octant // 2, 1.0
    else:
        phi, quarter, sign = _EIGHTH_TURN * (n - rem) / n, (octant + 1) // 2, -1.0
    p = phi * phi
    c0, s0 = _taylor(_COS_TERMS, p), phi * _taylor(_SIN_TERMS, p)
    s0 *= sign  # angle = quarter * pi/2 + sign * phi
    cq, sq = _QUARTER_TURNS[quarter % 4]
    return cq * c0 - sq * s0, sq * c0 + cq * s0


def pseudo_angle(x: float, y: float) -> float:
    """A strictly increasing function of the polar angle of ``(x, y)`` in ``[0, 4)`` (no trigonometry)."""

    if y >= 0.0:
        return y / (x + y) if x >= 0.0 else 1.0 + (-x) / (y - x)
    return 2.0 + (-y) / (-x - y) if x < 0.0 else 3.0 + x / (x - y)


def quat_to_matrix(q: Quat) -> tuple[Vec3, Vec3, Vec3]:
    """Row-major 3x3 rotation matrix of a (normalized) Hamilton quaternion."""
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
        (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
        (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
    )


def quat_multiply(a: Quat, b: Quat) -> Quat:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def matrix_to_quat(columns: tuple[Vec3, Vec3, Vec3]) -> Quat:
    """Unit quaternion (w >= 0) of the rotation whose matrix has the given columns (Shepperd's method)."""
    (m00, m10, m20), (m01, m11, m21), (m02, m12, m22) = columns
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = (0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s)
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        q = ((m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s)
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        q = ((m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s)
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        q = ((m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s)
    return q if q[0] >= 0.0 else (-q[0], -q[1], -q[2], -q[3])


def rotate(q: Quat, v: Vec3) -> Vec3:
    m = quat_to_matrix(q)
    return (dot(m[0], v), dot(m[1], v), dot(m[2], v))


def apply_pose(position: Vec3, q: Quat, local: Vec3) -> Vec3:
    return add(rotate(q, local), position)


def unit_quat(q: Quat) -> Quat:
    """``q / |q|``, sign kept (the components as given when the length is exactly 1)."""
    n = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    return q if n == 1.0 else (q[0] / n, q[1] / n, q[2] / n, q[3] / n)


def compose_pose(parent_position: Vec3, parent_q: Quat, local_position: Vec3, local_q: Quat) -> tuple[Vec3, Quat]:
    """A pose stated in the frame ``(parent_position, parent_q)``, in that frame's parent: computed from the rotations both
    quaternions state (``q / |q|``), the composed quaternion unit."""
    return apply_pose(parent_position, parent_q, local_position), unit_quat(quat_multiply(unit_quat(parent_q), unit_quat(local_q)))


def same_rotation(a: Quat, b: Quat, tol: float) -> bool:
    """``a`` and ``b`` describe the same rotation (``q`` and ``-q`` are equal)."""
    d = abs(sum(x * y for x, y in zip(a, b, strict=True)))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return abs(d / (na * nb) - 1.0) <= tol


def add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def scale(a: Vec3, s: float) -> Vec3:
    return (a[0] * s, a[1] * s, a[2] * s)


def dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a: Vec3, b: Vec3) -> Vec3:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def norm(a: Vec3) -> float:
    return math.sqrt(dot(a, a))


def unit(a: Vec3) -> Vec3:
    n = norm(a)
    return scale(a, 1.0 / n)


def perpendicular_basis(axis: Vec3) -> tuple[Vec3, Vec3]:
    """Two unit vectors completing a right-handed frame with unit ``axis`` (deterministic choice)."""
    helper = (1.0, 0.0, 0.0) if abs(axis[0]) < HELPER_AXIS_SWITCH else (0.0, 1.0, 0.0)
    e1 = unit(cross(helper, axis))
    return e1, cross(axis, e1)


@dataclass
class Mesh:
    """Indexed triangle mesh with per-vertex normals."""

    positions: list[Vec3] = field(default_factory=list)
    normals: list[Vec3] = field(default_factory=list)
    indices: list[int] = field(default_factory=list)

    def extend(self, other: Mesh) -> None:
        base = len(self.positions)
        self.positions.extend(other.positions)
        self.normals.extend(other.normals)
        self.indices.extend(i + base for i in other.indices)

    @property
    def empty(self) -> bool:
        return not self.indices


def unit_circle(segments: int) -> list[tuple[float, float]]:
    """``segments`` evenly spaced points of the unit circle, counter-clockwise from +X (``turn_cos_sin``)."""

    return [turn_cos_sin(k, segments) for k in range(segments)]


def circle_segments(radius: float, tolerance: float, *, minimum: int = 8) -> int:
    """The number of sides of a regular polygon inscribed in a circle of ``radius`` whose sides stay within
    ``tolerance`` of the circle (sagitta ``r (1 - cos(pi / n)) <= r (pi / n)^2 / 2 <= tolerance``), rounded up to a multiple
    of 8 (so the polygon keeps the circle's quarter-turn symmetry)."""

    n = math.ceil(math.pi * math.sqrt(max(radius, 0.0) / (2.0 * tolerance))) if radius > 0.0 else minimum
    n = max(n, minimum)
    return n + (-n) % 8


def _ring(radius: float, z: float, segments: int) -> list[Vec3]:
    return [(radius * c, radius * s, z) for c, s in unit_circle(segments)]


def disc(radius: float, z: float, *, facing: float = 1.0, segments: int = CIRCLE_SEGMENTS) -> Mesh:
    """Filled disc in the local plane ``z``, normal ``(0, 0, facing)``."""
    m = Mesh()
    m.positions.append((0.0, 0.0, z))
    m.positions.extend(_ring(radius, z, segments))
    m.normals = [(0.0, 0.0, facing)] * (segments + 1)
    for k in range(segments):
        a, b = 1 + k, 1 + (k + 1) % segments
        m.indices.extend((0, a, b) if facing > 0 else (0, b, a))
    return m


def cylinder(radius: float, z_min: float, z_max: float, *, segments: int = CIRCLE_SEGMENTS) -> Mesh:
    """Closed cylinder about local +Z: side wall plus both end discs."""
    m = Mesh()
    bottom, top = _ring(radius, z_min, segments), _ring(radius, z_max, segments)
    for k, (c, s) in enumerate(unit_circle(segments)):
        m.positions.extend((bottom[k], top[k]))
        m.normals.extend(((c, s, 0.0), (c, s, 0.0)))
    for k in range(segments):
        b0, t0, b1, t1 = 2 * k, 2 * k + 1, 2 * ((k + 1) % segments), 2 * ((k + 1) % segments) + 1
        m.indices.extend((b0, b1, t1, b0, t1, t0))
    m.extend(disc(radius, z_min, facing=-1.0, segments=segments))
    m.extend(disc(radius, z_max, facing=1.0, segments=segments))
    return m


def rectangle(half_u: float, half_v: float, z: float = 0.0, *, center: tuple[float, float] = (0.0, 0.0)) -> Mesh:
    cu, cv = center
    m = Mesh()
    m.positions = [(cu - half_u, cv - half_v, z), (cu + half_u, cv - half_v, z), (cu + half_u, cv + half_v, z), (cu - half_u, cv + half_v, z)]
    m.normals = [(0.0, 0.0, 1.0)] * 4
    m.indices = [0, 1, 2, 0, 2, 3]
    return m


def sag(radius_mm: float | None, r_mm: float) -> float:
    """Axial sag (z offset from the vertex) of a sphere of signed ``radius_mm`` (positive: centre on +Z) at transverse
    radius ``r_mm <= |radius_mm|``; 0 for a plane (``None``). ``math.sqrt`` is correctly rounded on every runtime."""
    if radius_mm is None:
        return 0.0
    root = math.sqrt(radius_mm * radius_mm - r_mm * r_mm)
    return radius_mm - root if radius_mm > 0 else radius_mm + root


def surface_rim(vertex_z: float, radius_mm: float | None, clear_radius: float, edge_radius: float) -> tuple[float, float]:
    """``(profile radius, rim z)`` of an optical surface drawn out to the element edge.

    A plane, or a sphere no steeper than the edge (``|R| >= edge``), follows its own profile to the edge radius. A steeper
    sphere follows its profile to its clear radius; ``surface_cap`` + ``annulus`` then close it with a flat ring out to the
    edge at the rim height (a display simplification of the bevel, outside the traced clear aperture).
    """
    r = edge_radius if radius_mm is None or abs(radius_mm) >= edge_radius else clear_radius
    return r, vertex_z + sag(radius_mm, r)


def surface_cap(vertex_z: float, radius_mm: float | None, profile_radius: float, *, facing: float = 1.0,
                rings: int = PROFILE_RINGS, segments: int = CIRCLE_SEGMENTS) -> Mesh:
    """An optical surface as a surface of revolution about local +Z: the vertex at ``(0, 0, vertex_z)`` and ``rings``
    rings out to ``profile_radius`` on the sphere of signed ``radius_mm`` (a flat disc for a plane). Normals point to
    ``(0, 0, facing)`` at the vertex and follow the sphere."""
    if radius_mm is None:
        return disc(profile_radius, vertex_z, facing=facing, segments=segments)
    m = Mesh()
    m.positions.append((0.0, 0.0, vertex_z))
    m.normals.append((0.0, 0.0, facing))
    centre_z = vertex_z + radius_mm
    unit_ring = _ring(1.0, 0.0, segments)
    for i in range(1, rings + 1):
        r = profile_radius * i / rings
        z = vertex_z + sag(radius_mm, r)
        for c, s, _ in unit_ring:
            x, y = r * c, r * s
            m.positions.append((x, y, z))
            # the sphere normal, oriented to +Z at the vertex whatever the sign of the radius, then to ``facing``
            m.normals.append((-x / radius_mm * facing, -y / radius_mm * facing, -(z - centre_z) / radius_mm * facing))
    for k in range(segments):
        a, b = 1 + k, 1 + (k + 1) % segments
        m.indices.extend((0, a, b) if facing > 0 else (0, b, a))
    for i in range(1, rings):
        inner, outer = 1 + (i - 1) * segments, 1 + i * segments
        for k in range(segments):
            a, b = inner + k, inner + (k + 1) % segments
            c, d = outer + (k + 1) % segments, outer + k
            m.indices.extend((a, d, c, a, c, b) if facing > 0 else (a, c, d, a, b, c))
    return m


def annulus(inner_radius: float, outer_radius: float, z: float, *, facing: float = 1.0, segments: int = CIRCLE_SEGMENTS) -> Mesh:
    """Flat ring in the local plane ``z`` between two radii, normal ``(0, 0, facing)``."""
    m = Mesh()
    m.positions = _ring(inner_radius, z, segments) + _ring(outer_radius, z, segments)
    m.normals = [(0.0, 0.0, facing)] * (2 * segments)
    for k in range(segments):
        a, b = k, (k + 1) % segments
        c, d = segments + (k + 1) % segments, segments + k
        m.indices.extend((a, d, c, a, c, b) if facing > 0 else (a, c, d, a, b, c))
    return m


def side_wall(radius: float, z_bottom: float, z_top: float, *, segments: int = CIRCLE_SEGMENTS) -> Mesh:
    """Open cylinder wall about local +Z from ``z_bottom`` to ``z_top`` (empty when ``z_top <= z_bottom``)."""
    m = Mesh()
    if not z_top > z_bottom:
        return m
    bottom, top = _ring(radius, z_bottom, segments), _ring(radius, z_top, segments)
    for k, (c, s) in enumerate(unit_circle(segments)):
        m.positions.extend((bottom[k], top[k]))
        m.normals.extend(((c, s, 0.0), (c, s, 0.0)))
    for k in range(segments):
        b0, t0, b1, t1 = 2 * k, 2 * k + 1, 2 * ((k + 1) % segments), 2 * ((k + 1) % segments) + 1
        m.indices.extend((b0, b1, t1, b0, t1, t0))
    return m


def box(x0: float, x1: float, y0: float, y1: float, z0: float, z1: float) -> Mesh:
    """Axis-aligned closed box ``[x0, x1] x [y0, y1] x [z0, z1]`` with flat outward normals (no trigonometry)."""
    m = Mesh()
    faces = (
        ((1.0, 0.0, 0.0), ((x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1))),
        ((-1.0, 0.0, 0.0), ((x0, y1, z0), (x0, y0, z0), (x0, y0, z1), (x0, y1, z1))),
        ((0.0, 1.0, 0.0), ((x1, y1, z0), (x0, y1, z0), (x0, y1, z1), (x1, y1, z1))),
        ((0.0, -1.0, 0.0), ((x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1))),
        ((0.0, 0.0, 1.0), ((x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1))),
        ((0.0, 0.0, -1.0), ((x0, y1, z0), (x1, y1, z0), (x1, y0, z0), (x0, y0, z0))),
    )
    for normal, corners in faces:
        base = len(m.positions)
        m.positions.extend(corners)
        m.normals.extend([normal] * 4)
        m.indices.extend((base, base + 1, base + 2, base, base + 2, base + 3))
    return m


def tube(points: list[Vec3], radius: float, *, sides: int = TUBE_SIDES) -> Mesh:
    """Open prism tube along a polyline (global coordinates), one straight piece per segment."""
    m = Mesh()
    for p0, p1 in zip(points, points[1:], strict=False):
        d = sub(p1, p0)
        if norm(d) <= 0.0:
            continue
        axis = unit(d)
        e1, e2 = perpendicular_basis(axis)
        base = len(m.positions)
        for c, s in unit_circle(sides):
            n = add(scale(e1, c), scale(e2, s))
            off = scale(n, radius)
            m.positions.extend((add(p0, off), add(p1, off)))
            m.normals.extend((n, n))
        for k in range(sides):
            a0, a1 = base + 2 * k, base + 2 * k + 1
            b0, b1 = base + 2 * ((k + 1) % sides), base + 2 * ((k + 1) % sides) + 1
            m.indices.extend((a0, b0, b1, a0, b1, a1))
    return m


def transformed(mesh: Mesh, position: Vec3, q: Quat) -> Mesh:
    out = Mesh(indices=list(mesh.indices))
    out.positions = [apply_pose(position, q, p) for p in mesh.positions]
    out.normals = [rotate(q, n) for n in mesh.normals]
    return out
