"""Rigid transforms for the assembly checker, in plain Python floats.

Only ``+ - * /`` and ``sqrt`` (both correctly rounded by IEEE 754) are used on the checked path, so CPython on any
platform and Pyodide produce bit-identical numbers and therefore identical result hashes. NumPy is deliberately
not used here: its vectorised reductions may change the summation order between builds.

Conventions are the Assembly Preview scene's: ``x_world = R(q) x_local + position_mm`` with ``q = (w, x, y, z)`` a
unit Hamilton quaternion of an active rotation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

Vec3 = tuple[float, float, float]
Mat3 = tuple[Vec3, Vec3, Vec3]  # rows

IDENTITY: Mat3 = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def quaternion_norm(q: tuple[float, float, float, float]) -> float:
    return math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])


def quaternion_matrix(q: tuple[float, float, float, float]) -> Mat3:
    """Rotation matrix of the rotation ``q`` states, ``q / |q|`` (the components as given when the length is exactly 1; the
    caller refuses a norm beyond its tolerance, and one within it is taken by its direction)."""

    n = quaternion_norm(q)
    w, x, y, z = q if n == 1.0 or n == 0.0 or not math.isfinite(n) else (q[0] / n, q[1] / n, q[2] / n, q[3] / n)
    return (
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)),
        (2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)),
        (2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)),
    )


def mat_vec(m: Mat3, v: Vec3) -> Vec3:
    return (
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    )


def mat_mul(a: Mat3, b: Mat3) -> Mat3:
    cols = tuple(zip(*b))
    return tuple(tuple(r[0] * c[0] + r[1] * c[1] + r[2] * c[2] for c in cols) for r in a)  # type: ignore[return-value]


def transpose(m: Mat3) -> Mat3:
    return tuple(zip(*m))  # type: ignore[return-value]


def sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a: Vec3, b: Vec3) -> Vec3:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def length(v: Vec3) -> float:
    return math.sqrt(dot(v, v))


@dataclass(frozen=True)
class Rigid:
    """``x -> rotation @ x + translation``."""

    rotation: Mat3 = IDENTITY
    translation: Vec3 = (0.0, 0.0, 0.0)

    @classmethod
    def from_pose(cls, position_mm: Vec3, quaternion_wxyz: tuple[float, float, float, float]) -> Rigid:
        return cls(quaternion_matrix(quaternion_wxyz), (float(position_mm[0]), float(position_mm[1]), float(position_mm[2])))

    def apply(self, p: Vec3) -> Vec3:
        r, t = self.rotation, self.translation
        return (
            r[0][0] * p[0] + r[0][1] * p[1] + r[0][2] * p[2] + t[0],
            r[1][0] * p[0] + r[1][1] * p[1] + r[1][2] * p[2] + t[1],
            r[2][0] * p[0] + r[2][1] * p[1] + r[2][2] * p[2] + t[2],
        )

    def apply_all(self, points: tuple[Vec3, ...]) -> tuple[Vec3, ...]:
        return tuple(self.apply(p) for p in points)

    def compose(self, inner: Rigid) -> Rigid:
        """``self o inner``: first ``inner``, then ``self``."""

        return Rigid(mat_mul(self.rotation, inner.rotation), self.apply(inner.translation))

    def inverse(self) -> Rigid:
        rt = transpose(self.rotation)
        t = mat_vec(rt, self.translation)
        return Rigid(rt, (-t[0], -t[1], -t[2]))

    def axis(self, index: int) -> Vec3:
        """The frame's local axis ``index`` expressed in the outer frame (a column of the rotation)."""

        return (self.rotation[0][index], self.rotation[1][index], self.rotation[2][index])


def rotation_gap(a: Mat3, b: Mat3) -> float:
    """``sin(theta / 2)`` of the rotation between ``a`` and ``b`` (0 = equal, 1 = half turn), from the trace.

    ``trace(a^T b) = 1 + 2 cos(theta)`` and ``sin^2(theta / 2) = (3 - trace) / 4``; clamped against rounding.
    """

    trace = sum(a[0][i] * b[0][i] + a[1][i] * b[1][i] + a[2][i] * b[2][i] for i in range(3))
    return math.sqrt(min(1.0, max(0.0, (3.0 - trace) / 4.0)))


def half_angle_sin(angle_deg: float) -> float:
    """``sin(angle / 2)`` for comparing against ``rotation_gap`` (a fixed tolerance, computed once per check)."""

    return math.sin(math.radians(angle_deg) / 2.0)


def gap_degrees(gap: float) -> float:
    """The rotation angle in degrees for a ``rotation_gap`` value (reporting only, rounded by the caller)."""

    return math.degrees(2.0 * math.asin(min(1.0, gap)))


def frame_along(origin: Vec3, direction: Vec3) -> Rigid | None:
    """An orthonormal frame at ``origin`` whose local +Z is ``direction`` (``None`` for a zero direction)."""

    norm = length(direction)
    if norm == 0.0:
        return None
    w = (direction[0] / norm, direction[1] / norm, direction[2] / norm)
    helper = (1.0, 0.0, 0.0) if abs(w[0]) <= abs(w[1]) and abs(w[0]) <= abs(w[2]) else (
        (0.0, 1.0, 0.0) if abs(w[1]) <= abs(w[2]) else (0.0, 0.0, 1.0))
    u = cross(helper, w)
    un = length(u)
    u = (u[0] / un, u[1] / un, u[2] / un)
    v = cross(w, u)
    return Rigid(((u[0], v[0], w[0]), (u[1], v[1], w[1]), (u[2], v[2], w[2])), origin)
