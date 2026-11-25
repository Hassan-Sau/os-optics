"""Rigid-body arithmetic of the composition operations, with ``+ - * /`` and ``sqrt`` only.

Every number a composition operation writes into a Project is hashed exactly (osc1), and the same operation runs in
CPython and in the browser runtime (Pyodide). Those IEEE-754 operations are correctly rounded on both, while the
platform ``sin`` / ``cos`` may differ in the last bit, so they are never called here:

* a rotation by a multiple of 90 degrees uses exact half-angle values (0, +-1, +-sqrt(1/2)), and a rotation whose matrix
  is a permutation with signs (every quarter turn about the frame axes) moves points exactly;
* any other angle uses a fixed reduction to [0, 45] degrees and Taylor polynomials of degree 19 (sine) and 18 (cosine)
  in Horner form (error below 1e-17 on that interval);
* every rotation is computed from the stated quaternion's direction ``q / |q|`` (``unit``), so a pose stated within the
  Project's 1e-6 tolerance of unit length moves points as its unit quaternion does;
* every rotated pose's quaternion is renormalized (``q / sqrt(q.q)``), so repeated transforms never drift from unit length,
  and written with ``w >= 0`` (``q`` and ``-q`` are the same rotation), so equal poses hash alike.

Conventions: quaternions are ``(w, x, y, z)``, Hamilton, active rotation (``PoseV1``); a pose maps local coordinates to
its parent frame by ``x_parent = R(q) x_local + position``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

Vector = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]

IDENTITY: Quaternion = (1.0, 0.0, 0.0, 0.0)
SQRT_HALF = math.sqrt(0.5)
FULL_TURN_DEG = 360.0
QUARTER_TURN_DEG = 90.0
EIGHTH_TURN_DEG = 45.0
#: A rotation matrix entry this close to 0 or +-1 makes the rotation a quarter turn (moved by an exact permutation).
PERMUTATION_TOLERANCE = 1e-12
#: The ``|1 - |q||`` up to which a quaternion an operation is given is used unchanged (the Engine's tolerance); beyond it
#: the quaternion is normalized (``stated_quaternion``).
UNIT_TOLERANCE = 1e-6
#: A rotation axis whose length is outside this range is scaled by its largest component before it is normalized (its
#: squares would otherwise overflow or lose precision).
SCALE_FREE_NORM = (1e-100, 1e100)
_DEG_TO_RAD = math.pi / 180.0
# 1/n! (n odd for the sine, even for the cosine), highest order first for Horner's scheme
_SIN = tuple((-1.0 if k % 2 else 1.0) / math.factorial(2 * k + 1) for k in range(9, -1, -1))
_COS = tuple((-1.0 if k % 2 else 1.0) / math.factorial(2 * k) for k in range(9, -1, -1))
#: (cos, sin) of the multiples of 45 degrees.
_EXACT = {0: (1.0, 0.0), 1: (SQRT_HALF, SQRT_HALF), 2: (0.0, 1.0), 3: (-SQRT_HALF, SQRT_HALF), 4: (-1.0, 0.0),
          5: (-SQRT_HALF, -SQRT_HALF), 6: (0.0, -1.0), 7: (SQRT_HALF, -SQRT_HALF)}


def _poly(coefficients: tuple[float, ...], x2: float) -> float:
    value = 0.0
    for c in coefficients:
        value = value * x2 + c
    return value


def _cos_sin_small(t_deg: float) -> tuple[float, float]:
    """(cos, sin) of ``t_deg`` in [0, 45] degrees."""

    x = t_deg * _DEG_TO_RAD
    x2 = x * x
    return _poly(_COS, x2), x * _poly(_SIN, x2)


def cos_sin_deg(angle_deg: float) -> tuple[float, float]:
    """(cos, sin) of an angle in degrees, without libm: exact for multiples of 45 degrees, else a fixed polynomial
    after reduction to [0, 45] degrees (module docstring)."""

    if not math.isfinite(angle_deg):
        raise ValueError("the angle must be a finite number")
    r = math.fmod(float(angle_deg), FULL_TURN_DEG)  # exact
    if r < 0.0:
        r = r + FULL_TURN_DEG
    if r >= FULL_TURN_DEG:
        r = 0.0
    if math.fmod(r, EIGHTH_TURN_DEG) == 0.0:
        return _EXACT[int(r / EIGHTH_TURN_DEG)]
    quadrant = int(r // QUARTER_TURN_DEG)
    a = r - QUARTER_TURN_DEG * quadrant  # in (0, 90)
    if a <= EIGHTH_TURN_DEG:
        c, s = _cos_sin_small(a)
    else:
        s, c = _cos_sin_small(QUARTER_TURN_DEG - a)
    return ((c, s), (-s, c), (-c, -s), (s, -c))[quadrant]


def norm(v: Sequence[float]) -> float:
    return math.sqrt(sum(c * c for c in v))


def normalize_quaternion(q: Sequence[float]) -> Quaternion:
    """``q / |q|`` with the sign that makes ``w`` positive (``q`` and ``-q`` are the same rotation; when ``w`` is zero, the
    first non-zero of ``x, y, z`` is made positive), so equal rotations are written alike whichever operation made them."""

    w, x, y, z = (float(c) for c in q)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n == 0.0 or not math.isfinite(n):
        raise ValueError("the quaternion must be non-zero and finite")
    unit = (w / n, x / n, y / n, z / n)
    leading = next((c for c in unit if c != 0.0), 0.0)
    return tuple(0.0 - c for c in unit) if leading < 0.0 else unit  # type: ignore[return-value]  # (0.0 - c: never -0.0)


def unit(q: Sequence[float]) -> Quaternion:
    """The rotation ``q`` states, ``q / |q|``, sign kept (the components as given when the length is exactly 1, so a unit
    quaternion moves no bit); ``ValueError`` for a zero or non-finite one. Every rotation is computed from it: a quaternion
    stated within ``UNIT_TOLERANCE`` of unit length moves points as its unit quaternion does (to rounding)."""

    w, x, y, z = (float(c) for c in q)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n == 0.0 or not math.isfinite(n):
        raise ValueError("the quaternion must be non-zero and finite")
    return (w, x, y, z) if n == 1.0 else (w / n, x / n, y / n, z / n)


def stated_quaternion(q: Sequence[float]) -> Quaternion:
    """A quaternion a caller states, taken by its direction: within ``UNIT_TOLERANCE`` of unit length its components as
    given, else ``normalize_quaternion(q)``; ``ValueError`` for a zero or non-finite one."""

    n = norm(q)
    if n == 0.0 or not math.isfinite(n):
        raise ValueError("the quaternion must be non-zero and finite")
    if abs(n - 1.0) <= UNIT_TOLERANCE:
        return tuple(float(c) for c in q)  # type: ignore[return-value]
    return normalize_quaternion(q)


def axis_angle_quaternion(axis: Sequence[float], angle_deg: float) -> Quaternion:
    """The unit quaternion of a rotation by ``angle_deg`` about ``axis`` (any non-zero length; right-hand rule)."""

    ax, ay, az = (float(c) for c in axis)
    if not all(math.isfinite(c) for c in (ax, ay, az)):
        raise ValueError("the rotation axis must be a non-zero finite vector")
    n = math.sqrt(ax * ax + ay * ay + az * az)
    if not SCALE_FREE_NORM[0] <= n <= SCALE_FREE_NORM[1]:  # very long or very short: scale by the largest component first
        largest = max(abs(ax), abs(ay), abs(az))
        if largest == 0.0:
            raise ValueError("the rotation axis must be a non-zero finite vector")
        ax, ay, az = ax / largest, ay / largest, az / largest
        n = math.sqrt(ax * ax + ay * ay + az * az)
    c, s = cos_sin_deg(float(angle_deg) / 2.0)
    return normalize_quaternion((c, s * (ax / n), s * (ay / n), s * (az / n)))


def multiply(a: Sequence[float], b: Sequence[float]) -> Quaternion:
    """The Hamilton product ``a * b`` (``b`` first, then ``a``), as ``PoseV1.compose``."""

    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)


def conjugate(q: Sequence[float]) -> Quaternion:
    w, x, y, z = q
    return (w, -x, -y, -z)


def is_identity(q: Sequence[float]) -> bool:
    return tuple(q) == IDENTITY


def signed_permutation(q: Sequence[float]) -> tuple[tuple[int, int], ...] | None:
    """For a quarter-turn rotation (the matrix of ``unit(q)`` is a signed permutation within 1e-12): per output axis, the
    input axis and the sign; else None."""

    w, x, y, z = unit(q)
    matrix = ((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
              (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
              (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)))
    rows = []
    for row in matrix:
        ones = [(j, 1 if v > 0 else -1) for j, v in enumerate(row) if abs(abs(v) - 1.0) <= PERMUTATION_TOLERANCE]
        zeros = [j for j, v in enumerate(row) if abs(v) <= PERMUTATION_TOLERANCE]
        if len(ones) != 1 or len(zeros) != 2:
            return None
        rows.append(ones[0])
    return tuple(rows) if len({r[0] for r in rows}) == 3 else None


def rotate(q: Sequence[float], v: Sequence[float]) -> Vector:
    """``R(q) v``, the rotation of ``unit(q)``: exactly (a permutation with signs) for a quarter-turn rotation, else
    ``v + 2 u x (u x v + w v)`` (as ``PoseV1.compose``)."""

    q = unit(q)
    permutation = signed_permutation(q)
    if permutation is not None:
        return tuple(sign * float(v[axis]) for axis, sign in permutation)  # type: ignore[return-value]
    w1, x1, y1, z1 = q
    vx, vy, vz = v
    cx, cy, cz = y1 * vz - z1 * vy + w1 * vx, z1 * vx - x1 * vz + w1 * vy, x1 * vy - y1 * vx + w1 * vz
    return (vx + 2.0 * (y1 * cz - z1 * cy), vy + 2.0 * (z1 * cx - x1 * cz), vz + 2.0 * (x1 * cy - y1 * cx))


def add(a: Sequence[float], b: Sequence[float]) -> Vector:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a: Sequence[float], b: Sequence[float]) -> Vector:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def compose_pose(parent_position: Sequence[float], parent_q: Sequence[float], position: Sequence[float],
                 q: Sequence[float]) -> tuple[Vector, Quaternion]:
    """A pose given in the frame ``(parent_position, parent_q)``, expressed in that frame's parent frame. The quaternion
    is renormalized unless the parent frame has no rotation (then it is copied exactly)."""

    if is_identity(parent_q):
        return add(position, parent_position), tuple(float(c) for c in q)  # type: ignore[return-value]
    return add(rotate(parent_q, position), parent_position), normalize_quaternion(multiply(parent_q, q))


def inverse_pose(position: Sequence[float], q: Sequence[float]) -> tuple[Vector, Quaternion]:
    """The inverse of a rigid pose: ``(-R^T p, q*)``."""

    qc = conjugate(q)
    rotated = rotate(qc, position)
    return (-rotated[0], -rotated[1], -rotated[2]), qc


__all__ = [
    "IDENTITY",
    "SQRT_HALF",
    "add",
    "axis_angle_quaternion",
    "compose_pose",
    "conjugate",
    "cos_sin_deg",
    "inverse_pose",
    "is_identity",
    "multiply",
    "norm",
    "normalize_quaternion",
    "rotate",
    "signed_permutation",
    "sub",
    "unit",
]
