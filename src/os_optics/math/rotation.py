"""Quaternion (Hamilton, ``(w, x, y, z)``, unit) utilities. Active rotations."""

from __future__ import annotations

import math

import numpy as np

from .vectors import Array, dot


#: The least length whose squared components keep their bits: below it, and where the squares overflow, ``quat_normalize``
#: divides by the largest component first.
LEAST_PLAIN_LENGTH = 1e-150


def quat_normalize(q: Array) -> Array:
    """``q / |q|`` for any non-zero, finite ``q``: a length whose squared components overflow or underflow is taken from
    ``q / max|c|`` (whose length lies in [1, 2]); ``ValueError`` for a zero or non-finite quaternion."""

    q = np.asarray(q, dtype=np.float64)
    n = math.sqrt(float(dot(q, q)))
    if math.isfinite(n) and n >= LEAST_PLAIN_LENGTH:
        return q / n
    largest = float(np.max(np.abs(q))) if q.size else 0.0
    if not math.isfinite(largest) or largest == 0.0:
        raise ValueError("cannot normalize a zero or non-finite quaternion")
    scaled = q / largest
    return scaled / math.sqrt(float(dot(scaled, scaled)))


def quat_to_matrix(q: Array) -> Array:
    w, x, y, z = quat_normalize(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_to_quat(m: Array) -> Array:
    """Shepperd's method; returns ``w >= 0`` representative."""
    m = np.asarray(m, dtype=np.float64)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s])
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    q = quat_normalize(q)
    return -q if q[0] < 0 else q


def quat_from_axis_angle(axis: Array, angle_rad: float) -> Array:
    axis = np.asarray(axis, dtype=np.float64)
    n = math.sqrt(float(dot(axis, axis)))
    if n == 0.0 or not math.isfinite(n):
        raise ValueError("rotation axis must be non-zero and finite")
    half = 0.5 * angle_rad
    return np.concatenate([[math.cos(half)], (math.sin(half) / n) * axis])


def quat_multiply(a: Array, b: Array) -> Array:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def rotation_matrix_axis_angle(axis: Array, angle_rad: float) -> Array:
    return quat_to_matrix(quat_from_axis_angle(axis, angle_rad))
