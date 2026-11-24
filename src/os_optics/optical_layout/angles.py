"""Deterministic angles for the Optical Layout Drawing: atan2 from ``+ - * /`` and ``sqrt`` only.

``math.atan2`` comes from the platform's libm, whose last bit may differ between CPython on Windows, Linux and Pyodide;
a printed 0.1 deg could then differ in a rounding tie. ``atan2_deg`` reduces the argument to |t| <= 1, halves the angle
twice (``atan t = 2 atan(t / (1 + sqrt(1 + t^2)))``, |t| <= tan(pi/16)) and sums the alternating series to full double
precision: every operation is a correctly rounded IEEE 754 operation, so every runtime gets the same bits.
"""

from __future__ import annotations

import math

PI = 3.141592653589793
HALF_PI = PI / 2.0
DEGREES = 180.0 / PI
SERIES_TERMS = 24
#: Below this horizontal length a direction is vertical: its azimuth is 0.
VERTICAL = 1e-12


def _atan_unit(t: float) -> float:
    """atan(t) for 0 <= t <= 1."""

    for _ in range(2):
        t = t / (1.0 + math.sqrt(1.0 + t * t))
    t2 = t * t
    term, total = t, t
    for n in range(1, SERIES_TERMS):
        term = -term * t2
        total += term / (2 * n + 1)
    return 4.0 * total


def atan(t: float) -> float:
    sign = -1.0 if t < 0.0 else 1.0
    a = abs(t)
    value = _atan_unit(a) if a <= 1.0 else HALF_PI - _atan_unit(1.0 / a)
    return sign * value


def atan2_deg(y: float, x: float) -> float:
    """The angle of (x, y) from +x, counter-clockwise, in degrees in (-180, 180] (0 for the zero vector)."""

    if x == 0.0 and y == 0.0:
        return 0.0
    if x > 0.0:
        angle = atan(y / x)
    elif x < 0.0:
        angle = atan(y / x) + (PI if y >= 0.0 else -PI)
    else:
        angle = HALF_PI if y > 0.0 else -HALF_PI
    return angle * DEGREES


def azimuth_elevation(direction: tuple[float, float, float]) -> tuple[float, float]:
    """(azimuth from +X in the X-Y plane, counter-clockwise seen from +Z; elevation above the X-Y plane), degrees."""

    x, y, z = direction
    horizontal = math.sqrt(x * x + y * y)
    azimuth = atan2_deg(y, x) if horizontal > VERTICAL else 0.0
    return azimuth, atan2_deg(z, horizontal)
