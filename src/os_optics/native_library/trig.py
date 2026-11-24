"""Trigonometry in exact decimal arithmetic for the native library (no binary float, no libm: the same digits in CPython
and Pyodide).

Every function works in ``WORK_CONTEXT`` (40 significant digits) with a few guard digits: ``pi`` by Machin's formula,
``sin`` / ``cos`` by their Taylor series after reduction to ``|x| <= pi / 4``, ``atan`` by its series after halving the
argument (``atan x = 2 atan(x / (1 + sqrt(1 + x^2)))``) until ``|x| < 0.1``, ``asin x = atan(x / sqrt(1 - x^2))``.
Angles in degrees are converted with that ``pi``. The results are rounded back to the working precision.
"""

from __future__ import annotations

from decimal import Decimal, localcontext

from .numbers import ONE, TWO, WORK_CONTEXT, ZERO

#: Guard digits added to the working precision inside a series.
GUARD_DIGITS = 10
DEGREES_PER_HALF_TURN = Decimal(180)
#: ``atan`` halves its argument until it is below this before summing the series.
ATAN_SERIES_LIMIT = Decimal("0.1")
_MACHIN = ((16, 5), (-4, 239))  # pi = 16 atan(1/5) - 4 atan(1/239)


def _context():
    context = WORK_CONTEXT.copy()
    context.prec = WORK_CONTEXT.prec + GUARD_DIGITS
    return context


def _atan_series(x: Decimal) -> Decimal:
    """``x - x^3/3 + x^5/5 - ...`` for ``|x| < 0.25`` (terms until they no longer change the sum)."""

    total, power, k, x2 = ZERO, x, 1, x * x
    while True:
        term = power / k
        if total + term == total:
            return total
        total += term
        power = -power * x2
        k += 2


def pi() -> Decimal:
    with localcontext(_context()):
        value = sum((Decimal(c) * _atan_series(ONE / Decimal(d)) for c, d in _MACHIN), ZERO)
    with localcontext(WORK_CONTEXT):
        return +value


def radians(degrees: Decimal) -> Decimal:
    with localcontext(_context()):
        value = degrees * pi() / DEGREES_PER_HALF_TURN
    with localcontext(WORK_CONTEXT):
        return +value


def degrees(radians_: Decimal) -> Decimal:
    with localcontext(_context()):
        value = radians_ * DEGREES_PER_HALF_TURN / pi()
    with localcontext(WORK_CONTEXT):
        return +value


def _sin_cos(x: Decimal) -> tuple[Decimal, Decimal]:
    """(sin x, cos x) in the guarded context."""

    half_pi = pi() / TWO
    quarter = int((x / half_pi).to_integral_value())  # x = quarter * pi/2 + r, |r| <= pi/4
    r = x - quarter * half_pi
    s, c = ZERO, ZERO
    term, k = r, 1  # sin series
    while True:
        if s + term == s:
            break
        s += term
        term = -term * r * r / ((k + 1) * (k + 2))
        k += 2
    term, k = ONE, 0  # cos series
    while True:
        if c + term == c:
            break
        c += term
        term = -term * r * r / ((k + 1) * (k + 2))
        k += 2
    return {0: (s, c), 1: (c, -s), 2: (-s, -c), 3: (-c, s)}[quarter % 4]


def sin_cos_deg(angle_deg: Decimal) -> tuple[Decimal, Decimal]:
    """(sin, cos) of an angle in degrees."""

    with localcontext(_context()):
        s, c = _sin_cos(angle_deg * pi() / DEGREES_PER_HALF_TURN)
    with localcontext(WORK_CONTEXT):
        return +s, +c


def tan_deg(angle_deg: Decimal) -> Decimal:
    with localcontext(_context()):
        s, c = _sin_cos(angle_deg * pi() / DEGREES_PER_HALF_TURN)
        value = s / c
    with localcontext(WORK_CONTEXT):
        return +value


def atan(x: Decimal) -> Decimal:
    with localcontext(_context()):
        halvings = 0
        while abs(x) >= ATAN_SERIES_LIMIT:
            x = x / (ONE + (ONE + x * x).sqrt())
            halvings += 1
        value = _atan_series(x) * (2 ** halvings)
    with localcontext(WORK_CONTEXT):
        return +value


def asin(x: Decimal) -> Decimal:
    """``asin x`` for ``|x| < 1`` (radians)."""

    if not -ONE < x < ONE:
        raise ValueError("asin needs |x| < 1")
    with localcontext(_context()):
        value = atan(x / (ONE - x * x).sqrt())
    with localcontext(WORK_CONTEXT):
        return +value


def atan_deg(x: Decimal) -> Decimal:
    return degrees(atan(x))


def asin_deg(x: Decimal) -> Decimal:
    return degrees(asin(x))


__all__ = ["asin", "asin_deg", "atan", "atan_deg", "degrees", "pi", "radians", "sin_cos_deg", "tan_deg"]
