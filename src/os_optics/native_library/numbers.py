"""Exact decimal arithmetic of the native component library.

Every value a native class produces is computed with Python's ``decimal`` module in ``WORK_CONTEXT`` (no binary float,
no libm function), so the same parameters give the same model bytes in CPython and in Pyodide. Results are written as
canonical decimal strings (``DecimalString``) at a stated resolution:

* surface radii and every derived length: ``LENGTH_QUANTUM_MM`` (1 nm);
* a thickness chosen by a thickening rule: rounded up to ``THICKNESS_QUANTUM_MM`` (1 um);
* derived dimensionless values (a conic constant, a refractive index in a report): ``RATIO_QUANTUM``.
"""

from __future__ import annotations

from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_EVEN, Context, Decimal, DivisionByZero, InvalidOperation, Overflow, localcontext

from .._vendor.shared_contracts.common import decimal_to_string

#: Working precision of every computation (40 significant digits).
WORK_CONTEXT = Context(prec=40, rounding=ROUND_HALF_EVEN)
#: The context a whole native build (and description) runs in, stated in full: Python's standard decimal context (28
#: significant digits, round half even). Arithmetic outside a ``WORK_CONTEXT`` block uses it, so a native model never
#: depends on the decimal context of the thread that asks for it.
BUILD_CONTEXT = Context(prec=28, rounding=ROUND_HALF_EVEN, Emin=-999999, Emax=999999, capitals=1, clamp=0, flags=[],
                        traps=[InvalidOperation, DivisionByZero, Overflow])
#: Resolution of derived lengths (radii, vertex positions, paraxial distances, outlines): 1e-6 mm.
LENGTH_QUANTUM_MM = Decimal("0.000001")
#: Resolution of a thickness chosen by a thickening rule (rounded up): 1e-3 mm.
THICKNESS_QUANTUM_MM = Decimal("0.001")
#: Resolution of derived dimensionless values (conic constants, indices in design reports).
RATIO_QUANTUM = Decimal("0.000000000001")
ZERO = Decimal(0)
ONE = Decimal(1)
TWO = Decimal(2)
HALF = Decimal("0.5")
NM_PER_UM = Decimal(1000)


def to_decimal(value: object) -> Decimal | None:
    """A finite ``Decimal`` from a JSON scalar (a decimal string, an int, or a float by its shortest repr) or a ``Decimal``;
    else None."""

    if isinstance(value, bool):
        return None
    if isinstance(value, Decimal):
        return value if value.is_finite() else None
    if isinstance(value, float):
        value = repr(value)
    if isinstance(value, int):
        return Decimal(value)
    if not isinstance(value, str):
        return None
    try:
        number = Decimal(value.strip())
    except (InvalidOperation, ValueError):
        return None
    return number if number.is_finite() else None


def text(value: Decimal, quantum: Decimal | None = None, rounding: str = ROUND_HALF_EVEN) -> str:
    """The canonical decimal string of ``value``, first rounded to a multiple of ``quantum`` when given."""

    if quantum is not None:
        value = quantize(value, quantum, rounding)
    return decimal_to_string(value)


def quantize(value: Decimal, quantum: Decimal, rounding: str = ROUND_HALF_EVEN) -> Decimal:
    """``value`` rounded to a multiple of ``quantum`` (``ROUND_CEILING`` / ``ROUND_FLOOR`` for outward bounds)."""

    with localcontext(WORK_CONTEXT):
        return (value / quantum).quantize(ONE, rounding=rounding) * quantum


def ceil_to(value: Decimal, quantum: Decimal) -> Decimal:
    return quantize(value, quantum, ROUND_CEILING)


def floor_to(value: Decimal, quantum: Decimal) -> Decimal:
    return quantize(value, quantum, ROUND_FLOOR)


def length(value: Decimal) -> str:
    """A derived length (mm) as a canonical string at ``LENGTH_QUANTUM_MM``."""

    return text(value, LENGTH_QUANTUM_MM)


def sag(curvature: Decimal, conic: Decimal, rho: Decimal) -> Decimal | None:
    """Sag ``c rho^2 / (1 + sqrt(1 - (1 + k) c^2 rho^2))`` of a conic profile at radius ``rho``; None outside its domain."""

    with localcontext(WORK_CONTEXT):
        if curvature == 0:
            return ZERO
        rho_sq = rho * rho
        discriminant = ONE - (ONE + conic) * curvature * curvature * rho_sq
        if discriminant < 0:
            return None
        return curvature * rho_sq / (ONE + discriminant.sqrt())


__all__ = [
    "HALF",
    "LENGTH_QUANTUM_MM",
    "NM_PER_UM",
    "ONE",
    "RATIO_QUANTUM",
    "THICKNESS_QUANTUM_MM",
    "TWO",
    "WORK_CONTEXT",
    "ZERO",
    "ceil_to",
    "floor_to",
    "length",
    "quantize",
    "sag",
    "text",
    "to_decimal",
]
