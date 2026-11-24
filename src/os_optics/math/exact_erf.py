"""The error function from ``+ - x /`` and the exact exponential only, for values that must have the same bits in every
runtime (``math.exact_trig`` explains why libm is avoided).

* ``erfc(x)`` for ``x >= 0``: below ``ERFC_SERIES_LIMIT`` as ``1 - erf(x)`` with ``erf(x) = 2/sqrt(pi) e^(-x^2) sum_n
  x (2 x^2)^n / (2n+1)!!`` (positive terms, ``ERF_SERIES_TERMS`` of them); above it the continued fraction
  ``erfc(x) = e^(-x^2) / sqrt(pi) / (x + (1/2) / (x + 1 / (x + (3/2) / (x + ...))))`` evaluated backward from
  ``ERFC_FRACTION_DEPTH``. Accuracy (checked against an independent implementation over ``[0, 27]``): absolute error
  below ``2e-15`` everywhere and relative error below ``4e-15`` from ``ERFC_SERIES_LIMIT`` on, where ``erfc`` is small,
  so far tails keep their relative accuracy.
* ``erf_interval(a, b) = (erf(b) - erf(a)) / 2`` for ``a <= b``: the probability that a normal variable lies in an
  interval, formed from ``erfc`` of ``|a|`` and ``|b|`` so that no difference of two numbers close to 1 is taken (an
  interval far in one tail keeps its relative accuracy, an interval covering nearly everything its absolute accuracy).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

import math

from .exact_trig import exp as exact_exp
from .exact_trig import exp_float

Array = NDArray[np.float64]

_SQRT_PI = 1.7724538509055159
_TWO_OVER_SQRT_PI = 1.1283791670955126
#: erfc switches from the series (1 - erf) to the continued fraction at this argument.
ERFC_SERIES_LIMIT = 1.5
#: Terms of the positive erf series below ERFC_SERIES_LIMIT (the first omitted one is below 1e-20 of the sum).
ERF_SERIES_TERMS = 40
#: Depth of the backward-evaluated continued fraction at and above ERFC_SERIES_LIMIT (converged to rounding there).
ERFC_FRACTION_DEPTH = 100
#: Arguments are capped here: erfc(40) = e^-1600 / ... underflows to 0 like any larger argument.
_ERFC_ARGUMENT_CAP = 40.0


def erfc(x: Array) -> Array:
    """Complementary error function of ``x >= 0`` (elementwise; negative entries are treated as 0)."""

    x = np.minimum(np.maximum(np.asarray(x, dtype=np.float64), 0.0), _ERFC_ARGUMENT_CAP)
    small = x < ERFC_SERIES_LIMIT
    xs = np.where(small, x, 0.0)
    x2 = xs * xs
    term = xs.copy()
    total = xs.copy()
    for n in range(1, ERF_SERIES_TERMS):
        term = term * (2.0 * x2) / (2 * n + 1)
        total = total + term
    series = 1.0 - _TWO_OVER_SQRT_PI * exact_exp(-x2) * total
    xl = np.where(small, ERFC_SERIES_LIMIT, x)
    fraction = xl.copy()
    for k in range(ERFC_FRACTION_DEPTH, 0, -1):
        fraction = xl + (0.5 * k) / fraction
    tail = exact_exp(-(xl * xl)) / (_SQRT_PI * fraction)
    return np.where(small, series, tail)


def erfc_float(x: float) -> float:
    """``erfc`` of one double: the same operations in the same order as ``erfc`` of a one-element array (the same bits;
    only the branch the array path selects is evaluated), on Python floats without NumPy's per-call overhead. A NaN
    takes the array path."""

    x = float(x)
    if math.isnan(x):
        return float(erfc(np.array([x]))[0])
    x = min(max(x, 0.0), _ERFC_ARGUMENT_CAP)
    if x < ERFC_SERIES_LIMIT:
        x2 = x * x
        term = x
        total = x
        for n in range(1, ERF_SERIES_TERMS):
            term = term * (2.0 * x2) / (2 * n + 1)
            total = total + term
        return 1.0 - _TWO_OVER_SQRT_PI * exp_float(-x2) * total
    fraction = x
    for k in range(ERFC_FRACTION_DEPTH, 0, -1):
        fraction = x + (0.5 * k) / fraction
    return exp_float(-(x * x)) / (_SQRT_PI * fraction)


def erf_interval(a: Array, b: Array) -> Array:
    """``(erf(b) - erf(a)) / 2`` elementwise for ``a <= b``."""

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ea, eb = erfc(np.abs(a)), erfc(np.abs(b))
    return np.where(a >= 0.0, 0.5 * (ea - eb), np.where(b <= 0.0, 0.5 * (eb - ea), 1.0 - 0.5 * (ea + eb)))


__all__ = ["ERFC_FRACTION_DEPTH", "ERFC_SERIES_LIMIT", "ERF_SERIES_TERMS", "erf_interval", "erfc", "erfc_float"]
