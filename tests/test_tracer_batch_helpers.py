"""The one-ray shortcuts of the vector and trigonometry helpers give the bits of their array forms.

A probe without pupil rays is a packet of one ray, so the tracer's per-step geometry runs on ``(1, 3)`` rows. ``dot``,
``cross``, ``normalize`` and ``acos_deg`` compute such a row in Python floats (the same correctly rounded ``+ - x /`` and
square root, in the same order) and ``cross`` computes float64 rows component by component instead of through
``numpy.cross``. These tests compare each against the array arithmetic it replaces, with ``==`` on the bytes, over seeded
random rows that include signed zeros, subnormals, huge and non-finite components (in both operands of ``dot``). A NaN
``dot`` is a NaN in both forms, but its sign bit is not fixed (adding two NaNs keeps either one's sign, even within one
interpreter), so the ``dot`` comparisons make NaN canonical first.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from os_optics.math import exact_trig
from os_optics.math.vectors import cross, dot, normalize

SPECIALS = np.array([0.0, -0.0, 5e-324, -5e-324, 1e-300, 1e300, -1e300, 1.0, -1.0, np.inf, -np.inf, np.nan])


def _bits(x) -> bytes:
    a = np.asarray(x)
    return a.dtype.str.encode() + str(a.shape).encode() + np.ascontiguousarray(a).tobytes()


def _bits_nan_canonical(x) -> bytes:
    """``_bits`` with every NaN replaced by the canonical NaN (a NaN stays a NaN; only its sign and payload are dropped)."""
    a = np.asarray(x)
    return _bits(np.where(np.isnan(a), np.nan, a))


def _rows(rng: np.random.Generator, n: int, special_share: float = 0.2) -> np.ndarray:
    rows = rng.standard_normal((n, 3)) * 10.0 ** rng.uniform(-8, 8, (n, 1))
    pick = rng.random((n, 3)) < special_share
    rows[pick] = rng.choice(SPECIALS, size=int(pick.sum()))
    return rows


def _dot_reference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    total = a[..., 0] * b[..., 0]
    for i in range(1, max(a.shape[-1], b.shape[-1])):
        total = total + a[..., i] * b[..., i]
    return total


def _normalize_reference(v: np.ndarray, zero_tol: float) -> tuple[np.ndarray, np.ndarray]:
    n = np.sqrt(_dot_reference(v, v))
    ok = np.isfinite(n) & (n > zero_tol)
    safe = np.where(ok, n, 1.0)
    return np.where(ok[..., None], v / safe[..., None], 0.0), ok


@pytest.mark.parametrize("seed", range(4))
def test_dot_of_one_row_has_the_bits_of_the_array_sum(seed):
    rng = np.random.default_rng(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for k in (1, 2, 3, 4):
            a, b = rng.standard_normal((400, k)), rng.standard_normal((400, k))
            for operand in (a, b):  # (special values drawn element by element, in both operands)
                pick = rng.random((400, k)) < 0.2
                operand[pick] = rng.choice(SPECIALS, size=int(pick.sum()))
            same = _bits_nan_canonical
            for i in range(a.shape[0]):
                row_a, row_b = a[i:i + 1], b[i:i + 1]
                assert same(dot(row_a, row_b)) == same(_dot_reference(row_a, row_b))
                assert same(dot(row_a, b[i])) == same(_dot_reference(row_a, b[i]))
                assert same(dot(a[i], row_b)) == same(_dot_reference(a[i], row_b))
                assert same(dot(a[i], b[i])) == same(_dot_reference(a[i], b[i]))
            # several rows: all products in one operation, columns added in order (a view with steps as a copy)
            assert same(dot(a, b)) == same(_dot_reference(a, b))
            assert same(dot(a[::-3], b[::-3])) == same(_dot_reference(a[::-3].copy(), b[::-3].copy()))
            assert same(dot(a, b[5])) == same(_dot_reference(a, b[5]))
            assert same(dot(a[5], b)) == same(_dot_reference(a[5], b))
            assert same(dot(np.asfortranarray(a), b)) == same(_dot_reference(a, b))
            assert same(dot(a[:0], b[:0])) == same(_dot_reference(a[:0], b[:0]))


@pytest.mark.parametrize("seed", range(4))
def test_cross_has_the_bits_of_numpy_cross(seed):
    rng = np.random.default_rng(100 + seed)
    a, b = _rows(rng, 300), _rows(rng, 300)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for i in range(a.shape[0]):
            for x, y in ((a[i:i + 1], b[i:i + 1]), (a[i], b[i]), (a[i:i + 1], b[i]), (a[i], b[i:i + 1])):
                assert _bits(cross(x, y)) == _bits(np.cross(x, y))
        for x, y in ((a, b), (a, b[7]), (a[3], b), (a[::-2], b[::-2]), (a[:1], b), (a, b[:1]), (a[:0], b[:0]),
                     (np.asfortranarray(a), b)):
            assert _bits(cross(x, y)) == _bits(np.cross(x, y))
    assert _bits(cross(np.array([[1, 0, 0]]), np.array([[0, 1, 0]]))) == _bits(np.cross([[1, 0, 0]], [[0, 1, 0]]))


@pytest.mark.parametrize("seed", range(4))
def test_normalize_of_one_row_has_the_bits_of_the_array_form(seed):
    rng = np.random.default_rng(200 + seed)
    v = _rows(rng, 600)
    v[rng.random(600) < 0.1] = 0.0
    v[rng.random(600) < 0.05] *= 1e-13  # (near the zero tolerances in use)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        # (a negative or NaN tolerance has no meaning for a length, but the one-row form answers it as the array form does)
        for tol in (0.0, 1e-12, -0.0, -1.0, -1e-300, -np.inf, np.nan):
            for i in range(v.shape[0]):
                got, want = normalize(v[i:i + 1], tol), _normalize_reference(v[i:i + 1], tol)
                assert _bits(got[0]) == _bits(want[0]) and _bits(got[1]) == _bits(want[1]), (tol, i, v[i])
        for row in ([0.0, 0.0, 0.0], [5e-324, 0.0, 0.0], [1e-200, 1e-200, 0.0]):  # (a length that is exactly zero)
            one = np.array([row])
            for tol in (-1.0, -np.inf):
                got, want = normalize(one, tol), _normalize_reference(one, tol)
                assert _bits(got[0]) == _bits(want[0]) and _bits(got[1]) == _bits(want[1]), (tol, row)


def test_acos_deg_of_a_one_element_array_has_the_bits_of_the_array_series():
    rng = np.random.default_rng(300)
    values = np.concatenate([rng.uniform(-0.2, 1.2, 3000), rng.uniform(0.999999, 1.0, 500), rng.uniform(0.0, 1e-6, 200),
                             [0.0, -0.0, 1.0, np.nextafter(1.0, 0.0), 2.0, -3.0, 0.2588190451025207, 0.9659258262890683]])
    whole = exact_trig.acos_deg(values)  # (the array series over all of them)
    for i, c in enumerate(values):
        assert _bits(exact_trig.acos_deg(values[i:i + 1])) == _bits(whole[i:i + 1])
        assert _bits(exact_trig.acos_deg(values[i:i + 1].reshape(1, 1))) == _bits(whole[i:i + 1].reshape(1, 1))
        assert float(exact_trig.acos_deg(c)) == float(whole[i])
    nan = exact_trig.acos_deg(np.array([np.nan]))
    assert nan.shape == (1,) and np.isnan(nan).all()
