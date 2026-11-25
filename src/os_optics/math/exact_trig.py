"""Trigonometry, exponential and logarithm from ``+ - x /`` and ``sqrt`` only, for values that must have the same bits in
every runtime.

IEEE-754 makes the basic operations and the square root correctly rounded on every platform (and scaling by a power of 2
exact), while libm's ``acos``, ``atan2``, ``sin``, ``cos``, ``exp``, ``pow`` and ``log`` may differ in the last bit between
CPython builds and Pyodide. The Engine uses these functions where a last-bit difference would reach a result: the angle
of incidence at which an angle-resolved coating table is read and that a surface exposure reports, the phase factors
``exp(i phi)`` of stated coating phases and of the network export, an attenuator's azimuth and ``10^-OD``, the radii
and angles of the pupil-ray samples, the Gaussian truncation integrals (and their Gauss-Legendre nodes), the phase and polarization ellipse of an arrival and the image rotation of its
first-order transfer. Accuracy:
a few units in the last place.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

_TWO_PI = 6.283185307179586
_PI = 3.141592653589793
_HALF_PI = 1.5707963267948966
#: ln 2 split into a part exact in a product with a small integer and the rest (Cody-Waite reduction of exp).
_LN2_HI = 0.6931471803691238
_LN2_LO = 1.9082149292705877e-10
_LN2 = 0.6931471805599453
_LN10 = 2.302585092994046
_SQRT_HALF = 0.7071067811865476
#: ``pow10`` splits off integer powers of 10 up to this exponent (beyond it the result over- or underflows anyway).
_POW10_MAX_EXPONENT = 400.0
_DECADE = 10  # (an exact integer: its powers convert to the nearest double)
#: ``exp`` scales by at most this power of 2 (beyond about 1075 any double over- or underflows).
_EXP_MAX_SCALE = 1100
#: exp(r) for |r| <= ln 2 / 2: Taylor terms (the first omitted one is below 1e-20 relative).
_EXP_TERMS = 22
#: For |r| <= 0.35 (the reduced argument, ln 2 / 2 = 0.3466 plus the rounding of ``k ln 2`` up to |x| of about 1e13) every
#: term ``r^n / n!`` with n >= 14 is below 4.8e-18, less than half the spacing of doubles at the partial sum (>= e^-0.35 =
#: 0.70, spacing >= 2^-53), so adding it leaves the sum unchanged: stopping after the term n = 13 gives the same bits as
#: ``_EXP_TERMS`` (a NaN or a larger |r| takes all of them).
_EXP_SHORT_TERMS = 14
_EXP_SHORT_MAX_R = 0.35
#: ln(m) = 2 atanh(z), |z| <= 0.172: odd terms (the first omitted one is below 1e-20).
_ATANH_TERMS = 16
_PI_OVER_6 = 0.5235987755982988
_DEG_PER_RAD = 57.29577951308232
_SQRT3 = 1.7320508075688772
#: tan(15 deg): arguments above it are reduced by atan(t) = pi / 6 + atan((sqrt3 t - 1) / (sqrt3 + t)).
_TAN_15_DEG = 0.2679491924311227
#: Odd terms of the arctangent series for |u| <= tan(15 deg): the first omitted term is below 1e-24.
_ATAN_TERMS = 20
#: exp(i phi): the reduced angle is halved this many times, expanded in a short series and doubled back.
_HALVINGS = 2
_SERIES_TERMS = 14


def _atan_unit(t: Array) -> Array:
    """arctan of ``t`` in ``[0, 1]`` (radians)."""

    big = t > _TAN_15_DEG
    u = np.where(big, (t * _SQRT3 - 1.0) / (_SQRT3 + t), t)
    minus_u2 = -(u * u)  # ((-term) * u2 and term * (-u2) are the same correctly rounded product)
    total = np.zeros_like(u)
    term = u
    for k in range(_ATAN_TERMS):
        total += term / (2 * k + 1)
        term = term * minus_u2
    return np.where(big, _PI_OVER_6 + total, total)


def _atan_unit_float(t: float) -> float:
    """``_atan_unit`` of one value in Python floats: the same operations in the same order (each ``+ - x /`` is the same
    correctly rounded IEEE operation as NumPy's), so the same bits, without NumPy's per-operation overhead."""

    big = t > _TAN_15_DEG
    u = (t * _SQRT3 - 1.0) / (_SQRT3 + t) if big else t
    u2 = u * u
    total = 0.0
    term = u
    for k in range(_ATAN_TERMS):
        total = total + term / (2 * k + 1)
        term = -term * u2
    return _PI_OVER_6 + total if big else total


def acos_deg(cos_value: Array) -> Array:
    """``arccos`` in degrees of values clipped to ``[0, 1]``: ``2 atan(sqrt((1 - c) / (1 + c)))``."""

    if np.ndim(cos_value) == 0:  # one value (most callers): Python floats, the same operations
        c = float(cos_value)
        if c == c:  # (not NaN)
            c = 1.0 if c > 1.0 else (0.0 if c < 0.0 else c)
            return np.float64(2.0 * _atan_unit_float(math.sqrt((1.0 - c) / (1.0 + c))) * _DEG_PER_RAD)
    values = np.asarray(cos_value, dtype=np.float64)
    if values.size == 1:  # one ray's array (a one-ray packet): the same operations in Python floats, in the array's shape
        c = float(values.reshape(-1)[0])
        if c == c:
            c = 1.0 if c > 1.0 else (0.0 if c < 0.0 else c)
            return np.full(values.shape, 2.0 * _atan_unit_float(math.sqrt((1.0 - c) / (1.0 + c))) * _DEG_PER_RAD)
    c = np.clip(values, 0.0, 1.0)
    return 2.0 * _atan_unit(np.sqrt((1.0 - c) / (1.0 + c))) * _DEG_PER_RAD


#: Below this ``1 - cos`` (an angle of incidence under about 0.81 deg) ``incidence_deg`` takes the angle from the sine:
#: next to 1 a cosine has lost the angle (one unit in its last place is 8.5e-7 deg, so a direction one unit off the normal
#: reads 0 or 8.5e-7 deg by rounding), while ``|n x d|`` keeps it. Above it both forms agree to about 1e-12 relative.
_NEAR_NORMAL_ONE_MINUS_COS = 1e-4
#: An angle of incidence below this (deg) is reported as 0: a unit direction's last bit is 6e-15 deg, but a direction
#: refracted through lenses carries more rounding -- up to about 4e-9 deg on the native confocal bench's on-axis hits,
#: whose true incidence is 0 -- so no angle this small is resolved. 1e-7 deg is 25 times that noise and a tenth of the
#: 1e-6 deg a group-delay record prints: normal incidence reads 0 whatever the rounding of the poses and the refractions.
_INCIDENCE_RESOLUTION_DEG = 1e-7


def _near_normal_deg_float(d: Array, n: Array) -> float:
    """``atan(|n x d| / |n . d|)`` in degrees of one direction and one normal, in Python floats (no FMA)."""

    dx, dy, dz = np.asarray(d, dtype=np.float64).reshape(-1)[:3].tolist()
    nx, ny, nz = np.asarray(n, dtype=np.float64).reshape(-1)[:3].tolist()
    cx, cy, cz = ny * dz - nz * dy, nz * dx - nx * dz, nx * dy - ny * dx
    along = abs(nx * dx + ny * dy + nz * dz)
    angle = _atan_unit_float(math.sqrt(cx * cx + cy * cy + cz * cz) / along) * _DEG_PER_RAD
    return angle if angle >= _INCIDENCE_RESOLUTION_DEG else 0.0


def incidence_deg(cos_value: Array, d: Array, n: Array) -> Array:
    """The angle of incidence in degrees of direction(s) ``d`` on a surface of normal(s) ``n``, given ``cos_value``, the
    ``|n . d|`` the caller computed (clipped to ``[0, 1]``; one value for one direction ``d`` of shape ``(3,)``, an array
    for rows of ``d``). Away from normal incidence it is ``acos_deg(cos_value)`` (the same bits); within
    ``_NEAR_NORMAL_ONE_MINUS_COS`` of 1 it is ``atan(|n x d| / |n . d|)``, which does not depend on the rounding of the
    cosine, and an angle below ``_INCIDENCE_RESOLUTION_DEG`` is 0: a direction a few units in the last place off the
    normal reads 0, not 0 or 8.5e-7 deg by the rounding of its cosine.

    Each value takes one arctangent: a single one is computed only in the form that is returned, and the rows of an array
    share one ``_atan_unit`` of ``|n x d| / |n . d|`` (near the normal) or ``acos_deg``'s ``sqrt((1 - c) / (1 + c))`` (away
    from it) -- elementwise operations, so each row has the bits of the form computed on its own."""

    if np.ndim(cos_value) == 0:
        c = float(cos_value)
        if 1.0 - c < _NEAR_NORMAL_ONE_MINUS_COS:  # (False for a NaN)
            return np.float64(_near_normal_deg_float(d, n))
        return acos_deg(cos_value)
    values = np.asarray(cos_value, dtype=np.float64)
    if values.size == 1:  # one ray's array: Python floats, in the array's shape
        if 1.0 - float(values.reshape(-1)[0]) < _NEAR_NORMAL_ONE_MINUS_COS:
            return np.full(values.shape, _near_normal_deg_float(d, n))
        return acos_deg(cos_value)
    near = (1.0 - values) < _NEAR_NORMAL_ONE_MINUS_COS
    if not bool(near.any()):
        return acos_deg(cos_value)
    d = np.asarray(d, dtype=np.float64).reshape(values.shape + (3,))
    dx, dy, dz = d[..., 0], d[..., 1], d[..., 2]
    n = np.asarray(n, dtype=np.float64)
    if n.ndim == 1:  # one normal for every row: its components as Python floats (the same products)
        nx, ny, nz = n.tolist()
    else:
        n = np.broadcast_to(n, d.shape)
        nx, ny, nz = n[..., 0], n[..., 1], n[..., 2]
    cx = ny * dz - nz * dy
    cy = nz * dx - nx * dz
    cz = nx * dy - ny * dx
    along = np.abs(nx * dx + ny * dy + nz * dz)
    ratio = np.divide(np.sqrt(cx * cx + cy * cy + cz * cz), along, out=np.zeros_like(along), where=near & (along > 0.0))
    c = np.clip(values, 0.0, 1.0)  # (acos_deg's argument for the rows away from the normal)
    arctangent = _atan_unit(np.where(near, ratio, np.sqrt((1.0 - c) / (1.0 + c))))
    small = arctangent * _DEG_PER_RAD
    return np.where(near, np.where(small >= _INCIDENCE_RESOLUTION_DEG, small, 0.0), 2.0 * arctangent * _DEG_PER_RAD)


def cis(phase_rad: float) -> complex:
    """``exp(i phase)``: the angle reduced to ``[-pi, pi]``, halved ``_HALVINGS`` times, its cosine and sine summed as power
    series and doubled back (``cos 2x = c^2 - s^2``, ``sin 2x = 2 s c``)."""

    x = float(phase_rad)
    x = x - round(x / _TWO_PI) * _TWO_PI
    x = x / 2.0 ** _HALVINGS
    x2 = x * x
    c, s, term_c, term_s = 0.0, 0.0, 1.0, x
    for k in range(_SERIES_TERMS):
        c += term_c
        s += term_s
        term_c = -term_c * x2 / ((2 * k + 1) * (2 * k + 2))
        term_s = -term_s * x2 / ((2 * k + 2) * (2 * k + 3))
    for _ in range(_HALVINGS):
        c, s = c * c - s * s, 2.0 * s * c
    return complex(c, s)


def cos_sin(phase_rad: Array) -> tuple[Array, Array]:
    """``(cos, sin)`` elementwise, the same reduction and series as ``cis`` (the same bits as ``cis`` of each element)."""

    x = np.asarray(phase_rad, dtype=np.float64)
    x = x - np.rint(x / _TWO_PI) * _TWO_PI
    x = x / 2.0 ** _HALVINGS
    x2 = x * x
    c, s = np.zeros_like(x), np.zeros_like(x)
    term_c, term_s = np.ones_like(x), x
    for k in range(_SERIES_TERMS):
        c = c + term_c
        s = s + term_s
        term_c = -term_c * x2 / ((2 * k + 1) * (2 * k + 2))
        term_s = -term_s * x2 / ((2 * k + 2) * (2 * k + 3))
    for _ in range(_HALVINGS):
        c, s = c * c - s * s, 2.0 * s * c
    return c, s


def asin(x: Array) -> Array:
    """``arcsin`` of values clipped to ``[-1, 1]`` (radians): ``atan2(x, sqrt((1 - x) (1 + x)))``."""

    x = np.clip(np.asarray(x, dtype=np.float64), -1.0, 1.0)
    return atan2(x, np.sqrt((1.0 - x) * (1.0 + x)))


def atan2(y: Array, x: Array) -> Array:
    """``atan2(y, x)`` in radians, in ``(-pi, pi]`` (``0`` at the origin), elementwise."""

    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if y.ndim == 0 and x.ndim == 0:  # one pair: Python floats, the same operations (NaN takes the array path)
        yf, xf = float(y), float(x)
        if yf == yf and xf == xf:
            ayf, axf = abs(yf), abs(xf)
            big_f, small_f = (axf, ayf) if axf >= ayf else (ayf, axf)
            a = _atan_unit_float(small_f / big_f if big_f > 0.0 else 0.0)
            a = _HALF_PI - a if ayf > axf else a
            a = _PI - a if xf < 0.0 else a
            return np.asarray(-a if yf < 0.0 else a)
    ay, ax = np.abs(y), np.abs(x)
    big, small = np.maximum(ax, ay), np.minimum(ax, ay)
    t = np.divide(small, big, out=np.zeros_like(big), where=big > 0.0)
    a = _atan_unit(t)
    a = np.where(ay > ax, _HALF_PI - a, a)
    a = np.where(x < 0.0, _PI - a, a)
    return np.where(y < 0.0, -a, a)


def atan2_deg(y: Array, x: Array) -> Array:
    """``atan2(y, x)`` in degrees, in ``(-180, 180]`` (``0`` at the origin), elementwise."""

    return atan2(y, x) * _DEG_PER_RAD


def exp(x: Array) -> Array:
    """``e^x`` elementwise: ``x = k ln 2 + r`` (|r| <= ln 2 / 2), a Taylor series for ``e^r``, scaled by ``2^k`` exactly.
    Underflows to 0 below about -745 and overflows to inf above about 709, like libm."""

    x = np.asarray(x, dtype=np.float64)
    k = np.rint(x / _LN2)
    r = (x - k * _LN2_HI) - k * _LN2_LO
    total = np.ones_like(r)
    term = np.ones_like(r)
    # within |r| <= _EXP_SHORT_MAX_R the terms from n = _EXP_SHORT_TERMS on cannot change the sum (see the constants)
    terms = _EXP_SHORT_TERMS if bool((np.abs(r) <= _EXP_SHORT_MAX_R).all()) else _EXP_TERMS
    for n in range(1, terms):
        term = term * r / n
        total = total + term
    with np.errstate(over="ignore", under="ignore"):
        return np.ldexp(total, np.clip(k, -_EXP_MAX_SCALE, _EXP_MAX_SCALE).astype(np.int64))


def exp_float(x: float) -> float:
    """``exp`` of one finite double, with the same operations in the same order as ``exp`` of a one-element array (the
    same bits) but on Python floats, without NumPy's per-call overhead. A non-finite ``x`` takes the array path."""

    x = float(x)
    if not math.isfinite(x):
        return float(exp(np.array([x]))[0])
    k = float(round(x / _LN2))  # (round half to even, as np.rint)
    r = (x - k * _LN2_HI) - k * _LN2_LO
    total = 1.0
    term = 1.0
    for n in range(1, _EXP_SHORT_TERMS if abs(r) <= _EXP_SHORT_MAX_R else _EXP_TERMS):
        term = term * r / n
        total = total + term
    with np.errstate(over="ignore", under="ignore"):
        return float(np.ldexp(np.float64(total), int(min(max(k, -_EXP_MAX_SCALE), _EXP_MAX_SCALE))))


def expm1(x: Array) -> Array:
    """``e^x - 1`` elementwise, accurate for small ``|x|``: the Taylor terms after the first summed directly for
    ``|x| <= ln 2 / 2``, else ``exp(x) - 1``."""

    x = np.asarray(x, dtype=np.float64)
    small = np.abs(x) <= 0.5 * _LN2
    r = np.where(small, x, 0.0)
    total = np.zeros_like(r)
    term = np.ones_like(r)
    for n in range(1, _EXP_TERMS):
        term = term * r / n
        total = total + term
    return np.where(small, total, exp(np.where(small, 0.0, x)) - 1.0)


def pow10(x: Array) -> Array:
    """``10^x`` elementwise: ``10^n`` (``n`` the nearest integer, an exactly converted integer power) times ``exp(f ln 10)``
    of the rest ``|f| <= 1/2``; exact for integer ``x`` (``10^-2`` is the double nearest 0.01)."""

    x = np.asarray(x, dtype=np.float64)
    n = np.rint(np.clip(x, -_POW10_MAX_EXPONENT, _POW10_MAX_EXPONENT))
    rest = exp((x - n) * _LN10)
    flat = n.ravel()
    scale = np.array([float(_DECADE ** int(abs(v))) for v in flat], dtype=np.float64).reshape(n.shape)
    with np.errstate(over="ignore", under="ignore"):
        return np.where(n >= 0.0, rest * scale, rest / scale)


def log(x: float) -> float:
    """Natural logarithm of a positive finite ``x``: ``x = m 2^e`` exactly, ``ln m = 2 atanh((m - 1) / (m + 1))`` as a
    series."""

    m, e = np.frexp(float(x))
    m, e = float(m), int(e)
    if m < _SQRT_HALF:
        m, e = 2.0 * m, e - 1
    z = (m - 1.0) / (m + 1.0)
    z2 = z * z
    total, term = 0.0, z
    for k in range(_ATANH_TERMS):
        total += term / (2 * k + 1)
        term *= z2
    return 2.0 * total + e * _LN2


def log10(x: float) -> float:
    """``log10(x)`` of a positive finite ``x``: ``log(x) / ln 10``."""

    return log(x) / _LN10


__all__ = ["acos_deg", "asin", "atan2", "atan2_deg", "cis", "cos_sin", "exp", "exp_float", "expm1", "incidence_deg", "log", "log10",
           "pow10"]
