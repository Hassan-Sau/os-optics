"""Jones calculus in a transported 3D transverse basis.

State per ray: unit direction ``d``, right-handed basis ``(e1, e2)`` with ``e1 x e2 = d``, complex Jones vector
``J = (J1, J2)`` such that ``E = J1 e1 + J2 e2``. Amplitudes are energy-normalized (``|J|^2`` is the power fraction),
so interface coefficients passed in are ``sqrt(R)e^{i phi}`` / energy-normalized transmission amplitudes.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ...math.exact_linalg import cmul, cvdot
from ...math.exact_trig import cis
from ...math.vectors import cross, dot, incidence_s_axis, normalize

Array = NDArray[np.float64]
CArray = NDArray[np.complex128]


def field_3d(jones: CArray, e1: Array, e2: Array) -> CArray:
    """Complex 3D electric-field vector ``J1 e1 + J2 e2`` (rows)."""
    return jones[:, 0:1] * e1 + jones[:, 1:2] * e2


def s_axis(d: Array, n_or: Array, e1: Array, zero_tol: float) -> Array:
    """Unit ``s`` (perpendicular to the plane of incidence): ``d x n`` normalized; the ray's ``e1`` within
    ``NORMAL_INCIDENCE_SIN`` of normal incidence (``math.vectors.incidence_s_axis``: frame-covariant, so a design turned
    as a whole gives the same result)."""
    s, _ = incidence_s_axis(d, n_or, e1, zero_tol)
    return s


def to_sp(jones: CArray, e1: Array, e2: Array, s: Array, d: Array) -> tuple[CArray, CArray]:
    """Components of ``J`` in the right-handed basis ``(s, p')`` with ``p' = d x s``."""
    p = cross(d, s)
    j_s = jones[:, 0] * dot(e1, s) + jones[:, 1] * dot(e2, s)
    j_p = jones[:, 0] * dot(e1, p) + jones[:, 1] * dot(e2, p)
    return j_s, j_p


def apply_interface(
    jones: CArray,
    e1: Array,
    e2: Array,
    d_in: Array,
    s: Array,
    coef_s: CArray,
    coef_p: CArray,
    d_out: Array,
) -> tuple[CArray, Array, Array]:
    """Apply ``diag(coef_s, coef_p)`` in the ``(s, p')`` basis and return ``(J_out, e1_out, e2_out)``.

    The output basis is ``(s, d_out x s)`` (right-handed about ``d_out``; ``s`` is perpendicular to ``d_out`` for
    reflection/refraction in the plane of incidence). For ``d_out == d_in`` (pure transmission of a thin element)
    the result is re-expressed in the original ``(e1, e2)`` basis so the basis does not spin needlessly.
    """
    j_s, j_p = to_sp(jones, e1, e2, s, d_in)
    out_s, out_p = cmul(coef_s, j_s), cmul(coef_p, j_p)  # no fused multiply-add: identical bits in every runtime
    # s is perpendicular to the plane of incidence, hence to d_out; near normal incidence (s = e1) only to first order in
    # the angle: the output axis is s made exactly perpendicular to d_out, so the basis stays orthonormal
    e1_out, _ = normalize(s - dot(s, d_out)[:, None] * d_out, 0.0)
    e2_out = cross(d_out, e1_out)
    return np.stack([out_s, out_p], axis=1), e1_out, e2_out


def apply_thin_transmission(
    jones: CArray, e1: Array, e2: Array, d: Array, s: Array, coef_s: CArray, coef_p: CArray
) -> CArray:
    """Diagonal action in ``(s, p')`` for an undeviated ray, returned in the *unchanged* ``(e1, e2)`` basis."""
    p = cross(d, s)
    s1, s2 = dot(s, e1), dot(s, e2)
    p1, p2 = dot(p, e1), dot(p, e2)
    j_s = jones[:, 0] * s1 + jones[:, 1] * s2
    j_p = jones[:, 0] * p1 + jones[:, 1] * p2
    a, b = cmul(coef_s, j_s), cmul(coef_p, j_p)
    return np.stack([a * s1 + b * p1, a * s2 + b * p2], axis=1)


def project_axis(axis: Array, d: Array, e1: Array, e2: Array, zero_tol: float) -> tuple[Array, Array, NDArray[np.bool_]]:
    """Components ``(a1, a2)`` of a unit axis projected onto the ray's transverse plane. ``ok=False`` if degenerate."""
    a_t = axis[None, :] - dot(np.broadcast_to(axis, d.shape), d)[:, None] * d
    a_t, ok = normalize(a_t, zero_tol)
    return dot(a_t, e1), dot(a_t, e2), ok


def _cis(phase: float | Array) -> complex | CArray:
    """``exp(i phase)`` from ``math.exact_trig.cis``, elementwise for an array."""
    if np.ndim(phase) == 0:
        return cis(float(phase))
    flat = np.asarray(phase, dtype=np.float64)
    return np.array([cis(float(x)) for x in flat.ravel()], dtype=np.complex128).reshape(flat.shape)


def retarder(jones: CArray, a1: Array, a2: Array, delta: float | Array) -> CArray:
    """``J -> e^{-i delta/2} a (a.J) + e^{+i delta/2} b (b.J)`` with ``b = d x a`` (fast axis ``a``, slow axis ``b``)."""
    b1, b2 = -a2, a1
    pa = a1 * jones[:, 0] + a2 * jones[:, 1]
    pb = b1 * jones[:, 0] + b2 * jones[:, 1]
    fast, slow = _cis(-0.5 * delta), _cis(0.5 * delta)  # libm-free: identical bits in every runtime (a QWP's e^{i pi/4})
    fa, sb = cmul(fast, pa), cmul(slow, pb)
    return np.stack([fa * a1 + sb * b1, fa * a2 + sb * b2], axis=1)


def faraday_rotation(jones: CArray, along_axis: Array, theta: float) -> CArray:
    """A field turned by ``theta`` right-handed about a fixed axis, per ray: ``J -> R(b) J`` with ``R`` the rotation of the
    ``(e1, e2)`` plane about ``d`` and ``b = sign(d . axis) theta`` (``along_axis``: ``d . axis``). The basis is right-handed
    about ``d``, so the same turn in the laboratory reads with the opposite sign for light travelling against the axis."""
    turn = cis(theta)
    cos_b = turn.real
    sin_b = np.where(along_axis < 0.0, -turn.imag, turn.imag)
    return np.stack([cmul(cos_b, jones[:, 0]) - cmul(sin_b, jones[:, 1]), cmul(sin_b, jones[:, 0]) + cmul(cos_b, jones[:, 1])], axis=1)


def polarizer(jones: CArray, a1: Array, a2: Array, t_par: float, t_perp: float) -> CArray:
    """Partial linear polarizer: amplitude ``sqrt(t_par)`` along ``a``, ``sqrt(t_perp)`` along ``b = d x a``."""
    b1, b2 = -a2, a1
    pa = a1 * jones[:, 0] + a2 * jones[:, 1]
    pb = b1 * jones[:, 0] + b2 * jones[:, 1]
    ka, kb = np.sqrt(t_par), np.sqrt(t_perp)
    return np.stack([ka * pa * a1 + kb * pb * b1, ka * pa * a2 + kb * pb * b2], axis=1)


def polarization_overlap(ea: CArray, eb: CArray) -> float:
    """Amplitude overlap ``|e_a^* . e_b|`` of two 3D field vectors, each normalized first. 0 if either is zero."""
    na, nb = float(np.sqrt(cvdot(ea, ea).real)), float(np.sqrt(cvdot(eb, eb).real))
    if na == 0.0 or nb == 0.0:
        return 0.0
    overlap = cvdot(ea / na, eb / nb)
    return float(np.sqrt(overlap.real * overlap.real + overlap.imag * overlap.imag))
