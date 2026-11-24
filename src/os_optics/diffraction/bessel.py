"""The vector Debye integral of a rotationally symmetric pupil with one polarization, reduced to three Bessel integrals in
``theta`` (Richards and Wolf 1959; Novotny and Hecht, Principles of Nano-Optics, section 3.6), evaluated by Gauss-Legendre
quadrature. With the pupil amplitude ``A(theta)`` (illumination at ``h(theta)``, ``sqrt(cos(theta))``, and any phase that
depends on ``theta`` alone) and the transmissions ``t_p``, ``t_s``:

    I0 = Integral A sin(theta) (t_s + t_p cos(theta)) J0(k rho sin(theta)) exp(i k z cos(theta)) dtheta
    I1 = Integral A sin(theta)^2 t_p J1(...) exp(...) dtheta
    I2 = Integral A sin(theta) (t_s - t_p cos(theta)) J2(...) exp(...) dtheta

and, with ``C = pi (-i f' / lambda_m)``, for the pupil Jones vector ``(J_x, J_y)`` at the observation point
``(rho, phi, z)``:

    E_x = C [J_x (I0 + I2 cos(2 phi)) + J_y I2 sin(2 phi)]
    E_y = C [J_x I2 sin(2 phi) + J_y (I0 - I2 cos(2 phi))]
    E_z = -2 i C I1 [J_x cos(phi) + J_y sin(phi)]

The magnetic field is the same expression for the Jones vector ``(-J_y, J_x)`` with ``t_p`` and ``t_s`` exchanged
(``s x e_p = e_s``, ``s x e_s = -e_p``). This is an algorithm independent of the direct sphere sum of ``debye.py``: the
azimuthal integral is done analytically.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy.special import j0, j1, jv

from .objective import Objective

Array = NDArray[np.float64]
CArray = NDArray[np.complex128]

#: Most products (points x theta nodes) held in memory at once.
CHUNK_PRODUCTS = 1 << 21
BESSEL_ORDER_TWO = 2
#: Below this argument ``J2`` is evaluated directly; above it by the recurrence ``J2 = 2 J1 / x - J0`` (relative rounding
#: ``1e-16 / x^2``).
RECURRENCE_FROM = 0.1


def _bessel_012(arg: Array) -> tuple[Array, Array, Array]:
    b0, b1 = j0(arg), j1(arg)
    small = arg < RECURRENCE_FROM
    b2 = np.where(small, 0.0, 2.0 * b1 / np.where(small, 1.0, arg) - b0)
    if small.any():
        b2[small] = jv(BESSEL_ORDER_TWO, arg[small])
    return b0, b1, b2


def _integrals(k: float, theta: Array, weight: Array, amplitude: CArray, tp: CArray, ts: CArray, rho: Array, z: Array,
               swap: bool) -> tuple[CArray, CArray, CArray]:
    if swap:
        tp, ts = ts, tp
    st, ct = np.sin(theta), np.cos(theta)
    base = weight * amplitude * st
    g0 = base * (ts + tp * ct)
    g1 = base * st * tp
    g2 = base * (ts - tp * ct)
    arg = k * rho[:, None] * st[None, :]
    axial = np.exp(1j * k * z[:, None] * ct[None, :])
    b0, b1, b2 = _bessel_012(arg)
    return (b0 * axial) @ g0, (b1 * axial) @ g1, (b2 * axial) @ g2


def _field(c: complex, jx: complex, jy: complex, i0: CArray, i1: CArray, i2: CArray, phi: Array) -> CArray:
    c2, s2 = np.cos(2.0 * phi), np.sin(2.0 * phi)
    return np.stack((c * (jx * (i0 + i2 * c2) + jy * i2 * s2),
                     c * (jx * i2 * s2 + jy * (i0 - i2 * c2)),
                     -2j * c * i1 * (jx * np.cos(phi) + jy * np.sin(phi))))


def symmetric_field(objective: Objective, theta: Array, weight: Array, amplitude: CArray, tp: CArray, ts: CArray,
                    jones: tuple[complex, complex], points: Array) -> tuple[CArray, CArray]:
    """``(E, H)`` (3, P) at ``points`` (3, P) in mm, of the symmetric pupil sampled at the Gauss-Legendre nodes ``theta``
    with weights ``weight`` (``dtheta``), amplitude ``A(theta)`` and transmissions (module docstring)."""

    x, y, z = points
    rho = np.sqrt(x * x + y * y)
    phi = np.arctan2(y, x)
    c = math.pi * objective.prefactor
    jx, jy = jones
    count = rho.size
    out_e = np.empty((len(points), count), dtype=np.complex128)
    out_h = np.empty((len(points), count), dtype=np.complex128)
    block = max(1, CHUNK_PRODUCTS // max(theta.size, 1))
    for start in range(0, count, block):
        part = slice(start, start + block)
        i0, i1, i2 = _integrals(objective.k, theta, weight, amplitude, tp, ts, rho[part], z[part], swap=False)
        out_e[:, part] = _field(c, jx, jy, i0, i1, i2, phi[part])
        i0, i1, i2 = _integrals(objective.k, theta, weight, amplitude, tp, ts, rho[part], z[part], swap=True)
        out_h[:, part] = _field(c, -jy, jx, i0, i1, i2, phi[part])
    return out_e, out_h


__all__ = ["symmetric_field"]
