"""The vector Debye integral of a field on the reference sphere (Richards and Wolf 1959; Leutenegger et al., Opt. Express 14,
11277 (2006) for its evaluation by matrix Fourier transform):

    E(r) = -i f' / lambda_m  Integral a(s) exp(i k s . r) dOmega,     H(r) likewise from s x a(s)

``at_points`` sums a sampling of the sphere (``quadrature.PolarRule`` or a ``CartesianGrid``'s cells) directly at any set of
points; ``scattered_on_grid`` / ``scattered_at_points`` sum samples at any directions (the traced rays of an imaging system, one
per ray tube) on a plane normal to the axis. ``on_grid`` evaluates a Cartesian sampling on a rectangular grid of a plane ``z`` as two matrix products,
``E_c = M_y G_c M_x^T`` with ``G = a w exp(i k cos(theta) z)`` and ``M_x[j, i] = exp(-i k u_i x_j)``. The Cartesian sampling
makes the result periodic in ``x`` and ``y`` with the period ``lambda_m / step``: a grid point is only meaningful within half
of it from the light's focus. On the grid ``x_j = j lambda_m / (M step)``, ``j = 0 .. M - 1``, the transform is a discrete
Fourier transform and the axial flux it carries equals the solid-angle sum of the sampled sphere exactly (Parseval;
``focus.Focus.plane_integrals`` is that check).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .objective import Objective, directions
from .quadrature import CartesianGrid

Array = NDArray[np.float64]
CArray = NDArray[np.complex128]

#: Most products (points x sphere samples) held in memory at once by ``at_points``.
CHUNK_PRODUCTS = 1 << 21


def at_points(objective: Objective, theta: Array, phi: Array, weight: Array, field: CArray, magnetic: CArray,
              points: Array) -> tuple[CArray, CArray]:
    """``(E, H)`` (3, P) at ``points`` (3, P), in mm, of the sphere samples ``(theta, phi, weight)`` with fields (3, J)."""

    s = directions(theta, phi)
    k = objective.k
    weighted_e = (field * weight).T
    weighted_h = (magnetic * weight).T
    count = points.shape[1]
    out_e = np.empty((len(s), count), dtype=np.complex128)
    out_h = np.empty((len(s), count), dtype=np.complex128)
    block = max(1, CHUNK_PRODUCTS // max(theta.size, 1))
    for start in range(0, count, block):
        p = points[:, start:start + block]
        phase = k * (p[0][:, None] * s[0][None, :] + p[1][:, None] * s[1][None, :] + p[2][:, None] * s[2][None, :])
        kernel = np.exp(1j * phase)
        out_e[:, start:start + block] = (kernel @ weighted_e).T
        out_h[:, start:start + block] = (kernel @ weighted_h).T
    return objective.prefactor * out_e, objective.prefactor * out_h


def on_grid(objective: Objective, grid: CartesianGrid, field: CArray, magnetic: CArray, x: Array, y: Array,
            z: float) -> tuple[CArray, CArray]:
    """``(E, H)`` (3, len(y), len(x)) on the plane ``z`` (mm) of a Cartesian sphere sampling with fields (3, N, N)."""

    k = objective.k
    axial = np.exp(1j * k * np.cos(grid.theta) * z) * grid.weight
    mx = np.exp(-1j * k * np.outer(x, grid.sines))
    my = np.exp(-1j * k * np.outer(y, grid.sines))
    out = []
    for values in (field, magnetic):
        out.append(np.stack([my @ (values[c] * axial) @ mx.T for c in range(len(values))]))
    return objective.prefactor * out[0], objective.prefactor * out[1]


def scattered_on_grid(k: float, directions: Array, amplitudes: CArray, a: Array, b: Array) -> CArray:
    """``sum_j A_cj exp(i k (s_tj a + s_sj b))`` (C, len(b), len(a)) of sphere samples at scattered directions: ``directions``
    (2, J) their transverse components along the plane's two axes, ``amplitudes`` (C, J) the weighted fields (the solid angle,
    the pupil phase and any prefactor included). The exponent is linear in ``a`` and ``b`` apart, so the sum is two matrix
    products, whatever the samples' layout."""

    mx = np.exp(1j * k * np.outer(directions[0], a))
    my = np.exp(1j * k * np.outer(b, directions[1]))
    return np.stack([my @ (amplitudes[c][:, None] * mx) for c in range(len(amplitudes))])


def scattered_at_points(k: float, directions: Array, amplitudes: CArray, a: Array, b: Array) -> CArray:
    """The same sum at the points ``(a_p, b_p)`` (C, P)."""

    out = np.empty((len(amplitudes), a.size), dtype=np.complex128)
    block = max(1, CHUNK_PRODUCTS // max(directions.shape[1], 1))
    for start in range(0, a.size, block):
        part = slice(start, start + block)
        kernel = np.exp(1j * k * (np.outer(a[part], directions[0]) + np.outer(b[part], directions[1])))
        out[:, part] = (kernel @ amplitudes.T).T
    return out


def axial_flux(e: CArray, h: CArray) -> Array:
    """``S_z = Re(E_x conj(H_y) - E_y conj(H_x))``."""

    return (e[0] * np.conj(h[1]) - e[1] * np.conj(h[0])).real


__all__ = ["CHUNK_PRODUCTS", "at_points", "axial_flux", "on_grid", "scattered_at_points", "scattered_on_grid"]
