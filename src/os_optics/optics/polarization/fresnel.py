"""Fresnel coefficients for a dielectric interface (Born & Wolf), complex-safe for TIR."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
CArray = NDArray[np.complex128]


@dataclass(frozen=True)
class FresnelResult:
    r_s: CArray  # amplitude reflection (E_s' / E_s), triad (s, d x s)
    r_p: CArray
    tau_s: CArray  # energy-normalized transmission amplitude: |tau|^2 = T
    tau_p: CArray
    R_s: Array
    R_p: Array
    T_s: Array
    T_p: Array
    tir: NDArray[np.bool_]


def fresnel(n1: float, n2: float, cos_i: Array) -> FresnelResult:
    """s/p amplitude and power coefficients.

    ``cos_i`` in ``(0, 1]``; callers must exclude grazing rows (``cos_i`` at the numerical floor). Under total
    internal reflection ``cos_t`` is evaluated as ``+i sqrt(.)`` (decaying evanescent wave for exp[i(k.r - wt)]),
    giving ``|r| = 1`` and ``T = 0``.
    """
    cos_i = np.asarray(cos_i, dtype=np.float64)
    eta = n1 / n2
    sin_t_sq = (eta * eta) * (1.0 - cos_i * cos_i)
    cos_t = np.sqrt((1.0 - sin_t_sq).astype(np.complex128))
    tir = sin_t_sq > 1.0
    a, b = n1 * cos_i, n2 * cos_t
    # (a grazing row -- a ray the caller retired, whose row it does not read -- divides by zero: its values are not used, so
    # the floating-point warnings are not raised for it)
    with np.errstate(divide="ignore", invalid="ignore"):
        r_s = (a - b) / (a + b)
        t_s = 2.0 * a / (a + b)
        c, d = n2 * cos_i, n1 * cos_t
        r_p = (c - d) / (c + d)
        t_p = 2.0 * a / (c + d)
        flux = np.where(tir, 0.0, np.real(b) / a)  # (n2 cos_t) / (n1 cos_i), real part
        flux_p = np.where(tir, 0.0, np.real(b) / a)
        tau_s = t_s * np.sqrt(flux)
        tau_p = t_p * np.sqrt(flux_p)
    return FresnelResult(
        r_s=r_s,
        r_p=r_p,
        tau_s=tau_s,
        tau_p=tau_p,
        R_s=np.abs(r_s) ** 2,
        R_p=np.abs(r_p) ** 2,
        T_s=np.abs(tau_s) ** 2,
        T_p=np.abs(tau_p) ** 2,
        tir=tir,
    )
