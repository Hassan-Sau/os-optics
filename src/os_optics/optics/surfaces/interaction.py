"""Direction laws: reflection, Snell refraction with TIR, ideal thin lens."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ...math.vectors import dot, normalize

Array = NDArray[np.float64]


@dataclass(frozen=True)
class RefractionResult:
    d_out: Array  # refracted direction; for TIR rows the reflected direction (flagged by ``tir``)
    tir: NDArray[np.bool_]
    cos_i: Array  # >= 0
    sin_t_sq: Array  # (n1/n2)^2 sin^2(theta_i)
    n_oriented: Array  # normal oriented against the incident ray
    critical_margin: Array  # 1 - sin_t_sq (relative distance to TIR; <=0 means TIR)


def orient_against(d: Array, n: Array) -> tuple[Array, Array]:
    """Return ``(n_or, cos_i)`` with ``n_or`` flipped so that ``d . n_or <= 0`` and ``cos_i = -d . n_or >= 0``."""
    dn = dot(d, n)
    n_or = np.where((dn > 0.0)[:, None], -n, n)
    return n_or, np.abs(dn)


def refract(d: Array, n: Array, n1: float | Array, n2: float | Array, zero_tol: float) -> RefractionResult:
    """Vector Snell law with TIR (``k = 1 - eta^2 (1 - c^2) < 0``). ``n`` need not be oriented. ``n1`` / ``n2`` are one
    index or one per ray (a gradient-index medium: each ray's index where it meets the surface)."""
    n_or, cos_i = orient_against(d, n)
    eta = n1 / n2
    sin_t_sq = eta * eta * (1.0 - cos_i * cos_i)
    k = 1.0 - sin_t_sq
    tir = k < 0.0
    root = np.sqrt(np.where(tir, 0.0, k))
    eta_rows = eta[:, None] if np.ndim(eta) else eta
    t_dir = eta_rows * d + (eta * cos_i - root)[:, None] * n_or
    r_dir = d + 2.0 * cos_i[:, None] * n_or  # reflection off the oriented normal
    out = np.where(tir[:, None], r_dir, t_dir)
    out, _ = normalize(out, zero_tol)
    return RefractionResult(out, tir, cos_i, sin_t_sq, n_or, k)


def reflect_dir(d: Array, n: Array, zero_tol: float) -> tuple[Array, Array, Array]:
    """Mirror reflection about ``n`` (any orientation). Returns ``(d_out, n_oriented, cos_i)``."""
    n_or, cos_i = orient_against(d, n)
    out = d + 2.0 * cos_i[:, None] * n_or
    out, _ = normalize(out, zero_tol)
    return out, n_or, cos_i


def thin_lens_dir(
    point: Array, center: Array, axis: Array, d: Array, focal_length_mm: float, zero_tol: float
) -> tuple[Array, NDArray[np.bool_]]:
    """Ideal thin lens, slope form (exact for all angles): ``s' = s - h/f``, ``d' = normalize(s' + a)``.

    ``axis`` is the lens axis (unit); it is oriented along each ray. Rows with ``|d.a| <= zero_tol`` (ray parallel to
    the lens plane) are returned unchanged with ``ok=False``.
    """
    dz = dot(d, axis)
    a_or = np.where((dz < 0.0)[:, None], -axis, axis)
    dz_abs = np.abs(dz)
    ok = dz_abs > zero_tol
    safe = np.where(ok, dz_abs, 1.0)
    d_t = d - dz[:, None] * axis
    slope = d_t / safe[:, None]
    rel = point - center
    h = rel - dot(rel, axis)[:, None] * axis
    slope_out = slope - h / focal_length_mm
    out, _ = normalize(slope_out + a_or, zero_tol)
    return np.where(ok[:, None], out, d), ok
