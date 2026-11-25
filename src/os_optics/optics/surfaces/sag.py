"""Conic / even-asphere sag profiles (``SagSurface``): sag, slope and curvature terms, closed-form conic intersection,
local normals and principal curvatures.

Local frame of a sag surface: vertex ``origin``, axis ``normal`` (``n``), transverse axes ``u``, ``v`` (``v = n x u``). A
point at transverse coordinates ``(a, b)`` lies at ``origin + a u + b v + z n`` with ``z = sag(s)``, where ``s = a^2 + b^2``
(rotational) or ``s = a^2`` (cylindrical: curved along ``u``, straight along ``v``):

    sag(s) = c s / (1 + S) + sum_j A_(2j+4) s^(j+2),        S = sqrt(1 - (1 + k) c^2 s)

``c`` is the vertex curvature (positive when the centre of curvature lies on the ``+n`` side), ``k`` the conic constant and
``A_4, A_6, ...`` the even-asphere coefficients. The profile is defined where ``1 - (1 + k) c^2 s >= 0`` (its *domain*).

Without polynomial terms the surface is the vertex branch of the quadric ``c (s + (1 + k) z^2) - 2 z = 0``: on that quadric
``1 - (1 + k) c z = +-S``, and the sag graph is exactly its part with ``1 - (1 + k) c z >= 0``.

Every derivative is written with ``S`` multiplied through, so it stays finite at the edge of the domain (``S = 0``, e.g.
the rim of a hemisphere):

* ``hS = c + 2 S P1(s)``  with ``P1 = d(polynomial)/ds``: the slope is ``dz/drho = rho hS / S``;
* unit normal toward ``+n``: ``(-a hS, -b hS, S) / WS`` with ``WS = sqrt(S^2 + s hS^2)``;
* principal curvatures (positive when the surface bends toward its ``+n`` side, i.e. the centre of curvature lies there):
  along the meridian (rotational) / the curved axis (cylindrical)
  ``kappa_1 = (hS S^2 + (1 + k) c^3 s + 4 s S^3 P2(s)) / WS^3`` (``P2 = d^2(polynomial)/ds^2``), and along the parallel
  circle ``kappa_2 = hS / WS`` (rotational) or ``0`` along the straight axis (cylindrical). The meridians and parallels of
  a surface of revolution (and the two axes of a cylinder) are its lines of curvature, so these are the principal
  directions.

Axicon term. A surface may add ``L rho`` (``SurfaceG.linear``; ``rho = sqrt(s)``): ``sag(s) += L sqrt(s)``,
``dz/ds += L / (2 sqrt(s))`` and the unit normal becomes ``(-(a hS + r_a L S), -(b hS + r_b L S), S) / |.|`` with the
radial unit components ``r_a = a / rho``, ``r_b = b / rho`` (a cylinder: ``r_a = sign(a)``, ``r_b = 0``). The apex
(``rho = 0``) has no tangent plane: within ``AXICON_APEX_RADIUS_MM`` of it ``r_a = r_b = 0`` (the normal of the profile
without the cone term, the surface axis for a pure cone). A surface with an axicon term has no principal curvatures for
the Gaussian beam model (``principal_curvatures`` returns ``None``; the tracer invalidates the Gaussian state there).
Every function below computes exactly what it did before when ``L = 0``.

Two-variable shapes (``SurfaceG.freeform``: Zernike, XY polynomial, biconic, toroid) are ``freeform.py``: the functions below
that take ``(a, b)`` hand such a surface over to it; ``transverse_s`` and the functions of ``s`` alone never see one.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from ...contracts.optics import AXICON_APEX_RADIUS_MM
from ...math.vectors import correctly_rounded_sum, dot
from ...scene.model import SurfaceG
from . import freeform

Array = NDArray[np.float64]


@dataclass(frozen=True)
class SagCurvature:
    """Local differential geometry of a sag surface at one point (global frame).

    ``normal``: unit normal toward the surface's ``+normal`` side; ``k1`` / ``k2``: principal normal curvatures (1/mm,
    positive when the centre of curvature lies on the ``+normal`` side) along the unit tangents ``dir1`` (meridian /
    curved axis) and ``dir2`` (parallel circle / straight axis)."""

    normal: Array
    k1: float
    k2: float
    dir1: Array
    dir2: Array


def transverse_s(surf: SurfaceG, a: Array, b: Array) -> Array:
    """``s = rho^2``: ``a^2 + b^2`` (rotational) or ``a^2`` (cylindrical)."""
    return a * a if surf.cylindrical else a * a + b * b


def sqrt_term(surf: SurfaceG, s: Array) -> tuple[Array, NDArray[np.bool_]]:
    """``S = sqrt(1 - (1 + k) c^2 s)`` and the domain mask (``S`` is 0 outside the domain)."""
    arg = 1.0 - (1.0 + surf.conic) * surf.curvature * surf.curvature * s
    ok = arg >= 0.0
    return np.sqrt(np.where(ok, arg, 0.0)), ok


def _poly(surf: SurfaceG, s: Array, order: int) -> Array:
    """``order``-th derivative with respect to ``s`` of ``sum_j A_(2j+4) s^(j+2)`` (order 0, 1 or 2)."""
    total = np.zeros_like(s)
    for j, coefficient in enumerate(surf.coefficients):
        power = j + 2
        factor = 1.0
        for m in range(order):
            factor *= power - m
        total = total + (coefficient * factor) * s ** (power - order)
    return total


def sag(surf: SurfaceG, s: Array) -> tuple[Array, NDArray[np.bool_]]:
    """Sag ``z(s)`` and the domain mask (``z`` is NaN outside the domain)."""
    big_s, ok = sqrt_term(surf, s)
    z = surf.curvature * s / (1.0 + big_s)
    if surf.coefficients:
        z = z + _poly(surf, s, 0)
    if surf.linear != 0.0:
        z = z + surf.linear * np.sqrt(s)
    return np.where(ok, z, np.nan), ok


def sag_ds(surf: SurfaceG, s: Array, big_s: Array) -> Array:
    """``dz/ds = c / (2 S) + P1(s)`` (infinite at the domain edge when ``c != 0``)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        base = np.where(big_s > 0.0, surf.curvature / (2.0 * np.where(big_s > 0.0, big_s, 1.0)), np.inf)
    base = np.where(surf.curvature == 0.0, 0.0, base)
    total = base + (_poly(surf, s, 1) if surf.coefficients else 0.0)
    if surf.linear != 0.0:  # d(L sqrt(s)) / ds, infinite at the apex
        with np.errstate(divide="ignore", invalid="ignore"):
            total = total + np.where(s > 0.0, surf.linear / (2.0 * np.sqrt(np.where(s > 0.0, s, 1.0))), np.inf)
    return total


def sag_dtau(surf: SurfaceG, a: Array, b: Array, da: Array, db: Array, s: Array, big_s: Array) -> Array:
    """Derivative of the sag along a ray ``(a, b) + tau (da, db)``: ``dz/ds ds/dtau``, with the axicon term's part written
    as ``L (r_a da + r_b db)`` (finite at the apex, where it is the one-sided ``L |d_t|`` of a ray leaving the apex)."""
    ds_dtau = 2.0 * a * da if surf.cylindrical else 2.0 * (a * da + b * db)
    if surf.linear == 0.0:
        return sag_ds(surf, s, big_s) * ds_dtau
    base = sag_ds(replace(surf, linear=0.0), s, big_s) * ds_dtau
    r_a, r_b = radial_units(surf, a, b)
    along = r_a * da + (0.0 if surf.cylindrical else r_b * db)
    at_apex = (r_a == 0.0) & (r_b == 0.0)
    transverse = np.abs(da) if surf.cylindrical else np.sqrt(da * da + db * db)
    return base + surf.linear * np.where(at_apex, transverse, along)


def conic_roots(surf: SurfaceG, a0: Array, b0: Array, z0: Array, da: Array, db: Array, dz: Array) -> tuple[Array, Array]:
    """Both ray parameters ``tau`` where ``(a0, b0, z0) + tau (da, db, dz)`` meets the quadric
    ``c (s + (1 + k) z^2) - 2 z = 0`` (``inf`` when there is no real root).

    ``alpha tau^2 + 2 beta tau + gamma = 0`` with ``alpha = c (|d_t|^2 + (1 + k) dz^2)``,
    ``beta = c (p_t.d_t + (1 + k) z0 dz) - dz``, ``gamma = c (|p_t|^2 + (1 + k) z0^2) - 2 z0`` (``p_t``, ``d_t``: the
    transverse parts, only ``a`` for a cylinder), solved without cancellation: ``q = -(beta + sgn(beta) sqrt(D))``,
    ``tau_1 = q / alpha``, ``tau_2 = gamma / q`` (a ray parallel to a paraboloid's axis has ``alpha = 0``: one root)."""
    c, k1 = surf.curvature, 1.0 + surf.conic
    if surf.cylindrical:
        p2, pd, d2 = a0 * a0, a0 * da, da * da
    else:
        p2, pd, d2 = a0 * a0 + b0 * b0, a0 * da + b0 * db, da * da + db * db
    alpha = c * (d2 + k1 * dz * dz)
    beta = c * (pd + k1 * z0 * dz) - dz
    gamma = c * (p2 + k1 * z0 * z0) - 2.0 * z0
    disc = beta * beta - alpha * gamma
    has = disc >= 0.0
    root = np.sqrt(np.where(has, disc, 0.0))
    q = -(beta + np.where(beta >= 0.0, root, -root))
    with np.errstate(divide="ignore", invalid="ignore"):
        tau1 = np.where(has & (alpha != 0.0), q / np.where(alpha != 0.0, alpha, 1.0), np.inf)
        tau2 = np.where(has & (q != 0.0), gamma / np.where(q != 0.0, q, 1.0), np.inf)
    return tau1, tau2


def on_conic_branch(surf: SurfaceG, z: Array) -> NDArray[np.bool_]:
    """Points of the quadric that belong to the sag graph: ``1 - (1 + k) c z >= 0``."""
    return 1.0 - (1.0 + surf.conic) * surf.curvature * z >= 0.0


def _slope_terms(surf: SurfaceG, s: Array) -> tuple[Array, Array, Array]:
    """``(S, hS, WS)`` at ``s`` (see the module docstring)."""
    big_s, _ = sqrt_term(surf, s)
    h_s = np.full_like(s, surf.curvature)
    if surf.coefficients:
        h_s = h_s + 2.0 * big_s * _poly(surf, s, 1)
    w_s = np.sqrt(big_s * big_s + s * h_s * h_s)
    return big_s, h_s, w_s


def local_normals(surf: SurfaceG, a: Array, b: Array) -> Array:
    """Global unit normals toward the ``+normal`` side at local points ``(a, b)`` on the surface (``(N, 3)``)."""
    if surf.freeform is not None:
        return freeform.local_normals(surf, a, b)
    s = transverse_s(surf, a, b)
    big_s, h_s, w_s = _slope_terms(surf, s)
    if surf.linear != 0.0:
        return _axicon_normals(surf, a, b, s, big_s, h_s)
    w_s = np.where(w_s > 0.0, w_s, 1.0)
    na = -a * h_s / w_s
    nb = np.zeros_like(a) if surf.cylindrical else -b * h_s / w_s
    nz = big_s / w_s
    return na[:, None] * surf.u_axis + nb[:, None] * surf.v_axis + nz[:, None] * surf.normal


def radial_units(surf: SurfaceG, a: Array, b: Array) -> tuple[Array, Array]:
    """Unit radial components ``(r_a, r_b)`` of the axicon term's slope at ``(a, b)``; ``(0, 0)`` within
    ``AXICON_APEX_RADIUS_MM`` of the apex."""
    if surf.cylindrical:
        away = np.abs(a) > AXICON_APEX_RADIUS_MM
        return np.where(away, np.sign(a), 0.0), np.zeros_like(a)
    rho = np.sqrt(a * a + b * b)
    away = rho > AXICON_APEX_RADIUS_MM
    safe = np.where(away, rho, 1.0)
    return np.where(away, a / safe, 0.0), np.where(away, b / safe, 0.0)


def _axicon_normals(surf: SurfaceG, a: Array, b: Array, s: Array, big_s: Array, h_s: Array) -> Array:
    """Unit normals toward ``+normal`` of a profile with an axicon term (module docstring)."""
    r_a, r_b = radial_units(surf, a, b)
    ga = a * h_s + r_a * surf.linear * big_s
    gb = np.zeros_like(a) if surf.cylindrical else b * h_s + r_b * surf.linear * big_s
    w = np.sqrt(ga * ga + gb * gb + big_s * big_s)
    w = np.where(w > 0.0, w, 1.0)
    return (-ga / w)[:, None] * surf.u_axis + (-gb / w)[:, None] * surf.v_axis + (big_s / w)[:, None] * surf.normal


def principal_curvatures(surf: SurfaceG, point: Array) -> SagCurvature | None:
    """Normal and principal curvatures at a surface point (global). ``None`` outside the profile's domain and on a
    surface with an axicon term (no curvature the Gaussian beam model can use)."""
    if surf.linear != 0.0:
        return None
    rel = point - surf.origin
    a, b = float(dot(rel, surf.u_axis)), float(dot(rel, surf.v_axis))
    if surf.freeform is not None:
        found = freeform.principal_curvatures(surf, a, b)
        return None if found is None else SagCurvature(*found)
    s_arr = transverse_s(surf, np.array([a]), np.array([b]))
    big_s_arr, ok = sqrt_term(surf, s_arr)
    if not bool(ok[0]):
        return None
    s, big_s = float(s_arr[0]), float(big_s_arr[0])
    _, h_arr, w_arr = _slope_terms(surf, s_arr)
    h_s, w_s = float(h_arr[0]), float(w_arr[0])
    if w_s <= 0.0:
        return None
    c, k1 = surf.curvature, 1.0 + surf.conic
    p2 = float(_poly(surf, s_arr, 2)[0]) if surf.coefficients else 0.0
    kappa_1 = (h_s * big_s * big_s + k1 * (c * c * c) * s + 4.0 * s * (big_s * big_s * big_s) * p2) / (w_s * w_s * w_s)
    u, v, n = surf.u_axis, surf.v_axis, surf.normal
    if surf.cylindrical:
        normal = (-a * h_s * u + big_s * n) / w_s
        dir1 = (big_s * u + a * h_s * n) / w_s
        return SagCurvature(normal, kappa_1, 0.0, dir1, v.copy())
    rho = float(np.sqrt(s))
    ra, rb = (a / rho, b / rho) if rho > 0.0 else (1.0, 0.0)
    normal = (-a * h_s * u - b * h_s * v + big_s * n) / w_s
    dir1 = (ra * big_s * u + rb * big_s * v + rho * h_s * n) / w_s
    dir2 = -rb * u + ra * v
    return SagCurvature(normal, kappa_1, h_s / w_s, dir1, dir2)


def domain_radius(surf: SurfaceG) -> float:
    """Largest ``rho`` of the profile's domain (``inf`` for paraboloids, hyperboloids and planes)."""
    k1c2 = (1.0 + surf.conic) * surf.curvature * surf.curvature
    return float(1.0 / np.sqrt(k1c2)) if k1c2 > 0.0 else float("inf")


def sag_bounds(surf: SurfaceG, rho_max: float) -> tuple[float, float]:
    """Bounds ``(z_min, z_max)`` of the sag over ``0 <= rho <= rho_max`` (within the domain): the conic part is monotonic
    in ``rho``, each polynomial term is bounded by ``|A| rho_max^(2j+4)`` and the axicon term lies between 0 and
    ``L rho_max``; a two-variable shape by ``freeform.sag_range``."""
    if surf.freeform is not None:
        return freeform.sag_range(surf, 0.0, rho_max)
    rho = min(rho_max, domain_radius(surf))
    s = np.array([rho * rho])
    big_s, _ = sqrt_term(surf, s)
    base = float(surf.curvature * s[0] / (1.0 + big_s[0]))
    poly = correctly_rounded_sum(abs(coefficient) * rho ** (2 * (j + 2)) for j, coefficient in enumerate(surf.coefficients))
    if surf.linear == 0.0:
        return min(0.0, base) - poly, max(0.0, base) + poly
    cone = surf.linear * rho
    return min(0.0, base) + min(0.0, cone) - poly, max(0.0, base) + max(0.0, cone) + poly
