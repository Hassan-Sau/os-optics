"""Two-variable sag surfaces (``SurfaceG.freeform``): Zernike, XY polynomial, biconic, toroid.

The shapes' formulas are ``_vendor/shared_contracts/common/sag_shapes.py`` (one kernel for floats, arrays and decimals). Here
they meet the Engine's surface frame: vertex ``origin``, axis ``normal``, transverse axes ``u``, ``v``, a point at ``(a, b)`` at
``origin + a u + b v + z n``.

A Zernike or XY polynomial shape adds its polynomial ``p(a, b)`` to the surface's rotational terms (``sag.py``), a function of
``s = a^2 + b^2``; a biconic or a toroid replaces them. Writing the rotational part as in ``sag.py`` with ``S`` multiplied
through, the unit normal toward ``+n`` is parallel to

    (-(a hS + S p_a), -(b hS + S p_b), S)

finite at the edge of the domain. The shapes' own slopes come in the same form, ``(z, h_a, h_b, w, ok)``.

Principal curvatures. With the first fundamental form ``E = 1 + z_a^2``, ``F = z_a z_b``, ``G = 1 + z_b^2`` and the second
``L = z_aa / W``, ``M = z_ab / W``, ``N = z_bb / W``, ``W = sqrt(1 + z_a^2 + z_b^2)``, the principal curvatures are the
eigenvalues of the shape operator: ``H +- sqrt(H^2 - K)``, ``H = (E N - 2 F M + G L) / (2 (E G - F^2))``, ``K = (L N - M^2) / (E G - F^2)``,
positive when the centre of curvature lies on the ``+n`` side; the principal direction of ``kappa`` is
``(du, dv) ~ (M - kappa F, -(L - kappa E))``. Second derivatives are analytic, in floats: only the chief ray's Gaussian step
needs them.

Determinism: ``+ - * /`` and square roots only, no libm function and no fused products (the same bits in every runtime).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..._vendor.shared_contracts.common.sag_shapes import BICONIC, SagOps, polynomial_bound, sag_slopes, sag_value
from ...math.vectors import dot
from ...scene.model import SurfaceG

Array = NDArray[np.float64]


def _array_root(value: Array) -> Array:
    return np.sqrt(np.where(value > 0.0, value, 0.0))


#: The kernel's arithmetic on arrays.
ARRAY_OPS = SagOps(float, _array_root)
#: A line meets a biconic in at most this many points (``(Q - z)^2 = z^2 (1 - S^2)`` has degree 4); a torus likewise.
QUARTIC_MEETINGS = 4


# ------------------------------------------------------------------------------------------------- the rotational part


def _rotational(surf: SurfaceG, a: Array, b: Array) -> tuple[Array, Array, Array, NDArray[np.bool_]]:
    """``(z, S, hS, ok)`` of the surface's rotational terms at ``s = a^2 + b^2`` (``sag.py``)."""
    s = a * a + b * b
    arg = 1.0 - (1.0 + surf.conic) * surf.curvature * surf.curvature * s
    ok = arg >= 0.0
    big_s = np.sqrt(np.where(ok, arg, 0.0))
    z = surf.curvature * s / (1.0 + big_s)
    if not surf.coefficients:
        return z, big_s, np.full_like(s, surf.curvature), ok
    first = np.zeros_like(s)
    power = s
    for j, coefficient in enumerate(surf.coefficients):
        z = z + coefficient * (power * s)
        first = first + (coefficient * (j + 2)) * power
        power = power * s
    return z, big_s, surf.curvature + 2.0 * big_s * first, ok


def sag_at(surf: SurfaceG, a: Array, b: Array) -> Array:
    """The sag at ``(a, b)`` (NaN outside the profile's domain)."""
    form = surf.freeform.form  # type: ignore[union-attr]
    z, ok = sag_value(form, a, b, ARRAY_OPS)
    if form.additive:
        base, _, _, base_ok = _rotational(surf, a, b)
        z, ok = base + z, base_ok
    return np.where(ok, z, np.nan)


def _slopes(surf: SurfaceG, a: Array, b: Array) -> tuple[Array, Array, Array, Array, NDArray[np.bool_]]:
    """``(z, h_a, h_b, w, ok)``: the unit normal toward ``+n`` is parallel to ``(-h_a, -h_b, w)``, and ``dz/da = h_a / w``."""
    form = surf.freeform.form  # type: ignore[union-attr]
    z, h_a, h_b, w, ok = sag_slopes(form, a, b, ARRAY_OPS)
    if not form.additive:
        return z, h_a, h_b, w, ok
    base, big_s, h_s, base_ok = _rotational(surf, a, b)
    return base + z, a * h_s + big_s * h_a, b * h_s + big_s * h_b, big_s, base_ok


def residual_slope(surf: SurfaceG, a: Array, b: Array, z: Array, da: Array, db: Array, dz: Array) -> tuple[Array, Array]:
    """``F = z - sag(a, b)`` (NaN outside the domain) and ``dF/dtau`` along ``(da, db, dz)``."""
    zs, h_a, h_b, w, ok = _slopes(surf, a, b)
    positive = w > 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        along = np.where(positive, (h_a * da + h_b * db) / np.where(positive, w, 1.0), np.inf)
    return np.where(ok, z - zs, np.nan), dz - along


def local_normals(surf: SurfaceG, a: Array, b: Array) -> Array:
    """Global unit normals toward the ``+normal`` side at local points ``(a, b)`` on the surface (``(N, 3)``)."""
    _, h_a, h_b, w, _ = _slopes(surf, a, b)
    length = np.sqrt(h_a * h_a + h_b * h_b + w * w)
    length = np.where(length > 0.0, length, 1.0)
    return (-h_a / length)[:, None] * surf.u_axis + (-h_b / length)[:, None] * surf.v_axis + (w / length)[:, None] * surf.normal


def record_solve(surf: SurfaceG, a: Array, b: Array, z: Array, done: NDArray[np.bool_], iterations: int) -> None:
    """Add one refinement to the surface's ``SolverStats``: how many rays it took, how many steps, the largest residual of a
    converged root, and how many rays did not converge."""
    stats = surf.freeform.stats  # type: ignore[union-attr]
    stats.solved_rays += int(done.size)
    stats.max_iterations = max(stats.max_iterations, iterations)
    stats.non_converged_rays += int(done.size - np.count_nonzero(done))
    if bool(done.any()):
        residual = np.abs(z[done] - sag_at(surf, a[done], b[done]))
        finite = residual[np.isfinite(residual)]
        if finite.size:
            stats.max_residual_mm = max(stats.max_residual_mm, float(finite.max()))


# --------------------------------------------------------------------------------------- second derivatives, curvature


def _polynomial_hessian(form: Any, a: float, b: float) -> tuple[float, float, float, float, float]:
    """``(p_a, p_b, p_aa, p_ab, p_bb)`` of a polynomial form at ``(a, b)``."""
    scale = form.scale
    big_x, big_y = a * scale, b * scale
    top_x, top_y = max(i for i, _, _ in form.terms), max(j for _, j, _ in form.terms)
    px, py = [1.0], [1.0]
    for _ in range(top_x):
        px.append(px[-1] * big_x)
    for _ in range(top_y):
        py.append(py[-1] * big_y)
    p_a = p_b = p_aa = p_ab = p_bb = 0.0
    for i, j, c in form.terms:
        if i:
            p_a += c * i * px[i - 1] * py[j]
        if j:
            p_b += c * j * px[i] * py[j - 1]
        if i > 1:
            p_aa += c * i * (i - 1) * px[i - 2] * py[j]
        if i and j:
            p_ab += c * i * j * px[i - 1] * py[j - 1]
        if j > 1:
            p_bb += c * j * (j - 1) * px[i] * py[j - 2]
    return p_a * scale, p_b * scale, p_aa * scale * scale, p_ab * scale * scale, p_bb * scale * scale


def _rotational_hessian(surf: SurfaceG, a: float, b: float) -> tuple[float, float, float, float, float] | None:
    """``(z_a, z_b, z_aa, z_ab, z_bb)`` of the rotational terms: ``z_a = 2 a g``, ``z_aa = 2 g + 4 a^2 g'``, ``z_ab = 4 a b g'``
    with ``g = dz/ds = c / (2 S) + P1``, ``g' = (1 + k) c^3 / (4 S^3) + P2``; ``None`` at or beyond the edge of the domain."""
    c, k1 = surf.curvature, 1.0 + surf.conic
    s = a * a + b * b
    arg = 1.0 - k1 * c * c * s
    if arg <= 0.0:
        return None
    big_s = math.sqrt(arg)
    g = c / (2.0 * big_s)
    dg = k1 * (c * c * c) / (4.0 * (big_s * big_s * big_s))
    power = s
    previous = 1.0
    for j, coefficient in enumerate(surf.coefficients):
        order = j + 2
        g += coefficient * order * power
        dg += coefficient * order * (order - 1) * previous
        previous, power = power, power * s
    return (2.0 * a * g, 2.0 * b * g, 2.0 * g + 4.0 * (a * a) * dg, 4.0 * (a * b) * dg, 2.0 * g + 4.0 * (b * b) * dg)


def _biconic_hessian(form: Any, a: float, b: float) -> tuple[float, float, float, float, float] | None:
    """``z = Q u``, ``Q = c_a a^2 + c_b b^2``, ``u = 1 / (1 + S)``, ``S^2 = 1 - A a^2 - B b^2``: product and chain rule."""
    big_a, big_b = form.conic_x, form.conic_y
    arg = 1.0 - big_a * (a * a) - big_b * (b * b)
    if arg <= 0.0:
        return None
    big_s = math.sqrt(arg)
    u = 1.0 / (1.0 + big_s)
    c_a, c_b = form.curvature_x, form.curvature_y
    q = c_a * (a * a) + c_b * (b * b)
    q_a, q_b = 2.0 * c_a * a, 2.0 * c_b * b
    s_a, s_b = -big_a * a / big_s, -big_b * b / big_s
    s_aa = -big_a / big_s - big_a * big_a * (a * a) / (big_s * big_s * big_s)
    s_bb = -big_b / big_s - big_b * big_b * (b * b) / (big_s * big_s * big_s)
    s_ab = -big_a * big_b * (a * b) / (big_s * big_s * big_s)
    u_a, u_b = -u * u * s_a, -u * u * s_b
    u_aa = 2.0 * (u * u * u) * (s_a * s_a) - u * u * s_aa
    u_bb = 2.0 * (u * u * u) * (s_b * s_b) - u * u * s_bb
    u_ab = 2.0 * (u * u * u) * (s_a * s_b) - u * u * s_ab
    return (q_a * u + q * u_a, q_b * u + q * u_b, 2.0 * c_a * u + 2.0 * q_a * u_a + q * u_aa,
            q_a * u_b + q_b * u_a + q * u_ab, 2.0 * c_b * u + 2.0 * q_b * u_b + q * u_bb)


def _toroid_hessian(form: Any, a: float, b: float) -> tuple[float, float, float, float, float] | None:
    """``z = z_b + p a^2 / (1 + V)``: ``z_a = p a / V``, ``z_b = z_b' / V``, ``z_aa = p / V^3``, ``z_ab = p^2 z_b' a / V^3``,
    ``z_bb = z_b'' / V + p^3 z_b'^2 a^2 / V^3`` (``p = 1 / (R - z_b)``, ``V^2 = 1 - p^2 a^2``)."""
    c, big_b = form.curvature, form.conic_y
    y2 = b * b
    arg = 1.0 - big_b * y2
    if arg <= 0.0:
        return None
    big_t = math.sqrt(arg)
    z = c * y2 / (1.0 + big_t)
    first = c * b / big_t
    second = c / (big_t * big_t * big_t) if big_b != 0.0 else c
    power, previous = y2, 1.0
    for i, alpha in enumerate(form.y_terms, start=1):
        z += alpha * power
        first += 2 * i * alpha * previous * b
        second += 2 * i * (2 * i - 1) * alpha * previous
        previous, power = power, power * y2
    if form.rotation is None:
        return 0.0, first, 0.0, 0.0, second
    distance = form.rotation - z
    if distance * form.rotation <= 0.0:
        return None
    p = 1.0 / distance
    root = 1.0 - p * p * (a * a)
    if root <= 0.0:
        return None
    v = math.sqrt(root)
    v3 = v * v * v
    return (p * a / v, first / v, p / v3, p * p * first * a / v3, second / v + (p * p * p) * (first * first) * (a * a) / v3)


def _hessian(surf: SurfaceG, a: float, b: float) -> tuple[float, float, float, float, float] | None:
    """``(z_a, z_b, z_aa, z_ab, z_bb)`` of the whole profile at ``(a, b)`` (``None``: at or beyond the edge of its domain)."""
    form = surf.freeform.form  # type: ignore[union-attr]
    if form.additive:
        base = _rotational_hessian(surf, a, b)
        if base is None:
            return None
        poly = _polynomial_hessian(form, a, b)
        return (base[0] + poly[0], base[1] + poly[1], base[2] + poly[2], base[3] + poly[3], base[4] + poly[4])
    return _biconic_hessian(form, a, b) if form.kind == BICONIC else _toroid_hessian(form, a, b)


def principal_curvatures(surf: SurfaceG, a: float, b: float) -> tuple[Array, float, float, Array, Array] | None:
    """``(normal, k1, k2, dir1, dir2)`` at ``(a, b)``, global (module docstring), ``dir2 = normal x dir1``; ``None`` at or beyond
    the edge of the profile's domain."""
    found = _hessian(surf, a, b)
    if found is None:
        return None
    z_a, z_b, z_aa, z_ab, z_bb = found
    u, v, n = surf.u_axis, surf.v_axis, surf.normal
    big_e, big_f, big_g = 1.0 + z_a * z_a, z_a * z_b, 1.0 + z_b * z_b
    w2 = 1.0 + z_a * z_a + z_b * z_b
    w = math.sqrt(w2)
    big_l, big_m, big_n = z_aa / w, z_ab / w, z_bb / w
    mean = (big_e * big_n - 2.0 * big_f * big_m + big_g * big_l) / (2.0 * w2)
    gauss = (big_l * big_n - big_m * big_m) / w2
    spread = math.sqrt(max(mean * mean - gauss, 0.0))
    k1, k2 = mean + spread, mean - spread
    # the principal direction of k1, from whichever row of (II - k I) (du, dv) = 0 is the stronger
    du, dv = big_m - k1 * big_f, -(big_l - k1 * big_e)
    du2, dv2 = big_n - k1 * big_g, -(big_m - k1 * big_f)
    if du2 * du2 + dv2 * dv2 > du * du + dv * dv:
        du, dv = du2, dv2
    if du == 0.0 and dv == 0.0:  # an umbilic: every direction is principal
        du, dv = 1.0, 0.0
    tangent = du * u + dv * v + (du * z_a + dv * z_b) * n
    dir1 = tangent / math.sqrt(float(dot(tangent, tangent)))
    normal = (-z_a * u - z_b * v + n) / w
    dir2 = np.cross(normal, dir1)
    return normal, k1, k2, dir1, dir2


# ---------------------------------------------------------------------------------------------------- bounds


def _rotational_range(surf: SurfaceG, rho_first: float, rho_last: float) -> tuple[float, float]:
    """Bounds of the rotational terms over ``rho_first <= rho <= rho_last`` (the conic part monotonic in ``rho``, each polynomial
    term at most ``|A| rho_last^(2j+4)``)."""
    k1c2 = (1.0 + surf.conic) * surf.curvature * surf.curvature
    if k1c2 > 0.0:
        rho_last = min(rho_last, 1.0 / math.sqrt(k1c2))
        rho_first = min(rho_first, rho_last)
    ends = []
    for rho in (rho_first, rho_last):
        s = rho * rho
        ends.append(surf.curvature * s / (1.0 + math.sqrt(max(1.0 - k1c2 * s, 0.0))))
    s = rho_last * rho_last
    poly, power = 0.0, s * s
    for coefficient in surf.coefficients:
        poly += abs(coefficient) * power
        power *= s
    return min(ends) - poly, max(ends) + poly


def _toroid_curve_range(form: Any, y: float) -> tuple[float, float, float, float]:
    """``(lo, hi, |z_b'|, |z_b''|)`` bounds of a toroid's curve over ``|b| <= y`` (within its domain)."""
    k1c2 = form.conic_y
    if k1c2 > 0.0:
        y = min(y, 1.0 / math.sqrt(k1c2))
    y2 = y * y
    big_t = math.sqrt(max(1.0 - k1c2 * y2, 0.0))
    c = form.curvature
    conic = c * y2 / (1.0 + big_t)
    lo, hi = min(0.0, conic), max(0.0, conic)
    if c != 0.0 and big_t <= 0.0:
        first, second = math.inf, math.inf
    else:
        smallest = min(big_t, 1.0)  # a hyperbolic section's T exceeds 1: 1 / T is at most 1
        first, second = abs(c) * y / smallest, abs(c) / (smallest * smallest * smallest)
    power, previous = y2, 1.0
    for i, alpha in enumerate(form.y_terms, start=1):
        size = abs(alpha) * power
        lo, hi = lo - size, hi + size
        first += 2 * i * abs(alpha) * previous * y
        second += 2 * i * (2 * i - 1) * abs(alpha) * previous
        previous, power = power, power * y2
    return lo, hi, first, second


def sag_range(surf: SurfaceG, rho_first: float, rho_last: float) -> tuple[float, float]:
    """Bounds of the sag over ``rho_first <= sqrt(a^2 + b^2) <= rho_last``, within the profile's domain. A polynomial: the
    rotational terms' range and ``sum |c| (rho_last / R)^(i + j)``. A biconic: ``|a|, |b| <= rho_last`` in ``Q / (1 + S)``. A toroid:
    its curve's range plus the rotation's part, at most ``|R - z_b| (1 - V)``."""
    if not math.isfinite(rho_last):
        return -math.inf, math.inf
    form = surf.freeform.form  # type: ignore[union-attr]
    if form.additive:
        lo, hi = _rotational_range(surf, rho_first, rho_last)
        bound = polynomial_bound(form.terms, rho_last * form.scale)
        return lo - bound, hi + bound
    if form.kind == BICONIC:
        s = rho_last * rho_last
        q_lo, q_hi = min(0.0, form.curvature_x, form.curvature_y) * s, max(0.0, form.curvature_x, form.curvature_y) * s
        root = math.sqrt(max(1.0 - max(0.0, form.conic_x, form.conic_y) * s, 0.0))
        return q_lo / (1.0 + root), q_hi / (1.0 + root)
    lo, hi, _, _ = _toroid_curve_range(form, rho_last)
    if form.rotation is None:
        return lo, hi
    near = min(abs(form.rotation - lo), abs(form.rotation - hi))
    far = max(abs(form.rotation - lo), abs(form.rotation - hi))
    if near <= 0.0:
        part = far
    else:
        p = 1.0 / near
        part = min(far, p * rho_last * rho_last / (1.0 + math.sqrt(max(1.0 - p * p * rho_last * rho_last, 0.0))))
    return (lo - part, hi) if form.rotation < 0.0 else (lo, hi + part)


def hessian_bound(surf: SurfaceG, rho_first: float, rho_last: float) -> float:
    """A bound of the norm of the sag's Hessian over ``rho_first <= sqrt(a^2 + b^2) <= rho_last`` (``inf`` where it reaches the
    edge of the profile's domain). A polynomial adds ``sum |c| d (d - 1) (rho_last / R)^(d - 2) / R^2``, ``d = i + j``, to the
    rotational terms' principal values; a biconic and a toroid use the product and chain rule of ``_biconic_hessian`` /
    ``_toroid_hessian`` with each factor at its largest."""
    del rho_first
    form = surf.freeform.form  # type: ignore[union-attr]
    if form.additive:
        k1c2 = (1.0 + surf.conic) * surf.curvature * surf.curvature
        root = math.sqrt(max(1.0 - k1c2 * rho_last * rho_last, 0.0)) if k1c2 > 0.0 else 1.0
        if surf.curvature != 0.0 and root <= 0.0:
            return math.inf
        total = abs(surf.curvature) / (root * root * root)
        s = rho_last * rho_last
        power = s
        for j, coefficient in enumerate(surf.coefficients):
            total += abs(coefficient) * (2 * j + 4) * (2 * j + 3) * power
            power *= s
        scale = form.scale
        reach = rho_last * scale
        for i, j, c in form.terms:
            degree = i + j
            size = abs(c) * degree * (degree - 1) * scale * scale
            for _ in range(degree - 2):
                size *= reach
            total += size
        return total
    if form.kind == BICONIC:
        s = rho_last * rho_last
        big_a = max(abs(form.conic_x), abs(form.conic_y))
        root = math.sqrt(max(1.0 - max(0.0, form.conic_x, form.conic_y) * s, 0.0))
        if big_a != 0.0 and root <= 0.0:
            return math.inf
        c = max(abs(form.curvature_x), abs(form.curvature_y))
        u = 1.0 / (1.0 + root)
        q, q1, q2 = c * s, 2.0 * c * rho_last, 2.0 * c
        if big_a == 0.0:
            return q2 * u
        s1 = big_a * rho_last / root
        s2 = big_a / root + big_a * big_a * s / (root * root * root)
        u1 = u * u * s1
        u2 = 2.0 * (u * u * u) * (s1 * s1) + u * u * s2
        return q2 * u + 2.0 * q1 * u1 + q * u2
    lo, hi, first, second = _toroid_curve_range(form, rho_last)
    if not (math.isfinite(first) and math.isfinite(second)):
        return math.inf
    if form.rotation is None:
        return second
    near = min(abs(form.rotation - lo), abs(form.rotation - hi))
    if (form.rotation - lo) * (form.rotation - hi) <= 0.0:
        return math.inf
    p = 1.0 / near
    root = 1.0 - p * p * rho_last * rho_last
    if root <= 0.0:
        return math.inf
    v = math.sqrt(root)
    v3 = v * v * v
    return p / v3 + 2.0 * p * p * first * rho_last / v3 + second / v + (p * p * p) * (first * first) * (rho_last * rho_last) / v3


def line_meetings(surf: SurfaceG) -> int:
    """The most points a straight line that does not run along the surface meets it at: the degree of the profile's equation
    along the line. A polynomial on rotational terms: ``e`` the larger of the polynomial's degree and the terms' even power,
    ``2 e`` where the base has a square root; a biconic: 4; a toroid: the curve's implicit degree, doubled by the rotation."""
    form = surf.freeform.form  # type: ignore[union-attr]
    if form.additive:
        highest = 0
        for j, coefficient in enumerate(surf.coefficients):
            if coefficient != 0.0:
                highest = j + 2
        e = max(max(i + j for i, j, _ in form.terms), 2 * highest, 1)
        return 2 * e if surf.curvature != 0.0 else e
    if form.kind == BICONIC:
        return QUARTIC_MEETINGS
    e = max(2 * len(form.y_terms), 1)
    implicit = 2 * e if form.curvature != 0.0 else e
    return 2 * implicit if form.rotation is not None else implicit


__all__ = ["ARRAY_OPS", "hessian_bound", "line_meetings", "local_normals", "principal_curvatures", "record_solve", "residual_slope",
           "sag_at", "sag_range"]
