"""Gaussian power on aperture regions other than a circle or a rectangle (``truncation.transmission_fraction``).

The integral ``I = integral_A exp(-2 d^T S d) dA`` (``d`` the offset from the chief-ray hit, ``S`` as in ``truncation.py``)
over one region ``A`` of ``optics/apertures.py``:

* circle and rectangle: ``truncation``'s own integrals (also used for holes of those shapes);
* ellipse (semi-axes ``a``, ``b`` along ``axis`` and across it): the affine map ``d = c + M z``, ``M = R diag(a, b)``, takes
  it onto the unit disc, where the beam is the Gaussian of ``S' = M^T S M`` centred at ``M^-1 c``:
  ``I = a b I_disc(S', M^-1 c, 1)`` -- exactly the circle integral, with its accuracy;
* d_shape and polygon: in the beam's whitened principal frame (``x = sqrt(lambda_min) e_min . d``, ``y = sqrt(lambda_max)
  e_max . d``) the beam is ``exp(-2 |P|^2)`` and, by the polar form of the area integral about the beam centre and
  Green's theorem, for any closed boundary ``P(t)`` (counter-clockwise)

      I = (1 / sqrt(det S)) oint (P x P') g(|P|^2) dt,       g(q) = (1 - exp(-2 q)) / (4 q),  g(0) = 1/2,

  which holds wherever the beam centre lies (inside: the boundary winds once around it). ``g`` is bounded and smooth
  (``-expm1(-2 q) / (4 q)``: no cancellation near the centre). A polygon's edges are straight lines in that frame, a
  d_shape's arc an elliptic arc. Each piece is integrated by Gauss-Legendre panels (``NumericalConfig.aperture_quad_sector_nodes``
  nodes) walked along it: a panel starting where the boundary lies ``rho`` from the beam centre spans at most
  ``max(1, rho) / (2 V)`` of the parameter (``V`` a bound of the boundary's speed), so the boundary stays at least
  ``max(1, rho) / 2`` from the centre over the panel and every panel is narrow against the scale on which the integrand
  changes there (``1`` near the centre, ``rho`` beyond it); an arc's panels are also at most ``ARC_PANEL_MAX_RAD`` wide.
  Nothing of the Gaussian is cut off; the result is exact to rounding (the panels resolve an analytic integrand whose
  nearest complex singularity lies at least about a panel width away), at a cost that grows with the logarithm of the
  region's size in beam radii.

Only ``+ - * /``, ``sqrt`` and the exact exponential and trigonometry of ``math.exact_trig``: the same bits in every runtime.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from ...config import NumericalConfig
from ...math.exact_trig import cos_sin, expm1
from ...scene.model import ApertureG
from .. import apertures
from .truncation import _circle_integral, _principal_axes, _rectangle_integral

Array = NDArray[np.float64]

#: Widest Gauss-Legendre panel (radians of the arc parameter) on a d_shape's arc.
ARC_PANEL_MAX_RAD = 0.5


def region_integral(ap: ApertureG, s: Array, det_s: float, chief_uv: tuple[float, float], cfg: NumericalConfig,
                    rule: tuple[Array, Array]) -> float:
    """``integral exp(-2 d^T S d) dA`` over the outer region of ``ap`` (its holes not removed; a circle's obstruction is),
    ``d`` measured from the chief-ray hit ``chief_uv``. The integrand is positive, so the integral lies between 0 and the
    whole plane's, ``pi / (2 sqrt(det S))``: the quadrature's rounding (a region far beside the beam, whose boundary terms
    cancel to a few ulps of either sign) is held within that range."""
    total = _outer_integral(ap, s, det_s, chief_uv, cfg, rule)
    whole = math.pi / (2.0 * math.sqrt(det_s)) if det_s > 0.0 else math.inf
    return min(max(total, 0.0), whole)


def _outer_integral(ap: ApertureG, s: Array, det_s: float, chief_uv: tuple[float, float], cfg: NumericalConfig,
                    rule: tuple[Array, Array]) -> float:
    cu, cv = chief_uv
    centre = (ap.offset_u - cu, ap.offset_v - cv)
    if ap.shape == apertures.CIRCLE:
        outer = _circle_integral(s, det_s, centre, ap.p1, cfg, rule)
        return outer - (_circle_integral(s, det_s, centre, ap.obstruction, cfg, rule) if ap.obstruction > 0 else 0.0)
    if ap.shape == apertures.RECTANGLE:
        return _rectangle_integral(s, det_s, centre, ap.p1, ap.p2, cfg, rule)
    if ap.shape == apertures.ELLIPSE:
        return _ellipse_integral(ap, s, det_s, centre, cfg, rule)
    return _boundary_integral(ap, s, det_s, centre, rule)


def _ellipse_integral(ap: ApertureG, s: Array, det_s: float, centre: tuple[float, float], cfg: NumericalConfig,
                      rule: tuple[Array, Array]) -> float:
    a, b = ap.p1, ap.p2
    c, sn = ap.axis
    m00, m01, m10, m11 = a * c, -b * sn, a * sn, b * c
    s00, s01, s11 = float(s[0, 0]), float(s[0, 1]), float(s[1, 1])
    t00 = m00 * m00 * s00 + 2.0 * m00 * m10 * s01 + m10 * m10 * s11
    t01 = m00 * m01 * s00 + (m00 * m11 + m10 * m01) * s01 + m10 * m11 * s11
    t11 = m01 * m01 * s00 + 2.0 * m01 * m11 * s01 + m11 * m11 * s11
    mapped = np.array([[t00, t01], [t01, t11]])
    det_mapped = det_s * (a * b) * (a * b)
    centre_mapped = ((c * centre[0] + sn * centre[1]) / a, (c * centre[1] - sn * centre[0]) / b)
    return a * b * _circle_integral(mapped, det_mapped, centre_mapped, 1.0, cfg, rule)


def _g(q: Array) -> Array:
    """``(1 - exp(-2 q)) / (4 q)`` (``1/2`` at ``q = 0``), without cancellation for small ``q``."""
    safe = np.where(q > 0.0, q, 1.0)
    return np.where(q > 0.0, -expm1(-2.0 * safe) / (4.0 * safe), 0.5)


def _boundary_integral(ap: ApertureG, s: Array, det_s: float, centre: tuple[float, float], rule: tuple[Array, Array]) -> float:
    lam_min, lam_max, e_min, e_max = _principal_axes(s, det_s)
    if not (lam_min > 0.0):
        return 0.0
    sa, sb = math.sqrt(lam_min), math.sqrt(lam_max)
    orientation = 1.0 if e_min[0] * e_max[1] - e_min[1] * e_max[0] > 0.0 else -1.0

    def white(du: float, dv: float) -> tuple[float, float]:
        return sa * (e_min[0] * du + e_min[1] * dv), sb * (e_max[0] * du + e_max[1] * dv)

    total = 0.0
    if ap.shape == apertures.POLYGON:
        corners = [white(centre[0] + x, centre[1] + y) for x, y in ap.vertices]
        for i in range(len(corners)):
            total += _segment(corners[i], corners[(i + 1) % len(corners)], rule)
    else:  # d_shape: the arc beyond the flat's corners, then the flat back to the arc's start
        half = apertures.flat_half_angle(ap)
        c, sn = ap.axis
        mid = white(*centre)
        along = white(ap.p1 * c, ap.p1 * sn)
        across = white(-ap.p1 * sn, ap.p1 * c)
        start, end = half, 2.0 * math.pi - half
        total += _arc(mid, along, across, start, end, rule)
        ce, se = _cos_sin(end)
        cs, ss = _cos_sin(start)
        first = (mid[0] + ce * along[0] + se * across[0], mid[1] + ce * along[1] + se * across[1])
        last = (mid[0] + cs * along[0] + ss * across[0], mid[1] + cs * along[1] + ss * across[1])
        total += _segment(first, last, rule)
    return orientation * total / (sa * sb)


def _segment(a: tuple[float, float], b: tuple[float, float], rule: tuple[Array, Array]) -> float:
    """``(A x E) int_0^1 g(|A + t E|^2) dt``, ``E = B - A``."""
    ex, ey = b[0] - a[0], b[1] - a[1]
    cross = a[0] * ey - a[1] * ex
    speed = math.sqrt(ex * ex + ey * ey)
    if cross == 0.0 or speed == 0.0:
        return 0.0
    panels = []
    t = 0.0
    while t < 1.0:
        px, py = a[0] + t * ex, a[1] + t * ey
        width = max(1.0, math.sqrt(px * px + py * py)) / (2.0 * speed)
        panels.append((t, min(1.0, t + width)))
        t = panels[-1][1]
    ts, ws = _nodes(panels, rule)
    px, py = a[0] + ts * ex, a[1] + ts * ey
    return cross * float(np.sum(ws * _g(px * px + py * py)))


def _arc(mid: tuple[float, float], along: tuple[float, float], across: tuple[float, float], start: float, end: float,
         rule: tuple[Array, Array]) -> float:
    """``int (P x P') g(|P|^2) dtheta`` over ``P = mid + cos(theta) along + sin(theta) across``, ``theta`` from ``start``
    to ``end``."""
    speed = math.sqrt(along[0] * along[0] + along[1] * along[1] + across[0] * across[0] + across[1] * across[1])
    panels = []
    theta = start
    while theta < end:
        c, sn = _cos_sin(theta)
        px, py = mid[0] + c * along[0] + sn * across[0], mid[1] + c * along[1] + sn * across[1]
        width = min(ARC_PANEL_MAX_RAD, max(1.0, math.sqrt(px * px + py * py)) / (2.0 * speed))
        panels.append((theta, min(end, theta + width)))
        theta = panels[-1][1]
    thetas, ws = _nodes(panels, rule)
    c, sn = cos_sin(thetas)
    px, py = mid[0] + c * along[0] + sn * across[0], mid[1] + c * along[1] + sn * across[1]
    dx, dy = -sn * along[0] + c * across[0], -sn * along[1] + c * across[1]
    return float(np.sum(ws * (px * dy - py * dx) * _g(px * px + py * py)))


def _cos_sin(theta: float) -> tuple[float, float]:
    c, sn = cos_sin(np.array([theta]))
    return float(c[0]), float(sn[0])


def _nodes(panels: list[tuple[float, float]], rule: tuple[Array, Array]) -> tuple[Array, Array]:
    nodes, weights = rule
    edges = np.array(panels)
    half = 0.5 * (edges[:, 1] - edges[:, 0])
    mid = 0.5 * (edges[:, 0] + edges[:, 1])
    return (mid[:, None] + half[:, None] * nodes[None, :]).ravel(), (half[:, None] * weights[None, :]).ravel()
