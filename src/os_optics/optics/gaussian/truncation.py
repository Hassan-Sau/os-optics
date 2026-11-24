"""Deterministic aperture transmission of an elliptical Gaussian beam.

Power through a surface aperture ``A`` (in the surface plane, coordinates ``(u, v)``):

    T = (cos(alpha) * 2 / (pi w_x w_y)) * integral_A exp(-2 * d^T S d) dA,      S = sum_k b_k b_k^T / w_k^2

where ``d`` is the offset from the chief-ray hit point, ``b_k`` are the in-plane ``(u, v)`` components of the beam
axes and ``alpha`` is the angle between the beam and the surface normal (``cos(alpha) = |d.n|``).

* circular aperture, footprint nearly round (aspect ratio at most ``CIRCLE_POLAR_MAX_ASPECT``): polar coordinates about
  the beam centre; the radial integral is closed form, ``int_0^R exp(-2 g r^2) r dr = (1 - exp(-2 g R^2)) / (4 g)``, the
  angular integral is a periodic trapezoid (beam centre inside) or a Gauss-Legendre sum over the subtended sector (beam
  centre outside);
* circular aperture, astigmatic footprint: in the beam's whitened principal frame (``x``, ``y``) the beam is
  ``exp(-2 (x^2 + y^2))`` and the disk an ellipse; ``x = x0 + A sin(theta)`` along its short semi-axis, the integral over
  the chord in ``y`` in closed form (``erf``), the one over ``theta`` by Gauss-Legendre panels sized by the scale of the
  factor that varies fastest on each piece (``_circle_integral_principal``);
* ellipse, d_shape and polygon apertures and holes: ``aperture_regions.py`` (an ellipse mapped onto the unit disc, the
  other shapes by a boundary integral in the beam's whitened frame);
* rectangular aperture: the square completed along one rectangle axis, ``d^T S d = m u^2 + c (v - v*(u))^2``; the inner
  integral over ``v`` in closed form (``erf``), the outer one over ``u`` by Gauss-Legendre panels sized by the scale of
  the factor that varies fastest on each piece; a beam whose axes are the rectangle's is the product of two ``erf``
  intervals (``_rectangle_integral``).

The polar sums sample the angular factor ``1 / g(phi)``, whose width about the footprint's long axis is the inverse aspect
ratio: they are exact to rounding for a nearly round footprint and wrong for a line-shaped one (a 100:1 line focus: 17 %
of its power), so astigmatic footprints use the principal frame. The cost of the other two methods does not grow with the
beam's astigmatism or orientation. The ``erf`` is ``math.exact_erf`` (``+ - x /``, ``sqrt`` and the exact exponential).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from ...config import NumericalConfig
from ...math.exact_erf import erf_interval
from ...math.exact_trig import asin, atan2, cos_sin
from ...math.exact_trig import exp as exact_exp
from ...math.vectors import dot
from ...run_control import checkpoint
from ...scene.model import ApertureG
from .state import GState

Array = NDArray[np.float64]


#: A circular aperture is integrated in polar form about the beam centre while the footprint's aspect ratio
#: ``sqrt(lambda_max / lambda_min)`` of ``S`` is at most this (exact to rounding there: the angular factor's Fourier
#: coefficients fall by ``(sqrt(lambda_max) - sqrt(lambda_min)) / (sqrt(lambda_max) + sqrt(lambda_min)) <= 1/3`` per
#: harmonic), else in the beam's whitened principal frame.
CIRCLE_POLAR_MAX_ASPECT = 2.0
#: Largest Gauss-Legendre panel, in radians of the ellipse angle, of the principal-frame circle integral.
CIRCLE_ANGLE_PANEL_MAX_RAD = 1.0

#: Newton steps on P_n for the Gauss-Legendre nodes: quadratic from the asymptotic first guess, converged (no change) within
#: a handful; the cap only bounds the loop.
LEGENDRE_NEWTON_MAX_STEPS = 100


class QuadratureRuleTooLarge(MemoryError):
    """The Gauss-Legendre rule of ``nodes`` nodes (``NumericalConfig.aperture_quad_sector_nodes``) needs more memory than
    the machine gives: the evaluation is refused by name (``validation/pipeline.py``: a FAIL naming the setting; a Project
    run: ``PROJECT_INVALID``), not ended by an internal error."""

    def __init__(self, nodes: int) -> None:
        super().__init__(self.refusal("numerical_settings.aperture_quad_sector_nodes", nodes))
        self.nodes = nodes

    @staticmethod
    def refusal(setting: str, nodes: int) -> str:
        """The refusal of ``nodes`` nodes, naming the setting as the caller states it (``setting``)."""
        return (f"{setting} is {nodes}: the Gauss-Legendre rule of that many nodes needs more memory than this machine gives "
                "(each of its arrays holds 8 bytes per node); lower the node count (Engine default 48)")


def gauss_legendre_rule(nodes: int) -> tuple[Array, Array]:
    """Gauss-Legendre nodes (ascending) / weights on [-1, 1]: Newton's method on ``P_n`` (three-term recurrence) from
    ``cos(pi (i - 1/4) / (n + 1/2))``, weights ``2 / ((1 - x^2) P_n'(x)^2)`` -- plain arithmetic, the same bits in every
    runtime (no LAPACK eigenvalues). Callers build it once per validation call and pass it down (no global cache).

    The nodes stop changing within a few steps, apart from last-bit corrections that may alternate between two
    neighbouring doubles; each step is a function of the nodes alone, so once the nodes repeat those of two steps
    before, every later step repeats too, and the nodes and derivatives that ``LEGENDRE_NEWTON_MAX_STEPS`` steps end
    on are known without taking them.

    Its work grows with ``nodes`` squared: each step of the recurrence is a checkpoint (``run_control``, no count), where the
    run's monitor may stop a long build; a rule the machine cannot hold raises ``QuadratureRuleTooLarge``."""
    n = int(nodes)
    if n > np.iinfo(np.intp).max:  # (no array of this platform indexes that many nodes: a 32-bit or WebAssembly build)
        raise QuadratureRuleTooLarge(n)
    try:
        return _gauss_legendre_rule(n)
    except MemoryError as exc:
        raise QuadratureRuleTooLarge(n) from exc


def _gauss_legendre_rule(n: int) -> tuple[Array, Array]:
    guess, _ = cos_sin(np.pi * (np.arange(1, n + 1) - 0.25) / (n + 0.5))
    x = guess
    derivative = np.ones_like(x)
    before: tuple[bytes, Array] | None = None  # the previous step's nodes (bytes) and derivatives
    for index in range(LEGENDRE_NEWTON_MAX_STEPS):
        p_prev, p = np.ones_like(x), x
        for k in range(2, n + 1):
            checkpoint(0)
            p_prev, p = p, ((2 * k - 1) * x * p - (k - 1) * p_prev) / k
        derivative = n * (x * p - p_prev) / ((x - 1.0) * (x + 1.0))
        step = p / derivative
        x_next = x - step
        if not bool(np.any(step != 0.0)):
            x = x_next
            break
        if before is not None and x_next.tobytes() == before[0]:  # a cycle of one or two steps from here on
            if (LEGENDRE_NEWTON_MAX_STEPS - index) % 2 == 0:  # the last step would start from the previous nodes, end on these
                derivative = before[1]
            else:
                x = x_next
            break
        before = (x.tobytes(), derivative)
        x = x_next
    weights = 2.0 / ((1.0 - x) * (1.0 + x) * derivative * derivative)
    return x[::-1].copy(), weights[::-1].copy()


def _s_matrix(state: GState, u_axis: Array, v_axis: Array) -> tuple[Array, float, float, float]:
    """2x2 matrix ``S`` in ``(u, v)`` coordinates, the beam radii and ``det S``.

    ``det S = ((b_x x b_y) / (w_x w_y))^2`` (the determinant of a sum of two rank-one terms): it keeps its relative accuracy
    for a strongly astigmatic beam turned against the ``(u, v)`` axes, where ``S_uu S_vv - S_uv^2`` would cancel."""
    wx, wy = state.radius("x"), state.radius("y")
    bxu = np.array([dot(state.bx, u_axis), dot(state.bx, v_axis)])
    byu = np.array([dot(state.by, u_axis), dot(state.by, v_axis)])
    s = np.outer(bxu, bxu) / (wx * wx) + np.outer(byu, byu) / (wy * wy)
    q = (float(bxu[0]) * float(byu[1]) - float(bxu[1]) * float(byu[0])) / (wx * wy)
    return s, wx, wy, q * q


def transmission_fraction(
    state: GState,
    aperture: ApertureG,
    chief_uv: tuple[float, float],
    u_axis: Array,
    v_axis: Array,
    cos_alpha: float,
    cfg: NumericalConfig,
    rule: tuple[Array, Array] | None = None,
) -> float:
    """Fraction of the beam power inside ``aperture`` (1.0 for an unbounded surface). Result is clipped to [0, 1].

    A loss below ``cfg.cross_runtime_abs_tol`` is quadrature rounding noise, not a physical loss: the fraction is then
    exactly 1.0. The quadrature (nodes, cosines, sines, exponentials, error functions) is libm- and LAPACK-free
    (``math.exact_trig``, ``math.exact_erf``), so the fraction has the same bits in every runtime.

    ``rule`` is an optional precomputed Gauss-Legendre rule of ``cfg.aperture_quad_sector_nodes`` nodes."""
    rule = rule if rule is not None else gauss_legendre_rule(cfg.aperture_quad_sector_nodes)
    if aperture.shape == "none":
        return 1.0
    s, wx, wy, det_s = _s_matrix(state, u_axis, v_axis)
    prefactor = cos_alpha * 2.0 / (math.pi * wx * wy)
    cu, cv = chief_uv
    if aperture.shape == "circle":
        outer = _circle_integral(s, det_s, (aperture.offset_u - cu, aperture.offset_v - cv), aperture.p1, cfg, rule)
        inner = _circle_integral(s, det_s, (aperture.offset_u - cu, aperture.offset_v - cv), aperture.obstruction, cfg, rule) if aperture.obstruction > 0 else 0.0
        total = outer - inner
    elif aperture.shape == "rectangle":
        total = _rectangle_integral(s, det_s, (aperture.offset_u - cu, aperture.offset_v - cv), aperture.p1, aperture.p2, cfg, rule)
    else:  # ellipse, d_shape, polygon (``aperture_regions``)
        from .aperture_regions import region_integral
        total = region_integral(aperture, s, det_s, (cu, cv), cfg, rule)
    if aperture.holes:  # openings through the element: their light is not on the surface
        from .aperture_regions import region_integral
        for hole in aperture.holes:
            total -= region_integral(hole, s, det_s, (cu, cv), cfg, rule)
    fraction = float(min(1.0, max(0.0, prefactor * total)))
    return 1.0 if 1.0 - fraction <= cfg.cross_runtime_abs_tol else fraction


def _g_of_angle(s: Array, c: Array, sn: Array) -> Array:
    """``d^T S d`` of the unit direction ``(cos phi, sin phi)`` given as ``c``, ``sn``."""
    return s[0, 0] * c * c + 2.0 * s[0, 1] * c * sn + s[1, 1] * sn * sn


def _principal_axes(s: Array, det_s: float) -> tuple[float, float, tuple[float, float], tuple[float, float]]:
    """``(lambda_min, lambda_max, e_min, e_max)`` of ``S`` in closed form: ``lambda_min = det S / lambda_max`` (no
    cancellation for a strongly astigmatic beam) and the unit eigenvector of ``lambda_max`` from the column of
    ``S - lambda_max I`` that does not cancel; ``e_min`` is ``e_max`` turned by +90 degrees."""
    a, b, c = float(s[0, 0]), float(s[0, 1]), float(s[1, 1])
    half = 0.5 * (a - c)
    r = math.sqrt(half * half + b * b)
    lam_max = 0.5 * (a + c) + r
    lam_min = det_s / lam_max
    vx, vy = (half + r, b) if half >= 0.0 else (b, r - half)
    norm = math.sqrt(vx * vx + vy * vy)
    e_max = (vx / norm, vy / norm) if norm > 0.0 else (1.0, 0.0)
    return lam_min, lam_max, (-e_max[1], e_max[0]), e_max


def _circle_integral(s: Array, det_s: float, center_rel: tuple[float, float], radius: float, cfg: NumericalConfig, rule: tuple[Array, Array]) -> float:
    """``integral_{disk} exp(-2 d^T S d) dA`` for a disk (centre relative to the beam centre): in polar form about the beam
    centre for a nearly round footprint, else in the beam's whitened principal frame (module docstring)."""
    if radius <= 0.0:
        return 0.0
    lam_min, lam_max, e_min, e_max = _principal_axes(s, det_s)
    if not (lam_min > 0.0) or lam_max <= CIRCLE_POLAR_MAX_ASPECT * CIRCLE_POLAR_MAX_ASPECT * lam_min:
        return _circle_integral_polar(s, center_rel, radius, cfg, rule)
    return _circle_integral_principal(lam_min, lam_max, e_min, e_max, center_rel, radius, cfg, rule)


def _acos_unit(c: float) -> float:
    """``arccos`` of ``c`` in ``[0, 1]`` (radians, libm-free)."""
    return float(atan2(math.sqrt((1.0 - c) * (1.0 + c)), c))


def _circle_integral_principal(
    lam_min: float, lam_max: float, e_min: tuple[float, float], e_max: tuple[float, float], center_rel: tuple[float, float], radius: float,
    cfg: NumericalConfig, rule: tuple[Array, Array],
) -> float:
    """``integral_{disk} exp(-2 d^T S d) dA`` in the beam's whitened principal frame.

    With ``x = sqrt(lambda_min) (e_min . d)``, ``y = sqrt(lambda_max) (e_max . d)`` the beam is ``exp(-2 (x^2 + y^2))``,
    ``dA = dx dy / sqrt(lambda_min lambda_max)``, and the disk is the ellipse of centre ``(x0, y0)`` and semi-axes
    ``A = sqrt(lambda_min) R`` along ``x`` and ``B = sqrt(lambda_max) R >= A`` along ``y``. Along its short axis
    ``x = x0 + A sin(theta)`` (``theta`` in ``[-pi/2, pi/2]``: no square-root end points), the chord
    ``|y - y0| <= B cos(theta)`` is integrated in closed form,

        integral = (1 / sqrt(lambda_min lambda_max)) int A cos(theta) exp(-2 x^2) sqrt(pi / 2) P(theta) d(theta),
        P = (erf(sqrt2 (y0 + B cos theta)) - erf(sqrt2 (y0 - B cos theta))) / 2,

    over ``|x| <= K`` (``K`` = ``aperture_quad_extent_sigmas``; the power left out is at most ``exp(-2 K^2)``) by
    Gauss-Legendre panels split at ``theta = 0`` and where a chord end lies within ``K`` of the beam axis (``P`` changes
    there, over ``1 / (B |sin theta|)``; elsewhere it is 0 or 1 to ``exp(-2 K^2)``). Each piece's panels are no wider than
    twice the smaller of ``1 / (A cos theta)`` (the Gaussian factor), that scale of ``P`` (inside those pieces) and half
    of ``CIRCLE_ANGLE_PANEL_MAX_RAD``, the factors' largest values on the piece: at most ``2 K + 1`` panels per piece."""
    nodes, weights = rule
    k = cfg.aperture_quad_extent_sigmas
    sa, sb = math.sqrt(lam_min), math.sqrt(lam_max)
    x0 = sa * (e_min[0] * center_rel[0] + e_min[1] * center_rel[1])
    y0 = sb * (e_max[0] * center_rel[0] + e_max[1] * center_rel[1])
    semi_a, semi_b = sa * radius, sb * radius
    sin_lo, sin_hi = max(-1.0, (-k - x0) / semi_a), min(1.0, (k - x0) / semi_a)
    if sin_hi <= sin_lo:
        return 0.0
    theta_lo, theta_hi = float(asin(sin_lo)), float(asin(sin_hi))
    windows: list[tuple[float, float]] = []  # theta where a chord end y0 +- B cos(theta) lies within K of the axis
    for lo, hi in (((-k - y0) / semi_b, (k - y0) / semi_b), ((y0 - k) / semi_b, (y0 + k) / semi_b)):
        c_lo, c_hi = max(0.0, lo), min(1.0, hi)
        if c_lo < c_hi:
            near, far = _acos_unit(c_hi), _acos_unit(c_lo)
            windows += [(near, far), (-far, -near)]
    cuts = sorted({theta_lo, theta_hi, *(t for w in windows for t in w if theta_lo < t < theta_hi), *((0.0,) if theta_lo < 0.0 < theta_hi else ())})
    ends = np.array(cuts)
    cos_end, sin_end = cos_sin(ends)
    segments: list[tuple[float, float, float]] = []
    for i in range(len(cuts) - 1):
        a, b = cuts[i], cuts[i + 1]
        if b <= a:
            continue
        scale = 0.5 * CIRCLE_ANGLE_PANEL_MAX_RAD
        max_cos = float(max(cos_end[i], cos_end[i + 1]))
        if semi_a * max_cos > 0.0:
            scale = min(scale, 1.0 / (semi_a * max_cos))
        middle = 0.5 * (a + b)
        if any(lo <= middle <= hi for lo, hi in windows):
            max_sin = float(max(abs(sin_end[i]), abs(sin_end[i + 1])))
            if semi_b * max_sin > 0.0:
                scale = min(scale, 1.0 / (semi_b * max_sin))
        segments.append((a, b, scale))
    if not segments:
        return 0.0
    theta, w = _composite_rule(segments, nodes, weights)
    cos_t, sin_t = cos_sin(theta)
    x = x0 + semi_a * sin_t
    half_chord = semi_b * cos_t
    root2 = math.sqrt(2.0)
    chord = erf_interval(root2 * (y0 - half_chord), root2 * (y0 + half_chord))
    total = float(np.sum(w * (semi_a * cos_t) * exact_exp(-2.0 * (x * x)) * chord))
    return total * math.sqrt(0.5 * math.pi) / (sa * sb)


def _circle_integral_polar(s: Array, center_rel: tuple[float, float], radius: float, cfg: NumericalConfig, rule: tuple[Array, Array]) -> float:
    """``integral_{disk} exp(-2 d^T S d) dA`` for a disk (centre relative to the beam centre) in polar form."""
    px, py = -center_rel[0], -center_rel[1]  # beam centre relative to the disk centre
    e = math.hypot(px, py)
    if e < radius:  # beam centre inside: full angle, radial interval [0, R(phi)]
        n = cfg.aperture_quad_angular
        phi = (np.arange(n) + 0.5) * (2.0 * math.pi / n)
        cos_phi, sin_phi = cos_sin(phi)
        b = cos_phi * px + sin_phi * py
        r_out = -b + np.sqrt(b * b + (radius * radius - e * e))
        g = _g_of_angle(s, cos_phi, sin_phi)
        vals = (1.0 - exact_exp(-2.0 * g * (r_out * r_out))) / (4.0 * g)
        return float(np.sum(vals) * (2.0 * math.pi / n))
    if e == 0.0 or radius <= 0.0:
        return 0.0
    # beam centre outside: sector centred on the direction to the disk centre, half angle asin(R/e)
    phi_c = float(atan2(-py, -px))
    delta = float(asin(min(1.0, radius / e)))
    x, w = rule
    psi = x * (math.pi / 2.0)  # phi = phi_c + delta sin(psi), psi in [-pi/2, pi/2]
    cos_psi, sin_psi = cos_sin(psi)
    phi = phi_c + delta * sin_psi
    jac = delta * cos_psi * (math.pi / 2.0)
    cos_phi, sin_phi = cos_sin(phi)
    b = cos_phi * px + sin_phi * py
    disc = np.maximum(b * b + (radius * radius - e * e), 0.0)
    root = np.sqrt(disc)
    r1, r2 = -b - root, -b + root
    g = _g_of_angle(s, cos_phi, sin_phi)
    vals = (exact_exp(-2.0 * g * (r1 * r1)) - exact_exp(-2.0 * g * (r2 * r2))) / (4.0 * g)
    return float(np.sum(vals * jac * w))


def _composite_rule(segments: list[tuple[float, float, float]], nodes: Array, weights: Array) -> tuple[Array, Array]:
    """Composite Gauss-Legendre points / weights on consecutive segments ``(a, b, scale)``: each split into equal panels no
    wider than ``2 * scale``."""
    points, panel_weights = [], []
    for a, b, scale in segments:
        count = max(1, int(math.ceil((b - a) / (2.0 * scale))))
        edges = np.linspace(a, b, count + 1)
        half = 0.5 * np.diff(edges)
        mid = 0.5 * (edges[:-1] + edges[1:])
        points.append((mid[:, None] + half[:, None] * nodes[None, :]).ravel())
        panel_weights.append((half[:, None] * weights[None, :]).ravel())
    return np.concatenate(points), np.concatenate(panel_weights)


def _outer_segments(u0: float, u1: float, scale: float, windows: list[tuple[float, float]], window_scale: float) -> list[tuple[float, float, float]]:
    """``[u0, u1]`` cut into segments of panel scale ``scale``, and ``window_scale`` (< ``scale``) inside ``windows``."""
    segments: list[tuple[float, float, float]] = []
    cursor = u0
    for lo, hi in sorted(windows):
        lo, hi = max(lo, cursor), min(hi, u1)
        if hi <= lo:
            continue
        if lo > cursor:
            segments.append((cursor, lo, scale))
        segments.append((lo, hi, window_scale))
        cursor = hi
    if cursor < u1:
        segments.append((cursor, u1, scale))
    return segments


def _rectangle_integral(
    s: Array, det_s: float, center_rel: tuple[float, float], half_u: float, half_v: float, cfg: NumericalConfig, rule: tuple[Array, Array],
) -> float:
    """``integral exp(-2 d^T S d) dA`` over a rectangle (centre relative to the beam centre, half sides along u and v).

    With the square completed along the rectangle axis of the larger diagonal entry of ``S`` (called ``v`` here, the other
    one ``u``),

        d^T S d = m u^2 + c (v - v*(u))^2,    c = S_vv,  v*(u) = -(S_uv / c) u,  m = det S / c,

    the inner integral over ``v`` is closed form, ``sqrt(pi / (2 c)) P(u)`` with
    ``P(u) = (erf(sqrt(2 c) (v_hi - v*(u))) - erf(sqrt(2 c) (v_lo - v*(u)))) / 2``. A beam whose axes are the rectangle's
    (``S_uv = 0``) is the product ``(pi / (2 sqrt(m c))) P_u P_v`` of two such intervals. Otherwise the outer integral of
    ``exp(-2 m u^2) P(u)`` runs over ``|u| <= K / sqrt(m)`` (``K`` = ``aperture_quad_extent_sigmas``; the power left out
    is at most ``exp(-2 K^2)``) on Gauss-Legendre panels no wider than ``2 / sqrt(m)``; where ``v*(u)`` crosses a
    rectangle edge ``v_e`` (at ``u_e = -v_e c / S_uv``), ``P`` changes over ``sqrt(c) / |S_uv|`` in ``u``, and within ``K``
    such scales of ``u_e`` the panels are no wider than twice that scale when it is the smaller one. At most
    ``3 ceil(K) + 5`` panels, however astigmatic or turned the beam: a line focus costs what a round beam costs."""
    nodes, weights = rule
    k = cfg.aperture_quad_extent_sigmas
    s_uu, s_uv, s_vv = float(s[0, 0]), float(s[0, 1]), float(s[1, 1])
    u_lo, u_hi = center_rel[0] - half_u, center_rel[0] + half_u
    v_lo, v_hi = center_rel[1] - half_v, center_rel[1] + half_v
    if s_uu > s_vv:  # the inner integral runs along the axis of the larger diagonal entry (|S_uv| / S_vv <= 1)
        s_uu, s_vv = s_vv, s_uu
        u_lo, u_hi, v_lo, v_hi = v_lo, v_hi, u_lo, u_hi
    if not (s_vv > 0.0 and det_s > 0.0):
        return 0.0
    c, m, slope = s_vv, det_s / s_vv, s_uv / s_vv
    root_c, root_m = math.sqrt(2.0 * c), math.sqrt(2.0 * m)
    if slope == 0.0:  # separable
        p_u = float(erf_interval(root_m * u_lo, root_m * u_hi))
        p_v = float(erf_interval(root_c * v_lo, root_c * v_hi))
        return (0.5 * math.pi / (math.sqrt(m) * math.sqrt(c))) * p_u * p_v
    scale_u = 1.0 / math.sqrt(m)
    reach_u = k * scale_u
    u0, u1 = max(u_lo, -reach_u), min(u_hi, reach_u)
    if u1 <= u0:
        return 0.0
    windows: list[tuple[float, float]] = []
    scale_edge = 1.0 / (math.sqrt(c) * abs(slope))
    if scale_edge < scale_u:
        windows = [(-edge / slope - k * scale_edge, -edge / slope + k * scale_edge) for edge in (v_lo, v_hi)]
    uu, wu = _composite_rule(_outer_segments(u0, u1, scale_u, windows, scale_edge), nodes, weights)
    p_v = erf_interval(root_c * (v_lo + slope * uu), root_c * (v_hi + slope * uu))
    return float(np.sum(wu * exact_exp(-2.0 * m * (uu * uu)) * p_v)) * math.sqrt(0.5 * math.pi / c)
