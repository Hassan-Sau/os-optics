"""Second-harmonic generation of a Gaussian beam in a uniaxial crystal slab: undepleted pump, paraxial coupled waves.

Conventions (SI inside, mm and nm at the interface). A field ``E(r, t) = A u(r) exp(i (k z - omega t)) + c.c.`` carries the
intensity ``I = 2 n eps0 c |A|^2``; the second-order polarization at ``2 omega`` is ``P = 2 eps0 d_eff E_a E_b`` for two
photons of one wave (type I, or type 0 under quasi-phase matching) and ``4 eps0 d_eff E_a E_b`` for one photon of each of
two waves (type II), ``d_eff`` the tensor ``d_ijk`` contracted with the unit electric-field vectors of the harmonic and the
two fundamental waves (``d_eff_from_tensor``). In the slowly varying envelope approximation a plane wave gives

    I_2 = 2 omega^2 d_eff^2 L^2 I_a I_b f / (n_a n_b n_2 eps0 c^3) sinc^2(dk L / 2),   f = 1 (type I), 4 (type II)

with ``I_a I_b = I_1^2`` for type I and ``dk = k_2 - k_a - k_b``. For a Gaussian beam (waist radii ``w0x``, ``w0y`` along the
beam's two axes, Rayleigh ranges ``zx``, ``zy``) every slice ``dz`` of the crystal radiates into the harmonic Gaussian mode of
the same reduced q-parameter, with the amplitude ``prod_j (1 + i tau_j(z))^(-1/2)`` per axis (``tau_j = (z - z_waist) / z_j``),
and a harmonic generated at ``z`` whose energy walks off the driving polarization by the angle vector ``s`` (per unit path)
is displaced by ``s (L - z)`` at the exit. The harmonic power is (Boyd and Kleinman, J. Appl. Phys. 39, 3597 (1968),
generalized to an elliptical, simply astigmatic beam and to a walk-off direction off its axes)

    P_2 = K f P_a P_b L^2 H / (pi w0x w0y),   K = 2 omega^2 d_eff^2 / (n_a n_b n_2 eps0 c^3)

    H = (1 / L^2) Re int_0^L int_0^L F(u) F*(v) exp(-sum_j s_j^2 (u - v)^2 / w0_j^2) du dv,
    F(z) = exp(-i dk z) G(z) prod_j (1 + i tau_j(z))^(-1/2),

``s_j`` the components of the walk-off between the harmonic and the mean of the two fundamental waves on the beam axes, and
``G(z) = exp(-sum_j p_j^2 z^2 / (2 w0_j^2 (1 + i tau_j(z))))`` the overlap of two fundamental waves that walk off each other by
``p`` (type II; ``G = 1`` for one wave). In an absorbing crystal ``F(z)`` carries the real factor ``exp(-a z - b (L - z))``:
the fundamental fields decay up to ``z`` (``a``: half the sum of the two waves' power attenuation coefficients per unit path)
and the harmonic generated at ``z`` decays from there to the exit (``b``: half its coefficient), so ``P_2`` is the harmonic
power at the exit face; with one coefficient ``alpha`` for every wave and no walk-off, focusing or mismatch it is
``P_2(alpha = 0) exp(-alpha L) ((1 - exp(-x)) / x)^2``, ``x = alpha L / 2``. ``focused`` evaluates ``tau_j`` along the crystal (the Boyd-Kleinman focusing
factor); otherwise ``tau_j`` is frozen at the crystal's middle (the near-field, thin-crystal limit: the beam radius does not
change along the crystal). In the near field without walk-off ``H = (w0x w0y / (wx wy)) sinc^2(dk L / 2)`` and the formula is
the plane-wave one integrated over the Gaussian intensity, ``P_2 = K P^2 L^2 sinc^2 / (pi wx wy)``; for a round beam focused
at the crystal's centre without walk-off ``H = (b / L) h(sigma, xi)`` in Boyd and Kleinman's notation (``xi = L / b``,
``sigma = -b dk / 2``), whose maximum ``h = 1.068`` lies at ``xi = 2.84``.

Every operation is ``+ - * /`` and ``sqrt`` or ``math.exact_trig`` (no libm transcendental), complex products are formed from
separately rounded real ones, matrix products with ``math.vectors`` and sums with ``correctly_rounded_sum`` (no fused
multiply-add, no BLAS): the same bits in every runtime.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ...math.exact_trig import cos_sin
from ...math.exact_trig import exp as exact_exp
from ...math.vectors import correctly_rounded_sum, dot, matvec
from ...units import NM_TO_MM, UM_TO_MM
from ..polarization.uniaxial import EXTRAORDINARY, Crystal

Array = NDArray[np.float64]

#: Vacuum permittivity (F/m) and speed of light (m/s), CODATA 2018.
EPS0_F_PER_M = 8.8541878128e-12
C_M_PER_S = 299792458.0
#: pm/V -> m/V, mm -> m, nm -> m.
PM_PER_V_TO_M_PER_V = 1e-12
MM_TO_M = 1e-3
NM_TO_M = 1e-9
#: ``x`` with ``sinc^2(x) = 1/2`` (``sin(x) / x = 1 / sqrt 2``): the half-power point of the plane-wave tuning curve.
SINC2_HALF_POWER_ARGUMENT = 1.3915573782515103
#: Full width at half maximum of ``sinc^2(dk L / 2)`` in ``dk L`` (rad).
FWHM_DK_L = 4.0 * SINC2_HALF_POWER_ARGUMENT
#: The first-order relative overstatement of the undepleted harmonic of one wave in a collimated Gaussian beam at phase matching,
#: per unit of conversion efficiency: ``(2/3) K^2 int I^3 / int K I^2 / eta = 8/9`` (``depletion_estimate``).
GAUSSIAN_DEPLETION_FACTOR = 8.0 / 9.0
#: Gauss-Legendre nodes per panel of the composite rule of ``overlap_factor``.
PANEL_NODES = 8
#: ``2 sqrt(2 ln 2)``: the full width at half maximum of a Gaussian in units of its standard deviation.
GAUSSIAN_FWHM_PER_SIGMA = 2.3548200450309493
#: ``2 sqrt(ln 2) / pi``: a Gaussian line of wavelength FWHM ``d_lambda`` has the coherence length
#: ``L_c = (2 sqrt(ln 2) / pi) lambda^2 / d_lambda`` (``scene/lower.py``, ``coherence_length_from_linewidth_mm``).
COHERENCE_PER_LINEWIDTH = 0.5300207270387938
#: ``line_average``: panels of ``PANEL_NODES`` Gauss-Legendre nodes over ``+- LINE_SIGMAS`` standard deviations of the line (a
#: line no wider than the spectral acceptance: the phase changes by about 1 rad per panel at most).
LINE_PANELS = 16
LINE_SIGMAS = 5.0
#: The most nodes ``overlap_factor`` uses over the crystal: one dimension (no walk-off), two dimensions (walk-off).
MAX_NODES_1D = 65536
MAX_NODES_2D = 2048


@dataclass(frozen=True)
class Overlap:
    """``H`` of the module docstring, the difference between its values on the last two node counts (an estimate of the
    quadrature error) and the node count."""

    value: float
    quadrature_error: float
    nodes: int


def wave_index(crystal: Crystal, mode: str, k_hat: Array) -> float:
    """The phase index of the ``mode`` wave of ``crystal`` along the unit wave normal ``k_hat``: ``n_o``, or ``n(theta)`` with
    ``1 / n^2 = cos^2 theta / n_o^2 + sin^2 theta / n_e^2``."""

    if mode != EXTRAORDINARY:
        return crystal.n_o
    c = float(dot(k_hat, crystal.axis))
    c2 = min(c * c, 1.0)
    return 1.0 / math.sqrt(c2 * crystal.a + (1.0 - c2) * crystal.b)


def grating_wavenumber_per_mm(period_um: float | None, order: int) -> float:
    """``2 pi m / Lambda`` (1/mm) of a quasi-phase-matching grating of period ``period_um``; 0 without one."""

    if period_um is None:
        return 0.0
    return 2.0 * math.pi * order / (period_um * UM_TO_MM)


def collinear_mismatch(k_hat: Array, fundamental: Crystal, harmonic: Crystal, modes: tuple[str, str], harmonic_mode: str,
                       grating_per_mm: float) -> float:
    """``dk = k_2 - k_a - k_b - K_G`` (1/mm) of collinear waves along ``k_hat`` (``fundamental`` at the fundamental's vacuum
    wavelength, ``harmonic`` at half of it)."""

    k0 = 2.0 * math.pi / (fundamental.wavelength_nm * NM_TO_MM)
    n_a = wave_index(fundamental, modes[0], k_hat)
    n_b = wave_index(fundamental, modes[1], k_hat)
    n_h = wave_index(harmonic, harmonic_mode, k_hat)
    return k0 * (2.0 * n_h - n_a - n_b) - grating_per_mm


def d_eff_from_tensor(d_pm_per_v: tuple[tuple[float, ...], ...], frame: Array, e_h: Array, e_a: Array, e_b: Array) -> float:
    """``d_eff = sum_ijk d_ijk e_h,i e_a,j e_b,k`` (pm/V) from the contracted 3 x 6 matrix ``d_il`` (``l``: xx, yy, zz, yz, xz,
    xy) in the crystal frame, whose unit axes X, Y, Z are the rows of ``frame`` (scene frame); ``e_*`` the unit electric-field
    vectors (scene frame) of the harmonic and the two fundamental waves."""

    h, a, b = (matvec(frame, np.asarray(v, dtype=float)).tolist() for v in (e_h, e_a, e_b))
    pairs = (a[0] * b[0], a[1] * b[1], a[2] * b[2], a[1] * b[2] + a[2] * b[1], a[0] * b[2] + a[2] * b[0], a[0] * b[1] + a[1] * b[0])
    total = 0.0
    for i in range(3):
        row = d_pm_per_v[i]
        total += h[i] * (row[0] * pairs[0] + row[1] * pairs[1] + row[2] * pairs[2] + row[3] * pairs[3] + row[4] * pairs[4]
                         + row[5] * pairs[5])
    return total


def coupling_per_W(fundamental_nm: float, d_eff_pm_per_v: float, n_a: float, n_b: float, n_h: float) -> float:
    """``K = 2 omega^2 d_eff^2 / (n_a n_b n_2 eps0 c^3)`` (1/W), ``omega`` the fundamental's angular frequency."""

    omega = 2.0 * math.pi * C_M_PER_S / (fundamental_nm * NM_TO_M)
    d = d_eff_pm_per_v * PM_PER_V_TO_M_PER_V
    return 2.0 * omega * omega * d * d / (n_a * n_b * n_h * EPS0_F_PER_M * C_M_PER_S * C_M_PER_S * C_M_PER_S)


def harmonic_power_W(coupling: float, power_a_W: float, power_b_W: float, two_waves: bool, length_mm: float, waist_x_mm: float,
                     waist_y_mm: float, overlap: float) -> float:
    """``P_2 = K f P_a P_b L^2 H / (pi w0x w0y)`` (module docstring; ``f`` 4 for two fundamental waves, else 1 with
    ``P_a = P_b``)."""

    factor = 4.0 if two_waves else 1.0
    return coupling * factor * power_a_W * power_b_W * length_mm * length_mm * overlap / (math.pi * waist_x_mm * waist_y_mm)


def depletion_estimate(efficiency: float, power_a_W: float, power_b_W: float, two_waves: bool) -> float:
    """The relative amount by which the undepleted-pump harmonic power exceeds the depleted one, to first order: ``8/9 eta``
    for one fundamental wave in a collimated Gaussian beam at phase matching (``tanh^2 x = x^2 (1 - 2/3 x^2 + ...)`` per
    point of the beam, integrated over its intensity), times ``(P_a + P_b)^2 / (4 P_a P_b)`` for two waves (the weaker one
    depletes first). ``efficiency``: harmonic power / power of the converting fundamental waves."""

    factor = 1.0
    if two_waves and power_a_W > 0.0 and power_b_W > 0.0:
        total = power_a_W + power_b_W
        factor = total * total / (4.0 * power_a_W * power_b_W)
    return GAUSSIAN_DEPLETION_FACTOR * efficiency * factor


def _inv_sqrt_one_plus_i(tau: Array) -> tuple[Array, Array]:
    """``(1 + i tau)^(-1/2)`` on the principal branch (real and imaginary parts), from real square roots."""

    size = np.sqrt(1.0 + tau * tau)
    re = np.sqrt((size + 1.0) / 2.0)
    im = np.sign(tau) * np.sqrt(np.maximum(size - 1.0, 0.0) / 2.0)
    return re / size, -im / size


def _times(ar: Array, ai: Array, br: Array, bi: Array) -> tuple[Array, Array]:
    """``(ar + i ai)(br + i bi)`` from separately rounded real products (no fused multiply-add)."""

    return ar * br - ai * bi, ar * bi + ai * br


def _nodes(length_mm: float, panels: int, rule: tuple[Array, Array]) -> tuple[Array, Array]:
    x, w = rule
    width = length_mm / panels
    starts = np.arange(panels) * width
    z = (starts[:, None] + (x[None, :] + 1.0) * (width / 2.0)).ravel()
    weights = np.tile(w * (width / 2.0), panels)
    return z, weights


def _overlap_on(length_mm: float, dk: float, rayleigh: tuple[float, float], offset: tuple[float, float],
                waist: tuple[float, float], drift: tuple[float, float], pair: tuple[float, float], focused: bool, panels: int,
                rule: tuple[Array, Array], absorption: tuple[float, float] = (0.0, 0.0)) -> float:
    z, w = _nodes(length_mm, panels, rule)
    where = z if focused else np.full_like(z, length_mm / 2.0)
    # F(z) w in real arithmetic: the phase exp(-i dk z), the focusing factors, G(z) of two walking-off waves, the absorption
    fr, fi = cos_sin(-dk * z)
    er = np.zeros_like(z)
    ei = np.zeros_like(z)
    paired = False
    for j in range(2):
        tau = (where + offset[j]) / rayleigh[j]
        fr, fi = _times(fr, fi, *_inv_sqrt_one_plus_i(tau))
        if pair[j] != 0.0:  # - c / (1 + i tau) = -c (1 - i tau) / (1 + tau^2)
            paired = True
            c = (pair[j] * pair[j] * z * z) / (2.0 * waist[j] * waist[j]) / (1.0 + tau * tau)
            er = er - c
            ei = ei + c * tau
    if paired:
        cos_e, sin_e = cos_sin(ei)
        m = exact_exp(er)
        fr, fi = _times(fr, fi, m * cos_e, m * sin_e)
    if absorption[0] != 0.0 or absorption[1] != 0.0:
        decay = exact_exp(-(absorption[0] * z + absorption[1] * (length_mm - z)))
        fr, fi = fr * decay, fi * decay
    fr, fi = fr * w, fi * w
    if drift[0] == 0.0 and drift[1] == 0.0:
        tr, ti = correctly_rounded_sum(fr.tolist()), correctly_rounded_sum(fi.tolist())
        return (tr * tr + ti * ti) / (length_mm * length_mm)
    gap = z[:, None] - z[None, :]
    rate = drift[0] * drift[0] / (waist[0] * waist[0]) + drift[1] * drift[1] / (waist[1] * waist[1])
    kernel = exact_exp(-rate * gap * gap)
    # Re F(u) F*(v) = Fr(u) Fr(v) + Fi(u) Fi(v)
    terms = (fr[:, None] * fr[None, :] + fi[:, None] * fi[None, :]) * kernel
    return correctly_rounded_sum(terms.ravel().tolist()) / (length_mm * length_mm)


def overlap_factor(length_mm: float, dk_per_mm: float, rayleigh_mm: tuple[float, float], waist_offset_mm: tuple[float, float],
                   waist_mm: tuple[float, float], drift: tuple[float, float], pair: tuple[float, float], focused: bool,
                   rule: tuple[Array, Array], rel_tol: float, absorption: tuple[float, float] = (0.0, 0.0)) -> Overlap:
    """``H`` of the module docstring over a crystal of path length ``length_mm`` with the phase mismatch ``dk_per_mm`` along
    it; per beam axis: Rayleigh range, ``waist_offset_mm`` (how far past its waist the beam is where it enters: ``tau(z) =
    (z + offset) / z_R``), waist radius, harmonic walk-off ``drift`` and type-II walk-off ``pair`` (per unit path);
    ``absorption``: the field attenuations ``(a, b)`` per unit path of the module docstring (``(0, 0)``: a transparent
    crystal). Composite Gauss-Legendre (``rule``: ``PANEL_NODES`` nodes on [-1, 1]) with panels enough to resolve the phase,
    the focusing, the walk-off and the absorption, doubled until two successive values agree to ``rel_tol`` of the larger
    (or the node limit is reached)."""

    variation = abs(dk_per_mm) * length_mm / math.pi
    if focused:
        variation += 4.0 * length_mm / min(rayleigh_mm)
    variation += 2.0 * max(abs(d) / w for d, w in zip(drift, waist_mm)) * length_mm
    variation += max(abs(p) / w for p, w in zip(pair, waist_mm)) * length_mm
    variation += (absorption[0] + absorption[1]) * length_mm
    panels = 2 + int(math.ceil(variation))
    limit = (MAX_NODES_1D if drift == (0.0, 0.0) else MAX_NODES_2D) // PANEL_NODES
    panels = min(panels, limit // 2) if limit >= 2 else 1
    args = (length_mm, dk_per_mm, rayleigh_mm, waist_offset_mm, waist_mm, drift, pair, focused)
    previous = _overlap_on(*args, panels, rule, absorption)
    while True:
        finer = min(2 * panels, limit)
        value = _overlap_on(*args, finer, rule, absorption)
        error = abs(value - previous)
        if error <= rel_tol * max(abs(value), abs(previous)) or finer == panels or finer >= limit:
            return Overlap(value, error, finer * PANEL_NODES)
        panels, previous = finer, value


def plane_factor(phase: float) -> float:
    """``sinc^2(phase)``, ``phase = dk L / 2`` (1 at 0)."""

    if phase == 0.0:
        return 1.0
    return (float(cos_sin(np.array([phase]))[1][0]) / phase) ** 2


def line_average(phase_at: Callable[[float], float | None], fwhm_nm: float, rule: tuple[Array, Array]) -> float | None:
    """The plane-wave tuning factor ``sinc^2(dk L / 2)`` averaged over a Gaussian spectral line of wavelength FWHM ``fwhm_nm``:
    ``phase_at(offset_nm)`` gives ``dk L / 2`` at the fundamental's vacuum wavelength offset from the line centre (``None``:
    not available there, and then neither is the average). Composite Gauss-Legendre (``rule``) over ``+- LINE_SIGMAS``
    standard deviations in ``LINE_PANELS`` panels, normalized by the weight it covers."""

    sigma = fwhm_nm / GAUSSIAN_FWHM_PER_SIGMA
    offsets, weights = _nodes(2.0 * LINE_SIGMAS * sigma, LINE_PANELS, rule)
    offsets = offsets - LINE_SIGMAS * sigma
    gauss = weights * exact_exp(-(offsets * offsets) / (2.0 * sigma * sigma))
    total = 0.0
    norm = 0.0
    for x, g in zip(offsets.tolist(), gauss.tolist()):
        phase = phase_at(x)
        if phase is None:
            return None
        total += g * plane_factor(phase)
        norm += g
    return total / norm


__all__ = [
    "COHERENCE_PER_LINEWIDTH",
    "C_M_PER_S",
    "EPS0_F_PER_M",
    "FWHM_DK_L",
    "GAUSSIAN_FWHM_PER_SIGMA",
    "LINE_PANELS",
    "LINE_SIGMAS",
    "MAX_NODES_1D",
    "MAX_NODES_2D",
    "Overlap",
    "PANEL_NODES",
    "SINC2_HALF_POWER_ARGUMENT",
    "collinear_mismatch",
    "coupling_per_W",
    "d_eff_from_tensor",
    "depletion_estimate",
    "grating_wavenumber_per_mm",
    "harmonic_power_W",
    "line_average",
    "overlap_factor",
    "plane_factor",
    "wave_index",
]
