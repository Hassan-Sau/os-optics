"""Second-harmonic generation formulas (``optics/nonlinear/shg.py``) against references computed here independently:

* the coupled-wave equations of second-harmonic generation integrated numerically (fourth-order Runge-Kutta, SI units,
  ``E = A exp(i(kz - wt)) + c.c.``, ``I = 2 n eps0 c |A|^2``, ``P(2w) = 2 eps0 d A1^2``): undepleted and with pump depletion, for a
  plane wave and point by point over a collimated Gaussian beam;
* the tanh^2 solution of the depleted plane wave at phase matching and the sinc^2 tuning curve of the undepleted one;
* a split-step Fourier solution of the paraxial coupled-wave equation of the harmonic driven by a focused Gaussian fundamental
  (with and without walk-off): an independent numerical calculation of the Boyd-Kleinman focusing factor;
* the closed-form near-field walk-off factor ``(sqrt(pi) t erf(t) - 1 + exp(-t^2)) / t^2`` (``t = rho L / w0``) and Boyd and
  Kleinman's published optimum ``h = 1.068`` at ``xi = 2.84``, ``sigma = 0.57`` (J. Appl. Phys. 39, 3597 (1968)).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from os_optics.optics.gaussian import gauss_legendre_rule
from os_optics.optics.nonlinear import shg

EPS0 = 8.8541878128e-12
C = 299792458.0
RULE = gauss_legendre_rule(shg.PANEL_NODES)
LAMBDA_NM = 1064.0
N1, N2 = 1.6545, 1.6545  # (phase-matched indices; the formulas take any)
D_PM = 2.0


def _rk4(f, y0: np.ndarray, length: float, steps: int) -> np.ndarray:
    y, h = y0.astype(complex), length / steps
    for k in range(steps):
        z = k * h
        k1 = f(z, y)
        k2 = f(z + h / 2, y + h / 2 * k1)
        k3 = f(z + h / 2, y + h / 2 * k2)
        k4 = f(z + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return y


def _coupled(intensity_W_m2: float, length_m: float, dk_per_m: float, deplete: bool, steps: int = 400) -> float:
    """Harmonic intensity (W/m^2) after ``length_m`` of the SHG coupled-wave equations (Boyd, Nonlinear Optics, 2.7):
    dA1/dz = i 2 w d / (n1 c) A2 A1* exp(-i dk z), dA2/dz = i 2 w d / (n2 c) A1^2 exp(i dk z) (dk = k2 - 2 k1, the second
    equation's phase written for this sign)."""
    omega = 2 * math.pi * C / (LAMBDA_NM * 1e-9)
    d = D_PM * 1e-12
    a1 = math.sqrt(intensity_W_m2 / (2 * N1 * EPS0 * C))
    g1, g2 = 2 * omega * d / (N1 * C), 2 * omega * d / (N2 * C)

    def f(z, y):
        A1, A2 = y
        dA1 = 1j * g1 * A2 * np.conj(A1) * np.exp(1j * dk_per_m * z) if deplete else 0.0
        dA2 = 1j * g2 * A1 * A1 * np.exp(-1j * dk_per_m * z)
        return np.array([dA1, dA2])

    out = _rk4(f, np.array([a1, 0.0]), length_m, steps)
    return 2 * N2 * EPS0 * C * abs(out[1]) ** 2


def _plane_wave(intensity: float, length_m: float, dk: float) -> float:
    """The plane-wave undepleted result through the module: P_2 / area for a flat beam = K I^2 L^2 sinc^2."""
    k = shg.coupling_per_W(LAMBDA_NM, D_PM, N1, N1, N2)
    phase = dk * length_m / 2
    return k * intensity * intensity * length_m * length_m * (1.0 if phase == 0 else (math.sin(phase) / phase) ** 2)


@pytest.mark.parametrize("phase", [0.0, 0.7, 1.39155737825, 3.0, 10.0])
def test_the_plane_wave_coupling_is_the_undepleted_coupled_wave_solution(phase):
    """K I^2 L^2 sinc^2(dk L / 2) equals the numerical undepleted coupled-wave solution (relative 1e-9) for phase mismatches
    from 0 to 10 rad (the sinc^2 tuning curve, through its half-power point)."""
    length = 5e-3
    intensity = 1e10
    dk = 2 * phase / length
    assert _plane_wave(intensity, length, dk) == pytest.approx(_coupled(intensity, length, dk, deplete=False, steps=4000), rel=1e-9)


@pytest.mark.parametrize("intensity", [1e12, 1e13, 1e14])
def test_the_depleted_plane_wave_is_tanh_squared_and_the_estimate_bounds_the_overstatement(intensity):
    """With pump depletion at phase matching the harmonic is I tanh^2(Gamma L) (numerically integrated; relative 1e-8) and the
    undepleted value overstates it by (2/3) eta to first order: the plane-wave reference of the depletion estimate."""
    length = 5e-3
    undepleted = _plane_wave(intensity, length, 0.0)
    gamma_l = math.sqrt(undepleted / intensity)
    depleted = _coupled(intensity, length, 0.0, deplete=True, steps=2000)
    assert depleted == pytest.approx(intensity * math.tanh(gamma_l) ** 2, rel=1e-8)
    eta = undepleted / intensity
    over = undepleted / depleted - 1.0
    assert over == pytest.approx(2.0 / 3.0 * eta, rel=0.15 + 2 * eta)


def test_a_collimated_gaussian_beam_integrates_the_plane_wave_over_its_profile_and_the_depletion_estimate_holds():
    """Near field, no walk-off, at phase matching: P2 = K P^2 L^2 / (pi w^2) is the undepleted coupled-wave solution integrated
    point by point over the Gaussian intensity (radial quadrature, relative 1e-6); with depletion the overstatement is
    8/9 eta to first order and at most the module's estimate (which is that value)."""
    length_mm, w_mm = 5.0, 0.02
    for power in (0.5, 50.0):
        h = shg.overlap_factor(length_mm, 0.0, (1e12, 1e12), (0.0, 0.0), (w_mm, w_mm), (0.0, 0.0), (0.0, 0.0), False, RULE, 1e-12)
        k = shg.coupling_per_W(LAMBDA_NM, D_PM, N1, N1, N2)
        p2 = shg.harmonic_power_W(k, power, power, False, length_mm, w_mm, w_mm, h.value)
        w = w_mm * 1e-3
        i0 = 2 * power / (math.pi * w * w)
        r = np.linspace(0.0, 3.5 * w, 1401)
        und = np.array([_coupled(i0 * math.exp(-2 * x * x / (w * w)), length_mm * 1e-3, 0.0, False, steps=8) for x in r])
        p_und = np.trapezoid(und * 2 * np.pi * r, r)
        assert p2 == pytest.approx(p_und, rel=1e-5)
        dep = np.array([_coupled(i0 * math.exp(-2 * x * x / (w * w)), length_mm * 1e-3, 0.0, True, steps=200) for x in r])
        p_dep = np.trapezoid(dep * 2 * np.pi * r, r)
        eta = p2 / power
        estimate = shg.depletion_estimate(eta, power, power, False)
        over = p_und / p_dep - 1.0  # (both on the same radial grid)
        assert over <= estimate * (1.0 + 2.0 * eta) and over == pytest.approx(estimate, rel=2.0 * eta + 1e-3)


def test_two_waves_share_the_power_and_deplete_as_the_type_ii_equations_say():
    """Type II: P2 = K 4 P_a P_b L^2 H / (pi w^2): at 45 deg (P_a = P_b = P/2) the type I value of the total power, and the
    depletion factor (P_a + P_b)^2 / (4 P_a P_b) is 1 there and grows for unequal powers."""
    k = shg.coupling_per_W(LAMBDA_NM, D_PM, N1, N1, N2)
    one = shg.harmonic_power_W(k, 1.0, 1.0, False, 5.0, 0.05, 0.05, 1.0)
    two = shg.harmonic_power_W(k, 0.5, 0.5, True, 5.0, 0.05, 0.05, 1.0)
    assert two == pytest.approx(one, rel=1e-15)
    assert shg.depletion_estimate(0.01, 0.5, 0.5, True) == pytest.approx(8.0 / 9.0 * 0.01, rel=1e-15)
    assert shg.depletion_estimate(0.01, 0.9, 0.1, True) == pytest.approx(8.0 / 9.0 * 0.01 / 0.36, rel=1e-12)


def test_the_near_field_walkoff_factor_is_the_closed_form():
    """Near field at phase matching with walk-off rho: H = (sqrt(pi) t erf(t) - 1 + exp(-t^2)) / t^2, t = rho L / w0."""
    length, w0 = 5.0, 0.05
    for t in (0.1, 1.0, 5.56, 20.0):
        rho = t * w0 / length
        h = shg.overlap_factor(length, 0.0, (1e12, 1e12), (0.0, 0.0), (w0, w0), (rho, 0.0), (0.0, 0.0), False, RULE, 1e-10)
        closed = (math.sqrt(math.pi) * t * math.erf(t) - 1.0 + math.exp(-t * t)) / (t * t)
        assert h.value == pytest.approx(closed, rel=1e-8)


def test_the_boyd_kleinman_optimum():
    """A round beam focused at the crystal's centre without walk-off: h = xi H peaks at h = 1.068 (sigma = 0.57, xi = 2.84),
    the published Boyd-Kleinman optimum; the optimum lies at dk = k2 - 2 k1 < 0 (sigma = -b dk / 2 > 0)."""
    length, xi = 10.0, 2.84
    b = length / xi
    best, at = 0.0, 0.0
    for sigma in np.linspace(0.40, 0.75, 71):
        h = xi * shg.overlap_factor(length, -2 * sigma / b, (b / 2, b / 2), (-length / 2, -length / 2), (0.03, 0.03), (0.0, 0.0),
                                    (0.0, 0.0), True, RULE, 1e-10).value
        if h > best:
            best, at = h, sigma
    assert best == pytest.approx(1.068, abs=6e-4) and at == pytest.approx(0.57, abs=0.01)
    wrong = xi * shg.overlap_factor(length, 2 * 0.57 / b, (b / 2, b / 2), (-length / 2, -length / 2), (0.03, 0.03), (0.0, 0.0),
                                    (0.0, 0.0), True, RULE, 1e-10).value
    assert wrong < 0.1


def _split_step(length_mm: float, w0_mm: float, focus_mm: float, dk_per_mm: float, rho: float, n: int = 256, steps: int = 800) -> float:
    """H from a split-step Fourier solution of dA2/dz = (i / 2 k2) lap A2 - rho dA2/dx + A1^2 exp(-i dk z) with the analytic
    focused TEM00 fundamental A1 = exp(-r^2 / (w0^2 (1 + i tau))) / (1 + i tau), tau = (z - focus) / zR (k2 = 2 k1, unit
    coupling); H = int |A2(L)|^2 / (L^2 pi w0^2 / 4) is 1 for a collimated beam at phase matching."""
    lam_mm = LAMBDA_NM * 1e-6
    k1 = 2 * math.pi * N1 / lam_mm
    k2 = 2 * k1
    zr = k1 * w0_mm * w0_mm / 2
    half = 6.0 * w0_mm * math.sqrt(1 + (max(abs(focus_mm), abs(length_mm - focus_mm)) / zr) ** 2) + rho * length_mm
    x = np.linspace(-half, half, n, endpoint=False)
    dx = x[1] - x[0]
    X, Y = np.meshgrid(x, x, indexing="ij")
    kx = 2 * math.pi * np.fft.fftfreq(n, dx)
    KX, KY = np.meshgrid(kx, kx, indexing="ij")
    dz = length_mm / steps
    half_step = np.exp((-1j * (KX ** 2 + KY ** 2) / (2 * k2) - 1j * KX * rho) * dz / 2)
    a2 = np.zeros((n, n), dtype=complex)
    for s in range(steps):
        z = (s + 0.5) * dz
        a2 = np.fft.ifft2(np.fft.fft2(a2) * half_step)
        tau = (z - focus_mm) / zr
        a1 = np.exp(-(X ** 2 + Y ** 2) / (w0_mm ** 2 * (1 + 1j * tau))) / (1 + 1j * tau)
        a2 = a2 + dz * a1 * a1 * np.exp(-1j * dk_per_mm * z)
        a2 = np.fft.ifft2(np.fft.fft2(a2) * half_step)
    total = float(np.sum(np.abs(a2) ** 2) * dx * dx)
    return total / (length_mm ** 2 * math.pi * w0_mm ** 2 / 4)


@pytest.mark.parametrize(("xi", "sigma", "rho_over"), [(1.0, 0.0, 0.0), (2.84, 0.57, 0.0), (0.5, 0.3, 0.0), (1.0, 0.5, 2.0)])
def test_the_focused_overlap_is_the_split_step_coupled_wave_solution(xi, sigma, rho_over):
    """The Boyd-Kleinman overlap H (focusing, phase mismatch and walk-off rho = rho_over w0 / L) equals the split-step Fourier
    solution of the paraxial coupled-wave equation (relative 1e-5; 256 x 256 grid, 800 steps: 5e-7 seen)."""
    length, w0 = 5.0, 0.02
    lam_mm = LAMBDA_NM * 1e-6
    zr = math.pi * N1 * w0 * w0 / lam_mm
    length = xi * 2 * zr
    b = 2 * zr
    dk = -2 * sigma / b
    rho = rho_over * w0 / length
    focus = length / 2
    module = shg.overlap_factor(length, dk, (zr, zr), (-focus, -focus), (w0, w0), (rho, 0.0), (0.0, 0.0), True, RULE, 1e-9).value
    numeric = _split_step(length, w0, focus, dk, rho)
    assert module == pytest.approx(numeric, rel=1e-5)


def test_the_overlap_reports_an_honest_quadrature_error():
    """Far from phase matching (dk L = 800 rad) with focusing and walk-off (a two-dimensional integral at its node limit): the
    stated quadrature error bounds the difference from the same integral on four times the nodes."""
    args = (20.0, 40.0, (3.0, 4.0), (-12.0, -9.0), (0.02, 0.03), (0.001, 0.0005), (0.0, 0.0), True)
    h = shg.overlap_factor(*args, RULE, 1e-9)
    finer = shg._overlap_on(*args, 4 * h.nodes // shg.PANEL_NODES, RULE)
    assert abs(h.value - finer) <= max(h.quadrature_error, 1e-12 * abs(finer)) and h.quadrature_error <= 1e-5 * abs(h.value)
