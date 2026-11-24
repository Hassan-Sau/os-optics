"""Second-harmonic generation traced by the Engine (``Tracer._second_harmonic``): a uniaxial crystal slab with a chi(2)
interaction converts part of a Gaussian beam entering it into a harmonic beam at half the wavelength, traced as a new
spectral line of the source, while the fundamental goes on with the power the harmonic took.

References are computed here, independently of the Engine's crystal and nonlinear code:

* the crystals' indices from their published Sellmeier formulas (congruent LiNbO3: Zelmon, Small and Jundt, J. Opt. Soc. Am. B
  14, 3319 (1997), 21 degC; beta-BBO: Tamosauskas et al., Opt. Mater. Express 8, 1410 (2018));
* the extraordinary index n(theta), the type I phase-matching angle ``sin^2 theta = (n_o1^-2 - n_o2^-2) / (n_e2^-2 - n_o2^-2)``, the
  walk-off angle and the phase mismatch ``dk = (4 pi / lambda)(n_2 - n_1) - 2 pi / Lambda``;
* the plane-wave coupled-wave result integrated over the collimated Gaussian beam, ``P_2 = 2 w^2 d^2 L^2 P^2 sinc^2(dk L / 2) /
  (n_1^2 n_2 eps0 c^3 pi w^2)`` (the coupled equations themselves are checked in ``test_shg_reference.py``), and the closed-form
  walk-off factor of the near field;
* acceptance bandwidths as ``5.566 / (L |d dk / dx|)`` from central differences of the test's own ``dk``;
* the isotropic Fresnel transmittance at normal incidence.

Commonly quoted handbook values for type I 1064 -> 532 nm SHG in BBO (phase-matching angle about 22.8 deg, walk-off about
3.2 deg) are checked loosely, as consistency with the literature, not as the reference.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from os_optics import builders as B
from os_optics.config import NumericalConfig
from os_optics.contracts import (
    CircularAperture,
    EngineeringComponentSnapshot,
    JonesVector,
    PlaneSurface,
    ProjectSnapshot,
    RefractInteraction,
    Target,
    UniaxialMaterial,
)
from os_optics.contracts.materials import CrystalTemperature, SecondHarmonicGeneration, SellmeierIndexLaw, ThermoOpticRow

from helpers import arrivals, issue_codes, run

EPS0 = 8.8541878128e-12
C = 299792458.0
LAMBDA = 1064.0
#: congruent LiNbO3 (Zelmon 1997), extraordinary and ordinary; beta-BBO (Tamosauskas 2018)
LN_E = ((2.9804, 0.5981, 8.9543), (0.02047, 0.0666, 416.08))
LN_O = ((2.6734, 1.2290, 12.614), (0.01764, 0.05914, 474.60))
BBO_O = ((0.90291, 0.83155, 0.76536), (0.003926, 0.018786, 60.01))
BBO_E = ((1.151075, 0.21803, 0.656), (0.007142, 0.02259, 263.0))


def sellmeier(law, wavelength_nm: float) -> float:
    l2 = (wavelength_nm / 1000.0) ** 2
    return math.sqrt(1.0 + sum(b * l2 / (l2 - c) for b, c in zip(*law)))


def n_theta(theta: float, n_o: float, n_e: float) -> float:
    return 1.0 / math.sqrt(math.cos(theta) ** 2 / n_o ** 2 + math.sin(theta) ** 2 / n_e ** 2)


def law(coefficients) -> SellmeierIndexLaw:
    return SellmeierIndexLaw(b=coefficients[0], c_um2=coefficients[1])


def fresnel_t(n1: float, n2: float) -> float:
    return 4.0 * n1 * n2 / (n1 + n2) ** 2


def slab(material_id: str, length: float, radius: float = 5.0) -> EngineeringComponentSnapshot:
    def face(sid: str, x: float, normal) -> PlaneSurface:
        return PlaneSurface(surface_id=sid, origin_mm=(x, 0.0, 0.0), normal=normal, aperture=CircularAperture(radius_mm=radius),
                            medium_positive="air", medium_negative=material_id, interaction=RefractInteraction())
    return EngineeringComponentSnapshot(instance_id="X", pose=B.Pose3D(), surfaces=(face("front", 0.0, (-1.0, 0.0, 0.0)),
                                                                                    face("back", length, (1.0, 0.0, 0.0))))


def ppln(length: float, period_um: float, *, focusing: str = "near_field", d33: float = 25.0, temperature=None) -> UniaxialMaterial:
    """Periodically poled congruent LiNbO3, optic axis along y (beam along x: no walk-off), type 0 (e + e -> e)."""
    return UniaxialMaterial(material_id="PPLN", valid_wavelength_nm=(400.0, 5000.0), ordinary=law(LN_O), extraordinary=law(LN_E),
                            optic_axis=(0.0, 1.0, 0.0), temperature=temperature,
                             absorption_coefficient_per_mm=0.0,  # Synthetic crystal: conversion without assumed real bulk loss.
                            nonlinear=SecondHarmonicGeneration(fundamental_waves="ee", harmonic_wave="e", d_eff_pm_per_V=d33,
                                                               qpm_period_um=period_um, focusing=focusing))


def laser(power: float = 1.0, waist: float = 0.05, waist_x: float = 0.0, polarization_deg: float = 0.0, **kw):
    """A beam along +x from x = -20 mm with its waist (in air) at ``waist_x``, linearly polarized at ``polarization_deg`` from y
    toward z (the source's transverse basis): 0 along y, 90 along z."""
    return B.laser("L", wavelength_nm=LAMBDA, waist_radius_mm=waist, position=(-20.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0),
                   waist_position_z_mm=waist_x + 20.0, polarization=JonesVector.linear(polarization_deg), power_W=power, **kw)


def scene(material, length: float, source, *targets) -> ProjectSnapshot:
    return ProjectSnapshot(materials=(material,), components=(source, slab(material.material_id, length)),
                           observations=(B.observation_plane("OUT", position=(length + 40.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0),
                                                             radius_mm=100.0),), targets=tuple(targets))


def matched_period(wavelength: float = LAMBDA) -> float:
    n1, n2 = sellmeier(LN_E, wavelength), sellmeier(LN_E, wavelength / 2)
    return wavelength / 1000.0 / (2.0 * (n2 - n1))


def ppln_dk(period_um: float, wavelength: float = LAMBDA) -> float:
    """dk (1/mm) of e + e -> e along x in congruent LiNbO3 with the grating."""
    n1, n2 = sellmeier(LN_E, wavelength), sellmeier(LN_E, wavelength / 2)
    return 4.0 * math.pi / (wavelength * 1e-6) * (n2 - n1) - 2.0 * math.pi / (period_um * 1e-3)


def gaussian_near_field(power: float, d_eff: float, length_mm: float, w_mm: float, n1: float, n2: float, dk: float) -> float:
    """The plane-wave coupled-wave result integrated over a collimated Gaussian beam of radius ``w_mm`` (module docstring), SI."""
    omega = 2.0 * math.pi * C / (LAMBDA * 1e-9)
    d, length, w = d_eff * 1e-12, length_mm * 1e-3, w_mm * 1e-3
    phase = dk * length_mm / 2.0
    sinc2 = 1.0 if phase == 0 else (math.sin(phase) / phase) ** 2
    return 2 * omega ** 2 * d ** 2 * length ** 2 * power ** 2 * sinc2 / (n1 ** 2 * n2 * EPS0 * C ** 3 * math.pi * w ** 2)


def radius_in_middle(w0_mm: float, length_mm: float, n: float) -> float:
    """The radius in the crystal's middle of a beam whose waist w0 lies on the entrance face (Rayleigh range pi n w0^2 / lambda
    inside the crystal)."""
    zr = math.pi * n * w0_mm * w0_mm / (LAMBDA * 1e-6)
    return w0_mm * math.sqrt(1.0 + (length_mm / 2.0 / zr) ** 2)


def conversion(result):
    (row,) = result.nonlinear_conversions
    return row


def line(result, index: int):
    return next(l for l in result.power_budget.lines if l.spectral_index == index)


# ----------------------------------------------------------------------------------------------- quasi-phase matching


def test_a_phase_matched_ppln_slab_generates_the_collimated_gaussian_harmonic_and_books_the_energy():
    """PPLN at the period that phase-matches 1064 -> 532 nm, a collimated 1 W beam: the harmonic power is the near-field
    coupled-wave value with d_eff = (2/pi) d33 (relative 1e-6), dk L / 2 = 0, the fundamental's line books it as
    NONLINEAR_CONVERSION, the harmonic is spectral line 1 (532 nm, generated from L#0) with that power as its input, and both
    beams leave the slab with the Fresnel transmittance of their own index."""
    length, period = 2.0, matched_period()
    result = run(scene(ppln(length, period), length, laser()))
    row = conversion(result)
    n1, n2 = sellmeier(LN_E, LAMBDA), sellmeier(LN_E, LAMBDA / 2)
    p_in = fresnel_t(1.0, n1)
    expected = gaussian_near_field(p_in, 2.0 / math.pi * 25.0, length, radius_in_middle(0.05, length, n1), n1, n2, 0.0)
    assert row.status.value == "PASS" and row.fundamental_waves == "ee" and row.harmonic_wave == "e"
    assert row.d_eff_pm_per_V == pytest.approx(50.0 / math.pi, rel=1e-15)
    assert abs(row.phase_mismatch_rad) < 1e-9 and row.walkoff_mrad == pytest.approx(0.0, abs=1e-9)
    assert row.fundamental_power_W == pytest.approx(p_in, rel=1e-12)
    # (relative 1e-3: the run carries the extraordinary beam with one diffraction index for its two axes, n_e across the optic
    # axis and n_o^2 / n_e along it; the reference diffracts it with n_e, which moves its radius in the middle by 1e-4)
    assert row.harmonic_power_W == pytest.approx(expected, rel=1e-3)
    assert row.conversion_efficiency == pytest.approx(expected / p_in, rel=1e-3)
    fundamental, harmonic = line(result, 0), line(result, 1)
    assert fundamental.sinks["NONLINEAR_CONVERSION"] == pytest.approx(row.harmonic_power_W, rel=1e-15)
    assert harmonic.wavelength_nm == 532.0 and harmonic.generated_from == "L#0"
    assert harmonic.input_power_W == pytest.approx(row.harmonic_power_W, rel=1e-15)
    assert fundamental.conserved and harmonic.conserved and result.power_budget.conserved
    out = {a.beam.spectral_index: a for a in arrivals(result, "OUT")}
    assert out[1].beam.wavelength_nm == 532.0 and out[1].beam.path_id == row.harmonic_path_id
    assert out[1].beam.power_W == pytest.approx(row.harmonic_power_W * fresnel_t(n2, 1.0), rel=1e-9)
    assert out[0].beam.power_W == pytest.approx((p_in - row.harmonic_power_W) * fresnel_t(n1, 1.0), rel=1e-9)
    # the harmonic is generated as the TEM00 of the fundamental's reduced q-parameter (waist w0 / sqrt 2 at the same place);
    # the two then cross the rest of the crystal with their own indices, which moves their waists in air apart by about
    # L (1 / n_1 - 1 / n_2) (each extraordinary beam diffracts with the run's one index for its two axes: 0.005 mm)
    g1, g2 = out[0].beam.gaussian, out[1].beam.gaussian
    assert g2.waist_radius_x_mm == pytest.approx(g1.waist_radius_x_mm / math.sqrt(2.0), rel=1e-9)
    assert g1.waist_distance_x_mm - g2.waist_distance_x_mm == pytest.approx(length * (1 / n1 - 1 / n2), abs=5e-3)
    assert np.allclose(out[1].beam.direction, (1.0, 0.0, 0.0), atol=1e-12)


@pytest.mark.parametrize("phase", [0.5, 1.3915573782515103, 2.5, 6.0])
def test_detuning_the_grating_follows_the_sinc_squared_tuning_curve(phase):
    """A grating period off the matched one gives dk L / 2 = phase (dk computed here from the Sellmeier formula): the run's
    phase mismatch is that value (1e-6 rad) and its efficiency relative to phase matching is sinc^2(phase) (relative 2e-3: a
    1 mm beam, L / b = 1.6e-4, whose Gouy phase across the crystal shifts the curve by less than that)."""
    length = 2.0
    k_matched = 2 * math.pi / (matched_period() * 1e-3)
    period = 2 * math.pi / (k_matched - 2 * phase / length) * 1e3  # dk = k_matched - K_G
    at = conversion(run(scene(ppln(length, period), length, laser(waist=1.0))))
    peak = conversion(run(scene(ppln(length, matched_period()), length, laser(waist=1.0))))
    assert at.status.value == "PASS" and at.near_field_error < 2e-3
    assert at.phase_mismatch_rad == pytest.approx(ppln_dk(period) * length / 2, abs=1e-6)
    assert at.phase_mismatch_rad == pytest.approx(phase, abs=1e-6)
    assert at.conversion_efficiency / peak.conversion_efficiency == pytest.approx((math.sin(phase) / phase) ** 2, rel=2e-3)
    assert at.plane_wave_factor == pytest.approx((math.sin(phase) / phase) ** 2, rel=1e-12)


def test_the_spectral_and_temperature_acceptance_are_the_widths_of_the_sinc_squared_curve():
    """Spectral acceptance 5.566 / (L |d dk / d lambda|) with the derivative of the test's own dk (relative 1e-4); with stated
    thermo-optic coefficients, the temperature acceptance 5.566 / (L |d dk / dT|) from dn/dT at both wavelengths (relative
    1e-4), and the phase mismatch follows the temperature through the indices."""
    length, period = 3.0, matched_period()
    h = 1e-3
    slope = (ppln_dk(period, LAMBDA + h) - ppln_dk(period, LAMBDA - h)) / (2 * h)
    row = conversion(run(scene(ppln(length, period), length, laser(waist=1.0))))
    assert row.spectral_acceptance_nm == pytest.approx(5.566229513006 / (length * abs(slope)), rel=1e-4)
    assert row.temperature_acceptance_K is None and row.angular_acceptance_mrad is not None
    # stated dn_e/dT of 4e-5 /K at 1064 nm and 9e-5 /K at 532 nm (test values), 10 K above the reference temperature
    rows = (ThermoOpticRow(wavelength_nm=500.0, dno_dT_per_K=0.0, dne_dT_per_K=9.0e-5),
            ThermoOpticRow(wavelength_nm=532.0, dno_dT_per_K=0.0, dne_dT_per_K=9.0e-5),
            ThermoOpticRow(wavelength_nm=1064.0, dno_dT_per_K=0.0, dne_dT_per_K=4.0e-5),
            ThermoOpticRow(wavelength_nm=1100.0, dno_dT_per_K=0.0, dne_dT_per_K=4.0e-5))
    warm = CrystalTemperature(temperature_c=31.0, reference_temperature_c=21.0, thermo_optic=rows)
    row_t = conversion(run(scene(ppln(length, period, temperature=warm), length, laser(waist=1.0))))
    d_dk_dT = 4.0 * math.pi / (LAMBDA * 1e-6) * (9.0e-5 - 4.0e-5)
    assert row_t.temperature_acceptance_K == pytest.approx(5.566229513006 / (length * d_dk_dT), rel=1e-4)
    assert row_t.phase_mismatch_rad == pytest.approx(d_dk_dT * 10.0 * length / 2, rel=1e-6)
    phase = row_t.phase_mismatch_rad
    assert row_t.conversion_efficiency / row.conversion_efficiency == pytest.approx((math.sin(phase) / phase) ** 2, rel=2e-3)


def test_a_tightly_focused_beam_needs_the_focusing_model():
    """A beam focused into the crystal centre at L/b = 2.84 (an ordinary fundamental, o + o -> e under quasi-phase matching:
    the fundamental diffracts with n_o): the near-field model differs from the focused one by more than
    nonlinear_focusing_rel_tol and is out of model scope (the run states why); boyd_kleinman focusing at the optimum mismatch
    sigma = 0.57 gives Boyd and Kleinman's h = 1.068 (P_2 = K P^2 L k_1 h / pi; relative 1e-3)."""
    n1, n2 = sellmeier(LN_O, LAMBDA), sellmeier(LN_E, LAMBDA / 2)
    w0, power = 0.02, 0.05
    zr = math.pi * n1 * w0 * w0 / (LAMBDA * 1e-6)
    length, b = 2.84 * 2 * zr, 2 * zr
    # (the waist stated in air lies inside the crystal at n times its distance from the entrance face: at L / 2 for x = L / (2 n))
    waist_x = length / (2 * n1)
    k0 = 2 * math.pi / (LAMBDA * 1e-6)

    def crystal(dk: float, focusing: str) -> UniaxialMaterial:
        period = 2 * math.pi / (2 * k0 * (n2 - n1) - dk) * 1e3
        return UniaxialMaterial(material_id="PPLN", valid_wavelength_nm=(400.0, 5000.0), ordinary=law(LN_O), extraordinary=law(LN_E),
                                optic_axis=(0.0, 1.0, 0.0), absorption_coefficient_per_mm=0.0,
                                nonlinear=SecondHarmonicGeneration(fundamental_waves="oo", harmonic_wave="e", d_eff_pm_per_V=5.0,
                                                                   qpm_period_um=period, focusing=focusing))

    beam = laser(power=power, waist=w0, waist_x=waist_x, polarization_deg=90.0)  # (along z: ordinary)
    near = conversion(run(scene(crystal(0.0, "near_field"), length, beam)))
    assert near.status.value == "OUT_OF_MODEL_SCOPE" and near.harmonic_power_W is None
    assert near.near_field_error > NumericalConfig().nonlinear_focusing_rel_tol and "boyd_kleinman" in near.reason
    focused = conversion(run(scene(crystal(-2 * 0.57 / b, "boyd_kleinman"), length, beam)))
    assert focused.status.value == "PASS" and focused.focusing_parameter == pytest.approx(2.84, rel=1e-9)
    k1 = k0 * n1
    omega = 2 * math.pi * C / (LAMBDA * 1e-9)
    d = 5.0 * 2.0 / math.pi * 1e-12
    p = focused.fundamental_power_W
    k_si = 2 * omega ** 2 * d ** 2 / (n1 ** 2 * n2 * EPS0 * C ** 3)
    h = focused.harmonic_power_W * math.pi / (k_si * p * p * length * 1e-3 * k1 * 1e3)
    assert h == pytest.approx(1.068, rel=1e-3)


# ----------------------------------------------------------------------------------------------- birefringent: BBO type I


def bbo_type_i(theta: float, length: float, **nonlinear) -> UniaxialMaterial:
    """BBO with its optic axis in the x-y plane at theta from the beam (+x); type I: o + o -> e."""
    spec = {"fundamental_waves": "oo", "harmonic_wave": "e", "d_eff_pm_per_V": 2.0, **nonlinear}
    return UniaxialMaterial(material_id="BBO", valid_wavelength_nm=(400.0, 1100.0), ordinary=law(BBO_O), extraordinary=law(BBO_E),
                            optic_axis=(math.cos(theta), math.sin(theta), 0.0), absorption_coefficient_per_mm=0.0,
                            nonlinear=SecondHarmonicGeneration(**spec))


def bbo_theta() -> float:
    no1, no2, ne2 = sellmeier(BBO_O, LAMBDA), sellmeier(BBO_O, LAMBDA / 2), sellmeier(BBO_E, LAMBDA / 2)
    return math.asin(math.sqrt((no1 ** -2 - no2 ** -2) / (ne2 ** -2 - no2 ** -2)))


def bbo_walkoff(theta: float, wavelength: float) -> float:
    n_o, n_e = sellmeier(BBO_O, wavelength), sellmeier(BBO_E, wavelength)
    s, c = math.sin(theta), math.cos(theta)
    return math.atan((n_o ** 2 - n_e ** 2) * s * c / (n_e ** 2 * c * c + n_o ** 2 * s * s))


def test_bbo_type_i_is_phase_matched_at_its_angle_with_the_walkoff_and_angular_acceptance_of_the_crystal():
    """Type I (o + o -> e) BBO cut at the phase-matching angle (22.83 deg from the Sellmeier formulas; handbooks quote about
    22.8 deg): dk = 0, the walk-off is the extraordinary harmonic's (3.19 deg; handbooks about 3.2 deg), the angular
    acceptance 5.566 / (L |d dk / d theta|) with the test's own derivative, the harmonic power the near-field value with the
    closed-form walk-off factor, and only the ordinary fundamental converts (polarization along z)."""
    theta, length = bbo_theta(), 5.0
    assert math.degrees(theta) == pytest.approx(22.8, abs=0.1)
    rho = bbo_walkoff(theta, LAMBDA / 2)
    assert math.degrees(rho) == pytest.approx(3.2, abs=0.05)
    result = run(scene(bbo_type_i(theta, length), length, laser(polarization_deg=90.0)))  # (o: along z)
    row = conversion(result)
    no1 = sellmeier(BBO_O, LAMBDA)
    assert abs(row.phase_mismatch_rad) < 1e-6 and row.walkoff_mrad == pytest.approx(rho * 1e3, rel=1e-9)
    no2, ne2 = sellmeier(BBO_O, LAMBDA / 2), sellmeier(BBO_E, LAMBDA / 2)

    def dk(t: float) -> float:
        return 4 * math.pi / (LAMBDA * 1e-6) * (n_theta(t, no2, ne2) - no1)

    slope = (dk(theta + 1e-7) - dk(theta - 1e-7)) / 2e-7
    assert row.angular_acceptance_mrad == pytest.approx(5.566229513006 / (length * abs(slope)) * 1e3, rel=1e-4)
    t = math.tan(rho) * length / 0.05
    walk = (math.sqrt(math.pi) * t * math.erf(t) - 1.0 + math.exp(-t * t)) / (t * t)
    p_in = fresnel_t(1.0, no1)
    n_h = n_theta(theta, no2, ne2)
    expected = gaussian_near_field(p_in, 2.0, length, radius_in_middle(0.05, length, no1), no1, n_h, 0.0) * walk
    assert row.harmonic_power_W == pytest.approx(expected, rel=1e-6)
    # the e-polarized fundamental (polarization along y) has no ordinary part: nothing converts
    crossed = conversion(run(scene(bbo_type_i(theta, length), length, laser(polarization_deg=0.0))))
    assert crossed.converting_power_W == 0.0 and crossed.harmonic_power_W == 0.0 and crossed.harmonic_path_id is None


def test_the_d_tensor_of_bbo_gives_its_effective_coefficient_with_the_traced_fields():
    """BBO (point group 3m) with d22 = 2.2 and d31 = 0.08 pm/V (test values), crystal X along z (phi = 90 deg): d_eff is the
    tensor contracted with the traced fields, the harmonic's electric field at theta + rho from the optic axis in the plane of
    the wave normal and the axis (its energy flow at theta + rho): |d22 cos(theta + rho) + d31 sin(theta + rho)|
    (relative 1e-9); the harmonic power scales with d_eff^2."""
    theta, length = bbo_theta(), 1.0
    d22, d31 = 2.2, 0.08
    tensor = ((0.0, 0.0, 0.0, 0.0, d31, -d22), (-d22, d22, 0.0, d31, 0.0, 0.0), (d31, d31, 0.0, 0.0, 0.0, 0.0))
    # crystal X perpendicular to the optic axis and to the plane of k and the axis: along z (then the o field is along -X / +X)
    material = bbo_type_i(theta, length, d_eff_pm_per_V=None, d_tensor_pm_per_V=tensor, crystal_x_axis=(0.0, 0.0, 1.0))
    row = conversion(run(scene(material, length, laser(polarization_deg=90.0))))
    rho = bbo_walkoff(theta, LAMBDA / 2)
    expected = abs(d22 * math.cos(theta + rho) + d31 * math.sin(theta + rho))
    assert abs(row.d_eff_pm_per_V) == pytest.approx(expected, rel=1e-9)
    stated = conversion(run(scene(bbo_type_i(theta, length, d_eff_pm_per_V=expected), length, laser(polarization_deg=90.0))))
    assert row.harmonic_power_W == pytest.approx(stated.harmonic_power_W, rel=1e-9)


def test_type_ii_converts_the_product_of_its_two_waves():
    """BBO type II (o + e -> e) at its phase-matching angle (solved here): the harmonic is K 4 P_o P_e L^2 H / (pi w^2), so it
    follows sin^2(2 alpha) of the input polarization angle alpha from the ordinary axis (relative 1e-9), the two fundamental
    waves give up half the harmonic power each, and the run states the walk-off of the two fundamental waves."""
    no1, ne1 = sellmeier(BBO_O, LAMBDA), sellmeier(BBO_E, LAMBDA)
    no2, ne2 = sellmeier(BBO_O, LAMBDA / 2), sellmeier(BBO_E, LAMBDA / 2)
    lo, hi = 0.3, 1.2
    for _ in range(200):  # n_e2(theta) = (n_o1 + n_e1(theta)) / 2
        mid = (lo + hi) / 2
        if n_theta(mid, no2, ne2) - (no1 + n_theta(mid, no1, ne1)) / 2 > 0:
            lo = mid
        else:
            hi = mid
    theta, length = (lo + hi) / 2, 1.0
    material = UniaxialMaterial(material_id="BBO", valid_wavelength_nm=(400.0, 1100.0), ordinary=law(BBO_O), extraordinary=law(BBO_E),
                                optic_axis=(math.cos(theta), math.sin(theta), 0.0), absorption_coefficient_per_mm=0.0,
                                nonlinear=SecondHarmonicGeneration(fundamental_waves="oe", harmonic_wave="e", d_eff_pm_per_V=2.0))
    rows = {}
    for alpha in (45.0, 30.0, 10.0):
        result = run(scene(material, length, laser(polarization_deg=90.0 - alpha, power=1.0)))
        rows[alpha] = conversion(result)
        assert abs(rows[alpha].phase_mismatch_rad) < 1e-6
    t_o, t_e = fresnel_t(1.0, no1), fresnel_t(1.0, n_theta(theta, no1, ne1))  # (normal incidence: each wave its own index)

    def product(alpha: float) -> float:  # P_o P_e inside the crystal, per W^2 of input
        return math.cos(math.radians(alpha)) ** 2 * t_o * math.sin(math.radians(alpha)) ** 2 * t_e

    for alpha in (30.0, 10.0):
        ratio = rows[alpha].harmonic_power_W / rows[45.0].harmonic_power_W
        assert ratio == pytest.approx(product(alpha) / product(45.0), rel=1e-9)


# ----------------------------------------------------------------------------------------------- validity and targets


def test_strong_conversion_is_out_of_model_scope_with_its_reason_and_the_powers_behind_it_are_too():
    """A conversion whose depletion estimate exceeds nonlinear_depletion_rel_tol: the pass is OUT_OF_MODEL_SCOPE with the
    estimate in its reason, NONLINEAR_CONVERSION_OUT_OF_SCOPE is reported, a target on the efficiency is OUT_OF_MODEL_SCOPE,
    and a power target on either beam behind the crystal is out of model scope; the ledger still closes."""
    length = 10.0
    targets = (Target(target_id="eta", metric="shg_conversion_efficiency", device_id="X", minimum=0.0),
               Target(target_id="p532", metric="power_W", observation_id="OUT", wavelength_nm=532.0, minimum=0.0))
    result = run(scene(ppln(length, matched_period()), length, laser(power=400.0, waist=0.1), *targets))
    row = conversion(result)
    assert row.status.value == "OUT_OF_MODEL_SCOPE" and "undepleted-pump result overstates" in row.reason
    assert row.depletion_estimate > NumericalConfig().nonlinear_depletion_rel_tol and row.harmonic_power_W is None
    assert "NONLINEAR_CONVERSION_OUT_OF_SCOPE" in issue_codes(result)
    by_id = {t.target_id: t for t in result.targets}
    assert by_id["eta"].status.value == "OUT_OF_MODEL_SCOPE" and by_id["p532"].status.value == "OUT_OF_MODEL_SCOPE"
    assert "NONLINEAR_CONVERSION_OUT_OF_SCOPE" in by_id["p532"].issue_codes
    assert result.power_budget.conserved


def test_a_mixture_of_lines_or_a_multimode_beam_is_out_of_model_scope():
    """A two-line source and an M^2 = 1.5 beam: the harmonic is not computed (out of model scope, the reason stated), no
    harmonic line is generated, and the fundamental goes on with its power out of model scope behind the crystal."""
    length = 2.0
    two = B.laser("L", wavelength_nm=None, waist_radius_mm=0.05, position=(-20.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0),
                  polarization=JonesVector.linear(0.0), power_W=1.0, spectrum=((LAMBDA, 0.5), (LAMBDA + 1.0, 0.5)))
    for source, words in ((two, "mixture"), (laser(m2=1.5), "TEM00")):
        result = run(scene(ppln(length, matched_period()), length, source))
        rows = result.nonlinear_conversions
        assert rows and all(r.status.value == "OUT_OF_MODEL_SCOPE" and words in r.reason for r in rows)
        assert all(r.harmonic_path_id is None for r in rows)
        assert all(a.beam.wavelength_nm > 1000.0 and a.power_out_of_scope_at for a in arrivals(result, "OUT"))


def test_targets_on_the_conversion_and_on_the_generated_line():
    """shg_conversion_efficiency / nonlinear_conversion_efficiency / shg_power_W of the crystal read the pass (PASS / FAIL against
    the limits); a power target on the harmonic line at the plane reads the harmonic beam; a target on an element that converts
    nothing selects nothing (UNKNOWN, TARGET_SELECTION_EMPTY)."""
    length = 2.0
    targets = (Target(target_id="eta", metric="shg_conversion_efficiency", device_id="X", minimum=1e-4),
               Target(target_id="eta2", metric="nonlinear_conversion_efficiency", device_id="X", maximum=1e-6),
               Target(target_id="p2", metric="shg_power_W", device_id="X", minimum=0.0),
               Target(target_id="p532", metric="power_W", observation_id="OUT", wavelength_nm=532.0, minimum=0.0),
               Target(target_id="none", metric="shg_conversion_efficiency", device_id="L", minimum=0.0))
    result = run(scene(ppln(length, matched_period()), length, laser(), *targets))
    row = conversion(result)
    by_id = {t.target_id: t for t in result.targets}
    assert by_id["eta"].status.value == "PASS" and by_id["eta"].value == pytest.approx(row.conversion_efficiency, rel=1e-15)
    assert by_id["eta2"].status.value == "FAIL"
    assert by_id["p2"].value == pytest.approx(row.harmonic_power_W, rel=1e-15)
    out = next(a for a in arrivals(result, "OUT") if a.beam.spectral_index == 1)
    assert by_id["p532"].status.value == "PASS" and by_id["p532"].value == pytest.approx(out.beam.power_W, rel=1e-12)
    assert by_id["none"].status.value == "UNKNOWN" and by_id["none"].issue_codes == ("TARGET_SELECTION_EMPTY",)
