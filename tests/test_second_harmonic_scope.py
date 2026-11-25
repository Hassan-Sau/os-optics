"""The conditions of the second-harmonic model (``Tracer._second_harmonic``) beyond one continuous-wave pass through a clear
crystal: a pulsed source, a line of stated width, cascaded crystals, an absorbing crystal and a detuning onto a zero of the
tuning curve. Each case was a review finding (a pulsed source graded PASS with the continuous-wave value, a crash on a stated
linewidth, cascaded crystals added incoherently, an absorbing crystal's harmonic generated at its entrance face, a refusal at
the first zero of sinc^2); each test fails on the code before the fix.

References are computed here, independently of the Engine's nonlinear code (helpers of ``test_second_harmonic``: the
Sellmeier formulas of congruent LiNbO3, the plane-wave coupled-wave result integrated over the collimated Gaussian beam,
the Fresnel transmittance):

* a Gaussian line of FWHM ``d_lambda``: ``sinc^2(dk(lambda) L / 2)`` averaged over its spectrum by a dense Riemann sum with
  ``dk`` from the Sellmeier formulas;
* the harmonic of a laser line whose coherence is phase noise: ``|gamma_2| = |gamma_1|^4``, so with the Gaussian-decay model
  ``L_c = (2 sqrt(ln 2) / pi) lambda^2 / d_lambda`` the harmonic's coherence length is half the fundamental's;
* the harmonic generated along an absorbing crystal (one attenuation coefficient ``alpha`` for both waves, phase matched,
  collimated): the exit power ``P_2(0) exp(-alpha L) ((1 - exp(-x)) / x)^2``, ``x = alpha L / 2`` (the field generated at
  ``z`` from the fundamental's ``exp(-alpha z)`` decays by ``exp(-alpha (L - z) / 2)`` to the exit);
* the first zero of the tuning curve, ``dk L / 2 = pi``, which the beam's focusing fills: the overlap integral
  ``|int_0^L exp(-i dk z) prod_j (1 + i (z - z_j) / z_Rj)^(-1/2) dz|^2 / L^2`` by composite Gauss-Legendre quadrature, with the
  extraordinary wave's own diffraction per section (Rayleigh range ``pi n_e w0^2 / lambda`` across the optic axis,
  ``pi n_o^2 w0^2 / (n_e lambda)`` along it, the waists where the paraxial reduced lengths put them);
* a pulsed harmonic's fluence from the pulse's peak power (a Gaussian pulse of the stated FWHM) against the rating, and the
  continuous-wave value the exposure used to grade.
"""

from __future__ import annotations

import math
import re

import numpy as np
import pytest

from os_optics import builders as B
from os_optics.contracts import (ConstantCoating, ConstantIndexMaterial, PowerRating, ProjectSnapshot, PulseTrain, RefractInteraction,
                                      TabulatedCoating, Target, UniaxialMaterial)
from os_optics.contracts.materials import SecondHarmonicGeneration

from helpers import arrivals, issue_codes, run
from os_optics.config import NumericalConfig
from test_second_harmonic import (
    LAMBDA,
    LN_E,
    LN_O,
    conversion,
    fresnel_t,
    gaussian_near_field,
    laser,
    law,
    line,
    matched_period,
    ppln,
    ppln_dk,
    radius_in_middle,
    scene,
    sellmeier,
    slab,
)

D_EFF = 2.0 / math.pi * 25.0
TARGETS = (Target(target_id="eta", metric="shg_conversion_efficiency", device_id="X", minimum=0.0),
           Target(target_id="p532", metric="power_W", observation_id="OUT", wavelength_nm=532.0, minimum=0.0))


def _source(**update):
    """``laser()`` with fields of its source model replaced (validated again)."""
    component = laser()
    source = component.source.model_copy(update=update)
    return type(component).model_validate({**component.model_dump(), "source": source.model_dump()})


def _targets(result) -> dict:
    return {t.target_id: t for t in result.targets}


# ----------------------------------------------------------------------------------------------------- a crystal that is not a slab


def _wedge(material_id: str, length: float, tilt_deg: float = 1.0):
    """``slab`` with its exit face turned by ``tilt_deg`` about z: not two parallel planes."""
    component = slab(material_id, length)
    front, back = component.surfaces
    tilt = math.radians(tilt_deg)
    return component.model_copy(update={"surfaces": (front, back.model_copy(update={"normal": (math.cos(tilt), math.sin(tilt), 0.0)}))})


def test_a_nonlinear_crystal_that_is_not_a_slab_is_traced_as_the_linear_crystal_with_its_powers_out_of_scope():
    """A PPLN wedge (exit face turned 1 deg). The run completes; the pass reports its conversion out of model scope with the
    reason; the fundamental goes where it goes in the same wedge of the same crystal without a nonlinear interaction -- every
    beam on the plane equal bit for bit -- and its power there, the undepleted number, is never a known value: it is out of
    model scope behind the entrance face, and the plane's power with it."""
    length = 2.0
    material = ppln(length, matched_period())
    linear = material.model_copy(update={"nonlinear": None})

    def wedged(crystal, *targets):
        project = scene(crystal, length, laser(), *targets)
        return run(project.model_copy(update={"components": (project.components[0], _wedge(crystal.material_id, length))}))

    result, reference = wedged(material, *TARGETS), wedged(linear)
    assert result.outcome.execution.outcome.value == "COMPLETED" and reference.status.value != "FAIL"
    row = conversion(result)
    assert (row.status.value, row.harmonic_power_W, row.conversion_efficiency) == ("OUT_OF_MODEL_SCOPE", None, None)
    assert "are not two parallel planes apart" in row.reason and row.surface_id == "X/front"
    (issue,) = [i for i in result.issues if i.code.value == "NONLINEAR_CONVERSION_OUT_OF_SCOPE"]
    assert issue.status.value == "OUT_OF_MODEL_SCOPE" and issue.location == "surface:X/front"
    beams, expected = arrivals(result, "OUT"), arrivals(reference, "OUT")
    assert beams and [a.model_dump(exclude={"power_out_of_scope_at"}) for a in beams] == [a.model_dump() for a in expected]
    assert all(a.power_out_of_scope_at == ("X/front",) for a in beams) and not any(a.power_out_of_scope_at for a in expected)
    plane = next(o for o in result.observations if o.observation_id == "OUT")
    assert plane.quantity_state.value == "OUT_OF_MODEL_SCOPE" and plane.power_upper_bound_W is None
    assert _targets(result)["eta"].status.value == "OUT_OF_MODEL_SCOPE" and _targets(result)["eta"].value is None
    assert result.power_budget.conserved and not reference.nonlinear_conversions


# ----------------------------------------------------------------------------------------------------- pulsed sources


def test_a_pulsed_source_is_out_of_model_scope_and_never_graded_with_the_average_power():
    """The same 1 W beam as a 10 ns / 10 Hz pulse train (1e7 W peak): the continuous-wave conversion of its average power
    is not the pulse train's (which follows the peak power and pulse shape). The pass is OUT_OF_MODEL_SCOPE with the reason,
    the efficiency target and the harmonic's power behind the crystal are out of model scope, the ledger still closes. Before
    the fix both targets passed with the continuous-wave value."""
    length = 2.0
    cw = run(scene(ppln(length, matched_period()), length, laser(), *TARGETS))
    assert conversion(cw).status.value == "PASS" and _targets(cw)["eta"].status.value == "PASS"
    pulsed = run(scene(ppln(length, matched_period()), length,
                       _source(pulse=PulseTrain(repetition_rate_hz=10.0, pulse_duration_s=1e-8)), *TARGETS))
    row = conversion(pulsed)
    assert row.status.value == "OUT_OF_MODEL_SCOPE" and "pulsed" in row.reason and row.harmonic_power_W is None
    assert row.conversion_efficiency is None and "NONLINEAR_CONVERSION_OUT_OF_SCOPE" in issue_codes(pulsed)
    by_id = _targets(pulsed)
    assert by_id["eta"].status.value == "OUT_OF_MODEL_SCOPE" and by_id["p532"].status.value == "OUT_OF_MODEL_SCOPE"
    assert all(a.power_out_of_scope_at for a in arrivals(pulsed, "OUT"))
    assert pulsed.power_budget.conserved


def _pulsed_doubler_with_a_rated_window(pulsed: bool) -> ProjectSnapshot:
    """1 W (average) at 1064 nm, 0.5 mm waist -> 2 mm PPLN (AR faces) -> a dichroic sending 1064 nm out of the line -> a window
    (AR faces) rated 1 J/cm2 for 10 ns pulses at 532 nm (or 1 W/cm2 CW), with power_rating_utilization <= 1 on it."""
    ar = ConstantCoating(reflectance_s=0.0, reflectance_p=0.0, transmittance_s=1.0, transmittance_p=1.0)

    def coated(component):
        return component.model_copy(update={"surfaces": tuple(s.model_copy(update={"interaction": RefractInteraction(coating=ar)})
                                                               for s in component.surfaces)})

    one, zero = [[1.0, 1.0]] * 2, [[0.0, 0.0]] * 2
    dichroic = TabulatedCoating(wavelength_nm=(500.0, 600.0, 1000.0, 1100.0), aoi_deg=(0.0, 90.0), reflectance_s=(*zero, *one),
                                reflectance_p=(*zero, *one), transmittance_s=(*one, *zero), transmittance_p=(*one, *zero))
    common = dict(status="DECLARED", basis="test fixture", value_definition="PEAK", wavelength_nm=532.0, beam_diameter_mm=0.7,
                  beam_diameter_definition="1/E2", polarization="LINEAR", polarization_condition="linear")
    rating = (PowerRating(regime="PULSED", value=1.0, unit="J/cm2", pulse_duration_s=1e-8, repetition_rate_hz=10.0, pulse_count=1,
                          test_protocol="S_ON_1", **common) if pulsed else PowerRating(regime="CW", value=1.0, unit="W/cm2", **common))
    window = coated(B.window("RATED", thickness_mm=1.0, clear_radius_mm=10.0, material="G", position=(40.0, 0.0, 0.0)))
    source = _source(waist_radius_mm=0.5, pulse=PulseTrain(repetition_rate_hz=10.0, pulse_duration_s=1e-8, total_pulses=1)) if pulsed \
        else laser(waist=0.5)
    targets = (Target(target_id="util", metric="power_rating_utilization", device_id="RATED", maximum=1.0),)
    return ProjectSnapshot(materials=(ppln(2.0, matched_period()), ConstantIndexMaterial(material_id="G", valid_wavelength_nm=(300.0, 2000.0),
                                                                                         refractive_index=1.5, absorption_coefficient_per_mm=0.0)),  # Synthetic rated window.
                           components=(source, coated(slab("PPLN", 2.0)),
                                       B.beam_splitter("D", position=(15.0, 0.0, 0.0), normal=(-1 / math.sqrt(2), 1 / math.sqrt(2), 0.0),
                                                       coating=dichroic),
                                       window.model_copy(update={"power_ratings": (rating,)})),
                           observations=(B.observation_plane("OUT", position=(60.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0), radius_mm=50.0),),
                           targets=targets)


def test_a_pulsed_harmonic_s_exposure_is_out_of_model_scope_never_the_continuous_wave_value():
    """The harmonic of a pulse train follows the peak power: 0.1 J pulses of 10 ns (peak about 9.4e6 W) convert most of the
    pump in a 2 mm PPLN (the undepleted efficiency is far above 1), so the 532 nm pulse fluence on the window is many J/cm2,
    above its 1 J/cm2 rating -- while the continuous-wave conversion of the average power gives 1.7e-4 J/cm2, which the
    exposure used to grade (power_rating_utilization PASS 1.7e-4). The window's hits behind the out-of-scope crystal are now
    OUT_OF_MODEL_SCOPE (NONLINEAR_CONVERSION_OUT_OF_SCOPE: no fluence, no pulse energy) and the utilization target is
    OUT_OF_MODEL_SCOPE; the continuous-wave doubler is graded as before."""
    n1, n2 = sellmeier(LN_E, LAMBDA), sellmeier(LN_E, LAMBDA / 2)
    energy = 1.0 / 10.0
    peak_power = energy / (1e-8 * math.sqrt(math.pi / (4.0 * math.log(2.0))))  # (a Gaussian pulse of 10 ns FWHM)
    assert gaussian_near_field(peak_power, D_EFF, 2.0, 0.5, n1, n2, 0.0) / peak_power > 1.0  # (the undepleted efficiency)
    cw_energy = gaussian_near_field(1.0, D_EFF, 2.0, 0.5, n1, n2, 0.0) / 10.0
    assert 2.0 * cw_energy / (math.pi * 0.05 ** 2) < 1e-3  # (J/cm2, w 0.5 mm: the continuous-wave basis, far within the rating)
    assert 2.0 * (0.1 * energy) / (math.pi * 0.05 ** 2) > 1.0  # (even a tenth of the pulse energy converted exceeds it)
    cw = run(_pulsed_doubler_with_a_rated_window(False))
    assert _targets(cw)["util"].status.value == "PASS"
    pulsed = run(_pulsed_doubler_with_a_rated_window(True))
    util = _targets(pulsed)["util"]
    assert (util.status.value, util.value) == ("OUT_OF_MODEL_SCOPE", None)
    assert "NONLINEAR_CONVERSION_OUT_OF_SCOPE" in util.issue_codes and "X/front" in util.message
    rated = [surface for surface in pulsed.exposures.surfaces if surface.element_id == "RATED"]
    assert rated and all(surface.peak_fluence_J_per_cm2 is None for surface in rated)
    for hit in (h for surface in rated for h in surface.hits):
        assert (hit.status.value, hit.reason) == ("OUT_OF_MODEL_SCOPE", "NONLINEAR_CONVERSION_OUT_OF_SCOPE")
        assert hit.peak_fluence_J_per_cm2 is None and hit.pulse_energy_J is None and hit.peak_irradiance_W_per_cm2 is None


# ------------------------------------------------------------------------------------------- linewidth and coherence


def _line_spread(linewidth_nm: float, length: float, period: float) -> float:
    """``|<sinc^2> - sinc^2(centre)|`` over a Gaussian line of FWHM ``linewidth_nm`` (Riemann sum over +-6 sigma, 40001 points)."""
    sigma = linewidth_nm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    wavelengths = np.linspace(LAMBDA - 6 * sigma, LAMBDA + 6 * sigma, 40001)
    weights = np.exp(-(wavelengths - LAMBDA) ** 2 / (2 * sigma * sigma))
    phases = np.array([ppln_dk(period, w) for w in wavelengths]) * length / 2.0
    sinc2 = np.where(phases == 0.0, 1.0, np.sin(phases) ** 2 / np.where(phases == 0.0, 1.0, phases) ** 2)
    centre = 1.0 if ppln_dk(period) == 0.0 else (math.sin(ppln_dk(period) * length / 2) / (ppln_dk(period) * length / 2)) ** 2
    return abs(float((weights * sinc2).sum() / weights.sum()) - centre)


def _stated_spread(reason: str) -> float:
    return float(re.search(r"averaged over the line by ([0-9.e+-]+)", reason).group(1))


def test_a_stated_linewidth_or_coherence_length_no_longer_stops_the_run():
    """Before the fix a source stating linewidth_nm or coherence_length_mm crashed the run (IndexError: the generated line had
    no coherence entry). A narrow line converts as the single frequency (the same harmonic power, bit for bit), and the
    harmonic beam's coherence length is half the fundamental's (twice the phase noise of the fundamental field)."""
    length = 2.0
    plain = conversion(run(scene(ppln(length, matched_period()), length, laser())))
    for update, fundamental_mm in (({"linewidth_nm": 0.001}, 2.0 * math.sqrt(math.log(2.0)) / math.pi * LAMBDA ** 2 / 0.001 * 1e-6),
                                   ({"coherence_length_mm": 1000.0}, 1000.0)):
        result = run(scene(ppln(length, matched_period()), length, _source(**update), *TARGETS))
        row = conversion(result)
        assert row.status.value == "PASS" and row.harmonic_power_W == plain.harmonic_power_W
        out = {a.beam.spectral_index: a for a in arrivals(result, "OUT")}
        assert out[0].coherence_length_mm == pytest.approx(fundamental_mm, rel=1e-12)
        assert out[1].coherence_length_mm == pytest.approx(fundamental_mm / 2.0, rel=1e-12)
        assert _targets(result)["p532"].status.value == "PASS"


@pytest.mark.parametrize(("linewidth_nm", "in_scope"), [(0.1, True), (0.4, False), (5.0, False)])
def test_a_line_wider_than_the_tolerance_allows_is_out_of_model_scope(linewidth_nm, in_scope):
    """A 2 mm PPLN doubler (spectral acceptance 1.02 nm) lit by Gaussian lines of 0.1, 0.4 and 5 nm FWHM: the single-frequency
    result holds while sinc^2 averaged over the line stays within nonlinear_linewidth_rel_tol (0.02) of the centre's value
    (independent changes of the average 0.0044, 0.065, 0.798). Before the fix the 5 nm line (average sinc^2 0.20) was graded PASS with the
    single-frequency efficiency."""
    length, period = 2.0, matched_period()
    reference = _line_spread(linewidth_nm, length, period)
    assert (reference <= 0.02) == in_scope
    result = run(scene(ppln(length, period), length, _source(linewidth_nm=linewidth_nm), *TARGETS))
    row = conversion(result)
    if in_scope:
        assert row.status.value == "PASS"
        assert row.harmonic_power_W == conversion(run(scene(ppln(length, period), length, laser()))).harmonic_power_W
    else:
        assert row.status.value == "OUT_OF_MODEL_SCOPE" and "line's width" in row.reason
        assert _stated_spread(row.reason) == pytest.approx(reference, rel=5e-3)  # (the reason states three digits)
        assert _targets(result)["eta"].status.value == "OUT_OF_MODEL_SCOPE"


# --------------------------------------------------------------------------------------------------- cascaded crystals


def _ppln_named(material_id: str, **extra) -> UniaxialMaterial:
    return UniaxialMaterial(material_id=material_id, valid_wavelength_nm=(250.0, 5000.0), ordinary=law(LN_O), extraordinary=law(LN_E),
                            optic_axis=(0.0, 1.0, 0.0), absorption_coefficient_per_mm=0.0,
                            nonlinear=SecondHarmonicGeneration(fundamental_waves="ee", harmonic_wave="e", d_eff_pm_per_V=25.0,
                                                               qpm_period_um=matched_period()), **extra)


def _second_slab(material_id: str, x0: float, length: float):
    first = slab(material_id, length)
    surfaces = tuple(s.model_copy(update={"origin_mm": (s.origin_mm[0] + x0, 0.0, 0.0)}) for s in first.surfaces)
    return first.model_copy(update={"instance_id": "Y", "surfaces": surfaces})


def test_cascaded_crystals_are_out_of_model_scope_in_the_second_one():
    """Two phase-matched PPLN slabs in series: the harmonic of the first travels with the fundamental into the second, where the
    fields add coherently (in phase: four times one slab's harmonic; the run used to add the powers, 1.56 times, and grade the
    second crystal PASS). The first crystal's pass is unchanged, the second crystal's passes (fundamental and harmonic) are out
    of model scope with their reasons, its efficiency target and the 532 nm power behind it too."""
    length = 2.0
    single = conversion(run(scene(ppln(length, matched_period()), length, laser(waist=1.0))))
    targets = (Target(target_id="p532", metric="power_W", observation_id="OUT", wavelength_nm=532.0, minimum=0.0),
               Target(target_id="etaY", metric="shg_conversion_efficiency", device_id="Y", wavelength_nm=LAMBDA, minimum=0.0))
    project = ProjectSnapshot(materials=(_ppln_named("PPLN"), _ppln_named("B")),
                              components=(laser(waist=1.0), slab("PPLN", length), _second_slab("B", length + 10.0, length)),
                              observations=(B.observation_plane("OUT", position=(2 * length + 50.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0),
                                                                radius_mm=100.0),), targets=targets)
    result = run(project)
    rows = {(r.element_id, r.wavelength_nm): r for r in result.nonlinear_conversions}
    first = rows[("X", LAMBDA)]
    assert first.status.value == "PASS" and first.harmonic_power_W == single.harmonic_power_W
    again, harmonic = rows[("Y", LAMBDA)], rows[("Y", LAMBDA / 2)]
    assert again.status.value == "OUT_OF_MODEL_SCOPE" and "already generated a second harmonic at X/front" in again.reason
    assert harmonic.status.value == "OUT_OF_MODEL_SCOPE" and "second harmonic of L#0 generated upstream" in harmonic.reason
    by_id = _targets(result)
    assert by_id["etaY"].status.value == "OUT_OF_MODEL_SCOPE" and by_id["p532"].status.value == "OUT_OF_MODEL_SCOPE"
    assert all("Y/front" in a.power_out_of_scope_at for a in arrivals(result, "OUT"))
    assert result.power_budget.conserved


# ---------------------------------------------------------------------------------------------------- absorbing crystal


@pytest.mark.parametrize("alpha", [0.5, 0.05])
def test_an_absorbing_crystal_generates_its_harmonic_along_its_length(alpha):
    """PPLN with a power attenuation alpha (both waves), 2 mm, phase matched, a collimated 1 mm beam: the harmonic power at the
    exit face is P_2(0) exp(-alpha L) ((1 - exp(-x)) / x)^2 (relative 1e-6; alpha L = 1: 0.228 of the clear crystal's), the run
    states it as harmonic_power_W and the harmonic arrives with it times the exit face's transmittance. Before the fix the
    harmonic was generated at the entrance face and absorbed over the whole crystal (1.615 times the exit power at alpha L = 1).
    The ledgers close: the harmonic line's input (the fundamental's NONLINEAR_CONVERSION) is the exit power over the harmonic's
    transmittance exp(-alpha L)."""
    length, waist = 2.0, 1.0
    material = ppln(length, matched_period()).model_copy(update={"absorption_coefficient_per_mm": alpha})
    result = run(scene(material, length, laser(waist=waist), *TARGETS))
    row = conversion(result)
    n1, n2 = sellmeier(LN_E, LAMBDA), sellmeier(LN_E, LAMBDA / 2)
    p_in = fresnel_t(1.0, n1)
    clear = gaussian_near_field(p_in, D_EFF, length, radius_in_middle(waist, length, n1), n1, n2, 0.0)
    x = alpha * length / 2.0
    expected = clear * math.exp(-alpha * length) * ((1.0 - math.exp(-x)) / x) ** 2
    assert row.status.value == "PASS"
    assert row.harmonic_power_W == pytest.approx(expected, rel=1e-6)
    assert row.conversion_efficiency == pytest.approx(expected / p_in, rel=1e-6)
    out = {a.beam.spectral_index: a for a in arrivals(result, "OUT")}
    assert out[1].beam.power_W == pytest.approx(expected * fresnel_t(n2, 1.0), rel=1e-6)
    booked = line(result, 1).input_power_W
    assert booked == pytest.approx(expected / math.exp(-alpha * length), rel=1e-6)
    assert line(result, 0).sinks["NONLINEAR_CONVERSION"] == booked
    assert out[0].beam.power_W == pytest.approx((p_in - booked) * math.exp(-alpha * length) * fresnel_t(n1, 1.0), rel=1e-9)
    assert line(result, 0).conserved and line(result, 1).conserved
    assert _targets(result)["p532"].value == pytest.approx(out[1].beam.power_W, rel=1e-12)


# ------------------------------------------------------------------------------------------- zeros of the tuning curve


def _focused_overlap(length: float, dk: float, waist: float) -> float:
    """``H`` of the module docstring for a beam whose waist (in air) lies at ``length / (2 n_e)`` behind the entrance face: its
    extraordinary wave diffracts with ``n_e`` across the optic axis and ``n_o^2 / n_e`` along it."""
    n_e, n_o = sellmeier(LN_E, LAMBDA), sellmeier(LN_O, LAMBDA)
    lam = LAMBDA * 1e-6
    sections = ((math.pi * n_e * waist ** 2 / lam, length / 2.0),
                (math.pi * n_o * n_o * waist ** 2 / (n_e * lam), length / 2.0 * (n_o / n_e) ** 2))

    nodes, weights = np.polynomial.legendre.leggauss(40)  # (composite Gauss-Legendre, 64 panels: the integrand is smooth)
    panel = length / 64
    z = (np.arange(64)[:, None] * panel + (nodes[None, :] + 1.0) * panel / 2.0).ravel()
    w = np.tile(weights * panel / 2.0, 64)
    value = np.exp(-1j * dk * z)
    for rayleigh, at in sections:
        value = value / np.sqrt(1.0 + 1j * (z - at) / rayleigh)
    total = complex(np.sum(value * w))
    return abs(total) ** 2 / (length * length)


def test_off_the_first_zero_the_near_field_result_stands():
    """A 10 mm PPLN at dk L / 2 = 4 with a collimated 1 mm beam: the near-field result agrees with the focused one to far below
    the tolerance of itself, and is the closed form (relative 1e-6)."""
    length, waist, phase = 10.0, 1.0, 4.0
    k_matched = 2 * math.pi / (matched_period() * 1e-3)
    n1, n2 = sellmeier(LN_E, LAMBDA), sellmeier(LN_E, LAMBDA / 2)
    p_in = fresnel_t(1.0, n1)
    period = 2 * math.pi / (k_matched - 2 * phase / length) * 1e3
    row = conversion(run(scene(ppln(length, period), length, laser(waist=waist))))
    assert row.status.value == "PASS" and row.near_field_error < 1e-3
    assert row.phase_mismatch_rad == pytest.approx(phase, abs=1e-6)
    expected = gaussian_near_field(p_in, D_EFF, length, radius_in_middle(waist, length, n1), n1, n2, 2 * phase / length)
    assert row.harmonic_power_W == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize(("waist", "passes"), [(1.0, True), (0.05, False)])
def test_at_the_first_zero_the_harmonic_is_the_focusing_s_not_the_near_field_zero(waist, passes):
    """A 10 mm PPLN at dk L / 2 = pi, the waist at its centre. The near-field sinc^2 vanishes there (1e-29 of the input), but
    the beam's focusing fills the zero: 6e-8 of the phase-matched harmonic for a 1 mm waist (L/b 8e-4), 6e-3 for 0.05 mm
    (L/b 0.31; efficiency 4e-5). Both are within nonlinear_focusing_rel_tol of the phase-matched value, so the near-field result
    used to be graded: a target eta <= 1e-9 passed with 1.1e-29 for the 0.05 mm waist. The pass now reports the focused result
    (the near-field one is not within the tolerance of it): the overlap integral of the module docstring (relative 2e-3: the
    Engine's Gaussian state in the crystal is one beam of the two sections' mean focusing, which moves this second-order value by
    6e-4), and the target FAILs for the 0.05 mm waist and PASSes (1e-12) for the 1 mm one."""
    length, phase = 10.0, math.pi
    n1, n2 = sellmeier(LN_E, LAMBDA), sellmeier(LN_E, LAMBDA / 2)
    k_matched = 2 * math.pi / (matched_period() * 1e-3)
    period = 2 * math.pi / (k_matched - 2 * phase / length) * 1e3
    target = Target(target_id="eta", metric="shg_conversion_efficiency", device_id="X", maximum=1e-9)
    result = run(scene(ppln(length, period), length, laser(waist=waist, waist_x=length / (2 * n1)), target))
    row = conversion(result)
    overlap = _focused_overlap(length, 2 * phase / length, waist)
    assert row.status.value == "PASS" and 0.0 < row.near_field_error <= NumericalConfig().nonlinear_focusing_rel_tol
    assert row.overlap_factor == pytest.approx(overlap, rel=2e-3)
    expected = gaussian_near_field(fresnel_t(1.0, n1), D_EFF, length, waist, n1, n2, 0.0) * overlap
    assert row.harmonic_power_W == pytest.approx(expected, rel=2e-3)
    assert expected > 1e12 * gaussian_near_field(fresnel_t(1.0, n1), D_EFF, length, waist, n1, n2, 2 * phase / length)
    assert _targets(result)["eta"].status.value == ("PASS" if passes else "FAIL")


@pytest.mark.parametrize("linewidth_nm", [0.05, 0.1, 0.15])
def test_a_line_centred_on_a_zero_of_the_tuning_curve_is_out_of_model_scope(linewidth_nm):
    """A 2 mm PPLN whose line centre sits on the first zero (dk L / 2 = pi): sinc^2 averaged over a 0.05 / 0.1 / 0.15 nm line is
    3.4e-4 / 1.35e-3 / 3.0e-3 of phase matching while the centre's is about 0 -- the single-frequency value is not the line's.
    The line check compared the difference with the phase-matched value 1 and passed (efficiency 1.3e-30); it now compares it
    with the larger of the line's average and the reported value's scale, the focused value that fills the zero (3.4e-4 of
    phase matching, about the 0.05 nm line's average): the relative difference is about 1 and the pass is out of model scope."""
    length = 2.0
    k_matched = 2 * math.pi / (matched_period() * 1e-3)
    period = 2 * math.pi / (k_matched - 2 * math.pi / length) * 1e3
    average = _line_spread(linewidth_nm, length, period)  # (|<sinc^2> - centre|, the centre about 0)
    assert 1e-4 < average < 1e-2
    result = run(scene(ppln(length, period), length, _source(linewidth_nm=linewidth_nm), *TARGETS))
    row = conversion(result)
    assert row.status.value == "OUT_OF_MODEL_SCOPE" and "line's width" in row.reason
    assert 0.5 < _stated_spread(row.reason) <= 1.0
    assert _targets(result)["eta"].status.value == "OUT_OF_MODEL_SCOPE"


def _focused_overlap_from_face(length: float, dk: float, waist: float) -> float:
    """``_focused_overlap`` for a beam whose waist lies on the entrance face (``laser()``'s default): both sections' waists at
    ``z = 0``."""
    n_e, n_o = sellmeier(LN_E, LAMBDA), sellmeier(LN_O, LAMBDA)
    lam = LAMBDA * 1e-6
    rayleighs = (math.pi * n_e * waist ** 2 / lam, math.pi * n_o * n_o * waist ** 2 / (n_e * lam))
    nodes, weights = np.polynomial.legendre.leggauss(40)
    panel = length / 64
    z = (np.arange(64)[:, None] * panel + (nodes[None, :] + 1.0) * panel / 2.0).ravel()
    w = np.tile(weights * panel / 2.0, 64)
    value = np.exp(-1j * dk * z)
    for rayleigh in rayleighs:
        value = value / np.sqrt(1.0 + 1j * z / rayleigh)
    return abs(complex(np.sum(value * w))) ** 2 / (length * length)


@pytest.mark.parametrize("linewidth_nm", [1e-6, 1e-4, 1e-3, 0.01])
def test_at_a_filled_zero_a_narrow_line_is_judged_against_the_focused_value(linewidth_nm):
    """The 2 mm PPLN on its first zero (dk L / 2 = pi), 0.05 mm waist: the harmonic reported there is the focused value that
    fills the zero (``H(pi) / H(0)`` = 3.4e-4 of phase matching, the independent overlap integral). A line of 1e-6 / 1e-4 /
    1e-3 nm changes sinc^2 averaged over it by 1.4e-13 / 1.4e-9 / 1.4e-7 of phase matching (Riemann average) -- negligible
    against that value -- and used to be refused because the check compared the change with the near-field centre (about 0):
    "changes ... by 1". It is now compared with the reported value's scale: the narrow lines convert as the single frequency
    (the same harmonic power, bit for bit), while 0.01 nm (1.35e-5, 4 % of the filled value) stays out of model scope with
    that share stated."""
    length = 2.0
    k_matched = 2 * math.pi / (matched_period() * 1e-3)
    period = 2 * math.pi / (k_matched - 2 * math.pi / length) * 1e3
    scale = _focused_overlap_from_face(length, 2 * math.pi / length, 0.05) / _focused_overlap_from_face(length, 0.0, 0.05)
    assert 1e-4 < scale < 1e-3
    change = _line_spread(linewidth_nm, length, period) / scale
    single = conversion(run(scene(ppln(length, period), length, laser(), *TARGETS)))
    assert single.status.value == "PASS"
    row = conversion(run(scene(ppln(length, period), length, _source(linewidth_nm=linewidth_nm), *TARGETS)))
    tolerance = NumericalConfig().nonlinear_linewidth_rel_tol
    if change < 0.5 * tolerance:
        assert row.status.value == "PASS" and row.harmonic_power_W == single.harmonic_power_W
    else:
        assert change > 1.5 * tolerance
        assert row.status.value == "OUT_OF_MODEL_SCOPE" and "line's width" in row.reason
        assert _stated_spread(row.reason) == pytest.approx(change, rel=0.05)
