"""Light injected into a cavity: what reaches the input mirror, which beam and polarization it brings, the detuning, target units and the
model's edges (``os_optics.local.cavity``, ``injection``).

Each test reproduces a case an independent review found answered wrongly and checks the analysis against a reference written out here:
the Gaussian power through a circular stop ``P (1 - exp(-2 a^2 / w^2))``, the normal-incidence Fresnel loss ``((n - 1) / (n + 1))^2``, the
mode-matching overlap of two Gaussian beams in closed form, a beam carried through the input mirror's substrate by hand-written ABCD
matrices, Malus's law for the share of a linear polarization along a crystal's optic axis, the Airy functions as complex field sums, the
Hermite-Gauss fractions summed exactly in rationals, and the impedance-matching condition of a lossy input mirror.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction

import numpy as np
import pytest

from declared_helpers import NBK7_SELLMEIER, declared_component, flint, nbk7, plano_convex_model, sequence, stop_model, surface, with_models
from local_helpers import AGAINST_X, ALONG_X, plane, project, source
from os_optics.local import json_api
from os_optics.local.advisories import project_warnings
from os_optics.local.catalog import load_project
from os_optics.local.cavity import CavityError, cavity_analysis
from os_optics.local.cavity_build_up import unresolved_fraction
from os_optics.local.lowering import lower_project
from test_cavity_injection import LAMBDA_MM, LAMBDA_NM, PARTIAL, POWER, _airy, _g_mode, cavity
from test_project_crystals import CALCITE, displacer

C_MM_PER_S = 299792458.0e3
PEAK = 0.009 * 0.009 / (1.0 - 0.99) ** 2  # the transmission of a mode-matched beam on resonance (R = 0.99, T = 0.009 mirrors)


def _analyse(doc: dict, injection: dict | None, components=("M1", "M2"), **request):
    body = {"components": list(components), "wavelength_nm": LAMBDA_NM, **request}
    if injection is not None:
        body["injection"] = injection
    return cavity_analysis(doc, body)


def _value(analysis, quantity_id: str) -> float:
    quantity = analysis.quantity(quantity_id)
    assert quantity.state.value == "KNOWN", (quantity_id, quantity.state, quantity.reason, quantity.message)
    return quantity.value


def _range(analysis, quantity_id: str) -> tuple[float, float]:
    quantity = analysis.quantity(quantity_id)
    if quantity.state.value == "KNOWN":
        return quantity.value, quantity.value
    assert quantity.state.value == "BOUNDED" and quantity.interval.kind.value == "PROVEN", (quantity_id, quantity.state, quantity.reason)
    return quantity.interval.lower, quantity.interval.upper


def _overlap(w1: float, z1: float, w2: float, z2: float) -> float:
    """The power coupled between two round coaxial TEM00 beams (waists w at positions z along the axis): the square of the one-section
    overlap 2 / sqrt((w1 / w2 + w2 / w1)^2 + (lambda d / (pi w1 w2))^2)."""

    d = z2 - z1
    one = 2.0 / math.sqrt((w1 / w2 + w2 / w1) ** 2 + (LAMBDA_MM * d / (math.pi * w1 * w2)) ** 2)
    return one * one


def _fed(position=(-80.0, 0.0, 0.0), *, waist=0.3, waist_position=0.0, extra=(), models=(), materials=None, **kwargs) -> dict:
    return cavity("500", "500", 300.0, extra=(source("S", position, wavelength_nm=LAMBDA_NM, waist_mm=waist, power_w=POWER,
                                                     waist_position_mm=waist_position, **kwargs), *extra), models=models, materials=materials)


# ------------------------------------------------------------------------------------------------ what reaches the input mirror (CAV-1)


def _own_quantities_stand(analysis, plain) -> None:
    """Every quantity of the request without the injection is in the document with the injection, unchanged."""

    with_injection = {quantity.quantity_id: quantity for quantity in analysis.quantities}
    for quantity in plain.quantities:
        assert with_injection[quantity.quantity_id] == quantity, quantity.quantity_id


def _beam_is_not_the_source_s(analysis) -> dict:
    """The quantities that need the source's beam say so; returns the injection record."""

    record = analysis.arrays["injection"]
    assert record["source_problem"]["code"] == "CAVITY_SOURCE_UNSUPPORTED"
    for quantity_id in ("mode_matching_efficiency", "higher_order_mode_power_fraction", "unresolved_mode_power_fraction"):
        quantity = analysis.quantity(quantity_id)
        assert quantity.state.value == "OUT_OF_MODEL_SCOPE" and quantity.reason == "CAVITY_SOURCE_UNSUPPORTED" and quantity.value is None
    assert _value(analysis, "resonant_power_build_up") == pytest.approx(0.009 / (1.0 - 0.99) ** 2, rel=1e-12)
    assert record["coupling"]["regime"] == "UNDER_COUPLED"
    return record


def _bounded_by(analysis, arriving: float) -> None:
    """No power is KNOWN: each is bounded from zero by the arriving power as light not resolved, any phase."""

    for quantity_id in POWERS:
        quantity = analysis.quantity(quantity_id)
        assert quantity.state.value == "BOUNDED" and quantity.reason == "CAVITY_SOURCE_UNSUPPORTED" and quantity.value is None
        assert quantity.interval.kind.value == "PROVEN" and quantity.interval.lower == 0.0
    assert analysis.quantity("circulating_power_W").interval.upper == pytest.approx(arriving * 0.009 / (1.0 - 0.99) ** 2, rel=1e-12)
    assert analysis.quantity("transmitted_power_W").interval.upper == pytest.approx(arriving * PEAK, rel=1e-12)
    assert analysis.quantity("reflected_power_W").interval.upper <= arriving and analysis.quantity("lost_power_W").interval.upper <= arriving


def test_a_stop_between_the_source_and_the_input_mirror_bounds_the_injection_by_the_traced_power():
    """A 0.05 mm opening 40 mm in front of a 0.3 mm source passes about 5 % of its 2 W; the injection once took all 2 W (180 W
    circulating). The source's light is traced to M1: the power arriving there, P (1 - exp(-2 a^2 / w^2)) = 0.106 W, is the injected
    power; the beam behind the stop is no longer the source's, so its mode matching and its mode fractions are outside the model and
    no power is KNOWN -- each is bounded from zero by the arriving 0.106 W as light not resolved (9.5 W circulating at most, not
    180 W). The cavity's own quantities are those of the request without the injection, quantity for quantity."""

    stop = declared_component("A", "aperture", "stop", (-40.0, 0.0, 0.0), ALONG_X)
    doc = _fed(extra=(stop,), models=(stop_model("stop", opening_radius_mm="0.05", edge_radius_mm="12"),))
    analysis = _analyse(doc, {"source_id": "S", "beam": "SOURCE", "detuning_Hz": 0.0})
    record = _beam_is_not_the_source_s(analysis)
    assert "A/stop" in record["source_problem"]["message"] and record["traced_arrival"]["met"] == ["A/stop:aperture_stop"]
    arriving = _value(analysis, "injected_power_W")
    z_r = math.pi * 0.3 ** 2 / LAMBDA_MM
    w = 0.3 * math.sqrt(1.0 + (40.0 / z_r) ** 2)
    assert arriving == pytest.approx(POWER * (1.0 - math.exp(-2.0 * 0.05 ** 2 / w ** 2)), rel=1e-3)
    assert record["traced_arrival"]["emitted_W"] == POWER and record["traced_arrival"]["arriving_W"] == arriving
    _bounded_by(analysis, arriving)
    assert analysis.quantity("circulating_power_W").interval.upper < 10.0  # (not the 180 W of the whole 2 W)
    _own_quantities_stand(analysis, _analyse(doc, None))
    assert _value(analysis, "detuning_Hz") == 0.0 and analysis.execution.complete


@pytest.mark.parametrize("offset", [12.3, 12.1, 12.02])
def test_a_beam_partly_cut_by_a_stop_its_chief_ray_misses_has_no_injected_power_and_states_the_share_cut(offset):
    """A stop (opaque from 0.5 to 12 mm) 40 mm in front of the 0.3 mm source, centred 12.3 / 12.1 / 12.02 mm off the axis: its rim
    lies 0.30 / 0.10 / 0.02 mm from the beam's axis, which it misses, and cuts (1/2) erfc(sqrt 2 g / w) = 0.024 / 0.255 / 0.448 of
    the beam. The chief-ray trace delivers all 2 W to M1 and the injection once took them (eta 0.818, 1.325 W transmitted). The
    run flags BEAM_PARTIALLY_INTERCEPTED: the chief ray's 2 W count the light the stop takes, so the injected power has no value and
    nothing bounds the powers; the message names the stop and the share its equal-power pupil samples give (96 samples: within
    0.03 of the Gaussian integral). The cavity's own quantities stand."""

    stop = declared_component("A", "aperture", "stop", (-40.0, offset, 0.0), ALONG_X)
    doc = _fed(extra=(stop,), models=(stop_model("stop", opening_radius_mm="0.5", edge_radius_mm="12"),))
    analysis = _analyse(doc, {"source_id": "S", "beam": "SOURCE", "detuning_Hz": 0.0})
    record = _beam_is_not_the_source_s(analysis)
    injected = analysis.quantity("injected_power_W")
    assert injected.state.value == "OUT_OF_MODEL_SCOPE" and injected.reason == "CAVITY_SOURCE_UNSUPPORTED" and injected.value is None
    assert "BEAM_PARTIALLY_INTERCEPTED at A (stop)" in injected.message and "delivers 2 W of its 2 W" in injected.message
    assert record["traced_arrival"]["model_conditions"] == [{"code": "BEAM_PARTIALLY_INTERCEPTED", "instance_id": "A", "surface_id": "stop"}]
    z_r = math.pi * 0.3 ** 2 / LAMBDA_MM
    w = 0.3 * math.sqrt(1.0 + (40.0 / z_r) ** 2)
    cut = 0.5 * math.erfc(math.sqrt(2.0) * (offset - 12.0) / w)
    stated = float(re.search(r"about ([0-9.e+-]+) of the power", injected.message).group(1))
    assert stated == pytest.approx(cut, abs=0.03)
    for quantity_id in POWERS:
        quantity = analysis.quantity(quantity_id)
        assert quantity.state.value == "OUT_OF_MODEL_SCOPE" and quantity.reason == "CAVITY_SOURCE_UNSUPPORTED"
        assert quantity.value is None and quantity.interval is None
    _own_quantities_stand(analysis, _analyse(doc, None))


def test_a_stop_well_clear_of_the_beam_leaves_the_injection_as_without_it():
    """The same stop 13.5 mm off the axis (its rim 1.5 mm = 5 beam radii away: e^-50 of the power) touches no pupil sample: the
    injection is the one without the stop, value for value."""

    stop = declared_component("A", "aperture", "stop", (-40.0, 13.5, 0.0), ALONG_X)
    beside = _analyse(_fed(extra=(stop,), models=(stop_model("stop", opening_radius_mm="0.5", edge_radius_mm="12"),)),
                      {"source_id": "S", "beam": "SOURCE", "detuning_Hz": 0.0})
    alone = _analyse(_fed(), {"source_id": "S", "beam": "SOURCE", "detuning_Hz": 0.0})
    for quantity_id in ("injected_power_W", "mode_matching_efficiency", "transmitted_power_W", "circulating_power_W"):
        assert beside.quantity(quantity_id) == alone.quantity(quantity_id), quantity_id


def test_a_lens_between_the_source_and_the_input_mirror_bounds_the_injection_by_the_traced_power():
    """An uncoated N-BK7 lens on the axis between the source and M1: the chief ray crosses its two faces at normal incidence, so the
    power arriving at M1 is P (1 - ((n - 1) / (n + 1))^2)^2 with the Sellmeier index written here (1.840 W of 2 W). The lens
    changes the beam: its mode matching is outside the model and the powers are bounded by the arriving power."""

    lens = declared_component("L", "lens", "pcx", (-40.0, 0.0, 0.0), ALONG_X)
    doc = _fed(extra=(lens,), models=(plano_convex_model("pcx"),), materials=[nbk7()])
    analysis = _analyse(doc, {"source_id": "S", "beam": "SOURCE", "detuning_Hz": 0.0})
    record = _beam_is_not_the_source_s(analysis)
    assert "L/S1" in record["source_problem"]["message"] and record["traced_arrival"]["met"] == ["L/S1:refract", "L/S2:refract"]
    lam = LAMBDA_NM * 1e-3
    n = math.sqrt(1.0 + sum(float(b) * lam * lam / (lam * lam - float(c)) for b, c in zip(NBK7_SELLMEIER["b"], NBK7_SELLMEIER["c_um2"])))
    arriving = _value(analysis, "injected_power_W")
    assert arriving == pytest.approx(POWER * (1.0 - ((n - 1.0) / (n + 1.0)) ** 2) ** 2, rel=1e-9)
    _bounded_by(analysis, arriving)
    _own_quantities_stand(analysis, _analyse(doc, None))


def test_a_source_of_several_lines_leaves_the_injection_without_a_power_and_the_cavity_as_it_is():
    """The injection takes one line: a source of two lines is not a single-line Gaussian beam. Its beam and the power of the line at
    the mirror are not formed (OUT_OF_MODEL_SCOPE, CAVITY_SOURCE_UNSUPPORTED); the cavity's quantities stand. A source the Project
    does not hold is a request that contradicts the Project: refused."""

    doc = _fed()
    model = doc["components"][-1]["source_model"]
    del model["wavelength_nm"], model["power_w"]
    model["spectral_lines"] = [{"wavelength_nm": LAMBDA_NM, "power_w": 1.0}, {"wavelength_nm": 532.0, "power_w": 1.0}]
    analysis = _analyse(doc, {"source_id": "S", "beam": "SOURCE", "detuning_Hz": 0.0})
    record = _beam_is_not_the_source_s(analysis)
    assert "not a single-line Gaussian beam" in record["source_problem"]["message"]
    for quantity_id in ("injected_power_W", *POWERS):
        assert analysis.quantity(quantity_id).state.value == "OUT_OF_MODEL_SCOPE" and analysis.quantity(quantity_id).value is None
    _own_quantities_stand(analysis, _analyse(doc, None))
    with pytest.raises(CavityError) as caught:
        _analyse(_fed(), {"source_id": "OTHER", "beam": "SOURCE", "detuning_Hz": 0.0})
    assert caught.value.problems[0]["code"] == "CAVITY_SOURCE_UNSUPPORTED" and caught.value.problems[0]["location"] == "injection.source_id"


def test_an_observation_plane_on_the_way_does_not_change_the_injection():
    doc = _fed()
    doc["observations"].append(plane("q", (-40.0, 0.0, 0.0)))
    analysis = _analyse(doc, {"source_id": "S", "beam": "SOURCE", "detuning_Hz": 0.0})
    assert _value(analysis, "injected_power_W") == POWER


def test_the_input_mirror_s_substrate_takes_its_fresnel_loss_and_refracts_the_source_s_beam():
    """M1 on a 6 mm plate of constant index n: the uncoated back face reflects ((n - 1) / (n + 1))^2 of the light, and the beam crosses
    the plate (a reduced length 6 / n) and the R = 500 mm face from glass into the cavity (power (1 - n) / 500)."""

    n = float(flint()["dispersion"]["refractive_index"])
    m1 = {"model_id": "m1", "model": sequence([surface(1, "-6", clear="12.7"),
                                               surface(2, "0", radius="500", clear="12.7", traced="REFLECTED", coating_id=PARTIAL["coating_id"])],
                                              ["FLINT"], edge="13")}
    doc = _fed(waist=0.3, waist_position=0.0)
    doc["optical_models"] = [m1 if m["model_id"] == "m1" else m for m in doc["optical_models"]]
    doc["model_materials"] = [flint()]
    analysis = _analyse(doc, {"source_id": "S", "beam": "SOURCE", "detuning_Hz": 0.0})
    assert _value(analysis, "injected_power_W") == pytest.approx(POWER * (1.0 - ((n - 1.0) / (n + 1.0)) ** 2), rel=1e-9)
    # the source's waist 80 mm before the reference plane, 74 mm before the back face: q = z - z_waist + i z_R there
    q = complex(74.0, math.pi * 0.3 ** 2 / LAMBDA_MM)
    q = q + 6.0 / n
    q = q / ((n - 1.0) / 500.0 * q + 1.0)  # (the face's matrix ((1, 0), (-(1 - n) / R, 1)))
    waist = math.sqrt(LAMBDA_MM * q.imag / math.pi)
    mode = _g_mode(500.0, 500.0, 300.0)
    assert _value(analysis, "mode_matching_efficiency") == pytest.approx(_overlap(waist, -q.real, mode["w0"], mode["z1"]), rel=1e-9)
    beam = analysis.arrays["injection"]["source_beam"]
    assert beam["waist_radius_x_mm"] == pytest.approx(waist, rel=1e-12) and beam["waist_after_reference_x_mm"] == pytest.approx(-q.real, rel=1e-12)


# ------------------------------------------------------------------------------------------------ the source's beam (CAV-2)


@pytest.mark.parametrize("waist, position", [(0.3, -80.0), (1.0, -80.0), (0.05, -200.0)])
def test_the_source_s_own_beam_is_injected_with_its_mode_matching(waist, position):
    """The source's waist lies at the source: its mode matching with the eigenmode (waist w0 at the centre) is the closed-form overlap
    (0.818, 0.266, 0.038), and only that share of TEM00 builds up on resonance; the analysis used to assume a perfectly matched beam
    (1.62 W transmitted in every case)."""

    analysis = _analyse(_fed((position, 0.0, 0.0), waist=waist), {"source_id": "S", "beam": "SOURCE", "detuning_Hz": 0.0})
    mode = _g_mode(500.0, 500.0, 300.0)
    eta = _overlap(waist, position, mode["w0"], mode["z1"])
    assert _value(analysis, "mode_matching_efficiency") == pytest.approx(eta, rel=1e-10)
    rows = analysis.arrays["injection"]["modes"]
    tem00 = next(row for row in rows if row["m"] == 0 and row["n"] == 0)
    assert tem00["power_fraction"] == pytest.approx(eta, rel=1e-10)
    assert tem00["transmitted_W"] == pytest.approx(POWER * eta * PEAK, rel=1e-10)
    for row in rows:  # every other mode at its Gouy-shifted phase, by the complex field sum
        delta = -(row["m"] + row["n"]) * mode["gouy"]
        assert row["transmitted_W"] == pytest.approx(POWER * row["power_fraction"] * _airy(0.99, 0.009, 0.99, 0.009, 1.0, delta)["transmitted"],
                                                      rel=1e-9, abs=1e-18)
    low, high = _range(analysis, "transmitted_power_W")
    assert POWER * eta * PEAK <= low * (1.0 + 1e-12) and high <= POWER * PEAK * (1.0 + 1e-12)
    assert high < POWER * PEAK * (1.0 - 1e-3) or eta > 0.999


def test_an_input_beam_that_contradicts_the_named_source_is_refused_and_a_matching_one_accepted():
    doc = _fed()
    with pytest.raises(CavityError) as caught:
        _analyse(doc, {"source_id": "S", "beam": "INPUT_BEAM", "detuning_Hz": 0.0},
                 input_beam={"waist_radius_mm": 0.2786, "waist_after_reference_mm": 150.0})
    assert caught.value.problems[0]["location"] == "input_beam" and "is not the beam of source 'S'" in caught.value.problems[0]["message"]
    stated = _analyse(doc, {"source_id": "S", "beam": "INPUT_BEAM", "detuning_Hz": 0.0}, input_beam={"waist_radius_mm": 0.3, "waist_after_reference_mm": -80.0})
    own = _analyse(doc, {"source_id": "S", "beam": "SOURCE", "detuning_Hz": 0.0})
    assert _range(stated, "transmitted_power_W") == _range(own, "transmitted_power_W")
    with pytest.raises(CavityError) as caught:  # a source's beam is its own: EIGENMODE takes power_w
        _analyse(doc, {"source_id": "S", "beam": "EIGENMODE", "detuning_Hz": 0.0})
    assert caught.value.problems[0]["location"] == "injection.beam"


# ------------------------------------------------------------------------------------------------ the source's polarization (CAV-3)


def _crystal_cavity(polarization_deg: float) -> dict:
    """A 200 mm cavity with a calcite plate whose optic axis is its local X, in its faces; the source's local X is the same direction,
    so a linear polarization at angle phi lies at phi to the optic axis."""

    doc = cavity("500", "500", 200.0, extra=(declared_component("X", "prism", "BD", (95.0, 0.0, 0.0), ALONG_X),
                                             source("S", (-80.0, 0.0, 0.0), wavelength_nm=LAMBDA_NM, waist_mm=0.3, power_w=1.0,
                                                    polarization_deg=polarization_deg)))
    doc["optical_models"].append(displacer("BD", axis=("1", "0", "0")))
    doc["model_materials"] = [CALCITE]
    return doc


@pytest.mark.parametrize("angle", [0.0, 30.0, 45.0, 90.0])
def test_the_source_s_polarization_divides_its_power_between_the_crystal_s_two_families(angle):
    """Malus: cos^2(phi) of the power is polarized along the optic axis (the extraordinary family), sin^2(phi) across it (ordinary).
    Each family's TEM00 transmits its share x its mode matching x the Airy peak; the two families together never exceed the input (they
    used to transmit 0.81 W each from 1 W)."""

    rows, shares, highs = {}, {"EXTRAORDINARY": math.cos(math.radians(angle)) ** 2, "ORDINARY": math.sin(math.radians(angle)) ** 2}, []
    for family in ("ORDINARY", "EXTRAORDINARY"):
        analysis = _analyse(_crystal_cavity(angle), {"source_id": "S", "beam": "SOURCE", "detuning_Hz": 0.0, "mode_order_max": 0},
                            components=("M1", "X", "M2"), polarization=family, round_trip_loss=0.0, reflectances={"M1": 0.99, "M2": 0.99},
                            transmittances={"M1": 0.009, "M2": 0.009})
        share = _value(analysis, "polarization_family_power_fraction")
        assert share == pytest.approx(shares[family], abs=1e-12)
        eta = _value(analysis, "mode_matching_efficiency")
        tem00 = analysis.arrays["injection"]["modes"][0]
        assert tem00["transmitted_W"] == pytest.approx(1.0 * share * eta * PEAK, rel=1e-10, abs=1e-15)
        budget = analysis.arrays["injection"]["power_budget"]
        assert budget["other_polarization_W"] == pytest.approx(1.0 - share, abs=1e-12) and abs(budget["closure_residual_W"]) < 1e-12
        highs.append(_range(analysis, "transmitted_power_W")[1])
        rows[family] = share
    assert sum(highs) <= 1.0 * PEAK * (1.0 + 1e-12)
    assert rows["ORDINARY"] + rows["EXTRAORDINARY"] == pytest.approx(1.0, abs=1e-12)


def test_an_unpolarized_source_gives_each_family_half():
    doc = _crystal_cavity(0.0)
    doc["components"][-1]["source_model"]["polarization"] = {"kind": "UNPOLARIZED"}
    analysis = _analyse(doc, {"source_id": "S", "beam": "SOURCE", "detuning_Hz": 0.0, "mode_order_max": 0}, components=("M1", "X", "M2"),
                        polarization="ORDINARY", round_trip_loss=0.0, reflectances={"M1": 0.99, "M2": 0.99}, transmittances={"M1": 0.009, "M2": 0.009})
    assert _value(analysis, "polarization_family_power_fraction") == pytest.approx(0.5, abs=1e-15)
    assert _value(analysis, "injected_power_W") == pytest.approx(1.0, rel=1e-15)  # (its two incoherent halves, traced apart, summed)


# ------------------------------------------------------------------------------------------------ target units (CAV-4)


def test_a_cavity_target_in_another_unit_is_not_assessed_and_the_check_warns():
    """A free-spectral-range target stated in MHz (400 - 600) used to be judged against the value in Hz (UNMET)."""

    for unit, assessment in (("MHz", "INDETERMINATE"), ("Hz", "UNMET"), (None, "UNMET")):
        target = {"target_id": "f", "metric": "cavity_free_spectral_range_Hz", "minimum": 400.0, "maximum": 600.0}
        if unit is not None:
            target["unit"] = unit
        doc = cavity("500", "500", 300.0, targets=[target], extra=(source("S", (-80.0, 0.0, 0.0), wavelength_nm=LAMBDA_NM),))
        (row,) = _analyse(doc, None, reflectances={"M1": 0.99, "M2": 0.99}).arrays["targets"]
        assert row["assessment"] == assessment and row["metric_unit"] == "Hz"
        warnings = project_warnings(load_project(doc), lower_project(doc, None))
        assert [w["code"] for w in warnings] == (["TARGET_UNIT_MISMATCH"] if unit == "MHz" else [])
        if unit == "MHz":
            assert row["reason"] == "TARGET_UNIT_MISMATCH" and "cavity analysis" in warnings[0]["message"]
    assert json_api.capabilities()["cavity_analysis"]["target_metric_units"]["cavity_linewidth_Hz"] == "Hz"


# ------------------------------------------------------------------------------------------------ the detuning (CAV-5)


@pytest.mark.parametrize("length", [300.0, 300.0001, 300.00027])
def test_without_a_stated_detuning_the_powers_are_bounded_over_every_detuning(length):
    """Nominal lengths 0.27 um apart gave transmissions from 4.5e-5 to 5.6e-4 W, each reported as known. A Project's positions do not
    fix the detuning: the powers are now the Airy function's range, T1 T2 / (1 + a)^2 to T1 T2 / (1 - a)^2 of the input, the same for
    every nominal length, and the values with TEM00 on resonance are listed apart."""

    analysis = _analyse(cavity("500", "500", length), {"power_w": POWER, "beam": "EIGENMODE"})
    detuning = analysis.quantity("detuning_Hz")
    assert detuning.state.value == "UNKNOWN" and detuning.reason == "DETUNING_NOT_STATED"
    a = 0.99
    assert _range(analysis, "transmitted_power_W") == pytest.approx((POWER * 0.009 * 0.009 / (1.0 + a) ** 2, POWER * PEAK), rel=1e-12)
    low, high = _range(analysis, "reflected_power_W")
    on, off = _airy(0.99, 0.009, 0.99, 0.009, 1.0, 0.0)["reflected"], _airy(0.99, 0.009, 0.99, 0.009, 1.0, math.pi)["reflected"]
    assert (low, high) == pytest.approx((POWER * min(on, off), POWER * max(on, off)), rel=1e-11)
    at = analysis.arrays["injection"]["at_tem00_resonance"]
    assert at["transmitted_W"] == pytest.approx(POWER * PEAK, rel=1e-12) and at["circulating_W"] == pytest.approx(POWER * 0.009 / (1.0 - a) ** 2, rel=1e-12)
    assert analysis.arrays["injection"]["detuning_from"] == "NOT_STATED"
    # no detuning, no value: the powers name the input that is missing; the channels with TEM00 on resonance are quantities of their own
    transmitted = analysis.quantity("transmitted_power_W")
    assert transmitted.value is None and transmitted.value_basis is None and transmitted.missing == ("injection.detuning_Hz",)
    assert _value(analysis, "at_tem00_resonance.transmitted_W") == at["transmitted_W"]
    assert _value(analysis, "at_tem00_resonance.circulating_W") == at["circulating_W"]
    channels = ("reflected_W", "transmitted_W", "input_mirror_loss_W", "output_mirror_loss_W", "inside_loss_W")
    assert sum(_value(analysis, f"at_tem00_resonance.{key}") for key in channels) == pytest.approx(POWER, rel=1e-13)


def test_without_a_stated_detuning_a_mismatched_beam_s_resonant_channels_are_lower_bounds():
    """A source whose beam is not the eigenmode, resolved to order 2, no detuning: the channels with TEM00 on resonance are the resolved
    modes' sums, lower bounds (the light above the order only adds, at most its share at any phase). TEM00 and the order-2 modes at
    their Gouy-shifted phases, as complex field sums, give the transmitted value; the channels' values and the light not resolved add
    up to the injected power."""

    analysis = _analyse(_fed(), {"source_id": "S", "beam": "SOURCE", "mode_order_max": 2})
    injection = analysis.arrays["injection"]
    unresolved = _value(analysis, "unresolved_mode_power_fraction")
    assert unresolved > 1e-3
    expected = POWER * sum(row["power_fraction"] * _airy(0.99, 0.009, 0.99, 0.009, 1.0, -row["gouy_shift_rad"])["transmitted"]
                           for row in injection["modes"])
    transmitted = analysis.quantity("at_tem00_resonance.transmitted_W")
    assert transmitted.state.value == "BOUNDED" and transmitted.value == pytest.approx(expected, rel=1e-9)
    assert transmitted.value_basis.kind.value == "LOWER_BOUND" and transmitted.value_basis.omitted[0].part.value == "UNRESOLVED_MODES"
    assert transmitted.interval.lower == transmitted.value
    assert transmitted.interval.upper == pytest.approx(transmitted.value + POWER * unresolved * PEAK, rel=1e-12)
    channels = ("reflected_W", "transmitted_W", "input_mirror_loss_W", "output_mirror_loss_W", "inside_loss_W")
    resolved = sum(analysis.quantity(f"at_tem00_resonance.{key}").value for key in channels)
    assert resolved + POWER * unresolved == pytest.approx(POWER, rel=1e-12)
    for key in ("circulating_W", "at_output_mirror_W", *channels):
        quantity = analysis.quantity(f"at_tem00_resonance.{key}")
        assert quantity.interval.lower >= quantity.value and quantity.interval.upper >= quantity.interval.lower
        assert quantity.value_basis.omitted[0].upper == pytest.approx(quantity.interval.upper - quantity.value, rel=1e-9, abs=1e-18)


def _airy_sum(rows: list, delta: np.ndarray) -> dict:
    """The resolved modes' channels at TEM00 round-trip phases ``delta``, each mode at its Gouy-shifted phase, as complex field sums
    (R = 0.99, T = 0.009 mirrors), per watt injected."""

    r, t = math.sqrt(0.99), math.sqrt(0.009)
    out = {"transmitted": np.zeros_like(delta), "reflected": np.zeros_like(delta), "circulating": np.zeros_like(delta)}
    for row in rows:
        e = np.exp(-1j * (delta - row["gouy_shift_rad"]))
        inside = t / (1.0 - r * r * e)
        out["circulating"] += row["power_fraction"] * np.abs(inside) ** 2
        out["transmitted"] += row["power_fraction"] * 0.009 * np.abs(inside) ** 2
        out["reflected"] += row["power_fraction"] * np.abs(-r + t * r * e * inside) ** 2
    return out


def _extremes(rows: list) -> dict:
    """Per channel the minimum and maximum over every detuning of its sum: a 200 001-point grid, then around each of its five
    largest local maxima (and smallest minima) 2001 points over the two grid steps beside it."""

    grid = np.linspace(-math.pi, math.pi, 200001)
    step = grid[1] - grid[0]
    sums = _airy_sum(rows, grid)
    out = {}
    for channel, values in sums.items():
        found = []
        for sign in (-1.0, 1.0):
            signed = sign * values
            peaks = np.flatnonzero((signed >= np.roll(signed, 1)) & (signed >= np.roll(signed, -1)))
            near = np.concatenate([np.linspace(grid[i] - step, grid[i] + step, 2001) for i in peaks[np.argsort(signed[peaks])[-5:]]])
            found.append(sign * max(float(signed.max()), float(np.max(sign * _airy_sum(rows, near)[channel]))))
        out[channel] = tuple(found)
    return out


@pytest.mark.parametrize("waist, position, order", [(0.3, -80.0, 10), (1.0, -80.0, 10), (0.05, -200.0, 60)])
def test_without_a_stated_detuning_the_bounds_are_those_of_the_mode_sum_not_of_each_mode(waist, position, order):
    """A mismatched source (eta 0.818 / 0.266 / 0.038) without a stated detuning: its modes resonate at different detunings, so
    no detuning puts them all on resonance. The bounds used to sum each mode's own extremes (transmitted up to 1.62 W in every
    case: 1.22 / 3.8 / 25 times the true maximum of the sum; reflected down to 0.014 W where the sum never falls below 1.92 W).
    They now bound the sum itself: the resolved modes' part of each bound is within 1e-4 of the extreme of the complex Airy sum
    over the detuning (a fine grid, refined around its extremes), and never inside it."""

    analysis = _analyse(_fed((position, 0.0, 0.0), waist=waist), {"source_id": "S", "beam": "SOURCE", "mode_order_max": order})
    injection = analysis.arrays["injection"]
    rows = injection["modes"]
    unresolved = injection["not_resolved_bounds"]
    extremes = _extremes(rows)
    for quantity_id, channel, key in (("transmitted_power_W", "transmitted", "transmitted_W"),
                                      ("circulating_power_W", "circulating", "circulating_W"),
                                      ("reflected_power_W", "reflected", "reflected_W")):
        low, high = _range(analysis, quantity_id)
        true_low, true_high = (POWER * x for x in extremes[channel])
        resolved_low, resolved_high = low - unresolved[key][0], high - unresolved[key][1]
        assert true_low * (1.0 - 1e-4) <= resolved_low <= true_low * (1.0 + 1e-12), (quantity_id, low, true_low)
        assert true_high * (1.0 - 1e-12) <= resolved_high <= true_high * (1.0 + 1e-4), (quantity_id, high, true_high)
        if quantity_id == "transmitted_power_W":
            assert high < POWER * PEAK * (1.0 - 1e-3) or waist == 0.3  # (no longer the perfectly matched maximum)


# ------------------------------------------------------------------------------------------------ the unresolved light (CAV-6)


@pytest.mark.parametrize("eta", [0.5, 0.02, 1e-3, 2e-4, 5e-5])
def test_the_unresolved_fraction_is_the_complement_of_the_listed_modes_and_never_above_one(eta):
    """The fraction above order 10 of a beam whose TEM00 share per section is eta, against 1 minus the listed fractions
    eta C(2k, k) 4^-k (1 - eta^2)^k summed exactly in rationals (it used to reach 11.3 for eta = 5e-5)."""

    e = Fraction(eta)
    rest = 1 - e * e
    one = [e * math.comb(2 * k, k) / 4 ** k * rest ** k for k in range(6)]
    listed = sum(one[i] * one[j] for i in range(6) for j in range(6) if 2 * (i + j) <= 10)
    value = unresolved_fraction(eta, eta, 10)
    assert value <= 1.0 and value == pytest.approx(float(1 - listed), rel=1e-12, abs=1e-15)


def _mirrors_only(analysis) -> dict:
    """The quantities that read the mirrors' coefficients alone, as dumped."""

    return {quantity_id: analysis.quantity(quantity_id).model_dump() for quantity_id in (
        "resonant_power_build_up", "resonant_transmission", "resonant_reflection", "impedance_matching_reflectance", "input_loss_matching_reflectance")}


POWERS = ("circulating_power_W", "transmitted_power_W", "reflected_power_W", "lost_power_W", "power_build_up", "transmission", "reflection")


def test_a_beam_wider_than_the_input_mirror_s_aperture_is_bounded_from_zero_as_light_not_resolved():
    """A beam 2000 times the eigenmode (557 mm) on a 12.7 mm mirror: its transmission used to be bounded by 1.05 of the injected power.
    The mirror's edge cuts the beam, so its transverse content is not the Gaussian beam's: all its light is light not resolved, each power
    between 0 (how much enters is not known) and the bound for any phase, never above the injected power for a power that leaves or is
    lost. The resonant quantities and the coupling read the mirrors alone: they are those of the same cavity fed with its eigenmode."""

    mode = _g_mode(500.0, 500.0, 300.0)
    mirrors = {"reflectances": {"M1": 0.99, "M2": 0.99}, "transmittances": {"M1": 0.01, "M2": 0.01}}
    matched = _analyse(cavity("500", "500", 300.0), {"power_w": 1.0, "beam": "EIGENMODE", "detuning_Hz": 0.0}, **mirrors)
    for scale, clipped in ((2000.0, True), (100.0, True), (5.0, False)):
        waist = mode["w0"] * scale
        analysis = _analyse(cavity("500", "500", 300.0), {"power_w": 1.0, "beam": "INPUT_BEAM", "detuning_Hz": 0.0, "mode_order_max": 60},
                            input_beam={"waist_radius_mm": waist, "waist_after_reference_mm": 150.0}, **mirrors)
        transmitted = analysis.quantity("transmitted_power_W")
        radius = waist * math.sqrt(1.0 + (150.0 * LAMBDA_MM / (math.pi * waist * waist)) ** 2)  # (the beam at the mirror, 150 mm from its waist)
        check = next(c for c in analysis.models[-1].validity if c.parameter == "input_mirror_clear_radius_over_beam_radius")
        assert check.observed == pytest.approx(12.7 / radius, rel=1e-12) and check.minimum == 3.0 and check.within is (not clipped)
        assert _mirrors_only(analysis) == _mirrors_only(matched) and _value(analysis, "detuning_Hz") == 0.0
        assert analysis.arrays["injection"]["coupling"] == matched.arrays["injection"]["coupling"]
        if clipped:
            assert check.code == "INPUT_BEAM_CLIPPED" and check.at == ("M1",)
            assert analysis.arrays["injection"]["not_decomposed"] == ["INPUT_BEAM_CLIPPED"] and analysis.arrays["injection"]["modes"] == []
            assert _value(analysis, "unresolved_mode_power_fraction") == 1.0
            for quantity_id in POWERS:
                quantity = analysis.quantity(quantity_id)
                assert quantity.state.value == "BOUNDED" and quantity.reason == "INPUT_BEAM_CLIPPED" and quantity.value is None
                assert quantity.interval.kind.value == "PROVEN" and quantity.interval.lower == 0.0
            for quantity_id in ("transmitted_power_W", "reflected_power_W", "lost_power_W", "transmission", "reflection"):
                assert analysis.quantity(quantity_id).interval.upper <= 1.0
            assert transmitted.interval.upper == pytest.approx(0.01 * 0.01 / (1.0 - 0.99) ** 2, rel=1e-12)  # (all of it matched, on resonance)
            assert analysis.quantity("circulating_power_W").interval.upper == pytest.approx(0.01 / (1.0 - 0.99) ** 2, rel=1e-12)
        else:
            assert check.code is None and "not_decomposed" not in analysis.arrays["injection"]
            low, high = _range(analysis, "transmitted_power_W")
            assert 0.0 <= low <= high <= 1.0


def _wide_mirror(model_id: str) -> dict:
    return {"model_id": model_id, "model": sequence([surface(1, "0", radius="500", clear="400", traced="REFLECTED", coating_id=PARTIAL["coating_id"])], [],
                                                    edge="401")}


def test_a_far_mismatched_beam_s_bounds_never_exceed_the_injected_power():
    """Mirrors of 400 mm clear radius take a beam 400 times the eigenmode (111 mm): about 1e-5 of it couples into each resolved mode, the
    rest is bounded; a lossless cavity (T1 T2 / (1 - a)^2 = 1) may transmit at most what enters."""

    doc = project([declared_component("M1", "mirror", "m1", (0.0, 0.0, 0.0), ALONG_X), declared_component("M2", "mirror", "m2", (300.0, 0.0, 0.0), AGAINST_X)],
                  [plane("p", (150.0, 0.0, 0.0))])
    doc = with_models(doc, [_wide_mirror("m1"), _wide_mirror("m2")], None, coatings=[PARTIAL])
    mode = _g_mode(500.0, 500.0, 300.0)
    analysis = _analyse(doc, {"power_w": 1.0, "beam": "INPUT_BEAM", "detuning_Hz": 0.0, "mode_order_max": 60},
                        reflectances={"M1": 0.99, "M2": 0.99}, transmittances={"M1": 0.01, "M2": 0.01},
                        input_beam={"waist_radius_mm": mode["w0"] * 400.0, "waist_after_reference_mm": 150.0})
    assert 0.0 < _value(analysis, "unresolved_mode_power_fraction") <= 1.0
    for quantity_id in ("transmitted_power_W", "reflected_power_W", "lost_power_W", "transmission", "reflection"):
        low, high = _range(analysis, quantity_id)
        assert 0.0 <= low <= high <= 1.0, quantity_id


# ------------------------------------------------------------------------------------------------ impedance matching (CAV-7)


def test_a_lossy_input_mirror_is_matched_at_its_own_loss_and_the_coupling_regime_is_named():
    """With A1 = 0.01 held, the reflection on resonance vanishes at R1 = (1 - A1)^2 R2 (0.950697 for R2 = 0.97), not at R1 = R2."""

    r2, t2, a1 = 0.97, 0.02, 0.01
    first = _analyse(cavity("500", "500", 300.0), {"power_w": 1.0, "beam": "EIGENMODE", "detuning_Hz": 0.0},
                     reflectances={"M1": 0.97, "M2": r2}, transmittances={"M1": 0.02, "M2": t2})
    matched = _value(first, "input_loss_matching_reflectance")
    assert matched == pytest.approx((1.0 - a1) ** 2 * r2, rel=1e-14) and _value(first, "impedance_matching_reflectance") == pytest.approx(r2, rel=1e-14)
    for r1, regime in ((matched, "CRITICAL"), (0.97, "UNDER_COUPLED"), (0.93, "OVER_COUPLED")):
        t1 = 1.0 - a1 - r1
        analysis = _analyse(cavity("500", "500", 300.0), {"power_w": 1.0, "beam": "EIGENMODE", "detuning_Hz": 0.0},
                            reflectances={"M1": r1, "M2": r2}, transmittances={"M1": t1, "M2": t2})
        reference = _airy(r1, t1, r2, t2, 1.0, 0.0)["reflected"]
        assert _value(analysis, "resonant_reflection") == pytest.approx(reference, rel=1e-9, abs=1e-14)
        assert analysis.arrays["injection"]["coupling"]["regime"] == regime
        if regime == "CRITICAL":
            assert reference < 1e-20
        else:  # the reflected field on resonance, -r1 + t1^2 b / (1 - r1 b): its sign tells which term dominates
            field = -math.sqrt(r1) + t1 * math.sqrt(r2) / (1.0 - math.sqrt(r1 * r2))
            assert (field < 0.0) == (regime == "UNDER_COUPLED")


# ------------------------------------------------------------------------------------------------ the model's edges (CAV-8, CAV-9, CAV-10)


def test_a_cavity_of_a_few_wavelengths_and_a_detuning_beyond_the_carrier_are_refused():
    inj = {"power_w": 1.0, "beam": "EIGENMODE", "detuning_Hz": 0.0}
    with pytest.raises(CavityError) as caught:
        _analyse(cavity("500", "500", 1e-6), inj, reflectances={"M1": 0.99, "M2": 0.99}, transmittances={"M1": 0.01, "M2": 0.01})
    assert caught.value.problems[0]["code"] == "CAVITY_TOO_SHORT"
    carrier = C_MM_PER_S / LAMBDA_MM
    for injection in ({**inj, "detuning_Hz": 1e15}, {**inj, "detuning_Hz": -carrier},
                      {**inj, "scan": {"from_Hz": -1e8, "to_Hz": 2.0 * carrier, "points": 3}}):
        with pytest.raises(CavityError) as caught:
            _analyse(cavity("500", "500", 300.0), injection)
        assert caught.value.problems[0]["code"] == "PARAXIAL_REQUEST_INVALID" and "frequency" in caught.value.problems[0]["message"]
    assert _value(_analyse(cavity("500", "500", 0.02), None, reflectances={"M1": 0.99, "M2": 0.99}), "cavity_length_mm") == pytest.approx(0.02)


def test_an_input_mirror_of_zero_reflectance_is_a_legitimate_input():
    """R1 = 0, T1 = 1: the light enters once, M2 transmits T2 and returns R2 of it, which leaves through M1 (closed forms; it used to be
    refused). The finesse of a round trip returning nothing is 0 and the resonances have no half maximum."""

    analysis = _analyse(cavity("500", "500", 300.0), {"power_w": 1.0, "beam": "EIGENMODE", "detuning_Hz": 0.0},
                        reflectances={"M1": 0.0, "M2": 0.99}, transmittances={"M1": 1.0, "M2": 0.01})
    assert _value(analysis, "circulating_power_W") == pytest.approx(1.0, rel=1e-15)
    assert _value(analysis, "transmitted_power_W") == pytest.approx(0.01, rel=1e-15)
    assert _value(analysis, "reflected_power_W") == pytest.approx(_airy(0.0, 1.0, 0.99, 0.01, 1.0, 0.0)["reflected"], rel=1e-14)
    assert _value(analysis, "finesse") == 0.0 and analysis.quantity("linewidth_fwhm_Hz").reason == "NO_HALF_MAXIMUM"


def test_a_source_with_m2_above_one_is_bounded_as_light_not_resolved_by_its_own_reason():
    """M^2 = 1.3 is a higher-order or multimode transverse beam, whose content in the cavity's modes the model does not decompose: its
    powers are bounded as light not resolved, between 0 and the bound for any phase (2 W in: at most 180 W circulating, 1.62 W
    transmitted, with every watt matched and on resonance), by SOURCE_NOT_FUNDAMENTAL_MODE (it used to be named
    SOURCE_NOT_SINGLE_FREQUENCY, which is the pulsed / broad-line reason). The detuning and the quantities of the mirrors alone are
    those of the same source with M^2 = 1."""

    doc = _fed()
    plain = _analyse(doc, {"source_id": "S", "beam": "SOURCE", "detuning_Hz": 0.0})
    doc["components"][-1]["source_model"]["m2"] = 1.3
    analysis = _analyse(doc, {"source_id": "S", "beam": "SOURCE", "detuning_Hz": 0.0})
    assert _mirrors_only(analysis) == _mirrors_only(plain) and _value(analysis, "resonant_power_build_up") == pytest.approx(0.009 / (1.0 - 0.99) ** 2)
    assert analysis.quantity("detuning_Hz") == plain.quantity("detuning_Hz") and _value(analysis, "injected_power_W") == POWER
    assert analysis.arrays["injection"]["coupling"] == plain.arrays["injection"]["coupling"]
    for quantity_id in POWERS:
        quantity = analysis.quantity(quantity_id)
        assert quantity.state.value == "BOUNDED" and quantity.reason == "SOURCE_NOT_FUNDAMENTAL_MODE" and quantity.value is None
        assert quantity.interval.kind.value == "PROVEN" and quantity.interval.lower == 0.0 and "M^2 is 1.3" in quantity.message
    assert analysis.quantity("circulating_power_W").interval.upper == pytest.approx(POWER * 0.009 / (1.0 - 0.99) ** 2, rel=1e-12)
    assert analysis.quantity("transmitted_power_W").interval.upper == pytest.approx(POWER * PEAK, rel=1e-12)
    for quantity_id in ("transmitted_power_W", "reflected_power_W", "lost_power_W"):
        assert analysis.quantity(quantity_id).interval.upper <= POWER
    (check,) = [c for c in analysis.models[-1].validity if c.parameter == "source_m2"]
    assert (check.observed, check.maximum, check.within, check.code) == (1.3, 1.0, False, "SOURCE_NOT_FUNDAMENTAL_MODE")
    assert analysis.arrays["injection"]["not_decomposed"] == ["SOURCE_NOT_FUNDAMENTAL_MODE"]
    bounded = json_api.capabilities()["cavity_analysis"]["injection"]["bounded"]
    assert any("M^2 > 1" in item and "SOURCE_NOT_FUNDAMENTAL_MODE" in item for item in bounded)
