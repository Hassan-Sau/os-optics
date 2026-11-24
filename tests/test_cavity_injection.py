"""Light injected into a linear cavity through its first mirror (``os_optics.local.cavity``, ``injection``): steady-state circulating,
transmitted, reflected and lost powers, the transverse-mode content of a mismatched beam, targets, the run's results query, and how the
single-pass trace and the exposure model's return bounds stand beside the analysis.

Every expected number is computed here independently of the code under test: the Fabry-Perot Airy functions written out with complex
field sums (``E_c = t1 E / (1 - r1 r2' e^{-i delta})``), the enhancement ``1 / (1 - sqrt(R1 R2))^2`` on resonance, impedance matching
(``R1 = R2 x losses`` gives no reflection), the coefficient-of-finesse form of the detuning curve, the Hermite-Gauss and Laguerre-Gauss
overlap integrals evaluated by quadrature on a grid (numpy polynomials), the closed-form waist mismatch ``(2 w0 w / (w0^2 + w^2))^2``,
the two-mirror g-parameter eigenmode and the round-trip phase from the exact rational number of wavelengths in the round trip.
"""

from __future__ import annotations

import io
import json
import math
from contextlib import redirect_stdout
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from declared_helpers import declared_component, nbk7, sequence, surface, with_models
from local_helpers import AGAINST_X, ALONG_X, plane, project, source
from os_optics.cli import main as cli_main
from os_optics.local import json_api
from os_optics.local.cavity import CavityError, cavity_analysis
from os_optics.local.errors import LocalRunError
from os_optics.local.cavity_build_up import lg_fractions, one_section_fractions
from os_optics.local.lowering import lower_project
from os_optics.local.run import run_optical_project
from os_optics.results import load_run

LAMBDA_NM = 1064.0
LAMBDA_MM = LAMBDA_NM * 1e-6
C_MM_PER_S = 299792458.0e3
POWER = 2.0
#: A partially transmitting reflector: R = 0.99, T = 0.009 (A = 0.001) at every angle up to 45 deg.
PARTIAL = {"coating_id": "DIELECTRIC_HIGH_REFLECTOR.400-1100R0.99T0.009.AOI0-45", "kind": "DIELECTRIC_HIGH_REFLECTOR", "display_name": "Partial reflector",
           "bands": [{"min_nm": "400", "max_nm": "1100", "reflectance": "0.99", "transmittance": "0.009", "status": "DESIGN_BASIS"}],
           "valid_aoi_deg": {"min_deg": "0", "max_deg": "45", "status": "SPECIFIED"}, "aoi_reference": "AIR_EQUIVALENT", "basis": "Test coating."}
#: An opaque reflector: R = 0.99, T = 0.
OPAQUE = {**PARTIAL, "coating_id": "METALLIC_REFLECTOR.400-1100R0.99T0.AOI0-45", "kind": "METALLIC_REFLECTOR", "display_name": "Opaque reflector",
          "bands": [{"min_nm": "400", "max_nm": "1100", "reflectance": "0.99", "transmittance": "0", "status": "DESIGN_BASIS"}]}


def _mirror(model_id: str, radius: str | None, coating: dict, clear: str = "12.7") -> dict:
    return {"model_id": model_id, "model": sequence([surface(1, "0", radius=radius, clear=clear, traced="REFLECTED", coating_id=coating["coating_id"])], [],
                                                    edge="13")}


def cavity(r1: str | None, r2: str | None, length: float, *, coating: dict = PARTIAL, extra: tuple = (), models: tuple = (), targets=None,
           clear: str = "12.7", materials: list | None = None) -> dict:
    """M1 at x = 0 facing +X (the input mirror), M2 at x = length facing -X; a radius > 0 is concave towards the cavity."""

    doc = project([declared_component("M1", "mirror", "m1", (0.0, 0.0, 0.0), ALONG_X), declared_component("M2", "mirror", "m2", (length, 0.0, 0.0), AGAINST_X),
                   *extra], [plane("p", (length / 2.0, 0.0, 0.0))], targets)
    return with_models(doc, [_mirror("m1", r1, coating, clear), _mirror("m2", r2, coating, clear), *models], materials, coatings=[coating])


def _analyse(doc: dict, injection: dict, **request):
    return cavity_analysis(doc, {"components": ["M1", "M2"], "wavelength_nm": LAMBDA_NM, **request, "injection": injection})


def _value(analysis, quantity_id: str) -> float:
    quantity = analysis.quantity(quantity_id)
    assert quantity.state.value == "KNOWN", (quantity_id, quantity.state, quantity.reason, quantity.message)
    return quantity.value


def _airy(r1: float, t1: float, r2: float, t2: float, passing: float, delta: float) -> dict:
    """The Fabry-Perot fields summed as complex amplitudes (an independent statement of the model)."""

    e = complex(math.cos(delta), -math.sin(delta))
    back = math.sqrt(r2) * passing
    inside = math.sqrt(t1) / (1.0 - math.sqrt(r1) * back * e)
    reflected = -math.sqrt(r1) + math.sqrt(t1) * back * e * inside
    return {"circulating": abs(inside) ** 2, "transmitted": t2 * passing * abs(inside) ** 2, "reflected": abs(reflected) ** 2}


def _g_mode(r1: float | None, r2: float | None, length: float) -> dict:
    g1 = 1.0 if r1 is None else 1.0 - length / r1
    g2 = 1.0 if r2 is None else 1.0 - length / r2
    p = g1 * g2
    if g1 == 0.0 and g2 == 0.0:  # the symmetric confocal cavity: w0^2 = lambda L / (2 pi) at the centre
        w0_sq, z1 = LAMBDA_MM * length / (2.0 * math.pi), length / 2.0
    else:
        w0_sq = LAMBDA_MM * length / math.pi * math.sqrt(p * (1.0 - p) / (g1 + g2 - 2.0 * p) ** 2)
        z1 = g2 * (1.0 - g1) * length / (g1 + g2 - 2.0 * p)
    return {"w0": math.sqrt(w0_sq), "z1": z1, "gouy": 2.0 * math.acos(math.copysign(math.sqrt(p), g1)), "q": complex(-z1, math.pi * w0_sq / LAMBDA_MM)}


# ------------------------------------------------------------------------------------------------ Airy functions


def test_on_resonance_the_build_up_is_the_enhancement_with_the_input_transmittance_and_the_power_budget_closes():
    analysis = _analyse(cavity("500", "500", 300.0), {"power_w": POWER, "beam": "EIGENMODE", "detuning_Hz": 0.0},
                        reflectances={"M1": 0.97, "M2": 0.995}, transmittances={"M1": 0.029, "M2": 0.004})
    enhancement = 1.0 / (1.0 - math.sqrt(0.97 * 0.995)) ** 2
    assert _value(analysis, "power_build_up") == pytest.approx(0.029 * enhancement, rel=1e-13)
    assert _value(analysis, "circulating_power_W") == pytest.approx(POWER * 0.029 * enhancement, rel=1e-13)
    assert _value(analysis, "transmitted_power_W") == pytest.approx(POWER * 0.029 * 0.004 * enhancement, rel=1e-13)
    assert _value(analysis, "resonant_power_build_up") == pytest.approx(0.029 * enhancement, rel=1e-13)
    airy = _airy(0.97, 0.029, 0.995, 0.004, 1.0, 0.0)
    assert _value(analysis, "reflected_power_W") == pytest.approx(POWER * airy["reflected"], rel=1e-11)
    assert _value(analysis, "resonant_reflection") == pytest.approx(airy["reflected"], rel=1e-11)
    budget = analysis.arrays["injection"]["power_budget"]
    channels = sum(budget[k] for k in ("reflected_W", "transmitted_W", "input_mirror_loss_W", "output_mirror_loss_W", "inside_loss_W", "not_resolved_W"))
    assert channels == pytest.approx(POWER, rel=1e-13) and abs(budget["closure_residual_W"]) < 1e-13
    # the losses: A1 of the injected and of the returned light at M1, A2 of the light at M2 (each with its own formula)
    circulating = POWER * 0.029 * enhancement
    assert budget["input_mirror_loss_W"] == pytest.approx(0.001 * (POWER + 0.995 * circulating), rel=1e-12)
    assert budget["output_mirror_loss_W"] == pytest.approx(0.001 * circulating, rel=1e-12)
    assert _value(analysis, "lost_power_W") == pytest.approx(budget["input_mirror_loss_W"] + budget["output_mirror_loss_W"], rel=1e-13)
    assert analysis.execution.complete


def test_impedance_matching_cancels_the_reflection_on_resonance():
    """A lossless input mirror with R1 = R2 (1 - L): no reflection on resonance, and all of the light leaves through M2 or the losses."""

    loss, r2 = 0.004, 0.999
    r1 = r2 * (1.0 - loss)
    analysis = _analyse(cavity("500", "500", 300.0), {"power_w": POWER, "beam": "EIGENMODE", "detuning_Hz": 0.0},
                        reflectances={"M1": r1, "M2": r2}, transmittances={"M1": 1.0 - r1, "M2": 1.0 - r2}, round_trip_loss=loss)
    assert _value(analysis, "impedance_matching_reflectance") == pytest.approx(r1, rel=1e-15)
    assert _value(analysis, "reflected_power_W") == pytest.approx(0.0, abs=1e-14 * POWER)
    assert _value(analysis, "resonant_reflection") == pytest.approx(0.0, abs=1e-14)
    a = math.sqrt(r1 * r2 * (1.0 - loss))
    assert _value(analysis, "transmission") == pytest.approx((1.0 - r1) * (1.0 - r2) * math.sqrt(1.0 - loss) / (1.0 - a) ** 2, rel=1e-12)
    # the stated loss enters the finesse, which is now a value, not a bound
    assert _value(analysis, "finesse") == pytest.approx(math.pi * math.sqrt(a) / (1.0 - a), rel=1e-12)
    # under-coupled (R1 below the match) and over-coupled (above) both reflect
    for other in (r1 - 0.01, r1 + 0.0009):
        off = _analyse(cavity("500", "500", 300.0), {"power_w": POWER, "beam": "EIGENMODE", "detuning_Hz": 0.0},
                       reflectances={"M1": other, "M2": r2}, transmittances={"M1": 1.0 - other, "M2": 1.0 - r2}, round_trip_loss=loss)
        assert _value(off, "reflection") == pytest.approx(_airy(other, 1.0 - other, r2, 1.0 - r2, math.sqrt(1.0 - loss), 0.0)["reflected"], rel=1e-9)
        assert _value(off, "reflection") > 1e-4


def test_the_detuning_curve_is_the_airy_function_and_falls_to_half_at_half_the_linewidth():
    r1, r2, t1, t2 = 0.98, 0.99, 0.019, 0.009
    analysis = _analyse(cavity("500", "500", 300.0), {"power_w": POWER, "beam": "EIGENMODE", "detuning_Hz": 0.0,
                                                      "scan": {"from_Hz": -2.0e8, "to_Hz": 2.0e8, "points": 41}},
                        reflectances={"M1": r1, "M2": r2}, transmittances={"M1": t1, "M2": t2})
    fsr = C_MM_PER_S / 600.0
    scan = analysis.arrays["injection"]["scan"]
    nu = np.array(scan["detuning_Hz"])
    assert nu == pytest.approx(np.linspace(-2.0e8, 2.0e8, 41), rel=1e-15)
    a = math.sqrt(r1 * r2)
    peak = t1 * t2 / (1.0 - a) ** 2
    coefficient = 4.0 * a / (1.0 - a) ** 2
    expected = POWER * peak / (1.0 + coefficient * np.sin(np.pi * nu / fsr) ** 2)
    assert np.array(scan["transmitted_W"]) == pytest.approx(expected, rel=1e-11)
    reflected = [POWER * _airy(r1, t1, r2, t2, 1.0, 2.0 * math.pi * f / fsr)["reflected"] for f in nu]
    assert np.array(scan["reflected_W"]) == pytest.approx(reflected, rel=1e-10)
    half = _value(analysis, "linewidth_fwhm_Hz") / 2.0
    at_half = _analyse(cavity("500", "500", 300.0), {"power_w": POWER, "beam": "EIGENMODE", "detuning_Hz": half},
                       reflectances={"M1": r1, "M2": r2}, transmittances={"M1": t1, "M2": t2})
    assert _value(at_half, "transmitted_power_W") == pytest.approx(POWER * peak / 2.0, rel=1e-10)


@pytest.mark.parametrize("offset_nm", [LAMBDA_NM / 4.0, LAMBDA_NM / 8.0, -LAMBDA_NM / 16.0, 3.0 * LAMBDA_NM / 2.0 + 10.0])
def test_a_length_offset_from_the_resonance_sets_the_round_trip_phase(offset_nm):
    """The cavity length less that of the TEM00 resonance: the round trip changes by twice the offset, a phase of 2 pi 2 offset / lambda
    (reduced to [-pi, pi]); the detuning it equals is that phase's share of the free spectral range."""

    analysis = _analyse(cavity("500", "500", 300.0), {"power_w": POWER, "beam": "EIGENMODE", "length_offset_nm": offset_nm},
                        reflectances={"M1": 0.99, "M2": 0.99}, transmittances={"M1": 0.01, "M2": 0.01})
    waves = 2 * Fraction(offset_nm) / Fraction(LAMBDA_NM)
    delta = 2.0 * math.pi * float(waves - round(waves))
    record = analysis.arrays["injection"]
    assert record["detuning_from"] == "LENGTH_OFFSET" and record["round_trip_phase_rad"] == pytest.approx(delta, abs=1e-12)
    fsr = C_MM_PER_S / 600.0
    assert _value(analysis, "detuning_Hz") == pytest.approx(delta * fsr / (2.0 * math.pi), rel=1e-12, abs=1e-6)
    assert _value(analysis, "transmission") == pytest.approx(_airy(0.99, 0.01, 0.99, 0.01, 1.0, delta)["transmitted"], rel=1e-9)
    if offset_nm == LAMBDA_NM / 4.0:  # half a wave per round trip: half-way between resonances, (1 - a)^2 / (1 + a)^2 of the peak
        assert _value(analysis, "transmission") == pytest.approx(0.01 * 0.01 / (1.0 + 0.99) ** 2, rel=1e-12)


# ------------------------------------------------------------------------------------------------ transverse modes


def _hg(n: int, x: np.ndarray, q: complex) -> np.ndarray:
    w = math.sqrt(-LAMBDA_MM / (math.pi * (1.0 / q).imag))
    coefficients = np.zeros(n + 1)
    coefficients[n] = 1.0
    norm = (2.0 / math.pi) ** 0.25 / math.sqrt(2.0 ** n * math.factorial(n) * w)
    return norm * np.polynomial.hermite.hermval(math.sqrt(2.0) * x / w, coefficients) * np.exp(-1j * math.pi / LAMBDA_MM * x * x / q)


def _overlaps(q_in: complex, q_mode: complex, orders: int) -> list[float]:
    """|integral conj(HG_n of the mode) x input|^2 by the trapezoidal rule on a fine grid."""

    x = np.linspace(-6.0, 6.0, 120001)
    beam = _hg(0, x, q_in)
    assert np.trapezoid(np.abs(beam) ** 2, x) == pytest.approx(1.0, rel=1e-10)
    return [abs(np.trapezoid(np.conj(_hg(n, x, q_mode)) * beam, x)) ** 2 for n in range(orders + 1)]


@pytest.mark.parametrize("scale, shift", [(1.3, 0.0), (0.6, 40.0), (1.0, -70.0)], ids=["waist", "waist-and-position", "position"])
def test_the_hermite_gauss_content_of_a_mismatched_beam_equals_the_overlap_integrals(scale, shift):
    mode = _g_mode(500.0, 500.0, 300.0)
    waist = mode["w0"] * scale
    q_in = complex(-(mode["z1"] + shift), math.pi * waist * waist / LAMBDA_MM)
    numeric = _overlaps(q_in, mode["q"], 8)
    assert max(numeric[1::2]) < 1e-12  # a coaxial beam: no odd order
    closed = one_section_fractions(numeric[0], 8)
    assert closed == pytest.approx(numeric, abs=1e-10)
    analysis = _analyse(cavity("500", "500", 300.0), {"power_w": POWER, "beam": "INPUT_BEAM", "detuning_Hz": 0.0, "mode_order_max": 8},
                        reflectances={"M1": 0.99, "M2": 0.99}, transmittances={"M1": 0.01, "M2": 0.01},
                        input_beam={"waist_radius_mm": waist, "waist_after_reference_mm": mode["z1"] + shift})
    rows = {(r["m"], r["n"]): r["power_fraction"] for r in analysis.arrays["injection"]["modes"]}
    for (m, n), fraction in rows.items():
        assert fraction == pytest.approx(numeric[m] * numeric[n], abs=1e-10)
    assert _value(analysis, "mode_matching_efficiency") == pytest.approx(numeric[0] ** 2, rel=1e-9)
    if shift == 0.0:  # the closed form of a waist mismatch at the waist
        assert numeric[0] == pytest.approx(2.0 * mode["w0"] * waist / (mode["w0"] ** 2 + waist ** 2), rel=1e-10)


def test_the_laguerre_gauss_content_of_a_round_beam_equals_the_radial_overlap_integrals():
    mode = _g_mode(500.0, 500.0, 300.0)
    waist = mode["w0"] * 1.5
    q_in = complex(-mode["z1"] + 25.0, math.pi * waist * waist / LAMBDA_MM)
    r = np.linspace(0.0, 4.0, 800001)

    def lg(p: int, q: complex) -> np.ndarray:
        w = math.sqrt(-LAMBDA_MM / (math.pi * (1.0 / q).imag))
        c = np.zeros(p + 1)
        c[p] = 1.0
        return math.sqrt(2.0 / math.pi) / w * np.polynomial.laguerre.lagval(2.0 * r * r / (w * w), c) * np.exp(-1j * math.pi / LAMBDA_MM * r * r / q)

    beam = lg(0, q_in)
    numeric = [abs(np.trapezoid(np.conj(lg(p, mode["q"])) * beam * 2.0 * np.pi * r, r)) ** 2 for p in range(5)]
    analysis = _analyse(cavity("500", "500", 300.0), {"power_w": POWER, "beam": "INPUT_BEAM", "detuning_Hz": 0.0, "mode_order_max": 8},
                        reflectances={"M1": 0.99, "M2": 0.99}, transmittances={"M1": 0.01, "M2": 0.01},
                        input_beam={"waist_radius_mm": waist, "waist_after_reference_mm": mode["z1"] - 25.0})
    rows = analysis.arrays["injection"]["lg_modes"]
    assert [row["power_fraction"] for row in rows] == pytest.approx(numeric, abs=5e-10)
    assert lg_fractions(numeric[0], 4) == pytest.approx(numeric, abs=5e-10)
    # LG_p0 is the degenerate HG group of order 2p of a round beam
    hg = analysis.arrays["injection"]["modes"]
    for row in rows:
        assert sum(m["power_fraction"] for m in hg if m["order"] == row["order"]) == pytest.approx(row["power_fraction"], rel=1e-12)


def test_each_mode_resonates_at_its_gouy_shifted_frequency():
    mode = _g_mode(500.0, 500.0, 300.0)
    fsr = C_MM_PER_S / 600.0
    offset = (2.0 * mode["gouy"]) % (2.0 * math.pi) * fsr / (2.0 * math.pi)
    request = {"reflectances": {"M1": 0.99, "M2": 0.99}, "transmittances": {"M1": 0.01, "M2": 0.01},
               "input_beam": {"waist_radius_mm": mode["w0"] * 1.4, "waist_after_reference_mm": mode["z1"]}}
    analysis = _analyse(cavity("500", "500", 300.0), {"power_w": POWER, "beam": "INPUT_BEAM", "detuning_Hz": offset, "mode_order_max": 6}, **request)
    peak = 0.01 * 0.01 / (1.0 - 0.99) ** 2
    for row in analysis.arrays["injection"]["modes"]:
        expected = (row["m"] * mode["gouy"] + row["n"] * mode["gouy"]) % (2.0 * math.pi) * fsr / (2.0 * math.pi)
        assert row["offset_from_tem00_Hz"] == pytest.approx(expected, rel=1e-12)
        if row["order"] == 2:  # on resonance at this detuning
            assert row["transmitted_W"] == pytest.approx(POWER * row["power_fraction"] * peak, rel=1e-9)
        else:
            delta = 2.0 * math.pi * offset / fsr - row["order"] * mode["gouy"]
            assert row["transmitted_W"] == pytest.approx(POWER * row["power_fraction"] * _airy(0.99, 0.01, 0.99, 0.01, 1.0, delta)["transmitted"], rel=1e-9)


def test_a_confocal_cavity_transmits_every_even_order_together():
    """g1 = g2 = 0: the round-trip Gouy phase is pi, so every even order resonates with TEM00 -- the transmission of a mismatched
    beam is the mode-matched one, less what is not resolved. Resolved to order 30 that is under 1e-12 of the input, below the
    negligible level (1e-9): the value is KNOWN, with the proven interval beside it and the most the light left out can add as its
    resolution. Resolved to order 2 it is 2e-3 of the input: the value is the resolved modes' sum, a lower bound of a BOUNDED quantity."""

    mode = _g_mode(100.0, 100.0, 100.0)
    analysis = _analyse(cavity("100", "100", 100.0), {"power_w": POWER, "beam": "INPUT_BEAM", "detuning_Hz": 0.0, "mode_order_max": 30},
                        reflectances={"M1": 0.99, "M2": 0.99}, transmittances={"M1": 0.009, "M2": 0.009},
                        input_beam={"waist_radius_mm": mode["w0"] * 1.25, "waist_after_reference_mm": mode["z1"]})
    assert _value(analysis, "mode_matching_efficiency") < 0.96
    peak = 0.009 * 0.009 / (1.0 - 0.99) ** 2
    transmitted = analysis.quantity("transmitted_power_W")
    unresolved = _value(analysis, "unresolved_mode_power_fraction")
    assert transmitted.state.value == "KNOWN" and 0.0 < unresolved < 1e-12 and transmitted.value_basis is None
    exact = POWER * peak
    assert transmitted.interval.kind.value == "PROVEN" and transmitted.interval.lower == pytest.approx(exact, rel=1e-11)
    assert transmitted.interval.lower <= exact * (1.0 + 1e-14) and exact <= transmitted.interval.upper * (1.0 + 1e-14)  # (within rounding)
    resolution = transmitted.convergence.resolution
    assert abs(transmitted.value - exact) <= resolution + 1e-14 * exact and resolution == pytest.approx(POWER * unresolved * peak, rel=1e-6)
    coarse = _analyse(cavity("100", "100", 100.0), {"power_w": POWER, "beam": "INPUT_BEAM", "detuning_Hz": 0.0, "mode_order_max": 2},
                      reflectances={"M1": 0.99, "M2": 0.99}, transmittances={"M1": 0.009, "M2": 0.009},
                      input_beam={"waist_radius_mm": mode["w0"] * 1.25, "waist_after_reference_mm": mode["z1"]})
    left = _value(coarse, "unresolved_mode_power_fraction")
    bounded = coarse.quantity("transmitted_power_W")
    assert left > 1e-3 and bounded.state.value == "BOUNDED" and bounded.value_basis.kind.value == "LOWER_BOUND"
    (omitted,) = bounded.value_basis.omitted
    assert omitted.part.value == "UNRESOLVED_MODES" and omitted.upper == pytest.approx(POWER * left * peak, rel=1e-9)
    assert bounded.value == pytest.approx(exact * (1.0 - left), rel=1e-9)  # (every resolved even order on resonance)
    assert bounded.value <= exact <= bounded.value + omitted.upper * (1.0 + 1e-12) and bounded.interval.upper == pytest.approx(exact, rel=1e-12)


def test_light_in_modes_above_the_resolved_order_is_bounded_and_the_bounds_hold():
    mode = _g_mode(500.0, 500.0, 300.0)
    request = {"reflectances": {"M1": 0.99, "M2": 0.995}, "transmittances": {"M1": 0.009, "M2": 0.004},
               "input_beam": {"waist_radius_mm": mode["w0"] * 2.0, "waist_after_reference_mm": mode["z1"] + 50.0}}
    coarse = _analyse(cavity("500", "500", 300.0), {"power_w": POWER, "beam": "INPUT_BEAM", "detuning_Hz": 3.0e7, "mode_order_max": 2}, **request)
    fine = _analyse(cavity("500", "500", 300.0), {"power_w": POWER, "beam": "INPUT_BEAM", "detuning_Hz": 3.0e7, "mode_order_max": 60}, **request)
    assert _value(coarse, "unresolved_mode_power_fraction") > 0.1 and _value(fine, "unresolved_mode_power_fraction") < 0.02
    for quantity_id in ("circulating_power_W", "transmitted_power_W", "reflected_power_W", "lost_power_W"):
        wide, narrow = coarse.quantity(quantity_id), fine.quantity(quantity_id)
        assert wide.state.value == "BOUNDED" and wide.interval.kind.value == "PROVEN"
        assert wide.interval.lower <= narrow.interval.lower and narrow.interval.upper <= wide.interval.upper + 1e-15
        # the value beside the interval: the resolved modes' sum, a lower bound (the light above the order only adds), no higher than
        # the finer resolution's and inside both intervals' upper ends
        assert wide.value_basis.kind.value == "LOWER_BOUND" and [o.part.value for o in wide.value_basis.omitted] == ["UNRESOLVED_MODES"]
        assert wide.value <= narrow.value * (1.0 + 1e-12) and wide.value <= wide.interval.upper and narrow.value <= narrow.interval.upper
        assert wide.value + wide.value_basis.omitted[0].upper == pytest.approx(wide.interval.upper, rel=1e-12)
    # the transmitted value of the coarse resolution, written out: HG_00, HG_20 and HG_02 (per section eta, eta (1 - eta^2) / 2), each
    # at its Gouy-shifted phase, as complex field sums
    q_in = complex(-(mode["z1"] + 50.0), math.pi * (2.0 * mode["w0"]) ** 2 / LAMBDA_MM)
    eta = 2.0 * math.sqrt(q_in.imag * mode["q"].imag) / abs(q_in - mode["q"].conjugate())
    delta = 2.0 * math.pi * 3.0e7 / (C_MM_PER_S / 600.0)
    listed = [(0, eta * eta), (2, 2.0 * eta * eta * (1.0 - eta * eta) / 2.0)]
    expected = POWER * sum(f * _airy(0.99, 0.009, 0.995, 0.004, 1.0, delta - order * mode["gouy"])["transmitted"] for order, f in listed)
    assert coarse.quantity("transmitted_power_W").value == pytest.approx(expected, rel=1e-9)
    budget = coarse.arrays["injection"]["power_budget"]
    assert budget["not_resolved_W"] == pytest.approx(POWER * _value(coarse, "unresolved_mode_power_fraction"), rel=1e-12)
    assert abs(budget["closure_residual_W"]) < 1e-13


def test_the_aperture_limits_the_resolved_order():
    mode = _g_mode(500.0, 500.0, 300.0)
    w1 = mode["w0"] * math.sqrt(1.0 + (150.0 / (math.pi * mode["w0"] ** 2 / LAMBDA_MM)) ** 2)
    clear = 3.0 * w1 * math.sqrt(4.5)  # orders up to N = 3 clear 3 sqrt(N + 1) mode radii
    analysis = _analyse(cavity("500", "500", 300.0, clear=f"{clear:.6f}"), {"power_w": POWER, "beam": "INPUT_BEAM", "detuning_Hz": 0.0},
                        reflectances={"M1": 0.99, "M2": 0.99}, transmittances={"M1": 0.01, "M2": 0.01},
                        input_beam={"waist_radius_mm": mode["w0"] * 1.5, "waist_after_reference_mm": mode["z1"]})
    record = analysis.arrays["injection"]
    assert record["order_limit"] == 3 and record["resolved_order"] == 3 and "clear aperture" in record["order_limited_by"]
    assert max(row["order"] for row in record["modes"]) == 2


# ------------------------------------------------------------------------------------------------ model scope and inputs


def test_without_stated_losses_of_an_intracavity_window_the_powers_are_bounded_for_any_loss():
    window = declared_component("L", "window", "window", (150.0, 0.0, 0.0), ALONG_X)
    model = {"model_id": "window", "model": sequence([surface(1, "0"), surface(2, "5")], ["N-BK7"])}
    doc = cavity("500", "500", 300.0, extra=(window,), models=(model,), materials=[nbk7()])
    request = {"components": ["M1", "L", "M2"], "wavelength_nm": LAMBDA_NM, "reflectances": {"M1": 0.99, "M2": 0.99},
               "transmittances": {"M1": 0.01, "M2": 0.01}}
    open_ = cavity_analysis(doc, {**request, "injection": {"power_w": POWER, "beam": "EIGENMODE", "detuning_Hz": 0.0}})
    circulating = open_.quantity("circulating_power_W")
    assert circulating.state.value == "BOUNDED" and "round_trip_loss" in circulating.message
    a = 0.99
    assert circulating.interval.lower == pytest.approx(POWER * 0.01 / (1.0 + a) ** 2) and circulating.interval.upper == pytest.approx(POWER * 0.01 / (1.0 - a) ** 2)
    # the value beside the interval is the power without the window's loss: an estimate that names the loss taken as zero
    assert circulating.value == pytest.approx(POWER * 0.01 / (1.0 - a) ** 2, rel=1e-12) and circulating.value_basis.kind.value == "ESTIMATE"
    assert [(x.input.value, x.at) for x in circulating.value_basis.assumed] == [("NO_UNSTATED_LOSS", ("L",))]
    assert circulating.missing == ("round_trip_loss",)
    # on resonance, a mode-matched beam: the values without the window's loss, bounded for any loss. The build-up, the transmission and
    # the matching reflectances fall with loss (upper bounds); a critically coupled cavity's reflection (0 without loss) rises to R1.
    reference = [_airy(0.99, 0.01, 0.99, 0.01, math.sqrt(1.0 - loss), 0.0) for loss in np.linspace(0.0, 1.0, 2001)]
    for quantity_id, channel, kind, least in (("resonant_power_build_up", "circulating", "UPPER_BOUND", 0.01),
                                              ("resonant_transmission", "transmitted", "UPPER_BOUND", 0.0),
                                              ("resonant_reflection", "reflected", "LOWER_BOUND", 0.0)):
        quantity = open_.quantity(quantity_id)
        assert quantity.state.value == "BOUNDED" and quantity.reason == "LOSSES_NOT_MODELED" and quantity.missing == ("round_trip_loss",)
        assert quantity.value == pytest.approx(reference[0][channel], rel=1e-9, abs=1e-12) and quantity.value_basis.kind.value == kind
        assert [(x.input.value, x.at) for x in quantity.value_basis.assumed] == [("NO_UNSTATED_LOSS", ("L",))]
        values = [row[channel] for row in reference]
        assert quantity.interval.kind.value == "PROVEN"
        assert quantity.interval.lower == pytest.approx(min(values), abs=1e-12) and quantity.interval.upper == pytest.approx(max(values), rel=1e-12)
        assert quantity.interval.lower == pytest.approx(least, abs=1e-12)
    matching = open_.quantity("impedance_matching_reflectance")
    assert matching.value == matching.interval.upper == pytest.approx(0.99, rel=1e-12) and matching.value_basis.kind.value == "UPPER_BOUND"
    stated = cavity_analysis(doc, {**request, "round_trip_loss": 0.002, "injection": {"power_w": POWER, "beam": "EIGENMODE", "detuning_Hz": 0.0}})
    a = math.sqrt(0.99 * 0.99 * 0.998)
    assert _value(stated, "circulating_power_W") == pytest.approx(POWER * 0.01 / (1.0 - a) ** 2, rel=1e-12)
    assert stated.arrays["injection"]["power_budget"]["inside_loss_W"] > 0.0
    for quantity_id in ("resonant_power_build_up", "resonant_transmission", "resonant_reflection", "impedance_matching_reflectance",
                        "input_loss_matching_reflectance", "circulating_power_W", "transmitted_power_W", "reflected_power_W"):
        interval = open_.quantity(quantity_id).interval  # (the stated loss's values lie in the intervals that hold for any loss)
        assert interval.lower <= _value(stated, quantity_id) <= interval.upper, quantity_id


@pytest.mark.parametrize("r1, t1, r2, kind", [(0.9, 0.1, 0.99, "ESTIMATE"), (0.995, 0.004, 0.9, "LOWER_BOUND"), (0.9, 0.08, 0.99, "ESTIMATE")],
                         ids=["over-coupled", "under-coupled", "over-coupled-lossy-input"])
def test_the_resonant_reflection_without_stated_losses_is_bounded_for_any_loss(r1, t1, r2, kind):
    """The field reflected on resonance, (r1 - (1 - A1) b) / (1 - r1 b), falls monotonically as the returned field b grows: an
    under-coupled cavity's reflection without loss is its least (a lower bound, up to R1 when nothing returns); an over-coupled
    cavity's passes through zero as loss is added, so its value without loss is an estimate inside [0, max(R1, that value)]. The
    interval's ends are the extremes of the complex field sum over every loss of one pass."""

    window = declared_component("L", "window", "window", (150.0, 0.0, 0.0), ALONG_X)
    model = {"model_id": "window", "model": sequence([surface(1, "0"), surface(2, "5")], ["N-BK7"])}
    doc = cavity("500", "500", 300.0, extra=(window,), models=(model,), materials=[nbk7()])
    analysis = cavity_analysis(doc, {"components": ["M1", "L", "M2"], "wavelength_nm": LAMBDA_NM, "reflectances": {"M1": r1, "M2": r2},
                                     "transmittances": {"M1": t1, "M2": 1.0 - r2},
                                     "injection": {"power_w": POWER, "beam": "EIGENMODE", "detuning_Hz": 0.0}})
    values = [_airy(r1, t1, r2, 1.0 - r2, passing, 0.0)["reflected"] for passing in np.linspace(0.0, 1.0, 20001)]
    reflection = analysis.quantity("resonant_reflection")
    assert reflection.state.value == "BOUNDED" and reflection.value_basis.kind.value == kind
    assert reflection.value == pytest.approx(values[-1], rel=1e-9)
    assert reflection.interval.lower == pytest.approx(min(values), abs=1e-6) and reflection.interval.upper == pytest.approx(max(values), rel=1e-9)
    assert reflection.interval.lower <= min(values) * (1.0 + 1e-12) + 1e-15 and max(values) <= reflection.interval.upper * (1.0 + 1e-12)


def test_a_cavity_whose_apertures_clip_its_mode_has_its_injected_powers_bounded_for_any_loss():
    """Mirrors of 0.6 mm clear radius where the R500 / L300 mode is 0.34 mm wide (1.8 mode radii): the edges add a loss nothing
    states, and the light the cavity takes is not its unclipped mode. The quantities of the mirrors on resonance are bounds for any
    loss (the build-up and the transmission upper bounds at the unclipped values); the seven powers are bounded from zero, their
    intervals holding the unclipped Airy value, and no power that leaves or is lost exceeds the injected power. A stated
    round_trip_loss states the elements' loss, not the clipping: the bounds stay."""

    mirrors = {"reflectances": {"M1": 0.99, "M2": 0.99}, "transmittances": {"M1": 0.009, "M2": 0.009}}
    injection = {"power_w": POWER, "beam": "EIGENMODE", "detuning_Hz": 0.0}
    clear = _analyse(cavity("500", "500", 300.0), injection, **mirrors)
    clipped = _analyse(cavity("500", "500", 300.0, clear="0.6"), injection, **mirrors)
    assert clipped.quantity("mode_radius_mm.M1").reason == "MODE_CLIPPED_BY_APERTURE"
    assert clipped.quantity("mode_radius_mm.M1").value == _value(clear, "mode_radius_mm.M1")
    for quantity_id in ("resonant_power_build_up", "resonant_transmission"):
        quantity = clipped.quantity(quantity_id)
        assert quantity.state.value == "BOUNDED" and quantity.value == _value(clear, quantity_id) == quantity.interval.upper
        assert quantity.value_basis.kind.value == "UPPER_BOUND" and quantity.missing == ()
        assert [(x.input.value, x.at, x.code) for x in quantity.value_basis.assumed] == [("NO_UNSTATED_LOSS", ("M1", "M2"), "MODE_CLIPPED_BY_APERTURE")]
    for quantity_id in ("circulating_power_W", "transmitted_power_W", "reflected_power_W", "lost_power_W"):
        quantity = clipped.quantity(quantity_id)
        assert quantity.state.value == "BOUNDED" and quantity.reason == "MODE_CLIPPED_BY_APERTURE" and quantity.value is None
        assert quantity.interval.lower == 0.0 <= _value(clear, quantity_id) <= quantity.interval.upper * (1.0 + 1e-12), quantity_id
        if quantity_id != "circulating_power_W":
            assert quantity.interval.upper <= POWER
    assert _value(clipped, "unresolved_mode_power_fraction") == 1.0 and clipped.arrays["injection"]["order_limit"] == -1
    stated = _analyse(cavity("500", "500", 300.0, clear="0.6"), injection, round_trip_loss=0.001, **mirrors)
    assert stated.quantity("resonant_power_build_up").state.value == "BOUNDED" and stated.quantity("finesse").state.value == "BOUNDED"
    a = math.sqrt(0.99 * 0.99 * 0.999)
    assert stated.quantity("resonant_power_build_up").value == pytest.approx(0.009 / (1.0 - a) ** 2, rel=1e-12)


def test_an_unstable_cavity_has_no_injected_build_up():
    analysis = _analyse(cavity("50", "50", 120.0), {"power_w": POWER, "beam": "EIGENMODE", "detuning_Hz": 0.0},
                        reflectances={"M1": 0.99, "M2": 0.99}, transmittances={"M1": 0.01, "M2": 0.01})
    for quantity_id in ("circulating_power_W", "transmitted_power_W", "resonant_power_build_up"):
        quantity = analysis.quantity(quantity_id)
        assert quantity.state.value == "OUT_OF_MODEL_SCOPE" and quantity.reason == "CAVITY_UNSTABLE"
    assert _value(analysis, "injected_power_W") == POWER and analysis.execution.complete


def test_the_transmittances_come_from_the_coatings_and_a_stated_reflectance_needs_its_transmittance():
    analysis = _analyse(cavity("500", "500", 300.0), {"power_w": POWER, "beam": "EIGENMODE", "detuning_Hz": 0.0})
    assert _value(analysis, "transmittance.M1") == pytest.approx(0.009) and _value(analysis, "transmittance.M2") == pytest.approx(0.009)
    enhancement = 1.0 / (1.0 - 0.99) ** 2
    assert _value(analysis, "transmission") == pytest.approx(0.009 * 0.009 * enhancement, rel=1e-12)
    half = _analyse(cavity("500", "500", 300.0), {"power_w": POWER, "beam": "EIGENMODE", "detuning_Hz": 0.0}, reflectances={"M1": 0.98})
    assert half.quantity("transmittance.M1").reason == "TRANSMITTANCE_NOT_STATED" and half.quantity("circulating_power_W").state.value == "UNKNOWN"
    # the stated R = 0.98 leaves 0 <= T <= 0.02: bounded, no value invented, the request field named
    unstated = half.quantity("transmittance.M1")
    assert unstated.state.value == "BOUNDED" and unstated.value is None and unstated.missing == ("transmittances.M1",)
    assert unstated.interval.kind.value == "PROVEN" and (unstated.interval.lower, unstated.interval.upper) == pytest.approx((0.0, 0.02), abs=1e-15)
    # the missing transmittance withholds only its readers, which name the request field; what does not read it stands
    for quantity_id in ("circulating_power_W", "transmitted_power_W", "resonant_power_build_up", "resonant_reflection", "input_loss_matching_reflectance"):
        quantity = half.quantity(quantity_id)
        assert quantity.state.value == "UNKNOWN" and quantity.reason == "TRANSMITTANCE_NOT_STATED" and quantity.missing == ("transmittances.M1",)
    assert _value(half, "impedance_matching_reflectance") == pytest.approx(0.99, rel=1e-15) and _value(half, "detuning_Hz") == 0.0
    assert _value(half, "unresolved_mode_power_fraction") == 0.0 and [row["order"] for row in half.arrays["injection"]["modes"]] == [0]
    opaque = _analyse(cavity("500", "500", 300.0, coating=OPAQUE), {"power_w": POWER, "beam": "EIGENMODE", "detuning_Hz": 0.0})
    assert _value(opaque, "circulating_power_W") == 0.0 and _value(opaque, "reflected_power_W") == pytest.approx(0.99 * POWER)


@pytest.mark.parametrize("change, code", [
    ({"injection": {"beam": "EIGENMODE"}}, "PARAXIAL_REQUEST_INVALID"),
    ({"injection": {"power_w": 1.0, "beam": "INPUT_BEAM"}}, "PARAXIAL_REQUEST_INVALID"),
    ({"injection": {"power_w": 1.0, "beam": "EIGENMODE", "detuning_Hz": 0.0, "length_offset_nm": 1.0}}, "PARAXIAL_REQUEST_INVALID"),
    ({"injection": {"power_w": 1.0, "beam": "EIGENMODE", "mode_order_max": 61}}, "PARAXIAL_REQUEST_INVALID"),
    ({"transmittances": {"M1": 0.5}, "reflectances": {"M1": 0.6}}, "PARAXIAL_REQUEST_INVALID"),
    ({"round_trip_loss": 1.0}, "PARAXIAL_REQUEST_INVALID"),
    ({"targets": [{"target_id": "t", "quantity_id": "transmission"}]}, "PARAXIAL_REQUEST_INVALID"),
], ids=["no-power", "input-beam-missing", "two-detunings", "order", "r-plus-t", "loss", "target-without-threshold"])
def test_requests_the_injection_cannot_answer_are_refused(change, code):
    with pytest.raises(CavityError) as caught:
        cavity_analysis(cavity("500", "500", 300.0), {"components": ["M1", "M2"], "wavelength_nm": LAMBDA_NM, **change})
    assert caught.value.problems[0]["code"] == code


# ------------------------------------------------------------------------------------------------ the source of the Project


def _fed(position=(-80.0, 0.0, 0.0), **kwargs) -> dict:
    return cavity("500", "500", 300.0, extra=(source("S", position, wavelength_nm=LAMBDA_NM, waist_mm=0.3, power_w=POWER, **kwargs),))


def _matched(position=(-80.0, 0.0, 0.0), **kwargs) -> dict:
    """A source whose beam is the R500 / L300 cavity's eigenmode: its waist (w0 of the g-parameters) at the cavity's centre."""

    mode = _g_mode(500.0, 500.0, 300.0)
    return cavity("500", "500", 300.0, extra=(source("S", position, wavelength_nm=LAMBDA_NM, waist_mm=mode["w0"], power_w=POWER,
                                                     waist_position_mm=mode["z1"] - position[0], **kwargs),))


def test_a_source_behind_the_first_mirror_feeds_the_cavity_with_its_power_and_line():
    analysis = _analyse(_matched(), {"source_id": "S", "beam": "SOURCE", "detuning_Hz": 0.0})
    assert _value(analysis, "injected_power_W") == POWER and analysis.quantity("injected_power_W").source == "source S"
    assert _value(analysis, "mode_matching_efficiency") == pytest.approx(1.0, rel=1e-12)
    transmitted = analysis.quantity("transmitted_power_W")  # (a beam matched within rounding: a value, or bounds ~1e-16 apart)
    for bound in ((transmitted.value,) if transmitted.state.value == "KNOWN" else (transmitted.interval.lower, transmitted.interval.upper)):
        assert bound == pytest.approx(POWER * 0.009 * 0.009 / (1.0 - 0.99) ** 2, rel=1e-12)
    # a source that does not feed the cavity through its first mirror: the cavity's own quantities stand, the injection's say why
    plain = cavity_analysis(_fed(), {"components": ["M1", "M2"], "wavelength_nm": LAMBDA_NM})
    beside = _analyse(_fed((-80.0, 1.0, 0.0)), {"source_id": "S", "beam": "SOURCE"})  # (1 mm off the axis, behind the mirror)
    assert "1 mm off the cavity's axis" in beside.arrays["injection"]["source_problem"]["message"]
    assert _value(beside, "injected_power_W") == POWER  # (its light still arrives at the mirror: the trace carries its power)
    within = _analyse(_fed((80.0, 0.0, 0.0)), {"source_id": "S", "beam": "SOURCE"})  # (between the mirrors)
    assert within.quantity("injected_power_W").state.value == "OUT_OF_MODEL_SCOPE" and "inside the cavity" in within.quantity("injected_power_W").message
    assert within.arrays["injection"]["traced_arrival"]["not_arriving"] and within.arrays["injection"]["traced_arrival"]["arriving_W"] == 0.0
    for document, state in ((beside, "BOUNDED"), (within, "OUT_OF_MODEL_SCOPE")):
        assert document.arrays["injection"]["source_problem"]["code"] == "CAVITY_SOURCE_UNSUPPORTED"
        for quantity_id in ("circulating_power_W", "transmitted_power_W"):
            quantity = document.quantity(quantity_id)
            assert quantity.state.value == state and quantity.reason == "CAVITY_SOURCE_UNSUPPORTED" and quantity.value is None
        assert document.quantity("mode_matching_efficiency").state.value == "OUT_OF_MODEL_SCOPE"
        assert _value(document, "finesse") == _value(plain, "finesse") and _value(document, "mode_radius_mm.M1") == _value(plain, "mode_radius_mm.M1")
    with pytest.raises(CavityError) as caught:
        cavity_analysis(_fed(), {"components": ["M1", "M2"], "wavelength_nm": 1000.0, "injection": {"source_id": "S", "beam": "SOURCE"}})
    assert caught.value.problems[0]["location"] == "wavelength_nm"


def test_a_source_broader_than_the_cavity_linewidth_is_bounded_over_every_detuning():
    """A mode-matched source with a linewidth of 265 kHz against a cavity linewidth of 1.6 MHz (0.166 of it, above the 0.01 a single
    frequency may have): every spectral component sits at some detuning, so the time-averaged transmitted power lies between the Airy
    function's extremes. The reference averages the Airy function over a Lorentzian line of that width centred on the resonance
    (nu = (width / 2) tan(phi), uniform in phi): 0.858 of the single-frequency peak, inside the interval and far from its upper end,
    which a single-frequency answer would have reported. The quantities of the mirrors alone and the detuning stand; a request that
    raises the fraction above 0.166 takes the source as one frequency and says which setting did."""

    doc = _matched()
    doc["components"][-1]["source_model"]["linewidth_nm"] = 1e-6
    width = C_MM_PER_S * 1e-6 / (LAMBDA_NM * LAMBDA_NM * 1e-6)
    fsr = C_MM_PER_S / 600.0
    cavity_width = 2.0 * fsr / math.pi * math.asin((1.0 - 0.99) / (2.0 * math.sqrt(0.99)))
    peak = 0.009 * 0.009 / (1.0 - 0.99) ** 2
    phi = (np.arange(400000) + 0.5) / 400000 * math.pi - math.pi / 2.0
    airy = peak / (1.0 + 4.0 * 0.99 / (1.0 - 0.99) ** 2 * np.sin(math.pi * 0.5 * width * np.tan(phi) / fsr) ** 2)
    averaged = POWER * float(np.mean(airy))
    assert averaged == pytest.approx(POWER * peak * cavity_width / (cavity_width + width), rel=2e-3)  # (two Lorentzians: widths add)
    plain = _analyse(_matched(), {"source_id": "S", "beam": "SOURCE", "detuning_Hz": 0.0})
    broad = _analyse(doc, {"source_id": "S", "beam": "SOURCE", "detuning_Hz": 0.0})
    transmitted = broad.quantity("transmitted_power_W")
    assert transmitted.state.value == "BOUNDED" and transmitted.reason == "SOURCE_NOT_SINGLE_FREQUENCY" and transmitted.value is None
    assert transmitted.interval.kind.value == "PROVEN" and transmitted.missing == ()
    assert transmitted.interval.lower == pytest.approx(POWER * 0.009 * 0.009 / (1.0 + 0.99) ** 2, rel=1e-9)
    assert transmitted.interval.upper == pytest.approx(POWER * peak, rel=1e-9)
    assert transmitted.interval.lower < averaged < 0.9 * transmitted.interval.upper
    for quantity_id in ("circulating_power_W", "reflected_power_W", "lost_power_W"):
        quantity = broad.quantity(quantity_id)
        assert quantity.state.value == "BOUNDED" and 0.0 <= quantity.interval.lower <= quantity.interval.upper
    for quantity_id in ("resonant_power_build_up", "resonant_transmission", "resonant_reflection", "impedance_matching_reflectance",
                        "input_loss_matching_reflectance", "detuning_Hz"):
        assert broad.quantity(quantity_id) == plain.quantity(quantity_id), quantity_id
    assert broad.arrays["injection"]["coupling"] == plain.arrays["injection"]["coupling"] and "at_tem00_resonance" not in broad.arrays["injection"]
    (check,) = [c for c in broad.models[-1].validity if c.parameter == "source_linewidth_over_cavity_linewidth"]
    assert check.observed == pytest.approx(width / cavity_width, rel=1e-9) and check.maximum == 0.01 and check.within is False
    assert check.setting == "injection.source_linewidth_fraction_max" and check.code == "SOURCE_NOT_SINGLE_FREQUENCY"
    assert broad.settings["limits"][check.setting] == 0.01
    # the caller's fraction: 0.2 takes this source as one frequency (the value a single frequency gives), and the check names the setting
    accepted = _analyse(doc, {"source_id": "S", "beam": "SOURCE", "detuning_Hz": 0.0, "source_linewidth_fraction_max": 0.2})
    assert accepted.quantity("transmitted_power_W").value == plain.quantity("transmitted_power_W").value
    (check,) = [c for c in accepted.models[-1].validity if c.parameter == "source_linewidth_over_cavity_linewidth"]
    assert check.within is True and check.maximum == 0.2 and check.setting == "injection.source_linewidth_fraction_max"
    assert accepted.settings["limits"]["injection.source_linewidth_fraction_max"] == 0.2 and "source_linewidth_fraction_max" not in accepted.settings["injection"]
    doc["components"][-1]["source_model"]["linewidth_nm"] = 1e-9
    narrow = _analyse(doc, {"source_id": "S", "beam": "SOURCE", "detuning_Hz": 0.0})
    assert narrow.quantity("transmitted_power_W").value == plain.quantity("transmitted_power_W").value
    (check,) = [c for c in narrow.models[-1].validity if c.parameter == "source_linewidth_over_cavity_linewidth"]
    assert check.within is True and check.setting is None and check.code is None
    # a pulsed source: a comb of lines, the same bound
    del doc["components"][-1]["source_model"]["linewidth_nm"]
    doc["components"][-1]["source_model"]["pulse"] = {"repetition_rate_hz": 80000000.0, "pulse_duration_s": 1e-9, "duration_definition": "FWHM"}
    pulsed = _analyse(doc, {"source_id": "S", "beam": "SOURCE", "detuning_Hz": 0.0})
    assert pulsed.quantity("transmitted_power_W").interval == transmitted.interval and pulsed.quantity("transmitted_power_W").reason == "SOURCE_NOT_SINGLE_FREQUENCY"
    assert _value(pulsed, "resonant_transmission") == _value(plain, "resonant_transmission")


# ------------------------------------------------------------------------------------------------ targets, the run, the trace


def test_targets_are_assessed_on_the_quantities_and_their_thresholds_stay_out_of_the_physical_identity():
    targets = [{"target_id": "tp", "metric": "cavity_transmission", "minimum": 0.5, "unit": "1"},
               {"target_id": "bu", "metric": "cavity_power_build_up", "maximum": 10.0, "unit": "1"},
               {"target_id": "other", "metric": "cavity_finesse", "minimum": 1.0, "device_instance_id": "S", "unit": "1"}]
    doc = cavity("500", "500", 300.0, targets=targets, extra=(source("S", (-80.0, 0.0, 0.0), wavelength_nm=LAMBDA_NM),))
    request = {"components": ["M1", "M2"], "wavelength_nm": LAMBDA_NM, "injection": {"power_w": POWER, "beam": "EIGENMODE", "detuning_Hz": 0.0},
               "targets": [{"target_id": "r", "quantity_id": "reflected_power_W", "target_value": 0.0, "tolerance_abs": 0.05}]}
    analysis = cavity_analysis(doc, request)
    rows = {row["target_id"]: row for row in analysis.arrays["targets"]}
    assert rows["tp"]["assessment"] == "MET" and rows["tp"]["origin"] == "PROJECT"  # 0.81 transmission
    assert rows["bu"]["assessment"] == "UNMET"  # build-up 90
    assert rows["other"]["assessment"] == "NOT_APPLICABLE"
    assert rows["r"]["assessment"] == "MET" and rows["r"]["origin"] == "REQUEST"  # 0.0198 W (0.0099 of the light)
    stricter = cavity_analysis(doc, {**request, "targets": [{"target_id": "r", "quantity_id": "reflected_power_W", "target_value": 0.0, "tolerance_abs": 1e-9}]})
    assert stricter.identity.physical == analysis.identity.physical and stricter.identity.requirements != analysis.identity.requirements
    assert next(r for r in stricter.arrays["targets"] if r["target_id"] == "r")["assessment"] == "UNMET"
    bounded = cavity_analysis(doc, {**request, "input_beam": {"waist_radius_mm": 0.5, "waist_after_reference_mm": 150.0},
                                    "injection": {"power_w": POWER, "beam": "INPUT_BEAM", "detuning_Hz": 0.0, "mode_order_max": 0},
                                    "targets": [{"target_id": "c", "quantity_id": "circulating_power_W", "minimum": 140.0}]})
    row = next(r for r in bounded.arrays["targets"] if r["target_id"] == "c")
    assert row["assessment"] == "INDETERMINATE" and row["reason"] == "INTERVAL_STRADDLES_THRESHOLD"
    assert row["value"] == bounded.quantity("circulating_power_W").value and row["value_basis"]["kind"] == "LOWER_BOUND"


def test_a_bounded_quantity_s_value_decides_no_target_its_interval_does():
    """A window between R = 0.99 mirrors, its loss not stated: the finesse value 312.6 is the mirrors' (an upper bound). A minimum of
    300, between the true finesse (whatever the window's loss makes it) and that value, is not met by the value: the interval
    (no lower end) straddles it. A maximum above the bound is met and a minimum above it unmet, by the interval."""

    window = declared_component("L", "window", "window", (150.0, 0.0, 0.0), ALONG_X)
    model = {"model_id": "window", "model": sequence([surface(1, "0"), surface(2, "5")], ["N-BK7"])}
    doc = cavity("500", "500", 300.0, extra=(window,), models=(model,), materials=[nbk7()])
    analysis = cavity_analysis(doc, {"components": ["M1", "L", "M2"], "wavelength_nm": LAMBDA_NM, "reflectances": {"M1": 0.99, "M2": 0.99},
                                     "targets": [{"target_id": "between", "quantity_id": "finesse", "minimum": 300.0},
                                                 {"target_id": "above", "quantity_id": "finesse", "maximum": 400.0},
                                                 {"target_id": "beyond", "quantity_id": "finesse", "minimum": 320.0},
                                                 {"target_id": "width", "quantity_id": "linewidth_fwhm_Hz", "maximum": 1.0e9}]})
    mirrors = math.pi * math.sqrt(0.99) / 0.01
    rows = {row["target_id"]: row for row in analysis.arrays["targets"]}
    assert rows["between"]["value"] == pytest.approx(mirrors, rel=1e-12) and rows["between"]["value"] > 300.0
    assert rows["between"]["assessment"] == "INDETERMINATE" and rows["between"]["reason"] == "INTERVAL_STRADDLES_THRESHOLD"
    assert rows["between"]["value_basis"]["kind"] == "UPPER_BOUND" and rows["between"]["state"] == "BOUNDED"
    assert rows["above"]["assessment"] == "MET" and rows["beyond"]["assessment"] == "UNMET"
    assert rows["width"]["assessment"] == "INDETERMINATE"  # (a lower bound of the linewidth below a maximum decides nothing)
    with_loss = cavity_analysis(doc, {"components": ["M1", "L", "M2"], "wavelength_nm": LAMBDA_NM, "reflectances": {"M1": 0.99, "M2": 0.99},
                                      "round_trip_loss": 0.01, "targets": [{"target_id": "between", "quantity_id": "finesse", "minimum": 300.0}]})
    assert _value(with_loss, "finesse") < 300.0 and with_loss.arrays["targets"][0]["assessment"] == "UNMET"


def test_a_project_s_negligible_level_decides_when_light_not_resolved_leaves_a_power_known():
    """A beam 1.25 times the confocal mode resolved to order 2 leaves 2e-3 of its power in higher modes: BOUNDED at the Engine's
    level (1e-9 of the input). A Project that sets negligible_power_fraction to 1e-2 reads the same number as KNOWN, and the value
    says so: a MODEL basis that names the setting and the most the light left out can add."""

    mode = _g_mode(100.0, 100.0, 100.0)
    request = {"reflectances": {"M1": 0.99, "M2": 0.99}, "transmittances": {"M1": 0.009, "M2": 0.009},
               "input_beam": {"waist_radius_mm": mode["w0"] * 1.25, "waist_after_reference_mm": mode["z1"]}}
    injection = {"power_w": POWER, "beam": "INPUT_BEAM", "detuning_Hz": 0.0, "mode_order_max": 2}
    default = _analyse(cavity("100", "100", 100.0), injection, **request).quantity("transmitted_power_W")
    doc = {**cavity("100", "100", 100.0), "numerical_settings": {"negligible_power_fraction": 1e-2}}
    raised = _analyse(doc, injection, **request).quantity("transmitted_power_W")
    assert default.state.value == "BOUNDED" and raised.state.value == "KNOWN" and raised.value == default.value
    assert raised.interval == default.interval and raised.value_basis.kind.value == "MODEL"
    (omitted,) = raised.value_basis.omitted
    assert omitted.part.value == "BELOW_NEGLIGIBLE_LEVEL" and omitted.setting == "negligible_power_fraction"
    assert omitted.upper == default.value_basis.omitted[0].upper == raised.convergence.resolution


def _assert_transmission_not_traced(result) -> None:
    """The run names the light through the mirror out of its model (not a block), with the coating's transmittance and the route."""

    codes = {issue.code.value for issue in result.issues}
    assert "BLOCKED_BY_BACK_SIDE" not in codes
    (issue,) = [i for i in result.issues if i.numerical_context.get("cause") == "MIRROR_TRANSMISSION_NOT_TRACED"]
    assert issue.code.value == "OUT_OF_MODEL_SCOPE" and issue.status.value == "OUT_OF_MODEL_SCOPE"
    assert issue.numerical_context["transmittance"] == pytest.approx(0.009) and "cavity analysis" in issue.message
    assert [p.termination for p in result.paths] == ["BLOCKED_BY_BACK_SIDE"]


def test_a_run_reports_cavity_targets_out_of_its_model_and_its_results_query_assesses_them(tmp_path):
    targets = [{"target_id": "tp", "metric": "cavity_transmission", "minimum": 0.5, "unit": "1"}]
    doc = cavity("500", "500", 300.0, extra=(source("S", (-80.0, 0.0, 0.0), wavelength_nm=LAMBDA_NM, waist_mm=0.3, power_w=POWER),), targets=targets)
    run = run_optical_project(doc, catalog=None, output_dir=tmp_path / "run")
    (target,) = run.result.targets
    assert target.status.value == "OUT_OF_MODEL_SCOPE" and "cavity analysis" in target.message
    _assert_transmission_not_traced(run.result)
    request = {"components": ["M1", "M2"], "wavelength_nm": LAMBDA_NM, "injection": {"source_id": "S", "beam": "SOURCE", "detuning_Hz": 0.0}}
    results = load_run(tmp_path / "run")
    analysis = results.cavity(request)
    assert analysis.identity.physical.run_result_hash == results.identity()["result_hash"]
    assert analysis.identity.physical.engine_input_hash == results.identity()["input_hash"]
    assert analysis.arrays["targets"][0]["assessment"] == "MET"
    files = {path.name: path.read_bytes() for path in (tmp_path / "run").iterdir() if path.is_file()}
    reply = json_api.run_results(files, {"kind": "cavity", "request": request})
    assert reply["ok"] and reply["result"]["analysis"] == "CAVITY"
    refused = json_api.run_results(files, {"kind": "cavity", "request": {**request, "components": ["M1"]}})
    assert not refused["ok"] and refused["error"]["code"] == "CAVITY_ANALYSIS_INVALID"
    buffer = io.StringIO()
    (tmp_path / "request.json").write_text(json.dumps(request), encoding="utf-8")
    with redirect_stdout(buffer):
        code = cli_main(["results", "cavity", str(tmp_path / "run"), "--request", str(tmp_path / "request.json")])
    assert code == 0 and json.loads(buffer.getvalue())["analysis"] == "CAVITY"


@pytest.mark.parametrize("verify", [True, False])
def test_a_results_cavity_of_a_run_whose_pinned_catalog_is_not_here_names_it_and_is_no_internal_error(tmp_path, monkeypatch,
                                                                                                       capsys, verify):
    """A run of a catalog design, read on a device without the catalog release it pins (an empty store, no reachable catalog
    service): the cavity analysis lowers the run's Project again, which the local runtime refuses. The command answers that
    refusal with its problems (CATALOG_NOT_AVAILABLE) and exit 65, as 'project cavity' does -- not INTERNAL_ERROR with a
    traceback (exit 70) -- and the JSON API gives the LOCAL_RUN_ERROR envelope whether or not it verifies the files."""
    from os_optics.local import clear_installed_catalogs, example_project

    run_optical_project(example_project(), output_dir=tmp_path / "run")
    files = {path.name: path.read_bytes() for path in (tmp_path / "run").iterdir() if path.is_file()}
    monkeypatch.setenv("OPENSCI_CATALOG_DIR", str(tmp_path / "empty-store"))
    monkeypatch.setenv("OPENSCI_CATALOG_URL", "http://127.0.0.1:9")  # nothing listens
    clear_installed_catalogs()
    request = {"components": ["SRC", "L1"], "wavelength_nm": 1064}
    (tmp_path / "request.json").write_text(json.dumps(request), encoding="utf-8")
    capsys.readouterr()
    code = cli_main(["results", "cavity", str(tmp_path / "run"), "--request", str(tmp_path / "request.json"),
                     *(() if verify else ("--no-verify",))])
    out, err = capsys.readouterr()
    reply = json.loads(out)
    assert code == 65, reply
    assert reply["status"] == "ERROR" and "CATALOG_NOT_AVAILABLE" in [p["code"] for p in reply["problems"]]
    assert "INTERNAL_ERROR" not in out and "Traceback" not in err
    answer = json_api.run_results(files, {"kind": "cavity", "request": request, "verify": verify})
    assert answer["ok"] is False and answer["error"]["code"] == "LOCAL_RUN_ERROR", answer
    assert "CATALOG_NOT_AVAILABLE" in [p["code"] for p in answer["error"]["problems"]]


def test_a_beam_meeting_the_back_of_an_opaque_mirror_is_still_blocked():
    doc = cavity("500", "500", 300.0, coating=OPAQUE, extra=(source("S", (-80.0, 0.0, 0.0), wavelength_nm=LAMBDA_NM, waist_mm=0.3, power_w=POWER),))
    issues = run_optical_project(doc, catalog=None).result.issues
    assert "BLOCKED_BY_BACK_SIDE" in {issue.code.value for issue in issues}
    assert not any(issue.numerical_context.get("cause") == "MIRROR_TRANSMISSION_NOT_TRACED" for issue in issues)


def test_the_exposure_model_s_return_bound_for_the_same_mirrors_covers_the_resonant_build_up():
    """The trace bounds the light two facing mirrors return by the pair's gain g = r_a r_b (amplitude reflectances, their maximum over the
    angles the pair's light meets them): all returns together carry at most (1 / (1 - g))^2 times one crossing. The analysis's build-up of
    the light that entered, at any detuning, is at most 1 / (1 - a)^2 with the round-trip field gain a = sqrt(R1 R2): a <= g, so the trace's
    bound stays an upper bound; with an angle-independent coating it is attained on resonance."""

    from os_optics.config import NumericalConfig
    from os_optics.optics.materials.library import MaterialLibrary
    from os_optics.scene.lower import lower_scene
    from os_optics.validation import resonance

    doc = _fed()
    snapshot = lower_project(doc, None).request.project
    library = MaterialLibrary(snapshot.materials, NumericalConfig())
    scene, _ = lower_scene(snapshot, NumericalConfig(), library)
    gains = resonance.return_gains(scene, library, [LAMBDA_NM])[LAMBDA_NM]
    (pair,) = gains.pairs
    analysis = _analyse(doc, {"source_id": "S", "beam": "SOURCE", "detuning_Hz": 0.0})
    a = analysis.arrays["injection"]["coupler"]["round_trip_field_gain"]
    assert pair.gain == pytest.approx(a, rel=1e-12)
    bound = (1.0 + pair.returns) ** 2
    assert _value(analysis, "resonant_power_build_up") / analysis.arrays["injection"]["coupler"]["T1"] == pytest.approx(bound, rel=1e-12)
    scan = _analyse(doc, {"source_id": "S", "beam": "SOURCE", "detuning_Hz": 0.0, "scan": {"from_Hz": -5e8, "to_Hz": 5e8, "points": 101}})
    assert max(scan.arrays["injection"]["scan"]["circulating_W"]) <= POWER * 0.009 * bound * (1.0 + 1e-12)


def test_the_cli_injects_through_flags_and_a_request_file(tmp_path):
    path = tmp_path / "cavity.json"
    path.write_text(json.dumps(cavity("500", "500", 300.0)), encoding="utf-8")
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = cli_main(["project", "cavity", str(path), "--components", "M1", "M2", "--wavelength-nm", str(LAMBDA_NM), "--inject-power-w", "2",
                         "--inject-beam", "EIGENMODE", "--detuning-hz", "0"])
    doc = json.loads(buffer.getvalue())
    assert code == 0 and {q["quantity_id"]: q["value"] for q in doc["quantities"]}["injected_power_W"] == 2.0
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"components": ["M1", "M2"], "wavelength_nm": LAMBDA_NM, "reflectances": {"M1": 0.98}, "transmittances": {"M1": 0.019},
                                   "injection": {"power_w": 1.0, "beam": "EIGENMODE", "detuning_Hz": 0.0}}), encoding="utf-8")
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = cli_main(["project", "cavity", str(path), "--request", str(request), "--detuning-hz", "1e6"])
    doc = json.loads(buffer.getvalue())
    assert code == 0 and doc["settings"]["injection"]["detuning_Hz"] == 1e6 and doc["settings"]["reflectances"] == {"M1": 0.98}
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = cli_main(["project", "cavity", str(path), "--inject-power-w", "1"])
    assert code == 64


def test_capabilities_and_metric_routes_name_the_cavity_targets():
    capability = json_api.capabilities()["cavity_analysis"]
    assert capability["target_metrics"]["cavity_transmission"] == "transmission" and "INPUT_BEAM" in capability["injection"]["beams"]
    lower_project(cavity("500", "500", 300.0, extra=(source("S", (-80.0, 0.0, 0.0)),),
                         targets=[{"target_id": "x", "metric": "cavity_finesse", "minimum": 1.0}]), None)
    unknown = lower_project(cavity("500", "500", 300.0, extra=(source("S", (-80.0, 0.0, 0.0)),),  # a name that is not a cavity
                                   targets=[{"target_id": "x", "metric": "cavity_modes", "minimum": 1.0}]), None)  # metric stays unknown
    assert "TARGET_INVALID" in {p.code.value for p in unknown.target_problems}


# ------------------------------------------------------------------------------------------------ the two-mirror probe geometry 

PROBE = Path(__file__).parent / "fixtures" / "cavity" / "two_mirror_cavity_probe.json"
PROBE_NM = 632.8


def _probe() -> dict:
    """Two native concave mirrors (R = 120 mm, metallic R = 0.99, T = 0) 120 mm apart -- a symmetric confocal cavity -- with a 1 mW
    632.8 nm Gaussian source between them (``tests/fixtures/cavity/two_mirror_cavity_probe.json``)."""

    document = json.loads(PROBE.read_text(encoding="utf-8"))
    # (in vacuum: the confocal closed forms below take the medium between the mirrors to have index 1 -- the Project states no
    # environment, which is air at 20 C and 101325 Pa)
    document["environment"] = {"pressure_pa": "0"}
    return document


def test_the_probe_cavity_s_run_stops_at_the_interaction_limit_and_does_not_complete():
    run = run_optical_project(_probe(), catalog=None)
    outcome = run.local_validation.simulation.outcome
    assert outcome.execution.outcome == "PARTIAL" and not outcome.execution.complete
    assert "max_interactions_per_path" in outcome.execution.budgets.exhausted
    assert [p.termination for p in run.result.paths] == ["MAX_INTERACTIONS"]


def test_the_probe_cavity_s_modes_and_resonances_equal_the_confocal_closed_forms():
    analysis = cavity_analysis(_probe(), {"components": ["M1", "M2"], "wavelength_nm": PROBE_NM})
    lam = PROBE_NM * 1e-6
    assert _value(analysis, "gap_1.waist_radius_mm") == pytest.approx(math.sqrt(lam * 120.0 / (2.0 * math.pi)), rel=1e-12)
    fsr = C_MM_PER_S / 240.0
    assert _value(analysis, "free_spectral_range_Hz") == pytest.approx(fsr, rel=1e-12)
    assert _value(analysis, "transverse_mode_spacing_Hz") == pytest.approx(fsr / 2.0, rel=1e-12)
    assert _value(analysis, "finesse") == pytest.approx(math.pi * math.sqrt(0.99) / 0.01, rel=1e-12)
    # its source is inside the cavity: it does not feed it through a mirror. The injection quantities say so; the cavity's stand.
    fed = cavity_analysis(_probe(), {"components": ["M1", "M2"], "wavelength_nm": PROBE_NM, "injection": {"source_id": "SRC", "beam": "SOURCE"}})
    injected = fed.quantity("injected_power_W")
    assert injected.state.value == "OUT_OF_MODEL_SCOPE" and injected.reason == "CAVITY_SOURCE_UNSUPPORTED" and "inside the cavity" in injected.message
    assert fed.quantity("circulating_power_W").state.value == "OUT_OF_MODEL_SCOPE" and fed.quantity("circulating_power_W").value is None
    for quantity_id in ("gap_1.waist_radius_mm", "free_spectral_range_Hz", "transverse_mode_spacing_Hz", "finesse"):
        assert fed.quantity(quantity_id) == analysis.quantity(quantity_id)
    # its metallic mirrors transmit nothing: injected light is reflected (R) or absorbed (A)
    opaque = cavity_analysis(_probe(), {"components": ["M1", "M2"], "wavelength_nm": PROBE_NM,
                                        "injection": {"power_w": 0.001, "beam": "EIGENMODE", "detuning_Hz": 0.0}})
    assert _value(opaque, "transmittance.M1") == 0.0 and _value(opaque, "circulating_power_W") == 0.0
    assert _value(opaque, "reflected_power_W") == pytest.approx(0.99e-3) and _value(opaque, "lost_power_W") == pytest.approx(0.01e-3)


def _fed_probe() -> dict:
    """The probe with partially transmitting mirrors (R = 0.99, T = 0.009) and its source moved 60 mm behind M1, a target on the
    cavity's transmission."""

    doc = _probe()
    coating = doc["model_coatings"][0]
    coating["kind"] = "DIELECTRIC_HIGH_REFLECTOR"
    coating["bands"][0]["transmittance"] = "0.009"
    doc["components"][0]["nominal_pose"]["position_mm"] = [-60.0, 0.0, 0.0]
    doc["targets"] = [{"target_id": "T", "metric": "cavity_transmission", "minimum": 0.5, "unit": "1"}]
    return doc


def test_the_probe_cavity_fed_through_its_mirror_builds_up_in_the_analysis_and_is_refused_by_name_in_the_run():
    run = run_optical_project(_fed_probe(), catalog=None)
    _assert_transmission_not_traced(run.result)
    (target,) = run.result.targets
    assert target.status.value == "OUT_OF_MODEL_SCOPE"
    # (input_beam is the source's own beam at the reference plane: its 0.5 mm waist 60 mm before M1; a stated input beam must be)
    analysis = cavity_analysis(_fed_probe(), {"components": ["M1", "M2"], "wavelength_nm": PROBE_NM,
                                              "input_beam": {"waist_radius_mm": 0.5, "waist_after_reference_mm": -60.0},
                                              "injection": {"source_id": "SRC", "beam": "INPUT_BEAM", "detuning_Hz": 0.0, "mode_order_max": 40}})
    peak = 0.009 * 0.009 / (1.0 - 0.99) ** 2
    # confocal: every even order resonates with TEM00, so the mismatched beam is transmitted as a matched one (the rest bounded)
    assert _value(analysis, "mode_matching_efficiency") < 0.5
    transmitted = analysis.quantity("transmitted_power_W")
    assert transmitted.interval.lower <= 0.001 * peak <= transmitted.interval.upper * (1.0 + 1e-12)
    assert transmitted.interval.upper == pytest.approx(0.001 * peak, rel=1e-12)  # (the resolved even orders and the rest's upper bound)
    fsr = C_MM_PER_S / 240.0
    assert all(min(row["offset_from_tem00_Hz"], fsr - row["offset_from_tem00_Hz"]) < 1e-3 for row in analysis.arrays["injection"]["modes"])
    assert analysis.arrays["targets"][0]["assessment"] in ("MET", "INDETERMINATE")


def test_a_target_inside_the_bound_of_the_light_left_out_is_not_decided_by_the_truncation():
    """W28 in the cavity analysis: at a Project level of 1e-2 the transmitted power is KNOWN, its PROVEN interval beside it (the
    light in modes not resolved may add). A maximum between the value and the interval's upper end is not decided; one above
    the upper end is met; one below the lower end is not met."""

    mode = _g_mode(100.0, 100.0, 100.0)
    request = {"reflectances": {"M1": 0.99, "M2": 0.99}, "transmittances": {"M1": 0.009, "M2": 0.009},
               "input_beam": {"waist_radius_mm": mode["w0"] * 1.25, "waist_after_reference_mm": mode["z1"]}}
    injection = {"power_w": POWER, "beam": "INPUT_BEAM", "detuning_Hz": 0.0, "mode_order_max": 2}
    doc = {**cavity("100", "100", 100.0), "numerical_settings": {"negligible_power_fraction": 1e-2}}
    quantity = _analyse(doc, injection, **request).quantity("transmitted_power_W")
    assert quantity.state.value == "KNOWN" and quantity.interval.lower <= quantity.value < quantity.interval.upper
    v, hi, lo = quantity.value, quantity.interval.upper, quantity.interval.lower
    targets = [{"target_id": "inside", "quantity_id": "transmitted_power_W", "maximum": 0.5 * (v + hi)},
               {"target_id": "above", "quantity_id": "transmitted_power_W", "maximum": hi * 1.001},
               {"target_id": "below", "quantity_id": "transmitted_power_W", "maximum": lo * 0.999}]
    rows = {row["target_id"]: row for row in _analyse(doc, injection, targets=targets, **request).arrays["targets"]}
    assert rows["inside"]["assessment"] == "INDETERMINATE" and rows["inside"]["reason"] == "INTERVAL_STRADDLES_THRESHOLD"
    assert rows["inside"]["value"] == v and rows["inside"]["interval"] == [lo, hi]
    assert rows["inside"]["value_basis"]["omitted"][0]["part"] == "BELOW_NEGLIGIBLE_LEVEL"
    assert "negligible_power_fraction" in rows["inside"]["message"]
    assert rows["above"]["assessment"] == "MET" and rows["below"]["assessment"] == "UNMET"
