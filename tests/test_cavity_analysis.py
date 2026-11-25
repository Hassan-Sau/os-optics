"""Cavity analysis of placed mirrors (``os_optics.local.cavity``): stability, the Gaussian eigenmode, Gouy phase, free spectral range,
transverse-mode spacing, finesse, linewidth and mode matching, as an analysis document.

Every expected number is computed here independently of the code under test: the two-mirror g-parameter formulas
(``w1^2 = (lambda L / pi) sqrt(g2 / (g1 (1 - g1 g2)))``, the waist ``w0^2 = (lambda L / pi) sqrt(g1 g2 (1 - g1 g2) / (g1 + g2 - 2 g1 g2)^2)`` at
``z1 = g2 (1 - g1) L / (g1 + g2 - 2 g1 g2)`` from the first mirror, the one-way Gouy phase ``acos(+-sqrt(g1 g2))``), the confocal and plane-concave closed forms,
``FSR = c / 2L``, the finesse ``pi sqrt(R) / (1 - R)`` and the Airy half width, the overlap of two Gaussians from their radii and wavefront curvatures,
and, for a cavity with a lens, round-trip matrices written here with numpy, the eigen-q from the quadratic's roots and the Gouy phase summed segment by
segment as ``atan(z / z_R)``.
"""

from __future__ import annotations

import copy
import io
import json
import math
from contextlib import redirect_stdout

import numpy as np
import pytest

from declared_helpers import NBK7_SELLMEIER, declared_component, nbk7, plano_convex_model, sequence, surface, thin_lens_model, with_models
from local_helpers import (AGAINST_X, ALONG_X, FLAT_MIRROR, FOLD_X_TO_Y, TEST_CATALOG_VERSION, arrivals, catalog_document, element, free, plane, project,
                           quat_multiply, run, source)
from os_optics.cli import main as cli_main
from os_optics.local import json_api
from os_optics.local.cavity import CavityError, cavity_analysis
from os_optics.local.lowering import lower_project

HR = next(c for c in catalog_document()["coatings"] if c["kind"] == "DIELECTRIC_HIGH_REFLECTOR")
R_HR = 0.99
LAMBDA_NM = 1064.0
LAMBDA_MM = LAMBDA_NM * 1e-6
C_MM_PER_S = 299792458.0e3
#: +Z along -Y: 90 deg about X.
AGAINST_Y = (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0)


def _mirror(model_id: str, radius: str | None, clear: str = "12.7", coating: dict = HR) -> dict:
    return {"model_id": model_id, "model": sequence([surface(1, "0", radius=radius, clear=clear, traced="REFLECTED", coating_id=coating["coating_id"])], [],
                                                    edge="13")}


def two_mirror_cavity(r1: str | None, r2: str | None, length: float, *, clear: str = "12.7", coating: dict = HR, extra: tuple = ()) -> dict:
    """M1 at x = 0 facing +X, M2 at x = length facing -X (radius > 0: concave towards the cavity)."""

    doc = project([declared_component("M1", "mirror", "m1", (0.0, 0.0, 0.0), ALONG_X), declared_component("M2", "mirror", "m2", (length, 0.0, 0.0), AGAINST_X),
                   *extra], [plane("p", (length / 2.0, 0.0, 0.0))])
    coatings = [HR] if coating is HR else [HR, coating]
    return with_models(doc, [_mirror("m1", r1, clear, coating), _mirror("m2", r2, clear, coating)], coatings=coatings)


def _q(doc: dict, request: dict | None = None, **kwargs):
    return cavity_analysis(doc, {"components": ["M1", "M2"], "wavelength_nm": LAMBDA_NM, **(request or {})}, **kwargs)


def _value(analysis, quantity_id: str) -> float:
    quantity = analysis.quantity(quantity_id)
    assert quantity.state.value == "KNOWN", (quantity_id, quantity.state, quantity.reason, quantity.message)
    return quantity.value


def _g_reference(r1: float | None, r2: float | None, length: float) -> dict:
    """The two-mirror eigenmode from the g parameters (standard resonator formulas)."""

    g1 = 1.0 if r1 is None else 1.0 - length / r1
    g2 = 1.0 if r2 is None else 1.0 - length / r2
    p = g1 * g2
    scale = LAMBDA_MM * length / math.pi
    w0_sq = scale * math.sqrt(p * (1.0 - p) / (g1 + g2 - 2.0 * p) ** 2)
    one_way = math.acos(math.copysign(math.sqrt(p), g1))
    return {"g1": g1, "g2": g2, "w1": math.sqrt(scale * math.sqrt(g2 / (g1 * (1.0 - p)))), "w2": math.sqrt(scale * math.sqrt(g1 / (g2 * (1.0 - p)))),
            "w0": math.sqrt(w0_sq), "z1": g2 * (1.0 - g1) * length / (g1 + g2 - 2.0 * p), "gouy": 2.0 * one_way}


# ------------------------------------------------------------------------------------------------ analytic references


def test_symmetric_confocal_cavity():
    """R1 = R2 = L = 100 mm: g1 = g2 = 0, w0^2 = lambda L / (2 pi) at the centre, sqrt(2) w0 on the mirrors, Gouy pi per round trip, so the transverse modes
    lie FSR / 2 apart; FSR = c / 2L; two R = 0.99 mirrors: finesse pi sqrt(R) / (1 - R). Its round trip is -I (every beam reproduces): the half round trip
    gives the mode."""

    analysis = _q(two_mirror_cavity("100", "100", 100.0))
    assert analysis.analysis == "CAVITY" and analysis.execution.complete
    w0 = math.sqrt(LAMBDA_MM * 100.0 / (2.0 * math.pi))
    assert _value(analysis, "g1") == 0.0 and _value(analysis, "g2") == 0.0 and _value(analysis, "g1g2") == pytest.approx(0.0, abs=1e-15)
    assert _value(analysis, "gap_1.waist_radius_mm") == pytest.approx(w0, rel=1e-12)
    assert _value(analysis, "gap_1.waist_along_axis_mm") == pytest.approx(50.0, rel=1e-12)
    assert _value(analysis, "mode_radius_mm.M1") == pytest.approx(math.sqrt(2.0) * w0, rel=1e-12)
    assert _value(analysis, "mode_radius_mm.M2") == pytest.approx(math.sqrt(2.0) * w0, rel=1e-12)
    assert _value(analysis, "round_trip_gouy_phase_rad") == pytest.approx(math.pi, rel=1e-12)
    fsr = C_MM_PER_S / 200.0
    assert _value(analysis, "free_spectral_range_Hz") == pytest.approx(fsr, rel=1e-12)
    assert _value(analysis, "transverse_mode_spacing_Hz") == pytest.approx(fsr / 2.0, rel=1e-12)
    finesse = math.pi * math.sqrt(R_HR) / (1.0 - R_HR)
    assert _value(analysis, "finesse") == pytest.approx(finesse, rel=1e-12)
    assert _value(analysis, "linewidth_fwhm_Hz") == pytest.approx(2.0 * fsr / math.pi * math.asin((1.0 - R_HR) / (2.0 * math.sqrt(R_HR))), rel=1e-12)
    assert _value(analysis, "linewidth_fwhm_Hz") == pytest.approx(fsr / finesse, rel=1e-4)  # high finesse: FSR / F
    assert analysis.arrays["eigenmode_from"] == "HALF_ROUND_TRIP" and np.array(analysis.arrays["round_trip_abcd"]) == pytest.approx(-np.eye(2), abs=1e-12)


@pytest.mark.parametrize("r1, r2, length", [(None, 500.0, 100.0), (None, 500.0, 250.0), (500.0, 500.0, 300.0), (300.0, 500.0, 200.0),
                                            (-1000.0, 500.0, 200.0), (105.0, 105.0, 200.0)],
                         ids=["plane-concave", "half-symmetric", "symmetric-R500", "asymmetric", "convex-concave", "near-concentric"])
def test_two_mirror_cavities_equal_the_g_parameter_formulas(r1, r2, length):
    analysis = _q(two_mirror_cavity(None if r1 is None else f"{r1:g}", f"{r2:g}", length))
    ref = _g_reference(r1, r2, length)
    assert _value(analysis, "g1") == pytest.approx(ref["g1"], rel=1e-12, abs=1e-15) and _value(analysis, "g2") == pytest.approx(ref["g2"], rel=1e-12)
    assert _value(analysis, "g1g2") == pytest.approx(ref["g1"] * ref["g2"], rel=1e-12)
    assert _value(analysis, "mode_radius_mm.M1") == pytest.approx(ref["w1"], rel=1e-11)
    assert _value(analysis, "mode_radius_mm.M2") == pytest.approx(ref["w2"], rel=1e-11)
    assert _value(analysis, "gap_1.waist_radius_mm") == pytest.approx(ref["w0"], rel=1e-11)
    assert _value(analysis, "gap_1.waist_along_axis_mm") == pytest.approx(ref["z1"], rel=1e-11, abs=1e-9)
    assert _value(analysis, "round_trip_gouy_phase_rad") == pytest.approx(ref["gouy"], rel=1e-12)
    fsr = C_MM_PER_S / (2.0 * length)
    assert _value(analysis, "free_spectral_range_Hz") == pytest.approx(fsr, rel=1e-12)
    assert _value(analysis, "transverse_mode_spacing_Hz") == pytest.approx(fsr * ref["gouy"] / (2.0 * math.pi), rel=1e-12)
    inside = analysis.arrays["waists"][0]["inside_gap"]
    assert inside is (-1e-9 <= ref["z1"] <= length + 1e-9)


def test_the_plane_concave_closed_form_and_the_virtual_waist_of_a_convex_concave_cavity():
    """Plane-concave: w0^2 = (lambda / pi) sqrt(L (R - L)) on the plane mirror. Convex-concave (g1 = 1.2, g2 = 0.6): stable, its waist lies outside the cavity,
    66.7 mm behind the convex mirror."""

    plane_concave = _q(two_mirror_cavity(None, "500", 100.0))
    assert _value(plane_concave, "gap_1.waist_radius_mm") == pytest.approx(math.sqrt(LAMBDA_MM / math.pi * math.sqrt(100.0 * 400.0)), rel=1e-12)
    assert _value(plane_concave, "gap_1.waist_along_axis_mm") == pytest.approx(0.0, abs=1e-9)
    convex = _q(two_mirror_cavity("-1000", "500", 200.0))
    assert _value(convex, "gap_1.waist_along_axis_mm") == pytest.approx(-200.0 / 3.0, rel=1e-11)
    assert convex.arrays["waists"][0]["inside_gap"] is False
    assert convex.arrays["waists"][0]["point_mm"] == pytest.approx([-200.0 / 3.0, 0.0, 0.0], abs=1e-9)


def test_an_unstable_cavity_has_no_eigenmode_and_keeps_its_other_quantities():
    """R = 50 mm, L = 120 mm: g = -1.4, g1 g2 = 1.96 > 1. The eigenmode, Gouy phase and mode spacing are outside the model; FSR stays; the finesse of the
    mirrors alone bounds the cavity's from above (its geometric loss lowers it)."""

    analysis = _q(two_mirror_cavity("50", "50", 120.0))
    # (an answered analysis completed: the quantities outside the model say so themselves)
    assert analysis.execution.outcome.value == "COMPLETED" and analysis.execution.complete
    assert analysis.execution.termination.value == "COMPLETED" and analysis.execution.reason == ""
    for quantity_id in ("round_trip_gouy_phase_rad", "mode_radius_mm.M1", "mode_radius_mm.M2", "gap_1.waist_radius_mm", "transverse_mode_spacing_Hz"):
        quantity = analysis.quantity(quantity_id)
        assert quantity.state.value == "OUT_OF_MODEL_SCOPE" and quantity.reason == "CAVITY_UNSTABLE" and quantity.value is None
        assert "g1 g2 = 1.96" in quantity.message
    assert _value(analysis, "g1g2") == pytest.approx(1.96, rel=1e-12) and _value(analysis, "round_trip_half_trace") == pytest.approx(2.92, rel=1e-12)
    assert _value(analysis, "free_spectral_range_Hz") == pytest.approx(C_MM_PER_S / 240.0, rel=1e-12)
    finesse = analysis.quantity("finesse")
    assert finesse.state.value == "BOUNDED" and finesse.interval.kind.value == "PROVEN" and finesse.interval.lower is None
    assert finesse.interval.upper == pytest.approx(math.pi * math.sqrt(R_HR) / (1.0 - R_HR), rel=1e-12)
    # the value beside the bound is the mirrors' finesse, named an upper bound with the loss it leaves out (the unstable geometry's)
    assert finesse.value == finesse.interval.upper and finesse.value_basis.kind.value == "UPPER_BOUND"
    assert [(a.input.value, a.at, a.code) for a in finesse.value_basis.assumed] == [("NO_UNSTATED_LOSS", (), "CAVITY_UNSTABLE")]
    width = analysis.quantity("linewidth_fwhm_Hz")
    assert width.state.value == "BOUNDED" and width.interval.upper is None and width.interval.lower > 0.0
    assert width.value == width.interval.lower == pytest.approx(2.0 * C_MM_PER_S / 240.0 / math.pi * math.asin((1.0 - R_HR) / (2.0 * math.sqrt(R_HR))), rel=1e-12)
    assert width.value_basis.kind.value == "LOWER_BOUND" and width.value_basis.assumed == finesse.value_basis.assumed
    stability = next(c for c in analysis.models[0].validity if c.parameter.endswith("half_trace"))
    assert stability.within is False


@pytest.mark.parametrize("r1, r2, length", [(None, None, 100.0), (None, "100", 100.0)], ids=["plane-plane", "hemispherical"])
def test_a_marginal_cavity_has_no_gaussian_eigenmode(r1, r2, length):
    analysis = _q(two_mirror_cavity(r1, r2, length))
    assert analysis.quantity("mode_radius_mm.M1").reason == "CAVITY_MARGINALLY_STABLE"
    assert analysis.quantity("free_spectral_range_Hz").state.value == "KNOWN"


# ------------------------------------------------------------------------------------------------ folds, lenses


def _folded() -> dict:
    """M1 (R 200) at the origin facing +X, a catalog plane mirror at x = 60 folding +X to +Y, M2 (R 200) at (60, 40, 0) facing -Y: unfolded L = 100."""

    doc = project([declared_component("M1", "mirror", "m1", (0.0, 0.0, 0.0), ALONG_X), element("FM", "mirror", FLAT_MIRROR, (60.0, 0.0, 0.0), FOLD_X_TO_Y),
                   declared_component("M2", "mirror", "m2", (60.0, 40.0, 0.0), AGAINST_Y)], [plane("p", (30.0, 0.0, 0.0))])
    return with_models(doc, [_mirror("m1", "200"), _mirror("m2", "200")], coatings=[HR])


def test_a_folded_cavity_equals_the_straight_one_and_meets_the_fold_twice():
    folded = cavity_analysis(_folded(), {"components": ["M1", "FM", "M2"], "wavelength_nm": LAMBDA_NM}, catalog=catalog_document())
    straight = _q(two_mirror_cavity("200", "200", 100.0))
    for quantity_id in ("round_trip_half_trace", "mode_radius_mm.M1", "mode_radius_mm.M2", "round_trip_gouy_phase_rad", "free_spectral_range_Hz",
                        "transverse_mode_spacing_Hz"):
        assert _value(folded, quantity_id) == pytest.approx(_value(straight, quantity_id), rel=1e-12)
    assert _value(folded, "gap_1.waist_radius_mm") == pytest.approx(_value(straight, "gap_1.waist_radius_mm"), rel=1e-12)
    # the waist at the centre of the unfolded cavity lies in the first gap; the second gap's beam has the same waist, virtual, behind the fold
    assert _value(folded, "gap_1.waist_along_axis_mm") == pytest.approx(50.0, rel=1e-12) and _value(folded, "gap_2.waist_along_axis_mm") == pytest.approx(50.0, rel=1e-12)
    assert [w["inside_gap"] for w in folded.arrays["waists"]] == [True, False]
    assert folded.arrays["waists"][0]["point_mm"] == pytest.approx([50.0, 0.0, 0.0], abs=1e-9)
    assert folded.arrays["waists"][1]["point_mm"] == pytest.approx([60.0, -10.0, 0.0], abs=1e-9)
    rho = R_HR ** 4  # M1 once, the fold twice, M2 once
    assert _value(folded, "round_trip_power_fraction") == pytest.approx(rho, rel=1e-12)
    g = math.sqrt(rho)
    assert _value(folded, "finesse") == pytest.approx(math.pi * math.sqrt(g) / (1.0 - g), rel=1e-12)
    assert [e["reflections_per_round_trip"] for e in folded.arrays["elements"]] == [1, 2, 1]


#: A fold coating whose reflectance depends on the polarization away from normal incidence: at 45 deg R_s = 0.996, R_p = 0.97.
SP = {"coating_id": "SP-FOLD", "kind": "DIELECTRIC_HIGH_REFLECTOR", "display_name": "Fold coating",
      "valid_aoi_deg": {"min_deg": "0", "max_deg": "45", "status": "DECLARED"}, "aoi_reference": "INCIDENT_MEDIUM", "basis": "fixture data",
      "angular_spectrum": {"wavelength_nm": ["400", "1100"], "aoi_deg": ["0", "20", "45"],
                           "reflectance_s": [["0.99", "0.992", "0.996"]] * 2, "reflectance_p": [["0.99", "0.985", "0.97"]] * 2,
                           "transmittance_s": [["0.009", "0.007", "0.003"]] * 2, "transmittance_p": [["0.009", "0.014", "0.029"]] * 2,
                           "status": "DECLARED"}}


def test_a_fold_mirror_s_reflectance_is_the_stated_polarization_s_where_s_and_p_are_eigen_polarizations():
    """The folded cavity with a fold whose coating reflects R_s = 0.996 and R_p = 0.97 at 45 deg. Both are reported; the round trip
    needs one of them. Its tilted mirror has one plane of incidence, so s and p are eigen-polarizations: polarization 'S' takes R_s
    and gives exactly the cavity with 0.996 typed in reflectances, 'P' the one with 0.97. A cavity folded in two planes has no such
    polarization (the fold's reflectance stays unknown and names reflectances); a cavity without a tilted mirror refuses S / P."""

    doc = project([declared_component("M1", "mirror", "m1", (0.0, 0.0, 0.0), ALONG_X), declared_component("FM", "mirror", "fm", (60.0, 0.0, 0.0), FOLD_X_TO_Y),
                   declared_component("M2", "mirror", "m2", (60.0, 40.0, 0.0), AGAINST_Y)], [plane("p", (30.0, 0.0, 0.0))])
    doc = with_models(doc, [_mirror("m1", "200"), _mirror("fm", None, coating=SP), _mirror("m2", "200")], coatings=[HR, SP])
    request = {"components": ["M1", "FM", "M2"], "wavelength_nm": LAMBDA_NM}
    plain = cavity_analysis(doc, request)
    fold = plain.quantity("reflectance.FM")
    assert fold.state.value == "UNKNOWN" and fold.reason == "REFLECTANCE_DEPENDS_ON_POLARIZATION" and fold.missing == ("polarization", "reflectances.FM")
    assert _value(plain, "reflectance_s.FM") == 0.996 and _value(plain, "reflectance_p.FM") == 0.97
    assert plain.quantity("finesse").state.value == "UNKNOWN"
    for polarization, value in (("S", 0.996), ("P", 0.97)):
        chosen = cavity_analysis(doc, {**request, "polarization": polarization})
        typed = cavity_analysis(doc, {**request, "reflectances": {"FM": value}})
        assert _value(chosen, "reflectance.FM") == value and chosen.quantity("reflectance.FM").origin.value == "MODELED"
        for quantity_id in ("round_trip_power_fraction", "finesse", "linewidth_fwhm_Hz"):
            assert _value(chosen, quantity_id) == _value(typed, quantity_id)
        assert _value(chosen, "round_trip_power_fraction") == pytest.approx(R_HR * value * value * R_HR, rel=1e-14)
        assert chosen.settings["polarization"] == polarization and any(f"{polarization.lower()}-polarized" in a for a in chosen.models[1].assumptions)
    # folded in two planes (+X to +Y, then +Y to +Z): the planes of incidence differ
    c, s = math.cos(math.radians(22.5)), math.sin(math.radians(22.5))
    twisted = project([declared_component("M1", "mirror", "m1", (0.0, 0.0, 0.0), ALONG_X), declared_component("FM1", "mirror", "fm", (60.0, 0.0, 0.0), FOLD_X_TO_Y),
                       declared_component("FM2", "mirror", "fm2", (60.0, 40.0, 0.0), (c, s, 0.0, 0.0)),
                       declared_component("M2", "mirror", "m2", (60.0, 40.0, 30.0), (0.0, 1.0, 0.0, 0.0))], [plane("p", (30.0, 0.0, 0.0))])
    twisted = with_models(twisted, [_mirror("m1", "200"), _mirror("fm", None, coating=SP), _mirror("fm2", None), _mirror("m2", "200")], coatings=[HR, SP])
    out_of_plane = cavity_analysis(twisted, {"components": ["M1", "FM1", "FM2", "M2"], "wavelength_nm": LAMBDA_NM, "polarization": "S"})
    assert _value(out_of_plane, "cavity_length_mm") == pytest.approx(130.0, rel=1e-12)
    fold = out_of_plane.quantity("reflectance.FM1")
    assert fold.state.value == "UNKNOWN" and fold.reason == "REFLECTANCE_DEPENDS_ON_POLARIZATION" and fold.missing == ("reflectances.FM1",)
    assert out_of_plane.quantity("finesse").state.value == "UNKNOWN" and _value(out_of_plane, "reflectance_s.FM1") == 0.996
    (problem,) = _problems(two_mirror_cavity("200", "200", 100.0), {"components": ["M1", "M2"], "wavelength_nm": LAMBDA_NM, "polarization": "S"})
    assert problem["location"] == "polarization" and "no tilted mirror" in problem["message"]


def _lens_cavity(lens: dict, lens_x: float, m2_x: float) -> dict:
    doc = project([declared_component("M1", "mirror", "m1", (0.0, 0.0, 0.0), ALONG_X), declared_component("L", "lens", lens["model_id"], (lens_x, 0.0, 0.0), ALONG_X),
                   declared_component("M2", "mirror", "m2", (m2_x, 0.0, 0.0), AGAINST_X)], [plane("p", (20.0, 0.0, 0.0))])
    return with_models(doc, [_mirror("m1", None), lens, _mirror("m2", None)], [nbk7()], coatings=[HR])


def test_a_plane_lens_plane_cavity_equals_an_independent_round_trip():
    """Plane mirrors 130 mm apart, a thin lens f = 100 mm 50 mm from M1: by hand P = T(80) L T(50) = ((0.2, 90), (-0.01, 0.5)), the round trip
    ((-0.8, ...), (..., -0.8)): stable, not symmetric. The eigen-q from the roots of C q^2 + (D - A) q - B, the Gouy phase summed as atan(z / z_R) over the
    four free segments. A thin lens states no thickness: its paraxial matrix takes it at one plane, so the group path counts none
    through it -- twice the gaps, FSR = c / (2 x 130 mm) -- as an estimate that names the lens; the finesse is an upper bound (the
    lens's losses are not evaluated)."""

    analysis = cavity_analysis(_lens_cavity(thin_lens_model("f100", "100", clear="12"), 50.0, 130.0),
                               {"components": ["M1", "L", "M2"], "wavelength_nm": LAMBDA_NM})
    lens = np.array([[1.0, 0.0], [-0.01, 1.0]])
    forward = free(80.0) @ lens @ free(50.0)
    back = free(50.0) @ lens @ free(80.0)
    round_trip = back @ forward
    assert round_trip[0, 0] == pytest.approx(-0.8, abs=1e-12) and np.array(analysis.arrays["round_trip_abcd"]) == pytest.approx(round_trip, abs=1e-12)
    (a, b), (c, d) = round_trip
    roots = np.roots([c, d - a, -b])
    q0 = next(complex(r) for r in roots if complex(r).imag > 0.0)
    radius = lambda q: math.sqrt(-LAMBDA_MM / (math.pi * (1.0 / q).imag))  # noqa: E731
    assert _value(analysis, "mode_radius_mm.M1") == pytest.approx(radius(q0), rel=1e-11)
    q_lens = q0 + 50.0
    q_m2 = (q_lens / (-0.01 * q_lens + 1.0)) + 80.0
    assert _value(analysis, "mode_radius_mm.M2") == pytest.approx(radius(q_m2), rel=1e-11)
    gouy = 0.0
    for q_start, length in ((q0, 50.0), (q_lens / (-0.01 * q_lens + 1.0), 80.0)):
        z_r, z = q_start.imag, q_start.real
        gouy += math.atan((z + length) / z_r) - math.atan(z / z_r)
    assert _value(analysis, "round_trip_gouy_phase_rad") == pytest.approx((2.0 * gouy) % (2.0 * math.pi), rel=1e-11)
    assert _value(analysis, "gap_1.waist_radius_mm") == pytest.approx(math.sqrt(LAMBDA_MM * q0.imag / math.pi), rel=1e-11)
    assert analysis.arrays["eigenmode_from"] == "ROUND_TRIP"
    fsr = C_MM_PER_S / 260.0
    expected = {"round_trip_group_path_mm": 260.0, "free_spectral_range_Hz": fsr,
                "transverse_mode_spacing_Hz": fsr * ((2.0 * gouy) % (2.0 * math.pi)) / (2.0 * math.pi),
                "linewidth_fwhm_Hz": 2.0 * fsr / math.pi * math.asin((1.0 - R_HR) / (2.0 * math.sqrt(R_HR)))}
    for quantity_id, value in expected.items():
        quantity = analysis.quantity(quantity_id)
        assert quantity.state.value == "UNKNOWN" and quantity.reason == "OPTICAL_THICKNESS_NOT_STATED" and "'L'" in quantity.message
        assert quantity.value == pytest.approx(value, rel=1e-11) and quantity.value_basis.kind.value == "ESTIMATE", quantity_id
        thin = [a for a in quantity.value_basis.assumed if a.input.value == "THIN_ELEMENT_PATH"]
        assert [(a.at, a.code) for a in thin] == [(("L",), "OPTICAL_THICKNESS_NOT_STATED")]
    # (the linewidth also takes the lens's unstated loss as zero: both stand-ins are named, and no bound is claimed)
    width = analysis.quantity("linewidth_fwhm_Hz")
    assert [a.input.value for a in width.value_basis.assumed] == ["THIN_ELEMENT_PATH", "NO_UNSTATED_LOSS"] and width.interval is None
    # injected light: a stated detuning of 1 MHz is a phase the estimated FSR gives, so the powers keep their bounds over every
    # detuning with the Airy value at that phase beside them; a length offset fixes the phase itself
    request = {"components": ["M1", "L", "M2"], "wavelength_nm": LAMBDA_NM, "reflectances": {"M1": 0.99, "M2": 0.99},
               "transmittances": {"M1": 0.01, "M2": 0.01}, "round_trip_loss": 0.0}
    doc = _lens_cavity(thin_lens_model("f100", "100", clear="12"), 50.0, 130.0)
    detuned = cavity_analysis(doc, {**request, "injection": {"power_w": 1.0, "beam": "EIGENMODE", "detuning_Hz": 1.0e6}})
    transmitted = detuned.quantity("transmitted_power_W")
    airy = 0.01 * 0.01 / ((1.0 - 0.99) ** 2 + 4.0 * 0.99 * math.sin(math.pi * 1.0e6 / fsr) ** 2)
    assert transmitted.state.value == "BOUNDED" and transmitted.value == pytest.approx(airy, rel=1e-9)
    assert transmitted.value_basis.kind.value == "ESTIMATE" and [a.input.value for a in transmitted.value_basis.assumed] == ["THIN_ELEMENT_PATH"]
    assert transmitted.interval.lower == pytest.approx(0.0001 / 1.99 ** 2, rel=1e-9) and transmitted.interval.upper == pytest.approx(1.0, rel=1e-9)
    assert _value(detuned, "detuning_Hz") == 1.0e6
    offset = cavity_analysis(doc, {**request, "injection": {"power_w": 1.0, "beam": "EIGENMODE", "length_offset_nm": 0.0}})
    assert _value(offset, "transmitted_power_W") == pytest.approx(1.0, rel=1e-12)  # (on resonance, whatever the FSR)
    derived = offset.quantity("detuning_Hz")
    assert derived.state.value == "UNKNOWN" and derived.value == 0.0 and derived.value_basis.kind.value == "ESTIMATE"
    finesse = analysis.quantity("finesse")
    assert finesse.state.value == "BOUNDED" and "'L'" in finesse.message
    # the mirrors' finesse pi sqrt(R) / (1 - R) beside its bound: an upper bound that takes the lens's unstated loss as zero
    assert finesse.value == finesse.interval.upper == pytest.approx(math.pi * math.sqrt(R_HR) / (1.0 - R_HR), rel=1e-12)
    assert finesse.value_basis.kind.value == "UPPER_BOUND" and finesse.missing == ("round_trip_loss",)
    assert [(a.input.value, a.at) for a in finesse.value_basis.assumed] == [("NO_UNSTATED_LOSS", ("L",))]
    assert analysis.execution.outcome.value == "COMPLETED" and analysis.execution.complete


def _group_index_nbk7(wavelength_nm: float) -> float:
    """n - lambda dn/dlambda from the Sellmeier formula written here, dn/dlambda by a central difference."""

    b = [float(v) for v in NBK7_SELLMEIER["b"]]
    c = [float(v) for v in NBK7_SELLMEIER["c_um2"]]

    def n(lam_um: float) -> float:
        return math.sqrt(1.0 + sum(bi * lam_um ** 2 / (lam_um ** 2 - ci) for bi, ci in zip(b, c)))

    lam, h = wavelength_nm * 1e-3, 1e-5
    return n(lam) - lam * (n(lam + h) - n(lam - h)) / (2.0 * h)


def test_the_free_spectral_range_counts_the_group_path_through_a_thick_lens():
    """An N-BK7 plano-convex lens (4.5 mm thick) between plane mirrors: L_rt = 2 (free gaps + n_g t)."""

    analysis = cavity_analysis(_lens_cavity(plano_convex_model("pcx"), 40.0, 130.0), {"components": ["M1", "L", "M2"], "wavelength_nm": LAMBDA_NM})
    path = 2.0 * ((130.0 - 4.5) + _group_index_nbk7(LAMBDA_NM) * 4.5)
    assert _value(analysis, "round_trip_group_path_mm") == pytest.approx(path, rel=1e-10)
    assert _value(analysis, "free_spectral_range_Hz") == pytest.approx(C_MM_PER_S / path, rel=1e-10)
    reply = json_api.paraxial_analysis(_lens_cavity(plano_convex_model("pcx"), 40.0, 130.0), {"components": ["L"], "wavelength_nm": LAMBDA_NM})
    lens = np.array(reply["system"]["abcd"])
    forward = free(130.0 - 40.0 - 4.5) @ lens @ free(40.0)
    back = free(40.0) @ np.array([[lens[1, 1], lens[0, 1]], [lens[1, 0], lens[0, 0]]]) @ free(130.0 - 40.0 - 4.5)
    assert np.array(analysis.arrays["round_trip_abcd"]) == pytest.approx(back @ forward, abs=1e-12)


# ------------------------------------------------------------------------------------------------ reflectances, apertures, mode matching


def test_stated_reflectances_replace_the_coating_and_set_the_finesse():
    analysis = _q(two_mirror_cavity("200", "200", 100.0), {"reflectances": {"M1": 0.95}})
    r1 = analysis.quantity("reflectance.M1")
    assert r1.value == 0.95 and r1.origin.value == "EXTERNALLY_CHARACTERIZED" and r1.source
    assert analysis.quantity("reflectance.M2").origin.value == "MODELED" and _value(analysis, "reflectance.M2") == R_HR
    product = 0.95 * R_HR
    assert _value(analysis, "finesse") == pytest.approx(math.pi * product ** 0.25 / (1.0 - math.sqrt(product)), rel=1e-12)
    low = _q(two_mirror_cavity("200", "200", 100.0), {"reflectances": {"M1": 0.1, "M2": 0.1}})
    assert low.quantity("linewidth_fwhm_Hz").state.value == "NOT_APPLICABLE" and low.quantity("linewidth_fwhm_Hz").reason == "NO_HALF_MAXIMUM"
    assert low.execution.complete


def test_a_reflectance_outside_the_coating_data_is_the_held_value_the_run_traces_with_as_an_estimate():
    """Mirrors characterized from 400 to 900 nm, analysed at 1064 nm. The compiled coating holds its nearest published value there,
    and a run traces with it: a beam reflected once by such a mirror returns 0.99 of its power. The analysis reports that number
    for the reflectance and what reads it, each UNKNOWN with an ESTIMATE basis that states the wavelength against the record's band
    and the mirrors it is held for; the injection waits for a stated coefficient. Inside the band nothing changes."""

    narrow = copy.deepcopy(HR)
    narrow["coating_id"] = "HR-400-900"
    narrow["bands"] = [{**narrow["bands"][0], "min_nm": "400", "max_nm": "900"}]
    doc = two_mirror_cavity("200", "200", 100.0, coating=narrow)
    for model in doc["optical_models"]:
        model["model"]["valid_band_nm"]["max_nm"] = "900"  # mirrors characterized from 400 to 900 nm, analysed at 1064 nm
    single = with_models(project([source("S", (-50.0, 0.0, 0.0), wavelength_nm=LAMBDA_NM), declared_component("M", "mirror", "m1", (0.0, 0.0, 0.0), AGAINST_X)],
                                 [plane("p", (-25.0, 0.0, 0.0))]), [doc["optical_models"][0]], coatings=[HR, narrow])
    forward, back = (arrival.beam.power_W for arrival in arrivals(run(single).result, "p"))
    traced = back / forward
    assert traced == pytest.approx(R_HR, rel=1e-12)
    analysis = _q(doc)
    fsr = C_MM_PER_S / 200.0
    expected = {"reflectance.M1": traced, "reflectance.M2": traced, "round_trip_power_fraction": traced * traced,
                "finesse": math.pi * math.sqrt(traced) / (1.0 - traced),
                "linewidth_fwhm_Hz": 2.0 * fsr / math.pi * math.asin((1.0 - traced) / (2.0 * math.sqrt(traced)))}
    for quantity_id, value in expected.items():
        quantity = analysis.quantity(quantity_id)
        assert quantity.state.value == "UNKNOWN" and quantity.reason == "REFLECTANCE_OUTSIDE_COATING_DATA", quantity_id
        assert quantity.value == pytest.approx(value, rel=1e-12) and quantity.value_basis.kind.value == "ESTIMATE"
    entry = analysis.quantity("reflectance.M1").value_basis
    assert [(a.input.value, a.at, a.code) for a in entry.assumed] == [("HELD_AT_NEAREST_STATED_POINT", ("M1",), "REFLECTANCE_OUTSIDE_COATING_DATA")]
    (row,) = entry.validity
    assert (row.parameter, row.minimum, row.maximum, row.observed, row.within, row.at) == ("coating_wavelength_nm", 400.0, 900.0, LAMBDA_NM, False, ("M1",))
    assert analysis.quantity("reflectance.M1").missing == ("reflectances.M1",)
    assert [a.at for a in analysis.quantity("finesse").value_basis.assumed] == [("M1",), ("M2",)]
    injected = _q(doc, {"injection": {"power_w": 1.0, "beam": "EIGENMODE", "detuning_Hz": 0.0}})
    assert injected.quantity("transmittance.M1").state.value == "UNKNOWN" and injected.quantity("transmittance.M1").value_basis.kind.value == "ESTIMATE"
    waiting = injected.quantity("circulating_power_W")
    assert waiting.state.value == "UNKNOWN" and waiting.value is None and waiting.missing == ("reflectances.M1",)
    inside = cavity_analysis(doc, {"components": ["M1", "M2"], "wavelength_nm": 800.0})
    assert _value(inside, "reflectance.M1") == R_HR and inside.quantity("reflectance.M1").value_basis is None
    assert _value(inside, "finesse") == pytest.approx(math.pi * math.sqrt(R_HR) / (1.0 - R_HR), rel=1e-12)
    assert _value(analysis, "free_spectral_range_Hz") == pytest.approx(C_MM_PER_S / 200.0, rel=1e-12)
    assert analysis.execution.outcome.value == "COMPLETED" and analysis.execution.complete
    stated = _q(doc, {"reflectances": {"M1": 0.99, "M2": 0.99}})
    assert stated.execution.complete and _value(stated, "finesse") == pytest.approx(math.pi * math.sqrt(0.99) / 0.01, rel=1e-12)


def test_a_mode_the_apertures_clip_keeps_its_paraxial_values_as_estimates_that_name_the_condition():
    """Clear radius 0.4 mm on mirrors where the mode is 0.198 mm: about 2 mode radii, below 3. The round-trip matrix still has its
    eigenmode: the mode quantities carry the g-parameter values, outside the model's validity (OUT_OF_MODEL_SCOPE), each an ESTIMATE
    whose basis states the failed condition: the ratio 0.4 / w against the limit 3, the setting that holds the limit and the mirror.
    Such a value decides no target. A request that accepts 2 mode radii reads the same numbers as KNOWN and names the setting."""

    doc = two_mirror_cavity("200", "200", 100.0, clear="0.4")
    analysis = _q(doc, {"targets": [{"target_id": "w", "quantity_id": "mode_radius_mm.M1", "maximum": 0.25}]})
    ref = _g_reference(200.0, 200.0, 100.0)
    w = ref["w1"]
    radius = analysis.quantity("mode_radius_mm.M1")
    assert radius.state.value == "OUT_OF_MODEL_SCOPE" and radius.reason == "MODE_CLIPPED_BY_APERTURE" and "'M1'" in radius.message
    assert radius.value == pytest.approx(w, rel=1e-11) and radius.value_basis.kind.value == "ESTIMATE"
    (row,) = radius.value_basis.validity
    assert (row.parameter, row.within, row.minimum, row.setting, row.at, row.code) == (
        "clear_aperture_over_mode_radius", False, 3.0, "aperture_over_mode_min", ("M1",), "MODE_CLIPPED_BY_APERTURE")
    assert row.observed == pytest.approx(0.4 / w, rel=1e-11) and analysis.settings["limits"]["aperture_over_mode_min"] == 3.0
    fsr = C_MM_PER_S / 200.0
    for quantity_id, expected in (("gap_1.waist_radius_mm", ref["w0"]), ("gap_1.waist_along_axis_mm", ref["z1"]),
                                  ("round_trip_gouy_phase_rad", ref["gouy"]), ("transverse_mode_spacing_Hz", fsr * ref["gouy"] / (2.0 * math.pi))):
        quantity = analysis.quantity(quantity_id)
        assert quantity.state.value == "OUT_OF_MODEL_SCOPE" and quantity.value == pytest.approx(expected, rel=1e-11), quantity_id
        assert quantity.value_basis == radius.value_basis
    other = analysis.quantity("mode_radius_mm.M2")  # (the other mirror: the failed condition, and its own aperture over its mode radius)
    assert [(r.at, r.within) for r in other.value_basis.validity] == [(("M1",), False), (("M2",), False)]
    assert other.value_basis.validity[1].observed == pytest.approx(0.4 / ref["w2"], rel=1e-11)
    assert analysis.arrays["waists"][0]["inside_gap"] is True and analysis.arrays["waists"][0]["point_mm"] == pytest.approx([50.0, 0.0, 0.0], abs=1e-9)
    (target,) = analysis.arrays["targets"]  # (0.198 mm is under the maximum 0.25 mm: the estimate does not decide it)
    assert target["assessment"] == "INDETERMINATE" and target["reason"] == "MODE_CLIPPED_BY_APERTURE"
    assert target["value"] == radius.value and target["value_basis"]["kind"] == "ESTIMATE"
    check = next(c for c in analysis.models[0].validity if c.parameter == "clear_aperture_over_mode_radius")
    assert check.within is False and check.observed == pytest.approx(0.4 / w, rel=1e-11)
    assert check == row
    relaxed = _q(doc, {"aperture_over_mode_min": 2.0, "targets": [{"target_id": "w", "quantity_id": "mode_radius_mm.M1", "maximum": 0.25}]})
    accepted = relaxed.quantity("mode_radius_mm.M1")
    assert accepted.state.value == "KNOWN" and accepted.value == radius.value and accepted.value_basis.kind.value == "MODEL"
    (row,) = accepted.value_basis.validity
    assert (row.within, row.minimum, row.setting, row.code) == (True, 2.0, "aperture_over_mode_min", None)
    assert relaxed.settings["limits"]["aperture_over_mode_min"] == 2.0 and relaxed.arrays["targets"][0]["assessment"] == "MET"
    assert relaxed.quantity("finesse").state.value == "KNOWN"
    assert relaxed.identity.physical.settings_hash != analysis.identity.physical.settings_hash
    finesse = analysis.quantity("finesse")
    assert finesse.state.value == "BOUNDED" and finesse.value == finesse.interval.upper == pytest.approx(math.pi * math.sqrt(R_HR) / (1.0 - R_HR), rel=1e-12)
    assert finesse.value_basis.kind.value == "UPPER_BOUND" and finesse.value_basis.assumed[0].code == "MODE_CLIPPED_BY_APERTURE"
    wide = _q(two_mirror_cavity("200", "200", 100.0))
    inside = next(c for c in wide.models[0].validity if c.parameter == "clear_aperture_over_mode_radius")
    assert inside.within is True and inside.setting is None and inside.at == () and wide.quantity("mode_radius_mm.M1").value_basis is None


def test_the_paraxial_limits_of_the_eigenmode_are_the_project_s_settings():
    """R = 200 mm mirrors 100 mm apart: the mode's far-field half angle is lambda / (pi w0) = 1.84 mrad, far inside the Engine's limit
    of 0.3 rad. A Project that sets gaussian_scope_divergence_rad to 1 mrad puts it outside: the mode quantities keep the g-parameter
    values as estimates that state the angle, the Project's limit and the setting; the resonances, which read the mirrors, stand."""

    doc = {**two_mirror_cavity("200", "200", 100.0), "numerical_settings": {"gaussian_scope_divergence_rad": 1e-3, "gaussian_warn_divergence_rad": 5e-4}}
    ref = _g_reference(200.0, 200.0, 100.0)
    plain = _q(two_mirror_cavity("200", "200", 100.0))
    analysis = _q(doc, {"injection": {"power_w": 1.0, "beam": "EIGENMODE", "detuning_Hz": 0.0}, "transmittances": {"M1": 0.009, "M2": 0.009},
                        "reflectances": {"M1": 0.99, "M2": 0.99}})
    waist = analysis.quantity("gap_1.waist_radius_mm")
    assert waist.state.value == "OUT_OF_MODEL_SCOPE" and waist.reason == "MODE_OUTSIDE_PARAXIAL_LIMITS"
    assert waist.value == _value(plain, "gap_1.waist_radius_mm") == pytest.approx(ref["w0"], rel=1e-11)
    (row,) = waist.value_basis.validity
    assert waist.value_basis.kind.value == "ESTIMATE" and (row.parameter, row.within, row.maximum, row.setting, row.code) == (
        "far_field_half_angle_rad", False, 1e-3, "gaussian_scope_divergence_rad", "MODE_OUTSIDE_PARAXIAL_LIMITS")
    assert row.observed == pytest.approx(LAMBDA_MM / (math.pi * ref["w0"]), rel=1e-11)
    assert analysis.settings["limits"]["gaussian_scope_divergence_rad"] == 1e-3 and plain.settings["limits"]["gaussian_scope_divergence_rad"] == 0.3
    # the mirrors' quantities stand; no mode of the injected light is resolved, so its powers are bounded from zero
    assert _value(analysis, "free_spectral_range_Hz") == _value(plain, "free_spectral_range_Hz")
    assert _value(analysis, "resonant_power_build_up") == pytest.approx(0.009 / (1.0 - 0.99) ** 2, rel=1e-12)
    circulating = analysis.quantity("circulating_power_W")
    assert circulating.state.value == "BOUNDED" and circulating.reason == "MODE_OUTSIDE_PARAXIAL_LIMITS" and circulating.value is None
    assert circulating.interval.lower == 0.0 and circulating.interval.upper == pytest.approx(0.009 / (1.0 - 0.99) ** 2, rel=1e-12)


def _overlap(w1: float, w2: float, inverse_r1: float, inverse_r2: float) -> float:
    """Power overlap of two coaxial TEM00 beams from their radii and wavefront curvatures at one plane."""

    return 4.0 / ((w1 / w2 + w2 / w1) ** 2 + (math.pi * w1 * w2 / LAMBDA_MM) ** 2 * (inverse_r1 - inverse_r2) ** 2)


def test_mode_matching_of_a_stated_input_beam():
    """Symmetric R = 200, L = 100: the mode's waist w0 is 50 mm inside. A beam with that waist there couples fully; a beam with its waist on the mirror
    meets the mode's radius and curvature there; an elliptical beam couples the product of its two one-dimensional overlaps."""

    doc = two_mirror_cavity("200", "200", 100.0)
    w0 = _g_reference(200.0, 200.0, 100.0)["w0"]
    matched = _q(doc, {"input_beam": {"waist_radius_mm": w0, "waist_after_reference_mm": 50.0}})
    assert _value(matched, "mode_matching_efficiency") == pytest.approx(1.0, rel=1e-12)
    larger = _q(doc, {"input_beam": {"waist_radius_mm": 1.3 * w0, "waist_after_reference_mm": 50.0}})
    assert _value(larger, "mode_matching_efficiency") == pytest.approx(_overlap(1.3 * w0, w0, 0.0, 0.0), rel=1e-12)
    z_r = math.pi * w0 * w0 / LAMBDA_MM
    q_mode = complex(-50.0, z_r)
    w_mode = math.sqrt(-LAMBDA_MM / (math.pi * (1.0 / q_mode).imag))
    at_mirror = _q(doc, {"input_beam": {"waist_radius_mm": 0.15, "waist_after_reference_mm": 0}})
    assert _value(at_mirror, "mode_matching_efficiency") == pytest.approx(_overlap(0.15, w_mode, 0.0, (1.0 / q_mode).real), rel=1e-12)
    elliptical = _q(doc, {"input_beam": {"waist_radius_mm": 1.2 * w0, "waist_radius_y_mm": 0.8 * w0, "waist_after_reference_mm": 50.0}})
    one_d = lambda a: 2.0 * a / (1.0 + a * a)  # noqa: E731
    assert _value(elliptical, "mode_matching_efficiency") == pytest.approx(one_d(1.2) * one_d(0.8), rel=1e-12)
    unstable = _q(two_mirror_cavity("50", "50", 120.0), {"input_beam": {"waist_radius_mm": 0.1, "waist_after_reference_mm": 0}})
    assert unstable.quantity("mode_matching_efficiency").state.value == "OUT_OF_MODEL_SCOPE"


# ------------------------------------------------------------------------------------------------ identity, refusals, surfaces


def test_the_document_is_bound_to_the_project_and_its_settings():
    doc = two_mirror_cavity("200", "200", 100.0)
    analysis = _q(doc)
    assert analysis.identity.physical.engine_input_hash == lower_project(doc, None, check_solids=False, require_source=False).engine_input_hash
    assert analysis.identity.physical.run_result_hash is None
    assert analysis.settings["components"] == ["M1", "M2"] and analysis.settings["wavelength_nm"] == LAMBDA_NM
    other = cavity_analysis(doc, {"components": ["M1", "M2"], "wavelength_nm": 633})
    assert other.identity.physical.settings_hash != analysis.identity.physical.settings_hash
    assert {m.model_id for m in analysis.models} == {q.basis for q in analysis.quantities}


def _problems(doc: dict, request: dict, **kwargs) -> list[dict]:
    with pytest.raises(CavityError) as info:
        cavity_analysis(doc, request, **kwargs)
    assert info.value.as_dict()["code"] == "CAVITY_ANALYSIS_INVALID"
    return list(info.value.problems)


def test_refusals_are_structured():
    doc = two_mirror_cavity("200", "200", 100.0, extra=(declared_component("L", "lens", "f100", (50.0, 0.0, 0.0), ALONG_X),))
    doc["optical_models"].append(thin_lens_model("f100", "100"))
    base = {"wavelength_nm": LAMBDA_NM}
    assert [p["code"] for p in _problems(doc, {**base, "components": ["L", "M2"]})] == ["CAVITY_ELEMENT_UNSUPPORTED"]
    assert [p["code"] for p in _problems(doc, {**base, "components": ["M1", "M2", "M1"]})] == ["PARAXIAL_REQUEST_INVALID"]
    assert [p["code"] for p in _problems(doc, {**base, "components": ["M1"]})] == ["PARAXIAL_REQUEST_INVALID"]
    assert [p["code"] for p in _problems(doc, {**base, "components": ["M1", "M2"], "extra": 1})] == ["PARAXIAL_REQUEST_INVALID"]
    assert [p["code"] for p in _problems(doc, {**base, "components": ["M1", "L", "M2"], "reflectances": {"L": 0.5}})] == ["PARAXIAL_REQUEST_INVALID"]
    assert [p["code"] for p in _problems(doc, {**base, "components": ["M1", "M2"], "reflectances": {"M1": 1}})] == ["PARAXIAL_REQUEST_INVALID"]
    assert [p["code"] for p in _problems(doc, {**base, "components": ["M1", "M2"], "input_beam": {"waist_radius_mm": 1}})] == ["PARAXIAL_REQUEST_INVALID"]
    turned = copy.deepcopy(doc)
    turned["components"][1]["nominal_pose"]["quaternion_wxyz"] = list(ALONG_X)
    assert [p["code"] for p in _problems(turned, {**base, "components": ["M1", "M2"]})] == ["PARAXIAL_REFLECTOR_BACK_FACE"]
    # a plane end mirror at 45 deg does not send the light back
    tilted = project([element("M1", "mirror", FLAT_MIRROR, (0.0, 0.0, 0.0), FOLD_X_TO_Y), declared_component("M2", "mirror", "m2", (0.0, 100.0, 0.0), AGAINST_Y)],
                     [plane("p", (0.0, 50.0, 0.0))])
    tilted = with_models(tilted, [_mirror("m2", "200")], coatings=[HR])
    (problem,) = _problems(tilted, {**base, "components": ["M1", "M2"]}, catalog=catalog_document())
    assert problem["code"] == "CAVITY_NOT_CLOSED" and "'M1'" in problem["message"]
    refused = json_api.cavity_analysis(doc, {**base, "components": ["L", "M2"]})
    assert refused["ok"] is False and refused["error"]["code"] == "CAVITY_ANALYSIS_INVALID"


def test_a_source_in_the_project_does_not_change_the_cavity():
    """The cavity's light is its own round trip: a source on the axis (the run's injection) leaves the analysis unchanged."""

    plain = _q(two_mirror_cavity("200", "200", 100.0))
    lit = _q(two_mirror_cavity("200", "200", 100.0, extra=(source(position=(-50.0, 0.0, 0.0), wavelength_nm=LAMBDA_NM),)))
    assert [q.model_dump() for q in lit.quantities] == [q.model_dump() for q in plain.quantities]


def _run_cli(doc: dict, tmp_path, *args: str) -> tuple[int, dict]:
    path = tmp_path / "cavity.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    out = io.StringIO()
    with redirect_stdout(out):
        code = cli_main(["project", "cavity", str(path), *args])
    return code, json.loads(out.getvalue())


def test_the_cli_exit_codes_follow_the_analysis(tmp_path):
    code, reply = _run_cli(two_mirror_cavity("100", "100", 100.0), tmp_path, "--components", "M1", "M2", "--wavelength-nm", "1064")
    assert code == 0 and reply["analysis"] == "CAVITY" and reply["schema"] == "opensci.analysis_result"
    code, reply = _run_cli(two_mirror_cavity("50", "50", 120.0), tmp_path, "--components", "M1", "M2", "--wavelength-nm", "1064")
    assert code == 0 and reply["execution"]["outcome"] == "COMPLETED"  # (an unstable cavity: answered, its eigenmode rows say why)
    states = {q["quantity_id"]: (q["state"], q.get("reason")) for q in reply["quantities"]}
    assert states["mode_radius_mm.M1"] == ("OUT_OF_MODEL_SCOPE", "CAVITY_UNSTABLE")
    unstable = {q["quantity_id"]: q for q in reply["quantities"]}["round_trip_gouy_phase_rad"]
    assert unstable["state"] == "OUT_OF_MODEL_SCOPE" and unstable["reason"] == "CAVITY_UNSTABLE" and unstable["value"] is None
    code, reply = _run_cli(two_mirror_cavity("100", "100", 100.0), tmp_path, "--components", "M1", "M1", "--wavelength-nm", "1064")
    assert code == 1 and reply["status"] == "ERROR" and reply["error"]["code"] == "CAVITY_ANALYSIS_INVALID"
    code, reply = _run_cli(two_mirror_cavity("100", "100", 100.0), tmp_path, "--components", "M1", "M2", "--wavelength-nm", "1064",
                           "--reflectance", "M1=0.95", "--input-waist-mm", "0.13", "--input-waist-after-mm", "50")
    quantities = {q["quantity_id"]: q for q in reply["quantities"]}
    assert code == 0 and quantities["reflectance.M1"]["value"] == 0.95 and 0.99 < quantities["mode_matching_efficiency"]["value"] <= 1.0
    path = tmp_path / "cavity.json"
    out = io.StringIO()
    with redirect_stdout(out):
        assert cli_main(["project", "cavity", str(path), "--components", "M1", "M2", "--wavelength-nm", "1064", "--reflectance", "M1"]) == 64


def test_json_api_and_capabilities():
    reply = json_api.cavity_analysis(two_mirror_cavity("200", "200", 100.0), {"components": ["M1", "M2"], "wavelength_nm": LAMBDA_NM})
    assert reply["ok"] is True and reply["analysis"] == "CAVITY"
    caps = json_api.capabilities()
    assert caps["cavity_analysis"]["analysis"] == "CAVITY" and caps["cavity_analysis"]["result_schema_kind"] == "analysis-result"
    assert json_api.json_schema("analysis-result")["schema"]["title"] == "AnalysisResultV1"


def test_the_paraxial_analysis_points_a_repeated_mirror_to_the_cavity_analysis():
    reply = json_api.paraxial_analysis(two_mirror_cavity("200", "200", 100.0), {"components": ["M1", "M2", "M1"], "wavelength_nm": LAMBDA_NM})
    problem = reply["error"]["problems"][0]
    assert problem["code"] == "PARAXIAL_REFLECTOR_REPEATED" and "project cavity" in problem["message"]


def test_the_catalog_mirror_pair_of_a_two_mirror_resonator():
    """Two catalog concave mirrors (R = 200 mm) 150 mm apart, by hand: g = 0.25, w on the mirrors (lambda L / pi)^(1/2) (1 / (1 - g^2))^(1/4), one-way Gouy
    acos(0.25), FSR c / 300 mm, finesse from the catalog coating's R = 0.99."""

    from local_helpers import CONCAVE_MIRROR

    doc = project([element("M1", "mirror", CONCAVE_MIRROR, (0.0, 0.0, 0.0), ALONG_X), element("M2", "mirror", CONCAVE_MIRROR, (150.0, 0.0, 0.0), AGAINST_X)],
                  [plane("p", (75.0, 0.0, 0.0))])
    analysis = cavity_analysis(doc, {"components": ["M1", "M2"], "wavelength_nm": 633}, catalog=catalog_document())
    lam, g = 633e-6, 0.25
    assert _value(analysis, "g1") == pytest.approx(g, rel=1e-12) and _value(analysis, "mode_radius_mm.M1") == pytest.approx(math.sqrt(lam * 150.0 / math.pi) * (1.0 / (1.0 - g * g)) ** 0.25,
                                                                                                 rel=1e-12)
    assert _value(analysis, "round_trip_gouy_phase_rad") == pytest.approx(2.0 * math.acos(g), rel=1e-12)
    assert _value(analysis, "free_spectral_range_Hz") == pytest.approx(C_MM_PER_S / 300.0, rel=1e-12)
    assert _value(analysis, "finesse") == pytest.approx(math.pi * math.sqrt(0.99) / 0.01, rel=1e-12)
    assert analysis.settings["catalog_version"] == TEST_CATALOG_VERSION


def test_a_stated_coaxial_tolerance_accepts_a_slightly_misaligned_cavity_and_the_mode_names_it():
    """M2 2e-6 mm off the axis: refused by default (the paraxial placement's tolerance), answered on the nominal axis with a
    stated coaxial_tolerance -- the aligned cavity's numbers, each eigenmode quantity KNOWN with a MODEL basis whose rows state
    the measured decentre, the tolerance that accepted it and the mirror. A curved end mirror tilted by 2e-6 rad is refused
    whatever the stated tilt tolerance (a tilted curved mirror has no single 2 x 2 matrix); a lens inside a plane-plane cavity
    tilted by 2e-6 rad is accepted with a stated tilt_rad and named in the element_tilt_rad row. arrays.elements and
    settings.limits state the same."""

    aligned = _q(two_mirror_cavity("200", "200", 100.0))
    assert [(e["decentre_mm"], e["tilt_rad"]) for e in aligned.arrays["elements"]] == [(0.0, 0.0), (0.0, 0.0)]
    assert aligned.settings["limits"]["coaxial_tolerance.lateral_mm"] == 1e-6 and aligned.quantity("mode_radius_mm.M1").value_basis is None
    moved = two_mirror_cavity("200", "200", 100.0)
    moved["components"][1]["nominal_pose"]["position_mm"] = [100.0, 2e-6, 0.0]
    (problem,) = _problems(moved, {"components": ["M1", "M2"], "wavelength_nm": LAMBDA_NM})
    assert problem["code"] == "PARAXIAL_NOT_COAXIAL" and "coaxial_tolerance.lateral_mm is 1e-06 mm" in problem["message"]
    analysis = _q(moved, {"coaxial_tolerance": {"lateral_mm": 1e-5}})
    for quantity_id in ("mode_radius_mm.M1", "mode_radius_mm.M2", "gap_1.waist_radius_mm", "round_trip_gouy_phase_rad"):
        quantity = analysis.quantity(quantity_id)
        assert quantity.state.value == "KNOWN" and quantity.value == pytest.approx(_value(aligned, quantity_id), rel=1e-9)
        assert quantity.value_basis.kind.value == "MODEL"
        rows = {row.parameter: row for row in quantity.value_basis.validity}
        assert rows["element_decentre_mm"].setting == "coaxial_tolerance.lateral_mm" and rows["element_decentre_mm"].maximum == 1e-5
        assert rows["element_decentre_mm"].observed == pytest.approx(2e-6, rel=1e-6) and rows["element_decentre_mm"].at == ("M2",)
        assert all(row.within is True for row in rows.values())
    assert _value(analysis, "finesse") == _value(aligned, "finesse")
    assert analysis.arrays["elements"][1]["decentre_mm"] == pytest.approx(2e-6, rel=1e-6) and analysis.arrays["elements"][1]["tilt_rad"] == 0.0
    assert analysis.settings["limits"]["coaxial_tolerance.lateral_mm"] == 1e-5
    named = {row.setting for quantity in analysis.quantities if quantity.value_basis is not None for row in quantity.value_basis.validity if row.setting}
    assert named <= set(analysis.settings["limits"])  # (every setting a value names is in the record of the limits in force)
    # the curved end mirror tilted by 2e-6 rad: refused with or without a stated tilt tolerance
    half = 1e-6
    moved["components"][1]["nominal_pose"]["quaternion_wxyz"] = list(quat_multiply((math.cos(half), 0.0, 0.0, math.sin(half)), AGAINST_X))
    for stated in ({"lateral_mm": 1e-5}, {"lateral_mm": 1e-5, "tilt_rad": 5e-6}):
        (problem,) = _problems(moved, {"components": ["M1", "M2"], "wavelength_nm": LAMBDA_NM, "coaxial_tolerance": stated})
        assert problem["code"] == "PARAXIAL_CURVED_REFLECTOR_TILTED" and "whatever coaxial_tolerance says" in problem["message"], stated
    # a lens tilted by 2e-6 rad in a plane-lens-plane cavity: refused by default, accepted on the nominal axis with a stated tilt
    request = {"components": ["M1", "L", "M2"], "wavelength_nm": LAMBDA_NM}
    straight = cavity_analysis(_lens_cavity(thin_lens_model("f100", "100", clear="12"), 50.0, 130.0), request)
    tilted = _lens_cavity(thin_lens_model("f100", "100", clear="12"), 50.0, 130.0)
    tilted["components"][1]["nominal_pose"]["quaternion_wxyz"] = list(quat_multiply((math.cos(half), 0.0, 0.0, math.sin(half)), ALONG_X))
    (problem,) = _problems(tilted, request)
    assert problem["code"] == "PARAXIAL_NOT_COAXIAL" and "coaxial_tolerance.tilt_rad" in problem["message"]
    analysis = cavity_analysis(tilted, {**request, "coaxial_tolerance": {"tilt_rad": 5e-6}})
    for quantity_id in ("mode_radius_mm.M1", "mode_radius_mm.M2", "round_trip_gouy_phase_rad"):
        quantity = analysis.quantity(quantity_id)
        assert quantity.state.value == "KNOWN" and quantity.value == pytest.approx(_value(straight, quantity_id), rel=1e-9)
        rows = {row.parameter: row for row in quantity.value_basis.validity}
        assert quantity.value_basis.kind.value == "MODEL" and rows["element_tilt_rad"].setting == "coaxial_tolerance.tilt_rad"
        assert rows["element_tilt_rad"].observed == pytest.approx(2e-6, rel=1e-6) and rows["element_tilt_rad"].at == ("L",)
        assert rows["element_tilt_rad"].maximum == 5e-6 and rows["element_tilt_rad"].within is True
    assert analysis.arrays["elements"][1]["tilt_rad"] == pytest.approx(2e-6, rel=1e-6)
    assert analysis.settings["limits"]["coaxial_tolerance.tilt_rad"] == 5e-6


def test_the_tilt_tolerance_texts_state_that_a_curved_mirror_is_analysed_only_at_normal_incidence(capsys):
    """The cavity request's coaxial_tolerance and the command's --tilt-tolerance-rad say what the refusal above does: tilt_rad
    admits a plane end mirror, a source and a transmissive element off the axis, never a curved mirror."""
    from os_optics.cli import main
    from os_optics.local import cavity as module

    text = " ".join((module.__doc__ or "").split())
    assert "a plane end mirror's normal" in text and "A curved mirror is analysed only at normal incidence" in text
    assert "PARAXIAL_CURVED_REFLECTOR_TILTED" in text
    with pytest.raises(SystemExit):
        main(["project", "cavity", "--help"])
    assert "a curved mirror only at normal incidence, whatever this says" in " ".join(capsys.readouterr().out.split())
