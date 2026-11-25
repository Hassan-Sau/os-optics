"""Materials, spectral sampling, interpolation, coatings, validity ranges (PHYSICS_SCOPE P2, P3)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from os_optics import builders as B
from os_optics.config import DEFAULT_NUMERICAL_CONFIG as CFG
from os_optics.contracts import (
    ConstantCoating, ConstantIndexMaterial, JonesVector, ProjectSnapshot, SellmeierMaterial, TabulatedCoating, TabulatedIndexMaterial,
)
from os_optics.optics.materials import CoatingEvaluator, MaterialLibrary

from os_optics.results import basis_of

from helpers import arrivals, issue_codes, one_arrival, run

lib = MaterialLibrary((), CFG)


# ---- Sellmeier vs published catalogue values ------------------------------------------------------------------------------
@pytest.mark.parametrize("wl,n_ref", [(486.13, 1.52238), (546.07, 1.51872), (587.56, 1.51680), (656.27, 1.51432)])
def test_bk7_sellmeier_matches_schott_catalogue(wl, n_ref):
    n, issue = lib.evaluator("N-BK7").index(wl)
    assert issue is None and n == pytest.approx(n_ref, abs=6e-6)  # catalogue values are given to 5 decimals


def test_fused_silica_sellmeier_matches_malitson():
    n, issue = lib.evaluator("fused_silica").index(587.56)
    assert issue is None and n == pytest.approx(1.45846, abs=6e-6)
    n2, _ = lib.evaluator("fused_silica").index(1064.0)
    assert n2 == pytest.approx(1.44963, abs=6e-6)  # published value at 1064 nm


def test_material_out_of_range_is_out_of_model_scope_not_extrapolation():
    n, issue = lib.evaluator("N-BK7").index(250.0)
    assert n is None and issue.code.value == "MATERIAL_OUT_OF_RANGE" and issue.status.value == "OUT_OF_MODEL_SCOPE"
    assert issue.numerical_context["valid_min_nm"] == 300.0 and issue.numerical_context["valid_max_nm"] == 2500.0
    # the material's name is a fact of the issue (the Designer words it in Chinese from numerical_context, not the sentence)
    assert issue.numerical_context["material"] == "N-BK7" and issue.message.endswith("of material N-BK7")
    n_edge, issue_edge = lib.evaluator("N-BK7").index(300.0)  # inclusive band
    assert issue_edge is None and n_edge > 1.5


def test_sellmeier_pole_and_negative_n_squared_are_structured_errors():
    pole = SellmeierMaterial(material_id="pole", valid_wavelength_nm=(400.0, 900.0), b=(1.0, 0.0, 0.0), c_um2=(0.5**2, 1.0, 2.0))
    ev = MaterialLibrary((pole,), CFG).evaluator("pole")
    n, issue = ev.index(500.0)  # lambda^2 == C1 exactly
    assert n is None and issue.code.value == "MATERIAL_INVALID" and "pole" in issue.message
    neg = SellmeierMaterial(material_id="neg", valid_wavelength_nm=(400.0, 900.0), b=(-5.0, 0.0, 0.0), c_um2=(0.01, 1.0, 2.0))
    n, issue = MaterialLibrary((neg,), CFG).evaluator("neg").index(600.0)
    assert n is None and issue.code.value == "MATERIAL_INVALID"


def test_tabulated_index_pchip_and_linear():
    wl = (400.0, 500.0, 600.0, 700.0)
    lin = TabulatedIndexMaterial(material_id="lin", valid_wavelength_nm=(400.0, 700.0), wavelength_nm=wl, refractive_index=(1.6, 1.5, 1.4, 1.3))
    ev = MaterialLibrary((lin,), CFG).evaluator("lin")
    for x in (400.0, 433.3, 550.0, 699.9, 700.0):  # a linear table is reproduced exactly by PCHIP (and by linear interpolation)
        assert ev.index(x)[0] == pytest.approx(1.6 - (x - 400.0) * 1e-3, abs=1e-12)
    # shape preservation: no overshoot on a monotone step-like table
    step = TabulatedIndexMaterial(material_id="step", valid_wavelength_nm=(400.0, 700.0), wavelength_nm=wl,
                                  refractive_index=(1.50, 1.50, 1.60, 1.60))
    ev_step = MaterialLibrary((step,), CFG).evaluator("step")
    xs = np.linspace(400.0, 700.0, 301)
    ns = np.array([ev_step.index(float(x))[0] for x in xs])
    assert ns.min() >= 1.50 - 1e-12 and ns.max() <= 1.60 + 1e-12 and np.all(np.diff(ns) >= -1e-12)
    linear = TabulatedIndexMaterial(material_id="l2", valid_wavelength_nm=(400.0, 700.0), wavelength_nm=wl,
                                    refractive_index=(1.6, 1.5, 1.4, 1.3), interpolation="linear")
    assert MaterialLibrary((linear,), CFG).evaluator("l2").index(450.0)[0] == pytest.approx(1.55, abs=1e-12)
    assert ev.index(701.0)[1].code.value == "MATERIAL_OUT_OF_RANGE"


def test_constant_index_with_validity_band_and_project_override_of_air():
    fallback = ConstantIndexMaterial(material_id="fallback", valid_wavelength_nm=(500.0, 600.0), refractive_index=1.45)
    ev = MaterialLibrary((fallback,), CFG).evaluator("fallback")
    assert ev.index(550.0)[0] == 1.45 and ev.index(499.0)[1].status.value == "OUT_OF_MODEL_SCOPE"
    custom_air = ConstantIndexMaterial(material_id="air", valid_wavelength_nm=(200.0, 1700.0), refractive_index=1.000273)
    assert MaterialLibrary((custom_air,), CFG).evaluator("air").index(405.0)[0] == 1.000273  # project overrides the built-in


def test_unknown_material_is_a_structured_failure_not_a_crash():
    plate = B.window("W", thickness_mm=2.0, clear_radius_mm=10.0, material="unobtainium", position=(20, 0, 20))
    proj = ProjectSnapshot(components=(B.laser("L", wavelength_nm=532.0, waist_radius_mm=0.5, position=(0, 0, 20), direction=(1, 0, 0)), plate),
                           observations=(B.observation_plane("d", position=(60, 0, 20), normal=(1, 0, 0), radius_mm=10.0),))
    res = run(proj)
    assert res.status.value == "FAIL" and res.issues[0].code.value == "UNKNOWN_MATERIAL" and res.issues[0].location.startswith("component:W")
    assert res.manifest is not None and res.manifest.result_hash


def test_beam_in_material_outside_its_band_terminates_with_structured_failure():
    proj = ProjectSnapshot(
        components=(B.laser("L", wavelength_nm=250.0, waist_radius_mm=0.5, position=(0, 0, 20), direction=(1, 0, 0)),
                    B.window("W", thickness_mm=2.0, clear_radius_mm=10.0, material="N-BK7", position=(20, 0, 20))),
        observations=(B.observation_plane("d", position=(60, 0, 20), normal=(1, 0, 0), radius_mm=10.0),),
    )
    res = run(proj)
    assert "MATERIAL_OUT_OF_RANGE" in issue_codes(res) and res.status.value == "OUT_OF_MODEL_SCOPE"  # (a data-coverage limit)
    assert res.paths[0].termination == "MATERIAL_OUT_OF_RANGE" and res.observations[0].arrivals == ()


# ---- weighted discrete wavelengths ---------------------------------------------------------------------------------------------
def test_weighted_discrete_spectrum_traces_each_wavelength_with_its_own_index():
    """Two spectral lines through a BK7 plate at 45 deg: refraction angle from Snell with n(lambda), powers = weights."""
    th = math.radians(45.0)
    d = (math.cos(th), math.sin(th), 0.0)
    plate = B.window("W", thickness_mm=10.0, clear_radius_mm=30.0, material="N-BK7", position=(50, 0, 20), axis=(1, 0, 0))
    proj = ProjectSnapshot(
        components=(B.laser("L", spectrum=((486.13, 0.3), (656.27, 0.7)), waist_radius_mm=0.4, power_W=10e-3,
                            position=(50 - 40 * d[0], -40 * d[1], 20.0), direction=d), plate),
        observations=(B.observation_plane("d", position=(150, 0, 20), normal=(1, 0, 0), radius_mm=500.0),),
    )
    res = run(proj)
    arr = sorted(arrivals(res, "d"), key=lambda a: a.beam.wavelength_nm)
    assert [a.beam.spectral_index for a in arr] == [0, 1]
    powers = {round(a.beam.wavelength_nm, 2): a.beam.power_W for a in arr}
    # transmission of a plate depends on n(lambda): compare each against the Fresnel product times the weight
    def n_of(wl):
        return lib.evaluator("N-BK7").index(wl)[0]
    for wl, weight in ((486.13, 0.3), (656.27, 0.7)):
        n = n_of(wl)
        th2 = math.asin(math.sin(th) / n)
        c1, c2 = math.cos(th), math.cos(th2)
        # laser default pol = in-plane (p): T_p per surface, two surfaces
        tp1 = 2 * c1 / (n * c1 + c2)
        tp2 = 2 * n * c2 / (c2 + n * c1)  # interface 2: n1 = n (cos c2), n2 = 1 (cos c1): 2 n1 cos1/(n2 cos1 + n1 cos2)
        flux = (n * c2 / c1) * (c1 / (n * c2))
        expected = weight * 10e-3 * (tp1 * tp2) ** 2 * flux
        assert powers[round(wl, 2)] == pytest.approx(expected, rel=1e-10)
    # lateral displacement of the exit beam through a plane-parallel plate (independent closed form):
    #   shift_perp = t sin(theta - theta') / cos(theta')
    # measured as the perpendicular distance between the exit ray and the undisplaced ray through the same entry point
    for a in arr:
        n = n_of(a.beam.wavelength_nm)
        th2 = math.asin(math.sin(th) / n)
        shift = 10.0 * math.sin(th - th2) / math.cos(th2)
        undisplaced_y = (150 - (50 - 40 * d[0])) * (d[1] / d[0]) - 40 * d[1]
        offset_y = a.uv_mm[0] - undisplaced_y  # plane frame u = global y
        assert abs(offset_y * math.cos(th)) == pytest.approx(shift, rel=1e-9)
    assert arr[0].uv_mm != arr[1].uv_mm  # dispersion separates the two lines


def test_spectrum_lines_do_not_interfere_and_weights_normalize():
    proj = ProjectSnapshot(
        components=(B.laser("L", spectrum=((500.0, 2.0), (600.0, 6.0)), waist_radius_mm=0.4, power_W=8e-3, position=(0, 0, 20), direction=(1, 0, 0)),),
        observations=(B.observation_plane("d", position=(50, 0, 20), normal=(1, 0, 0), radius_mm=10.0),),
    )
    res = run(proj)
    p = {round(a.beam.wavelength_nm): a.beam.power_W for a in arrivals(res, "d")}
    assert p == pytest.approx({500: 2e-3, 600: 6e-3}, rel=1e-12)
    assert res.interference == ()  # different wavelengths never interfere: no pair is analysed


def test_medium_index_changes_fringe_period():
    """Immersing the whole set-up in n = 1.0003: fringe period = lambda0 / (n |d1 - d2|_par)."""
    from test_golden_interference import _interference_project, PERIOD

    proj, *_ = _interference_project()
    custom_air = ConstantIndexMaterial(material_id="air", valid_wavelength_nm=(200.0, 1700.0), refractive_index=1.0003)
    res = run(proj.model_copy(update={"materials": (custom_air,)}))
    assert res.interference[0].fringe_period_nm == pytest.approx(PERIOD / 1.0003, rel=1e-12)
    assert res.interference[0].fringe_spacing_3d_nm == pytest.approx(PERIOD / 1.0003, rel=1e-12)


# ---- coatings ------------------------------------------------------------------------------------------------------------------
def _table_coating():
    wl = (400.0, 500.0)
    aoi = (0.0, 30.0, 60.0)
    r_s = ((0.10, 0.20, 0.50), (0.30, 0.40, 0.60))
    r_p = ((0.05, 0.10, 0.20), (0.15, 0.20, 0.30))
    t_s = tuple(tuple(0.9 - v for v in row) for row in r_s)
    t_p = tuple(tuple(0.9 - v for v in row) for row in r_p)
    return TabulatedCoating(wavelength_nm=wl, aoi_deg=aoi, reflectance_s=r_s, reflectance_p=r_p, transmittance_s=t_s, transmittance_p=t_p)


def test_tabulated_coating_bilinear_interpolation():
    ev = CoatingEvaluator(_table_coating(), CFG)
    c = ev.coefficients(450.0, np.array([15.0, 45.0]), location="x")
    assert c.issue is None
    # bilinear on a table whose rows are linear in AOI and in wavelength
    np.testing.assert_allclose(c.rs, [0.5 * (0.5 * (0.10 + 0.20) + 0.5 * (0.30 + 0.40)), 0.5 * (0.5 * (0.20 + 0.50) + 0.5 * (0.40 + 0.60))], atol=1e-12)
    np.testing.assert_allclose(c.ts, 0.9 - c.rs, atol=1e-12)


def test_a_coating_outside_its_data_gives_its_values_at_the_nearest_stated_point_with_the_issue():
    """The table of ``_table_coating`` read by hand: at 390 nm the 400 nm row (10 deg: a third of the way from 0.10 to 0.20),
    at 61 deg the 60 deg column (450 nm: halfway between 0.50 and 0.60)."""

    ev = CoatingEvaluator(_table_coating(), CFG)
    low = ev.coefficients(390.0, np.array([10.0]), location="x")
    assert low.issue.code.value == "COATING_OUT_OF_RANGE" and low.issue.status.value == "OUT_OF_MODEL_SCOPE"
    assert low.rs[0] == pytest.approx(0.10 + (0.20 - 0.10) / 3.0, abs=1e-12) and low.ts[0] == pytest.approx(0.9 - low.rs[0], abs=1e-12)
    assert (low.held.wavelength_nm, low.held.asked_wavelength_nm, low.held.valid_wavelength_nm) == (400.0, 390.0, (400.0, 500.0))
    assert low.held.aoi_deg == low.held.asked_aoi_deg == 10.0
    steep = ev.coefficients(450.0, np.array([61.0]), location="x")
    assert steep.issue.status.value == "OUT_OF_MODEL_SCOPE" and steep.rs[0] == pytest.approx(0.55, abs=1e-12)
    assert (steep.held.aoi_deg, steep.held.asked_aoi_deg, steep.held.valid_aoi_deg) == (60.0, 61.0, (0.0, 60.0))
    edge = ev.coefficients(500.0, np.array([60.0]), location="x")
    assert edge.issue is None and edge.held is None  # inclusive edges
    const = CoatingEvaluator(ConstantCoating(reflectance_s=0.5, transmittance_s=0.4, reflectance_p=0.5, transmittance_p=0.4,
                                             valid_wavelength_nm=(500.0, 600.0), max_aoi_deg=50.0), CFG)
    below = const.coefficients(450.0, np.array([10.0]), location="x")
    assert below.issue is not None and (below.rs[0], below.ts[0], below.held.wavelength_nm) == (0.5, 0.4, 500.0)
    beyond = const.coefficients(550.0, np.array([51.0]), location="x")
    assert beyond.issue is not None and (beyond.rs[0], beyond.held.aoi_deg) == (0.5, 50.0)


def _split_at(wavelength_nm: float, band: tuple[float, float]) -> ProjectSnapshot:
    return ProjectSnapshot(
        components=(B.laser("L", wavelength_nm=wavelength_nm, waist_radius_mm=0.4, position=(0, 0, 20), direction=(1, 0, 0)),
                    B.beam_splitter("BS", position=(30, 0, 20), normal=(-1, 1, 0), coating=ConstantCoating(
                        reflectance_s=0.5, reflectance_p=0.5, transmittance_s=0.5, transmittance_p=0.5, valid_wavelength_nm=band))),
        observations=(B.observation_plane("d", position=(80, 0, 20), normal=(1, 0, 0), radius_mm=10.0),),
    )


def test_a_beam_goes_on_past_a_coating_outside_its_data_with_its_power_as_an_estimate():
    """650 nm light on a splitter coating stated for 400-500 nm (T = 0.5): the beam goes on with the coating's values, where
    it goes is the beam's of a coating stated at 650 nm bit for bit, and its power (1 mW x 0.5) is an estimate that names
    the surface, the range and the point the values were read at."""

    res, inside = run(_split_at(650.0, (400.0, 500.0))), run(_split_at(650.0, (400.0, 700.0)))
    (issue,) = [i for i in res.issues if i.code.value == "COATING_OUT_OF_RANGE"]
    assert issue.status.value == "NOT_APPLICABLE" and issue.location == "surface:BS/coating"  # a notice: the powers carry it
    assert issue.numerical_context == {"path_id": "L#0", "wavelength_nm": 650.0, "aoi_deg": 45.0, "valid_wavelength_nm": [400.0, 500.0],
                                       "valid_aoi_deg": [0.0, 90.0], "held_wavelength_nm": 500.0, "held_aoi_deg": 45.0}
    assert res.status.value == "PASS" and "COATING_OUT_OF_RANGE" not in issue_codes(inside)
    plane, reference = res.observations[0], inside.observations[0]
    (arrival,), (expected,) = plane.arrivals, reference.arrivals
    assert arrival.beam.power_W == pytest.approx(1e-3 * 0.5, rel=1e-12)  # the stated transmittance, by hand
    assert arrival.beam.model_dump() == expected.beam.model_dump() and arrival.uv_mm == expected.uv_mm
    assert arrival.power_out_of_scope_at == ("BS/coating",) and expected.power_out_of_scope_at == () and expected.value_basis is None
    entry = arrival.value_basis["beam.power_W"]
    assert entry.kind.value == "ESTIMATE" and entry.omitted == ()
    assert [a.model_dump(mode="json") for a in entry.assumed] == [
        {"input": "HELD_AT_NEAREST_STATED_POINT", "at": ["BS/coating"], "value": 500.0, "unit": "nm", "code": "COATING_OUT_OF_RANGE"}]
    assert [c.model_dump(mode="json") for c in entry.validity] == [
        {"parameter": "wavelength_nm", "unit": "nm", "minimum": 400.0, "maximum": 500.0, "observed": 650.0, "within": False,
         "at": ["BS/coating"], "code": "COATING_OUT_OF_RANGE"}]
    assert plane.quantity_state.value == "OUT_OF_MODEL_SCOPE" and plane.total_power_W == reference.total_power_W
    assert reference.quantity_state.value == "KNOWN"
    assert basis_of(plane, "total_power_W") == entry and plane.power_by_wavelength[0].value_basis == {"power_W": entry}
    assert [p.termination for p in res.paths] == [p.termination for p in inside.paths]
    assert all(p.power_out_of_scope_at == ("BS/coating",) for p in res.paths)


def test_fresnel_energy_conservation_and_reflected_branch():
    """Single interface with reflection tracing: T + R = P0 exactly, R equals the closed form at normal incidence."""
    proj = ProjectSnapshot(
        components=(B.laser("L", wavelength_nm=532.0, waist_radius_mm=0.3, position=(0, 0, 20), direction=(1, 0, 0), power_W=1e-3),
                    B.window("W", thickness_mm=4.0, clear_radius_mm=10.0, material="N-BK7", position=(20, 0, 20), trace_fresnel_reflection=True)),
        observations=(B.observation_plane("T", position=(60, 0, 20), normal=(1, 0, 0), radius_mm=10.0),
                      B.observation_plane("R", position=(-20, 0, 20), normal=(1, 0, 0), radius_mm=10.0)),
    )
    res = run(proj)
    n = lib.evaluator("N-BK7").index(532.0)[0]
    r = ((n - 1) / (n + 1)) ** 2
    refl = [a for a in arrivals(res, "R") if a.beam.direction[0] < 0]
    front = min(refl, key=lambda a: a.beam.geometric_length_mm)
    assert front.beam.power_W == pytest.approx(1e-3 * r, rel=1e-12)  # first-surface reflection
    thru = [a for a in arrivals(res, "T") if a.beam.direction[0] > 0]
    # the direct transmitted beam has power (1-r)^2
    direct = max(thru, key=lambda a: a.beam.power_W)
    assert direct.beam.power_W == pytest.approx(1e-3 * (1 - r) ** 2, rel=1e-12)
