"""An acousto-optic deflector as a grating (``deflector.acousto_optic``), and the chain deflector -> pupil -> focal field -> detector, against numbers derived by hand here. Tolerances set
before the first run: directions 1e-9 (exact geometry of the grating relation); powers 1e-12; the focal spots as in ``test_modulator_chain``.

Hand derivation, with a velocity and tones chosen for the test (no real device): v = 650 m/s, 852 nm, tones 25, 27, ... 55 MHz (16 tones, 2 MHz apart, df = 1 MHz):

* the period the model states is v / df = 650 um, and tone f is grating order f / df;
* sin(theta) = lambda f / v: 25 MHz gives 852e-9 x 25e6 / 650 = 0.0327692; spacing lambda 2 MHz / v = 2.62154e-3 in sine;
* a screen 100 mm behind, across the zero order: x = 100 tan(theta), so 3.27868 mm for 25 MHz;
* a lens of f = 200 mm focuses tone f at u = 200 sin(theta): 6.55385 mm for 25 MHz, 524.31 um apart;
* 20 nW through 5 % per tone is 1 nW per tone: w = 0.5 mm focuses to w_f = lambda f / (pi w) = 108.48 um, peak 2 P / (pi w_f^2) = 5.410e-8 W/mm^2;
* a camera pixel of 6.5 um on a peak sees 5.410e-8 x 4.225e-5 = 2.286e-12 W: in 1 ms, 2.286e-12 x 1e-3 / 2.33153e-19 = 9.804e3 photons.
"""

from __future__ import annotations

import json
import math

import pytest

from os_optics.cli import main as cli_main
from os_optics.local import json_api
from os_optics.native_library import NativeComponentError, build_native_model, native_components
from os_optics.results import detector_response, load_run, pupil_field

from declared_helpers import declared_component, nbk7, sequence, surface, with_models
from local_helpers import plane, project, quat_multiply, run, source

LAMBDA_NM = 852.0
LAMBDA_M = 852e-9
VELOCITY = 650.0
TONES_MHZ = [25 + 2 * i for i in range(16)]
ALONG_X = (0.5, 0.5, 0.5, 0.5)
ASSUMED = {"data_kind": "ASSUMED", "basis": "values chosen for this test; no real device"}


def table(rows_per_tone: float = 0.05, zero: float | None = 0.1) -> dict:
    rows = [{"wavelength_nm": "852", "rf_mhz": str(f), "efficiency": str(rows_per_tone)} for f in TONES_MHZ]
    if zero is not None:
        rows.append({"wavelength_nm": "852", "rf_mhz": "0", "efficiency": str(zero)})
    return {"rows": rows, "valid_aoi_deg": ["-2", "2"], **ASSUMED}


def aod(**more) -> dict:
    return {"acoustic_velocity_m_per_s": str(VELOCITY), "rf_tones_mhz": [str(f) for f in TONES_MHZ], "width_mm": "12", "height_mm": "12", **more}


def bench(parameters: dict, observations: list[dict], power_w: float = 20e-9) -> dict:
    build = json_api.build_native_model("deflector.acousto_optic", parameters)
    assert build["ok"] is True, build
    laser = source(position=(-50.0, 0.0, 0.0), wavelength_nm=LAMBDA_NM, power_w=power_w, waist_mm=0.5)
    component = {**build["component_template"], "instance_id": "AOD", "nominal_pose": {"position_mm": [0.0, 0.0, 0.0], "quaternion_wxyz": list(ALONG_X)}}
    return with_models(project([laser, component], observations), [build["optical_model"]])


def test_the_class_compiles_onto_a_grating_of_period_v_over_df():
    build = build_native_model("deflector.acousto_optic", aod(design_wavelength_nm="852")).as_dict()
    model = build["optical_model"]["model"]
    assert build["class_id"] == "deflector.acousto_optic" and model["model_kind"] == "DIFFRACTION_GRATING" and model["mode"] == "TRANSMISSION"
    assert float(model["period_nm"]) == pytest.approx(650e3, rel=1e-15)
    assert model["orders"] == [0, *TONES_MHZ]  # the zero order stays; a tone of f MHz is order f of the 1 MHz grating
    tones = build["design"]["tones"]
    assert [row["rf_mhz"] for row in tones] == [str(f) for f in TONES_MHZ] and [row["grating_order"] for row in tones] == TONES_MHZ
    assert float(tones[0]["optical_frequency_shift_hz"]) == 25e6
    assert float(tones[0]["direction_sine_at_design_wavelength"]) == pytest.approx(0.0327692, rel=1e-5)
    spacing = float(tones[1]["direction_sine_at_design_wavelength"]) - float(tones[0]["direction_sine_at_design_wavelength"])
    assert spacing == pytest.approx(2.62154e-3, rel=1e-5)
    assert float(build["design"]["rf_tone_step_hz"]) == 1e6
    assert "response" not in model


def test_the_grating_step_is_the_tones_greatest_common_divisor():
    build = build_native_model("deflector.acousto_optic", {**aod(), "rf_tones_mhz": ["80.5", "90.25"]}).as_dict()
    step = float(build["design"]["rf_tone_step_hz"])
    assert step == math.gcd(80500000, 90250000) == 250000 and [row["grating_order"] for row in build["design"]["tones"]] == [322, 361]
    assert float(build["optical_model"]["model"]["period_nm"]) == pytest.approx(650.0 / 250000.0 * 1e9, rel=1e-15)
    # tones on a step of 500 Hz: a grating of 1.3 m, with millions of propagating orders: refused, with the step that would do
    with pytest.raises(NativeComponentError) as exc:
        build_native_model("deflector.acousto_optic", {**aod(), "rf_tones_mhz": ["80.0005", "90.25"]})
    assert exc.value.problems[0].parameter == "rf_tones_mhz" and "65000" in exc.value.problems[0].message


def test_the_minus_one_order_deflects_away_and_shifts_down():
    build = build_native_model("deflector.acousto_optic", aod(diffraction_order="-1")).as_dict()
    assert build["optical_model"]["model"]["orders"] == [*(-f for f in reversed(TONES_MHZ)), 0]
    assert float(build["design"]["tones"][0]["optical_frequency_shift_hz"]) == -25e6


def test_each_tone_lands_at_the_tangent_of_its_own_angle_and_the_zero_order_stays():
    """No efficiency stated: every direction is decided, every power is not known."""

    document = bench(aod(), [plane("SCREEN", (100.0, 0.0, 0.0), radius_mm=40.0)])
    results = load_run(run(document))
    beams = results.handoff("SCREEN")["beams"]
    assert len(beams) == 17
    us = sorted(beam["chief_ray"]["uv_mm"][0] for beam in beams)
    assert us[0] == pytest.approx(0.0, abs=1e-9)
    for f, u in zip(TONES_MHZ, us[1:]):
        sine = LAMBDA_M * f * 1e6 / VELOCITY
        assert u == pytest.approx(100.0 * math.tan(math.asin(sine)), rel=1e-9)
    assert us[1] == pytest.approx(3.27868, rel=1e-5)
    # tone spacing in sine, read back from the directions
    sines = sorted(beam["chief_ray"]["direction_uvn"][0] for beam in beams)
    assert sines[2] - sines[1] == pytest.approx(2.62154e-3, rel=1e-5)
    assert all(beam["power_W"] is None for beam in beams)


def test_a_stated_efficiency_gives_each_tones_power():
    document = bench(aod(efficiency=table()), [plane("SCREEN", (100.0, 0.0, 0.0), radius_mm=40.0)])
    results = load_run(run(document))
    beams = sorted(results.handoff("SCREEN")["beams"], key=lambda beam: beam["chief_ray"]["uv_mm"][0])
    assert beams[0]["power_W"] == pytest.approx(0.1 * 20e-9, rel=1e-12)  # the zero order
    assert all(beam["power_W"] == pytest.approx(0.05 * 20e-9, rel=1e-12) for beam in beams[1:])
    # a period of 650 um has 1525 propagating orders at 852 nm: the list holds those followed
    orders = results.grating_orders("AOD")
    assert sorted(row["order"] for row in orders) == [0, *TONES_MHZ]
    assert {row["response_basis"] for row in orders} == {"ASSUMED"}
    assert {row["order"]: row["efficiency"] for row in orders}[25] == pytest.approx(0.05, rel=1e-12)


def test_what_the_design_report_states_of_each_quantity():
    no_table = build_native_model("deflector.acousto_optic", aod()).design["quantities"]
    assert no_table["direction"]["status"] == "MODELED" and no_table["optical_frequency_shift_hz"]["status"] == "MODELED"
    assert no_table["order_power"]["status"] == "UNAVAILABLE" and no_table["order_power"]["missing"] == ["efficiency"]
    assert {no_table[name]["status"] for name in ("tone_phases", "timing", "rf_crosstalk", "intermodulation")} == {"UNAVAILABLE"}
    stated = build_native_model("deflector.acousto_optic", aod(efficiency=table())).design["quantities"]["order_power"]
    assert stated["status"] == "EXTERNALLY_CHARACTERIZED" and stated["source"]["data_kind"] == "ASSUMED" and stated["zero_order_stated"] is True
    assert build_native_model("deflector.acousto_optic", aod(efficiency=table(zero=None))).design["quantities"]["order_power"]["zero_order_stated"] is False


@pytest.mark.parametrize("change, parameter", [
    ({"rf_tones_mhz": []}, "rf_tones_mhz"), ({"rf_tones_mhz": ["-1"]}, "rf_tones_mhz"), ({"rf_tones_mhz": ["80", "80"]}, "rf_tones_mhz"), ({"rf_tones_mhz": ["80.00000000001"]}, "rf_tones_mhz"),
    ({"efficiency": {"rows": []}}, "efficiency"), ({"efficiency": "high"}, "efficiency"),
    ({"efficiency": {**table(), "rows": table()["rows"][:3]}}, "efficiency"),  # not every tone
    ({"efficiency": {**table(), "rows": [*table()["rows"], {"wavelength_nm": "852", "rf_mhz": "99", "efficiency": "0.1"}]}}, "efficiency"),  # not a tone
    ({"efficiency": {**table(), "valid_aoi_deg": ["2", "-2"]}}, "efficiency"),
    ({"efficiency": table(rows_per_tone=0.1)}, "response"),  # 16 x 0.1 + 0.1 > 1 of the incident power
])
def test_a_deflector_that_cannot_be_built_says_why(change, parameter):
    with pytest.raises(NativeComponentError) as exc:
        build_native_model("deflector.acousto_optic", {**aod(), **change})
    assert exc.value.problems[0].parameter == parameter


def test_the_class_is_discovered_beside_the_others(capsys):
    described = next(c for c in native_components() if c["class_id"] == "deflector.acousto_optic")
    assert described["model_kind"] == "DIFFRACTION_GRATING" and described["category"] == "MODULATOR" and "AOD" in described["keywords"]
    assert "CUSTOMER_SUPPLIED" in described["availability"] and described["catalog_equivalents"] == []
    assert "deflector.acousto_optic" in json_api.capabilities()["native_component_classes"]
    assert cli_main(["native", "list", "--category", "modulator"]) == 0
    assert [c["class_id"] for c in json.loads(capsys.readouterr().out)["classes"]] == ["deflector.acousto_optic"]


# ------------------------------------------------------------------------------------------------- the chain


def _tones_at_pupil(parameters: dict) -> tuple[object, dict]:
    document = bench(parameters, [plane("PUPIL", (5.0, 0.0, 0.0), radius_mm=40.0)])
    results = load_run(run(document))
    beams = sorted(results.handoff("PUPIL")["beams"], key=lambda beam: beam["chief_ray"]["uv_mm"][0])
    request = {"wavelength_nm": LAMBDA_NM, "focal_length_mm": 200.0, "sampling": {"pixel_count": [640, 640], "pitch_mm": [0.005, 0.005], "fft_size": [2048, 2048]},
               "illumination": {"kind": "HANDOFF_BEAMS", "composition": "INCOHERENT", "path_ids": [beam["path_id"] for beam in beams[1:]]},
               "output": {"center_uv_mm": [10.487, 0.0], "half_extent_mm": [4.5, 0.5]}}
    return results, request


def test_sixteen_tones_go_from_the_deflector_through_the_pupil_to_the_detector():
    """Power, direction and coordinates carried by the run, the focal field and the camera, against the hand numbers of the module docstring."""

    results, request = _tones_at_pupil(aod(efficiency=table()))
    field = results.pupil_field("PUPIL", request)
    assert field["status"] == "OK" and field["composition"] == "INCOHERENT_SUM"
    assert field["power"]["incident_W"] == pytest.approx(16e-9, rel=1e-12)
    assert field["power"]["window_W"] == pytest.approx(16e-9, rel=2e-3)
    rows = sorted(field["spots"], key=lambda row: row["u_mm"])
    assert rows[0]["u_mm"] == pytest.approx(6.55385, rel=1e-5)
    assert rows[1]["u_mm"] - rows[0]["u_mm"] == pytest.approx(0.52431, rel=1e-4)
    # the largest focal sample near each spot, at most half a step of 16.64 um off: 2 (0.5 h / w_f)^2 = 1.2 %, and the grating's anamorphic stretch of 0.3 %
    assert all(row["peak_W_per_mm2"] == pytest.approx(5.410e-8, rel=0.016) for row in rows)
    assert field["array"]["uniformity_power"] > 0.99
    assert field["quantities"]["focal_phase"]["status"] == "UNAVAILABLE"
    camera = detector_response(field, {"exposure_s": 1e-3, "pixel_pitch_mm": [0.0065, 0.0065], "fill_factor": 1.0, "pixel_count": [1, 1], "center_uv_mm": [rows[0]["u_mm"], 0.0]})
    assert camera["quantities"]["photons"]["values"][0][0] == pytest.approx(9.804e3, rel=0.016)
    assert camera["quantities"]["signal_electrons"]["reason"] == "PARAMETER_NOT_STATED"


def test_a_deflector_with_no_efficiency_gives_where_the_spots_are_and_no_power():
    results, request = _tones_at_pupil(aod())
    field = results.pupil_field("PUPIL", request)
    assert field["status"] == "UNAVAILABLE" and field["reason"] == "BEAM_POWER_UNKNOWN"
    positions = sorted(row["u_mm"] for row in field["spot_positions"])
    assert positions[0] == pytest.approx(6.55385, rel=1e-5) and len(positions) == 16
    camera = detector_response(field, {"exposure_s": 1e-3})
    assert camera["status"] == "UNAVAILABLE" and camera["upstream_reason"] == "BEAM_POWER_UNKNOWN"


def test_a_table_with_no_zero_order_row_leaves_the_zero_order_out_of_the_camera_and_says_so():
    """The table states the 16 tones, not ``rf_mhz: 0``: the zero order has a direction and up to the rest of the power, 20 - 16 = 4 nW, and the camera reads the tones only."""

    results, _request = _tones_at_pupil(aod(efficiency=table(zero=None)))
    field = results.irradiance_map("PUPIL", samples=21)
    assert field["status"] == "OK" and field["map_complete"] is False
    (zero_order,) = field["omitted_beams"]
    assert zero_order["uv_mm"] == pytest.approx([0.0, 0.0], abs=1e-9)
    assert zero_order["power_upper_bound_W"] == pytest.approx(4e-9, rel=1e-9)
    detector = {"exposure_s": 1e-3, "pixel_pitch_mm": [0.1, 0.1], "fill_factor": 1.0}
    # (one coherence group, so a field of unknown power has no floor, until the reader states what the class says: tones add in irradiance)
    assert detector_response(field, detector)["quantities"]["photons"]["reason"] == "BEAM_POWER_UNKNOWN"
    camera = detector_response(field, detector, "INCOHERENT")
    assert camera["map_complete"] is False and camera["quantities"]["photons"]["status"] == "BOUNDED"


def test_a_table_holds_whichever_polarization_arrives_as_the_class_says():
    """A beam at 45 deg through the deflector, then a glass plate met off normal, whose share depends on the polarization: each order keeps the state that arrives, so its power stays known."""

    build = json_api.build_native_model("deflector.acousto_optic", aod(efficiency=table()))
    laser = source(position=(-50.0, 0.0, 0.0), wavelength_nm=LAMBDA_NM, power_w=1e-3, waist_mm=0.5, polarization_deg=45.0)
    component = {**build["component_template"], "instance_id": "AOD", "nominal_pose": {"position_mm": [0.0, 0.0, 0.0], "quaternion_wxyz": list(ALONG_X)}}
    half = math.radians(10.0) / 2.0
    plate = {"model_id": "plate", "model": sequence([surface(1, "0", clear="30"), surface(2, "3", clear="30")], ["N-BK7"], edge="31")}
    tilted = declared_component("PLATE", "other_optical", "plate", (30.0, 0.0, 0.0), quat_multiply(ALONG_X, (math.cos(half), 0.0, math.sin(half), 0.0)))
    document = with_models(project([laser, component, tilted], [{**plane("SCREEN", (120.0, 0.0, 0.0), radius_mm=60.0), "absorbing": True}]), [build["optical_model"], plate], [nbk7()])
    result = run(document).result
    assert all(arrival.beam.power_W is not None for arrival in result.observations[0].arrivals)
    assert "GRATING_RESPONSE_UNAVAILABLE" not in {issue.code.value for issue in result.issues}


def test_tones_have_no_phase_to_add_coherently():
    results, request = _tones_at_pupil(aod(efficiency=table()))
    coherent = {**request, "illumination": {**request["illumination"], "composition": "COHERENT"}}
    assert results.pupil_field("PUPIL", coherent)["reason"] == "PHASE_NOT_DEFINED"


def test_the_zero_order_beyond_the_window_is_refused_until_it_is_left_out():
    results, request = _tones_at_pupil(aod(efficiency=table()))
    every = {**request, "illumination": {"kind": "HANDOFF_BEAMS", "composition": "INCOHERENT"}, "sampling": {**request["sampling"], "pitch_mm": [0.02, 0.02], "pixel_count": [256, 256], "fft_size": [1024, 1024]}}
    assert results.pupil_field("PUPIL", every)["reason"] == "GRID_INVALID"
    assert results.pupil_field("PUPIL", {**request, "illumination": {**request["illumination"], "path_ids": ["nope"]}})["reason"] == "NO_BEAMS"
    assert pupil_field(request)["reason"] == "HANDOFF_REQUIRED"


def test_independent_sources_are_no_substitute_and_stay_valid():
    """A bench of independent lasers at a pupil still runs and reads as it did: the deflector adds a model beside them, not a change to them. Each is its own beam at f times its direction."""

    lasers = [source(f"S{i}", (-50.0, 2.0 * i, 0.0), wavelength_nm=LAMBDA_NM, power_w=1e-9, waist_mm=0.5) for i in range(3)]
    results = load_run(run(project(lasers, [plane("PUPIL", (5.0, 0.0, 0.0), radius_mm=40.0)])))
    beams = results.handoff("PUPIL")["beams"]
    assert len(beams) == 3 and all(beam["power_W"] == pytest.approx(1e-9, rel=1e-12) and beam["validity"]["tem00_field_applicable"] for beam in beams)
    field = results.pupil_field("PUPIL", {"wavelength_nm": LAMBDA_NM, "focal_length_mm": 200.0, "sampling": {"pixel_count": [256, 256], "pitch_mm": [0.05, 0.05]},
                                          "illumination": {"kind": "HANDOFF_BEAMS", "composition": "INCOHERENT"}, "output": {"half_extent_mm": [0.5, 0.5]}})
    # parallel beams at different places: one focal spot, at the axis; the pupil's offset is a phase, which independent beams do not share
    assert field["status"] == "OK" and all(abs(row["u_mm"]) < 1e-9 for row in field["spots"])
