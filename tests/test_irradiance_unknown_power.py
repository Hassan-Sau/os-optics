"""An irradiance map and a detector read of a plane where light may arrive that the map does not hold: the map says how much, and each quantity is judged against that power, taken whole on one
pixel, in its own units. References: the two ends of a pixel by hand, the same chain with the efficiency stated (100 %), and the known beam's own power for the map's total. A small ghost keeps its
results with a range; a real unknown order of a size with the known light gives bounds or no value; light outside the map's scope never gives a false ``saturated``."""

from __future__ import annotations

import math

import pytest

from os_optics.local import run_optical_project
from os_optics.results import RunResults, detector_response

from declared_helpers import declared_component, nbk7, sequence, stop_model, surface, with_models
from local_helpers import AGAINST_X, ALONG_X, catalog_document, plane, project, quat_multiply, source

POWER_W = 1e-3
SOURCE = {"kind": "CUSTOMER_STATEMENT", "reference": "fixture"}
DETECTOR = {"exposure_s": 1e-3, "pixel_pitch_mm": [0.1, 0.1], "fill_factor": 1.0, "full_well_e": 1e12, "read_noise_e": 2.0, "dark_current_e_per_s": 1.0,
            "gain_e_per_adu": 1.0, "bit_depth": 16, "characterization": SOURCE,
            "quantum_efficiency": {"rows": [{"wavelength_nm": 400, "qe": 0.5}, {"wavelength_nm": 800, "qe": 0.5}], "source": SOURCE}}


def grating(orders=(0,), **extra):
    return {"model_id": "g", "model": {"model_kind": "DIFFRACTION_GRATING", "model_version": "1", "mode": "TRANSMISSION", "line_density_per_mm": "100", "orders": list(orders),
                                       "clear_half_widths_mm": ["10", "10"], "edge_half_widths_mm": ["11", "11"], "status": "SPECIFIED", **extra}}


def response(orders=((0, "1"),)):
    rows = [{"wavelength_nm": "633", "aoi_deg": a, "order": m, "efficiency_te": e, "efficiency_tm": e} for a in ("-5", "5") for m, e in orders]
    return {"normalization": "ABSOLUTE", "polarization_basis": "TE_TM", "interpolation": "LINEAR", "rows": rows, "data_kind": "ASSUMED", "basis": "fixture", "max_conical_angle_deg": "1"}


def run_of(*components, models=(grating(),), screen_x=100.0):
    screen = {**plane("SCREEN", (screen_x, 0.0, 0.0), radius_mm=60.0), "absorbing": True}
    document = with_models(project(list(components), [screen], None), list(models), None, design_intent="OPTICAL_SYSTEM")
    return RunResults(result=run_optical_project(document, catalog=None).result)


def laser(name, position, wavelength_nm=633.0, coherence_length_mm=None):
    made = source(name, position, wavelength_nm=wavelength_nm, waist_mm=0.5, quaternion=ALONG_X, power_w=POWER_W)
    if coherence_length_mm is not None:
        made["source_model"]["coherence_length_mm"] = coherence_length_mm
    return made


GRATING = declared_component("G", "other_optical", "g", (0.0, 0.0, 0.0), ALONG_X)


def known_and_unknown():
    return run_of(laser("A", (-100.0, 0.0, 0.0)), GRATING, laser("B", (50.0, 3.0, 0.0)))


def test_a_map_names_the_beams_of_unknown_power_it_leaves_out():
    field = known_and_unknown().irradiance_map("SCREEN")
    assert field["status"] == "OK" and field["map_complete"] is False
    (left_out,) = field["omitted_beams"]
    assert left_out["path_id"].startswith("A#") and left_out["path_id"] not in field["path_ids"]
    assert left_out["uv_mm"] == pytest.approx([0.0, 0.0], abs=1e-9)
    assert left_out["direction_uvn"] == pytest.approx([0.0, 0.0, 1.0], abs=1e-12)
    assert left_out["power_upper_bound_W"] == pytest.approx(POWER_W)
    assert field["grid_power_incoherent_W"] == pytest.approx(POWER_W, rel=1e-4)  # the known beam only


def test_a_map_of_every_arriving_beam_is_complete():
    field = run_of(laser("B", (50.0, 3.0, 0.0)), declared_component("G", "other_optical", "g", (0.0, 40.0, 0.0), ALONG_X)).irradiance_map("SCREEN")
    assert field["map_complete"] is True and field["omitted_beams"] == []


def test_a_beam_of_another_source_is_not_left_out_of_a_map_of_one_source():
    field = known_and_unknown().irradiance_map("SCREEN", source_id="B")
    assert field["map_complete"] is True and field["omitted_beams"] == []


def test_a_beam_of_unknown_power_at_another_wavelength_leaves_the_sum_over_the_wavelengths_incomplete():
    """A known 532 nm beam and a 633 nm beam of unknown power: the map of 532 nm alone is complete and 633 nm alone has none.
    Without a wavelength the map is the sum over both lines: it holds the 532 nm beam (its own map on the same grid, bit for bit),
    says the 633 nm line has no map and names its beam, and is not complete."""

    results = run_of(laser("A", (-100.0, 0.0, 0.0), wavelength_nm=633.0), GRATING, laser("B", (50.0, 3.0, 0.0), wavelength_nm=532.0))
    green = results.irradiance_map("SCREEN", wavelength_nm=532.0)
    assert green["map_complete"] is True and green["wavelength_nm"] == 532.0
    assert results.irradiance_map("SCREEN", wavelength_nm=633.0)["reason"] == "BEAM_POWER_UNKNOWN"
    total = results.irradiance_map("SCREEN")
    assert total["status"] == "OK" and total["wavelength_nm"] is None and total["wavelengths_nm"] == [532.0, 633.0]
    assert total["scope"] == {"wavelength_nm": None, "source_id": None} and total["composition_over_wavelengths"] == "INCOHERENT_SUM"
    assert total["map_complete"] is False and total["untraced_power_W"] is None and total["untraced_fraction"] is None
    (left_out,) = total["omitted_beams"]
    assert left_out["path_id"].startswith("A#") and total["path_ids"] == green["path_ids"]
    rows = {row["wavelength_nm"]: row for row in total["per_wavelength"]}
    assert rows[532.0]["map_complete"] is True and rows[532.0]["grid_power_incoherent_W"] == green["grid_power_incoherent_W"]
    assert rows[633.0]["reason"] == "BEAM_POWER_UNKNOWN" and rows[633.0]["grid_power_incoherent_W"] is None and rows[633.0]["map_complete"] is False
    # (the 532 nm beam is the only usable light, so the sum's grid is its own map's)
    assert total["grid"] == green["grid"] and total["incoherent_W_per_mm2"] == green["incoherent_W_per_mm2"]
    # a camera reads the power on its pixels from the sum; photons and electrons need one wavelength
    reply = detector_response(total, DETECTOR)
    assert reply["status"] == "OK" and reply["wavelength_nm"] is None and reply["wavelengths_nm"] == [532.0, 633.0]
    assert reply["quantities"]["pixel_power_W"]["status"] == "BOUNDED" and reply["quantities"]["pixel_power_W"]["bound"] == "LOWER"
    for name in ("photon_energy_J", "photon_flux_density", "photons", "signal_electrons", "snr", "saturated", "adu"):
        q = reply["quantities"][name]
        assert q["status"] == "UNAVAILABLE" and q["reason"] == "WAVELENGTH_REQUIRED" and q["missing"] == ["wavelength_nm"], name
    assert reply["quantities"]["dark_electrons"]["status"] == "MODELED" and reply["quantities"]["read_noise_variance_e2"]["status"] == "MODELED"
    assert reply["inputs"]["quantum_efficiency"]["reason"] == "WAVELENGTH_REQUIRED"


def test_no_uniformity_is_stated_over_a_map_that_leaves_a_beam_out():
    roi = {"shape": "circle", "center_uv_mm": [0.0, 0.0], "radius_mm": 5.0}
    uniformity = known_and_unknown().irradiance_map("SCREEN", roi=roi)["uniformity"]
    assert uniformity["min_over_mean"] is None and uniformity["std_over_mean"] is None and uniformity["reason"] == "BEAM_POWER_UNKNOWN"
    # the withheld figure names the share and the limit that withholds it
    assert uniformity["untraced_fraction"] == pytest.approx(1.0) and uniformity["max_untraced_fraction"] == 0.1


def test_the_coherent_map_is_withheld_when_a_left_out_beam_may_interfere():
    # (one source, a table that states order 0 and leaves order 1 out: one coherence group, and a field of unknown power may also darken the map)
    results = run_of(laser("A", (-100.0, 0.0, 0.0), coherence_length_mm=1e6), GRATING, models=(grating(orders=(0, 1), response=response(((0, "0.5"),))),))
    field = results.irradiance_map("SCREEN")
    assert len(field["omitted_beams"]) == 1
    assert field["coherent_W_per_mm2"] is None and field["coherent_reason"] == "BEAM_POWER_UNKNOWN" and field["grid_power_coherent_W"] is None
    assert detector_response(field, DETECTOR)["composition"] == "INCOHERENT"


def test_a_detector_reads_a_map_that_leaves_a_beam_out_as_lower_bounds():
    reply = known_and_unknown().detector_response("SCREEN", DETECTOR)
    assert reply["map_complete"] is False and reply["omitted_beams"][0]["path_id"].startswith("A#")
    quantities = reply["quantities"]
    for name in ("pixel_power_W", "photons", "signal_electrons", "shot_noise_variance_e2", "total_noise_variance_e2", "adu", "photon_flux_density"):
        assert quantities[name]["status"] == "BOUNDED" and quantities[name]["bound"] == "LOWER", name
    assert quantities["snr"]["status"] == "UNAVAILABLE" and quantities["snr"]["reason"] == "BEAM_POWER_UNKNOWN"
    assert "snr" in reply["unavailable_quantities"]
    assert reply["bounded_quantities"] == sorted(["adc_clipped", "adu", "photon_flux_density", "photons", "pixel_power_W", "saturated", "shot_noise_variance_e2",
                                                  "signal_electrons", "total_noise_variance_e2"])
    # what does not read the light keeps its status
    assert quantities["dark_electrons"]["status"] == "MODELED" and quantities["read_noise_variance_e2"]["status"] == "MODELED"
    assert quantities["photon_energy_J"]["status"] == "MODELED"
    assert reply["totals"]["bound"] == "LOWER"
    assert reply["totals"]["power_on_pixels_W"] == pytest.approx(POWER_W, rel=1e-3)


def test_a_detector_states_no_false_saturation_while_a_beam_is_left_out():
    results = known_and_unknown()
    # within reach of the omitted beam's whole power on one pixel: 1 mW for 1 ms at 633 nm and QE 0.5 is 1.6e12 e
    quantities = results.detector_response("SCREEN", {**DETECTOR, "full_well_e": 1e12, "gain_e_per_adu": 1e11})["quantities"]
    assert False not in [value for row in quantities["saturated"]["values"] for value in row]
    # beyond its reach: not saturated, whatever it does
    far = results.detector_response("SCREEN", {**DETECTOR, "full_well_e": 1e15, "gain_e_per_adu": 1e11})["quantities"]
    assert all(value is False for row in far["saturated"]["values"] for value in row)
    # a pixel the known beam alone saturates is saturated whatever else arrives
    low_well = results.detector_response("SCREEN", {**DETECTOR, "full_well_e": 1e9})
    flat = [value for row in low_well["quantities"]["saturated"]["values"] for value in row]
    assert True in flat and None in flat and False not in flat
    assert low_well["totals"]["saturated_pixels"] == sum(1 for value in flat if value is True)


def test_the_lower_bound_holds_against_the_same_chain_with_the_efficiency_stated():
    # the same light with a stated 100 % order: the complete read is MODELED, and every bounded value of the incomplete read is at most its value
    window = {"half_extent_mm": 5.0, "center_uv_mm": [0.0, 0.0], "samples": 101}
    full = run_of(laser("A", (-100.0, 0.0, 0.0)), GRATING, laser("B", (50.0, 3.0, 0.0)), models=(grating(response=response()),))
    complete = full.detector_response("SCREEN", DETECTOR, composition="INCOHERENT", **window)
    partial = known_and_unknown().detector_response("SCREEN", DETECTOR, composition="INCOHERENT", **window)
    assert complete["map_complete"] is True and complete["quantities"]["signal_electrons"]["status"] == "MODELED"
    assert complete["quantities"]["saturated"]["status"] == "MODELED" and "bound" not in complete["totals"]
    for name in ("pixel_power_W", "signal_electrons", "adu"):
        # (the map is read by cubic convolution, whose small negative lobes the detector's module docstring states: a tolerance of 1e-3 of the peak, chosen before the run)
        peak = max(max(row) for row in complete["quantities"][name]["values"])
        for low, high in zip(partial["quantities"][name]["values"], complete["quantities"][name]["values"]):
            assert all(a <= b + 1e-3 * peak for a, b in zip(low, high)), name
    assert complete["totals"]["power_on_pixels_W"] == pytest.approx(2.0 * partial["totals"]["power_on_pixels_W"], rel=1e-3)


def test_a_map_is_not_complete_when_its_plane_is_not():
    """Order 1 is in the table at 0.5 and is not followed: no beam of unknown power reaches the plane, yet up to 1 mW may. The plane says so, and the camera must not answer with a false `saturated`."""

    results = run_of(laser("A", (-100.0, 0.0, 0.0)), GRATING, models=(grating(orders=(0,), response=response(((0, "0.5"), (1, "0.5")))),))
    (obs,) = results.result.observations
    assert obs.power_complete is False
    field = results.irradiance_map("SCREEN")
    assert field["status"] == "OK" and field["omitted_beams"] == [] and field["map_complete"] is False
    assert field["untraced_power_W"] == pytest.approx(0.5 * POWER_W, rel=1e-9)
    reply = detector_response(field, DETECTOR)
    assert reply["map_complete"] is False and reply["untraced_power_W"] == pytest.approx(0.5 * POWER_W, rel=1e-9)
    assert reply["quantities"]["signal_electrons"]["status"] == "UNAVAILABLE"


def test_the_incompleteness_of_another_source_does_not_touch_a_map_of_one_source():
    """The plane is open because of source A's grating: a map of B alone holds all of B's light. A's own map is open."""

    results = run_of(laser("A", (-100.0, 0.0, 0.0)), GRATING, laser("B", (50.0, 3.0, 0.0)), models=(grating(orders=(0,), response=response(((0, "0.5"), (1, "0.5")))),))
    assert results.result.observations[0].power_complete is False
    assert results.irradiance_map("SCREEN", source_id="B")["map_complete"] is True
    assert results.irradiance_map("SCREEN", source_id="A")["map_complete"] is False


def test_a_dark_fringe_is_not_read_as_a_lower_bound():
    """Orders 0 and 1 of one coherent source, the table states order 0 only, 1 mm behind a coarse grating (20 lines/mm): the beams overlap with a fringe period of lambda / sin = 50 um, and a field
    of unknown power may cancel the known beam at a fringe minimum. Nothing that reads the light has a floor; stated as an incoherent sum by the reader, it has."""

    results = run_of(laser("A", (-100.0, 0.0, 0.0), coherence_length_mm=1e6), GRATING,
                     models=(grating(orders=(0, 1), response=response(((0, "0.5"),)), line_density_per_mm="20"),), screen_x=1.0)
    field = results.irradiance_map("SCREEN")
    assert field["coherent_reason"] == "BEAM_POWER_UNKNOWN"
    reply = detector_response(field, DETECTOR)
    for name in ("photon_flux_density", "pixel_power_W", "photons", "signal_electrons", "shot_noise_variance_e2", "total_noise_variance_e2", "snr", "saturated", "adu", "adc_clipped"):
        assert reply["quantities"][name]["status"] == "UNAVAILABLE" and reply["quantities"][name]["reason"] == "BEAM_POWER_UNKNOWN", name
        assert "values" not in reply["quantities"][name]
    assert reply["bounded_quantities"] == [] and "power_on_pixels_W" not in reply["totals"]
    assert reply["quantities"]["dark_electrons"]["status"] == "MODELED"
    stated = detector_response(field, DETECTOR, "INCOHERENT")
    assert stated["composition"] == "INCOHERENT" and stated["quantities"]["signal_electrons"]["status"] == "BOUNDED"


def _singlet_run(absorbing=True):
    """1 mW through an uncoated glass singlet onto a plane: its ghost is about 0.2 % of the light, 4 % per face reflected twice."""

    lens = {"model_id": "L", "model": sequence([surface(1, "0", radius="100", clear="12"), surface(2, "4", radius="-100", clear="12")], ["N-BK7"], edge="13")}
    screen = {**plane("SCREEN", (60.0, 0.0, 0.0), radius_mm=60.0), "absorbing": absorbing}
    # This synthetic control assumes zero bulk absorption so only Fresnel ghosts open its power.
    document = with_models(project([laser("A", (-50.0, 0.0, 0.0)), declared_component("L1", "lens", "L", (0.0, 0.0, 0.0), ALONG_X)], [screen], None), [lens], [{**nbk7(), "absorption_coefficient_per_mm": "0"}],
                           design_intent="OPTICAL_SYSTEM")
    return RunResults(result=run_optical_project(document, catalog=None).result)


@pytest.mark.parametrize("absorbing", [True, False])
def test_a_small_ghost_keeps_the_results_with_their_range(absorbing):
    """The control: the ghost is about 1.6 uW of 1 mW, so a camera far from its full well reads not saturated, and SNR and values stand, with a range. A plane that does not absorb has
    no bound on its own (that light may cross it more than once); a detector placed there receives one crossing."""

    results = _singlet_run(absorbing)
    field = results.irradiance_map("SCREEN")
    assert field["status"] == "OK" and field["coherent_W_per_mm2"] is not None and field["coherent_reason"] is None
    assert 0.0 < field["untraced_power_W"] < 0.01 * POWER_W and 0.0 < field["untraced_fraction"] < 0.01
    assert field["map_complete"] is False and field["interference_open"] is True
    reply = results.detector_response("SCREEN", {**DETECTOR, "full_well_e": 1e15, "gain_e_per_adu": 1e10}, composition="COHERENT")
    quantities = reply["quantities"]
    flat = [value for row in quantities["saturated"]["values"] for value in row]
    assert flat and all(value is False for value in flat)
    assert quantities["snr"]["status"] == "BOUNDED" and quantities["snr"]["bound"] == "INTERVAL"
    assert quantities["signal_electrons"]["status"] == "BOUNDED" and quantities["signal_electrons"]["bound"] == "INTERVAL"
    u = field["untraced_power_W"]
    # the two ends by hand at the brightest pixel: the field of the ghost whole on it
    powers = quantities["pixel_power_W"]
    (j, i) = max(((j, i) for j in range(len(powers["values"])) for i in range(len(powers["values"][0]))), key=lambda ji: powers["values"][ji[0]][ji[1]])
    p = powers["values"][j][i]
    assert powers["upper"][j][i] == pytest.approx((math.sqrt(p) + math.sqrt(u)) ** 2, rel=1e-12)
    assert powers["lower"][j][i] == pytest.approx((math.sqrt(p) - math.sqrt(u)) ** 2, rel=1e-12)
    assert quantities["snr"]["lower"][j][i] < quantities["snr"]["values"][j][i] < quantities["snr"]["upper"][j][i]


def test_a_full_well_the_worst_case_could_reach_reads_not_known_and_one_the_known_light_reaches_reads_saturated():
    """The rule in electrons: k = QE t / E, a pixel of power p with the ghost U on it. Saturated false only below (sqrt(p) + sqrt(U))^2 k, true at or above (sqrt(p) - sqrt(U))^2 k, between: null."""

    results = _singlet_run()
    field = results.irradiance_map("SCREEN")
    base = results.detector_response("SCREEN", {**DETECTOR, "full_well_e": 1e15}, composition="COHERENT")["quantities"]
    u, values = field["untraced_power_W"], base["pixel_power_W"]["values"]
    e_per_w = base["signal_electrons"]["values"][0][0] / values[0][0]
    p = max(max(row) for row in values)
    between = ((math.sqrt(p) - math.sqrt(u)) ** 2 + (math.sqrt(p) + math.sqrt(u)) ** 2) / 2.0 * e_per_w
    flat = [v for row in results.detector_response("SCREEN", {**DETECTOR, "full_well_e": between}, composition="COHERENT")["quantities"]["saturated"]["values"] for v in row]
    assert None in flat and False in flat
    high = (math.sqrt(p) + math.sqrt(u)) ** 2 * e_per_w * 1.001
    assert True not in [v for row in results.detector_response("SCREEN", {**DETECTOR, "full_well_e": high}, composition="COHERENT")["quantities"]["saturated"]["values"] for v in row]
    low = (math.sqrt(p) - math.sqrt(u)) ** 2 * e_per_w * 0.999
    assert True in [v for row in results.detector_response("SCREEN", {**DETECTOR, "full_well_e": low}, composition="COHERENT")["quantities"]["saturated"]["values"] for v in row]


def _cavity_run():
    """A source between mirrors of R = 0.99, a plane that does not absorb between them: the interaction limit stops the trace at 64 surfaces, the rest goes round. By the series the plane is crossed by
    P / (1 - R) = 100 P in all, far above what the map sums and what any one crossing brings."""

    coating = next(c for c in catalog_document()["coatings"] if c["kind"] == "DIELECTRIC_HIGH_REFLECTOR")
    mirror = {"model_id": "m", "model": sequence([surface(1, "0", clear="30", traced="REFLECTED", coating_id=coating["coating_id"])], [], edge="31")}
    parts = [laser("A", (-5.0, 0.0, 0.0)), declared_component("M1", "mirror", "m", (50.0, 0.0, 0.0), AGAINST_X), declared_component("M2", "mirror", "m", (-10.0, 0.0, 0.0), ALONG_X)]
    document = with_models(project(parts, [plane("SCREEN", (25.0, 0.0, 0.0), radius_mm=25.0)], None), [mirror], None, [coating], design_intent="OPTICAL_SYSTEM")
    return RunResults(result=run_optical_project(document, catalog=None).result)


CAVITY_DETECTOR = {**DETECTOR, "pixel_pitch_mm": [1.0, 1.0], "gain_e_per_adu": 1e13}


def test_a_cavity_on_a_plane_that_does_not_absorb_has_no_upper_end():
    """What the map sums and one crossing of what is left cannot bound a plane the light goes on crossing: a full well above that and below the series truth must not read `false`."""

    results = _cavity_run()
    (obs,) = results.result.observations
    assert obs.power_upper_bound_W is None and obs.absorbed_power_upper_bound_W is None
    assert results.irradiance_map("SCREEN")["untraced_power_W"] is None
    known = results.detector_response("SCREEN", CAVITY_DETECTOR, composition="INCOHERENT")["quantities"]
    values = known["pixel_power_W"]["values"]
    assert known["pixel_power_W"]["bound"] == "LOWER" and "upper" not in known["pixel_power_W"]
    e_per_w = known["signal_electrons"]["values"][0][0] / values[0][0]
    peak = max(max(row) for row in values)
    truth = 100.0 * POWER_W  # (the whole series on the plane; the peak pixel holds a share of it)
    assert peak < truth
    for full_well, expected in ((1.5 * peak * e_per_w, {None}), (truth * e_per_w, {None})):
        flat = {v for row in results.detector_response("SCREEN", {**CAVITY_DETECTOR, "full_well_e": full_well}, composition="INCOHERENT")["quantities"]["saturated"]["values"] for v in row}
        assert flat == expected
    below = {v for row in results.detector_response("SCREEN", {**CAVITY_DETECTOR, "full_well_e": 0.5 * peak * e_per_w}, composition="INCOHERENT")["quantities"]["saturated"]["values"] for v in row}
    assert True in below and False not in below
    # and with the light free to interfere, as a laser's is, nothing that reads it has an end
    assert results.detector_response("SCREEN", CAVITY_DETECTOR)["quantities"]["saturated"]["status"] == "UNAVAILABLE"


def _emitter_lens_run(*, lossless=True):
    emitter = {"instance_id": "EM", "optical_role": "source", "custom_part": {"custom_part_key": "emitter", "description": "e"},
               "nominal_pose": {"position_mm": [-30.0, 0.0, 0.0], "quaternion_wxyz": list(ALONG_X)},
               "source_model": {"model_kind": "EXTENDED_EMITTER", "wavelength_nm": 633.0, "power_w": 1.0, "emitting_surface": {"shape": "DISC", "diameter_mm": 1.0},
                                "angular_law": "LAMBERTIAN", "sampling": {"sample_count": 4096, "seed": 3}}}
    lens = {"model_id": "L", "model": sequence([surface(1, "0", radius="60", clear="12"), surface(2, "4", radius="-60", clear="12")], ["N-BK7"], edge="13")}
    # Only the Fresnel control assumes lossless bulk; omitting absorption in the second
    # case must not produce an invented uniformity or crash on its unknown power.
    material = {**nbk7(), "absorption_coefficient_per_mm": "0"} if lossless else {
        k: v for k, v in nbk7().items() if k != "absorption_coefficient_per_mm"}
    components = [emitter, declared_component("LENS", "lens", "L", (0.0, 0.0, 0.0), ALONG_X)]
    if not lossless:
        # The sampled emitter alone has its own upper-ray model; a separate
        # Gaussian beam genuinely traverses this glass with unstated bulk loss.
        components.append(source("GAUSS", (-30, 0, 0), wavelength_nm=633, waist_mm=.5))
    document = with_models(project(components, [{**plane("D", (60.0, 0.0, 0.0), radius_mm=50.0), "absorbing": True}], None), [lens], [material],
                           design_intent="OPTICAL_SYSTEM")
    return RunResults(result=run_optical_project(document, catalog=None).result)


def test_the_uniformity_of_an_emitter_through_a_lens_stands_with_its_error():
    """Fresnel light of the lens is not a reason to withhold a figure the emitter chain exists to give: about 5 % of the light on the plane, stated."""

    results = _emitter_lens_run()
    roi = {"shape": "circle", "center_uv_mm": [0.0, 0.0], "radius_mm": 10.0}
    field = results.irradiance_map("D", samples=32, half_extent_mm=20.0, center_uv_mm=[0.0, 0.0], roi=roi)
    uniformity = field["uniformity"]
    assert field["map_complete"] is False and field["coherent_reason"] == "INDEPENDENT_SAMPLES" and field["interference_open"] is False
    assert uniformity["min_over_mean"] is not None and uniformity["std_over_mean"] is not None and "reason" not in uniformity
    assert 0.0 < uniformity["untraced_fraction"] <= 0.1
    assert uniformity["relative_error_bound"] == pytest.approx(uniformity["roi_untraced_fraction"], rel=1e-12)  # (independent rays add)
    assert uniformity["roi_untraced_fraction"] >= uniformity["untraced_fraction"]


def test_sampled_emitter_through_unknown_loss_withholds_uniformity_instead_of_crashing():
    results = _emitter_lens_run(lossless=False)
    roi = {"shape": "circle", "center_uv_mm": [0.0, 0.0], "radius_mm": 10.0}
    field = results.irradiance_map("D", samples=32, half_extent_mm=20.0, center_uv_mm=[0.0, 0.0], roi=roi)
    assert field["status"] == "OK" and not field["map_complete"]
    assert "_of_unknown_power" not in field
    assert field["uniformity"]["min_over_mean"] is None
    assert field["uniformity"]["reason"] == "BEAM_POWER_UNKNOWN"


def test_a_beam_of_unknown_power_outside_the_acceptance_cone_does_not_open_the_map():
    """A's orders at 1200 lines/mm, 633 nm: 0 and +-1 (sin 0.76, 49.5 deg), all followed, all outside a 3 deg cone about a normal at 25 deg; B arrives along that normal."""

    half = math.radians(25.0) / 2.0
    tilted = quat_multiply((math.cos(half), 0.0, 0.0, math.sin(half)), ALONG_X)
    normal = (math.cos(math.radians(25.0)), math.sin(math.radians(25.0)), 0.0)
    screen = {**plane("SCREEN", (100.0 * normal[0], 100.0 * normal[1], 0.0), tilted, radius_mm=300.0), "absorbing": True, "acceptance_half_angle_deg": 3.0}
    other = source("B", (20.0 * normal[0], 20.0 * normal[1], 0.0), wavelength_nm=633.0, waist_mm=0.5, quaternion=tilted, power_w=POWER_W)
    wide = grating(orders=(-1, 0, 1), line_density_per_mm="1200", clear_half_widths_mm=["5", "5"], edge_half_widths_mm=["6", "6"])
    document = with_models(project([laser("A", (-100.0, 0.0, 0.0)), GRATING, other], [screen], None), [wide], None, design_intent="OPTICAL_SYSTEM")
    results = RunResults(result=run_optical_project(document, catalog=None).result)
    (obs,) = results.result.observations
    assert all(a.within_acceptance is False for a in obs.arrivals if a.beam.source_id == "A") and any(a.beam.power_W is None for a in obs.arrivals)
    field = results.irradiance_map("SCREEN")
    assert field["omitted_beams"] == [] and field["untraced_power_W"] == 0.0 and field["map_complete"] is True


def test_an_order_not_followed_may_still_reach_the_cone():
    """Order 0 is in the table and not followed, order 1 is outside the cone: what order 0 carries has no such statement."""

    screen = {**plane("SCREEN", (100.0, 0.0, 0.0), radius_mm=300.0), "absorbing": True, "acceptance_half_angle_deg": 1.0}
    document = with_models(project([laser("A", (-100.0, 0.0, 0.0)), GRATING, laser("B", (50.0, 3.0, 0.0))], [screen], None), [grating(orders=(1,))], None, design_intent="OPTICAL_SYSTEM")
    field = RunResults(result=run_optical_project(document, catalog=None).result).irradiance_map("SCREEN")
    assert field["omitted_beams"] == [] and field["map_complete"] is False and field["untraced_power_W"] == pytest.approx(POWER_W)


def test_an_order_not_followed_of_the_same_source_may_cancel_the_light():
    """The same source's order in the table, not followed: as large as the known light, and it can interfere. The coherent map is withheld and nothing reading the light has an end, until the reader says the sum is incoherent."""

    results = run_of(laser("A", (-100.0, 0.0, 0.0)), GRATING, models=(grating(orders=(0,), response=response(((0, "0.5"), (1, "0.5")))),))
    field = results.irradiance_map("SCREEN")
    assert field["interference_open"] is True and field["coherent_reason"] == "BEAM_POWER_UNKNOWN" and field["coherent_W_per_mm2"] is None
    reply = detector_response(field, DETECTOR)
    assert reply["quantities"]["signal_electrons"]["status"] == "UNAVAILABLE" and reply["quantities"]["saturated"]["status"] == "UNAVAILABLE"
    stated = detector_response(field, DETECTOR, "INCOHERENT")["quantities"]["signal_electrons"]
    assert stated["status"] == "BOUNDED" and stated["bound"] == "LOWER" and "upper" in stated


def test_a_camera_read_of_one_wavelength_says_so_and_states_no_false_saturation():
    """An unknown 532 nm beam beside a known 633 nm map: the 633 nm values are of their scope, and the camera, which sees the 532 nm light too, cannot be called unsaturated."""

    window = {"wavelength_nm": 633.0}
    results = run_of(laser("A", (-100.0, 0.0, 0.0), wavelength_nm=532.0), GRATING, laser("B", (50.0, 3.0, 0.0), wavelength_nm=633.0))
    reply = results.detector_response("SCREEN", {**DETECTOR, "full_well_e": 1e15, "gain_e_per_adu": 1e10}, **window)
    assert reply["scope"] == {"wavelength_nm": 633.0, "source_id": None}
    (other,) = reply["outside_scope"]
    assert other["wavelength_nm"] == 532.0 and other["known_power_W"] == 0.0 and other["power_upper_bound_W"] == pytest.approx(POWER_W)
    assert reply["quantities"]["signal_electrons"]["status"] == "MODELED"
    flat = [v for row in reply["quantities"]["saturated"]["values"] for v in row]
    assert reply["quantities"]["saturated"]["status"] == "BOUNDED" and set(flat) == {None}


def test_a_camera_read_of_one_source_says_so_and_states_no_false_saturation():
    results = known_and_unknown()
    reply = results.detector_response("SCREEN", {**DETECTOR, "full_well_e": 1e15, "gain_e_per_adu": 1e10}, source_id="B")
    assert reply["map_complete"] is True and reply["scope"] == {"wavelength_nm": 633.0, "source_id": "B"}
    (other,) = reply["outside_scope"]
    assert other["sources"] == ["A"] and other["power_upper_bound_W"] == pytest.approx(POWER_W)
    assert reply["quantities"]["signal_electrons"]["status"] == "MODELED"
    assert set(v for row in reply["quantities"]["saturated"]["values"] for v in row) == {None}
    alone = run_of(laser("B", (50.0, 3.0, 0.0)), declared_component("G", "other_optical", "g", (0.0, 40.0, 0.0), ALONG_X))
    full = alone.detector_response("SCREEN", {**DETECTOR, "full_well_e": 1e15, "gain_e_per_adu": 1e10})
    assert full["outside_scope"] == [] and full["quantities"]["saturated"]["status"] == "MODELED"


def test_a_field_with_no_statement_of_completeness_stays_modeled():
    # a focal field of pupil_field refuses beams of unknown power: its document omits nothing
    axis = [-1.0 + 0.05 * i for i in range(41)]
    field = {"status": "OK", "model": "test", "wavelength_nm": 633.0, "grid": {"u_mm": axis, "v_mm": axis, "shape": [41, 41]},
             "incoherent_W_per_mm2": [[1e-5] * 41 for _ in axis], "coherent_W_per_mm2": None}
    reply = detector_response(field, DETECTOR)
    assert reply["map_complete"] is True and reply["omitted_beams"] == [] and reply["bounded_quantities"] == []
    assert reply["quantities"]["snr"]["status"] == "MODELED" and math.isfinite(reply["quantities"]["snr"]["values"][0][0])


def test_the_power_of_sampled_rays_left_without_a_ray_is_stated():
    # an extended emitter through a grating that gives order 0 at 70 % and leaves the rest to an order of unknown power: the rays have no ray for that power
    emitter = {"instance_id": "EM", "optical_role": "source", "custom_part": {"custom_part_key": "emitter", "description": "e"},
               "nominal_pose": {"position_mm": [-20.0, 0.0, 0.0], "quaternion_wxyz": list(ALONG_X)},
               "source_model": {"model_kind": "EXTENDED_EMITTER", "wavelength_nm": 633.0, "power_w": 1.0, "emitting_surface": {"shape": "DISC", "diameter_mm": 0.2},
                                "angular_law": "LAMBERTIAN", "sampling": {"sample_count": 1024, "seed": 1}}}
    wide = {"model_id": "g", "model": {**grating(orders=(0, 1))["model"], "clear_half_widths_mm": ["300", "300"], "edge_half_widths_mm": ["301", "301"], "response": {
        **response(), "rows": [{"wavelength_nm": "633", "aoi_deg": a, "order": 0, "efficiency_te": "0.7", "efficiency_tm": "0.7"} for a in ("-89.9", "89.9")], "max_conical_angle_deg": "89.9"}}}
    results = run_of(emitter, GRATING, models=(wide,))
    field = results.irradiance_map("SCREEN", samples=8)
    assert field["map_complete"] is False
    # the 30 % of the emitted watt the response leaves to order 1, up to the rays that miss the grating
    assert 0.0 < field["omitted_sampled_power_W"] <= 0.3 * (1.0 + 1e-12)
    assert detector_response(field, DETECTOR)["quantities"]["signal_electrons"]["status"] == "BOUNDED"


def test_the_exported_table_says_what_the_map_left_out():
    rows, summary = known_and_unknown().table("irradiance", monitor="SCREEN", samples=5)
    assert rows and summary["map_complete"] is False and summary["omitted_beams"][0]["path_id"].startswith("A#")


# ------------------------------------------------------------------------------------------------- the limit is the caller's


def test_the_map_and_the_detector_state_the_limit_they_apply():
    field = known_and_unknown().irradiance_map("SCREEN")
    assert field["max_untraced_fraction"] == 0.1 and field["untraced_fraction"] == pytest.approx(1.0)
    assert detector_response(field, DETECTOR)["max_untraced_fraction"] == 0.1
    refused = known_and_unknown().irradiance_map("SCREEN", max_untraced_fraction=-1.0)
    assert refused["status"] == "UNAVAILABLE" and refused["reason"] == "LIMIT_INVALID"


def test_a_raised_limit_states_the_coherent_map_and_the_detector_quantities_with_their_ends():
    """The same source's order in the table at 0.5, not followed: as large as the known light (f = 1), and it may interfere. At the
    default limit (0.1) the coherent map and everything that reads the light are withheld. With the limit raised to 2 the coherent
    map of the known beam is stated with f, and each pixel between (sqrt(p) - sqrt(U))^2 and (sqrt(p) + sqrt(U))^2, by hand at the
    brightest pixel: with U = 0.5 mW whole on a pixel that holds less, the lower end is 0."""

    results = run_of(laser("A", (-100.0, 0.0, 0.0)), GRATING, models=(grating(orders=(0,), response=response(((0, "0.5"), (1, "0.5")))),))
    roi = {"shape": "circle", "center_uv_mm": [0.0, 0.0], "radius_mm": 0.5}
    default = results.irradiance_map("SCREEN", roi=roi)
    assert default["coherent_W_per_mm2"] is None and default["coherent_reason"] == "BEAM_POWER_UNKNOWN" and default["uniformity"]["min_over_mean"] is None
    field = results.irradiance_map("SCREEN", roi=roi, max_untraced_fraction=2.0)
    assert field["max_untraced_fraction"] == 2.0 and field["untraced_fraction"] == pytest.approx(1.0) and field["interference_open"] is True
    assert field["coherent_reason"] is None and field["coherent_W_per_mm2"] == field["incoherent_W_per_mm2"]  # (one beam: its own map)
    assert field["incoherent_W_per_mm2"] == default["incoherent_W_per_mm2"] and field["map_complete"] is False
    uniformity = field["uniformity"]
    share = uniformity["roi_untraced_fraction"]
    assert uniformity["min_over_mean"] is not None and uniformity["relative_error_bound"] == pytest.approx((1.0 + math.sqrt(share)) ** 2 - 1.0, rel=1e-12)
    reply = detector_response(field, {**DETECTOR, "full_well_e": 1e15})  # (a well no end of a pixel reaches)
    assert reply["composition"] == "COHERENT" and reply["max_untraced_fraction"] == 2.0
    powers = reply["quantities"]["pixel_power_W"]
    assert powers["status"] == "BOUNDED" and powers["bound"] == "INTERVAL"
    u = field["untraced_power_W"]
    assert u == pytest.approx(0.5 * POWER_W, rel=1e-9)
    (j, i) = max(((j, i) for j in range(len(powers["values"])) for i in range(len(powers["values"][0]))), key=lambda ji: powers["values"][ji[0]][ji[1]])
    p = powers["values"][j][i]
    assert 0.0 < p < u
    assert powers["upper"][j][i] == pytest.approx((math.sqrt(p) + math.sqrt(u)) ** 2, rel=1e-12) and powers["lower"][j][i] == 0.0
    snr = reply["quantities"]["snr"]
    assert snr["status"] == "BOUNDED" and snr["bound"] == "INTERVAL" and snr["lower"][j][i] == 0.0 and snr["upper"][j][i] > snr["values"][j][i]
    # the same limit rules the detector: the default map's document refuses, and a document of the raised limit answers
    assert detector_response(default, DETECTOR)["quantities"]["signal_electrons"]["status"] == "UNAVAILABLE"
    assert reply["quantities"]["signal_electrons"]["status"] == "BOUNDED"


def test_without_a_limit_and_without_a_bound_a_pixel_has_no_end_and_no_snr():
    """The cavity plane: nothing bounds the light that goes on crossing it, and it may interfere. Without a limit the coherent map
    of the known beams is stated with untraced_fraction null; a pixel then lies between 0 (that light may cancel it: a dark fringe
    is not a lower bound) and no upper end, no saturation flag reads false, and the SNR has no end at all."""

    results = _cavity_run()
    default = results.irradiance_map("SCREEN")
    assert default["untraced_power_W"] is None and default["coherent_W_per_mm2"] is None
    field = results.irradiance_map("SCREEN", max_untraced_fraction=None)
    assert field["max_untraced_fraction"] is None and field["untraced_fraction"] is None and field["untraced_power_W"] is None
    assert field["interference_open"] is True and field["coherent_reason"] in (None, "FRINGES_UNDERSAMPLED", "COHERENCE_UNDEFINED")
    assert field["incoherent_W_per_mm2"] == default["incoherent_W_per_mm2"]
    reply = detector_response(field, {**CAVITY_DETECTOR, "full_well_e": 1e3})
    powers = reply["quantities"]["pixel_power_W"]
    assert powers["status"] == "BOUNDED" and powers["bound"] == "INTERVAL" and powers["upper"] is None
    assert all(value == 0.0 for row in powers["lower"] for value in row) and max(max(row) for row in powers["values"]) > 0.0
    assert reply["quantities"]["snr"]["status"] == "UNAVAILABLE" and reply["quantities"]["snr"]["reason"] in ("BEAM_POWER_UNKNOWN", "UNTRACED_POWER_TOO_LARGE")
    assert {value for row in reply["quantities"]["saturated"]["values"] for value in row} == {None}


# ------------------------------------------------------------------------------------------------- beams a map cannot hold


def _two_beams(*extra, models=(grating(),)):
    """Two known 1 mW beams of two sources on the screen, 3 mm apart (the grating stands aside)."""
    return run_of(laser("A", (-100.0, 0.0, 0.0)), laser("B", (50.0, 3.0, 0.0)), declared_component("G", "other_optical", "g", (0.0, 40.0, 0.0), ALONG_X), *extra,
                  models=models)


def _marked(results, source_id, **facts):
    """The same result with the arrival of ``source_id`` carrying ``facts``: what the trace states of a beam whose power is out
    of model scope, or part of which surfaces take before the plane."""
    result = results.result
    observations = tuple(o.model_copy(update={"arrivals": tuple(a.model_copy(update=facts) if a.beam.source_id == source_id else a for a in o.arrivals)})
                         for o in result.observations)
    return RunResults(result=result.model_copy(update={"observations": observations}))


WINDOW = {"half_extent_mm": 6.0, "center_uv_mm": [1.5, 0.0], "samples": 61}


def test_a_beam_no_map_can_hold_is_left_out_and_the_map_of_the_others_stays():
    """Two beams on a plane. Where the power of one is out of model scope, or surfaces take part of it before the plane, the map
    was refused whole. It now holds the other beam -- its own map, bit for bit -- names the excluded beam with its reason, its
    surfaces and what is known of its power, counts the most it can carry as light the map does not hold, and is not complete."""

    two = _two_beams()
    both = two.irradiance_map("SCREEN", **WINDOW)
    alone = two.irradiance_map("SCREEN", source_id="B", **WINDOW)
    assert both["map_complete"] is True and both["excluded_beams"] == [] and alone["grid_power_incoherent_W"] == pytest.approx(POWER_W, rel=1e-6)
    a_path = next(a.beam.path_id for a in two.result.observations[0].arrivals if a.beam.source_id == "A")

    scoped = _marked(two, "A", power_out_of_scope_at=("X/S1",)).irradiance_map("SCREEN", **WINDOW)
    assert scoped["status"] == "OK" and scoped["incoherent_W_per_mm2"] == alone["incoherent_W_per_mm2"] and scoped["path_ids"] == alone["path_ids"]
    (row,) = scoped["excluded_beams"]
    assert row["path_id"] == a_path and row["reason"] == "POWER_OUT_OF_MODEL_SCOPE" and row["surfaces"] == ["X/S1"]
    assert row["uv_mm"] == pytest.approx([0.0, 0.0], abs=1e-9)
    assert row["power_W"] is None and row["power_upper_bound_W"] is None and row["power_lower_bound_W"] is None
    assert scoped["map_complete"] is False and scoped["untraced_power_W"] is None and scoped["untraced_fraction"] is None
    # (two sources: no common coherence group, so the excluded beam cannot darken the map; its coherent sum is the beam's own)
    assert scoped["interference_open"] is False and scoped["coherent_W_per_mm2"] == alone["coherent_W_per_mm2"]

    taken = _marked(two, "A", power_taken_before_at=("X/S1",), power_taken_before_upper_W=0.4 * POWER_W).irradiance_map("SCREEN", **WINDOW)
    assert taken["status"] == "OK" and taken["incoherent_W_per_mm2"] == alone["incoherent_W_per_mm2"]
    (row,) = taken["excluded_beams"]
    assert row["reason"] == "POWER_TAKEN_BEFORE_PLANE" and row["surfaces"] == ["X/S1"] and row["power_W"] is None
    assert row["power_upper_bound_W"] == pytest.approx(POWER_W, rel=1e-9) and row["power_lower_bound_W"] == pytest.approx(0.6 * POWER_W, rel=1e-9)
    assert taken["map_complete"] is False and taken["untraced_power_W"] >= row["power_upper_bound_W"]
    assert taken["untraced_fraction"] == pytest.approx(taken["untraced_power_W"] / alone["grid_power_incoherent_W"], rel=1e-4)

    # the same beam alone on its plane: no beam is left for a map, and the reply is the refusal it was
    for facts, reason in (({"power_out_of_scope_at": ("X/S1",)}, "POWER_OUT_OF_MODEL_SCOPE"),
                          ({"power_taken_before_at": ("X/S1",), "power_taken_before_upper_W": 0.4 * POWER_W}, "POWER_TAKEN_BEFORE_PLANE")):
        refused = _marked(two, "A", **facts).irradiance_map("SCREEN", source_id="A")
        assert refused["status"] == "UNAVAILABLE" and refused["reason"] == reason and refused["path_ids"] == [a_path]


def test_an_excluded_beam_of_the_maps_coherence_group_follows_the_limit_and_never_reads_as_unsaturated():
    """The excluded beam shares the coherence group of the beam in the map, and nothing bounds its power: it may cancel the light.
    At the default limit the coherent map is withheld; without a limit it is stated, and a pixel has no end. Either way no
    saturation flag reads false where the excluded light could saturate the pixel."""

    two = _two_beams()
    group = next(a.coherence_group for a in two.result.observations[0].arrivals if a.beam.source_id == "B")
    marked = _marked(two, "A", power_out_of_scope_at=("X/S1",), coherence_group=group)
    field = marked.irradiance_map("SCREEN", **WINDOW)
    assert field["interference_open"] is True and field["coherent_W_per_mm2"] is None and field["coherent_reason"] == "UNTRACED_POWER_TOO_LARGE"
    assert field["untraced_fraction"] is None and field["max_untraced_fraction"] == 0.1
    reply = detector_response(field, {**DETECTOR, "full_well_e": 1e15}, "INCOHERENT")
    assert reply["excluded_beams"] == field["excluded_beams"]
    assert {value for row in reply["quantities"]["saturated"]["values"] for value in row} == {None}
    free = marked.irradiance_map("SCREEN", max_untraced_fraction=None, **WINDOW)
    assert free["coherent_reason"] is None and free["coherent_W_per_mm2"] is not None and free["untraced_fraction"] is None
    open_reply = detector_response(free, {**DETECTOR, "full_well_e": 1e15})
    assert {value for row in open_reply["quantities"]["saturated"]["values"] for value in row} == {None}
    # a taken beam's bound is finite: a full well the worst case could reach is not known, one beyond it reads false
    taken = _marked(two, "A", power_taken_before_at=("X/S1",), power_taken_before_upper_W=0.4 * POWER_W).irradiance_map("SCREEN", **WINDOW)
    base = detector_response(taken, {**DETECTOR, "full_well_e": 1e15}, "INCOHERENT")["quantities"]
    e_per_w = base["signal_electrons"]["values"][30][30] / base["pixel_power_W"]["values"][30][30]
    peak = max(max(row) for row in base["pixel_power_W"]["values"])
    within_reach = detector_response(taken, {**DETECTOR, "full_well_e": (peak + 0.5 * POWER_W) * e_per_w}, "INCOHERENT")["quantities"]["saturated"]["values"]
    assert False not in [value for row in within_reach for value in row]
    beyond = detector_response(taken, {**DETECTOR, "full_well_e": (peak + 2.0 * POWER_W) * e_per_w}, "INCOHERENT")["quantities"]["saturated"]["values"]
    assert all(value is False for row in beyond for value in row)


def test_a_beam_without_a_gaussian_state_is_left_out_with_its_power():
    """Beam A passes a stop of 0.3 mm (its radius there: 0.5 mm): 51 % of it is cut and it has no Gaussian state behind the stop,
    so no profile; its power is known. Alone on the plane it has no map (GAUSSIAN_NOT_AVAILABLE, as before); beside beam B the
    map holds B and names A with its power, which counts as light the map does not hold."""

    stop = stop_model("stop", edge_radius_mm="2", opening_radius_mm="0.3")
    two = _two_beams(declared_component("STOP", "aperture", "stop", (-50.0, 0.0, 0.0), ALONG_X), models=(grating(), stop))
    (a,) = [x for x in two.result.observations[0].arrivals if x.beam.source_id == "A"]
    w_stop = 0.5 * math.sqrt(1.0 + (50.0 / (math.pi * 0.5 ** 2 / 633e-6)) ** 2)  # the beam's radius at the stop, 50 mm behind its waist
    assert a.beam.power_W == pytest.approx(POWER_W * (1.0 - math.exp(-2.0 * 0.3 ** 2 / w_stop ** 2)), rel=1e-6) and not a.beam.gaussian.valid
    alone = two.irradiance_map("SCREEN", source_id="B", **WINDOW)
    field = two.irradiance_map("SCREEN", **WINDOW)
    assert field["status"] == "OK" and field["incoherent_W_per_mm2"] == alone["incoherent_W_per_mm2"]
    (row,) = field["excluded_beams"]
    assert row["reason"] == "GAUSSIAN_NOT_AVAILABLE" and row["surfaces"] == [] and row["power_W"] == a.beam.power_W
    assert row["power_upper_bound_W"] == a.beam.power_W == row["power_lower_bound_W"]
    assert field["map_complete"] is False and field["untraced_power_W"] >= a.beam.power_W
    refused = two.irradiance_map("SCREEN", source_id="A")
    assert refused["status"] == "UNAVAILABLE" and refused["reason"] == "GAUSSIAN_NOT_AVAILABLE"
