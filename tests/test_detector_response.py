"""Detector response of an irradiance map (``os_optics.results.detector``), against closed forms derived here: the photon count of a known
power, wavelength and exposure; the integral of a Gaussian over a pixel (erf); the sums of the noise variances. Tolerances were set before the
first run: exact arithmetic to 1e-12. A bilinear integral of a Gaussian is a trapezoid rule, of known error: over an interval, per axis, ``h^2 / 12 [f'(b) - f'(a)]``,
so its relative error at a point is about ``h^2 / 12 |f'' / f|`` with ``f'' / f = (4 x / w^2)^2 - 4 / w^2``: at 20 samples per 1/e^2 radius 0.1 % near the axis and a few percent
far in the tail. The cubic convolution of the module must stay within that bound."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from os_optics.cli import main as cli_main
from os_optics.local import json_api
from os_optics.results import DetectorError, detector_response, load_run

from local_helpers import plane, project, run, source

# CODATA 2019 exact values, typed here on purpose: independent of the module's constants.
H = 6.62607015e-34
C = 299792458.0
LAMBDA_NM = 852.0
PHOTON_J = H * C / (LAMBDA_NM * 1e-9)
SOURCE = {"kind": "DATASHEET", "reference": "camera datasheet rev A (test values)"}


def uniform_map(irradiance_w_per_mm2: float, half_mm: float = 1.0, samples: int = 41) -> dict:
    axis = [-half_mm + 2.0 * half_mm * i / (samples - 1) for i in range(samples)]
    rows = [[irradiance_w_per_mm2] * samples for _ in axis]
    return {"status": "OK", "model": "test", "wavelength_nm": LAMBDA_NM, "grid": {"u_mm": axis, "v_mm": axis, "shape": [samples, samples]},
            "incoherent_W_per_mm2": rows, "coherent_W_per_mm2": None, "coherent_reason": "COHERENCE_UNDEFINED"}


def gaussian_map(power_w: float, w_mm: float, half_mm: float, step_mm: float, center=(0.0, 0.0)) -> dict:
    n = int(round(2.0 * half_mm / step_mm)) + 1
    axis = [-half_mm + i * step_mm for i in range(n)]
    peak = 2.0 * power_w / (math.pi * w_mm * w_mm)
    rows = [[peak * math.exp(-2.0 * ((u - center[0]) ** 2 + (v - center[1]) ** 2) / (w_mm * w_mm)) for u in axis] for v in axis]
    return {"status": "OK", "model": "test", "wavelength_nm": LAMBDA_NM, "grid": {"u_mm": axis, "v_mm": axis, "shape": [n, n]},
            "incoherent_W_per_mm2": rows, "coherent_W_per_mm2": rows}


FULL = {"quantum_efficiency": {"rows": [{"wavelength_nm": 800.0, "qe": 0.5}, {"wavelength_nm": 900.0, "qe": 0.3}], "source": SOURCE},
        "exposure_s": 2e-3, "pixel_pitch_mm": [0.01, 0.01], "fill_factor": 1.0, "pixel_count": [4, 4], "full_well_e": 3.0e4, "read_noise_e": 2.0,
        "dark_current_e_per_s": 100.0, "gain_e_per_adu": 0.5, "bit_depth": 12, "characterization": SOURCE}
#: QE at 852 nm, linear between the rows.
QE_852 = 0.5 + (852.0 - 800.0) / 100.0 * (0.3 - 0.5)


def test_photon_count_of_a_known_power_wavelength_and_exposure():
    """A pixel of (10 um)^2: P = I p^2, N = P t lambda / (h c)."""

    irradiance = 2e-9  # W/mm^2
    doc = detector_response(uniform_map(irradiance), FULL)
    assert doc["status"] == "OK" and doc["unavailable_quantities"] == []
    pixel_power = irradiance * 0.01 * 0.01
    photons = pixel_power * 2e-3 / PHOTON_J
    assert doc["quantities"]["photon_energy_J"]["value"] == pytest.approx(PHOTON_J, rel=1e-12)
    assert doc["quantities"]["pixel_power_W"]["values"][1][2] == pytest.approx(pixel_power, rel=1e-12)
    assert doc["quantities"]["photons"]["values"][0][0] == pytest.approx(photons, rel=1e-12)
    assert doc["quantities"]["signal_electrons"]["values"][3][3] == pytest.approx(photons * QE_852, rel=1e-12)
    assert doc["quantities"]["photon_flux_density"]["values"][0][0] == pytest.approx(irradiance / PHOTON_J, rel=1e-12)
    assert doc["totals"]["pixel_count"] == 16 and doc["totals"]["power_on_pixels_W"] == pytest.approx(16 * pixel_power, rel=1e-12)
    assert doc["composition"] == "INCOHERENT"  # the only map of the document, and why the coherent one is withheld
    assert doc["coherent_reason"] == "COHERENCE_UNDEFINED"
    assert doc["inputs"]["quantum_efficiency"] == {"status": "EXTERNALLY_CHARACTERIZED", "unit": None, "value": pytest.approx(QE_852, rel=1e-12), "source": SOURCE}
    assert doc["inputs"]["read_noise_e"]["source"] == SOURCE and doc["inputs"]["exposure_s"]["source"] is None


def test_noise_variances_add_and_give_the_snr():
    irradiance = 1e-9
    doc = detector_response(uniform_map(irradiance), FULL)["quantities"]
    signal = irradiance * 1e-4 * 2e-3 / PHOTON_J * QE_852
    dark, read = 100.0 * 2e-3, 2.0 * 2.0
    assert doc["dark_electrons"]["value"] == pytest.approx(dark, rel=1e-12)
    assert doc["shot_noise_variance_e2"]["values"][0][0] == pytest.approx(signal, rel=1e-12)
    assert doc["dark_noise_variance_e2"]["value"] == pytest.approx(dark, rel=1e-12)
    assert doc["read_noise_variance_e2"]["value"] == pytest.approx(read, rel=1e-12)
    total = signal + dark + read
    assert doc["total_noise_variance_e2"]["values"][0][0] == pytest.approx(total, rel=1e-12)
    assert doc["snr"]["values"][0][0] == pytest.approx(signal / math.sqrt(total), rel=1e-12)
    assert doc["adu"]["values"][0][0] == pytest.approx((signal + dark) / 0.5, rel=1e-12)
    assert doc["saturated"]["values"][0][0] is False and doc["adc_clipped"]["values"][0][0] is False


def test_saturation_and_adc_limit():
    """Well 30000 e at gain 0.5 e/ADU is 60000 ADU, above the 4095 of 12 bits: the ADC clips long before the well fills."""

    bright = detector_response(uniform_map(1.0), FULL)
    q = bright["quantities"]
    assert q["saturated"]["values"][0][0] is True and q["adc_clipped"]["values"][0][0] is True
    assert q["adu"]["values"][0][0] == 4095.0
    assert q["snr"]["values"][0][0] is None  # no SNR at a saturated pixel
    assert bright["totals"]["saturated_pixels"] == 16
    # the ADC limit reached below the well: signal + dark = 2100 e at gain 0.5 is 4200 ADU
    signal = 2100.0 - 0.2
    irradiance = signal * PHOTON_J / (QE_852 * 1e-4 * 2e-3)
    q = detector_response(uniform_map(irradiance), FULL)["quantities"]
    assert q["saturated"]["values"][0][0] is False and q["adc_clipped"]["values"][0][0] is True and q["adu"]["values"][0][0] == 4095.0
    assert q["snr"]["values"][0][0] == pytest.approx(signal / math.sqrt(signal + 0.2 + 4.0), rel=1e-12)
    # a dark pixel with no noise stated as present: the variance is 0 and the SNR has no value
    quiet = {**FULL, "read_noise_e": 0.0, "dark_current_e_per_s": 0.0}
    assert detector_response(uniform_map(0.0), quiet)["quantities"]["snr"]["values"][0][0] is None


@pytest.mark.parametrize("step_per_w, fill, pixel", [(20, 1.0, 0.5), (20, 0.5, 0.5), (40, 1.0, 0.5)])
def test_pixel_integration_of_a_gaussian_against_erf(step_per_w, fill, pixel):
    """P / 4 [erf(sqrt(2) (u + a/2) / w) - erf(sqrt(2) (u - a/2) / w)] [the same in v], with a the active width, within the trapezoid rule's error
    (module docstring): relative to the pixel near the axis, relative to the central pixel everywhere."""

    w, power = 1.0, 1e-3
    doc = detector_response(gaussian_map(power, w, half_mm=4.0, step_mm=w / step_per_w), {"pixel_pitch_mm": [pixel, pixel], "fill_factor": fill, "pixel_count": [9, 9]},
                            composition="INCOHERENT")
    active = pixel * math.sqrt(fill)

    def axis(c: float) -> float:
        return 0.5 * (math.erf(math.sqrt(2.0) * (c + active / 2.0) / w) - math.erf(math.sqrt(2.0) * (c - active / 2.0) / w))

    h = w / step_per_w
    near_axis = 2.0 * h * h / 12.0 * 4.0 / (w * w)  # both axes, |f'' / f| = 4 / w^2 at the axis, with a factor 2 of margin
    near_axis *= 2.0
    centers = [(i - 4) * pixel for i in range(9)]
    values = doc["quantities"]["pixel_power_W"]["values"]
    central = power * axis(0.0) * axis(0.0)
    for j, v in enumerate(centers):
        for i, u in enumerate(centers):
            want = power * axis(u) * axis(v)
            if abs(u) <= 0.5 and abs(v) <= 0.5:
                assert values[j][i] == pytest.approx(want, rel=near_axis), (u, v)
            assert abs(values[j][i] - want) <= near_axis * central, (u, v)
    assert doc["pixels"]["active_width_mm"] == pytest.approx([active, active], rel=1e-12)


def test_the_integration_error_falls_at_least_as_the_step_squared():
    """A cubic convolution of third order: halving the step divides the error by at least 4."""

    w, power = 1.0, 1e-3
    errors = []
    for step in (0.1, 0.05):
        doc = detector_response(gaussian_map(power, w, 4.0, step), {"pixel_pitch_mm": [0.5, 0.5], "fill_factor": 1.0, "pixel_count": [1, 1]}, composition="COHERENT")
        exact = power * math.erf(math.sqrt(2.0) * 0.25) ** 2
        errors.append(abs(doc["quantities"]["pixel_power_W"]["values"][0][0] - exact))
    assert errors[0] / errors[1] >= 4.0


def test_a_decentred_rectangular_pixel_array_reads_the_right_cells():
    w, power = 1.0, 1e-3
    doc = detector_response(gaussian_map(power, w, 4.0, 0.05), {"pixel_pitch_mm": [0.5, 0.25], "fill_factor": 1.0, "pixel_count": [3, 2], "center_uv_mm": [1.0, -0.5]},
                            composition="COHERENT")
    grid = doc["pixels"]
    assert grid["shape"] == [2, 3] and grid["u_mm"] == pytest.approx([0.5, 1.0, 1.5]) and grid["v_mm"] == pytest.approx([-0.625, -0.375])

    def part(low: float, high: float) -> float:
        return 0.5 * (math.erf(math.sqrt(2.0) * high / w) - math.erf(math.sqrt(2.0) * low / w))

    want = power * part(1.25, 1.75) * part(-0.5, -0.25)
    # the trapezoid rule's error at step 0.05 (module docstring): |f'' / f| = (4 x 1.5)^2 - 4 = 32 at the pixel's centre along u
    assert doc["quantities"]["pixel_power_W"]["values"][1][2] == pytest.approx(want, rel=2.0 * 0.05 * 0.05 / 12.0 * 32.0)


def test_an_airy_peak_pixel_about_as_wide_as_the_nyquist_step():
    """(2 J1(x) / x)^2 sampled at the focal step lambda f / (2 D) of a just-resolved pupil, a pixel of that width on the peak, against the midpoint rule on 2000^2 points of the closed form:
    within 4 % (the coarsest sampling a focal field has: a first run gave 3.2 %), and the reported estimate is at least the error made."""

    special = pytest.importorskip("scipy.special")
    scale, diameter = 0.1704, 8.0
    step = scale / (2.0 * diameter)
    k = math.pi * diameter / scale
    n = 201
    axis = [(i - (n - 1) / 2) * step for i in range(n)]

    def airy(radius):
        x = k * radius
        return ((2.0 * special.j1(x) / x) ** 2) if x > 1e-9 else 1.0

    rows = [[float(airy(math.hypot(u, v))) for u in axis] for v in axis]
    field = {"status": "OK", "wavelength_nm": LAMBDA_NM, "grid": {"u_mm": axis, "v_mm": axis, "shape": [n, n]}, "coherent_W_per_mm2": rows}
    doc = detector_response(field, {"pixel_pitch_mm": [step, step], "fill_factor": 1.0, "pixel_count": [1, 1]})
    fine = (np.arange(2000) + 0.5) / 2000 * step - step / 2.0
    uu, vv = np.meshgrid(fine, fine)
    x = k * np.hypot(uu, vv)
    exact = float(((2.0 * special.j1(x) / x) ** 2).mean()) * step * step
    value = doc["quantities"]["pixel_power_W"]["values"][0][0]
    assert value == pytest.approx(exact, rel=0.04)
    estimate = doc["pixels"]["integration"]["relative_error_estimate"]
    assert doc["pixels"]["integration"]["method"] == "CUBIC_CONVOLUTION" and estimate >= abs(value - exact) / exact


def test_a_histogram_map_is_constant_in_each_bin():
    edges = [0.0, 1.0, 2.0]
    doc = {"status": "OK", "model": "test", "wavelength_nm": LAMBDA_NM, "incoherent_W_per_mm2": [[1.0, 3.0], [5.0, 7.0]], "coherent_W_per_mm2": None,
           "grid": {"u_mm": [0.5, 1.5], "v_mm": [0.5, 1.5], "shape": [2, 2], "u_edges_mm": edges, "v_edges_mm": edges}}
    reply = detector_response(doc, {"pixel_pitch_mm": [1.0, 1.0], "fill_factor": 1.0, "pixel_count": [1, 1], "center_uv_mm": [1.0, 1.0]})
    # the unit square centred on the corner of the four bins holds a quarter of each
    assert reply["quantities"]["pixel_power_W"]["values"][0][0] == pytest.approx((1.0 + 3.0 + 5.0 + 7.0) / 4.0, rel=1e-12)


@pytest.mark.parametrize("left_out, needing", [
    ("exposure_s", {"photons", "signal_electrons", "dark_electrons", "shot_noise_variance_e2", "dark_noise_variance_e2", "total_noise_variance_e2", "snr", "saturated", "adu", "adc_clipped"}),
    ("quantum_efficiency", {"signal_electrons", "shot_noise_variance_e2", "total_noise_variance_e2", "snr", "saturated", "adu", "adc_clipped"}),
    ("pixel_pitch_mm", {"pixel_power_W", "photons", "signal_electrons", "shot_noise_variance_e2", "total_noise_variance_e2", "snr", "saturated", "adu", "adc_clipped"}),
    ("fill_factor", {"pixel_power_W", "photons", "signal_electrons", "shot_noise_variance_e2", "total_noise_variance_e2", "snr", "saturated", "adu", "adc_clipped"}),
    ("read_noise_e", {"read_noise_variance_e2", "total_noise_variance_e2", "snr"}),
    ("dark_current_e_per_s", {"dark_electrons", "dark_noise_variance_e2", "total_noise_variance_e2", "snr", "saturated", "adu", "adc_clipped"}),
    ("full_well_e", {"saturated", "adu", "adc_clipped"}),
    ("gain_e_per_adu", {"adu", "adc_clipped"}),
    ("bit_depth", {"adu", "adc_clipped"}),
])
def test_a_parameter_left_out_leaves_exactly_what_needs_it_unavailable(left_out, needing):
    irradiance = uniform_map(1e-9)
    complete = detector_response(irradiance, FULL)
    doc = detector_response(irradiance, {k: v for k, v in FULL.items() if k != left_out})
    assert doc["status"] == "OK"
    assert set(doc["unavailable_quantities"]) == needing
    assert doc["inputs"][left_out]["status"] == "UNAVAILABLE"
    for name, quantity in doc["quantities"].items():
        if name in needing:
            assert quantity["status"] == "UNAVAILABLE" and left_out in quantity["missing"] and quantity["reason"] == "PARAMETER_NOT_STATED"
            assert "values" not in quantity and "value" not in quantity
        elif name == "snr" and left_out == "full_well_e":  # without a well no pixel is known to saturate; the values below it are unchanged
            assert quantity["values"] == complete["quantities"]["snr"]["values"]
        else:
            assert quantity == complete["quantities"][name], name


def test_an_empty_detector_still_gives_the_photon_flux_density():
    doc = detector_response(uniform_map(1e-5), {})
    assert doc["status"] == "OK" and doc["pixels"] is None and doc["totals"] == {}
    assert doc["quantities"]["photon_flux_density"]["status"] == "MODELED"
    assert set(doc["unavailable_quantities"]) == set(doc["quantities"]) - {"photon_energy_J", "photon_flux_density"}


def test_quantum_efficiency_is_not_extrapolated():
    table = {"rows": [{"wavelength_nm": 400.0, "qe": 0.8}, {"wavelength_nm": 700.0, "qe": 0.6}], "source": SOURCE}
    doc = detector_response(uniform_map(1e-5), {**FULL, "quantum_efficiency": table})
    q = doc["quantities"]
    assert doc["inputs"]["quantum_efficiency"]["reason"] == "WAVELENGTH_OUTSIDE_QE_TABLE"
    assert q["signal_electrons"]["reason"] == "WAVELENGTH_OUTSIDE_QE_TABLE" and q["photons"]["status"] == "MODELED"


def test_a_field_with_no_map_is_not_replaced():
    upstream = {"status": "UNAVAILABLE", "reason": "NO_ARRIVALS", "message": "no beam counts as arriving at this plane"}
    doc = detector_response(upstream, FULL)
    assert doc["status"] == "UNAVAILABLE" and doc["reason"] == "IRRADIANCE_UNAVAILABLE" and doc["upstream_reason"] == "NO_ARRIVALS"
    assert "quantities" not in doc


def test_two_different_maps_need_a_composition():
    field = gaussian_map(1e-3, 1.0, 4.0, 0.1)
    field["coherent_W_per_mm2"] = [[2.0 * x for x in row] for row in field["incoherent_W_per_mm2"]]
    detector = {"pixel_pitch_mm": [0.5, 0.5], "fill_factor": 1.0, "pixel_count": [1, 1]}
    refused = detector_response(field, detector)
    assert refused["status"] == "UNAVAILABLE" and refused["reason"] == "COMPOSITION_REQUIRED"
    coherent = detector_response(field, detector, "COHERENT")["quantities"]["pixel_power_W"]["values"][0][0]
    incoherent = detector_response(field, detector, "INCOHERENT")["quantities"]["pixel_power_W"]["values"][0][0]
    assert coherent == pytest.approx(2.0 * incoherent, rel=1e-12)
    assert detector_response(field, detector, "NEITHER")["reason"] == "COMPOSITION_INVALID"
    withheld = {**field, "coherent_W_per_mm2": None, "coherent_reason": "FRINGES_UNDERSAMPLED"}
    assert detector_response(withheld, detector, "COHERENT")["upstream_reason"] == "FRINGES_UNDERSAMPLED"


def test_a_pixel_array_beyond_the_map_is_not_read_as_dark():
    doc = detector_response(uniform_map(1e-5, half_mm=0.02), {**FULL, "pixel_count": [100, 100]})
    assert doc["status"] == "OK"
    assert doc["quantities"]["pixel_power_W"]["reason"] == "PIXEL_GRID_OUTSIDE_MAP" and doc["pixels"] is None
    assert doc["quantities"]["photon_flux_density"]["status"] == "MODELED"
    assert doc["quantities"]["dark_electrons"]["status"] == "MODELED"  # needs no pixel array on the map
    # no pixel_count: as many whole pixels as fit
    fitted = detector_response(uniform_map(1e-5, half_mm=0.025), {k: v for k, v in FULL.items() if k != "pixel_count"})
    assert fitted["pixels"]["count"] == [5, 5]


def test_an_invalid_detector_is_refused():
    with pytest.raises(DetectorError):
        detector_response(uniform_map(1e-5), {"fill_factor": 2.0})
    reply = json_api.detector_response(uniform_map(1e-5), {"bit_depth": 0})
    assert reply["ok"] is False and reply["error"]["code"] == "DETECTOR_INVALID"
    assert json_api.detector_response(uniform_map(1e-5), {})["ok"] is True


def test_a_value_stated_without_its_source_is_used_as_written_and_its_source_reads_null():
    """The same detector with and without where its values come from: every quantity is the same number; only the
    ``source`` of the stated inputs differs (null)."""

    irradiance = uniform_map(2.0e-4)
    bare = {k: v for k, v in FULL.items() if k != "characterization"}
    bare["quantum_efficiency"] = {"rows": FULL["quantum_efficiency"]["rows"]}
    stated, unstated = detector_response(irradiance, FULL), detector_response(irradiance, bare)
    assert unstated["quantities"] == stated["quantities"]
    assert {q["status"] for q in unstated["quantities"].values()} == {"MODELED"}  # read noise, full well, ADU: all computed
    for name in ("quantum_efficiency", "read_noise_e", "full_well_e", "dark_current_e_per_s", "gain_e_per_adu"):
        assert stated["inputs"][name]["source"] == SOURCE and unstated["inputs"][name]["source"] is None
        assert unstated["inputs"][name]["value"] == stated["inputs"][name]["value"]
        assert unstated["inputs"][name]["status"] == "EXTERNALLY_CHARACTERIZED"
    # the electrical response the same way
    electrical = {"bandwidth_Hz": 1.0e5, "bandwidth_definition": "FIRST_ORDER_3DB", "dark_current_A": 1.0e-9}
    with_source = detector_response(irradiance, {**bare, "electrical": {**electrical, "source": SOURCE}})
    without = detector_response(irradiance, {**bare, "electrical": electrical})
    assert with_source["inputs"]["electrical"]["source"] == SOURCE and without["inputs"]["electrical"]["source"] is None
    assert without["inputs"]["electrical"]["value"] == with_source["inputs"]["electrical"]["value"]
    assert without["quantities"] == with_source["quantities"]


# ------------------------------------------------------------------------------------------------- a run's irradiance


def _beam_project() -> dict:
    return project([source(position=(-100.0, 0.0, 0.0), wavelength_nm=LAMBDA_NM, power_w=2e-6, waist_mm=0.5)], [plane("CAM", (100.0, 0.0, 0.0), radius_mm=20.0)])


def test_a_runs_map_gives_the_photons_of_the_sources_power():
    """A pixel array holding the whole beam: every photon of the source's 2 uW (the Gaussian tail beyond it is below 1e-3)."""

    results = load_run(run(_beam_project()))
    camera = {"exposure_s": 1e-3, "pixel_pitch_mm": [0.05, 0.05], "fill_factor": 1.0}
    doc = results.detector_response("CAM", camera, half_extent_mm=1.5, samples=301)
    assert doc["status"] == "OK" and doc["observation_id"] == "CAM"
    assert doc["totals"]["power_on_pixels_W"] == pytest.approx(2e-6, rel=2e-3)
    assert doc["totals"]["photons"] == pytest.approx(2e-6 * 1e-3 / PHOTON_J, rel=2e-3)
    assert doc["quantities"]["signal_electrons"]["status"] == "UNAVAILABLE"
    assert results.detector_response("CAM", camera, wavelength_nm=532.0)["reason"] == "IRRADIANCE_UNAVAILABLE"


def test_the_command_line_reads_a_run_or_a_field(tmp_path, capsys):
    run_dir = tmp_path / "run"
    from os_optics.local import run_optical_project

    run_optical_project(_beam_project(), output_dir=str(run_dir))
    detector = tmp_path / "detector.json"
    detector.write_text(json.dumps({"pixel_pitch_mm": [0.05, 0.05], "fill_factor": 1.0, "exposure_s": 1e-3}), encoding="utf-8")
    assert cli_main(["results", "detector", str(run_dir), "--monitor", "CAM", "--detector", str(detector), "--samples", "101"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["quantities"]["photons"]["status"] == "MODELED"
    field = tmp_path / "field.json"
    field.write_text(json.dumps(uniform_map(1e-5)), encoding="utf-8")
    assert cli_main(["results", "detector", "--field", str(field), "--detector", str(detector)]) == 0
    assert json.loads(capsys.readouterr().out)["pixels"]["count"] == [40, 40]
    assert cli_main(["results", "detector", "--detector", str(detector)]) == 64
    capsys.readouterr()
    assert cli_main(["results", "detector", str(run_dir), "--monitor", "CAM", "--detector", str(detector), "--wavelength-nm", "532"]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "UNAVAILABLE"
    detector.write_text(json.dumps({"bit_depth": 0}), encoding="utf-8")
    assert cli_main(["results", "detector", "--field", str(field), "--detector", str(detector)]) == 64
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "DETECTOR_INVALID"


#: Irradiance documents of status OK that lack a member the response reads, or hold one of the wrong form, and the member
#: each refusal names.
MALFORMED_FIELDS = [
    ({"status": "OK", "wavelength_nm": 1064, "incoherent_W_per_mm2": [[1, 2], [3, 4]]}, "grid"),
    ({"status": "OK", "wavelength_nm": 1064, "grid": {"u_mm": [0.0, 0.1]}, "incoherent_W_per_mm2": [[1, 2], [3, 4]]}, "grid.v_mm"),
    ({"status": "OK", "wavelength_nm": 1064, "grid": {"u_mm": [0.0, 0.1], "v_mm": [0.0, 0.1]}, "incoherent_W_per_mm2": [[1, 2, 3], [3, 4, 5]]},
     "incoherent_W_per_mm2"),
    ({"status": "OK", "wavelength_nm": "1064", "grid": {"u_mm": [0.0, 0.1], "v_mm": [0.0, 0.1]}, "incoherent_W_per_mm2": [[1, 2], [3, 4]]},
     "wavelength_nm"),
]


@pytest.mark.parametrize("field, member", MALFORMED_FIELDS)
def test_a_malformed_irradiance_document_is_refused_by_name_not_as_an_internal_error(tmp_path, capsys, field, member):
    """A caller's own irradiance document (--field) that lacks the grid, a grid axis, holds a map of another shape or a
    wavelength that is no number: the library, the JSON API and the command refuse it with IRRADIANCE_INVALID naming the
    member (exit 64), never as an internal error of os-optics (exit 70, a traceback)."""
    detector = {"pixel_pitch_mm": [0.05, 0.05], "fill_factor": 1.0, "pixel_count": [4, 4]}
    with pytest.raises(DetectorError) as raised:
        detector_response(field, detector)
    assert raised.value.code == "IRRADIANCE_INVALID" and f"{member}:" in raised.value.message
    reply = json_api.detector_response(field, detector)
    assert reply["ok"] is False and reply["error"]["code"] == "IRRADIANCE_INVALID"
    (tmp_path / "field.json").write_text(json.dumps(field), encoding="utf-8")
    (tmp_path / "det.json").write_text(json.dumps(detector), encoding="utf-8")
    capsys.readouterr()
    code = cli_main(["results", "detector", "--field", str(tmp_path / "field.json"), "--detector", str(tmp_path / "det.json")])
    out, err = capsys.readouterr()
    assert code == 64, out
    assert json.loads(out)["error"]["code"] == "IRRADIANCE_INVALID"
    assert "INTERNAL_ERROR" not in out and "Traceback" not in err


def test_a_field_document_the_structure_check_misses_is_still_the_callers_input(tmp_path, capsys, monkeypatch):
    """Whatever shape of a --field document the response stumbles on, the document is the caller's own input, not input
    known to be good: a usage error naming the file (exit 64), not INTERNAL_ERROR (exit 70)."""
    import os_optics.results as results_module

    def stumbles(*args, **kwargs):
        raise KeyError("grid")

    monkeypatch.setattr(results_module, "detector_response", stumbles)
    (tmp_path / "field.json").write_text(json.dumps(uniform_map(1e-5)), encoding="utf-8")
    (tmp_path / "det.json").write_text(json.dumps({"pixel_pitch_mm": [0.05, 0.05], "fill_factor": 1.0}), encoding="utf-8")
    capsys.readouterr()
    code = cli_main(["results", "detector", "--field", str(tmp_path / "field.json"), "--detector", str(tmp_path / "det.json")])
    out, err = capsys.readouterr()
    assert code == 64 and json.loads(out)["error"]["code"] == "IRRADIANCE_INVALID"
    assert "field.json" in json.loads(out)["error"]["message"] and "Traceback" not in err
