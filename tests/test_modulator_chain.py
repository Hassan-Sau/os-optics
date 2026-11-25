"""The worked chain generator -> pupil -> focal field -> detector, each stage a separate model, against numbers derived by hand here. Bounds were set before the first run,
except the array's efficiency bound, which a first run showed had ignored the 83.8 % a perfect spot holds in its Airy radius.

Setup: 852 nm, f = 200 mm, an 8 mm round pupil on a phase modulator of 12.5 um pixels, a camera of 6.5 um pixels with stated values, a 0.1 ms exposure.

Hand derivation, for 1 nW in one spot:

* lambda f = 0.1704 mm^2; focal step lambda f / (N p) = 0.1704 / (2048 x 0.0125) = 6.65625 um; Airy radius 1.22 lambda f / D = 25.986 um;
* peak irradiance I0 = P (pi D^2 / 4) / (lambda f)^2 = 1e-9 x 50.265 / 0.029036 = 1.7311e-6 W/mm^2;
* a pixel on the peak averages I over a square of side p = 6.5 um: I = I0 (1 - x^2 / 4 + ...), x = pi D r / (lambda f), <r^2> = p^2 / 6, so the factor is
  1 - (pi D / (lambda f))^2 p^2 / 24 = 1 - 21755 x 4.225e-5 / 24 = 0.9617, and the pixel's power is 1.7311e-6 x 4.225e-5 x 0.9617 = 7.034e-11 W;
* photon energy h c / lambda = 2.33153e-19 J: 7.034e-11 x 1e-4 / 2.33153e-19 = 3.017e4 photons; QE 0.6: 1.810e4 electrons; dark 100 e/s x 1e-4 s = 0.01 e;
* variance 18100 + 0.01 + 2^2 = 18104; SNR 18100 / sqrt(18104) = 134.5; at 6 e/ADU 3017 ADU, below the 16-bit limit and the 30000 e well.

A blazed ramp of 8 pixels per period sends that spot to u = lambda f / (8 p) = 1.704 mm with sinc^2(1/8) = 0.9496 of its peak: 2.865e4 photons on the pixel there.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from os_optics.results import detector_response, pupil_field

LAMBDA_NM = 852.0
F_MM = 200.0
D_MM = 8.0
PITCH = 0.0125
SOURCE = {"kind": "CUSTOMER_STATEMENT", "reference": "values chosen for this worked example; no real camera"}
CAMERA = {"quantum_efficiency": {"rows": [{"wavelength_nm": 800.0, "qe": 0.6}, {"wavelength_nm": 900.0, "qe": 0.6}], "source": SOURCE},
          "exposure_s": 1e-4, "pixel_pitch_mm": [0.0065, 0.0065], "fill_factor": 1.0, "full_well_e": 30000.0, "read_noise_e": 2.0, "dark_current_e_per_s": 100.0,
          "gain_e_per_adu": 6.0, "bit_depth": 16, "characterization": SOURCE}
# The hand derivation of the module docstring.
HAND_PHOTONS_ON_PEAK = 3.017e4
HAND_SINC2 = 0.9496


def slm(power: float, mask: dict | None = None, half: tuple[float, float] | None = None, fft: int = 4096, center: tuple[float, float] = (0.0, 0.0)) -> dict:
    doc = {"wavelength_nm": LAMBDA_NM, "focal_length_mm": F_MM, "sampling": {"pixel_count": [640, 640], "pitch_mm": [PITCH, PITCH], "fill_factor": 1.0, "fft_size": [fft, fft]},
           "aperture": {"shape": "CIRCLE", "diameter_mm": D_MM}, "illumination": {"kind": "UNIFORM", "power_w": power}}
    if mask is not None:
        doc["mask"] = mask
    if half is not None:
        doc["output"] = {"half_extent_mm": list(half), "center_uv_mm": list(center)}
    return doc


def test_one_spot_gives_the_hand_derived_numbers_at_every_stage():
    field = pupil_field(slm(1e-9, half=(0.3, 0.3)))
    assert pupil_field(slm(1e-9, half=(0.1, 0.1), fft=2048))["sampling"]["grid_step_mm"][0] == pytest.approx(0.00665625, rel=1e-9)
    peak = max(max(row) for row in field["coherent_W_per_mm2"])
    assert peak == pytest.approx(1.7311e-6, rel=2e-3)
    assert field["power"]["window_W"] + field["power"]["higher_orders_W"] == pytest.approx(1e-9, rel=1e-12)
    camera = detector_response(field, {**CAMERA, "pixel_count": [33, 33]})
    assert camera["status"] == "OK" and camera["unavailable_quantities"] == []
    q = camera["quantities"]
    center = 16  # an odd array: its middle pixel lies on the axis
    assert camera["pixels"]["u_mm"][center] == 0.0
    assert q["pixel_power_W"]["values"][center][center] == pytest.approx(7.034e-11, rel=3e-3)
    assert q["photons"]["values"][center][center] == pytest.approx(HAND_PHOTONS_ON_PEAK, rel=3e-3)
    assert q["signal_electrons"]["values"][center][center] == pytest.approx(0.6 * HAND_PHOTONS_ON_PEAK, rel=3e-3)
    assert q["dark_electrons"]["value"] == pytest.approx(0.01, rel=1e-12)
    assert q["total_noise_variance_e2"]["values"][center][center] == pytest.approx(0.6 * HAND_PHOTONS_ON_PEAK + 4.01, rel=3e-3)
    assert q["snr"]["values"][center][center] == pytest.approx(134.5, rel=3e-3)
    assert q["adu"]["values"][center][center] == pytest.approx(0.6 * HAND_PHOTONS_ON_PEAK / 6.0, rel=3e-3)
    assert camera["totals"]["saturated_pixels"] == 0 and camera["totals"]["adc_clipped_pixels"] == 0


def test_the_power_on_the_pixels_is_the_power_of_the_field_they_cover():
    """No stage invents or loses power: a pixel array covering the grid reads the grid's power, a 1 % box of tails aside."""

    field = pupil_field(slm(1e-9, half=(0.3, 0.3)))
    camera = detector_response(field, {**CAMERA, "pixel_pitch_mm": [0.0065, 0.0065]})
    grid_power = field["grid_power_coherent_W"]
    assert grid_power + field["power_outside_grid_W"] == pytest.approx(1e-9, rel=1e-9)
    assert camera["totals"]["power_on_pixels_W"] == pytest.approx(grid_power, rel=0.01)
    assert camera["totals"]["photons"] == pytest.approx(camera["totals"]["power_on_pixels_W"] * 1e-4 / 2.33153e-19, rel=1e-4)


def test_a_ramp_moves_the_spot_and_the_detector_finds_it_there():
    """+x tilt -> +u on the focal plane -> the same pixel coordinates on the camera; the pixel step's sinc^2 scales its photons."""

    spot = 0.1704 / (8 * PITCH)
    assert spot == pytest.approx(1.704, rel=1e-12)
    field = pupil_field(slm(1e-9, {"kind": "BLAZED_RAMP", "sin_theta": [spot / F_MM, 0.0]}, half=(0.3, 0.3), center=(spot, 0.0)))
    assert field["status"] == "OK" and field["validity"]["field_sine"] > spot / F_MM
    camera = detector_response(field, {**CAMERA, "pixel_count": [33, 33], "center_uv_mm": [spot, 0.0]})
    photons = np.array(camera["quantities"]["photons"]["values"])
    j, i = np.unravel_index(int(np.argmax(photons)), photons.shape)
    assert (i, j) == (16, 16) and camera["pixels"]["u_mm"][16] == pytest.approx(spot, rel=1e-12)
    assert photons[j, i] == pytest.approx(HAND_SINC2 * HAND_PHOTONS_ON_PEAK, rel=5e-3)


def test_a_five_by_five_array_reaches_the_camera_at_its_targets():
    """25 nW into 5 x 5 spots at a pitch of 46 camera pixels (0.299 mm, so each target lies on a pixel's centre): at most the 1 nW per spot of the derivation on each peak pixel; a
    perfect array with the mask's diffraction efficiency of about 90 % gives about 0.9 x 3.017e4 photons there. Bounds: every spot between 0.8 and 1 of 3.017e4; the brightest pixel
    near each target within one camera pixel."""

    spots = [{"u_mm": (i - 2) * 0.299, "v_mm": (j - 2) * 0.299} for j in range(5) for i in range(5)]
    field = pupil_field(slm(25e-9, {"kind": "SPOT_ARRAY", "spots": spots, "iterations": 30}))
    assert field["array"]["uniformity_power"] > 0.95
    assert field["validity"]["marginal_ray_sine"] < 0.05
    camera = detector_response(field, {**CAMERA, "pixel_count": [201, 201]})
    u, v = np.array(camera["pixels"]["u_mm"]), np.array(camera["pixels"]["v_mm"])
    photons = np.array(camera["quantities"]["photons"]["values"])
    values = []
    for spot in spots:
        rows = np.abs(v - spot["v_mm"]) < 0.05
        columns = np.abs(u - spot["u_mm"]) < 0.05
        patch = photons[np.ix_(rows, columns)]
        j, i = np.unravel_index(int(np.argmax(patch)), patch.shape)
        assert abs(u[columns][i] - spot["u_mm"]) <= 0.0065 and abs(v[rows][j] - spot["v_mm"]) <= 0.0065
        values.append(float(patch[j, i]))
    assert 0.8 * HAND_PHOTONS_ON_PEAK < min(values) and max(values) <= HAND_PHOTONS_ON_PEAK
    assert (max(values) - min(values)) / (max(values) + min(values)) < 0.05  # the array's stated uniformity survives the pixels


def test_sixteen_independent_tones_keep_their_spacing_power_and_detector_counts():
    """Tones of 70 to 100 MHz (2 MHz steps) from a generator of 650 m/s acoustic velocity (test values): sin(theta) = lambda f_RF / v, spacing lambda 2 MHz / v = 2.6215 mrad, so
    524.3 um on the focal plane; relayed about 85 MHz. Each tone of 1 nW and w = 1.5 mm focuses to w_f = lambda f / (pi w) = 36.16 um, peak 2 P / (pi w_f^2) = 4.868e-7 W/mm^2, which is
    the same for every tone: they add in irradiance (different optical frequencies), not in field. A spot's peak is the largest focal sample near it, at most half a step of 6.656 um away:
    within 2 (0.5 h / w_f)^2 = 1.7 % of the closed form."""

    velocity_mm_per_s = 650e3
    tones_hz = [70e6 + 2e6 * i for i in range(16)]
    lam_mm = LAMBDA_NM * 1e-6
    beams = [{"label": f"{f / 1e6:.0f} MHz", "power_w": 1e-9, "radius_mm": [1.5, 1.5], "direction_cosines": [lam_mm * (f - 85e6) / velocity_mm_per_s, 0.0]} for f in tones_hz]
    assert beams[1]["direction_cosines"][0] - beams[0]["direction_cosines"][0] == pytest.approx(2.6215e-3, rel=1e-4)
    field = pupil_field({"wavelength_nm": LAMBDA_NM, "focal_length_mm": F_MM, "sampling": {"pixel_count": [1024, 1024], "pitch_mm": [PITCH, PITCH], "fft_size": [2048, 2048]},
                         "illumination": {"kind": "BEAMS", "beams": beams, "composition": "INCOHERENT"}})
    assert field["status"] == "OK" and field["composition"] == "INCOHERENT_SUM" and field["quantities"]["focal_phase"]["status"] == "UNAVAILABLE"
    spacing = [b["u_mm"] - a["u_mm"] for a, b in zip(field["spots"], field["spots"][1:])]
    assert all(s == pytest.approx(0.5243, rel=1e-3) for s in spacing)
    for row in field["spots"]:
        assert row["peak_W_per_mm2"] == pytest.approx(4.868e-7, rel=0.017)
    assert field["power"]["window_W"] == pytest.approx(16e-9, rel=1e-3)
    assert field["array"]["uniformity_power"] > 0.999
    camera = detector_response(field, {**CAMERA, "pixel_pitch_mm": [0.0065, 0.0065]})
    assert camera["composition"] == "INCOHERENT" and camera["totals"]["power_on_pixels_W"] == pytest.approx(16e-9, rel=0.01)


def test_each_quantity_says_how_it_is_known():
    field = pupil_field({**slm(1e-9, half=(0.2, 0.2)), "device_efficiency": {"value": 0.9, "source": SOURCE}})
    statuses = {name: q["status"] for name, q in field["quantities"].items()}
    assert statuses == {"pupil_power_W": "MODELED", "mask_phase": "MODELED", "device_efficiency": "EXTERNALLY_CHARACTERIZED", "polarization": "UNAVAILABLE",
                        "focal_irradiance": "MODELED", "focal_phase": "MODELED"}
    camera = detector_response(field, {"exposure_s": 1e-4, "pixel_pitch_mm": [0.0065, 0.0065], "fill_factor": 1.0})
    assert camera["inputs"]["exposure_s"]["status"] == "EXTERNALLY_CHARACTERIZED" and camera["inputs"]["read_noise_e"]["status"] == "UNAVAILABLE"
    assert camera["quantities"]["photons"]["status"] == "MODELED" and camera["quantities"]["snr"]["status"] == "UNAVAILABLE"


def test_missing_detector_data_changes_no_field_result():
    field = pupil_field(slm(1e-9, half=(0.2, 0.2)))
    before = dict(field)
    camera = detector_response(field, {"pixel_pitch_mm": [0.0065, 0.0065], "fill_factor": 1.0})
    assert field == before and camera["status"] == "OK"
    assert camera["quantities"]["pixel_power_W"]["status"] == "MODELED"
    assert "photons" in camera["unavailable_quantities"] and "snr" in camera["unavailable_quantities"]


def test_a_chain_above_the_scalar_limit_keeps_its_upstream_stages():
    """NA 0.25: refused at the focal stage, stated at the pupil stage, and the camera is not fed anything in the field's place."""

    field = pupil_field({"wavelength_nm": LAMBDA_NM, "focal_length_mm": 100.0, "sampling": {"pixel_count": [512, 512], "pitch_mm": [0.1, 0.1]},
                         "aperture": {"shape": "CIRCLE", "diameter_mm": 50.0}, "illumination": {"kind": "UNIFORM", "power_w": 1e-3},
                         "mask": {"kind": "BLAZED_RAMP", "sin_theta": [1e-3, 0.0]}, "output": {"half_extent_mm": [0.1, 0.1]}})
    assert field["status"] == "PARTIAL" and field["quantities"]["focal_irradiance"]["reason"] == "NA_ABOVE_SCALAR_LIMIT"
    assert field["mask"]["kind"] == "BLAZED_RAMP" and field["power"]["delivered_W"] == pytest.approx(1e-3, rel=1e-12)
    camera = detector_response(field, CAMERA)
    assert camera["status"] == "UNAVAILABLE" and camera["upstream_reason"] == "NA_ABOVE_SCALAR_LIMIT" and "quantities" not in camera


def test_the_hand_derivation_is_the_formula():
    """The literals of the module docstring, computed once more from their formulas."""

    lam_f = LAMBDA_NM * 1e-6 * F_MM
    area = math.pi * D_MM * D_MM / 4.0
    i0 = 1e-9 * area / (lam_f * lam_f)
    factor = 1.0 - (math.pi * D_MM / lam_f) ** 2 * 0.0065 ** 2 / 24.0
    energy = 6.62607015e-34 * 299792458.0 / (LAMBDA_NM * 1e-9)
    assert i0 == pytest.approx(1.7311e-6, rel=2e-4) and factor == pytest.approx(0.9617, rel=2e-4) and energy == pytest.approx(2.33153e-19, rel=1e-5)
    assert i0 * 0.0065 ** 2 * factor * 1e-4 / energy == pytest.approx(HAND_PHOTONS_ON_PEAK, rel=1e-3)
    assert np.sinc(1.0 / 8.0) ** 2 == pytest.approx(HAND_SINC2, rel=1e-4)
    assert (2.0 * 1e-9 / (math.pi * (lam_f / (math.pi * 1.5)) ** 2)) == pytest.approx(4.868e-7, rel=1e-3)
