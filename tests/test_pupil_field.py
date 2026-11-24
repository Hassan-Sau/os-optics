"""Scalar focal field of a pupil mask (``os_optics.results.pupil_field``), against closed forms derived here, with tolerances set before the
first run:

* uniform circular pupil: Airy radius ``1.22 lambda f / D`` (1 %), peak ``P A / (lambda f)^2`` (exact for the sampled area, 1e-9), encircled energy
  ``1 - J0^2 - J1^2`` at the first zero (1 %);
* blazed ramp of ``q`` pixels per period: spot at ``u = lambda f / (q p)``, peak ``sinc^2(1 / q) P A / (lambda f)^2`` (1e-9), window power ``fill sinc^2(sqrt(fill) / q)`` of
  the active power (1e-3 at 1024 pixels: the side lobes of the aperture leave the peak's envelope value by a share falling as 1 / N);
* binary 0-pi grating, whole periods on the grid: first order ``(2 / pi)^2`` = 40.5 % (1e-6), no zero order and no even orders;
* Parseval across pupil -> focal window, 1e-12; the replicas of the pixel grid summed directly against the ledger (1e-3);
* phases: ``U_f(0) = -i sqrt(P A) / (lambda f)`` (1e-9 rad) and a shift of the pupil by ``x0`` is ``exp(-i 2 pi x0 u / (lambda f))``;
* a Gaussian pupil: focal radius ``lambda f / (pi w)`` and peak ``2 P / (pi w_f^2)`` (1e-3).
"""

from __future__ import annotations

import hashlib
import io
import json
import math

import numpy as np
import pytest
from scipy import special

from os_optics.cli import main as cli_main
from os_optics.local import json_api
from os_optics.results import PupilFieldError, detector_response, load_run, pupil_field

from local_helpers import ALONG_X, plane, project, quat_multiply, run, source, turn_about_z

LAMBDA_NM = 852.0
LAMBDA_MM = LAMBDA_NM * 1e-6
F_MM = 200.0
SCALE = LAMBDA_MM * F_MM
SOURCE = {"kind": "CUSTOMER_STATEMENT", "reference": "test mask"}


def request(pixels=256, pitch=0.03125, *, power=1e-3, aperture=None, mask=None, fill=None, fft=None, illumination=None, output=None, focal=F_MM, efficiency=None) -> dict:
    doc = {"wavelength_nm": LAMBDA_NM, "focal_length_mm": focal,
           "sampling": {"pixel_count": [pixels, pixels], "pitch_mm": [pitch, pitch]},
           "illumination": illumination or {"kind": "UNIFORM", "power_w": power}}
    if fill is not None:
        doc["sampling"]["fill_factor"] = fill
    if fft is not None:
        doc["sampling"]["fft_size"] = [fft, fft]
    if aperture is not None:
        doc["aperture"] = aperture
    if mask is not None:
        doc["mask"] = mask
    if output is not None:
        doc["output"] = output
    if efficiency is not None:
        doc["device_efficiency"] = {"value": efficiency, "source": SOURCE}
    return doc


def at(doc: dict, u: float, v: float, name: str = "coherent_W_per_mm2") -> float:
    """The map's value at the grid point nearest ``(u, v)``."""

    grid = doc["grid"]
    return doc[name][int(np.argmin(np.abs(np.array(grid["v_mm"]) - v)))][int(np.argmin(np.abs(np.array(grid["u_mm"]) - u)))]


def nearest(axis: list[float], value: float) -> float:
    return axis[int(np.argmin(np.abs(np.array(axis) - value)))]


# ------------------------------------------------------------------------------------------------- a uniform circular pupil


D_MM = 8.0
CIRCLE = {"shape": "CIRCLE", "diameter_mm": D_MM}


def _airy() -> dict:
    return pupil_field(request(pixels=512, pitch=D_MM / 400.0, aperture=CIRCLE, fft=2048, output={"half_extent_mm": [0.2, 0.2], "include_field": True}))


def test_uniform_circular_pupil_gives_the_airy_pattern():
    doc = _airy()
    assert doc["status"] == "OK" and doc["composition"] == "COHERENT"
    u = np.array(doc["grid"]["u_mm"])
    v = np.array(doc["grid"]["v_mm"])
    map_ = np.array(doc["coherent_W_per_mm2"])
    dx = D_MM / 400.0
    du = SCALE / (2048 * dx)
    assert doc["sampling"]["grid_step_mm"] == pytest.approx([du, du], rel=1e-12)  # x = lambda f / (N dx)
    # the peak, for the sampled area of the opening (the number of cells inside it)
    xs = (np.arange(512) - 255.5) * dx
    xx, yy = np.meshgrid(xs, xs)
    area = float((xx * xx + yy * yy <= 16.0).sum()) * dx * dx
    assert area == pytest.approx(math.pi * 16.0, rel=2e-3)
    peak = 1e-3 * area / (SCALE * SCALE)
    assert float(map_.max()) == pytest.approx(peak, rel=1e-9)
    assert at(doc, 0.0, 0.0) == pytest.approx(peak, rel=1e-9)
    # the first zero along +u: the field is real up to the factor -i, so it changes sign there (linear between the two focal samples)
    j = int(np.argmin(np.abs(v)))
    center = int(np.argmin(np.abs(u)))
    along = np.array(doc["field_sqrt_W_per_mm2"]["im"][j])[center:]
    crossing = int(np.nonzero(along[:-1] * along[1:] <= 0.0)[0][0])
    first_zero = float(u[center + crossing] + du * along[crossing] / (along[crossing] - along[crossing + 1]))
    assert first_zero == pytest.approx(1.22 * SCALE / D_MM, rel=0.01)
    # the pattern itself along that axis, within the staircase of the sampled opening: (2 J1(x) / x)^2, x = pi D u / (lambda f)
    x = math.pi * D_MM * u[center + 1:center + 40] / SCALE
    np.testing.assert_allclose(map_[j, center + 1:center + 40] / map_[j, center], (2.0 * special.j1(x) / x) ** 2, atol=2e-3)
    # encircled energy within the first zero: 1 - J0^2(x) - J1^2(x), x the first zero of J1
    zero = float(special.jn_zeros(1, 1)[0])
    radius = zero * SCALE / (math.pi * D_MM)
    uu, vv = np.meshgrid(u, v)
    inside = float(map_[uu * uu + vv * vv <= radius * radius].sum()) * du * du
    assert inside / 1e-3 == pytest.approx(1.0 - special.j0(zero) ** 2 - special.j1(zero) ** 2, rel=0.01)


def test_parseval_holds_from_the_pupil_to_the_focal_window():
    doc = _airy()
    power = doc["power"]
    assert power["parseval_relative_residual"] < 1e-12
    assert power["on_pupil_W"] == pytest.approx(1e-3, rel=1e-12)
    assert power["window_W"] == pytest.approx(1e-3, rel=1e-12)
    assert power["higher_orders_W"] == pytest.approx(0.0, abs=1e-15)
    assert doc["grid_power_coherent_W"] + doc["power_outside_grid_W"] == pytest.approx(1e-3, rel=1e-9)
    assert doc["quantities"]["focal_irradiance"]["status"] == "MODELED"
    assert doc["quantities"]["device_efficiency"]["status"] == "UNAVAILABLE" and doc["quantities"]["polarization"]["status"] == "UNAVAILABLE"


def test_field_phase_conventions():
    """U_f(0, 0) = -i sqrt(P A) / (lambda f) for an even or odd grid, a constant mask phase adds, and a shifted pupil gives the phase ramp of the Fourier shift theorem."""

    for pixels in (64, 65):
        doc = pupil_field(request(pixels=pixels, pitch=0.1, fft=256, output={"half_extent_mm": [0.2, 0.2], "include_field": True}))
        grid = doc["grid"]
        i, j = grid["u_mm"].index(0.0), grid["v_mm"].index(0.0)
        field = complex(doc["field_sqrt_W_per_mm2"]["re"][j][i], doc["field_sqrt_W_per_mm2"]["im"][j][i])
        area = (pixels * 0.1) ** 2
        assert abs(field) == pytest.approx(math.sqrt(1e-3 * area) / SCALE, rel=1e-9)
        assert math.atan2(field.imag, field.real) == pytest.approx(-math.pi / 2.0, abs=1e-9)
        # the grid is centred: the field is purely imaginary all along the axis (a real, even pupil)
        row = np.array(doc["field_sqrt_W_per_mm2"]["re"][j])
        assert np.abs(row).max() < 1e-9 * abs(field)
    # a ramp about the grid's centre: the transform of exp(i a x) over a symmetric grid is real, so the field's real part is 0, not rounding noise blown up to its own scale
    ramp = pupil_field(request(pixels=64, pitch=0.1, fft=256, mask={"kind": "BLAZED_RAMP", "sin_theta": [2e-3, 0.0]}, output={"half_extent_mm": [0.6, 0.1], "include_field": True}))
    assert np.abs(ramp["field_sqrt_W_per_mm2"]["re"]).max() == 0.0 and np.abs(ramp["field_sqrt_W_per_mm2"]["im"]).max() > 0.0
    constant = [[0.7] * 64 for _ in range(64)]
    doc = pupil_field(request(pixels=64, pitch=0.1, fft=256, mask={"kind": "SAMPLED", "phase_rad": constant, "source": SOURCE},
                              output={"half_extent_mm": [0.2, 0.2], "include_field": True}))
    i = doc["grid"]["u_mm"].index(0.0)
    field = complex(doc["field_sqrt_W_per_mm2"]["re"][i][i], doc["field_sqrt_W_per_mm2"]["im"][i][i])
    assert math.atan2(field.imag, field.real) == pytest.approx(-math.pi / 2.0 + 0.7, abs=1e-9)
    # a Gaussian beam centred at x0: phase -pi/2 - 2 pi x0 u / (lambda f) on the focal plane
    x0 = 0.5
    doc = pupil_field(request(pixels=256, pitch=0.05, fft=1024,
                              illumination={"kind": "GAUSSIAN", "power_w": 1e-3, "radius_mm": [0.8, 0.8], "center_mm": [x0, 0.0]},
                              output={"half_extent_mm": [0.05, 0.05], "include_field": True}))
    grid = doc["grid"]
    j = grid["v_mm"].index(0.0)
    for i, u in list(enumerate(grid["u_mm"]))[::7]:
        field = complex(doc["field_sqrt_W_per_mm2"]["re"][j][i], doc["field_sqrt_W_per_mm2"]["im"][j][i])
        want = -math.pi / 2.0 - 2.0 * math.pi * x0 * u / SCALE
        assert np.angle(field * np.exp(-1j * want)) == pytest.approx(0.0, abs=1e-9)
    assert doc["conventions"]["focal_scale"].startswith("u_k = k lambda f / (Nx dx)")


# ------------------------------------------------------------------------------------------------- a blazed ramp


Q = 8  # pixels per period
PITCH = 0.0125
RAMP_SINE = LAMBDA_MM / (Q * PITCH)


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_blazed_ramp_steers_to_the_stated_spot_with_the_pixel_step_efficiency(sign):
    """1024 pixels, fully filled: the whole-period ramp puts one transform bin at u = f sin(theta) = lambda f / (q p), of relative peak sinc^2(1 / q)."""

    expected = sign * F_MM * RAMP_SINE
    doc = pupil_field(request(pixels=1024, pitch=PITCH, fill=1.0, fft=2048, mask={"kind": "BLAZED_RAMP", "sin_theta": [sign * RAMP_SINE, 0.0]},
                              output={"half_extent_mm": [2.5, 0.2]}))
    u = np.array(doc["grid"]["u_mm"])
    v = np.array(doc["grid"]["v_mm"])
    map_ = np.array(doc["coherent_W_per_mm2"])
    j, i = np.unravel_index(int(np.argmax(map_)), map_.shape)
    assert u[i] == pytest.approx(expected, rel=1e-9) and v[j] == 0.0  # +x tilt -> +u
    assert doc["mask"]["steered_uv_mm"][0] == pytest.approx(expected, rel=1e-12)
    area = (1024 * PITCH) ** 2
    sinc2 = np.sinc(1.0 / Q) ** 2
    assert map_[j, i] == pytest.approx(sinc2 * 1e-3 * area / (SCALE * SCALE), rel=1e-9)
    # the ledger: sinc^2(1 / q) in the window, the rest in the replicas of the pixel grid
    power = doc["power"]
    assert power["window_W"] / power["delivered_W"] == pytest.approx(sinc2, rel=1e-3)
    assert power["higher_orders_W"] == pytest.approx(1e-3 * (1.0 - sinc2), rel=2e-2)
    assert power["parseval_relative_residual"] < 1e-12


def test_the_replicas_of_the_pixel_grid_hold_the_higher_orders_of_the_ledger():
    """Summed directly from the closed form: Sum_m sinc^2(1 / q + m) = 1, each order at u = lambda f (1 / q + m) / p (Parseval on the continuous field)."""

    orders = range(-2000, 2001)
    total = sum(np.sinc(1.0 / Q + m) ** 2 for m in orders)
    assert total == pytest.approx(1.0, abs=2e-3)
    doc = pupil_field(request(pixels=1024, pitch=PITCH, fill=1.0, fft=2048, mask={"kind": "BLAZED_RAMP", "sin_theta": [RAMP_SINE, 0.0]},
                              output={"half_extent_mm": [0.5, 0.5]}))
    higher = sum(np.sinc(1.0 / Q + m) ** 2 for m in orders if m != 0)
    assert doc["power"]["higher_orders_W"] / 1e-3 == pytest.approx(higher, rel=2e-2)


def test_a_fill_factor_gives_the_envelope_and_the_dead_area_of_the_ledger():
    """An active width a = p sqrt(fill): window power fill sinc^2(sqrt(fill) / q) of the active power, and (1 - fill) P on the gaps."""

    fill = 0.81
    doc = pupil_field(request(pixels=1024, pitch=PITCH, fill=fill, fft=2048, mask={"kind": "BLAZED_RAMP", "sin_theta": [RAMP_SINE, 0.0]},
                              output={"half_extent_mm": [0.5, 0.5]}))
    power = doc["power"]
    assert power["active_W"] == pytest.approx(fill * 1e-3, rel=1e-12) and power["dead_area_W"] == pytest.approx((1.0 - fill) * 1e-3, rel=1e-12)
    assert power["window_W"] == pytest.approx(power["active_W"] * fill * np.sinc(math.sqrt(fill) / Q) ** 2, rel=1e-3)
    assert power["window_W"] + power["higher_orders_W"] == pytest.approx(power["delivered_W"], rel=1e-12)


def test_a_stated_device_efficiency_scales_the_field_and_keeps_the_rest_in_the_ledger():
    plain = pupil_field(request(pixels=64, pitch=0.1, fft=256, output={"half_extent_mm": [0.3, 0.3]}))
    doc = pupil_field(request(pixels=64, pitch=0.1, fft=256, output={"half_extent_mm": [0.3, 0.3]}, efficiency=0.8))
    assert doc["quantities"]["device_efficiency"] == {"status": "EXTERNALLY_CHARACTERIZED", "unit": None, "value": 0.8, "source": SOURCE}
    assert at(doc, 0.0, 0.0) == pytest.approx(0.8 * at(plain, 0.0, 0.0), rel=1e-9)
    assert doc["power"]["not_delivered_W"] == pytest.approx(0.2e-3, rel=1e-12) and doc["power"]["window_W"] == pytest.approx(0.8e-3, rel=1e-12)


# ------------------------------------------------------------------------------------------------- a binary grating


def test_binary_grating_first_order_holds_forty_point_five_percent():
    """A 0-pi square wave of 16 pixels per period, 256 pixels (16 whole periods), fully filled: amplitude 2 / (pi m) in each odd order m, none in the even ones."""

    period = 16 * PITCH
    doc = pupil_field(request(pixels=256, pitch=PITCH, fill=1.0, fft=512, mask={"kind": "BINARY_GRATING", "period_mm": period, "axis": "X", "offset_mm": -128 * PITCH},
                              output={"half_extent_mm": [3.0, 0.1]}))
    assert doc["mask"]["samples_per_period"] == pytest.approx(16.0)
    uniform_peak = 1e-3 * (256 * PITCH) ** 2 / (SCALE * SCALE)
    first = SCALE / period
    relative = lambda u: at(doc, u, 0.0) / uniform_peak  # noqa: E731
    assert relative(first) == pytest.approx((2.0 / math.pi) ** 2, rel=1e-6)
    assert relative(-first) == pytest.approx((2.0 / math.pi) ** 2, rel=1e-6)
    assert (2.0 / math.pi) ** 2 == pytest.approx(0.405, abs=5e-4)
    assert relative(0.0) < 1e-9 and relative(2.0 * first) < 1e-9
    assert relative(3.0 * first) == pytest.approx((2.0 / (3.0 * math.pi)) ** 2, rel=1e-6)


def test_binary_grating_along_y_lies_along_v():
    period = 16 * PITCH
    doc = pupil_field(request(pixels=256, pitch=PITCH, fill=1.0, fft=512, mask={"kind": "BINARY_GRATING", "period_mm": period, "axis": "Y", "offset_mm": -128 * PITCH},
                              output={"half_extent_mm": [0.1, 1.0]}))
    uniform_peak = 1e-3 * (256 * PITCH) ** 2 / (SCALE * SCALE)
    assert at(doc, 0.0, SCALE / period) / uniform_peak == pytest.approx((2.0 / math.pi) ** 2, rel=1e-6)


# ------------------------------------------------------------------------------------------------- a spot array


def _array_spots(n: int, pitch_mm: float) -> list[dict]:
    return [{"u_mm": (i - (n - 1) / 2.0) * pitch_mm, "v_mm": (j - (n - 1) / 2.0) * pitch_mm} for j in range(n) for i in range(n)]


def test_gerchberg_saxton_mask_generates_a_uniform_array():
    """5 x 5 spots at 0.3 mm from an 8 mm pupil of 12.5 um pixels: the stated uniformity and efficiency, and the array where it was asked. Bounds set before the first run:
    power uniformity above 0.95, at least 70 % of the power in the Airy radii of the spots (a perfect spot holds 83.8 % there, and such an array diffracts about 90 % into its spots)."""

    spots = _array_spots(5, 0.3)
    doc = pupil_field(request(pixels=640, pitch=PITCH, fill=1.0, fft=2048, aperture=CIRCLE, mask={"kind": "SPOT_ARRAY", "spots": spots, "iterations": 30}))
    assert doc["status"] == "OK" and doc["mask"]["spot_count"] == 25
    array = doc["array"]
    assert array["spot_count"] == 25 and array["uniformity_power"] > 0.95 and array["efficiency"] > 0.7
    assert array["uniformity_peak"] > 0.9
    radius = 1.22 * SCALE / D_MM
    assert doc["spots"][0]["radius_mm"] == pytest.approx(radius, rel=2e-3)
    assert all(abs(row["u_mm"]) <= 0.6 + 1e-9 and row["power_W"] > 0.0 for row in doc["spots"])
    # every spot's peak lies within a focal step or two of its target on the map
    map_ = np.array(doc["coherent_W_per_mm2"])
    u, v = np.array(doc["grid"]["u_mm"]), np.array(doc["grid"]["v_mm"])
    for row in doc["spots"]:
        box = (np.abs(u - row["u_mm"]) < 0.1)
        rows = (np.abs(v - row["v_mm"]) < 0.1)
        patch = map_[np.ix_(rows, box)]
        j, i = np.unravel_index(int(np.argmax(patch)), patch.shape)
        assert abs(u[box][i] - row["u_mm"]) <= 2.0 * doc["sampling"]["grid_step_mm"][0]
        assert abs(v[rows][j] - row["v_mm"]) <= 2.0 * doc["sampling"]["grid_step_mm"][1]
    # the same request gives the same mask
    assert pupil_field(request(pixels=640, pitch=PITCH, fill=1.0, fft=2048, aperture=CIRCLE, mask={"kind": "SPOT_ARRAY", "spots": spots, "iterations": 30}))["array"] == array


@pytest.mark.parametrize("spots, pixels, pitch, diameter", [
    (_array_spots(2, 0.6), 64, 0.1, 6.0),
    (_array_spots(5, 0.3), 320, 0.025, 8.0),
    ([{"u_mm": (i - 7.5) * 0.3, "v_mm": 0.0} for i in range(16)], 320, 0.025, 8.0),
])
def test_the_generator_does_not_amplify_rounding(spots, pixels, pitch, diameter):
    """A shift of the targets by 1e-13 mm, the size of a rounding error, moves the mask by no more than what such a shift gives directly (2 pi x du / (lambda f) ~ 1e-11 rad) times a modest factor:
    it does not grow to a different mask, in ten iterations or thirty. This is what makes the mask the same in Pyodide and native Python."""

    def mask(shift: float, iterations: int) -> np.ndarray:
        moved = [{**spot, "u_mm": spot["u_mm"] + shift} for spot in spots]
        doc = pupil_field(request(pixels=pixels, pitch=pitch, aperture={"shape": "CIRCLE", "diameter_mm": diameter}, mask={"kind": "SPOT_ARRAY", "spots": moved, "iterations": iterations},
                                  output={"half_extent_mm": [0.1, 0.1], "include_mask": True}))
        return np.array(doc["mask"]["phase_rad"])

    x = (np.arange(pixels) - 0.5 * (pixels - 1)) * pitch
    xx, yy = np.meshgrid(x, x)
    inside = xx * xx + yy * yy <= diameter * diameter / 4.0
    for iterations in (10, 30):
        difference = np.angle(np.exp(1j * (mask(0.0, iterations) - mask(1e-13, iterations))))[inside]
        assert np.abs(difference - difference.mean()).max() < 1e-8, iterations


def test_a_small_array_is_uniform_after_few_iterations():
    doc = pupil_field(request(pixels=64, pitch=0.1, fft=256, aperture={"shape": "CIRCLE", "diameter_mm": 6.0}, mask={"kind": "SPOT_ARRAY", "spots": _array_spots(2, 0.6), "iterations": 10}))
    assert doc["array"]["uniformity_power"] > 0.95


def test_weights_set_the_powers_of_a_spot_array():
    spots = [{"u_mm": -0.4, "v_mm": 0.0, "weight": 1.0}, {"u_mm": 0.4, "v_mm": 0.0, "weight": 4.0}]
    doc = pupil_field(request(pixels=320, pitch=PITCH * 2, aperture={"shape": "CIRCLE", "diameter_mm": 7.0}, fft=1024, mask={"kind": "SPOT_ARRAY", "spots": spots, "iterations": 40}))
    (a, b) = doc["spots"]
    assert b["power_W"] / a["power_W"] == pytest.approx(4.0, rel=0.02)


def test_a_generated_mask_written_back_gives_the_same_focal_field():
    spots = _array_spots(2, 0.4)
    generator = request(pixels=64, pitch=0.1, fft=256, mask={"kind": "SPOT_ARRAY", "spots": spots}, output={"half_extent_mm": [1.0, 1.0], "include_mask": True})
    doc = pupil_field(generator)
    written = {**generator, "mask": {"kind": "SAMPLED", "phase_rad": doc["mask"]["phase_rad"], "source": SOURCE}}
    again = pupil_field(written)
    np.testing.assert_allclose(again["coherent_W_per_mm2"], doc["coherent_W_per_mm2"], rtol=1e-9, atol=1e-12 * np.max(doc["coherent_W_per_mm2"]))
    assert again["quantities"]["mask_phase"]["status"] == "EXTERNALLY_CHARACTERIZED" and doc["quantities"]["mask_phase"]["status"] == "MODELED"


# ------------------------------------------------------------------------------------------------- illumination


def test_a_gaussian_pupil_focuses_to_the_gaussian_spot():
    """w_f = lambda f / (pi w), peak 2 P / (pi w_f^2); the opening is far outside the beam (4.3 w of grid half width) and clips nothing."""

    w = 1.5
    doc = pupil_field(request(pixels=256, pitch=0.05, fft=2048, illumination={"kind": "GAUSSIAN", "power_w": 2e-3, "radius_mm": [w, w]}, output={"half_extent_mm": [0.3, 0.3]}))
    w_f = SCALE / (math.pi * w)
    assert doc["power"]["clipped_W"] == pytest.approx(0.0, abs=2e-3 * 2e-3)
    assert at(doc, 0.0, 0.0) == pytest.approx(2.0 * 2e-3 / (math.pi * w_f * w_f), rel=1e-3)
    axis = np.array(doc["grid"]["u_mm"])
    point = float(axis[int(np.argmin(np.abs(axis - w_f)))])
    assert at(doc, point, 0.0) / at(doc, 0.0, 0.0) == pytest.approx(math.exp(-2.0 * point * point / (w_f * w_f)), rel=2e-3)


def test_a_clipping_opening_is_in_the_ledger():
    """A Gaussian of 1/e^2 radius w through an opening of radius a passes 1 - exp(-2 a^2 / w^2)."""

    w, a = 2.0, 2.0
    doc = pupil_field(request(pixels=256, pitch=0.05, fft=512, aperture={"shape": "CIRCLE", "diameter_mm": 2.0 * a}, illumination={"kind": "GAUSSIAN", "power_w": 1e-3, "radius_mm": [w, w]},
                              output={"half_extent_mm": [0.3, 0.3]}))
    passed = 1.0 - math.exp(-2.0 * a * a / (w * w))
    assert doc["power"]["on_pupil_W"] == pytest.approx(1e-3 * passed, rel=2e-3)
    assert doc["power"]["clipped_W"] == pytest.approx(1e-3 * (1.0 - passed), rel=2e-2)
    assert doc["power"]["window_W"] == pytest.approx(doc["power"]["on_pupil_W"], rel=1e-12)


def _tones(count: int) -> list[dict]:
    return [{"label": f"tone {i}", "power_w": 1e-3, "radius_mm": [1.5, 1.5], "direction_cosines": [i * 1e-3, 0.0]} for i in range(count)]


def test_independent_beams_add_in_irradiance_each_at_f_times_its_direction():
    beams = _tones(4)
    doc = pupil_field(request(pixels=256, pitch=0.05, fft=1024, illumination={"kind": "BEAMS", "beams": beams, "composition": "INCOHERENT"}))
    assert doc["composition"] == "INCOHERENT_SUM" and doc["coherent_W_per_mm2"] is None and doc["coherent_reason"] == "BEAMS_INCOHERENT"
    assert doc["quantities"]["focal_phase"]["reason"] == "BEAMS_INCOHERENT"
    w_f = SCALE / (math.pi * 1.5)
    for i, row in enumerate(doc["spots"]):
        assert row["u_mm"] == pytest.approx(F_MM * i * 1e-3, rel=1e-12)
        assert row["power_W"] == pytest.approx(1e-3 * (1.0 - math.exp(-2.0 * 1.5 * 1.5)), rel=5e-3)  # the circle of 1.5 w_f
        assert row["peak_W_per_mm2"] == pytest.approx(2e-3 / (math.pi * w_f * w_f), rel=5e-3)
    assert doc["array"]["efficiency"] == pytest.approx(1.0, abs=0.02) and doc["array"]["uniformity_power"] > 0.999
    assert doc["power"]["window_W"] == pytest.approx(4e-3, rel=1e-3)


def test_coherent_beams_add_their_fields():
    """Two equal beams in phase: four times the peak of one; in antiphase: none."""

    one = {"label": "a", "power_w": 1e-3, "radius_mm": [1.5, 1.5], "phase_rad": 0.0}
    kwargs = {"pixels": 128, "pitch": 0.1, "fft": 512, "output": {"half_extent_mm": [0.2, 0.2]}}
    single = pupil_field(request(illumination={"kind": "BEAMS", "beams": [one], "composition": "COHERENT"}, **kwargs))
    double = pupil_field(request(illumination={"kind": "BEAMS", "beams": [one, {**one, "label": "b"}], "composition": "COHERENT"}, **kwargs))
    dark = pupil_field(request(illumination={"kind": "BEAMS", "beams": [one, {**one, "label": "b", "phase_rad": math.pi}], "composition": "COHERENT"}, **kwargs))
    assert at(double, 0.0, 0.0) == pytest.approx(4.0 * at(single, 0.0, 0.0), rel=1e-9)
    assert at(dark, 0.0, 0.0) < 1e-12 * at(single, 0.0, 0.0)
    # one beam without a stated phase is the same field up to a constant phase: computed, with the phase reference said
    no_phase = pupil_field(request(illumination={"kind": "BEAMS", "beams": [{**one, "phase_rad": None}], "composition": "COHERENT"}, **kwargs))
    assert no_phase["status"] == "OK" and no_phase["phase_reference"] == "UNDETERMINED_CONSTANT" and "phase_reference" not in single
    assert no_phase["coherent_W_per_mm2"] == single["coherent_W_per_mm2"] and no_phase["power"] == single["power"]
    # several beams add with their phases: one without a phase leaves the pattern undetermined, and is refused
    undetermined = pupil_field(request(illumination={"kind": "BEAMS", "beams": [one, {**one, "label": "b", "phase_rad": None}], "composition": "COHERENT"}, **kwargs))
    assert undetermined["status"] == "UNAVAILABLE" and undetermined["reason"] == "PHASE_NOT_DEFINED" and undetermined["path_ids"] == ["b"]
    assert "INCOHERENT" in undetermined["message"]


def test_a_beam_or_a_target_beyond_the_window_is_refused():
    """The window of 0.1 mm pixels at lambda f = 0.1704 mm^2 reaches +-0.852 mm: a spot beyond it would fold back into the map."""

    beyond = {"label": "far", "power_w": 1e-3, "radius_mm": [1.0, 1.0], "direction_cosines": [0.9 / F_MM, 0.0]}
    doc = pupil_field(request(pixels=64, pitch=0.1, illumination={"kind": "BEAMS", "beams": [beyond], "composition": "INCOHERENT"}))
    assert doc["reason"] == "GRID_INVALID" and "0.852" in doc["message"]
    assert pupil_field(request(pixels=64, pitch=0.1, mask={"kind": "SPOT_ARRAY", "spots": [{"u_mm": 0.0, "v_mm": 0.9}]}))["reason"] == "GRID_INVALID"
    assert pupil_field(request(pixels=64, pitch=0.1, mask={"kind": "BLAZED_RAMP", "sin_theta": [0.9 / F_MM, 0.0]}))["reason"] == "GRID_INVALID"
    # pixels: the ramp folds back as a pixelated device does, one window over, and the window holds the replica
    folded = pupil_field(request(pixels=64, pitch=0.1, fill=1.0, fft=256, mask={"kind": "BLAZED_RAMP", "sin_theta": [(0.9 - 1.704) / F_MM, 0.0]}, output={"half_extent_mm": [0.84, 0.1]}))
    wrapped = pupil_field(request(pixels=64, pitch=0.1, fill=1.0, fft=256, mask={"kind": "BLAZED_RAMP", "sin_theta": [0.9 / F_MM, 0.0]}, output={"half_extent_mm": [0.84, 0.1]}))
    assert folded["status"] == wrapped["status"] == "OK"
    np.testing.assert_allclose(wrapped["coherent_W_per_mm2"], folded["coherent_W_per_mm2"], rtol=1e-6, atol=1e-9 * np.max(folded["coherent_W_per_mm2"]))


# ------------------------------------------------------------------------------------------------- the validity limit


def test_above_the_validity_limit_the_focal_quantities_are_unavailable_and_the_pupil_stage_stays():
    """D = 50 mm at f = 100 mm: NA 0.25 above the limit 0.2."""

    doc = pupil_field(request(pixels=512, pitch=0.1, focal=100.0, aperture={"shape": "CIRCLE", "diameter_mm": 50.0}, power=1e-3, output={"half_extent_mm": [0.1, 0.1]}))
    assert doc["status"] == "PARTIAL" and doc["reason"] == "NA_ABOVE_SCALAR_LIMIT"
    assert doc["validity"]["pupil_na"] == pytest.approx(0.25, rel=2e-3) and doc["validity"]["in_domain"] is False and doc["validity"]["limit"] == 0.2
    assert doc["quantities"]["focal_irradiance"]["status"] == "UNAVAILABLE" and doc["quantities"]["focal_phase"]["status"] == "UNAVAILABLE"
    assert "coherent_W_per_mm2" not in doc and "grid" not in doc
    assert doc["quantities"]["pupil_power_W"]["status"] == "MODELED" and doc["power"]["on_pupil_W"] == pytest.approx(1e-3, rel=1e-12)
    # the detector reports what is upstream of it, and nothing takes the field's place
    camera = detector_response(doc, {"pixel_pitch_mm": [0.01, 0.01], "fill_factor": 1.0})
    assert camera["status"] == "UNAVAILABLE" and camera["upstream_reason"] == "NA_ABOVE_SCALAR_LIMIT"


def test_the_validity_limit_counts_the_output_grids_corner():
    """A pupil of NA 0.05 is in the domain until the grid's corner reaches a sine of 0.15."""

    near = pupil_field(request(pixels=256, pitch=0.5, focal=1000.0, aperture={"shape": "CIRCLE", "diameter_mm": 100.0}, output={"half_extent_mm": [1.0, 1.0]}))
    assert near["status"] == "OK" and near["validity"]["pupil_na"] == pytest.approx(0.05, rel=0.02)
    assert near["validity"]["marginal_ray_sine"] == pytest.approx(near["validity"]["pupil_na"] + near["validity"]["field_sine"], rel=1e-12)
    # the same pupil, a grid far off the axis: 200 mm at f = 1000 mm is a sine of 0.2
    far = pupil_field(request(pixels=64, pitch=0.5, focal=1000.0, output={"center_uv_mm": [0.0, 0.0], "half_extent_mm": [0.3, 0.3]}))
    assert far["status"] == "OK"
    wide = pupil_field(request(pixels=64, pitch=0.002, focal=50.0, output={"half_extent_mm": [10.0, 10.0]}))
    assert wide["status"] == "PARTIAL" and wide["validity"]["field_sine"] > 0.2


# ------------------------------------------------------------------------------------------------- masks and refusals


def _npz(**arrays) -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return buffer.getvalue()


def test_a_mask_file_is_read_and_its_hash_checked():
    ramp = 2.0 * math.pi * RAMP_SINE * ((np.arange(64) - 31.5) * PITCH)[None, :] / LAMBDA_MM
    phase = np.broadcast_to(ramp, (64, 64)).copy()
    data = _npz(phase_rad=phase)
    digest = hashlib.sha256(data).hexdigest()
    mask = {"kind": "FILE", "file_name": "mask.npz", "sha256": digest, "source": SOURCE}
    kwargs = {"pixels": 64, "pitch": PITCH, "fill": 1.0, "fft": 256, "output": {"half_extent_mm": [3.0, 0.2]}}
    doc = pupil_field(request(mask=mask, **kwargs), {"mask.npz": data})
    parametric = pupil_field(request(mask={"kind": "BLAZED_RAMP", "sin_theta": [RAMP_SINE, 0.0]}, **kwargs))
    assert doc["coherent_W_per_mm2"] == parametric["coherent_W_per_mm2"]
    assert doc["mask"]["sha256"] == digest and doc["quantities"]["mask_phase"]["status"] == "EXTERNALLY_CHARACTERIZED"
    assert pupil_field(request(mask=mask, **kwargs))["reason"] == "MASK_FILE_MISSING"
    assert pupil_field(request(mask={**mask, "sha256": "0" * 64}, **kwargs), {"mask.npz": data})["reason"] == "MASK_FILE_HASH_MISMATCH"
    bad_shape = _npz(phase_rad=np.zeros((3, 3)))
    assert pupil_field(request(mask={**mask, "sha256": hashlib.sha256(bad_shape).hexdigest()}, **kwargs), {"mask.npz": bad_shape})["reason"] == "MASK_FILE_INVALID"
    junk = b"not a mask"
    assert pupil_field(request(mask={**mask, "sha256": hashlib.sha256(junk).hexdigest()}, **kwargs), {"mask.npz": junk})["reason"] == "MASK_FILE_INVALID"
    # an .npy file holds the phase alone; an .npz may add an amplitude
    plain = io.BytesIO()
    np.save(plain, phase)
    npy = plain.getvalue()
    same = pupil_field(request(mask={**mask, "file_name": "mask.npy", "sha256": hashlib.sha256(npy).hexdigest()}, **kwargs), {"mask.npy": npy})
    assert same["coherent_W_per_mm2"] == parametric["coherent_W_per_mm2"]
    half = _npz(phase_rad=phase, amplitude=np.full((64, 64), 0.5))
    dim = pupil_field(request(mask={**mask, "sha256": hashlib.sha256(half).hexdigest()}, **kwargs), {"mask.npz": half})
    assert dim["power"]["after_mask_W"] == pytest.approx(0.25e-3, rel=1e-12) and dim["power"]["absorbed_by_mask_W"] == pytest.approx(0.75e-3, rel=1e-12)


def test_an_inline_amplitude_mask_is_in_the_ledger():
    phase = [[0.0] * 8 for _ in range(8)]
    amplitude = [[1.0 if i < 4 else 0.0 for i in range(8)] for _ in range(8)]
    doc = pupil_field(request(pixels=8, pitch=1.0, fft=32, focal=1e5, mask={"kind": "SAMPLED", "phase_rad": phase, "amplitude": amplitude, "source": SOURCE}, output={"half_extent_mm": [50.0, 50.0]}))
    assert doc["power"]["after_mask_W"] == pytest.approx(0.5e-3, rel=1e-12) and doc["power"]["window_W"] == pytest.approx(0.5e-3, rel=1e-12)


@pytest.mark.parametrize("change", [
    {"wavelength_nm": 0.0}, {"focal_length_mm": -1.0},
    {"sampling": {"pixel_count": [0, 8], "pitch_mm": [0.1, 0.1]}},
    {"sampling": {"pixel_count": [8, 8], "pitch_mm": [0.1, 0.1], "fft_size": [4, 4]}},
    {"sampling": {"pixel_count": [8, 8], "pitch_mm": [0.1, -0.1]}},
    {"mask": {"kind": "SAMPLED", "phase_rad": [[0.0]], "source": SOURCE}},
    {"mask": {"kind": "SAMPLED", "phase_rad": [[0.0] * 8] * 8}},
    {"mask": {"kind": "FILE", "file_name": "../mask.npz", "sha256": "0" * 64, "source": SOURCE}},
    {"mask": {"kind": "NOPE"}},
    {"illumination": {"kind": "BEAMS", "beams": [], "composition": "COHERENT"}},
    {"illumination": {"kind": "BEAMS", "beams": _tones(1), "composition": "INCOHERENT"}, "mask": {"kind": "SPOT_ARRAY", "spots": [{"u_mm": 0.0, "v_mm": 0.0}]}},
    {"device_efficiency": {"value": 0.5}},
])
def test_an_invalid_request_is_refused_with_its_problems(change):
    doc = {**request(pixels=8, pitch=0.1), **change}
    with pytest.raises(PupilFieldError) as exc:
        pupil_field(doc)
    assert exc.value.code == "PUPIL_FIELD_INVALID" and exc.value.message
    reply = json_api.pupil_field(doc)
    assert reply["ok"] is False and reply["error"]["code"] == "PUPIL_FIELD_INVALID"


def test_a_transform_that_does_not_sample_the_irradiance_is_refused():
    doc = pupil_field(request(pixels=64, pitch=0.1, fft=64))
    assert doc["status"] == "UNAVAILABLE" and doc["reason"] == "GRID_INVALID" and "twice" in doc["message"]
    # the same size is enough when the light fills only part of the grid
    small = pupil_field(request(pixels=64, pitch=0.1, fft=64, aperture={"shape": "CIRCLE", "diameter_mm": 3.0}))
    assert small["status"] == "OK"
    assert pupil_field(request(pixels=64, pitch=0.1, aperture={"shape": "CIRCLE", "diameter_mm": 1e-6}))["reason"] == "NO_LIGHT"
    assert pupil_field(request(pixels=64, pitch=0.1, fft=128, output={"half_extent_mm": [1e-6, 1e-6]}))["reason"] == "GRID_INVALID"
    too_many = pupil_field(request(pixels=2048, pitch=1e-3, focal=1e4, aperture={"shape": "CIRCLE", "diameter_mm": 0.5}, output={"half_extent_mm": [1e6, 1e6]}))
    assert too_many["reason"] == "GRID_INVALID"


# ------------------------------------------------------------------------------------------------- a run's beams


def _pupil_plane_run(m2: float = 1.0) -> dict:
    """A beam of a large waist 100 mm before the pupil plane (Rayleigh range 8 m): the run delivers it nearly collimated."""

    laser = source(position=(-100.0, 0.0, 0.0), wavelength_nm=LAMBDA_NM, power_w=2e-3, waist_mm=1.5)
    laser["source_model"]["m2"] = m2
    return project([laser], [plane("PUPIL", (0.0, 0.0, 0.0), radius_mm=20.0)])


def test_a_runs_handoff_lights_the_pupil_with_the_beam_it_delivered():
    """Power, wavelength and radius come from the run; the focus is the closed form of that beam."""

    results = load_run(run(_pupil_plane_run()))
    beam = results.handoff("PUPIL")["beams"][0]
    wanted = {"wavelength_nm": LAMBDA_NM, "focal_length_mm": F_MM, "sampling": {"pixel_count": [256, 256], "pitch_mm": [0.05, 0.05], "fft_size": [2048, 2048]},
              "illumination": {"kind": "HANDOFF_BEAMS", "composition": "INCOHERENT"}, "output": {"half_extent_mm": [0.3, 0.3]}}
    doc = results.pupil_field("PUPIL", wanted)
    assert doc["status"] == "OK" and doc["composition"] == "INCOHERENT_SUM"
    assert doc["power"]["incident_W"] == pytest.approx(beam["power_W"], rel=1e-12) and beam["power_W"] == pytest.approx(2e-3, rel=1e-9)
    w = beam["gaussian"]["beam_radius_x_mm"]
    w_f = SCALE / (math.pi * w)
    assert w == pytest.approx(1.5, rel=1e-3)
    assert at(doc, 0.0, 0.0, "incoherent_W_per_mm2") == pytest.approx(2.0 * 2e-3 / (math.pi * w_f * w_f), rel=2e-3)
    assert doc["spots"][0]["label"] == beam["path_id"]
    # no beam of another wavelength, and no handoff at all: refused, nothing assumed
    assert results.pupil_field("PUPIL", {**wanted, "wavelength_nm": 532.0})["reason"] == "NO_BEAMS"
    assert pupil_field(wanted)["reason"] == "HANDOFF_REQUIRED"
    # the same through the JSON API, with the handoff result of run_results
    reply = json_api.pupil_field(wanted, None, results.handoff("PUPIL"))
    assert reply["ok"] is True and reply["array"] == doc["array"]


def test_a_beam_that_is_not_tem00_builds_no_field():
    results = load_run(run(_pupil_plane_run(m2=1.5)))
    wanted = {"wavelength_nm": LAMBDA_NM, "focal_length_mm": F_MM, "sampling": {"pixel_count": [64, 64], "pitch_mm": [0.2, 0.2]},
              "illumination": {"kind": "HANDOFF_BEAMS", "composition": "INCOHERENT"}}
    doc = results.pupil_field("PUPIL", wanted)
    assert doc["status"] == "UNAVAILABLE" and doc["reason"] == "BEAM_NOT_TEM00"


def _stopped_run(opening_mm: str) -> dict:
    """The beam of ``_pupil_plane_run`` (1.5 mm radius) through a round stop 50 mm before the pupil plane."""

    from declared_helpers import declared_component, stop_model, with_models

    laser = source(position=(-100.0, 0.0, 0.0), wavelength_nm=LAMBDA_NM, power_w=2e-3, waist_mm=1.5)
    doc = project([laser, declared_component("STOP", "aperture", "stop", (-50.0, 0.0, 0.0))], [plane("PUPIL", (0.0, 0.0, 0.0), radius_mm=20.0)])
    return with_models(doc, [stop_model("stop", edge_radius_mm="12.7", opening_radius_mm=opening_mm)])


def test_a_beam_cut_below_the_runs_warning_share_lights_the_pupil_within_its_stated_ripple():
    """A stop of 2.5 beam radii (3.75 mm) cuts 3.7e-6 of the beam: the handoff keeps TEM00 and the pupil field is built from it.
    Against the same beam through a stop that cuts nothing (10 mm), the focal peak differs by the cut share, far inside the
    ripple figure the handoff states for the beam. A stop of 1.2 beam radii builds no field, and the refusal states the share."""

    wanted = {"wavelength_nm": LAMBDA_NM, "focal_length_mm": F_MM, "sampling": {"pixel_count": [256, 256], "pitch_mm": [0.05, 0.05], "fft_size": [1024, 1024]},
              "illumination": {"kind": "HANDOFF_BEAMS", "composition": "INCOHERENT"}, "output": {"half_extent_mm": [0.3, 0.3]}}
    cut_results, open_results = load_run(run(_stopped_run("3.75"))), load_run(run(_stopped_run("10")))
    validity = cut_results.handoff("PUPIL")["beams"][0]["validity"]
    w_stop = 1.5 * math.sqrt(1.0 + (50.0 / (math.pi * 1.5**2 / LAMBDA_MM)) ** 2)  # the beam's radius at the stop, 50 mm behind its waist
    share = math.exp(-2.0 * 3.75**2 / w_stop**2)
    assert validity["beam_parameters"] == "TEM00" and validity["truncated_power_fraction"] == pytest.approx(share, rel=1e-6, abs=2e-12)
    assert open_results.handoff("PUPIL")["beams"][0]["validity"]["truncated_upstream"] == []
    cut, whole = cut_results.pupil_field("PUPIL", wanted), open_results.pupil_field("PUPIL", wanted)
    assert cut["status"] == "OK" and whole["status"] == "OK"
    ratio = at(cut, 0.0, 0.0, "incoherent_W_per_mm2") / at(whole, 0.0, 0.0, "incoherent_W_per_mm2")
    assert ratio == pytest.approx(1.0 - validity["truncated_power_fraction"], rel=1e-9)
    assert abs(ratio - 1.0) <= validity["field_intensity_ripple"]
    assert cut["power"]["incident_W"] == pytest.approx(2e-3 * (1.0 - share), rel=1e-6)
    refused = load_run(run(_stopped_run("1.8"))).pupil_field("PUPIL", wanted)
    assert refused["status"] == "UNAVAILABLE" and refused["reason"] == "BEAM_NOT_TEM00" and refused["beam_parameters"] == "TRUNCATED_GAUSSIAN"
    assert refused["truncated_power_fraction"] == pytest.approx(math.exp(-2.0 * 1.8**2 / w_stop**2), rel=1e-6)


def _elliptical_run(turn_deg: float) -> dict:
    """An elliptical beam (1.5 mm x 0.75 mm waist) on a pupil plane whose (u, v) axes are turned by ``turn_deg`` about the beam."""

    laser = source(position=(-100.0, 0.0, 0.0), wavelength_nm=LAMBDA_NM, power_w=2e-3, waist_mm=1.5)
    laser["source_model"]["waist_radius_y_mm"] = 0.75
    half = math.radians(turn_deg) / 2.0
    return project([laser], [plane("PUPIL", (0.0, 0.0, 0.0), quat_multiply(ALONG_X, turn_about_z(math.cos(half), math.sin(half))), radius_mm=20.0)])


@pytest.mark.parametrize("turn_deg", [0.0, 90.0, 30.0, -55.0])
def test_an_elliptical_beam_focuses_along_its_own_axes_whatever_the_planes_orientation(turn_deg):
    """An elliptical Gaussian beam's focal spot is the elliptical Gaussian of radii lambda f / (pi w_i) sqrt(1 + (pi w_i^2 /
    (lambda R_i))^2) along the beam's own axes (the Fourier transform of exp(-x^2 / w^2 + i k x^2 / (2 R)) per axis), peak
    2 P / (pi w_f1 w_f2). The plane's (u, v) axes are aligned with the beam's, swapped, and turned by 30 and -55 degrees (a turned
    ellipse was refused: BEAM_AXES_ROTATED): the map at grid points along both beam axes and off them is that closed form, so a
    mirrored or mis-turned ellipse (its radii differ by a factor of two) would miss it."""

    results = load_run(run(_elliptical_run(turn_deg)))
    beam = results.handoff("PUPIL")["beams"][0]
    g = beam["gaussian"]
    e1, e2 = g["axis_x_uvn"][:2], g["axis_y_uvn"][:2]
    angle = math.degrees(math.atan2(e1[1], e1[0]))
    assert min(abs((angle - sign * turn_deg + 180.0) % 360.0 - 180.0) for sign in (1.0, -1.0)) < 1e-6  # (the plane is turned as asked)
    wanted = {"wavelength_nm": LAMBDA_NM, "focal_length_mm": F_MM, "sampling": {"pixel_count": [256, 256], "pitch_mm": [0.05, 0.05], "fft_size": [2048, 2048]},
              "illumination": {"kind": "HANDOFF_BEAMS", "composition": "COHERENT"}, "output": {"half_extent_mm": [0.2, 0.2]}}
    doc = results.pupil_field("PUPIL", wanted)
    assert doc["status"] == "OK", doc.get("reason")
    assert doc["power"]["on_pupil_W"] == pytest.approx(beam["power_W"], rel=1e-6) and beam["power_W"] == pytest.approx(2e-3, rel=1e-9)
    radii = []
    for w, curvature in ((g["beam_radius_x_mm"], g["curvature_radius_x_mm"]), (g["beam_radius_y_mm"], g["curvature_radius_y_mm"])):
        defocus = 0.0 if curvature is None else math.pi * w * w / (LAMBDA_MM * curvature)
        radii.append(SCALE / (math.pi * w) * math.sqrt(1.0 + defocus * defocus))
    assert radii[1] == pytest.approx(2.0 * radii[0], rel=5e-3)  # (the narrow pupil axis makes the wide focal axis)
    peak = 2.0 * beam["power_W"] / (math.pi * radii[0] * radii[1])
    u_axis, v_axis = np.array(doc["grid"]["u_mm"]), np.array(doc["grid"]["v_mm"])
    samples = [(0.0, 0.0)] + [(s * a * e1[0] + t * b * e2[0], s * a * e1[1] + t * b * e2[1])
                              for s, t, a, b in ((1, 0, radii[0], 0.0), (-1, 0, 0.7 * radii[0], 0.0), (0, 1, 0.0, radii[1]), (0, -1, 0.0, 0.6 * radii[1]),
                                                 (1, 1, 0.6 * radii[0], 0.5 * radii[1]), (1, -1, 0.8 * radii[0], 0.4 * radii[1]))]
    for u, v in samples:
        i, j = int(np.argmin(np.abs(u_axis - u))), int(np.argmin(np.abs(v_axis - v)))
        s1 = u_axis[i] * e1[0] + v_axis[j] * e1[1]
        s2 = u_axis[i] * e2[0] + v_axis[j] * e2[1]
        expected = peak * math.exp(-2.0 * (s1 * s1 / radii[0] ** 2 + s2 * s2 / radii[1] ** 2))
        assert doc["coherent_W_per_mm2"][j][i] == pytest.approx(expected, rel=2e-3), (turn_deg, u, v)


def test_a_tilted_beam_of_any_orientation_keeps_its_power_and_its_own_profile_on_the_plane():
    """A beam that crosses the plane at 0.3 rad with its elliptical axes turned about its direction. Its irradiance on the plane
    is its own transverse Gaussian at each plane point's transverse coordinates (3-D projections on the beam's axes, written
    out here), times the cosine of the tilt; and the power through the plane, integrated here, is the beam's -- not P / cos,
    which a profile without the flux factor gives, and not a circle where the footprint is an ellipse."""

    from os_optics.results.pupil_field import HandoffBeam, _beam_field

    def about_z(a: float) -> np.ndarray:
        return np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])

    theta, power, w1, w2 = 0.3, 2e-3, 1.2, 0.6
    tilt = np.array([[math.cos(theta), 0.0, math.sin(theta)], [0.0, 1.0, 0.0], [-math.sin(theta), 0.0, math.cos(theta)]])
    e1, e2, d = (about_z(0.4) @ tilt @ about_z(0.7)).T  # the beam's two axes and its direction, as (u, v, n) components
    assert abs(d[2] - math.cos(theta)) < 1e-12 and abs(float(np.dot(e1, e2))) < 1e-12
    beam = HandoffBeam(label="b", power_w=power, beam_radius_mm=(w1, w2), radius_mm=(w1 / math.hypot(e1[0], e1[1]), w2 / math.hypot(e2[0], e2[1])),
                       center_mm=(0.3, -0.2), direction_cosines=(float(d[0]), float(d[1])), curvature_radius_mm=(None, None), phase_rad=None,
                       axes=((float(e1[0]), float(e1[1])), (float(e2[0]), float(e2[1]))))
    axis = (np.arange(1024) - 511.5) * 0.02
    xx, yy = np.meshgrid(axis, axis)
    irradiance = np.abs(_beam_field(beam, 2.0 * math.pi / LAMBDA_MM, xx, yy, False)) ** 2
    du, dv = xx - 0.3, yy + 0.2
    x1, x2 = e1[0] * du + e1[1] * dv, e2[0] * du + e2[1] * dv
    expected = 2.0 * power / (math.pi * w1 * w2) * math.cos(theta) * np.exp(-2.0 * (x1 ** 2 / w1 ** 2 + x2 ** 2 / w2 ** 2))
    assert np.allclose(irradiance, expected, rtol=1e-12, atol=1e-15 * float(expected.max()))  # (the far tail underflows)
    assert float(irradiance.sum()) * 0.02 * 0.02 == pytest.approx(power, rel=1e-9)
    # a stated beam (PupilBeamV1: radii along the plane's axes) is the same formula with the plane's own axes
    stated = pupil_field(request(pixels=64, pitch=0.2, illumination={"kind": "BEAMS", "composition": "INCOHERENT",
                                                                     "beams": [{"label": "s", "power_w": 1e-3, "radius_mm": [1.2, 0.6]}]}))
    assert stated["status"] == "OK" and stated["power"]["on_pupil_W"] == pytest.approx(1e-3, rel=1e-9)


def _two_beam_pupil_run() -> dict:
    """Two beams on the pupil plane, 2 mm apart: A is TEM00, B has M^2 = 1.5 (no field follows from it)."""

    a = source("A", position=(-100.0, 0.0, 0.0), wavelength_nm=LAMBDA_NM, power_w=2e-3, waist_mm=1.5)
    b = source("B", position=(-100.0, 2.0, 0.0), wavelength_nm=LAMBDA_NM, power_w=1e-3, waist_mm=1.0)
    b["source_model"]["m2"] = 1.5
    return project([a, b], [plane("PUPIL", (0.0, 0.0, 0.0), radius_mm=20.0)])


def test_an_incoherent_sum_leaves_out_the_beams_no_field_follows_from_and_names_them():
    """Beam B (M^2 = 1.5) gives no field. An incoherent sum built the field of nothing (BEAM_NOT_TEM00); it now builds beam A's
    -- the map of a request that names A alone, value for value -- and states B as excluded with its power, which a detector
    counts as light the map does not hold. A coherent sum still refuses: the field left out could cancel the others."""

    results = load_run(run(_two_beam_pupil_run()))
    beams = {beam["source_id"]: beam for beam in results.handoff("PUPIL")["beams"]}
    wanted = {"wavelength_nm": LAMBDA_NM, "focal_length_mm": F_MM, "sampling": {"pixel_count": [256, 256], "pitch_mm": [0.05, 0.05], "fft_size": [1024, 1024]},
              "illumination": {"kind": "HANDOFF_BEAMS", "composition": "INCOHERENT"}, "output": {"half_extent_mm": [0.3, 0.3]}}
    alone = results.pupil_field("PUPIL", {**wanted, "illumination": {**wanted["illumination"], "path_ids": [beams["A"]["path_id"]]}})
    doc = results.pupil_field("PUPIL", wanted)
    assert alone["status"] == "OK" and "excluded_beams" not in alone and "illumination_complete" not in alone
    assert doc["status"] == "OK" and doc["incoherent_W_per_mm2"] == alone["incoherent_W_per_mm2"] and doc["power"] == alone["power"]
    (row,) = doc["excluded_beams"]
    assert row["path_id"] == beams["B"]["path_id"] and row["reason"] == "BEAM_NOT_TEM00" and row["power_W"] == pytest.approx(1e-3, rel=1e-9)
    assert row["spot_uv_mm"] == pytest.approx([0.0, 0.0], abs=1e-9) and row["surfaces"] == []
    assert doc["illumination_complete"] is False and doc["map_complete"] is False and doc["interference_open"] is False
    assert doc["untraced_power_W"] == row["power_W"] and doc["untraced_fraction"] == pytest.approx(0.5, rel=1e-9)
    # the detector reads lower bounds, and no pixel the excluded beam's power could saturate reads as unsaturated
    detector = {"exposure_s": 1e-3, "pixel_pitch_mm": [0.05, 0.05], "fill_factor": 1.0, "pixel_count": [5, 5], "read_noise_e": 2.0, "dark_current_e_per_s": 1.0,
                "characterization": SOURCE, "quantum_efficiency": {"rows": [{"wavelength_nm": 400, "qe": 0.5}, {"wavelength_nm": 1000, "qe": 0.5}], "source": SOURCE}}
    reply = detector_response(doc, {**detector, "full_well_e": 1e15})
    powers = reply["quantities"]["pixel_power_W"]
    assert reply["map_complete"] is False and reply["excluded_beams"] == doc["excluded_beams"] and powers["status"] == "BOUNDED" and powers["bound"] == "LOWER"
    e_per_w = reply["quantities"]["signal_electrons"]["values"][2][2] / powers["values"][2][2]
    (j, i) = max(((j, i) for j in range(5) for i in range(5)), key=lambda ji: powers["values"][ji[0]][ji[1]])
    peak = powers["values"][j][i]
    assert powers["upper"][j][i] == pytest.approx(peak + row["power_W"], rel=1e-12)  # (the excluded beam whole on the pixel)
    # a full well between the brightest pixel's known power and that power plus the excluded beam's: not known there
    within_reach = detector_response(doc, {**detector, "full_well_e": (peak + 0.5 * row["power_W"]) * e_per_w})["quantities"]["saturated"]["values"]
    assert within_reach[j][i] is None
    beyond = detector_response(doc, {**detector, "full_well_e": (peak + 2.0 * row["power_W"]) * e_per_w})["quantities"]["saturated"]["values"]
    assert all(value is False for r in beyond for value in r)
    assert detector_response(alone, {**detector, "full_well_e": 1e15})["quantities"]["pixel_power_W"]["status"] == "MODELED"
    # coherent: refused, naming the beams and the composition that leaves them out
    refused = results.pupil_field("PUPIL", {**wanted, "illumination": {"kind": "HANDOFF_BEAMS", "composition": "COHERENT"}})
    assert refused["status"] == "UNAVAILABLE" and refused["reason"] == "BEAM_NOT_TEM00" and refused["path_id"] == beams["B"]["path_id"]
    assert "INCOHERENT" in refused["message"] and beams["B"]["path_id"] in refused["message"]
    # B alone: no usable beam, the refusal it was
    only = results.pupil_field("PUPIL", {**wanted, "illumination": {**wanted["illumination"], "path_ids": [beams["B"]["path_id"]]}})
    assert only["status"] == "UNAVAILABLE" and only["reason"] == "BEAM_NOT_TEM00" and "INCOHERENT" not in only["message"]


# ------------------------------------------------------------------------------------------------- discovery and the command line


def test_discovery_names_the_field_models_and_the_schemas():
    caps = json_api.capabilities()
    assert caps["native_field_models"] == ["detector.camera_response", "slm.pupil_mask"]
    assert caps["field_analyses"]["pupil_field"]["scalar_sine_limit"] == 0.2
    assert caps["field_analyses"]["quantity_statuses"] == ["MODELED", "EXTERNALLY_CHARACTERIZED", "BOUNDED", "UNAVAILABLE"]
    assert {"pupil-field-request", "detector-model"} <= set(caps["schema_kinds"])
    assert caps["json_api_methods"]["pupil_field"]["file_map"] == "mask_files"
    assert json_api.json_schema("pupil-field-request")["schema"]["title"] == "PupilFieldRequestV1"
    assert json_api.json_schema("detector-model")["schema"]["title"] == "DetectorModelV1"
    models = {m["model_id"]: m for m in json_api.native_components()["field_models"]}
    assert models["slm.pupil_mask"]["json_api"] == "pupil_field" and models["detector.camera_response"]["json_api"] == "detector_response"
    assert "CUSTOMER_SUPPLIED" in models["slm.pupil_mask"]["availability"]


def test_the_command_line_runs_the_stages(tmp_path, capsys):
    assert cli_main(["native", "list"]) == 0
    assert {m["model_id"] for m in json.loads(capsys.readouterr().out)["field_models"]} == {"slm.pupil_mask", "detector.camera_response"}
    assert cli_main(["native", "describe", "slm.pupil_mask"]) == 0
    assert json.loads(capsys.readouterr().out)["json_api"] == "pupil_field"
    assert cli_main(["schema", "pupil-field-request"]) == 0
    assert json.loads(capsys.readouterr().out)["title"] == "PupilFieldRequestV1"
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request(pixels=64, pitch=0.1, fft=256, output={"half_extent_mm": [0.3, 0.3]})), encoding="utf-8")
    assert cli_main(["results", "pupil-field", "--request", str(path)]) == 0
    field = capsys.readouterr().out
    assert json.loads(field)["status"] == "OK"
    field_path = tmp_path / "field.json"
    field_path.write_text(field, encoding="utf-8")
    detector = tmp_path / "detector.json"
    detector.write_text(json.dumps({"pixel_pitch_mm": [0.05, 0.05], "fill_factor": 1.0}), encoding="utf-8")
    assert cli_main(["results", "detector", "--field", str(field_path), "--detector", str(detector)]) == 0
    assert json.loads(capsys.readouterr().out)["composition"] == "COHERENT"
    # above the limit: JSON with the pupil stage and the focal quantities UNAVAILABLE with their reason: answered, exit code 0
    path.write_text(json.dumps(request(pixels=512, pitch=0.1, focal=100.0, aperture={"shape": "CIRCLE", "diameter_mm": 50.0}, output={"half_extent_mm": [0.1, 0.1]})), encoding="utf-8")
    assert cli_main(["results", "pupil-field", "--request", str(path)]) == 0
    above = json.loads(capsys.readouterr().out)
    assert above["status"] == "PARTIAL" and above["quantities"]["focal_irradiance"]["status"] == "UNAVAILABLE"
    assert above["quantities"]["focal_irradiance"]["reason"] == "NA_ABOVE_SCALAR_LIMIT" and above["quantities"]["pupil_power_W"]["status"] == "MODELED"
    # no field at all: UNAVAILABLE, exit code 2
    path.write_text(json.dumps(request(pixels=64, pitch=0.1, aperture={"shape": "CIRCLE", "diameter_mm": 1e-6})), encoding="utf-8")
    assert cli_main(["results", "pupil-field", "--request", str(path)]) == 2
    assert json.loads(capsys.readouterr().out)["reason"] == "NO_LIGHT"
    path.write_text(json.dumps({"wavelength_nm": 852.0}), encoding="utf-8")
    assert cli_main(["results", "pupil-field", "--request", str(path)]) == 64
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "PUPIL_FIELD_INVALID"
    assert cli_main(["results", "pupil-field", "--request", str(path), "--monitor", "M"]) == 64
    capsys.readouterr()
    # a FILE mask named on the command line, and a run's beams
    phase = np.zeros((64, 64))
    data = _npz(phase_rad=phase)
    (tmp_path / "mask.npz").write_bytes(data)
    masked = request(pixels=64, pitch=0.1, fft=256, mask={"kind": "FILE", "file_name": "mask.npz", "sha256": hashlib.sha256(data).hexdigest(), "source": SOURCE},
                     output={"half_extent_mm": [0.3, 0.3]})
    path.write_text(json.dumps(masked), encoding="utf-8")
    assert cli_main(["results", "pupil-field", "--request", str(path), "--mask-file", str(tmp_path / "mask.npz")]) == 0
    capsys.readouterr()
    from os_optics.local import run_optical_project

    run_dir = tmp_path / "run"
    run_optical_project(_pupil_plane_run(), output_dir=str(run_dir))
    wanted = {"wavelength_nm": LAMBDA_NM, "focal_length_mm": F_MM, "sampling": {"pixel_count": [128, 128], "pitch_mm": [0.1, 0.1]},
              "illumination": {"kind": "HANDOFF_BEAMS", "composition": "INCOHERENT"}, "output": {"half_extent_mm": [0.3, 0.3]}}
    path.write_text(json.dumps(wanted), encoding="utf-8")
    assert cli_main(["results", "pupil-field", str(run_dir), "--monitor", "PUPIL", "--request", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["composition"] == "INCOHERENT_SUM"
