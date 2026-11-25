"""Detector response of an irradiance map: photons, electrons, ADU, saturation, noise and SNR per pixel.

``detector_response(irradiance, detector)`` is a pure post-process of an irradiance document: ``irradiance_map`` of a run or the
focal field of ``pupil_field``. It simulates no light and never changes the field it reads; a detector parameter that is not
stated leaves the quantities that need it ``UNAVAILABLE`` (``results.quantity``) and the others as they are.

Model, for one wavelength ``lambda`` over one exposure ``t``:

* photon energy ``E = h c / lambda``; photon flux density ``I / E`` from the irradiance ``I`` (W/mm^2);
* pixels: a rectangle of ``nu x nv`` cells of pitch ``(pu, pv)`` centred at ``center_uv_mm`` (default: the grid's centre), each with a
  light-sensitive rectangle centred in its cell, ``sqrt(fill)`` of the cell's width on each axis. ``pixel_power_W`` is ``I`` integrated
  over that rectangle: a map of bins is constant in each bin; a map of grid points is read as Keys' cubic convolution between its points
  (third order in the grid step, and tiny negative overshoots beside a dark zone set to 0), integrated exactly, and
  ``pixels.integration.relative_error_estimate`` is its difference from the bilinear reading at the brightest pixel. A focal field
  sampled just at its Nyquist step with a pixel as wide as the step reads about 3 % low at the peak, and 1 % at 1.6 times that density;
  a larger ``fft_size`` refines it;
* ``photons = pixel_power_W t / E``; ``signal_electrons = QE(lambda) photons``, QE linear between the stated rows and unavailable
  outside them; ``dark_electrons = i_d t``;
* noise variances in e^2: shot ``signal_electrons`` and dark ``dark_electrons`` (Poisson), read ``sigma_r^2``, and their sum. ADC
  quantization noise is not in the sum;
* ``saturated``: ``signal_electrons + dark_electrons >= full well``. ``snr = signal / sqrt(total variance)``, null at a saturated pixel
  and where the variance is 0;
* ``adu`` is the mean count above the black level: ``min(signal + dark, full well) / gain``, limited to ``2^bits - 1``, not rounded to a
  digital number; ``adc_clipped``: the count reaches that limit.
* with a stated electrical response (``electrical``: each pixel a photodetector element of that response; a detector that
  does not state it gives none of these, and its document is as before), continuous detection without an exposure: ``photocurrent_A`` ``R(lambda) pixel_power_W`` with ``R = QE q lambda / (h c)``,
  ``electrical_noise_current_A`` ``sqrt((2 q I + floor^2) B)`` (the light's shot noise and the stated dark-noise floor in the
  noise-equivalent bandwidth ``B``, ``os_optics.validation.detection``) and ``electrical_snr`` their ratio, null where the noise
  is 0 and, where the detector states ``electrical.saturation_current_A`` (the linear range of one element), where the pixel's
  photocurrent is above it.

A map that does not hold all the light (``results.irradiance``): the reply passes on ``scope``, ``outside_scope``, ``map_complete``, ``untraced_power_W`` (``U``), ``untraced_fraction`` (``f``),
``max_untraced_fraction`` (the limit the map was made with: the same rule applies here), ``interference_open``, ``omitted_beams``, ``excluded_beams`` and ``omitted_sampled_power_W``, and each
quantity is judged against ``U`` in its own units. ``U`` is taken whole on one pixel, the worst case, so a pixel of
known power ``p`` holds between ``p`` and ``p + U``; where that light may interfere with the map's beams (``interference_open``, and ``composition`` is not ``INCOHERENT``), between ``(sqrt(p) - sqrt(U))^2``
(0 when ``U >= p``) and ``(sqrt(p) + sqrt(U))^2``; where nothing bounds ``U`` (a map made without a limit), from ``p`` up, or from 0 up where it may interfere. So:

* ``pixel_power_W``, ``photons``, ``signal_electrons``, the shot and total variances and ``adu`` are ``BOUNDED``: ``values`` from the known light, with ``upper`` (``bound`` ``LOWER``) or ``lower`` and ``upper``
  (``bound`` ``INTERVAL``), each the same quantity at the pixel's two ends. No ``upper`` where nothing bounds ``U``. ``photon_flux_density``, at a point, has no end from a power: ``bound`` ``LOWER`` or ``NOMINAL``;
* ``saturated`` and ``adc_clipped``: ``true`` when the pixel's lower end reaches the limit, ``false`` when its upper end stays below it, ``null`` otherwise;
* ``snr``: ``values`` with ``lower`` and ``upper`` while ``f`` is at most the map's ``max_untraced_fraction`` (default ``MAX_UNTRACED_FRACTION``, ``results.quantity``, which says why), ``UNAVAILABLE`` above it
  and where nothing bounds ``U``;
* where the map's coherent sum is withheld for that light (``coherent_reason``), it is more than that fraction and may cancel the light, so no end holds: all of the above are ``UNAVAILABLE``, unless ``composition``
  ``INCOHERENT`` states that those beams add in irradiance (tones of different frequencies). A map made with a larger ``max_untraced_fraction`` (``irradiance_map``) holds its coherent sum, and the
  quantities are stated with their ends.

A map summed over several wavelengths (``wavelength_nm`` null, ``wavelengths_nm``): ``pixel_power_W`` is stated, as are the dark and read quantities; the photon energy and every quantity that
counts photons or electrons are ``UNAVAILABLE`` (``WAVELENGTH_REQUIRED``, ``missing`` ``wavelength_nm``): read the map of one wavelength for them.

The camera sees the whole plane and the map one wavelength or one source: the quantities are of ``scope``. ``saturated`` and ``adc_clipped`` count all the light on the pixel, so while ``outside_scope`` lists other
light (whose electrons this document cannot tell) they hold ``true`` or ``null``, never ``false``. Parameters and dark quantities do not read the light and stay as they are.

A grid that does not hold the pixel array gives ``PIXEL_GRID_OUTSIDE_MAP`` for the pixel quantities: it is not read as dark. The
document's own maps stay usable; ask for a wider map (``half_extent_mm``) or a smaller array (``pixel_count``).

Coherent or incoherent map: an irradiance document with two different maps needs ``composition``; a document holding one map uses it,
and a withheld coherent map stays withheld (``coherent_reason``).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from pydantic import ValidationError

from .._vendor.shared_contracts.optical_design.detector import MAX_DETECTOR_PIXELS, DetectorModelV1
from ..validation.detection import ELEMENTARY_CHARGE_C, dark_floor, noise_bandwidth_Hz, responsivity_A_per_W
from .handoff import mark_quantities
from .quantity import (
    BEAM_POWER_UNKNOWN,
    BOUNDED,
    LOWER,
    MAX_UNTRACED_FRACTION,
    NOMINAL,
    NOT_COMPUTED,
    PARAMETER_NOT_STATED,
    UNAVAILABLE,
    UNTRACED_POWER_TOO_LARGE,
    lower_bound,
    modeled,
    nominal,
    stated,
    unavailable,
    within,
)

DETECTOR_SCHEMA = "opensci.detector_response"
DETECTOR_SCHEMA_VERSION = "1.0"
# Exact SI values (2019 redefinition).
PLANCK_J_S = 6.62607015e-34
SPEED_OF_LIGHT_M_PER_S = 299792458.0
NM_TO_M = 1e-9
# A pixel edge this far beyond the grid, as a share of the grid step, still counts as on the grid (rounding of the axes).
EDGE_TOLERANCE = 1e-9
# The integral of Keys' kernel: the coefficients 2.5 / 3, 1.5 / 4 and 0.5 / 4 of its parts, its integral from 0 to 1, and the fewest points per axis it needs.
KEYS_B = 2.5 / 3.0
KEYS_A = 0.375
KEYS_C = 0.125
KEYS_INNER = 1.0 - 2.5 / 3.0 + 0.375
MIN_CUBIC_POINTS = 3

DETECTOR_MODEL = ("Linear detector over one exposure: photons from the irradiance integrated over each pixel's light-sensitive area, "
                  "electrons by the stated quantum efficiency, Poisson shot and dark noise, stated read noise; no pixel crosstalk, "
                  "lag, nonlinearity or ADC quantization noise.")
NO_FLOOR = "light of unknown power that may interfere reaches this plane: its field may add to the light here or cancel it. State the element's efficiency, or composition INCOHERENT where they add in irradiance"
#: Why a value stands on light the plane bounds would state, in an evaluation that did not compute them.
BOUNDS_NOT_COMPUTED = ("the plane bounds were not computed, so the light the map leaves out is not known: it may add to the light here or "
                       "cancel it. Evaluate with plane_bounds, or read composition INCOHERENT where the light adds in irradiance")
WITHHELD = (BEAM_POWER_UNKNOWN, UNTRACED_POWER_TOO_LARGE, NOT_COMPUTED)
DETECTOR_INVALID = "DETECTOR_INVALID"
#: An irradiance document that lacks a member the response reads, or holds one of the wrong form (``_irradiance_problem``).
IRRADIANCE_INVALID = "IRRADIANCE_INVALID"
IRRADIANCE_UNAVAILABLE = "IRRADIANCE_UNAVAILABLE"
COMPOSITION_REQUIRED = "COMPOSITION_REQUIRED"
COMPOSITION_INVALID = "COMPOSITION_INVALID"
WAVELENGTH_REQUIRED = "WAVELENGTH_REQUIRED"
#: Why a photon or electron quantity is not stated for a map summed over several wavelengths.
SEVERAL_WAVELENGTHS = "the map sums several wavelengths, and the photon energy and the quantum efficiency are those of one: read the map of one wavelength (wavelength_nm)"
#: The quantities that do not read the photon energy: the power on a pixel, and the dark and read quantities.
WAVELENGTH_FREE = ("pixel_power_W", "dark_electrons", "dark_noise_variance_e2", "read_noise_variance_e2")
PIXEL_GRID_OUTSIDE_MAP = "PIXEL_GRID_OUTSIDE_MAP"
WAVELENGTH_OUTSIDE_QE_TABLE = "WAVELENGTH_OUTSIDE_QE_TABLE"
COHERENT = "COHERENT"
INCOHERENT = "INCOHERENT"
ORDER = "row-major: rows are v, columns u"

#: The stated parameters each quantity needs, whichever quantity they reach it through.
PIXEL_PARAMETERS = ("pixel_pitch_mm", "fill_factor")
PHOTON_PARAMETERS = (*PIXEL_PARAMETERS, "exposure_s")
ELECTRON_PARAMETERS = (*PHOTON_PARAMETERS, "quantum_efficiency")
DARK_PARAMETERS = ("dark_current_e_per_s", "exposure_s")
QUANTITY_PARAMETERS: dict[str, tuple[str, ...]] = {
    "pixel_power_W": PIXEL_PARAMETERS,
    "photons": PHOTON_PARAMETERS,
    "signal_electrons": ELECTRON_PARAMETERS,
    "dark_electrons": DARK_PARAMETERS,
    "shot_noise_variance_e2": ELECTRON_PARAMETERS,
    "dark_noise_variance_e2": DARK_PARAMETERS,
    "read_noise_variance_e2": ("read_noise_e",),
    "total_noise_variance_e2": (*ELECTRON_PARAMETERS, "dark_current_e_per_s", "read_noise_e"),
    "snr": (*ELECTRON_PARAMETERS, "dark_current_e_per_s", "read_noise_e"),
    "saturated": (*ELECTRON_PARAMETERS, "dark_current_e_per_s", "full_well_e"),
    "adu": (*ELECTRON_PARAMETERS, "dark_current_e_per_s", "full_well_e", "gain_e_per_adu", "bit_depth"),
    "adc_clipped": (*ELECTRON_PARAMETERS, "dark_current_e_per_s", "full_well_e", "gain_e_per_adu", "bit_depth"),
}
#: The quantities of a detector that states ``electrical`` (continuous detection; ``os_optics.validation.detection``).
ELECTRICAL_PARAMETERS: dict[str, tuple[str, ...]] = {
    "photocurrent_A": (*PIXEL_PARAMETERS, "quantum_efficiency"),
    "electrical_noise_current_A": (*PIXEL_PARAMETERS, "quantum_efficiency", "electrical"),
    "electrical_snr": (*PIXEL_PARAMETERS, "quantum_efficiency", "electrical"),
}
#: The quantities that need the pixel array on the map.
PIXEL_QUANTITIES = ("pixel_power_W", "photons", "signal_electrons", "shot_noise_variance_e2", "total_noise_variance_e2", "snr", "saturated",
                    "adu", "adc_clipped", "photocurrent_A", "electrical_noise_current_A", "electrical_snr")
#: Ratios whose value is null where their noise is 0, bounded like the SNR on a map that does not hold all the light.
SNR_QUANTITIES = ("snr", "electrical_snr")
QUANTITY_UNITS = {"pixel_power_W": "W", "photons": "photon", "signal_electrons": "e", "dark_electrons": "e", "shot_noise_variance_e2": "e^2",
                  "dark_noise_variance_e2": "e^2", "read_noise_variance_e2": "e^2", "total_noise_variance_e2": "e^2", "snr": None, "saturated": None,
                  "adu": "ADU", "adc_clipped": None, "photocurrent_A": "A", "electrical_noise_current_A": "A", "electrical_snr": None}
STATED_INPUTS = {"exposure_s": "s", "pixel_pitch_mm": "mm", "fill_factor": None, "full_well_e": "e", "read_noise_e": "e rms",
                 "dark_current_e_per_s": "e/s", "gain_e_per_adu": "e/ADU", "bit_depth": "bit"}


class DetectorError(Exception):
    """A detector document that is not a valid ``DetectorModelV1``."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code, self.message = code, message


def parse_detector(detector: Any) -> DetectorModelV1:
    """``detector`` as a ``DetectorModelV1``, or ``DetectorError`` with what is wrong."""

    if isinstance(detector, DetectorModelV1):
        return detector
    try:
        return DetectorModelV1.model_validate(detector)
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(str(p) for p in e['loc']) or 'detector'}: {e['msg']}" for e in exc.errors(include_url=False))
        raise DetectorError(DETECTOR_INVALID, problems) from None


def _refusal(reason: str, message: str, **context: Any) -> dict[str, Any]:
    return {"status": UNAVAILABLE, "schema": DETECTOR_SCHEMA, "schema_version": DETECTOR_SCHEMA_VERSION, "reason": reason, "message": message, **context}


def _axis(value: Any) -> int | None:
    """The length of a 1-D array of finite numbers with at least one member; ``None`` where ``value`` is not one."""

    if not isinstance(value, list) or not value:
        return None
    if not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in value):
        return None
    return len(value)


def _irradiance_problem(irradiance: dict[str, Any]) -> str | None:
    """What is wrong with the structure of an irradiance document whose ``status`` is ``OK`` (the member at fault), or
    ``None``: ``grid`` an object with ``u_mm`` and ``v_mm`` 1-D arrays of finite numbers (and ``u_edges_mm`` / ``v_edges_mm``,
    where stated, one longer), each map stated a 2-D array of numbers of shape ``(len(v_mm), len(u_mm))`` (rows v, columns
    u), and the members the response reads of their types."""

    grid = irradiance.get("grid")
    if not isinstance(grid, dict):
        return "grid: an object with u_mm and v_mm is required"
    sizes = {}
    for axis in ("u_mm", "v_mm"):
        sizes[axis] = _axis(grid.get(axis))
        if sizes[axis] is None:
            return f"grid.{axis}: a non-empty array of finite numbers (mm) is required"
    for edges, axis in (("u_edges_mm", "u_mm"), ("v_edges_mm", "v_mm")):
        if grid.get(edges) is not None and _axis(grid[edges]) != sizes[axis] + 1:
            return f"grid.{edges}: an array of {sizes[axis] + 1} finite numbers (the bin edges of grid.{axis}) is required"
    for name in ("coherent_W_per_mm2", "incoherent_W_per_mm2"):
        rows = irradiance.get(name)
        if rows is None:
            continue
        if (not isinstance(rows, list) or len(rows) != sizes["v_mm"]
                or not all(isinstance(row, list) and len(row) == sizes["u_mm"] for row in rows)
                or not all(isinstance(x, (int, float)) and not isinstance(x, bool) for row in rows for x in row)):
            return (f"{name}: a 2-D array of numbers of {sizes['v_mm']} rows (grid.v_mm) of {sizes['u_mm']} values (grid.u_mm) "
                    "is required")
    kinds: dict[str, tuple[type, ...]] = {"map_complete": (bool,), "interference_open": (bool,), "coherent_reason": (str,),
                                          "outside_scope": (list,), "omitted_beams": (list,), "excluded_beams": (list,), "needs": (list,)}
    for name, types in kinds.items():
        if irradiance.get(name) is not None and not isinstance(irradiance[name], types):
            return f"{name}: {'true or false' if types == (bool,) else 'a string' if types == (str,) else 'an array'} or null is required"
    for name in ("wavelength_nm", "untraced_power_W", "untraced_fraction", "max_untraced_fraction"):
        value = irradiance.get(name)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not value >= 0.0
                                  or (name == "wavelength_nm" and not 0.0 < value < math.inf)):
            return f"{name}: {'a positive number (nm)' if name == 'wavelength_nm' else 'a number of at least 0'} or null is required"
    return None


def _select_map(irradiance: dict[str, Any], composition: str | None) -> tuple[np.ndarray | None, str | None, dict[str, Any] | None]:
    """The map to read, its composition and, when there is none to read, the refusal."""

    coherent, incoherent = irradiance.get("coherent_W_per_mm2"), irradiance.get("incoherent_W_per_mm2")
    if composition not in (None, COHERENT, INCOHERENT):
        return None, None, _refusal(COMPOSITION_INVALID, f"composition is {COHERENT} or {INCOHERENT}, not {composition!r}")
    if composition == COHERENT and coherent is None:
        return None, None, _refusal(IRRADIANCE_UNAVAILABLE, "this irradiance document holds no coherent map", upstream_reason=irradiance.get("coherent_reason"))
    if composition == INCOHERENT and incoherent is None:
        return None, None, _refusal(IRRADIANCE_UNAVAILABLE, "this irradiance document holds no incoherent map")
    if composition is None and coherent is not None and incoherent is not None:
        if coherent != incoherent:
            return None, None, _refusal(COMPOSITION_REQUIRED, f"this irradiance document holds two different maps; choose one with composition {COHERENT} or {INCOHERENT}")
        composition = COHERENT
    if composition is None:
        composition = COHERENT if coherent is not None else INCOHERENT
    chosen = coherent if composition == COHERENT else incoherent
    if chosen is None:
        return None, None, _refusal(IRRADIANCE_UNAVAILABLE, "this irradiance document holds no map")
    return np.array(chosen, dtype=np.float64), composition, None


def _hat_integral(s: np.ndarray) -> np.ndarray:
    """The running integral, in steps, of a unit hat function centred at ``s = 0``."""

    return np.where(s <= -1.0, 0.0, np.where(s <= 0.0, 0.5 * (s + 1.0) ** 2, np.where(s <= 1.0, 1.0 - 0.5 * (1.0 - s) ** 2, 1.0)))


def _cubic_integral(s: np.ndarray) -> np.ndarray:
    """The running integral, in steps, of Keys' cubic convolution kernel (a = -1/2) centred at ``s = 0``: the kernel is
    ``1.5 |s|^3 - 2.5 |s|^2 + 1`` up to ``|s| = 1`` and ``-0.5 |s|^3 + 2.5 |s|^2 - 4 |s| + 2`` up to ``|s| = 2``; it integrates to 1."""

    a = np.abs(s)
    a2, a3, a4 = a * a, a * a * a, a * a * a * a
    inner = a - KEYS_B * a3 + KEYS_A * a4
    outer = KEYS_INNER + 2.0 * (a - 1.0) - 2.0 * (a2 - 1.0) + KEYS_B * (a3 - 1.0) - KEYS_C * (a4 - 1.0)
    half = np.where(a <= 1.0, inner, np.where(a <= 2.0, outer, 0.5))
    return 0.5 + np.sign(s) * half


def _point_weights(integral: Any, axis: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Per pixel and grid point: the integral over ``[low, high]`` of the point's kernel, whose running integral is ``integral``."""

    step = (axis[-1] - axis[0]) / (axis.size - 1)
    index = np.arange(axis.size)
    return step * (integral((high[:, None] - axis[0]) / step - index) - integral((low[:, None] - axis[0]) / step - index))


def _with_ghost_points(map_: np.ndarray) -> np.ndarray:
    """``map_`` with one more row and column on each side, by Keys' boundary condition ``g = 3 m0 - 3 m1 + m2`` (it keeps the kernel's order)."""

    rows = np.vstack([3.0 * map_[0] - 3.0 * map_[1] + map_[2], map_, 3.0 * map_[-1] - 3.0 * map_[-2] + map_[-3]])
    first = 3.0 * rows[:, 0] - 3.0 * rows[:, 1] + rows[:, 2]
    last = 3.0 * rows[:, -1] - 3.0 * rows[:, -2] + rows[:, -3]
    return np.hstack([first[:, None], rows, last[:, None]])


def _grid_point_power(map_: np.ndarray, u: np.ndarray, v: np.ndarray, u_range: tuple[np.ndarray, np.ndarray],
                      v_range: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:
    """The power per pixel of a map of grid points, by cubic convolution, and what is known of its accuracy (module docstring)."""

    bilinear = _point_weights(_hat_integral, v, *v_range) @ map_ @ _point_weights(_hat_integral, u, *u_range).T
    if min(u.size, v.size) < MIN_CUBIC_POINTS:
        return bilinear, {"method": "BILINEAR", "relative_error_estimate": None}
    step_u, step_v = (u[-1] - u[0]) / (u.size - 1), (v[-1] - v[0]) / (v.size - 1)
    padded_u = np.concatenate([[u[0] - step_u], u, [u[-1] + step_u]])
    padded_v = np.concatenate([[v[0] - step_v], v, [v[-1] + step_v]])
    cubic = _point_weights(_cubic_integral, padded_v, *v_range) @ _with_ghost_points(map_) @ _point_weights(_cubic_integral, padded_u, *u_range).T
    cubic = np.clip(cubic, 0.0, None)
    brightest = np.unravel_index(int(np.argmax(cubic)), cubic.shape)
    peak = float(cubic[brightest])
    estimate = abs(float(bilinear[brightest]) - peak) / peak if peak > 0.0 else None
    return cubic, {"method": "CUBIC_CONVOLUTION", "relative_error_estimate": estimate, "grid_step_mm": [float(step_u), float(step_v)]}


def _bin_weights(edges: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Per pixel and bin: the overlap of ``[low, high]`` with the bin (a map constant in each bin)."""

    return np.clip(np.minimum(high[:, None], edges[1:]) - np.maximum(low[:, None], edges[:-1]), 0.0, None)


def _pixel_power(map_: np.ndarray, grid: dict[str, Any], model: DetectorModelV1) -> tuple[np.ndarray | None, dict[str, Any] | None, str | None]:
    """``(power per pixel, the pixel array, why there is none)``. Rows are v, columns u."""

    u, v = np.array(grid["u_mm"], dtype=np.float64), np.array(grid["v_mm"], dtype=np.float64)
    u_edges, v_edges = grid.get("u_edges_mm"), grid.get("v_edges_mm")
    binned = u_edges is not None and v_edges is not None
    u_span = (float(u_edges[0]), float(u_edges[-1])) if binned else (float(u[0]), float(u[-1]))
    v_span = (float(v_edges[0]), float(v_edges[-1])) if binned else (float(v[0]), float(v[-1]))
    center = model.center_uv_mm or (0.5 * (u_span[0] + u_span[1]), 0.5 * (v_span[0] + v_span[1]))
    pitch = model.pixel_pitch_mm
    tolerance = EDGE_TOLERANCE * max((u_span[1] - u_span[0]) / u.size, (v_span[1] - v_span[0]) / v.size)
    count = model.pixel_count
    if count is None:
        fitting = [int((min(span[1] - c, c - span[0]) + tolerance) // (0.5 * p)) for span, c, p in zip((u_span, v_span), center, pitch)]
        count = (fitting[0], fitting[1])
        if not all(1 <= n <= MAX_DETECTOR_PIXELS for n in count):
            return None, None, f"{count[0]} x {count[1]} pixels of this pitch fit the grid; state pixel_count from 1 to {MAX_DETECTOR_PIXELS}, or read another map"
    active = [p * math.sqrt(model.fill_factor) for p in pitch]  # type: ignore[arg-type]
    u_center = center[0] + (np.arange(count[0]) - 0.5 * (count[0] - 1)) * pitch[0]
    v_center = center[1] + (np.arange(count[1]) - 0.5 * (count[1] - 1)) * pitch[1]
    u_low, u_high = u_center - 0.5 * active[0], u_center + 0.5 * active[0]
    v_low, v_high = v_center - 0.5 * active[1], v_center + 0.5 * active[1]
    if (u_low[0] < u_span[0] - tolerance or u_high[-1] > u_span[1] + tolerance or v_low[0] < v_span[0] - tolerance or v_high[-1] > v_span[1] + tolerance):
        return None, None, ("the pixel array reaches beyond the irradiance grid; read a wider map (half_extent_mm) or state a smaller "
                            "pixel_count")
    u_low, u_high = np.clip(u_low, *u_span), np.clip(u_high, *u_span)
    v_low, v_high = np.clip(v_low, *v_span), np.clip(v_high, *v_span)
    if binned:
        wu, wv = _bin_weights(np.array(u_edges, dtype=np.float64), u_low, u_high), _bin_weights(np.array(v_edges, dtype=np.float64), v_low, v_high)
        power, integration = wv @ map_ @ wu.T, {"method": "CONSTANT_PER_BIN", "relative_error_estimate": None}
    else:
        power, integration = _grid_point_power(map_, u, v, (u_low, u_high), (v_low, v_high))
    pixels = {"count": [count[0], count[1]], "shape": [count[1], count[0]], "order": ORDER, "pitch_mm": list(pitch), "active_width_mm": active,  # type: ignore[arg-type]
              "center_uv_mm": [center[0], center[1]], "u_mm": u_center.tolist(), "v_mm": v_center.tolist(), "integration": integration}
    return power, pixels, None


def _quantum_efficiency(model: DetectorModelV1, wavelength_nm: float) -> float | None:
    rows = model.quantum_efficiency.rows  # type: ignore[union-attr]
    wavelengths = [row.wavelength_nm for row in rows]
    if wavelength_nm < wavelengths[0] or wavelength_nm > wavelengths[-1]:
        return None
    return float(np.interp(wavelength_nm, wavelengths, [row.qe for row in rows]))


def _source(source: Any) -> dict[str, Any] | None:
    """Where a stated value comes from as the detector states it; ``None``: the detector does not state it."""

    return None if source is None else source.model_dump(mode="json")


def _inputs(model: DetectorModelV1, qe: float | None, wavelength_nm: float | None) -> dict[str, Any]:
    """Every parameter as stated or absent: what the quantities were computed from. A stated value's ``source`` is where
    the detector says it comes from, ``null`` when it does not say."""

    source = _source(model.characterization)
    inputs: dict[str, Any] = {}
    for name, unit in STATED_INPUTS.items():
        value = getattr(model, name)
        if value is None:
            inputs[name] = unavailable(unit, PARAMETER_NOT_STATED, f"{name} is not stated", [name])
        else:
            value = list(value) if isinstance(value, tuple) else value
            inputs[name] = stated(unit, value, source if name in ("full_well_e", "read_noise_e", "dark_current_e_per_s", "gain_e_per_adu") else None)
    if model.quantum_efficiency is None:
        inputs["quantum_efficiency"] = unavailable(None, PARAMETER_NOT_STATED, "quantum_efficiency is not stated", ["quantum_efficiency"])
    elif wavelength_nm is None:
        inputs["quantum_efficiency"] = unavailable(None, WAVELENGTH_REQUIRED, SEVERAL_WAVELENGTHS, ["wavelength_nm"])
    elif qe is None:
        inputs["quantum_efficiency"] = unavailable(None, WAVELENGTH_OUTSIDE_QE_TABLE, f"{wavelength_nm} nm is outside the stated table",
                                                   ["quantum_efficiency"])
    else:
        inputs["quantum_efficiency"] = stated(None, qe, _source(model.quantum_efficiency.source))
    if model.electrical is not None:
        floor, reason = dark_floor(model)
        if floor is None:
            inputs["electrical"] = unavailable(None, reason or PARAMETER_NOT_STATED, "the dark-noise floor needs the quantum efficiency at "
                                               "nep_wavelength_nm", ["quantum_efficiency"])
        else:
            electrical = model.electrical.model_dump(mode="json", exclude_none=True)
            electrical.pop("source", None)
            inputs["electrical"] = stated(None, {**electrical, "noise_equivalent_bandwidth_Hz": noise_bandwidth_Hz(model.electrical),
                                                 "dark_noise_density_A_per_rtHz": floor},
                                          _source(model.electrical.source))
    return inputs


def _missing_parameters(inputs: dict[str, Any], parameters: tuple[str, ...]) -> list[str]:
    return sorted({name for name in parameters if inputs[name]["status"] == UNAVAILABLE})


def _maps(pixel_power: np.ndarray, model: DetectorModelV1, photon_energy_J: float | None, qe: float | None,
          wavelength_nm: float | None = 0.0) -> dict[str, np.ndarray | float]:
    """Every value the stated parameters give (``detector_response`` decides which ones to report). Without a photon energy (a
    map summed over several wavelengths): the power on each pixel alone."""

    out: dict[str, np.ndarray | float] = {"pixel_power_W": pixel_power}
    if photon_energy_J is None or wavelength_nm is None:
        return out
    if qe is not None and model.electrical is not None:  # continuous detection: no exposure needed
        current = responsivity_A_per_W(qe, wavelength_nm) * pixel_power
        out["photocurrent_A"] = current
        floor = None if model.electrical is None else dark_floor(model)[0]
        if floor is not None:
            noise = np.sqrt((2.0 * ELEMENTARY_CHARGE_C * current + floor * floor) * noise_bandwidth_Hz(model.electrical))
            out["electrical_noise_current_A"] = noise
            out["electrical_snr"] = np.divide(current, noise, out=np.zeros_like(current), where=noise > 0.0)
            saturation = model.electrical.saturation_current_A
            # (null where the noise is 0, and, like a saturated pixel's snr, above a stated linear range)
            out["electrical_snr_null"] = noise <= 0.0 if saturation is None else (noise <= 0.0) | (current > saturation)
    exposure, read, dark_current, well, gain, bits = (model.exposure_s, model.read_noise_e, model.dark_current_e_per_s, model.full_well_e,
                                                      model.gain_e_per_adu, model.bit_depth)
    if exposure is None:
        return out
    out["photons"] = pixel_power * exposure / photon_energy_J
    if qe is None:
        return out
    signal = qe * pixel_power * exposure / photon_energy_J
    out["signal_electrons"] = signal
    out["shot_noise_variance_e2"] = signal
    if dark_current is None:
        return out
    dark = dark_current * exposure
    saturated = None if well is None else signal + dark >= well
    if read is not None:
        total = signal + dark + read * read
        out["total_noise_variance_e2"] = total
        out["snr"] = np.divide(signal, np.sqrt(total), out=np.zeros_like(signal), where=total > 0.0)
        out["snr_null"] = total <= 0.0 if saturated is None else (total <= 0.0) | saturated
    if well is not None:
        out["saturated"] = saturated  # type: ignore[assignment]
        if gain is not None and bits is not None:
            limit = 2.0 ** bits - 1.0
            counts = np.minimum(signal + dark, well) / gain
            out["adu"] = np.minimum(counts, limit)
            out["adc_clipped"] = counts >= limit
    return out


def detector_response(irradiance: dict[str, Any], detector: Any, composition: str | None = None) -> dict[str, Any]:
    """The detector response of an irradiance document (module docstring). ``composition``: ``COHERENT`` or ``INCOHERENT``, the map
    to read when the document holds two different ones. A document without a map gives ``status`` ``UNAVAILABLE`` with the reason;
    otherwise ``status`` ``OK`` with ``quantities``, each ``MODELED``, ``BOUNDED`` or ``UNAVAILABLE`` (``results.quantity``), and ``inputs``.
    Raises ``DetectorError``: ``DETECTOR_INVALID`` (the detector), ``IRRADIANCE_INVALID`` (a document of status ``OK`` that lacks a
    member the response reads or holds one of the wrong form, the member named)."""

    model = parse_detector(detector)
    if not isinstance(irradiance, dict) or irradiance.get("status") != "OK":
        upstream = irradiance.get("reason") if isinstance(irradiance, dict) else None
        return _refusal(IRRADIANCE_UNAVAILABLE, "the irradiance document holds no map; its stage reports why", upstream_reason=upstream)
    problem = _irradiance_problem(irradiance)
    if problem is not None:
        raise DetectorError(IRRADIANCE_INVALID, f"the irradiance document is not one 'results irradiance' or 'results pupil-field' "
                                                f"gives: {problem}")
    map_, chosen, refusal = _select_map(irradiance, composition)
    if refusal is not None:
        return refusal
    # (a map summed over several wavelengths states none: the power on a pixel does not need one, a photon count does)
    wavelength_nm = None if irradiance.get("wavelength_nm") is None else float(irradiance["wavelength_nm"])
    photon_energy_J = None if wavelength_nm is None else PLANCK_J_S * SPEED_OF_LIGHT_M_PER_S / (wavelength_nm * NM_TO_M)
    qe = None if model.quantum_efficiency is None or wavelength_nm is None else _quantum_efficiency(model, wavelength_nm)
    inputs = _inputs(model, qe, wavelength_nm)
    grid = irradiance["grid"]
    complete = irradiance.get("map_complete", True)
    interfering = bool(irradiance.get("interference_open")) and composition != INCOHERENT
    of_light = modeled if complete else nominal if interfering else lower_bound
    # (more than the fraction, and it may cancel the light: no end holds, unless the reader states the sum is incoherent)
    withheld = irradiance.get("coherent_reason") if irradiance.get("coherent_reason") in WITHHELD else None
    no_floor = withheld is not None and composition != INCOHERENT
    floor_text = BOUNDS_NOT_COMPUTED if withheld == NOT_COMPUTED else NO_FLOOR
    if photon_energy_J is None:
        quantities: dict[str, Any] = {name: unavailable(unit, WAVELENGTH_REQUIRED, SEVERAL_WAVELENGTHS, ["wavelength_nm"])
                                      for name, unit in (("photon_energy_J", "J"), ("photon_flux_density", "photon/(s mm^2)"))}
    else:
        quantities = {
            "photon_energy_J": modeled("J", "h c / lambda", value=photon_energy_J),
            "photon_flux_density": (unavailable("photon/(s mm^2)", withheld, floor_text) if no_floor else
                                    of_light("photon/(s mm^2)", "irradiance / photon energy, on the irradiance grid", values=(map_ / photon_energy_J).tolist())),
        }
    pixel_power, pixels, outside = (None, None, None)
    if model.pixel_pitch_mm is not None and model.fill_factor is not None:
        pixel_power, pixels, outside = _pixel_power(map_, grid, model)
    values = {} if pixel_power is None or no_floor else _maps(pixel_power, model, photon_energy_J, qe, wavelength_nm)
    outside_scope = irradiance.get("outside_scope", [])
    ends = None
    if values:
        low, high = _pixel_ends(pixel_power, None if complete else irradiance.get("untraced_power_W"), complete, interfering)
        ends = (_maps(low, model, photon_energy_J, qe, wavelength_nm), None if high is None else _maps(high, model, photon_energy_J, qe, wavelength_nm))
    fraction = irradiance.get("untraced_fraction")
    # (the limit the map was made with: one rule for the map and for what is read from it; ``None``: no limit)
    limit = irradiance["max_untraced_fraction"] if "max_untraced_fraction" in irradiance else MAX_UNTRACED_FRACTION
    bounds_missing = "plane_bounds" in (irradiance.get("needs") or ())
    too_large = not complete and (bounds_missing or (limit is not None and (fraction is None or fraction > limit)))
    # (no limit, and nothing bounds the light the map leaves out: the SNR has no end)
    unbounded = not complete and not too_large and irradiance.get("untraced_power_W") is None
    snr_refusal = None
    if too_large and withheld == NOT_COMPUTED:
        snr_refusal = unavailable(QUANTITY_UNITS["snr"], NOT_COMPUTED, "the plane bounds were not computed, so the light the map leaves "
                                  "out is not known: it adds signal and noise here; evaluate with plane_bounds")
    elif too_large or unbounded:
        share = "nothing bounds the light the map leaves out" if fraction is None else f"the light the map leaves out may be {fraction:.3g} of the light in it"
        above = "" if unbounded or limit is None else f", above max_untraced_fraction {limit:g}"
        snr_refusal = unavailable(QUANTITY_UNITS["snr"], BEAM_POWER_UNKNOWN if irradiance.get("omitted_beams") or withheld == BEAM_POWER_UNKNOWN else UNTRACED_POWER_TOO_LARGE,
                                  f"{share}{above}: it adds signal and noise here, and may saturate the pixel")
    dark_electrons = None if model.dark_current_e_per_s is None or model.exposure_s is None else model.dark_current_e_per_s * model.exposure_s
    for name, parameters in {**QUANTITY_PARAMETERS, **(ELECTRICAL_PARAMETERS if model.electrical is not None else {})}.items():
        missing = _missing_parameters(inputs, parameters)
        unit = QUANTITY_UNITS[name]
        if photon_energy_J is None and name not in WAVELENGTH_FREE:
            quantities[name] = unavailable(unit, WAVELENGTH_REQUIRED, SEVERAL_WAVELENGTHS, ["wavelength_nm"])
        elif missing:
            reason = WAVELENGTH_OUTSIDE_QE_TABLE if inputs["quantum_efficiency"].get("reason") == WAVELENGTH_OUTSIDE_QE_TABLE and "quantum_efficiency" in missing else PARAMETER_NOT_STATED
            quantities[name] = unavailable(unit, reason, f"needs {', '.join(missing)}", missing)
        elif name == "dark_electrons":
            quantities[name] = modeled(unit, "dark current x exposure", value=dark_electrons)
        elif name == "dark_noise_variance_e2":
            quantities[name] = modeled(unit, "Poisson: the dark electrons", value=dark_electrons)
        elif name == "read_noise_variance_e2":
            quantities[name] = modeled(unit, "read noise squared", value=model.read_noise_e * model.read_noise_e)  # type: ignore[operator]
        elif name in PIXEL_QUANTITIES and no_floor:
            quantities[name] = unavailable(unit, withheld, floor_text)
        elif name in PIXEL_QUANTITIES and pixel_power is None:
            quantities[name] = unavailable(unit, PIXEL_GRID_OUTSIDE_MAP, outside or "the pixel array is not on the map")
        else:
            quantities[name] = _pixel_quantity(name, unit, values, complete, ends, bool(outside_scope), interfering, snr_refusal)
    unavailable_names = sorted(name for name, q in quantities.items() if q["status"] == UNAVAILABLE)
    totals = _totals(values, pixels)
    if totals and ends is not None and (not complete or outside_scope):
        for name, key in (("saturated", "saturated_pixels"), ("adc_clipped", "adc_clipped_pixels")):
            if name in values:
                flags = [flag for row in _yes_no(name, ends, bool(outside_scope)) for flag in row]
                totals[key] = sum(1 for flag in flags if flag is True)
                totals[f"{key}_not_known"] = sum(1 for flag in flags if flag is None)
        if not complete:
            totals["bound"] = NOMINAL if interfering else LOWER
    omitted = {"map_complete": complete, "scope": irradiance.get("scope"), "outside_scope": outside_scope, "omitted_beams": irradiance.get("omitted_beams", []),
               "excluded_beams": irradiance.get("excluded_beams", []),
               "untraced_power_W": irradiance.get("untraced_power_W", 0.0), "untraced_fraction": irradiance.get("untraced_fraction", 0.0),
               "max_untraced_fraction": limit, "interference_open": bool(irradiance.get("interference_open"))}
    if "omitted_sampled_power_W" in irradiance:
        omitted["omitted_sampled_power_W"] = irradiance["omitted_sampled_power_W"]
    if irradiance.get("wavelengths_nm") is not None:
        omitted["wavelengths_nm"] = irradiance["wavelengths_nm"]
    scope = irradiance.get("model_scope") or []
    if scope:
        # (light a model-scope finding reaches: every number read from it carries the finding's ESTIMATE, as the map does)
        mark_quantities(quantities, scope, skip=LIGHT_FREE_QUANTITIES)
        omitted["model_scope"] = scope
    return {
        "status": "OK", "schema": DETECTOR_SCHEMA, "schema_version": DETECTOR_SCHEMA_VERSION, "model": DETECTOR_MODEL,
        "wavelength_nm": wavelength_nm, "composition": chosen, "coherent_reason": irradiance.get("coherent_reason"),
        "irradiance_model": irradiance.get("model"), "observation_id": irradiance.get("observation_id"),
        "irradiance_grid": {key: grid[key] for key in ("u_mm", "v_mm", "shape", "order") if key in grid},
        "inputs": inputs, "pixels": pixels, "quantities": quantities, "unavailable_quantities": unavailable_names,
        "bounded_quantities": sorted(name for name, q in quantities.items() if q["status"] == BOUNDED), **omitted,
        "totals": totals,
    }


#: What each pixel quantity is computed from (its ``basis``).
PIXEL_BASIS = {
    "pixel_power_W": "irradiance integrated over the light-sensitive area of each pixel (pixels.integration)",
    "photons": "pixel power x exposure / photon energy",
    "signal_electrons": "quantum efficiency x photons",
    "shot_noise_variance_e2": "Poisson: the signal electrons",
    "total_noise_variance_e2": "shot + dark + read variance",
    "snr": "signal electrons / sqrt(total variance); null where the variance is 0 and, when a full well is stated, at a saturated pixel",
    "saturated": "signal + dark electrons >= full well",
    "adu": "min(signal + dark, full well) / gain, limited to 2^bits - 1: the mean count above the black level, not rounded",
    "adc_clipped": "the count reaches 2^bits - 1",
    "photocurrent_A": "QE q lambda / (h c) x pixel power: continuous detection, no exposure",
    "electrical_noise_current_A": "sqrt((2 q I + dark-noise density^2) B): shot noise and the stated floor in the noise-equivalent bandwidth",
    "electrical_snr": "photocurrent / electrical noise current (amplitude ratio); null where the noise is 0",
}


#: The quantities that do not read the light: the photon energy and the dark and read noise.
LIGHT_FREE_QUANTITIES = frozenset({"photon_energy_J", "dark_electrons", "dark_noise_variance_e2", "read_noise_variance_e2"})


#: Pixel quantities that say yes or no.
YES_NO_QUANTITIES = ("saturated", "adc_clipped")


def _pixel_ends(pixel_power: np.ndarray, untraced_W: float | None, complete: bool, interfering: bool) -> tuple[np.ndarray, np.ndarray | None]:
    """The two ends of each pixel's power with the light the map leaves out taken whole on it (module docstring); ``None``: nothing bounds it above."""

    if complete:
        return pixel_power, pixel_power
    if untraced_W is None:
        # (nothing bounds that light: no upper end; where it may interfere it may also cancel the light here, so the lower end is 0)
        return (np.zeros_like(pixel_power) if interfering else pixel_power), None
    if not interfering:
        return pixel_power, pixel_power + untraced_W
    root, extra = np.sqrt(pixel_power), math.sqrt(untraced_W)
    return np.clip(root - extra, 0.0, None) ** 2, (root + extra) ** 2


def _yes_no(name: str, ends: tuple[dict[str, Any], dict[str, Any] | None], outside: bool) -> list[list[bool | None]]:
    """``true`` where the lower end reaches the limit, ``false`` where nothing else on the plane can and the upper end does not, else not known (module docstring)."""

    low = ends[0][name].tolist()
    high = None if ends[1] is None or outside else ends[1][name].tolist()
    return [[True if low[j][i] else False if high is not None and not high[j][i] else None for i in range(len(low[j]))] for j in range(len(low))]


def _snr_rows(values: dict[str, Any], name: str = "snr") -> list[list[float | None]]:
    return [[None if null else snr for snr, null in zip(row, flags)] for row, flags in zip(values[name].tolist(), values[f"{name}_null"].tolist())]


def _pixel_quantity(name: str, unit: str | None, values: dict[str, Any], complete: bool, ends: tuple[dict[str, Any], dict[str, Any] | None] | None, outside: bool, interfering: bool,
                    snr_refusal: dict[str, Any] | None) -> dict[str, Any]:
    assert ends is not None  # (a pixel quantity is read only where the pixel array is on the map)
    if name in YES_NO_QUANTITIES and (outside or not complete):
        return lower_bound(unit, PIXEL_BASIS[name], values=_yes_no(name, ends, outside))
    if complete:
        return modeled(unit, PIXEL_BASIS[name], values=_snr_rows(values, name) if name in SNR_QUANTITIES else values[name].tolist())
    if name in SNR_QUANTITIES:
        if snr_refusal is not None or ends[1] is None:
            return snr_refusal  # type: ignore[return-value]
        return within(unit, PIXEL_BASIS[name], _snr_rows(values, name), _snr_rows(ends[0], name), _snr_rows(ends[1], name))
    upper = None if ends[1] is None else ends[1][name].tolist()
    if interfering:
        return within(unit, PIXEL_BASIS[name], values[name].tolist(), ends[0][name].tolist(), upper)
    return lower_bound(unit, PIXEL_BASIS[name], values=values[name].tolist(), **({} if upper is None else {"upper": upper}))


def _totals(values: dict[str, Any], pixels: dict[str, Any] | None) -> dict[str, Any]:
    """Sums and peaks over the pixel array, of what is available."""

    if pixels is None:
        return {}
    totals: dict[str, Any] = {"pixel_count": pixels["count"][0] * pixels["count"][1]}
    for name, key in (("pixel_power_W", "power_on_pixels_W"), ("photons", "photons"), ("signal_electrons", "signal_electrons")):
        if name in values:
            totals[key] = float(values[name].sum())
            totals[f"peak_{name}"] = float(values[name].max())
    if "saturated" in values:
        totals["saturated_pixels"] = int(values["saturated"].sum())
    if "adc_clipped" in values:
        totals["adc_clipped_pixels"] = int(values["adc_clipped"].sum())
    return totals
