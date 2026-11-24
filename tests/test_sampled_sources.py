"""Sampled sources: an arbitrary intensity pattern (SAMPLED_INTENSITY), a sampled complex field (SAMPLED_FIELD) and a
speckle field (SPECKLE), traced through a Project and propagated by ``results.field``, against independent references.

References, each computed here from the stated geometry, independently of the Engine:

* a uniform map's ray positions are the uniform rectangle's (the same Sobol coordinates through the inverse distribution);
* a two-level map sends each half the share of its integral; a ramp map's mean position is the ramp's centroid (2/3 of the
  way up); a uniform cone's rays fill projected solid angle (the share within theta is sin^2(theta) / sin^2(theta_c));
* a Gaussian map's sampled irradiance equals the exact cell integrals of 2P / (pi w^2) exp(-2 r^2 / w^2), and the Engine's own
  Gaussian beam source's irradiance at the same plane;
* a linear phase 2 pi x / Lambda turns every ray of a field by the grating equation, sin(theta) = lambda / Lambda; the field's
  angular-spectrum centroid at z is z tan(theta), the paraxial transfer's z sin(theta);
* a Gaussian field focused by a lens has the 1/e^2 radius and peak irradiance of the Engine's Gaussian beam source traced through
  the same lens (its q-parameter propagation, a separate implementation from the first-order matrix the field uses);
* fully developed speckle has contrast 1; the average of M independent patterns has contrast 1 / sqrt(M);
* a Gaussian waist through a plano-convex lens (``_pcx_reference``): an exact real-ray trace of its wavefront normals, the
  exit-face field by the ray mapping and the Fresnel-Hankel integral, beside the same field with the first-order lens's phase;
* an uncoated window passes ``(1 - R)^2`` of a field that its clear aperture does not cut, wherever the field lies on its grid.

Tolerances were set before the first run from each estimate's expected error (stated at each test).
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib
import math

import numpy as np
import pytest
from local_helpers import ALONG_X, PCX_F100, catalog_document, element, plane, project, run as _fixture_run, source

from os_optics.errors import ErrorCode
from os_optics.local.errors import LocalRunError
from os_optics.optics.fields.source_fields import (
    PatternG,
    SourceGrid,
    sample_positions,
    speckle_screen,
)
from os_optics.optics.rays.emitter_samples import local_samples, sobol_points
from os_optics.results import load_run, pupil_field
from os_optics.results.field import FieldError, field_npz
from os_optics.scene.model import EmitterG

def run(document, **kwargs):
    """Fixture lenses/windows are explicitly ideal lossless for sampled-field comparisons."""
    fixture = catalog_document()
    fixture["schema_version"] = "3.6"  # Required for material bulk-loss evidence.
    for material in fixture["materials"]:
        material["bulk_loss"] = {"kind": "INTERNAL_TRANSMITTANCE",
                                 "wavelength_nm": [material["valid_band_nm"]["min_nm"], material["valid_band_nm"]["max_nm"]],
                                 "internal_transmittance": ["1", "1"], "thickness_mm": ["1", "1"],
                                 "basis": "Assumed lossless synthetic test fixture."}
    return _fixture_run(document, catalog=fixture, **kwargs)


LAM_NM = 633.0
LAM_MM = LAM_NM * 1e-6


def _component(model: dict, instance_id: str = "S") -> dict:
    return {"instance_id": instance_id, "optical_role": "source", "custom_part": {"custom_part_key": "sampled", "description": "Sampled source."},
            "nominal_pose": {"position_mm": [0.0, 0.0, 0.0], "quaternion_wxyz": list(ALONG_X)}, "source_model": model}


def _axis(count: int, pitch: float) -> np.ndarray:
    return (np.arange(count) - 0.5 * (count - 1)) * pitch


def _gaussian_rows(count: int, pitch: float, radius: float, *, amplitude: bool = False) -> list:
    x = _axis(count, pitch)
    xx, yy = np.meshgrid(x, x)
    power_shape = np.exp(-2.0 * (xx ** 2 + yy ** 2) / radius ** 2)
    return (np.sqrt(power_shape) if amplitude else power_shape).tolist()


def _intensity(values: list, pitch: float, *, law: str = "COLLIMATED", interpolation: str = "PIECEWISE_CONSTANT", rays: int = 4096,
               power_w: float = 0.001, **extra) -> dict:
    model = {"model_kind": "SAMPLED_INTENSITY", "wavelength_nm": LAM_NM, "power_w": power_w,
             "intensity_map": {"grid": {"column_count": len(values[0]), "row_count": len(values), "pitch_mm": [pitch, pitch]},
                               "values": values, "unit": "RELATIVE", "interpolation": interpolation},
             "angular_law": law, "sampling": {"sample_count": rays, "seed": 5}}
    model.update(extra)
    return model


def _field(amplitude: list, phase: list, pitch: float, *, rays: int = 1024, power_w: float = 0.001, wavelength_nm: float = LAM_NM) -> dict:
    return {"model_kind": "SAMPLED_FIELD", "wavelength_nm": wavelength_nm, "power_w": power_w,
            "grid": {"column_count": len(amplitude[0]), "row_count": len(amplitude), "pitch_mm": [pitch, pitch]},
            "amplitude": amplitude, "phase_rad": phase, "amplitude_unit": "RELATIVE", "polarization": {"kind": "LINEAR", "angle_deg": 0.0},
            "ray_sampling": {"sample_count": rays, "seed": 0}}


def _quantity(doc, name: str):
    return next(q for q in doc.quantities if q.quantity_id.endswith("." + name))


# ------------------------------------------------------------------------------------------------- sampling


def _pattern(values: np.ndarray, pitch: float, interpolation: str = "PIECEWISE_CONSTANT") -> PatternG:
    return PatternG(kind="map", grid=SourceGrid(values.shape[1], values.shape[0], pitch, pitch), interpolation=interpolation, weights=values)


def test_a_uniform_map_samples_the_uniform_rectangle():
    """The map's inverse distribution of a uniform map is the identity on each axis (rounding only)."""
    u = sobol_points(4096, 9)
    x, y = sample_positions(_pattern(np.ones((5, 8)), 0.25), u[:, 0], u[:, 1])
    assert np.max(np.abs(x - (u[:, 0] - 0.5) * 2.0)) < 1e-14 and np.max(np.abs(y - (u[:, 1] - 0.5) * 1.25)) < 1e-14


def test_a_two_level_map_shares_its_power_by_area_and_level():
    values = np.ones((4, 4))
    values[:, :2] = 3.0
    u = sobol_points(4096, 3)
    x, _y = sample_positions(_pattern(values, 1.0), u[:, 0], u[:, 1])
    # 3 / (3 + 1) of the power on x < 0; a stratified estimate of 4096 points is within 1 / 4096 per stratum boundary
    assert abs(float(np.mean(x < 0.0)) - 0.75) <= 2.0 / 4096


def test_a_bilinear_ramp_has_the_ramp_centroid():
    """Values rising linearly along x from 0: density proportional to (x - x0) between the outer points, mean at 2/3 of the span."""
    values = np.tile(np.arange(6, dtype=float), (3, 1))
    u = sobol_points(16384, 4)
    x, y = sample_positions(_pattern(values, 0.5, "BILINEAR"), u[:, 0], u[:, 1])
    x0, span = -1.25, 2.5
    assert float(x.min()) >= x0 - 1e-12 and float(x.max()) <= x0 + span + 1e-12 and float(np.abs(y).max()) <= 0.5 + 1e-12
    assert abs(float(x.mean()) - (x0 + 2.0 * span / 3.0)) < 2e-3  # (sampling error of 16384 stratified points: well below 1e-3 mm)
    # and the cumulative distribution (x - x0)^2 / span^2 at the quarter points
    for t in (0.25, 0.5, 0.75):
        assert abs(float(np.mean(x <= x0 + t * span)) - t * t) < 2e-3


def _emitter_g(law: str, **extra) -> EmitterG:
    z = np.array([0.0, 0.0, 1.0])
    return EmitterG("E", "E", np.zeros(3), z, np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), ((550.0, 1.0),), 1.0, "air",
                    "rectangle", 1.0, 1.0, law, 16384, 3, **extra)


def test_a_uniform_cone_fills_projected_solid_angle():
    half = 20.0
    sine_c = math.sin(math.radians(half))
    _p, d, _s = local_samples(_emitter_g("UNIFORM_CONE", cone_half_angle_deg=half, cone_sine=sine_c))
    sines = np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2)
    assert float(sines.max()) <= sine_c * (1 + 1e-12)
    for theta in (5.0, 10.0, 15.0):
        expected = math.sin(math.radians(theta)) ** 2 / sine_c ** 2
        assert abs(float(np.mean(sines <= math.sin(math.radians(theta)))) - expected) < 2e-3


def test_collimated_rays_leave_along_the_axis():
    _p, d, _s = local_samples(_emitter_g("COLLIMATED"))
    assert np.array_equal(d, np.tile([0.0, 0.0, 1.0], (16384, 1)))


# ------------------------------------------------------------------------------------------------- intensity map


def test_a_gaussian_map_reproduces_the_gaussian_irradiance():
    """A collimated Gaussian map of 1/e^2 radius w = 1 mm against (a) the exact cell integrals of the Gaussian and (b) the Engine's
    Gaussian beam source of the same waist at the same plane (Rayleigh range 4.96 m: unchanged 1 mm downstream). 65536 rays over
    a 16 x 16 map of 4 x 4 mm: a central cell holds about 2500 rays (plain Monte Carlo 2 %, stratified less): 3 % on every cell
    above a tenth of the peak."""
    w, power = 1.0, 0.001
    model = _intensity(_gaussian_rows(128, 0.04, w), 0.04, rays=65536, power_w=power)
    sampled = load_run(run(project([_component(model)], [plane("P", (1.0, 0.0, 0.0), radius_mm=10.0)])))
    beam = load_run(run(project([source("G", (0.0, 0.0, 0.0), wavelength_nm=LAM_NM, waist_mm=w, power_w=power)],
                                [plane("P", (1.0, 0.0, 0.0), radius_mm=10.0)])))
    a = sampled.irradiance_map("P", half_extent_mm=2.0, samples=16)
    # the Gaussian source's map at the histogram's bin centres: 16 points from -1.875 to 1.875 mm
    b = beam.irradiance_map("P", half_extent_mm=1.875, samples=16)
    hist = np.array(a["incoherent_W_per_mm2"])
    gauss = np.array(b["incoherent_W_per_mm2"])
    edges = np.array(a["grid"]["u_edges_mm"])
    # (a) the exact integral of the Gaussian over each cell, over the cell's area
    s = math.sqrt(2.0) / w
    share = 0.5 * np.diff([math.erf(s * e) for e in edges])
    exact = power * np.outer(share, share) / np.outer(np.diff(edges), np.diff(edges))
    lit = exact > 0.1 * exact.max()
    assert np.max(np.abs(hist[lit] / exact[lit] - 1.0)) < 0.03
    # (b) the Gaussian source's map at the cell centres against the closed form there, and against the sampled map
    centres = np.array(b["grid"]["u_mm"])
    assert np.allclose(centres, np.array(a["grid"]["u_mm"]), atol=1e-12) and np.allclose(b["grid"]["v_mm"], a["grid"]["v_mm"], atol=1e-12)
    point = 2.0 * power / (math.pi * w * w) * np.exp(-2.0 * (centres[:, None] ** 2 + centres[None, :] ** 2) / (w * w))
    assert np.max(np.abs(gauss - point)) < 1e-6 * point.max()  # (w(1 mm) / w0 - 1 = 2e-8)
    ratio = hist[lit] / gauss[lit]
    curvature = np.abs(exact[lit] / point[lit] - 1.0).max()  # (a cell's average against its centre value)
    assert curvature < 0.03 and np.max(np.abs(ratio - 1.0)) < 0.03 + curvature
    total = sampled.sampled_light("P")[0]
    assert abs(total["power_W"] - power) < 1e-15


def test_a_sampled_source_states_its_pattern_and_normalization():
    values = [[1.0, 2.0], [3.0, 4.0]]
    result = run(project([_component(_intensity(values, 0.5, law="LAMBERTIAN"))], [plane("P", (5.0, 0.0, 0.0))])).result
    line = result.sampled_sources[0]
    assert line.pattern == "INTENSITY_MAP" and line.angular_law == "LAMBERTIAN" and line.area_mm2 == pytest.approx(1.0)
    assert line.exitance_mean_W_per_mm2 == pytest.approx(0.001) and line.exitance_peak_W_per_mm2 == pytest.approx(0.001 * 4.0 / 2.5)
    assert line.radiance_W_per_m2_sr is None  # (the radiance follows the map)


# ------------------------------------------------------------------------------------------------- sampled field


def _tilted_field(period: float, *, n: int = 64, pitch: float = 0.01, w: float = 0.15) -> dict:
    x = _axis(n, pitch)
    xx, yy = np.meshgrid(x, x)
    return _field(np.exp(-(xx ** 2 + yy ** 2) / w ** 2).tolist(), (2.0 * math.pi * xx / period).tolist(), pitch)


def test_a_tilted_phase_steers_the_rays_by_the_grating_equation():
    period = 0.1
    result = run(project([_component(_tilted_field(period))], [plane("P", (20.0, 0.0, 0.0), radius_mm=5.0)])).result
    sine = LAM_MM / period
    directions = np.array([p.segments[0].end_mm for p in result.paths]) - np.array([p.segments[0].start_mm for p in result.paths])
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    transverse = np.sqrt(directions[:, 1] ** 2 + directions[:, 2] ** 2)
    assert np.max(np.abs(transverse - sine)) < 1e-12
    line = result.sampled_sources[0]
    assert line.pattern == "SAMPLED_FIELD" and line.angular_law == "FIELD_WAVEVECTOR" and line.domain_sr is None


def test_a_tilted_field_propagates_to_the_grating_angle():
    """Angular spectrum (exact scalar): centroid z tan(theta) to 1e-5; paraxial transfer: z sin(theta) to 1e-5."""
    period, z = 0.1, 20.0
    results = load_run(run(project([_component(_tilted_field(period))], [plane("P", (z, 0.0, 0.0), radius_mm=5.0)])))
    sine = LAM_MM / period
    asm = results.field("P")
    assert asm.arrays["method"] == "ANGULAR_SPECTRUM" and _stated(asm)
    u, v = _quantity(asm, "centroid_u_mm").value, _quantity(asm, "centroid_v_mm").value
    shift = math.hypot(u, v)
    assert abs(shift / (z * sine / math.sqrt(1.0 - sine * sine)) - 1.0) < 1e-5
    assert _quantity(asm, "power_window_W").value == pytest.approx(0.001, rel=1e-9)
    paraxial = results.field("P", {"method": "PARAXIAL_TRANSFER"})
    assert paraxial.arrays["method"] == "PARAXIAL_TRANSFER"
    shift_p = math.hypot(_quantity(paraxial, "centroid_u_mm").value, _quantity(paraxial, "centroid_v_mm").value)
    assert abs(shift_p / (z * sine) - 1.0) < 1e-5


def test_a_gaussian_field_through_a_lens_matches_the_gaussian_beam():
    """A Gaussian field of w0 = 1 mm (flat phase) through a 100 mm plano-convex lens, at a plane near its focus: the field's
    1/e^2 radius (twice its rms width) and peak irradiance against the Engine's Gaussian beam source of the same waist through the
    same lens. Both are first order; the grid samples the focal spot with about 3 points per radius: 1 %."""
    w0, power = 1.0, 0.001
    lens = element("L", "lens", PCX_F100, (10.0, 0.0, 0.0))
    observations = [plane("F", (106.0, 0.0, 0.0), radius_mm=5.0)]
    amplitude = _gaussian_rows(128, 0.04, w0, amplitude=True)
    phase = np.zeros((128, 128)).tolist()
    field_run = load_run(run(project([_component(_field(amplitude, phase, 0.04, wavelength_nm=780.0, power_w=power)), lens], observations)))
    beam_run = run(project([source("G", (0.0, 0.0, 0.0), wavelength_nm=780.0, waist_mm=w0, power_w=power), lens], observations))
    (arrival,) = [a for o in beam_run.result.observations for a in o.arrivals]
    g = arrival.beam.gaussian
    doc = field_run.field("F")
    assert doc.arrays["method"] == "PARAXIAL_TRANSFER" and _stated(doc), doc.models
    radius_u, radius_v = 2.0 * _quantity(doc, "rms_radius_u_mm").value, 2.0 * _quantity(doc, "rms_radius_v_mm").value
    assert radius_u == pytest.approx(g.beam_radius_x_mm, rel=0.01) and radius_v == pytest.approx(g.beam_radius_y_mm, rel=0.01)
    peak = 2.0 * arrival.beam.power_W / (math.pi * g.beam_radius_x_mm * g.beam_radius_y_mm)
    assert _quantity(doc, "peak_irradiance_W_per_mm2").value == pytest.approx(peak, rel=0.01)
    assert _quantity(doc, "power_window_W").value == pytest.approx(arrival.beam.power_W, rel=1e-6)


def test_a_field_at_a_plane_lights_a_pupil_field():
    """The field a run delivers to a plane (``--field-out``) as a pupil-field FIELD illumination: the focal spot of a Gaussian of
    w = 0.3 mm (on a grid of 4.3 w half width: no truncation) in front of a 100 mm ideal lens is lambda f / (pi w); peak 2P / (pi w_f^2)
    within 1 %."""
    w, power = 0.3, 0.001
    results = load_run(run(project([_component(_field(_gaussian_rows(128, 0.02, w, amplitude=True), np.zeros((128, 128)).tolist(), 0.02,
                                                      power_w=power))], [plane("P", (5.0, 0.0, 0.0), radius_mm=5.0)])))
    doc = results.field("P", {"include_field": True, "padding": 1, "refinement": False})
    data = field_npz(doc)
    count = len(doc.arrays["field_u_mm"])
    pitch = doc.arrays["field_u_mm"][1] - doc.arrays["field_u_mm"][0]
    request = {"wavelength_nm": LAM_NM, "focal_length_mm": 100.0, "sampling": {"pixel_count": [count, count], "pitch_mm": [pitch, pitch]},
               "illumination": {"kind": "FIELD", "file_name": "f.npz", "sha256": hashlib.sha256(data).hexdigest()},
               "output": {"half_extent_mm": [0.3, 0.3]}}
    reply = pupil_field(request, {"f.npz": data})
    assert reply["status"] == "OK", reply
    wf = LAM_MM * 100.0 / (math.pi * w)
    peak = max(max(row) for row in reply["coherent_W_per_mm2"])
    assert peak == pytest.approx(2.0 * power / (math.pi * wf * wf), rel=0.01)


def _untraced_field(result, code: str):
    """The issue of a sampled field the run did not trace, after checking what that means for the run: the run completed, no
    ray of the source was launched, its whole power is booked as not traced and its plane's power is not a known value."""
    engine = result.result
    (issue,) = [i for i in engine.issues if i.code.value == code]
    assert issue.status.value == "OUT_OF_MODEL_SCOPE" and issue.location.endswith("source:S")
    assert engine.outcome.execution.outcome.value == "COMPLETED"
    (line,) = [entry for entry in engine.power_budget.lines if entry.source_id == "S"]
    assert line.sinks == {"SOURCE_NOT_TRACED": line.input_power_W} and line.input_power_W == 0.001
    plane_result = next(o for o in engine.observations if o.observation_id == "P")
    assert plane_result.total_power_W is None and plane_result.quantity_state.value == "UNKNOWN"
    # (the plane lists every sampled line: this one with no ray launched and none counted)
    assert all(line.samples_traced == 0 and not line.counted_crossing_count for line in plane_result.sampled_light)
    return issue


def test_an_undersampled_or_evanescent_field_is_not_traced_and_the_run_completes():
    n, pitch = 16, 0.01
    steep = (1.8 * np.add.outer(np.zeros(n), np.arange(n))).tolist()  # 1.8 rad per grid point along x
    flat = np.ones((n, n)).tolist()
    issue = _untraced_field(run(project([_component(_field(flat, steep, pitch))], [plane("P", (5.0, 0.0, 0.0))])), "SOURCE_FIELD_UNDERSAMPLED")
    assert "1.8 rad" in issue.message and issue.numerical_context["largest_phase_step_rad"] == pytest.approx(1.8, abs=1e-12)
    tiny = 0.0002
    diagonal = (1.5 * np.add.outer(np.arange(n), np.arange(n))).tolist()  # |grad| = 1.5 sqrt(2) / 0.2 um > 2 pi / 633 nm
    issue = _untraced_field(run(project([_component(_field(flat, diagonal, tiny))], [plane("P", (5.0, 0.0, 0.0))])), "SOURCE_FIELD_EVANESCENT")
    # 1.5 sqrt 2 rad per 0.2 um against k = 2 pi / 633 nm (in vacuum; air's index differs by 3e-4)
    assert issue.numerical_context["largest_transverse_sine"] == pytest.approx(1.5 * math.sqrt(2.0) / tiny / (2.0 * math.pi / LAM_MM), rel=1e-3)
    assert ErrorCode.SOURCE_FIELD_UNDERSAMPLED.value == "SOURCE_FIELD_UNDERSAMPLED"


def test_a_field_that_is_not_traced_leaves_the_other_source_s_values_as_they_are():
    """A Gaussian source 50 mm beside an undersampled field. The plane in front of the field reads UNKNOWN -- never a known
    zero, never a known power; the plane only the Gaussian beam reaches holds, bit for bit, the beam and the power of the
    run without the field, and says that light was not followed (the field's 1 mW is of unknown fate)."""
    n, pitch = 16, 0.01
    steep = (1.8 * np.add.outer(np.zeros(n), np.arange(n))).tolist()
    planes = [plane("P", (5.0, 0.0, 0.0)), plane("only", (20.0, 50.0, 0.0), radius_mm=2.0)]
    gaussian = source("G", (0.0, 50.0, 0.0), wavelength_nm=780.0)
    both = run(project([_component(_field(np.ones((n, n)).tolist(), steep, pitch)), gaussian], planes))
    alone = run(project([gaussian], planes))
    _untraced_field(both, "SOURCE_FIELD_UNDERSAMPLED")
    lit, reference = (next(o for o in r.result.observations if o.observation_id == "only") for r in (both, alone))
    assert [a.model_dump() for a in lit.arrivals] == [a.model_dump() for a in reference.arrivals] and len(lit.arrivals) == 1
    assert lit.total_power_W == reference.total_power_W == 0.001
    # (lit, with light of unknown fate that may add: BOUNDED from its traced value, the upper end open)
    assert reference.quantity_state.value == "KNOWN" and lit.quantity_state.value == "BOUNDED" and not lit.power_complete
    assert lit.power_upper_bound_W is None
    assert lit.untraced_power_W >= 0.001
    with pytest.raises(FieldError) as refused:  # (the field analysis propagates no field the run did not trace)
        load_run(both).field("P", {})
    assert refused.value.code == "SOURCE_FIELD_UNDERSAMPLED"


# ------------------------------------------------------------------------------------------------- speckle


def _speckle(realizations: int, *, seed: int = 3) -> dict:
    return {"model_kind": "SPECKLE", "wavelength_nm": LAM_NM, "power_w": 0.001, "polarization": {"kind": "LINEAR", "angle_deg": 0.0},
            "illumination": {"shape": "GAUSSIAN", "radius_mm": 2.0},
            "phase_screen": {"statistics": "GAUSSIAN_CORRELATED", "phase_rms_rad": 3.0, "correlation_length_mm": 0.2, "seed": seed},
            "grid": {"column_count": 512, "row_count": 512, "pitch_mm": [0.01, 0.01]}, "realizations": realizations,
            "ray_sampling": {"sample_count": 4096, "seed": 0}}


def test_a_speckle_screen_is_reproducible_and_has_its_statistics():
    grid = SourceGrid(512, 512, 0.01, 0.01)
    screen = speckle_screen(grid, 3.0, 0.2, 7)
    assert screen.tobytes() == speckle_screen(grid, 3.0, 0.2, 7).tobytes()
    assert not np.array_equal(screen, speckle_screen(grid, 3.0, 0.2, 8))
    # rms phase 3 rad: over 512^2 points of about (5.12 / 0.2)^2 = 655 correlation cells, the sample rms is within 5 %
    assert float(screen.std()) == pytest.approx(3.0, rel=0.05)
    # autocorrelation exp(-r^2 / lc^2): at one correlation length (20 points) 1/e, within 0.05
    centred = screen - screen.mean()
    shifted = float((centred[:, 20:] * centred[:, :-20]).mean()) / float((centred * centred).mean())
    assert shifted == pytest.approx(math.exp(-1.0), abs=0.05)


def test_fully_developed_speckle_and_its_average_over_realizations():
    """A strong diffuser (rms phase 3 rad, correlation length 0.2 mm) lit by a 2 mm Gaussian, seen 1 m away through free space
    (paraxial transfer: the angular spectrum's window does not hold the spread light there). Over the region of +-1 mm, where the
    mean irradiance (rms width about 2.3 mm) varies by under 2 %, about 200 speckle grains of lambda z / D = 0.16 mm: one
    pattern's contrast is 1 within its sampling error (7 %, mean of 16: 2 %); 0.1. The average of 16 independent patterns has
    contrast 1 / 4 (0.05), and mean(C_m) / 4 (0.04)."""
    results = load_run(run(project([_component(_speckle(16))], [plane("P", (1000.0, 0.0, 0.0), radius_mm=50.0)])))
    doc = results.field("P", {"region": {"half_width_mm": [1.0, 1.0]}, "refinement": False})
    assert doc.arrays["method"] == "PARAXIAL_TRANSFER" and _stated(doc)
    contrast = _quantity(doc, "contrast")
    assert contrast.name == "speckle contrast of one pattern" and abs(contrast.value - 1.0) < 0.1
    assert contrast.convergence.method.value == "HALF_SAMPLE" and contrast.convergence.difference < 0.05
    averaged = _quantity(doc, "contrast_averaged").value
    assert abs(averaged - 0.25) < 0.05 and abs(averaged - _quantity(doc, "contrast_averaged_independent").value) < 0.04
    assert _quantity(doc, "power_window_W").value == pytest.approx(0.001, rel=1e-6)
    assert len(doc.arrays["contrast_per_realization"]) == 16
    # the run's rays: the geometric transport of the diffused light reaches the plane whole
    assert _quantity(doc, "geometric_power_W").value == pytest.approx(0.001, rel=1e-12)


# ------------------------------------------------------------------------------------------------- review regressions
#
# Each case reproduces a finding of the independent review against a closed form or a separate computation; before the fix
# each one was reported complete with a wrong value or passed a non-physical ray to the tracer.


def _gaussian_field(count: int, pitch: float, w: float, *, x0: float = 0.0, sine: float = 0.0, power_w: float = 0.001,
                    rays: int = 1024, lam_nm: float = LAM_NM) -> dict:
    """``exp(-((x - x0)^2 + y^2) / w^2)`` with the linear phase of a tilt of ``sine`` toward +x (in air at ``lam_nm``, 633 nm)."""
    x = _axis(count, pitch)
    xx, yy = np.meshgrid(x, x)
    return _field(np.exp(-((xx - x0) ** 2 + yy ** 2) / w ** 2).tolist(), (2.0 * math.pi * sine * xx / (lam_nm * 1e-6)).tolist(), pitch,
                  rays=rays, power_w=power_w, wavelength_nm=lam_nm)


def _check(doc, parameter: str):
    return next((c for m in doc.models for c in (m.validity or ()) if c.parameter == parameter), None)


def _stated(doc) -> bool:
    """The model's conditions hold on the plane: no quantity of the document is outside its model. (The execution record
    says only that the analysis ran to its end, which a field analysis always does.)"""

    assert doc.execution.complete and doc.execution.outcome.value == "COMPLETED"
    return all(q.state.value != "OUT_OF_MODEL_SCOPE" for q in doc.quantities)


def _value(doc, name: str):
    q = _quantity(doc, name)
    return q.value if q.state.value == "KNOWN" else (q.state.value, q.reason)


def test_light_that_leaves_the_window_is_not_reported():
    """SRC-1. A tilted Gaussian spot off the grid's centre (w 0.08 mm at x0 0.45 mm, sine 0.012) 80 mm away: its light lands at
    x0 + z tan(theta) = 1.4101 mm, beyond the angular spectrum's window (+-1.28 mm at padding 2), and folded back to -0.855 mm
    while the window still held all the power (the old condition). The window twice as wide holds it outside the first: refused
    (WINDOW_LOSES_POWER); AUTO takes the paraxial transfer, whose centroid is x0 + z sin(theta) (z (tan - sin) = 7e-5 from the
    true one): 2e-4. Padding 1 at 20 mm (w 0.06 at x0 0.40): refused by both methods; padding 2 lands at 0.64 mm (1e-4)."""
    z, x0, s = 80.0, 0.45, 0.012
    results = load_run(run(project([_component(_gaussian_field(128, 0.01, 0.08, x0=x0, sine=s, rays=2048))],
                                   [plane("P", (z, 0.0, 0.0), radius_mm=10.0)])))
    asm = results.field("P", {"method": "ANGULAR_SPECTRUM"})
    assert asm.execution.complete and not _stated(asm) and _quantity(asm, "centroid_u_mm").reason == "WINDOW_LOSES_POWER"
    assert _check(asm, "window_power_fraction").within is False and _check(asm, "window_power_fraction").observed < 0.5
    auto = results.field("P")
    assert auto.arrays["method"] == "PARAXIAL_TRANSFER" and _stated(auto)
    true = x0 + z * s / math.sqrt(1.0 - s * s)
    assert abs(_value(auto, "centroid_u_mm") - true) < 2e-4
    z, x0 = 20.0, 0.40
    results = load_run(run(project([_component(_gaussian_field(128, 0.01, 0.06, x0=x0, sine=s))], [plane("P", (z, 0.0, 0.0), radius_mm=10.0)])))
    for method in ("ANGULAR_SPECTRUM", "PARAXIAL_TRANSFER"):
        narrow = results.field("P", {"method": method, "padding": 1, "refinement": False})
        assert narrow.execution.complete and not _stated(narrow) and _quantity(narrow, "centroid_u_mm").reason == "WINDOW_LOSES_POWER", method
        wide = results.field("P", {"method": method, "padding": 2, "refinement": False})
        assert _stated(wide) and abs(_value(wide, "centroid_u_mm") - (x0 + z * s / math.sqrt(1.0 - s * s))) < 1e-4, method


def test_a_band_limited_window_that_changes_the_peak_is_not_reported():
    """SRC-2. A Gaussian field of w0 0.15 mm (SQRT_W_PER_MM2, 5 mW on the grid) at 5 Rayleigh ranges: the angular spectrum's band
    limit at padding 2 cuts the spectrum where its amplitude is still 5.6 % and its peak came out 3.8 % high with every condition
    met. The window twice as wide changes it by that much: refused (WINDOW_NOT_CONVERGED); AUTO's paraxial transfer and, at 0.5
    and 1 Rayleigh range, the angular spectrum give the Gaussian beam's peak 2 P / (pi w(z)^2) at the grid point nearest the axis
    (p / 2 off in each axis) and 1/e^2 radius w(z) to 1e-3, each with its window refinement as a convergence record."""
    count, pitch, w0, power = 128, 0.01, 0.15, 0.005
    x = _axis(count, pitch)
    xx, yy = np.meshgrid(x, x)
    amplitude = np.sqrt(2.0 * power / (math.pi * w0 ** 2)) * np.exp(-(xx ** 2 + yy ** 2) / w0 ** 2)
    grid_power = float((amplitude ** 2).sum()) * pitch * pitch
    model = {"model_kind": "SAMPLED_FIELD", "wavelength_nm": LAM_NM, "grid": {"column_count": count, "row_count": count, "pitch_mm": [pitch, pitch]},
             "amplitude": amplitude.tolist(), "phase_rad": np.zeros((count, count)).tolist(), "amplitude_unit": "SQRT_W_PER_MM2",
             "polarization": {"kind": "LINEAR", "angle_deg": 0.0}, "ray_sampling": {"sample_count": 1024, "seed": 0}}
    rayleigh = math.pi * w0 ** 2 / LAM_MM
    for factor, method in ((5.0, "PARAXIAL_TRANSFER"), (0.5, "ANGULAR_SPECTRUM"), (1.0, "ANGULAR_SPECTRUM")):
        results = load_run(run(project([_component(model)], [plane("P", (factor * rayleigh, 0.0, 0.0), radius_mm=10.0)])))
        if factor == 5.0:
            asm = results.field("P", {"method": "ANGULAR_SPECTRUM"})
            change = _check(asm, "window_refinement_change")
            assert asm.execution.complete and not _stated(asm) and _quantity(asm, "peak_irradiance_W_per_mm2").reason == "WINDOW_NOT_CONVERGED"
            assert change.within is False and change.observed == pytest.approx(0.038, abs=0.005)
        doc = results.field("P")
        assert doc.arrays["method"] == method and _stated(doc), (factor, doc.models)
        w = w0 * math.sqrt(1.0 + factor ** 2)
        peak = 2.0 * grid_power / (math.pi * w * w) * math.exp(-4.0 * (0.5 * pitch) ** 2 / (w * w))
        assert _value(doc, "peak_irradiance_W_per_mm2") == pytest.approx(peak, rel=1e-3)
        assert 2.0 * _value(doc, "rms_radius_u_mm") == pytest.approx(w, rel=1e-3)
        for name in ("peak_irradiance_W_per_mm2", "centroid_u_mm", "rms_radius_u_mm", "power_region_W"):
            record = _quantity(doc, name).convergence
            assert record is not None and record.method.value == "REFINEMENT", name
        assert _quantity(doc, "peak_irradiance_W_per_mm2").convergence.difference < 1e-3 * peak


def test_apertures_that_cut_the_field_are_detected():
    """SRC-3. A uniform disc of 12 mm radius through a lens of 11.43 mm clear radius: 0.9087 of its area lies inside (counted on
    the grid). The probe beam that gives the axis's transmission was itself cut (0.829), so the rays' power over it was 1.0818 and
    passed the old one-sided check. Now the launched rays that reach the plane along the axis's path are counted (their share is
    the area share, to the sampling error of 4096 rays: 0.015) and the rays' power over the probe's must lie within 0.1 %:
    refused (APERTURES_CUT_FIELD). A 1 mm Gaussian well inside the aperture passes both."""
    count, pitch, radius, clear = 128, 0.2, 12.0, 11.43
    x = _axis(count, pitch)
    xx, yy = np.meshgrid(x, x)
    disc = (xx ** 2 + yy ** 2 <= radius * radius).astype(float)
    inside = float(((xx ** 2 + yy ** 2 <= clear * clear) * disc).sum() / disc.sum())
    lens = element("L", "lens", PCX_F100, (10.0, 0.0, 0.0))
    results = load_run(run(project([_component(_field(disc.tolist(), np.zeros((count, count)).tolist(), pitch, rays=4096)), lens],
                                   [plane("P", (100.0, 0.0, 0.0), radius_mm=4.0)])))
    doc = results.field("P", {"refinement": False})
    assert doc.execution.complete and not _stated(doc) and _quantity(doc, "peak_irradiance_W_per_mm2").reason == "APERTURES_CUT_FIELD"
    assert _check(doc, "aperture_pass_share").observed == pytest.approx(inside, abs=0.015)
    # the rays' power over the axis's transmission is the same share: the probe beam's own cut at the lens rim (its Gaussian
    # tail, 17 % of it) no longer enters the axis's transmission, so the figure is the field's loss and not the probe's
    assert _check(doc, "geometric_transmission").within is False
    assert _check(doc, "geometric_transmission").observed == pytest.approx(_check(doc, "aperture_pass_share").observed, abs=5e-3)
    # every failed condition is named on each withheld value, not only the first
    failed = tuple(c.parameter for m in doc.models for c in m.validity if c.within is False)
    assert _quantity(doc, "peak_irradiance_W_per_mm2").depends_on == failed and failed[:2] == ("aperture_pass_share", "geometric_transmission")
    assert all(q.value is None and q.value_basis is None for q in doc.quantities
               if q.state.value == "OUT_OF_MODEL_SCOPE" and q.quantity_id != "S.geometric_power_W")
    # (the disc's rim meets the lens's coating beyond its published angles, and the run's finding lists the field's line: the
    # rays' power keeps its number with that finding's ESTIMATE)
    rays = _quantity(doc, "geometric_power_W")
    assert (rays.state.value, rays.reason) == ("OUT_OF_MODEL_SCOPE", "COATING_AOI_OUTSIDE_RANGE") and rays.value is not None
    assert [(c.code, c.parameter) for c in rays.value_basis.validity] == [("COATING_AOI_OUTSIDE_RANGE", "angle_of_incidence_deg")]
    gaussian = load_run(run(project([_component(_gaussian_field(128, 0.05, 1.0, rays=4096)), lens], [plane("P", (60.0, 0.0, 0.0), radius_mm=10.0)])))
    doc = gaussian.field("P", {"refinement": False})
    assert _stated(doc)
    assert _check(doc, "aperture_pass_share").observed == 1.0
    assert _check(doc, "geometric_transmission").observed == pytest.approx(1.0, abs=1e-3)


def _exit_vertex(sku: str) -> float:
    from local_helpers import product_document

    return float(product_document(sku)["simulation_models"][0]["surfaces"][-1]["vertex_z_mm"])


def _back_focus_x(sku: str, lens_x: float, lam_nm: float = LAM_NM) -> float:
    from local_helpers import lens_matrix, product_document

    m = lens_matrix(sku, lam_nm)
    exit_vertex = float(product_document(sku)["simulation_models"][0]["surfaces"][-1]["vertex_z_mm"])
    return lens_x + exit_vertex - m[0, 0] / m[1, 0]


def _pcx_trace(sku: str, w0: float, lens_x: float, lam_nm: float = LAM_NM) -> dict:
    """An exact real-ray trace of a Gaussian waist ``w0`` (flat phase, at the source) through a plano-convex lens, curved side
    first, its vertex ``lens_x`` from the source (no Engine code): the Gaussian beam's q to a plane 1 mm before the vertex
    (outside the sag); rays along its wavefront normals there, traced by vector Snell through the sphere (the product's radius,
    N-BK7 by the product document's Sellmeier) and the flat exit face. Returns the rays' start heights ``h``, the beam radius
    ``w`` and wavefront radius ``bend`` there, exit heights, optical paths from that plane, exit direction sines, and the
    first-order exit wavefront's radius (the paraxial ray's through the same surfaces; negative: converging)."""
    from local_helpers import product_document, sellmeier_index

    model = product_document(sku)["simulation_models"][0]
    curvature, thickness = float(model["surfaces"][0]["radius_mm"]), float(model["surfaces"][1]["vertex_z_mm"])
    n = sellmeier_index(model["media"][0], lam_nm)
    lam_mm = lam_nm * 1e-6
    z0 = lens_x - 1.0
    q = complex(z0, math.pi * w0 * w0 / lam_mm)
    w, bend = math.sqrt(-lam_mm / (math.pi * (1.0 / q).imag)), 1.0 / (1.0 / q).real
    h = np.linspace(0.0, 3.5 * w, 100001)
    tilt = np.arctan(h / bend)
    d = np.column_stack([np.sin(tilt), np.cos(tilt)])  # (r, z)
    p = np.column_stack([h, np.full(h.size, z0)])

    def refract(d, normal, n1, n2):  # (normal against the incidence)
        cos_i = -(d * normal).sum(1)
        eta = n1 / n2
        return eta * d + (eta * cos_i - np.sqrt(1.0 - eta * eta * (1.0 - cos_i * cos_i)))[:, None] * normal

    centre = np.array([0.0, lens_x + curvature])
    oc = p - centre
    b = (oc * d).sum(1)
    s1 = -b - np.sqrt(b * b - ((oc * oc).sum(1) - curvature * curvature))
    p1 = p + s1[:, None] * d
    d1 = refract(d, (p1 - centre) / curvature, 1.0, n)
    s2 = (lens_x + thickness - p1[:, 1]) / d1[:, 1]
    lens = np.array([[1.0, thickness / n], [0.0, 1.0]]) @ np.array([[1.0, 0.0], [-(n - 1.0) / curvature, 1.0]])
    height, slope = lens @ np.array([1.0 + 1.0 / bend, 1.0 / bend])  # (the paraxial ray of the wavefront, 1 mm to the vertex)
    return {"h": h, "w": w, "bend": bend, "lens": lens, "landing": p1[:, 0] + s2 * d1[:, 0],
            "path": s1 + n * s2 + (np.sqrt(h * h + bend * bend) - bend), "sine": n * d1[:, 0], "sphere": height / slope}


def _pcx_reference(sku: str, w0: float, lens_x: float, distances: list[float], radius: np.ndarray, lam_nm: float = LAM_NM
                   ) -> list[tuple[np.ndarray, np.ndarray]]:
    """An independent diffraction reference (``_pcx_trace``; no Engine code): the exit-face field by the ray mapping (power
    ``h dh -> r dr``, phase ``k0`` times the optical path) and the Fresnel-Hankel integral to each distance after the exit face.
    Each distance gives ``(truth, first_order)`` irradiances at ``radius``: the real exit phase, and the same amplitude with the
    first-order lens's spherical exit phase (aberration-free, by the same transport, so their difference is the aberration's
    alone)."""
    from scipy.special import j0

    trace = _pcx_trace(sku, w0, lens_x, lam_nm)
    h, w, landing, path, sphere = trace["h"], trace["w"], trace["landing"], trace["path"], trace["sphere"]
    k = 2.0 * math.pi / (lam_nm * 1e-6)
    assert np.all(np.diff(landing) > 0.0)
    step = w / 5000.0
    r = np.arange(0.0, landing[-1], step)
    source_h = np.interp(r, landing, h)
    stretch = np.interp(r, landing, np.gradient(landing, h))
    ratio = np.where(r > 0.0, source_h / np.maximum(r, 1e-300) / stretch, 1.0 / stretch[0])
    amplitude = np.exp(-source_h ** 2 / (w * w)) * np.sqrt(ratio)
    # the Fresnel integral focuses a paraxial (quadratic) sphere perfectly, not an exact one: the exit phase is taken as the
    # quadratic sphere plus the aberration (the exact optical path less the exact sphere's)
    exact_sphere = np.sign(sphere) * (np.sqrt(r * r + sphere * sphere) - abs(sphere))
    first = amplitude * np.exp(1j * k * r * r / (2.0 * sphere))
    exact = first * np.exp(1j * k * ((np.interp(r, landing, path) - path[0]) - exact_sphere))
    weights = np.full(r.size, step) * r
    weights[-1] *= 0.5
    out = []
    for z in distances:
        kernel = j0(k * radius[:, None] * r[None, :] / z) * np.exp(1j * k * r * r / (2.0 * z))[None, :]
        out.append(tuple(np.abs(kernel @ (field * weights)) ** 2 for field in (exact, first)))
    return out


def _profile_values(irradiance: np.ndarray, u: np.ndarray, v: np.ndarray) -> tuple[float, float]:
    """Peak over power and rms width along u of an irradiance on the grid ``[v][u]``."""
    total = float(irradiance.sum())
    centroid = float((irradiance.sum(axis=0) * u).sum()) / total
    return float(irradiance.max()) / total, math.sqrt(float((irradiance.sum(axis=0) * (u - centroid) ** 2).sum()) / total)


def _on_grid(radial: np.ndarray, radius: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    uu, vv = np.meshgrid(u, v)
    return np.interp(np.hypot(uu, vv), radius, radial, right=0.0)


def _errors(doc, truth: np.ndarray, radius: np.ndarray, u: np.ndarray | None = None, v: np.ndarray | None = None,
            computed: np.ndarray | None = None) -> tuple[float, float]:
    """Relative errors of peak over power and of rms width of ``computed`` (default: the document's field) against ``truth``."""
    if computed is None:
        u, v = np.array(doc.arrays["field_u_mm"]), np.array(doc.arrays["field_v_mm"])
        computed = np.abs(np.array(doc.arrays["field_re"]) + 1j * np.array(doc.arrays["field_im"])) ** 2
    peak, width = _profile_values(computed, u, v)
    true_peak, true_width = _profile_values(_on_grid(truth, radius, u, v), u, v)
    return peak / true_peak - 1.0, width / true_width - 1.0


def _gaussian_waist_rayleigh(w0: float, lens_x: float, sku: str, lam_nm: float = LAM_NM) -> tuple[float, float]:
    """The image waist of the Gaussian ``w0`` behind the lens (from its exit face) and its Rayleigh range (q propagation)."""
    from local_helpers import lens_matrix

    m = lens_matrix(sku, lam_nm)
    q = complex(lens_x, math.pi * w0 * w0 / (lam_nm * 1e-6))
    q = (m[0, 0] * q + m[0, 1]) / (m[1, 0] * q + m[1, 1])
    waist = -q.real
    return waist, q.imag


def test_the_aberration_check_measures_the_error_it_admits():
    """SRC3-1. The rays' aberration beyond first order, fitted as a phase where it is one, is applied to the field and propagated
    by the same transfer: the plane's values may change by at most 1 %. Reference: ``_pcx_reference`` (exact ray trace, ray-mapped
    exit field, Fresnel-Hankel integral; its first-order twin is the Collins model's field).

    (a) A Gaussian of w 1.25 mm through a 50 mm plano-convex lens, 0.3 mm before the paraxial focus: the old bound (0.091) accepted
    it with the peak 12.7 % below the truth. The check's change is the reference's error of the first-order model (counted
    ``ABERRATION_PROFILE_MARGIN`` times, to 15 %:
    the fit's phase against the exact optical path, the field's ray mapping): refused, ABERRATED_BEYOND_FIRST_ORDER. At the focus
    (the error is the Strehl ratio's, 0.958: 4.4 %), likewise.
    (b) The same lens with w 0.5 mm at its focus, 0.5 and 1.5 Rayleigh ranges either side: complete, peak and rms width within 1 %
    of the reference (the window and grid add 1e-3). Half a Rayleigh range from the waist the reference's peak is 0.7 % off the
    first-order value while the fit predicts 0.35 %: the profile's change counts twice (E4-4).
    (c) A small waist (w0 20 um, flat phase) 100 mm before the lens, imaged at about 1:1: its own rays all leave parallel, only its
    diffraction fills the lens (1 mm), where the old check saw no aberration and accepted a peak 11 % from the truth half a
    Rayleigh range from the image waist. The check's rays carry the field's diffraction spread: refused, the change equal to the
    reference's error to 15 %.
    (d) E4-4: 1064 nm, w 0.7 mm, 0.6 Rayleigh ranges (1 mm) either side of the image waist: the fit predicts a 0.78 % change of the
    peak, the reference shows about 1.1 % against the first-order Gaussian beam (q through the paraxial matrix), beyond the 1 % the
    check states. It was accepted; counted twice, the change is refused (ABERRATED_BEYOND_FIRST_ORDER). Its waist is complete."""
    from local_helpers import PCX_F50

    from os_optics.results.field import ABERRATION_PROFILE_MARGIN

    lens_x = 10.0
    lens = element("L", "lens", PCX_F50, (lens_x, 0.0, 0.0))
    exit_x = lens_x + _exit_vertex(PCX_F50)
    count, w = 256, 1.25
    focus = _back_focus_x(PCX_F50, lens_x) - exit_x
    distances = [focus - 0.3, focus]
    results = load_run(run(project([_component(_gaussian_field(count, 8.4 * w / count, w, rays=4096)), lens],
                                   [plane(f"P{i}", (exit_x + z, 0.0, 0.0), radius_mm=12.0) for i, z in enumerate(distances)])))
    radius = np.arange(0.0, 0.12, 0.0002)
    reference = _pcx_reference(PCX_F50, w, lens_x, distances, radius)
    for i, (truth, first) in enumerate(reference):
        doc = results.field(f"P{i}", {"refinement": False})
        assert doc.execution.complete and not _stated(doc) and _quantity(doc, "peak_irradiance_W_per_mm2").reason == "ABERRATED_BEYOND_FIRST_ORDER"
        u = v = np.arange(-0.06, 0.06, 0.0015)
        error = max(abs(e) for e in _errors(None, truth, radius, u, v, _on_grid(first, radius, u, v)))
        assert error > 0.04, (i, error)
        assert _check(doc, "aberration_value_change").observed == pytest.approx(ABERRATION_PROFILE_MARGIN * error, rel=0.15), i
    w = 0.5
    waist, rayleigh = _gaussian_waist_rayleigh(w, lens_x, PCX_F50)
    distances = [waist - 1.5 * rayleigh, waist - 0.5 * rayleigh, waist, waist + 0.5 * rayleigh, waist + 1.5 * rayleigh]
    results = load_run(run(project([_component(_gaussian_field(128, 8.4 * w / 128, w, rays=4096)), lens],
                                   [plane(f"P{i}", (exit_x + z, 0.0, 0.0), radius_mm=12.0) for i, z in enumerate(distances)])))
    radius = np.arange(0.0, 0.25, 0.0002)
    for i, (truth, _first) in enumerate(_pcx_reference(PCX_F50, w, lens_x, distances, radius)):
        doc = results.field(f"P{i}", {"include_field": True, "refinement": False})
        assert _stated(doc), (i, doc.models)
        assert _check(doc, "aberration_value_change").observed < 0.01
        peak, width = _errors(doc, truth, radius)
        assert abs(peak) < 0.01 and abs(width) < 0.01, (i, peak, width)
    w0, distance = 0.02, 100.0
    lens = element("L", "lens", PCX_F50, (distance, 0.0, 0.0))
    waist, rayleigh = _gaussian_waist_rayleigh(w0, distance, PCX_F50)
    planes = [waist - 0.5 * rayleigh]
    results = load_run(run(project([_component(_gaussian_field(64, 12.8 * w0 / 64, w0, rays=4096)), lens],
                                   [plane("P0", (distance + _exit_vertex(PCX_F50) + planes[0], 0.0, 0.0), radius_mm=12.0)])))
    radius = np.arange(0.0, 0.12, 0.0002)
    ((truth, first),) = _pcx_reference(PCX_F50, w0, distance, planes, radius)
    doc = results.field("P0", {"refinement": False})
    assert doc.execution.complete and not _stated(doc) and _quantity(doc, "peak_irradiance_W_per_mm2").reason == "ABERRATED_BEYOND_FIRST_ORDER"
    u = v = np.arange(-0.1, 0.1, 0.002)
    error = max(abs(e) for e in _errors(None, truth, radius, u, v, _on_grid(first, radius, u, v)))
    assert error > 0.08 and _check(doc, "aberration_value_change").observed == pytest.approx(ABERRATION_PROFILE_MARGIN * error, rel=0.15)
    lam, w = 1064.0, 0.7
    lens = element("L", "lens", PCX_F50, (lens_x, 0.0, 0.0))
    waist, rayleigh = _gaussian_waist_rayleigh(w, lens_x, PCX_F50, lam)
    distances = [waist - 0.6 * rayleigh, waist, waist + 0.6 * rayleigh]
    results = load_run(run(project([_component(_gaussian_field(128, 8.4 * w / 128, w, rays=4096, lam_nm=lam)), lens],
                                   [plane(f"P{i}", (exit_x + z, 0.0, 0.0), radius_mm=12.0) for i, z in enumerate(distances)])))
    radius = np.arange(0.0, 0.25, 0.0002)
    for i, ((truth, _first), z) in enumerate(zip(_pcx_reference(PCX_F50, w, lens_x, distances, radius, lam), distances)):
        doc = results.field(f"P{i}", {"refinement": False})
        if i == 1:
            assert _stated(doc), doc.models
            continue
        assert doc.execution.complete and not _stated(doc) and _quantity(doc, "peak_irradiance_W_per_mm2").reason == "ABERRATED_BEYOND_FIRST_ORDER"
        # (the first-order value the check refuses: the Gaussian beam of q through the paraxial matrix, on the reference's grid)
        beam = rayleigh * math.sqrt(1.0 + ((z - waist) / rayleigh) ** 2) * math.sqrt(lam * 1e-6 / (math.pi * rayleigh))
        u = v = np.arange(-4.0 * beam, 4.0 * beam, beam / 40.0)
        first_order = _on_grid(np.exp(-2.0 * radius ** 2 / beam ** 2), radius, u, v)
        peak, _width = _errors(None, truth, radius, u, v, first_order)
        assert abs(peak) > 0.01, (i, peak)
        # (e) the accuracy asked is the request's: with value_tolerance 0.03 the same plane is stated, its values are within
        # that of the exact reference, and each says which accuracy it was computed to
        refused = _check(doc, "aberration_value_change")
        assert refused.within is False and refused.maximum == 0.01 and refused.setting == "value_tolerance"
        assert _quantity(doc, "peak_irradiance_W_per_mm2").depends_on == ("aberration_value_change",)
        assert "value_tolerance" not in doc.settings["request"]
        asked = results.field(f"P{i}", {"refinement": False, "value_tolerance": 0.03})
        assert _stated(asked) and asked.settings["request"]["value_tolerance"] == 0.03
        change = _check(asked, "aberration_value_change")
        assert change.within is True and change.maximum == 0.03 and change.observed == refused.observed and change.setting == "value_tolerance"
        computed = np.array(asked.arrays["irradiance_mean_W_per_mm2"])
        peak_error, width_error = _errors(None, truth, radius, np.array(asked.arrays["map_u_mm"]), np.array(asked.arrays["map_v_mm"]), computed)
        assert 0.005 < abs(peak_error) < 0.03 and abs(width_error) < 0.03, (i, peak_error, width_error)
        for name in ("power_window_W", "peak_irradiance_W_per_mm2", "centroid_u_mm", "rms_radius_u_mm", "contrast"):
            entry = _quantity(asked, name).value_basis
            assert entry.kind.value == "MODEL" and entry.settings() == ("value_tolerance",), name
            assert {c.parameter: (c.maximum, c.within) for c in entry.validity} == {"aberration_value_change": (0.03, True), "window_refinement_change": (0.03, True)}
        assert _quantity(asked, "geometric_power_W").value_basis is None
    # at the default accuracy no value carries an entry and no check names the setting
    assert all(q.value_basis is None for q in results.field("P1", {"refinement": False}).quantities)
    assert all(c.setting is None for m in results.field("P1", {"refinement": False}).models for c in m.validity)
    with pytest.raises(FieldError) as invalid:
        results.field("P1", {"value_tolerance": 0.2})
    assert invalid.value.code == "FIELD_INVALID"


def test_irradiance_far_from_a_focus_is_reported_where_only_the_phase_is_aberrated():
    """SRC3-3. A Gaussian of w 4 mm through a 100 mm plano-convex lens, 20 mm and 60 mm behind it: its complex field is aberrated
    (relative to first order, about a third of its rms), its irradiance hardly. Reference (``_pcx_trace``, second moments): the
    wave's rms width is the first-order Gaussian beam's (q through the lens's paraxial matrix) plus the difference between the
    real rays' and the paraxial rays' second moments at the plane (the diffraction term is common to both). The irradiance
    values are reported (their change under the aberration is below 1e-3) and the rms width is that reference's (1e-3). With
    ``include_field`` the complex field is withheld (aberration_field_error above 0.1: ``arrays.field_withheld``) and the
    plane's values are those of the request without it, quantity for quantity: that condition is the field's alone. At the
    focus the aberration is far beyond first order: refused. The old field-error bound refused all three planes (0.72, 0.94,
    1.23)."""
    from local_helpers import lens_matrix

    from os_optics.results.field import ABERRATION_PROFILE_MARGIN

    lens_x, w = 10.0, 4.0
    lens = element("L", "lens", PCX_F100, (lens_x, 0.0, 0.0))
    exit_x = lens_x + _exit_vertex(PCX_F100)
    distances = [20.0 - exit_x, 60.0 - exit_x]
    results = load_run(run(project([_component(_gaussian_field(256, 8.4 * w / 256, w, rays=2048)), lens],
                                   [plane(f"P{i}", (exit_x + z, 0.0, 0.0), radius_mm=12.0) for i, z in enumerate(distances)]
                                   + [plane("F", (_back_focus_x(PCX_F100, lens_x), 0.0, 0.0), radius_mm=12.0)])))
    trace = _pcx_trace(PCX_F100, w, lens_x)
    h, landing, sine, bend = trace["h"], trace["landing"], trace["sine"], trace["bend"]
    weight = np.exp(-2.0 * h * h / trace["w"] ** 2) * h  # (the rays' power per radial step)
    paraxial = trace["lens"] @ np.vstack([h * (1.0 + 1.0 / bend), h / bend])  # (height, slope at the exit face)

    def moment(x: np.ndarray) -> float:  # (per axis: half the radial second moment)
        return 0.5 * float((weight * x * x).sum() / weight.sum())

    m = lens_matrix(PCX_F100, LAM_NM)
    q0 = complex(lens_x, math.pi * w * w / LAM_MM)
    q_exit = (m[0, 0] * q0 + m[0, 1]) / (m[1, 0] * q0 + m[1, 1])
    for i, z in enumerate(distances):
        doc = results.field(f"P{i}", {"refinement": False})
        assert _stated(doc), (i, doc.models)
        assert _check(doc, "aberration_value_change").observed < ABERRATION_PROFILE_MARGIN * 1e-3  # (the profile's change, counted twice)
        real = landing + z * sine / np.sqrt(1.0 - sine * sine)
        first = paraxial[0] + z * paraxial[1]
        gaussian = 0.5 * math.sqrt(-LAM_MM / (math.pi * (1.0 / (q_exit + z)).imag))  # (the rms width: half the 1/e^2 radius)
        expected = math.sqrt(gaussian * gaussian + moment(real) - moment(first))
        assert _value(doc, "rms_radius_u_mm") == pytest.approx(expected, rel=1e-3), (i, expected / gaussian - 1.0)
        with_field = results.field(f"P{i}", {"refinement": False, "include_field": True})
        error = _check(with_field, "aberration_field_error")
        assert error.observed > 0.1 and error.maximum == 0.1 and error.within is False
        # the plane's values do not depend on that condition: stated, and bit for bit those of the request without the field
        assert _stated(with_field) and [q.model_dump() for q in with_field.quantities] == [q.model_dump() for q in doc.quantities]
        assert with_field.arrays["irradiance_mean_W_per_mm2"] == doc.arrays["irradiance_mean_W_per_mm2"]
        assert "field_re" not in with_field.arrays and "field_im" not in with_field.arrays
        assert with_field.arrays["field_withheld"] == {"check": "aberration_field_error", "observed": error.observed, "maximum": 0.1}
        assert "field_withheld" not in doc.arrays
    focus = results.field("F", {"refinement": False})
    assert focus.execution.complete and not _stated(focus) and _quantity(focus, "rms_radius_u_mm").reason == "ABERRATED_BEYOND_FIRST_ORDER"


def test_a_field_off_the_grid_centre_keeps_its_power_through_an_aperture():
    """SRC3-2. A Gaussian of w 1 mm centred 0, 5 and 9 mm off its grid's centre (its 1e-6 edge 2.6 mm further), through an uncoated
    N-BK7 window of clear radius 11.43 mm: nothing cuts it, and the power on the plane is P (1 - R)^2 at normal incidence (R from
    the product document's Sellmeier, 1e-6). The probe beam that gives the axis's transmission was sized by the field's rms about the
    grid centre (2 sqrt(x0^2 + w^2 / 4)) and cut by the window: refused as APERTURES_CUT_FIELD (before that fix: 4.6 % / 20.5 % low).
    At 10.5 mm the field's edge is cut (3 % of it beyond the clear radius): refused, APERTURES_CUT_FIELD."""
    from local_helpers import WINDOW, product_document, sellmeier_index

    model = product_document(WINDOW)["simulation_models"][0]
    n = sellmeier_index(model["media"][0], LAM_NM)
    transmitted = 0.001 * (1.0 - ((n - 1.0) / (n + 1.0)) ** 2) ** 2
    for x0, cut in ((0.0, False), (5.0, False), (9.0, False), (10.5, True)):
        results = load_run(run(project([_component(_gaussian_field(256, 0.1, 1.0, x0=x0, rays=2048)), element("W", "window", WINDOW, (10.0, 0.0, 0.0))],
                                       [plane("P", (60.0, 0.0, 0.0), radius_mm=30.0)])))
        doc = results.field("P", {"refinement": False})
        if cut:
            assert doc.execution.complete and not _stated(doc) and _quantity(doc, "power_window_W").reason == "APERTURES_CUT_FIELD"
            assert _check(doc, "aperture_pass_share").within is False
            continue
        assert _stated(doc), (x0, doc.models)
        assert _value(doc, "power_window_W") == pytest.approx(transmitted, rel=1e-6), x0
        assert _value(doc, "centroid_u_mm") == pytest.approx(x0, abs=1e-6)


def test_an_aperture_that_cuts_the_probe_beam_and_not_the_field_leaves_the_power_right(monkeypatch):
    """SRC3-2 again, with the probe beam that gives the axis's transmission made 8 and 12 times wider than the field (the
    geometry before the probe was sized by the field): the window's clear aperture (11.43 mm) cuts 1.7 % and 16 % of the probe
    and nothing of the field (w 1 mm). The probe's cut is its own Gaussian tail's, not the field's: it is undone, so the power
    on the plane is P (1 - R)^2 (closed form, 1e-6) and the centroid and width are those of the run with the probe as it is.
    Without that the power read 1.7 % / 16 % low and the plane was refused (APERTURES_CUT_FIELD)."""
    from local_helpers import WINDOW, product_document, sellmeier_index

    field_module = importlib.import_module("os_optics.results.field")  # (the module: the package exports a function of that name)

    model = product_document(WINDOW)["simulation_models"][0]
    n = sellmeier_index(model["media"][0], LAM_NM)
    transmitted = 0.001 * (1.0 - ((n - 1.0) / (n + 1.0)) ** 2) ** 2
    results = load_run(run(project([_component(_gaussian_field(256, 0.1, 1.0, rays=2048)), element("W", "window", WINDOW, (10.0, 0.0, 0.0))],
                                   [plane("P", (60.0, 0.0, 0.0), radius_mm=30.0)])))
    plain = results.field("P", {"refinement": False})
    assert _stated(plain) and _value(plain, "power_window_W") == pytest.approx(transmitted, rel=1e-6)
    probe = field_module._probe
    for factor, cut in ((8.0, 0.017), (12.0, 0.16)):
        def wide(emitter, wavelength_nm, cfg, factor=factor):
            made = probe(emitter, wavelength_nm, cfg)
            return dataclasses.replace(made, w0x=factor * made.w0x, w0y=factor * made.w0y)

        monkeypatch.setattr(field_module, "_probe", wide)
        doc = results.field("P", {"refinement": False})
        assert _stated(doc), (factor, doc.models)
        assert _check(doc, "geometric_transmission").observed == pytest.approx(1.0, abs=1e-6)
        assert _value(doc, "power_window_W") == pytest.approx(transmitted, rel=1e-6), factor
        for name in ("centroid_u_mm", "rms_radius_u_mm", "rms_radius_v_mm", "peak_irradiance_W_per_mm2"):
            assert _value(doc, name) == pytest.approx(_value(plain, name), rel=1e-6, abs=1e-9), (factor, name)
        # the same run with the probe's power taken as it arrives: low by the probe's cut, and refused
        monkeypatch.setattr(field_module, "_axis_transmission", lambda out, arrival: float(arrival.power_W))
        raw = results.field("P", {"refinement": False})
        assert not _stated(raw) and _quantity(raw, "power_window_W").reason == "APERTURES_CUT_FIELD"
        assert _check(raw, "geometric_transmission").observed == pytest.approx(1.0 / (1.0 - cut), rel=5e-2)
        monkeypatch.undo()


def test_rays_never_leave_evanescent_and_unchecked_phases_are_not_traced():
    """SRC-5 / SRC-6. (a) A lit half (flat phase) beside a dim half (0.5e-6 of the peak) with a pi checkerboard phase on a 0.2 um
    grid: the dim points' gradient is sqrt(2) pi / p, a sine of sqrt(2) lambda / (2 p) = 2.24 (evanescent), and rays drawn in the
    lit cells next to them took it (bilinear), leaving with |d| up to 1.22 while the run passed. Now every grid point a ray's
    direction is interpolated from is checked: the run is refused SOURCE_FIELD_EVANESCENT, and the samples refuse a non-unit
    direction. (b) One bright point over a floor of 0.99e-6 of the peak with a random phase: the floor holds 6.1 % of the power
    (closed form), and its phase was never checked (below the 1e-6 'lit' level). The lit points now hold all but 1e-3 of the
    power: refused SOURCE_FIELD_UNDERSAMPLED."""
    count, pitch = 32, 0.0002
    x = _axis(count, pitch)
    xx, _yy = np.meshgrid(x, x)
    amplitude = np.where(xx < 0.0, 1.0, math.sqrt(0.5e-6))
    index = np.add.outer(np.arange(count), np.arange(count))
    phase = np.where(xx < 0.0, 0.0, np.where(index % 2 == 0, 0.0, math.pi))
    issue = _untraced_field(run(project([_component(_field(amplitude.tolist(), phase.tolist(), pitch, rays=4096))],
                                        [plane("P", (0.01, 0.0, 0.0))])), "SOURCE_FIELD_EVANESCENT")
    sine = math.sqrt(2.0) * LAM_MM / (2.0 * pitch)  # (in vacuum; air's index differs by 3e-4)
    reported = float(issue.message.split("phase gradient reaches ")[1].split(" ")[0])
    assert reported == pytest.approx(sine, rel=1e-3) and issue.numerical_context["largest_transverse_sine"] == pytest.approx(sine, rel=1e-3)
    pattern = PatternG(kind="field", grid=SourceGrid(count, count, pitch, pitch), interpolation="PIECEWISE_CONSTANT",
                       weights=amplitude ** 2, phase=phase, jones=np.array([1.0 + 0j, 0j]))
    emitter = EmitterG("E", "E", np.zeros(3), np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]),
                       ((LAM_NM, 1.0),), 1.0, "air", "rectangle", 0.5 * count * pitch, 0.5 * count * pitch, "COLLIMATED", 4096, 0,
                       pattern=pattern)
    with pytest.raises(AssertionError, match="transverse sine"):
        local_samples(emitter, 1.0)
    big, floor = 256, 0.99e-6
    amplitude = np.full((big, big), math.sqrt(floor))
    amplitude[big // 2, big // 2] = 1.0
    phase = np.random.default_rng(1).uniform(-math.pi, math.pi, (big, big))
    share = floor * (big * big - 1) / (1.0 + floor * (big * big - 1))
    assert share == pytest.approx(0.0609, abs=1e-4)
    _untraced_field(run(project([_component(_field(amplitude.tolist(), phase.tolist(), 0.01, rays=4096))], [plane("P", (10.0, 0.0, 0.0))])),
                    "SOURCE_FIELD_UNDERSAMPLED")


def test_widths_and_contrast_need_enough_points():
    """SRC-7. A region two grid points tall: no rms width along v and no contrast (NOT_APPLICABLE, TOO_FEW_POINTS); the rms width
    along u (40 points) is the Gaussian's over the region (within the +-0.2 mm cut: computed on the region's points from the
    closed form of w(z), 1e-3)."""
    w0, z = 0.1, 2.0
    results = load_run(run(project([_component(_gaussian_field(64, 0.01, w0))], [plane("P", (z, 0.0, 0.0), radius_mm=5.0)])))
    doc = results.field("P", {"region": {"half_width_mm": [0.2, 0.012]}})
    assert doc.arrays["method"] == "ANGULAR_SPECTRUM" and _stated(doc)
    for name in ("rms_radius_v_mm", "contrast"):
        q = _quantity(doc, name)
        assert q.state.value == "NOT_APPLICABLE" and q.reason == "TOO_FEW_POINTS", name
    w = w0 * math.sqrt(1.0 + (z * LAM_MM / (math.pi * w0 * w0)) ** 2)
    u = (np.arange(-20, 20) + 0.5) * 0.01  # (the grid's points are centred: half a pitch off the axis)
    profile = np.exp(-2.0 * u * u / (w * w))
    assert _value(doc, "rms_radius_u_mm") == pytest.approx(math.sqrt(float((profile * u * u).sum() / profile.sum())), rel=1e-3)


_OVERFLOW_MAP = {"model_kind": "SAMPLED_INTENSITY", "wavelength_nm": LAM_NM,
                 "intensity_map": {"grid": {"column_count": 2, "row_count": 2, "pitch_mm": [10.0, 10.0]},
                                   "values": [[1e306, 1e306], [1e306, 1e306]], "unit": "W_PER_MM2", "interpolation": "PIECEWISE_CONSTANT"},
                 "angular_law": "COLLIMATED", "sampling": {"sample_count": 256, "seed": 5}}


def _sqrt_w(amplitude: list) -> dict:
    model = _field(amplitude, np.zeros((len(amplitude), len(amplitude[0]))).tolist(), 0.01)
    model["amplitude_unit"] = "SQRT_W_PER_MM2"
    del model["power_w"]
    return model


@pytest.mark.parametrize(("label", "model"), [
    ("amplitude whose square overflows", _field((np.ones((8, 8)) * 1e160).tolist(), np.zeros((8, 8)).tolist(), 0.01)),
    ("amplitude whose square underflows", _field((np.ones((8, 8)) * 1e-170).tolist(), np.zeros((8, 8)).tolist(), 0.01)),
    ("SQRT_W amplitude whose power overflows", _sqrt_w((np.ones((8, 8)) * 1e154).tolist())),
    ("W_PER_MM2 map whose integral overflows", _OVERFLOW_MAP),
])
def test_values_beyond_float_range_are_refused_as_invalid_projects(label, model):
    """SRC-8. Values whose squares or sums leave float64 raised a bare OverflowError, a pydantic ValidationError of a derived line
    or a numerical failure of the run; they are a PROJECT_INVALID refusal of the Project."""
    with pytest.raises(LocalRunError) as caught:
        run(project([_component(model)], [plane("P", (5.0, 0.0, 0.0))]))
    assert "PROJECT_INVALID" in str(caught.value), label


def test_a_weak_diffuser_is_not_reported_as_fully_developed_speckle():
    """SRC-9. A 1 mm disc through screens of rms 0.3 rad (correlation 0.1 mm), 200 mm away: one pattern's contrast is 0.81, almost all
    of it the disc's diffraction ripple on the unscattered beam (exp(-0.09) = 0.914 of the power). The document states the screens'
    rms phase (the stated 0.3 rad, 5 %), the unscattered share (|<exp(i phi)>|^2 against exp(-sigma^2) of the screens' rms: 0.01) and
    that the speckle is not fully developed; a strong screen (2.5 rad) is."""
    def speckle(rms: float) -> dict:
        return {"model_kind": "SPECKLE", "wavelength_nm": LAM_NM, "power_w": 0.001, "polarization": {"kind": "LINEAR", "angle_deg": 0.0},
                "illumination": {"shape": "UNIFORM_DISC", "diameter_mm": 1.0},
                "phase_screen": {"statistics": "GAUSSIAN_CORRELATED", "phase_rms_rad": rms, "correlation_length_mm": 0.1, "seed": 3},
                "grid": {"column_count": 256, "row_count": 256, "pitch_mm": [0.01, 0.01]}, "realizations": 4,
                "ray_sampling": {"sample_count": 1024, "seed": 0}}

    for rms, developed in ((0.3, False), (2.5, True)):
        results = load_run(run(project([_component(speckle(rms))], [plane("P", (200.0, 0.0, 0.0), radius_mm=50.0)])))
        doc = results.field("P", {"region": {"half_width_mm": [0.5, 0.5]}, "refinement": False})
        assert _stated(doc) and doc.arrays["speckle_fully_developed"] is developed
        sigma = _value(doc, "screen_phase_rms_rad")
        assert sigma == pytest.approx(rms, rel=0.05)
        if not developed:
            assert _value(doc, "unscattered_power_share") == pytest.approx(math.exp(-sigma * sigma), abs=0.01)
            assert "fully developed" in _quantity(doc, "contrast").message
