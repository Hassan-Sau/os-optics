"""Extended emitters: sampling, normalization, power accounting and results, against closed-form references.

Every reference below is computed here, independently of the Engine, from the geometry stated in the test. Tolerances were set
before the first run from the sampling's expected error (a stratified 2-D estimate of a fraction F over N rays is well inside
the Monte Carlo sigma sqrt(F (1 - F) / N)):

* collection fractions at 4096 rays: absolute 0.004 (about 0.6 sigma of plain Monte Carlo at F = 0.2);
* an irradiance annulus at 16384 rays: relative 3 %, a coarse map cell at 16384 rays: relative 10 %.

Sources of the closed forms: the configuration factor of a differential Lambertian element to a coaxial parallel disc is
``sin^2(theta)`` and the coaxial parallel disc-to-disc factor (Siegel & Howell, Thermal Radiation Heat Transfer) and the aligned
identical-rectangles factor (Incropera & DeWitt, table of view factors) are written out below.
"""

from __future__ import annotations

import csv
import io
import math

import numpy as np
import pytest
from local_helpers import ALONG_X, PCX_F50, catalog_document, element, plane, project, run as _fixture_run, source

from os_optics.config import NumericalConfig
from os_optics.contracts.results import ValidationResult
from os_optics.errors import ErrorCode, Status
from os_optics.optics.materials.library import MaterialLibrary
from os_optics.optics.rays.emitter_samples import _shifts, local_samples, sobol_points
from os_optics.optics.rays.sampled import DISPLAY_SAMPLES, SampleTracer
from os_optics.results import RunResults
from os_optics.scene.lower import lower_scene
from os_optics.scene.model import EmitterG

COLLECTION_ABS_TOL = 0.004
ANNULUS_REL_TOL = 0.03
MAP_CELL_REL_TOL = 0.10
RAYS = 4096


def run(document, **kwargs):
    """Synthetic fixture lens assumes no bulk absorption for geometric radiometric accounting."""
    fixture = catalog_document()
    fixture["schema_version"] = "3.6"  # Required for material bulk-loss evidence.
    for material in fixture["materials"]:
        material["bulk_loss"] = {"kind": "INTERNAL_TRANSMITTANCE",
                                 "wavelength_nm": [material["valid_band_nm"]["min_nm"], material["valid_band_nm"]["max_nm"]],
                                 "internal_transmittance": ["1", "1"], "thickness_mm": ["1", "1"],
                                 "basis": "Assumed lossless synthetic test fixture."}
    return _fixture_run(document, catalog=fixture, **kwargs)
MAP_RAYS = 16384


def emitter(law: str = "LAMBERTIAN", *, rays: int = RAYS, seed: int = 1, surface: dict | None = None, position=(0.0, 0.0, 0.0),
            power_w: float = 1.0, wavelength_nm: float = 550.0, instance_id: str = "EM", quaternion=ALONG_X, lines: list | None = None) -> dict:
    model = {"model_kind": "EXTENDED_EMITTER", "emitting_surface": surface or {"shape": "DISC", "diameter_mm": 0.02}, "angular_law": law,
             "sampling": {"sample_count": rays, "seed": seed}}
    model.update({"spectral_lines": lines} if lines else {"wavelength_nm": wavelength_nm, "power_w": power_w})
    return {"instance_id": instance_id, "optical_role": "source", "custom_part": {"custom_part_key": "emitter", "description": "Extended emitter."},
            "nominal_pose": {"position_mm": list(position), "quaternion_wxyz": list(quaternion)}, "source_model": model}


def detector(oid: str, x: float, radius_mm: float, *, absorbing: bool = True) -> dict:
    return {**plane(oid, (x, 0.0, 0.0), radius_mm=radius_mm), "absorbing": absorbing}


def fraction_of(document: dict, oid: str = "D") -> float:
    observation = next(o for o in run(document).result.observations if o.observation_id == oid)
    return observation.sampled_light[0].collection_fraction


def disc_to_disc_view_factor(r1: float, r2: float, h: float) -> float:
    """F(1 -> 2) of coaxial parallel discs of radii r1, r2 a distance h apart."""
    big_r1, big_r2 = r1 / h, r2 / h
    x = 1.0 + (1.0 + big_r2 * big_r2) / (big_r1 * big_r1)
    return 0.5 * (x - math.sqrt(x * x - 4.0 * (big_r2 / big_r1) ** 2))


def aligned_rectangles_view_factor(a: float, b: float, c: float) -> float:
    """F of identical aligned parallel rectangles a x b a distance c apart."""
    x, y = a / c, b / c
    return (2.0 / (math.pi * x * y)) * (
        0.5 * math.log((1.0 + x * x) * (1.0 + y * y) / (1.0 + x * x + y * y))
        + x * math.sqrt(1.0 + y * y) * math.atan(x / math.sqrt(1.0 + y * y))
        + y * math.sqrt(1.0 + x * x) * math.atan(y / math.sqrt(1.0 + x * x))
        - x * math.atan(x) - y * math.atan(y))


# ------------------------------------------------------------------------------------------------- the sequence


def test_sobol_points_are_the_published_sequence_before_the_shift():
    # Joe-Kuo's first points (Gray-code order, 5 dimensions), by hand from the direction numbers
    expected = [(0.0, 0.0, 0.0, 0.0, 0.0), (0.5, 0.5, 0.5, 0.5, 0.5), (0.75, 0.25, 0.25, 0.25, 0.75), (0.25, 0.75, 0.75, 0.75, 0.25)]
    u = sobol_points(4, 0)
    raw = np.floor(u * 2.0 ** 32).astype(np.uint64) ^ np.array(_shifts(0), dtype=np.uint64)
    assert (raw / 2.0 ** 32).tolist() == [list(p) for p in expected]


@pytest.mark.parametrize("pair", [(0, 1), (2, 3), (0, 2), (1, 3), (0, 4), (2, 4)])
def test_each_prefix_is_stratified_in_every_pair_used(pair):
    for count in (1024, 512):  # (the full set and the half-sample estimate)
        u = sobol_points(count, 12345)[:, pair]
        cells = np.floor(u * 8).astype(int)
        counts = np.bincount(cells[:, 0] * 8 + cells[:, 1], minlength=64)
        assert counts.min() >= count // 64 - count // 128 and counts.max() <= count // 64 + count // 128


def test_sampling_is_bit_reproducible_and_seeded():
    a = sobol_points(256, 7)
    assert a.tobytes() == sobol_points(256, 7).tobytes()
    assert a.tobytes() != sobol_points(256, 8).tobytes()
    assert np.array_equal(sobol_points(128, 7), a[:128])  # a prefix: the half-sample estimate needs no second trace
    assert float(a.min()) > 0.0 and float(a.max()) < 1.0


def _emitter_g(law: str, shape: str = "disc") -> EmitterG:
    z = np.array([0.0, 0.0, 1.0])
    return EmitterG("E", "E", np.zeros(3), z, np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), ((550.0, 1.0),), 1.0, "air", shape, 2.0, 1.0, law, 4096, 3)


def test_sampled_directions_follow_the_angular_law():
    _, d, _ = local_samples(_emitter_g("LAMBERTIAN"))
    assert np.allclose(np.sum(d * d, axis=1), 1.0, atol=1e-14)
    assert float(d[:, 2].min()) > 0.0  # the hemisphere around +z
    # cos(theta) dOmega: E[cos^2] = 1/2 and E[cos] = 2/3 under the Lambertian law (uniform: 1/3 and 0 over the sphere)
    assert abs(float(np.mean(d[:, 2] ** 2)) - 0.5) < 1e-3
    assert abs(float(np.mean(d[:, 2])) - 2.0 / 3.0) < 1e-3
    _, d, _ = local_samples(_emitter_g("ISOTROPIC"))
    assert np.allclose(np.sum(d * d, axis=1), 1.0, atol=1e-14)
    assert abs(float(np.mean(d[:, 2] ** 2)) - 1.0 / 3.0) < 1e-3
    assert abs(float(np.mean(d[:, 2]))) < 1e-3
    assert float(d[:, 2].min()) < 0.0 < float(d[:, 2].max())  # both sides


def test_sampled_points_fill_the_area_uniformly():
    p, _, state = local_samples(_emitter_g("LAMBERTIAN", "rectangle"))
    assert abs(float(np.max(np.abs(p[:, 0]))) - 2.0) < 1e-2 and abs(float(np.max(np.abs(p[:, 1]))) - 1.0) < 1e-2
    assert abs(float(np.mean(p[:, 0] ** 2)) - 4.0 / 3.0) < 1e-3  # a^2 / 3 of a uniform interval
    assert 0.48 < float(state.mean()) < 0.52  # the two orthogonal states
    p, _, _ = local_samples(_emitter_g("LAMBERTIAN", "disc"))
    assert float(np.max(np.hypot(p[:, 0], p[:, 1]))) <= 2.0
    assert abs(float(np.mean(p[:, 0] ** 2 + p[:, 1] ** 2)) - 2.0) < 1e-2  # R^2 / 2 of a uniform disc


# ----------------------------------------------------------------------------------------------- analytic references


def test_small_lambertian_disc_collection_is_sin_squared_theta():
    # emitter 0.02 mm in diameter, disc aperture of radius 5 mm at 10 mm: sin^2(theta) = 25 / 125
    fraction = fraction_of(project([emitter()], [detector("D", 10.0, 5.0)]))
    reference = 5.0 ** 2 / (5.0 ** 2 + 10.0 ** 2)
    assert fraction == pytest.approx(reference, abs=COLLECTION_ABS_TOL)
    assert abs(reference - disc_to_disc_view_factor(0.01, 5.0, 10.0)) < 1e-6


def test_finite_lambertian_disc_matches_the_disc_to_disc_view_factor():
    # 6 mm emitter, aperture radius 5 mm at 10 mm: the source's extent lowers the factor well beyond the tolerance
    fraction = fraction_of(project([emitter(surface={"shape": "DISC", "diameter_mm": 6.0})], [detector("D", 10.0, 5.0)]))
    reference = disc_to_disc_view_factor(3.0, 5.0, 10.0)
    assert fraction == pytest.approx(reference, abs=COLLECTION_ABS_TOL)
    assert 0.2 - reference > 3.0 * COLLECTION_ABS_TOL * 0.5


def test_lambertian_rectangle_matches_the_aligned_rectangles_view_factor():
    surface = {"shape": "RECTANGLE", "width_mm": 10.0, "height_mm": 6.0}
    doc = project([emitter(surface=surface)], [_rect_plane("D", 10.0, 5.0, 3.0)])
    fraction = fraction_of(doc)
    assert fraction == pytest.approx(aligned_rectangles_view_factor(10.0, 6.0, 10.0), abs=COLLECTION_ABS_TOL)


def _rect_plane(oid: str, x: float, half_u: float, half_v: float) -> dict:
    return {**plane(oid, (x, 0.0, 0.0)), "aperture": {"shape": "rectangle", "half_width_mm": half_u, "half_height_mm": half_v}, "absorbing": True}


def test_isotropic_emitter_collection_is_solid_angle_over_four_pi():
    # Omega / 4 pi = (1 - cos(theta)) / 2: not the Lambertian sin^2(theta)
    fraction = fraction_of(project([emitter("ISOTROPIC")], [detector("D", 10.0, 5.0)]))
    reference = (1.0 - 10.0 / math.hypot(5.0, 10.0)) / 2.0
    assert fraction == pytest.approx(reference, abs=COLLECTION_ABS_TOL)
    assert abs(reference - 0.2) > 3.0 * COLLECTION_ABS_TOL


def test_isotropic_light_reaches_both_sides_and_lambertian_only_the_front():
    back = {**plane("B", (-10.0, 0.0, 0.0), radius_mm=5.0), "absorbing": True}
    reference = (1.0 - 10.0 / math.hypot(5.0, 10.0)) / 2.0
    result = run(project([emitter("ISOTROPIC")], [back])).result
    assert result.observations[0].sampled_light[0].collection_fraction == pytest.approx(reference, abs=COLLECTION_ABS_TOL)
    result = run(project([emitter("LAMBERTIAN")], [back])).result
    assert not result.observations[0].lit


def test_irradiance_on_a_parallel_plane_falls_as_cos_fourth():
    # small Lambertian source, plane at 20 mm: E(r) = P cos^4(theta) / (pi d^2); the annulus between r1 and r2 holds
    # P (sin^2(theta_2) - sin^2(theta_1))
    distance = 20.0
    doc = project([emitter(rays=MAP_RAYS)], [plane("D", (distance, 0.0, 0.0), radius_mm=40.0)])
    observation = run(doc).result.observations[0]
    crossings = observation.sampled_light[0].crossings
    r = np.hypot(np.array(crossings.u_mm), np.array(crossings.v_mm))
    power = np.array(crossings.power_W)
    for r1, r2 in ((0.0, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, 40.0)):
        measured = float(power[(r >= r1) & (r < r2)].sum())
        sin2 = lambda x: x * x / (x * x + distance * distance)  # noqa: E731
        assert measured == pytest.approx(sin2(r2) - sin2(r1), rel=ANNULUS_REL_TOL)
    # the map's cells against the law at each cell centre (cells small enough for the law to be nearly linear across them)
    results = RunResults(result=run(doc).result)
    grid = results.irradiance_map("D", samples=8, half_extent_mm=16.0, center_uv_mm=[0.0, 0.0])
    assert grid["status"] == "OK"
    assert grid["coherent_W_per_mm2"] is None and grid["coherent_reason"] == "INDEPENDENT_SAMPLES"
    assert grid["composition"] == "INCOHERENT_SUM"
    assert grid["grid"]["bin_area_mm2"] == pytest.approx(16.0)
    assert len(grid["grid"]["u_edges_mm"]) == 9
    for j, v in enumerate(grid["grid"]["v_mm"]):
        for i, u in enumerate(grid["grid"]["u_mm"]):
            cos2 = distance ** 2 / (distance ** 2 + u * u + v * v)
            assert grid["incoherent_W_per_mm2"][j][i] == pytest.approx(cos2 ** 2 / (math.pi * distance ** 2), rel=MAP_CELL_REL_TOL)
    in_grid = grid["grid_power_incoherent_W"] + grid["power_outside_grid_W"]
    assert in_grid == pytest.approx(math.fsum(crossings.power_W), rel=1e-12)


# ------------------------------------------------------------------------------------------ normalization and reporting


def test_reported_normalization_units_and_sampling_domain():
    result = run(project([emitter(surface={"shape": "RECTANGLE", "width_mm": 2.0, "height_mm": 0.5}, power_w=0.25, rays=64)],
                         [detector("D", 10.0, 5.0)])).result
    line = result.sampled_sources[0]
    assert line.emitted_W == 0.25 and line.area_mm2 == 1.0
    assert line.angular_law == "LAMBERTIAN" and line.solid_angle_measure == "PROJECTED_SOLID_ANGLE"
    assert line.domain_sr == pytest.approx(math.pi)
    assert line.radiance_W_per_m2_sr == pytest.approx(0.25 / (math.pi * 1e-6))  # L = P / (pi A), A in m^2
    assert line.intensity_W_per_sr is None
    assert line.ray_weight_W == 0.25 / 64 and line.sample_count == 64 and line.samples_traced == 64
    result = run(project([emitter("ISOTROPIC", power_w=2.0, rays=64)], [detector("D", 10.0, 5.0)])).result
    line = result.sampled_sources[0]
    assert line.solid_angle_measure == "SOLID_ANGLE" and line.domain_sr == pytest.approx(4.0 * math.pi)
    assert line.intensity_W_per_sr == pytest.approx(2.0 / (4.0 * math.pi)) and line.radiance_W_per_m2_sr is None


def test_convergence_evidence_is_two_estimates_from_one_run():
    light = run(project([emitter()], [detector("D", 10.0, 5.0)])).result.observations[0].sampled_light[0]
    assert light.half_sample_count == RAYS // 2
    half_rows = [p for p, k, c in zip(light.crossings.power_W, light.crossings.sample_index, light.crossings.counted) if c and k < RAYS // 2]
    assert light.collection_fraction_half_samples == pytest.approx(math.fsum(half_rows) / 0.5)
    assert light.collection_fraction_difference == pytest.approx(abs(light.collection_fraction - light.collection_fraction_half_samples))
    assert light.collection_fraction_difference < 0.01
    # F = 0.2, each ray in or out: sqrt(F (1 - F) / N), written out
    assert light.collection_fraction_sigma == pytest.approx(math.sqrt(light.collection_fraction * (1.0 - light.collection_fraction) / RAYS), rel=1e-9)


def test_a_seed_gives_the_same_result_and_another_seed_an_independent_estimate():
    doc = project([emitter(seed=11, rays=1024)], [detector("D", 10.0, 5.0)])
    a, b = run(doc).result, run(doc).result
    assert a.manifest.result_hash == b.manifest.result_hash and a.manifest.trace_hash == b.manifest.trace_hash
    other = run(project([emitter(seed=12, rays=1024)], [detector("D", 10.0, 5.0)])).result
    assert other.manifest.result_hash != a.manifest.result_hash
    assert abs(other.observations[0].sampled_light[0].collection_fraction - a.observations[0].sampled_light[0].collection_fraction) < 0.02


def test_each_spectral_line_is_sampled_and_accounted_on_its_own():
    lines = [{"wavelength_nm": 450.0, "power_w": 0.25}, {"wavelength_nm": 650.0, "power_w": 0.75}]
    result = run(project([emitter(lines=lines, rays=1024)], [detector("D", 10.0, 5.0)])).result
    assert [s.emitted_W for s in result.sampled_sources] == [0.25, 0.75]
    observation = result.observations[0]
    assert [line.wavelength_nm for line in observation.sampled_light] == [450.0, 650.0]
    fractions = [line.collection_fraction for line in observation.sampled_light]
    assert fractions[0] == pytest.approx(fractions[1])
    assert [s.wavelength_nm for s in observation.power_by_wavelength] == [450.0, 650.0]
    assert observation.total_power_W == pytest.approx(fractions[0])
    # without a wavelength: the two lines' histograms on one grid, added (lines of different wavelengths add in irradiance)
    results = RunResults(result=result)
    total = results.irradiance_map("D")
    assert total["status"] == "OK" and total["wavelength_nm"] is None and total["wavelengths_nm"] == [450.0, 650.0]
    window = {"center_uv_mm": total["grid"]["center_uv_mm"], "samples": total["grid"]["shape"][0], "half_extent_mm": total["grid"]["half_extent_mm"]}
    blue, red = (results.irradiance_map("D", wavelength_nm=w, **window) for w in (450.0, 650.0))
    assert red["status"] == "OK" and red["wavelength_nm"] == 650.0 and blue["grid"] == red["grid"] == total["grid"]
    assert np.array_equal(np.array(total["incoherent_W_per_mm2"]), np.array(blue["incoherent_W_per_mm2"]) + np.array(red["incoherent_W_per_mm2"]))
    assert np.array_equal(np.array(total["sample_counts"]), np.array(blue["sample_counts"]) + np.array(red["sample_counts"]))
    assert total["grid_power_incoherent_W"] == pytest.approx(blue["grid_power_incoherent_W"] + red["grid_power_incoherent_W"], rel=1e-12)
    # (a crossing counts wherever it lands on the plane: the grid's power and the power beyond it are all the crossings')
    crossing_power = math.fsum(math.fsum(line.crossings.power_W) for line in observation.sampled_light)
    assert total["grid_power_incoherent_W"] + total["power_outside_grid_W"] == pytest.approx(crossing_power, rel=1e-9)
    assert total["coherent_W_per_mm2"] is None and total["coherent_reason"] == "INDEPENDENT_SAMPLES"
    assert [row["wavelength_nm"] for row in total["per_wavelength"]] == [450.0, 650.0] and len(total["sampled_lines"]) == 2


# ----------------------------------------------------------------------------------------------------- power accounting


def test_ledger_closes_through_optics_and_counts_every_sink():
    doc = project([emitter(surface={"shape": "DISC", "diameter_mm": 2.0}), element("L", "lens", PCX_F50, (50.0, 0.0, 0.0))],
                  [detector("D", 100.0, 5.0), plane("M", (40.0, 0.0, 0.0), radius_mm=2.0)])
    result = run(doc).result
    line = result.power_budget.lines[0]
    assert line.conserved
    assert line.input_power_W == 1.0
    assert sum(line.sinks.values()) == pytest.approx(1.0, abs=1e-9)
    assert line.sinks["DETECTED"] > 0.0 and line.sinks["CLIPPED"] > 0.0 and line.sinks["ESCAPED"] > 0.0
    assert line.sinks["UNTRACED_REFLECTION"] > 0.0
    observed = {o.observation_id: o.sampled_light[0].power_W for o in result.observations}
    assert line.observed_W == pytest.approx(observed)
    assert observed["D"] == pytest.approx(line.sinks["DETECTED"])  # the absorbing plane took exactly what it counted


def test_light_a_ray_leaves_behind_is_stated_on_the_plane_never_silently_dropped():
    doc = project([emitter(surface={"shape": "DISC", "diameter_mm": 2.0}), element("L", "lens", PCX_F50, (50.0, 0.0, 0.0))],
                  [detector("D", 100.0, 5.0), plane("M", (40.0, 0.0, 0.0), radius_mm=2.0)])
    by_id = {o.observation_id: o for o in run(doc).result.observations}
    reflected = run(doc).result.power_budget.lines[0].sinks["UNTRACED_REFLECTION"]
    assert not by_id["D"].power_complete and by_id["D"].untraced_power_W == pytest.approx(reflected)
    assert by_id["D"].power_upper_bound_W == pytest.approx(by_id["D"].total_power_W + reflected)  # absorbed at most once
    assert by_id["M"].power_upper_bound_W is None  # a transparent plane may be crossed again
    # each plane's entry names the surfaces where the rays left that light behind
    for oid in ("D", "M"):
        (part,) = by_id[oid].value_basis["total_power_W"].omitted
        assert part.part.value == "NOT_FOLLOWED" and part.at == ("surface:L/S1", "surface:L/S2")


def test_a_plane_no_sampled_ray_reaches_is_a_sampled_zero_never_a_known_zero():
    # a pinhole a few rays may miss: the sampling reads 0 W there, resolved to one ray's weight (emitted / sample_count),
    # which is not a proof that the plane is dark
    def pinhole(rays: int, *targets):
        doc = project([emitter(rays=rays)], [detector("D", 200.0, 0.5)])
        if targets:
            doc["targets"] = list(targets)
        return run(doc).result

    result = pinhole(16)
    observation = result.observations[0]
    (line,) = observation.sampled_light
    emitted = line.emitted_W
    assert not observation.lit and observation.total_power_W == 0.0
    assert observation.quantity_state.value == "KNOWN" and observation.power_complete and observation.untraced_power_W == 0.0
    assert observation.power_sampling_resolution_W == pytest.approx(emitted / 16, rel=1e-15)
    assert (line.power_W, line.counted_crossing_count, line.sample_count, line.samples_traced) == (0.0, 0, 16, 16)
    # (no bound is taken from the sampled zero; this absorbing plane receives at most the launched power)
    assert observation.status is Status.PASS and observation.power_upper_bound_W == emitted
    issue = next(i for i in result.issues if i.code is ErrorCode.OBSERVATION_NOT_REACHED)
    assert issue.status is Status.NOT_APPLICABLE and issue.suggested_action is None
    assert issue.numerical_context["power_sampling_resolution_W"] == observation.power_sampling_resolution_W
    # doubling the rays halves the resolution
    assert pinhole(32).observations[0].power_sampling_resolution_W == pytest.approx(emitted / 32, rel=1e-15)
    # a power maximum within the resolution is not decided; one above it is met; a minimum above it fails
    small = {"target_id": "small", "observation_id": "D", "metric": "power_W", "maximum": "0.000000001"}
    large = {"target_id": "large", "observation_id": "D", "metric": "power_W", "maximum": str(2.0 * emitted / 16)}
    floor = {"target_id": "floor", "observation_id": "D", "metric": "power_W", "minimum": str(2.0 * emitted / 16)}
    graded = {t.target_id: t for t in pinhole(16, small, large, floor).targets}
    assert graded["small"].assessment.value == "INDETERMINATE" and graded["small"].value == 0.0
    assert graded["large"].assessment.value == "MET"
    assert graded["floor"].assessment.value == "UNMET"


@pytest.mark.parametrize(("radius_mm", "counted"), [(0.5, False), (50.0, True)])
def test_a_beam_metric_on_an_extended_emitters_light_is_out_of_scope_whether_or_not_a_ray_is_counted(radius_mm, counted):
    """A beam metric describes one beam; an extended emitter's light is independent rays. On a plane its counted rays reach
    the metric is out of model scope, and on a pinhole none of the 16 rays is counted on (a sampled zero, resolved to one
    ray's weight, not a dark plane) it is too: the sampling does not show that no light of the emitter arrives there, so
    the target is not UNMET for want of an arrival."""
    doc = project([emitter(rays=16)], [detector("D", 200.0, radius_mm)])
    doc["targets"] = [{"target_id": "spot", "observation_id": "D", "metric": "beam_radius_mm", "unit": "mm", "maximum": "0.05"}]
    result = run(doc).result
    (line,) = result.observations[0].sampled_light
    assert bool(line.counted_crossing_count) is counted
    (target,) = result.targets
    assert (target.status, target.assessment.value, target.value) == (Status.OUT_OF_MODEL_SCOPE, "INDETERMINATE", None)
    assert target.issue_codes == ("OUT_OF_MODEL_SCOPE",) and "independent rays of an extended emitter" in target.message


def test_the_ray_step_budget_books_what_it_leaves_untraced():
    doc = project([emitter(rays=64)], [detector("D", 10.0, 5.0)])
    from os_optics.local import ProjectDocumentV2, lower_project

    from local_helpers import load_test_catalog

    snapshot = lower_project(ProjectDocumentV2.model_validate(doc), catalog=load_test_catalog()).request.project
    cfg = NumericalConfig()
    library = MaterialLibrary(snapshot.materials, cfg)
    scene, _ = lower_scene(snapshot, cfg, library)
    tracer = SampleTracer(scene, library, cfg)
    tracer.sampled.step_budget = tracer.schedule.max_trace_steps = 10
    sampled = tracer.run_samples()
    line = sampled.lines[0]
    assert line.launched < 64
    flows = {f.kind: f.power_W for f in tracer.out.power_flows}
    assert flows["PATH_LIMIT"] >= (64 - line.launched) * line.weight_W
    assert sum(f.power_W for f in tracer.out.power_flows) == pytest.approx(1.0)
    issue = next(i for i in tracer.out.issues if i.code is ErrorCode.MAX_PATHS)
    assert issue.status is Status.WARNING and issue.numerical_context["untraced_power_W"] == pytest.approx(flows["PATH_LIMIT"])


def test_unpolarized_ensemble_through_a_polarizer_keeps_half_of_its_transmission():
    # at incidence within 12 deg a linear polarizer of pass transmittance t and leak t_perp passes (t + t_perp) / 2 of
    # unpolarized light, whichever way its axis lies: the two orthogonal states are stratified over the rays
    from local_helpers import POLARIZER

    target = plane("B", (15.0, 0.0, 0.0), radius_mm=3.0)
    free = run(project([emitter()], [target])).result.observations[0].sampled_light[0].power_W
    passed = run(project([emitter(), element("P", "polarizer", POLARIZER, (10.0, 0.0, 0.0))], [target])).result.observations[0].sampled_light[0].power_W
    model = next(p for p in catalog_document()["products"] if p["optical_sku"] == POLARIZER)["simulation_models"][0]
    t = float(model["transmittance"]["value"])
    reference = (t + t / float(model["extinction_ratio"]["value"])) / 2.0
    assert passed / free == pytest.approx(reference, abs=0.03)


# ---------------------------------------------------------------------------------- beside the beams; beam quantities


def test_gaussian_quantities_of_rays_are_unavailable_and_the_power_stays_measured():
    targets = [{"target_id": "P", "observation_id": "D", "metric": "power_W", "minimum": 0.1},
               {"target_id": "R", "observation_id": "D", "metric": "beam_radius_x_mm", "maximum": 1.0}]
    result = run(project([emitter(rays=256)], [detector("D", 10.0, 5.0)], targets)).result
    by_id = {t.target_id: t for t in result.targets}
    assert by_id["P"].status is Status.PASS and by_id["P"].value == pytest.approx(result.observations[0].total_power_W)
    assert by_id["R"].status is Status.OUT_OF_MODEL_SCOPE and by_id["R"].value is None


def test_a_gaussian_beam_beside_an_emitter_keeps_its_own_results():
    beam = source("LASER", (-5.0, 0.0, 0.0), wavelength_nm=550.0, power_w=0.5)
    targets = [{"target_id": "R", "observation_id": "D", "source_id": "LASER", "metric": "beam_radius_x_mm", "maximum": 5.0},
               {"target_id": "P", "observation_id": "D", "source_id": "EM", "metric": "power_W", "minimum": 0.1}]
    document = project([emitter(rays=1024), beam], [detector("D", 10.0, 5.0)], targets)
    result = run(document).result
    by_id = {t.target_id: t for t in result.targets}
    assert by_id["R"].status is Status.PASS and by_id["R"].value > 0.0  # the beam's metric, from the beam alone
    assert by_id["P"].status is Status.PASS
    observation = result.observations[0]
    assert len(observation.arrivals) == 1 and observation.arrivals[0].beam.gaussian.valid
    rays_W = observation.sampled_light[0].power_W
    assert observation.total_power_W == pytest.approx(rays_W + observation.arrivals[0].detected_power_W)
    assert not result.interference and not observation.superpositions  # no phase relation with the rays; one beam alone
    grid = RunResults(result=result).irradiance_map("D", samples=16, half_extent_mm=5.0, center_uv_mm=[0.0, 0.0])
    assert grid["coherent_W_per_mm2"] is None
    laser_only = RunResults(result=result).irradiance_map("D", samples=16, half_extent_mm=5.0, center_uv_mm=[0.0, 0.0], source_id="LASER")
    rays_only = RunResults(result=result).irradiance_map("D", samples=16, half_extent_mm=5.0, center_uv_mm=[0.0, 0.0], source_id="EM")
    assert laser_only["coherent_W_per_mm2"] is not None and rays_only["coherent_reason"] == "INDEPENDENT_SAMPLES"


def test_a_thin_lens_images_the_emitter_inverted_at_unit_magnification():
    # object and image at 2f of an ideal thin lens: each ray through the lens lands at minus its launch point (within 1 um)
    from declared_helpers import declared_component, thin_lens_model, with_models

    from os_optics.scene.model import EmitterG

    surface = {"shape": "RECTANGLE", "width_mm": 4.0, "height_mm": 2.0}
    document = project([emitter(surface=surface), declared_component("L", "lens", "thin", (100.0, 0.0, 0.0))],
                       [plane("image", (200.0, 0.0, 0.0), radius_mm=30.0)])
    result = run(with_models(document, [thin_lens_model("thin", "50")], [])).result
    light = result.observations[0].sampled_light[0]
    assert light.counted_crossing_count > 0
    z, x, y = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])
    points, _, _ = local_samples(EmitterG("EM", "EM", np.zeros(3), z, x, y, ((550.0, 1.0),), 1.0, "air", "rectangle", 2.0, 1.0, "LAMBERTIAN", RAYS, 1))
    u, v = np.array(light.crossings.u_mm), np.array(light.crossings.v_mm)
    through_lens = np.array(light.crossings.counted) & (np.hypot(u, v) < 5.0)
    assert int(through_lens.sum()) > 40
    launched = points[np.array(light.crossings.sample_index)[through_lens]]
    assert np.max(np.abs(u[through_lens] + launched[:, 0])) < 1e-3
    assert np.max(np.abs(v[through_lens] + launched[:, 1])) < 1e-3
    assert np.max(np.abs(u[through_lens])) <= 2.0 and np.max(np.abs(v[through_lens])) <= 1.0
    # the other rays pass beside the lens, straight on, far outside the image
    beside = np.array(light.crossings.counted) & ~through_lens
    assert not beside.any() or float(np.min(np.hypot(u[beside], v[beside]))) > 15.0
    assert result.power_budget.lines[0].conserved


def test_a_stop_passes_the_rays_inside_its_opening():
    from declared_helpers import declared_component, stop_model, with_models

    document = project([emitter(), declared_component("S", "aperture", "stop", (10.0, 0.0, 0.0))], [plane("D", (20.0, 0.0, 0.0), radius_mm=30.0)])
    result = run(with_models(document, [stop_model("stop", plane_z_mm="0", opening_radius_mm="2", edge_radius_mm="50")], [])).result
    # the opening's radius 2 mm at 10 mm, sin^2(theta) = 4 / 104; a plate of radius 50 mm stops the rest up to its rim
    assert result.observations[0].sampled_light[0].collection_fraction == pytest.approx(4.0 / 104.0, abs=COLLECTION_ABS_TOL)
    assert result.power_budget.lines[0].sinks["CLIPPED"] > 0.9


def test_a_rated_surface_reads_the_rays_power_and_no_fake_irradiance():
    from declared_helpers import declared_component, nbk7, with_models
    from test_power_exposure_ratings import _rating, _window

    rating = _rating("CW", 1000.0, "W/cm2", beam_diameter_mm=1.0, beam_diameter_definition="1/E2", polarization_condition=None)
    document = project([emitter(rays=256, wavelength_nm=1064.0, power_w=2.0, surface={"shape": "DISC", "diameter_mm": 1.0}),
                        declared_component("W", "window", "win", (20.0, 0.0, 0.0))], [plane("screen", (40.0, 0.0, 0.0), radius_mm=20.0)])
    result = run(with_models(document, [_window([rating])], [nbk7()])).result
    front = next(s for s in result.exposures.surfaces if s.surface_id == "W/S1")
    assert front.incident_power_W == pytest.approx(front.hits[0].incident_power_W)
    assert front.incident_power_W > 0.0 and front.peak_irradiance_W_per_cm2 is None and not front.complete
    assert front.rating_status != "WITHIN_RATING"


def test_a_network_export_and_a_paraxial_beam_refuse_an_emitter():
    from os_optics.local import export_linear_optical_network

    from local_helpers import catalog_document as catalog
    from os_optics.local.paraxial import ParaxialError, paraxial_analysis
    from os_optics.network import NetworkExportError

    with pytest.raises(NetworkExportError) as refused:
        export_linear_optical_network(project([emitter(rays=16)], [detector("D", 10.0, 5.0)]))
    assert refused.value.problems[0].code.value == "OUT_OF_MODEL_SCOPE"
    document = project([emitter(rays=16), element("L", "lens", PCX_F50, (50.0, 0.0, 0.0))], [detector("D", 100.0, 5.0)])
    with pytest.raises(ParaxialError, match="extended emitter"):
        paraxial_analysis(document, {"components": ["L"], "wavelength_nm": 550.0, "source_id": "EM"}, catalog=catalog())


@pytest.mark.parametrize("change, expected", [
    ({"sampling": {"sample_count": 100, "seed": 1}}, "components[0].source_model.sampling: "),
    ({"emitting_surface": {"shape": "DISC", "diameter_mm": -1.0}}, "components[0].source_model.emitting_surface.diameter_mm: "),
    ({"waist_radius_mm": 1.0}, "components[0].source_model.waist_radius_mm: "),
    ({"power_w": None}, "components[0].source_model: "),
])
def test_a_wrong_emitter_is_refused_at_the_field_the_document_spells(change, expected):
    from os_optics.local import LocalRunError, load_project

    document = project([emitter()], [detector("D", 10.0, 5.0)])
    document["components"][0]["source_model"].update(change)
    with pytest.raises(LocalRunError) as refused:
        load_project(document)
    assert expected in str(refused.value)
    assert "EXTENDED_EMITTER." not in str(refused.value)


def test_gaussian_only_result_has_no_trace_of_this_feature():
    result = run(project([source("LASER", (-5.0, 0.0, 0.0))], [detector("D", 10.0, 5.0)])).result
    dumped = result.model_dump(mode="json")
    assert "sampled_sources" not in dumped and "sampled_light" not in dumped["observations"][0]
    assert ValidationResult.model_validate(dumped).sampled_sources == ()


# ---------------------------------------------------------------------------------------------- uniformity and exports


def test_roi_uniformity_from_the_map_equals_a_calculation_from_the_exported_rows():
    # the supported export alone gives the same number as the native metric
    doc = project([emitter(rays=MAP_RAYS, surface={"shape": "RECTANGLE", "width_mm": 4.0, "height_mm": 4.0})], [plane("D", (30.0, 0.0, 0.0), radius_mm=60.0)])
    results = RunResults(result=run(doc).result)
    roi = {"shape": "rectangle", "center_uv_mm": [0.0, 0.0], "half_width_mm": [8.0, 8.0]}
    grid = results.irradiance_map("D", samples=8, half_extent_mm=8.0, center_uv_mm=[0.0, 0.0], roi=roi)
    uniformity = grid["uniformity"]
    assert uniformity["cell_count"] == 64
    rows = list(csv.DictReader(io.StringIO(results.export("sampled-crossings", "csv", monitor="D").decode("utf-8"))))
    u, v, power = (np.array([float(r[k]) for r in rows]) for k in ("u_mm", "v_mm", "power_W"))
    keep = (u >= -8.0) & (u < 8.0) & (v >= -8.0) & (v < 8.0)
    i, j = np.floor((u[keep] + 8.0) / 2.0).astype(int), np.floor((v[keep] + 8.0) / 2.0).astype(int)
    bins = np.bincount(j * 8 + i, weights=power[keep], minlength=64) / 4.0
    assert uniformity["min_over_mean"] == pytest.approx(float(bins.min() / bins.mean()), rel=1e-9)
    assert uniformity["std_over_mean"] == pytest.approx(float(bins.std() / bins.mean()), rel=1e-9)
    # cos^4 over 8 mm at 30 mm: about 1 - 2 (r / d)^2 at the corner, r = 11.3 mm
    assert 0.6 < uniformity["min_over_mean"] < 1.0
    assert "half_samples" in uniformity and uniformity["mean_crossings_per_cell"] > 15.0
    assert grid["sample_counts"][4][4] > 15


def test_exports_hold_the_crossings_and_the_collection():
    results = RunResults(result=run(project([emitter(rays=64)], [detector("D", 10.0, 5.0)])).result)
    rows = list(csv.DictReader(io.StringIO(results.export("sampled-light", "csv").decode("utf-8"))))
    assert rows[0]["observation_id"] == "D" and rows[0]["source_id"] == "EM"
    assert float(rows[0]["collection_fraction"]) == pytest.approx(results.sampled_light()[0]["collection_fraction"])
    assert results.describe()["available"]["irradiance_map"] is True
    assert results.describe()["available"]["sampled_sources"][0]["sample_count"] == 64
    assert "crossings" not in results.monitor("D")["sampled_light"][0]
    assert results.export("irradiance", "npz", monitor="D", samples=4)[:2] == b"PK"


@pytest.mark.parametrize("roi", [{"shape": "triangle"}, {"shape": "circle", "center_uv_mm": [0.0, 0.0], "radius_mm": -1.0},
                                 {"shape": "rectangle", "center_uv_mm": [1e6, 0.0], "half_width_mm": [1.0, 1.0]}])
def test_a_bad_region_is_refused(roi):
    results = RunResults(result=run(project([emitter(rays=64)], [detector("D", 10.0, 5.0)])).result)
    assert results.irradiance_map("D", roi=roi)["reason"] == "ROI_INVALID"


# --------------------------------------------------------------------------------------------------------------- the view


def test_view_shows_the_emitter_as_an_area_and_sample_rays():
    doc = project([emitter(rays=256, surface={"shape": "RECTANGLE", "width_mm": 6.0, "height_mm": 2.0}), element("L", "lens", PCX_F50, (50.0, 0.0, 0.0))],
                  [detector("D", 100.0, 8.0)])
    outcome = run(doc)
    scene = outcome.optical_view
    em = next(e for e in scene.elements if e.element_id == "EM")
    assert em.geometry.emitting_area.shape == "RECTANGLE"
    assert em.geometry.emitting_area.size_x_mm == 6.0 and em.geometry.emitter_radius_mm == pytest.approx(math.hypot(6.0, 2.0) / 2.0)
    paths = [p for p in scene.beam_paths if p.source_element_id == "EM"]
    assert DISPLAY_SAMPLES <= len(paths)
    assert all(not p.pupil_rays for p in paths)
    assert len({tuple(p.chief_ray.points_mm[0]) for p in paths}) == len(paths)  # each from its own point of the area
    from os_optics.optical_view.section import optical_section, scene_section_elements, section_svg

    section = optical_section(scene_section_elements(scene), "L", "XZ", beam_paths=scene.beam_paths)
    outline = section["emitters"][0]
    assert outline["element_id"] == "EM"
    assert {round(p[1], 6) for p in outline["outline"]} == {-50.0}  # the emitter's plane, 50 mm before the lens
    assert sorted({abs(round(p[0], 6)) for p in outline["outline"]})[-1] in (3.0, 1.0)  # its true half size along the section
    assert "emitting surface" in section_svg(section)
    assert optical_section(scene_section_elements(scene), "EM", "XZ")["emitters"][0]["element_id"] == "EM"
    from os_optics.optical_view.section import system_sections
    from os_optics.viewer.section_render import section_png

    whole = system_sections(scene_section_elements(scene), "XZ", beam_paths=scene.beam_paths, observation_planes=scene.observation_planes)
    assert len(whole) == 1 and whole[0]["reference_element_id"] == "L" and whole[0]["emitters"][0]["element_id"] == "EM"
    assert whole[0]["bounds"]["optics_z_mm"][0] == -50.0  # the frame holds the emitter
    assert section_png(whole[0])[1:4] == b"PNG"
    alone = run(project([emitter(rays=64)], [detector("D", 10.0, 5.0)])).optical_view
    only = system_sections(scene_section_elements(alone), "XZ", beam_paths=alone.beam_paths, observation_planes=alone.observation_planes)
    assert [s["reference_element_id"] for s in only] == ["EM"] and only[0]["elements"] == []
