"""The scientific result layer of a local run (``os_optics.results``, ``os-optics results`` / ``export``,
``json_api.run_results``): every number comes from the run's own files, the tables agree with the Engine result they are
read from, exports round-trip, and the wavefront agrees with closed-form and independently traced references.

References never call the production wavefront code: a diverging Gaussian launch is checked against the sphere about its
own centre of curvature (zero OPD, closed form), a plane window against a flat wavefront, and a plano-convex singlet
against a meridional ray trace written here and against the Seidel thin-lens spherical aberration W040."""

from __future__ import annotations

import csv
import io
import json
import math

import numpy as np
import pytest

from os_optics import EngineConfiguration, ValidationRequest, validate_with_ray_paths
from os_optics.builders import aperture_stop, laser, observation_plane, singlet_lens, thin_lens, window
from os_optics.cli import main
from os_optics.contracts import ConstantIndexMaterial, ProjectSnapshot
from os_optics.local import example_project_of_kind, json_api, run_optical_project
from os_optics.local.artifacts import run_artifact_files
from os_optics.results import EXPORT_KINDS, ResultsError, columns, load_run, results_from_engine
from os_optics.results.irradiance import irradiance_map

from local_helpers import catalog_document


def _reference_catalog():
    """Fixture-only optics assume zero absorption while testing result numerics, not a claim about published glass."""
    document = catalog_document()
    document["schema_version"] = "3.6"  # Required for material bulk-loss evidence.
    for material in document["materials"]:
        material["bulk_loss"] = {"kind": "INTERNAL_TRANSMITTANCE",
                                 "wavelength_nm": [material["valid_band_nm"]["min_nm"], material["valid_band_nm"]["max_nm"]],
                                 "internal_transmittance": ["1", "1"], "thickness_mm": ["1", "1"],
                                 "basis": "Assumed lossless synthetic test fixture."}
    return document

GLASS = ConstantIndexMaterial(material_id="glass15", valid_wavelength_nm=(100.0, 20000.0), refractive_index=1.5,
                              absorption_coefficient_per_mm=0.0)  # Test-only reference glass.
X = (1.0, 0.0, 0.0)
NM = 1e-6


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """Reference runs written to run directories: a focus, a branching interferometer and a two-source dichroic design."""
    out = {}
    for kind in ("focus", "two_beam_interference", "dichroic_routing", "folded_expander"):
        directory = tmp_path_factory.mktemp(kind)
        run = run_optical_project(example_project_of_kind(kind), catalog=_reference_catalog(), output_dir=directory)
        out[kind] = (run, directory)
    return out


def _engine(components, observations=()):
    request = ValidationRequest(project=ProjectSnapshot(components=components, observations=observations, materials=(GLASS,)),
                                configuration=EngineConfiguration(run_layout=False, run_control=False))
    result, trace = validate_with_ray_paths(request)
    return results_from_engine(result, trace)


def _json_safe(value):
    json.dumps(value, allow_nan=False)
    return value


# ---------------------------------------------------------------------------------------------------- loading / describe
@pytest.mark.parametrize("source", ["directory", "files", "run"])
def test_a_run_loads_from_its_directory_its_files_or_the_run_itself(runs, source):
    run, directory = runs["focus"]
    given = {"directory": directory, "files": run_artifact_files(run), "run": run}[source]
    results = load_run(given)
    described = _json_safe(results.describe())
    # the results are read as stored; the files are verified when asked
    assert described["integrity"] == {"checked": False, "consistent": None, "problems": []}
    verified = load_run(given, verify=True).describe()
    assert verified["integrity"] == {"checked": True, "consistent": True, "problems": []}
    assert {k: v for k, v in verified.items() if k != "integrity"} == {k: v for k, v in described.items() if k != "integrity"}
    assert described["run"]["result_hash"] == run.result.manifest.result_hash and described["run"]["trace_hash"] == run.result.manifest.trace_hash
    assert described["available"]["monitors"] == ["focal_plane"] and described["available"]["ray_data"] is True
    assert described["available"]["wavefront"] is True and described["available"]["power_budget"] is True
    assert set(described["conventions"]) >= {"plane_frame", "polarization", "first_order", "power", "wavefront", "not_modelled"}
    assert [f["name"] for f in described["files"]] == [name for name in run.artifacts.manifest and
                                                      [f["name"] for f in run.artifacts.manifest["files"]]]
    assert set(described["exports"]["kinds"]) == set(EXPORT_KINDS)


def test_a_tampered_run_still_loads_and_says_it_does_not_verify(runs):
    run, _ = runs["focus"]
    files = run_artifact_files(run)
    files["optical_scene.json"] = files["optical_scene.json"] + b" "
    described = load_run(files, verify=True).describe()
    assert described["integrity"]["consistent"] is False and described["integrity"]["problems"]
    unasked = load_run(files).describe()
    assert unasked["integrity"] == {"checked": False, "consistent": None, "problems": []}
    assert {k: v for k, v in unasked.items() if k != "integrity"} == {k: v for k, v in described.items() if k != "integrity"}


def test_a_run_without_ray_data_answers_everything_else_and_refuses_rays(runs):
    run, _ = runs["focus"]
    files = run_artifact_files(run)
    files.pop("ray_data.json")
    results = load_run(files, verify=False)
    assert results.describe()["available"]["ray_data"] is False and results.monitors()
    with pytest.raises(ResultsError) as exc:
        results.rays()
    assert exc.value.code == "RAY_DATA_NOT_AVAILABLE"
    assert results.wavefront("focal_plane", "SRC#0")["reason"] == "RAY_DATA_NOT_AVAILABLE"


def test_unreadable_runs_and_unknown_names_are_structured_errors(runs, tmp_path):
    with pytest.raises(ResultsError) as exc:
        load_run(tmp_path / "missing")
    assert exc.value.code == "RUN_NOT_READABLE"
    results = load_run(runs["focus"][1])
    for call, code in ((lambda: results.monitor("nope"), "MONITOR_NOT_FOUND"), (lambda: results.paths("nope"), "PATH_NOT_FOUND"),
                       (lambda: results.export("nope"), "EXPORT_INVALID"), (lambda: results.export("monitors", "npz"), "EXPORT_INVALID"),
                       (lambda: results.export("monitors", "csv", path="SRC#0"), "EXPORT_INVALID")):
        with pytest.raises(ResultsError) as exc:
            call()
        assert exc.value.code == code


# ---------------------------------------------------------------------------------------------------- monitors / paths
def test_monitors_carry_the_resolved_frame_and_every_arrival(runs):
    run, directory = runs["dichroic_routing"]
    results = load_run(directory)
    monitors = {m["observation_id"]: m for m in _json_safe(results.monitors())}
    for observation in run.result.observations:
        m = monitors[observation.observation_id]
        assert m["plane"] == observation.plane.model_dump(mode="json")
        assert m["total_power_W"] == observation.total_power_W and m["arrival_count"] == len(observation.arrivals)
        full = _json_safe(results.monitor(observation.observation_id))
        assert full["arrivals"] == [a.model_dump(mode="json") for a in observation.arrivals]
        assert full["superpositions"] == [s.model_dump(mode="json") for s in observation.superpositions]
        assert {t["target_id"] for t in full["targets"]} == {t.target_id for t in run.result.targets if t.observation_id == observation.observation_id}


def test_the_path_tree_names_parents_children_and_crossed_planes(runs):
    run, directory = runs["two_beam_interference"]
    paths = {p["path_id"]: p for p in _json_safe(load_run(directory).paths())}
    assert paths["SRC#0"]["children"] == ["SRC#0.0"] and paths["SRC#0.0"]["parent_path_id"] == "SRC#0"
    assert paths["SRC#0"]["depth"] == 0 and paths["SRC#0.0"]["depth"] == 1
    assert paths["SRC#0"]["observations"] == ["crossing_plane"] == paths["SRC#0.0"]["observations"]
    for p in run.result.paths:
        assert paths[p.path_id]["optical_path_length_mm"] == p.optical_path_length_mm
        assert paths[p.path_id]["power_start_W"] == p.segments[0].power_start_W and paths[p.path_id]["final_power_W"] == p.final_power_W


def test_power_budget_rows_balance_per_line(runs):
    run, directory = runs["two_beam_interference"]
    budget = _json_safe(load_run(directory).power_budget())
    assert budget["available"] is True and budget["conserved"] is True
    for line in run.result.power_budget.lines:
        rows = [r for r in budget["rows"] if r["source_id"] == line.source_id and r["spectral_index"] == line.spectral_index]
        (inp,) = [r["power_W"] for r in rows if r["category"] == "INPUT"]
        (res,) = [r["power_W"] for r in rows if r["category"] == "RESIDUAL"]
        sinks = sum(r["power_W"] for r in rows if r["category"] == "SINK")
        assert inp == pytest.approx(sinks + res, rel=1e-12)
        assert {r["name"] for r in rows if r["category"] == "OBSERVED"} == set(line.observed_W)


# ---------------------------------------------------------------------------------------------------------------- rays
@pytest.mark.parametrize("kind", ["focus", "two_beam_interference", "folded_expander"])
def test_ray_rows_at_a_monitor_reproduce_the_arrival_statistics(runs, kind):
    run, directory = runs[kind]
    rays = _json_safe(load_run(directory).rays())
    for observation in run.result.observations:
        for arrival in observation.arrivals:
            rows = [r for r in rays if r["observation_id"] == observation.observation_id and r["path_id"] == arrival.beam.path_id
                    and r["crossing_index"] == arrival.crossing_index]
            chief = [r for r in rows if r["chief"]]
            pupil = [r for r in rows if not r["chief"]]
            assert len(chief) == 1 and len(pupil) == arrival.pupil_rays_arrived > 0
            assert (chief[0]["u_mm"], chief[0]["v_mm"]) == pytest.approx(arrival.uv_mm, abs=1e-9)
            assert chief[0]["opl_mm"] == pytest.approx(arrival.beam.optical_path_length_mm, rel=1e-12)
            centroid = (np.mean([r["u_mm"] for r in pupil]), np.mean([r["v_mm"] for r in pupil]))
            assert centroid == pytest.approx(arrival.pupil_centroid_uv_mm, abs=1e-9)
            assert all(r["counted"] is True and r["power_weight"] == pytest.approx(1.0 / arrival.pupil_ray_count) for r in pupil)
            d = np.array([[r["direction_x"], r["direction_y"], r["direction_z"]] for r in pupil])
            uvn = np.array([[r["direction_u"], r["direction_v"], r["direction_n"]] for r in pupil])
            plane = observation.plane
            assert np.allclose(uvn, d @ np.array([plane.u_axis, plane.v_axis, plane.normal]).T, atol=1e-12)


def test_ray_filters_select_one_monitor_path_and_crossing(runs):
    results = load_run(runs["two_beam_interference"][1])
    rows = results.rays(monitor="crossing_plane", path="SRC#0.0", crossing_index=0)
    assert rows and {(r["path_id"], r["crossing_index"]) for r in rows} == {("SRC#0.0", 0)}
    assert results.rays(monitor="crossing_plane", path="SRC#0.0", crossing_index=1) == []


# ----------------------------------------------------------------------------------------------------------- exports
def _parse_cell(text, reference):
    if text == "":
        return None
    if isinstance(reference, bool):
        return text == "true"
    if isinstance(reference, int):
        return int(text)
    if isinstance(reference, float):
        return float(text)
    if isinstance(reference, list):
        return [type(reference[0])(x) if reference else x for x in _unformula(text).split(";")]
    if isinstance(reference, dict):  # the value_basis column: compact JSON
        return json.loads(text)
    return _unformula(text)


def _unformula(text):
    """A text cell a spreadsheet would read as a formula is exported with a leading ' ."""

    return text[1:] if text[:1] == "'" and text[1:2] in ("=", "+", "-", "@", "\t", "\r") else text


@pytest.mark.parametrize("kind", ["monitors", "arrivals", "rays", "paths", "segments", "power-budget", "superpositions", "metrics"])
def test_csv_jsonl_and_json_exports_round_trip_to_the_api_rows(runs, kind):
    results = load_run(runs["two_beam_interference"][1])
    rows, _ = results.table(kind)
    assert rows, kind
    names = [c["name"] for c in columns(kind)]
    parsed = list(csv.reader(io.StringIO(results.export(kind, "csv").decode("utf-8"))))
    assert parsed[0] == names and len(parsed) == len(rows) + 1
    for line, row in zip(parsed[1:], rows):
        assert list(row) == names
        for text, name in zip(line, names):
            assert _parse_cell(text, row[name]) == (row[name] if row[name] not in ([], "") else None), (kind, name)  # '' = undefined
    assert [json.loads(line) for line in results.export(kind, "jsonl").decode("utf-8").splitlines()] == rows
    document = json.loads(results.export(kind, "json"))
    assert document["schema"] == "opensci.results_export" and document["kind"] == kind and document["rows"] == rows
    assert document["columns"] == columns(kind) and document["run"]["result_hash"] == results.result.manifest.result_hash
    assert results.export(kind, "csv") == results.export(kind, "csv")  # deterministic


def test_column_names_carry_units_and_every_column_is_documented():
    for kind in EXPORT_KINDS:
        cols = columns(kind)
        assert len({c["name"] for c in cols}) == len(cols)
        for c in cols:
            assert c["description"]
            if c["unit"] in ("mm", "W", "nm", "deg", "rad", "waves"):
                assert c["name"].endswith("_" + c["unit"]), c


# --------------------------------------------------------------------------------------------------------- irradiance
def test_irradiance_map_integrates_to_the_arriving_power_and_exports_npz(runs):
    run, directory = runs["two_beam_interference"]
    results = load_run(directory)
    (observation,) = run.result.observations
    (superposition,) = observation.superpositions
    grid = _json_safe(results.irradiance_map("crossing_plane", samples=201))
    assert grid["status"] == "OK" and sorted(grid["path_ids"]) == ["SRC#0", "SRC#0.0"]
    assert grid["grid_power_incoherent_W"] == pytest.approx(superposition.incoherent_power_W, rel=1e-3)
    # the crossing beams make sub-micrometre fringes: a whole-beam grid cannot resolve them, so the coherent map is withheld
    (pair,) = run.result.interference
    period_mm = pair.fringe_period_nm * 1e-6
    assert grid["coherent_W_per_mm2"] is None and grid["coherent_reason"] == "FRINGES_UNDERSAMPLED"
    assert grid["sampling"]["shortest_fringe_period_mm"] == pytest.approx(period_mm, rel=1e-9)
    # a grid that resolves them: the arms cross the catalog splitter's coating and coated mirrors, whose group delays the
    # catalog does not state -- their degree of coherence is not known, so the coherent map is withheld; at one frequency
    # (those passes taken alike: the records cleared) the map shows fringes of exactly that period along the fringe direction
    half = grid["sampling"]["resolving_half_extent_mm"] / 4.0
    withheld = results.irradiance_map("crossing_plane", half_extent_mm=half, samples=201)
    assert withheld["coherent_W_per_mm2"] is None and withheld["coherent_reason"] == "COHERENCE_UNDEFINED"
    single = run.result.model_copy(update={"observations": tuple(
        o.model_copy(update={"arrivals": tuple(a.model_copy(update={"group_delay_unknown_at": ()}) for a in o.arrivals)})
        for o in run.result.observations)})
    zoom = irradiance_map(single, "crossing_plane", half_extent_mm=half, samples=201)
    assert zoom["coherent_reason"] is None and zoom["sampling"]["grid_step_mm"] <= period_mm / 4.0
    assert zoom["sampling"]["samples_basis"] == "STATED"
    # with no samples stated the count is the one that resolves the fringes (four points per period), not 101: the coherent
    # map is there, and its measured period is the pair's
    wide = 3.0 * grid["sampling"]["resolving_half_extent_mm"]  # (three times what 101 points resolve)
    auto = irradiance_map(single, "crossing_plane", half_extent_mm=wide)
    count = auto["grid"]["shape"][0]
    assert auto["sampling"]["samples_basis"] == "FRINGE_PERIOD" and 101 < count <= 1001 and auto["grid"]["shape"] == [count, count]
    assert auto["coherent_reason"] is None and auto["sampling"]["grid_step_mm"] <= period_mm / 4.0
    assert auto["sampling"]["grid_step_mm"] > period_mm / 4.0 * (count - 3) / (count - 1)  # (the smallest count that does)
    auto_profile = np.array(auto["coherent_W_per_mm2"]).mean(axis=0) if abs(pair.fringe_direction[0]) else np.array(auto["coherent_W_per_mm2"]).mean(axis=1)
    auto_spectrum = np.abs(np.fft.rfft(auto_profile - auto_profile.mean()))
    assert 1.0 / (np.argmax(auto_spectrum) / (len(auto_profile) * auto["sampling"]["grid_step_mm"])) == pytest.approx(period_mm, rel=0.05)
    profile = np.array(zoom["coherent_W_per_mm2"]).mean(axis=0) if abs(pair.fringe_direction[0]) else np.array(zoom["coherent_W_per_mm2"]).mean(axis=1)
    spectrum = np.abs(np.fft.rfft(profile - profile.mean()))
    step = zoom["sampling"]["grid_step_mm"]
    measured = 1.0 / (np.argmax(spectrum) / (len(profile) * step))
    assert measured == pytest.approx(period_mm, rel=0.05)
    data = results.export("irradiance", "npz", monitor="crossing_plane", samples=21)
    assert data == results.export("irradiance", "npz", monitor="crossing_plane", samples=21)
    arrays = np.load(io.BytesIO(data))
    small = results.irradiance_map("crossing_plane", samples=21)
    assert np.array_equal(arrays["u_mm"], small["grid"]["u_mm"]) and "irradiance_coherent_W_per_mm2" not in arrays.files  # withheld: undersampled
    assert np.array_equal(arrays["irradiance_incoherent_W_per_mm2"], np.array(small["incoherent_W_per_mm2"]))
    rows = list(csv.DictReader(io.StringIO(results.export("irradiance", "csv", monitor="crossing_plane", samples=21).decode())))
    assert len(rows) == 21 * 21 and float(rows[22]["u_mm"]) == small["grid"]["u_mm"][1] and float(rows[22]["v_mm"]) == small["grid"]["v_mm"][1]


def test_irradiance_without_a_wavelength_is_the_sum_over_the_wavelengths_that_arrive(runs):
    """Two wavelengths on one plane: lines of different wavelengths are mutually incoherent, so the map without a wavelength
    is the sum of the two lines' maps on one grid. Reference: the two single-wavelength maps on the same grid, added here."""

    results = load_run(runs["dichroic_routing"][1])
    monitor = next(m["observation_id"] for m in results.monitors() if len({a["wavelength_nm"] for a in m["arrivals"]}) > 1)
    total = results.irradiance_map(monitor)
    assert total["status"] == "OK" and total["wavelength_nm"] is None and len(total["wavelengths_nm"]) == 2
    assert total["scope"]["wavelength_nm"] is None and total["composition_over_wavelengths"] == "INCOHERENT_SUM"
    assert total["grid"]["shape"] == [101, 101] and total["sampling"]["samples_basis"] == "DEFAULT"
    window = {"center_uv_mm": total["grid"]["center_uv_mm"], "half_extent_mm": total["grid"]["half_extent_mm"], "samples": 101}
    lines = [results.irradiance_map(monitor, wavelength_nm=w, **window) for w in total["wavelengths_nm"]]
    assert all(line["status"] == "OK" and line["wavelength_nm"] == w for line, w in zip(lines, total["wavelengths_nm"]))
    again = results.irradiance_map(monitor, **window)
    summed = np.array(lines[0]["incoherent_W_per_mm2"]) + np.array(lines[1]["incoherent_W_per_mm2"])
    assert np.array_equal(np.array(again["incoherent_W_per_mm2"]), summed)
    assert again["grid_power_incoherent_W"] == pytest.approx(lines[0]["grid_power_incoherent_W"] + lines[1]["grid_power_incoherent_W"], rel=1e-12)
    assert np.array_equal(np.array(total["incoherent_W_per_mm2"]), summed) and total["grid"] == again["grid"]
    assert [row["wavelength_nm"] for row in again["per_wavelength"]] == total["wavelengths_nm"]
    for row, line in zip(again["per_wavelength"], lines):
        assert row["path_ids"] == line["path_ids"] and row["grid_power_incoherent_W"] == line["grid_power_incoherent_W"]
        assert row["map_complete"] == line["map_complete"] and row["untraced_power_W"] == line["untraced_power_W"]
    assert again["path_ids"] == lines[0]["path_ids"] + lines[1]["path_ids"]
    assert again["map_complete"] == (lines[0]["map_complete"] and lines[1]["map_complete"])
    # a stated wavelength gives that line's map alone, as before
    one = results.irradiance_map(monitor, wavelength_nm=total["wavelengths_nm"][0], samples=11)
    assert one["status"] == "OK" and one["wavelength_nm"] == total["wavelengths_nm"][0] and "per_wavelength" not in one


# ---------------------------------------------------------------------------------------------------------- wavefront
def test_a_plane_window_leaves_a_collimated_wavefront_flat():
    results = _engine((laser("L", wavelength_nm=633.0, waist_radius_mm=1.0),
                       window("W", thickness_mm=5.0, clear_radius_mm=10.0, material="glass15", position=(50.0, 0.0, 0.0))),
                      (observation_plane("D", position=(100.0, 0.0, 0.0), normal=X),))
    wave = _json_safe(results.wavefront("D", "L#0"))
    assert wave["status"] == "OK" and wave["ray_count"] == wave["pupil_ray_count"] == 96
    assert wave["rms_waves"] < 1e-9 and wave["pv_waves"] < 1e-9


def test_a_diverging_gaussian_launch_is_a_sphere_about_its_centre_of_curvature():
    """Rays launched along the normals of the source's Gaussian phase front (radius R = d + z_R^2 / d for a waist d behind
    the source) are a perfect spherical wave about the point R behind the source: zero OPD against that sphere."""
    wavelength, w0, d = 633.0, 0.05, 50.0
    z_r = math.pi * w0**2 / (wavelength * NM)
    radius = d + z_r**2 / d
    results = _engine((laser("L", wavelength_nm=wavelength, waist_radius_mm=w0, waist_position_z_mm=-d),),
                      (observation_plane("D", position=(100.0, 0.0, 0.0), normal=X),))
    wave = results.wavefront("D", "L#0", reference="sphere", center=(-radius, 0.0, 0.0))
    assert wave["status"] == "OK" and wave["reference"]["radius_mm"] == pytest.approx(100.0 + radius)
    assert wave["rms_waves"] < 1e-8 and wave["pv_waves"] < 1e-8
    focus = results.wavefront("D", "L#0", reference="sphere", center="ray_focus")  # the rays meet exactly there
    assert focus["reference"]["center_mm"] == pytest.approx([-radius, 0.0, 0.0], abs=1e-6) and focus["rms_waves"] < 1e-8
    flat = results.wavefront("D", "L#0")  # against a plane: the defocus of a sphere of radius 100 + R
    rho2 = np.array([(r["u_mm"] ** 2 + r["v_mm"] ** 2) for r in flat["rays"]])
    extra = np.sqrt((100.0 + radius) ** 2 + rho2) - (100.0 + radius)  # closed form: |hit - centre| - |chief hit - centre|
    expected = (extra - extra.mean()) / (wavelength * NM)
    assert np.allclose([r["opd_waves"] for r in flat["rays"]], expected, atol=1e-6)


def _meridional_opl(h, *, vertex, rc, t, n, z_plane, center, wavelength):
    """Independent reference: exact meridional trace of a ray parallel to the axis at height ``h`` from the source plane
    (x = 0) through a plano-convex lens (sphere of radius ``rc`` with its vertex at ``vertex``, flat back at
    ``vertex + t``, index ``n``) to the sphere through (z_plane, 0) about (center, 0); returns the optical path in mm."""
    c = np.array([vertex + rc, 0.0])
    p1 = np.array([c[0] - math.sqrt(rc**2 - h**2), h])
    d = np.array([1.0, 0.0])
    normal = (p1 - c) / rc  # outward, against the incoming ray
    cos_i = -float(d @ normal)
    mu = 1.0 / n
    cos_t = math.sqrt(1.0 - mu**2 * (1.0 - cos_i**2))
    d1 = mu * d + (mu * cos_i - cos_t) * normal
    s1 = (vertex + t - p1[0]) / d1[0]
    p2 = p1 + s1 * d1
    sin_out = n * d1[1]
    d2 = np.array([math.sqrt(1.0 - sin_out**2), sin_out])
    radius = center - z_plane
    rel = p2 - np.array([center, 0.0])
    b = float(d2 @ rel)
    s2 = -b - math.sqrt(b * b - (float(rel @ rel) - radius**2))
    return p1[0] + n * s1 + s2


def test_a_spherical_singlet_matches_an_independent_trace_and_the_seidel_spherical_aberration():
    rc, t, n, wavelength, beam = 51.5, 1.0, 1.5, 633.0, 2.0
    mid = 50.0
    vertex = mid - t / 2.0
    # paraxial focus of the thick lens (curved side first): back focal distance f - t / n with f = rc / (n - 1)
    f = rc / (n - 1.0)
    focus = vertex + t + f - t / n
    z_plane = focus - 20.0
    lens = singlet_lens("S", r1_mm=rc, r2_mm=math.inf, thickness_mm=t, clear_radius_mm=12.0, material="glass15", position=(mid, 0.0, 0.0))
    results = _engine((laser("L", wavelength_nm=wavelength, waist_radius_mm=beam), lens),
                      (observation_plane("D", position=(z_plane, 0.0, 0.0), normal=X),))
    wave = results.wavefront("D", "L#0", reference="sphere", center=(focus, 0.0, 0.0))
    assert wave["status"] == "OK" and wave["reference"]["converging"] is True and wave["ray_count"] == 96
    heights = np.array([math.hypot(r["pupil_x"], r["pupil_y"]) * beam for r in wave["rays"]])
    chief = _meridional_opl(0.0, vertex=vertex, rc=rc, t=t, n=n, z_plane=z_plane, center=focus, wavelength=wavelength)
    reference = np.array([_meridional_opl(h, vertex=vertex, rc=rc, t=t, n=n, z_plane=z_plane, center=focus, wavelength=wavelength)
                          for h in heights]) - chief
    reference -= reference.mean()
    assert np.allclose([r["opd_mm"] for r in wave["rays"]], reference, atol=1e-10)
    # Seidel: W040 = S_I / 8 with S_I = h^4 K^3 / 4 [(n/(n-1))^2 + (n+2)/(n(n-1)^2) (B + 2(n^2-1)/(n+2) C)^2 - n/(n+2) C^2],
    # shape B = 1 (curved side to the collimated beam), conjugate C = -1 (object at infinity). OPD here is the ray's optical
    # path minus the chief's (positive = lags), the opposite sign of the classical wavefront aberration W.
    shape, conjugate = 1.0, -1.0
    bracket = ((n / (n - 1)) ** 2 + (n + 2) / (n * (n - 1) ** 2) * (shape + 2 * (n * n - 1) / (n + 2) * conjugate) ** 2
               - n / (n + 2) * conjugate**2)
    h_max = heights.max()
    w040 = h_max**4 / (f**3) / 4.0 * bracket / 8.0
    design = np.column_stack([np.ones_like(heights), heights**2, heights**4, heights**6])
    fitted = np.linalg.lstsq(design, np.array([r["opd_mm"] for r in wave["rays"]]), rcond=None)[0][2] * h_max**4
    assert -fitted == pytest.approx(w040, rel=0.02)


def test_wavefront_refusals_are_structured():
    collimated = _engine((laser("L", wavelength_nm=633.0, waist_radius_mm=1.0),), (observation_plane("D", position=(100.0, 0.0, 0.0), normal=X),))
    assert collimated.wavefront("D", "L#0", reference="sphere", center="ray_focus")["reason"] == "REFERENCE_UNDEFINED"
    assert collimated.wavefront("D", "L#0", reference="sphere")["reason"] == "REFERENCE_UNDEFINED"
    assert collimated.wavefront("D", "L#0", reference="sphere", center=(100.0, 0.0, 0.0))["reason"] == "REFERENCE_UNDEFINED"
    assert collimated.wavefront("D", "L#0", crossing_index=1)["reason"] == "CROSSING_NOT_FOUND"
    assert collimated.wavefront("D", "L#0", min_rays=97)["reason"] == "TOO_FEW_RAYS"
    assert collimated.wavefront("D", "L#0", reference="cylinder")["reason"] == "REFERENCE_UNDEFINED"
    ideal = _engine((laser("L", wavelength_nm=633.0, waist_radius_mm=1.0),
                     thin_lens("T", focal_length_mm=100.0, clear_radius_mm=10.0, position=(50.0, 0.0, 0.0))),
                    (observation_plane("D", position=(100.0, 0.0, 0.0), normal=X),))
    refused = _json_safe(ideal.wavefront("D", "L#0"))
    assert refused["status"] == "UNAVAILABLE" and refused["reason"] == "RAY_OPL_UNDEFINED" and "Diffraction" in refused["model"]
    assert ideal.result.observations[0].arrivals[0].ray_opl_defined is False


def test_a_double_pass_has_rays_and_a_wavefront_per_crossing():
    from os_optics.builders import mirror

    results = _engine((laser("L", wavelength_nm=633.0, waist_radius_mm=1.0), mirror("M", position=(100.0, 0.0, 0.0), normal=(-1.0, 0.0, 0.0))),
                      (observation_plane("D", position=(50.0, 0.0, 0.0), normal=X),))
    arrivals = {a["crossing_index"]: a for a in results.arrivals("D")}
    assert set(arrivals) == {0, 1} and arrivals[0]["direction_n"] == pytest.approx(1.0) and arrivals[1]["direction_n"] == pytest.approx(-1.0)
    for crossing in (0, 1):
        rows = results.rays("D", "L#0", crossing_index=crossing)
        assert len(rows) == 97 and {r["crossing_index"] for r in rows} == {crossing}
        chief = rows[0]
        assert chief["opl_mm"] == pytest.approx(50.0 + 100.0 * crossing)
        wave = results.wavefront("D", "L#0", crossing_index=crossing)
        assert wave["status"] == "OK" and wave["rms_waves"] < 1e-9
    assert results.paths()[0]["observations"] == ["D", "D"]


def test_wavefront_of_a_branch_starts_at_its_source_phase_front(runs):
    """Both arms of the interferometer are collimated at the crossing plane: flat against a plane perpendicular to each."""
    results = load_run(runs["two_beam_interference"][1])
    for path in ("SRC#0", "SRC#0.0"):
        wave = results.wavefront("crossing_plane", path)
        assert wave["status"] == "OK" and wave["rms_waves"] < 1e-6


# --------------------------------------------------------------------------------------------------------------- CLI
def _cli(argv, capsys):
    code = main(argv)
    return code, capsys.readouterr()


def test_results_commands_print_the_api_answers(runs, capsys):
    run, directory = runs["two_beam_interference"]
    results = load_run(directory)
    d = str(directory)
    for argv, expected in ((["results", "monitors", d], results.monitors()), (["results", "monitor", d, "crossing_plane"], results.monitor("crossing_plane")),
                           (["results", "paths", d], results.paths()), (["results", "power-budget", d], results.power_budget()),
                           (["results", "rays", d, "--monitor", "crossing_plane", "--path", "SRC#0"], results.rays("crossing_plane", "SRC#0")),
                           (["results", "wavefront", d, "--monitor", "crossing_plane", "--path", "SRC#0.0"], results.wavefront("crossing_plane", "SRC#0.0")),
                           (["results", "irradiance", d, "--monitor", "crossing_plane", "--samples", "5"], results.irradiance_map("crossing_plane", samples=5)),
                           (["results", "metrics"], results.metrics()), (["results", "columns", "rays"], {"kind": "rays", "columns": columns("rays")})):
        code, out = _cli(argv, capsys)
        assert code == 0 and json.loads(out.out) == json.loads(json.dumps(expected)), argv
    code, out = _cli(["results", "describe", d], capsys)
    assert code == 0 and json.loads(out.out)["run"]["trace_hash"] == run.result.manifest.trace_hash


def test_results_exit_codes(runs, capsys, tmp_path):
    d = str(runs["focus"][1])
    code, out = _cli(["results", "monitor", d, "nope"], capsys)
    assert code == 64 and json.loads(out.out)["error"]["code"] == "MONITOR_NOT_FOUND"
    code, out = _cli(["results", "describe", str(tmp_path / "none")], capsys)
    assert code == 64 and json.loads(out.out)["error"]["code"] == "RUN_NOT_READABLE"
    code, out = _cli(["results", "wavefront", d, "--monitor", "focal_plane", "--path", "SRC#0", "--reference", "sphere", "--center", "1,2"], capsys)
    assert code == 64
    code, out = _cli(["results", "wavefront", d, "--monitor", "focal_plane", "--path", "SRC#0", "--min-rays", "500"], capsys)
    assert code == 2 and json.loads(out.out)["reason"] == "TOO_FEW_RAYS"
    code, out = _cli(["results", "metrics", "--name", "optical_path_difference_mm"], capsys)
    assert code == 0 and "OPL(a) - OPL(b)" in json.loads(out.out)[0]["definition"]


def test_the_untraced_limit_of_a_map_is_the_callers_through_every_surface(runs, capsys):
    """max_untraced_fraction: the default 0.1 when not given; a number or no limit through the Python API, the command line
    ('none'), the JSON API (null) and the export, each stated in the document."""
    run, directory = runs["two_beam_interference"]
    results = load_run(directory)
    d = str(directory)
    assert results.irradiance_map("crossing_plane", samples=5)["max_untraced_fraction"] == 0.1
    for argv, limit in ((["--max-untraced-fraction", "0.5"], 0.5), (["--max-untraced-fraction", "none"], None), ([], 0.1)):
        code, out = _cli(["results", "irradiance", d, "--monitor", "crossing_plane", "--samples", "5", *argv], capsys)
        expected = results.irradiance_map("crossing_plane", samples=5, max_untraced_fraction=limit)
        assert code == 0 and json.loads(out.out) == json.loads(json.dumps(expected)) and expected["max_untraced_fraction"] == limit, argv
    files = run_artifact_files(run)
    for query, limit in (({"max_untraced_fraction": 0.5}, 0.5), ({"max_untraced_fraction": None}, None), ({}, 0.1)):
        reply = json_api.run_results(files, {"kind": "irradiance", "monitor": "crossing_plane", "samples": 5, **query})
        assert reply["ok"] is True and reply["result"]["max_untraced_fraction"] == limit, query
    summary = json.loads(results.export("irradiance", "json", monitor="crossing_plane", samples=5, max_untraced_fraction=None))["summary"]
    assert summary["max_untraced_fraction"] is None and summary["sampling"]["samples_basis"] == "STATED"
    # (the JSON API's export keeps a stated null as no limit, as its irradiance query does)
    for query, limit in (({"max_untraced_fraction": 0.5}, 0.5), ({"max_untraced_fraction": None}, None), ({}, 0.1)):
        reply = json_api.run_results(files, {"kind": "export", "export_kind": "irradiance", "format": "json",
                                             "monitor": "crossing_plane", "samples": 5, **query})
        assert reply["ok"] is True, reply
        assert json.loads(reply["result"]["data"])["summary"]["max_untraced_fraction"] == limit, query
    with pytest.raises(SystemExit) as refused:  # (not a number of at least 0: a usage error)
        main(["results", "irradiance", d, "--monitor", "crossing_plane", "--max-untraced-fraction", "-1"])
    assert refused.value.code == 64 and "a number of at least 0, or 'none'" in capsys.readouterr().err


@pytest.mark.parametrize("samples", ["0", "1", "-1", "1002"])
def test_an_explicit_irradiance_sample_count_is_never_replaced_by_the_default(runs, capsys, tmp_path, samples):
    """``--samples 0`` used to mean "use the default" (a 101 x 101 grid); every explicit value outside [2, 1001]
    is now refused like 1 or 1002 always were (``GRID_INVALID``, exit 2, no grid, no export file)."""

    d = str(runs["focus"][1])
    code, out = _cli(["results", "irradiance", d, "--monitor", "focal_plane", "--samples", samples], capsys)
    refused = json.loads(out.out)
    assert code == 2 and refused["status"] == "UNAVAILABLE" and refused["reason"] == "GRID_INVALID" and "grid" not in refused
    target = tmp_path / "irradiance.json"
    code, out = _cli(["export", "irradiance", d, "--monitor", "focal_plane", "--samples", samples, "--format", "json", "--out",
                      str(target)], capsys)
    assert code == 2 and json.loads(out.out)["reason"] == "GRID_INVALID" and not target.exists()


def test_the_irradiance_default_applies_only_when_samples_is_not_given(runs, capsys):
    run, directory = runs["focus"]
    results = load_run(directory)
    code, out = _cli(["results", "irradiance", str(directory), "--monitor", "focal_plane"], capsys)
    assert code == 0 and json.loads(out.out)["grid"]["shape"] == [101, 101]
    assert json.loads(results.export("irradiance", "json", monitor="focal_plane"))["summary"]["grid"]["shape"] == [101, 101]
    with pytest.raises(ResultsError) as refused:
        results.export("irradiance", "json", monitor="focal_plane", samples=0)
    assert refused.value.code == "GRID_INVALID"
    files = run_artifact_files(run)
    queried = json_api.run_results(files, {"kind": "irradiance", "monitor": "focal_plane", "samples": 0})
    assert queried["ok"] is True and queried["result"]["reason"] == "GRID_INVALID"
    exported = json_api.run_results(files, {"kind": "export", "export_kind": "irradiance", "format": "json", "monitor": "focal_plane",
                                            "samples": 0})
    assert exported["ok"] is False and exported["error"]["code"] == "GRID_INVALID"


def test_explicit_wavefront_export_filters_are_passed_as_given(runs):
    """The export took ``min_rays or 8``, ``reference or 'plane'`` and ``crossing_index or 0``: an explicit value was
    silently replaced. The export now answers exactly as the wavefront query with the same values."""

    clipped = _engine((laser("L", wavelength_nm=633.0, waist_radius_mm=1.0),
                       aperture_stop("S", position=(50.0, 0.1, 0.0), axis=X, radius_mm=0.15)),
                      (observation_plane("D", position=(100.0, 0.0, 0.0), normal=X),))
    assert clipped.wavefront("D", "L#0")["reason"] == "TOO_FEW_RAYS"  # 3 pupil rays pass the stop; 8 by default
    explicit = clipped.wavefront("D", "L#0", min_rays=3)
    assert explicit["status"] == "OK" and explicit["ray_count"] == 3
    assert clipped.wavefront("D", "L#0", min_rays=4)["reason"] == "TOO_FEW_RAYS"
    exported = json.loads(clipped.export("wavefront", "json", monitor="D", path="L#0", min_rays=3))
    assert exported["summary"]["rms_waves"] == explicit["rms_waves"] and len(exported["rows"]) == 3
    with pytest.raises(ResultsError) as refused:
        clipped.export("wavefront", "json", monitor="D", path="L#0")
    assert refused.value.code == "TOO_FEW_RAYS"
    files = run_artifact_files(runs["two_beam_interference"][0])
    base = {"monitor": "crossing_plane", "path": "SRC#0"}
    for extra in ({"reference": ""}, {"crossing_index": 0.5}, {"min_rays": 0}):
        queried = json_api.run_results(files, {"kind": "wavefront", **base, **extra})
        exported = json_api.run_results(files, {"kind": "export", "export_kind": "wavefront", "format": "json", **base, **extra})
        why = queried["result"]["reason"] if queried["ok"] else queried["error"]["code"]
        assert exported["ok"] is False and exported["error"]["code"] == why, extra


@pytest.mark.parametrize("value", [0, -1, 2.5, True, "8"])
def test_min_rays_is_an_integer_of_at_least_one(runs, value):
    """Review P3-2: ``min_rays`` 0 / -1 / 2.5 / true were floored to 1 by ``max(min_rays, 1)``; they are refused."""

    run, directory = runs["focus"]
    with pytest.raises(ResultsError) as refused:
        load_run(directory).wavefront("focal_plane", "SRC#0", min_rays=value)
    assert refused.value.code == "RESULTS_QUERY_INVALID" and "min_rays" in refused.value.message
    reply = json_api.run_results(run_artifact_files(run), {"kind": "wavefront", "monitor": "focal_plane", "path": "SRC#0", "min_rays": value})
    assert reply["ok"] is False and reply["error"]["code"] == "RESULTS_QUERY_INVALID"


@pytest.mark.parametrize("value", ["0", 0.5, True, False, [0]], ids=repr)
def test_a_crossing_index_query_is_an_integer(runs, value):
    """Review P3-2: rays / export rays / wavefront read crossing_index "0" or 0.5 as no match (0 rows), true as 1, false
    as 0, and a wavefront raised a raw TypeError; every one is now RESULTS_QUERY_INVALID."""

    run, directory = runs["two_beam_interference"]
    files = run_artifact_files(run)
    for query in ({"kind": "rays", "monitor": "crossing_plane", "crossing_index": value},
                  {"kind": "export", "export_kind": "rays", "format": "csv", "crossing_index": value},
                  {"kind": "wavefront", "monitor": "crossing_plane", "path": "SRC#0", "crossing_index": value}):
        reply = json_api.run_results(files, query)
        assert reply["ok"] is False and reply["error"]["code"] == "RESULTS_QUERY_INVALID", query
        assert "Traceback" not in json.dumps(reply) and "TypeError" not in json.dumps(reply)


def test_valid_crossing_and_min_rays_queries_are_unchanged(runs):
    run, directory = runs["two_beam_interference"]
    results = load_run(directory)
    files = run_artifact_files(run)
    rows = results.rays("crossing_plane", "SRC#0", 0)
    assert rows and json_api.run_results(files, {"kind": "rays", "monitor": "crossing_plane", "path": "SRC#0", "crossing_index": 0,
                                                 "limit": None})["result"]["rows"] == rows
    assert results.rays("crossing_plane", "SRC#0", 1) == [] and results.rays("crossing_plane", "SRC#0", -1) == []
    assert results.wavefront("crossing_plane", "SRC#0", crossing_index=1)["reason"] == "CROSSING_NOT_FOUND"
    assert results.wavefront("crossing_plane", "SRC#0", min_rays=1)["status"] == "OK"
    assert results.wavefront("crossing_plane", "SRC#0", crossing_index=0.0) == results.wavefront("crossing_plane", "SRC#0")


def test_cli_min_rays_is_passed_as_given_and_checked(runs, capsys, monkeypatch):
    from os_optics.results import RunResults

    seen = []
    original = RunResults.wavefront
    monkeypatch.setattr(RunResults, "wavefront", lambda self, *a, **k: seen.append(k["min_rays"]) or original(self, *a, **k))
    d = str(runs["focus"][1])
    for argv, expected in ((["--min-rays", "1"], 1), ([], 8)):
        assert _cli(["results", "wavefront", d, "--monitor", "focal_plane", "--path", "SRC#0", *argv], capsys)[0] == 0
        assert seen.pop() == expected
    for value in ("0", "-1"):
        code, out = _cli(["results", "wavefront", d, "--monitor", "focal_plane", "--path", "SRC#0", "--min-rays", value], capsys)
        assert code == 64 and json.loads(out.out)["error"]["code"] == "RESULTS_QUERY_INVALID"


# ------------------------------------------------------------------------------------- internal errors are named as such
def _defect(*_args, **_kwargs):
    raise TypeError("a defect inside the analysis")


#: One command of each handler of ``cli_results`` with the ``RunResults`` method it calls (RUN: the run directory).
INTERNAL_CASES = [
    (["results", "monitors", "RUN"], "monitors"),
    (["results", "irradiance", "RUN", "--monitor", "focal_plane"], "irradiance_map"),
    (["results", "imaging", "RUN", "--monitor", "focal_plane"], "imaging"),
    (["results", "ghosts", "RUN", "--monitor", "focal_plane"], "ghosts"),
    (["results", "field", "RUN", "--monitor", "focal_plane"], "field"),
    (["results", "cavity", "RUN", "--request", "REQUEST"], "cavity"),
    (["results", "focal-field", "RUN", "--monitor", "focal_plane", "--request", "REQUEST"], "focal_field"),
    (["results", "pupil-field", "RUN", "--monitor", "focal_plane", "--request", "REQUEST"], "pupil_field"),
    (["results", "detector", "RUN", "--monitor", "focal_plane", "--detector", "REQUEST"], "irradiance_map"),
    (["export", "monitors", "RUN"], "export"),
]


def _argv(argv, run_dir, request):
    return [{"RUN": str(run_dir), "REQUEST": str(request)}.get(word, word) for word in argv]


@pytest.mark.parametrize(("argv", "method"), INTERNAL_CASES, ids=lambda v: " ".join(v[:2]) if isinstance(v, list) else v)
def test_a_defect_on_a_run_that_verifies_is_an_internal_error_and_on_a_damaged_run_an_unreadable_run(runs, capsys, tmp_path,
                                                                                                    monkeypatch, argv, method):
    """An exception no results command documents: on a run that verifies the input is good, so the command says the
    failure is its own (``INTERNAL_ERROR`` with the exception's type, the traceback on stderr, exit 70, never the 64 of a
    usage error); on a run of which one file was altered the files may hold anything: ``RUN_NOT_READABLE``, exit 64, with
    the command that checks it."""
    import shutil

    from os_optics.results import RunResults

    request = tmp_path / "request.json"
    request.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(RunResults, method, _defect)
    good = runs["focus"][1]
    code, out = _cli(_argv(argv, good, request), capsys)
    error = json.loads(out.out)["error"]
    assert (code, error["code"], error["type"]) == (70, "INTERNAL_ERROR", "TypeError")
    assert "a defect inside the analysis" in error["message"] and "Traceback" in out.err and "_defect" in out.err
    assert "unexpected structure" not in error["message"]
    damaged = tmp_path / "damaged"
    shutil.copytree(good, damaged)
    with (damaged / "optical_scene.json").open("ab") as handle:
        handle.write(b" ")
    code, out = _cli(_argv(argv, damaged, request), capsys)
    error = json.loads(out.out)["error"]
    assert (code, error["code"]) == (64, "RUN_NOT_READABLE") and "type" not in error
    assert f"'os-optics project verify {damaged}'" in error["message"] and "Traceback" not in out.err


@pytest.mark.parametrize(("command", "module", "name"), [("pupil-field", "pupil_field", "pupil_field"),
                                                         ("focal-field", "focal_field", "focal_field")])
def test_a_defect_of_an_analysis_that_reads_no_run_is_an_internal_error(capsys, tmp_path, monkeypatch, command, module, name):
    import importlib

    request = tmp_path / "request.json"
    request.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(importlib.import_module(f"os_optics.results.{module}"), name, _defect)
    code, out = _cli(["results", command, "--request", str(request)], capsys)
    error = json.loads(out.out)["error"]
    assert (code, error["code"], error["type"]) == (70, "INTERNAL_ERROR", "TypeError") and "Traceback" in out.err


def test_a_defect_of_an_export_that_needs_no_run_is_an_internal_error(capsys, monkeypatch):
    from os_optics.results import RunResults

    monkeypatch.setattr(RunResults, "export", _defect)
    code, out = _cli(["export", "metrics"], capsys)
    error = json.loads(out.out)["error"]
    assert (code, error["code"], error["type"]) == (70, "INTERNAL_ERROR", "TypeError") and "Traceback" in out.err


def test_a_run_that_cannot_be_loaded_is_unreadable_unless_it_verifies(runs, capsys, tmp_path, monkeypatch):
    """The loader itself failing: the directory is checked as ``project verify`` checks it."""
    import shutil

    import os_optics.results as results_package

    monkeypatch.setattr(results_package, "load_run", _defect)
    good = runs["focus"][1]
    code, out = _cli(["results", "describe", str(good)], capsys)
    assert (code, json.loads(out.out)["error"]["code"]) == (70, "INTERNAL_ERROR")
    damaged = tmp_path / "damaged"
    shutil.copytree(good, damaged)
    (damaged / "ray_data.json").write_bytes(b"{}")
    code, out = _cli(["results", "describe", str(damaged)], capsys)
    assert (code, json.loads(out.out)["error"]["code"]) == (64, "RUN_NOT_READABLE")


def test_any_command_that_fails_inside_os_optics_exits_70_with_the_internal_error_document(capsys, monkeypatch):
    """A defect in any command is never read as the exit code of a result (1 = a target not met): one JSON document with
    the exception's type on stdout, the traceback on stderr, exit 70. Ctrl-C keeps 130."""
    from os_optics import cli
    from os_optics.cli_exit import INTERNAL_ERROR_EXIT

    def crash(*_args, **_kwargs):
        raise ZeroDivisionError("division inside a command")

    monkeypatch.setattr(cli, "_main", crash)
    code, out = _cli(["catalog", "list"], capsys)
    document = json.loads(out.out)
    assert code == INTERNAL_ERROR_EXIT == 70 and document["status"] == "ERROR"
    assert (document["error"]["code"], document["error"]["type"]) == ("INTERNAL_ERROR", "ZeroDivisionError")
    assert "division inside a command" in document["error"]["message"] and "Traceback" in out.err

    def interrupt(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "_main", interrupt)
    assert _cli(["catalog", "list"], capsys)[0] == 130
    monkeypatch.undo()
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    assert "70 = an internal error of os-optics" in " ".join(capsys.readouterr().out.split())


def test_json_api_results_internal_errors_and_query_value_types(runs, monkeypatch):
    """``run_results``: a query value of the wrong JSON type is refused by name before anything is read; what a query
    raises after that on files that verify (``verify`` true) is os-optics's own -- ``INTERNAL_ERROR`` with the exception's
    type --, and on files that were not checked the earlier answers stand."""
    from os_optics.results import RunResults

    files = run_artifact_files(runs["focus"][0])
    for query, field in (({"kind": "monitor", "monitor": 5}, "monitor"), ({"kind": "paths", "path": ["SRC#0"]}, "path"),
                         ({"kind": "irradiance", "monitor": "focal_plane", "wavelength_nm": "852"}, "wavelength_nm"),
                         ({"kind": "irradiance", "monitor": "focal_plane", "samples": True}, "samples"),
                         ({"kind": "irradiance", "monitor": "focal_plane", "center_uv_mm": [0.0]}, "center_uv_mm"),
                         ({"kind": "irradiance", "monitor": "focal_plane", "roi": [1, 2]}, "roi"),
                         ({"kind": "wavefront", "monitor": "focal_plane", "path": "SRC#0", "center": [0.0, "a", 0.0]}, "center"),
                         ({"kind": "imaging", "monitor": "focal_plane", "request": "{}"}, "request"),
                         ({"kind": "export", "export_kind": "rays", "format": 3}, "format")):
        for verify in (False, True):
            reply = json_api.run_results(files, {**query, "verify": verify})
            assert reply["ok"] is False and reply["error"]["code"] == "RESULTS_QUERY_INVALID", query
            assert reply["error"]["problems"][0]["message"].startswith(f"invalid query value: {field} is "), query
    # values of the right type are answered as before (an integer where a number is taken, a list as a centre)
    assert json_api.run_results(files, {"kind": "irradiance", "monitor": "focal_plane", "samples": 5, "wavelength_nm": 852,
                                        "center_uv_mm": [0, 0.0], "verify": True})["ok"] is True
    assert json_api.run_results(files, {"kind": "wavefront", "monitor": "focal_plane", "path": "SRC#0", "reference": "sphere",
                                        "center": "ray_focus"})["ok"] is True
    monkeypatch.setattr(RunResults, "monitors", _defect)
    internal = json_api.run_results(files, {"kind": "monitors", "verify": True})
    assert internal["ok"] is False and (internal["error"]["code"], internal["error"]["type"]) == ("INTERNAL_ERROR", "TypeError")
    assert "a defect inside the analysis" in internal["error"]["problems"][0]["message"]
    unchecked = json_api.run_results(files, {"kind": "monitors"})
    assert unchecked["error"]["code"] == "RESULTS_QUERY_INVALID" and "type" not in unchecked["error"]
    altered = {**files, "optical_scene.json": files["optical_scene.json"] + b" "}
    assert json_api.run_results(altered, {"kind": "monitors", "verify": True})["error"]["code"] == "RESULTS_QUERY_INVALID"


def test_export_writes_the_api_bytes(runs, capsys, tmp_path):
    directory = runs["two_beam_interference"][1]
    results = load_run(directory)
    target = tmp_path / "arrivals.csv"
    code, out = _cli(["export", "arrivals", str(directory), "--format", "csv", "--out", str(target)], capsys)
    assert code == 0 and json.loads(out.out)["bytes"] == target.stat().st_size and target.read_bytes() == results.export("arrivals", "csv")
    target = tmp_path / "wave.json"
    code, _ = _cli(["export", "wavefront", str(directory), "--monitor", "crossing_plane", "--path", "SRC#0", "--format", "json", "--out",
                    str(target)], capsys)
    assert code == 0 and json.loads(target.read_bytes())["summary"]["rms_waves"] == results.wavefront("crossing_plane", "SRC#0")["rms_waves"]
    code, out = _cli(["export", "irradiance", str(directory), "--monitor", "crossing_plane", "--format", "npz"], capsys)
    assert code == 64  # binary output needs --out
    target = tmp_path / "metrics.jsonl"
    assert _cli(["export", "metrics", "--format", "jsonl", "--out", str(target)], capsys)[0] == 0
    assert target.read_bytes() == results.export("metrics", "jsonl")


def test_export_without_out_writes_the_file_to_standard_output(runs, capsysbinary):
    directory = runs["two_beam_interference"][1]
    assert main(["export", "paths", str(directory)]) == 0
    assert capsysbinary.readouterr().out == load_run(directory).export("paths", "csv")


# ----------------------------------------------------------------------------------------------------------- JSON API
def test_json_api_results_queries_answer_over_the_run_files(runs):
    run, directory = runs["two_beam_interference"]
    files = run_artifact_files(run)
    results = load_run(directory)
    assert json_api.metric_descriptors()["descriptors"] == results.metrics()
    caps = json_api.capabilities()["results"]
    assert "ray_data.json" in caps["run_files"] and caps["export_formats"]["irradiance"][-1] == "npz"
    for query, expected in (({"kind": "monitors"}, results.monitors()), ({"kind": "monitor", "monitor": "crossing_plane"}, results.monitor("crossing_plane")),
                            ({"kind": "paths"}, results.paths()), ({"kind": "power_budget"}, results.power_budget()),
                            ({"kind": "superpositions"}, results.superpositions()), ({"kind": "columns", "export_kind": "rays"}, columns("rays")),
                            ({"kind": "wavefront", "monitor": "crossing_plane", "path": "SRC#0"}, results.wavefront("crossing_plane", "SRC#0")),
                            ({"kind": "irradiance", "monitor": "crossing_plane", "samples": 7}, results.irradiance_map("crossing_plane", samples=7))):
        reply = json_api.run_results(files, query)
        json.dumps(reply, allow_nan=False)
        assert reply["ok"] is True and reply["result"] == expected, query
    page = json_api.run_results(files, {"kind": "rays", "offset": 3, "limit": 5})["result"]
    assert page["total"] == len(results.rays()) and page["rows"] == results.rays()[3:8]
    exported = json_api.run_results(files, {"kind": "export", "export_kind": "rays", "format": "jsonl", "monitor": "crossing_plane"})["result"]
    assert exported["file_name"] == "rays.jsonl" and exported["data"] == results.export("rays", "jsonl", monitor="crossing_plane")
    described = json_api.run_results(files, {"kind": "describe", "verify": True})["result"]
    assert described["integrity"]["consistent"] is True
    assert json_api.run_results(files)["result"]["integrity"]["checked"] is False


@pytest.mark.parametrize(("files", "query", "code"), [
    ("not files", {"kind": "monitors"}, "INPUT_INVALID"),
    (None, {"kind": "nope"}, "RESULTS_QUERY_INVALID"),
    (None, {"kind": "monitors", "monitor": "x"}, "RESULTS_QUERY_INVALID"),
    (None, {"kind": "monitor", "monitor": "nope"}, "MONITOR_NOT_FOUND"),
    (None, {"kind": "rays", "limit": -1}, "RESULTS_QUERY_INVALID"),
    (None, {"kind": "export", "export_kind": "rays", "format": "npz"}, "EXPORT_INVALID"),
])
def test_json_api_results_refusals_are_envelopes(runs, files, query, code):
    run, _ = runs["focus"]
    reply = json_api.run_results(run_artifact_files(run) if files is None else files, query)
    assert reply["ok"] is False and reply["error"]["code"] == code and reply["error"]["problems"][0]["code"] == code


def test_describe_states_each_elements_simulation_model(runs):
    described = load_run(runs["dichroic_routing"][1]).describe()
    models = described["element_models"]
    assert models and all({"instance_id", "optical_sku", "model_kind", "fidelity", "approximations"} <= set(m) for m in models)
    assert "element_models" in described["conventions"] and "phase" in described["conventions"]
