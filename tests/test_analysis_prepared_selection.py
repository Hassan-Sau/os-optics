"""Selections inside the analyses of a run: the imaging request's ``convergence`` and ``diffraction`` and the ghost
request's ``convergence`` leave out the doubled discretization and the diffraction quantities. What a request leaves out is
not computed (the doubled grid is never traced, the point spread function never summed) and is stated in each quantity:
``convergence.method`` ``NONE`` for a value whose discretization was not checked, ``state`` ``NOT_COMPUTED`` with reason
``NOT_REQUESTED`` for a quantity that was not computed. A default request's document is unchanged, and a value computed
under a selection is the value the full analysis computes on the same discretization (its ``convergence.coarse``).

A selection changes what is computed, never what the analysis says about the rest: without the diffraction quantities a
field whose wavefront or geometric quantities are not available states them as the full analysis does (a thin ideal lens,
whose rays carry no optical path; a source's two polarization components that part), and a reason of the diffraction
quantities alone (the pupil's sampling, the map's window) stays with them. The execution record says whether the analysis
ran to its end, which every one of these does: a quantity's state never makes it partial.

The options reach the analyses unchanged through ``json_api.run_results`` and ``os-optics results imaging|ghosts
--request``; an option the analysis does not know, or a map without the point spread function, is an invalid request.

An analysis read from an in-memory ``readings`` evaluation of a Project equals the same analysis of the Project's complete
run: it reads the arrivals the readings retain, and nothing it needs is outside them (imaging, ghosts, field; and the
wavefront query of the readings).
"""

from __future__ import annotations

import importlib
import json
import math

import pytest

from os_optics.cli import main as cli_main
from os_optics.contracts.analysis import ConvergenceMethod
from os_optics.errors.status import QuantityState
from os_optics.local import EvaluationSession, evaluate_project, example_project_of_kind, json_api, run_optical_project
from os_optics.results import ImagingError, load_run

from declared_helpers import declared_component, thin_lens_model, with_models
from local_helpers import catalog_document, plane, project, run, source
from test_ghosts import _prescription
from test_imaging import _hyperbolic, _lens_project
from test_imaging_polarization_components import _singlet_behind_polarizer

# (the modules: the package exports functions of the same names)
imaging_module = importlib.import_module("os_optics.results.imaging")
ghosts_module = importlib.import_module("os_optics.results.ghosts")
#: The default ray grid (a coarser one undersamples this pupil for the diffraction quantities) and a small map.
BASE = {"launch_grid": 64, "psf_samples": 32}
LINE = "SRC#0."
GEOMETRIC = ("image_space_na", "geometric_spot_rms_radius")
WAVEFRONT = ("opd_rms_waves", "opd_pv_waves")
DIFFRACTION = ("strehl_ratio", "mtf_tangential", "mtf_sagittal", "diffraction_limited_mtf_tangential", "diffraction_limited_mtf_sagittal",
               "encircled_radius_50", "encircled_radius_80", "diffraction_limited_encircled_radius_80")


def _f8_project() -> dict:
    return _lens_project(_hyperbolic(8.0, 40.0, 587.5618), waist_mm=1.0)


@pytest.fixture(scope="module")
def f8():
    return load_run(run(_f8_project()))


@pytest.fixture(scope="module")
def full(f8):
    return f8.imaging("image", BASE)


def _dump(doc) -> dict:
    return doc.model_dump(mode="json", by_alias=True)


def test_the_default_options_leave_the_imaging_document_unchanged(f8, full):
    explicit = f8.imaging("image", {**BASE, "convergence": True, "diffraction": True})
    assert _dump(explicit) == _dump(full)
    assert "convergence" not in full.settings["request"] and "diffraction" not in full.settings["request"]
    assert full.quantity(LINE + "strehl_ratio").convergence.method is ConvergenceMethod.REFINEMENT


def test_without_convergence_imaging_traces_no_doubled_grid_and_claims_no_convergence(f8, full, monkeypatch):
    grids: list[int] = []
    traced = imaging_module.trace_fields

    def recording(request, monitor, wanted, preferred, grid, *args, **kwargs):
        grids.append(grid.count)
        return traced(request, monitor, wanted, preferred, grid, *args, **kwargs)

    monkeypatch.setattr(imaging_module, "trace_fields", recording)
    doc = f8.imaging("image", {**BASE, "convergence": False})
    assert grids and set(grids) == {BASE["launch_grid"]}
    assert doc.execution.complete and doc.settings["request"]["convergence"] is False
    known = [q for q in doc.quantities if q.state is QuantityState.KNOWN]
    assert {q.quantity_id for q in known} == {q.quantity_id for q in full.quantities if q.state is QuantityState.KNOWN}
    for q in known:
        assert q.convergence is not None and q.convergence.method is ConvergenceMethod.NONE, q.quantity_id
        assert q.convergence.converged is None and q.convergence.coarse is None and q.convergence.fine is None, q.quantity_id
    # the geometric and wavefront values are those of the full analysis' launch_grid rays
    for name in (*GEOMETRIC, *WAVEFRONT):
        assert doc.quantity(LINE + name).value == full.quantity(LINE + name).convergence.coarse, name
    rays = doc.arrays[LINE + "rays"]
    assert rays["launch_grid"] == [BASE["launch_grid"]] and len(rays["rays_arrived"]) == 1 == len(rays["rays_without_neighbours"])
    assert len(doc.arrays[LINE + "psf"]["window_light_share"]) == 1


def test_without_diffraction_imaging_sums_no_point_spread_function(f8, full, monkeypatch):
    sums: list[str] = []
    for name in ("psf_map", "psf_at"):
        original = getattr(imaging_module._Level, name)
        monkeypatch.setattr(imaging_module._Level, name, lambda self, *a, _name=name, _original=original, **k: sums.append(_name) or _original(self, *a, **k))
    doc = f8.imaging("image", {**BASE, "diffraction": False})
    assert sums == []
    assert doc.execution.complete and doc.settings["request"]["diffraction"] is False
    for name in DIFFRACTION:
        q = doc.quantity(LINE + name)
        assert q.state is QuantityState.NOT_COMPUTED and q.reason == "NOT_REQUESTED" and q.value is None and q.values_ref is None, name
    # the geometric and wavefront quantities are the full analysis' (the same two grids, the same convergence records)
    for name in (*GEOMETRIC, *WAVEFRONT, "geometric_mtf_tangential", "geometric_mtf_sagittal"):
        assert doc.quantity(LINE + name).model_dump() == full.quantity(LINE + name).model_dump(), name
    assert LINE + "psf" not in doc.arrays and LINE + "encircled_energy" not in doc.arrays
    assert doc.arrays[LINE + "rays"] == full.arrays[LINE + "rays"]


def test_both_options_off_leave_one_grid_and_the_geometric_quantities(f8, full):
    doc = f8.imaging("image", {**BASE, "convergence": False, "diffraction": False})
    assert doc.execution.complete
    for name in (*GEOMETRIC, *WAVEFRONT):
        q = doc.quantity(LINE + name)
        assert q.value == full.quantity(LINE + name).convergence.coarse and q.convergence.method is ConvergenceMethod.NONE, name
    assert all(doc.quantity(LINE + name).state is QuantityState.NOT_COMPUTED for name in DIFFRACTION)


def test_a_psf_map_needs_the_diffraction_quantities(f8):
    with pytest.raises(ImagingError) as refused:
        f8.imaging("image", {"diffraction": False, "include_psf_map": True})
    assert refused.value.code == "IMAGING_INVALID" and "include_psf_map" in refused.value.message


# ------------------------------------------------------------------------------------------------- a field that lacks a quantity

#: The selections that leave the diffraction quantities out.
WITHOUT_DIFFRACTION = ({"diffraction": False}, {"convergence": False, "diffraction": False})


def _thin_lens_project() -> dict:
    """A thin ideal lens: its rays carry no optical path, so its image has no wavefront (``RAY_OPL_UNDEFINED``)."""

    doc = project([source(position=(-50.0, 0.0, 0.0), waist_mm=1.0)], [plane("image", (100.0, 0.0, 0.0), radius_mm=20.0)])
    doc["components"].append(declared_component("L", "lens", "thin", (0.0, 0.0, 0.0)))
    return with_models(doc, [thin_lens_model("thin", "100", clear="2")])


def test_without_diffraction_a_field_without_a_wavefront_states_what_the_full_analysis_states(tmp_path, capsys):
    run_dir = tmp_path / "run"
    results = load_run(run(_thin_lens_project(), output_dir=str(run_dir)))
    default = results.imaging("image", {"launch_grid": 32})
    # the analysis ran to its end; the missing wavefront is the quantities' own state
    assert default.execution.outcome.value == "COMPLETED" and default.execution.complete and default.execution.reason == ""
    assert all(default.quantity(LINE + name).reason == "RAY_OPL_UNDEFINED" for name in (*WAVEFRONT, "strehl_ratio"))
    for selection in WITHOUT_DIFFRACTION:
        doc = results.imaging("image", {"launch_grid": 32, **selection})
        assert doc.execution == default.execution, selection
        for name in WAVEFRONT:
            q = doc.quantity(LINE + name)
            assert q.state is QuantityState.OUT_OF_MODEL_SCOPE and q.reason == "RAY_OPL_UNDEFINED", (selection, name)
        geometric = doc.quantity(LINE + "geometric_mtf_tangential")
        assert geometric.state is QuantityState.UNKNOWN and geometric.reason == "GEOMETRIC_REGIME_UNKNOWN", selection
        assert doc.quantity(LINE + "geometric_spot_rms_radius").state is QuantityState.KNOWN, selection
        for name in DIFFRACTION:
            q = doc.quantity(LINE + name)
            assert q.state is QuantityState.NOT_COMPUTED and q.reason == "NOT_REQUESTED", (selection, name)
    # the command answers: exit code 0, with the states in the rows
    request = tmp_path / "imaging.json"
    for selection in WITHOUT_DIFFRACTION:
        request.write_text(json.dumps({"launch_grid": 32, **selection}), encoding="utf-8")
        assert cli_main(["results", "imaging", str(run_dir), "--monitor", "image", "--request", str(request)]) == 0, selection
        printed = json.loads(capsys.readouterr().out)
        assert printed["execution"]["outcome"] == "COMPLETED" and printed["execution"]["complete"] is True, selection
        states = {q["quantity_id"]: (q["state"], q["reason"]) for q in printed["quantities"]}
        assert states[LINE + "opd_rms_waves"] == ("OUT_OF_MODEL_SCOPE", "RAY_OPL_UNDEFINED"), selection


def test_without_diffraction_parted_polarization_components_state_what_the_full_analysis_states(monkeypatch):
    import os_optics.results.imaging_trace as imaging_trace

    monkeypatch.setattr(imaging_trace, "ROUTE_TOLERANCE_MM", -1.0)  # (no two crossings agree: every pair reads as parted)
    results = load_run(run(_singlet_behind_polarizer(45.0)))
    default = results.imaging("image", {"launch_grid": 32, "psf_samples": 32})
    assert default.execution.complete and all(q.state is not QuantityState.KNOWN for q in default.quantities)
    for selection in WITHOUT_DIFFRACTION:
        doc = results.imaging("image", {"launch_grid": 32, "psf_samples": 32, **selection})
        assert doc.execution == default.execution, selection
        for name in (*GEOMETRIC, *WAVEFRONT, "geometric_mtf_tangential", "geometric_mtf_sagittal"):
            assert doc.quantity(LINE + name).reason == "POLARIZATION_COMPONENT_ROUTES_DIFFER", (selection, name)
        assert all(q.state is not QuantityState.KNOWN for q in doc.quantities), selection


@pytest.mark.parametrize(("request_fields", "reason"), [
    ({"launch_grid": 16, "psf_samples": 32}, "PUPIL_UNDERSAMPLED"),
    ({**BASE, "psf_half_width_um": 0.5}, "PSF_WINDOW_TOO_SMALL"),
])
def test_a_reason_of_the_diffraction_quantities_alone_stays_with_them(f8, request_fields, reason):
    default = f8.imaging("image", request_fields)
    blocked = default.quantity(LINE + "mtf_tangential")
    assert default.execution.complete and blocked.state is QuantityState.OUT_OF_MODEL_SCOPE and blocked.reason == reason
    assert all(default.quantity(LINE + name).state is QuantityState.KNOWN for name in (*GEOMETRIC, *WAVEFRONT))
    doc = f8.imaging("image", {**request_fields, "diffraction": False})
    assert doc.execution.complete
    for name in (*GEOMETRIC, *WAVEFRONT):
        assert doc.quantity(LINE + name).model_dump() == default.quantity(LINE + name).model_dump(), name
    assert all(doc.quantity(LINE + name).state is QuantityState.NOT_COMPUTED for name in DIFFRACTION)


# ------------------------------------------------------------------------------------------------- the JSON API and the command


def test_the_options_pass_through_the_json_api_and_the_command(tmp_path, capsys):
    run_dir = tmp_path / "run"
    run(_f8_project(), output_dir=str(run_dir))
    files = {path.name: path.read_bytes() for path in run_dir.iterdir() if path.is_file()}
    narrowed = {**BASE, "convergence": False, "diffraction": False}
    reply = json_api.run_results(files, {"kind": "imaging", "monitor": "image", "request": narrowed})
    assert reply["ok"] is True, reply
    assert reply["result"]["settings"]["request"]["convergence"] is False and reply["result"]["settings"]["request"]["diffraction"] is False
    assert {q["state"] for q in reply["result"]["quantities"] if q["quantity_id"].split(".")[-1] in DIFFRACTION} == {"NOT_COMPUTED"}
    request = tmp_path / "request.json"
    request.write_text(json.dumps(narrowed), encoding="utf-8")
    assert cli_main(["results", "imaging", str(run_dir), "--monitor", "image", "--request", str(request)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["settings"]["request"] == reply["result"]["settings"]["request"] and out["quantities"] == reply["result"]["quantities"]
    request.write_text(json.dumps({"diffraction": False, "include_psf_map": True}), encoding="utf-8")
    assert cli_main(["results", "imaging", str(run_dir), "--monitor", "image", "--request", str(request)]) == 1
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "IMAGING_INVALID"
    # the ghost request has convergence and no diffraction option
    reply = json_api.run_results(files, {"kind": "ghosts", "monitor": "image", "request": {"convergence": False}})
    assert reply["ok"] is True and reply["result"]["settings"]["request"]["convergence"] is False, reply
    request.write_text(json.dumps({"convergence": False}), encoding="utf-8")
    code = cli_main(["results", "ghosts", str(run_dir), "--monitor", "image", "--request", str(request)])
    out = json.loads(capsys.readouterr().out)
    assert code == (0 if out["execution"]["complete"] else 2) and out["quantities"] == reply["result"]["quantities"]
    refused = json_api.run_results(files, {"kind": "ghosts", "monitor": "image", "request": {"diffraction": False}})
    assert refused["ok"] is False and refused["error"]["code"] == "GHOSTS_INVALID", refused


# ------------------------------------------------------------------------------------------------- ghosts


@pytest.fixture(scope="module")
def plate():
    return load_run(run_optical_project(_lens_project(_prescription(None, None, 5.0, 20.0), waist_mm=1.0)))


def test_ghosts_without_convergence_trace_once_with_the_sources_own_sampling(plate, monkeypatch):
    full = plate.ghosts("image")
    assert _dump(plate.ghosts("image", {"convergence": True})) == _dump(full)
    assert "convergence" not in full.settings["request"]
    scales: list[int] = []
    traced = ghosts_module._trace

    def recording(request, wanted, parsed, scale):
        scales.append(scale)
        return traced(request, wanted, parsed, scale)

    monkeypatch.setattr(ghosts_module, "_trace", recording)
    single = plate.ghosts("image", {"convergence": False})
    assert scales == [1]
    assert single.settings["request"]["convergence"] is False
    by_id = {q.quantity_id: q for q in full.quantities}
    compared = 0
    for q in single.quantities:
        if q.convergence is not None:
            assert q.convergence.method is ConvergenceMethod.NONE and q.convergence.converged is None, q.quantity_id
            assert q.convergence.coarse is None and q.convergence.fine is None, q.quantity_id
        reference = by_id.get(q.quantity_id)
        if q.state is QuantityState.KNOWN and reference is not None and reference.convergence is not None and reference.convergence.coarse is not None:
            # (the single trace is the full analysis' coarser one)
            assert q.value == reference.convergence.coarse, q.quantity_id
            compared += 1
    assert compared >= 3
    assert single.completeness is not None and single.completeness.enumerated == full.completeness.enumerated


# ------------------------------------------------------------------------------------------------- a readings evaluation


def test_imaging_of_a_readings_evaluation_equals_the_complete_runs(full):
    evaluation = evaluate_project(_f8_project(), outputs=("readings",), catalog=catalog_document())
    assert evaluation.result.evaluation is not None  # (a narrowed evaluation: no untraced light, no bounds)
    doc = evaluation.results().imaging("image", BASE)
    ours, theirs = _dump(doc), _dump(full)
    assert ours["identity"]["physical"]["engine_input_hash"] == theirs["identity"]["physical"]["engine_input_hash"]
    for dump in (ours, theirs):
        dump["identity"]["physical"].pop("run_result_hash")
    assert ours == theirs


def _sampled_field_project() -> dict:
    """A sampled Gaussian field (w = 0.3 mm, 32 x 32 points of 0.08 mm, 633 nm, 256 rays) and a plane 5 mm on in free space."""

    count, pitch, w = 32, 0.08, 0.3
    axis = [(i - 0.5 * (count - 1)) * pitch for i in range(count)]
    amplitude = [[math.sqrt(math.exp(-2.0 * (x ** 2 + y ** 2) / w ** 2)) for x in axis] for y in axis]
    source = {"instance_id": "S", "optical_role": "source", "custom_part": {"custom_part_key": "sampled", "description": "Sampled source."},
              "nominal_pose": {"position_mm": [0.0, 0.0, 0.0], "quaternion_wxyz": [0.5, 0.5, 0.5, 0.5]},
              "source_model": {"model_kind": "SAMPLED_FIELD", "wavelength_nm": 633.0, "power_w": 0.001,
                               "grid": {"column_count": count, "row_count": count, "pitch_mm": [pitch, pitch]},
                               "amplitude": amplitude, "phase_rad": [[0.0] * count for _ in range(count)], "amplitude_unit": "RELATIVE",
                               "polarization": {"kind": "LINEAR", "angle_deg": 0.0}, "ray_sampling": {"sample_count": 256, "seed": 0}}}
    plane = {"observation_id": "P", "pose": {"position_mm": [5.0, 0.0, 0.0], "quaternion_wxyz": [0.5, 0.5, 0.5, 0.5]},
             "aperture": {"shape": "circle", "radius_mm": 5.0}}
    return {"schema": "opensci.project", "project_schema_version": "2.0", "components": [source], "observations": [plane], "targets": []}


def test_the_field_and_the_wavefront_of_a_readings_evaluation_equal_the_complete_runs():
    document = _sampled_field_project()
    complete = load_run(run_optical_project(document))
    evaluation = evaluate_project(document, outputs=("readings",))
    assert evaluation.result.evaluation is not None  # (a narrowed evaluation)
    for request in (None, {"refinement": False}):
        theirs = complete.field("P", request)
        assert theirs.execution.complete and all(q.state is QuantityState.KNOWN for q in theirs.quantities), request
        ours, theirs = _dump(evaluation.results().field("P", request)), _dump(theirs)
        for dump in (ours, theirs):
            dump["identity"]["physical"].pop("run_result_hash")
        assert ours == theirs, request
    two = example_project_of_kind("two_beam_interference")
    readings = EvaluationSession().evaluate(two, outputs=("readings",))
    assert readings.result.evaluation is not None
    ours, theirs = readings.results(), load_run(run_optical_project(two))
    for path in ("SRC#0", "SRC#0.0"):
        wavefront = ours.wavefront("crossing_plane", path)
        assert wavefront["status"] == "OK" and wavefront == theirs.wavefront("crossing_plane", path), path
