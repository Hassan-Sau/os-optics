"""The results of an in-memory narrowed evaluation, audited query by query (``RunResults.QUERY_REQUIRES``): a query reads
every capability it needs and is refused (``NOT_COMPUTED``, naming each one not computed) when one did not run; a
composite document leaves out and names the parts it did not compute; the analyses run on a narrowed evaluation give the
complete run's document where everything they read was computed, and state what was not computed otherwise -- never an
empty answer, a zero or a reason that belongs to something else."""

from __future__ import annotations

import pytest

from os_optics.local import EvaluationSession, evaluate_project, example_project, example_project_of_kind, run_optical_project
from os_optics.results import NOT_COMPUTED, QUERY_REQUIRES, ResultsError, load_run

from local_helpers import catalog_document, project, source, with_lossless_bulk
from test_analysis_prepared_selection import _dump, _lens_project, _prescription
from test_extended_source_radiometry import detector, emitter
from test_transverse_handoff import _stop_project

QE = {"rows": [{"wavelength_nm": 700.0, "qe": 0.5}, {"wavelength_nm": 900.0, "qe": 0.5}],
      "source": {"kind": "DATASHEET", "reference": "test"}}
DETECTOR = {"quantum_efficiency": QE, "exposure_s": 1e-3, "pixel_pitch_mm": [0.05, 0.05], "fill_factor": 1.0,
            "pixel_count": [4, 4], "full_well_e": 1.0e9, "read_noise_e": 2.0,
            "dark_current_e_per_s": 100.0, "gain_e_per_adu": 0.5, "bit_depth": 12, "characterization": QE["source"]}


def _refused(query, *capabilities: str) -> str:
    with pytest.raises(ResultsError) as refused:
        query()
    assert refused.value.code == NOT_COMPUTED, refused.value.message
    for capability in capabilities:
        assert repr(capability) in refused.value.message, (capability, refused.value.message)
    return refused.value.message


def test_every_query_names_capabilities_of_the_table():
    from os_optics.evaluation.capabilities import CAPABILITY_BY_ID

    for query, capabilities in QUERY_REQUIRES.items():
        assert capabilities and set(capabilities) <= set(CAPABILITY_BY_ID), query
    # (the handoff -- and the fields lit by it -- read each beam's truncations and the plane bounds' issues)
    for query in ("handoff", "focal_field", "pupil_field"):
        assert QUERY_REQUIRES[query] == ("plane_readings", "power_ledger", "plane_bounds")
    assert QUERY_REQUIRES["rays"] == QUERY_REQUIRES["wavefront"] == ("ray_paths", "plane_readings")
    # (the path list is the beam trace's and the secondary rays'; the grating orders and conversions are the beam trace's)
    assert QUERY_REQUIRES["paths"] == QUERY_REQUIRES["segments"] == ("beam_trace", "secondary_rays")
    assert QUERY_REQUIRES["grating_orders"] == QUERY_REQUIRES["nonlinear"] == ("beam_trace",)


def test_the_paths_of_a_trace_without_the_secondary_rays_are_refused_never_a_shorter_list():
    """An extended emitter beside a laser: the emitter's rays are paths of the result, added by the secondary rays. An
    evaluation of the beam trace alone holds the laser's path only, so its path queries are refused and its scope names
    what the secondary rays would have produced; extended, they are the complete evaluation's."""
    document = project([emitter(rays=64), source("LASER", (-5.0, 0.0, 0.0), wavelength_nm=550.0, power_w=0.5)],
                       [detector("D", 10.0, 5.0)])
    complete = EvaluationSession().evaluate(document, outputs=("complete",))
    whole = complete.results()
    assert len(whole.paths()) > 1 and {row["source_id"] for row in whole.paths()} == {"EM", "LASER"}
    assert len(whole.paths()) == len(complete.result.paths) == complete.result.summary.path_count
    for outputs in (("beam_trace",), ("ray_paths",)):
        traced = EvaluationSession().evaluate(document, outputs=outputs)
        assert len(traced.result.paths) == 1 and traced.result.manifest.trace_hash != complete.result.manifest.trace_hash
        results = traced.results()
        for query in (lambda: results.paths(), lambda: results.segments(), lambda: results.paths(path="LASER#0"),
                      lambda: results.table("paths"), lambda: results.export("segments", "csv")):
            assert "which this evaluation did not compute" in _refused(query, "secondary_rays")
        described = results.describe()
        assert described["available"]["paths"] is None and described["available"]["sampled_sources"] is None
        (missing,) = [row for row in described["evaluation"]["not_computed"] if row["capability"] == "secondary_rays"]
        assert {"paths", "summary.path_count", "manifest.trace_hash", "sampled_sources"} <= set(missing["produces"])
        assert results.grating_orders() == [] and results.nonlinear() == []  # (the beam trace's own: answered)
        extended = traced.extend(outputs=("secondary_rays",))
        assert extended.results().paths() == whole.paths() and extended.results().segments() == whole.segments()
        assert extended.result.manifest.trace_hash == complete.result.manifest.trace_hash
        assert extended.results().describe()["available"]["paths"] == whole.describe()["available"]["paths"]


def test_a_refusal_names_the_call_that_computes_what_is_missing():
    """The results of a Project evaluation are extended with outputs; the result of an Engine evaluation is evaluated
    again with capabilities -- the refusal names the call its reader made."""
    import os_optics
    from os_optics.results import handoff, results_from_engine

    evaluation = EvaluationSession().evaluate(example_project())
    local = _refused(lambda: evaluation.results().power_budget(), "power_ledger")
    assert local.endswith("evaluate with outputs that include 'power_ledger' (Evaluation.extend(outputs=['power_ledger']))")
    request = evaluation.request
    bare = results_from_engine(os_optics.evaluate(request, capabilities=("readings",)))
    for query, capability in ((lambda: bare.power_budget(), "power_ledger"), (lambda: bare.targets(), "requirements"),
                              (lambda: bare.exposures(), "surface_exposures")):
        message = _refused(query, capability)
        assert message.endswith(f"evaluate with capabilities that include '{capability}' "
                                f"(os_optics.evaluate(request, capabilities=['{capability}']))")
        assert "outputs" not in message and "Evaluation.extend" not in message
    message = _refused(lambda: bare.handoff("focal_plane"), "power_ledger", "plane_bounds")
    assert "os_optics.evaluate(request, capabilities=['power_ledger', 'plane_bounds'])" in message
    # (the way out the message names computes it)
    again = results_from_engine(os_optics.evaluate(request, capabilities=("readings", "power_ledger")))
    assert again.power_budget()["available"] is True
    # (the module-level handoff reads what RunResults.handoff reads: without the plane bounds its validity is incomplete)
    partial = os_optics.evaluate(request, capabilities=("plane_readings", "power_ledger"))
    with pytest.raises(ResultsError) as refused:
        handoff(partial, request.model_dump(mode="json"), "focal_plane")
    assert refused.value.code == NOT_COMPUTED
    assert refused.value.message == ("handoff reads plane_bounds, which this evaluation did not compute: evaluate with "
                                     "capabilities that include 'plane_bounds' "
                                     "(os_optics.evaluate(request, capabilities=['plane_bounds']))")
    whole = os_optics.evaluate(request, capabilities=("plane_readings", "power_ledger", "plane_bounds"))
    assert handoff(whole, request.model_dump(mode="json"), "focal_plane") == handoff(
        os_optics.validate(request), request.model_dump(mode="json"), "focal_plane")


def test_a_refusal_of_an_evaluation_made_with_an_exclusion_names_a_way_out_that_works():
    """``Evaluation.extend`` keeps the exclusions the evaluation was made with: where it would refuse the outputs a query
    reads, the refusal names every exclusion that stands in the way, as the caller wrote it, and a new evaluation
    without them, not ``extend``; where it computes them, the refusal names ``extend`` as before."""
    from os_optics.evaluation.plan import EXCLUDED_CAPABILITY_REQUIRED, EvaluationPlanError

    document = example_project()
    session = EvaluationSession()
    evaluation = session.evaluate(document, outputs=("readings",), exclude=("power_ledger",))
    results = evaluation.results()
    # (the excluded capability itself; one that was not requested and cannot be computed without the excluded one)
    for query, capability, needs in (
            (lambda: results.power_budget(), "power_ledger", ""),
            (lambda: results.detection("focal_plane"), "plane_bounds", ", which plane_bounds cannot be computed without"),
            (lambda: results.exposures(), "surface_exposures", ", which surface_exposures cannot be computed without")):
        message = _refused(query, capability)
        assert "Evaluation.extend(outputs=" not in message, message
        assert (f"it was made with exclude=power_ledger, and Evaluation.extend keeps the exclusion of power_ledger{needs}. "
                f"Evaluate the Project again without that exclusion, with outputs that include '{capability}' "
                "(EvaluationSession.evaluate(project, outputs=[...])") in message, message
        # (the call the refusal no longer names is refused)
        with pytest.raises(EvaluationPlanError) as kept:
            evaluation.extend(outputs=[capability])
        assert kept.value.code == EXCLUDED_CAPABILITY_REQUIRED and kept.value.context["capability"] == "power_ledger"
    # (the way out it names computes it, in the same session without a second beam trace)
    again = session.evaluate(document, outputs=("readings", "power_ledger"))
    assert again.results().power_budget()["available"] is True
    assert session.statistics()["executions"]["beam_trace"] == 1
    # (an exclusion the missing capability does not need: extend computes it, and the refusal names extend)
    other = EvaluationSession().evaluate(document, outputs=("readings",), exclude=("plane_bounds",))
    message = _refused(lambda: other.results().power_budget(), "power_ledger")
    assert message.endswith("evaluate with outputs that include 'power_ledger' (Evaluation.extend(outputs=['power_ledger']))")
    assert other.extend(outputs=["power_ledger"]).results().power_budget()["available"] is True
    # (two exclusions that both stand between the evaluation and the query: the refusal names both, and the evaluation
    # without them answers the query)
    session = EvaluationSession()
    both = session.evaluate(document, outputs=("readings",), exclude=("power_ledger", "untraced_light")).results()
    message = _refused(lambda: both.detection("focal_plane"), "plane_bounds")
    assert ("it was made with exclude=power_ledger, untraced_light, and Evaluation.extend keeps the exclusions of "
            "power_ledger and untraced_light, which plane_bounds cannot be computed without. Evaluate the Project again "
            "without those exclusions, with outputs that include 'plane_bounds' "
            "(EvaluationSession.evaluate(project, outputs=[...])") in message, message
    message = _refused(lambda: both.handoff("focal_plane"), "power_ledger", "plane_bounds")
    assert ("keeps the exclusions of power_ledger, and of untraced_light, which power_ledger and plane_bounds cannot be "
            "computed without. Evaluate the Project again without those exclusions, with outputs that include "
            "'power_ledger', 'plane_bounds' ") in message, message
    # (the clause names everything the query reads, never one capability that does not need the other exclusion)
    bounds = EvaluationSession().evaluate(document, outputs=("readings",), exclude=("untraced_light", "plane_bounds"))
    message = _refused(lambda: bounds.results().handoff("focal_plane"), "power_ledger", "plane_bounds")
    assert ("keeps the exclusions of plane_bounds, and of untraced_light, which power_ledger and plane_bounds cannot be "
            "computed without. ") in message, message
    again = session.evaluate(document, outputs=("readings", "plane_bounds"))
    assert again.results().detection("focal_plane") is not None
    assert session.statistics()["executions"]["beam_trace"] == 1
    # (a preset in exclude is named as the caller wrote it, never by a member id)
    session = EvaluationSession()
    preset = session.evaluate(document, outputs=("beam_trace",), exclude=("readings",)).results()
    message = _refused(lambda: preset.power_budget(), "power_ledger")
    assert ("it was made with exclude=readings, and Evaluation.extend keeps the exclusion of readings, which power_ledger "
            "cannot be computed without. Evaluate the Project again without that exclusion, ") in message, message
    assert "plane_readings" not in message, message
    assert session.evaluate(document, outputs=("beam_trace", "power_ledger")).results().power_budget()["available"] is True
    # (the preset and one of its members by id: both are written, both are named, and the evaluation without them works)
    session = EvaluationSession()
    twice = session.evaluate(document, outputs=("beam_trace",), exclude=("readings", "plane_readings")).results()
    message = _refused(lambda: twice.power_budget(), "power_ledger")
    assert ("it was made with exclude=readings, plane_readings, and Evaluation.extend keeps the exclusions of "
            "plane_readings and readings, which power_ledger cannot be computed without. Evaluate the Project again "
            "without those exclusions, ") in message, message
    with pytest.raises(EvaluationPlanError) as kept:  # (without one of the two only, it is refused)
        session.evaluate(document, outputs=("beam_trace", "power_ledger"), exclude=("readings",))
    assert kept.value.code == EXCLUDED_CAPABILITY_REQUIRED
    assert session.evaluate(document, outputs=("beam_trace", "power_ledger")).results().power_budget()["available"] is True


def test_the_handoff_of_a_narrowed_evaluation_is_refused_until_its_validity_is_computed():
    """A stop clips the beam: the readings alone would call it an unclipped TEM00 beam (no truncation booked, no plane-bound
    issue), so the handoff, its table and the fields lit by it are refused; extended, it is the complete run's."""
    document = _stop_project({"opening_radius_mm": "1.2"})
    complete = load_run(run_optical_project(document)).handoff("after")
    assert complete["beams"][0]["validity"]["beam_parameters"] == "TRUNCATED_GAUSSIAN"
    readings = EvaluationSession().evaluate(document)
    results = readings.results()
    for query in (lambda: results.handoff("after"), lambda: results.table("handoff", monitor="after"),
                  lambda: results.export("handoff", "json", monitor="after"),
                  lambda: results.focal_field("after", {}), lambda: results.pupil_field("after", {})):
        _refused(query, "power_ledger", "plane_bounds")
    extended = readings.extend(outputs=("power_ledger", "plane_bounds")).results().handoff("after")
    assert extended == complete
    # (the module-level handoff of a bare narrowed result refuses what it cannot state, too)
    from os_optics.results import handoff

    with pytest.raises(ResultsError) as bare:
        handoff(readings.result, readings.request.model_dump(mode="json"), "after")
    assert bare.value.code == NOT_COMPUTED and "power_ledger" in bare.value.message


def test_the_irradiance_map_without_the_plane_bounds_states_the_coherent_map_not_computed():
    catalog = with_lossless_bulk(catalog_document())
    document = example_project(catalog)
    complete = load_run(run_optical_project(document, catalog=catalog)).irradiance_map("focal_plane", roi=_ROI)
    readings = EvaluationSession(catalog=catalog).evaluate(document)
    narrowed = readings.results().irradiance_map("focal_plane", roi=_ROI)
    # (the incoherent map reads the plane readings only: the complete run's)
    assert narrowed["incoherent_W_per_mm2"] == complete["incoherent_W_per_mm2"]
    assert narrowed["grid_power_incoherent_W"] == complete["grid_power_incoherent_W"]
    # (the coherent map and the untraced share read the plane bounds: not computed, never UNTRACED_POWER_TOO_LARGE)
    assert complete["coherent_W_per_mm2"] is not None and complete["coherent_reason"] is None
    assert narrowed["coherent_W_per_mm2"] is None and narrowed["coherent_reason"] == "NOT_COMPUTED"
    assert narrowed["untraced_power_W"] is None and narrowed["untraced_fraction"] is None and narrowed["map_complete"] is False
    assert narrowed["needs"] == ["plane_bounds"] and "needs" not in complete
    assert narrowed["uniformity"]["reason"] == "NOT_COMPUTED" and complete["uniformity"]["min_over_mean"] is not None
    _rows, summary = readings.results().table("irradiance", monitor="focal_plane")
    assert summary["coherent_reason"] == "NOT_COMPUTED" and summary["needs"] == ["plane_bounds"]
    # (a detector reading the narrowed map states its signal-to-noise ratio not computed, for the same reason)
    snr = readings.results().detector_response("focal_plane", DETECTOR)["quantities"]["snr"]
    assert snr["reason"] == "NOT_COMPUTED" and "plane_bounds" in snr["message"]
    extended = readings.extend(outputs=("plane_bounds",)).results().irradiance_map("focal_plane", roi=_ROI)
    assert extended == complete


_ROI = {"shape": "circle", "center_uv_mm": [0.0, 0.0], "radius_mm": 0.05}


def test_a_monitor_document_names_what_it_did_not_compute():
    catalog = with_lossless_bulk(catalog_document())
    document = example_project_of_kind("two_beam_interference", catalog)
    complete = load_run(run_optical_project(document, catalog=catalog))
    session = EvaluationSession(catalog=catalog)
    readings = session.evaluate(document)
    plane = readings.results().monitor("crossing_plane")
    for section in ("interference", "targets", "detection", "polarization_mixtures"):
        assert section not in plane
    assert plane["not_computed"] == [{"section": "interference", "capability": "beam_pairs"},
                                     {"section": "targets", "capability": "requirements"},
                                     {"section": "detection", "capability": "plane_bounds"},
                                     {"section": "polarization_mixtures", "capability": "power_ledger"}]
    their_plane = complete.monitor("crossing_plane")
    assert their_plane["interference"] and their_plane["targets"] and "not_computed" not in their_plane
    # (the targets selected are on the plane; the others are named as not selected)
    declared = [t.target_id for t in readings.request.project.targets]
    selected = session.evaluate(document, targets=declared[:1]).results().monitor("crossing_plane")
    assert [t["target_id"] for t in selected["targets"]] == declared[:1]
    assert [t["target_id"] for t in selected["not_evaluated"]] == declared[1:]
    extended = readings.extend(outputs=("requirements", "beam_pairs", "plane_bounds")).results().monitor("crossing_plane")
    assert extended == their_plane
    # (describe() counts nothing it did not compute: null, not 0 or false)
    available = readings.results().describe()["available"]
    assert available["two_beam_interference"] is None and available["targets"] is None and available["power_budget"] is None
    assert available["exposures"] is None and available["monitors"] == ["crossing_plane"] and available["ray_data"] is True
    assert complete.describe()["available"]["two_beam_interference"] == 1
    # (the wavefront reads the ray paths and the plane readings: with the rays alone it is not computed, never false)
    rays_only = session.evaluate(document, outputs=("ray_paths",)).results().describe()["available"]
    assert rays_only["wavefront"] is None and rays_only["ray_data"] is True and rays_only["ray_data_recording"] == "full"
    assert rays_only["monitors"] is None and rays_only["paths"] is None
    assert available["wavefront"] is True is complete.describe()["available"]["wavefront"]


def test_an_unknown_element_is_refused_on_a_narrowed_evaluation():
    results = EvaluationSession().evaluate(example_project()).results()
    for query in (lambda: results.nonlinear("no_such_element"), lambda: results.grating_orders("no_such_element"),
                  lambda: results.table("grating-orders", element="no_such_element"),
                  lambda: results.export("nonlinear", "json", element="no_such_element")):
        with pytest.raises(ResultsError) as refused:
            query()
        assert refused.value.code == "ELEMENT_NOT_FOUND" and "['L1']" in refused.value.message
    assert results.nonlinear("L1") == [] and results.grating_orders("L1") == []


def test_the_ghosts_of_a_readings_evaluation_equal_the_complete_runs():
    document = _lens_project(_prescription(None, None, 5.0, 20.0), waist_mm=1.0)
    complete = load_run(run_optical_project(document, catalog=catalog_document())).ghosts("image")
    evaluation = evaluate_project(document, outputs=("readings",), catalog=catalog_document())
    assert evaluation.result.evaluation is not None
    ours, theirs = _dump(evaluation.results().ghosts("image")), _dump(complete)
    for dump in (ours, theirs):
        dump["identity"]["physical"].pop("run_result_hash")
    assert ours == theirs
    # (the document carries the session's build id: the build is not identified a second time)
    assert ours["identity"]["physical"]["engine_build_id"] == evaluation.session.build_id
