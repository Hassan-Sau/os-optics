"""Selective evaluation (``os_optics.evaluate``): what a narrowed scope computes, and what it says it did not.

A selection runs the capabilities it names and what they require, through the nodes the complete evaluation runs. The
quantities it retains equal the complete run's field by field; a capability outside the scope is never entered, the
sections it produces are named as not computed, a plane whose bounds did not run reports ``NOT_COMPUTED`` (never a known
zero, never a bound), and every target the evaluation was asked to assess has a result: one that a capability outside the
scope would decide is undetermined, one on a stage the request turns off reads as in ``validate``.
"""

from __future__ import annotations

import json
import math
from collections import Counter

import numpy as np
import pytest

from os_optics import EngineConfiguration, ValidationRequest, evaluate, validate
from os_optics import builders as B
from os_optics.contracts import ProjectSnapshot, Target
from os_optics.evaluation import EvaluationPlanError, EvaluationState, capability
from os_optics.local import lower_project
from os_optics.run_control import monitored

from fixture_scenes import FIXTURES
from test_power_exposure_ratings import _document, _rating

READINGS = ("scene", "beam_trace", "secondary_rays", "plane_readings")
#: What the plane readings report of a plane, whatever else the evaluation runs.
READING_FIELDS = ("observation_id", "arrivals", "plane", "superpositions", "power_basis", "sampled_light", "outside_active_area",
                  "power_sampling_resolution_W")
#: Fixtures of several kinds: a lens train, two-beam interference, branching beams (a Mach-Zehnder with a window; nineteen
#: splitters with layout checks and thirty targets), secondary rays, a clipped beam, crystals.
SELECTIVE = ("keplerian_expander", "interference_405_500", "mzi_window", "benchmark_standard", "trace_extended_emitter",
             "aperture_clipping", "trace_birefringent_crystals")


def _rated_request() -> ValidationRequest:
    """A 2 W beam focused onto a window that states a power rating, with a utilization, a power and a beam-size target."""
    targets = [{"target_id": "util", "metric": "power_rating_utilization", "device_instance_id": "W", "maximum": 1.0},
               {"target_id": "p", "metric": "power_W", "observation_id": "screen", "minimum": 0.1},
               {"target_id": "r", "metric": "beam_radius_mm", "observation_id": "screen", "maximum": 5.0}]
    return lower_project(_document([_rating("CW", 1.0e6, "W/cm2")], targets=targets)).request


def _requests() -> dict[str, ValidationRequest]:
    return {**{name: FIXTURES[name]() for name in SELECTIVE}, "rated_window": _rated_request()}


def _dark_and_lit(targets: tuple[Target, ...] = ()) -> ValidationRequest:
    """One laser, a plane on its beam and a plane beside it that no traced beam reaches."""
    project = ProjectSnapshot(
        name="a lit plane and a plane beside the beam",
        components=(B.laser("L", wavelength_nm=633.0, waist_radius_mm=0.5, position=(0, 0, 20), direction=(1, 0, 0), power_W=1e-3),),
        observations=(B.observation_plane("lit", position=(100, 0, 20), normal=(1, 0, 0), radius_mm=10.0),
                      B.observation_plane("beside", position=(100, 80, 20), normal=(1, 0, 0), radius_mm=5.0)),
        targets=targets)
    return ValidationRequest(project=project, configuration=EngineConfiguration(run_layout=False, run_control=False))


#: How the complete run ends the message of a plane a beam of out-of-scope power reaches.
BOUNDS_NOT_STATED = ", so the plane's power and its bounds are not stated"


def _assert_readings_equal(selective, complete) -> None:
    """The plane readings of ``selective`` are the complete run's, field by field."""
    assert [o.observation_id for o in selective.observations] == [o.observation_id for o in complete.observations]
    for got, whole in zip(selective.observations, complete.observations):
        for name in READING_FIELDS:
            assert getattr(got, name) == getattr(whole, name), (got.observation_id, name)
        # (every launched wavelength is listed, as in the complete run)
        assert [s.wavelength_nm for s in got.power_by_wavelength] == [s.wavelength_nm for s in whole.power_by_wavelength]
        if whole.power_lit:
            assert got.total_power_W == whole.total_power_W and got.status == whole.status
            if whole.quantity_state == "OUT_OF_MODEL_SCOPE":
                # (a beam of out-of-scope power reaches the plane: it says so and states no lower bound, as in the complete run)
                assert got.power_lower_bound_W is None, got.observation_id
                assert got.message.startswith(f"{whole.message.removesuffix(BOUNDS_NOT_STATED)}; "), got.observation_id
            else:
                assert got.power_lower_bound_W == whole.power_lower_bound_W, got.observation_id
            # (each line carries the complete run's traced power, except that a line whose light does not arrive has no
            # power value where the complete run states a known zero: that is the plane bounds' to say)
            for s, w in zip(got.power_by_wavelength, whole.power_by_wavelength):
                if s.power_W is None and w.power_W == 0.0:
                    continue
                assert (s.power_W, s.power_out_of_scope_at) == (w.power_W, w.power_out_of_scope_at), (got.observation_id, s.wavelength_nm)
        elif whole.arrivals:
            # (beams of unknown power only: no power value, graded as the complete run grades it)
            assert got.total_power_W is None and got.status == whole.status
    assert selective.manifest.trace_hash == complete.manifest.trace_hash
    for name in ("paths", "grating_orders", "nonlinear_conversions", "sampled_sources", "source_spectra"):
        assert getattr(selective, name) == getattr(complete, name), name
    assert selective.summary.path_count == complete.summary.path_count
    # (every issue of the narrowed result is one the complete run raises at the same place)
    assert not Counter((i.code, i.location) for i in selective.issues) - Counter((i.code, i.location) for i in complete.issues)


# ----------------------------------------------------------------------------------------------- retained quantities


@pytest.fixture(scope="module")
def complete_results() -> dict:
    return {name: (request, validate(request)) for name, request in _requests().items()}


@pytest.mark.parametrize("name", (*SELECTIVE, "rated_window"))
def test_the_readings_of_a_selective_evaluation_are_the_complete_run_s(complete_results, name):
    request, complete = complete_results[name]
    readings = evaluate(request, capabilities=("readings",))
    assert readings.evaluation.requested == ("readings",) and readings.evaluation.executed == READINGS
    _assert_readings_equal(readings, complete)
    assert all(o.quantity_state == "NOT_COMPUTED" and o.power_complete is False for o in readings.observations)
    assert readings.power_budget is None and readings.exposures is None and readings.interference == () and readings.targets == ()
    assert readings.outcome.execution.outcome == complete.outcome.execution.outcome == "COMPLETED"
    assert readings.outcome.requirements.declared is bool(request.project.targets)


@pytest.mark.parametrize("name", (*SELECTIVE, "rated_window"))
def test_each_narrower_scope_retains_the_complete_run_s_sections(complete_results, name):
    request, complete = complete_results[name]
    ledger = evaluate(request, capabilities=("power_ledger",))
    assert ledger.power_budget == complete.power_budget
    assert [o.polarization_mixtures for o in ledger.observations] == [o.polarization_mixtures for o in complete.observations]
    pairs = evaluate(request, capabilities=("plane_readings", "beam_pairs"), exclude=("untraced_light",))
    assert pairs.interference == complete.interference and pairs.incoherent_pairs == complete.incoherent_pairs
    _assert_readings_equal(pairs, complete)
    bounds = evaluate(request, capabilities=("plane_bounds",))
    # (the planes with their bounds, states, grades and detections are the complete run's whole plane results)
    assert bounds.observations == complete.observations and bounds.power_budget == complete.power_budget
    assert bounds.evaluation.executed == (*READINGS, "power_ledger", "untraced_light", "plane_bounds")
    with_light = evaluate(request, capabilities=("surface_exposures", "untraced_light"))
    assert with_light.exposures == complete.exposures and with_light.exposures.untraced_light_included == complete.exposures.untraced_light_included
    alone = evaluate(request, capabilities=("surface_exposures",))
    if any(c.power_ratings for c in request.project.components):
        # (a design with power ratings: the exposures are compared with them, which needs the untraced light followed)
        assert "untraced_light" in alone.evaluation.executed and alone.exposures == complete.exposures
        assert alone.exposures.untraced_light_included is True
    else:
        # (without ratings the report covers the traced beams, and following the untraced light adds nothing to it)
        assert alone.exposures == complete.exposures and alone.exposures.untraced_light_included is False
        missing = {n.capability: n for n in alone.evaluation.not_computed}["untraced_light"]
        assert missing.produces == capability("untraced_light").produces and "exposures.untraced_light_included" in missing.produces


def _near_rating_request() -> ValidationRequest:
    """The focused 2 W beam on a window rated between its traced exposure and the upper bound with the untraced light."""
    return lower_project(_document([_rating("CW", 2900.0, "W/cm2")])).request


def test_the_exposures_of_a_rated_design_never_state_a_verdict_on_the_traced_beams_alone():
    request = _near_rating_request()
    complete = validate(request)
    # (the complete run withholds the comparison: the untraced light may lift the exposure above the rating)
    assert {s.surface_id: s.rating_status for s in complete.exposures.surfaces if s.ratings} == {"W/S1": "UNKNOWN", "W/S2": "UNKNOWN"}
    assert all(not s.complete for s in complete.exposures.surfaces if s.ratings)
    assert {e.element_id: e.rating_status for e in complete.exposures.elements}["W"] == "UNKNOWN"
    state = EvaluationState()
    alone = evaluate(request, capabilities=("surface_exposures",), state=state)
    assert alone.evaluation.executed == (*READINGS, "power_ledger", "untraced_light", "surface_exposures")
    assert state.executions["untraced_light"] == 1
    assert alone.exposures == complete.exposures and alone.exposures.untraced_light_included is True
    assert "WITHIN_RATING" not in {s.rating_status for s in alone.exposures.surfaces} | {e.rating_status for e in alone.exposures.elements}
    with pytest.raises(EvaluationPlanError) as refused:
        evaluate(request, capabilities=("surface_exposures",), exclude=("untraced_light",))
    assert refused.value.code == "EXCLUDED_CAPABILITY_REQUIRED"
    assert refused.value.message == (
        "untraced_light is excluded, and surface_exposures requires untraced_light: the evaluation plan runs nothing that is "
        "excluded and computes nothing in its place. Remove untraced_light from exclude, or do not request surface_exposures.")


def test_a_rated_surface_only_untraced_light_reaches_is_in_the_exposures_alone():
    from test_closed_returns import faint_ring

    request = ValidationRequest(project=faint_ring(), configuration=EngineConfiguration(run_layout=False, run_control=False))
    complete = validate(request)
    assert "B/face" in [s.surface_id for s in complete.exposures.surfaces]
    alone = evaluate(request, capabilities=("surface_exposures",))
    assert alone.exposures == complete.exposures and "untraced_light" in alone.evaluation.executed
    assert [(e.element_id, e.rating_status, e.max_utilization) for e in alone.exposures.elements] == [
        (e.element_id, e.rating_status, e.max_utilization) for e in complete.exposures.elements]
    # (the issues of the surfaces the untraced light reaches are the complete run's)
    assert [i for i in alone.issues if (i.location or "").startswith("surface:")] == [
        i for i in complete.issues if (i.location or "").startswith("surface:")]


def test_the_requirements_alone_evaluate_every_target_as_the_complete_run_does(complete_results):
    for name in ("interference_405_500", "keplerian_expander", "benchmark_standard", "rated_window"):
        request, complete = complete_results[name]
        selective = evaluate(request, capabilities=("requirements",))
        assert selective.targets == complete.targets and selective.outcome.requirements == complete.outcome.requirements
        assert selective.evaluation.targets_not_evaluated == ()
        assert "requirements" not in [n.capability for n in selective.evaluation.not_computed]


# ----------------------------------------------------------------------------------------------- compositions


def test_a_and_c_without_b_plane_readings_and_beam_pairs_without_the_untraced_light(complete_results):
    request, complete = complete_results["interference_405_500"]
    geometric = ("period", "half_angle", "opd", "coherence")
    state = EvaluationState()
    result = evaluate(request, capabilities=("plane_readings", "beam_pairs"), exclude=("untraced_light",), targets=geometric, state=state)
    assert result.evaluation.executed == (*READINGS, "beam_pairs", "requirements")
    assert state.executions == {c: 1 for c in result.evaluation.executed}
    by_id = {t.target_id: t for t in complete.targets}
    assert result.targets == tuple(by_id[t] for t in geometric)  # (the same value, status, message and codes)
    skipped = {t.target_id: t for t in result.evaluation.targets_not_evaluated}
    assert set(skipped) == {"visibility", "exposure"} and all(t.reason == "NOT_REQUESTED" for t in skipped.values())
    assert result.outcome.requirements.declared and result.outcome.requirements.met == geometric
    not_computed = {n.capability: n for n in result.evaluation.not_computed}
    assert list(not_computed) == ["power_ledger", "untraced_light", "plane_bounds", "surface_exposures"]
    assert not_computed["untraced_light"].reason == "EXCLUDED" and not_computed["plane_bounds"].reason == "NOT_REQUESTED"
    assert not_computed["plane_bounds"].produces == capability("plane_bounds").produces
    assert "observations[].quantity_state" in not_computed["plane_bounds"].produces and not_computed["power_ledger"].produces[0] == "power_budget"
    assert result.status == "PASS"


def test_a_capability_excluded_through_a_preset_is_excluded_in_every_statement_of_the_evaluation():
    request = FIXTURES["mzi_window"]()
    by_preset = evaluate(request, capabilities=("beam_trace",), exclude=("readings",))
    by_id = evaluate(request, capabilities=("beam_trace",), exclude=("plane_readings",))
    assert {n.capability: n.reason for n in by_preset.evaluation.not_computed}["plane_readings"] == "EXCLUDED"
    assert by_preset.evaluation.not_computed == by_id.evaluation.not_computed
    # (a Project evaluation: its own scope and the Engine result's scope state one reason)
    from os_optics.local import json_api
    from os_optics.local.evaluate import EvaluationSession

    document = json_api.example_project_document()["project"]
    evaluation = EvaluationSession().evaluate(document, outputs=("beam_trace",), exclude=("readings",))
    local = {n["capability"]: n["reason"] for n in evaluation.describe_scope()["not_computed"]}
    engine = {n.capability: n.reason for n in evaluation.result.evaluation.not_computed}
    assert local["plane_readings"] == engine["plane_readings"] == "EXCLUDED" and local["model_scope"] == "EXCLUDED"
    assert all(local[c] == reason for c, reason in engine.items())
    reply = json_api.evaluate_project(document, ["beam_trace"], exclude=["readings"])
    assert {n["capability"]: n["reason"] for n in reply["evaluation"]["not_computed"]}["plane_readings"] == "EXCLUDED"


def test_b_only_the_beam_trace_alone(complete_results):
    request, complete = complete_results["mzi_window"]
    result = evaluate(request, capabilities=("beam_trace",))
    assert result.evaluation.executed == ("scene", "beam_trace")
    assert result.paths == complete.paths and result.summary.path_count == complete.summary.path_count
    assert result.manifest.trace_hash == complete.manifest.trace_hash
    assert result.observations == () and result.summary.observation_count == 0 and result.power_budget is None
    assert [n.capability for n in result.evaluation.not_computed][:2] == ["secondary_rays", "plane_readings"]
    assert result.evaluation.not_computed[1].produces == ("observations",)


def test_b_only_the_layout_checks_alone():
    traced = FIXTURES["layout_example"]()
    complete = validate(traced)
    alone = evaluate(traced, capabilities=("layout",))
    # (the layout checks test the traced beams against the envelopes: the trace runs, nothing after it does)
    assert alone.evaluation.executed == ("scene", "beam_trace", "secondary_rays", "layout") and alone.layout == complete.layout
    # (the layout's status enters the run status only through a declared layout target: alone, nothing else counts)
    assert alone.observations == () and alone.layout.status == complete.layout.status
    assert alone.status.value == "NOT_APPLICABLE"
    untraced = traced.model_copy(update={"configuration": traced.configuration.model_copy(update={"run_physics": False})})
    state = EvaluationState()
    result = evaluate(untraced, capabilities=("layout",), state=state)
    assert state.executions == {"layout": 1} and result.layout == validate(untraced).layout
    # (the request traces nothing: its complete scope is the layout checks and the requirements)
    assert [n.capability for n in result.evaluation.not_computed] == ["requirements"]
    assert [n.capability for n in evaluate(untraced, capabilities=("requirements",)).evaluation.not_computed] == ["layout"]


def test_b_only_the_control_analysis_alone():
    request = FIXTURES["control_recoverability"]()
    complete = validate(request)
    state = EvaluationState()
    alone = evaluate(request, capabilities=("control",), state=state)
    # (the control analysis traces its own readings: nothing of the design's evaluation runs for it)
    assert state.executions == {"control": 1} and alone.evaluation.executed == ("control",)
    assert alone.control == complete.control and alone.manifest.random_seed == complete.manifest.random_seed
    assert alone.observations == () and alone.manifest.trace_hash is None and alone.status == complete.control.status
    assert evaluate(request, state=state).manifest.result_hash == complete.manifest.result_hash and state.executions["control"] == 1


def test_an_exclusion_a_selection_requires_is_refused_before_anything_runs():
    request = FIXTURES["interference_405_500"]()
    state = EvaluationState()
    with pytest.raises(EvaluationPlanError) as refused:
        evaluate(request, capabilities=("plane_bounds",), exclude=("untraced_light",), state=state)
    assert refused.value.code == "EXCLUDED_CAPABILITY_REQUIRED"
    assert refused.value.message == (
        "untraced_light is excluded, and plane_bounds requires untraced_light: the evaluation plan runs nothing that is excluded "
        "and computes nothing in its place. Remove untraced_light from exclude, or do not request plane_bounds.")
    with pytest.raises(EvaluationPlanError) as by_target:
        evaluate(request, capabilities=("readings",), targets=("visibility",), exclude=("untraced_light",), state=state)
    assert "target 'visibility' (metric fringe_visibility) reads plane_bounds" in by_target.value.message
    assert state.executions == {} and state.retained == ()
    with pytest.raises(EvaluationPlanError) as unknown:
        evaluate(request, capabilities=("record",))
    assert unknown.value.code == "UNKNOWN_CAPABILITY"


# ----------------------------------------------------------------------------------------------- not computed


def test_a_complete_evaluation_states_no_scope(complete_results):
    request, complete = complete_results["interference_405_500"]
    assert complete.evaluation is None and "evaluation" not in json.loads(complete.model_dump_json())
    assert evaluate(request, capabilities=("complete",)).manifest.result_hash == complete.manifest.result_hash
    narrowed = json.loads(evaluate(request, capabilities=("readings",)).model_dump_json())
    assert narrowed["evaluation"]["requested"] == ["readings"] and narrowed["evaluation"]["executed"] == list(READINGS)


def test_a_plane_without_bounds_is_not_computed_never_a_known_zero_or_a_bound():
    request = _dark_and_lit()
    complete = validate(request)
    whole = {o.observation_id: o for o in complete.observations}
    assert whole["beside"].quantity_state == "KNOWN_ZERO" and whole["beside"].total_power_W == 0.0 and whole["beside"].status == "PASS"
    readings = evaluate(request, capabilities=("readings",))
    planes = {o.observation_id: o for o in readings.observations}
    lit, beside = planes["lit"], planes["beside"]
    assert lit.quantity_state == beside.quantity_state == "NOT_COMPUTED"
    assert lit.total_power_W == whole["lit"].total_power_W and lit.status == whole["lit"].status == "PASS"
    assert lit.power_complete is False and all(s.complete is False for s in lit.power_by_wavelength)
    assert beside.total_power_W is None and beside.status == "NOT_APPLICABLE" and beside.power_complete is False
    # (what the plane bounds compute is not a number here: the untraced power is null, per plane and per wavelength,
    # where the complete run states it)
    assert whole["lit"].untraced_power_W is not None and all(s.untraced_power_W is not None for s in whole["lit"].power_by_wavelength)
    assert lit.power_by_wavelength
    # (every launched wavelength is listed on every plane, as in the complete run; where none of its light arrives it has
    # no power value: whether that is a known zero is the plane bounds' to say)
    assert [(s.wavelength_nm, s.power_W, s.complete, s.untraced_power_W) for s in beside.power_by_wavelength] == [(633.0, None, False, None)]
    assert [s.wavelength_nm for s in whole["beside"].power_by_wavelength] == [633.0]
    for plane in (lit, beside):
        assert plane.untraced_power_W is None
        assert all(s.untraced_power_W is None and s.complete is False for s in plane.power_by_wavelength)
        dumped = plane.model_dump(mode="json")
        assert dumped["untraced_power_W"] is None and all(s["untraced_power_W"] is None for s in dumped["power_by_wavelength"])
    assert type(readings).model_validate_json(readings.model_dump_json()) == readings
    missing = {n.capability: n.produces for n in readings.evaluation.not_computed}["plane_bounds"]
    assert {"observations[].untraced_power_W", "observations[].power_by_wavelength[].untraced_power_W"} <= set(missing)
    assert beside.message == ("no traced beam reaches this observation plane; whether light the trace does not follow reaches it "
                              "was not computed (the evaluation did not run the plane bounds)")
    for plane in (lit, beside):
        dumped = plane.model_dump(mode="json")
        assert "power_upper_bound_W" not in dumped and "absorbed_power_upper_bound_W" not in dumped and "detection" not in dumped
    (issue,) = [i for i in readings.issues if i.location == "observation:beside"]
    assert issue.code == "OBSERVATION_NOT_REACHED" and issue.status == "NOT_APPLICABLE"
    assert issue.message.endswith("was not computed (the evaluation did not run the plane bounds)")
    # (nothing that was not computed grades the result; the complete run's dark plane is a notice too)
    assert complete.status == "PASS" and readings.status == "PASS"
    assert [n.capability for n in readings.evaluation.not_computed] == [
        "power_ledger", "untraced_light", "plane_bounds", "surface_exposures", "beam_pairs", "requirements"]
    assert all(n.reason == "NOT_REQUESTED" and n.produces == capability(n.capability).produces for n in readings.evaluation.not_computed)


def test_a_lit_plane_lists_the_launched_lines_that_do_not_reach_it_without_the_plane_bounds():
    """Two lasers, a plane on one beam only: without the plane bounds the plane lists both lines, the other one with no
    power value (the complete run states it as a known zero)."""
    project = ProjectSnapshot(
        name="two lines, one plane on the red beam",
        components=(B.laser("R", wavelength_nm=633.0, waist_radius_mm=0.5, position=(0, 0, 20), direction=(1, 0, 0), power_W=1e-3),
                    B.laser("G", wavelength_nm=532.0, waist_radius_mm=0.5, position=(0, 80, 20), direction=(1, 0, 0), power_W=1e-3)),
        observations=(B.observation_plane("red", position=(100, 0, 20), normal=(1, 0, 0), radius_mm=10.0),))
    request = ValidationRequest(project=project, configuration=EngineConfiguration(run_layout=False, run_control=False))
    (whole,) = validate(request).observations
    assert [(s.wavelength_nm, s.power_W, s.complete) for s in whole.power_by_wavelength] == [(532.0, 0.0, True), (633.0, 1e-3, True)]
    for selection in (("readings",), ("power_ledger",)):
        narrowed = evaluate(request, capabilities=selection)
        (plane,) = narrowed.observations
        assert [(s.wavelength_nm, s.power_W, s.complete, s.untraced_power_W) for s in plane.power_by_wavelength] == [
            (532.0, None, False, None), (633.0, 1e-3, False, None)]
        missing = {n.capability: n.produces for n in narrowed.evaluation.not_computed}["plane_bounds"]
        # (the plane bounds state the power of a wavelength no traced light brings; an arriving wavelength's is the readings')
        assert "observations[].power_by_wavelength[].power_W (of a wavelength no traced light brings)" in missing
        assert "observations[].power_by_wavelength[].power_W" not in missing


def test_a_plane_part_of_a_traced_beam_reaches_says_so_without_the_plane_bounds():
    """The partial crossing is a fact of the trace: a narrowed result states it as the complete one does, and never says
    that no traced beam reaches the plane."""
    request = FIXTURES["trace_partial_beams"]()
    complete = {o.observation_id: o for o in validate(request).observations}
    reached = "part of a traced beam reaches this observation plane before observation plane D, while the beam's chief ray does not cross it"
    assert complete["T"].message.startswith(f"{reached}: it receives between 0 W and ")
    for selection in (("readings",), ("plane_readings",)):
        readings = evaluate(request, capabilities=selection)
        plane = next(o for o in readings.observations if o.observation_id == "T")
        assert plane.message == (f"{reached}; its power there, and whether light the trace does not follow reaches the plane, "
                                 "were not computed (the evaluation did not run the plane bounds)")
        assert (plane.status, plane.quantity_state, plane.total_power_W, plane.untraced_power_W) == (
            "NOT_APPLICABLE", "NOT_COMPUTED", None, None)
        (issue,) = [i for i in readings.issues if i.location == "observation:T" and i.code == "OBSERVATION_NOT_REACHED"]
        assert issue.message == plane.message and issue.status == "NOT_APPLICABLE"
        assert issue.numerical_context["partial_before_surface_ids"] == ["observation/D"]
        assert issue.suggested_action == ("place the plane where the beam's chief ray crosses it, or outside the beam's envelope, "
                                          "to read its power; evaluate with plane_bounds for the upper bound of the power there")
        assert not any("no traced beam reaches" in o.message or "no beam reaches" in o.message for o in readings.observations
                       if o.observation_id == "T")


def test_a_plane_only_beams_of_unknown_power_reach_has_no_power_value_without_the_plane_bounds():
    """A grating that states no order efficiency: the beams it sends on reach the plane at known positions and directions
    with no known power. Without the plane bounds the plane has no power value and reads UNKNOWN, as in the complete run
    -- never a traced 0 W that passes."""
    request = FIXTURES["trace_grating_orders"]()
    complete = validate(request)
    whole = next(o for o in complete.observations if o.arrivals and not o.power_lit)
    assert whole.total_power_W is None and whole.status == "UNKNOWN"
    for selection in (("readings",), ("plane_readings",), ("power_ledger",), ("surface_exposures",), ("beam_pairs",)):
        narrowed = evaluate(request, capabilities=selection)
        _assert_readings_equal(narrowed, complete)
        plane = next(o for o in narrowed.observations if o.observation_id == whole.observation_id)
        assert (plane.status, plane.quantity_state, plane.total_power_W, plane.untraced_power_W) == (
            "UNKNOWN", "NOT_COMPUTED", None, None), selection
        assert plane.message == (f"{len(whole.arrivals)} beam(s) reach this observation plane at known positions and directions; "
                                 "their power is not known; whether light the trace does not follow reaches the plane was not "
                                 "computed (the evaluation did not run the plane bounds)")
        assert all(s.power_W is None for s in plane.power_by_wavelength)
        assert narrowed.status == "UNKNOWN"


def _out_of_scope_and_taken_before() -> ValidationRequest:
    """Two beams on the plane T2: the power of L is out of model scope behind its fold mirror (an elliptical beam folded
    out of plane has no Gaussian state, and the mirror's support reaches beyond the plane 0.25 mm behind the fold), and the
    dump D2 just behind the plane may take a bounded share of A before it. The plane T, on a third beam S whose tail the
    dump D may meet, has a bounded share only."""
    tilt = math.radians(45.0)
    normal = np.array((0.956828365, -0.064123474, 0.283491904))
    place = np.array((0.22612451, 0.105151786, -0.013936101))
    across = np.cross(normal, (0.0, 0.0, 1.0))
    across = across / np.linalg.norm(across)
    centre = place + 12.0 * across  # (where A crosses T2)

    def vec(v) -> tuple[float, float, float]:
        return tuple(float(x) for x in v)

    components = (
        B.laser("S", wavelength_nm=780.0, power_W=1e-3, waist_radius_mm=0.088, position=(-60.0, 0, 300.0), direction=(1.0, 0.0, 0.0),
                waist_position_z_mm=38.0),
        B.beam_dump("D", position=(-2.30460535, -1.44934377, 300.91578842), axis=(0.75095129, 0.57052169, -0.33253143),
                    radius_mm=2.2339150461982724),
        B.beam_dump("END", position=(30.0, 0.0, 300.0), axis=(1.0, 0.0, 0.0), radius_mm=20.0),
        B.laser("L", wavelength_nm=780.0, waist_radius_mm=0.056094031752207646, waist_radius_y_mm=0.14037063609031555, m2=1.0,
                m2_y=2.101783895176921, power_W=1e-3, position=(-30.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0),
                waist_position_z_mm=9.987034811675102),
        B.mirror("M", position=(0.0, 0.0, 0.0), normal=(-0.217551273, 0.967588023, -0.12823752), reflectance=0.99, clear_radius_mm=20.0),
        B.laser("A", wavelength_nm=780.0, power_W=1e-3, waist_radius_mm=0.3, position=vec(centre - 20.0 * normal), direction=vec(normal)),
        B.beam_dump("D2", position=vec(centre + 0.1 * 0.3 * math.tan(tilt) * normal),
                    axis=vec(-(math.cos(tilt) * normal + math.sin(tilt) * across)), radius_mm=6.0))
    planes = (B.observation_plane("T", position=(0, 0, 300.0), normal=(1.0, 0.0, 0.0), radius_mm=30),
              B.observation_plane("T2", position=vec(place), normal=vec(normal), radius_mm=60.0))
    project = ProjectSnapshot(name="an out-of-scope beam and a beam with a share taken before the plane",
                              components=components, observations=planes)
    return ValidationRequest(project=project, configuration=EngineConfiguration(run_layout=False, run_control=False))


def test_a_plane_a_beam_of_out_of_scope_power_reaches_says_so_and_states_no_lower_bound_without_the_plane_bounds():
    """That the traced value of such a plane is not the light's is a fact of the trace: a narrowed result states it as the
    complete one does, and never a lower bound taken from the other beams' shares."""
    request = _out_of_scope_and_taken_before()
    complete = validate(request)
    whole = {o.observation_id: o for o in complete.observations}
    reached = ("the power of a beam here is out of model scope behind M/face (power_out_of_scope_at): the traced beams bring "
               "0.00199 W by their chief rays' powers, which is not the light's")
    assert whole["T2"].message == f"{reached}{BOUNDS_NOT_STATED}"
    total = whole["T2"].total_power_W
    assert total == pytest.approx(1.99e-3, rel=1e-12)
    assert (whole["T2"].quantity_state, whole["T2"].power_lower_bound_W) == ("OUT_OF_MODEL_SCOPE", None)
    # (the readings' own lower bound of T2 would take the share D2 may take of A from a total that is not the light's)
    assert [a.power_taken_before_at for a in whole["T2"].arrivals if a.power_taken_before_upper_W is not None] == [("D2/dump",)]
    assert whole["T"].quantity_state == "BOUNDED" and 0.0 < whole["T"].power_lower_bound_W < whole["T"].total_power_W
    for selection in (("readings",), ("readings", "power_ledger"), ("readings", "beam_pairs")):
        narrowed = evaluate(request, capabilities=selection)
        _assert_readings_equal(narrowed, complete)
        planes = {o.observation_id: o for o in narrowed.observations}
        plane = planes["T2"]
        assert (plane.quantity_state, plane.status, plane.total_power_W) == ("NOT_COMPUTED", whole["T2"].status, total), selection
        assert plane.power_lower_bound_W is None and "power_lower_bound_W" not in plane.model_dump(mode="json"), selection
        assert plane.message == (f"{reached}; whether light the trace does not follow reaches the plane was not computed (the "
                                 "evaluation did not run the plane bounds)"), selection
        # (a plane only a bounded share is taken before keeps the lower bound of the complete run, and no message)
        assert (planes["T"].power_lower_bound_W, planes["T"].message) == (whole["T"].power_lower_bound_W, ""), selection


def test_a_plane_without_a_power_value_states_no_lower_bound_without_the_plane_bounds():
    """No power value, no lower bound, as in a complete evaluation: a plane no traced beam reaches and a plane only beams
    of unknown power reach keep no lower bound of the readings, whatever the readings carry there."""
    from os_optics.validation.pipeline import _bounds_not_computed

    dark = next(o for o in validate(_dark_and_lit()).observations if o.observation_id == "beside")
    unknown = next(o for o in validate(FIXTURES["trace_grating_orders"]()).observations if o.arrivals and not o.power_lit)
    assert not dark.lit and dark.power_lower_bound_W is None and unknown.power_lower_bound_W is None
    for whole in (dark, unknown):
        (plane,), _issues = _bounds_not_computed([whole.model_copy(update={"power_lower_bound_W": 0.0})], [], {}, [633.0])
        assert (plane.total_power_W, plane.power_lower_bound_W, plane.quantity_state) == (None, None, "NOT_COMPUTED")
        assert "power_lower_bound_W" not in plane.model_dump(mode="json")


def test_the_local_rows_of_a_plane_only_beams_of_unknown_power_reach_have_no_power_value():
    """The compact reply, the monitors, one monitor, the monitors table and its CSV of the default (narrowed) evaluation of a
    Project whose grating states no order efficiency: no 0 W, no PASS."""
    import csv
    import io

    from os_optics.local import json_api
    from os_optics.local.evaluate import EvaluationSession

    pose = {"quaternion_wxyz": [0.5, 0.5, 0.5, 0.5]}
    document = {
        "schema": "opensci.project", "project_schema_version": "2.0",
        "components": [
            {"instance_id": "SRC", "optical_role": "source", "custom_part": {"custom_part_key": "test-laser", "description": "Test laser."},
             "nominal_pose": {"position_mm": [-200.0, 0.0, 0.0], **pose},
             "source_model": {"model_kind": "GAUSSIAN_BEAM", "wavelength_nm": 600.0, "power_w": 0.001, "waist_radius_mm": 1.0,
                              "waist_position_mm": 0.0, "m2": 1.0, "polarization": {"kind": "LINEAR", "angle_deg": 0.0}}},
            {"instance_id": "G", "optical_role": "other_optical", "custom_part": {"custom_part_key": "own-g", "description": "My own optic."},
             "optical_model_id": "g", "nominal_pose": {"position_mm": [50.0, 0.0, 0.0], **pose}}],
        "observations": [{"observation_id": "zero", "pose": {"position_mm": [150.0, 0.0, 0.0], **pose},
                          "aperture": {"shape": "circle", "radius_mm": 10.0}}],
        "optical_models": [{"model_id": "g", "model": {
            "model_kind": "DIFFRACTION_GRATING", "model_version": "1", "mode": "TRANSMISSION", "line_density_per_mm": "300",
            "orders": [0, 1], "clear_half_widths_mm": ["12", "12"], "edge_half_widths_mm": ["12.5", "12.5"], "status": "SPECIFIED"}}]}
    reply = json_api.evaluate_project(document, ["readings"])
    assert reply["ok"]
    (row,) = reply["monitors"]
    assert (row["status"], row["quantity_state"], row["total_power_W"]) == ("UNKNOWN", "NOT_COMPUTED", None)
    assert "their power is not known" in row["message"]
    results = EvaluationSession().evaluate(document).results()
    (monitor,) = results.monitors()
    assert (monitor["status"], monitor["total_power_W"]) == ("UNKNOWN", None)
    one = results.monitor("zero")
    assert one["total_power_W"] is None and one["status"] == "UNKNOWN"
    (table_row,), _summary = results.table("monitors")
    assert table_row["total_power_W"] is None and table_row["status"] == "UNKNOWN"
    (csv_row,) = csv.DictReader(io.StringIO(results.export("monitors").decode("utf-8")))
    assert csv_row["total_power_W"] == "" and csv_row["status"] == "UNKNOWN"


def test_the_local_rows_of_a_plane_without_bounds_carry_no_untraced_power():
    """The compact reply and the in-memory monitors state null for the untraced power of a NOT_COMPUTED plane, never 0."""
    from os_optics.local import json_api
    from os_optics.local.evaluate import EvaluationSession

    document = json_api.example_project_document()["project"]
    reply = json_api.evaluate_project(document, ["readings"])
    assert reply["ok"] and reply["monitors"]
    for row in reply["monitors"]:
        assert row["quantity_state"] == "NOT_COMPUTED" and row["untraced_power_W"] is None
    whole = json_api.evaluate_project(document, ["plane_bounds"])
    assert all(isinstance(row["untraced_power_W"], float) for row in whole["monitors"])
    results = EvaluationSession().evaluate(document, outputs=("readings",)).results()
    assert all(row["untraced_power_W"] is None for row in results.monitors())


def test_a_target_whose_verdict_needs_the_plane_bounds_is_undetermined_without_them():
    targets = (Target(target_id="size", metric="beam_radius_mm", observation_id="lit", maximum=5.0),
               Target(target_id="dark_size", metric="beam_radius_mm", observation_id="beside", maximum=5.0),
               Target(target_id="power", metric="power_W", observation_id="lit", minimum=1e-4))
    request = _dark_and_lit(targets)
    complete = validate(request)
    whole = {t.target_id: t for t in complete.targets}
    assert whole["dark_size"].status == "FAIL" and "OBSERVATION_NOT_REACHED" in whole["dark_size"].issue_codes
    selective = evaluate(request, capabilities=("readings",), targets=("size", "dark_size"))
    assert selective.evaluation.executed == (*READINGS, "requirements")
    # (every target the evaluation was asked to assess has a result; the one the plane bounds would decide is undetermined)
    size, dark = selective.targets
    assert size == whole["size"]
    assert (dark.target_id, dark.status, dark.assessment, dark.value, dark.issue_codes) == (
        "dark_size", "UNKNOWN", "INDETERMINATE", None, ("CAPABILITY_NOT_COMPUTED",))
    assert dark.message == ("no beam matching the target is among the traced beams on the observation plane; whether light the "
                            "trace does not follow reaches the plane was not computed: evaluate with plane_bounds")
    # (the requirement record alone carries the outcome: the target is undetermined, and the status counts it)
    record = selective.outcome.requirements
    assert record.declared and record.met == ("size",) and record.indeterminate == ("dark_size",) and record.unmet == ()
    assert selective.status == "UNKNOWN" and selective.summary.target_counts == {"PASS": 1, "UNKNOWN": 1}
    # (the scope names only the target the caller did not select)
    assert [(t.target_id, t.metric, t.reason) for t in selective.evaluation.targets_not_evaluated] == [
        ("power", "power_W", "NOT_REQUESTED")]
    # (with the plane bounds in the scope the same target is evaluated, as the complete run evaluates it)
    with_bounds = evaluate(request, capabilities=("plane_bounds",), targets=("dark_size",))
    assert with_bounds.targets == (whole["dark_size"],) and with_bounds.status == "FAIL"


def test_a_two_beam_target_no_traced_pair_matches_is_undetermined_without_the_plane_bounds():
    targets = (Target(target_id="period", metric="fringe_period_nm", observation_id="beside", minimum=1.0),
               Target(target_id="separation", metric="centroid_separation_mm", observation_id="beside", maximum=1.0))
    request = _dark_and_lit(targets)
    complete = validate(request)
    assert [(t.status.value, t.issue_codes) for t in complete.targets] == [("FAIL", ("OBSERVATION_NOT_REACHED",))] * 2
    narrowed = evaluate(request, capabilities=("requirements",))
    # (a fringe metric reads the beam pairs, a geometric two-beam metric the plane readings: neither asks for the bounds)
    assert narrowed.evaluation.executed == (*READINGS, "beam_pairs", "requirements")
    for target in narrowed.targets:
        assert (target.status, target.assessment, target.value, target.issue_codes) == (
            "UNKNOWN", "INDETERMINATE", None, ("CAPABILITY_NOT_COMPUTED",))
        assert target.message == ("no pair of beams for this target is among the traced beams on the observation plane; whether "
                                  "light the trace does not follow reaches the plane was not computed: evaluate with plane_bounds")
    assert narrowed.outcome.requirements.indeterminate == ("period", "separation") and narrowed.status == "UNKNOWN"
    assert narrowed.evaluation.targets_not_evaluated == ()
    assert evaluate(request, capabilities=("requirements", "plane_bounds")).targets == complete.targets


#: What ``validate`` says of a target whose stage the request turns off, as on main: status, assessment, codes, message.
STAGE_OFF = {
    "layout": ("UNKNOWN", "INDETERMINATE", ("INVALID_INPUT",), "layout stage did not run"),
    "control": ("UNKNOWN", "INDETERMINATE", ("INVALID_INPUT",), "control stage did not run"),
    "trace": ("UNKNOWN", "INDETERMINATE", ("UNKNOWN_REFERENCE",), "unknown observation 'lit'"),
}


def _facts(target) -> tuple:
    return target.status.value, target.assessment.value, target.issue_codes, target.message


def test_a_target_on_a_stage_the_request_turns_off_has_the_result_validate_gives_it():
    base = FIXTURES["layout_example"]()
    clearance = Target(target_id="clear", metric="layout_min_clearance_mm", minimum=0.0)
    rank = Target(target_id="rank", metric="numerical_rank", minimum=1.0)
    size = Target(target_id="size", metric="beam_radius_mm", observation_id="d", maximum=50.0)
    project = base.project.model_copy(update={"targets": (clearance, rank, size)})
    with_layout = validate(base.model_copy(update={"project": project}))
    # (the Project states no control specification: the control analysis has nothing to run under the default flags)
    assert with_layout.evaluation is None and [t.target_id for t in with_layout.targets] == ["clear", "rank", "size"]
    assert _facts(with_layout.targets[1]) == STAGE_OFF["control"] and with_layout.targets[0].status == "PASS"
    off = base.model_copy(update={"project": project, "configuration": base.configuration.model_copy(update={"run_layout": False})})
    result = validate(off)
    # (a complete evaluation of the request: every declared target has a result, and the result states no scope)
    assert result.evaluation is None and "evaluation" not in json.loads(result.model_dump_json())
    clear, control, traced = result.targets
    assert _facts(clear) == STAGE_OFF["layout"] and _facts(control) == STAGE_OFF["control"] and clear.value is None
    assert traced == with_layout.targets[2] and result.layout is None
    record = result.outcome.requirements
    assert record.indeterminate == ("clear", "rank") and record.met == ("size",) and result.status == "UNKNOWN"
    assert result.summary.target_counts == {"PASS": 1, "UNKNOWN": 2}
    # (a selection of the same request gives the target the same result: the request, not the selection, turns the stage off)
    narrowed = evaluate(off, capabilities=("requirements",), targets=("clear",))
    assert narrowed.targets == (clear,) and narrowed.outcome.requirements.indeterminate == ("clear",)
    assert [t.target_id for t in narrowed.evaluation.targets_not_evaluated] == ["rank", "size"]


def test_a_request_that_does_not_trace_gives_its_targets_the_results_validate_gives():
    request = _dark_and_lit((Target(target_id="size", metric="beam_radius_mm", observation_id="lit", maximum=5.0),
                             Target(target_id="clear", metric="layout_min_clearance_mm", minimum=0.0)))
    off = request.model_copy(update={"configuration": request.configuration.model_copy(update={"run_physics": False})})
    result = validate(off)
    assert result.evaluation is None and result.observations == () and result.status == "UNKNOWN"
    assert [_facts(t) for t in result.targets] == [STAGE_OFF["trace"], STAGE_OFF["layout"]]
    assert result.outcome.requirements.indeterminate == ("size", "clear") and result.outcome.requirements.declared
    # (a two-beam target of a request that does not trace is not met, as in ``validate`` before selections existed: no
    # pair reaches the plane, and no selection left the plane bounds out)
    pairs = FIXTURES["interference_405_500"]()
    untraced = pairs.model_copy(update={"configuration": pairs.configuration.model_copy(update={"run_physics": False})})
    dark = validate(untraced)
    assert dark.evaluation is None and dark.status == "FAIL" and {t.status.value for t in dark.targets} == {"FAIL"}
    assert all("OBSERVATION_NOT_REACHED" in t.issue_codes for t in dark.targets)
    assert dark.outcome.requirements.unmet == tuple(t.target_id for t in pairs.project.targets)


# ----------------------------------------------------------------------------------------------- skipped work


class _Phases:
    """A run monitor that records the phases a run enters and the checkpoints it counts in each."""

    def __init__(self) -> None:
        self.entered: list[str] = []
        self.stack: list[str] = []
        self.steps: Counter = Counter()

    def enter(self, name: str, total: int | None, unit: str) -> None:
        self.entered.append(name)
        self.stack.append(name)

    def leave(self, name: str) -> None:
        self.stack.pop()

    def unwind(self, name: str) -> None:
        self.stack.pop()

    def step(self, count: int) -> None:
        self.steps[self.stack[-1] if self.stack else ""] += count


def _phases(request, **selection) -> _Phases:
    monitor = _Phases()
    with monitored(monitor):
        evaluate(request, **selection)
    return monitor


def test_a_capability_outside_the_scope_is_never_entered():
    request = FIXTURES["mzi_window"]()  # (a window's ghosts: the complete run follows them with probe traces)
    complete = _phases(request)
    assert [p for p in complete.entered if p != "return_analysis"] == [
        "trace", "emitters", "observation_results", "untraced_light", "bounds", "exposures", "interference", "targets", "result"]
    assert complete.steps["untraced_light"] > 0
    readings = _phases(request, capabilities=("readings",))
    assert readings.entered == ["trace", "emitters", "observation_results", "result"]
    assert set(readings.steps) <= {"trace", "observation_results"} and readings.steps["trace"] == complete.steps["trace"]
    assert _phases(request, capabilities=("beam_trace",)).entered == ["trace", "result"]
    pairs = _phases(request, capabilities=("plane_readings", "beam_pairs"), targets=())
    assert pairs.entered == ["trace", "emitters", "observation_results", "interference", "targets", "result"]
    exposures = _phases(request, capabilities=("surface_exposures",))
    assert exposures.entered == ["trace", "emitters", "observation_results", "exposures", "result"]
    untraced = FIXTURES["layout_example"]()
    untraced = untraced.model_copy(update={"configuration": untraced.configuration.model_copy(update={"run_physics": False})})
    assert _phases(untraced, capabilities=("layout",)).entered == ["result"]


def test_a_retained_capability_is_not_entered_again():
    request = FIXTURES["mzi_window"]()
    state = EvaluationState()
    assert _phases(request, capabilities=("readings",), state=state).entered == ["trace", "emitters", "observation_results", "result"]
    extended = _phases(request, state=state)
    assert [p for p in extended.entered if p != "return_analysis"] == [
        "untraced_light", "bounds", "exposures", "interference", "targets", "result"]
    assert _phases(request, state=state).entered == ["targets", "result"]
