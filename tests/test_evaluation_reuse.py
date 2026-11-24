"""Reuse of an evaluation's products (``os_optics.evaluation.EvaluationState``).

A state keeps each executed capability's product for the next evaluation of the same physical inputs, numerical
configuration and Engine build. A follow-up evaluation runs only what the state does not hold; a result built over a
retained state equals a fresh evaluation of the same scope bit for bit; a changed target threshold reuses every physical
product; a changed pose, material, numerical setting or build drops them; an evaluation a monitor stops leaves the
products of the capabilities it completed, and the next evaluation continues from them.
"""

from __future__ import annotations

import pytest

from os_optics import ValidationRequest, evaluate, validate, validate_with_ray_paths
from os_optics.evaluation import EvaluationState, PhysicalIdentity
from os_optics.local import lower_project
from os_optics.manifest import engine_build_id
from os_optics.run_control import STOP_CALLER, RunStopped, monitored
from os_optics.validation.ray_paths import ray_path_recorder, recorded_ray_paths

from fixture_scenes import FIXTURES
from test_evaluation_scope import READINGS, _rated_request
from test_power_exposure_ratings import _document, _rating

PHYSICS = (*READINGS, "power_ledger", "untraced_light", "plane_bounds", "surface_exposures", "beam_pairs")
#: The sections of a result that are facts of the physics: a target's thresholds are no part of them.
PHYSICAL_SECTIONS = ("observations", "paths", "interference", "incoherent_pairs", "power_budget", "exposures", "grating_orders",
                     "nonlinear_conversions", "sampled_sources", "surface_profiles", "source_spectra")


def _moved_thresholds(request):
    """``request`` with every target's limits changed (the same metrics and selectors)."""
    targets = tuple(t.model_copy(update={"minimum": None, "maximum": 1.0e9, "target_value": None, "tolerance_abs": None,
                                         "tolerance_rel": None}) for t in request.project.targets)
    return request.model_copy(update={"project": request.project.model_copy(update={"targets": targets})})


def test_shared_prerequisites_execute_once():
    request = FIXTURES["interference_405_500"]()
    state = EvaluationState()
    result = evaluate(request, capabilities=("plane_bounds", "beam_pairs", "surface_exposures"), state=state)
    # (the trace and the readings serve the bounds, the pairs and the exposures; the ledger the bounds and the exposures)
    assert result.evaluation.executed == PHYSICS and state.executions == {c: 1 for c in PHYSICS}
    assert state.last_executed == PHYSICS and state.last_reused == () and state.retained == PHYSICS
    complete = evaluate(request, state=state)
    assert state.last_executed == ("requirements",) and state.last_reused == PHYSICS
    assert state.executions == {**{c: 1 for c in PHYSICS}, "requirements": 1}
    assert complete.manifest.result_hash == validate(request).manifest.result_hash and state.invalidations == {}


@pytest.mark.parametrize("name", ("interference_405_500", "mzi_window", "trace_extended_emitter", "trace_return_chain",
                                  "trace_probe_budget", "trace_partial_beams", "trace_grating_orders", "benchmark_standard",
                                  "control_recoverability", "layout_example"))
def test_an_extension_to_complete_equals_a_fresh_complete_evaluation(name):
    request = FIXTURES[name]()
    fresh = validate(request)
    state = EvaluationState()
    steps = (("beam_trace",), ("readings",), ("plane_readings", "beam_pairs"), ("power_ledger",), ("complete",))
    for capabilities in steps:
        result = evaluate(request, capabilities=capabilities, state=state)
    assert result.manifest.result_hash == fresh.manifest.result_hash and result.manifest.trace_hash == fresh.manifest.trace_hash
    assert result.model_dump(mode="json", exclude={"manifest"}) == fresh.model_dump(mode="json", exclude={"manifest"})
    assert result.evaluation is None and all(count == 1 for capability, count in state.executions.items() if capability != "requirements")
    assert state.invalidations == {}
    # (and each narrower evaluation over the state that now holds everything equals a fresh one of its scope)
    for capabilities in steps[:-1]:
        assert (evaluate(request, capabilities=capabilities, state=state).manifest.result_hash
                == evaluate(request, capabilities=capabilities).manifest.result_hash), capabilities
    assert all(count == 1 for capability, count in state.executions.items() if capability != "requirements")


def test_exposures_made_before_the_plane_bounds_are_made_again_in_the_complete_order():
    """The return-light analysis is read in the order of a complete run (the plane bounds, then the exposures): exposures a
    narrower evaluation made without the plane bounds are made again when the bounds join the scope."""
    request = _rated_request()
    fresh = validate(request)
    state = EvaluationState()
    first = evaluate(request, capabilities=("surface_exposures", "untraced_light"), state=state)
    assert first.exposures == fresh.exposures and state.executions["surface_exposures"] == 1
    complete = evaluate(request, state=state)
    assert complete.manifest.result_hash == fresh.manifest.result_hash
    assert state.executions["surface_exposures"] == 2 and state.executions["untraced_light"] == state.executions["beam_trace"] == 1
    assert state.invalidations == {"parameters": 1}


def _two_wavelength_ring(limit: int):
    """Three mirrors with a transparent plane beside them, the third component rated, and a second laser at 532 nm placed
    first: two wavelengths leave light out, so the return-light analysis' work limit is spent in the order they are read."""
    from os_optics import EngineConfiguration, NumericalConfig, ValidationRequest
    from os_optics.contracts import PowerRating, Target

    from test_return_analysis_budget import LAM, three_mirrors

    project = three_mirrors(plane=True)
    parts = list(project.components)
    parts[2] = parts[2].model_copy(update={"power_ratings": (PowerRating(
        regime="CW", value=100.0, unit="W/cm2", wavelength_nm=LAM, status="DECLARED", basis="test rating",
        polarization="LINEAR", polarization_condition="linear"),)})
    source = parts[0].source.model_copy(update={"source_id": "S2", "wavelength_nm": 532.0})
    second = parts[0].model_copy(update={"instance_id": "S2", "source": source})
    targets = (*project.targets, Target(target_id="util", metric="power_rating_utilization", device_id=parts[2].instance_id, maximum=1.0))
    project = project.model_copy(update={"components": (second, *parts), "targets": targets})
    return ValidationRequest(project=project, configuration=EngineConfiguration(
        run_layout=False, run_control=False, numerical=NumericalConfig(max_return_analysis_work=limit)))


def test_exposures_without_the_plane_bounds_spend_the_work_limit_as_the_complete_evaluation_does():
    """Exposures made without the plane bounds read the return-light analysis in the plane bounds' order: the work limit
    falls on the same wavelength, so the exposures and the analysis records equal the complete evaluation's also when the
    limit is reached."""
    from test_return_analysis_budget import _work_of, three_mirrors

    work = _work_of(three_mirrors(plane=True))

    def withheld(result) -> list:
        return [i.numerical_context["wavelength_nm"] for i in result.issues if i.code == "RETURN_ANALYSIS_BUDGET"]

    for limit, partial in ((work + 5, True), (10 * work, False)):
        request = _two_wavelength_ring(limit)
        complete = validate(request)
        records = complete.outcome.execution.budgets.return_analysis
        # (the return-light analysis is auxiliary: its exhausted budget is recorded and leaves the execution COMPLETED)
        assert [r.wavelength_nm for r in records] == [532.0, 1064.0] and complete.outcome.execution.outcome == "COMPLETED"
        assert ("max_return_analysis_work" in complete.outcome.execution.budgets.exhausted) is partial
        # (the plane bounds read 532 nm first: with the limit reached, 1064 nm is the wavelength left without an analysis)
        assert withheld(complete) == ([[1064.0]] if partial else [])
        for selection in (dict(capabilities=("surface_exposures",)), dict(capabilities=("requirements",), targets=("util",))):
            alone = evaluate(request, **selection)
            assert "plane_bounds" not in alone.evaluation.executed and "untraced_light" in alone.evaluation.executed
            assert alone.exposures == complete.exposures, (limit, selection)
            assert alone.outcome.execution.budgets.return_analysis == records and withheld(alone) == withheld(complete)
            assert alone.outcome.execution.outcome == complete.outcome.execution.outcome
        util = evaluate(request, capabilities=("requirements",), targets=("util",)).targets[0]
        assert util == next(t for t in complete.targets if t.target_id == "util")


class _StopOnEntry:
    """A run monitor that stops the run when it enters one phase for the ``at``-th time."""

    def __init__(self, phase_name: str, at: int = 1) -> None:
        self.phase_name, self.at, self.count = phase_name, at, 0

    def enter(self, name: str, total: int | None, unit: str) -> None:
        if name == self.phase_name:
            self.count += 1
            if self.count == self.at:
                raise RunStopped(STOP_CALLER, "stopped by the test", phase=name)

    def leave(self, name: str) -> None:
        pass

    def unwind(self, name: str) -> None:
        pass

    def step(self, count: int) -> None:
        pass


def test_exposures_stopped_in_the_return_analysis_continue_as_a_fresh_evaluation():
    """An exposures-only evaluation stopped while it reads the return-light analysis in the plane bounds' order, then
    continued over its state: the analysis is read in the same order, so the result equals a fresh one (exposures,
    analysis records, result hash), whether or not the work limit is reached."""
    from test_return_analysis_budget import _work_of, three_mirrors

    work = _work_of(three_mirrors(plane=True))
    for limit in (work + 5, 10 * work):
        request = _two_wavelength_ring(limit)
        fresh = evaluate(request, capabilities=("surface_exposures",))
        for at in (1, 2):
            state = EvaluationState()
            with pytest.raises(RunStopped) as stopped, monitored(_StopOnEntry("return_analysis", at=at)):
                evaluate(request, capabilities=("surface_exposures",), state=state)
            assert stopped.value.phase == "return_analysis" and "untraced_light" in state.retained
            assert "surface_exposures" not in state.retained
            for _ in range(2):  # (and again over the same state: the exposures are retained, the analysis read in order)
                continued = evaluate(request, capabilities=("surface_exposures",), state=state)
                assert continued.exposures == fresh.exposures, (limit, at)
                assert continued.outcome.execution.budgets.return_analysis == fresh.outcome.execution.budgets.return_analysis
                assert continued.manifest.result_hash == fresh.manifest.result_hash


def test_the_exposures_are_dropped_with_the_untraced_light_they_read():
    from os_optics.evaluation.state import DEPENDENTS

    assert "surface_exposures" in DEPENDENTS["untraced_light"] and "plane_bounds" in DEPENDENTS["untraced_light"]
    # (a design without ratings: its exposures cover the traced beams and stay when the untraced light first runs)
    request = FIXTURES["mzi_window"]()
    state = EvaluationState()
    first = evaluate(request, capabilities=("surface_exposures",), state=state)
    evaluate(request, capabilities=("plane_bounds",), state=state)
    again = evaluate(request, capabilities=("surface_exposures",), state=state)
    assert state.executions["surface_exposures"] == 1 and state.invalidations == {} and again.exposures == first.exposures
    # (a design with ratings: its exposures read the untraced light, and go with it)
    rated = _rated_request()
    state = EvaluationState()
    whole = evaluate(rated, state=state)
    state.discard("untraced_light")
    assert state.retained == (*READINGS, "power_ledger", "beam_pairs") and state.invalidations == {"parameters": 1}
    assert evaluate(rated, state=state).manifest.result_hash == whole.manifest.result_hash
    assert state.executions["surface_exposures"] == state.executions["untraced_light"] == 2 and state.executions["beam_trace"] == 1


def test_the_return_analysis_records_are_not_computed_where_no_capability_read_the_analysis():
    """The untraced light makes the return-light analysis when the plane bounds or the surface exposures read it: an
    evaluation that runs it without them has no records, and names them as not computed (with the plane bounds, its first
    reader); one with a reader has the complete evaluation's records and names nothing."""
    from os_optics.evaluation import capability

    from test_return_analysis_budget import _work_of, three_mirrors

    request = _two_wavelength_ring(_work_of(three_mirrors(plane=True)) + 5)
    records = validate(request).outcome.execution.budgets.return_analysis
    assert records
    for selection in (("untraced_light",), ("readings", "untraced_light")):
        alone = evaluate(request, capabilities=selection)
        assert "untraced_light" in alone.evaluation.executed and not alone.outcome.execution.budgets.return_analysis
        produces = {n.capability: n.produces for n in alone.evaluation.not_computed}
        assert produces["plane_bounds"] == (*capability("plane_bounds").produces, "outcome.execution.budgets.return_analysis")
    for selection in (("surface_exposures",), ("plane_bounds",)):
        read = evaluate(request, capabilities=selection)
        assert read.outcome.execution.budgets.return_analysis == records
        assert all(n.produces == capability(n.capability).produces for n in read.evaluation.not_computed)
    assert capability("untraced_light").phases == ("untraced_light",)
    assert "return_analysis" in capability("plane_bounds").phases and "return_analysis" in capability("surface_exposures").phases


def test_exposures_of_a_design_without_ratings_make_and_record_no_return_analysis():
    """The exposures of a design that states no power rating cover its traced beams and read no return-light analysis:
    with the untraced light and without the plane bounds none is made, so the result has no record of one, no budget that
    ran out, and names the records as not computed with the plane bounds -- where the complete evaluation's analysis
    stops at its work limit, and where it finishes."""
    from os_optics.evaluation import capability
    from os_optics.local import evaluate_project

    fixture = FIXTURES["trace_return_chain"]()
    assert not any(c.power_ratings for c in fixture.project.components) and fixture.project.observations
    selection = ("surface_exposures", "untraced_light")
    named = (*capability("plane_bounds").produces, "outcome.execution.budgets.return_analysis")
    for limit in (500, fixture.configuration.numerical.max_return_analysis_work):
        numerical = fixture.configuration.numerical.model_copy(update={"max_return_analysis_work": limit})
        request = fixture.model_copy(update={"configuration": fixture.configuration.model_copy(update={"numerical": numerical})})
        complete = validate(request)
        budgets = complete.outcome.execution.budgets
        assert len(budgets.return_analysis) == 1
        assert (budgets.exhausted, complete.outcome.execution.outcome) == (
            (("max_return_analysis_work",), "COMPLETED") if limit == 500 else ((), "COMPLETED"))
        alone = evaluate(request, capabilities=selection)
        assert alone.evaluation.executed[-2:] == ("untraced_light", "surface_exposures")
        execution = alone.outcome.execution
        assert (execution.budgets.return_analysis, execution.budgets.exhausted, execution.outcome) == ((), (), "COMPLETED"), limit
        assert not [i for i in alone.issues if i.code == "RETURN_ANALYSIS_BUDGET"]
        assert alone.exposures == complete.exposures
        assert {n.capability: n.produces for n in alone.evaluation.not_computed}["plane_bounds"] == named
    # (a Project evaluation lists the Engine capabilities it did not compute as its result does: without a rating the
    # exposures are no reader of the analysis, with one they are)
    for ratings, records_named in ((None, True), ([_rating("CW", 1.0e6, "W/cm2")], False)):
        evaluation = evaluate_project(_document(ratings), outputs=selection)
        listed = {row["capability"]: tuple(row["produces"]) for row in evaluation.not_computed()}
        assert all(listed[n.capability] == n.produces for n in evaluation.result.evaluation.not_computed)
        assert (listed["plane_bounds"] == named) is records_named and bool(evaluation.result.exposures.untraced_light_included) is not records_named


def test_a_threshold_change_reuses_every_physical_product():
    for request in (FIXTURES["interference_405_500"](), FIXTURES["benchmark_standard"](), _rated_request()):
        state = EvaluationState()
        first = evaluate(request, state=state)
        ran = dict(state.executions)
        moved = _moved_thresholds(request)
        second = evaluate(moved, state=state)
        assert state.last_executed == ("requirements",) and state.invalidations == {}
        assert state.executions == {**ran, "requirements": 2}
        for section in PHYSICAL_SECTIONS:
            assert getattr(second, section) == getattr(first, section), section
        assert second.manifest.design_projection_hash == first.manifest.design_projection_hash
        assert second.manifest.trace_hash == first.manifest.trace_hash and second.manifest.input_hash != first.manifest.input_hash
        fresh = validate(moved)
        assert second.manifest.result_hash == fresh.manifest.result_hash and second.targets == fresh.targets
        assert [t.maximum for t in second.targets] == [1.0e9] * len(second.targets)


def test_a_changed_target_selection_runs_only_the_capabilities_that_read_it():
    """Which pairs a pair target names and which elements a utilization target assesses are read by the beam pairs and the
    exposures; nothing else of the physics reads a target."""
    util = {"target_id": "util", "metric": "power_rating_utilization", "device_instance_id": "W", "maximum": 1.0}
    exceeded = [_rating("CW", 1.0, "W/cm2")]  # (2 W focused onto a window rated for 1 W/cm2: the rating's notice names the target)
    assessed = lower_project(_document(exceeded, targets=[util])).request
    unassessed = lower_project(_document(exceeded)).request
    state = EvaluationState()
    first = evaluate(assessed, state=state)
    second = evaluate(unassessed, state=state)
    assert state.last_executed == ("surface_exposures", "requirements") and state.invalidations == {"parameters": 1}
    assert second.manifest.result_hash == validate(unassessed).manifest.result_hash != first.manifest.result_hash
    notice = {r.manifest.input_hash: next(i for i in r.issues if i.code == "POWER_RATING_EXCEEDED") for r in (first, second)}
    assert notice[first.manifest.input_hash].numerical_context["utilization_target"] is True
    assert notice[second.manifest.input_hash].numerical_context["utilization_target"] is False
    request = FIXTURES["interference_405_500"]()
    state = EvaluationState()
    evaluate(request, state=state)
    untargeted = request.model_copy(update={"project": request.project.model_copy(update={"targets": ()})})
    result = evaluate(untargeted, state=state)
    assert state.last_executed == ("beam_pairs", "requirements")
    assert result.manifest.result_hash == validate(untargeted).manifest.result_hash


def _with_component(request, index: int, **pose_update):
    component = request.project.components[index]
    moved = component.model_copy(update={"pose": component.pose.model_copy(update=pose_update)})
    components = (*request.project.components[:index], moved, *request.project.components[index + 1:])
    return request.model_copy(update={"project": request.project.model_copy(update={"components": components})})


def test_a_changed_input_material_numerical_setting_or_build_drops_the_products():
    request = _rated_request()
    state = EvaluationState(build_id="build-a")
    first = evaluate(request, state=state)
    assert first.manifest.engine_build_id == "build-a" and state.identity.engine_build_id == "build-a"
    assert state.identity == PhysicalIdentity("build-a", first.manifest.design_projection_hash, state.identity.numerical_hash,
                                              request.project.project_schema_version)
    ran = dict(state.executions)
    assert all(count == 1 for count in ran.values())

    def twice(capability_counts: dict) -> dict:
        return {capability: 2 * count for capability, count in capability_counts.items()}

    # ---- a moved component
    x, y, z = request.project.components[-1].pose.position_mm
    moved = _with_component(request, len(request.project.components) - 1, position_mm=(x + 1.0, y, z))
    result = evaluate(moved, state=state)
    assert state.invalidations == {"inputs": 1} and state.executions == twice(ran)
    assert result.manifest.result_hash == validate(moved).manifest.result_hash != first.manifest.result_hash
    # ---- a changed material
    glass = moved.project.materials[0]
    other_glass = glass.model_copy(update={"absorption_coefficient_per_mm": 1.0e-3})
    remade = moved.model_copy(update={"project": moved.project.model_copy(update={"materials": (other_glass, *moved.project.materials[1:])})})
    result = evaluate(remade, state=state)
    assert state.invalidations == {"inputs": 2} and state.last_reused == ()
    assert result.manifest.result_hash == validate(remade).manifest.result_hash
    # ---- a changed numerical setting
    numerical = remade.configuration.numerical.model_copy(update={"max_paths": remade.configuration.numerical.max_paths - 1})
    stricter = remade.model_copy(update={"configuration": remade.configuration.model_copy(update={"numerical": numerical})})
    result = evaluate(stricter, state=state)
    assert state.invalidations == {"inputs": 2, "numerical_configuration": 1} and state.last_reused == ()
    assert result.manifest.result_hash == validate(stricter).manifest.result_hash
    # ---- another Engine build (the caller states the build the products belong to)
    count = dict(state.executions)
    evaluate(stricter, state=state)
    assert state.last_executed == ("requirements",) and state.invalidations == {"inputs": 2, "numerical_configuration": 1}
    state.build_id = "build-b"
    result = evaluate(stricter, state=state)
    assert state.invalidations == {"inputs": 2, "numerical_configuration": 1, "engine_build": 1} and state.last_reused == ()
    assert result.manifest.engine_build_id == "build-b" and state.executions["beam_trace"] == count["beam_trace"] + 1
    # ---- and back to the first request: nothing of it is retained
    evaluate(request, state=state)
    assert state.invalidations == {"inputs": 3, "numerical_configuration": 2, "engine_build": 1} and state.last_reused == ()



@pytest.mark.parametrize("setting", ["max_trace_steps", "aperture_quad_sector_nodes", "aperture_edge_tol_mm"])
def test_each_numerical_setting_of_the_trace_evidence_drops_the_retained_trace(setting):
    """A changed trace-step budget, aperture quadrature or aperture edge tolerance makes the retained trace evidence of
    another configuration: the next evaluation traces again, reuses nothing, and equals a fresh evaluation."""
    from os_optics import NumericalConfig

    request = FIXTURES["keplerian_expander"]()
    state = EvaluationState()
    evaluate(request, capabilities=("readings",), state=state)
    settings = request.configuration.numerical.model_dump()
    settings[setting] = settings[setting] * 2
    numerical = NumericalConfig.model_validate(settings)
    changed = request.model_copy(update={"configuration": request.configuration.model_copy(update={"numerical": numerical})})
    second = evaluate(changed, capabilities=("readings",), state=state)
    assert state.invalidations == {"numerical_configuration": 1} and state.last_reused == () and "beam_trace" in state.last_executed
    assert second.model_dump(mode="json") == evaluate(changed, capabilities=("readings",)).model_dump(mode="json")

def test_the_build_id_is_computed_once_per_state_and_a_stateless_evaluation_computes_its_own():
    request = FIXTURES["keplerian_expander"]()
    state = EvaluationState()
    assert state.build_id is None
    first = evaluate(request, capabilities=("readings",), state=state)
    assert state.build_id == engine_build_id() == first.manifest.engine_build_id == validate(request).manifest.engine_build_id
    assert evaluate(request, state=state).manifest.engine_build_id == state.build_id


def test_a_request_of_an_unsupported_schema_version_runs_and_reads_nothing_of_the_state():
    request = FIXTURES["keplerian_expander"]()
    state = EvaluationState()
    evaluate(request, capabilities=("beam_trace",), state=state)
    evaluate(request, capabilities=("beam_trace",), state=state)
    assert state.last_reused == ("scene", "beam_trace")
    retained, executions = state.retained, dict(state.executions)
    other = ValidationRequest.model_validate({**request.model_dump(mode="json"),
                                              "project": {**request.project.model_dump(mode="json"), "project_schema_version": "9.0.0"}})
    refused = evaluate(other, capabilities=("beam_trace",), state=state)
    assert [i.code for i in refused.issues] == ["UNSUPPORTED_SCHEMA_VERSION"]
    assert state.last_executed == () and state.last_reused == ()
    assert state.retained == retained and state.executions == executions


class _StopIn:
    """A run monitor that stops the run at the ``at``-th checkpoint of one phase."""

    def __init__(self, phase_name: str, at: int = 2) -> None:
        self.phase_name, self.at, self.stack, self.entered, self.steps = phase_name, at, [], [], 0

    def enter(self, name: str, total: int | None, unit: str) -> None:
        self.entered.append(name)
        self.stack.append(name)

    def leave(self, name: str) -> None:
        self.stack.pop()

    def unwind(self, name: str) -> None:
        self.stack.pop()

    def step(self, count: int) -> None:
        if self.stack and self.stack[-1] == self.phase_name:
            self.steps += 1
            if self.steps == self.at:
                raise RunStopped(STOP_CALLER, "stopped by the test", phase=self.phase_name)


def test_an_evaluation_stopped_in_the_untraced_light_continues_without_tracing_again():
    request = FIXTURES["mzi_window"]()  # (a window's ghosts: the probe pass follows them)
    state = EvaluationState()
    monitor = _StopIn("untraced_light")
    with pytest.raises(RunStopped) as stopped, monitored(monitor):
        evaluate(request, state=state)
    assert stopped.value.phase == "untraced_light" and monitor.entered[-1] == "untraced_light"
    done = (*READINGS, "power_ledger")
    assert state.retained == done and state.executions == {c: 1 for c in done} and state.last_executed == done
    result = evaluate(request, state=state)
    assert state.last_reused == done and state.last_executed == ("untraced_light", "plane_bounds", "surface_exposures", "beam_pairs",
                                                                 "requirements")
    assert all(count == 1 for count in state.executions.values())
    fresh = validate(request)
    assert result.manifest.result_hash == fresh.manifest.result_hash and result.surface_profiles == fresh.surface_profiles


def test_a_stop_before_a_capability_finished_keeps_nothing_of_it():
    request = FIXTURES["mzi_window"]()
    state = EvaluationState()
    with pytest.raises(RunStopped), monitored(_StopIn("trace", at=1)):
        evaluate(request, capabilities=("readings",), state=state)
    assert state.retained == ("scene",) and "beam_trace" not in state.executions
    assert evaluate(request, state=state).manifest.result_hash == validate(request).manifest.result_hash
    assert state.executions["scene"] == state.executions["beam_trace"] == 1


def test_the_solver_statistics_of_a_result_are_those_of_the_capabilities_it_ran():
    request = FIXTURES["trace_freeform_surfaces"]()
    state = EvaluationState()
    readings = evaluate(request, capabilities=("readings",), state=state)
    # (a probe pass a monitor stops part way leaves none of its rays in the retained scene's statistics)
    with pytest.raises(RunStopped), monitored(_StopIn("untraced_light", at=3)):
        evaluate(request, state=state)
    assert state.retained == (*READINGS, "power_ledger")
    complete = evaluate(request, state=state)
    fresh = validate(request)
    assert complete.surface_profiles == fresh.surface_profiles
    traced = {p.surface_id: p.solved_rays for p in readings.surface_profiles}
    assert traced and all(0 < traced[p.surface_id] < p.solved_rays for p in fresh.surface_profiles if p.solved_rays)
    # (a narrower evaluation over the complete state reports its own capabilities' rays, as a fresh one does)
    assert evaluate(request, capabilities=("readings",), state=state).surface_profiles == readings.surface_profiles


def test_a_trace_a_recorder_did_not_observe_is_traced_again_for_it():
    request = FIXTURES["mzi_window"]()
    result, paths = validate_with_ray_paths(request)
    assert result.manifest.result_hash == validate(request).manifest.result_hash and paths.status == "TRACED"
    state = EvaluationState()
    evaluate(request, capabilities=("readings",), state=state)
    recorder = ray_path_recorder()
    observed = evaluate(request, capabilities=("readings",), state=state, recorder=recorder)
    # (the lowered scene is kept: the trace and what was built on it are made again, observed)
    assert state.invalidations == {"recorder": 1} and state.executions["beam_trace"] == 2 and state.last_reused == ("scene",)
    assert state.executions["scene"] == 1 and state.last_executed == ("beam_trace", "secondary_rays", "plane_readings")
    # (the same recorder again, and an evaluation that passes none: the observed trace is reused)
    complete = evaluate(request, state=state, recorder=recorder)
    again = evaluate(request, state=state)
    assert state.executions["beam_trace"] == 2 and state.invalidations == {"recorder": 1}
    assert complete.manifest.result_hash == again.manifest.result_hash == result.manifest.result_hash
    assert recorded_ray_paths(request, complete, recorder)[1] == paths
    assert recorded_ray_paths(request, observed, recorder)[1].paths == paths.paths


def _ids(trace) -> list[str]:
    return [path.path_id for path in trace.paths]


def test_a_recorder_holds_the_record_of_the_latest_trace_it_observed():
    """The ray-path record belongs to the trace it observed: a trace made again for a recorder, or continued after a stop,
    replaces what the recorder held -- no path twice, none of another trace."""
    request = FIXTURES["trace_extended_emitter"]()  # (an extended emitter: its rays are secondary rays)
    result, paths = validate_with_ray_paths(request)
    assert paths.paths and paths.sampled is not None and len(set(_ids(paths))) == len(paths.paths)
    # recorder A, then B, then A again over one state: each time the trace is made again for the recorder that asks
    state = EvaluationState()
    first, second = ray_path_recorder(), ray_path_recorder()
    for recorder in (first, second, first):
        observed = evaluate(request, state=state, recorder=recorder)
        assert recorded_ray_paths(request, observed, recorder)[1] == paths
    assert state.invalidations == {"recorder": 2} and observed.manifest.result_hash == result.manifest.result_hash
    # a stop inside the secondary rays, then the evaluation continued with the same recorder, and with none
    for again in (True, False):
        state, recorder = EvaluationState(), ray_path_recorder()
        with pytest.raises(RunStopped), monitored(_StopIn("emitters")):
            evaluate(request, capabilities=("readings",), state=state, recorder=recorder)
        assert state.retained == ("scene", "beam_trace")
        continued = evaluate(request, state=state, recorder=recorder if again else None)
        assert recorded_ray_paths(request, continued, recorder)[1] == paths and state.executions["beam_trace"] == 1
    # a stop inside the secondary rays, then an evaluation that is not given the recorder (another recorder over the same
    # state, or another state): the recorder holds part of the stopped secondary rays, and never answers TRACED with them
    # under that result's trace hash
    for at in (1, 2):
        for other_state in (False, True):
            state, recorder = EvaluationState(), ray_path_recorder()
            with pytest.raises(RunStopped), monitored(_StopIn("emitters", at=at)):
                evaluate(request, state=state, recorder=recorder)
            other = (evaluate(request, state=EvaluationState()) if other_state
                     else evaluate(request, state=state, recorder=ray_path_recorder()))
            assert other.manifest.trace_hash is not None
            _result, partial = recorded_ray_paths(request, other, recorder)
            assert (partial.status, partial.trace_hash, partial.paths) == ("NOT_TRACED", None, ())
            assert partial.message.startswith("the recorder holds part of the secondary rays a monitor stopped: give the recorder to")
            if other_state:
                # (the beam trace's record is whole: a result of the beam trace alone, over the stopped state, reads it)
                beam_only = evaluate(request, capabilities=("beam_trace",), state=state, recorder=recorder)
                assert recorded_ray_paths(request, beam_only, recorder)[1].status == "TRACED" and state.executions["beam_trace"] == 1
            # (given the recorder, the evaluation that continues traces the secondary rays for it and the record is whole)
            assert recorded_ray_paths(request, evaluate(request, state=state, recorder=recorder), recorder)[1] == paths
    # a stop inside the beam trace: the trace starts again, and the recorder with it
    beams = FIXTURES["mzi_window"]()
    whole = validate_with_ray_paths(beams)[1]
    state, recorder = EvaluationState(), ray_path_recorder()
    with pytest.raises(RunStopped), monitored(_StopIn("trace")):
        evaluate(beams, capabilities=("readings",), state=state, recorder=recorder)
    continued = evaluate(beams, state=state, recorder=recorder)
    assert recorded_ray_paths(beams, continued, recorder)[1] == whole and len(set(_ids(whole))) == len(whole.paths)
    # a stop inside the beam trace, then the evaluation continued without the recorder: the recorder holds part of the
    # stopped trace, and never answers TRACED with it under the continued result's trace hash
    for at in (1, 2):
        state, recorder = EvaluationState(), ray_path_recorder()
        with pytest.raises(RunStopped), monitored(_StopIn("trace", at=at)):
            evaluate(beams, capabilities=("readings",), state=state, recorder=recorder)
        continued = evaluate(beams, state=state)
        assert continued.manifest.trace_hash is not None
        _result, partial = recorded_ray_paths(beams, continued, recorder)
        assert (partial.status, partial.trace_hash, partial.paths) == ("NOT_TRACED", None, ())
        assert partial.message.startswith("the recorder holds part of a beam trace a monitor stopped: give the recorder to")
        # (given the recorder, the evaluation that continues traces again for it and the record is whole)
        assert recorded_ray_paths(beams, evaluate(beams, state=state, recorder=recorder), recorder)[1] == whole


def test_the_ray_paths_of_a_result_are_those_of_the_capabilities_it_ran():
    """A beam-trace result read after an evaluation that also traced the secondary rays carries the beam trace's rays alone,
    as a fresh evaluation of the same scope does: the paths its ``trace_hash`` covers."""
    request = FIXTURES["trace_extended_emitter"]()
    fresh_recorder = ray_path_recorder()
    fresh = recorded_ray_paths(request, evaluate(request, capabilities=("beam_trace",), recorder=fresh_recorder), fresh_recorder)[1]
    assert fresh.sampled is None
    state, recorder = EvaluationState(), ray_path_recorder()
    readings = evaluate(request, capabilities=("readings",), state=state, recorder=recorder)
    with_secondary = recorded_ray_paths(request, readings, recorder)[1]
    assert with_secondary == validate_with_ray_paths(request)[1] and len(with_secondary.paths) > len(fresh.paths)
    beam_only = evaluate(request, capabilities=("beam_trace",), state=state, recorder=recorder)
    assert state.last_executed == () and state.executions["beam_trace"] == 1
    assert recorded_ray_paths(request, beam_only, recorder)[1] == fresh
    # (the readings' record is still the recorder's: reading it again gives the same paths)
    assert recorded_ray_paths(request, readings, recorder)[1] == with_secondary


def test_a_recorder_passed_to_an_evaluation_that_does_not_trace_drops_nothing():
    request = FIXTURES["mzi_window"]()
    state = EvaluationState()
    complete = evaluate(request, state=state)
    retained = state.retained
    requirements = evaluate(request, capabilities=("requirements",), state=state, recorder=ray_path_recorder())
    assert state.last_executed == ("requirements",) and state.invalidations == {} and state.retained == retained
    assert requirements.targets == complete.targets
    assert evaluate(request, state=state).manifest.result_hash == complete.manifest.result_hash
    assert state.executions["beam_trace"] == 1 and state.invalidations == {}
    # (an evaluation that traces with a recorder the retained trace did not have is traced again, as before)
    evaluate(request, capabilities=("readings",), state=state, recorder=ray_path_recorder())
    assert state.invalidations == {"recorder": 1} and state.executions["beam_trace"] == 2
