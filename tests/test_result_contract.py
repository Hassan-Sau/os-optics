"""The scientific-result contract of a run (docs/engine/RESULTS_AND_EXPORT.md): execution outcome,
resource budgets, the requirement assessment of declared targets, and the state of a plane's power. None of these
changes a number; each answers a question the aggregate status cannot."""

from __future__ import annotations

import json

import pytest

from os_optics import builders as B
from os_optics.config import NumericalConfig
from os_optics.contracts import ProjectSnapshot, Target, ValidationResult
from os_optics.errors.status import ExecutionOutcome, QuantityState, RequirementAssessment, TerminationReason

from helpers import arrivals, issue_codes, run

CFG = NumericalConfig()


def _focus(maximum: float | None = 0.05) -> ProjectSnapshot:
    """A 1 mW 780 nm beam through an ideal thin lens onto a plane 100 mm on."""

    src = B.laser("S", wavelength_nm=780.0, power_W=1e-3, waist_radius_mm=1.0, position=(-60.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0))
    lens = B.thin_lens("L1", focal_length_mm=100.0, clear_radius_mm=12.7, position=(0.0, 0.0, 0.0))
    plane = B.observation_plane("focal", position=(100.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0), radius_mm=12.7)
    targets = () if maximum is None else (Target(target_id="spot", observation_id="focal", metric="beam_radius_mm",
                                                  maximum=maximum, unit="mm"),)
    return ProjectSnapshot(components=(src, lens), observations=(plane,), targets=targets)


def test_a_run_without_targets_reports_results_and_no_requirement_assessment():
    """No targets means no requirement assessment -- which is not universal success, and not a failure either."""

    result = run(_focus(maximum=None))
    assert result.outcome is not None
    assert result.outcome.execution.outcome is ExecutionOutcome.COMPLETED
    assert result.outcome.execution.termination is TerminationReason.COMPLETED
    assert result.outcome.execution.complete is True
    assert result.outcome.requirements.declared is False
    assert result.outcome.requirements.met == () and result.outcome.requirements.unmet == ()
    (obs,) = result.observations
    assert obs.arrivals and obs.quantity_state is QuantityState.KNOWN
    assert arrivals(result, "focal")[0].beam.power_W > 0.0


def test_a_valid_run_that_misses_its_target_is_completed_and_unmet():
    """A design that misses a customer target is a completed calculation with an unmet requirement, not a failed run."""

    result = run(_focus(maximum=1e-6))
    (target,) = result.targets
    assert target.status.value == "FAIL"
    assert target.assessment is RequirementAssessment.UNMET
    assert result.outcome.execution.outcome is ExecutionOutcome.COMPLETED
    assert result.outcome.requirements.declared is True
    assert result.outcome.requirements.unmet == (target.target_id,)
    (obs,) = result.observations
    assert obs.arrivals and obs.arrivals[0].beam.power_W > 0.0


def _physical(result: ValidationResult) -> dict:
    """Everything about a result that a target threshold must not change."""

    body = result.model_dump(mode="json")
    # (the scientific quantities themselves; status and issues legitimately follow the verdict, targets are the verdict)
    return {key: body[key] for key in ("observations", "power_budget", "interference", "paths", "exposures") if key in body}


@pytest.mark.parametrize("threshold", [0.01, 0.05, 1.0])
def test_changing_only_a_target_threshold_changes_no_physical_value(threshold: float):
    """The probes follow the metric a Project targets, never the bounds it sets on it."""

    reference = run(_focus(maximum=0.05))
    changed = run(_focus(maximum=threshold))
    assert _physical(changed) == _physical(reference)
    assert changed.outcome.execution == reference.outcome.execution
    assert [o.quantity_state for o in changed.observations] == [o.quantity_state for o in reference.observations]
    assert [t.value for t in changed.targets] == [t.value for t in reference.targets]
    # (the verdict is the only thing the threshold decides, and it is the correct one)
    (target,) = changed.targets
    expected = RequirementAssessment.MET if target.value <= threshold else RequirementAssessment.UNMET
    assert target.assessment is expected
    assert (target.issue_codes == ("TARGET_NOT_MET",)) is (expected is RequirementAssessment.UNMET)


def test_a_known_dark_plane_is_known_zero():
    """A plane no light reaches, with every launched line traced to a known end, is 0 W by evidence."""

    src = B.laser("S", wavelength_nm=780.0, power_W=1e-3, waist_radius_mm=0.5, position=(-60.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0))
    dump = B.beam_dump("D", position=(0.0, 0.0, 0.0), axis=(1.0, 0.0, 0.0), radius_mm=12.7)
    behind = B.observation_plane("behind", position=(150.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0), radius_mm=12.7)
    result = run(ProjectSnapshot(components=(src, dump), observations=(behind,), targets=()))
    (obs,) = result.observations
    assert obs.total_power_W == pytest.approx(0.0)
    assert obs.quantity_state is QuantityState.KNOWN_ZERO


def test_budget_exhaustion_is_partial_and_names_the_budget():
    """A run that used up a budget reports PARTIAL, the budget that ran out and the settings it used."""

    project = _focus(maximum=None)
    # Two physical surfaces need two interactions; free propagation to a monitor is not a third.
    project = project.model_copy(update={"components": (*project.components, B.thin_lens(
        "L2", focal_length_mm=1000.0, clear_radius_mm=12.7, position=(50.0, 0.0, 0.0)))})
    tight = run(project, numerical=CFG.model_copy(update={"max_interactions_per_path": 1}))
    assert tight.outcome.execution.outcome is ExecutionOutcome.PARTIAL
    assert tight.outcome.execution.termination is TerminationReason.BUDGET_EXHAUSTED
    assert tight.outcome.execution.budgets.exhausted
    assert tight.outcome.execution.budgets.limits["max_interactions_per_path"] == 1
    assert tight.observations is not None
    loose = run(project)
    assert loose.outcome.execution.outcome is ExecutionOutcome.COMPLETED
    assert loose.outcome.execution.budgets.exhausted == ()
    assert loose.outcome.execution.budgets.limits["max_interactions_per_path"] == CFG.max_interactions_per_path


def test_an_interaction_limit_names_the_path_and_the_source_it_cut():
    """The per-path interaction limit of the primary trace names the path it ended and that path's source, as the trace
    budget names its own (the source list was once empty)."""

    project = _focus(maximum=None)
    project = project.model_copy(update={"components": (*project.components, B.thin_lens(
        "L2", focal_length_mm=1000.0, clear_radius_mm=12.7, position=(50.0, 0.0, 0.0)))})
    budgets = run(project, numerical=CFG.model_copy(update={"max_interactions_per_path": 1})).outcome.execution.budgets
    assert budgets.exhausted == ("max_interactions_per_path",)
    assert budgets.exhausted_by_phase == {"primary": ("max_interactions_per_path",)}
    assert budgets.affected_source_ids == ("S",)
    assert budgets.affected_path_ids == ("S#0",)
    assert budgets.untraced_power_W == pytest.approx(1e-3, rel=1e-9)


def test_a_stored_result_without_the_new_fields_still_reads():
    """Old run directories carry no outcome, assessment or quantity_state; they must keep their meaning."""

    body = json.loads(run(_focus()).model_dump_json())
    body.pop("outcome", None)
    for observation in body.get("observations", []):
        observation.pop("quantity_state", None)
    for target in body.get("targets", []):
        target.pop("assessment", None)
    restored = ValidationResult.model_validate(body)
    assert restored.outcome is None
    assert all(o.quantity_state is None for o in restored.observations)
    assert [t.assessment for t in restored.targets] == [RequirementAssessment.MET]


def test_the_assessment_is_derived_from_the_status_of_a_stored_target():
    body = json.loads(run(_focus(maximum=1e-6)).model_dump_json())
    for target in body["targets"]:
        target.pop("assessment", None)
    restored = ValidationResult.model_validate(body)
    assert [t.assessment for t in restored.targets] == [RequirementAssessment.UNMET]




def test_a_project_budget_outside_the_published_range_is_refused():
    """The published range of each budget field is part of the contract, not a comment."""

    from pydantic import ValidationError

    from os_optics._vendor.shared_contracts.optical_design.compute_budget import ComputeBudgetV1

    for field, bad in (("max_paths", 0), ("max_paths", -5), ("max_interactions_per_path", 0),
                       ("max_trace_steps", 2 ** 53), ("max_interference_pairs", 0)):
        with pytest.raises(ValidationError):
            ComputeBudgetV1.model_validate({field: bad})
    assert ComputeBudgetV1.model_validate({"max_paths": 4096}).overrides() == {"max_paths": 4096}


def test_a_bounded_plane_is_only_bounded_where_the_bound_is_supported():
    """A transparent plane may be crossed by the untraced light more than once: no upper bound, so UNKNOWN."""

    from os_optics.contracts.results import ObservationPlaneFrame, ObservationResult
    from os_optics.validation.outcome import quantity_state_of

    def plane(absorbing: bool) -> ObservationPlaneFrame:
        frame = ObservationPlaneFrame(origin_mm=(0.0, 0.0, 0.0), u_axis=(0.0, 1.0, 0.0), v_axis=(0.0, 0.0, 1.0),
                                      normal=(1.0, 0.0, 0.0), pose_frame="DESIGN", absorbing=absorbing,
                                      absorbing_extent="ACTIVE_AREA" if absorbing else "NONE")
        return frame

    bounded = ObservationResult(observation_id="a", status="UNKNOWN", plane=plane(True), total_power_W=1e-3,
                                power_complete=False, untraced_power_W=1e-4)
    assert quantity_state_of(bounded) is QuantityState.BOUNDED
    transparent = ObservationResult(observation_id="b", status="UNKNOWN", plane=plane(False), total_power_W=1e-3,
                                   power_complete=False, untraced_power_W=1e-4)
    assert quantity_state_of(transparent) is QuantityState.UNKNOWN
    dark = ObservationResult(observation_id="c", status="PASS", plane=plane(True), total_power_W=0.0)
    assert quantity_state_of(dark) is QuantityState.KNOWN_ZERO


def test_the_cli_reports_requirements_by_default_and_execution_or_the_aggregate_on_request():
    """``--status-contract`` (``cli_exit``): requirements is the default; execution and the aggregate stay selectable, and
    ``EXIT_CODES`` (the aggregate) is unchanged."""

    from types import SimpleNamespace

    from os_optics.cli_exit import DEFAULT_STATUS_CONTRACT, EXIT_CODES, STATUS_CONTRACTS, run_exit_code

    assert STATUS_CONTRACTS == ("aggregate", "execution", "requirements") and DEFAULT_STATUS_CONTRACT == "requirements"
    assert EXIT_CODES == {"PASS": 0, "WARNING": 0, "NOT_APPLICABLE": 0, "FAIL": 1, "UNKNOWN": 2, "OUT_OF_MODEL_SCOPE": 3}

    def outcome(execution: str, declared: bool, met=(), unmet=(), indeterminate=(), not_applicable=(), termination=None):
        return SimpleNamespace(execution=SimpleNamespace(outcome=SimpleNamespace(value=execution),
                                                         termination=SimpleNamespace(value=termination or (
                                                             "COMPLETED" if execution == "COMPLETED" else "BUDGET_EXHAUSTED"))),
                               requirements=SimpleNamespace(declared=declared, met=tuple(met), unmet=tuple(unmet),
                                                            indeterminate=tuple(indeterminate),
                                                            not_applicable=tuple(not_applicable)))

    def local(status: str, result_outcome) -> SimpleNamespace:
        return SimpleNamespace(status=SimpleNamespace(value=status), result=SimpleNamespace(outcome=result_outcome))

    ok = local("PASS", outcome("COMPLETED", True, met=("a",)))
    missed = local("FAIL", outcome("COMPLETED", True, unmet=("a",)))
    undecided = local("UNKNOWN", outcome("COMPLETED", True, indeterminate=("a",)))
    partial = local("WARNING", outcome("PARTIAL", True))
    targetless = local("PASS", outcome("COMPLETED", False))
    # (a target that does not apply to the light it names -- a two-beam metric on beams of different wavelengths -- is
    # not met: the aggregate ignores it, the requirements contract does not report it as met)
    inapplicable = local("PASS", outcome("COMPLETED", True, met=("a",), not_applicable=("b",)))
    # an unrelated quantity out of the model's scope: the aggregate reads it, the requirements do not
    scoped = local("OUT_OF_MODEL_SCOPE", outcome("COMPLETED", True, met=("a",)))
    failed = local("FAIL", outcome("FAILED", True, termination="NUMERICAL_FAILURE"))
    invalid = local("FAIL", outcome("FAILED", False, termination="INVALID_INPUT"))
    runs = (ok, missed, undecided, partial, targetless, inapplicable, scoped, failed, invalid)

    assert [run_exit_code("aggregate", r) for r in runs] == [0, 1, 2, 0, 0, 0, 3, 1, 1]
    assert [run_exit_code("execution", r) for r in runs] == [0, 0, 0, 1, 0, 0, 0, 1, 1]
    assert [run_exit_code("requirements", r) for r in runs] == [0, 1, 2, 3, 0, 2, 0, 3, 1]


def test_a_budget_that_stops_the_trace_is_reported_with_what_it_left_out():
    """The outcome names the budget, its effective value and the power left untraced -- not a blanket zero inference."""

    src = B.laser("S", wavelength_nm=780.0, power_W=1e-3, waist_radius_mm=0.5, position=(-60.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0))
    plane = B.observation_plane("far", position=(150.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0), radius_mm=12.7)
    project = ProjectSnapshot(components=(src,), observations=(plane,), targets=())
    tight = run(project, numerical=CFG.model_copy(update={"max_trace_steps": 1}))
    budgets = tight.outcome.execution.budgets
    assert "max_trace_steps" in budgets.exhausted
    assert budgets.limits["max_trace_steps"] == 1
    assert budgets.untraced_power_W is not None and budgets.untraced_power_W > 0.0
    assert "MAX_PATHS" in issue_codes(tight)


def test_requirements_exit_code_uses_local_model_applicability():
    from os_optics.cli_exit import run_exit_code
    from test_local_validity import _lens_run
    local = _lens_run(1200)
    assert local.result.outcome.requirements.met == ("power",)
    assert run_exit_code("execution", local) == 0
    assert run_exit_code("requirements", local) == 2
