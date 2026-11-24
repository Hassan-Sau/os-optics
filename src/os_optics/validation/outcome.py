"""Build the scientific-result outcome block of a run (``contracts/outcome.py``).

Pure functions of what the run already computed: the issues it raised, the budgets it was given and the
targets it assessed. Nothing here changes a number, a status or a decision -- the block *reports* the run's
own facts in the vocabulary of the scientific-result contract, so no consumer has to infer an execution
outcome or a requirement result from the aggregate ``Status``.
"""

from __future__ import annotations

from ..config import NumericalConfig
from ..contracts.outcome import BudgetUsage, ExecutionRecord, RequirementRecord, ResultOutcome, ReturnAnalysisRecord, budget_limits
from ..contracts.results import ObservationResult, TargetResult
from ..errors.codes import ErrorCode
from ..errors.status import BUDGET_NAMES, ExecutionOutcome, QuantityState, RequirementAssessment, TerminationReason


#: The phase of the primary trace in ``BudgetUsage.exhausted_by_phase``: only its budgets make an execution ``PARTIAL``.
PRIMARY_PHASE = "primary"


def _context(issues, code: ErrorCode) -> list[dict]:
    return [dict(issue.numerical_context) for issue in issues if issue.code is code]


def budget_usage(issues, cfg: NumericalConfig, analyses: tuple[ReturnAnalysisRecord, ...] = ()) -> BudgetUsage:
    """Which of the run's resource budgets it exhausted, and what each left out.

    The trace budget (``max_paths`` / ``max_trace_steps``) reports itself through one ``MAX_PATHS`` issue whose
    ``numerical_context.budget`` names the limit that was reached, so the two remain distinguishable without a
    second issue code: the code stays ``MAX_PATHS`` for readers that know it. The return-light analysis budget
    (``max_return_analysis_work``) reports itself through ``RETURN_ANALYSIS_BUDGET`` (phase ``return_analysis``) where a
    bound read what it withheld; ``analyses`` records every wavelength the analysis ran for (``return_analysis``: its
    screen's answer, work and storage), whether or not it withheld anything.
    """

    exhausted: list[str] = []
    untraced = 0.0
    seen_power = False
    paths: list[str] = []
    sources: set[str] = set()
    phases: dict[str, set[str]] = {}
    for context in _context(issues, ErrorCode.MAX_PATHS):
        phase = context.get("phase", "primary")
        sources.update(context.get("affected_source_ids", ()))
        paths.extend(p for p in context.get("affected_path_ids", ()) if p not in paths)
        for name in context.get("budget") or ():
            if isinstance(name, str) and name in ("max_paths", "max_trace_steps") and name not in exhausted:
                exhausted.append(name)
            if name in ("max_paths", "max_trace_steps"):
                phases.setdefault(phase, set()).add(name)
        power = context.get("untraced_power_W")
        if phase == "primary" and isinstance(power, (int, float)):
            untraced += float(power)
            seen_power = True
    for issue in issues:
        if issue.code is ErrorCode.MAX_INTERACTIONS:
            context = issue.numerical_context
            phase = context.get("phase", "primary")
            phases.setdefault(phase, set()).add("max_interactions_per_path")
            sources.update(context.get("affected_source_ids", ()))
            paths.extend(p for p in context.get("affected_path_ids", ()) if p not in paths)
            if "max_interactions_per_path" not in exhausted:
                exhausted.append("max_interactions_per_path")
            power = issue.numerical_context.get("untraced_power_W")
            if phase == "primary" and isinstance(power, (int, float)):
                untraced += float(power)
                seen_power = True
            if issue.location and issue.location.startswith("path:"):
                path_id = issue.location.removeprefix("path:")
                if path_id not in paths:
                    paths.append(path_id)
    observations: list[str] = []
    for issue in issues:
        if issue.code is ErrorCode.MAX_BEAM_PAIRS:
            phases.setdefault("interference", set()).add("max_interference_pairs")
            if "max_interference_pairs" not in exhausted:
                exhausted.append("max_interference_pairs")
            if issue.location and issue.location.startswith("observation:"):
                observation_id = issue.location.removeprefix("observation:")
                if observation_id not in observations:
                    observations.append(observation_id)
    for context in _context(issues, ErrorCode.RETURN_ANALYSIS_BUDGET):
        # (the return-light analysis ran out of its work: the bounds it withheld name their planes)
        phases.setdefault("return_analysis", set()).add("max_return_analysis_work")
        if "max_return_analysis_work" not in exhausted:
            exhausted.append("max_return_analysis_work")
        observations.extend(o for o in context.get("affected_observation_ids", ()) if o not in observations)
    order = BUDGET_NAMES
    return BudgetUsage(
        limits=budget_limits(max_interactions_per_path=cfg.max_interactions_per_path, max_paths=cfg.max_paths,
                             max_trace_steps=cfg.max_trace_steps, max_interference_pairs=cfg.max_interference_pairs,
                             max_return_analysis_work=cfg.max_return_analysis_work),
        exhausted=tuple(name for name in order if name in exhausted),
        exhausted_by_phase={phase: tuple(name for name in order if name in values) for phase, values in sorted(phases.items())},
        affected_source_ids=tuple(sorted(sources)),
        untraced_power_W=untraced if seen_power else None,
        affected_path_ids=tuple(paths),
        affected_observation_ids=tuple(observations),
        return_analysis=tuple(analyses),
    )


def execution_record(issues, cfg: NumericalConfig, *, failed: TerminationReason | None = None, reason: str = "",
                     cancelled: bool = False, analyses: tuple[ReturnAnalysisRecord, ...] = ()) -> ExecutionRecord:
    """The execution outcome: failed (input / numerical), cancelled, partial or complete.

    ``PARTIAL`` only when the primary trace exhausted a budget (``budgets.exhausted_by_phase["primary"]``): the traced
    beams themselves are then incomplete. A budget an auxiliary analysis exhausted -- the probes of untraced light
    (``untraced_light``), the return-light analysis (``return_analysis``), the pair listing (``interference``) -- is
    recorded in ``budgets.exhausted`` / ``exhausted_by_phase`` and on the quantities that lost a bound; the calculation
    itself completed."""

    budgets = budget_usage(issues, cfg, analyses)
    if failed is not None:
        return ExecutionRecord(outcome=ExecutionOutcome.FAILED, termination=failed, reason=reason, complete=False,
                               budgets=budgets)
    if cancelled:
        return ExecutionRecord(outcome=ExecutionOutcome.CANCELLED, termination=TerminationReason.CANCELLED,
                               reason=reason or "the caller cancelled the run", complete=False, budgets=budgets)
    primary = budgets.exhausted_by_phase.get(PRIMARY_PHASE, ())
    if primary:
        return ExecutionRecord(
            outcome=ExecutionOutcome.PARTIAL, termination=TerminationReason.BUDGET_EXHAUSTED,
            reason="a resource budget of the trace ran out (" + ", ".join(primary) + "); the results count the light that "
                   "was traced, and each affected quantity says what it leaves out",
            complete=False, budgets=budgets)
    return ExecutionRecord(outcome=ExecutionOutcome.COMPLETED, termination=TerminationReason.COMPLETED, complete=True,
                           budgets=budgets)


def requirement_record(targets: tuple[TargetResult, ...], *, declared: bool | None = None) -> RequirementRecord:
    """Which declared targets were met, unmet, undecided -- or that none was declared at all.

    ``declared`` overrides the inference from ``targets``: a run that stopped before evaluating the targets still knows
    whether the Project declared any, and must not report "no requirement assessment" for a Project that has one.
    """

    if declared is None:
        declared = bool(targets)
    if not declared:
        return RequirementRecord(declared=False)
    if not targets:
        return RequirementRecord(declared=True)

    def ids(assessment: RequirementAssessment) -> tuple[str, ...]:
        return tuple(t.target_id for t in targets if (t.assessment or RequirementAssessment.INDETERMINATE) is assessment)

    return RequirementRecord(declared=True, met=ids(RequirementAssessment.MET), unmet=ids(RequirementAssessment.UNMET),
                             indeterminate=ids(RequirementAssessment.INDETERMINATE),
                             not_applicable=ids(RequirementAssessment.NOT_APPLICABLE))


def result_outcome(issues, cfg: NumericalConfig, targets: tuple[TargetResult, ...], *,
                   failed: TerminationReason | None = None, reason: str = "", cancelled: bool = False,
                   declared: bool | None = None, analyses: tuple[ReturnAnalysisRecord, ...] = ()) -> ResultOutcome:
    """The whole block: the execution record and the requirement record (``analyses``: the return-light analysis of each
    wavelength the run analysed, ``validation/pipeline.return_analysis_records``)."""

    return ResultOutcome(execution=execution_record(issues, cfg, failed=failed, reason=reason, cancelled=cancelled,
                                                    analyses=analyses),
                         requirements=requirement_record(targets, declared=declared))


def quantity_state_of(observation: ObservationResult) -> QuantityState:
    """Whether a plane's power is a known quantity, a known zero, or unknown (``QuantityState``); ``OUT_OF_MODEL_SCOPE`` where the
    power of a beam on it, or of the light a surface scattered out of such a beam, is out of model scope (``power_out_of_scope_at``).

    A plane no traced beam reaches is a known zero only when every launched line was traced to a known end
    (``power_complete``). Light of unknown fate that may be detected there (``untraced_power_W``) leaves it ``UNKNOWN``,
    never 0 W, whatever bound is supported. Where the only light that may arrive is light the probe traces followed to the
    plane under a supported bound (``followed_light``), its power is ``BOUNDED`` between ``power_lower_bound_W`` 0.0 and
    ``power_upper_bound_W``, with no value (``total_power_W`` null): the probes' powers are bounds, not the plane's reading;
    that interval grades the plane's power targets (``metrics._no_arrival``). Light left out as negligible (below the lines' negligible level) is the stated truncation of a value and
    changes no state while the most it brings to the plane stays within that level (``LineFate.faint_accounted``; above it,
    surfaces facing each other or many crossings may send it back: the plane is ``BOUNDED`` from 0, ``UNKNOWN`` without a
    bound): such a plane is ``KNOWN_ZERO`` where no beam arrives and ``KNOWN`` where one does, with the most that
    light brings in ``power_upper_bound_W``. ``BOUNDED`` is a lit plane whose power lies in a supported interval: from its
    traced value up to ``power_upper_bound_W``, or with an open upper end where nothing bounds the light not followed
    (``power_upper_bound_W`` null; the plane's ``value_basis`` entry names the causes). A lit plane a
    beam of which loses a bounded share to a surface before
    it (``ArrivalResult.power_taken_before_upper_W``) is ``BOUNDED`` too. A power on the geometric pupil-sample basis
    (``ArrivalResult.power_basis``) is a ``KNOWN`` sampled value with its sampling resolution beside it
    (``ObservationResult.power_sampling_resolution_W``): convergence evidence, not a supported interval, so neither
    ``BOUNDED`` nor an upper bound is taken from it. So is a plane no counted ray of a sampled line reaches (a sampled
    zero, ``SampledLightAtPlane``): ``KNOWN`` 0 W with one ray's weight per such line as its resolution, never ``KNOWN_ZERO``.
    """

    if any(a.power_out_of_scope_at for a in observation.arrivals) or any(line.power_out_of_scope_at for line in observation.sampled_light):
        return QuantityState.OUT_OF_MODEL_SCOPE
    if not observation.power_complete:
        if not observation.lit and not observation.total_power_W and observation.untraced_power_W > 0.0:
            # (only light the probes followed reaches it, under a supported bound: the stated interval [0, upper], no value)
            if observation.power_lower_bound_W == 0.0 and observation.power_upper_bound_W is not None:
                return QuantityState.BOUNDED
            return QuantityState.UNKNOWN
        if "power_upper_bound_W" in observation.model_fields_set:
            if observation.power_upper_bound_W is not None:
                return QuantityState.BOUNDED
            # (no supported upper end: the traced value of a lit plane is the lower end of its power -- BOUNDED with an open
            # upper end, the cause in its value_basis entry; a plane with no value, or a traced 0 W, stays UNKNOWN)
            return QuantityState.BOUNDED if observation.power_lit and observation.total_power_W else QuantityState.UNKNOWN
        # (a bounded power needs a *supported* bound: on an absorbing plane the untraced light the probes followed adds
        # at most once, so the power lies in [total_power_W, total_power_W + untraced_power_W]. On a transparent plane
        # that light may cross the plane several times and nothing bounds it from above -- the same reason
        # ``metrics._untraced_bounds`` refuses a bound there. No plane, no absorbing area: unknown.)
        plane = observation.plane
        if observation.total_power_W is not None and observation.untraced_power_W > 0.0 and plane is not None and plane.absorbing:
            return QuantityState.BOUNDED
        return QuantityState.UNKNOWN
    if any(a.power_taken_before_upper_W is not None for a in observation.arrivals):
        # (part of a beam is taken before the plane by a bounded share: the plane's power lies in a supported interval)
        return QuantityState.BOUNDED
    if observation.lit:
        return QuantityState.KNOWN
    if observation.total_power_W == 0.0 and observation.power_sampling_resolution_W is None:
        # (a sampled zero -- no ray of a sampled line counted, power_sampling_resolution_W beside it -- stays KNOWN)
        return QuantityState.KNOWN_ZERO
    return QuantityState.KNOWN
__all__ = ["budget_usage", "execution_record", "quantity_state_of", "requirement_record", "result_outcome"]
