"""The scientific-result outcome block of a run (``ValidationResult.outcome``).

Four independent questions, four independent answers (the scientific-result contract, docs/engine/RESULTS_AND_EXPORT.md):

* ``ExecutionRecord`` -- did the calculation complete, and if not, why and what did it cover?
* ``RequirementRecord`` -- which *declared* customer targets were assessed, and how?

Integrity and freshness are not here: they are properties of the artifact and of the Project it was made
from (``os_optics.local.freshness``), not of the physics. Commercial eligibility is not here either:
it is a property of the requested deliverable, not of the design.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import Field, model_validator

from ..errors.status import BUDGET_NAMES, ExecutionOutcome, TerminationReason
from .base import OmitAbsentFields, StrictModel


class ReturnAnalysisRecord(StrictModel):
    """One wavelength's return-light analysis: whether the light left out of the trace may come back.

    ``closed_return_screen`` is the screen's answer: ``COMPLETED_NO_RETURN`` (every trail of three or more surfaces of
    separate elements was examined and none closes: the open paths' count bounds that light), ``POTENTIAL_RETURN`` (a
    closed return the screen cannot rule out, ``closed_return_surfaces``: bounded by ``closed_series_bound`` where an
    all-orders series of parallel planes applies, else by nothing) or ``INCOMPLETE`` (the work limit came first: not
    known, so it bounds nothing). ``ended_in`` is where the analysis ended (``pairs``: the screen of surfaces facing each
    other, ``solid_folds``: the internal folds of plane-faced solids, ``closed_returns``), ``termination`` why
    (``COMPLETED``, ``RETURN_FOUND`` -- the search stops at the first closed return, shortest first, and names that one:
    others may remain --, ``BUDGET_EXHAUSTED``, ``NOT_STARTED``: an earlier stage used the work first, ``PARALLEL_STACK``:
    the reflecting surfaces are parallel planes, not searched for trails -- ``closed_return_surfaces`` are then the stack's
    groups of surfaces, not paths, and ``closed_series_bound`` the all-orders series that bounds them where one applies).
    ``closed_return_kinds`` says what each entry of ``closed_return_surfaces`` is: ``path`` (a closed trail, its surfaces
    in order), ``parallel_stack`` (the surfaces of a stack of parallel planes) or ``solid_folds`` (the faces of a
    plane-faced solid whose internal folds have no certified sum).
    ``max_return_analysis_work`` is the run's limit, shared by its wavelengths in the order the run reads them:
    ``work_before`` units the wavelengths analysed before this one spent, ``work_limit`` what that left this one (the run's
    limit less ``work_before``), ``work_done`` the units it spent, ``solid_fold_work`` of them in the fold-solid analysis;
    ``search_depth`` the longest trail examined (surfaces, its return to the first counted); ``frames_peak`` the most steps
    of a trail the search held at once, at most ``frames_limit`` (one per directed leg between the screened surfaces, plus
    one). Counts of work, never time: the same on every machine and runtime. A run cancelled or stopped during the analysis
    has no result, and so no record.
    """

    wavelength_nm: float = Field(gt=0)
    closed_return_screen: Literal["COMPLETED_NO_RETURN", "POTENTIAL_RETURN", "INCOMPLETE"]
    ended_in: Literal["pairs", "solid_folds", "closed_returns"]
    termination: Literal["COMPLETED", "RETURN_FOUND", "BUDGET_EXHAUSTED", "NOT_STARTED", "PARALLEL_STACK"]
    work_done: int = Field(ge=0)
    work_limit: int | None = Field(None, ge=0)
    search_depth: int = Field(0, ge=0)
    frames_peak: int = Field(0, ge=0)
    frames_limit: int = Field(0, ge=0)
    closed_return_surfaces: tuple[tuple[str, ...], ...] = ()
    closed_series_bound: float | None = Field(None, ge=1)
    work_before: int = Field(0, ge=0)
    solid_fold_work: int = Field(0, ge=0)
    closed_return_kinds: tuple[Literal["path", "parallel_stack", "solid_folds"], ...] = ()

    @model_validator(mode="after")
    def _consistent(self):
        incomplete = self.termination in {"BUDGET_EXHAUSTED", "NOT_STARTED"}
        if (self.closed_return_screen == "INCOMPLETE") != incomplete:
            raise ValueError("an incomplete screen is exactly one that its work limit stopped")
        if (self.closed_return_screen == "POTENTIAL_RETURN") != bool(self.closed_return_surfaces):
            raise ValueError("a potential return names its surfaces, and only a potential return does")
        if self.closed_series_bound is not None and self.closed_return_screen != "POTENTIAL_RETURN":
            raise ValueError("an all-orders series bounds a potential return only")
        if self.termination == "NOT_STARTED" and self.ended_in == "closed_returns":
            raise ValueError("a search not started ended before the closed-return screen")
        if self.termination == "PARALLEL_STACK" and self.closed_return_screen != "POTENTIAL_RETURN":
            raise ValueError("a stack of parallel planes is a potential return, bounded or not by its series")
        if self.frames_peak > self.frames_limit or (self.work_limit is not None and self.work_done > self.work_limit):
            raise ValueError("work and storage stay within their limits")
        if self.solid_fold_work > self.work_done:
            raise ValueError("the fold-solid analysis' work is part of the work done")
        if len(self.closed_return_kinds) != len(self.closed_return_surfaces):
            raise ValueError("each closed return named has its kind")
        return self


class BudgetUsage(OmitAbsentFields, StrictModel):
    """The effective resource budgets of a run, which of them it exhausted, and what that left out.

    ``limits`` records the settings actually used (``NumericalConfig`` field -> value), so a result states
    its own resource conditions and a later reading never has to assume the defaults of the runtime that
    produced it. ``exhausted`` names the budgets that ran out (a subset of ``limits``, in ``BUDGET_NAMES``
    order); the issues say in words what each one left untraced. ``return_analysis``: per wavelength the run
    analysed, the return-light analysis's answer, work and storage (``ReturnAnalysisRecord``; absent where no
    bound read it). Nothing here is a physical quantity.
    """

    absent_omitted: ClassVar[tuple[str, ...]] = ("return_analysis",)

    limits: dict[str, int] = Field(default_factory=dict, description="The effective limit of each resource budget.")
    exhausted: tuple[str, ...] = Field((), description="The budgets that ran out, in BUDGET_NAMES order.")
    exhausted_by_phase: dict[str, tuple[str, ...]] = Field(default_factory=dict,
        description="The budgets each analysis ran out of: its own termination record. primary: the beam trace (the only "
        "phase that makes the execution PARTIAL); untraced_light: the probe traces of light the trace does not follow "
        "(capability untraced_light); return_analysis: the return-light analysis read by the plane bounds and the surface "
        "exposures (capabilities plane_bounds, surface_exposures); interference: the pair listing (capability beam_pairs). A "
        "phase other than primary leaves the execution COMPLETED; the quantities that lost a bound say so themselves.")
    affected_source_ids: tuple[str, ...] = ()
    untraced_power_W: float | None = Field(
        None, ge=0.0, description="Power the primary trace budgets left untraced. Auxiliary probe omissions remain in their issues and are not added again."
    )
    affected_path_ids: tuple[str, ...] = Field(
        (), description="The beam paths the run cut short (a path that met the interaction limit or the trace budget)."
    )
    affected_observation_ids: tuple[str, ...] = Field(
        (), description="Observation planes whose beam pairs the interference-pair budget left partly unanalysed."
    )
    return_analysis: tuple[ReturnAnalysisRecord, ...] = Field(
        (), description="Per wavelength analysed, the return-light analysis: the closed-return screen's answer "
        "(COMPLETED_NO_RETURN, POTENTIAL_RETURN or INCOMPLETE), where and why it ended, its work against what "
        "max_return_analysis_work (the run's) left it and its trail storage against its limit. Absent where no bound read "
        "the analysis."
    )


class ExecutionRecord(StrictModel):
    """Did the requested calculation complete, and what does that mean for the results?

    ``complete`` is the direct answer for the reader who only needs one: true exactly for ``COMPLETED``.
    ``PARTIAL``: the primary trace exhausted a budget (``budgets.exhausted_by_phase["primary"]``). A budget an auxiliary
    analysis exhausted (untraced-light probes, return-light analysis, pair listing) is listed in ``budgets`` and leaves the
    execution ``COMPLETED``. A ``PARTIAL`` run still carries every value it did calculate -- a budget that ran out never
    makes a calculated quantity zero or invalid.
    """

    outcome: ExecutionOutcome
    termination: TerminationReason
    reason: str = Field("", description="Why, in one sentence, when the outcome is not COMPLETED.")
    complete: bool
    budgets: BudgetUsage = Field(default_factory=BudgetUsage)


class RequirementRecord(StrictModel):
    """Which declared customer targets were assessed, and how they came out.

    ``declared`` is false for a Project that states no targets: there is then no requirement assessment at
    all, which is not universal success. The four id lists hold every target that was assessed, in Project order: they
    alone say how the requirements came out. A declared target in none of them was not assessed: the run stopped before
    the targets, or the caller of a narrowed evaluation did not select it (``ValidationResult.evaluation`` names it in
    ``targets_not_evaluated``).
    """

    declared: bool
    met: tuple[str, ...] = ()
    unmet: tuple[str, ...] = ()
    indeterminate: tuple[str, ...] = ()
    not_applicable: tuple[str, ...] = ()


class ResultOutcome(StrictModel):
    """The run's scientific-result contract, independent of any customer assessment of the design."""

    execution: ExecutionRecord
    requirements: RequirementRecord


class RunStop(StrictModel):
    """Where and why a run stopped before it completed (``StoppedRun.stop``; ``os_optics.run_control``).

    ``cause``: ``CALLER`` (the caller cancelled it: ``RunMonitor.cancel`` or Ctrl-C), ``MEMORY_LIMIT``
    or ``ARTIFACT_LIMIT`` (a resource limit the caller set was reached, or -- ``phase`` ``preflight`` -- the run's estimate
    exceeded it before the trace). ``phase`` is the innermost phase the run was in (``run_control.RUN_PHASES``), ``phases``
    every phase it was in, outermost first, ``completed_phases`` those it finished; ``done`` / ``total`` / ``unit`` count its
    work in that phase (``total`` null when not known in advance)."""

    cause: Literal["CALLER", "MEMORY_LIMIT", "ARTIFACT_LIMIT"]
    message: str = Field(description="Why the run stopped, in one sentence.")
    phase: str
    phases: tuple[str, ...] = ()
    completed_phases: tuple[str, ...] = ()
    done: int | None = Field(None, ge=0)
    total: int | None = Field(None, ge=0)
    unit: str | None = None
    limit_bytes: int | None = Field(None, ge=1, description="The memory or run-file limit that stopped the run (bytes).")
    measured_bytes: int | None = Field(None, ge=0, description="The memory in use, or the run-file bytes, that reached it.")
    estimated_bytes: int | None = Field(None, ge=0, description="The estimate that exceeded it before the trace (phase preflight).")


class StoppedRun(StrictModel):
    """A run that stopped before it completed: it has no result.

    Nothing the run computed before it stopped is reported -- no observation, target, path or bound --, and no file of
    it is written but this record (``run_stopped.json`` in the directory it was given; an earlier run there stays as it
    was). ``outcome``
    states it in the scientific-result vocabulary: execution ``CANCELLED`` (termination ``CANCELLED``, ``complete``
    false), and requirements with none of the declared targets assessed. ``affected_observation_ids`` /
    ``affected_target_ids`` list every observation plane and target of the Project: none has a value from this run.
    ``progress`` is the run's monitor record (elapsed time, seconds per phase, peak memory read, run-file bytes written,
    the limits): facts of this execution on this machine, not of the design."""

    kind: Literal["STOPPED_RUN"] = "STOPPED_RUN"
    project_hash: str
    engine_input_hash: str | None = None
    outcome: ResultOutcome
    stop: RunStop
    affected_observation_ids: tuple[str, ...] = ()
    affected_target_ids: tuple[str, ...] = ()
    progress: dict[str, Any] = Field(default_factory=dict)


def budget_limits(**limits: int) -> dict[str, int]:
    """The budgets a run used, in the vocabulary's order (unknown keys are dropped, not guessed)."""

    return {name: int(limits[name]) for name in BUDGET_NAMES if name in limits}


__all__ = ["BudgetUsage", "ExecutionRecord", "RequirementRecord", "ResultOutcome", "ReturnAnalysisRecord", "RunStop", "StoppedRun",
           "budget_limits"]
