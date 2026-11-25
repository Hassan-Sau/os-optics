"""Scientific outcomes carried with a local run and its submitted evidence."""

from __future__ import annotations

from typing import Literal

from pydantic import model_serializer, Field, model_validator

from ..base import StrictContractModel
from ..common.safe_int import SafeJsonInt

RequirementAssessmentV1 = Literal["MET", "UNMET", "INDETERMINATE", "NOT_APPLICABLE"]
QuantityStateV1 = Literal["KNOWN", "KNOWN_ZERO", "BOUNDED", "UNKNOWN", "OUT_OF_MODEL_SCOPE", "NOT_APPLICABLE"]
BudgetNameV1 = Literal["max_interactions_per_path", "max_paths", "max_trace_steps", "max_interference_pairs",
                       "max_return_analysis_work"]


class ReturnAnalysisRecordV1(StrictContractModel):
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
    work_done: SafeJsonInt = Field(ge=0)
    work_limit: SafeJsonInt | None = Field(None, ge=0)
    search_depth: SafeJsonInt = Field(0, ge=0)
    frames_peak: SafeJsonInt = Field(0, ge=0)
    frames_limit: SafeJsonInt = Field(0, ge=0)
    closed_return_surfaces: tuple[tuple[str, ...], ...] = ()
    closed_series_bound: float | None = Field(None, ge=1)
    work_before: SafeJsonInt = Field(0, ge=0)
    solid_fold_work: SafeJsonInt = Field(0, ge=0)
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


class BudgetUsageV1(StrictContractModel):
    limits: dict[BudgetNameV1, SafeJsonInt] = Field(default_factory=dict)
    exhausted: tuple[BudgetNameV1, ...] = ()
    exhausted_by_phase: dict[Literal["primary", "untraced_light", "interference", "return_analysis"], tuple[BudgetNameV1, ...]] = Field(default_factory=dict)
    affected_source_ids: tuple[str, ...] = ()
    untraced_power_W: float | None = Field(None, ge=0)
    affected_path_ids: tuple[str, ...] = ()
    affected_observation_ids: tuple[str, ...] = ()
    # Per wavelength analysed, the return-light analysis (ReturnAnalysisRecordV1). Omitted when empty.
    return_analysis: tuple[ReturnAnalysisRecordV1, ...] = ()

    @model_serializer(mode="wrap")
    def _omit_absent_return_analysis(self, handler):
        data = handler(self)
        if not self.return_analysis:
            data.pop("return_analysis", None)
        return data

    @model_validator(mode="after")
    def _consistent(self):
        if any(value < 1 for value in self.limits.values()):
            raise ValueError("resource limits must be positive")
        if len(set(self.exhausted)) != len(self.exhausted) or not set(self.exhausted) <= self.limits.keys():
            raise ValueError("exhausted budgets must uniquely name recorded limits")
        if self.exhausted_by_phase:
            phases = tuple(self.exhausted_by_phase.values())
            if any(len(names) != len(set(names)) for names in phases) or set().union(*map(set, phases)) != set(self.exhausted):
                raise ValueError("phase exhaustion must uniquely name budgets and cover exactly the exhausted budgets")
        return self


class ExecutionRecordV1(StrictContractModel):
    # CANCELLED: the run stopped before it completed (by its caller or a resource limit it set) and has no result.
    outcome: Literal["COMPLETED", "PARTIAL", "FAILED", "CANCELLED"]
    termination: Literal["COMPLETED", "BUDGET_EXHAUSTED", "NUMERICAL_FAILURE", "INVALID_INPUT", "OUT_OF_MODEL_SCOPE", "CANCELLED"]
    reason: str = ""
    complete: bool
    budgets: BudgetUsageV1 = Field(default_factory=BudgetUsageV1)

    @model_validator(mode="after")
    def _consistent(self):
        allowed = {"COMPLETED": {"COMPLETED"}, "PARTIAL": {"BUDGET_EXHAUSTED"},
                   "FAILED": {"NUMERICAL_FAILURE", "INVALID_INPUT", "OUT_OF_MODEL_SCOPE"}, "CANCELLED": {"CANCELLED"}}
        if self.complete != (self.outcome == "COMPLETED") or self.termination not in allowed[self.outcome]:
            raise ValueError("execution outcome, termination and completeness disagree")
        # PARTIAL exactly when the primary trace exhausted a budget. A budget an auxiliary analysis exhausted (the probes of
        # untraced light, the return-light analysis, the pair listing) is recorded in exhausted / exhausted_by_phase and
        # leaves the execution COMPLETED. A record without the phase map (written before it existed) keeps the earlier
        # rule: PARTIAL exactly when any budget is listed.
        phases = self.budgets.exhausted_by_phase
        partial = "primary" in phases if phases else bool(self.budgets.exhausted)
        if partial != (self.outcome == "PARTIAL") and self.outcome in {"COMPLETED", "PARTIAL"}:
            raise ValueError("completed/partial execution disagrees with the budgets the primary trace exhausted")
        return self


class RequirementRecordV1(StrictContractModel):
    declared: bool
    met: tuple[str, ...] = ()
    unmet: tuple[str, ...] = ()
    indeterminate: tuple[str, ...] = ()
    not_applicable: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _consistent(self):
        ids = self.met + self.unmet + self.indeterminate + self.not_applicable
        if len(ids) != len(set(ids)) or (ids and not self.declared):
            raise ValueError("requirement groups must uniquely partition declared targets")
        return self


class ResultOutcomeV1(StrictContractModel):
    execution: ExecutionRecordV1
    requirements: RequirementRecordV1


class ObservationQuantityV1(StrictContractModel):
    """Power evidence for one observation, including supported bounds when available."""

    observation_id: str
    quantity_state: QuantityStateV1 | None = None
    total_power_W: float | None = Field(None, ge=0)
    power_complete: bool
    untraced_power_W: float = Field(0, ge=0)
    power_upper_bound_W: float | None = Field(None, ge=0)
    # A lit plane with complete evidence whose power lies in [power_lower_bound_W, total_power_W]: part of a beam is taken
    # by a surface before the plane, by a bounded share (quantity_state BOUNDED). Omitted elsewhere.
    power_lower_bound_W: float | None = Field(None, ge=0)
    # A sampled total_power_W (the Engine's geometric pupil-sample basis): the sum of its beams' sampling resolutions,
    # convergence evidence and not a bound. A KNOWN sampled value takes no upper bound from it. Omitted elsewhere.
    # KNOWN / KNOWN_ZERO mean that nothing above the stated negligible level is left out of total_power_W: light left out
    # below that level is the truncation of the value, so power_upper_bound_W is the value, a supported bound above it
    # (the value plus the most that light may add), or null where nothing bounds that light.
    power_sampling_resolution_W: float | None = Field(None, ge=0)
    issue_codes: tuple[str, ...] = ()

    @model_serializer(mode="wrap")
    def _omit_historical_bound(self, handler):
        data = handler(self)
        if self.power_lower_bound_W is None:
            data.pop("power_lower_bound_W", None)
        if self.power_sampling_resolution_W is None:
            data.pop("power_sampling_resolution_W", None)
        if "power_upper_bound_W" not in self.model_fields_set:
            data.pop("power_upper_bound_W", None)
        elif self.power_upper_bound_W is None:
            # Explicit null means unsupported, not historical absence. Keep it in compact payloads.
            data["power_upper_bound_W"] = None
        return data

    @model_validator(mode="after")
    def _consistent(self):
        if self.quantity_state in {"KNOWN", "KNOWN_ZERO"} and self.untraced_power_W != 0:
            raise ValueError("known power cannot have unresolved power")
        if self.quantity_state == "KNOWN_ZERO" and (not self.power_complete or self.total_power_W != 0):
            raise ValueError("known zero requires complete evidence and zero power")
        if self.quantity_state == "KNOWN" and (not self.power_complete or self.total_power_W is None):
            raise ValueError("known power requires complete evidence and a value")
        explicit = "power_upper_bound_W" in self.model_fields_set
        if explicit and self.power_upper_bound_W is not None and self.power_upper_bound_W < (self.total_power_W or 0):
            raise ValueError("power upper bound cannot be below traced power")
        lower = self.power_lower_bound_W
        if lower is not None and (self.total_power_W is None or lower > self.total_power_W):
            raise ValueError("power lower bound requires a value at or above it")
        if self.quantity_state == "BOUNDED" and self.power_complete:
            if lower is None:
                raise ValueError("bounded complete power requires a lower bound")
        elif self.quantity_state == "BOUNDED" and explicit and self.power_upper_bound_W is None:
            # An open upper end: nothing bounds the light not followed, and the traced value is the lower end.
            if self.total_power_W is None or self.total_power_W <= 0:
                raise ValueError("bounded power with an open upper end requires a positive value")
        elif self.quantity_state == "BOUNDED" and (
                self.power_upper_bound_W <= 0 if explicit else self.total_power_W is None or self.untraced_power_W <= 0):
            raise ValueError("bounded power requires a value and unresolved positive power")
        return self


class ModelScopeApplicabilityV1(StrictContractModel):
    """Post-trace model applicability; raw Engine values and hashes remain unchanged."""

    instance_id: str
    code: str
    message: str
    surface_id: str | None = None
    member_id: str | None = None
    affected_observation_ids: tuple[str, ...] = ()
    affected_target_ids: tuple[str, ...] = ()
    affected_element_ids: tuple[str, ...] | None = None
