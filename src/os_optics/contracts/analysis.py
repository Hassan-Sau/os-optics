"""The result of an analysis derived from a run: an imaging analysis, a focal field, a ghost search, a configuration comparison.

An analysis reads a completed run (or a stated optical request) and reports scientific quantities. It is not part of the run's
result and does not change it: the run's result hash, its targets and its outcome are what they were. Each analysis document says

* ``execution`` -- whether the analysis itself completed, and which of its own budgets ran out (``ExecutionRecord``);
* ``models`` -- the physical model each quantity rests on, its assumptions and the conditions of validity the case was checked
  against;
* ``quantities`` -- every reported value with its unit, the scope it belongs to, its state (``QuantityState``: the same words a
  plane's power uses), the interval when only bounds are known, and the evidence that the discretization resolved it;
* ``completeness`` -- for an enumeration (ghost paths), what was listed, what was not followed and a bound on what was left out.

There is no verdict on the design. A quantity outside its model's validity is ``OUT_OF_MODEL_SCOPE``; it has no value, or a
value with a ``value_basis`` that says what the number is (``contracts/basis.py``), while the quantities that do not depend on it
keep theirs. Only a ``KNOWN`` value, a ``PROVEN`` interval, or a qualified value in the direction its kind proves may decide a
threshold, and a value whose ``convergence.resolution`` straddles a threshold does not decide it.

Identity. ``identity.physical`` names what the numbers depend on (the Engine build, the run's input and result hashes, the
analysis settings); ``identity.requirements`` names the thresholds an evaluation used. Changing a threshold moves only the latter.
"""

from __future__ import annotations

from enum import StrEnum
from typing import ClassVar

from pydantic import Field, model_validator

from ..errors import ValidationIssue
from ..errors.status import QuantityState
from .base import Id, OmitAbsentFields, StrictModel
from .basis import ValidityCheck, ValueBasis, check_state, decides_threshold
from .outcome import ExecutionRecord

ANALYSIS_RESULT_SCHEMA = "opensci.analysis_result"
ANALYSIS_RESULT_SCHEMA_VERSION = "1.0"

#: States whose quantity carries a single value.
VALUED_STATES = frozenset({QuantityState.KNOWN, QuantityState.KNOWN_ZERO})


class QuantityOrigin(StrEnum):
    """Where a value comes from: computed by the stated model, or stated from outside it (a datasheet, a measurement)."""

    MODELED = "MODELED"
    EXTERNALLY_CHARACTERIZED = "EXTERNALLY_CHARACTERIZED"


class IntervalKind(StrEnum):
    """How an interval is known.

    ``PROVEN``: the true value lies in it under the stated model (it may decide a threshold). ``SAMPLING``: the spread of the
    value across the discretizations tried; it states resolution, not a bound. ``NOMINAL``: a scale of what may be missing,
    which decides nothing.
    """

    PROVEN = "PROVEN"
    SAMPLING = "SAMPLING"
    NOMINAL = "NOMINAL"


class ConvergenceMethod(StrEnum):
    """How the discretization of a value was checked.

    ``ANALYTIC``: closed form, no discretization. ``REFINEMENT``: recomputed on a finer discretization (``coarse`` / ``fine``).
    ``HALF_SAMPLE``: recomputed from each half of a sample set. ``REFERENCE``: compared with an independent method. ``NONE``:
    not checked.
    """

    ANALYTIC = "ANALYTIC"
    REFINEMENT = "REFINEMENT"
    HALF_SAMPLE = "HALF_SAMPLE"
    REFERENCE = "REFERENCE"
    NONE = "NONE"


class ModelBasis(StrictModel):
    """The physical model a quantity rests on."""

    model_id: Id
    name: str
    version: str = Field(description="The model's algorithm version: part of the analysis's physical identity.")
    assumptions: tuple[str, ...] = ()
    validity: tuple[ValidityCheck, ...] = ()


class Interval(StrictModel):
    """Bounds of a quantity; an open side is ``null``."""

    lower: float | None = None
    upper: float | None = None
    kind: IntervalKind

    @model_validator(mode="after")
    def _ordered(self) -> "Interval":
        if self.lower is not None and self.upper is not None and self.lower > self.upper:
            raise ValueError("interval lower exceeds upper")
        return self


class Convergence(StrictModel):
    """Evidence that the discretization resolved a value."""

    method: ConvergenceMethod
    parameter: str | None = Field(None, description="The discretization parameter that was varied (grid size, pupil samples).")
    coarse: float | None = None
    fine: float | None = None
    difference: float | None = Field(None, ge=0.0, description="|fine - coarse| or the half-sample difference.")
    resolution: float | None = Field(None, ge=0.0, description="The smallest difference in this value the analysis resolves.")
    converged: bool | None = None


class QuantityScope(StrictModel):
    """What a quantity belongs to. Unset members do not restrict it."""

    configuration_id: Id | None = None
    observation_id: Id | None = None
    field_id: Id | None = None
    source_id: Id | None = None
    path_id: str | None = None
    surface_id: str | None = None
    wavelength_nm: float | None = None


class Quantity(OmitAbsentFields, StrictModel):
    """One reported scientific quantity.

    ``value`` is set for ``KNOWN`` / ``KNOWN_ZERO`` (a ``BOUNDED`` quantity may also carry its nominal value beside its
    interval); ``values_ref`` names an array in the document's ``arrays`` instead of a scalar. Any other state carries a
    ``reason`` code and, when a stated input is missing, ``missing``; it carries a value exactly when ``value_basis`` says what
    that number is (``contracts/basis.py``: a lower or upper bound, or an estimate outside a validity limit or with a
    stand-in input). ``value_basis`` is omitted when absent.
    """

    absent_omitted: ClassVar[tuple[str, ...]] = ("value_basis",)

    quantity_id: Id
    name: str
    unit: str | None = None
    scope: QuantityScope = Field(default_factory=QuantityScope)
    state: QuantityState
    origin: QuantityOrigin = QuantityOrigin.MODELED
    value: float | None = None
    values_ref: str | None = None
    interval: Interval | None = None
    basis: Id | None = Field(None, description="The model_id of the ModelBasis this quantity rests on.")
    convergence: Convergence | None = None
    reason: str | None = None
    message: str = ""
    missing: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = Field((), description="Quantity ids, issue codes or surface ids this value depends on.")
    source: str | None = Field(None, description="Where an externally characterized value comes from.")
    value_basis: ValueBasis | None = Field(
        None, description="What the number in value / values_ref is, where it is not simply the model's value (a lower or upper "
        "bound, an estimate) or a caller setting other than the Engine default produced it. Omitted when absent.")

    @model_validator(mode="after")
    def _state_consistent(self) -> "Quantity":
        valued = self.value is not None or self.values_ref is not None
        if self.state in VALUED_STATES and not valued:
            raise ValueError(f"{self.state.value} quantity {self.quantity_id!r} has no value")
        if self.state is QuantityState.BOUNDED and self.interval is None:
            raise ValueError(f"BOUNDED quantity {self.quantity_id!r} has no interval")
        if self.state not in VALUED_STATES and self.state is not QuantityState.BOUNDED:
            if valued and self.value_basis is None:
                raise ValueError(f"{self.state.value} quantity {self.quantity_id!r} carries a value without a value_basis")
            if not self.reason:
                raise ValueError(f"{self.state.value} quantity {self.quantity_id!r} has no reason")
        if self.value_basis is not None and not valued:
            raise ValueError(f"quantity {self.quantity_id!r} has a value_basis but no value")
        check_state(self.state, self.value_basis, self.quantity_id)
        if self.state is QuantityState.KNOWN_ZERO and self.value is not None and self.value != 0.0:
            raise ValueError(f"KNOWN_ZERO quantity {self.quantity_id!r} is not zero")
        return self

    def decides(self, threshold: float) -> bool:
        """Whether this quantity can decide a comparison with ``threshold``: a known value outside its resolution of the
        threshold, a proven interval wholly on one side, or a qualified value in the direction its kind proves (a lower bound
        above the threshold, an upper bound below it; an estimate never)."""

        resolution = self.convergence.resolution if self.convergence and self.convergence.resolution else 0.0
        if self.state is QuantityState.BOUNDED and self.interval is not None and self.interval.kind is IntervalKind.PROVEN:
            lower, upper = self.interval.lower, self.interval.upper
            return (lower is not None and lower > threshold) or (upper is not None and upper < threshold)
        if self.value is None:
            return False
        if self.state in VALUED_STATES or self.value_basis is not None:
            return decides_threshold(self.value, threshold, basis=self.value_basis, resolution=resolution)
        return False


class NotFollowed(StrictModel):
    """An item an enumeration did not follow to its end, and why; ``power_upper_W`` bounds its power when known."""

    item_id: str
    reason: str
    power_upper_W: float | None = Field(None, ge=0.0)


class Completeness(StrictModel):
    """What an enumeration covered. ``proven`` is true when ``unenumerated_power_upper_W`` is a proven bound on every item not
    listed."""

    statement: str
    enumerated: int = Field(ge=0)
    not_followed: tuple[NotFollowed, ...] = ()
    unenumerated_power_upper_W: float | None = Field(None, ge=0.0)
    proven: bool = False


class PhysicalIdentity(StrictModel):
    """What the numbers of an analysis depend on."""

    engine_build_id: str
    engine_input_hash: str | None = None
    run_result_hash: str | None = None
    settings_hash: str = Field(description="sha256 of the analysis's canonical settings.")


class RequirementIdentity(StrictModel):
    """The thresholds an evaluation of the analysis used."""

    targets_hash: str


class AnalysisIdentity(StrictModel):
    physical: PhysicalIdentity
    requirements: RequirementIdentity | None = None


class AnalysisResultV1(StrictModel):
    """One analysis document. See the module docstring."""

    schema_: str = Field(ANALYSIS_RESULT_SCHEMA, alias="schema")
    schema_version: str = ANALYSIS_RESULT_SCHEMA_VERSION
    analysis: str = Field(description="The analysis kind, e.g. IMAGING, FOCAL_FIELD, GHOST_PATHS, CONFIGURATIONS.")
    analysis_version: str
    identity: AnalysisIdentity
    execution: ExecutionRecord
    settings: dict[str, object] = Field(default_factory=dict, description="The effective settings, as hashed in settings_hash.")
    models: tuple[ModelBasis, ...] = ()
    quantities: tuple[Quantity, ...] = ()
    arrays: dict[str, object] = Field(default_factory=dict, description="Named arrays (curves, maps) that quantities reference.")
    completeness: Completeness | None = None
    issues: tuple[ValidationIssue, ...] = ()

    model_config = StrictModel.model_config | {"populate_by_name": True}

    @model_validator(mode="after")
    def _references(self) -> "AnalysisResultV1":
        model_ids = {m.model_id for m in self.models}
        seen: set[str] = set()
        for q in self.quantities:
            if q.quantity_id in seen:
                raise ValueError(f"duplicate quantity_id {q.quantity_id!r}")
            seen.add(q.quantity_id)
            if q.basis is not None and q.basis not in model_ids:
                raise ValueError(f"quantity {q.quantity_id!r} names an unknown model {q.basis!r}")
            if q.values_ref is not None and q.values_ref not in self.arrays:
                raise ValueError(f"quantity {q.quantity_id!r} names an unknown array {q.values_ref!r}")
        return self

    def quantity(self, quantity_id: str) -> Quantity:
        for q in self.quantities:
            if q.quantity_id == quantity_id:
                return q
        raise KeyError(quantity_id)
