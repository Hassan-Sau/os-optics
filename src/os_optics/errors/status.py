"""Unified status vocabulary and aggregation.

Two kinds of word live here, and they answer different questions:

* ``Status`` -- the *diagnostic severity* of one finding (a target outcome, an issue, an observation). It is
  what ``worst`` aggregates.
* ``ExecutionOutcome``, ``TerminationReason``, ``RequirementAssessment`` and ``QuantityState`` -- the
  scientific-result contract (docs/engine/RESULTS_AND_EXPORT.md): did the calculation
  complete, *why* did it stop, is a declared customer target met, and is a quantity known.

The single aggregate ``Status`` a run reports (``ValidationResult.status``) is a **summary of diagnostic severity**, never
an approval of the design: the worst status over its targets, issues, observation planes and enabled stages. It may read
``WARNING`` or ``UNKNOWN`` beside a target that is ``MET`` (a dark plane's ``WARNING`` is the plane's, whatever a target asks
of it). A target's own result says what it decides, a quantity's own state whether its value is usable. A met target does
not make an invalid calculation valid, and an unmet, undecided or absent target does not make an independently calculated
quantity unusable.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum


class Status(StrEnum):
    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    OUT_OF_MODEL_SCOPE = "OUT_OF_MODEL_SCOPE"


# Severity for "worst of" aggregation. NOT_APPLICABLE never dominates: it means "does not apply".
_SEVERITY = {
    Status.NOT_APPLICABLE: -1,
    Status.PASS: 0,
    Status.WARNING: 1,
    Status.UNKNOWN: 2,
    Status.OUT_OF_MODEL_SCOPE: 3,
    Status.FAIL: 4,
}


def worst(statuses: Iterable[Status], *, empty: Status = Status.NOT_APPLICABLE) -> Status:
    """Worst status of a collection.

    An empty collection (or only NOT_APPLICABLE members) yields ``empty`` (NOT_APPLICABLE by
    default): nothing was checked, and that must never read as PASS.
    """
    relevant = [s for s in statuses if s is not Status.NOT_APPLICABLE]
    if not relevant:
        return empty
    return max(relevant, key=lambda s: _SEVERITY[s])


def severity(status: Status) -> int:
    return _SEVERITY[status]


# ------------------------------------------------------------------------------------- execution outcome


class ExecutionOutcome(StrEnum):
    """Did the requested calculation run to its end?

    ``COMPLETED``: every launched spectral line and every branch the model follows was traced to a known
    end, and every requested stage ran. ``PARTIAL``: the run produced results but a resource budget stopped part
    of the primary trace, so some quantities may be missing or bounded only. A budget an auxiliary analysis exhausted
    (the probes of untraced light, the return-light analysis, the pair listing) is listed in the run's budget record and
    on the quantities that lost a bound; the execution is still ``COMPLETED``. ``FAILED``:
    no usable result was produced (invalid input, a numerical failure, or a valid request outside the traced model,
    refused). ``CANCELLED``: the run was stopped before it completed -- by its caller, or by a memory or run-file
    limit the caller set (``contracts/outcome.py`` ``StoppedRun``) -- and has no result.

    This says nothing about whether a customer target is met: a design that misses every target still
    ``COMPLETED`` when the calculation is sound.
    """

    COMPLETED = "COMPLETED"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class TerminationReason(StrEnum):
    """Why the run stopped, in the run's own terms (structured; each issue says it in words).

    ``INVALID_INPUT``: the request is not valid. ``OUT_OF_MODEL_SCOPE``: the request is valid and states something the traced
    model does not cover (a facet surface): it is refused, with the issue naming the route that carries it.
    ``CANCELLED``: the run was stopped before it completed (``StoppedRun.stop`` says where, by its caller or by which
    resource limit)."""

    COMPLETED = "COMPLETED"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    NUMERICAL_FAILURE = "NUMERICAL_FAILURE"
    INVALID_INPUT = "INVALID_INPUT"
    OUT_OF_MODEL_SCOPE = "OUT_OF_MODEL_SCOPE"
    CANCELLED = "CANCELLED"


#: The resource budgets a run can exhaust (``NumericalConfig`` fields; ``ExecutionRecord.budgets``).
BUDGET_NAMES = (
    "max_interactions_per_path",
    "max_paths",
    "max_trace_steps",
    "max_interference_pairs",
    "max_return_analysis_work",
)


# ---------------------------------------------------------------------------------- requirement outcome


class RequirementAssessment(StrEnum):
    """The outcome of evaluating one *declared* customer target against the calculated quantities.

    ``MET`` / ``UNMET`` are decided either way. ``INDETERMINATE`` means the available evidence does not decide
    it: no value could be calculated, a supported interval straddles the threshold, or the light the trace
    does not follow is bounded by an interval that straddles it. ``NOT_APPLICABLE``: the target does not
    describe this light at all (a power rating whose conditions differ, say).

    A run with no targets has no requirement assessment at all -- that is not universal success.
    """

    MET = "MET"
    UNMET = "UNMET"
    INDETERMINATE = "INDETERMINATE"
    NOT_APPLICABLE = "NOT_APPLICABLE"


#: From a target's diagnostic ``Status`` to its requirement assessment. A WARNING target is one that is met
#: with a caveat (a bounded interval, a bounded contribution of left-out light); it is still met.
_ASSESSMENT = {
    Status.PASS: RequirementAssessment.MET,
    Status.WARNING: RequirementAssessment.MET,
    Status.FAIL: RequirementAssessment.UNMET,
    Status.UNKNOWN: RequirementAssessment.INDETERMINATE,
    Status.OUT_OF_MODEL_SCOPE: RequirementAssessment.INDETERMINATE,
    Status.NOT_APPLICABLE: RequirementAssessment.NOT_APPLICABLE,
}


def assessment_of(status: Status, applicability_codes: Iterable[str] = ()) -> RequirementAssessment:
    """The requirement assessment of a target graded ``status``. ``applicability_codes``: the model-scope findings that
    can change what the target's metric reads (a Local Validation record's ``applicability_issue_codes``); with any, the
    value is not one the model supports, so the requirement is ``INDETERMINATE`` whatever its status."""
    if tuple(applicability_codes):
        return RequirementAssessment.INDETERMINATE
    return _ASSESSMENT[status]


# ------------------------------------------------------------------------------------ quantity availability


class QuantityState(StrEnum):
    """Whether a scientific quantity at one scope (a plane's power, a surface's exposure, an analysis quantity) is known.

    ``KNOWN``: the model's value (it may be zero or non-zero). ``KNOWN_ZERO``: the model's value is exactly zero
    *and* no light reaches it above its lines' negligible level (``negligible_power_fraction`` of their input) -- distinct from a plane no traced beam happens to hit while
    light the trace does not follow may still arrive. Both mean that nothing above that level is left out: light left out
    below it is the stated truncation of the value and changes no state (its bound stands in the result's own fields).
    ``BOUNDED``: the value lies in a supported interval that is not a single number (a lit plane's power, or the power of a
    plane no traced beam reaches that only light the trace followed reaches under a supported bound, [0, bound]; the result
    states its ends, and an upper end nothing bounds is an explicit null with the cause in the quantity's ``value_basis``
    entry). ``UNKNOWN``: not known (part of the launched power was not followed, a budget stopped the trace, an input
    the quantity needs is missing, light the trace followed may reach a plane no traced beam reaches with no finite bound).
    ``OUT_OF_MODEL_SCOPE``: the quantity is not modelled
    here. ``NOT_APPLICABLE``: nothing to report at this scope (beam metrics on a plane no beam crosses).
    ``NOT_COMPUTED``: not computed, because the evaluation's scope or an option of the analysis request left it out. In an
    evaluation's result: the capability that produces the quantity was not in the executed scope
    (``ValidationResult.evaluation`` names the scope and the capability to request); the result of a complete evaluation
    never reports it. In an analysis document: an option of the analysis request left the quantity out (reason
    ``NOT_REQUESTED``; its message names the request field that turns it on).
    """

    KNOWN = "KNOWN"
    KNOWN_ZERO = "KNOWN_ZERO"
    BOUNDED = "BOUNDED"
    UNKNOWN = "UNKNOWN"
    OUT_OF_MODEL_SCOPE = "OUT_OF_MODEL_SCOPE"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    NOT_COMPUTED = "NOT_COMPUTED"


# ---------------------------------------------------------------------------------------------- value basis


class ValueKind(StrEnum):
    """How the model's true value of a quantity stands to the number reported for it (``contracts/basis.py``).

    The quantity's state word says what is proven about the quantity; this word says what the number in its field is.
    """

    #: The number is the model's value; the entry records only the conditions or caller settings under which that holds.
    MODEL = "MODEL"
    #: The model's true value is at least the number: what the number does not count can only add.
    LOWER_BOUND = "LOWER_BOUND"
    #: The model's true value is at most the number: what the number counts that may not all contribute can only lower it.
    UPPER_BOUND = "UPPER_BOUND"
    #: The stated model's number outside a validity limit or with a stand-in input; no proven relation to the true value.
    ESTIMATE = "ESTIMATE"


class OmittedPart(StrEnum):
    """A contribution a reported number does not count (``Omitted.part``).

    Two classes (``ACCOUNTING_PARTS``): an *accounting* part is light left out by a stated level -- the number is the
    model's value truncated there and the part changes no state or status, and decides no target whose limit lies inside
    its bound (such a target is INDETERMINATE: the truncation does not decide it); every other part is an *uncertainty*
    part -- what it would add is not known to be below a stated level.
    """

    #: Light below the caller's negligible level of its source line, not followed, probed or listed.
    BELOW_NEGLIGIBLE_LEVEL = "BELOW_NEGLIGIBLE_LEVEL"
    #: Light on paths ended at the absolute power floor.
    BELOW_POWER_FLOOR = "BELOW_POWER_FLOOR"
    #: Light a resource budget stopped before its end (paths, interactions, trace steps, a line not launched, an analysis's work).
    BUDGET = "BUDGET"
    #: Light on a path that ended where a surface, medium or coefficient is outside the traced model.
    OUT_OF_MODEL = "OUT_OF_MODEL"
    #: Light above the negligible level the trace does not follow as beams: ghost reflections, coating leakage, scatter,
    #: light passing beside an element, a beam's tail beside an active area.
    NOT_FOLLOWED = "NOT_FOLLOWED"
    #: Light that may come back between facing surfaces or around a closed path beyond the passes that were counted.
    POSSIBLE_RETURN = "POSSIBLE_RETURN"
    #: The part of a beam that leaves it beside its chief ray (met another surface first, taken before the plane, attenuated
    #: differently across the footprint) while the number applies the chief ray's power to the whole beam.
    BEAM_PART_ELSEWHERE = "BEAM_PART_ELSEWHERE"
    #: The accounting of left-out light was not computed for this value (the capability that bounds it did not run).
    NOT_ACCOUNTED = "NOT_ACCOUNTED"
    #: The effect of a hard edge on a beam's profile beyond the geometric or Gaussian truncation of its power.
    EDGE_DIFFRACTION = "EDGE_DIFFRACTION"
    #: The light of a source whose emitted power the model does not derive (a sample whose emission is not derived).
    SOURCE_NOT_DERIVED = "SOURCE_NOT_DERIVED"
    #: Power coupled into resonator modes above the highest mode order resolved.
    UNRESOLVED_MODES = "UNRESOLVED_MODES"


#: The accounting parts of ``OmittedPart``: left out by a stated level, they change no state or status; a target whose
#: limit lies inside their bound is not decided by the truncation (INDETERMINATE).
ACCOUNTING_PARTS = frozenset({OmittedPart.BELOW_NEGLIGIBLE_LEVEL, OmittedPart.BELOW_POWER_FLOOR})


class AssumedInput(StrEnum):
    """A stand-in a calculation used for an input the design or the data does not state (``Assumed.input``)."""

    #: A response (coefficient, retardance, rotation, index) used outside the angle or wavelength domain where it is stated,
    #: held at the value of the nearest stated point (a single design point included).
    HELD_AT_NEAREST_STATED_POINT = "HELD_AT_NEAREST_STATED_POINT"
    #: A dispersion or response law evaluated outside the range its data states.
    LAW_BEYOND_STATED_RANGE = "LAW_BEYOND_STATED_RANGE"
    #: A pass whose group delay the design does not state adds none.
    UNSTATED_GROUP_DELAY_ZERO = "UNSTATED_GROUP_DELAY_ZERO"
    #: A coating whose phase the data does not state is given the phases of the bare interface.
    UNSTATED_PHASE_AS_INTERFACE = "UNSTATED_PHASE_AS_INTERFACE"
    #: An element stated without a thickness adds the thin-element optical path (an ideal lens -h^2/2f at the chief ray's
    #: height, any other element none) and no substrate path.
    THIN_ELEMENT_PATH = "THIN_ELEMENT_PATH"
    #: A source that declares no coherence length is coherent at every path difference.
    FULL_COHERENCE = "FULL_COHERENCE"
    #: A loss the design does not state (the round-trip loss of an intracavity element, scatter) is zero.
    NO_UNSTATED_LOSS = "NO_UNSTATED_LOSS"
    #: A test condition a rating does not state equals the design's light and the Engine's own exposure definition.
    UNSTATED_CONDITION_AS_DESIGN = "UNSTATED_CONDITION_AS_DESIGN"
    #: The quantity is evaluated at a named operating point (``Assumed.value``) because the request states none.
    NOMINAL_OPERATING_POINT = "NOMINAL_OPERATING_POINT"
    #: Light whose profile is not known to be Gaussian (truncated, behind an obstruction) is a Gaussian beam of its second moments.
    GAUSSIAN_PROFILE = "GAUSSIAN_PROFILE"
    #: One ray (the chief ray, or the passing pupil ray that carries the beam's reference) stands for a beam that spreads around it.
    SINGLE_RAY_FOR_BEAM = "SINGLE_RAY_FOR_BEAM"
    #: Decentres and tilts accepted within their tolerance are taken as zero: the value is computed on the nominal axis.
    NOMINAL_AXIS = "NOMINAL_AXIS"
    #: An element's response model is used for a quantity its data does not state as validated (a tabulated thin lens, an
    #: assumed detector response).
    RESPONSE_MODEL_UNVALIDATED = "RESPONSE_MODEL_UNVALIDATED"
    #: The pupil is lit uniformly (a point object), not with the beam's own profile.
    UNIFORM_PUPIL = "UNIFORM_PUPIL"

