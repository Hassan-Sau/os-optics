"""The value basis of a reported number: what the number is, and what it does not count, assume or hold within.

A value field that is null because a proof is missing, a validity limit is exceeded or an input is unstated becomes a
number only together with a ``value_basis`` entry that names it. The entry says, in closed vocabularies and numbers only:

* ``kind`` (``ValueKind``) -- how the model's true value stands to the number: ``MODEL`` (it is the model's value),
  ``LOWER_BOUND`` / ``UPPER_BOUND`` (the true value is at least / at most it), ``ESTIMATE`` (no proven relation);
* ``omitted`` -- contributions the number does not count (``OmittedPart``), with the proven most each can add (``upper``)
  where one is known;
* ``assumed`` -- stand-ins for inputs the design or the data does not state (``AssumedInput``);
* ``validity`` -- the validity conditions checked for this value (``ValidityCheck``).

``setting`` names the caller setting that holds a limit or level (a key of the run's settings record), ``at`` where the
fact applies (issue-location grammar or bare surface / element ids), ``code`` the ``ErrorCode`` of the issue stating the
same fact. No basis type holds prose.

A carrier holds ``value_basis: {field name: ValueBasis}`` (``"*"``: every value field of the record), omitted when
empty, so a result with nothing to qualify keeps its bytes. The state word beside a quantity (``QuantityState``, a
status) keeps saying what is proven; the entry says what the number is. Read a number as the model's value only when no
entry names it or its kind is ``MODEL`` (``os_optics.results.basis_of`` also derives entries from the flat fields of a
result). An entry feeds nothing: no aggregate status, no exit code, no execution outcome.

``grade_point`` is the one decision rule of a target on a point value (rule T1): a ``LOWER_BOUND`` decides a minimum it
meets and a maximum it exceeds, an ``UPPER_BOUND`` the mirror image, an ``ESTIMATE`` nothing, and a number whose
measurement is ``UNKNOWN`` / ``OUT_OF_MODEL_SCOPE`` decides nothing either.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, Literal

from pydantic import BeforeValidator, Field, model_validator

from ..errors.status import (
    ACCOUNTING_PARTS,
    AssumedInput,
    OmittedPart,
    QuantityState,
    Status,
    ValueKind,
    worst,
)
from .base import OmitAbsentFields, StrictModel

#: The carrier field that holds the entries.
BASIS_FIELD = "value_basis"
#: The key of an entry that qualifies every value field of its record (numbers that stand or fall together).
WHOLE_RECORD = "*"


class ValidityCheck(OmitAbsentFields, StrictModel):
    """One condition of a model's validity, checked against the case: ``observed`` within ``[minimum, maximum]``.

    ``setting``: the caller setting that holds the limit (a key of the run's settings record), present where the case is
    outside the limit or the limit in force is not the Engine default. ``at``: where the case was checked. ``code``: the
    ``ErrorCode`` of the issue that states the same fact."""

    absent_omitted: ClassVar[tuple[str, ...]] = ("setting", "at", "code")

    parameter: str
    unit: str | None = None
    minimum: float | None = None
    maximum: float | None = None
    observed: float | None = None
    within: bool | None = Field(None, description="True inside the range, false outside, null when it could not be evaluated.")
    setting: str | None = None
    at: tuple[str, ...] = ()
    code: str | None = None


class Omitted(OmitAbsentFields, StrictModel):
    """A contribution the number does not count. ``upper``: the proven most this part can move the true value away from the
    number, in the number's unit (absent: no proven bound). ``setting``: the caller setting that decided it is left out (a
    negligible level, a budget)."""

    absent_omitted: ClassVar[tuple[str, ...]] = ("at", "upper", "setting", "code")

    part: OmittedPart
    at: tuple[str, ...] = ()
    upper: float | None = Field(None, ge=0.0)
    setting: str | None = None
    code: str | None = None

    @property
    def accounting(self) -> bool:
        """Left out by a stated level: changes no state or status; a target whose limit lies inside its bound is not
        decided by the truncation (INDETERMINATE)."""

        return self.part in ACCOUNTING_PARTS


class Assumed(OmitAbsentFields, StrictModel):
    """A stand-in the calculation used for an input the design or the data does not state. ``value`` / ``unit``: the
    stand-in used, when it is a number or a named form; ``at``: the passes, surfaces, elements or conditions it applies to."""

    absent_omitted: ClassVar[tuple[str, ...]] = ("at", "value", "unit", "code")

    input: AssumedInput
    at: tuple[str, ...] = ()
    value: float | str | None = None
    unit: str | None = None
    code: str | None = None


class ValueBasis(OmitAbsentFields, StrictModel):
    """What a reported number is (``kind``) and the facts it stands on. At least one of the three lists is not empty; a
    ``MODEL`` entry holds only checks that are within their range and omissions of accounting parts (light left out by a
    stated level), and no assumption."""

    absent_omitted: ClassVar[tuple[str, ...]] = ("omitted", "assumed", "validity")

    kind: ValueKind
    omitted: tuple[Omitted, ...] = ()
    assumed: tuple[Assumed, ...] = ()
    validity: tuple[ValidityCheck, ...] = ()

    @model_validator(mode="after")
    def _states_something(self) -> "ValueBasis":
        if not (self.omitted or self.assumed or self.validity):
            raise ValueError("a value basis states at least one omission, assumption or validity check")
        if self.kind is ValueKind.MODEL:
            if any(check.within is not True for check in self.validity):
                raise ValueError("a MODEL value basis holds only validity checks that are within their range")
            if any(not part.accounting for part in self.omitted):
                raise ValueError("a MODEL value basis omits only accounting parts (light left out by a stated level)")
            if self.assumed:
                raise ValueError("a MODEL value basis holds no assumption")
        return self

    @property
    def decides_as_model(self) -> bool:
        """The number may be read as the model's value (the allow-list rule: ``MODEL`` only)."""

        return self.kind is ValueKind.MODEL

    @property
    def out_of_validity(self) -> bool:
        """A validity check of this value failed (``within`` false)."""

        return any(check.within is False for check in self.validity)

    def settings(self) -> tuple[str, ...]:
        """Every caller setting the entry names, in order of appearance (each a key of the run's settings record)."""

        names = [item.setting for item in (*self.omitted, *self.validity) if item.setting is not None]
        return tuple(dict.fromkeys(names))


ValueBases = dict[str, ValueBasis]

# --------------------------------------------------------------------------------------------- carrier rules

_NO_ENTRY_STATES = frozenset({QuantityState.KNOWN_ZERO, QuantityState.NOT_APPLICABLE})


def _is_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, (tuple, list)) and value:
        return all(_is_number(v) for v in value)
    return False


def resolve(record: Any, key: str) -> Any:
    """The value a basis key names on ``record``: a field name, or a dotted path into a nested record (``beam.power_W``).
    ``KeyError`` when the key names no field."""

    value = record
    for name in key.split("."):
        fields = getattr(type(value), "model_fields", None)
        if fields is None or name not in fields:
            raise KeyError(key)
        value = getattr(value, name)
    return value


def normalize_bases(value: Any) -> Any:
    """``value_basis`` as stored: an empty mapping is no entry (``None``, omitted from the JSON form)."""

    if value is None or (isinstance(value, Mapping) and not value):
        return None
    return value


def check_carrier(record: Any, bases: Mapping[str, ValueBasis] | None, *, state: QuantityState | None = None,
                  state_keys: Iterable[str] = ()) -> None:
    """The carrier rules of a record's ``value_basis``: every key is ``"*"`` or names a value field of the record that holds a
    number; a quantity in state ``KNOWN_ZERO`` / ``NOT_APPLICABLE`` carries no entry and one in state ``KNOWN`` only a
    ``MODEL`` entry (``state`` is the state word of the fields ``state_keys`` and of ``"*"``). ``ValueError`` otherwise."""

    if not bases:
        return
    governed = {WHOLE_RECORD, *state_keys}
    for key, entry in bases.items():
        if key != WHOLE_RECORD:
            try:
                value = resolve(record, key)
            except KeyError:
                raise ValueError(f"value_basis key {key!r} names no field of {type(record).__name__}") from None
            if value is None:
                raise ValueError(f"value_basis key {key!r} names a field without a value")
            if not _is_number(value):
                raise ValueError(f"value_basis key {key!r} names a field that holds no number")
        if state is not None and key in governed:
            check_state(state, entry, key)


def check_state(state: QuantityState, entry: ValueBasis | None, key: str = "value") -> None:
    """A ``KNOWN_ZERO`` / ``NOT_APPLICABLE`` quantity carries no entry, a ``KNOWN`` one only a ``MODEL`` entry."""

    if entry is None:
        return
    if state in _NO_ENTRY_STATES:
        raise ValueError(f"a {state.value} quantity carries no value basis ({key!r})")
    if state is QuantityState.KNOWN and entry.kind is not ValueKind.MODEL:
        raise ValueError(f"a KNOWN quantity carries only a MODEL value basis ({key!r} is {entry.kind.value})")


#: The type of a carrier's ``value_basis`` field: entries keyed by the value field they qualify (``"*"``: every value field),
#: ``None`` (omitted from the JSON form) when there is none.
ValueBasisField = Annotated[
    dict[str, ValueBasis] | None,
    BeforeValidator(normalize_bases),
    Field(description="What a number of this record is where it is not simply the model's value: entries keyed by the "
          "name of the value field they qualify ('*': every value field of the record), each stating its kind (MODEL, "
          "LOWER_BOUND, UPPER_BOUND, ESTIMATE) and the contributions omitted, inputs assumed and validity checks it stands "
          "on. Read a number as the model's value only when no entry names it or its kind is MODEL. Omitted when empty."),
]


class BasisCarrier:
    """Mixin of a result record with a ``value_basis`` field (declare it as ``value_basis: ValueBasisField = None`` and list
    it in the record's omitted-when-absent fields). ``basis_state_field`` names the record's ``QuantityState`` field, which
    governs the keys ``basis_state_keys`` and ``"*"`` (``check_carrier``)."""

    basis_state_field: ClassVar[str | None] = None
    basis_state_keys: ClassVar[tuple[str, ...]] = ()

    @model_validator(mode="after")
    def _value_basis_rules(self) -> Any:
        bases = getattr(self, BASIS_FIELD, None)
        if bases:
            state = getattr(self, self.basis_state_field) if self.basis_state_field else None
            check_carrier(self, bases, state=state, state_keys=self.basis_state_keys)
        return self


# ----------------------------------------------------------------------------------------------- derivation


def _union(items: Iterable[Any]) -> tuple[Any, ...]:
    out: list[Any] = []
    for item in items:
        if item not in out:
            out.append(item)
    return tuple(out)


_FLIP = {ValueKind.LOWER_BOUND: ValueKind.UPPER_BOUND, ValueKind.UPPER_BOUND: ValueKind.LOWER_BOUND}


def merge_bases(*bases: ValueBasis | None, monotone: Literal["increasing", "decreasing"] | None = "increasing") -> ValueBasis | None:
    """The entry of a quantity derived from inputs whose entries are ``bases`` (``None``: an input with no entry, the model's
    value): the union of their lists, and the kind

    * ``MODEL`` when every qualified input is ``MODEL``;
    * the inputs' bound kind when the derived value is monotone in every qualified input (``monotone="increasing"``: it
      keeps the kind; ``"decreasing"``: a lower bound becomes an upper bound and the reverse) and their bound kinds agree;
    * ``ESTIMATE`` otherwise (an ``ESTIMATE`` input, kinds that disagree, ``monotone=None``).

    ``None`` when no input has an entry. A producer calls it with the entries of exactly the inputs it reads."""

    present = [b for b in bases if b is not None]
    if not present:
        return None
    omitted = _union(o for b in present for o in b.omitted)
    assumed = _union(a for b in present for a in b.assumed)
    validity = _union(v for b in present for v in b.validity)
    bounds = {b.kind for b in present if b.kind is not ValueKind.MODEL}
    if not bounds:
        kind = ValueKind.MODEL
    elif monotone is None or ValueKind.ESTIMATE in bounds or len(bounds) > 1:
        kind = ValueKind.ESTIMATE
    else:
        kind = bounds.pop()
        if monotone == "decreasing":
            kind = _FLIP[kind]
    return ValueBasis(kind=kind, omitted=omitted, assumed=assumed, validity=validity)


# ------------------------------------------------------------------------------------- the decision rule T1


@dataclass(frozen=True)
class Requirement:
    """The values that meet a target: at least ``minimum``, at most ``maximum``, within ``tolerance`` of ``target_value``."""

    minimum: float | None = None
    maximum: float | None = None
    target_value: float | None = None
    tolerance: float | None = None

    def _off_target(self, value: float) -> bool:
        return self.target_value is not None and self.tolerance is not None and abs(value - self.target_value) > self.tolerance

    def below(self, value: float) -> bool:
        """``value`` is under every value that meets the target."""

        return ((self.minimum is not None and value < self.minimum)
                or (self.target_value is not None and value < self.target_value and self._off_target(value)))

    def above(self, value: float) -> bool:
        """``value`` is over every value that meets the target."""

        return ((self.maximum is not None and value > self.maximum)
                or (self.target_value is not None and value > self.target_value and self._off_target(value)))

    def meets(self, value: float) -> bool:
        return not self.below(value) and not self.above(value)

    @property
    def open_above(self) -> bool:
        """Every value above one that meets the target meets it too (no maximum, no target value)."""

        return self.maximum is None and not (self.target_value is not None and self.tolerance is not None)

    @property
    def open_below(self) -> bool:
        """Every value below one that meets the target meets it too (no minimum, no target value)."""

        return self.minimum is None and not (self.target_value is not None and self.tolerance is not None)


#: Measurement states under which a number decides nothing: the Engine itself does not call it known.
UNDECIDING_MEASUREMENTS = frozenset({Status.UNKNOWN, Status.OUT_OF_MODEL_SCOPE})


def undecided_status(basis: ValueBasis | None, measured: Status = Status.UNKNOWN) -> Status:
    """The status of a target its value does not decide: ``OUT_OF_MODEL_SCOPE`` when the value is outside a validity limit
    (or its measurement is), ``UNKNOWN`` otherwise."""

    if measured is Status.OUT_OF_MODEL_SCOPE or (basis is not None and basis.out_of_validity):
        return Status.OUT_OF_MODEL_SCOPE
    return Status.UNKNOWN


def grade_point(value: float, requirement: Requirement, *, basis: ValueBasis | None = None,
                measured: Status = Status.PASS) -> Status:
    """The status of a target graded on the point value ``value`` (rule T1). ``measured`` is the status of the measurement
    before the comparison, ``basis`` the value's entry.

    * the measurement is ``UNKNOWN`` / ``OUT_OF_MODEL_SCOPE``: undecided, that status (never ``FAIL``: the Engine does
      not call the number known);
    * no entry, or a ``MODEL`` one: met -> the measurement's status (``PASS`` when it has none), missed -> ``FAIL``;
    * ``LOWER_BOUND``: a value over every met value -> ``FAIL``; a met value of a target with no upper end -> ``WARNING``
      (met with a caveat); otherwise undecided (``UNKNOWN``). ``UPPER_BOUND``: the mirror image;
    * ``ESTIMATE``: undecided, ``OUT_OF_MODEL_SCOPE`` when a validity check failed, else ``UNKNOWN``.

    A proven interval (both ends known) is graded by its ends, not here. Its assessment follows from the status
    (``assessment_of``)."""

    if measured in UNDECIDING_MEASUREMENTS:
        return measured
    kind = ValueKind.MODEL if basis is None else basis.kind
    if kind is ValueKind.MODEL:
        return worst([measured] + ([] if requirement.meets(value) else [Status.FAIL]), empty=Status.PASS)
    if kind is ValueKind.LOWER_BOUND:
        if requirement.above(value):
            return Status.FAIL
        if requirement.open_above and requirement.meets(value):
            return worst([measured, Status.WARNING], empty=Status.WARNING)
        return undecided_status(basis)
    if kind is ValueKind.UPPER_BOUND:
        if requirement.below(value):
            return Status.FAIL
        if requirement.open_below and requirement.meets(value):
            return worst([measured, Status.WARNING], empty=Status.WARNING)
        return undecided_status(basis)
    return undecided_status(basis)


def decides_threshold(value: float, threshold: float, *, basis: ValueBasis | None = None, resolution: float = 0.0) -> bool:
    """Whether ``value`` decides a comparison with ``threshold``: a model value outside ``resolution`` of it; a lower bound
    above it; an upper bound below it; an estimate never."""

    if not math.isfinite(value):
        return False
    kind = ValueKind.MODEL if basis is None else basis.kind
    if kind is ValueKind.MODEL:
        return abs(value - threshold) > resolution
    if kind is ValueKind.LOWER_BOUND:
        return value - resolution > threshold
    if kind is ValueKind.UPPER_BOUND:
        return value + resolution < threshold
    return False


__all__ = [
    "ACCOUNTING_PARTS",
    "Assumed",
    "AssumedInput",
    "BASIS_FIELD",
    "BasisCarrier",
    "ValueBasisField",
    "Omitted",
    "OmittedPart",
    "Requirement",
    "UNDECIDING_MEASUREMENTS",
    "ValidityCheck",
    "ValueBases",
    "ValueBasis",
    "ValueKind",
    "WHOLE_RECORD",
    "check_carrier",
    "check_state",
    "decides_threshold",
    "grade_point",
    "merge_bases",
    "normalize_bases",
    "resolve",
    "undecided_status",
]
