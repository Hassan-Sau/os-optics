"""ValidationIssue: the structured error / warning record (public contract)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .codes import ErrorCode
from .status import Status

ContextValue = float | int | str | bool | None | list[float] | list[int] | list[str]


class ValidationIssue(BaseModel):
    """A structured, machine-readable finding.

    ``status`` says how this finding counts in aggregation (WARNING, FAIL, UNKNOWN,
    OUT_OF_MODEL_SCOPE, NOT_APPLICABLE). PASS is never an issue. ``NOT_APPLICABLE`` marks a notice that does not count
    toward the run's status: a report that holds whatever the design's targets are (an exceeded power rating,
    ``POWER_RATING_EXCEEDED``, which only a ``power_rating_utilization`` target turns into a result; a plane known to
    receive no light; two beams that do not interfere).

    Text contract: ``message`` states the finding (what was found, where, and the numbers behind it); ``suggested_action`` is
    a step inside the tool that obtains the missing value or decision -- a budget or setting to raise, a capability to add,
    a field to read, a model statement to make -- and is absent where no such step exists. The action is never part of the
    message, and a design change is not an action.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    code: ErrorCode
    status: Status = Field(description="How this issue counts in aggregation.")
    message: str = Field(description="The finding: what was found, where, and the numbers behind it; never the action.")
    location: str | None = Field(None, description="Object path, e.g. 'component:M1/surface:front'.")
    affected_targets: tuple[str, ...] = Field(
        (), description="The targets this issue leaves UNKNOWN or OUT_OF_MODEL_SCOPE, in Project order.")
    numerical_context: dict[str, ContextValue] = Field(default_factory=dict)
    suggested_action: str | None = Field(
        None, description="A step inside the tool that obtains the missing value or decision (a budget or setting to raise, a "
                          "capability to add, a field to read, a model statement to make); absent where none exists.")


def make_issue(
    code: ErrorCode,
    status: Status,
    message: str,
    *,
    location: str | None = None,
    affected_targets: tuple[str, ...] = (),
    suggested_action: str | None = None,
    **numerical_context: Any,
) -> ValidationIssue:
    """Build an issue, coercing numpy values and non-finite numbers into JSON-safe context values."""
    ctx: dict[str, ContextValue] = {key: _coerce(value) for key, value in numerical_context.items()}
    return ValidationIssue(
        code=code,
        status=status,
        message=message,
        location=location,
        affected_targets=affected_targets,
        numerical_context=ctx,
        suggested_action=suggested_action,
    )


def notice(
    code: ErrorCode,
    message: str,
    *,
    location: str,
    suggested_action: str | None = None,
    **numerical_context: Any,
) -> ValidationIssue:
    """A finding whose whole content a quantity carries (its ``value_basis`` entry or a flat field), issued as a notice.

    Its status is ``NOT_APPLICABLE``: it does not count toward any aggregate status (a plane's, the run's) or the exit code.
    ``location`` is the quantity it belongs to, in the issue-location grammar (``observation:<id>``, ``surface:<id>``,
    ``target:<id>``, ...); the basis entry that states the same fact names ``code`` (``Omitted.code``, ``Assumed.code``,
    ``ValidityCheck.code``), the key a reader joins the two on. It names no target in ``affected_targets``: a target's own
    result says what the quantity does to it."""

    if not location:
        raise ValueError("a notice is located at the quantity that carries its content")
    return make_issue(code, Status.NOT_APPLICABLE, message, location=location, suggested_action=suggested_action,
                      **numerical_context)


def _coerce(value: Any) -> ContextValue:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)  # never NaN/Inf in JSON
    if isinstance(value, (bool, int, str)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_coerce(v) for v in value]  # type: ignore[misc]
    return str(value)
