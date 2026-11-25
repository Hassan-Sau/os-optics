"""Targets on the stated limits of a retained degree of freedom: travel, drift and repeatability.

A component's actuators (``ActuatorSpec``; a Project's retained DOF lowers into one named by its axis, ``Tx`` ... ``Rz``,
when it states a travel) carry the limits of the stage that provides the adjustment, as the design states them. A target
compares those stated numbers with what the design needs; nothing here is computed from the trace, and the run's optics
do not depend on them.

* ``dof_travel_margin_mm`` / ``_deg``: ``min(travel_max - required_max, required_min - travel_min)`` over the required
  adjustment ``[parameters.required_min, parameters.required_max]`` (relative to the nominal pose; either end alone uses
  that side only). Zero or more: the required adjustment lies inside the travel; negative: it exceeds the travel by that
  much. A target ``minimum: 0`` asks "the adjustment the design needs stays inside the stage travel".
* ``dof_drift_mm`` / ``_deg``: ``drift_per_kelvin * parameters.temperature_span_K + drift_per_hour * parameters.duration_h``
  (the terms given), the drift magnitude over that span added linearly; a target ``maximum`` is the drift the design
  tolerates.
* ``dof_repeatability_mm`` / ``_deg``: the stated repeatability; a target ``maximum`` is the repeatability the design
  needs.

The target names the DOF by ``device_id`` (the component) and ``parameters.axis`` (the actuator id). ``_mm`` metrics apply
to translations, ``_deg`` metrics to rotations. A DOF that states no travel (no actuator), or no drift term or
repeatability the target reads, gives no value: ``UNKNOWN`` with ``MECHANICAL_LIMIT_NOT_DECLARED``, and the target is
``INDETERMINATE`` -- an adjustment without stated limits is not verified.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple

from ..contracts.components import ActuatorSpec
from ..contracts.project import Target
from ..errors import ErrorCode, Status

#: metric -> (unit, kind, actuator motion).
MECHANICAL_METRICS: dict[str, tuple[str, str, str]] = {
    "dof_travel_margin_mm": ("mm", "travel", "translate"),
    "dof_travel_margin_deg": ("deg", "travel", "rotate"),
    "dof_drift_mm": ("mm", "drift", "translate"),
    "dof_drift_deg": ("deg", "drift", "rotate"),
    "dof_repeatability_mm": ("mm", "repeatability", "translate"),
    "dof_repeatability_deg": ("deg", "repeatability", "rotate"),
}
#: The metric-specific ``parameters`` of each kind (``parameters.axis`` names the DOF of every one).
KIND_PARAMETERS: dict[str, tuple[str, ...]] = {
    "travel": ("required_min", "required_max"),
    "drift": ("temperature_span_K", "duration_h"),
    "repeatability": (),
}
MOTION_WORDS = {"translate": "translation", "rotate": "rotation"}


class DeclaredValue(NamedTuple):
    """A mechanical metric's value, unit, measurement status, codes and message (``metrics.MetricValue`` fields)."""

    value: float | None
    unit: str
    status: Status
    codes: tuple[str, ...] = ()
    message: str = ""


def _number(t: Target, key: str) -> float | None:
    value = t.parameters.get(key)  # (the Target contract: a number where given)
    return None if value is None else float(value)


def _undeclared(unit: str, message: str) -> DeclaredValue:
    return DeclaredValue(None, unit, Status.UNKNOWN, (ErrorCode.MECHANICAL_LIMIT_NOT_DECLARED.value,), message)


def mechanical_value(t: Target, actuators: Mapping[str, tuple[ActuatorSpec, ...]] | None) -> DeclaredValue:
    """The value of ``t`` (a ``MECHANICAL_METRICS`` target) from the actuators of its ``device_id`` (module docstring)."""

    unit, kind, motion = MECHANICAL_METRICS[t.metric]
    axis = t.parameters.get("axis")
    device = t.device_id
    if device is None or axis is None:
        return DeclaredValue(None, unit, Status.UNKNOWN, (ErrorCode.INVALID_INPUT.value,),
                             f"{t.metric} names a DOF by device_id and parameters.axis")
    spec = next((a for a in (actuators or {}).get(device, ()) if a.actuator_id == axis), None)
    dof = f"{device} {axis}"
    if spec is None:
        return _undeclared(unit, f"{dof} states no travel: the adjustment is not verified (state the DOF's travel)")
    if spec.dof != motion:
        return DeclaredValue(None, unit, Status.UNKNOWN, (ErrorCode.TARGET_SELECTION_EMPTY.value,),
                             f"{dof} is a {MOTION_WORDS[spec.dof]}; {t.metric} applies to a {MOTION_WORDS[motion]}")
    if kind == "travel":
        low, high = _number(t, "required_min"), _number(t, "required_max")
        margins = [m for m in ((spec.travel_max - high) if high is not None else None,
                               (low - spec.travel_min) if low is not None else None) if m is not None]
        if not margins:
            return DeclaredValue(None, unit, Status.UNKNOWN, (ErrorCode.INVALID_INPUT.value,),
                                 f"{t.metric} needs parameters.required_min and / or required_max")
        margin = min(margins)
        return DeclaredValue(margin, unit, Status.PASS, (), f"{dof} travel [{spec.travel_min:g}, {spec.travel_max:g}] {unit}: "
                             f"{margin:g} {unit} beyond the required adjustment")
    if kind == "drift":
        terms = ((spec.drift_per_kelvin, _number(t, "temperature_span_K"), "per kelvin"),
                 (spec.drift_per_hour, _number(t, "duration_h"), "per hour"))
        asked = [(rate, span, word) for rate, span, word in terms if span is not None]
        if not asked:
            return DeclaredValue(None, unit, Status.UNKNOWN, (ErrorCode.INVALID_INPUT.value,),
                                 f"{t.metric} needs parameters.temperature_span_K and / or duration_h")
        missing = [word for rate, _span, word in asked if rate is None]
        if missing:
            return _undeclared(unit, f"{dof} states no drift {' or '.join(missing)}: its drift over the span is not verified")
        drift = 0.0
        for rate, span, _word in asked:
            drift = drift + rate * abs(span)  # (stated magnitudes added linearly: the worst case of the stated terms)
        return DeclaredValue(drift, unit, Status.PASS, (), f"{dof} drifts by at most {drift:g} {unit} over the stated span")
    if spec.repeatability is None:
        return _undeclared(unit, f"{dof} states no repeatability: it is not verified")
    return DeclaredValue(spec.repeatability, unit, Status.PASS, (), f"{dof} repeatability +/- {spec.repeatability:g} {unit}")


__all__ = ["KIND_PARAMETERS", "MECHANICAL_METRICS", "DeclaredValue", "mechanical_value"]
