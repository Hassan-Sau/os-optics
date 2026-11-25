"""Target parameters: which ``parameters`` keys a target's metric takes, and the value type of each.

A target's ``parameters`` either select the beam(s) its metric reads -- the ``parameters.*`` entries of the metric's
``selectors`` (``validation/descriptors.py``) -- or carry a metric-specific number (``METRIC_PARAMETERS``). Every key a
target gives is applied; a key its metric does not take, or a value of the wrong type, is refused as input before any
physics runs, so a selector can never silently drop out and leave the target to pass on another beam:

* ``path_id`` / ``path_id_a`` / ``path_id_b``: a string (a beam path id such as ``"SRC#0"``);
* ``crossing_index`` / ``crossing_index_a`` / ``crossing_index_b``: an integer (a JSON number without a fractional
  part; a boolean, a string or a fraction is refused, never rounded). A crossing that does not occur selects no beam;
* ``dose_mj_per_cm2`` (``estimated_exposure_time_s``): a number greater than 0, and required (``REQUIRED_PARAMETERS``:
  a target on that metric without it is refused);
* ``axis`` (a retained DOF's limits, ``dof_*``): a string; ``required_min`` / ``required_max`` (``dof_travel_margin_*``) and
  ``temperature_span_K`` / ``duration_h`` (``dof_drift_*``): numbers (``validation/mechanical.py``);
* a two-beam target's ``crossing_index_a`` / ``crossing_index_b`` come together and only next to ``path_id_a`` and
  ``path_id_b`` (they number the crossings of those two paths; alone they would select nothing).

A metric that reads an observation plane (its descriptor lists the selector ``observation_id``: the single-beam, two-beam,
geometric pair and detection metrics) needs the target's ``observation_id`` (``requires_observation``); the Target
contract refuses a target on one without it.

Metrics without a descriptor (out-of-scope metrics) take no parameters; an unknown metric is reported as such and its
parameters are not checked here. A boolean is never a parameter value.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .descriptors import DESCRIPTOR_BY_NAME
from .mechanical import KIND_PARAMETERS, MECHANICAL_METRICS
from .scope import out_of_scope_metric

#: The value type of every parameter key a metric can take.
PARAMETER_TYPES: dict[str, str] = {
    "path_id": "string",
    "path_id_a": "string",
    "path_id_b": "string",
    "crossing_index": "integer",
    "crossing_index_a": "integer",
    "crossing_index_b": "integer",
    "dose_mj_per_cm2": "number",
    "axis": "string",
    "required_min": "number",
    "required_max": "number",
    "temperature_span_K": "number",
    "duration_h": "number",
}
#: Metric-specific parameters (not beam selectors): metric -> keys.
METRIC_PARAMETERS: dict[str, tuple[str, ...]] = {
    "estimated_exposure_time_s": ("dose_mj_per_cm2",),
    **{metric: KIND_PARAMETERS[kind] for metric, (_unit, kind, _motion) in MECHANICAL_METRICS.items()},
}
#: Parameters a target on a metric must give: metric -> keys.
REQUIRED_PARAMETERS: dict[str, tuple[str, ...]] = {"estimated_exposure_time_s": ("dose_mj_per_cm2",)}
#: Parameters whose value is greater than 0.
POSITIVE_PARAMETERS = frozenset({"dose_mj_per_cm2"})
#: The selector of a metric that reads an observation plane (``MetricDescriptor.selectors``).
OBSERVATION_SELECTOR = "observation_id"
#: The two-beam selection by paths and crossings (``crossing_index_a`` / ``_b`` apply only with all four).
PAIR_PATH_SELECTION = ("path_id_a", "path_id_b", "crossing_index_a", "crossing_index_b")
SELECTOR_PREFIX = "parameters."
#: A refused value is quoted in the message up to this many characters.
MAX_SHOWN_CHARS = 40
ELLIPSIS = "..."
VALUE_WORDS = {str: "a string", int: "an integer", float: "a number"}
TYPE_WORDS = {"string": "a string", "integer": "an integer (a JSON number without a fractional part)", "number": "a number"}


def accepted_parameters(metric: str) -> tuple[str, ...] | None:
    """The ``parameters`` keys ``metric`` takes, sorted (``()`` for a metric that takes none); ``None`` for a metric the
    Engine does not know (its parameters are not checked: the metric itself is reported)."""

    descriptor = DESCRIPTOR_BY_NAME.get(metric)
    if descriptor is None:
        return () if out_of_scope_metric(metric) is not None else None
    selectors = [s[len(SELECTOR_PREFIX):] for s in descriptor.selectors if s.startswith(SELECTOR_PREFIX)]
    return tuple(sorted({*selectors, *METRIC_PARAMETERS.get(metric, ())}))


def requires_observation(metric: str) -> bool:
    """Whether ``metric`` reads an observation plane, so that its target must name one (``observation_id``)."""

    descriptor = DESCRIPTOR_BY_NAME.get(metric)
    return descriptor is not None and OBSERVATION_SELECTOR in descriptor.selectors


def _type_ok(kind: str, value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if kind == "string":
        return isinstance(value, str)
    if not isinstance(value, (int, float)) or (isinstance(value, float) and not math.isfinite(value)):
        return False
    return kind == "number" or isinstance(value, int) or value.is_integer()


def _shown(value: Any) -> str:
    text = repr(value) if isinstance(value, str) else ("true" if value is True else "false" if value is False else str(value))
    return text if len(text) <= MAX_SHOWN_CHARS else text[:MAX_SHOWN_CHARS - len(ELLIPSIS)] + ELLIPSIS


def parameter_problems(metric: str, parameters: Mapping[str, Any]) -> list[tuple[str, str]]:
    """``(key, message)`` for every parameter of a target on ``metric`` that is not taken, not of its type or not in its
    range, for a parameter the metric needs and the target does not give, and for an incomplete two-beam path selection;
    empty when every parameter is applied as given."""

    accepted = accepted_parameters(metric)
    problems: list[tuple[str, str]] = []
    for key in sorted(parameters):
        value = parameters[key]
        if isinstance(value, bool):
            problems.append((key, f"parameter {key!r} is the boolean {_shown(value)}; a parameter value is a JSON number or string"))
            continue
        if accepted is not None and key not in accepted:
            takes = f"takes the parameters {list(accepted)}" if accepted else "takes no parameters"
            problems.append((key, f"metric {metric!r} {takes}, not {key!r} ('os-optics results metrics' lists each "
                                  "metric's selectors)"))
            continue
        kind = PARAMETER_TYPES.get(key)
        if kind is not None and not _type_ok(kind, value):
            problems.append((key, f"parameter {key!r} must be {TYPE_WORDS[kind]}, not {_shown(value)} "
                                  f"({VALUE_WORDS.get(type(value), type(value).__name__)})"))
        elif key in POSITIVE_PARAMETERS and not value > 0:
            problems.append((key, f"parameter {key!r} must be greater than 0, not {_shown(value)}"))
    for key in REQUIRED_PARAMETERS.get(metric, ()):
        if key not in parameters:
            problems.append((key, f"metric {metric!r} needs parameter {key!r}: add it to the target's parameters"))
    selection, crossings = set(PAIR_PATH_SELECTION), {"crossing_index_a", "crossing_index_b"} & set(parameters)
    if crossings and accepted is not None and selection <= set(accepted) and not selection <= set(parameters):
        problems.append((min(crossings), "crossing_index_a and crossing_index_b select crossings of the two paths: give both, "
                                         "together with path_id_a and path_id_b"))
    return problems


def check_target_parameters(metric: str, parameters: Any) -> None:
    """Raise ``ValueError`` naming every problem of ``parameters`` (``parameter_problems``); a non-mapping is left to the
    contract's own type check."""

    if not isinstance(parameters, Mapping):
        return
    problems = parameter_problems(metric, parameters)
    if problems:
        raise ValueError("; ".join(message for _key, message in problems))
