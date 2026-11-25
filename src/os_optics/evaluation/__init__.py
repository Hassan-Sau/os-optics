"""Composable evaluation: an evaluation is a plan over named capabilities.

A capability is a named part of an evaluation with one implementation (``capability_table()``: the scene, the beam trace, the
secondary rays, the plane readings, the power ledger, the untraced light, the plane bounds, the surface exposures, the
beam pairs, layout, control and the requirements; the local workflow adds its own). ``validate(request)`` runs every
capability the request enables. A selective evaluation names the ones it needs, and the plan adds what they require::

    from os_optics.evaluation import EvaluationState, evaluate, plan_evaluation

    plan = plan_evaluation(("readings",), request=request)
    plan.executed      # ('scene', 'beam_trace', 'secondary_rays', 'plane_readings')
    plan.added         # what the plan runs that was not named, each with what requires it

    state = EvaluationState()
    readings = evaluate(request, capabilities=("readings",), state=state)
    readings.evaluation.not_computed          # the capabilities that did not run, with the sections each produces
    complete = evaluate(request, state=state)  # runs what the readings did not; equal to validate(request)
    state.executions                           # {'scene': 1, 'beam_trace': 1, ...}: each capability ran once

* ``plan_evaluation(requested, exclude=(), targets=None, request=None)`` adds what the requested capabilities require and
  what each selected target's metric reads (``MetricDescriptor.requires``), and refuses a plan an exclusion makes
  impossible (``EvaluationPlanError``): nothing excluded is run.
* ``evaluate(request, capabilities=("complete",), exclude=(), targets=None, state=None, recorder=None)`` runs the plan and
  returns a ``ValidationResult``. A capability outside the executed scope is not run, and what it would have produced is
  not computed: ``result.evaluation`` names it and a plane whose bounds did not run reports ``quantity_state``
  ``NOT_COMPUTED``. Every target the evaluation is asked to assess has a result, and ``outcome.requirements`` alone says
  how the requirements came out.
* ``EvaluationState`` keeps each executed capability's product for the next evaluation of the same physical inputs,
  numerical configuration and Engine build; a changed target threshold reuses every one of them.
"""

from typing import Any

from .capabilities import (
    ANALYSES,
    CAPABILITIES,
    COMPLETE,
    ENGINE,
    LOCAL,
    PRESETS,
    READINGS,
    Capability,
    analyses,
    capability,
    capability_ids,
    capability_table,
)
from .plan import (
    AddedCapability,
    EvaluationPlan,
    EvaluationPlanError,
    TargetRequirement,
    enabled_capabilities,
    plan_evaluation,
)
from .state import EvaluationState, PhysicalIdentity

__all__ = [
    "ANALYSES",
    "CAPABILITIES",
    "COMPLETE",
    "ENGINE",
    "LOCAL",
    "PRESETS",
    "READINGS",
    "AddedCapability",
    "Capability",
    "EvaluationPlan",
    "EvaluationPlanError",
    "EvaluationState",
    "PhysicalIdentity",
    "TargetRequirement",
    "analyses",
    "capability",
    "capability_ids",
    "capability_table",
    "enabled_capabilities",
    "evaluate",
    "plan_evaluation",
]


def __getattr__(name: str) -> Any:
    # (``evaluate`` lives with the capabilities' nodes in ``os_optics.validation.pipeline``, which reads this package's
    # table: it is bound here on first use)
    if name == "evaluate":
        from ..validation.pipeline import evaluate

        return evaluate
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
