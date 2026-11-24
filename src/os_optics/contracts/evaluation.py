"""The scope of an evaluation: what was requested, what ran, and what was not computed (``ValidationResult.evaluation``).

An evaluation runs a plan over named capabilities (``os_optics.evaluation``). A complete evaluation -- every capability the
request's configuration enables, every declared target assessed -- carries no scope record: its result is ``validate``'s.
A narrowed one states its scope here, so a reader never takes an empty section for "nothing found" or a missing bound for
"complete". Every target the evaluation was asked to assess has a ``TargetResult`` and its place in
``outcome.requirements``; the scope names only the declared targets the caller did not select.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from .base import StrictModel

#: Why a capability of the request's complete scope did not run: the caller did not request it (and nothing requested needs
#: it), or the caller excluded it.
NOT_REQUESTED = "NOT_REQUESTED"
EXCLUDED = "EXCLUDED"


class NotComputed(StrictModel):
    """One capability of the request's complete scope that this evaluation did not run, and what it would have produced."""

    capability: str = Field(description="The capability id (os_optics.evaluation.CAPABILITIES).")
    reason: Literal["NOT_REQUESTED", "EXCLUDED"] = Field(
        description="NOT_REQUESTED: not among the requested capabilities, and none of them requires it. EXCLUDED: the caller "
        "excluded it.")
    produces: tuple[str, ...] = Field(
        (), description="The result sections and fields it produces: each is absent, empty, null or false in this result, "
        "or carries the state NOT_COMPUTED, because the capability did not run. A list, a count or a hash that a "
        "capability which ran produces as well (the beam paths, their count and the trace hash, which the beam trace "
        "and the secondary rays both produce) holds the part of the capability that ran only.")


class TargetNotEvaluated(StrictModel):
    """One declared target the caller did not ask this evaluation to assess. It has no ``TargetResult``, no status and no
    assessment, takes no part in the result's ``status`` and is in no list of ``outcome.requirements``."""

    target_id: str
    metric: str
    reason: Literal["NOT_REQUESTED"] = Field(
        description="NOT_REQUESTED: the requirements were not requested, or this target is not among the targets selected.")


class EvaluationScope(StrictModel):
    """What a narrowed evaluation computed. ``requested``: the capabilities the caller of the Engine evaluation
    (``os_optics.evaluate``) asked for, ids and presets as given; a Project evaluation (``os_optics.local``) asks the
    Engine for the Engine capabilities of its plan by id, and the local caller's own selection is ``Evaluation.scope`` /
    ``Evaluation.describe_scope()``. ``executed``: those and the capabilities they require, in execution order;
    ``not_computed``: the capabilities of the request's complete scope that did not run; ``targets_not_evaluated``: the
    declared targets the caller did not select."""

    requested: tuple[str, ...] = ()
    executed: tuple[str, ...] = ()
    not_computed: tuple[NotComputed, ...] = ()
    targets_not_evaluated: tuple[TargetNotEvaluated, ...] = ()


__all__ = ["EXCLUDED", "NOT_REQUESTED", "EvaluationScope", "NotComputed", "TargetNotEvaluated"]
