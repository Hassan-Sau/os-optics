"""Analyses derived from a run or a stated optical request (``contracts.analysis.AnalysisResultV1``).

``document(...)`` assembles an analysis document with its identity, so every analysis states the same things in the same
places; the analysis modules compute the quantities.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from ..contracts.analysis import (
    AnalysisIdentity,
    AnalysisResultV1,
    Completeness,
    ModelBasis,
    PhysicalIdentity,
    Quantity,
    RequirementIdentity,
)
from ..contracts.outcome import BudgetUsage, ExecutionRecord
from ..errors import ValidationIssue
from ..errors.status import ExecutionOutcome, TerminationReason
from ..manifest import engine_build_id
from ..manifest.canonical import hash_exact

__all__ = ["completed", "document", "partial", "settings_hash"]


def settings_hash(settings: Mapping[str, Any]) -> str:
    """sha256 of an analysis's canonical settings (floats exact)."""

    return hash_exact(dict(settings))


def completed(budgets: BudgetUsage | None = None) -> ExecutionRecord:
    """The execution record of an analysis that ran to its end. What it states of each quantity is that quantity's own
    state and reason (and an enumeration's ``completeness``), never this record."""

    return ExecutionRecord(outcome=ExecutionOutcome.COMPLETED, termination=TerminationReason.COMPLETED, complete=True,
                           budgets=budgets or BudgetUsage())


def partial(termination: TerminationReason, reason: str, budgets: BudgetUsage | None = None) -> ExecutionRecord:
    """The execution record of an analysis that stopped early: a resource budget ran out, or a run it reads did not
    complete. What it computed stands. A quantity that is unknown or outside its model does not make an analysis partial."""

    return ExecutionRecord(outcome=ExecutionOutcome.PARTIAL, termination=termination, reason=reason, complete=False,
                           budgets=budgets or BudgetUsage())


def _identified_build() -> str:
    """The Engine build id, identified now (``manifest.engine_build_id``: it reads every source file of the Engine)."""

    return engine_build_id()


def document(
    analysis: str,
    analysis_version: str,
    settings: Mapping[str, Any],
    execution: ExecutionRecord,
    *,
    engine_input_hash: str | None = None,
    run_result_hash: str | None = None,
    targets_hash: str | None = None,
    models: Iterable[ModelBasis] = (),
    quantities: Iterable[Quantity] = (),
    arrays: Mapping[str, Any] | None = None,
    completeness: Completeness | None = None,
    issues: Iterable[ValidationIssue] = (),
    engine_build_id: str | None = None,
) -> AnalysisResultV1:
    """An analysis document bound to the Engine build, the run it reads and its own settings. ``engine_build_id``: the
    build id the caller already holds (an evaluation session's, the run's it reads); without one the build is identified
    here, which reads every source file of the Engine."""

    identity = AnalysisIdentity(
        physical=PhysicalIdentity(engine_build_id=engine_build_id if engine_build_id is not None else _identified_build(),
                                  engine_input_hash=engine_input_hash,
                                  run_result_hash=run_result_hash, settings_hash=settings_hash(settings)),
        requirements=RequirementIdentity(targets_hash=targets_hash) if targets_hash is not None else None,
    )
    return AnalysisResultV1(
        analysis=analysis, analysis_version=analysis_version, identity=identity, execution=execution,
        settings=dict(settings), models=tuple(models), quantities=tuple(quantities), arrays=dict(arrays or {}),
        completeness=completeness, issues=tuple(issues),
    )
