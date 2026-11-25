"""Model kinds the Project format does not have, refused before the Project is read with what serves the request.

A declared optical model ``model_kind`` of ``CHOPPER`` or ``LOCK_IN`` names a modulation and a lock-in, which a Project
declares in an observation plane's ``detection``, not as optical elements (``validation.scope.MODEL_SCOPE_BOUNDARY``).
Instead of the generic "unknown model kind" of the Project schema, each is refused as ``SIMULATION_MODEL_UNSUPPORTED`` with
``details``: ``boundary_item``, ``disposition``, ``missing_model`` (empty for a supported item) and ``route`` (what serves
the request). Any other unknown kind keeps the schema's own refusal.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..validation.scope import BOUNDARY_BY_ITEM, IN_SCOPE_SUPPORTED, OUT_OF_SCOPE_OPTICAL_MODEL_KINDS, OUT_OF_SCOPE_SOURCE_KINDS
from .errors import LocalRunErrorCode, LocalRunProblem


def _kind(node: Any) -> str | None:
    kind = node.get("model_kind") if isinstance(node, Mapping) else None
    return kind.strip().upper() if isinstance(kind, str) else None


def _named(item: Any, key: str, kind: str, position: int) -> str:
    name = item.get(key) if isinstance(item, Mapping) else None
    return f"{kind} {name}" if isinstance(name, str) and name else f"{kind} {position + 1}"


def _problem(what: str, model_kind: str, item: str, location: str) -> LocalRunProblem:
    boundary = BOUNDARY_BY_ITEM[item]
    if boundary["disposition"] == IN_SCOPE_SUPPORTED:
        message = f"{what} model_kind {model_kind} is not a kind of the Project format: it is declared as {boundary['route']}"
    else:
        message = (f"{what} model_kind {model_kind} is outside the model's scope: it needs {boundary['missing_model']}, which the "
                   f"Engine does not model. Route: {boundary['route']}")
    return LocalRunProblem(LocalRunErrorCode.SIMULATION_MODEL_UNSUPPORTED, message, location,
                           {"model_kind": model_kind, "boundary_item": item, "disposition": boundary["disposition"],
                            "missing_model": boundary["missing_model"], "route": boundary["route"]})


def out_of_scope_kind_problems(document: Any) -> list[LocalRunProblem]:
    """One ``SIMULATION_MODEL_UNSUPPORTED`` problem per source or declared optical model of ``document`` (a Project as a
    JSON-shaped mapping) whose ``model_kind`` names an out-of-scope model; empty for anything else."""

    if not isinstance(document, Mapping):
        return []
    problems = []
    components = document.get("components")
    for position, component in enumerate(components if isinstance(components, list) else ()):
        source = component.get("source_model") if isinstance(component, Mapping) else None
        kind = _kind(source)
        if kind in OUT_OF_SCOPE_SOURCE_KINDS:
            problems.append(_problem("source", kind, OUT_OF_SCOPE_SOURCE_KINDS[kind], _named(component, "instance_id", "component", position)))
    models = document.get("optical_models")
    for position, declared in enumerate(models if isinstance(models, list) else ()):
        kind = _kind(declared.get("model") if isinstance(declared, Mapping) else None)
        if kind in OUT_OF_SCOPE_OPTICAL_MODEL_KINDS:
            problems.append(_problem("optical model", kind, OUT_OF_SCOPE_OPTICAL_MODEL_KINDS[kind],
                                     _named(declared, "model_id", "optical model", position)))
    return problems


__all__ = ["out_of_scope_kind_problems"]
