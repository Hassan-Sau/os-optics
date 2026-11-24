"""Exceptions of the Engine request: an input that violates its contract, and a valid request the model does not take.

Physical and numerical findings are never raised: they are returned as ``ValidationIssue`` records at the item they
concern. Which refusals end a whole operation and which are answered per item is recorded in ``errors/registry.py``
(every code vocabulary and exception class of the Engine, with the refusals still to be converted).
"""

from __future__ import annotations

from .codes import ErrorCode


class EngineInputError(ValueError):
    """The request violates its contract (malformed geometry, duplicate ids, unknown references): it is not evaluated."""

    def __init__(self, code: ErrorCode, message: str, *, location: str | None = None, **context: object) -> None:
        super().__init__(f"{code}: {message}" + (f" [{location}]" if location else ""))
        self.code = code
        self.message = message
        self.location = location
        self.context = context


class ModelScopeRefusal(EngineInputError):
    """The request is valid but states something outside the traced model for which the trace has no per-item form yet (a
    facet surface the trace does not take: an open sheet refracting between two media, a crystal on a side of its facets, more
    planar faces than it traces): the run is refused as out of the model's scope (``TerminationReason.OUT_OF_MODEL_SCOPE``),
    not as invalid input. ``errors/registry.py`` lists what is still raised this way."""
