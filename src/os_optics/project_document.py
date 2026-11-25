"""The one place where a Project document becomes a ``ProjectDocumentV2`` on the Engine side.

``parse_project_document`` reads a Project document (a JSON-shaped mapping) given to the Engine from outside -- a file,
a request, a run directory, a Project an import or a native component is added to -- and raises the contract's
``pydantic.ValidationError`` when it is not a valid v2 Project; each caller states the refusal in its own terms
(``local.catalog.load_project``: ``PROJECT_INVALID``). A Project the Engine builds itself from parts it already holds
(an example, a probe of an import's own unit, a re-pinned copy of a parsed Project) is constructed with the contract
directly. A change of how a Project document is read lands here.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ._vendor.shared_contracts.optical_design import ProjectDocumentV2


#: The union members pydantic names in an error's location after a discriminated field: no field a customer writes.
_UNION_TAGS = {"source_model": ("GAUSSIAN_BEAM", "EXTENDED_EMITTER", "FLUORESCENCE", "RAMAN", "SAMPLED_INTENSITY", "SAMPLED_FIELD", "SPECKLE"),
               "emitting_surface": ("RECTANGLE", "DISC"), "illumination": ("UNIFORM_DISC", "UNIFORM_RECTANGLE", "GAUSSIAN")}


def plain_location(loc: tuple[Any, ...]) -> tuple[Any, ...]:
    """A validation error's location as the document spells it: ``source_model.m2``, not ``source_model.GAUSSIAN_BEAM.m2``."""

    out: list[Any] = []
    for part in loc:
        if not (out and isinstance(out[-1], str) and part in _UNION_TAGS.get(out[-1], ())):
            out.append(part)
    return tuple(out)


def parse_project_document(document: Mapping[str, Any]) -> ProjectDocumentV2:
    """The ``ProjectDocumentV2`` of a Project document (``pydantic.ValidationError`` when it is not a valid v2 Project)."""

    return ProjectDocumentV2.model_validate(document)


__all__ = ["parse_project_document"]
