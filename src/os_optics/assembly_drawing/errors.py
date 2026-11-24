"""Stable error codes of the local Detailed Assembly Drawing generator.

Every refusal raises ``AssemblyDrawingError`` with one ``DrawingErrorCode`` and a tuple of fixed, human-safe problem
texts (they name package-local references and contract fields).
"""

from __future__ import annotations

from enum import StrEnum


class DrawingErrorCode(StrEnum):
    #: The Assembly Preview Package does not verify (manifest, digests, schema, geometry).
    PREVIEW_PACKAGE_INVALID = "PREVIEW_PACKAGE_INVALID"
    #: The preview verifies but is not bound to a commercial resolution and its Customer Purchase BOM.
    PREVIEW_NOT_COMMERCIALLY_BOUND = "PREVIEW_NOT_COMMERCIALLY_BOUND"
    #: The detail document is not JSON / not the contract's shape / not in canonical file form.
    DETAIL_PACKAGE_MALFORMED = "DETAIL_PACKAGE_MALFORMED"
    #: The detail declares another schema id or version than this generator supports.
    DETAIL_UNSUPPORTED_VERSION = "DETAIL_UNSUPPORTED_VERSION"
    #: ``detail_content_hash`` is not the digest of the document (or not the expected one).
    DETAIL_CONTENT_HASH_MISMATCH = "DETAIL_CONTENT_HASH_MISMATCH"
    #: The detail is bound to another preview package, Project, Local Validation, resolution or BOM.
    DETAIL_BINDING_MISMATCH = "DETAIL_BINDING_MISMATCH"
    #: A dimension does not resolve in, or does not hold for, the bound preview scene.
    DETAIL_DIMENSION_INVALID = "DETAIL_DIMENSION_INVALID"
    #: A fact the sheet must show is absent (a part without a Customer Purchase BOM line reference, a fixation
    #: without board holes on a gridded board, an INSERTION_DEPTH setting without exactly one INSERTION_DEPTH
    #: dimension measured from the poses, ...).
    MISSING_REQUIRED_FACT = "MISSING_REQUIRED_FACT"
    #: A part GLB cannot be read as a closed triangle mesh.
    PART_GEOMETRY_UNREADABLE = "PART_GEOMETRY_UNREADABLE"
    #: The semantic breadboard (outline, thickness, top) disagrees with the breadboard's own mesh.
    BOARD_GEOMETRY_MISMATCH = "BOARD_GEOMETRY_MISMATCH"
    #: The local assembly validation of the same package FAILED and diagnostic mode was not requested.
    DRAWING_REFUSED_VALIDATION_FAILED = "DRAWING_REFUSED_VALIDATION_FAILED"
    #: The local assembly validation is UNKNOWN or OUT_OF_MODEL_SCOPE (nothing established that the layout holds) and
    #: diagnostic mode was not requested.
    DRAWING_REFUSED_VALIDATION_INCONCLUSIVE = "DRAWING_REFUSED_VALIDATION_INCONCLUSIVE"
    #: The assembly does not fit the A3 sheet at any supported scale.
    LAYOUT_DOES_NOT_FIT = "LAYOUT_DOES_NOT_FIT"
    #: A drawing output set does not verify (``verify_assembly_drawing``).
    DRAWING_OUTPUTS_INVALID = "DRAWING_OUTPUTS_INVALID"
    #: An input is not of an accepted type (e.g. a path given to the JSON API).
    INPUT_INVALID = "INPUT_INVALID"


class AssemblyDrawingError(ValueError):
    """The generator refused; ``code`` is a ``DrawingErrorCode``, ``problems`` lists why."""

    def __init__(self, code: DrawingErrorCode, problems: tuple[str, ...] | list[str]):
        self.code = code
        self.problems = tuple(problems) or (code.value,)
        super().__init__(f"{code.value}: " + "; ".join(self.problems))

    def as_dict(self) -> dict[str, object]:
        return {"code": self.code.value, "problems": [{"code": self.code.value, "message": p} for p in self.problems]}
