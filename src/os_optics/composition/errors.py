"""Structured refusals of the composition operations.

An operation either returns its complete result or raises ``CompositionError`` listing every problem it found; the
workspace it was given is never changed. Each problem has a stable ``CompositionErrorCode`` ``code``, a ``message``, an
optional ``location`` and optional structured ``details`` (the ids, SKUs or digests a caller needs to correct the input).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

#: The ``error.code`` of a refused composition call in ``json_api`` replies.
WORKSPACE_OPERATION_REFUSED = "WORKSPACE_OPERATION_REFUSED"


class CompositionErrorCode(StrEnum):
    """Why a composition operation was refused (stable values).

    Input: ``WORKSPACE_INVALID`` (not a workspace or a Project), ``MODULE_INVALID`` (the module to import is not a
    workspace or a Project), ``REQUEST_INVALID`` (a malformed request), ``MODEL_MISSING`` (a component or a compound
    member names an optical model the document does not declare), ``REFERENCE_INVALID`` (any other id that does not
    resolve: a material, a coating, an observation's device, a target's component / observation / source).

    Retry and concurrency: ``OP_ID_REUSED`` (the op_id was already used for another request), ``BASE_CHANGED`` (the
    workspace is not the one ``expected_base`` names; ``details.current_digest``).

    Instances and ids: ``INSTANCE_KEY_EXISTS`` (a different module or placement under an existing key: a second instance
    needs a new key), ``INSTANCE_UNKNOWN``, ``INSTANCE_EDITED`` (replacing an instance with local edits -- members of it
    or of an instance nested in it changed, deleted or added since the import -- without ``discard_edits``;
    ``details.edited``), ``ID_TOO_LONG`` (a namespaced id exceeds its grammar's length),
    ``ID_CONFLICT`` (a namespaced id is already used in the Project, or is the id of a member an instance records as
    deleted), ``CATALOG_BINDING_CONFLICT`` (the catalog components would pin different catalog releases and could not
    all be moved to the active release unchanged; ``details.components`` lists them).

    Fixtures: ``FIXTURE_INVALID`` (an override or rebind names an object that is not a fixture of the module, or a
    rebind target that is not in the Project or is of another kind), ``FIXTURE_REBIND_REQUIRED`` (targets use a removed
    fixture: rebind it or leave the targets out), ``FIXTURE_DEPENDENCY`` (a retained object depends on a removed fixture).

    Selection and transforms: ``SELECTION_INVALID`` (unknown ids, groups or instances), ``SELECTION_AMBIGUOUS`` (an id
    names both a component and an observation: give its kind), ``SELECTION_EMPTY``, ``FRAME_INVALID``, ``PIVOT_INVALID``,
    ``ROTATION_INVALID``, ``TRANSLATION_INVALID``.

    Groups: ``GROUP_EXISTS``, ``GROUP_UNKNOWN``.

    Results: ``CHECK_PROJECT_REFUSED`` (a request with ``require_check`` whose result has a problem the local check
    refuses that the workspace did not have, e.g. two optical bodies overlapping; ``details.problems``),
    ``REFERENCE_BROKEN`` (an operation would leave a target
    or an observation pointing at an object that no longer exists), ``RUN_REFUSED`` (the local run refused the
    Project), ``VALIDATION_PROJECT_MISMATCH`` (a recorded validation ran on another Project), ``BATCH_INVALID``.
    """

    WORKSPACE_INVALID = "WORKSPACE_INVALID"
    MODULE_INVALID = "MODULE_INVALID"
    REQUEST_INVALID = "REQUEST_INVALID"
    MODEL_MISSING = "MODEL_MISSING"
    REFERENCE_INVALID = "REFERENCE_INVALID"
    OP_ID_REUSED = "OP_ID_REUSED"
    BASE_CHANGED = "BASE_CHANGED"
    INSTANCE_KEY_EXISTS = "INSTANCE_KEY_EXISTS"
    INSTANCE_UNKNOWN = "INSTANCE_UNKNOWN"
    INSTANCE_EDITED = "INSTANCE_EDITED"
    ID_TOO_LONG = "ID_TOO_LONG"
    ID_CONFLICT = "ID_CONFLICT"
    CATALOG_BINDING_CONFLICT = "CATALOG_BINDING_CONFLICT"
    FIXTURE_INVALID = "FIXTURE_INVALID"
    FIXTURE_REBIND_REQUIRED = "FIXTURE_REBIND_REQUIRED"
    FIXTURE_DEPENDENCY = "FIXTURE_DEPENDENCY"
    SELECTION_INVALID = "SELECTION_INVALID"
    SELECTION_AMBIGUOUS = "SELECTION_AMBIGUOUS"
    SELECTION_EMPTY = "SELECTION_EMPTY"
    FRAME_INVALID = "FRAME_INVALID"
    PIVOT_INVALID = "PIVOT_INVALID"
    ROTATION_INVALID = "ROTATION_INVALID"
    TRANSLATION_INVALID = "TRANSLATION_INVALID"
    GROUP_EXISTS = "GROUP_EXISTS"
    GROUP_UNKNOWN = "GROUP_UNKNOWN"
    CHECK_PROJECT_REFUSED = "CHECK_PROJECT_REFUSED"
    REFERENCE_BROKEN = "REFERENCE_BROKEN"
    RUN_REFUSED = "RUN_REFUSED"
    VALIDATION_PROJECT_MISMATCH = "VALIDATION_PROJECT_MISMATCH"
    BATCH_INVALID = "BATCH_INVALID"


class CompositionWarningCode(StrEnum):
    """What a successful operation reports for review (stable values): ``UNMARKED_SOURCE_IMPORTED`` (an imported source
    that the module does not mark as a test source: the system now has its light too), ``ABSORBING_PLANE_IMPORTED`` (an
    imported absorbing observation plane that is not marked as a test observation and is not the sensing face of an
    imported detector or beam dump: it stops the beams that reach it),
    ``TARGET_REBOUND`` (a target now reads the light of another source: its wavelength, power and coherence group are
    that source's), ``PATH_SELECTOR_RENAMED`` (a target's beam-path selector was renamed with its source; the branch part
    counts the interactions of the composed system), ``DESIGN_INTENT_DIFFERS`` (the module states another design
    intent than the workspace's Project; the Project's is kept), ``INSTANCE_MEMBERS_REMOVED`` (objects of an instance were
    deleted from the Project; the instance records them as deleted members, an edit of it)."""

    UNMARKED_SOURCE_IMPORTED = "UNMARKED_SOURCE_IMPORTED"
    ABSORBING_PLANE_IMPORTED = "ABSORBING_PLANE_IMPORTED"
    TARGET_REBOUND = "TARGET_REBOUND"
    PATH_SELECTOR_RENAMED = "PATH_SELECTOR_RENAMED"
    DESIGN_INTENT_DIFFERS = "DESIGN_INTENT_DIFFERS"
    INSTANCE_MEMBERS_REMOVED = "INSTANCE_MEMBERS_REMOVED"


@dataclass(frozen=True)
class CompositionProblem:
    """One reason an operation was refused, or one warning of an applied operation."""

    code: str
    message: str
    location: str | None = None
    details: dict[str, Any] | None = field(default=None, compare=False)

    def as_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"code": str(self.code), "message": self.message}
        if self.location is not None:
            data["location"] = self.location
        if self.details:
            data["details"] = dict(self.details)
        return data


class CompositionError(ValueError):
    """The operation was refused; ``problems`` lists every reason. ``workspace_digest``: the digest of the workspace the
    operation was given (unchanged), when it could be read."""

    def __init__(self, problems: list[CompositionProblem] | tuple[CompositionProblem, ...], *, workspace_digest: str | None = None) -> None:
        self.problems: tuple[CompositionProblem, ...] = tuple(problems)
        self.workspace_digest = workspace_digest
        super().__init__("; ".join(f"{p.code}{f' at {p.location}' if p.location else ''}: {p.message}" for p in self.problems))

    @property
    def codes(self) -> tuple[str, ...]:
        return tuple(str(p.code) for p in self.problems)

    def as_dict(self) -> dict[str, Any]:
        return {"code": WORKSPACE_OPERATION_REFUSED, "problems": [p.as_dict() for p in self.problems]}


def refuse(code: CompositionErrorCode, message: str, location: str | None = None, **details: Any) -> CompositionError:
    return CompositionError([CompositionProblem(code, message, location, details or None)])


def warning(code: CompositionWarningCode, message: str, location: str | None = None, **details: Any) -> CompositionProblem:
    return CompositionProblem(code, message, location, details or None)


__all__ = [
    "WORKSPACE_OPERATION_REFUSED",
    "CompositionError",
    "CompositionErrorCode",
    "CompositionProblem",
    "CompositionWarningCode",
    "refuse",
    "warning",
]
