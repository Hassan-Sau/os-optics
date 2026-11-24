"""`OpticalWorkspaceV1`: a Project together with what composing and editing it needs (``schema =
"opensci.optical_workspace"``, ``schema_version = "1.0"``).

A workspace wraps one ``ProjectDocumentV2`` -- the physics, unchanged and hashed exactly as on its own -- with:

* ``module``: what makes the Project a reusable module (``ModuleDescriptorV1``): its identity, which of its objects are
  test fixtures (a test source, a test observation plane, a test target, a test element) and what happens to them on
  import, its optical interfaces (input and output ports in the module's design frame) and which of its targets carry
  over to a system that imports it;
* ``instances``: the modules imported into this Project (``LayoutInstanceV1``): the source module and its digest, where
  it was placed, how each source id maps to the Project id it became, what happened to its fixtures, the content digest
  of every imported member at import (so later edits of an instance are recognised, deletions included: a member
  deleted from the Project keeps its record) and of the module object each component and observation plane was copied
  from, the module's interface ports and its own validations;
* ``groups``: editing groups (``EditingGroupV1``): named sets of components and observation planes that are selected and
  moved together, each with an optional reference frame; groups may overlap and never change the Project's physics or
  its hash;
* ``operations``: the journal of the last ``MAX_OPERATIONS`` operations applied (``OperationRecordV1``), which makes
  an operation safe to retry: the same ``op_id`` with the same request returns the recorded result;
* ``validations``: validation runs of the system and of imported modules (``ValidationEntryV1``).

``workspace_digest()`` is osc1 over ``project``, ``module``, ``instances`` without their copied ``module_validations``,
and ``groups``: the organization and the physics only (the journal and every validation record excluded, so the runtime
that validated a module never changes it). It is the ``expected_base`` an operation names. ``module_digest()`` is osc1 over ``project`` and ``module``: the
identity of the workspace exported as a module.

Freshness of a validation is derived, never stored (``validation_freshness``): a ``SYSTEM`` validation is ``CURRENT``
exactly when its ``project_hash`` and ``engine_input_hash`` equal the current ones, otherwise ``HISTORICAL`` with the
reasons; a ``MODULE`` validation describes the module on its own and is never current for the system that imports it.
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import Field, JsonValue, StringConstraints, TypeAdapter, field_validator, model_validator

from ..base import StrictContractModel
from ..common.geometry import PoseV1, Vector3
from ..common.hashes import Sha256Digest, osc1_digest
from ..common.identifiers import BuildVersion, OpenSciId
from ..common.osc1 import Osc1Error, canonicalize_osc1_text
from ..common.ordering import duplicate_keys
from ..common.text import AuthorName, AuthorText
from ..design_catalog.optical_catalog import CoatingId, MaterialId
from .component import ModelId, OpticalRoleV1, require_unit_quaternion
from .local_validation import LocalRuntimeKindV1, LocalValidationStatusV1
from .scientific_result import ModelScopeApplicabilityV1, ObservationQuantityV1, RequirementAssessmentV1, ResultOutcomeV1
from .project import ProjectDocumentV2


OPTICAL_WORKSPACE_SCHEMA_ID: Literal["opensci.optical_workspace"] = "opensci.optical_workspace"
OPTICAL_WORKSPACE_SCHEMA_VERSION: Literal["1.0"] = "1.0"
#: The journal keeps the last this many operations.
MAX_OPERATIONS = 256
#: The most validation entries a workspace (and an instance) keeps.
MAX_VALIDATIONS = 256
#: The largest canonical (osc1 / JCS) size of one operation summary, in characters.
MAX_SUMMARY_CHARS = 16384
#: A port direction's length may differ from 1 by this much (the Engine's quaternion-norm tolerance).
DIRECTION_TOLERANCE = 1e-6

InstanceKeyV1 = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^[a-z][a-z0-9_]{0,23}$"),
    Field(description="Key of an imported module instance: a lowercase letter, then lowercase letters, digits or '_' (at most 24)."),
]
OperationId = Annotated[
    str,
    StringConstraints(strict=True, min_length=1, max_length=128, pattern=r"^[!-~]+$"),
    Field(description="Id of one operation (printable ASCII, no spaces): chosen by the caller, who reuses it to retry, or generated "
                      "(unique in the journal) when the caller gives none."),
]
OperationKind = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^[a-z][a-z0-9_]{0,63}$"),
    Field(description="The operation's name (for example 'import', 'transform', 'group_create')."),
]
Revision = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=64, pattern=r"^[ -~]*[!-~][ -~]*$")]

ObjectKind = Literal["COMPONENT", "OBSERVATION", "TARGET"]
IdMapKind = Literal["COMPONENT", "OBSERVATION", "TARGET", "MODEL", "MATERIAL", "COATING"]
_ID_GRAMMAR: dict[str, TypeAdapter] = {
    "MODEL": TypeAdapter(ModelId), "MATERIAL": TypeAdapter(MaterialId), "COATING": TypeAdapter(CoatingId),
}
_DEFAULT_GRAMMAR = TypeAdapter(OpenSciId)


def _check_id(kind: str, value: str, where: str) -> None:
    try:
        _ID_GRAMMAR.get(kind, _DEFAULT_GRAMMAR).validate_python(value)
    except ValueError:
        raise ValueError(f"{where}: {value!r} is not a valid {kind.lower()} id") from None


# ------------------------------------------------------------------------------------------------- module


class FixtureRoleV1(StrictContractModel):
    """An object of the module that exists only to test the module on its own: ``TEST_SOURCE`` (a source component),
    ``TEST_ELEMENT`` (any other component), ``TEST_OBSERVATION`` (an observation plane) or ``TEST_TARGET`` (a target).
    ``default_action`` says what an import does with it unless the import states otherwise: ``REMOVE`` or ``RETAIN``."""

    object_kind: ObjectKind
    object_id: OpenSciId
    role: Literal["TEST_SOURCE", "TEST_OBSERVATION", "TEST_TARGET", "TEST_ELEMENT"]
    default_action: Literal["REMOVE", "RETAIN"]

    @model_validator(mode="after")
    def _kind_matches_role(self) -> FixtureRoleV1:
        expected = {"TEST_SOURCE": "COMPONENT", "TEST_ELEMENT": "COMPONENT", "TEST_OBSERVATION": "OBSERVATION",
                    "TEST_TARGET": "TARGET"}[self.role]
        if self.object_kind != expected:
            raise ValueError(f"fixture {self.object_id}: a {self.role} is a {expected}, not a {self.object_kind}")
        return self


class InterfacePortV1(StrictContractModel):
    """An optical interface of the module, in the module's design frame: light enters (``INPUT``) or leaves
    (``OUTPUT``) through ``point_mm`` travelling along ``direction`` (a unit vector). ``wavelength_band_nm`` (min, max)
    is the band the port is meant for, ``beam_reference`` the source component whose beam defines the port's boundary
    condition (its waist, size and polarization), ``aperture_radius_mm`` the usable radius around the point."""

    port_id: OpenSciId
    role: Literal["INPUT", "OUTPUT"]
    point_mm: Vector3
    direction: Vector3
    wavelength_band_nm: tuple[float, float] | None = None
    beam_reference: OpenSciId | None = None
    aperture_radius_mm: float | None = Field(None, gt=0)
    description: AuthorText | None = None

    @model_validator(mode="after")
    def _consistent(self) -> InterfacePortV1:
        length = math.sqrt(sum(c * c for c in self.direction))
        if abs(length - 1.0) > DIRECTION_TOLERANCE:
            raise ValueError(f"port {self.port_id}: direction must be a unit vector (length {length!r})")
        if self.wavelength_band_nm is not None and not 0 < self.wavelength_band_nm[0] <= self.wavelength_band_nm[1]:
            raise ValueError(f"port {self.port_id}: wavelength_band_nm is (min, max) with 0 < min <= max")
        return self


class RequirementScopeV1(StrictContractModel):
    """Where one of the module's targets applies: ``MODULE_ONLY`` (it checks the module on its own and stays out of an
    importing system) or ``CARRY_TO_PARENT`` (an import copies it into the system, renamed with the instance)."""

    target_id: OpenSciId
    scope: Literal["MODULE_ONLY", "CARRY_TO_PARENT"]


class ModuleDescriptorV1(StrictContractModel):
    """What makes a Project a reusable module: ``module_id`` and ``title`` (with an optional ``revision`` label and
    ``description``), its test ``fixtures``, its optical ``interfaces`` and the scope of its ``requirements``. Every
    object, port source and target it names is in the workspace's Project."""

    module_id: OpenSciId
    title: AuthorName
    revision: Revision | None = None
    description: AuthorText | None = None
    fixtures: tuple[FixtureRoleV1, ...] = ()
    interfaces: tuple[InterfacePortV1, ...] = ()
    requirements: tuple[RequirementScopeV1, ...] = ()

    @model_validator(mode="after")
    def _unique(self) -> ModuleDescriptorV1:
        keys = [(f.object_kind, f.object_id) for f in self.fixtures]
        if len(set(keys)) != len(keys):
            raise ValueError("an object is marked as a fixture once")
        for label, items, key in (("port_id", self.interfaces, "port_id"), ("requirement target_id", self.requirements, "target_id")):
            duplicates = duplicate_keys(items, key)
            if duplicates:
                raise ValueError(f"duplicate {label} in the module: {duplicates}")
        return self


# ---------------------------------------------------------------------------------------------- instances


class InstanceSourceV1(StrictContractModel):
    """The module an instance was imported from: its ``module_id``, ``module_digest`` (``OpticalWorkspaceV1.module_digest``
    of the module workspace), ``title``, ``revision`` and an optional ``label`` the importer gave it."""

    module_id: OpenSciId
    module_digest: Sha256Digest
    title: AuthorName
    revision: Revision | None = None
    label: AuthorName | None = None


class IdMapEntryV1(StrictContractModel):
    """One id of the module (``source_id``) and the Project id it became (``parent_id``). Every import copies each
    definition it needs (a model, material or coating) under its own instance-namespaced id, so ``shared`` is false; a
    ``true`` (written by earlier builds, which reused an identical definition the Project already had) marks a definition
    this instance uses but another instance owns."""

    kind: IdMapKind
    source_id: str = Field(min_length=1, max_length=128)
    parent_id: str = Field(min_length=1, max_length=128)
    shared: bool = False

    @model_validator(mode="after")
    def _grammar(self) -> IdMapEntryV1:
        _check_id(self.kind, self.source_id, "source_id")
        _check_id(self.kind, self.parent_id, "parent_id")
        if self.shared and self.kind not in ("MODEL", "MATERIAL", "COATING"):
            raise ValueError(f"{self.kind} {self.source_id}: only a model, material or coating definition is shared")
        return self


class FixtureOutcomeV1(StrictContractModel):
    """What an import did with one of the module's fixtures: ``RETAINED`` (copied, as ``parent_id``), ``REMOVED`` (not
    copied) or ``REBOUND`` (not copied; what referred to it now refers to the Project object ``rebound_to``, for example
    the system's own source instead of the module's test source). ``affected_targets`` lists the Project targets whose
    meaning the outcome changed."""

    object_kind: ObjectKind
    source_id: OpenSciId
    action: Literal["RETAINED", "REMOVED", "REBOUND"]
    parent_id: OpenSciId | None = None
    rebound_to: OpenSciId | None = None
    affected_targets: tuple[OpenSciId, ...] = ()

    @model_validator(mode="after")
    def _consistent(self) -> FixtureOutcomeV1:
        if (self.action == "RETAINED") != (self.parent_id is not None):
            raise ValueError(f"fixture {self.source_id}: a RETAINED fixture names its parent_id, and only it does")
        if (self.action == "REBOUND") != (self.rebound_to is not None):
            raise ValueError(f"fixture {self.source_id}: a REBOUND fixture names rebound_to, and only it does")
        if len(set(self.affected_targets)) != len(self.affected_targets):
            raise ValueError(f"fixture {self.source_id}: duplicate affected target")
        return self


class MemberDigestV1(StrictContractModel):
    """osc1 of one imported member's content (``parent_id``) as the import wrote it; a different digest now means the
    member was edited after the import, and a member the Project no longer has was deleted after the import.

    ``kind`` names the member's kind when the instance has members of different kinds with the same id (a module may
    give a component and an observation plane the same id); without it the digest is of the instance's first id map
    entry with that id that is not ``shared``. Moving the whole instance gives each unedited member the digest of its
    moved content; an edited or deleted component or observation plane would have had its import content moved too,
    which is not known, so its digest becomes one that no content has, derived from its digest before the move (records
    that held the same content and moved together keep the same digest): it stays an edit (also when a later move
    brings it back onto its old digest, or when it is added back). A member an enclosing instance records as added by the
    replacement of an instance nested in it (``LayoutInstanceV1``) also has a digest no content has: it is an edit of the
    enclosing instance whatever its content. So does a member an instance records although its revision lacks it
    (``LayoutInstanceV1``: a member of a nested instance's part of an enclosing import, or a definition that objects
    outside the instance use): no edit of the instance while the Project lacks it too or, for such a definition, while
    objects outside the instance use it; otherwise a member added to it.

    ``source_digest`` (components and observation planes only) is osc1 of the module object the member was copied from
    (the object with its source id in the module's Project, without its catalog release pin ``product.catalog_version``,
    which the catalog binding sets for the whole Project) when the member's content is that object placed at this
    instance's placement: the instance's own import writes it, and an instance of a composed module carries it from the
    module's record of that instance for each member that was unedited there. A replacement keeps the exact content of
    an unedited member when the new revision's object for it has the same source digest (after a whole-instance move or
    rotation a fresh placement differs from the moved content by floating-point rounding alone); in every other case it
    writes the new revision's content.

    The records of the instances enclosing a replaced instance follow the replacement for each component and observation
    plane they hold as the replaced instance imported it (the same digest, with a source digest in the replaced
    instance's record): when the new revision re-creates the same object (the same source digest) at the unchanged
    placement, also an edited member the replacement discards, the digest becomes the re-created content's. When the new
    revision changes that object or leaves it out, the digest stays (an edit of the enclosing instance, or a deletion
    while the Project lacks the member) and ``nested_revision_digest`` names what the enclosing import content was: osc1
    of ``{"source_digest": ..., "placement": ...}``, the replaced instance's source digest of the member and its
    placement. A later replacement of that instance that re-creates that object at that placement gives the member its
    import content back: the digest becomes the re-created content's and ``nested_revision_digest`` goes. Moving the
    enclosing instance as a whole removes it (the move changes the placement it names)."""

    parent_id: str = Field(min_length=1, max_length=128)
    kind: IdMapKind | None = None
    digest: Sha256Digest
    source_digest: Sha256Digest | None = None
    nested_revision_digest: Sha256Digest | None = None


class TargetOutcomeEntryV1(StrictContractModel):
    target_id: OpenSciId
    status: LocalValidationStatusV1
    assessment: RequirementAssessmentV1 | None = None
    applicability_issue_codes: tuple[str, ...] | None = None


class ValidationEntryV1(StrictContractModel):
    """One validation run: of the whole system (``SYSTEM``) or of a module on its own (``MODULE``, with the
    ``module_digest`` it ran on and, in the workspace's list, the ``instance_key`` of the instance it belongs to).

    ``project_hash`` and ``engine_input_hash`` identify exactly what ran (the Project's osc1 hash and the Engine's input
    hash); ``design_projection_hash`` (the physics without targets) and ``targets_digest``
    (``project_targets_digest``) let a later state tell what changed. ``engine_build_id`` and ``runtime_kind`` identify
    the runtime; ``validation_status`` and ``target_outcomes`` are the result."""

    scope: Literal["SYSTEM", "MODULE"]
    instance_key: InstanceKeyV1 | None = None
    project_hash: Sha256Digest
    engine_input_hash: Sha256Digest
    design_projection_hash: Sha256Digest | None = None
    targets_digest: Sha256Digest | None = None
    engine_build_id: BuildVersion
    validation_status: LocalValidationStatusV1
    target_outcomes: tuple[TargetOutcomeEntryV1, ...] = ()
    outcome: ResultOutcomeV1 | None = None
    observation_quantities: tuple[ObservationQuantityV1, ...] | None = None
    model_scope_issues: tuple[ModelScopeApplicabilityV1, ...] | None = None
    runtime_kind: LocalRuntimeKindV1
    module_digest: Sha256Digest | None = None

    @model_validator(mode="after")
    def _consistent(self) -> ValidationEntryV1:
        if self.scope == "SYSTEM" and (self.instance_key is not None or self.module_digest is not None):
            raise ValueError("a SYSTEM validation names no instance_key or module_digest")
        if self.scope == "MODULE" and self.module_digest is None:
            raise ValueError("a MODULE validation names the module_digest it ran on")
        duplicates = duplicate_keys(self.target_outcomes, "target_id")
        if duplicates:
            raise ValueError(f"duplicate target outcome: {duplicates}")
        return self


class LayoutInstanceV1(StrictContractModel):
    """One module imported into the Project.

    ``placement`` is the pose of the module's design frame in the Project's design frame at import (updated when the
    whole instance is moved by a coordinated transform; an instance nested in it moves with it, also one with no
    component or observation plane left in the Project). ``id_map`` maps every imported id; ``fixtures`` records what
    happened to each fixture; ``member_digests`` records the content of each imported member at import (the members
    the import copied). A copied member deleted from the Project later keeps its id map entry, digest and fixture
    outcome: the deletion is an edit of the instance until an object of the same kind and id is back in the Project
    (then it is the member again, edited or not by its digest) or the instance is replaced
    (``OpticalWorkspaceV1.deleted_members`` lists them). An instance nested in this one (through ``parent_instance``)
    replaced on its own changes this instance's content, and this record says how: a member the nested instance's new
    revision adds gets an id map entry (source id: its id without this instance's key prefix) and a digest no content
    has (an added member, an edit of this instance); a member the new revision no longer has stays recorded as deleted,
    and a changed one keeps its import digest (edited), both naming the object and placement the nested instance had
    for them (``MemberDigestV1.nested_revision_digest``): the nested instance replaced back by a revision with that
    object makes them this instance's import content again, also after a whole-instance rotation made before; a member
    the replacement re-creates from the same object at the same placement (an edited one it discards included) is this
    instance's import content (its digest follows). A member of a nested instance's part of an enclosing import (one
    an enclosing record holds with its import digest) that a replacement leaves out is recorded, with an id map entry
    and a digest no content has, by the innermost instance whose revision lacks it and by every instance between that
    one and the replaced one: the member stays their part (replacing any of them may bring the member back, and an
    object with its id in the Project is a member added to them), not an edit of them while the Project lacks the
    member too. A definition (optical model, material, coating) of the replaced instances that objects outside them
    use stays in the Project with its content; when the new revision lacks it, it is recorded in the same way by the
    innermost instance whose revision lacks it and by every instance between: no edit of them while objects outside
    them use it, a member added to them once nothing outside them does.
    ``interfaces`` are the module's interface ports as its descriptor states them (the module's design frame and ids:
    composed with ``placement`` they give the ports in the Project's design frame); ``module_validations`` are the
    module's own validations, copied with it (scope ``MODULE``). ``parent_instance`` names the instance whose import
    created this one, when it came in as an instance of a composed module (replacing that instance also replaces this
    one)."""

    instance_key: InstanceKeyV1
    source: InstanceSourceV1
    parent_instance: InstanceKeyV1 | None = None
    placement: PoseV1
    id_map: tuple[IdMapEntryV1, ...] = ()
    fixtures: tuple[FixtureOutcomeV1, ...] = ()
    member_digests: tuple[MemberDigestV1, ...] = ()
    interfaces: tuple[InterfacePortV1, ...] = ()
    module_validations: tuple[ValidationEntryV1, ...] = Field((), max_length=MAX_VALIDATIONS)

    @model_validator(mode="after")
    def _consistent(self) -> LayoutInstanceV1:
        require_unit_quaternion(self.placement, f"instance {self.instance_key} placement")
        sources = [(entry.kind, entry.source_id) for entry in self.id_map]
        if len(set(sources)) != len(sources):
            raise ValueError(f"instance {self.instance_key}: a source id is mapped once per kind")
        fixtures = [(f.object_kind, f.source_id) for f in self.fixtures]
        if len(set(fixtures)) != len(fixtures):
            raise ValueError(f"instance {self.instance_key}: one outcome per fixture")
        if self.parent_instance == self.instance_key:
            raise ValueError(f"instance {self.instance_key}: parent_instance names another instance")
        mapped = {(entry.kind, entry.parent_id) for entry in self.id_map}
        members = [(self.member_kind(member), member.parent_id) for member in self.member_digests]
        for member, ref in zip(self.member_digests, members):
            if ref not in mapped:
                what = f"{member.kind.lower()} {member.parent_id!r}" if member.kind is not None else repr(member.parent_id)
                raise ValueError(f"instance {self.instance_key}: member digest of {what}, which the id map does not list")
        if len(set(members)) != len(members):
            raise ValueError(f"instance {self.instance_key}: one digest per member")
        for member, (kind, parent_id) in zip(self.member_digests, members):
            if member.source_digest is not None and kind not in ("COMPONENT", "OBSERVATION"):
                raise ValueError(f"instance {self.instance_key}: {str(kind).lower()} {parent_id!r} has a source digest (only components and "
                                 "observation planes have one)")
            if member.nested_revision_digest is not None and kind not in ("COMPONENT", "OBSERVATION"):
                raise ValueError(f"instance {self.instance_key}: {str(kind).lower()} {parent_id!r} has a nested revision digest (only components "
                                 "and observation planes have one)")
        if len({port.port_id for port in self.interfaces}) != len(self.interfaces):
            raise ValueError(f"instance {self.instance_key}: one entry per interface port")
        if any(v.scope != "MODULE" for v in self.module_validations):
            raise ValueError(f"instance {self.instance_key}: module_validations have scope MODULE")
        return self

    def removed_fixtures(self) -> frozenset[tuple[str, str]]:
        """``(object_kind, source_id)`` of every fixture the import did not copy (REMOVED or REBOUND)."""

        return frozenset((f.object_kind, f.source_id) for f in self.fixtures if f.action != "RETAINED")

    def member_kind(self, member: MemberDigestV1) -> str | None:
        """The kind of the member ``member`` is the digest of: its ``kind``, else the kind of the first id map entry
        with its id that is not ``shared`` (``None`` when the id map has no entry with that id)."""

        if member.kind is not None:
            return member.kind
        kinds = [entry.kind for entry in self.id_map if entry.parent_id == member.parent_id]
        own = [entry.kind for entry in self.id_map if entry.parent_id == member.parent_id and not entry.shared]
        return (own or kinds or [None])[0]

    def member_digest_map(self) -> dict[tuple[str, str], str]:
        """``(kind, parent_id)`` -> digest of every member this instance holds a digest of."""

        return {(kind, member.parent_id): member.digest for member in self.member_digests
                if (kind := self.member_kind(member)) is not None}

    def source_digest_map(self) -> dict[tuple[str, str], str]:
        """``(kind, parent_id)`` -> source digest of every member this instance holds one of (``MemberDigestV1``)."""

        return {(kind, member.parent_id): member.source_digest for member in self.member_digests
                if member.source_digest is not None and (kind := self.member_kind(member)) is not None}

    def copied_members(self) -> frozenset[tuple[str, str]]:
        """``(kind, parent_id)`` of every member the import copied: the id map entries this instance holds a member
        digest of (not a definition another instance owns)."""

        digests = self.member_digest_map()
        return frozenset((entry.kind, entry.parent_id) for entry in self.id_map if not entry.shared and (entry.kind, entry.parent_id) in digests)


# ------------------------------------------------------------------------------------------------- groups


class ObjectRefV1(StrictContractModel):
    """A component or an observation plane of the Project."""

    kind: Literal["COMPONENT", "OBSERVATION"]
    id: OpenSciId


class EditingGroupV1(StrictContractModel):
    """A named set of components and observation planes selected and moved together. ``origin`` is ``IMPORT`` for the
    group an import creates (with its ``instance_key``) and ``USER`` otherwise. ``frame`` is the group's reference frame
    for transforms stated relative to it (an import group's frame starts as the instance's placement). Groups may
    overlap; a group never changes the Project's physics or hash."""

    group_id: OpenSciId
    label: AuthorName | None = None
    members: tuple[ObjectRefV1, ...] = ()
    origin: Literal["IMPORT", "USER"]
    instance_key: InstanceKeyV1 | None = None
    frame: PoseV1 | None = None

    @model_validator(mode="after")
    def _consistent(self) -> EditingGroupV1:
        refs = [(m.kind, m.id) for m in self.members]
        if len(set(refs)) != len(refs):
            raise ValueError(f"group {self.group_id}: a member is listed once")
        if (self.origin == "IMPORT") != (self.instance_key is not None):
            raise ValueError(f"group {self.group_id}: an IMPORT group names its instance_key, and only it does")
        if self.frame is not None:
            require_unit_quaternion(self.frame, f"group {self.group_id} frame")
        return self


# ---------------------------------------------------------------------------------------------- journal


class OperationRecordV1(StrictContractModel):
    """One applied operation: its ``op_id`` and ``kind``, the osc1 digest of its request (``request_digest``), the
    workspace digest it was applied to (``base_digest``) and the one it produced (``result_digest``), and a bounded
    ``summary`` of its result (at most ``MAX_SUMMARY_CHARS`` canonical characters) that a retry returns."""

    op_id: OperationId
    kind: OperationKind
    request_digest: Sha256Digest
    base_digest: Sha256Digest
    result_digest: Sha256Digest
    summary: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("summary")
    @classmethod
    def _bounded(cls, value: dict[str, Any]) -> dict[str, Any]:
        try:
            text = canonicalize_osc1_text(value)
        except Osc1Error as exc:
            raise ValueError(f"summary is not canonical JSON: {exc}") from None
        if len(text) > MAX_SUMMARY_CHARS:
            raise ValueError(f"summary holds at most {MAX_SUMMARY_CHARS} canonical characters, got {len(text)}")
        return value


# ---------------------------------------------------------------------------------------------- workspace


class ValidationFreshness(StrEnum):
    CURRENT = "CURRENT"
    HISTORICAL = "HISTORICAL"


class StalenessReason(StrEnum):
    """Why a validation is historical: the physics (``PHYSICS_CHANGED``), the targets (``TARGETS_CHANGED``), only other
    Project fields (``OTHER_PROJECT_CHANGED``: for example the design intent, the name or the description, with the same
    physics, targets and Engine input), the catalog release the Project ran against (``CATALOG_CHANGED``: the Engine
    input changed while the Project did not) or the Engine build (``ENGINE_CHANGED``) differ from now, or it validated a
    module on its own (``MODULE_SCOPE``)."""

    PHYSICS_CHANGED = "PHYSICS_CHANGED"
    TARGETS_CHANGED = "TARGETS_CHANGED"
    OTHER_PROJECT_CHANGED = "OTHER_PROJECT_CHANGED"
    CATALOG_CHANGED = "CATALOG_CHANGED"
    ENGINE_CHANGED = "ENGINE_CHANGED"
    MODULE_SCOPE = "MODULE_SCOPE"


def project_targets_digest(project: ProjectDocumentV2) -> str:
    """osc1 of the Project's targets (as its payload writes them): the ``targets_digest`` of a validation."""

    return osc1_digest(project.payload().get("targets", []))


def validation_freshness(entry: ValidationEntryV1, *, project_hash: str, engine_input_hash: str,
                         design_projection_hash: str | None = None, targets_digest: str | None = None,
                         engine_build_id: str | None = None) -> tuple[ValidationFreshness, tuple[StalenessReason, ...]]:
    """The freshness of ``entry`` against the current state: the current Project hash and Engine input hash and, when
    known, the current design projection hash, targets digest and Engine build id.

    A ``SYSTEM`` entry is ``CURRENT`` exactly when its ``project_hash`` and ``engine_input_hash`` equal the current ones.
    Otherwise it is ``HISTORICAL`` with the reasons: when the Project changed, ``TARGETS_CHANGED`` if the targets digest
    differs and ``PHYSICS_CHANGED`` if the design projection differs or the change is not shown to be the targets'
    alone; ``OTHER_PROJECT_CHANGED`` instead when the design projection, the targets digest and the Engine input are all
    known to be unchanged (another Project field changed, such as the design intent); ``ENGINE_CHANGED`` if the Engine
    build differs; ``CATALOG_CHANGED`` if the same Project on the same Engine
    build gives another Engine input (the catalog release changed). A ``MODULE`` entry describes a module on its own and
    is always ``HISTORICAL`` for the system (``MODULE_SCOPE``)."""

    if entry.scope == "MODULE":
        return ValidationFreshness.HISTORICAL, (StalenessReason.MODULE_SCOPE,)
    if entry.project_hash == project_hash and entry.engine_input_hash == engine_input_hash:
        return ValidationFreshness.CURRENT, ()
    reasons: list[StalenessReason] = []
    if entry.project_hash != project_hash:
        projection_known = design_projection_hash is not None and entry.design_projection_hash is not None
        physics = projection_known and entry.design_projection_hash != design_projection_hash
        targets_known = targets_digest is not None and entry.targets_digest is not None
        targets = targets_known and entry.targets_digest != targets_digest
        if projection_known and targets_known and not physics and not targets and entry.engine_input_hash == engine_input_hash:
            reasons.append(StalenessReason.OTHER_PROJECT_CHANGED)
        elif physics or not targets:
            reasons.append(StalenessReason.PHYSICS_CHANGED)
        if targets:
            reasons.append(StalenessReason.TARGETS_CHANGED)
    engine = engine_build_id is not None and entry.engine_build_id != engine_build_id
    if engine:
        reasons.append(StalenessReason.ENGINE_CHANGED)
    if entry.project_hash == project_hash and not engine:
        reasons.append(StalenessReason.CATALOG_CHANGED)
    return ValidationFreshness.HISTORICAL, tuple(reasons)


_DIGEST_FIELDS = {"project", "module", "instances", "groups"}
#: Instance fields left out of the workspace digest: the module's validation records copied with it.
_DIGEST_INSTANCE_EXCLUDED = {"module_validations"}
_MODULE_FIELDS = {"project", "module"}


class OpticalWorkspaceV1(StrictContractModel):
    """A Project with its module description, imported instances, editing groups, operation journal and validation
    history (module docstring). Every id the workspace names in the Project exists there, except an imported id the
    instance records as a fixture it did not copy, and a member the instance copied that was deleted from the Project
    after the import (its id map entry, fixture outcome and affected targets keep naming it; ``deleted_members`` lists
    them); group ids, instance keys and operation ids are unique."""

    schema_id: Literal["opensci.optical_workspace"] = Field(alias="schema")
    schema_version: Literal["1.0"]
    project: ProjectDocumentV2
    module: ModuleDescriptorV1 | None = None
    instances: tuple[LayoutInstanceV1, ...] = ()
    groups: tuple[EditingGroupV1, ...] = ()
    operations: tuple[OperationRecordV1, ...] = Field((), max_length=MAX_OPERATIONS)
    validations: tuple[ValidationEntryV1, ...] = Field((), max_length=MAX_VALIDATIONS)

    @model_validator(mode="after")
    def _check(self) -> OpticalWorkspaceV1:
        for label, items, key in (("group_id", self.groups, "group_id"), ("instance_key", self.instances, "instance_key"),
                                  ("op_id", self.operations, "op_id")):
            duplicates = duplicate_keys(items, key)
            if duplicates:
                raise ValueError(f"duplicate {label} in OpticalWorkspaceV1: {duplicates}")
        ids = self._project_ids()
        if self.module is not None:
            self._check_module(self.module, ids)
        keys = {instance.instance_key for instance in self.instances}
        for instance in self.instances:
            if instance.parent_instance is not None and instance.parent_instance not in keys:
                raise ValueError(f"instance {instance.instance_key}: parent_instance {instance.parent_instance!r} is not in the workspace")
            removed = instance.removed_fixtures()
            copied = instance.copied_members()  # a copied member may have been deleted from the Project since
            for entry in instance.id_map:
                absent = entry.parent_id not in ids[entry.kind]
                if absent and (entry.kind, entry.source_id) not in removed and (entry.kind, entry.parent_id) not in copied:
                    raise ValueError(f"instance {instance.instance_key}: {entry.kind.lower()} {entry.parent_id!r} (from "
                                     f"{entry.source_id!r}) is not in the Project")
            for outcome in instance.fixtures:
                for name in ("parent_id", "rebound_to"):
                    value = getattr(outcome, name)
                    if value is not None and value not in ids[outcome.object_kind] and (outcome.object_kind, value) not in copied:
                        raise ValueError(f"instance {instance.instance_key}: fixture {outcome.source_id} {name} {value!r} is not in "
                                         "the Project")
                for target in outcome.affected_targets:
                    if target not in ids["TARGET"] and ("TARGET", target) not in copied:
                        raise ValueError(f"instance {instance.instance_key}: affected target {target!r} is not in the Project")
        for group in self.groups:
            if group.instance_key is not None and group.instance_key not in keys:
                raise ValueError(f"group {group.group_id}: instance {group.instance_key!r} is not in the workspace")
            for member in group.members:
                if member.id not in ids[member.kind]:
                    raise ValueError(f"group {group.group_id}: {member.kind.lower()} {member.id!r} is not in the Project")
        for validation in self.validations:
            if validation.instance_key is not None and validation.instance_key not in keys:
                raise ValueError(f"validation of instance {validation.instance_key!r}, which is not in the workspace")
        return self

    def _project_ids(self) -> dict[str, frozenset[str]]:
        project = self.project
        return {
            "COMPONENT": frozenset(c.instance_id for c in project.components),
            "OBSERVATION": frozenset(o.observation_id for o in project.observations),
            "TARGET": frozenset(t.target_id for t in project.targets),
            "MODEL": frozenset(m.model_id for m in project.optical_models or ()),
            "MATERIAL": frozenset(m.material_id for m in project.model_materials or ()),
            "COATING": frozenset(c.coating_id for c in project.model_coatings or ()),
        }

    def _check_module(self, module: ModuleDescriptorV1, ids: dict[str, frozenset[str]]) -> None:
        for fixture in module.fixtures:
            if fixture.object_id not in ids[fixture.object_kind]:
                raise ValueError(f"module fixture {fixture.object_kind.lower()} {fixture.object_id!r} is not in the Project")
            if fixture.role == "TEST_SOURCE":
                component = next(c for c in self.project.components if c.instance_id == fixture.object_id)
                if component.optical_role is not OpticalRoleV1.SOURCE:
                    raise ValueError(f"module fixture {fixture.object_id!r}: a TEST_SOURCE is a source component")
        sources = {c.instance_id for c in self.project.components if c.optical_role is OpticalRoleV1.SOURCE}
        for port in module.interfaces:
            if port.beam_reference is not None and port.beam_reference not in sources:
                raise ValueError(f"module port {port.port_id}: beam_reference {port.beam_reference!r} is not a source component of the Project")
        for requirement in module.requirements:
            if requirement.target_id not in ids["TARGET"]:
                raise ValueError(f"module requirement {requirement.target_id!r} is not a target of the Project")

    def payload(self) -> dict[str, Any]:
        """Canonical persisted form (``None`` fields omitted)."""

        return self.model_dump(mode="json", by_alias=True, exclude_none=True)

    def workspace_digest(self) -> str:
        """osc1 over ``project``, ``module``, ``instances`` (without their copied ``module_validations``) and ``groups``:
        the ``expected_base`` of an operation. The journal and every validation record are excluded."""

        body = self.model_dump(mode="json", by_alias=True, exclude_none=True, include=_DIGEST_FIELDS)
        body["instances"] = [{k: v for k, v in instance.items() if k not in _DIGEST_INSTANCE_EXCLUDED} for instance in body["instances"]]
        return osc1_digest(body)

    def module_digest(self) -> str:
        """osc1 over ``project`` and ``module``: the identity of this workspace as an importable module."""

        return osc1_digest(self.model_dump(mode="json", by_alias=True, exclude_none=True, include=_MODULE_FIELDS))

    def instance(self, instance_key: str) -> LayoutInstanceV1 | None:
        return next((instance for instance in self.instances if instance.instance_key == instance_key), None)

    def deleted_members(self, instance_key: str) -> tuple[IdMapEntryV1, ...]:
        """The id map entries of the members instance ``instance_key`` copied that the Project no longer has: deleted
        after the import (empty for an unknown instance)."""

        instance = self.instance(instance_key)
        if instance is None:
            return ()
        ids = self._project_ids()
        copied = instance.copied_members()
        return tuple(entry for entry in instance.id_map if (entry.kind, entry.parent_id) in copied and entry.parent_id not in ids[entry.kind])

    def group(self, group_id: str) -> EditingGroupV1 | None:
        return next((group for group in self.groups if group.group_id == group_id), None)

    def operation(self, op_id: str) -> OperationRecordV1 | None:
        return next((operation for operation in self.operations if operation.op_id == op_id), None)


__all__ = [
    "DIRECTION_TOLERANCE",
    "MAX_OPERATIONS",
    "MAX_SUMMARY_CHARS",
    "MAX_VALIDATIONS",
    "OPTICAL_WORKSPACE_SCHEMA_ID",
    "OPTICAL_WORKSPACE_SCHEMA_VERSION",
    "EditingGroupV1",
    "FixtureOutcomeV1",
    "FixtureRoleV1",
    "IdMapEntryV1",
    "InstanceKeyV1",
    "InstanceSourceV1",
    "InterfacePortV1",
    "LayoutInstanceV1",
    "MemberDigestV1",
    "ModuleDescriptorV1",
    "ObjectRefV1",
    "OperationRecordV1",
    "OpticalWorkspaceV1",
    "RequirementScopeV1",
    "StalenessReason",
    "TargetOutcomeEntryV1",
    "ValidationEntryV1",
    "ValidationFreshness",
    "project_targets_digest",
    "validation_freshness",
]
