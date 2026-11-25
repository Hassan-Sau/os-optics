"""`LocalValidationRecordV1`: the structured result of validating a pure-optical Project on the user's
own device.

The record is produced by the os-optics engine (headless Python or the same package in the browser Designer) and is
self-contained: it needs no OpenSci server and no account. When the user later submits the design for commercial
resolution, the same record is what their Agent submits together with the Project; ``check_local_validation_binding``
is the structural consistency check between the two.

Hash profiles mirror what the engine (``os-optics``) emits in its ``ValidationManifest``:
``project.project_hash`` is osc1 of the ``ProjectDocumentV2`` payload, ``input_hash`` is the Engine's exact
input hash (``eng-exact-py1``, subject ``engine_input``) and ``result_hash`` the Engine's quantized result
hash (``eng-quant-py1``, subject ``engine_result``, with its ``sig_digits`` / ``abs_floor`` params).
The adapter from an Engine ``ValidationResult`` to this record is part of the local runtime
(``os_optics.local``).

``simulated_parts`` records, per catalog component, exactly what the public optical catalog supplied to the
run: ``optical_sku``, the public ``catalog_version`` and the simulation model (``model_kind`` /
``model_version``). A catalog release may re-characterize a product's model values under a new ``catalog_version``
while its ``optical_sku`` stays, so this list binds the ``project_hash`` to the published models the results were
computed with. ``resolve_simulated_parts`` derives the list from a ``PublicOpticalCatalogV2``.

The validated configuration is always the Project's nominal pose: ``retained_dofs`` records which DOFs
are retained and with what control intent.

``payload()`` omits ``None`` fields -- see the hash evolution rule in ``project.py``.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import Field, StringConstraints, field_validator, model_validator

from ..base import StrictContractModel
from ..common.hashes import HashProfile, HashRef, osc1_digest, require_engine_quantized_params, require_hash_profile
from ..common.identifiers import BuildVersion, InstanceKey, OpenSciId
from ..common.ordering import duplicate_keys
from ..common.safe_int import SafeJsonInt
from ..design_catalog.optical_catalog import CatalogVersion, OpticalSku, PublicOpticalCatalogV2
from ._ordering import sort_raw
from .dof import (
    DOF_LIMIT_FIELDS,
    DRIFT_DESCRIPTION,
    HARD_LIMITS_DESCRIPTION,
    REPEATABILITY_DESCRIPTION,
    TRAVEL_DESCRIPTION,
    UNIT_DESCRIPTION,
    DofAxisV1,
    DofControlIntentV1,
    DofDriftV1,
    DofIntervalV1,
    check_dof_limits,
    dof_sort_index,
)
from .project import PROJECT_HASH_SUBJECT, ProjectDocumentV2
from .scientific_result import ModelScopeApplicabilityV1, ObservationQuantityV1, RequirementAssessmentV1, ResultOutcomeV1


LOCAL_VALIDATION_SCHEMA_ID: Literal["opensci.local_validation"] = "opensci.local_validation"
LOCAL_VALIDATION_SCHEMA_VERSION: Literal["1.0"] = "1.0"
#: The PyPI distribution that produces the records an engine writes (``engine_package`` of every engine-produced
#: identity): ``importlib.metadata.distribution(ENGINE_PACKAGE_NAME)`` is the installed engine; its import package is
#: ``os_optics``. Engines write this name.
ENGINE_PACKAGE_NAME: Literal["os-optics"] = "os-optics"
#: The engine's distribution name up to 0.9.0, which os-optics 0.10.0 and 0.11.0 still wrote. Read only: a record
#: stored or kept from those engines (a submission the server re-reads, a run directory on the user's disk) still parses;
#: no current engine writes it.
EARLIER_ENGINE_PACKAGE_NAME: Literal["opensci-engine"] = "opensci-engine"
#: ``engine_package`` as read: the distribution that produced the document. Written as ``ENGINE_PACKAGE_NAME``.
EnginePackageName = Annotated[
    Literal["os-optics", "opensci-engine"],
    Field(description="The PyPI distribution of the engine that produced this document: os-optics (written by every "
                      "current engine); opensci-engine only in documents written by an engine release up to 0.11.0."),
]
ENGINE_INPUT_HASH_SUBJECT = "engine_input"
ENGINE_RESULT_HASH_SUBJECT = "engine_result"

IssueCode = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=128)]

_SIG_DIGITS_PATTERN = r"^[1-9][0-9]*$"
_ABS_FLOOR_PATTERN = r"^(0|[1-9][0-9]*)(\.[0-9]+)?$"


class LocalRuntimeKindV1(StrEnum):
    """Where the OpenSci engine ran: ``PYTHON_LOCAL`` (os-optics in Python on the user's computer) or
    ``BROWSER_PYTHON`` (the same package in the browser Designer). Both run on the user's own device."""

    PYTHON_LOCAL = "PYTHON_LOCAL"
    BROWSER_PYTHON = "BROWSER_PYTHON"


class LocalValidationStatusV1(StrEnum):
    """Status of a local validation."""

    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    OUT_OF_MODEL_SCOPE = "OUT_OF_MODEL_SCOPE"


# Same "worst of" severity as the Engine (NOT_APPLICABLE never dominates).
_STATUS_SEVERITY = {
    LocalValidationStatusV1.NOT_APPLICABLE: -1,
    LocalValidationStatusV1.PASS: 0,
    LocalValidationStatusV1.WARNING: 1,
    LocalValidationStatusV1.UNKNOWN: 2,
    LocalValidationStatusV1.OUT_OF_MODEL_SCOPE: 3,
    LocalValidationStatusV1.FAIL: 4,
}


def status_severity(status: LocalValidationStatusV1) -> int:
    return _STATUS_SEVERITY[status]


class ValidatedProjectRefV1(StrictContractModel):
    """Which Project revision was validated: schema identity + osc1 content hash of its payload."""

    project_schema: Literal["opensci.project"] = "opensci.project"
    project_schema_version: Literal["2.0"] = "2.0"
    project_hash: HashRef

    @field_validator("project_hash")
    @classmethod
    def _osc1(cls, value: HashRef) -> HashRef:
        value = require_hash_profile(value, HashProfile.OSC1, field_name="ValidatedProjectRefV1.project_hash")
        if value.subject != PROJECT_HASH_SUBJECT:
            raise ValueError(f"ValidatedProjectRefV1.project_hash subject must be {PROJECT_HASH_SUBJECT!r}")
        return value


class LocalRuntimeIdentityV1(StrictContractModel):
    """Identity of the local engineering runtime that produced the record."""

    runtime_kind: LocalRuntimeKindV1
    engine_package: EnginePackageName = ENGINE_PACKAGE_NAME
    engine_version: BuildVersion
    engine_build_id: BuildVersion | None = None
    engine_schema_version: BuildVersion | None = Field(None, description="The Engine's public contract schema version.")
    numerical_config_version: BuildVersion | None = None


SimulationModelKind = Annotated[str, StringConstraints(strict=True, max_length=64, pattern=r"^[A-Z][A-Z0-9_]*$")]
SimulationModelVersion = Annotated[str, StringConstraints(strict=True, max_length=16, pattern=r"^[1-9][0-9]*$")]


class SimulationModelRefV1(StrictContractModel):
    """A published simulation model of a catalog product (its ``model_kind`` / ``model_version``, e.g. ``SURFACE_SEQUENCE`` 1)."""

    model_kind: SimulationModelKind
    model_version: SimulationModelVersion


class SimulatedPartV1(StrictContractModel):
    """What the public optical catalog supplied to the run for one catalog component."""

    instance_id: InstanceKey
    optical_sku: OpticalSku
    catalog_version: CatalogVersion
    simulation_model: SimulationModelRefV1


class TargetOutcomeV1(StrictContractModel):
    target_id: OpenSciId
    metric: OpenSciId
    status: LocalValidationStatusV1
    value: float | None = None
    unit: str | None = Field(None, min_length=1, max_length=40)
    assessment: RequirementAssessmentV1 | None = None
    applicability_issue_codes: tuple[str, ...] | None = None

    @model_validator(mode="after")
    def _assessment_consistent(self):
        expected = {"PASS": "MET", "WARNING": "MET", "FAIL": "UNMET", "UNKNOWN": "INDETERMINATE",
                    "OUT_OF_MODEL_SCOPE": "INDETERMINATE", "NOT_APPLICABLE": "NOT_APPLICABLE"}[self.status.value]
        if self.applicability_issue_codes:
            expected = "INDETERMINATE"
            if self.assessment != expected:
                raise ValueError("model applicability requires an explicit INDETERMINATE assessment")
        if self.assessment is not None and self.assessment != expected:
            raise ValueError("target assessment contradicts its diagnostic status")
        return self


class IssueCountV1(StrictContractModel):
    code: IssueCode
    count: SafeJsonInt = Field(ge=1)


class SimulationSummaryV1(StrictContractModel):
    """Compact summary of the local optical simulation (the full result stays local).

    ``status`` can never be less severe than its worst target outcome.
    """

    status: LocalValidationStatusV1
    target_outcomes: tuple[TargetOutcomeV1, ...] = ()
    issue_counts: tuple[IssueCountV1, ...] = ()
    path_count: SafeJsonInt = Field(0, ge=0)
    observation_count: SafeJsonInt = Field(0, ge=0)
    outcome: ResultOutcomeV1 | None = None
    observation_quantities: tuple[ObservationQuantityV1, ...] | None = None
    model_scope_issues: tuple[ModelScopeApplicabilityV1, ...] | None = None

    @field_validator("target_outcomes", mode="before")
    @classmethod
    def _canonicalize_targets(cls, value: Any) -> Any:
        return sort_raw(value, lambda get, item: get(item, "target_id"))

    @field_validator("issue_counts", mode="before")
    @classmethod
    def _canonicalize_issues(cls, value: Any) -> Any:
        return sort_raw(value, lambda get, item: get(item, "code"))

    @model_validator(mode="after")
    def _check(self) -> SimulationSummaryV1:
        for label, items, key in (("target_id", self.target_outcomes, "target_id"), ("issue code", self.issue_counts, "code")):
            duplicates = duplicate_keys(items, key)
            if duplicates:
                raise ValueError(f"duplicate {label} in SimulationSummaryV1: {duplicates}")
        worst_target = max((status_severity(t.status) for t in self.target_outcomes), default=status_severity(LocalValidationStatusV1.NOT_APPLICABLE))
        if status_severity(self.status) < worst_target:
            raise ValueError(f"simulation status {self.status.value} is less severe than a target outcome")
        if self.outcome is not None:
            requirements = self.outcome.requirements
            expected = {"MET": set(requirements.met), "UNMET": set(requirements.unmet),
                        "INDETERMINATE": set(requirements.indeterminate), "NOT_APPLICABLE": set(requirements.not_applicable)}
            for target in self.target_outcomes:
                if target.assessment is None or target.target_id not in expected[target.assessment]:
                    raise ValueError("requirement summary contradicts target assessments")
            if set().union(*expected.values()) != {t.target_id for t in self.target_outcomes}:
                raise ValueError("requirement summary must cover exactly the recorded targets")
            unevaluated = requirements.declared and not self.target_outcomes and self.outcome.execution.outcome in {"FAILED", "CANCELLED"}
            if requirements.declared != bool(self.target_outcomes) and not unevaluated:
                raise ValueError("declared requirements contradict target outcomes")
        if self.observation_quantities is not None:
            ids = [q.observation_id for q in self.observation_quantities]
            if len(ids) != len(set(ids)) or len(ids) != self.observation_count:
                raise ValueError("observation quantities must uniquely cover the recorded observations")
        if self.model_scope_issues is not None:
            quantities = {q.observation_id: q for q in self.observation_quantities or ()}
            targets = {t.target_id: t for t in self.target_outcomes}
            for issue in self.model_scope_issues:
                if not set(issue.affected_observation_ids) <= quantities.keys() or not set(issue.affected_target_ids) <= targets.keys():
                    raise ValueError("model applicability refers to unrecorded observations or targets")
                for oid in issue.affected_observation_ids:
                    q = quantities[oid]
                    if q.quantity_state != "OUT_OF_MODEL_SCOPE" or issue.code not in q.issue_codes:
                        raise ValueError("model applicability contradicts observation evidence")
            for target in self.target_outcomes:
                codes = {i.code for i in self.model_scope_issues if target.target_id in i.affected_target_ids}
                if set(target.applicability_issue_codes or ()) != codes:
                    raise ValueError("target applicability differs from model scope evidence")
        elif any(t.applicability_issue_codes for t in self.target_outcomes):
            raise ValueError("target applicability requires model scope evidence")
        return self


class RetainedDofStateV1(StrictContractModel):
    """One retained DOF of the validated (nominal) configuration, its control intent and the limits the Project states for it
    (``dof.DOF_LIMIT_FIELDS``; absent when it states none)."""

    instance_id: InstanceKey
    axis: DofAxisV1
    control: DofControlIntentV1
    travel: DofIntervalV1 | None = Field(None, description=TRAVEL_DESCRIPTION)
    hard_limits: DofIntervalV1 | None = Field(None, description=HARD_LIMITS_DESCRIPTION)
    repeatability: float | None = Field(None, gt=0.0, description=REPEATABILITY_DESCRIPTION)
    drift: DofDriftV1 | None = Field(None, description=DRIFT_DESCRIPTION)
    unit: Literal["mm", "deg"] | None = Field(None, description=UNIT_DESCRIPTION)

    @model_validator(mode="after")
    def _limits(self) -> RetainedDofStateV1:
        check_dof_limits(self, self.axis)
        return self


class ModelVersionEntryV1(StrictContractModel):
    key: OpenSciId
    version: BuildVersion


class ReproducibilityV1(StrictContractModel):
    """What is needed to reproduce the local run bit-for-bit (or within the quantized result hash)."""

    deterministic: bool = Field(description="True when the run used no pseudo-random sampling.")
    random_seed: SafeJsonInt | None = Field(None, ge=0)
    prng_id: str | None = Field(None, min_length=1, max_length=64)
    model_versions: tuple[ModelVersionEntryV1, ...] = ()

    @field_validator("model_versions", mode="before")
    @classmethod
    def _canonicalize(cls, value: Any) -> Any:
        return sort_raw(value, lambda get, item: get(item, "key"))

    @model_validator(mode="after")
    def _check(self) -> ReproducibilityV1:
        duplicates = duplicate_keys(self.model_versions, "key")
        if duplicates:
            raise ValueError(f"duplicate model_versions key: {duplicates}")
        if not self.deterministic and (self.random_seed is None or self.prng_id is None):
            raise ValueError("a non-deterministic run must record random_seed and prng_id")
        return self


def _pinned_subject(value: HashRef, subject: str, field_name: str) -> HashRef:
    if value.subject != subject:
        raise ValueError(f"{field_name} subject must be {subject!r}, got {value.subject!r}")
    return value


class LocalValidationRecordV1(StrictContractModel):
    """One local validation of one pure-optical Project revision, computed on the user's device.

    Status rules: ``validation_status`` is never less severe than ``simulation.status``, and a simulation
    that checked nothing (``NOT_APPLICABLE``) can never be reported as a ``PASS`` validation.
    """

    schema_id: Literal["opensci.local_validation"] = Field(alias="schema")
    schema_version: Literal["1.0"]
    trust_basis: Literal["LOCAL_RUNTIME"] = Field(
        "LOCAL_RUNTIME",
        description="Computed by os-optics on the user's device.",
    )
    project: ValidatedProjectRefV1
    runtime: LocalRuntimeIdentityV1
    simulated_parts: tuple[SimulatedPartV1, ...] = ()
    input_hash: HashRef
    result_hash: HashRef
    simulation: SimulationSummaryV1
    validation_status: LocalValidationStatusV1
    retained_dofs: tuple[RetainedDofStateV1, ...] = ()
    reproducibility: ReproducibilityV1

    @field_validator("input_hash")
    @classmethod
    def _input_hash(cls, value: HashRef) -> HashRef:
        value = require_hash_profile(value, HashProfile.ENGINE_EXACT_PY1, field_name="LocalValidationRecordV1.input_hash")
        return _pinned_subject(value, ENGINE_INPUT_HASH_SUBJECT, "LocalValidationRecordV1.input_hash")

    @field_validator("result_hash")
    @classmethod
    def _result_hash(cls, value: HashRef) -> HashRef:
        value = require_engine_quantized_params(value, field_name="LocalValidationRecordV1.result_hash")
        if not re.fullmatch(_SIG_DIGITS_PATTERN, value.params["sig_digits"]):
            raise ValueError("LocalValidationRecordV1.result_hash params.sig_digits must be a positive integer string")
        if not re.fullmatch(_ABS_FLOOR_PATTERN, value.params["abs_floor"]):
            raise ValueError("LocalValidationRecordV1.result_hash params.abs_floor must be a non-negative plain decimal string")
        return _pinned_subject(value, ENGINE_RESULT_HASH_SUBJECT, "LocalValidationRecordV1.result_hash")

    @field_validator("simulated_parts", mode="before")
    @classmethod
    def _canonicalize_parts(cls, value: Any) -> Any:
        return sort_raw(value, lambda get, item: get(item, "instance_id"))

    @field_validator("retained_dofs", mode="before")
    @classmethod
    def _canonicalize_dofs(cls, value: Any) -> Any:
        return sort_raw(value, lambda get, item: (get(item, "instance_id"), dof_sort_index(get(item, "axis"))))

    @model_validator(mode="after")
    def _check(self) -> LocalValidationRecordV1:
        if self.simulation.model_scope_issues and status_severity(self.validation_status) < status_severity(LocalValidationStatusV1.OUT_OF_MODEL_SCOPE):
            raise ValueError("model applicability requires validation_status at least OUT_OF_MODEL_SCOPE")
        duplicates = duplicate_keys(self.retained_dofs, "instance_id", "axis")
        if duplicates:
            raise ValueError(f"duplicate retained DOF state: {duplicates}")
        part_duplicates = duplicate_keys(self.simulated_parts, "instance_id")
        if part_duplicates:
            raise ValueError(f"duplicate simulated_parts instance_id: {part_duplicates}")
        if status_severity(self.validation_status) < status_severity(self.simulation.status):
            raise ValueError(
                f"validation_status {self.validation_status.value} cannot be less severe than simulation status {self.simulation.status.value}"
            )
        if self.simulation.status is LocalValidationStatusV1.NOT_APPLICABLE and self.validation_status is LocalValidationStatusV1.PASS:
            raise ValueError("validation_status cannot be PASS when the simulation checked nothing (NOT_APPLICABLE)")
        return self

    def payload(self) -> dict[str, Any]:
        """Canonical hashed/persisted form; ``None`` fields are omitted."""

        return self.model_dump(mode="json", by_alias=True, exclude_none=True)

    def record_hash(self) -> str:
        return osc1_digest(self.payload())


def validated_project_ref(project: ProjectDocumentV2) -> ValidatedProjectRefV1:
    return ValidatedProjectRefV1(project_hash=project.content_hash_ref())


def nominal_retained_dof_states(project: ProjectDocumentV2) -> tuple[RetainedDofStateV1, ...]:
    """Every retained DOF the Project declares, with its control intent."""

    return tuple(
        RetainedDofStateV1(instance_id=component.instance_id, axis=dof.axis, control=dof.control,
                           **{name: getattr(dof, name) for name in DOF_LIMIT_FIELDS if getattr(dof, name) is not None})
        for component in project.components
        for dof in component.retained_dofs
    )


def _dof_key(key: tuple[str, DofAxisV1]) -> tuple[str, int]:
    return (key[0], dof_sort_index(key[1]))


def check_local_validation_binding(record: LocalValidationRecordV1, project: ProjectDocumentV2) -> tuple[str, ...]:
    """Structural consistency between a record and the Project it claims to validate. Empty tuple = consistent.

    Pure and local: usable before submission and on submission alike. Checks the project hash, that every catalog
    component has exactly one simulated part with the same SKU (and the same ``catalog_version`` when the Project pins
    one), that the target outcomes are exactly the Project's targets with matching metrics, and that the retained DOFs
    and control intents match (``input_hash`` / ``result_hash`` are Engine hashes, not compared here).
    """

    problems: list[str] = []
    expected_hash = project.content_hash()
    if record.project.project_hash.digest != expected_hash:
        problems.append(f"project_hash mismatch: record {record.project.project_hash.digest} != project {expected_hash}")

    products = {c.instance_id: c.product for c in project.components if c.product is not None}
    component_ids = {c.instance_id for c in project.components}
    for issue in record.simulation.model_scope_issues or ():
        for instance_id in {issue.instance_id, *(issue.affected_element_ids or ())} - component_ids:
            problems.append(f"model applicability refers to component {instance_id} absent from the project")
    simulated = {p.instance_id: p for p in record.simulated_parts}
    for instance_id in sorted(products.keys() - simulated.keys()):
        problems.append(f"catalog component {instance_id} has no simulated_parts entry")
    for instance_id in sorted(simulated.keys() - products.keys()):
        problems.append(f"simulated_parts entry {instance_id} is not a catalog component of the project")
    for instance_id in sorted(products.keys() & simulated.keys()):
        product, part = products[instance_id], simulated[instance_id]
        if part.optical_sku != product.optical_sku:
            problems.append(f"simulated part {instance_id} sku {part.optical_sku} != project sku {product.optical_sku}")
        if product.catalog_version is not None and part.catalog_version != product.catalog_version:
            problems.append(f"simulated part {instance_id} catalog_version {part.catalog_version} != pinned {product.catalog_version}")

    targets = {t.target_id: t.metric for t in project.targets}
    outcomes = {o.target_id: o.metric for o in record.simulation.target_outcomes}
    for target_id in sorted(targets.keys() - outcomes.keys()):
        problems.append(f"target {target_id} declared by the project has no outcome in the record")
    for target_id in sorted(outcomes.keys() - targets.keys()):
        problems.append(f"target outcome {target_id} in the record is not declared by the project")
    for target_id in sorted(targets.keys() & outcomes.keys()):
        if targets[target_id] != outcomes[target_id]:
            problems.append(f"target {target_id} metric {outcomes[target_id]} != declared {targets[target_id]}")

    if record.simulation.observation_quantities is not None:
        observations = {o.observation_id: o for o in project.observations}
        quantities = {q.observation_id: q for q in record.simulation.observation_quantities}
        if observations.keys() != quantities.keys():
            problems.append("observation quantities do not cover the project's observation planes")
        for oid in observations.keys() & quantities.keys():
            if (quantities[oid].quantity_state == "BOUNDED" and not observations[oid].absorbing
                    and "power_upper_bound_W" not in quantities[oid].model_fields_set):
                problems.append(f"observation {oid} claims a single-pass power bound on a transparent plane")

    declared = {(component.instance_id, dof.axis): dof for component in project.components for dof in component.retained_dofs}
    recorded = {(state.instance_id, state.axis): state for state in record.retained_dofs}
    for key in sorted(declared.keys() - recorded.keys(), key=_dof_key):
        problems.append(f"retained DOF {key[0]}/{key[1].value} declared by the project is missing from the record")
    for key in sorted(recorded.keys() - declared.keys(), key=_dof_key):
        problems.append(f"retained DOF {key[0]}/{key[1].value} in the record is not declared by the project")
    for key in sorted(declared.keys() & recorded.keys(), key=_dof_key):
        if declared[key].control is not recorded[key].control:
            problems.append(f"retained DOF {key[0]}/{key[1].value} control {recorded[key].control.value} != declared {declared[key].control.value}")
        differing = [name for name in DOF_LIMIT_FIELDS if getattr(declared[key], name) != getattr(recorded[key], name)]
        if differing:
            problems.append(f"retained DOF {key[0]}/{key[1].value} {', '.join(differing)} differ from the project's")
    return tuple(problems)


def resolve_simulated_parts(
    project: ProjectDocumentV2, catalog: PublicOpticalCatalogV2, *, model_kind: str | None = None
) -> tuple[SimulatedPartV1, ...]:
    """Resolve every catalog component of ``project`` against the public optical ``catalog``.

    Fails closed (``ValueError`` listing every problem) when a SKU is not published, when the Project pins a
    different ``catalog_version``, or when the product's simulation model is missing or ambiguous (pass
    ``model_kind`` to choose among several).
    """

    parts: list[SimulatedPartV1] = []
    problems: list[str] = []
    for component in project.components:
        ref = component.product
        if ref is None:
            continue
        product = catalog.get(ref.optical_sku)
        if product is None:
            problems.append(f"{component.instance_id}: {ref.optical_sku} is not in public catalog {catalog.catalog_version}")
            continue
        if ref.catalog_version is not None and ref.catalog_version != catalog.catalog_version:
            problems.append(f"{component.instance_id}: designed against catalog {ref.catalog_version}, resolving against {catalog.catalog_version}")
            continue
        models = [m for m in product.simulation_models if model_kind is None or m.model_kind == model_kind]
        if len(models) != 1:
            problems.append(
                f"{component.instance_id}: {len(models)} simulation models of {ref.optical_sku} match model_kind={model_kind!r}; need exactly one"
            )
            continue
        parts.append(
            SimulatedPartV1(
                instance_id=component.instance_id,
                optical_sku=ref.optical_sku,
                catalog_version=catalog.catalog_version,
                simulation_model=SimulationModelRefV1(model_kind=models[0].model_kind, model_version=models[0].model_version),
            )
        )
    if problems:
        raise ValueError("cannot resolve the project against the public optical catalog: " + "; ".join(problems))
    return tuple(parts)
