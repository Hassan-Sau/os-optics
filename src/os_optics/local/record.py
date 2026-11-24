"""Adapt a real Engine ``ValidationResult`` to the accepted ``LocalValidationRecordV1``.

Every hash and version comes from the run itself, never from a constant:

* ``project.project_hash``: osc1 of the validated ``ProjectDocumentV2`` payload;
* ``input_hash``: the Engine manifest ``input_hash`` (``eng-exact-py1``). It must equal the hash of the
  request the lowering built, otherwise the adapter refuses to write the record;
* ``result_hash``: the Engine manifest ``result_hash`` (``eng-quant-py1``) with the ``sig_digits`` /
  ``abs_floor`` actually used by the run's numerical configuration;
* ``runtime``: the manifest's Engine version, build id (version + digest of the installed sources),
  schema version and numerical-config version;
* ``simulated_parts``: the public optical SKU, catalog release and simulation model the lowering used;
* ``reproducibility``: the manifest's seed / PRNG / model versions.

``simulation.status`` is the Engine's overall status, unchanged: an ``OUT_OF_MODEL_SCOPE``, ``UNKNOWN`` or
``FAIL`` run is recorded as exactly that. ``validation_status`` equals it, except that a run outside a
published model's validity (``validity.py``) is at least ``OUT_OF_MODEL_SCOPE`` -- never ``PASS``.
"""

from __future__ import annotations

import sys

from .._vendor.shared_contracts.common.hashes import HashProfile, HashRef
from .._vendor.shared_contracts.design_catalog.optical_catalog import parse_query_decimal
from .._vendor.shared_contracts.optical_design import (
    IssueCountV1,
    LocalRuntimeIdentityV1,
    LocalRuntimeKindV1,
    LocalValidationRecordV1,
    LocalValidationStatusV1,
    ModelVersionEntryV1,
    ProjectDocumentV2,
    ReproducibilityV1,
    SimulationSummaryV1,
    TargetOutcomeV1,
    nominal_retained_dof_states,
    validated_project_ref,
)
from .._vendor.shared_contracts.optical_design.local_validation import status_severity
from .._vendor.shared_contracts.optical_design.scientific_result import ModelScopeApplicabilityV1, ObservationQuantityV1, ResultOutcomeV1
from .._version import DISTRIBUTION_NAME
from ..contracts.results import ValidationResult
from ..errors import QuantityState, assessment_of
from .errors import LocalRunErrorCode, fail
from .lowering import LoweredProject

BROWSER_PLATFORM = "emscripten"


def detect_runtime_kind() -> LocalRuntimeKindV1:
    """``BROWSER_PYTHON`` inside Pyodide (Emscripten), ``PYTHON_LOCAL`` otherwise."""

    return LocalRuntimeKindV1.BROWSER_PYTHON if sys.platform == BROWSER_PLATFORM else LocalRuntimeKindV1.PYTHON_LOCAL


def _plain_decimal(value: float) -> str:
    """``1e-12`` -> ``"0.000000000001"``: the exact decimal of the float's shortest repr, never exponent notation."""

    return format(parse_query_decimal(repr(value), "abs_floor"), "f")


#: The fields of a model-scope finding that stay with the run's own listing (its summary and manifest): its structured facts
#: (``details``, ``validity``), the arrivals and sampled lines it reaches (``affected_arrivals``, ``affected_sampled_lines``)
#: and the light of each check (``validity_reach``).
LISTING_ONLY_FIELDS = frozenset({"details", "validity", "affected_arrivals", "affected_sampled_lines", "validity_reach"})


def applicability_fields(issue) -> dict:
    """The fields of a model-scope finding (a ``ModelScopeIssue`` or its row) that the Local Validation record holds
    (``ModelScopeApplicabilityV1``): every field but ``LISTING_ONLY_FIELDS``."""

    row = issue.as_dict() if hasattr(issue, "as_dict") else issue
    return {key: value for key, value in row.items() if key not in LISTING_ONLY_FIELDS}


def target_row_bases(target, scope_issues=()) -> dict:
    """The ``value_basis`` member of a target's row (``results.basis.row_bases``): its value's entry and, where it has a value
    and model-scope findings name it (``affected_target_ids``), an ``ESTIMATE`` with the validity checks they state beside
    it -- the value is the model's number outside a published model's validity. ``scope_issues``: ``ModelScopeIssue`` objects
    or their rows; a row without its checks (a record's) stands for one check of its code at its element. Of a finding's
    checks the row keeps those whose light the target reads (``validity_reach``: the wavelength, angle or intercepted rays
    its own beams met the element at); a finding that does not say per check (a run of an earlier Engine), or whose checks
    name none of the target's light, gives all of its checks."""

    from ..contracts.basis import ValidityCheck, ValueBasis, merge_bases
    from ..errors.status import ValueKind
    from ..results.basis import basis_of

    entry = basis_of(target, "value")
    if target.value is not None:
        checks = []
        for issue in scope_issues:
            row = issue.as_dict() if hasattr(issue, "as_dict") else (issue.model_dump(mode="json") if hasattr(issue, "model_dump") else issue)
            if target.target_id in (row.get("affected_target_ids") or ()):
                stated = [ValidityCheck.model_validate(c) for c in row.get("validity") or ()]
                reach = row.get("validity_reach") or ()
                if len(reach) == len(stated):
                    stated = [c for c, light in zip(stated, reach, strict=True)
                              if target.target_id in (light.get("affected_target_ids") or ())] or stated
                checks += stated or [ValidityCheck(parameter=row["code"].lower(), within=False, at=(row["instance_id"],), code=row["code"])]
        if checks:
            entry = merge_bases(entry, ValueBasis(kind=ValueKind.ESTIMATE, validity=tuple(checks)), monotone=None)
    return {"value_basis": {"value": entry.model_dump(mode="json")}} if entry is not None else {}


def scientific_projection(result: ValidationResult, scope_issues=()) -> SimulationSummaryV1:
    """Apply explicit local model applicability without modifying/hash-replacing the raw result."""
    scope = tuple(i if isinstance(i, ModelScopeApplicabilityV1) else ModelScopeApplicabilityV1.model_validate(applicability_fields(i))
                  for i in scope_issues)
    targets = []
    for target in result.targets:
        codes = tuple(sorted({i.code for i in scope if target.target_id in i.affected_target_ids}))
        targets.append(TargetOutcomeV1(
            target_id=target.target_id, metric=target.metric, status=target.status.value, value=target.value, unit=target.unit or None,
            assessment=assessment_of(target.status, codes).value if codes or target.assessment is not None else None,
            **({"applicability_issue_codes": codes} if codes else {})))
    outcome = None if result.outcome is None else result.outcome.model_dump(mode="json")
    if outcome is not None:
        outcome["requirements"] = {"declared": outcome["requirements"]["declared"], **{
            key: [t.target_id for t in targets if t.assessment == assessment]
            for key, assessment in (("met", "MET"), ("unmet", "UNMET"), ("indeterminate", "INDETERMINATE"),
                                    ("not_applicable", "NOT_APPLICABLE"))}}
    observations = []
    for o in result.observations:
        codes = {i.code for i in scope if o.observation_id in i.affected_observation_ids}
        # (NOT_COMPUTED -- an evaluation without the plane bounds -- is the Engine's own state: a record is made of complete
        # runs only, and the results read it from the Engine result: results/model.py _quantity_state)
        state = None if o.quantity_state in (None, QuantityState.NOT_COMPUTED) else o.quantity_state.value
        quantity = dict(
            observation_id=o.observation_id,
            quantity_state="OUT_OF_MODEL_SCOPE" if codes else state,
            total_power_W=o.total_power_W, power_complete=o.power_complete, untraced_power_W=o.untraced_power_W,
            **({"power_upper_bound_W": o.power_upper_bound_W} if "power_upper_bound_W" in o.model_fields_set else {}),
            # (the record states a lower bound beside a value only: a dark plane's interval is its null value and its upper bound)
            **({"power_lower_bound_W": o.power_lower_bound_W}
               if o.power_lower_bound_W is not None and o.total_power_W is not None and not codes else {}),
            **({"power_sampling_resolution_W": o.power_sampling_resolution_W}
               if o.power_sampling_resolution_W is not None and not codes else {}),
            issue_codes=tuple(sorted(codes | {i.code.value for i in result.issues if i.location == f"observation:{o.observation_id}"})))
        # (a plane whose bounds did not run states no untraced power -- null, which the record's contract does not hold: such
        # a summary is read by the results, never recorded)
        observations.append(ObservationQuantityV1(**quantity) if o.untraced_power_W is not None
                            else ObservationQuantityV1.model_construct(**quantity))
    fields = dict(
        status=result.status.value, outcome=outcome, observation_quantities=observations, target_outcomes=targets,
        model_scope_issues=scope,
        issue_counts=[IssueCountV1(code=code, count=count) for code, count in result.summary.issue_counts.items() if count > 0],
        path_count=result.summary.path_count, observation_count=result.summary.observation_count)
    if result.evaluation is not None:
        # (a selective evaluation lists the targets it evaluated, not every declared one: its summary is read by the
        # results, never recorded -- only a complete run makes a Local Validation record)
        return SimulationSummaryV1.model_construct(
            **{**fields, "outcome": None if outcome is None else ResultOutcomeV1.model_validate(outcome)})
    return SimulationSummaryV1(**fields)


def validation_status_of(result: ValidationResult, outside_model_scope: bool) -> LocalValidationStatusV1:
    """The diagnostic summary of a run: the Engine status, made at least ``OUT_OF_MODEL_SCOPE`` when a model-scope finding
    exists."""

    status = LocalValidationStatusV1(result.status.value)
    if outside_model_scope and status_severity(status) < status_severity(LocalValidationStatusV1.OUT_OF_MODEL_SCOPE):
        status = LocalValidationStatusV1.OUT_OF_MODEL_SCOPE
    return status


def local_validation_record(
    project: ProjectDocumentV2,
    lowered: LoweredProject,
    result: ValidationResult,
    *,
    runtime_kind: LocalRuntimeKindV1 | str | None = None,
    outside_model_scope: bool = False,
    scope_issues=(),
) -> LocalValidationRecordV1:
    if result.evaluation is not None:
        raise fail(LocalRunErrorCode.BINDING_INCONSISTENT, "a Local Validation record is made of a complete evaluation; this "
                   "result states a narrowed scope (result.evaluation): complete it with Evaluation.run()")
    manifest = result.manifest
    if manifest is None:
        raise fail(LocalRunErrorCode.BINDING_INCONSISTENT, "the Engine result carries no manifest")
    if manifest.input_hash != lowered.engine_input_hash:
        raise fail(LocalRunErrorCode.BINDING_INCONSISTENT, "the Engine manifest input_hash is not the hash of the lowered request")
    numerical = lowered.request.configuration.numerical
    validation_status = validation_status_of(result, outside_model_scope)
    return LocalValidationRecordV1(
        schema="opensci.local_validation",
        schema_version="1.0",
        project=validated_project_ref(project),
        runtime=LocalRuntimeIdentityV1(
            runtime_kind=LocalRuntimeKindV1(runtime_kind) if runtime_kind is not None else detect_runtime_kind(),
            engine_package=DISTRIBUTION_NAME,
            engine_version=manifest.engine_version,
            engine_build_id=manifest.engine_build_id,
            engine_schema_version=manifest.schema_version,
            numerical_config_version=manifest.numerical_config_version,
        ),
        simulated_parts=lowered.simulated_parts,
        input_hash=HashRef(profile=HashProfile.ENGINE_EXACT_PY1, digest=manifest.input_hash, subject="engine_input"),
        result_hash=HashRef(
            profile=HashProfile.ENGINE_QUANTIZED_PY1,
            digest=manifest.result_hash,
            subject="engine_result",
            params={"sig_digits": str(numerical.hash_float_sig_digits), "abs_floor": _plain_decimal(numerical.hash_float_abs_floor)},
        ),
        simulation=scientific_projection(result, scope_issues),
        validation_status=validation_status,
        retained_dofs=nominal_retained_dof_states(project),
        reproducibility=ReproducibilityV1(
            deterministic=manifest.random_seed is None,
            random_seed=manifest.random_seed,
            prng_id=manifest.prng_id if manifest.random_seed is not None else None,
            model_versions=[ModelVersionEntryV1(key=key, version=version) for key, version in manifest.model_versions.items()],
        ),
    )
