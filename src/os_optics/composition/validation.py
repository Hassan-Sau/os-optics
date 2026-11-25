"""Validations of a workspace and their freshness.

``run_workspace`` runs the workspace's Project through the normal local run (``os_optics.local.run_optical_project``:
the same Engine, the same lowering, the same Local Validation record as any Project) and appends a ``SYSTEM``
``ValidationEntryV1``; ``record_validation`` appends one for an externally run Project
from its ``local_validation.json``. Freshness is derived, never stored (``status``): a ``SYSTEM`` validation is ``CURRENT``
exactly while the Project hash and the Engine input hash are the ones it ran on; otherwise it is ``HISTORICAL`` with the
reasons (``PHYSICS_CHANGED``, ``TARGETS_CHANGED``, ``CATALOG_CHANGED``, ``ENGINE_CHANGED``). Moving an element, changing a
fixture binding or a target makes earlier validations historical; regrouping does not (groups are not part of the
Project). A module's own validations travel with its instance and are never current for the system (``MODULE_SCOPE``).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .._vendor.shared_contracts.optical_design import LocalValidationRecordV1, OpticalWorkspaceV1, project_targets_digest, validation_freshness
from .._vendor.shared_contracts.optical_design.workspace import MAX_VALIDATIONS
from ..local.catalog import _validation_message
from ..local.errors import LocalRunError
from ..manifest import engine_build_id
from ..manifest.canonical import hash_exact
from .document import load_workspace, validate_payload
from .errors import CompositionError, CompositionErrorCode, CompositionProblem, refuse
from .importing import instance_ports
from .operations import edited_members, kept_definitions
from .selection import instance_members

_E = CompositionErrorCode
PROJECTION_FIELDS = ("baseplate", "components", "observations", "materials")
#: Characters of a SHA-256 digest (hexadecimal).
DIGEST_LENGTH = 64


def design_projection_hash(request: Any) -> str:
    """The Engine's ``design_projection_hash`` of a lowered request: the physics without the targets (as the Engine's
    validation manifest computes it)."""

    dump = request.project.model_dump(mode="json")
    return hash_exact({k: dump[k] for k in PROJECTION_FIELDS})


def physics_input_hash(request: Any) -> str:
    """The identity of a lowered request's physics: the exact hash of the request (a ``ValidationRequest`` or its JSON
    document) without its targets -- every component, observation plane and material, the whole numerical configuration
    (budgets and settings) and the schema stamp. Two requests that differ only in their targets have the same one; any
    other difference of the Engine input gives another."""

    dump = request.model_dump(mode="json") if hasattr(request, "model_dump") else dict(request)
    return hash_exact({**dump, "project": {key: value for key, value in dump["project"].items() if key != "targets"}})


def current_identity(workspace: OpticalWorkspaceV1, *, catalog: Any = None) -> dict[str, Any]:
    """The identity a validation is compared with: Project hash, Engine input hash, design projection hash, targets
    digest and Engine build id, and the physics input hash (``physics_input_hash`` of the lowered request; ``None`` when
    the Project does not lower); ``check``: whether the Project lowers (``problems`` when it does not, and then the Engine
    input hash is ``None``) and the advisory ``warnings`` of an accepted Project (``os_optics.local.advisories``)."""

    from ..local.advisories import project_warnings
    from ..local.lowering import lower_project

    project = workspace.project
    identity: dict[str, Any] = {"project_hash": project.content_hash(), "engine_input_hash": None, "design_projection_hash": None,
                                "physics_input_hash": None,
                                "targets_digest": project_targets_digest(project), "engine_build_id": engine_build_id()}
    try:
        lowered = lower_project(project, catalog)
    except LocalRunError as exc:
        identity["check"] = {"ok": False, "problems": [p.as_dict() for p in exc.problems], "warnings": []}
        return identity
    identity.update(engine_input_hash=lowered.engine_input_hash, design_projection_hash=design_projection_hash(lowered.request),
                    physics_input_hash=physics_input_hash(lowered.request),
                    check={"ok": True, "problems": [], "warnings": project_warnings(project, lowered)})
    return identity


def _append(workspace: OpticalWorkspaceV1, entry: dict[str, Any]) -> tuple[OpticalWorkspaceV1, bool]:
    payload = workspace.payload()
    validations = list(payload.get("validations") or [])
    if entry in validations:
        return workspace, False
    validations.append(entry)
    payload["validations"] = validations[-MAX_VALIDATIONS:]
    return validate_payload(payload), True


def _entry(record: LocalValidationRecordV1, projection: str | None, targets_digest: str) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "scope": "SYSTEM",
        "project_hash": record.project.project_hash.digest,
        "engine_input_hash": record.input_hash.digest,
        "targets_digest": targets_digest,
        "engine_build_id": record.runtime.engine_build_id,
        "validation_status": record.validation_status.value,
        "target_outcomes": [{"target_id": o.target_id, "status": o.status.value,
                             **({"assessment": o.assessment} if o.assessment is not None else {}),
                             **({"applicability_issue_codes": list(o.applicability_issue_codes)} if o.applicability_issue_codes else {})}
                            for o in record.simulation.target_outcomes],
        "runtime_kind": record.runtime.runtime_kind.value,
    }
    if projection is not None:
        entry["design_projection_hash"] = projection
    if record.simulation.outcome is not None:
        entry["outcome"] = record.simulation.outcome.model_dump(mode="json", exclude_none=True)
    if record.simulation.observation_quantities is not None:
        entry["observation_quantities"] = [q.model_dump(mode="json", exclude_none=True) for q in record.simulation.observation_quantities]
    if record.simulation.model_scope_issues is not None:
        entry["model_scope_issues"] = [i.model_dump(mode="json", exclude_none=True) for i in record.simulation.model_scope_issues]
    return entry


def run_workspace(workspace: Any, *, catalog: Any = None, output_dir: Path | str | None = None, view_settings: Any = None
                  ) -> tuple[OpticalWorkspaceV1, Any, dict[str, Any]]:
    """Run the workspace's Project (``run_optical_project``; ``output_dir`` receives the run directory) and append its
    ``SYSTEM`` validation. Returns ``(workspace, run, entry)``; a Project the local run refuses raises
    ``CompositionError`` (``RUN_REFUSED``, with the run's problems in ``details.problems``)."""

    from ..local.run import run_optical_project

    ws = load_workspace(workspace)
    try:
        run = run_optical_project(ws.project, catalog=catalog, output_dir=output_dir, view_settings=view_settings)
    except LocalRunError as exc:
        raise CompositionError([CompositionProblem(_E.RUN_REFUSED, "the local run refused the Project: "
                                                   + "; ".join(f"{p.code.value}: {p.message}" for p in exc.problems[:5]), None,
                                                   {"problems": [p.as_dict() for p in exc.problems]})],
                               workspace_digest=ws.workspace_digest()) from None
    manifest = run.result.manifest
    entry = _entry(run.local_validation, manifest.design_projection_hash if manifest is not None else None, project_targets_digest(ws.project))
    new_ws, _added = _append(ws, entry)
    return new_ws, run, entry


#: The members of a run summary's target row a workspace run reply carries: the value with its entry (``value_basis``),
#: its error against ``target_value`` (``error``: one fact with the value, W5) and its failed checks (``validity``), as
#: the summary states them.
WORKSPACE_TARGET_KEYS = ("target_id", "status", "assessment", "value", "unit", "target_value", "error", "value_basis", "validity")


def workspace_target_rows(summary_targets: Any) -> list[dict[str, Any]]:
    """The ``targets`` of a workspace run reply (``os-optics workspace run``, ``json_api.workspace_run``) from the run
    summary's target rows (``LocalOpticalRun.summary``, ``json_api.run_summary``): the effective assessment, and a value
    that is a bound or an estimate with its entry under ``value_basis`` (B8: a number without an entry is the model's
    value), its error against ``target_value`` beside it (``error``, W5) and its failed checks under ``validity``."""

    return [{key: row[key] for key in WORKSPACE_TARGET_KEYS if key in row} for row in summary_targets]


def record_validation(workspace: Any, request: Any, *, catalog: Any = None) -> tuple[OpticalWorkspaceV1, dict[str, Any]]:
    """Append the ``SYSTEM`` validation of a run made elsewhere: ``request["local_validation"]`` is its Local Validation
    record (``local_validation.json``); ``request["design_projection_hash"]`` (optional) the run manifest's. The record
    must have run on this workspace's Project (``VALIDATION_PROJECT_MISMATCH`` otherwise). An identical entry is not
    added twice."""

    ws = load_workspace(workspace)
    try:
        if not isinstance(request, Mapping) or "local_validation" not in request or set(request) - {"local_validation", "design_projection_hash"}:
            raise refuse(_E.REQUEST_INVALID, "a record request is {local_validation, design_projection_hash?}")
        try:
            record = LocalValidationRecordV1.model_validate(request["local_validation"])
        except ValidationError as exc:
            raise refuse(_E.REQUEST_INVALID, f"local_validation is not a Local Validation record: {_validation_message(exc)}",
                         "local_validation") from None
        project_hash = ws.project.content_hash()
        if record.project.project_hash.digest != project_hash:
            raise refuse(_E.VALIDATION_PROJECT_MISMATCH, "the record validated another Project (its project_hash differs from the workspace's)",
                         "local_validation", record_project_hash=record.project.project_hash.digest, project_hash=project_hash)
        projection = request.get("design_projection_hash")
        if projection is not None and not (isinstance(projection, str) and len(projection) == DIGEST_LENGTH):
            raise refuse(_E.REQUEST_INVALID, "design_projection_hash is a 64-character digest", "design_projection_hash")
        if projection is None:
            identity = current_identity(ws, catalog=catalog)
            if identity["engine_input_hash"] == record.input_hash.digest:
                projection = identity["design_projection_hash"]
        entry = _entry(record, projection, project_targets_digest(ws.project))
        new_ws, _added = _append(ws, entry)
        return new_ws, entry
    except CompositionError as exc:
        exc.workspace_digest = ws.workspace_digest()
        raise


def freshness_rows(workspace: OpticalWorkspaceV1, identity: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for index, entry in enumerate(workspace.validations):
        unchecked = identity["engine_input_hash"] is None
        if unchecked and entry.scope == "SYSTEM" and entry.project_hash == identity["project_hash"]:
            state, reasons = "UNDETERMINED", ["PROJECT_NOT_CHECKED"]  # the same Project, but its Engine input is unknown now
        else:
            fresh, why = validation_freshness(entry, project_hash=identity["project_hash"],
                                              engine_input_hash=entry.engine_input_hash if unchecked else identity["engine_input_hash"],
                                              design_projection_hash=identity["design_projection_hash"], targets_digest=identity["targets_digest"],
                                              engine_build_id=identity["engine_build_id"])
            state, reasons = fresh.value, [r.value for r in why]
        rows.append({"index": index, "scope": entry.scope, "validation_status": entry.validation_status.value, "freshness": state,
                     "reasons": reasons, "project_hash": entry.project_hash, "engine_input_hash": entry.engine_input_hash,
                     "engine_build_id": entry.engine_build_id, "runtime_kind": entry.runtime_kind.value,
                     "target_outcomes": [o.model_dump(mode="json", exclude_none=True) for o in entry.target_outcomes],
                     "outcome": None if entry.outcome is None else entry.outcome.model_dump(mode="json"),
                     "model_scope_issues": None if entry.model_scope_issues is None else [i.model_dump(mode="json", exclude_none=True) for i in entry.model_scope_issues],
                     "observation_quantities": None if entry.observation_quantities is None else
                         [q.model_dump(mode="json") for q in entry.observation_quantities]})
    return rows


def workspace_status(workspace: Any, *, catalog: Any = None) -> dict[str, Any]:
    """Everything an Agent needs to know about a workspace: its digests and the current Project / Engine identity, whether
    the Project passes the local check, each validation with its freshness (``current_validation``: the index of the
    latest ``CURRENT`` one, or ``None``), each instance (source, placement, the members the Project has, its local edits
    and those of the instances nested in it -- what replacing it would discard (``edited``: ``EDITED`` / ``REMOVED`` /
    ``ADDED``, ``edited_members``) --, the definitions a replacement keeps because objects outside it use them
    (``kept_definitions``), fixture outcomes, interface ports in the Project's design frame
    at the instance's current placement, module validations), each group and the module descriptor."""

    ws = load_workspace(workspace)
    identity = current_identity(ws, catalog=catalog)
    rows = freshness_rows(ws, identity)
    current = [r["index"] for r in rows if r["freshness"] == "CURRENT"]
    instances = []
    targets = {t.target_id for t in ws.project.targets}
    components = {c.instance_id for c in ws.project.components}
    for instance in ws.instances:
        members_c, members_o = instance_members(ws, instance.instance_key)
        instances.append({
            "instance_key": instance.instance_key,
            "source": instance.source.model_dump(mode="json", exclude_none=True),
            "placement": instance.placement.model_dump(mode="json"),
            "components": sorted(members_c),
            "observations": sorted(members_o),
            "targets": sorted(e.parent_id for e in instance.id_map if e.kind == "TARGET" and e.parent_id in targets),
            "shared_definitions": sorted(f"{e.kind}:{e.parent_id}" for e in instance.id_map if e.shared),
            "edited": edited_members(ws, instance.instance_key),
            "kept_definitions": kept_definitions(ws, instance.instance_key),
            "fixtures": [f.model_dump(mode="json", exclude_none=True) for f in instance.fixtures],
            "interfaces": instance_ports(instance, components),
            "module_validations": [{"validation_status": v.validation_status.value, "freshness": "HISTORICAL", "reasons": ["MODULE_SCOPE"],
                                    "project_hash": v.project_hash, "module_digest": v.module_digest,
                                    "outcome": None if v.outcome is None else v.outcome.model_dump(mode="json")}
                                   for v in instance.module_validations],
        })
    return {
        "workspace_digest": ws.workspace_digest(),
        "module_digest": ws.module_digest(),
        **identity,
        "validations": rows,
        "current_validation": current[-1] if current else None,
        "instances": instances,
        "groups": [g.model_dump(mode="json", exclude_none=True) for g in ws.groups],
        "module": ws.module.model_dump(mode="json", exclude_none=True) if ws.module is not None else None,
        "operations": len(ws.operations),
    }


__all__ = ["current_identity", "design_projection_hash", "physics_input_hash", "record_validation", "run_workspace", "workspace_status",
           "workspace_target_rows"]
