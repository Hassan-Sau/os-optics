"""Reading workspaces, modules and requests, and the content digests the composition operations use."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from .._vendor.shared_contracts.common.hashes import osc1_digest
from .._vendor.shared_contracts.optical_design import ModuleDescriptorV1, OpticalWorkspaceV1, ProjectDocumentV2
from ..local.catalog import _validation_message
from .errors import CompositionError, CompositionErrorCode, CompositionProblem, refuse

_E = CompositionErrorCode
WORKSPACE_SCHEMA = "opensci.optical_workspace"
PROJECT_SCHEMA = "opensci.project"


def _raw_document(source: Any, what: str, code: CompositionErrorCode) -> Any:
    if isinstance(source, (OpticalWorkspaceV1, ProjectDocumentV2)):
        return source.payload()
    if isinstance(source, BaseModel):
        return source.model_dump(mode="json", by_alias=True, exclude_none=True)
    if isinstance(source, Mapping):
        return copy.deepcopy(dict(source))
    if isinstance(source, memoryview):
        source = source.tobytes()
    if isinstance(source, str) and not source.lstrip().startswith("{"):
        source = Path(source)
    if isinstance(source, Path):
        try:
            source = source.read_bytes()
        except (OSError, ValueError) as exc:
            raise refuse(code, f"cannot read the {what} file: {exc}") from None
    if isinstance(source, str):
        source = source.encode("utf-8")
    if isinstance(source, (bytes, bytearray)):
        try:
            return json.loads(bytes(source))
        except (ValueError, RecursionError) as exc:
            raise refuse(code, f"the {what} is not valid JSON: {exc}") from None
    raise refuse(code, f"a {what} is a JSON object, JSON text, bytes or a file path, not a {type(source).__name__}")


def reference_problems(project: Any, code_prefix: str = "") -> list[CompositionProblem]:
    """The references of a raw Project document that do not resolve: components naming an undeclared optical model or a
    compound member naming one (``MODEL_MISSING``), and materials, coatings, devices, observations and sources that are
    not in the document (``REFERENCE_INVALID``). Lists every one, so a caller can correct them all at once."""

    problems: list[CompositionProblem] = []
    if not isinstance(project, Mapping):
        return problems

    def items(key: str) -> list[Mapping[str, Any]]:
        value = project.get(key) or []
        return [item for item in value if isinstance(item, Mapping)] if isinstance(value, list) else []

    models = {m.get("model_id"): m for m in items("optical_models")}
    materials = {m.get("material_id") for m in items("model_materials")}
    coatings = {c.get("coating_id") for c in items("model_coatings")}
    components = {c.get("instance_id"): c for c in items("components")}
    observations = {o.get("observation_id") for o in items("observations")}
    sources = {i for i, c in components.items() if c.get("optical_role") == "source"}
    where = f"{code_prefix}" if code_prefix else ""
    for instance_id, component in components.items():
        model_id = component.get("optical_model_id")
        if model_id is not None and model_id not in models:
            problems.append(CompositionProblem(_E.MODEL_MISSING, f"component {instance_id!r} uses optical model {model_id!r}, which the document "
                                               "does not declare", f"{where}component {instance_id}", {"model_id": model_id}))
    for model_id, entry in models.items():
        model = entry.get("model") if isinstance(entry.get("model"), Mapping) else {}
        for member in model.get("members") or []:
            if isinstance(member, Mapping) and member.get("model_id") not in models:
                problems.append(CompositionProblem(_E.MODEL_MISSING, f"compound unit {model_id!r} member {member.get('member_id')!r} uses optical "
                                                   f"model {member.get('model_id')!r}, which the document does not declare",
                                                   f"{where}optical model {model_id}", {"model_id": member.get("model_id")}))
        for name, known, label in (("material", materials, "material"), ("coating", coatings, "coating")):
            for value in sorted(_referenced(model, name)):
                if value not in known:
                    problems.append(CompositionProblem(_E.REFERENCE_INVALID, f"optical model {model_id!r} uses {label} {value!r}, which the document "
                                                       f"does not declare in model_{label}s", f"{where}optical model {model_id}",
                                                       {label + "_id": value}))
    for observation in items("observations"):
        device = observation.get("device_instance_id")
        if device is not None and device not in components:
            problems.append(CompositionProblem(_E.REFERENCE_INVALID, f"observation {observation.get('observation_id')!r} is attached to component "
                                               f"{device!r}, which the document does not have", f"{where}observation {observation.get('observation_id')}"))
    for target in items("targets"):
        tid = target.get("target_id")
        for key, known, label in (("device_instance_id", components, "component"), ("observation_id", observations, "observation"),
                                  ("source_id", sources, "source"), ("source_id_b", sources, "source")):
            value = target.get(key)
            if value is not None and value not in known:
                problems.append(CompositionProblem(_E.REFERENCE_INVALID, f"target {tid!r} {key} names {label} {value!r}, which the document does not have",
                                                   f"{where}target {tid}", {key: value}))
    return problems


MATERIAL_KEYS = ("medium_front", "medium_back", "medium")


def _referenced(model: Any, kind: str) -> set[str]:
    """Material or coating ids a raw model payload names."""

    found: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                if kind == "material" and key in MATERIAL_KEYS and isinstance(value, str) and value != "AIR":
                    found.add(value)
                elif kind == "material" and key == "media" and isinstance(value, list):
                    found.update(v for v in value if isinstance(v, str) and v != "AIR")
                elif kind == "coating" and key == "coating_id" and isinstance(value, str):
                    found.add(value)
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(model)
    return found


def _validated_workspace(document: Any, what: str, code: CompositionErrorCode) -> OpticalWorkspaceV1:
    if not isinstance(document, Mapping):
        raise refuse(code, f"a {what} is a JSON object")
    if document.get("schema") == PROJECT_SCHEMA:
        project_doc = document
        document = {"schema": WORKSPACE_SCHEMA, "schema_version": "1.0", "project": document}
    elif document.get("schema") == WORKSPACE_SCHEMA:
        project_doc = document.get("project")
    else:
        raise refuse(code, f"a {what} has schema {WORKSPACE_SCHEMA!r} (a workspace) or {PROJECT_SCHEMA!r} (a Project), "
                           f"not {document.get('schema')!r}")
    try:
        return OpticalWorkspaceV1.model_validate(document)
    except ValidationError as exc:
        problems = reference_problems(project_doc)
        if problems:
            raise CompositionError(problems) from None
        raise refuse(code, f"the {what} is not valid: {_validation_message(exc)}") from None
    except RecursionError:
        raise refuse(code, f"the {what} is nested too deeply") from None


def load_workspace(source: Any) -> OpticalWorkspaceV1:
    """An ``OpticalWorkspaceV1`` from a workspace or a Project (a model, a JSON document, JSON text / bytes or a file
    path). A Project is wrapped as a workspace without module, instances, groups, journal or validations. Refuses
    (``WORKSPACE_INVALID``, or ``MODEL_MISSING`` / ``REFERENCE_INVALID`` listing every unresolved reference) anything
    else."""

    if isinstance(source, OpticalWorkspaceV1):
        return source
    return _validated_workspace(_raw_document(source, "workspace", _E.WORKSPACE_INVALID), "workspace", _E.WORKSPACE_INVALID)


def load_module(source: Any) -> OpticalWorkspaceV1:
    """A module to import: a workspace (normally one written by ``export_module``) or a bare Project (``MODULE_INVALID``
    otherwise)."""

    if isinstance(source, OpticalWorkspaceV1):
        return source
    return _validated_workspace(_raw_document(source, "module", _E.MODULE_INVALID), "module", _E.MODULE_INVALID)


def new_workspace(project: Any, *, module: Any = None) -> OpticalWorkspaceV1:
    """A workspace holding ``project`` (a Project, or a workspace whose module descriptor is replaced when ``module`` is
    given), optionally with a module descriptor (``ModuleDescriptorV1`` or its JSON document)."""

    workspace = load_workspace(project)
    if module is None:
        return workspace
    try:
        descriptor = module if isinstance(module, ModuleDescriptorV1) else ModuleDescriptorV1.model_validate(module)
        return OpticalWorkspaceV1.model_validate({**workspace.payload(), "module": descriptor.model_dump(mode="json", exclude_none=True)})
    except ValidationError as exc:
        raise refuse(_E.REQUEST_INVALID, f"the module descriptor is not valid: {_validation_message(exc)}") from None


def validate_payload(payload: dict[str, Any]) -> OpticalWorkspaceV1:
    """The workspace a composition step produced (refused as ``REFERENCE_BROKEN`` / ``ID_CONFLICT`` / ``REQUEST_INVALID``
    when the contract refuses it)."""

    try:
        return OpticalWorkspaceV1.model_validate(payload)
    except ValidationError as exc:
        problems = reference_problems(payload.get("project"))
        if problems:
            raise CompositionError([CompositionProblem(_E.REFERENCE_BROKEN, p.message, p.location, p.details) for p in problems]) from None
        message = _validation_message(exc)
        code = _E.ID_CONFLICT if "duplicate" in message else _E.REFERENCE_BROKEN if "not in the Project" in message else _E.REQUEST_INVALID
        raise refuse(code, f"the resulting workspace is not valid: {message}") from None


def digest(value: Any) -> str:
    return osc1_digest(value)


def project_payload(workspace: OpticalWorkspaceV1) -> dict[str, Any]:
    return workspace.project.payload()


__all__ = ["load_module", "load_workspace", "new_workspace", "reference_problems", "validate_payload"]
