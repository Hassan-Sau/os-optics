"""Composing optical systems from reusable layouts: ``OpticalWorkspaceV1`` operations.

A workspace (``shared_contracts.optical_design.OpticalWorkspaceV1``, ``schema "opensci.optical_workspace"``) is a Project
together with its module description, imported instances, editing groups, operation journal and validation history. The
Project inside is an ordinary ``ProjectDocumentV2``: it runs, validates and exports exactly as on its own, and its hash
never depends on groups, instances or the journal.

Workflow:

1. Author each module as an ordinary Project; ``init_workspace(project, module=...)`` marks its test fixtures, interfaces
   and requirement scopes; ``run_workspace`` validates it on its own; ``export_module`` writes the reusable module.
2. ``import_layout(workspace, module, {"instance_key", "placement", "fixtures", ...})`` copies a module into a system,
   as many times as needed (one ``instance_key`` per instance), with namespaced ids (also of the models, materials and
   coatings each instance copies for itself), the module's poses composed with the placement, fixtures retained /
   removed / rebound and the module's requirements carried; it creates one editing group per import.
3. ``select`` / ``group_*`` organize the objects (groups may overlap, span imports and include directly authored
   elements; they never change the Project); ``transform`` moves and rotates any selection as one rigid body about an
   explicit frame and pivot; ``replace_instance`` swaps an unedited instance for a new module revision.
4. ``run_workspace`` validates the whole system through the normal local run; ``workspace_status`` says which
   validations are current. ``export_module`` turns a composed workspace into a module again.

Every changing operation is atomic and retry-safe (``op_id``, ``expected_base``; ``operations`` module docstring) and
returns structured results; refusals raise ``CompositionError`` with stable codes (``CompositionErrorCode``). The
rotation arithmetic uses ``+ - * /`` and ``sqrt`` only (``rotation``), so Python and the browser runtime write
identical Projects. ``os_optics.local.json_api`` exposes the same operations as ``workspace_*`` JSON functions, and
``os-optics workspace ...`` as commands.
"""

from __future__ import annotations

from .document import load_module, load_workspace, new_workspace
from .errors import WORKSPACE_OPERATION_REFUSED, CompositionError, CompositionErrorCode, CompositionProblem, CompositionWarningCode
from .operations import (
    BATCH_KINDS,
    OperationResult,
    batch,
    edited_members,
    export_module,
    group_add,
    group_create,
    group_dissolve,
    group_remove,
    group_set_members,
    history,
    import_layout,
    kept_definitions,
    replace_instance,
    select,
    transform,
    update_project,
)
from .rotation import axis_angle_quaternion, cos_sin_deg
from .validation import current_identity, record_validation, run_workspace, workspace_status


def init_workspace(project: object, *, module: object = None):
    """A workspace holding ``project`` (a Project, or a workspace to normalize), optionally with a module descriptor
    (``ModuleDescriptorV1`` or its JSON document): its test fixtures, interfaces and requirement scopes."""

    return new_workspace(project, module=module)


__all__ = [
    "BATCH_KINDS",
    "WORKSPACE_OPERATION_REFUSED",
    "CompositionError",
    "CompositionErrorCode",
    "CompositionProblem",
    "CompositionWarningCode",
    "OperationResult",
    "axis_angle_quaternion",
    "batch",
    "cos_sin_deg",
    "current_identity",
    "edited_members",
    "export_module",
    "group_add",
    "group_create",
    "group_dissolve",
    "group_remove",
    "group_set_members",
    "history",
    "import_layout",
    "init_workspace",
    "kept_definitions",
    "load_module",
    "load_workspace",
    "record_validation",
    "replace_instance",
    "run_workspace",
    "select",
    "transform",
    "update_project",
    "workspace_status",
]
