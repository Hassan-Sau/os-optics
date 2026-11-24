"""JSON-in / JSON-out composition calls (also re-exported by ``os_optics.local.json_api``).

Success: ``{"ok": True, "workspace", "workspace_digest", "replayed", "result"}`` for an operation that changes the
workspace; a read-only call returns its own fields. Refusal: ``{"ok": False, "error": {"code":
"WORKSPACE_OPERATION_REFUSED", "problems": [{"code", "message", "location"?, "details"?}]}, "workspace_digest"}`` (the
digest of the unchanged input, or ``None`` when it could not be read).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from . import operations as ops
from . import validation as val
from .document import load_workspace, new_workspace
from .errors import CompositionError, CompositionErrorCode, refuse

_E = CompositionErrorCode


def _refused(exc: CompositionError) -> dict[str, Any]:
    return {"ok": False, "error": exc.as_dict(), "workspace_digest": exc.workspace_digest}


def _guard(call: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        return call()
    except CompositionError as exc:
        return _refused(exc)
    except RecursionError:
        return _refused(refuse(_E.REQUEST_INVALID, "an input document is nested too deeply"))
    except (ValueError, TypeError, OverflowError) as exc:  # a value outside the JSON / number domain (e.g. not a JSON type)
        return _refused(refuse(_E.REQUEST_INVALID, f"an input holds a value that is not plain JSON or not in range: {exc}"))


def _operation(result: ops.OperationResult) -> dict[str, Any]:
    return {"ok": True, **result.as_dict()}


def workspace_init(document: Any, options: Any = None) -> dict[str, Any]:
    """A workspace from a Project (or a workspace to normalize); ``options.module``: a module descriptor to set."""

    def call() -> dict[str, Any]:
        if options is not None and (not isinstance(options, Mapping) or set(options) - {"module"}):
            raise refuse(_E.REQUEST_INVALID, "options is {module?}")
        ws = new_workspace(document, module=(options or {}).get("module"))
        return {"ok": True, "workspace": ws.payload(), "workspace_digest": ws.workspace_digest(), "module_digest": ws.module_digest()}

    return _guard(call)


def workspace_import(workspace: Any, module: Any, request: Any) -> dict[str, Any]:
    """Import a module as a new instance (``composition.import_layout``)."""

    return _guard(lambda: _operation(ops.import_layout(workspace, module, request)))


def workspace_select(workspace: Any, selector: Any) -> dict[str, Any]:
    """The objects a selector selects: ``{"ok": True, "selection": {components, observations, attached_observations, count}}``."""

    return _guard(lambda: {"ok": True, "selection": ops.select(workspace, selector)})


def workspace_transform(workspace: Any, request: Any) -> dict[str, Any]:
    """Translate / rotate a selection as one rigid body (``composition.transform``)."""

    return _guard(lambda: _operation(ops.transform(workspace, request)))


def workspace_group_create(workspace: Any, request: Any) -> dict[str, Any]:
    """Create an editing group (``composition.group_create``)."""

    return _guard(lambda: _operation(ops.group_create(workspace, request)))


def workspace_group_dissolve(workspace: Any, request: Any) -> dict[str, Any]:
    """Remove an editing group (``composition.group_dissolve``)."""

    return _guard(lambda: _operation(ops.group_dissolve(workspace, request)))


def workspace_group_set_members(workspace: Any, request: Any) -> dict[str, Any]:
    """Replace a group's members (``composition.group_set_members``)."""

    return _guard(lambda: _operation(ops.group_set_members(workspace, request)))


def workspace_group_add(workspace: Any, request: Any) -> dict[str, Any]:
    """Add members to a group (``composition.group_add``)."""

    return _guard(lambda: _operation(ops.group_add(workspace, request)))


def workspace_group_remove(workspace: Any, request: Any) -> dict[str, Any]:
    """Remove members from a group (``composition.group_remove``)."""

    return _guard(lambda: _operation(ops.group_remove(workspace, request)))


def workspace_replace_instance(workspace: Any, module: Any, request: Any) -> dict[str, Any]:
    """Replace an instance with another module revision (``composition.replace_instance``)."""

    return _guard(lambda: _operation(ops.replace_instance(workspace, module, request)))


def workspace_update_project(workspace: Any, request: Any) -> dict[str, Any]:
    """Replace the workspace's Project by a revision edited elsewhere (``composition.update_project``)."""

    return _guard(lambda: _operation(ops.update_project(workspace, request)))


def workspace_batch(workspace: Any, request: Any) -> dict[str, Any]:
    """Apply several operations atomically (``composition.batch``)."""

    return _guard(lambda: _operation(ops.batch(workspace, request)))


def workspace_export_module(workspace: Any, request: Any) -> dict[str, Any]:
    """The workspace as a reusable module: ``{"ok": True, "module", "module_digest"}``."""

    def call() -> dict[str, Any]:
        module = ops.export_module(workspace, request)
        return {"ok": True, "module": module.payload(), "module_digest": module.module_digest()}

    return _guard(call)


def workspace_run(workspace: Any, options: Any = None) -> dict[str, Any]:
    """Run the workspace's Project and append its system validation: ``{"ok": True, "workspace", "workspace_digest",
    "status", "outcome", "targets", "validation", "summary", "local_validation", "files"?}`` (``options``: ``{view_settings?,
    include_files?}``; ``files``: the run directory's ``{name: bytes}``). As in the reply of ``os-optics workspace run``,
    ``outcome`` (execution and the declared targets' requirements) and each target's ``assessment`` say what was decided;
    ``status`` is the diagnostic summary, never an approval of the design."""

    def call() -> dict[str, Any]:
        from ..local.artifacts import run_artifact_files
        from ..local.json_api import _view_settings, run_summary
        from ..local.errors import LocalRunError

        if options is not None and (not isinstance(options, Mapping) or set(options) - {"view_settings", "include_files"}):
            raise refuse(_E.REQUEST_INVALID, "options is {view_settings?, include_files?}")
        opts = dict(options or {})
        try:
            settings = _view_settings(opts.get("view_settings"))
        except LocalRunError as exc:
            raise refuse(_E.REQUEST_INVALID, "; ".join(p.message for p in exc.problems), "view_settings") from None
        ws, run, entry = val.run_workspace(workspace, view_settings=settings)
        record = run.local_validation.simulation
        summary = run_summary(run)
        # (the summary's rows, at its quantization: each value with its entry and failed checks, the record's assessment)
        reply: dict[str, Any] = {"ok": True, "workspace": ws.payload(), "workspace_digest": ws.workspace_digest(), "status": run.status.value,
                                 "outcome": None if record.outcome is None else record.outcome.model_dump(mode="json"),
                                 "targets": val.workspace_target_rows(summary["targets"]),
                                 "validation": entry, "summary": summary, "local_validation": run.local_validation.payload()}
        if opts.get("include_files"):
            reply["files"] = run_artifact_files(run)
        return reply

    return _guard(call)


def workspace_record_validation(workspace: Any, request: Any) -> dict[str, Any]:
    """Append the system validation of a run made elsewhere: ``{"ok": True, "workspace", "workspace_digest", "validation",
    "freshness"}``."""

    def call() -> dict[str, Any]:
        ws, entry = val.record_validation(workspace, request)
        identity = val.current_identity(ws)
        rows = val.freshness_rows(ws, identity)
        row = next(r for r in reversed(rows) if r["project_hash"] == entry["project_hash"] and r["engine_input_hash"] == entry["engine_input_hash"])
        return {"ok": True, "workspace": ws.payload(), "workspace_digest": ws.workspace_digest(), "validation": entry,
                "freshness": {"freshness": row["freshness"], "reasons": row["reasons"]}}

    return _guard(call)


def workspace_status(workspace: Any) -> dict[str, Any]:
    """``{"ok": True, "status": {...}}`` (``composition.workspace_status``)."""

    return _guard(lambda: {"ok": True, "status": val.workspace_status(workspace)})


def workspace_history(workspace: Any, query: Any = None) -> dict[str, Any]:
    """The operation journal, or whether operation ``query["op_id"]`` was applied: ``{"ok": True, "operations"}`` /
    ``{"ok": True, "found", "operation"}`` (``query``: ``{op_id?, limit?}``)."""

    def call() -> dict[str, Any]:
        if query is not None and (not isinstance(query, Mapping) or set(query) - {"op_id", "limit"}):
            raise refuse(_E.REQUEST_INVALID, "query is {op_id?, limit?}")
        q = dict(query or {})
        return {"ok": True, **ops.history(workspace, op_id=q.get("op_id"), limit=q.get("limit"))}

    return _guard(call)


#: The composition functions of ``json_api`` (name -> function).
WORKSPACE_FUNCTIONS = {
    "workspace_init": workspace_init,
    "workspace_import": workspace_import,
    "workspace_select": workspace_select,
    "workspace_transform": workspace_transform,
    "workspace_group_create": workspace_group_create,
    "workspace_group_dissolve": workspace_group_dissolve,
    "workspace_group_set_members": workspace_group_set_members,
    "workspace_group_add": workspace_group_add,
    "workspace_group_remove": workspace_group_remove,
    "workspace_replace_instance": workspace_replace_instance,
    "workspace_update_project": workspace_update_project,
    "workspace_batch": workspace_batch,
    "workspace_export_module": workspace_export_module,
    "workspace_run": workspace_run,
    "workspace_record_validation": workspace_record_validation,
    "workspace_status": workspace_status,
    "workspace_history": workspace_history,
}

__all__ = ["WORKSPACE_FUNCTIONS", *WORKSPACE_FUNCTIONS]

