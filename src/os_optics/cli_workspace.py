"""``os-optics workspace``: compose optical systems from reusable layouts (``os_optics.composition``).

Each command reads a workspace file (a bare Project file is read as a workspace), applies one operation and writes the
resulting workspace back (or to ``--out``) in one step (a complete new file replaces the old one); a refused operation
writes nothing. Results are printed as JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .cli_exit import DEFAULT_STATUS_CONTRACT, add_status_contract_argument, run_exit_code

USAGE_ERROR = 64
REFUSED = 65
WORKSPACE_DESCRIPTION = (
    "Compose optical systems from reusable layouts. A workspace is a Project with its module description (test fixtures, "
    "interfaces, requirement scopes), imported instances, editing groups, operation journal and validations. Import "
    "modules as independent instances, select and group their elements, move or rotate any selection as one rigid body "
    "about an explicit frame and pivot, run the whole system and see which validations are current. Every change is "
    "atomic: a refused operation writes nothing. An operation is applied whatever the local check says of its result; the "
    "reply's check states whether a run accepts the result (ok), the problems the operation introduced (new_problems) and "
    "the ones the workspace already had (carried_problems). --require-check refuses an operation that introduces a problem."
)
WORKSPACE_EPILOG = (
    "Retries: give each operation an --op-id and repeat it unchanged after a lost response; the same operation is never "
    "applied twice (a reused op id with another request is refused). Without --op-id every command is applied as a new "
    "operation. --expect-base DIGEST refuses the operation when the "
    "workspace changed since you read that digest. --request FILE gives the whole request as JSON (options on the command "
    "line override it). Exit codes: 0 = applied (or already applied), 64 = usage error or unwritable file, 65 = refused "
    "(JSON error on stdout); 'workspace run' follows --status-contract (default requirements, or OS_OPTICS_STATUS_CONTRACT): "
    "0 = the calculation completed and every declared target is met (or none is declared), 1 = a declared target is not "
    "met, 2 = a declared target is undetermined or does not apply and none is unmet, 3 = the calculation did not complete "
    "(PARTIAL, FAILED or CANCELLED) (execution: 0 = completed, 1 = anything else; aggregate: 0 = PASS / WARNING / "
    "NOT_APPLICABLE, 1 = FAIL, 2 = UNKNOWN, 3 = OUT_OF_MODEL_SCOPE); 65 = the Project cannot be run; 130 = Ctrl-C."
)


class _Usage(Exception):
    pass


def _out(doc: object) -> None:
    print(json.dumps(doc, indent=2, ensure_ascii=False))


def _numbers(text: str | None, count: int, label: str) -> list[float] | None:
    if text is None:
        return None
    try:
        values = [float(v) for v in text.split(",")]
    except ValueError:
        raise _Usage(f"{label} takes {count} comma-separated numbers") from None
    if len(values) != count:
        raise _Usage(f"{label} takes {count} comma-separated numbers")
    return values


def _json_file(path: str | None, label: str) -> Any:
    if path is None:
        return None
    try:
        text = sys.stdin.read() if path == "-" else Path(path).read_text(encoding="utf-8")
        return json.loads(text)
    except (OSError, ValueError) as exc:
        raise _Usage(f"cannot read {label}: {exc}") from None


def _json_text(text: str | None, label: str) -> Any:
    if text is None:
        return None
    if not text.lstrip().startswith(("{", "[")):
        return _json_file(text, label)
    try:
        return json.loads(text)
    except ValueError as exc:
        raise _Usage(f"{label} is not valid JSON: {exc}") from None


def _common(parser: argparse.ArgumentParser, *, writes: bool = True) -> None:
    parser.add_argument("workspace", help="the workspace file (a Project file is read as a workspace)")
    if writes:
        parser.add_argument("--op-id", help="this operation's id; repeat it unchanged to retry safely")
        parser.add_argument("--expect-base", metavar="DIGEST", help="refuse unless the workspace still has this workspace_digest")
        parser.add_argument("--require-check", action="store_true", help="refuse the operation when its result has a problem the "
                            "local check refuses and the workspace did not have before (default: the operation is applied and "
                            "the reply's check lists the problems)")
        parser.add_argument("--request", metavar="FILE", help="the whole request as a JSON file ('-' for stdin)")
        parser.add_argument("-o", "--out", help="write the resulting workspace here instead of rewriting WORKSPACE")
    parser.add_argument("--json", action="store_true", help="print the whole reply, including the workspace")


def _selector_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ids", help="comma-separated component / observation ids")
    parser.add_argument("--group", action="append", default=[], help="an editing group (repeat for several)")
    parser.add_argument("--instance", action="append", default=[], help="an imported instance (repeat for several)")
    parser.add_argument("--kinds", help="COMPONENT, OBSERVATION or both (comma-separated)")


def _selector(args: argparse.Namespace) -> dict[str, Any] | None:
    selector: dict[str, Any] = {}
    if args.ids:
        selector["ids"] = [i for i in args.ids.split(",") if i]
    if args.group:
        selector["groups"] = list(args.group)
    if args.instance:
        selector["instances"] = list(args.instance)
    if args.kinds:
        selector["kinds"] = [k.strip().upper() for k in args.kinds.split(",") if k.strip()]
    return selector or None


def _ref(text: str | None, label: str, names: tuple[str, ...]) -> Any:
    """``design`` / ``frame_origin`` style words, ``name:id`` references, or ``x,y,z`` points."""

    if text is None:
        return None
    if text in names:
        return text
    head, sep, rest = text.partition(":")
    if sep and head in ("group", "component", "instance") and rest:
        return {head: rest}
    if "," in text:
        return {"point_mm": _numbers(text, 3, label)}
    raise _Usage(f"{label}: {text!r} is not one of {list(names)}, group:ID, component:ID, instance:KEY or X,Y,Z")


def register(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("workspace", help="compose optical systems from reusable layouts: import, group, move, run",
                                   description=WORKSPACE_DESCRIPTION, epilog=WORKSPACE_EPILOG)
    sub = parser.add_subparsers(dest="workspace_command", required=True)
    p_init = sub.add_parser("init", help="make a workspace from a Project (optionally describing it as a module)", epilog=WORKSPACE_EPILOG)
    p_init.add_argument("project", help="a Project (or workspace) file")
    p_init.add_argument("--module", metavar="JSON", help="module descriptor (JSON text or file): module_id, title, fixtures, interfaces, "
                                                         "requirements")
    p_init.add_argument("-o", "--out", required=True, help="the workspace file to write")
    p_init.add_argument("--json", action="store_true", help="print the whole reply, including the workspace")

    p_import = sub.add_parser("import", help="import a module as a new instance", epilog=WORKSPACE_EPILOG)
    _common(p_import)
    p_import.add_argument("module", help="the module file (a workspace written by 'workspace export', or a Project)")
    p_import.add_argument("--key", help="the new instance's key (lowercase letters, digits, '_'; at most 24)")
    p_import.add_argument("--position", metavar="X,Y,Z", help="where the module's design-frame origin goes (mm)")
    p_import.add_argument("--quaternion", metavar="W,X,Y,Z", help="the module's design-frame orientation")
    p_import.add_argument("--placement", metavar="JSON", help="the placement pose as JSON {position_mm, quaternion_wxyz}")
    p_import.add_argument("--fixtures", metavar="JSON", help="fixture handling: {policy, overrides, rebind, unbound_targets}")
    p_import.add_argument("--rebind", action="append", default=[], metavar="FIXTURE=ID",
                          help="rebind a removed fixture (e.g. the module's test source) to an object of the system")
    p_import.add_argument("--label", help="a label for the instance and its group")
    p_import.add_argument("--group-id", help="the id of the import's editing group (default: the key)")
    p_import.add_argument("--catalog-binding", choices=("repin", "refuse"), help="when catalog pins differ (default repin)")

    p_select = sub.add_parser("select", help="print the objects a selector selects", epilog=WORKSPACE_EPILOG)
    _common(p_select, writes=False)
    _selector_args(p_select)

    p_transform = sub.add_parser("transform", help="move / rotate a selection as one rigid body", epilog=WORKSPACE_EPILOG)
    _common(p_transform)
    _selector_args(p_transform)
    p_transform.add_argument("--translate", metavar="X,Y,Z", help="translation (mm) in the frame's axes")
    p_transform.add_argument("--axis", metavar="X,Y,Z", help="rotation axis in the frame's axes")
    p_transform.add_argument("--angle", type=float, metavar="DEG", help="rotation angle in degrees (right-hand rule)")
    p_transform.add_argument("--rotation-quaternion", metavar="W,X,Y,Z", help="rotation as a unit quaternion in the frame's axes")
    p_transform.add_argument("--frame", help="design (default), group:ID, component:ID or instance:KEY")
    p_transform.add_argument("--pivot", help="frame_origin (default), selection_centroid, component:ID or X,Y,Z (in the frame)")

    p_group = sub.add_parser("group", help="create, dissolve or edit an editing group", epilog=WORKSPACE_EPILOG)
    p_group.add_argument("action", choices=("create", "dissolve", "set", "add", "remove"))
    _common(p_group)
    p_group.add_argument("group_id", help="the group's id")
    _selector_args(p_group)
    p_group.add_argument("--label", help="create: the group's label")
    p_group.add_argument("--frame", help="create: the group's reference frame: component:ID, instance:KEY, group:ID or a pose JSON")

    p_replace = sub.add_parser("replace", help="replace an instance with another revision of its module", epilog=WORKSPACE_EPILOG)
    _common(p_replace)
    p_replace.add_argument("key", help="the instance to replace")
    p_replace.add_argument("module", help="the new module file")
    p_replace.add_argument("--discard-edits", action="store_true", help="replace even if the instance or an instance nested in it was edited since its import (members changed, deleted or added)")
    p_replace.add_argument("--fixtures", metavar="JSON", help="fixture handling, as for import")
    p_replace.add_argument("--rebind", action="append", default=[], metavar="FIXTURE=ID", help="as for import")

    p_update = sub.add_parser("update-project", help="replace the workspace's Project by an edited revision of it",
                              epilog=WORKSPACE_EPILOG)
    _common(p_update)
    p_update.add_argument("project", nargs="?", help="the edited Project file (or a workspace file: its Project); groups drop the objects it no "
                                          "longer has, and an instance member it no longer has stays recorded as deleted (an edit "
                                          "of its instance)")
    p_update.add_argument("--ops", help="apply Project operations from this JSON file to the workspace's Project, e.g. set_compute_budget")

    p_batch = sub.add_parser("batch", help="apply several operations atomically", epilog=WORKSPACE_EPILOG)
    _common(p_batch)
    p_batch.add_argument("operations", help="JSON file: [{kind, request, module?}, ...] (module: a JSON document or a file path)")

    p_export = sub.add_parser("export", help="write the workspace as a reusable module", epilog=WORKSPACE_EPILOG)
    _common(p_export, writes=False)
    p_export.add_argument("--module-id", help="the module's id")
    p_export.add_argument("--title", help="the module's title")
    p_export.add_argument("--revision", help="a revision label")
    p_export.add_argument("--descriptor", metavar="JSON", help="the whole module descriptor (JSON text or file); options override it")
    p_export.add_argument("-o", "--out", required=True, help="the module file to write")

    p_status = sub.add_parser("status", help="validations and their freshness, instances, groups, the module", epilog=WORKSPACE_EPILOG)
    _common(p_status, writes=False)

    p_run = sub.add_parser("run", help="run the whole system and record its validation", epilog=WORKSPACE_EPILOG)
    _common(p_run, writes=False)
    p_run.add_argument("--out-dir", help="write the run directory here")
    p_run.add_argument("-o", "--out", help="write the workspace with the new validation here instead of rewriting WORKSPACE")
    add_status_contract_argument(p_run)

    p_record = sub.add_parser("record", help="record the validation of a run directory made elsewhere", epilog=WORKSPACE_EPILOG)
    _common(p_record, writes=False)
    p_record.add_argument("run_dir", help="the run directory (holds local_validation.json and manifest.json)")
    p_record.add_argument("-o", "--out", help="write the workspace here instead of rewriting WORKSPACE")

    p_history = sub.add_parser("history", help="the operation journal, or whether one operation was applied", epilog=WORKSPACE_EPILOG)
    _common(p_history, writes=False)
    p_history.add_argument("--op-id", help="only this operation: was it applied?")
    p_history.add_argument("--limit", type=int, help="the last LIMIT operations")

    for p in (p_init, p_import, p_select, p_transform, p_group, p_replace, p_update, p_batch, p_export, p_status, p_run, p_record,
              p_history):
        p.set_defaults(handler=command)
    return parser


def _write(path: str, document: Any) -> None:
    """Write ``document`` to ``path`` in one step: a complete temporary file next to it replaces it, so an interrupted
    write never leaves a truncated workspace."""

    target = Path(path)
    temporary = target.with_name(f".{target.name}.tmp")
    try:
        temporary.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(target)
    except OSError as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise _Usage(f"cannot write {path}: {exc.strerror}") from None


def _rebind(items: list[str]) -> dict[str, str]:
    rebind = {}
    for item in items:
        name, sep, target = item.partition("=")
        if not sep or not name or not target:
            raise _Usage(f"--rebind takes FIXTURE=ID, got {item!r}")
        rebind[name] = target
    return rebind


def _module_document(value: Any) -> Any:
    if isinstance(value, str):
        return _json_file(value, "module")
    return value


def _request(args: argparse.Namespace) -> dict[str, Any]:
    request = _json_file(getattr(args, "request", None), "--request") or {}
    if not isinstance(request, dict):
        raise _Usage("--request holds a JSON object")
    if getattr(args, "op_id", None):
        request["op_id"] = args.op_id
    if getattr(args, "expect_base", None):
        request["expected_base"] = args.expect_base
    if getattr(args, "require_check", False):
        request["require_check"] = True
    return request


def _apply(args: argparse.Namespace, ws: Any) -> tuple[Any, dict[str, Any], str]:
    """(reply, request, what) for the changing commands."""

    from .local import json_api as J

    request = _request(args)
    command_name = args.workspace_command
    if command_name == "import":
        if args.key:
            request["instance_key"] = args.key
        position, quaternion = _numbers(args.position, 3, "--position"), _numbers(args.quaternion, 4, "--quaternion")
        if args.placement:
            request["placement"] = _json_text(args.placement, "--placement")
        elif position is not None or quaternion is not None:
            request["placement"] = {"position_mm": position or [0.0, 0.0, 0.0], "quaternion_wxyz": quaternion or [1.0, 0.0, 0.0, 0.0]}
        _fixture_options(args, request)
        for name in ("label", "group_id", "catalog_binding"):
            if getattr(args, name):
                request[name] = getattr(args, name)
        return J.workspace_import(ws, _json_file(args.module, "module"), request), request, "import"
    if command_name == "transform":
        selector = _selector(args)
        if selector is not None:
            request["selector"] = selector
        translation = _numbers(args.translate, 3, "--translate")
        if translation is not None:
            request["translation_mm"] = translation
        if args.rotation_quaternion:
            request["rotation"] = {"quaternion_wxyz": _numbers(args.rotation_quaternion, 4, "--rotation-quaternion")}
        elif args.axis is not None or args.angle is not None:
            if args.axis is None or args.angle is None:
                raise _Usage("a rotation needs --axis and --angle")
            request["rotation"] = {"axis": _numbers(args.axis, 3, "--axis"), "angle_deg": args.angle}
        if args.frame:
            request["frame"] = _ref(args.frame, "--frame", ("design",))
        if args.pivot:
            request["pivot"] = _ref(args.pivot, "--pivot", ("frame_origin", "selection_centroid"))
        return J.workspace_transform(ws, request), request, "transform"
    if command_name == "group":
        request["group_id"] = args.group_id
        selector = _selector(args)
        if args.action != "dissolve" and (selector is not None or args.action != "create"):
            request["members"] = selector or {}
        if args.label:
            request["label"] = args.label
        if args.frame:
            request["frame"] = _json_text(args.frame, "--frame") if args.frame.lstrip().startswith("{") else _ref(args.frame, "--frame", ())
        name = {"create": "workspace_group_create", "dissolve": "workspace_group_dissolve", "set": "workspace_group_set_members",
                "add": "workspace_group_add", "remove": "workspace_group_remove"}[args.action]
        return getattr(J, name)(ws, request), request, f"group {args.action}"
    if command_name == "replace":
        request["instance_key"] = args.key
        if args.discard_edits:
            request["discard_edits"] = True
        _fixture_options(args, request)
        return J.workspace_replace_instance(ws, _json_file(args.module, "module"), request), request, "replace"
    if command_name == "update-project":
        if bool(args.project) == bool(args.ops):
            raise _Usage("update-project takes either a Project file or --ops FILE")
        if args.ops:
            from .composition import load_workspace

            operations = _json_file(args.ops, "operations")
            if isinstance(operations, dict) and set(operations) == {"ops"}:
                operations = operations["ops"]
            edited = J.update_project(load_workspace(ws).project.payload(), operations,
                                      {"require_check": True} if request.get("require_check") is True else None)
            if not edited["ok"]:
                return edited, request, "update-project"
            project = edited["project"]
        else:
            project = _json_file(args.project, "project")
        if isinstance(project, dict) and project.get("schema") == "opensci.optical_workspace":
            project = project.get("project")
        request["project"] = project
        return J.workspace_update_project(ws, request), request, "update-project"
    operations = _json_file(args.operations, "operations")
    if not isinstance(operations, list):
        raise _Usage("the operations file holds a JSON list of {kind, request, module?}")
    for op in operations:
        if isinstance(op, dict) and isinstance(op.get("module"), str):
            op["module"] = _module_document(op["module"])
    request["operations"] = operations
    return J.workspace_batch(ws, request), request, "batch"


def _fixture_options(args: argparse.Namespace, request: dict[str, Any]) -> None:
    fixtures = _json_text(args.fixtures, "--fixtures") if args.fixtures else request.get("fixtures")
    rebind = _rebind(args.rebind)
    if rebind:
        fixtures = dict(fixtures or {})
        fixtures["rebind"] = {**(fixtures.get("rebind") or {}), **rebind}
    if fixtures is not None:
        request["fixtures"] = fixtures


def _print_reply(args: argparse.Namespace, reply: dict[str, Any], written: str | None) -> None:
    if args.json:
        _out(reply)
        return
    shown = {k: v for k, v in reply.items() if k not in ("workspace", "module", "files")}
    if written is not None:
        shown = {"written": written, **shown}
    _out(shown)


def command(args: argparse.Namespace) -> int:
    from .local import json_api as J

    try:
        if args.workspace_command == "init":
            module = _json_text(args.module, "--module") if args.module else None
            reply = J.workspace_init(_json_file(args.project, "project"), {"module": module} if module is not None else None)
            if not reply["ok"]:
                _out(reply)
                return REFUSED
            _write(args.out, reply["workspace"])
            _print_reply(args, reply, args.out)
            return 0
        ws = _json_file(args.workspace, "workspace")
        if args.workspace_command == "select":
            selector = _selector(args) or {}
            reply = J.workspace_select(ws, selector)
        elif args.workspace_command == "status":
            reply = J.workspace_status(ws)
        elif args.workspace_command == "history":
            query = {k: v for k, v in (("op_id", args.op_id), ("limit", args.limit)) if v is not None}
            reply = J.workspace_history(ws, query or None)
        elif args.workspace_command == "export":
            descriptor = _json_text(args.descriptor, "--descriptor") if args.descriptor else {}
            for name in ("module_id", "title", "revision"):
                if getattr(args, name):
                    descriptor[name] = getattr(args, name)
            reply = J.workspace_export_module(ws, descriptor)
            if reply["ok"]:
                _write(args.out, reply["module"])
                _print_reply(args, reply, args.out)
                return 0
        elif args.workspace_command == "run":
            return _run(args, ws)
        elif args.workspace_command == "record":
            return _record(args, ws)
        else:
            reply, _request_doc, _what = _apply(args, ws)
            if reply["ok"]:
                target = args.out or args.workspace
                _write(target, reply["workspace"])
                _print_reply(args, reply, target)
                return 0
        if not reply["ok"]:
            _out(reply)
            return REFUSED
        _print_reply(args, reply, None)
        return 0
    except _Usage as exc:
        print(f"error: {exc}", file=sys.stderr)
        return USAGE_ERROR


def _run(args: argparse.Namespace, ws: Any) -> int:
    from .composition import CompositionError
    from .composition.validation import run_workspace, workspace_target_rows

    contract = getattr(args, "status_contract", None) or DEFAULT_STATUS_CONTRACT  # (resolved by cli.py, with the environment)
    try:
        new_ws, run, entry = run_workspace(ws, output_dir=args.out_dir)
    except CompositionError as exc:
        _out({"ok": False, "error": exc.as_dict(), "workspace_digest": exc.workspace_digest})
        return REFUSED
    except OSError as exc:  # (a run directory that holds a file of the person's own, a link, or cannot be written)
        print(f"error: cannot write the run directory: {exc}", file=sys.stderr)
        return USAGE_ERROR
    target = args.out or args.workspace
    _write(target, new_ws.payload())
    # (the run summary's rows: the effective assessment -- the record's, which also counts the published models'
    # applicability -- and each value with its entry and failed checks, as LocalOpticalRun.summary states them)
    summary = run.summary()
    reply = {"ok": True, "written": target, "status": run.status.value, "workspace_digest": new_ws.workspace_digest(), "validation": entry,
             "targets": workspace_target_rows(summary["targets"]),
             "run_directory": str(run.artifacts.directory) if run.artifacts is not None else None,
             "outcome": None if run.local_validation.simulation.outcome is None else run.local_validation.simulation.outcome.model_dump(mode="json"),
             "exit_contract": contract}
    _out(reply if not args.json else {**reply, "workspace": new_ws.payload(), "summary": {**summary, "exit_contract": contract}})
    return run_exit_code(contract, run)


def _record(args: argparse.Namespace, ws: Any) -> int:
    from .local import json_api as J

    run_dir = Path(args.run_dir)
    record = _json_file(str(run_dir / "local_validation.json"), "local_validation.json")
    request: dict[str, Any] = {"local_validation": record}
    manifest_path = run_dir / "simulation_result.json"
    if manifest_path.exists():
        result = _json_file(str(manifest_path), "simulation_result.json")
        projection = (result.get("manifest") or {}).get("design_projection_hash") if isinstance(result, dict) else None
        if isinstance(projection, str):
            request["design_projection_hash"] = projection
    reply = J.workspace_record_validation(ws, request)
    if not reply["ok"]:
        _out(reply)
        return REFUSED
    target = args.out or args.workspace
    _write(target, reply["workspace"])
    _print_reply(args, reply, target)
    return 0


__all__ = ["register"]
