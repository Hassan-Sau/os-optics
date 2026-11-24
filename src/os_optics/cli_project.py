"""``os-optics capabilities``, ``os-optics project check PROJECT`` and ``os-optics project update PROJECT --ops
OPS.json``: what this runtime offers, a Project's identity and check without a run, and headless Project edit
operations from the shell (``os_optics.local.project_ops``)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

USAGE_ERROR = 64
LOCAL_INPUT_ERROR = 65
CAPABILITIES_DESCRIPTION = (
    "Print what this runtime is and offers as JSON (the same document as os_optics.local.json_api.capabilities()): engine "
    "version and build id, json_api version, the active catalog release with product counts per category, form and model "
    "kind, optical roles, DOF axes, target metrics with their units, example kinds, Project features, result queries and "
    "export kinds, the composition vocabularies, the Project edit operations (project_ops), the advisory warning codes of a "
    "check and the run freshness states and reasons. No network; without a catalog, catalog and catalog_version are null."
)
CHECK_DESCRIPTION = (
    "Check a Project without running it: the Project contract, then the same lowering a run performs (catalog binding, "
    "declared models, source physics, metrics, optical solids). Prints its identity -- project_hash (osc1) and "
    "engine_input_hash, the values a run records and 'results freshness' compares --, the catalog release and design intent, "
    "the budgets and numerical settings in force (compute_budget, numerical_settings), the catalog parts, the declared models, the optical-solid status and the advisory warnings: "
    "TARGET_INVALID, TARGET_PARAMETER_INVALID, UNKNOWN_SOURCE_REFERENCE (a target that cannot be read as written: the run "
    "reports it UNKNOWN with that code and evaluates everything else), TARGET_UNIT_MISMATCH (a "
    "target whose unit is not its metric's; the run reports that target's value in the metric's unit and does not compare its "
    "bounds; its status is UNKNOWN, and a target the "
    "run cannot evaluate at all keeps that outcome), "
    "SOURCE_WAVELENGTH_OUTSIDE_COMPONENT_BAND (no source line of the Project lies inside a component's model band, or a "
    "tabulated thin lens's or Faraday rotator's table domain: every beam that meets the component is outside its model, "
    "and the run reports MODEL_WAVELENGTH_OUTSIDE_BAND in "
    "model_scope_issues (for those two also as an Engine issue in issues; details.reported_in is then engine_issues); a "
    "component some source line fits has no warning, and the run reports the other lines that reach it), "
    "FRAME_SYMMETRY_DERIVED and DESIGN_GEOMETRY_EXTENDED (a declared model's stated frame symmetry, or a stated cylinder or "
    "box design_geometry that does not hold its surfaces: the derived symmetry, or the smallest body of that kind that holds "
    "both, is used), and INPUT_NOT_SIMULATED (an input "
    "the simulation does not read: a selected parameter that is not an attenuator position, an entry of the Project's materials "
    "list; the run's results are those of the Project without it). Warnings never "
    "refuse a Project."
)
UPDATE_DESCRIPTION = (
    "Apply Project edit operations, including rigid transform, and check "
    "the result like 'project check'. OPS.json holds a JSON array of operations (or {\"ops\": [...]}), each {\"op\": NAME, "
    "...}: set_component_pose, set_source_model {instance_id, changes}, set_observation_pose, set_observation, set_target "
    "{target_id, changes}, set_retained_dof(s), set_component_product, set_component_model, set_design_intent, "
    "set_compute_budget {compute_budget: {max_interactions_per_path, max_paths, max_trace_steps, max_interference_pairs, "
    "max_return_analysis_work, pupil_ring_count, pupil_points_per_ring, max_scatter_origins, max_mesh_patches} | null}, "
    "set_numerical_settings {numerical_settings: {power_epsilon_W, negligible_power_fraction, gaussian_*, interference_*, "
    "nonlinear_*, rating_*, capture_min_power_fraction, aperture_quad_sector_nodes, design_point: {...}} | null} (the "
    "validity limits, levels and match windows; 'capabilities' lists each with its default, unit and direction under "
    "numerical_settings.fields; the ranges are the Project schema's, os-optics schema project), "
    "add_/remove_component, add_/remove_observation, add_/remove_target, add_/update_/remove_optical_model, "
    "add_/update_/remove_model_material, add_/remove_model_coating and transform {selector, translation_mm, rotation, frame, "
    "pivot}; in 'changes' a null value removes that field ('capabilities' lists the operations under project_ops; "
    "os_optics.local.project_ops documents each). The operations apply together or not at all, and the edited document is a "
    "Project or nothing is written. The check is reported, not required: the report's check.ok says whether a run accepts "
    "the edited Project and check.problems lists what it refuses (engine_input_hash, catalog_version and compute_budget are "
    "then null), so a design is built and repaired one edit at a time. --require-check refuses the update when the check "
    "does not pass. Without "
    "-o the updated Project is printed with the report; with -o it is written to that file (atomically; the input file is "
    "rewritten only when -o names it, and not at all when nothing changed) and the report is printed. --expect-hash refuses "
    "the update (PROJECT_CHANGED) when the Project's project_hash is not the one the operations were written against. "
    "Retries: absolute operations applied again change nothing (changed: false); an add of an item identical to the one "
    "already present is listed in already_present. transform is relative, and a list that adds an item and then changes "
    "or removes it is refused when applied again to its own result (DUPLICATE_ID / NOT_FOUND), so retry with "
    "--expect-hash: PROJECT_CHANGED then reports the current project_hash."
)


def _out(document: object) -> None:
    print(json.dumps(document, indent=2, ensure_ascii=True))


def _capabilities(args: argparse.Namespace) -> int:
    from .local import json_api

    _out({key: value for key, value in json_api.capabilities().items() if key != "ok"})
    return 0


def _read_bytes(path: str, what: str) -> bytes | None:
    try:
        return sys.stdin.buffer.read() if path == "-" else Path(path).read_bytes()
    except OSError as exc:
        print(f"error: cannot read the {what}: {exc}", file=sys.stderr)
        return None


def _check(args: argparse.Namespace) -> int:
    from .local import json_api

    project = _read_bytes(args.project, "project")
    catalog = _read_bytes(args.catalog, "catalog") if args.catalog else None
    if project is None or (args.catalog and catalog is None):
        return USAGE_ERROR
    reply = json_api.check_project(project, catalog=catalog)
    if not reply["ok"]:
        _out({"status": "ERROR", "problems": reply["error"]["problems"],
              **({"optical_solids": reply["optical_solids"]} if reply.get("optical_solids") is not None else {})})
        return LOCAL_INPUT_ERROR
    solids = reply["optical_solids"]
    _out({
        "status": "ACCEPTED",
        "project_hash": reply["project_hash"],
        "engine_input_hash": reply["engine_input_hash"],
        "catalog_version": reply["catalog_version"],
        "design_intent": reply["design_intent"],
        "compute_budget": reply["compute_budget"],
        "numerical_settings": reply["numerical_settings"],
        "simulated_parts": [{"instance_id": p["instance_id"], "optical_sku": p["optical_sku"]} for p in reply["simulated_parts"]],
        "declared_models": reply["declared_models"],
        "optical_solids": None if solids is None else {"status": solids.get("status"), "issues": solids.get("issues", [])},
        "warnings": reply["warnings"],
        **({"configurations": [{**{k: v for k, v in row.items() if k != "optical_solids"},
                                "optical_solids": None if row["optical_solids"] is None else
                                {"status": row["optical_solids"].get("status"), "issues": row["optical_solids"].get("issues", [])}}
                               for row in reply["configurations"]]} if "configurations" in reply else {}),
    })
    return 0


def _same_file(a: str, b: str) -> bool:
    try:
        return Path(a).samefile(b)
    except OSError:
        return False


def _write_atomically(path: str, data: bytes) -> None:
    """Write ``data`` to a temporary file beside ``path``, then rename it over ``path`` (a reader never sees a part)."""

    target = Path(path)
    temporary = target.with_name(f".{target.name}.update.tmp")
    try:
        temporary.write_bytes(data)
        temporary.replace(target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _ops(data: bytes) -> Any:
    document = json.loads(data)
    if isinstance(document, dict) and set(document) == {"ops"}:
        return document["ops"]
    return document


def _update(args: argparse.Namespace) -> int:
    from .local.project_ops import ProjectUpdateError, update_project

    if args.project == "-" and args.ops == "-":
        print("error: the Project and the operations cannot both come from standard input", file=sys.stderr)
        return USAGE_ERROR
    project = _read_bytes(args.project, "project")
    ops_data = _read_bytes(args.ops, "operations")
    if project is None or ops_data is None:
        return USAGE_ERROR
    try:
        ops = _ops(ops_data)
    except (ValueError, RecursionError) as exc:
        print(f"error: the operations file is not JSON: {exc}", file=sys.stderr)
        return USAGE_ERROR
    try:
        update = update_project(project, ops, expected_project_hash=args.expect_hash, require_check=args.require_check)
    except ProjectUpdateError as exc:
        _out({"status": "ERROR", "error": exc.as_dict()})
        return LOCAL_INPUT_ERROR
    report = update.as_dict()
    project_doc = report.pop("project")
    report.pop("optical_solids", None)
    written = False
    if args.out:
        unchanged_in_place = not update.changed and args.project != "-" and _same_file(args.out, args.project)
        if not unchanged_in_place:
            try:
                _write_atomically(args.out, (json.dumps(project_doc, indent=2, ensure_ascii=True) + "\n").encode("utf-8"))
            except OSError as exc:
                print(f"error: cannot write the Project: {exc}", file=sys.stderr)
                return USAGE_ERROR
            written = True
    document: dict[str, Any] = {"status": "UPDATED" if update.changed else "UNCHANGED", **report, "out": args.out, "project_written": written}
    if not args.out:
        document["project"] = project_doc
    _out(document)
    return 0


def register(subparsers: argparse._SubParsersAction) -> None:
    """Add ``os-optics capabilities``."""

    capabilities = subparsers.add_parser("capabilities", help="print what this runtime offers (JSON): versions, catalog counts, roles, "
                                                              "metrics with units, operations, result queries",
                                         description=CAPABILITIES_DESCRIPTION, epilog="exit codes: 0 = printed")
    capabilities.set_defaults(handler=_capabilities)


def register_project(project_commands: argparse._SubParsersAction) -> None:
    """Add ``os-optics project check`` and ``os-optics project update``."""

    check = project_commands.add_parser("check", help="check a Project without running it: identity, problems, advisory warnings",
                                        description=CHECK_DESCRIPTION,
                                        epilog="exit codes: 0 = accepted (warnings may be listed), 64 = unreadable file, 65 = refused "
                                               "(JSON with status ERROR and the problems)")
    check.add_argument("project", help="path to a Project JSON file ('-' = standard input)")
    check.add_argument("--catalog", help="check against this public optical catalog JSON instead of the release the Project pins")
    check.set_defaults(handler=_check)
    update = project_commands.add_parser("update", help="apply headless Project edit operations and report the check of the result",
                                         description=UPDATE_DESCRIPTION,
                                         epilog="exit codes: 0 = updated or unchanged (check.ok says whether a run accepts the "
                                                "Project), 64 = unreadable / unwritable file or operations "
                                                "that are not JSON, 65 = refused (JSON error: PROJECT_OP_INVALID with op_index and "
                                                "op_code, PROJECT_REJECTED with the problems of a document that is no Project or, with "
                                                "--require-check, of the check, PROJECT_CHANGED with the current "
                                                "project_hash); nothing is written on a refusal")
    update.add_argument("project", help="path to a Project JSON file ('-' = standard input)")
    update.add_argument("--ops", required=True, help="JSON file with the operations ('-' = standard input)")
    update.add_argument("--expect-hash", help="refuse unless the Project's project_hash is this one (PROJECT_CHANGED)")
    update.add_argument("--require-check", action="store_true", help="refuse the update (PROJECT_REJECTED) when the check does not "
                        "accept the edited Project")
    update.add_argument("-o", "--out", help="write the updated Project here (may be the input file) instead of printing it")
    update.set_defaults(handler=_update)
