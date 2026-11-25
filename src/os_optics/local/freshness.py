"""Is a run current for a Project? One call, no workspace needed.

``run_freshness(project, run)`` compares a run with a Project by the rule the workspace uses for its validations
(``validation_freshness`` of the optical-design contracts; ``os_optics.composition.validation``): the run is
``CURRENT`` exactly when the Project's osc1 hash and the Engine input hash it lowers to equal those the run recorded;
otherwise it is ``HISTORICAL`` with the reasons ``PHYSICS_CHANGED`` (the design apart from its targets changed -- its optics,
sources, observation planes or ``compute_budget`` --, or the change cannot be attributed to the targets alone), ``TARGETS_CHANGED``, ``OTHER_PROJECT_CHANGED`` (only other Project
fields, such as the design intent), ``CATALOG_CHANGED`` (the same Project gives another Engine input: another catalog
release) and ``ENGINE_CHANGED`` (added to the reasons of a ``HISTORICAL`` run that another Engine build made). ``UNDETERMINED`` (``PROJECT_NOT_CHECKED``): the same Project, but it does not pass the
check now, so its Engine input is unknown. The rule keeps a run of the same Project and Engine input ``CURRENT`` when
another Engine build made it, as the workspace does; ``engine_build_matches`` says whether the installed build made it.

``run`` is a run directory (path), its files (``{file name: bytes}``: ``json_api.run_project``'s ``files``; a file given
as text is read as its UTF-8 bytes), a
``run_project`` reply, or a Local Validation record (``local_validation.json``: its document or its path). With the run's
files, the run's ``project.json`` gives the targets digest and ``simulation_result.json`` the design projection hash, so
the reasons can tell targets from physics. A bare record carries neither: a changed Project is then reported
``PHYSICS_CHANGED``.

Freshness and integrity are two answers. Freshness is the hash rule above, read from the record as it is stored: is this
the run of this Project? Integrity is whether the run's files are the ones the run wrote (their hashes and
cross-references, ``verify_run_files``): ``integrity`` is ``{"checked": false, "consistent": null, "problems": []}``
unless ``verify`` is true, and a run that does not verify is reported there, with the freshness its hashes give. A run
is verified where its bytes cross a trust boundary (``project verify``, a submission, the server's intake:
``local/artifacts.py``), not by every freshness query.

Beside the aggregate, the reply states the two identities separately. ``physics``: ``{"freshness", "reasons"}`` -- is the
run's physics the Project's? The physics identity is the exact hash of the Engine request without its targets
(``composition.validation.physics_input_hash``: every component, plane and material, the numerical configuration with
its budgets and settings), read for the run from its ``engine_request.json`` and for the Project from its lowering:
equal means ``CURRENT`` whatever the targets did, so after an edit of the targets alone the stored quantities are still
the Project's and only the requirements need assessing again. ``HISTORICAL`` carries ``PHYSICS_CHANGED`` or
``CATALOG_CHANGED`` (and ``ENGINE_CHANGED`` as the aggregate does); ``UNDETERMINED`` carries
``PROJECT_NOT_CHECKED``, or ``ENGINE_REQUEST_NOT_AVAILABLE`` for a bare record whose Project changed (the record alone
cannot tell a physics change from a target change). ``requirements``: ``{"freshness"}`` -- ``CURRENT`` when the physics
is and the targets are the ones the run assessed, ``HISTORICAL`` when either changed. A re-run is needed for the rows
whose physics is not ``CURRENT``.

A Project with configurations (``ProjectDocumentV2.configurations``): a run of its configurations (``run_configured_project``)
is compared configuration by configuration (``os_optics.local.configurations.configurations_freshness``: each
configuration's run against ``configured(id)``, ``CONFIGURATIONS_CHANGED`` when the Project's configurations are not the
ones run); an ordinary run is ``HISTORICAL`` for it (``CONFIGURATIONS_CHANGED``).
"""

from __future__ import annotations

import json
from collections import namedtuple
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import ValidationError

#: What the workspace rule reads of a workspace: its Project (identity) and its validations (freshness rows).
_ProjectOnly = namedtuple("_ProjectOnly", "project")
_Validations = namedtuple("_Validations", "validations")

RUN_NOT_READABLE = "RUN_NOT_READABLE"
FRESHNESS_STATES = ("CURRENT", "HISTORICAL", "UNDETERMINED")
CONFIGURATIONS_CHANGED = "CONFIGURATIONS_CHANGED"
ENGINE_REQUEST_NOT_AVAILABLE = "ENGINE_REQUEST_NOT_AVAILABLE"
FRESHNESS_REASONS = ("PHYSICS_CHANGED", "TARGETS_CHANGED", "OTHER_PROJECT_CHANGED", "CATALOG_CHANGED", "ENGINE_CHANGED",
                     "PROJECT_NOT_CHECKED", CONFIGURATIONS_CHANGED, ENGINE_REQUEST_NOT_AVAILABLE)
#: The aggregate's reasons that say the physics changed (the others concern targets and other Project fields).
PHYSICS_REASONS = ("PHYSICS_CHANGED", "CATALOG_CHANGED")
#: ``integrity`` of a run whose files were not verified.
NOT_VERIFIED = {"checked": False, "consistent": None, "problems": []}


class RunNotReadable(ValueError):
    """The run cannot be read as a run or a Local Validation record."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

    def as_dict(self) -> dict[str, Any]:
        return {"code": RUN_NOT_READABLE, "problems": [{"code": RUN_NOT_READABLE, "message": self.message}]}


def _files_of(directory: Path) -> dict[str, bytes]:
    from ..results.model import RUN_FILE_NAMES

    files: dict[str, bytes] = {}
    for name in RUN_FILE_NAMES:
        target = directory / name
        if target.is_file():
            try:
                files[name] = target.read_bytes()
            except OSError as exc:
                raise RunNotReadable(f"{name}: cannot be read ({exc})") from None
    return files


def _json(data: Any, name: str) -> Any:
    try:
        return json.loads(bytes(data.tobytes() if isinstance(data, memoryview) else data))
    except (ValueError, TypeError, RecursionError) as exc:
        raise RunNotReadable(f"{name}: not JSON ({type(exc).__name__})") from None


def _run_inputs(run: Any) -> tuple[Any, dict[str, bytes] | None]:
    """``(local validation document, run files or None)``."""

    from .artifacts import LOCAL_VALIDATION_FILE, MANIFEST_FILE

    if isinstance(run, (str, Path)):
        path = Path(run)
        if path.is_dir():
            if not (path / MANIFEST_FILE).is_file():  # (the directory rule: a run is there once its manifest is)
                raise RunNotReadable(f"{path} has no {MANIFEST_FILE}: no run was completed into it")
            files = _files_of(path)
        elif path.is_file():
            try:
                return _json(path.read_bytes(), path.name), None
            except OSError as exc:
                raise RunNotReadable(f"{path} cannot be read ({exc})") from None
        else:
            raise RunNotReadable(f"{path} is neither a run directory nor a Local Validation record file")
    elif isinstance(run, Mapping) and LOCAL_VALIDATION_FILE in run:
        files = {}
        for name, value in run.items():
            if isinstance(value, str):
                files[str(name)] = value.encode("utf-8")
            elif isinstance(value, (bytes, bytearray, memoryview)):
                files[str(name)] = bytes(value.tobytes() if isinstance(value, memoryview) else value)
            else:
                raise RunNotReadable(f"{name}: a run file is bytes or text, not {type(value).__name__}")
    elif isinstance(run, Mapping) and isinstance(run.get("files"), Mapping):
        return _run_inputs(run["files"])
    elif isinstance(run, Mapping) and isinstance(run.get("local_validation"), Mapping):
        return run["local_validation"], None
    elif isinstance(run, Mapping):
        return run, None
    else:
        raise RunNotReadable("a run is a run directory, its {file name: bytes} files, a run_project reply or a Local Validation record")
    if LOCAL_VALIDATION_FILE not in files:
        raise RunNotReadable(f"the run has no {LOCAL_VALIDATION_FILE}")
    return _json(files[LOCAL_VALIDATION_FILE], LOCAL_VALIDATION_FILE), files


def _run_facts(files: dict[str, bytes] | None, verify: bool) -> tuple[str | None, str | None, dict[str, Any]]:
    """``(design projection hash, targets digest, integrity)`` of the run's files; the files are verified only with
    ``verify``."""

    from .._vendor.shared_contracts.optical_design import project_targets_digest
    from .artifacts import PROJECT_FILE, SIMULATION_RESULT_FILE, verify_run_files
    from .catalog import load_project
    from .errors import LocalRunError

    if files is None:
        return None, None, dict(NOT_VERIFIED)
    integrity = dict(NOT_VERIFIED)
    if verify:
        problems = list(verify_run_files(files))
        integrity = {"checked": True, "consistent": not problems, "problems": problems}
    projection = targets = None
    if SIMULATION_RESULT_FILE in files:
        try:
            manifest = _json(files[SIMULATION_RESULT_FILE], SIMULATION_RESULT_FILE).get("manifest")
            value = manifest.get("design_projection_hash") if isinstance(manifest, dict) else None
            projection = value if isinstance(value, str) else None
        except (RunNotReadable, AttributeError):
            projection = None
    if PROJECT_FILE in files:
        try:
            targets = project_targets_digest(load_project(bytes(files[PROJECT_FILE])))
        except LocalRunError:
            targets = None
    return projection, targets, integrity


def _run_physics_hash(files: dict[str, bytes] | None) -> str | None:
    """The physics input hash of the run's ``engine_request.json`` (``None`` without the file or when it cannot be read)."""

    from ..composition.validation import physics_input_hash
    from .artifacts import ENGINE_REQUEST_FILE

    if files is None or ENGINE_REQUEST_FILE not in files:
        return None
    try:
        return physics_input_hash(_json(files[ENGINE_REQUEST_FILE], ENGINE_REQUEST_FILE))
    except (RunNotReadable, KeyError, TypeError, AttributeError, ValueError):
        return None


def physics_and_requirements(freshness: str, reasons: list[str], *, run_physics: str | None, current_physics: str | None,
                             run_targets: str | None, current_targets: str | None) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(physics, requirements)`` of a run (module docstring) from its aggregate ``freshness`` / ``reasons`` and the two
    identities: the physics input hashes and the targets digests of the run and of the Project (``None``: not known)."""

    carried = [r for r in reasons if r == "ENGINE_CHANGED"]
    if freshness == "CURRENT":
        physics: dict[str, Any] = {"freshness": "CURRENT", "reasons": []}
    elif CONFIGURATIONS_CHANGED in reasons:
        physics = {"freshness": "HISTORICAL", "reasons": [CONFIGURATIONS_CHANGED]}
    elif current_physics is None:
        physics = {"freshness": "UNDETERMINED", "reasons": ["PROJECT_NOT_CHECKED"]}
    elif run_physics is None:
        stated = [r for r in reasons if r in PHYSICS_REASONS and r != "PHYSICS_CHANGED"]  # (a catalog change is certain)
        physics = ({"freshness": "HISTORICAL", "reasons": stated + carried} if stated else
                   {"freshness": "UNDETERMINED", "reasons": [ENGINE_REQUEST_NOT_AVAILABLE]})
    elif run_physics == current_physics:
        physics = {"freshness": "CURRENT", "reasons": []}
    else:
        physics = {"freshness": "HISTORICAL", "reasons": ([r for r in reasons if r in PHYSICS_REASONS] or ["PHYSICS_CHANGED"]) + carried}
    if physics["freshness"] == "CURRENT" and run_targets is not None and run_targets == current_targets:
        requirements = "CURRENT"
    elif physics["freshness"] == "HISTORICAL" or (run_targets is not None and run_targets != current_targets):
        requirements = "HISTORICAL"
    else:
        requirements = "UNDETERMINED"
    return physics, {"freshness": requirements}


def run_freshness(project: Any, run: Any, *, catalog: Any = None, verify: bool = False) -> dict[str, Any]:
    """The freshness of ``run`` for ``project`` (module docstring): ``{"freshness", "reasons", "physics", "requirements",
    "engine_build_matches", "run", "current", "integrity"}``. ``run``: the run's identity and outcome (``project_hash``, ``engine_input_hash``, ``engine_build_id``,
    ``validation_status``, ``target_outcomes``, ``design_projection_hash``, ``targets_digest``, ``runtime_kind``);
    ``current``: the Project's (``project_hash``, ``engine_input_hash``, ``design_projection_hash``, ``targets_digest``,
    ``engine_build_id`` installed, ``check``). ``verify``: also verify the run's files and report it in ``integrity``
    (which never changes ``freshness`` or ``reasons``). Raises ``LocalRunError`` for a document that is not a Project and
    ``RunNotReadable`` for a run that cannot be read."""

    from .._vendor.shared_contracts.optical_design import LocalValidationRecordV1, ValidationEntryV1
    from ..composition.validation import _entry, current_identity, freshness_rows
    from .catalog import _validation_message, load_project

    from .configurations import configurations_freshness, run_reader

    document = load_project(project)
    read = run_reader(run)
    if read is not None:
        return configurations_freshness(document, read, catalog=catalog, verify=verify)
    record_document, files = _run_inputs(run)
    try:
        record = LocalValidationRecordV1.model_validate(record_document)
    except ValidationError as exc:
        raise RunNotReadable(f"not a Local Validation record: {_validation_message(exc)}") from None
    projection, targets, integrity = _run_facts(files, verify)
    entry_document = _entry(record, projection, targets or "")
    if targets is None:  # a bare record does not say which targets it ran with
        entry_document.pop("targets_digest")
    entry = ValidationEntryV1.model_validate(entry_document)
    identity = current_identity(_ProjectOnly(document), catalog=catalog)  # type: ignore[arg-type]
    row = freshness_rows(_Validations((entry,)), identity)[0]  # type: ignore[arg-type]
    freshness, reasons = row["freshness"], list(row["reasons"])
    if document.configurations:  # an ordinary run is not a run of the Project's configurations
        freshness, reasons = "HISTORICAL", [CONFIGURATIONS_CHANGED]
    run_physics = _run_physics_hash(files)
    physics, requirements = physics_and_requirements(
        freshness, reasons, run_physics=run_physics, current_physics=identity["physics_input_hash"],
        run_targets=targets, current_targets=identity["targets_digest"])
    return {
        "freshness": freshness,
        "reasons": reasons,
        "physics": physics,
        "requirements": requirements,
        "engine_build_matches": entry.engine_build_id == identity["engine_build_id"],
        "run": {"project_hash": entry.project_hash, "engine_input_hash": entry.engine_input_hash, "engine_build_id": entry.engine_build_id,
                "validation_status": entry.validation_status.value, "target_outcomes": row["target_outcomes"],
                "outcome": row["outcome"], "observation_quantities": row["observation_quantities"],
                "model_scope_issues": row["model_scope_issues"],
                "design_projection_hash": entry.design_projection_hash, "targets_digest": entry.targets_digest,
                "physics_input_hash": run_physics, "runtime_kind": entry.runtime_kind.value},
        "current": {key: identity[key] for key in ("project_hash", "engine_input_hash", "design_projection_hash", "targets_digest",
                                                   "physics_input_hash", "engine_build_id", "check")},
        "integrity": integrity,
    }


__all__ = ["CONFIGURATIONS_CHANGED", "ENGINE_REQUEST_NOT_AVAILABLE", "FRESHNESS_REASONS", "FRESHNESS_STATES", "PHYSICS_REASONS",
           "RUN_NOT_READABLE", "RunNotReadable", "physics_and_requirements", "run_freshness"]
