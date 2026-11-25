"""Projects with configurations: every configuration run as its own Project, written side by side, and their comparison.

A Project may declare ``configurations`` (``ProjectDocumentV2.configurations``): named states of one design -- the
positions of a zoom, the states of a switched system. ``run_configured_project(project, output_dir="run")`` (``os-optics
project run project.json --out-dir run``) evaluates each configuration as the ordinary Project
``project.configured(id)``: its own Engine input (and input hash), its own budgets, its own complete run. A configuration's
run is exactly the run of that Project on its own. The run directory::

    run/
      manifest.json             <- start here: the Project, the run's configurations input hash, per configuration its
                                   directory, Project hash, Engine input and result hashes, execution and the SHA-256
                                   of its manifest; the files below with their sizes and SHA-256
      project.json              the Project with its configurations
      comparison.json           the configurations side by side (AnalysisResultV1, analysis CONFIGURATIONS;
                                os_optics.analysis.configurations)
      configurations/<id>/      an ordinary run directory of the configuration's Project (manifest.json, project.json,
                                simulation_result.json, local_validation.json, optical_scene.json, ...): every reader of
                                a run directory -- 'os-optics results ...', 'os-optics view', load_run -- opens it

Budgets apply per configuration (the Project's ``compute_budget``, with a configuration's own fields in place of the
Project's): a configuration that runs out of a budget reports its own ``PARTIAL`` execution and termination, and the
other configurations keep their complete results. A configuration whose Project cannot be lowered is recorded ``FAILED``
(termination ``INVALID_INPUT``) with the problems of its lowering, each located ``configuration <id> ...``, and no Engine
input hash; when no selected configuration can be lowered the run is refused before anything is traced
(``LocalRunError``) and the output directory is left as it was. A configuration whose Engine execution fails is
recorded ``FAILED`` with its problems; a ``FAILED`` configuration has no run directory, the others are written.

Identity. The run's ``engine_input_hash`` is ``hash_exact({"configurations": [[id, engine_input_hash], ...]})`` over the
configurations it ran, in the Project's order (``configurations_input_hash``); each configuration's input hash is that of
its own Engine request. There is no overall PASS: each configuration keeps its own status, outcome and requirements.

``verify_configurations_run`` re-checks a run directory offline (the manifest's file list, every configuration's run
directory with ``verify_run_artifacts``, each configuration's Project against ``configured(id)`` of the Project, the input
hashes and the comparison's binding); ``verify_run_artifacts`` and ``verify_run_files`` do the same for such a directory.
``configurations_freshness`` says, per configuration, whether the run is current for a Project.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .._vendor.shared_contracts.optical_design import LocalRuntimeKindV1, ProjectDocumentV2, validated_project_ref
from .._vendor.shared_contracts.optical_view import OpticalViewSettingsV1
from .._version import DISTRIBUTION_NAME, ENGINE_VERSION
from ..analysis.configurations import ConfigurationEvidence, configurations_comparison
from ..contracts.analysis import AnalysisResultV1
from ..manifest import hash_exact
from ..optical_view import canonical_json_bytes
from ..run_control import RunStopped, active_monitor, begin_attempt, end_attempt, phase
from ..validation.ray_paths import RAY_DATA_FULL, RAY_DATA_LEVELS
from .artifacts import (
    COMPARISON_FILE,
    CONFIGURATIONS_DIRECTORY,
    JSON_MEDIA_TYPE,
    LOCAL_VALIDATION_FILE,
    MANIFEST_FILE,
    PROJECT_FILE,
    SIMULATION_RESULT_FILE,
    RUN_FILES,
    _commit_run,
    _open_run_write,
    _stage_run,
    _staging_order,
    _verify_run,
    run_artifact_files,
    write_run_files,
)
from .catalog import load_project
from .errors import LocalRunError, LocalRunErrorCode, LocalRunProblem, fail
from .evaluate import EvaluationSession
from .lowering import LoweredProject
from .run import LocalOpticalRun, completed_write, session_run, stop_run

CONFIGURATIONS_RUN_SCHEMA = "opensci.local_configurations_run"
CONFIGURATIONS_RUN_VERSION = "1.0"
CONFIGURATIONS_NOTICE = (
    "Local pure-optical run of a Project with configurations: each configuration is the run of its own Project in "
    "configurations/<id> (an ordinary run directory), from one os-optics execution per configuration on this device; "
    "comparison.json sets them side by side."
)
#: (role, file name, media type, what it is) of the top-level files, in this order.
CONFIGURATIONS_FILES = (
    ("project", PROJECT_FILE, JSON_MEDIA_TYPE, "the Project with its configurations"),
    ("comparison", COMPARISON_FILE, JSON_MEDIA_TYPE, "the configurations side by side (analysis CONFIGURATIONS)"),
)
FAILED = "FAILED"
COMPLETED = "COMPLETED"
NUMERICAL_FAILURE = "NUMERICAL_FAILURE"
INVALID_INPUT = "INVALID_INPUT"
#: The freshness reason of a run whose configurations are not the Project's (added, removed, or not run).
CONFIGURATIONS_CHANGED = "CONFIGURATIONS_CHANGED"


def configuration_directory(configuration_id: str) -> str:
    """The run directory of a configuration, relative to the run: ``configurations/<id>``."""

    return f"{CONFIGURATIONS_DIRECTORY}/{configuration_id}"


def configurations_input_hash(pairs: Sequence[tuple[str, str | None]]) -> str:
    """The input hash of a run of configurations: ``hash_exact({"configurations": [[id, engine_input_hash], ...]})``; a
    configuration that could not be lowered has no Engine input and enters with ``null``."""

    return hash_exact({"configurations": [[cid, digest] for cid, digest in pairs]})


def select_configurations(document: ProjectDocumentV2, configuration_ids: Sequence[str] | None = None) -> tuple[str, ...]:
    """The configurations to run, in the Project's order: all of them, or those ``configuration_ids`` names
    (``CONFIGURATION_UNKNOWN`` for an unknown or repeated id, an empty selection or a Project without configurations)."""

    declared = [c.configuration_id for c in document.configurations or ()]
    if not declared:
        raise fail(LocalRunErrorCode.CONFIGURATION_UNKNOWN, "the Project declares no configurations: run it as one Project",
                   "configurations")
    if configuration_ids is None:
        return tuple(declared)
    chosen = list(configuration_ids)
    if not chosen or not all(isinstance(i, str) for i in chosen):
        raise fail(LocalRunErrorCode.CONFIGURATION_UNKNOWN, "a configuration selection is a non-empty list of configuration ids")
    problems = [LocalRunProblem(LocalRunErrorCode.CONFIGURATION_UNKNOWN,
                                f"the Project declares no configuration {cid!r} (it declares {', '.join(declared)})")
                for cid in dict.fromkeys(chosen) if cid not in declared]
    if len(set(chosen)) != len(chosen):
        problems.append(LocalRunProblem(LocalRunErrorCode.CONFIGURATION_UNKNOWN, "a configuration is selected twice"))
    if problems:
        raise LocalRunError(problems)
    return tuple(cid for cid in declared if cid in chosen)


def configured_project(project: Any, configuration_id: str | None = None) -> ProjectDocumentV2:
    """The Project one state is evaluated on: ``project.configured(configuration_id)`` for a Project that declares
    configurations (``CONFIGURATION_UNKNOWN`` for an id it does not declare, or for an id given to a Project without
    configurations); the Project itself when no id is given. A Project with configurations and no id is returned as it
    is: lowering it refuses with ``CONFIGURATION_NOT_SELECTED`` -- no configuration is chosen for the caller."""

    document = load_project(project)
    if configuration_id is None:
        return document
    if not isinstance(configuration_id, str):
        raise fail(LocalRunErrorCode.CONFIGURATION_UNKNOWN, "configuration_id is the id of one of the Project's configurations (a string)",
                   "configurations")
    (chosen,) = select_configurations(document, [configuration_id])
    return document.configured(chosen)


@dataclass(frozen=True)
class ConfigurationLowering:
    """One selected configuration lowered: its ``LoweredProject``, or (``lowered`` ``None``) the ``problems`` that keep
    its Project from being simulated, each located ``configuration <id> ...``."""

    configuration_id: str
    lowered: LoweredProject | None
    problems: tuple[LocalRunProblem, ...] = ()


def lower_configurations(project: Any, catalog: Any = None, *, model_kind: str | None = None,
                         configuration_ids: Sequence[str] | None = None, check_solids: bool = True,
                         session: EvaluationSession | None = None) -> tuple[ConfigurationLowering, ...]:
    """Lower each selected configuration's Project (``lower_project(project.configured(id))``), each on its own: a
    configuration that cannot be lowered keeps its problems (``ConfigurationLowering.problems``, located ``configuration
    <id> ...``) and the others are lowered. When no selected configuration can be lowered the call refuses with every
    problem (``LocalRunError``). ``session``: the ``EvaluationSession`` whose catalogs and compiled elements the lowerings
    share (default: one of their own, with ``catalog`` and ``model_kind``); configurations change poses, sources and
    budgets, never which models the components use, so each element is compiled once."""

    document = load_project(project)
    session = session if session is not None else EvaluationSession(catalog=catalog, model_kind=model_kind)
    lowerings: list[ConfigurationLowering] = []
    for cid in select_configurations(document, configuration_ids):
        try:
            lowerings.append(ConfigurationLowering(cid, session.lowering(document.configured(cid), check_solids=check_solids)))
        except LocalRunError as exc:
            lowerings.append(ConfigurationLowering(cid, None, tuple(
                replace(p, location=f"configuration {cid}" + (f" {p.location}" if p.location else "")) for p in exc.problems)))
    if all(item.lowered is None for item in lowerings):
        raise LocalRunError([problem for item in lowerings for problem in item.problems])
    return tuple(lowerings)


@dataclass(frozen=True)
class ConfigurationRun:
    """One configuration of a run: its Project (``configured(id)``), Engine input hash and run -- or, for a run the
    Engine could not complete, the ``problems`` that stopped it (``run`` is then ``None``). A configuration whose Project
    could not be lowered has no Engine input (``engine_input_hash`` ``None``) and the problems of its lowering."""

    configuration_id: str
    label: str | None
    project: ProjectDocumentV2
    engine_input_hash: str | None
    run: LocalOpticalRun | None
    problems: tuple[LocalRunProblem, ...] = ()

    @property
    def directory(self) -> str:
        return configuration_directory(self.configuration_id)

    @property
    def execution(self) -> dict[str, Any]:
        """The execution record (JSON): the run's own, or ``FAILED`` for a run that produced no result."""

        if self.run is not None:
            outcome = self.run.local_validation.simulation.outcome
            if outcome is not None:
                return outcome.execution.model_dump(mode="json")
            return self.run.result.outcome.execution.model_dump(mode="json")  # type: ignore[union-attr]
        numerical = any(NUMERICAL_FAILURE in p.message for p in self.problems)
        return {"outcome": FAILED, "termination": NUMERICAL_FAILURE if numerical else INVALID_INPUT,
                "reason": self.problems[0].message if self.problems else "", "complete": False, "budgets": {}}

    def evidence(self) -> ConfigurationEvidence:
        return ConfigurationEvidence(
            configuration_id=self.configuration_id, label=self.label, directory=self.directory, engine_input_hash=self.engine_input_hash,
            execution=self.execution, record=None if self.run is None else self.run.local_validation,
            result=None if self.run is None else self.run.result, project=self.project,
            problems=tuple(p.as_dict() for p in self.problems))


@dataclass(frozen=True)
class ConfigurationsRunArtifacts:
    """Where a run of configurations was written: ``directory``, its ``manifest`` and each configuration's run
    directory (``configuration_paths``: id -> path, the configurations that have one)."""

    directory: Path
    manifest: dict[str, Any]

    @property
    def manifest_path(self) -> Path:
        return self.directory / MANIFEST_FILE

    @property
    def comparison_path(self) -> Path:
        return self.directory / COMPARISON_FILE

    @property
    def configuration_paths(self) -> dict[str, Path]:
        return {entry["configuration_id"]: self.directory / entry["directory"] for entry in self.manifest["configurations"]
                if entry["manifest_sha256"] is not None}


@dataclass(frozen=True)
class ConfiguredProjectRun:
    """The run of a Project with configurations (module docstring): ``configurations`` in the Project's order,
    ``engine_input_hash`` over them, the ``comparison`` document, the ``engine_build_id`` of the Engine that ran them
    and, once written, ``artifacts``."""

    project: ProjectDocumentV2
    configurations: tuple[ConfigurationRun, ...]
    engine_input_hash: str
    comparison: AnalysisResultV1
    engine_build_id: str
    artifacts: ConfigurationsRunArtifacts | None = None

    @property
    def complete(self) -> bool:
        """Every configuration's run completed (its execution outcome is ``COMPLETED``)."""

        return all(c.execution["outcome"] == COMPLETED for c in self.configurations)

    def configuration(self, configuration_id: str) -> ConfigurationRun:
        for item in self.configurations:
            if item.configuration_id == configuration_id:
                return item
        raise KeyError(configuration_id)

    def write_artifacts(self, out_dir: Path | str) -> ConfigurationsRunArtifacts:
        return write_configurations_artifacts(self, out_dir)

    def summary(self) -> dict[str, Any]:
        """A JSON-safe report: the Project, the run's input hash, the execution of each configuration and the comparison;
        per configuration its run's own ``summary()`` (status, outcome, targets, issues, record), or its problems."""

        artifacts = None
        if self.artifacts is not None:
            artifacts = {"directory": str(self.artifacts.directory), "manifest": str(self.artifacts.manifest_path),
                         "comparison": str(self.artifacts.comparison_path),
                         "configurations": {cid: str(path) for cid, path in self.artifacts.configuration_paths.items()}}
        return {
            "project_hash": self.project.content_hash(),
            "design_intent": self.project.effective_design_intent.value,
            "engine_input_hash": self.engine_input_hash,
            "execution": _execution_summary(self.configurations),
            "configurations": [
                {"configuration_id": c.configuration_id, "label": c.label, "directory": c.directory,
                 "engine_input_hash": c.engine_input_hash, "execution": c.execution,
                 "problems": [p.as_dict() for p in c.problems],
                 "summary": None if c.run is None else c.run.summary()}
                for c in self.configurations],
            "comparison": self.comparison.model_dump(mode="json", by_alias=True),
            "artifacts": artifacts,
        }


def _execution_summary(configurations: Sequence[ConfigurationRun]) -> dict[str, Any]:
    completed = [c.configuration_id for c in configurations if c.execution["outcome"] == COMPLETED]
    return {"complete": len(completed) == len(configurations), "completed": completed,
            "not_completed": [{"configuration_id": c.configuration_id, "outcome": c.execution["outcome"],
                               "termination": c.execution["termination"]}
                              for c in configurations if c.execution["outcome"] != COMPLETED]}


def run_configured_project(
    project: Any,
    *,
    catalog: Any = None,
    model_kind: str | None = None,
    runtime_kind: LocalRuntimeKindV1 | str | None = None,
    output_dir: Path | str | None = None,
    view_settings: OpticalViewSettingsV1 | None = None,
    ray_data: str = RAY_DATA_FULL,
    configuration_ids: Sequence[str] | None = None,
) -> ConfiguredProjectRun:
    """Run every configuration of a Project (module docstring), or those ``configuration_ids`` names, each as its own
    Project -- the run ``run_optical_project`` makes of it, with the same ``catalog``, ``model_kind``, ``runtime_kind``,
    ``view_settings`` and ``ray_data`` -- and set them side by side (``comparison``). ``output_dir``: write the run
    directory there, once every configuration has run (a run that stops or fails before that leaves the directory as it
    found it: ``local/artifacts.py``). The configurations run through one ``EvaluationSession``: each element is compiled once, each
    configuration is lowered once, and the Engine build is identified once.

    Every selected configuration is lowered before any is traced. A configuration whose Project cannot be lowered is kept
    as ``FAILED`` (termination ``INVALID_INPUT``, no Engine input hash) with its problems, and the others run; a selection
    that names no configuration of the Project, or of which none can be lowered, raises ``LocalRunError``: nothing runs
    and ``output_dir`` is left as it was. A configuration whose Engine execution fails is kept as ``FAILED`` with its
    problems; budgets that run out make that configuration's run ``PARTIAL``, never another's. A monitored run that is
    stopped (``run_control.monitored``) raises ``RunStopped`` with its record, written into ``output_dir``
    (``run_stopped.json``), as ``run_optical_project`` does."""

    if ray_data not in RAY_DATA_LEVELS:
        raise ValueError(f"ray_data is one of {', '.join(RAY_DATA_LEVELS)}, not {ray_data!r}")
    document = load_project(project)
    held: dict[str, Any] = {}  # (the run being written: what a stop after its move into place answers with)
    attempt = begin_attempt()
    try:
        # (one session for every configuration: the catalog, the compiled elements and the Engine build id are shared, and
        # each configuration's run uses the lowering made here, before anything is traced)
        session = EvaluationSession(catalog=catalog, model_kind=model_kind, runtime_kind=runtime_kind)
        lowerings = lower_configurations(document, catalog, model_kind=model_kind, configuration_ids=configuration_ids,
                                         session=session)
        run = _run_configured(document, session, lowerings, view_settings=view_settings, ray_data=ray_data)
        if output_dir is not None:
            held["run"] = run
            run = replace(run, artifacts=run.write_artifacts(output_dir))
        return run
    except RunStopped as stopped:
        if (written := completed_write(held, output_dir)) is not None:
            return written
        stopped.stopped = None  # (the record of the run of the whole Project, not of the configuration it stopped in)
        raise stop_run(stopped, document, active_monitor(), output_dir=output_dir) from None
    except KeyboardInterrupt:
        if (written := completed_write(held, output_dir)) is not None:
            return written  # (every configuration's run is in place: the Ctrl-C came after the run completed)
        raise
    finally:
        end_attempt(attempt)


def _run_configured(document: ProjectDocumentV2, session: EvaluationSession, lowerings: Sequence[ConfigurationLowering], *,
                    view_settings: Any, ray_data: str) -> ConfiguredProjectRun:
    labels = {c.configuration_id: c.label for c in document.configurations or ()}
    runs = []
    for item in lowerings:
        cid, low = item.configuration_id, item.lowered
        configured = document.configured(cid)
        if low is None:  # (its Project cannot be lowered: FAILED / INVALID_INPUT with its problems, nothing traced)
            runs.append(ConfigurationRun(cid, labels[cid], configured, None, None, item.problems))
            continue
        try:
            run = session_run(session, configured, lowered=low, view_settings=view_settings, ray_data=ray_data)
        except LocalRunError as exc:
            runs.append(ConfigurationRun(cid, labels[cid], configured, low.engine_input_hash, None,
                                         tuple(replace(p, location=f"configuration {cid}" + (f" {p.location}" if p.location else ""))
                                               for p in exc.problems)))
            continue
        runs.append(ConfigurationRun(cid, labels[cid], configured, low.engine_input_hash, run))
    input_hash = configurations_input_hash([(c.configuration_id, c.engine_input_hash) for c in runs])
    comparison = configurations_comparison(document, [c.evidence() for c in runs], engine_input_hash=input_hash,
                                           engine_build_id=session.build_id)
    return ConfiguredProjectRun(project=document, configurations=tuple(runs), engine_input_hash=input_hash, comparison=comparison,
                                engine_build_id=session.build_id)


# ------------------------------------------------------------------------------------------------------------ files


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _top_files(run: ConfiguredProjectRun) -> dict[str, bytes]:
    return {PROJECT_FILE: canonical_json_bytes(run.project.payload()),
            COMPARISON_FILE: canonical_json_bytes(run.comparison.model_dump(mode="json", by_alias=True))}


def _manifest(run: ConfiguredProjectRun, top: Mapping[str, bytes], manifests: Mapping[str, bytes | None]) -> dict[str, Any]:
    """The run's manifest; ``manifests``: per configuration the bytes of its run directory's manifest (None: no run)."""

    entries = []
    for c in run.configurations:
        record = None if c.run is None else c.run.local_validation
        data = manifests.get(c.configuration_id)
        entries.append({
            "configuration_id": c.configuration_id, "label": c.label, "directory": c.directory,
            "project_hash": c.project.content_hash(), "engine_input_hash": c.engine_input_hash,
            "result_hash": None if record is None else record.result_hash.digest,
            "validation_status": None if record is None else record.validation_status.value,
            "execution": c.execution, "problems": [p.as_dict() for p in c.problems],
            "manifest_sha256": None if data is None else _sha256(data),
        })
    return {
        "schema": CONFIGURATIONS_RUN_SCHEMA,
        "schema_version": CONFIGURATIONS_RUN_VERSION,
        "notice": CONFIGURATIONS_NOTICE,
        "project": validated_project_ref(run.project).model_dump(mode="json", exclude_none=True),
        "engine_input_hash": run.engine_input_hash,
        "engine": {"engine_package": DISTRIBUTION_NAME, "engine_version": ENGINE_VERSION, "engine_build_id": run.engine_build_id},
        "execution": _execution_summary(run.configurations),
        "configurations": entries,
        "comparison": {"analysis": run.comparison.analysis, "execution": run.comparison.execution.model_dump(mode="json")},
        "files": [{"role": role, "name": name, "media_type": media, "description": what, "bytes": len(top[name]),
                   "sha256": _sha256(top[name])} for role, name, media, what in CONFIGURATIONS_FILES],
    }


def configurations_run_files(run: ConfiguredProjectRun) -> dict[str, bytes]:
    """Every file of the run directory, in memory: ``configurations/<id>/<file>`` for each configuration's run directory,
    then ``project.json``, ``comparison.json`` and ``manifest.json`` (last)."""

    files: dict[str, bytes] = {}
    manifests: dict[str, bytes | None] = {}
    for c in run.configurations:
        if c.run is None:
            manifests[c.configuration_id] = None
            continue
        sub = run_artifact_files(c.run)
        manifests[c.configuration_id] = sub[MANIFEST_FILE]
        files.update({f"{c.directory}/{name}": data for name, data in sub.items()})
    top = _top_files(run)
    files.update(top)
    files[MANIFEST_FILE] = canonical_json_bytes(_manifest(run, top, manifests))
    return files


def write_configurations_artifacts(run: ConfiguredProjectRun, out_dir: Path | str) -> ConfigurationsRunArtifacts:
    """Write the run directory (module docstring) by the run directory rule (``local/artifacts.py``): every file of every
    configuration's run directory, then ``comparison.json``, ``project.json`` and the manifest are written to their
    temporary names, and the whole run is moved into place once all are written -- each configuration's run directory, the
    directory's own files, its manifest last. A configuration without a run has what an earlier run left in its directory
    removed. A write that stops or fails before the move leaves the directory as it found it. The bytes are those of
    ``configurations_run_files``. ``configurations/`` or a ``configurations/<id>`` of the run that is a link is refused
    (``RunDirectoryLinkError``), before anything is written: a run never writes through a link."""

    top = _top_files(run)
    names = [f"{c.directory}/{name}" for c in run.configurations if c.run is not None
             for name in (*(name for _, name, _, _ in RUN_FILES), MANIFEST_FILE)]
    write = _open_run_write(out_dir, (*names, *top, MANIFEST_FILE), top[PROJECT_FILE],
                            [c.directory for c in run.configurations if c.run is None],
                            {f"{c.directory}/{PROJECT_FILE}": canonical_json_bytes(c.run.project.payload())
                             for c in run.configurations if c.run is not None})
    manifests: dict[str, bytes | None] = {}
    try:
        for c in run.configurations:
            if c.run is None:
                manifests[c.configuration_id] = None
                continue
            with phase("write", unit="bytes"):
                manifests[c.configuration_id] = canonical_json_bytes(_stage_run(write.staged, c.run, f"{c.directory}/"))
        for name in _staging_order(top):
            write.staged.stage(name, top[name])
        manifest = _manifest(run, top, manifests)
        write.staged.stage(MANIFEST_FILE, canonical_json_bytes(manifest))
        _commit_run(write, ConfigurationsRunArtifacts(directory=write.directory, manifest=manifest))
    finally:
        write.staged.discard()
    return ConfigurationsRunArtifacts(directory=write.directory, manifest=manifest)


def write_configurations_files(files: Mapping[str, bytes], out_dir: Path | str) -> ConfigurationsRunArtifacts:
    """Write a run directory from its files held in memory (``configurations_run_files``) by the same rule."""

    manifest = json.loads(files[MANIFEST_FILE])
    without_run = [configuration_directory(str(entry["configuration_id"])) for entry in manifest.get("configurations") or ()
                   if isinstance(entry, dict) and entry.get("manifest_sha256") is None and "configuration_id" in entry]
    written = write_run_files(files, out_dir, without_run=without_run)
    return ConfigurationsRunArtifacts(directory=written.directory, manifest=written.manifest)


def configuration_files(files: Mapping[str, Any], configuration_id: str) -> dict[str, Any]:
    """The files of one configuration's run directory out of a run's files held in memory (names without the
    ``configurations/<id>/`` prefix): an ordinary run's files, for ``load_run``, ``json_api.run_results``, ..."""

    prefix = configuration_directory(configuration_id) + "/"
    return {name[len(prefix):]: data for name, data in files.items() if isinstance(name, str) and name.startswith(prefix)}


# ----------------------------------------------------------------------------------------------------------- reading


def is_configurations_manifest(manifest: Any) -> bool:
    return isinstance(manifest, dict) and manifest.get("schema") == CONFIGURATIONS_RUN_SCHEMA


def run_reader(run: Any) -> Callable[[str], bytes] | None:
    """A reader of the files of a run of configurations -- a directory, its ``{name: bytes or text}`` files or a
    ``run_project`` reply (its ``files``) -- or ``None`` when ``run`` is not one (an ordinary run, a record, ...)."""

    if isinstance(run, Mapping) and isinstance(run.get("files"), Mapping):
        return run_reader(run["files"])
    if isinstance(run, (str, Path)):
        directory = Path(run)
        if not (directory / MANIFEST_FILE).is_file():
            return None

        def read_path(name: str) -> bytes:
            return (directory / name).read_bytes()

        read: Callable[[str], bytes] = read_path
    elif isinstance(run, Mapping) and MANIFEST_FILE in run:
        def read_files(name: str) -> bytes:
            data = run.get(name)
            if data is None:
                raise FileNotFoundError(f"no file named {name}")
            if isinstance(data, str):
                return data.encode("utf-8")
            return bytes(data.tobytes() if isinstance(data, memoryview) else data)

        read = read_files
    else:
        return None
    try:
        manifest = json.loads(read(MANIFEST_FILE))
    except (OSError, ValueError, TypeError, RecursionError):
        return None
    return read if is_configurations_manifest(manifest) else None


def _json(read: Callable[[str], bytes], name: str, problems: list[str]) -> Any:
    try:
        return json.loads(read(name))
    except FileNotFoundError:
        problems.append(f"{name}: missing")
    except OSError as exc:
        problems.append(f"{name}: cannot be read ({exc.strerror or type(exc).__name__})")
    except (ValueError, RecursionError):
        problems.append(f"{name}: not valid JSON")
    return None


def _prefixed(read: Callable[[str], bytes], prefix: str) -> Callable[[str], bytes]:
    return lambda name: read(f"{prefix}/{name}")


def verify_configurations_run(read: Callable[[str], bytes]) -> tuple[str, ...]:
    """Check a run of configurations read by ``read(name) -> bytes`` (module docstring). Returns the problems found
    (empty = consistent)."""

    problems: list[str] = []
    try:
        return _verify_configurations(read, problems)
    except (AttributeError, KeyError, TypeError, ValueError, RecursionError, ArithmeticError) as exc:
        return (*problems, f"the run files cannot be checked: a file has an unexpected structure ({type(exc).__name__})")


def _verify_configurations(read: Callable[[str], bytes], problems: list[str]) -> tuple[str, ...]:  # noqa: C901 - one check list
    from ..project_document import parse_project_document

    manifest = _json(read, MANIFEST_FILE, problems)
    if not is_configurations_manifest(manifest) or manifest.get("schema_version") != CONFIGURATIONS_RUN_VERSION:
        return (*problems, f"{MANIFEST_FILE}: not an {CONFIGURATIONS_RUN_SCHEMA} {CONFIGURATIONS_RUN_VERSION} manifest")
    listed = {entry.get("role"): entry for entry in manifest.get("files") or [] if isinstance(entry, dict)}
    for role, name, _media, _what in CONFIGURATIONS_FILES:
        entry = listed.get(role)
        if entry is None or entry.get("name") != name:
            problems.append(f"{MANIFEST_FILE}: the {role} file is not listed as {name}")
            continue
        try:
            data = read(name)
        except OSError:
            problems.append(f"{name}: missing")
            continue
        if len(data) != entry.get("bytes") or _sha256(data) != entry.get("sha256"):
            problems.append(f"{name}: size or SHA-256 differs from {MANIFEST_FILE}")
    project = None
    document = _json(read, PROJECT_FILE, problems)
    if document is not None:
        try:
            project = parse_project_document(document)
        except ValidationError as exc:
            problems.append(f"{PROJECT_FILE}: not a valid ProjectDocumentV2 ({exc.error_count()} errors)")
    if project is not None:
        if not project.configurations:
            problems.append(f"{PROJECT_FILE}: the Project declares no configurations")
        if (manifest.get("project") or {}).get("project_hash", {}).get("digest") != project.content_hash():
            problems.append(f"{MANIFEST_FILE}: its Project hash differs from {PROJECT_FILE}'s")
    declared = [c.configuration_id for c in (project.configurations or ())] if project is not None else []
    entries = manifest.get("configurations")
    if not isinstance(entries, list) or not all(isinstance(e, dict) for e in entries):
        return (*problems, f"{MANIFEST_FILE}: its configurations are not a list of entries")
    ids = [e.get("configuration_id") for e in entries]
    if project is not None and [i for i in declared if i in ids] != ids:
        problems.append(f"{MANIFEST_FILE}: its configurations are not configurations of the Project, in its order")
    pairs = []
    for entry in entries:
        cid = str(entry.get("configuration_id"))
        where = configuration_directory(cid)
        if entry.get("directory") != where:
            problems.append(f"{MANIFEST_FILE}: configuration {cid} is not in {where}")
            continue
        pairs.append((cid, entry.get("engine_input_hash")))
        if entry.get("manifest_sha256") is None:
            if (entry.get("execution") or {}).get("outcome") != FAILED:
                problems.append(f"{MANIFEST_FILE}: configuration {cid} has no run directory but did not fail")
            continue
        sub = _prefixed(read, where)
        try:
            sub_manifest = sub(MANIFEST_FILE)
        except OSError:
            problems.append(f"{where}/{MANIFEST_FILE}: missing")
            continue
        if _sha256(sub_manifest) != entry.get("manifest_sha256"):
            problems.append(f"{where}/{MANIFEST_FILE}: SHA-256 differs from {MANIFEST_FILE}")
        problems.extend(f"{where}/{p}" for p in _verify_run(sub))
        configured = None
        if project is not None and cid in declared:
            configured = project.configured(cid)
            try:
                if sub(PROJECT_FILE) != canonical_json_bytes(configured.payload()):
                    problems.append(f"{where}/{PROJECT_FILE}: is not configuration {cid} of {PROJECT_FILE}")
            except OSError:
                pass  # reported by the run directory's own check
            if entry.get("project_hash") != configured.content_hash():
                problems.append(f"{MANIFEST_FILE}: configuration {cid}'s Project hash is not that of configuration {cid}")
        record = _json(sub, LOCAL_VALIDATION_FILE, [])
        if isinstance(record, dict):
            if (record.get("input_hash") or {}).get("digest") != entry.get("engine_input_hash"):
                problems.append(f"{MANIFEST_FILE}: configuration {cid}'s Engine input hash differs from its record's")
            if (record.get("result_hash") or {}).get("digest") != entry.get("result_hash"):
                problems.append(f"{MANIFEST_FILE}: configuration {cid}'s Engine result hash differs from its record's")
    if configurations_input_hash(pairs) != manifest.get("engine_input_hash"):
        problems.append(f"{MANIFEST_FILE}: its engine_input_hash is not the hash of its configurations' input hashes")
    comparison = _json(read, COMPARISON_FILE, [])
    if comparison is not None:
        try:
            parsed = AnalysisResultV1.model_validate(comparison)
        except ValidationError as exc:
            problems.append(f"{COMPARISON_FILE}: not a valid analysis document ({exc.error_count()} errors)")
        else:
            if parsed.analysis != "CONFIGURATIONS" or parsed.identity.physical.engine_input_hash != manifest.get("engine_input_hash"):
                problems.append(f"{COMPARISON_FILE}: is not the comparison of this run's configurations")
            if [row.get("configuration_id") for row in parsed.arrays.get("configurations", [])] != ids:  # type: ignore[union-attr]
                problems.append(f"{COMPARISON_FILE}: compares other configurations than {MANIFEST_FILE} lists")
    return tuple(problems)


def comparison_from_files(read: Callable[[str], bytes], *, paraxial: Mapping[str, Any] | None = None, catalog: Any = None) -> AnalysisResultV1:
    """The comparison of a written run of configurations, recomputed from its files (with an optional ``paraxial``
    request, ``os_optics.analysis.configurations``); ``ValueError`` when the files are not such a run."""

    from .._vendor.shared_contracts.optical_design import LocalValidationRecordV1
    from ..contracts.results import ValidationResult
    from ..project_document import parse_project_document

    manifest = json.loads(read(MANIFEST_FILE))
    if not is_configurations_manifest(manifest):
        raise ValueError("not a run of configurations")
    project = parse_project_document(json.loads(read(PROJECT_FILE)))
    evidence = []
    for entry in manifest["configurations"]:
        cid = entry["configuration_id"]
        record = result = None
        if entry.get("manifest_sha256") is not None:
            sub = _prefixed(read, entry["directory"])
            record = LocalValidationRecordV1.model_validate(json.loads(sub(LOCAL_VALIDATION_FILE)))
            result = ValidationResult.model_validate(json.loads(sub(SIMULATION_RESULT_FILE)))
        evidence.append(ConfigurationEvidence(
            configuration_id=cid, label=entry.get("label"), directory=entry["directory"], engine_input_hash=entry["engine_input_hash"],
            execution=entry["execution"], record=record, result=result, project=project.configured(cid),
            problems=tuple(entry.get("problems") or ())))
    return configurations_comparison(project, evidence, engine_input_hash=manifest["engine_input_hash"], paraxial=paraxial,
                                     catalog=catalog)


# --------------------------------------------------------------------------------------------------------- freshness


def configurations_freshness(project: Any, read: Callable[[str], bytes], *, catalog: Any = None, verify: bool = False) -> dict[str, Any]:
    """The freshness of a run of configurations (read by ``read``) for ``project``, per configuration: each run
    configuration's own ``run_freshness`` against ``project.configured(id)`` (``configurations``), and overall ``CURRENT``
    exactly when the run holds a completed run of every configuration of the Project, each ``CURRENT``. Otherwise
    ``HISTORICAL`` (a configuration ``HISTORICAL``, ``CONFIGURATIONS_CHANGED`` when the Project's configurations are not
    the ones run -- added, removed or not run) or ``UNDETERMINED``. ``verify``: also verify the run's files
    (``verify_configurations_run``) and report it in ``integrity``, which never changes the freshness.

    ``coverage``: ``declared`` (the Project's configurations), ``run`` (those the run holds a completed run of and the
    Project still declares), ``not_run``, ``no_longer_declared`` (run, but not declared any more) and ``complete`` (every
    declared configuration was run). ``physics`` / ``requirements``: the two identities of ``run_freshness`` taken over
    the configurations in ``coverage.run`` -- ``physics`` ``CURRENT`` when each of them is (a subset run, or a zoom
    position added to the Project, leaves the unchanged configurations' physics ``CURRENT``; ``coverage`` names what is
    missing), ``HISTORICAL`` when one is, else ``UNDETERMINED``."""

    from ..results.model import RUN_FILE_NAMES
    from .freshness import NOT_VERIFIED, run_freshness

    document = load_project(project)
    manifest = json.loads(read(MANIFEST_FILE))
    integrity = dict(NOT_VERIFIED)
    if verify:
        problems = list(verify_configurations_run(read))
        integrity = {"checked": True, "consistent": not problems, "problems": problems}
    declared = [c.configuration_id for c in document.configurations or ()]
    rows, reasons = [], []
    run_ids = [str(e.get("configuration_id")) for e in manifest.get("configurations") or []]
    completed_ids = []
    for entry in manifest.get("configurations") or []:
        cid = str(entry.get("configuration_id"))
        if cid not in declared or entry.get("manifest_sha256") is None:
            continue
        completed_ids.append(cid)
        sub = _prefixed(read, str(entry.get("directory")))
        files = {}
        for name in RUN_FILE_NAMES:
            try:
                files[name] = sub(name)
            except OSError:
                continue
        rows.append({"configuration_id": cid, **run_freshness(document.configured(cid), files, catalog=catalog, verify=verify)})
    for row in rows:
        reasons.extend(r for r in row["reasons"] if r not in reasons)
    if completed_ids != declared:
        reasons.append(CONFIGURATIONS_CHANGED)
    states = {row["freshness"] for row in rows}
    if not reasons and states == {"CURRENT"}:
        freshness = "CURRENT"
    elif "HISTORICAL" in states or CONFIGURATIONS_CHANGED in reasons:
        freshness = "HISTORICAL"
    else:
        freshness = "UNDETERMINED"
    def over(part: str) -> str:
        found = {row[part]["freshness"] for row in rows}
        if "HISTORICAL" in found:
            return "HISTORICAL"
        return "CURRENT" if found == {"CURRENT"} else "UNDETERMINED"

    physics_reasons: list[str] = []
    for row in rows:
        physics_reasons.extend(r for r in row["physics"]["reasons"] if r not in physics_reasons)
    not_run = [cid for cid in declared if cid not in completed_ids]
    return {
        "freshness": freshness, "reasons": reasons,
        "physics": {"freshness": over("physics"), "reasons": physics_reasons},
        "requirements": {"freshness": over("requirements")},
        "coverage": {"declared": declared, "run": completed_ids, "not_run": not_run,
                     "no_longer_declared": [cid for cid in run_ids if cid not in declared], "complete": not not_run},
        "engine_build_matches": all(row["engine_build_matches"] for row in rows),
        "configurations": rows,
        "not_run": not_run,
        "run": {"project_hash": (manifest.get("project") or {}).get("project_hash", {}).get("digest"),
                "engine_input_hash": manifest.get("engine_input_hash"), "configuration_ids": run_ids},
        "current": {"project_hash": document.content_hash(), "configuration_ids": declared},
        "integrity": integrity,
    }


__all__ = [
    "COMPARISON_FILE",
    "CONFIGURATIONS_CHANGED",
    "CONFIGURATIONS_DIRECTORY",
    "CONFIGURATIONS_RUN_SCHEMA",
    "CONFIGURATIONS_RUN_VERSION",
    "ConfigurationLowering",
    "ConfigurationRun",
    "ConfigurationsRunArtifacts",
    "ConfiguredProjectRun",
    "comparison_from_files",
    "configuration_directory",
    "configured_project",
    "configuration_files",
    "configurations_freshness",
    "configurations_input_hash",
    "configurations_run_files",
    "is_configurations_manifest",
    "lower_configurations",
    "run_configured_project",
    "run_reader",
    "select_configurations",
    "verify_configurations_run",
    "write_configurations_artifacts",
    "write_configurations_files",
]
