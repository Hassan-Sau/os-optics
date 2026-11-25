"""The run directory: every artifact of one local run, bound together by one manifest.

``run_optical_project(project, output_dir="run")`` (or ``run.write_artifacts("run")``, or
``os-optics project run project.json --out-dir run``) writes::

    run/
      manifest.json           <- start here: status, identities, hashes, the file list, binding checks
      project.json            the ProjectDocumentV2 that was run (its osc1 hash is the record's project_hash)
      engine_request.json     the Engine ValidationRequest lowered from it (its eng-exact hash is the input_hash)
      simulation_result.json  the Engine ValidationResult (targets, observations, beam paths, issues, power budget, manifest)
      ray_data.json           every chief and pupil ray the Engine traced (opensci.ray_data): per path, the point,
                              direction and optical path length of each ray at every surface it reached (with
                              ``ray_data="summary"`` / ``--ray-data summary``: the chief rays only, stated in the file)
      local_validation.json   the LocalValidationRecordV1 (keep it; it is what a later submission carries)
      optical_scene.json      the Optical Simulation View scene (renderer-independent JSON)
      optical_scene.glb       the same view as a glTF 2.0 binary; open it in any glTF viewer

``os_optics.results`` (``load_run("run")``) and ``os-optics results ... run`` / ``os-optics export ...``
read the scientific results of a run directory: monitors, arrivals, rays, wavefronts, paths, the power budget.

All files come from ONE Engine execution: the scene and the record carry the same Engine input and result hashes, the
ray data carries the Engine input and trace hashes of the result, and the manifest states that check. Contents are deterministic for a given Project, catalog and Engine build (no
timestamps).

The run directory rule (``os_optics.output_files``: the rule of every output a command regenerates):

* **A completed run replaces the run in its directory.** Every file is first written to a temporary name beside its own
  (``.<name>.partial``), one at a time, each serialized piece by piece (``manifest.streaming``: the long arrays element by
  element) and hashed as it is written, so no two whole documents are held at once; the bytes are those of
  ``run_artifact_files``. Only when all are written is the run moved into place: each manifest the move removes is first
  copied beside it (``.manifest.json.earlier``), the stop record and the manifest there are removed, the files are
  renamed (``project.json`` first), what the earlier run held and the new one does not
  write is removed by name (a configuration's run directory, ``comparison.json``, the files of a single run where a run
  of configurations is written), the manifest is moved last and the copies are removed. A process killed during these
  steps -- or a step that fails there, a file held open elsewhere -- leaves no manifest, which is no run, and the copies,
  which list by name what the earlier run still holds there, so the same run started again -- or any other run --
  completes there and removes by name what the earlier run and the interrupted commit (its staged manifest) left there
  and it does not write. A run that finds such a staged manifest where it stages its own first keeps it under its own
  name (``.manifest.json.interrupted-<n>``), so a second interruption loses no record of what the first moved there;
  those are removed when a commit completes there. A file of the earlier run the commit cannot remove (held open
  elsewhere) stays and is recorded by name (``.manifest.json.leftover``), so the next run accepts and removes it; a name
  the commit did remove is not recorded. A run replaces only what the Engine wrote: a plain file at a run-file
  name is replaced when the Engine manifest there (or its copy a killed commit left) lists that name, or a staged
  manifest an interrupted commit left (``.manifest.json.partial``, or kept as ``.manifest.json.interrupted-<n>``) does --
  read by name, no file of an earlier run is hashed -- and a ``configurations/<id>`` the run writes or removes when the
  directory's manifest (or its copy, or its staged one) names it. Anything else there is the person's own: the run refuses
  (``RunDirectoryOccupiedError``) before anything is written. ``project.json``, which is also the name of a design file,
  is replaced only when it is missing, holds the Project being run or is listed with its present SHA-256 by the manifest
  there (or its copy) or by a staged manifest of a killed commit. The stop record, the copies, the kept staged manifests,
  the record of what a commit left over and the temporary names are the Engine's. Other files are never touched. A run never writes through a link (``RunDirectoryLinkError``).
* **An attempt that does not complete leaves the directory as it found it.** A refused Project, a usage error, a failure
  before the run is moved into place, a stop while tracing or while the files are staged: the temporary files of the
  attempt are removed and nothing else changes, so an earlier run there still verifies. A failure while the run is moved
  into place leaves what a process killed there leaves (above): any run then completes there. A stopped run adds its record, ``run_stopped.json``, beside it
  (``later_attempt`` tells a reader of the kept run). Whether the kept run is current for a Project is
  ``run_freshness``: the manifest binds the Project and Engine input hashes of the run it describes.
* **A run is verified where its bytes cross a trust boundary, and not again between two steps of one process.** The
  points that verify are ``project verify`` (``verify_run_artifacts``), a submission (``submission_document`` of a
  directory, a file map or a ``LocalOpticalRun``), the server's intake, every ``results`` / ``export`` command of the
  command line (``--no-verify`` reads as stored), and any read that asks (``verify=True`` of ``load_run``,
  ``results_from_files``, ``run_freshness``; ``results freshness --verify``). Every other read takes the files as stored
  and says so (``integrity.checked`` false).

``verify_run_artifacts("run")`` re-reads a run directory and checks every hash and cross-reference, offline;
``verify_run_files(files)`` runs the same checks on the directory's files held in memory (``{file name: bytes}``).
"""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from ..run_control import artifact_bytes, attempt_commits, phase
from .._vendor.shared_contracts.optical_design import LocalValidationRecordV1, ProjectDocumentV2, check_local_validation_binding
from .._vendor.shared_contracts.optical_view import OpticalSimulationSceneV1
from ..contracts.request import ValidationRequest
from ..contracts.results import ValidationResult
from ..manifest import hash_exact, hash_quantized
from ..manifest.streaming import dict_parts, hash_quantized_parts, json_parts, model_parts
from ..optical_view import GLB_FILE, SCENE_FILE, GlbFormatError, canonical_json_bytes, optical_view_files, parse_glb
from ..output_files import StagedFiles, staged_path
from ..project_document import parse_project_document
from ..validation.ray_paths import RAY_DATA_FULL, RAY_DATA_SCHEMA, RAY_PATHS_VERSION, TRACED, ray_data_recording, ray_paths_parts

if TYPE_CHECKING:  # pragma: no cover
    from .run import LocalOpticalRun

MANIFEST_FILE = "manifest.json"
PROJECT_FILE = "project.json"
ENGINE_REQUEST_FILE = "engine_request.json"
SIMULATION_RESULT_FILE = "simulation_result.json"
RAY_DATA_FILE = "ray_data.json"
LOCAL_VALIDATION_FILE = "local_validation.json"
RUN_MANIFEST_SCHEMA = "opensci.local_run"
# 1.2: ``engine_run.engine_package``, the distribution that produced the run (``os-optics``; the record's
# ``runtime.engine_package``). A 1.1 directory (os-optics 0.10.0 / 0.11.0) has none and still verifies.
RUN_MANIFEST_VERSION = "1.2"
RUN_MANIFEST_READ_VERSIONS = ("1.1", RUN_MANIFEST_VERSION)
RAY_DATA_ROLE = "ray_data"
#: The record of a run that stopped before it completed (``StoppedRun``), written into the directory the run was given:
#: beside an earlier run there, which stays as it was, or alone.
STOPPED_FILE = "run_stopped.json"
#: The copy of a manifest that a commit removes (``_commit_run``), written beside it before the commit removes anything and
#: removed when the commit ends. A commit killed, or failing, after the manifest is gone leaves it: the record of what the
#: earlier run still holds there, which the next run reads in place of the manifest (``_earlier_manifest``).
EARLIER_MANIFEST_FILE = ".manifest.json.earlier"
#: The staged manifest an interrupted commit left (``.manifest.json.partial``), kept under its own name -- this prefix and a
#: number -- by the next run before it stages its own manifest at that name (``_keep_interrupted``): each lists the files
#: its commit had moved there, which the next runs read by name (``_staged_manifests``) until a commit completes there.
INTERRUPTED_MANIFEST_PREFIX = ".manifest.json.interrupted-"
#: What a commit that completed could not remove of the earlier run (a file another program held open while the commit
#: removed it), by name relative to the run directory, with the ``configurations/<id>`` directories that hold it
#: (``_record_leftover``): the next run, the same one or another, accepts and removes those names as the Engine's. A
#: commit that completes rewrites it, or removes it when nothing is left over.
LEFTOVER_FILE = ".manifest.json.leftover"
LEFTOVER_SCHEMA = "opensci.run_leftover"
#: A run of a Project with configurations (``local/configurations.py``): each configuration's run directory in
#: ``configurations/<id>``, and the configurations side by side in ``comparison.json``.
CONFIGURATIONS_DIRECTORY = "configurations"
COMPARISON_FILE = "comparison.json"
JSON_MEDIA_TYPE = "application/json"
GLB_MEDIA_TYPE = "model/gltf-binary"
# (role, file name, media type, what it is) -- the manifest's file list, in this order.
RUN_FILES = (
    ("project", PROJECT_FILE, JSON_MEDIA_TYPE, "the Project that was run"),
    ("engine_request", ENGINE_REQUEST_FILE, JSON_MEDIA_TYPE, "the simulation input built from the Project ('os-optics "
                                                             "validate' runs it again)"),
    ("simulation_result", SIMULATION_RESULT_FILE, JSON_MEDIA_TYPE, "the simulation result: targets, beams at every observation "
                                                                   "plane, paths, power budget, issues"),
    (RAY_DATA_ROLE, RAY_DATA_FILE, JSON_MEDIA_TYPE, "every chief and pupil ray of the run (points, directions, optical path lengths "
                                                    "per surface reached)"),
    ("local_validation", LOCAL_VALIDATION_FILE, JSON_MEDIA_TYPE, "the Local Validation record of the run"),
    ("optical_scene", SCENE_FILE, JSON_MEDIA_TYPE, "the Optical Simulation View of the run (renderer-independent)"),
    ("optical_scene_glb", GLB_FILE, GLB_MEDIA_TYPE, "glTF 2.0 binary rendering of the optical scene"),
)
#: The ray data file's description in the manifest of a run that recorded the chief rays only.
RAY_DATA_SUMMARY_DESCRIPTION = ("the chief ray of every path (points, directions, optical path lengths per surface reached) "
                                "and each path's pupil sampling; pupil rays not recorded (ray data recording 'summary')")
RUN_NOTICE = (
    "Local pure-optical run: every file below comes from one os-optics execution on this device. The 3D file is "
    "the Optical Simulation View of the design (optical elements, Engine-traced beams, observation planes)."
)


@dataclass(frozen=True)
class RunArtifacts:
    """Where one run's files were written. ``paths`` maps each role (``project``, ``engine_request``,
    ``simulation_result``, ``ray_data``, ``local_validation``, ``optical_scene``, ``optical_scene_glb``, ``manifest``) to its
    file."""

    directory: Path
    manifest: dict[str, Any]

    @property
    def paths(self) -> dict[str, Path]:
        out = {entry["role"]: self.directory / entry["name"] for entry in self.manifest["files"]}
        out["manifest"] = self.directory / MANIFEST_FILE
        return out

    @property
    def manifest_path(self) -> Path:
        return self.directory / MANIFEST_FILE

    @property
    def local_validation_path(self) -> Path:
        return self.directory / LOCAL_VALIDATION_FILE

    @property
    def simulation_result_path(self) -> Path:
        return self.directory / SIMULATION_RESULT_FILE

    @property
    def ray_data_path(self) -> Path:
        return self.directory / RAY_DATA_FILE

    @property
    def optical_scene_path(self) -> Path:
        return self.directory / SCENE_FILE

    @property
    def optical_scene_glb_path(self) -> Path:
        return self.directory / GLB_FILE


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def cross_artifact_problems(record: LocalValidationRecordV1, scene: OpticalSimulationSceneV1, result: ValidationResult) -> tuple[str, ...]:
    """The scene, the record and the Engine result describe the same Engine run of the same Project (empty = yes)."""

    manifest = result.manifest
    problems = []
    if scene.project != record.project:
        problems.append("the optical scene and the local validation record name different Projects")
    if scene.engine_run.input_hash != record.input_hash:
        problems.append("the optical scene and the local validation record have different Engine input hashes")
    if scene.engine_run.result_hash != record.result_hash:
        problems.append("the optical scene and the local validation record have different Engine result hashes")
    if (scene.engine_run.engine_package, scene.engine_run.engine_version) != (record.runtime.engine_package, record.runtime.engine_version):
        problems.append("the optical scene and the local validation record were made by different Engine distributions or versions")
    if manifest is None:
        problems.append("the Engine result carries no manifest")
    else:
        if manifest.input_hash != record.input_hash.digest or manifest.result_hash != record.result_hash.digest:
            problems.append("the Engine result's manifest hashes differ from the local validation record's")
        if manifest.trace_hash != scene.engine_run.trace_hash:
            problems.append("the optical scene's trace hash differs from the Engine result's")
        if manifest.engine_build_id != record.runtime.engine_build_id:
            problems.append("the Engine result's build id differs from the local validation record's")
    # A matching result_hash string alone does not bind a compact record's claims to the result.
    # Compare only new fields actually present, preserving historical absent-field records.
    def same(left: Any, right: Any) -> bool:
        params = record.result_hash.params
        return hash_quantized(left, sig_digits=int(params["sig_digits"]), abs_floor=float(params["abs_floor"])) == \
            hash_quantized(right, sig_digits=int(params["sig_digits"]), abs_floor=float(params["abs_floor"]))

    from .record import scientific_projection
    projected = scientific_projection(result, record.simulation.model_scope_issues or ())
    if record.simulation.outcome is not None and not same(
        record.simulation.outcome.model_dump(mode="json"),
        None if projected.outcome is None else projected.outcome.model_dump(mode="json"),
    ):
        problems.append("the local validation's scientific outcome differs from the Engine result's")
    targets = {t.target_id: t for t in projected.target_outcomes}
    for target in record.simulation.target_outcomes:
        actual = targets.get(target.target_id)
        if actual is None or target.metric != actual.metric or target.status.value != actual.status.value or not same(target.value, actual.value):
            problems.append(f"the local validation's target {target.target_id} differs from the Engine result's")
        elif target.assessment is not None and target.assessment != actual.assessment:
            problems.append(f"the local validation's target {target.target_id} assessment differs from the Engine result's")
    if record.simulation.observation_quantities is not None:
        observations = {o.observation_id: o for o in projected.observation_quantities or ()}
        for quantity in record.simulation.observation_quantities:
            actual = observations.get(quantity.observation_id)
            expected = None if actual is None else actual.model_dump(mode="json")
            if not same(quantity.model_dump(mode="json"), expected):
                problems.append(f"the local validation's observation {quantity.observation_id} differs from the Engine result's")
    return tuple(problems)


def ray_data_problems(ray_data: Any, result: ValidationResult) -> tuple[str, ...]:
    """The ray data document belongs to the Engine result: schema and version, the same Engine input and trace hashes,
    ``TRACED`` exactly when the result has a trace, and the same beam paths (empty = yes)."""

    if not isinstance(ray_data, dict):
        return ("the ray data is not a JSON object",)
    if ray_data.get("schema") != RAY_DATA_SCHEMA or ray_data.get("schema_version") != RAY_PATHS_VERSION:
        return (f"the ray data is not an {RAY_DATA_SCHEMA} {RAY_PATHS_VERSION} document",)
    manifest = result.manifest
    problems = []
    if manifest is None:
        return ("the Engine result carries no manifest",)
    if ray_data.get("input_hash") != manifest.input_hash:
        problems.append("the ray data's Engine input hash differs from the Engine result's")
    if ray_data.get("trace_hash") != manifest.trace_hash:
        problems.append("the ray data's trace hash differs from the Engine result's")
    traced = ray_data.get("status") == TRACED
    if traced != (manifest.trace_hash is not None):
        problems.append("the ray data's status does not match whether the Engine result has a trace")
    paths = ray_data.get("paths")
    if not isinstance(paths, list) or not all(isinstance(p, dict) for p in paths):
        problems.append("the ray data's paths are not a list of objects")
    elif traced and sorted(str(p.get("path_id")) for p in paths) != sorted(p.path_id for p in result.paths):
        problems.append("the ray data's beam paths differ from the Engine result's")
    return tuple(problems)


def recorded_status(record: Any, model_scope_issues: Any) -> dict[str, Any]:
    """The ``status`` block of a run: its record's validation and simulation status, the model-scope findings and each
    target's diagnostic status -- what the run directory manifest states, and the results of a complete evaluation held in
    memory with it (``Evaluation.results()``)."""

    return {
        "validation_status": record.validation_status.value,
        "simulation_status": record.simulation.status.value,
        "model_scope_issues": [issue.as_dict() for issue in model_scope_issues],
        "targets": [{"target_id": t.target_id, "metric": t.metric, "status": t.status.value}
                    for t in record.simulation.target_outcomes],
    }


def _manifest(run: LocalOpticalRun, files: Mapping[str, tuple[int, str]]) -> dict[str, Any]:
    """The run directory manifest; ``files``: per file name its size and SHA-256."""
    record, scene, result = run.local_validation, run.optical_view, run.result
    level = getattr(run, "ray_data", RAY_DATA_FULL)
    manifest = result.manifest
    assert manifest is not None  # a completed local run always has one
    from .lowering import inputs_not_simulated

    not_simulated = inputs_not_simulated(run.project)
    return {
        "schema": RUN_MANIFEST_SCHEMA,
        "schema_version": RUN_MANIFEST_VERSION,
        "notice": RUN_NOTICE,
        "status": recorded_status(record, run.model_scope_issues),
        "project": record.project.model_dump(mode="json", by_alias=True, exclude_none=True),
        "catalog_version": run.catalog_version,
        "simulated_parts": [part.model_dump(mode="json", exclude_none=True) for part in run.simulated_parts],
        "element_models": [dict(model) for model in run.element_models],
        # (inputs of the Project the simulation did not read, lowering.inputs_not_simulated: present only when there are any)
        **({"inputs_not_simulated": [dict(row) for row in not_simulated]} if not_simulated else {}),
        "retained_dofs": [dof.model_dump(mode="json", exclude_none=True) for dof in record.retained_dofs],
        "engine_run": {
            "engine_package": record.runtime.engine_package,
            "engine_version": manifest.engine_version,
            "engine_build_id": manifest.engine_build_id,
            "input_hash": record.input_hash.model_dump(mode="json", exclude_none=True),
            "result_hash": record.result_hash.model_dump(mode="json", exclude_none=True),
            "trace_hash": manifest.trace_hash,
            "ray_paths_version": run.ray_paths.ray_paths_version,
        },
        "optical_view": {"view_kind": scene.view_kind, "scene_content_hash_osc1": scene.content_hash()},
        "ray_data": {"schema": RAY_DATA_SCHEMA, "schema_version": run.ray_paths.ray_paths_version, "status": run.ray_paths.status,
                     "trace_hash": run.ray_paths.trace_hash, "path_count": len(run.ray_paths.paths),
                     **({} if level == RAY_DATA_FULL else {"recording": level})},
        "binding": {
            "one_engine_execution": True,
            "record_binding_problems": list(run.binding_problems()),
            "cross_artifact_problems": list(cross_artifact_problems(record, scene, result)),
        },
        "files": [
            {"role": role, "name": name, "media_type": media,
             "description": RAY_DATA_SUMMARY_DESCRIPTION if role == RAY_DATA_ROLE and level != RAY_DATA_FULL else what,
             "bytes": files[name][0], "sha256": files[name][1]}
            for role, name, media, what in RUN_FILES
        ],
    }


def _view_files(run: LocalOpticalRun) -> dict[str, bytes]:
    """The scene JSON and GLB; a scene whose coordinates the GLB's 32-bit floats cannot hold (positions far beyond any
    optical table) is refused (``OPTICAL_VIEW_BINDING_FAILED``), never raised raw."""

    from .errors import LocalRunErrorCode, fail

    try:
        return optical_view_files(run.optical_view)
    except (ArithmeticError, ValueError, struct.error) as exc:
        raise fail(LocalRunErrorCode.OPTICAL_VIEW_BINDING_FAILED,
                   f"the 3D Optical Simulation View cannot represent this run: its coordinates are too large to draw "
                   f"({type(exc).__name__}); use positions of a realistic magnitude") from None


def _run_files(run: LocalOpticalRun) -> Iterator[tuple[str, Iterable[str] | bytes]]:
    """Every file of the run directory but the manifest, in ``RUN_FILES`` order: its name and its content -- bytes, or the
    pieces of its canonical JSON text (``manifest.streaming``), produced one file at a time."""

    yield PROJECT_FILE, canonical_json_bytes(run.project.payload())
    yield ENGINE_REQUEST_FILE, canonical_json_bytes(run.request.model_dump(mode="json"))
    yield SIMULATION_RESULT_FILE, json_parts(model_parts(run.result))
    yield RAY_DATA_FILE, json_parts(ray_paths_parts(run.ray_paths, getattr(run, "ray_data", RAY_DATA_FULL)))
    yield LOCAL_VALIDATION_FILE, canonical_json_bytes(run.local_validation.payload())
    yield from _view_files(run).items()


def fixed_run_file_bytes(project: Any, request: Any) -> int:
    """The bytes of a run's ``project.json`` and ``engine_request.json`` (``_run_files``): known exactly once the Project
    (``ProjectDocumentV2``) is lowered to its Engine request, before the trace (``local/resources.py``)."""

    return len(canonical_json_bytes(project.payload())) + len(canonical_json_bytes(request.model_dump(mode="json")))


def _encoded(content: Iterable[str] | bytes) -> Iterator[bytes]:
    if isinstance(content, bytes):
        yield content
    else:
        for piece in content:
            yield piece.encode("utf-8")


def _kept(pieces: Iterable[bytes]) -> Iterator[bytes]:
    """``pieces``, each counted against the run's artifact limit before it is kept (``run_control.artifact_bytes``)."""

    for piece in pieces:
        artifact_bytes(len(piece))
        yield piece


def run_artifact_files(run: LocalOpticalRun) -> dict[str, bytes]:
    """Every file of the run directory, in memory (``manifest.json`` last). A monitored run (``os_optics.run_control``)
    counts each piece against its artifact limit as it is produced, and may be stopped here."""

    with phase("write", unit="bytes"):
        files = {name: b"".join(_kept(_encoded(content))) for name, content in _run_files(run)}
        manifest = canonical_json_bytes(_manifest(run, {name: (len(data), _sha256(data)) for name, data in files.items()}))
        artifact_bytes(len(manifest))
        files[MANIFEST_FILE] = manifest
    return files


class RunDirectoryLinkError(OSError):
    """A run would write through a link -- a symbolic link, a Windows directory junction or another reparse point -- in its
    run directory. A run never writes through a link and never replaces one: it refuses to write (``project run`` exits
    64), and the link and what it points to stay as they were."""


class RunDirectoryOccupiedError(OSError):
    """A run would replace a file it did not write: a plain file at a run-file name that no Engine manifest there lists
    (nor a staged manifest of an interrupted commit, nor the record of what a completed commit could not remove), a ``configurations/<id>`` holding a run that the directory's manifest
    does not name, or a ``project.json`` holding another design that no manifest lists with its bytes. A run never replaces
    it: it refuses to write (``project run`` exits 64) and every file stays as it was."""


def _refuse_link(path: Path) -> Path:
    if _is_link(path):
        raise RunDirectoryLinkError(f"{path} is a link (a symbolic link or a junction); a run never writes through a link or "
                                    "replaces one: remove the link, or write the run into another directory")
    return path


def run_subdirectory(directory: Path | str, relative: str, *, make: bool = True) -> Path:
    """``directory / relative`` (such as ``configurations/<id>``), made if missing (unless not ``make``), after checking
    that no part of it below ``directory`` is a link (``RunDirectoryLinkError``): a run never writes through one."""

    path = Path(directory)
    for part in Path(relative).parts:
        path = _refuse_link(path / part)
    if make:
        path.mkdir(parents=True, exist_ok=True)
    return path


def record_stopped_run(out_dir: Path | str, stopped: Any) -> None:
    """Write the ``StoppedRun`` record of a run that stopped before it completed into ``out_dir`` (``run_stopped.json``),
    beside whatever is there: an earlier run in the directory stays as it was. A directory that cannot be written, or a
    directory or a ``run_stopped.json`` that is a link, keeps no record."""

    try:
        directory = Path(out_dir)
        if _is_link(directory):
            return  # (a run directory that is a link is never written through)
        directory.mkdir(parents=True, exist_ok=True)
        staged = StagedFiles(directory, check=_refuse_link)
        try:
            staged.stage(STOPPED_FILE, canonical_json_bytes(stopped.model_dump(mode="json")))
            staged.commit()
        finally:
            staged.discard()
    except (OSError, ValueError):
        pass


#: Windows' ``FILE_ATTRIBUTE_REPARSE_POINT`` (``stat.FILE_ATTRIBUTE_REPARSE_POINT``): set on a directory junction and on
#: every other reparse point.
REPARSE_POINT_ATTRIBUTE = 0x400
#: Hexadecimal digits of a SHA-256 digest, as a manifest records it.
SHA256_HEX_DIGITS = 64
#: ``StoppedRun.kind``: what makes a ``run_stopped.json`` a stop record.
STOPPED_RUN_KIND = "STOPPED_RUN"
#: Bytes read at a time when a file is hashed.
HASH_CHUNK_BYTES = 1 << 20


def _is_link(path: Path) -> bool:
    """``path`` is a symbolic link, a Windows directory junction or another reparse point (``Path.is_symlink`` is False for
    a junction, and ``Path.is_junction`` needs Python 3.12): a run never writes through one and never removes through
    one."""

    try:
        info = path.lstat()
    except (OSError, ValueError):
        return False
    return path.is_symlink() or bool(getattr(info, "st_file_attributes", 0) & REPARSE_POINT_ATTRIBUTE)


def _is_plain_file(path: Path) -> bool:
    return not _is_link(path) and path.is_file()


def _is_plain_directory(path: Path) -> bool:
    return not _is_link(path) and path.is_dir()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_bytes())
    except (OSError, ValueError, RecursionError):
        return None


def _listed_files(manifest: Any) -> dict[str, str] | None:
    """What an Engine manifest (of a run, or of a run of a Project with configurations) lists: per Engine file name the
    SHA-256 it recorded, from its well-formed entries only (a ``name`` that is one of the Engine's file names, a
    ``sha256`` of 64 characters); an entry of any other shape, or a file list that is no list, lists nothing. ``None``:
    ``manifest`` is no Engine manifest."""

    from .configurations import CONFIGURATIONS_RUN_SCHEMA

    if not isinstance(manifest, dict) or manifest.get("schema") not in (RUN_MANIFEST_SCHEMA, CONFIGURATIONS_RUN_SCHEMA):
        return None
    names = {name for _, name, _, _ in RUN_FILES} | {COMPARISON_FILE}
    entries = manifest.get("files")
    listed: dict[str, str] = {}
    for entry in entries if isinstance(entries, list) else ():
        if not isinstance(entry, dict):
            continue
        name, sha256 = entry.get("name"), entry.get("sha256")
        if isinstance(name, str) and name in names and isinstance(sha256, str) and len(sha256) == SHA256_HEX_DIGITS:
            listed[name] = sha256
    return listed


def _engine_manifest(directory: Path) -> Any:
    """The Engine manifest in ``directory`` (of a run, or of a run of configurations) as read; ``None``: there is none (no
    file, a link, no JSON, or a ``manifest.json`` that is no Engine manifest)."""

    path = directory / MANIFEST_FILE
    manifest = _read_json(path) if _is_plain_file(path) else None
    return manifest if _listed_files(manifest) is not None else None


def _configuration_directories(directory: Path, manifest: Any) -> list[str]:
    """The ``configurations/<id>`` directories an Engine manifest of a run of configurations names, each a plain directory
    under a plain ``configurations/`` (a link is never entered)."""

    from .configurations import is_configurations_manifest

    if not is_configurations_manifest(manifest) or not _is_plain_directory(directory / CONFIGURATIONS_DIRECTORY):
        return []
    found: list[str] = []
    entries = manifest.get("configurations")
    for entry in entries if isinstance(entries, list) else ():
        name = entry.get("directory") if isinstance(entry, dict) else None
        parts = name.split("/") if isinstance(name, str) else []
        if len(parts) != 2 or parts[0] != CONFIGURATIONS_DIRECTORY or parts[1] in ("", ".", "..") or "\\" in parts[1]:
            continue
        if name not in found and _is_plain_directory(directory / CONFIGURATIONS_DIRECTORY / parts[1]):
            found.append(name)
    return found


def _interrupted_manifest(directory: Path) -> Any:
    """The copy of the earlier manifest that a commit killed after removing it left in ``directory``
    (``EARLIER_MANIFEST_FILE``), as read; ``None``: there is none, or it is no Engine manifest."""

    path = directory / EARLIER_MANIFEST_FILE
    manifest = _read_json(path) if _is_plain_file(path) else None
    return manifest if _listed_files(manifest) is not None else None


def _earlier_manifest(directory: Path) -> Any:
    """The Engine manifest that describes the earlier run in ``directory``: the manifest there, or -- where there is no
    ``manifest.json`` at all -- the copy a commit killed after removing it left (``EARLIER_MANIFEST_FILE``), which lists
    what that run held there, by name. ``None``: neither."""

    manifest = _engine_manifest(directory)
    if manifest is None and not _is_link(directory / MANIFEST_FILE) and not (directory / MANIFEST_FILE).exists():
        manifest = _interrupted_manifest(directory)
    return manifest


def _leftover_names(directory: Path) -> list[str]:
    """The names the leftover record in ``directory`` lists (``LEFTOVER_FILE``, ``_record_leftover``), relative to it: a
    run-file name of the directory (not its manifest or ``project.json``), or of a ``configurations/<id>`` (not its
    manifest), or such a ``configurations/<id>`` itself. A name of any other shape, or a record that is no leftover
    record, lists nothing; a link is never read."""

    path = directory / LEFTOVER_FILE
    record = _read_json(path) if _is_plain_file(path) else None
    names = record.get("names") if isinstance(record, dict) and record.get("schema") == LEFTOVER_SCHEMA else None
    run_files = {name for _, name, _, _ in RUN_FILES} - {MANIFEST_FILE}
    found: list[str] = []
    for name in names if isinstance(names, list) else ():
        parts = name.split("/") if isinstance(name, str) and "\\" not in name else []
        top = len(parts) == 1 and parts[0] in (run_files | {COMPARISON_FILE}) - {PROJECT_FILE}
        nested = (len(parts) in (2, 3) and parts[0] == CONFIGURATIONS_DIRECTORY and parts[1] not in ("", ".", "..")
                  and (len(parts) == 2 or parts[2] in run_files))
        if top or nested:
            found.append(name)
    return list(dict.fromkeys(found))


def _record_leftover(directory: Path, names: Iterable[str]) -> None:
    """Before a commit moves the directory's manifest into place (``StagedFiles.commit`` ``before_last``): of ``names``
    (relative; the earlier run's that it removes by name), those still there as plain files -- a file another program held
    open while the commit removed it -- are recorded by name in
    ``LEFTOVER_FILE``, with the ``configurations/<id>`` directories that hold them, so the next run, the same one or
    another, accepts them (``_refuse_unlisted``) and removes them by name (``_earlier_run_names``). With none left over the
    record is removed. A name the commit did remove is not listed: a file placed there later is the person's own. Written
    before the manifest, so no kill between the two leaves a held file listed by no record (a kill while it is written
    leaves no record or the whole one, through its temporary file); a commit that fails or is killed at the manifest's move
    keeps it: it lists only names still there as plain files, which every later run accepts and removes by name."""

    left = [name for name in names if _is_plain_file(directory / name)]
    path = directory / LEFTOVER_FILE
    if not left:
        if _is_link(path) or path.exists():
            _remove_quietly([path])
        return
    holders = dict.fromkeys(Path(name).parent.as_posix() for name in left if "/" in name)
    temporary = path.with_name(f"{LEFTOVER_FILE}.partial")
    try:
        temporary.unlink(missing_ok=True)
        with temporary.open("xb") as handle:
            handle.write(canonical_json_bytes({"schema": LEFTOVER_SCHEMA, "names": [*left, *holders]}))
        temporary.replace(path)  # (a link at the record's name is replaced, never written through)
    except OSError:
        _remove_quietly([temporary])


def _sub_run_names(relative: str) -> list[str]:
    """The names of a configuration's run directory ``relative`` a run removes by name: its run files, then the directory
    (removed when empty)."""

    return [*(f"{relative}/{name}" for _, name, _, _ in RUN_FILES), relative]


def _earlier_run_names(directory: Path) -> tuple[list[str], list[str]]:
    """What the run that the Engine manifest in ``directory`` describes (``_earlier_manifest``: the manifest there, or the
    copy a killed commit left) holds there, with what the staged manifests of interrupted commits list
    (``_staged_manifests``: the files those commits had moved into place), by name (only those manifests are read; no file is hashed): ``(manifests, others)`` -- the
    manifests and stop records (the directory's, and those of each configuration they name), and every other file and
    directory of those runs, a directory after the files in it. With no Engine manifest there: the directory's own two
    names, and nothing else."""

    manifests, others = [MANIFEST_FILE, STOPPED_FILE], []
    configurations: list[str] = []
    for manifest in (_earlier_manifest(directory), *_staged_manifests(directory)):
        if manifest is None:
            continue
        others.extend(_listed_files(manifest) or ())
        configurations += _configuration_directories(directory, manifest)
    for relative in dict.fromkeys(configurations):
        manifests += [f"{relative}/{MANIFEST_FILE}", f"{relative}/{STOPPED_FILE}"]
        others += _sub_run_names(relative)
    # (what a completed commit could not remove: by name, its files before the directories that hold them)
    leftover = _leftover_names(directory)
    others += leftover
    if configurations or any(name.startswith(f"{CONFIGURATIONS_DIRECTORY}/") for name in leftover):
        others.append(CONFIGURATIONS_DIRECTORY)
    return manifests, list(dict.fromkeys(others))


def _same_project(path: Path, project: bytes) -> bool:
    """The file ``path`` holds the same Project as ``project`` (the canonical ``project.json`` a run writes): the same
    ``ProjectDocumentV2`` payload, whatever the file's formatting -- the Project being run, which the run writes again in
    its canonical form. Anything else (another design, a file that is no Project) is not."""

    try:
        if not _is_plain_file(path):
            return False
        return parse_project_document(json.loads(path.read_bytes())).payload() == parse_project_document(json.loads(project)).payload()
    except Exception:  # (unreadable, no JSON, no valid Project: not the Project being run)
        return False


def _refuse_another_design(directory: Path, project: bytes) -> None:
    """``project.json`` is also the name of a design file, so it keeps one protection: a run replaces it when it is
    missing, holds the Project being run (``_same_project``) or is listed with its present SHA-256 by the Engine manifest
    there (or the copy of it a killed commit left: ``_earlier_manifest``; an earlier run's) or by a staged manifest of an
    interrupted commit (``_staged_manifests``: the ``project.json`` that commit moved into place). A ``project.json`` holding another design that no
    manifest lists is the person's own: refused (``RunDirectoryOccupiedError``) before anything is written."""

    path = directory / PROJECT_FILE
    if _is_link(path) or not path.exists():
        return  # (a link there is refused by the write itself: RunDirectoryLinkError)
    listed = {(_listed_files(manifest) or {}).get(PROJECT_FILE)
              for manifest in (_earlier_manifest(directory), *_staged_manifests(directory)) if manifest is not None} - {None}
    try:
        if listed and _is_plain_file(path) and _file_sha256(path) in listed:
            return
    except OSError:
        pass
    if _same_project(path, project):
        return
    raise RunDirectoryOccupiedError(
        f"{path} is no file of an earlier run there (no Engine manifest lists it with these bytes) (it holds another Project "
        "than the one run); a run never replaces it: move or delete it, or write the run into another directory")


def _staged_manifest(directory: Path) -> Any:
    """The staged manifest an interrupted commit left in ``directory`` (``.manifest.json.partial``: written before the commit
    began, moved last), as read; ``None``: there is none, or it is no Engine manifest. It lists every file of that commit."""

    path = staged_path(directory / MANIFEST_FILE)
    manifest = _read_json(path) if _is_plain_file(path) else None
    return manifest if _listed_files(manifest) is not None else None


def _interrupted_records(directory: Path) -> list[Path]:
    """The staged manifests of earlier interrupted commits kept in ``directory`` under their own names
    (``INTERRUPTED_MANIFEST_PREFIX`` and a number), in the order they were kept; links are never read."""

    if not _is_plain_directory(directory):
        return []
    found = [(int(path.name[len(INTERRUPTED_MANIFEST_PREFIX):]), path) for path in directory.iterdir()
             if path.name.startswith(INTERRUPTED_MANIFEST_PREFIX) and path.name[len(INTERRUPTED_MANIFEST_PREFIX):].isdecimal()
             and _is_plain_file(path)]
    return [path for _, path in sorted(found)]


def _staged_manifests(directory: Path) -> list[Any]:
    """Every staged manifest interrupted commits left in ``directory``, as read: the one at its staged name
    (``_staged_manifest``) and those kept under their own names (``_interrupted_records``); each Engine manifest only. Each
    lists the files its commit had moved there or was moving."""

    manifests = [_read_json(path) for path in _interrupted_records(directory)]
    manifests.append(_staged_manifest(directory))
    return [manifest for manifest in manifests if _listed_files(manifest) is not None]


def _keep_interrupted(directory: Path, relatives: Iterable[str]) -> None:
    """Before a run stages a manifest in each directory ``relatives`` (``.``: the run directory), the staged manifest an
    interrupted commit left there -- an Engine manifest at ``.manifest.json.partial``, a file of one name -- is kept under
    its own name (``INTERRUPTED_MANIFEST_PREFIX`` and the next number), so staging this run's manifest at that name does
    not lose the record of the files that commit moved there. A name that is a link is refused
    (``RunDirectoryLinkError``)."""

    for relative in relatives:
        where = directory if relative == "." else directory / relative
        staged = staged_path(where / MANIFEST_FILE)
        if not _is_plain_directory(where) or _staged_manifest(where) is None or staged.lstat().st_nlink != 1:
            continue  # (a staged manifest with another name elsewhere -- a hard link -- is no commit's: staging deletes it)
        kept = _interrupted_records(where)
        number = 1 + (int(kept[-1].name[len(INTERRUPTED_MANIFEST_PREFIX):]) if kept else 0)
        staged.replace(_refuse_link(where / f"{INTERRUPTED_MANIFEST_PREFIX}{number}"))


def _engine_names(directory: Path) -> tuple[set[str], set[str]]:
    """What the Engine's records in ``directory`` name, by name only: the run-file names the manifest there (or the copy
    of it a killed commit left: ``_earlier_manifest``) and the staged manifests of interrupted commits
    (``_staged_manifests``) list, and the ``configurations/<id>`` directories they name."""

    files: set[str] = set()
    configurations: set[str] = set()
    for manifest in (_earlier_manifest(directory), *_staged_manifests(directory)):
        if manifest is None:
            continue
        files |= set(_listed_files(manifest) or ())
        configurations |= set(_configuration_directories(directory, manifest))
    return files, configurations


#: Every name of a run directory a run writes or removes as the Engine's, besides ``comparison.json``.
_RUN_FILE_NAMES = (*(name for _, name, _, _ in RUN_FILES), MANIFEST_FILE)


def _refuse_unlisted(directory: Path, names: Iterable[str], without_run: Iterable[str], projects: Mapping[str, bytes]) -> None:
    """A run replaces only what the Engine wrote (keep item of the run directory), decided by name, no file hashed: before
    anything is written,

    * every ``configurations/<id>`` the run writes or removes that holds an Engine manifest -- a run -- must be named by the
      directory's manifest or by a staged manifest of an interrupted commit there (a run the person keeps there stays);
    * every plain file at a name the run writes -- or removes: every run-file name of a configuration that did not run
      (``without_run``) -- must be listed by name by the Engine manifest in its directory, by a staged manifest an
      interrupted commit left there or by the record of what a completed commit could not remove (``_leftover_names``); a ``manifest.json`` must be an Engine manifest; a configuration's ``project.json`` may
      also hold the Project being run there (``projects``: the bytes the run writes at that name).

    ``project.json`` of the directory keeps its own rule (``_refuse_another_design``); the stop record and the temporary
    names are the Engine's. ``RunDirectoryOccupiedError`` otherwise."""

    groups: dict[str, list[str]] = {}
    for name in names:
        groups.setdefault(Path(name).parent.as_posix(), []).append(Path(name).name)
    for relative in without_run:
        # (a configuration that did not run: the commit removes every run-file name there, so each is checked as written)
        group = groups.setdefault(relative, [])
        group.extend(name for name in _RUN_FILE_NAMES if name not in group)
    _, configurations = _engine_names(directory)
    leftover = _leftover_names(directory)
    for parent in sorted(groups, key=lambda group: group != "."):
        where = directory if parent == "." else directory / parent
        if parent != ".":
            if not _is_plain_directory(where):
                continue
            if _engine_manifest(where) is not None and parent not in configurations:
                raise RunDirectoryOccupiedError(
                    f"{where} holds a run that is no part of the earlier run there (the directory's manifest does not name it); a run "
                    "never replaces it: move or delete it, or write the run into another directory")
        listed, _ = _engine_names(where)
        listed |= {Path(name).name for name in leftover if Path(name).parent.as_posix() == parent}
        for name in groups[parent]:
            path = where / name
            if (parent == "." and name == PROJECT_FILE) or not _is_plain_file(path):
                continue
            if name == MANIFEST_FILE and _engine_manifest(where) is not None:
                continue
            if name != MANIFEST_FILE and name in listed:
                continue
            relative = name if parent == "." else f"{parent}/{name}"
            if name == PROJECT_FILE and relative in projects and _same_project(path, projects[relative]):
                continue
            raise RunDirectoryOccupiedError(
                f"{path} is no file of an earlier run there (no Engine manifest there lists it); a run never replaces it: move "
                "or delete it, or write the run into another directory")


@dataclass
class _RunWrite:
    """One write of a run into ``directory``: its staged files and what the earlier run there holds (read before anything
    is written)."""

    directory: Path
    staged: StagedFiles
    earlier: tuple[list[str], list[str]]
    #: Configuration directories of this run that hold no run (a configuration that failed): what is there goes.
    without_run: tuple[str, ...] = ()


def _open_run_write(out_dir: Path | str, names: Iterable[str], project: bytes | None, without_run: Iterable[str] = (),
                    projects: Mapping[str, bytes] | None = None) -> _RunWrite:
    """Before a run is written into ``out_dir``: the directory is made; the directory, every name the run writes
    (``names``, relative; ``configurations/<id>/<file>`` for a configuration's run) with the directories on its way and its
    temporary name must be no link (``RunDirectoryLinkError``); every file at those names must be the Engine's
    (``_refuse_unlisted``) and the ``project.json`` there must not be another design of the person's own (``project``: the
    bytes the run writes there; ``projects``: those of a configuration's ``project.json``; ``RunDirectoryOccupiedError``).
    Nothing in the directory has changed when this raises. When it returns, the staged manifest an interrupted commit left
    where the run stages a manifest is kept under its own name (``_keep_interrupted``); nothing else has changed."""

    directory = _refuse_link(Path(out_dir))
    directory.mkdir(parents=True, exist_ok=True)
    names, without_run = tuple(names), tuple(without_run)
    for relative in without_run:
        run_subdirectory(directory, relative, make=False)
    for name in names:
        parent = Path(name).parent.as_posix()
        if parent != ".":
            run_subdirectory(directory, parent, make=False)
        _refuse_link(directory / name)
        _refuse_link(staged_path(directory / name))
    if project is not None:
        _refuse_another_design(directory, project)
    _refuse_unlisted(directory, names, without_run, projects or {})
    _keep_interrupted(directory, dict.fromkeys(Path(name).parent.as_posix() for name in names if Path(name).name == MANIFEST_FILE))
    return _RunWrite(directory, StagedFiles(directory, check=_refuse_link), _earlier_run_names(directory), without_run)


# An attempt to run (``run_control.begin_attempt`` / ``end_attempt``) records the directories whose commit it
# completed: its stop handlers ask ``committed_run`` whether the run they stop was already moved into place, so that a
# Ctrl-C or a stop that lands in the clean-up after the manifest's move answers with the completed run and writes no
# stop record beside it.


def _directory_key(directory: Path | str) -> str:
    try:
        return str(Path(directory).resolve())
    except (OSError, ValueError, RuntimeError):
        return str(Path(directory).absolute())


def committed_run(directory: Path | str | None) -> Any:
    """What the write into ``directory`` returns (``RunArtifacts`` / ``ConfigurationsRunArtifacts``) when the current
    attempt's commit there completed -- its manifest is in place --; ``None`` otherwise, and outside an attempt."""

    commits = attempt_commits()
    if commits is None or directory is None:
        return None
    return commits.get(_directory_key(directory))


def _commit_run(write: _RunWrite, artifacts: Any = None) -> None:
    """Every file of the run is staged: replace the run in the directory. ``artifacts``: what the write returns, recorded
    for the current attempt (``committed_run``) the moment the directory's manifest is in place.

    0. Beside each Engine manifest the commit removes, its copy (``EARLIER_MANIFEST_FILE``) is written
       (``_keep_earlier_manifests``); the copies are removed when the commit ends. A commit killed after a manifest is gone
       leaves its copy: the record, by name, of what the earlier run still holds there, which the next run reads in place
       of the manifest (``_earlier_manifest``) -- to accept the earlier run's ``project.json`` with its bytes, to accept
       the earlier run's files by name, and to remove by name what it held and the next run does not write. A commit that
       fails (a name it cannot delete or move into place: a file held open elsewhere) leaves what a kill at that step
       leaves: the copy of each manifest it removed -- the copy of one still in place is removed -- and, once the
       manifests are gone, its staged manifests (``StagedFiles.commit`` ``keep``), which list the files it moved there;
       any run, the same one or the earlier design, then completes there. The next run keeps each such staged manifest
       under its own name before it stages its own (``_keep_interrupted``, in ``_open_run_write``); they are removed when
       a commit completes (in a configuration's run directory it removes, before the directory).
    1. The run files of each configuration that did not run (``without_run``) are removed, then the stop records and
       manifests there (the earlier run's, and any at a name this run writes), the
       directory's own manifest last of them: a commit that is interrupted after this leaves no manifest -- no run -- rather
       than a manifest beside files of another run, and a name that cannot be deleted stops the commit while the
       directory's manifest still describes what is there.
    2. ``project.json`` is moved into place, first of the files.
    3. The other files are moved; what the earlier run held and this one does not write is removed by name (a
       configuration's run directory, after its copy of the manifest; ``comparison.json``; the files of a single run).
    4. What step 3 could not remove (a file held open elsewhere) stays and is recorded by name (``_record_leftover``),
       then the directory's manifest is moved last, then the copies and the kept staged manifests are removed."""

    written = set(write.staged.names)
    subs = list(dict.fromkeys([*(Path(name).parent.as_posix() for name in write.staged.names if "/" in name), *write.without_run]))
    manifests, others = write.earlier
    named = list(dict.fromkeys([*manifests, *(f"{sub}/{name}" for sub in subs for name in (MANIFEST_FILE, STOPPED_FILE))]))
    # (a configuration that did not run stages no manifest of its own: its run files go before any manifest, so a kill
    # never leaves them in a directory whose manifest is gone, where no manifest would list them as the Engine's)
    first = [*(name for sub in write.without_run for name in _sub_run_names(sub)[:-1]),
             *(name for name in named if Path(name).name == STOPPED_FILE),
             *(name for name in named if Path(name).name == MANIFEST_FILE and name != MANIFEST_FILE), MANIFEST_FILE]
    held = list(dict.fromkeys([*others, *(name for sub in write.without_run for name in _sub_run_names(sub))]))
    stale = [name for name in held if name not in written and not any(other.startswith(name + "/") for other in written)]
    directories = list(dict.fromkeys(Path(name).parent.as_posix() for name in first if Path(name).name == MANIFEST_FILE))
    copies, kept = _keep_earlier_manifests(write.directory, directories)
    # (the staged manifests interrupted commits left where this run writes: removed once it completes; in a configuration's
    # run directory it removes, they go with the directory's copy, before the directory)
    records = [record for relative in dict.fromkeys([".", *subs])
               for record in _interrupted_records(write.directory if relative == "." else write.directory / relative)]
    remove = [entry for name in stale
              for entry in ((*((f"{name}/{EARLIER_MANIFEST_FILE}",) if name in directories else ()),
                             *(f"{name}/{record.name}" for record in _interrupted_records(write.directory / name)), name)
                            if Path(name).parent.as_posix() == CONFIGURATIONS_DIRECTORY else (name,))]
    try:
        commits = attempt_commits()

        def completed() -> None:  # (the directory's manifest is in place: this attempt's run is there)
            if commits is not None:
                commits[_directory_key(write.directory)] = artifacts

        write.staged.commit(remove_first=first, first=PROJECT_FILE, remove=remove, last=MANIFEST_FILE,
                            keep=[name for name in written if Path(name).name == MANIFEST_FILE],
                            before_last=lambda: _record_leftover(write.directory, stale), after_last=completed)
    except Exception:
        # (a commit that fails leaves what a kill at the same step leaves: the copy of each manifest it removed and, once
        # the manifests are gone, its staged manifests; the copy of a manifest still in place is removed)
        # (a link at the manifest's name counts as a manifest in place)
        _remove_quietly([copy for copy in copies
                         if _is_link(copy.parent / MANIFEST_FILE) or (copy.parent / MANIFEST_FILE).exists()])
        raise
    _remove_quietly([*copies, *kept, *records])


def _keep_earlier_manifests(directory: Path, relatives: Iterable[str]) -> tuple[list[Path], list[Path]]:
    """Before a commit removes anything: in each directory whose manifest it removes (``relatives``; ``.``: the run
    directory) the copy of that manifest (``EARLIER_MANIFEST_FILE``), written when the manifest there is an Engine
    manifest, or -- where there is no manifest -- the copy a killed commit left there, kept as it is. Returns ``(written,
    kept)``. A copy name that is a link is refused (``RunDirectoryLinkError``) before anything is removed."""

    written: list[Path] = []
    kept: list[Path] = []
    for relative in relatives:
        where = directory if relative == "." else directory / relative
        if not _is_plain_directory(where):
            continue
        copy = _refuse_link(where / EARLIER_MANIFEST_FILE)
        if _engine_manifest(where) is not None:
            data = (where / MANIFEST_FILE).read_bytes()
            if copy.exists():
                copy.unlink()  # (a copy an earlier killed commit left)
            with copy.open("xb") as handle:
                handle.write(data)
            written.append(copy)
        elif _earlier_manifest(where) is not None:
            kept.append(copy)
    return written, kept


def _remove_quietly(paths: Iterable[Path]) -> None:
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass  # (one that cannot be deleted stays; it is read only where there is no manifest)


def _stage_run(staged: StagedFiles, run: LocalOpticalRun, prefix: str = "") -> dict[str, Any]:
    """Stage every file of ``run`` (under ``prefix``: ``configurations/<id>/``), each written piece by piece and hashed as
    it is written and counted against the run's artifact limit, and return its manifest (staged last)."""

    written = {name: staged.stage(prefix + name, _kept(_encoded(content))) for name, content in _run_files(run)}
    manifest = _manifest(run, written)
    data = canonical_json_bytes(manifest)
    artifact_bytes(len(data))
    staged.stage(prefix + MANIFEST_FILE, data)
    return manifest


def write_run_artifacts(run: LocalOpticalRun, out_dir: Path | str) -> RunArtifacts:
    """Write the run directory (see the module docstring) and return where everything is: every file is written to its
    temporary name first, one at a time, piece by piece and hashed as it is written (the same bytes as
    ``run_artifact_files``), and the run is moved into place once all are written. A write that stops or fails before that
    removes its temporary files and leaves the directory as it found it. A run file name, or its temporary name, that is
    a link is refused (``RunDirectoryLinkError``) and nothing is written through it."""

    write = _open_run_write(out_dir, (*(name for _, name, _, _ in RUN_FILES), MANIFEST_FILE),
                            canonical_json_bytes(run.project.payload()))
    with phase("write", unit="bytes"):
        try:
            manifest = _stage_run(write.staged, run)
            _commit_run(write, RunArtifacts(directory=write.directory, manifest=manifest))
        finally:
            write.staged.discard()
    return RunArtifacts(directory=write.directory, manifest=manifest)


def _staging_order(names: Iterable[str]) -> list[str]:
    """``names`` (relative; ``configurations/<id>/<file>`` for a configuration's run) in the order a run is staged: each
    configuration's run directory, then the directory's own files; in each, the manifest last (``_commit_run`` moves the
    directory's ``project.json`` first and its manifest last)."""

    groups: dict[str, list[str]] = {}
    for name in names:
        groups.setdefault(Path(name).parent.as_posix(), []).append(name)
    ordered: list[str] = []
    for parent in sorted(groups, key=lambda group: group == "."):
        ordered += sorted(groups[parent], key=lambda name: Path(name).name == MANIFEST_FILE)
    return ordered


def write_run_files(files: Mapping[str, bytes], out_dir: Path | str, *, without_run: Iterable[str] = ()) -> RunArtifacts:
    """Write a run directory from its files held in memory (``run_artifact_files``, or ``configurations_run_files`` with
    ``configurations/<id>/<file>`` names) by the rule of ``write_run_artifacts``, and return where everything is.
    ``without_run``: the configuration directories of the run that hold no run."""

    write = _open_run_write(out_dir, files, files.get(PROJECT_FILE), without_run,
                            {name: data for name, data in files.items() if Path(name).name == PROJECT_FILE})
    try:
        for name in _staging_order(files):
            write.staged.stage(name, files[name])
        _commit_run(write, RunArtifacts(directory=write.directory, manifest=json.loads(files[MANIFEST_FILE])))
    finally:
        write.staged.discard()
    return RunArtifacts(directory=write.directory, manifest=json.loads(files[MANIFEST_FILE]))


#: Bytes read at a time when a written file is compared with the bytes it was written from.
READ_BACK_CHUNK_BYTES = 1 << 20


def written_run_problems(files: Mapping[str, bytes], out_dir: Path | str) -> tuple[str, ...]:
    """Check that a run directory written by ``write_run_files`` holds exactly ``files`` (size and SHA-256 of each file,
    read back in chunks, so no file is held twice or parsed again). Returns the problems found (empty = consistent)."""

    directory = Path(out_dir)
    problems: list[str] = []
    for name, data in files.items():
        digest = hashlib.sha256()
        size = 0
        try:
            with (directory / name).open("rb") as handle:
                while chunk := handle.read(READ_BACK_CHUNK_BYTES):
                    digest.update(chunk)
                    size += len(chunk)
        except OSError:
            problems.append(f"{name}: missing")
            continue
        if size != len(data) or digest.hexdigest() != _sha256(data):
            problems.append(f"{name}: size or SHA-256 differs from the run's {name}")
    return tuple(problems)


def _load_json(read: Callable[[str], bytes], name: str, problems: list[str]) -> Any:
    try:
        return json.loads(read(name))
    except FileNotFoundError:
        problems.append(f"{name}: missing")
    except OSError as exc:
        problems.append(f"{name}: cannot be read ({exc.strerror or type(exc).__name__})")
    except ValueError:
        problems.append(f"{name}: not valid JSON")
    except RecursionError:
        problems.append(f"{name}: nested too deeply to be a run file")
    return None


def _object(value: Any) -> dict[str, Any]:
    """A manifest member that must be a JSON object; anything else reads as an empty one (and fails the checks)."""

    return value if isinstance(value, dict) else {}


def _parse(model: Any, document: Any, name: str, problems: list[str]) -> Any:
    if document is None:
        return None
    try:
        return parse_project_document(document) if model is ProjectDocumentV2 else model.model_validate(document)
    except ValidationError as exc:
        problems.append(f"{name}: not a valid {model.__name__} ({exc.error_count()} errors)")
        return None


#: The run files a commercial submission needs (``require`` of ``verify_run_artifacts`` / ``verify_run_files``): the
#: manifest, the Project, the Engine request and result and the Local Validation record, whose hashes and bindings vouch for
#: what it sends. The display files (optical scene, GLB, ray data) may be absent; a present one is still checked.
SUBMISSION_FILES = frozenset({MANIFEST_FILE, PROJECT_FILE, ENGINE_REQUEST_FILE, SIMULATION_RESULT_FILE, LOCAL_VALIDATION_FILE})


def verify_run_artifacts(out_dir: Path | str, *, require: frozenset[str] | None = None) -> tuple[str, ...]:
    """Check a run directory written by ``write_run_artifacts``. Returns the problems found (empty = consistent).

    Checks: every listed file exists with the manifest's size and SHA-256; each document parses as its contract;
    the Project hashes to the record's ``project_hash`` and the record passes ``check_local_validation_binding``;
    the Engine request hashes to the record's ``input_hash``; the Engine result is unaltered (its quantized hash equals
    its manifest's ``result_hash``) and its manifest matches the record; the optical scene names the same Project, input
    hash, result hash and trace hash; the GLB is structurally valid and carries the scene's content hash; the ray data
    names the Engine result's input and trace hashes and its beam paths (``ray_data_problems``). The run directory of a
    Project with configurations is checked as one (``os_optics.local.configurations.verify_configurations_run``).

    The run that is there is checked: a stop record (``run_stopped.json``) beside the manifest is the record of a later
    attempt that did not complete and left the run as it was (``later_attempt``); a directory holding a stop record and
    no manifest holds no run. Files that are links are read like the files they point to.

    ``require``: the run files that must be there (``SUBMISSION_FILES`` for a submission); a run file outside it may be
    absent, and when present it is checked as above. ``None`` (the default): every run file.
    """

    directory = Path(out_dir)
    return _verify_run(lambda name: (directory / name).read_bytes(), require)


def later_attempt(read: Callable[[str], bytes]) -> dict[str, Any] | None:
    """The attempt that followed the run read by ``read(name) -> bytes`` and did not complete, from the stop record beside
    it (``run_stopped.json``): ``{"execution", "cause", "phase", "project_hash", "same_project"}`` -- its execution outcome
    (``CANCELLED``), why and in which phase it stopped, the hash of the Project it ran (empty: stopped before the Project
    was read) and whether that is the Project of the run kept. ``None``: no stop record is there (a run that completes
    removes it). The run kept is the earlier one: its manifest states the Project and Engine input it is the run of."""

    try:
        record = json.loads(read(STOPPED_FILE))
        if not isinstance(record, dict) or record.get("kind") != STOPPED_RUN_KIND:
            return None
        attempt = {"execution": record["outcome"]["execution"]["outcome"], "cause": record["stop"]["cause"],
                   "phase": record["stop"]["phase"], "project_hash": record.get("project_hash") or ""}
    except (OSError, ValueError, KeyError, TypeError, RecursionError):
        return None
    try:
        kept = _object(_object(_object(json.loads(read(MANIFEST_FILE))).get("project")).get("project_hash")).get("digest")
    except (OSError, ValueError, RecursionError):
        kept = None
    return {**attempt, "same_project": bool(attempt["project_hash"]) and attempt["project_hash"] == kept}


def later_attempt_in(directory: Path | str) -> dict[str, Any] | None:
    """``later_attempt`` of a run directory."""

    return later_attempt(lambda name: (Path(directory) / name).read_bytes())


def verify_run_files(files: Mapping[str, bytes], *, require: frozenset[str] | None = None) -> tuple[str, ...]:
    """``verify_run_artifacts`` on a run directory held in memory: ``files`` maps each file name to its bytes (the
    ``files`` of ``json_api.run_project`` / ``run_artifact_files``). A missing file is a problem (one outside ``require``
    is not); other names are ignored, like other files in a run directory. Returns the problems found (empty = consistent)."""

    def read(name: str) -> bytes:
        data = files.get(name)
        if data is None:
            raise FileNotFoundError(f"no file named {name}")
        return bytes(data)

    return _verify_run(read, require)


def _verify_run(read: Callable[[str], bytes], require: frozenset[str] | None = None) -> tuple[str, ...]:
    problems: list[str] = []
    try:
        return _verify_run_checked(read, problems, require)
    except (AttributeError, KeyError, TypeError, ValueError, RecursionError, ArithmeticError) as exc:
        # a run file of the wrong shape that a check above did not anticipate: inconsistent, never a crash
        return (*problems, f"the run files cannot be checked: a file has an unexpected structure ({type(exc).__name__})")


def _stopped_note(read: Callable[[str], bytes]) -> str | None:
    """The problem of a directory that holds no run but the record of a run that stopped before it completed
    (``run_stopped.json``), else ``None``."""

    try:
        record = json.loads(read(STOPPED_FILE))
        stop = record["stop"]
        execution = record["outcome"]["execution"]
        return (f"not a run directory: the run written here stopped before it completed (execution {execution['outcome']}, "
                f"{stop['cause']}, in phase {stop['phase']}: {stop['message']}); {STOPPED_FILE} holds its record and no "
                "result of it is reported")
    except (OSError, ValueError, KeyError, TypeError, RecursionError):
        return None


def _absent(read: Callable[[str], bytes], require: frozenset[str] | None) -> frozenset[str]:
    """The run files outside ``require`` that are not there (none when every file is required)."""

    if require is None:
        return frozenset()
    absent = set()
    for _role, name, _media, _what in RUN_FILES:
        if name not in require:
            try:
                read(name)
            except FileNotFoundError:
                absent.add(name)
    return frozenset(absent)


def _verify_run_checked(read: Callable[[str], bytes], problems: list[str],
                        require: frozenset[str] | None = None) -> tuple[str, ...]:
    try:
        read(MANIFEST_FILE)
    except FileNotFoundError:
        stopped = _stopped_note(read)
        if stopped:
            return (stopped,)
        return (f"not a run directory: it has no {MANIFEST_FILE} (write one with 'os-optics project run PROJECT "
                "--out-dir DIR')",)
    except OSError:
        pass  # reported by _load_json
    manifest = _load_json(read, MANIFEST_FILE, problems)
    if not isinstance(manifest, dict):
        return tuple(problems or [f"{MANIFEST_FILE}: not a JSON object"])
    from .configurations import is_configurations_manifest, verify_configurations_run

    if is_configurations_manifest(manifest):  # a Project with configurations: each configuration's run directory and the whole
        return verify_configurations_run(read)
    version = manifest.get("schema_version")
    if manifest.get("schema") != RUN_MANIFEST_SCHEMA or version not in RUN_MANIFEST_READ_VERSIONS:
        return (f"{MANIFEST_FILE}: not an {RUN_MANIFEST_SCHEMA} {' or '.join(RUN_MANIFEST_READ_VERSIONS)} manifest",)
    with_ray_data = True
    entries = manifest.get("files")
    if not isinstance(entries, list) or not all(isinstance(entry, dict) for entry in entries):
        problems.append(f"{MANIFEST_FILE}: its file list is not a list of file entries")
        entries = []
    listed = {entry.get("role"): entry for entry in entries}
    absent = _absent(read, require)
    for role, name, _media, _what in RUN_FILES:
        if (role == RAY_DATA_ROLE and not with_ray_data) or name in absent:
            continue
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
    project = _parse(ProjectDocumentV2, _load_json(read, PROJECT_FILE, problems), PROJECT_FILE, problems)
    request = _parse(ValidationRequest, _load_json(read, ENGINE_REQUEST_FILE, problems), ENGINE_REQUEST_FILE, problems)
    stored_result = _load_json(read, SIMULATION_RESULT_FILE, problems)
    result = _parse(ValidationResult, stored_result, SIMULATION_RESULT_FILE, problems)
    record = _parse(LocalValidationRecordV1, _load_json(read, LOCAL_VALIDATION_FILE, problems), LOCAL_VALIDATION_FILE, problems)
    scene = None if SCENE_FILE in absent else _parse(OpticalSimulationSceneV1, _load_json(read, SCENE_FILE, problems), SCENE_FILE,
                                                     problems)
    if project is not None and record is not None:
        problems.extend(f"{LOCAL_VALIDATION_FILE}: {p}" for p in check_local_validation_binding(record, project))
    if record is not None:
        # the manifest is read first ("start here"): what it states must be the record's
        if manifest.get("project") != record.project.model_dump(mode="json", by_alias=True, exclude_none=True):
            problems.append(f"{MANIFEST_FILE}: its project differs from the local validation record's")
        status = _object(manifest.get("status"))
        # (a finding's structured details stay with the manifest's listing: the record holds its applicability fields)
        from .record import applicability_fields

        scope_rows = status.get("model_scope_issues")
        if isinstance(scope_rows, list):
            scope_rows = [applicability_fields(row) if isinstance(row, dict) else row for row in scope_rows]
        if record.simulation.model_scope_issues is not None and scope_rows != [
                i.model_dump(mode="json", exclude_none=True) for i in record.simulation.model_scope_issues]:
            problems.append(f"{MANIFEST_FILE}: its model applicability differs from the local validation record's")
        if (status.get("validation_status") != record.validation_status.value
                or status.get("simulation_status") != record.simulation.status.value):
            problems.append(f"{MANIFEST_FILE}: its status differs from the local validation record's")
        if version != "1.1" and _object(manifest.get("engine_run")).get("engine_package") != record.runtime.engine_package:
            problems.append(f"{MANIFEST_FILE}: its engine package differs from the local validation record's")
    if request is not None and record is not None and hash_exact(request.model_dump(mode="json")) != record.input_hash.digest:
        problems.append(f"{ENGINE_REQUEST_FILE}: does not hash to the record's Engine input_hash")
    if isinstance(stored_result, dict) and request is not None:
        cfg = request.configuration.numerical
        # (the body as stored: a file written before a contract field existed gains that field when it is read as a
        # model, so re-serializing would change its hash and no historical run directory would verify. A dict as read,
        # with its manifest removed, is the body the manifest hashed.)
        body = dict(stored_result)
        body["manifest"] = None
        expected_hash = _object(stored_result.get("manifest")).get("result_hash")
        if expected_hash is None or hash_quantized_parts(dict_parts(body), cfg.hash_float_sig_digits,
                                                         cfg.hash_float_abs_floor) != expected_hash:
            problems.append(f"{SIMULATION_RESULT_FILE}: does not match its own manifest result_hash")
        del body
    del stored_result  # (the parsed result model stands for it below; the ray data is read next)
    if record is not None and scene is not None and result is not None:
        problems.extend(cross_artifact_problems(record, scene, result))
    if with_ray_data and RAY_DATA_FILE not in absent:
        ray_data = _load_json(read, RAY_DATA_FILE, problems)
        if ray_data is not None and result is not None:
            problems.extend(f"{RAY_DATA_FILE}: {p}" for p in ray_data_problems(ray_data, result))
        if isinstance(ray_data, dict) and _object(manifest.get(RAY_DATA_ROLE)).get("trace_hash") != ray_data.get("trace_hash"):
            problems.append(f"{RAY_DATA_FILE}: its trace hash differs from {MANIFEST_FILE}")
        if isinstance(ray_data, dict) and ray_data_recording(ray_data) != _object(manifest.get(RAY_DATA_ROLE)).get("recording", RAY_DATA_FULL):
            problems.append(f"{RAY_DATA_FILE}: its recording level differs from {MANIFEST_FILE}")
        del ray_data
    if scene is not None:
        if scene.content_hash() != _object(manifest.get("optical_view")).get("scene_content_hash_osc1"):
            problems.append(f"{SCENE_FILE}: content hash differs from {MANIFEST_FILE}")
    if scene is not None and GLB_FILE not in absent:
        try:
            gltf, _binary = parse_glb(read(GLB_FILE))
            extras = gltf["asset"].get("extras", {})
            if extras.get("scene_content_hash") != scene.content_hash():
                problems.append(f"{GLB_FILE}: was not rendered from {SCENE_FILE}")
            if record is not None and extras.get("engine_result_hash") != record.result_hash.digest:
                problems.append(f"{GLB_FILE}: its Engine result hash differs from the local validation record's")
        except (OSError, GlbFormatError, KeyError, ValueError) as exc:
            problems.append(f"{GLB_FILE}: not a valid GLB ({exc})")
    return tuple(problems)
