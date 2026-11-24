"""The commercial submission of one local run, and the idempotency keys of commercial calls -- transport-free.

``submission_document(run)`` builds what ``POST /api/v1/commercial-submissions`` takes: the Project, its osc1 hash and
the Local Validation record of one local run. ``canonical_json_bytes`` is the request-body form, and
``idempotency_key_for(operation, *parts)`` the deterministic ``Idempotency-Key`` of each commercial call
(``opensci-<operation>-<sha256>``): ``("submit", canonical_json_bytes(document))``, ``("order", resolution_id)``,
``("bank-transfer-report", order_id)``, ``("artifact-set", artifact_set_digest)``.

This module does no network, environment, token or clock access, so a local JSON host can use it
through ``os_optics.local.json_api.commercial_submission``; ``os_optics.client`` re-exports these names for
Python callers (``OpenSciClient.submit``, ``os-optics resolve submit``).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SUBMISSION_SCHEMA = "opensci.commercial_submission"
SUBMISSION_SCHEMA_VERSION = "2.0"
#: The file name of a submission body (``json_api.commercial_submission`` returns it under this name).
SUBMISSION_FILE = "submission.json"
RUN_DIRECTORY_INCONSISTENT = "RUN_DIRECTORY_INCONSISTENT"
CONFIGURATIONS_NOT_SUBMITTABLE = "CONFIGURATIONS_NOT_SUBMITTABLE"
MAX_REPORTED_PROBLEMS = 20


class RunDirectoryInconsistentError(ValueError):
    """The run directory (or its files held in memory) does not verify (``verify_run_artifacts`` /
    ``verify_run_files``): it is not submitted. ``problems`` lists at most ``MAX_REPORTED_PROBLEMS`` of them."""

    code = RUN_DIRECTORY_INCONSISTENT

    def __init__(self, problems: tuple[str, ...] | list[str], *, run_dir: str | None = None) -> None:
        self.problems = tuple(problems)[:MAX_REPORTED_PROBLEMS]
        self.run_dir = run_dir
        super().__init__("the run directory does not verify; run the Project again")

    @property
    def details(self) -> dict[str, Any]:
        details: dict[str, Any] = {} if self.run_dir is None else {"run_dir": self.run_dir}
        details["problems"] = list(self.problems)
        return details


class ConfigurationsNotSubmittableError(RunDirectoryInconsistentError):
    """The run is the run of a Project with configurations: a commercial submission carries one Project and the record of
    its one run, so it is not submitted. A configuration's own run directory (``configurations/<id>``) is the run of
    that configuration's ordinary Project and is submitted as such."""

    code = CONFIGURATIONS_NOT_SUBMITTABLE

    def __init__(self, configuration_ids: tuple[str, ...] | list[str], *, run_dir: str | None = None) -> None:
        ids = ", ".join(configuration_ids)
        super().__init__((f"the run is a run of a Project with configurations ({ids}); a submission carries one Project and "
                          "one run: submit the run directory of one configuration (configurations/<id>)",), run_dir=run_dir)
        self.args = (f"a Project with configurations ({ids}) is not submitted; submit one configuration's run directory "
                     "(configurations/<id>)",)


def _refuse_configured(project: Any, run_dir: str | None = None) -> None:
    if project.configurations:
        raise ConfigurationsNotSubmittableError([c.configuration_id for c in project.configurations], run_dir=run_dir)


def canonical_json_bytes(document: Any) -> bytes:
    """Sorted-key, compact UTF-8 JSON: the form idempotency keys are derived from."""

    return json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def idempotency_key_for(operation: str, *parts: str | bytes) -> str:
    """The deterministic ``Idempotency-Key`` of one operation on one input: ``opensci-<operation>-<sha256>``.

    The same operation on the same input always gets the same key, so a repeated call (after a lost response, a crash
    or a second run of a script) returns the original result instead of creating a second one. The digest is SHA-256
    over the UTF-8 operation, then for each part a zero byte and the part (bytes as given, text as UTF-8)."""

    digest = hashlib.sha256(operation.encode("utf-8"))
    for part in parts:
        digest.update(b"\x00")
        digest.update(part if isinstance(part, bytes) else part.encode("utf-8"))
    return f"opensci-{operation}-{digest.hexdigest()}"


def submission_document(run: Any, *, customer_constraints: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The commercial submission of one local run: its Project, Project hash and Local Validation record.

    ``run`` is the ``LocalOpticalRun`` of ``os_optics.local.run_optical_project`` (its record checked against its Project:
    ``run.binding_problems()``), the run directory written by ``os-optics project run --out-dir`` (checked with
    ``verify_run_artifacts`` first), or that directory's files held in memory as ``{file name: bytes}``
    (``json_api.run_project``'s ``files``; checked with ``verify_run_files``). This is where a run leaves the device, so
    it is where it is verified: a run, a run directory or a file set that does not verify raises
    ``RunDirectoryInconsistentError``. A submission needs the files of ``SUBMISSION_FILES`` (manifest, Project, Engine
    request and result, Local Validation record); the optical scene, its GLB and the ray data may be absent and are
    checked when present.
    ``customer_constraints`` (optional): ``max_board_size_mm`` ``[x, y]``, ``require_all_retained_dofs`` and the
    purchase options ``include_breadboard`` (default true; false: the customer's own optical table) and
    ``assembly_service`` (default false; requires the breadboard), ``customer_supplied_components`` (instance ids of
    design components the customer supplies themselves; they are not on the Customer Purchase BOM), ``purchase_scope``
    (``COMPLETE_SYSTEM`` by default, or ``OPTICS_ONLY``: the optical elements only, no mounting hardware) and
    ``spare_units`` (``{OpenSci SKU: count}``: spare units of the design's OpenSci optics); these keys encode the order
    options mechanical supply / platform / assembly. With a laboratory layout the beam height is derived by the service;
    ``mounting`` (``{instance_id: {"beam_height_mm", "incidence_deg"?}}``) states it for each OpenSci optic of an
    ``OPTICAL_SYSTEM`` design bought with OpenSci mounts and no platform.

    A run of a Project with configurations (``os_optics.local.configurations``) is refused with
    ``ConfigurationsNotSubmittableError`` (code ``CONFIGURATIONS_NOT_SUBMITTABLE``, a ``RunDirectoryInconsistentError``):
    one configuration's run directory is the run of an ordinary Project and is submitted instead."""

    from .artifacts import LOCAL_VALIDATION_FILE, PROJECT_FILE, SUBMISSION_FILES, verify_run_artifacts, verify_run_files
    from .catalog import load_project
    from .run import LocalOpticalRun

    from .configurations import ConfiguredProjectRun

    if isinstance(run, ConfiguredProjectRun):
        _refuse_configured(run.project)
    if isinstance(run, LocalOpticalRun):
        problems = run.binding_problems()
        if problems:
            raise RunDirectoryInconsistentError(problems)
        project, record = run.project, run.local_validation.payload()
    elif isinstance(run, (str, Path)):
        directory = Path(run)
        problems = verify_run_artifacts(directory, require=SUBMISSION_FILES)
        if problems:
            raise RunDirectoryInconsistentError(problems, run_dir=str(directory))
        project = load_project((directory / PROJECT_FILE).read_bytes())
        _refuse_configured(project, str(directory))
        record = json.loads((directory / LOCAL_VALIDATION_FILE).read_bytes())
    elif isinstance(run, Mapping):
        problems = verify_run_files(run, require=SUBMISSION_FILES)
        if problems:
            raise RunDirectoryInconsistentError(problems)
        project = load_project(bytes(run[PROJECT_FILE]))
        _refuse_configured(project)
        record = json.loads(bytes(run[LOCAL_VALIDATION_FILE]))
    else:
        raise TypeError("run must be a LocalOpticalRun, the path of a run directory or its {file name: bytes} files")
    document: dict[str, Any] = {"schema": SUBMISSION_SCHEMA, "schema_version": SUBMISSION_SCHEMA_VERSION,
                                "project_hash": project.content_hash(), "project": project.payload(),
                                "local_validation": record}
    if customer_constraints is not None:
        document["customer_constraints"] = dict(customer_constraints)
    return document


__all__ = [
    "CONFIGURATIONS_NOT_SUBMITTABLE",
    "MAX_REPORTED_PROBLEMS",
    "RUN_DIRECTORY_INCONSISTENT",
    "SUBMISSION_FILE",
    "SUBMISSION_SCHEMA",
    "SUBMISSION_SCHEMA_VERSION",
    "ConfigurationsNotSubmittableError",
    "RunDirectoryInconsistentError",
    "canonical_json_bytes",
    "idempotency_key_for",
    "submission_document",
]
