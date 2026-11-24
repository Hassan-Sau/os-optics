"""Export a Project's optics as an ``opensci-linear-optical-network/v1`` document (``os_optics.network``).

The Project is lowered exactly as a run lowers it (``lower_project``: catalog products, sources, observation planes), so
the network describes the same optics a run simulates. The published-model validity of a run applies too, to the light the
export describes: when the exported light is outside a published model's characterization (``model_scope_issues``) the
export is refused with ``OUT_OF_MODEL_SCOPE``; a violation whose light reaches only planes the export leaves out
(``affected_observation_ids``) is listed in ``validity.warnings`` instead. It is checked on the run of the export's selection
(``network.export.export_scope_projects``: the selected sources, each emitting only its line at the exported wavelength; a
source the selection leaves out, or a line it does not export, never refuses the export) with the declared polarizations
and on the runs with every exported source lab-H and lab-V polarized (the two input columns of every source), so a branch
only one polarization feeds (a polarizing splitter's other port) is checked too.
"""

from __future__ import annotations

from typing import Any

from ..contracts.network import LinearOpticalNetworkV1, NetworkExportOptionsV1, ValidityIssueV1
from ..contracts.request import ValidationRequest
from ..manifest import hash_exact
from ..network import NetworkErrorCode, NetworkExportError, NetworkProblem, build_linear_optical_network
from ..network.export import export_scope_projects
from ..validation.ray_paths import validate_with_ray_paths
from .catalog import installed_catalogs, load_project, load_public_catalog, pinned_catalog_version
from .lowering import lower_project
from .validity import model_scope_issues


def _catalog_sha256(catalog_version: str) -> str | None:
    """SHA-256 of the release bytes when this runtime holds them installed (the browser runtime, ``install_catalog``) or
    as the local store's active release; ``None`` otherwise (never fetched for this)."""

    for item in installed_catalogs():
        if item.catalog.catalog_version == catalog_version:
            return item.sha256
    from .catalog import catalog_resolution_status

    active = catalog_resolution_status().get("active")
    if active is not None and active.get("catalog_version") == catalog_version:
        return active.get("sha256")
    return None


def export_linear_optical_network(project: Any, *, catalog: Any = None, options: NetworkExportOptionsV1 | dict | None = None,
                                  model_kind: str | None = None, configuration_id: str | None = None) -> LinearOpticalNetworkV1:
    """The linear optical network of a Project (``ProjectDocumentV2``, its JSON document / text, or a file path).

    ``catalog`` defaults to the release the Project pins (else the active one), as for a run; a Project without catalog
    components loads none and its provenance names none. ``options``:
    ``NetworkExportOptionsV1`` or its JSON form (``wavelength_nm``, ``source_ids``, ``observation_ids``,
    ``mode_overlap_tolerance``). ``configuration_id``: the configuration to export, of a Project that declares
    configurations (``configurations.configured_project``). Raises ``LocalRunError`` for a Project the runtime cannot
    simulate and ``NetworkExportError`` for optics that are not a linear optical network."""

    from .configurations import configured_project  # (configurations.py imports the run, which imports this package's lowering)

    document = configured_project(project, configuration_id)
    explicit = catalog is not None
    uses_catalog = any(c.product is not None for c in document.components)
    public_catalog = None  # as for a run: a Project without catalog components needs (and names) no catalog
    if uses_catalog:
        public_catalog = load_public_catalog(catalog, catalog_version=pinned_catalog_version(document) if catalog is None else None)
    lowered = lower_project(document, public_catalog, model_kind=model_kind)
    options = _options(options)
    request = lowered.request
    warnings: dict[tuple[str, str], ValidityIssueV1] = {}
    if lowered.elements:  # a published model used outside its characterization is not a faithful network either
        problems: list[NetworkProblem] = []
        outputs = set(options.observation_ids if options.observation_ids is not None
                      else (o.observation_id for o in request.project.observations))
        for project in export_scope_projects(request.project, request.configuration.numerical, options):
            result, ray_paths = validate_with_ray_paths(request.model_copy(update={"project": project}))
            # (a tabulated thin lens's own domain issues are Engine issues of the probe run, which the export refuses itself)
            for issue in model_scope_issues(lowered, result, ray_paths, tabulated_domains=False):
                location = f"component {issue.instance_id}" + (f" surface {issue.surface_id}" if issue.surface_id else "")
                exported = set(issue.affected_observation_ids) & outputs
                if issue.affected_observation_ids and not exported:  # light no exported output receives
                    warnings.setdefault((issue.code, location), ValidityIssueV1(
                        code=issue.code, location=location, message="outside a published model's characterization on light that "
                        "reaches no exported output"))
                    continue
                problem = NetworkProblem(NetworkErrorCode.OUT_OF_MODEL_SCOPE, f"{issue.code}: {issue.message}", location,
                                         observation_id=next(iter(exported)) if len(exported) == 1 else None)
                if problem not in problems:
                    problems.append(problem)
        if problems:
            raise NetworkExportError(problems)
    return build_linear_optical_network(
        request.project, request.configuration.numerical, options, engine_input_hash=lowered.engine_input_hash,
        project_hash=document.content_hash(),
        catalog_version=lowered.catalog_version if uses_catalog else None,
        catalog_sha256=None if not uses_catalog or explicit else _catalog_sha256(lowered.catalog_version),
        warnings=tuple(warnings.values()))


def export_request_network(request: ValidationRequest | dict | str | bytes, *, options: NetworkExportOptionsV1 | dict | None = None
                           ) -> LinearOpticalNetworkV1:
    """The linear optical network of an Engine ``ValidationRequest`` (low-level input: its ``project`` snapshot and numerical
    configuration; no catalog)."""

    if not isinstance(request, ValidationRequest):
        request = ValidationRequest.model_validate_json(request) if isinstance(request, (str, bytes)) else ValidationRequest.model_validate(request)
    return build_linear_optical_network(request.project, request.configuration.numerical, _options(options),
                                        engine_input_hash=hash_exact(request.model_dump(mode="json")))


def _options(options: Any) -> NetworkExportOptionsV1:
    if options is None:
        return NetworkExportOptionsV1()
    if isinstance(options, NetworkExportOptionsV1):
        return options
    from pydantic import ValidationError

    try:
        return NetworkExportOptionsV1.model_validate(options)
    except ValidationError as exc:
        errors = "; ".join(f"{'.'.join(str(p) for p in e['loc']) or 'options'}: {e['msg']}" for e in exc.errors(include_url=False))
        raise NetworkExportError([NetworkProblem(NetworkErrorCode.OPTIONS_INVALID, errors)]) from None
