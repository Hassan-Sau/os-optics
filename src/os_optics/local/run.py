"""``run_optical_project``: public catalog + pure-optical Project -> ONE real Engine run -> simulation result, local
validation record and Optical Simulation View, optionally written to a run directory."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path, PurePath
from typing import Any

from pydantic import ValidationError

from .._vendor.shared_contracts.optical_design import (
    LocalRuntimeKindV1,
    LocalValidationRecordV1,
    LocalValidationStatusV1,
    ProjectDocumentV2,
    SimulatedPartV1,
    check_local_validation_binding,
)
from .._vendor.shared_contracts.design_catalog import ModelFidelity, SimulationApproximation
from .._vendor.shared_contracts.optical_design.optical_model import DERIVED_APPROXIMATIONS
from .._vendor.shared_contracts.optical_view import OpticalSimulationSceneV1, OpticalViewSettingsV1
from ..contracts.outcome import ExecutionRecord, RequirementRecord, ResultOutcome, RunStop, StoppedRun
from ..contracts.request import ValidationRequest
from ..contracts.results import ValidationResult
from ..errors import ExecutionOutcome, Status, TerminationReason
from ..run_control import RunStopped, active_monitor, begin_attempt, checkpoint, end_attempt, monitored, phase
from ..optical_view import OpticalViewBindingError, build_optical_simulation_scene
from ..validation.pipeline import NUMERICAL_FAILURE_MESSAGE
from ..validation.pipeline import evaluate, quadrature_refusal
from ..validation.ray_paths import RAY_DATA_FULL, RAY_DATA_LEVELS, RayPathTrace, recorded_ray_paths
from .artifacts import RunArtifacts, committed_run, fixed_run_file_bytes, record_stopped_run, write_run_artifacts
from .catalog import _raw, _validation_message, load_project, load_public_catalog, pinned_catalog_version, resolved_release
from .errors import LocalRunError, LocalRunErrorCode, LocalRunProblem, fail
from ..optics.gaussian.truncation import QuadratureRuleTooLarge
from ..optics.surfaces.facets import coplanar_tolerance, planar_patches
from .compile import DIFFRACTION_GRATING, FACET_SOLID, FARADAY_ROTATOR, IDEAL_THIN_LENS, TABULATED_THIN_LENS, engine_supported_metrics
from .lowering import inputs_not_simulated, lower_project, solid_check
from .sample_emission import check_excitation
from .record import local_validation_record, target_row_bases
from .resources import estimate_after_trace, estimate_run_resources
from .validity import ModelScopeIssue, model_scope_issues


@dataclass(frozen=True)
class LocalOpticalRun:
    """One local run of a pure-optical Project. Everything here comes from the same single Engine execution.

    * ``status``: the diagnostic summary -- ``PASS`` / ``WARNING`` / ``FAIL`` / ``UNKNOWN`` / ``OUT_OF_MODEL_SCOPE`` /
      ``NOT_APPLICABLE`` (``local_validation.validation_status``), never an approval of the design; ``engine_status`` is the
      Engine's own; what was decided is ``local_validation.simulation.outcome`` (execution, requirements) with each target's
      ``assessment`` and each plane's ``quantity_state``;
    * ``result``: the full Engine ``ValidationResult`` (targets with values, observations, beam paths, issues);
    * ``local_validation``: the ``LocalValidationRecordV1`` to keep (hashes, runtime identity, retained DOFs);
    * ``optical_view``: the ``OpticalSimulationSceneV1`` (3D Optical Simulation View) of this run;
    * ``artifacts``: where the run directory was written (``RunArtifacts``), or ``None`` if not written yet --
      call ``write_artifacts(dir)``;
    * ``model_scope_issues``: every use of a model outside its validity found after the trace (a published or declared
      model's band, coatings and angles, light meeting an element in a way the trace does not follow) and every use of a
      tabulated thin lens or Faraday rotator outside its table's domain (also an Engine issue in ``result.issues``, status
      ``OUT_OF_MODEL_SCOPE``); each entry names the observations, targets and elements it affects. The run used a
      model outside its validity when this list is not empty or any Engine issue or target has status
      ``OUT_OF_MODEL_SCOPE``; the status is the worst outcome, so a failing target
      makes such a run ``FAIL``. ``results.load_run(run).targets()`` gives ``in_model_scope`` and every finding in
      ``out_of_model_issues``.
    """

    project: ProjectDocumentV2
    catalog_version: str | None
    simulated_parts: tuple[SimulatedPartV1, ...]
    request: ValidationRequest
    engine_input_hash: str
    result: ValidationResult
    local_validation: LocalValidationRecordV1
    ray_paths: RayPathTrace
    optical_view: OpticalSimulationSceneV1
    model_scope_issues: tuple[ModelScopeIssue, ...] = ()
    artifacts: RunArtifacts | None = None
    element_models: tuple[dict[str, Any], ...] = ()
    #: What ``ray_data.json`` records (``validation/ray_paths.py`` ``RAY_DATA_LEVELS``): ``full`` every chief and pupil
    #: ray, ``summary`` the chief rays only. The trace is the same either way.
    ray_data: str = RAY_DATA_FULL

    @property
    def status(self) -> LocalValidationStatusV1:
        """The validation outcome (``local_validation.validation_status``): the Engine status, made at least
        ``OUT_OF_MODEL_SCOPE`` when a ``model_scope_issues`` entry exists."""

        return self.local_validation.validation_status

    @property
    def engine_status(self) -> Status:
        """The Engine's own overall status (``local_validation.simulation.status``)."""

        return self.result.status

    def binding_problems(self) -> tuple[str, ...]:
        """``check_local_validation_binding`` of the record against the Project (empty = consistent). A run does not
        refuse itself on it: the run directory's manifest records the problems (``binding``), ``verify_run_artifacts``
        reports them and a submission of the run is refused (``local/submission.py``)."""

        return check_local_validation_binding(self.local_validation, self.project)

    def write_artifacts(self, out_dir: Path | str) -> RunArtifacts:
        """Write the run directory (``manifest.json``, ``project.json``, ``engine_request.json``,
        ``simulation_result.json``, ``ray_data.json``, ``local_validation.json``, ``optical_scene.json``,
        ``optical_scene.glb``)."""

        return write_run_artifacts(self, out_dir)

    def summary(self) -> dict[str, Any]:
        """A compact JSON-safe report: status, per-target outcomes, public model binding, artifact paths and the record."""

        artifacts = None
        effective = {t.target_id: t for t in self.local_validation.simulation.target_outcomes}
        if self.artifacts is not None:
            artifacts = {"directory": str(self.artifacts.directory), **{role: str(path) for role, path in self.artifacts.paths.items()}}
        return {
            # (first what decides: did the calculation complete, and how did the declared targets come out; then the
            # aggregate diagnostic severity under its own name -- the same value as "status", which decides nothing)
            **outcome_lead(self.local_validation.simulation.outcome, self.status.value),
            "status": self.status.value,
            "engine_status": self.result.status.value,
            "outcome": None if self.local_validation.simulation.outcome is None else self.local_validation.simulation.outcome.model_dump(mode="json"),
            "project_hash": self.local_validation.project.project_hash.digest,
            "design_intent": self.project.effective_design_intent.value,
            "catalog_version": self.catalog_version,
            "simulated_parts": [part.model_dump(mode="json", exclude_none=True) for part in self.simulated_parts],
            "element_models": [dict(model) for model in self.element_models],
            # (inputs of the Project the simulation did not read: present only when there are any)
            **({"inputs_not_simulated": [dict(row) for row in inputs_not_simulated(self.project)]}
               if inputs_not_simulated(self.project) else {}),
            "targets": [
                {"target_id": t.target_id, "metric": t.metric, "status": t.status.value,
                 "assessment": effective[t.target_id].assessment, "value": t.value, "unit": t.unit,
                 "target_value": t.target_value, "error": t.error,
                 "message": t.message, "issue_codes": sorted(set(t.issue_codes) | set(effective[t.target_id].applicability_issue_codes or ())),
                 "applicability_issue_codes": list(effective[t.target_id].applicability_issue_codes or ()),
                 **target_row_bases(t, self.model_scope_issues),
                 **({"validity": [v.model_dump(mode="json") for v in t.validity]} if t.validity else {})}
                for t in self.result.targets
            ],
            # each issue in full, as in simulation_result.json: where it is, its numbers and what to do about it
            "issues": [i.model_dump(mode="json") for i in self.result.issues],
            "model_scope_issues": [issue.as_dict() for issue in self.model_scope_issues],
            "retained_dofs": [dof.model_dump(mode="json", exclude_none=True) for dof in self.local_validation.retained_dofs],
            "optical_view": {
                "scene_content_hash_osc1": self.optical_view.content_hash(),
                "beam_paths": len(self.optical_view.beam_paths),
                "elements": [e.element_id for e in self.optical_view.elements],
                "observation_planes": [p.observation_id for p in self.optical_view.observation_planes],
                # how to look at it: the local 3D viewer (the same text in every runtime, so runs stay byte-comparable)
                **({"view_command": VIEW_COMMAND} if self.artifacts is not None else {}),
            },
            "artifacts": artifacts,
            "local_validation": self.local_validation.payload(),
        }


def _approximations(element: Any) -> list[str]:
    """The model's approximation labels, sorted: those it states, with the two that follow from the model itself
    (``DERIVED_APPROXIMATIONS``) as they hold -- ``TYPICAL_SPECTRAL_CURVE`` when a coating it uses is a typical curve,
    ``UNTRACED_FACES_BLOCK`` when a face of its body carries no traced surface -- whether or not a declared model lists them."""

    derived = {label.value for label in DERIVED_APPROXIMATIONS}
    labels = {a.value for a in element.model.approximations} - derived
    if any(envelope.coating is not None and envelope.coating.is_typical for envelope in element.envelopes):
        labels.add(SimulationApproximation.TYPICAL_SPECTRAL_CURVE.value)
    if getattr(element.model, "has_untraced_faces", False):
        labels.add(SimulationApproximation.UNTRACED_FACES_BLOCK.value)
    return sorted(labels)


def outcome_lead(outcome: Any, status: str) -> dict[str, Any]:
    """The first keys of a run summary or reply, in reading order: ``execution`` (did the calculation complete: its
    outcome, termination and ``complete``), ``requirements`` (the assessed targets by assessment: ``declared``, ``met``,
    ``unmet``, ``indeterminate``, ``not_applicable``; a run assesses every declared target, and an evaluation's reply adds
    the declared targets it did not assess: ``requirements_coverage``) and ``diagnostic_summary`` (the aggregate diagnostic severity, the
    worst of the findings and target statuses: the value of ``status``). ``outcome``: the record's or the result's outcome
    block; ``None`` (a record written before the block existed) gives null for the first two."""
    execution = requirements = None
    if outcome is not None:
        record = outcome.model_dump(mode="json")
        execution = {key: record["execution"][key] for key in ("outcome", "termination", "complete")}
        requirements = record["requirements"]
    return {"execution": execution, "requirements": requirements, "diagnostic_summary": status}


def requirements_coverage(requirements: dict[str, Any] | None, scope: Any) -> dict[str, Any] | None:
    """The requirements block of an evaluation's reply: ``requirements`` (``outcome_lead``'s: ``declared`` and the targets
    by assessment) with ``not_assessed`` -- each declared target the evaluation did not assess, in Project order, with
    its reason (``scope``: ``ValidationResult.evaluation``; ``None`` for a complete evaluation, which assesses every one).
    ``met``, ``unmet``, ``indeterminate``, ``not_applicable`` and ``not_assessed`` together name every declared target
    once."""
    if requirements is None:
        return None
    return {**requirements, "not_assessed": [] if scope is None else [
        {"target_id": t.target_id, "reason": t.reason} for t in scope.targets_not_evaluated]}


def _model_row(element: Any) -> dict[str, Any]:
    row: dict[str, Any] = {"model_kind": element.stated_kind}
    fidelity = getattr(element.model, "fidelity", None)
    if fidelity is not None:
        row["fidelity"] = fidelity.value
        row["approximations"] = _approximations(element)
    elif element.model_kind in (IDEAL_THIN_LENS, TABULATED_THIN_LENS, FARADAY_ROTATOR, DIFFRACTION_GRATING):
        row["fidelity"] = ModelFidelity.IDEAL_THIN_ELEMENT.value
    if element.model_kind == TABULATED_THIN_LENS:  # a response model: where its table comes from stays with the run
        row["response"] = tabulated_response_summary(element.model)
    if element.model_kind == FARADAY_ROTATOR:
        row["response"] = faraday_response_summary(element.model)
    if element.model_kind == DIFFRACTION_GRATING:
        row["response"] = grating_response_summary(element.model)
    if element.model_kind == FACET_SOLID:  # what its facets stand for, how many faces the run traces, how flat they are
        model = element.model
        vertices = model.vertex_floats()
        (surface,) = element.surfaces
        faces = planar_patches(vertices, model.triangles, surface.facet_group_of())
        row["facets"] = {"approximation": model.approximation, "triangles": len(model.triangles), "traced_faces": len(element.envelopes),
                         **({"max_chord_deviation_mm": model.max_chord_deviation_mm} if model.max_chord_deviation_mm is not None else {}),
                         **({"source": model.source} if model.source is not None else {}),
                         # (facets within this distance of one plane are one face; its corners lie within the largest
                         # deviation of the face's plane: optics/surfaces/facets.py)
                         "coplanar_tolerance_mm": float(f"{coplanar_tolerance(vertices):.3g}"),
                         "max_face_deviation_mm": float(f"{max(face.deviation_mm for face in faces):.3g}")}
    ideal = [envelope.surface_id for envelope in element.envelopes if envelope.ideal_reflector]
    if ideal:  # reflecting surfaces that state no coating: ideal reflectors (R = 1, reflection phases pi / 0)
        row["ideal_reflectors"] = ideal
    crystal = next((m for m in element.materials if getattr(m, "nonlinear", None) is not None), None)
    if crystal is not None:  # a NONLINEAR_CRYSTAL: the slab and the interaction the run computes
        spec = crystal.nonlinear
        row["second_harmonic"] = {"fundamental_waves": spec.fundamental_waves, "harmonic_wave": spec.harmonic_wave,
                                  "focusing": spec.focusing, **({"qpm_period_um": spec.qpm_period_um} if spec.qpm_period_um else {}),
                                  **({"temperature_c": crystal.temperature.temperature_c} if crystal.temperature is not None else {})}
    notes = model_notes(element.model)
    if notes:
        row["notes"] = notes
    return row


def model_notes(model: Any) -> list[str]:
    """What a model states that its results do not apply: a variable neutral density wheel's stated linearity (the run uses
    the nominal linear law of azimuth, without widening the transmittance by the stated deviation)."""

    law = getattr(model, "law", None)
    linearity = getattr(law, "linearity_percent", None)
    if linearity is None:
        return []
    return [f"The optical density follows the nominal linear law of azimuth from {law.od_start} to {law.od_end}; the stated "
            f"linearity of ±{linearity} % is not applied to the results."]


def tabulated_response_summary(model: Any) -> dict[str, Any]:
    """What a run reports about a tabulated thin lens: its wavelength domain, validity domain, the metrics it was validated
    for (with their error bounds and the Engine metrics they cover) and its evidence (method, tools, source data and
    configuration digests, fit, validation summary, date) -- so a result keeps the response's external origin.
    ``data_kind`` is ``EVIDENCED`` for a table that states its evidence and ``ASSUMED`` for one that does not; a limit, a
    metric list or evidence the model does not state reads ``null`` (``max_field_angle_deg``, ``supported_metrics``,
    ``engine_metrics``, ``evidence``): that limit is not checked."""

    domain: dict[str, Any] = {"wavelength_nm": [row.wavelength_nm for row in model.rows], "focal_interpolation": model.focal_interpolation,
                              "transmittance_interpolation": model.transmittance_interpolation, "max_field_angle_deg": model.max_field_angle_deg}
    if model.max_beam_radius_mm is not None:
        domain["max_beam_radius_mm"] = model.max_beam_radius_mm
    if model.validated_polarizations is not None:
        domain["validated_polarizations"] = list(model.validated_polarizations)
    metrics, engine_metrics, evidence = model.supported_metrics, engine_supported_metrics(model), model.evidence
    return {
        "data_kind": model.data_kind,
        "domain": domain,
        "supported_metrics": None if metrics is None else [entry.model_dump(mode="json", exclude_none=True) for entry in metrics],
        "engine_metrics": None if engine_metrics is None else list(engine_metrics),
        "evidence": None if evidence is None else evidence.model_dump(mode="json", exclude_none=True),
        "status": model.status.value,
    }


def faraday_response_summary(model: Any) -> dict[str, Any]:
    """What a run reports about a Faraday rotator: the sense of its rotation, the stated response, the operating conditions
    and evidence it rests on. ``data_kind`` is ``EVIDENCED`` for a table that states its evidence and ``ASSUMED`` for one
    that does not; an angle range, operating conditions or evidence the model does not state reads ``null``
    (``domain.valid_aoi_deg``: no angle of incidence is checked)."""

    angles, evidence = model.valid_aoi_deg, model.evidence
    return {
        "data_kind": model.data_kind,
        "rotation_sense": "right-handed about the element's local +Z, whichever way the light travels",
        "rows": [row.model_dump(mode="json") for row in model.rows],
        "domain": {"interpolation": model.interpolation, "valid_aoi_deg": None if angles is None else [angles.min_deg, angles.max_deg]},
        "operating_conditions": model.operating_conditions,
        "evidence": None if evidence is None else evidence.model_dump(mode="json", exclude_none=True),
        "status": model.status.value,
    }


def grating_response_summary(model: Any) -> dict[str, Any]:
    """What a run reports about a diffraction grating: its direction model (always available) and its stated order response
    with its provenance; phase is not stated."""

    response = model.response
    return {
        "mode": model.mode, "period_nm": format(model.period, "f"), "grating_vector_angle_deg": model.grating_vector_angle_deg,
        "direction_model": "n_exit t_exit = n_in t_in + m (lambda / period) K",
        "orders": list(model.orders) if model.orders is not None else None,
        "all_propagating_orders": list(model.all_propagating_orders) if model.all_propagating_orders is not None else None,
        "phase": "not stated",
        "order_response": None if response is None else {
            "normalization": response.normalization, "polarization_basis": response.polarization_basis, "interpolation": response.interpolation,
            "max_conical_angle_deg": response.max_conical_angle_deg, "data_kind": response.data_kind, "basis": response.basis,
            "rows": [row.model_dump(mode="json", exclude_none=True) for row in response.rows],
            "absorption": [row.model_dump(mode="json", exclude_none=True) for row in response.absorption],
            "evidence": None if response.evidence is None else response.evidence.model_dump(mode="json", exclude_none=True)},
        "status": model.status.value,
    }


def element_models(lowered: Any) -> tuple[dict[str, Any], ...]:
    """What each element's simulation is, in Project order: a catalog product's optical SKU, model kind, fidelity and the
    named approximations of its published model (for example a coated plane whose substrate is not modelled); a declared
    model's ``optical_model_id`` and model kind (with its fidelity and approximations when the model states them; an ideal
    or tabulated thin lens, a Faraday rotator and a diffraction grating are ``IDEAL_THIN_ELEMENT``; a tabulated thin lens adds
    its ``response``: ``tabulated_response_summary``, a Faraday rotator its own: ``faraday_response_summary``, a grating its own:
    ``grating_response_summary``), ``ideal_reflectors`` (the ids of its reflecting surfaces that state no coating: each is an
    ideal reflector, R = 1 with reflection phases pi for s and 0 for p, so the light behind it carries no coating loss),
    ``notes`` for what the model states that the results do not apply (``model_notes``:
    a variable neutral density wheel's linearity), and for a ``COMPOUND`` unit its ``members`` (``member_id``,
    ``optical_model_id``, model kind, ...)."""

    rows = []
    for e in lowered.elements:
        identity = {"optical_sku": e.optical_sku} if e.optical_sku is not None else {"optical_model_id": e.optical_model_id}
        row = {"instance_id": e.instance_id, **identity, **_model_row(e)}
        if e.is_compound:
            row["members"] = [{"member_id": m.member_id, "optical_model_id": m.optical_model_id, **_model_row(m)} for m in e.members]
        rows.append(row)
    return tuple(rows)


VIEW_COMMAND = "os-optics view <run directory>  (3D view in the browser; --snapshot DIR writes PNG images)"

def _numerical_failure(exc: BaseException) -> LocalRunError:
    return fail(LocalRunErrorCode.ENGINE_RUN_INCOMPLETE,
                f"the Engine could not evaluate the Project (NUMERICAL_FAILURE): {NUMERICAL_FAILURE_MESSAGE} "
                f"({type(exc).__name__})")


def targets_accounted(request: ValidationRequest, result: ValidationResult) -> bool:
    """Whether ``result`` names every declared target of ``request`` once, in Project order, either evaluated
    (``result.targets``) or not (``result.evaluation.targets_not_evaluated``): the partition the requirements of an
    evaluation's reply are read from (``requirements_coverage``)."""
    declared = [t.target_id for t in request.project.targets]
    evaluated = [t.target_id for t in result.targets]
    skipped = [t.target_id for t in result.evaluation.targets_not_evaluated] if result.evaluation is not None else []
    named = set(evaluated) | set(skipped)
    return (len(evaluated) + len(skipped) == len(declared) and named == set(declared)
            and evaluated == [i for i in declared if i in set(evaluated)] and skipped == [i for i in declared if i in set(skipped)])


def require_complete(request: ValidationRequest, result: ValidationResult) -> None:
    """An Engine run that rejected its input (or failed numerically) evaluates no targets; report the Engine's own
    diagnosis. A Project without targets is checked by the execution outcome itself, so its failed run is not returned
    as a run (``project run`` exits 65 with the Engine's problems, as for a Project with targets). A selective evaluation
    accounts for every declared target too, by id: each is evaluated or named in
    ``result.evaluation.targets_not_evaluated``, once, in Project order."""

    failed = result.outcome is not None and result.outcome.execution.outcome is ExecutionOutcome.FAILED
    if not failed and targets_accounted(request, result):
        return
    problems = [
        LocalRunProblem(LocalRunErrorCode.ENGINE_RUN_INCOMPLETE, f"the Engine could not evaluate the Project ({issue.code.value}): {issue.message}", issue.location)
        for issue in result.issues
    ] or [LocalRunProblem(LocalRunErrorCode.ENGINE_RUN_INCOMPLETE, f"the Engine evaluated {len(result.targets)} of {len(request.project.targets)} targets")]
    raise LocalRunError(problems)


def run_optical_project(
    project: Any,
    *,
    catalog: Any = None,
    model_kind: str | None = None,
    runtime_kind: LocalRuntimeKindV1 | str | None = None,
    output_dir: Path | str | None = None,
    view_settings: OpticalViewSettingsV1 | None = None,
    ray_data: str = RAY_DATA_FULL,
    monitor: Any = None,
) -> LocalOpticalRun:
    """Simulate a pure-optical Project with the real Engine on this device and produce its local validation record and
    its 3D Optical Simulation View -- all from one Engine execution.

    ``project`` is a ``ProjectDocumentV2``, its JSON document (dict), JSON text, or a file path (``pathlib.Path`` or
    ``str``); ``example_project()`` returns a complete reference design. ``catalog`` defaults to the public optical
    catalog release the Project's components pin (from the local catalog store, fetched once when missing), else the
    active local catalog (``load_public_catalog``). ``output_dir``, when given, receives the run directory (``manifest.json``,
    ``project.json``, ``engine_request.json``, ``simulation_result.json``, ``ray_data.json``, ``local_validation.json``,
    ``optical_scene.json``, ``optical_scene.glb``); ``run.artifacts`` then says where each file is.
    ``view_settings`` (``OpticalViewSettingsV1``) only changes what the view draws, never a number. ``ray_data``: what
    ``ray_data.json`` records -- ``full`` (default) every chief and pupil ray, ``summary`` the chief rays and each path's
    pupil sampling without its pupil rays (about 1 % of the size, for runs of hundreds of beams); the trace, the results
    and every hash are the same either way, and a query that needs pupil rays refuses a ``summary`` run
    (``PUPIL_RAYS_NOT_RECORDED``).

    ``monitor`` (``os_optics.run_monitor.RunMonitor``; default: the one installed by ``run_control.monitored``, if any):
    progress reports, cancellation and memory / run-file limits. A run it stops -- cancelled, at a limit, or refused
    before the trace because its estimate (``local/resources.py``) exceeds a limit -- raises ``RunStopped`` whose
    ``stopped`` is the run's ``StoppedRun`` record (execution ``CANCELLED``; every observation and target unevaluated);
    its ``run_stopped.json`` in ``output_dir`` holds that record. A monitored
    run interrupted by Ctrl-C (``KeyboardInterrupt``) stops the same way. The monitor never changes a number.

    ``output_dir`` follows the run directory rule (``local/artifacts.py``): a run that completes replaces the run there;
    an attempt that does not complete -- stopped, refused, failed before the run is moved into place -- leaves the
    directory as it found it (a stop adds its record beside an earlier run, which still verifies). A failure while the run
    is moved into place (a file held open elsewhere) leaves what a process killed there leaves: past the move's first
    step, no manifest, so no run that verifies, and the Engine's records of what is there. The first step removes the run
    files of each configuration that did not run this time, then the stop records, then the configurations' manifests,
    then the directory's own: stopped while those run files are removed, it leaves the directory's manifest and that
    configuration's without the files removed before (``project verify``: INCONSISTENT, naming them as missing); stopped
    while the configurations' manifests are removed, the directory's manifest beside a configuration whose manifest is
    gone, recorded by its copy (``configurations/<id>/.manifest.json.earlier``; ``project verify``: INCONSISTENT). Either
    way the records let any run then complete there. A stop or a Ctrl-C that lands once the directory's manifest is in
    place (while the move cleans up, or as the write ends) follows a complete run of this attempt: the run is returned
    with its artifacts, and no stop record is written.

    Outcomes: an Engine ``FAIL`` / ``UNKNOWN`` / ``OUT_OF_MODEL_SCOPE`` run is returned (``run.status``), not raised,
    and so is a run that leaves a published model's validity: it did when ``run.model_scope_issues`` is not empty or any
    Engine issue (``run.result.issues``) or target has status ``OUT_OF_MODEL_SCOPE``. The status is the worst of all
    outcomes, so a failing target makes such a run ``FAIL``. A Project that cannot be simulated as given raises ``LocalRunError`` whose ``problems``
    carry stable ``LocalRunErrorCode`` values. The simulation runs on this device; only the catalog may be
    synchronized (``os_optics.catalog``).
    """

    if ray_data not in RAY_DATA_LEVELS:
        raise ValueError(f"ray_data is one of {', '.join(RAY_DATA_LEVELS)}, not {ray_data!r}")
    from .evaluate import EvaluationSession

    return session_run(EvaluationSession(catalog=catalog, model_kind=model_kind, runtime_kind=runtime_kind), project,
                       output_dir=output_dir, view_settings=view_settings, ray_data=ray_data, monitor=monitor)


def session_run(session: Any, project: Any, *, lowered: Any = None, output_dir: Path | str | None = None,
                view_settings: OpticalViewSettingsV1 | None = None, ray_data: str = RAY_DATA_FULL,
                monitor: Any = None) -> LocalOpticalRun:
    """``run_optical_project`` through the ``EvaluationSession`` ``session`` (its catalogs, compiled elements, Engine build
    id and Engine state): the run of every configuration of a Project shares one (``local/configurations.py``).
    ``lowered``: ``project`` already lowered by ``session.lowering`` (every configuration is lowered before any is
    traced); its run uses that lowering."""

    monitor = monitor if monitor is not None else active_monitor()
    facts: dict[str, Any] = {}
    held: dict[str, Any] = {}  # (the run being written: what a stop after its move into place answers with)
    attempt = begin_attempt()
    try:
        with monitored(monitor):
            # (the complete composition of the evaluation's capabilities: os_optics.local.evaluate)
            from .evaluate import COMPLETE

            evaluation = session.evaluation(project, (COMPLETE,), view_settings=view_settings, ray_data=ray_data,
                                            writes=output_dir is not None, facts=facts, lowered=lowered)
            run = evaluation.run(ray_data=ray_data)
            if output_dir is not None:
                held["run"] = run
                run = replace(run, artifacts=run.write_artifacts(output_dir))
            return run
    except RunStopped as stopped:
        if (written := completed_write(held, output_dir)) is not None:
            return written
        raise stop_run(stopped, facts.get("document"), monitor, engine_input_hash=facts.get("engine_input_hash"),
                       output_dir=output_dir) from None
    except KeyboardInterrupt:
        if (written := completed_write(held, output_dir)) is not None:
            return written
        if monitor is None or not hasattr(monitor, "stopped_by_interrupt"):
            raise
        raise stop_run(monitor.stopped_by_interrupt(), facts.get("document"), monitor,
                       engine_input_hash=facts.get("engine_input_hash"), output_dir=output_dir) from None
    finally:
        end_attempt(attempt)


def completed_write(held: dict[str, Any], output_dir: Path | str | None) -> Any:
    """The run ``held["run"]`` with its artifacts where this attempt already moved it into ``output_dir`` (its manifest in
    place: ``artifacts.committed_run``), so a stop or a Ctrl-C that lands after that -- in the clean-up of the move, or as
    the write's phase ends -- answers with the completed run, not a stop; ``None`` otherwise."""

    artifacts = committed_run(output_dir)
    if artifacts is None or "run" not in held:
        return None
    return replace(held["run"], artifacts=artifacts)


def stopped_run_record(stopped: RunStopped, document: ProjectDocumentV2 | None, monitor: Any = None, *,
                       engine_input_hash: str | None = None) -> StoppedRun:
    """The ``StoppedRun`` record of a run ``stopped`` before it completed (``document``: its Project, when it was read)."""

    observations = tuple(o.observation_id for o in document.observations) if document is not None else ()
    targets = tuple(t.target_id for t in document.targets) if document is not None else ()
    progress = dict(monitor.summary()) if monitor is not None and hasattr(monitor, "summary") else {}
    if stopped.progress:
        progress["at_stop"] = stopped.progress
    return StoppedRun(
        project_hash=document.content_hash() if document is not None else "",
        engine_input_hash=engine_input_hash,
        outcome=ResultOutcome(
            execution=ExecutionRecord(outcome=ExecutionOutcome.CANCELLED, termination=TerminationReason.CANCELLED,
                                      reason=stopped.message, complete=False),
            requirements=RequirementRecord(declared=bool(targets))),
        stop=RunStop(cause=stopped.cause, message=stopped.message, phase=stopped.phase, phases=stopped.phases,
                     completed_phases=stopped.completed_phases, done=stopped.done, total=stopped.total, unit=stopped.unit,
                     limit_bytes=stopped.limit_bytes, measured_bytes=stopped.measured_bytes,
                     estimated_bytes=stopped.estimated_bytes),
        affected_observation_ids=observations, affected_target_ids=targets, progress=progress)


def stop_run(stopped: RunStopped, document: ProjectDocumentV2 | None, monitor: Any = None, *,
             engine_input_hash: str | None = None, output_dir: Path | str | None = None) -> RunStopped:
    """Give ``stopped`` its record (``stopped.stopped``, once) and write it into ``output_dir`` (``run_stopped.json``);
    returns ``stopped`` to raise. Nothing else in the directory changes: an earlier run there stays as it was."""

    if stopped.stopped is None:
        stopped.stopped = stopped_run_record(stopped, document, monitor, engine_input_hash=engine_input_hash)
    # (not beside a run this very attempt moved into place before the stop landed: that run is no earlier one)
    if output_dir is not None and committed_run(output_dir) is None:
        record_stopped_run(output_dir, stopped.stopped)
    return stopped


# ------------------------------------------------------------------------------------------------------------- stages
# The capabilities of a local run, one function each (os_optics.evaluation.capabilities): ``run_optical_project`` and an
# ``EvaluationSession`` evaluation (local/evaluate.py) compose these and nothing else.


def project_catalog(document: ProjectDocumentV2, catalog: Any, catalogs: dict[Any, Any] | None = None) -> Any:
    """The public catalog a Project's run uses: ``catalog`` when given, else the release its components pin (else the
    active one); ``None`` for a Project without catalog components and no ``catalog``. ``catalogs``: a dictionary the
    caller keeps, holding each catalog loaded (an ``EvaluationSession``'s): a catalog file (a path) by its resolved path
    and the SHA-256 of its bytes, read at each call (the file edited on disk is loaded, and its elements compiled, anew;
    the dictionary keeps the latest content of each file), another given catalog by the identity of the object given
    (another object is loaded, and its elements compiled, anew), and without a catalog the release this call resolves to
    (``catalog.resolved_release``: the release the parts pin when this runtime holds it, else the active one), checked at
    each call -- a release installed, or made active, after an earlier call is the one used, and the dictionary keeps only
    the release in force."""

    if catalog is None and not any(c.product is not None for c in document.components):
        return None
    source = catalog
    pin = pinned_catalog_version(document) if catalog is None else None
    if catalog is None:
        release = resolved_release(pin)
        key: tuple[Any, ...] = ("pinned", pin) if release is None else ("release", *release)
        if catalogs is not None:
            for stale in [k for k in catalogs if k[0] in ("release", "pinned") and k != key]:
                del catalogs[stale]  # (a release no longer in force)
    elif catalogs is not None and _is_catalog_path(catalog):
        source = _raw(catalog, "public optical catalog", LocalRunErrorCode.CATALOG_INVALID)  # (the bytes, read once per call)
        key = ("file", str(Path(catalog).resolve()), hashlib.sha256(source).hexdigest())
        for stale in [k for k in catalogs if k[:2] == key[:2] and k != key]:
            del catalogs[stale]  # (an earlier content of the same file)
    else:
        key = ("given", id(catalog))
    if catalogs is not None and key in catalogs:
        return catalogs[key][1]
    loaded = load_public_catalog(source, catalog_version=pin)
    if catalogs is not None:
        catalogs[key] = (catalog, loaded)  # (the given object is held, so its id names it while the session lives)
    return loaded


def _is_catalog_path(catalog: Any) -> bool:
    """Whether ``catalog`` names a file (``load_public_catalog`` reads it): a path, or a ``str`` that is not JSON text."""

    return isinstance(catalog, PurePath) or (isinstance(catalog, str) and not catalog.lstrip().startswith(("{", "[")))


def lowered_project(document: ProjectDocumentV2, catalog: Any, *, model_kind: str | None, check_solids: bool = True,
                    catalogs: dict[Any, Any] | None = None, compiled: dict[Any, Any] | None = None,
                    engine_state: Any = None) -> Any:
    """The Project lowered to its Engine request against its catalog (``project_catalog``) with the capability
    ``solid_check`` (refused on an ERROR finding when ``check_solids``): ``lower_project`` with the caller's ``catalogs``,
    ``compiled`` elements and ``engine_state``, outside any phase -- a lowering made before the runs that use it
    (``prepare_project(..., lowered=...)``)."""

    public_catalog = project_catalog(document, catalog, catalogs)  # a Project without catalog components needs none
    return lower_project(document, public_catalog, model_kind=model_kind, check_solids=check_solids, compiled=compiled,
                         engine_state=engine_state)


def prepare_project(document: ProjectDocumentV2, catalog: Any, *, model_kind: str | None, solids: bool = True,
                    catalogs: dict[Any, Any] | None = None, compiled: dict[Any, Any] | None = None,
                    lowered: Any = None, engine_state: Any = None, refuse_solids: bool = True) -> tuple[Any, Any]:
    """Phase ``lower``: the catalog read and the Project lowered to its Engine request (with the capability
    ``solid_check`` when ``solids``; an ERROR finding of it refuses the Project when ``refuse_solids``, else the report
    states it). Returns ``(catalog, LoweredProject)``. The lowering computes the catalog binding
    (``LoweredProject.simulated_parts``) and checks the catalog pin itself; that it equals ``resolve_simulated_parts`` --
    the function the server recomputes at intake -- is a property of the lowering, held by a test over the reference
    Projects, not checked again in every run. ``lowered``: the Project already lowered with the same ``catalogs``
    (``lowered_project``, before this phase: the configurations of a Project). ``engine_state``: the ``EvaluationState``
    of the evaluation that follows, whose Engine build a sample's excitation trace uses (``lower_project``)."""

    with phase("lower"):
        public_catalog = project_catalog(document, catalog, catalogs)  # a Project without catalog components needs none
        if public_catalog is not None:
            checkpoint()  # (the catalog is read and checked)
        if lowered is None:
            lowered = lower_project(document, public_catalog, model_kind=model_kind, inspect_solids=solids, compiled=compiled,
                                    engine_state=engine_state, check_solids=refuse_solids)
        checkpoint()  # (every component lowered and every optical body checked: checkpoints inside lower_project)
    return public_catalog, lowered


def checked_solids(document: ProjectDocumentV2, lowered: Any, *, refuse: bool = True) -> Any:
    """The capability ``solid_check`` of an already lowered Project (phase ``lower``): ``lowered`` with its optical-solid
    report, refused as ``lower_project`` refuses it when ``refuse``."""

    with phase("lower"):
        return replace(lowered, solids=solid_check(lowered.elements, document, refuse=refuse))


def require_valid_solids(lowered: Any) -> None:
    """Refuse (``LocalRunError`` carrying the whole report, as ``solid_check`` refuses) a lowered Project whose
    optical-solid report holds an ERROR finding: the Local Validation record is made only of a design whose optical
    bodies are valid solids."""

    solids = lowered.solids
    if solids is not None and not solids.valid:
        raise LocalRunError(solids.local_run_problems(), optical_solids=solids.as_dict())


def install_preflight(monitor: Any, document: ProjectDocumentV2, lowered: Any, *, ray_data: str,
                      view_settings: OpticalViewSettingsV1 | None, writes: bool) -> None:
    """With a monitor that refuses a run on its estimate: the estimate of the complete run (``local/resources.py``) checked
    before the trace, and its re-estimate installed for the end of the trace."""

    if monitor is None or not hasattr(monitor, "preflight"):
        return
    # (refused here when the run's estimate exceeds a memory or run-file limit, and again when the trace has ended:
    # local/resources.py; the memory once lowered -- the catalog read -- is read now)
    drawn = view_settings is None or view_settings.pupil_ray_selection.value != "NONE"
    in_memory = bool(getattr(monitor, "files_in_memory", False))
    base = monitor.memory_now() if hasattr(monitor, "memory_now") else getattr(monitor, "memory_bytes", None)
    monitor.preflight(estimate_run_resources(lowered.request, ray_data=ray_data, pupil_rays_drawn=drawn,
                                             base_memory_bytes=base, files_in_memory=in_memory))
    if hasattr(monitor, "reestimate"):
        request = lowered.request
        # (what the run writes for certain counts only when it writes run files: into output_dir, or held in memory; and it
        # is counted -- the Project and the request serialized once more -- only for a monitor that would refuse the run
        # on it: one with a run-file limit that refuses on its estimates)
        writes = writes or in_memory
        certain = bool(getattr(monitor, "refuse_on_estimate", True)) and getattr(monitor, "max_artifact_bytes", True) is not None
        monitor.reestimate = lambda paths, steps, memory: estimate_after_trace(
            request, paths=paths, steps=steps, ray_data=ray_data, pupil_rays_drawn=drawn, memory_bytes=memory,
            files_in_memory=in_memory, certain_files=certain,
            fixed_file_bytes=fixed_run_file_bytes(document, request) if writes and certain else None)


def traced_evaluation(request: ValidationRequest, *, capabilities: Any = ("complete",), exclude: Any = (), targets: Any = None,
                      state: Any = None, recorder: Any = None, ray_paths: bool = True) -> tuple[ValidationResult, RayPathTrace | None]:
    """The Engine evaluation of ``request`` (``os_optics.evaluate``) and, with a ``recorder`` (``ray_path_recorder``) and
    ``ray_paths``, the capability ``ray_paths``: the chief and pupil rays of its beam trace (phase ``ray_paths``). With a
    ``recorder`` and ``ray_paths`` false the trace is observed and its record left in the recorder, unread: a later
    evaluation over ``state`` that reads the ray paths passes the same recorder and does not trace again."""

    result = evaluate(request, capabilities=capabilities, exclude=exclude, targets=targets, state=state, recorder=recorder)
    nodes = quadrature_refusal(result)
    if nodes is not None:  # (a node count whose rule the machine cannot hold: the Project's setting, named)
        raise fail(LocalRunErrorCode.PROJECT_INVALID, str(QuadratureRuleTooLarge(nodes)))
    if recorder is None or not ray_paths:
        return result, None
    return recorded_ray_paths(request, result, recorder)


def scope_findings(lowered: Any, result: ValidationResult, ray_paths: RayPathTrace) -> tuple[ModelScopeIssue, ...]:
    """The capability ``model_scope`` (phase ``model_scope``): every use of a model outside its validity in the traced
    rays, each naming the targets, planes and elements it affects."""

    try:
        with phase("model_scope", total=len(result.paths), unit="paths"):
            return model_scope_issues(lowered, result, ray_paths)
    except LocalRunError:
        raise
    except (ArithmeticError, ValueError, RecursionError) as exc:  # numbers of an extreme magnitude
        raise _numerical_failure(exc) from None


def run_record(document: ProjectDocumentV2, lowered: Any, result: ValidationResult, scope_issues: tuple[ModelScopeIssue, ...], *,
               runtime_kind: LocalRuntimeKindV1 | str | None) -> LocalValidationRecordV1:
    """The capability ``record`` (phase ``record``): the Local Validation record of a complete run."""

    try:
        with phase("record"):
            return local_validation_record(document, lowered, result, runtime_kind=runtime_kind,
                                           outside_model_scope=bool(scope_issues), scope_issues=scope_issues)
    except LocalRunError:
        raise
    except (ArithmeticError, ValueError, RecursionError) as exc:  # numbers of an extreme magnitude
        raise _numerical_failure(exc) from None


def run_view(document: ProjectDocumentV2, catalog: Any, lowered: Any, result: ValidationResult, ray_paths: RayPathTrace, *,
             view_settings: OpticalViewSettingsV1 | None, model_kind: str | None) -> OpticalSimulationSceneV1:
    """The capability ``view`` (phase ``scene``): the Optical Simulation View scene of the evaluation."""

    try:
        with phase("scene", total=len(ray_paths.paths), unit="beam paths"):
            # (the result, the ray paths and the request are this evaluation's own objects: their hashes are compared as
            # stated, and the result is not serialized and hashed again)
            return build_optical_simulation_scene(project=document, catalog=catalog, engine_request=lowered.request,
                                                  engine_result=result, ray_paths=ray_paths, settings=view_settings,
                                                  model_kind=model_kind, same_evaluation=True)
    except OpticalViewBindingError as exc:
        raise fail(LocalRunErrorCode.OPTICAL_VIEW_BINDING_FAILED, str(exc)) from None
    except ValidationError as exc:  # defensive: a run the scene contract cannot express is refused, never raised raw
        raise fail(LocalRunErrorCode.OPTICAL_VIEW_BINDING_FAILED,
                   f"the Optical Simulation View cannot represent this run: {_validation_message(exc)}") from None
    except (ArithmeticError, ValueError, RecursionError) as exc:
        raise _numerical_failure(exc) from None
