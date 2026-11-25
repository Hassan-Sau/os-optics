"""``RunResults``: the scientific results of one local run, loaded from its run directory (or its files in memory)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import ValidationError

from ..config import NumericalConfig
from ..contracts.request import ValidationRequest
from ..contracts.results import ValidationResult
from ..optics.interference.source_coherence import request_source_coherence
from ..local.artifacts import (
    ENGINE_REQUEST_FILE,
    MANIFEST_FILE,
    PROJECT_FILE,
    RAY_DATA_FILE,
    RUN_FILES,
    SIMULATION_RESULT_FILE,
    STOPPED_FILE,
    later_attempt,
    run_artifact_files,
    verify_run_files,
)
from ..local.run import LocalOpticalRun
from ..validation.descriptors import metric_descriptors
from ..validation.ray_paths import RAY_DATA_FULL, TRACED, ray_data_recording
from . import export as formats
from .basis import SAMPLED_LINE_FIELDS, dump_with_bases, row_bases, scoped_interference, scoped_observation
from .detector import detector_response
from .handoff import handoff, mark_document, model_scope
from .irradiance import IRRADIANCE_MODEL, irradiance_map
from .quantity import MAX_UNTRACED_FRACTION
from .focal_field import focal_field
from .pupil_field import pupil_field
from .rays import (MIN_WAVEFRONT_RAYS, PUPIL_RAYS_NOT_RECORDED, PUPIL_RAYS_NOT_RECORDED_MESSAGE, WAVEFRONT_MODEL,
                   observation_crossings, ray_rows, wavefront)
from .tables import (
    COLUMNS,
    arrival_row,
    columns_of,
    exposure_rows,
    handoff_row,
    metric_rows,
    monitor_row,
    path_rows,
    power_budget_rows,
    power_rating_rows,
    grating_order_rows,
    nonlinear_rows,
    segment_rows,
    sampled_crossing_rows,
    sampled_light_rows,
    superposition_rows,
)

RESULTS_SCHEMA = "opensci.run_results"
RESULTS_SCHEMA_VERSION = "1.0"
EXPORT_KINDS = tuple(COLUMNS)
#: What a monitor's document says of each sampled line (its crossings' columns are the ``sampled-crossings`` export).
SAMPLED_SUMMARY_FIELDS = ("source_id", "spectral_index", "wavelength_nm", "emitted_W", "power_W", "collection_fraction", "sample_count",
                          "samples_traced", "half_sample_count", "collection_fraction_half_samples", "collection_fraction_difference",
                          "collection_fraction_sigma", "crossing_count", "counted_crossing_count")


def sampled_summary(line: Any) -> dict[str, Any]:
    """A sampled line's ``SAMPLED_SUMMARY_FIELDS``, with ``power_out_of_scope_at`` where light scattered out of a beam whose power
    is out of model scope makes its powers the chief ray's, not the light's, and the ``value_basis`` of its power and
    collection fractions (``SAMPLED_LINE_FIELDS``: a model-scope finding that reaches the line, on a plane as the results
    report it, ``scoped_observation``)."""

    return {**{k: getattr(line, k) for k in SAMPLED_SUMMARY_FIELDS},
            **({"power_out_of_scope_at": list(line.power_out_of_scope_at)} if line.power_out_of_scope_at else {}),
            **row_bases([(name, line, name) for name in SAMPLED_LINE_FIELDS])}
TABLE_FILTERS = {
    "monitors": (), "arrivals": ("monitor", "path"), "rays": ("monitor", "path", "crossing_index"), "paths": ("path",),
    "segments": ("path",), "power-budget": (), "superpositions": ("monitor",), "metrics": (),
    "wavefront": ("monitor", "path", "crossing_index", "reference", "center", "min_rays"),
    "irradiance": ("monitor", "wavelength_nm", "half_extent_mm", "samples", "source_id", "center_uv_mm", "roi", "max_untraced_fraction"),
    "sampled-light": ("monitor",), "sampled-crossings": ("monitor",),
    "handoff": ("monitor", "path", "crossing_index", "element"), "exposures": ("element",), "power-ratings": ("element",),
    "grating-orders": ("element",), "nonlinear": ("element",),
}

# Error codes of ResultsError.
RUN_NOT_READABLE = "RUN_NOT_READABLE"
#: The directory (or files) of a run of a Project with configurations, given where one run is read.
CONFIGURATIONS_RUN = "CONFIGURATIONS_RUN"
MONITOR_NOT_FOUND = "MONITOR_NOT_FOUND"
PATH_NOT_FOUND = "PATH_NOT_FOUND"
ELEMENT_NOT_FOUND = "ELEMENT_NOT_FOUND"
RAY_DATA_NOT_AVAILABLE = "RAY_DATA_NOT_AVAILABLE"
EXPORT_INVALID = "EXPORT_INVALID"
#: A run whose Project no longer lowers to the run's Engine input (the cavity analysis of a run).
RUN_INPUT_MISMATCH = "RUN_INPUT_MISMATCH"
#: A query option of the wrong type or outside its bound (the json_api code of an invalid results query).
RESULTS_QUERY_INVALID = "RESULTS_QUERY_INVALID"
#: A query of a part of an evaluation that was not computed (``os_optics.local.Evaluation.results``): the message names the
#: capability to request.
NOT_COMPUTED = "NOT_COMPUTED"
_READINGS = ("plane_readings",)
#: The handoff of a plane's beams: the readings, and each beam's validity -- the Gaussian truncations the power ledger books
#: and the issues of the plane bounds (``focal_field`` / ``pupil_field`` with ``HANDOFF_BEAMS`` read it).
_HANDOFF = ("plane_readings", "power_ledger", "plane_bounds")
#: The beam paths and their segments: the beam trace's, and those the rays of extended emitters and scattered light add
#: (the secondary rays do no work in a scene without either).
_PATHS = ("beam_trace", "secondary_rays")
#: Every capability each query and export kind reads (``os_optics.evaluation.capabilities``); a query of an evaluation that
#: did not compute one of them is refused with ``NOT_COMPUTED`` naming it. ``monitor(id)`` leaves out the parts of other
#: capabilities that did not run and names them (``MONITOR_SECTIONS``); ``irradiance_map`` without the plane bounds gives
#: the incoherent map and states the coherent one not computed (``os_optics.results.irradiance``).
QUERY_REQUIRES: dict[str, tuple[str, ...]] = {
    "targets": ("requirements",), "monitors": _READINGS, "monitor": _READINGS, "arrivals": _READINGS,
    "superpositions": _READINGS, "sampled_light": _READINGS, "sampled-light": _READINGS, "sampled-crossings": _READINGS,
    "irradiance_map": _READINGS, "irradiance": _READINGS, "detector_response": _READINGS,
    "handoff": _HANDOFF, "focal_field": _HANDOFF, "pupil_field": _HANDOFF,
    "imaging": _READINGS, "ghosts": _READINGS, "field": _READINGS, "sample_emission": _READINGS,
    "detection": ("plane_bounds",), "paths": _PATHS, "segments": _PATHS,
    "grating_orders": ("beam_trace",), "grating-orders": ("beam_trace",), "nonlinear": ("beam_trace",),
    "rays": ("ray_paths", "plane_readings"), "wavefront": ("ray_paths", "plane_readings"),
    "power_budget": ("power_ledger",), "power-budget": ("power_ledger",),
    "exposures": ("surface_exposures",), "power-ratings": ("surface_exposures",),
}
#: The sections of ``monitor(id)`` another capability than the plane readings produces: left out, and named under
#: ``not_computed``, when it did not run.
MONITOR_SECTIONS = {"interference": "beam_pairs", "targets": "requirements", "detection": "plane_bounds",
                    "polarization_mixtures": "power_ledger"}
#: What ``describe()["available"]`` counts, by the capabilities that produce it: ``None`` where one of them did not run
#: (the path list and the wavefront by the capabilities their queries read, ``QUERY_REQUIRES``).
AVAILABLE_REQUIRES: dict[str, tuple[str, ...]] = {
    "monitors": _READINGS, "arrivals": _READINGS, "outside_active_area_crossings": _READINGS,
    "superpositions": _READINGS, "paths": QUERY_REQUIRES["paths"], "two_beam_interference": ("beam_pairs",),
    "targets": ("requirements",), "power_budget": ("power_ledger",), "ray_data": ("ray_paths",),
    "ray_data_version": ("ray_paths",), "ray_data_recording": ("ray_paths",),
    "wavefront": QUERY_REQUIRES["wavefront"], "irradiance_map": _READINGS, "sampled_light": _READINGS,
    "sampled_sources": ("secondary_rays",), "exposures": ("surface_exposures",),
}
#: The smallest ``min_rays`` a wavefront takes: it needs at least one arrived pupil ray.
MIN_RAYS_LOWER_BOUND = 1

CONVENTIONS = {
    "design_frame": "Right-handed design frame in millimetres (+Z up). Positions (position_*, origin_*, start_* / end_*) and "
                    "directions (direction_x/y/z, unit vectors) are design-frame values.",
    "plane_frame": "Each observation plane (monitor) has origin_mm and unit axes u, v and normal n = u x v (monitors(): the "
                   "resolved design-frame pose; pose_frame DEVICE_LOCAL: the Project gave the pose relative to device_id). "
                   "u_mm = (point - origin) . u, v_mm = (point - origin) . v; direction_u/v/n are direction cosines on u, v, "
                   "n (direction_n < 0: travelling against the normal).",
    "absorbing": "absorbing_extent NONE: a transparent monitor (beams pass on); ACTIVE_AREA: absorbs exactly the beams that "
                 "count as arriving (chief ray inside the active area, or at least capture_min_power_fraction of the Gaussian "
                 "power inside it), other beams pass on and are listed under outside_active_area, except one whose chief ray "
                 "would cross the plane beyond the scene's reach along it (laboratory_reach_mm or the farthest surface, source "
                 "point or active area along the ray, whichever is farther): it is not listed; UNBOUNDED_PLANE: absorbing "
                 "without an aperture, the whole plane absorbs.",
    "crossing_index": "0 for a path's first crossing of a plane, 1 for its second (folded / double pass), ... A branch path "
                      "counts its own crossings from 0.",
    "gaussian_frame": "Gaussian quantities (*_x / *_y) refer to the beam's own transverse axes gaussian_e1 / gaussian_e2, not to "
                      "design or plane axes. waist_distance > 0: the waist lies upstream of the plane (diverging beam); "
                      "waist position = hit point - waist_distance x direction.",
    "polarization": "Jones vector (Ex, Ey) in the transported basis (e1, e2), e1 x e2 = propagation direction; field "
                    "E ~ exp[i(k.r - w t)]. Stokes s1..s3 normalized by S0; s3 = -1 right circular (clockwise viewed facing "
                    "the source); fully polarized light (degree of polarization 1: partial polarization is not modelled).",
    "first_order": "fo_d<out>_d<in>: the first-order ray-transfer matrix d(u, v, l_u, l_v) / d(x0, y0, a0, b0) from the source "
                   "plane to the crossing: x0, y0 parallel displacements of the chief ray at the source along the source frame "
                   "x / y (mm), a0, b0 tilts toward x / y (rad); u, v the hit point on the plane (mm), l_u, l_v the direction "
                   "cosines on u, v. Central differences of differential rays traced with the chief ray.",
    "power": "power_W of an arrival is the beam power reaching the plane (chief-ray packet model: Fresnel, coatings, absorption, "
             "Gaussian aperture truncation). The power budget lists every sink where power leaves the traced beams; power counted "
             "at a monitor is OBSERVED, not a sink (a transparent monitor does not consume power). residual = input - sum of "
             "sinks.",
    "optical_path": "optical_path_length_mm / opl_mm: sum of n x geometric length from the source (a branch includes its "
                    "ancestors). Every pupil ray starts on the source plane with 0. phase_rad is a plane-wave phase without the "
                    "Gaussian Gouy phase.",
    "rays": "rays(): the chief ray (ray_index 0) and the pupil sample rays (equal-power Gaussian rings) at each plane crossing, "
            "where the trace placed them; launch_offset_x/y label each pupil ray by its start offset at the source in units of "
            "the source's 1/e^2 radii.",
    "wavefront": WAVEFRONT_MODEL,
    "irradiance": IRRADIANCE_MODEL,
    "handoff": "handoff(monitor): the beams at an observation plane in its (u, v, n) frame -- q, waist, curvature, M^2, Jones vector, "
               "power, medium -- with the TEM00 field-construction convention and each beam's validity, and for a named element its "
               "clear apertures in the same frame. After GAUSSIAN_TRUNCATION (an aperture cut more than the run's "
               "gaussian_clip_warn_fraction) the power is exact while q, spot size and irradiance are those of the untruncated "
               "Gaussian (validity.beam_parameters TRUNCATED_GAUSSIAN); a smaller cut keeps TEM00 and is stated as "
               "validity.truncated_power_fraction and field_intensity_ripple.",
    "element_models": "element_models: what each element's simulation is (a catalog product's SKU or a declared model's "
                      "optical_model_id; model kind, fidelity, approximations; a compound unit's members). "
                      "SUBSTRATE_NOT_MODELLED / IDEAL_THIN_ELEMENT: one coated plane or a thin Jones element, so no plate "
                      "thickness, refraction, lateral shift or optical path through the substrate is in the numbers.",
    "phase": "phase_rad refers to each beam's own transported polarization basis: phases of two beams are not comparable "
             "(two parallel linear fields may have opposite basis vectors); use the pair results (phase_difference_rad) or "
             "superpositions for interference.",
    "irradiance_definitions": "Pair metrics *_irradiance_W_per_mm2 are beam irradiance (power per unit area perpendicular to "
                              "the beams); the irradiance map is power flux through the plane per unit plane area (smaller by "
                              "cos AOI).",
    "files": "Run files are UTF-8 JSON; command-line JSON output is ASCII (non-ASCII text escaped as JSON unicode escapes).",
    "metrics": "metrics() / 'os-optics results metrics': every target metric's unit, definition, sign convention, operand "
               "order, frame, selectors, model basis and validity.",
    "not_modelled": "Diffraction (beyond the Gaussian beam model), the Gouy phase in phase_rad, partial polarization and "
                    "depolarization, scattering, untraced ghost reflections (their power is listed as UNTRACED_REFLECTION) and "
                    "coherent sums beyond the paraxial Gaussian superposition.",
}


def not_computed_message(query: str, missing: Any, *, engine_result: bool,
                         kept: tuple[tuple[str, ...], tuple[str, ...]] | None = None) -> str:
    """The ``NOT_COMPUTED`` refusal of ``query``, naming the capabilities it reads that did not run and the call that
    computes them: ``os_optics.evaluate(request, capabilities=...)`` for the result of an Engine evaluation
    (``engine_result``), ``Evaluation.extend(outputs=...)`` for the results of a Project evaluation. ``kept``: the
    written exclusions that refuse it and ``exclude`` as the caller wrote it, when ``Evaluation.extend(outputs=missing)``
    is refused by exclusions the evaluation was made with -- the refusal then names each of them and a new evaluation
    without them."""

    names = ", ".join(repr(c) for c in missing)
    if kept is not None:
        exclusions, written = kept
        own = [c for c in exclusions if c in missing]
        others = [c for c in exclusions if c not in missing]
        listed = " and ".join(own)
        if others:
            needing = list(missing)  # (everything the query reads, as the text for one exclusion says it)
            listed += ((", and of " if own else "") + " and ".join(others)
                       + f", which {' and '.join(needing)} cannot be computed without")
        one = len(exclusions) == 1
        return (f"{query} reads {' and '.join(missing)}, which this evaluation did not compute: it was made with "
                f"exclude={', '.join(written)}, and Evaluation.extend keeps the {'exclusion' if one else 'exclusions'} of "
                f"{listed}. Evaluate the Project again without {'that exclusion' if one else 'those exclusions'}, with "
                f"outputs that include {names} "
                "(EvaluationSession.evaluate(project, outputs=[...]); in the same session it reuses the Engine products "
                "this one left)")
    way = (f"evaluate with capabilities that include {names} (os_optics.evaluate(request, capabilities=[{names}]))"
           if engine_result else f"evaluate with outputs that include {names} (Evaluation.extend(outputs=[{names}]))")
    return f"{query} reads {' and '.join(missing)}, which this evaluation did not compute: {way}"


class ResultsError(ValueError):
    """A results request that cannot be answered: ``code`` (stable) and ``message``."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message}


#: The code of a results request whose re-lowering of the run's Project the local runtime refuses (``RunProjectRefused``).
LOCAL_RUN_ERROR = "LOCAL_RUN_ERROR"


class RunProjectRefused(ResultsError):
    """A results request that lowers the run's Project again (the cavity analysis) and that the local runtime refuses here --
    for example a pinned catalog release that is not on this device and cannot be fetched: code ``LOCAL_RUN_ERROR``;
    ``refusal``: the ``LocalRunError``, whose problems keep their codes (``CATALOG_NOT_AVAILABLE``, ...)."""

    def __init__(self, refusal: Any) -> None:
        super().__init__(LOCAL_RUN_ERROR, str(refusal))
        self.refusal = refusal

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "problems": [p.as_dict() for p in self.refusal.problems]}


@dataclass(frozen=True)
class RunResults:
    """The scientific results of one Engine run (see ``os_optics.results``).

    ``result``: the Engine ``ValidationResult`` (``simulation_result.json``); ``ray_data``: the ray data document
    (``ray_data.json``; ``None`` when the run has none); ``manifest``: the run directory manifest (``None`` for a bare
    Engine result); ``numerical``: the run's ``NumericalConfig``; ``integrity_problems``: what ``verify_run_files`` found
    (empty = the run directory is consistent; ``None`` = not checked); ``request``: the Engine request document
    (``engine_request.json``; ``None`` for a bare Engine result)."""

    result: ValidationResult
    ray_data: dict[str, Any] | None = None
    manifest: dict[str, Any] | None = None
    numerical: NumericalConfig = field(default_factory=NumericalConfig)
    integrity_problems: tuple[str, ...] | None = None
    request: dict[str, Any] | None = None
    #: The Project that was run (``project.json``; ``None`` for a bare Engine result).
    project: dict[str, Any] | None = None
    #: The attempt that followed this run into its directory and did not complete (``local.artifacts.later_attempt``:
    #: from ``run_stopped.json`` beside the run); ``None``: there is none.
    later_attempt: dict[str, Any] | None = None
    #: What an evaluation held in memory states where a run directory's manifest would (``status`` with
    #: ``model_scope_issues`` when the model scope was computed, ``project``, ``catalog_version``, ``element_models``,
    #: ``retained_dofs``) and its ``evaluation`` scope; ``None`` for a run directory or a bare Engine result.
    recorded: dict[str, Any] | None = None
    #: The capabilities the evaluation computed (``os_optics.evaluation``); ``None``: every one a run computes.
    executed: tuple[str, ...] | None = None
    #: The Engine build id of the Engine that computes this object's analyses, when its holder knows it (an evaluation held
    #: in memory: its session's); ``None``: an analysis identifies the build itself (``os_optics.analysis.document``).
    engine_build_id: str | None = None

    # ------------------------------------------------------------------------------------------------- identity
    def identity(self) -> dict[str, Any]:
        """Engine version and build, input / result / trace hashes, Project hash and catalog release of the run.
        ``engine_input_hash`` is ``input_hash`` under the name the network export's provenance and ``check_project`` use."""

        m = self.result.manifest
        out: dict[str, Any] = {
            "engine_version": m.engine_version if m else None, "engine_build_id": m.engine_build_id if m else None,
            "input_hash": m.input_hash if m else None, "engine_input_hash": m.input_hash if m else None,
            "result_hash": m.result_hash if m else None,
            "trace_hash": m.trace_hash if m else None,
        }
        facts = self._facts()
        if facts:
            project = _object(facts.get("project"))
            out["project_hash"] = _object(project.get("project_hash")).get("digest")
            out["project_id"] = project.get("project_id")
            out["catalog_version"] = facts.get("catalog_version")
        return out

    @property
    def has_ray_data(self) -> bool:
        return self.ray_data is not None and self.ray_data.get("status") == TRACED

    def _observation(self, monitor: str):
        observation = next((o for o in self.result.observations if o.observation_id == monitor), None)
        if observation is None:
            known = [o.observation_id for o in self.result.observations]
            raise ResultsError(MONITOR_NOT_FOUND, f"no observation plane {monitor!r} in this run; monitors: {known}")
        return observation

    def _scoped_observations(self) -> list[Any]:
        """The run's planes as the results report them: each with the value basis the run's model-scope findings give its
        powers (``basis.scoped_observation``; the Engine result itself is unchanged)."""

        findings = self._scope_findings() or ()
        return [scoped_observation(o, findings) for o in self.result.observations]

    def _element(self, element: str | None) -> str | None:
        """``element`` once it names an optical element of the run (``ELEMENT_NOT_FOUND`` otherwise): checked against the
        exposure report when it was computed, else against the request's components with surfaces."""

        if element is None:
            return None
        if self.computed("surface_exposures") and self.result.exposures is not None:
            self.exposures(element)  # raises ELEMENT_NOT_FOUND
        elif self.request is not None:
            components = (self.request.get("project") or {}).get("components") or []
            known = [c.get("instance_id") for c in components if isinstance(c, dict) and c.get("surfaces")]
            if element not in known:
                raise ResultsError(ELEMENT_NOT_FOUND, f"no optical element {element!r} in this run; elements: {known}")
        return element

    def _check_path(self, path: str | None) -> None:
        if path is not None and path not in {p.path_id for p in self.result.paths}:
            raise ResultsError(PATH_NOT_FOUND, f"no beam path {path!r} in this run; paths: {[p.path_id for p in self.result.paths]}")

    def _require_ray_data(self) -> dict[str, Any]:
        if not self.has_ray_data:
            raise ResultsError(RAY_DATA_NOT_AVAILABLE, "this run carries no ray data (ray_data.json): run the Project again with "
                                                       "this Engine to get it")
        assert self.ray_data is not None
        if ray_data_recording(self.ray_data) != RAY_DATA_FULL:
            raise ResultsError(PUPIL_RAYS_NOT_RECORDED, PUPIL_RAYS_NOT_RECORDED_MESSAGE)
        return self.ray_data

    def _facts(self) -> dict[str, Any]:
        """The run directory's manifest, else what an in-memory evaluation states in its place."""

        return self.manifest if self.manifest is not None else self.recorded or {}

    def computed(self, capability: str) -> bool:
        """Whether the evaluation computed ``capability`` (a run directory: every capability its run computes)."""

        if self.executed is not None:
            return capability in self.executed
        scope = self.result.evaluation  # (a bare Engine result of a selective evaluation)
        return scope is None or capability not in {n.capability for n in scope.not_computed}

    def _evaluation_scope(self) -> dict[str, Any] | None:
        """The scope of an evaluation that did not compute everything (``None``: a complete run)."""

        recorded = (self.recorded or {}).get("evaluation")
        if isinstance(recorded, dict):
            return None if recorded.get("complete") else recorded
        scope = self.result.evaluation
        return None if scope is None else scope.model_dump(mode="json")

    def _need(self, query: str) -> None:
        """Refuse a query of a part the evaluation did not compute (``NOT_COMPUTED``, naming every capability it reads that
        did not run)."""

        missing = [c for c in QUERY_REQUIRES.get(query, ()) if not self.computed(c)]
        if missing:
            raise ResultsError(NOT_COMPUTED, not_computed_message(query, missing, engine_result=self.executed is None,
                                                                  kept=self._kept_exclusion(missing)))

    def _kept_exclusion(self, missing: list[str]) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
        """The exclusions that refuse ``Evaluation.extend(outputs=missing)``: every written exclusion that stands between
        the evaluation and them, in the words the caller wrote (a preset, not its member id), and ``exclude`` as the
        caller wrote it, asked of the plan ``extend`` makes; ``None``: ``extend`` computes them."""

        scope = _object((self.recorded or {}).get("evaluation"))
        written = tuple(scope.get("excluded") or ())
        if not written or self.request is None:
            return None
        from ..evaluation.capabilities import LOCAL, REQUIREMENTS
        from ..evaluation.plan import EXCLUDED_CAPABILITY_REQUIRED, EvaluationPlanError, plan_evaluation

        requested = tuple(scope.get("requested") or ())
        # (extend evaluates the targets this evaluation selected)
        targets = [t.target_id for t in self.result.targets] if REQUIREMENTS in (self.executed or ()) else None
        outputs = (*requested, *(c for c in missing if c not in requested))
        request = ValidationRequest.model_validate(self.request)
        remaining, kept = list(written), []
        for _ in written:  # (each pass takes out the written exclusion the plan meets first)
            try:
                plan_evaluation(outputs, exclude=remaining, targets=targets, request=request, level=LOCAL, what="outputs")
            except EvaluationPlanError as refused:
                name = str(refused.context.get("excluded_by", refused.context.get("capability")))
                if refused.code != EXCLUDED_CAPABILITY_REQUIRED or name not in remaining:
                    break
                kept.append(name)
                remaining.remove(name)
            else:
                break
        return (tuple(kept), written) if kept else None

    # ------------------------------------------------------------------------------------------------- queries
    def _scope_findings(self) -> list[dict[str, Any]] | None:
        """The model-scope findings the run recorded (``None`` when the evaluation did not compute the model scope)."""

        status = _object(self._facts().get("status"))
        if self.manifest is None and "model_scope_issues" not in status:
            return None
        return _objects(status.get("model_scope_issues"))

    def _scientific_summary(self):
        from ..local.record import scientific_projection
        return scientific_projection(self.result, self._scope_findings() or [])

    def describe(self) -> dict[str, Any]:
        """What this run is and which results it holds: identity, status, model scope, the result types present, the
        frame / sign conventions, where the metric descriptors are, the export kinds and the run files."""

        r = self.result
        scientific = self._scientific_summary()
        facts = self._facts()
        status: dict[str, Any] = {"engine_status": r.status.value}
        if facts:
            status.update({k: v for k, v in _object(facts.get("status")).items()})
            # (each target's assessment beside the manifest's diagnostic status: the record's, which counts applicability)
            assessed = {t.target_id: t.assessment for t in scientific.target_outcomes}
            if isinstance(status.get("targets"), list):
                status["targets"] = [{**row, "assessment": assessed.get(row.get("target_id"))} if isinstance(row, dict) else row
                                     for row in status["targets"]]
        pupil_rays = self.has_ray_data and ray_data_recording(self.ray_data) == RAY_DATA_FULL
        wavefront_ready = pupil_rays and any(a.ray_opl_defined for o in r.observations for a in o.arrivals)
        return {
            "schema": RESULTS_SCHEMA, "schema_version": RESULTS_SCHEMA_VERSION,
            "run": self.identity(), "status": status,
            "outcome": None if scientific.outcome is None else scientific.outcome.model_dump(mode="json"),
            "integrity": {"checked": self.integrity_problems is not None,
                          "consistent": None if self.integrity_problems is None else not self.integrity_problems,
                          "problems": list(self.integrity_problems or ())},
            # (present only beside a stop record: a later run into this directory stopped and left this run as it was)
            **({"later_attempt": dict(self.later_attempt)} if self.later_attempt is not None else {}),
            "issues": [{"code": i.code.value, "status": i.status.value, "message": i.message, "location": i.location} for i in r.issues],
            "available": self._available({
                "monitors": [o.observation_id for o in r.observations],
                "arrivals": sum(len(o.arrivals) for o in r.observations),
                "outside_active_area_crossings": sum(len(o.outside_active_area) for o in r.observations),
                "superpositions": sum(len(o.superpositions) for o in r.observations),
                "paths": [p.path_id for p in r.paths],
                "two_beam_interference": len(r.interference),
                "targets": len(r.targets),
                "power_budget": r.power_budget is not None,
                "ray_data": self.has_ray_data,
                "ray_data_version": self.ray_data.get("schema_version") if self.ray_data else None,
                "ray_data_recording": ray_data_recording(self.ray_data) if self.has_ray_data else None,
                "wavefront": wavefront_ready,
                "irradiance_map": any(o.lit for o in r.observations),
                "sampled_light": sum(len(o.sampled_light) for o in r.observations),
                "sampled_sources": [s.model_dump(mode="json") for s in r.sampled_sources],
                "source_spectra": [s.model_dump(mode="json") for s in r.source_spectra],
                "exposures": None if r.exposures is None else len(r.exposures.surfaces),
            }),
            "element_models": _objects(facts.get("element_models")),
            # each retained DOF with the limits the Project states for it (travel, hard limits, repeatability, drift)
            "retained_dofs": _objects(facts.get("retained_dofs")),
            # (an evaluation that did not compute everything: its scope, and the queries refused as NOT_COMPUTED)
            **({"evaluation": scope} if (scope := self._evaluation_scope()) is not None else {}),
            "conventions": dict(CONVENTIONS),
            "queries": ["describe()", "targets()", "monitors()", "monitor(id)", "paths()", "segments(path=None)", "power_budget()",
                        "rays(monitor=None, path=None, crossing_index=None)",
                        "wavefront(monitor, path, crossing_index=0, reference='plane'|'sphere', center=None)",
                        "irradiance_map(monitor, wavelength_nm=None, half_extent_mm=None, samples=None, source_id=None, max_untraced_fraction=0.1, "
                        "center_uv_mm=None, roi=None)", "sampled_light(monitor=None)",
                        "superpositions(monitor=None)", "handoff(monitor, path=None, crossing_index=None, element=None)",
                        "exposures(element=None)", "grating_orders(element=None)", "nonlinear(element=None)", "metrics()", "table(kind, ...)", "export(kind, fmt, ...)"],
            "exports": {"kinds": list(EXPORT_KINDS), "formats": {k: list(formats.FORMATS if k in formats.NPZ_KINDS else formats.FORMATS[:-1])
                                                               for k in EXPORT_KINDS}},
            "metric_descriptors": "metrics(): the meaning, unit, sign convention and operand order of every target metric",
            "files": [{k: f.get(k) for k in ("role", "name", "media_type", "bytes", "sha256", "description")}
                      for f in _objects((self.manifest or {}).get("files"))],
        }

    def _available(self, counts: dict[str, Any]) -> dict[str, Any]:
        """``describe()["available"]``: ``None`` for what a capability the evaluation did not compute would hold."""

        return {k: None if not all(self.computed(c) for c in AVAILABLE_REQUIRES.get(k, ())) else v for k, v in counts.items()}

    def targets(self) -> dict[str, Any]:
        """Every target outcome of the run in one compact document: per target its ``assessment`` (MET / UNMET /
        INDETERMINATE / NOT_APPLICABLE: the requirement outcome, which decides; the Local Validation record's, so a published
        model used outside its validity makes it INDETERMINATE), diagnostic ``status``, ``value`` and ``unit`` (full
        precision; ``None`` when not evaluated), the ``bounds`` it was judged against (``minimum`` / ``maximum`` /
        ``target_value`` with its resolved ``tolerance``), ``error`` (value minus target value), ``message`` and
        ``issue_codes``, the value's ``value_basis`` entry where the number is a bound or an estimate, and ``validity``
        where a check of the model the metric needs failed or stays open (a Gaussian beam past an accuracy limit: no value,
        the parameter, observed value, limit, its setting and where); ``counts`` per status and ``assessments`` per assessment; the run's ``outcome``,
        ``validation_status`` and ``engine_status``; and its ``model_scope_issues`` (``None`` for an Engine result without a
        run manifest).

        The run used a model outside its validity when ``model_scope_issues`` is not empty or any Engine issue or target
        has status ``OUT_OF_MODEL_SCOPE`` (a tabulated thin lens and a Faraday rotator report their domain as Engine issues,
        also listed in ``model_scope_issues``).
        ``out_of_model_issues`` lists all of them in one place -- each ``{source, code, message, ...}`` with ``source``
        ``ENGINE_ISSUE`` (plus ``location``), ``MODEL_SCOPE`` (plus ``instance_id``, ``surface_id`` / ``member_id``) or
        ``TARGET`` (plus ``target_id``), one entry per distinct source, code and place (the run's ``issues`` keep every
        message) -- and ``in_model_scope`` is ``False`` when that list is not empty (``None`` when it is empty but the
        run has no manifest, so the model-scope check is not available). The status is the worst of all outcomes, so a
        failing target makes such a run ``FAIL``: read ``in_model_scope``, not the status alone."""

        from ..local.record import target_row_bases

        self._need("targets")
        status = _object(self._facts().get("status"))
        scientific = self._scientific_summary()
        effective = {t.target_id: t for t in scientific.target_outcomes}
        findings = self._scope_findings() or ()
        rows = []
        for t in self.result.targets:
            bounds = {k: v for k, v in (("minimum", t.minimum), ("maximum", t.maximum), ("target_value", t.target_value),
                                        ("tolerance", t.tolerance)) if v is not None}
            row: dict[str, Any] = {"target_id": t.target_id, "metric": t.metric, "observation_id": t.observation_id, "status": t.status.value,
                                   "assessment": effective[t.target_id].assessment,
                                   "value": t.value, "unit": t.unit, "bounds": bounds, "error": t.error, "message": t.message,
                                   "issue_codes": sorted(set(t.issue_codes) | set(effective[t.target_id].applicability_issue_codes or ())),
                                   "applicability_issue_codes": list(effective[t.target_id].applicability_issue_codes or ()),
                                   **target_row_bases(t, findings),
                                   **({"validity": [v.model_dump(mode="json") for v in t.validity]} if t.validity else {})}
            if t.device_id is not None:
                row["device_id"] = t.device_id
            rows.append(row)
        counts: dict[str, int] = {}
        assessments: dict[str, int] = {}
        for row in rows:
            counts[row["status"]] = counts.get(row["status"], 0) + 1
            assessments[str(row["assessment"])] = assessments.get(str(row["assessment"]), 0) + 1
        scope = self._scope_findings()
        out_of_model: list[dict[str, Any]] = [
            {"source": "ENGINE_ISSUE", "code": i.code.value, "message": i.message, "location": i.location}
            for i in self.result.issues if i.status.value == "OUT_OF_MODEL_SCOPE"]
        out_of_model += [{"source": "MODEL_SCOPE", **issue} for issue in scope or ()]
        out_of_model += [{"source": "TARGET", "code": row["issue_codes"][0] if row["issue_codes"] else "OUT_OF_MODEL_SCOPE",
                          "message": row["message"], "target_id": row["target_id"]}
                         for row in rows if row["status"] == "OUT_OF_MODEL_SCOPE"]
        out_of_model = _distinct_findings(out_of_model)
        in_model_scope = False if out_of_model else (None if scope is None else True)
        return {"run": self.identity(), "validation_status": status.get("validation_status"), "engine_status": self.result.status.value,
                "outcome": None if scientific.outcome is None else scientific.outcome.model_dump(mode="json"),
                "in_model_scope": in_model_scope, "targets": rows, "counts": dict(sorted(counts.items())),
                "assessments": dict(sorted(assessments.items())),
                "model_scope_issues": scope, "out_of_model_issues": out_of_model,
                # (a selective evaluation: each declared target the caller did not select)
                **({"not_evaluated": [t.model_dump(mode="json") for t in self.result.evaluation.targets_not_evaluated]}
                   if self.result.evaluation is not None else {})}

    def monitors(self) -> list[dict[str, Any]]:
        """Every observation plane (monitor): its frame, active area, absorbing behaviour, arrivals and power."""

        self._need("monitors")
        out = []
        quantities = {q.observation_id: q for q in self._scientific_summary().observation_quantities or ()}
        for o in self._scoped_observations():
            out.append({
                "observation_id": o.observation_id, "status": o.status.value, "message": o.message,
                "quantity_state": _quantity_state(quantities[o.observation_id], o),
                "issue_codes": list(quantities[o.observation_id].issue_codes),
                "plane": o.plane.model_dump(mode="json") if o.plane else None,
                "arrival_count": len(o.arrivals), "total_power_W": o.total_power_W,
                **({"power_basis": o.power_basis} if o.power_basis is not None else {}),
                "power_complete": o.power_complete, "untraced_power_W": o.untraced_power_W,
                **({"power_upper_bound_W": o.power_upper_bound_W} if "power_upper_bound_W" in o.model_fields_set else {}),
                **({"power_lower_bound_W": o.power_lower_bound_W} if o.power_lower_bound_W is not None else {}),
                **({"power_sampling_resolution_W": o.power_sampling_resolution_W} if o.power_sampling_resolution_W is not None else {}),
                "power_by_wavelength": [dump_with_bases(s) for s in o.power_by_wavelength],
                "arrivals": [{"path_id": a.beam.path_id, "crossing_index": a.crossing_index, "source_id": a.beam.source_id,
                              "wavelength_nm": a.beam.wavelength_nm, "power_W": a.beam.power_W,
                              # (its power is its chief ray's, not the light's, behind these surfaces)
                              **({"power_out_of_scope_at": list(a.power_out_of_scope_at)} if a.power_out_of_scope_at else {}),
                              **row_bases([("power_W", a, "beam.power_W")])}
                             for a in o.arrivals],
                "sampled_light": [sampled_summary(line) for line in o.sampled_light],
                "outside_active_area_count": len(o.outside_active_area),
                "superposition_count": len(o.superpositions),
                **row_bases([("total_power_W", o, "total_power_W")]),
            })
        return out

    def monitor(self, monitor: str) -> dict[str, Any]:
        """One observation plane in full: frame, every arrival's complete beam state (Gaussian, polarization, first-order
        transfer, lineage), crossings outside the active area, power by wavelength, coherent superpositions, the partially
        polarized light of each route (``polarization_mixtures``, when a partially polarized source's light arrives), two-beam
        interference, the stated detector's reading (``detection``, as ``detection()`` gives it, when the plane states a
        detector) and the targets evaluated on it (each with the value basis ``targets()`` states). Of an evaluation that
        did not compute everything, the sections another capability produces (``MONITOR_SECTIONS``: the interference, the
        targets, the detection, the polarization mixtures) are left out when it did not run, and ``not_computed`` names
        each with that capability; the targets the evaluation did not select are listed under ``not_evaluated``."""

        from ..local.record import target_row_bases

        self._need("monitor")
        o = scoped_observation(self._observation(monitor), self._scope_findings() or ())
        scientific = self._scientific_summary()
        findings = self._scope_findings() or ()
        quantity = next(q for q in scientific.observation_quantities or () if q.observation_id == monitor)
        effective = {t.target_id: t for t in scientific.target_outcomes}
        missing = [(section, capability) for section, capability in MONITOR_SECTIONS.items() if not self.computed(capability)]
        doc = {
            "observation_id": o.observation_id, "status": o.status.value, "message": o.message,
            "quantity_state": _quantity_state(quantity, o), "issue_codes": list(quantity.issue_codes),
            "plane": o.plane.model_dump(mode="json") if o.plane else None, "total_power_W": o.total_power_W,
            **row_bases([("total_power_W", o, "total_power_W")]),
            **({"power_basis": o.power_basis} if o.power_basis is not None else {}),
            "power_complete": o.power_complete, "untraced_power_W": o.untraced_power_W,
                **({"power_upper_bound_W": o.power_upper_bound_W} if "power_upper_bound_W" in o.model_fields_set else {}),
                **({"power_lower_bound_W": o.power_lower_bound_W} if o.power_lower_bound_W is not None else {}),
                **({"power_sampling_resolution_W": o.power_sampling_resolution_W} if o.power_sampling_resolution_W is not None else {}),
            "power_by_wavelength": [dump_with_bases(s) for s in o.power_by_wavelength],
            "arrivals": [dump_with_bases(a, exclude={"relative_path_mm"}) for a in o.arrivals],
            "sampled_light": [sampled_summary(line) for line in o.sampled_light],
            "outside_active_area": [c.model_dump(mode="json") for c in o.outside_active_area],
            "superpositions": [dump_with_bases(s) for s in o.superpositions],
            **({"polarization_mixtures": [dump_with_bases(m) for m in o.polarization_mixtures]} if o.polarization_mixtures else {}),
            "interference": [dump_with_bases(scoped_interference(i, findings)) for i in self.result.interference
                             if i.observation_id == monitor],
            **({"detection": o.detection.model_dump(mode="json")} if o.detection is not None else {}),
            "targets": [{**t.model_dump(mode="json", exclude={"value_basis"}), "assessment": effective[t.target_id].assessment,
                         "applicability_issue_codes": list(effective[t.target_id].applicability_issue_codes or ()),
                         "issue_codes": sorted(set(t.issue_codes) | set(effective[t.target_id].applicability_issue_codes or ())),
                         **target_row_bases(t, findings)}
                        for t in self.result.targets if t.observation_id == monitor],
        }
        for section, _capability in missing:
            doc.pop(section, None)
        if missing:
            doc["not_computed"] = [{"section": section, "capability": capability} for section, capability in missing]
        scope = self.result.evaluation
        if scope is not None and self.computed("requirements"):
            targets = ((self.request or {}).get("project") or {}).get("targets") or []
            declared = {t.get("target_id"): t.get("observation_id") for t in targets if isinstance(t, dict)}
            unselected = [t.model_dump(mode="json") for t in scope.targets_not_evaluated if declared.get(t.target_id) == monitor]
            if unselected:
                doc["not_evaluated"] = unselected
        return doc

    def arrivals(self, monitor: str | None = None, path: str | None = None) -> list[dict[str, Any]]:
        """Flat arrival rows (export kind ``arrivals``), optionally for one monitor / one path."""

        self._need("arrivals")
        if monitor is not None:
            self._observation(monitor)
        self._check_path(path)
        return [arrival_row(o.observation_id, a) for o in self._scoped_observations() if monitor in (None, o.observation_id)
                for a in o.arrivals if path in (None, a.beam.path_id)]

    def paths(self, path: str | None = None) -> list[dict[str, Any]]:
        """The beam paths as a lineage tree: parent / children, branch surface and kind, termination, power at start and
        end, optical path length and the observation planes each crosses."""

        self._need("paths")
        self._check_path(path)
        crossings = observation_crossings(self.ray_data) if self.has_ray_data and self.ray_data else self._segment_crossings()
        return [row for row in path_rows(self.result, crossings) if path in (None, row["path_id"])]

    def _segment_crossings(self) -> dict[str, list[str]]:
        prefix = "observation/"
        return {p.path_id: [s.end_surface_id[len(prefix):] for s in p.segments if s.end_surface_id and s.end_surface_id.startswith(prefix)]
                for p in self.result.paths}

    def segments(self, path: str | None = None) -> list[dict[str, Any]]:
        """Every straight chief-ray segment of every path (or of one): start, end, medium, index, OPL, power, AOI."""

        self._need("segments")
        self._check_path(path)
        return segment_rows(self.result, path)

    def power_budget(self) -> dict[str, Any]:
        """The power ledger: per source spectral line the input power, every sink, the power observed at each monitor and the
        residual; per element the losses; every individual sink entry (``available`` False when the run has none)."""

        self._need("power_budget")
        budget = self.result.power_budget
        if budget is None:
            return {"available": False}
        return {"available": True, **budget.model_dump(mode="json"), "rows": power_budget_rows(self.result)}

    def exposures(self, element: str | None = None) -> dict[str, Any]:
        """The per-surface beam exposure of the run and its comparison with the elements' power ratings
        (``ValidationResult.exposures``: every beam meeting every element surface with its power, both Gaussian radii, AOI,
        footprint, peak irradiance, P / d and pulsed peak fluence; per surface the combined values, the groups of mutually
        coherent beams and the rating comparisons with their bounds, every condition's state and the applicability; per element the rating
        status and largest utilization; per source line the light left out as negligible), optionally for one element. ``available`` False for a run
        without it (a run of an earlier Engine)."""

        self._need("exposures")
        report = self.result.exposures
        if report is None:
            return {"available": False}
        doc = report.model_dump(mode="json")
        recorded_scope = _object(self._facts().get("status")).get("model_scope_issues")
        scope = _objects(recorded_scope)
        scoped = isinstance(recorded_scope, list) and all(isinstance(issue.get("affected_element_ids"), list) for issue in scope)
        def applicability(issues):
            return {"state": "OUT_OF_MODEL_SCOPE" if issues else "NO_RECORDED_LIMITATION" if scoped else "UNRECORDED",
                    "issue_codes": sorted({i["code"] for i in issues})}
        for row in [*doc["surfaces"], *doc["elements"]]:
            issues = [i for i in scope if row["element_id"] in (i.get("affected_element_ids") or ())]
            row["local_model_applicability"] = applicability(issues)
            row["model_scope_issues"] = issues
            for rating in row.get("ratings", ()):
                rating["local_model_applicability"] = applicability(issues)
        if element is not None:
            if element not in {e.element_id for e in report.elements}:
                raise ResultsError(ELEMENT_NOT_FOUND, f"no optical element {element!r} in this run; elements: "
                                                      f"{[e.element_id for e in report.elements]}")
            doc["surfaces"] = [s for s in doc["surfaces"] if s["element_id"] == element]
            doc["elements"] = [e for e in doc["elements"] if e["element_id"] == element]
        selected_scope = scope if element is None else [i for i in scope if element in (i.get("affected_element_ids") or ())]
        return {"available": True, **doc, "local_model_applicability": applicability(selected_scope), "model_scope_issues": selected_scope}

    def grating_orders(self, element: str | None = None) -> list[dict[str, Any]]:
        """Every diffraction order of every beam's pass through a grating (``GratingOrderResult``), optionally for one
        element: its direction, and the states of its power, polarization and phase."""

        self._need("grating_orders")
        element_id = self._element(element)
        return [row.model_dump(mode="json") for row in self.result.grating_orders if element_id in (None, row.element_id)]

    def nonlinear(self, element: str | None = None) -> list[dict[str, Any]]:
        """Every beam's pass into a crystal that generates its second harmonic (``NonlinearConversionResult``), optionally for
        one element: the harmonic power and efficiency, the phase mismatch, walk-off, the model's checks and the acceptance
        bandwidths."""

        self._need("nonlinear")
        element_id = self._element(element)
        return [row.model_dump(mode="json") for row in self.result.nonlinear_conversions if element_id in (None, row.element_id)]

    def superpositions(self, monitor: str | None = None) -> list[dict[str, Any]]:
        """Coherent sums of the beams on each plane, per wavelength (two or more beams): the value, or where a differing
        coating states no phase the range over every phase with ``phase_statement``."""

        self._need("superpositions")
        if monitor is not None:
            self._observation(monitor)
        return superposition_rows(self._scoped_observations(), monitor)

    def rays(self, monitor: str | None = None, path: str | None = None, crossing_index: int | None = None) -> list[dict[str, Any]]:
        """The chief and pupil rays at every observation-plane crossing (export kind ``rays``): plane u / v, global position
        and direction, plane direction cosines, OPL, launch offset and power weight."""

        self._need("rays")
        if monitor is not None:
            self._observation(monitor)
        self._check_path(path)
        crossing_index = None if crossing_index is None else _query_integer("crossing_index", crossing_index)
        return ray_rows(self.result, self._require_ray_data(), monitor=monitor, path=path, crossing_index=crossing_index)

    def wavefront(self, monitor: str, path: str, crossing_index: int = 0, reference: str = "plane", center: Any = None,
                  min_rays: int = MIN_WAVEFRONT_RAYS) -> dict[str, Any]:
        """The geometric wavefront (optical path difference per pupil ray, piston removed, RMS / PV) of one beam at one
        crossing of a monitor; ``status`` ``UNAVAILABLE`` with a ``reason`` when it cannot be computed. See
        ``os_optics.results.rays.wavefront``."""

        self._need("wavefront")
        self._observation(monitor)
        self._check_path(path)
        crossing_index = _query_integer("crossing_index", crossing_index)
        min_rays = _query_integer("min_rays", min_rays, minimum=MIN_RAYS_LOWER_BOUND)
        return wavefront(self.result, self.ray_data, monitor, path, crossing_index=crossing_index, reference=reference, center=center,
                         min_rays=min_rays)

    def irradiance_map(self, monitor: str, wavelength_nm: float | None = None, half_extent_mm: float | None = None,
                       samples: int | None = None, source_id: str | None = None, center_uv_mm: Any = None,
                       roi: dict[str, Any] | None = None, max_untraced_fraction: float | None = MAX_UNTRACED_FRACTION) -> dict[str, Any]:
        """Irradiance on a grid of the plane: the paraxial Gaussian superposition model (incoherent and coherent sums) for
        beams, a histogram with explicit bins (incoherent) for the sampled rays of extended emitters; ``roi`` adds the
        uniformity over a region. Without ``wavelength_nm`` on a plane several wavelengths reach: their sum.
        ``max_untraced_fraction``: the share of the map's light that may be missing from it above which a coherent map and a
        uniformity are withheld (``None``: no limit). A map of light a model-scope finding reaches lists it (``model_scope``)
        and carries its ``ESTIMATE`` on its maps and grid powers (``handoff.mark_document``). See
        ``os_optics.results.irradiance``."""

        self._need("irradiance_map")
        self._observation(monitor)
        doc = irradiance_map(self.result, monitor, cfg=self.numerical, wavelength_nm=wavelength_nm, half_extent_mm=half_extent_mm,
                             samples=samples, sources=request_source_coherence(self.request), source_id=source_id,
                             center_uv_mm=center_uv_mm, roi=roi, max_untraced_fraction=max_untraced_fraction)
        findings = self._scope_findings()
        if doc.get("status") != "OK" or not findings:
            return doc
        light = {(monitor, path_id, index) for path_id, index in zip(doc.get("path_ids") or (), doc.get("crossing_indices") or ())}
        lines = {(monitor, row["source_id"], row["spectral_index"]) for row in doc.get("sampled_lines") or ()}
        return mark_document(doc, model_scope(findings, monitor, light, lines))

    def detector_response(self, monitor: str, detector: Any, composition: str | None = None, **irradiance: Any) -> dict[str, Any]:
        """The response of a stated detector to this plane's irradiance map (``irradiance``: the arguments of ``irradiance_map``). A
        parameter it does not state leaves what needs it ``UNAVAILABLE``. See ``os_optics.results.detector``."""

        self._need("detector_response")
        return detector_response(self.irradiance_map(monitor, **irradiance), detector, composition)

    def pupil_field(self, monitor: str, request: Any, mask_files: Any = None) -> dict[str, Any]:
        """The scalar low-NA focal field of ``request`` (``PupilFieldRequestV1``), whose ``HANDOFF_BEAMS`` illumination reads the beams this
        run delivers to the plane ``monitor``. See ``os_optics.results.pupil_field``."""

        self._need("pupil_field")
        return pupil_field(request, mask_files, self.handoff(monitor))

    def focal_field(self, monitor: str, request: Any) -> Any:
        """The vector focal field of an ideal aplanatic objective (``FocalFieldRequestV1``) whose ``HANDOFF_BEAMS`` illumination reads
        the beams this run delivers to the plane ``monitor``: an ``AnalysisResultV1`` bound to this run's identity. See
        ``os_optics.results.focal_field``."""

        self._need("focal_field")
        return focal_field(request, self.handoff(monitor), run_identity=self.identity(), config=self.numerical,
                           engine_build_id=self.engine_build_id)

    def imaging(self, monitor: str, request: Any = None) -> Any:
        """The imaging analysis of the plane ``monitor`` (``ImagingRequestV1``; ``None``: the defaults): point spread function, MTF and
        encircled energy per field from a dense trace of the run's own request, an ``AnalysisResultV1`` bound to this run. See
        ``os_optics.results.imaging``."""

        self._need("imaging")
        from .imaging import imaging

        return imaging(self, monitor, request)

    def field(self, monitor: str, request: Any = None) -> Any:
        """The coherent field of the run's sampled-field or speckle source at the plane ``monitor`` (``FieldRequestV1``; ``None``:
        the defaults): its propagation by the angular spectrum or the paraxial transfer, irradiance maps, power, contrast and
        their convergence over realizations; an ``AnalysisResultV1`` bound to this run. See ``os_optics.results.field``."""

        self._need("field")
        from .field import field

        return field(self, monitor, request)

    def ghosts(self, monitor: str, request: Any = None) -> Any:
        """The ghost paths of the plane ``monitor`` (``GhostRequestV1``; ``None``: the defaults): every ghost reflection of the
        transmitting interfaces up to the bounce order, where it lands and how bright it is against the primary image, with a
        proven bound on every ghost not listed; an ``AnalysisResultV1`` bound to this run. See ``os_optics.results.ghosts``."""

        self._need("ghosts")
        from .ghosts import ghosts

        return ghosts(self, monitor, request)

    def detection(self, monitor: str | None = None) -> list[dict[str, Any]]:
        """Per observation plane that states a detector (``Observation.detection``): the light it reads, each source at each
        wavelength with its responsivity and modulated share, and every quantity -- photocurrent, shot / dark / thermal noise,
        ``detector_snr``, and with a lock-in its noise-equivalent bandwidth, demodulated signal, noise and ``lock_in_snr`` --
        with its status (``MODELED``, ``EXTERNALLY_CHARACTERIZED``, or ``UNAVAILABLE`` and why), and the model's conditions
        (``os_optics.validation.detection``). ``monitor``: only that plane (``MONITOR_NOT_FOUND`` for an unknown one; an empty list
        when it states no detector). A model-scope finding on the light the detector reads is carried in each quantity that
        reads that light (``value_basis["value"]``: an ``ESTIMATE`` with its checks), not in ``power_state``, which is the
        plane's power state as the Engine computed it."""

        self._need("detection")
        findings = self._scope_findings() or ()
        observations = [self._observation(monitor)] if monitor is not None else list(self.result.observations)
        return [{"observation_id": o.observation_id, **o.detection.model_dump(mode="json")}
                for o in (scoped_observation(plane, findings) for plane in observations) if o.detection is not None]

    def sample_emission(self) -> list[dict[str, Any]]:
        """Per FLUORESCENCE / RAMAN sample: how its emission follows from its excitation -- the excitation plane and wavelength, the
        power delivered there (and by which sources), the absorbed share and quantum yield or the Raman conversion, and per line
        its photons per excitation photon, energy ratio, photon rate and power -- as the Engine input records it
        (``contracts/detection.py`` ``SampleEmission``), with the emitter's collection at each plane (``sampled_light``). Empty
        for a run without samples, or a bare Engine result without its request."""

        self._need("sample_emission")
        components = ((self.request or {}).get("project") or {}).get("components") or []
        out = []
        for component in components:
            record = component.get("sample_emission") if isinstance(component, dict) else None
            if record is None:
                continue
            source_id = component.get("instance_id")
            collected = [row for row in sampled_light_rows(self._scoped_observations(), None) if row.get("source_id") == source_id]
            out.append({"source_id": source_id, **record, "traced": "emitter" in component, "collection": collected})
        return out

    def cavity(self, request: Any, catalog: Any = None) -> Any:
        """The cavity analysis (``os_optics.local.cavity``) of the Project this run ran: its stability, eigenmode, resonances and, with
        ``injection``, the resonant build-up of injected light, with the request's targets and the Project's cavity targets assessed;
        an ``AnalysisResultV1`` bound to this run (its Engine input hash must be the run's). Raises ``ResultsError``
        (``RUN_NOT_READABLE``: no Project in the run; ``RUN_INPUT_MISMATCH``: the Project lowers to another Engine input here, e.g.
        under another catalog release; ``CAVITY_ANALYSIS_INVALID``: a refused request, with the problems in the message) and
        ``RunProjectRefused`` (``LOCAL_RUN_ERROR``) where the runtime cannot lower the run's Project here, e.g. a pinned catalog
        release that is not on this device (``CATALOG_NOT_AVAILABLE``: ``os-optics catalog sync`` or ``catalog import``)."""

        from ..local.cavity import CavityError, cavity_analysis
        from ..local.errors import LocalRunError

        if self.project is None:
            raise ResultsError(RUN_NOT_READABLE, f"this run carries no {PROJECT_FILE}: a cavity analysis reads the Project that was run")
        try:
            analysis = cavity_analysis(self.project, request, catalog=catalog, run_result_hash=self.identity()["result_hash"],
                                       engine_build_id=self.engine_build_id)
        except CavityError as exc:
            raise ResultsError(exc.as_dict()["code"], "; ".join(f"{p['code']}: {p['message']}" for p in exc.problems)) from None
        except LocalRunError as exc:
            raise RunProjectRefused(exc) from None
        expected = self.identity()["input_hash"]
        if expected is not None and analysis.identity.physical.engine_input_hash != expected:
            raise ResultsError(RUN_INPUT_MISMATCH, f"the run's Project lowers here to the Engine input {analysis.identity.physical.engine_input_hash}, "
                                                   f"not the run's {expected} (another catalog release or Engine): run the Project again, or "
                                                   "analyse it with 'os-optics project cavity'")
        return analysis

    def sampled_light(self, monitor: str | None = None) -> list[dict[str, Any]]:
        """Per plane and sampled source line: the emitted power, the collection fraction with the same fraction from the
        first half of the rays, and the crossing counts (``sampled-crossings`` exports the rows)."""

        self._need("sampled_light")
        return sampled_light_rows(self._scoped_observations(), monitor)

    def handoff(self, monitor: str, path: str | None = None, crossing_index: int | None = None,
                element: str | None = None) -> dict[str, Any]:
        """The transverse-plane handoff of an observation plane (``os_optics.results.handoff``): the arriving beams' Engine
        state in the plane's (u, v, n) frame with the TEM00 field-construction convention and each beam's validity, and with
        ``element`` (a component instance id) that element's clear apertures in the same frame."""

        self._need("handoff")
        self._observation(monitor)
        self._check_path(path)
        crossing_index = None if crossing_index is None else _query_integer("crossing_index", crossing_index)
        if element is not None and not isinstance(element, str):
            raise ResultsError(RESULTS_QUERY_INVALID, f"element is a component instance id, not {element!r}")
        return handoff(self.result, self.request, monitor, path=path, crossing_index=crossing_index, element=element, cfg=self.numerical,
                       findings=self._scope_findings() or ())

    def metrics(self, include_stage_metrics: bool = False) -> list[dict[str, Any]]:
        """Every public target metric's descriptor: unit, definition, sign convention, operands, frame, selectors, model
        basis, validity and version."""

        return [d.model_dump(mode="json") for d in metric_descriptors(include_stage_metrics=include_stage_metrics)]

    # ------------------------------------------------------------------------------------------------- tables / export
    def columns(self, kind: str) -> list[dict[str, str]]:
        """The columns of an export kind: name, unit and description, in order."""

        if kind not in COLUMNS:
            raise ResultsError(EXPORT_INVALID, f"unknown export kind {kind!r}; kinds: {list(EXPORT_KINDS)}")
        return columns_of(kind)

    def _derived(self, kind: str, filters: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if kind == "wavefront":
            if filters.get("monitor") is None or filters.get("path") is None:
                raise ResultsError(EXPORT_INVALID, "a wavefront export needs a monitor and a path")
            # An absent filter takes its default; a given one is passed as given (an explicit 0 is not "absent").
            doc = self.wavefront(filters["monitor"], filters["path"], crossing_index=_given(filters, "crossing_index", 0),
                                 reference=_given(filters, "reference", "plane"), center=filters.get("center"),
                                 min_rays=_given(filters, "min_rays", MIN_WAVEFRONT_RAYS))
            summary_keys = ("observation_id", "path_id", "crossing_index", "wavelength_nm", "medium_index", "reference", "ray_count",
                            "pupil_ray_count", "piston_waves", "rms_waves", "pv_waves", "pupil_coordinates", "model")
        else:
            if filters.get("monitor") is None:
                raise ResultsError(EXPORT_INVALID, "an irradiance export needs a monitor")
            doc = self.irradiance_map(filters["monitor"], wavelength_nm=filters.get("wavelength_nm"),
                                      half_extent_mm=filters.get("half_extent_mm"), samples=filters.get("samples"),
                                      source_id=filters.get("source_id"), center_uv_mm=filters.get("center_uv_mm"), roi=filters.get("roi"),
                                      **({"max_untraced_fraction": filters["max_untraced_fraction"]} if "max_untraced_fraction" in filters else {}))
            summary_keys = ("observation_id", "wavelength_nm", "path_ids", "crossing_indices", "coherent_reason",
                            "grid_power_incoherent_W", "grid_power_coherent_W", "model")
            summary_keys += tuple(k for k in ("composition", "sampled_lines", "power_outside_grid_W", "uniformity", "map_complete", "scope",
                                              "outside_scope", "omitted_beams", "excluded_beams", "omitted_sampled_power_W", "untraced_power_W",
                                              "untraced_fraction", "max_untraced_fraction", "interference_open", "needs", "sampling",
                                              "wavelengths_nm", "per_wavelength", "composition_over_wavelengths") if k in doc)
        if doc["status"] != "OK":
            raise ResultsError(doc["reason"], doc["message"])
        summary = {k: doc[k] for k in summary_keys}
        if kind == "wavefront":
            return doc["rays"], summary
        summary["grid"] = {k: doc["grid"][k] for k in ("shape", "order", "bin_area_mm2", "u_edges_mm", "v_edges_mm") if k in doc["grid"]}
        rows = []
        coherent = doc["coherent_W_per_mm2"]
        counts = doc.get("sample_counts")
        for j, v in enumerate(doc["grid"]["v_mm"]):
            for i, u in enumerate(doc["grid"]["u_mm"]):
                rows.append({"u_mm": u, "v_mm": v, "irradiance_coherent_W_per_mm2": coherent[j][i] if coherent is not None else None,
                             "irradiance_incoherent_W_per_mm2": doc["incoherent_W_per_mm2"][j][i],
                             "sample_count": None if counts is None else counts[j][i]})
        return rows, {**summary, "_grid": doc}

    def _monitor_rows(self) -> list[dict[str, Any]]:
        """The monitors export's rows, each plane's quantity state as ``monitors()`` states it."""

        quantities = {q.observation_id: q for q in self._scientific_summary().observation_quantities or ()}
        return [monitor_row(o, _quantity_state(quantities[o.observation_id], o)) for o in self._scoped_observations()]

    def table(self, kind: str, **filters: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """``(rows, summary)`` of an export kind (``EXPORT_KINDS``) with its filters (``TABLE_FILTERS``)."""

        if kind not in COLUMNS:
            raise ResultsError(EXPORT_INVALID, f"unknown export kind {kind!r}; kinds: {list(EXPORT_KINDS)}")
        unknown = sorted(k for k, v in filters.items() if v is not None and k not in TABLE_FILTERS[kind])
        if unknown:
            raise ResultsError(EXPORT_INVALID, f"export kind {kind!r} takes the filters {list(TABLE_FILTERS[kind])}, not {unknown}")
        self._need(kind)
        if kind in ("wavefront", "irradiance"):
            return self._derived(kind, filters)
        if kind == "handoff":
            if filters.get("monitor") is None:
                raise ResultsError(EXPORT_INVALID, "a handoff export needs a monitor")
            doc = self.handoff(filters["monitor"], filters.get("path"), filters.get("crossing_index"), filters.get("element"))
            summary = {k: doc[k] for k in ("schema", "schema_version", "observation_id", "plane", "element", "conventions", "scope")}
            return [handoff_row(doc["observation_id"], beam) for beam in doc["beams"]], summary
        monitor, path = filters.get("monitor"), filters.get("path")
        rows = {
            "monitors": lambda: self._monitor_rows(),
            "arrivals": lambda: self.arrivals(monitor, path),
            "rays": lambda: self.rays(monitor, path, filters.get("crossing_index")),
            "paths": lambda: self.paths(path),
            "segments": lambda: self.segments(path),
            "power-budget": lambda: power_budget_rows(self.result),
            "superpositions": lambda: self.superpositions(monitor),
            "sampled-light": lambda: sampled_light_rows(self._scoped_observations(), monitor),
            "sampled-crossings": lambda: sampled_crossing_rows(self._scoped_observations(), monitor),
            "metrics": lambda: metric_rows(metric_descriptors()),
            "exposures": lambda: exposure_rows(self.result, self._element(filters.get("element"))),
            "power-ratings": lambda: power_rating_rows(self.result, self._element(filters.get("element"))),
            "grating-orders": lambda: grating_order_rows(self.result, self._element(filters.get("element"))),
            "nonlinear": lambda: nonlinear_rows(self.result, self._element(filters.get("element"))),
        }[kind]()
        return rows, {}

    def export(self, kind: str, fmt: str = "csv", **filters: Any) -> bytes:
        """The export of one kind in one format (``csv``, ``jsonl``, ``json``; ``npz`` for ``irradiance``) as bytes.
        ``filters``: ``monitor``, ``path``, ``crossing_index`` (rays / wavefront), ``reference`` / ``center`` / ``min_rays``
        (wavefront), ``wavelength_nm`` / ``half_extent_mm`` / ``samples`` / ``source_id`` / ``center_uv_mm`` / ``roi``
        (irradiance); see ``TABLE_FILTERS``."""

        if fmt not in formats.FORMATS or (fmt == "npz" and kind not in formats.NPZ_KINDS):
            allowed = list(formats.FORMATS) if kind in formats.NPZ_KINDS else list(formats.FORMATS[:-1])
            raise ResultsError(EXPORT_INVALID, f"export kind {kind!r} is written as {allowed}, not {fmt!r}")
        rows, summary = self.table(kind, **filters)
        grid = summary.pop("_grid", None)
        if fmt == "csv":
            return formats.csv_bytes(kind, rows)
        if fmt == "jsonl":
            return formats.jsonl_bytes(rows)
        if fmt == "npz":
            assert grid is not None
            arrays = {"u_mm": np.array(grid["grid"]["u_mm"]), "v_mm": np.array(grid["grid"]["v_mm"]),
                      "irradiance_incoherent_W_per_mm2": np.array(grid["incoherent_W_per_mm2"])}
            if grid["coherent_W_per_mm2"] is not None:
                arrays["irradiance_coherent_W_per_mm2"] = np.array(grid["coherent_W_per_mm2"])
            if "sample_counts" in grid:
                arrays["sample_counts"] = np.array(grid["sample_counts"])
                arrays["u_edges_mm"], arrays["v_edges_mm"] = np.array(grid["grid"]["u_edges_mm"]), np.array(grid["grid"]["v_edges_mm"])
            return formats.npz_bytes(arrays)
        used = {k: v for k, v in filters.items() if v is not None}
        if "center" in used and not isinstance(used["center"], str):
            used["center"] = [float(c) for c in used["center"]]
        run = {k: v for k, v in self.identity().items() if k in ("engine_version", "input_hash", "result_hash", "trace_hash", "project_hash")}
        return formats.json_bytes(formats.json_document(kind, rows, run=run, filters=used, summary=summary))


# ------------------------------------------------------------------------------------------------------------- loading


def _quantity_state(quantity: Any, observation: Any) -> str | None:
    """A plane's quantity state as the results report it: the Local Validation projection's (``OUT_OF_MODEL_SCOPE`` under a
    model-scope finding), else the Engine's own -- ``NOT_COMPUTED`` where the evaluation did not compute the plane bounds."""

    if quantity.quantity_state is not None:
        return quantity.quantity_state
    return None if observation.quantity_state is None else observation.quantity_state.value


def _query_integer(name: str, value: Any, *, minimum: int | None = None) -> int:
    """``value`` as an int when it is a JSON integer (a number without a fractional part, never a boolean) not below
    ``minimum``; otherwise ``ResultsError`` ``RESULTS_QUERY_INVALID`` (never rounded, floored or read as 0 / 1)."""

    integral = not isinstance(value, bool) and (isinstance(value, int) or (isinstance(value, float) and value.is_integer()))
    if not integral:
        raise ResultsError(RESULTS_QUERY_INVALID, f"{name} must be an integer, not {value!r}")
    if minimum is not None and value < minimum:
        raise ResultsError(RESULTS_QUERY_INVALID, f"{name} must be at least {minimum}, not {value!r}")
    return int(value)


def _given(filters: Mapping[str, Any], name: str, default: Any) -> Any:
    """The filter's value, or ``default`` only when it is absent (``None``): ``samples=0`` stays 0 and is refused."""

    value = filters.get(name)
    return default if value is None else value


def _parse_json(data: bytes | None, name: str) -> Any:
    if data is None:
        return None
    try:
        return json.loads(data)
    except ValueError as exc:
        raise ResultsError(RUN_NOT_READABLE, f"{name}: not JSON ({exc})") from None
    except RecursionError:
        raise ResultsError(RUN_NOT_READABLE, f"{name}: nested too deeply to be a run file") from None


def _object(value: Any) -> dict[str, Any]:
    """A manifest member that should be a JSON object (a run directory that does not verify may hold anything)."""

    return value if isinstance(value, dict) else {}


def _objects(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


_FINDING_PLACE = ("location", "instance_id", "surface_id", "member_id", "target_id")


def _distinct_findings(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The first entry of each distinct ``source``, ``code`` and place (an Engine issue reported twice at one location)."""

    seen: set[str] = set()
    out = []
    for finding in findings:
        key = json.dumps([finding.get("source"), finding.get("code"), *(finding.get(k) for k in _FINDING_PLACE)],
                         default=str)
        if key not in seen:
            seen.add(key)
            out.append(finding)
    return out


def _refuse_configurations(manifest_bytes: bytes | None) -> None:
    """The top of a run of a Project with configurations holds no run: name the configurations' run directories."""

    from ..local.configurations import is_configurations_manifest

    try:
        manifest = None if manifest_bytes is None else json.loads(manifest_bytes)
    except (ValueError, RecursionError):
        return
    if is_configurations_manifest(manifest):
        directories = [str(e.get("directory")) for e in manifest.get("configurations") or [] if isinstance(e, dict)
                       and e.get("manifest_sha256") is not None]
        raise ResultsError(CONFIGURATIONS_RUN, "a run of a Project with configurations: each configuration's results are in its "
                                               f"own run directory ({', '.join(directories) or 'none was written'}); "
                                               "'os-optics results configurations' compares them")


def results_from_files(files: Mapping[str, bytes], *, verify: bool = False) -> RunResults:
    """``RunResults`` of a run directory held in memory (``{file name: bytes}``: ``json_api.run_project``'s ``files``).
    ``verify``: also check the files' hashes and cross-references (``describe()["integrity"]``)."""

    def get(name: str) -> bytes | None:
        data = files.get(name)
        return None if data is None else bytes(data)

    raw_result = get(SIMULATION_RESULT_FILE)
    if raw_result is None:
        _refuse_configurations(get(MANIFEST_FILE))
        raise ResultsError(RUN_NOT_READABLE, f"the run has no {SIMULATION_RESULT_FILE}")
    try:
        result = ValidationResult.model_validate(_parse_json(raw_result, SIMULATION_RESULT_FILE))
    except ValidationError as exc:
        raise ResultsError(RUN_NOT_READABLE, f"{SIMULATION_RESULT_FILE}: not an Engine ValidationResult ({exc.error_count()} errors)") from None
    numerical = NumericalConfig()
    request_doc = _parse_json(get(ENGINE_REQUEST_FILE), ENGINE_REQUEST_FILE)
    if request_doc is not None:
        try:
            numerical = ValidationRequest.model_validate(request_doc).configuration.numerical
        except ValidationError:
            numerical = NumericalConfig()
    ray_data = _parse_json(get(RAY_DATA_FILE), RAY_DATA_FILE)
    manifest = _parse_json(get(MANIFEST_FILE), MANIFEST_FILE)
    project = _parse_json(get(PROJECT_FILE), PROJECT_FILE)
    problems = tuple(verify_run_files(files)) if verify else None

    def read(name: str) -> bytes:
        data = get(name)
        if data is None:
            raise FileNotFoundError(name)
        return data

    return RunResults(result=result, ray_data=ray_data if isinstance(ray_data, dict) else None,
                      manifest=manifest if isinstance(manifest, dict) else None, numerical=numerical, integrity_problems=problems,
                      request=request_doc if isinstance(request_doc, dict) else None, project=project if isinstance(project, dict) else None,
                      later_attempt=later_attempt(read))


RUN_FILE_NAMES = (MANIFEST_FILE, *(name for _role, name, _media, _what in RUN_FILES))


def load_run(run: Any, *, verify: bool = False) -> RunResults:
    """Load the scientific results of a run: a run directory (``str`` / ``pathlib.Path``, written by ``project run --out-dir``
    or ``run_optical_project(..., output_dir=...)``), its files held in memory (``{file name: bytes}``) or a
    ``LocalOpticalRun``. The results are read as stored. With ``verify`` the run's hashes and cross-references are
    checked too (``describe()["integrity"]``; without it ``integrity.checked`` is false); a run that does not verify
    still loads. A run is verified where its bytes cross a trust boundary -- ``project verify``, a submission
    (``local/artifacts.py``) -- not by every query. Raises ``ResultsError`` (``RUN_NOT_READABLE``)
    when the Engine result cannot be read, or when a directory has no ``manifest.json`` (no run was completed into it).
    A directory that also holds the record of a later attempt that stopped (``run_stopped.json``) loads the run that is
    there, with ``describe()["later_attempt"]``."""

    if isinstance(run, Mapping):
        return results_from_files(run, verify=verify)
    if isinstance(run, (str, Path)):
        directory = Path(run)
        if not directory.is_dir():
            raise ResultsError(RUN_NOT_READABLE, f"{directory} is not a run directory")
        if not (directory / MANIFEST_FILE).is_file():
            raise ResultsError(RUN_NOT_READABLE, f"{directory} has no {MANIFEST_FILE}: no run was completed into it; run "
                                                 "'os-optics project run' into it")
        files = {}
        for name in (*RUN_FILE_NAMES, STOPPED_FILE):
            target = directory / name
            if target.is_file():
                try:
                    files[name] = target.read_bytes()
                except OSError as exc:
                    raise ResultsError(RUN_NOT_READABLE, f"{name}: cannot be read ({exc})") from None
        return results_from_files(files, verify=verify)
    if isinstance(run, LocalOpticalRun):
        return results_from_files(run_artifact_files(run), verify=verify)
    raise ResultsError(RUN_NOT_READABLE, "a run is a run directory path, its {file name: bytes} files or a LocalOpticalRun")


def results_from_engine(result: ValidationResult, ray_paths: Any = None, numerical: NumericalConfig | None = None) -> RunResults:
    """``RunResults`` of a low-level Engine run: ``validate_with_ray_paths(request)`` gives ``(result, ray_paths)``
    (``ray_paths``: a ``RayPathTrace`` or its ``ray_paths_document``; ``None``: no ray data)."""

    from ..validation.ray_paths import RayPathTrace, ray_paths_document

    document = ray_paths_document(ray_paths) if isinstance(ray_paths, RayPathTrace) else ray_paths
    return RunResults(result=result, ray_data=document, numerical=numerical or NumericalConfig())
