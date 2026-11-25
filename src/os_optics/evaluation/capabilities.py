"""The capabilities of an evaluation: one constant table.

A capability is a named part of an evaluation with one implementation. ``CAPABILITIES`` lists them in execution order: each
row states what the capability requires, which result sections it produces, the numerical settings that affect it and the
run phases it passes through. ``validate`` runs every capability the request's configuration enables; a selective
evaluation (``os_optics.evaluate``) runs the ones it asks for and what those require (``os_optics.evaluation.plan``).

Engine capabilities compute a ``ValidationResult``; local capabilities (``os_optics.local``) record and present a run.
A Project evaluation (the local level) runs the physics capabilities and the local ones; the layout checks and the control
analysis are capabilities of an Engine request only (``os_optics.evaluate``). ``ANALYSES`` are run on demand -- on an
evaluation's or a run's results, or on a Project or request of their own: each names what it reads of an evaluation, its
Python call with its arguments and its command, and is not a node of an evaluation plan. ``EXPORTS`` is how a complete
run's files are written (``Evaluation.write_run``): not an output of an evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

ENGINE = "engine"
LOCAL = "local"
COMPUTATION = "computation"
RECORDING = "recording"
PRESENTATION = "presentation"
ANALYSIS = "analysis"
EXPORT = "export"

SCENE = "scene"
BEAM_TRACE = "beam_trace"
SECONDARY_RAYS = "secondary_rays"
PLANE_READINGS = "plane_readings"
POWER_LEDGER = "power_ledger"
UNTRACED_LIGHT = "untraced_light"
PLANE_BOUNDS = "plane_bounds"
SURFACE_EXPOSURES = "surface_exposures"
BEAM_PAIRS = "beam_pairs"
LAYOUT = "layout"
CONTROL = "control"
REQUIREMENTS = "requirements"
SOLID_CHECK = "solid_check"
RAY_PATHS = "ray_paths"
MODEL_SCOPE = "model_scope"
RECORD = "record"
VIEW = "view"
RUN_FILES = "run_files"

#: Presets, usable wherever capability ids are. ``complete``: every Engine capability the request's configuration enables
#: (what ``validate`` runs). ``readings``: the plane readings and what they require -- the beams' arrival quantities and
#: traced powers, without the probe traces of untraced light.
COMPLETE = "complete"
READINGS = "readings"

#: The records of the return-light analysis: made by the untraced light when a reader -- the plane bounds or the surface
#: exposures -- reads the analysis (``not_computed_produces``).
RETURN_ANALYSIS_RECORDS = "outcome.execution.budgets.return_analysis"

#: The request flags (``EngineConfiguration``) that turn a capability on.
FLAG_PHYSICS = "run_physics"
FLAG_LAYOUT = "run_layout"
FLAG_CONTROL = "run_control"


@dataclass(frozen=True)
class Capability:
    """One row of the table.

    ``level``: ``engine`` or ``local``. ``kind``: ``computation`` (it computes physical quantities), ``recording`` (it keeps
    what a computation produced), ``presentation`` (it draws it), ``analysis`` (run on demand) or ``export`` (written by
    ``Evaluation.write_run``). ``requires``: the capabilities whose products it always reads. ``local_requires``: the
    capabilities a Project evaluation (the local level) runs with it and that cannot be excluded while it runs: the
    model-scope findings that say whether its numbers apply. ``traced_requires``: capabilities it reads as well when the request traces
    beams (``run_physics``): the layout checks test the traced beam paths against the envelopes. ``rated_requires``:
    capabilities it requires as well when a component of the request states a power rating (the exposures of such a design
    are compared with its ratings, and no comparison is made on the traced beams alone); it reads them whenever they are in
    the evaluation. ``complete_scope``: it also requires every Engine capability of the request's complete scope. ``produces``: the sections and fields of a
    ``ValidationResult`` (or the documents of a local run) it produces. ``controls``: the ``NumericalConfig`` fields that
    affect it. ``phases``: the run phases it passes through (``os_optics.run_control.RUN_PHASES``). ``request_flag``: the
    ``EngineConfiguration`` flag that enables it (empty: always enabled). ``entry_point`` / ``request_schema``: for an
    analysis or an export, the Python call that runs it, with its arguments, and the JSON schema kind of its request (empty:
    it takes query fields). ``command``: the ``os-optics`` command that runs it.
    """

    id: str
    level: str
    kind: str
    summary: str
    requires: tuple[str, ...] = ()
    produces: tuple[str, ...] = ()
    controls: tuple[str, ...] = ()
    phases: tuple[str, ...] = ()
    method: str = ""
    version: str = "1"
    request_flag: str = ""
    traced_requires: tuple[str, ...] = ()
    rated_requires: tuple[str, ...] = ()
    complete_scope: bool = False
    entry_point: str = ""
    request_schema: str = ""
    local_requires: tuple[str, ...] = ()
    command: str = ""


CAPABILITIES: tuple[Capability, ...] = (
    Capability(
        SOLID_CHECK, LOCAL, COMPUTATION,
        "The optical bodies of the lowered Project checked as solids: every profile defined over its outline, a positive "
        "thickness between consecutive surfaces, no two bodies sharing volume, every member inside its unit's envelope. "
        "The report states each finding with its zone and the thickness of every gap; the record is refused for a finding "
        "that makes a body impossible.",
        produces=("optical_solids",), phases=("lower",),
        method="Each element's profiles and gaps sampled over its outline; every pair of optical bodies tested for a common point."),
    Capability(
        SCENE, ENGINE, COMPUTATION,
        "The Project lowered to traced surfaces, sources and media.",
        controls=("max_mesh_patches",), request_flag=FLAG_PHYSICS,
        method="Each component's surfaces placed in the design frame with its interaction model and materials."),
    Capability(
        BEAM_TRACE, ENGINE, COMPUTATION,
        "The traced beams: every beam path of every source line, with its chief ray, pupil rays and Gaussian state.",
        requires=(SCENE,),
        produces=("paths", "grating_orders", "nonlinear_conversions", "surface_profiles", "summary.path_count", "manifest.trace_hash"),
        controls=("max_paths", "max_trace_steps", "max_interactions_per_path", "negligible_power_fraction", "pupil_ring_count",
                  "pupil_points_per_ring", "power_epsilon_W", "gaussian_warn_divergence_rad", "gaussian_scope_divergence_rad",
                  "gaussian_max_w_over_radius", "gaussian_max_aoi_deg", "gaussian_clip_warn_fraction",
                  "gaussian_clip_scope_fraction", "nonlinear_depletion_rel_tol", "nonlinear_focusing_rel_tol",
                  "nonlinear_quadrature_rel_tol", "nonlinear_harmonic_gaussian_rel_tol", "nonlinear_beam_quality_tol",
                  "nonlinear_linewidth_rel_tol", "aperture_quad_sector_nodes", "capture_min_power_fraction"),
        phases=("trace",), request_flag=FLAG_PHYSICS,
        method="Sequential-by-encounter exact ray trace of chief and pupil rays with paraxial Gaussian beam transport."),
    Capability(
        SECONDARY_RAYS, ENGINE, COMPUTATION,
        "The rays of extended emitters and of light the surfaces scatter out of the beams, with the paths and trace events "
        "they add to the beam trace's; no work in a scene without either.",
        requires=(BEAM_TRACE,),
        produces=("sampled_sources", "observations[].sampled_light", "paths", "summary.path_count", "manifest.trace_hash"),
        controls=("max_scatter_origins",), phases=("emitters",), request_flag=FLAG_PHYSICS,
        method="Deterministic sample sequences traced ray by ray."),
    Capability(
        PLANE_READINGS, ENGINE, COMPUTATION,
        "What arrives at each observation plane: per-beam position, direction, size, phase, polarization and power, the traced "
        "power per plane and wavelength, and the coherent sums.",
        requires=(BEAM_TRACE, SECONDARY_RAYS), produces=("observations",), local_requires=(MODEL_SCOPE,),
        controls=("gaussian_max_ray_spread_ratio", "capture_min_power_fraction", "aperture_quad_sector_nodes"),
        phases=("observation_results",), request_flag=FLAG_PHYSICS,
        method="The traced arrivals evaluated on each plane's frame, active area and acceptance cone."),
    Capability(
        POWER_LEDGER, ENGINE, COMPUTATION,
        "Where every launched watt went: the power budget per source line, sink and element, and the light of each route of "
        "a partially polarized source on a plane.",
        requires=(PLANE_READINGS,), produces=("power_budget", "observations[].polarization_mixtures"),
        controls=("energy_conservation_tol", "power_ledger_fraction_resolution"),
        request_flag=FLAG_PHYSICS, method="The trace's power flows booked per line and checked for conservation."),
    Capability(
        UNTRACED_LIGHT, ENGINE, COMPUTATION,
        "The light the beam trace does not follow (ghost reflections, coating transmissions, clipped light), followed by "
        "probe traces, and the analysis of how surfaces may return the light left out as negligible, made when the plane "
        "bounds or the surface exposures read it. The plane bounds read it, and so do the surface exposures of a design with "
        "power ratings: that light on each surface, the upper bounds and the rating comparisons.",
        requires=(BEAM_TRACE, SECONDARY_RAYS),
        produces=("exposures.untraced_light_included", "exposures.faint_light", "exposures.surfaces[].untraced_hits",
                  RETURN_ANALYSIS_RECORDS),
        controls=("max_paths", "max_trace_steps", "max_interactions_per_path", "negligible_power_fraction",
                  "max_return_analysis_work"),
        phases=("untraced_light",), request_flag=FLAG_PHYSICS,
        method="Probe traces of every untraced branch, strongest first, down to the negligible level of its line."),
    Capability(
        PLANE_BOUNDS, ENGINE, COMPUTATION,
        "Whether each plane's power is complete: the untraced light that may reach it, its upper bounds, the grade of a plane "
        "no traced beam reaches, the state of its power and what a stated detector reads.",
        requires=(PLANE_READINGS, POWER_LEDGER, UNTRACED_LIGHT),
        produces=("observations[].quantity_state", "observations[].power_complete", "observations[].untraced_power_W",
                 "observations[].power_upper_bound_W", "observations[].absorbed_power_upper_bound_W",
                 "observations[].power_by_wavelength[].power_W (of a wavelength no traced light brings)",
                 "observations[].power_by_wavelength[].complete", "observations[].power_by_wavelength[].untraced_power_W",
                 "observations[].detection"),
        controls=("negligible_power_fraction", "laboratory_reach_mm", "max_return_analysis_work"),
        phases=("bounds", "return_analysis"), request_flag=FLAG_PHYSICS,
        method="The ledger's sinks and the probes' arrivals combined per plane and source line."),
    Capability(
        SURFACE_EXPOSURES, ENGINE, COMPUTATION,
        "The exposure of every surface a beam meets. A design that states a power rating has the untraced light followed "
        "(untraced_light is then required): its report includes that light, the upper bounds and the comparison with the "
        "ratings. A design without ratings is reported on its traced beams.",
        requires=(POWER_LEDGER,), rated_requires=(UNTRACED_LIGHT,), produces=("exposures",),
        controls=("negligible_power_fraction", "max_return_analysis_work", "rating_wavelength_tol_rel",
                  "rating_pulse_duration_tol_rel", "rating_repetition_rate_tol_rel", "rating_beam_diameter_tol_rel",
                  "rating_polarization_fraction_tol", "rating_unlocated_light_share"),
        phases=("exposures", "return_analysis"), request_flag=FLAG_PHYSICS,
        method="Peak irradiance and fluence of each beam's footprint, summed coherently per coherence group."),
    Capability(
        BEAM_PAIRS, ENGINE, COMPUTATION,
        "The two-beam interference of the beam pairs on each plane: crossing angle, fringe period, path and phase difference, "
        "coherence, overlap and visibility.",
        requires=(PLANE_READINGS,), produces=("interference", "incoherent_pairs"),
        controls=("max_interference_pairs", "interference_wavelength_tol_nm", "interference_min_overlap",
                  "interference_min_crossing_deg"),
        phases=("interference",), request_flag=FLAG_PHYSICS,
        method="Analytic two-beam interference of the chief rays' plane waves with their Gaussian envelopes."),
    Capability(
        LAYOUT, ENGINE, COMPUTATION,
        "The mechanical layout checks of the components' envelopes on the baseplate.",
        produces=("layout",), controls=("layout_geometry_tol_mm", "beam_envelope_k"),
        request_flag=FLAG_LAYOUT, traced_requires=(SECONDARY_RAYS,), method="2.5D footprint, clearance and beam-envelope checks."),
    Capability(
        CONTROL, ENGINE, COMPUTATION,
        "The control analysis of the Project's control specification: sensitivity, recoverability and Monte Carlo.",
        produces=("control",),
        controls=("lm_max_iterations", "fd_step_translation_mm", "fd_step_rotation_rad", "capture_min_power_fraction"),
        request_flag=FLAG_CONTROL, method="Finite-difference sensor Jacobian and Levenberg-Marquardt recovery."),
    Capability(
        REQUIREMENTS, ENGINE, COMPUTATION,
        "The declared targets evaluated against the computed quantities; each target requires the capabilities its metric "
        "reads.",
        produces=("targets", "outcome.requirements"), phases=("targets",), local_requires=(MODEL_SCOPE,),
        method="Each target's metric read from the results and graded on its value or supported interval."),
    Capability(
        RAY_PATHS, LOCAL, RECORDING,
        "The chief and pupil rays of the beam trace, point by point.",
        requires=(BEAM_TRACE,), produces=("ray_data",), phases=("ray_paths",),
        method="A record of the tracer's own ray positions; nothing is traced again."),
    Capability(
        MODEL_SCOPE, LOCAL, COMPUTATION,
        "Whether each published element model applies to the light that meets it, and the targets and planes a finding "
        "affects.",
        requires=(RAY_PATHS, PLANE_READINGS), produces=("model_scope_issues",), phases=("model_scope",),
        controls=("design_point_wavelength_rel", "design_point_retarder_wavelength_rel", "design_point_angle_metallic_reflector_deg",
                  "design_point_angle_dielectric_high_reflector_deg", "design_point_angle_anti_reflection_deg",
                  "design_point_angle_beamsplitter_deg", "design_point_interference_shift_fraction",
                  "design_point_interference_angle_max_deg"),
        method="Each element's stated validity range tested against the traced rays that meet it; a single stated design "
               "wavelength or angle is matched within the design-point tolerance in force."),
    Capability(
        RECORD, LOCAL, RECORDING,
        "The Local Validation record of a complete run: a run of every capability the request enables, its optical bodies "
        "checked as solids.",
        requires=(SOLID_CHECK, MODEL_SCOPE), produces=("local_validation",), phases=("record",), complete_scope=True,
        method="The run's identity, hashes, target outcomes and plane quantities bound to the Project."),
    Capability(
        VIEW, LOCAL, PRESENTATION,
        "The Optical Simulation View scene of a run.",
        requires=(RAY_PATHS, PLANE_READINGS), produces=("optical_scene",), phases=("scene",),
        method="The recorded rays and the planes' arrivals drawn with the elements' display geometry."),
)

def _retrace_controls(*replaced: str) -> tuple[str, ...]:
    """The ``NumericalConfig`` fields an analysis that lowers and traces the Project again reads: the scene lowering's and
    the beam trace's (``SCENE``, ``BEAM_TRACE``), less ``replaced``, those its own request or launch grid sets instead."""

    rows = {row.id: row for row in CAPABILITIES}
    return tuple(c for c in (*rows[SCENE].controls, *rows[BEAM_TRACE].controls) if c not in replaced)


#: The analyses run on demand (not plan nodes): what each reads of an evaluation (``requires``; an analysis of its own
#: Project or request reads none), its Python call with every argument a caller passes (the identity a run hands on is left
#: out) and its command, and the schema kind of its request (``os_optics.local.json_api.json_schema``). ``RunResults`` is
#: ``Evaluation.results()`` or ``os_optics.results.load_run(RUN)``.
ANALYSES: tuple[Capability, ...] = (
    Capability("imaging", LOCAL, ANALYSIS, "Point spread function, MTF and encircled energy of an image plane.",
               requires=(PLANE_READINGS,), entry_point="os_optics.results.RunResults.imaging(monitor, request=None)",
               command="os-optics results imaging RUN --monitor ID [--request FILE]", request_schema="imaging-request",
               controls=_retrace_controls("pupil_ring_count", "pupil_points_per_ring"),
               method="The run's fields traced again with a dense pupil grid and once more with twice the rays per axis: the values "
               "are the finer grid's, each with its difference from the first as its convergence, and the point spread function's "
               "quantities are computed from the traced pupil. A request can leave out the finer grid (convergence false: the "
               "values are then the single grid's, with no convergence) and the point spread function's quantities (diffraction "
               "false)."),
    Capability("ghosts", LOCAL, ANALYSIS, "The interface-reflection ghost images of an image plane, with a bound on those "
               "not listed.", requires=(PLANE_READINGS,), entry_point="os_optics.results.RunResults.ghosts(monitor, request=None)",
               command="os-optics results ghosts RUN --monitor ID [--request FILE]",
               request_schema="ghost-request", controls=_retrace_controls("max_paths", "max_trace_steps"),
               method="Every lens or window face's reflection followed up to a bounce order, traced "
               "again with twice the pupil sampling: the values are the finer trace's, each with its difference from the first as "
               "its convergence. A request can leave out the finer trace (convergence false: the values are then the single "
               "trace's, with no convergence)."),
    Capability("field", LOCAL, ANALYSIS, "The coherent field of a sampled-field or speckle source propagated to a plane.",
               requires=(PLANE_READINGS,), entry_point="os_optics.results.RunResults.field(monitor, request=None)",
               command="os-optics results field RUN --monitor ID [--request FILE]", request_schema="field-request",
               controls=_retrace_controls(),
               method="Angular spectrum in free space; the Collins integral of the first-order transfer through elements."),
    Capability("focal_field", LOCAL, ANALYSIS, "The vector focal field of an ideal aplanatic objective. Lit by the beams of a "
               "plane (HANDOFF_BEAMS) it reads that plane's handoff: the plane readings, the power ledger and the plane bounds.",
               entry_point="os_optics.results.RunResults.focal_field(monitor, request), or "
                           "os_optics.results.focal_field(request, handoff=None) for a stated illumination",
               command="os-optics results focal-field [RUN --monitor ID] --request FILE", request_schema="focal-field-request",
               controls=("gaussian_scope_divergence_rad",), method="Richards-Wolf vector diffraction integral."),
    Capability("pupil_field", LOCAL, ANALYSIS, "The scalar focal irradiance of a pupil field at low numerical aperture. Lit by "
               "the beams of a plane (HANDOFF_BEAMS) it reads that plane's handoff: the plane readings, the power ledger and the "
               "plane bounds.",
               entry_point="os_optics.results.RunResults.pupil_field(monitor, request, mask_files=None), or "
                           "os_optics.results.pupil_field(request, mask_files=None, handoff=None) for a stated illumination",
               command="os-optics results pupil-field [RUN --monitor ID] --request FILE [--mask-file FILE]",
               request_schema="pupil-field-request", method="Scalar Fraunhofer diffraction of the pupil field."),
    Capability("irradiance_map", LOCAL, ANALYSIS, "The irradiance map of a plane's arriving light: of one wavelength, or without "
               "one the sum over the wavelengths that arrive. The coherent map and the "
               "untraced light's share of the map read the plane bounds as well: without them the incoherent map is given and "
               "the coherent one is not computed. max_untraced_fraction is the share of the map's light that may be missing "
               "from it above which the coherent map and a uniformity are withheld (None: no limit).",
               requires=(PLANE_READINGS,), entry_point="os_optics.results.RunResults.irradiance_map(monitor, wavelength_nm=None, "
               "half_extent_mm=None, samples=None, source_id=None, center_uv_mm=None, roi=None, max_untraced_fraction=0.1)",
               command="os-optics results irradiance RUN --monitor ID", controls=("interference_wavelength_tol_nm",),
               method="The arriving beams' Gaussian envelopes and the sampled rays' crossings binned on the plane."),
    Capability("detector_response", LOCAL, ANALYSIS, "Detector counts from an irradiance map: a plane's (it reads the plane "
               "readings; further keyword arguments of RunResults.detector_response are those of RunResults.irradiance_map) or "
               "an irradiance document's.",
               entry_point="os_optics.results.RunResults.detector_response(monitor, detector, composition=None), or "
                           "os_optics.results.detector_response(irradiance, detector, composition=None)",
               command="os-optics results detector [RUN --monitor ID | --field FILE] --detector FILE", request_schema="detector-model",
               controls=("interference_wavelength_tol_nm",), method="Pixel integration with the stated detector model."),
    Capability("cavity", LOCAL, ANALYSIS, "The modes, stability and injection of a linear cavity of a Project.",
               entry_point="os_optics.local.cavity_analysis(project, request, catalog=None), or "
                           "os_optics.results.RunResults.cavity(request, catalog=None)",
               command="os-optics project cavity PROJECT --request FILE, or os-optics results cavity RUN --request FILE",
               controls=("gaussian_scope_divergence_rad", "gaussian_max_w_over_radius", "negligible_power_fraction"),
               method="Round-trip ray-transfer matrix of the cavity."),
    Capability("paraxial", LOCAL, ANALYSIS, "The first-order properties of a design's axis and its afocal and collimation "
               "solves.", entry_point="os_optics.local.paraxial_analysis(project, request, catalog=None)",
               command="os-optics project paraxial PROJECT --components A B --wavelength-nm W [--solve afocal|collimate]",
               method="Paraxial ray-transfer matrices."),
)

#: How a complete run's files are written: by ``Evaluation.write_run``, never as an output of an evaluation.
EXPORTS: tuple[Capability, ...] = (
    Capability(RUN_FILES, LOCAL, EXPORT,
               "The run directory of a complete run, written by Evaluation.write_run(directory), "
               "run_optical_project(project, output_dir=...) or 'os-optics project run PROJECT --out-dir DIR'. An evaluation "
               "writes nothing, so run_files is not one of its outputs.",
               requires=(RECORD, VIEW, RAY_PATHS), produces=("run_directory",), phases=("write",),
               entry_point="os_optics.local.Evaluation.write_run(output_dir, ray_data='full', monitor=None)",
               command="os-optics project run PROJECT --out-dir DIR",
               method="The run's documents written with a manifest of their hashes."),
)

PRESETS: tuple[str, ...] = (COMPLETE, READINGS)
#: What each preset names at the Engine level beside the capabilities the request enables (``complete``: all of them), and
#: what it adds at the local level: a complete local run checks the solids, records the rays, checks the model scope and
#: makes the record and the view; local readings come with the model-scope findings that say whether a number applies.
ENGINE_PRESET_MEMBERS: dict[str, tuple[str, ...]] = {COMPLETE: (), READINGS: (PLANE_READINGS,)}
LOCAL_PRESET_MEMBERS: dict[str, tuple[str, ...]] = {COMPLETE: (SOLID_CHECK, RAY_PATHS, MODEL_SCOPE, RECORD, VIEW), READINGS: (MODEL_SCOPE,)}
CAPABILITY_BY_ID: dict[str, Capability] = {row.id: row for row in CAPABILITIES}
ANALYSIS_BY_ID: dict[str, Capability] = {row.id: row for row in ANALYSES}
EXPORT_BY_ID: dict[str, Capability] = {row.id: row for row in EXPORTS}
#: The Engine capabilities of an Engine request only: a Project's request runs neither the layout checks nor the control
#: analysis, so a Project evaluation (the local level) does not list or accept them.
ENGINE_REQUEST_ONLY: tuple[str, ...] = (LAYOUT, CONTROL)
ENGINE_CAPABILITIES: tuple[str, ...] = tuple(row.id for row in CAPABILITIES if row.level == ENGINE)
LOCAL_CAPABILITIES: tuple[str, ...] = tuple(row.id for row in CAPABILITIES if row.level == LOCAL)
#: The capabilities of the physics run (``run_physics``): the Engine capabilities the ``run_physics`` flag enables.
PHYSICS_CAPABILITIES: tuple[str, ...] = tuple(row.id for row in CAPABILITIES if row.request_flag == FLAG_PHYSICS)


def capability(capability_id: str) -> Capability:
    """The row of one capability, analysis or export (``KeyError`` for an id the table does not hold)."""
    found = CAPABILITY_BY_ID.get(capability_id) or ANALYSIS_BY_ID.get(capability_id) or EXPORT_BY_ID.get(capability_id)
    if found is None:
        raise KeyError(capability_id)
    return found


def not_computed_produces(capability_id: str, executed, exposures_read_analysis: bool = True) -> tuple[str, ...]:
    """What a capability an evaluation did not run would have produced, in the result of the evaluation that ran
    ``executed``: the row's ``produces``; for the plane bounds also the return-light analysis' records when the evaluation
    ran the untraced light without a reader of that analysis (neither the plane bounds nor the surface exposures): the
    analysis is made when they read it, and the plane bounds are its first reader. ``exposures_read_analysis`` false: the
    design states no power rating, so the surface exposures read no analysis and are no reader of it."""
    row = CAPABILITY_BY_ID[capability_id]
    ran = set(executed)
    if capability_id == PLANE_BOUNDS and UNTRACED_LIGHT in ran and (SURFACE_EXPOSURES not in ran or not exposures_read_analysis):
        return (*row.produces, RETURN_ANALYSIS_RECORDS)
    return row.produces


def capability_ids(level: str | None = None) -> tuple[str, ...]:
    """The capability ids in execution order (``level``: only the ``engine`` or the ``local`` ones)."""
    return tuple(row.id for row in CAPABILITIES if level is None or row.level == level)


def level_capabilities(level: str) -> tuple[Capability, ...]:
    """The rows an evaluation at ``level`` selects from, in execution order: ``engine`` (``os_optics.evaluate``) every Engine
    capability; ``local`` (a Project evaluation) the Engine ones a Project's request can run and the local ones."""
    if level == ENGINE:
        return tuple(row for row in CAPABILITIES if row.level == ENGINE)
    return tuple(row for row in CAPABILITIES if row.id not in ENGINE_REQUEST_ONLY)


def capability_table() -> tuple[Capability, ...]:
    """Every capability row, in execution order."""
    return CAPABILITIES


def analyses() -> tuple[Capability, ...]:
    """Every analysis row."""
    return ANALYSES


def exports() -> tuple[Capability, ...]:
    """Every export row."""
    return EXPORTS


def canonical(ids) -> tuple[str, ...]:
    """``ids`` once each, in the table's execution order."""
    wanted = set(ids)
    return tuple(row.id for row in CAPABILITIES if row.id in wanted)


__all__ = [
    "ANALYSES", "ANALYSIS", "BEAM_PAIRS", "BEAM_TRACE", "CAPABILITIES", "COMPLETE", "COMPUTATION", "CONTROL", "ENGINE",
    "ENGINE_CAPABILITIES", "ENGINE_PRESET_MEMBERS", "ENGINE_REQUEST_ONLY", "EXPORT", "EXPORTS", "EXPORT_BY_ID", "FLAG_CONTROL",
    "FLAG_LAYOUT", "FLAG_PHYSICS", "LAYOUT", "LOCAL",
    "LOCAL_CAPABILITIES", "LOCAL_PRESET_MEMBERS", "MODEL_SCOPE",
    "PHYSICS_CAPABILITIES", "PLANE_BOUNDS", "PLANE_READINGS", "POWER_LEDGER", "PRESENTATION", "PRESETS", "RAY_PATHS", "READINGS",
    "RECORD", "RECORDING", "REQUIREMENTS", "RUN_FILES", "SCENE", "SECONDARY_RAYS", "SOLID_CHECK", "SURFACE_EXPOSURES", "UNTRACED_LIGHT",
    "VIEW",
    "Capability", "analyses", "canonical", "capability", "capability_ids", "capability_table", "exports", "level_capabilities",
]
