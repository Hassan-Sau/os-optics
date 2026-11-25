"""The public entry points: ``validate``, ``evaluate`` and ``validate_json``.

Pure function of the request: no I/O, no clocks, no global state. Input-contract violations and non-finite results are
returned as structured FAIL results, never as exceptions or NaN.

An evaluation is a plan over capabilities (``os_optics.evaluation``): each physics capability is one node function here
(``PHYSICS_NODES``), run at most once per evaluation over an explicit ``EvaluationState`` that keeps its product.
``validate`` runs every capability the request enables; ``evaluate`` runs a selection with what it requires; the Engine's
own compositions (control readings, the network export) run the same nodes with their own parameters (``PhysicsPlan``).
Each node keeps the issues it raised, and a result lists them capability by capability in execution order, so a result
built over a retained state equals one computed afresh.
"""

from __future__ import annotations

import itertools
import math

from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import ValidationError

from .._version import ENGINE_VERSION, MODEL_VERSIONS, NUMERICAL_CONFIG_VERSION, PRNG_ID, SCHEMA_VERSION
from ..config import NumericalConfig
from ..contracts.evaluation import EXCLUDED, NOT_REQUESTED, EvaluationScope, NotComputed, TargetNotEvaluated
from ..contracts.manifest import ValidationManifest
from ..contracts.optics import OpticalPath
from ..contracts.outcome import ReturnAnalysisRecord
from ..contracts.project import ProjectSnapshot, Target
from ..contracts.request import ValidationRequest
from ..contracts.results import (
    ExposureReport,
    IncoherentPairs,
    InterferenceResult,
    ObservationResult,
    PowerBudget,
    SampledSourceLine,
    SourceSpectrum,
    SpectralPower,
    TargetResult,
    ValidationResult,
    ValidationSummary,
)
from ..errors import EngineInputError, ErrorCode, ModelScopeRefusal, Status, TerminationReason, ValidationIssue, make_issue, notice, worst
from ..errors.status import QuantityState
from ..evaluation.capabilities import (
    BEAM_PAIRS,
    BEAM_TRACE,
    COMPLETE,
    CONTROL,
    LAYOUT,
    PHYSICS_CAPABILITIES,
    PLANE_BOUNDS,
    PLANE_READINGS,
    POWER_LEDGER,
    REQUIREMENTS,
    SCENE,
    SECONDARY_RAYS,
    SURFACE_EXPOSURES,
    UNTRACED_LIGHT,
    not_computed_produces,
)
from ..evaluation.plan import EvaluationPlan, plan_evaluation
from ..evaluation.state import RECORDER_CHANGED, EvaluationState, PhysicalIdentity
from ..manifest import engine_build_id, hash_exact, hash_quantized, runtime_info
from ..manifest.streaming import hash_quantized_parts, model_parts
from ..math.pose import Pose
from ..optics.interference import analyse_pair
from ..optics.materials.library import MaterialLibrary
from ..optics.gaussian.truncation import QuadratureRuleTooLarge
from ..optics.rays.tracer import ArrivalRecord, TraceObserver, TraceOutput, Tracer
from ..run_control import checkpoint, phase, traced
from ..scene.lower import lower_scene
from ..scene.model import Scene, line_coherence_length_mm
from ..contracts.results import SurfaceProfileEvidence
from ..units import NM_TO_MM
from .affected import lit_reach_issues, name_affected_targets
from .coating_use import coating_uses, declared_air_equivalent
from .exposures import (POWER_RATING_UTILIZATION, build_exposure_report, closed_return_kinds, closed_return_text, kinds_context,
                        named_returns, pair_facts, rated_surfaces, spectral_factor, unbounded_pairs, closed_return_facts,
                        SCENE_WIDE)
from .detection import detection_result
from .metrics import PAIR_METRICS, assess_targets
from .pair_selection import plane_pairs, same_wavelength_pair_count
from .outcome import quantity_state_of, result_outcome
from .power import PARTIAL_CROSSING_PLACEMENT, build_power_budget
from .observations import (
    attribute_power_floor,
    build_observation_results,
    check_gaussian_aberration,
    complete_power_by_wavelength,
    grade_unlit_planes,
    partial_reach,
    plane_power_scope,
    power_scope_reach,
    split_arrivals,
    unknown_power_reach,
)
from .scope import MODEL_SCOPE_SUPPORTED, OUT_OF_SCOPE_METRICS
from .target_parameters import PAIR_PATH_SELECTION
from .resonance import ReturnAnalysis, ReturnGains
from .sampled_light import combine_surface_hits, sampled_reach, sampled_sources, trace_sampled_light
from .scattered_light import scatter_issues, scattered_reach, scattered_sources, trace_scattered_light
from ..optics.interference.source_coherence import source_coherence
from ..optics.interference.spectrum import source_spectra
from .untraced import (FAINT_POWER, STORED_KINDS, PlaneReach, Probes, follow_untraced, merge_reach, partial_surfaces, plane_reach, probe_budget_issues,
                       surface_reach)
from .window_bound import surface_room
from .polarization_mixture import with_polarization_mixtures
from ..optics.polarization.partial import GROUP_MARK, component_pair

#: The most deeper probe passes a run makes to follow light left out as negligible where it alone keeps a power-rating
#: comparison from being decided (``validation/exposures.py``: ``ExposureOutcome.deeper_levels``).
MAX_DEEPER_PASSES = 4
#: Notices (status NOT_APPLICABLE: no effect on the run's status) the manifest's ``warnings`` list too.
MANIFEST_NOTICE_CODES = frozenset({ErrorCode.POWER_RATING_EXCEEDED, ErrorCode.EXPOSURE_BOUND_WITHHELD})
MAX_REPORTED_CONTRACT_ERRORS = 20  # first N pydantic errors are echoed in the INVALID_INPUT issue
MAX_PARSE_ERROR_CHARS = 200  # the JSON parser's own message is echoed up to this length
MAX_FIRST_ERROR_CHARS = 140  # the INVALID_INPUT sentence names the first contract error up to this length
#: Where the refusal of an aperture rule the machine cannot hold is located (``quadrature_refusal``).
QUADRATURE_SETTING = "configuration.numerical.aperture_quad_sector_nodes"
NUMERICAL_FAILURE_MESSAGE = (
    "a number of the design (a position, a size or a beam parameter) is too large or too small to compute with; "
    "the result is withheld -- use values of a realistic magnitude"
)


@dataclass
class PhysicsOutcome:
    scene: Scene
    trace: TraceOutput
    observations: list[ObservationResult]
    interference: list[InterferenceResult]
    issues: list[ValidationIssue]
    power_budget: PowerBudget | None = None
    # per plane and source line no traced beam brings to it: light the trace does not follow that may reach it
    reach: PlaneReach | None = None
    # per-surface beam exposure and the power-rating comparison (validation/exposures.py)
    exposures: ExposureReport | None = None
    # surface id -> the target metrics a response model there was validated for (tabulated thin lenses)
    metric_scope: dict[str, frozenset[str]] = field(default_factory=dict)
    # element id -> the response-model surfaces not validated for the exposure of its rated surfaces behind them
    element_scope: dict[str, tuple[str, ...]] = field(default_factory=dict)
    # every spectral line of the extended emitters traced (validation/sampled_light.py)
    sampled_sources: tuple[SampledSourceLine, ...] = ()
    # per vacuum wavelength the returns between surfaces facing each other (validation/resonance.py; None: no probe pass)
    gains: Mapping[float, ReturnGains] | None = None
    # per plane the same-wavelength beam pairs of mutually incoherent sources, counted (validation/pair_selection.py)
    incoherent_pairs: tuple[IncoherentPairs, ...] = ()
    # how each Gaussian-beam source's spectrum is traced (optics/interference/spectrum.py: ValidationResult.source_spectra)
    source_spectra: tuple[SourceSpectrum, ...] = ()
    # element id -> the surfaces behind which the power of light reaching its rated surfaces is out of model scope
    element_power_scope: dict[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class PhysicsPlan:
    """The physics capabilities one composition runs (``os_optics.evaluation.capabilities``) and the parameters of their
    nodes. An evaluation's plan runs the capabilities it names with every parameter at its default; the Engine's own
    compositions below run all of them but the beam pairs with the parameters a reading of the beams alone needs.

    ``pupil_rays``: the beam trace carries each beam's pupil and differential rays (without them: the chief rays and the
    Gaussian states -- no pupil spot, no first-order transfer, no aberration check of the Gaussian state).
    ``rated_surfaces``: the beam trace watches the surfaces that state a power rating for beams whose chief ray misses them
    (``SurfaceHit.miss_distance_mm``: their exposure is then bounded, never a known zero) and keeps the branches it leaves
    out as negligible (a deeper probe pass may follow them). ``secondary_rays``: the rays of extended emitters and of
    light the surfaces scatter out of the beams are traced. ``follow_untraced``: a probe pass follows every untraced
    branch of every source line (the bounds of lit planes and of the exposures); without it the plane bounds probe the
    lines dark on some plane themselves and the exposures are the traced beams'.
    """

    capabilities: frozenset[str] = frozenset(PHYSICS_CAPABILITIES)
    pupil_rays: bool = True
    rated_surfaces: bool = True
    secondary_rays: bool = True
    follow_untraced: bool = True

    def without(self, *capabilities: str) -> PhysicsPlan:
        """This plan without ``capabilities`` (only one nothing else of the plan requires: the beam pairs, the exposures)."""
        return replace(self, capabilities=self.capabilities - frozenset(capabilities))


#: Every physics capability at its defaults: the physics of a complete evaluation.
COMPLETE_PHYSICS = PhysicsPlan()
#: The traced beams alone, with their pupil rays (the network export): no beam pairs, no secondary rays, no probe pass
#: over every line, no watch on rated surfaces.
BEAMS_ALONE = PhysicsPlan(capabilities=frozenset(PHYSICS_CAPABILITIES) - {BEAM_PAIRS}, rated_surfaces=False, secondary_rays=False,
                          follow_untraced=False)
#: The same without pupil rays: what a sensor of the control analysis reads, once per actuator state.
SENSOR_READINGS = replace(BEAMS_ALONE, pupil_rays=False)
#: Every physics capability but the beam pairs, traced without pupil rays: the chief rays with the complete bounds.
CHIEF_RAYS = replace(COMPLETE_PHYSICS.without(BEAM_PAIRS), pupil_rays=False)

#: Which reader of the return-light analysis an exposure report was made after: the plane bounds (``CHAIN``: the order of a
#: complete evaluation) or none (``SOLO``); ``TRACED``: the exposures of the traced beams alone, which read no analysis.
CHAIN = "chain"
SOLO = "solo"
TRACED = "traced"
#: A surface's solver statistics as a capability leaves them: (solved rays, most steps, largest residual, rays not converged).
SolverRecord = tuple[int, int, float, int]


@dataclass
class _Lowered:
    """``scene``: the lowered scene and its material library."""

    scene: Scene
    library: MaterialLibrary
    issues: tuple[ValidationIssue, ...]
    freeform: tuple[Any, ...]  # (the surfaces with a two-variable sag shape: their solver statistics are taken per capability)
    metric_scope: dict[str, frozenset[str]]
    media: Callable[[str, float], float] | None = None
    solver: dict[str, SolverRecord] = field(default_factory=dict)


@dataclass
class _Traced:
    """``beam_trace``: the trace of the beams, as the tracer left it."""

    trace: TraceOutput
    quad_rule: Any
    watch: frozenset[str]
    issues: tuple[ValidationIssue, ...]
    solver: dict[str, SolverRecord] = field(default_factory=dict)


@dataclass
class _Secondary:
    """``secondary_rays``: the beam trace with what the secondary rays booked (inputs, ledger, paths, hits, events)."""

    trace: TraceOutput
    sampled: Any
    scattered: Any
    issues: tuple[ValidationIssue, ...]
    solver: dict[str, SolverRecord] = field(default_factory=dict)


@dataclass
class _Readings:
    """``plane_readings``: the arrivals counted and not counted, the sources' spectra and coherence, each plane's readings."""

    kept: list[ArrivalRecord]
    rejected: list[ArrivalRecord]
    spectra: tuple[SourceSpectrum, ...]
    sources: Any
    observations: list[ObservationResult]
    issues: tuple[ValidationIssue, ...]
    solver: dict[str, SolverRecord] = field(default_factory=dict)


@dataclass
class _Ledger:
    """``power_ledger``."""

    budget: PowerBudget
    issues: tuple[ValidationIssue, ...]
    solver: dict[str, SolverRecord] = field(default_factory=dict)


@dataclass
class _Untraced:
    """``untraced_light``: the probe pass over every line (``None``: none was made) and the return-light analyses read on
    demand, one per chain of readers (``CHAIN``, ``SOLO``)."""

    probes: Probes | None
    issues: tuple[ValidationIssue, ...]
    gains: dict[str, ReturnAnalysis] = field(default_factory=dict)
    solver: dict[str, SolverRecord] = field(default_factory=dict)


@dataclass
class _Bounds:
    """``plane_bounds``: the reach of untraced light per plane, the planes with their bounds, and what the return-light
    analysis had been asked when the bounds were done (its work issue and its records)."""

    reach: PlaneReach
    observations: list[ObservationResult]
    issues: tuple[ValidationIssue, ...]
    return_work: tuple[ValidationIssue, ...] = ()
    analyses: tuple[ReturnAnalysisRecord, ...] = ()
    solver: dict[str, SolverRecord] = field(default_factory=dict)


@dataclass
class _Exposed:
    """``surface_exposures``: the exposure report with its scopes, and the return-light analysis' state after it."""

    exposure: Any
    issues: tuple[ValidationIssue, ...]
    return_work: tuple[ValidationIssue, ...] = ()
    analyses: tuple[ReturnAnalysisRecord, ...] = ()
    solver: dict[str, SolverRecord] = field(default_factory=dict)


@dataclass
class _Pairs:
    """``beam_pairs``: the pairs analysed, the pairs counted, and the planes whose pairs the pair budget cut."""

    interference: list[InterferenceResult]
    incoherent: tuple[IncoherentPairs, ...]
    issues: tuple[ValidationIssue, ...]
    truncated: frozenset[str]
    solver: dict[str, SolverRecord] = field(default_factory=dict)


@dataclass
class _LaidOut:
    """``layout``."""

    layout: Any
    issues: tuple[ValidationIssue, ...]
    solver: dict[str, SolverRecord] = field(default_factory=dict)


@dataclass
class _Controlled:
    """``control``: the analysis and the seed it drew its Monte Carlo samples with."""

    control: Any
    issues: tuple[ValidationIssue, ...]
    seed: int | None
    solver: dict[str, SolverRecord] = field(default_factory=dict)


@dataclass
class _Evaluation:
    """One evaluation over a state: the request's parts, the capabilities it runs (``executed``, in execution order), the
    parameters of its physics nodes, and what it ran and what it reused."""

    project: ProjectSnapshot
    cfg: NumericalConfig
    state: EvaluationState
    executed: tuple[str, ...]
    physics: PhysicsPlan
    recorder: TraceObserver | None = None
    quad_rule: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None
    component_poses: dict[str, Pose] | None = None
    seed: int | None = None
    ran: list[str] = field(default_factory=list)
    reused: list[str] = field(default_factory=list)

    def runs(self, capability: str) -> bool:
        return capability in self.executed

    def product(self, capability: str) -> Any:
        return self.state.product(capability)


def _take_solver_stats(surfaces: tuple[Any, ...]) -> dict[str, SolverRecord]:
    """What the intersection solver did on each freeform surface since the statistics were last taken, and reset them: each
    capability that traces accounts for its own rays, so a result sums the capabilities it ran and a retained scene never
    carries the rays of another evaluation."""
    taken: dict[str, SolverRecord] = {}
    for surface in surfaces:
        stats = surface.freeform.stats
        if stats.solved_rays or stats.max_iterations or stats.non_converged_rays or stats.max_residual_mm:
            taken[surface.surface_id] = (stats.solved_rays, stats.max_iterations, stats.max_residual_mm, stats.non_converged_rays)
            stats.solved_rays = stats.max_iterations = stats.non_converged_rays = 0
            stats.max_residual_mm = 0.0
    return taken


def _node(ev: _Evaluation, capability: str, parameters: Any, compute: Callable[[_Evaluation], Any]) -> Any:
    """The capability's product: the state's when it holds one made with these parameters, else ``compute``'s, stored once
    the node finished (a node a monitor stops stores nothing: the next evaluation runs it again)."""
    state = ev.state
    if state.holds(capability, parameters):
        ev.reused.append(capability)
        return state.product(capability)
    if state.has(capability):
        # (a product made with other parameters, and what was built on it. Where the state holds none, nothing it holds
        # was built on one: a retained product of a reader that ran without this capability stays)
        state.discard(capability)
    surfaces = state.product(SCENE).freeform if capability != SCENE and state.has(SCENE) else ()
    try:
        product = compute(ev)
    finally:
        solver = _take_solver_stats(surfaces)
    product.solver = solver
    state.store(capability, parameters, product)
    ev.ran.append(capability)
    return product


def _scene_node(ev: _Evaluation) -> _Lowered:
    library = MaterialLibrary(ev.project.materials, ev.cfg)
    scene, _poses = lower_scene(ev.project, ev.cfg, library, component_poses=ev.component_poses)
    metric_scope = {surface.surface_id: surface.interaction.table.supported_metrics for surface in scene.surfaces
                    if surface.interaction.table is not None and surface.interaction.table.supported_metrics is not None}
    return _Lowered(scene, library, tuple(scene.notices), tuple(s for s in scene.surfaces if s.freeform is not None), metric_scope)


def _beam_trace_node(ev: _Evaluation) -> _Traced:
    lowered: _Lowered = ev.product(SCENE)
    # with power ratings, the rated surfaces are watched for beams whose chief ray misses them while their footprint reaches
    # them (``SurfaceHit.miss_distance_mm``): their exposure is then bounded, never a known zero
    watch = rated_surfaces(ev.project) if ev.physics.rated_surfaces else frozenset()
    # (a ray-path recorder keeps the record of the trace that starts here, and of no earlier one)
    started = getattr(ev.recorder, "beam_trace_started", None)
    if started is not None:
        started()
    with phase("trace", unit="beam paths"):
        # (with ratings the trace also keeps the branches it leaves out as negligible: a deeper probe pass may follow them;
        # the tracer builds the aperture rule, whose checkpoints are the trace's work)
        tracer = Tracer(lowered.scene, lowered.library, ev.cfg, include_pupil=ev.physics.pupil_rays, recorder=ev.recorder,
                        watch=watch, quad_rule=ev.quad_rule, keep_faint=bool(watch), hold_coating_data=True)
        trace = tracer.run()
    # (the recorder's record is whole: a monitor that stopped the trace above leaves it open)
    finished = getattr(ev.recorder, "beam_trace_finished", None)
    if finished is not None:
        finished()
    # (a monitored run: what the trace counted, for the estimate of the rest of the run -- os_optics.run_control)
    traced(len(trace.paths), sum(len(path.segments) for path in trace.paths))
    ev.state.recorder = ev.recorder
    return _Traced(trace, tracer.quad_rule, watch, tuple(trace.issues))


def _secondary_rays_node(ev: _Evaluation) -> _Secondary:
    lowered: _Lowered = ev.product(SCENE)
    beams: _Traced = ev.product(BEAM_TRACE)
    scene, library, primary = lowered.scene, lowered.library, beams.trace
    # (the secondary rays book into a trace of their own lists: the beam trace stays as the tracer left it)
    trace = replace(primary, inputs=list(primary.inputs), power_flows=list(primary.power_flows), issues=list(primary.issues),
                    paths=list(primary.paths), surface_hits=list(primary.surface_hits), trace_events=list(primary.trace_events))
    # (the observer of the beam trace observes the extended emitters' rays of its secondary rays too)
    observer = ev.recorder if ev.recorder is not None else ev.state.recorder
    started = getattr(observer, "secondary_rays_started", None)
    if started is not None:
        started()
    with phase("emitters"):
        # (the rays of extended emitters; control readings and the network export read the beams alone)
        sampled = (trace_sampled_light(scene, library, ev.cfg, trace, beams.quad_rule, observer)
                   if scene.emitters and ev.physics.secondary_rays else None)
        # (the light the surfaces scatter out of the beams, traced as secondary emitters: validation/scattered_light.py)
        scattered = (trace_scattered_light(scene, library, ev.cfg, trace, beams.quad_rule)
                     if trace.scatter_origins and ev.physics.secondary_rays else None)
        # (the observer reads what the model-scope check needs of the scattered rays: ScatterLine.scope)
        scattered_finished = getattr(observer, "scattered_finished", None)
        if scattered is not None and scattered_finished is not None:
            scattered_finished(scattered)
    # (the recorder's record of the secondary rays is whole: a monitor that stopped them above leaves it open)
    finished = getattr(observer, "secondary_rays_finished", None)
    if finished is not None:
        finished()
    trace.issues.extend(scatter_issues(trace, library, scattered, ev.cfg))
    return _Secondary(trace, sampled, scattered, tuple(trace.issues[len(primary.issues):]))


def _plane_readings_node(ev: _Evaluation) -> _Readings:
    lowered: _Lowered = ev.product(SCENE)
    beams: _Traced = ev.product(BEAM_TRACE)
    secondary: _Secondary = ev.product(SECONDARY_RAYS)
    project, cfg, scene, trace = ev.project, ev.cfg, lowered.scene, secondary.trace
    obs_ids = [o.observation_id for o in project.observations]
    arrivals, aberration_issues = check_gaussian_aberration(trace.arrivals, cfg)
    kept, rejected = split_arrivals(arrivals, scene, cfg, beams.quad_rule)
    # (what the sources state beyond their lines: pulse trains, continuous spectra, partially polarized sources sharing a
    # coherence group -- optics/interference/source_coherence.py)
    spectra = source_spectra(project.components)
    sources = source_coherence(project.components, spectra)
    with phase("observation_results", total=len(kept), unit="arrivals"):
        obs_results, obs_issues = build_observation_results(kept, scene, obs_ids, cfg, rejected, beams.quad_rule, project.observations,
                                                            sources, secondary.sampled, secondary.scattered)
    issues = [*aberration_issues, *attribute_power_floor(obs_issues, trace.paths, scene.sources, cfg)]
    return _Readings(kept, rejected, spectra, sources, obs_results, tuple(issues))


def _power_ledger_node(ev: _Evaluation) -> _Ledger:
    budget, issues = build_power_budget(ev.product(SECONDARY_RAYS).trace, ev.product(PLANE_READINGS).observations, ev.cfg)
    return _Ledger(budget, tuple(issues))


def _rated(ev: _Evaluation) -> bool:
    """The design states a power rating: its exposures read the untraced light."""
    return bool(ev.product(BEAM_TRACE).watch) or any(c.power_ratings for c in ev.project.components)


def _untraced_light_node(ev: _Evaluation) -> _Untraced:
    lowered: _Lowered = ev.product(SCENE)
    trace = ev.product(SECONDARY_RAYS).trace
    # with power ratings or an observation plane, every untraced branch is followed (one probe pass for the planes and the
    # surfaces): light the trace does not follow bounds a lit plane's power and a surface's exposure from above too; a
    # coherent power and a pair's light at its peak position it may lower as well.
    # (Which planes and ratings the Project has, never which bounds a target sets: changing a target's threshold must not
    # change what the run computes -- the probes, the untraced bound and the result hash. Every observation reports power,
    # including when the user declares no optional target; adding or removing a requirement must not change whether
    # missing light is followed.)
    wants_bounded_power = bool(ev.project.observations)
    with phase("untraced_light", unit="probe paths"):
        probes = (follow_untraced(trace, lowered.scene, lowered.library, ev.cfg, watch=ev.product(BEAM_TRACE).watch)
                  if ev.physics.follow_untraced and (_rated(ev) or wants_bounded_power) else None)
    # (the pass's own budget, over every line it followed: the planes and the surfaces both read it)
    issues = probe_budget_issues(probes, ev.cfg, {(i.source_id, i.spectral_index) for i in trace.inputs}) if probes is not None else []
    return _Untraced(probes, tuple(issues))


def _return_gains(ev: _Evaluation, readers: str) -> ReturnAnalysis | None:
    """The return-light analysis of the evaluation's probe pass for one chain of readers (surfaces facing each other that
    may return light left out as negligible, and their factors: validation/resonance.py). A wavelength is analysed when a
    bound first reads it -- light left out of the trace at that wavelength --, never for a run whose bounds need none of
    it, and the run's work limit is spent in the order the wavelengths are first read: so the plane bounds and the
    exposures after them (``CHAIN``) share one analysis, read in that order, and exposures made without the plane bounds
    (``SOLO``) have their own, read first in the order the plane bounds read theirs (the wavelengths of the lines that left
    light out, in the order of the trace's inputs, where the Project has an observation plane). The limit is then spent on
    the same wavelengths as in a complete evaluation, and the exposures and the analysis records equal its own whether or
    not the limit is reached. The exposures of a design that states no power rating read no analysis: without the plane
    bounds none is made for them and none recorded. ``None``: no probe pass."""
    untraced: _Untraced = ev.product(UNTRACED_LIGHT)
    probes = untraced.probes
    if probes is None:
        return None
    lowered: _Lowered = ev.product(SCENE)
    scene, library, cfg, trace = lowered.scene, lowered.library, ev.cfg, ev.product(SECONDARY_RAYS).trace
    found = untraced.gains.get(readers)
    if found is None:
        found = untraced.gains[readers] = ReturnAnalysis(scene, library, {i.wavelength_nm for i in trace.inputs},
                                                         lambda: _uses(trace, scene, library, cfg, probes),
                                                         cfg.max_return_analysis_work)
    if readers == SOLO and ev.project.observations and _rated(ev):
        # (``plane_reach`` reads, for its first plane, every line with left-out light in the order of the inputs; repeated
        # on every read, since a monitor may have stopped the reads of an earlier evaluation over the same state part way:
        # the analysed wavelengths are kept, and the rest are read in the same order)
        left = surface_reach(trace, probes, cfg).left_out_root
        for line in trace.inputs:
            if left.get((line.source_id, line.spectral_index), 0.0) > 0.0:
                found[line.wavelength_nm]
    return found


def _media(ev: _Evaluation) -> Callable[[str, float], float]:
    lowered: _Lowered = ev.product(SCENE)
    if lowered.media is None:
        lowered.media = media_index(lowered.scene, lowered.library)
    return lowered.media


def _plane_bounds_node(ev: _Evaluation) -> _Bounds:
    lowered: _Lowered = ev.product(SCENE)
    secondary: _Secondary = ev.product(SECONDARY_RAYS)
    readings: _Readings = ev.product(PLANE_READINGS)
    budget = ev.product(POWER_LEDGER).budget
    probes = ev.product(UNTRACED_LIGHT).probes
    project, cfg, scene, trace = ev.project, ev.cfg, lowered.scene, secondary.trace
    obs_ids = [o.observation_id for o in project.observations]
    gains = _return_gains(ev, CHAIN)
    with phase("bounds"):
        followed: dict = {}
        reach, reach_issues = plane_reach(trace, scene, lowered.library, readings.kept, readings.rejected, obs_ids, cfg, probes, gains,
                                          diffraction_roots(project, scene, _media(ev)), followed)
        reach = sampled_reach(reach, secondary.sampled, trace, obs_ids, cfg)
        reach = scattered_reach(reach, secondary.scattered, obs_ids, cfg)
        unknown_harmonics = tuple(sorted({c.harmonic_wavelength_nm for c in trace.nonlinear_conversions
                                          if c.status is Status.UNKNOWN}))
        observations = complete_power_by_wavelength(readings.observations, budget, cfg, reach, followed, unknown_harmonics)
        observations = with_polarization_mixtures(observations, budget, cfg)
    # (a probe pass over every line reports its own budget, ``_untraced_light_node``; without one the reach probes the
    # lines dark on some plane itself, and its budget is reported here)
    return _Bounds(reach, observations, tuple(reach_issues) if probes is None else (), tuple(return_work_issues(gains, scene)),
                   return_analysis_records(gains))


def _exposure_readers(ev: _Evaluation) -> str:
    return TRACED if not ev.runs(UNTRACED_LIGHT) else CHAIN if ev.runs(PLANE_BOUNDS) else SOLO


def _exposure_parameters(ev: _Evaluation) -> tuple[str, tuple[str, ...]]:
    """What an exposure report depends on beside the physical inputs: the readers of the return-light analysis before it,
    and the elements a ``power_rating_utilization`` target assesses (an exceeded rating's notice says whether one does;
    never a target's thresholds)."""
    assessed = {t.device_id for t in ev.project.targets if t.metric == POWER_RATING_UTILIZATION and t.device_id is not None}
    return _exposure_readers(ev), tuple(sorted(assessed))


def _surface_exposures_node(ev: _Evaluation) -> _Exposed:
    lowered: _Lowered = ev.product(SCENE)
    beams: _Traced = ev.product(BEAM_TRACE)
    secondary: _Secondary = ev.product(SECONDARY_RAYS)
    budget = ev.product(POWER_LEDGER).budget
    scene, cfg, trace = lowered.scene, ev.cfg, secondary.trace
    readers = _exposure_readers(ev)
    probes = ev.product(UNTRACED_LIGHT).probes if readers != TRACED else None
    gains = _return_gains(ev, readers) if readers != TRACED else None
    untraced = surface_reach(trace, probes, cfg) if probes is not None and _rated(ev) else None
    room = surface_room(scene)
    surface_hits = combine_surface_hits(trace.surface_hits, secondary.sampled)
    issues: list[ValidationIssue] = []
    with phase("exposures"):
        exposure = _exposures(trace, scene, lowered.library, cfg, ev.project, beams.watch, surface_hits, budget, untraced,
                              lowered.metric_scope, _media(ev), gains, room, issues)
    issues.extend(exposure.issues)
    return _Exposed(exposure, tuple(issues), tuple(return_work_issues(gains, scene)), return_analysis_records(gains))


def _pair_selections(ev: _Evaluation) -> tuple[str, ...]:
    """What the declared pair-metric targets select (planes, sources, wavelengths, paths and crossings): the pairs of
    mutually incoherent sources a target names are analysed with the others. A target's thresholds are no part of it."""
    return tuple(sorted({repr((t.observation_id, t.source_id, t.source_id_b, t.wavelength_nm,
                               tuple((key, t.parameters.get(key)) for key in PAIR_PATH_SELECTION)))
                         for t in ev.project.targets if t.metric in PAIR_METRICS and t.observation_id is not None}))


def _beam_pairs_node(ev: _Evaluation) -> _Pairs:
    readings: _Readings = ev.product(PLANE_READINGS)
    interference: list[InterferenceResult] = []
    incoherent: list[IncoherentPairs] = []
    issues: list[ValidationIssue] = []
    with phase("interference", unit="beam pairs"):
        _interference(ev.project, ev.cfg, readings.kept, [o.observation_id for o in ev.project.observations], readings.sources, issues,
                      interference, incoherent)
    truncated = frozenset(i.location.removeprefix("observation:") for i in issues
                          if i.code is ErrorCode.MAX_BEAM_PAIRS and i.location is not None)
    return _Pairs(interference, tuple(incoherent), tuple(issues), truncated)


def _no_parameters(ev: _Evaluation) -> tuple:
    return ()


#: The physics capabilities in execution order, each with the parameters its product depends on and its node.
PHYSICS_NODES: tuple[tuple[str, Callable[[_Evaluation], Any], Callable[[_Evaluation], Any]], ...] = (
    (SCENE, _no_parameters, _scene_node),
    (BEAM_TRACE, lambda ev: (ev.physics.pupil_rays, ev.physics.rated_surfaces), _beam_trace_node),
    (SECONDARY_RAYS, lambda ev: (ev.physics.secondary_rays,), _secondary_rays_node),
    (PLANE_READINGS, _no_parameters, _plane_readings_node),
    (POWER_LEDGER, _no_parameters, _power_ledger_node),
    (UNTRACED_LIGHT, lambda ev: (ev.physics.follow_untraced,), _untraced_light_node),
    (PLANE_BOUNDS, _no_parameters, _plane_bounds_node),
    (SURFACE_EXPOSURES, _exposure_parameters, _surface_exposures_node),
    (BEAM_PAIRS, _pair_selections, _beam_pairs_node),
)


def _run_physics_nodes(ev: _Evaluation) -> None:
    """Run the evaluation's physics capabilities in execution order, each once, over the evaluation's state."""
    if ev.recorder is not None and ev.runs(BEAM_TRACE) and ev.state.has(BEAM_TRACE) and ev.state.recorder is not ev.recorder:
        # (the retained trace was not observed by this recorder; an evaluation that does not trace has nothing to observe
        # and drops nothing)
        ev.state.discard(BEAM_TRACE, RECORDER_CHANGED)
    for capability, parameters, compute in PHYSICS_NODES:
        if ev.runs(capability):
            _node(ev, capability, parameters(ev), compute)


def _last_reader(ev: _Evaluation) -> str | None:
    """The last capability of the evaluation that read the return-light analysis."""
    if ev.runs(SURFACE_EXPOSURES) and ev.runs(UNTRACED_LIGHT):
        return SURFACE_EXPOSURES
    return PLANE_BOUNDS if ev.runs(PLANE_BOUNDS) else None


def _physics_issues(ev: _Evaluation) -> list[ValidationIssue]:
    """The issues of the physics capabilities the evaluation ran, capability by capability in execution order (the order a
    result lists them in, whichever evaluation ran each one), with the return-light analysis' work issue after its last
    reader."""
    reader = _last_reader(ev)
    issues: list[ValidationIssue] = []
    for capability in PHYSICS_CAPABILITIES:
        if ev.runs(capability):
            product = ev.product(capability)
            issues.extend(product.issues)
            if capability == reader:
                issues.extend(product.return_work)
    return issues


def _solver_totals(ev: _Evaluation) -> dict[str, SolverRecord]:
    """Per freeform surface, what the solver did in the capabilities the evaluation ran."""
    totals: dict[str, SolverRecord] = {}
    for capability in PHYSICS_CAPABILITIES:
        if ev.runs(capability):
            for surface_id, (solved, steps, residual, lost) in ev.product(capability).solver.items():
                before = totals.get(surface_id, (0, 0, 0.0, 0))
                totals[surface_id] = (before[0] + solved, max(before[1], steps), max(before[2], residual), before[3] + lost)
    return totals


def _result_trace(ev: _Evaluation) -> TraceOutput | None:
    """The trace a result reports: the beam trace, with the secondary rays when the evaluation ran them."""
    if ev.runs(SECONDARY_RAYS):
        return ev.product(SECONDARY_RAYS).trace
    return ev.product(BEAM_TRACE).trace if ev.runs(BEAM_TRACE) else None


def run_physics(
    project: ProjectSnapshot,
    cfg: NumericalConfig,
    *,
    plan: PhysicsPlan = COMPLETE_PHYSICS,
    component_poses: dict[str, Pose] | None = None,
    recorder: TraceObserver | None = None,
    quad_rule: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None,
) -> PhysicsOutcome:
    """Lower the scene, trace it, and derive observation and interference results: the physics capabilities of ``plan``,
    each run once by the node an evaluation runs (``evaluate``), over a state of this call's own.

    ``plan``: ``COMPLETE_PHYSICS``, or one of the Engine's own compositions -- ``BEAMS_ALONE`` (the network export),
    ``SENSOR_READINGS`` (the control analysis), ``CHIEF_RAYS`` -- which run every physics capability but the beam pairs.
    ``recorder`` observes the trace step by step (``validation/ray_paths.py``); it cannot change the trace.
    ``quad_rule``: the Gauss-Legendre rule of ``cfg.aperture_quad_sector_nodes`` nodes, from a caller that traces many times
    in one validation call (control readings) and builds it once."""
    ev = _Evaluation(project, cfg, EvaluationState(), tuple(c for c in PHYSICS_CAPABILITIES if c in plan.capabilities), plan,
                     recorder, quad_rule, component_poses)
    _run_physics_nodes(ev)
    lowered: _Lowered = ev.product(SCENE)
    secondary: _Secondary = ev.product(SECONDARY_RAYS)
    readings: _Readings = ev.product(PLANE_READINGS)
    bounds: _Bounds | None = ev.product(PLANE_BOUNDS) if ev.runs(PLANE_BOUNDS) else None
    exposed: _Exposed | None = ev.product(SURFACE_EXPOSURES) if ev.runs(SURFACE_EXPOSURES) else None
    pairs: _Pairs | None = ev.product(BEAM_PAIRS) if ev.runs(BEAM_PAIRS) else None
    return PhysicsOutcome(
        lowered.scene, secondary.trace, bounds.observations if bounds is not None else readings.observations,
        pairs.interference if pairs is not None else [], _physics_issues(ev),
        ev.product(POWER_LEDGER).budget if ev.runs(POWER_LEDGER) else None, bounds.reach if bounds is not None else None,
        exposed.exposure.report if exposed is not None else None, lowered.metric_scope,
        exposed.exposure.element_scope if exposed is not None else {},
        sampled_sources(lowered.scene, secondary.sampled) + scattered_sources(secondary.scattered),
        _return_gains(ev, CHAIN) if ev.runs(UNTRACED_LIGHT) else None, pairs.incoherent if pairs is not None else (),
        readings.spectra, element_power_scope=exposed.exposure.power_scope if exposed is not None else {})


def _exposures(trace, scene, library, cfg, project, watch, surface_hits, power_budget, untraced, metric_scope, media, gains, room,
               issues):
    """The per-surface exposures and power-rating comparisons, with the deeper probe passes they ask for
    (``_surface_exposures_node``)."""
    # (the crystals whose second-harmonic generation is out of model scope: the reason code of the light behind them)
    converters = frozenset(c.surface_id for c in trace.nonlinear_conversions)
    exposure = build_exposure_report(surface_hits, project, cfg, power_budget, untraced, metric_scope, media, gains,
                                     room, converters)
    passes = 0
    while untraced is not None and exposure.deeper_levels and passes < MAX_DEEPER_PASSES:
        # light left out as negligible alone keeps a rating comparison from being decided (its diffraction limit, or its
        # share of the light for a condition): follow those lines' light deeper, down to the level the decision needs,
        # and rebuild the exposures -- unless a pass runs out of its budget, meets light whose fate it cannot follow, or
        # leaves no less light out (the sum of its pieces' amplitudes, which the bounds use; then the last pass stands,
        # bounded). A pass whose light meets the interaction limit found light that may be stored in a cavity: it stands
        # (that light has no upper bound), and no pass follows
        passes += 1
        lines = set(exposure.deeper_levels)
        deeper = follow_untraced(trace, scene, library, cfg, watch=watch, levels=exposure.deeper_levels)
        issues.extend(probe_budget_issues(deeper, cfg, lines))
        if deeper.budget_reached:
            break
        found = surface_reach(trace, deeper, cfg)
        stored = any(kind in STORED_KINDS for line in lines for kind in found.unfollowed.get(line, {}))
        if not stored and (any(line in found.unfollowed for line in lines) or all(
                found.left_out_root.get(line, 0.0) >= untraced.left_out_root.get(line, 0.0) for line in lines)):
            break
        untraced = merge_reach(untraced, found, lines)
        exposure = build_exposure_report(surface_hits, project, cfg, power_budget, untraced, metric_scope, media,
                                         gains, room, converters)
        if stored:
            break
    return exposure


def _interference(project, cfg, kept, obs_ids, sources, issues, interference, incoherent) -> None:
    """The beam pairs of every plane (``_beam_pairs_node``): the pairs that can interfere analysed into ``interference``, the
    pairs of mutually incoherent sources counted into ``incoherent``, their issues into ``issues``."""
    by_obs: dict[str, list[ArrivalRecord]] = {}
    for rec in kept:
        if rec.power_bound_W is None and not rec.polarization_open:  # (a beam of unknown field forms no pair)
            by_obs.setdefault(rec.observation_id, []).append(rec)
    pair_targets: dict[str, list] = {}
    for t in project.targets:
        if t.metric in PAIR_METRICS and t.observation_id is not None:
            pair_targets.setdefault(t.observation_id, []).append(t)
    for oid in obs_ids:
        recs = sorted(by_obs.get(oid, []), key=lambda r: (r.path_id, r.crossing_index, r.spectral_index))
        # beams of different wavelengths never interfere, and beams of mutually incoherent sources form no fringe: the
        # same-wavelength pairs of one coherence group (and the pairs a target selects) are analysed and counted against
        # the cap (validation/pair_selection.py), the others counted. The pairs are enumerated lazily, so a plane many
        # beams reach costs the cap, not every pair. The two polarization components of a partially polarized line never
        # interfere: no pair of them is formed
        same = (pair for pair in plane_pairs(recs, pair_targets.get(oid, ()), cfg) if not component_pair(*pair))
        pairs = list(itertools.islice(same, cfg.max_interference_pairs + 1))
        listed = len(pairs)
        if len(pairs) > cfg.max_interference_pairs:
            pair_count = listed = len(pairs) + sum(1 for _ in same)
            # (a notice: the pair listing is an auxiliary analysis; the planes it truncates are named in the run's budget
            # record and a pair target beyond the cap is UNKNOWN by itself)
            issues.append(notice(ErrorCode.MAX_BEAM_PAIRS,
                                 f"{pair_count} beam pairs reach this plane; only the first {cfg.max_interference_pairs} are analysed",
                                 location=f"observation:{oid}", pair_count=pair_count, limit=cfg.max_interference_pairs,
                                 suggested_action="raise compute_budget.max_interference_pairs to analyse the other pairs"))
            pairs = pairs[: cfg.max_interference_pairs]
        for a, b in pairs:
            res, pair_issues = analyse_pair(a, b, cfg, sources)
            interference.append(res)
            issues.extend(pair_issues)
            checkpoint()
        # (the two components of a partially polarized line are counted on their own: they are one source's light)
        components = component_pair_count(recs, cfg)
        unlisted = same_wavelength_pair_count([r.wavelength_nm for r in recs], cfg) - listed - components
        if components > 0:
            issues.append(make_issue(
                ErrorCode.PARTIAL_POLARIZATION_COMPONENTS, Status.NOT_APPLICABLE,
                f"same-wavelength beam pairs of the two polarization components of a partially polarized source's line on "
                f"this plane: {components}; the two components of one line are mutually incoherent (an unpolarized share and "
                "a polarized one), form no fringes and their powers add, so they are counted here, not analysed pair by pair",
                location=f"observation:{oid}", observation_id=oid, pair_count=components))
        if unlisted > 0:
            # (the sources' coherence groups; where the sources share one, the group of a partially polarized source's
            # unpolarized share -- incoherent with the other sources -- is the second)
            groups = len({_source_group(r.coherence_group) for r in recs})
            if groups < 2:
                groups = len({r.coherence_group for r in recs})
            incoherent.append(IncoherentPairs(observation_id=oid, pair_count=unlisted, coherence_group_count=groups))
            issues.append(make_issue(
                ErrorCode.INCOHERENT_SOURCES, Status.NOT_APPLICABLE,
                f"same-wavelength beam pairs of mutually incoherent sources on this plane: {unlisted} ({groups} coherence "
                "groups); they form no fringes and their irradiances add, so they are counted in incoherent_pairs, not "
                "analysed pair by pair (a pair-metric target naming two of these sources analyses its pair)",
                location=f"observation:{oid}", observation_id=oid, pair_count=unlisted, coherence_group_count=groups))


def _source_group(group: str) -> str:
    """A coherence group without the mark of a partially polarized line's second component (its source's own group)."""
    return group[: -len(GROUP_MARK)] if group.endswith(GROUP_MARK) else group


def component_pair_count(recs: list[ArrivalRecord], cfg: NumericalConfig) -> int:
    """The same-wavelength pairs of a plane's beams that are the two polarization components of one partially polarized
    line (``optics/polarization/partial.py``: ``component_pair``), counted per (source, line) with each component's beams."""
    counts: dict[tuple[str, int, float], list[int]] = {}
    for r in recs:
        if r.polarization_component is not None:
            key = (r.source_id, r.spectral_index, r.wavelength_nm)
            counts.setdefault(key, [0, 0])[1 if r.polarization_component else 0] += 1
    total = 0
    for first, second in counts.values():
        total += first * second  # (integers)
    return total


def return_work_issues(gains: Mapping[float, ReturnGains] | None, scene: Scene) -> list[ValidationIssue]:
    """One ``RETURN_ANALYSIS_BUDGET`` notice (``NOT_APPLICABLE``) where the return-light analysis of some wavelength reached
    the run's work limit (``NumericalConfig.max_return_analysis_work``) and a bound read it: whether the left-out light may
    circulate is not known there, so the bounds that read it are withheld (``ReturnGains.withheld_by_work``). The analysis
    is auxiliary: the quantities that lost a bound carry the fact (a plane's ``power_upper_bound_W`` null, an exposure's
    own rating status), and the notice raises no status. It names the wavelengths, the work done and the run's
    limit, and exactly the bounds withheld: the power upper bounds of the planes whose power read the crossings
    (``ReturnGains.power_withheld_by_work``: a transparent plane; an absorbing plane's power takes each watt once and stays
    reported), the coherent-power and irradiance upper bounds of the planes whose field read them, and the exposure upper
    bounds of the surfaces; the execution stays ``COMPLETED`` with the budget listed in ``budgets.exhausted`` (phase
    ``return_analysis``, ``validation/outcome.py``)."""
    if not isinstance(gains, ReturnAnalysis):
        return []
    spent = [g for g in gains.analysed if not g.screened and g.withheld_by_work]
    if not spent:
        return []
    planes = {s.surface_id: s.interaction.observation_id for s in scene.surfaces if s.interaction.kind == "observe"}
    read = set().union(*(g.withheld_by_work for g in spent))
    power_read = set().union(*(g.power_withheld_by_work for g in spent))
    observations = sorted({planes[s] for s in read if s in planes})
    power_planes = sorted({planes[s] for s in power_read if s in planes})
    surfaces = sorted(s for s in read if s and s not in planes)
    first = spent[0].work
    limit = None if first.limit is None else first.before + first.limit  # (the run's limit)
    wavelengths = ", ".join(f"{g.wavelength_nm:g}" for g in spent)
    held = "; ".join(part for part in (
        f"the power upper bounds of {', '.join(power_planes)}" if power_planes else "",
        f"the coherent-power and irradiance upper bounds of {', '.join(observations)}" if observations else "",
        f"the exposure upper bounds of {', '.join(surfaces)}" if surfaces else "") if part) or "the upper bounds that read it"
    action = "raise compute_budget.max_return_analysis_work to obtain these bounds"
    return [make_issue(
        ErrorCode.RETURN_ANALYSIS_BUDGET, Status.NOT_APPLICABLE,
        f"the return-light analysis at {wavelengths} nm reached the run's work limit of {limit} units before it could bound "
        f"how often surfaces facing each other or a closed path of three or more surfaces may return the light left out of "
        f"the trace: {held} are withheld",
        phase="return_analysis", budget=["max_return_analysis_work"], limit=limit,
        wavelength_nm=[g.wavelength_nm for g in spent], work_done=[g.work.steps for g in spent],
        stage=[g.work.stage for g in spent],
        affected_observation_ids=observations, power_bound_observation_ids=power_planes, affected_surface_ids=surfaces,
        suggested_action=action)]


def return_analysis_records(gains: Mapping[float, ReturnGains] | None) -> tuple[ReturnAnalysisRecord, ...]:
    """The record of each wavelength the return-light analysis ran for (``BudgetUsage.return_analysis``), by ascending
    wavelength: its closed-return screen's answer (``ReturnGains.closed_return_screen``), where and why it ended, its work
    and its trail storage. None for a run whose bounds read no analysis."""
    if not isinstance(gains, ReturnAnalysis):
        return ()
    out = []
    for g in gains.analysed:
        search = g.search
        out.append(ReturnAnalysisRecord(
            wavelength_nm=g.wavelength_nm, closed_return_screen=g.closed_return_screen,
            ended_in=g.work.stage if g.work.exhausted and g.work.stage is not None else "closed_returns",
            termination=search.termination, work_done=g.work.steps, work_limit=g.work.limit, search_depth=search.depth,
            frames_peak=search.frames_peak, frames_limit=search.frames_limit,
            closed_return_surfaces=g.closed_cycles if not g.work.exhausted else (),
            closed_series_bound=g.closed_series if g.closed_cycles and not g.work.exhausted else None,
            work_before=g.work.before, solid_fold_work=g.work.fold_steps,
            closed_return_kinds=g.closed_return_kinds if not g.work.exhausted else ()))
    return tuple(out)


def plane_bound_notices(gains: Mapping[float, ReturnGains] | None, reach: PlaneReach | None, scene: Scene, trace: TraceOutput,
                        observations: list[ObservationResult]) -> list[ValidationIssue]:
    """The run-level notices of a plane whose power upper bound a pair or a closed return withholds
    (``EXPOSURE_BOUND_WITHHELD``, status ``NOT_APPLICABLE``; a run with no rating has no surface to carry one): no bound, target or
    status moves. A plane without an upper bound through light at the interaction limit has that light's own issue."""
    if not gains or not reach:
        return []
    planes = {s.interaction.observation_id: s.surface_id for s in scene.surfaces if s.interaction.kind == "observe"}
    wavelengths = {(i.source_id, i.spectral_index): i.wavelength_nm for i in trace.inputs}
    out: list[ValidationIssue] = []
    for observation in observations:
        oid = observation.observation_id
        if observation.power_upper_bound_W is not None or oid not in reach or oid not in planes:
            continue
        found = sorted({wavelengths[line] for line, kinds in reach[oid].items()
                        if line in wavelengths and not math.isfinite(kinds.get(FAINT_POWER, 0.0))})
        held = [w for w in found if w in gains and not math.isfinite(gains[w].factor(planes[oid])) and unbounded_pairs(gains[w])]
        if held:
            pairs, named, counted = named_returns(sorted({pair for w in held for pair in unbounded_pairs(gains[w])}),
                                                  lambda pair: f"{pair[0]} and {pair[1]}", "pairs of surfaces")
            action, facts = pair_facts(pairs, [record for w in held for record in gains[w].saturated_pairs()])
            out.append(make_issue(
                ErrorCode.EXPOSURE_BOUND_WITHHELD, Status.NOT_APPLICABLE,
                f"The power upper bound on the plane {oid} is withheld at {', '.join(f'{w:g}' for w in held)} nm because {named} return left-out "
                f"light with a round-trip gain bound of 1.",
                location=f"observation:{oid}", observation_id=oid, quantity="power_upper_bound", wavelength_nm=held,
                return_pair_surfaces=[", ".join(pair) for pair in pairs], cause="RETURN_PAIR_GAIN_BOUND_1", suggested_action=action,
                **counted, **facts))
        closed = [w for w in found if w in gains and gains[w].closed_cycles and gains[w].closed_series is None]
        if closed:
            kinds = closed_return_kinds(gains, closed)
            cycles, named, counted = named_returns(sorted({cycle for w in closed for cycle in gains[w].closed_cycles}),
                                                   closed_return_text(kinds, ", "), "closed paths")
            counted = {**counted, **kinds_context(cycles, kinds)}
            # (the search names the shortest closed return it finds; the groups it screens say where any other can be: a fact of
            # the message. No step inside the tool obtains this bound, so the notice has no action)
            where, facts = closed_return_facts(gains, closed)
            out.append(make_issue(
                ErrorCode.EXPOSURE_BOUND_WITHHELD, Status.NOT_APPLICABLE,
                f"The power upper bound on the plane {oid} is withheld at {', '.join(f'{w:g}' for w in closed)} nm because no series bounds the "
                f"left-out light's closed return through {named}; {where}.",
                location=f"observation:{oid}", observation_id=oid, quantity="power_upper_bound", wavelength_nm=closed,
                closed_return_surfaces=[", ".join(cycle) for cycle in cycles], cause="CLOSED_RETURN_NOT_BOUNDED",
                return_scope=SCENE_WIDE, **counted, **facts))
    return out


def _uses(trace: TraceOutput, scene: Scene, library: MaterialLibrary, cfg: NumericalConfig, probes: Probes):
    """How the left-out light of a probe pass meets coatings from their dense sides (``validation/coating_use.py``)."""
    if not any(declared_air_equivalent(s) is not None for s in scene.surfaces):
        return {}
    hits = [*trace.surface_hits, *(hit for _, outcome in probes.results for hit in outcome.surface_hits)]
    open_lines = frozenset(line for line, kinds in surface_reach(trace, probes, cfg).unfollowed.items() if kinds)
    return coating_uses(scene, library, cfg, hits, {(i.source_id, i.spectral_index): i.power_W for i in trace.inputs}, open_lines)


def diffraction_roots(project: ProjectSnapshot, scene: Scene,
                      media: Callable[[str, float], float]) -> Callable[[str, tuple[str, int]], float]:
    """(surface id, source line) -> the square root of the diffraction limit of 1 W of that line on the surface (W/mm^2):
    ``sqrt(2 pi s) n / lambda`` -- of ``I_max = 2 pi n^2 P s / lambda^2`` (``validation/exposures.py``; ``n`` the larger
    index of the surface's media, ``s`` the line's spectral factor)."""
    pulses = {c.source.source_id: c.source.pulse for c in project.components if c.source is not None}
    lines = {}
    counts = {}
    sources = {}
    for src in (*scene.sources, *scene.emitters):
        counts[src.source_id] = len(src.spectrum)
        sources[src.source_id] = src
        for index, (wavelength_nm, _weight) in enumerate(src.spectrum):
            length = line_coherence_length_mm(src, index)
            lines[(src.source_id, index)] = (wavelength_nm, spectral_factor(wavelength_nm, length, pulses.get(src.source_id)))

    def line_of(line: tuple[str, int]) -> tuple[float, float]:
        # a line a crystal generates (index = its fundamental's + the source's line count, half the wavelength)
        if line not in lines:
            source_id, index = line
            wavelength_nm, _spectral = line_of((source_id, index - counts[source_id]))
            length = line_coherence_length_mm(sources[source_id], index)
            lines[line] = (wavelength_nm / 2.0, spectral_factor(wavelength_nm / 2.0, length, pulses.get(source_id)))
        return lines[line]

    def of(surface_id: str, line: tuple[str, int]) -> float:
        wavelength_nm, spectral = line_of(line)
        return math.sqrt(2.0 * math.pi * spectral) * media(surface_id, wavelength_nm) / (wavelength_nm * NM_TO_MM)

    return of


def media_index(scene: Scene, library: MaterialLibrary) -> Callable[[str, float], float]:
    """(surface id, vacuum wavelength nm) -> the larger refractive index of the surface's two media there (the exposures'
    diffraction limit of light left out as negligible); a medium whose index cannot be evaluated there counts as the
    largest index of the scene's media that can be (1 when none can)."""
    by_id = {surface.surface_id: (surface.medium_pos, surface.medium_neg) for surface in scene.surfaces}
    media = sorted({m for pair in by_id.values() for m in pair} | {source.medium for source in (*scene.sources, *scene.emitters)})
    cache: dict[tuple[str, float], float | None] = {}

    def index(medium: str, wavelength_nm: float) -> float | None:
        key = (medium, wavelength_nm)
        if key not in cache:
            try:
                n, issue = library.evaluator(medium).index(wavelength_nm, location=f"medium:{medium}")
                crystal = library.uniaxial(medium)
                if crystal is not None and issue is None and n is not None:  # (a crystal: the larger of its two indices)
                    n_e, issue = crystal.extraordinary_index.index(wavelength_nm, location=f"medium:{medium}")
                    n = None if n_e is None else max(n, n_e)
            except EngineInputError:
                n, issue = None, None
            cache[key] = None if issue is not None or n is None else float(n)
        return cache[key]

    def largest(wavelength_nm: float) -> float:
        known = [n for n in (index(m, wavelength_nm) for m in media) if n is not None]
        return max(known) if known else 1.0

    def of(surface_id: str, wavelength_nm: float) -> float:
        pair = by_id.get(surface_id)
        values = [index(m, wavelength_nm) for m in pair] if pair is not None else [None]
        return largest(wavelength_nm) if any(n is None for n in values) else max(values)  # type: ignore[type-var]

    return of


def _failure_result(text_hash: str | None, cfg: NumericalConfig, issue: ValidationIssue, request_hash: str | None = None, *,
                    failure: TerminationReason = TerminationReason.INVALID_INPUT,
                    declared: bool | None = None, build_id: str | None = None) -> ValidationResult:
    result = ValidationResult(
        status=issue.status, summary=ValidationSummary(issue_counts={issue.code.value: 1}), issues=(issue,),
        outcome=result_outcome((issue,), cfg, (), failed=failure, reason=issue.message, declared=declared),
    )
    manifest = _manifest(result, cfg, request_hash or text_hash or "", "", "", None, None, [], build_id=build_id)
    return result.model_copy(update={"manifest": manifest})


def _manifest(
    result: ValidationResult,
    cfg: NumericalConfig,
    input_hash: str,
    project_hash: str,
    projection_hash: str,
    trace_events: list[dict] | None,
    seed: int | None,
    issues: list[ValidationIssue],
    project_schema_version: str = SCHEMA_VERSION,
    build_id: str | None = None,
) -> ValidationManifest:
    # (``build_id``: the Engine build an ``EvaluationState`` identified once; without a state it is computed here)
    # (the result's quantized hash, its long arrays dumped and hashed element by element: never the whole document at once)
    body = model_parts(result, update={"manifest": None})
    tolerances = {k: float(v) for k, v in cfg.model_dump().items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    # the WARNING issues, and the rating notices (informational: they do not change the run's status)
    warnings = tuple(sorted({f"{i.code.value}: {i.message}" for i in issues
                             if i.status is Status.WARNING or i.code in MANIFEST_NOTICE_CODES}))
    return ValidationManifest(
        engine_version=ENGINE_VERSION,
        engine_build_id=build_id if build_id is not None else engine_build_id(),
        schema_version=SCHEMA_VERSION,
        project_schema_version=project_schema_version,
        numerical_config_version=NUMERICAL_CONFIG_VERSION,
        input_hash=input_hash,
        project_hash=project_hash,
        design_projection_hash=projection_hash,
        result_hash=hash_quantized_parts(body, cfg.hash_float_sig_digits, cfg.hash_float_abs_floor),
        trace_hash=None if trace_events is None else hash_quantized(trace_events, cfg.hash_float_sig_digits, cfg.hash_float_abs_floor),
        random_seed=seed,
        prng_id=PRNG_ID if seed is not None else None,
        model_versions=dict(MODEL_VERSIONS),
        numerical_tolerances=tolerances,
        warnings=warnings,
        model_scope=tuple(MODEL_SCOPE_SUPPORTED) + tuple(f"unsupported:{k}" for k in sorted(OUT_OF_SCOPE_METRICS)),
        runtime=runtime_info(),
    )


def _surface_profiles(scene: Scene, solver: dict[str, SolverRecord]) -> tuple[tuple[SurfaceProfileEvidence, ...], list[ValidationIssue]]:
    """The evidence of every surface with a two-variable sag shape (``solver``: what the intersection solver did on each in
    the capabilities the evaluation ran, ``_solver_totals``), and an issue for each one where the solver did not converge
    for some ray: such a ray passed the surface as if it missed it, so the result is not known."""

    profiles, issues = [], []
    for surface in scene.surfaces:
        shape = surface.freeform
        if shape is None:
            continue
        solved_rays, max_iterations, max_residual_mm, non_converged_rays = solver.get(surface.surface_id, (0, 0, 0.0, 0))
        profiles.append(SurfaceProfileEvidence(
            surface_id=surface.surface_id, kind=shape.kind, indexing=shape.indexing, normalization=shape.normalization,
            normalization_radius_mm=shape.normalization_radius, aperture_reach_mm=shape.aperture_reach,
            vertex_sag_mm=shape.vertex_sag, solved_rays=solved_rays,
            max_iterations=max_iterations, max_residual_mm=max_residual_mm, non_converged_rays=non_converged_rays))
        if non_converged_rays:
            issues.append(make_issue(
                ErrorCode.SAG_INTERSECTION_NOT_CONVERGED, Status.UNKNOWN,
                f"the intersection of {non_converged_rays} ray(s) with {surface.surface_id} did not converge within "
                f"{max_iterations} steps: those rays continued as if they missed the surface",
                location=f"surface:{surface.surface_id}", non_converged_rays=non_converged_rays,
                suggested_action="raise sag_max_iterations in the numerical configuration, or reduce the shape's steepest terms"))
    return tuple(profiles), issues


def _schema_ok(version: str) -> bool:
    return version.split(".")[0] == SCHEMA_VERSION.split(".")[0]


def quadrature_refusal(result: ValidationResult) -> int | None:
    """The node count of ``aperture_quad_sector_nodes`` an evaluation was refused for (its rule needs more memory than the
    machine gives: a FAIL with an ``INVALID_INPUT`` issue at ``QUADRATURE_SETTING``), else ``None``: a caller that states
    the setting under another name (a Project's ``numerical_settings``) refuses it by that name."""
    for issue in result.issues:
        nodes = issue.numerical_context.get("nodes")
        if issue.code is ErrorCode.INVALID_INPUT and issue.location == QUADRATURE_SETTING and isinstance(nodes, int):
            return nodes
    return None


def validate(request: ValidationRequest) -> ValidationResult:
    """Run a full validation: the complete evaluation, every capability the request's configuration enables. Never raises
    for physical or contract problems; see module docstring."""
    return evaluate(request)


def evaluate(request: ValidationRequest, *, capabilities: Iterable[str] | str = (COMPLETE,), exclude: Iterable[str] | str = (),
             targets: Iterable[str] | None = None, state: EvaluationState | None = None,
             recorder: TraceObserver | None = None) -> ValidationResult:
    """Evaluate ``request`` for the capabilities asked for (``os_optics.evaluation``: capability ids and the presets
    ``complete`` and ``readings``), with what they require.

    ``validate(request)`` is ``evaluate(request)``: the complete evaluation, every capability run by the node a selective
    evaluation runs. A selection runs the capabilities it names and their prerequisites (``plan_evaluation``); a
    capability outside that scope is not run -- its phase is never entered -- and what it would have produced is reported
    as not computed: the result's ``evaluation`` names the requested and the executed scope, each capability of the
    request's complete scope that did not run with the sections it produces, and each declared target the caller did not
    select. A plane whose bounds did not run reports its traced power with ``quantity_state`` ``NOT_COMPUTED``. Every
    target the evaluation was asked to assess has a result: the plan runs the capabilities its metric reads; one no traced
    beam matches where the plane bounds did not run is undetermined (``UNKNOWN``, issue code ``CAPABILITY_NOT_COMPUTED``),
    and one on a stage the request's configuration turns off reads as in ``validate``.

    ``exclude``: capabilities that must not run; a selection that requires one is refused. ``targets``: the declared
    targets to evaluate (naming targets requests the requirements). ``state``: an ``EvaluationState`` the caller holds:
    each capability's product is kept in it and reused by the next evaluation of the same physical inputs, numerical
    configuration and Engine build, so a follow-up evaluation runs only what the state does not hold, a changed target
    threshold runs the requirements alone, and the result equals a fresh evaluation of the same scope. ``recorder``
    observes the beam trace and the extended emitters' rays of its secondary rays (``validation/ray_paths.py``) and holds
    the record of the latest of each;
    a retained trace it did not observe is traced again when the evaluation runs the beam trace, and an evaluation that
    does not trace leaves the state as it is.

    Raises ``EvaluationPlanError`` for a selection that cannot be planned (an unknown id, a capability the request does not
    enable, an undeclared target, a required capability that is excluded); never for physical or contract problems.
    """
    plan = plan_evaluation(capabilities, exclude=exclude, targets=targets, request=request, what="capabilities")
    return _evaluate_request(request, plan, state, recorder)


def _evaluate_request(request: ValidationRequest, plan: EvaluationPlan, state: EvaluationState | None,
                      recorder: TraceObserver | None) -> ValidationResult:
    cfg = request.configuration.numerical
    project = request.project
    input_hash = hash_exact(request.model_dump(mode="json"))
    build_id = None
    if state is not None:
        # (the Engine build is identified once per state: its products belong to that build)
        if state.build_id is None:
            state.build_id = engine_build_id()
        build_id = state.build_id
        # (what this evaluation ran and read: nothing, until it runs -- also when it is refused below)
        state.last_executed = state.last_reused = ()
    if not _schema_ok(project.project_schema_version):
        return _failure_result(None, cfg, make_issue(
            ErrorCode.UNSUPPORTED_SCHEMA_VERSION, Status.FAIL,
            f"project_schema_version {project.project_schema_version!r} is not supported (engine schema {SCHEMA_VERSION})",
            supported=SCHEMA_VERSION, received=project.project_schema_version), input_hash, declared=bool(project.targets),
            build_id=build_id)
    project_dump = project.model_dump(mode="json")
    project_hash = hash_exact(project_dump)
    projection_hash = hash_exact({k: project_dump[k] for k in ("baseplate", "components", "observations", "materials")})
    if state is None:
        state = EvaluationState()
    else:
        assert build_id is not None
        state.bind(PhysicalIdentity(build_id, projection_hash, hash_exact(cfg.model_dump(mode="json")),
                                    project.project_schema_version))
    physics = PhysicsPlan(capabilities=frozenset(c for c in plan.executed if c in PHYSICS_CAPABILITIES))
    ev = _Evaluation(project, cfg, state, plan.executed, physics, recorder, seed=request.configuration.random_seed)
    try:
        return _evaluate_checked(request, plan, ev, input_hash, project_hash, projection_hash, build_id)
    except EngineInputError as exc:
        # (a valid request outside the traced model is refused as out of the model's scope, not as invalid input)
        scope = isinstance(exc, ModelScopeRefusal)
        issue = make_issue(exc.code, Status.OUT_OF_MODEL_SCOPE if scope else Status.FAIL, exc.message, location=exc.location,
                           **{k: v for k, v in exc.context.items() if isinstance(v, (int, float, str, bool, list))})
        return _failure_result(None, cfg, issue, input_hash, declared=bool(project.targets), build_id=build_id,
                               failure=TerminationReason.OUT_OF_MODEL_SCOPE if scope else TerminationReason.INVALID_INPUT)
    except QuadratureRuleTooLarge as exc:  # (a node count whose rule the machine cannot hold: the caller's setting, named)
        issue = make_issue(ErrorCode.INVALID_INPUT, Status.FAIL, QuadratureRuleTooLarge.refusal(QUADRATURE_SETTING, exc.nodes),
                           location=QUADRATURE_SETTING, nodes=exc.nodes)
        return _failure_result(None, cfg, issue, input_hash, declared=bool(project.targets), build_id=build_id)
    except ValidationError as exc:  # a non-finite / invalid number reached a result contract
        errors = [f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:5]]
        stage = exc.title or "result"
        quantity = ".".join(str(p) for p in exc.errors()[0]["loc"]) if exc.errors() else "a value"
        issue = make_issue(ErrorCode.NUMERICAL_FAILURE, Status.FAIL,
                           f"the {stage} record received a non-finite or invalid number at {quantity[:MAX_FIRST_ERROR_CHARS]}, so no result is stated",
                           errors=errors, stage=stage, quantity=quantity)
        return _failure_result(None, cfg, issue, input_hash, failure=TerminationReason.NUMERICAL_FAILURE,
                               declared=bool(project.targets), build_id=build_id)
    except (ArithmeticError, ValueError, RecursionError) as exc:  # e.g. a position of 1e300 mm or a waist of 1e-300 mm
        issue = make_issue(ErrorCode.NUMERICAL_FAILURE, Status.FAIL, NUMERICAL_FAILURE_MESSAGE, error=type(exc).__name__)
        return _failure_result(None, cfg, issue, input_hash, failure=TerminationReason.NUMERICAL_FAILURE,
                               declared=bool(project.targets), build_id=build_id)
    finally:
        state.last_executed, state.last_reused = tuple(ev.ran), tuple(ev.reused)


def _layout_node(ev: _Evaluation) -> _LaidOut:
    try:
        from ..layout import run_layout_checks
    except ImportError as exc:  # pragma: no cover - environment without shapely
        return _LaidOut(None, (make_issue(ErrorCode.DEPENDENCY_UNAVAILABLE, Status.UNKNOWN, "layout checks need the 'shapely' package",
                                          detail=str(exc)),))
    return _LaidOut(run_layout_checks(ev.project, ev.cfg, _result_trace(ev)), ())


def _control_node(ev: _Evaluation) -> _Controlled:
    from ..control import run_control_analysis

    spec = ev.project.control_spec
    assert spec is not None
    control, issues = run_control_analysis(ev.project, ev.cfg, ev.seed)
    seed = ev.seed
    if spec.monte_carlo is not None:  # the seed its samples were drawn with (the documented default when none is stated)
        seed = (control.monte_carlo_seed if control.monte_carlo_seed is not None
                else spec.monte_carlo.seed if spec.monte_carlo.seed is not None else seed)
    return _Controlled(control, tuple(issues), seed)


def _observations(ev: _Evaluation) -> list[ObservationResult]:
    """The planes as the evaluation's targets read them: with their bounds where the plane bounds ran, else the readings
    (with the light of a partially polarized source's routes where the ledger ran)."""
    if ev.runs(PLANE_BOUNDS):
        return ev.product(PLANE_BOUNDS).observations
    if not ev.runs(PLANE_READINGS):
        return []
    observations = ev.product(PLANE_READINGS).observations
    return with_polarization_mixtures(observations, ev.product(POWER_LEDGER).budget, ev.cfg) if ev.runs(POWER_LEDGER) else observations


def _requirements(plan: EvaluationPlan, ev: _Evaluation, observations: list[ObservationResult],
                  layout: Any, control: Any) -> tuple[tuple[Target, ...], list[TargetResult], tuple[TargetNotEvaluated, ...]]:
    """The declared targets the evaluation assesses, their results, and the declared targets the caller did not select
    (all of them, when the requirements were not requested), in Project order. Every target that is assessed has a result:
    one whose metric reads a stage the request turns off is assessed on what the run has (it reads as in ``validate``), and
    one no traced beam matches where the plane bounds did not run is undetermined."""
    project, cfg = ev.project, ev.cfg
    by_id = {t.target_id: t for t in project.targets}
    skipped = tuple(TargetNotEvaluated(target_id=target_id, metric=by_id[target_id].metric, reason=NOT_REQUESTED)
                    for target_id in plan.unselected_targets)
    evaluated: tuple[Target, ...] = ()
    results: list[TargetResult] = []
    if ev.runs(REQUIREMENTS):
        lowered: _Lowered | None = ev.product(SCENE) if ev.runs(SCENE) else None
        beams: _Traced | None = ev.product(BEAM_TRACE) if ev.runs(BEAM_TRACE) else None
        exposed: _Exposed | None = ev.product(SURFACE_EXPOSURES) if ev.runs(SURFACE_EXPOSURES) else None
        pairs: _Pairs | None = ev.product(BEAM_PAIRS) if ev.runs(BEAM_PAIRS) else None
        detections = {o.observation_id: o.detection for o in project.observations if o.detection is not None}
        evaluated = tuple(by_id[requirement.target_id] for requirement in plan.targets)
        with phase("targets", total=len(evaluated), unit="targets"):
            results = assess_targets(
                evaluated, {o.observation_id: o for o in observations}, pairs.interference if pairs is not None else [],
                layout, control, cfg, ev.product(POWER_LEDGER).budget if ev.runs(POWER_LEDGER) else None,
                ev.product(PLANE_BOUNDS).reach if ev.runs(PLANE_BOUNDS) else None,
                exposed.exposure.report if exposed is not None else None,
                metric_scope=lowered.metric_scope if lowered is not None else {},
                truncated_pairs=pairs.truncated if pairs is not None else frozenset(),
                element_scope=exposed.exposure.element_scope if exposed is not None else {},
                element_power_scope=exposed.exposure.power_scope if exposed is not None else {},
                attenuators=frozenset(s.surface_id for s in lowered.scene.surfaces
                                      if s.interaction.kind == "neutral_density") if lowered is not None else None,
                met_first={(rec.observation_id, rec.path_id, rec.crossing_index): (rec.met_first, rec.left_beyond)
                           for rec in beams.trace.arrivals if rec.met_first or rec.left_beyond}
                if beams is not None else None,
                actuators={c.instance_id: c.actuators for c in project.components if c.actuators},
                detections=detections,
                conversions=tuple(beams.trace.nonlinear_conversions) if beams is not None else (),
                # (the plane bounds left out of an evaluation whose request enables them; a request that does not trace
                # has none to leave out, and its targets read as in ``validate``)
                plane_bounds=ev.runs(PLANE_BOUNDS) or PLANE_BOUNDS not in plan.complete_scope)
        ev.state.count(REQUIREMENTS)
        ev.ran.append(REQUIREMENTS)
    return evaluated, results, skipped


def _evaluate_checked(request: ValidationRequest, plan: EvaluationPlan, ev: _Evaluation, input_hash: str, project_hash: str,
                      projection_hash: str, build_id: str | None) -> ValidationResult:
    project = ev.project
    _run_physics_nodes(ev)
    issues = _physics_issues(ev)
    surface_profiles: tuple[SurfaceProfileEvidence, ...] = ()
    if ev.runs(BEAM_TRACE):
        surface_profiles, profile_issues = _surface_profiles(ev.product(SCENE).scene, _solver_totals(ev))
        issues.extend(profile_issues)
    layout = None
    if ev.runs(LAYOUT):
        laid: _LaidOut = _node(ev, LAYOUT, (ev.runs(BEAM_TRACE),), _layout_node)
        layout = laid.layout
        issues.extend(laid.issues)
    control = None
    seed = ev.seed
    if ev.runs(CONTROL):
        spec = project.control_spec
        controlled: _Controlled = _node(ev, CONTROL, (None if spec is None else hash_exact(spec.model_dump(mode="json")), ev.seed),
                                        _control_node)
        control, seed = controlled.control, controlled.seed
        issues.extend(controlled.issues)
    observations = _observations(ev)
    evaluated, results, not_evaluated = _requirements(plan, ev, observations, layout, control)
    with phase("result"):
        return _assembled(request, plan, ev, input_hash, project_hash, projection_hash, build_id, issues, observations,
                          surface_profiles, layout, control, seed, evaluated, results, not_evaluated)


#: What a plane no traced beam reaches says where the plane bounds did not run, what its issue adds, and what a plane only
#: part of a traced beam reaches says there (after ``partial_reach``: the trace itself knows the partial crossing).
UNLIT_BOUNDS_NOT_COMPUTED = ("no traced beam reaches this observation plane; whether light the trace does not follow reaches it "
                             "was not computed (the evaluation did not run the plane bounds)")
ISSUE_BOUNDS_NOT_COMPUTED = ("whether light the trace does not follow reaches the plane was not computed (the evaluation did not "
                             "run the plane bounds)")
PARTIAL_BOUNDS_NOT_COMPUTED = ("its power there, and whether light the trace does not follow reaches the plane, were not computed "
                               "(the evaluation did not run the plane bounds)")
PARTIAL_BOUNDS_ACTION = f"{PARTIAL_CROSSING_PLACEMENT}; evaluate with {PLANE_BOUNDS} for the upper bound of the power there"


def _partial_crossings(trace: TraceOutput) -> dict[str, tuple[str | None, ...]]:
    """Per observation plane, the surfaces part of a traced beam reaches it before while the beam's chief ray does not cross
    it (``untraced.partial_surfaces`` without the parts a severe truncation diffracts: the plane bounds word those)."""
    found: dict[str, list[str | None]] = {}
    for part in trace.partial_crossings:
        if not part.diffracted:
            seen = found.setdefault(part.observation_id, [])
            if part.ends_at not in seen:
                seen.append(part.ends_at)
    return {oid: tuple(surfaces) for oid, surfaces in found.items()}


def _bounds_not_computed(observations: list[ObservationResult], issues: list[ValidationIssue],
                         partial: Mapping[str, tuple[str | None, ...]],
                         launched: Iterable[float]) -> tuple[list[ObservationResult], list[ValidationIssue]]:
    """The plane readings of an evaluation that did not run the plane bounds: every plane's power state is ``NOT_COMPUTED``
    and nothing says its power is complete (no upper bound, no completeness per wavelength, no untraced power: null, never
    0). Every launched wavelength (``launched``) is listed per plane, as in a complete evaluation; one no traced light
    brings has no power value (null, never 0: whether it is a known zero is the plane bounds' to say). A plane traced light
    of known power reaches keeps its traced power and its status; a plane only beams of unknown
    power reach has no power value and reads ``UNKNOWN``, as in a complete evaluation (that their power is not known is a
    fact of the trace); a lit plane a beam of out-of-scope power reaches says so and states no lower bound, as in a
    complete evaluation; a plane no traced beam reaches has no power value and no grade (``NOT_APPLICABLE``), and the issue
    that names it grades nothing either: whether it is dark is the plane bounds' to say. ``partial``: the planes part of a
    traced beam reaches (``_partial_crossings``) -- such a plane says so, as in a complete evaluation, never that no traced
    beam reaches it."""
    unlit = {o.observation_id for o in observations if not o.lit}
    wavelengths = sorted(set(launched))
    planes = []
    for o in observations:
        listed = {s.wavelength_nm for s in o.power_by_wavelength}
        spectral = tuple(sorted(
            [*(s.model_copy(update={"complete": False, "untraced_power_W": None}) for s in o.power_by_wavelength),
             *(SpectralPower(wavelength_nm=w, power_W=None, complete=False, untraced_power_W=None)
               for w in wavelengths if w not in listed)], key=lambda s: s.wavelength_nm))
        update: dict[str, Any] = {"quantity_state": QuantityState.NOT_COMPUTED, "power_complete": False, "untraced_power_W": None,
                                  "power_by_wavelength": spectral}
        if o.observation_id in unlit:
            message = (f"{partial_reach(partial[o.observation_id])}; {PARTIAL_BOUNDS_NOT_COMPUTED}"
                       if o.observation_id in partial else UNLIT_BOUNDS_NOT_COMPUTED)
            # (no power value, no lower bound: as in a complete evaluation, here and in the next branch)
            update.update(status=Status.NOT_APPLICABLE, total_power_W=None, power_lower_bound_W=None, message=message)
        elif not o.power_lit:
            update.update(status=worst([o.status, Status.UNKNOWN]), total_power_W=None, power_lower_bound_W=None,
                          message=f"{unknown_power_reach(len(o.arrivals))}; {ISSUE_BOUNDS_NOT_COMPUTED}")
        elif scope := plane_power_scope(o):
            # (the traced value adds chief-ray powers, not the light's: it bounds nothing, and neither does a share taken
            # from it -- ``complete_power_by_wavelength``)
            update.update(power_lower_bound_W=None,
                          message=f"{power_scope_reach(scope, o.total_power_W or 0.0)}; {ISSUE_BOUNDS_NOT_COMPUTED}")
        planes.append(o.model_copy(update=update))
    out = []
    for issue in issues:
        oid = issue.location.removeprefix("observation:") if issue.location else None
        if oid in unlit and issue.code in (ErrorCode.OBSERVATION_NOT_REACHED, ErrorCode.ZERO_POWER):
            update = {"status": Status.NOT_APPLICABLE, "message": f"{issue.message}; {ISSUE_BOUNDS_NOT_COMPUTED}"}
            if oid in partial:
                names = sorted({s for s in partial[oid] if s})
                update["message"] = f"{partial_reach(partial[oid])}; {PARTIAL_BOUNDS_NOT_COMPUTED}"
                update["numerical_context"] = {**issue.numerical_context, "partial_before_surface_id": names[0] if names else None,
                                               "partial_before_surface_ids": names}
                if issue.code is ErrorCode.OBSERVATION_NOT_REACHED:
                    update["suggested_action"] = PARTIAL_BOUNDS_ACTION
            issue = issue.model_copy(update=update)
        out.append(issue)
    return planes, out


def _scope(plan: EvaluationPlan, not_evaluated: tuple[TargetNotEvaluated, ...],
           exposures_read_analysis: bool = True) -> EvaluationScope | None:
    """The scope a narrowed evaluation's result states; ``None`` when every capability of the request's complete scope ran
    and every declared target was evaluated. ``exposures_read_analysis``: ``not_computed_produces``."""
    missing = plan.not_executed
    if not missing and not not_evaluated:
        return None
    return EvaluationScope(
        requested=plan.requested, executed=plan.executed,
        not_computed=tuple(NotComputed(capability=c, reason=EXCLUDED if c in plan.excluded_capabilities else NOT_REQUESTED,
                                       produces=not_computed_produces(c, plan.executed, exposures_read_analysis))
                           for c in missing),
        targets_not_evaluated=not_evaluated)


def _assembled(request: ValidationRequest, plan: EvaluationPlan, ev: _Evaluation, input_hash: str, project_hash: str,
               projection_hash: str, build_id: str | None, issues: list[ValidationIssue], obs_results: list[ObservationResult],
               surface_profiles: tuple[SurfaceProfileEvidence, ...], layout: Any, control: Any, seed: int | None,
               evaluated: tuple[Target, ...], targets: list[TargetResult],
               not_evaluated: tuple[TargetNotEvaluated, ...]) -> ValidationResult:
    """The evaluation's ``ValidationResult`` from the products of the capabilities it ran: the planes no beam reaches
    graded (where the plane bounds ran), the run-level notices, each plane's quantity state, the status, the summary, the
    scope of a narrowed evaluation and the manifest with its hashes."""
    project, cfg = ev.project, ev.cfg
    trace = _result_trace(ev)
    bounds: _Bounds | None = ev.product(PLANE_BOUNDS) if ev.runs(PLANE_BOUNDS) else None
    power_budget = ev.product(POWER_LEDGER).budget if ev.runs(POWER_LEDGER) else None
    reach = bounds.reach if bounds is not None else None
    exposed: _Exposed | None = ev.product(SURFACE_EXPOSURES) if ev.runs(SURFACE_EXPOSURES) else None
    pairs: _Pairs | None = ev.product(BEAM_PAIRS) if ev.runs(BEAM_PAIRS) else None
    secondary: _Secondary | None = ev.product(SECONDARY_RAYS) if ev.runs(SECONDARY_RAYS) else None
    if bounds is not None:
        # a plane no traced beam reaches: dark (0 W), or not known when light was not followed; targets play no part
        assert trace is not None
        obs_results, issues = grade_unlit_planes(obs_results, issues, power_budget, cfg, reach, partial_surfaces(trace))
    elif ev.runs(PLANE_READINGS):
        assert trace is not None
        wavelengths = {line.wavelength_nm for line in trace.inputs}
        wavelengths.update(c.harmonic_wavelength_nm for c in trace.nonlinear_conversions if c.status is Status.UNKNOWN)
        obs_results, issues = _bounds_not_computed(obs_results, issues, _partial_crossings(trace), wavelengths)
    issues = issues + lit_reach_issues(evaluated, targets, {o.observation_id: o for o in obs_results}, power_budget, reach, cfg)
    if bounds is not None:
        assert trace is not None
        issues = issues + plane_bound_notices(_return_gains(ev, CHAIN), reach, ev.product(SCENE).scene, trace, obs_results)
    issues = name_affected_targets(issues, evaluated, targets, {o.observation_id: o for o in obs_results}, power_budget, reach, cfg)
    if bounds is not None:
        # (whether each plane's power is a known quantity, a known zero or unknown; facts of the physics, not of a target)
        obs_results = [o.model_copy(update={"quantity_state": quantity_state_of(o)}) for o in obs_results]
        # (a plane that states a detector: what the detector reads of that light, validation/detection.py)
        detections = {o.observation_id: o.detection for o in project.observations if o.detection is not None}
        obs_results = [o.model_copy(update={"detection": detection_result(o, detections[o.observation_id], o.quantity_state)})
                       if o.observation_id in detections else o for o in obs_results]
    interference = pairs.interference if pairs is not None else []
    # (the layout analysis' findings are its own outcome, ``layout.status``: a mechanical finding reaches the run's status
    # only through a declared layout target, ``layout_violation_count`` / ``layout_min_clearance_mm``)
    statuses = [t.status for t in targets] + [i.status for i in issues] + [o.status for o in obs_results]
    if control is not None:
        statuses.append(control.status)
    overall = worst(statuses)
    if not targets and overall is Status.PASS and not obs_results and not interference and layout is None and control is None:
        overall = Status.NOT_APPLICABLE
    if not targets and not_evaluated and overall is Status.PASS:
        # (the request declares targets and this evaluation assessed none of them: what it computed found nothing, and
        # that is no verdict on the requirements -- evaluation.targets_not_evaluated names them)
        overall = Status.NOT_APPLICABLE
    counts = Counter(t.status.value for t in targets)
    issue_counts = Counter(i.code.value for i in issues)
    summary = ValidationSummary(target_counts=dict(sorted(counts.items())), issue_counts=dict(sorted(issue_counts.items())),
                                path_count=len(trace.paths) if trace is not None else 0, observation_count=len(obs_results))
    reader = _last_reader(ev)
    paths: tuple[OpticalPath, ...] = ()
    if trace is not None and request.configuration.include_paths:
        paths = tuple(sorted(trace.paths, key=lambda p: p.path_id))
    result = ValidationResult(status=overall, summary=summary, targets=tuple(targets), observations=tuple(obs_results),
                              grating_orders=tuple(trace.grating_orders) if trace is not None else (),
                              nonlinear_conversions=tuple(trace.nonlinear_conversions) if trace is not None else (),
                              paths=paths, interference=tuple(interference),
                              incoherent_pairs=pairs.incoherent if pairs is not None else (), layout=layout,
                              control=control, issues=tuple(issues),
                              power_budget=power_budget, exposures=exposed.exposure.report if exposed is not None else None,
                              surface_profiles=surface_profiles,
                              sampled_sources=(sampled_sources(ev.product(SCENE).scene, secondary.sampled)
                                               + scattered_sources(secondary.scattered)) if secondary is not None else (),
                              source_spectra=(ev.product(PLANE_READINGS).spectra if ev.runs(PLANE_READINGS)
                                              else source_spectra(project.components)),
                              outcome=result_outcome(issues, cfg, tuple(targets), declared=bool(project.targets),
                                                     analyses=ev.product(reader).analyses if reader is not None else ()),
                              evaluation=_scope(plan, not_evaluated, exposed is None or _rated(ev)))
    manifest = _manifest(result, cfg, input_hash, project_hash, projection_hash, trace.trace_events if trace is not None else None,
                         seed, issues, project.project_schema_version, build_id)
    return result.model_copy(update={"manifest": manifest})


def validate_json(text: str | bytes) -> str:
    """JSON in, JSON out (the Pyodide bridge / CLI entry). Malformed input yields a structured FAIL result."""
    try:
        request = ValidationRequest.model_validate_json(text)
    except ValidationError as exc:
        raw = text if isinstance(text, str) else text.decode("utf-8", errors="replace")
        cfg = NumericalConfig()
        errors = [f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:MAX_REPORTED_CONTRACT_ERRORS]]
        issue = make_issue(
            ErrorCode.INVALID_INPUT, Status.FAIL,
            f"the request does not satisfy the public contract at {errors[0][:MAX_FIRST_ERROR_CHARS]}" if errors else
            "the request does not satisfy the public contract", errors=errors, error_count=len(exc.errors()),
        )
        result = _failure_result(hash_exact(raw), cfg, issue)
        return result.model_dump_json()
    except (ValueError, RecursionError) as exc:  # input the JSON parser itself cannot take (nesting, encoding)
        raw = text if isinstance(text, str) else text.decode("utf-8", errors="replace")
        issue = make_issue(ErrorCode.INVALID_INPUT, Status.FAIL, "the request is not valid JSON",
                           errors=[f"{type(exc).__name__}: {str(exc)[:MAX_PARSE_ERROR_CHARS]}"], error_count=1)
        return _failure_result(hash_exact(raw), NumericalConfig(), issue).model_dump_json()
    return validate(request).model_dump_json()
