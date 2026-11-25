"""Ghost paths of a run's image plane: where each ghost reflection's light lands, how bright it is against the primary image,
and a bound on every ghost not listed (``ghosts``, ``GhostRequestV1``), as an analysis document (``AnalysisResultV1``,
analysis ``GHOST_PATHS``) that changes no run.

Trace. The run's request is traced again with every transmitting interface's reflection followed as a ghost branch up to the
declared bounce order (``results.ghost_trace``): the bare Fresnel reflection of an uncoated face, the stated reflectance of a
coated one, each at its chief ray's angle of incidence and polarization, applied to the whole beam as the run applies it to the
transmitted light. The primary light is traced as in the run. A ghost path's bounce order is its number of such reflections; an
even order travels on toward the image side (``FORWARD``), an odd order back toward the source (``BACKWARD``). The trace is done
with the sources' own pupil sampling and again with twice the rings and twice the points per ring (``convergence``); the values
are the finer trace's. With the request's ``convergence`` false the trace is done once, with the sources' own pupil sampling, and
each value's convergence is ``NONE`` (not checked); its settings name ``convergence`` only then.

Per listed ghost path and line (a source's spectral line): its bounce surfaces, direction and end, the power its last bounce
reflected (relative to the line's input), and where it crosses the image plane (an observation plane the request names):

* ``relative_power`` -- its power on the plane over the primary light's (the paths without a ghost bounce) on the same plane,
  same line: an incoherent ratio of powers (no interference with the primary, no absolute radiometry);
* ``focus_offset_mm`` -- the least-squares point closest to its pupil rays (``results.rays._ray_focus``) as a signed distance
  from the plane along its normal, oriented with the primary light (positive: beyond the plane). Parallel rays have no focus
  (``RAY_FOCUS_UNDEFINED``);
* ``waist_offset_x_mm`` / ``_y_mm`` -- the same for the waists of its paraxial Gaussian beam, where that beam is valid (else
  ``OUT_OF_MODEL_SCOPE`` with the Gaussian state's reason code);
* ``spot_rms_radius_um`` -- the RMS radius of its pupil rays on the plane about their centroid;
* ``irradiance_ratio`` -- its peak irradiance over the primary's: ``2P / (pi w_x w_y)`` of both Gaussian beams where both are
  valid, else the estimate ``relative_power x (r_primary / r_ghost)^2`` from RMS radii widened in quadrature by the diffraction
  radius ``lambda / (pi NA)`` of each bundle (an estimate, not a bound).

A power on the geometric pupil-sample basis (an aperture cut the pupil rays of a beam without a valid Gaussian state on its way)
is a sampled value: its quantity states the sampling resolution (``convergence.resolution``, in the quantity's own units).

Completeness. Every part of a line's light the ghost trace does not follow -- reflections beyond the bounce order, below the
floor, light the budget leaves untraced, paths that end outside the model, scattered light, light a response model does not
deliver, other untraced light, and the enumerated ghost paths beyond ``max_listed`` -- enters, with an upper bound on its
power, the bound on every ghost not listed: a passive system never raises a beam's power, so that sum (at most the line's input:
the pieces are disjoint parts of its light) bounds the power any unlisted ghost brings to any plane. The bound takes, at every
ghost surface, an upper bound on the beam's mean reflectance and transmittance over its angles of incidence instead of its
chief ray's (``results.ghost_trace``): proven over a valid Gaussian beam on a plane or a sphere, else sampled over the beam's
pupil rays, which leaves the line's interval ``NOMINAL`` and ``completeness.proven`` false. It does not bound irradiance (a
weak ghost may focus). Light that ended outside the model, scattered light and light a response model does not deliver are bounded but
their own ghosts are not enumerated: ``completeness.not_followed`` lists each with its reason and bound.

Model scope. The run's model-scope findings (``model_scope_issues``) mark the numbers of the light they reach, line by line:
a finding that lists an arrival of the line on the plane (its ``affected_arrivals``: the run's paths of the line, both
polarization components, every branch) reaches every power, ratio and bound of the line; one that reaches the line anywhere
and names an element a ghost bounces at also reaches that ghost's numbers (the ghost reflects that element's coefficients),
and every finding that reaches the line reaches the bound on the ghosts not listed. Such a number reads
``OUT_OF_MODEL_SCOPE``, keeps its value and carries the findings' ``ESTIMATE`` in ``value_basis`` (``imaging.outside_model``);
where the light lands (focus, waists, spot) is marked only by a finding that touches the geometry. ``arrays["model_scope"]``
lists the findings applied (``handoff.model_scope``). A power that is itself out of model scope (``POWER_OUT_OF_MODEL_SCOPE``)
states no number and keeps its own reason.

Execution. ``COMPLETED`` when the enumeration ran to its end, ``PARTIAL`` (``BUDGET_EXHAUSTED``) when its budget ran out. Light
that is not followed, a power that is out of model scope, a missing primary and a requested field that is not a beam source do
not change it: they are stated in ``not_followed``, in ``completeness.proven`` and in the quantities' own states and reasons.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from pydantic import ValidationError

from ..analysis import completed, document, partial
from ..contracts.analysis import (
    AnalysisResultV1,
    Completeness,
    Convergence,
    ConvergenceMethod,
    Interval,
    IntervalKind,
    ModelBasis,
    NotFollowed,
    Quantity,
    QuantityScope,
)
from ..contracts.basis import ValidityCheck
from ..contracts.ghosts import GhostRequestV1
from ..contracts.outcome import BudgetUsage, budget_limits
from ..contracts.request import ValidationRequest
from ..errors.status import QuantityState, TerminationReason
from ..math.vectors import correctly_rounded_sum, dot
from ..optics.materials.library import MaterialLibrary
from ..optics.rays.path_ids import BRANCH_SUFFIX_PATTERN, root_path_id
from ..optics.rays.tracer import ArrivalRecord, TraceOutput
from ..scene.lower import lower_scene
from ..units import NM_TO_MM
from .ghost_trace import (
    BELOW_FLOOR,
    BOUNCE_LIMIT,
    OUT_OF_MODEL,
    PATH_BUDGET,
    SCATTERED,
    UNMODELLED_RESPONSE,
    UNTRACED_PREFIX,
    GhostTracer,
    Unfollowed,
    ghost_bounces,
)
from .imaging import MM_TO_UM, NOT_CHECKED, RAY_TOLERANCE, _sig, settings_request
from .handoff import merged_scope
from .model import ResultsError
from .quantity import POWER_OUT_OF_MODEL_SCOPE, power_scope_message
from .rays import _ray_focus

GHOST_ANALYSIS = "GHOST_PATHS"
GHOST_ANALYSIS_VERSION = "1.3"
GHOSTS_INVALID = "GHOSTS_INVALID"
FORWARD = "FORWARD"
BACKWARD = "BACKWARD"
#: The finer trace has this many times the rings and the points per ring of the source's pupil sampling.
REFINEMENT_FACTOR = 2
#: The request's selection at its default (the full analysis), left out of the settings (``imaging.settings_request``).
SELECTION_DEFAULTS = {"convergence": True}
#: A focus or a spot needs at least this many pupil rays on the plane.
MIN_FOCUS_RAYS = 8
#: A requested wavelength matches a spectral line within this, in nm.
WAVELENGTH_MATCH_NM = 1e-6
#: Focus and waist offsets agree between the two traces within this share of the larger, or within this many mm.
OFFSET_ABS_TOLERANCE_MM = 1e-6
FOCUS_MODEL_ID = "ray_focus"
FRESNEL_MODEL_ID = "interface_reflection_ghosts"
WAIST_MODEL_ID = "gaussian_waist"
GAUSSIAN_IRRADIANCE_MODEL_ID = "gaussian_peak_irradiance"
GEOMETRIC_IRRADIANCE_MODEL_ID = "geometric_area_estimate"
BOUND_MODEL_ID = "unlisted_ghost_bound"

PRIMARY_NOT_ON_PLANE = "PRIMARY_NOT_ON_PLANE"
RAY_FOCUS_UNDEFINED = "RAY_FOCUS_UNDEFINED"
TOO_FEW_RAYS = "TOO_FEW_RAYS"
GAUSSIAN_STATE_INVALID = "GAUSSIAN_STATE_INVALID"
POWER_NOT_KNOWN = "POWER_NOT_KNOWN"
NOT_LISTED = "NOT_LISTED"
#: The quantities of a ghost read from its power: none is stated where that power is out of model scope (``GhostTracer.power_scope``,
#: ``ArrivalRecord.power_scope``); where the light lands (focus, waist, spot) stays.
GHOST_POWER_QUANTITIES = frozenset({"reflected_power_relative_to_line", "end_power_relative_to_line", "power_relative_to_line",
                                    "relative_power", "irradiance_ratio"})
#: Of those, the ratios to the primary light's: out of model scope also where the primary's power is.
RELATIVE_TO_PRIMARY = frozenset({"relative_power", "irradiance_ratio"})
#: The quantities that read where the light lands only: a finding that touches no geometry (a coefficient used outside its
#: range) leaves them as they are (``imaging.outside_model``).
GEOMETRY_ONLY = frozenset({"focus_offset_mm", "waist_offset_x_mm", "waist_offset_y_mm", "spot_rms_radius_um", "primary_spot_rms_radius_um"})
#: The bound on every ghost not listed, over the line's input and over the primary's.
UNLISTED_QUANTITIES = frozenset({"unlisted_power_upper_relative_to_line", "unlisted_power_upper_relative_to_primary"})
SAMPLING_NOT_RESOLVED = "SAMPLING_NOT_RESOLVED"
MODELS = (
    ModelBasis(model_id=FRESNEL_MODEL_ID, name="sequential interface-reflection ghosts", version=GHOST_ANALYSIS_VERSION, assumptions=(
        "each transmitting interface reflects its bare Fresnel reflectance, or its coating's stated reflectance, at the chief ray's "
        "angle of incidence and polarization; the whole beam takes its chief ray's coefficients (the listed powers; the bound on "
        "the light not listed takes an upper bound on the beam's mean coefficients instead)",
        "powers, not fields: a ghost does not interfere with the primary light or with another ghost",
        "a reflection the design itself follows (a splitter, a mirror, total internal reflection) is the design's light",
        "the power of a ghost past a clear aperture that cuts its pupil rays without a valid Gaussian state is the share of its "
        "equal-power pupil samples that pass, within the stated sampling resolution")),
    ModelBasis(model_id=FOCUS_MODEL_ID, name="least-squares focus of the pupil rays", version=GHOST_ANALYSIS_VERSION, assumptions=(
        "the point closest to every pupil ray of the ghost (equal weights), as a signed distance from the image plane along its "
        "normal oriented with the primary light; parallel rays have no focus",)),
    ModelBasis(model_id=WAIST_MODEL_ID, name="paraxial Gaussian beam waist", version=GHOST_ANALYSIS_VERSION, assumptions=(
        "the waists of the ghost's paraxial Gaussian beam along its chief ray, as signed distances from the image plane",)),
    ModelBasis(model_id=GAUSSIAN_IRRADIANCE_MODEL_ID, name="Gaussian peak irradiance ratio", version=GHOST_ANALYSIS_VERSION, assumptions=(
        "peak irradiance 2P / (pi w_x w_y) of each beam in its transverse plane at the image plane",)),
    ModelBasis(model_id=GEOMETRIC_IRRADIANCE_MODEL_ID, name="geometric spot-area estimate", version=GHOST_ANALYSIS_VERSION, assumptions=(
        "an estimate, not a bound: the power ratio times the ratio of the spots' areas, each spot's RMS radius widened in quadrature "
        "by lambda / (pi NA) of its bundle (NA: the medium index times the largest sine of its rays about their mean direction)",)),
    ModelBasis(model_id=BOUND_MODEL_ID, name="power bound on the ghosts not listed", version=GHOST_ANALYSIS_VERSION, assumptions=(
        "a passive system never raises a beam's power: the power of every piece of light not followed, where it leaves the traced "
        "beams, bounds what it and its own ghosts bring to any plane",
        "past a clear aperture that cuts the pupil rays of a beam without a valid Gaussian state, the bound takes the light that met it",
        "at each transmitting interface whose reflection is a ghost bounce, the bound takes an upper bound on the beam's mean power "
        "reflectance and transmittance: over its paraxial Gaussian beam's light, shell by shell in its whitened position and "
        "direction (radius rho = 0.25 to 6), each shell's share times the largest coefficient of either polarization over the angles "
        "of incidence it reaches (within rho far-field half angles of the chief ray's direction, and on a sphere of radius R at "
        "normals whose sine to the chief ray's line differs by at most (rho w + 2 rho theta_0 R) / R), the light beyond counted "
        "whole: proven on a plane or a sphere",
        "a beam without a valid Gaussian state, an annular beam or a sag surface: the mean over its live pupil rays of each ray's "
        "larger coefficient at its own angle of incidence, a sampled value that leaves the bound not proven (NOMINAL); a crystal "
        "face: the chief ray's coefficients, not proven",
        "the pieces are disjoint parts of the line's light: together they carry at most its input",
        "every other interaction acts on the whole beam with its chief ray's coefficients, as in the run",
        "a bound on power, not on irradiance: a weak ghost may focus")),
)


class GhostError(ResultsError):
    """A request that is not valid: no analysis is made."""


def parse_request(request: Any) -> GhostRequestV1:
    if isinstance(request, GhostRequestV1):
        return request
    try:
        return GhostRequestV1.model_validate({} if request is None else request)
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(str(p) for p in e['loc']) or 'request'}: {e['msg']}" for e in exc.errors(include_url=False))
        raise GhostError(GHOSTS_INVALID, problems) from None


# ---------------------------------------------------------------------------------------------- the traces


@dataclass
class _Trace:
    tracer: GhostTracer
    out: TraceOutput
    inputs: dict[tuple[str, int], tuple[float, float]]  # line -> (wavelength nm, input power W)


def _trace(request: dict[str, Any], wanted: set[str], parsed: GhostRequestV1, scale: int) -> _Trace:
    validated = ValidationRequest.model_validate(request)
    cfg = validated.configuration.numerical.model_copy(update={"max_paths": parsed.max_paths, "max_trace_steps": parsed.max_trace_steps})
    library = MaterialLibrary(validated.project.materials, cfg)
    scene, _poses = lower_scene(validated.project, cfg, library)
    sources = [replace(src, ring_count=src.ring_count * scale, points_per_ring=src.points_per_ring * scale) for src in scene.sources
               if src.source_id in wanted]
    scene = replace(scene, sources=sources, emitters=[])
    floor = {(src.source_id, j): parsed.relative_power_floor * src.power_W * weight for src in sources for j, (_wl, weight) in enumerate(src.spectrum)}
    tracer = GhostTracer(scene, library, cfg, max_order=parsed.max_bounce_order, floor_W=floor)
    out = tracer.run()
    inputs: dict[tuple[str, int], tuple[float, float]] = {}
    for item in out.inputs:
        key = (item.source_id, item.spectral_index)
        inputs[key] = (item.wavelength_nm, inputs.get(key, (0.0, 0.0))[1] + item.power_W)
    return _Trace(tracer, out, inputs)


def _beam_sources(results: Any) -> dict[str, list[float]]:
    lines: dict[str, list[float]] = {}
    for path in results.result.paths:  # the run's beam paths: its Gaussian sources' spectral lines (an extended emitter has none)
        found = lines.setdefault(path.source_id, [])
        if path.wavelength_nm not in found:
            found.append(path.wavelength_nm)
    return lines


# ---------------------------------------------------------------------------------------------- per crossing


@dataclass(frozen=True)
class _Crossing:
    """What one beam's crossing of the image plane gives."""

    power_W: float
    sampling: float | None
    focus_offset: float | None
    focus_point: tuple[float, ...] | None
    focus_reason: str | None
    waist: tuple[float, float] | None
    waist_reason: str | None
    spot_rms: float | None
    diffraction_radius: float | None
    gaussian_radii: tuple[float, float] | None


def _normal(arrival: ArrivalRecord, primary: ArrivalRecord | None) -> np.ndarray:
    normal = np.array(arrival.plane_normal, dtype=np.float64)
    reference = primary.direction if primary is not None else arrival.direction
    return normal if float(dot(normal, reference)) >= 0.0 else -normal


def _crossing(arrival: ArrivalRecord, primary: ArrivalRecord | None) -> _Crossing:
    normal = _normal(arrival, primary)
    origin = np.array(arrival.plane_origin, dtype=np.float64)
    points, directions = np.asarray(arrival.pupil_points, dtype=np.float64), np.asarray(arrival.pupil_directions, dtype=np.float64)
    count = int(points.shape[0])
    focus_offset = focus_point = None
    focus_reason = None
    spot = radius_d = None
    if count < MIN_FOCUS_RAYS:
        focus_reason = TOO_FEW_RAYS
    else:
        focus = _ray_focus(points, directions)
        if focus is None:
            focus_reason = RAY_FOCUS_UNDEFINED
        else:
            focus_offset = float(dot(focus - origin, normal))
            focus_point = tuple(_sig(float(c)) for c in focus)
        uv = np.asarray(arrival.pupil_uv, dtype=np.float64)
        centred = uv - uv.mean(axis=0)
        spot = math.sqrt(float((centred * centred).sum(axis=1).mean()))
        mean = directions.mean(axis=0)
        mean = mean / math.sqrt(float(dot(mean, mean)))
        across = directions - dot(directions, mean)[:, None] * mean[None, :]
        sine = float(np.sqrt((across * across).sum(axis=1)).max())
        na = arrival.n * sine
        radius_d = arrival.wavelength_nm * NM_TO_MM / (math.pi * na) if na > 0.0 else 0.0
    waist = gaussian = None
    waist_reason = None
    g = arrival.gstate
    if g.valid:
        position, direction = np.array(arrival.position, dtype=np.float64), np.array(arrival.direction, dtype=np.float64)
        waist = tuple(float(dot(position - g.waist_distance(axis) * direction - origin, normal)) for axis in ("x", "y"))  # type: ignore[assignment]
        gaussian = (g.radius("x"), g.radius("y"))
    else:
        waist_reason = g.reason.value if g.reason is not None else GAUSSIAN_STATE_INVALID
    return _Crossing(arrival.power_W, arrival.sampling, focus_offset, focus_point, focus_reason, waist, waist_reason, spot, radius_d, gaussian)


def _irradiance(ghost: _Crossing, primary: _Crossing, relative: float) -> tuple[float | None, str, str | None]:
    """The peak irradiance ratio: (value, basis, reason when not available)."""

    if ghost.gaussian_radii is not None and primary.gaussian_radii is not None:
        gx, gy = ghost.gaussian_radii
        px, py = primary.gaussian_radii
        return relative * (px * py) / (gx * gy), GAUSSIAN_IRRADIANCE_MODEL_ID, None
    if ghost.spot_rms is None or primary.spot_rms is None or ghost.diffraction_radius is None or primary.diffraction_radius is None:
        return None, GEOMETRIC_IRRADIANCE_MODEL_ID, TOO_FEW_RAYS
    r_ghost = ghost.spot_rms * ghost.spot_rms + ghost.diffraction_radius * ghost.diffraction_radius
    r_primary = primary.spot_rms * primary.spot_rms + primary.diffraction_radius * primary.diffraction_radius
    if r_ghost <= 0.0:
        return None, GEOMETRIC_IRRADIANCE_MODEL_ID, TOO_FEW_RAYS
    return relative * r_primary / r_ghost, GEOMETRIC_IRRADIANCE_MODEL_ID, None


def _converged(coarse: float | None, fine: float | None, parameter: str, *, absolute: float = 0.0,
               resolution: float | None = None) -> Convergence:
    """The refinement evidence of a value; ``resolution``: its sampling resolution on the geometric pupil-sample basis (in the
    value's units; ``None``: the Gaussian model's value), which also widens what counts as converged."""
    res = None if resolution is None else _sig(resolution)
    if coarse is None or fine is None:
        return Convergence(method=ConvergenceMethod.REFINEMENT, parameter=parameter, resolution=res, converged=None)
    difference = abs(fine - coarse)
    return Convergence(method=ConvergenceMethod.REFINEMENT, parameter=parameter, coarse=_sig(coarse), fine=_sig(fine), difference=_sig(difference),
                       resolution=res, converged=difference <= RAY_TOLERANCE * max(abs(fine), abs(coarse)) or difference <= absolute)


def _unchecked(coarse: float | None, fine: float | None, parameter: str, *, absolute: float = 0.0,
               resolution: float | None = None) -> Convergence:
    """The convergence of a value of a single trace (``convergence`` false): not checked; its sampling resolution stays."""
    if resolution is None:
        return NOT_CHECKED
    return Convergence(method=ConvergenceMethod.NONE, resolution=_sig(resolution))


def _resolution(power_W: float, sampling: float | None) -> float | None:
    """The sampling resolution (W) of a power on the geometric pupil-sample basis (``sampling``: as a fraction of it)."""
    return None if sampling is None else sampling * power_W


def _ratio_resolution(num: float, num_res: float | None, den: float, den_res: float | None) -> float | None | bool:
    """The resolution of ``num / den`` from those of its parts: ``None`` when neither is sampled, ``False`` when the
    denominator's resolution reaches zero (the ratio is not resolved)."""
    if num_res is None and den_res is None:
        return None
    rn, rd = num_res or 0.0, den_res or 0.0
    if den - rd <= 0.0:
        return False
    return (num + rn) / (den - rd) - num / den


def _basis_note(resolution: float | None, unit: str = "") -> str:
    return "" if resolution is None else f"on the geometric pupil-sample basis: sampling resolution {_sig(resolution):.3g}{unit}"


# ---------------------------------------------------------------------------------------------- the analysis


def ghosts(results: Any, monitor: str, request: Any) -> AnalysisResultV1:
    """The ghost-path analysis of the run ``results`` at the observation plane ``monitor`` (module docstring)."""

    parsed = parse_request(request)
    if results.request is None:
        raise ResultsError("REQUEST_NOT_AVAILABLE", "this run carries no Engine request (engine_request.json): its light cannot be traced again")
    results._observation(monitor)  # (an unknown plane is refused)
    beam_sources = _beam_sources(results)
    wanted = list(parsed.fields) if parsed.fields is not None else sorted(beam_sources)
    refused = [source for source in wanted if source not in beam_sources]
    traced = {source for source in wanted if source in beam_sources}
    settings = {"request": settings_request(parsed, SELECTION_DEFAULTS), "monitor": monitor, "random_seed": None,
                "pupil_sampling": f"the sources' own pupil rays; convergence against {REFINEMENT_FACTOR} x the rings and the points per ring"
                if parsed.convergence else "the sources' own pupil rays, traced once"}
    quantities: list[Quantity] = []
    arrays: dict[str, Any] = {"conventions": {
        "direction": "FORWARD: an even number of ghost bounces (on toward the image side); BACKWARD: an odd number (back toward the source)",
        "offsets": "signed distances from the image plane along its normal, oriented with the primary light: positive beyond the plane",
        "powers": "relative to the primary light on the same plane and line, or to the line's input power; no absolute radiometry",
    }}
    not_followed: list[NotFollowed] = []
    enumerated = 0
    upper_total: list[float] = []
    proven = True
    for source in refused:
        quantities.append(Quantity(quantity_id=f"{source}.field", name="ghost field", scope=QuantityScope(source_id=source, observation_id=monitor),
                                   state=QuantityState.NOT_APPLICABLE, reason="FIELD_NOT_A_BEAM_SOURCE",
                                   message=f"{source} is not a beam source of this run (an extended emitter, or no such source)"))
    budgets = BudgetUsage()
    # (the run's model-scope findings: each line's quantities carry those that reach its light, ``_mark_line``)
    findings = results._scope_findings() or ()
    applied: list[list[dict[str, Any]]] = []
    if traced:
        coarse = _trace(results.request, traced, parsed, 1)
        fine = _trace(results.request, traced, parsed, REFINEMENT_FACTOR) if parsed.convergence else coarse
        exhausted = tuple(sorted(set(coarse.tracer.schedule.exhausted) | set(fine.tracer.schedule.exhausted)))
        budgets = BudgetUsage(limits=budget_limits(max_paths=parsed.max_paths, max_trace_steps=parsed.max_trace_steps,
                                                   max_interactions_per_path=fine.tracer.cfg.max_interactions_per_path),
                              exhausted=exhausted, affected_source_ids=fine.tracer.schedule.affected_source_ids,
                              affected_path_ids=fine.tracer.schedule.affected_path_ids)
        for key in sorted(fine.inputs):
            wavelength, line_W = fine.inputs[key]
            if parsed.wavelengths_nm is not None and not any(abs(wavelength - w) <= WAVELENGTH_MATCH_NM for w in parsed.wavelengths_nm):
                continue
            line = _Line(key, wavelength, line_W, monitor, parsed, coarse, fine)
            start = len(quantities)
            line.analyse(quantities, arrays, not_followed)
            if findings:
                applied.append(_mark_line(results, findings, monitor, key, wavelength, quantities, start, arrays.get(line.prefix + "ghosts") or ()))
            enumerated += line.enumerated
            upper_total.append(line.upper_W)
            proven = proven and line.proven
    statement = (f"Every ghost path with up to {parsed.max_bounce_order} interface reflections whose reflection carries more than "
                 f"{parsed.relative_power_floor:g} of its line's input is enumerated; the strongest {parsed.max_listed} per line are listed. "
                 "Every ghost not listed, and every other piece of light not followed, carries in total at most unenumerated_power_upper_W "
                 "to any plane (a bound on power, not on irradiance)" + ("." if proven else
                 "; it is not proven: it does not cover the light whose power the trace does not know (POWER_NOT_KNOWN) or whose power is "
                 "out of model scope (POWER_OUT_OF_MODEL_SCOPE), or it rests at some ghost surfaces on the angles of the beam's pupil rays "
                 "(see each line's bound)."))
    completeness = Completeness(statement=statement, enumerated=enumerated, not_followed=tuple(not_followed),
                                unenumerated_power_upper_W=_sig(correctly_rounded_sum(upper_total)) if upper_total else 0.0, proven=proven)
    # (the execution says whether the enumeration ran to its end: only its budget stops it early. Light that is not followed
    # -- scattered, ended outside the model, of unknown power -- is in not_followed and the bound; a power that is out of
    # model scope, a missing primary and a field that is not a beam source are their quantities' own states)
    if budgets.exhausted:
        execution = partial(TerminationReason.BUDGET_EXHAUSTED, "the ghost trace's budget left light untraced: its power is in the bound", budgets)
    else:
        execution = completed(budgets)
    scope = merged_scope(applied)
    if scope:
        arrays["model_scope"] = scope
    identity = results.identity()
    return document(GHOST_ANALYSIS, GHOST_ANALYSIS_VERSION, settings, execution, engine_input_hash=identity.get("engine_input_hash"),
                    run_result_hash=identity.get("result_hash"), models=MODELS, quantities=quantities, arrays=arrays, completeness=completeness,
                    engine_build_id=results.engine_build_id)


def _line_pattern(key: tuple[str, int]) -> re.Pattern[str]:
    """The path ids of the line ``key`` (``source_id``, spectral index): both polarization components and every branch."""
    return re.compile(re.escape(root_path_id(*key)) + BRANCH_SUFFIX_PATTERN)


def _line_findings(findings: Any, key: tuple[str, int], wavelength: float) -> list[dict[str, Any]]:
    """The model-scope findings (rows) that reach the light of the line ``key`` anywhere, as ``{code, instance_id, touches,
    validity}`` (``handoff.model_scope``): a finding that lists an arrival of the line's paths or its sampled line (with the
    checks whose light is among them), and a finding that lists no light at all (with its wavelength checks of the line's
    wavelength alone; one whose wavelength checks all name another wavelength does not reach it)."""

    from ..local.validity import finding_quantity_kinds
    from .basis import WAVELENGTH_CHECK, _arrival_keys, _finding_row, _line_keys

    pattern = _line_pattern(key)
    out = []
    for row in map(_finding_row, findings or ()):
        arrivals = {a for a in _arrival_keys(row.get("affected_arrivals")) if pattern.fullmatch(a[1])}
        lines = {line for line in _line_keys(row.get("affected_sampled_lines")) if (line[1], line[2]) == key}
        checks = [ValidityCheck.model_validate(c) for c in row.get("validity") or ()]
        if arrivals or lines:
            reach = row.get("validity_reach") or ()
            if len(reach) == len(checks):
                checks = [c for c, met in zip(checks, reach, strict=True)
                          if not arrivals.isdisjoint(_arrival_keys(met.get("affected_arrivals")))
                          or not lines.isdisjoint(_line_keys(met.get("affected_sampled_lines")))] or checks
        elif row.get("affected_arrivals") or row.get("affected_sampled_lines"):
            continue  # (it lists the light it reaches: another line's)
        elif any(c.parameter == WAVELENGTH_CHECK for c in checks):
            checks = [c for c in checks if c.parameter != WAVELENGTH_CHECK
                      or (c.observed is not None and abs(c.observed - wavelength) <= WAVELENGTH_MATCH_NM)]
            if not any(c.parameter == WAVELENGTH_CHECK for c in checks):
                continue
        if not checks:  # (a finding a run of an earlier Engine listed without its checks)
            checks = [ValidityCheck(parameter=row["code"].lower(), within=False, at=(row["instance_id"],), code=row["code"])]
        out.append({"code": row["code"], "instance_id": row["instance_id"],
                    "touches": sorted(kind.value for kind in finding_quantity_kinds(row["code"])),
                    "validity": [check.model_dump(mode="json") for check in checks]})
    return out


def _mark_line(results: Any, findings: Any, monitor: str, key: tuple[str, int], wavelength: float, quantities: list[Quantity],
               start: int, table: Any) -> list[dict[str, Any]]:
    """Mark the quantities ``quantities[start:]`` of the line ``key`` with the model-scope findings that reach the light they
    read (module docstring, *Model scope*); returns the findings applied (``handoff.model_scope`` entries)."""

    from .handoff import model_scope, scope_marks
    from .imaging import outside_model

    pattern = _line_pattern(key)
    light = {(monitor, a.beam.path_id, a.crossing_index) for a in results._observation(monitor).arrivals if pattern.fullmatch(a.beam.path_id)}
    line_scope = model_scope(findings, monitor, light, wavelength_nm=wavelength) if light else []
    on_line = _line_findings(findings, key, wavelength)
    ghost_of = {qid: row for row in table for qid in row.get("quantities", ())}
    applied: list[list[dict[str, Any]]] = []
    for index in range(start, len(quantities)):
        quantity = quantities[index]
        row = ghost_of.get(quantity.quantity_id)
        if row is not None:  # (a ghost's own: also the findings at an element it bounces at)
            bounces = row.get("bounce_surfaces") or ()
            extra = [f for f in on_line if any(s == f["instance_id"] or s.startswith(f["instance_id"] + "/") for s in bounces)]
            scope = merged_scope([line_scope, extra])
        elif quantity.quantity_id.rsplit(".", 1)[-1] in UNLISTED_QUANTITIES:  # (every ghost not listed: every finding on the line)
            scope = merged_scope([line_scope, on_line])
        else:
            scope = line_scope
        if scope and (quantity.value is not None or quantity.values_ref is not None):
            marked = outside_model(quantity, scope_marks(scope), geometry=GEOMETRY_ONLY)
            if marked is not quantity:
                quantities[index] = marked
                applied.append(scope)
    return merged_scope(applied)


class _Line:
    """One source line's ghosts (``ghosts``)."""

    def __init__(self, key: tuple[str, int], wavelength: float, line_W: float, monitor: str, request: GhostRequestV1, coarse: _Trace, fine: _Trace) -> None:
        self.key, self.wavelength, self.line_W, self.monitor, self.request = key, wavelength, line_W, monitor, request
        self.coarse, self.fine = coarse, fine
        # (one trace, ``convergence`` false: no value's discretization is checked)
        self.evidence = _converged if request.convergence else _unchecked
        # the line's spectral index is its identity (as in its path ids, <source>#<index>); the wavelength is in the scope
        self.prefix = f"{key[0]}#{key[1]}."
        self.scope = QuantityScope(source_id=key[0], observation_id=monitor, wavelength_nm=wavelength)
        self.enumerated = 0
        self.upper_W = 0.0
        self.proven = True

    def _of(self, trace: _Trace) -> tuple[dict[str, Any], dict[str, list[ArrivalRecord]]]:
        surfaces = trace.tracer.surfaces
        paths = {p.path_id: p for p in trace.out.paths if (p.source_id, p.spectral_index) == self.key}
        arrivals: dict[str, list[ArrivalRecord]] = {}
        for a in trace.out.arrivals:
            if a.observation_id == self.monitor and a.counted and (a.source_id, a.spectral_index) == self.key:
                arrivals.setdefault(a.path_id, []).append(a)
        orders = {pid: len(ghost_bounces(p.lineage, surfaces)) for pid, p in paths.items()}
        return {"paths": paths, "orders": orders}, arrivals

    def analyse(self, quantities: list[Quantity], arrays: dict[str, Any], not_followed: list[NotFollowed]) -> None:
        fine_paths, fine_arrivals = self._of(self.fine)
        coarse_paths, coarse_arrivals = self._of(self.coarse)
        orders, paths = fine_paths["orders"], fine_paths["paths"]
        surfaces = self.fine.tracer.surfaces
        p = self.prefix
        refine = f"pupil rings and points per ring x {REFINEMENT_FACTOR}"

        def primary_of(arrivals: dict[str, list[ArrivalRecord]], ords: dict[str, int]) -> tuple[float, ArrivalRecord | None, float | None]:
            primaries = [a for pid, items in arrivals.items() if ords.get(pid, 0) == 0 for a in items]
            if not primaries:
                return 0.0, None, None
            sampled = [a for a in primaries if a.sampling is not None]
            resolution = correctly_rounded_sum(a.sampling * a.power_W for a in sampled) if sampled else None
            return (correctly_rounded_sum(a.power_W for a in primaries), max(primaries, key=lambda a: (a.power_W, a.path_id)),
                    resolution)

        primary_W, primary, primary_res = primary_of(fine_arrivals, orders)
        coarse_primary_W, coarse_primary, _coarse_res = primary_of(coarse_arrivals, coarse_paths["orders"])
        primary_crossing = None if primary is None else _crossing(primary, primary)
        coarse_primary_crossing = None if coarse_primary is None else _crossing(coarse_primary, coarse_primary)
        # (light whose power is out of model scope, ``ArrivalResult.power_out_of_scope_at``: its power is its chief ray's, not the
        # light's, so no power, ratio or bound that reads it is stated; where the light lands stays)
        scope_of = self.fine.tracer.power_scope

        def path_scope(pid: str) -> set[str]:
            return {*scope_of.get(pid, ()), *(s for a in fine_arrivals.get(pid, ()) for s in a.power_scope)}

        primary_scope = sorted({s for pid, items in fine_arrivals.items() if orders.get(pid, 0) == 0
                                for s in (*path_scope(pid), *(x for a in items for x in a.power_scope))})
        out_of_scope = primary_scope
        if primary is None:
            quantities.append(Quantity(quantity_id=p + "primary_power_relative_to_line", name="primary light on the image plane over the line's input",
                                       scope=self.scope, state=QuantityState.UNKNOWN, reason=PRIMARY_NOT_ON_PLANE, basis=FRESNEL_MODEL_ID,
                                       message="no light of this line without a ghost bounce reaches the image plane: ghost powers are stated "
                                               "relative to the line's input only"))
        else:
            if out_of_scope:
                quantities.append(Quantity(quantity_id=p + "primary_power_relative_to_line", name="primary light on the image plane over the line's input",
                                           scope=self.scope, state=QuantityState.OUT_OF_MODEL_SCOPE, reason=POWER_OUT_OF_MODEL_SCOPE, basis=FRESNEL_MODEL_ID,
                                           message=power_scope_message(out_of_scope, "the primary light's power")))
            else:
                resolution = None if primary_res is None else primary_res / self.line_W
                quantities.append(Quantity(quantity_id=p + "primary_power_relative_to_line", name="primary light on the image plane over the line's input",
                                           scope=self.scope, state=QuantityState.KNOWN, value=_sig(primary_W / self.line_W), basis=FRESNEL_MODEL_ID,
                                           convergence=self.evidence(coarse_primary_W / self.line_W if coarse_primary is not None else None,
                                                                  primary_W / self.line_W, refine, resolution=resolution),
                                           message=_basis_note(resolution)))
            assert primary_crossing is not None
            if primary_crossing.spot_rms is not None:
                quantities.append(Quantity(quantity_id=p + "primary_spot_rms_radius_um", name="RMS radius of the primary light's pupil rays on the plane",
                                           unit="um", scope=self.scope, state=QuantityState.KNOWN, value=_sig(primary_crossing.spot_rms * MM_TO_UM),
                                           basis=FOCUS_MODEL_ID))

        # the enumerated ghost paths, the strongest on the plane first
        ghost_ids = [pid for pid, order in orders.items() if order >= 1]
        self.enumerated = len(ghost_ids)

        def strength(pid: str) -> tuple[float, float, str]:
            on_plane = max((a.power_W for a in fine_arrivals.get(pid, ())), default=0.0)
            return (-on_plane, -self.fine.tracer.birth_W.get(pid, 0.0), pid)

        ranked = sorted(ghost_ids, key=strength)
        listed, rest = ranked[:self.request.max_listed], ranked[self.request.max_listed:]
        table = []
        for pid in listed:
            path = paths[pid]
            bounces = ghost_bounces(path.lineage, surfaces)
            row: dict[str, Any] = {
                "path_id": pid, "bounce_order": len(bounces), "bounce_surfaces": list(bounces),
                "direction": FORWARD if len(bounces) % 2 == 0 else BACKWARD, "termination": path.termination,
                "termination_surface_id": path.termination_surface_id, "reaches_image_plane": pid in fine_arrivals, "quantities": [],
            }
            promoted = sorted({surface for a in fine_arrivals.get(pid, ()) for surface in a.reference_promoted})
            if promoted:  # (its chief ray fell beside a clear aperture: the ghost goes on as its pupil rays that passed)
                row["reference_promoted_at"] = promoted
            q = f"{p}{pid}."
            scope = self.scope.model_copy(update={"path_id": pid})
            own_scope = path_scope(pid)
            birth = self.fine.tracer.birth_W.get(pid)
            coarse_birth = self.coarse.tracer.birth_W.get(pid) if pid in coarse_paths["paths"] else None

            def add(name: str, label: str, unit: str | None, value: float | None, *, basis: str, convergence: Convergence | None = None,
                    reason: str | None = None, message: str = "", state: QuantityState = QuantityState.OUT_OF_MODEL_SCOPE, prefix: str = q) -> None:
                quantity_id = prefix + name
                row["quantities"].append(quantity_id)
                # (its own power, and against the primary's for the ratios)
                behind = own_scope | (set(primary_scope) if name in RELATIVE_TO_PRIMARY else set())
                if behind and name in GHOST_POWER_QUANTITIES:
                    value, state, reason, convergence = None, QuantityState.OUT_OF_MODEL_SCOPE, POWER_OUT_OF_MODEL_SCOPE, None
                    message = power_scope_message(sorted(behind), label)
                if value is None:
                    quantities.append(Quantity(quantity_id=quantity_id, name=label, unit=unit, scope=scope, state=state, reason=reason or "NOT_DETERMINED",
                                               message=message, basis=basis))
                    return
                quantities.append(Quantity(quantity_id=quantity_id, name=label, unit=unit, scope=scope, state=QuantityState.KNOWN, value=_sig(value),
                                           basis=basis, convergence=convergence, message=message))

            tracer = self.fine.tracer
            if birth is not None:
                res = _resolution(birth, tracer.birth_sampling.get(pid))
                res = None if res is None else res / self.line_W
                add("reflected_power_relative_to_line", "power its last ghost bounce reflects over the line's input", None, birth / self.line_W,
                    basis=FRESNEL_MODEL_ID, convergence=self.evidence(None if coarse_birth is None else coarse_birth / self.line_W, birth / self.line_W, refine,
                                                                   resolution=res), message=_basis_note(res))
            if pid not in fine_arrivals:
                end = path.final_power_W
                coarse_path = coarse_paths["paths"].get(pid)
                coarse_end = None if coarse_path is None else coarse_path.final_power_W
                if end is not None:
                    res = _resolution(end, tracer.end_sampling.get(pid))
                    res = None if res is None else res / self.line_W
                    add("end_power_relative_to_line", f"power where the path ends ({path.termination}) over the line's input", None, end / self.line_W,
                        basis=FRESNEL_MODEL_ID, convergence=self.evidence(None if coarse_end is None else coarse_end / self.line_W, end / self.line_W, refine,
                                                                       resolution=res), message=_basis_note(res))
            crossings = sorted(fine_arrivals.get(pid, ()), key=lambda a: a.crossing_index)
            row["crossing_indices"] = [a.crossing_index for a in crossings]
            for arrival in crossings:
                c = arrival.crossing_index
                cq = q if len(crossings) == 1 else f"{q}c{c}."
                ghost = _crossing(arrival, primary)
                match = next((a for a in coarse_arrivals.get(pid, ()) if a.crossing_index == c), None)
                rough = None if match is None else _crossing(match, coarse_primary)
                ghost_res = _resolution(ghost.power_W, ghost.sampling)
                res = None if ghost_res is None else ghost_res / self.line_W
                add("power_relative_to_line", "power on the image plane over the line's input", None, ghost.power_W / self.line_W, basis=FRESNEL_MODEL_ID,
                    convergence=self.evidence(None if rough is None else rough.power_W / self.line_W, ghost.power_W / self.line_W, refine, resolution=res),
                    message=_basis_note(res), prefix=cq)
                if primary is None or primary_W <= 0.0:
                    add("relative_power", "power on the image plane over the primary light's", None, None, basis=FRESNEL_MODEL_ID,
                        state=QuantityState.UNKNOWN, reason=PRIMARY_NOT_ON_PLANE, message="no primary light of this line reaches the image plane", prefix=cq)
                else:
                    relative = ghost.power_W / primary_W
                    rough_relative = None if rough is None or coarse_primary_W <= 0.0 else rough.power_W / coarse_primary_W
                    ratio_res = _ratio_resolution(ghost.power_W, ghost_res, primary_W, primary_res)
                    if ratio_res is False:
                        add("relative_power", "power on the image plane over the primary light's", None, None, basis=FRESNEL_MODEL_ID,
                            state=QuantityState.UNKNOWN, reason=SAMPLING_NOT_RESOLVED,
                            message="the primary light's power on the geometric pupil-sample basis is not resolved from zero", prefix=cq)
                    else:
                        add("relative_power", "power on the image plane over the primary light's", None, relative, basis=FRESNEL_MODEL_ID,
                            convergence=self.evidence(rough_relative, relative, refine, resolution=ratio_res),
                            message=_basis_note(ratio_res), prefix=cq)
                if ghost.focus_offset is not None:
                    add("focus_offset_mm", "least-squares focus of its pupil rays from the image plane", "mm", ghost.focus_offset, basis=FOCUS_MODEL_ID,
                        convergence=self.evidence(None if rough is None else rough.focus_offset, ghost.focus_offset, refine, absolute=OFFSET_ABS_TOLERANCE_MM),
                        prefix=cq)
                    row.setdefault("focus_point_mm", {})[str(c)] = list(ghost.focus_point or ())
                else:
                    message = ("its pupil rays are parallel: a collimated ghost has no focus" if ghost.focus_reason == RAY_FOCUS_UNDEFINED
                               else f"fewer than {MIN_FOCUS_RAYS} of its pupil rays reach the plane")
                    add("focus_offset_mm", "least-squares focus of its pupil rays from the image plane", "mm", None, basis=FOCUS_MODEL_ID,
                        state=QuantityState.NOT_APPLICABLE if ghost.focus_reason == RAY_FOCUS_UNDEFINED else QuantityState.UNKNOWN,
                        reason=ghost.focus_reason, message=message, prefix=cq)
                for index, axis in enumerate(("x", "y")):
                    if ghost.waist is not None:
                        add(f"waist_offset_{axis}_mm", f"waist of its Gaussian beam ({axis}) from the image plane", "mm", ghost.waist[index], basis=WAIST_MODEL_ID,
                            convergence=self.evidence(None if rough is None or rough.waist is None else rough.waist[index], ghost.waist[index], refine,
                                                   absolute=OFFSET_ABS_TOLERANCE_MM), prefix=cq)
                    else:
                        add(f"waist_offset_{axis}_mm", f"waist of its Gaussian beam ({axis}) from the image plane", "mm", None, basis=WAIST_MODEL_ID,
                            reason=ghost.waist_reason, message="its paraxial Gaussian beam is not valid here; the ray focus stands", prefix=cq)
                if ghost.spot_rms is not None:
                    add("spot_rms_radius_um", "RMS radius of its pupil rays on the plane", "um", ghost.spot_rms * MM_TO_UM, basis=FOCUS_MODEL_ID,
                        convergence=self.evidence(None if rough is None or rough.spot_rms is None else rough.spot_rms * MM_TO_UM, ghost.spot_rms * MM_TO_UM, refine),
                        prefix=cq)
                else:
                    add("spot_rms_radius_um", "RMS radius of its pupil rays on the plane", "um", None, basis=FOCUS_MODEL_ID, state=QuantityState.UNKNOWN,
                        reason=TOO_FEW_RAYS, message=f"fewer than {MIN_FOCUS_RAYS} of its pupil rays reach the plane", prefix=cq)
                if primary_crossing is None or primary_W <= 0.0:
                    add("irradiance_ratio", "peak irradiance on the plane over the primary light's", None, None, basis=GEOMETRIC_IRRADIANCE_MODEL_ID,
                        state=QuantityState.UNKNOWN, reason=PRIMARY_NOT_ON_PLANE, message="no primary light of this line reaches the image plane", prefix=cq)
                else:
                    value, basis, reason = _irradiance(ghost, primary_crossing, ghost.power_W / primary_W)
                    rough_value = None
                    if rough is not None and coarse_primary_crossing is not None and coarse_primary_W > 0.0:
                        rough_value, rough_basis, _ = _irradiance(rough, coarse_primary_crossing, rough.power_W / coarse_primary_W)
                        rough_value = rough_value if rough_basis == basis else None
                    add("irradiance_ratio", "peak irradiance on the plane over the primary light's", None, value, basis=basis, reason=reason,
                        message="fewer than the needed pupil rays of the ghost or of the primary reach the plane",
                        convergence=self.evidence(rough_value, value, refine) if value is not None else None, prefix=cq)
            table.append(row)
        arrays[p + "ghosts"] = table

        # what is not followed, and the bound on every ghost not listed
        pieces = [u for u in self.fine.tracer.unfollowed if (u.source_id, u.spectral_index) == self.key]
        unknown = sorted({pid for source, index, pid in self.fine.tracer.unknown_power if (source, index) == self.key})
        terms: dict[str, list[Unfollowed]] = {}
        for piece in pieces:
            terms.setdefault(piece.reason if not piece.reason.startswith(UNTRACED_PREFIX) else UNTRACED_PREFIX + "LIGHT", []).append(piece)
        grouped: dict[tuple[str, str], list[float]] = {}
        for piece in pieces:
            entry = grouped.setdefault((piece.path_id, piece.reason), [0.0, 0.0])
            entry[0] += piece.power_W
            entry[1] += piece.power_upper_W
        for (pid, reason), (_power, upper) in sorted(grouped.items()):
            not_followed.append(NotFollowed(item_id=f"{pid}:{reason}", reason=reason, power_upper_W=_sig(upper)))
        rest_W = [self.fine.tracer.birth_W.get(pid, 0.0) for pid in rest]
        rest_upper = [self.fine.tracer.birth_upper_W.get(pid, 0.0) for pid in rest]
        sampled = any(not piece.proven for piece in pieces) or any(pid in self.fine.tracer.unproven for pid in rest)
        if sampled:
            self.proven = False
        for pid, upper in zip(rest, rest_upper):
            not_followed.append(NotFollowed(item_id=f"{pid}:{NOT_LISTED}", reason=NOT_LISTED, power_upper_W=_sig(upper)))
        for pid in unknown:
            not_followed.append(NotFollowed(item_id=f"{pid}:{POWER_NOT_KNOWN}", reason=POWER_NOT_KNOWN))
            self.proven = False
        breakdown: dict[str, dict[str, float]] = {}
        for name in (BOUNCE_LIMIT, BELOW_FLOOR, PATH_BUDGET, OUT_OF_MODEL, SCATTERED, UNMODELLED_RESPONSE, UNTRACED_PREFIX + "LIGHT"):
            items = terms.get(name, [])
            breakdown[name.lower()] = {"power_W": _sig(correctly_rounded_sum(i.power_W for i in items)),
                                       "power_upper_W": _sig(correctly_rounded_sum(i.power_upper_W for i in items)), "pieces": len(items)}
        breakdown[NOT_LISTED.lower()] = {"power_W": _sig(correctly_rounded_sum(rest_W)), "power_upper_W": _sig(correctly_rounded_sum(rest_upper)),
                                        "pieces": len(rest)}
        nominal = correctly_rounded_sum([*(i.power_W for i in pieces), *rest_W])
        # (the pieces are disjoint parts of the line's light: together they carry at most its input)
        self.upper_W = min(correctly_rounded_sum([*(i.power_upper_W for i in pieces), *rest_upper]), self.line_W)
        faces = [f for f in self.fine.tracer.faces if f.path_id in paths or f.path_id == self.key[0]]
        arrays[p + "completeness"] = {"line_input_W": _sig(self.line_W), "enumerated_ghost_paths": len(ghost_ids), "listed": len(listed),
                                      "terms": breakdown, "power_not_known_paths": unknown,
                                      "ghost_surface_meetings": len(faces),
                                      "ghost_surface_meetings_with_sampled_angles": sum(1 for f in faces if not f.proven)}
        kind = IntervalKind.NOMINAL if sampled else IntervalKind.PROVEN
        message = ("value: the trace's own powers of the pieces; interval: their bound, not proven: at some ghost surfaces the beam had "
                   "no valid Gaussian state (or met a sag surface or a crystal face), and the reflectance and transmittance there are "
                   "bounded over the angles of its pupil rays only" if sampled else
                   "value: the trace's own powers of the pieces; interval: their proven bound")
        bound = Interval(lower=0.0, upper=_sig(self.upper_W / self.line_W), kind=kind)
        unlisted_scope = sorted({*(s for piece in pieces for s in scope_of.get(piece.path_id, ())),
                                 *(s for pid in rest for s in path_scope(pid))})
        if unlisted_scope:
            self.proven = False
            for name, label in (("unlisted_power_upper_relative_to_line", "power every ghost not listed brings to any plane, over the line's input"),
                                ("unlisted_power_upper_relative_to_primary", "power every ghost not listed brings to any plane, over the primary image's")):
                quantities.append(Quantity(quantity_id=p + name, name=label, scope=self.scope, state=QuantityState.OUT_OF_MODEL_SCOPE,
                                           reason=POWER_OUT_OF_MODEL_SCOPE, basis=BOUND_MODEL_ID, message=power_scope_message(unlisted_scope, "this bound")))
            return
        quantities.append(Quantity(quantity_id=p + "unlisted_power_upper_relative_to_line", name="power every ghost not listed brings to any plane, over the line's input",
                                   scope=self.scope, state=QuantityState.BOUNDED, value=_sig(nominal / self.line_W), interval=bound, basis=BOUND_MODEL_ID,
                                   message=message))
        if primary_W > 0.0:
            quantities.append(Quantity(quantity_id=p + "unlisted_power_upper_relative_to_primary", name="power every ghost not listed brings to any plane, over the primary image's",
                                       scope=self.scope, state=QuantityState.BOUNDED, value=_sig(nominal / primary_W),
                                       interval=Interval(lower=0.0, upper=_sig(self.upper_W / primary_W), kind=kind), basis=BOUND_MODEL_ID,
                                       message=message))


__all__ = ["BACKWARD", "FORWARD", "GHOST_ANALYSIS", "GHOSTS_INVALID", "GhostError", "ghosts", "parse_request"]
