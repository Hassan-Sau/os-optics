"""Target metric registry and evaluation. Unsupported physics never returns PASS.

Metric kinds:

* ``observation`` metrics read one beam arriving at an observation plane;
* ``pair`` metrics read the two-beam interference analysis on an observation plane;
* ``layout`` / ``control`` metrics read those stage results;
* geometric pair metrics (``validation/alignment.py``: centroid / angular / waist separation, envelope overlap) read two
  arrivals of any wavelengths (``wavelength_nm`` / ``wavelength_nm_b``), never the interference analysis;
* ``power_rating_utilization`` reads the run's exposure report (``validation/exposures.py``) for the element ``device_id``.

Anything in the out-of-scope registry returns ``OUT_OF_MODEL_SCOPE``; an unknown name returns ``UNKNOWN``.

A target that no traced beam matches is one of three distinct cases, read from the power ledger of the target's source
lines (its ``source_id`` / ``source_id_b`` and ``wavelength_nm``; every line when it names none; ``validation/power.py``):

* **empty selection** -- a ``source_id``, ``wavelength_nm``, ``path_id`` or ``crossing_index`` (k >= 1) that selects no beam
  of the run: ``UNKNOWN`` with ``TARGET_SELECTION_EMPTY`` (naming what the run has), never a value;
* **incomplete trace** -- part of those lines' power has an unknown fate (the trace budget: ``MAX_PATHS``; the per-path
  interaction limit: ``MAX_INTERACTIONS``; a path ended out of model scope or a line not launched: ``OUT_OF_MODEL_SCOPE``;
  a non-negligible part at the power floor: ``ZERO_POWER``), or, for a power summed over the lines, light the trace does
  not follow may reach the plane (``validation/untraced.py``: an untraced reflection or coating transmission, light
  passing beside an element, or part of a traced beam crossing the plane where its chief ray does not, each followed by
  the probes to a known amount: ``UNTRACED_LIGHT_BOUNDED``, whatever else is unknown; the part of a beam booked
  ``CLIPPED`` that passes the opening: ``CLIPPED``; a beam's tail on an active area it is not counted on:
  ``INCOMPLETE_ENVELOPE``): that light may reach the plane, which has no power value. A ``power_W`` /
  ``coherent_power_W`` target is graded on the interval from 0 to the bound of that light where every watt of it is light
  the probes followed to the plane and the bound is finite (a maximum at or above the bound passes, a minimum above it
  fails, any other limit is ``UNKNOWN``; the plane's own power is ``BOUNDED`` from 0 with no value), and is ``UNKNOWN`` with those codes
  where part of it has an unknown fate. A beam metric is ``UNKNOWN`` there. This holds with any selector: a ``path_id`` or a
  ``crossing_index`` that selects no arrival reads this light as well;
* **known zero** -- every watt of those lines was traced to a known end and none of the light the trace does not follow
  (followed by probe traces to every order above ``negligible_power_fraction`` x the line power)
  can reach the plane: it is dark for them. ``power_W`` and ``coherent_power_W`` are then ``0.0`` W (code
  ``NO_TRACED_ARRIVAL``; a maximum passes, a minimum above 0 fails); a metric that needs a beam (radius, position, ...)
  fails with ``OBSERVATION_NOT_REACHED``. Light left out as negligible within the lines' negligible level is the stated
  truncation of such a value (its ``value_basis`` entry holds that light's bound; the plane stays ``KNOWN_ZERO`` /
  ``KNOWN``), and the target is graded on the value and that bound together: a limit inside the interval they give (a
  maximum below the bound, a minimum the bound could reach) is ``UNKNOWN`` (``_left_out_reach``). The light left out as negligible
  may still reach the plane: its power then lies between 0 and that light's bound (on an
  absorbing plane its power, on a transparent one as often as its open paths cross the plane -- once per straight leg
  after each number of reflections, at least twice, ``ReturnGains.crossings`` --, more where surfaces facing each other
  may hold it and send it across:
  ``validation/untraced.py``, ``FAINT_POWER``; ``NO_TRACED_ARRIVAL``, ``FAINT_LIGHT_BOUNDED``), no value, and its
  coherent power between 0 and the square of that light's field.

Where some matching beams arrive but the lines are incomplete, a power summed over them is bounded, not known: the value
is the traced light's, and the plane's power lies in an interval (``_untraced_bounds``): for ``power_W`` from the traced
value up to the traced value plus the untraced power ``U`` (``LineFate.untraced_W``) on an absorbing plane, and on a
transparent one when ``U`` is light the probe traces followed to every crossing of the plane (no upper bound when part of
it has an unknown fate: it may cross the plane any number of times); for ``coherent_power_W`` from
``(sqrt(P) - S)^2`` (0 when ``S >= sqrt(P)``) to ``(sqrt(P) + S)^2`` with ``S = sum_i sqrt(U_i)`` over the separate
contributions the probes followed to the plane (each crossing: mutually coherent fields add their amplitudes), and from 0
to unbounded when part of the untraced light has an unknown fate (its multiplicity is not known). Every run with an
observation plane follows every untraced branch (``validation/pipeline.py``), so ``U`` includes
the untraced light of the target's lines that reaches the plane also when their traced beams arrive there (a weak
traced beam does not stand for a strong coating transmission of its line folded onto the same plane). A target is
graded on that interval: ``PASS`` only when every value in it meets the target (a minimum: the lower end meets it; a
maximum: the upper end meets it), ``FAIL`` only when none does (a minimum above the upper end; a maximum below the lower
end), otherwise ``UNKNOWN`` -- with the cause codes in each case. ``power_rating_utilization`` is graded the same way on the
element's utilization interval when its exposure is not exact (``validation/exposures.py``: from the traced beams up to
the light the trace does not follow, unbounded when power of unknown fate may reach the element -- e.g. the light a
tabulated thin lens does not deliver into its focused beam, which makes every rated element's maximum utilization
UNKNOWN, upstream of the lens too). A ``crossing_index`` is looked for among the crossings of the selected sources and
wavelength only.

A pair's light on the plane. The two-beam metrics describe the two traced beams' interference: their crossing angle,
fringe period, path and phase difference, coherence, spatial and polarization overlap and balance. The values of the
light on the plane -- the power-weighted ``fringe_visibility`` over the plane and, at the pair's peak position,
``peak_irradiance_W_per_mm2``, ``mean_irradiance_W_per_mm2``, ``estimated_exposure_time_s`` from the mean and the local
contrast ``fringe_visibility_peak`` -- also hold the light of the pair's wavelength that the trace does not follow: a run
with a target on one of them follows that light (``validation/pipeline.py``), and the target is graded on the interval it
leaves (``_pair_light``). ``fringe_visibility`` is the fringe's contrast over the plane as beam a's phase varies: with
``P = P_a + P_b`` the two beams' power on the plane (``incoherent_power_sum_W``), ``V`` the two beams' value and ``R`` the
largest field amplitude, as the square root of a power (sqrt(W)), that light may add across the plane (the same as
``coherent_power_W``'s: the followed contributions' square roots, and the light left out as negligible, the sum of its
pieces' square roots times its open paths' crossing field and the plane's return factor), the fringe's amplitude lies
within ``V P +- 2 sqrt(P) R``
and the mean power between ``(sqrt(P) - R)^2`` and ``(sqrt(P) + R)^2`` (Cauchy-Schwarz and Minkowski over the plane, any
phase, place and polarization of that light), so with ``rho = R / sqrt(P)`` the contrast lies between
``max(0, V - 2 rho) / (1 + rho)^2`` and ``min(1, (V + 2 rho) / (1 - rho)^2)`` (1 when ``rho >= 1``). With ``F`` the
largest field amplitude that light may have anywhere on the plane
(``untraced.IRRADIANCE_ROOT``: each crossing of followed light at its Gaussian peak, or the diffraction limit of its power
where its Gaussian state is not valid there; the light left out as negligible at the diffraction limit of its pieces'
amplitudes added, times its open paths' crossing field and the plane's return factor), an irradiance ``v`` there lies
between
``max(0, sqrt(v) - F)^2`` and ``(sqrt(v) + F)^2`` (Minkowski: any phase, any spread), and the contrast ``(I_max - I_min) /
(I_max + I_min)`` of the fringe there (``I_max`` the peak irradiance, ``I_min = 2 mean - peak``) between its values at
the ends of those intervals: at least ``(I_max- - I_min+) / (I_max- + I_min+)`` (0 if negative), at most ``(I_max+ -
I_min-) / (I_max+ + I_min-)``. Light of unknown fate leaves the irradiances without an upper bound and the contrast
between 0 and 1 (``UNTRACED_LIGHT_BOUNDED`` / ``FAINT_LIGHT_BOUNDED`` and the causes). Two mutually incoherent beams
(``INCOHERENT_SOURCES``) form no lasting fringe: their ``fringe_visibility`` and ``fringe_visibility_peak`` are 0, the first
bounded the same way, the second between 0 and 1 where such light reaches the plane (their irradiances are not formed).

Metric-level scope of response models. A tabulated thin lens (``TabulatedThinLensInteraction``) states the target metrics
its response was validated for (``supported_metrics``). A target whose beams crossed such a surface -- the beams arriving at
its observation plane that it selects (its ``source_id`` / ``wavelength_nm`` / ``path_id`` / ``crossing_index``; for a pair
metric or a geometric pair metric the two beams of the pair it selects, and where it selects no single pair every beam of
its ``source_id`` / ``source_id_b``) -- and whose metric that surface does not list is
``OUT_OF_MODEL_SCOPE`` with code ``METRIC_NOT_VALIDATED_FOR_ELEMENT``: its value is the number the Engine computes with the
element's ideal thin-lens step, reported with the entry ``RESPONSE_MODEL_UNVALIDATED`` (``value_basis["value"]``, at the
surfaces crossed) and not graded, because the response model states no accuracy for it. ``power_rating_utilization`` of an
element is out of scope without a value when a
beam on one of its rated surfaces crossed such a lens whose ``supported_metrics`` do not cover the exposure behind it
(``power_rating_utilization`` itself, or ``power_W`` with ``beam_radius_mm`` or both ``beam_radius_x_mm`` and
``beam_radius_y_mm``; ``validation/exposures.py``).

Power out of model scope (``ArrivalResult.power_out_of_scope_at``). A beam that crossed a variable attenuator whose density
varies across it by more than the tolerance (``ND_DENSITY_VARIES_ACROSS_BEAM``) carries the chief ray's transmittance, not
the beam's; part of a beam that met an absorbing observation plane its chief ray does not cross (before the plane of the
target, ``OUT_OF_MODEL_SCOPE``; the plane id appears as ``observation/<id>``) left the beam there, and part of a beam that
met a physical surface (a fold mirror, a lens; its surface id) before a plane its chief ray crosses first goes on from
there untraced, while the beam's power follows its chief ray; so does part of a beam that met a physical surface beside its
chief ray on an earlier leg, where it would have reached the surface the chief ray met inside its clear aperture. A target of a power-derived metric
(``POWER_DERIVED_METRICS``) whose beams include such a beam is ``OUT_OF_MODEL_SCOPE`` with that code and no value. The
variable attenuators are the run's ``neutral_density`` surfaces (``assess_targets(attenuators=...)``; without them every
surface other than an observation plane is taken as one). ``power_rating_utilization`` of an element is ``OUT_OF_MODEL_SCOPE``
the same way when light on one of its rated surfaces carries such a power (``ExposureHit`` ``OUT_OF_MODEL_SCOPE``;
``assess_targets(element_power_scope=...)``): behind a crystal whose second-harmonic generation is out of model scope
(``NONLINEAR_CONVERSION_OUT_OF_SCOPE``; e.g. a pulsed source, whose harmonic follows the peak power, not the average power
the continuous-wave conversion uses) and behind every surface above -- the exposure of that light is not known.

Chief-ray metrics behind a promoted reference (``ArrivalResult.reference_ray_promoted_at``). Where a clear aperture took the
chief ray while some pupil rays passed, the beam goes on as those rays and one of them carries its reference: the passing ray
nearest their centroid, the lowest row on a tie. The chief-ray hit point, direction and angle of incidence on a plane behind
it (``REFERENCE_RAY_METRICS``) are then that ray's -- one sample of the light that passed, whose choice among samples equally
near the centroid (a mirror-symmetric pair) is arbitrary --, not a property of the beam. A target on one whose beams include
such a beam is ``OUT_OF_MODEL_SCOPE`` with code ``CLIPPED``: its value is the reference ray's (the arrival's own ``uv_mm``,
``direction_uvn``, ``aoi_deg``), reported with the entry ``SINGLE_RAY_FOR_BEAM`` at the promoting surfaces
(``value_basis["value"]``) and not graded: a verdict on one sample could change with the row order. The plane's spot
centroid (``spot_centroid_u_mm`` / ``spot_centroid_v_mm``) is the beam's position there; a metric keeps one definition, so
the centroid is never reported under a chief-ray metric's name.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, replace
from typing import Any, Callable

from ..config import NumericalConfig
from ..contracts.basis import (BASIS_FIELD, Assumed, Omitted, Requirement, ValidityCheck, ValueBasis, grade_point,
                               merge_bases, undecided_status)
from ..contracts.project import Target
from ..contracts.results import (
    ArrivalResult,
    ControlResult,
    ExposureReport,
    InterferenceResult,
    LayoutResult,
    ObservationResult,
    SampledLightAtPlane,
    PowerBudget,
    TargetResult,
)
from ..errors import ErrorCode, Status, worst
from ..errors.status import AssumedInput, OmittedPart, ValueKind
from ..evaluation.capabilities import (BEAM_PAIRS, CONTROL, LAYOUT, PLANE_BOUNDS, PLANE_READINGS, POWER_LEDGER, SURFACE_EXPOSURES,
                                       UNTRACED_LIGHT)
from ..units import W_PER_MM2_TO_MW_PER_CM2
from ..optics.rays.tracer import REFLECTION_EVENTS, SINK_BULK_LOSS_UNKNOWN
from ..optics.interference.coherence import MM_PER_FS, delay_bound_fs, pass_id
from ..math.phase import reported_phase
from ..math.vectors import correctly_rounded_sum
from .pair_selection import target_selects_pair
from .alignment import ALIGNMENT_METRICS, alignment_operands, measure_alignment
from .exposures import ELEMENT_METRICS, measure_utilization
from .observations import POWER_SUM_METRICS, capped_by_launch
from .power import LineFate, left_out_parts, line_fate
from .scope import out_of_scope_explanation, out_of_scope_metric
from .mechanical import MECHANICAL_METRICS, mechanical_value
from .detection import DETECTION_METRICS, detect
from .outcome import quantity_state_of
from .nonlinear import NONLINEAR_METRICS, nonlinear_value
from .superposition import received_power_W
from .untraced import (PARTIAL_CROSSING, PlaneReach, faint_power_sum, faint_root_sum, hazard_root_sum, hazards, irradiance_root_sum,
                       return_causes, withheld_by_work)
from .polarization_mixture import POLARIZATION_METRICS, component_twins, mixture_metric, route_arrivals
from .time_delay import GROUP_DELAY_METRICS, PAIR_DELAY_METRICS, PHASE_DELAY_METRICS, group_delay, phase_delay, time_delay

#: Metrics of the field at the chief-ray point: not defined behind a central obstruction (no light on the chief ray there).
CHIEF_FIELD_METRICS = frozenset({"phase_rad"})
#: Metrics with no value behind an element that states no optical path (``ArrivalResult.ray_opl_defined`` false), in words.
OPTICAL_PATH_METRICS = {"phase_rad": "phase", "optical_path_length_mm": "optical path length", "phase_delay_ps": "phase delay"}
#: Pair metrics with no value where a pass one beam makes and the other does not states no optical path.
OPTICAL_PATH_PAIR_METRICS = frozenset({"optical_path_difference_mm", "optical_path_difference_abs_mm"})
CHIEF_FIELD_PAIR_METRICS = frozenset({"phase_difference_rad"})


@dataclass(frozen=True)
class MetricValue:
    value: float | None
    unit: str
    status: Status = Status.PASS  # status of the *measurement* (before comparison)
    codes: tuple[str, ...] = ()
    message: str = ""
    # a lower bound of ``value`` when the value is not exact (an annular beam counted whole on an active area that may miss
    # part of it: ``ArrivalResult.detected_power_min_W``); the target is judged on [lower, value]
    lower: float | None = None
    # where the plane's power lies when part of the target's light was not traced (``_untraced_bounds``): ``(lower, upper)``,
    # ``upper`` ``math.inf`` when nothing bounds it; the target is judged on that interval, ``value`` being the traced light's
    untraced: tuple[float, float] | None = None
    # what lies in the ``untraced`` interval, for the message
    subject: str = "the plane receives"
    # what the number is where it is not simply the model's value (``contracts/basis.py``): the target reports it under
    # ``value_basis["value"]`` and is graded by rule T1 (``grade_point``)
    basis: ValueBasis | None = None
    # false: the value is reported with the measurement's status and is not compared with the target (it describes
    # something the measurement says does not apply, e.g. the period of fringes that do not form)
    graded: bool = True
    # the failed validity check of the model the measurement needs (a Gaussian state beyond an accuracy limit): the target
    # reports it under ``validity`` with no value; or the check light left out as negligible leaves open (a detector's linear
    # range or bandwidth, ``_detection_metric``), reported beside the value
    validity: tuple[ValidityCheck, ...] = ()


def _unknown_power(t: Target, arrivals: list[ArrivalResult], unit: str, obs: ObservationResult | None, fate: LineFate) -> MetricValue:
    """A power summed over beams none of whose power is known (a grating's orders without a stated response): no value, and the
    interval it lies in. A target that selects its path reads the beams' own bounds; the plane's power on an absorbing plane
    is at most the power of unknown fate, on a transparent one nothing bounds it (that light may cross it more than once)."""
    selected = t.parameters.get("path_id") is not None
    if selected:
        upper = correctly_rounded_sum(a.power_upper_bound_W or 0.0 for a in arrivals)
    else:
        upper = fate.untraced_W + fate.faint_W if _absorbing(obs) else math.inf
    if _absorbing(obs):
        upper = capped_by_launch(upper, fate.input_W)
    return MetricValue(None, unit, Status.PASS, fate.cause_codes(),
                       f"{len(arrivals)} beam(s) reach the plane at known positions and directions, with a power that is not known "
                       f"({fate.describe()})", untraced=(0.0, upper))


def _received_power(arrivals: list[ArrivalResult], unit: str) -> MetricValue:
    """The power the plane receives from ``arrivals``, with its lower bound where a detected power is not exact, and the
    interval it lies in (``_power_interval``) where a beam's power is a pupil-sample estimate or part of it is taken before
    the plane. Its entry is the merge of the entries of the powers it adds (``results.basis.basis_of``: a share taken
    before the plane, an annular beam's detected power -- each an upper bound with ``BEAM_PART_ELSEWHERE``); a beam whose
    power is out of model scope behind a surface is ``_power_scope``'s, which states it for the target."""
    from ..results.basis import basis_of  # (imported here: the results package imports modules that import this one)

    value = correctly_rounded_sum(received_power_W(a) for a in arrivals)
    lower = correctly_rounded_sum(a.detected_power_min_W if a.detected_power_min_W is not None else received_power_W(a) for a in arrivals)
    basis = merge_bases(*(basis_of(a, "detected_power_W" if a.detected_power_W is not None else "beam.power_W") for a in arrivals
                          if received_power_W(a) > 0.0 and not a.power_out_of_scope_at), monotone="increasing")
    return _power_interval(MetricValue(value, unit, lower=lower if lower < value else None, basis=basis), arrivals)


def _power_interval(mv: MetricValue, arrivals: list[ArrivalResult]) -> MetricValue:
    """``mv`` judged on the interval the received power lies in when it is not exact on its own terms:

    * a beam on the geometric pupil-sample basis (``ArrivalResult.power_basis``, ``GEOMETRIC_POWER_BASIS``): its power
      within plus or minus its ``power_sampling_resolution_W`` -- convergence evidence of a sampled value, not a proven bound
      (as ``AnalysisResultV1`` ``Quantity.decides``): a limit outside it is decided, one within it is not; the message says so
      and names why the beam is sampled (``_geometric_causes``);
    * a beam part of which surfaces take before the plane (``power_taken_before_upper_W``, ``POWER_TAKEN_BEFORE_PLANE``):
      the power the plane receives from it is at most what it reads (on a transparent plane less
      ``power_taken_before_lower_W``, the share a stopping surface certainly takes; not on an absorbing plane, whose reading is
      its active area's share, which the light certainly taken may have missed) and at least that reading less the bound.
      Each beam's ends lie within what it reads: one outside the plane's acceptance cone (received 0) adds nothing.

    The ends add over the beams (the lower end at least 0). ``mv`` unchanged when no beam is either."""
    geometric = [a for a in arrivals if a.power_basis is not None]
    taken = [a for a in arrivals if a.power_taken_before_upper_W is not None and received_power_W(a) > 0.0]
    if not geometric and not taken or mv.value is None:
        return mv
    sampling = correctly_rounded_sum(a.power_sampling_resolution_W or 0.0 for a in geometric)
    share = correctly_rounded_sum(min(a.power_taken_before_upper_W or 0.0, received_power_W(a)) for a in taken)
    certain = correctly_rounded_sum(min(a.power_taken_before_lower_W or 0.0, received_power_W(a)) for a in taken
                                    if a.detected_power_W is None)
    low = max((mv.lower if mv.lower is not None else mv.value) - sampling - share, 0.0)
    high = max(mv.value + sampling - certain, low)
    codes, notes = list(mv.codes), []
    if geometric:
        codes.append(ErrorCode.GEOMETRIC_POWER_BASIS.value)
        notes.append(f"{len(geometric)} beam(s) carry a power sampled by their pupil rays ({_geometric_causes(geometric)}), within a "
                     f"sampling resolution of {sampling:.4g} W (convergence evidence of the sampled value, not a proven bound)")
    if taken:
        codes.append(ErrorCode.POWER_TAKEN_BEFORE_PLANE.value)
        surfaces = sorted({surface for a in taken for surface in a.power_taken_before_at})
        notes.append(f"{', '.join(surfaces)} {'take' if len(surfaces) > 1 else 'takes'} at most {share:.4g} W of the beam before "
                     "this plane" + (f" and at least {certain:.4g} W" if certain > 0.0 else ""))
    subject = "within that resolution the plane receives" if geometric else "the plane receives"
    return MetricValue(mv.value, mv.unit, mv.status, tuple(dict.fromkeys(codes)), "; ".join(notes), lower=mv.lower,
                       untraced=(low, high), subject=subject, basis=mv.basis)


#: Why a beam's power is on the geometric pupil-sample basis, by the code that ended its Gaussian state (``GaussianBeamState.reason``).
_GEOMETRIC_CAUSE = {
    ErrorCode.TRANSMISSION_MAP_EVANESCENT.value: "a thin sample turned part of the beam beyond grazing, and its evanescent share left it",
}
_GEOMETRIC_EDGE_CAUSE = ("the beam has no valid Gaussian state, and an edge took part of its pupil samples or passed within a sample "
                         "spacing of them: a clear aperture, or the active area of this absorbing plane")


def _geometric_causes(arrivals: list[ArrivalResult]) -> str:
    """Why the ``arrivals``' powers are on the geometric pupil-sample basis, each cause once."""
    causes = [_GEOMETRIC_CAUSE.get(a.beam.gaussian.reason or "", _GEOMETRIC_EDGE_CAUSE) if a.beam.gaussian is not None
              else _GEOMETRIC_EDGE_CAUSE for a in arrivals]
    return "; ".join(dict.fromkeys(causes))


def _get(obj: object, name: str) -> float | None:
    v = getattr(obj, name, None)
    return None if v is None else float(v)


# metric -> (unit, extractor(ArrivalResult) -> float|None)
def _gauss(field_name: str) -> Callable[[ArrivalResult], float | None]:
    def f(a: ArrivalResult) -> float | None:
        g = a.beam.gaussian
        if g is None or not g.valid:
            return None
        return _get(g, field_name)

    return f


def _pol(field_name: str) -> Callable[[ArrivalResult], float | None]:
    def f(a: ArrivalResult) -> float | None:
        return None if a.polarization_state is None else _get(a.polarization_state, field_name)

    return f


def _first_order(field_name: str) -> Callable[[ArrivalResult], float | None]:
    def f(a: ArrivalResult) -> float | None:
        return None if a.first_order is None else _get(a.first_order, field_name)

    return f


def _parity(a: ArrivalResult) -> float | None:
    if a.first_order is not None:
        return 1.0 if a.first_order.handedness_preserved else -1.0
    reflections = sum(1 for step in a.lineage if step.rsplit(":", 1)[-1] in REFLECTION_EVENTS)
    return 1.0 if reflections % 2 == 0 else -1.0


def _centroid(index: int) -> Callable[[ArrivalResult], float | None]:
    def f(a: ArrivalResult) -> float | None:
        return None if a.pupil_centroid_uv_mm is None else a.pupil_centroid_uv_mm[index]

    return f


def _direction(index: int) -> Callable[[ArrivalResult], float | None]:
    def f(a: ArrivalResult) -> float | None:
        return None if a.direction_uvn is None else a.direction_uvn[index]

    return f


def _geo_mean_radius(a: ArrivalResult) -> float | None:
    g = a.beam.gaussian
    if g is None or not g.valid or g.beam_radius_x_mm is None or g.beam_radius_y_mm is None:
        return None
    return math.sqrt(g.beam_radius_x_mm * g.beam_radius_y_mm)


OBSERVATION_METRICS: dict[str, tuple[str, Callable[[ArrivalResult], float | None], bool]] = {
    # name: (unit, extractor, needs_gaussian)
    "power_W": ("W", lambda a: a.beam.power_W, False),
    "coherent_power_W": ("W", lambda a: a.beam.power_W, False),  # evaluated by _coherent_power
    "beam_radius_x_mm": ("mm", _gauss("beam_radius_x_mm"), True),
    "beam_radius_y_mm": ("mm", _gauss("beam_radius_y_mm"), True),
    "beam_radius_mm": ("mm", _geo_mean_radius, True),
    "waist_radius_x_mm": ("mm", _gauss("waist_radius_x_mm"), True),
    "waist_radius_y_mm": ("mm", _gauss("waist_radius_y_mm"), True),
    "waist_distance_x_mm": ("mm", _gauss("waist_distance_x_mm"), True),
    "waist_distance_y_mm": ("mm", _gauss("waist_distance_y_mm"), True),
    "rayleigh_range_x_mm": ("mm", _gauss("rayleigh_range_x_mm"), True),
    "rayleigh_range_y_mm": ("mm", _gauss("rayleigh_range_y_mm"), True),
    "divergence_half_angle_x_rad": ("rad", _gauss("far_field_half_angle_x_rad"), True),
    "divergence_half_angle_y_rad": ("rad", _gauss("far_field_half_angle_y_rad"), True),
    "aoi_deg": ("deg", lambda a: a.aoi_deg, False),
    "optical_path_length_mm": ("mm", lambda a: a.beam.optical_path_length_mm, False),
    "geometric_path_length_mm": ("mm", lambda a: a.beam.geometric_length_mm, False),
    "phase_rad": ("rad", lambda a: a.beam.phase_rad, False),
    "position_u_mm": ("mm", lambda a: a.uv_mm[0], False),
    "position_v_mm": ("mm", lambda a: a.uv_mm[1], False),
    "wavelength_nm": ("nm", lambda a: a.beam.wavelength_nm, False),
    "spot_rms_mm": ("mm", lambda a: a.pupil_rms_radius_mm, False),
    "capture_margin_mm": ("mm", lambda a: a.capture_margin_mm, True),
    "active_area_power_fraction": ("1", lambda a: a.active_area_power_fraction, True),
    "acceptance_margin_deg": ("deg", lambda a: a.acceptance_margin_deg, False),
    "spot_centroid_u_mm": ("mm", _centroid(0), False),
    "spot_centroid_v_mm": ("mm", _centroid(1), False),
    "direction_cosine_u": ("1", _direction(0), False),
    "direction_cosine_v": ("1", _direction(1), False),
    "stokes_s1": ("1", _pol("s1"), False),
    "stokes_s2": ("1", _pol("s2"), False),
    "stokes_s3": ("1", _pol("s3"), False),
    "polarization_ellipticity_deg": ("deg", _pol("ellipticity_deg"), False),
    "polarization_azimuth_deg": ("deg", _pol("azimuth_in_plane_deg"), False),
    "degree_of_polarization": ("1", _pol("degree_of_polarization"), False),
    # (evaluated by validation/time_delay.py; the extractors compare the two polarization components of one route)
    "group_delay_ps": ("ps", lambda a: a.group_path_length_mm, False),
    "phase_delay_ps": ("ps", lambda a: a.beam.optical_path_length_mm, False),
    "image_parity": ("1", _parity, False),
    "image_rotation_deg": ("deg", _first_order("image_rotation_deg"), False),
    "transverse_magnification": ("1", _first_order("transverse_magnification"), False),
}

PAIR_METRICS: dict[str, tuple[str, str]] = {
    # metric -> (unit, InterferenceResult field)
    "beam_crossing_angle_deg": ("deg", "beam_crossing_angle_deg"),
    "half_angle_deg": ("deg", "half_angle_deg"),
    "fringe_period_nm": ("nm", "fringe_period_nm"),
    "fringe_spacing_3d_nm": ("nm", "fringe_spacing_3d_nm"),
    "optical_path_difference_mm": ("mm", "optical_path_difference_mm"),
    "optical_path_difference_abs_mm": ("mm", "optical_path_difference_mm"),  # |OPD|
    "phase_difference_rad": ("rad", "phase_difference_rad"),
    "coherence_length_mm": ("mm", "coherence_length_mm"),
    "coherence_margin": ("1", "coherence_margin"),
    "coherence_factor": ("1", "coherence_factor"),
    "spatial_overlap_ratio": ("1", "spatial_overlap_ratio"),
    "polarization_overlap": ("1", "polarization_overlap"),
    "intensity_balance": ("1", "intensity_balance"),
    "fringe_visibility": ("1", "fringe_visibility"),
    "fringe_visibility_peak": ("1", "fringe_visibility_peak"),
    "mean_irradiance_W_per_mm2": ("W/mm^2", "mean_irradiance_W_per_mm2"),
    "peak_irradiance_W_per_mm2": ("W/mm^2", "peak_irradiance_W_per_mm2"),
    "overlap_effective_area_mm2": ("mm^2", "overlap_effective_area_mm2"),
    "estimated_exposure_time_s": ("s", "mean_irradiance_W_per_mm2"),  # derived; needs parameters['dose_mj_per_cm2']
    "time_delay_ps": ("ps", "group_path_difference_mm"),  # evaluated by validation/time_delay.py
}

#: Two-beam metrics that read the beams' degree of coherence: the source's declared coherence
#: (``SourceSpec.coherence_length_mm`` / ``linewidth_nm``; without one a line is taken as monochromatic, with a
#: ``FULL_COHERENCE`` entry) and the relative group delay of the two paths (where a pass states none, the value at no added
#: delay there, an ``ESTIMATE``). They are not computed (issue ``COHERENCE_UNDEFINED``, a target on one of them ``UNKNOWN``)
#: where a medium states no group index, a pulsed source states no coherence, or a pulse train's revival may change them;
#: ``coherence_length_mm`` needs the declared coherence.
COHERENCE_METRICS = frozenset({
    "coherence_length_mm", "coherence_margin", "coherence_factor", "fringe_visibility", "fringe_visibility_peak",
    "peak_irradiance_W_per_mm2",
})

LAYOUT_METRICS = {"layout_violation_count": "1", "layout_min_clearance_mm": "mm"}
CONTROL_METRICS = {
    "recoverability_fraction": "1",
    "jacobian_condition_number": "1",
    "monte_carlo_recovered_fraction": "1",
    "numerical_rank": "1",
}

ALL_SUPPORTED = (set(OBSERVATION_METRICS) | set(PAIR_METRICS) | set(ALIGNMENT_METRICS) | set(ELEMENT_METRICS) | set(LAYOUT_METRICS)
                 | set(CONTROL_METRICS) | set(MECHANICAL_METRICS) | set(DETECTION_METRICS) | set(NONLINEAR_METRICS))
#: Two-beam metrics that describe the geometry of the fringe pattern. Where the two beams do not overlap on the plane
#: (``BEAMS_NOT_OVERLAPPING``: their spatial overlap is at or below ``interference_min_overlap``) no fringe forms: a target
#: on one reports the period of the two wavevectors and is not graded (``NOT_APPLICABLE``, never ``PASS``). The metrics of
#: the light on the plane (the visibilities, the irradiances, the balance, the exposure time) are compared on the values
#: the pair analysis computed for that overlap.
FRINGE_GEOMETRY_METRICS = frozenset({"fringe_period_nm", "fringe_spacing_3d_nm"})
#: Two-beam metrics whose value for mutually incoherent beams (different coherence groups: ``INCOHERENT_SOURCES``) is 0:
#: their fringe averages out (the visibility over the plane and the contrast at any point, with a coherence factor of 0).
INCOHERENT_ZERO_METRICS = frozenset({"fringe_visibility", "fringe_visibility_peak"})
INCOHERENT_NOTE = ("the two beams come from mutually incoherent sources: their fringe averages out over time, so its "
                   "visibility is 0")
#: Two-beam metrics of the light on the plane (over the plane, or at the pair's peak position): light of the pair's
#: wavelength the trace does not follow adds to it (``_pair_light``); a target on one follows that light
#: (``validation/pipeline.py``).
PAIR_PLANE_METRICS = frozenset({"fringe_visibility", "peak_irradiance_W_per_mm2", "mean_irradiance_W_per_mm2",
                                "estimated_exposure_time_s", "fringe_visibility_peak"})
#: Two-beam metrics whose sign follows the operand order (a, b): reported for the target's own order.
ANTISYMMETRIC_PAIR_METRICS = frozenset({"optical_path_difference_mm", "phase_difference_rad"})
#: Metrics read from the chief ray's hit and direction on the plane: behind a promoted reference ray they are one pupil
#: sample's (module docstring), and a target on one is out of model scope, its value that sample's and not graded.
REFERENCE_RAY_METRICS = frozenset({
    "position_u_mm", "position_v_mm", "direction_cosine_u", "direction_cosine_v", "aoi_deg", "acceptance_margin_deg",
})
#: The surface id prefix of an observation plane (``scene/lower.py``) in ``ArrivalResult.power_out_of_scope_at``.
OBSERVATION_SURFACE_PREFIX = "observation/"
#: Metrics computed from the beams' powers: out of model scope when a beam's power is (``ArrivalResult.power_out_of_scope_at``).
POWER_DERIVED_METRICS = frozenset({
    "power_W", "coherent_power_W", "intensity_balance", "fringe_visibility", "fringe_visibility_peak",
    "mean_irradiance_W_per_mm2", "peak_irradiance_W_per_mm2", "estimated_exposure_time_s",
})
#: The evaluation capabilities a target on each metric reads (``os_optics.evaluation``; published as
#: ``MetricDescriptor.requires``): an evaluation that asks for the requirements runs them for each selected target; a
#: target on a stage the request turns off is assessed without it and reads as in ``validate``. A metric of one beam on a plane reads the plane
#: readings, and the power ledger for the light of a partially polarized source's route; a power on a plane and a detector's
#: quantity read the plane bounds; a two-beam metric reads the beam pairs, and the plane bounds for the pair's light on the
#: plane; the rating utilization reads the exposures with the untraced light; a conversion reads the ledger; a declared
#: limit reads nothing. A target's thresholds never enter it.
METRIC_REQUIRES: dict[str, tuple[str, ...]] = {
    **{metric: (PLANE_BOUNDS,) if metric in POWER_SUM_METRICS else (PLANE_READINGS, POWER_LEDGER) if metric in POLARIZATION_METRICS
       else (PLANE_READINGS,) for metric in OBSERVATION_METRICS},
    **{metric: (PLANE_READINGS,) for metric in ALIGNMENT_METRICS},
    **{metric: (PLANE_BOUNDS,) for metric in DETECTION_METRICS},
    **{metric: (BEAM_PAIRS, PLANE_BOUNDS) if metric in PAIR_PLANE_METRICS else (BEAM_PAIRS,) for metric in PAIR_METRICS},
    **{metric: (SURFACE_EXPOSURES, UNTRACED_LIGHT) for metric in ELEMENT_METRICS},
    **{metric: (POWER_LEDGER,) for metric in NONLINEAR_METRICS},
    **{metric: () for metric in MECHANICAL_METRICS},
    **{metric: (LAYOUT,) for metric in LAYOUT_METRICS},
    **{metric: (CONTROL,) for metric in CONTROL_METRICS},
}
#: Significant digits of the ends of a bounded value's interval in a target message: at least the usual 6, more (up to the
#: 17 that tell any two doubles apart) until the ends differ.
INTERVAL_MIN_DIGITS = 6
INTERVAL_MAX_DIGITS = 17
#: The unit of a dimensionless metric (not printed after a value).
DIMENSIONLESS_UNIT = "1"


def _select_arrivals(obs: ObservationResult | None, t: Target, cfg: NumericalConfig) -> tuple[list[ArrivalResult], str]:
    if obs is None:
        return [], f"unknown observation {t.observation_id!r}"
    arr = list(obs.arrivals)
    if t.source_id is not None:
        arr = [a for a in arr if a.beam.source_id == t.source_id]
    if t.wavelength_nm is not None:
        arr = [a for a in arr if abs(a.beam.wavelength_nm - t.wavelength_nm) <= cfg.interference_wavelength_tol_nm]
    # The Target contract has checked every parameter (``target_parameters.py``): each one given is applied here.
    pid = t.parameters.get("path_id")
    if pid is not None:
        arr = [a for a in arr if a.beam.path_id == pid]
    crossing = t.parameters.get("crossing_index")
    if crossing is not None:
        arr = [a for a in arr if a.crossing_index == crossing]
    return arr, ""


def _select_sampled(obs: ObservationResult | None, t: Target, cfg: NumericalConfig) -> list[SampledLightAtPlane]:
    """The sampled lines a target on a plane selects: by source and wavelength (a path or crossing selector names a beam)."""
    if obs is None or t.parameters.get("path_id") is not None or t.parameters.get("crossing_index") is not None:
        return []
    return [line for line in obs.sampled_light if line.counted_crossing_count
            and (t.source_id is None or line.source_id == t.source_id)
            and (t.wavelength_nm is None or abs(line.wavelength_nm - t.wavelength_nm) <= cfg.interference_wavelength_tol_nm)]


def _select_sampled_zero(obs: ObservationResult | None, t: Target, cfg: NumericalConfig) -> list[SampledLightAtPlane]:
    """The sampled lines a target on a plane selects none of whose rays is counted there (a sampled zero)."""
    if obs is None or t.parameters.get("path_id") is not None or t.parameters.get("crossing_index") is not None:
        return []
    return [line for line in obs.sampled_light if not line.counted_crossing_count
            and (t.source_id is None or line.source_id == t.source_id)
            and (t.wavelength_nm is None or abs(line.wavelength_nm - t.wavelength_nm) <= cfg.interference_wavelength_tol_nm)]


def _sampled_resolution(mv: MetricValue, zero: list[SampledLightAtPlane]) -> MetricValue:
    """``mv`` judged within plus or minus one ray's weight per selected sampled line none of whose rays is counted on the
    plane (``SampledLightAtPlane``: the sampled zero), as a pupil-sampled beam power is judged (``_power_interval``): a limit
    outside that interval is decided, one within it is not."""
    if not zero or mv.value is None:
        return mv
    resolution = correctly_rounded_sum(line.emitted_W / line.sample_count for line in zero)
    low, high = mv.untraced if mv.untraced is not None else (mv.lower if mv.lower is not None else mv.value, mv.value)
    rays = int(math.fsum(line.samples_traced for line in zero))
    note = (f"none of the {rays} sampled rays of {len(zero)} line(s) is counted on this plane, which resolves their power to "
            f"{resolution:.4g} W (one ray's weight per line)")
    return MetricValue(mv.value, mv.unit, mv.status, mv.codes, f"{mv.message}; {note}" if mv.message else note, lower=mv.lower,
                       untraced=(max(low - resolution, 0.0), high + resolution), subject="within that resolution the plane receives",
                       basis=mv.basis)


def _coherent_power(obs: ObservationResult, arrivals: list[ArrivalResult], unit: str, cfg: NumericalConfig) -> MetricValue:
    """Power through the plane of the selected wavelength's beams summed coherently (``ObservationResult.superpositions``);
    on an absorbing plane the detected power, so a single beam's coherent power equals its ``power_W``."""
    wavelengths = sorted({a.beam.wavelength_nm for a in arrivals})
    if len(wavelengths) > 1:
        return MetricValue(None, unit, Status.UNKNOWN, (ErrorCode.INVALID_INPUT.value,),
                           "beams of several wavelengths match; select one with wavelength_nm")
    if len(arrivals) == 1:
        return _received_power(arrivals, unit)
    for s in obs.superpositions:
        if abs(s.wavelength_nm - wavelengths[0]) <= cfg.interference_wavelength_tol_nm:
            if s.status is Status.OUT_OF_MODEL_SCOPE and any(a.power_out_of_scope_at for a in arrivals):
                # (no coherent sum of a beam whose power is out of model scope: _power_scope names the surfaces)
                return MetricValue(None, unit, Status.UNKNOWN, (), "the coherent sum is not available")
            ranged = s.total_power_min_W is not None and s.total_power_max_W is not None
            if s.total_power_W is None and not ranged:
                return MetricValue(None, unit, s.status, s.issue_codes, "the coherent sum is not available (see issue codes)")
            if len(s.path_ids) != len(arrivals):
                return MetricValue(None, unit, Status.UNKNOWN, (ErrorCode.INVALID_INPUT.value,),
                                   "coherent_power_W sums every beam of the wavelength on the plane; do not narrow it by source or path")
            whole_plane = ErrorCode.SUPERPOSITION_WHOLE_PLANE.value in s.issue_codes
            status = Status.WARNING if s.status is Status.WARNING or whole_plane else Status.PASS
            if s.total_power_W is None:
                # beams of it meet coatings differently that state no phase: the phase between their fields is open, and the
                # power has a range over every phase, no value (validation/superposition.py)
                return MetricValue(None, unit, status, s.issue_codes,
                                   "beams of the coherent sum meet coatings differently that state no phase, so the phase "
                                   "between their fields is open", untraced=(s.total_power_min_W, s.total_power_max_W),  # type: ignore[arg-type]
                                   subject="over every phase the coherent power is")
            if ranged:
                # the sum at the stated model's conventions -- a monochromatic stand-in for a line that states no coherence,
                # the traced delays, the default interface phases -- with its entry, and the range those leave
                # (validation/superposition.py); the entry: the stored one, or -- behind bounded group delays alone, which
                # store none -- the derived ESTIMATE of the traced delays (``results/basis.py`` ``_superposition``)
                from ..results.basis import basis_of  # (imported here: the results package imports modules that import this one)

                entry = basis_of(s, "total_power_W")
                return MetricValue(s.total_power_W, unit, status, s.issue_codes, _coherent_sum_text(s, entry, arrivals),
                                   untraced=(s.total_power_min_W, s.total_power_max_W), subject="the coherent power is",  # type: ignore[arg-type]
                                   basis=entry)
            # (a sum on the trace's conventions with no range beside it -- its open terms carry no power -- keeps its entry)
            return MetricValue(s.total_power_W, unit, status, s.issue_codes, basis=(s.value_basis or {}).get("total_power_W"))
    return MetricValue(None, unit, Status.UNKNOWN, (ErrorCode.NUMERICAL_FAILURE.value,), "no coherent sum was formed")


def _coherent_sum_text(s: Any, entry: ValueBasis | None, arrivals: list[ArrivalResult]) -> str:
    """What a ranged coherent sum's value stands on, from its entry's inputs: a monochromatic stand-in -- at full coherence
    where no line of its pairs states a length, else at the other source's stated length and the traced delays --, and the
    passes the beams meet differently -- of bounded group delay only, or of a group delay or phase the design does not
    state."""
    by = {} if entry is None else {x.input: x for x in entry.assumed}
    full = by.get(AssumedInput.FULL_COHERENCE)
    if full is not None:
        sources = [x.removeprefix("source:") for x in full.at]
        groups = {a.coherence_group for a in arrivals if a.beam.source_id in sources}
        if any(a.coherence_length_mm is not None and a.coherence_group in groups for a in arrivals):
            lead = (f"the coherent sum with source {', '.join(sources)} taken as monochromatic, at the other source's stated "
                    f"coherence length and the traced delays, is {s.total_power_W:.6g} W")
        else:
            lead = f"the coherent sum at full coherence is {s.total_power_W:.6g} W; the source states no coherence"
    else:
        lead = f"the coherent sum at the traced delays is {s.total_power_W:.6g} W"
    delay = by.get(AssumedInput.UNSTATED_GROUP_DELAY_ZERO)
    unbounded = {pass_id(key) for a in arrivals for key in a.group_delay_unknown_at if delay_bound_fs(key) is None}
    if AssumedInput.UNSTATED_PHASE_AS_INTERFACE in by or (delay is not None and unbounded & set(delay.at)):
        return f"{lead}, and beams of it meet passes differently whose group delay or phase the design does not state"
    if delay is not None or entry is None:
        return f"{lead}, and beams of it meet coatings differently whose group delays are known within their stated bounds"
    return lead


def _wavelength_mismatch(obs: ObservationResult | None, t: Target, cfg: NumericalConfig) -> str | None:
    """Why a two-beam target of two named sources has no pair although both reach the plane: two-beam metrics compare
    beams of one wavelength, and these two sources' beams arrive at different wavelengths."""

    if obs is None or t.source_id is None or t.source_id_b is None or t.source_id == t.source_id_b:
        return None
    a = sorted({x.beam.wavelength_nm for x in obs.arrivals if x.beam.source_id == t.source_id})
    b = sorted({x.beam.wavelength_nm for x in obs.arrivals if x.beam.source_id == t.source_id_b})
    if not a or not b:
        return None
    if any(abs(wa - wb) <= cfg.interference_wavelength_tol_nm for wa in a for wb in b):
        return None
    fmt = lambda values: "/".join(f"{v:g} nm" for v in values)  # noqa: E731
    return (f"both beams reach the observation plane, but at different wavelengths ({t.source_id}: {fmt(a)}, "
            f"{t.source_id_b}: {fmt(b)}); two-beam metrics compare two beams of the same wavelength")


def _components_named(t: Target, obs: ObservationResult | None) -> bool:
    """Whether the pair target names, with parameters path_id_a / path_id_b, the two polarization components of one partially
    polarized line (they form no pair: ``optics/polarization/partial.py``)."""
    pa, pb = t.parameters.get("path_id_a"), t.parameters.get("path_id_b")
    if obs is None or pa is None or pb is None:
        return False
    first = {a.beam.path_id: a for a in obs.arrivals}
    a, b = first.get(pa), first.get(pb)
    return (a is not None and b is not None and a.polarization_component is not None and b.polarization_component is not None
            and a.polarization_component != b.polarization_component and a.beam.source_id == b.beam.source_id
            and a.beam.spectral_index == b.beam.spectral_index)


def _target_reverses_pair(t: Target, r: InterferenceResult) -> bool:
    """Whether the target's operand order (a, b) is the reverse of the result's (``path_id_a``, ``path_id_b``)."""
    pa, pb = t.parameters.get("path_id_a"), t.parameters.get("path_id_b")
    if pa is not None:
        if r.path_id_a == r.path_id_b:  # two crossings of one path: the crossing indices give the order
            ca = t.parameters.get("crossing_index_a")
            return ca is not None and ca == r.crossing_index_b and r.crossing_index_a != r.crossing_index_b
        return pa == r.path_id_b and pa != r.path_id_a
    if pb is not None:
        return pb == r.path_id_a and pb != r.path_id_b
    if t.source_id is not None and t.source_id_b is not None and t.source_id != t.source_id_b:
        return t.source_id == r.source_id_b and t.source_id != r.source_id_a
    return False


def _needs_plane_bounds(unit: str, what: str) -> MetricValue:
    """A target no traced beam (pair) matches in an evaluation that did not run the plane bounds: whether it is not met,
    selects nothing or waits on light the trace does not follow is theirs to decide. The target is undetermined
    (``UNKNOWN``, ``CAPABILITY_NOT_COMPUTED``, no value) and its message names the capability to add."""
    return MetricValue(None, unit, Status.UNKNOWN, (ErrorCode.CAPABILITY_NOT_COMPUTED.value,),
                       f"no {what} is among the traced beams on the observation plane; whether light the trace does not "
                       f"follow reaches the plane was not computed: evaluate with {PLANE_BOUNDS}")


def assess_targets(
    targets: tuple[Target, ...],
    observations: dict[str, ObservationResult],
    interference: list[InterferenceResult],
    layout: LayoutResult | None,
    control: ControlResult | None,
    cfg: NumericalConfig,
    power_budget: PowerBudget | None = None,
    reach: PlaneReach | None = None,
    exposures: ExposureReport | None = None,
    metric_scope: dict[str, frozenset[str]] | None = None,
    truncated_pairs: frozenset[str] = frozenset(),
    element_scope: dict[str, tuple[str, ...]] | None = None,
    attenuators: frozenset[str] | None = None,
    met_first: dict[tuple[str, str, int], tuple[tuple[str, ...], tuple[str, ...]]] | None = None,
    actuators: dict[str, tuple[Any, ...]] | None = None,
    detections: dict[str, Any] | None = None,
    conversions: tuple[Any, ...] = (),
    element_power_scope: dict[str, tuple[str, ...]] | None = None,
    plane_bounds: bool = True,
) -> list[TargetResult]:
    """Every target's result. A target whose metric reads a stage the request turns off (the layout checks, the control
    analysis, the trace) is assessed on what the run has: a layout or control target says the stage did not run; without
    the trace a plane target reads its plane as unknown (``UNKNOWN_REFERENCE``), a two-beam target as not met, and the
    rating utilization says the physics stage did not run. ``plane_bounds`` false
    (an evaluation that left out the plane bounds its request enables): a target no traced beam matches is undetermined
    (``CAPABILITY_NOT_COMPUTED``) -- whether light the trace does not follow reaches its plane is the plane bounds' to
    say. ``exposures``: the run's per-surface exposure report (element metrics); ``metric_scope``:
    surface id -> the metrics a response model there was validated for (module docstring); ``truncated_pairs``: the
    observation planes on which only the first ``max_interference_pairs`` beam pairs were analysed (``MAX_BEAM_PAIRS``);
    ``element_scope``: element id -> the response-model surfaces not validated for the exposure of its rated surfaces;
    ``attenuators``: the variable attenuators' surface ids, ``met_first``: per arrival (plane, path, crossing) the surfaces
    part of its beam met first and the one that sent part of it beyond the plane (``_power_scope``); ``actuators``: component
    id -> its ``ActuatorSpec``s, the stated limits the mechanical metrics read (``validation/mechanical.py``); ``detections``:
    observation id -> its ``Detection``, which the detection metrics read (``validation/detection.py``); ``conversions``: the
    run's ``NonlinearConversionResult``s, which the second-harmonic metrics read (``validation/nonlinear.py``);
    ``element_power_scope``: element id -> the surfaces behind which the power of light on its rated surfaces is out of model
    scope (``validation/exposures.py``: ``ExposureOutcome.power_scope``).

    A target that is not assessable (``Target.not_assessable``) reads nothing: its result is ``UNKNOWN`` with the marker's
    code and message, in its place among the others."""
    declared, targets = targets, tuple(t for t in targets if t.not_assessable is None)
    results = [_evaluate(t, observations, interference, layout, control, cfg, power_budget, reach, exposures, truncated_pairs,
                         actuators=actuators, detections=detections, conversions=conversions, plane_bounds=plane_bounds)
               for t in targets]
    converters = frozenset(c.surface_id for c in conversions)
    results = [_power_scope(t, r, observations.get(t.observation_id or ""), cfg, attenuators, met_first, converters)
               for t, r in zip(targets, results)]
    results = [_power_labels(t, r, observations.get(t.observation_id or ""), cfg) for t, r in zip(targets, results)]
    results = [_reference_scope(t, r, observations.get(t.observation_id or ""), cfg) for t, r in zip(targets, results)]
    if element_power_scope:
        results = [_element_power_scope(t, r, element_power_scope, converters) for t, r in zip(targets, results)]
    if metric_scope:
        results = [_metric_scope(t, r, observations.get(t.observation_id or ""), metric_scope, cfg, element_scope or {}, interference)
                   for t, r in zip(targets, results)]
    if len(targets) == len(declared):
        return results
    assessed = iter(results)
    return [next(assessed) if t.not_assessable is None else _not_assessable(t) for t in declared]


def _not_assessable(t: Target) -> TargetResult:
    """The result of a requirement that cannot be read as written: no value, undetermined, its input problem's code."""
    assert t.not_assessable is not None
    return TargetResult(target_id=t.target_id, metric=t.metric, status=Status.UNKNOWN, value=None,
                        message=t.not_assessable.message, issue_codes=(t.not_assessable.code,))


def selected_light(t: Target, obs: ObservationResult | None, cfg: NumericalConfig) -> tuple[list[ArrivalResult], list[SampledLightAtPlane]]:
    """The beams and the sampled lines on its plane whose light a target's value reads, by its selection alone (no pair
    analysis needed, so the answer is the same whatever an evaluation computed): a one-beam metric the beams and sampled
    lines it selects (``_select_arrivals``, ``_select_sampled`` with the lines none of whose rays is counted there; a
    polarization metric of a polarization component both components of its route, ``_route_light``); a geometric pair metric the beams of its two operands (``alignment_operands``); a two-beam metric both beams of every
    pair of one wavelength it selects (``target_selects_pair``), and where it reads the light on the plane
    (``PAIR_PLANE_METRICS``) every beam and sampled line of such a pair's wavelength too. Where a selection matches more
    than one beam or pair, all of them: the target reads one of them at most, and which one it would read is not
    narrowed here."""
    if obs is None:
        return [], []
    tol = cfg.interference_wavelength_tol_nm
    if t.metric in ALIGNMENT_METRICS:
        operands = {id(a) for beams in alignment_operands(t, obs, cfg) for a in beams}
        return [a for a in obs.arrivals if id(a) in operands], []
    if t.metric in PAIR_METRICS:
        arrivals = list(obs.arrivals)
        chosen: set[int] = set()
        wavelengths: list[float] = []
        for i, a in enumerate(arrivals):
            for j in range(i + 1, len(arrivals)):
                b = arrivals[j]
                if abs(a.beam.wavelength_nm - b.beam.wavelength_nm) <= tol and target_selects_pair(
                        t, a.beam.source_id, b.beam.source_id, a.beam.wavelength_nm, a.beam.path_id, a.crossing_index,
                        b.beam.path_id, b.crossing_index, cfg):
                    chosen |= {i, j}
                    wavelengths.append(a.beam.wavelength_nm)
        if t.metric not in PAIR_PLANE_METRICS:
            return [arrivals[k] for k in sorted(chosen)], []

        def of_pair(wavelength: float) -> bool:
            return any(abs(wavelength - w) <= tol for w in wavelengths)

        return ([a for k, a in enumerate(arrivals) if k in chosen or of_pair(a.beam.wavelength_nm)],
                [line for line in obs.sampled_light if of_pair(line.wavelength_nm)])
    return (_route_light(t, obs, _select_arrivals(obs, t, cfg)[0]),
            [*_select_sampled(obs, t, cfg), *_select_sampled_zero(obs, t, cfg)])


def _route_light(t: Target, obs: ObservationResult, arrivals: list[ArrivalResult]) -> list[ArrivalResult]:
    """The beams a target's value reads among ``arrivals`` it selects: a polarization metric (``POLARIZATION_METRICS``)
    of a polarization component reads the light of its route, both components (``mixture_metric``), whichever component's
    path id selects it; any other metric the selected beams."""
    if t.metric not in POLARIZATION_METRICS:
        return arrivals
    return route_arrivals(arrivals, obs)


def _scope_arrivals(t: Target, obs: ObservationResult | None, cfg: NumericalConfig,
                    interference: list[InterferenceResult] | None = None) -> list[ArrivalResult]:
    """The beams whose lineage decides a target's scope (module docstring). ``interference``: the run's pairs, so that a pair
    or geometric pair target reads the two beams of the pair it selects (without them, or where it selects no single pair,
    every beam of its sources)."""
    if obs is None:
        return []
    if t.metric in PAIR_METRICS or t.metric in ALIGNMENT_METRICS:
        pair = None if interference is None else _selected_pair(t, obs, cfg, interference)
        if pair is not None:
            return pair
        sources = {t.source_id, t.source_id_b} - {None}
        return [a for a in obs.arrivals if not sources or a.beam.source_id in sources]
    return _route_light(t, obs, _select_arrivals(obs, t, cfg)[0])


def _selected_pair(t: Target, obs: ObservationResult, cfg: NumericalConfig,
                   interference: list[InterferenceResult]) -> list[ArrivalResult] | None:
    """The two arrivals a pair or geometric pair target reads, where it selects exactly one pair; ``None`` otherwise."""
    if t.metric in ALIGNMENT_METRICS:
        a_beams, b_beams = alignment_operands(t, obs, cfg)
        return [a_beams[0], b_beams[0]] if len(a_beams) == 1 and len(b_beams) == 1 else None
    pairs = [r for r in interference if r.observation_id == obs.observation_id and target_selects_pair(
        t, r.source_id_a, r.source_id_b, r.wavelength_nm, r.path_id_a, r.crossing_index_a, r.path_id_b, r.crossing_index_b, cfg)]
    if len(pairs) != 1:
        return None
    keys = {(pairs[0].path_id_a, pairs[0].crossing_index_a), (pairs[0].path_id_b, pairs[0].crossing_index_b)}
    return [a for a in obs.arrivals if (a.beam.path_id, a.crossing_index) in keys]


def _power_scope(t: Target, result: TargetResult, obs: ObservationResult | None, cfg: NumericalConfig,
                 attenuators: frozenset[str] | None = None,
                 met_first: dict[tuple[str, str, int], tuple[tuple[str, ...], tuple[str, ...]]] | None = None,
                 converters: frozenset[str] = frozenset()) -> TargetResult:
    """``result``, or ``OUT_OF_MODEL_SCOPE`` when a power-derived target's beams include one whose power is out of model
    scope (module docstring; ``attenuators``: the variable attenuators' surface ids, ``None``: every surface other than an
    observation plane; ``met_first``: per arrival the surfaces part of its beam met before the plane and the surface whose
    reflection sent part of it beyond the plane -- an attenuator among them is met, not crossed)."""
    if t.metric not in POWER_DERIVED_METRICS or result.status is Status.OUT_OF_MODEL_SCOPE:
        return result
    scoped = _scope_arrivals(t, obs, cfg)
    # (a bounded share taken before the plane bounds a beam's power only: ``power_W`` is judged on its interval,
    # ``_power_interval``; the other power-derived metrics are out of scope as for an unbounded share)
    taken = set() if t.metric == "power_W" else {surface for a in scoped for surface in a.power_taken_before_at}
    at = sorted({surface for a in scoped for surface in a.power_out_of_scope_at} | taken)
    # (light a surface scattered out of a beam whose power is out of model scope: SampledLightAtPlane.power_out_of_scope_at)
    scattered = [] if t.metric in PAIR_METRICS or t.metric in ALIGNMENT_METRICS else [
        line for line in _select_sampled(obs, t, cfg) if line.power_out_of_scope_at]
    if not at and not scattered:
        return result
    kinds = [(met_first or {}).get((obs.observation_id if obs is not None else "", a.beam.path_id, a.crossing_index), ((), ()))
             for a in scoped]
    first = {surface for met_ids, _ in kinds for surface in met_ids} | taken
    left = {surface for _, left_ids in kinds for surface in left_ids}
    # (a coating a beam met outside its data, its values held at the nearest stated point: the beam's stored entry names it)
    entries = [entry for a in scoped if (entry := (a.value_basis or {}).get("beam.power_W")) is not None]
    held = sorted({surface for entry in entries for assumed in entry.assumed
                   if assumed.input is AssumedInput.HELD_AT_NEAREST_STATED_POINT for surface in assumed.at} & set(at))
    at = [surface for surface in at if surface not in held]
    crystals = [surface for surface in at if surface in converters and surface not in first and surface not in left]
    at = [surface for surface in at if surface not in crystals]
    planes = [surface for surface in at if surface.startswith(OBSERVATION_SURFACE_PREFIX)]
    nd = [surface for surface in at if not surface.startswith(OBSERVATION_SURFACE_PREFIX) and surface not in first
          and surface not in left and (attenuators is None or surface in attenuators)]
    beyond = [surface for surface in at if surface in left and surface not in first]
    met = [surface for surface in at if surface not in planes and surface not in nd and surface not in beyond]
    parts, codes = [], []
    if scattered:
        behind = sorted({surface for line in scattered for surface in line.power_out_of_scope_at})
        by = sorted({line.scatter_surface_id or line.source_id for line in scattered})
        parts.append(f"light {', '.join(by)} scattered out of a beam whose power is out of model scope behind {', '.join(behind)} "
                     "reaches this plane: its power follows that beam's chief ray, not the light's")
        codes.append(ErrorCode.NONLINEAR_CONVERSION_OUT_OF_SCOPE if any(s in converters for s in behind) else ErrorCode.OUT_OF_MODEL_SCOPE)
    if crystals:
        parts.append(f"a beam of this target entered the crystal at {', '.join(crystals)} whose second-harmonic generation is out of "
                     "model scope there: the power the fundamental and the harmonic carry behind it is not known within the model")
        codes.append(ErrorCode.NONLINEAR_CONVERSION_OUT_OF_SCOPE)
    if nd:
        parts.append(f"a beam of this target crossed the variable attenuator at {', '.join(nd)} where its optical "
                     "density varies across the beam beyond the tolerance: its power is the chief ray's transmittance applied to "
                     "the whole beam")
        codes.append(ErrorCode.ND_DENSITY_VARIES_ACROSS_BEAM)
    if planes:
        names = ", ".join(surface.removeprefix(OBSERVATION_SURFACE_PREFIX) for surface in planes)
        parts.append(f"part of a beam of this target meets the absorbing plane {names} before this plane, where the beam's chief "
                     "ray does not: the power that part takes away is not computed")
        codes.append(ErrorCode.OUT_OF_MODEL_SCOPE)
    if met or beyond:
        how = ([f"meets {', '.join(met)} before the plane"] if met else []) + (
            [f"is reflected by {', '.join(beyond)} beyond the plane"] if beyond else [])
        parts.append(f"part of a beam of this target does not reach this plane with the beam: it {' or '.join(how)}, while the "
                     "power recorded here is the whole beam's; the power that reaches the plane is not computed")
        if ErrorCode.OUT_OF_MODEL_SCOPE not in codes:
            codes.append(ErrorCode.OUT_OF_MODEL_SCOPE)
    if held:
        only = not parts and result.value is not None
        parts.append(f"a beam of this target met the coating at {', '.join(held)} outside the wavelengths or angles its data "
                     "covers: its power behind it uses the coating's values at the nearest stated point")
        codes.append(ErrorCode.COATING_OUT_OF_RANGE)
        if only:
            # (nothing else qualifies the value: it is reported as the estimate it is, and decides nothing)
            scoped = result.model_copy(update={"issue_codes": _scope_codes(result, ErrorCode.COATING_OUT_OF_RANGE)})
            return scoped.model_copy(update={
                # (the target's own entry -- light not followed, a stated truncation -- and the held coatings' entries: B6)
                "status": Status.OUT_OF_MODEL_SCOPE,
                BASIS_FIELD: {"value": merge_bases((result.value_basis or {}).get("value"), *entries, monotone=None)},
                "message": parts[0] + f", so {t.metric} is an estimate (value_basis) and does not decide the target"})
    message = "; ".join(parts) + f", so {t.metric} is out of model scope"
    scoped = result
    for code in codes:
        scoped = scoped.model_copy(update={"issue_codes": _scope_codes(scoped, code)})
    return scoped.model_copy(update={"status": Status.OUT_OF_MODEL_SCOPE, "value": None, "value_basis": None, "error": None, "message": message})


def _element_power_scope(t: Target, result: TargetResult, scope: dict[str, tuple[str, ...]],
                         converters: frozenset[str]) -> TargetResult:
    """``result``, or ``OUT_OF_MODEL_SCOPE`` when an element metric's element has light on its rated surfaces whose power is
    out of model scope (module docstring)."""
    if t.metric not in ELEMENT_METRICS or result.status is Status.OUT_OF_MODEL_SCOPE:
        return result
    at = scope.get(t.device_id or "", ())
    if not at:
        return result
    crystals = [surface for surface in at if surface in converters]
    others = [surface for surface in at if surface not in converters]
    parts, codes = [], []
    if crystals:
        parts.append(f"light on {t.device_id}'s rated surfaces passed the crystal at {', '.join(crystals)}, whose second-harmonic "
                     "generation is out of model scope there: the power and pulse energy the fundamental and the harmonic carry "
                     "behind it are not known within the model")
        codes.append(ErrorCode.NONLINEAR_CONVERSION_OUT_OF_SCOPE)
    if others:
        parts.append(f"light on {t.device_id}'s rated surfaces carries a power out of model scope behind {', '.join(others)} "
                     "(the beam's power follows its chief ray there, not the light's)")
        codes.append(ErrorCode.OUT_OF_MODEL_SCOPE)
    scoped = result
    for code in codes:
        scoped = scoped.model_copy(update={"issue_codes": _scope_codes(scoped, code)})
    message = "; ".join(parts) + f", so its exposure and {t.metric} are out of model scope"
    return scoped.model_copy(update={"status": Status.OUT_OF_MODEL_SCOPE, "value": None, "value_basis": None, "error": None, "message": message})


def _reference_scope(t: Target, result: TargetResult, obs: ObservationResult | None, cfg: NumericalConfig) -> TargetResult:
    """``result``, or ``OUT_OF_MODEL_SCOPE`` (``CLIPPED``) when a target on a chief-ray metric (``REFERENCE_RAY_METRICS``)
    reads a beam whose reference ray was promoted at a clear aperture that took its chief ray (module docstring): the value
    stays the reference ray's, as an estimate that is not graded (``_not_graded``)."""
    if t.metric not in REFERENCE_RAY_METRICS or result.status is Status.OUT_OF_MODEL_SCOPE or obs is None:
        return result
    at = sorted({surface for a in _select_arrivals(obs, t, cfg)[0] for surface in a.reference_ray_promoted_at})
    if not at:
        return result
    message = (f"the chief ray of a beam of this target fell outside the clear aperture of {', '.join(at)} while some of its "
               f"pupil rays passed: the beam goes on as those rays, and {t.metric} on this plane reads the one pupil ray "
               "that carries its reference (the passing ray nearest their centroid), not a property of the beam, so "
               f"{t.metric} is out of model scope and its value, that ray's, is not graded; spot_centroid_u_mm and "
               "spot_centroid_v_mm give the centroid of the pupil samples that pass, the beam's position within the samples' "
               "resolution")
    return _not_graded(result, ErrorCode.CLIPPED, message,
                       Assumed(input=AssumedInput.SINGLE_RAY_FOR_BEAM, at=tuple(at), code=ErrorCode.CLIPPED.value))


def _scope_codes(result: TargetResult, code: ErrorCode) -> tuple[str, ...]:
    """A target out of model scope is not compared: its codes say why, never ``TARGET_NOT_MET``."""
    return tuple(dict.fromkeys(c for c in (*result.issue_codes, code.value) if c != ErrorCode.TARGET_NOT_MET.value))


def _not_graded(result: TargetResult, code: ErrorCode, message: str, assumed: Assumed | None) -> TargetResult:
    """``result`` out of model scope with ``code``: where it has a value and ``assumed`` names the stand-in that makes the
    number an estimate, the value stays, with that entry merged into ``value_basis["value"]`` (rule T1: reported, with its
    error against the target, and not decided); otherwise no value."""
    update: dict[str, Any] = {"status": Status.OUT_OF_MODEL_SCOPE, "message": message, "issue_codes": _scope_codes(result, code)}
    if result.value is None or assumed is None:
        update.update(value=None, error=None, value_basis=None)
    else:
        own = ValueBasis(kind=ValueKind.ESTIMATE, assumed=(assumed,))
        update["value_basis"] = {"value": merge_bases((result.value_basis or {}).get("value"), own, monotone=None)}
    return result.model_copy(update=update)


def _metric_scope(t: Target, result: TargetResult, obs: ObservationResult | None, scope: dict[str, frozenset[str]],
                  cfg: NumericalConfig, element_scope: dict[str, tuple[str, ...]],
                  interference: list[InterferenceResult] | None = None) -> TargetResult:
    """``result``, or ``OUT_OF_MODEL_SCOPE`` when the target's beams crossed a response model not validated for its metric
    (for an element metric: for the exposure of the element's rated surfaces behind it). A beam, pair or alignment metric
    keeps the value the Engine computed, as an estimate that is not graded (``_not_graded``); an element metric has no
    value there (no rating verdict rests on an exposure the response model was not validated for)."""
    element = t.metric in ELEMENT_METRICS
    if element:
        crossed = sorted(set(element_scope.get(t.device_id or "", ())))
        what = f"the exposure of {t.device_id}'s rated surfaces behind it (power_W with a beam-size metric, or {t.metric})"
    else:
        crossed = sorted({step.rsplit(":", 1)[0] for a in _scope_arrivals(t, obs, cfg, interference) for step in a.lineage
                          if step.rsplit(":", 1)[0] in scope and t.metric not in scope[step.rsplit(":", 1)[0]]})
        what = t.metric
    if not crossed:
        return result
    validated = "; ".join(f"{surface}: {', '.join(sorted(scope[surface]))}" for surface in crossed)
    message = (f"the beams of this target crossed the response model at {', '.join(crossed)}, which is not validated for "
               f"{what} (validated metrics -- {validated})")
    kept = not element and result.value is not None
    if kept:
        message += "; the value is computed with the element's ideal thin-lens step and is not graded"
    assumed = None if element else Assumed(input=AssumedInput.RESPONSE_MODEL_UNVALIDATED, at=tuple(crossed),
                                           code=ErrorCode.METRIC_NOT_VALIDATED_FOR_ELEMENT.value)
    return _not_graded(result, ErrorCode.METRIC_NOT_VALIDATED_FOR_ELEMENT, message, assumed)


def _target_lines(t: Target, power_budget: PowerBudget | None, cfg: NumericalConfig) -> list:
    """The target's source lines: its ``source_id`` / ``source_id_b`` and ``wavelength_nm`` (every line when it names none)."""
    if power_budget is None:
        return []
    sources = {t.source_id, t.source_id_b} - {None}
    return [line for line in power_budget.lines
            if (not sources or line.source_id in sources)
            and (t.wavelength_nm is None or abs(line.wavelength_nm - t.wavelength_nm) <= cfg.interference_wavelength_tol_nm)]


def _fate(t: Target, power_budget: PowerBudget | None, cfg: NumericalConfig, reach: PlaneReach | None = None,
          obs: ObservationResult | None = None) -> LineFate:
    """The fate of the target's source lines. For a power summed over the selected lines, it includes the light the trace does
    not follow that may reach the target's plane from those lines (``reach``: from the lines no traced beam brings to it,
    and -- in a run that followed every untraced branch -- from the lines whose traced beams arrive there too). A ``path_id``
    or ``crossing_index`` that selects an arrival reads that arrival: the light beside it is not its power, so the ledger alone.
    With no selected arrival use ``_dark_fate``."""
    lines = _target_lines(t, power_budget, cfg)
    if reach is None or obs is None or t.parameters.get("path_id") is not None or t.parameters.get("crossing_index") is not None:
        return line_fate(lines, cfg)
    return _reach_fate(lines, t.observation_id, reach, cfg)


def _reach_fate(lines: list, observation_id: str | None, reach: PlaneReach, cfg: NumericalConfig) -> LineFate:
    keys = [(line.source_id, line.spectral_index) for line in lines]
    return line_fate(lines, cfg, hazards(reach, observation_id, keys), hazard_root_sum(reach, observation_id, keys),
                     faint_root_sum(reach, observation_id, keys), faint_power_sum(reach, observation_id, keys),
                     withheld_by_work(reach, observation_id, keys), return_causes(reach, observation_id, keys))


def _dark_fate(t: Target, power_budget: PowerBudget | None, cfg: NumericalConfig, reach: PlaneReach | None,
               obs: ObservationResult | None) -> LineFate:
    """The fate of the target's source lines where it selects no arrival: always with the light that may reach the plane,
    whatever its selectors, since part of a traced beam crossing the plane where its chief ray does not is that very path's
    own light (a ``path_id``: of that path's line)."""
    lines = _target_lines(t, power_budget, cfg)
    if reach is None or obs is None:
        return line_fate(lines, cfg)
    path = t.parameters.get("path_id")
    if path is not None and power_budget is not None:
        owners = {(e.source_id, e.spectral_index) for e in power_budget.entries if e.path_id == path}
        lines = [line for line in lines if not owners or (line.source_id, line.spectral_index) in owners]
    return _reach_fate(lines, t.observation_id, reach, cfg)


def _fmt_list(values, limit: int = 8) -> str:
    items = [str(v) for v in values]
    return ", ".join(items[:limit]) + (f", ... ({len(items)} in all)" if len(items) > limit else "") if items else "none"


def _selection_problem(t: Target, obs: ObservationResult | None, power_budget: PowerBudget | None, cfg: NumericalConfig,
                       fate: LineFate, *, pair: bool) -> str | None:
    """Why the target's selectors select no beam of this run (``None``: they are satisfiable). A source or wavelength that
    no launched line has is always empty. A path or a later crossing (``crossing_index`` >= 1) that does not occur is empty
    only when the selected lines were traced completely (otherwise the untraced light might have formed it)."""
    if power_budget is None:
        return None
    launched = sorted({line.source_id for line in power_budget.lines})
    for name in ("source_id", "source_id_b"):
        source = getattr(t, name)
        if source is not None and source not in launched:
            return f"{name} {source!r} is not a source of this run (sources: {_fmt_list(launched)})"
    sources = {t.source_id, t.source_id_b} - {None}
    lines = [line for line in power_budget.lines if not sources or line.source_id in sources]
    if t.wavelength_nm is not None and not any(abs(line.wavelength_nm - t.wavelength_nm) <= cfg.interference_wavelength_tol_nm
                                               for line in lines):
        offered = sorted({line.wavelength_nm for line in lines})
        return (f"no source line of the selected source(s) is at {t.wavelength_nm:g} nm (launched lines: "
                f"{_fmt_list(f'{w:g} nm' for w in offered)})")
    if not fate.complete:
        return None
    keys = {(line.source_id, line.spectral_index) for line in _target_lines(t, power_budget, cfg)}
    paths = sorted({e.path_id for e in power_budget.entries if e.path_id is not None and (e.source_id, e.spectral_index) in keys})
    # the crossings the selected beams make: of the selected sources and wavelength only (another source's later crossing
    # of the plane does not make this one occur)
    arrivals = [a for a in ([] if obs is None else obs.arrivals) if (a.beam.source_id, a.beam.spectral_index) in keys]
    if pair:
        selectors = (("path_id_a", "crossing_index_a"), ("path_id_b", "crossing_index_b"))
    else:
        selectors = (("path_id", "crossing_index"),)
    for path_key, crossing_key in selectors:
        path = t.parameters.get(path_key)
        if path is not None and path not in paths:
            return f"{path_key} {path!r} is not a beam path of the selected source lines in this run (paths: {_fmt_list(paths)})"
        crossing = t.parameters.get(crossing_key)
        if crossing is None or crossing == 0:  # the first crossing: a path that misses the plane is not reached
            continue
        matching = [a for a in arrivals if path is None or a.beam.path_id == path]
        if crossing < 0 or not any(a.crossing_index == crossing for a in matching):
            seen = sorted({a.crossing_index for a in matching})
            return (f"{crossing_key} {int(crossing)} does not occur: the selected beams cross this plane "
                    f"{'at crossing indices ' + _fmt_list(seen) if seen else 'never'}")
    return None


def _empty_selection(unit: str, problem: str) -> MetricValue:
    return MetricValue(None, unit, Status.UNKNOWN, (ErrorCode.TARGET_SELECTION_EMPTY.value,),
                       f"the target selects no beam: {problem}")


def _not_followed(fate: LineFate) -> str:
    """What happened to the light of the target's source lines that the result does not count: light not followed, or part
    of a traced beam reaching the plane where its chief ray does not cross it (``PARTIAL_CROSSING``), or both."""
    if fate.partial_only:
        return "part of a traced beam of the target's source lines reaches it where the beam's chief ray does not cross it"
    if PARTIAL_CROSSING in fate.causes():
        return f"part of the target's source lines was not followed or reaches it only in part ({fate.describe()})"
    return f"part of the target's source lines was not followed ({fate.describe()})"


def _not_traced(unit: str, fate: LineFate, what: str) -> MetricValue:
    return MetricValue(None, unit, Status.UNKNOWN, fate.cause_codes(),
                       f"no traced {what} reaches the observation plane, and {_not_followed(fate)}: whether that light reaches it "
                       "is not known")


def _absorbing(obs: ObservationResult | None) -> bool:
    """Whether the plane stops the light it receives (each watt is received at most once)."""
    return obs is not None and obs.plane is not None and obs.plane.absorbing


def _square(x: float) -> float:
    return x * x


def _untraced_bounds(mv: MetricValue, fate: LineFate, obs: ObservationResult | None, *, coherent: bool) -> MetricValue:
    """``_untraced_interval``, whose upper end for ``power_W`` on an absorbing plane is at most the power the target's lines
    launch (``observations.capped_by_launch``: each watt is absorbed once): an interval end above it says nothing."""
    out = _untraced_interval(mv, fate, obs, coherent=coherent)
    if mv.basis is not None and out.basis is None and out.value is not None:  # (the value keeps the entry it was stated with)
        out = replace(out, basis=mv.basis)
    if coherent or out.untraced is None or not _absorbing(obs) or out.untraced[1] <= fate.input_W:
        return out
    low, high = out.untraced
    capped = capped_by_launch(high, max(fate.input_W, low))
    # The cap always applies numerically. Explain it when it changes the displayed bound.
    if f"{high:.4g}" == f"{capped:.4g}":
        return replace(out, untraced=(low, capped))
    note = f"this absorbing plane receives at most the {fate.input_W:.4g} W its lines launch"
    return replace(out, untraced=(low, capped), message=f"{out.message}; {note}" if out.message else note)


def _untraced_interval(mv: MetricValue, fate: LineFate, obs: ObservationResult | None, *, coherent: bool) -> MetricValue:
    """A power summed over the traced beams while part of the target's source lines was not followed (module docstring):
    the interval the plane's power lies in, from the traced value and the untraced power ``U = fate.untraced_W``.

    * ``power_W`` (beam powers added): the traced light is there whatever the rest does, so the traced value is a lower
      bound; on an absorbing plane each untraced watt is received at most once, so traced + ``U`` is an upper bound. A
      transparent plane may be crossed again and again by light the trace did not follow (a cavity, a retro-reflector
      behind it): no upper bound.
    * ``coherent_power_W`` (fields added): untraced light of the same wavelength may interfere with the traced light either
      way, and several untraced contributions with one another. With ``P`` the traced coherent power and ``U_i`` the
      power of each separate contribution the probes followed to the plane (a crossing, a tail), the triangle inequality
      of the field amplitudes gives ``(sqrt(P) - S)^2 <= power <= (sqrt(P) + S)^2`` with ``S = sum_i sqrt(U_i)`` (the lower
      end 0 when ``S >= sqrt(P)``). Untraced power of unknown fate (the ledger's unknown-fate sinks, light the probes did
      not follow) may arrive in any number of beams: then 0 and no upper bound. In a run that followed every untraced
      branch, ``S`` also holds the amplitude of the light the plane's power does not count (``LineFate.faint_root``: the
      light left out as negligible, the sum of the square roots of its pieces' powers times its open paths' crossing
      field, and the followed contributions at or below the negligible level): however faint, its field adds to the traced
      light's (``FAINT_LIGHT_BOUNDED``).
    * In such a run ``power_W``'s upper bound also adds the power that light may bring (``LineFate.faint_W``: on an
      absorbing plane its power, on a transparent one times its open paths' crossings, more where surfaces facing each
      other may hold it and send it across the plane -- no upper bound where nothing bounds their returns;
      ``FAINT_LIGHT_BOUNDED``).

    Where every launched watt of the target's lines has a known end (``fate.complete``), the light left out as negligible is
    the stated truncation of the value and nothing else: the result names ``FAINT_LIGHT_BOUNDED``, its ``value_basis`` entry
    holds the most that light may add (``_truncation``), and the target is judged on the traced value and that bound
    (``_left_out_reach``)."""
    if mv.value is not None and not coherent and fate.complete and fate.faint_W > 0.0 and fate.faint_accounted:
        # (the stated truncation of the value: the entry holds the bound, which the grading reads)
        basis = merge_bases(mv.basis, _truncation(fate, fate.faint_W, fate.faint_W))
        truncated = replace(mv, codes=tuple(mv.codes) + (_faint_code(fate, fate.faint_W),),
                            message=_unbounded_note(mv.message, fate, fate.faint_W, "power"), basis=basis)
        if basis is not None and basis.kind is not ValueKind.MODEL:
            # (a value that is not the model's value -- a share taken before the plane, an annular beam's detected power:
            # the grading does not read the left-out bound from its entry (``_left_out_reach``), so the interval itself
            # counts that light, as for a coherent sum (W28))
            low = mv.untraced[0] if mv.untraced is not None else (mv.lower if mv.lower is not None else mv.value)
            high = mv.untraced[1] if mv.untraced is not None else mv.value
            return replace(truncated, untraced=(low, high + fate.faint_W))
        return truncated
    if mv.value is not None and not coherent and fate.complete and fate.faint_W > 0.0:
        # (above the negligible level here -- its returns or crossings -- that light may add: the interval it leaves, from
        # the traced value; a power already known only within an interval, ``_power_interval``, widens from its ends)
        low = mv.untraced[0] if mv.untraced is not None else (mv.lower if mv.lower is not None else mv.value)
        upper = (mv.untraced[1] if mv.untraced is not None else mv.value) + fate.faint_W
        message = (f"the traced light gives {mv.value:.6g} W; light the trace leaves out as negligible may add "
                   + (f"at most {fate.faint_W:.4g} W" if math.isfinite(upper) else
                      f"power without bound ({fate.unbounded_returns()})"))
        if mv.untraced is not None:
            message = f"{mv.message}; {message}"
        return MetricValue(mv.value, mv.unit, mv.status, tuple(mv.codes) + (_faint_code(fate, fate.faint_W),), message,
                           lower=mv.lower, untraced=(low, upper), basis=merge_bases(mv.basis, _returned(fate, fate.faint_W)))
    # (a coherent sum whose beams meet coatings differently with bounded group delays is itself an interval, ``mv.untraced``,
    # and so is a power sampled by pupil rays or taken in part before the plane, ``_power_interval``: the light the trace
    # does not follow widens it from its ends)
    banded = mv.untraced is not None
    # (what the traced light gives: its value, or the coherent sum's range where it has no value)
    traced = f"the traced light gives {mv.value:.6g} W" if mv.value is not None else mv.message
    if (mv.value is not None or banded) and coherent and fate.complete and fate.faint_root > 0.0:
        low = mv.untraced[0] if banded else (mv.lower if mv.lower is not None else mv.value)  # type: ignore[index]
        high = mv.untraced[1] if banded else mv.value  # type: ignore[index]
        # (the most that light's field may add to the sum, ``(sqrt(P) + root)^2 - P``, and the interval it leaves)
        added = _square(math.sqrt(high) + fate.faint_root) - high
        gap = math.sqrt(low) - fate.faint_root
        widened = (gap * gap if gap > 0.0 else 0.0, _square(math.sqrt(high) + fate.faint_root))
        if not (fate.faint_accounted and added <= fate.level_W):
            # (above the negligible level here -- its power, or the bound of its field at this sum (W1) -- that light's
            # field may move the coherent sum either way: the interval it leaves)
            message = (f"{traced}; light the trace leaves out as negligible may add a field of at "
                       f"most sqrt({fate.faint_root ** 2:.4g} W) to it, with an open phase" if math.isfinite(fate.faint_root) else
                       f"{traced}; light the trace leaves out as negligible may add a field without "
                       f"bound to it ({fate.unbounded_returns()})")
            if banded and mv.value is not None:
                message = f"{mv.message}; {message}"
            # (the entry's bound: the most the value may move inside that interval -- for an unbanded sum the rise ``added``,
            # which exceeds the fall)
            moves = max(widened[1] - mv.value, mv.value - widened[0]) if mv.value is not None else None
            return MetricValue(mv.value, mv.unit, mv.status, tuple(mv.codes) + (_faint_code(fate, fate.faint_root),), message,
                               lower=mv.lower, untraced=widened,
                               basis=merge_bases(mv.basis, _returned(fate, fate.faint_root, coherent=True, upper=moves))
                               if mv.value is not None else mv.basis)
        # (as for the power: the traced coherent sum stands; the entry holds the most that light's field may add to it)
        basis = merge_bases(mv.basis, _truncation(fate, fate.faint_root, added)) if mv.value is not None else mv.basis
        codes, note = tuple(mv.codes) + (_faint_code(fate, fate.faint_root),), _unbounded_note(mv.message, fate, fate.faint_root, "field")
        if mv.value is None or basis is None or basis.kind is not ValueKind.MODEL:
            # (a sum with no value, or one that is not the model's value: the grading does not read the left-out bound from
            # a MODEL entry (``_left_out_reach``), so the interval itself counts that light's field (W28))
            return replace(mv, codes=codes, message=note, untraced=widened, basis=basis)
        return replace(mv, codes=codes, message=note, basis=basis)
    if (mv.value is None and not banded) or fate.complete:
        return mv
    untraced = fate.untraced_W
    low = mv.untraced[0] if banded else (mv.lower if mv.lower is not None else mv.value)  # type: ignore[index]
    high = mv.untraced[1] if banded else mv.value  # type: ignore[index]
    absorbing = _absorbing(obs)
    # light the probes followed to the plane is counted at every crossing: a finite bound on any plane; light of unknown
    # fate may cross a transparent plane any number of times
    bounded = absorbing or not fate.unbounded
    if coherent:
        # several mutually coherent untraced contributions add their field amplitudes: the bound uses the sum of their square
        # roots (each a separately probed contribution); light of unknown fate has an unknown multiplicity -- no bound
        known = not fate.unbounded
        root_u = fate.root_sum + fate.faint_root
        gap = math.sqrt(low) - root_u
        lower = gap * gap if known and gap > 0.0 else 0.0
        upper = (math.sqrt(high) + root_u) ** 2 if known else math.inf
    else:
        lower, upper = low, (high + untraced + fate.faint_W if bounded else math.inf)
    if coherent:
        reach = (f"{untraced:.4g} W of that light in separate contributions, followed to every crossing of this plane, may "
                 f"interfere with the traced light (their amplitudes add up to at most sqrt({fate.root_sum ** 2:.4g} W))"
                 if not fate.unbounded else
                 "part of that light has an unknown fate and may interfere with the traced light in any number of beams, so "
                 "nothing bounds the coherent sum from above")
    else:
        reach = (f"that part adds at most {untraced:.4g} W" if fate.partial_only else
                 f"on this absorbing plane that light adds at most {untraced:.4g} W" if absorbing else
                 f"that light, followed through the optics to every crossing of this plane, adds at most {untraced:.4g} W"
                 if bounded else
                 "this plane does not absorb, so that light may cross it more than once and bounds it from above by nothing")
    if not fate.unbounded and not math.isfinite(fate.faint_root if coherent else fate.faint_W):
        # (what leaves the interval open above when the followed light itself is bounded)
        reach += (f"; light the trace leaves out as negligible may add {'a field' if coherent else 'power'} without bound "
                  f"({fate.unbounded_returns()})")
    message = f"{traced}, and {_not_followed(fate)}{':' if fate.partial_only else ';'} {reach}"
    if banded and mv.value is not None:
        message = f"{mv.message}; {message}"
    faint_amount = fate.faint_root if coherent else fate.faint_W
    faint = (_faint_code(fate, faint_amount),) if not fate.unbounded and faint_amount > 0.0 else ()
    basis = mv.basis
    if mv.value is not None:
        # (what the value does not count, by cause: the traced value is the lower end of a power that adds beams' powers; a
        # coherent sum, or a value already known only within an interval, is moved either way)
        kind = ValueKind.LOWER_BOUND if not coherent and not banded and mv.lower is None else ValueKind.ESTIMATE
        gap = upper - mv.value if not coherent and math.isfinite(upper) else None
        # (a coherent sum's parts: the most each one's field moves the value from the traced ends, ``left_out_parts``)
        basis = merge_bases(basis, ValueBasis(kind=kind, omitted=left_out_parts(
                                fate, absorbing=absorbing, gap=gap, coherent=(low, high, mv.value) if coherent else None)),
                            monotone="increasing" if kind is ValueKind.LOWER_BOUND else None)
    return MetricValue(mv.value, mv.unit, mv.status, tuple(mv.codes) + fate.cause_codes() + faint, message, lower=mv.lower,
                       untraced=(lower, upper), basis=basis)


def _truncation(fate: LineFate, amount: float, added: float) -> ValueBasis:
    """The entry of a power whose only uncounted light is left out as negligible (below the lines' negligible level):
    the model's value with its stated truncation -- ``added`` (W) the most that light may add to it, absent where nothing
    bounds its returns; ``amount`` what ``_faint_code`` reads."""
    return ValueBasis(kind=ValueKind.MODEL, omitted=(Omitted(
        part=OmittedPart.BELOW_NEGLIGIBLE_LEVEL, upper=added if math.isfinite(added) else None,
        setting="negligible_power_fraction", code=_faint_code(fate, amount)),))


def _left_out_reach(t: Target, basis: ValueBasis | None) -> float | None:
    """The most the light left out as negligible may move a plane power target's value (W): the sum of the ``upper`` of the
    accounting omissions of its ``MODEL`` entry (``_truncation``; ``math.inf`` where one states no bound), ``None`` when
    there is none. The plane keeps its state and its value (B4); the grading reads this bound (``_left_out_ends``). A
    bound kind already decides only in the direction it proves (``grade_point``), and an interval that counts that light
    holds it already (``_untraced_interval``)."""
    if t.metric not in POWER_SUM_METRICS or basis is None or basis.kind is not ValueKind.MODEL:
        return None
    parts = [o for o in basis.omitted if o.accounting]
    if not parts:
        return None
    reach = math.inf if any(o.upper is None for o in parts) else math.fsum(o.upper for o in parts)  # type: ignore[misc]
    return reach if reach > 0.0 else None


def _left_out_ends(t: Target, value: float, reach: float) -> tuple[float, float]:
    """Where a power ``value`` lies when light left out as negligible may move it by at most ``reach`` (W): ``power_W`` adds
    powers, so that light can only add (``[value, value + reach]``); its field may lower a coherent sum as well, by less than
    it may raise it (``_truncation``: ``reach`` is the rise ``(sqrt(P) + r)^2 - P``, which exceeds the fall
    ``P - (sqrt(P) - r)^2``), so ``[max(0, value - reach), value + reach]``."""
    low = value if t.metric == "power_W" else max(0.0, value - reach)
    return low, value + reach


def _undecided_on(requirement: Requirement, low: float, high: float) -> bool:
    """Whether a target is met at some values of ``[low, high]`` and missed at others (rule T1: an interval decides only
    when every value in it is on one side)."""
    met = requirement.meets(low) and requirement.meets(high)
    return not met and not requirement.above(low) and not requirement.below(high)


def _left_out_note(message: str, value: float, reach: float, ends: tuple[float, float], fmt: Callable[[float, int], str]) -> str:
    """The message of a target the light left out as negligible leaves undecided: the counted value, that light's bound and
    the interval they give."""
    span = (f"between {fmt(ends[0], INTERVAL_MIN_DIGITS)} and {fmt(ends[1], INTERVAL_MIN_DIGITS)}" if math.isfinite(reach)
            else f"at least {fmt(ends[0], INTERVAL_MIN_DIGITS)}")
    added = f"at most {fmt(reach, INTERVAL_MIN_DIGITS)}" if math.isfinite(reach) else "power without a bound"
    note = (f"the counted light gives {fmt(value, INTERVAL_MIN_DIGITS)}; light left out as negligible "
            f"(negligible_power_fraction) may move it by {added}, so the value lies {span}: the target is met at some "
            "values of that interval and not at others, so it is not decided")
    return f"{message}; {note}" if message else note


def _returned(fate: LineFate, amount: float, *, coherent: bool = False, upper: float | None = None) -> ValueBasis:
    """The entry of a value the light left out as negligible may add to above the negligible level of its lines at this plane
    (its returns or crossings, ``LineFate.faint_accounted``): the traced value is the lower end of a power (an estimate of a
    coherent sum, which that field may lower), and the omission is the light's possible return. Its ``upper``: for a power
    ``amount`` (W), for a coherent sum ``upper`` (the most the field ``amount``, sqrt(W), may move the value), absent where
    either is not finite (nothing bounds that light)."""
    bound = upper if coherent else amount
    return ValueBasis(kind=ValueKind.ESTIMATE if coherent else ValueKind.LOWER_BOUND, omitted=(Omitted(
        part=OmittedPart.POSSIBLE_RETURN,
        upper=bound if bound is not None and math.isfinite(bound) and math.isfinite(amount) else None,
        setting="negligible_power_fraction", code=_faint_code(fate, amount)),))


def _unbounded_note(message: str, fate: LineFate, amount: float, what: str) -> str:
    """``message`` with the cause where nothing bounds the ``what`` (power, field) the light left out as negligible may add
    (``amount`` not finite: ``LineFate.unbounded_returns``); unchanged where it is bounded (the entry holds the amount)."""
    if math.isfinite(amount):
        return message
    note = f"nothing bounds the {what} of the light the trace leaves out as negligible ({fate.unbounded_returns()})"
    return f"{message}; {note}" if message else note


def _faint_code(fate: LineFate, amount: float) -> str:
    """The code of the light left out as negligible that may add ``amount`` (W, or sqrt(W) for a field) to a result:
    ``FAINT_LIGHT_BOUNDED`` (that light, bounded by its diffraction limit and its returns between surfaces facing each other,
    or by nothing where those returns are not bounded -- the exposures' convention), except where the amount is not finite
    because the return-light analysis reached its work limit first (``LineFate.withheld_by_work``): then the result carries
    that cause, ``RETURN_ANALYSIS_BUDGET``, as its message does."""
    return fate.faint_code(amount)


def _power_labels(t: Target, result: TargetResult, obs: ObservationResult | None, cfg: NumericalConfig) -> TargetResult:
    """What a power target's reading is, on its own terms; the assessment does not change. ``SURFACE_COINCIDENT_ARRIVAL``: a
    selected arrival is the power at a physical surface, not that of the beam that leaves it.
    ``POWER_INCOHERENT_SUM``: a ``power_W`` value adds two or more beams of one wavelength on a plane where their coherent
    sum exists (``ObservationResult.power_basis``), and its message says so."""
    if t.metric not in POWER_SUM_METRICS or obs is None:
        return result
    selected = _select_arrivals(obs, t, cfg)[0]
    codes = list(result.issue_codes)
    if any(a.surface_coincidence is not None for a in selected):
        codes.append(ErrorCode.SURFACE_COINCIDENT_ARRIVAL.value)
    message = result.message
    if t.metric == "power_W" and obs.power_basis is not None and result.value is not None:
        wavelengths = sorted(a.beam.wavelength_nm for a in selected)
        if any(b - a <= cfg.interference_wavelength_tol_nm for a, b in zip(wavelengths, wavelengths[1:])):
            codes.append(ErrorCode.POWER_INCOHERENT_SUM.value)
            note = (f"the value is the sum of the powers of the {len(selected)} beams it selects, with no interference term; "
                    "coherent_power_W states the power of their fields added")
            message = f"{message}; {note}" if message else note
    if tuple(codes) == tuple(result.issue_codes):
        return result
    return result.model_copy(update={"issue_codes": tuple(dict.fromkeys(codes)), "message": message})


def _no_arrival(t: Target, unit: str, obs: ObservationResult | None, power_budget: PowerBudget | None, cfg: NumericalConfig,
                *, pair: bool, what: str, known_zero: bool, reach: PlaneReach | None = None) -> MetricValue:
    """A target no traced beam (pair) matches: empty selection, incomplete trace, or a known zero (module docstring)."""
    ledger = _fate(t, power_budget, cfg)
    problem = _selection_problem(t, obs, power_budget, cfg, ledger, pair=pair)
    if problem is not None:
        return _empty_selection(unit, problem)
    # a power is dark only if none of the lines' untraced light can reach the plane either, and a beam metric, or a pair metric
    # on a plane nothing lights, has nothing to fail on while that light may be there
    fate = _dark_fate(t, power_budget, cfg, reach, obs) if known_zero or not pair or (obs is not None and not obs.lit) else ledger
    if not known_zero and fate.complete:
        if not math.isfinite(fate.faint_W):  # (nothing bounds the light left out as negligible on this plane: not known to be dark)
            return MetricValue(None, unit, Status.UNKNOWN, (_faint_code(fate, fate.faint_W),),
                               f"no {what} reaches the observation plane; light the trace leaves out as negligible may add power without "
                               f"bound ({fate.unbounded_returns()})")
        fate = ledger
    if known_zero and fate.complete and t.metric == "coherent_power_W" and fate.faint_root > 0.0 \
            and not (fate.faint_accounted and _square(fate.faint_root) <= fate.level_W):
        # (dark for the traced light, but light left out as negligible may bring a field above the negligible level here:
        # its power, or the bound of its field at the sum, r^2 with P = 0 (W1))
        return MetricValue(None, unit, Status.PASS, (_faint_code(fate, fate.faint_root),),
                           f"no {what} reaches the observation plane; light the trace leaves out as negligible may, with a field "
                           + (f"of at most sqrt({fate.faint_root ** 2:.4g} W)" if math.isfinite(fate.faint_root) else
                              f"without bound ({fate.unbounded_returns()})"),
                           untraced=(0.0, _square(fate.faint_root)))
    if known_zero and fate.complete and not fate.faint_accounted and t.metric == "power_W":
        # (light left out as negligible may bring more than the negligible level here: no known zero for the power)
        reach_text = (f"with at most {fate.faint_W:.4g} W" if math.isfinite(fate.faint_W) else
                      f"without bound ({fate.unbounded_returns()})")
        return MetricValue(None, unit, Status.PASS, (ErrorCode.NO_TRACED_ARRIVAL.value, _faint_code(fate, fate.faint_W)),
                           f"no {what} reaches the observation plane; light the trace leaves out as negligible may, {reach_text}",
                           untraced=(0.0, fate.faint_W))
    if not fate.complete and known_zero and not fate.unbounded:
        # no traced beam, and every watt that may reach the plane is light the probes followed to it: the plane's power has
        # no value and lies between 0 and the bound of that light (coherent: the square of the summed amplitudes). A finite
        # bound is proven, and the target is graded on that interval: a maximum at or above it is met, a minimum above it is
        # not, anything else is not decided. The plane's own power is BOUNDED from 0 with no value (validation/outcome.py)
        upper = _square(fate.root_sum + fate.faint_root) if t.metric == "coherent_power_W" else fate.untraced_W + fate.faint_W
        if _absorbing(obs) and t.metric == "power_W":
            upper = capped_by_launch(upper, fate.input_W)
        reach_text = (f"no {what} reaches the observation plane as a whole: part of a traced beam reaches it where the beam's chief "
                      "ray does not cross it" if fate.partial_only else
                      f"no {what} reaches the observation plane; light the trace does not follow may ({fate.describe()})")
        if not math.isfinite(upper):
            return MetricValue(None, unit, Status.UNKNOWN, fate.cause_codes(),
                               f"{reach_text}: it is not traced and nothing bounds it, so the target is not decided")
        return MetricValue(None, unit, Status.PASS, fate.cause_codes(),
                           f"{reach_text}: it is not traced, and it brings at most {upper:.4g} W", untraced=(0.0, upper))
    if not fate.complete:
        return _not_traced(unit, fate, what)
    if known_zero:
        # (light left out as negligible is the stated truncation of the 0 W: the entry holds the most it may bring)
        amount = fate.faint_root if t.metric == "coherent_power_W" else fate.faint_W
        added = _square(fate.faint_root) if t.metric == "coherent_power_W" else fate.faint_W
        return MetricValue(0.0, unit, Status.PASS,
                           (ErrorCode.NO_TRACED_ARRIVAL.value,) + ((_faint_code(fate, amount),) if amount > 0.0 else ()),
                           _unbounded_note(
                               f"no {what} reaches the observation plane, all of the target's source lines were traced to known "
                               "ends, and none of their untraced reflections, coating transmissions or clipped light reaches it "
                               "(followed through the optics to every order above negligible_power_fraction of the line power): the "
                               "plane receives 0 W of them"
                               + ("; a plane without an active area counts a beam that leaves the scene along the scene's length, "
                                  f"at least {cfg.laboratory_reach_mm:g} mm" if obs is not None and obs.plane is not None and obs.plane.aperture is None else ""),
                               fate, amount, "field" if t.metric == "coherent_power_W" else "power"),
                           basis=_truncation(fate, amount, added) if amount > 0.0 else None)
    return MetricValue(None, unit, Status.FAIL, (ErrorCode.OBSERVATION_NOT_REACHED.value,),
                       f"no {what} reaches the observation plane")


def _measure(
    t: Target,
    observations: dict[str, ObservationResult],
    interference: list[InterferenceResult],
    layout: LayoutResult | None,
    control: ControlResult | None,
    cfg: NumericalConfig,
    power_budget: PowerBudget | None = None,
    reach: PlaneReach | None = None,
    exposures: ExposureReport | None = None,
    truncated_pairs: frozenset[str] = frozenset(),
    actuators: dict[str, tuple[Any, ...]] | None = None,
    detections: dict[str, Any] | None = None,
    conversions: tuple[Any, ...] = (),
    plane_bounds: bool = True,
) -> MetricValue:
    m = t.metric
    scope_key = out_of_scope_metric(m)
    if scope_key is not None:
        return MetricValue(None, "", Status.OUT_OF_MODEL_SCOPE, (ErrorCode.OUT_OF_MODEL_SCOPE.value,), out_of_scope_explanation(scope_key))
    if m not in ALL_SUPPORTED:
        return MetricValue(None, "", Status.UNKNOWN, (ErrorCode.UNSUPPORTED_METRIC.value,), f"unknown metric {m!r}")
    if m in DETECTION_METRICS:
        return _detection_metric(t, observations, detections or {}, _detection_left_out(t, observations, power_budget, cfg, reach))
    if m in MECHANICAL_METRICS:
        declared = mechanical_value(t, actuators)
        return MetricValue(declared.value, declared.unit, declared.status, declared.codes, declared.message)
    if m in NONLINEAR_METRICS:
        got = nonlinear_value(t, conversions, power_budget, cfg)
        return MetricValue(got.value, got.unit, got.status, got.codes, got.message)
    if m in ELEMENT_METRICS:
        # an exposure that is not exact gives an interval of utilizations (validation/exposures.py), graded like a bounded power
        got = measure_utilization(t, exposures)
        return MetricValue(got.value, ELEMENT_METRICS[m], got.status, got.codes, got.message, untraced=got.bounds,
                           subject="the utilization is", basis=got.basis)
    if m in ALIGNMENT_METRICS:
        return _alignment(t, observations, cfg, power_budget, reach, plane_bounds)
    if m in OBSERVATION_METRICS:
        unit, fn, needs_g = OBSERVATION_METRICS[m]
        assert t.observation_id is not None  # (the Target contract: a metric that reads a plane names it)
        obs = observations.get(t.observation_id)
        arrivals, err = _select_arrivals(obs, t, cfg)
        if err:
            return MetricValue(None, unit, Status.UNKNOWN, (ErrorCode.UNKNOWN_REFERENCE.value,), err)
        selected_path = bool(arrivals) and t.parameters.get("path_id") is not None
        selected_crossing = selected_path and t.parameters.get("crossing_index") is not None
        if m in POWER_SUM_METRICS and t.wavelength_nm is not None and not selected_path:
            harmonic = next((c for c in conversions if c.status is Status.UNKNOWN
                             and (t.source_id is None or t.source_id == c.source_id)
                             and abs(t.wavelength_nm - c.harmonic_wavelength_nm) <= cfg.interference_wavelength_tol_nm), None)
            if harmonic is not None:
                return MetricValue(None, unit, Status.UNKNOWN, (ErrorCode.BULK_LOSS_NOT_ESTABLISHED.value,),
                                   f"the {harmonic.harmonic_wavelength_nm:g} nm harmonic from {harmonic.element_id} "
                                   f"has no established power: {harmonic.reason}")
        sampled = _select_sampled(obs, t, cfg)
        zero = _select_sampled_zero(obs, t, cfg) if m == "power_W" else []
        if (sampled or zero) and m == "power_W":  # (sampled rays add to the beams' power; each line's own power is its sum of rays)
            received = _received_power(arrivals, unit)
            added = math.fsum(line.power_W for line in sampled)
            with_rays = replace(received, value=(received.value or 0.0) + added,
                                lower=None if received.lower is None else received.lower + added,
                                untraced=None if received.untraced is None else (received.untraced[0] + added, received.untraced[1] + added))
            return _untraced_bounds(_sampled_resolution(with_rays, zero), _fate(t, power_budget, cfg, reach, obs), obs, coherent=False)
        # (only a power adds over independent rays: no beam stands behind this metric -- also where none of a sampled
        # line's rays is counted on the plane, a sampled zero, which does not show that none of its light arrives)
        if (sampled or _select_sampled_zero(obs, t, cfg)) and not arrivals:
            return MetricValue(None, unit, Status.OUT_OF_MODEL_SCOPE, (ErrorCode.OUT_OF_MODEL_SCOPE.value,),
                               f"{m} describes one beam; the light here is independent rays of an extended emitter, with a power "
                               "and an irradiance distribution (ObservationResult.sampled_light)")
        if not arrivals:
            if not plane_bounds:
                return _needs_plane_bounds(unit, "beam matching the target")
            return _no_arrival(t, unit, obs, power_budget, cfg, pair=False, what="beam matching the target",
                               known_zero=m in POWER_SUM_METRICS, reach=reach)
        fate = _fate(t, power_budget, cfg, reach, obs) if m in POWER_SUM_METRICS else _fate(t, power_budget, cfg)
        if m in POWER_SUM_METRICS:
            known = [a for a in arrivals if a.beam.power_W is not None]
            if not known:
                return _unknown_power(t, arrivals, unit, obs, fate)
        if m == "power_W":
            if selected_crossing and len(known) == len(arrivals) and SINK_BULK_LOSS_UNKNOWN in fate.unknown_W:
                # The matched crossing's traced power already accounts for its traversal. Unknown bulk loss elsewhere
                # in the source line may affect other crossings, but cannot add power to this selected arrival.
                fate = replace(fate, unknown_W={kind: power for kind, power in fate.unknown_W.items()
                                               if kind != SINK_BULK_LOSS_UNKNOWN})
            return _untraced_bounds(_received_power(known, unit), fate, obs, coherent=False)
        if m == "coherent_power_W":
            if len(arrivals) > 1 and (len(known) != len(arrivals) or any(a.beam.polarization is None for a in arrivals)):
                # (a beam whose power or polarization is not known has no field to add)
                return MetricValue(None, unit, Status.UNKNOWN, fate.cause_codes(),
                                   "a beam of the coherent sum left a diffraction grating with a power or polarization that is not known")
            return _untraced_bounds(_coherent_power(observations[t.observation_id], arrivals, unit, cfg), fate, obs, coherent=True)
        if m in POLARIZATION_METRICS and obs is not None and any(a.polarization_component is not None for a in arrivals):
            return mixture_metric(m, unit, arrivals, obs)  # (the light on the route: both polarization components)
        if component_twins(arrivals) and fn(arrivals[0]) == fn(arrivals[1]):
            arrivals = arrivals[:1]  # (the two polarization components of one route agree on this quantity)
        if len(arrivals) > 1:
            return MetricValue(None, unit, Status.UNKNOWN, (ErrorCode.INVALID_INPUT.value,),
                               f"{len(arrivals)} beams match; select one with source_id / wavelength_nm / parameters['path_id']")
        a = arrivals[0]
        if m in OPTICAL_PATH_METRICS and not a.ray_opl_defined:
            return MetricValue(None, unit, Status.UNKNOWN, (ErrorCode.PHASE_UNDEFINED.value,),
                               "the beam crossed an ideal or tabulated thin lens, a Faraday rotator or a diffraction grating, whose model "
                               f"states no optical path, so its {OPTICAL_PATH_METRICS[m]} has no value")
        if m in GROUP_DELAY_METRICS:
            return group_delay(a)
        if m in PHASE_DELAY_METRICS:
            return phase_delay(a)
        if m in CHIEF_FIELD_METRICS and a.chief_ray_obstructed:
            return MetricValue(None, unit, Status.OUT_OF_MODEL_SCOPE, (ErrorCode.GAUSSIAN_CENTRAL_OBSTRUCTION.value,),
                               "the beam passed a central obstruction its chief ray fell on: no light is on the chief ray, so the "
                               "field there (its phase) is not defined in the geometric model")
        val = fn(a)
        st = Status.PASS
        codes: tuple[str, ...] = ()
        g = a.beam.gaussian
        if needs_g and (g is None or not g.valid):
            reason = g.reason if g is not None else None
            # (no number of a state outside the model's scope is reported: the failed accuracy check it records is)
            return MetricValue(None, unit, Status.OUT_OF_MODEL_SCOPE, (reason or ErrorCode.OUT_OF_MODEL_SCOPE.value,),
                               "Gaussian beam model is out of scope for this beam", validity=g.validity if g is not None else ())
        if needs_g and g is not None and g.status is Status.WARNING:
            # (a caveat of the Gaussian state marks the metrics computed from that state, and only them)
            st = Status.WARNING
            codes = (g.reason,) if g.reason else ()
        if val is None:
            return MetricValue(None, unit, Status.UNKNOWN, (ErrorCode.NUMERICAL_FAILURE.value,), "metric is undefined for this beam")
        return MetricValue(val, unit, st, codes)
    if m in PAIR_METRICS:
        unit, field = PAIR_METRICS[m]
        assert t.observation_id is not None  # (the Target contract)
        # Checked by the Target contract (``target_parameters.py``): every source / path / crossing selector given is applied
        # (``pair_selection.target_selects_pair``; the run lists every pair it selects, of mutually incoherent sources too).
        pairs = [r for r in interference if r.observation_id == t.observation_id and target_selects_pair(
            t, r.source_id_a, r.source_id_b, r.wavelength_nm, r.path_id_a, r.crossing_index_a, r.path_id_b, r.crossing_index_b, cfg)]
        pa, pb = t.parameters.get("path_id_a"), t.parameters.get("path_id_b")
        if t.observation_id in truncated_pairs and len(pairs) <= 1 and (pa is None or pb is None or not pairs):
            # only the first max_interference_pairs pairs of this plane were analysed: the pair the target asks for, or a
            # second pair that would make its selection ambiguous, may be among the others
            return MetricValue(None, unit, Status.UNKNOWN, (ErrorCode.MAX_BEAM_PAIRS.value,),
                               f"only the first {cfg.max_interference_pairs} beam pairs on this plane were analysed, and the "
                               "pair this target selects is not known to be among them: name both beams with parameters "
                               "path_id_a / path_id_b, or reduce the beams reaching the plane")
        if len(pairs) <= 1 and _open_field_pair(t, observations.get(t.observation_id), cfg):
            # (a pair of unknown field forms no InterferenceResult: it is there, and may be the pair this target selects)
            return MetricValue(None, unit, Status.UNKNOWN, (ErrorCode.PHASE_UNDEFINED.value,),
                               "two beams reach this plane and the field of one is not known: it left a diffraction grating with a power or "
                               "polarization that is not known; state its efficiency per polarization, or name another pair with parameters path_id_a / path_id_b")
        if not pairs and _components_named(t, observations.get(t.observation_id)):
            return MetricValue(None, unit, Status.NOT_APPLICABLE, (ErrorCode.PARTIAL_POLARIZATION_COMPONENTS.value,),
                               "the two beams are the two polarization components of a partially polarized source's line: they "
                               "are mutually incoherent, with no interference term and no two-beam quantity between them; the "
                               "light on their route is ObservationResult.polarization_mixtures")
        if not pairs:
            mismatch = _wavelength_mismatch(observations.get(t.observation_id), t, cfg)
            if mismatch is not None:
                return MetricValue(None, unit, Status.NOT_APPLICABLE, (ErrorCode.WAVELENGTH_MISMATCH.value,), mismatch)
            if not plane_bounds:
                return _needs_plane_bounds(unit, "pair of beams for this target")
            return _no_arrival(t, unit, observations.get(t.observation_id), power_budget, cfg, pair=True,
                               what="pair of beams for this target", known_zero=False, reach=reach)
        if len(pairs) > 1:
            return MetricValue(None, unit, Status.UNKNOWN, (ErrorCode.INVALID_INPUT.value,),
                               f"{len(pairs)} beam pairs match; select with source_id/source_id_b/parameters path_id_a,path_id_b")
        r = pairs[0]
        if m in PAIR_DELAY_METRICS:
            return time_delay(r, _target_reverses_pair(t, r), observations.get(t.observation_id))
        if m in CHIEF_FIELD_PAIR_METRICS and _obstructed_pair(observations.get(t.observation_id), r):
            return MetricValue(None, unit, Status.OUT_OF_MODEL_SCOPE, (ErrorCode.GAUSSIAN_CENTRAL_OBSTRUCTION.value,),
                               "a beam of the pair passed a central obstruction its chief ray fell on: no light is on its chief "
                               "ray, so the phase difference of the fields there is not defined in the geometric model")
        if m in FRINGE_GEOMETRY_METRICS and ErrorCode.BEAMS_NOT_OVERLAPPING.value in r.issue_codes:
            period = _get(r, field)
            no_fringes = "the two beams do not overlap on the observation plane: no fringes form there"
            if period is None:
                return MetricValue(None, unit, Status.NOT_APPLICABLE, r.issue_codes, no_fringes)
            overlap = ValidityCheck(parameter="spatial_overlap_ratio", unit=DIMENSIONLESS_UNIT, minimum=cfg.interference_min_overlap,
                                    observed=r.spatial_overlap_ratio, within=False, setting="interference_min_overlap",
                                    code=ErrorCode.BEAMS_NOT_OVERLAPPING.value)
            return MetricValue(period, unit, Status.NOT_APPLICABLE, r.issue_codes,
                               f"{no_fringes}; the value is the period of the two beams' wavevectors and is not graded",
                               basis=ValueBasis(kind=ValueKind.ESTIMATE, validity=(overlap,)), graded=False)
        val = _get(r, field)
        incoherent = val is None and m in INCOHERENT_ZERO_METRICS and ErrorCode.INCOHERENT_SOURCES.value in r.issue_codes
        if incoherent:
            # mutually incoherent beams: the cross term of their fields averages out over the detection time, their
            # irradiances add, and the time-averaged visibility of their fringe is 0 (the visibility with a coherence
            # factor of 0)
            val = 0.0
        if val is not None and m == "optical_path_difference_abs_mm":
            val = abs(val)
        elif val is not None and m in ANTISYMMETRIC_PAIR_METRICS and _target_reverses_pair(t, r):
            # (a reported phase lies in (-pi, pi]: its negation lies in [-pi, pi), which reported_phase reads in (-pi, pi])
            val = -val if m == "optical_path_difference_mm" else reported_phase(-val)
        if m == "estimated_exposure_time_s":
            dose = float(t.parameters["dose_mj_per_cm2"])  # (the Target contract: required, and greater than 0)
            val = None if val is None or val <= 0.0 else dose / (val * W_PER_MM2_TO_MW_PER_CM2)
        if val is None:
            st = r.status if r.status in (Status.OUT_OF_MODEL_SCOPE, Status.NOT_APPLICABLE) else Status.UNKNOWN
            # (reached with mutually incoherent beams too: the difference is formed before that verdict, and has no other reason to be absent)
            if m in OPTICAL_PATH_PAIR_METRICS and {ErrorCode.PHASE_UNDEFINED.value, ErrorCode.INCOHERENT_SOURCES.value} & set(r.issue_codes):
                return MetricValue(None, unit, Status.UNKNOWN, tuple(sorted({*r.issue_codes, ErrorCode.PHASE_UNDEFINED.value})),
                                   "the optical path difference of the two beams has no value: one beam crosses an ideal or tabulated thin "
                                   "lens, a Faraday rotator or a diffraction grating, whose model states no optical path, and the other does not; "
                                   "model the element with its surfaces and material")
            if m == "phase_difference_rad" and ErrorCode.PHASE_UNDEFINED.value in r.issue_codes:
                return MetricValue(None, unit, Status.UNKNOWN, r.issue_codes,
                                   "the phase difference of the two beams is not known: one beam crosses an ideal or tabulated "
                                   "thin lens, a Faraday rotator or a diffraction grating, which states neither a phase nor an "
                                   "optical path, and the other does not; model the element with its surfaces and material")
            if (m in COHERENCE_METRICS and ErrorCode.COHERENCE_UNDEFINED.value in r.issue_codes and r.coherence_length_mm is not None
                    and r.group_path_difference_mm is not None):
                return MetricValue(None, unit, st, r.issue_codes,
                                   f"{m} depends on the beams' degree of coherence, which their pulse train leaves open at this "
                                   "delay: its pulses may be mutually coherent, and near a whole multiple of its period length "
                                   "c / f_rep one beam's pulse overlaps a later pulse of the other")
            if m in COHERENCE_METRICS and ErrorCode.COHERENCE_UNDEFINED.value in r.issue_codes and r.coherence_length_mm is not None:
                return MetricValue(None, unit, st, r.issue_codes,
                                   f"{m} depends on the beams' degree of coherence, which is not computed: a medium on a beam's "
                                   "path states no group index at the wavelength; state the medium's dispersion there")
            if m in COHERENCE_METRICS and (ErrorCode.COHERENCE_UNDEFINED.value in r.issue_codes or _full_coherence_assumed(r)):
                # (also a pair whose source states no coherence and is taken as monochromatic: it has no coherence length,
                # so no margin against one)
                return MetricValue(None, unit, st, tuple(sorted({*r.issue_codes, ErrorCode.COHERENCE_UNDEFINED.value})),
                                   f"{m} depends on the source coherence, which is not declared: declare the source's "
                                   "coherence_length_mm or linewidth_nm")
            return MetricValue(None, unit, st, r.issue_codes, "metric is not available for this beam pair (see issue codes)")
        status = Status.WARNING if r.status is Status.WARNING else Status.PASS
        note = INCOHERENT_NOTE if incoherent else ""
        ends = _coherence_ends(r) if m in COHERENCE_METRICS else None
        # (a number on the trace's convention -- no added delay, default interface phases -- carries the pair's entry)
        entry = None if incoherent else (r.value_basis or {}).get(field)
        if entry is not None and not _stand_in_only(r, field):
            # (an estimate: reported with its entry, graded as a point -- it decides nothing, rule T1; the monochromatic
            # stand-in alone is judged on the interval every degree of coherence gives, _assumed_coherence)
            return MetricValue(val, unit, status, r.issue_codes, note, basis=entry)
        if m in PAIR_PLANE_METRICS and reach is not None and power_budget is not None:
            if ends is None:
                # (a line that states no coherence beside one that does, behind bounded coating delays: the light's interval
                # of the pair at the upper end of its degree of coherence, the value the traced one; _stand_in_top)
                top = _stand_in_top(r) if _stand_in_only(r, field) else r
                bounded = _pair_light(t, top, val if top is r else _get(top, field), unit, status, power_budget, cfg, reach)  # type: ignore[arg-type]
                if bounded is not None and top is not r:
                    bounded = replace(bounded, value=val)
                if bounded is not None:
                    return _assumed_coherence(replace(bounded, message=f"{note}; {bounded.message}") if note else bounded, r, field)
            else:
                lights = [_pair_light(t, end, _get(end, field), unit, status, power_budget, cfg, reach) for end in ends]  # type: ignore[arg-type]
                if lights[0] is not None and lights[1] is not None:
                    assert lights[0].untraced is not None and lights[1].untraced is not None
                    return MetricValue(val, unit, status, tuple(dict.fromkeys((*lights[0].codes, *lights[1].codes))),
                                       f"{_coherence_note(r)}; {lights[0].message}",
                                       untraced=(min(lights[0].untraced[0], lights[1].untraced[0]),
                                                 max(lights[0].untraced[1], lights[1].untraced[1])), subject=lights[0].subject,
                                       basis=merge_bases(_traced_delay_entry(r, field, incoherent), lights[0].basis,
                                                         lights[1].basis, monotone=None))
        if ends is not None:
            low, high = sorted((_get(ends[0], field), _get(ends[1], field)))  # type: ignore[type-var]
            if low is not None and high is not None and m != "coherence_length_mm":
                return MetricValue(val, unit, status, r.issue_codes, _coherence_note(r), untraced=(low, high),
                                   subject=COHERENCE_SUBJECTS.get(m, "the value is"), basis=_traced_delay_entry(r, field, incoherent))
        return _assumed_coherence(MetricValue(val, unit, status, r.issue_codes, note), r, field)
    if m in LAYOUT_METRICS:
        unit = LAYOUT_METRICS[m]
        if layout is None:
            return MetricValue(None, unit, Status.UNKNOWN, (ErrorCode.INVALID_INPUT.value,), "layout stage did not run")
        if m == "layout_violation_count":
            return MetricValue(float(sum(1 for i in layout.issues if i.status is Status.FAIL)), unit)
        if layout.min_clearance_mm is None:
            return MetricValue(None, unit, Status.UNKNOWN, (ErrorCode.INVALID_INPUT.value,), "no clearance was evaluated")
        return MetricValue(layout.min_clearance_mm, unit)
    unit = CONTROL_METRICS[m]
    if control is None:
        return MetricValue(None, unit, Status.UNKNOWN, (ErrorCode.INVALID_INPUT.value,), "control stage did not run")
    val = {
        "recoverability_fraction": control.recoverability_fraction,
        "monte_carlo_recovered_fraction": control.monte_carlo_recovered_fraction,
        "jacobian_condition_number": None if control.sensitivity is None else control.sensitivity.condition_number,
        "numerical_rank": None if control.sensitivity is None else float(control.sensitivity.numerical_rank),
    }[m]
    if val is None:
        return MetricValue(None, unit, Status.UNKNOWN, (ErrorCode.INVALID_INPUT.value,), "control metric is not available")
    return MetricValue(float(val), unit)


#: What a coherence metric's interval is of, in a target's message (``_coherence_ends``).
COHERENCE_SUBJECTS = {
    "coherence_factor": "the coherence factor is", "coherence_margin": "the coherence margin is",
    "fringe_visibility": "the visibility over the plane is", "fringe_visibility_peak": "the local visibility is",
    "peak_irradiance_W_per_mm2": "the peak irradiance is",
}


def _full_coherence_assumed(r: InterferenceResult) -> bool:
    """Whether the pair's degree of coherence stands on the monochromatic stand-in of a line that states no coherence
    (``FULL_COHERENCE`` in the entry of its ``coherence_factor``): such a pair has no coherence length."""
    entry = (r.value_basis or {}).get("coherence_factor")
    return entry is not None and any(a.input is AssumedInput.FULL_COHERENCE for a in entry.assumed)


def _stand_in_only(r: InterferenceResult, field: str) -> bool:
    """Whether the entry of the pair's ``field`` states only the monochromatic stand-in of a line that states no coherence
    (``FULL_COHERENCE``, ``_assumed_coherence``); an entry that also stands on another of the trace's conventions (no added
    delay, the default interface phases, a share taken before the plane) is an estimate graded as a point."""
    entry = (r.value_basis or {}).get(field)
    # (where its values are no upper end -- behind bounded coating delays, or a dispersion difference -- such a pair states
    # only the upper end its degree of coherence may reach, ``coherence_factor_max`` without a ``coherence_factor_min``: its
    # values are at the traced delay, no added delay at those passes, and the interval up to that end covers every delay
    # the bounds allow and every length of the unstated line)
    ranged = r.coherence_factor_max is not None and r.coherence_factor_min is None
    return (entry is not None and bool(entry.assumed) and not entry.omitted and not entry.validity
            and any(a.input is AssumedInput.FULL_COHERENCE for a in entry.assumed)
            and all(a.input is AssumedInput.FULL_COHERENCE or (ranged and a.input is AssumedInput.UNSTATED_GROUP_DELAY_ZERO)
                    for a in entry.assumed))


def _assumed_coherence(mv: MetricValue, r: InterferenceResult, field: str) -> MetricValue:
    """``mv`` of a pair whose line states no coherence and is taken as monochromatic (``InterferenceResult.value_basis``:
    ``FULL_COHERENCE``): the value at a degree of coherence of 1, with its entry, judged on the interval every degree of
    coherence from 0 to 1 gives: from 0 (the peak irradiance: from the mean irradiance) to the value; the combined power,
    which the cross term may lower or raise, from 0 to the larger of the value and the two beams' incoherent sum (its
    ``combined_power_collinear_max_W`` where the pair states that range over every phase). Where light
    the trace does not follow widens it (``mv.untraced``), from 0 to its upper end. ``mv`` unchanged for a pair of stated
    coherence."""
    entry = (r.value_basis or {}).get(field)
    if entry is None or mv.value is None:
        return mv
    if mv.untraced is not None:
        # (``mv.untraced``: the light's interval of the pair at its upper end, ``_stand_in_top``)
        return replace(mv, basis=entry, untraced=(0.0, mv.untraced[1]))
    top = _stand_in_top(r)
    upper = _get(top, field) if top is not r else None
    most = mv.value if upper is None else max(mv.value, upper)
    if field == "combined_power_collinear_W":
        if r.incoherent_power_sum_W is None:
            return replace(mv, basis=entry)
        # (where the phase of the degree of coherence moves with the unstated length or the bounded delays, the pair states
        # the power's range over every phase up to its upper end, ``optics/interference/two_beam.py``)
        reach = r.combined_power_collinear_max_W
        ends = (0.0, max(mv.value, r.incoherent_power_sum_W if reach is None else reach))
    elif field == "peak_irradiance_W_per_mm2":
        if r.mean_irradiance_W_per_mm2 is None:
            return replace(mv, basis=entry)
        ends = (min(r.mean_irradiance_W_per_mm2, mv.value), most)
    else:
        ends = (0.0, most)
    if top is not r and r.group_delay_bound_mm is not None:
        assert r.coherence_factor_max is not None
        note = (f"a source states no coherence and is taken as monochromatic, and the two beams meet coatings differently whose "
                f"group delays are known within {r.group_delay_bound_mm / MM_PER_FS:.6g} fs in all: the value is at the traced "
                f"delay and the other source's stated coherence length, the upper end at the degree of coherence "
                f"{r.coherence_factor_max:.6g} those bounds allow at any shorter coherence length of the first, and every "
                "degree of coherence from 0 lies in the interval")
    elif top is not r:
        assert r.coherence_factor_max is not None
        note = ("a source states no coherence and is taken as monochromatic, and the two paths' group-delay dispersions "
                "differ: the value is at the other source's stated coherence length, the upper end at the degree of coherence "
                f"{r.coherence_factor_max:.6g} a shorter coherence length of the first gives, and every degree of coherence "
                "from 0 lies in the interval")
    else:
        note = ("a source states no coherence and is taken as monochromatic: the value is the upper end (at full coherence, or at "
                "the other source's stated coherence length), and every degree of coherence lies in the interval")
    return replace(mv, basis=entry, message=f"{mv.message}; {note}" if mv.message else note, untraced=ends,
                   subject=COHERENCE_SUBJECTS.get(field, "the value is"))


def _traced_delay_entry(r: InterferenceResult, field: str, incoherent: bool) -> ValueBasis | None:
    """The entry of a pair's coherence quantity read at the traced delay while bounded coating delays leave it an interval:
    the stored one, else the derived ESTIMATE of that convention (``results/basis.py`` ``basis_of``; ``None`` where none
    applies)."""
    if incoherent:
        return None
    from ..results.basis import basis_of  # (imported here: the results package imports modules that import this one)

    return basis_of(r, field)


def _coherence_ends(r: InterferenceResult) -> tuple[InterferenceResult, InterferenceResult] | None:
    """The pair at the two ends of the degree of coherence its coatings' group-delay bounds allow
    (``InterferenceResult.coherence_factor_min`` / ``coherence_factor_max``; ``optics/interference/coherence.py``): every
    quantity the degree of coherence scales -- the fringe visibilities, the peak irradiance's excess over the mean -- at
    each end, and the coherence margin at the farther and the nearer end of the relative group delay's interval. ``None``
    when the pair's degree of coherence is exact."""
    g, g_min, g_max = r.coherence_factor, r.coherence_factor_min, r.coherence_factor_max
    if r.group_delay_bound_mm is None or g is None or g_min is None or g_max is None:
        return None
    return _at_coherence(r, g_min, False), _at_coherence(r, g_max, True)


#: The smallest positive normal float: a degree of coherence below it holds too few digits to scale a quantity by.
_SMALLEST_NORMAL = 2.2250738585072014e-308


def _at_coherence(r: InterferenceResult, gamma: float, high: bool) -> InterferenceResult:
    """The pair at the degree of coherence ``gamma`` (``high``: an upper end) in place of its traced one: every quantity the
    degree of coherence scales -- the fringe visibilities, the peak irradiance's excess over the mean -- and the coherence
    margin at the nearer (``high``) or farther end of the relative group delay's interval."""
    g = r.coherence_factor
    assert g is not None
    # (with a degree of coherence of 0 at the traced delay -- or one below the smallest normal float, where the scaled
    # quantities have lost their digits -- the scale of each quantity is not in the result: the bound every one obeys -- a
    # visibility at most the degree of coherence, the excess at most the mean -- stands for it)
    update: dict[str, float] = {"coherence_factor": gamma}
    for name in ("fringe_visibility", "fringe_visibility_peak"):
        value = getattr(r, name)
        if value is not None:
            update[name] = value * gamma / g if g >= _SMALLEST_NORMAL else (gamma if high else 0.0)
    if r.peak_irradiance_W_per_mm2 is not None and r.mean_irradiance_W_per_mm2 is not None:
        mean = r.mean_irradiance_W_per_mm2
        balance, overlap = r.intensity_balance, r.polarization_overlap
        if balance is not None and overlap is not None:
            # (the excess 2 sqrt(i1 i2) eta gamma from the stored balance b = min/max: 2 sqrt(i1 i2) / (i1 + i2) =
            # 2 sqrt(b) / (1 + b); no subtraction of the mean and no division by the traced degree, either of which loses
            # the excess where that degree is tiny)
            excess = mean * (2.0 * math.sqrt(balance) / (1.0 + balance)) * overlap * gamma
        else:
            excess = mean * gamma if high else 0.0
        update["peak_irradiance_W_per_mm2"] = mean + excess
    if (r.coherence_margin is not None and r.group_path_difference_mm is not None and r.coherence_length_mm
            and r.group_delay_bound_mm is not None):
        delay, bound = abs(r.group_path_difference_mm), r.group_delay_bound_mm
        update["coherence_margin"] = 1.0 - (max(0.0, delay - bound) if high else delay + bound) / r.coherence_length_mm
    return r.model_copy(update=update)


def _stand_in_top(r: InterferenceResult) -> InterferenceResult:
    """A pair whose line states no coherence beside one that does, where its values are no upper end -- behind bounded
    coating delays, or where the paths' group-delay dispersions differ (``coherence_factor_max`` without a
    ``coherence_factor_min``, ``optics/interference/two_beam.py``): the pair at that upper end, which every delay the
    bounds allow and every coherence length of the unstated line stays below; ``r`` itself otherwise (its values are the
    upper ends)."""
    if r.coherence_factor is None or r.coherence_factor_max is None or r.coherence_factor_min is not None:
        return r
    return _at_coherence(r, r.coherence_factor_max, True)


def _coherence_note(r: InterferenceResult) -> str:
    """Why a coherence metric is an interval (``_coherence_ends``)."""
    assert r.group_delay_bound_mm is not None
    return (f"the two beams meet coatings differently whose group delays are known within "
            f"{r.group_delay_bound_mm / MM_PER_FS:.6g} fs in all, so their degree of coherence lies between "
            f"{r.coherence_factor_min:.6g} and {r.coherence_factor_max:.6g}")


def _pair_light(t: Target, r: InterferenceResult, value: float, unit: str, status: Status, power_budget: PowerBudget,
                cfg: NumericalConfig, reach: PlaneReach) -> MetricValue | None:
    """A pair's light on the plane with the light of its wavelength the trace does not follow (module docstring): the
    interval ``[max(0, sqrt(v) - F)^2, (sqrt(v) + F)^2]`` of an irradiance ``v`` at the peak position (the exposure time
    from the mean's; the local contrast from the peak's and the minimum's), unbounded above where part of that light has an
    unknown fate; the power-weighted visibility from the power that light may add across the plane (``_visibility_light``);
    ``None`` when no such light reaches the plane (the value is exact)."""
    lines = [line for line in power_budget.lines if abs(line.wavelength_nm - r.wavelength_nm) <= cfg.interference_wavelength_tol_nm]
    keys = [(line.source_id, line.spectral_index) for line in lines]
    oid = r.observation_id
    if t.metric == "fringe_visibility":
        return _visibility_light(r, value, unit, status, line_fate(
            lines, cfg, hazards(reach, oid, keys), hazard_root_sum(reach, oid, keys), faint_root_sum(reach, oid, keys)))
    fate = line_fate(lines, cfg, hazards(reach, oid, keys), withheld_by_work=withheld_by_work(reach, oid, keys),
                     returns=return_causes(reach, oid, keys))
    extra = irradiance_root_sum(reach, oid, keys)
    faint = faint_root_sum(reach, oid, keys) > 0.0
    if fate.complete and extra <= 0.0:
        return None
    irradiance = r.peak_irradiance_W_per_mm2 if t.metric in ("peak_irradiance_W_per_mm2", "fringe_visibility_peak") \
        else r.mean_irradiance_W_per_mm2
    if irradiance is None and t.metric == "fringe_visibility_peak":
        # (mutually incoherent beams: no irradiance is formed at a peak position, so the light that may add leaves the contrast open)
        what = f"light of their wavelength the trace does not follow ({fate.describe() if not fate.complete else 'left out as negligible'})"
        return MetricValue(value, unit, status, tuple(dict.fromkeys(tuple(r.issue_codes) + (fate.cause_codes() if not fate.complete else ()))),
                           f"the two beams are mutually incoherent, with no fringe of their own; {what} may add to them",
                           untraced=(0.0, 1.0), subject="the contrast there is")
    assert irradiance is not None
    open_above = (not fate.complete and fate.unbounded) or not math.isfinite(extra)

    def ends(v: float) -> tuple[float, float]:
        if open_above:
            return 0.0, math.inf
        gap = math.sqrt(v) - extra
        return (gap * gap if gap > 0.0 else 0.0), _square(math.sqrt(v) + extra)

    low, high = ends(irradiance)
    subject = "the irradiance there is"
    if t.metric == "fringe_visibility_peak":
        assert r.mean_irradiance_W_per_mm2 is not None
        floor = 2.0 * r.mean_irradiance_W_per_mm2 - irradiance  # (the fringe's minimum there)
        floor_low, floor_high = ends(floor if floor > 0.0 else 0.0)
        if open_above:
            low, high = 0.0, 1.0
        else:
            least = (low - floor_high) / (low + floor_high) if low + floor_high > 0.0 else 0.0
            most = (high - floor_low) / (high + floor_low) if high + floor_low > 0.0 else 1.0
            low, high = (least if least > 0.0 else 0.0), (most if most < 1.0 else 1.0)
        subject = "the contrast there is"
    if t.metric == "estimated_exposure_time_s":
        dose = float(t.parameters["dose_mj_per_cm2"])
        low, high = (0.0 if not math.isfinite(high) else dose / (high * W_PER_MM2_TO_MW_PER_CM2),
                     math.inf if not low > 0.0 else dose / (low * W_PER_MM2_TO_MW_PER_CM2))
        subject = "the exposure time is"
    what = ("light of their wavelength the trace does not follow (followed to this plane, or left out as negligible)"
            if fate.complete or not fate.unbounded else f"light of their wavelength the trace does not follow ({fate.describe()})")
    reach_text = (f"may add a field of at most sqrt({extra * extra:.4g} W/mm^2) anywhere on the plane, with an open phase"
                  if not open_above else
                  f"may reach the plane in any amount: part of it has an unknown fate, or {fate.unbounded_returns('the plane')}")
    message = (f"the two beams give {r.mean_irradiance_W_per_mm2 if t.metric not in ('peak_irradiance_W_per_mm2', 'fringe_visibility_peak') else irradiance:.6g} "
               f"W/mm^2 at their peak position; {what} {reach_text}")
    codes = tuple(r.issue_codes) + (fate.cause_codes() if not fate.complete else ()) + (
        (ErrorCode.FAINT_LIGHT_BOUNDED.value,) if faint and not open_above else ())
    return MetricValue(value, unit, status, tuple(dict.fromkeys(codes)), message, untraced=(low, high), subject=subject)


def _visibility_light(r: InterferenceResult, value: float, unit: str, status: Status, fate: LineFate) -> MetricValue | None:
    """The power-weighted visibility over the plane with the light of the pair's wavelength the trace does not follow
    (module docstring): with ``rho = R / sqrt(P_a + P_b)``, ``R = fate.root_sum + fate.faint_root`` (sqrt(W)), between
    ``max(0, V - 2 rho) / (1 + rho)^2`` and ``min(1, (V + 2 rho) / (1 - rho)^2)``; between 0 and 1 where part of that light
    has an unknown fate or nothing bounds its returns; ``None`` when no such light reaches the plane (the value is exact)."""
    root = fate.root_sum + fate.faint_root
    if fate.complete and root <= 0.0:
        return None
    power = r.incoherent_power_sum_W
    known = not fate.unbounded and math.isfinite(root) and power is not None and power > 0.0
    if known:
        assert power is not None
        rho = root / math.sqrt(power)
        least = (value - 2.0 * rho) / _square(1.0 + rho)
        most = (value + 2.0 * rho) / _square(1.0 - rho) if rho < 1.0 else 1.0
        low, high = (least if least > 0.0 else 0.0), (most if most < 1.0 else 1.0)
        reach_text = (f"may add a field of at most sqrt({root * root:.4g} W) across the plane, with an open phase, place and "
                      f"polarization (the two beams bring {power:.6g} W)")
    else:
        low, high = 0.0, 1.0
        reach_text = ("may reach the plane in any amount: part of it has an unknown fate, or two surfaces facing each "
                      "other across the plane may hold it")
    what = ("light of their wavelength the trace does not follow (followed to this plane, or left out as negligible)"
            if not fate.unbounded else f"light of their wavelength the trace does not follow ({fate.describe()})")
    message = f"the two beams' fringe visibility is {value:.6g}; {what} {reach_text}"
    codes = tuple(r.issue_codes) + (fate.cause_codes() if not fate.complete else ()) + (
        (ErrorCode.FAINT_LIGHT_BOUNDED.value,) if fate.faint_root > 0.0 and known else ())
    return MetricValue(value, unit, status, tuple(dict.fromkeys(codes)), message, untraced=(low, high),
                       subject="the visibility over the plane is")


def _open_field_pair(t: Target, obs: ObservationResult | None, cfg: NumericalConfig) -> bool:
    """Whether two arrivals of the plane that this pair target selects form no pair because the field of one is not known (``pipeline``: such a beam is left out of the pairs)."""
    if obs is None:
        return False
    wanted = {t.source_id, t.source_id_b} - {None}
    named = {t.parameters.get("path_id_a"), t.parameters.get("path_id_b")} - {None}
    for a, b in itertools.combinations(obs.arrivals, 2):
        known = all(x.beam.power_W is not None and x.beam.polarization is not None for x in (a, b))
        if known or abs(a.beam.wavelength_nm - b.beam.wavelength_nm) > cfg.interference_wavelength_tol_nm:
            continue
        if t.wavelength_nm is not None and abs(a.beam.wavelength_nm - t.wavelength_nm) > cfg.interference_wavelength_tol_nm:
            continue
        sources, paths = {a.beam.source_id, b.beam.source_id}, {a.beam.path_id, b.beam.path_id}
        if (len(wanted) == 1 and not wanted <= sources) or (len(wanted) == 2 and sources != wanted):
            continue
        if not named <= paths:
            continue
        return True
    return False


def _alignment(t: Target, observations: dict[str, ObservationResult], cfg: NumericalConfig,
               power_budget: PowerBudget | None, reach: PlaneReach | None = None, plane_bounds: bool = True) -> MetricValue:
    """A geometric pair metric (``validation/alignment.py``): two beams of any wavelengths, never WAVELENGTH_MISMATCH."""
    unit = ALIGNMENT_METRICS[t.metric][0]
    assert t.observation_id is not None  # (the Target contract)
    obs = observations.get(t.observation_id)
    got = measure_alignment(t, obs, cfg)
    if not got.empty:
        return MetricValue(got.value, unit, got.status, got.codes, got.message, validity=got.validity)
    if power_budget is not None and t.wavelength_nm_b is not None:
        sources = {t.source_id_b if t.source_id_b is not None else t.source_id} - {None}
        lines = [line for line in power_budget.lines if not sources or line.source_id in sources]
        if not any(abs(line.wavelength_nm - t.wavelength_nm_b) <= cfg.interference_wavelength_tol_nm for line in lines):
            offered = sorted({line.wavelength_nm for line in lines})
            return _empty_selection(unit, f"no source line of the second operand's source(s) is at {t.wavelength_nm_b:g} nm "
                                          f"(launched lines: {_fmt_list(f'{w:g} nm' for w in offered)})")
    if not plane_bounds:
        return _needs_plane_bounds(unit, "pair of beams for this target")
    return _no_arrival(t, unit, obs, power_budget, cfg, pair=True, what="pair of beams for this target", known_zero=False, reach=reach)


def _obstructed_pair(obs: ObservationResult | None, r: InterferenceResult) -> bool:
    """Whether a beam of the pair ``r`` passed a central obstruction its chief ray fell on."""
    if obs is None:
        return False
    keys = {(r.path_id_a, r.crossing_index_a), (r.path_id_b, r.crossing_index_b)}
    return any(a.chief_ray_obstructed for a in obs.arrivals if (a.beam.path_id, a.crossing_index) in keys)


def _detection_left_out(t: Target, observations: dict[str, ObservationResult], power_budget: PowerBudget | None,
                        cfg: NumericalConfig, reach: PlaneReach | None) -> float | None:
    """The most the light left out as negligible may add to the power of a detection target's plane (W; ``math.inf`` where
    nothing bounds it), as a ``power_W`` target there reads it (``_untraced_interval`` / ``_no_arrival``): ``None`` where the
    plane's power has no such stated truncation."""
    obs = observations.get(t.observation_id) if t.observation_id is not None else None
    if obs is None or reach is None:
        return None
    fate = _fate(t, power_budget, cfg, reach, obs) if obs.arrivals else _dark_fate(t, power_budget, cfg, reach, obs)
    if not fate.complete or not fate.faint_W > 0.0 or not fate.faint_accounted:
        return None
    return fate.faint_W


def _detection_metric(t: Target, observations: dict[str, ObservationResult], detections: dict[str, Any],
                      left_out_W: float | None = None) -> MetricValue:
    """A detection metric of the target's plane: the quantity of its detection (``validation/detection.py``), read from all the
    light the plane counts. ``UNKNOWN`` with ``DETECTION_QUANTITY_UNAVAILABLE`` and the reason where the quantity is not
    available; a target that selects a source or a wavelength is refused (a detector reads every source and wavelength).
    A quantity computed from the traced light of a plane whose power is not simply the model's value carries its entry and,
    where one is proven, the interval it lies in. On a plane of the model's value whose light left out as negligible may add
    at most ``left_out_W`` (``_detection_left_out``), a quantity that depends on the light is graded on the interval that
    light leaves (decision W28: the truncation does not decide a limit inside it), with the stated truncation as its entry."""
    unit = DETECTION_METRICS[t.metric]
    assert t.observation_id is not None  # (the Target contract)
    if t.source_id is not None or t.wavelength_nm is not None or t.source_id_b is not None:
        return MetricValue(None, unit, Status.UNKNOWN, (ErrorCode.INVALID_INPUT.value,),
                           f"{t.metric} reads every source and wavelength the detector receives: a target on it selects none")
    spec = detections.get(t.observation_id)
    obs = observations.get(t.observation_id)
    if spec is None or obs is None:
        return MetricValue(None, unit, Status.UNKNOWN, (ErrorCode.INVALID_INPUT.value,),
                           f"observation {t.observation_id} states no detector (detection): {t.metric} needs one")
    finite = left_out_W is not None and math.isfinite(left_out_W)
    detected = detect(obs, spec, quantity_state_of(obs), left_out_W)
    quantity = detected.result.quantity(t.metric)
    if quantity is None or quantity.value is None:
        reason = "not computed" if quantity is None else quantity.reason
        missing = "" if quantity is None or not quantity.missing else f" (needs {', '.join(quantity.missing)})"
        return MetricValue(None, unit, Status.UNKNOWN, (ErrorCode.DETECTION_QUANTITY_UNAVAILABLE.value,),
                           f"{t.metric} is not available: {reason}{missing}")
    entry = (quantity.value_basis or {}).get("value")
    if entry is None and left_out_W is not None and t.metric != "detector_bandwidth_Hz":
        # (the model's value, truncated where light below the negligible level was left out: that light may add, so the
        # target is graded on the interval it leaves. Where nothing proves one -- nothing bounds the light, or within its
        # reach a validity condition of the detector model is open (``Detected.open_checks``, carried in the target's
        # ``validity``) -- the SNR of the photocurrent at least keeps its value (more light only raises it) while the model
        # holds, and the lock-in's SNR, which light added as steady current lowers towards 0, keeps only its sign.)
        open_checks = detected.open_checks.get(t.metric, ())
        ends = detected.bounds.get(t.metric) if finite and not open_checks else None
        if ends is None:
            ends = (quantity.value, math.inf) if t.metric == "detector_snr" and not open_checks else (0.0, math.inf)
        low, high = min(ends[0], quantity.value), max(ends[1], quantity.value)
        reach = [x for x in (high - quantity.value, quantity.value - low) if math.isfinite(x)]
        truncated = ValueBasis(kind=ValueKind.MODEL, omitted=(Omitted(
            part=OmittedPart.BELOW_NEGLIGIBLE_LEVEL, upper=max(reach) if len(reach) == 2 else None,
            setting="negligible_power_fraction", code=ErrorCode.FAINT_LIGHT_BOUNDED.value),))
        added = f"at most {left_out_W:.4g} W" if finite else "power without a bound"
        unproven = (f"; with that light the detector's {' and '.join(c.parameter for c in open_checks)} may exceed its stated "
                    "limit, where the detector model does not hold" if open_checks else "")
        return MetricValue(quantity.value, unit, Status.PASS, (ErrorCode.FAINT_LIGHT_BOUNDED.value,),
                           f"{t.metric} of the light on the plane is {quantity.value:.6g}; light left out as negligible "
                           f"(negligible_power_fraction) may add {added} to the plane{unproven}",
                           untraced=(low, high), subject="the detector reads", basis=truncated, validity=open_checks)
    if entry is None:
        return MetricValue(quantity.value, unit)
    # (why the plane's power is not simply the model's value: a share taken before the plane, an annular beam counted whole)
    bound_codes = []
    if any(a.power_taken_before_upper_W is not None for a in obs.arrivals):
        bound_codes.append(ErrorCode.POWER_TAKEN_BEFORE_PLANE.value)
    if any(a.detected_power_min_W is not None and a.detected_power_min_W < received_power_W(a) for a in obs.arrivals):
        bound_codes.append(ErrorCode.INCOMPLETE_ENVELOPE.value)
    if left_out_W is not None and t.metric == "detector_snr" and t.metric in detected.bounds:
        # (a plane whose power is a bound -- a share taken before it, an annular beam -- whose light left out as negligible
        # may add too: graded on the interval from the least the entry allows to the most that light brings (``detect``;
        # unbounded above where nothing bounds it or a validity condition it may leave open applies, those carried in the
        # target's ``validity``), the entry merged with the stated truncation -- decision W28, as ``power_W`` there)
        open_checks = detected.open_checks.get(t.metric, ())
        low, high = detected.bounds[t.metric]
        truncated = ValueBasis(kind=ValueKind.MODEL, omitted=(Omitted(
            part=OmittedPart.BELOW_NEGLIGIBLE_LEVEL, upper=high - quantity.value if math.isfinite(high) else None,
            setting="negligible_power_fraction", code=ErrorCode.FAINT_LIGHT_BOUNDED.value),))
        added = f"at most {left_out_W:.4g} W" if finite else "power without a bound"
        unproven = (f"; with that light the detector's {' and '.join(c.parameter for c in open_checks)} may exceed its stated "
                    "limit, where the detector model does not hold" if open_checks else "")
        return MetricValue(quantity.value, unit, Status.PASS, tuple(dict.fromkeys((*bound_codes, ErrorCode.FAINT_LIGHT_BOUNDED.value))),
                           f"{t.metric} of the traced light on the plane is {quantity.value:.6g}; the plane's power is "
                           f"{detected.result.power_state.value if detected.result.power_state is not None else 'not that light alone'} "
                           f"({', '.join(part.part.value for part in entry.omitted) or entry.kind.value}), and light left out "
                           f"as negligible (negligible_power_fraction) may add {added} to it{unproven}",
                           untraced=(low, high), subject="the detector reads", basis=merge_bases(entry, truncated),
                           validity=open_checks)
    if left_out_W is not None and t.metric in detected.reach:
        # (a plane whose power is an upper end whose light left out as negligible may add too, on a quantity graded with its
        # entry -- the lock-in's SNR there: no proven interval, P9d -- the entry merged with the stated truncation, how far
        # that light may move the value (``Detected.reach``), and the codes of both; graded on the entry, decision W28)
        open_checks = detected.open_checks.get(t.metric, ())
        truncated = ValueBasis(kind=ValueKind.MODEL, omitted=(Omitted(
            part=OmittedPart.BELOW_NEGLIGIBLE_LEVEL, upper=detected.reach[t.metric], setting="negligible_power_fraction",
            code=ErrorCode.FAINT_LIGHT_BOUNDED.value),))
        added = f"at most {left_out_W:.4g} W" if finite else "power without a bound"
        unproven = (f"; with that light the detector's {' and '.join(c.parameter for c in open_checks)} may exceed its stated "
                    "limit, where the detector model does not hold" if open_checks else "")
        return MetricValue(quantity.value, unit, Status.PASS, tuple(dict.fromkeys((*bound_codes, ErrorCode.FAINT_LIGHT_BOUNDED.value))),
                           f"{t.metric} of the traced light on the plane is {quantity.value:.6g}; the plane's power is "
                           f"{detected.result.power_state.value if detected.result.power_state is not None else 'not that light alone'} "
                           f"({', '.join(part.part.value for part in entry.omitted) or entry.kind.value}), and light left out "
                           f"as negligible (negligible_power_fraction) may add {added} to it{unproven}",
                           subject="the detector reads", basis=merge_bases(entry, truncated), validity=open_checks)
    # the detector reads the traced light of a plane whose power is not simply the model's value: the number with what it
    # is (rule T1), graded on the quantity's interval where one is proven (``validation/detection.py``)
    codes = []
    if not obs.power_complete:
        codes.append(ErrorCode.UNTRACED_LIGHT_BOUNDED.value if obs.untraced_power_W else ErrorCode.FAINT_LIGHT_BOUNDED.value)
    codes.extend(bound_codes)
    return MetricValue(quantity.value, unit, Status.PASS, tuple(codes),
                       f"{t.metric} of the traced light on the plane is {quantity.value:.6g}; the plane's power is "
                       f"{detected.result.power_state.value if detected.result.power_state is not None else 'not that light alone'} "
                       f"({', '.join(part.part.value for part in entry.omitted) or entry.kind.value})",
                       untraced=detected.bounds.get(t.metric), subject="the detector reads", basis=entry)


def _evaluate(t: Target, observations: dict[str, ObservationResult], interference: list[InterferenceResult],
              layout: LayoutResult | None, control: ControlResult | None, cfg: NumericalConfig,
              power_budget: PowerBudget | None = None, reach: PlaneReach | None = None,
              exposures: ExposureReport | None = None, truncated_pairs: frozenset[str] = frozenset(),
              actuators: dict[str, tuple[Any, ...]] | None = None, detections: dict[str, Any] | None = None,
              conversions: tuple[Any, ...] = (), plane_bounds: bool = True) -> TargetResult:
    mv = _measure(t, observations, interference, layout, control, cfg, power_budget, reach, exposures, truncated_pairs, actuators,
                  detections, conversions, plane_bounds)
    base = dict(target_id=t.target_id, device_id=t.device_id, observation_id=t.observation_id, metric=t.metric,
                target_value=t.target_value, minimum=t.minimum, maximum=t.maximum, unit=mv.unit or t.unit,
                source_revision=t.source_revision, issue_codes=tuple(mv.codes))
    tol = None
    if t.target_value is not None:
        cands = []
        if t.tolerance_abs is not None:
            cands.append(t.tolerance_abs)
        if t.tolerance_rel is not None:
            cands.append(t.tolerance_rel * abs(t.target_value))
        tol = max(cands)
    if mv.basis is not None and mv.value is not None:
        base["value_basis"] = {"value": mv.basis}
    if mv.validity:
        base["validity"] = mv.validity
    if mv.value is None and mv.untraced is None:
        return TargetResult(status=mv.status, value=None, tolerance=tol, message=mv.message, **base)
    if t.unit is not None and mv.unit and t.unit != mv.unit:
        # the measured value is reported in the metric's unit; the bounds, stated in another unit, are not compared and
        # nothing is converted
        return TargetResult(status=Status.UNKNOWN, value=mv.value, tolerance=tol,
                            message=f"unit mismatch: the value is in {mv.unit!r}; the target's bounds are stated in {t.unit!r} "
                                    "and were not compared",
                            **{**base, "issue_codes": (ErrorCode.INVALID_INPUT.value,)})
    if not mv.graded:
        return TargetResult(status=mv.status, value=mv.value, tolerance=tol, message=mv.message, **base)
    unit = mv.unit or t.unit or ""
    suffix = f" {unit}" if unit and unit != DIMENSIONLESS_UNIT else ""

    def judge(value: float) -> list[str]:
        """Why ``value`` fails the target (empty: it meets it)."""
        out = []
        if t.target_value is not None and tol is not None and abs(value - t.target_value) > tol:
            out.append(f"|error| {abs(value - t.target_value):.6g}{suffix} exceeds the tolerance {tol:.6g}{suffix}")
        if t.minimum is not None and value < t.minimum:
            out.append(f"value {value:.6g}{suffix} is below the minimum {t.minimum:.6g}{suffix}")
        if t.maximum is not None and value > t.maximum:
            out.append(f"value {value:.6g}{suffix} is above the maximum {t.maximum:.6g}{suffix}")
        return out

    requirement = Requirement(minimum=t.minimum, maximum=t.maximum, target_value=t.target_value, tolerance=tol)
    below, above = requirement.below, requirement.above

    def fmt(value: float, digits: int) -> str:
        return f"{value:.{digits}g}{suffix}"

    err = None if t.target_value is None or tol is None or mv.value is None else mv.value - t.target_value
    codes = tuple(base["issue_codes"])
    if mv.lower is None and mv.untraced is None:
        # a point value: rule T1 (a number the measurement does not call known, or an estimate, decides nothing; a one-sided
        # bound decides in the direction it proves)
        msgs = judge(mv.value)
        status = grade_point(mv.value, requirement, basis=mv.basis, measured=mv.status)
        reach = _left_out_reach(t, mv.basis)
        if reach is not None and status in (Status.PASS, Status.WARNING, Status.FAIL):
            ends = _left_out_ends(t, mv.value, reach)
            if _undecided_on(requirement, *ends):
                base["issue_codes"] = codes
                return TargetResult(status=undecided_status(mv.basis), value=mv.value, tolerance=tol, error=err,
                                    message=_left_out_note(mv.message, mv.value, reach, ends, fmt), **base)
        unmet = status is Status.FAIL
        base["issue_codes"] = codes + ((ErrorCode.TARGET_NOT_MET.value,) if unmet else ())
        if unmet or status in (Status.PASS, Status.WARNING, Status.NOT_APPLICABLE):
            message = "; ".join(msgs) or mv.message
        else:  # undecided: why the number decides nothing, then how it stands to the target
            message = "; ".join(m for m in (mv.message, *msgs) if m)
        return TargetResult(status=status, value=mv.value, tolerance=tol, error=err, message=message, **base)
    # The value is only known to lie in an interval [low, high]: an annular beam counted on an active area that may miss part
    # of it ([lower, value], INCOMPLETE_ENVELOPE), and/or light the trace did not follow (``mv.untraced``). The target is
    # met when both ends meet it (the met values form one interval), fails when the whole interval lies on one side of the
    # met values, and is otherwise not decided (UNKNOWN). The reported value stays the traced value.
    low, high = mv.untraced if mv.untraced is not None else (mv.lower, mv.value)
    assert low is not None
    notes = []
    if mv.lower is not None:
        codes = codes + (ErrorCode.INCOMPLETE_ENVELOPE.value,)
        digits = next((d for d in range(INTERVAL_MIN_DIGITS, INTERVAL_MAX_DIGITS + 1)  # enough to tell the ends apart
                       if f"{mv.lower:.{d}g}" != f"{mv.value:.{d}g}"), INTERVAL_MAX_DIGITS)
        notes.append(f"the value lies between {fmt(mv.lower, digits)} and {fmt(mv.value, digits)}, a width of "
                     f"{mv.value - mv.lower:.3g}{suffix} (an annular beam counted on an active area that may miss part of it)")
    if mv.untraced is not None:
        digits = INTERVAL_MIN_DIGITS if not math.isfinite(high) else next(
            (d for d in range(INTERVAL_MIN_DIGITS, INTERVAL_MAX_DIGITS + 1) if f"{low:.{d}g}" != f"{high:.{d}g}"), INTERVAL_MAX_DIGITS)
        span = (f"between {fmt(low, digits)} and {fmt(high, digits)}" if math.isfinite(high) else f"at least {fmt(low, digits)}")
        notes.append(f"{mv.message}: {mv.subject} {span}")
    bound = "; ".join(notes)
    reach = _left_out_reach(t, mv.basis)
    if reach is not None:
        # (the interval is the counted light's; light left out as negligible widens it by its bound, ``_left_out_ends``)
        wide = (_left_out_ends(t, low, reach)[0], _left_out_ends(t, high, reach)[1])
        decided = (not judge(low) and not judge(high)) or above(low) or below(high)
        if decided and _undecided_on(requirement, *wide):
            base["issue_codes"] = codes
            return TargetResult(status=Status.UNKNOWN, value=mv.value, tolerance=tol, error=err,
                                message=_left_out_note(bound, mv.value, reach, wide, fmt), **base)
    met_low, met_high = not judge(low), not judge(high)  # (an unbounded upper end meets a minimum only)
    if met_low and met_high:
        # decided: met across the interval (an annular bound keeps its WARNING; light the trace did not follow is bounded)
        status = worst([mv.status] + ([Status.WARNING] if mv.lower is not None else []), empty=Status.PASS)
        base["issue_codes"] = codes
        return TargetResult(status=status, value=mv.value, tolerance=tol, error=err, message=f"{bound}: the target is met", **base)
    if above(low) or below(high):
        reasons = judge(low) if above(low) else judge(high)
        base["issue_codes"] = codes + (ErrorCode.TARGET_NOT_MET.value,)
        end = "lower" if above(low) else "upper"
        return TargetResult(status=Status.FAIL, value=mv.value, tolerance=tol, error=err,
                            message=f"{bound}: the target is not met even at the {end} end ({'; '.join(reasons)})", **base)
    base["issue_codes"] = codes
    return TargetResult(status=Status.UNKNOWN, value=mv.value, tolerance=tol, error=err,
                        message=f"{bound}: the target is met at some values of that interval and not at others, so it is not decided",
                        **base)
