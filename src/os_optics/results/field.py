"""The coherent field of a run's sampled-field or speckle source at an observation plane (``FieldRequestV1``), as an analysis
document (``AnalysisResultV1``, analysis ``FIELD_PROPAGATION``).

What is computed. The source's field ``U`` (``SampledFieldSourceV1``; for ``SpeckleSourceV1`` each realization's field, the
illumination through its phase screen) is built again from the run's Engine request and propagated as a scalar field to the
plane (``optics/fields/propagation.py``):

* ``ANGULAR_SPECTRUM``: where the source sees the plane through free space (no element on the source axis before it, the same
  medium) and the plane is parallel to the source plane with its axes along the source's: the exact scalar transfer function
  over the axial distance, band-limited; non-paraxial. Its conditions: the band limit keeps at least ``MIN_WINDOW_POWER`` of
  the field's power (``band_power_fraction``), and the window holds the light (below).
* ``PARAXIAL_TRANSFER``: the generalized Collins integral of the source axis's first-order ray-transfer matrix to the plane
  (the run's own real-ray finite-difference matrix of a probe beam launched along the source axis; positions in mm,
  directions as ``n`` times the direction cosines), separable along the plane's axes. Its conditions: the transfer does not
  couple the axes (no image rotation other than by a multiple of 90 degrees), it is symplectic within
  ``MAX_SYMPLECTIC_RESIDUAL``, one of its discretizations is sampled, the field's angular spread (the sine holding 99 % of its
  angular spectrum) is within ``SCALAR_SINE_LIMIT``, the window holds the light (below), and the field's own rays (the run's
  geometric trace of the field, traced again here; only the rays that reach the plane along the axis's path, through the
  same interactions in the same order) show that

  - the apertures pass the field: at least ``MIN_GEOMETRIC_TRANSMISSION`` of the launched rays reach the plane
    (``aperture_pass_share``), and their power over the source's power times the axis's transmission is within
    ``[MIN_GEOMETRIC_TRANSMISSION, MAX_GEOMETRIC_TRANSMISSION]`` (``geometric_transmission``: outside it the field's rays are
    cut, or the elements transmit them differently across the field than along its axis);
  - first order describes the system over the field (``_aberration_fit``, ``_aberrated_source``): rays over the field's phase
    space (spread as its amplitude, its diffraction included) are traced through the elements; each one's landing point and
    direction against the matrix's prediction from its launch, carried back to the source plane by the matrix's inverse, is
    its position and direction error there. Free space over ``s`` (taken where the position errors are smallest) brings them
    to a plane where the aberration is a phase, ``grad(phi) = k0 n`` times the direction errors, fitted by a polynomial
    (degree ``ABERRATION_DEGREE``). The field with that phase applied at that plane is propagated by the same transfer: the
    plane's values over the region change by at most the request's ``value_tolerance`` (``aberration_value_change``: relative
    for the powers, irradiances, rms widths and contrast, counted ``ABERRATION_PROFILE_MARGIN`` times -- near an image waist the
    fit predicts as little as half of their change --, relative to the rms width for the centroid), the phase is sampled
    (``aberration_phase_step``, at most ``MAX_ABERRATION_PHASE_STEP`` between neighbouring samples where the light is), and
    what the phase leaves out (the fit's residual, the position errors left at ``s``) bounds the field's relative rms error
    by at most ``MAX_ABERRATION_UNMODELLED`` (``aberration_unmodelled``; the Poincare inequality of a Gaussian weight). With
    ``include_field`` the complex field's relative rms change over the region (up to a constant phase) must be at most
    ``MAX_ABERRATION_FIELD_ERROR`` too (``aberration_field_error``): away from a focus an aberrated phase may leave the
    irradiance within 1e-3 while the field itself differs. That condition is the exported field's alone: where it fails the
    plane's values are those of the same request without ``include_field``, and the field arrays are left out
    (``arrays.field_withheld``: the check, its observed value and its limit). The change is the aberration's own effect at this plane, not a
    bound: a plane near a focus, where an aberration first order in the phase moves the peak, is judged as one at the focus
    (where the Strehl ratio's second order governs) or far from it.

  The elements' power transmission along the axis (a probe beam's: interface reflections, coatings, absorption) scales the
  field. The probe is a Gaussian beam of the field's rms size, and where a clear aperture cuts that Gaussian's tail the cut
  is undone (each aperture's share of the probe's arriving power): the apertures are the field's rays' to pass or not
  (``aperture_pass_share``), not the probe profile's. The values are the first-order field's (the check above measures how far the rays' aberration would move them); the
  apertures' diffraction and the elements' action on the polarization are not
  modelled.
* ``AUTO``: the angular spectrum where it applies and all its conditions hold, else the paraxial transfer.

The window holds the light (both methods; realization 0): the computation is periodic over its output window (a circular
convolution, or the chirp route's periodic sum), so light that leaves the window folds back into it with its power
unchanged. The same field is therefore propagated again on a window twice as wide on the same output lattice (twice the
padding; the chirp route on the input interpolated to half the pitch, ``optics/fields/propagation.reference_operator``), and
(``window_power_fraction``) at least ``MIN_WINDOW_POWER`` of its power must lie inside the first window, and
(``window_refinement_change``) the plane's values over the region (window and region power, mean and peak irradiance,
centroid, rms widths, contrast) may change by at most the request's ``value_tolerance`` (relative; the centroid relative to the
rms width), else ``WINDOW_LOSES_POWER`` / ``WINDOW_NOT_CONVERGED``. With ``refinement`` these values are the convergence
records of a single pattern's quantities.

The accuracy asked. ``value_tolerance`` (default ``DEFAULT_VALUE_TOLERANCE``, at most ``MAX_VALUE_TOLERANCE``) is the largest
change of the plane's values the two accuracy checks admit. A request that states another value has it in the settings, the
two checks name it (``setting``), and each plane value carries a ``value_basis`` of kind ``MODEL`` with those checks: the
number is the model's within the accuracy the caller asked for.

A plane outside a model's conditions gets no values: its quantities are ``OUT_OF_MODEL_SCOPE`` with the first failed
condition as reason and every failed condition in ``depends_on``, and the models list every condition with its observed
value. Light a run's model-scope finding reaches (the finding lists the field's line on the plane, ``affected_sampled_lines``):
every quantity with a number is ``OUT_OF_MODEL_SCOPE`` with the finding's ``ESTIMATE``, its number kept
(``imaging.outside_model``), and ``arrays["model_scope"]`` lists the findings, which ``field_npz`` carries into the field file. The analysis has no resource budget of its own: its
execution is ``COMPLETED`` either way, and the quantities' states say what is stated.

Statistics, over the request's ``region`` of the plane (default: the whole output window), per realization ``m``: the power
``P_m`` and mean irradiance on the region, and the contrast ``C_m = std(I_m) / mean(I_m)`` over the region's grid points
(population standard deviation). Realizations are mutually incoherent (their irradiances add): the averaged irradiance is
their mean, with its own contrast ``C_avg``; for ``M`` independent, identically distributed patterns ``C_avg`` is
``mean(C_m) / sqrt(M)`` (reported beside it: a check of their independence), and fully developed speckle (a strong
diffuser seen through many correlation cells) has ``C_m`` near 1. A speckle source's document also gives its screens' rms
phase over the illumination and the share of the power they leave unscattered (``|<exp(i phi)>|^2``, ``exp(-sigma^2)`` for a
Gaussian screen), and whether the pattern is fully developed (``exp(-sigma^2)`` at most ``MAX_UNSCATTERED_SHARE``): a weaker
screen's pattern holds the unscattered beam and its diffraction by the illumination's edge, which the contrast also counts.
The rms widths need ``MIN_STATISTICS_POINTS`` grid points of the region along their axis and the contrasts as many along both
(else ``NOT_APPLICABLE``, ``TOO_FEW_POINTS``). Convergence: realizations' half-sample estimates (the first ``M / 2``
realizations against all ``M``) and the window's refinement (above).

Conventions: the plane's coordinates ``(u, v)`` in mm; arrays ``[v][u]``; irradiance in W/mm^2; the field in sqrt(W/mm^2) with
time dependence ``exp(-i w t)``, its phase relative to the source's stated phase plus the chief ray's optical path
(``PARAXIAL_TRANSFER``) or ``k z`` (``ANGULAR_SPECTRUM``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from pydantic import ValidationError

from ..analysis import completed, document
from ..contracts.analysis import (
    AnalysisResultV1,
    Convergence,
    ConvergenceMethod,
    Interval,
    IntervalKind,
    ModelBasis,
    Quantity,
    QuantityScope,
    ValidityCheck,
)
from ..contracts.basis import ValueBasis
from ..contracts.request import ValidationRequest
from ..contracts.source_field import DEFAULT_VALUE_TOLERANCE, FieldRequestV1
from ..errors.status import QuantityState, ValueKind
from ..math.vectors import dot, matvec
from ..optics.fields.propagation import (
    CHIRP,
    AngularSpectrum,
    angular_extent_sine,
    angular_spectrum,
    paraxial_conditions,
    paraxial_transfer,
    reference_angular_spectrum,
    reference_operator,
)
from ..optics.fields.source_fields import complex_field, realization_phase
from ..optics.materials.library import MaterialLibrary
from ..optics.rays.emitter_samples import EmitterSamples, field_samples, sobol_points
from ..optics.rays.sampled import SampledLine, SampledTrace, SampleTracer, field_bases
from ..optics.rays.tracer import SINK_APERTURE_TRUNCATION, SINK_CLIPPED, ArrivalRecord, SourceInput, Tracer
from ..scene.lower import lower_scene
from ..scene.model import EmitterG, SourceG
from ..units import NM_TO_MM
from .imaging import _sig
from .model import ResultsError
from .quantity import POWER_OUT_OF_MODEL_SCOPE, power_scope_message
from .pupil_field import SCALAR_SINE_LIMIT

FIELD_ANALYSIS = "FIELD_PROPAGATION"
FIELD_ANALYSIS_VERSION = "1.4"
FIELD_INVALID = "FIELD_INVALID"
ANGULAR_SPECTRUM = "ANGULAR_SPECTRUM"
PARAXIAL_TRANSFER = "PARAXIAL_TRANSFER"
#: The window must keep this share of the field's power.
MIN_WINDOW_POWER = 0.99
#: The source's rays must bring this share of its power to the plane (the paraxial transfer has no apertures).
MIN_GEOMETRIC_TRANSMISSION = 0.999
#: The rays' power over the probe-transmitted power is at most this: above it the probe beam that gives the axis's
#: transmission is cut by an aperture.
MAX_GEOMETRIC_TRANSMISSION = 1.001
#: The request setting that holds the accuracy of the plane's values: the largest change the window refinement
#: (``window_refinement_change``) and the rays' aberration beyond first order (``aberration_value_change``) may make
#: (relative for the powers, irradiances, rms widths and contrast, relative to the rms width for the centroid).
VALUE_TOLERANCE_SETTING = "value_tolerance"
#: The conditions of the exported complex field alone (``include_field``): the plane's values do not depend on them.
FIELD_ONLY_CHECKS = ("aberration_field_error",)
#: The change of the plane's profile values (powers, irradiances, rms widths, contrast) under the fitted aberration counts this
#: many times in ``aberration_value_change`` (the centroid's once). Within about one Rayleigh range of an image waist the
#: first-order fit predicts as little as 0.35-0.5 of the peak change an exact diffraction reference shows (plano-convex
#: singlets, 405-1064 nm, beam radii 0.4-0.9 mm: the reference exceeds the predicted change by at most 0.47 % of the value
#: there); counted twice, an admitted change of at most 0.5 % leaves the value within 1 % of that reference. The centroid's
#: shift is predicted exactly (decentred and tilted lenses).
ABERRATION_PROFILE_MARGIN = 2.0
#: With ``include_field``: the largest relative rms difference of the complex field over the region that the rays' aberration
#: makes (``aberration_field_error``, up to a constant phase).
MAX_ABERRATION_FIELD_ERROR = 0.1
#: The largest bound on the field's relative rms error from what the fitted aberration phase leaves out
#: (``aberration_unmodelled``).
MAX_ABERRATION_UNMODELLED = 0.01
#: The largest step of the fitted aberration phase between neighbouring samples where the light is (rad): above it the phase
#: is not sampled.
MAX_ABERRATION_PHASE_STEP = 0.5 * math.pi
#: The polynomial degree of the fitted aberration phase.
ABERRATION_DEGREE = 8
#: The fit needs at least this many of the phase-space rays on the plane.
MIN_ABERRATION_RAYS = 64
#: The phase-space rays' diffraction offsets are this times the amplitude's rms angular spread (a Gaussian's amplitude is
#: sqrt(2) wider than its irradiance).
AMPLITUDE_SPREAD = math.sqrt(2.0)
#: Where the light is, for the phase-step check: irradiance at least this share of the peak.
LIT_SHARE = 1e-6
#: The contrast's change is taken relative to at least this contrast.
CONTRAST_CHANGE_FLOOR = 0.01
#: The rms width along an axis needs at least this many grid points of the region along it, and the contrast as many along
#: both.
MIN_STATISTICS_POINTS = 8
#: A speckle pattern is fully developed when the screen leaves at most this share of the power unscattered (|<exp(i phi)>|^2
#: over the illumination; exp(-sigma^2) for a Gaussian screen of rms sigma: sigma >= 2.15 rad).
MAX_UNSCATTERED_SHARE = 0.01
#: The largest |ad - bc - 1| of an axis of the paraxial transfer.
MAX_SYMPLECTIC_RESIDUAL = 1e-3
#: The floor of an angle row's scale when the coupling between the source's axes is measured (a row of the ray matrix's angles).
ANGLE_ROW_SCALE_FLOOR = 1e-3
#: Couplings between the axes up to this (relative to the paired entries) are rounding of the finite differences.
COUPLING_TOLERANCE = 1e-6
#: A plane's axes count as along the source's within this (direction cosines).
AXIS_TOLERANCE = 1e-9
#: Two output lattices share a point within this share of the pitch.
LATTICE_TOLERANCE = 1e-6
#: The field arrays of include_field hold at most this many points per axis.
MAX_FIELD_POINTS = 1024
#: Percent per unit share (a share written in a message).
PERCENT = 100.0

NOT_A_FIELD_SOURCE = "NOT_A_FIELD_SOURCE"
SOURCE_REQUIRED = "SOURCE_REQUIRED"
PLANE_NOT_REACHED = "PLANE_NOT_REACHED"
TRANSFER_UNAVAILABLE = "TRANSFER_UNAVAILABLE"
COUPLED_TRANSFER = "COUPLED_TRANSFER"
NOT_FREE_SPACE = "NOT_FREE_SPACE"
PLANE_NOT_PARALLEL = "PLANE_NOT_PARALLEL"
GRID_INVALID = "GRID_INVALID"
APERTURES_CUT_FIELD = "APERTURES_CUT_FIELD"
NOT_PARAXIAL = "NOT_PARAXIAL"
WINDOW_LOSES_POWER = "WINDOW_LOSES_POWER"
NOT_SYMPLECTIC = "NOT_SYMPLECTIC"
REGION_EMPTY = "REGION_EMPTY"
ABERRATED = "ABERRATED_BEYOND_FIRST_ORDER"
WINDOW_NOT_CONVERGED = "WINDOW_NOT_CONVERGED"
TOO_FEW_POINTS = "TOO_FEW_POINTS"
FIELD_TOO_LARGE = "FIELD_TOO_LARGE"

ASM_MODEL_ID = "angular_spectrum"
PARAXIAL_MODEL_ID = "paraxial_transfer"
STATISTICS_MODEL_ID = "irradiance_statistics"
GEOMETRIC_MODEL_ID = "geometric_rays"

STATISTICS_MODEL = ModelBasis(model_id=STATISTICS_MODEL_ID, name="spatial irradiance statistics over the region",
                              version=FIELD_ANALYSIS_VERSION, assumptions=(
    "contrast std(I) / mean(I) over the region's grid points (population standard deviation); it includes the variation of the "
    "mean irradiance across the region, so a region over which the mean is flat measures speckle alone",
    "realizations are mutually incoherent: their irradiances add (a moving diffuser over an exposure); M independent, identically "
    "distributed patterns average to the contrast mean(C_m) / sqrt(M)",
    "fully developed speckle (each point of the plane lit through many correlation cells of a screen of rms phase well above "
    "1 rad) has a negative-exponential irradiance and contrast 1"))
GEOMETRIC_MODEL = ModelBasis(model_id=GEOMETRIC_MODEL_ID, name="the field's ray decomposition", version=FIELD_ANALYSIS_VERSION,
                             assumptions=("the source's rays (positions by |U|^2, directions along the field's local wavevector), traced as "
                                          "in the run; their powers add: geometric transport, no interference",))


class FieldError(ResultsError):
    """A request that is not valid or names no field source: no analysis is made."""


class _OutOfModel(Exception):
    def __init__(self, reason: str, message: str, state: QuantityState = QuantityState.NOT_APPLICABLE) -> None:
        super().__init__(message)
        self.reason, self.message, self.state = reason, message, state


def parse_request(request: Any) -> FieldRequestV1:
    if isinstance(request, FieldRequestV1):
        return request
    try:
        return FieldRequestV1.model_validate({} if request is None else request)
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(str(p) for p in e['loc']) or 'request'}: {e['msg']}" for e in exc.errors(include_url=False))
        raise FieldError(FIELD_INVALID, problems) from None


@dataclass
class _AxisRays:
    """The source's rays that cross the plane along the axis's path (the probe's interactions, in order): their launch sample,
    landing point and direction on the plane, and power; and how many rays the run launched."""

    sample_index: np.ndarray
    u: np.ndarray
    v: np.ndarray
    du: np.ndarray
    dv: np.ndarray
    power_W: np.ndarray
    launched: int
    #: the launch ``(x, y, a, b)`` of each ray (source frame: position, direction sines), when the rays are not the run's
    launch: np.ndarray | None = None


@dataclass
class _Setup:
    emitter: EmitterG
    wavelength_nm: float
    power_W: float
    n_source: float
    arrival: ArrivalRecord
    geometric_W: float
    axis_transmission: float
    rays: _AxisRays
    #: what traces the field's phase-space rays (``_phase_space_rays``) when the paraxial transfer needs them
    trace: tuple[Any, Any, Any, tuple[str, ...], str] | None = None
    _phase_rays: _AxisRays | None = None

    def phase_rays(self) -> _AxisRays:
        if self._phase_rays is None:
            assert self.trace is not None
            self._phase_rays = _phase_space_rays(self, *self.trace)
        return self._phase_rays


class _LineageTracer(SampleTracer):
    """The run's sampled trace, keeping the interactions before each plane crossing (one entry per crossing row)."""

    def _do_observe(self, pk, surf, h, inter) -> str | None:  # type: ignore[no-untyped-def]
        if pk.power_bound_W is None:
            key = (inter.observation_id or "", pk.src.source_id, pk.spectral_index)
            self.crossing_lineages.setdefault(key, []).append(tuple(pk.lineage))
        return super()._do_observe(pk, surf, h, inter)


class _PhaseSpaceTracer(_LineageTracer):
    """Traces given rays of a field source (``_phase_space_rays``) with the run's physics, keeping each crossing's lineage."""

    def trace_rays(self, emitter: EmitterG, points: np.ndarray, directions: np.ndarray, opl: np.ndarray) -> SampledTrace:
        """Rays launched at ``points`` along ``directions`` (unit; the emitter's frame), each of an equal share of its power."""

        wavelength_nm, weight = emitter.spectrum[0]
        power, count = emitter.power_W * weight, points.shape[0]
        self.out.inputs.append(SourceInput(emitter.source_id, emitter.owner_id, 0, wavelength_nm, power))
        line = SampledLine(emitter.source_id, emitter.owner_id, 0, wavelength_nm, power, power / max(count, 1), count, emitter.seed)
        self.sampled.lines.append(line)
        self._line = line
        state = self._sample_state(emitter, wavelength_nm)
        if state is None or count == 0:
            return self.sampled
        gstate, n = state
        axes = np.column_stack([emitter.e1, emitter.e2, emitter.direction])
        pattern = emitter.pattern
        jones = None if pattern is None or pattern.jones is None else np.tile(pattern.jones, (count, 1))
        samples = EmitterSamples(emitter.position + matvec(axes, points), matvec(axes, directions), np.zeros(count, dtype=bool), jones, opl)
        bases = field_bases(samples.directions, emitter.e1, emitter.e2)
        self.recorder = None
        for k in range(count):
            if self.schedule.steps >= self.sampled.step_budget:
                break
            line.launched = k + 1
            self._sample_index = k
            self._keep = False
            self._enqueue(self._packet(emitter, 0, k, wavelength_nm, line.weight_W, gstate, n, samples, bases))
            self._drain()
        return self.sampled


def _axis_rows(trace: SampledTrace, lineages: dict, monitor: str, source_id: str, axis: tuple[str, ...]
               ) -> list[tuple[int, float, float, float, float, float]]:
    """The crossings of ``monitor`` by the source's rays along the axis's path: (sample, u, v, l_u, l_v, power)."""

    keep: list[tuple[int, float, float, float, float, float]] = []
    for key, r in trace.rows.items():
        if key[0] != monitor or key[1] != source_id:
            continue
        for i, lineage in enumerate(lineages.get(key, [])):
            if _interactions(lineage) == axis:  # (the axis's path: not a ghost, not another crossing)
                keep.append((r.sample_index[i], r.u_mm[i], r.v_mm[i], r.direction_u[i], r.direction_v[i], r.power_W[i]))
    return keep


def _envelope_sines(u_field: np.ndarray, px: float, py: float, wavelength_mm: float, n: float) -> tuple[float, float]:
    """The rms direction sines (along x, y) of the field's amplitude ``|U|`` alone: its diffraction spread about the field's
    local wavevector."""

    ny, nx = u_field.shape
    spectrum = np.abs(np.fft.fft2(np.abs(u_field), s=(2 * ny, 2 * nx))) ** 2
    total = float(spectrum.sum())
    fx, fy = np.fft.fftfreq(2 * nx, px), np.fft.fftfreq(2 * ny, py)
    sx = math.sqrt(float((spectrum.sum(axis=0) * fx * fx).sum()) / total)
    sy = math.sqrt(float((spectrum.sum(axis=1) * fy * fy).sum()) / total)
    return wavelength_mm * sx / n, wavelength_mm * sy / n


def _phase_space_rays(setup: _Setup, scene, library, cfg, axis: tuple[str, ...], monitor: str) -> _AxisRays:  # type: ignore[no-untyped-def]
    """The field's rays over its phase space (the aberration check's), spread as the field's amplitude rather than its
    irradiance (a focus's peak and the field's first-order errors weight the aberration by the amplitude, whose tails reach
    further): positions distributed as ``|U|`` (the run's Sobol sequence through the inverse distribution of ``|U|``), each
    along the field's local wavevector there turned by a diffraction offset drawn from a Gaussian of ``sqrt(2)`` times the
    amplitude's own rms angular spread (``_envelope_sines``; the amplitude of a Gaussian spread is ``sqrt(2)`` wider than its
    irradiance; Box-Muller on the samples' third and fourth Sobol coordinates, which a field's rays leave unused), so that the
    rays fill the beam wherever it goes, a small waist's diffraction cone included. Traced as the run's rays; those that cross
    the plane along the axis's path, with their launch."""

    emitter = setup.emitter
    pattern = emitter.pattern
    assert pattern is not None
    spread = replace(emitter, pattern=replace(pattern, weights=np.sqrt(pattern.weights), _cdf={}))
    points, directions, opl = field_samples(spread, setup.n_source)
    lam = setup.wavelength_nm * NM_TO_MM
    sigma_x, sigma_y = _envelope_sines(complex_field(pattern, 0, setup.power_W), pattern.grid.px, pattern.grid.py, lam, setup.n_source)
    u = sobol_points(emitter.sample_count, emitter.seed)
    radius = np.sqrt(-2.0 * np.log(u[:, 2]))
    angle = 2.0 * math.pi * u[:, 3]
    sx = directions[:, 0] + AMPLITUDE_SPREAD * sigma_x * radius * np.cos(angle)
    sy = directions[:, 1] + AMPLITUDE_SPREAD * sigma_y * radius * np.sin(angle)
    inside = sx * sx + sy * sy < 1.0
    points, sx, sy, opl = points[inside], sx[inside], sy[inside], opl[inside]
    turned = np.column_stack([sx, sy, np.sqrt(np.maximum((1.0 - sx * sx) - sy * sy, 0.0))])
    tracer = _PhaseSpaceTracer(replace(scene, sources=[], emitters=[emitter]), library, cfg)
    tracer.crossing_lineages = {}
    trace = tracer.trace_rays(emitter, points, turned, opl)
    keep = _axis_rows(trace, tracer.crossing_lineages, monitor, emitter.source_id, axis)
    columns = [np.array([row[c] for row in keep], dtype=np.float64) for c in range(6)]
    index = columns[0].astype(np.intp)
    launch = np.column_stack([points[index, 0], points[index, 1], sx[index], sy[index]])
    return _AxisRays(index, columns[1], columns[2], columns[3], columns[4], columns[5], points.shape[0], launch)


def _interactions(lineage: tuple[str, ...]) -> tuple[str, ...]:
    """A path's interactions with the elements (its plane crossings left out)."""

    return tuple(step for step in lineage if not step.startswith("observation/") and not step.endswith(":observe"))


def _probe(emitter: EmitterG, wavelength_nm: float, cfg) -> SourceG:
    """A Gaussian beam along the source axis, of the field's rms size about its centroid (wherever the field lies on its grid):
    its arrival gives the chief ray and the first-order transfer to the plane, and its power the elements' transmission along
    the axis. (A probe as wide as the field's distance from the axis would be cut by apertures the field itself passes.)"""

    pattern = emitter.pattern
    assert pattern is not None
    w = pattern.weights / float(pattern.weights.sum())
    x, y = pattern.grid.x(), pattern.grid.y()
    wx, wy = w.sum(axis=0), w.sum(axis=1)
    mx, my = float((wx * x).sum()), float((wy * y).sum())
    sx = math.sqrt(max(float((wx * x * x).sum()) - mx * mx, 0.0))
    sy = math.sqrt(max(float((wy * y * y).sum()) - my * my, 0.0))
    return SourceG(source_id=emitter.source_id, owner_id=emitter.owner_id, position=emitter.position, direction=emitter.direction,
                   e1=emitter.e1, e2=emitter.e2, spectrum=((wavelength_nm, 1.0),), power_W=1.0,
                   w0x=max(2.0 * sx, pattern.grid.px), w0y=max(2.0 * sy, pattern.grid.py), z_waist=0.0, m2x=1.0, m2y=1.0,
                   jones=pattern.jones if pattern.jones is not None else np.array([1.0 + 0j, 0j]), medium=emitter.medium,
                   coherence_lengths_mm=None, coherence_group=f"source:{emitter.source_id}", ring_count=cfg.pupil_ring_count,
                   points_per_ring=cfg.pupil_points_per_ring)


def _axis_transmission(out: Any, arrival: ArrivalRecord) -> float:
    """The elements' power transmission along the source axis: the probe beam's power on the plane (it is launched with 1 W),
    with the cuts its own Gaussian profile took at clear apertures undone.

    The probe's profile is not the field's: an aperture that passes every ray of a flat-top field still cuts the tail of a
    Gaussian of the field's rms size. At each aperture on the probe's way the tracer booked the cut (``APERTURE_TRUNCATION``,
    and ``CLIPPED`` for a central obstruction) and went on with the rest, so the share kept there is one less the booked
    power over the power arriving, and the probe's power divided by those shares is what interfaces, coatings and absorption
    alone transmit. Where the bookings and the arrivals at a surface cannot be paired (a path that meets an aperture
    several times, cut at some of them only) the probe's power is returned as it is."""

    paths = {path.path_id: path for path in out.paths}
    chain: list[str] = []
    current: str | None = arrival.path_id
    while current is not None and current in paths and current not in chain:
        chain.append(current)
        current = paths[current].parent_path_id
    upstream = {entry.rpartition(":")[0] for entry in arrival.lineage}
    booked: dict[tuple[str, str], list[float]] = {}
    for flow in out.power_flows:
        if flow.kind in (SINK_APERTURE_TRUNCATION, SINK_CLIPPED) and flow.path_id in chain and flow.surface_id in upstream and flow.power_W > 0.0:
            booked.setdefault((flow.path_id, flow.surface_id), []).append(float(flow.power_W))
    kept = 1.0
    for (path_id, surface_id), cuts in booked.items():
        arriving = [segment.power_end_W for segment in paths[path_id].segments if segment.end_surface_id == surface_id]
        if any(power is None or power <= 0.0 for power in arriving) or not arriving:
            return float(arrival.power_W)
        if len(arriving) == 1:
            pairs = [(math.fsum(cuts), arriving[0])]
        elif len(arriving) == len(cuts):
            pairs = list(zip(cuts, arriving))
        else:
            return float(arrival.power_W)
        for cut, power in pairs:
            share = 1.0 - cut / power
            if not share > 0.0:
                return float(arrival.power_W)
            kept *= share
    return float(arrival.power_W) / kept


def _setup(results: Any, monitor: str, parsed: FieldRequestV1) -> _Setup:
    validated = ValidationRequest.model_validate(results.request)
    cfg = validated.configuration.numerical
    library = MaterialLibrary(validated.project.materials, cfg)
    scene, _poses = lower_scene(validated.project, cfg, library)
    fields = [e for e in scene.emitters if e.pattern is not None and e.pattern.coherent]
    if parsed.source_id is not None:
        emitter = next((e for e in fields if e.source_id == parsed.source_id), None)
        if emitter is None:
            raise FieldError(NOT_A_FIELD_SOURCE, f"{parsed.source_id!r} is not a sampled-field or speckle source of this run; "
                                                 f"those are {[e.source_id for e in fields]}")
    elif len(fields) != 1:
        raise FieldError(SOURCE_REQUIRED if fields else NOT_A_FIELD_SOURCE,
                         f"name the source (source_id): this run's sampled-field and speckle sources are {[e.source_id for e in fields]}"
                         if fields else "this run has no sampled-field or speckle source")
    else:
        emitter = fields[0]
    refused = emitter.pattern.out_of_scope  # type: ignore[union-attr]
    if refused is not None:  # (a field the run did not trace: undersampled phase or evanescent light)
        raise FieldError(refused.code.value, refused.message)
    wavelength = emitter.spectrum[0][0]
    n_source, _issue = library.evaluator(emitter.medium).index(wavelength)
    if n_source is None:
        raise FieldError(NOT_A_FIELD_SOURCE, f"source {emitter.source_id}'s medium {emitter.medium} has no index at {wavelength} nm")
    probe_scene = replace(scene, sources=[_probe(emitter, wavelength, cfg)], emitters=[])
    out = Tracer(probe_scene, library, cfg, include_pupil=True).run()
    crossings = sorted((a for a in out.arrivals if a.observation_id == monitor and a.crossing_index == parsed.crossing_index),
                       key=lambda a: (-a.power_W, a.path_id))
    if not crossings:
        raise _OutOfModel(PLANE_NOT_REACHED, f"the axis of source {emitter.source_id} does not reach plane {monitor!r}"
                                             + (f" a {parsed.crossing_index + 1}. time" if parsed.crossing_index else ""))
    if crossings[0].power_scope:
        # (the axis's power behind these surfaces is its chief ray's, not the light's: ArrivalResult.power_out_of_scope_at)
        raise _OutOfModel(POWER_OUT_OF_MODEL_SCOPE, power_scope_message(list(crossings[0].power_scope), "the field's power"),
                          QuantityState.OUT_OF_MODEL_SCOPE)
    tracer = _LineageTracer(replace(scene, sources=[], emitters=[emitter]), library, cfg)
    tracer.crossing_lineages = {}
    trace = tracer.run_samples()
    axis = _interactions(crossings[0].lineage)
    keep = _axis_rows(trace, tracer.crossing_lineages, monitor, emitter.source_id, axis)
    launched = next((line.launched for line in trace.lines if line.source_id == emitter.source_id), 0)  # (a field has one line)
    columns = [np.array([row[c] for row in keep], dtype=np.float64) for c in range(6)]
    rays = _AxisRays(columns[0].astype(np.intp), columns[1], columns[2], columns[3], columns[4], columns[5], launched)
    geometric = math.fsum(row[5] for row in keep)
    return _Setup(emitter, wavelength, emitter.power_W, float(n_source), crossings[0], geometric, _axis_transmission(out, crossings[0]), rays,
                  (scene, library, cfg, axis, monitor))


@dataclass
class _Plane:
    """A propagated field on the plane: ``field`` ``[v][u]`` at the coordinates ``u``, ``v`` (mm, ascending)."""

    field: np.ndarray
    u: np.ndarray
    v: np.ndarray


def _signed_permutation(matrix: np.ndarray) -> bool:
    rounded = np.round(matrix)
    return bool(np.all(np.abs(matrix - rounded) <= AXIS_TOLERANCE)) and sorted(np.abs(rounded).ravel().tolist()) == [0.0, 0.0, 1.0, 1.0]


def _ascending(plane: _Plane) -> _Plane:
    field, u, v = plane.field, plane.u, plane.v
    if u.size > 1 and u[0] > u[-1]:
        field, u = field[:, ::-1], u[::-1]
    if v.size > 1 and v[0] > v[-1]:
        field, v = field[::-1, :], v[::-1]
    return _Plane(field, u, v)


def _free_space(setup: _Setup) -> tuple[float, np.ndarray] | None:
    """``(z, R)`` when the source sees the plane through free space and the plane is parallel to the source plane with its axes
    along the source's (``R``: the plane's (u, v) per the source's (x, y)); ``None`` otherwise."""

    arrival, emitter = setup.arrival, setup.emitter
    if any(not step.startswith("observation/") for step in arrival.lineage) or arrival.medium != emitter.medium:
        return None
    if abs(abs(float(dot(arrival.plane_normal, emitter.direction))) - 1.0) > AXIS_TOLERANCE:
        return None
    rotation = np.array([[float(dot(arrival.plane_u, emitter.e1)), float(dot(arrival.plane_u, emitter.e2))],
                         [float(dot(arrival.plane_v, emitter.e1)), float(dot(arrival.plane_v, emitter.e2))]])
    if not _signed_permutation(rotation):
        return None
    return float(dot(arrival.position - emitter.position, emitter.direction)), np.round(rotation)


def _asm_plane(setup: _Setup, result: AngularSpectrum, rotation: np.ndarray) -> _Plane:
    uc, vc = setup.arrival.uv
    if rotation[0, 0] != 0.0:
        plane = _Plane(result.field, uc + rotation[0, 0] * result.x, vc + rotation[1, 1] * result.y)
    else:
        plane = _Plane(result.field.T, uc + rotation[0, 1] * result.y, vc + rotation[1, 0] * result.x)
    return _ascending(plane)


def _window_check(plane: _Plane, reference: _Plane | None) -> ValidityCheck:
    """The share of the light inside the output window: the reference's power (the same field on a window twice as wide, on
    the same lattice) inside ``plane``'s window over its whole power. Light that leaves the window folds back into it in the
    computation (circular convolution, periodic sum) without any change of its power; the wider window shows it outside."""

    if reference is None:
        return ValidityCheck(parameter="window_power_fraction", minimum=MIN_WINDOW_POWER, observed=None, within=False)
    irradiance = np.abs(reference.field) ** 2
    du = float(plane.u[1] - plane.u[0]) if plane.u.size > 1 else 0.0
    dv = float(plane.v[1] - plane.v[0]) if plane.v.size > 1 else 0.0
    inside_u = (reference.u >= plane.u[0] - 0.25 * du) & (reference.u <= plane.u[-1] + 0.25 * du)
    inside_v = (reference.v >= plane.v[0] - 0.25 * dv) & (reference.v <= plane.v[-1] + 0.25 * dv)
    total = float(irradiance.sum())
    share = float(irradiance[np.ix_(inside_v, inside_u)].sum()) / total if total > 0.0 else 0.0
    return ValidityCheck(parameter="window_power_fraction", minimum=MIN_WINDOW_POWER, observed=_sig(share), within=share >= MIN_WINDOW_POWER)


def _angular_spectrum(setup: _Setup, u_field: np.ndarray, padding: int, geometry: tuple[float, np.ndarray], checked: bool
                      ) -> tuple[_Plane, list[ValidityCheck], _Plane | None]:
    pattern = setup.emitter.pattern
    assert pattern is not None
    z, rotation = geometry
    lam = setup.wavelength_nm * NM_TO_MM
    result = angular_spectrum(u_field, pattern.grid.px, pattern.grid.py, lam, setup.n_source, z, padding)
    plane = _asm_plane(setup, result, rotation)
    if not checked:
        return plane, [], None
    cell = pattern.grid.cell_area
    kept = float(np.sum(np.abs(result.field) ** 2)) * cell / (float(np.sum(np.abs(u_field) ** 2)) * cell)
    reference = _asm_plane(setup, reference_angular_spectrum(u_field, pattern.grid.px, pattern.grid.py, lam, setup.n_source, z, padding),
                           rotation)
    checks = [ValidityCheck(parameter="band_power_fraction", minimum=MIN_WINDOW_POWER, observed=_sig(kept), within=kept >= MIN_WINDOW_POWER),
              _window_check(plane, reference),
              ValidityCheck(parameter="propagation_distance", unit="mm", observed=_sig(z), within=True)]
    return plane, checks, reference


def _pairing(matrix: np.ndarray) -> tuple[int, float]:
    """The source axis paired with the plane's u axis (0: x, 1: y) and the largest relative coupling to the other one."""

    identity = abs(matrix[0, 0]) + abs(matrix[0, 2]) + abs(matrix[1, 1]) + abs(matrix[1, 3])
    swapped = abs(matrix[0, 1]) + abs(matrix[0, 3]) + abs(matrix[1, 0]) + abs(matrix[1, 2])
    j = 0 if identity >= swapped else 1
    coupling = 0.0
    for i in (0, 1):
        jj = j if i == 0 else 1 - j
        other = 1 - jj
        for row, scale_floor in ((i, 1.0), (2 + i, ANGLE_ROW_SCALE_FLOOR)):
            paired = max(abs(matrix[row, jj]), abs(matrix[row, 2 + jj]), scale_floor)
            coupling = max(coupling, abs(matrix[row, other]) / paired, abs(matrix[row, 2 + other]) / paired)
    return j, coupling


def _abcd(matrix: np.ndarray, i: int, j: int, n1: float, n2: float) -> tuple[float, float, float, float]:
    """The canonical ``(a, b, c, d)`` of plane axis ``i`` from source axis ``j`` (positions, ``n`` times direction cosines)."""

    return (float(matrix[i, j]), float(matrix[i, 2 + j]) / n1, n2 * float(matrix[2 + i, j]), n2 * float(matrix[2 + i, 2 + j]) / n1)


@dataclass
class _AberrationFit:
    """The rays' aberration as a phase ``phi`` at a plane ``distance`` (mm, in the source's medium) from the source plane:
    ``phi = sum c_ij xi^i eta^j`` over ``xi = (x - center_x) / scale_x``, ``eta = (y - center_y) / scale_y`` (clipped to the
    rays' range), and the bound on what the phase leaves out (``unmodelled``, a relative rms error of the field)."""

    distance: float
    center: tuple[float, float]
    scale: tuple[float, float]
    low: tuple[float, float]
    high: tuple[float, float]
    terms: list[tuple[int, int]]
    coefficients: np.ndarray
    unmodelled: float

    def phase(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """``phi`` on the separable grid ``[y][x]`` (1-D coordinate arrays)."""

        xi = np.clip((x - self.center[0]) / self.scale[0], self.low[0], self.high[0])
        eta = np.clip((y - self.center[1]) / self.scale[1], self.low[1], self.high[1])
        powers_x = np.vstack([xi ** i for i in range(ABERRATION_DEGREE + 1)])
        powers_y = np.vstack([eta ** j for j in range(ABERRATION_DEGREE + 1)])
        table = np.zeros((ABERRATION_DEGREE + 1, ABERRATION_DEGREE + 1))
        for (i, j), c in zip(self.terms, self.coefficients):
            table[j, i] = c
        return powers_y.T @ table @ powers_x


def _aberration_fit(setup: _Setup, matrix: np.ndarray) -> _AberrationFit | None:
    """The aberration beyond first order that the field's phase-space rays (``_phase_space_rays``) show (module docstring).

    Each ray lands at ``(u, v)`` with directions ``(l_u, l_v)``; the first-order matrix predicts them from its launch
    ``(x, y, a, b)``; the difference, carried back to the source plane by the matrix's inverse, is the ray's position and
    direction error ``(dx, dy, da, db)`` there. Free space over ``s`` (in the source's medium) turns it into
    ``(dx + s da, dy + s db, da, db)``: ``s`` is taken where the position errors are smallest (least squares over the rays'
    power), the plane where the aberration is a phase: ``grad(phi) = k0 n (da, db)`` at the rays' first-order points
    ``(x + s a, y + s b)``, fitted by a polynomial of degree ``ABERRATION_DEGREE``. What the phase leaves out, the fit's
    residual gradient ``r`` and the position errors left at ``s``, bounds the field's relative rms error by about
    ``sqrt(sum over x, y of (sigma_s r_rms)^2 + (k0 n sigma_a dx_rms)^2)`` (the Poincare inequality of a Gaussian weight;
    ``sigma_s`` the rays' rms width at ``s``, ``sigma_a`` their rms direction sine). ``None`` without rays on the plane."""

    rays, arrival = setup.phase_rays(), setup.arrival
    if rays.launch is None or rays.sample_index.size < MIN_ABERRATION_RAYS:
        return None
    launch = rays.launch
    uc, vc = arrival.uv
    chief = np.array([uc, vc, float(dot(arrival.direction, arrival.plane_u)), float(dot(arrival.direction, arrival.plane_v))])
    landing = np.column_stack([rays.u, rays.v, rays.du, rays.dv]) - chief[None, :]
    residual = landing - launch @ matrix.T
    back = np.linalg.solve(matrix, residual.T).T  # (dx, dy, da, db) at the source plane
    weights = rays.power_W / float(rays.power_W.sum())
    dx, dy, da, db = back[:, 0], back[:, 1], back[:, 2], back[:, 3]
    turn = float((weights * (da * da + db * db)).sum())
    distance = -float((weights * (dx * da + dy * db)).sum()) / turn if turn > 0.0 else 0.0
    xs, ys = launch[:, 0] + distance * launch[:, 2], launch[:, 1] + distance * launch[:, 3]
    k0n = 2.0 * math.pi / (setup.wavelength_nm * NM_TO_MM) * setup.n_source
    gx, gy = k0n * da, k0n * db
    pattern = setup.emitter.pattern
    assert pattern is not None
    center, scale, low, high = [], [], [], []
    normalized = []
    for coordinate, pitch in ((xs, pattern.grid.px), (ys, pattern.grid.py)):
        mean = float((weights * coordinate).sum())
        spread = max(math.sqrt(float((weights * (coordinate - mean) ** 2).sum())), pitch)
        t = (coordinate - mean) / spread
        center.append(mean), scale.append(spread), low.append(float(t.min())), high.append(float(t.max()))
        normalized.append(t)
    xi, eta = normalized
    terms = [(i, total - i) for total in range(1, ABERRATION_DEGREE + 1) for i in range(total, -1, -1)]
    along_x = np.column_stack([i * xi ** max(i - 1, 0) * eta ** j / scale[0] for i, j in terms])
    along_y = np.column_stack([j * xi ** i * eta ** max(j - 1, 0) / scale[1] for i, j in terms])
    root = np.sqrt(weights)
    coefficients = np.linalg.lstsq(np.vstack([along_x * root[:, None], along_y * root[:, None]]), np.concatenate([gx * root, gy * root]),
                                   rcond=None)[0]
    rx, ry = gx - along_x @ coefficients, gy - along_y @ coefficients
    fitted = scale[0] ** 2 * float((weights * rx * rx).sum()) + scale[1] ** 2 * float((weights * ry * ry).sum())
    a, b = launch[:, 2], launch[:, 3]
    spread_a = float((weights * (a - float((weights * a).sum())) ** 2).sum())
    spread_b = float((weights * (b - float((weights * b).sum())) ** 2).sum())
    left_x, left_y = dx + distance * da, dy + distance * db
    positioned = k0n * k0n * (spread_a * float((weights * left_x * left_x).sum()) + spread_b * float((weights * left_y * left_y).sum()))
    return _AberrationFit(distance, (center[0], center[1]), (scale[0], scale[1]), (low[0], low[1]), (high[0], high[1]), terms,
                          coefficients, math.sqrt(fitted + positioned))


def _aberrated_source(fit: _AberrationFit, u_field: np.ndarray, px: float, py: float, wavelength_mm: float, n: float
                      ) -> tuple[np.ndarray, float]:
    """The source field ``V`` that the first-order transfer turns into the aberrated field (on the source grid; light it
    spreads beyond the grid is left out): free space to the fit's plane, its phase ``exp(i phi)``, free space back, on a grid
    twice as wide. Per axis, the Fresnel transfer function where it is sampled on that grid (``lambda0 |s| <= n N p^2``),
    else the Fourier form ``chirp, FFT`` (positions ``lambda0 s f / n``; its chirp is sampled on the field's grid then);
    without the phase both are exactly the identity. With the largest phase step between neighbouring samples where the
    light is (above ``LIT_SHARE`` of its peak irradiance there)."""

    ny, nx = u_field.shape
    s = fit.distance
    padded = np.zeros((2 * ny, 2 * nx), dtype=np.complex128)
    oy, ox = ny // 2, nx // 2
    padded[oy:oy + ny, ox:ox + nx] = u_field
    coordinates, forward, backward = [], [], []
    for axis, count, pitch in ((1, nx, px), (0, ny, py)):
        size = 2 * count
        x = (np.arange(size, dtype=np.float64) - (count // 2) - 0.5 * (count - 1)) * pitch
        f = np.fft.fftfreq(size, pitch)
        if wavelength_mm * abs(s) <= n * size * pitch * pitch:
            transfer = np.exp(-1j * math.pi * wavelength_mm * s * f * f / n)
            forward.append(lambda a, ax=axis, h=transfer: np.fft.ifft(np.fft.fft(a, axis=ax) * _along(h, ax), axis=ax))
            backward.append(lambda a, ax=axis, h=transfer: np.fft.ifft(np.fft.fft(a, axis=ax) * _along(np.conj(h), ax), axis=ax))
            coordinates.append(x)
        else:
            chirp = np.exp(1j * math.pi * n * x * x / (wavelength_mm * s))
            forward.append(lambda a, ax=axis, q=chirp: np.fft.fft(a * _along(q, ax), axis=ax))
            backward.append(lambda a, ax=axis, q=chirp: np.fft.ifft(a, axis=ax) * _along(np.conj(q), ax))
            coordinates.append(wavelength_mm * s * f / n)
    at_plane = forward[1](forward[0](padded))
    phi = fit.phase(coordinates[0], coordinates[1])
    irradiance = np.abs(at_plane) ** 2
    lit = irradiance >= LIT_SHARE * float(irradiance.max())
    step = 0.0
    for axis, coordinate in ((1, coordinates[0]), (0, coordinates[1])):
        order = np.argsort(coordinate, kind="stable")
        ordered = np.take(phi, order, axis=axis)
        on = np.take(lit, order, axis=axis)
        jumps = np.abs(np.diff(ordered, axis=axis))
        pairs = (on[:, :-1] | on[:, 1:]) if axis == 1 else (on[:-1, :] | on[1:, :])
        if pairs.any():
            step = max(step, float(jumps[pairs].max()))
    out = backward[0](backward[1](at_plane * np.exp(1j * phi)))
    return out[oy:oy + ny, ox:ox + nx], step


def _along(values: np.ndarray, axis: int) -> np.ndarray:
    return values[None, :] if axis == 1 else values[:, None]


def _paraxial(setup: _Setup, u_field: np.ndarray, padding: int, checked: bool
              ) -> tuple[_Plane | None, list[ValidityCheck], str | None, _Plane | None, _Plane | None]:
    """The paraxial transfer and (``checked``) its conditions, its reference on a window twice as wide and the field with the
    rays' aberration (``_aberrated_source`` through the same transfer, on the same lattice); the plane is ``None`` with the
    reason of a condition that leaves no field."""

    arrival = setup.arrival
    pattern = setup.emitter.pattern
    assert pattern is not None
    if arrival.transfer is None:
        return None, [], f"{TRANSFER_UNAVAILABLE}: {arrival.transfer_issue or 'no first-order transfer at this plane'}", None, None
    matrix = np.asarray(arrival.transfer, dtype=np.float64)
    j, coupling = _pairing(matrix)
    n1, n2 = setup.n_source, float(arrival.n)
    abcd_u, abcd_v = _abcd(matrix, 0, j, n1, n2), _abcd(matrix, 1, 1 - j, n1, n2)
    lam = setup.wavelength_nm * NM_TO_MM
    field = u_field if j == 0 else u_field.T
    pu, pv = (pattern.grid.px, pattern.grid.py) if j == 0 else (pattern.grid.py, pattern.grid.px)
    checks: list[ValidityCheck] = []
    if checked:
        checks.append(ValidityCheck(parameter="axis_coupling", maximum=COUPLING_TOLERANCE, observed=_sig(coupling),
                                    within=coupling <= COUPLING_TOLERANCE))
        for name, abcd in (("u", abcd_u), ("v", abcd_v)):
            a, b, c, d = abcd
            residual = abs(a * d - b * c - 1.0)
            checks.append(ValidityCheck(parameter=f"symplectic_residual_{name}", maximum=MAX_SYMPLECTIC_RESIDUAL, observed=_sig(residual),
                                        within=residual <= MAX_SYMPLECTIC_RESIDUAL))
        sine = angular_extent_sine(u_field, pattern.grid.px, pattern.grid.py, lam, n1)
        checks.append(ValidityCheck(parameter="angular_spread_sine", maximum=SCALAR_SINE_LIMIT, observed=_sig(sine), within=sine <= SCALAR_SINE_LIMIT))
        # the apertures pass the field: the share of the launched rays that reach the plane along the axis's path
        launched = setup.rays.launched
        passed = setup.rays.sample_index.size / launched if launched else 0.0
        checks.append(ValidityCheck(parameter="aperture_pass_share", minimum=MIN_GEOMETRIC_TRANSMISSION, observed=_sig(passed),
                                    within=passed >= MIN_GEOMETRIC_TRANSMISSION))
        # those rays' power over the source's power times the axis's transmission (which scales the field): 1 when the probe
        # beam that gives that transmission is not cut by an aperture either
        transmission = setup.geometric_W / (setup.power_W * setup.axis_transmission) if setup.axis_transmission > 0.0 else 0.0
        checks.append(ValidityCheck(parameter="geometric_transmission", minimum=MIN_GEOMETRIC_TRANSMISSION, maximum=MAX_GEOMETRIC_TRANSMISSION,
                                    observed=_sig(transmission),
                                    within=MIN_GEOMETRIC_TRANSMISSION <= transmission <= MAX_GEOMETRIC_TRANSMISSION))
        fit = _aberration_fit(setup, matrix)
        checks.append(ValidityCheck(parameter="aberration_unmodelled", maximum=MAX_ABERRATION_UNMODELLED,
                                    observed=_sig(fit.unmodelled) if fit is not None else None,
                                    within=None if fit is None else fit.unmodelled <= MAX_ABERRATION_UNMODELLED))
    if coupling > COUPLING_TOLERANCE:
        return None, checks, f"{COUPLED_TRANSFER}: the transfer to this plane couples its axes (relative coupling {coupling:.3g})", None, None
    result = paraxial_transfer(field, pu, pv, abcd_u, abcd_v, lam, padding)
    if checked:
        for name, count, pitch, abcd in (("u", field.shape[1], pu, abcd_u), ("v", field.shape[0], pv, abcd_v)):
            conditions = paraxial_conditions(count, pitch, abcd, lam, padding)
            best = min(conditions.values())
            checks.append(ValidityCheck(parameter=f"sampling_{name}", maximum=1.0, observed=_sig(best) if math.isfinite(best) else None,
                                        within=best <= 1.0))
    if result is None:
        return None, checks, (f"{GRID_INVALID}: no discretization of the transfer is sampled by this grid and padding (the field "
                              "spreads beyond the window or its chirp is undersampled): use a finer grid or a larger padding"), None, None
    out, ku, kv = result
    scale = math.sqrt(setup.axis_transmission)
    phase = np.exp(2j * math.pi * float(arrival.opl_mm) / lam) if arrival.ray_opl_defined else None
    out = out * scale
    if phase is not None:
        out = out * phase
    uc, vc = arrival.uv
    plane = _ascending(_Plane(out, uc + ku.coordinates, vc + kv.coordinates))
    if not checked:
        return plane, checks, None, None, None
    aberrated = None
    if fit is not None:
        source, step = _aberrated_source(fit, u_field, pattern.grid.px, pattern.grid.py, lam, n1)
        checks.append(ValidityCheck(parameter="aberration_phase_step", unit="rad", maximum=MAX_ABERRATION_PHASE_STEP, observed=_sig(step),
                                    within=step <= MAX_ABERRATION_PHASE_STEP))
        turned = (kv.matrix @ (source if j == 0 else source.T) @ ku.matrix.T) * scale
        if phase is not None:
            turned = turned * phase
        aberrated = _ascending(_Plane(turned, uc + ku.coordinates, vc + kv.coordinates))
    reference = None
    ru = reference_operator(field.shape[1], pu, abcd_u, lam, padding, ku)
    rv = reference_operator(field.shape[0], pv, abcd_v, lam, padding, kv)
    if ru is not None and rv is not None:
        wide = (rv.matrix @ field @ ru.matrix.T) * scale
        if phase is not None:
            wide = wide * phase
        reference = _ascending(_Plane(wide, uc + ru.coordinates, vc + rv.coordinates))
    checks.append(ValidityCheck(parameter="axis_power_transmission", observed=_sig(setup.axis_transmission), within=True))
    checks.append(_window_check(plane, reference))
    checks.append(ValidityCheck(parameter="route_u_chirp" if ku.route == CHIRP else "route_u_factored", observed=_sig(ku.condition), within=True))
    checks.append(ValidityCheck(parameter="route_v_chirp" if kv.route == CHIRP else "route_v_factored", observed=_sig(kv.condition), within=True))
    return plane, checks, None, reference, aberrated


def _failed(checks: list[ValidityCheck]) -> list[str]:
    """The failed conditions of the plane's values (those of the exported field alone are ``_field_failed``)."""

    return [c.parameter for c in checks if c.within is False and c.parameter not in FIELD_ONLY_CHECKS]


def _field_failed(checks: list[ValidityCheck]) -> list[ValidityCheck]:
    """The failed conditions of the exported complex field alone (``FIELD_ONLY_CHECKS``)."""

    return [c for c in checks if c.within is False and c.parameter in FIELD_ONLY_CHECKS]


def _tolerance_check(parameter: str, change: float | None, tolerance: float) -> ValidityCheck:
    """One of the two accuracy checks against the request's ``value_tolerance``. It names the setting where the case is
    outside the limit or the limit is not the default."""

    within = change is not None and change <= tolerance
    named = {} if within and tolerance == DEFAULT_VALUE_TOLERANCE else {"setting": VALUE_TOLERANCE_SETTING}
    return ValidityCheck(parameter=parameter, maximum=tolerance, observed=_sig(change) if change is not None else None, within=within, **named)


_REASONS = {"window_power_fraction": WINDOW_LOSES_POWER, "axis_coupling": COUPLED_TRANSFER, "symplectic_residual_u": NOT_SYMPLECTIC,
            "symplectic_residual_v": NOT_SYMPLECTIC, "angular_spread_sine": NOT_PARAXIAL, "geometric_transmission": APERTURES_CUT_FIELD,
            "sampling_u": GRID_INVALID, "sampling_v": GRID_INVALID, "band_power_fraction": WINDOW_LOSES_POWER,
            "aperture_pass_share": APERTURES_CUT_FIELD, "aberration_field_error": ABERRATED, "window_refinement_change": WINDOW_NOT_CONVERGED,
            "aberration_value_change": ABERRATED, "aberration_unmodelled": ABERRATED, "aberration_phase_step": ABERRATED}


@dataclass
class _Route:
    method: str
    checks: list[ValidityCheck]
    failed: list[str]
    reason: str | None
    reference: _Plane | None = None
    # the plane's values of realization 0 and the reference's at the same points (``_values``)
    values: tuple[dict[str, float | None], dict[str, float | None] | None] | None = None
    #: the request's value_tolerance (the accuracy checks' limit)
    tolerance: float = DEFAULT_VALUE_TOLERANCE

    @property
    def model(self) -> ModelBasis:
        return _model(self.method, self.checks, self.tolerance)


def _refine(plane: _Plane | None, checks: list[ValidityCheck], reference: _Plane | None, parsed: FieldRequestV1 | None
            ) -> tuple[dict[str, float | None], dict[str, float | None] | None] | None:
    """The window refinement's condition (``window_refinement_change``, appended to ``checks``): the plane's values of the
    region against the reference's at the same points; ``None`` when there is no plane or the region holds no point of it."""

    if plane is None or parsed is None or _failed(checks):
        return None
    mask = _region_mask(plane, _bounds(plane, parsed))
    if not mask.any():
        return None
    enough = _enough(mask)
    coarse = _values(np.abs(plane.field) ** 2, plane, mask, _cell(plane), enough)
    wide = _on_lattice(reference, plane)
    fine = None if wide is None else _values(np.abs(wide) ** 2, plane, mask, _cell(plane), enough)
    checks.append(_tolerance_check("window_refinement_change", _change(coarse, fine), parsed.value_tolerance))
    return coarse, fine


def _aberration_change(plane: _Plane | None, checks: list[ValidityCheck], aberrated: _Plane | None, parsed: FieldRequestV1 | None) -> None:
    """The aberration's conditions over the region (appended to ``checks``): the plane's values with the rays' aberration
    against without (``aberration_value_change``, ``_change``), and the complex field's relative rms difference up to a
    constant phase (``aberration_field_error``; a condition with ``include_field``)."""

    if plane is None or parsed is None or aberrated is None or _failed(checks):
        return
    mask = _region_mask(plane, _bounds(plane, parsed))
    if not mask.any():
        return
    enough = _enough(mask)
    parts = _change_parts(_values(np.abs(plane.field) ** 2, plane, mask, _cell(plane), enough),
                          _values(np.abs(aberrated.field) ** 2, plane, mask, _cell(plane), enough))
    change = None if parts is None else max(ABERRATION_PROFILE_MARGIN * parts[0], parts[1])
    checks.append(_tolerance_check("aberration_value_change", change, parsed.value_tolerance))
    first, second = plane.field[mask], aberrated.field[mask]
    norm = float(np.vdot(first, first).real)
    overlap = abs(complex(np.vdot(first, second)))
    error = math.sqrt(max(norm + float(np.vdot(second, second).real) - 2.0 * overlap, 0.0) / norm) if norm > 0.0 else math.inf
    checks.append(ValidityCheck(parameter="aberration_field_error", maximum=MAX_ABERRATION_FIELD_ERROR if parsed.include_field else None,
                                observed=_sig(error) if math.isfinite(error) else None,
                                within=error <= MAX_ABERRATION_FIELD_ERROR if parsed.include_field else True))


def _propagate(setup: _Setup, u_field: np.ndarray, padding: int, method: str, parsed: FieldRequestV1 | None = None,
               checked: bool = True) -> tuple[_Plane | None, _Route]:
    """The field on the plane by the request's method (module docstring); ``checked``: with the model's conditions, the
    reference on a window twice as wide and the window refinement over ``parsed``'s region (realization 0)."""

    geometry = _free_space(setup)
    tolerance = DEFAULT_VALUE_TOLERANCE if parsed is None else parsed.value_tolerance
    if method in ("AUTO", ANGULAR_SPECTRUM) and geometry is not None:
        plane, checks, reference = _angular_spectrum(setup, u_field, padding, geometry, checked)
        values = _refine(plane, checks, reference, parsed) if checked else None
        failed = _failed(checks)
        if not failed or method == ANGULAR_SPECTRUM:
            return plane, _Route(ANGULAR_SPECTRUM, checks, failed, None, reference, values, tolerance)
    if method == ANGULAR_SPECTRUM:
        free = setup.arrival.medium == setup.emitter.medium and all(s.startswith("observation/") for s in setup.arrival.lineage)
        reason = (f"{PLANE_NOT_PARALLEL}: the plane is not parallel to the source plane with its axes along the source's" if free else
                  f"{NOT_FREE_SPACE}: elements or another medium lie between the source and the plane: use PARAXIAL_TRANSFER")
        return None, _Route(ANGULAR_SPECTRUM, [], [reason.split(":")[0]], reason)
    plane, checks, reason, reference, aberrated = _paraxial(setup, u_field, padding, checked)
    if checked and reason is None:
        _aberration_change(plane, checks, aberrated, parsed)
    values = _refine(plane, checks, reference, parsed) if checked and reason is None else None
    return plane, _Route(PARAXIAL_TRANSFER, checks, _failed(checks), reason, reference, values, tolerance)


def _percent(share: float) -> str:
    return f"{PERCENT * share:g} %"


def _model(method: str, checks: list[ValidityCheck], tolerance: float = DEFAULT_VALUE_TOLERANCE) -> ModelBasis:
    if method == ANGULAR_SPECTRUM:
        return ModelBasis(model_id=ASM_MODEL_ID, name="scalar angular-spectrum propagation in free space", version=FIELD_ANALYSIS_VERSION,
                          assumptions=("scalar field (the polarization is carried unchanged), exact scalar transfer function over the axial "
                                       "distance with the evanescent waves removed, band-limited against the window's folding",
                                       "the field is the band-limited field of its grid samples, zero beyond the grid",
                                       "the light stays in the output window: the same field on a window twice as wide (twice the padding) "
                                       f"holds at least 0.99 of its power inside it and changes the plane's values by at most {_percent(tolerance)}"),
                          validity=tuple(checks))
    return ModelBasis(model_id=PARAXIAL_MODEL_ID, name="Collins integral of the first-order transfer", version=FIELD_ANALYSIS_VERSION,
                      assumptions=("scalar paraxial field about the source axis, through the first-order (real-ray finite-difference) "
                                   "ray-transfer matrix of the axis to the plane: no aberration beyond first order (the field's rays give the "
                                   f"aberration as a phase; applied, it changes the plane's profile values by at most {_percent(tolerance / ABERRATION_PROFILE_MARGIN)} and its centroid by at most {_percent(tolerance)} "
                                   "of its rms width, aberration_value_change)",
                                   "no aperture: the elements' clear apertures are checked against the field's rays, not applied, and "
                                   "their diffraction is not modelled",
                                   "the light stays in the output window: the same transfer on a window twice as wide holds at least 0.99 "
                                   f"of its power inside it and changes the plane's values by at most {_percent(tolerance)}",
                                   "the elements' power transmission along the axis (interface reflections, coatings, absorption: a probe beam "
                                   "launched along it) scales the whole field; the polarization is carried as the source's, not changed",
                                   "the field is the band-limited field of its grid samples, zero beyond the grid"),
                      validity=tuple(checks))


# ------------------------------------------------------------------------------------------------- statistics


def _bounds(plane: _Plane, parsed: FieldRequestV1) -> tuple[float, float, float, float]:
    """The region ``(u_min, u_max, v_min, v_max)``: the request's, or the output window of the requested padding (which the
    refinement's larger window keeps as its region)."""

    if parsed.region is None:
        return float(plane.u[0]), float(plane.u[-1]), float(plane.v[0]), float(plane.v[-1])
    (cu, cv), (hu, hv) = parsed.region.center_uv_mm, parsed.region.half_width_mm
    return cu - hu, cu + hu, cv - hv, cv + hv


def _region_mask(plane: _Plane, bounds: tuple[float, float, float, float]) -> np.ndarray:
    u0, u1, v0, v1 = bounds
    return ((plane.v >= v0) & (plane.v <= v1))[:, None] & ((plane.u >= u0) & (plane.u <= u1))[None, :]


def _cell(plane: _Plane) -> float:
    du = float(plane.u[1] - plane.u[0]) if plane.u.size > 1 else 0.0
    dv = float(plane.v[1] - plane.v[0]) if plane.v.size > 1 else 0.0
    return du * dv


@dataclass
class _Stats:
    window_W: float
    region_W: float
    mean: float
    contrast: float


def _stats(irradiance: np.ndarray, mask: np.ndarray, cell: float) -> _Stats:
    values = irradiance[mask]
    mean = float(values.mean())
    contrast = float(values.std()) / mean if mean > 0.0 else math.nan
    return _Stats(float(irradiance.sum()) * cell, float(values.sum()) * cell, mean, contrast)


def _enough(mask: np.ndarray) -> dict[str, bool]:
    """Whether the region holds ``MIN_STATISTICS_POINTS`` grid points along u and along v."""

    return {"u": int(mask.any(axis=0).sum()) >= MIN_STATISTICS_POINTS, "v": int(mask.any(axis=1).sum()) >= MIN_STATISTICS_POINTS}


def _on_lattice(reference: _Plane | None, plane: _Plane) -> np.ndarray | None:
    """The reference's field at ``plane``'s points (the reference's lattice holds them), or ``None``."""

    if reference is None:
        return None
    picks = []
    for coarse, fine in ((plane.u, reference.u), (plane.v, reference.v)):
        pitch = float(coarse[1] - coarse[0]) if coarse.size > 1 else float(fine[1] - fine[0]) if fine.size > 1 else 1.0
        start = int(np.argmin(np.abs(fine - coarse[0])))
        if start + coarse.size > fine.size or np.max(np.abs(fine[start:start + coarse.size] - coarse)) > LATTICE_TOLERANCE * abs(pitch):
            return None
        picks.append(slice(start, start + coarse.size))
    return reference.field[picks[1], picks[0]]


def _values(irradiance: np.ndarray, plane: _Plane, mask: np.ndarray, cell: float, enough: dict[str, bool]) -> dict[str, float | None]:
    """The plane's values of one irradiance over the region (the window refinement compares them)."""

    stats = _stats(irradiance, mask, cell)
    weights = irradiance * mask
    total = float(weights.sum())
    values: dict[str, float | None] = {"window": stats.window_W, "region": stats.region_W, "mean": stats.mean,
                                       "peak": float(irradiance[mask].max()), "centroid_u": None, "centroid_v": None, "rms_u": None,
                                       "rms_v": None, "contrast": stats.contrast if enough["u"] and enough["v"] else None}
    if total > 0.0:
        cu = float((weights.sum(axis=0) * plane.u).sum()) / total
        cv = float((weights.sum(axis=1) * plane.v).sum()) / total
        values["centroid_u"], values["centroid_v"] = cu, cv
        values["rms_u"] = math.sqrt(float((weights.sum(axis=0) * (plane.u - cu) ** 2).sum()) / total)
        values["rms_v"] = math.sqrt(float((weights.sum(axis=1) * (plane.v - cv) ** 2).sum()) / total)
    return values


def _change(coarse: dict[str, float | None], fine: dict[str, float | None] | None) -> float | None:
    """The largest change of the plane's values on the wider window: relative for the powers, irradiances, rms widths and
    contrast, relative to the rms width for the centroid along that axis (``None`` without the reference)."""

    parts = _change_parts(coarse, fine)
    return None if parts is None else max(parts)


def _change_parts(coarse: dict[str, float | None], fine: dict[str, float | None] | None) -> tuple[float, float] | None:
    """``_change`` as ``(the profile's: powers, irradiances, rms widths and contrast; the centroid's)``."""

    if fine is None:
        return None
    worst = shift = 0.0
    for name in ("window", "region", "mean", "peak", "rms_u", "rms_v", "contrast"):
        a, b = coarse.get(name), fine.get(name)
        if a is None or b is None:
            continue
        floor = CONTRAST_CHANGE_FLOOR if name == "contrast" else 0.0
        scale = max(abs(b), floor)
        if scale > 0.0:
            worst = max(worst, abs(a - b) / scale)
    for axis in ("u", "v"):
        a, b, width = coarse.get(f"centroid_{axis}"), fine.get(f"centroid_{axis}"), fine.get(f"rms_{axis}")
        if a is not None and b is not None and width:
            shift = max(shift, abs(a - b) / width)
    return worst, shift


#: The contrast message's note on speckle that is not fully developed.
_SPECKLE_NOTE = ("; fully developed speckle (contrast 1) needs a screen that scatters nearly all the light (arrays.speckle_fully_developed, "
                 "screen_phase_rms_rad, unscattered_power_share); a weaker screen's pattern holds the unscattered beam and its "
                 "diffraction by the illumination's edge, whose ripple also counts in this contrast")


def _speckle_screens(pattern, count: int, add) -> bool:  # type: ignore[no-untyped-def]
    """The screens' statistics over the illumination (``|U|^2``-weighted), mean over the realizations used: their rms phase
    ``sigma`` and the unscattered share ``|<exp(i phi)>|^2`` (``exp(-sigma^2)`` for a Gaussian screen over many correlation
    cells; over few cells it keeps a residual of about the inverse of their number). Returns whether the pattern is fully
    developed: ``exp(-sigma^2)`` at most ``MAX_UNSCATTERED_SHARE``."""

    weights = pattern.weights / float(pattern.weights.sum())
    rms, unscattered = [], []
    for m in range(count):
        phase = realization_phase(pattern, m)
        mean = float((weights * phase).sum())
        rms.append(math.sqrt(float((weights * (phase - mean) ** 2).sum())))
        unscattered.append(abs(complex((weights * np.exp(1j * phase)).sum())) ** 2)
    sigma, share = math.fsum(rms) / count, math.fsum(unscattered) / count
    stated = pattern.speckle.phase_rms_rad
    add("screen_phase_rms_rad", "rms phase of the screens over the illumination", "rad", sigma, STATISTICS_MODEL_ID,
        f"irradiance-weighted, mean over the realizations (stated: {_sig(stated)} rad)")
    add("unscattered_power_share", "share of the power the screens leave unscattered", None, share, STATISTICS_MODEL_ID,
        f"|<exp(i phi)>|^2 over the illumination, mean over the realizations (exp(-sigma^2) = {_sig(math.exp(-sigma * sigma))} for the "
        f"screens' rms, {_sig(math.exp(-stated * stated))} for the stated one); the speckle is fully developed when exp(-sigma^2) is at "
        f"most {MAX_UNSCATTERED_SHARE} (arrays.speckle_fully_developed)")
    return math.exp(-sigma * sigma) <= MAX_UNSCATTERED_SHARE


def _binned(irradiance: np.ndarray, u: np.ndarray, v: np.ndarray, points: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Block means of at most ``points`` blocks per axis (each block's power is the grid's) and the blocks' centres."""

    fv, fu = max(1, math.ceil(irradiance.shape[0] / points)), max(1, math.ceil(irradiance.shape[1] / points))
    nv, nu = irradiance.shape[0] // fv, irradiance.shape[1] // fu
    block = irradiance[:nv * fv, :nu * fu].reshape(nv, fv, nu, fu).mean(axis=(1, 3))
    return block, u[:nu * fu].reshape(nu, fu).mean(axis=1), v[:nv * fv].reshape(nv, fv).mean(axis=1)


def _rounded(values: np.ndarray) -> list[Any]:
    return [[_sig(float(x)) for x in row] for row in values] if values.ndim == 2 else [_sig(float(x)) for x in values]


def _line_scope(results: Any, monitor: str, source_id: str, wavelength_nm: float) -> list[dict[str, Any]]:
    """The run's model-scope findings that reach the field's line (``source_id``, spectral index 0: a field has one line) on
    ``monitor`` (its ``affected_sampled_lines``; a finding of an earlier Engine, which lists no sampled line, by the plane and
    the line's wavelength), each as ``{code, instance_id, touches, validity}`` (``handoff.model_scope``). Empty where the
    model scope was not computed (a bare Engine result)."""

    from .handoff import model_scope

    return model_scope(results._scope_findings() or (), monitor, set(), {(monitor, source_id, 0)}, wavelength_nm=wavelength_nm)


def field(results: Any, monitor: str, request: Any = None) -> AnalysisResultV1:
    """The field analysis of the run ``results`` at the observation plane ``monitor`` (module docstring)."""

    parsed = parse_request(request)
    if results.request is None:
        raise ResultsError("REQUEST_NOT_AVAILABLE", "this run carries no Engine request (engine_request.json): its source cannot be built again")
    results._observation(monitor)  # (an unknown plane is refused)
    # (the settings name the accuracy only when the request states one other than the default: the settings of a request that
    # does not, and their hash, do not depend on the option existing)
    stated = parsed.model_dump(mode="json")
    if parsed.value_tolerance == DEFAULT_VALUE_TOLERANCE:
        stated.pop(VALUE_TOLERANCE_SETTING)
    settings = {"request": stated, "monitor": monitor}
    identity = results.identity()

    #: (the model-scope findings that reach the field's own line on the plane: ``_line_scope``)
    line_scope: list[dict[str, Any]] = []

    def finish(quantities: list[Quantity], models: list[ModelBasis], arrays: dict[str, Any]) -> AnalysisResultV1:
        if line_scope:
            from .handoff import scope_marks
            from .imaging import outside_model

            marks = scope_marks(line_scope)
            quantities = [outside_model(q, marks, geometry=frozenset()) for q in quantities]
            arrays = {**arrays, "model_scope": line_scope}
        return document(FIELD_ANALYSIS, FIELD_ANALYSIS_VERSION, settings, completed(), engine_input_hash=identity.get("engine_input_hash"),
                        run_result_hash=identity.get("result_hash"), models=models, quantities=quantities, arrays=arrays,
                        engine_build_id=results.engine_build_id)

    try:
        setup = _setup(results, monitor, parsed)
    except _OutOfModel as exc:
        q = Quantity(quantity_id="field", name="field on the plane", scope=QuantityScope(observation_id=monitor), state=exc.state,
                     reason=exc.reason, message=exc.message)
        return finish([q], [], {})
    emitter = setup.emitter
    pattern = emitter.pattern
    assert pattern is not None
    source_id = emitter.source_id
    line_scope = _line_scope(results, monitor, source_id, setup.wavelength_nm)
    speckle = pattern.kind == "speckle"
    count = min(parsed.realizations or pattern.realizations, pattern.realizations)
    scope = QuantityScope(observation_id=monitor, source_id=source_id, wavelength_nm=setup.wavelength_nm)
    models: list[ModelBasis] = [GEOMETRIC_MODEL]
    quantities: list[Quantity] = [Quantity(
        quantity_id=f"{source_id}.geometric_power_W", name="power of the source's rays on the plane", unit="W", scope=scope,
        state=QuantityState.KNOWN, value=_sig(setup.geometric_W), basis=GEOMETRIC_MODEL_ID,
        message="the run's geometric trace of the field (its rays add their powers): the light the elements' apertures pass")]

    first = complex_field(pattern, 0, setup.power_W)
    plane, route = _propagate(setup, first, parsed.padding, parsed.method, parsed)
    names = {"contrast": "speckle contrast of one pattern" if speckle else "irradiance contrast"}
    plane_ids = ("power_window_W", "power_region_W", "mean_irradiance_W_per_mm2", "peak_irradiance_W_per_mm2", "centroid_u_mm",
                 "centroid_v_mm", "rms_radius_u_mm", "rms_radius_v_mm", "contrast")

    def out_of_scope(reason: str, code: str) -> AnalysisResultV1:
        models.append(route.model)
        for name in plane_ids:
            quantities.append(Quantity(quantity_id=f"{source_id}.{name}", name=names.get(name, name), scope=scope,
                                       state=QuantityState.OUT_OF_MODEL_SCOPE, basis=route.model.model_id, reason=code, message=reason,
                                       depends_on=tuple(route.failed)))
        return finish(quantities, models, {"method": route.method})

    if plane is None or route.failed:
        reason = route.reason or "; ".join(f"{_REASONS.get(p, p)}: {p} outside its range" for p in route.failed)
        return out_of_scope(reason, reason.split(":")[0] if route.reason else _REASONS.get(route.failed[0], route.failed[0]))

    bounds = _bounds(plane, parsed)
    mask = _region_mask(plane, bounds)
    if not mask.any():
        raise FieldError(REGION_EMPTY, "the region holds no grid point of the output window "
                                       f"(u {plane.u[0]:.6g} .. {plane.u[-1]:.6g}, v {plane.v[0]:.6g} .. {plane.v[-1]:.6g} mm)")
    cell = _cell(plane)
    points_u, points_v = int(mask.any(axis=0).sum()), int(mask.any(axis=1).sum())
    enough = _enough(mask)
    total = np.abs(plane.field) ** 2
    wide = _on_lattice(route.reference, plane)
    fine_values = route.values[1] if route.values is not None else None
    models.append(route.model)
    stats = [_stats(total, mask, cell)]
    for m in range(1, count):
        other, _route = _propagate(setup, complex_field(pattern, m, setup.power_W), parsed.padding, route.method, checked=False)
        assert other is not None
        irradiance = np.abs(other.field) ** 2
        stats.append(_stats(irradiance, mask, cell))
        total = total + irradiance
    mean_map = total / count
    averaged = _stats(mean_map, mask, cell)
    refined = None if wide is None else _stats(np.abs(wide) ** 2, mask, cell)
    refine = parsed.refinement and fine_values is not None and count == 1

    def mean_of(values: list[float]) -> float:
        return math.fsum(values) / len(values)

    def converge(name: str, full: float, values: list[float]) -> Convergence | None:
        half = len(values) // 2
        if half >= 1:
            coarse = mean_of(values[:half])
            return Convergence(method=ConvergenceMethod.HALF_SAMPLE, parameter="realizations", coarse=_sig(coarse), fine=_sig(full),
                               difference=_sig(abs(full - coarse)))
        if parsed.refinement and refined is not None:
            fine = getattr(refined, name)
            return Convergence(method=ConvergenceMethod.REFINEMENT, parameter="padding", coarse=_sig(full), fine=_sig(fine),
                               difference=_sig(abs(fine - full)))
        return None

    def refinement(name: str, value: float) -> Convergence | None:
        if not refine or fine_values is None or fine_values.get(name) is None:
            return None
        fine = fine_values[name]
        return Convergence(method=ConvergenceMethod.REFINEMENT, parameter="padding", coarse=_sig(value), fine=_sig(fine),
                           difference=_sig(abs(fine - value)))

    # (a request that states its own accuracy: each plane value is the model's within it, and says so with the two checks)
    asked = [c for c in route.checks if c.setting == VALUE_TOLERANCE_SETTING]
    accuracy = ValueBasis(kind=ValueKind.MODEL, validity=tuple(asked)) if asked and parsed.value_tolerance != DEFAULT_VALUE_TOLERANCE else None

    def add(name: str, label: str, unit: str | None, value: float, basis: str, message: str, convergence: Convergence | None = None,
            interval: Interval | None = None) -> None:
        quantities.append(Quantity(quantity_id=f"{source_id}.{name}", name=label, unit=unit, scope=scope, state=QuantityState.KNOWN,
                                   value=_sig(value), basis=basis, convergence=convergence, interval=interval, message=message,
                                   value_basis=accuracy if name in plane_ids else None))

    def unavailable(name: str, label: str, axes: str) -> None:
        quantities.append(Quantity(quantity_id=f"{source_id}.{name}", name=label, scope=scope, state=QuantityState.NOT_APPLICABLE,
                                   reason=TOO_FEW_POINTS, message=f"the region holds {points_u} x {points_v} grid points (u x v); this value "
                                   f"needs at least {MIN_STATISTICS_POINTS} along {axes}: state a larger region or a finer grid"))

    basis = route.model.model_id
    window = [s.window_W for s in stats]
    region = [s.region_W for s in stats]
    means = [s.mean for s in stats]
    contrasts = [s.contrast for s in stats]
    add("power_window_W", "power in the output window", "W", mean_of(window), basis,
        f"of {_sig(setup.power_W)} W at the source; the window's share is a condition of the model", converge("window_W", mean_of(window), window))
    add("power_region_W", "power in the region", "W", mean_of(region), basis, "mean over the realizations", converge("region_W", mean_of(region), region))
    add("mean_irradiance_W_per_mm2", "mean irradiance over the region", "W/mm^2", mean_of(means), basis, "mean over the realizations",
        converge("mean", mean_of(means), means))
    peak = float(mean_map[mask].max())
    add("peak_irradiance_W_per_mm2", "peak irradiance in the region" + (" of the averaged patterns" if count > 1 else ""), "W/mm^2", peak, basis,
        "the largest grid value of the region (the grid's sampling of the pattern)", refinement("peak", peak))
    weights = mean_map * mask
    centroid_u = float((weights.sum(axis=0) * plane.u).sum()) / float(weights.sum())
    centroid_v = float((weights.sum(axis=1) * plane.v).sum()) / float(weights.sum())
    add("centroid_u_mm", "irradiance centroid along u over the region", "mm", centroid_u, basis, "of the averaged irradiance",
        refinement("centroid_u", centroid_u))
    add("centroid_v_mm", "irradiance centroid along v over the region", "mm", centroid_v, basis, "of the averaged irradiance",
        refinement("centroid_v", centroid_v))
    for axis, coordinate, centroid, along in (("u", plane.u, centroid_u, 0), ("v", plane.v, centroid_v, 1)):
        label = f"rms width along {axis} over the region"
        if not enough[axis]:
            unavailable(f"rms_radius_{axis}_mm", label, axis)
            continue
        rms = math.sqrt(float((weights.sum(axis=along) * (coordinate - centroid) ** 2).sum()) / float(weights.sum()))
        add(f"rms_radius_{axis}_mm", label, "mm", rms, basis,
            "second moment of the averaged irradiance about its centroid (a Gaussian beam's 1/e^2 radius is twice it)", refinement(f"rms_{axis}", rms))
    statistics = enough["u"] and enough["v"]
    if statistics:
        spread = None
        if count >= 2:
            sd = float(np.std(contrasts, ddof=1)) / math.sqrt(count)
            spread = Interval(lower=_sig(mean_of(contrasts) - sd), upper=_sig(mean_of(contrasts) + sd), kind=IntervalKind.SAMPLING)
        add("contrast", names["contrast"], None, mean_of(contrasts), STATISTICS_MODEL_ID,
            "std / mean of the irradiance over the region's grid points, mean over the realizations (the interval: one standard error)"
            + (_SPECKLE_NOTE if speckle else ""), converge("contrast", mean_of(contrasts), contrasts), spread)
    else:
        unavailable("contrast", names["contrast"], "u and v")
    models.append(STATISTICS_MODEL)
    if count >= 2 and statistics:
        add("contrast_averaged", f"contrast of the average of {count} mutually incoherent patterns", None, averaged.contrast, STATISTICS_MODEL_ID,
            "std / mean of the realizations' mean irradiance over the region")
        add("contrast_averaged_independent", f"contrast of the average of {count} independent patterns", None,
            mean_of(contrasts) / math.sqrt(count), STATISTICS_MODEL_ID,
            "mean(C_m) / sqrt(M): the averaged contrast if the realizations' patterns are independent and identically distributed")
    developed = None
    if speckle:
        developed = _speckle_screens(pattern, count, add)
    rows, u_axis, v_axis = _binned(np.where(mask, mean_map, 0.0), plane.u, plane.v, parsed.map_points)
    arrays: dict[str, Any] = {
        "method": route.method, "realizations": count, "grid": {"u_points": int(plane.u.size), "v_points": int(plane.v.size),
                                                                 "u_pitch_mm": _sig(float(plane.u[1] - plane.u[0])) if plane.u.size > 1 else None,
                                                                 "v_pitch_mm": _sig(float(plane.v[1] - plane.v[0])) if plane.v.size > 1 else None},
        "map_u_mm": _rounded(u_axis), "map_v_mm": _rounded(v_axis), "irradiance_mean_W_per_mm2": _rounded(rows),
        "contrast_per_realization": [_sig(c) for c in contrasts], "power_region_per_realization_W": [_sig(p) for p in region],
        "conventions": {"frame": "the plane's (u, v) in mm; maps [v][u], zero outside the region",
                        "map": "block means of the computed grid (each block's power is the grid's)",
                        "field": "U in sqrt(W/mm^2), exp(-i w t); phase relative to the source's stated phase plus the chief ray's optical "
                                 "path (PARAXIAL_TRANSFER) or k z (ANGULAR_SPECTRUM)"}}
    if developed is not None:
        arrays["speckle_fully_developed"] = developed
    withheld = _field_failed(route.checks)
    if parsed.include_field and withheld:
        # (the plane's values stand; the complex field itself differs from the aberrated one by more than its own limit)
        arrays["field_withheld"] = {"check": withheld[0].parameter, "observed": withheld[0].observed, "maximum": withheld[0].maximum}
    elif parsed.include_field:
        rows_v, cols_u = np.nonzero(mask.any(axis=1))[0], np.nonzero(mask.any(axis=0))[0]
        crop = plane.field[rows_v[0]:rows_v[-1] + 1, cols_u[0]:cols_u[-1] + 1]
        if max(crop.shape) > MAX_FIELD_POINTS:
            raise FieldError(FIELD_TOO_LARGE, f"the field over the region is {crop.shape[1]} x {crop.shape[0]} points; include_field returns at "
                                              f"most {MAX_FIELD_POINTS} per axis: state a smaller region")
        arrays["field_u_mm"] = [float(x) for x in plane.u[cols_u[0]:cols_u[-1] + 1]]
        arrays["field_v_mm"] = [float(x) for x in plane.v[rows_v[0]:rows_v[-1] + 1]]
        arrays["field_re"] = crop.real.tolist()
        arrays["field_im"] = crop.imag.tolist()
    return finish(quantities, models, arrays)

def field_npz(doc: AnalysisResultV1) -> bytes:
    """The ``field_re`` / ``field_im`` of a field document as an ``.npz`` archive (a pupil-field ``FIELD`` illumination file, with
    ``u_mm`` / ``v_mm`` and the pitch the pupil grid takes). Where a model-scope finding reaches the field's line, the archive
    also holds them as JSON text (``model_scope_json``, the document's ``arrays["model_scope"]``): a pupil field lit by the file
    carries them on (``pupil_field``)."""

    import json

    from .export import npz_bytes

    arrays = doc.arrays
    if "field_re" not in arrays:
        raise FieldError(FIELD_INVALID, "this field document holds no field: request include_field")
    u, v = np.array(arrays["field_u_mm"]), np.array(arrays["field_v_mm"])
    members = dict(field_re=np.array(arrays["field_re"]), field_im=np.array(arrays["field_im"]), u_mm=u, v_mm=v,
                   pitch_mm=np.array([u[1] - u[0] if u.size > 1 else 0.0, v[1] - v[0] if v.size > 1 else 0.0]))
    if arrays.get("model_scope"):
        members["model_scope_json"] = np.array([json.dumps(arrays["model_scope"], sort_keys=True, allow_nan=False)])
    return npz_bytes(members)


__all__ = ["FIELD_ANALYSIS", "FieldError", "field", "field_npz", "parse_request"]
