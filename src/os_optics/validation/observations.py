"""Turn raw tracer arrivals into public ObservationResult / ArrivalResult records (capture, acceptance, pupil stats), after
checking each arrival's Gaussian beam against its own pupil rays (``check_gaussian_aberration``).

Capture and acceptance are measurements, not requirements: a beam that overfills a plane's active area is a notice
(``INCOMPLETE_ENVELOPE``, with the captured fraction), and so is a beam outside the acceptance cone (``MONITOR_MISSED``: it
is not detected, the plane receives 0 W from it). A notice has status ``NOT_APPLICABLE`` and raises no status: neither the
plane's nor the run's. A requirement for full capture is a target
(``active_area_power_fraction`` / ``capture_margin_mm`` minimum, ``acceptance_margin_deg``), which fails when violated.

A plane no traced beam reaches is graded by the physics alone (``grade_unlit_planes``): UNKNOWN when launched power has an
unknown fate or light the trace does not follow may reach it (``validation/untraced.py``), otherwise it is dark, a measured
0 W (status PASS, ``quantity_state`` ``KNOWN_ZERO``), and its ``OBSERVATION_NOT_REACHED`` issue a notice. What a target
asks of the plane is the target's result.

A plane within ``ray_min_advance_mm`` of a physical surface's chief-ray hit point records the power arriving at that surface
before the interaction (``decisions.SurfaceCoincidenceRecord``): each such arrival says so
(``ArrivalResult.surface_coincidence``), and the plane carries one informational ``SURFACE_COINCIDENT_ARRIVAL`` per surface."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from ..config import NumericalConfig
from ..contracts.basis import Assumed, Omitted, ValidityCheck, ValueBasis
from ..contracts.optics import BeamState
from ..contracts.polarization import JonesVector, PolarizationBasis
from ..contracts.project import Observation
from ..contracts.results import (
    ArrivalResult,
    FirstOrderTransfer,
    FollowedLight,
    GratingOrderStep,
    ObservationPlaneFrame,
    ObservationResult,
    OutsideCrossing,
    PolarizationStateSummary,
    PowerBudget,
    SpectralPower,
    SurfaceCoincidence,
)
from ..errors import ErrorCode, Status, ValidationIssue, make_issue, notice, worst
from ..errors.status import AssumedInput, OmittedPart, ValueKind
from ..math.exact_linalg import cmul, det2
from ..math.exact_trig import atan2, atan2_deg, cis
from ..math.phase import reported_phase
from ..math.vectors import correctly_rounded_sum, dot, matvec
from ..optics.gaussian.truncation import transmission_fraction
from ..optics.rays.sampled import SampledTrace
from ..optics.interference.source_coherence import SourceCoherence
from ..optics.rays.tracer import REFLECTION_EVENTS, SINK_ESCAPED, ArrivalRecord
from ..optics.rays.tracer import ZERO_POWER as ZERO_POWER_TERMINATION
from ..optics.surfaces.geometry import aperture_miss_distance
from ..scene.model import ApertureG, Scene, SurfaceG
from ..units import NM_TO_MM
from ..run_control import checkpoint
from .power import DIFFRACTED_BEAM_ACTION, PARTIAL_CROSSING_ACTION, LineFate, left_out_parts, line_fate
from .sampled_light import sampled_at_plane
from .scattered_light import scattered_at_plane
from .superposition import superpositions
from .untraced import DIFFRACTED_BEAM, PlaneReach, faint_once_sum, faint_power_sum, hazards, return_causes, withheld_by_work

Array = NDArray[np.float64]

# |E.E| / |E|^2 above this is circular light: the ellipse has no major axis.
CIRCULAR_LIGHT_TOL = 1e-9
# A projected major axis shorter than this (unit vector component) lies along the plane normal: no in-plane azimuth.
IN_PLANE_AXIS_MIN = 1e-9
# The sign of a major axis is set by its first component (x, y, z) whose magnitude is within this relative margin of the
# largest: an axis at 45 degrees in the design frame has two components equal up to the rounding of their transport (about
# 1e-16 relative), and taking the larger by its last bits would flip the reported sign with that noise.
MAJOR_AXIS_TIE_REL_TOL = 1e-9
# |det A| below this fraction of |A|^2 / 2 is a singular (focusing / defocused) position map: no image rotation.
SINGULAR_MAP_REL = 1e-6
DEG_PER_HALF_TURN = 180.0
# Angles within this of a wrap point (0 / 180 deg) are snapped onto it, so last-bit noise cannot move a value across the wrap.
ANGLE_WRAP_TOL_DEG = 1e-9


def _wrapped(angle_deg: float, period_deg: float, low_deg: float) -> float:
    """``angle`` in [low, low + period), with values within ``ANGLE_WRAP_TOL_DEG`` of the upper end mapped to ``low``
    and negative zero removed."""
    value = (angle_deg - low_deg) % period_deg + low_deg
    if value > low_deg + period_deg - ANGLE_WRAP_TOL_DEG:
        value = low_deg
    return value + 0.0
# Lineage steps of observation planes (virtual: not part of an arrival's interaction history).
OBSERVE_SUFFIX = ":observe"


def polarization_summary(rec: ArrivalRecord) -> PolarizationStateSummary | None:
    """Stokes parameters and the polarization ellipse of the arrival's Jones vector (``PolarizationStateSummary``)."""
    if rec.polarization_open:  # (a grating's response states powers: the relative phase of its TE and TM parts is open)
        return None
    return jones_summary(complex(rec.jones[0]), complex(rec.jones[1]), rec.e1, rec.e2, rec.plane_u, rec.plane_v)


def jones_summary(ex: complex, ey: complex, e1, e2, plane_u, plane_v) -> PolarizationStateSummary | None:
    """Stokes parameters and the polarization ellipse of the Jones vector ``(ex, ey)`` in the transverse basis ``(e1, e2)``,
    its major axis's azimuth in the plane of axes ``plane_u``, ``plane_v`` (``PolarizationStateSummary``)."""
    # same bits in every runtime: |.|^2 as re^2 + im^2, complex products from separate real products, libm-free angles
    e1, e2 = np.asarray(e1, dtype=float), np.asarray(e2, dtype=float)
    plane_u, plane_v = np.asarray(plane_u, dtype=float), np.asarray(plane_v, dtype=float)
    ix, iy = ex.real * ex.real + ex.imag * ex.imag, ey.real * ey.real + ey.imag * ey.imag
    s0 = ix + iy
    if s0 <= 0.0:
        return None
    cross = complex(ex.real * ey.real + ex.imag * ey.imag, ex.real * ey.imag - ex.imag * ey.real)  # conj(ex) ey
    s1, s2, s3 = (ix - iy) / s0, 2.0 * cross.real / s0, 2.0 * cross.imag / s0
    # 2 chi = atan2(s3, sqrt(s1^2 + s2^2)): well conditioned also next to circular light (asin(s3) is not)
    chi = 0.5 * float(atan2_deg(s3, math.sqrt(s1 * s1 + s2 * s2)))
    field = cmul(ex, e1) + cmul(ey, e2)
    squares = cmul(field, field)
    self_dot = complex(float(squares[0].real) + float(squares[1].real) + float(squares[2].real),
                       float(squares[0].imag) + float(squares[1].imag) + float(squares[2].imag))
    major, azimuth = None, None
    if math.sqrt(self_dot.real * self_dot.real + self_dot.imag * self_dot.imag) > CIRCULAR_LIGHT_TOL * s0:
        theta = -0.5 * float(atan2(self_dot.imag, self_dot.real))
        axis = np.real(cmul(field, cis(theta)))
        axis = axis / math.sqrt(float(dot(axis, axis)))
        # an axis has no sign: report the one whose leading component -- the first (x, y, z) within MAJOR_AXIS_TIE_REL_TOL of
        # the largest magnitude -- is positive (independent of last-bit noise up to that resolution)
        mags = np.abs(axis)
        top = float(np.max(mags))
        lead = next(i for i in range(3) if float(mags[i]) >= top * (1.0 - MAJOR_AXIS_TIE_REL_TOL))
        if axis[lead] < 0.0:
            axis = -axis
        major = (float(axis[0]) + 0.0, float(axis[1]) + 0.0, float(axis[2]) + 0.0)
        pu, pv = float(dot(axis, plane_u)), float(dot(axis, plane_v))
        if math.sqrt(pu * pu + pv * pv) > IN_PLANE_AXIS_MIN:
            azimuth = _wrapped(float(atan2_deg(pv, pu)), DEG_PER_HALF_TURN, 0.0)
    handedness = "LINEAR" if abs(s3) <= CIRCULAR_LIGHT_TOL else ("RIGHT" if s3 < 0.0 else "LEFT")
    return PolarizationStateSummary(s1=s1, s2=s2, s3=s3, ellipticity_deg=chi, handedness=handedness, major_axis=major,
                                    azimuth_in_plane_deg=azimuth)


def first_order_transfer(rec: ArrivalRecord) -> FirstOrderTransfer | None:
    """The arrival's ``FirstOrderTransfer`` (``None`` without a transfer matrix: ``rec.transfer_issue`` says why)."""
    if rec.transfer is None:
        return None
    t = rec.transfer
    reflections = sum(1 for step in rec.lineage if step.rsplit(":", 1)[-1] in REFLECTION_EVENTS)
    preserved = reflections % 2 == 0
    a = t[:2, :2].copy()
    if float(dot(rec.direction, rec.plane_normal)) < 0.0:
        a[1, :] = -a[1, :]  # view the plane along the propagation: frame (u, -v)
    det = det2(a)
    scale = 0.5 * float(np.sum(a * a))
    rotation, mirror, magnification = None, None, math.sqrt(abs(det))
    regular = scale > 0.0 and abs(det) > SINGULAR_MAP_REL * scale
    if regular and preserved and det > 0.0:
        # in (-180, 180]: an inverted image is +180 whatever the sign of a rounded-off zero entry
        rotation = -_wrapped(-float(atan2_deg(a[1, 0] - a[0, 1], a[0, 0] + a[1, 1])), 2.0 * DEG_PER_HALF_TURN, -DEG_PER_HALF_TURN)
        rotation = rotation + 0.0
    elif regular and not preserved and det < 0.0:
        mirror = _wrapped(0.5 * float(atan2_deg(a[1, 0] + a[0, 1], a[0, 0] - a[1, 1])), DEG_PER_HALF_TURN, 0.0)
    return FirstOrderTransfer(
        matrix=tuple(tuple(float(v) for v in row) for row in t),  # type: ignore[arg-type]
        reflection_count=reflections, handedness_preserved=preserved, image_rotation_deg=rotation, mirror_axis_deg=mirror,
        transverse_magnification=magnification if regular else None,
    )


def plane_frame(surf: SurfaceG, observation: Observation | None) -> ObservationPlaneFrame:
    """The resolved design-frame geometry and semantics of one observation plane."""
    inter = surf.interaction
    vec = lambda v: (float(v[0]), float(v[1]), float(v[2]))  # noqa: E731
    absorbing = inter.absorbing
    extent = "NONE" if not absorbing else ("ACTIVE_AREA" if inter.active_aperture is not None else "UNBOUNDED_PLANE")
    return ObservationPlaneFrame(
        origin_mm=vec(surf.origin), u_axis=vec(surf.u_axis), v_axis=vec(surf.v_axis), normal=vec(surf.normal),
        pose_frame="DEVICE_LOCAL" if observation is not None and observation.device_id is not None else "DESIGN",
        device_id=None if observation is None else observation.device_id,
        aperture=None if observation is None else observation.aperture, absorbing=absorbing, absorbing_extent=extent,
        acceptance_half_angle_deg=inter.acceptance_half_angle_deg, medium=inter.observation_medium or "air",
    )


def orthonormal_basis(d: Array, e1: Array, e2: Array) -> tuple[Array, Array, Array, float]:
    """Gram-Schmidt re-orthonormalization; returns ``(d, e1, e2, max_deviation)``."""
    d = d / math.sqrt(float(dot(d, d)))
    dev = max(abs(float(dot(e1, d))), abs(float(dot(e2, d))), abs(float(dot(e1, e2))), abs(float(dot(e1, e1)) - 1.0), abs(float(dot(e2, e2)) - 1.0))
    e1n = e1 - float(dot(e1, d)) * d
    e1n = e1n / math.sqrt(float(dot(e1n, e1n)))
    e2n = np.cross(d, e1n)
    return d, e1n, e2n, dev


#: The phase of a beam is referred to its Jones x component (along ``basis.e1``) unless the y component's power exceeds it by
#: more than this relative margin: the components of a beam polarized at 45 degrees are equal up to the rounding of their
#: transport (about 1e-15 relative), and an exact comparison would refer its phase to either by its last bits -- a jump of
#: pi where they have opposite signs.
DOMINANT_COMPONENT_REL_TOL = 1e-9


def beam_state(rec: ArrivalRecord, cfg: NumericalConfig, issues: list[ValidationIssue]) -> BeamState:
    d, e1, e2, dev = orthonormal_basis(rec.direction, rec.e1, rec.e2)
    if dev > cfg.basis_orthonormality_tol:
        issues.append(make_issue(ErrorCode.POLARIZATION_BASIS_DRIFT, Status.WARNING, "transported polarization basis drifted from orthonormality; re-orthonormalized",
                                 location=f"observation:{rec.observation_id}", deviation=dev, path_id=rec.path_id))
    jones = rec.jones
    k0 = 2.0 * math.pi / (rec.wavelength_nm * NM_TO_MM)
    j0, j1 = complex(jones[0]), complex(jones[1])
    p0, p1 = j0.real * j0.real + j0.imag * j0.imag, j1.real * j1.real + j1.imag * j1.imag
    dominant = j1 if p1 > p0 * (1.0 + DOMINANT_COMPONENT_REL_TOL) else j0
    phase = None
    # (behind a central obstruction no light is at the chief point; a thin lens, Faraday rotator or grating states no optical path)
    if dominant != 0 and not rec.chief_obstructed and rec.ray_opl_defined:
        wrapped = (k0 * rec.opl_mm + float(atan2(dominant.imag, dominant.real)) + math.pi) % (2.0 * math.pi) - math.pi
        phase = reported_phase(wrapped)  # the digits the optical path length determines (math.phase)
    return BeamState(
        source_id=rec.source_id, path_id=rec.path_id, spectral_index=rec.spectral_index, wavelength_nm=rec.wavelength_nm,
        position_mm=tuple(float(x) for x in rec.position), direction=tuple(float(x) for x in d),  # type: ignore[arg-type]
        power_W=None if rec.power_bound_W is not None else rec.power_W, geometric_length_mm=rec.length_mm,
        optical_path_length_mm=rec.opl_mm if rec.ray_opl_defined else None,
        phase_rad=phase,
        polarization=None if rec.power_bound_W is not None or rec.polarization_open else JonesVector(
            ex=(float(jones[0].real), float(jones[0].imag)), ey=(float(jones[1].real), float(jones[1].imag))),
        basis=PolarizationBasis(direction=tuple(float(x) for x in d), e1=tuple(float(x) for x in e1), e2=tuple(float(x) for x in e2)),  # type: ignore[arg-type]
        gaussian=rec.gstate.to_contract(),
    )


def _capture_margin(rec: ArrivalRecord, ap: ApertureG, cfg: NumericalConfig) -> float | None:
    """Distance (mm) from the 1/e^2 beam ellipse to the active-area edge (negative = the ellipse crosses the edge): along the
    radial direction of a circular area, along both axes of a rectangular one; ``None`` for another shape (its
    ``active_area_power_fraction`` states the capture) or without a valid Gaussian state."""
    g = rec.gstate
    if not g.valid or ap.shape not in ("circle", "rectangle") or ap.holes:
        return None
    u, v = rec.plane_u, rec.plane_v
    wx, wy = g.radius("x"), g.radius("y")
    bxu = np.array([dot(g.bx, u), dot(g.bx, v)])
    byu = np.array([dot(g.by, u), dot(g.by, v)])
    s = np.outer(bxu, bxu) / (wx * wx) + np.outer(byu, byu) / (wy * wy)
    s_inv = np.array([[s[1, 1], -s[0, 1]], [-s[1, 0], s[0, 0]]]) / det2(s)
    cu, cv = rec.uv
    if ap.shape == "circle":
        du = cu - ap.offset_u
        dv = cv - ap.offset_v
        dist = math.sqrt(du * du + dv * dv)
        if dist <= cfg.aperture_edge_tol_mm:
            # The radial direction is geometrically undefined at the centre; use the same deterministic +u
            # convention the exact-zero branch always intended (avoids turning last-bit position noise into a
            # direction-dependent, platform-dependent margin for centred astigmatic beams).
            n = np.array([1.0, 0.0])
        else:
            n = np.array([du, dv]) / dist
        extent = math.sqrt(float(dot(n, matvec(s_inv, n))))  # ellipse half-extent along the radial direction
        return ap.p1 - (dist + extent)
    margins = []
    for axis, half, off, c in ((np.array([1.0, 0.0]), ap.p1, ap.offset_u, cu), (np.array([0.0, 1.0]), ap.p2, ap.offset_v, cv)):
        extent = math.sqrt(float(dot(axis, matvec(s_inv, axis))))
        margins.append(half - (abs(c - off) + extent))
    return min(margins)


def _miss_distance(rec: ArrivalRecord, ap: ApertureG) -> float:
    """Distance from the chief-ray hit to the active-area edge (mm, positive outside)."""
    return aperture_miss_distance(ap, rec.uv[0], rec.uv[1])


@dataclass(frozen=True)
class RaySpread:
    """The worst ratio of one arrival's pupil-ray spread to the Gaussian bound (``gaussian_ray_spread``)."""

    ratio: float
    axis: str  # "x" | "y": the Gaussian beam axis
    at_waist: bool  # False: at the observation plane; True: on the plane of that axis' Gaussian waist
    distance_mm: float  # axial position of that plane from the chief-ray hit point (along the beam)
    geometric_rms_mm: float
    gaussian_bound_mm: float


def gaussian_ray_spread(rec: ArrivalRecord, cfg: NumericalConfig) -> RaySpread | None:
    """Compare the arrival's live pupil rays with its Gaussian beam.

    The pupil rays are launched on the source's Gaussian envelope (ring radii ``rho_k w``, slopes of its wavefront
    curvature). In any aberration-free first-order system a Gaussian beam is exactly the pair of real rays of the
    two-ray (complex-ray) decomposition, ``w^2 = x_A^2 + x_B^2`` per axis, and every geometric pupil ray is ``a_i x_A``
    (``a_i`` its launch offset in units of the source radius), so on every transverse plane its RMS spread is at most
    ``sqrt(m_a) w`` (``m_a``: the rays' launch second moment along that axis). Divided by the rotation-invariant
    ``sqrt(m) w``, ``m = mean((a_x^2 + a_y^2) / 2)``, the aberration-free ratio is <= 1 for a complete or symmetrically
    truncated sample and <= sqrt(2) for any subset of rays or rotation of the axes (``m_a <= 2 m``).
    This returns, per Gaussian axis, the RMS spread of the rays about their centroid divided by that bound, on the
    observation plane and on the plane of that axis' Gaussian waist (the rays extended along straight lines in the
    medium of the plane, as the Gaussian's own waist is), and keeps the largest. ``None``: no valid Gaussian state or
    fewer than two live pupil rays. A ray travelling against the beam gives an infinite ratio.
    """
    g = rec.gstate
    if not g.valid or rec.pupil_points.shape[0] < 2:
        return None
    # (plain ufunc reductions: this runs for every arrival of every run, so numpy call overhead matters)
    count = rec.pupil_points.shape[0]
    d0 = rec.direction / math.sqrt(float(dot(rec.direction, rec.direction)))
    rel = rec.pupil_points - rec.position
    along = dot(rec.pupil_directions, d0)
    if float(along.min()) <= cfg.zero_vector_tol:
        return RaySpread(math.inf, "x", False, 0.0, math.inf, 0.0)
    z = dot(rel, d0)
    second_moment = float((rec.pupil_launch * rec.pupil_launch).sum()) / (2.0 * count)
    # every (axis, plane) case at once, columns: x on the plane, x on its waist plane, y on the plane, y on its waist plane
    distances = np.array((0.0, -g.waist_distance("x"), 0.0, -g.waist_distance("y")))
    rel_x, rel_y = dot(rel, g.bx), dot(rel, g.by)  # (each axis projected once: the x and y columns repeat)
    dir_x, dir_y = dot(rec.pupil_directions, g.bx), dot(rec.pupil_directions, g.by)
    spread = np.stack((rel_x, rel_x, rel_y, rel_y), axis=1) + np.stack((dir_x, dir_x, dir_y, dir_y), axis=1) / along[:, None] * (
        distances - z[:, None])
    spread -= spread.sum(axis=0) / count
    rms_cases = np.sqrt((spread * spread).sum(axis=0) / count)
    worst_spread: RaySpread | None = None
    for column, (axis, at_waist) in enumerate((("x", False), ("x", True), ("y", False), ("y", True))):
        distance = float(distances[column])
        rms = float(rms_cases[column])
        # the Gaussian radius on that plane: the arrival's own, or the waist radius (no state propagation needed)
        bound = math.sqrt(second_moment) * (g.waist_radius(axis) if at_waist else g.radius(axis))
        ratio = rms / bound if bound > 0.0 else math.inf
        if worst_spread is None or ratio > worst_spread.ratio:
            worst_spread = RaySpread(ratio, axis, at_waist, distance, rms, bound)
    return worst_spread


def check_gaussian_aberration(arrivals: list[ArrivalRecord], cfg: NumericalConfig) -> tuple[list[ArrivalRecord], list[ValidationIssue]]:
    """Mark the Gaussian state of every arrival whose pupil rays spread more than ``gaussian_max_ray_spread_ratio``
    times the aberration-free bound (``gaussian_ray_spread``) as out of model scope (``GAUSSIAN_ABERRATION_DOMINATED``):
    the rays of the Engine's own trace show aberrations the paraxial Gaussian beam does not describe, so its metrics at
    that plane (radius, waist, divergence, capture, interference) are withheld. Arrivals are returned in order; the trace
    itself is not changed."""
    checked: list[ArrivalRecord] = []
    issues: list[ValidationIssue] = []
    for rec in arrivals:
        spread = gaussian_ray_spread(rec, cfg)
        if spread is None or spread.ratio <= cfg.gaussian_max_ray_spread_ratio:
            checked.append(rec)
            continue
        where = (f"on the plane of the beam's {spread.axis} waist, {abs(spread.distance_mm):.4g} mm "
                 f"{'before' if spread.distance_mm < 0.0 else 'after'} the observation plane" if spread.at_waist else "on the observation plane")
        text = (f"the Engine's pupil rays spread {spread.ratio:.3g} times as wide as an aberration-free beam with this Gaussian envelope can "
                f"({where}; limit {cfg.gaussian_max_ray_spread_ratio:g}): the beam is aberration dominated and its Gaussian metrics are not valid")
        checked.append(replace(rec, gstate=rec.gstate.invalidated(ErrorCode.GAUSSIAN_ABERRATION_DOMINATED, text)))
        issues.append(make_issue(
            ErrorCode.GAUSSIAN_ABERRATION_DOMINATED, Status.OUT_OF_MODEL_SCOPE, text, location=f"observation:{rec.observation_id}",
            path_id=rec.path_id, ray_spread_ratio=spread.ratio, ray_spread_limit=cfg.gaussian_max_ray_spread_ratio, beam_axis=spread.axis,
            at_waist=spread.at_waist, axial_distance_mm=spread.distance_mm, geometric_rms_mm=spread.geometric_rms_mm,
            gaussian_bound_rms_mm=spread.gaussian_bound_mm,
            suggested_action="read the ray spot (pupil_rms_radius_mm) instead of the Gaussian metrics"))
    return checked, issues


def split_arrivals(
    arrivals: list[ArrivalRecord], scene: Scene, cfg: NumericalConfig, rule: tuple[Array, Array] | None = None
) -> tuple[list[ArrivalRecord], list[ArrivalRecord]]:
    """Separate arrivals that reach a device from beams that merely cross its plane outside the active area.

    The tracer decides it once per crossing (``ArrivalRecord.counted``, the rule an absorbing plane also absorbs by): with
    no active area every crossing counts; with one, a beam counts if its chief ray is inside the area or (valid Gaussian) at
    least ``capture_min_power_fraction`` of its power falls inside it. ``scene`` / ``cfg`` / ``rule`` are not needed."""
    kept = [rec for rec in arrivals if rec.counted]
    rejected = [rec for rec in arrivals if not rec.counted]
    return kept, rejected


def build_observation_results(
    arrivals: list[ArrivalRecord], scene: Scene, observation_ids: list[str], cfg: NumericalConfig,
    rejected: list[ArrivalRecord] | None = None,
    rule: tuple[Array, Array] | None = None,
    observations: tuple[Observation, ...] = (),
    sources: SourceCoherence | None = None,
    sampled: SampledTrace | None = None,
    scattered=None,
) -> tuple[list[ObservationResult], list[ValidationIssue]]:
    """``arrivals`` must already be the *kept* arrivals (see ``split_arrivals``); ``rejected`` only feeds the diagnostics.
    ``sampled``: the rays of the run's extended emitters (``validation/sampled_light.py``), which add to each plane's power;
    ``scattered``: the rays of the light its surfaces scatter (``validation/scattered_light.py``), which add the same way."""
    issues: list[ValidationIssue] = []
    by_id: dict[str, SurfaceG] = {s.interaction.observation_id: s for s in scene.surfaces if s.interaction.kind == "observe"}  # type: ignore[misc]
    results: list[ObservationResult] = []
    declared = {o.observation_id: o for o in observations}
    for oid in observation_ids:
        surf = by_id[oid]
        inter = surf.interaction
        recs = sorted((r for r in arrivals if r.observation_id == oid), key=lambda r: (r.path_id, r.crossing_index, r.spectral_index))
        sampled_lines = sampled_at_plane(sampled, oid) + scattered_at_plane(scattered, oid)
        arr_results: list[ArrivalResult] = []
        statuses: list[Status] = []
        for rec in recs:
            checkpoint()  # (one more arrival: the run's monitor may stop the run here, os_optics/run_control.py)
            state = beam_state(rec, cfg, issues)
            if rec.met_first or rec.left_beyond:
                statuses.append(Status.OUT_OF_MODEL_SCOPE)
                issues.append(_met_first_issue(oid, rec))
            cos_a = abs(float(dot(rec.direction, rec.plane_normal)))
            within, margin_deg = (None, None)
            if inter.acceptance_half_angle_deg is not None:
                within = rec.within_acceptance if rec.within_acceptance is not None else rec.aoi_deg <= inter.acceptance_half_angle_deg
                margin_deg = inter.acceptance_half_angle_deg - rec.aoi_deg
                if not within:
                    # a measurement: the beam is not detected (0 W); a power or acceptance_margin_deg target decides. A
                    # notice: the arrival's own fields carry it (within_acceptance, acceptance_margin_deg)
                    issues.append(notice(
                        ErrorCode.MONITOR_MISSED,
                        "the beam arrives outside the detector acceptance angle: it is not detected (the plane receives 0 W from it)",
                        location=f"observation:{oid}", aoi_deg=rec.aoi_deg, acceptance_half_angle_deg=inter.acceptance_half_angle_deg,
                        path_id=rec.path_id, **({"incident_power_W": rec.power_W} if rec.power_bound_W is None else {"power_upper_bound_W": rec.power_bound_W})))
            frac, margin = None, None
            if inter.active_aperture is not None:
                if rec.gstate.valid:
                    frac = transmission_fraction(rec.gstate, inter.active_aperture, rec.uv, rec.plane_u, rec.plane_v, cos_a, cfg, rule)
                margin = _capture_margin(rec, inter.active_aperture, cfg)
                if margin is not None and margin < 0.0:
                    # partial capture is a measured fraction, not a failure: a pinhole or a bucket may take part of a beam on
                    # purpose. Full capture is required by a target (active_area_power_fraction / capture_margin_mm minimum).
                    # A notice: the arrival's own fields carry it (active_area_power_fraction, capture_margin_mm)
                    captured = "" if frac is None else f": {frac:.4g} of its power falls on it"
                    issues.append(notice(
                        ErrorCode.INCOMPLETE_ENVELOPE, f"the beam is not fully inside the detector active area{captured}",
                        location=f"observation:{oid}", capture_margin_mm=margin, active_area_power_fraction=frac,
                        chief_ray_inside=inside_active(rec, inter.active_aperture, cfg), path_id=rec.path_id,
                        suggested_action="state the requirement as an active_area_power_fraction target"))
            pup = rec.pupil_uv
            centroid, rms = None, None
            if pup.shape[0] > 0:
                c = pup.mean(axis=0)
                centroid = (float(c[0]), float(c[1]))
                rms = float(np.sqrt(np.mean(np.sum((pup - c) ** 2, axis=1))))
            # (a beam that met a coating outside its data: what its powers are, ``Tracer._live_coating``)
            held = held_basis(rec.power_scope, rec.held)
            bases = {} if held is None else {key: held for key, number in (
                ("beam.power_W", state.power_W), ("detected_power_W", rec.detected_power_W)) if number is not None}
            arr_results.append(
                ArrivalResult(
                    value_basis={**bases, **(_samples_basis(rec, rms) or {})}, beam=state, uv_mm=rec.uv, aoi_deg=rec.aoi_deg, within_acceptance=within, acceptance_margin_deg=margin_deg,
                    active_area_power_fraction=frac, capture_margin_mm=margin, pupil_ray_count=rec.pupil_total,
                    pupil_rays_arrived=int(pup.shape[0]), pupil_centroid_uv_mm=centroid, pupil_rms_radius_mm=rms,
                    crossing_index=rec.crossing_index, parent_path_id=rec.parent_path_id, detected_power_W=rec.detected_power_W,
                    detected_power_min_W=rec.detected_power_min_W,
                    lineage=tuple(step for step in rec.lineage if not step.endswith(OBSERVE_SUFFIX)),
                    direction_uvn=(float(dot(np.asarray(state.direction), rec.plane_u)),
                                   float(dot(np.asarray(state.direction), rec.plane_v)), float(dot(np.asarray(state.direction), rec.plane_normal))),
                    polarization_state=polarization_summary(rec), first_order=first_order_transfer(rec),
                    first_order_issue=rec.transfer_issue, ray_opl_defined=rec.ray_opl_defined,
                    relative_path_mm=None if rec.ray_opl_defined else rec.opl_mm,
                    coherence_group=rec.coherence_group, coherence_length_mm=rec.coherence_length_mm,
                    group_path_length_mm=rec.group_opl_mm, group_delay_unknown_at=rec.delays,
                    group_delay_dispersion_mm2=None if rec.group_opl_mm is None else rec.gdd_mm2,
                    chief_ray_obstructed=rec.chief_obstructed, power_out_of_scope_at=rec.power_scope,
                    grating_orders=tuple(GratingOrderStep(surface_id=surface, order=order) for surface, order in rec.grating_steps),
                    power_upper_bound_W=rec.power_bound_W,
                    surface_coincidence=None if rec.coincidence is None else SurfaceCoincidence(
                        surface_id=rec.coincidence.surface_id, role=rec.coincidence.role,  # type: ignore[arg-type]
                        surface_event=rec.coincidence.event,  # type: ignore[arg-type]
                        crossing_power_upper_bound_W=rec.coincidence.crossing_upper_W),
                    polarization_component=rec.polarization_component,
                    power_basis=None if rec.sampling is None else "GEOMETRIC_PUPIL_SAMPLES",
                    power_sampling_resolution_W=None if rec.sampling is None or rec.power_bound_W is not None else rec.sampling * received_power_W(rec),
                    power_taken_before_at=rec.taken_at, power_taken_before_upper_W=rec.taken_upper_W,
                    power_taken_before_lower_W=rec.taken_lower_W,
                    reference_ray_promoted_at=rec.reference_promoted,
                )
            )
            if rec.gstate.status is not Status.PASS:
                statuses.append(rec.gstate.status)
        issues.extend(_coincidence_issues(oid, recs))
        if not recs and not any(line.counted_crossing_count for line in sampled_lines):
            # graded by grade_unlit_planes (dark or unknown); until then the plane and its issue grade nothing
            statuses.append(Status.NOT_APPLICABLE)
            missed = [r for r in (rejected or []) if r.observation_id == oid]
            ap = inter.active_aperture
            ctx: dict = {}
            if missed and ap is not None:
                ctx = {"beams_crossing_outside_active_area": len(missed), "nearest_miss_mm": min(_miss_distance(r, ap) for r in missed)}
            issues.append(make_issue(ErrorCode.OBSERVATION_NOT_REACHED, Status.NOT_APPLICABLE,
                                     "no beam reaches this observation plane's active area" if missed else "no beam reaches this observation plane",
                                     location=f"observation:{oid}",
                                     suggested_action="check the beam path, apertures and element orientations upstream, or recentre the detector"
                                     + ("" if surf.owner_id == "observation" else
                                        f"; this plane is attached to {surf.owner_id!r}, so its pose is in that component's "
                                        "local frame (the identity pose puts it at the component's position), not in the design frame"),
                                     **ctx))
        spectral: dict[float, float] = {}
        # (a wavelength some of whose light's power is out of model scope: the surfaces, ArrivalResult.power_out_of_scope_at)
        spectral_scope: dict[float, set[str]] = {}
        spectral_held: dict[float, list] = {}
        # (a wavelength a surface took a bounded share of before the plane: the surfaces and the most they took, per beam at
        # most what the plane receives from it -- ArrivalResult.power_taken_before_at)
        spectral_taken: dict[float, tuple[set[str], list[float]]] = {}
        for rec in recs:
            if rec.power_bound_W is None:  # (a beam whose power is not known adds nothing known to a wavelength's value)
                spectral[rec.wavelength_nm] = spectral.get(rec.wavelength_nm, 0.0) + received_power_W(rec)
            if rec.power_scope:
                spectral_scope.setdefault(rec.wavelength_nm, set()).update(rec.power_scope)
                spectral_held.setdefault(rec.wavelength_nm, []).extend(item for item in rec.held if item not in spectral_held[rec.wavelength_nm])
            if rec.taken_upper_W is not None and rec.power_bound_W is None:
                at, shares = spectral_taken.setdefault(rec.wavelength_nm, (set(), []))
                at.update(rec.taken_at)
                shares.append(min(rec.taken_upper_W, received_power_W(rec)))
        for line in sampled_lines:
            if line.counted_crossing_count:
                spectral[line.wavelength_nm] = spectral.get(line.wavelength_nm, 0.0) + line.power_W
            if line.power_out_of_scope_at:
                spectral_scope.setdefault(line.wavelength_nm, set()).update(line.power_out_of_scope_at)
        outside = []
        if inter.active_aperture is not None:
            for rec in sorted((r for r in (rejected or []) if r.observation_id == oid), key=lambda r: (r.path_id, r.crossing_index)):
                outside.append(OutsideCrossing(path_id=rec.path_id, crossing_index=rec.crossing_index, uv_mm=rec.uv,
                                               miss_distance_mm=_miss_distance(rec, inter.active_aperture),
                                               power_W=rec.power_W if rec.power_bound_W is None else None, power_upper_bound_W=rec.power_bound_W,
                                               power_out_of_scope_at=(*rec.power_scope, *(at for at in rec.taken_at if at not in rec.power_scope))))
        frame = plane_frame(surf, declared.get(oid))
        sums = superpositions(tuple(arr_results), frame, cfg, sources)
        # A bounded ray's Jones field is only its unattenuated upper envelope;
        # it contributes no *known* watts to the plane's traced subtotal.
        total_W = (float(correctly_rounded_sum(received_power_W(r) for r in recs if r.power_bound_W is None))
                   + math.fsum(line.power_W for line in sampled_lines))
        # (each beam's share taken counts up to what the plane receives from it: nothing outside the acceptance cone)
        taken = [min(r.taken_upper_W, received_power_W(r)) for r in recs
                 if r.power_bound_W is None and r.taken_upper_W is not None]
        resolutions = [a.power_sampling_resolution_W for a in arr_results if a.power_sampling_resolution_W is not None]
        # (a sampled line none of whose rays is counted here: a sampled zero, resolved to one ray's weight)
        resolutions += sampled_zero_weights(sampled_lines)
        results.append(ObservationResult(
            observation_id=oid, status=worst(statuses, empty=Status.PASS), arrivals=tuple(arr_results),
            total_power_W=total_W,
            power_lower_bound_W=(max(total_W - float(correctly_rounded_sum(taken)), 0.0)
                                 if taken or (total_W > 0.0 and any(r.power_bound_W is not None for r in recs)) else None),
            power_sampling_resolution_W=float(correctly_rounded_sum(resolutions)) if resolutions else None,
            message="" if recs or any(line.counted_crossing_count for line in sampled_lines) else "no beam reached this observation plane",
            sampled_light=sampled_lines,
            plane=frame, superpositions=sums, power_basis="INCOHERENT_SUM" if sums else None,
            # (a wavelength out of model scope behind some surface: the held coatings' entry; else a bounded share taken
            # before the plane: that entry)
            power_by_wavelength=tuple(SpectralPower(
                wavelength_nm=w, power_W=p, power_out_of_scope_at=tuple(sorted(spectral_scope.get(w, ()))),
                value_basis=(_line_basis(sorted(spectral_scope[w]), spectral_held.get(w, ())) if w in spectral_scope
                             else _taken_basis(spectral_taken.get(w))))
                for w, p in sorted(spectral.items())),
            outside_active_area=tuple(outside),
        ))
    return results, issues


def held_basis(scope: tuple[str, ...] | list[str], held) -> ValueBasis | None:
    """The entry of a power behind the surfaces ``scope`` (``power_out_of_scope_at``) some of which are there because the
    beam met their coating outside its data and went on with the data's values at the nearest stated point (``held``:
    ``ArrivalRecord.held``). An ``ESTIMATE``: per held surface and parameter one ``HELD_AT_NEAREST_STATED_POINT`` assumption
    with the value the data was read at and one failed validity check with the stated range and the value asked for;
    ``BEAM_PART_ELSEWHERE`` for the other surfaces of ``scope``. ``None`` when no surface is held (the flat field says it)."""
    assumed: list[Assumed] = []
    validity: list[ValidityCheck] = []
    for item in held:
        at = (item.surface_id,)
        for parameter, unit, asked, read_at, minimum, maximum in item.parameters:
            assumed.append(Assumed(input=AssumedInput.HELD_AT_NEAREST_STATED_POINT, at=at, value=read_at, unit=unit, code=item.code))
            validity.append(ValidityCheck(parameter=parameter, unit=unit, minimum=minimum, maximum=maximum, observed=asked,
                                          within=False, at=at, code=item.code))
    if not assumed:
        return None
    held_at = {item.surface_id for item in held}
    rest = tuple(surface for surface in scope if surface not in held_at)
    return ValueBasis(kind=ValueKind.ESTIMATE, assumed=tuple(assumed), validity=tuple(validity),
                      omitted=(Omitted(part=OmittedPart.BEAM_PART_ELSEWHERE, at=rest),) if rest else ())


def _line_basis(scope: list[str], held: list) -> dict[str, ValueBasis] | None:
    """The stored entry of a wavelength's power some of whose beams met a coating outside its data (``held_basis``)."""
    entry = held_basis(scope, held)
    return None if entry is None else {"power_W": entry}


def sampled_zero_weights(lines) -> list[float]:
    """One ray's weight (``emitted_W / sample_count``) per sampled line none of whose rays is counted on the plane: the
    resolution of that line's sampled zero (``SampledLightAtPlane``)."""
    return [line.emitted_W / line.sample_count for line in lines if not line.counted_crossing_count]


def _sampled_zero_text(obs: ObservationResult) -> str:
    """What a plane no counted ray of the sampled lines reaches receives: 0 W, resolved to the rays' weight."""
    lines = [line for line in obs.sampled_light if not line.counted_crossing_count]
    rays = int(math.fsum(line.samples_traced for line in lines))
    weight = max(line.emitted_W / line.sample_count for line in lines)
    return (f"none of the {rays} sampled rays is counted on this observation plane; one ray carries up to {weight:.4g} W: it "
            f"receives 0 W within a sampling resolution of {obs.power_sampling_resolution_W:.4g} W")


def _taken_basis(taken: tuple[set[str], list[float]] | None) -> dict[str, ValueBasis] | None:
    """The entry of a wavelength's power on a plane where surfaces took a bounded share of its beams before the plane: the
    value is the beams' own (their chief rays'), the light there is at most that and at least it less ``upper``; whether
    other light adds is the wavelength's ``complete`` (so the kind states no relation)."""
    if taken is None:
        return None
    at, shares = taken
    return {"power_W": ValueBasis(kind=ValueKind.ESTIMATE, omitted=(Omitted(
        part=OmittedPart.BEAM_PART_ELSEWHERE, at=tuple(sorted(at)), upper=float(correctly_rounded_sum(shares)),
        code=ErrorCode.POWER_TAKEN_BEFORE_PLANE.value),))}


def _samples_basis(rec: ArrivalRecord, rms: float | None) -> dict[str, ValueBasis] | None:
    """The entry of the pupil samples' rms radius behind a surface where they stopped following the beam's diffraction
    (``ArrivalRecord.samples_unresolved``): the radius is the geometric samples', the beam spreads beyond them, so as the
    beam's size it is an estimate; the check states the spread they do not follow and the most they follow there."""
    if rec.samples_unresolved is None or rms is None:
        return None
    surface_id, spread, limit, code = rec.samples_unresolved
    return {"pupil_rms_radius_mm": ValueBasis(kind=ValueKind.ESTIMATE, validity=(ValidityCheck(
        parameter="diffraction_spread_mm", unit="mm", maximum=limit, observed=spread, within=False, at=(surface_id,), code=code),))}


def _met_first_issue(oid: str, rec: ArrivalRecord) -> ValidationIssue:
    """``OUT_OF_MODEL_SCOPE`` at the plane: part of the arriving beam meets physical surfaces before the plane, or a surface
    the leg began at reflected part of it beyond the plane (``ArrivalRecord.met_first`` / ``left_beyond``), without a bound
    of that part -- e.g. a beam whose chief ray passes a hole or beside an element that the rest of the beam meets. The beam's
    power follows its chief ray, so the power recorded here is the whole beam's and the power that reaches the plane is not
    computed (``ArrivalResult.power_out_of_scope_at``; the plane's power is ``OUT_OF_MODEL_SCOPE``)."""
    how = ([f"meets {', '.join(rec.met_first)} before this plane"] if rec.met_first else []) + (
        [f"is reflected by {', '.join(rec.left_beyond)} beyond this plane"] if rec.left_beyond else [])
    return make_issue(
        ErrorCode.OUT_OF_MODEL_SCOPE, Status.OUT_OF_MODEL_SCOPE,
        f"part of beam {rec.path_id} {' or '.join(how)}, while its chief ray does not: the beam's power follows its chief ray, "
        "so the power recorded here is the whole beam's, and the power that reaches the plane is not computed",
        location=f"observation:{oid}", path_id=rec.path_id, surface_ids=sorted({*rec.met_first, *rec.left_beyond}),
        suggested_action="read the power at a plane before that surface")


def _coincidence_issues(oid: str, recs: list[ArrivalRecord]) -> list[ValidationIssue]:
    """One informational ``SURFACE_COINCIDENT_ARRIVAL`` per physical surface whose chief-ray hit point this plane passes
    through: the arrivals there are the power at that surface, not the power of the beam that leaves it."""
    roles: dict[str, set[str]] = {}
    for rec in recs:
        if rec.coincidence is not None:
            roles.setdefault(rec.coincidence.surface_id, set()).add(rec.coincidence.role)
    return [make_issue(ErrorCode.SURFACE_COINCIDENT_ARRIVAL, Status.NOT_APPLICABLE,
                       f"this observation plane passes through the hit point of {surface} (within ray_min_advance_mm): an ARRIVING "
                       "arrival is the power arriving at that surface before the interaction, a RETURNING arrival the beam that "
                       "leaves it back across the plane", location=f"observation:{oid}", surface_id=surface, roles=sorted(roles[surface]))
            for surface in sorted(roles)]


def plane_fate(obs: ObservationResult, lines, cfg: NumericalConfig, reach: PlaneReach | None) -> LineFate:
    """``line_fate`` of ``lines`` for this plane: their ledger, and the light the trace does not follow that may reach the
    plane from them (``validation/untraced.py``): from the lines no traced beam brings to it and, in a run that followed
    every untraced branch, from the lines whose traced beams arrive there too (a ghost beside its own traced beam) -- the
    same light a ``power_W`` target on the plane is graded on (``metrics._fate``)."""
    lines = list(lines)
    keys = [(line.source_id, line.spectral_index) for line in lines]
    return line_fate(lines, cfg, hazards(reach, obs.observation_id, keys), faint_W=faint_power_sum(reach, obs.observation_id, keys),
                     withheld_by_work=withheld_by_work(reach, obs.observation_id, keys),
                     returns=return_causes(reach, obs.observation_id, keys))


def complete_power_by_wavelength(observations: list[ObservationResult], budget: PowerBudget,
                                 cfg: NumericalConfig, reach: PlaneReach | None = None,
                                 followed: dict[str, list[FollowedLight]] | None = None,
                                 unknown_harmonics_nm: tuple[float, ...] = ()) -> list[ObservationResult]:
    """List every launched wavelength in each plane's ``power_by_wavelength`` and state whether the trace accounts for all
    of it (``complete`` / ``untraced_power_W``, per wavelength and for the plane). A wavelength none of whose light
    arrives reads 0.0 when that is a known zero (all its launched power traced to known ends, and none of the light the
    trace does not follow can reach the plane), and ``None`` -- not a number -- when it is not known; the plane's
    ``total_power_W`` likewise when nothing arrives. With some light arriving the value counts the traced light, and
    ``complete`` false says the plane may receive more: ``untraced_power_W`` is the light the trace does not follow that may
    reach it (beside its traced beams too, ``plane_fate``) and ``power_upper_bound_W`` the traced light plus that light and
    the light left out as negligible (``None`` where nothing bounds it) -- with light not followed, the upper end of the
    interval a ``power_W`` target without selectors is graded on (``metrics._untraced_bounds``). Light left out as negligible
    is the stated truncation of a value while the most it brings here stays within the lines' negligible level
    (``LineFate.faint_accounted``): it changes no completeness (``complete``, ``power_complete``) and no value, only that
    bound; above that level it leaves the power incomplete (its possible return). A traced value on the geometric pupil-sample basis
    (``ObservationResult.power_sampling_resolution_W``) is a sampled value whose resolution is no proven bound: no upper
    bound is taken from it. On an absorbing plane the bound is at most the lines' launched power: each watt is absorbed
    once (``capped_by_launch``). A harmonic whose generation is unknown opens its spectral total. The pump ledger
    already carries the unresolved energy. Its share at the harmonic wavelength or at this plane is not established.
    Individual traced arrivals still report their power."""
    by_wavelength: dict[float, list] = {}
    for line in budget.lines:
        by_wavelength.setdefault(line.wavelength_nm, []).append(line)
    out = []
    for obs in observations:
        received = {s.wavelength_nm: s.power_W for s in obs.power_by_wavelength}
        scoped_at = {s.wavelength_nm: s.power_out_of_scope_at for s in obs.power_by_wavelength}
        bases = {s.wavelength_nm: s.value_basis for s in obs.power_by_wavelength}
        spectral = []
        for wavelength, lines in sorted(by_wavelength.items()):
            fate = plane_fate(obs, lines, cfg, reach)
            power = received.get(wavelength)
            if power is None:
                power = 0.0 if fate.complete and math.isfinite(fate.faint_W) else None
            # (light left out as negligible is the stated truncation of the value while it stays within the lines' negligible
            # level here: it changes no completeness; above it, it may add -- LineFate.faint_accounted)
            complete = fate.complete and fate.faint_accounted
            entry = (bases.get(wavelength) or {}).get("power_W") if power is not None else None
            if entry is not None and complete and not scoped_at.get(wavelength):
                # (a bounded share taken before the plane and no other light adds: the value is the upper end)
                entry = entry.model_copy(update={"kind": ValueKind.UPPER_BOUND})
            spectral.append(SpectralPower(wavelength_nm=wavelength, power_W=power, complete=complete,
                                          untraced_power_W=fate.untraced_W, power_out_of_scope_at=scoped_at.get(wavelength, ()),
                                          value_basis=None if entry is None else {"power_W": entry}))
        extra = [s for s in obs.power_by_wavelength if s.wavelength_nm not in by_wavelength]  # (none: arrivals are launched lines)
        if unknown_harmonics_nm:
            by_reported_wavelength = {row.wavelength_nm: row for row in (*spectral, *extra)}
            for harmonic_nm in unknown_harmonics_nm:
                matching = [w for w in by_reported_wavelength if abs(w - harmonic_nm) <= cfg.interference_wavelength_tol_nm]
                for wavelength in matching or [harmonic_nm]:
                    previous = by_reported_wavelength.get(wavelength)
                    by_reported_wavelength[wavelength] = SpectralPower(
                        wavelength_nm=wavelength, power_W=None, complete=False, untraced_power_W=None,
                        power_out_of_scope_at=previous.power_out_of_scope_at if previous is not None else ())
            spectral = list(by_reported_wavelength.values())
            extra = []
        overall = plane_fate(obs, budget.lines, cfg, reach)
        # (dark with every launched watt at a known end: 0 W, unless nothing bounds the light left out as negligible)
        total = obs.total_power_W if obs.power_lit or (overall.complete and math.isfinite(overall.faint_W)) else None
        upper = None
        absorbing = obs.plane is not None and obs.plane.absorbing
        # An absorbing plane receives the remaining energy at most once. On a transparent
        # plane only followed crossings support a bound; unknown returns remain unbounded.
        # A sampled traced value (the geometric pupil-sample basis) supports no bound of its own.
        sampled = obs.power_sampling_resolution_W is not None
        if not sampled and math.isfinite(overall.faint_W) and (not overall.unbounded or absorbing):
            upper = (total or 0.0) + overall.untraced_W + overall.faint_W
        if absorbing:
            upper = capped_by_launch(upper, max(overall.input_W, total or 0.0))
        faint_once = faint_once_sum(reach, obs.observation_id, [(line.source_id, line.spectral_index) for line in budget.lines])
        # (light of unknown fate may go on crossing a plane that does not absorb, as in a cavity: nothing bounds that, whatever one crossing brings)
        once = (total or 0.0) + overall.untraced_W + faint_once if (
            upper is None and not sampled and math.isfinite(faint_once) and not overall.unbounded) else None
        once = None if once is None else capped_by_launch(once, max(overall.input_W, total or 0.0))
        message = obs.message
        lower = obs.power_lower_bound_W if total is not None else None
        if not obs.lit and not overall.complete and not overall.unbounded and upper is not None:
            # (no traced beam, and every watt that may arrive is light the probes followed to the plane under a supported
            # bound: its power lies between 0 and that bound -- no value is claimed for it)
            lower = 0.0
        scope = plane_power_scope(obs)
        if scope:
            # (the power of a beam here, or of light scattered out of one, is out of model scope: the traced value adds chief-ray
            # powers, not the light's, so it bounds nothing either way -- the plane's power is OUT_OF_MODEL_SCOPE)
            upper = once = lower = None
            message = f"{power_scope_reach(scope, total or 0.0)}, so the plane's power and its bounds are not stated"
        elif obs.power_lit and upper is None:  # (a plane no traced beam reaches is worded by grade_unlit_planes)
            message = _unbounded_text(obs, overall)
        bases = dict(obs.value_basis or {})
        entry = None if scope else open_power_basis(obs, overall, budget, total, upper, lower, absorbing)
        if entry is not None:
            bases["total_power_W"] = entry
        out.append(obs.model_copy(update={
            "power_upper_bound_W": upper, "absorbed_power_upper_bound_W": once, "message": message,
            "power_lower_bound_W": lower, "value_basis": bases or None,
            "followed_light": tuple((followed or {}).get(obs.observation_id, ())),
            "power_by_wavelength": tuple(sorted(spectral + extra, key=lambda s: s.wavelength_nm)), "total_power_W": total,
            "power_complete": overall.complete and overall.faint_accounted, "untraced_power_W": overall.untraced_W}))
    return out


def open_power_basis(obs: ObservationResult, fate: LineFate, budget: PowerBudget, total: float | None, upper: float | None,
                     lower: float | None, absorbing: bool) -> ValueBasis | None:
    """The stored entry of a lit plane's ``total_power_W`` that light not followed may add to, where the plane's flat fields
    do not hold the facts: the upper end of its power is open (``power_upper_bound_W`` null), or light of unknown fate is
    among what it does not count. The traced value is the lower end (``LOWER_BOUND``; ``ESTIMATE`` where part of a beam is
    also taken before the plane, ``power_lower_bound_W``), and each omission names its cause: the part, the code, where the
    light left the trace and the most it adds (``power.left_out_parts``). ``None`` elsewhere: a plane traced completely, one
    no beam of known power reaches, and a lit plane whose only uncounted light the probes followed to it under a supported
    bound (``untraced_power_W`` and ``power_upper_bound_W`` hold that; ``results.basis_of`` derives its entry)."""
    if fate.complete or not obs.power_lit or total is None or not (upper is None or fate.unbounded):
        return None
    omitted = list(left_out_parts(fate, budget, absorbing=absorbing, gap=None if upper is None else max(upper - total, 0.0)))
    taken = lower is not None and lower < total
    if taken:
        at = tuple(sorted({s for a in obs.arrivals for s in a.power_taken_before_at}))
        omitted.insert(0, Omitted(part=OmittedPart.BEAM_PART_ELSEWHERE, at=at, upper=max(0.0, total - lower)))
    return ValueBasis(kind=ValueKind.ESTIMATE if taken else ValueKind.LOWER_BOUND, omitted=tuple(omitted))


def plane_power_scope(obs: ObservationResult) -> list[str]:
    """The surfaces behind which the power of a beam on the plane, or of light scattered out of one, is out of model scope
    (``ArrivalResult.power_out_of_scope_at``, ``SampledLightAtPlane.power_out_of_scope_at``), in order; empty: none."""
    return sorted({*(at for a in obs.arrivals for at in a.power_out_of_scope_at),
                   *(at for line in obs.sampled_light for at in line.power_out_of_scope_at)})


def capped_by_launch(upper: float | None, launched_W: float) -> float:
    """An upper bound of the power an absorbing plane (or a detector placed on a plane) receives, at most ``launched_W``,
    the power its source lines launch: each watt is absorbed once, and no light is made on the way. ``None`` (nothing else
    bounds it): that power."""
    return launched_W if upper is None else min(upper, launched_W)


def _unbounded_text(obs: ObservationResult, fate: LineFate) -> str:
    """Why the power of a plane traced beams reach has no upper bound (``power_upper_bound_W`` null): a traced value sampled
    by pupil rays (``power_sampling_resolution_W``), light of unknown fate that may cross a transparent plane any number of
    times, or light left out as negligible that surfaces facing each other may send across the plane without bound
    (``validation/untraced.py``, ``FAINT_POWER``)."""
    traced = f"the traced beams bring {obs.total_power_W:.6g} W"
    if obs.power_sampling_resolution_W is not None:
        traced += (f", a value sampled by pupil rays within {obs.power_sampling_resolution_W:.4g} W (convergence evidence, "
                   "not a bound)")
        if not (fate.unbounded and not (obs.plane is not None and obs.plane.absorbing)):
            return f"{traced}, so no upper bound is taken from it"
    if fate.unbounded and not (obs.plane is not None and obs.plane.absorbing):
        return (f"{traced}; part of the launched power was not followed ({fate.describe()}), and this plane does not absorb: "
                "that light may cross it more than once, so the plane's power has no upper bound")
    return (f"{traced}; light the trace leaves out as negligible may add power without bound ({fate.unbounded_returns()}), so "
            "the plane's power has no upper bound")


#: Metrics whose value is the power a plane receives: they measure a dark plane as 0 W.
POWER_SUM_METRICS = frozenset({"power_W", "coherent_power_W"})


def surface_name(surface_id: str) -> str:
    """A surface as a customer reads it: ``observation plane D`` for ``observation/D``, an element's surface by its id."""
    return f"observation plane {surface_id.removeprefix('observation/')}" if surface_id.startswith("observation/") else surface_id


def partial_reach(surfaces: tuple[str | None, ...]) -> str:
    """Where part of a traced beam reaches a plane its chief ray does not cross (``PARTIAL_CROSSING``)."""
    names = sorted({s for s in surfaces if s})
    where = f"before {', '.join(surface_name(s) for s in names)}" if names else "on its way out of the scene"
    return f"part of a traced beam reaches this observation plane {where}, while the beam's chief ray does not cross it"


def unknown_power_reach(count: int) -> str:
    """What a plane only beams of unknown power reach says of them (a fact of the trace alone)."""
    return f"{count} beam(s) reach this observation plane at known positions and directions; their power is not known"


def power_scope_reach(scope: list[str], traced_W: float) -> str:
    """What a plane a beam of out-of-scope power reaches says of its traced value (a fact of the trace alone)."""
    return (f"the power of a beam here is out of model scope behind {', '.join(scope)} (power_out_of_scope_at): "
            f"the traced beams bring {traced_W:.6g} W by their chief rays' powers, which is not the light's")


def _partial_text(obs: ObservationResult, fate: LineFate, surfaces: tuple[str | None, ...]) -> str:
    """What a plane only part of a traced beam reaches receives (``PARTIAL_CROSSING``): before which surfaces, and the range
    of its power."""
    upper = obs.power_upper_bound_W if obs.power_upper_bound_W is not None else fate.untraced_W + fate.faint_W
    return f"{partial_reach(surfaces)}; its power there is not known, and at most {upper:.4g} W"


def _beyond_the_reach(obs: ObservationResult, budget: PowerBudget, cfg: NumericalConfig) -> str:
    """The clause a dark plane without an active area carries when a beam leaves the scene: how far along that beam it counts."""
    if obs.plane is None or obs.plane.aperture is not None or not any(line.sinks.get(SINK_ESCAPED, 0.0) > 0.0 for line in budget.lines):
        return ""
    return (f" within {cfg.laboratory_reach_mm:g} mm of the start of a leg that leaves the scene, or the scene's own extent along it if "
            "that is farther, which is as far as a plane with no active area counts (to read light beyond it, state an active "
            "area or end the leg on a dump)")


def grade_unlit_planes(observations: list[ObservationResult], issues: list[ValidationIssue], budget: PowerBudget | None,
                       cfg: NumericalConfig, reach: PlaneReach | None = None,
                       partial_before: dict[str, tuple[str | None, ...]] | None = None,
                       ) -> tuple[list[ObservationResult], list[ValidationIssue]]:
    """Grade each plane that no traced beam reaches (the ``OBSERVATION_NOT_REACHED`` / ``ZERO_POWER`` issue built with it).
    The grade, the message and the issue are functions of the physics only (the target's result says what it decides).

    * Launched power of unknown fate (trace budget, interaction limit, a path ended out of model scope, a line not launched,
      light at the power floor) may reach the plane: whether light reaches it is not known -- ``UNKNOWN``. Its power stays
      ``UNKNOWN`` too (``outcome.quantity_state_of``), and ``untraced_power_W`` and any supported bound are reported beside it.
    * Only light the probe traces followed reaches it (``reach``: an untraced reflection or coating transmission, light
      passing an aperture or an element its chief ray misses, a beam's tail on the active area, part of a beam crossing the
      plane where its chief ray does not -- ``partial_before``: plane -> the surfaces that part reaches the plane before,
      ``tracer.PartialCrossing.ends_at``, in the issue's context ``partial_before_surface_ids``), under a supported bound:
      its power lies between 0 and ``power_upper_bound_W`` (``power_lower_bound_W`` 0.0, ``BOUNDED``, no value claimed) and
      ``followed_light`` lists the contributions; the plane's status is ``PASS`` and its issue a notice. Without a supported
      bound it is ``UNKNOWN``.
    * Otherwise the plane is dark: the traced model sends it 0 W, a measured fact -- the plane's status is ``PASS`` and its
      issue a notice (``NOT_APPLICABLE``); a target on the plane says whether that is what the design wants. Light left out as
      negligible (below the lines' negligible level) is the stated truncation of that 0 W and changes no grade: the most it
      brings is the issue's ``faint_power_upper_bound_W`` and the plane's ``power_upper_bound_W`` (null where nothing bounds
      its returns). A plane without an active area beside a beam that leaves the scene says how far that holds
      (``_beyond_the_reach``)."""
    if budget is None:
        return observations, issues
    graded: dict[str, Status] = {}
    fates: dict[str, LineFate] = {}
    out_obs = []
    for obs in observations:
        if obs.power_lit:
            out_obs.append(obs)
            continue
        if obs.arrivals:  # (its beams' positions and directions are known, their powers are not)
            fate = plane_fate(obs, budget.lines, cfg, reach)
            bound = (f"between 0 W and {obs.power_upper_bound_W:.4g} W" if obs.power_upper_bound_W is not None else
                     "without an upper bound, since this plane does not absorb and that light may cross it more than once")
            out_obs.append(obs.model_copy(update={"status": worst([obs.status, Status.UNKNOWN]), "message": (
                f"{unknown_power_reach(len(obs.arrivals))} ({fate.describe()}): the plane receives {bound}")}))
            continue
        fate = fates[obs.observation_id] = plane_fate(obs, budget.lines, cfg, reach)
        if fate.complete and obs.power_sampling_resolution_W is not None:
            # (a sampled zero: no ray of the sampled lines is counted on it -- 0 W, resolved to the rays' weight)
            status, message = Status.PASS, _sampled_zero_text(obs)
        elif fate.complete and fate.faint_accounted:
            # (dark: a measured 0 W, which grades nothing -- the plane may be meant to stay dark; light left out as
            # negligible is the stated truncation of the 0 W while it stays within the lines' negligible level here)
            status, message = Status.PASS, f"no light reaches this observation plane{_beyond_the_reach(obs, budget, cfg)}: it receives 0 W"
        elif fate.complete:
            # (dark for the traced light, but the light left out as negligible may bring more than the negligible level here:
            # its returns or crossings -- the range covers it)
            status = Status.UNKNOWN
            message = ("no traced beam reaches this observation plane; omitted faint light may reach it: "
                       + ("received power has no supported finite upper bound" if obs.power_upper_bound_W is None else
                          f"received power lies between 0 W and {obs.power_upper_bound_W:.4g} W"))
        elif obs.power_lower_bound_W == 0.0:
            # (only light the probes followed reaches it, under a supported bound: a stated interval, which decides a power target
            # there only in the direction it proves -- validation/metrics.py _no_arrival)
            status = Status.PASS
            lead = (partial_reach((partial_before or {}).get(obs.observation_id, ())) if fate.partial_only else
                    f"no traced beam reaches this observation plane; light the trace does not follow as beams reaches it ({fate.describe()})")
            message = f"{lead}: it receives between 0 W and {obs.power_upper_bound_W:.4g} W"
        elif fate.faint_W > 0:
            status = Status.UNKNOWN
            # (the range covers every light that may arrive: the faint light, and a partly crossing or untraced beam's)
            lead = (f"{partial_reach((partial_before or {}).get(obs.observation_id, ()))}, and omitted faint light may reach it"
                    if fate.partial_only else
                    "no traced beam reaches this observation plane; omitted faint light and light the trace does not follow may "
                    "reach it")
            message = (f"{lead}: "
                       + ("received power has no supported finite upper bound" if obs.power_upper_bound_W is None else
                          f"its power there is not known, and at most {obs.power_upper_bound_W:.4g} W"))
        elif fate.partial_only:
            status, message = Status.UNKNOWN, _partial_text(obs, fate, (partial_before or {}).get(obs.observation_id, ()))
        else:
            status, message = Status.UNKNOWN, ("no traced beam reaches this observation plane, and part of the launched power was "
                                               f"not followed ({fate.describe()}): whether light reaches it is not known")
        graded[obs.observation_id] = status
        out_obs.append(obs.model_copy(update={"status": status, "message": message}))
    out_issues = []
    for issue in issues:
        oid = issue.location.removeprefix("observation:") if issue.location else None
        if oid not in graded or issue.code not in (ErrorCode.OBSERVATION_NOT_REACHED, ErrorCode.ZERO_POWER):
            out_issues.append(issue)
            continue
        context = dict(issue.numerical_context)
        action = issue.suggested_action
        fate = fates[oid]
        partial = None
        if fate.partial_only:  # (the surfaces part of a traced beam reaches the plane before, for the reader to name)
            before = (partial_before or {}).get(oid, ())
            names = sorted({s for s in before if s})
            context.update(partial_before_surface_id=names[0] if names else None, partial_before_surface_ids=names)
            partial = _partial_text(next(o for o in out_obs if o.observation_id == oid), fate, before)
        follow = ("let the trace follow that light (a larger max_paths or max_interactions_per_path, the issue that ended a "
                  "path out of model scope, or a traced reflection / a beam that passes the aperture whole), then read "
                  "this plane again" + (f"; if light should reach it: {issue.suggested_action}" if issue.suggested_action else ""))
        if DIFFRACTED_BEAM in fate.unknown_W:
            follow = DIFFRACTED_BEAM_ACTION if len(fate.unknown_W) == 1 else f"{DIFFRACTED_BEAM_ACTION}; for the other light: {follow}"
        plane = next(o for o in out_obs if o.observation_id == oid)
        if fate.complete and plane.power_sampling_resolution_W is not None:
            status = Status.NOT_APPLICABLE  # (a notice: the plane's sampled 0 W and its resolution carry the fact)
            text = plane.message
            context.update(power_sampling_resolution_W=plane.power_sampling_resolution_W,
                           sampled_lines_not_counted=sum(1 for line in plane.sampled_light if not line.counted_crossing_count))
            action = None
        elif fate.complete and fate.faint_accounted:
            status = Status.NOT_APPLICABLE  # (a notice: the plane's KNOWN_ZERO and its 0 W carry the fact)
            text = (f"{issue.message}{_beyond_the_reach(plane, budget, cfg)}: all launched light was traced to known ends, and no untraced "
                    "reflection, coating transmission or clipped light reaches it (followed through the optics to every order above "
                    "negligible_power_fraction of its line's power), so the plane receives 0 W")
            if fate.faint_W > 0:
                # (the light left out below that level: the most it brings here, null where nothing bounds its returns)
                context.update(faint_power_upper_bound_W=fate.faint_W if math.isfinite(fate.faint_W) else None)
        elif graded[oid] is Status.PASS:
            # (only followed light under a supported bound: a notice; the plane's BOUNDED power, its bounds and its
            # followed_light carry the fact)
            status = Status.NOT_APPLICABLE
            text = next(o.message for o in out_obs if o.observation_id == oid)
            context.update(untraced_power_W=fate.untraced_W, untraced_causes=list(fate.cause_codes()), untraced_light=fate.causes(),
                           **({"faint_power_upper_bound_W": fate.faint_W} if fate.faint_W > 0 else {}))
            if issue.code is ErrorCode.OBSERVATION_NOT_REACHED:
                action = PARTIAL_CROSSING_ACTION if fate.partial_only else follow
        elif fate.faint_W > 0:
            status = Status.UNKNOWN
            text = next(o.message for o in out_obs if o.observation_id == oid)
            if not fate.complete and not fate.partial_only:
                text += f"; {fate.describe()}"
            context.update(faint_power_upper_bound_W=fate.faint_W if math.isfinite(fate.faint_W) else None,
                           untraced_power_W=fate.untraced_W,
                           untraced_causes=list(fate.cause_codes()),
                           untraced_light=fate.causes())
            if issue.code is ErrorCode.OBSERVATION_NOT_REACHED and not fate.complete:
                action = PARTIAL_CROSSING_ACTION if fate.partial_only else follow
        else:
            status = Status.UNKNOWN
            if issue.code is ErrorCode.ZERO_POWER:  # (attribute_power_floor: every source is at the floor) already says why
                text = f"{issue.message}: whether that light reaches the plane is not known"
            elif partial is not None:
                text = partial
            else:
                text = (f"{issue.message}; part of the launched power was not followed ({fate.describe()}): whether light reaches "
                        "it is not known")
            context.update(untraced_power_W=fate.untraced_W,
                           untraced_causes=list(fate.cause_codes()), untraced_light=fate.causes())
            if issue.code is ErrorCode.OBSERVATION_NOT_REACHED and fate.partial_only:
                action = PARTIAL_CROSSING_ACTION
            elif issue.code is ErrorCode.OBSERVATION_NOT_REACHED:
                # the plane may also simply be misplaced: keep the advice on the beam path (and an attached plane's frame)
                action = follow
        out_issues.append(issue.model_copy(update={"status": status, "message": text, "numerical_context": context,
                                                   "suggested_action": action}))
    return out_obs, out_issues


def received_power_W(rec: ArrivalRecord) -> float:
    """The power the plane receives from one counted arrival: the detected power on an absorbing plane (its active area's
    share of the beam), the crossing beam's power on a transparent one; nothing from a beam outside the plane's acceptance
    cone."""
    if rec.detected_power_W is not None:
        return rec.detected_power_W
    return 0.0 if rec.within_acceptance is False else rec.power_W


def inside_active(rec: ArrivalRecord, ap: ApertureG, cfg: NumericalConfig) -> bool:
    """Chief-ray membership of the active area."""
    from ..optics.surfaces.geometry import aperture_inside

    return bool(aperture_inside(ap, np.array([rec.uv[0]]), np.array([rec.uv[1]]), cfg.aperture_edge_tol_mm)[0])


def attribute_power_floor(issues: list[ValidationIssue], paths, sources, cfg: NumericalConfig) -> list[ValidationIssue]:
    """Name the power floor when it is why no beam reaches a plane: when every source's power is at or below
    ``power_epsilon_W`` (e.g. 1e-16 W), every traced path ends at once (``ZERO_POWER``) and each plane's
    ``OBSERVATION_NOT_REACHED`` becomes ``ZERO_POWER`` stating the floor, instead of advice to check the beam path. A beam
    a component extinguishes (e.g. crossed polarizers) is still a plane no beam reaches."""

    if not sources or any(source.power_W > cfg.power_epsilon_W for source in sources):
        return issues
    if any(path.termination != ZERO_POWER_TERMINATION for path in paths):
        return issues
    strongest = max(source.power_W for source in sources)
    action = (f"set numerical_settings.power_epsilon_W below the source power (for example {strongest * cfg.negligible_power_fraction:.0e} W)"
              if strongest > 0.0 else "give the source a power above 0 W")
    return [
        make_issue(ErrorCode.ZERO_POWER, issue.status,
                   f"no beam reaches this observation plane: the source power is at or below the power floor "
                   f"({cfg.power_epsilon_W:g} W), where the trace stops", location=issue.location,
                   power_floor_W=cfg.power_epsilon_W, source_power_W=strongest, setting="power_epsilon_W",
                   suggested_action=action)
        if issue.code is ErrorCode.OBSERVATION_NOT_REACHED else issue
        for issue in issues
    ]
