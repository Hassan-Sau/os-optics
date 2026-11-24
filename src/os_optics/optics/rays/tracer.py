"""Non-sequential 3D tracer: chief ray + deterministic pupil samples, Gaussian q, Jones, OPL, branching.

Each *packet* is a bundle of rays sharing one source, one wavelength, one lineage and one medium; ray 0 is the chief
ray. A packet travels in straight legs from one physical surface to the next. Every step takes the chief ray to its next
crossing: a physical surface, which every live ray then reaches from where the leg began (pupil rays whose first surface
that stops or turns them -- a physical surface or an absorbing plane -- is another one, or none, are retired as
``DIVERGED``: marginal stray light is not followed, the packet's power and Gaussian state stay those of the chief's
path), or observation planes alone. Observation planes are virtual: each samples every live ray that crosses it before
the surface that would stop or turn that ray, where that ray crosses it, and none moves, turns, retires or counts a ray
-- the chief ray too stays where the leg began (``_Packet.leg_t``), so adding a transparent plane changes no other plane's,
target's or exposure's result (it is one more trace step, and an escaping path's length ends at the chief ray's last
event), and planes through one point, coplanar or tilted, sample the beam independently of each other and of their declaration
order. An absorbing plane that counts the beam ends the path. At the end of each leg, every observation plane part of the
beam reaches on the leg while the chief ray does not cross it is recorded (``TraceOutput.partial_crossings``) with a
bound of the power that part carries (the Gaussian beam's share, ``decisions.leg_share``): ``validation/untraced.py`` counts
it as light that may reach the plane instead of reading a known zero, and an absorbing such plane puts the beam's power out
of model scope from there on (``power_scope``). Splitting elements create child packets. Which queued packet is traced next, and whether the trace
budget (``max_paths``, ``max_trace_steps``, ``max_interactions_per_path``) still allows a launch, a branch or a step, is
decided by the ``TraceScheduler`` (``scheduler.py``); the tracer books the power the budget leaves untraced. Which
surfaces a crossing holds, what part of the beam crosses a plane its chief ray does not, which surfaces take part of
the beam before a plane, and what a plane reads of a crossing are decided by pure functions over explicit inputs
(``decisions.py``); the tracer applies what they return.

A diffraction grating (``optics/surfaces/grating.py``) ends the packet that meets it and starts one branch per order it
follows. Where its response does not give an order's power, the incident power goes to ``UNMODELLED_RESPONSE`` once and the
branch carries no power at all (``_Packet.power_bound_W``: its Jones vector is zero, nothing it does is booked and no power
floor ends it): its rays are geometry -- positions, directions, Gaussian beam -- and every observation plane reads it
with its power not known and bounded, never as a zero. Such a branch follows every branch of its own unconditionally, so
geometric results behind a beam splitter stay available too.

An optional ``recorder`` (``TraceObserver``) is told about each step of each packet so it can copy the per-ray state the
tracer produced (``validation/ray_paths.py`` builds the pupil / chief ray paths from it), including which surface each
retired ``DIVERGED`` ray would have reached instead. It is observation only: the tracer never reads anything back from
it, and with ``recorder=None`` every call site is skipped.

Ray layout of a packet: index 0 is the chief ray, ``1 .. n_pupil`` the pupil samples and, when pupil sampling is on, the
last ``DIFFERENTIAL_RAY_COUNT`` rays are *differential* rays: copies of the chief ray displaced by ``+-dx``, ``+-dy`` at the
source plane (along the source frame x / y) and tilted by ``+-da``, ``+-db`` (toward the source x / y), with
``dx = first_order_position_step_mm`` and ``da = first_order_angle_step_rad``. They are traced exactly like every other
ray and give, by central differences at each observation crossing, the first-order ray-transfer matrix from the source
plane to that plane (``ArrivalRecord.transfer``). They never count as pupil samples.

Power ledger: every place where traced power leaves the modelled beams is recorded as a ``PowerFlow`` (``SINK_*`` kinds:
detected, absorbed, escaped, clipped, truncated at an aperture, a reflection or transmission the trace does not follow,
coating / bulk / polarizer / element losses, and the out-of-model terminations), together with every source's input
power (``TraceOutput.inputs``). ``validation/power.py`` turns them into the public ``PowerBudget``. Light that leaves the
traced beams by a model approximation but goes on physically (an untraced reflection or coating transmission, the part of
a clipped beam inside the opening, light beyond an element's support) is also recorded where it starts
(``TraceOutput.untraced_branches``): ``validation/untraced.py`` decides from it whether a plane no traced beam reaches is
dark. It is never traced on or added to a result.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field, replace
from collections.abc import Callable
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from ...config import NumericalConfig
from ...contracts.optics import POLARIZATION_MATCH_FRACTION, OpticalPath, RaySegment
from ...contracts.results import GratingOrderResult, NonlinearConversionResult
from ...errors import ErrorCode, QuantityState, Status, ValidationIssue, make_issue, notice
from ...math.vectors import correctly_rounded_sum, dot, matmul, normalize, parallel_transport_basis, transverse_basis
from ...run_control import checkpoint
from ...scene.model import ApertureG, GradientFaceG, InteractionG, Scene, SourceG, SurfaceG, line_coherence_length_mm
from ..gaussian import (
    GState,
    LocalCurvature,
    apply_curved_or_flat_interface,
    apply_flat,
    apply_grating,
    apply_thin_lens,
    gauss_legendre_rule,
    initial_state,
    mark_truncation,
    propagate,
    transmission_fraction,
)
from ..materials.coatings import CoatingCoefficients, CoatingEvaluator
from ..polarization.phases import PhaseValues, coated_interface, mirror_reflection, mirror_transmission, splitter_amplitudes
from ...math.exact_linalg import cmul, power2
from ...math.exact_trig import atan2_deg, cis, cos_sin, expm1, incidence_deg
from ...math.exact_trig import exp as exact_exp
from ...math.exact_trig import exp_float
from ...math.exact_trig import atan2 as exact_atan2
from ..materials.group import NM3_PER_NM2_TO_MM
from ..materials.library import MaterialLibrary
from ..gaussian.state import check_divergence
from . import gradient
from ..polarization import (
    apply_interface,
    apply_thin_transmission,
    faraday_rotation,
    field_3d,
    fresnel,
    polarizer,
    project_axis,
    retarder,
    s_axis,
    to_sp,
)
from ..surfaces import Hits, PlaneBatch, SphereBatch, aperture_inside, aperture_miss_distance, intersect, reflect_dir, refract, thin_lens_dir
from ..surfaces.freeform import hessian_bound as freeform_hessian_bound
from ..surfaces.freeform import sag_range as freeform_sag_range
from ..surfaces.geometry import support_of
from .. import apertures
from ..surfaces.grating import AOI_MATCH_DEG, GratingG, GratingReading, diffracted, propagating_orders
from ..surfaces.sag import local_normals as sag_local_normals
from ..surfaces.sag import principal_curvatures
from ..surfaces.tabulated_lens import (
    DOMAIN_LIMIT_ABS_TOLERANCE_DEG,
    DOMAIN_LIMIT_ABS_TOLERANCE_MM,
    ThinLensTable,
    beyond_limit,
    nearest_row_quantities,
    orthogonal_fractions,
    response_at,
)
from ..attenuation import ND_FOOTPRINT_OD_TOLERANCE, transmission_error_estimate
from .envelope import NO_SPREAD, PARALLEL_PLATE_TOL, Spread, add_spread, plate_spread
from ..interference.coherence import delay_key, polarization_state
from .hits import HeldData, SurfaceHit
from .path_ids import child_path_id, root_path_id
from ..polarization.partial import source_components
from ..polarization import uniaxial
from ..nonlinear import shg
from ..polarization.uniaxial import Crystal, Waves
from .pupil import pupil_offsets
from .scatter import SINK_BULK_SCATTER, SINK_SURFACE_SCATTER, ScatterOrigin, bulk_shares
from ..complex_transmission import PHASE_RESIDUAL_TOL_RAD, POWER_FACTOR_TOL, RAY_POWER_SPREAD_TOL, TWO_PI, beam_power_fraction
from ..complex_transmission import evaluate as transmission_evaluate
from ..complex_transmission import evanescent as transmission_evanescent
from ..complex_transmission import evanescent_share_bound
from ..complex_transmission import footprint as transmission_footprint
from ..complex_transmission import phase_residual as transmission_phase_residual
from ...units import NM_TO_MM
from .scheduler import BRANCH_NOT_TRACED, LINE_NOT_TRACED, TraceScheduler, order_power, order_quantum
from .lower_share import EITHER_SIDE, certain_meeting_share, possible_meeting_share
from ...units import RAD_TO_MRAD
from ...math.exact_erf import erfc_float
from ...units import NM_TO_MM
from .decisions import (
    PREFILTER_REL_MARGIN,
    SETTLE_MARGIN,
    SETTLE_STEPS,
    START_SURFACE,
    TAIL_BESIDE,
    AnnulusOnArea,
    BeamView,
    TakenBefore,
    beam_part_W,
    Leg,
    NextSurface,
    PartialCrossing,
    SceneIndex,
    Sheet,
    StartSurface,
    SurfaceCoincidenceRecord,
    literal_crossing_share,
    counts_on_active_area,
    crossing_group,
    envelope_sigmas,
    first_blocking,
    flow_stretch,
    reach_stretch,
    _surface_balls,
    leg_end_candidates,
    leg_partial_crossings,
    observation_reading,
    taken_before,
    widest,
)

Array = NDArray[np.float64]
CArray = NDArray[np.complex128]

# Termination reasons (strings appear in OpticalPath.termination)
#: A grating with more propagating orders than this (a coarse period: an acousto-optic deflector's) lists only the orders it follows in ``grating_orders``.
MAX_LISTED_PROPAGATING_ORDERS = 129
#: The chief ray's leg in a gradient-index rod is recorded as this many straight pieces per period of its path (at most
#: ``GRADIENT_MAX_SEGMENTS``); the first-order transfer of the Gaussian beam is taken by central differences of this step
#: (mm across the chief ray, rad of tilt) and must be symplectic to ``GRADIENT_SYMPLECTIC_TOL``.
#: Corners of an element's projected support outline and of a hole's outline in the shadow check (``Tracer._holes_in_shadow``),
#: and the fewest annulus rays the affine map is fitted from.
SHADOW_RING_SEGMENTS = 128
SHADOW_FIT_MIN_RAYS = 6
#: A curved support (sag surface, sphere) that obstructs the beam is taken as met along its axis when the chief ray's direction
#: lies within this sine of it (rounding of rotated unit vectors); met farther off it, the share it stops is not computed
#: (``Tracer._off_axis_curved``): the annulus's power behind it is out of model scope.
CURVED_SUPPORT_AXIS_SIN_TOL = 1e-7
GRADIENT_SEGMENTS_PER_PERIOD = 16
GRADIENT_MAX_SEGMENTS = 256
GRADIENT_JACOBIAN_STEP = 1e-6
GRADIENT_SYMPLECTIC_TOL = 1e-6
#: A chief ray within this cosine of running along a rod's end face has no first-order transfer through the rod.
GRADIENT_GRAZING_COS = 1e-9
#: Angles per turn of the beam's envelope rays checked against a rod's side (``Tracer._gradient_envelope``).
GRADIENT_ENVELOPE_STEPS = 16
#: The chief ray's leg in an axial gradient-index medium is recorded as this many straight pieces.
GRADIENT_AXIAL_SEGMENTS = 8
ESCAPED = "ESCAPED"
CLIPPED = "CLIPPED"
ABSORBED = "ABSORBED"
DETECTED = "DETECTED"
BLOCKED_BACK = "BLOCKED_BY_BACK_SIDE"
#: The cause (``numerical_context["cause"]``) of the OUT_OF_MODEL_SCOPE issue of a beam meeting the back of a mirror whose coating
#: transmits: the trace does not follow light through a mirror (``os_optics.local.cavity`` evaluates a cavity fed that way).
MIRROR_TRANSMISSION_NOT_TRACED = "MIRROR_TRANSMISSION_NOT_TRACED"
MAX_INTERACTIONS = "MAX_INTERACTIONS"
ZERO_POWER = "ZERO_POWER"
GRAZING = "GRAZING_INCIDENCE"
MEDIUM_MISMATCH = "MEDIUM_MISMATCH"
MATERIAL_RANGE = "MATERIAL_OUT_OF_RANGE"
COATING_RANGE = "COATING_OUT_OF_RANGE"
DEGENERATE = "DEGENERATE_SURFACE"
TRACE_BUDGET = "TRACE_BUDGET"  # the run's trace budget (max_trace_steps) was spent while this path was traced
NOT_ACCEPTED = "OUTSIDE_ACCEPTANCE"  # an absorbing plane stopped the beam outside its detector's acceptance cone (not detected)
ANNULUS_NOT_FOLLOWED = "ANNULAR_BEAM_NOT_FOLLOWED"  # the annulus behind a central obstruction is cut further on: not followed
# a beam without a valid Gaussian state (e.g. severely truncated) meets an aperture that cuts its rays: the power that
# passes is not computed, so the path is not followed further
TRUNCATED_NOT_FOLLOWED = "TRUNCATED_BEAM_NOT_FOLLOWED"
# the surface id prefix of an observation plane (``scene/lower.py``)
OBSERVATION_PREFIX = "observation/"
# ``Tracer._sag_tangent`` looks for a window radius ``r`` with ``F(r) < r`` from the tangent plane's own estimate, growing
# the trial radius by this factor at most this many times
SAG_WINDOW_GROWTH = 1.25
SAG_WINDOW_STEPS = 12
# ``_sag_region_share``: the beam's profile within ``K`` radii is cut into REGION_GRID x REGION_GRID cells, each split in four
# up to REGION_CELL_DEPTH times where its lines may meet the surface before the plane, the chief ray's stretch halved up to
# REGION_STRETCH_DEPTH times per cell; past REGION_BOX_BUDGET weighed boxes the region states no bound
REGION_GRID = 8
REGION_CELL_DEPTH = 4
REGION_STRETCH_DEPTH = 12
REGION_BOX_BUDGET = 600_000
# the chief ray crosses an attenuator where its density law leaves the plane uncoated and no uncoated transmittance is stated
# (these three end a probe; a traced beam goes on there with a power that is not computed)
ND_UNCOATED = "ND_UNCOATED_REGION"
# light of a wavelength a Faraday rotator states no response for
FARADAY_UNAVAILABLE = "FARADAY_RESPONSE_UNAVAILABLE"
# a retarder whose retardance is stated per wavelength met by light of another wavelength (out of model scope)
RETARDANCE_UNAVAILABLE = "RETARDANCE_UNAVAILABLE"
# the beam met a diffraction grating: its light goes on in the orders' branches
GRATING_ORDERS = "GRATING_ORDERS"
# central-difference steps of the acceptance bandwidths of second-harmonic generation (``Tracer._acceptance``): internal angle
# (rad), temperature (K) and fundamental wavelength (nm), for the first and the second derivative of the phase mismatch
ACCEPTANCE_ANGLE_STEP_RAD = 1e-6
ACCEPTANCE_ANGLE_STEP2_RAD = 1e-4
ACCEPTANCE_TEMPERATURE_STEP_K = 1e-2
ACCEPTANCE_TEMPERATURE_STEP2_K = 1.0
ACCEPTANCE_WAVELENGTH_STEP_NM = 1e-3
ACCEPTANCE_WAVELENGTH_STEP2_NM = 0.1

# Power ledger sink kinds (where traced power leaves the modelled beams).
SINK_DETECTED = "DETECTED"  # absorbed by an absorbing observation plane (a detector / dump face) on its active area
SINK_OUTSIDE_ACTIVE_AREA = "OUTSIDE_ACTIVE_AREA"  # the part of a beam stopped at an absorbing plane that falls beside its active area
SINK_OUTSIDE_ACCEPTANCE = "OUTSIDE_ACCEPTANCE"  # a beam an absorbing plane stops on its active area outside the acceptance cone
SINK_ABSORBED = "ABSORBED"  # an absorber surface
SINK_ESCAPED = "ESCAPED"  # left the scene without meeting another surface
SINK_CLIPPED = "CLIPPED"  # chief ray on an opaque mount rim (the whole beam), or the part a central obstruction blocks
SINK_APERTURE_TRUNCATION = "APERTURE_TRUNCATION"  # Gaussian power outside a clear aperture the chief ray passes
SINK_UNTRACED_REFLECTION = "UNTRACED_REFLECTION"  # reflection at a transmitting interface that is not followed
SINK_UNTRACED_TRANSMISSION = "UNTRACED_TRANSMISSION"  # power a reflector's coating transmits (not followed)
SINK_COATING_ABSORPTION = "COATING_ABSORPTION"  # 1 - R - T of a coated surface (absorption / scatter in the coating)
SINK_BULK_ABSORPTION = "BULK_ABSORPTION"  # exp(-alpha L) in an absorbing medium
SINK_POLARIZER_REJECTION = "POLARIZER_REJECTION"
SINK_ELEMENT_TRANSMISSION_LOSS = "ELEMENT_TRANSMISSION_LOSS"  # retarder / ideal lens / aperture stop / Faraday rotator transmittance < 1, a grating's stated absorption
SINK_ATTENUATOR_ABSORPTION = "ATTENUATOR_ABSORPTION"  # 1 - R - 10^-OD of a neutral-density attenuator
SINK_BLOCKED_BACK_SIDE = "BLOCKED_BACK_SIDE"
SINK_TERMINATED_OUT_OF_MODEL = "TERMINATED_OUT_OF_MODEL"  # grazing, medium mismatch, data out of range, degenerate axis
SINK_PATH_LIMIT = "PATH_LIMIT"  # power the trace budget left untraced: a source line or branch not traced, a path cut short
SINK_INTERACTION_LIMIT = "INTERACTION_LIMIT"
SINK_BELOW_POWER_FLOOR = "BELOW_POWER_FLOOR"  # a path ended because its power fell to power_epsilon_W
SINK_SOURCE_NOT_TRACED = "SOURCE_NOT_TRACED"  # a source line the tracer could not launch (medium data missing)
# 1 - T(lambda) of a tabulated thin lens: light its response model does not deliver into the focused beam (other orders,
# reflection, scatter, absorption); the power of a grating's orders its response does not give, and of the orders it does not
# follow; not followed as a power, so it may still reach any plane
SINK_UNMODELLED_RESPONSE = "UNMODELLED_RESPONSE"
# Power entering a material whose absorption is not stated at this wavelength: the
# ray continues geometrically with zero *assumed* attenuation, but carries only an
# upper bound on power. Book it once, before the bound is attached to the packet.
SINK_BULK_LOSS_UNKNOWN = "BULK_LOSS_UNKNOWN"
# power a crystal converts into a generated second harmonic (booked on the fundamental's line; the harmonic's own line
# starts with it, ``SourceInput.generated_from``)
SINK_NONLINEAR_CONVERSION = "NONLINEAR_CONVERSION"
#: Every sink kind, in reporting order.
SINK_KINDS = (
    SINK_DETECTED, SINK_OUTSIDE_ACTIVE_AREA, SINK_OUTSIDE_ACCEPTANCE, SINK_ABSORBED, SINK_ESCAPED, SINK_CLIPPED, SINK_APERTURE_TRUNCATION, SINK_UNTRACED_REFLECTION,
    SINK_UNTRACED_TRANSMISSION, SINK_COATING_ABSORPTION, SINK_BULK_ABSORPTION, SINK_POLARIZER_REJECTION,
    SINK_ELEMENT_TRANSMISSION_LOSS, SINK_ATTENUATOR_ABSORPTION, SINK_BLOCKED_BACK_SIDE, SINK_TERMINATED_OUT_OF_MODEL, SINK_PATH_LIMIT,
    SINK_INTERACTION_LIMIT, SINK_BELOW_POWER_FLOOR, SINK_SOURCE_NOT_TRACED, SINK_UNMODELLED_RESPONSE, SINK_SURFACE_SCATTER,
    SINK_BULK_SCATTER, SINK_NONLINEAR_CONVERSION, SINK_BULK_LOSS_UNKNOWN,
)
# Path termination -> the sink its remaining power goes to.
TERMINATION_SINKS = {
    ESCAPED: SINK_ESCAPED, CLIPPED: SINK_CLIPPED, ABSORBED: SINK_ABSORBED, DETECTED: SINK_DETECTED,
    BLOCKED_BACK: SINK_BLOCKED_BACK_SIDE, MAX_INTERACTIONS: SINK_INTERACTION_LIMIT, ZERO_POWER: SINK_BELOW_POWER_FLOOR,
    GRAZING: SINK_TERMINATED_OUT_OF_MODEL, MEDIUM_MISMATCH: SINK_TERMINATED_OUT_OF_MODEL, MATERIAL_RANGE: SINK_TERMINATED_OUT_OF_MODEL,
    COATING_RANGE: SINK_TERMINATED_OUT_OF_MODEL, DEGENERATE: SINK_TERMINATED_OUT_OF_MODEL, TRACE_BUDGET: SINK_PATH_LIMIT,
    NOT_ACCEPTED: SINK_OUTSIDE_ACCEPTANCE, ANNULUS_NOT_FOLLOWED: SINK_TERMINATED_OUT_OF_MODEL,
    TRUNCATED_NOT_FOLLOWED: SINK_TERMINATED_OUT_OF_MODEL, ND_UNCOATED: SINK_TERMINATED_OUT_OF_MODEL,
    FARADAY_UNAVAILABLE: SINK_TERMINATED_OUT_OF_MODEL, RETARDANCE_UNAVAILABLE: SINK_TERMINATED_OUT_OF_MODEL,
}
#: Sinks of power whose further fate the trace does not know: it may still reach any observation plane (the trace budget,
#: the per-path interaction limit, a source line not launched, a path ended out of model scope). A plane no traced beam
#: reaches is dark only when no launched line has power here (``validation/power.py``: ``line_fate``).
UNKNOWN_FATE_SINKS = (SINK_PATH_LIMIT, SINK_INTERACTION_LIMIT, SINK_SOURCE_NOT_TRACED, SINK_TERMINATED_OUT_OF_MODEL,
                      SINK_UNMODELLED_RESPONSE, SINK_SURFACE_SCATTER, SINK_BULK_SCATTER, SINK_BULK_LOSS_UNKNOWN)
#: Light that leaves the traced beams by a model approximation but goes on physically (``UntracedBranch``): an interface
#: reflection or a coating transmission the trace does not follow, the part of a beam that passes a clear aperture its
#: chief ray misses (the whole beam is booked ``CLIPPED``), the part of a beam that passes beside an element's support
#: (inside ``APERTURE_TRUNCATION``). On a plane a traced beam reaches, the power counts the traced beams (the declared
#: approximation); a plane no traced beam reaches is dark only where none of it can arrive (``validation/untraced.py``).
UNTRACED_BRANCH_KINDS = (SINK_UNTRACED_REFLECTION, SINK_UNTRACED_TRANSMISSION, SINK_CLIPPED, SINK_APERTURE_TRUNCATION)
# Interaction kind -> the sink of whatever power its interaction loses beyond the explicitly split parts.
INTERACTION_LOSS_SINKS = {
    "refract": SINK_COATING_ABSORPTION, "mirror": SINK_COATING_ABSORPTION, "beam_splitter": SINK_COATING_ABSORPTION,
    "polarizer": SINK_POLARIZER_REJECTION, "retarder": SINK_ELEMENT_TRANSMISSION_LOSS, "thin_lens": SINK_ELEMENT_TRANSMISSION_LOSS,
    "aperture_stop": SINK_ELEMENT_TRANSMISSION_LOSS, "neutral_density": SINK_ATTENUATOR_ABSORPTION,
    "faraday_rotator": SINK_ELEMENT_TRANSMISSION_LOSS, "diffraction_grating": SINK_UNMODELLED_RESPONSE,
    "complex_transmission": SINK_ELEMENT_TRANSMISSION_LOSS,
}
#: Interaction kinds that send the beam they receive back on the side it came from, as a whole or in part.
REFLECTIVE_KINDS = frozenset({"mirror", "beam_splitter"})
# Lineage events that reverse the handedness of the transverse frame (a reflection).
#: Lineage events of total internal reflection: at an isotropic face, and into either wave of a crystal (``_crystal_event``).
TIR_EVENTS = frozenset({"tir", "tir_o", "tir_e"})
REFLECTION_EVENTS = frozenset({"reflect", "split_reflect", "fresnel_reflection", "coating_reflection", "grating_reflect",
                               "fresnel_reflection_o", "fresnel_reflection_e", "coating_reflection_o", "coating_reflection_e"}) | TIR_EVENTS
# Corners of the polygon around the image of the opening that cut an annulus (``Tracer._edge_inside``).
EDGE_POLYGON_CORNERS = 64
# The annulus rays show the sign of the paraxial map from the opening on a Gaussian axis when their offsets there and at
# the surface correlate at least this well (otherwise both signs are taken).
EDGE_SIGN_CORRELATION_MIN = 0.5
# Differential rays per packet: +-x, +-y (position) and +-a, +-b (direction) at the source plane.
DIFFERENTIAL_RAY_COUNT = 8
#: The geometric pupil-sample basis (``Tracer._sample_picture``) holds while the diffraction spread its rays do not follow
#: is at most this fraction of the radius they picture (``sqrt(2)`` times the live samples' rms radius about the chief ray,
#: a Gaussian's 1/e^2 radius): the beam's radius then exceeds the rays' picture by at most 3.1 % (``sqrt(1 + 0.25^2)``),
#: which moves the power an edge passes by at most about 2.3 % of the beam (``2 x^2 exp(-x^2) <= 0.74`` per unit of
#: relative scale, ``x`` the edge in radii), below the sampling resolution of a cut through the default sampling (one ring
#: of 12 samples in 96). For a collimated beam of radius ``w`` over a path ``L``: a Fresnel number ``w^2 / (lambda L)`` of
#: about 1.3 or more.
GEOMETRIC_DIFFRACTION_RATIO = 0.25
#: sqrt(2): a Gaussian's 1/e^2 radius over the rms radius of its power about its centre.
SQRT_TWO = math.sqrt(2.0)
# States of a pupil sample's cell at a cut (``Tracer._cell_grid``): retired earlier, passing, cut; no neighbour there.
_GONE, _PASSED, _TAKEN, _NONE = 0, 1, 2, 3
#: The bounded openings a packet keeps (``_Packet.openings``, the most recent ones): the light beyond their edges left it.
OPENINGS_KEPT = 4
#: The power of the extrapolation of the samples' affine map from their farthest sample to an opening's edge
#: (``Tracer._edge_held``): its residual there is taken to grow at most as the cube of the distance (third-order aberration).
EDGE_MAP_GROWTH = 3
#: An affine fit of pupil samples whose position covariance has ``det <= this x the product of its diagonal`` (samples near a
#: line) is not formed (``_affine_map``).
AFFINE_FIT_MIN_SPREAD = 1e-12


def _affine_map(x: Array, y: Array, u: Array, v: Array
                ) -> tuple[tuple[float, float, float, float], tuple[float, float], float, tuple[float, float], float] | None:
    """The least-squares affine map ``(x, y) -> (u, v)`` of the points (``M`` row-major, ``b``), its largest residual, the
    centroid of ``(x, y)`` and their farthest distance from it; ``None`` when the points lie on a line. Ordered sums only
    (``correctly_rounded_sum``): the same bits in every runtime."""
    n = float(x.size)
    mx, my = correctly_rounded_sum(x.tolist()) / n, correctly_rounded_sum(y.tolist()) / n
    mu, mv = correctly_rounded_sum(u.tolist()) / n, correctly_rounded_sum(v.tolist()) / n
    dx, dy, du, dv = x - mx, y - my, u - mu, v - mv
    sxx, syy, sxy = (correctly_rounded_sum((a * b).tolist()) for a, b in ((dx, dx), (dy, dy), (dx, dy)))
    sxu, syu, sxv, syv = (correctly_rounded_sum((a * b).tolist()) for a, b in ((dx, du), (dy, du), (dx, dv), (dy, dv)))
    det = sxx * syy - sxy * sxy
    if not det > AFFINE_FIT_MIN_SPREAD * (sxx * syy):
        return None
    m00, m01 = (syy * sxu - sxy * syu) / det, (sxx * syu - sxy * sxu) / det
    m10, m11 = (syy * sxv - sxy * syv) / det, (sxx * syv - sxy * sxv) / det
    ru, rv = du - (m00 * dx + m01 * dy), dv - (m10 * dx + m11 * dy)
    residual = float(np.sqrt(np.max(ru * ru + rv * rv)))
    far = float(np.sqrt(np.max(dx * dx + dy * dy)))
    return (m00, m01, m10, m11), (mu - (m00 * mx + m01 * my), mv - (m10 * mx + m11 * my)), residual, (mx, my), far


def _neighbours(grid: NDArray[Any], none: int) -> list[NDArray[Any]]:
    """For each cell of a pupil sampling's ring layout (``grid``: rings x points per ring, ``pupil.py``), its six neighbours'
    entries: the two beside it on its ring and the two nearest on the rings inside and outside it (ring ``k``'s point ``m``
    lies between the points ``m - 1`` and ``m`` of ring ``k + 1`` when ``k`` is even, ``m`` and ``m + 1`` when it is odd);
    ``none`` inside the innermost and outside the outermost ring."""
    rings = grid.shape[0]
    out = [np.roll(grid, 1, axis=1), np.roll(grid, -1, axis=1), *(np.full_like(grid, none) for _ in range(4))]
    for k in range(rings - 1):
        step = -1 if k % 2 == 0 else 1
        out[2][k], out[3][k] = grid[k + 1], np.roll(grid[k + 1], -step)
        out[4][k + 1], out[5][k + 1] = grid[k], np.roll(grid[k], step)
    return out


# First-order transfer is unavailable when a differential ray was retired before the plane.
TRANSFER_LOST = "a differential ray of the first-order transfer did not follow the chief ray to this plane"
TRANSFER_NOT_SAMPLED = "the trace ran without pupil / differential rays"
TRANSFER_AXICON = "the beam crossed an axicon (cone) surface, whose map of rays is not linear about the chief ray"
TRANSFER_FACETS = ("the beam crossed flat facets approximating a curved surface (TESSELLATED_FROM_CAD), whose map of rays is not "
                   "linear about the chief ray")
TRANSFER_PROMOTED = ("the chief ray fell outside a clear aperture upstream and a passing pupil ray carries the beam's reference from "
                     "there (reference_ray_promoted_at): no first-order transfer about the chief ray reaches this plane")
#: How far before the surface the differential rays are relaunched about a promoted reference ray (mm): far beyond
#: ``ray_min_advance_mm`` and short against any surface's curvature, so each meets the surface at the reference's hit.
RELAUNCH_BACKOFF_MM = 0.1


def _inverse4(m: list[list[float]]) -> list[list[float]] | None:
    """The inverse of a 4 x 4 matrix by Gauss-Jordan elimination with partial pivoting, in IEEE double operations in a
    fixed order (identical bits in every runtime; no LAPACK). ``None`` when a pivot is zero (singular)."""
    a = [list(row) + [1.0 if i == j else 0.0 for j in range(4)] for i, row in enumerate(m)]
    for col in range(4):
        pivot = max(range(col, 4), key=lambda r: abs(a[r][col]))
        if a[pivot][col] == 0.0:
            return None
        a[col], a[pivot] = a[pivot], a[col]
        p = a[col][col]
        a[col] = [x / p for x in a[col]]
        for r in range(4):
            if r != col and a[r][col] != 0.0:
                f = a[r][col]
                a[r] = [x - f * y for x, y in zip(a[r], a[col])]
    return [row[4:] for row in a]


@dataclass(frozen=True)
class PowerFlow:
    """Power leaving the traced beams at one place (``kind``: a ``SINK_*`` value). ``surface_id``: the scene surface
    (``None``: free space or the source); ``owner_id``: the component owning it; ``detail``: the termination, event or
    medium that caused it."""

    kind: str
    power_W: float
    source_id: str
    spectral_index: int
    wavelength_nm: float
    path_id: str | None
    surface_id: str | None
    owner_id: str | None
    detail: str


@dataclass(frozen=True)
class SourceInput:
    """Power launched by one source at one spectral line."""

    source_id: str
    owner_id: str
    spectral_index: int
    wavelength_nm: float
    power_W: float
    # a line a crystal generates (a second harmonic): the line it is generated from (``<source>#<index>``); its power is
    # the harmonic power generated, booked on that line as ``NONLINEAR_CONVERSION``
    generated_from: str | None = None


@dataclass
class ArrivalRecord:
    observation_id: str
    surface_id: str
    path_id: str
    source_id: str
    spectral_index: int
    wavelength_nm: float
    medium: str
    n: float
    position: Array
    direction: Array
    power_W: float
    opl_mm: float
    length_mm: float
    jones: CArray
    e1: Array
    e2: Array
    gstate: GState
    uv: tuple[float, float]
    aoi_deg: float
    pupil_uv: Array  # (M_alive, 2)
    pupil_total: int
    coherence_group: str
    coherence_length_mm: float | None
    lineage: tuple[str, ...]
    plane_normal: Array = field(default_factory=lambda: np.zeros(3))
    plane_u: Array = field(default_factory=lambda: np.zeros(3))
    plane_v: Array = field(default_factory=lambda: np.zeros(3))
    plane_origin: Array = field(default_factory=lambda: np.zeros(3))
    # The live pupil rays at the plane, in the order of ``pupil_uv``: hit points and directions (design frame) and launch
    # offsets in units of the source beam radii (``pupil_offsets(1, 1, ...)``: x along the source's e1, y along e2). The
    # Gaussian aberration check (``validation/observations.py``) compares their spread with the Gaussian beam.
    pupil_points: Array = field(default_factory=lambda: np.zeros((0, 3)))
    pupil_directions: Array = field(default_factory=lambda: np.zeros((0, 3)))
    pupil_launch: Array = field(default_factory=lambda: np.zeros((0, 2)))
    # Which crossing of this plane by this path it is (0 = first): a folded or double-pass path can cross a plane twice.
    crossing_index: int = 0
    # The chief ray's AOI is within the plane's acceptance cone (None: the plane declares none). A beam outside it is not
    # detected: on an absorbing plane it is stopped without being detected (``detected_power_W`` 0), and a transparent
    # plane receives no power from it.
    within_acceptance: bool | None = None
    # A crossing that does not count (``counted`` False) on a plane with an acceptance cone: does the beam's Gaussian tail on
    # the active area arrive inside the cone? The tail is off the chief ray, so it travels along the local wavefront normal
    # there, tilted from the chief by up to rho_max / |R| (rho_max: the farthest point of the area from the chief hit point,
    # R: the wavefront's radius of curvature), with a local angular spread whose 1/e^2 half-width is at most the far-field
    # half angle theta_0 (Gaussian; beyond aperture_quad_extent_sigmas of it the power is negligible, as for the beam's
    # spatial tails). ``False``: every direction within that bound of the chief is outside the cone (the tail is not
    # detected); ``True``: every one is inside; ``None``: not decided (no cone, no valid Gaussian state, or the bound
    # straddles the cone's edge) -- then the tail is taken as possibly arriving.
    tail_within_acceptance: bool | None = None
    # The chief ray fell on a central obstruction upstream: it is an unpowered reference axis of the annulus that continues
    # (the packet power is the annulus's; the Gaussian state is invalid; phase at the chief point is not defined).
    chief_obstructed: bool = False
    # Surfaces behind which the beam's power is out of model scope (``_Packet.power_scope``).
    power_scope: tuple[str, ...] = ()
    # Of those, the surfaces whose data the beam met outside its stated range, held at the nearest stated point (``_Packet.held``).
    held: tuple[HeldData, ...] = ()
    # Which polarization component of a partially polarized source's line the beam is (``SourceG.polarization_component``;
    # None: fully polarized light).
    polarization_component: int | None = None
    # Of those, the physical surfaces that part of the beam met before this plane on the chief ray's leg, and the one whose
    # reflection sent part of it beyond the plane at the leg's start (``_Packet.met_first`` / ``left_beyond``)
    met_first: tuple[str, ...] = ()
    left_beyond: tuple[str, ...] = ()
    # Whether the beam counts as reaching the plane's active area (always True without an active area): the chief ray is
    # inside it, or at least capture_min_power_fraction of the Gaussian power falls inside it. Only counted arrivals are
    # results; an absorbing plane absorbs only counted beams, the others pass it unchanged.
    counted: bool = True
    # A crossing that does not count: the fraction of the beam's power on the active area (its Gaussian tail; behind a
    # central obstruction that of the unobstructed envelope; 0.0 beyond aperture_quad_extent_sigmas beam radii), None when
    # it is not computed (no valid Gaussian state). A plane no traced beam reaches is dark only where this is negligible.
    area_fraction: float | None = None
    # Power an absorbing plane detects from this beam (the ledger's DETECTED): its power on the plane's active area, or the
    # whole beam without one; None on a transparent plane and for a beam an absorbing plane does not count.
    detected_power_W: float | None = None
    # A lower bound of ``detected_power_W`` when it is not exact (an annulus counted on an active area that may miss up to
    # ``capture_min_power_fraction`` of it); None when it is exact.
    detected_power_min_W: float | None = None
    parent_path_id: str | None = None
    # First-order ray-transfer matrix d(u, v, l_u, l_v)/d(x0, y0, a0, b0) from the source plane (None: ``transfer_issue``).
    transfer: Array | None = None
    transfer_issue: str | None = None
    # The per-ray optical path lengths are defined (False after an ideal thin lens, a Faraday rotator or a grating, whose
    # per-ray OPL is not modelled).
    ray_opl_defined: bool = True
    source_e1: Array = field(default_factory=lambda: np.zeros(3))
    source_e2: Array = field(default_factory=lambda: np.zeros(3))
    # The chief ray's group optical path (sum of n_g x length, mm; None: a medium's group index is not available), its group-
    # delay dispersion (sum of k2 x length, mm^2) and whether it crossed a dispersive medium (``_Packet.gpl``)
    group_opl_mm: float | None = None
    gdd_mm2: float = 0.0
    dispersive: bool = False
    # the passes on its way whose group delay the model does not state (``_Packet.delays``)
    delays: tuple[str, ...] = ()
    # Its power is not known: at most this (``_Packet.power_bound_W``; ``None``: ``power_W`` is the model's value), its
    # polarization is not known (``_Packet.polarization_open``), and the grating orders on its way: (surface id, order).
    power_bound_W: float | None = None
    polarization_open: bool = False
    grating_steps: tuple[tuple[str, int], ...] = ()
    # The plane passes through a physical surface's chief-ray hit point (``_Packet.coincidence``)
    coincidence: SurfaceCoincidenceRecord | None = None
    # The power is on the geometric pupil-sample basis (``_Packet.sampling``): its sampling resolution as a fraction of
    # ``power_W`` (``None``: the Gaussian model's power).
    sampling: float | None = None
    # The physical surfaces and absorbing planes that take a bounded share of the beam before this plane on the chief
    # ray's leg (``Tracer._taken_before``), and the most power they take together: the beam's crossing power lies in
    # ``[power_W - taken_upper_W, power_W]``; where a stopping surface certainly takes part of it, the least power they take
    # together (``taken_lower_W``): it lies in ``[power_W - taken_upper_W, power_W - taken_lower_W]``.
    taken_at: tuple[str, ...] = ()
    taken_upper_W: float | None = None
    taken_lower_W: float | None = None
    # The surfaces where the chief ray fell outside a clear aperture and a passing pupil sample became the reference ray
    # (``_Packet.reference_promoted``): position, direction, AOI and optical path are that ray's.
    reference_promoted: tuple[str, ...] = ()
    # The pupil samples do not follow the beam's diffraction from that surface on (``_Packet.samples_unresolved``)
    samples_unresolved: tuple[str, float, float, str] | None = None


@dataclass(frozen=True)
class UntracedBranch:
    """Light that leaves the traced beams by a model approximation but goes on physically (``UNTRACED_BRANCH_KINDS``).

    It starts at ``origin`` (the chief-ray hit on ``surface_id``) along ``direction`` with ``power_W``; ``gstate`` is its
    Gaussian beam there (the reflected beam of an untraced reflection, the incident beam otherwise), ``None`` when it is
    not known. ``seed`` is that light as a one-ray packet (chief ray, Jones vector, basis, Gaussian state, medium) ready
    to be followed by a probe trace (``Tracer.trace_probes``). It only decides whether a plane no traced beam reaches is
    dark (``validation/untraced.py``); it is never added to a result."""

    kind: str
    source_id: str
    spectral_index: int
    path_id: str
    surface_id: str
    origin: Array
    direction: Array
    gstate: GState | None
    power_W: float
    seed: Any = None  # _Packet


def _sag_range(surf: SurfaceG, rho_max: float) -> tuple[float, float]:
    """Bounds ``(z_lo, z_hi)`` of a sag profile over ``0 <= rho <= rho_max`` within its domain (``optics/surfaces/sag.py``,
    ``sag_bounds``; here with products and ``sqrt`` only, so the same bits in every runtime): the conic part is monotonic
    in ``rho``, each polynomial term at most ``|A| rho^(2j+4)``, the axicon term between 0 and ``L rho``; a two-variable
    shape by ``freeform.sag_range``."""
    if surf.freeform is not None:
        return freeform_sag_range(surf, 0.0, rho_max)
    k1c2 = (1.0 + surf.conic) * surf.curvature * surf.curvature
    rho = min(rho_max, 1.0 / math.sqrt(k1c2)) if k1c2 > 0.0 else rho_max
    if not math.isfinite(rho):
        return -math.inf, math.inf
    s = rho * rho
    big_s = math.sqrt(max(1.0 - k1c2 * s, 0.0))
    base = surf.curvature * s / (1.0 + big_s)
    poly, power = 0.0, s * s
    for coefficient in surf.coefficients:
        poly += abs(coefficient) * power
        power *= s
    cone = surf.linear * rho
    return min(0.0, base) + min(0.0, cone) - poly, max(0.0, base) + max(0.0, cone) + poly


def _support_rho(surf: SurfaceG, support: ApertureG) -> tuple[float, float]:
    """The range of the profile's ``rho`` (``sqrt(a^2 + b^2)``; a cylinder: ``|a|``) over a sag surface's support."""
    ou, p1 = support.offset_u, support.p1
    if surf.cylindrical:
        if support.shape in ("circle", "rectangle"):
            return max(abs(ou) - p1, 0.0), abs(ou) + p1
        u_lo, u_hi, _v_lo, _v_hi = apertures.bounds(support)
        return (0.0 if u_lo <= 0.0 <= u_hi else min(abs(u_lo), abs(u_hi))), apertures.far_u(support)
    return apertures.rho_range(support)


def _room_in_support(surf: SurfaceG, support: ApertureG, a: float, b: float) -> float:
    """How far the point ``(a, b)`` (across a sag surface's axis) lies inside its support and its profile's domain: the
    radius of the largest disc around it within both (0 outside)."""
    room = apertures.room(support, a, b)
    k1c2 = (1.0 + surf.conic) * surf.curvature * surf.curvature
    if k1c2 > 0.0:
        rho = abs(a) if surf.cylindrical else math.sqrt(a * a + b * b)
        room = min(room, 1.0 / math.sqrt(k1c2) - rho)
    return max(room, 0.0)


def _sag_window(surf: SurfaceG, a: float, b: float, radius: float, rho_lo: float, rho_hi: float) -> tuple[float, float]:
    """The range of the profile's ``rho`` over the points within ``radius`` of ``(a, b)`` across the axis, within the
    support's range ``[rho_lo, rho_hi]``."""
    rho = abs(a) if surf.cylindrical else math.sqrt(a * a + b * b)
    first, last = max(rho - radius, rho_lo, 0.0), min(rho + radius, rho_hi)
    return (first, last) if first <= last else (rho, rho)


def _sag_range_between(surf: SurfaceG, rho_first: float, rho_last: float) -> tuple[float, float]:
    """Bounds of a sag profile over ``rho_first <= rho <= rho_last`` within its domain: the conic part monotonic in ``rho``
    (between its values at the ends), each polynomial term at most ``|A| rho_last^(2j+4)``, the axicon term between
    ``L rho_first`` and ``L rho_last``; a two-variable shape by ``freeform.sag_range``."""
    if surf.freeform is not None:
        return freeform_sag_range(surf, rho_first, rho_last)
    k1c2 = (1.0 + surf.conic) * surf.curvature * surf.curvature
    domain = 1.0 / math.sqrt(k1c2) if k1c2 > 0.0 else math.inf
    rho_last = min(rho_last, domain)
    rho_first = min(rho_first, rho_last)
    if not math.isfinite(rho_last):
        return -math.inf, math.inf
    ends = []
    for rho in (rho_first, rho_last):
        s = rho * rho
        ends.append(surf.curvature * s / (1.0 + math.sqrt(max(1.0 - k1c2 * s, 0.0))))
    s = rho_last * rho_last
    poly, power = 0.0, s * s
    for coefficient in surf.coefficients:
        poly += abs(coefficient) * power
        power *= s
    cones = (surf.linear * rho_first, surf.linear * rho_last)
    return min(ends) + min(cones) - poly, max(ends) + max(cones) + poly


def _sag_hessian_bound(surf: SurfaceG, rho_first: float, rho_last: float) -> float:
    """A bound of the norm of a sag profile's Hessian (without its axicon term) over ``rho_first <= rho <= rho_last``: the
    conic part's principal values are ``c / S^3`` (radial) and ``c / S`` (across), ``S`` monotonic in ``rho``; the
    polynomial's at most ``sum |A_j| (2j+4)(2j+3) rho_last^(2j+2)``; a two-variable shape by ``freeform.hessian_bound``.
    ``inf`` at the edge of the profile's domain."""
    if surf.freeform is not None:
        return freeform_hessian_bound(surf, rho_first, rho_last)
    k1c2 = (1.0 + surf.conic) * surf.curvature * surf.curvature
    smallest = math.inf
    for rho in (rho_first, rho_last):
        smallest = min(smallest, math.sqrt(max(1.0 - k1c2 * rho * rho, 0.0)))
    if surf.curvature != 0.0 and smallest <= 0.0:
        return math.inf
    curvature = abs(surf.curvature)
    conic = curvature / (smallest * smallest * smallest) if curvature else 0.0
    conic = max(conic, curvature / smallest) if curvature else 0.0
    s = rho_last * rho_last
    poly, power = 0.0, s
    for j, coefficient in enumerate(surf.coefficients):
        poly += abs(coefficient) * (2 * j + 4) * (2 * j + 3) * power
        power *= s
    return conic + poly


def _profile_beyond(z: float) -> float:
    """The share of a Gaussian profile (density ``exp(-2 x^2)``, ``x`` in units of the 1/e^2 radius) beyond ``x = z``."""
    tail = 0.5 * erfc_float(math.sqrt(2.0) * abs(z))
    return tail if z >= 0.0 else 1.0 - tail


def _radius_ranges(q: complex, m2: float, scale: float, t0: Array, t1: Array) -> tuple[Array, Array]:
    """A Gaussian axis's narrowest and widest 1/e^2 radius over each stretch ``[t0, t1]`` from where its parameter is ``q``
    (element by element)."""
    ends = [np.sqrt(m2 * scale * ((q.real + t) * (q.real + t) + q.imag * q.imag) / q.imag) for t in (t0, t1)]
    waist = (t0 <= -q.real) & (-q.real <= t1)
    return np.where(waist, math.sqrt(m2 * scale * q.imag), np.minimum(ends[0], ends[1])), np.maximum(ends[0], ends[1])


def _box_terms(s: tuple[Array, Array], x: tuple[Array, Array], y: tuple[Array, Array]) -> tuple[Array, ...]:
    """The midpoints and half-widths ``(s, x, y)`` of the boxes ``s x x x y`` (element by element) that
    ``_linear_ranges_of`` weighs: computed once for every linear function weighed over the same boxes."""
    return (0.5 * (s[0] + s[1]), 0.5 * (x[0] + x[1]), 0.5 * (y[0] + y[1]),
            0.5 * (s[1] - s[0]), 0.5 * (x[1] - x[0]), 0.5 * (y[1] - y[0]))


def _linear_ranges_of(c: tuple[float, float, float, float], terms: tuple[Array, ...]) -> tuple[Array, Array]:
    """The exact ranges of ``c0 + cs s + cx x + cy y`` over the boxes ``s x x x y`` given by their ``_box_terms`` (element
    by element): the value at the box's centre minus and plus ``|cs| hs + |cx| hx + |cy| hy``."""
    mid_s, mid_x, mid_y, half_s, half_x, half_y = terms
    mid = c[0] + c[1] * mid_s + c[2] * mid_x + c[3] * mid_y
    half = abs(c[1]) * half_s + abs(c[2]) * half_x + abs(c[3]) * half_y
    return mid - half, mid + half


def _sag_ranges_between(surf: SurfaceG, rho_first: Array, rho_last: Array) -> tuple[Array, Array]:
    """``_sag_range_between`` element by element (a two-variable shape one element at a time)."""
    if surf.freeform is not None:
        pairs = [_sag_range_between(surf, float(a), float(b)) for a, b in zip(rho_first, rho_last)]
        return np.array([p[0] for p in pairs], dtype=float), np.array([p[1] for p in pairs], dtype=float)
    k1c2 = (1.0 + surf.conic) * surf.curvature * surf.curvature
    domain = 1.0 / math.sqrt(k1c2) if k1c2 > 0.0 else math.inf
    last = np.minimum(rho_last, domain)
    first = np.minimum(rho_first, last)
    ends = [surf.curvature * (r * r) / (1.0 + np.sqrt(np.maximum(1.0 - k1c2 * (r * r), 0.0))) for r in (first, last)]
    s = last * last
    poly, power = np.zeros_like(last), s * s
    for coefficient in surf.coefficients:
        poly = poly + abs(coefficient) * power
        power = power * s
    cones = (surf.linear * first, surf.linear * last)
    lo = np.minimum(ends[0], ends[1]) + np.minimum(cones[0], cones[1]) - poly
    hi = np.maximum(ends[0], ends[1]) + np.maximum(cones[0], cones[1]) + poly
    finite = np.isfinite(last)
    return np.where(finite, lo, -math.inf), np.where(finite, hi, math.inf)


def _sag_region_share(surf: SurfaceG, leg: Leg, k_sigma: float | None, cfg: NumericalConfig) -> float | None:
    """A bound of the fraction of a beam that may meet the sag surface ``surf`` before the plane ``leg`` ends at (its end
    plane: an observation plane the chief ray crosses first), from the surface itself rather than a plane standing for it.

    The beam's energy flows along lines ``xi w_x(s) b_x + eta w_y(s) b_y`` off the chief ray (``xi``, ``eta``: fixed places
    in its profile, density ``exp(-2 (xi^2 + eta^2))``); the parallel rays the other bounds read lie at ``xi W_x``,
    ``eta W_y`` (``W`` the widest radius on the stretch). For a cell of the profile and a stretch ``[s0, s1]`` of the chief
    ray every such point lies in the box of ``s``, ``x = xi w`` and ``y = eta w`` with ``w`` between the narrowest radius on
    the stretch and ``W``, and on that box the plane's signed distance and the surface's local coordinates ``(a, b, z)`` are
    linear: their exact ranges. A line of the cell may meet the surface before the plane there only if part of the box lies
    before the plane, its ``(a, b)`` range meets the surface's support and the profile's domain, and ``z`` minus the sag
    can vanish (the sag between its bounds over that range of ``rho``, ``_sag_range_between``). The profile within ``K``
    radii is cut into REGION_GRID x REGION_GRID cells; a cell is clear when every stretch of the chief ray, halved up to
    REGION_STRETCH_DEPTH times (only as far as a line of the cell may lie before the plane), fails a test; the others are
    split in four, up to REGION_CELL_DEPTH times, and those still not clear count whole, with the profile beyond ``K``
    radii. Every box is weighed element by element (the same bits in every runtime). ``None``: no bound this way (no stated
    ``K``, a leg without a finite end plane, an unbounded support, more than REGION_BOX_BUDGET boxes)."""
    support = support_of(surf)
    if k_sigma is None or support.shape == "none" or not math.isfinite(leg.length) or leg.end_offset is None:
        return None
    d, g = leg.d, leg.beam
    dn_t = float(dot(d, leg.end_normal))
    if abs(dn_t) <= cfg.parallel_dot_tol:
        return None
    sigma = 1.0 if dn_t > 0.0 else -1.0
    s_lo, s_hi = reach_stretch(leg, k_sigma, cfg)
    scale = g.wavelength_nm * NM_TO_MM / (math.pi * g.n)
    whole = (np.array([s_lo - leg.first]), np.array([s_hi - leg.first]))
    narrow_x_all, wide_x = (float(v[0]) for v in _radius_ranges(g.qx, g.m2x, scale, *whole))
    narrow_y_all, wide_y = (float(v[0]) for v in _radius_ranges(g.qy, g.m2y, scale, *whole))
    start, p_t = leg.start, leg.start + leg.end_offset * d
    rel = start - surf.origin

    def coefficients(axis: Array, base: float) -> tuple[float, float, float, float]:
        return (base, float(dot(axis, d)), float(dot(axis, g.bx)), float(dot(axis, g.by)))

    plane_c = coefficients(leg.end_normal, float(dot(leg.end_normal, start - p_t)))
    a_c = coefficients(surf.u_axis, float(dot(surf.u_axis, rel)))
    b_c = coefficients(surf.v_axis, float(dot(surf.v_axis, rel)))
    z_c = coefficients(surf.normal, float(dot(surf.normal, rel)))
    k1c2 = (1.0 + surf.conic) * surf.curvature * surf.curvature
    domain = 1.0 / math.sqrt(k1c2) if k1c2 > 0.0 and surf.freeform is None else math.inf
    tol = cfg.aperture_edge_tol_mm
    # the finest grid of the profile: cells are index ranges on it, their edges ``-K + i (2K / fine)``
    fine = REGION_GRID * 2 ** REGION_CELL_DEPTH
    edges = -k_sigma + np.arange(fine + 1) * (2.0 * k_sigma / fine)
    beyond = np.array([_profile_beyond(float(e)) for e in edges])

    def spans(lo: Array, hi: Array, narrow_r: Array, wide_r: float) -> tuple[Array, Array]:
        """The range of ``xi w`` for ``xi`` in ``[lo, hi]`` and ``w`` between ``narrow_r`` and ``wide_r``."""
        corners = (lo * narrow_r, lo * wide_r, hi * narrow_r, hi * wide_r)
        return (np.minimum(np.minimum(corners[0], corners[1]), np.minimum(corners[2], corners[3])),
                np.maximum(np.maximum(corners[0], corners[1]), np.maximum(corners[2], corners[3])))

    def may_meet(cells: NDArray[np.intp], s0: Array, s1: Array) -> NDArray[np.bool_]:
        """Whether a line of each cell (rows of index ranges ``i0, i1, j0, j1``) may meet the surface before the plane on
        its stretch ``[s0, s1]``."""
        narrow_x, _ = _radius_ranges(g.qx, g.m2x, scale, s0 - leg.first, s1 - leg.first)
        narrow_y, _ = _radius_ranges(g.qy, g.m2y, scale, s0 - leg.first, s1 - leg.first)
        x = spans(edges[cells[:, 0]], edges[cells[:, 1]], narrow_x, wide_x)
        y = spans(edges[cells[:, 2]], edges[cells[:, 3]], narrow_y, wide_y)
        terms = _box_terms((s0, s1), x, y)
        lo, hi = _linear_ranges_of(plane_c, terms)
        before = (lo if sigma > 0.0 else -hi) <= tol
        a_lo, a_hi = _linear_ranges_of(a_c, terms)
        b_lo, b_hi = _linear_ranges_of(b_c, terms)
        if support.shape == "circle":
            du = np.maximum(np.maximum(support.offset_u - a_hi, a_lo - support.offset_u), 0.0)
            dv = np.maximum(np.maximum(support.offset_v - b_hi, b_lo - support.offset_v), 0.0)
            on_support = du * du + dv * dv <= (support.p1 + tol) * (support.p1 + tol)
        elif support.shape == "rectangle":
            on_support = ((a_hi >= support.offset_u - support.p1 - tol) & (a_lo <= support.offset_u + support.p1 + tol)
                          & (b_hi >= support.offset_v - support.p2 - tol) & (b_lo <= support.offset_v + support.p2 + tol))
        else:
            on_support = np.ones(s0.shape, dtype=bool)
        straddles_a = (a_lo <= 0.0) & (a_hi >= 0.0)
        near_a = np.where(straddles_a, 0.0, np.minimum(np.abs(a_lo), np.abs(a_hi)))
        far_a = np.maximum(np.abs(a_lo), np.abs(a_hi))
        if surf.cylindrical:
            rho_lo, rho_hi = near_a, far_a
        else:
            straddles_b = (b_lo <= 0.0) & (b_hi >= 0.0)
            near_b = np.where(straddles_b, 0.0, np.minimum(np.abs(b_lo), np.abs(b_hi)))
            far_b = np.maximum(np.abs(b_lo), np.abs(b_hi))
            rho_lo, rho_hi = np.sqrt(near_a * near_a + near_b * near_b), np.sqrt(far_a * far_a + far_b * far_b)
        in_domain = rho_lo <= domain + tol
        sag_lo, sag_hi = _sag_ranges_between(surf, rho_lo, rho_hi)
        z_lo, z_hi = _linear_ranges_of(z_c, terms)
        return before & on_support & in_domain & (z_lo - sag_hi <= tol) & (z_hi - sag_lo >= -tol)

    def last_before(cells: NDArray[np.intp]) -> Array:
        """The farthest chief-ray parameter at which a line of each cell may still lie before the plane (its points over
        the whole stretch: the plane's signed distance grows along the chief ray)."""
        x = spans(edges[cells[:, 0]], edges[cells[:, 1]], np.full(len(cells), narrow_x_all), wide_x)
        y = spans(edges[cells[:, 2]], edges[cells[:, 3]], np.full(len(cells), narrow_y_all), wide_y)
        across = sigma * (plane_c[2] * (0.5 * (x[0] + x[1])) + plane_c[3] * (0.5 * (y[0] + y[1])))
        spread = abs(plane_c[2]) * (0.5 * (x[1] - x[0])) + abs(plane_c[3]) * (0.5 * (y[1] - y[0]))
        return (tol - sigma * plane_c[0] - across + spread) / abs(plane_c[1])

    weighed = 0

    def open_cells(cells: NDArray[np.intp]) -> NDArray[np.bool_] | None:
        """Which cells are not clear: a stretch of the chief ray on which a line of the cell may meet the surface first."""
        nonlocal weighed
        found = np.zeros(len(cells), dtype=bool)
        last = last_before(cells)
        rows = np.arange(len(cells))
        s0, s1 = np.full(len(cells), s_lo), np.full(len(cells), s_hi)
        for depth in range(REGION_STRETCH_DEPTH + 1):
            keep = (s0 <= last[rows]) & ~found[rows]
            rows, s0, s1 = rows[keep], s0[keep], s1[keep]
            weighed += len(rows)
            if weighed > REGION_BOX_BUDGET:
                return None
            if not len(rows):
                break
            meets = may_meet(cells[rows], s0, s1)
            rows, s0, s1 = rows[meets], s0[meets], s1[meets]
            if depth == REGION_STRETCH_DEPTH:
                found[rows] = True
                break
            middle = 0.5 * (s0 + s1)
            rows = np.repeat(rows, 2)
            s0, s1 = np.stack([s0, middle], axis=1).ravel(), np.stack([middle, s1], axis=1).ravel()
        return found

    step = 2 ** REGION_CELL_DEPTH
    ii, jj = np.meshgrid(np.arange(REGION_GRID) * step, np.arange(REGION_GRID) * step, indexing="ij")
    cells = np.stack([ii.ravel(), ii.ravel() + step, jj.ravel(), jj.ravel() + step], axis=1).astype(np.intp)
    for level in range(REGION_CELL_DEPTH + 1):
        found = open_cells(cells)
        if found is None:
            return None
        cells = cells[found]
        if level == REGION_CELL_DEPTH or not len(cells):
            break
        half = (cells[:, 1] - cells[:, 0]) // 2
        i_mid, j_mid = cells[:, 0] + half, cells[:, 2] + half
        cells = np.concatenate([np.stack([cells[:, 0], i_mid, cells[:, 2], j_mid], axis=1), np.stack([i_mid, cells[:, 1], cells[:, 2], j_mid], axis=1),
                                np.stack([cells[:, 0], i_mid, j_mid, cells[:, 3]], axis=1), np.stack([i_mid, cells[:, 1], j_mid, cells[:, 3]], axis=1)])
    shares = [(float(beyond[i0]) - float(beyond[i1])) * (float(beyond[j0]) - float(beyond[j1])) for i0, i1, j0, j1 in cells.tolist()]
    inside = 1.0 - 2.0 * _profile_beyond(k_sigma)
    return min(1.0, correctly_rounded_sum(shares) + (1.0 - inside * inside))


@dataclass(frozen=True)
class ProbeOutcome:
    """What a probe trace of one ``UntracedBranch`` did: its crossings of observation planes (counted or not) and where its
    power left the probe's beams (``traced`` False: the probe budget was spent before it, nothing is known)."""

    traced: bool
    arrivals: tuple[ArrivalRecord, ...] = ()
    flows: tuple[PowerFlow, ...] = ()
    not_followed_W: float = 0.0  # its descendants' untraced branches the budget did not reach (their power at their origin)
    not_followed: int = 0
    # its (and its descendants') arrivals at physical surfaces (the exposure of the surfaces by untraced light)
    surface_hits: tuple[SurfaceHit, ...] = ()
    # planes part of its beams crosses where their chief rays do not (``PartialCrossing``)
    partial_crossings: tuple[PartialCrossing, ...] = ()


#: An angle of incidence (deg) below which an undetermined pass's record prints 0 (``coherence.delay_key``, 1e-6 deg): a
#: coating met there treats every polarization state alike, so its record carries none.
NORMAL_INCIDENCE_RECORD_DEG = 5e-7


@dataclass
class TraceOutput:
    paths: list[OpticalPath] = field(default_factory=list)
    arrivals: list[ArrivalRecord] = field(default_factory=list)
    issues: list[ValidationIssue] = field(default_factory=list)
    trace_events: list[dict[str, Any]] = field(default_factory=list)  # canonical event list for trace_hash
    power_flows: list[PowerFlow] = field(default_factory=list)
    inputs: list[SourceInput] = field(default_factory=list)
    untraced_branches: list[UntracedBranch] = field(default_factory=list)
    # every beam's meeting with a scattering surface, in trace order (``optics/rays/scatter.py``)
    scatter_origins: list[ScatterOrigin] = field(default_factory=list)
    # every traced packet's arrival at a physical surface (``optics/rays/hits.py``; a probe trace's in ``ProbeOutcome``)
    surface_hits: list[SurfaceHit] = field(default_factory=list)
    # transparent planes part of a traced beam crosses on a leg its chief ray does not cross them on (``PartialCrossing``)
    partial_crossings: list[PartialCrossing] = field(default_factory=list)
    # per source line (source_id, spectral_index): the total power of the branches of light the trace left out as negligible
    # for its line (each at most ``negligible_power_fraction`` x the line input, or the line's ``follow_W`` level; not
    # followed): wherever it goes, its field may still add to the light of its line (the exposures bound it by its
    # diffraction limit, ``validation/exposures.py``)
    left_out_W: dict[tuple[str, int], float] = field(default_factory=dict)
    # per source line: the sum of the square roots of those branches' powers (sqrt(W)) -- the branches of a line are
    # mutually coherent, so where they meet their fields add in amplitude: the bounds use this sum, not the total power
    left_out_root: dict[tuple[str, int], float] = field(default_factory=dict)
    # with ``Tracer(keep_faint=True)``: those left-out branches themselves, with their seeds (a deeper probe pass can follow
    # them; ``validation/untraced.py``: ``follow_untraced(levels=...)``)
    faint_branches: list[UntracedBranch] = field(default_factory=list)
    # every order of every traced beam's pass through a diffraction grating
    grating_orders: list[GratingOrderResult] = field(default_factory=list)
    # every traced beam's pass into a crystal that generates its second harmonic
    nonlinear_conversions: list[NonlinearConversionResult] = field(default_factory=list)


@dataclass(frozen=True)
class TakenShares:
    """What ``Tracer._taken_before`` found for one plane: the surfaces behind which the plane's power is out of model scope
    (``scope``; of them the reflecting start surface, ``beyond``), or the surfaces that take a bounded share of the beam
    before it (``bounded``) and the most power they take together (``upper_W``), and the least (``lower_W``: ``None`` where
    none is certain)."""

    scope: tuple[str, ...]
    beyond: tuple[str, ...]
    bounded: tuple[str, ...] = ()
    upper_W: float | None = None
    lower_W: float | None = None


@dataclass(frozen=True)
class _Opening:
    """The opening that cut an annulus, seen from the chief ray where it did, on the Gaussian axes ``bx``, ``by`` there:
    its radius and centre (``centre``: (x, y) of the opening's centre relative to the chief ray), the envelope's radii
    there, and each annulus ray's offset from the chief ray there (rows of the packet; NaN for the other rays)."""

    radius: float
    centre: tuple[float, float]
    beam_radii: tuple[float, float]
    ray_offsets: Array


@dataclass(frozen=True)
class _Passed:
    """A bounded opening the packet's chief ray passed (a clear aperture, in that surface's ``(u, v)``), with where each of
    its live pupil samples crossed it (rows by source sample, ``pupil_offsets`` order; NaN where a sample was not live): the
    light beyond the opening's edge left the beam there (``Tracer._edge_held``)."""

    surface_id: str
    aperture: Any
    uv: Array


@dataclass(frozen=True)
class _OrderShare:
    """What one order of a grating pass receives: ``fraction`` of the incident power (``None``: not known, at most ``bound``
    of it), with the TE and TM efficiencies it scales the field's parts by; ``stated``: a table's value that is not this
    beam's share (an unpolarized mean)."""

    fraction: float | None = None
    bound: float = 1.0
    te: float = 0.0
    tm: float = 0.0
    polarization_open: bool = False
    stated: float | None = None


@dataclass(frozen=True)
class _GratingShares:
    """What a grating's response says of one pass. ``reading`` (with the beam's TE and TM power fractions ``f_s``, ``f_p``):
    the orders' powers are known; without it they are not, and ``bounds`` holds what the table bounds per order.
    ``relative``: a relative table's values. ``absorbed``: the share the table says the grating absorbs."""

    reading: GratingReading | None = None
    f_s: float = 1.0
    f_p: float = 0.0
    polarization_open: bool = False
    absorbed: float = 0.0
    bounds: dict[int, float] = field(default_factory=dict)
    stated: dict[int, float] = field(default_factory=dict)
    relative: dict[int, float] = field(default_factory=dict)

    def of(self, m: int) -> _OrderShare:
        if self.reading is None:
            return _OrderShare(bound=self.bounds.get(m, 1.0), stated=self.stated.get(m))
        if m not in self.reading.efficiency_te:  # an order the table does not represent: at most what it leaves
            listed = correctly_rounded_sum(self.f_s * self.reading.efficiency_te[k] + self.f_p * self.reading.efficiency_tm[k] for k in self.reading.efficiency_te)
            return _OrderShare(bound=max(1.0 - listed - self.absorbed, 0.0))
        te, tm = self.reading.efficiency_te[m], self.reading.efficiency_tm[m]
        return _OrderShare(fraction=self.f_s * te + self.f_p * tm, te=te, tm=tm, polarization_open=self.polarization_open)


@dataclass(frozen=True)
class _CrystalLeg:
    """A beam travelling in a uniaxial crystal as one of its two waves (``optics/polarization/uniaxial.py``): which wave, the
    crystal at the beam's wavelength, each ray's wave vector and phase index along its ray (its optical path per mm), the
    group index and dispersion along the chief ray, the error per mm of reduced Gaussian path of carrying the beam with one
    diffraction index, and the chief ray's path length where the leg began."""

    mode: str
    crystal: Crystal
    wave: Array
    ray_index: Array
    group: tuple[float, float] | None
    gaussian_error_per_mm: float
    start_length: float

    def rows(self, rows: NDArray[np.intp]) -> _CrystalLeg:
        return replace(self, wave=self.wave[rows], ray_index=self.ray_index[rows])


@dataclass(frozen=True)
class _Shadow:
    """The element whose back the chief ray met inside the beam (``Tracer._obstruct``): the plane across the chief ray at that
    point (``origin``, unit axes ``ax``, ``ay``), the element's support outline projected along the chief ray onto it (``ring``,
    plane coordinates, counter-clockwise), the most that projection may be off by (``margin``, mm: a curved element's sag), and
    the annulus rays' positions on that plane (``ray_points``, plane coordinates; NaN: not an annulus ray). Behind it a hole
    of a later surface passes nothing of the annulus where the rays' affine map from this plane carries the whole hole inside
    the shadow (``Tracer._holes_in_shadow``). ``holed``: the element's support has holes of its own; light passing them goes on
    inside that outline, so no later hole is taken as lying in the shadow."""

    surface_id: str
    origin: Array
    ax: Array
    ay: Array
    ring: Array
    margin: float
    ray_points: Array
    holed: bool = False


@dataclass
class _Packet:
    path_id: str
    src: SourceG
    spectral_index: int
    wavelength_nm: float
    power0_W: float
    medium: str
    n: float
    o: Array
    d: Array
    e1: Array
    e2: Array
    jones: CArray
    alive: NDArray[np.bool_]
    opl: Array
    length: Array
    gstate: GState
    lineage: list[str]
    parent_path_id: str | None = None
    branch_surface_id: str | None = None
    branch_kind: str | None = None
    segments: list[RaySegment] = field(default_factory=list)
    interactions: int = 0
    children: int = 0
    pupil_total: int = 0  # pupil samples: rays 1 .. pupil_total
    differential: int = 0  # differential rays: the last ``differential`` rays (0 or DIFFERENTIAL_RAY_COUNT)
    opl_undefined: bool = False  # a thin lens, Faraday rotator or grating was crossed (per-ray OPL undefined from there on)
    crossings: dict[str, int] = field(default_factory=dict)  # observation id -> crossings of it by this path so far
    # Observation planes through the hit point of the physical surface just met (scene index, sign of the incident
    # chief direction along the plane normal, that surface's scene index): a beam leaving that surface back toward the
    # incident side crosses them again there (``Tracer._pending_crossings``).
    pending_observers: tuple[tuple[int, float, int], ...] = ()
    # What the plane being crossed now samples at a physical surface's hit point, for its arrival record (the crossing
    # sets it, and clears it after the plane has sampled): which surface, and the beam arriving at it or returning from it.
    coincidence: SurfaceCoincidenceRecord | None = None
    # Observation planes (scene indices) the chief ray crossed since the packet last met a physical surface (or started). The
    # rays stay where they left that surface until the chief ray meets the next one (``Tracer._observation_event``): these
    # planes may still lie ahead of them; they are not met again on the leg and do not stop them (``Tracer._stays``).
    leg_crossed: tuple[int, ...] = ()
    # Observation planes already recorded as crossed by part of the beam on this leg (``Tracer._leg_end``).
    leg_partial: tuple[int, ...] = ()
    # How far along the leg (from where it began, along the chief ray) the chief ray's last event is: the observation planes
    # it crossed sample the rays without moving them (``Tracer._observation_event``), so every ray stays where the leg began.
    leg_t: float = 0.0
    # The chief ray fell on a central obstruction: from there it is a reference axis (no light travels along it in the
    # geometric model), the packet carries the annulus's power and its live pupil rays are the annulus (``_pass_obstruction``).
    # ``gstate`` is then the *unobstructed* beam's Gaussian state, carried on only as the envelope that bounds where the
    # annulus can be (an arrival reports it out of model scope, ``GAUSSIAN_CENTRAL_OBSTRUCTION``).
    chief_obstructed: bool = False
    # Behind a central obstruction: the share of the incident beam the annulus carried there (its power / the unobstructed
    # beam's), and the radius of the opening that cut it, in units of the envelope's radii per Gaussian axis (the opening
    # edge maps paraxially inside that many envelope radii everywhere downstream: ``Tracer._edge_inside``).
    annulus_fraction: float = 1.0
    opening: _Opening | None = None
    # An untraced branch of light passing beside an element (beyond its support): it does not meet that element's surfaces
    # until it has met another physical surface (``Tracer._trace``); never inherited by a fork.
    passes_owner: str | None = None
    # Surfaces behind which the packet's power is out of model scope (a variable attenuator whose density varies across the
    # beam: ``ND_DENSITY_VARIES_ACROSS_BEAM``); inherited by every branch it makes (``ArrivalResult.power_out_of_scope_at``).
    power_scope: tuple[str, ...] = ()
    # Surfaces whose coating the packet met outside the wavelengths or angles its data covers: the trace went on with the
    # data's values at the nearest stated point (``Tracer._live_coating``), each such surface is in ``power_scope``, and the
    # powers behind it say so in their ``value_basis``; inherited by every branch it makes.
    held: tuple[HeldData, ...] = ()
    # the physical surfaces part of the beam met before the observation plane being sampled (``_taken_before``; that sample
    # only, never forked), and the surface whose reflection sent part of it beyond that plane at the leg's start
    met_first: tuple[str, ...] = ()
    left_beyond: tuple[str, ...] = ()
    # The chief ray's group optical path from its source (sum of n_g x length over the media it crossed, mm; ``None`` when a
    # medium's group index is not available), its group-delay dispersion (sum of k2 x length, mm^2) and whether it crossed a
    # dispersive medium (n_g != n): two beams' degree of coherence follows their relative group delay (``SurfaceHit``)
    gpl: float | None = 0.0
    gdd: float = 0.0
    dispersive: bool = False
    # how much the envelope delay can vary over the beam's ray families (``optics/rays/envelope.py``; ``None``: not bounded
    # -- it crossed a dispersive medium other than between two parallel planes), the plane where its present segment began
    # (its unit normal; ``None``: a curved surface) and the chief ray's length since
    spread: Spread | None = NO_SPREAD
    segment_normal: tuple[float, float, float] | None = None
    # the curved surface (scene index) the present segment began at (``None``: a plane, the source, or not known), and the
    # physical surface it began at (its id; ``None``: the source, or not known)
    segment_surface: int | None = None
    segment_start: str | None = None
    segment_mm: float = 0.0
    # which of the source's pupil samples the packet's pupil rays are (``pupil_offsets`` order, rays 1 .. pupil_total;
    # ``None``: all of them in order): a probe keeps two whole rings of its parent's (``Tracer._branch``)
    launch_rows: NDArray[np.intp] | None = None
    # the passes on its way whose group delay the model does not state (``optics/interference/coherence.py``: an element
    # without a thickness, a coating without a stated constant phase), as ``delay_key`` records
    delays: tuple[str, ...] = ()
    # A grating order whose power is not known: ``jones`` is zero and this is the most power it can carry (the power incident
    # on the grating, or the share its response bounds). Unknown bulk loss instead keeps Jones as the *unattenuated upper*
    # field for downstream interactions and geometry; its actual amplitude and power remain unknown.
    power_bound_W: float | None = None
    bulk_loss_unknown: bool = False
    # Missing absorption in a nonlinear crystal leaves the pump split between wavelengths unknown. The packet's bound then
    # uses the incident power without tightening it by the attenuation along a pump ray that never changes wavelength.
    conversion_loss_unknown: bool = False
    # The order's power is known but not its polarization (a TE / TM response leaves the relative phase of a mixed state open:
    # ``jones`` holds the right magnitudes with an arbitrary relative phase): a surface whose share depends on it opens its power.
    polarization_open: bool = False
    # the grating orders on its way: (surface id, order)
    grating_steps: tuple[tuple[str, int], ...] = ()
    # The packet's power is on the geometric pupil-sample basis (``Tracer._geometric_cut``): a clear aperture cut its pupil
    # rays while it had no valid Gaussian state, and its power is the incident power times the share of its equal-power
    # pupil samples that passed. The value is the sampling resolution of that power, as a fraction of it (``None``: the
    # Gaussian model's power); inherited by every branch it makes.
    sampling: float | None = None
    # The surfaces that take a bounded share of the beam before the observation plane being sampled, and the most power
    # they take together, and the least where it is certain (``Tracer._taken_before``; that sample only, never forked)
    taken_at: tuple[str, ...] = ()
    taken_upper_W: float | None = None
    taken_lower_W: float | None = None
    # In a uniaxial crystal: the wave the beam travels as (``None``: an isotropic medium, where ``n`` is the medium's index);
    # ``n`` is then the chief ray's phase index along its ray. ``gaussian_error_mm``: the reduced q-parameter error the crystal
    # legs behind it accumulated (``optics/polarization/uniaxial.py``, ``birefringent_gaussian_rel_tol``).
    crystal: _CrystalLeg | None = None
    gaussian_error_mm: float = 0.0
    # The far-field half angles (rad, M^2 included) of the Gaussian beam the pupil rays were launched from, along the
    # source's x / y axes, at the source plane: ``lambda M^2 / (pi n w)``, ``w`` the beam radius there (``None``: no pupil
    # rays launched). The rays are the beam's energy-flow tangents at the source; through a paraxial system of ray-transfer
    # block ``B`` the beam's radius is ``sqrt(w_rays^2 + (B theta)^2)``, so ``B theta`` is the diffraction spread the rays
    # do not follow (``Tracer._sample_picture``).
    launch_spread: tuple[float, float] | None = None
    # The source's pupil-sample cells (``pupil_offsets`` order) an edge may already have cut without cutting the sample
    # (``Tracer._frontier_cells``): their shares are in ``sampling`` once, and a later edge near them adds nothing more.
    edge_cells: NDArray[np.bool_] | None = None
    # The bounded openings its chief ray passed, the most recent last (``OPENINGS_KEPT``; ``_Passed``): the light beyond
    # their edges left the beam there, so a later edge that holds an opening's image cuts nothing (``Tracer._edge_held``).
    # Inherited by a fork; not by an untraced branch (light beside an element did not pass its opening).
    openings: tuple[_Passed, ...] = ()
    # The pupil samples this step retired because another planar patch of the facet surface the chief ray meets lies first
    # on their way (``Tracer._facet_edge``: their light falls beyond the patch's edge); this step only, never forked.
    facet_diverged: int = 0
    # The surfaces where the chief ray fell outside the clear aperture while pupil samples passed (``Tracer._promote_reference``):
    # from the first of them, row 0 is not the source's chief ray but the passing sample nearest the centroid of those that
    # passed, which carries the beam's reference geometry (position, direction, optical path, the surface sequence).
    reference_promoted: tuple[str, ...] = ()
    # Behind a promoted reference: the 4 x 4 matrix ``P`` that turns the relaunched differential rays' output differences into
    # the first-order transfer from the source plane (``D_out P``: ``P = D_in^-1 M_1``, ``M_1`` the transfer up to the
    # promotion, ``D_in`` the relaunched rays' input differences there; ``Tracer._first_order``). ``None``: the source's own
    # differential rays (no promotion), or no transfer behind the promotion.
    transfer_base: Array | None = None
    # The element whose back the chief ray met inside the beam, the annulus passing beside it (``Tracer._obstruct``)
    shadow: _Shadow | None = None
    # The crystal faces where this light generated a second harmonic that travels on with it (``Tracer._second_harmonic``):
    # a later conversion of it would add coherently to that harmonic. Inherited by a fork.
    harmonic_at: tuple[str, ...] = ()
    # One of two crystal waves (ordinary, extraordinary) a face split this light into, both carrying more than the line's
    # negligible level (``Tracer._do_crystal_face``): the two are one coherent field until something separates them, and a
    # later face that splits this wave again into two waves mixes them. Inherited by a fork.
    coherent_twin: bool = False
    # The bounded shares surfaces took of the beam on the legs behind it: (surface id, the most power it took there in W,
    # the leg -- ``interactions`` when it was taken). A surface beside the chief ray the beam's tail may meet
    # (``Tracer._tail_beside_chief``) or pupil rays meet first (``_met_beside_chief``), an absorbing plane part of the beam
    # crosses (``_leg_end``). The beam's power follows its chief ray and still counts that light; every later element is
    # passive, so the light missing behind any of them is at most the light taken: each bound stays in W, unscaled, and
    # is inherited whole by every branch. An observation plane reads the beam on ``[P - S, P]`` (``_do_observe``).
    taken_path: tuple[tuple[str, float, int], ...] = ()
    # Where its pupil samples first did not follow the beam's diffraction within reach of an edge (``Tracer._not_computed``):
    # (surface id, the spread beyond them in mm, the most spread they follow there in mm, the code of the beam's lost state).
    # Its rays go on as geometry; an arrival behind states its sample radius as an estimate. Inherited by a fork.
    samples_unresolved: tuple[str, float, float, str] | None = None

    def fork(self, path_id: str, surface_id: str, kind: str) -> _Packet:
        return _Packet(
            path_id, self.src, self.spectral_index, self.wavelength_nm, self.power0_W, self.medium, self.n,
            self.o.copy(), self.d.copy(), self.e1.copy(), self.e2.copy(), self.jones.copy(), self.alive.copy(),
            self.opl.copy(), self.length.copy(), self.gstate, list(self.lineage), self.path_id, surface_id, kind,
            [], self.interactions, 0, self.pupil_total, self.differential, self.opl_undefined,
            pending_observers=self.pending_observers, leg_crossed=self.leg_crossed, chief_obstructed=self.chief_obstructed,
            annulus_fraction=self.annulus_fraction, opening=self.opening, power_scope=self.power_scope, held=self.held,
            gpl=self.gpl, gdd=self.gdd, dispersive=self.dispersive, spread=self.spread, segment_normal=self.segment_normal, segment_surface=self.segment_surface, segment_start=self.segment_start,
            segment_mm=self.segment_mm, launch_rows=self.launch_rows, delays=self.delays, power_bound_W=self.power_bound_W,
            bulk_loss_unknown=self.bulk_loss_unknown, conversion_loss_unknown=self.conversion_loss_unknown,
            polarization_open=self.polarization_open, grating_steps=self.grating_steps, sampling=self.sampling, crystal=self.crystal,
            gaussian_error_mm=self.gaussian_error_mm, launch_spread=self.launch_spread, edge_cells=self.edge_cells,
            openings=self.openings, reference_promoted=self.reference_promoted, transfer_base=self.transfer_base,
            shadow=self.shadow, harmonic_at=self.harmonic_at, coherent_twin=self.coherent_twin,
            taken_path=self.taken_path, samples_unresolved=self.samples_unresolved,
        )

    def take(self, surface_id: str, bound_W: float) -> None:
        """A surface took at most ``bound_W`` of the beam beside its chief ray on the present leg (``taken_path``)."""
        self.taken_path = (*self.taken_path, (surface_id, bound_W, self.interactions))

    @property
    def taken_path_at(self) -> tuple[str, ...]:
        """The surfaces of ``taken_path``, each once, in the order they took their shares."""
        return tuple(dict.fromkeys(entry[0] for entry in self.taken_path))

    def scope_behind(self) -> tuple[str, ...]:
        """The surfaces behind which the beam's power is not the light's: out of model scope (``power_scope``) or less a
        bounded share (``taken_path_at``). What reads the beam's profile or peak there states no value behind either."""
        return (*self.power_scope, *(at for at in self.taken_path_at if at not in self.power_scope))

    def pupil_slice(self) -> slice:
        return slice(1, 1 + self.pupil_total)

    def chief_power(self) -> float:
        return self.power0_W * power2(self.jones[0])  # (no fused multiply-add: identical bits in every runtime)


def _to_scene(grin: Any, local: Array, *, point: bool = True) -> Array:
    """Local gradient-index coordinates (transverse ``ex``, ``ey``, axial; last axis) in the scene frame (a point, or a direction)."""
    out = local[..., 0:1] * grin.ex + local[..., 1:2] * grin.ey + local[..., 2:3] * grin.axis
    return grin.origin + out if point else out


def _inside_by(points: Array, ring: Array, margin: float) -> bool:
    """Is every point inside the counter-clockwise polygon ``ring`` and farther than ``margin`` from its edges?"""
    count = ring.shape[0]
    for x, y in points:
        crossings = 0
        for k in range(count):
            (x0, y0), (x1, y1) = ring[k - 1], ring[k]
            if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
                crossings += 1
            ex, ey = x1 - x0, y1 - y0
            length2 = ex * ex + ey * ey
            s = 0.0 if length2 == 0.0 else min(1.0, max(0.0, ((x - x0) * ex + (y - y0) * ey) / length2))
            dx, dy = x - (x0 + s * ex), y - (y0 + s * ey)
            if dx * dx + dy * dy <= margin * margin:
                return False
        if crossings % 2 == 0:
            return False
    return True


def _support_side(surf: SurfaceG, point: Array, normal: Array, side: float, tol: float) -> int:
    """Where a plane surface's support lies against the plane through ``point`` with unit ``normal`` (``side``: the sign of
    the light's side): ``1`` wholly on the far side (``side * (p - point) . normal > tol`` for every point), ``-1`` wholly on
    the near side, ``0`` otherwise. Exact for a circle and a rectangle (their extent along the normal), the circle about the
    offset that holds any other bounded shape; an unbounded support is ``0``."""
    support = support_of(surf)
    if not apertures.bounded(support):
        return 0
    centre = surf.origin + support.offset_u * surf.u_axis + support.offset_v * surf.v_axis
    a_u, a_v = abs(float(dot(surf.u_axis, normal))), abs(float(dot(surf.v_axis, normal)))
    if support.shape == "rectangle":
        half = support.p1 * a_u + support.p2 * a_v
    else:
        half = (support.p1 if support.shape == "circle" else apertures.reach(support)) * math.sqrt(a_u * a_u + a_v * a_v)
    at = side * float(dot(centre - point, normal))
    return 1 if at - half > tol else -1 if at + half < -tol else 0


def _is_round_state(state: GState, cfg: NumericalConfig) -> bool:
    """Both axes of a Gaussian state carry the same ``q`` and ``M^2`` (``gaussian_round_tol``): it has no preferred axes."""
    round_q = abs(state.qx - state.qy) <= cfg.gaussian_round_tol * max(abs(state.qx), abs(state.qy))
    round_m2 = abs(state.m2x - state.m2y) <= cfg.gaussian_round_tol * max(state.m2x, state.m2y)
    return round_q and round_m2


def _facet_list(surf: SurfaceG, shown: int = 8) -> str:
    """The triangles of a facet patch, at most ``shown`` of them listed."""
    triangles = surf.facet.triangles if surf.facet is not None else ()
    listed = ", ".join(str(k) for k in triangles[:shown])
    return listed + (f" and {len(triangles) - shown} more" if len(triangles) > shown else "")


def _max_radius(pk: _Packet) -> float | None:
    """The Gaussian beam radius drawn along a segment (``None``: no valid Gaussian beam; behind a central obstruction the
    state is only the envelope of the unobstructed beam, not the annulus's radius)."""
    g = pk.gstate
    return max(g.radius("x"), g.radius("y")) if g.valid and not pk.chief_obstructed else None


#: Behind a central obstruction the light is not a Gaussian beam (``ArrivalRecord.gstate`` of an obstructed beam).
OBSTRUCTED_TEXT = ("the chief ray fell on a central obstruction: the light passing it is an annulus, not a Gaussian beam")


def _cdot(field: CArray, axis: Array) -> CArray:
    """``field . axis`` per row of a complex 3D field and a real axis (separately rounded products, index order)."""
    return field[:, 0] * axis[:, 0] + field[:, 1] * axis[:, 1] + field[:, 2] * axis[:, 2]


class TraceObserver(Protocol):
    """Read-only observer of a trace, called at fixed points of each packet's life.

    A packet's *steps* are its start (source or branch point) and every surface its chief ray is advanced to. The
    observer may read ``pk.path_id``, ``pk.src``, ``pk.o`` (per-ray positions), ``pk.d`` (per-ray directions) and
    ``pk.alive`` (per-ray mask); it must copy what it keeps and must never write to the packet or the tracer.
    """

    def sampled_finished(self, sampled: Any) -> None:
        """The rays of the extended emitters are traced (``sampled.SampledTrace``). Read only."""

    def packet_started(self, pk: _Packet, surface_id: str | None) -> None:
        """A root packet was created at its source (``surface_id`` None) or a branch was spawned at ``surface_id``."""

    def pupil_rays_diverged(self, pk: _Packet, rays: NDArray[np.intp], next_surface_ids: tuple[str | None, ...],
                            delivered: NDArray[np.bool_]) -> None:
        """Live pupil rays ``rays`` (indices into the packet) are about to be retired because the surface that would stop or
        turn each first, ``next_surface_ids`` (``None``: none), is not the physical surface the chief ray reaches next (called
        just before ``packet_advanced``; observation planes never retire a ray). ``delivered[k]``: ray ``rays[k]`` would otherwise reach the chief's surface
        inside its clear aperture, i.e. the packet's power and Gaussian beam still count it as travelling with the chief."""

    def packet_advanced(self, pk: _Packet, surface_id: str | None) -> None:
        """Every ray in ``pk.alive`` is at ``surface_id`` (rays that would hit another surface already removed from it; at an
        observation plane ``pk.alive`` holds the live rays the plane samples, each where it crosses the plane, for this call
        only: a live ray the plane does not sample goes on and is in ``pk.alive`` again at ``packet_interacted``);
        ``None``: the chief hits nothing and the packet is about to leave the scene."""

    def packet_interacted(self, pk: _Packet) -> None:
        """The interaction with the surface just reached was applied (the path may still end there)."""

    def branch_spawned(self, parent: _Packet, child: _Packet) -> None:
        """A splitting interaction set the state of the branch ``child`` it split from ``parent`` (its direction, Jones vector,
        basis and lineage), before ``parent`` itself is changed by that interaction and before the branch is queued."""

    def path_finished(self, pk: _Packet, path: OpticalPath) -> None:
        """The packet ended and the tracer kept it as ``path``."""

    def path_dropped(self, pk: _Packet) -> None:
        """The packet ended but the path limit discarded it."""


class Tracer:
    #: Whether the scattered light of a scattering surface is recorded to be traced as rays (``ScatterOrigin``): for the
    #: beams of a run; a probe's and a sampled ray's scatter is booked and not followed
    _follows_scatter = True
    #: ``_leg_end`` skips an observation plane no live pupil ray reaches first and whose Gaussian share is certainly at or
    #: below the line's negligible level (``decisions.beyond_reach``); False checks every plane in full (the same results, slower)
    _prefilter_leg_ends = True
    #: Whether light entering a crystal that generates a second harmonic generates it (``_second_harmonic``): for the beams of a
    #: run; a probe's light and an extended emitter's sampled rays pass the crystal unconverted
    _converts_harmonics = True

    def __init__(self, scene: Scene, library: MaterialLibrary, cfg: NumericalConfig, *, include_pupil: bool = True,
                 recorder: TraceObserver | None = None, watch: frozenset[str] = frozenset(),
                 quad_rule: tuple[Array, Array] | None = None, follow_W: dict[tuple[str, int], float] | None = None,
                 keep_faint: bool = False, bounding: bool = False, hold_coating_data: bool = False) -> None:
        self.scene, self.library, self.cfg = scene, library, cfg
        # ``hold_coating_data``: a beam whose chief ray meets a coating outside its data goes on with the data's values at
        # the nearest stated point, its power behind marked (``_live_coating``, ``ArrivalRecord.held``). Only for a caller
        # that carries that mark to every quantity it reports (the beams of a run); elsewhere the path ends there.
        self.hold_coating_data = hold_coating_data
        # ``bounding``: a probe trace, whose light only ever raises the bounds of a result -- where a traced beam's power that
        # passes is not computed (a beam without a valid Gaussian state whose pupil rays an aperture cuts) it goes on with its
        # whole power, and a beam whose chief ray falls on a central obstruction goes on without pupil rays (the bounds of
        # its annulus alone decide, ``_pass_obstruction``)
        self.bounding = bounding
        # ``follow_W``: per source line, the power at or below which a branch is left out (default: the line's negligible
        # level, ``negligible_power_fraction`` x its input); ``keep_faint``: keep the left-out branches' seeds
        # (``TraceOutput.faint_branches``)
        self.follow_W = dict(follow_W or {})
        self.keep_faint = keep_faint
        # surfaces (ids) a beam whose chief ray misses them but whose footprint reaches them is recorded for (``_beside``):
        # the rated surfaces of a run with power ratings
        self._watch = [i for i, s in enumerate(scene.surfaces) if s.surface_id in watch and s.interaction.kind != "observe"]
        self.include_pupil = include_pupil
        self.recorder = recorder
        self.out = TraceOutput()
        # the packets waiting to be traced and the trace budget: every scheduling decision and its accounting
        # (``scheduler.py``); the tracer books the power the budget leaves untraced
        # the resolution at which beams are ordered by power, one grid for the whole run (``scheduler.order_quantum``): a
        # binary fraction, at most ``power_ledger_fraction_resolution``, of the power its sources and emitters launch
        self.order_quantum_W = order_quantum(cfg.power_ledger_fraction_resolution, correctly_rounded_sum(
            [light.power_W for light in (*scene.sources, *scene.emitters)]))
        self.schedule: TraceScheduler[_Packet] = TraceScheduler(cfg.max_paths, cfg.max_trace_steps, cfg.max_interactions_per_path,
                                                                self.order_quantum_W)
        # the Gauss-Legendre rule of ``cfg.aperture_quad_sector_nodes`` nodes: ``quad_rule`` when the caller built it once for
        # a validation call that traces many times (control), else built here
        self.quad_rule = quad_rule if quad_rule is not None else gauss_legendre_rule(cfg.aperture_quad_sector_nodes)
        # the Gauss-Legendre panel rule of second-harmonic generation's overlap integral (``optics/nonlinear/shg.py``)
        self.quad_rule_nl = gauss_legendre_rule(shg.PANEL_NODES) if scene.nonlinear else None
        self._plane_ids = [i for i, s in enumerate(scene.surfaces) if s.geometry == "plane"]
        self._sphere_ids = [i for i, s in enumerate(scene.surfaces) if s.geometry == "sphere"]
        self._other_ids = [i for i, s in enumerate(scene.surfaces) if s.geometry not in ("plane", "sphere")]
        # observation planes, physical surfaces and the surfaces that stop or turn a ray (every physical surface and the
        # absorbing planes; a transparent observation plane is virtual: it never changes, stops or retires a ray), with the
        # observation planes' frames and active areas: the scene as the decisions see it (``decisions.py``)
        self.index = SceneIndex.of(scene.surfaces, tuple(light.position for light in (*scene.sources, *scene.emitters)),
                                   self.cfg.laboratory_reach_mm)
        # the absorbing planes with an active area: they stop only the rays that cross them on it (``_blocking_view``)
        self._absorbing_areas = [i for i in self.index.observe_ids if scene.surfaces[i].interaction.absorbing
                                 and scene.surfaces[i].interaction.active_aperture is not None
                                 and apertures.bounded(scene.surfaces[i].interaction.active_aperture)]
        # the observation planes with an active area (``_off_area_far``)
        self._area_planes = [i for i in self.index.observe_ids if int(self.index.area_shape[i]) != 0]
        self._step_blocking: tuple[Array, Array] | None = None  # (this step's times, their blocking view)
        self._surface_by_id = {s.surface_id: s for s in scene.surfaces}
        self._surface_index = {s.surface_id: i for i, s in enumerate(scene.surfaces)}
        self._balls: dict[int, list[tuple[Array, float]]] = {}  # (``_tail_surfaces``: each surface's bounding balls)
        self._owners = np.array([s.owner_id for s in scene.surfaces], dtype=object)
        # the media inside each element's own body: those of its surfaces that are not the surround (a source's medium or an
        # observation plane's); light in one of them is inside the element (``_spill_beside``)
        surround = {src.medium for src in scene.sources} | {emitter.medium for emitter in scene.emitters} | {
            s.interaction.observation_medium or "air" for s in scene.surfaces if s.interaction.kind == "observe"}
        self._interior: dict[str, set[str]] = {}
        for s in scene.surfaces:
            if s.interaction.kind != "observe":
                self._interior.setdefault(s.owner_id, set()).update({s.medium_pos, s.medium_neg} - surround)
        self._planes = PlaneBatch([scene.surfaces[i] for i in self._plane_ids]) if self._plane_ids else None
        self._spheres = SphereBatch([scene.surfaces[i] for i in self._sphere_ids]) if self._sphere_ids else None
        # unit launch offsets per pupil ring layout, computed once per run (every observation records them)
        self._unit_offsets: dict[tuple[int, int], Array] = {}
        # per-interaction ledger scratch: power handed to child paths and explicitly attributed losses
        self._branch_power = 0.0
        self._split: dict[str, float] = {}
        self._split_detail = ""
        # (path id, Gaussian reason) already reported, so each path reports a lost Gaussian state once
        self._gaussian_reported: set[tuple[str, str]] = set()
        # surfaces with an axicon term (no first-order transfer behind them)
        self._axicon_ids = frozenset(s.surface_id for s in scene.surfaces if s.linear != 0.0)
        # planar patches of facet surfaces that approximate curved ones (no first-order transfer behind them either)
        self._tessellated_ids = frozenset(s.surface_id for s in scene.surfaces if s.facet is not None and s.facet.tessellated)
        # (path id, surface id, code) of tabulated thin-lens domain issues already reported
        self._table_reported: set[tuple[str, str, str]] = set()
        # (surface id, wavelength, what) of a grating's response issues already reported
        self._grating_reported: set[tuple[str, float, str]] = set()
        self._unknown_left = 0  # paths of unknown power the trace budget left untraced (``_report_budget``)
        # uniaxial crystals at a wavelength (``_crystal_at``) and the crystal faces whose oblique incidence was reported
        self._crystals: dict[tuple[str, float], Crystal] = {}
        self._crystal_reported: set[str] = set()
        # the lines crystals generate (a second harmonic): (source id, spectral index) -> [owner, wavelength, power, parent line]
        self._generated: dict[tuple[str, int], list[Any]] = {}
        # radial gradient-index media (``optics/rays/gradient.py``): light in one goes from face to face in one closed-form leg
        self._gradients = scene.gradients
        # surfaces with holes and the same surfaces without them: an annulus's reference axis that crosses a hole while the
        # annulus meets the surface around it takes the surface's continued profile there (``_filled_chief``)
        self._filled = {i: replace(s, aperture=replace(s.aperture, holes=())) for i, s in enumerate(scene.surfaces)
                        if s.interaction.kind != "observe" and s.aperture.holes}

    # ------------------------------------------------------------------------------- public
    def run(self) -> TraceOutput:
        lines = [(src, j, wl, weight) for src in self.scene.sources for j, (wl, weight) in enumerate(src.spectrum)]
        for src, j, wl, weight in lines:
            self.out.inputs.append(SourceInput(src.source_id, src.owner_id, j, wl, src.power_W * weight))
        # what each line launches: one root, or one per polarization component of a partially polarized source (each with its
        # share of the line's power: ``optics/polarization/partial.py``); a fully polarized line's share is 1.0 (exact)
        launches = [(src_k, j, wl, weight * share) for src, j, wl, weight in lines for src_k, share in source_components(src)]
        # the path budget launches the strongest source lines first (ties in source order); the rest are not traced
        roots: dict[int, _Packet | None] = {}
        for i in sorted(range(len(launches)), key=lambda k: (-launches[k][0].power_W * launches[k][3], k)):
            if not self.schedule.paths_left():
                break
            src, j, wl, weight = launches[i]
            roots[i] = self._root_packet(src, j, wl, weight)
            if roots[i] is not None:
                self.schedule.admit_root()
        for i, (src, j, wl, weight) in enumerate(launches):
            if i not in roots:  # the path budget is spent: this line is not traced
                self.schedule.refuse_line(src.source_id)
                self.out.power_flows.append(PowerFlow(SINK_PATH_LIMIT, src.power_W * weight, src.source_id, j, wl, None, None,
                                                      src.owner_id, LINE_NOT_TRACED))
                continue
            pk = roots[i]
            if pk is None:
                self.out.power_flows.append(PowerFlow(SINK_SOURCE_NOT_TRACED, src.power_W * weight, src.source_id, j, wl, None,
                                                      None, src.owner_id, "source"))
            else:
                if self.recorder is not None:
                    self.recorder.packet_started(pk, None)
                self._enqueue(pk)
        self._drain()
        for (source_id, index), (owner, wavelength_nm, power_W, parent) in sorted(self._generated.items()):
            self.out.inputs.append(SourceInput(source_id, owner, index, wavelength_nm, power_W, parent))
        self._report_budget()
        return self.out

    def _drain(self) -> None:
        """Trace the queued packets, strongest first, while the step budget lasts."""
        while self.schedule.pending():
            pk, traced = self.schedule.pop()
            if not traced:  # the step budget is spent: queued branches are not traced
                self._flow(pk, SINK_PATH_LIMIT, pk.chief_power(), None, BRANCH_NOT_TRACED)
                if self.recorder is not None:
                    self.recorder.path_dropped(pk)
                continue
            self._trace(pk)
            checkpoint()  # (one more beam path traced: the run's monitor may stop it here, os_optics/run_control.py)

    def trace_probes(self, roots: list[UntracedBranch]) -> list[ProbeOutcome]:
        """Follow each root ``UntracedBranch`` (its one-ray seed packet) through the scene with the same physics, and,
        recursively, every untraced branch its probe makes in turn (a ghost of the ghost, ...) down to the negligible level
        of its line (``_branch``: at most ``negligible_power_fraction`` x the line's input), strongest first over all roots
        (powers equal on the run's ordering grid in the order met, ``scheduler.order_power``).
        Returns per root the observation crossings, physical-surface hits and power flows of it and all its descendants.
        The budget (``max_paths``, ``max_trace_steps``) is the probes' together: a branch met after it is spent is not
        followed (``ProbeOutcome.not_followed_W``; a root itself not followed: ``traced`` False)."""
        traced = [False] * len(roots)
        arrivals: list[list[ArrivalRecord]] = [[] for _ in roots]
        hits: list[list[SurfaceHit]] = [[] for _ in roots]
        partial: list[list[PartialCrossing]] = [[] for _ in roots]
        flows: list[list[PowerFlow]] = [[] for _ in roots]
        not_followed = [0.0] * len(roots)
        not_followed_count = [0] * len(roots)
        # (strongest first on the run's ordering grid, ``scheduler.order_power``; equal there: in the order met)
        quantum = self.order_quantum_W
        heap: list[tuple[float, int, int, UntracedBranch]] = [(-order_power(b.power_W, quantum), k, k, b) for k, b in enumerate(roots)]
        heapq.heapify(heap)
        order = len(roots)
        while heap:
            _, ordinal, root, branch = heapq.heappop(heap)
            if not self.schedule.admit_probe(branch.path_id, branch.source_id):  # the budget is spent: not followed
                not_followed[root] += branch.power_W
                not_followed_count[root] += 1
                continue
            if ordinal == root:
                traced[root] = True
            seed = branch.seed
            first_arrival, first_flow, first_branch = len(self.out.arrivals), len(self.out.power_flows), len(self.out.untraced_branches)
            first_hit, first_partial = len(self.out.surface_hits), len(self.out.partial_crossings)
            self._enqueue(replace(seed, o=seed.o.copy(), d=seed.d.copy(), e1=seed.e1.copy(), e2=seed.e2.copy(),
                                  jones=seed.jones.copy(), alive=seed.alive.copy(), opl=seed.opl.copy(), length=seed.length.copy(),
                                  lineage=list(seed.lineage), segments=[], crossings=dict(seed.crossings)))
            self._drain()
            arrivals[root].extend(self.out.arrivals[first_arrival:])
            hits[root].extend(self.out.surface_hits[first_hit:])
            partial[root].extend(self.out.partial_crossings[first_partial:])
            flows[root].extend(self.out.power_flows[first_flow:])
            for child in self.out.untraced_branches[first_branch:]:
                heapq.heappush(heap, (-order_power(child.power_W, quantum), order, root, child))
                order += 1
        return [ProbeOutcome(traced[k], tuple(arrivals[k]), tuple(flows[k]), not_followed[k], not_followed_count[k], tuple(hits[k]),
                             tuple(partial[k]))
                for k in range(len(roots))]

    def _report_budget(self) -> None:
        """One ``MAX_PATHS`` issue for the whole run when the trace budget left power untraced (``TraceScheduler.report``):
        the power the ledger's ``PATH_LIMIT`` holds, of the launched power."""
        if not self.schedule.exhausted:
            return
        untraced_W = correctly_rounded_sum(f.power_W for f in self.out.power_flows if f.kind == SINK_PATH_LIMIT)
        launched_W = correctly_rounded_sum(i.power_W for i in self.out.inputs)
        issue = self.schedule.report(len(self.out.paths), untraced_W, launched_W, self._unknown_left)
        if issue is not None:
            self._issue(issue)

    # ------------------------------------------------------------------------------ helpers
    def _enqueue(self, pk: _Packet) -> None:
        """Queue a packet: the strongest queued packet is traced next (powers equal on the run's ordering grid in queue
        order, ``scheduler.order_power``), so a budget that runs out leaves the weakest branches untraced."""
        self.schedule.push(pk, pk.chief_power(), pk.path_id, pk.src.source_id)

    def _issue(self, issue: ValidationIssue) -> None:
        self.out.issues.append(issue)

    def _extinction(self, pk: _Packet, medium: str | None = None, wavelength_nm: float | None = None) -> float:
        """Extinction along a packet's medium; unknown absorption opens its power, not its geometry.

        A default scalar zero is not a stated zero. The unknown incident power is booked once
        in the ledger and the packet carries an upper-bound Jones field thereafter.
        """
        medium = pk.medium if medium is None else medium
        evaluator = self.library.evaluator(medium)
        wavelength = pk.wavelength_nm if wavelength_nm is None else wavelength_nm
        if self.library.loss_unstated(medium) or not evaluator.loss_established(wavelength):
            key = (medium, wavelength)
            reported = self.__dict__.setdefault("_loss_unknown", set())
            if key not in reported:
                reported.add(key)
                name = medium.rsplit(":", 1)[-1]
                self._issue(make_issue(
                    ErrorCode.BULK_LOSS_NOT_ESTABLISHED, Status.WARNING,
                    f"the bulk absorption of {name} is not established at {wavelength} nm (no stated bulk "
                    "loss): the geometric path remains valid but its power may be anywhere from zero to the unattenuated bound",
                    location=f"material:{medium}", wavelength_nm=wavelength,
                    suggested_action="supply bulk-loss data at this wavelength or explicitly assume a scalar bulk absorption coefficient"))
            if medium == pk.medium and wavelength == pk.wavelength_nm and pk.power_bound_W is None:
                # The same watt may be absorbed or survive; preserve the Jones upper
                # envelope for downstream geometry and modelled surface losses.
                self._flow(pk, SINK_BULK_LOSS_UNKNOWN, pk.chief_power(), None, medium)
                pk.power_bound_W = pk.chief_power()
                pk.bulk_loss_unknown = True
        return evaluator.extinction_at(wavelength)

    def _root_packet(self, src: SourceG, index: int, wavelength_nm: float, weight: float) -> _Packet | None:
        cfg = self.cfg
        n_src, issue = self.library.evaluator(src.medium).index(wavelength_nm, location=f"source:{src.source_id}")
        if issue is not None or n_src is None:
            if issue is not None:
                self._issue(issue)
            return None
        gstate = initial_state(
            wavelength_nm=wavelength_nm, n=n_src, w0x=src.w0x, w0y=src.w0y, z_waist=src.z_waist,
            m2x=src.m2x, m2y=src.m2y, bx=src.e1, by=src.e2, cfg=cfg,
        ).located(src.source_id)
        if gstate.reason is not None and gstate.status is not Status.PASS:
            self._issue(make_issue(gstate.reason, gstate.status, gstate.reason_text, location=f"source:{src.source_id}",
                                   wavelength_nm=wavelength_nm))
        origin, direction = src.position, src.direction
        n_pupil = 0
        launch_spread = None
        if self.include_pupil and gstate.valid:
            offs = (pupil_offsets(src.w0x, src.w0y, src.ring_count, src.points_per_ring) if src.pupil_layout is None
                    else src.pupil_layout * np.array([src.w0x, src.w0y]))  # an analysis-only layout (SourceG.pupil_layout)
            # actual beam radius at the source plane (waist may not sit at the origin)
            offs = offs * np.array([gstate.radius("x") / src.w0x, gstate.radius("y") / src.w0y])
            gx, gy = (1.0 / gstate.qx).real, (1.0 / gstate.qy).real  # wavefront slopes per mm of offset
            lam_mm = wavelength_nm * NM_TO_MM
            launch_spread = (src.m2x * lam_mm / (math.pi * n_src * gstate.radius("x")),
                             src.m2y * lam_mm / (math.pi * n_src * gstate.radius("y")))
            o = np.vstack([origin, origin + offs[:, 0:1] * src.e1 + offs[:, 1:2] * src.e2])
            dirs = direction + (offs[:, 0:1] * gx) * src.e1 + (offs[:, 1:2] * gy) * src.e2
            d_pup, _ = normalize(dirs, cfg.zero_vector_tol)
            d = np.vstack([direction, d_pup])
            n_pupil = offs.shape[0]
        else:
            o, d = origin[None, :].copy(), direction[None, :].copy()
        n_diff = 0
        if self.include_pupil:
            dx, da = cfg.first_order_position_step_mm, cfg.first_order_angle_step_rad
            shifts = np.array([dx * src.e1, -dx * src.e1, dx * src.e2, -dx * src.e2])
            tilts = np.array([da * src.e1, -da * src.e1, da * src.e2, -da * src.e2])
            d_tilt, _ = normalize(direction + tilts, cfg.zero_vector_tol)
            o = np.vstack([o, origin + shifts, np.repeat(origin[None, :], 4, 0)])
            d = np.vstack([d, np.repeat(direction[None, :], 4, 0), d_tilt])
            n_diff = DIFFERENTIAL_RAY_COUNT
        n_rays = o.shape[0]
        e1c, e2c = src.e1[None, :], src.e2[None, :]
        if n_rays > 1:
            e1, e2 = parallel_transport_basis(np.repeat(e1c, n_rays, 0), np.repeat(e2c, n_rays, 0),
                                              np.repeat(direction[None, :], n_rays, 0), d, cfg.antiparallel_tol)
        else:
            e1, e2 = e1c.copy(), e2c.copy()
        return _Packet(
            path_id=root_path_id(src.source_id, index, src.polarization_component), src=src, spectral_index=index, wavelength_nm=wavelength_nm,
            power0_W=src.power_W * weight, medium=src.medium, n=n_src, o=o, d=d, e1=e1, e2=e2,
            jones=np.repeat(src.jones[None, :], n_rays, 0).astype(np.complex128), alive=np.ones(n_rays, dtype=bool),
            opl=np.zeros(n_rays), length=np.zeros(n_rays), gstate=gstate, lineage=[], pupil_total=n_pupil, differential=n_diff,
            segment_normal=(float(direction[0]), float(direction[1]), float(direction[2])),  # (the source plane)
            launch_spread=launch_spread,
        )

    def _flow(self, pk: _Packet, kind: str, power_W: float, surface: SurfaceG | None, detail: str) -> None:
        """Record traced power leaving the beams (``PowerFlow``). A branch whose power is not known books nothing: the power
        it may carry was booked once where it became unknown."""
        if pk.power_bound_W is not None:
            self._unknown_left += kind == SINK_PATH_LIMIT
            return
        for kind, power_W in self._bulk_split(pk, kind, power_W):
            self.out.power_flows.append(PowerFlow(
                kind, float(power_W), pk.src.source_id, pk.spectral_index, pk.wavelength_nm, pk.path_id,
                None if surface is None else surface.surface_id, None if surface is None else surface.owner_id, detail))

    def _bulk_split(self, pk: _Packet, kind: str, power_W: float) -> tuple[tuple[str, float], ...]:
        """A bulk extinction booked in the packet's medium as its absorbed and scattered parts (``scatter.bulk_shares``); any
        other entry as it is."""
        if kind != SINK_BULK_ABSORPTION:
            return ((kind, power_W),)
        medium = self.library.evaluator(pk.medium)
        return bulk_shares(medium.absorption_at(pk.wavelength_nm) or 0.0, medium.scattering_per_mm, power_W)

    def _negligible(self, pk: _Packet, power_W: float) -> bool:
        """Power at most the negligible level of the packet's source line (``negligible_power_fraction`` x line
        input): light a result leaves out by statement -- or at most the line's ``follow_W`` level when one is given."""
        return power_W <= self._negligible_level(pk)

    def _negligible_level(self, pk: _Packet) -> float:
        """The power at or below which ``_negligible`` holds for the packet's line."""
        level = self.follow_W.get((pk.src.source_id, pk.spectral_index)) if self.follow_W else None
        return self.cfg.negligible_power_fraction * pk.power0_W if level is None else level

    def _leave_out(self, pk: _Packet, power_W: float) -> None:
        """Note a branch of the packet's line left out as negligible (``TraceOutput.left_out_W``: their total per line;
        ``left_out_root``: the sum of their square roots)."""
        if power_W > 0.0:
            key = (pk.src.source_id, pk.spectral_index)
            self.out.left_out_W[key] = self.out.left_out_W.get(key, 0.0) + float(power_W)
            self.out.left_out_root[key] = self.out.left_out_root.get(key, 0.0) + math.sqrt(float(power_W))

    def _branch(self, pk: _Packet, kind: str, surf: SurfaceG, direction: Array, gstate: GState, power_W: float, *,
                jones: CArray | None = None, e1: Array | None = None, e2: Array | None = None,
                rays: Callable[[], tuple[Array, Array, Array, CArray]] | None = None,
                alive: NDArray[np.bool_] | None = None, delay: str | None = None,
                wave: tuple[float, _CrystalLeg | None] | None = None) -> None:
        """Record light that leaves the traced beams by a model approximation but goes on (``UntracedBranch``) from the chief
        hit ``pk.o[0]`` along ``direction``, unless it is negligible for its line. ``jones`` / ``e1`` / ``e2``: its Jones
        vector and basis (default: the packet's chief ray scaled to ``power_W``). In a probe trace (second order: the light
        of a probe that leaves it in turn) it is recorded too and followed by ``trace_probes``.

        The branch keeps the packet's pupil rays (not its differential rays), so a probe trace checks its Gaussian state
        against them where it meets a surface, as the trace checks its own beams (``validation/exposures.py``): ``rays``
        gives every ray's (direction, e1, e2, Jones vector) in the branch (rows as the packet's; default: the packet's rays
        going on unchanged, their Jones vectors scaled as the chief's), ``alive`` which rays carry it (default: the packet's
        live rays); ``delay``: the record of its own pass when the model does not state its group delay (``_delay``);
        ``wave``: its index along the ray and crystal wave where it leaves as another wave of a crystal (default: the packet's)."""
        if pk.power_bound_W is not None:  # (its power was booked once: nothing separate leaves it)
            return
        faint = self._negligible(pk, power_W)
        if faint:
            self._leave_out(pk, power_W)
            if not self.keep_faint:
                return
        p_chief = pk.chief_power()
        scale = math.sqrt(power_W / p_chief) if p_chief > 0.0 else 0.0
        if jones is None:
            jones = pk.jones[0] * scale if p_chief > 0.0 else pk.jones[0] * 0.0
        e1 = pk.e1[0] if e1 is None else e1
        e2 = pk.e2[0] if e2 is None else e2
        # (an annulus behind a central obstruction leaves its light without pupil rays: its bounds alone decide); a branch
        # keeps the parent's pupil rays of two whole rings (the outermost and the middle one: a symmetric sample, so the
        # aberration check's bound holds for it as for all of them)
        rows = None if pk.chief_obstructed or not pk.pupil_total else self._probe_rows(pk)
        pupil_total = 0 if rows is None else int(rows.size)
        count = 1 + pupil_total
        d_rows = np.array(direction, dtype=float)[None, :]
        e1_rows, e2_rows = np.array(e1, dtype=float)[None, :], np.array(e2, dtype=float)[None, :]
        j_rows = np.array(jones, dtype=np.complex128)[None, :]
        live = np.ones(1, dtype=bool)
        o_rows = pk.o[np.concatenate([[0], rows])] if rows is not None else pk.o[:1].copy()
        if rows is not None:
            # (the chief ray's row as computed for it alone: the probe's chief ray is the same with or without pupil rays)
            with np.errstate(invalid="ignore", divide="ignore", over="ignore"):  # (retired rays' rows may hold no hit)
                d_all, e1_all, e2_all, j_all = rays() if rays is not None else (pk.d, pk.e1, pk.e2, pk.jones * scale)
            d_rows = np.vstack([d_rows, d_all[rows]])
            e1_rows, e2_rows = np.vstack([e1_rows, e1_all[rows]]), np.vstack([e2_rows, e2_all[rows]])
            j_rows = np.vstack([j_rows, np.asarray(j_all, dtype=np.complex128)[rows]])
            live = np.concatenate([live, (pk.alive if alive is None else alive)[rows]])
            finite = (np.isfinite(d_rows).all(axis=1) & np.isfinite(e1_rows).all(axis=1) & np.isfinite(e2_rows).all(axis=1)
                      & np.isfinite(j_rows.real).all(axis=1) & np.isfinite(j_rows.imag).all(axis=1)
                      & np.isfinite(pk.o[:count]).all(axis=1))
            if not bool(finite.all()):  # a retired ray without a hit: kept retired, on the chief ray's values
                live = live & finite
                d_rows = np.where(finite[:, None], d_rows, d_rows[0])
                e1_rows, e2_rows = np.where(finite[:, None], e1_rows, e1_rows[0]), np.where(finite[:, None], e2_rows, e2_rows[0])
                j_rows = np.where(finite[:, None], j_rows, 0.0)
                o_rows = np.where(finite[:, None], o_rows, o_rows[0])
        launch_rows = None
        if rows is not None:
            launch_rows = rows - 1 if pk.launch_rows is None else pk.launch_rows[rows - 1]
        # Opening offsets use packet row indices, just like origins and Jones vectors.
        # An annular branch intentionally retains only its reference ray; preserve the
        # opening geometry but no discarded pupil evidence for the map's sign.
        opening = pk.opening
        if opening is not None:
            kept_rows = np.concatenate([[0], rows]) if rows is not None else np.array([0])
            opening = replace(opening, ray_offsets=opening.ray_offsets[kept_rows].copy())
        kept = np.concatenate([[0], rows]).astype(np.intp) if rows is not None else np.zeros(1, dtype=np.intp)
        n_seed, leg = (pk.n, pk.crystal) if wave is None else wave
        seed = _Packet(
            path_id=pk.path_id, src=pk.src, spectral_index=pk.spectral_index, wavelength_nm=pk.wavelength_nm,
            power0_W=pk.power0_W, medium=pk.medium, n=n_seed, o=o_rows, d=d_rows,
            e1=e1_rows, e2=e2_rows, jones=j_rows, alive=live,
            opl=pk.opl[np.concatenate([[0], rows])] if rows is not None else pk.opl[:1].copy(),
            length=pk.length[np.concatenate([[0], rows])] if rows is not None else pk.length[:1].copy(), gstate=gstate, lineage=[*pk.lineage, f"{surf.surface_id}:{kind.lower()}"],
            parent_path_id=pk.path_id, branch_surface_id=surf.surface_id, branch_kind=kind,
            interactions=pk.interactions + 1, pupil_total=pupil_total, opl_undefined=pk.opl_undefined, crossings=dict(pk.crossings),
            pending_observers=pk.pending_observers, leg_crossed=pk.leg_crossed, chief_obstructed=pk.chief_obstructed,
            annulus_fraction=pk.annulus_fraction, opening=opening,
            passes_owner=surf.owner_id if kind == SINK_APERTURE_TRUNCATION else None,
            gpl=pk.gpl, gdd=pk.gdd, dispersive=pk.dispersive, spread=pk.spread, segment_normal=pk.segment_normal,
            segment_surface=pk.segment_surface, segment_start=pk.segment_start, segment_mm=pk.segment_mm, launch_rows=launch_rows, delays=pk.delays if delay is None else (*pk.delays, delay),
            crystal=None if leg is None else leg.rows(kept), gaussian_error_mm=pk.gaussian_error_mm,
            reference_promoted=pk.reference_promoted)
        (self.out.faint_branches if faint else self.out.untraced_branches).append(UntracedBranch(
            kind, pk.src.source_id, pk.spectral_index, pk.path_id, surf.surface_id, pk.o[0].copy(),
            np.array(direction, dtype=float), gstate if gstate.valid else None, float(power_W), seed))

    def _probe_rows(self, pk: _Packet) -> NDArray[np.intp] | None:
        """The packet's pupil rays (rows) a branch keeps: those of two whole rings of the source's sampling -- the
        outermost and the middle one -- that the packet still carries (all of its pupil rays when it carries no more)."""
        if pk.launch_rows is not None:
            return np.arange(1, 1 + pk.pupil_total)
        rings, per_ring = pk.src.ring_count, pk.src.points_per_ring
        keep = sorted({rings - 1, rings // 2})
        rows = np.concatenate([1 + ring * per_ring + np.arange(per_ring) for ring in keep]).astype(np.intp)
        return rows[rows <= pk.pupil_total] if rows.size else None

    def _spill_beside(self, pk: _Packet, surf: SurfaceG, h: Hits, chief_cos: float, gstate: GState, p_in: float,
                      opening_fraction: float) -> None:
        """The part of a beam whose chief ray passes a clear aperture that falls beyond the element's support (clear
        aperture grown by the mount ring, or the outline): it passes beside the element, booked inside
        ``APERTURE_TRUNCATION`` but going on physically (``UntracedBranch``). Only for light travelling in the element's
        surround: inside the element's own body (a lens's glass) the part beyond the support meets the element's edge or
        mount, an ordinary known loss like the main beam's (booked ``APERTURE_TRUNCATION``, no branch)."""
        if not gstate.valid:
            return
        if pk.medium in self._interior.get(surf.owner_id, ()):
            return
        support = support_of(surf)
        if not apertures.bounded(support):
            return
        beyond_W = self._beyond_share(opening_fraction) * p_in  # (at most this much passes beside the support)
        if self._negligible(pk, beyond_W):
            if self.keep_faint:
                self._branch(pk, SINK_APERTURE_TRUNCATION, surf, pk.d[0], gstate, beyond_W)
            else:
                self._leave_out(pk, beyond_W)
            return
        ap_u, ap_v, ap_cos = self._aperture_frame(surf, h, chief_cos)
        on_support = transmission_fraction(gstate, replace(support, obstruction=0.0), (float(h.u[0]), float(h.v[0])), ap_u, ap_v,
                                           ap_cos, self.cfg, self.quad_rule)
        self._branch(pk, SINK_APERTURE_TRUNCATION, surf, pk.d[0], gstate, self._beyond_share(on_support) * p_in)

    def _beyond_share(self, inside: float) -> float:
        """The share of a beam beyond an aperture or support of which the fraction ``inside`` (``transmission_fraction``)
        falls inside it, resolved upward to ``power_ledger_fraction_resolution`` (the resolution of the beam's
        ``APERTURE_TRUNCATION`` ledger entry): ``1 - inside`` of a fraction next to 1 keeps the quadrature's rounding of the
        whole beam (a few 1e-15), which is all the digits a share of 1e-10 has, and which follows the last bits of the poses.
        Upward: the share bounds the light left out or branched beside the support from above (a faint-light bound)."""
        quantum = self.cfg.power_ledger_fraction_resolution
        return math.ceil(max(1.0 - inside, 0.0) / quantum) * quantum

    def _facet_diverged(self, pk: _Packet, diverged: NDArray[np.bool_], nearest: NDArray[np.intp], surf: SurfaceG) -> int:
        """How many of the pupil samples ``diverged`` retires at this step meet another planar patch of the facet surface of
        ``surf`` first (``nearest``: each ray's first surface): their light falls beyond the edge of ``surf``'s patch."""
        if surf.facet is None or not bool(diverged.any()):
            return 0
        mesh = surf.facet.mesh_id
        surfaces = self.scene.surfaces
        rays = np.flatnonzero(diverged & self._annulus_rays(pk))
        return sum(1 for r in rays if nearest[r] >= 0 and (other := surfaces[int(nearest[r])].facet) is not None and other.mesh_id == mesh)

    def _facet_edge(self, pk: _Packet, surf: SurfaceG, p_in: float, share: float) -> None:
        """A beam with a valid Gaussian state on a planar patch of a facet surface whose outline cuts part of it: that part
        (``share`` of the power, the Gaussian integral beyond the outline) falls on the neighbouring facets (or, beyond an
        outer edge of the part, past it), which this beam does not follow. Its power goes to ``TERMINATED_OUT_OF_MODEL``
        (its further fate is not known; it may reach any plane), the beam goes on with the rest, and
        ``FACET_EDGE_DIVIDES_BEAM`` states the share. Beyond
        ``gaussian_clip_scope_fraction`` the part on the patch is not a Gaussian beam: its Gaussian state ends (reason
        ``FACET_EDGE_DIVIDES_BEAM``), and the power from there on is the share of its pupil samples (as behind any severe
        truncation)."""
        cfg = self.cfg
        quantum = cfg.power_ledger_fraction_resolution
        share = round(max(share, 0.0) / quantum) * quantum
        if share <= 0.0:
            return
        pk.jones = pk.jones.copy()
        pk.jones[0] = pk.jones[0] * math.sqrt(max(1.0 - share, 0.0))
        lost = p_in * share
        self._flow(pk, SINK_TERMINATED_OUT_OF_MODEL, lost, surf, "facet edge")
        severe = share > cfg.gaussian_clip_scope_fraction
        if severe:
            pk.gstate = pk.gstate.invalidated(
                ErrorCode.FACET_EDGE_DIVIDES_BEAM,
                f"the edge of the facet patch {surf.surface_id} divides the beam ({share:.3g} of its power falls beyond the edge): the "
                "part on the patch is not a Gaussian beam")
        if self._negligible(pk, lost):
            return
        self._issue(make_issue(
            ErrorCode.FACET_EDGE_DIVIDES_BEAM, Status.OUT_OF_MODEL_SCOPE if severe else Status.WARNING,
            f"the edge of the facet patch {surf.surface_id} (facets {_facet_list(surf)} of {surf.facet.mesh_id}) divides the beam: "
            f"{share:.4g} of its power ({lost:.4g} W) falls beyond the edge (on the neighbouring facets, or past the part) and is "
            "not followed by this beam, so its further fate is not known; the beam goes on with the rest"
            + ("; its Gaussian beam metrics are not available behind the edge" if severe else ""),
            location=f"surface:{surf.surface_id}", path_id=pk.path_id, loss_fraction=share, untraced_power_W=lost,
            facet_surface=surf.facet.mesh_id, suggested_action="aim the beam inside one facet, or reduce its radius at the facets"))

    def _facet_edge_geometric(self, pk: _Packet, surf: SurfaceG, h: Hits, p_in: float) -> str | None:
        """A beam without a valid Gaussian state on a planar patch of a facet surface, some of whose pupil samples this step
        retired toward the surface's other patches (``_Packet.facet_diverged``): on the geometric pupil-sample basis their
        share falls beyond the patch's edge and is not followed (``TERMINATED_OUT_OF_MODEL``, ``FACET_EDGE_DIVIDES_BEAM``);
        the beam goes on with the share of the samples on the patch. No sample on the patch: the power that goes on is not
        computed (``_not_computed``)."""
        here = int((pk.alive & self._annulus_rays(pk)).sum())
        beyond = pk.facet_diverged
        pk.facet_diverged = 0
        if self.bounding:  # (a probe goes on with its whole power: an upper bound)
            return None
        if here == 0:
            return self._not_computed(pk, surf, f"the edge of the facet patch {surf.surface_id}", pupil_rays_cut=beyond)
        share = beyond / (here + beyond)
        pk.jones = pk.jones.copy()
        pk.jones[0] = pk.jones[0] * math.sqrt(1.0 - share)
        lost = p_in * share
        self._flow(pk, SINK_TERMINATED_OUT_OF_MODEL, lost, surf, "facet edge")
        known = pk.power_bound_W is None
        if known and self._negligible(pk, lost):
            return None
        reason = pk.gstate.reason or ErrorCode.OUT_OF_MODEL_SCOPE
        self._issue(make_issue(
            ErrorCode.FACET_EDGE_DIVIDES_BEAM, Status.OUT_OF_MODEL_SCOPE,
            f"the beam has no valid Gaussian state ({pk.gstate.reason_text or reason.value}) and {beyond} of its {here + beyond} pupil "
            f"samples meet other facets of {surf.facet.mesh_id} than the patch {surf.surface_id} its chief ray meets: their share "
            f"({share:.4g}" + (f", {lost:.4g} W" if known else "") + ") falls beyond the patch's edge and is not followed by this "
            "beam, so its further fate is not known; the beam goes on with the share of the samples on the patch",
            location=f"surface:{surf.surface_id}", path_id=pk.path_id, loss_fraction=share, pupil_rays_cut=beyond,
            pupil_rays_live=here + beyond, facet_surface=surf.facet.mesh_id,
            **({"untraced_power_W": lost} if known else {}),
            suggested_action="aim the beam inside one facet, or reduce its radius at the facets"))
        return None

    def _finish(self, pk: _Packet, termination: str, surface_id: str | None) -> None:
        surface = None if surface_id is None else self._surface_by_id.get(surface_id)
        if termination != GRATING_ORDERS:  # (a grating's handler booked all of it)
            self._flow(pk, TERMINATION_SINKS.get(termination, SINK_TERMINATED_OUT_OF_MODEL), pk.chief_power(), surface, termination)
        if not self.schedule.keep_path(len(self.out.paths)):
            if self.recorder is not None:
                self.recorder.path_dropped(pk)
            return
        self.out.paths.append(
            OpticalPath(
                path_id=pk.path_id, source_id=pk.src.source_id, spectral_index=pk.spectral_index,
                wavelength_nm=pk.wavelength_nm, parent_path_id=pk.parent_path_id, branch_surface_id=pk.branch_surface_id,
                branch_kind=pk.branch_kind, lineage=tuple(pk.lineage), segments=tuple(pk.segments), termination=termination,
                termination_surface_id=surface_id, final_power_W=None if pk.power_bound_W is not None else pk.chief_power(),
                power_out_of_scope_at=pk.scope_behind(), geometric_path_length_mm=float(pk.length[0]),
                optical_path_length_mm=None if pk.opl_undefined else float(pk.opl[0]),
            )
        )
        self.out.trace_events.append({"path": pk.path_id, "end": termination, "surface": surface_id,
                                      "opl": float(pk.opl[0]), "len": float(pk.length[0]), "power": pk.chief_power()})
        if self.recorder is not None:
            self.recorder.path_finished(pk, self.out.paths[-1])

    def _spawn(self, pk: _Packet, surf: SurfaceG, kind: str, power_W: float) -> _Packet | None:
        """A branch of ``pk`` the trace follows, born at ``surf`` by the event ``kind`` with ``power_W``: its packet (the
        caller sets its state and queues it), or ``None`` when the path budget refuses it -- its power is then booked to the
        ledger's ``PATH_LIMIT`` here."""
        if not self.schedule.reserve_branch(pk.path_id, pk.src.source_id):
            self._flow(pk, SINK_PATH_LIMIT, power_W, surf, kind)
            return None
        root = root_path_id(pk.src.source_id, pk.spectral_index, pk.src.polarization_component)
        child = pk.fork(child_path_id(pk.path_id, root, pk.children), surf.surface_id, kind)
        # the branch is born at a physical interaction (a split, a traced reflection): it counts toward
        # max_interactions_per_path for the branch as it does for the parent (counted after the handler), so a cavity
        # followed through its reflected branches stops at the limit like one followed by a single path
        child.interactions = pk.interactions + 1
        pk.children += 1
        if self.recorder is not None:
            self.recorder.packet_started(child, surf.surface_id)
        return child

    def _first_blocking(self, times: Array, exclude: list[int] | tuple[int, ...] = (), *,
                        whole_planes: bool = False) -> tuple[Array, NDArray[np.intp]]:
        """Per ray, the parameter and scene index (-1: none) of the first surface that stops or turns it (a physical surface or
        an absorbing plane on its active area, ``_blocking_view``; ``whole_planes``: anywhere on it; a transparent
        observation plane is virtual), leaving out ``exclude`` (``decisions.first_blocking``)."""
        return first_blocking(self.index.blocking_ids, times if whole_planes else self._blocking_view(times), exclude)

    def _blocking_view(self, times: Array) -> Array:
        """``times`` (this step's ray parameters per surface) as the surfaces that stop or turn a ray see them: an absorbing
        plane with an active area stops a ray only where the ray crosses it on that area (elsewhere ``inf``), as it stops
        the beam only when its active area takes it (``_stops_beam``) and bounds the part of a beam it takes on that area
        (``decisions.part_on_sheet``)."""
        held = self._step_blocking
        return held[1] if held is not None and held[0] is times else times

    def _set_blocking(self, pk: _Packet, times: Array) -> None:
        """Form this step's ``_blocking_view`` of ``times`` (after its last change)."""
        if not self._absorbing_areas:
            self._step_blocking = None
            return
        view = times
        for si in self._absorbing_areas:
            t = times[si]
            rays = np.flatnonzero(np.isfinite(t))
            if rays.size == 0:
                continue
            surf = self.scene.surfaces[si]
            rel = pk.o[rays] + t[rays][:, None] * pk.d[rays] - surf.origin
            inside = aperture_inside(surf.interaction.active_aperture, dot(rel, surf.u_axis), dot(rel, surf.v_axis),
                                     self.cfg.aperture_edge_tol_mm)
            if bool(inside.all()):
                continue
            if view is times:
                view = times.copy()
            view[si, rays[~inside]] = np.inf
        self._step_blocking = (times, view)

    def _stays(self, pk: _Packet, times: Array, crossing: list[int], lead: int) -> tuple[NDArray[np.bool_], NDArray[np.intp], Array]:
        """Which rays reach the physical surface ``lead`` of the chief ray's crossing with it: those whose first surface that
        stops or turns them (``_first_blocking``; the crossing's coincident observation planes and the planes the chief ray
        crossed on this leg aside) is ``lead``. Also returns, per ray, that first surface and its parameter. Only a physical
        surface or an absorbing plane takes a pupil ray out of the beam, and only here, where the chief ray meets one:
        transparent observation planes never do."""
        t_block, nearest = self._first_blocking(times, (*crossing[:-1], *pk.leg_crossed))
        return nearest == lead, nearest, t_block

    def _envelope_sigmas(self, pk: _Packet) -> float | None:
        """``K`` of the packet's beam (``decisions.envelope_sigmas``): beyond ``K`` beam radii from the chief ray a Gaussian
        beam's share past a line is at most an eighth of the line's negligible level."""
        return envelope_sigmas(self._negligible_level(pk), pk.chief_power(), pk.chief_obstructed, pk.annulus_fraction)

    def _view(self, pk: _Packet) -> BeamView:
        """The packet as the decisions read it, now (``decisions.BeamView``)."""
        return BeamView(pk.path_id, pk.src.source_id, pk.spectral_index, pk.wavelength_nm, pk.o, pk.d, pk.alive,
                        self._annulus_rays(pk), pk.gstate, pk.chief_power(), pk.chief_obstructed, pk.annulus_fraction,
                        self._negligible_level(pk))

    def _reach_along(self, si: int | None, point: Array, d: Array, reach: float | None, side: float,
                     every_hit: bool = False) -> float | None:
        """How far a curved surface ``si`` lies beyond (``side`` +1: along ``d``) or back from (``side`` -1) the plane across
        ``d`` through its point ``point``, for the rays parallel to ``d`` within ``reach`` of ``point`` that meet it (0 when
        it does not lie that way): ``None`` for a plane surface; ``inf`` where nothing bounds it (no reach, an unbounded
        support). Each bound below is sound by itself; the smallest is taken.

        * A sphere of centre ``C`` within the envelope, where the envelope stays inside the cap (every such ray meets the
          cap at the root nearest the point): a ray at offset ``rho`` meets it ``z = -a + sgn(a) sqrt(a^2 - 2 rho . u -
          |rho|^2)`` along ``d`` from the plane (``u = point - C``, ``a = d . u``). With ``p`` the distance of ``u`` from
          the ray, it lies at most ``|a| - sqrt(a^2 - 2 reach p - reach^2)`` toward ``-sgn(a) d`` (the way it bends) and at
          most ``sqrt(a^2 + m) - |a|`` the other way (``m``: ``p^2``, or ``2 reach p - reach^2`` for ``p > reach``).
        * A sphere's whole cap (``_cap_extent``): the range of ``d . (P - point)`` over its points (also where a ray of the
          envelope passes the cap's rim and meets its far part).
        ``every_hit`` (the surface a ray meets first, ``_taken_before``): a sphere's nearest-root bound only where no ray of
        the envelope meets the cap first at the other root of its line (``_other_root_clear``), else its whole cap.
        * A sag surface (``_sag_extent``): the slab of its sag range over the part of its support such a ray can meet, and
          the box of its whole support's extent along the ray."""
        if si is None:
            return None
        surf = self.scene.surfaces[si]
        if surf.geometry == "plane":
            return None
        if reach is None or not math.isfinite(reach):
            return math.inf
        if surf.geometry == "sphere":
            lo, hi = self._cap_extent(surf, point, d)
            rel = point - surf.center
            a = float(dot(d, rel))
            if a != 0.0:
                p = math.sqrt(max(float(dot(rel, rel)) - a * a, 0.0))
                bends = (side > 0.0) == (a < 0.0)  # (the side the surface bends to: a sphere whose centre lies that way)
                disc = a * a - 2.0 * reach * p - reach * reach
                if bends:
                    bound = abs(a) - math.sqrt(disc) if disc > 0.0 else math.inf
                else:
                    bound = max(math.sqrt(a * a + (p * p if p <= reach else 2.0 * reach * p - reach * reach)) - abs(a), 0.0)
                m = p * p if p <= reach else 2.0 * reach * p - reach * reach
                far = max(abs(a) - math.sqrt(disc) if disc > 0.0 else math.inf,
                          math.sqrt(a * a + m) - abs(a))  # (the farthest root either way)
                spread = math.sqrt(reach * reach + far * far)
                if math.isfinite(bound) and self._inside_cap(surf, point, spread) and (
                        not every_hit or self._other_root_clear(surf, point, d, a, spread)):
                    lo, hi = (lo, min(hi, bound)) if side > 0.0 else (max(lo, -bound), hi)
        elif surf.geometry == "sag":
            lo, hi = self._sag_extent(surf, point, d, reach)
        else:
            return math.inf
        return max(hi, 0.0) if side > 0.0 else max(-lo, 0.0)

    def _other_root_clear(self, surf: SurfaceG, point: Array, d: Array, a: float, radius: float) -> bool:
        """Whether every ray of the envelope meets a sphere first at the root of its line nearest the chief ray's hit
        ``point``, not at the other one: always where the chief line enters the sphere at ``point`` (``a = d . (point -
        C) < 0``: the other roots lie ahead); else the chief line crossed the sphere before, at ``P2 = point - 2 a d``, and
        the other root of each envelope line lies within ``radius`` of ``P2`` (across ``d`` within the reach, along it
        within the nearest roots' spread, ``_reach_along``). None of them is on the cap when every point within
        ``radius`` of ``P2`` lies below the cap's rim plane, ``(P2 - C) . cap_axis + radius < rim`` (that coordinate
        changes by at most the distance; the rim from the support with the edge tolerance, as ``_cap_extent``)."""
        if a < 0.0:
            return True
        sphere_r = abs(surf.radius)
        support = surf.aperture_radius + surf.mount_margin + self.cfg.aperture_edge_tol_mm
        rim = math.sqrt(max(sphere_r * sphere_r - support * support, 0.0))
        other = point - (2.0 * a) * d
        return float(dot(other - surf.center, surf.cap_axis)) + radius < rim

    def _inside_cap(self, surf: SurfaceG, point: Array, radius: float) -> bool:
        """Whether every point of a sphere within ``radius`` of its point ``point`` lies inside the cap (the points with
        ``(P - C) . cap_axis >= rim``, within its support): ``radius`` below the point's distance to the cap's rim circle,
        which is at least its distance from the rim's plane along the axis, or from the rim's cylinder across it while it
        stays on the cap's side of the centre."""
        sphere_r = abs(surf.radius)
        support = min(surf.aperture_radius + surf.mount_margin, sphere_r)
        rim = math.sqrt(max(sphere_r * sphere_r - support * support, 0.0))
        rel = point - surf.center
        axial = float(dot(rel, surf.cap_axis))
        across = math.sqrt(max(float(dot(rel, rel)) - axial * axial, 0.0))
        return radius < max(axial - rim, min(support - across, axial))

    def _cap_extent(self, surf: SurfaceG, point: Array, d: Array) -> tuple[float, float]:
        """The range of ``d . (P - point)`` over the points ``P`` of a sphere's cap (centre ``C``, radius ``R``, the points
        with ``(P - C) . cap_axis >= rim``, ``rim`` from its support radius): the sphere's extreme ``+-R`` along ``d`` where
        the cap holds it, else on the rim circle."""
        radius = abs(surf.radius)
        support = surf.aperture_radius + surf.mount_margin + self.cfg.aperture_edge_tol_mm
        rim = math.sqrt(max(radius * radius - support * support, 0.0))
        axis = surf.cap_axis
        along = float(dot(d, axis))
        across = math.sqrt(max(1.0 - along * along, 0.0))
        ring = math.sqrt(max(radius * radius - rim * rim, 0.0))
        top = radius if along * radius >= rim else rim * along + ring * across
        bottom = -radius if -along * radius >= rim else rim * along - ring * across
        offset = float(dot(d, surf.center - point))
        return offset + bottom, offset + top

    def _sag_extent(self, surf: SurfaceG, point: Array, d: Array, reach: float) -> tuple[float, float]:
        """The range of ``z_d = d . (P - point)`` over the points ``P`` of a sag surface (``optics/surfaces/sag.py``) that
        a ray parallel to ``d`` within ``reach`` of ``point`` meets inside the surface's support: the tighter of
        * the box: every point of the support, its sag within the support's sag range;
        * the slab: such a hit lies within ``(reach + dz |n x d|) / |d . n|`` of ``point`` across the axis ``n`` (``dz`` the
          support's sag depth: a vector across ``n`` shrinks by at most ``|d . n|`` across ``d``), so its sag lies in the
          range over that window of the support (``_sag_window``), and ``z_d (d . n) = z - z_point - rho . n`` with
          ``|rho . n| <= reach |n x d|``.
        ``(-inf, inf)`` for an unbounded support."""
        support = support_of(surf)
        if not apertures.bounded(support):
            return -math.inf, math.inf
        n, u_axis, v_axis = surf.normal, surf.u_axis, surf.v_axis
        ou, ov = support.offset_u, support.offset_v
        rho_lo, rho_hi = _support_rho(surf, support)
        z_lo, z_hi = _sag_range(surf, rho_hi)
        dn = float(dot(d, n))
        du, dv = float(dot(d, u_axis)), float(dot(d, v_axis))
        # the box: every point of the support, its sag within the range
        centre = float(dot(d, surf.origin - point)) + ou * du + ov * dv
        spread = apertures.spread(support, du, dv)
        lo = centre - spread + min(z_lo * dn, z_hi * dn)
        hi = centre + spread + max(z_lo * dn, z_hi * dn)
        if abs(dn) > self.cfg.parallel_dot_tol and math.isfinite(z_hi - z_lo):  # the slab, over the window
            side = n - dn * d
            across = math.sqrt(max(float(dot(side, side)), 0.0))
            tilt = reach * across
            rel = point - surf.origin
            window = _sag_window(surf, float(dot(rel, u_axis)), float(dot(rel, v_axis)),
                                 (reach + (z_hi - z_lo) * across) / abs(dn), rho_lo, rho_hi)
            w_lo, w_hi = _sag_range_between(surf, *window)
            z_point = float(dot(rel, n))
            first, second = (w_lo - z_point - tilt) / dn, (w_hi - z_point + tilt) / dn
            lo, hi = max(lo, min(first, second)), min(hi, max(first, second))
        return lo, hi

    def _curved_plane(self, si: int | None, point: Array, d: Array, reach: float | None,
                      side: float, every_hit: bool = False) -> tuple[Array, float, float] | None:
        """A curved surface ``si`` at its point ``point``, for the rays parallel to ``d`` within ``reach`` of it: a plane
        normal and how far back and ahead along ``d`` the surface lies from that plane through ``point`` -- it lies between
        the plane shifted ``back`` against ``d`` and ``ahead`` along it (``inf``: not bounded). ``None``: a plane surface.
        A sphere: the plane across ``d`` (``_reach_along``). A sag surface: of its tangent plane at the point, the surface
        within ``|D| (n_t . n) / |d . n_t|`` of it along a ray (``_sag_tangent``), and the plane across ``d``, the one that
        lies nearer the point on the ``side`` used (+1: ahead, an end surface; -1: back, a start surface).
        ``every_hit``: a sag surface's tangent plane only where it holds every hit of such a ray (``_sag_tangent``'s
        axis window), not only the one that continues the point's, and a sphere's nearest roots only where no such ray
        meets the cap first at the other root of its line (the surface a ray meets first, ``_taken_before``)."""
        if si is None or self.scene.surfaces[si].geometry == "plane":
            return None
        surf = self.scene.surfaces[si]
        back = self._reach_along(si, point, d, reach, -1.0, every_hit)
        ahead = self._reach_along(si, point, d, reach, 1.0, every_hit)
        across = (d, math.inf if back is None else back, math.inf if ahead is None else ahead)
        if surf.geometry == "sag" and reach is not None and math.isfinite(reach):
            tangent = self._sag_tangent(surf, point, d, reach, every_hit)
            used = 2 if side > 0.0 else 1
            if tangent is not None and tangent[used] <= across[used]:
                return tangent
        return across

    def _sag_tangent(self, surf: SurfaceG, point: Array, d: Array, reach: float,
                     every_hit: bool = False) -> tuple[Array, float, float] | None:
        """A sag surface near its point ``point`` as its tangent plane there (unit normal ``n_t``, ``local_normals``) and the
        surface's deviation from it. A hit of a ray parallel to ``d`` within ``reach`` lies within ``r`` of the point
        across the axis, where the profile departs from its tangent plane by at most ``D(r) = H r^2 / 2 + 2 |L| r`` (``H``
        a bound of its Hessian over that window of ``rho``, ``_sag_hessian_bound``; ``L`` the axicon term, by its slope);
        along the ray that is ``D(r) (n_t . n) / |d . n_t|``. ``r`` is the smaller of
        * ``(reach + dz |n x d|) / |d . n|`` (``_sag_extent``'s window; every hit), and
        * where that window does not localize the hits (wider than the support's range of ``rho``; a ray across the axis'
          plane, as on an off-axis paraboloid's reflected leg) and not ``every_hit``, a radius ``r`` with ``F(r) < r``,
          ``F(r) = sqrt(reach^2 + ((D(r) (n_t . n) + reach |n_t x d|) / |d . n_t|)^2)`` the farthest a hit within ``r``
          can lie, the disc of radius ``r`` inside the support (``_room_in_support``): the hit that continues the chief
          ray's as the offset grows cannot leave it. Another hit of the same line (a fold of the surface across the ray,
          the near rim of a steep surface the beam passes) is not taken into account there.
        ``None``: not bounded this way (a ray along the tangent plane, an unbounded support, no such radius; with
        ``every_hit``, an axis window that does not localize the hits)."""
        cfg = self.cfg
        support = support_of(surf)
        if not apertures.bounded(support):
            return None
        n = surf.normal
        dn = float(dot(d, n))
        rel = point - surf.origin
        a_point, b_point = float(dot(rel, surf.u_axis)), float(dot(rel, surf.v_axis))
        normal = sag_local_normals(surf, np.array([a_point]), np.array([b_point]))[0]
        dn_t = float(dot(d, normal))
        if abs(dn_t) <= cfg.parallel_dot_tol:
            return None
        rho_lo, rho_hi = _support_rho(surf, support)
        tilt_n = n - dn * d
        tilt_t = normal - dn_t * d
        across_n = math.sqrt(max(float(dot(tilt_n, tilt_n)), 0.0))
        across_t = math.sqrt(max(float(dot(tilt_t, tilt_t)), 0.0))
        along = abs(float(dot(normal, n)))

        def deviation(radius: float) -> float:
            first, last = _sag_window(surf, a_point, b_point, radius, rho_lo, rho_hi)
            return 0.5 * _sag_hessian_bound(surf, first, last) * radius * radius + 2.0 * abs(surf.linear) * radius

        radius = math.inf
        if abs(dn) > cfg.parallel_dot_tol:
            z_lo, z_hi = _sag_range(surf, rho_hi)
            radius = (reach + (z_hi - z_lo) * across_n) / abs(dn)
        if not radius <= rho_hi - rho_lo:  # (the axis window does not localize the hits: the continuation's, inside the support)
            if every_hit:
                return None
            room = _room_in_support(surf, support, a_point, b_point)

            def farthest(trial: float) -> float:  # (F: the farthest a hit within ``trial`` can lie)
                far = (deviation(trial) * along + reach * across_t) / abs(dn_t)
                return math.sqrt(reach * reach + far * far)

            trial = reach * (1.0 + across_t / abs(dn_t))
            for _ in range(SAG_WINDOW_STEPS):
                if trial >= min(radius, room):
                    break
                if farthest(trial) < trial:
                    # (F increases with r: each F(r) (1 + margin) from a radius with F(r) < r has the same property)
                    for _ in range(SAG_WINDOW_STEPS):
                        smaller = farthest(trial) * (1.0 + PREFILTER_REL_MARGIN)
                        if not smaller < trial:
                            break
                        trial = smaller
                    radius = trial
                    break
                trial *= SAG_WINDOW_GROWTH
        if not math.isfinite(radius):
            return None
        gap = deviation(radius) * along / abs(dn_t)
        if not math.isfinite(gap):
            return None
        if surf.freeform is not None or surf.coefficients or surf.linear != 0.0 or surf.curvature == 0.0:
            return normal, gap, gap
        # a conic profile bends one way (toward +n for c > 0): along the ray the surface lies on one side of the plane
        ahead = (surf.curvature > 0.0) == (dn_t > 0.0)
        return normal, 0.0 if ahead else gap, gap if ahead else 0.0

    def _leg(self, pk: _Packet, end: tuple[Array, Array, float] | None, end_si: int | None = None, *,
             start: tuple[Array, GState] | None = None) -> Leg | None:
        """The packet's present straight leg for the partial-crossing bounds (``Leg``), ending at ``end`` (the chief-ray hit
        point, the surface's normal there and the chief-ray parameter; ``end_si``: that surface, for a curved one; ``None``:
        a leg that leaves the scene). ``start``: where the leg began and its Gaussian beam there, when the packet has moved
        along it to sample a plane (``_open_leg``). A curved start or end surface is taken as a plane shifted as far out as
        the surface reaches within ``K`` beam radii (``_envelope_sigmas``, ``_curved_plane``: a sphere across the ray, a sag
        surface its tangent plane; the share beyond them is below the negligible level). ``None``: no valid Gaussian state,
        or a curved surface whose reach is not bounded (a sag surface without a bounded support, no stated negligible
        level)."""
        g = pk.gstate if start is None else start[1]
        if not g.valid:
            return None
        a_point, d = (pk.o[0] if start is None else start[0]), pk.d[0]
        k_sigma = self._envelope_sigmas(pk)
        cap_s = self._curved_plane(pk.segment_surface, a_point, d,
                                   None if k_sigma is None else k_sigma * widest(g, (0.0,)), -1.0)
        if end is None:
            if cap_s is not None and not math.isfinite(cap_s[1]):
                return None
            normal_s = cap_s[0] if cap_s is not None else (d if pk.segment_normal is None else np.array(pk.segment_normal, dtype=float))
            reach_s = 0.0 if cap_s is None else cap_s[1]
            return Leg(a_point, d, math.inf, normal_s, -reach_s, d, None, -reach_s, math.inf, propagate(g, -reach_s) if reach_s else g,
                       self.index.reach_along(a_point, d))
        point_e, normal_e, s_e = end
        cap_e = self._curved_plane(end_si, point_e, d, None if k_sigma is None else k_sigma * widest(g, (s_e,)), 1.0)
        if (cap_s is not None and not math.isfinite(cap_s[1])) or (cap_e is not None and not math.isfinite(cap_e[2])):
            return None
        reach_s = None if cap_s is None else cap_s[1]
        reach_e = None if cap_e is None else cap_e[2]
        if cap_s is None:
            normal_s = d if pk.segment_normal is None else np.array(pk.segment_normal, dtype=float)
            start_offset = 0.0
        else:
            normal_s, start_offset = cap_s[0], -reach_s  # (the plane through the start point shifted back by reach_s)
        if cap_e is not None and cap_e[0] is d:  # (across the ray: shifted ahead by reach_e)
            normal_e, end_offset = d, s_e + reach_e
        else:
            if cap_e is not None:  # (a tangent plane, shifted ahead by reach_e)
                normal_e, point_e = cap_e[0], point_e + reach_e * d
            dn_e = float(dot(d, normal_e))
            end_offset = float(dot(point_e - a_point, normal_e)) / dn_e if abs(dn_e) > self.cfg.parallel_dot_tol else None
        if cap_s is None and cap_e is None:
            first, length, beam = 0.0, s_e, g
        else:
            first = -(reach_s or 0.0)
            length = s_e + (reach_s or 0.0) + (reach_e or 0.0)
            beam = propagate(g, first)
        return Leg(a_point, d, s_e, normal_s, start_offset, normal_e, end_offset, first, length, beam)

    def _open_leg(self, pk: _Packet, h: Hits) -> Leg | None:
        """The leg so far, while the packet samples a plane (``h``: the rays' hits on it, from where the leg began; ``_sample``
        has moved the packet there): it has no end yet, so it runs out of the scene."""
        t = float(h.t[0])
        return self._leg(pk, None, start=(h.point[0] - t * pk.d[0], propagate(pk.gstate, -t)))

    def _leg_end(self, pk: _Packet, times: Array, crossed: tuple[int, ...], end: tuple[Array, Array, float] | None,
                 ends_at: str | None, end_si: int | None = None) -> None:
        """The straight leg ends (the chief ray reaches a physical surface or an absorbing plane that stops the beam, or leaves
        the scene): every observation plane the chief ray does not cross on it (not among ``crossed``) that part of the beam
        crosses before the leg's end is recorded (``PartialCrossing``, ``decisions.leg_partial_crossings``: with a bound of
        the power that part carries; a part at or below the line's negligible level is not recorded). An absorbing plane
        takes that part out of the beam, whose power from there on follows the chief ray only: the plane and the record's
        bound join the shares taken on the way (``_Packet.take``: the planes behind read ``[P - S, P]``). A plane through the hit point of the surface the
        leg starts from that counted an arrival there is among ``crossed`` (``pk.leg_crossed``): the light of this leg in or
        near it is not added to that reading, which is the power at the surface. A plane that counted none there is a candidate
        like any other. (A later report per side of a plane gives that light to the other side of the reading plane.)"""
        candidates = leg_end_candidates(self.index, crossed, pk.leg_partial)
        if not candidates:
            return
        leg = self._leg(pk, end, end_si)
        decided = leg_partial_crossings(self.index, self._view(pk), times, self._blocking_view(times), candidates, crossed, end_si,
                                        leg, ends_at, self.cfg, self.quad_rule, prefilter=self._prefilter_leg_ends, traced=not self.bounding,
                                        beside=self._beside_end(pk, end_si))
        for decision in decided:
            record = decision.crossing
            if record is None:
                continue
            pk.leg_partial = (*pk.leg_partial, decision.surface_index)
            self.out.partial_crossings.append(record)
            if record.absorbing:
                pk.take(record.surface_id, record.power_W)

    def _beside_end(self, pk: _Packet, end_si: int | None) -> Array | None:
        """Per ray, where a live pupil ray crosses the plane of the physical surface the leg ends at beyond that surface's
        support -- through one of its holes or past its outline (``inf`` for the others; ``None``: no such ray, or the
        surface is not a plane, or the beam has no valid Gaussian state there). That light is the surface's own: where the
        chief ray meets the surface, the Gaussian beam's part beyond the support is booked there and goes on as its untraced
        branch (``_spill_beside``; on the opaque rim, ``_clipped_branch``), followed by a probe trace -- so it does not also
        cross the planes beyond as part of this leg (``decisions.leg_partial_crossings``: those rays' light ends at the
        surface's plane). Without a valid Gaussian state (or behind a central obstruction) no such branch carries it, and
        the rays keep counting."""
        if end_si is None or not pk.gstate.valid or pk.chief_obstructed:
            return None
        surf = self.scene.surfaces[end_si]
        if surf.geometry != "plane" or surf.interaction.kind == "observe" or not apertures.bounded(surf.aperture):
            return None
        rows = np.flatnonzero(pk.alive & self._annulus_rays(pk))
        if rows.size == 0:
            return None
        cfg = self.cfg
        denom = dot(pk.d[rows], surf.normal)
        numer = dot(surf.origin - pk.o[rows], surf.normal)
        ok = np.abs(denom) > cfg.parallel_dot_tol
        t = np.where(ok, numer / np.where(ok, denom, 1.0), np.inf)
        ok &= t > cfg.ray_min_advance_mm
        rel = pk.o[rows] + np.where(ok, t, 0.0)[:, None] * pk.d[rows] - surf.origin
        outside = ok & ~aperture_inside(support_of(surf), dot(rel, surf.u_axis), dot(rel, surf.v_axis), cfg.aperture_edge_tol_mm)
        if not bool(outside.any()):
            return None
        beside = np.full(pk.o.shape[0], np.inf)
        beside[rows[outside]] = t[outside]
        return beside

    def _cursor(self, pk: _Packet) -> tuple[Array, float, float | None]:
        """The chief ray's last event on this leg (``_Packet.leg_t`` from where the leg began): its point, the beam's power
        and largest Gaussian radius there."""
        t = pk.leg_t
        a = self._extinction(pk)
        power = pk.chief_power()
        if a > 0.0 and t > 0.0:  # (libm-free: the same bits in every runtime)
            power = power * exp_float(-a * t)
        g = propagate(pk.gstate, t)
        radius = max(g.radius("x"), g.radius("y")) if g.valid and not pk.chief_obstructed else None
        return pk.o[0] + t * pk.d[0], power, radius

    def _to_cursor(self, pk: _Packet) -> None:
        """Move the chief ray to its last event on this leg (a path that ends between surfaces: it left the scene, or the
        trace budget stopped it there): its length, optical path, Gaussian beam and power (the bulk absorption on the way
        booked)."""
        t = pk.leg_t
        if t == 0.0:
            return
        a = self._extinction(pk)
        if a > 0.0:
            power = pk.chief_power()
            self._flow(pk, SINK_BULK_ABSORPTION, power * -float(expm1(-a * t)), None, pk.medium)
            pk.jones = pk.jones.copy()
            pk.jones[0] = pk.jones[0] * exp_float(-0.5 * a * t)
        pk.o, pk.opl, pk.length = pk.o.copy(), pk.opl.copy(), pk.length.copy()
        pk.o[0] = pk.o[0] + t * pk.d[0]
        pk.opl[0] = pk.opl[0] + pk.n * t
        pk.length[0] = pk.length[0] + t
        self._advance_group(pk, t)
        pk.gstate = propagate(pk.gstate, t)
        pk.leg_t = 0.0

    def _sample(self, pk: _Packet, hit: Hits, rows: NDArray[np.bool_]) -> tuple:
        """Put the rays ``rows`` where each crosses an observation plane (``hit``, from where the leg began), for that plane
        to sample: positions, optical paths, lengths, the chief ray's power after the bulk absorption on the way, Gaussian beam
        and group delay. Returns what ``_restore`` puts back (observation planes never move a ray)."""
        held = (pk.o, pk.opl, pk.length, pk.jones, pk.gstate, pk.gpl, pk.gdd, pk.dispersive, pk.segment_mm, pk.alive, pk.power_scope,
                pk.met_first, pk.left_beyond, pk.taken_at, pk.taken_upper_W, pk.taken_lower_W)
        dt = np.where(rows, hit.t, 0.0)
        pk.o = np.where(rows[:, None], hit.point, pk.o)
        pk.opl = pk.opl + self._ray_index(pk) * dt
        pk.length = pk.length + dt
        a = self._extinction(pk)
        if a > 0.0:
            pk.jones = pk.jones * exact_exp(-0.5 * a * dt)[:, None]
        pk.gstate = propagate(pk.gstate, float(dt[0]))
        self._advance_group(pk, float(dt[0]))
        pk.alive = rows.copy()
        return held

    @staticmethod
    def _restore(pk: _Packet, held: tuple) -> None:
        (pk.o, pk.opl, pk.length, pk.jones, pk.gstate, pk.gpl, pk.gdd, pk.dispersive, pk.segment_mm, pk.alive,
         pk.power_scope, pk.met_first, pk.left_beyond, pk.taken_at, pk.taken_upper_W, pk.taken_lower_W) = held

    def _pending_crossings(self, pk: _Packet) -> bool:
        """Cross the observation planes through the point where ``pk`` left a physical surface (``pk.pending_observers``)
        when it leaves toward their incident side, as the beam crosses a plane just on the incident side on its way back.
        Each plane samples every ray where that ray crosses it; no ray moves (a plane never moves a ray). Returns True when
        the path ended there."""
        cfg, surfaces = self.cfg, self.scene.surfaces
        pending, pk.pending_observers = pk.pending_observers, ()
        for si, incident_sign, surface_si in pending:
            surf = surfaces[si]
            along = float(dot(pk.d[0], surf.normal))
            if abs(along) <= cfg.parallel_dot_tol or (along > 0.0) == (incident_sign > 0.0):
                continue  # it continues past the plane (or along it): no second crossing here
            # an observation plane is virtual: crossing it is not a physical interaction (max_interactions_per_path)
            denom = dot(pk.d, surf.normal)
            ok = np.abs(denom) > cfg.parallel_dot_tol
            t = np.where(ok, dot(surf.origin - pk.o, surf.normal) / np.where(ok, denom, 1.0), np.inf)
            reached = pk.alive & ok
            step = np.where(reached, t, 0.0)
            point = pk.o + step[:, None] * pk.d
            rel = point - surf.origin
            hit = Hits(np.where(reached, t, np.inf), point, np.broadcast_to(surf.normal, pk.o.shape).copy(),
                       np.ones(pk.o.shape[0], dtype=bool), dot(rel, surf.u_axis), dot(rel, surf.v_axis))
            start, power_start, w_start = self._cursor(pk)
            live = pk.alive.copy()
            held = self._sample(pk, hit, reached)
            self._segment(pk, start, point[0], float(step[0]) - pk.leg_t, surf, power_start=power_start, w_start=w_start,
                          w_end=_max_radius(pk))
            if self.recorder is not None:
                self.recorder.packet_advanced(pk, surf.surface_id)
            self.out.trace_events.append({"path": pk.path_id, "surface": surf.surface_id, "t": float(step[0])})
            # (a beam leaving a surface toward the side it came from was reflected there, on this path or a split-off one)
            pk.coincidence = SurfaceCoincidenceRecord(surfaces[surface_si].surface_id, "RETURNING", "REFLECTIVE")
            terminated = self._interact(pk, surf, hit)
            pk.coincidence = None
            if terminated is not None:
                if self.recorder is not None:
                    self.recorder.packet_interacted(pk)
                self._finish(pk, terminated, surf.surface_id)
                return True
            self._restore(pk, held)
            pk.alive = live
            if self.recorder is not None:
                self.recorder.packet_interacted(pk)
            pk.leg_crossed = (*pk.leg_crossed, si)
            pk.leg_t = float(step[0])
        return False

    def _stops_beam(self, pk: _Packet, si: int, hit: Hits, rows: NDArray[np.bool_]) -> bool:
        """Whether the absorbing plane ``si`` stops the beam where the chief ray crosses it (it counts the beam: its active
        area takes it, ``decisions.counts_on_active_area``; behind a central obstruction, the annulus is not wholly beside it)."""
        surf = self.scene.surfaces[si]
        if not surf.interaction.absorbing:
            return False
        held = self._sample(pk, hit, rows)
        cos_i = abs(float(dot(pk.d[0], surf.normal)))
        stops = self._annulus_on_area(pk, surf, hit, cos_i)[0] is not False if pk.chief_obstructed else \
            counts_on_active_area(pk.gstate, surf, hit, cos_i, self.cfg, self.quad_rule)[0]
        self._restore(pk, held)
        return bool(stops)

    def _observation_event(self, pk: _Packet, crossing: list[int], times: Array) -> bool:
        """The chief ray's next crossing is observation planes only (``decisions.observation_crossing``). They are virtual: none
        moves, turns, retires or counts a ray, so each samples the beam independently of the others -- every live ray that
        crosses it before the surface that would stop or turn that ray, where that ray crosses it (its point, optical path
        length, length and power there, also in the ray data), whatever the order the planes are declared in or the ray
        crosses them in. No ray moves: the chief ray too stays where the leg began (``_Packet.leg_t`` keeps how far along the
        leg its last event is), so a plane the chief ray crosses later samples each ray where it crosses it. An absorbing
        plane that counts the beam ends the path after the transparent planes: they sample only the rays that reach them no
        later (relative to the chief ray) than it, and when part of the beam meets it first their power is out of model
        scope (``power_scope``). Returns True when the path ended."""
        cfg, surfaces = self.cfg, self.scene.surfaces
        hits = [intersect(surfaces[i], pk.o, pk.d, cfg) for i in crossing]
        live = pk.alive.copy()
        t_block, nearest = self._first_blocking(times, (*crossing, *pk.leg_crossed))
        reach = [live & np.isfinite(hit.t) & (hit.t < t_block) for hit in hits]
        for rows in reach:
            rows[0] = live[0]  # (the chief ray defines the crossing)
        first = min(range(len(hits)), key=lambda k: (float(hits[k].t[0]), k))
        t0 = float(hits[first].t[0])
        stop = next((k for k, si in enumerate(crossing) if self._stops_beam(pk, si, hits[k], reach[k])), None)
        scope: dict[int, tuple[str, ...]] = {}
        beyond: dict[int, tuple[str, ...]] = {}
        taken: dict[int, TakenShares] = {}
        if stop is not None:
            # the leg ends at that plane: the planes part of the beam crosses first, and the transparent planes of this
            # crossing sample only the rays that reach them no later (relative to the chief ray) than it does
            end_hit = hits[stop]
            self._leg_end(pk, times, (*pk.leg_crossed, *crossing),
                          (end_hit.point[0], surfaces[crossing[stop]].normal, float(end_hit.t[0])), surfaces[crossing[stop]].surface_id,
                          crossing[stop])
            for k, si in enumerate(crossing):
                if surfaces[si].interaction.absorbing:
                    continue
                gap = hits[k].t - end_hit.t
                later = np.isfinite(gap) & (gap - gap[0] > cfg.ray_min_advance_mm)
                reach[k] = reach[k] & ~later
        for k, si in enumerate(crossing):
            taken[k] = self._taken_before(pk, times, si, hits[k], t_block, nearest)
            if surfaces[si].interaction.absorbing:
                # (an absorbing plane detects the beam's share on its active area: light taken before it lowers that by at
                # most the share taken, while the share certainly taken may be light that would have missed the area)
                taken[k] = replace(taken[k], lower_W=None)
            scope[k], beyond[k] = taken[k].scope, taken[k].beyond
        start, power_start, w_start = self._cursor(pk)
        power_leg = pk.chief_power()
        for ordinal, (si, hit) in enumerate(zip(crossing, hits)):
            observed = surfaces[si]
            held = self._sample(pk, hit, reach[ordinal])
            if scope.get(ordinal):
                pk.power_scope = (*pk.power_scope, *(at for at in scope[ordinal] if at not in pk.power_scope))
                pk.left_beyond = beyond.get(ordinal, ())
                pk.met_first = tuple(at for at in scope[ordinal] if not at.startswith(OBSERVATION_PREFIX) and at not in pk.left_beyond)
            if ordinal in taken and taken[ordinal].bounded:
                pk.taken_at, pk.taken_upper_W, pk.taken_lower_W = taken[ordinal].bounded, taken[ordinal].upper_W, taken[ordinal].lower_W
            if ordinal == 0:  # (one step: from the last event to the chief ray's first crossing of these planes)
                self._segment(pk, start, hits[first].point[0], t0 - pk.leg_t, observed, power_start=power_start,
                              w_start=w_start, w_end=_max_radius(pk))
            else:
                self._segment(pk, hits[first].point[0], hits[first].point[0], 0.0, observed, w_start=_max_radius(pk),
                              w_end=_max_radius(pk))
            if self.recorder is not None:
                self.recorder.packet_advanced(pk, observed.surface_id)
            self.out.trace_events.append({"path": pk.path_id, "surface": observed.surface_id,
                                          "t": t0 - pk.leg_t if ordinal == 0 else 0.0})
            terminated = self._interact(pk, observed, hit)
            if terminated is not None:
                a = self._extinction(pk)
                if a > 0.0:  # (the bulk absorption from where the leg began to this plane)
                    self._flow(pk, SINK_BULK_ABSORPTION, power_leg * -float(expm1(-a * float(hit.t[0]))), observed, pk.medium)
                if self.recorder is not None:
                    self.recorder.packet_interacted(pk)
                self._finish(pk, terminated, observed.surface_id)
                return True
            self._restore(pk, held)
            pk.alive = live.copy()
            if self.recorder is not None:
                self.recorder.packet_interacted(pk)
        pk.leg_crossed = (*pk.leg_crossed, *crossing)
        pk.leg_t = t0
        return False

    def _taken_before(self, pk: _Packet, times: Array, si: int, hit: Hits, t_block: Array,
                      nearest: NDArray[np.intp]) -> TakenShares:
        """The surfaces that take part of the beam out of it, or turn it, before it reaches the plane ``si`` (``hit``; an
        absorbing plane as a transparent one) that the chief ray crosses on this leg (``decisions.taken_before``: absorbing planes, surfaces a live
        pupil ray meets first -- ``t_block`` / ``nearest``, each ray's first surface that stops or turns it --, the
        physical surface the chief ray meets next, ``_next_surface``, and the reflecting surface the leg began at,
        ``_start_surface``): the power the plane records (the beam's, which follows its chief ray) is then not the power that
        reaches it (``power_scope``). Returns all of them, and apart the start surface when it is one; or, when every one
        counted states a supported bound of the power it takes (``TakenBefore.bound_W``), none of them as out of scope but
        all as taken before the plane with the sum of their bounds (``TakenShares``)."""
        leg = self._leg(pk, (hit.point[0], self.scene.surfaces[si].normal, float(hit.t[0])))
        def beside(other: int) -> float | None:
            # (a surface beside the chief ray a pupil ray meets first; not a face of the body the beam travels in: a beam
            # that meets its own body's other face beside the chief ray may leave the body where no surface is traced)
            surf = self.scene.surfaces[other]
            if pk.medium in self._interior.get(surf.owner_id, ()):
                return None
            return possible_meeting_share(surf, leg, self._envelope_sigmas(pk), self.cfg)  # type: ignore[arg-type]

        lead_surface = self._next_surface(pk, times, leg)
        weighed = taken_before(self.index, self._view(pk), self._blocking_view(times), si, hit.t, t_block, nearest, leg,
                               pk.leg_crossed, lead_surface, self._start_surface(pk, leg), self.cfg, self.quad_rule, beside)
        # (surfaces beside the chief ray the beam's tail may meet before the plane while no pupil ray does)
        lead = None if lead_surface is None else self._surface_index.get(lead_surface.surface_id)
        tail = self._tail_surfaces(pk, leg, {entry.surface_id for entry in weighed}, lead)
        weighed = (*weighed, *(TakenBefore(self.scene.surfaces[i].surface_id, TAIL_BESIDE, None, True,
                                           None if share is None else beam_part_W(self._view(pk), share)) for i, share in tail))
        counted = [entry for entry in weighed if entry.taken]
        if counted and all(entry.bound_W is not None for entry in counted):
            # every share taken is bounded: the plane reads the beam, whose crossing power lies in [P - S, P]
            share = correctly_rounded_sum(float(entry.bound_W) for entry in counted)  # type: ignore[arg-type]
            # (a line certain to meet a stopping surface first may meet another counted surface before it that turns light
            # rather than stopping it -- a lens, a mirror beside the chief ray -- and reach the plane after all: then no share
            # is certain. The surface the leg began at sends its lines on before every such meeting)
            turning = any(entry.lower_W is None and entry.why != START_SURFACE and not self._stops_light(entry.surface_id)
                          for entry in counted)
            certain = [] if turning else [entry.lower_W for entry in counted if entry.lower_W is not None]
            least = min(correctly_rounded_sum(certain), share) if certain else None
            return TakenShares((), (), tuple(entry.surface_id for entry in counted), min(share, pk.chief_power()),
                               None if least is None or least <= 0.0 else min(least, pk.chief_power()))
        found = tuple(entry.surface_id for entry in counted)
        beyond = tuple(entry.surface_id for entry in counted if entry.why == START_SURFACE)
        return TakenShares(found, beyond)

    def _tail_surfaces(self, pk: _Packet, leg: Leg | None, skip: set[str], lead: int | None) -> list[tuple[int, float | None]]:
        """The physical surfaces beside the chief ray that the beam's energy-flow lines within ``K`` radii may meet on the leg
        ``leg`` before its end plane while no pupil ray need meet them -- launched on a converging beam the pupil rays close
        toward its waist and pass well inside the Gaussian envelope there --, with the fraction of the beam that may meet each
        first (``lower_share.possible_meeting_share``; ``None``: not bounded), those above the line's negligible level only.
        A surface's balls (``decisions._surface_balls``) must come within ``sqrt(2) K`` times the beam's widest radius of the
        chief ray's stretch the leg's light occupies (``reach_stretch``) and not lie wholly beyond the end plane or behind the
        start plane. Left out: ``skip`` (ids weighed already), the surface the chief ray meets next (``lead``) and the one the
        leg began at, the other surfaces of their elements (light reaching them first passes beside the surface met: its clear
        aperture books it), and the faces of the body the beam travels in. None without a leg (no valid Gaussian state), for
        a probe, a beam of unknown power or behind a central obstruction (``_met_beside_obstruction``)."""
        if (leg is None or self.bounding or pk.power_bound_W is not None or pk.chief_obstructed or not pk.gstate.valid
                or leg.end_offset is None or not math.isfinite(leg.length)):
            return []
        k_sigma = self._envelope_sigmas(pk)
        if k_sigma is None:
            return []
        cfg, surfaces = self.cfg, self.scene.surfaces
        ends = tuple(i for i in (lead, self._surface_index.get(pk.segment_start or "")) if i is not None)
        owners = {surfaces[i].owner_id for i in ends}
        s_lo, s_hi = reach_stretch(leg, k_sigma, cfg)
        w_most = widest(leg.beam, (min(s_lo - leg.first, 0.0), max(s_hi - leg.first, leg.length))) * (1.0 + PREFILTER_REL_MARGIN)
        reach = math.sqrt(2.0) * k_sigma * w_most + cfg.aperture_edge_tol_mm
        level = self._negligible_level(pk)
        d = leg.d
        a_point, span = leg.start + s_lo * d, s_hi - s_lo
        end_point, start_point = leg.start + leg.end_offset * d, leg.start + leg.start_offset * d
        side_e = 1.0 if float(dot(d, leg.end_normal)) > 0.0 else -1.0
        side_s = 1.0 if float(dot(d, leg.start_normal)) > 0.0 else -1.0
        tol = cfg.aperture_edge_tol_mm
        power = pk.chief_power()
        out: list[tuple[int, float | None]] = []
        for si in self.index.physical_ids:
            surf = surfaces[si]
            if (si in ends or surf.surface_id in skip or surf.owner_id in owners
                    or pk.medium in self._interior.get(surf.owner_id, ())):
                continue
            if surf.geometry == "plane" and not apertures.bounded(support_of(surf)):
                along = float(dot(d, surf.normal))
                f0 = float(dot(a_point - surf.origin, surf.normal))
                f1 = f0 + span * along
                near = f0 * f1 <= 0.0 or min(abs(f0), abs(f1)) <= reach * math.sqrt(max(1.0 - along * along, 0.0)) + tol
            else:
                near = False
                balls = self._balls.get(si)
                if balls is None:  # (the scene does not move during the trace)
                    balls = self._balls[si] = _surface_balls(surf)
                # (a plane surface lies at least as far from the stretch as its plane when the stretch keeps to one side)
                plane_gap = 0.0
                if surf.geometry == "plane":
                    f0 = float(dot(a_point - surf.origin, surf.normal))
                    f1 = f0 + span * float(dot(d, surf.normal))
                    plane_gap = min(abs(f0), abs(f1)) if f0 * f1 > 0.0 else 0.0
                    # (its support, exactly: wholly beyond the end plane or wholly behind the start plane, nothing of the leg's
                    # light meets it on the leg -- a parallel plate beside the leg's start or end, closer than its radius)
                    if (_support_side(surf, end_point, leg.end_normal, side_e, tol) > 0
                            or _support_side(surf, start_point, leg.start_normal, side_s, tol) < 0):
                        continue
                for centre, radius in balls:
                    rel = centre - a_point
                    along = min(max(float(dot(rel, d)), 0.0), span)
                    off = rel - along * d
                    gap = max(math.sqrt(float(dot(off, off))) - radius, plane_gap) - tol
                    # (a line at profile place xi lies within |xi| W of the chief ray on the stretch: one reaching the ball there
                    # has |xi| >= gap / W, a share of at most exp(-2 (gap / W)^2) of the beam; the lines beyond the K square, which
                    # may reach past the stretch, hold at most half the negligible level, ``envelope_sigmas``)
                    if (gap <= reach and not (gap > 0.0 and exp_float(-2.0 * (gap / w_most) ** 2) * power <= 0.5 * level)
                            and side_e * float(dot(centre - end_point, leg.end_normal)) - radius <= tol
                            and side_s * float(dot(centre - start_point, leg.start_normal)) + radius >= -tol):
                        near = True
                        break
            if not near:
                continue
            share = possible_meeting_share(surf, leg, k_sigma, cfg)
            if share is not None and self._negligible(pk, share * power):
                continue
            out.append((si, share))
        return out

    def _tail_beside_chief(self, pk: _Packet, lead: int, lead_hit: Hits, met: tuple[str, ...] = ()) -> None:
        """The leg ends where the chief ray meets ``lead``: a physical surface beside the chief ray that the beam's tail may
        meet first (``_tail_surfaces``; one a pupil ray meets on this leg is counted already, ``_met_beside_chief``:
        ``met``) takes light that the beam's power, which follows its chief ray, still counts. Where the share it may take
        is bounded and the surface stops the light it meets (``_stops_light``: the share never reaches a plane again) it
        joins the shares taken on the way (``_Packet.take``: the planes behind read ``[P - S, P]``) with a
        ``POWER_TAKEN_BEFORE_PLANE`` warning stating the bound; otherwise -- no bound, or a surface that turns the light (a
        mirror, a lens or plate face, a splitter, a scattering absorber), which may send it to a plane behind, so ``P`` is no
        upper end there -- it joins ``power_scope`` (the beam's power behind it is out of model scope) with an
        ``OUT_OF_MODEL_SCOPE`` issue."""
        if self.bounding or pk.power_bound_W is not None or pk.chief_obstructed or not pk.gstate.valid:
            return
        leg = self._leg(pk, (lead_hit.point[0], lead_hit.normal[0], float(lead_hit.t[0])), lead)
        p_in = pk.chief_power()
        lead_id = self.scene.surfaces[lead].surface_id
        for si, share in self._tail_surfaces(pk, leg, {*pk.power_scope, *met}, lead):
            surface_id = self.scene.surfaces[si].surface_id
            action = ("move the element or the beam so the beam clears it by more of its radius, or read the power at a plane "
                      "before that surface")
            if share is None or not self._stops_light(surface_id):
                pk.power_scope = (*pk.power_scope, surface_id)
                why = "is not bounded" if share is None else "is turned by it and may reach a plane behind"
                self._issue(make_issue(
                    ErrorCode.OUT_OF_MODEL_SCOPE, Status.OUT_OF_MODEL_SCOPE,
                    f"the share of beam {pk.path_id} that may meet {surface_id} before {lead_id} {why}: its power "
                    f"behind {lead_id} is not computed",
                    location=f"surface:{surface_id}", path_id=pk.path_id, surface_ids=[surface_id], next_surface_id=lead_id,
                    suggested_action=action))
                continue
            pk.take(surface_id, p_in * share)
            self._issue(make_issue(
                ErrorCode.POWER_TAKEN_BEFORE_PLANE, Status.WARNING,
                f"at most {p_in * share:.4g} W of beam {pk.path_id} may meet {surface_id} before {lead_id}",
                location=f"surface:{surface_id}", path_id=pk.path_id, surface_ids=[surface_id], next_surface_id=lead_id,
                tail_fraction_upper=share, tail_power_upper_W=p_in * share, suggested_action=action))

    def _stops_light(self, surface_id: str) -> bool:
        """Whether a surface stops every light it meets: an absorber that does not scatter, an absorbing observation plane."""
        surf = next((s for s in self.scene.surfaces if s.surface_id == surface_id), None)
        if surf is None:
            return False
        inter = surf.interaction
        return (inter.kind == "absorber" and inter.scatter is None) or (inter.kind == "observe" and bool(inter.absorbing))

    def _next_surface(self, pk: _Packet, times: Array, leg: Leg | None) -> NextSurface | None:
        """The physical surface the chief ray meets next (``None``: none, or no leg), as ``taken_before`` bounds the part of
        the beam that meets it first: the surface as its own plane at the chief ray's hit, a curved one as a plane shifted
        as far back as it reaches within the beam's envelope (``_curved_plane`` with ``every_hit``: for a sag surface its
        tangent plane only where the axis window holds every hit of the envelope's rays, else the plane across the ray
        before every point of the support those rays can meet; for a sphere the plane across the ray before the root of
        each line nearest the chief ray's hit while the envelope stays inside the cap there and no envelope line can meet
        the cap first at its other root -- the near rim of a steep cap the beam passes, ``_other_root_clear`` --, else
        before the whole cap; ``NextSurface.sheet`` ``None``: that reach is not bounded). The plane is taken without its
        support, so the share may be larger than the part that meets the surface."""
        surfaces = self.scene.surfaces
        ahead = [i for i in self.index.physical_ids if np.isfinite(times[i, 0])]
        if not ahead:
            return None
        lead = min(ahead, key=lambda i: (float(times[i, 0]), i))
        if leg is None:  # (no Gaussian bound: taken_before weighs where the surface's support lies)
            return NextSurface(surfaces[lead].surface_id, None)

        def sheet() -> Sheet | None:  # resolved only when taken_before weighs this surface
            lead_hit = intersect(surfaces[lead], pk.o[:1], pk.d[:1], self.cfg)
            point, d = lead_hit.point[0], pk.d[0]
            k_sigma = self._envelope_sigmas(pk)
            reach = None if k_sigma is None else k_sigma * widest(pk.gstate, (float(lead_hit.t[0]),))
            cap = self._curved_plane(lead, point, d, reach, -1.0, every_hit=True)
            if cap is None:
                return Sheet(point, lead_hit.normal[0])
            if math.isfinite(cap[1]):
                return Sheet(point - cap[1] * d, cap[0])
            return None

        def region() -> float | None:  # resolved only for a sag surface counted as taking a share
            return _sag_region_share(surfaces[lead], leg, self._envelope_sigmas(pk), self.cfg)

        inter = surfaces[lead].interaction
        stop_side: int | None = None  # (the sign of the surface function on the side that stops what it meets)
        if inter.kind == "absorber":
            stop_side = EITHER_SIDE
        elif inter.kind == "mirror" and inter.reflective_side != "both" and inter.coating is None:
            stop_side = -1 if inter.reflective_side == "positive" else 1

        def certain() -> float:  # resolved only for a stopping surface counted with a bound
            return certain_meeting_share(surfaces[lead], leg, self._envelope_sigmas(pk), stop_side, self.cfg)  # type: ignore[arg-type]

        return NextSurface(surfaces[lead].surface_id, sheet, region if surfaces[lead].geometry == "sag" else None,
                           certain if stop_side is not None else None)

    @staticmethod
    def _turned_back(pk: _Packet) -> bool:
        """Whether the leg began with a reflection (the last interaction in the packet's lineage that is not an observation:
        a mirror, a coating or Fresnel reflection, a splitter's or an attenuator's reflected beam, total internal
        reflection)."""
        for entry in reversed(pk.lineage):
            event = entry.rsplit(":", 1)[-1]
            if event != "observe":
                return "reflect" in event or event in TIR_EVENTS
        return False

    def _start_surface(self, pk: _Packet, leg: Leg | None) -> StartSurface | None:
        """The physical surface the leg began at, when it reflected the beam (``_turned_back``; ``None`` otherwise), with the
        plane standing for it in ``decisions.start_beyond``: its own plane, a curved one as a plane (``_curved_plane``: a
        sphere's across the ray, a sag surface's across it or its tangent plane at the chief ray's start) shifted as far
        beyond the chief ray's start as the surface reaches within the beam's envelope (``None``: not bounded); for the
        energy flow (``decisions.start_flow_share``) the same with the reach grown to hold the lines within ``K`` radii over
        the stretch they meet it on, and for a sphere also its tangent plane (``_sphere_tangent``). Without a leg (no valid
        Gaussian state) the surface without planes: only the pupil rays decide."""
        if pk.segment_start is None or not self._turned_back(pk):
            return None
        if leg is None:  # (no Gaussian bound: only the pupil rays that do not cross the plane tell, ``taken_before``)
            return StartSurface(pk.segment_start, None)
        a_point, d = leg.start, leg.d
        if pk.segment_surface is None:
            normal_s = d if pk.segment_normal is None else np.array(pk.segment_normal, dtype=float)
            return StartSurface(pk.segment_start, (normal_s, 0.0), (normal_s, 0.0))
        segment_surface = pk.segment_surface

        def plane() -> tuple[Array, float] | None:  # resolved only when taken_before weighs this surface
            k_sigma = self._envelope_sigmas(pk)
            cap = self._curved_plane(segment_surface, a_point, d, None if k_sigma is None else
                                     k_sigma * widest(pk.gstate, (0.0,)), 1.0)
            if cap is None or not math.isfinite(cap[2]):
                return None
            return (cap[0], cap[2])  # (the plane through start + ahead d)

        def settled_plane(stand_in: Callable[[float], tuple[Array, float, float] | None]) -> tuple[Array, float] | None:
            # (a plane standing for the surface for the energy-flow lines within K radii: its reach grown until it holds K
            # times the beam's widest radius over the stretch from the surface's back to the end of the stretch the lines
            # meet the plane on, ``decisions.flow_stretch``)
            k_sigma = self._envelope_sigmas(pk)
            if k_sigma is None:
                return None
            reach = k_sigma * widest(pk.gstate, (0.0,))
            for _ in range(SETTLE_STEPS):
                cap = stand_in(reach)
                if cap is None or not (math.isfinite(cap[1]) and math.isfinite(cap[2])):
                    return None
                stretch = flow_stretch(leg, cap[0], cap[2], k_sigma, self.cfg)
                if stretch is None:
                    return None
                need = k_sigma * widest(pk.gstate, (min(stretch[0], -cap[1]), stretch[1]))
                if need <= reach:
                    return (cap[0], cap[2])
                reach = SETTLE_MARGIN * need
            return None

        def flow() -> tuple[Array, float] | None:
            return settled_plane(lambda reach: self._curved_plane(segment_surface, a_point, d, reach, 1.0))

        tangent = None
        if self.scene.surfaces[segment_surface].geometry == "sphere":
            def tangent() -> tuple[Array, float] | None:
                return settled_plane(lambda reach: self._sphere_tangent(self.scene.surfaces[segment_surface], a_point, d, reach))

        return StartSurface(pk.segment_start, plane, flow, tangent)

    def _sphere_tangent(self, surf: SurfaceG, point: Array, d: Array, reach: float) -> tuple[Array, float, float] | None:
        """A sphere near its point ``point`` as its tangent plane there (unit normal ``n_t = (point - C) / R``) and how far back
        and ahead along ``d`` it lies from that plane for the lines parallel to ``d`` within ``reach`` of ``point`` that meet
        it at the root that continues the point's. A point ``X`` of the sphere lies ``|X - point|^2 / (2 R)`` from the
        tangent plane, toward the centre. Such a hit lies within ``l`` of the point when ``F(l) < l``, ``F(l) = sqrt(reach^2
        + ((reach sin + l^2 / (2 R)) / cos)^2)`` the farthest a hit within ``l`` can lie (``cos = |d . n_t|``; as ``rho``
        grows the root cannot leave the disc), so the sphere lies within ``l^2 / (2 R cos)`` of the plane along ``d``, on
        the centre's side; and only where the cap holds every point within ``l`` (``_inside_cap``). ``None``: no such ``l``
        (a ray near grazing), or the cap's rim within it."""
        radius = abs(surf.radius)
        rel = point - surf.center
        n_t = rel / math.sqrt(float(dot(rel, rel)))
        dn = float(dot(d, n_t))
        cos = abs(dn)
        if cos <= self.cfg.parallel_dot_tol:
            return None
        sin = math.sqrt(max(1.0 - cos * cos, 0.0))

        def farthest(length: float) -> float:
            along = (reach * sin + length * length / (2.0 * radius)) / cos
            return math.sqrt(reach * reach + along * along)

        length = reach / cos
        for _ in range(SAG_WINDOW_STEPS):
            if length >= radius:
                return None
            if farthest(length) < length:
                break
            length *= SAG_WINDOW_GROWTH
        else:
            return None
        if not self._inside_cap(surf, point, length):
            return None
        shift = length * length / (2.0 * radius * cos)
        # (toward the centre: behind the plane along d where d leaves it outward, d . n_t > 0, else ahead)
        return (n_t, shift, 0.0) if dn > 0.0 else (n_t, 0.0, shift)

    def _off_area_far(self, pk: _Packet, times: Array) -> None:
        """The chief ray does not cross an observation plane with an active area beyond the scene's reach along it
        (``SceneIndex.reach_along``, which holds every active area): such a crossing lies off the area, and it is where a
        ray nearly parallel to the plane meets it (10^14 mm on for a ray 2e-12 off parallel to a plane 200 mm away), so
        whether it exists at all would follow the last bits of the ray's direction (``parallel_dot_tol``). What part of the
        beam crosses the plane on its area is the leg's partial crossing (``decisions.leg_partial_crossings``), as for a chief ray
        parallel to the plane."""
        chief = times[self._area_planes, 0]
        finite = np.isfinite(chief)
        if not finite.any():
            return
        reach = self.index.reach_along(pk.o[0], pk.d[0])
        beyond = [i for i, t, ok in zip(self._area_planes, chief.tolist(), finite.tolist()) if ok and t > reach]
        if beyond:
            times[beyond, 0] = np.inf

    def _trace(self, pk: _Packet) -> None:
        cfg, scene = self.cfg, self.scene
        surfaces = scene.surfaces
        while True:
            if pk.pending_observers and self._pending_crossings(pk):
                return
            if not self.schedule.take_step(pk.path_id, pk.src.source_id):  # the step budget is spent: the path stops here
                self._to_cursor(pk)
                self._finish(pk, TRACE_BUDGET, None)
                return
            if pk.medium in self._gradients:  # (in a GRIN rod: one closed-form leg to the end face ahead, then that face)
                if self._gradient_leg(pk):
                    return
                continue
            n_rays = pk.o.shape[0]
            times = np.full((len(surfaces), n_rays), np.inf)
            if self._planes is not None:
                times[self._plane_ids] = self._planes.times(pk.o, pk.d, cfg)
            if self._spheres is not None:
                times[self._sphere_ids] = self._spheres.times(pk.o, pk.d, cfg)
            for si in self._other_ids:
                times[si] = intersect(surfaces[si], pk.o, pk.d, cfg).t
            times[:, ~pk.alive] = np.inf
            if pk.leg_crossed:  # (planes the chief ray crossed on this leg: no ray meets them again before the leg ends)
                times[list(pk.leg_crossed)] = np.inf
            if pk.passes_owner is not None:  # light beside an element: none of its surfaces is met on this pass
                times[self._owners == pk.passes_owner] = np.inf
            if self._area_planes:
                self._off_area_far(pk, times)
            if pk.chief_obstructed and self._filled:
                self._filled_chief(pk, times)
            self._set_blocking(pk, times)
            if pk.chief_obstructed and self._annulus_diverges(pk, times):
                return
            if self._watch and pk.power_bound_W is None:  # (also for a beam that leaves the scene: its tail may still meet a rated element on the way)
                self._beside(pk, times, float(np.min(times[:, 0])))
            if not surfaces or not np.isfinite(times[:, 0]).any():
                self._leg_end(pk, times, pk.leg_crossed, None, None)
                # the beam leaves the scene: a pupil ray a surface stops or turns on the way leaves the beam there
                _t, nearest = self._first_blocking(times, pk.leg_crossed)
                diverged = pk.alive & (nearest >= 0)
                if self.recorder is not None and bool(diverged.any()):
                    rays = np.flatnonzero(diverged)
                    self.recorder.pupil_rays_diverged(pk, rays, tuple(surfaces[int(nearest[r])].surface_id for r in rays),
                                                      np.ones(rays.size, dtype=bool))
                pk.alive = pk.alive & ~diverged
                self._to_cursor(pk)
                self._segment(pk, None, None, None, None)
                if self.recorder is not None:
                    self.recorder.packet_advanced(pk, None)
                self._finish(pk, ESCAPED, None)
                return
            crossing, lead = crossing_group(self.index, int(np.argmin(times[:, 0])), times, self.cfg)
            if lead is None:
                if self._observation_event(pk, crossing, times):
                    return
                continue
            coincident = len(crossing) > 1
            chief_s = crossing[0]
            surf = surfaces[chief_s]
            hits = [self._hits(pk, i) for i in crossing]
            h = hits[0]
            # every ray advances to the physical surface; coincident planes sample it on the way
            crossing_times = hits[-1].t
            stays, nearest, _t = self._stays(pk, times, crossing, lead)
            # the straight leg ends at this surface: the observation planes part of the beam crosses on it, before that
            # surface, while the chief ray does not
            lead_hit = hits[-1]
            self._leg_end(pk, times, (*pk.leg_crossed, *crossing[:-1]), (lead_hit.point[0], lead_hit.normal[0], float(lead_hit.t[0])),
                          surfaces[lead].surface_id, lead)
            if not coincident and not pk.chief_obstructed and self._obstructs(pk, surf, h):
                if self._obstruct(pk, surf, h, times, chief_s):
                    return
                continue
            diverged = pk.alive & ~stays
            if self.recorder is not None and bool(diverged.any()):
                rays = np.flatnonzero(diverged)
                self.recorder.pupil_rays_diverged(pk, rays, tuple(
                    surfaces[int(nearest[r])].surface_id if nearest[r] >= 0 else None for r in rays),
                    np.isfinite(hits[-1].t[rays]) & hits[-1].inside[rays])
            pk.facet_diverged = self._facet_diverged(pk, diverged, nearest, surfaces[lead])
            if surfaces[lead].facet is None:
                parted = self._geometric_divergence(pk, surfaces[lead], diverged)
                if parted is not None:  # (every sample parted: the power the reference ray goes on with is not computed)
                    self._to_cursor(pk)
                    self._finish(pk, parted, None)
                    return
            met = self._met_beside_chief(pk, diverged, nearest, lead, hits[-1])
            self._tail_beside_chief(pk, lead, hits[-1], met)
            pk.alive = pk.alive & ~diverged
            # advance all live rays to the surface, each from where the leg began
            t = np.where(pk.alive, crossing_times, 0.0)
            extinction = self._extinction(pk)
            start, power_start, w_start = self._cursor(pk)
            power_leg = pk.chief_power()
            # (libm-free: the same bits in every runtime); a medium without absorption: e^-0 is exactly 1 for every ray
            atten = np.ones(t.shape) if extinction == 0.0 else exact_exp(-0.5 * extinction * t)
            pk.jones = pk.jones * atten[:, None]
            if extinction > 0.0:
                self._flow(pk, SINK_BULK_ABSORPTION, power_leg * -float(expm1(-extinction * float(t[0]))),
                           surfaces[chief_s], pk.medium)
            point = hits[-1].point
            pk.o = np.where(pk.alive[:, None], point, pk.o)
            pk.opl = pk.opl + self._ray_index(pk) * t
            pk.length = pk.length + t
            self._advance_group(pk, float(t[0]))
            pk.gstate = propagate(pk.gstate, float(t[0]))
            self._segment(pk, start, point[0] if coincident else h.point[0], float(t[0]) - pk.leg_t, surf, power_start=power_start,
                          w_start=w_start, w_end=_max_radius(pk))
            step_t = float(t[0]) - pk.leg_t
            pk.leg_t = 0.0
            crossing_alive = pk.alive.copy()
            first_arrival = len(self.out.arrivals)
            counted_here: list[int] = []  # (the planes of this crossing that counted an arrival: they read the power here)
            for ordinal, (si, hit) in enumerate(zip(crossing, hits)):
                observed = surfaces[si]
                if ordinal:
                    self._segment(pk, pk.o[0], pk.o[0], 0.0, observed,
                                  w_start=_max_radius(pk), w_end=_max_radius(pk))
                if coincident:
                    pk.alive = crossing_alive & np.isfinite(hit.t)
                physical = observed.interaction.kind != "observe"
                held = None
                if coincident and not physical:
                    # the plane samples each ray where it crosses the plane, not where it meets the (tilted or curved) surface
                    held = (pk.o, pk.opl, pk.length, pk.gpl, pk.gdd, pk.dispersive, pk.segment_mm)
                    on_plane = pk.alive & np.isfinite(hit.t)
                    # (the difference only where the ray is on the plane: a dead ray's times may both be inf, inf - inf)
                    dt = np.zeros(np.broadcast(hit.t, crossing_times).shape)
                    np.subtract(hit.t, crossing_times, out=dt, where=on_plane)
                    pk.o = np.where(on_plane[:, None], hit.point, pk.o)
                    pk.opl = pk.opl + self._ray_index(pk) * dt
                    pk.length = pk.length + dt
                    self._advance_group(pk, float(dt[0]))
                if self.recorder is not None:
                    self.recorder.packet_advanced(pk, observed.surface_id)
                self.out.trace_events.append({"path": pk.path_id, "surface": observed.surface_id,
                                              "t": step_t if ordinal == 0 else 0.0})
                # The allowance counts interactions, not free propagation. Record the arrival at
                # the next optic in both path segments and ray data, then stop before interacting.
                if physical and self.schedule.interaction_spent(pk.interactions):
                    self._interaction_limit(pk)
                    return
                if physical:
                    # a new straight leg starts at the surface (for the branches it makes too); its coincident planes lie at
                    # that start: crossed (``_stays``), and crossed again by a beam leaving toward their incident side
                    pk.leg_crossed, pk.leg_partial = tuple(counted_here), ()
                    if coincident:
                        pk.pending_observers = tuple((i, float(dot(pk.d[0], surfaces[i].normal)), lead) for i in crossing[:-1])
                    arriving_d, arriving_W, marked = pk.d[0].copy(), pk.chief_power(), self.out.arrivals[first_arrival:]
                elif coincident:
                    pk.coincidence = self._arriving(pk, observed, hit, surfaces[lead], lead_hit)
                terminated = self._interact(pk, observed, hit)
                pk.coincidence = None
                if physical and coincident:
                    self._settle_surface_event(pk, observed, lead_hit, arriving_d, arriving_W, terminated, marked)
                elif coincident and terminated is None and self.out.arrivals[-1].counted:
                    counted_here.append(si)
                if coincident and terminated is None and not physical:
                    pk.alive = crossing_alive.copy()
                if self.recorder is not None:
                    self.recorder.packet_interacted(pk)
                if held is not None and terminated is None:  # back to the surface the rays go on to
                    pk.o, pk.opl, pk.length, pk.gpl, pk.gdd, pk.dispersive, pk.segment_mm = held
                if terminated is not None:
                    self._finish(pk, terminated, observed.surface_id)
                    return
                if physical:  # observation planes are virtual: they never use the per-path interaction budget
                    pk.passes_owner = None
                    pk.interactions += 1

    def _settle_surface_event(self, pk: _Packet, surf: SurfaceG, h: Hits, arriving_d: Array, arriving_W: float, terminated: str | None,
                              marked: list[ArrivalRecord]) -> None:
        """What the chief ray did at the surface, for the arrivals of the planes through its hit point (they sampled the beam
        before it acted): a chief ray that left on the side it came from (a mirror, total internal reflection) was reflected,
        so was one a coated refracting face sent back more of than it passed; a mirror's back side or a stop's plate stopped it."""
        normal = h.normal[0]
        event = None
        if float(dot(arriving_d, normal)) * float(dot(pk.d[0], normal)) < 0.0:
            event = "REFLECTIVE"
        elif terminated in (BLOCKED_BACK, CLIPPED):
            event = "ABSORBING"
        elif surf.interaction.kind == "refract" and surf.interaction.coating is not None and pk.chief_power() < 0.5 * arriving_W:
            event = "REFLECTIVE"
        if event is None:
            return
        for rec in marked:
            if rec.coincidence is not None and rec.coincidence.event != "ABSORBING":
                rec.coincidence = replace(rec.coincidence, event=event)

    def _arriving(self, pk: _Packet, plane: SurfaceG, hit: Hits, surf: SurfaceG, surface_hit: Hits) -> SurfaceCoincidenceRecord:
        """What the plane through ``surf``'s chief-ray hit point records of the beam about to meet it: the power arriving at
        the surface, and the auxiliary bound of what crosses the literal plane before the surface (``literal_crossing_share``):
        for a flat surface tilted to the plane and a valid Gaussian beam of known power that does meet the surface."""
        inter = surf.interaction
        event = ("ABSORBING" if plane.interaction.absorbing or inter.kind == "absorber" else
                 "REFLECTIVE" if inter.kind in REFLECTIVE_KINDS or (inter.grating is not None and inter.grating.reflection) else "TRANSMISSIVE")
        share = None
        if (surf.geometry == "plane" and not plane.interaction.absorbing and pk.gstate.valid and not pk.chief_obstructed
                and pk.power_bound_W is None and bool(surface_hit.inside[0])):
            share = literal_crossing_share(pk.gstate, pk.d[0], plane.normal, surf.normal, float(surface_hit.t[0]) - float(hit.t[0]),
                                           float(surface_hit.t[0]), self._envelope_sigmas(pk), self.cfg)
        if share is None:
            return SurfaceCoincidenceRecord(surf.surface_id, "ARRIVING", event)
        cos_i = abs(float(dot(pk.d[0], surf.normal)))
        ap_u, ap_v, ap_cos = self._aperture_frame(surf, surface_hit, cos_i)
        beside = self._beyond_share(transmission_fraction(pk.gstate, surf.aperture, (float(surface_hit.u[0]), float(surface_hit.v[0])), ap_u,
                                                          ap_v, ap_cos, self.cfg, self.quad_rule))
        return SurfaceCoincidenceRecord(surf.surface_id, "ARRIVING", event, pk.chief_power() * min(1.0, share + beside))

    def _beside(self, pk: _Packet, times: Array, t_next: float) -> None:
        """Watched surfaces the chief ray misses on its way to ``t_next`` (its next surface) while the beam's footprint reaches
        their support: one ``SurfaceHit`` each with ``miss_distance_mm`` (``hits.py``). The support is taken in the surface's
        vertex plane (a sphere: the plane of its rim); the in-plane distance ``s`` from the chief ray's crossing to the
        support's edge gives the transverse distance ``s |cos(AOI)|``, a lower bound of the nearest support point's distance
        from the chief ray. The footprint reaches it when that is within ``aperture_quad_extent_sigmas`` of the larger 1/e^2
        radius there; without a valid Gaussian state, within the reach its live pupil rays imply (``_pupil_reach``; every
        crossed surface without pupil rays) -- then the distance is recorded as ``inf``: nothing bounds what arrives."""
        cfg = self.cfg
        surfaces = self.scene.surfaces
        o, d = pk.o[0], pk.d[0]
        for si in self._watch:
            surf = surfaces[si]
            if np.isfinite(times[si, 0]) or (pk.passes_owner is not None and surf.owner_id == pk.passes_owner):
                continue  # the chief ray meets it (or the beam passes beside this element by construction)
            if surf.geometry == "sphere":
                support_r = surf.aperture_radius + surf.mount_margin
                rim = math.sqrt(max(surf.radius * surf.radius - support_r * support_r, 0.0))
                normal, origin = surf.cap_axis, surf.center + rim * surf.cap_axis
                u_axis, v_axis = surf.u_axis, surf.v_axis
                support = None
            else:
                normal, origin, u_axis, v_axis = surf.normal, surf.origin, surf.u_axis, surf.v_axis
                support = support_of(surf)
                support_r = 0.0
            denom = float(dot(d, normal))
            if abs(denom) <= cfg.parallel_dot_tol:
                continue
            t = float(dot((origin - o), normal)) / denom
            if not (max(cfg.ray_min_advance_mm, pk.leg_t) < t <= t_next):
                continue  # the chief ray does not cross the surface's plane after its last event, up to its next one (the
                # steps' windows partition the leg: a plane crossed exactly at the next event is this step's)
            point = o + t * d
            rel = point - origin
            u, v = float(dot(rel, u_axis)), float(dot(rel, v_axis))
            if support is None:
                outside = math.sqrt(u * u + v * v) - support_r
            elif not apertures.bounded(support):
                continue
            else:
                outside = aperture_miss_distance(support, u, v)
            if outside <= 0.0:
                continue
            cos_aoi = min(1.0, abs(denom))
            distance = outside * cos_aoi
            gstate = pk.gstate
            if gstate.valid and not pk.chief_obstructed:
                gstate = propagate(gstate, t)
                if distance > cfg.aperture_quad_extent_sigmas * max(gstate.radius("x"), gstate.radius("y")):
                    continue
            else:  # no valid Gaussian state: the extent beyond the pupil rays is not known; the pupil rays' reach bounds it
                if distance > self._pupil_reach(pk, times[si], normal, origin, point, d):
                    continue
                distance = math.inf
            self.out.surface_hits.append(SurfaceHit(
                surface_id=surf.surface_id, owner_id=surf.owner_id, path_id=pk.path_id, source_id=pk.src.source_id,
                spectral_index=pk.spectral_index, wavelength_nm=pk.wavelength_nm, power_W=pk.chief_power(), gstate=gstate,
                position=point, direction=d.copy(), normal=np.array(normal, dtype=float), aoi_deg=float(incidence_deg(cos_aoi, d, normal)),
                cos_aoi=cos_aoi, chief_obstructed=pk.chief_obstructed, lineage=tuple(pk.lineage), power_scope=pk.scope_behind(),
                **self._hit_taken(pk), miss_distance_mm=distance, surface_flat=surf.geometry == "plane",
                **self._hit_field(pk, float(pk.opl[0]) + pk.n * t, step=t)))

    def _pupil_reach(self, pk: _Packet, surface_times: Array, normal: Array, origin: Array, point: Array, d: Array) -> float:
        """How far from the chief ray a beam without a valid Gaussian state may still carry light where it crosses a plane
        (``normal``, ``origin``; the chief crosses at ``point``): each live pupil ray launched ``rho`` source radii off the
        chief and found ``r`` mm off it there implies a local beam radius ``r / rho``; the reach is
        ``aperture_quad_extent_sigmas`` times the largest of them (the extent of a Gaussian of that size). ``inf`` when a
        live pupil ray meets the surface itself (it is reached) and without live pupil rays (a probe: nothing bounds it)."""
        rays = np.flatnonzero(pk.alive[pk.pupil_slice()]) + 1
        if rays.size == 0 or bool(np.isfinite(surface_times[rays]).any()):
            return math.inf
        dn = dot(pk.d[rays], normal)
        ok = np.abs(dn) > self.cfg.parallel_dot_tol
        t = np.where(ok, dot(origin - pk.o[rays], normal) / np.where(ok, dn, 1.0), 0.0)
        rel = pk.o[rays] + t[:, None] * pk.d[rays] - point
        trans = rel - dot(rel, d)[:, None] * d
        offsets = np.sqrt(np.sum(trans * trans, axis=1))
        launch = self._pupil_launch(pk, rays)
        rho = np.sqrt(np.sum(launch * launch, axis=1))
        usable = ok & (rho > 0.0)
        if not bool(usable.any()):
            return math.inf
        return self.cfg.aperture_quad_extent_sigmas * float(np.max(offsets[usable] / rho[usable]))

    def _annulus_diverges(self, pk: _Packet, times: Array) -> bool:
        """An obstructed packet's annulus must meet the surface its reference axis meets next (or leave the scene with it).
        When one of its live pupil rays would meet another surface first (a transparent observation plane aside, which
        never splits light), or none, the light splits up: the power that goes on is not computed (``_annulus_not_followed``),
        the rays that part are retired and the rest goes on. Returns True when the path ended (a probe)."""
        rays = np.flatnonzero(pk.alive & self._annulus_rays(pk))
        if rays.size == 0:  # no ray of the annulus (none sampled it, or no pupil rays): the bounds alone decide, at the surfaces
            return False
        surfaces = self.scene.surfaces
        chief_s = int(np.argmin(times[:, 0])) if np.isfinite(times[:, 0]).any() else None
        crossing, lead = crossing_group(self.index, chief_s, times, self.cfg) if chief_s is not None else ([], None)
        if lead is not None:
            stays, nearest, _t = self._stays(pk, times, crossing, lead)
            apart = ~stays[rays]
        else:  # (observation planes alone, or none: the first surface that stops or turns the rays must be the axis's; an
            # absorbing plane as a whole: where its active area takes part of the annulus, the plane decides, ``_annulus_on_area``)
            _t, nearest = self._first_blocking(times, pk.leg_crossed, whole_planes=True)
            apart = nearest[rays] != nearest[0]
        if not bool(apart.any()):
            return False
        names = sorted({surfaces[int(nearest[r])].surface_id if nearest[r] >= 0 else "(none)" for r in rays[apart]})
        ahead = "leaves the scene" if chief_s is None else f"meets {surfaces[chief_s].surface_id}"
        term = self._annulus_not_followed(
            pk, None, f"its reference axis {ahead} but {int(apart.sum())} of its {rays.size} pupil rays meet "
            f"{', '.join(names)}", pupil_rays_cut=int(apart.sum()), pupil_rays_live=int(rays.size))
        if term is not None:
            self._to_cursor(pk)
            self._finish(pk, term, None)
            return True
        # the rays that part leave the beam here; the rest goes on with its reference axis
        pk.alive = pk.alive.copy()
        pk.alive[rays[apart]] = False
        times[:, ~pk.alive] = np.inf
        self._set_blocking(pk, times)
        return False

    def _interaction_limit(self, pk: _Packet) -> None:
        """End a path that met ``max_interactions_per_path`` physical surfaces (the guard against endless cavity loops;
        observation planes do not count). Its remaining power is not followed (the ledger's ``INTERACTION_LIMIT``): the
        trace is incomplete there, so the issue is ``UNKNOWN`` and so is every result that light could still change."""
        self._issue(self.schedule.interaction_issue(pk.path_id, pk.src.source_id, pk.chief_power(), float(pk.length[0]), pk.power_bound_W))
        self._finish(pk, MAX_INTERACTIONS, None)

    def _segment(self, pk: _Packet, start: Array | None, end: Array | None, length: float | None, surf: SurfaceG | None,
                 *, power_start: float | None = None, w_start: float | None = None, w_end: float | None = None) -> None:
        if start is None:
            start = pk.o[0]
        p_start = pk.chief_power() if power_start is None else power_start
        known = pk.power_bound_W is None
        pk.segments.append(
            RaySegment(
                start_mm=tuple(float(v) for v in start), end_mm=None if end is None else tuple(float(v) for v in end),  # type: ignore[arg-type]
                length_mm=length, medium_id=pk.medium, refractive_index=pk.n,
                optical_path_length_mm=None if length is None else pk.n * length, power_start_W=p_start if known else None,
                power_end_W=pk.chief_power() if known else None, end_surface_id=None if surf is None else surf.surface_id,
                beam_radius_start_mm=w_start, beam_radius_end_mm=w_end,
                crystal_mode=None if pk.crystal is None else uniaxial.MODE_NAMES[pk.crystal.mode],  # type: ignore[arg-type]
            )
        )

    def _end_event(self, pk: _Packet, event: str, aoi_deg: float | None = None) -> None:
        if pk.segments:
            seg = pk.segments[-1]
            pk.segments[-1] = seg.model_copy(update={"end_event": event, "angle_of_incidence_deg": aoi_deg})

    # -------------------------------------------------------------------- interaction dispatch
    def _interact(self, pk: _Packet, surf: SurfaceG, h: Hits) -> str | None:
        """Apply the surface interaction. Returns a termination reason (the path ends) or ``None`` (continues)."""
        cfg = self.cfg
        inter = surf.interaction
        if pk.gstate.limit is not None and pk.gstate.limit.at is None:  # (a branch: the limit was exceeded where it began)
            pk.gstate = pk.gstate.located(pk.branch_surface_id or pk.src.source_id)
        plane_normal = self._chief_plane_normal(surf, h)
        chief_cos = abs(float(dot(pk.d[0], plane_normal)))

        p_in = pk.chief_power()
        if inter.kind != "observe":
            self._close_segment(pk, surf)
            if pk.power_bound_W is None:  # (the exposures read the ledger's unknown power for a branch without one)
                self._record_hit(pk, surf, h, p_in)
        # --- clear aperture --------------------------------------------------------------------------------
        if inter.kind != "observe" and pk.chief_obstructed:
            # the chief ray is a reference axis: the annulus must pass whole, else it is not followed
            term = self._annulus_through_aperture(pk, surf, h, chief_cos)
            if term is not None:
                return term
        elif inter.kind != "observe":
            pup_out = pk.alive & ~h.inside
            if not bool(h.inside[0]):
                if not self._in_central_obstruction(surf.aperture, h):
                    promoted = self._promote_reference(pk, surf, h, chief_cos, p_in)
                    if promoted is None:
                        self._report_clip(pk, surf, h, chief_cos)
                        self._clipped_branch(pk, surf, h, chief_cos, p_in)
                        self._end_event(pk, "clipped", float(incidence_deg(chief_cos, pk.d[0], plane_normal)))
                        pk.alive[:] = False
                        return CLIPPED
                    term, h = promoted
                    if term is not None:
                        return term
                    # (from here the promoted reference ray is row 0: its hit, its local normal)
                    if surf.geometry == "sag":
                        plane_normal = h.normal[0]
                    chief_cos = abs(float(dot(pk.d[0], plane_normal)))
                    p_in = pk.chief_power()
                else:
                    term = self._pass_obstruction(pk, surf, h, chief_cos, p_in)
                    if term is not None:
                        return term
            else:
                cut = pup_out & self._annulus_rays(pk)
                if surf.facet is not None and not pk.gstate.valid and pk.facet_diverged:
                    term = self._facet_edge_geometric(pk, surf, h, p_in)
                    if term is not None:
                        return term
                    p_in = pk.chief_power()
                if not pk.gstate.valid and not self.bounding and apertures.bounded(surf.aperture):
                    # no valid Gaussian (e.g. severely truncated upstream): the power this aperture passes is the share of
                    # the equal-power pupil samples that pass it, where they represent the beam (a probe goes on with its
                    # whole power: an upper bound of what passes)
                    term = self._geometric_aperture(pk, surf, h, cut, p_in)
                    if term is not None:
                        return term
                pk.alive = pk.alive & ~pup_out
            if apertures.bounded(surf.aperture) and pk.gstate.valid and not pk.chief_obstructed:
                ap_u, ap_v, ap_cos = self._aperture_frame(surf, h, chief_cos)
                frac = transmission_fraction(pk.gstate, surf.aperture, (float(h.u[0]), float(h.v[0])), ap_u, ap_v,
                                             ap_cos, cfg, self.quad_rule)
                if surf.facet is not None:  # (beyond a patch's edge lie the neighbouring facets, not the surround)
                    self._facet_edge(pk, surf, p_in, 1.0 - frac)
                    if self._spent(pk):
                        self._end_event(pk, "zero_power")
                        return ZERO_POWER
                    frac = 1.0
                else:
                    self._spill_beside(pk, surf, h, chief_cos, pk.gstate, p_in, frac)
                pk.gstate, code = (pk.gstate, None) if surf.facet is not None else mark_truncation(pk.gstate, 1.0 - frac, cfg)
                if code is not None:
                    status = Status.OUT_OF_MODEL_SCOPE if code is ErrorCode.GAUSSIAN_SEVERE_TRUNCATION else Status.WARNING
                    self._issue(make_issue(code, status, f"aperture truncates {1.0 - frac:.4g} of the beam power",
                                           location=f"surface:{surf.surface_id}", loss_fraction=1.0 - frac,
                                           beam_radius_x_mm=pk.gstate.radius("x") if pk.gstate.valid else None,
                                           suggested_action="enlarge the clear aperture or reduce the beam radius"))
                if frac < 1.0:
                    pk.jones = pk.jones.copy()
                    pk.jones[0] = pk.jones[0] * math.sqrt(max(frac, 0.0))
                    quantum = cfg.power_ledger_fraction_resolution
                    self._flow(pk, SINK_APERTURE_TRUNCATION, p_in * round((1.0 - frac) / quantum) * quantum, surf, "aperture")
                if self._spent(pk):
                    self._end_event(pk, "zero_power")
                    return ZERO_POWER
            if self._spent(pk):
                self._end_event(pk, "zero_power")
                return ZERO_POWER
            if apertures.bounded(surf.aperture) and not pk.chief_obstructed:
                self._note_opening(pk, surf.surface_id, surf.aperture, h)

        if pk.polarization_open and (self._weighs_polarization(inter, incidence_deg(chief_cos, pk.d[0], plane_normal))
                                     or self._crystal_face(surf)):
            self._open_power(pk, surf)
        if pk.crystal is not None and inter.kind not in ("refract", "observe"):
            return self._crystal_meets(pk, surf, inter)
        handler = {
            "refract": self._do_refract, "mirror": self._do_mirror, "thin_lens": self._do_thin_lens,
            "beam_splitter": self._do_splitter, "polarizer": self._do_polarizer, "retarder": self._do_retarder,
            "absorber": self._do_absorber, "aperture_stop": self._do_aperture_stop, "observe": self._do_observe,
            "neutral_density": self._do_neutral_density, "faraday_rotator": self._do_faraday_rotator,
            "diffraction_grating": self._do_diffraction_grating, "complex_transmission": self._do_complex_transmission,
        }[inter.kind]
        p_before = pk.chief_power()
        self._branch_power, self._split, self._split_detail = 0.0, {}, ""
        annulus_before = int((pk.alive & self._annulus_rays(pk)).sum()) if pk.chief_obstructed else 0
        if inter.scatter is not None:  # (a rough surface: its scattered share leaves first, optics/rays/scatter.py)
            self._scatter_off(pk, surf, h, inter)
        termination = handler(pk, surf, h, inter)
        pk.gstate = pk.gstate.located(surf.surface_id)  # (an accuracy limit first exceeded here)
        self._record_interaction_loss(pk, surf, inter.kind, p_before)
        if termination is None and pk.chief_obstructed and inter.kind != "observe":
            left = int((pk.alive & self._annulus_rays(pk)).sum())
            if left < annulus_before:  # the interaction retired rays of the annulus (other side, TIR, grazing, coating range)
                return self._annulus_not_followed(
                    pk, surf, f"{annulus_before - left} of its {annulus_before} pupil rays do not take the interaction its reference "
                    f"axis takes at {surf.surface_id}", pupil_rays_cut=annulus_before - left)
        return termination

    def _geometry_orders(self) -> bool:
        """Whether a grating's orders of unknown power are followed as geometry and its passes reported: for a traced beam. A
        probe's light of unknown fate may reach anything, and an extended emitter's rays have nothing to report per ray."""
        return not self.bounding

    def _spent(self, pk: _Packet) -> bool:
        """The path ends at the power floor: its power is known and at or below it (a branch whose power is not known is
        followed whatever it may carry)."""
        return pk.power_bound_W is None and pk.chief_power() <= self.cfg.power_epsilon_W

    @staticmethod
    def _chief_plane_normal(surf: SurfaceG, h: Hits) -> Array:
        """The normal the chief ray's angle of incidence is taken on: a plane's, the local one at the chief-ray hit of a sag
        surface, a sphere's cap axis."""
        if surf.geometry == "plane":
            return surf.normal
        if surf.geometry == "sag":
            return h.normal[0]
        return surf.cap_axis

    @staticmethod
    def _weighs_polarization(inter: InteractionG, chief_aoi_deg: float) -> bool:
        """Whether the share of the power this surface passes depends on the polarization (a polarizer, a splitter's coating,
        a grating's response, a coated reflector; an interface off normal incidence, ``chief_aoi_deg`` from
        ``incidence_deg``)."""
        if inter.kind in ("polarizer", "beam_splitter", "diffraction_grating"):
            return True
        if inter.kind == "mirror":
            return inter.coating is not None
        return inter.kind == "refract" and float(chief_aoi_deg) >= NORMAL_INCIDENCE_RECORD_DEG

    def _open_power(self, pk: _Packet, surf: SurfaceG) -> None:
        """A beam of known power and unknown polarization meets a surface whose share depends on it: its power goes to the
        ledger's ``UNMODELLED_RESPONSE`` and from here on is not known, at most what it is now."""
        power = pk.chief_power()
        key = (surf.surface_id, pk.wavelength_nm, "polarization")
        if key not in self._grating_reported:
            self._grating_reported.add(key)
            self._issue(make_issue(ErrorCode.GRATING_RESPONSE_UNAVAILABLE, Status.UNKNOWN,
                                   f"{pk.wavelength_nm!r} nm light from a grating order of mixed TE and TM meets {surf.surface_id}, whose share "
                                   "depends on the polarization: the response states powers, not the relative phase of the two "
                                   "components, so its power is not known from here on and its direction is",
                                   location=f"surface:{surf.surface_id}", path_id=pk.path_id, wavelength_nm=pk.wavelength_nm, incident_power_W=power, reason="POLARIZATION",
                                   suggested_action="send purely TE or purely TM light to the grating to keep the order's polarization known"))
        self._flow(pk, SINK_UNMODELLED_RESPONSE, power, surf, "grating order of unknown polarization")
        pk.jones = np.zeros_like(pk.jones)
        pk.power_bound_W, pk.polarization_open = power, False
        pk.bulk_loss_unknown = False  # Jones has been cleared; preserve this bound, not its zero envelope.

    def _pupil_launch(self, pk: _Packet, rays: NDArray[np.intp]) -> Array:
        """Launch offsets (units of the source radii) of the packet's pupil rays ``rays``."""
        unit = getattr(pk.src, "pupil_layout", None)  # an emitter's packet (sampled.py) has none
        if unit is None:
            layout = (pk.src.ring_count, pk.src.points_per_ring)
            if layout not in self._unit_offsets:
                self._unit_offsets[layout] = pupil_offsets(1.0, 1.0, *layout)
            unit = self._unit_offsets[layout]
        if not rays.size:
            return np.zeros((0, 2))
        samples = rays - 1 if pk.launch_rows is None else pk.launch_rows[rays - 1]
        return unit[samples]

    def _record_hit(self, pk: _Packet, surf: SurfaceG, h: Hits, p_in: float) -> None:
        """The beam arriving at a physical surface (``SurfaceHit``): power, Gaussian state, chief AOI on the local normal."""
        normal = h.normal[0]
        cos_aoi = min(1.0, abs(float(dot(pk.d[0], normal))))
        aoi = float(incidence_deg(cos_aoi, pk.d[0], normal))  # libm-free: the exposure results hold it (identical bits in every runtime)
        alive_pupil = np.zeros_like(pk.alive)
        alive_pupil[pk.pupil_slice()] = pk.alive[pk.pupil_slice()] & np.isfinite(h.t[pk.pupil_slice()])
        rays = np.flatnonzero(alive_pupil)
        self.out.surface_hits.append(SurfaceHit(
            surface_id=surf.surface_id, owner_id=surf.owner_id, path_id=pk.path_id, source_id=pk.src.source_id,
            spectral_index=pk.spectral_index, wavelength_nm=pk.wavelength_nm, power_W=p_in, gstate=pk.gstate,
            position=h.point[0].copy(), direction=pk.d[0].copy(), normal=np.array(normal, dtype=float), aoi_deg=aoi,
            cos_aoi=cos_aoi, chief_obstructed=pk.chief_obstructed, pupil_points=h.point[rays], pupil_directions=pk.d[rays],
            pupil_launch=self._pupil_launch(pk, rays), lineage=tuple(pk.lineage), power_scope=pk.scope_behind(),
            surface_flat=surf.geometry == "plane", **self._hit_taken(pk), **self._hit_field(pk, float(pk.opl[0]))))

    @staticmethod
    def _hit_taken(pk: _Packet) -> dict[str, Any]:
        """For a ``SurfaceHit``: the surfaces that took a bounded share of the beam before it and their bound, where no
        other surface puts its power out of model scope (the beam's power is then an upper value of the light's)."""
        if not pk.taken_path_at or pk.power_scope or pk.power_bound_W is not None:
            return {}
        taken_W = correctly_rounded_sum(entry[1] for entry in pk.taken_path)
        return {"power_taken_at": pk.taken_path_at, "power_taken_upper_W": min(taken_W, pk.chief_power())}

    def _hit_field(self, pk: _Packet, opl_mm: float, *, step: float = 0.0) -> dict[str, Any]:
        """The ``SurfaceHit`` field and coherence entries of the packet's chief ray at a surface (optical path ``opl_mm``;
        ``step``: the chief ray's distance beyond the packet's position there, in its medium)."""
        gpl, gdd, dispersive = self._group_after(pk, step)
        # (a hit beside a surface inside a dispersive medium: the spread of its unfinished segment is not bounded)
        spread = pk.spread if step == 0.0 or not self._in_dispersive(pk) else None
        return {"jones": pk.jones[0].copy(), "e1": pk.e1[0].copy(), "e2": pk.e2[0].copy(), "opl_mm": opl_mm,
                "opl_defined": not pk.opl_undefined, "coherence_group": pk.src.coherence_group,
                "coherence_length_mm": line_coherence_length_mm(pk.src, pk.spectral_index),
                "group_opl_mm": gpl, "gdd_mm2": gdd, "dispersive": dispersive, "envelope_spread": spread,
                "delays": pk.delays}

    @staticmethod
    def _ray_index(pk: _Packet) -> Array | float:
        """Each ray's optical path per mm of ray in the packet's medium: its index (one value), or in a crystal each ray's
        phase index along its ray (``_CrystalLeg``)."""
        return pk.n if pk.crystal is None else pk.crystal.ray_index

    def _medium_group(self, pk: _Packet) -> tuple[float, float] | None:
        """``(n_g, k2)`` along the chief ray in the packet's medium (``None``: not available): the medium's, or in a crystal its
        wave's (``optics/polarization/uniaxial.py``)."""
        if pk.crystal is not None:
            return pk.crystal.group
        return self.library.group(pk.medium, pk.wavelength_nm)

    def _in_dispersive(self, pk: _Packet) -> bool:
        """Whether the packet's medium disperses (``n_g != n`` or ``k2 != 0`` at its wavelength; not available: yes)."""
        group = self._medium_group(pk)
        return group is None or group[0] != pk.n or group[1] != 0.0

    def _close_segment(self, pk: _Packet, surf: SurfaceG) -> None:
        """The packet meets a physical surface: the segment it travelled since the last one ends. Through a dispersive
        medium between two parallel planes it adds its ``plate_spread`` to the envelope delay's spread (otherwise the
        spread is no longer bounded); the next segment begins at this surface."""
        normal = (float(surf.normal[0]), float(surf.normal[1]), float(surf.normal[2])) if surf.geometry == "plane" else None
        if pk.spread is not None and pk.segment_mm != 0.0 and self._in_dispersive(pk):
            start = pk.segment_normal
            group = self._medium_group(pk)
            added: Spread | None = None
            # (an extraordinary wave's group index changes with its direction: the plate bound does not hold for it)
            extraordinary = pk.crystal is not None and pk.crystal.mode == uniaxial.EXTRAORDINARY
            if (not extraordinary and start is not None and normal is not None and group is not None and pk.gstate.valid and not pk.chief_obstructed
                    and abs(start[0] * normal[0] + start[1] * normal[1] + start[2] * normal[2]) >= 1.0 - PARALLEL_PLATE_TOL):
                d = pk.d[0]
                cos_chief = min(1.0, abs(float(d[0]) * normal[0] + float(d[1]) * normal[1] + float(d[2]) * normal[2]))
                slope = 0.0
                for q, axis in ((pk.gstate.qx, "x"), (pk.gstate.qy, "y")):
                    inverse = 1.0 / q
                    slope = max(slope, pk.gstate.radius(axis) * math.sqrt(inverse.real * inverse.real + inverse.imag * inverse.imag))
                ng, k2 = group
                added = plate_spread(abs(ng - pk.n), abs(k2), abs(pk.segment_mm), cos_chief, slope)
            pk.spread = add_spread(pk.spread, added)
        pk.segment_normal = normal
        pk.segment_surface = None if normal is not None else self._surface_index.get(surf.surface_id)
        pk.segment_start = surf.surface_id
        pk.segment_mm = 0.0

    def _group_after(self, pk: _Packet, step: float) -> tuple[float | None, float, bool]:
        """The chief ray's (group optical path, group-delay dispersion, crossed a dispersive medium) after ``step`` mm more in
        its medium (``_Packet.gpl``)."""
        if step == 0.0:
            return pk.gpl, pk.gdd, pk.dispersive
        group = self._medium_group(pk)
        if group is None or pk.gpl is None:
            return None, pk.gdd, True
        ng, k2 = group
        return pk.gpl + ng * step, pk.gdd + k2 * step, pk.dispersive or ng != pk.n or k2 != 0.0

    def _advance_group(self, pk: _Packet, step: float) -> None:
        """Advance the chief ray's group optical path by ``step`` mm in its medium (``_Packet.gpl``)."""
        pk.gpl, pk.gdd, pk.dispersive = self._group_after(pk, step)
        pk.segment_mm = pk.segment_mm + step

    def _aperture_frame(self, surf: SurfaceG, h: Hits, chief_cos: float) -> tuple[Array, Array, float]:
        """In-plane axes and projection factor of the beam footprint on the aperture coordinates (``transmission_fraction``).

        Planes and spheres: the surface's ``(u, v)`` axes and the chief-ray ``cos``. A sag surface measures its clear
        aperture on the transverse coordinates ``(a, b)``: locally the beam meets the tangent plane at the hit (normal
        ``N``), whose points map to ``(a, b)`` along the axis ``n``; the axes are ``u`` and ``v`` lifted onto that plane
        along ``n`` (``u - (u.N)/(n.N) n``) and the area factor is ``|d.N| / (n.N)``."""
        if surf.geometry != "sag":
            return surf.u_axis, surf.v_axis, chief_cos
        nrm = h.normal[0]
        along = max(float(dot(nrm, surf.normal)), self.cfg.zero_vector_tol)
        u = surf.u_axis - (float(dot(surf.u_axis, nrm)) / along) * surf.normal
        v = surf.v_axis - (float(dot(surf.v_axis, nrm)) / along) * surf.normal
        return u, v, chief_cos / along

    @staticmethod
    def _annulus_rays(pk: _Packet) -> NDArray[np.bool_]:
        """The packet's pupil samples (rays ``1 .. pupil_total``): the light of an obstructed packet. The chief ray and the
        differential rays are reference rays."""
        mask = np.zeros(pk.o.shape[0], dtype=bool)
        mask[pk.pupil_slice()] = True
        return mask

    def _in_central_obstruction(self, ap: Any, h: Hits) -> bool:
        """Is the chief-ray hit (outside the clear aperture) on the aperture's central obstruction, i.e. within its outer
        radius? (Otherwise it is on the opaque rim beyond the opening.)"""
        if ap.shape != "circle" or ap.obstruction <= 0.0:
            return False
        return math.hypot(float(h.u[0]) - ap.offset_u, float(h.v[0]) - ap.offset_v) <= ap.p1 + self.cfg.aperture_edge_tol_mm

    def _pass_obstruction(self, pk: _Packet, surf: SurfaceG, h: Hits, chief_cos: float, p_in: float) -> str | None:
        """The chief ray lands on the central obstruction of a circular clear aperture: the light passes the
        annulus between the obstruction and the opening.

        The incident beam is Gaussian, so the annulus's power is exact: ``transmission_fraction`` of the annulus. The part
        on the obstruction is booked ``CLIPPED`` and the part beyond the opening ``APERTURE_TRUNCATION``; the packet goes on
        with the annulus's power. The pupil rays inside the annulus go on as its geometric rays; the chief ray (and the
        differential rays beside it) go on as an unpowered reference axis that keeps the surface sequence. The transmitted
        beam is not a Gaussian beam (no Gaussian metric is computed downstream: ``GAUSSIAN_CENTRAL_OBSTRUCTION``), and
        diffraction by the obstruction is not modelled. Without a valid Gaussian state the annulus's power is not known:
        the path ends ``ANNULAR_BEAM_NOT_FOLLOWED``.

        Behind the obstruction the packet's Gaussian state is the *unobstructed* beam's, carried on as the envelope that
        bounds where the annulus can be (the annulus is part of that beam), and the opening's radius is kept in units of the
        envelope's radii (``opening``): the annulus is carried through a later aperture or onto an active area only
        while it provably passes whole (``_annulus_through_aperture``, ``_annulus_on_area``). Its pupil rays, where any are
        left, must also take the surfaces and interactions its axis takes; with none (an obstruction wider than the sampled
        rings, or a trace without pupil rays: control readings, probes) the bounds alone decide."""
        cfg, ap = self.cfg, surf.aperture
        if not pk.gstate.valid:
            term = self._annulus_not_followed(
                pk, surf, f"the chief ray falls on the central obstruction of {surf.surface_id} and the beam has no valid Gaussian state")
            if term is not None:
                return term
            # the annulus goes on as geometry: its pupil rays inside the opening, the chief ray as a reference axis; no
            # envelope bounds it (no opening is kept), so every later aperture and active area finds its power not known
            annulus = self._annulus_rays(pk)
            pk.alive = pk.alive & (h.inside | ~annulus)
            pk.chief_obstructed = True
            pk.opening = None
            return None
        ap_u, ap_v, ap_cos = self._aperture_frame(surf, h, chief_cos)
        centre = (float(h.u[0]), float(h.v[0]))
        passed = transmission_fraction(pk.gstate, ap, centre, ap_u, ap_v, ap_cos, cfg, self.quad_rule)
        opening = transmission_fraction(pk.gstate, replace(ap, obstruction=0.0), centre, ap_u, ap_v, ap_cos, cfg, self.quad_rule)
        quantum = cfg.power_ledger_fraction_resolution
        blocked = round(max(opening - passed, 0.0) / quantum) * quantum
        beyond = round(max(1.0 - opening, 0.0) / quantum) * quantum
        if blocked > 0.0:
            self._flow(pk, SINK_CLIPPED, p_in * blocked, surf, "central obstruction")
        if beyond > 0.0:
            self._flow(pk, SINK_APERTURE_TRUNCATION, p_in * beyond, surf, "aperture")
        self._spill_beside(pk, surf, h, chief_cos, pk.gstate, p_in, opening)
        pk.jones = pk.jones.copy()
        pk.jones[0] = pk.jones[0] * math.sqrt(max(passed, 0.0))
        annulus = self._annulus_rays(pk)
        pk.alive = pk.alive & (h.inside | ~annulus)  # pupil rays on the obstruction or beyond the opening are stopped
        if self.bounding:
            pk.alive = pk.alive & ~annulus  # (a probe: the bounds of its annulus alone decide)
        pk.chief_obstructed = True
        pk.annulus_fraction = passed
        # the opening that cut the annulus, seen from the chief ray on the Gaussian axes, and the annulus rays' offsets there
        # (``_edge_inside`` maps it paraxially to later apertures)
        bx, by, d0 = pk.gstate.bx, pk.gstate.by, pk.d[0]
        centre_3d = (ap.offset_u - centre[0]) * ap_u + (ap.offset_v - centre[1]) * ap_v
        rel = pk.o - pk.o[0]
        rel = rel - np.outer(dot(rel, d0), d0)  # transverse offsets of the rays from the chief ray
        offsets = np.full((pk.o.shape[0], 2), np.nan)
        rays = np.flatnonzero(pk.alive & annulus)
        offsets[rays] = np.column_stack([dot(rel[rays], bx), dot(rel[rays], by)])
        pk.opening = _Opening(ap.p1, (float(dot(centre_3d, bx)), float(dot(centre_3d, by))),
                              (pk.gstate.radius("x"), pk.gstate.radius("y")), offsets)
        text = (f"the chief ray falls on the central obstruction (radius {ap.obstruction:.4g} mm) of the clear aperture: the "
                "light passing it is an annulus, not a Gaussian beam")
        self._gaussian_reported.add((pk.path_id, ErrorCode.GAUSSIAN_CENTRAL_OBSTRUCTION.value))
        continuing = int((pk.alive & annulus).sum())
        self._issue(make_issue(
            ErrorCode.GAUSSIAN_CENTRAL_OBSTRUCTION, Status.OUT_OF_MODEL_SCOPE,
            f"{text}; the annulus transmits {passed:.4g} of the beam power ({p_in * passed:.4g} W), the obstruction blocks "
            f"{max(opening - passed, 0.0):.4g}; behind this surface the power and the geometric rays of the annulus are traced "
            "while it provably passes the apertures whole, the Gaussian beam metrics are not available "
            "and diffraction by the obstruction is not modelled",
            location=f"surface:{surf.surface_id}", path_id=pk.path_id, obstruction_radius_mm=ap.obstruction, aperture_radius_mm=ap.p1,
            chief_offset_mm=math.hypot(centre[0] - ap.offset_u, centre[1] - ap.offset_v), transmitted_fraction=passed,
            obstructed_fraction=max(opening - passed, 0.0), outside_aperture_fraction=max(1.0 - opening, 0.0),
            transmitted_power_W=p_in * passed, pupil_rays_continuing=continuing, pupil_ray_count=pk.pupil_total,
            suggested_action="read power_W and the ray data (spot size, per-ray paths) behind the obstruction; Gaussian beam "
                             "metrics are withheld there"))
        return None

    # ------------------------------------------------------------------ obscured pupils: an element inside the beam
    def _obstructs(self, pk: _Packet, surf: SurfaceG, h: Hits) -> bool:
        """Does the chief ray meet an opaque element whose whole support lies inside the beam, a non-negligible part of the
        beam passing beside it (``_blocker``: the back of a mirror -- the secondary mirror of a Cassegrain telescope, met by the
        incoming beam --, an absorber, or the opaque plate of an aperture stop beside its opening)? Only for a traced beam with
        pupil rays and a valid Gaussian state in the element's surround."""
        if (self.bounding or not self.include_pupil or pk.pupil_total == 0 or not pk.gstate.valid or pk.crystal is not None
                or pk.power_bound_W is not None):
            return False
        if not bool(np.isfinite(h.t[0])) or pk.medium in self._interior.get(surf.owner_id, ()):
            return False
        if self._blocker(pk, surf, h) is None:
            return False
        support = support_of(surf)
        if not apertures.bounded(support):
            return False
        beside = 1.0 - self._on_support(pk, surf, h)
        return not self._negligible(pk, beside * pk.chief_power())

    def _blocker(self, pk: _Packet, surf: SurfaceG, h: Hits) -> str | None:
        """How the element the chief ray meets stops the light on its support, when it stops all of it there: ``"back"`` -- the
        back of a mirror that does not transmit there --, ``"absorber"`` -- an absorber that does not scatter --, ``"stop"`` --
        the opaque plate of an aperture stop, the chief ray beside its opening (not on a central obstruction) and the beam's
        share through the opening negligible --; else ``None`` (the element keeps its own rule; an absorbing facet patch too)."""
        inter = surf.interaction
        if inter.kind == "mirror":
            if inter.reflective_side == "both":
                return None
            from_pos, _same = self._side(pk, surf, h)
            if (inter.reflective_side == "positive") == from_pos or self._back_transmittance(pk, surf, h, inter) is not None:
                return None
            return "back"
        if inter.kind == "absorber":  # (a facet patch of a solid keeps the facet rules: its neighbours lie beside it)
            return "absorber" if inter.scatter is None and surf.facet is None else None
        if inter.kind == "aperture_stop" and surf.geometry == "plane" and not bool(h.inside[0]):
            if self._in_central_obstruction(surf.aperture, h) or not apertures.bounded(surf.aperture):
                return None
            g = propagate(pk.gstate, float(h.t[0]))
            chief_cos = abs(float(dot(pk.d[0], surf.normal)))
            ap_u, ap_v, ap_cos = self._aperture_frame(surf, h, chief_cos)
            opening = transmission_fraction(g, surf.aperture, (float(h.u[0]), float(h.v[0])), ap_u, ap_v, ap_cos, self.cfg, self.quad_rule)
            return "stop" if self._negligible(pk, opening * pk.chief_power()) else None
        return None

    def _met_beside_obstruction(self, pk: _Packet, surf: SurfaceG, met: NDArray[np.bool_], nearest: NDArray[np.intp]) -> None:
        """Pupil rays of a beam whose chief ray meets an obstructing element (``_obstruct``) that meet another physical surface
        first, beside the element's support (``met``): they leave the beam there, but the annulus's power is the Gaussian share
        beside the support, which still counts their light. So, as in ``_met_beside_chief``, each such surface joins
        ``power_scope`` (the annulus's power behind it is out of model scope, never a known value) and an ``OUT_OF_MODEL_SCOPE``
        issue states the share of the pupil samples it takes. Observation planes keep their own rule (``_leg_end``)."""
        if self.bounding or pk.power_bound_W is not None or not bool(met.any()):
            return
        surfaces = self.scene.surfaces
        counts: dict[int, int] = {}
        for ray in np.flatnonzero(met):
            si = int(nearest[ray])
            if si < 0 or surfaces[si].interaction.kind == "observe":
                continue
            counts[si] = counts.get(si, 0) + 1
        n_live = int((pk.alive & self._annulus_rays(pk)).sum())
        for si in sorted(counts):
            surface_id = surfaces[si].surface_id
            if surface_id in pk.power_scope:
                continue
            count = counts[si]
            pk.power_scope = (*pk.power_scope, surface_id)
            self._issue(make_issue(
                ErrorCode.OUT_OF_MODEL_SCOPE, Status.OUT_OF_MODEL_SCOPE,
                f"{count} of the {n_live} pupil rays of beam {pk.path_id} (about {count / n_live:.3g} of its power by its "
                f"equal-power samples) meet {surface_id} beside the support of {surf.surface_id}, which its chief ray meets: the "
                f"annulus passing beside {surf.surface_id} carries the Gaussian share beside that support, which still counts that "
                f"light, so the power it carries behind {surf.surface_id} is not computed",
                location=f"surface:{surface_id}", path_id=pk.path_id, surface_ids=[surface_id], next_surface_id=surf.surface_id,
                pupil_rays_met=count, pupil_rays_live=n_live, met_fraction=count / n_live,
                suggested_action="move the element or the beam so the beam passes the element whole or clears it, or read "
                                 "the power at a plane before that surface"))

    def _on_support(self, pk: _Packet, surf: SurfaceG, h: Hits) -> float:
        """The share of the packet's Gaussian beam, propagated to the chief ray's hit, that falls on the surface's support."""
        g = propagate(pk.gstate, float(h.t[0]))
        normal = surf.normal if surf.geometry == "plane" else h.normal[0]
        chief_cos = abs(float(dot(pk.d[0], normal)))
        ap_u, ap_v, ap_cos = self._aperture_frame(surf, h, chief_cos)
        support = support_of(surf)
        return transmission_fraction(g, replace(support, obstruction=0.0), (float(h.u[0]), float(h.v[0])), ap_u, ap_v, ap_cos,
                                     self.cfg, self.quad_rule)

    def _obstruct(self, pk: _Packet, surf: SurfaceG, h: Hits, times: Array, si: int) -> bool:
        """The chief ray meets an opaque element inside the beam (``_obstructs``): the element is a central obstruction. The
        share of the Gaussian beam on its support is stopped there (a mirror's back: ``BLOCKED_BACK_SIDE``; an absorber:
        ``ABSORBED``; a stop's plate: ``CLIPPED``); the rest passes beside it as an
        annulus with that exact power, its pupil rays outside the support going on as its geometric rays, the chief ray as an
        unpowered reference axis (``chief_obstructed``) through the element, the Gaussian state as the envelope of the
        unobstructed beam. Every live ray is carried to the plane across the chief ray at its hit; a pupil ray that meets the
        element, or another surface first, leaves the beam. The element's support projected onto that plane is the annulus's
        shadow (``_Shadow``), against which later holes are checked. Gaussian metrics are withheld behind it
        (``GAUSSIAN_CENTRAL_OBSTRUCTION``) and diffraction by the element is not modelled. Returns whether the path ended."""
        cfg = self.cfg
        p_in = pk.chief_power()
        blocker = self._blocker(pk, surf, h)
        on_support = self._on_support(pk, surf, h)
        t0 = float(h.t[0])
        d0, p_hit = pk.d[0].copy(), h.point[0].copy()
        along = dot(pk.d, d0)
        t_plane = np.where(along > 0.0, dot(p_hit - pk.o, d0) / np.where(along > 0.0, along, 1.0), np.inf)
        meets = np.isfinite(h.t)  # (a pupil ray on the element's support meets it, before or beyond the plane across the chief ray)
        _t, nearest = self._first_blocking(times, (si,))
        first = np.array([nearest[r] >= 0 and times[nearest[r], r] < t_plane[r] for r in range(times.shape[1])], dtype=bool)
        self._met_beside_obstruction(pk, surf, pk.alive & first & ~meets & self._annulus_rays(pk), nearest)
        self._tail_beside_chief(pk, si, h)  # (and those the beam's tail may meet between and beyond the pupil samples)
        annulus = pk.alive & ~meets & ~first & np.isfinite(t_plane)
        annulus[0] = True
        retired = pk.alive & ~annulus
        if self.recorder is not None and bool(retired[1:].any()):
            rays = np.flatnonzero(retired)
            self.recorder.pupil_rays_diverged(pk, rays, tuple(surf.surface_id if meets[r] else None for r in rays), meets[rays])
        pk.alive = annulus
        if int(pk.alive[pk.pupil_slice()].sum()) == 0:
            return self._finish_obstructed_without_rays(pk, surf)
        t = np.where(pk.alive, t_plane, 0.0)
        t[0] = t0
        start, power_start, w_start = self._cursor(pk)
        extinction = self._extinction(pk)
        if extinction > 0.0:
            pk.jones = pk.jones * exact_exp(-0.5 * extinction * t)[:, None]
            self._flow(pk, SINK_BULK_ABSORPTION, p_in * -float(expm1(-extinction * t0)), surf, pk.medium)
            p_in = pk.chief_power()
        pk.o = np.where(pk.alive[:, None], pk.o + t[:, None] * pk.d, pk.o)
        pk.opl = pk.opl + self._ray_index(pk) * t
        pk.length = pk.length + t
        self._advance_group(pk, t0)
        pk.gstate = propagate(pk.gstate, t0)
        self._segment(pk, start, p_hit, t0 - pk.leg_t, surf, power_start=power_start, w_start=w_start, w_end=_max_radius(pk))
        pk.leg_t, pk.leg_crossed, pk.leg_partial = 0.0, (), ()
        if self.recorder is not None:
            self.recorder.packet_advanced(pk, surf.surface_id)
        self.out.trace_events.append({"path": pk.path_id, "surface": surf.surface_id, "t": t0})
        if self.schedule.interaction_spent(pk.interactions):
            self._interaction_limit(pk)
            return True
        self._close_segment(pk, surf)
        if pk.power_bound_W is None:
            self._record_hit(pk, surf, h, p_in)
        quantum = cfg.power_ledger_fraction_resolution
        blocked = round(on_support / quantum) * quantum
        sink = {"absorber": SINK_ABSORBED, "stop": SINK_CLIPPED}.get(blocker or "back", SINK_BLOCKED_BACK_SIDE)
        if blocked > 0.0:
            self._flow(pk, sink, p_in * blocked, surf, "obstruction")
        pk.jones = pk.jones.copy()
        pk.jones[0] = pk.jones[0] * math.sqrt(max(1.0 - on_support, 0.0))
        pk.chief_obstructed = True
        pk.annulus_fraction = max(1.0 - on_support, 0.0)
        pk.opening = None
        pk.shadow = self._shadow_of(pk, surf, h, p_hit, d0)
        if self._off_axis_curved(surf, d0) and surf.surface_id not in pk.power_scope:
            # a curved support met at an angle to its axis: the share above maps the support onto the plane at the chief ray's
            # hit, not the shadow of the sagged surface along the beam (exact along the axis only), so it is neither the share
            # stopped nor a bound of it -- the annulus's power behind the element is out of model scope
            pk.power_scope = (*pk.power_scope, surf.surface_id)
            self._issue(make_issue(
                ErrorCode.OUT_OF_MODEL_SCOPE, Status.OUT_OF_MODEL_SCOPE,
                f"the chief ray of beam {pk.path_id} meets the curved support of {surf.surface_id} at an angle to its axis: the "
                f"share of the beam it stops ({on_support:.4g} on the support mapped onto the plane at the chief ray's hit) is not "
                f"the share its shadow along the beam stops, which lies up to {pk.shadow.margin:.4g} mm (its depth over the support "
                "times the tangent of that angle) from that outline, so the power passing beside it is not computed",
                location=f"surface:{surf.surface_id}", path_id=pk.path_id, surface_ids=[surf.surface_id],
                obstructed_fraction=on_support, shadow_margin_mm=pk.shadow.margin,
                suggested_action="meet the element along its axis, or read the power at a plane before it"))
        self._gaussian_reported.add((pk.path_id, ErrorCode.GAUSSIAN_CENTRAL_OBSTRUCTION.value))
        continuing = int(pk.alive[pk.pupil_slice()].sum())
        met = {"absorber": f"the absorber {surf.surface_id}", "stop": f"the opaque plate of the stop {surf.surface_id} beside its opening"
               }.get(blocker or "back", f"the back of {surf.surface_id}")
        self._issue(make_issue(
            ErrorCode.GAUSSIAN_CENTRAL_OBSTRUCTION, Status.OUT_OF_MODEL_SCOPE,
            f"the chief ray meets {met}, whose support lies inside the beam: the element blocks "
            f"{on_support:.4g} of the beam power and the light passing beside it is an annulus ({p_in * (1.0 - on_support):.4g} W), "
            "not a Gaussian beam; behind it the power and the geometric rays of the annulus are traced while it provably passes the "
            "apertures and holes whole, the Gaussian beam metrics are not available and diffraction by the element is not modelled",
            location=f"surface:{surf.surface_id}", path_id=pk.path_id, transmitted_fraction=1.0 - on_support, obstructed_fraction=on_support,
            transmitted_power_W=p_in * (1.0 - on_support), pupil_rays_continuing=continuing, pupil_ray_count=pk.pupil_total,
            suggested_action="read power_W and the ray data (spot size, per-ray paths) behind the obstruction; Gaussian beam "
                             "metrics are withheld there"))
        self._end_event(pk, "obstruction")
        pk.lineage.append(f"{surf.surface_id}:obstruction")
        if self.recorder is not None:
            self.recorder.packet_interacted(pk)
        pk.interactions += 1
        pk.passes_owner = None
        return False

    @staticmethod
    def _off_axis_curved(surf: SurfaceG, d0: Array) -> bool:
        """A curved support (a sag surface, a sphere) met by the chief ray ``d0`` more than ``CURVED_SUPPORT_AXIS_SIN_TOL`` (sine)
        from its axis: ``_on_support`` maps the support onto the plane at the chief ray's hit, which is the shadow of the
        sagged surface along the beam only along that axis."""
        if surf.geometry == "plane":
            return False
        axis = surf.normal if surf.geometry == "sag" else surf.cap_axis
        along = float(dot(d0, axis))
        return math.sqrt(max(0.0, 1.0 - along * along)) > CURVED_SUPPORT_AXIS_SIN_TOL

    def _finish_obstructed_without_rays(self, pk: _Packet, surf: SurfaceG) -> bool:
        """No pupil ray passes beside the element: the annulus is not resolved by the samples, its path ends."""
        self._end_event(pk, "blocked_back_side")
        pk.alive[:] = False
        self._finish(pk, ANNULUS_NOT_FOLLOWED, surf.surface_id)
        return True

    def _shadow_of(self, pk: _Packet, surf: SurfaceG, h: Hits, p_hit: Array, d0: Array) -> _Shadow:
        """The element's shadow on the plane across the chief ray at its hit (``_Shadow``): its support outline in the surface's
        (u, v), projected along the chief ray onto the plane; the margin is a curved element's largest sag over its support
        times the tangent of the chief ray's angle to its axis."""
        support = support_of(surf)
        ring = np.array(apertures.ring(support, SHADOW_RING_SEGMENTS), dtype=np.float64)
        normal = surf.normal if surf.geometry in ("plane", "sag") else surf.cap_axis
        base = surf.origin if surf.geometry in ("plane", "sag") else surf.center + surf.radius * surf.cap_axis
        points = base + np.outer(ring[:, 0], surf.u_axis) + np.outer(ring[:, 1], surf.v_axis)
        along = float(dot(d0, normal))
        points = points - np.outer(dot(points - p_hit, d0), d0)  # (onto the plane across the chief ray, along it)
        ax, ay = pk.gstate.bx, pk.gstate.by
        rel = points - p_hit
        plane_ring = np.column_stack([dot(rel, ax), dot(rel, ay)])
        if float(np.sum(plane_ring[:-1, 0] * plane_ring[1:, 1] - plane_ring[1:, 0] * plane_ring[:-1, 1])) < 0.0:
            plane_ring = plane_ring[::-1]
        margin = 0.0
        if surf.geometry != "plane":
            reach = apertures.reach(support)
            low, high = _sag_range(surf, reach) if surf.geometry == "sag" else (0.0, surf.radius)
            tilt = math.sqrt(max(0.0, 1.0 - along * along)) / max(abs(along), self.cfg.parallel_dot_tol)
            margin = max(abs(low), abs(high)) * tilt
        rays = np.full((pk.o.shape[0], 2), np.nan)
        live = np.flatnonzero(pk.alive)
        rel_rays = pk.o[live] - p_hit
        rays[live] = np.column_stack([dot(rel_rays, ax), dot(rel_rays, ay)])
        return _Shadow(surf.surface_id, p_hit.copy(), ax.copy(), ay.copy(), plane_ring, margin, rays, bool(support.holes))

    def _holes_in_shadow(self, pk: _Packet, surf: SurfaceG, h: Hits) -> bool:
        """Do the surface's holes lie inside the shadow of the element the annulus passed (``_Shadow``), so that no light of
        the annulus reaches them? The annulus rays' least-squares affine map from the shadow's plane to the surface's (u, v)
        (``_affine_map``) carries the shadow's outline onto the surface; every point of every hole's outline must lie inside
        the mapped outline by more than the map's largest residual plus the shadow's margin (scaled by the map). Not where the
        element itself has holes (``_Shadow.holed``): light through them goes on inside its outline and may reach a later hole."""
        shadow = pk.shadow
        if shadow is None or shadow.holed:
            return False
        rows = np.flatnonzero(pk.alive & self._annulus_rays(pk) & np.isfinite(shadow.ray_points[:, 0]) & np.isfinite(h.u))
        if rows.size < SHADOW_FIT_MIN_RAYS:
            return False
        fit = _affine_map(shadow.ray_points[rows, 0], shadow.ray_points[rows, 1], h.u[rows], h.v[rows])
        if fit is None:
            return False
        (m00, m01, m10, m11), (bu, bv), residual, _centre, _far = fit
        ring = shadow.ring
        mapped = np.column_stack([m00 * ring[:, 0] + m01 * ring[:, 1] + bu, m10 * ring[:, 0] + m11 * ring[:, 1] + bv])
        if m00 * m11 - m01 * m10 < 0.0:
            mapped = mapped[::-1]
        scale = math.sqrt(max(m00 * m00 + m10 * m10, m01 * m01 + m11 * m11))
        margin = residual + scale * shadow.margin + self.cfg.aperture_edge_tol_mm
        for hole in surf.aperture.holes:
            points = np.array(apertures.ring(hole, SHADOW_RING_SEGMENTS), dtype=np.float64)
            if not _inside_by(points, mapped, margin):
                return False
        return True

    def _filled_chief(self, pk: _Packet, times: Array) -> None:
        """An annulus's reference axis crossing a hole of a surface the annulus meets around it (a Cassegrain primary's
        central hole, behind the secondary's shadow): its time to that surface is taken on the surface without its holes, so
        it keeps the sequence the annulus takes; whether the hole passes any of the annulus is decided there
        (``_holes_in_shadow``)."""
        rays = np.flatnonzero(pk.alive & self._annulus_rays(pk))
        if rays.size == 0:
            return
        first = np.argmin(times[:, rays], axis=0)
        chief_t = float(np.min(times[:, 0]))
        for si, filled in self._filled.items():
            if np.isfinite(times[si, 0]) or not bool(np.any(first == si)):
                continue
            t = float(intersect(filled, pk.o[:1], pk.d[:1], self.cfg).t[0])
            if np.isfinite(t) and t < chief_t:
                times[si, 0] = t
                chief_t = t

    def _hits(self, pk: _Packet, si: int) -> Hits:
        """``intersect`` of the packet's rays with surface ``si``; an obstructed packet's reference axis that crosses a hole
        there takes the surface without its holes (``_filled_chief``)."""
        hits = intersect(self.scene.surfaces[si], pk.o, pk.d, self.cfg)
        filled = self._filled.get(si)
        if filled is None or not pk.chief_obstructed or bool(np.isfinite(hits.t[0])):
            return hits
        chief = intersect(filled, pk.o[:1], pk.d[:1], self.cfg)
        if not bool(np.isfinite(chief.t[0])):
            return hits
        update = {}
        for name in hits.__dataclass_fields__:
            value = getattr(hits, name)
            if isinstance(value, np.ndarray) and value.shape[:1] == hits.t.shape:
                value = value.copy()
                value[0] = getattr(chief, name)[0]
            update[name] = value
        return replace(hits, **update)

    def _whole_annulus_inside(self, pk: _Packet, surf: SurfaceG, h: Hits, ap: Any, outside_envelope: float) -> bool:
        """Does the whole annulus lie inside the aperture / active area ``ap``? Yes when at most ``negligible_power_fraction``
        of the envelope's power falls outside it, or when the opening that cut the annulus provably maps inside it
        (``_edge_inside``)."""
        return outside_envelope <= self.cfg.negligible_power_fraction or self._edge_inside(pk, surf, h, ap)

    def _edge_inside(self, pk: _Packet, surf: SurfaceG, h: Hits, ap: Any) -> bool:
        """Does the opening that cut the annulus map inside ``ap`` (a convex aperture: no obstruction)?

        Paraxially a ray launched along the Gaussian phase front at transverse offset ``x`` from the chief ray (Gaussian
        axis ``x``) reaches ``m_x x`` with ``m = A + B / R``, while the envelope grows by ``w' / w = sqrt(m^2 + (B M^2 lambda
        / (pi n w^2))^2) >= |m|``: ``|m_x| <= M_x = w'_x / w_x`` and likewise on ``y``. The light that passed the opening
        (a disc ``D`` about its centre, which holds the chief ray) therefore lies in ``{(a s_x M_x u, b s_y M_y v): (u, v)
        in D, a, b in [0, 1]}`` for the signs ``s`` of ``m``, i.e. in the convex hull of ``K = diag(s M) D`` (the 64-gon
        around it), the projections of ``K`` onto the two axes and the chief ray. The annulus rays give each sign where
        they show it (their offsets at the opening and here correlate); otherwise both signs are taken. Every such hull,
        projected along the chief ray onto the surface, must lie inside ``ap``."""
        opening = pk.opening
        if opening is None or not pk.gstate.valid or not apertures.bounded(ap) or not apertures.convex(ap):
            return False
        g, d0 = pk.gstate, pk.d[0]
        scale_x, scale_y = g.radius("x") / opening.beam_radii[0], g.radius("y") / opening.beam_radii[1]
        signs = [self._edge_sign(pk, opening, axis, d0) for axis in (g.bx, g.by)]
        corners = EDGE_POLYGON_CORNERS
        phi = np.arange(corners) * (2.0 * math.pi / corners)
        polygon = opening.radius / cis(math.pi / corners).real
        normal = surf.cap_axis if surf.geometry == "sphere" else surf.normal
        along = float(dot(d0, normal))
        if abs(along) <= self.cfg.parallel_dot_tol:
            return False
        for sign_x in signs[0]:
            for sign_y in signs[1]:
                cos_phi, sin_phi = cos_sin(phi)
                x = sign_x * scale_x * (opening.centre[0] + polygon * cos_phi)
                y = sign_y * scale_y * (opening.centre[1] + polygon * sin_phi)
                zeros = np.zeros_like(x)
                xs = np.concatenate([x, x, zeros, [0.0]])
                ys = np.concatenate([y, zeros, y, [0.0]])
                offsets = np.outer(xs, g.bx) + np.outer(ys, g.by)
                on_plane = offsets - np.outer(dot(offsets, normal) / along, d0)
                u = float(h.u[0]) + dot(on_plane, surf.u_axis)
                v = float(h.v[0]) + dot(on_plane, surf.v_axis)
                if not bool(aperture_inside(ap, u, v, 0.0).all()):
                    return False
        return True

    def _edge_sign(self, pk: _Packet, opening: _Opening, axis: Array, d0: Array) -> tuple[float, ...]:
        """The sign of the paraxial map from the opening on one Gaussian axis, from the live annulus rays (their offsets
        from the chief ray at the opening and here along that axis); both signs when they do not show it."""
        column = 0 if axis is pk.gstate.bx else 1
        rays = np.flatnonzero(pk.alive & np.isfinite(opening.ray_offsets[:, column]))
        if rays.size == 0:
            return (1.0, -1.0)
        before = opening.ray_offsets[rays, column]
        rel = pk.o[rays] - pk.o[0]
        now = dot(rel - np.outer(dot(rel, d0), d0), axis)
        norm = math.sqrt(float(dot(before, before)) * float(dot(now, now)))
        if norm <= 0.0:
            return (1.0, -1.0)
        correlation = float(dot(before, now)) / norm
        if abs(correlation) < EDGE_SIGN_CORRELATION_MIN:
            return (1.0, -1.0)
        return (1.0,) if correlation > 0.0 else (-1.0,)

    def _annulus_through_aperture(self, pk: _Packet, surf: SurfaceG, h: Hits, chief_cos: float) -> str | None:
        """An obstructed packet meets a surface with its reference axis: the annulus passes the clear aperture only when
        every live ray of it does and the whole envelope of the unobstructed beam (its Gaussian state) fits inside it (at
        most ``negligible_power_fraction`` of the envelope's power outside: the annulus is part of that beam). Otherwise the
        power that passes is not computed (``_annulus_not_followed``: the issue ``ANNULAR_BEAM_NOT_FOLLOWED``; the rays that
        pass go on with a power that is not known)."""
        annulus = pk.alive & self._annulus_rays(pk)
        cut = annulus & ~h.inside
        if bool(cut.any()):
            term = self._annulus_not_followed(
                pk, surf, f"{int(cut.sum())} of its {int(annulus.sum())} pupil rays fall outside the clear aperture of "
                f"{surf.surface_id}", pupil_rays_cut=int(cut.sum()), pupil_rays_live=int(annulus.sum()))
            if term is None:
                pk.alive = pk.alive & ~cut  # (the rays the aperture cuts are retired; the rest goes on)
            return term
        if not apertures.bounded(surf.aperture):
            return None
        if not pk.gstate.valid:
            return self._annulus_not_followed(
                pk, surf, f"the envelope of the unobstructed beam is not known at {surf.surface_id} ({pk.gstate.reason_text}), "
                "so whether the whole annulus passes its clear aperture is not known")
        aperture = surf.aperture
        if aperture.holes:
            if not self._holes_in_shadow(pk, surf, h):
                return self._annulus_not_followed(
                    pk, surf, f"the holes of {surf.surface_id} are not proven to lie inside the shadow of the element the annulus "
                    "passed, so the part of the annulus they pass is not computed")
            aperture = replace(aperture, holes=())  # (inside the shadow: no light of the annulus reaches them)
        ap_u, ap_v, ap_cos = self._aperture_frame(surf, h, chief_cos)
        inside = transmission_fraction(pk.gstate, aperture, (float(h.u[0]), float(h.v[0])), ap_u, ap_v, ap_cos, self.cfg,
                                       self.quad_rule)
        if not self._whole_annulus_inside(pk, surf, h, surf.aperture, 1.0 - inside):
            return self._annulus_not_followed(
                pk, surf, f"the clear aperture of {surf.surface_id} does not contain the whole envelope of the unobstructed beam "
                f"({1.0 - inside:.4g} of its power falls outside) nor the image of the opening that cut the annulus, so the part "
                "of the annulus it passes is not computed", envelope_outside_fraction=1.0 - inside)
        return None

    def _clipped_branch(self, pk: _Packet, surf: SurfaceG, h: Hits, chief_cos: float, p_in: float) -> None:
        """The chief ray falls on the opaque rim beyond the clear aperture: the whole beam is booked ``CLIPPED``, but the part
        of it inside the opening passes physically (``UntracedBranch``; the whole beam when its Gaussian state is not valid)."""
        if not pk.gstate.valid:
            self._branch(pk, SINK_CLIPPED, surf, pk.d[0], pk.gstate, p_in)
            return
        ap_u, ap_v, ap_cos = self._aperture_frame(surf, h, chief_cos)
        opening = transmission_fraction(pk.gstate, surf.aperture, (float(h.u[0]), float(h.v[0])), ap_u, ap_v, ap_cos, self.cfg,
                                        self.quad_rule)
        # (its pupil rays: those inside the opening)
        self._branch(pk, SINK_CLIPPED, surf, pk.d[0], pk.gstate, opening * p_in, alive=pk.alive & h.inside)
        self._spill_beside(pk, surf, h, chief_cos, pk.gstate, p_in, opening)

    def _promote_reference(self, pk: _Packet, surf: SurfaceG, h: Hits, chief_cos: float,
                           p_in: float) -> tuple[str | None, Hits] | None:
        """The chief ray falls on the opaque rim beyond the clear aperture while live pupil samples pass the opening: the
        beam goes on as those samples, and the passing sample nearest their centroid becomes its reference ray (row 0)
        from here on -- every later decision (the next surface, the interaction coefficients, the plane crossings) is taken
        on it, as on the chief ray before. ``None``: the beam is not continued (the trace keeps the whole-beam ``CLIPPED``
        outcome): a probe, no pupil samples or none passing, a facet patch, a crystal leg, a power or polarization that is
        not known. Otherwise ``(termination, h)``: the hits with the reference's row (``termination`` set: the path ends).

        Power behind the aperture:
        * a valid Gaussian state: the Gaussian power inside the opening (``transmission_fraction``, exact for the incident
          Gaussian beam); the rest is booked ``APERTURE_TRUNCATION`` (the part beyond the element's support handed on as
          light passing beside it, ``_spill_beside``). The light that passed is not a Gaussian beam about its new reference:
          the state becomes invalid (``GAUSSIAN_SEVERE_TRUNCATION``) and later edges take the geometric pupil-sample basis.
        * no valid Gaussian state: the geometric pupil-sample basis (``_geometric_aperture``: the share of the equal-power
          samples that pass, with its sampling resolution; not computed where the samples do not follow the beam's
          diffraction).

        The reference's position, direction, polarization basis, Jones vector (scaled to the power that passes), optical
        path and length are the promoted sample's; the arrivals behind it say so (``reference_promoted``). The differential
        rays are relaunched about it, so the samples' diffraction check behind it keeps a first-order transfer (composed with
        the chief ray's up to here: ``_first_order``); an arrival reports no first-order transfer (``TRANSFER_PROMOTED``).
        The group path keeps the chief ray's up to here. Edge diffraction is not modelled, as for every cut."""
        cfg = self.cfg
        if (self.bounding or not pk.pupil_total or surf.facet is not None or pk.crystal is not None
                or pk.power_bound_W is not None or pk.polarization_open):
            return None
        annulus = self._annulus_rays(pk)
        live = pk.alive & annulus
        passing = live & h.inside & np.isfinite(h.t)
        if not bool(passing.any()):
            return None
        transfer, _why = self._first_order(pk, surf, h, promoted=True)  # (about the chief ray, before its rays retire)
        cut = live & ~passing
        if pk.gstate.valid:
            ap_u, ap_v, ap_cos = self._aperture_frame(surf, h, chief_cos)
            frac = transmission_fraction(pk.gstate, surf.aperture, (float(h.u[0]), float(h.v[0])), ap_u, ap_v, ap_cos, cfg,
                                         self.quad_rule)
            if frac <= 0.0:
                return None
            self._spill_beside(pk, surf, h, chief_cos, pk.gstate, p_in, frac)
            gstate, _code = mark_truncation(pk.gstate, 1.0 - frac, cfg)
            text = (f"the chief ray falls outside the clear aperture of {surf.surface_id}, which passes {frac:.4g} of the beam "
                    "power: the light that passes is not a Gaussian beam about the pupil ray that carries its reference from there")
            behind = gstate if not gstate.valid else gstate.invalidated(ErrorCode.GAUSSIAN_SEVERE_TRUNCATION, text)
            self._issue(make_issue(ErrorCode.GAUSSIAN_SEVERE_TRUNCATION, Status.OUT_OF_MODEL_SCOPE,
                                   f"aperture truncates {1.0 - frac:.4g} of the beam power", location=f"surface:{surf.surface_id}",
                                   loss_fraction=1.0 - frac, path_id=pk.path_id,
                                   suggested_action="enlarge the clear aperture or reduce the beam radius"))
            pk.jones = pk.jones.copy()
            pk.jones[0] = pk.jones[0] * math.sqrt(frac)
            quantum = cfg.power_ledger_fraction_resolution
            lost = round((1.0 - frac) / quantum) * quantum
            if lost > 0.0:
                self._flow(pk, SINK_APERTURE_TRUNCATION, p_in * lost, surf, "aperture")
        else:
            behind = pk.gstate
            term = self._geometric_aperture(pk, surf, h, cut, p_in)
            if term is not None:
                self._report_clip(pk, surf, h, chief_cos)
                return term, h
        pk.alive = pk.alive & ~cut
        rows = np.flatnonzero(pk.alive & annulus)
        n_pass, n_live = int(rows.size), int(live.sum())
        # the reference: the passing sample nearest the centroid of the passing samples (on the surface's u, v; the lowest
        # row on a tie)
        cu = correctly_rounded_sum(float(x) for x in h.u[rows]) / n_pass
        cv = correctly_rounded_sum(float(x) for x in h.v[rows]) / n_pass
        du, dv = h.u[rows] - cu, h.v[rows] - cv
        k = int(rows[int(np.argmin(du * du + dv * dv))])
        su, sv = float(h.u[k]) - float(h.u[0]), float(h.v[k]) - float(h.v[0])
        shift = math.sqrt(su * su + sv * sv)
        # (the power behind: computed, or -- where the samples do not follow the diffraction -- not computed, at most the bound)
        passed = ({"power_upper_bound_W": pk.power_bound_W, "untraced_power_W": pk.power_bound_W} if pk.power_bound_W is not None
                  else {"transmitted_power_W": pk.chief_power(), "transmitted_fraction": pk.chief_power() / p_in if p_in > 0.0 else 0.0})
        self._report_clip(pk, surf, h, chief_cos, promoted={
            "pupil_rays_passing": n_pass, "pupil_rays_live": n_live, "reference_pupil_ray": k, "reference_shift_mm": shift,
            **passed})
        pk.gstate = behind
        # row 0 becomes the promoted sample
        pk.o, pk.d, pk.e1, pk.e2 = pk.o.copy(), pk.d.copy(), pk.e1.copy(), pk.e2.copy()
        pk.opl, pk.length, pk.jones, pk.alive = pk.opl.copy(), pk.length.copy(), pk.jones.copy(), pk.alive.copy()
        for arr in (pk.o, pk.d, pk.e1, pk.e2, pk.opl, pk.length):
            arr[0] = arr[k]
        chief_p, sample_p = power2(pk.jones[0]), power2(pk.jones[k])
        if sample_p > 0.0:
            pk.jones[0] = pk.jones[k] * math.sqrt(chief_p / sample_p)
        pk.alive[0] = True
        h = Hits(h.t.copy(), h.point.copy(), h.normal.copy(), h.inside.copy(), h.u.copy(), h.v.copy())
        for arr in (h.t, h.point, h.normal, h.inside, h.u, h.v):
            arr[0] = arr[k]
        pk.reference_promoted = (*pk.reference_promoted, surf.surface_id)
        pk.transfer_base = None
        if pk.differential:
            self._relaunch_differential(pk, surf, h, transfer)
        return None, h

    def _relaunch_differential(self, pk: _Packet, surf: SurfaceG, h: Hits, transfer: Array | None) -> None:
        """Relaunch the differential rays about the promoted reference ray (row 0, at its hit on ``surf``; ``h`` is updated
        in place) as ``_root_packet`` launches them about the chief ray: displaced by ``+-dx`` along its basis vectors, and
        tilted by ``+-da``, each started ``RELAUNCH_BACKOFF_MM`` before the surface on its way to it. Their input differences
        on the surface's ``(u, v, l_u, l_v)`` (``D_in``) and the chief ray's transfer up to here (``transfer``, ``M_1``) give
        ``_Packet.transfer_base = D_in^-1 M_1``. Without ``M_1`` (it was not sampled or lost) or where a relaunched ray
        misses the surface, the differential rays retire (no transfer behind the promotion)."""
        cfg = self.cfg
        n_rays = pk.o.shape[0]
        rows = np.arange(n_rays - pk.differential, n_rays)
        if transfer is None or rows.size != DIFFERENTIAL_RAY_COUNT:
            pk.alive[rows] = False
            return
        d0, e1, e2, p0 = pk.d[0], pk.e1[0], pk.e2[0], h.point[0]
        dx, da = cfg.first_order_position_step_mm, cfg.first_order_angle_step_rad
        shifts = np.array([dx * e1, -dx * e1, dx * e2, -dx * e2])
        d_tilt, _ = normalize(d0 + np.array([da * e1, -da * e1, da * e2, -da * e2]), cfg.zero_vector_tol)
        dirs = np.vstack([np.repeat(d0[None, :], 4, 0), d_tilt])
        ends = np.vstack([p0 + shifts, np.repeat(p0[None, :], 4, 0)])
        hd = intersect(surf, ends - RELAUNCH_BACKOFF_MM * dirs, dirs, cfg)
        if not bool(np.isfinite(hd.t).all()):
            pk.alive[rows] = False
            return
        state = np.column_stack([hd.u, hd.v, dot(dirs, surf.u_axis), dot(dirs, surf.v_axis)])
        d_in = [[float(state[2 * j][i] - state[2 * j + 1][i]) for j in range(4)] for i in range(4)]
        inverse = _inverse4(d_in)
        if inverse is None:
            pk.alive[rows] = False
            return
        pk.transfer_base = matmul(np.array(inverse, dtype=np.float64), np.asarray(transfer, dtype=np.float64))
        e1_rows, e2_rows = parallel_transport_basis(np.repeat(e1[None, :], 8, 0), np.repeat(e2[None, :], 8, 0),
                                                    np.repeat(d0[None, :], 8, 0), dirs, cfg.antiparallel_tol)
        step = hd.t - RELAUNCH_BACKOFF_MM
        pk.o[rows], pk.d[rows], pk.e1[rows], pk.e2[rows] = hd.point, dirs, e1_rows, e2_rows
        pk.jones[rows] = pk.jones[0]
        pk.opl[rows] = pk.opl[0] + pk.n * step
        pk.length[rows] = pk.length[0] + step
        pk.alive[rows] = True
        h.t[rows], h.point[rows], h.normal[rows], h.inside[rows], h.u[rows], h.v[rows] = hd.t, hd.point, hd.normal, hd.inside, hd.u, hd.v

    def _geometric_aperture(self, pk: _Packet, surf: SurfaceG, h: Hits, cut: NDArray[np.bool_], p_in: float) -> str | None:
        """A beam without a valid Gaussian state whose chief ray passes a bounded clear aperture (``cut``: the live pupil rays
        it cuts): the geometric pupil-sample basis, where its samples represent the beam.

        * The samples do not follow the beam's diffraction (``_sample_picture``: the spread of the Gaussian beam they were
          launched from beyond them is more than ``GEOMETRIC_DIFFRACTION_RATIO`` of the radius they picture) and the aperture's
          edge comes within reach of the beam (``_unresolved_reach``): what passes is not computed, the path ends
          (``_not_computed``). An aperture beyond that reach passes the beam whole.
        * Some samples are cut: ``_geometric_cut``.
        * None is cut, but the edge passes within one sample spacing beyond a sample at the frontier of the live samples
          (``_frontier_cells``): the edge may cut part of that sample's cell, which no sample resolves. The power stays and
          its sampling resolution grows by those samples' shares (``GEOMETRIC_POWER_BASIS``).

        Without differential rays to the surface (an axicon crossed, a differential ray retired) the diffraction spread is
        not known and the samples are taken to represent the beam."""
        live = pk.alive & self._annulus_rays(pk)
        if not bool(live.any()):
            return None
        why = self._unresolved_reach(pk, surf, h, live, surf.aperture)
        if why is not None:
            return self._not_computed(pk, surf, f"the clear aperture of {surf.surface_id}", **why)
        if bool(cut.any()):
            return self._geometric_cut(pk, surf, h, cut, p_in)
        cells = self._frontier_cells(pk, live, h, surf.aperture)
        frontier = int(cells.sum())
        if not frontier:
            return None
        self._mark_edge_cells(pk, cells)
        n_live = int(live.sum())
        self._add_sampling(pk, frontier / n_live)
        known = pk.power_bound_W is None
        reason = pk.gstate.reason or ErrorCode.OUT_OF_MODEL_SCOPE
        power_text = (f"its {p_in:.4g} W pass within a sampling resolution of {pk.sampling * p_in:.3g} W" if known
                      else "its power is not known")
        self._issue(make_issue(
            ErrorCode.GEOMETRIC_POWER_BASIS, Status.WARNING,
            f"the beam has no valid Gaussian state ({pk.gstate.reason_text or reason.value}) and the edge of the clear aperture "
            f"of {surf.surface_id} passes within one sample spacing beyond {frontier} of its {n_live} pupil rays, cutting none: "
            f"part of those samples' cells may be cut, which the samples do not resolve ({power_text}); its rays are traced on "
            "and its Gaussian beam metrics are not available", location=f"surface:{surf.surface_id}", path_id=pk.path_id,
            reason_code=reason.value, pupil_rays_cut=0, pupil_rays_live=n_live, pupil_rays_on_boundary=frontier,
            transmitted_fraction=1.0, **({"transmitted_power_W": p_in, "sampling_resolution_W": pk.sampling * p_in} if known else {}),
            suggested_action="read power_W within its sampling resolution; raise compute_budget.pupil_ring_count and pupil_points_per_ring together to narrow it (about as 1 / rings)"))
        return None

    def _geometric_divergence(self, pk: _Packet, surf: SurfaceG, diverged: NDArray[np.bool_]) -> str | None:
        """A beam without a valid Gaussian state some of whose live pupil samples part from it on the way to the surface
        ``surf`` its reference ray meets next (``diverged``: they pass beside that element's support, or meet another surface
        first): on the geometric pupil-sample basis the light they carry leaves the beam there. The beam goes on with the
        share ``n_stay / n_live`` of its power; the rest is booked ``APERTURE_TRUNCATION`` at ``surf`` and handed on as an
        ``UntracedBranch`` with the parted rays (light passing beside the element, followed by the untraced-light bounds).
        The power's sampling resolution grows by the samples whose cells may straddle the parting (a live neighbour with
        the other fate) over those that stay (``GEOMETRIC_POWER_BASIS``). Where every live sample parts, no sample is left to
        share the power the reference ray goes on with: it is not computed and the path ends (``_not_computed``; the
        termination is returned). A probe, a beam behind a central obstruction (its envelope decides), one of unknown power
        or without live samples is left as it is. With a valid Gaussian state the Gaussian integrals at each support and
        aperture decide instead (``_spill_beside``)."""
        if pk.gstate.valid or self.bounding or pk.power_bound_W is not None or pk.chief_obstructed or not pk.pupil_total:
            return None
        annulus = self._annulus_rays(pk)
        live = pk.alive & annulus
        gone = diverged & annulus
        n_live, n_gone = int(live.sum()), int(gone.sum())
        if n_gone == 0 or n_live == 0:
            return None
        if n_gone >= n_live:
            return self._not_computed(pk, surf, f"the way to {surf.surface_id}", pupil_rays_parted=n_gone)
        n_stay = n_live - n_gone
        p_in = pk.chief_power()
        gone_W = p_in * (n_gone / n_live)
        boundary = int(self._boundary_cells(pk, live, gone, outer=False).sum())
        self._branch(pk, SINK_APERTURE_TRUNCATION, surf, pk.d[0], pk.gstate, gone_W, alive=pk.alive & gone)
        pk.jones = pk.jones.copy()
        pk.jones[0] = pk.jones[0] * math.sqrt(n_stay / n_live)
        self._flow(pk, SINK_APERTURE_TRUNCATION, gone_W, surf, "beside")
        self._add_sampling(pk, boundary / n_stay)
        reason = pk.gstate.reason or ErrorCode.OUT_OF_MODEL_SCOPE
        self._issue(make_issue(
            ErrorCode.GEOMETRIC_POWER_BASIS, Status.WARNING,
            f"the beam has no valid Gaussian state ({pk.gstate.reason_text or reason.value}) and {n_gone} of its {n_live} pupil "
            f"rays part from it before {surf.surface_id} (beside the element or onto another surface): their share "
            f"({n_gone / n_live:.4g}, {gone_W:.4g} W) leaves the beam there and goes on as untraced light, and the beam carries "
            f"{p_in - gone_W:.4g} W within a sampling resolution of {pk.sampling * (p_in - gone_W):.3g} W",
            location=f"surface:{surf.surface_id}", path_id=pk.path_id, reason_code=reason.value, pupil_rays_cut=n_gone,
            pupil_rays_live=n_live, pupil_rays_on_boundary=boundary, transmitted_fraction=n_stay / n_live,
            transmitted_power_W=p_in - gone_W, cut_power_W=gone_W, sampling_resolution_W=pk.sampling * (p_in - gone_W),
            suggested_action="read power_W within its sampling resolution; raise compute_budget.pupil_ring_count and pupil_points_per_ring together to narrow it (about as 1 / rings)"))
        return None

    def _met_beside_chief(self, pk: _Packet, diverged: NDArray[np.bool_], nearest: NDArray[np.intp], lead: int,
                          lead_hit: Hits) -> tuple[str, ...]:
        """A beam with a valid Gaussian state some of whose pupil samples meet another physical surface first on the way to
        the surface ``lead`` its chief ray meets, at points where they would have reached ``lead`` inside its clear aperture
        (``diverged`` / ``nearest``: e.g. the back of a Cassegrain secondary mirror beside the chief ray of a decentred beam).
        The beam's power follows its chief ray: the Gaussian integrals at ``lead`` count that light as the beam's, which no
        longer carries it. Where the share of the beam whose energy-flow lines may meet the surface on this leg is bounded
        (``lower_share.possible_meeting_share``, as ``_taken_before`` bounds it for a plane on the leg; not a face of the
        body the beam travels in) the surface joins the shares taken on the way (``_Packet.take``: the beam's own light that
        reaches a plane behind lies between the power stated there less that bound and the power stated) with a
        ``POWER_TAKEN_BEFORE_PLANE`` warning -- only for a surface that stops the light it meets (``_stops_light``): light a
        mirror, a lens or plate face, a splitter or a scattering absorber turns may reach a plane behind, where the power
        stated is then no upper end. Otherwise it joins ``power_scope`` (the beam's power behind it is out of model scope,
        never a known value) with an ``OUT_OF_MODEL_SCOPE`` issue. Either issue states the share of the pupil
        samples the surface takes (equal-power samples: an estimate of the power share) and, for a plane or sag surface,
        the Gaussian integral of the beam over its support in the plane of its vertex (``_support_share_beside``). Returns
        the surfaces counted. Samples that would have missed ``lead``'s clear aperture are
        not counted: the Gaussian integral at ``lead`` already takes that light out of the beam. Without a valid Gaussian
        state ``_geometric_divergence`` books their share instead; a beam behind a central obstruction (``_obstruct``), a
        probe, a beam of unknown power, an observation plane (``_leg_end``) and another patch of the same facet surface
        (``FACET_EDGE_DIVIDES_BEAM``) keep their own rules."""
        if not pk.gstate.valid or self.bounding or pk.power_bound_W is not None or pk.chief_obstructed or not pk.pupil_total:
            return ()
        surfaces = self.scene.surfaces
        target = surfaces[lead]
        live = pk.alive & self._annulus_rays(pk)
        met = live & diverged & (nearest >= 0) & (nearest != lead) & np.isfinite(lead_hit.t) & lead_hit.inside
        if not bool(met.any()):
            return ()
        counts: dict[int, int] = {}
        for ray in np.flatnonzero(met):
            si = int(nearest[ray])
            other = surfaces[si]
            if other.interaction.kind == "observe":
                continue
            if target.facet is not None and other.facet is not None and other.facet.mesh_id == target.facet.mesh_id:
                continue
            counts[si] = counts.get(si, 0) + 1
        n_live = int(live.sum())
        counted: list[str] = []
        leg = self._leg(pk, (lead_hit.point[0], lead_hit.normal[0], float(lead_hit.t[0])), lead)
        k_sigma = self._envelope_sigmas(pk)
        for si in sorted(counts):
            surface_id = surfaces[si].surface_id
            if surface_id in pk.power_scope:
                continue
            count = counts[si]
            counted.append(surface_id)
            p_in = pk.chief_power()
            share = self._support_share_beside(pk, surfaces[si])
            on_support = {} if share is None else {"support_fraction": share, "support_power_W": p_in * share}
            context = dict(location=f"surface:{surface_id}", path_id=pk.path_id, surface_ids=[surface_id],
                           next_surface_id=target.surface_id, pupil_rays_met=count, pupil_rays_live=n_live,
                           met_fraction=count / n_live, **on_support,
                           suggested_action="move the element or the beam so the beam passes the element whole or clears it, or "
                                            "read the power at a plane before that surface")
            bound = None
            if leg is not None and pk.medium not in self._interior.get(surfaces[si].owner_id, ()):
                bound = possible_meeting_share(surfaces[si], leg, k_sigma, self.cfg)
            if bound is not None and self._stops_light(surface_id):
                pk.take(surface_id, p_in * bound)
                self._issue(make_issue(
                    ErrorCode.POWER_TAKEN_BEFORE_PLANE, Status.WARNING,
                    f"at most {p_in * bound:.4g} W of beam {pk.path_id} meets {surface_id} before {target.surface_id}",
                    taken_fraction_upper=bound, taken_power_upper_W=p_in * bound, **context))
                continue
            pk.power_scope = (*pk.power_scope, surface_id)
            self._issue(make_issue(
                ErrorCode.OUT_OF_MODEL_SCOPE, Status.OUT_OF_MODEL_SCOPE,
                f"{count} of the {n_live} pupil rays of beam {pk.path_id} meet {surface_id} before {target.surface_id} and the "
                f"share of the beam it takes {'is not bounded' if bound is None else 'is turned by it and may reach a plane behind'}: "
                f"its power behind {target.surface_id} is not computed", **context))
        return tuple(counted)

    def _support_share_beside(self, pk: _Packet, surf: SurfaceG) -> float | None:
        """The share of the packet's Gaussian beam on the support of a plane or sag surface its chief ray does not meet: the
        Gaussian integral (``transmission_fraction``) over the support in the plane of the surface's vertex, the beam
        propagated to where the chief ray crosses that plane, on the surface's ``(u, v)`` (a sag surface's clear aperture is
        measured on them along its axis). ``None`` for another geometry, or when the chief ray does not cross the plane ahead."""
        if surf.geometry not in ("plane", "sag"):
            return None
        d0 = pk.d[0]
        along = float(dot(d0, surf.normal))
        if abs(along) <= self.cfg.parallel_dot_tol:
            return None
        t = float(dot(surf.origin - pk.o[0], surf.normal)) / along
        if not t > 0.0:
            return None
        rel = pk.o[0] + t * d0 - surf.origin
        g = propagate(pk.gstate, t)
        return transmission_fraction(g, replace(support_of(surf), obstruction=0.0), (float(dot(rel, surf.u_axis)), float(dot(rel, surf.v_axis))),
                                     surf.u_axis, surf.v_axis, abs(along), self.cfg, self.quad_rule)

    @staticmethod
    def _add_sampling(pk: _Packet, added: float) -> None:
        """Compound the packet's relative sampling resolution with a step's ``added`` (``r + r' + r r'``)."""
        before = 0.0 if pk.sampling is None else pk.sampling
        pk.sampling = before + added + before * added

    def _geometric_cut(self, pk: _Packet, surf: SurfaceG, h: Hits, cut: NDArray[np.bool_], p_in: float) -> str | None:
        """A beam without a valid Gaussian state (``GAUSSIAN_SEVERE_TRUNCATION`` upstream, ``GAUSSIAN_SURFACE_TOO_STRONG``, ...)
        whose chief ray passes a clear aperture that cuts some of its pupil rays (``cut``): the geometric pupil-sample basis.

        The pupil samples carry equal shares of the source line's power (``pupil.py``: equal-power rings and azimuths) and
        each keeps its share along its ray, so the aperture passes the share ``f = n_pass / n_live`` of the live pupil
        samples. The cut rays retire (as with a valid Gaussian state), the packet's power is scaled by ``f`` and the rest,
        ``p_in (1 - f)``, is booked ``APERTURE_TRUNCATION``. Outside the element's own body the cut light beyond the element's
        support goes on physically beside it (as ``_spill_beside`` for a Gaussian beam): the share of the cut samples that
        land beyond the support, and of those on the support whose cells may reach beyond it, is recorded as an
        ``UntracedBranch`` with the cut rays. The power is a sampled quantity: its resolution grows by the samples whose cell
        may straddle the cut (``_boundary_cells``, and the passing samples at the frontier of the live ones whose cell the
        edge may reach, ``_frontier_cells``) over the samples that pass, as a fraction of the power that passes
        (``_Packet.sampling``). Diffraction at the edge is not modelled. The Gaussian beam metrics stay unavailable behind it
        (the state keeps its reason). With no pupil sample left the power that passes is not computed: ``_not_computed``."""
        annulus = pk.alive & self._annulus_rays(pk)
        n_live, n_cut = int(annulus.sum()), int(cut.sum())
        n_pass = n_live - n_cut
        if n_pass <= 0:
            return self._not_computed(pk, surf, f"the clear aperture of {surf.surface_id}", pupil_rays_cut=n_cut)
        cells = self._frontier_cells(pk, annulus & ~cut, h, surf.aperture)
        boundary = int((self._boundary_cells(pk, annulus, cut) | cells).sum())
        self._mark_edge_cells(pk, cells)
        fraction = n_pass / n_live
        cut_W = p_in * (n_cut / n_live)
        support = support_of(surf)
        if pk.medium not in self._interior.get(surf.owner_id, ()) and support.shape != "none":
            # the cut light beyond the element's support passes beside it: the samples cut there, and those cut on the
            # support whose cells may reach beyond it (``_sampling_boundary`` between the two)
            beside = cut & ~aperture_inside(support, h.u, h.v, 0.0)
            reach = int(beside.sum()) + self._sampling_boundary(pk, cut, beside)
            if reach:
                self._branch(pk, SINK_APERTURE_TRUNCATION, surf, pk.d[0], pk.gstate, cut_W * min(reach / n_cut, 1.0),
                             alive=pk.alive & cut)
        pk.jones = pk.jones.copy()
        pk.jones[0] = pk.jones[0] * math.sqrt(fraction)
        self._flow(pk, SINK_APERTURE_TRUNCATION, cut_W, surf, "aperture")
        self._add_sampling(pk, boundary / n_pass)
        known = pk.power_bound_W is None
        reason = pk.gstate.reason or ErrorCode.OUT_OF_MODEL_SCOPE
        power_text = (f"{p_in * fraction:.4g} W pass and {cut_W:.4g} W are cut, within a sampling resolution of "
                      f"{pk.sampling * p_in * fraction:.3g} W" if known else "its power is not known")
        self._issue(make_issue(
            ErrorCode.GEOMETRIC_POWER_BASIS, Status.WARNING,
            f"the beam has no valid Gaussian state ({pk.gstate.reason_text or reason.value}) and the clear aperture of "
            f"{surf.surface_id} cuts {n_cut} of its {n_live} pupil rays: its power behind this surface is the share of the "
            f"equal-power pupil samples that pass ({fraction:.4g}; {power_text}); its rays are traced on and its Gaussian beam "
            "metrics are not available", location=f"surface:{surf.surface_id}", path_id=pk.path_id, reason_code=reason.value,
            pupil_rays_cut=n_cut, pupil_rays_live=n_live, pupil_rays_on_boundary=boundary, transmitted_fraction=fraction,
            **({"transmitted_power_W": p_in * fraction, "cut_power_W": cut_W,
                "sampling_resolution_W": pk.sampling * p_in * fraction} if known else {}),
            suggested_action="read power_W within its sampling resolution, and the ray results (spot, angles of incidence, "
                             "per-ray paths); raise compute_budget.pupil_ring_count and pupil_points_per_ring together to narrow the resolution (about as 1 / rings)"))
        return None

    def _geometric_area(self, pk: _Packet, surf: SurfaceG, h: Hits, area: ApertureG,
                        oid: str) -> tuple[float, float] | dict[str, Any]:
        """A beam without a valid Gaussian state whose chief ray falls on the bounded active area ``area`` of the absorbing
        plane ``oid``: the share of its power the area takes on the geometric pupil-sample basis (as ``_geometric_cut``, the
        area the opening; ``GEOMETRIC_POWER_BASIS`` where it is not all), with the packet's sampling resolution compounded
        with that step's (``(share, sampling)``). Where it is not computable, why (a ``dict`` for ``_not_computed``): no live
        pupil sample, none on the area, or samples that do not follow the beam's diffraction while the area's edge is within
        its reach (``_unresolved_reach``)."""
        live = pk.alive & self._annulus_rays(pk)
        if not bool(live.any()):
            return {"pupil_rays_live": 0}
        why = self._unresolved_reach(pk, surf, h, live, area)
        if why is not None:
            return why
        on = live & aperture_inside(area, h.u, h.v, self.cfg.aperture_edge_tol_mm)
        off = live & ~on
        n_live, n_on = int(live.sum()), int(on.sum())
        if n_on == 0:
            return {"pupil_rays_live": n_live, "pupil_rays_on_area": 0}
        boundary = self._frontier_cells(pk, on, h, area)
        if bool(off.any()):
            boundary = boundary | self._boundary_cells(pk, live, off)
        n_b = int(boundary.sum())
        before = 0.0 if pk.sampling is None else pk.sampling
        if not n_b and n_on == n_live:
            return 1.0, pk.sampling  # (the area takes every sample, and no edge reaches a cell)
        added = n_b / n_on
        sampling = before + added + before * added
        share = n_on / n_live
        known = pk.power_bound_W is None
        reason = pk.gstate.reason or ErrorCode.OUT_OF_MODEL_SCOPE
        power = pk.chief_power() * share
        power_text = f"{power:.4g} W within a sampling resolution of {sampling * power:.3g} W" if known else "its power is not known"
        self._issue(make_issue(
            ErrorCode.GEOMETRIC_POWER_BASIS, Status.WARNING,
            f"the beam has no valid Gaussian state ({pk.gstate.reason_text or reason.value}) and {n_on} of its {n_live} pupil rays "
            f"fall on the active area of observation {oid!r}: the power it detects is the share of the equal-power pupil samples "
            f"on it ({share:.4g}; {power_text})", location=f"observation:{oid}", path_id=pk.path_id, reason_code=reason.value,
            pupil_rays_on_area=n_on, pupil_rays_live=n_live, pupil_rays_on_boundary=n_b, active_area_power_fraction=share,
            **({"detected_power_W": power, "sampling_resolution_W": sampling * power} if known else {}),
            suggested_action="read the detected power within its sampling resolution; raise compute_budget.pupil_ring_count and pupil_points_per_ring together to narrow it (about as 1 / rings)"))
        return share, sampling

    def _unresolved_reach(self, pk: _Packet, surf: SurfaceG, h: Hits, live: NDArray[np.bool_],
                          region: ApertureG) -> dict[str, Any] | None:
        """``None`` where the live pupil samples ``live`` represent the beam for the region ``region`` of the surface (its
        clear aperture, a plane's active area): their picture follows the beam's diffraction (``_sample_picture``), or the
        region holds the beam in either picture (a disc about the chief ray of ``aperture_quad_extent_sigmas`` times the
        larger of the samples' farthest radius and the spread beyond them, where the Gaussian tail is nil, fits in it).
        Otherwise the numbers that say why (for ``_not_computed``)."""
        spread, rms, far = self._sample_picture(pk, surf, h, live)
        if spread is None or spread <= GEOMETRIC_DIFFRACTION_RATIO * SQRT_TWO * rms:
            return None
        reach = self.cfg.aperture_quad_extent_sigmas * math.sqrt(far * far + spread * spread)
        if apertures.room(region, float(h.u[0]), float(h.v[0])) >= reach:
            return None
        return {"diffraction_spread_mm": spread, "pupil_rms_radius_mm": rms}

    def _sample_picture(self, pk: _Packet, surf: SurfaceG, h: Hits, live: NDArray[np.bool_]) -> tuple[float | None, float, float]:
        """How the live pupil samples ``live`` picture the beam at the surface: the diffraction spread they do not follow
        (``None``: not known), and their rms and farthest radius about the chief ray (mm, in the surface's ``(u, v)``).

        The pupil rays leave the source as the energy-flow tangents of its Gaussian beam (``_root_packet``: offsets at the
        beam's radius, slopes of its wavefront), so through a paraxial system of ray-transfer matrix ``[[A, B], [C, D]]``
        from the source plane they land at ``(A + B / R) x0`` while the beam's radius is ``|A + B / q| w``: the rays'
        picture and ``B theta`` added in quadrature (``theta = lambda M^2 / (pi n w)``, ``_Packet.launch_spread``). ``B`` is
        the differential rays' (``_first_order``); the spread is the largest semi-axis of the image of the launch angles'
        ellipse, ``B diag(theta_x, theta_y)``. A beam truncated on its way spreads at least as fast as that Gaussian."""
        rows = np.flatnonzero(live)
        du, dv = h.u[rows] - h.u[0], h.v[rows] - h.v[0]
        r2 = du * du + dv * dv
        rms = math.sqrt(correctly_rounded_sum(float(x) for x in r2) / rows.size)
        far = math.sqrt(float(r2.max()))
        if pk.launch_spread is None:
            return None, rms, far
        matrix, _why = self._first_order(pk, surf, h, promoted=True)
        if matrix is None:
            return None, rms, far
        tx, ty = pk.launch_spread
        p, q, r, t = float(matrix[0, 2]) * tx, float(matrix[0, 3]) * ty, float(matrix[1, 2]) * tx, float(matrix[1, 3]) * ty
        trace = p * p + q * q + r * r + t * t
        det = p * t - q * r
        return math.sqrt(0.5 * (trace + math.sqrt(max(trace * trace - 4.0 * det * det, 0.0)))), rms, far

    @staticmethod
    def _cell_grid(pk: _Packet, live: NDArray[np.bool_], cut: NDArray[np.bool_]) -> tuple[NDArray[np.int8], list[NDArray[np.int8]]]:
        """The source's ring layout of pupil samples (``pupil.py``: ``rings x per_ring``): each cell's state (``_GONE``: not
        among ``live``; ``_PASSED``; ``_TAKEN``: among ``cut``) and the states of its six neighbours: the two beside it on
        its ring and the two nearest on the rings inside and outside it (ring ``k``'s point ``m`` lies between the points
        ``m - 1`` and ``m`` of ring ``k + 1`` when ``k`` is even, ``m`` and ``m + 1`` when it is odd; ``_NONE`` inside the
        innermost and outside the outermost ring)."""
        rings, per_ring = pk.src.ring_count, pk.src.points_per_ring
        rows = np.flatnonzero(live)
        samples = rows - 1 if pk.launch_rows is None else pk.launch_rows[rows - 1]
        state = np.full(rings * per_ring, _GONE, dtype=np.int8)
        state[samples] = np.where(cut[rows], _TAKEN, _PASSED)
        grid = state.reshape(rings, per_ring)
        return grid, _neighbours(grid, _NONE)

    @classmethod
    def _boundary_cells(cls, pk: _Packet, live: NDArray[np.bool_], cut: NDArray[np.bool_], *,
                        outer: bool = True) -> NDArray[np.bool_]:
        """The cells (``_cell_grid``) of the live pupil samples ``live`` that may straddle the cut of a clear aperture
        (``cut``), in samples' shares: one of whose live neighbours met the other fate there (passed / cut), and (``outer``)
        a passing sample of the outermost ring (its cell reaches out to the Gaussian's tail, which a finite aperture cuts):
        up to its whole share may be on the other side. Samples retired earlier do not count as neighbours: the live
        samples share the power that reaches the aperture equally."""
        grid, others = cls._cell_grid(pk, live, cut)
        straddles = np.zeros(grid.shape, dtype=bool)
        for other in others:
            straddles |= (other != _GONE) & (other != _NONE) & (other != grid)
        if outer:
            straddles[-1] |= grid[-1] == _PASSED
        return straddles & (grid != _GONE)

    @classmethod
    def _sampling_boundary(cls, pk: _Packet, live: NDArray[np.bool_], cut: NDArray[np.bool_]) -> int:
        """How many live pupil samples ``live`` may have a cell that straddles the cut of a clear aperture (``cut``):
        ``_boundary_cells`` counted."""
        return int(cls._boundary_cells(pk, live, cut).sum())

    @staticmethod
    def _mark_edge_cells(pk: _Packet, cells: NDArray[np.bool_]) -> None:
        """Record the frontier cells an edge may have cut (``_Packet.edge_cells``)."""
        if bool(cells.any()):
            pk.edge_cells = cells.ravel().copy() if pk.edge_cells is None else pk.edge_cells | cells.ravel()

    def _frontier_cells(self, pk: _Packet, passing: NDArray[np.bool_], h: Hits, region: ApertureG) -> NDArray[np.bool_]:
        """The cells (``_cell_grid``) of the passing live pupil samples ``passing`` at the frontier of the live samples --
        on the outermost ring (the cell reaches out to the Gaussian's tail), or beside a sample retired earlier or cut here
        (it reaches out to that cut's edge) -- that the edge of the region ``region`` may cut although the sample lies inside
        it: the region holds no disc about the sample as wide as the sample's spacing to its nearest passing neighbour at
        the surface (a cell's size; ``apertures.room``). An edge farther out is taken to cut nothing of the cell. A cell an edge may already have
        cut (``_Packet.edge_cells``) is not counted again: its share can be lost once. None where the region holds the image
        of an opening the beam passed earlier (``_edge_held``): all light that passed that opening is inside the region."""
        grid, others = self._cell_grid(pk, passing, np.zeros_like(passing))
        frontier = np.zeros(grid.shape, dtype=bool)
        for other in others:
            frontier |= other == _GONE
        frontier[-1] = True
        frontier &= grid == _PASSED
        if pk.edge_cells is not None:
            frontier &= ~pk.edge_cells.reshape(grid.shape)
        if not bool(frontier.any()):
            return frontier
        if self._edge_held(pk, passing, h, region):
            return np.zeros(grid.shape, dtype=bool)
        rings, per_ring = grid.shape
        rows = np.flatnonzero(passing)
        samples = rows - 1 if pk.launch_rows is None else pk.launch_rows[rows - 1]
        row_of = np.full(rings * per_ring, -1, dtype=np.intp)
        row_of[samples] = rows
        cells = row_of.reshape(rings, per_ring)
        near = _neighbours(cells, -1)
        out = np.zeros(grid.shape, dtype=bool)
        for k, m in zip(*np.nonzero(frontier)):
            row = int(cells[k, m])
            u, v = float(h.u[row]), float(h.v[row])
            spacing = math.inf
            for nb in near:
                other = int(nb[k, m])
                if other >= 0 and other != row:
                    du, dv = float(h.u[other]) - u, float(h.v[other]) - v
                    spacing = min(spacing, math.sqrt(du * du + dv * dv))
            out[k, m] = apertures.room(region, u, v) < spacing
        return out

    def _note_opening(self, pk: _Packet, surface_id: str, region: ApertureG, h: Hits) -> None:
        """Keep the bounded opening ``region`` the packet's chief ray passed here, with where its live pupil samples cross it
        (``_Packet.openings``)."""
        if self.bounding or not pk.pupil_total:
            return
        rows = np.flatnonzero(pk.alive & self._annulus_rays(pk))
        if rows.size == 0:
            return
        samples = rows - 1 if pk.launch_rows is None else pk.launch_rows[rows - 1]
        uv = np.full((pk.src.ring_count * pk.src.points_per_ring, 2), np.nan)
        uv[samples, 0], uv[samples, 1] = h.u[rows], h.v[rows]
        pk.openings = (*pk.openings[-(OPENINGS_KEPT - 1):], _Passed(surface_id, region, uv))

    def _edge_held(self, pk: _Packet, passing: NDArray[np.bool_], h: Hits, region: ApertureG) -> bool:
        """Does the region ``region`` hold all light that passed an opening the beam passed earlier (``_Packet.openings``,
        the most recent first)? The live samples ``passing`` map that opening's ``(u, v)`` to this surface's: their
        least-squares affine map ``p -> M p + b`` (``_affine_map``), with its largest residual over them grown to the
        opening's edge as the cube of the ratio of the edge's distance to the farthest sample's from their centroid
        (``EDGE_MAP_GROWTH``: the map's departure from affine is taken to grow no faster than third-order aberration). The
        region must hold the image of the opening grown by that margin, within ``aperture_edge_tol_mm``
        (``apertures.holds_affine_image``): then no light that passed the opening reaches the region's edge -- the same
        opening met again, or one that contains it, in a frame where the beam keeps its size or shrinks. At least three
        samples that crossed the opening, not on a line, are needed."""
        rows = np.flatnonzero(passing)
        if rows.size < 3 or not pk.openings:
            return False
        samples = rows - 1 if pk.launch_rows is None else pk.launch_rows[rows - 1]
        for opening in reversed(pk.openings):
            before = opening.uv[samples]
            known = np.isfinite(before[:, 0])
            if int(known.sum()) < 3:
                continue
            fit = _affine_map(before[known, 0], before[known, 1], h.u[rows][known], h.v[rows][known])
            if fit is None:
                continue
            matrix, offset, residual, centroid, far = fit
            reach = apertures.farthest(opening.aperture, opening.aperture.offset_u - centroid[0],
                                       opening.aperture.offset_v - centroid[1])
            margin = residual * max(1.0, reach / far) ** EDGE_MAP_GROWTH if far > 0.0 else math.inf
            if apertures.holds_affine_image(region, opening.aperture, matrix, offset, margin, self.cfg.aperture_edge_tol_mm):
                return True
        return False

    def _power_not_computed(self, pk: _Packet, surf: SurfaceG | None, detail: str) -> None:
        """The power the packet carries from here on is not computed. Its present power goes once to the ledger's
        ``TERMINATED_OUT_OF_MODEL`` (light of unknown fate), its Jones vector to zero, and ``power_bound_W`` holds what it had
        (what passes a passive element is at most what arrives). The packet goes on as geometry: nothing more is booked
        for it (``_flow``; inside a handler too, so ``_record_interaction_loss`` books nothing after it), no power floor
        ends it (``_spent``), and every plane reads it with a null power and ``power_upper_bound_W``."""
        if pk.power_bound_W is None:
            power = pk.chief_power()
            self._flow(pk, SINK_TERMINATED_OUT_OF_MODEL, power, surf, detail)
            pk.power_bound_W = power
        pk.jones = np.zeros_like(pk.jones)
        pk.polarization_open = False

    def _not_computed(self, pk: _Packet, surf: SurfaceG, where: str, **ctx: Any) -> str | None:
        """A beam without a valid Gaussian state (``GAUSSIAN_SEVERE_TRUNCATION`` upstream, or another reason) whose power at
        ``where`` (a clear aperture, a plane's active area) the geometric pupil-sample basis does not give: every pupil
        sample is cut while the chief ray passes, no sample falls on the area, none is left, or the samples do not follow
        the beam's diffraction there (``ctx``: the numbers). The power that passes is not computed (the Gaussian integral
        needs a Gaussian): from here on the beam goes on as geometry with a power that is not known, at most what arrived
        (``_power_not_computed``). Returns ``None`` (the caller goes on and retires the rays the surface cuts). A beam whose
        power is already not known raises nothing more."""
        reason = pk.gstate.reason or ErrorCode.OUT_OF_MODEL_SCOPE
        if "diffraction_spread_mm" in ctx and pk.samples_unresolved is None:
            # (the first surface where the samples do not follow the diffraction: arrivals behind state their radius as an estimate)
            pk.samples_unresolved = (surf.surface_id, float(ctx["diffraction_spread_mm"]),
                                     GEOMETRIC_DIFFRACTION_RATIO * SQRT_TWO * float(ctx["pupil_rms_radius_mm"]), reason.value)
        if pk.power_bound_W is not None:
            return None
        if "diffraction_spread_mm" in ctx:
            why = (f"its pupil rays do not follow its diffraction within reach of {where} (it spreads "
                   f"{ctx['diffraction_spread_mm']:.3g} mm beyond them, against their rms radius of {ctx['pupil_rms_radius_mm']:.3g} mm)")
        elif ctx.get("pupil_rays_parted"):
            why = f"all {ctx['pupil_rays_parted']} of its live pupil rays part from it on {where}"
        elif ctx.get("pupil_rays_cut"):
            why = f"{where} cuts all {ctx['pupil_rays_cut']} of its live pupil rays"
        else:
            why = f"no pupil ray of it falls on {where}"
        power = pk.chief_power()
        self._issue(make_issue(
            reason, Status.OUT_OF_MODEL_SCOPE,
            f"the power of beam {pk.path_id} behind {surf.surface_id} is not computed, at most {power:.4g} W: {why}",
            location=f"surface:{surf.surface_id}", path_id=pk.path_id, power_upper_bound_W=power, untraced_power_W=power,
            reason_code=reason.value, **ctx,
            suggested_action="enlarge the clear apertures so the beam passes whole, or read the power at a plane before this surface"))
        self._power_not_computed(pk, surf, TRUNCATED_NOT_FOLLOWED)
        return None

    def _annulus_not_followed(self, pk: _Packet, surf: SurfaceG | None, why: str, **ctx: Any) -> str | None:
        """An obstructed packet whose annulus is cut further on, or whose whole passing cannot be decided: the power that
        continues is not computed. A traced beam goes on as geometry with a power that is not known, at most the annulus's
        (``_power_not_computed``; ``None`` is returned and the caller retires the rays that are cut); a beam whose power is
        already not known raises nothing more. A probe ends there (``ANNULAR_BEAM_NOT_FOLLOWED`` is returned): its light of
        unknown fate is booked whole."""
        if pk.power_bound_W is not None and not self.bounding:
            return None
        location = f"surface:{surf.surface_id}" if surf is not None else f"path:{pk.path_id}"
        power = pk.chief_power()
        self._issue(make_issue(
            ErrorCode.ANNULAR_BEAM_NOT_FOLLOWED, Status.OUT_OF_MODEL_SCOPE,
            f"the power of the annular beam {pk.path_id} behind a central obstruction is not computed from here on, at most "
            f"{power:.4g} W: {why}", location=location, path_id=pk.path_id, power_upper_bound_W=power, untraced_power_W=power, **ctx,
            suggested_action="enlarge the clear apertures behind the obstruction so the whole annulus passes, or read the "
                             "power at a plane before this surface"))
        if self.bounding:
            self._end_event(pk, "annulus_not_followed")
            pk.alive[:] = False
            return ANNULUS_NOT_FOLLOWED
        self._power_not_computed(pk, surf, ANNULUS_NOT_FOLLOWED)
        return None

    def _local_curvature(self, surf: SurfaceG, h: Hits, d_in: Array, convention: str) -> LocalCurvature | None:
        """Principal curvatures of a sag surface at the chief-ray hit, signed for the Gaussian interface step:
        ``convention`` "refract": positive when the centre of curvature lies ahead along the ray; "mirror" (reflection, TIR):
        positive when concave (centre on the incident side). ``None`` for planes and spheres; a planar patch of a facet surface
        that approximates a curved one (``TESSELLATED_FROM_CAD``) has no curvature the Gaussian model can use."""
        if surf.facet is not None and surf.facet.tessellated:
            chord = surf.facet.chord_deviation_mm
            return LocalCurvature(0.0, 0.0, surf.u_axis, surf.v_axis, undefined=(
                ErrorCode.GAUSSIAN_UNDEFINED_AFTER_FACETS,
                f"the beam meets {surf.surface_id}, a flat facet of {surf.facet.mesh_id}, whose facets approximate a curved surface "
                f"(TESSELLATED_FROM_CAD, chord deviation {chord:g} mm): flat facets carry none of its curvature, so behind them the "
                "Gaussian beam metrics are out of model scope (rays and powers are traced on the facets)"))
        if surf.geometry != "sag":
            return None
        if surf.linear != 0.0:  # a cone has no curvature the Gaussian model can use (its apex is a point, its sides a cone)
            return LocalCurvature(0.0, 0.0, surf.u_axis, surf.v_axis, undefined=(
                ErrorCode.GAUSSIAN_UNDEFINED_AFTER_AXICON,
                f"the beam meets the axicon (cone) surface {surf.surface_id}: behind it the light forms a Bessel-like zone, not a "
                "Gaussian beam, so its Gaussian beam metrics are out of model scope (rays and powers are traced)"))
        geo = principal_curvatures(surf, h.point[0])
        if geo is None:  # pragma: no cover - a hit always lies inside the profile's domain
            return LocalCurvature(math.inf, math.inf, surf.u_axis, surf.v_axis)
        toward_positive = float(dot(d_in, geo.normal)) > 0.0  # the ray continues toward the surface's +normal side
        sign = 1.0 if toward_positive == (convention == "refract") else -1.0
        return LocalCurvature(sign * geo.k1, sign * geo.k2, geo.dir1, geo.dir2)

    def _record_interaction_loss(self, pk: _Packet, surf: SurfaceG, kind: str, p_before: float) -> None:
        """Ledger of one interaction: the power it removed from this path beyond what it handed to child paths, split into
        the parts the handler attributed (``self._split``) and the rest (``INTERACTION_LOSS_SINKS`` of its kind)."""
        loss = p_before - pk.chief_power() - self._branch_power
        for sink, power in self._split.items():
            if power > 0.0:
                self._flow(pk, sink, power, surf, self._split_detail or kind)
                loss -= power
        if abs(loss) > self.cfg.energy_conservation_tol * p_before and kind in INTERACTION_LOSS_SINKS:
            self._flow(pk, INTERACTION_LOSS_SINKS[kind], loss, surf, kind)

    def _report_clip(self, pk: _Packet, surf: SurfaceG, h: Hits, chief_cos: float, *,
                     promoted: dict[str, Any] | None = None) -> None:
        """``CLIPPED``: the chief ray falls outside the clear aperture. ``promoted``: the beam goes on as its pupil samples
        that pass (``_promote_reference``; the numbers: how many pass, the reference ray, the power that passes)."""
        ap = surf.aperture
        aoi = float(incidence_deg(chief_cos, pk.d[0], self._chief_plane_normal(surf, h)))
        u0, v0 = float(h.u[0]), float(h.v[0])
        ctx: dict[str, Any] = {
            "position_mm": [float(x) for x in h.point[0]], "aoi_deg": aoi, "aperture_shape": ap.shape,
            "surface_uv_mm": [u0, v0], "path_id": pk.path_id,
        }
        stated, required = apertures.clip_context(ap, u0, v0, chief_cos)
        ctx.update(stated)
        if pk.gstate.valid:
            ctx.update(beam_radius_x_mm=pk.gstate.radius("x"), beam_radius_y_mm=pk.gstate.radius("y"))
        ctx["suggested_min_aperture_radius_mm"] = required + (max(pk.gstate.radius("x"), pk.gstate.radius("y")) if pk.gstate.valid else 0.0)
        message = "chief ray falls outside the clear aperture: beam blocked"
        if promoted is not None:
            ctx.update(promoted)
            message = (f"chief ray falls outside the clear aperture: {promoted['pupil_rays_passing']} of its "
                       f"{promoted['pupil_rays_live']} pupil rays pass ("
                       + (f"{promoted['transmitted_power_W']:.4g} W" if "transmitted_power_W" in promoted
                          else f"power not computed, at most {promoted['power_upper_bound_W']:.4g} W")
                       + "); the beam goes on as "
                       "them, the one nearest their centroid its reference ray from here (the position, direction, angles of "
                       "incidence and optical path reported behind this surface are that ray's), its Gaussian beam metrics not "
                       "available")
        self._issue(make_issue(
            # (a physical outcome the calculation completes: the beam is stopped there, a target on what it reaches decides)
            ErrorCode.CLIPPED, Status.WARNING, message,
            location=f"surface:{surf.surface_id}", suggested_action="move the element or enlarge the clear aperture", **ctx,
        ))

    # --------------------------------------------------------------------------- AOI helpers
    def _aoi_guard(self, pk: _Packet, surf: SurfaceG, aoi: Array) -> str | None:
        """Grazing checks on the angles of incidence ``aoi`` (degrees, ``incidence_deg``). Chief at/above the fail angle terminates;
        pupil rays there are retired."""
        cfg = self.cfg
        pk.alive = pk.alive & (aoi < cfg.grazing_fail_aoi_deg)
        if aoi[0] >= cfg.grazing_fail_aoi_deg:
            self._issue(make_issue(ErrorCode.GRAZING_INCIDENCE, Status.OUT_OF_MODEL_SCOPE,
                                   f"beam {pk.path_id} meets {surf.surface_id} at {float(aoi[0]):.6g} deg, beyond the {cfg.grazing_fail_aoi_deg:g} deg "
                                   "the interface model evaluates: its light behind the surface is not traced",
                                   location=f"surface:{surf.surface_id}", aoi_deg=float(aoi[0]), limit_deg=cfg.grazing_fail_aoi_deg,
                                   path_id=pk.path_id))
            self._end_event(pk, "grazing", float(aoi[0]))
            pk.alive[:] = False
            return GRAZING
        if aoi[0] >= cfg.grazing_warn_aoi_deg:
            self._issue(make_issue(ErrorCode.GRAZING_INCIDENCE, Status.WARNING, "grazing incidence: Fresnel and aperture results are sensitive",
                                   location=f"surface:{surf.surface_id}", aoi_deg=float(aoi[0]), warn_deg=cfg.grazing_warn_aoi_deg))
        return None

    def _side(self, pk: _Packet, surf: SurfaceG, h: Hits) -> tuple[bool, NDArray[np.bool_]]:
        """(chief arrives from positive side, per-ray mask of rays on the same side as the chief)."""
        dn = dot(pk.d, h.normal)
        from_pos = bool(dn[0] < 0.0)
        return from_pos, (dn < 0.0) == from_pos

    def _live_coating(self, pk: _Packet, surf: SurfaceG, coating: CoatingEvaluator, aoi_deg: Array) -> CoatingCoefficients | None:
        """Coating coefficients of the rays this packet still follows.

        Only live rays are judged: a retired pupil ray's row holds whatever its last (possibly missed) intersection left, so
        it is evaluated at the chief ray's AOI instead.

        * a live pupil ray outside the coating's angle range while the chief ray is inside it goes on with the coefficient
          held at the range's edge (``CoatingEvaluator.coefficients_held``; one ``COATING_OUT_OF_RANGE`` warning per
          surface with the count). The beam's power is its chief ray's, so no power uses a held value; a beam on the
          geometric pupil-sample basis (no valid Gaussian state), whose power is its samples' share, and a probe retire such
          rays;
        * a chief ray outside the range goes on with the coating's values at the nearest stated wavelength and angle
          (``CoatingCoefficients.held``): where the light goes does not depend on them, its power does, so the surface
          joins ``power_scope`` and ``held`` (every power behind it is an estimate that names the surface and the data's
          range) and a ``COATING_OUT_OF_RANGE`` notice is recorded. A trace that keeps no such mark per ray
          (``hold_coating_data`` false: sampled rays, probes, analyses) ends the path there instead (``None``: the issue is
          recorded and every ray retired).

        ``aoi_deg``: the rays' angles of incidence from ``incidence_deg`` (``+ - x /`` and sqrt only: the same table value in every
        runtime).
        """
        location = f"surface:{surf.surface_id}"
        aoi = np.where(pk.alive, aoi_deg, aoi_deg[0])
        ok = coating.in_range(pk.wavelength_nm, aoi)
        if not bool(ok[0]):
            if not self.hold_coating_data:
                chief = coating.coefficients(pk.wavelength_nm, aoi[:1], location=location)
                if chief.issue is not None:
                    self._issue(chief.issue)
                pk.alive[:] = False
                return None
            values = coating.coefficients(pk.wavelength_nm, aoi, location=location)
            self._hold_coating(pk, surf, values)
            return values
        outside = pk.alive & ~ok
        if not bool(outside.any()):
            return coating.coefficients(pk.wavelength_nm, aoi, location=location)
        if not pk.gstate.valid or self.bounding:
            pk.alive = pk.alive & ok
            return coating.coefficients(pk.wavelength_nm, np.where(pk.alive, aoi, aoi[0]), location=location)
        lo, hi = coating.aoi_range()
        self._issue(make_issue(
            ErrorCode.COATING_OUT_OF_RANGE, Status.WARNING,
            f"{int(outside.sum())} pupil rays meet {surf.surface_id} outside the coating's angles {lo:g}-{hi:g} deg; they go on "
            "with the coefficient at the nearest stated angle", location=location, path_id=pk.path_id, wavelength_nm=pk.wavelength_nm,
            pupil_rays_held=int(outside.sum()), aoi_max_deg=float(np.max(aoi[outside])), aoi_min_deg=float(np.min(aoi[outside])),
            valid_aoi_deg=[lo, hi], suggested_action="state the coating's data over the angles the beam meets it at"))
        return coating.coefficients_held(pk.wavelength_nm, aoi, location=location)

    def _pass_not_computed(self, pk: _Packet, surf: SurfaceG, event: str, aoi_deg: float, detail: str) -> None:
        """A thin element (retarder, Faraday rotator, attenuator) whose response at the packet's line the model does not
        state: it does not deflect the light, which passes straight on with a power and polarization that are not computed
        (``_power_not_computed``, at most what arrived), and the pass's group delay is recorded as not stated. The caller
        has recorded the issue."""
        self._power_not_computed(pk, surf, detail)
        pk.gstate = apply_flat(pk.gstate, pk.d[0], pk.d[0], self.cfg)
        self._end_event(pk, event, aoi_deg)
        pk.lineage.append(f"{surf.surface_id}:{event}")
        pk.delays = (*pk.delays, self._delay(pk, surf, event, aoi_deg, float(dot(pk.d[0], surf.normal)) < 0.0))

    def _hold_coating(self, pk: _Packet, surf: SurfaceG, values: CoatingCoefficients) -> None:
        """The chief ray met ``surf``'s coating outside its data and goes on with the values held at the nearest stated point
        (``_live_coating``): the surface joins the packet's ``power_scope`` and ``held``, once, with a notice that names the
        path, the wavelength and angle asked for, the data's range and the point the values were read at."""
        point = values.held
        if point is None or any(item.surface_id == surf.surface_id for item in pk.held):
            return
        parameters = []
        if point.wavelength_nm != point.asked_wavelength_nm:
            parameters.append(("wavelength_nm", "nm", point.asked_wavelength_nm, point.wavelength_nm, *point.valid_wavelength_nm))
        if point.aoi_deg != point.asked_aoi_deg:
            parameters.append(("aoi_deg", "deg", point.asked_aoi_deg, point.aoi_deg, *point.valid_aoi_deg))
        pk.held = (*pk.held, HeldData(surf.surface_id, ErrorCode.COATING_OUT_OF_RANGE.value, tuple(parameters)))
        if surf.surface_id not in pk.power_scope:
            pk.power_scope = (*pk.power_scope, surf.surface_id)
        asked = " and ".join(f"{value:.6g} {unit}" for _name, unit, value, *_rest in parameters)
        read_at = " and ".join(f"{value:.6g} {unit}" for _name, unit, _asked, value, *_rest in parameters)
        self._issue(notice(
            ErrorCode.COATING_OUT_OF_RANGE,
            f"the coating's data do not cover {asked}: the beam goes on with the coating's values at {read_at}, and its power "
            "behind this surface is an estimate that names the surface",
            location=f"surface:{surf.surface_id}", path_id=pk.path_id, wavelength_nm=point.asked_wavelength_nm,
            aoi_deg=point.asked_aoi_deg, valid_wavelength_nm=list(point.valid_wavelength_nm), valid_aoi_deg=list(point.valid_aoi_deg),
            held_wavelength_nm=point.wavelength_nm, held_aoi_deg=point.aoi_deg,
            suggested_action="restrict the operating wavelength/AOI or provide a coating table covering them"))

    @staticmethod
    def _delay_stated(inter: InteractionG, which: str) -> bool:
        """Whether the group delay of the interaction's ``which`` pass ("reflection" / "transmission") is stated: a bare
        interface or an uncoated mirror, or stated constant phases (``optics/interference/coherence.py``)."""
        if inter.kind in ("refract", "mirror") and inter.coating is None:
            return True
        if inter.kind not in ("refract", "mirror", "beam_splitter") or inter.phases is None:
            return False
        model = inter.phases.model
        if model.wavelength_nm is not None:
            return False
        return model.states_reflection if which == "reflection" else model.states_transmission

    def _delay(self, pk: _Packet, surf: SurfaceG, event: str, aoi_deg: float, from_positive: bool,
               at: tuple[float, float] | None = None, state: str | None = None, inter: InteractionG | None = None) -> str:
        """The record of a pass of ``pk`` whose group delay the model does not state (``delay_key``); ``state``: the
        packet's polarization state there where that delay may depend on it (``_sp_state``, ``_axis_state``); ``inter``:
        the coated interface of a coating pass, whose group-delay bound the record then carries (none beyond the critical
        angle, where the coating's data do not hold)."""
        bound, stated = None, False
        if inter is not None and inter.coating is not None and event != "coated_tir":
            bound = inter.coating.group_delay_bound_fs
            # (a stated phase table gives the phase at the line, not its derivative: the record says so)
            model = inter.phases.model if inter.phases is not None and inter.phases.covers(pk.wavelength_nm) else None
            if model is not None:
                stated = model.states_reflection if event in REFLECTION_EVENTS else model.states_transmission
        return delay_key(surf.surface_id, event, from_positive, aoi_deg, at, state, bound, stated)

    @staticmethod
    def _sp_state(pk: _Packet, s: Array, aoi_deg: float) -> str | None:
        """The chief ray's polarization state as it meets a coating, in the s / p axes of its plane of incidence
        (``coherence.polarization_state``): a coating's s and p delays may differ. ``None`` at normal incidence (to the
        record's 1e-6 deg), where a coating treats every state alike."""
        if not aoi_deg >= NORMAL_INCIDENCE_RECORD_DEG:
            return None
        j_s, j_p = to_sp(pk.jones[:1], pk.e1[:1], pk.e2[:1], s[:1], pk.d[:1])
        return polarization_state(complex(j_s[0]), complex(j_p[0]))

    @staticmethod
    def _axis_state(pk: _Packet, a1: Array, a2: Array) -> str:
        """The chief ray's polarization state as it meets a retarder or polarizer, along its axis ``a`` and ``b = d x a``
        (``optics/polarization``: ``retarder``, ``polarizer``)."""
        j0, j1 = complex(pk.jones[0, 0]), complex(pk.jones[0, 1])
        x1, x2 = float(a1[0]), float(a2[0])
        return polarization_state(x1 * j0 + x2 * j1, -x2 * j0 + x1 * j1)

    def _stated_phases(self, pk: _Packet, surf: SurfaceG, inter: InteractionG) -> PhaseValues | None:
        """The interface's stated phases at the packet's wavelength (``None`` when it states none). A phase table that does
        not cover the wavelength states no phase there: the interface's own phases apply, as for a coating without stated
        phases, the pass is recorded as one whose phase is not stated (``_delay``), and the ``COATING_OUT_OF_RANGE`` issue
        names the path."""
        if inter.phases is None:
            return None
        location = f"surface:{surf.surface_id}"
        values, issue = inter.phases.at(pk.wavelength_nm, location=location)
        if issue is not None:
            self._issue(issue.model_copy(update={"numerical_context": {**issue.numerical_context, "path_id": pk.path_id}}))
        return values

    # --------------------------------------------------------------------------- interactions
    def _do_refract(self, pk: _Packet, surf: SurfaceG, h: Hits, inter: InteractionG) -> str | None:
        cfg = self.cfg
        from_pos, same_side = self._side(pk, surf, h)
        pk.alive = pk.alive & same_side
        med_from, med_to = (surf.medium_pos, surf.medium_neg) if from_pos else (surf.medium_neg, surf.medium_pos)
        if pk.medium != med_from:
            self._issue(make_issue(ErrorCode.MEDIUM_MISMATCH, Status.FAIL,
                                   f"ray is in medium {pk.medium!r} but the surface expects {med_from!r} on this side",
                                   location=f"surface:{surf.surface_id}", tracked_medium=pk.medium, expected_medium=med_from,
                                   suggested_action="fix the media on the two sides of the surfaces (overlapping or missing solids)"))
            self._end_event(pk, "medium_mismatch")
            pk.alive[:] = False
            return MEDIUM_MISMATCH
        if pk.crystal is not None or self._crystal_face(surf):
            return self._do_crystal_face(pk, surf, h, inter, from_pos, med_to)
        n1 = pk.n
        n2, issue = self.library.evaluator(med_to).index(pk.wavelength_nm, location=f"surface:{surf.surface_id}")
        if issue is not None or n2 is None:
            if issue is not None:
                self._issue(issue)
            pk.alive[:] = False
            return MATERIAL_RANGE
        # (a gradient-index medium on a side: each ray's index where it meets the face; n1c / n2c: the chief ray's)
        n1 = self._gradient_rays(med_from, pk, h.point, n1)
        n2 = self._gradient_rays(med_to, pk, h.point, n2)
        n1c, n2c = (float(n1[0]) if np.ndim(n1) else n1), (float(n2[0]) if np.ndim(n2) else n2)
        res = refract(pk.d, h.normal, n1, n2, cfg.zero_vector_tol)
        aoi_deg = incidence_deg(res.cos_i, pk.d, h.normal)
        term = self._aoi_guard(pk, surf, aoi_deg)
        if term:
            return term
        pk.alive = pk.alive & (res.tir == bool(res.tir[0]))
        chief_tir = bool(res.tir[0])
        if abs(float(res.critical_margin[0])) < cfg.tir_warn_margin * 1.0 and n1c > n2c:
            self._issue(make_issue(ErrorCode.TIR_BOUNDARY, Status.WARNING, "incidence is within the TIR warning margin of the critical angle",
                                   location=f"surface:{surf.surface_id}", critical_margin=float(res.critical_margin[0])))
        fr = fresnel(n1, n2, np.maximum(res.cos_i, np.finfo(float).tiny))
        tau_s, tau_p, r_s, r_p = fr.tau_s, fr.tau_p, fr.r_s, fr.r_p
        if inter.coating is not None and not chief_tir:
            cc = self._live_coating(pk, surf, inter.coating, aoi_deg)
            if cc is None:
                return COATING_RANGE
            ph = self._stated_phases(pk, surf, inter)
            # the coating's magnitudes; its stated phases (when given) instead of the bare Fresnel ones
            tau_s, tau_p, r_s, r_p = coated_interface(cc.rs, cc.rp, cc.ts, cc.tp, tau_s, tau_p, r_s, r_p, ph)
            self._twins_polarized(pk, surf, max(abs(float(cc.rs[0]) - float(cc.rp[0])), abs(float(cc.ts[0]) - float(cc.tp[0]))),
                                  "a coated interface")
        elif not chief_tir:
            self._twins_polarized(pk, surf, max(abs(float(fr.R_s[0]) - float(fr.R_p[0])), abs(float(fr.T_s[0]) - float(fr.T_p[0]))),
                                  "an interface met at an angle")
        s = s_axis(pk.d, res.n_oriented, pk.e1, cfg.zero_vector_tol)
        # (the polarization state the coating meets, where its delay is not stated: s and p may be delayed differently)
        state = self._sp_state(pk, s, float(aoi_deg[0])) if inter.coating is not None else None
        radius_signed: float | None = None
        if surf.geometry == "sphere":
            ahead = float(dot((surf.center - h.point[0]), pk.d[0])) > 0.0
            radius_signed = surf.radius if ahead else -surf.radius
        d_in = pk.d.copy()
        curv_refract = self._local_curvature(surf, h, d_in[0], "refract")
        curv_reflect = self._local_curvature(surf, h, d_in[0], "mirror")
        # Fresnel reflection branch (optional)
        child_state: tuple[Array, CArray, Array, Array] | None = None
        if inter.trace_reflection and not chief_tir:
            d_ref, n_or, cos_i = reflect_dir(d_in, h.normal, cfg.zero_vector_tol)
            j_ref, e1_ref, e2_ref = apply_interface(pk.jones, pk.e1, pk.e2, d_in, s, r_s, r_p, d_ref)
            child_state = (d_ref, j_ref, e1_ref, e2_ref)
        if chief_tir:
            coef_s, coef_p = r_s, r_p
        else:
            coef_s, coef_p = tau_s, tau_p
        j_new, e1_new, e2_new = apply_interface(pk.jones, pk.e1, pk.e2, d_in, s, coef_s, coef_p, res.d_out)
        if not chief_tir:
            # the power the interface reflects (traced as a child path below, or recorded as not followed)
            d_ref0, _, _ = reflect_dir(d_in[:1], h.normal[:1], cfg.zero_vector_tol)
            j_ref0, e1_ref0, e2_ref0 = apply_interface(pk.jones[:1], pk.e1[:1], pk.e2[:1], d_in[:1], s[:1], r_s[:1], r_p[:1], d_ref0)
            reflected_power = pk.power0_W * power2(j_ref0[0])
            if child_state is None:
                self._split[SINK_UNTRACED_REFLECTION] = reflected_power
                self._split_detail = "interface reflection (not followed)"
                if self.keep_faint or not self._negligible(pk, reflected_power):  # (where it goes decides whether a dark plane is dark)
                    conc = None if radius_signed is None else (-abs(radius_signed) if radius_signed > 0 else abs(radius_signed))
                    g_ref = apply_curved_or_flat_interface(
                        pk.gstate, kind="mirror", d_in=d_in[0], d_out=d_ref0[0], n_or=res.n_oriented[0], cos_i=float(res.cos_i[0]),
                        n1=n1c, n2=n1c, cos_t=float(res.cos_i[0]), tir=False, radius_signed_mm=conc, cfg=cfg, curvature=curv_reflect)

                    def reflected_rays(d_in: Array = d_in, s: Array = s, r_s: CArray = r_s, r_p: CArray = r_p,
                                       jones: CArray = pk.jones, e1: Array = pk.e1, e2: Array = pk.e2) -> tuple[Array, Array, Array, CArray]:
                        d_all, _, _ = reflect_dir(d_in, h.normal, cfg.zero_vector_tol)
                        j_all, e1_all, e2_all = apply_interface(jones, e1, e2, d_in, s, r_s, r_p, d_all)
                        return d_all, e1_all, e2_all, j_all

                    self._branch(pk, SINK_UNTRACED_REFLECTION, surf, d_ref0[0], g_ref, reflected_power,
                                 jones=j_ref0[0], e1=e1_ref0[0], e2=e2_ref0[0], rays=reflected_rays,
                                 delay=None if self._delay_stated(inter, "reflection") else
                                 self._delay(pk, surf, "coating_reflection", float(aoi_deg[0]), from_pos, state=state,
                                             inter=inter))
                else:
                    self._leave_out(pk, reflected_power)
        if child_state is not None:
            # the reflected branch of a transmitting interface: bare Fresnel, or the published coating's reflectance
            branch_event = "fresnel_reflection" if inter.coating is None else "coating_reflection"
            child = self._spawn(pk, surf, branch_event, reflected_power)
            if child is None:
                self._branch_power += reflected_power
            else:
                c_d, c_j, c_e1, c_e2 = child_state
                child.d, child.jones, child.e1, child.e2 = c_d, c_j, c_e1, c_e2
                conc = None
                if radius_signed is not None:
                    conc = -abs(radius_signed) if radius_signed > 0 else abs(radius_signed)  # convex from the incident side if centre is ahead
                child.gstate = apply_curved_or_flat_interface(
                    pk.gstate, kind="mirror", d_in=d_in[0], d_out=c_d[0], n_or=res.n_oriented[0], cos_i=float(res.cos_i[0]),
                    n1=n1c, n2=n1c, cos_t=float(res.cos_i[0]), tir=False, radius_signed_mm=conc, cfg=cfg, curvature=curv_reflect)
                child.lineage.append(f"{surf.surface_id}:{branch_event}")
                if not self._delay_stated(inter, "reflection"):
                    child.delays = (*child.delays, self._delay(pk, surf, branch_event, float(aoi_deg[0]), from_pos, state=state,
                                                               inter=inter))
                if self.recorder is not None:
                    self.recorder.branch_spawned(pk, child)
                child.alive = child.alive.copy()
                self._branch_power += child.chief_power()
                if not self._spent(child):
                    self._enqueue(child)
                else:
                    self._finish(child, ZERO_POWER, surf.surface_id)
        cos_t0 = math.sqrt(max(0.0, 1.0 - float(res.sin_t_sq[0])))
        if chief_tir:
            conc = None if radius_signed is None else (-abs(radius_signed) if radius_signed > 0 else abs(radius_signed))
            pk.gstate = apply_curved_or_flat_interface(pk.gstate, kind="mirror", d_in=d_in[0], d_out=res.d_out[0],
                                                       n_or=res.n_oriented[0], cos_i=float(res.cos_i[0]), n1=n1c, n2=n1c,
                                                       cos_t=float(res.cos_i[0]), tir=True, radius_signed_mm=conc, cfg=cfg,
                                                       curvature=curv_reflect)
            event = "tir"
            if inter.coating is not None and not self._delay_stated(inter, "reflection"):
                # (a coated face beyond the critical angle: the coating's total reflection has a delay the design does not
                # state -- not the bare interface's)
                pk.delays = (*pk.delays, self._delay(pk, surf, "coated_tir", float(aoi_deg[0]), from_pos, state=state))
        else:
            pk.gstate = apply_curved_or_flat_interface(pk.gstate, kind="refract", d_in=d_in[0], d_out=res.d_out[0],
                                                       n_or=res.n_oriented[0], cos_i=float(res.cos_i[0]), n1=n1c, n2=n2c,
                                                       cos_t=cos_t0, tir=False, radius_signed_mm=radius_signed, cfg=cfg,
                                                       curvature=curv_refract)
            pk.medium, pk.n = med_to, n2c
            event = "refract"
            if not self._delay_stated(inter, "transmission"):
                pk.delays = (*pk.delays, self._delay(pk, surf, event, float(aoi_deg[0]), from_pos, state=state, inter=inter))
        self._report_gaussian_state(pk, surf)
        pk.jones, pk.e1, pk.e2, pk.d = j_new, e1_new, e2_new, np.where(pk.alive[:, None], res.d_out, pk.d)
        self._end_event(pk, event, float(aoi_deg[0]))
        pk.lineage.append(f"{surf.surface_id}:{event}")
        if self._spent(pk):
            return ZERO_POWER
        return None

    # ---------------------------------------------------------------------------- radial gradient-index media
    def _gradient_rays(self, medium: str, pk: _Packet, points: Array, n: float) -> Array | float:
        """Each ray's index at ``points`` when ``medium`` is a radial gradient-index medium (``n0 sqrt(1 - g^2 r^2)``, ``n0`` its
        index on the axis), else ``n``."""
        grin = self._gradients.get(medium)
        if grin is None:
            return n
        laws = self.library.gradient(medium)
        assert laws is not None
        n0, _issue = laws.base_index.index(pk.wavelength_nm)
        n = n if n0 is None else n0  # (outside its band: the caller has already stopped on the medium's evaluator)
        g, _dg = laws.gradient(pk.wavelength_nm)
        rel = np.where(np.isfinite(points), points, 0.0) - grin.origin
        if laws.axial is not None:  # (an axial medium: its index at each point's position along the axis)
            return gradient.AxialProfile(n, laws.axial).index_z(dot(rel, grin.axis))
        r2 = dot(rel, grin.ex) ** 2 + dot(rel, grin.ey) ** 2
        return n * np.sqrt(np.maximum(0.0, 1.0 - (g * g) * r2))

    def _gradient_profile(self, pk: _Packet) -> tuple[gradient.Profile | gradient.AxialProfile | None, ValidationIssue | None]:
        """The packet's gradient-index medium at its wavelength (``gradient.Profile``, an axial one ``gradient.AxialProfile``), or
        the material issue outside its band."""
        laws = self.library.gradient(pk.medium)
        assert laws is not None
        n0, issue = laws.base_index.index(pk.wavelength_nm, location=f"material:{pk.medium}")
        if n0 is None:
            return None, issue
        derivatives = self.library.gradient_derivatives(laws, pk.wavelength_nm, n0)
        if laws.axial is not None:
            if derivatives is None:
                return gradient.AxialProfile(n0, laws.axial), None
            return gradient.AxialProfile(n0, laws.axial, derivatives[0], derivatives[1]), None
        g, dg = laws.gradient(pk.wavelength_nm)
        if derivatives is None:
            return gradient.Profile(n0, g, dg=dg), None
        return gradient.Profile(n0, g, derivatives[0], derivatives[1], dg), None

    def _gradient_refusal(self, pk: _Packet, message: str, location: str, **context: Any) -> bool:
        """The path ends out of model scope in a gradient-index medium (``GRADIENT_INDEX_UNSUPPORTED``)."""
        self._issue(make_issue(ErrorCode.GRADIENT_INDEX_UNSUPPORTED, Status.OUT_OF_MODEL_SCOPE, message, location=location,
                               path_id=pk.path_id, material=pk.medium, **context))
        self._end_event(pk, "gradient_index_unsupported")
        pk.alive[:] = False
        self._finish(pk, ErrorCode.GRADIENT_INDEX_UNSUPPORTED.value, None)
        return True

    def _gradient_leg(self, pk: _Packet) -> bool:
        """Light in a radial gradient-index medium (``optics/rays/gradient.py``): every live ray goes in closed form to the end
        face ahead along the rod's axis -- its position, direction, optical path and length there, its polarization carried
        along the ray -- and meets that face (``_interact``). A pupil ray that reaches the rod's side or another surface on
        the way is retired; the chief ray doing so ends the path out of model scope. The Gaussian beam takes the leg's
        first-order transfer about the chief ray (``_gradient_gaussian``). Returns whether the path ended."""
        cfg, scene = self.cfg, self.scene
        grin = self._gradients[pk.medium]
        profile, issue = self._gradient_profile(pk)
        if profile is None:
            if issue is not None:
                self._issue(issue)
            pk.alive[:] = False
            self._finish(pk, MATERIAL_RANGE, None)
            return True
        rel = pk.o - grin.origin
        d_local = np.stack([dot(pk.d, grin.ex), dot(pk.d, grin.ey), dot(pk.d, grin.axis)], axis=-1)
        leg, along = gradient.legs(profile, dot(rel, grin.ex), dot(rel, grin.ey), dot(rel, grin.axis), d_local, grin.z_lo, grin.z_hi)
        chief_sign = float(leg.sign[0])
        if isinstance(profile, gradient.AxialProfile):
            least = self._axial_least(profile, grin)
            if least < 1.0:
                return self._gradient_refusal(
                    pk, f"at {pk.wavelength_nm:.6g} nm the index of the axial gradient-index medium {pk.medium} falls to {least:.6g} "
                    "within the rod: below 1 the medium is outside the model", f"material:{pk.medium}", least_index=least)
        # (rays in the rod's frame that do not reach the end face ahead: too oblique to a slanted or curved face to cross it once,
        # or turning back along the axis of an axial medium)
        odd = np.zeros(along.shape, dtype=bool)
        with np.errstate(invalid="ignore", divide="ignore"):
            if grin.faces is not None:
                leg, reached = gradient.to_face(leg, grin.faces[1] if chief_sign > 0.0 else grin.faces[0])
                odd = odd | ~reached
            odd = odd | leg.turns(profile)
            reach = leg.largest_r2()
        side = ~along | (reach > grin.radius * grin.radius) | (leg.sign != chief_sign)
        if bool(odd[0]):
            return self._gradient_refusal(
                pk, f"in the gradient-index medium {pk.medium} the chief ray "
                + ("turns back along the rod's axis or meets its end face too obliquely to cross it once (its slope across the "
                   "axis times the face's largest slope reaches 1)" if grin.faces is not None else
                   f"comes within {gradient.AXIAL_MIN_COS:.3g} of the cosine 0 of its angle to the rod's axis, near where it turns "
                   "back") + ": such light is not followed", f"material:{pk.medium}", radius_mm=grin.radius)
        if bool(side[0]):
            return self._gradient_refusal(
                pk, f"in the gradient-index medium {pk.medium} the chief ray reaches the rod's side ({math.sqrt(float(reach[0])):.6g} mm from "
                f"its axis, radius {grin.radius:.6g} mm): light leaving a rod through its side is not followed",
                f"material:{pk.medium}", radius_mm=grin.radius)
        side = side | odd
        retired = pk.alive & side
        # the leg as points along each ray (scene frame), for the polarization and the check against the scene's other surfaces
        count = leg.point_count()
        local, tangents_local = leg.points(count)
        points = _to_scene(grin, local)
        tangents = _to_scene(grin, tangents_local, point=False)
        faces = (grin.face_lo, grin.face_hi)
        target = faces[1] if chief_sign > 0.0 else faces[0]
        met: dict[int, str] = {}
        starts, chords = points[:-1].reshape(-1, 3), (points[1:] - points[:-1]).reshape(-1, 3)
        length = np.sqrt(np.sum(chords * chords, axis=-1))
        unit = chords / np.where(length > 0.0, length, 1.0)[:, None]
        for si, surf in enumerate(scene.surfaces):
            if si in faces:
                continue
            t = intersect(surf, starts, unit, cfg).t
            hit = (np.isfinite(t) & (t < length)).reshape(count, -1).any(axis=0) & pk.alive
            for ray in np.flatnonzero(hit):
                met.setdefault(int(ray), surf.surface_id)
        if 0 in met:
            return self._gradient_refusal(
                pk, f"light travelling in the gradient-index medium {pk.medium} meets {met[0]} inside it: only the rod's end faces "
                "are traced in a gradient-index medium, so the light stops there", f"surface:{met[0]}", surface_id=met[0],
                suggested_action="place elements and observation planes outside the rod's volume")
        for ray in met:
            retired[ray] = True
        if self.recorder is not None and bool(retired.any()):
            rays = np.flatnonzero(retired)
            self.recorder.pupil_rays_diverged(pk, rays, tuple(met.get(int(r)) for r in rays), np.zeros(rays.size, dtype=bool))
        self._gradient_side(pk, retired, side, grin.radius, scene.surfaces[target],
                            self._gradient_envelope(pk, profile, grin, chief_sign), odd)
        pk.alive = pk.alive & ~retired
        # where every ray leaves the leg
        x, y, z, dx, dy = leg.at(leg.z_len)
        exit_local = np.stack([x, y, z], axis=-1)
        exit_dir = np.stack([dx, dy, leg.sign], axis=-1)
        exit_dir = exit_dir / np.sqrt(np.sum(exit_dir * exit_dir, axis=-1, keepdims=True))
        exit_point, exit_d = _to_scene(grin, exit_local), _to_scene(grin, exit_dir, point=False)
        opl = leg.optical_path(profile)
        lengths = leg.lengths(profile)
        if isinstance(leg, gradient.AxialLegs):  # (each ray turns in one plane: its polarization in closed form)
            e1 = gradient.planar_transport(pk.e1, pk.d, exit_d, grin.axis)
        else:
            e1 = gradient.transported(points, tangents, pk.e1)
        e2 = np.cross(exit_d, e1)
        start, power_start, w_start = pk.o[0].copy(), pk.chief_power(), _max_radius(pk)
        extinction = self._extinction(pk)
        if extinction > 0.0:
            atten = exact_exp(-0.5 * extinction * lengths)
            self._flow(pk, SINK_BULK_ABSORPTION, power_start * -float(expm1(-extinction * float(lengths[0]))),
                       scene.surfaces[target], pk.medium)
            pk.jones = pk.jones * atten[:, None]
        pk.gstate = self._gradient_gaussian(pk, profile, grin, exit_point[0], exit_d[0], points[:, 0], tangents[:, 0])
        if pk.gpl is not None and profile is not None:
            group = leg.group(profile, pk.wavelength_nm, NM3_PER_NM2_TO_MM)
            disperses = profile.dn0 != 0.0 or profile.d2n0 != 0.0 or getattr(profile, "dg", 0.0) != 0.0
            pk.gpl, pk.gdd, pk.dispersive = pk.gpl + group[0], pk.gdd + group[1], pk.dispersive or disperses
        pk.spread = None  # (the envelope delay's spread over the ray families through a gradient-index rod is not bounded)
        live = pk.alive[:, None]
        pk.o = np.where(live, exit_point, pk.o)
        pk.d = np.where(live, exit_d, pk.d)
        pk.e1, pk.e2 = np.where(live, e1, pk.e1), np.where(live, e2, pk.e2)
        pk.opl = pk.opl + np.where(pk.alive, opl, 0.0)
        pk.length = pk.length + np.where(pk.alive, lengths, 0.0)
        if isinstance(leg, gradient.AxialLegs):
            pk.n = float(leg.exit_index()[0])
        else:
            r2_exit = float(leg.r2(leg.z_len[:1])[0])
            pk.n = profile.n0 * math.sqrt(max(0.0, 1.0 - profile.g * profile.g * r2_exit))
        self._gradient_segments(pk, leg, profile, grin, start, power_start, w_start, scene.surfaces[target])
        pk.segment_mm = 0.0
        # the end face: the rays reach it where the leg ends (the face's own intersection, from just short of it)
        face = scene.surfaces[target]
        back = 2.0 * cfg.ray_min_advance_mm
        h = intersect(face, pk.o - back * pk.d, pk.d, cfg)
        missed = pk.alive & ~np.isfinite(h.t)
        if bool(missed[0]):
            return self._gradient_refusal(
                pk, f"in the gradient-index medium {pk.medium} the chief ray reaches the plane of its end face {face.surface_id} outside "
                "the face: light leaving a rod beside its end face is not followed", f"surface:{face.surface_id}")
        pk.alive = pk.alive & ~missed
        if self.recorder is not None:
            self.recorder.packet_advanced(pk, face.surface_id)
        self.out.trace_events.append({"path": pk.path_id, "surface": face.surface_id, "t": float(lengths[0])})
        if self.schedule.interaction_spent(pk.interactions):
            self._interaction_limit(pk)
            return True
        pk.leg_crossed, pk.leg_partial, pk.leg_t = (), (), 0.0
        terminated = self._interact(pk, face, h)
        if self.recorder is not None:
            self.recorder.packet_interacted(pk)
        if terminated is not None:
            self._finish(pk, terminated, face.surface_id)
            return True
        pk.passes_owner = None
        pk.interactions += 1
        return False

    def _gradient_envelope(self, pk: _Packet, profile: gradient.Profile | gradient.AxialProfile, grin: Any,
                           chief_sign: float) -> float | None:
        """``K`` (``_envelope_sigmas``) when the beam's envelope reaches the rod's side on this leg, else ``None``: the rays of
        the beam's family at ``K`` beam radii -- per Gaussian axis, from its waist (``w0``, ``qx`` / ``qy``), the rays at
        ``K w0 cos phi`` across and ``K (w0 / z_R) sin phi`` in angle, carried to where the leg starts (exact for a Gaussian
        beam), and their combinations ``cos psi`` along one axis with ``sin psi`` along the other, 16 values of each angle and
        ``K`` widened by ``1 / cos(pi / 16)^2`` over the gaps between them -- followed in closed form through the rod. Beyond
        ``K`` radii the beam's share past any line is negligible, so a beam whose envelope stays inside loses none there.
        ``None`` too without a valid Gaussian state or a stated ``K``."""
        g = pk.gstate
        k_sigma = self._envelope_sigmas(pk)
        if not g.valid or k_sigma is None or k_sigma <= 0.0 or g.qx.imag <= 0.0 or g.qy.imag <= 0.0:
            return None
        steps = GRADIENT_ENVELOPE_STEPS
        angles = np.arange(steps, dtype=np.float64) * (2.0 * math.pi / steps)
        cos_a, sin_a = cos_sin(angles)
        gap = float(cos_sin(np.array([math.pi / steps]))[0][0])
        k = k_sigma / (gap * gap)
        cos_phi, sin_phi = np.repeat(cos_a, steps), np.repeat(sin_a, steps)
        cos_psi, sin_psi = np.tile(cos_a, steps), np.tile(sin_a, steps)
        offset = np.zeros((steps * steps, 3))
        tilt = np.zeros((steps * steps, 3))
        for axis, q, share in (("x", g.qx, cos_psi), ("y", g.qy, sin_psi)):
            w0 = g.waist_radius(axis)
            theta = w0 / q.imag
            across, angle = k * w0 * cos_phi * share, k * theta * sin_phi * share
            frame = g.bx if axis == "x" else g.by
            offset = offset + (across + q.real * angle)[:, None] * frame[None, :]
            tilt = tilt + angle[:, None] * frame[None, :]
        d = pk.d[0][None, :] + tilt
        d = d / np.sqrt(np.sum(d * d, axis=-1, keepdims=True))
        start = pk.o[0][None, :] + offset
        rel = start - grin.origin
        d_local = np.stack([dot(d, grin.ex), dot(d, grin.ey), dot(d, grin.axis)], axis=-1)
        if grin.faces is None:
            z0 = dot(rel, grin.axis)
            # (each ray from where it crosses the plane of the face the leg starts at)
            z_face = grin.z_lo if chief_sign > 0.0 else grin.z_hi
            safe = np.where(np.abs(d_local[:, 2]) > 0.0, d_local[:, 2], 1.0)
            back = (z_face - z0) / safe
        else:  # (a slanted or curved entrance face: from where each crosses its tangent plane at the chief ray's start)
            normal = self._gradient_normal(grin, grin.faces[0] if chief_sign > 0.0 else grin.faces[1], pk.o[0])
            along_n = dot(d, normal)
            back = dot(pk.o[0] - start, normal) / np.where(np.abs(along_n) > 0.0, along_n, 1.0)
        rel = rel + back[:, None] * d
        leg, along = gradient.legs(profile, dot(rel, grin.ex), dot(rel, grin.ey), dot(rel, grin.axis), d_local, grin.z_lo, grin.z_hi)
        with np.errstate(invalid="ignore", divide="ignore"):
            odd = leg.turns(profile)
            if grin.faces is not None:
                leg, reached = gradient.to_face(leg, grin.faces[1] if chief_sign > 0.0 else grin.faces[0])
                odd = odd | ~reached | leg.turns(profile)
            reach = leg.largest_r2()
            out = ~along | ~np.isfinite(reach) | (reach > grin.radius * grin.radius) | (leg.sign != chief_sign) | odd
        return k_sigma if bool(out.any()) else None

    @staticmethod
    def _axial_least(profile: gradient.AxialProfile, grin: Any) -> float:
        """The least index of an axial medium over its rod (between its end faces' outermost axial positions)."""
        z_lo, z_hi = (grin.z_lo, grin.z_hi) if grin.faces is None else (grin.faces[0].z_min, grin.faces[1].z_max)
        return float(profile.least(np.array([z_lo]), np.array([z_hi]))[0])

    @staticmethod
    def _gradient_normal(grin: Any, face: GradientFaceG, point: Array) -> Array:
        """The unit normal (toward +axis) of a rod's end face at a scene point on it."""
        rel = point - grin.origin
        x, y = float(dot(rel, grin.ex)), float(dot(rel, grin.ey))
        m = gradient.face_normal(face, x, y)
        return m[0] * grin.ex + m[1] * grin.ey + m[2] * grin.axis

    def _gradient_side(self, pk: _Packet, retired: NDArray[np.bool_], side: NDArray[np.bool_], radius_mm: float,
                       face: SurfaceG, envelope: float | None = None, odd: NDArray[np.bool_] | None = None) -> None:
        """Pupil samples a gradient-index leg retires before the end face ``face`` (``retired``: they reach the rod's side --
        ``side`` -- or meet another surface inside it) while the chief ray goes on. The light they carry leaves the model there:
        what the side does with it (refraction out of the rod, or reflection back into it) is not followed. So the beam's
        power behind ``face`` is out of model scope, never a known value: ``face`` joins ``power_scope`` and an
        ``OUT_OF_MODEL_SCOPE`` issue (``GRADIENT_INDEX_UNSUPPORTED``) states the share of the pupil samples retired (equal-power
        samples: an estimate of the power share). On the geometric pupil-sample basis (no valid Gaussian state) the beam's
        power is its samples' share: theirs leaves the beam and is booked ``TERMINATED_OUT_OF_MODEL`` (its fate not
        followed), the beam goes on with the rest. With a valid Gaussian state the beam's power follows its chief ray (the
        leg's first-order transfer has no side), so it keeps its power, whose value behind ``face`` the scope withholds. A
        probe and a beam of unknown power keep their own rules."""
        if self.bounding or pk.power_bound_W is not None or not pk.pupil_total:
            return
        annulus = self._annulus_rays(pk)
        live = pk.alive & annulus
        gone = retired & annulus
        n_live, n_gone = int(live.sum()), int(gone.sum())
        if n_live == 0:
            return
        if n_gone == 0:
            if envelope is not None and face.surface_id not in pk.power_scope:
                # (no pupil sample reaches the side, but the beam's envelope does: its tail beyond the samples may leave there)
                pk.power_scope = (*pk.power_scope, face.surface_id)
                self._issue(make_issue(
                    ErrorCode.GRADIENT_INDEX_UNSUPPORTED, Status.OUT_OF_MODEL_SCOPE,
                    f"the envelope of beam {pk.path_id} ({envelope:.3g} beam radii about its chief ray, beyond which its share is "
                    f"negligible) reaches the rod's side in the gradient-index medium {pk.medium} (rod radius {radius_mm:.6g} mm), "
                    f"though none of its {n_live} pupil rays does: light leaving a rod through its side is not followed, so the "
                    f"power the beam carries behind {face.surface_id} is not computed",
                    location=f"material:{pk.medium}", path_id=pk.path_id, material=pk.medium, surface_ids=[face.surface_id],
                    next_surface_id=face.surface_id, pupil_rays_left=0, pupil_rays_at_side=0, pupil_rays_live=n_live,
                    envelope_sigmas=envelope, radius_mm=radius_mm,
                    suggested_action="aim the beam closer to the rod's axis, narrow it, or use a rod of larger radius so the beam "
                                     "stays inside the rod; or read the power at a plane before the rod"))
            return
        n_side = int((gone & side).sum())
        p_in = pk.chief_power()
        moved: dict[str, Any] = {}
        if not pk.gstate.valid and n_gone < n_live:
            n_stay = n_live - n_gone
            gone_W = p_in * (n_gone / n_live)
            boundary = int(self._boundary_cells(pk, live, gone, outer=False).sum())
            pk.jones = pk.jones.copy()
            pk.jones[0] = pk.jones[0] * math.sqrt(n_stay / n_live)
            self._flow(pk, SINK_TERMINATED_OUT_OF_MODEL, gone_W, face, "gradient_index_side")
            self._add_sampling(pk, boundary / n_stay)
            moved = {"left_power_W": gone_W}
        if face.surface_id not in pk.power_scope:
            pk.power_scope = (*pk.power_scope, face.surface_id)
        where = ("reach the rod's side" if n_side == n_gone else "meet another surface inside the rod" if n_side == 0 else
                 "reach the rod's side or meet another surface inside it")
        if odd is not None and bool((gone & odd).any()):
            where = ("reach the rod's side, meet another surface inside it, turn back along its axis or meet its end face too "
                     "obliquely to cross it once")
        booked = (f"; their share ({moved['left_power_W']:.4g} W) leaves the beam, booked as light whose fate is not followed"
                  if moved else "")
        self._issue(make_issue(
            ErrorCode.GRADIENT_INDEX_UNSUPPORTED, Status.OUT_OF_MODEL_SCOPE,
            f"{n_gone} of the {n_live} pupil rays of beam {pk.path_id} (about {n_gone / n_live:.3g} of its power by its "
            f"equal-power samples) {where} in the gradient-index medium {pk.medium} (rod radius {radius_mm:.6g} mm) while its "
            f"chief ray goes on to {face.surface_id}: light leaving a rod through its side is not followed, so the power the "
            f"beam carries behind {face.surface_id} is not computed{booked}",
            location=f"material:{pk.medium}", path_id=pk.path_id, material=pk.medium, surface_ids=[face.surface_id],
            next_surface_id=face.surface_id, pupil_rays_left=n_gone, pupil_rays_at_side=n_side, pupil_rays_live=n_live,
            left_fraction=n_gone / n_live, radius_mm=radius_mm, **moved,
            suggested_action="aim the beam closer to the rod's axis, narrow it, or use a rod of larger radius so the beam "
                             "stays inside the rod; or read the power at a plane before the rod"))

    def _gradient_segments(self, pk: _Packet, leg: gradient.Legs | gradient.AxialLegs, profile: gradient.Profile | gradient.AxialProfile,
                           grin: Any, start: Array,
                           power_start: float, w_start: float | None, face: SurfaceG) -> None:
        """The chief ray's leg as ``GRADIENT_SEGMENTS_PER_PERIOD`` straight pieces per period (at least one), each from one point
        of the curved ray to the next with that stretch's own length and optical path (closed form / quadrature, ``gradient``)."""
        if isinstance(leg, gradient.AxialLegs):
            pieces = GRADIENT_AXIAL_SEGMENTS
        else:
            periods = float(leg.w[0] * leg.z_len[0]) / gradient.TWO_PI
            pieces = max(1, min(GRADIENT_MAX_SEGMENTS, int(math.ceil(periods * GRADIENT_SEGMENTS_PER_PERIOD))))
        chief = leg.rows(0)
        bounds = np.arange(pieces + 1, dtype=np.float64) / pieces * float(chief.z_len[0])
        x, y, z, _dx, _dy = chief.at(bounds)
        points = _to_scene(grin, np.stack([x, y, z], axis=-1))
        points[0] = start
        power_end = pk.chief_power()
        for k in range(pieces):
            part = chief.between(float(bounds[k]), float(bounds[k + 1]))
            length = float(part.lengths(profile)[0])
            opl = float(part.optical_path(profile)[0])
            last = k == pieces - 1
            pk.segments.append(RaySegment(
                start_mm=tuple(float(v) for v in points[k]), end_mm=tuple(float(v) for v in points[k + 1]),  # type: ignore[arg-type]
                length_mm=length, medium_id=pk.medium, refractive_index=opl / length if length > 0.0 else profile.n0,
                optical_path_length_mm=opl, power_start_W=power_start if pk.power_bound_W is None else None,
                power_end_W=power_end if pk.power_bound_W is None else None, end_surface_id=face.surface_id if last else None,
                beam_radius_start_mm=w_start if k == 0 else None, beam_radius_end_mm=_max_radius(pk) if last else None))

    def _gradient_gaussian(self, pk: _Packet, profile: gradient.Profile | gradient.AxialProfile, grin: Any, exit_point: Array,
                           exit_d: Array, path: Array, tangents: Array) -> GState:
        """The Gaussian beam behind a gradient-index leg: its first-order transfer about the chief ray, from central differences
        of the closed-form leg (``GRADIENT_JACOBIAN_STEP``), in reduced coordinates (``x``, ``n theta``) from the beam's axes at the
        leg's start to those axes carried along the chief ray: ``q'/n_out = (A q/n_in + B / n_in) / (n_out C q/n_in + n_out D /
        n_in)`` per axis. A transfer that couples the axes (a skew chief ray) leaves the x/y-independent model unless the beam
        is round and the chief ray's meridional plane decouples it (``GAUSSIAN_TWISTED_ASTIGMATISM``)."""
        g0 = pk.gstate
        if not g0.valid:
            return g0
        d_in, p_in = pk.d[0], pk.o[0]
        axis = grin.axis
        axial = isinstance(profile, gradient.AxialProfile)

        def carried(v: Array) -> Array:
            if axial:
                return gradient.planar_transport(v[None, :], d_in[None, :], exit_d[None, :], axis)[0]
            return gradient.transported(path[:, None, :], tangents[:, None, :], v[None, :])[0]

        bx_out = carried(g0.bx)
        transfer = self._gradient_transfer(profile, grin, p_in, d_in, g0.bx, g0.by, exit_point, exit_d, bx_out)
        if transfer is None:
            return g0.invalidated(ErrorCode.GAUSSIAN_LARGE_AOI, "the beam meets a gradient-index rod's end face near grazing")
        jac, bx, by, bx_out = transfer
        coupling = max(abs(jac[i, j]) for i, j in ((0, 1), (0, 3), (1, 0), (1, 2), (2, 1), (2, 3), (3, 0), (3, 2)))
        scale = max(1.0, float(np.max(np.abs(jac))))
        if coupling > self.cfg.gaussian_astig_align_tol * scale:
            rel = p_in - grin.origin
            # (the chief ray's meridional plane: through the axis and the chief ray's start, or for an axial medium the plane
            # of the axis and its direction, in which it stays)
            across = d_in - float(dot(d_in, axis)) * axis if axial else rel - float(dot(rel, axis)) * axis
            if float(dot(across, across)) == 0.0:
                across = d_in - float(dot(d_in, axis)) * axis
            across = across - float(dot(across, d_in)) * d_in
            norm = math.sqrt(float(dot(across, across)))
            if norm == 0.0 or not _is_round_state(g0, self.cfg):
                return g0.invalidated(ErrorCode.GAUSSIAN_TWISTED_ASTIGMATISM, "the gradient-index rod's first-order transfer couples the "
                                      "beam's two axes (a skew chief ray): twisted astigmatism is outside the x/y-independent model")
            mx = across / norm
            my = np.cross(d_in, mx)
            mx_out = carried(mx)
            transfer = self._gradient_transfer(profile, grin, p_in, d_in, mx, my, exit_point, exit_d, mx_out)
            if transfer is None:
                return g0.invalidated(ErrorCode.GAUSSIAN_LARGE_AOI, "the beam meets a gradient-index rod's end face near grazing")
            jac, bx, by, bx_out = transfer
            coupling = max(abs(jac[i, j]) for i, j in ((0, 1), (0, 3), (1, 0), (1, 2), (2, 1), (2, 3), (3, 0), (3, 2)))
            if coupling > self.cfg.gaussian_astig_align_tol * max(1.0, float(np.max(np.abs(jac)))):
                return g0.invalidated(ErrorCode.GAUSSIAN_TWISTED_ASTIGMATISM, "the gradient-index rod's first-order transfer couples the "
                                      "beam's two axes (a skew chief ray): twisted astigmatism is outside the x/y-independent model")
        n_in = g0.n
        r_out = exit_point - grin.origin
        if axial:
            n_out = float(profile.index_z(np.array([float(dot(r_out, axis))]))[0])  # type: ignore[union-attr]
        else:
            r2 = float(dot(r_out, grin.ex)) ** 2 + float(dot(r_out, grin.ey)) ** 2
            n_out = profile.n0 * math.sqrt(max(0.0, 1.0 - profile.g * profile.g * r2))  # type: ignore[union-attr]
        qs = []
        for k, q in ((0, g0.qx), (1, g0.qy)):
            a, b, c, d = jac[k, k], jac[k, 2 + k] / n_in, n_out * jac[2 + k, k], n_out * jac[2 + k, 2 + k] / n_in
            if abs(a * d - b * c - 1.0) > GRADIENT_SYMPLECTIC_TOL:
                return g0.invalidated(ErrorCode.GAUSSIAN_LARGE_AOI, "the gradient-index rod's first-order transfer about the chief ray "
                                      f"is not resolved (determinant {a * d - b * c:.6g})")
            reduced = q / n_in
            qs.append(n_out * (a * reduced + b) / (c * reduced + d))
        by_out = np.cross(exit_d, bx_out)
        state = g0.with_(qx=qs[0], qy=qs[1], n=n_out, bx=bx_out, by=by_out)
        return check_divergence(state, self.cfg)

    def _gradient_transfer(self, profile: gradient.Profile | gradient.AxialProfile, grin: Any, p_in: Array, d_in: Array, bx: Array,
                           by: Array, exit_point: Array, exit_d: Array, bx_out: Array) -> tuple[Array, Array, Array, Array] | None:
        """``J`` (4 x 4) of the leg's map from (position across the chief ray along ``bx``, ``by``; direction tilt toward them) at its
        start to the same at its end (axes ``bx_out``, ``d x bx_out``), by central differences of the closed-form leg; the
        deviated rays start on the chief ray's face plane and end on the end face, then are moved along their own direction
        to the planes across the chief ray. ``None`` when the chief ray runs along a face."""
        normal = grin.axis
        normal_out = grin.axis
        exit_face = None
        if grin.faces is not None:  # (a slanted or curved face: its tangent planes at the chief ray's start and end)
            ahead = float(dot(d_in, grin.axis)) > 0.0
            normal = self._gradient_normal(grin, grin.faces[0] if ahead else grin.faces[1], p_in)
            exit_face = grin.faces[1] if ahead else grin.faces[0]
            normal_out = self._gradient_normal(grin, exit_face, exit_point)
        along_in = float(dot(d_in, normal))
        along_out = float(dot(exit_d, normal_out))
        if abs(along_in) < GRADIENT_GRAZING_COS or abs(along_out) < GRADIENT_GRAZING_COS:
            return None
        by_out = np.cross(exit_d, bx_out)
        step = GRADIENT_JACOBIAN_STEP
        starts, dirs = [], []
        for sign in (1.0, -1.0):
            for k in range(4):
                axis_vec = (bx, by, bx, by)[k]
                if k < 2:
                    offset = sign * step * axis_vec
                    starts.append(p_in + offset - (float(dot(offset, normal)) / along_in) * d_in)
                    dirs.append(d_in)
                else:
                    tilted = d_in + sign * step * axis_vec
                    starts.append(p_in)
                    dirs.append(tilted / math.sqrt(float(dot(tilted, tilted))))
        o = np.array(starts)
        d = np.array(dirs)
        rel = o - grin.origin
        d_local = np.stack([dot(d, grin.ex), dot(d, grin.ey), dot(d, grin.axis)], axis=-1)
        leg, _along = gradient.legs(profile, dot(rel, grin.ex), dot(rel, grin.ey), dot(rel, grin.axis), d_local, grin.z_lo, grin.z_hi)
        if exit_face is not None:
            with np.errstate(invalid="ignore", divide="ignore"):
                leg, _reached = gradient.to_face(leg, exit_face)
        x, y, z, dx, dy = leg.at(leg.z_len)
        out_dir = np.stack([dx, dy, leg.sign], axis=-1)
        out_dir = out_dir / np.sqrt(np.sum(out_dir * out_dir, axis=-1, keepdims=True))
        out_point, out_d = _to_scene(grin, np.stack([x, y, z], axis=-1)), _to_scene(grin, out_dir, point=False)
        delta = out_point - exit_point
        moved = delta - (dot(delta, exit_d) / dot(out_d, exit_d))[:, None] * out_d
        cols = np.stack([dot(moved, bx_out), dot(moved, by_out), dot(out_d, bx_out), dot(out_d, by_out)], axis=-1)
        jac = ((cols[:4] - cols[4:]) / (2.0 * step)).T  # rows: outputs, columns: inputs
        return jac, bx, by, bx_out

    # ---------------------------------------------------------------------------- uniaxial crystals
    def _crystal_face(self, surf: SurfaceG) -> bool:
        """Whether a uniaxial crystal lies on a side of ``surf`` (``scene/lower.py`` checked it is a refracting plane)."""
        return surf.medium_pos in self.scene.optic_axes or surf.medium_neg in self.scene.optic_axes

    def _crystal_at(self, medium: str, wavelength_nm: float, location: str) -> tuple[Crystal | None, ValidationIssue | None]:
        """The uniaxial crystal ``medium`` at a vacuum wavelength: both indices (outside their band: the material issue), its optic
        axis in the scene frame and both indices' group index and dispersion."""
        key = (medium, wavelength_nm)
        if key in self._crystals:
            return self._crystals[key], None
        laws = self.library.uniaxial(medium)
        assert laws is not None
        n_o, issue = laws.ordinary_index.index(wavelength_nm, location=location)
        if issue is not None or n_o is None:
            return None, issue
        n_e, issue = laws.extraordinary_index.index(wavelength_nm, location=location)
        if issue is not None or n_e is None:
            return None, issue
        crystal = Crystal(n_o, n_e, self.scene.optic_axes[medium], self.library.crystal_group(laws, False, wavelength_nm),
                          self.library.crystal_group(laws, True, wavelength_nm), wavelength_nm)
        self._crystals[key] = crystal
        return crystal, None

    def _crystal_meets(self, pk: _Packet, surf: SurfaceG, inter: InteractionG) -> str:
        """Light inside a crystal meets a surface that is not one of its refracting faces (an element placed inside the
        crystal's volume): outside the model, the path ends there."""
        self._issue(make_issue(ErrorCode.BIREFRINGENT_MEDIUM_UNSUPPORTED, Status.OUT_OF_MODEL_SCOPE,
                               f"light travelling in the uniaxial crystal {pk.medium} meets {surf.surface_id} ({inter.kind}) inside "
                               "it: only the crystal's own refracting faces are traced in a crystal, so the light stops there",
                               location=f"surface:{surf.surface_id}", path_id=pk.path_id, material=pk.medium,
                               interaction=inter.kind, suggested_action="place elements outside the crystal's volume"))
        self._end_event(pk, "birefringent_unsupported")
        pk.alive[:] = False
        return ErrorCode.BIREFRINGENT_MEDIUM_UNSUPPORTED.value

    @staticmethod
    def _crystal_event(event: str, waves: Waves) -> str:
        """A lineage event, marked with the crystal wave the light leaves as (``event_o`` / ``event_e``; the one wave along the
        optic axis is ordinary)."""
        return event if waves.mode is None else f"{event}_{uniaxial.ORDINARY if waves.mode == uniaxial.AXIAL else waves.mode}"

    def _crystal_gaussian(self, pk: _Packet, surf: SurfaceG, waves: Waves, crystal: Crystal | None, n_iso: float | None,
                          error_mm: float) -> tuple[GState, float]:
        """The Gaussian state of the beam leaving a crystal face as ``waves`` (``optics/polarization/uniaxial.py``): the
        footprint maps from the plane across the incident ray onto the face and across the outgoing ray (a twisted
        astigmatic map is outside the x/y-independent model), and the beam diffracts with the outgoing wave's index. Returns
        the state and its index's error per mm of reduced path; ``error_mm`` (the reduced q error the crystal legs behind it
        accumulated) beyond ``birefringent_gaussian_rel_tol`` of its reduced Rayleigh range puts the state out of model
        scope."""
        index, per_mm = uniaxial.gaussian_index(crystal, waves, n_iso)
        g = apply_grating(pk.gstate, pk.d[0], waves.S[0], surf.normal, pk.gstate.n, index, self.cfg)
        if g.valid and error_mm > 0.0:
            reach = min(g.qx.imag, g.qy.imag) / g.n
            if error_mm > self.cfg.birefringent_gaussian_rel_tol * reach:
                g = g.invalidated(ErrorCode.GAUSSIAN_ANISOTROPIC_MEDIUM, (
                    f"the beam crossed a uniaxial crystal as an extraordinary wave, whose two transverse axes diffract with "
                    f"different indices: one Gaussian index leaves its q-parameter uncertain by {error_mm:.3g} mm of reduced path, "
                    f"more than {self.cfg.birefringent_gaussian_rel_tol:g} of its reduced Rayleigh range {reach:.4g} mm, so its "
                    "Gaussian beam metrics are out of model scope (rays and powers are traced)"))
        return g, per_mm

    def _do_crystal_face(self, pk: _Packet, surf: SurfaceG, h: Hits, inter: InteractionG, from_pos: bool, med_to: str) -> str | None:
        """A plane face with a uniaxial crystal on a side (``optics/polarization/uniaxial.py``): the light leaving it is up to
        two transmitted and two reflected plane waves -- one per isotropic medium, the ordinary and the extraordinary wave in
        a crystal -- each a beam of its own along its ray direction, with its share of the power. The packet goes on as the
        first transmitted wave that carries power (ordinary before extraordinary); the other transmitted wave is a branch
        (``birefringent_split``); the reflected waves are traced branches (``trace_fresnel_reflection``) or untraced light
        (``UNTRACED_REFLECTION``). Where no wave is transmitted (total internal reflection of the incident wave) the packet
        goes on as the first reflected wave and the other is a branch. Light of a crystal on the far side whose wave is
        evanescent is reflected (its share of the face's reflection)."""
        cfg = self.cfg
        tol = cfg.zero_vector_tol
        location = f"surface:{surf.surface_id}"
        before = None if pk.crystal is None else pk.crystal.crystal
        if before is None and pk.medium in self.scene.optic_axes:
            return self._crystal_meets(pk, surf, inter)  # (light in a crystal without its wave: not one of the model's beams)
        after: Crystal | None = None
        n_after: float | None = None
        if med_to in self.scene.optic_axes:
            after, issue = self._crystal_at(med_to, pk.wavelength_nm, location)
        else:
            n_after, issue = self.library.evaluator(med_to).index(pk.wavelength_nm, location=location)
        if issue is not None or (after is None and n_after is None):
            if issue is not None:
                self._issue(issue)
            pk.alive[:] = False
            return MATERIAL_RANGE
        normal = h.normal * (-1.0 if from_pos else 1.0)  # rows: toward the medium beyond the face
        k_in = pk.d * pk.n if pk.crystal is None else pk.crystal.wave
        k_size = np.sqrt(dot(k_in, k_in))
        k_hat = k_in / k_size[:, None]
        cos_i = np.minimum(np.maximum(dot(k_hat, normal), 0.0), 1.0)
        aoi_deg = incidence_deg(cos_i, k_hat, normal)
        term = self._aoi_guard(pk, surf, aoi_deg)
        if term:
            return term
        kt = k_in - dot(k_in, normal)[:, None] * normal
        s = s_axis(k_hat, -normal, pk.e1, tol)
        j_s, j_p, p_in = uniaxial.transverse_field(field_3d(pk.jones, pk.e1, pk.e2), k_hat, s, tol)
        k_ref = k_hat - (2.0 * dot(k_hat, normal))[:, None] * normal  # the incident wave normal mirrored in the face
        p_ref = np.cross(k_ref, s)
        floor = np.maximum(cos_i, np.finfo(float).tiny)
        coated = inter.coating is not None
        cc: CoatingCoefficients | None = None
        ph: PhaseValues | None = None
        if coated:
            cc = self._live_coating(pk, surf, inter.coating, aoi_deg)  # type: ignore[arg-type]
            if cc is None:
                return COATING_RANGE
            ph = self._stated_phases(pk, surf, inter)
        state = self._sp_state(pk, s, float(aoi_deg[0])) if coated else None
        transmitted: list[tuple[Waves, CArray]] = []
        ordinary = None if after is None else uniaxial.outgoing(kt, normal, after, None, uniaxial.ORDINARY, True, s, tol)
        # (along the crystal's optic axis its two waves are one: the light passes as into an isotropic medium of index n_o)
        axial_t = after is not None and uniaxial.axial(after, ordinary)  # type: ignore[arg-type]
        if after is None or axial_t:
            n_t = n_after if after is None else after.n_o
            waves = (uniaxial.outgoing(kt, normal, None, n_t, None, True, s, tol) if after is None
                     else uniaxial.axial_wave(kt, normal, after, True, s, tol))
            fr = fresnel(k_size, float(n_t), floor)  # type: ignore[arg-type]
            tau_s, tau_p, r_s, r_p = fr.tau_s, fr.tau_p, fr.r_s, fr.r_p
            chief_tir = bool(fr.tir[0])
            pk.alive = pk.alive & (fr.tir == chief_tir)
            if coated and not chief_tir:
                tau_s, tau_p, r_s, r_p = coated_interface(cc.rs, cc.rp, cc.ts, cc.tp, tau_s, tau_p, r_s, r_p, ph)  # type: ignore[union-attr]
            if not chief_tir:
                transmitted.append((waves, np.stack([cmul(tau_s, j_s), cmul(tau_p, j_p)], axis=1)))
            field_ref = cmul(r_s, j_s)[:, None] * s + cmul(r_p, j_p)[:, None] * p_ref
        else:
            extraordinary = uniaxial.outgoing(kt, normal, after, None, uniaxial.EXTRAORDINARY, True, s, tol)
            u_o, u_e = uniaxial.split_axes(ordinary.e1, k_hat, s, tol)  # type: ignore[union-attr]
            field_ref = np.zeros((k_in.shape[0], 3), dtype=np.complex128)
            reflected_W = np.zeros(k_in.shape[0])
            for waves, u in ((ordinary, u_o), (extraordinary, u_e)):
                x, c_s, c_p = uniaxial.along(j_s, j_p, s, p_in, u)
                index = np.where(waves.ok, np.sqrt(dot(waves.K, waves.K)), k_size)
                fr = fresnel(k_size, index, floor)
                t_s, t_p, r_s, r_p = fr.T_s, fr.T_p, fr.r_s, fr.r_p
                big_r_s, big_r_p = fr.R_s, fr.R_p
                if coated:
                    t_s, t_p, big_r_s, big_r_p = cc.ts, cc.tp, cc.rs, cc.rp  # type: ignore[union-attr]
                    _ts, _tp, r_s, r_p = coated_interface(cc.rs, cc.rp, cc.ts, cc.tp, fr.tau_s, fr.tau_p, fr.r_s, fr.r_p, ph)  # type: ignore[union-attr]
                share_t = np.where(waves.ok, c_s * c_s * t_s + c_p * c_p * t_p, 0.0)
                # (an evanescent wave reflects its whole part, with a phase the model does not resolve)
                share_r = np.where(waves.ok, c_s * c_s * big_r_s + c_p * c_p * big_r_p, 1.0)
                r_s = np.where(waves.ok, r_s, 1.0)
                r_p = np.where(waves.ok, r_p, 1.0)
                along = uniaxial.oriented(waves, u)
                if bool(waves.ok[0]):
                    transmitted.append((along, uniaxial.mode_jones(along, x * np.sqrt(share_t))))
                field_ref = field_ref + cmul(x * c_s, r_s)[:, None] * s + cmul(x * c_p, r_p)[:, None] * p_ref
                reflected_W = reflected_W + (x.real * x.real + x.imag * x.imag) * share_r
            pk.alive = pk.alive & (ordinary.ok == bool(ordinary.ok[0])) & (extraordinary.ok == bool(extraordinary.ok[0]))  # type: ignore[union-attr]
            # the parts' reflections add to the face's reflected power (its conservation): their sum keeps its polarization
            parts = field_ref.real * field_ref.real + field_ref.imag * field_ref.imag
            size = parts[:, 0] + parts[:, 1] + parts[:, 2]
            field_ref = field_ref * np.sqrt(np.where(size > 0.0, reflected_W, 0.0) / np.where(size > 0.0, size, 1.0))[:, None]
        reflected: list[tuple[Waves, CArray]] = []
        if before is None:
            waves = uniaxial.outgoing(kt, normal, None, pk.n, None, False, s, tol)
            reflected.append((waves, np.stack([_cdot(field_ref, waves.e1), _cdot(field_ref, waves.e2)], axis=1)))
        elif uniaxial.axial(before, back := uniaxial.outgoing(kt, normal, before, None, uniaxial.ORDINARY, False, s, tol)):
            # (reflected along the optic axis: one wave, the reflected field whole on its transverse pair)
            waves = uniaxial.axial_wave(kt, normal, before, False, s, tol)
            v_1, v_2 = uniaxial.split_axes(waves.e1, k_ref, s, tol)
            reflected.append((waves, np.stack([_cdot(field_ref, v_1), _cdot(field_ref, v_2)], axis=1)))
        else:
            ordinary = back
            extraordinary = uniaxial.outgoing(kt, normal, before, None, uniaxial.EXTRAORDINARY, False, s, tol)
            v_o, v_e = uniaxial.split_axes(ordinary.e1, k_ref, s, tol)
            y_o, y_e = _cdot(field_ref, v_o), _cdot(field_ref, v_e)
            # (an evanescent reflected wave: the reflection goes to the other wave)
            total = (y_o.real * y_o.real + y_o.imag * y_o.imag) + (y_e.real * y_e.real + y_e.imag * y_e.imag)
            for waves, y, other_ok in ((ordinary, y_o, extraordinary.ok), (extraordinary, y_e, ordinary.ok)):
                own = y.real * y.real + y.imag * y.imag
                lone = waves.ok & ~other_ok & (own > 0.0)
                y = np.where(lone, y * np.sqrt(total / np.where(own > 0.0, own, 1.0)), y)
                if bool(waves.ok[0]):
                    along = uniaxial.oriented(waves, v_o if waves.mode == uniaxial.ORDINARY else v_e)
                    reflected.append((along, uniaxial.mode_jones(along, y)))
        axes = [c.axis for c in (before, after) if c is not None]
        if (float(aoi_deg[0]) > cfg.birefringent_interface_warn_aoi_deg and not uniaxial.decoupled(axes, s[0])
                and surf.surface_id not in self._crystal_reported):
            self._crystal_reported.add(surf.surface_id)
            self._issue(make_issue(ErrorCode.BIREFRINGENT_INTERFACE_APPROXIMATE, Status.WARNING,
                                   f"light meets the crystal face {surf.surface_id} at {float(aoi_deg[0]):.3g} deg: its split between the "
                                   "ordinary and extraordinary waves uses the isotropic Fresnel transmittance at each wave's index (exact at "
                                   "normal incidence), so the powers of the waves leaving the face are approximate",
                                   location=location, aoi_deg=float(aoi_deg[0]), warn_deg=cfg.birefringent_interface_warn_aoi_deg,
                                   suggested_action="meet the crystal faces near normal incidence where the wave powers matter"))
        # the crystal legs behind this face: their Gaussian error
        error_mm = pk.gaussian_error_mm
        if pk.crystal is not None:
            error_mm = error_mm + (float(pk.length[0]) - pk.crystal.start_length) * pk.crystal.gaussian_error_per_mm
        # (light that may have generated a second harmonic here goes on with it, unlike the light this face reflects)
        with_harmonic: tuple[str, ...] | None = None
        unknown_modes: tuple[str, ...] = ()
        if after is not None and med_to in self.scene.nonlinear and self._converts_harmonics and not self.bounding:
            transmitted, generated, unknown_modes = self._second_harmonic(pk, surf, med_to, after, transmitted, normal, kt, s, error_mm)
            if generated or unknown_modes:
                with_harmonic = (*pk.harmonic_at, surf.surface_id)

        def power(jones: CArray) -> float:
            return pk.power0_W * power2(jones[0])

        def pass_delay(event: str) -> str:
            # (a coated face beyond the critical angle: the coating's total reflection has a delay the design does not state)
            return (self._delay(pk, surf, event, float(aoi_deg[0]), from_pos, state=state, inter=inter) if event == "refract"
                    else self._delay(pk, surf, "coated_tir", float(aoi_deg[0]), from_pos, state=state))

        def carries(jones: CArray) -> bool:
            return pk.power_bound_W is not None or power(jones) > 0.0

        def take(target: _Packet, waves: Waves, jones: CArray, medium: str, crystal: Crystal | None, n_iso: float | None) -> None:
            alive = pk.alive & waves.ok
            gstate, per_mm = self._crystal_gaussian(pk, surf, waves, crystal, n_iso, error_mm)
            target.d = np.where(alive[:, None], waves.S, pk.d)
            target.e1 = np.where(alive[:, None], waves.e1, pk.e1)
            target.e2 = np.where(alive[:, None], waves.e2, pk.e2)
            target.jones = jones
            target.alive = alive
            target.medium, target.n = medium, float(waves.ray_index[0])
            target.gstate = gstate
            target.gaussian_error_mm = error_mm
            target.crystal = None if crystal is None else _CrystalLeg(
                waves.mode, crystal, waves.K, waves.ray_index, uniaxial.ray_group(crystal, waves), per_mm,  # type: ignore[arg-type]
                float(pk.length[0]))

        def open_converting_pump(target: _Packet) -> None:
            incident_W = target.chief_power()
            self._flow(target, SINK_BULK_LOSS_UNKNOWN, incident_W, surf, "second harmonic with unstated bulk absorption")
            # The watts may remain in the pump or change wavelength inside the slab. Keep the full incident power as the
            # bound for this pump wave, without tightening it using attenuation along an unconverted pump's path.
            target.power_bound_W = incident_W
            target.conversion_loss_unknown = True

        # A stated surface scatter share was booked immediately before the interface handler.
        incident_W = pk.chief_power() + self._branch_power if unknown_modes else 0.0
        med_from = pk.medium
        transmits = any(carries(j) for _, j in transmitted)
        if transmits:
            goes_on = [(w, j, med_to, after, n_after, "refract") for w, j in transmitted]
            branches = reflected
        else:
            goes_on = [(w, j, med_from, before, pk.n, "tir") for w, j in reflected]
            branches = []
        lead = next((k for k, (_w, j, *_rest) in enumerate(goes_on) if carries(j)), 0) if goes_on else None
        # (the face splits the light into two crystal waves that both carry more than the negligible level)
        split = sum(1 for w, j, _m, crystal, *_r in goes_on
                    if crystal is not None and w.mode is not None and carries(j) and not self._negligible(pk, power(j))) >= 2
        if split and pk.coherent_twin:
            self._twins_mixed(pk, surf)
        # the other wave of the light that goes on: a branch of its own
        for k, (waves, jones, medium, crystal, n_iso, event) in enumerate(goes_on):
            if k == lead or not carries(jones):
                continue
            p = power(jones)
            child = self._spawn(pk, surf, "birefringent_split", p)
            if child is None:
                self._branch_power += p
                continue
            take(child, waves, jones, medium, crystal, n_iso)
            if event == "refract" and waves.mode in unknown_modes:
                open_converting_pump(child)
            if with_harmonic is not None:
                child.harmonic_at = with_harmonic
            if split:
                child.coherent_twin = True
            child.lineage.append(f"{surf.surface_id}:{self._crystal_event(event, waves)}")
            if coated and not self._delay_stated(inter, "transmission" if event == "refract" else "reflection"):
                child.delays = (*child.delays, pass_delay(event))
            if self.recorder is not None:
                self.recorder.branch_spawned(pk, child)
            self._branch_power += child.chief_power()
            if not self._spent(child):
                self._enqueue(child)
            else:
                self._finish(child, ZERO_POWER, surf.surface_id)
        # the reflected waves of light that is transmitted
        branch_event = "fresnel_reflection" if not coated else "coating_reflection"
        for waves, jones in branches:
            if not carries(jones):
                continue
            p = power(jones)
            crystal_r = before
            if inter.trace_reflection:
                child = self._spawn(pk, surf, branch_event, p)
                if child is None:
                    self._branch_power += p
                    continue
                take(child, waves, jones, med_from, crystal_r, pk.n)
                child.lineage.append(f"{surf.surface_id}:{self._crystal_event(branch_event, waves)}")
                if not self._delay_stated(inter, "reflection"):
                    child.delays = (*child.delays, self._delay(pk, surf, branch_event, float(aoi_deg[0]), from_pos, state=state,
                                                               inter=inter))
                if self.recorder is not None:
                    self.recorder.branch_spawned(pk, child)
                self._branch_power += child.chief_power()
                if not self._spent(child):
                    self._enqueue(child)
                else:
                    self._finish(child, ZERO_POWER, surf.surface_id)
            else:
                self._split[SINK_UNTRACED_REFLECTION] = self._split.get(SINK_UNTRACED_REFLECTION, 0.0) + p
                self._split_detail = "interface reflection (not followed)"
                g_ref, per_mm = self._crystal_gaussian(pk, surf, waves, crystal_r, pk.n, error_mm)
                leg = None if crystal_r is None else _CrystalLeg(waves.mode, crystal_r, waves.K, waves.ray_index,  # type: ignore[arg-type]
                                                                 uniaxial.ray_group(crystal_r, waves), per_mm, float(pk.length[0]))
                self._branch(pk, SINK_UNTRACED_REFLECTION, surf, waves.S[0], g_ref, p, jones=jones[0], e1=waves.e1[0], e2=waves.e2[0],
                             rays=lambda waves=waves, jones=jones: (waves.S, waves.e1, waves.e2, jones), alive=pk.alive & waves.ok,
                             delay=None if self._delay_stated(inter, "reflection") else
                             self._delay(pk, surf, branch_event, float(aoi_deg[0]), from_pos, state=state, inter=inter),
                             wave=(float(waves.ray_index[0]), leg))
        if lead is None:  # (no wave leaves the face: nothing is traced on)
            self._end_event(pk, "refract", float(aoi_deg[0]))
            pk.alive[:] = False
            pk.jones = pk.jones * 0.0
            return ZERO_POWER
        waves, jones, medium, crystal, n_iso, event = goes_on[lead]
        take(pk, waves, jones, medium, crystal, n_iso)
        if event == "refract" and waves.mode in unknown_modes:
            # Close the known interface split before opening the transmitted packet's power. The caller's ledger step
            # then skips this packet, as it does for other transitions to unknown power.
            self._record_interaction_loss(pk, surf, inter.kind, incident_W)
            open_converting_pump(pk)
        if with_harmonic is not None:
            pk.harmonic_at = with_harmonic
        if split:
            pk.coherent_twin = True
        self._report_gaussian_state(pk, surf)
        self._end_event(pk, event, float(aoi_deg[0]))
        pk.lineage.append(f"{surf.surface_id}:{self._crystal_event(event, waves)}")
        if coated and not self._delay_stated(inter, "transmission" if event == "refract" else "reflection"):
            pk.delays = (*pk.delays, pass_delay(event))
        return ZERO_POWER if self._spent(pk) else None

    def _twins_polarized(self, pk: _Packet, surf: SurfaceG, diattenuation: float, what: str) -> None:
        """``pk`` is one of two coherent crystal waves of an earlier split (``_Packet.coherent_twin``; the two leave a crystal
        overlapping, orthogonally polarized) and ``surf`` transmits (or reflects) s and p, or a polarizer's two axes, by
        powers that differ by ``diattenuation``: the power it passes of the two waves' sum is not the sum of what it passes of
        each (their cross term, up to ``diattenuation`` times the line's power, depends on their relative phase, which the
        trace does not carry; a crystal plate between a polarizer pair reads the same power at every thickness by the sum).
        Above the line's negligible level the power behind the surface is out of model scope (``_twins_mixed``). A surface
        that treats both polarizations alike (normal incidence on an isotropic interface, an unpolarized coating) mixes
        nothing; neither does a polarization-preserving one (a retarder: its phase delay mixes them only at a later
        diattenuating surface, which this rule meets there). Whether the two waves still overlap there is not tested: two
        waves a displacer has separated are taken as overlapping too."""
        if not pk.coherent_twin or self.bounding or not (diattenuation * pk.power0_W > self._negligible_level(pk)):
            return
        self._twins_mixed(pk, surf, what)

    def _twins_mixed(self, pk: _Packet, surf: SurfaceG, what: str | None = None) -> None:
        """``pk`` is one of two coherent crystal waves of an earlier split (``_Packet.coherent_twin``) and ``surf`` splits it
        again into two waves: its twin, split there too, feeds the same two waves, and the light leaving in each is their
        fields added with a relative phase the trace does not carry (the two waves stay one coherent field: along the optic
        axis, or near it, their phase indices differ by about ``(n_o - n_e) theta^2``). Adding their powers is no bound of the
        light's: from here on the power is out of model scope (``power_scope``, an ``OUT_OF_MODEL_SCOPE`` issue). A face that
        sends each wave into one wave of the next crystal (a Wollaston or Rochon prism in its own plane) mixes nothing."""
        if surf.surface_id in pk.power_scope:
            return
        pk.power_scope = (*pk.power_scope, surf.surface_id)
        mixing = (f"{surf.surface_id} splits it again into two waves that its twin also feeds" if what is None else
                  f"{surf.surface_id} ({what}) passes its two polarizations by different powers, and its twin meets it too")
        self._issue(make_issue(
            ErrorCode.OUT_OF_MODEL_SCOPE, Status.OUT_OF_MODEL_SCOPE,
            f"beam {pk.path_id} is one of the two crystal waves an earlier face split the light into, and {mixing}: the light "
            "leaving is the two fields added with a relative phase the trace does not follow, so its power behind this surface "
            "is not computed",
            location=f"surface:{surf.surface_id}", path_id=pk.path_id, surface_ids=[surf.surface_id],
            suggested_action="align the crystals' optic axes with the plane of incidence (or across it) at this face, so that each "
                             "wave passes into one wave of the next crystal"))

    def _back_transmittance(self, pk: _Packet, surf: SurfaceG, h: Hits, inter: InteractionG) -> tuple[float, float] | None:
        """``(T, AOI deg)`` of a mirror's coating for the chief ray meeting its back, where its published data give a transmittance above
        zero at that wavelength and angle (the larger of T_s and T_p); ``None`` for an opaque mirror, an uncoated (ideal) one or an
        angle or wavelength outside its data."""

        if inter.coating is None:
            return None
        _d, _n, cos_i = reflect_dir(pk.d[:1], h.normal[:1], self.cfg.zero_vector_tol)
        aoi = incidence_deg(cos_i, pk.d[:1], h.normal[:1])
        if not bool(inter.coating.in_range(pk.wavelength_nm, aoi)[0]):
            return None
        values = inter.coating.coefficients(pk.wavelength_nm, aoi, location=f"surface:{surf.surface_id}")
        t = max(float(values.ts[0]), float(values.tp[0]))
        return (t, float(aoi[0])) if t > 0.0 else None

    def _crystal_with(self, medium: str, wavelength_nm: float, temperature_c: float | None = None) -> Crystal | None:
        """The crystal ``medium`` at a vacuum wavelength and a temperature (default: its own), indices only (``None`` where an
        index is not stated)."""
        got = self.library.crystal_indices(medium, wavelength_nm, temperature_c)
        return None if got is None else Crystal(got[0], got[1], self.scene.optic_axes[medium], None, None, wavelength_nm)

    def _acceptance(self, medium: str, k_hat: Array, modes: tuple[str, str], mode_h: str, length_mm: float,
                    wavelength_nm: float, grating: Callable[[float | None], float]) -> tuple[float | None, float | None, float | None]:
        """Full widths at half maximum of ``sinc^2(dk L / 2)`` about the traced beam (``NonlinearConversionResult``): internal
        angle (mrad), temperature (K), fundamental vacuum wavelength (nm). Each is the narrower of the widths the first and the
        second derivative of the collinear ``dk`` give (central differences); ``None`` where neither is defined."""

        def mismatch(direction: Array, wl: float, temperature: float | None) -> float | None:
            f, h = self._crystal_with(medium, wl, temperature), self._crystal_with(medium, wl / 2.0, temperature)
            if f is None or h is None:
                return None
            return shg.collinear_mismatch(direction, f, h, modes, mode_h, grating(temperature))

        def width(at: Callable[[float], float | None], step1: float, step2: float) -> float | None:
            values = [at(x) for x in (-step1, step1, -step2, 0.0, step2)]
            if any(v is None for v in values):
                return None
            m1, p1, m2, mid, p2 = values  # type: ignore[misc]
            first = abs((p1 - m1) / (2.0 * step1))
            second = abs((p2 - 2.0 * mid + m2) / (step2 * step2))
            widths = []
            if first > 0.0:
                widths.append(shg.FWHM_DK_L / (length_mm * first))
            if second > 0.0:
                widths.append(2.0 * math.sqrt(shg.FWHM_DK_L / (length_mm * second)))
            return min(widths) if widths else None

        axis = self.scene.optic_axes[medium]
        toward = axis - float(dot(axis, k_hat)) * k_hat
        size = math.sqrt(float(dot(toward, toward)))
        if size <= self.cfg.zero_vector_tol:  # (along the optic axis: any plane through it)
            toward = transverse_basis(k_hat, self.cfg.zero_vector_tol)[0][0]
        else:
            toward = toward / size

        def turned(delta: float) -> float | None:
            d = k_hat + delta * toward
            return mismatch(d / math.sqrt(float(dot(d, d))), wavelength_nm, None)

        angular = width(turned, ACCEPTANCE_ANGLE_STEP_RAD, ACCEPTANCE_ANGLE_STEP2_RAD)
        temperature = None
        state = self.library.uniaxial(medium).material.temperature  # type: ignore[union-attr]
        if state is not None:
            temperature = width(lambda dt: mismatch(k_hat, wavelength_nm, state.temperature_c + dt), ACCEPTANCE_TEMPERATURE_STEP_K,
                                ACCEPTANCE_TEMPERATURE_STEP2_K)
        spectral = width(lambda dl: mismatch(k_hat, wavelength_nm + dl, None), ACCEPTANCE_WAVELENGTH_STEP_NM,
                         ACCEPTANCE_WAVELENGTH_STEP2_NM)
        return (None if angular is None else angular * RAD_TO_MRAD), temperature, spectral

    def _line_average(self, medium: str, k_hat: Array, modes: tuple[str, str], mode_h: str, length_mm: float, wavelength_nm: float,
                      grating: Callable[[float | None], float], phase: float, fwhm_nm: float, rule: tuple[Array, Array]) -> float | None:
        """``sinc^2(dk L / 2)`` averaged over a Gaussian fundamental line of FWHM ``fwhm_nm`` about the traced beam
        (``shg.line_average``): ``dk L / 2`` at an offset wavelength is the traced ``phase`` plus the change of the collinear
        ``dk`` along ``k_hat`` (``None``: an index is not stated over the line)."""

        def mismatch(wl: float) -> float | None:
            f, h = self._crystal_with(medium, wl), self._crystal_with(medium, wl / 2.0)
            if f is None or h is None:
                return None
            return shg.collinear_mismatch(k_hat, f, h, modes, mode_h, grating(None))

        centre = mismatch(wavelength_nm)
        if centre is None:
            return None

        def phase_at(offset_nm: float) -> float | None:
            dk = mismatch(wavelength_nm + offset_nm)
            return None if dk is None else phase + (dk - centre) * length_mm / 2.0

        return shg.line_average(phase_at, fwhm_nm, rule)

    def _second_harmonic(self, pk: _Packet, surf: SurfaceG, medium: str, crystal: Crystal, transmitted: list[tuple[Waves, CArray]],
                         normal: Array, kt: Array, s_rows: Array,
                         error_mm: float) -> tuple[list[tuple[Waves, CArray]], bool, tuple[str, ...]]:
        """Light entering a crystal that generates its second harmonic (``optics/nonlinear/shg.py``): the undepleted-pump
        harmonic of the beam's converting waves, generated as a beam of its own -- a new spectral line of the source (index
        ``spectral index + number of the source's lines``, half the wavelength), traced from this face as the crystal's
        ``harmonic_wave`` -- while the fundamental waves go on with the power the harmonic took (``NONLINEAR_CONVERSION`` on
        the fundamental's line, the harmonic line's input). Returns ``transmitted`` with the fundamental waves' fields
        reduced, whether a harmonic beam was generated, and which transmitted modes carry unknown pump power when bulk
        absorption data are missing. Outside the model's conditions the harmonic quantities are out of model scope
        (the beams that go on carry the crystal in ``power_scope``); one ``NonlinearConversionResult`` per pass.

        In an absorbing crystal the harmonic is generated along it while the waves decay (``shg.overlap_factor``):
        ``harmonic_power_W`` is the power reaching the exit face, and the harmonic beam -- traced from the entrance face,
        absorbed over its path -- starts with that power divided by its transmittance through the crystal (the harmonic
        line's input and the fundamental's ``NONLINEAR_CONVERSION``), so it arrives with it."""
        cfg = self.cfg
        nl = self.scene.nonlinear[medium]
        spec = nl.spec
        location = f"surface:{surf.surface_id}"
        modes = (spec.fundamental_waves[0], spec.fundamental_waves[1])
        two_waves = modes[0] != modes[1]
        mode_h = spec.harmonic_wave
        by_mode = {w.mode: (k, w, j) for k, (w, j) in enumerate(transmitted)}

        def power(jones: CArray) -> float:
            return pk.power0_W * power2(jones[0])

        known = pk.power_bound_W is None
        total_W = correctly_rounded_sum(power(j) for _, j in transmitted) if known else 0.0
        p_a = power(by_mode[modes[0]][2]) if modes[0] in by_mode and known else 0.0
        p_b = power(by_mode[modes[1]][2]) if modes[1] in by_mode and known else 0.0
        converting_W = p_a if not two_waves else (p_a + p_b if p_a > 0.0 and p_b > 0.0 else 0.0)
        n_lines = len(pk.src.spectrum)
        h_index = pk.spectral_index + n_lines
        h_wavelength = pk.wavelength_nm / 2.0
        thickness = nl.thickness_mm
        state = self.library.uniaxial(medium).material.temperature  # type: ignore[union-attr]
        base = dict(element_id=surf.owner_id, surface_id=surf.surface_id, path_id=pk.path_id, source_id=pk.src.source_id,
                    spectral_index=pk.spectral_index, wavelength_nm=pk.wavelength_nm, harmonic_spectral_index=h_index,
                    harmonic_wavelength_nm=h_wavelength, fundamental_waves=spec.fundamental_waves, harmonic_wave=mode_h,
                    focusing=spec.focusing, fundamental_power_W=total_W if known else None,
                     converting_power_W=converting_W if known else None, length_mm=thickness,
                    temperature_c=None if state is None else state.temperature_c, qpm_period_um=spec.qpm_period_um)

        def report(reasons: list[str]) -> None:
            self._issue(make_issue(ErrorCode.NONLINEAR_CONVERSION_OUT_OF_SCOPE, Status.OUT_OF_MODEL_SCOPE,
                                   f"the second harmonic generated in {surf.owner_id} is out of model scope: {'; '.join(reasons)}",
                                   location=location, path_id=pk.path_id, wavelength_nm=pk.wavelength_nm,
                                   suggested_action="keep the conversion inside the second-harmonic model's conditions "
                                   "('os-optics results nonlinear' lists each pass's checks)"))
            if surf.surface_id not in pk.power_scope:
                pk.power_scope = (*pk.power_scope, surf.surface_id)

        def refuse(reason: str) -> tuple[list[tuple[Waves, CArray]], bool, tuple[str, ...]]:
            self.out.nonlinear_conversions.append(NonlinearConversionResult(status=Status.OUT_OF_MODEL_SCOPE, reason=reason, **base))
            report([reason])
            return transmitted, False, ()

        def unknown_absorption(reason: str, missing: tuple[float, ...] = ()) -> None:
            details = {"missing_bulk_loss_wavelengths_nm": missing} if missing else {}
            self._issue(make_issue(ErrorCode.BULK_LOSS_NOT_ESTABLISHED, Status.WARNING, reason, location=location,
                                   path_id=pk.path_id, wavelength_nm=missing[0] if missing else pk.wavelength_nm,
                                   suggested_action="supply the missing bulk-loss data or explicitly assume a scalar bulk "
                                   "absorption coefficient", **details))
            self.out.nonlinear_conversions.append(NonlinearConversionResult(status=Status.UNKNOWN, reason=reason, **base))

        if nl.out_of_scope is not None:  # (not a slab between two parallel plane faces: no conversion is computed in it)
            return refuse(nl.out_of_scope)
        # --- the fundamental: one coherent, polarized TEM00 line of known power with a Gaussian state
        if not known:
            if pk.bulk_loss_unknown or pk.conversion_loss_unknown:
                unknown_absorption("upstream bulk absorption is not established, so the fundamental power entering the crystal "
                                   "and its second-harmonic power are unknown")
                return transmitted, False, ()
            return refuse("the fundamental power entering the crystal is not established; an upper bound on it does not "
                          "determine the second-harmonic power")
        if pk.polarization_open:
            return refuse("the fundamental's polarization is not known here")
        if any(w.mode == uniaxial.AXIAL for w, _ in transmitted):
            return refuse("the fundamental travels along the crystal's optic axis, where its ordinary and extraordinary waves are one "
                          "wave: the model converts a fundamental split into the two waves")
        # (the beam's power is out of model scope behind these surfaces -- part of it met an element its chief ray misses, a
        # density or transmission that varies across it, an absorbing plane crossed beside it, an earlier conversion out of
        # scope: the power entering is its chief ray's, not the light's, so no conversion is computed from it)
        scoped_by = pk.scope_behind()
        scope_reason = (f"the fundamental's power entering the crystal is out of model scope behind {', '.join(scoped_by)} "
                        "(power_out_of_scope_at): the power that enters is not computed, and the harmonic grows with its square"
                        ) if scoped_by else None
        if converting_W <= 0.0:  # (none of the fundamental is in the converting waves: nothing is generated)
            if scope_reason is not None:
                return refuse(scope_reason)
            self.out.nonlinear_conversions.append(NonlinearConversionResult(status=Status.PASS, harmonic_power_W=0.0,
                                                                            conversion_efficiency=0.0, **base))
            return transmitted, False, ()
        if n_lines > 1 or pk.src.polarization_component is not None:
            return refuse("the fundamental is a mixture of mutually incoherent parts (spectral lines or polarization components of "
                          "one source): the harmonic of a mixture is not the sum of its parts' harmonics, and sum-frequency mixing "
                          "between lines is not modelled")
        if pk.chief_obstructed or pk.sampling is not None or not pk.gstate.valid:
            return refuse("the fundamental beam has no valid Gaussian state here (the conversion is computed for a Gaussian beam)")
        if max(pk.gstate.m2x, pk.gstate.m2y) > 1.0 + cfg.nonlinear_beam_quality_tol:
            return refuse(f"the fundamental's M^2 ({max(pk.gstate.m2x, pk.gstate.m2y):.4g}) exceeds 1 + "
                          f"{cfg.nonlinear_beam_quality_tol:g}: the conversion is computed for a TEM00 beam")
        # --- conditions under which the harmonic is generated but its power is not the model's (the beam is still traced)
        reasons = [] if scope_reason is None else [scope_reason]
        if pk.spectral_index >= n_lines:
            reasons.append(f"the light entering is the second harmonic of {pk.src.source_id}#{pk.spectral_index - n_lines} generated "
                           "upstream: a cascaded conversion (with its fundamental possibly entering too: sum-frequency mixing, and the "
                           "coherent addition of the harmonics) is not modelled")
        if pk.harmonic_at:
            reasons.append(f"this light already generated a second harmonic at {', '.join(pk.harmonic_at)} that travels with it: "
                           "the harmonic generated here adds to it coherently (cascaded crystals, a second pass), which is not modelled")
        pulse = getattr(pk.src, "pulse", None)
        if pulse is not None:
            reasons.append(f"the source is pulsed ({pulse[1]:.3g} s pulses at {pulse[0]:.3g} Hz): the conversion of a pulse train "
                           "follows its peak power and pulse shape (and, for short pulses, its bandwidth and the group-velocity "
                           "mismatch), not the average power the continuous-wave model uses")
        k_a, wave_a, j_a = by_mode[modes[0]]
        _k_b, wave_b, _j_b = by_mode[modes[1]]
        # --- the harmonic wave and the phase mismatch across the slab
        crystal_h, issue = self._crystal_at(medium, h_wavelength, location)
        if crystal_h is None:
            return refuse(f"the crystal's indices are not stated at the harmonic wavelength {h_wavelength:g} nm"
                          + (f" ({issue.message})" if issue is not None else ""))
        tol = cfg.zero_vector_tol
        root2 = math.sqrt(2.0)
        rows = np.arange(kt.shape[0])
        pupil = (rows >= 1) & (rows <= pk.pupil_total)
        kt_h = np.where(pupil[:, None], kt[0] + (kt - kt[0]) / root2, kt)
        waves_h = uniaxial.outgoing(kt_h, normal, crystal_h, None, mode_h, True, s_rows, tol)
        if not bool(waves_h.ok[0]):
            return refuse("no harmonic wave propagates into the crystal at this angle")
        k0 = 2.0 * math.pi / (pk.wavelength_nm * NM_TO_MM)
        n_hat = normal[0]

        def across(waves: Waves) -> float:
            return float(dot(waves.K[0], n_hat))

        def grating(temperature: float | None) -> float:
            if spec.qpm_period_um is None:
                return 0.0
            period = spec.qpm_period_um
            if state is not None:
                t = state.temperature_c if temperature is None else temperature
                period = period * (1.0 + state.thermal_expansion_per_K * (t - state.reference_temperature_c))
            return shg.grating_wavenumber_per_mm(period, spec.qpm_order)

        dk_z = k0 * (2.0 * across(waves_h) - across(wave_a) - across(wave_b)) - grating(None)
        phase = dk_z * thickness / 2.0
        k_drive = wave_a.K[0] + wave_b.K[0]
        k_hat = k_drive / math.sqrt(float(dot(k_drive, k_drive)))
        path = thickness / float(dot(k_hat, n_hat))
        dk_path = 2.0 * phase / path

        # --- walk-off (per unit path, transverse to the driving wave normal) and the Gaussian beam in the crystal
        def drift(waves: Waves) -> Array:
            ray = waves.S[0]
            return ray / float(dot(ray, k_hat)) - k_hat

        s_a, s_b, s_h = drift(wave_a), drift(wave_b), drift(waves_h)
        walk_vector = (s_a + s_b) / 2.0 - s_h
        g_a, _per_mm = self._crystal_gaussian(pk, surf, wave_a, crystal, None, error_mm)
        if not g_a.valid:
            return refuse(f"the fundamental beam has no valid Gaussian state in the crystal ({g_a.reason_text})")
        axes = (g_a.bx, g_a.by)
        walk = (float(dot(walk_vector, axes[0])), float(dot(walk_vector, axes[1])))
        pair = (float(dot(s_a - s_b, axes[0])), float(dot(s_a - s_b, axes[1]))) if two_waves else (0.0, 0.0)
        rayleigh = (g_a.rayleigh("x"), g_a.rayleigh("y"))
        offset = (g_a.waist_distance("x"), g_a.waist_distance("y"))
        waist = (g_a.waist_radius("x"), g_a.waist_radius("y"))
        walk_size = math.sqrt(float(dot(walk_vector, walk_vector)))
        walkoff_mrad = float(exact_atan2(np.array([walk_size]), np.array([1.0]))[0]) * RAD_TO_MRAD
        # --- the coefficient: stated, or the d tensor contracted with the traced waves' electric fields
        if spec.d_tensor_pm_per_V is not None:
            d_eff = shg.d_eff_from_tensor(spec.d_tensor_pm_per_V, nl.frame, waves_h.field_axis()[0], wave_a.field_axis()[0],
                                          wave_b.field_axis()[0])
        else:
            d_eff = float(spec.d_eff_pm_per_V)
        if spec.qpm_period_um is not None:
            d_eff = d_eff * 2.0 / (spec.qpm_order * math.pi)
        n_a = math.sqrt(float(dot(wave_a.K[0], wave_a.K[0])))
        n_b = math.sqrt(float(dot(wave_b.K[0], wave_b.K[0])))
        n_h = math.sqrt(float(dot(waves_h.K[0], waves_h.K[0])))
        coupling = shg.coupling_per_W(pk.wavelength_nm, d_eff, n_a, n_b, n_h)
        evaluator = self.library.evaluator(medium)
        unstated = self.library.loss_unstated(medium)
        missing_loss = tuple(wavelength for wavelength in (pk.wavelength_nm, h_wavelength)
                             if unstated or not evaluator.loss_established(wavelength))
        if missing_loss and coupling > 0.0:
            if reasons:
                return refuse("; ".join(reasons))
            wavelengths = ", ".join(f"{wavelength:g} nm" for wavelength in missing_loss)
            reason = (f"bulk absorption in {surf.owner_id} is not established at {wavelengths}; it changes the coherent "
                      "addition of the harmonic generated along the crystal and the power drawn from the pump")
            unknown_absorption(reason, missing_loss)
            return transmitted, False, modes
        # the crystal's attenuation along each wave's ray (as the tracer applies it: exp(-alpha x ray length) in power)
        # (a spectral loss: the fundamental's and the harmonic's own coefficients; a scalar one: the same for both)
        alpha = self._extinction(pk, medium)
        alpha_h = self._extinction(pk, medium, h_wavelength) if evaluator.spectral_loss else alpha
        absorption = (0.0, 0.0)
        transmit_h = 1.0
        if alpha > 0.0 or alpha_h > 0.0:
            def ray_length(waves: Waves) -> float:
                return thickness / abs(float(dot(waves.S[0], n_hat)))

            length_h = ray_length(waves_h)
            absorption = (alpha * (ray_length(wave_a) + ray_length(wave_b)) / (2.0 * path), alpha_h * length_h / (2.0 * path))
            amplitude_h = float(exact_exp(np.array([-0.5 * alpha_h * length_h]))[0])
            transmit_h = amplitude_h * amplitude_h
        absorbing = absorption != (0.0, 0.0)
        focused = spec.focusing == "boyd_kleinman"
        rule = self.quad_rule_nl if self.quad_rule_nl is not None else gauss_legendre_rule(shg.PANEL_NODES)
        tol_q = cfg.nonlinear_quadrature_rel_tol
        overlap = shg.overlap_factor(path, dk_path, rayleigh, offset, waist, walk, pair, focused, rule, tol_q, absorption)
        focused_here = focused  # (the result taken: the focused one in boyd_kleinman, or near a tuning-curve zero)
        near_error = None
        if not focused:
            exact = shg.overlap_factor(path, dk_path, rayleigh, offset, waist, walk, pair, True, rule, tol_q, absorption)
            # (relative to the larger of the focused value and the focused value at phase matching: near a zero of the tuning
            # curve the harmonic is a negligible part of what the crystal gives)
            peak = exact.value if dk_path == 0.0 else shg.overlap_factor(path, 0.0, rayleigh, offset, waist, walk, pair, True, rule,
                                                                         tol_q, absorption).value
            if exact.value > 0.0 and exact.value >= peak:
                near_error = abs(overlap.value / exact.value - 1.0)
            elif peak > 0.0:
                near_error = abs(overlap.value - exact.value) / peak
                if near_error <= cfg.nonlinear_focusing_rel_tol and \
                        abs(overlap.value - exact.value) > cfg.nonlinear_focusing_rel_tol * exact.value:
                    # near a zero of the tuning curve: the near-field value is close to the focused one against what the crystal
                    # gives at phase matching, but not against itself -- there the focusing fills the zero, and the harmonic is
                    # the focused result (the near-field one would be graded as a value it is not)
                    focused_here = True
                    overlap = exact
        power_a = p_a if two_waves else converting_W
        power_b = p_b if two_waves else converting_W
        harmonic_W = shg.harmonic_power_W(coupling, power_a, power_b, two_waves, path, waist[0], waist[1], overlap.value)
        exit_W = harmonic_W
        if absorbing:  # (the depletion estimate and power check use the lossless reference model)
            clear = shg.overlap_factor(path, dk_path, rayleigh, offset, waist, walk, pair, focused_here, rule, tol_q)
            harmonic_W = shg.harmonic_power_W(coupling, power_a, power_b, two_waves, path, waist[0], waist[1], clear.value)
        depletion = shg.depletion_estimate(harmonic_W / converting_W, power_a, power_b, two_waves)
        quadrature = overlap.quadrature_error / overlap.value if overlap.value > 0.0 else 0.0
        plane = shg.plane_factor(phase)
        angular, temperature_w, spectral = self._acceptance(medium, k_hat, modes, mode_h, path, pk.wavelength_nm, grating)
        b_min = 2.0 * min(rayleigh)
        coherence = line_coherence_length_mm(pk.src, pk.spectral_index)
        if coherence is not None:  # a line of stated width: sinc^2 averaged over it against its centre's
            fwhm = shg.COHERENCE_PER_LINEWIDTH * pk.wavelength_nm * pk.wavelength_nm * NM_TO_MM / coherence
            average = self._line_average(medium, k_hat, modes, mode_h, path, pk.wavelength_nm, grating, phase, fwhm, rule)
            if average is None:
                reasons.append(f"the crystal's indices are not stated over the fundamental line's width ({fwhm:.3g} nm FWHM), so "
                               "its effect on the phase matching is not known")
            else:
                # (relative to the larger of the two: the centre's value is graded, so near a zero of the tuning curve the line's
                # average must agree with it, not merely be small against the phase-matched value. Where the near-field run reports
                # the focused value that fills a zero (``focused_here``) that value is graded, not the near-field sinc^2 there: its
                # scale is its share of the focused value at phase matching, in the same units as the sinc^2 average)
                centre = plane
                if focused_here and not focused and peak > 0.0:
                    centre = max(plane, overlap.value / peak)
                spread = abs(average - plane) / max(centre, average) if max(centre, average) > 0.0 else 0.0
                if spread > cfg.nonlinear_linewidth_rel_tol:
                    reasons.append(f"the fundamental line's width ({fwhm:.3g} nm FWHM; spectral acceptance "
                                   + ("not defined" if spectral is None else f"{spectral:.3g} nm")
                                   + f") changes the phase-matching factor sinc^2(dk L / 2) averaged over the line by {spread:.3g}, "
                                   f"more than {cfg.nonlinear_linewidth_rel_tol:g}: the conversion is computed for one frequency")
        if depletion > cfg.nonlinear_depletion_rel_tol:
            reasons.append(f"the undepleted-pump result overstates the harmonic by an estimated {depletion:.3g} of it, more than "
                           f"{cfg.nonlinear_depletion_rel_tol:g} (conversion efficiency {harmonic_W / total_W:.3g}): pump depletion "
                           "is not modelled")
        if near_error is not None and near_error > cfg.nonlinear_focusing_rel_tol:
            reasons.append(f"the beam's radius changes along the crystal (L/b = {path / b_min:.3g}): the near-field result differs "
                           f"from the focused one by {near_error:.3g}, more than {cfg.nonlinear_focusing_rel_tol:g} (boyd_kleinman "
                           "focusing follows it for one fundamental wave)")
        if quadrature > cfg.nonlinear_focusing_rel_tol:
            reasons.append(f"the overlap integral did not converge (its last two values differ by {quadrature:.3g} of it)")
        available = converting_W if not two_waves else 2.0 * min(p_a, p_b)
        if harmonic_W >= available:
            reasons.append("the undepleted-pump harmonic power reaches the power of the converting fundamental")
            harmonic_W = available / 2.0
            exit_W = min(exit_W, harmonic_W)
        if absorbing:  # (the power the traced harmonic starts with: it reaches the exit face with exit_W)
            harmonic_W = min(exit_W / transmit_h, available / 2.0)
        efficiency = exit_W / total_W
        # --- energy: the converting waves give up the harmonic power (two waves: half each, one photon of each per pair)
        updated = list(transmitted)
        if two_waves:
            for mode in modes:
                k, waves, jones = by_mode[mode]
                p = power(jones)
                updated[k] = (waves, jones * math.sqrt(max(p - harmonic_W / 2.0, 0.0) / p))
        else:
            updated[k_a] = (wave_a, j_a * math.sqrt(max(p_a - harmonic_W, 0.0) / p_a))
        self._flow(pk, SINK_NONLINEAR_CONVERSION, harmonic_W, surf, "second_harmonic")
        self._branch_power += harmonic_W  # (booked here: not a loss of the interface)
        if reasons:
            report(reasons)
        # --- the harmonic beam
        child_id = self._harmonic_child(pk, surf, medium, h_index, h_wavelength, harmonic_W, crystal_h, waves_h, g_a, walk_vector,
                                        path, normal, pupil)
        entry = self._generated.setdefault((pk.src.source_id, h_index),
                                           [pk.src.owner_id, h_wavelength, 0.0, f"{pk.src.source_id}#{pk.spectral_index}"])
        entry[2] += harmonic_W
        scoped = bool(reasons)
        self.out.nonlinear_conversions.append(NonlinearConversionResult(
            status=Status.OUT_OF_MODEL_SCOPE if scoped else Status.PASS, reason="; ".join(reasons) or None,
            harmonic_path_id=child_id, harmonic_power_W=None if scoped else exit_W,
            conversion_efficiency=None if scoped else efficiency, d_eff_pm_per_V=d_eff, delta_k_per_mm=dk_z,
            phase_mismatch_rad=phase, plane_wave_factor=plane, overlap_factor=overlap.value, focusing_parameter=path / b_min,
            walkoff_mrad=walkoff_mrad, depletion_estimate=depletion, near_field_error=near_error, quadrature_error=quadrature,
            angular_acceptance_mrad=angular, temperature_acceptance_K=temperature_w, spectral_acceptance_nm=spectral,
            path_length_mm=path, **base))
        return updated, harmonic_W > 0.0, ()

    def _harmonic_child(self, pk: _Packet, surf: SurfaceG, medium: str, index: int, wavelength_nm: float, power_W: float,
                        crystal: Crystal, waves: Waves, g_f: GState, walk: Array, path: float, normal: Array,
                        pupil: NDArray[np.bool_]) -> str | None:
        """Queue the harmonic beam generated at ``surf`` (``_second_harmonic``) and return its path id (``None``: the trace
        budget refused it; its power is booked to the harmonic line's ``PATH_LIMIT``). Its rays are the fundamental's, with the
        pupil rays drawn in toward the chief ray by ``1/sqrt 2`` in position and in transverse wave vector (the harmonic of a
        Gaussian beam is a Gaussian beam of the same reduced q-parameter and ``1/sqrt 2`` of its radius; a pupil ray's optical
        path from the chief ray's scales by ``1/2`` on that wavefront); its rays start on the line of the harmonic's power
        centroid (half the walk-off across the crystal from the driving beam's ray); its Gaussian state is the fundamental's
        reduced q-parameter with ``M^2 = 1``, unless walk-off widens the harmonic beyond ``nonlinear_harmonic_gaussian_rel_tol``
        (``HARMONIC_BEAM_NOT_GAUSSIAN``)."""
        cfg = self.cfg
        if not self.schedule.reserve_branch(pk.path_id, pk.src.source_id):
            self.out.power_flows.append(PowerFlow(SINK_PATH_LIMIT, power_W, pk.src.source_id, index, wavelength_nm, None,
                                                  surf.surface_id, surf.owner_id, "second_harmonic"))
            return None
        root_f = root_path_id(pk.src.source_id, pk.spectral_index, pk.src.polarization_component)
        root_h = root_path_id(pk.src.source_id, index, pk.src.polarization_component)
        child = pk.fork(child_path_id(root_h + pk.path_id[len(root_f):], root_h, pk.children), surf.surface_id, "second_harmonic")
        child.interactions = pk.interactions + 1
        pk.children += 1
        root2 = math.sqrt(2.0)
        o = np.where(pupil[:, None], pk.o[0] + (pk.o - pk.o[0]) / root2, pk.o)
        ray = waves.S[0]
        shift = walk * (path / 2.0)
        shift = shift - ray * (float(dot(shift, normal[0])) / float(dot(ray, normal[0])))
        child.o = o + shift
        child.opl = np.where(pupil, pk.opl[0] + (pk.opl - pk.opl[0]) / 2.0, pk.opl)
        alive = pk.alive & waves.ok
        child.d = np.where(alive[:, None], waves.S, pk.d)
        child.e1 = np.where(alive[:, None], waves.e1, pk.e1)
        child.e2 = np.where(alive[:, None], waves.e2, pk.e2)
        child.alive = alive
        child.spectral_index, child.wavelength_nm = index, wavelength_nm
        child.power0_W = power_W
        child.jones = uniaxial.mode_jones(waves, np.ones(waves.K.shape[0], dtype=np.complex128))
        child.medium, child.n = medium, float(waves.ray_index[0])
        index_h, per_mm = uniaxial.gaussian_index(crystal, waves, None)
        scale = index_h / g_f.n
        gstate = g_f.with_(wavelength_nm=wavelength_nm, n=index_h, qx=g_f.qx * scale, qy=g_f.qy * scale, m2x=1.0, m2y=1.0)
        widening = max((float(dot(walk, b)) * path / gstate.waist_radius(a)) ** 2 / 3.0 for a, b in (("x", gstate.bx), ("y", gstate.by)))
        if widening > cfg.nonlinear_harmonic_gaussian_rel_tol:
            gstate = gstate.invalidated(ErrorCode.HARMONIC_BEAM_NOT_GAUSSIAN, (
                f"walk-off spreads the harmonic generated along the crystal over a strip: its width grows by about {widening:.3g}, more "
                f"than {cfg.nonlinear_harmonic_gaussian_rel_tol:g}, so it is traced without a Gaussian state (its power is known)"))
        child.gstate = gstate
        child.gaussian_error_mm = 0.0
        child.crystal = _CrystalLeg(waves.mode, crystal, waves.K, waves.ray_index, uniaxial.ray_group(crystal, waves), per_mm,  # type: ignore[arg-type]
                                    float(pk.length[0]))
        if child.launch_spread is not None:
            child.launch_spread = (child.launch_spread[0] / root2, child.launch_spread[1] / root2)
        child.lineage.append(f"{surf.surface_id}:{self._crystal_event('second_harmonic', waves)}")
        if self.recorder is not None:
            self.recorder.packet_started(child, surf.surface_id)
            self.recorder.branch_spawned(pk, child)
        if self._spent(child):
            self._finish(child, ZERO_POWER, surf.surface_id)
        else:
            self._enqueue(child)
        return child.path_id

    def _do_mirror(self, pk: _Packet, surf: SurfaceG, h: Hits, inter: InteractionG) -> str | None:
        cfg = self.cfg
        from_pos, same_side = self._side(pk, surf, h)
        allowed = inter.reflective_side == "both" or (inter.reflective_side == "positive") == from_pos
        if not allowed:
            transmitted = self._back_transmittance(pk, surf, h, inter)
            if transmitted is not None:
                t, aoi = transmitted
                self._issue(make_issue(ErrorCode.OUT_OF_MODEL_SCOPE, Status.OUT_OF_MODEL_SCOPE,
                                       f"the beam meets the back of a mirror whose coating transmits T = {t:.6g} at {aoi:.6g} deg: the trace "
                                       "does not follow light through a mirror, so the beam ends here; the cavity analysis evaluates light fed "
                                       "into a cavity through its mirror ('os-optics project cavity' with --inject-power-w or --inject-source)",
                                       location=f"surface:{surf.surface_id}", cause=MIRROR_TRANSMISSION_NOT_TRACED,
                                       reflective_side=inter.reflective_side, transmittance=t, aoi_deg=aoi, path_id=pk.path_id,
                                       suggested_action="analyse the injected cavity with the cavity analysis, or turn the mirror's reflective "
                                                        "side toward the beam"))
            else:
                self._issue(make_issue(ErrorCode.BLOCKED_BY_BACK_SIDE, Status.WARNING,
                                       "the beam hits the non-reflective side of a mirror and is blocked",
                                       location=f"surface:{surf.surface_id}", reflective_side=inter.reflective_side,
                                       suggested_action="flip the mirror normal or change the geometry"))
            self._end_event(pk, "blocked_back_side")
            pk.alive[:] = False
            return BLOCKED_BACK
        pk.alive = pk.alive & same_side
        d_out, n_or, cos_i = reflect_dir(pk.d, h.normal, cfg.zero_vector_tol)
        aoi_deg = incidence_deg(cos_i, pk.d, h.normal)
        term = self._aoi_guard(pk, surf, aoi_deg)
        if term:
            return term
        if inter.coating is not None:
            cc = self._live_coating(pk, surf, inter.coating, aoi_deg)
            if cc is None:
                return COATING_RANGE
            rs, rp = np.sqrt(cc.rs), np.sqrt(cc.rp)
            self._twins_polarized(pk, surf, max(abs(float(cc.rs[0]) - float(cc.rp[0])), abs(float(cc.ts[0]) - float(cc.tp[0]))),
                                  "a coated mirror")
        else:
            rs = rp = np.ones(pk.o.shape[0])
        ph = self._stated_phases(pk, surf, inter)
        coef_s, coef_p = mirror_reflection(rs, rp, inter.phase_s, inter.phase_p, ph)
        s = s_axis(pk.d, n_or, pk.e1, cfg.zero_vector_tol)
        state = self._sp_state(pk, s, float(aoi_deg[0])) if inter.coating is not None else None
        if inter.coating is not None:
            t_s0, t_p0 = mirror_transmission(np.sqrt(cc.ts[:1]), np.sqrt(cc.tp[:1]), ph)
            j_t0 = apply_thin_transmission(pk.jones[:1], pk.e1[:1], pk.e2[:1], pk.d[:1], s[:1], t_s0, t_p0)
            self._split[SINK_UNTRACED_TRANSMISSION] = pk.power0_W * power2(j_t0[0])

            def transmitted_rays(s: Array = s, ts: Array = cc.ts, tp: Array = cc.tp, jones: CArray = pk.jones, e1: Array = pk.e1,
                                 e2: Array = pk.e2, d: Array = pk.d) -> tuple[Array, Array, Array, CArray]:
                t_s, t_p = mirror_transmission(np.sqrt(ts), np.sqrt(tp), ph)
                return d, e1, e2, apply_thin_transmission(jones, e1, e2, d, s, t_s, t_p)

            # the coating's transmission goes on through the reflector (as thin, like a splitter's transmitted beam)
            self._branch(pk, SINK_UNTRACED_TRANSMISSION, surf, pk.d[0], pk.gstate, self._split[SINK_UNTRACED_TRANSMISSION],
                         jones=j_t0[0], e1=pk.e1[0], e2=pk.e2[0], rays=transmitted_rays,
                         delay=None if self._delay_stated(inter, "transmission") else
                         self._delay(pk, surf, "coating_transmission", float(aoi_deg[0]), from_pos, state=state,
                                     inter=inter))
        j_new, e1_new, e2_new = apply_interface(pk.jones, pk.e1, pk.e2, pk.d, s, coef_s, coef_p, d_out)
        conc: float | None = None
        if surf.geometry == "sphere":
            ahead = float(dot((surf.center - h.point[0]), pk.d[0])) > 0.0  # centre beyond the surface: convex toward the beam
            conc = -surf.radius if ahead else surf.radius
        pk.gstate = apply_curved_or_flat_interface(pk.gstate, kind="mirror", d_in=pk.d[0], d_out=d_out[0], n_or=n_or[0],
                                                   cos_i=float(cos_i[0]), n1=pk.n, n2=pk.n, cos_t=float(cos_i[0]), tir=False,
                                                   radius_signed_mm=conc, cfg=cfg,
                                                   curvature=self._local_curvature(surf, h, pk.d[0], "mirror"))
        self._report_gaussian_state(pk, surf)
        pk.jones, pk.e1, pk.e2, pk.d = j_new, e1_new, e2_new, np.where(pk.alive[:, None], d_out, pk.d)
        self._end_event(pk, "reflect", float(aoi_deg[0]))
        pk.lineage.append(f"{surf.surface_id}:reflect")
        if not self._delay_stated(inter, "reflection"):
            pk.delays = (*pk.delays, self._delay(pk, surf, "reflect", float(aoi_deg[0]), from_pos, state=state, inter=inter))
        if self._spent(pk):
            return ZERO_POWER
        return None

    def _do_thin_lens(self, pk: _Packet, surf: SurfaceG, h: Hits, inter: InteractionG) -> str | None:
        cfg = self.cfg
        axis = surf.normal
        focal_length, transmittance = inter.focal_length, inter.transmittance
        if inter.table is not None:  # a tabulated thin lens: this line's focal length and transmittance
            focal_length, transmittance, in_domain = response_at(inter.table, pk.wavelength_nm)
            if not in_domain:
                taken = " and ".join(nearest_row_quantities(inter.table, pk.wavelength_nm))
                self._table_issue(pk, surf, ErrorCode.MODEL_WAVELENGTH_OUTSIDE_BAND,
                                  f"{pk.wavelength_nm!r} nm light meets the tabulated thin lens {surf.surface_id}, whose response is "
                                  f"characterized for {inter.table.domain_text()}; the nearest row's {taken} "
                                  f"{'were' if ' and ' in taken else 'was'} used. Use a source wavelength inside that domain, or a "
                                  "response model that covers it.",
                                  wavelength_nm=pk.wavelength_nm)
        d_in0 = pk.d[0].copy()
        d_out, ok = thin_lens_dir(h.point, surf.origin, axis, pk.d, focal_length, cfg.parallel_dot_tol)
        cos_i = np.abs(dot(pk.d, axis))
        term = self._aoi_guard(pk, surf, incidence_deg(cos_i, pk.d, axis))
        if term:
            return term
        if not bool(ok[0]):
            self._issue(make_issue(ErrorCode.PARALLEL_RAY, Status.OUT_OF_MODEL_SCOPE,
                                   f"beam {pk.path_id} travels parallel to the thin-lens plane {surf.surface_id}: its light there is not traced",
                                   location=f"surface:{surf.surface_id}", path_id=pk.path_id))
            pk.alive[:] = False
            return GRAZING
        pk.alive = pk.alive & ok
        aoi0 = float(incidence_deg(float(cos_i[0]), pk.d[0], axis))
        if inter.table is not None:
            self._check_table_domain(pk, surf, inter.table, aoi0)
        e1n, e2n = parallel_transport_basis(pk.e1, pk.e2, pk.d, d_out, cfg.antiparallel_tol)
        pk.gstate = apply_thin_lens(pk.gstate, pk.d[0], d_out[0], float(cos_i[0]), focal_length, cfg, normal=axis)
        self._report_gaussian_state(pk, surf)
        if inter.table is not None and transmittance < 1.0:
            self._split[SINK_UNMODELLED_RESPONSE] = pk.chief_power() * (1.0 - transmittance)
            self._split_detail = "tabulated thin lens: response not delivered into the focused beam (1 - transmittance)"
        pk.jones = pk.jones * math.sqrt(transmittance)
        pk.e1, pk.e2, pk.d = e1n, e2n, np.where(pk.alive[:, None], d_out, pk.d)
        pk.opl_undefined = True
        self._end_event(pk, "thin_lens", aoi0)
        pk.lineage.append(f"{surf.surface_id}:thin_lens")
        pk.delays = (*pk.delays, self._delay(pk, surf, "thin_lens", aoi0, float(dot(d_in0, axis)) < 0.0,
                                             (float(h.u[0]), float(h.v[0]))))
        return ZERO_POWER if self._spent(pk) else None

    def _table_issue(self, pk: _Packet, surf: SurfaceG, code: ErrorCode, message: str, **context: Any) -> None:
        """One validity-domain issue of a tabulated thin lens (``OUT_OF_MODEL_SCOPE``), once per path, surface and code."""
        key = (pk.path_id, surf.surface_id, code.value)
        if key in self._table_reported:
            return
        self._table_reported.add(key)
        self._issue(make_issue(code, Status.OUT_OF_MODEL_SCOPE, message, location=f"surface:{surf.surface_id}", path_id=pk.path_id,
                               suggested_action="keep the light inside the response model's validated domain, or use a response "
                                                "model validated for these conditions", **context))

    def _check_table_domain(self, pk: _Packet, surf: SurfaceG, table: ThinLensTable, field_angle_deg: float) -> None:
        """The validity domain of a tabulated thin lens at the chief ray's arrival: field angle, beam size, polarization."""
        if table.max_field_angle_deg is not None and beyond_limit(field_angle_deg, table.max_field_angle_deg, DOMAIN_LIMIT_ABS_TOLERANCE_DEG):
            self._table_issue(pk, surf, ErrorCode.MODEL_FIELD_ANGLE_EXCEEDED,
                              f"the chief ray meets the tabulated thin lens {surf.surface_id} at {field_angle_deg:.6g} deg from its "
                              f"axis; its response is validated up to {table.max_field_angle_deg:.6g} deg. Keep the field angle "
                              "within that limit, or use a response model validated for it.",
                              field_angle_deg=field_angle_deg, max_field_angle_deg=table.max_field_angle_deg)
        if table.max_beam_radius_mm is not None:
            radius = max(pk.gstate.radius("x"), pk.gstate.radius("y")) if pk.gstate.valid else None
            if radius is None:
                self._table_issue(pk, surf, ErrorCode.MODEL_BEAM_SIZE_EXCEEDED,
                                  f"the beam arriving at the tabulated thin lens {surf.surface_id} has no valid Gaussian state "
                                  f"({pk.gstate.reason_text or 'unknown'}), so its size cannot be checked against the validated "
                                  f"{table.max_beam_radius_mm:.6g} mm", max_beam_radius_mm=table.max_beam_radius_mm)
            elif beyond_limit(radius, table.max_beam_radius_mm, DOMAIN_LIMIT_ABS_TOLERANCE_MM):
                self._table_issue(pk, surf, ErrorCode.MODEL_BEAM_SIZE_EXCEEDED,
                                  f"the beam arriving at the tabulated thin lens {surf.surface_id} has a 1/e^2 radius of {radius:.6g} mm; "
                                  f"its response is validated up to {table.max_beam_radius_mm:.6g} mm. Use a smaller beam at the lens, "
                                  "or a response model validated for that beam size.",
                                  beam_radius_mm=radius, max_beam_radius_mm=table.max_beam_radius_mm)
        if table.validated_polarizations is not None and table.u_axis is not None and table.v_axis is not None:
            fractions = orthogonal_fractions(pk.jones[0], pk.e1[0], pk.e2[0], pk.d[0], table.u_axis, table.v_axis)
            if not any(fractions[name] <= POLARIZATION_MATCH_FRACTION for name in table.validated_polarizations):
                names = {"linear_u": "linear along the element's local X", "linear_v": "linear along the element's local Y"}
                wanted = " or ".join(names[n] for n in table.validated_polarizations)
                self._table_issue(pk, surf, ErrorCode.MODEL_POLARIZATION_NOT_VALIDATED,
                                  f"the light arriving at the tabulated thin lens {surf.surface_id} is not polarized {wanted}, the "
                                  "polarization its response was validated for. Use light polarized that way, or a response model "
                                  "validated for the arriving polarization.",
                                  orthogonal_power_fraction=min(fractions[n] for n in table.validated_polarizations))

    def _do_splitter(self, pk: _Packet, surf: SurfaceG, h: Hits, inter: InteractionG) -> str | None:
        cfg = self.cfg
        assert inter.coating is not None
        from_pos, same_side = self._side(pk, surf, h)
        pk.alive = pk.alive & same_side
        d_ref, n_or, cos_i = reflect_dir(pk.d, h.normal, cfg.zero_vector_tol)
        aoi_deg = incidence_deg(cos_i, pk.d, h.normal)
        term = self._aoi_guard(pk, surf, aoi_deg)
        if term:
            return term
        cc = self._live_coating(pk, surf, inter.coating, aoi_deg)
        if cc is None:
            return COATING_RANGE
        ph = self._stated_phases(pk, surf, inter)
        s = s_axis(pk.d, n_or, pk.e1, cfg.zero_vector_tol)
        state = self._sp_state(pk, s, float(aoi_deg[0]))
        # A physical reflection: in the (s, d_out x s) triad r_p = -r_s (the ideal metal mirror's (pi, 0) and Fresnel at normal
        # incidence), so the splitter reflects like a mirror (no hidden half-wave plate). For a non-polarizing coating
        # (R_s = R_p, T_s = T_p) the result then never depends on the arbitrary s axis (e1) at normal incidence; a polarizing
        # coating (R_s != R_p or T_s != T_p) there still does: which field component it treats as s is that internal axis
        # (the network export refuses it, POLARIZING_NORMAL_INCIDENCE). reflection_phase is the phase of r_s relative to t; the
        # 2 x 2 splitter stays unitary for each polarization (t r* + r t* = 0 for r = +-i sqrt(R)) on both sides.
        # stated phases, when given, replace these in the same (s, d_out x s) triad
        t_s, t_p, r_s, r_p = splitter_amplitudes(cc.rs, cc.rp, cc.ts, cc.tp, inter.reflection_phase, inter.transmission_phase, ph)
        self._twins_polarized(pk, surf, max(abs(float(cc.rs[0]) - float(cc.rp[0])), abs(float(cc.ts[0]) - float(cc.tp[0]))),
                              "a beam splitter")
        d_in = pk.d.copy()
        # reflected branch
        j_ref, e1_ref, e2_ref = apply_interface(pk.jones, pk.e1, pk.e2, d_in, s, r_s, r_p, d_ref)
        reflected_power = pk.power0_W * power2(j_ref[0])
        self._branch_power += reflected_power
        child = self._spawn(pk, surf, "beam_splitter_reflection", reflected_power)
        if child is not None:
            child.d, child.jones, child.e1, child.e2 = d_ref, j_ref, e1_ref, e2_ref
            child.gstate = apply_curved_or_flat_interface(
                pk.gstate, kind="mirror", d_in=d_in[0], d_out=d_ref[0], n_or=n_or[0], cos_i=float(cos_i[0]), n1=pk.n, n2=pk.n,
                cos_t=float(cos_i[0]), tir=False, radius_signed_mm=None, cfg=cfg,
                curvature=self._local_curvature(surf, h, d_in[0], "mirror"))
            child.lineage.append(f"{surf.surface_id}:split_reflect")
            if not self._delay_stated(inter, "reflection"):
                child.delays = (*child.delays, self._delay(pk, surf, "split_reflect", float(aoi_deg[0]), from_pos, state=state,
                                                           inter=inter))
            if self.recorder is not None:
                self.recorder.branch_spawned(pk, child)
            if not self._spent(child):
                self._enqueue(child)
            else:
                self._finish(child, ZERO_POWER, surf.surface_id)
        # transmitted branch continues in this packet
        pk.jones = apply_thin_transmission(pk.jones, pk.e1, pk.e2, d_in, s, t_s, t_p)
        pk.gstate = apply_flat(pk.gstate, d_in[0], d_in[0], cfg)
        self._report_gaussian_state(pk, surf)
        self._end_event(pk, "split_transmit", float(aoi_deg[0]))
        pk.lineage.append(f"{surf.surface_id}:split_transmit")
        if not self._delay_stated(inter, "transmission"):
            pk.delays = (*pk.delays, self._delay(pk, surf, "split_transmit", float(aoi_deg[0]), from_pos, state=state,
                                                 inter=inter))
        return ZERO_POWER if self._spent(pk) else None

    def _oblique_warn(self, pk: _Packet, surf: SurfaceG, aoi: float, what: str) -> None:
        if aoi > self.cfg.retarder_warn_aoi_deg:
            self._issue(make_issue(ErrorCode.RETARDER_OBLIQUE_INCIDENCE, Status.WARNING,
                                   f"{what} at {aoi:.3g} deg incidence: thin-element polarization model loses accuracy",
                                   location=f"surface:{surf.surface_id}", aoi_deg=aoi, warn_deg=self.cfg.retarder_warn_aoi_deg))

    def _axis_components(self, pk: _Packet, surf: SurfaceG, inter: InteractionG) -> tuple[Array, Array] | None:
        assert inter.axis is not None
        a1, a2, ok = project_axis(inter.axis, pk.d, pk.e1, pk.e2, self.cfg.zero_vector_tol)
        if not bool(ok[0]):
            self._issue(make_issue(ErrorCode.DEGENERATE_SURFACE, Status.FAIL, "element axis is parallel to the ray direction",
                                   location=f"surface:{surf.surface_id}"))
            return None
        pk.alive = pk.alive & ok
        return a1, a2

    def _do_polarizer(self, pk: _Packet, surf: SurfaceG, h: Hits, inter: InteractionG) -> str | None:
        aoi = float(incidence_deg(abs(float(dot(pk.d[0], surf.normal))), pk.d[0], surf.normal))
        self._oblique_warn(pk, surf, aoi, "polarizer")
        comps = self._axis_components(pk, surf, inter)
        if comps is None:
            pk.alive[:] = False
            return DEGENERATE
        a1, a2 = comps
        t_par, t_perp = inter.transmittances(pk.wavelength_nm)
        self._twins_polarized(pk, surf, abs(t_par - t_perp), "a polarizer")
        # (light it passes along its blocking axis too may be delayed differently from that along its axis)
        state = self._axis_state(pk, a1, a2) if t_perp > 0.0 else None
        field_along = complex(a1[0]) * complex(pk.jones[0, 0]) + complex(a2[0]) * complex(pk.jones[0, 1])  # (plain Python: no FMA)
        along = pk.power0_W * (field_along.real * field_along.real + field_along.imag * field_along.imag)
        chief = pk.chief_power()
        # (light along the axis, or across it, leaves the other share as a small difference of two powers: rounding of a few
        # 1e-19 W of a 1 mW beam that follows the poses' last bits. Resolved to the ledger's resolution of such a difference,
        # power_ledger_fraction_resolution of the beam's power, it is 0, not a loss)
        resolution = self.cfg.power_ledger_fraction_resolution * chief
        if chief - along <= resolution:
            along, across = chief, 0.0
        elif along <= resolution:
            along, across = 0.0, chief
        else:
            across = chief - along
        self._split[SINK_POLARIZER_REJECTION] = across * (1.0 - t_perp)
        self._split[SINK_ELEMENT_TRANSMISSION_LOSS] = along * (1.0 - t_par)
        pk.jones = polarizer(pk.jones, comps[0], comps[1], t_par, t_perp)
        pk.gstate = apply_flat(pk.gstate, pk.d[0], pk.d[0], self.cfg)
        self._end_event(pk, "polarizer", aoi)
        pk.lineage.append(f"{surf.surface_id}:polarizer")
        pk.delays = (*pk.delays, self._delay(pk, surf, "polarizer", aoi, float(dot(pk.d[0], surf.normal)) < 0.0,
                                             state=state))
        return ZERO_POWER if self._spent(pk) else None

    def _do_retarder(self, pk: _Packet, surf: SurfaceG, h: Hits, inter: InteractionG) -> str | None:
        aoi = float(incidence_deg(abs(float(dot(pk.d[0], surf.normal))), pk.d[0], surf.normal))
        self._oblique_warn(pk, surf, aoi, "retarder")
        comps = self._axis_components(pk, surf, inter)
        if comps is None:
            pk.alive[:] = False
            return DEGENERATE
        if inter.retardance_table is not None:
            stated = inter.retardance_table.at(pk.wavelength_nm)
            if stated is None:
                bound = pk.chief_power() if pk.power_bound_W is None else pk.power_bound_W
                self._issue(make_issue(ErrorCode.MODEL_WAVELENGTH_OUTSIDE_BAND, Status.OUT_OF_MODEL_SCOPE,
                                       f"{pk.wavelength_nm!r} nm light meets the retarder {surf.surface_id}, whose retardance is stated for "
                                       f"{inter.retardance_table.domain_text()}: its power and polarization behind the element are not "
                                       f"computed, at most {bound:.4g} W",
                                       location=f"surface:{surf.surface_id}", path_id=pk.path_id, wavelength_nm=pk.wavelength_nm,
                                       power_upper_bound_W=bound,
                                       suggested_action="use a source wavelength inside that domain, or state the retardance at this wavelength"))
                if self.bounding:
                    self._end_event(pk, "retarder", aoi)
                    pk.alive[:] = False
                    return RETARDANCE_UNAVAILABLE
                return self._pass_not_computed(pk, surf, "retarder", aoi, RETARDANCE_UNAVAILABLE)
            delta = stated
        else:
            delta = inter.retardance if inter.achromatic else inter.retardance * inter.design_wavelength_nm / pk.wavelength_nm
        state = self._axis_state(pk, comps[0], comps[1])  # (its fast and slow axes' delays differ by an unstated amount)
        pk.jones = retarder(pk.jones, comps[0], comps[1], delta) * math.sqrt(inter.transmittances(pk.wavelength_nm)[0])
        pk.gstate = apply_flat(pk.gstate, pk.d[0], pk.d[0], self.cfg)
        self._end_event(pk, "retarder", aoi)
        pk.lineage.append(f"{surf.surface_id}:retarder")
        pk.delays = (*pk.delays, self._delay(pk, surf, "retarder", aoi, float(dot(pk.d[0], surf.normal)) < 0.0,
                                             state=state))
        return ZERO_POWER if self._spent(pk) else None

    def _do_faraday_rotator(self, pk: _Packet, surf: SurfaceG, h: Hits, inter: InteractionG) -> str | None:
        """A Faraday rotator (``contracts.optics.FaradayRotatorInteraction``): each ray's field turns about the surface
        normal, the same way in the laboratory whichever way the ray travels (``optics/polarization/jones.py``). Light of a
        wavelength outside its stated response passes on with a power that is not computed (``_pass_not_computed``; a probe
        stops here); the element states no optical path, so per-ray
        optical path lengths are undefined from here on and the pass's group delay is recorded as not stated, with the
        beam's polarization state (its two circular components may be delayed differently)."""
        assert inter.faraday is not None
        along = dot(pk.d, surf.normal)
        aoi = float(incidence_deg(abs(float(along[0])), pk.d[0], surf.normal))
        response = inter.faraday.at(pk.wavelength_nm)
        if response is None:
            bound = pk.chief_power() if pk.power_bound_W is None else pk.power_bound_W
            self._issue(make_issue(ErrorCode.MODEL_WAVELENGTH_OUTSIDE_BAND, Status.OUT_OF_MODEL_SCOPE,
                                   f"{pk.wavelength_nm!r} nm light meets the Faraday rotator {surf.surface_id}, whose response is stated for "
                                   f"{inter.faraday.domain_text()}: its power and polarization behind the element are not computed, at "
                                   f"most {bound:.4g} W",
                                   location=f"surface:{surf.surface_id}", path_id=pk.path_id, wavelength_nm=pk.wavelength_nm,
                                   power_upper_bound_W=bound,
                                   suggested_action="use a source wavelength inside that domain, or state the response at this wavelength"))
            if self.bounding:
                self._end_event(pk, "faraday_rotator", aoi)
                pk.alive[:] = False
                return FARADAY_UNAVAILABLE
            pk.opl_undefined = True  # (the element states no optical path)
            return self._pass_not_computed(pk, surf, "faraday_rotator", aoi, FARADAY_UNAVAILABLE)
        rotation, transmittance = response
        u1, u2, ok = project_axis(surf.u_axis, pk.d, pk.e1, pk.e2, self.cfg.zero_vector_tol)
        state = self._axis_state(pk, u1, u2) if bool(ok[0]) else None  # (read along an axis of the element, so it does not depend on the ray's basis)
        pk.jones = faraday_rotation(pk.jones, along, rotation) * math.sqrt(transmittance)
        pk.gstate = apply_flat(pk.gstate, pk.d[0], pk.d[0], self.cfg)
        pk.opl_undefined = True
        self._end_event(pk, "faraday_rotator", aoi)
        pk.lineage.append(f"{surf.surface_id}:faraday_rotator")
        pk.delays = (*pk.delays, self._delay(pk, surf, "faraday_rotator", aoi, float(along[0]) < 0.0, state=state))
        return ZERO_POWER if self._spent(pk) else None

    def _do_diffraction_grating(self, pk: _Packet, surf: SurfaceG, h: Hits, inter: InteractionG) -> str | None:
        """A diffraction grating (``contracts.optics.DiffractionGratingInteraction``): the packet ends here and each order it
        follows starts a branch, with every ray's direction from the vector grating equation, an order that does not
        propagate for a ray retiring it (for the chief ray: no branch). The branch's basis is ``(s, d x s)`` with ``s`` the
        groove axis across the ray: in a planar mount it is the TE axis, in a conical one the basis rule stays orthonormal
        about the leaving direction. The shares come from ``_grating_shares``; the part of the incident power no followed
        order carries and no stated absorption takes goes to ``UNMODELLED_RESPONSE``. The element states no optical path."""
        cfg = self.cfg
        g = inter.grating
        assert g is not None
        # (its orders' shares may depend on the polarization: taken as mixing two crystal twins, whatever its response)
        self._twins_polarized(pk, surf, 1.0, "a diffraction grating")
        from_pos, same_side = self._side(pk, surf, h)
        if g.reflection and not from_pos:
            self._issue(make_issue(ErrorCode.BLOCKED_BY_BACK_SIDE, Status.WARNING, "the beam hits the back of a reflection grating and is blocked",
                                   location=f"surface:{surf.surface_id}", suggested_action="turn the grating's ruled face (local +Z) toward the beam"))
            self._end_event(pk, "blocked_back_side")
            pk.alive[:] = False
            return BLOCKED_BACK
        pk.alive = pk.alive & same_side
        med_from, med_to = (surf.medium_pos, surf.medium_neg) if from_pos else (surf.medium_neg, surf.medium_pos)
        if pk.medium != med_from:
            self._issue(make_issue(ErrorCode.MEDIUM_MISMATCH, Status.FAIL,
                                   f"ray is in medium {pk.medium!r} but the grating expects {med_from!r} on this side",
                                   location=f"surface:{surf.surface_id}", tracked_medium=pk.medium, expected_medium=med_from,
                                   suggested_action="state the grating's media as the media on its two sides"))
            self._end_event(pk, "medium_mismatch")
            pk.alive[:] = False
            return MEDIUM_MISMATCH
        n1, n2, exit_medium = pk.n, pk.n, pk.medium
        if not g.reflection:
            found, issue = self.library.evaluator(med_to).index(pk.wavelength_nm, location=f"surface:{surf.surface_id}")
            if issue is not None or found is None:
                if issue is not None:
                    self._issue(issue)
                pk.alive[:] = False
                return MATERIAL_RANGE
            n2, exit_medium = found, med_to
        cos_i = np.abs(dot(pk.d, surf.normal))
        term = self._aoi_guard(pk, surf, incidence_deg(cos_i, pk.d, surf.normal))
        if term:
            return term
        aoi = float(incidence_deg(float(cos_i[0]), pk.d[0], surf.normal))
        d_in = pk.d.copy()
        along_k, across_k = float(dot(d_in[0], g.k_axis)), float(dot(d_in[0], g.groove_axis))
        incidence = float(atan2_deg(np.array([along_k]), cos_i[:1])[0])
        conical = float(atan2_deg(np.array([abs(across_k)]), np.array([math.sqrt(along_k * along_k + float(cos_i[0]) * float(cos_i[0]))]))[0])
        # the TE (groove) and TM axes about the arriving rays, the field along them
        s_in, ok_s = normalize(g.groove_axis - dot(d_in, g.groove_axis)[:, None] * d_in, cfg.zero_vector_tol)
        p_in = np.cross(d_in, s_in)
        j_s = pk.jones[:, 0] * dot(pk.e1, s_in) + pk.jones[:, 1] * dot(pk.e2, s_in)
        j_p = pk.jones[:, 0] * dot(pk.e1, p_in) + pk.jones[:, 1] * dot(pk.e2, p_in)
        known = pk.power_bound_W is None
        power = pk.chief_power() if known else float(pk.power_bound_W)  # type: ignore[arg-type]
        ratio = pk.wavelength_nm / g.period_nm
        propagating = propagating_orders(along_k, across_k, ratio, n1, n2)
        followed = tuple(m for m in g.orders if m in propagating) if g.all_propagating else g.orders
        shares = self._grating_shares(pk, surf, g, from_pos, incidence, conical, j_s, j_p, power, propagating)
        basis = "NONE" if g.response is None else g.response.data_kind.upper()
        rows: dict[int, dict[str, Any]] = {}
        for m in followed:
            d_out, ok = diffracted(d_in, surf.normal, g.k_axis, m, ratio, n1, n2, g.reflection)
            share = shares.of(m)
            if not bool(ok[0]):
                rows[m] = {"followed": False}
                if not g.all_propagating:
                    key = (surf.surface_id, pk.wavelength_nm, f"order{m}")
                    if key not in self._grating_reported:
                        self._grating_reported.add(key)
                        self._issue(make_issue(ErrorCode.GRATING_ORDER_NONPROPAGATING, Status.WARNING,
                                               f"order {m:+d} of the grating {surf.surface_id} does not propagate at {pk.wavelength_nm!r} nm and "
                                               f"{incidence:.6g} deg: its tangential wavevector exceeds the exit medium's, so it is not traced",
                                               location=f"surface:{surf.surface_id}", path_id=pk.path_id, wavelength_nm=pk.wavelength_nm, order=m,
                                               incidence_deg=incidence))
                continue
            order_power = power * share.fraction if share.fraction is not None else 0.0
            self._branch_power += order_power if known else 0.0
            if share.fraction is None and not self._geometry_orders():
                continue
            child = self._spawn(pk, surf, f"grating_order_{m:+d}", order_power if known else 0.0)
            if child is None:
                self._unknown_left += known and share.fraction is None  # (a branch of unknown power born of known power: nothing of it reached _flow)
                rows[m] = {"followed": False, "direction": d_out[0]}
                continue
            self._order_state(child, pk, surf, inter, d_in, d_out, ok, s_in, ok_s, p_in, j_s, j_p, share, m, power, n1, n2, exit_medium, aoi, from_pos, h)
            rows[m] = {"followed": True, "direction": d_out[0], "path": child.path_id}
            if self.recorder is not None:
                self.recorder.branch_spawned(pk, child)
            if self._spent(child) or (child.power_bound_W is not None and child.power_bound_W <= 0.0):
                child.power_bound_W = None
                self._finish(child, ZERO_POWER, surf.surface_id)
            else:
                self._enqueue(child)
        if known:
            if shares.absorbed > 0.0:
                self._split[SINK_ELEMENT_TRANSMISSION_LOSS] = power * shares.absorbed
            self._split_detail = "grating"
        self._record_orders(pk, surf, g, rows, propagating, followed, shares, power, incidence, conical, basis)
        self._end_event(pk, "grating", aoi)
        pk.jones = np.zeros_like(pk.jones)
        pk.alive[:] = False
        return GRATING_ORDERS

    def _order_state(self, child: _Packet, pk: _Packet, surf: SurfaceG, inter: InteractionG, d_in: Array, d_out: Array, ok: NDArray[np.bool_], s_in: Array,
                     ok_s: NDArray[np.bool_], p_in: Array, j_s: CArray, j_p: CArray, share: _OrderShare, m: int, power: float, n1: float, n2: float, exit_medium: str,
                     aoi: float, from_pos: bool, h: Hits) -> None:
        """The branch of order ``m``: its rays' directions, basis ``(s, d x s)`` and field, Gaussian beam and record of the pass."""
        cfg, g = self.cfg, inter.grating
        assert g is not None
        child.alive = pk.alive & ok & ok_s
        child.d = np.where(child.alive[:, None], d_out, d_in)
        s_out, ok_out = normalize(g.groove_axis - dot(child.d, g.groove_axis)[:, None] * child.d, cfg.zero_vector_tol)
        # (a retired ray keeps an orthonormal basis; a ray along the grooves has no groove axis across it: it retires)
        child.alive = child.alive & ok_out
        s_out = np.where(ok_out[:, None], s_out, pk.e1)
        child.e1, child.e2 = s_out, np.cross(child.d, s_out)
        if share.fraction is None:
            child.jones = np.zeros_like(pk.jones)
            child.power_bound_W = power * share.bound
            child.bulk_loss_unknown = False
        else:
            child.jones = np.stack([math.sqrt(share.te) * j_s, math.sqrt(share.tm) * j_p], axis=1)
            child.polarization_open = pk.polarization_open or share.polarization_open
        child.medium, child.n = exit_medium, n2
        child.gstate = apply_grating(pk.gstate, d_in[0], d_out[0], surf.normal, n1, n2, cfg)
        child.opl_undefined = True
        child.grating_steps = (*pk.grating_steps, (surf.surface_id, m))
        child.lineage.append(f"{surf.surface_id}:{'grating_reflect' if g.reflection else 'grating_transmit'}")
        state = None if share.fraction is None else self._sp_state(pk, s_in, aoi)
        child.delays = (*pk.delays, self._delay(pk, surf, f"grating_order_{m:+d}", aoi, from_pos, (float(h.u[0]), float(h.v[0])), state=state))
        self._report_gaussian_state(child, surf)

    def _grating_issue(self, pk: _Packet, surf: SurfaceG, what: str, message: str, power: float, action: str) -> None:
        """Why a grating's order powers are not known at this wavelength, once per surface, wavelength and reason."""
        key = (surf.surface_id, pk.wavelength_nm, what)
        if key in self._grating_reported:
            return
        self._grating_reported.add(key)
        self._issue(make_issue(ErrorCode.GRATING_RESPONSE_UNAVAILABLE, Status.UNKNOWN,
                               f"{pk.wavelength_nm!r} nm light meets the grating {surf.surface_id}: {message}; the orders' directions are traced, "
                               f"their powers are not known and together at most {power:.6g} W",
                               location=f"surface:{surf.surface_id}", path_id=pk.path_id, wavelength_nm=pk.wavelength_nm, incident_power_W=power,
                               reason=what, suggested_action=action))

    def _grating_shares(self, pk: _Packet, surf: SurfaceG, g: GratingG, from_pos: bool, incidence: float, conical: float, j_s: CArray, j_p: CArray,
                        power: float, propagating: tuple[int, ...]) -> _GratingShares:
        """What the response says of this pass (``_GratingShares``). The orders' powers are known only for light of known
        power, a ``te_tm`` absolute table, from the side it is stated for, inside its domain and conical limit; there the
        field's TE and TM parts add incoherently in each order, whatever their relative phase, and the order's polarization
        is known for light purely TE or TM, and where the table states TE = TM for every order of this reading: the response then does
        not depend on the polarization, and each order keeps the state that arrives. Elsewhere the share is unknown and bounded: by what the table bounds where
        it holds as a bound (twice an unpolarized mean), else by the whole incident power."""
        r = g.response
        known_power = pk.power_bound_W is None
        if r is None:
            if known_power:
                self._grating_issue(pk, surf, "NO_RESPONSE", "it states no order response", power, "state the grating's response per wavelength, "
                                    "angle of incidence and order to obtain the order powers")
            return _GratingShares()
        if not g.reflection and from_pos:
            if known_power:
                self._grating_issue(pk, surf, "OTHER_SIDE", "its response holds for light travelling along its local +Z, and this light travels "
                                    "against it", power, "turn the grating, or state the response of this side in another model")
            return _GratingShares()
        if not conical <= r.max_conical_angle_deg + AOI_MATCH_DEG:
            if known_power:
                self._grating_issue(pk, surf, "CONICAL", f"the chief ray arrives {conical:.6g} deg from the plane perpendicular to the grooves, and "
                                    f"its response holds up to {r.max_conical_angle_deg:g} deg", power,
                                    "state the conical angle the response holds for (max_conical_angle_deg), or align the grooves")
            return _GratingShares()
        reading = r.at(pk.wavelength_nm, incidence)
        if reading is None:
            if known_power:
                self._grating_issue(pk, surf, "OUT_OF_DOMAIN", f"at {incidence:.6g} deg it lies outside the response, stated for {r.domain_text()}", power,
                                    "state the response at this wavelength and angle, or use a source and pose inside its domain")
            return _GratingShares()
        if r.normalization == "relative":
            if known_power:
                self._grating_issue(pk, surf, "RELATIVE", "its response is relative, not a fraction of the incident power", power,
                                    "state an absolute response to obtain the order powers")
            return _GratingShares(relative=dict(reading.efficiency_te))
        if r.polarization_basis == "unpolarized":
            if known_power:
                self._grating_issue(pk, surf, "UNPOLARIZED", "its response states the mean of TE and TM, and the traced beam is fully polarized", power,
                                    "state the TE and TM response (equal, where the response does not depend on the polarization)")
            return _GratingShares(bounds={m: min(1.0, 2.0 * v) for m, v in reading.efficiency_te.items()},
                                  stated=dict(reading.efficiency_te))
        if not known_power:
            return _GratingShares(bounds={m: max(reading.efficiency_te[m], reading.efficiency_tm[m]) for m in reading.efficiency_te})
        p_s, p_p = float(np.abs(j_s[0]) ** 2), float(np.abs(j_p[0]) ** 2)
        total = p_s + p_p
        f_s, f_p = (p_s / total, p_p / total) if total > 0.0 else (1.0, 0.0)
        independent = all(reading.efficiency_te[m] == reading.efficiency_tm[m] for m in reading.efficiency_te)
        return _GratingShares(reading=reading, f_s=f_s, f_p=f_p, polarization_open=min(f_s, f_p) > POLARIZATION_MATCH_FRACTION and not independent,
                              absorbed=f_s * reading.absorption_te + f_p * reading.absorption_tm)

    def _record_orders(self, pk: _Packet, surf: SurfaceG, g: GratingG, rows: dict[int, dict[str, Any]], propagating: tuple[int, ...], followed: tuple[int, ...],
                       shares: _GratingShares, power: float, incidence: float, conical: float, basis: str) -> None:
        """One ``GratingOrderResult`` per order of this pass (``_geometry_orders``)."""
        if not self._geometry_orders():
            return
        known_power = pk.power_bound_W is None
        # (the beam's power is out of model scope behind these surfaces: it is its chief ray's, not the light's, so no order's
        # power -- nor a bound from it -- is stated; the response's share, direction and polarization stay)
        scoped = pk.scope_behind()
        n1 = pk.n
        listed = {*followed, *propagating} if len(propagating) <= MAX_LISTED_PROPAGATING_ORDERS else set(followed)
        for m in sorted(listed):
            row = rows.get(m, {"followed": False})
            direction = row.get("direction")
            propagates = m in propagating
            if direction is None and propagates:
                direction, _ok = diffracted(pk.d[:1], surf.normal, g.k_axis, m, pk.wavelength_nm / g.period_nm, n1, self._exit_index(pk, surf, g), g.reflection)
                direction = direction[0]
            share = shares.of(m)
            exit_angle = None
            if direction is not None:
                exit_angle = float(atan2_deg(np.array([float(dot(direction, g.k_axis))]), np.array([abs(float(dot(direction, surf.normal)))]))[0])
            known = share.fraction is not None and known_power
            stated = known and not scoped
            power_state = (QuantityState.NOT_APPLICABLE if not propagates else QuantityState.OUT_OF_MODEL_SCOPE if scoped and known_power
                           else QuantityState.KNOWN if known else QuantityState.BOUNDED)
            self.out.grating_orders.append(GratingOrderResult(
                surface_id=surf.surface_id, element_id=surf.owner_id, path_id=pk.path_id, source_id=pk.src.source_id, spectral_index=pk.spectral_index,
                wavelength_nm=pk.wavelength_nm, incidence_deg=incidence, conical_angle_deg=conical, order=m,
                propagation="PROPAGATING" if propagates else "NONPROPAGATING", followed=bool(row.get("followed")), order_path_id=row.get("path"),
                direction=None if direction is None else (float(direction[0]), float(direction[1]), float(direction[2])), exit_angle_deg=exit_angle,
                power_state=power_state, power_W=power * share.fraction if stated and propagates else None,  # type: ignore[operator]
                power_upper_bound_W=None if power_state is not QuantityState.BOUNDED else power * share.bound,
                power_out_of_scope_at=scoped if propagates and known_power else (),
                efficiency=(share.fraction if known else share.stated) if propagates else None, relative_response=shares.relative.get(m) if propagates else None,
                polarization_state=QuantityState.NOT_APPLICABLE if not propagates else QuantityState.KNOWN if known and not (
                    share.polarization_open or pk.polarization_open) else QuantityState.UNKNOWN,
                phase_state=QuantityState.NOT_APPLICABLE if not propagates else QuantityState.UNKNOWN, response_basis=basis))  # type: ignore[arg-type]

    def _exit_index(self, pk: _Packet, surf: SurfaceG, g: GratingG) -> float:
        """The index of the medium an order leaves in (an order was already evaluated in it: no new data is needed)."""
        if g.reflection:
            return pk.n
        from_pos = float(dot(pk.d[0], surf.normal)) < 0.0
        found, _issue = self.library.evaluator(surf.medium_neg if from_pos else surf.medium_pos).index(pk.wavelength_nm, location=f"surface:{surf.surface_id}")
        return float(found)  # type: ignore[arg-type]

    def _scatter_off(self, pk: _Packet, surf: SurfaceG, h: Hits, inter: InteractionG) -> None:
        """A surface that scatters (``optics/rays/scatter.py``): ``fraction`` of the power meeting it leaves as scattered light,
        booked ``SURFACE_SCATTER`` (handed on, like a branch's power) and, for a traced beam, recorded as a ``ScatterOrigin``;
        the packet goes on with the rest to the surface's own interaction. Light that does not meet the surface's active side
        (a reflector's back, a medium other than the one the surface expects) does not scatter: the interaction decides its fate."""
        sc = inter.scatter
        if sc is None or pk.power_bound_W is not None:  # (light of unknown power was booked where it became unknown)
            return
        normal = h.normal[0]
        from_pos = float(dot(pk.d[0], normal)) < 0.0
        if inter.kind == "mirror" and not (inter.reflective_side == "both" or (inter.reflective_side == "positive") == from_pos):
            return
        med_from, med_to = (surf.medium_pos, surf.medium_neg) if from_pos else (surf.medium_neg, surf.medium_pos)
        if inter.kind == "refract" and pk.medium != med_from:
            return
        p_in = pk.chief_power()
        scattered = sc.fraction * p_in
        pk.jones = pk.jones * math.sqrt(1.0 - sc.fraction)
        self._branch_power += scattered
        law = sc.law if sc.law == "LAMBERTIAN" else f"{sc.law} n={sc.exponent:g}"
        self._flow(pk, SINK_SURFACE_SCATTER, scattered, surf, f"surface scatter ({law}, {sc.side})")
        if not self._follows_scatter or self.bounding or scattered <= 0.0:
            return
        reflect = sc.side == "reflect"
        axis = np.array(normal, dtype=float) * (1.0 if from_pos == reflect else -1.0)
        g = pk.gstate
        self.out.scatter_origins.append(ScatterOrigin(
            ordinal=len(self.out.scatter_origins), source_id=pk.src.source_id, source_owner_id=pk.src.owner_id,
            spectral_index=pk.spectral_index, wavelength_nm=pk.wavelength_nm, path_id=pk.path_id, surface_id=surf.surface_id,
            surface_owner_id=surf.owner_id, point=h.point[0].copy(), axis=axis, medium=pk.medium if reflect else med_to,
            incident_power_W=p_in, fraction=sc.fraction, power_W=scattered, side=sc.side, law=sc.law, exponent=sc.exponent,
            sample_count=sc.sample_count, seed=sc.seed, beam_radius_mm=max(g.radius("x"), g.radius("y")) if g.valid and not pk.chief_obstructed else None,
            power_scope=pk.scope_behind()))

    def _do_complex_transmission(self, pk: _Packet, surf: SurfaceG, h: Hits, inter: InteractionG) -> str | None:
        """A thin complex-transmission element (``optics/complex_transmission.py``): every live ray gains the optical path
        ``phi / k0`` and the tangential wavevector ``grad phi`` at its hit point, keeps its polarization (its basis transported to
        its new direction) and its field is scaled by ``A`` there; the beam's power by the Gaussian-weighted mean of ``A^2`` over
        its footprint (the chief ray's ``A^2`` for a beam without a Gaussian footprint), the rest booked as the element's loss. A
        beam whose phase is not a plane across its footprint is no longer a Gaussian beam (``TRANSMISSION_MAP_PHASE_STRUCTURE``);
        a power factor the quadrature does not resolve, or a beam without a Gaussian footprint across which ``A`` varies, puts its
        power out of model scope (``TRANSMISSION_MAP_VARIES_ACROSS_BEAM``). A ray turned beyond grazing is evanescent: retired,
        and where it is the chief ray the path ends out of model scope (``TRANSMISSION_MAP_EVANESCENT``). Where only part of the
        beam is turned beyond grazing -- some of its pupil rays, or nodes of its Gaussian footprint within the clear aperture
        (``complex_transmission.evanescent``, the Gaussian's local ray directions) -- the beam is no longer Gaussian, and its
        power goes on the geometric pupil-sample basis (``_evanescent_share``): the evanescent share leaves it, booked
        ``TERMINATED_OUT_OF_MODEL`` (light of unknown fate), with the sampling resolution of the share that goes on. Where no
        pupil ray is turned and the footprint's turned nodes stand for at most the line's negligible level
        (``complex_transmission.evanescent_share_bound`` x the incident power, at most ``negligible_power_fraction`` x the line
        input: the far Gaussian tail), that bound is left out as negligible (``_leave_out``, as a negligible untraced branch)
        and the beam goes on as if no node were turned: its Gaussian state and power stand."""
        cfg, m = self.cfg, inter.transmission_map
        assert m is not None
        normal = surf.normal
        lam_mm = pk.wavelength_nm * NM_TO_MM
        amp, phi, g_u, g_v = transmission_evaluate(m, h.u, h.v)
        d_in = pk.d.copy()
        along = dot(d_in, normal)
        tangential = d_in - along[:, None] * normal[None, :] + (lam_mm / (TWO_PI * pk.n)) * (
            g_u[:, None] * surf.u_axis[None, :] + g_v[:, None] * surf.v_axis[None, :])
        sin2 = dot(tangential, tangential)
        aoi = float(incidence_deg(abs(float(along[0])), d_in[0], normal))
        if not float(sin2[0]) < 1.0:
            self._issue(make_issue(ErrorCode.TRANSMISSION_MAP_EVANESCENT, Status.OUT_OF_MODEL_SCOPE,
                                   "the element's phase gradient turns the chief ray beyond grazing: the light it transmits there is "
                                   "evanescent, outside the model of propagating rays", location=f"surface:{surf.surface_id}",
                                   path_id=pk.path_id, wavelength_nm=pk.wavelength_nm, tangential_direction=math.sqrt(float(sin2[0])),
                                   suggested_action="reduce the phase gradient (d phi / d u, d phi / d v) below 2 pi n / lambda"))
            self._end_event(pk, "complex_transmission", aoi)
            pk.alive[:] = False
            return GRAZING
        propagating = sin2 < 1.0
        annulus = pk.alive & self._annulus_rays(pk)  # (the live pupil samples meeting the element)
        turned = annulus & ~propagating
        pk.alive = pk.alive & propagating
        sign = np.where(along >= 0.0, 1.0, -1.0)
        d_out = tangential + (sign * np.sqrt(np.maximum(1.0 - sin2, 0.0)))[:, None] * normal[None, :]
        d_out = np.where(pk.alive[:, None], d_out, d_in)
        p_in = pk.chief_power()
        factor = m.amplitude * m.amplitude
        varies = ""
        valid = pk.gstate.valid and not pk.chief_obstructed
        fine = None
        footprint_turned = 0
        if valid:
            g = pk.gstate
            frame = (h.point[0], d_in[0], normal, surf.origin, surf.u_axis, surf.v_axis, g.bx, g.by, g.radius("x"), g.radius("y"))
            curvature = tuple(0.0 if g.curvature_radius(axis) is None else 1.0 / g.curvature_radius(axis) for axis in ("x", "y"))
            fine = transmission_footprint(self.quad_rule, self.quad_rule[0].size, cfg.aperture_quad_extent_sigmas, *frame, *curvature)
            nodes = transmission_evanescent(m, fine, normal, surf.u_axis, surf.v_axis, lam_mm / (TWO_PI * pk.n))
            turned_nodes = nodes & aperture_inside(surf.aperture, fine.u, fine.v, 0.0)
            footprint_turned = int(turned_nodes.sum())
            # (the far Gaussian tail turned beyond grazing: at most the line's negligible level is left out, bounded)
            tail_W = p_in * evanescent_share_bound(fine, turned_nodes) if footprint_turned else 0.0
            if footprint_turned and not turned.any() and self._negligible(pk, tail_W):
                footprint_turned = 0
                if not pk.chief_obstructed and pk.power_bound_W is None:
                    self._leave_out(pk, tail_W)
        if not pk.chief_obstructed and pk.power_bound_W is None and (footprint_turned or turned.any()):
            return self._evanescent_share(pk, surf, inter, d_in, d_out, float(along[0]) < 0.0, annulus, turned, amp, phi, footprint_turned, aoi)
        if valid and (m.grid is not None and m.grid.amplitude is not None or m.phase_structured):
            if m.grid is not None and m.grid.amplitude is not None:
                factor = beam_power_fraction(m, fine)
                half = gauss_legendre_rule(max(self.quad_rule[0].size // 2, 2))
                coarse = beam_power_fraction(m, transmission_footprint(half, half[0].size, cfg.aperture_quad_extent_sigmas, *frame))
                if abs(factor - coarse) > POWER_FACTOR_TOL:
                    varies = (f"the mean of A^2 over the beam's footprint is not resolved by the quadrature ({factor:.9g} with the full "
                              f"rule, {coarse:.9g} with half its nodes)")
        elif m.grid is not None and m.grid.amplitude is not None:
            factor = float(amp[0]) * float(amp[0])
            live = np.flatnonzero(pk.alive[pk.pupil_slice()]) + 1
            if live.size:
                spread = float(np.max(amp[live] * amp[live])) - float(np.min(amp[live] * amp[live]))
                if spread > RAY_POWER_SPREAD_TOL * max(factor, cfg.power_epsilon_W):
                    varies = f"A^2 varies by {spread:.4g} across the beam's live pupil rays (it has no valid Gaussian state)"
        if varies:
            self._issue(make_issue(ErrorCode.TRANSMISSION_MAP_VARIES_ACROSS_BEAM, Status.OUT_OF_MODEL_SCOPE,
                                   f"{varies}: the beam's power behind the element is out of model scope", location=f"surface:{surf.surface_id}",
                                   path_id=pk.path_id, power_factor=factor,
                                   suggested_action="use a smaller beam at the element, or a map whose amplitude changes more slowly across it"))
            if surf.surface_id not in pk.power_scope:
                pk.power_scope = (*pk.power_scope, surf.surface_id)
        if valid and m.phase_structured and fine is not None:
            residual = transmission_phase_residual(m, fine)
            if residual > PHASE_RESIDUAL_TOL_RAD:
                text = (f"the element's phase, less its best-fitting plane, varies by {residual:.4g} rad RMS across the beam's footprint "
                        f"(tolerance {PHASE_RESIDUAL_TOL_RAD:g} rad): the transmitted light is not a Gaussian beam; its rays and power "
                        "go on")
                pk.gstate = pk.gstate.invalidated(ErrorCode.TRANSMISSION_MAP_PHASE_STRUCTURE, text)
        if pk.gstate.valid:
            pk.gstate = apply_grating(pk.gstate, d_in[0], d_out[0], normal, pk.n, pk.n, cfg)
        self._report_gaussian_state(pk, surf)
        scale = amp.astype(np.complex128)
        scale[0] = math.sqrt(factor)
        pk.jones = pk.jones * scale[:, None]
        pk.e1, pk.e2 = parallel_transport_basis(pk.e1, pk.e2, d_in, d_out, cfg.antiparallel_tol)
        pk.d = d_out
        pk.opl = pk.opl + phi * (lam_mm / TWO_PI)
        self._split[SINK_ELEMENT_TRANSMISSION_LOSS] = p_in * (1.0 - factor)
        self._split_detail = "complex transmission"
        self._end_event(pk, "complex_transmission", aoi)
        pk.lineage.append(f"{surf.surface_id}:complex_transmission")
        pk.delays = (*pk.delays, self._delay(pk, surf, "complex_transmission", aoi, float(along[0]) < 0.0))
        return ZERO_POWER if self._spent(pk) else None

    def _evanescent_share(self, pk: _Packet, surf: SurfaceG, inter: InteractionG, d_in: Array, d_out: Array, from_pos: bool,
                          annulus: NDArray[np.bool_], turned: NDArray[np.bool_], amp: Array, phi: Array, footprint_turned: int,
                          aoi: float) -> str | None:
        """A complex-transmission element turns part of a beam beyond grazing (``_do_complex_transmission``: ``d_out`` its
        rays' new directions, the evanescent pupil rays retired). The beam behind it is not a Gaussian beam: its state is invalidated
        (``TRANSMISSION_MAP_EVANESCENT``) and its power goes on the geometric pupil-sample basis (``_geometric_cut``): of the
        ``n_live`` live equal-power pupil samples that met the element, the turned ones carry the evanescent share
        ``sum A^2 / n_live`` over them, booked ``TERMINATED_OUT_OF_MODEL`` (light of unknown fate), the others go on with
        ``sum A^2 / n_live`` over them, the rest is the element's loss. The share's sampling resolution grows by the samples whose
        cell may straddle the evanescent edge, and the outermost ring's (its cell reaches the Gaussian tail, which may be turned
        though no sample is: ``_sampling_boundary``), over the samples that go on (``_Packet.sampling``). With no live pupil
        sample the share is not computed and with none left the power that goes on is not known: the path ends out of model scope
        (``TERMINATED_OUT_OF_MODEL``)."""
        cfg, m = self.cfg, inter.transmission_map
        assert m is not None
        location = f"surface:{surf.surface_id}"
        n_live, n_turned = int(annulus.sum()), int(turned.sum())
        n_pass = n_live - n_turned
        where = (f"{n_turned} of its {n_live} live pupil rays" if n_turned else "no pupil ray") + (
            f" and {footprint_turned} nodes of its Gaussian footprint within the clear aperture" if footprint_turned else "")
        action = "reduce the phase gradient (d phi / d u, d phi / d v) below 2 pi n / lambda across the whole beam, or use a smaller beam"
        p_in = pk.chief_power()
        if n_pass <= 0:
            self._issue(make_issue(ErrorCode.TRANSMISSION_MAP_EVANESCENT, Status.OUT_OF_MODEL_SCOPE,
                                   f"the element's phase gradient turns {where} beyond grazing: the light it transmits there is evanescent, "
                                   f"and no pupil sample is left to share the power that goes on, so its {p_in:.4g} W are not traced on",
                                   location=location, path_id=pk.path_id, wavelength_nm=pk.wavelength_nm, pupil_rays_turned=n_turned,
                                   pupil_rays_live=n_live, untraced_power_W=p_in, suggested_action=action))
            self._end_event(pk, "complex_transmission", aoi)
            pk.alive[:] = False
            return GRAZING
        rows = np.flatnonzero(annulus)
        a2 = amp[rows] * amp[rows]
        gone = turned[rows]
        kept_share = correctly_rounded_sum(a2[~gone].tolist()) / n_live
        turned_share = correctly_rounded_sum(a2[gone].tolist()) / n_live
        if m.grid is not None and m.grid.amplitude is not None:
            spread = float(np.max(a2)) - float(np.min(a2))
            if spread > RAY_POWER_SPREAD_TOL * max(float(np.max(a2)), cfg.power_epsilon_W):
                self._issue(make_issue(ErrorCode.TRANSMISSION_MAP_VARIES_ACROSS_BEAM, Status.OUT_OF_MODEL_SCOPE,
                                       f"A^2 varies by {spread:.4g} across the beam's live pupil rays where part of the beam is turned beyond "
                                       "grazing: the beam's power behind the element is out of model scope", location=location,
                                       path_id=pk.path_id, power_factor=kept_share,
                                       suggested_action="use a smaller beam at the element, or a map whose amplitude changes more slowly across it"))
                if surf.surface_id not in pk.power_scope:
                    pk.power_scope = (*pk.power_scope, surf.surface_id)
        boundary = self._sampling_boundary(pk, annulus, turned)
        added = boundary / n_pass
        before = 0.0 if pk.sampling is None else pk.sampling
        pk.sampling = before + added + before * added
        evanescent_W, kept_W = p_in * turned_share, p_in * kept_share
        text = (f"the element's phase gradient turns {where} beyond grazing: the light it transmits there is evanescent, outside the "
                f"model of propagating rays; {evanescent_W:.4g} W (the turned pupil samples' share) leave the beam with that fate unknown, "
                f"and {kept_W:.4g} W go on, within a sampling resolution of {pk.sampling * kept_W:.3g} W, on the geometric pupil-sample "
                "basis (the beam behind the element is not a Gaussian beam)")
        self._issue(make_issue(ErrorCode.TRANSMISSION_MAP_EVANESCENT, Status.OUT_OF_MODEL_SCOPE, text, location=location,
                               path_id=pk.path_id, wavelength_nm=pk.wavelength_nm, pupil_rays_turned=n_turned, pupil_rays_live=n_live,
                               footprint_nodes_turned=footprint_turned, evanescent_power_W=evanescent_W, transmitted_power_W=kept_W,
                               sampling_resolution_W=pk.sampling * kept_W, suggested_action=action))
        if pk.gstate.valid:
            pk.gstate = pk.gstate.invalidated(ErrorCode.TRANSMISSION_MAP_EVANESCENT, text)
        self._report_gaussian_state(pk, surf)
        scale = amp.astype(np.complex128)
        scale[0] = math.sqrt(kept_share)
        pk.jones = pk.jones * scale[:, None]
        pk.e1, pk.e2 = parallel_transport_basis(pk.e1, pk.e2, d_in, d_out, cfg.antiparallel_tol)
        pk.d = d_out
        pk.opl = pk.opl + phi * (pk.wavelength_nm * NM_TO_MM / TWO_PI)
        self._split[SINK_TERMINATED_OUT_OF_MODEL] = evanescent_W
        self._split[SINK_ELEMENT_TRANSMISSION_LOSS] = p_in * max(0.0, 1.0 - kept_share - turned_share)
        self._split_detail = "complex transmission"
        self._end_event(pk, "complex_transmission", aoi)
        pk.lineage.append(f"{surf.surface_id}:complex_transmission")
        pk.delays = (*pk.delays, self._delay(pk, surf, "complex_transmission", aoi, from_pos))
        return ZERO_POWER if self._spent(pk) else None

    def _do_absorber(self, pk: _Packet, surf: SurfaceG, h: Hits, inter: InteractionG) -> str | None:
        self._end_event(pk, "absorbed")
        pk.lineage.append(f"{surf.surface_id}:absorbed")
        pk.alive[:] = False
        return ABSORBED

    def _do_aperture_stop(self, pk: _Packet, surf: SurfaceG, h: Hits, inter: InteractionG) -> str | None:
        """An aperture: it passes the light inside it, scaled by its transmittance. An attenuating one (transmittance below
        1) is an attenuator without a thickness, like a thin neutral-density filter: its group delay is not stated (a
        ``delay_key`` record); a fully open one only bounds the beam."""
        pk.jones = pk.jones * math.sqrt(inter.transmittance)
        pk.gstate = apply_flat(pk.gstate, pk.d[0], pk.d[0], self.cfg)
        aoi = float(incidence_deg(abs(float(dot(pk.d[0], surf.normal))), pk.d[0], surf.normal))
        self._end_event(pk, "aperture_stop", aoi)
        pk.lineage.append(f"{surf.surface_id}:aperture_stop")
        if inter.transmittance < 1.0:
            pk.delays = (*pk.delays, self._delay(pk, surf, "aperture_stop", aoi, float(dot(pk.d[0], surf.normal)) < 0.0))
        return ZERO_POWER if self._spent(pk) else None

    def _do_neutral_density(self, pk: _Packet, surf: SurfaceG, h: Hits, inter: InteractionG) -> str | None:
        """A thin attenuator (``optics/attenuation.py``): every live ray's field is scaled by the square root of the
        transmittance at its own hit point; the packet's power is the chief ray's. The constant reflectance is a traced
        branch (ideal-reflector phases); the rest is absorbed (``ATTENUATOR_ABSORPTION``). A beam whose density varies by
        more than ``ND_FOOTPRINT_OD_TOLERANCE`` across its 1/e^2 footprint is no longer represented by the chief-ray value:
        out of model scope (``ND_DENSITY_VARIES_ACROSS_BEAM``; its Gaussian state is withheld from there on)."""
        cfg = self.cfg
        att = inter.attenuator
        assert att is not None
        cos_i = abs(float(dot(pk.d[0], surf.normal)))
        aoi = float(incidence_deg(cos_i, pk.d[0], surf.normal))
        t_rays = att.transmittance(h.point)
        if not math.isfinite(float(t_rays[0])):
            bound = pk.chief_power() if pk.power_bound_W is None else pk.power_bound_W
            self._issue(make_issue(ErrorCode.ND_UNCOATED_REGION, Status.OUT_OF_MODEL_SCOPE,
                                   "the beam crosses the attenuator where its density law leaves the plane uncoated, and the element "
                                   f"states no uncoated transmittance: its power behind the element is not computed, at most {bound:.4g} W",
                                   location=f"surface:{surf.surface_id}", path_id=pk.path_id, wavelength_nm=pk.wavelength_nm,
                                   power_upper_bound_W=bound,
                                   suggested_action="move the beam onto the coated region, or state the element's uncoated_transmittance"))
            if self.bounding:
                self._end_event(pk, "neutral_density", aoi)
                pk.alive[:] = False
                return ND_UNCOATED
            return self._pass_not_computed(pk, surf, "neutral_density", aoi, ND_UNCOATED)
        footprint, across, varies = None, "", False
        if pk.gstate.valid and not pk.chief_obstructed:
            footprint = att.footprint_points(h.point[0], pk.d[0], pk.gstate.bx, pk.gstate.by, pk.gstate.radius("x"), pk.gstate.radius("y"))
            across = "the beam's 1/e^2 footprint"
        else:  # no Gaussian footprint (invalid upstream, or an annulus): the live pupil rays sample the beam
            rays = np.flatnonzero(pk.alive[pk.pupil_slice()]) + 1
            rays = rays[np.isfinite(h.point[rays]).all(axis=1)]
            if rays.size:
                footprint = h.point[rays]
                across = "the beam's live pupil rays (it has no valid Gaussian state)"
        if footprint is not None and att.kind != "fixed":
            od_min, od_max, defined = att.footprint_density_range(footprint)
            delta = od_max - od_min
            if not defined or delta > ND_FOOTPRINT_OD_TOLERANCE:
                text = (f"the optical density varies by {delta:.3g} across {across} (tolerance "
                        f"{ND_FOOTPRINT_OD_TOLERANCE:g}){'' if defined else ', part of which is uncoated without a stated transmittance'}: "
                        "the chief-ray transmittance no longer represents the beam, so its power is out of model scope from here on")
                self._issue(make_issue(ErrorCode.ND_DENSITY_VARIES_ACROSS_BEAM, Status.OUT_OF_MODEL_SCOPE, text,
                                       location=f"surface:{surf.surface_id}", path_id=pk.path_id, od_min=od_min, od_max=od_max,
                                       od_variation=delta, od_variation_tolerance=ND_FOOTPRINT_OD_TOLERANCE,
                                       transmission_error_estimate=transmission_error_estimate(delta) if defined else None,
                                       suggested_action="use a smaller beam at the attenuator, or move it to a part of the filter where "
                                                        "the density changes more slowly"))
                if pk.gstate.valid:
                    pk.gstate = pk.gstate.invalidated(ErrorCode.ND_DENSITY_VARIES_ACROSS_BEAM, text)
                    self._gaussian_reported.add((pk.path_id, ErrorCode.ND_DENSITY_VARIES_ACROSS_BEAM.value))  # reported here, once
                varies = True
        uncoated = ~np.isfinite(t_rays)
        if bool(uncoated.any()):
            pk.alive = pk.alive & ~uncoated
        if att.reflectance > 0.0:
            d_in = pk.d.copy()
            normals = np.repeat(surf.normal[None, :], pk.d.shape[0], 0)
            d_ref, n_or, cos_r = reflect_dir(pk.d, normals, cfg.zero_vector_tol)
            s = s_axis(pk.d, n_or, pk.e1, cfg.zero_vector_tol)
            amplitude = np.full(pk.o.shape[0], math.sqrt(att.reflectance))
            j_ref, e1_ref, e2_ref = apply_interface(pk.jones, pk.e1, pk.e2, d_in, s, -amplitude + 0j, amplitude + 0j, d_ref)
            reflected_power = pk.power0_W * power2(j_ref[0])
            self._branch_power += reflected_power
            child = self._spawn(pk, surf, "attenuator_reflection", reflected_power)
            if child is not None:
                child.d, child.jones, child.e1, child.e2 = d_ref, j_ref, e1_ref, e2_ref
                child.gstate = apply_curved_or_flat_interface(
                    pk.gstate, kind="mirror", d_in=d_in[0], d_out=d_ref[0], n_or=n_or[0], cos_i=float(cos_r[0]), n1=pk.n, n2=pk.n,
                    cos_t=float(cos_r[0]), tir=False, radius_signed_mm=None, cfg=cfg,
                    curvature=self._local_curvature(surf, h, d_in[0], "mirror"))
                child.lineage.append(f"{surf.surface_id}:attenuator_reflect")
                child.delays = (*child.delays, self._delay(pk, surf, "attenuator_reflect", aoi, float(dot(d_in[0], surf.normal)) < 0.0))
                if self.recorder is not None:
                    self.recorder.branch_spawned(pk, child)
                if not self._spent(child):
                    self._enqueue(child)
                else:
                    self._finish(child, ZERO_POWER, surf.surface_id)
        if varies and surf.surface_id not in pk.power_scope:
            # (after the reflected branch: its power, the constant reflectance's share, does not depend on the density)
            pk.power_scope = (*pk.power_scope, surf.surface_id)
        pk.jones = pk.jones * np.sqrt(np.where(np.isfinite(t_rays), t_rays, 0.0))[:, None]
        pk.gstate = apply_flat(pk.gstate, pk.d[0], pk.d[0], cfg)
        self._report_gaussian_state(pk, surf)
        self._end_event(pk, "neutral_density", aoi)
        pk.lineage.append(f"{surf.surface_id}:neutral_density")
        pk.delays = (*pk.delays, self._delay(pk, surf, "neutral_density", aoi, float(dot(pk.d[0], surf.normal)) < 0.0))
        return ZERO_POWER if self._spent(pk) else None

    def _annulus_on_area(self, pk: _Packet, surf: SurfaceG, h: Hits, cos_a: float) -> tuple[bool | None, float | None, str, float]:
        """Does an obstructed packet's annulus count on the plane's active area?

        ``True``: every live ray of the annulus is on it and the whole annulus provably is (``_whole_annulus_inside``), or
        at most ``capture_min_power_fraction`` of the annulus's power can fall beside it (the envelope's power outside the
        area relative to the annulus's share of it; the last value returned, reported as a WARNING); ``False``: none of it
        is (at most ``negligible_power_fraction`` of the envelope, no ray); ``None``: part of it, so the share on the area is
        not known (the reason is returned; the last value is then the share of the annulus that may fall beside the area
        where every ray is on it and only ``capture_min_power_fraction`` decides, else 0). The rays alone never decide: they
        sample the annulus only to 1.18 beam radii. Also returns the envelope's fraction on the area."""
        cfg = self.cfg
        ap = surf.interaction.active_aperture
        if ap is None:
            return True, None, "", 0.0
        annulus = np.flatnonzero(pk.alive & self._annulus_rays(pk))
        on_area = aperture_inside(ap, h.u[annulus], h.v[annulus], cfg.aperture_edge_tol_mm)
        if not pk.gstate.valid:
            return None, None, f"the envelope of the unobstructed beam is not known there ({pk.gstate.reason_text})", 0.0
        envelope = transmission_fraction(pk.gstate, ap, (float(h.u[0]), float(h.v[0])), surf.u_axis, surf.v_axis, cos_a, cfg,
                                         self.quad_rule)
        over = 0.0  # (the share beside the area where only the capture level keeps the annulus from counting whole)
        if bool(on_area.all()):
            if self._whole_annulus_inside(pk, surf, h, ap, 1.0 - envelope):
                return True, None, "", 0.0
            beside = (1.0 - envelope) / pk.annulus_fraction if pk.annulus_fraction > 0.0 else math.inf
            if beside <= cfg.capture_min_power_fraction:
                return True, None, "", beside
            over = beside if math.isfinite(beside) else 0.0
        if envelope <= cfg.negligible_power_fraction and not bool(on_area.any()):
            return False, envelope, "", 0.0
        level = (f"; up to {over:.3g} of the annulus's power may fall beside it, more than capture_min_power_fraction "
                 f"({cfg.capture_min_power_fraction:g})" if over > 0.0 else "")
        return None, envelope, (f"{envelope:.4g} of the envelope of the unobstructed beam and {int(on_area.sum())} of the "
                                f"{annulus.size} pupil rays of the annulus fall on it{level}"), over

    def _first_order(self, pk: _Packet, surf: SurfaceG, h: Hits, *, promoted: bool = False) -> tuple[Array | None, str | None]:
        """Central-difference ray-transfer matrix ``d(u, v, l_u, l_v) / d(x0, y0, a0, b0)`` from the source plane to this
        plane (``l``: direction cosines on the plane's u / v axes), from the packet's differential rays.

        Behind a promoted reference ray (``_promote_reference``) the differential rays were relaunched about it, and the
        matrix composes their transfer from the promotion with the chief ray's up to it (``_Packet.transfer_base``): it
        is about two different rays, an approximation within the aberration between them. It is used only where
        ``promoted`` is asked for (the samples' diffraction spread, ``_sample_picture``); an arrival reports none
        (``TRANSFER_PROMOTED``)."""
        if pk.reference_promoted and not promoted:
            return None, TRANSFER_PROMOTED
        if pk.differential == 0:
            return None, TRANSFER_NOT_SAMPLED
        if self._axicon_ids and any(step.rsplit(":", 1)[0] in self._axicon_ids for step in pk.lineage):
            return None, TRANSFER_AXICON
        if self._tessellated_ids and any(step.rsplit(":", 1)[0] in self._tessellated_ids for step in pk.lineage):
            return None, TRANSFER_FACETS
        first = pk.o.shape[0] - pk.differential
        rays = np.arange(first, first + pk.differential)
        if not bool(pk.alive[rays].all()) or not bool(np.isfinite(h.t[rays]).all()):
            return None, TRANSFER_LOST
        state = np.column_stack([h.u[rays], h.v[rays], dot(pk.d[rays], surf.u_axis), dot(pk.d[rays], surf.v_axis)])  # (8, 4)
        if pk.transfer_base is not None:
            matrix = matmul(np.column_stack([state[2 * j] - state[2 * j + 1] for j in range(4)]), pk.transfer_base)
        else:
            steps = (self.cfg.first_order_position_step_mm, self.cfg.first_order_position_step_mm,
                     self.cfg.first_order_angle_step_rad, self.cfg.first_order_angle_step_rad)
            matrix = np.column_stack([(state[2 * j] - state[2 * j + 1]) / (2.0 * steps[j]) for j in range(4)])
        # report each block at its resolution: the finite-difference roundoff stays far below it, so every runtime agrees
        cfg = self.cfg
        unitless, b_res, c_res = cfg.first_order_unitless_resolution, cfg.first_order_b_resolution_mm, cfg.first_order_c_resolution_per_mm
        resolution = np.array([[unitless, unitless, b_res, b_res], [unitless, unitless, b_res, b_res],
                               [c_res, c_res, unitless, unitless], [c_res, c_res, unitless, unitless]])
        return np.round(matrix / resolution) * resolution, None

    def _do_observe(self, pk: _Packet, surf: SurfaceG, h: Hits, inter: InteractionG) -> str | None:
        cos_i = abs(float(dot(pk.d[0], surf.normal)))
        aoi = float(incidence_deg(cos_i, pk.d[0], surf.normal))
        if inter.observation_medium is not None and inter.observation_medium != pk.medium:
            self._issue(make_issue(ErrorCode.MEDIUM_MISMATCH, Status.WARNING,
                                   "the beam medium at the observation plane differs from the declared plane medium",
                                   location=f"observation:{inter.observation_id}", tracked_medium=pk.medium,
                                   declared_medium=inter.observation_medium))
        oid = inter.observation_id or ""
        if pk.chief_obstructed:
            # behind a central obstruction the light is the annulus, bounded by the envelope of the unobstructed beam: an
            # active area counts it whole or not at all; one that takes part of it has no computable share (diffraction
            # fills the geometric shadow, so a detector there is not dark either): the path is not followed
            whole, area_fraction, why, beside = self._annulus_on_area(pk, surf, h, cos_i)
            if whole is None:
                # (where only the capture level decided, the issue names it: it decides what the plane's power leaves out)
                level = ({"setting": "capture_min_power_fraction", "capture_min_power_fraction": self.cfg.capture_min_power_fraction,
                          "annulus_power_beside_bound": beside} if beside > 0.0 else {})
                term = self._annulus_not_followed(
                    pk, surf, f"observation {oid!r} has an active area that takes part of the annulus ({why})",
                    envelope_on_area_fraction=area_fraction, **level)
                if term is not None:
                    return term
                whole, beside = True, 0.0  # (the plane takes part of it: the arrival is recorded, its power not known)
            annulus = AnnulusOnArea(whole, area_fraction, beside)
            if beside > 0.0:
                self._issue(make_issue(
                    ErrorCode.INCOMPLETE_ENVELOPE, Status.WARNING,
                    f"at most {beside:.3g} of the annular beam's power ({beside * pk.chief_power():.4g} W) may fall beside the "
                    f"active area of observation {oid!r} (less than capture_min_power_fraction); its whole power is counted",
                    location=f"observation:{oid}", path_id=pk.path_id, annulus_power_beside_bound=beside,
                    suggested_action="use a larger active area if the bound matters"))
        else:
            annulus = None
        area_share, sampling = None, pk.sampling
        area = inter.active_aperture
        if (inter.absorbing and area is not None and apertures.bounded(area) and not pk.chief_obstructed and not pk.gstate.valid
                and not self.bounding and bool(aperture_inside(area, h.u[:1], h.v[:1], self.cfg.aperture_edge_tol_mm)[0])):
            # no valid Gaussian state and the chief ray on the active area: the area takes the share of the equal-power pupil
            # samples on it, where they represent the beam; else the power it detects is not computed (a probe counts its
            # whole power: an upper bound)
            share = self._geometric_area(pk, surf, h, area, oid)
            if isinstance(share, dict):  # (the arrival is recorded, the power it detects not known)
                self._not_computed(pk, surf, f"the active area of observation {oid!r}", **share)
            else:
                area_share, sampling = share
        reading = observation_reading(self._view(pk), surf, h, cos_i, aoi, annulus, self.cfg, self.quad_rule,
                                      None if inter.active_aperture is None else lambda: self._open_leg(pk, h), area_share=area_share)
        counted, outside, accepted = reading.counted, reading.outside, reading.accepted
        crossing = pk.crossings.get(oid, 0)
        pk.crossings[oid] = crossing + 1
        alive_pupil = np.zeros_like(pk.alive)
        alive_pupil[pk.pupil_slice()] = pk.alive[pk.pupil_slice()]
        pupil_uv = np.column_stack([h.u[alive_pupil], h.v[alive_pupil]]) if alive_pupil.any() else np.zeros((0, 2))
        rays = np.flatnonzero(alive_pupil)
        launch = self._pupil_launch(pk, rays)
        transfer, transfer_issue = self._first_order(pk, surf, h)
        incident_W = pk.chief_power()
        taken_at, taken_upper_W = pk.taken_at, pk.taken_upper_W
        # the shares taken on the way add to the shares taken before this plane on its own leg (``_taken_before``; a surface
        # weighed there is not counted again for this leg): the plane reads [P - S, P]
        earlier = [(at, bound_W) for at, bound_W, leg in pk.taken_path if not (leg == pk.interactions and at in pk.taken_at)]
        if earlier and pk.power_bound_W is None:
            taken_at = (*taken_at, *dict.fromkeys(at for at, _ in earlier if at not in taken_at))
            taken_upper_W = min(incident_W, correctly_rounded_sum([taken_upper_W or 0.0, *(bound_W for _, bound_W in earlier)]))
        self.out.arrivals.append(
            ArrivalRecord(
                observation_id=oid, surface_id=surf.surface_id, path_id=pk.path_id, source_id=pk.src.source_id,
                spectral_index=pk.spectral_index, wavelength_nm=pk.wavelength_nm, medium=pk.medium, n=pk.n,
                position=h.point[0].copy(), direction=pk.d[0].copy(), power_W=incident_W, opl_mm=float(pk.opl[0]),
                length_mm=float(pk.length[0]), jones=pk.jones[0].copy(), e1=pk.e1[0].copy(), e2=pk.e2[0].copy(),
                gstate=(pk.gstate.invalidated(ErrorCode.GAUSSIAN_CENTRAL_OBSTRUCTION, OBSTRUCTED_TEXT) if pk.chief_obstructed
                        else pk.gstate), uv=(float(h.u[0]), float(h.v[0])), aoi_deg=aoi, pupil_uv=pupil_uv,
                pupil_total=pk.pupil_total, coherence_group=pk.src.coherence_group,
                coherence_length_mm=line_coherence_length_mm(pk.src, pk.spectral_index),
                lineage=tuple(pk.lineage),
                plane_normal=surf.normal.copy(), plane_u=surf.u_axis.copy(), plane_v=surf.v_axis.copy(), plane_origin=surf.origin.copy(),
                pupil_points=h.point[rays], pupil_directions=pk.d[rays], pupil_launch=launch,  # fancy indexing copies
                crossing_index=crossing, counted=counted, area_fraction=None if counted else reading.area_fraction,
                detected_power_W=None if pk.power_bound_W is not None else reading.detected_W,
                detected_power_min_W=None if pk.power_bound_W is not None else reading.detected_min_W, parent_path_id=pk.parent_path_id,
                within_acceptance=accepted, tail_within_acceptance=reading.tail_accepted, chief_obstructed=pk.chief_obstructed,
                transfer=transfer,
                transfer_issue=transfer_issue, ray_opl_defined=not pk.opl_undefined, source_e1=pk.src.e1.copy(), source_e2=pk.src.e2.copy(),
                power_scope=pk.power_scope, held=pk.held, met_first=pk.met_first, left_beyond=pk.left_beyond, group_opl_mm=pk.gpl, gdd_mm2=pk.gdd,
                dispersive=pk.dispersive, delays=pk.delays,
                power_bound_W=(min(pk.power_bound_W, pk.chief_power()) if pk.bulk_loss_unknown and pk.power_bound_W is not None
                                else pk.power_bound_W), polarization_open=pk.polarization_open,
                grating_steps=pk.grating_steps, coincidence=pk.coincidence, polarization_component=pk.src.polarization_component,
                sampling=sampling, taken_at=taken_at, taken_upper_W=taken_upper_W, taken_lower_W=pk.taken_lower_W,
                reference_promoted=pk.reference_promoted, samples_unresolved=pk.samples_unresolved,
            )
        )
        self.out.trace_events.append({"path": pk.path_id, "observe": inter.observation_id, "power": pk.chief_power()})
        pk.lineage.append(f"{surf.surface_id}:observe")
        self._end_event(pk, "observed", aoi)
        if inter.absorbing and counted:
            if outside > 0.0:
                self._flow(pk, SINK_OUTSIDE_ACTIVE_AREA, pk.chief_power() * outside, surf, "observe")
                pk.jones = pk.jones * math.sqrt(1.0 - outside)
            pk.alive[:] = False
            # outside the acceptance cone the plane stops the beam on its active area without detecting it
            return DETECTED if accepted is not False else NOT_ACCEPTED
        return None

    def _report_gaussian_state(self, pk: _Packet, surf: SurfaceG) -> None:
        g = pk.gstate
        if not g.valid and g.reason is not None:
            key = (pk.path_id, g.reason.value)
            if key not in self._gaussian_reported:
                self._gaussian_reported.add(key)
                self._issue(make_issue(g.reason, g.status, g.reason_text, location=f"surface:{surf.surface_id}", path_id=key[0]))
