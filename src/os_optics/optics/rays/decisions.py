"""The tracer's partial-beam and observation decisions, as pure functions over explicit inputs.

``Tracer`` (``tracer.py``) moves packets; at four points it decides something about the light *around* the chief ray, and
those decisions live here, each a function of frozen inputs that returns what it decided, so a decision can be tested
without building a scene or tracing a beam:

* ``crossing_group``: which surfaces the chief ray's next crossing holds, in the order they act (observation planes
  within the ray-advance guard of each other or of a physical surface, then that surface);
* ``leg_partial_crossings``: at the end of a straight leg, per observation plane the chief ray does not cross on it,
  whether part of the beam crosses it first, and the bound of that part's power (``PartialCrossing``, ``leg_share``), with
  an explicit outcome per plane (recorded, skipped by the pre-filter, negligible, or no bound and no pupil ray);
* ``taken_before``: the surfaces that take part of the beam out before a transparent plane the chief ray crosses (or, the
  reflecting surface the leg began at, send it beyond the plane), each with the Gaussian share that decided it, its bound
  and, for a stopping surface, the share it certainly takes (``TakenBefore``; ``sheet_flow_share``, ``start_flow_share``,
  ``lower_share``);
* ``observation_reading``: what an observation plane reads of a crossing (counted on its active area, inside its
  acceptance cone, the power detected).

Inputs: ``SceneIndex`` (the scene's observation planes and blocking surfaces, built once per ``Tracer``), ``BeamView``
(the packet fields a decision reads, at that moment), the step's ray parameters per surface (``times``) and their blocking
view, a leg (``Leg``, built by ``Tracer._leg``), and geometry the tracer resolves for a curved surface (``NextSurface``,
``StartSurface``: the plane that stands for it). The functions keep the tracer's operation order and its libm-free
helpers (``exact_erfc_float``, ``exact_log``); they never write an input. The tracer applies their results (appends,
power scope) itself.
"""

from __future__ import annotations

from collections.abc import Callable

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ...config import NumericalConfig
from ...errors import ErrorCode
from ...math.exact_erf import erfc_float as exact_erfc_float
from ...math.exact_trig import incidence_deg
from ...math.exact_trig import exp_float
from ...math.exact_trig import log as exact_log
from ...math.vectors import dot
from ...scene.model import SurfaceG
from ...units import NM_TO_MM
from ..gaussian import GState, propagate, transmission_fraction
from ..surfaces import Hits, aperture_inside, aperture_miss_distance
from ..surfaces import sag as sag_profile
from ..surfaces.geometry import support_of
from .. import apertures

Array = NDArray[np.float64]

# a Gaussian beam whose two axes' beam parameters (q, M^2) agree to this relative difference is round: its profile across
# any direction scales as one along a leg (``gaussian_crossing``)
ROUND_BEAM_REL_TOL = 1e-9
# ``beyond_reach`` decides a plane is beyond the beam's reach only with this margin (relative to the distances compared,
# and an absolute floor in mm), so the rounding of its sums never decides a plane that the full check would keep
PREFILTER_REL_MARGIN = 1e-9
PREFILTER_ABS_MARGIN_MM = 1e-12
# A beam whose profile across a plane does not scale as one: an energy-flow line may cross the plane this many times on a
# leg (``flow_crossing``: its distance from the plane has at most two inflections), so its crossings add at most this
# many times its share. ``K`` (``envelope_sigmas``) keeps the tail's share at this multiple at the line's negligible level.
MOST_LINE_CROSSINGS = 4.0
# Without a Gaussian state the live pupil rays are the only evidence of where the light is: the outermost ring lies 1.18 beam
# radii out on a launched beam, and ``K`` is about 3.3 radii at ``negligible_power_fraction``, so a plane more than this many
# envelopes of those rays from the chief ray stays out of the light's reach (``within_ray_envelope``).
RAY_ENVELOPE_MARGIN = 3.0
# Fixed-point steps of the chief-ray stretch where a tilted start surface has let the beam's envelope go (``settled``)
SETTLE_STEPS = 8
SETTLE_MARGIN = 1.01


# ------------------------------------------------------------------------------------------------------------ inputs
@dataclass(frozen=True)
class SceneIndex:
    """The scene's surfaces as the decisions see them (built once per trace, ``SceneIndex.of``): the observation planes,
    the physical surfaces, the surfaces that stop or turn a ray (every physical surface and the absorbing planes; a
    transparent observation plane is virtual: it never changes, stops or retires a ray), per scene index the
    observation planes' frames and active areas (shape 0 none, 1 circle, 2 rectangle; centre, axes, half sizes), and the
    balls that hold the scene's physical surfaces, source points and active areas (``ball_centres``, ``ball_radii``)."""

    surfaces: tuple[SurfaceG, ...]
    observe_ids: tuple[int, ...]
    physical_ids: tuple[int, ...]
    blocking_ids: NDArray[np.intp]
    observe_normal: Array
    observe_origin: Array
    area_shape: NDArray[np.intp]
    area_centre: Array
    area_u: Array
    area_v: Array
    area_half: Array
    ball_centres: Array
    ball_radii: Array
    #: ``NumericalConfig.laboratory_reach_mm``
    laboratory_reach_mm: float = NumericalConfig.model_fields["laboratory_reach_mm"].default

    def reach_along(self, start: Array, d: Array) -> float:
        """How far along the ray ``start + s d`` a plane without an active area counts the light of a leg that leaves the scene:
        ``max(laboratory_reach_mm, support)``, ``support`` the largest ``(c - start) . d + r`` over the balls that hold the scene's
        physical surfaces, source points and active areas. It is measured along the ray, so an element off to the side does
        not extend it and turning the whole design does not change it. No plane's origin counts: on an unbounded plane it
        means nothing."""
        if self.ball_radii.size == 0:
            return self.laboratory_reach_mm
        return max(self.laboratory_reach_mm, float(np.max(dot(self.ball_centres - start, d) + self.ball_radii)))

    @classmethod
    def of(cls, surfaces: list[SurfaceG] | tuple[SurfaceG, ...], points: tuple[Array, ...] = (),
           laboratory_reach_mm: float = NumericalConfig.model_fields["laboratory_reach_mm"].default) -> SceneIndex:
        observe_ids = tuple(i for i, s in enumerate(surfaces) if s.interaction.kind == "observe")
        blocking_ids = np.array([i for i, s in enumerate(surfaces) if s.interaction.kind != "observe" or s.interaction.absorbing],
                                dtype=np.intp)
        observe_normal = np.zeros((len(surfaces), 3))
        observe_origin = np.zeros((len(surfaces), 3))
        area_shape = np.zeros(len(surfaces), dtype=np.intp)
        area_centre = np.zeros((len(surfaces), 3))
        area_u = np.zeros((len(surfaces), 3))
        area_v = np.zeros((len(surfaces), 3))
        area_half = np.zeros((len(surfaces), 2))
        for i in observe_ids:
            surf = surfaces[i]
            observe_normal[i] = surf.normal
            observe_origin[i] = surf.origin
            active = surf.interaction.active_aperture
            if active is not None and active.shape != "none":
                code, half_u, half_v = apertures.index_area(active)
                area_shape[i] = code
                area_centre[i] = surf.origin + active.offset_u * surf.u_axis + active.offset_v * surf.v_axis
                area_u[i], area_v[i] = surf.u_axis, surf.v_axis
                area_half[i] = (half_u, half_v)
        physical_ids = tuple(i for i, s in enumerate(surfaces) if s.interaction.kind != "observe")
        balls = [ball for surf in surfaces for ball in _surface_balls(surf)] + [(np.asarray(point, dtype=float), 0.0) for point in points]
        ball_centres = np.array([centre for centre, _ in balls], dtype=float).reshape(-1, 3)
        ball_radii = np.array([radius for _, radius in balls], dtype=float)
        return cls(tuple(surfaces), observe_ids, physical_ids, blocking_ids, observe_normal, observe_origin, area_shape, area_centre,
                   area_u, area_v, area_half, ball_centres, ball_radii, laboratory_reach_mm)


def _surface_balls(surf: SurfaceG) -> list[tuple[Array, float]]:
    """Points and radii whose balls hold a surface: an observation plane's active area (none without one), a physical surface's
    support (an unbounded one: its origin), a sag surface's depth over its support by the profile at the support's centre and
    edge."""
    if surf.interaction.kind == "observe":
        active = surf.interaction.active_aperture
        if active is None or active.shape == "none":
            return []
        reach = apertures.reach(active)
        return [(surf.origin + active.offset_u * surf.u_axis + active.offset_v * surf.v_axis, reach)]
    if surf.geometry == "sphere":
        support = surf.aperture_radius + surf.mount_margin
        return [(surf.center + abs(surf.radius) * surf.cap_axis, math.sqrt(2.0) * support)]
    support = support_of(surf)
    if support.shape == "none":
        return [(surf.origin, 0.0)]
    reach = apertures.reach(support)
    middle = surf.origin + support.offset_u * surf.u_axis + support.offset_v * surf.v_axis
    if surf.geometry != "sag":
        return [(middle, reach)]
    a = support.offset_u + reach * np.array([0.0, 1.0, -1.0, 0.0, 0.0])
    b = support.offset_v + reach * np.array([0.0, 0.0, 0.0, 1.0, -1.0])
    depth, ok = sag_profile.sag(surf, sag_profile.transverse_s(surf, a, b))
    return [(middle + float(z) * surf.normal, reach) for z, good in zip(depth, ok) if good and math.isfinite(float(z))] or [(middle, reach)]


@dataclass(frozen=True)
class BeamView:
    """The packet fields a decision reads, as they are when it is taken: its identity, the rays (``o``, ``d``, ``alive``;
    ``annulus``: the pupil samples, rays ``1 .. pupil_total``), the Gaussian state, the chief ray's power, the central
    obstruction (``annulus_fraction``: the annulus's share of the unobstructed beam) and the line's negligible level
    (``negligible_power_fraction`` x the line's input, or its ``follow_W`` level). The arrays are the packet's own (not
    copied): a decision never writes them, and the tracer applies the result before it changes the packet."""

    path_id: str
    source_id: str
    spectral_index: int
    wavelength_nm: float
    o: Array
    d: Array
    alive: NDArray[np.bool_]
    annulus: NDArray[np.bool_]
    gstate: GState
    power_W: float
    chief_obstructed: bool
    annulus_fraction: float
    negligible_W: float

    def negligible(self, power_W: float) -> bool:
        """Power at most the line's negligible level (numerically zero for the line)."""
        return power_W <= self.negligible_W


@dataclass(frozen=True)
class _NoArea:
    active_aperture: None = None


@dataclass(frozen=True)
class Sheet:
    """A plane standing for a physical surface in ``part_on_sheet`` (its point and unit normal; no active area)."""

    origin: Array
    normal: Array
    interaction: _NoArea = _NoArea()
    u_axis: None = None
    v_axis: None = None


@dataclass(frozen=True)
class Leg:
    """A straight leg as the partial-crossing bounds take it (``Tracer._leg``): the chief ray ``start + s d``, its end at
    ``s_end``; the start surface as the plane ``s = start_offset - rho . start_normal / (d . start_normal)`` of the ray at
    transverse offset ``rho`` (a plane surface: its own plane; a curved one: a plane, ``_curved_plane`` -- a sphere's
    across the ray, a sag surface's across it or its tangent plane at the chief ray's start -- shifted ``start_offset``
    before the chief ray's start, the farthest the surface reaches back within the beam's envelope), the end surface
    likewise (``end_offset``: after ``s_end`` by the farthest a curved end surface reaches on), and the Gaussian beam over
    the stretch ``[first, first + length]`` of the chief ray those planes span (``beam`` there at ``first``). A leg that
    leaves the scene has ``s_end`` and ``length`` ``inf`` and no end surface (``end_offset`` ``None``); ``scene_end``: how far
    along it a plane without an active area counts its light (``SceneIndex.reach_along``)."""

    start: Array
    d: Array
    s_end: float
    start_normal: Array
    start_offset: float
    end_normal: Array
    end_offset: float | None  # (None: an end surface along the ray)
    first: float
    length: float
    beam: GState
    scene_end: float = math.inf


@dataclass(frozen=True)
class PartialCrossing:
    """Part of a traced beam crosses an observation plane on a straight leg its chief ray does not cross that plane on,
    before ``ends_at`` (the surface the leg ends at: a physical surface, an absorbing plane that stops the beam; ``None``:
    the beam leaves the scene). The power that crosses is not computed (the packet's power follows its chief ray); it is at
    most ``power_W`` (``leg_share``). On an ``absorbing`` plane that part leaves the beam there. ``validation/
    untraced.py`` counts it as light that may reach the plane, so the plane is not a known zero for the beam's source
    line. It is never added to a result. ``diffracted``: the beam was severely truncated by a hard edge, so its light is not
    localized and the whole beam may reach this plane ahead of it (``ahead_of_leg``); it counts only for a plane its line
    does not light."""

    observation_id: str
    surface_id: str
    source_id: str
    spectral_index: int
    wavelength_nm: float
    path_id: str
    power_W: float
    rays: int  # of its ``pupil_rays`` live pupil rays, those that cross the plane before the leg's end
    pupil_rays: int
    ends_at: str | None
    absorbing: bool = False
    diffracted: bool = False


@dataclass(frozen=True)
class SurfaceCoincidenceRecord:
    """A plane within ``ray_min_advance_mm`` of a physical surface's chief-ray hit point records the beam arriving at that
    surface before the interaction (``crossing_group``, role ``ARRIVING``) or, crossed again where the beam leaves it toward
    the side it came from, the beam returning (``Tracer._pending_crossings``, role ``RETURNING``, a splitter's child path too).
    ``event``: what the chief ray does at that hit (``REFLECTIVE``, ``TRANSMISSIVE``, ``ABSORBING``: an absorbing plane stops
    the beam at the arrival, as an absorber does; ``Tracer._settle_surface_event``). ``crossing_upper_W``: the auxiliary bound
    of the power that crosses the literal plane before the surface (``literal_crossing_share``); it never replaces the arrival."""

    surface_id: str
    role: str
    event: str
    crossing_upper_W: float | None = None


@dataclass(frozen=True)
class NextSurface:
    """The physical surface the chief ray meets next, as ``taken_before`` bounds the part of the beam that meets it before
    the plane: its id and the plane standing for it (``Tracer._next_surface``: the surface's own plane at the chief ray's
    hit, a curved one shifted as far back as it reaches within the beam's envelope; ``None``: that reach is not bounded)."""

    surface_id: str
    #: or a function that resolves it, called only when ``taken_before`` weighs the surface (as the tracer did before the
    #: extraction: a surface already counted is never resolved)
    sheet: Sheet | None | Callable[[], Sheet | None]
    #: a sag surface: a function bounding the fraction of the beam that may meet the surface itself before the plane
    #: (``Tracer._sag_region_share``; ``None``: no bound that way), called only for a surface counted as taking a share
    region: Callable[[], float | None] | None = None
    #: a surface that stops the light it meets from the side the beam comes from (the back of an uncoated mirror, an
    #: absorber): a function giving a lower bound of the fraction of the beam that certainly meets it first, before the
    #: plane (``lower_share.certain_meeting_share``), called only for a surface counted with a bound
    certain: Callable[[], float] | None = None

    def resolved_sheet(self) -> Sheet | None:
        return self.sheet() if callable(self.sheet) else self.sheet


@dataclass(frozen=True)
class StartSurface:
    """The physical surface the leg began at, when it reflected the beam: its id and its plane (unit normal, and the offset
    along the chief ray of the plane's crossing beyond the chief ray's start; ``Tracer._start_surface``; ``None``: not
    bounded)."""

    surface_id: str
    #: or a function that resolves it, called only when ``taken_before`` weighs the surface
    plane: tuple[Array, float] | None | Callable[[], tuple[Array, float] | None]
    #: the plane standing for the surface for the energy-flow lines within ``K`` beam radii over the stretch they meet it
    #: on (``start_flow_share``; a plane surface: ``plane`` itself; ``None``: not bounded), or a function that resolves it
    flow: tuple[Array, float] | None | Callable[[], tuple[Array, float] | None] = None
    #: a sphere: its tangent plane at the chief ray's start shifted as far beyond it as the cap reaches for the lines
    #: within ``K`` radii over that stretch (``Tracer._sphere_tangent``), standing for it in both pictures beside ``plane``
    #: and ``flow`` (the smaller share of each picture counts); ``None``: none, or a function
    tangent: tuple[Array, float] | None | Callable[[], tuple[Array, float] | None] = None

    def resolved_plane(self) -> tuple[Array, float] | None:
        return self.plane() if callable(self.plane) else self.plane

    def resolved_flow(self) -> tuple[Array, float] | None:
        return self.flow() if callable(self.flow) else self.flow

    def resolved_tangent(self) -> tuple[Array, float] | None:
        return self.tangent() if callable(self.tangent) else self.tangent


# ----------------------------------------------------------------------------------------------- Gaussian shares
def widest(g: GState, at: tuple[float, ...]) -> float:
    """The beam's largest 1/e^2 radius (of both axes) at the chief-ray parameters ``at`` (from where it is ``g``)."""
    scale = g.wavelength_nm * NM_TO_MM / (math.pi * g.n)
    return max(math.sqrt(m2 * scale * ((q.real + s) * (q.real + s) + q.imag * q.imag) / q.imag)
               for q, m2 in ((g.qx, g.m2x), (g.qy, g.m2y)) for s in at)


def envelope_sigmas(level: float, power: float, chief_obstructed: bool, annulus_fraction: float) -> float | None:
    """``K``: beyond ``K`` beam radii from the chief ray a Gaussian beam's share past a line is at most an eighth of the
    line's negligible ``level`` for the beam's ``power`` (``0.5 exp(-2 K^2)`` of it; behind a central obstruction, of the
    annulus's power over its fraction). ``None`` where that is not stated (no level, a power that is not finite)."""
    if chief_obstructed and annulus_fraction > 0.0:
        power = power / annulus_fraction  # (``part_on_sheet`` bounds the annulus by the envelope's part)
    if not (level > 0.0 and math.isfinite(power) and power >= 0.0):
        return None
    ratio = 4.0 * power / level
    return math.sqrt(0.5 * exact_log(ratio)) if ratio > 1.0 else 0.0


def beam_sigmas(beam: BeamView) -> float | None:
    """``envelope_sigmas`` of a beam."""
    return envelope_sigmas(beam.negligible_W, beam.power_W, beam.chief_obstructed, beam.annulus_fraction)


def gaussian_part(g: GState, s_end: float, a: float, b: Array) -> float:
    """The fraction of a Gaussian beam's power in the transverse half-plane ``a + b . rho > 0`` (``rho``: the offset from
    the chief ray, perpendicular to it; ``b`` perpendicular too), on a leg of length ``s_end`` from where the beam is
    ``g``: with the beam's radius along ``b`` at its widest on the leg when the half-plane leaves out the chief ray (more
    of the tail), at its narrowest when it holds it, so the fraction is never understated (``erfc``, libm-free)."""
    bn = math.sqrt(float(dot(b, b)))
    if bn == 0.0:
        return 1.0 if a > 0.0 else 0.0
    u = b / bn
    c = -a / bn  # the half-plane is u . rho > c
    scale = g.wavelength_nm * NM_TO_MM / (math.pi * g.n)
    radii2 = 0.0
    for q, axis, m2 in ((g.qx, g.bx, g.m2x), (g.qy, g.by, g.m2y)):
        ends = [math.sqrt(m2 * scale * ((q.real + s) * (q.real + s) + q.imag * q.imag) / q.imag) for s in (0.0, s_end)]
        waist = -q.real
        wide = max(ends)
        narrow = math.sqrt(m2 * scale * q.imag) if 0.0 <= waist <= s_end else min(ends)
        w = wide if c >= 0.0 else narrow
        along = float(dot(u, axis))
        radii2 += along * along * w * w
    if radii2 <= 0.0:
        return 1.0 if c < 0.0 else 0.0
    tail = 0.5 * exact_erfc_float(math.sqrt(2.0) * abs(c) / math.sqrt(radii2))
    return tail if c >= 0.0 else 1.0 - tail


def gaussian_crossing(g: GState, s_end: float, a: float, b: Array) -> float:
    """The fraction of a Gaussian beam's power that crosses the plane ``a + b . rho = 0`` along the beam (parallel to its
    chief ray; ``a < 0``: the chief ray on its near side) on a stretch of length ``s_end`` from where the beam is ``g``.
    A paraxial Gaussian beam's energy flows along lines that keep their place in its profile (``rho / w(s)`` fixed),
    so where the beam's profile across the plane scales as one (a round beam, ``ROUND_BEAM_REL_TOL``, or the plane
    along one of its axes) the
    lines that cross the plane on the stretch are those beyond it where the beam is widest there and not where it is
    narrowest: the share beyond it at the one less the share at the other (twice over, toward the waist and back, for
    a waist inside the stretch). Otherwise the share beyond it at its widest (``gaussian_part``)."""
    bn = math.sqrt(float(dot(b, b)))
    if bn == 0.0 or a >= 0.0:
        return gaussian_part(g, s_end, a, b)
    u = b / bn
    c = -a / bn
    along = [float(dot(u, g.bx)), float(dot(u, g.by))]
    if not _profile_scales_as_one(g, along[0], along[1]):
        return gaussian_part(g, s_end, a, b)
    scale = g.wavelength_nm * NM_TO_MM / (math.pi * g.n)

    def radius(s: float) -> float:
        r2 = 0.0
        for q, m2, k in ((g.qx, g.m2x, along[0]), (g.qy, g.m2y, along[1])):
            r2 += k * k * m2 * scale * ((q.real + s) * (q.real + s) + q.imag * q.imag) / q.imag
        return math.sqrt(r2)

    def beyond(r: float) -> float:
        return 0.5 * exact_erfc_float(math.sqrt(2.0) * c / r) if r > 0.0 else 0.0

    first, last = beyond(radius(0.0)), beyond(radius(s_end))
    waists = [-q.real for q, k in ((g.qx, along[0]), (g.qy, along[1])) if k != 0.0 and 0.0 < -q.real < s_end]
    if waists:
        least = min(beyond(radius(w)) for w in waists)
        return max(first - least, 0.0) + max(last - least, 0.0)
    return abs(last - first)


def _area_extent(active, u_axis: Array, v_axis: Array, centre: Array, m: Array) -> tuple[float, float]:
    """The range of ``(x - p) . m`` over an active area (``centre``: its centre less ``p``): a linear function's exact extent
    over a disc or a rectangle."""
    mu, mv = float(dot(u_axis, m)), float(dot(v_axis, m))
    half = apertures.spread(active, mu, mv)
    mid = float(dot(centre, m))
    return mid - half, mid + half


def along_area(surf: SurfaceG | Sheet, leg: Leg, cfg: NumericalConfig) -> tuple[float, float, list[tuple[float, Array]]] | None:
    """Where a leg (``Leg``) that runs along observation plane ``surf`` (parallel to its chief ray ``a_point + s d``)
    meets the plane's active area: the light that meets the plane on the area meets it at a point ``x`` of the area, at
    ``s = (x - a_point) . d`` after the leg's start plane and before its end plane, its offset from the chief ray across
    the plane ``(x - a_point) . w``. Returns the stretch of the
    leg along the area (its start ``s`` and length; a negative length: the area lies wholly before the leg's start or
    beyond its end) and the half-planes
    ``(a, b)`` of the beam (``a + b . rho > 0``) the area's offsets across the plane bound; ``None`` without an active
    area. Each range is the area's exact extent (a linear function over a disc or a rectangle)."""
    active = surf.interaction.active_aperture
    if active is None or active.shape == "none":
        return None
    a_point, d = leg.start, leg.d
    u_axis, v_axis = surf.u_axis, surf.v_axis
    centre = surf.origin + active.offset_u * u_axis + active.offset_v * v_axis - a_point

    def extent(m: Array) -> tuple[float, float]:
        low, high = _area_extent(active, u_axis, v_axis, centre, m)
        return low - cfg.aperture_edge_tol_mm, high + cfg.aperture_edge_tol_mm

    lo, hi = extent(d)
    dn_s = float(dot(d, leg.start_normal))
    if abs(dn_s) > cfg.parallel_dot_tol:  # after the start surface: s >= s_start(rho), linear over the area
        lo = max(lo, extent(d - leg.start_normal / dn_s)[0] + leg.start_offset)
    if leg.end_offset is not None:  # before the end surface: s <= s_end(rho)
        hi = min(hi, extent(d - leg.end_normal / float(dot(d, leg.end_normal)))[1] + leg.end_offset)
    n = surf.normal
    w = np.array([float(n[1]) * float(d[2]) - float(n[2]) * float(d[1]), float(n[2]) * float(d[0]) - float(n[0]) * float(d[2]),
                  float(n[0]) * float(d[1]) - float(n[1]) * float(d[0])])
    w = w - float(dot(w, d)) * d
    wn = math.sqrt(float(dot(w, w)))
    limits: list[tuple[float, Array]] = []
    if wn > 0.0:
        w = w / wn
        lo_w, hi_w = extent(w)
        if lo_w > 0.0:
            limits.append((-lo_w, w))
        elif hi_w < 0.0:
            limits.append((hi_w, -w))
    return lo, hi - lo, limits


def part_on_sheet(beam: BeamView, surf: SurfaceG | Sheet, leg: Leg | None, cfg: NumericalConfig, quad_rule: tuple[Array, Array], *,
                  area: bool = True) -> float | None:
    """A bound of the power of the beam that crosses the plane ``surf`` (an observation plane, or a plane standing for a
    physical surface: ``normal``, ``origin``, ``interaction.active_aperture``, ``u_axis``, ``v_axis``) on the leg after
    its start surface and before its end surface (``leg``) and on its active area (``area``), while the chief ray does not
    cross the plane on the leg. The beam is taken as its
    rays parallel to the chief ray with the Gaussian profile across them, its radius at its widest (narrowest, for a
    share that holds the chief ray) over the leg: the part that meets the plane after the start plane and before the end
    plane is two transverse half-planes (``gaussian_part`` each), and the share on an active area its Gaussian
    footprint there (``transmission_fraction``). A plane along the leg (parallel to the chief ray) with an active area:
    the share whose energy flow crosses it on the stretch of the leg along that area (``along_area``,
    ``gaussian_crossing``; nothing when the area lies before the leg's start or beyond its end); without one, the
    share beyond it where the beam is widest on the leg. ``None``: no bound (no valid Gaussian state, a curved start or
    end surface whose reach is not bounded) -- the pupil rays decide. This is the rays' picture on a finite leg;
    ``leg_share`` adds the energy flow's for a partial crossing, and a surface that takes part of the beam first reads
    this one alone (``taken_before``: the near-surface refusal band)."""
    if leg is None:
        return None
    a_point, d, g, s_e = leg.start, leg.d, leg.beam, leg.length

    def across(v: Array) -> Array:
        return v - float(dot(v, d)) * d

    normal_s = leg.start_normal
    n_p = surf.normal
    dn_p, dn_s = float(dot(d, n_p)), float(dot(d, normal_s))
    offset_p = float(dot(surf.origin - a_point, n_p))
    if abs(dn_p) <= cfg.parallel_dot_tol:
        # a plane along the ray: the part beyond it (on its far side from the chief ray), where the leg runs along it
        side = 1.0 if offset_p > 0.0 else -1.0  # (the part on the plane's far side from the chief ray)
        window = along_area(surf, leg, cfg) if area else None
        if window is None:
            fraction = gaussian_part(g, s_e, -abs(offset_p), across(side * n_p))
        elif window[1] < 0.0:
            fraction = 0.0  # (the active area lies before the leg's start or beyond its end)
        else:
            start, length, limits = window
            g_area = propagate(leg.beam, start - leg.first)
            fraction = gaussian_crossing(g_area, length, -abs(offset_p), across(side * n_p))
            for a_w, b_w in limits:
                fraction = min(fraction, gaussian_part(g_area, length, a_w, b_w))
    else:
        s_p = offset_p / dn_p
        fraction = 1.0
        if abs(dn_s) > cfg.parallel_dot_tol:  # after the start surface: s_p(rho) > s_s(rho)
            fraction = min(fraction, gaussian_part(g, s_e, s_p - leg.start_offset, across(normal_s / dn_s - n_p / dn_p)))
        if leg.end_offset is not None:  # before the end surface: s_p(rho) < s_e(rho)
            dn_e = float(dot(d, leg.end_normal))
            fraction = min(fraction, gaussian_part(g, s_e, leg.end_offset - s_p, across(n_p / dn_p - leg.end_normal / dn_e)))
        active = surf.interaction.active_aperture if area else None
        if active is not None and fraction > 0.0:
            at = min(max(s_p, leg.first), leg.first + leg.length)
            centre = a_point + s_p * d - surf.origin
            footprint = transmission_fraction(propagate(beam.gstate, at), active, (float(dot(centre, surf.u_axis)), float(dot(centre, surf.v_axis))),
                                              surf.u_axis, surf.v_axis, abs(dn_p), cfg, quad_rule)
            fraction = min(fraction, footprint)
    return beam_part_W(beam, fraction)


def beam_part_W(beam: BeamView, fraction: float) -> float:
    """The power of a fraction of the beam's Gaussian profile (behind a central obstruction the envelope's part bounds the
    annulus's)."""
    power = beam.power_W
    if beam.chief_obstructed and beam.annulus_fraction > 0.0:
        return min(power, fraction * power / beam.annulus_fraction)
    return fraction * power


# ---------------------------------------------------------------------------------------------------- the energy flow
def _profile_scales_as_one(g: GState, kx: float, ky: float) -> bool:
    """Whether the beam's profile across a direction with components ``kx``, ``ky`` on its axes scales as one along a leg:
    a round beam (``ROUND_BEAM_REL_TOL``), or a direction along one of its axes."""
    if kx == 0.0 or ky == 0.0:
        return True
    dq = g.qx - g.qy  # (exact parts: no libm hypot in the decision)
    return (dq.real * dq.real + dq.imag * dq.imag <= ROUND_BEAM_REL_TOL * ROUND_BEAM_REL_TOL * (g.qx.real * g.qx.real + g.qx.imag * g.qx.imag)
            and abs(g.m2x - g.m2y) <= ROUND_BEAM_REL_TOL * g.m2x)


def _beyond(z: float) -> float:
    """The share of a Gaussian profile beyond the normalized offset ``z`` (in units of its 1/e^2 radius, signed)."""
    tail = 0.5 * exact_erfc_float(math.sqrt(2.0) * abs(z))
    return tail if z >= 0.0 else 1.0 - tail


def _flow_axis(q: complex, m2: float, scale: float, s0: float, s1: float) -> tuple[float, float, complex, float]:
    """One beam axis over ``[s0, s1]`` from where its parameter is ``q``: its radius at its narrowest and at its widest
    (both ends compared, and the waist inside), the parameter where it is widest, and the largest ``|w' / w|`` there
    (``|z| / (z^2 + z_R^2)``, at most ``1 / (2 z_R)``, reached at ``|z| = z_R``)."""
    def radius(s: float) -> float:
        return math.sqrt(m2 * scale * ((q.real + s) * (q.real + s) + q.imag * q.imag) / q.imag)

    def slope(s: float) -> float:
        return abs(q.real + s) / ((q.real + s) * (q.real + s) + q.imag * q.imag)

    ends = (radius(s0), radius(s1))
    narrow = math.sqrt(m2 * scale * q.imag) if s0 <= -q.real <= s1 else min(ends)
    z0, z1 = q.real + s0, q.real + s1
    steepest = max(slope(s0), slope(s1))
    if z0 <= q.imag <= z1 or z0 <= -q.imag <= z1:
        steepest = max(steepest, 0.5 / q.imag)
    return narrow, max(ends), complex(q.real + (s0 if ends[0] >= ends[1] else s1), q.imag), steepest


def flow_crossing(surf: SurfaceG, leg: Leg, first: float, last: float, cfg: NumericalConfig, quad_rule: tuple[Array, Array]) -> float:
    """The most power (a fraction of the beam's Gaussian profile) that crosses the observation plane ``surf`` on the stretch
    ``[first, last]`` of the chief ray (``last`` may be ``inf``), on its active area, by the paraxial beam's energy flow.

    A paraxial Gaussian beam's energy flows along lines that keep their place in its profile: a line is ``(xi, eta)``, at
    ``xi w_x(s)`` and ``eta w_y(s)`` along the beam's axes from the chief ray. Let ``f(s) = p + q s`` be the chief ray's signed
    distance from the plane along its normal ``n``, ``N = |n_perp|`` the normal's length across the ray, ``k_i`` the
    components on the beam's axes of the unit vector along ``n_perp``. A line meets the plane at ``s`` when
    ``xi k_x w_x(s) + eta k_y w_y(s) = -f(s) / N``. With ``w_along(s)^2 = sum k_i^2 w_i(s)^2``, a quadratic
    ``alpha s^2 + beta s + gamma`` in ``s``, ``z(s) = -f(s) / (N w_along(s))`` is the normalized place across the beam the
    plane holds at ``s``. Differentiating ``z^2``, a ratio of quadratics, gives ``f`` times a linear factor: ``z`` has one
    stationary point besides, ``s* = (p beta - 2 q gamma) / (q beta - 2 p alpha)``, and ``z -> -q / (N sqrt(alpha))`` as the
    stretch runs out of the scene.

    * Where the profile across the plane scales as one (``_profile_scales_as_one``) ``zeta = xi k_x + eta k_y`` is a fixed,
      Gaussian coordinate of each line: exactly the lines whose ``zeta`` is one of the values ``z(s)`` takes meet the plane,
      each once per ``s`` it takes that value. So what crosses, counted once per crossing as a transparent plane counts
      it, is the total variation of ``G(z(s))`` over the stretch, ``G`` the share beyond ``z`` (``_beyond``): the difference
      of the shares at the ends of each monotonic piece. (It is at most ``0.5 erfc(sqrt(2) h_min)``, ``h_min = min |z|``,
      per piece, and is less: the lines beyond the plane at both ends of a piece do not cross it.)
    * Otherwise the lines that meet the plane at ``s`` lie, by Cauchy-Schwarz, at a radial distance of at least ``|z(s)|`` in
      the profile, with the half-plane turning along the leg: at most the radial tail ``exp(-2 h_min^2)``,
      ``MOST_LINE_CROSSINGS`` times over.

    On a leg that leaves the scene a plane without an active area counts what crosses it within the reach of a laboratory scene
    or the scene's own parts along the ray, whichever is farther (``Leg.scene_end``, ``SceneIndex.reach_along``). An infinite
    paraxial beam reaches any infinite plane eventually, 400 m on for a plane 100 mm off a 0.5 mm beam; no optic or monitor of
    the design lies there. A plane with an active area has that area's stretch instead.

    With an active area the stretch is cut to the area's extent along the ray and the leg's start and end surfaces there
    (``along_area``), and a line that meets the area keeps within its offsets across the plane: across ``n_perp`` the
    profile is independent, so where it scales as one the share beyond those offsets multiplies; otherwise it bounds
    the radial tail. A strip of the plane is more than a small area, so the area also bounds the power by its footprint
    (below). Both are sound for any stretch; the smaller is taken.

    The footprint: on the stretch the irradiance at a transverse offset is at most ``W_x W_y / (w_x w_y)`` times that of a
    beam at its widest radii ``W_i`` (``w_i`` the narrowest), and the flow's component along ``n`` at most
    ``|d . n| + N rho max|w' / w|`` (``rho``: the farthest transverse offset on the area): ``transmission_fraction`` of the
    widest beam, times both. It needs a finite stretch and a plane the chief ray is not parallel to."""
    d, g = leg.d, leg.beam
    n = surf.normal
    dn = float(dot(d, n))
    side = n - dn * d
    np2 = float(dot(side, side))
    if np2 <= 0.0:
        return 0.0  # (a plane across the ray: every line meets it where the chief ray does)
    across = math.sqrt(np2)
    kx, ky = float(dot(side, g.bx)) / across, float(dot(side, g.by)) / across
    limits: list[tuple[float, Array]] = []
    active = surf.interaction.active_aperture
    if active is not None and active.shape != "none":
        window = along_area(surf, leg, cfg)
        if window is not None:
            if window[1] < 0.0:
                return 0.0
            first, last, limits = max(first, window[0]), min(last, window[0] + window[1]), window[2]
    else:
        last = min(last, leg.scene_end)
    if first > last:
        return 0.0
    scale = g.wavelength_nm * NM_TO_MM / (math.pi * g.n)
    ax, ay = g.m2x * scale / g.qx.imag * kx * kx, g.m2y * scale / g.qy.imag * ky * ky
    alpha = ax + ay
    beta = 2.0 * (ax * g.qx.real + ay * g.qy.real)
    gamma = ax * (g.qx.real * g.qx.real + g.qx.imag * g.qx.imag) + ay * (g.qy.real * g.qy.real + g.qy.imag * g.qy.imag)
    p = float(dot(leg.start - surf.origin, n)) + leg.first * dn
    s0, s1 = first - leg.first, last - leg.first

    def place(s: float) -> float:
        return -(p + dn * s) / (across * math.sqrt(alpha * s * s + beta * s + gamma))

    nodes = [s0]
    stationary_den = dn * beta - 2.0 * p * alpha
    if stationary_den != 0.0:
        stationary = (p * beta - 2.0 * dn * gamma) / stationary_den
        if s0 < stationary < s1:
            nodes.append(stationary)
    places = [place(s) for s in nodes]
    places.append(place(s1) if math.isfinite(s1) else -dn / (across * math.sqrt(alpha)))
    scales = _profile_scales_as_one(g, kx, ky)
    if scales:
        fraction = 0.0
        for z0, z1 in zip(places, places[1:]):
            fraction += abs(_beyond(z1) - _beyond(z0))
    else:
        least = 0.0 if min(places) < 0.0 < max(places) else min(abs(z) for z in places)
        fraction = MOST_LINE_CROSSINGS * exp_float(-2.0 * least * least)
    if not math.isfinite(s1):
        return fraction
    g_area = propagate(g, s0)
    for a_w, b_w in limits:
        fraction = (fraction * gaussian_part(g_area, s1 - s0, a_w, b_w) if scales else
                    min(fraction, MOST_LINE_CROSSINGS * _radial_tail(g_area, s1 - s0, a_w, b_w)))
    if active is None or active.shape == "none" or abs(dn) <= cfg.parallel_dot_tol or fraction <= 0.0:
        return fraction
    # the footprint of the beam at its widest on the stretch
    narrow_x, wide_x, q_x, slope_x = _flow_axis(g.qx, g.m2x, scale, s0, s1)
    narrow_y, wide_y, q_y, slope_y = _flow_axis(g.qy, g.m2y, scale, s0, s1)
    centre = leg.start - (float(dot(leg.start - surf.origin, n)) / dn) * d - surf.origin
    centre_uv = (float(dot(centre, surf.u_axis)), float(dot(centre, surf.v_axis)))
    foot = transmission_fraction(g.with_(qx=q_x, qy=q_y), active, centre_uv, surf.u_axis, surf.v_axis, abs(dn), cfg, quad_rule)
    rel = surf.origin + active.offset_u * surf.u_axis + active.offset_v * surf.v_axis - leg.start
    far2 = 0.0
    for unit in (side / across, np.cross(n, d) / across):
        low, high = _area_extent(active, surf.u_axis, surf.v_axis, rel, unit)
        far2 += max(low * low, high * high)
    flow = 1.0 + max(slope_x, slope_y) * math.sqrt(far2) * across / abs(dn)
    return min(fraction, foot * (wide_x * wide_y) / (narrow_x * narrow_y) * flow)


def _radial_tail(g: GState, length: float, a: float, b: Array) -> float:
    """``exp(-2 c^2 / W^2)``, ``c`` the distance of the half-plane ``a + b . rho > 0`` from the chief ray and ``W`` the beam's
    widest radius along ``b`` on a stretch of ``length``: the radial share a turning half-plane can hold."""
    bn = math.sqrt(float(dot(b, b)))
    if bn == 0.0 or a >= 0.0:
        return 1.0
    scale = g.wavelength_nm * NM_TO_MM / (math.pi * g.n)
    radii2 = 0.0
    for q, axis, m2 in ((g.qx, g.bx, g.m2x), (g.qy, g.by, g.m2y)):
        along = float(dot(b, axis)) / bn
        radii2 += along * along * max(m2 * scale * ((q.real + s) * (q.real + s) + q.imag * q.imag) / q.imag for s in (0.0, length))
    return exp_float(-2.0 * a * a / (bn * bn * radii2)) if radii2 > 0.0 else 0.0


def literal_crossing_share(g: GState, d: Array, plane_normal: Array, surface_normal: Array, gap: float, travelled: float,
                           k_sigma: float | None, cfg: NumericalConfig) -> float | None:
    """A bound of the fraction of a Gaussian beam that crosses a plane through a flat surface's chief-ray hit point before it
    meets the surface (``gap``: the surface's chief-ray parameter less the plane's, within the advance guard; ``travelled``: how
    far the chief ray came on this leg). ``None`` where the two planes do not cut across the beam (the plane parallel to the
    surface: nothing of the beam tells them apart), where there is no stated negligible level, and where a line within ``K``
    radii may start beyond either plane on this leg (``K w tan >= travelled``): that light is on neither side of the half-plane.

    A ray at transverse offset ``rho`` parallel to the chief ray meets the plane at ``s_T - rho . n_T / (d . n_T)`` and the surface
    at ``s_S - rho . n_S / (d . n_S)``, so it crosses the plane first in the half-plane ``gap + rho . b > 0``, ``b`` the difference
    of those two vectors across the ray (``gaussian_part``). The flow's lines are not parallel: a line at offset ``rho`` has a
    slope of at most ``|rho| max|w' / w|``, and over the at most ``|rho| tan`` of chief ray the two crossings lie apart (``tan``
    the larger tilt of the two planes to the ray) it moves the half-plane's edge by at most
    ``c |rho|^2``, ``c = 2 max|w' / w| tan^2 / |b|``. So at most the band of height ``c |rho|^2`` beside that edge joins the
    half-plane: its share is at most the profile's peak ``sqrt(2 / pi) / w`` across the edge times ``c`` times the mean
    of ``|rho|^2``, ``w^2 / 2``. That holds where the band is thin against the beam (``c K^2 w <= 1`` within ``K`` radii; ``None``
    otherwise, a beam diverging as fast as the planes' tilt lets through), with the tail beyond ``K`` radii added. It is about
    0.5 of the beam for a 45 degree fold and a plane across the beam."""
    dn_t, dn_s = float(dot(d, plane_normal)), float(dot(d, surface_normal))
    if k_sigma is None or abs(dn_t) <= cfg.parallel_dot_tol or abs(dn_s) <= cfg.parallel_dot_tol:
        return None
    b = plane_normal / dn_t - surface_normal / dn_s
    b = b - float(dot(b, d)) * d
    bn = math.sqrt(float(dot(b, b)))
    if bn <= cfg.parallel_dot_tol:
        return None
    w = widest(g, (0.0,))
    scale = g.wavelength_nm * NM_TO_MM / (math.pi * g.n)
    slope = max(_flow_axis(g.qx, g.m2x, scale, 0.0, 0.0)[3], _flow_axis(g.qy, g.m2y, scale, 0.0, 0.0)[3])
    tan = max(_tilt(plane_normal, d, dn_t), _tilt(surface_normal, d, dn_s))
    if k_sigma * widest(g, (0.0, -travelled)) * tan >= travelled:
        return None
    thin = 2.0 * slope * tan * tan / bn
    if thin * k_sigma * k_sigma * w > 1.0:
        return None
    band = math.sqrt(2.0 / math.pi) * thin * 0.5 * w
    return min(1.0, gaussian_part(g, 0.0, gap, b) + band + exp_float(-2.0 * k_sigma * k_sigma))


def _tilt(normal: Array, d: Array, dn: float) -> float:
    """``|normal across d| / |d . normal|``: how far along the ray a plane moves per unit of transverse offset."""
    side = normal - dn * d
    return math.sqrt(max(float(dot(side, side)), 0.0)) / abs(dn)


def reach_stretch(leg: Leg, k_sigma: float, cfg: NumericalConfig) -> tuple[float, float]:
    """The stretch of the chief ray a finite leg's light can occupy within ``K`` beam radii: the leg's, out to the farthest
    its tilted start and end planes reach within ``K`` times the beam's widest radius."""
    reach = k_sigma * widest(leg.beam, (0.0, leg.length)) * (1.0 + PREFILTER_REL_MARGIN)
    d = leg.d
    dn_s = float(dot(d, leg.start_normal))
    lo = min(leg.start_offset, 0.0)
    if abs(dn_s) > cfg.parallel_dot_tol:
        lo -= reach * _tilt(leg.start_normal, d, dn_s)
    hi = leg.s_end
    if leg.end_offset is not None:
        dn_e = float(dot(d, leg.end_normal))
        hi = max(leg.s_end, leg.end_offset) + reach * _tilt(leg.end_normal, d, dn_e)
    return lo, hi


def settled(leg: Leg, k_sigma: float | None, cfg: NumericalConfig) -> float:
    """Where on a leg that leaves the scene every line within ``K`` beam radii has left its start surface, the chief ray too:
    the least ``S >= 0`` with ``S >= start_offset + K tan w(S)``, ``tan`` the start plane's tilt to the ray (0 for a plane
    across the ray; 0 too where the beam's far field outruns the tilted plane, ``K tan theta >= 1``, so no such ``S``
    exists: then the bound is the flow's from the chief ray's start)."""
    dn_s = float(dot(leg.d, leg.start_normal))
    if k_sigma is None or abs(dn_s) <= cfg.parallel_dot_tol:
        return 0.0
    pitch = k_sigma * _tilt(leg.start_normal, leg.d, dn_s)
    if pitch <= 0.0:
        return 0.0

    def bound(s: float) -> float:
        return leg.start_offset + pitch * widest(leg.beam, (s - leg.first,))

    s = 0.0
    for _ in range(SETTLE_STEPS):
        s = max(s, bound(s))
    s = leg.start_offset + SETTLE_MARGIN * (s - leg.start_offset)
    return s if s >= bound(s) else 0.0


def near_leg(leg: Leg, end: float) -> Leg:
    """The stretch of a leg that leaves the scene before ``end`` (``settled``), as a finite leg ending at a plane across the ray."""
    return Leg(leg.start, leg.d, end, leg.start_normal, leg.start_offset, leg.d, end, leg.first, end - leg.first, leg.beam)


def leg_share(beam: BeamView, surf: SurfaceG, leg: Leg | None, cfg: NumericalConfig, quad_rule: tuple[Array, Array]) -> float | None:
    """The bound of the power of a straight leg's beam that crosses the observation plane ``surf`` while the chief ray does not.

    * On a finite leg: the larger of the rays' picture (``part_on_sheet``, which also bounds the light that meets the plane
      where a tilted or curved start or end surface lets it) and the energy flow's on the stretch the light occupies
      (``flow_crossing``). The chief ray does not cross the plane between its start and its end (``[0, s_end]``), so the
      stretch keeps out to the reach of those surfaces (``Leg.first``, ``reach_stretch``) only on a side where it does not
      take in the chief ray's crossing behind them.
    * On a leg that leaves the scene there is no widest radius, so no rays' picture: the flow's on the stretch from where
      every line has left the start surface (``settled``) out of the scene, and the finite leg's bound before that.

    ``None``: no leg (``Tracer._leg``: no valid Gaussian state, a curved surface whose reach is not bounded)."""
    if leg is None:
        return None
    k_sigma = beam_sigmas(beam)
    if not math.isfinite(leg.length):
        start = settled(leg, k_sigma, cfg)
        far = beam_part_W(beam, flow_crossing(surf, leg, start, math.inf, cfg, quad_rule))
        if start <= leg.first:  # (no stretch before it: a plane across the ray that is the surface itself)
            return far
        return far + float(leg_share(beam, surf, near_leg(leg, start), cfg, quad_rule))  # type: ignore[arg-type]
    rays = part_on_sheet(beam, surf, leg, cfg, quad_rule)
    window = along_area(surf, leg, cfg)
    if window is not None and window[1] < 0.0:  # (the active area lies wholly before the leg's start or beyond its end)
        return 0.0
    dn = float(dot(leg.d, surf.normal))
    base = float(dot(leg.start - surf.origin, surf.normal))

    def beyond_chief(s: float, chief: float) -> bool:  # (the stretch out to ``s`` does not take in the chief ray's crossing)
        return (base + s * dn) * (base + chief * dn) >= 0.0

    lo, hi = (leg.first, leg.first + leg.length) if k_sigma is None else reach_stretch(leg, k_sigma, cfg)
    first = next((s for s in (lo, leg.first) if beyond_chief(s, 0.0)), 0.0)
    last = next((s for s in (hi, leg.first + leg.length) if beyond_chief(s, leg.s_end)), leg.s_end)
    return max(rays, beam_part_W(beam, flow_crossing(surf, leg, first, last, cfg, quad_rule)))  # type: ignore[type-var]


def beside_area_share(surf: SurfaceG, leg: Leg, k_sigma: float | None, cfg: NumericalConfig, quad_rule: tuple[Array, Array]) -> float:
    """The energy flow's fraction on the active area of a plane the chief ray crosses beside that area (an uncounted crossing,
    ``leg``: from the leg's start, with no end surface): the parallel-ray footprint at the crossing says nothing about a nearly
    parallel plane's area far from it."""
    return flow_crossing(surf, leg, settled(leg, k_sigma, cfg), math.inf, cfg, quad_rule)


def within_ray_envelope(surf: SurfaceG, beam: BeamView, rows: NDArray[np.bool_], s_end: float) -> bool:
    """Without a Gaussian bound: whether the observation plane ``surf`` comes within ``RAY_ENVELOPE_MARGIN`` times the envelope
    of the live pupil rays ``rows`` of the chief ray on the leg up to ``s_end``. The rays are the evidence of where
    the light is: at chief-ray parameter ``s`` they lie within ``R0 + sigma s`` of it (their largest offset at the leg's
    start and largest slope). ``False`` without a live ray: nothing says where the light is."""
    live = np.flatnonzero(rows)
    if live.size == 0:
        return False
    o0, d0 = beam.o[0], beam.d[0]
    rel = beam.o[live] - o0
    rho = rel - dot(rel, d0)[:, None] * d0
    slopes = beam.d[live] / dot(beam.d[live], d0)[:, None] - d0
    r0 = RAY_ENVELOPE_MARGIN * math.sqrt(float(dot(rho, rho).max()))
    sigma = RAY_ENVELOPE_MARGIN * math.sqrt(float(dot(slopes, slopes).max()))
    n = surf.normal
    dn = float(dot(d0, n))
    side = n - dn * d0
    across = math.sqrt(max(float(dot(side, side)), 0.0))
    if across <= 0.0:
        return False
    base = float(dot(o0 - surf.origin, n))
    s0, s1 = 0.0, s_end
    active = surf.interaction.active_aperture
    if active is not None and active.shape != "none":
        centre = surf.origin + active.offset_u * surf.u_axis + active.offset_v * surf.v_axis - o0
        low, high = _area_extent(active, surf.u_axis, surf.v_axis, centre, d0)
        s0, s1 = max(s0, low), min(s1, high)
        if s0 > s1:
            return False
        low, high = _area_extent(active, surf.u_axis, surf.v_axis, centre, np.cross(n, d0) / across)
        if (low > 0.0 or high < 0.0) and min(abs(low), abs(high)) > r0 + sigma * s1:
            return False
    if not math.isfinite(s1):
        return abs(dn) < across * sigma or abs(base + s0 * dn) <= across * (r0 + sigma * s0)
    f0, f1 = base + s0 * dn, base + s1 * dn
    return f0 * f1 <= 0.0 or abs(f0) <= across * (r0 + sigma * s0) or abs(f1) <= across * (r0 + sigma * s1)


def ahead_of_leg(index: SceneIndex, surf: SurfaceG, beam: BeamView) -> bool:
    """Whether any point of the plane ``surf`` lies ahead of the leg's start: of its active area, or without one of its part
    within the reach of the leg's start (``SceneIndex.reach_along``, a disc)."""
    o0, d0 = beam.o[0], beam.d[0]
    active = surf.interaction.active_aperture
    if active is not None and active.shape != "none":
        centre = surf.origin + active.offset_u * surf.u_axis + active.offset_v * surf.v_axis - o0
        return _area_extent(active, surf.u_axis, surf.v_axis, centre, d0)[1] > 0.0
    n = surf.normal
    distance = float(dot(o0 - surf.origin, n))
    reach = index.reach_along(o0, d0)
    if abs(distance) > reach:
        return False
    in_plane = d0 - float(dot(d0, n)) * n
    return -distance * float(dot(n, d0)) + math.sqrt(reach * reach - distance * distance) * math.sqrt(max(float(dot(in_plane, in_plane)), 0.0)) > 0.0


def start_beyond(beam: BeamView, surf: SurfaceG, leg: Leg, plane: tuple[Array, float] | None, cfg: NumericalConfig) -> float | None:
    """The most power of the beam that leaves the leg's start surface beyond the plane ``surf`` the chief ray crosses
    on the leg (those rays never cross it there): the Gaussian share of the half-plane ``s_p(rho) < s_s(rho)`` (the
    complement of ``part_on_sheet``'s start half-plane), with the start surface as the plane ``plane`` (its unit normal
    and the offset beyond the chief ray's start of its crossing: ``StartSurface``). ``None``: not bounded."""
    a_point, d = leg.start, leg.d
    n_p = surf.normal
    dn_p = float(dot(d, n_p))
    if abs(dn_p) <= cfg.parallel_dot_tol:
        return None
    if plane is None:
        return None
    normal_s, start_offset = plane
    dn_s = float(dot(d, normal_s))
    if abs(dn_s) <= cfg.parallel_dot_tol:
        return None
    s_p = float(dot(surf.origin - a_point, n_p)) / dn_p
    b = normal_s / dn_s - n_p / dn_p
    b = b - float(dot(b, d)) * d
    return beam_part_W(beam, gaussian_part(leg.beam, leg.length, start_offset - s_p, -b))


def flow_stretch(leg: Leg, normal: Array, offset: float, k_sigma: float, cfg: NumericalConfig) -> tuple[float, float] | None:
    """The stretch ``[offset - h, offset + h]`` of the chief ray (parameters from its start, ``leg.start``) on which every
    energy-flow line within ``K`` beam radii meets the plane that crosses the chief ray at ``offset`` with the unit
    ``normal``.

    A line at a place ``xi`` of the profile (``|xi| <= K``) lies at ``rho(s)``, ``|rho(s)| <= K W(s)`` (``W`` the larger
    radius of the two axes), and meets the plane where ``s = offset + beta . rho(s)``, ``|beta|`` the plane's tilt to the
    ray (``_tilt``). So a root lies within ``h`` of ``offset`` when ``h >= K |beta| W`` over that stretch (the least such
    ``h`` by ``settled``'s steps; ``W`` is widest at an end of a stretch). Beyond the stretch, ``W`` grows at most at the
    far-field half angle ``theta``, so a root there would need ``K |beta| theta >= 1``. ``None``: that holds (no such
    stretch), or the plane runs along the ray."""
    d = leg.d
    dn = float(dot(d, normal))
    if abs(dn) <= cfg.parallel_dot_tol:
        return None
    pitch = k_sigma * _tilt(normal, d, dn)
    if pitch <= 0.0:
        return offset, offset
    g = leg.beam
    scale = g.wavelength_nm * NM_TO_MM / (math.pi * g.n)
    theta = max(math.sqrt(m2 * scale / q.imag) for q, m2 in ((g.qx, g.m2x), (g.qy, g.m2y)))
    if pitch * theta >= 1.0:
        return None

    def reach(h: float) -> float:  # (K |beta| W over the stretch of half length h)
        return pitch * widest(g, (offset - h - leg.first, offset + h - leg.first))

    h = reach(0.0)
    for _ in range(SETTLE_STEPS):
        h = max(h, reach(h))
    h *= SETTLE_MARGIN
    return (offset - h, offset + h) if h >= reach(h) else None


def sheet_flow_share(beam: BeamView, surf: SurfaceG, leg: Leg, plane: tuple[Array, float], beyond: bool,
                     cfg: NumericalConfig) -> float | None:
    """The paraxial energy flow's bound of the fraction of the beam whose lines meet the plane ``plane`` (unit normal, and
    the offset of its crossing along the chief ray from the chief ray's start, ``leg.start``) beyond (``beyond``) or
    before the plane ``surf`` the chief ray crosses on the leg: the leg's reflecting start surface sends that part beyond
    ``surf`` (``start_flow_share``); the surface the chief ray meets next takes that part before it.

    A line at ``xi`` (density ``exp(-2 |xi|^2)``) lies at ``rho = (xi_x w_x(s), xi_y w_y(s))`` on the beam's axes; within
    ``K`` radii it meets ``plane`` at a root ``s*`` on ``flow_stretch`` (behind the chief ray's start too, where the plane is
    tilted), and lies beyond ``surf`` there when ``a + c . rho(s*) > 0`` (``a`` the plane's offset less ``surf``'s, ``c`` the
    difference of their tilts across the ray: the parallel rays' half-plane of ``start_beyond``), before it when
    ``-a - c . rho(s*) > 0``. Write the event ``alpha + gamma . rho > 0``. Each ``w_i(s*)`` lies within the axis's narrowest
    and widest radius ``m_i <= w_i <= M_i`` on that stretch, so ``gamma . rho <= sum_i M_i (gamma_i xi_i)^+``. With
    ``A = M_x gamma_x xi_x`` and ``B = M_y gamma_y xi_y`` (independent Gaussians), ``P(A^+ + B^+ > -alpha) <= P(A > -alpha)
    / 2 + P(B > -alpha) / 2 + P(A + B > -alpha)``; where the profile across ``gamma`` scales as one (a round beam, ``gamma``
    along an axis) the radii are widest at one ``s`` and the share is ``P(A + B > -alpha)`` alone. The lines beyond ``K``
    radii add ``exp(-2 K^2)``; ``alpha >= 0`` (the chief ray's own crossing in the event) gives 1. ``None``: not bounded (no
    stated ``K``, a plane along the ray, no stretch)."""
    k_sigma = beam_sigmas(beam)
    if k_sigma is None:
        return None
    d = leg.d
    n_p = surf.normal
    dn_p = float(dot(d, n_p))
    normal_s, offset = plane
    dn_s = float(dot(d, normal_s))
    if abs(dn_p) <= cfg.parallel_dot_tol or abs(dn_s) <= cfg.parallel_dot_tol:
        return None
    stretch = flow_stretch(leg, normal_s, offset, k_sigma, cfg)
    if stretch is None:
        return None
    s_p = float(dot(surf.origin - leg.start, n_p)) / dn_p
    c = n_p / dn_p - normal_s / dn_s
    c = c - float(dot(c, d)) * d
    alpha = offset - s_p if beyond else s_p - offset
    if alpha >= 0.0:
        return 1.0
    if not beyond:
        c = -c
    g = leg.beam
    scale = g.wavelength_nm * NM_TO_MM / (math.pi * g.n)
    c_x, c_y = float(dot(c, g.bx)), float(dot(c, g.by))
    s0, s1 = stretch[0] - leg.first, stretch[1] - leg.first
    m_x = _flow_axis(g.qx, g.m2x, scale, s0, s1)[1]
    m_y = _flow_axis(g.qy, g.m2y, scale, s0, s1)[1]
    sx, sy = abs(c_x) * m_x, abs(c_y) * m_y

    def tail(sigma: float) -> float:  # P(N(0, sigma^2 / 4) > -alpha)
        return 0.5 * exact_erfc_float(math.sqrt(2.0) * -alpha / sigma) if sigma > 0.0 else 0.0

    both = tail(math.sqrt(sx * sx + sy * sy))
    share = both if _profile_scales_as_one(g, c_x, c_y) else 0.5 * tail(sx) + 0.5 * tail(sy) + both
    return min(1.0, share + exp_float(-2.0 * k_sigma * k_sigma))


def start_flow_share(beam: BeamView, surf: SurfaceG, leg: Leg, plane: tuple[Array, float], end: tuple[Array, float] | None,
                     cfg: NumericalConfig) -> float | None:
    """The paraxial energy flow's bound of the fraction of the beam that leaves the leg's start surface beyond the plane
    ``surf`` the chief ray crosses on the leg (so never crosses it on the leg), behind the chief ray's start too
    (``sheet_flow_share`` of the start surface's plane, ``beyond``).

    The start surface is the plane ``plane`` (``StartSurface.flow``: a curved surface shifted as far beyond the chief ray's
    start as it reaches for the lines within ``K`` radii over the stretch they meet it on, so a line leaves the surface where
    it meets that plane or before it). A line that leaves the start before the plane and does not meet the leg's next
    surface first crosses the plane (it ends beyond it); one that leaves the actual surface beyond the plane but meets the
    start plane before it crosses the plane between the two, provided it is not stopped first: the next surface must lie
    beyond the stretch (``end``: the plane that stands for the next physical surface before every hit of the lines within
    ``K`` radii, its unit normal and the offset of its crossing; ``None``: none ahead). ``None``: not bounded (also an end
    surface within the stretch)."""
    k_sigma = beam_sigmas(beam)
    if k_sigma is None:
        return None
    if end is not None:
        stretch = flow_stretch(leg, plane[0], plane[1], k_sigma, cfg)
        before_end = flow_stretch(leg, end[0], end[1], k_sigma, cfg)
        if stretch is None or before_end is None or not before_end[0] > stretch[1]:
            return None
    return sheet_flow_share(beam, surf, leg, plane, True, cfg)


# ------------------------------------------------------------------------------------------------- crossing group
def observation_crossing(index: SceneIndex, chief_s: int, times: Array, cfg: NumericalConfig) -> list[int]:
    """The observation planes of one crossing: the chief ray's next surface ``chief_s`` is an observation plane with no
    physical surface within ``ray_min_advance_mm`` of it (else ``coincident_crossing``). Every observation plane the
    chief ray crosses within ``ray_min_advance_mm`` of it belongs to the crossing, whatever its orientation (planes
    through one point, coplanar or opposite planes, one just behind another): the tracer cannot step between them, and
    a plane left behind the minimum advance would never be crossed. Transparent planes come before absorbing ones, then
    by id (never by declaration order); each samples every ray where that ray crosses it (``Tracer._observation_event``),
    so nearby planes are not collapsed onto one point."""
    surfaces = index.surfaces
    reach = times[chief_s, 0] + cfg.ray_min_advance_mm
    group = [i for i in index.observe_ids if times[i, 0] <= reach]
    return sorted(group, key=lambda i: (surfaces[i].interaction.absorbing, surfaces[i].surface_id))


def coincident_crossing(index: SceneIndex, chief_s: int, times: Array, cfg: NumericalConfig) -> list[int] | None:
    """Observation planes the chief ray crosses at a physical surface's hit point, then that surface.

    A plane within ``ray_min_advance_mm`` of the nearest physical surface along the chief ray (a monitor exactly on a
    mirror face, a lens vertex, a cube face, an element plane), on either side of it, could not be crossed after the
    surface's interaction: its intersection would lie behind the minimum advance. Such planes record the beam arriving
    at the surface first, exactly as a plane just on the incident side would (transparent observers before an absorbing
    one, then by id), each ray where it crosses the plane (its point and optical path length there, also in the ray
    data); a beam that leaves the surface back toward the incident side crosses them again there
    (``Tracer._pending_crossings``). ``None``: no physical surface with a coincident observation plane."""
    surfaces = index.surfaces
    window = times[chief_s, 0] + cfg.ray_min_advance_mm
    physical = [int(i) for i in np.flatnonzero(times[:, 0] <= window) if surfaces[i].interaction.kind != "observe"]
    if not physical:
        return None
    lead = min(physical, key=lambda i: (float(times[i, 0]), i))
    reach = times[lead, 0] + cfg.ray_min_advance_mm
    observers = [i for i in index.observe_ids if times[i, 0] <= reach]
    if not observers:
        return None
    observers.sort(key=lambda i: (surfaces[i].interaction.absorbing, surfaces[i].surface_id))
    return [*observers, lead]


def crossing_group(index: SceneIndex, chief_s: int, times: Array, cfg: NumericalConfig) -> tuple[list[int], int | None]:
    """The surfaces of the chief ray's next crossing (``chief_s``: its nearest surface; ``times``: the ray parameters per
    surface, the chief ray's in column 0), in the order they act, and its physical surface (``None``: only observation
    planes, ``observation_crossing``)."""
    coincident = coincident_crossing(index, chief_s, times, cfg)
    if coincident is not None:
        return coincident, coincident[-1]
    if index.surfaces[chief_s].interaction.kind == "observe":
        return observation_crossing(index, chief_s, times, cfg), None
    return [chief_s], chief_s


def first_blocking(blocking_ids: NDArray[np.intp], times: Array, exclude: list[int] | tuple[int, ...] = ()) -> tuple[Array, NDArray[np.intp]]:
    """Per ray, the parameter and scene index (-1: none) of the first surface that stops or turns it among
    ``blocking_ids``, leaving out ``exclude``. ``times``: the step's ray parameters per surface as the blocking surfaces
    see them (``Tracer._blocking_view``: an absorbing plane with an active area stops only the rays that cross it on that
    area), or the whole planes."""
    ids = blocking_ids[~np.isin(blocking_ids, exclude)] if exclude else blocking_ids
    n_rays = times.shape[1]
    if ids.size == 0:
        return np.full(n_rays, np.inf), np.full(n_rays, -1, dtype=np.intp)
    sub = times[ids]
    k = np.argmin(sub, axis=0)
    t = sub[k, np.arange(n_rays)]
    return t, np.where(np.isfinite(t), ids[k], -1)


# ------------------------------------------------------------------------------------------- leg partial crossings
#: ``LegCrossing.outcome``: the plane is recorded (``PartialCrossing``)
RECORDED = "recorded"
#: no live pupil ray reaches it first and its Gaussian share is certainly below the negligible level (``beyond_reach``)
PREFILTERED = "prefiltered"
#: the bound (the Gaussian share, or the whole beam where a pupil ray reaches it first) is zero or negligible
NEGLIGIBLE = "negligible"
#: no Gaussian bound (no valid Gaussian state, a curved surface whose reach is not bounded), no live pupil ray crosses the
#: plane first and the plane lies beyond the reach of those rays' envelope (``within_ray_envelope``): nothing is recorded
NO_BOUND_NO_RAY = "no_bound_no_ray"


@dataclass(frozen=True)
class LegCrossing:
    """What ``leg_partial_crossings`` decided for one observation plane (scene index ``surface_index``): its ``outcome``,
    the Gaussian share of the beam that crosses it first (``share_W``, ``leg_share``; ``None``: no bound, or not computed
    for a pre-filtered plane), the bound compared with the negligible level (``bound_W``: the share; the whole beam where
    a pupil ray crosses the plane first and the share is nil or unknown or the leg leaves the scene, and where no
    Gaussian bound exists and the plane comes within the rays' envelope; 0 otherwise), the live pupil rays that cross it
    first, and the record (``RECORDED`` only)."""

    surface_index: int
    outcome: str
    share_W: float | None
    bound_W: float | None
    rays: int
    crossing: PartialCrossing | None = None


def leg_end_candidates(index: SceneIndex, crossed: tuple[int, ...], leg_partial: tuple[int, ...]) -> list[int]:
    """The observation planes a leg's end checks: those the chief ray does not cross on it and not already recorded."""
    return [si for si in index.observe_ids if si not in crossed and si not in leg_partial]


def pupil_part(surf: SurfaceG, beam: BeamView, times: Array, rows: NDArray[np.bool_], si: int, ends: Array, cfg: NumericalConfig) -> int:
    """How many of the live pupil rays ``rows`` cross observation plane ``surf`` (scene index ``si``) before ``ends`` (per
    ray: where its light stops or turns on this leg), on the plane's active area and inside its acceptance cone."""
    rays = np.flatnonzero(rows & beam.annulus)
    if rays.size == 0:
        return 0
    t = times[si, rays]
    hit = np.isfinite(t) & (t < ends[rays])
    inter = surf.interaction
    if bool(hit.any()) and (inter.active_aperture is not None or inter.acceptance_half_angle_deg is not None):
        crossing = np.flatnonzero(hit)
        rel = beam.o[rays[crossing]] + t[crossing][:, None] * beam.d[rays[crossing]] - surf.origin
        received = np.ones(crossing.size, dtype=bool)
        if inter.active_aperture is not None:
            received &= aperture_inside(inter.active_aperture, dot(rel, surf.u_axis), dot(rel, surf.v_axis),
                                        cfg.aperture_edge_tol_mm)
        if inter.acceptance_half_angle_deg is not None:
            arriving = beam.d[rays[crossing]]
            received &= incidence_deg(np.abs(dot(arriving, surf.normal)), arriving, surf.normal) <= inter.acceptance_half_angle_deg
        hit[crossing] = received
    return int(hit.sum())


def _tube_clear(index: SceneIndex, leg: Leg, cand: NDArray[np.intp], k_sigma: float, cfg: NumericalConfig) -> NDArray[np.bool_]:
    """Of the planes ``cand``, those the beam's ``K``-radius tube misses on a finite leg (``beyond_reach``); all False where the
    tube is not cheap: a start or end surface along the ray."""
    a_point, d, s_e = leg.start, leg.d, leg.s_end
    dn_s = float(dot(d, leg.start_normal))
    if abs(dn_s) <= cfg.parallel_dot_tol or leg.end_offset is None or not math.isfinite(s_e):
        return np.zeros(cand.size, dtype=bool)
    g = leg.beam
    scale = g.wavelength_nm * NM_TO_MM / (math.pi * g.n)
    reach = k_sigma * widest(g, (0.0, leg.length)) * (1.0 + PREFILTER_REL_MARGIN)
    s_lo, s_hi = reach_stretch(leg, k_sigma, cfg)
    normals = index.observe_normal[cand]
    along = dot(normals, d)
    base = dot(a_point - index.observe_origin[cand], normals)  # (the signed distance of the leg's start from each plane)
    side = normals - along[:, None] * d
    radius = reach * np.sqrt(np.maximum(dot(side, side), 0.0))
    at_lo, at_hi = base + s_lo * along, base + s_hi * along
    margin = PREFILTER_REL_MARGIN * (np.abs(base) + np.abs(s_lo * along) + np.abs(s_hi * along) + radius) + PREFILTER_ABS_MARGIN_MM
    clear = (np.minimum(at_lo, at_hi) - radius > margin) | (np.maximum(at_lo, at_hi) + radius < -margin)
    parallel = np.abs(along) <= cfg.parallel_dot_tol
    if not bool(parallel.any()):
        return clear
    # planes along the leg: the beam's widest radius over the leg and the active area's stretch along it
    shape = index.area_shape[cand]
    u_d, v_d = dot(index.area_u[cand], d), dot(index.area_v[cand], d)
    half = index.area_half[cand]
    extent = np.where(shape == 1, half[:, 0] * np.sqrt(u_d * u_d + v_d * v_d),
                      half[:, 0] * np.abs(u_d) + half[:, 1] * np.abs(v_d)) + cfg.aperture_edge_tol_mm
    centre = dot(index.area_centre[cand] - a_point, d)
    s_first = np.where(shape == 0, leg.first, np.minimum(centre - extent, leg.first)) - leg.first
    s_last = np.where(shape == 0, leg.first + leg.length, np.maximum(centre + extent, leg.first + leg.length)) - leg.first
    widest2 = np.zeros(cand.size)
    for q, m2 in ((g.qx, g.m2x), (g.qy, g.m2y)):
        for s_at in (s_first, s_last):
            widest2 = np.maximum(widest2, m2 * scale * ((q.real + s_at) * (q.real + s_at) + q.imag * q.imag) / q.imag)
    apart = k_sigma * np.sqrt(widest2) * (1.0 + PREFILTER_REL_MARGIN)
    clear_along = np.abs(base) > apart + PREFILTER_REL_MARGIN * (np.abs(base) + apart) + PREFILTER_ABS_MARGIN_MM
    return np.where(parallel, clear_along, clear)


def _flow_clear(index: SceneIndex, leg: Leg, cand: NDArray[np.intp], k_sigma: float, s_lo: float, s_hi: float,
                cfg: NumericalConfig) -> NDArray[np.bool_]:
    """Of the planes ``cand``, those whose normalized distance from the chief ray ``h(s) = |f(s)| / (N w_along(s))``
    (``flow_crossing``) stays above ``K`` on the stretch ``[s_lo, s_hi]`` (``inf``: out of the scene) cut to the active area's
    extent along the ray: at either end, at the stationary point inside, or in the limit. Every line that meets the plane
    there lies beyond ``K`` radii in the profile, so ``flow_crossing`` is at most ``MOST_LINE_CROSSINGS exp(-2 K^2)`` of the
    beam, the line's negligible level at most. A plane across the ray has no such line, and neither has a stretch the area
    lies off."""
    d, g = leg.d, leg.beam
    normals = index.observe_normal[cand]
    along = dot(normals, d)
    side = normals - along[:, None] * d
    kx, ky = dot(side, g.bx), dot(side, g.by)  # (not normalized: N^2 w_along^2 is the quadratic)
    scale = g.wavelength_nm * NM_TO_MM / (math.pi * g.n)
    ax, ay = g.m2x * scale / g.qx.imag * kx * kx, g.m2y * scale / g.qy.imag * ky * ky
    alpha = ax + ay
    beta = 2.0 * (ax * g.qx.real + ay * g.qy.real)
    gamma = ax * (g.qx.real * g.qx.real + g.qx.imag * g.qx.imag) + ay * (g.qy.real * g.qy.real + g.qy.imag * g.qy.imag)
    p = dot(leg.start - index.observe_origin[cand], normals) + leg.first * along
    shape = index.area_shape[cand]
    u_d, v_d = dot(index.area_u[cand], d), dot(index.area_v[cand], d)
    half = index.area_half[cand]
    extent = np.where(shape == 1, half[:, 0] * np.sqrt(u_d * u_d + v_d * v_d),
                      half[:, 0] * np.abs(u_d) + half[:, 1] * np.abs(v_d)) + cfg.aperture_edge_tol_mm
    centre = dot(index.area_centre[cand] - leg.start, d)
    lo = np.where(shape == 0, s_lo, np.maximum(s_lo, centre - extent)) - leg.first
    hi = np.where(shape == 0, min(s_hi, leg.scene_end), np.minimum(s_hi, centre + extent)) - leg.first
    finite = np.isfinite(hi)
    hi_at = np.where(finite, hi, 0.0)

    def h2(s: Array) -> Array:
        f = p + along * s
        return f * f / (alpha * s * s + beta * s + gamma)

    with np.errstate(divide="ignore", invalid="ignore"):
        f_lo, f_hi = p + along * lo, np.where(finite, p + along * hi_at, along)
        den = along * beta - 2.0 * p * alpha
        star = (p * beta - 2.0 * along * gamma) / den
        inside = (den != 0.0) & (lo < star) & (star < hi)
        at_star = np.where(inside, h2(np.where(inside, star, 0.0)), np.inf)
        at_hi = np.where(finite, h2(hi_at), along * along / alpha)
        least = np.minimum(np.minimum(h2(lo), at_hi), at_star)
    limit = k_sigma * k_sigma * (1.0 + PREFILTER_REL_MARGIN) ** 2
    return (alpha <= 0.0) | (lo > hi) | ((least > limit) & ~(f_lo * f_hi < 0.0))


def beyond_reach(index: SceneIndex, beam: BeamView, leg: Leg | None, cand: NDArray[np.intp], cfg: NumericalConfig) -> NDArray[np.bool_]:
    """Of the observation planes ``cand`` (not crossed by the chief ray on this leg), those whose bound (``leg_share``) is
    certainly at most the line's negligible level (``K`` makes ``0.5 erfc(sqrt(2) K) <= 0.5 exp(-2 K^2)`` an eighth of it for
    the beam's power, ``MOST_LINE_CROSSINGS exp(-2 K^2)`` all of it):

    * a plane across the leg that misses the tube of radius ``R = K w`` around the chief ray between the leg's start
      and end surfaces (``w`` the beam's widest radius on the leg; the tube's ends the farthest the start and end
      planes reach within ``R``): every ray parallel to the chief ray within ``R`` meets the plane before the start
      surface, or every one after the end surface, so one of the two half-planes of ``part_on_sheet`` lies at least
      ``R`` from the chief ray and its share is at most that eighth;
    * a plane along the leg (parallel to the chief ray) more than ``K`` times the beam's widest radius from it, that
      radius taken over the leg and its active area's stretch along the ray (``along_area`` keeps to it): its share
      (``gaussian_crossing``: at most twice the share beyond it there) is at most a quarter;
    * and, for both, the energy flow's (``flow_crossing``) on the stretch ``reach_stretch`` is below that level too
      (``_flow_clear``: the tube's radius is the leg's widest, the stretch runs beyond it).

    A leg that leaves the scene has no tube of its own: the flow's from where every line has left the start surface out of
    the scene (``settled``), and both tests on the finite stretch before that (``near_leg``). The leg's start and end planes
    and the stretch of the chief ray the widest radius is taken over are the leg's (``Tracer._leg``: a curved surface as a
    plane, ``_curved_plane``, shifted as far out as it reaches). Nothing is marked (all False) where that does not hold
    cheaply: no leg (no valid Gaussian state, a curved surface whose reach is not bounded), a start or end surface along
    the ray."""
    far = np.zeros(cand.size, dtype=bool)
    if leg is None or cand.size == 0:
        return far
    k_sigma = beam_sigmas(beam)
    if k_sigma is None or abs(float(dot(leg.d, leg.start_normal))) <= cfg.parallel_dot_tol:
        return far
    if not math.isfinite(leg.length):
        start = settled(leg, k_sigma, cfg)
        clear = _flow_clear(index, leg, cand, k_sigma, start, math.inf, cfg)
        return clear if start <= leg.first else clear & beyond_reach(index, beam, near_leg(leg, start), cand, cfg)
    clear = _tube_clear(index, leg, cand, k_sigma, cfg)
    if not bool(clear.any()):
        return clear
    return clear & _flow_clear(index, leg, cand, k_sigma, *reach_stretch(leg, k_sigma, cfg), cfg)


def leg_partial_crossings(index: SceneIndex, beam: BeamView, times: Array, blocking: Array, candidates: list[int],
                          crossed: tuple[int, ...], end_si: int | None, leg: Leg | None, ends_at: str | None,
                          cfg: NumericalConfig, quad_rule: tuple[Array, Array], *, prefilter: bool = True,
                          traced: bool = True, beside: Array | None = None) -> tuple[LegCrossing, ...]:
    """The straight leg ends (the chief ray reaches a physical surface or an absorbing plane that stops the beam, or leaves
    the scene): per observation plane of ``candidates`` (``leg_end_candidates``: not among ``crossed``, the planes the
    chief ray crossed on the leg, nor recorded on it before) whether part of the beam crosses it before the leg's end
    (``LegCrossing``), with a bound of the power that part carries (``leg_share``: the Gaussian beam's share; the whole beam
    when a live pupil ray crosses the plane first, ``pupil_part``, where that share is nil or not known -- without a valid
    Gaussian state -- or the leg leaves the scene). Without a bound and without such a ray the rule is that light that may be
    detected and is not traced is no known zero: the whole beam when the plane comes within the reach of the live pupil
    rays' envelope (``within_ray_envelope``), nothing when it lies beyond (no evidence of light there, or no live ray
    to tell). A part at or below the line's negligible level is not recorded. ``blocking``: ``times`` as the surfaces that stop or turn a ray see them (``Tracer._blocking_view``);
    ``end_si``: the surface the leg ends at; ``leg``: ``Tracer._leg`` (``None``: no Gaussian bound); ``prefilter``: skip
    the planes ``beyond_reach`` marks that no live pupil ray reaches first (the same results, faster); ``beside``: per ray,
    where it crosses the plane of the end surface beyond that surface's support -- through a hole or past its outline
    (``Tracer._beside_end``; ``inf`` elsewhere): that light is booked at the end surface and followed as its untraced branch,
    so its ray ends there for this leg (counted once, by that branch's probe).

    A traced beam a hard edge truncated severely (``GAUSSIAN_SEVERE_TRUNCATION``) has no Gaussian state, and its diffraction falls
    off too slowly for its rays to say where it goes: the whole beam may reach any plane ahead of the leg's start
    (``ahead_of_leg``, ``PartialCrossing.diffracted``). That light is not taken out of the beam, even on an absorbing plane. A
    milder truncation keeps its Gaussian state. A beam without one for any other reason, and light the trace follows only
    to bound it (``traced`` false: a ghost a lens's rim truncates, a tiny share of the line), keep the rays' envelope.

    The tracer records each ``RECORDED`` plane's ``PartialCrossing``; an absorbing one takes that part out of the beam,
    whose power from there on follows the chief ray only: the plane joins ``power_scope``."""
    surfaces = index.surfaces
    power = beam.power_W
    # where each ray's light stops or turns (the leg's end surface ``end_si`` included), once for the leg: an absorbing
    # plane is left out of it for its own rays (those whose first such surface it is: the next one then)
    base = tuple(c for c in crossed if c != end_si)
    ends0, nearest0 = first_blocking(index.blocking_ids, blocking, base)
    if beside is not None:
        ends0 = np.minimum(ends0, beside)
    rows = beam.alive & beam.annulus
    cand = np.array(candidates, dtype=np.intp)
    if prefilter:
        far = beyond_reach(index, beam, leg, cand, cfg)
        if bool(rows.any()):  # (a live pupil ray may cross the plane before its light stops or turns)
            t_rows = times[np.ix_(cand, np.flatnonzero(rows))]
            reach = (np.isfinite(t_rows) & (t_rows < ends0[rows])).any(axis=1)
        else:
            reach = np.zeros(cand.size, dtype=bool)
    out: list[LegCrossing] = []
    for k, si in enumerate(candidates):
        absorbing = surfaces[si].interaction.absorbing
        own = absorbing and bool((nearest0[rows] == si).any())
        if prefilter and far[k] and not reach[k] and not own:
            out.append(LegCrossing(si, PREFILTERED, None, None, 0))
            continue  # (no live pupil ray reaches it first, and its Gaussian share is below the negligible level)
        if not prefilter:
            ends = first_blocking(index.blocking_ids, blocking, base + ((si,) if absorbing else ()))[0]
        else:
            ends = first_blocking(index.blocking_ids, blocking, base + (si,))[0] if own else ends0
        if beside is not None:
            ends = np.minimum(ends, beside)
        active = surfaces[si].interaction.active_aperture
        if leg is not None and math.isfinite(leg.scene_end) and (active is None or active.shape == "none"):
            ends = np.minimum(ends, leg.scene_end)  # (a plane without an active area counts what crosses it within the scene)
        rays = pupil_part(surfaces[si], beam, times, beam.alive, si, ends, cfg)
        diffracted = False
        share = leg_share(beam, surfaces[si], leg, cfg, quad_rule)
        bound = share
        # (pupil rays reach the plane there first where the parallel-ray share is nil or there is none: the whole beam)
        if rays and (leg is None or not math.isfinite(leg.length) or beam.negligible(part_on_sheet(beam, surfaces[si], leg, cfg, quad_rule))):
            bound = power
        elif bound is None:
            s_end = math.inf if end_si is None else float(times[end_si, 0])
            diffracted = traced and beam.gstate.reason is ErrorCode.GAUSSIAN_SEVERE_TRUNCATION and ahead_of_leg(index, surfaces[si], beam)
            bound = power if diffracted or within_ray_envelope(surfaces[si], beam, rows, s_end) else 0.0
        if bound <= 0.0 or beam.negligible(bound):
            out.append(LegCrossing(si, NO_BOUND_NO_RAY if share is None and not rays else NEGLIGIBLE, share, bound, rays))
            continue
        surf = surfaces[si]
        out.append(LegCrossing(si, RECORDED, share, bound, rays, PartialCrossing(
            observation_id=surf.interaction.observation_id or "", surface_id=surf.surface_id, source_id=beam.source_id,
            spectral_index=beam.spectral_index, wavelength_nm=beam.wavelength_nm, path_id=beam.path_id, power_W=min(bound, power),
            rays=rays, pupil_rays=int(rows.sum()), ends_at=ends_at, absorbing=surf.interaction.absorbing and not diffracted,
            diffracted=diffracted)))
    return tuple(out)


# ---------------------------------------------------------------------------------------------------- taken before
#: ``TakenBefore.why``: an absorbing plane whose active area takes a share of the beam before the plane
ABSORBING_PLANE = "absorbing_plane"
#: a physical surface a live pupil ray meets before the plane
PUPIL_RAY_FIRST = "pupil_ray_first"
#: the physical surface the chief ray meets next, met by a share of the beam before the plane
NEXT_SURFACE = "next_surface"
#: the reflecting surface the leg began at, that sends a share of the beam beyond the plane
START_SURFACE = "start_surface"
#: a physical surface beside the chief ray that the beam's tail may meet before the plane while no pupil ray does
#: (``Tracer._tail_surfaces``)
TAIL_BESIDE = "tail_beside"


@dataclass(frozen=True)
class TakenBefore:
    """One surface ``taken_before`` weighed: its id, why (``ABSORBING_PLANE``, ``PUPIL_RAY_FIRST``, ``NEXT_SURFACE``,
    ``START_SURFACE``; ``TAIL_BESIDE``: added by ``Tracer._tail_surfaces``), the Gaussian share of the beam it takes (``share_W``; ``None``: not bounded -- for an absorbing
    plane the pupil rays then decide, for the next or start surface it is not counted; a pupil ray's surface has none)
    and whether it counts (``taken``: above the line's negligible level, or a pupil ray meets it first). ``bound_W``: for a
    counted surface, a supported upper bound of the power it takes before the plane -- the larger of the rays' picture
    (``share_W``) and the paraxial energy flow's (``leg_share``, ``sheet_flow_share``) -- where one is stated (an absorbing
    plane, the next surface or the start surface with a Gaussian share and an energy-flow bound); ``None`` otherwise (a
    pupil ray's surface, a start surface a pupil ray leaves beyond the plane without a Gaussian bound: the plane's power
    is then out of model scope). ``lower_W``: for the next surface with a bound that stops what it meets from the beam's
    side, the power its certain share takes (``lower_share.certain_meeting_share``; ``None``: none certain)."""

    surface_id: str
    why: str
    share_W: float | None
    taken: bool
    bound_W: float | None = None
    lower_W: float | None = None


def taken_before(index: SceneIndex, beam: BeamView, blocking: Array, si: int, hit_t: Array, t_block: Array, nearest: NDArray[np.intp],
                 leg: Leg | None, leg_crossed: tuple[int, ...], lead: NextSurface | None, start: StartSurface | None,
                 cfg: NumericalConfig, quad_rule: tuple[Array, Array],
                 beside: Callable[[int], float | None] | None = None) -> tuple[TakenBefore, ...]:
    """The surfaces that take part of the beam out of it, or turn it, before it reaches the transparent plane ``si``
    (``hit_t``: each ray's parameter there) that the chief ray crosses on this leg (``leg``, ``Tracer._leg`` to that
    plane): the power the plane records (the beam's, which follows its chief ray) is then not the power that reaches it
    (``power_scope``). Weighed, in this order (a surface already counted is not weighed again):
    * an absorbing plane (the planes the chief ray already crossed on the leg, ``leg_crossed``, aside; one of the same
      crossing that stops the beam included) whose active area takes a Gaussian share of the beam before ``si`` above
      the line's negligible level (``part_on_sheet``; without a bound, when a live pupil ray meets it there first:
      ``blocking``, the step's blocking view of the ray parameters);
    * a physical surface a live pupil ray meets before ``si`` (``t_block`` / ``nearest``: each ray's first surface
      that stops or turns it, ``first_blocking``), with the bound of the fraction of the beam that may meet it before
      ``si`` where ``beside`` states one (``lower_share.possible_meeting_share``);
    * the physical surface the chief ray meets next (``lead``), when the Gaussian share of the beam that meets it before
      ``si`` is above that level or a live pupil ray meets it first (its plane, ``NextSurface``, taken without its support,
      so the share may be larger than the part that meets the surface; without a Gaussian bound (``leg`` ``None``), also
      where its support may reach the near side of ``si``, ``support_reaches``);
    * the physical surface the leg began at, when it reflected the beam (``start``) and the Gaussian share of the beam
      that leaves it beyond ``si`` (and so never crosses ``si`` on the leg) is above that level (``start_beyond`` and
      ``start_flow_share``, the smaller over the planes standing for it, the larger of the two pictures), or a live pupil
      ray does not cross ``si`` ahead on the leg (it left the surface beyond it), or, without a Gaussian bound, its support
      may reach beyond ``si`` (``support_reaches``: the pupil rays do not bound such a beam's footprint).
      Light that leaves a transmitting surface beyond the plane crossed the plane on its way to it (a partial crossing
      of the leg before), unless it lay beyond the plane all along.
    The surfaces counted are those with ``taken``. Each counted absorbing plane with a Gaussian share, the next surface
    with a plane standing for it, and the start surface with both pictures also state a supported bound of the power they
    take (``TakenBefore.bound_W``: the larger of the rays' and the energy flow's pictures; for the next surface also the
    flow's lines that meet it beyond the plane's chief-ray crossing, ``sheet_flow_share``); a pupil ray's other surface
    states none. The next surface that stops what it meets from the beam's side (``NextSurface.certain``) also states the
    power it certainly takes (``TakenBefore.lower_W``)."""
    surfaces = index.surfaces
    rows = np.flatnonzero(beam.alive & beam.annulus)
    found: list[str] = []
    out: list[TakenBefore] = []
    for other in index.observe_ids:
        surf = surfaces[other]
        if other == si or not surf.interaction.absorbing or other in leg_crossed:
            continue
        share = part_on_sheet(beam, surf, leg, cfg, quad_rule)
        part = share
        if part is None:
            t = blocking[other, rows]
            part = beam.power_W if bool((np.isfinite(t) & (t < hit_t[rows])).any()) else 0.0
        taken = part > 0.0 and not beam.negligible(part)
        bound = None
        if taken:
            found.append(surf.surface_id)
            bound = None if share is None else leg_share(beam, surf, leg, cfg, quad_rule)
        out.append(TakenBefore(surf.surface_id, ABSORBING_PLANE, share, taken, bound))
    first = rows[np.isfinite(hit_t[rows]) & (t_block[rows] <= hit_t[rows]) & (nearest[rows] >= 0)]
    met_by_ray = False  # (a live pupil ray meets the next surface itself first: weighed with it below)
    for r in first:
        other = int(nearest[r])
        if lead is not None and surfaces[other].surface_id == lead.surface_id:
            met_by_ray = True
            continue
        if surfaces[other].interaction.kind != "observe" and surfaces[other].surface_id not in found:
            found.append(surfaces[other].surface_id)
            share = None if beside is None or leg is None else beside(other)
            bound = None if share is None else beam_part_W(beam, share)
            if bound is not None and beam.negligible(bound):
                bound = None  # (a pupil ray meets it where the Gaussian bound says nothing does: no bound)
            out.append(TakenBefore(surfaces[other].surface_id, PUPIL_RAY_FIRST, None, True, bound))
    plane_o, plane_n = surfaces[si].origin, surfaces[si].normal
    chief_side = 1.0 if float(dot(beam.o[0] - plane_o, plane_n)) >= 0.0 else -1.0
    if lead is not None and lead.surface_id not in found:
        sheet = lead.resolved_sheet()
        part = part_on_sheet(beam, sheet, leg, cfg, quad_rule, area=False) if sheet is not None else None
        taken = met_by_ray or (part is not None and part > 0.0 and not beam.negligible(part))
        if leg is None and not taken:
            # (no Gaussian bound: where the surface's support may reach the near side of the plane, light may meet it there
            # first -- the pupil rays do not show the beam's extent: near a waist they lie well inside it)
            taken = support_reaches(_surface_named(surfaces, lead.surface_id), plane_o, plane_n, chief_side)
        bound = None
        if sheet is not None and leg is not None:  # (the energy flow's lines that meet the sheet before the plane)
            dn_e = float(dot(leg.d, sheet.normal))
            flow = None if abs(dn_e) <= cfg.parallel_dot_tol else sheet_flow_share(
                beam, surfaces[si], leg, (sheet.normal, float(dot(sheet.origin - leg.start, sheet.normal)) / dn_e), False, cfg)
            flow_W = None if flow is None else beam_part_W(beam, flow)
            if flow_W is not None and part is not None and flow_W > part:
                taken = taken or not beam.negligible(flow_W)
        else:
            flow_W = None
        if taken:
            bound = None if sheet is None else leg_share(beam, sheet, leg, cfg, quad_rule)
            if bound is not None and flow_W is not None:
                bound = max(bound, flow_W)
            region = None if lead.region is None else lead.region()
            if region is not None:  # (a sag surface itself, not a plane standing for it: the tighter bound holds)
                region_W = beam_part_W(beam, region)
                bound = region_W if bound is None else min(bound, region_W)
                taken = met_by_ray or not beam.negligible(bound)
            if met_by_ray and bound is not None and beam.negligible(bound):
                bound = None  # (a pupil ray meets it first where the Gaussian bound says nothing does: no bound)
        lower = None
        if taken:
            found.append(lead.surface_id)
            if bound is not None and lead.certain is not None and not beam.chief_obstructed:
                least = lead.certain() * beam.power_W
                lower = min(least, bound) if least > 0.0 else None
        out.append(TakenBefore(lead.surface_id, PUPIL_RAY_FIRST if met_by_ray and bound is None else NEXT_SURFACE, part, taken,
                               bound if taken else None, lower))
    if start is not None and start.surface_id not in found:
        # a live pupil ray that does not cross the plane ahead on this leg left the start surface beyond it; without a
        # Gaussian bound, also where the start surface's support may reach beyond the plane
        gone = bool((~np.isfinite(hit_t[rows])).any())
        if leg is None and not gone:
            # (no Gaussian bound: where the surface's support may reach beyond the plane, light may leave it there -- the
            # pupil rays' spread does not bound the beam's footprint: near a waist they lie well inside it)
            gone = support_reaches(_surface_named(surfaces, start.surface_id), plane_o, plane_n, -chief_side)
        part = most = flow = None
        if leg is not None:
            tangent = start.resolved_tangent()
            shares = [start_beyond(beam, surfaces[si], leg, p, cfg) for p in (start.resolved_plane(), tangent) if p is not None]
            part = min((v for v in shares if v is not None), default=None)
            if part is not None:  # (the energy flow's bound beside the rays' share: the larger decides and bounds)
                end = None
                usable = True
                if lead is not None:  # (the next surface, before every hit of the envelope's lines: ``NextSurface.sheet``)
                    sheet = lead.resolved_sheet()
                    dn_e = None if sheet is None else float(dot(leg.d, sheet.normal))
                    if sheet is None or abs(dn_e) <= cfg.parallel_dot_tol:  # type: ignore[arg-type]
                        usable = False
                    else:
                        end = (sheet.normal, float(dot(sheet.origin - leg.start, sheet.normal)) / dn_e)  # type: ignore[operator]
                if usable:
                    flows = [start_flow_share(beam, surfaces[si], leg, p, end, cfg) for p in (start.resolved_flow(), tangent) if p is not None]
                    least = min((v for v in flows if v is not None), default=None)
                    flow = None if least is None else beam_part_W(beam, least)
                most = part if flow is None else max(part, flow)
        taken = gone or (most is not None and most > 0.0 and not beam.negligible(most))
        bound = most if taken and flow is not None and not (gone and beam.negligible(most)) else None  # type: ignore[arg-type]
        out.append(TakenBefore(start.surface_id, START_SURFACE, part, taken, bound))
    return tuple(out)


def _surface_named(surfaces: tuple[SurfaceG, ...], surface_id: str) -> SurfaceG:
    return next(surf for surf in surfaces if surf.surface_id == surface_id)


def support_reaches(surf: SurfaceG, origin: Array, normal: Array, side: float) -> bool:
    """Whether part of a physical surface may lie on the ``side`` (+1 / -1: the sign of ``(p - origin) . normal``) of a plane:
    a ball that holds its support (``_surface_balls``) reaches that side; an unbounded support always may."""
    if surf.geometry != "sphere" and not apertures.bounded(support_of(surf)):
        return True
    return any(side * float(dot(centre - origin, normal)) + radius > 0.0 for centre, radius in _surface_balls(surf))


# --------------------------------------------------------------------------------------------- observation reading
def counts_on_active_area(gstate: GState, surf: SurfaceG, h: Hits, cos_a: float, cfg: NumericalConfig,
                          quad_rule: tuple[Array, Array]) -> tuple[bool, float | None]:
    """Does this crossing (``h``: the rays' hits on the plane ``surf``; the chief ray's first; ``cos_a``: its incidence
    cosine) count as reaching the plane's active area? Without one, always. With one: the chief ray is inside it, or
    (valid Gaussian state ``gstate``) at least ``capture_min_power_fraction`` of the beam's power falls inside it. Also
    returns, for a crossing that does not count, the fraction of the beam's power on the area (``None``: not computed, no
    valid Gaussian state)."""
    ap = surf.interaction.active_aperture
    if ap is None:
        return True, None
    u0, v0 = float(h.u[0]), float(h.v[0])
    if bool(aperture_inside(ap, h.u[:1], h.v[:1], cfg.aperture_edge_tol_mm)[0]):
        return True, None
    if not gstate.valid:
        return False, None
    # beyond aperture_quad_extent_sigmas beam radii (projected onto the plane) the Gaussian power is negligible
    reach = cfg.aperture_quad_extent_sigmas * max(gstate.radius("x"), gstate.radius("y")) / max(cos_a, cfg.zero_vector_tol)
    if aperture_miss_distance(ap, u0, v0) > reach:
        return False, 0.0
    frac = transmission_fraction(gstate, ap, (u0, v0), surf.u_axis, surf.v_axis, cos_a, cfg, quad_rule)
    return frac >= cfg.capture_min_power_fraction, frac


def tail_within_acceptance(gstate: GState, surf: SurfaceG, h: Hits, aoi_deg: float, half_angle_deg: float,
                           cfg: NumericalConfig) -> bool | None:
    """Does the Gaussian tail of an uncounted crossing reach the active area inside the acceptance cone
    (``ArrivalRecord.tail_within_acceptance``)? The light at a transverse offset ``rho`` from the chief ray of a paraxial
    Gaussian beam travels, on average, along the local wavefront normal: the chief direction tilted by ``rho / R`` per
    beam axis (``1 / R = Re(1 / q)``, zero at a waist); its angular spread about that direction has a 1/e^2 half-width
    ``theta_0 w_0 / w(z) <= theta_0`` (the far-field half angle, M^2 included). Every point of the active area lies within
    ``rho_max`` of the chief hit point (in the plane; the transverse offset is no larger), so the tail travels within
    ``delta = rho_max max_axes |1 / R| + K theta_0`` of the chief direction (``K = aperture_quad_extent_sigmas``: the
    angular power beyond it is negligible) and meets the plane's normal line at ``aoi +- delta``. ``False`` when that
    whole range is outside the cone, ``True`` when inside, ``None`` otherwise or without a valid Gaussian state."""
    g = gstate
    ap = surf.interaction.active_aperture
    if not g.valid or ap is None or ap.shape == "none":
        return None
    du, dv = float(h.u[0]) - ap.offset_u, float(h.v[0]) - ap.offset_v
    rho_max = apertures.farthest(ap, du, dv)
    inverse_r = max(abs(q.real) / (q.real * q.real + q.imag * q.imag) for q in (g.qx, g.qy))
    spread = cfg.aperture_quad_extent_sigmas * max(g.far_field_half_angle("x"), g.far_field_half_angle("y"))
    delta = math.degrees(math.atan(rho_max * inverse_r) + spread)
    if aoi_deg - delta > half_angle_deg:
        return False
    if aoi_deg + delta <= half_angle_deg:
        return True
    return None


@dataclass(frozen=True)
class AnnulusOnArea:
    """Behind a central obstruction, whether the plane's active area counts the annulus whole (``Tracer._annulus_on_area``,
    decided: ``True`` or ``False``), the envelope's fraction on the area, and the bound of the annulus's power that may
    fall beside it (counted whole all the same: ``detected_min_W``)."""

    counted: bool
    area_fraction: float | None
    beside: float


@dataclass(frozen=True)
class ObservationReading:
    """What an observation plane reads of one crossing (``observation_reading``): whether it counts on the active area
    (``counted``; ``area_fraction``: the beam's power fraction on the area for one that does not, ``None`` when not
    computed), whether the chief ray is inside the acceptance cone (``accepted``; ``None``: no cone) and, for an uncounted
    crossing, whether the Gaussian tail is (``tail_accepted``), the share of the beam an absorbing plane stops beside its
    active area (``outside``, quantized to ``power_ledger_fraction_resolution``), and the power it detects
    (``detected_W``, ``None``: not an absorbing plane that counts; ``detected_min_W``: the least of it behind a central
    obstruction where part of the annulus may fall beside the area)."""

    counted: bool
    area_fraction: float | None
    accepted: bool | None
    tail_accepted: bool | None
    outside: float
    detected_W: float | None
    detected_min_W: float | None


def observation_reading(beam: BeamView, surf: SurfaceG, h: Hits, cos_i: float, aoi_deg: float, annulus: AnnulusOnArea | None,
                        cfg: NumericalConfig, quad_rule: tuple[Array, Array],
                        open_leg: Callable[[], Leg | None] | None = None, *, area_share: float | None = None) -> ObservationReading:
    """What the observation plane ``surf`` reads where the beam crosses it (``h``; ``cos_i`` / ``aoi_deg``: the chief
    ray's incidence there): on its active area (``counts_on_active_area``; behind a central obstruction the decided
    ``annulus``), inside its acceptance cone (the chief ray's angle of incidence; the Gaussian tail of an uncounted
    crossing, ``tail_within_acceptance``) and, on an absorbing plane that counts the beam, the power on its active area
    it detects (the rest is stopped beside it: ``outside``). ``open_leg``: the leg so far (``Tracer._open_leg``), called only
    for an uncounted crossing: the beam's tail on the active area is then at least what the energy flow may carry
    across the area (``beside_area_share``); the crossing stays uncounted, whatever that fraction. ``area_share``: for a
    beam without a valid Gaussian state, the share of its power on the active area of an absorbing plane that counts it,
    from its pupil samples (``Tracer._geometric_area``); without either, the whole beam is detected (a probe: an upper
    bound)."""
    inter = surf.interaction
    if annulus is not None:
        counted, area_fraction, beside = annulus.counted, annulus.area_fraction, annulus.beside
    else:
        counted, area_fraction = counts_on_active_area(beam.gstate, surf, h, cos_i, cfg, quad_rule)
        beside = 0.0
        if not counted and area_fraction is not None and open_leg is not None:
            leg = open_leg()
            if leg is not None:
                area_fraction = max(area_fraction, min(1.0, beside_area_share(surf, leg, beam_sigmas(beam), cfg, quad_rule)))
    incident_W = beam.power_W
    # the detector's acceptance cone (chief-ray AOI): a beam outside it reaches the plane but is not detected
    accepted = None if inter.acceptance_half_angle_deg is None else aoi_deg <= inter.acceptance_half_angle_deg
    tail_accepted = None
    if not counted and inter.acceptance_half_angle_deg is not None and area_fraction:
        tail_accepted = tail_within_acceptance(beam.gstate, surf, h, aoi_deg, inter.acceptance_half_angle_deg, cfg)
    outside = 0.0
    if inter.absorbing and counted and inter.active_aperture is not None and not beam.chief_obstructed:
        # only the beam's power on the active area is detected; the rest is stopped beside it (not followed)
        frac = (transmission_fraction(beam.gstate, inter.active_aperture, (float(h.u[0]), float(h.v[0])), surf.u_axis,
                                      surf.v_axis, cos_i, cfg, quad_rule) if beam.gstate.valid else area_share)
        if frac is not None:
            quantum = cfg.power_ledger_fraction_resolution
            outside = round((1.0 - min(max(frac, 0.0), 1.0)) / quantum) * quantum
    detected_W = (incident_W * (1.0 - outside) if accepted is not False else 0.0) if inter.absorbing and counted else None
    detected_min_W = detected_W * (1.0 - beside) if detected_W is not None and beside > 0.0 else None
    return ObservationReading(counted, area_fraction, accepted, tail_accepted, outside, detected_W, detected_min_W)
