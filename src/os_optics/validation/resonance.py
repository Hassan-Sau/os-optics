"""Surfaces facing each other that may return light the trace leaves out as negligible, and how much they raise its field.

Light the trace leaves out as negligible is bounded by its diffraction limit (``validation/exposures.py``): it is left out
in pieces -- each a branch the trace stops following at a known place, with a known power ``p`` -- and the field a piece
brings to a surface from one side by crossing it is at most the amplitude of the diffraction limit of ``p``; the pieces of
one source line are mutually coherent, so their fields add in amplitude (``sum sqrt(p)``). Light that goes back and forth
between two reflecting surfaces facing each other -- the two faces of a window, a lens or a cube, a window and a mirror,
the two mirrors of a linear cavity, with lenses or plates between them -- crosses those two surfaces and every surface
between them again and again, and leaves them toward the rest of the scene. This module finds such pairs and bounds the
field one piece can bring to any surface as a multiple ``f`` of the amplitude of one crossing of its power
(``ReturnGains``); a scene without a pair keeps ``f = 1``. It also counts how often the light's open paths between the
returns may bring it to a surface or across a plane (below).

The model. After each round trip between the two surfaces of a pair the light's field is at most ``g = r_a r_b`` of what
it was (``r`` the amplitude reflectances, bounded over every angle of incidence at which the pair's light can meet the
surface, below), so all its returns together bring at most ``g / (1 - g)`` times one crossing more on the two surfaces and
on every surface between them (the geometric series of its round trips; between parallel planes, which the light walks
along, fewer: below). The light a pair holds also leaves it -- through either of its surfaces, past the edge of an
aperture, reflected by a surface between them -- and the light any return sends out is at most that return's field: it
brings no more than that bound to any surface it reaches, and light left out as negligible may reach every surface. So
every surface and observation plane, from either side, gets ``f = 1 + sum`` of the pairs' bounds (each pair's returns add
in amplitude); ``g = 1`` bounds nothing anywhere. Closed paths of three or more surfaces -- a ring, a folded or coupled
cavity, a periscope's windows through its folds, a stack of three or more reflecting planes whose reflections chain, the
light one pair sends out returned by another pair -- are not pairs: ``validation/closed_returns.py`` screens the surfaces
that may reflect the light for them (``ReturnGains.closed_cycles``) and gives a stack of parallel planes an all-orders
series that counts them (``closed_series``); where a possible closed return has no such series, the open paths' count
bounds nothing (``crossings``, ``crossing_field``: inf) and no surface it enters has an upper bound. The closed paths inside
one plane-faced solid (a prism's or a cube's internal folds, total internal reflection there) are bounded by
``validation/solid_folds.py``: its certified closed walks add to the open paths' count (``ReturnGains.loops``), or the solid
is a closed return without a series. Counted neither by the pairs nor by the open paths: light trapped by total internal
reflection in a curved lens body (at or beyond the critical angle of a face it meets from its denser side); light that refraction turns back onto a lens or a refracting surface it already passed,
before it is reflected again (below); parallel planes combined with curved refracting surfaces. Light the trace or a
probe trace follows into any cavity meets the per-path interaction limit and leaves every surface it may reach without an
upper bound (``validation/exposures.py``).

Open paths. Between two returns the left-out light goes on along an open path, on which no surface reflects it twice (a
second reflection at a surface after one at another facing it is a pair's return; after two or more, outside the model),
and between two reflections it meets each surface that turns it -- a thin lens, a refracting surface between media of
known, different index -- at most once (``ReturnGains.once``; light turned back onto one it already passed before it is
reflected again is outside the model, above). Between two reflections it travels in straight legs, a new one only where
a lens or a refracting surface turns it, so in at most ``L = T + 1`` legs (``ReturnGains.legs``; ``T`` the thin lenses
and the refracting surfaces between media of different index, those of unknown index counted), and a straight leg meets
any other surface at most ``K`` times, the most points a straight line meets it at (``ReturnGains.per_leg``, below): a
plane once (a mirror, a splitter, an attenuator, a refracting plane between media of one index, an observation plane), a
sphere or a conic twice. So between two reflections one piece's light meets a lens or a refracting surface between
different media at most once and any other surface at most ``K L`` times (``m``, ``ReturnGains.meetings``) -- also where
refraction or a lens turns it back across a plane or onto a mirror or a splitter, plane or curved, that it already met.
It is reflected at most once by each of the ``N`` surfaces that may reflect it (``ReturnGains.reflectors``), so it meets
a surface after at most ``N + 1`` numbers ``j`` of reflections; and each reflection takes at most its surface's
reflectance bound ``R_s`` (over every angle, from either side) of the light that meets the surface, which the light
after ``j`` reflections meets at most ``m_s`` times: the light after ``j + 1`` reflections carries at most ``R = sum m_s
R_s`` (``ReturnGains.reflectance``) times the power of the light after ``j``, and never more than it, so at most
``min(1, R^j)`` of the piece's power. Its crossings bring at most ``m sum_{j=0}^{N} min(1, R^j)`` times its power across
a plane or onto a surface (``ReturnGains.crossings``). Its field: the light after ``j`` reflections is made of pieces
from different sequences of ``j`` distinct reflecting surfaces, each reflection at one of the surface's ``m_s``
meetings, mutually coherent, so their fields add. Each reflection takes at most ``sqrt(R_s)`` of the field, so they sum
to at most the sum over those ordered sequences of the products of ``m_s sqrt(R_s)`` -- ``j! e_j`` of those weights,
``e_j`` the elementary symmetric polynomial (at most ``(sum m_s sqrt(R_s))^j``) --; and they are at most ``j! e_j(m_s)``
pieces sharing at most ``min(1, j! e_j(m_s R_s))`` of the power (at most ``min(1, R^j)``), so they sum to at most the
square root of the product (Cauchy-Schwarz). With ``a_j`` the smaller, its crossings bring, each time split between the
two sides, at most ``sqrt(2) m sum_{j=0}^{N} a_j`` times the amplitude of one crossing of its whole power
(``ReturnGains.crossing_field``) -- never less than ``LEAST_CROSSINGS`` = 2, the piece once from either side: 2.26 on the
faces of four lenses with 0.25 % coatings and about 20.3 on an observation plane after them (``L = 9``), 5.41 on the
faces of an uncoated window (a bare face counts 1: the reflectance just below its critical angle).

Meetings per leg (``_line_meetings``). A straight line that does not run along a surface meets a plane at most once and
a sphere at most twice. Every point of a sag surface (``optics/surfaces/sag.py``: ``z = c s / (1 + S) + P(s) + L
sqrt(s)``, ``s = rho^2``, ``P`` the even-asphere polynomial, of degree ``p`` in ``s``, ``L`` the axicon term) lies on
the conic's quadric, ``c s + c (1 + k) w^2 - 2 w = 0`` with ``w = z - P(s) - L sqrt(s)``. A line whose projection across
the axis stands still (one parallel to the axis; on a cylinder one whose coordinate along the curved axis is fixed)
meets the surface at most once: ``s`` and so the sag stay fixed along it. Along any other, ``s = b^2 + t^2`` (``t`` the
distance along that projection, ``b`` its distance from the axis; ``b = 0`` on a cylinder) and ``z`` is linear in ``t``,
so ``V = z - P(s)`` is a polynomial in ``t`` of degree ``e = 2p`` (1 without asphere terms) and the quadric reads ``A +
sqrt(s) B = 0`` with the polynomials ``A = c s + c (1 + k) (V^2 + L^2 s) - 2 V`` and ``B = 2 L (1 - c (1 + k) V)``.
Without an axicon term every point the line meets is a root of ``A``, of degree at most ``2e`` where ``c (1 + k) != 0``,
``max(e, 2)`` on a paraboloid and ``e`` on a flat base (1: a plane). With one it is a root of ``A^2 - s B^2`` (which
vanishes identically only with ``A`` and ``B``: ``b^2 + t^2`` is no square) or, through the axis, of ``A + t B`` on one
side of it and of ``A - t B`` on the other: at most twice as many. Where the polynomial vanishes identically the line
follows the quadric, so it runs along the surface or meets it only where the two roots of the quadric meet, at the edge
of the profile's domain (``S = 0``: at most twice). So ``K`` is 2 on a sphere or a conic, twice the highest power of
``rho`` in the sag of an even asphere (that power itself on a paraboloid or a flat base), and twice that with an axicon
term (2 on a bare cone, 4 on a curved base).

Legs. Between two surfaces light travels straight: from a point of one surface's aperture to a point of the next one's,
every direction of such a leg lies within the angle the two bounding spheres subtend of the line between their centres
(any direction where they overlap), and it arrives at the far surface from the side the near one lies on -- from either
side where their extents across it overlap (light after a lens may meet either side of a surface the lens straddles). An
interface takes only light in its own medium on each side (the trace ends any other there): a leg arrives at an
interface only from a side whose index is that of the medium the leg starts in, within ``INDEX_TOL`` (air and vacuum
hold the same light); otherwise its light crossed another interface on the way, whose own leg brings it. A thin lens
passes the light within its largest deflection, ``2 atan(rho / 2|f|)`` (the slope-form lens turns a ray met at height
``h`` above its axis by up to ``2 atan(h / 2f)``; ``rho`` the largest height of its clear aperture above its axis, a
decentred aperture's offset included); an interface refracts a cone into one about its refracted axis, its half angle times
the largest angular magnification of Snell's law over the cone's angles of incidence (``_snell``; a curved one also
turned by its normals' cone, ``_snell_curved``); a cone that reaches grazing incidence or the critical angle there goes on
in every direction toward that side.

The pairs. Two reflecting surfaces (interfaces, mirrors, splitters, attenuators) face each other when light one of them
reflects can reach the other and come back: each sends light arriving from the other back toward it (a reflection's
image of the arriving cone, widened by twice the cone of its normals, meets the leg back), directly or through a thin
lens or an interface between them, which may turn the light toward the other one. The surfaces between them: those that
meet the convex hull of the two (``_Faces.crossing``), and every other thin lens or refracting interface that two of the
pair's surfaces and the surfaces between them already found lie on opposite sides of, or straddle (light may go from one
to the other through it, turned off the line between them: ``_Scene.relays``); two parallel planes with such a surface
off their hull that is not a plane parallel to them (a lens, a tilted face of a plate or a wedge: it turns the light and
shifts it sideways) are screened through them where a chain of the surfaces around them can pass light from one plane
to the other (each turning it within its largest deflection: a thin lens's, or ``acos(n_min / n_max)`` at a refracting
surface, ``_Scene.passes``), else as parallel planes; a refracting plane parallel to them keeps the light's momentum
along them wherever it lies, its medium counting in the walk (below). Light of the pair arrives at one of its
surfaces along the leg from the other one, or within the cone a surface between them passes the light from the other
one (or from another such surface) on in and the leg from that surface. Between two parallel planes with nothing between
them that turns light but planes parallel to them, every reflection and refraction keeps the light's momentum along the
planes, ``k = n sin(angle)``: the momenta that reach a plane of a class of parallel planes are those the rest of the
scene brings in -- along the leg from every other surface that redirects light (from a plane, within the mirror images
and refractions of the legs into it, ``images``) or within a source's beam (``RETURN_ANGLE_SIGMAS`` far-field divergences
of its direction) -- passed on from plane to plane of the class as far as the light's walk along them lets it reach the
other plane's aperture (``momenta``). The light of a pair of parallel planes has a momentum both planes meet; between them
it walks ``s(k) = sum t_m k / sqrt(n_m^2 - k^2)`` along them per pass, its hits on each plane ``2 s`` apart: on apertures
of radii ``rho_1 >= rho_2`` it comes back at most ``M = min(rho_1 / s, rho_2 / s + 1)`` times (each return onto one plane
needs a hit on the other between), so its returns bring at most ``min(M g, g / (1 - g))`` (``_parallel_returns``).
Elsewhere the angles are every angle of incidence of the arriving cones on the cone of the surface's normals.

Reflectance bounds (both polarizations). A bare interface: the s amplitude reflectance at the widest angle (it grows with
the angle and is never below the p one); from its denser side only angles below the critical angle count (beyond, the
light is totally reflected and held, outside the model), so a range reaching the critical angle counts 1, the
reflectance just below it, and a range wholly beyond it 0 -- the faces of an uncoated lens, which light inside it can
meet up to the critical angle, bound nothing. A coating: its largest reflectance over the range, at the range's ends and
its table's angles between them; a range not wholly inside its data (angles of incidence and wavelength) counts as
reflecting everything. A coating whose record states an air-equivalent angle range, met from a medium denser than air, is read at
the air-equivalent angles of the left-out light's own bounded extent there, not over the range above. Where that extent leaves
the record's range or has no real air-equivalent angle the use is refused (``validation/coating_use.py``), which has two readings:
the factors and counts here keep the record's stated value, and a plane's power bound reads them; the exposure upper bound of the
refused surface, and of the surfaces of a pair the refusal leaves with a gain bound of 1 (each refused surface counting as reflecting
everything), reads no factor (``ReturnGains.exposure_factor``). A mirror's back side stops the light; an uncoated mirror or splitter reflects everything; an
attenuator its stated reflectance; an interface whose media's indices are not known everything.

Arithmetic: ``+ - x /`` and ``sqrt`` on Python floats and elementwise NumPy, and the libm-free trigonometry of
``math.exact_trig``: the factors have identical bits in every runtime.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field

import numpy as np

from ..errors import EngineInputError
from ..math.exact_trig import atan2, cos_sin
from ..optics import apertures
from ..optics.materials.library import MaterialLibrary
from ..optics.surfaces.freeform import line_meetings
from ..optics.surfaces.sag import local_normals
from ..optics.surfaces.tabulated_lens import response_at
from ..scene.model import ApertureG, Scene, SurfaceG
from ..units import NM_TO_MM
from ..run_control import phase
from .return_work import (BUDGET_EXHAUSTED, COMPLETED_NO_RETURN, INCOMPLETE, NOT_STARTED, PAIR_SCREEN_WORK, PAIR_STAGE,
                          PARALLEL_STACK, POTENTIAL_RETURN, RETURN_FOUND, SEARCH_COMPLETED, SOLID_FOLD_STAGE, ClosedReturnSearch,
                          OutOfReturnWork, ReturnWork, WorkCounter, momentum_work, pair_cone_work)

#: A source's beam travels within this many far-field divergences of its direction and starts within this many radii of
#: its axis (beyond, a Gaussian beam carries less than exp(-2 x 8^2) of its power: below the rounding of any bound it
#: enters).
RETURN_ANGLE_SIGMAS = 8.0
#: Radii x angles at which a sag surface's normals are sampled for the cone of its normals, and the factor the sine of the
#: widest sampled angle is widened by (between samples).
SAG_SAMPLES = (24, 24)
SAG_CONE_MARGIN = 1.1
#: The reflecting interactions.
REFLECTING = frozenset({"refract", "mirror", "beam_splitter", "neutral_density", "diffraction_grating"})
#: The interactions that return no light (amplitude reflectance and transmittance bounds 0 in ``_coefficients``). Every
#: lowered interaction kind is in exactly one of ``REFLECTING`` and ``NON_RETURNING``; a kind in neither is refused
#: (``KeyError``) rather than taken to return no light (``tests/test_element_kind_coverage.py``).
NON_RETURNING = frozenset({"thin_lens", "polarizer", "retarder", "faraday_rotator", "absorber", "aperture_stop", "observe",
                           "complex_transmission"})
#: Interactions other than refraction that turn the light they pass.
BENDING = frozenset({"thin_lens"})
#: Interactions after which light may leave in another direction than it came (the start of a leg).
REDIRECTING = REFLECTING | BENDING
#: Two planes are parallel when the cosine of the angle between their normals is within this of 1.
PARALLEL_TOL = 1e-12
#: The length resolution of a side or overlap test on surface extents (mm): ``SIDE_ABS_TOL_MM + SIDE_REL_TOL`` times the
#: largest finite coordinate the test compares. A body lies on one side of a surface only when its extent clears the
#: surface's by more than this, and a surface meets a hull when the two extents come within it; inside it the answer is the
#: conservative one (``EITHER``, met). Poses enter as ``q / |q|``, which for a quaternion stated off unit length within its
#: tolerance differs from the unit quaternion in the last bit, so coplanar faces (a window's, a cube splitter's) are a few
#: 1e-14 mm apart or exactly level by rounding; an exact comparison would let that bit decide the screen's work and
#: outcome.
SIDE_ABS_TOL_MM = 1e-9
SIDE_REL_TOL = 1e-12
#: The cosine resolution of an overlap test on two cones of directions (``_overlap``, ``_any_meets``,
#: ``_Scene.light_into``): two cones share a direction when the cosine of the angle between their axes comes within this of
#: the cosine of the sum of their half angles; inside it the answer is the conservative one (they overlap). Cones that
#: touch (a leg's cone tangent to a reflection's image) meet with a margin of a few 1e-16 of either sign, and poses entering
#: as ``q / |q|`` move that margin by its last bits; an exact comparison would let that bit decide the screen's work and
#: outcome.
OVERLAP_COS_RESOLUTION = 1e-12
#: Two media whose indices differ by at most this may hold the same light (air and vacuum do); between media further
#: apart light crosses an interface.
INDEX_TOL = 1e-3
#: Sides a surface is met from: from its positive side (against its normal), from its negative side, or either.
POSITIVE, NEGATIVE, EITHER = 0, 1, 2

#: The field of light split between a plane's or a surface's two sides: at most sqrt(2) times that of all of it on one
#: (``sqrt(a) + sqrt(b) <= sqrt(2 (a + b))``).
SQRT_TWO = math.sqrt(2.0)
#: The least the open paths' crossings count, for the power and the field: the piece once from either side.
LEAST_CROSSINGS = 2.0
#: The relative allowance for rounding the sums of ``ReturnGains._looped`` carry (far above the rounding of their sums).
LOOP_ROUNDING = 1e-12

_PI = 3.141592653589793
_HALF_PI = 1.5707963267948966
_DEGREES = 57.29577951308232  # (180 / pi)
_GRAZING_DEG = 90.0


def _dot(a, b) -> float:
    return float(a[0]) * float(b[0]) + float(a[1]) * float(b[1]) + float(a[2]) * float(b[2])


def _unit(v) -> tuple[float, float, float]:
    length = math.sqrt(_dot(v, v))
    return (float(v[0]) / length, float(v[1]) / length, float(v[2]) / length)


def _angle(a, b) -> float:
    """The angle (rad, in [0, pi]) between two unit vectors (libm-free)."""
    cx = float(a[1]) * float(b[2]) - float(a[2]) * float(b[1])
    cy = float(a[2]) * float(b[0]) - float(a[0]) * float(b[2])
    cz = float(a[0]) * float(b[1]) - float(a[1]) * float(b[0])
    return float(atan2(math.sqrt(cx * cx + cy * cy + cz * cz), _dot(a, b)))


#: At most this many angles' (cos, sin) are remembered (a scene's pairs meet the same few angles again and again).
COS_SIN_REMEMBERED = 65536
_COS_SIN: dict[float, tuple[float, float]] = {}


def _cos_sin(angle: float) -> tuple[float, float]:
    """(cos, sin) of an angle (libm-free; remembered: the same angles recur for every pair)."""
    found = _COS_SIN.get(angle)
    if found is None:
        c, s = cos_sin(np.array([angle]))
        found = (float(c[0]), float(s[0]))
        if len(_COS_SIN) < COS_SIN_REMEMBERED:
            _COS_SIN[angle] = found
    return found


def _asin(x: float) -> float:
    """The angle (rad, in [0, pi/2]) whose sine is ``x`` (clipped to [0, 1]; libm-free: ``atan2(x, sqrt(1 - x^2))``)."""
    x = min(max(x, 0.0), 1.0)
    return float(atan2(x, math.sqrt((1.0 - x) * (1.0 + x))))


def _reflected(d, n):
    c = _dot(d, n)
    return _unit((d[0] - 2.0 * c * n[0], d[1] - 2.0 * c * n[1], d[2] - 2.0 * c * n[2]))


def _refracted(d, n, n_from: float, n_to: float):
    """The direction refracted at a plane of normal ``n`` (``None``: total internal reflection)."""
    c = _dot(d, n)
    ratio = n_from / n_to
    tangential = (d[0] - c * n[0], d[1] - c * n[1], d[2] - c * n[2])
    sin2 = ratio * ratio * _dot(tangential, tangential)
    if sin2 >= 1.0:
        return None
    along = math.sqrt(1.0 - sin2) * (1.0 if c > 0.0 else -1.0)
    return _unit((ratio * tangential[0] + along * n[0], ratio * tangential[1] + along * n[1], ratio * tangential[2] + along * n[2]))


# ------------------------------------------------------------------------------------------------ surfaces as seen here


@dataclass(frozen=True)
class _Face:
    """A surface as the screen sees it: index, centre of its aperture, bounding radius (inf: unbounded), the axis of its
    normals, the half angle of their cone (rad; 0 for a plane), whether it is a plane, its extent along its axis from the
    centre (``low``, ``high``, mm: a curved surface's sag) and, for a plane with a rectangular aperture, its in-plane axes
    and half widths (``rectangle``: u, v, half width along u, along v)."""

    index: int
    centre: tuple[float, float, float]
    radius: float
    axis: tuple[float, float, float]
    cone: float
    plane: bool
    low: float = 0.0
    high: float = 0.0
    rectangle: tuple[tuple[float, float, float], tuple[float, float, float], float, float] | None = None

    def side(self, axis) -> float:
        """Half the extent of the aperture along ``axis`` beyond its centre's (inf: unbounded): a rectangle's exactly, else
        the bounding radius times the sine of the tilt."""
        if self.rectangle is not None:
            u, v, hu, hv = self.rectangle
            return hu * abs(_dot(u, axis)) + hv * abs(_dot(v, axis))
        a = _dot(self.axis, axis)
        tilt = math.sqrt(max(0.0, (1.0 - a) * (1.0 + a)))
        return self.radius * tilt if tilt > 0.0 else 0.0


def _aperture_radius(ap: ApertureG | None) -> float:
    """The radius of a circle about the surface's origin that holds its aperture (inf: no aperture)."""
    if ap is None or ap.shape == "none":
        return math.inf
    offset = math.sqrt(ap.offset_u * ap.offset_u + ap.offset_v * ap.offset_v)
    return apertures.reach(ap) + offset


def _face(index: int, s: SurfaceG) -> _Face:
    if s.interaction.kind == "diffraction_grating":
        # (its orders leave in directions the law of reflection does not give: for these bounds, a flat face with normals in
        # every direction, which may send light it meets anywhere)
        return _Face(index, tuple(float(x) for x in s.origin), _aperture_radius(s.aperture), _unit(s.normal), _HALF_PI, False, 0.0, 0.0)  # type: ignore[arg-type]
    if s.geometry == "sphere":
        rim = math.sqrt(max(s.radius * s.radius - s.aperture_radius * s.aperture_radius, 0.0))
        centre = s.center + rim * s.cap_axis
        sine = min(s.aperture_radius / s.radius, 1.0) if s.radius > 0.0 else 1.0
        cone = _HALF_PI if sine >= 1.0 else float(atan2(sine, math.sqrt((1.0 - sine) * (1.0 + sine))))
        return _Face(index, tuple(float(x) for x in centre), s.aperture_radius + (s.radius - rim), _unit(s.cap_axis),  # type: ignore[arg-type]
                     cone, False, 0.0, s.radius - rim)
    ap = s.interaction.active_aperture if s.interaction.kind == "observe" else s.aperture
    radius = _aperture_radius(ap)
    if s.geometry == "plane":
        if ap is None or ap.shape == "none":
            return _Face(index, tuple(float(x) for x in s.origin), radius, _unit(s.normal), 0.0, True)  # type: ignore[arg-type]
        # (the aperture's own centre; a rectangle kept as one)
        centre = s.origin + ap.offset_u * s.u_axis + ap.offset_v * s.v_axis
        u, v = tuple(float(x) for x in s.u_axis), tuple(float(x) for x in s.v_axis)
        if ap.shape != "rectangle":  # (a circle, or the disc about its centre that holds any other region)
            return _Face(index, tuple(float(x) for x in centre), float(apertures.reach(ap)), _unit(s.normal), 0.0, True)  # type: ignore[arg-type]
        return _Face(index, tuple(float(x) for x in centre), math.sqrt(ap.p1 * ap.p1 + ap.p2 * ap.p2), _unit(s.normal), 0.0,  # type: ignore[arg-type]
                     True, rectangle=(u, v, float(ap.p1), float(ap.p2)))  # type: ignore[arg-type]
    # a sag surface: the cone of its normals over the clear aperture, from samples (widened between them); its sag at most
    # the radius times the tangent of that angle
    if not math.isfinite(radius):
        return _Face(index, tuple(float(x) for x in s.origin), radius, _unit(s.normal), _PI, False, -math.inf, math.inf)  # type: ignore[arg-type]
    radii, angles = SAG_SAMPLES
    rho = np.repeat(np.linspace(0.0, radius, radii), angles)
    phi = np.tile(np.arange(angles, dtype=np.float64) * (2.0 * math.pi / angles), radii)
    cos_phi, sin_phi = cos_sin(phi)
    normals = local_normals(s, rho * cos_phi, rho * sin_phi)
    axis = _unit(s.normal)
    least = 1.0
    for row in normals:
        length2 = _dot(row, row)
        if not (math.isfinite(length2) and length2 > 0.0):
            least = -1.0
            break
        least = min(least, abs(_dot(row, axis)) / math.sqrt(length2))
    if least < 0.0:
        return _Face(index, tuple(float(x) for x in s.origin), radius, axis, _PI, False, -math.inf, math.inf)  # type: ignore[arg-type]
    sine = SAG_CONE_MARGIN * math.sqrt(max(0.0, (1.0 - least) * (1.0 + least)))
    if sine >= 1.0:
        return _Face(index, tuple(float(x) for x in s.origin), radius, axis, _HALF_PI, False, -math.inf, math.inf)  # type: ignore[arg-type]
    cosine = math.sqrt((1.0 - sine) * (1.0 + sine))
    depth = radius * sine / cosine + (abs(s.freeform.vertex_sag) if s.freeform is not None else 0.0)
    return _Face(index, tuple(float(x) for x in s.origin), radius, axis, float(atan2(sine, cosine)), False, -depth, depth)  # type: ignore[arg-type]


def _point(centre, radius: float, axis) -> _Face:
    """A source as the screen sees it: a ball of ``radius`` about its position (its beam's extent there)."""
    return _Face(-1, tuple(float(x) for x in centre), radius, axis, _PI, False, -radius, radius)  # type: ignore[arg-type]


def _extent(face: _Face, axis) -> tuple[float, float]:
    """The extent of a surface along ``axis`` (infinite when not bounded): its sag along its own axis, projected, and its
    aperture's radius times the sine of its tilt."""
    a = _dot(face.axis, axis)
    side = face.side(axis)
    if not (math.isfinite(face.low) and math.isfinite(face.high) and math.isfinite(side)):
        return -math.inf, math.inf
    base = _dot(face.centre, axis)
    ends = (base + face.low * a, base + face.high * a)
    return min(ends) - side, max(ends) + side


def _side_resolution(*values: float) -> float:
    """``SIDE_ABS_TOL_MM + SIDE_REL_TOL`` times the largest finite magnitude of ``values`` (mm)."""
    largest = max((abs(v) for v in values if math.isfinite(v)), default=0.0)
    return SIDE_ABS_TOL_MM + SIDE_REL_TOL * largest


def _side_from(face: _Face, low: float, high: float) -> int:
    """The side of ``face`` a body whose extent along the face's axis is [``low``, ``high``] lies on (``EITHER`` when the
    two extents overlap or come within the side resolution of each other, ``_side_resolution``)."""
    f_low, f_high = _extent(face, face.axis)
    eps = _side_resolution(low, high, f_low, f_high)
    if low > f_high + eps:
        return POSITIVE
    if high < f_low - eps:
        return NEGATIVE
    return EITHER


class _Faces:
    """Every surface's centre, bounding radius, axis and extent along its axis, as arrays (``crossing``)."""

    def __init__(self, faces: list[_Face]) -> None:
        self.centre = np.array([f.centre for f in faces], dtype=np.float64).reshape(-1, 3)
        self.radius = np.array([f.radius for f in faces], dtype=np.float64)
        self.axis = np.array([f.axis for f in faces], dtype=np.float64).reshape(-1, 3)
        base = self.centre[:, 0] * self.axis[:, 0] + self.centre[:, 1] * self.axis[:, 1] + self.centre[:, 2] * self.axis[:, 2]
        self.low = base + np.array([f.low for f in faces], dtype=np.float64)
        self.high = base + np.array([f.high for f in faces], dtype=np.float64)

    def _extent(self, f: _Face) -> tuple[np.ndarray, np.ndarray]:
        """``_extent`` of ``f`` along every surface's axis."""
        ax, ay, az = self.axis[:, 0], self.axis[:, 1], self.axis[:, 2]
        a = f.axis[0] * ax + f.axis[1] * ay + f.axis[2] * az
        tilt = np.sqrt(np.maximum(0.0, (1.0 - a) * (1.0 + a)))
        base = f.centre[0] * ax + f.centre[1] * ay + f.centre[2] * az
        if not (math.isfinite(f.low) and math.isfinite(f.high)):
            return np.full(a.shape, -math.inf), np.full(a.shape, math.inf)
        with np.errstate(invalid="ignore"):
            side = np.where(tilt > 0.0, f.radius * tilt, 0.0)
        if f.rectangle is not None:
            u, v, hu, hv = f.rectangle
            side = (hu * np.abs(u[0] * ax + u[1] * ay + u[2] * az) + hv * np.abs(v[0] * ax + v[1] * ay + v[2] * az))
        ends_low, ends_high = base + f.low * a, base + f.high * a
        low = np.minimum(ends_low, ends_high) - side
        high = np.maximum(ends_low, ends_high) + side
        return np.where(np.isnan(low), -math.inf, low), np.where(np.isnan(high), math.inf, high)

    def crossing(self, a: _Face, b: _Face) -> np.ndarray:
        """Per surface: whether it meets the convex hull of two surfaces (where light going from one of them to the other
        travels): its aperture comes within the larger of their bounding radii of the line between their centres, and its
        extent along its own axis overlaps the hull's."""
        width = max(a.radius, b.radius)
        c = self.centre
        ab = (b.centre[0] - a.centre[0], b.centre[1] - a.centre[1], b.centre[2] - a.centre[2])
        length2 = _dot(ab, ab)
        apx, apy, apz = c[:, 0] - a.centre[0], c[:, 1] - a.centre[1], c[:, 2] - a.centre[2]
        if length2 > 0.0:
            t = np.clip((apx * ab[0] + apy * ab[1] + apz * ab[2]) / length2, 0.0, 1.0)
        else:
            t = np.zeros(c.shape[0])
        qx, qy, qz = a.centre[0] + t * ab[0] - c[:, 0], a.centre[1] + t * ab[1] - c[:, 1], a.centre[2] + t * ab[2] - c[:, 2]
        near = np.ones(c.shape[0], dtype=bool)
        if math.isfinite(width):
            with np.errstate(invalid="ignore"):
                far = np.isfinite(self.radius) & (np.sqrt(qx * qx + qy * qy + qz * qz) > width + self.radius)
            near = ~far
        low_a, high_a = self._extent(a)
        low_b, high_b = self._extent(b)
        low, high = np.minimum(low_a, low_b), np.maximum(high_a, high_b)
        # (within the side resolution of each other the extents meet: _side_resolution over the four ends compared)
        eps = SIDE_ABS_TOL_MM + SIDE_REL_TOL * np.maximum(np.maximum(_finite_abs(self.low), _finite_abs(self.high)),
                                                          np.maximum(_finite_abs(low), _finite_abs(high)))
        return near & (self.high >= low - eps) & (self.low <= high + eps)


def _finite_abs(x: np.ndarray) -> np.ndarray:
    """``|x|``, 0 where ``x`` is not finite."""
    return np.where(np.isfinite(x), np.abs(x), 0.0)


# ------------------------------------------------------------------------------------------------ media, beams, lenses


def _index(library: MaterialLibrary, medium: str, wavelength_nm: float) -> float | None:
    try:
        n, issue = library.evaluator(medium).index(wavelength_nm)
    except EngineInputError:
        return None
    return None if issue is not None else n


def _index_range(library: MaterialLibrary, medium: str, wavelength_nm: float) -> tuple[float, float] | None:
    """The least and the largest index light in ``medium`` may meet at this wavelength: ``(n, n)`` for an isotropic medium; a
    uniaxial crystal's ordinary and extraordinary indices (an extraordinary wave's ``n(theta)`` lies between them, and
    ``_index`` reads the ordinary one); ``None``: not known."""
    laws = library.uniaxial(medium)
    if laws is None:
        n = _index(library, medium, wavelength_nm)
        return None if n is None else (n, n)
    n_o, issue_o = laws.ordinary_index.index(wavelength_nm)
    n_e, issue_e = laws.extraordinary_index.index(wavelength_nm)
    if n_o is None or n_e is None or issue_o is not None or issue_e is not None:
        return None
    return (n_o, n_e) if n_o <= n_e else (n_e, n_o)


def _divergence(scene: Scene, library: MaterialLibrary, wavelength_nm: float) -> float:
    """The largest far-field half angle (rad) of the scene's sources at this wavelength (``M^2 lambda / (pi n w0)``)."""
    widest = 0.0
    for src in scene.sources:
        n = _index(library, src.medium, wavelength_nm) or 1.0
        for w0, m2 in ((src.w0x, src.m2x), (src.w0y, src.m2y)):
            widest = max(widest, m2 * wavelength_nm * NM_TO_MM / (math.pi * n * w0))
    return widest


def _source_radius(src, library: MaterialLibrary, wavelength_nm: float) -> float:
    """The larger 1/e^2 radius of a source's beam where it starts, at this wavelength: ``w0 sqrt(1 + (z / z_R)^2)`` with
    ``z`` the distance to its waist (which may lie ahead of the source or behind it) and ``z_R = pi n w0^2 / (M^2 lambda)``."""
    n = _index(library, src.medium, wavelength_nm) or 1.0
    lam = wavelength_nm * NM_TO_MM
    widest = 0.0
    for w0, m2 in ((src.w0x, src.m2x), (src.w0y, src.m2y)):
        z = src.z_waist / (math.pi * n * w0 * w0 / (m2 * lam))
        widest = max(widest, w0 * math.sqrt(1.0 + z * z))
    return widest


def _lens_bend(lens: SurfaceG, wavelength_nm: float, radius: float) -> float:
    """The largest deflection (rad) of a thin lens over its aperture (pi: not bounded), ``radius`` the largest height of
    its aperture above its axis (``_aperture_radius``: about the surface's origin, where the lens's axis passes): the
    slope-form lens (``s' = s - h / f``) turns a ray met at height ``h`` by ``atan(s) - atan(s - h / f)``, which is largest,
    ``2 atan(h / 2|f|)``, for the slope ``s = h / 2f``."""
    focal = lens.interaction.focal_length
    if lens.interaction.table is not None:
        focal = response_at(lens.interaction.table, wavelength_nm)[0]
    if not (math.isfinite(radius) and math.isfinite(focal) and focal != 0.0):
        return _PI
    return 2.0 * float(atan2(radius, 2.0 * abs(focal)))


# ------------------------------------------------------------------------------------------------ reflectance bounds


def _fresnel(n1: float, n2: float, cos_i: float) -> float:
    """The larger of the s and p amplitude reflectances of a bare interface from ``n1`` into ``n2`` (1 at and above the
    critical angle)."""
    cos_i = min(max(cos_i, 0.0), 1.0)
    ratio = n1 / n2
    sin2_t = ratio * ratio * (1.0 - cos_i) * (1.0 + cos_i)
    if sin2_t >= 1.0:
        return 1.0
    cos_t = math.sqrt(1.0 - sin2_t)
    rs = abs((n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t))
    rp = abs((n2 * cos_i - n1 * cos_t) / (n2 * cos_i + n1 * cos_t))
    return rs if rs >= rp else rp


_CRITICAL: dict[tuple[float, float], float] = {}


def _critical(n1: float, n2: float) -> float | None:
    """The critical angle (rad) of light going from ``n1`` into ``n2`` (``None``: there is none, ``n1 <= n2``;
    remembered: a scene has few pairs of indices)."""
    if n1 <= n2:
        return None
    found = _CRITICAL.get((n1, n2))
    if found is None:
        found = _asin(n2 / n1)
        if len(_CRITICAL) < COS_SIN_REMEMBERED:
            _CRITICAL[(n1, n2)] = found
    return found


#: Why a surface's reflectance bound over a range of angles is 1 (``ReturnPair.saturation``): the range reaches the
#: critical angle from the denser side; it reaches grazing incidence; it is not wholly inside the coating's data; the
#: coating's evaluation reported an issue; an index is not known; the surface's record states a reflectance of 1 (a
#: mirror or a splitter without a coating, a grating, a coating value of 1); a coating use the left-out light makes was
#: refused. ``TOTAL_REFLECTION``: not a cause of 1 -- every angle of the range is at or beyond the critical angle, which
#: the pair model leaves out (the bound there is 0).
CRITICAL_ANGLE_IN_RANGE = "CRITICAL_ANGLE_IN_RANGE"
GRAZING_INCIDENCE_IN_RANGE = "GRAZING_INCIDENCE_IN_RANGE"
COATING_DATA_RANGE = "COATING_DATA_RANGE"
COATING_EVALUATION_ISSUE = "COATING_EVALUATION_ISSUE"
INDEX_UNKNOWN = "INDEX_UNKNOWN"
REFLECTS_EVERYTHING = "REFLECTS_EVERYTHING"
REFUSED_COATING_USE = "REFUSED_COATING_USE"
TOTAL_REFLECTION = "TOTAL_REFLECTION"


def _interface(n1: float, n2: float, low: float, high: float, why: list[str] | None = None) -> float:
    """The amplitude reflectance bound of a bare interface from ``n1`` into ``n2`` over the angles of incidence [``low``,
    ``high``] (rad) below the critical angle (module docstring). ``why``: receives the code of a bound of 1 or 0 that is
    not the Fresnel value (``CRITICAL_ANGLE_IN_RANGE``, ``TOTAL_REFLECTION``); it changes no value."""
    critical = _critical(n1, n2)
    if critical is not None:
        if low >= critical:
            if why is not None:
                why.append(TOTAL_REFLECTION)
            return 0.0  # (totally reflected: outside the model)
        if high >= critical:
            if why is not None:
                why.append(CRITICAL_ANGLE_IN_RANGE)
            return 1.0  # (the reflectance just below the critical angle)
    return _fresnel(n1, n2, _cos_sin(high)[0])


def _interface_range(n1: tuple[float, float], n2: tuple[float, float], low: float, high: float,
                     why: list[str] | None = None) -> float:
    """``_interface`` over every pair of indices ``n1`` in ``[n1_low, n1_high]`` and ``n2`` in ``[n2_low, n2_high]`` (a crystal's
    side: its ordinary and extraordinary indices bound the index its waves meet). The larger amplitude reflectance is the s one,
    which grows with the angle of incidence and as the ratio ``m = n2 / n1`` moves away from 1 either way: its largest value lies
    at the ends of ``m``'s range, unless some ``m`` of the range puts a critical angle ``asin(m)`` within the angles (then 1)."""
    if n1[0] == n1[1] and n2[0] == n2[1]:
        return _interface(n1[0], n2[0], low, high, why)
    m_low, m_high = n2[0] / n1[1], n2[1] / n1[0]
    if m_low < 1.0 and m_low <= _cos_sin(high)[1] and m_high > _cos_sin(low)[1]:
        if why is not None:
            why.append(CRITICAL_ANGLE_IN_RANGE)
        return 1.0
    return max(_interface(n1[1], n2[0], low, high, why), _interface(n1[0], n2[1], low, high, why))


def _air_index(library: MaterialLibrary, wavelength_nm: float) -> float:
    """The index of the request's ``air`` (the environment's air in a native run; 1 in the Engine's own convention): an
    air-equivalent angle is the angle in that air with the same ``n sin(theta)``."""
    n = _index(library, "air", wavelength_nm)
    return 1.0 if n is None else n


def _dense_coating(evaluator, n: float | None, low: float, high: float, use, wavelength_nm: float,
                   n_low: float | None = None, why: list[str] | None = None, *, air: float = 1.0) -> tuple[float, float] | None:
    """(largest reflectance, largest transmittance) of a coating whose record states an air-equivalent angle range, met from a
    medium denser than air (``validation/coating_use.py``; ``None``: any other use, read as its angles say). A use the left-out
    light makes of it: at the air-equivalent angles of its extent, (1, 1) where it is refused (``use`` is given for a refused
    use only in the refusal's reading, ``_coefficients``); a side no left-out light reaches on its way to its next reflection,
    or a refused use in the stated reading: at the air-equivalent angles of the range the geometry gives (its top at most the
    critical angle: 90 deg). ``n_low``: a crystal's least index (``n`` its largest; the air-equivalent angles run from the least
    index's at ``low`` to the largest's at ``high``)."""
    declared = evaluator.declared_validity
    if n is not None and air != 1.0:  # (indices relative to the surrounding air, ``_air_index``)
        n, n_low = n / air, None if n_low is None else n_low / air
    if declared is None or declared.aoi_reference != "AIR_EQUIVALENT" or n is None or abs(n - 1.0) <= INDEX_TOL:
        return None
    if use is not None:
        if use.refused is not None:
            if why is not None:
                why.append(REFUSED_COATING_USE)
            return 1.0, 1.0
        return _coating(evaluator, wavelength_nm, use.air_deg[0], use.air_deg[1], why)
    return _coating(evaluator, wavelength_nm, _asin((n if n_low is None else n_low) * _cos_sin(low)[1]) * _DEGREES,
                    _asin(n * _cos_sin(high)[1]) * _DEGREES if high < _HALF_PI else _GRAZING_DEG, why)


def _coating(evaluator, wavelength_nm: float, low_deg: float, high_deg: float,
             why: list[str] | None = None) -> tuple[float, float]:
    """(largest reflectance, largest transmittance) of a coating over both polarizations and the angles of incidence
    [``low_deg``, ``high_deg``]: its values at the ends and at its table's angles between them (linear between them);
    (1, 1) when the range is not wholly inside its data (its angles of incidence and its wavelength band)."""
    if not bool(np.all(evaluator.in_range(wavelength_nm, np.array([low_deg, high_deg])))):
        if why is not None:
            why.append(COATING_DATA_RANGE)
        return 1.0, 1.0
    angles = [low_deg, high_deg] + [a for a in evaluator.aoi_nodes() if low_deg < a < high_deg]
    c = evaluator.coefficients(wavelength_nm, np.array(angles), location="return screen")
    if c.issue is not None:
        if why is not None:
            why.append(COATING_EVALUATION_ISSUE)
        return 1.0, 1.0
    r = max(float(np.max(c.rs)), float(np.max(c.rp)))
    t = max(float(np.max(c.ts)), float(np.max(c.tp)))
    return min(r, 1.0), min(t, 1.0)


def _coefficients(s: SurfaceG, side: int, low: float, high: float, library: MaterialLibrary,
                  wavelength_nm: float, uses: dict | None = None, refusing: bool = False,
                  why: list[str] | None = None) -> tuple[float, float]:
    """(amplitude reflectance, amplitude transmittance) bounds of light meeting ``s`` from ``side`` at the angles of
    incidence [``low``, ``high``] (rad; module docstring). ``uses``: this wavelength's ``CoatingUse`` per (surface id,
    side); ``refusing``: a refused use counts (1, 1) (the refusal's reading; else the record's stated value). ``why``:
    receives the codes of the bounds that are not a stated or computed reflectance (``_saturation``); it changes no value."""
    inter = s.interaction
    kind = inter.kind
    air = _air_index(library, wavelength_nm) if inter.coating is not None else 1.0

    def use_of(from_side: int):
        use = uses.get((s.surface_id, from_side)) if uses else None
        return use if use is None or use.refused is None or refusing else None

    def thin(mirror: bool) -> tuple[float, float]:
        """A coated mirror or splitter: from a side a medium denser than air holds, by ``_dense_coating``, else over the
        range as it is."""
        r_best = t_best = 0.0
        for from_side in _sides(side):
            if mirror and side == EITHER and inter.reflective_side != "both" and (inter.reflective_side == "positive") != (from_side == POSITIVE):
                continue
            n = _index(library, s.medium_pos if from_side == POSITIVE else s.medium_neg, wavelength_nm)
            found = _dense_coating(inter.coating, n, low, high, use_of(from_side), wavelength_nm, why=why, air=air)
            r, t = _coating(inter.coating, wavelength_nm, low * _DEGREES, high * _DEGREES, why) if found is None else found
            r_best, t_best = max(r_best, math.sqrt(r)), max(t_best, math.sqrt(t))
        return r_best, t_best

    if kind == "neutral_density":
        r = inter.attenuator.reflectance if inter.attenuator is not None else 1.0
        return math.sqrt(r), 1.0
    if kind == "mirror":
        if s.geometry == "plane" and inter.reflective_side != "both" and side != EITHER:
            if (inter.reflective_side == "positive") != (side == POSITIVE):
                return 0.0, 0.0  # (its back side stops the light)
        if inter.coating is None:
            return 1.0, 0.0
        return thin(True)
    if kind == "beam_splitter":
        if inter.coating is None:
            return 1.0, 1.0
        return thin(False)
    if kind == "diffraction_grating":
        if inter.grating.reflection:  # (every order on the side it faces: nothing is known of its efficiencies here)
            return (0.0, 0.0) if side == NEGATIVE else (1.0, 0.0)
        return 1.0, 1.0
    if kind in NON_RETURNING:
        return 0.0, 0.0
    if kind != "refract":
        raise KeyError(f"interaction kind {kind!r} at {s.surface_id} is neither REFLECTING nor NON_RETURNING")
    n_pos = _index(library, s.medium_pos, wavelength_nm)
    n_neg = _index(library, s.medium_neg, wavelength_nm)
    if n_pos is None or n_neg is None:
        if why is not None:
            why.append(INDEX_UNKNOWN)
        return 1.0, 1.0
    range_pos, range_neg = _index_range(library, s.medium_pos, wavelength_nm), _index_range(library, s.medium_neg, wavelength_nm)
    if range_pos is None or range_neg is None:
        if why is not None:
            why.append(INDEX_UNKNOWN)
        return 1.0, 1.0
    r_best = t_best = 0.0
    if range_pos[0] != range_pos[1] or range_neg[0] != range_neg[1]:
        # (a crystal on a side: its waves meet any index between its ordinary and extraordinary ones, ``_index_range``)
        for r1, r2, from_side in {POSITIVE: [(range_pos, range_neg, POSITIVE)], NEGATIVE: [(range_neg, range_pos, NEGATIVE)],
                                  EITHER: [(range_pos, range_neg, POSITIVE), (range_neg, range_pos, NEGATIVE)]}[side]:
            if inter.coating is None:
                r_best = max(r_best, _interface_range(r1, r2, low, high, why))
                t_best = 1.0
                continue
            top = high
            m_low, m_high = r2[0] / r1[1], r2[1] / r1[0]
            if m_low < 1.0:
                largest_critical = _HALF_PI if m_high >= 1.0 else _asin(m_high)
                if low >= largest_critical and use_of(from_side) is None:
                    continue  # (totally reflected for every index: outside the model)
                top = min(high, largest_critical)
            dense = _dense_coating(inter.coating, r1[1], min(low, top), top, use_of(from_side), wavelength_nm, r1[0], why, air=air)
            r, t = _coating(inter.coating, wavelength_nm, low * _DEGREES, top * _DEGREES, why) if dense is None else dense
            r_best, t_best = max(r_best, math.sqrt(r)), max(t_best, math.sqrt(t))
        return r_best, t_best
    media = {POSITIVE: [(n_pos, n_neg, POSITIVE)], NEGATIVE: [(n_neg, n_pos, NEGATIVE)],
             EITHER: [(n_pos, n_neg, POSITIVE), (n_neg, n_pos, NEGATIVE)]}[side]
    for n1, n2, from_side in media:
        if inter.coating is None:
            r_best = max(r_best, _interface(n1, n2, low, high, why))
            t_best = 1.0
            continue
        top = high
        critical = _critical(n1, n2)
        if critical is not None:
            if low >= critical and use_of(from_side) is None:
                continue  # (totally reflected: outside the model)
            top = min(high, critical)
        dense = _dense_coating(inter.coating, n1, min(low, top), top, use_of(from_side), wavelength_nm, why=why, air=air)
        r, t = _coating(inter.coating, wavelength_nm, low * _DEGREES, top * _DEGREES, why) if dense is None else dense
        r_best, t_best = max(r_best, math.sqrt(r)), max(t_best, math.sqrt(t))
    return r_best, t_best


#: The order in which the causes of a reflectance bound of 1 are named when a range meets several.
_SATURATION_ORDER = (REFUSED_COATING_USE, INDEX_UNKNOWN, COATING_DATA_RANGE, COATING_EVALUATION_ISSUE, CRITICAL_ANGLE_IN_RANGE)


def _saturation(s: SurfaceG, side: int, low: float, high: float, library: MaterialLibrary, wavelength_nm: float,
                uses: dict | None = None, refusing: bool = False) -> str:
    """Why the amplitude reflectance bound of ``s`` over [``low``, ``high``] is 1 (called only for a surface of a pair whose
    gain bound is 1): the first of ``_SATURATION_ORDER`` its evaluation met; else, a bare interface whose range reaches
    grazing incidence (the Fresnel reflectance there is 1), or a surface whose stated reflectance is 1."""
    why: list[str] = []
    _coefficients(s, side, low, high, library, wavelength_nm, uses, refusing, why)
    for code in _SATURATION_ORDER:
        if code in why:
            return code
    if s.interaction.kind == "refract" and s.interaction.coating is None:
        return GRAZING_INCIDENCE_IN_RANGE
    return REFLECTS_EVERYTHING


def _nominal_reflectance(s: SurfaceG, side: int, angle: float, library: MaterialLibrary, wavelength_nm: float,
                         uses: dict | None = None) -> float | None:
    """The amplitude reflectance of ``s`` met from ``side`` at the single angle of incidence ``angle`` (rad), by the record's
    stated values: ``None`` where that angle is at or beyond a critical angle, outside the coating's data, met with an
    evaluation issue, or an index is not known (no number stands for the reflectance there)."""
    why: list[str] = []
    r = _coefficients(s, side, angle, angle, library, wavelength_nm, uses, False, why)[0]
    return None if why else r


# ------------------------------------------------------------------------------------------------ legs


@dataclass(frozen=True)
class _Leg:
    """Light going straight: the line it travels along (``u``), the half angle about it of every direction it may take
    (``reach``, rad; at least pi/2: any direction toward the far surface) with its cosine and sine, and the side of the
    far surface it arrives from (for an image at an emitter: the side of the emitter it leaves into)."""

    u: tuple[float, float, float]
    reach: float
    cos: float
    sin: float
    side: int

    @property
    def wide(self) -> bool:
        return self.reach >= _HALF_PI


def _cone(u, reach: float, side: int) -> _Leg:
    if reach >= _HALF_PI:
        return _Leg(u, _HALF_PI, 0.0, 1.0, side)
    c, s = _cos_sin(reach)
    return _Leg(u, reach, c, s, side)


def _leg(start: _Face, end: _Face) -> _Leg:
    """The leg from ``start`` to ``end``: every direction from a point of one bounding sphere to a point of the other lies
    within ``asin((rho_s + rho_e) / d)`` of the line between their centres (any direction when the spheres overlap)."""
    w = (end.centre[0] - start.centre[0], end.centre[1] - start.centre[1], end.centre[2] - start.centre[2])
    dist = math.sqrt(_dot(w, w))
    total = start.radius + end.radius
    low, high = _extent(start, end.axis)
    side = _side_from(end, low, high)
    if dist <= 0.0 or not math.isfinite(total) or total >= dist:
        return _Leg(_unit(w) if dist > 0.0 else end.axis, _HALF_PI, 0.0, 1.0, side)
    t = total / dist
    return _Leg((w[0] / dist, w[1] / dist, w[2] / dist), _asin(t), math.sqrt((1.0 - t) * (1.0 + t)), t, side)


def _overlap(a: _Leg, b: _Leg) -> bool:
    """Whether two cones of directions share a direction (the angle between their axes at most the sum of their half
    angles), resolved to ``OVERLAP_COS_RESOLUTION``."""
    if a.wide or b.wide:
        return True
    return _dot(a.u, b.u) >= a.cos * b.cos - a.sin * b.sin - OVERLAP_COS_RESOLUTION


def _momenta(leg: _Leg, axis, n: float) -> tuple[float, float]:
    """The least and the largest ``n sin(angle)`` of the cone's directions to the normal ``axis`` (``n`` the index of the
    medium the light is in)."""
    if leg.wide:
        return 0.0, n
    c = abs(_dot(leg.u, axis))
    s = math.sqrt(max(0.0, (1.0 - c) * (1.0 + c)))
    high = 1.0 if c * leg.cos - s * leg.sin <= 0.0 else min(1.0, s * leg.cos + c * leg.sin)
    low = s * leg.cos - c * leg.sin
    return n * (low if low > 0.0 else 0.0), n * high


def _incidence(leg: _Leg, face: _Face) -> tuple[float, float]:
    """The angles of incidence (rad, low and high) at which a leg's light can meet the surface: its directions on every
    normal of the surface's cone."""
    if leg.wide or face.cone >= _HALF_PI:
        return 0.0, _HALF_PI
    phi = _angle(leg.u, face.axis)
    psi = phi if phi <= _HALF_PI else _PI - phi
    width = leg.reach + face.cone
    return max(0.0, psi - width), min(_HALF_PI, psi + width)


@dataclass(frozen=True)
class _Legs:
    """Legs as arrays (one row each): axes, half angles with their cosines and sines, wide flags and sides."""

    u: np.ndarray
    reach: np.ndarray
    cos: np.ndarray
    sin: np.ndarray
    wide: np.ndarray
    side: np.ndarray

    @staticmethod
    def of(legs: list[_Leg]) -> _Legs:
        return _Legs(np.array([leg.u for leg in legs], dtype=np.float64).reshape(-1, 3),
                     np.array([leg.reach for leg in legs], dtype=np.float64),
                     np.array([leg.cos for leg in legs], dtype=np.float64),
                     np.array([leg.sin for leg in legs], dtype=np.float64),
                     np.array([leg.wide for leg in legs], dtype=bool),
                     np.array([leg.side for leg in legs], dtype=np.int64))

    def take(self, rows) -> _Legs:
        return _Legs(self.u[rows], self.reach[rows], self.cos[rows], self.sin[rows], self.wide[rows], self.side[rows])

    @staticmethod
    def join(parts: list[_Legs]) -> _Legs:
        return _Legs(*(np.concatenate([getattr(part, name) for part in parts]) for name in ("u", "reach", "cos", "sin", "wide", "side")))

    @staticmethod
    def choose(mask: np.ndarray, a: _Legs, b: _Legs) -> _Legs:
        """Row by row ``a`` where ``mask``, else ``b``."""
        return _Legs(np.where(mask[:, None], a.u, b.u), np.where(mask, a.reach, b.reach), np.where(mask, a.cos, b.cos),
                     np.where(mask, a.sin, b.sin), np.where(mask, a.wide, b.wide), np.where(mask, a.side, b.side))


def _any_meets(legs: _Legs, face: _Face, ways: list[_Leg]) -> bool:
    """Whether light of any of the legs reflected at ``face`` can go off along the directions of any of the ways: the
    reflection's image of a leg's directions, widened by twice the cone of the normals (turning a mirror by an angle
    turns its reflection by twice it), meets them -- the angle between their axes at most the sum of the half angles,
    compared through cosines and resolved to ``OVERLAP_COS_RESOLUTION`` (``_overlap``); every pair elementwise."""
    if legs.reach.shape[0] == 0 or not ways:
        return False
    if face.cone >= _HALF_PI or bool(np.any(legs.wide)) or any(way.wide for way in ways):
        return True
    c2, s2 = _cos_sin(2.0 * face.cone) if face.cone > 0.0 else (1.0, 0.0)
    i_reach = (legs.reach + 2.0 * face.cone)[:, None]
    i_cos = (legs.cos * c2 - legs.sin * s2)[:, None]
    i_sin = (legs.sin * c2 + legs.cos * s2)[:, None]
    w_reach = np.array([way.reach for way in ways], dtype=np.float64)[None, :]
    if bool(np.any(i_reach + w_reach >= _PI)):
        return True
    # the reflected axes (``_reflected``: d - 2 (d . n) n, normalized)
    n = face.axis
    u = legs.u
    c = u[:, 0] * n[0] + u[:, 1] * n[1] + u[:, 2] * n[2]
    v0, v1, v2 = u[:, 0] - 2.0 * c * n[0], u[:, 1] - 2.0 * c * n[1], u[:, 2] - 2.0 * c * n[2]
    length = np.sqrt(v0 * v0 + v1 * v1 + v2 * v2)
    r0, r1, r2 = (v0 / length)[:, None], (v1 / length)[:, None], (v2 / length)[:, None]
    wu = np.array([way.u for way in ways], dtype=np.float64)
    dot = r0 * wu[None, :, 0] + r1 * wu[None, :, 1] + r2 * wu[None, :, 2]
    w_cos = np.array([way.cos for way in ways], dtype=np.float64)[None, :]
    w_sin = np.array([way.sin for way in ways], dtype=np.float64)[None, :]
    return bool(np.any(dot >= i_cos * w_cos - i_sin * w_sin - OVERLAP_COS_RESOLUTION))


def _incidences(legs: _Legs, face: _Face) -> tuple[np.ndarray, np.ndarray]:
    """``_incidence`` of every leg, elementwise (the same operations on the same values)."""
    k = legs.reach.shape[0]
    if face.cone >= _HALF_PI:
        return np.zeros(k), np.full(k, _HALF_PI)
    a, b = legs.u, face.axis
    cx = a[:, 1] * b[2] - a[:, 2] * b[1]
    cy = a[:, 2] * b[0] - a[:, 0] * b[2]
    cz = a[:, 0] * b[1] - a[:, 1] * b[0]
    phi = atan2(np.sqrt(cx * cx + cy * cy + cz * cz), a[:, 0] * b[0] + a[:, 1] * b[1] + a[:, 2] * b[2])
    psi = np.where(phi <= _HALF_PI, phi, _PI - phi)
    width = legs.reach + face.cone
    low, high = psi - width, psi + width
    low = np.where(legs.wide, 0.0, np.where(low > 0.0, low, 0.0))
    high = np.where(legs.wide, _HALF_PI, np.where(high < _HALF_PI, high, _HALF_PI))
    return low, high


def _snell(leg: _Leg, axis, n1: float, n2: float, side: int) -> _Leg:
    """The cone light of a leg goes on in through a plane of normal ``axis`` from index ``n1`` into ``n2`` (leaving into
    ``side``): its central direction refracted, its half angle times the largest angular magnification of Snell's law over
    the cone's angles of incidence, ``max(n1 cos i / (n2 cos t), n1 / n2)`` (meridional, sagittal; the meridional one is
    monotone in the angle); every direction toward that side where part of the cone meets the plane at grazing incidence
    or at or beyond the critical angle (what it passes then leaves near grazing)."""
    if leg.wide:
        return _Leg(leg.u, _HALF_PI, 0.0, 1.0, side)
    c = abs(_dot(leg.u, axis))
    s = math.sqrt(max(0.0, (1.0 - c) * (1.0 + c)))
    cos_high = c * leg.cos - s * leg.sin  # (the cosine of the widest angle of incidence)
    sin_high = 1.0 if cos_high <= 0.0 else s * leg.cos + c * leg.sin
    cos_low = 1.0 if s * leg.cos - c * leg.sin <= 0.0 else c * leg.cos + s * leg.sin
    ratio = n1 / n2
    if cos_high <= 0.0 or ratio * sin_high >= 1.0:
        return _Leg(leg.u, _HALF_PI, 0.0, 1.0, side)
    centre = _refracted(leg.u, axis, n1, n2)
    if centre is None:
        return _Leg(leg.u, _HALF_PI, 0.0, 1.0, side)

    def magnification(cos_i: float) -> float:
        sin2 = (1.0 - cos_i) * (1.0 + cos_i)
        cos_t = math.sqrt(max(0.0, 1.0 - ratio * ratio * sin2))
        return math.inf if cos_t <= 0.0 else ratio * cos_i / cos_t

    largest = max(magnification(cos_low), magnification(cos_high), ratio)
    return _cone(centre, largest * leg.reach, side)


def _snell_curved(leg: _Leg, face: _Face, n1: float, n2: float, side: int) -> _Leg:
    """The cone light of a leg goes on in through a curved interface whose normals lie within ``face.cone`` of its axis,
    from index ``n1`` into ``n2``: about its central direction refracted at the axis, within ``M (r + c) + c`` (``r`` the
    leg's half angle, ``c`` the normals' cone, ``M`` the largest angular magnification of Snell's law over the angles of
    incidence the leg meets the normals at: turning the normal by ``c`` changes the angle of incidence by at most ``c`` and
    the refracted direction by at most ``(1 + M) c``); every direction toward that side where those angles reach grazing
    incidence or the critical angle."""
    if leg.wide or face.cone >= _HALF_PI:
        return _Leg(leg.u, _HALF_PI, 0.0, 1.0, side)
    wider = _cone(leg.u, leg.reach + face.cone, side)
    plane = _snell(wider, face.axis, n1, n2, side)
    if plane.wide or wider.reach <= 0.0:
        return plane if not plane.wide else _Leg(leg.u, _HALF_PI, 0.0, 1.0, side)
    magnification = plane.reach / wider.reach  # (the largest magnification _snell found over the wider cone's angles)
    return _cone(plane.u, magnification * (leg.reach + face.cone) + face.cone, side)


def _other(side: int) -> int:
    return NEGATIVE if side == POSITIVE else POSITIVE if side == NEGATIVE else EITHER


def _sides(side: int) -> tuple[int, ...]:
    return (POSITIVE, NEGATIVE) if side == EITHER else (side,)


# ------------------------------------------------------------------------------------------------ results


@dataclass(frozen=True)
class ReturnPair:
    """Two surfaces facing each other (ids), the amplitude gain ``g = r_a r_b`` of a round trip between them, the angles
    of incidence (deg, low and high) at which their light meets each of them, the surfaces between them its returns
    cross, and the bound of all its returns over one crossing (``g / (1 - g)``, fewer between parallel planes; ``inf``:
    the pair bounds nothing)."""

    surfaces: tuple[str, str]
    gain: float
    angles_deg: tuple[tuple[float, float], tuple[float, float]] = ((0.0, 0.0), (0.0, 0.0))
    crossed: tuple[str, ...] = ()
    returns: float = 0.0
    #: A pair that bounds nothing (``returns`` ``inf``): per surface why its reflectance bound over its angles is 1
    #: (``_saturation``), and the pair at its centre-line incidence -- the angle (deg) at which the line between the two
    #: surfaces' centres meets each (between parallel planes: the least angle their light can have), the gain ``g0`` of a
    #: round trip at those angles by the records' stated values and ``g0 / (1 - g0)``. ``None`` where a surface has no
    #: stated reflectance at that angle or ``g0`` is 1. The centre-line value is one ray's: light of the pair off that line
    #: (up to the critical angle inside a lens) is not in it, so it bounds nothing and enters no proven field.
    saturation: tuple[str, str] | None = None
    nominal_angles_deg: tuple[float, float] | None = None
    nominal_gain: float | None = None
    nominal_returns: float | None = None


#: The most pairs a return cause names (``ReturnGains.return_causes``); the rest are counted.
NAMED_PAIRS = 3


@dataclass(frozen=True)
class ReturnGains:
    """Per surface id (physical surfaces and observation planes) the factors ``(f_positive, f_negative)`` by which the
    field of the light left out as negligible arriving from each side may exceed one crossing of its whole power
    (``inf``: not bounded), at one vacuum wavelength, and the pairs found (module docstring). The pairs' returns leave them
    toward every surface, so every surface has the same factor from either side, ``1 + sum`` of the pairs' bounds; a
    surface absent from ``by_surface`` (no pair in the scene) has 1."""

    wavelength_nm: float
    by_surface: dict[str, tuple[float, float]] = field(default_factory=dict)
    pairs: tuple[ReturnPair, ...] = ()
    #: The open paths (module docstring): the number of surfaces that may reflect light, ``N``; the most straight legs the
    #: light travels between two reflections, ``L = T + 1`` (``T`` the lenses and refracting surfaces that may turn it);
    #: the surfaces it meets at most once between two reflections (``once``: every thin lens and every refracting surface
    #: between media of known, different index); per other surface the most times a straight leg meets it, ``K``, where
    #: it is more than 1 (``per_leg``: a sphere or a conic 2, an asphere or an axicon more; a plane surface or an
    #: observation plane, absent, 1); and ``R``, the sum over the reflecting surfaces of their power reflectance bounds
    #: over every angle and side, each times the most times the light meets the surface between two reflections (1 for
    #: those of ``once``, ``K L`` for the others).
    reflectors: int = 0
    reflectance: float = 0.0
    legs: int = 1
    once: frozenset[str] = frozenset()
    per_leg: dict[str, int] = field(default_factory=dict)
    #: The same reflecting surfaces, each as (its amplitude reflectance bound ``sqrt(R_s)``, its meetings ``m_s``), in the
    #: scene's order (``crossing_field``: pieces of one reflection count from different sequences of surfaces are mutually
    #: coherent, their fields add).
    reflecting: tuple[tuple[float, int], ...] = ()
    #: Potential multi-surface returns not covered by the pair model. The
    #: all-orders series, when available, includes repeated reflectors.
    closed_cycles: tuple[tuple[str, ...], ...] = ()
    closed_series: float | None = None
    #: The coating uses refused for the left-out light (``validation/coating_use.py``), the pairs whose gain bound that makes 1
    #: (each refused surface counting as reflecting everything; a pair already at 1 is in ``pairs``), and the surfaces of both:
    #: their exposure has no factor (``exposure_factor``). Every other reading here is the record's stated value.
    refused_coatings: tuple = ()
    refused_pairs: tuple[ReturnPair, ...] = ()
    withheld: frozenset[str] = frozenset()
    #: The analysis' work (``validation/return_work.py``). ``exhausted``: it reached the work the run's limit
    #: (``NumericalConfig.max_return_analysis_work``) left it before it finished. In the pair screen (``stage`` ``pairs``):
    #: the pairs and their factors are not known (``factor``, ``power_factor``, ``exposure_factor``: inf; no ``pairs``)
    #: and the closed returns were not screened; in the fold-solid analysis (``solid_folds``) or the closed-return screen:
    #: whether a closed return exists is not known (the pairs are). Either way the open paths' counts bound nothing
    #: (``crossings``, ``crossing_field``: inf).
    work: ReturnWork = field(default_factory=ReturnWork)
    #: What the closed-return screen's trail search did (``validation/closed_returns.py``): the longest trail it examined,
    #: the frames it held against their limit, and how it ended (``NOT_STARTED``: the pair screen or the fold-solid
    #: analysis reached the work limit; ``PARALLEL_STACK``: parallel planes, no trail search).
    search: ClosedReturnSearch = field(default_factory=ClosedReturnSearch)
    #: The surfaces and observation planes (``""``: a reading for no surface) whose bound read a value an exhausted
    #: analysis withholds (filled as they are read: the bounds it withheld), and of them those read for the power the
    #: light carries across (``crossings``, ``power_factor``: a transparent plane's power upper bound; the others read its
    #: field, ``factor``, ``crossing_field``: irradiance, coherent power, exposure).
    withheld_by_work: set[str] = field(default_factory=set, compare=False, repr=False)
    power_withheld_by_work: set[str] = field(default_factory=set, compare=False, repr=False)
    #: The internal returns of plane-faced solids (``validation/solid_folds.py``): per entry of ``reflecting``, the sums of the
    #: closed walks inside its solid at that surface and of the alternations with a paired surface alone, in amplitude and in
    #: power (``G``, ``G2``, ``A``, ``A2``; empty: no solid with a fold), and the solids analysed. The open paths' counts add the
    #: walks with other loops: ``_looped``.
    loops: tuple[tuple[float, float, float, float], ...] = ()
    solid_folds: tuple = ()

    @property
    def screened(self) -> bool:
        """The analysis finished within its work limit."""
        return not self.work.exhausted

    @property
    def closed_return_screen(self) -> str:
        """The state of the closed-return screen (``validation/return_work.py``): ``INCOMPLETE`` (the work limit came first,
        in either stage: whether the left-out light may circulate is not known), ``POTENTIAL_RETURN`` (closed returns the
        screen cannot rule out, ``closed_cycles``, a plane-faced fold solid without a certified sum among them
        (``validation/solid_folds.py``): bounded by ``closed_series`` where that is not ``None``, else by nothing)
        or ``COMPLETED_NO_RETURN`` (none: the open paths' count bounds the light)."""
        if self.work.exhausted:
            return INCOMPLETE
        return POTENTIAL_RETURN if self.closed_cycles else COMPLETED_NO_RETURN

    def return_causes(self, surface_id: str | None = None) -> tuple[str, ...]:
        """Why nothing bounds the returns of the left-out light across the plane or onto the surface ``surface_id`` in a
        finished analysis (none where it did not finish: ``withheld_by_work`` says that), each as a clause: a pair of
        surfaces facing each other whose returns no factor bounds there, and each closed return without a series -- a
        closed path of the trail search (its surfaces in order), the parallel surfaces of a stack (``PARALLEL_STACK``) or
        the internal folds of a plane-faced solid (``validation/solid_folds.py``). Reads nothing an exhausted analysis
        withholds."""
        if not self.screened:
            return ()
        out = []
        if surface_id is not None:
            a, b = self.sides(surface_id)
            if not (math.isfinite(a) and math.isfinite(b)):
                held = self.saturated_pairs()
                named = "; ".join(f"{p.surfaces[0]} and {p.surfaces[1]}" for p in held[:NAMED_PAIRS])
                more = f"; and {len(held) - NAMED_PAIRS} more pairs" if len(held) > NAMED_PAIRS else ""
                out.append(f"the surfaces {named}{more}, facing each other, may hold it" if held else
                           "two surfaces facing each other may hold it")
        if self.closed_cycles and self.closed_series is None:
            solids = {fold.surfaces: fold.owner_id for fold in self.solid_folds if not fold.bounded}
            reasons = {fold.surfaces: fold.uncertified for fold in self.solid_folds if not fold.bounded}
            for cycle, kind in zip(self.closed_cycles, self.closed_return_kinds):
                if kind == "solid_folds":
                    out.append(f"the internal folds of {solids[cycle]} ({', '.join(cycle)}) may hold it (the fold analysis "
                               f"has no certified sum for this solid: {reasons[cycle]})")
                elif kind == "parallel_stack":
                    out.append(f"the parallel surfaces {', '.join(cycle)} may hold it between them")
                else:
                    out.append(f"a closed path through {' -> '.join(cycle)} may circulate it")
        return tuple(out)

    @property
    def closed_return_kinds(self) -> tuple[str, ...]:
        """What each of ``closed_cycles`` is: ``path`` (a closed trail of the search, its surfaces in order),
        ``parallel_stack`` (the surfaces of a stack of parallel planes: ``PARALLEL_STACK``) or ``solid_folds`` (the faces of a
        plane-faced solid whose internal folds have no certified sum, ``validation/solid_folds.py``)."""
        solids = {fold.surfaces for fold in self.solid_folds if not fold.bounded}
        return tuple("solid_folds" if cycle in solids else "parallel_stack" if self.search.termination == PARALLEL_STACK
                     else "path" for cycle in self.closed_cycles)

    def _pairs_open(self, surface_id: str | None) -> bool:
        """The pair screen did not finish: no pair factor is known (and the reading is recorded)."""
        if self.work.exhausted and self.work.stage == PAIR_STAGE:
            self.withheld_by_work.add(surface_id or "")
            return True
        return False

    def sides(self, surface_id: str) -> tuple[float, float]:
        return self.by_surface.get(surface_id, (1.0, 1.0))

    def factor(self, surface_id: str) -> float:
        """The factor on the field of the left-out light arriving from either side, over its open paths' crossings
        (``crossing_field``): ``(f_positive + f_negative) / 2`` (1 where no pair returns light to the surface; inf where the
        pair screen did not finish)."""
        if self._pairs_open(surface_id):
            return math.inf
        a, b = self.sides(surface_id)
        return math.inf if not (math.isfinite(a) and math.isfinite(b)) else 0.5 * (a + b)

    def exposure_factor(self, surface_id: str) -> float:
        """``factor`` as an exposure upper bound reads it: ``inf`` on a surface a refused coating use withholds (``withheld``)."""
        return math.inf if surface_id in self.withheld else self.factor(surface_id)

    def nominal_factor(self, surface_id: str) -> float | None:
        """``factor`` with each pair that bounds nothing counted at its centre-line incidence (``ReturnPair.nominal_returns``):
        ``1 + sum`` of the pairs' proven bounds where finite and of their centre-line values elsewhere. An estimate, never a
        bound: it enters no proven field and no status. ``None`` where ``factor`` is finite (nothing to estimate), where the
        pair screen did not finish, where a pair has no centre-line value, and on a surface a refused coating use withholds."""
        if not self.screened and self.work.stage == PAIR_STAGE:
            return None
        if surface_id in self.withheld or math.isfinite(self.factor(surface_id)):
            return None
        total = 1.0
        for pair in self.pairs:
            if math.isfinite(pair.returns):
                total += pair.returns
            elif pair.nominal_returns is None:
                return None
            else:
                total += pair.nominal_returns
        return total

    def saturated_pairs(self) -> tuple[ReturnPair, ...]:
        """The pairs whose returns nothing bounds by the records' stated values (gain bound 1)."""
        return tuple(pair for pair in self.pairs if not math.isfinite(pair.returns))

    def power_factor(self, surface_id: str) -> float:
        """The factor on the power the left-out light may carry across the surface, over its open paths' crossings
        (``crossings``): ``(f_positive^2 + f_negative^2) / 2`` (inf where the pair screen did not finish)."""
        if self._pairs_open(surface_id):
            self.power_withheld_by_work.add(surface_id or "")
            return math.inf
        a, b = self.sides(surface_id)
        return math.inf if not (math.isfinite(a) and math.isfinite(b)) else 0.5 * (a * a + b * b)

    def meetings(self, surface_id: str | None = None) -> int:
        """The most times the light between two reflections meets the surface or crosses the plane ``surface_id``: once a
        lens or a refracting surface between different media (``once``), any other surface ``K`` times per leg
        (``per_leg``; once per leg a plane surface or an observation plane, also where ``surface_id`` is ``None``)."""
        if surface_id is None:
            return self.legs
        if surface_id in self.once:
            return 1
        return self.legs * self.per_leg.get(surface_id, 1)

    def crossings(self, surface_id: str | None = None) -> float:
        """The most power one piece of left-out light brings across the plane or onto the surface ``surface_id`` on its
        open paths, its crossings' powers added, as a multiple of the piece's power: ``m sum_{j=0}^{N} min(1, R^j)`` (the
        light after ``j`` reflections, meeting it at most ``m`` times: ``meetings``), at least ``LEAST_CROSSINGS``."""
        if not self.screened:
            self.withheld_by_work.add(surface_id or "")
            self.power_withheld_by_work.add(surface_id or "")
            return math.inf
        if self.closed_cycles:
            return math.inf if self.closed_series is None else max(LEAST_CROSSINGS, self.meetings(surface_id) * self.closed_series)
        return self._open_crossings(surface_id)

    def _open_crossings(self, surface_id: str | None, certified_only: bool = False) -> float:
        """``crossings`` of the open paths (``certified_only``: without the loops of the solids whose folds have no
        certified sum, ``nominal_crossing_field``)."""
        total, term = 1.0, 1.0
        for _ in range(self.reflectors):
            term = min(1.0, term * self.reflectance)
            total += term
        if self.loops:
            total += self._looped(2, certified_only)
        return max(LEAST_CROSSINGS, self.meetings(surface_id) * total)

    def crossing_field(self, surface_id: str | None = None) -> float:
        """The most field one piece of left-out light brings to a point of the plane or the surface ``surface_id`` on its
        open paths, as a multiple of the amplitude of the diffraction limit of the piece's power: ``sqrt(2) m sum_{j=0}^{N}
        a_j``. The light after ``j`` reflections is made of pieces from different sequences of reflecting surfaces -- on an
        open path no surface reflects it twice, so ``j`` distinct ones, each at one of its ``m_s`` meetings --, mutually
        coherent, whose fields add: each reflection takes at most ``sqrt(R_s)`` of the field, so their amplitudes sum to at
        most ``j! e_j(m_s sqrt(R_s))`` (``e_j`` the elementary symmetric polynomial over the reflecting surfaces: the sum
        over ordered sequences of distinct surfaces; at most ``A^j``, ``A = sum m_s sqrt(R_s)``), and at most ``j!
        e_j(m_s)`` pieces share at most ``min(1, j! e_j(m_s R_s))`` of the power (at most ``min(1, R^j)``), so their
        amplitudes sum to at most the square root of the product (Cauchy-Schwarz): ``a_j`` the smaller. Meeting the surface
        at most ``m`` times, each time split between the two sides; at least ``LEAST_CROSSINGS``."""
        if not self.screened:
            self.withheld_by_work.add(surface_id or "")
            return math.inf
        if self.closed_cycles:
            return math.inf if self.closed_series is None else max(LEAST_CROSSINGS, SQRT_TWO * self.meetings(surface_id) * self.closed_series)
        return self._open_crossing_field(surface_id)

    def nominal_returns_factor(self) -> float | None:
        """Where closed returns without a series leave the open paths' counts unbounded (``crossings``,
        ``crossing_field``: inf): the factor ``1 / (1 - G)`` of the returns around the one closed trail the search found,
        ``G`` its round gain bound (``ClosedReturnSearch.cycle_gain``) -- 1 where the only such returns are the folds of
        solids without a certified sum (their loops are then left out of the count). ``None``: the counts are bounded
        (nothing to estimate), the analysis did not finish, ``G`` is 1, or a stack of parallel planes holds the light
        (its quantitative answer is a cavity's, not a series over one trail). An estimate: the other closed returns among
        the search's candidates and the uncertified folds are not in it."""
        if not self.screened or not self.closed_cycles or self.closed_series is not None:
            return None
        factor = 1.0
        for kind in self.closed_return_kinds:
            if kind == "parallel_stack":
                return None
            if kind == "path":
                gain = self.search.cycle_gain
                if gain is None or not gain < 1.0:
                    return None
                factor = factor / (1.0 - gain)
        return factor

    def nominal_crossing_field(self, surface_id: str | None = None) -> float | None:
        """``crossing_field`` of the open paths times ``nominal_returns_factor`` (``None`` with it)."""
        factor = self.nominal_returns_factor()
        return None if factor is None else self._open_crossing_field(surface_id, True) * factor

    def _open_crossing_field(self, surface_id: str | None, certified_only: bool = False) -> float:
        """``crossing_field`` of the open paths (``certified_only``: as ``_open_crossings``)."""
        n = len(self.reflecting)
        amplitude, pieces, power = [1.0] + [0.0] * n, [1.0] + [0.0] * n, [1.0] + [0.0] * n
        for r, meets in self.reflecting:
            # (ordered sums j! e_j: adding a surface of weight x gives o_j + j x o_(j-1); highest first)
            for j in range(n, 0, -1):
                amplitude[j] += j * (meets * r) * amplitude[j - 1]
                pieces[j] += j * meets * pieces[j - 1]
                power[j] += j * (meets * (r * r)) * power[j - 1]
        total = 1.0
        for j in range(1, n + 1):
            total += min(amplitude[j], math.sqrt(pieces[j] * min(1.0, power[j])))
        if self.loops:
            total += self._looped(1, certified_only)
        return max(LEAST_CROSSINGS, SQRT_TWO * (self.meetings(surface_id) * total))

    def _looped(self, power: int, certified_only: bool = False) -> float:
        """What the walks with loops inside plane-faced solids add (``validation/solid_folds.py``), as a multiple of one crossing
        of the piece (``power`` 1: its field; 2: its power): over the open paths of distinct reflecting surfaces ``s_1 .. s_j``,
        each reflection at one of its ``m`` meetings with weight ``m r`` (``m R`` in power), the walks that close at least one
        loop other than alternations with a paired surface (which the pairs count) add at most ``prod m r (prod G - prod A)``
        (``G - A`` holds those loops at a surface, ``A`` the alternations alone; telescoping over the path): ``sum_j j!
        (e_j(m r G) - e_j(m r A))`` (the ordered sums of the elementary symmetric polynomials, each bounded on its own side of
        rounding). inf where a loop sum is not bounded (``certified_only``: such a surface counts without its loops)."""
        n = len(self.reflecting)
        with_loops, without = [1.0] + [0.0] * n, [1.0] + [0.0] * n
        for (r, meets), (g, g2, a, a2) in zip(self.reflecting, self.loops):
            weight = meets * r if power == 1 else meets * (r * r)
            factor, paired = (g, a) if power == 1 else (g2, a2)
            if not (math.isfinite(factor) and math.isfinite(paired)):
                if not certified_only:
                    return math.inf
                factor = paired = 1.0
            for j in range(n, 0, -1):
                with_loops[j] += j * (weight * factor) * with_loops[j - 1]
                without[j] += j * (weight * paired) * without[j - 1]
        extra = 0.0
        for j in range(1, n + 1):
            extra += with_loops[j] * (1.0 + LOOP_ROUNDING) - without[j] * (1.0 - LOOP_ROUNDING)
        return max(extra, 0.0)


# ------------------------------------------------------------------------------------------------ the scene


#: Rounds of the passing on of momenta between parallel planes, per plane of a class (each round can only widen a plane's
#: range to an end already found, so they settle within that many rounds; beyond, every momentum counts).
MOMENTUM_ROUNDS = 4


class _Scene:
    """What the screen needs of a scene at one wavelength: the faces, the media's indices, the legs (cached), the sources'
    beams, the thin lenses' largest deflections, and per class of parallel planes the momenta of the light that can meet
    each of its planes (``momenta``), and the analysis' work budget (``work``, ``validation/return_work.py``)."""

    def __init__(self, scene: Scene, library: MaterialLibrary, wavelength_nm: float, faces: list[_Face],
                 uses: dict | None = None, refusing: bool = False, work: WorkCounter | None = None) -> None:
        self.scene, self.library, self.wavelength_nm, self.faces = scene, library, wavelength_nm, faces
        self.uses, self.refusing, self.work = uses, refusing, work
        #: The stage the work now spent belongs to (``closed_returns`` sets its own).
        self.stage = PAIR_STAGE
        #: What the closed-return trail search did (``closed_returns`` leaves it here).
        self.search: ClosedReturnSearch | None = None
        self.table = _Faces(faces)
        self.surfaces = scene.surfaces
        self.kinds = [s.interaction.kind for s in scene.surfaces]
        unclassified = sorted(set(self.kinds) - REFLECTING - NON_RETURNING)
        if unclassified:
            raise KeyError(f"interaction kinds {unclassified} are neither REFLECTING nor NON_RETURNING")
        self.redirecting = [i for i, kind in enumerate(self.kinds) if kind in REDIRECTING]
        self._indices: dict[str, float | None] = {}
        self._ranges: dict[str, tuple[float, float] | None] = {}
        self._legs: dict[tuple[int, int], _Leg] = {}
        spread = min(RETURN_ANGLE_SIGMAS * _divergence(scene, library, wavelength_nm), _HALF_PI)
        self.beams: list[tuple[_Face, _Leg]] = []
        for src in scene.sources:
            direction = _unit(src.direction)
            ball = _point(src.position, RETURN_ANGLE_SIGMAS * _source_radius(src, library, wavelength_nm), direction)
            self.beams.append((ball, _cone(direction, spread, EITHER)))
        # (a thin lens turns a ray by its height above the lens's axis, through the surface's origin: a decentred aperture
        # reaches its offset plus its radius)
        self.bends = {i: _lens_bend(scene.surfaces[i], wavelength_nm, _aperture_radius(scene.surfaces[i].aperture))
                      for i, kind in enumerate(self.kinds) if kind in BENDING}
        self._images: dict[int, list[_Leg]] = {}
        # the classes of parallel reflecting planes (each plane -> the first plane of its class)
        self.classes: dict[int, int] = {}
        for i, kind in enumerate(self.kinds):
            if kind not in REFLECTING or not faces[i].plane:
                continue
            for root in sorted(set(self.classes.values())):
                if abs(_dot(faces[root].axis, faces[i].axis)) >= 1.0 - PARALLEL_TOL:
                    self.classes[i] = root
                    break
            else:
                self.classes[i] = i
        self._momenta: dict[int, dict[int, tuple[float, float]]] = {}
        self._bounds: dict[tuple[int, int], tuple[float, float, float, float]] = {}
        self._walks: dict[tuple[int, int], float] = {}
        self._toward: dict[tuple[int, int], int] = {}
        self._passed: dict[tuple[int, int], _Leg] = {}
        self._coefficients: dict[tuple[int, int, str, str], tuple[float, float]] = {}
        self._rows: dict[int, _Legs] = {}
        #: Every surface that may turn light it passes (thin lenses, refracting surfaces between different media), and per
        #: such surface the side of it every surface lies on (``toward``), as an array (``relays``).
        self.turning = [c for c in range(len(faces)) if self.turns(c)]
        self._side_matrix: np.ndarray | None = None

    def passed_rows(self, c: int) -> _Legs:
        """``passed(src, c)`` for every surface index ``src`` (the surface itself: an empty cone), remembered per turning
        surface."""
        found = self._rows.get(c)
        if found is None:
            found = self._rows[c] = _Legs.of([self.passed(src, c) if src != c else _Leg(self.faces[c].axis, 0.0, 1.0, 0.0, EITHER)
                                              for src in range(len(self.faces))])
        return found

    def light_into(self, me: int, other: int, bending: list[int]) -> tuple[_Legs, _Legs]:
        """The pair's light at ``me`` as (cones, legs) row by row: straight from ``other`` (the leg twice), and for every
        turning surface ``c`` the cones it passes the light of ``other`` and of the other turning surfaces on in that meet
        the leg from ``c`` (``_overlap``, elementwise)."""
        direct = _Legs.of([self.leg(other, me)])
        if not bending:
            return direct, direct
        turns = np.array(bending, dtype=np.int64)
        sources = np.array([other, *bending], dtype=np.int64)
        rows = [self.passed_rows(c) for c in bending]
        u = np.stack([row.u for row in rows])[:, sources]
        cos = np.stack([row.cos for row in rows])[:, sources]
        sin = np.stack([row.sin for row in rows])[:, sources]
        wide = np.stack([row.wide for row in rows])[:, sources]
        lasts = _Legs.of([self.leg(c, me) for c in bending])
        dot = (u[:, :, 0] * lasts.u[:, 0:1] + u[:, :, 1] * lasts.u[:, 1:2]) + u[:, :, 2] * lasts.u[:, 2:3]
        meets = wide | lasts.wide[:, None] | (dot >= cos * lasts.cos[:, None] - sin * lasts.sin[:, None] - OVERLAP_COS_RESOLUTION)
        meets &= sources[None, :] != turns[:, None]
        which, column = np.nonzero(meets)
        cones = _Legs(u[which, column], np.stack([row.reach for row in rows])[:, sources][which, column], cos[which, column],
                      sin[which, column], wide[which, column], np.stack([row.side for row in rows])[:, sources][which, column])
        return _Legs.join([direct, cones]), _Legs.join([direct, lasts.take(which)])

    def coefficients(self, i: int, side: int, low: float, high: float) -> tuple[float, float]:
        """``_coefficients`` of surface ``i``, remembered per side and angle range (the same range recurs across pairs)."""
        key = (i, side, low.hex(), high.hex())
        found = self._coefficients.get(key)
        if found is None:
            found = self._coefficients[key] = _coefficients(self.surfaces[i], side, low, high, self.library, self.wavelength_nm, self.uses, self.refusing)
        return found

    def index(self, medium: str) -> float | None:
        if medium not in self._indices:
            self._indices[medium] = _index(self.library, medium, self.wavelength_nm)
        return self._indices[medium]

    def crystal(self, medium: str) -> bool:
        """Whether ``medium`` is a uniaxial crystal (two indices; its extraordinary wave walks off its wave normal)."""
        return self.library.uniaxial(medium) is not None

    def crystal_face(self, i: int) -> bool:
        """Whether a crystal lies on either side of surface ``i``."""
        s = self.surfaces[i]
        return self.crystal(s.medium_pos) or self.crystal(s.medium_neg)

    def side_range(self, i: int, side: int) -> tuple[float, float] | None:
        """The least and the largest index light meets on ``side`` of surface ``i`` (``_index_range``; both sides for
        ``EITHER``; ``None``: not known)."""
        s = self.surfaces[i]
        found = []
        for medium in ((s.medium_pos,) if side == POSITIVE else (s.medium_neg,) if side == NEGATIVE else (s.medium_pos, s.medium_neg)):
            if medium not in self._ranges:
                self._ranges[medium] = _index_range(self.library, medium, self.wavelength_nm)
            found.append(self._ranges[medium])
        if any(r is None for r in found):
            return None
        return min(r[0] for r in found), max(r[1] for r in found)  # type: ignore[index]

    def walk_off(self, i: int, side: int) -> float:
        """The largest walk-off angle (rad) of the extraordinary wave of a crystal on ``side`` of surface ``i`` (0 for an
        isotropic medium): ``tan rho_max = |n_o^2 - n_e^2| / (2 n_o n_e)``, its value at ``tan theta = n_o / n_e``."""
        s = self.surfaces[i]
        if not self.crystal(s.medium_pos if side == POSITIVE else s.medium_neg):
            return 0.0
        found = self.side_range(i, side)
        if found is None:
            return _HALF_PI
        low, high = found
        return float(atan2(high * high - low * low, 2.0 * low * high))

    def refracted(self, leg: _Leg, c: int, axis, from_side: int, to_side: int) -> _Leg:
        """``_snell`` through the plane surface ``c`` from ``from_side`` into ``to_side``; a crystal's face over its index range:
        the wave normal refracts with an index between its ordinary and extraordinary ones (the refracted direction moves
        monotonically with the ratio of the indices, so the cones at the ratio's two ends hold every one between), and a
        crystal's ray walks off its wave normal by at most ``walk_off`` on either side."""
        if not self.crystal_face(c):
            return _snell(leg, axis, self.side_index(c, from_side), self.side_index(c, to_side), to_side)  # type: ignore[arg-type]
        r1, r2 = self.side_range(c, from_side), self.side_range(c, to_side)
        if r1 is None or r2 is None or leg.wide:
            return _Leg(leg.u, _HALF_PI, 0.0, 1.0, to_side)
        start = _cone(leg.u, leg.reach + self.walk_off(c, from_side), leg.side) if self.walk_off(c, from_side) else leg
        first, second = _snell(start, axis, r1[1], r2[0], to_side), _snell(start, axis, r1[0], r2[1], to_side)
        if first.wide or second.wide:
            return _Leg(leg.u, _HALF_PI, 0.0, 1.0, to_side)
        reach = max(first.reach, second.reach) + _angle(first.u, second.u) + self.walk_off(c, to_side)
        return _cone(first.u, reach, to_side)

    def side_index(self, i: int, side: int) -> float | None:
        """The index of the medium on ``side`` of surface ``i`` (the larger of the two for ``EITHER``; ``None``: not known)."""
        s = self.surfaces[i]
        if side == POSITIVE:
            return self.index(s.medium_pos)
        if side == NEGATIVE:
            return self.index(s.medium_neg)
        a, b = self.index(s.medium_pos), self.index(s.medium_neg)
        return None if a is None or b is None else max(a, b)

    def leg(self, start: int, end: int) -> _Leg:
        key = (start, end)
        if key not in self._legs:
            self._legs[key] = _leg(self.faces[start], self.faces[end])
        return self._legs[key]

    def toward(self, e: int, t: int) -> int:
        """The side of surface ``e`` that surface ``t`` lies on (``EITHER``: their extents across ``e`` overlap)."""
        key = (e, t)
        side = self._toward.get(key)
        if side is None:
            side = self._toward[key] = _side_from(self.faces[e], *_extent(self.faces[t], self.faces[e].axis))
        return side

    def arrives(self, n_light: float | None, t: int, side: int) -> bool:
        """Whether light in a medium of index ``n_light`` can meet surface ``t`` from ``side``: an interface takes only light
        in its own medium on that side (the trace ends any other there, ``MEDIUM_MISMATCH``), indices compared within
        ``INDEX_TOL`` (either side for ``EITHER``; any light where an index is not known); any other surface takes all."""
        return self.arrival_index(n_light, t, side) is not False

    def arrival_index(self, n_light: float | None, t: int, side: int) -> float | None | bool:
        """The index of the medium light in a medium of index ``n_light`` (``None``: not known) meets surface ``t`` in from
        ``side``: at an interface the index of its own medium on a side that holds that light (the larger of the two
        sides for ``EITHER`` and light of unknown medium; ``False``: no side does, the light does not meet it), at any
        other surface the light's own (it keeps its medium there; the surface's side where not known). ``None``: not
        known."""
        sides = _sides(side)
        if self.kinds[t] != "refract":
            if n_light is not None:
                return n_light
            return self.side_index(t, side)
        found: float | None | bool = False
        for s_ in sides:
            n = self.side_index(t, s_)
            if n is None:
                return None
            if n_light is None or abs(n - n_light) <= INDEX_TOL:
                found = n if found is False or found is None else max(found, n)  # type: ignore[operator]
        return found

    def emitted(self, e: int, t: int) -> float | None:
        """The index of the medium light from surface ``e`` toward surface ``t`` starts in: ``e``'s on the side ``t`` lies
        on (``None``: either side, or not known)."""
        toward = self.toward(e, t)
        return None if toward == EITHER else self.side_index(e, toward)

    def met_once(self, c: int) -> bool:
        """Whether surface ``c`` is one the open paths meet at most once between two reflections: a thin lens, or a
        refracting surface between media of known, different index (one between media of unknown index counts per leg)."""
        kind = self.kinds[c]
        if kind in BENDING:
            return True
        if kind != "refract":
            return False
        s = self.surfaces[c]
        if self.crystal_face(c):
            return False  # (a crystal's waves meet its faces at indices between its two: counted per leg, as an unknown index)
        n_pos, n_neg = self.index(s.medium_pos), self.index(s.medium_neg)
        return n_pos is not None and n_neg is not None and n_pos != n_neg

    def turns(self, c: int, axis=None) -> bool:
        """Whether surface ``c`` may turn light it passes: a thin lens, or a refracting surface between media of different
        index -- unless it is a plane parallel to ``axis`` (between parallel planes its refraction keeps the light's
        momentum along them)."""
        kind = self.kinds[c]
        if kind in BENDING:
            return True
        if kind != "refract":
            return False
        s = self.surfaces[c]
        if self.crystal_face(c):
            return True  # (a crystal's face splits light into waves of different indices, the extraordinary one walking off)
        n_pos, n_neg = self.index(s.medium_pos), self.index(s.medium_neg)
        if n_pos is not None and n_neg is not None and n_pos == n_neg:
            return False
        face = self.faces[c]
        return not (axis is not None and face.plane and abs(_dot(face.axis, axis)) >= 1.0 - PARALLEL_TOL)

    def relays(self, i: int, j: int, between: list[int]) -> list[int]:
        """The surfaces off the hull of surfaces ``i`` and ``j`` through which light may still go from one of them to the
        other: every thin lens or refracting surface that two of ``i``, ``j``, the surfaces ``between`` them and the surfaces
        already found lie on opposite sides of, or one of them straddles (``toward``), found again and again until no other
        is (a chain of them turning the light off the line between the two)."""
        if not self.turning:
            return []
        if self._side_matrix is None:
            self._side_matrix = np.array([[self.toward(c, u) for u in range(len(self.faces))] for c in self.turning], dtype=np.int64)
        turning = np.array(self.turning, dtype=np.int64)
        known = np.zeros(len(self.faces), dtype=bool)
        known[[i, j, *between]] = True
        open_ = ~known[turning]
        found: list[int] = []
        while bool(np.any(open_)):
            sides = self._side_matrix[:, known]
            hit = open_ & (np.any(sides == EITHER, axis=1) | (np.any(sides == POSITIVE, axis=1) & np.any(sides == NEGATIVE, axis=1)))
            if not bool(np.any(hit)):
                break
            added = turning[hit]
            found += [int(c) for c in added]
            known[added] = True
            open_ &= ~hit
        return found

    def passes(self, start: int, c: int, end: int) -> bool:
        """Whether light going from surface ``start`` to the turning surface ``c`` may go on through it to ``end``: within a
        thin lens's largest deflection, or within the largest deviation of a refraction, ``acos(n_min / n_max)`` (from the
        denser side at the critical angle, or into it at grazing incidence; any direction where an index is not known),
        between the legs' cones."""
        into, out = self.leg(start, c), self.leg(c, end)
        if into.wide or out.wide:
            return True
        if self.kinds[c] in BENDING:
            turn = self.bends[c]
        else:
            r_a, r_b = self.side_range(c, POSITIVE), self.side_range(c, NEGATIVE)
            if r_a is None or r_b is None:
                return True
            low, high = min(r_a[0], r_b[0]), max(r_a[1], r_b[1])
            turn = _HALF_PI - _asin(low / high)  # (acos(n_min / n_max))
        return _angle(into.u, out.u) <= into.reach + turn + out.reach

    def chained(self, i: int, j: int, relays: list[int]) -> bool:
        """Whether light may go from surface ``i`` to surface ``j`` through a chain of the turning surfaces ``relays``
        (each step within ``passes``)."""
        frontier = [(i, c) for c in relays]
        seen = set(frontier)
        while frontier:
            step = []
            for previous, c in frontier:
                if self.passes(previous, c, j):
                    return True
                for nxt in relays:
                    if nxt != c and (c, nxt) not in seen and self.passes(previous, c, nxt):
                        seen.add((c, nxt))
                        step.append((c, nxt))
            frontier = step
        return False

    def onward(self, start: int, c: int, end: int) -> bool:
        """Whether light going from ``start`` to the turning surface ``c`` can go on from it to ``end``: through a thin lens
        within its largest deflection; through a refracting surface in any direction."""
        if self.kinds[c] not in BENDING:
            return True
        into, out = self.leg(start, c), self.leg(c, end)
        if into.wide or out.wide:
            return True
        return _angle(into.u, out.u) <= into.reach + self.bends[c] + out.reach

    def passed(self, src: int, c: int) -> _Leg:
        """The cone light from surface ``src`` leaves the turning surface ``c`` in, passed on (its transmission; a
        reflection there would be a third reflecting surface): through a thin lens within the lens's largest deflection,
        through a plane interface by ``_snell``, through a curved one by ``_snell_curved``; every direction where the light
        may arrive from either side or an index is not known."""
        key = (src, c)
        found = self._passed.get(key)
        if found is not None:
            return found
        leg, face = self.leg(src, c), self.faces[c]
        if leg.side == EITHER or leg.wide:
            found = _Leg(leg.u, _HALF_PI, 0.0, 1.0, EITHER)
        elif self.kinds[c] in BENDING:
            found = _cone(leg.u, leg.reach + self.bends[c], _other(leg.side))
        else:
            n1, n2 = self.side_index(c, leg.side), self.side_index(c, _other(leg.side))
            if n1 is None or n2 is None or (self.crystal_face(c) and not face.plane):
                found = _Leg(leg.u, _HALF_PI, 0.0, 1.0, _other(leg.side))
            elif face.plane:
                found = self.refracted(leg, c, face.axis, leg.side, _other(leg.side))
            else:
                found = _snell_curved(leg, face, n1, n2, _other(leg.side))
        self._passed[key] = found
        return found

    # -------------------------------------------------------------------------------------------- parallel planes

    def members(self, i: int) -> list[int]:
        root = self.classes[i]
        return [j for j, r in self.classes.items() if r == root]

    def images(self, e: int) -> list[_Leg]:
        """The cones light leaves the plane ``e`` in, from every leg into it (from every other surface that redirects
        light, and every source's beam that reaches it): its reflection's mirror image, and what it passes on -- straight
        on through a thin element, refracted through an interface (``_snell``, with the plane's indices: an interface takes
        only light in its own medium) --, each with the side it leaves into."""
        if e in self._images:
            return self._images[e]
        face = self.faces[e]
        axis = face.axis
        incoming = [self.leg(f, e) for f in self.redirecting
                    if f != e and self.arrives(self.emitted(f, e), e, self.leg(f, e).side)]
        for k, (ball, beam) in enumerate(self.beams):
            geometric = _leg(ball, face)
            if _overlap(beam, geometric) and self.arrives(self.index(self.scene.sources[k].medium), e, geometric.side):
                incoming.append(_Leg(beam.u, beam.reach, beam.cos, beam.sin, geometric.side))
        out: list[_Leg] = []
        refracts = self.kinds[e] == "refract"
        for leg in incoming:
            for side in _sides(leg.side):
                away = _other(side)
                out.append(_Leg(_reflected(leg.u, axis), leg.reach, leg.cos, leg.sin, side))
                if not refracts:
                    out.append(_Leg(leg.u, leg.reach, leg.cos, leg.sin, away))
                    continue
                n1, n2 = self.side_index(e, side), self.side_index(e, away)
                out.append(_Leg(leg.u, _HALF_PI, 0.0, 1.0, away) if n1 is None or n2 is None
                           else self.refracted(leg, e, axis, side, away))
        self._images[e] = out
        return out

    def momenta_in(self, leg: _Leg, t: int, side: int, n: float) -> tuple[float, float]:
        """``_momenta`` of a leg meeting surface ``t`` from ``side`` in a medium of index ``n``; where a crystal lies on that side,
        over its index range (its waves meet any index between its ordinary and extraordinary ones)."""
        if not self.crystal_face(t):
            return _momenta(leg, self.faces[t].axis, n)
        found = self.side_range(t, side)
        if found is None:
            return 0.0, math.inf
        low, high = _momenta(leg, self.faces[t].axis, found[0]), _momenta(leg, self.faces[t].axis, found[1])
        return min(low[0], high[0]), max(low[1], high[1])

    def entries(self, t: int, members: set[int]) -> tuple[float, float] | None:
        """The least and the largest momentum along the plane ``t``, ``n sin(angle of incidence)``, of the light that enters
        its class of parallel planes (``members``) at it: along the leg from every surface outside the class that redirects
        light -- from a plane within the images of the legs into it (``images``), from any other surface within the leg
        itself -- and within every source's beam that reaches it; ``n`` the index of the plane's medium on the side the leg
        arrives from (the larger of the two where it may arrive from either; a refracting surface takes only light in its
        own medium, the trace ends any other there) (``None``: none; inf: an index not known)."""
        face = self.faces[t]
        low, high, found = math.inf, 0.0, False

        def add(k: tuple[float, float]) -> None:
            nonlocal low, high, found
            if k[0] <= k[1]:
                low, high, found = min(low, k[0]), max(high, k[1]), True

        for e in self.redirecting:
            if e == t or e in members:
                continue
            geometric = self.leg(e, t)
            n = self.arrival_index(self.emitted(e, t), t, geometric.side)
            if n is False:
                continue  # (its light crosses an interface on the way: that interface's leg brings it)
            if n is None:
                add((0.0, math.inf))
                continue
            if self.faces[e].plane and self.kinds[e] in REFLECTING:
                toward = self.toward(e, t)
                for image in self.images(e):
                    if toward != EITHER and image.side != EITHER and image.side != toward:
                        continue
                    if _overlap(image, geometric):
                        a, b = self.momenta_in(image, t, geometric.side, n), self.momenta_in(geometric, t, geometric.side, n)
                        add((max(a[0], b[0]), min(a[1], b[1])))
            else:
                add(self.momenta_in(geometric, t, geometric.side, n))
        for k, (ball, beam) in enumerate(self.beams):
            geometric = _leg(ball, face)
            if not _overlap(beam, geometric):
                continue
            n = self.arrival_index(self.index(self.scene.sources[k].medium), t, geometric.side)
            if n is not False:
                add((0.0, math.inf) if n is None else self.momenta_in(beam, t, geometric.side, n))
        return (low, high) if found else None

    def bounds(self, a: int, b: int, members: list[int]) -> tuple[float, float, float, float]:
        key = (a, b)
        if key not in self._bounds:
            self._bounds[key] = self._bounds_of(a, b, members)
        return self._bounds[key]

    def _bounds_of(self, a: int, b: int, members: list[int]) -> tuple[float, float, float, float]:
        """For two planes of one class: (the gap between them along the normal, the offset of their centres along the
        planes, the largest and the least index of the media light may cross between them -- on their sides toward each
        other and on both sides of every refracting plane of the class between them; inf and 0 where one is not known)."""
        axis = self.faces[a].axis
        ca, cb = self.faces[a].centre, self.faces[b].centre
        da, db = _dot(ca, axis), _dot(cb, axis)
        w = (cb[0] - ca[0], cb[1] - ca[1], cb[2] - ca[2])
        along = db - da
        offset2 = _dot(w, w) - along * along
        offset = math.sqrt(offset2) if offset2 > 0.0 else 0.0
        indices = [self.side_index(a, self.toward(a, b)), self.side_index(b, self.toward(b, a))]
        low, high = min(da, db), max(da, db)
        for c in members:
            if c in (a, b) or self.kinds[c] != "refract":
                continue
            depth = _dot(self.faces[c].centre, axis)
            if low < depth < high:
                indices += [self.side_index(c, POSITIVE), self.side_index(c, NEGATIVE)]
        known = [n for n in indices if n is not None]
        if len(known) < len(indices):
            return abs(along), offset, math.inf, 0.0
        between = [c for c in members if c not in (a, b) and self.kinds[c] == "refract" and low < _dot(self.faces[c].centre, axis) < high]
        crystal = [c for c in (a, b, *between) if self.crystal_face(c)]
        if crystal:
            # (a crystal: its waves meet any index between its two, and its rays walk off their wave normals by up to its
            # walk-off angle, ``walks``)
            sided = [(a, self.toward(a, b)), (b, self.toward(b, a))] + [(c, x) for c in between for x in (POSITIVE, NEGATIVE)]
            ranges = [self.side_range(c, x) for c, x in sided]
            if any(r is None for r in ranges):
                return abs(along), offset, math.inf, 0.0
            self._walks[(a, b)] = max(self.walk_off(c, x) for c in crystal for x in (POSITIVE, NEGATIVE))
            return abs(along), offset, max(r[1] for r in ranges), min(r[0] for r in ranges)  # type: ignore[index]
        return abs(along), offset, max(known), min(known)

    def walks(self, a: int, b: int) -> float:
        """The largest walk-off angle (rad) of a crystal on or between planes ``a`` and ``b`` of one class (after ``bounds``;
        0 without one): a ray's angle to the normal is within it of its wave normal's."""
        return self._walks.get((a, b), 0.0)

    def momenta(self, i: int) -> dict[int, tuple[float, float]]:
        """Per plane of the class of plane ``i``: the least and the largest momentum along the planes of the light that can
        meet it -- what enters at it (``entries``), and what reaches it from another plane of the class, which keeps its
        momentum: from one plane's aperture to the other's it moves along them by ``s(k) = sum t_m k / sqrt(n_m^2 - k^2)``
        over the media between, which must lie within the offset of their centres plus or minus both radii (with the gap
        ``t`` and the largest index, ``s`` is at least ``t k / sqrt(n_max^2 - k^2)``; with the least, at most ``t k /
        sqrt(n_min^2 - k^2)``)."""
        root = self.classes[i]
        if root in self._momenta:
            return self._momenta[root]
        members = self.members(i)
        member_set = set(members)
        if self.work is not None:
            # (its work: per plane of the class, the leg from every redirecting surface outside it, or that surface's images)
            tests = 0
            for t in members:
                for e in self.redirecting:
                    if e != t and e not in member_set:
                        tests += len(self.images(e)) if self.faces[e].plane and self.kinds[e] in REFLECTING else 1
            self.work.spend(momentum_work(tests), self.stage)
        sets: dict[int, tuple[float, float]] = {}
        for t in members:
            found = self.entries(t, member_set)
            if found is not None:
                sets[t] = found
        windows: dict[tuple[int, int], tuple[float, float]] = {}
        for p in members:
            for t in members:
                if p == t:
                    continue
                gap, offset, n_high, n_low = self.bounds(p, t, members)
                if gap <= 0.0:
                    continue  # (coplanar: no light goes from one to the other between them)
                rp, rt = self.faces[p].radius, self.faces[t].radius
                if not (math.isfinite(rp) and math.isfinite(rt)):
                    windows[(p, t)] = (0.0, math.inf)
                    continue
                far, near = offset + rp + rt, offset - rp - rt
                walk_off = self.walks(p, t)
                if walk_off > 0.0:
                    # (a ray within ``walk_off`` of its wave normal: the wave normal's angle within that of the ray's)
                    top = math.inf if not math.isfinite(n_high) else n_high * _sin_at_most(float(atan2(far, gap)) + walk_off)
                    bottom = (0.0 if near <= 0.0 or n_low <= 0.0
                              else n_low * _sin_at_least(float(atan2(near, gap)) - walk_off))
                else:
                    top = math.inf if not math.isfinite(n_high) else n_high * far / math.sqrt(gap * gap + far * far)
                    bottom = 0.0 if near <= 0.0 or n_low <= 0.0 else n_low * near / math.sqrt(gap * gap + near * near)
                windows[(p, t)] = (bottom, top)
        settled = False
        for _ in range(MOMENTUM_ROUNDS * len(members)):
            changed = False
            for (p, t), (bottom, top) in windows.items():
                if p not in sets:
                    continue
                low, high = max(sets[p][0], bottom), min(sets[p][1], top)
                if low > high:
                    continue
                if t not in sets:
                    sets[t] = (low, high)
                    changed = True
                elif low < sets[t][0] or high > sets[t][1]:
                    sets[t] = (min(low, sets[t][0]), max(high, sets[t][1]))
                    changed = True
            if not changed:
                settled = True
                break
        if not settled:
            sets = {t: (0.0, math.inf) for t in members}  # (not settled: every momentum)
        self._momenta[root] = sets
        return sets


#: Pieces the range of momenta of a pair of parallel planes is cut into for the bound of its returns (``_parallel_returns``).
MOMENTUM_PIECES = 32


def _angles(ks: list[float], n: float | None) -> list[float]:
    """The angle of incidence (rad) of light of momentum ``k`` along the planes in a medium of index ``n``, for every
    ``k``: pi / 2 where ``k`` is not finite or at least ``n`` (or ``n`` not known), else ``asin(k / n)`` (``_asin``),
    elementwise."""
    if n is None:
        return [_HALF_PI] * len(ks)
    k = np.array(ks, dtype=np.float64)
    x = k / n
    x = np.where(0.0 > x, 0.0, x)
    x = np.where(1.0 < x, 1.0, x)
    a = atan2(x, np.sqrt((1.0 - x) * (1.0 + x)))
    return np.where(np.isfinite(k) & (k < n), a, _HALF_PI).tolist()


def _sin_at_most(angle: float) -> float:
    """``sin(min(angle, pi / 2))`` (libm-free)."""
    return 1.0 if angle >= _HALF_PI else _cos_sin(angle)[1]


def _sin_at_least(angle: float) -> float:
    """``sin(max(angle, 0))`` (libm-free)."""
    return 0.0 if angle <= 0.0 else _cos_sin(angle)[1]


def _plane_indices(sc: _Scene, i: int, side: int) -> tuple[float | None, float | None]:
    """(largest, least) index on ``side`` of plane ``i``: both its index for an isotropic medium, a crystal's two indices
    (``None``: not known)."""
    if not sc.crystal_face(i):
        n = sc.side_index(i, side)
        return n, n
    found = sc.side_range(i, side)
    return (None, None) if found is None else (found[1], found[0])


def _parallel_returns(sc: _Scene, i: int, j: int, side_i: int,
                      side_j: int) -> tuple[float, float, tuple[tuple[float, float], tuple[float, float]]] | None:
    """For light going back and forth between the parallel planes ``i`` and ``j`` (``None``: none can): (the bound of all
    its returns over one crossing, ``e``; the largest round-trip gain ``g`` over its momenta; the angles of incidence, rad,
    at which it meets each plane from its side toward the other). The light keeps its momentum along the planes, ``k = n
    sin(angle)``, which it brings from the rest of the scene and must have at both planes (``_Scene.momenta``). Going from
    one plane to the other it moves along them by its walk ``s(k) = sum t_m k / sqrt(n_m^2 - k^2)`` (at least ``t k /
    sqrt(n_max^2 - k^2)`` with the gap ``t`` and the largest index of the media between): it reaches the other aperture
    only when ``s`` is at least the offset of their centres less both radii, and each round trip moves its hits on each
    plane by ``2 s`` in one direction, its hits on the two planes alternating. On apertures of radii ``rho_1 >= rho_2``
    (a chord of each at most its diameter) a ray's hits on a plane of radius ``rho`` span at most ``2 rho``, and each
    return onto one plane needs a hit on the other one before it: it comes back onto either plane at most ``M(k) =
    min(floor(rho_1 / s), floor(rho_2 / s) + 1)`` times. Its returns bring at most ``sum_{m <= M} g^m <= min(M g, g / (1 -
    g))``: over the momenta cut into ``MOMENTUM_PIECES``, on each piece with ``M`` at its lower end and ``g`` (the
    reflectances over its angles) at its upper one, the largest of those -- finite also where a face reflects everything
    at the top (light that entered the planes at grazing incidence meets their inner sides at the critical angle and
    walks off)."""
    sets = sc.momenta(i)
    if i not in sets or j not in sets:
        return None
    k_low, k_high = max(sets[i][0], sets[j][0]), min(sets[i][1], sets[j][1])
    if k_low > k_high:
        return None
    gap, offset, n_high, n_low = sc.bounds(i, j, sc.members(i))
    ra, rb = sc.faces[i].radius, sc.faces[j].radius
    small, large = min(ra, rb), max(ra, rb)
    walking = gap > 0.0 and math.isfinite(small) and math.isfinite(n_high)
    if gap > 0.0 and math.isfinite(ra) and math.isfinite(rb):
        # (back onto an aperture: a walk of at most the larger radius per round trip; from one aperture to the other: at
        # least the offset of their centres less both radii)
        need = offset - ra - rb
        walk_off = sc.walks(i, j)
        if walk_off > 0.0:
            if math.isfinite(n_high):
                k_high = min(k_high, n_high * _sin_at_most(float(atan2(large, gap)) + walk_off))
            if need > 0.0 and n_low > 0.0:
                k_low = max(k_low, n_low * _sin_at_least(float(atan2(need, gap)) - walk_off))
        else:
            if math.isfinite(n_high):
                k_high = min(k_high, n_high * large / math.sqrt(gap * gap + large * large))
            if need > 0.0 and n_low > 0.0:
                k_low = max(k_low, n_low * need / math.sqrt(gap * gap + need * need))
        if k_low > k_high:
            return None
    # (the angles of momentum k at a plane: the least at its side's largest index, the largest at its least -- a crystal's two)
    hi_i, lo_i = _plane_indices(sc, i, side_i)
    hi_j, lo_j = _plane_indices(sc, j, side_j)

    def angle(k: float, n: float | None) -> float:
        if n is None or not math.isfinite(k) or k >= n:
            return _HALF_PI
        return _asin(k / n)

    def gain(k0: float, k1: float) -> float:
        r_i = sc.coefficients(i, side_i, angle(k0, hi_i), angle(k1, lo_i))[0]
        r_j = sc.coefficients(j, side_j, angle(k0, hi_j), angle(k1, lo_j))[0]
        return r_i * r_j

    angles = ((angle(k_low, hi_i), angle(k_high, lo_i)), (angle(k_low, hi_j), angle(k_high, lo_j)))
    if not math.isfinite(k_high):
        g = gain(k_low, k_high)
        return (math.inf if g >= 1.0 else g / (1.0 - g)), g, angles
    pieces = MOMENTUM_PIECES if walking and k_high > k_low else 1
    # (the pieces' ends: k_low + (k_high - k_low) m / pieces, the last one k_high, and their angles at either plane)
    ks = (k_low + (k_high - k_low) * np.arange(pieces, dtype=np.float64) / pieces).tolist() + [k_high]
    a_i, a_j = _angles(ks, hi_i), _angles(ks, hi_j)
    b_i, b_j = (a_i if lo_i == hi_i else _angles(ks, lo_i)), (a_j if lo_j == hi_j else _angles(ks, lo_j))
    excess, largest = 0.0, 0.0
    for m in range(pieces):
        k0 = ks[m]
        g = sc.coefficients(i, side_i, a_i[m], b_i[m + 1])[0] * sc.coefficients(j, side_j, a_j[m], b_j[m + 1])[0]
        largest = max(largest, g)
        bound = math.inf if g >= 1.0 else g / (1.0 - g)
        walk = 0.0
        if walking and k0 > 0.0 and sc.walks(i, j) > 0.0:
            # (a crystal: the ray's angle to the normal at least the wave normal's less the walk-off, its walk at least that)
            least = _asin(k0 / n_high) - sc.walks(i, j)
            if least > 0.0:
                c, s_ = _cos_sin(least)
                walk = gap * s_ / c
        elif walking and k0 > 0.0:
            walk = gap * k0 / math.sqrt((n_high - k0) * (n_high + k0))
        if walk > 0.0:
            most = math.floor(small / walk) + 1.0  # (each return onto one plane needs a hit on the other one between)
            if math.isfinite(large):
                most = min(most, math.floor(large / walk))
            bound = min(bound, most * g)
        excess = max(excess, bound)
    return excess, largest, angles


def _general_angles(sc: _Scene, i: int, j: int,
                    turning: list[int]) -> tuple[tuple[float, float, float, set[int]], tuple[float, float, float, set[int]]] | None:
    """For a pair met along legs from each other and through the surfaces between them that turn light: whether each
    sends light arriving that way back toward the other (directly or through a turning surface), and per surface the
    reflectance bound, the angles of incidence (rad) and the sides its light arrives at (``None``: they do not face each
    other). Light of the pair that reaches a surface through a turning surface comes to it from the other surface or from
    another turning one: it travels within the cones that surface passes those legs on in (``_Scene.passed``) and within
    the leg from it. Light that reaches a plane through planes parallel to it keeps its momentum along them: where the
    turning surfaces include planes of the member's own class, the light they bring meets it at the angles of that class's
    momenta there (``_Scene.momenta``), and for whether the member sends light back they pass it straight on."""
    found = []
    for me, other in ((i, j), (j, i)):
        face = sc.faces[me]
        own = sc.classes.get(me)
        alike = [c for c in turning if own is not None and sc.classes.get(c) == own]
        bending = [c for c in turning if c not in alike]
        # the pair's light at this surface: straight from the other one, or passed on by a turning surface (from the other
        # one or another turning surface), within that surface's leg to this one
        cones, lasts = sc.light_into(me, other, bending)
        outgoing = [(sc.leg(me, other), None)] + [(sc.leg(me, c), c) for c in bending]
        # (of the two cones the light lies in, the narrower)
        arriving = _Legs.choose(cones.wide | (~lasts.wide & (lasts.reach <= cones.reach)), lasts, cones)
        ways = [away for away, c in outgoing if c is None or sc.onward(me, c, other)]
        if not _any_meets(arriving, face, ways):
            return None
        found.append((me, face, alike, cones, lasts))
    out = []
    for me, face, alike, cones, lasts in found:
        r, low, high, sides = 0.0, _HALF_PI, 0.0, set()
        lo_a, hi_a = _incidences(cones, face)
        lo_b, hi_b = _incidences(lasts, face)
        lo, hi = np.where(lo_b > lo_a, lo_b, lo_a), np.where(hi_b < hi_a, hi_b, hi_a)
        keep = ~(lo > hi)
        if bool(np.any(keep)):
            kept_side, kept_lo, kept_hi = lasts.side[keep].tolist(), lo[keep], hi[keep]
            for key in set(zip(kept_side, kept_lo.tolist(), kept_hi.tolist())):
                r = max(r, sc.coefficients(me, *key)[0])
            low, high = min(low, float(np.min(kept_lo))), max(high, float(np.max(kept_hi)))
            sides.update(kept_side)
        if alike:
            momenta = sc.momenta(me).get(me)
            for side in sorted({sc.leg(c, me).side for c in alike}):
                for s_ in _sides(side):
                    n, n_least = _plane_indices(sc, me, s_)
                    if momenta is None:
                        continue
                    if n is None or n_least is None or not math.isfinite(momenta[1]):
                        lo, hi = 0.0, _HALF_PI
                    else:
                        lo = _HALF_PI if momenta[0] >= n else _asin(momenta[0] / n)
                        hi = _HALF_PI if momenta[1] >= n_least else _asin(momenta[1] / n_least)
                    r = max(r, sc.coefficients(me, s_, lo, hi)[0])
                    low, high = min(low, lo), max(high, hi)
                    sides.add(s_)
        if not sides:
            return None
        out.append((r, low, high, sides))
    return (out[0][0], out[0][1], out[0][2], out[0][3]), (out[1][0], out[1][1], out[1][2], out[1][3])


def _pairs(sc: _Scene) -> tuple[float, list[ReturnPair]]:
    """The summed bound ``sum g / (1 - g)`` of the returns of every pair over one crossing (``inf``: a pair bounds
    nothing), and the pairs (module docstring). ``sc.work``: the analysis' work budget, ``PAIR_SCREEN_WORK`` units per pair
    examined and ``pair_cone_work`` for its light through the surfaces that turn it; one that runs out raises
    ``OutOfReturnWork``."""
    counter = sc.work
    faces, surfaces = sc.faces, sc.surfaces
    ids = [s.surface_id for s in surfaces]
    reflecting = [i for i, kind in enumerate(sc.kinds) if kind in REFLECTING]
    # (a surface that reflects nothing at any angle on either side pairs with nothing)
    reflects = {i for i in reflecting if sc.coefficients(i, EITHER, 0.0, _HALF_PI)[0] > 0.0}
    excess = 0.0
    found: list[ReturnPair] = []

    for k, i in enumerate(reflecting):
        if i not in reflects:
            continue
        for j in reflecting[k + 1:]:
            if j not in reflects:
                continue
            if counter is not None:
                counter.spend(PAIR_SCREEN_WORK, PAIR_STAGE)
            fa, fb = faces[i], faces[j]
            crossed = [int(c) for c in np.flatnonzero(sc.table.crossing(fa, fb)) if int(c) not in (i, j)]
            parallel = fa.plane and fb.plane and abs(_dot(fa.axis, fb.axis)) >= 1.0 - PARALLEL_TOL
            side_i = _side_from(fa, *_extent(fb, fa.axis))
            side_j = _side_from(fb, *_extent(fa, fb.axis))
            turning = [c for c in crossed if sc.turns(c)]
            relays = sc.relays(i, j, crossed)
            # (off their hull: a refracting plane parallel to them keeps the light's momentum along them, and the walk counts
            # its medium; any other lens or refracting surface turns the light and shifts it sideways, so where a chain of
            # the surfaces around them may pass light from one plane to the other the walk-off bound does not hold)
            turning_off = [c for c in relays if sc.turns(c, fa.axis)]
            if (parallel and side_i != EITHER and side_j != EITHER
                    and not any(sc.turns(c, fa.axis) for c in turning)
                    and not (turning_off and sc.chained(i, j, turning + relays))):
                # (two parallel planes, nothing between them turning light but planes parallel to them)
                returns = _parallel_returns(sc, i, j, side_i, side_j)
                if returns is None:
                    continue
                value, gain, ((lo_i, hi_i), (lo_j, hi_j)) = returns
                at_i, at_j = side_i, side_j
                # (the centre-line incidence between parallel planes: the least angle their light can have)
                psi_i, psi_j = lo_i, lo_j
            else:
                # (the surfaces between them that turn light, those off their hull included; the light through them is
                # work of its own: pair_cone_work)
                if counter is not None:
                    counter.spend(pair_cone_work(len(turning) + len(relays)), PAIR_STAGE)
                general = _general_angles(sc, i, j, turning + relays)
                if general is None:
                    continue
                crossed += relays
                (r_i, lo_i, hi_i, sides_i), (r_j, lo_j, hi_j, sides_j) = general
                gain = r_i * r_j
                value = math.inf if gain >= 1.0 else gain / (1.0 - gain)
                at_i = next(iter(sides_i)) if len(sides_i) == 1 else EITHER
                at_j = next(iter(sides_j)) if len(sides_j) == 1 else EITHER
                psi_i, psi_j = _centre_incidence(sc.leg(j, i), fa), _centre_incidence(sc.leg(i, j), fb)
            if gain <= 0.0 or value <= 0.0:
                continue
            excess = excess + value
            extra: dict = {}
            if not math.isfinite(value):
                # (a pair that bounds nothing: why each bound is 1, and the pair at its centre-line incidence -- read for
                # these pairs only, from the surfaces' own records: no factor, count or work of the screen moves)
                extra["saturation"] = (
                    _saturation(surfaces[i], at_i, lo_i, hi_i, sc.library, sc.wavelength_nm, sc.uses, sc.refusing),
                    _saturation(surfaces[j], at_j, lo_j, hi_j, sc.library, sc.wavelength_nm, sc.uses, sc.refusing))
                r0_i = _nominal_reflectance(surfaces[i], at_i, psi_i, sc.library, sc.wavelength_nm, sc.uses)
                r0_j = _nominal_reflectance(surfaces[j], at_j, psi_j, sc.library, sc.wavelength_nm, sc.uses)
                if r0_i is not None and r0_j is not None and r0_i * r0_j < 1.0:
                    g0 = r0_i * r0_j
                    extra.update(nominal_angles_deg=(psi_i * _DEGREES, psi_j * _DEGREES), nominal_gain=g0,
                                 nominal_returns=g0 / (1.0 - g0))
            found.append(ReturnPair((ids[i], ids[j]), gain, ((lo_i * _DEGREES, hi_i * _DEGREES), (lo_j * _DEGREES, hi_j * _DEGREES)),
                                    tuple(ids[c] for c in crossed), value, **extra))
    return excess, found


def _centre_incidence(leg: _Leg, face: _Face) -> float:
    """The angle of incidence (rad) at which the line of ``leg`` (between two surfaces' centres) meets ``face``'s axis."""
    phi = _angle(leg.u, face.axis)
    return phi if phi <= _HALF_PI else _PI - phi


def return_gains(scene: Scene, library: MaterialLibrary, wavelengths: Iterable[float],
                 uses: dict[float, dict] | None = None) -> dict[float, ReturnGains]:
    """Per vacuum wavelength, the factor of the left-out light's field on every surface and observation plane, from either
    side: 1 plus the bounds of the returns of every pair (module docstring; none where no pair is found), and the surfaces
    that may reflect it on its open paths, the legs between two reflections, the surfaces met at most once between them and
    the others' meetings per leg, with the sum of the reflectance bounds. ``uses``: the coating uses of the left-out light
    (``validation/coating_use.py``): a refused use keeps the record's stated value in all of these and withholds exposure
    upper bounds (``ReturnGains.withheld``). Every wavelength is analysed now (``ReturnAnalysis``: on first read)."""
    analysis = ReturnAnalysis(scene, library, wavelengths, uses)
    return {wavelength: analysis[wavelength] for wavelength in analysis}


class ReturnAnalysis(Mapping[float, ReturnGains]):
    """``return_gains`` as a mapping read on demand: the analysis of a wavelength (its pairs and its closed returns) runs
    the first time a consumer reads that wavelength, and never for a wavelength no consumer reads. Membership, length and
    iteration name the wavelengths without analysing them. ``uses``: the coating uses (a mapping, or a callable that
    returns it, called once on the first analysis). ``work_limit``: the run's work limit (``validation/return_work.py``),
    shared by its wavelengths in the order they are first read -- each analysis gets what the earlier ones left
    (``ReturnWork.before``), so all of them together do at most ``work_limit`` units. Within the limit a wavelength's
    ``ReturnGains`` is the same whenever it is analysed: reading it on demand changes no number."""

    def __init__(self, scene: Scene, library: MaterialLibrary, wavelengths: Iterable[float],
                 uses: dict[float, dict] | Callable[[], dict[float, dict]] | None = None, work_limit: int | None = None) -> None:
        self._scene, self._library, self._work_limit = scene, library, work_limit
        self._wavelengths = tuple(sorted(set(wavelengths)))
        self._uses = uses
        self._done: dict[float, ReturnGains] = {}
        self._spent = 0  # (the units the analyses so far spent of ``work_limit``)
        self._shared: tuple[list[_Face], list[str], list[int]] | None = None

    def __contains__(self, wavelength: object) -> bool:
        return wavelength in self._wavelengths

    def __iter__(self) -> Iterator[float]:
        return iter(self._wavelengths)

    def __len__(self) -> int:
        return len(self._wavelengths)

    def __getitem__(self, wavelength: float) -> ReturnGains:
        if wavelength not in self._wavelengths:
            raise KeyError(wavelength)
        found = self._done.get(wavelength)
        if found is None:
            if callable(self._uses):
                self._uses = self._uses()
            if self._shared is None:
                surfaces = self._scene.surfaces
                # (the most points a straight line meets each surface at: per straight leg)
                self._shared = ([_face(i, s) for i, s in enumerate(surfaces)], [s.surface_id for s in surfaces],
                                [_line_meetings(s) for s in surfaces])
            faces, ids, per_line = self._shared
            left = None if self._work_limit is None else self._work_limit - self._spent
            with phase("return_analysis", total=left, unit="work units"):
                found = _gains_at(self._scene, self._library, wavelength, (self._uses or {}).get(wavelength), faces, ids, per_line,
                                  left, self._spent)
            self._spent += found.work.steps
            self._done[wavelength] = found
        return found

    @property
    def analysed(self) -> tuple[ReturnGains, ...]:
        """The analyses made so far, by ascending wavelength."""
        return tuple(self._done[w] for w in sorted(self._done))


def _gains_at(scene: Scene, library: MaterialLibrary, wavelength: float, here: dict | None, faces: list[_Face],
              ids: list[str], per_line: list[int], work_limit: int | None = None, before: int = 0) -> ReturnGains:
    """The ``ReturnGains`` of one vacuum wavelength (``return_gains``), within ``work_limit`` units (what the run's limit
    left after ``before`` units of its earlier wavelengths)."""
    counter = WorkCounter(work_limit, before)
    sc = _Scene(scene, library, wavelength, faces, here, work=counter)
    try:
        excess, pairs = _pairs(sc)
    except OutOfReturnWork:
        # (the pairs are not known: no factor, no count of the open paths -- every bound that reads them is withheld)
        return ReturnGains(wavelength, work=counter.record(), search=ClosedReturnSearch(termination=NOT_STARTED))
    factor = 1.0 + excess
    by_surface = {sid: (factor, factor) for sid in ids} if excess > 0.0 else {}
    refused = tuple(use for _, use in sorted((here or {}).items()) if use.refused is not None)
    refused_pairs: tuple[ReturnPair, ...] = ()
    if refused:
        stated = {p.surfaces for p in pairs if not math.isfinite(p.returns)}
        try:
            refusing = _pairs(_Scene(scene, library, wavelength, faces, here, refusing=True, work=counter))[1]
        except OutOfReturnWork:
            return ReturnGains(wavelength, work=counter.record(), search=ClosedReturnSearch(termination=NOT_STARTED))
        refused_pairs = tuple(p for p in refusing if not math.isfinite(p.returns) and p.surfaces not in stated)
    withheld = frozenset({use.surface_id for use in refused} | {sid for p in refused_pairs for sid in p.surfaces})
    legs = len(sc.turning) + 1
    # (met at most once between two reflections: every thin lens and every refracting surface between media of known,
    # different index -- the open paths' premise; every other surface per straight leg)
    once = {c for c in sc.turning if sc.met_once(c)}
    per_leg = {ids[i]: per_line[i] for i in range(len(ids)) if i not in once and per_line[i] > 1}
    # (the internal returns of plane-faced solids with a fold: a face's step bound, total internal reflection included, and
    # its closed walks' sums; a solid whose returns are not bounded is a closed return without a series. Its work counts
    # against the same limit: where it reaches the limit the closed returns are not known -- INCOMPLETE, the trail search not
    # started --, while the pairs it follows stay known)
    from .solid_folds import solid_folds
    sc.stage = SOLID_FOLD_STAGE
    try:
        folds = solid_folds(sc, frozenset(frozenset(p.surfaces) for p in pairs if math.isfinite(p.returns)))
        folded = True
    except OutOfReturnWork:
        folds, folded = (), False
    in_solid = {sid: (step, g, g2, a, a2) for fold in folds
                for sid, step, g, g2, a, a2 in zip(fold.surfaces, fold.step, fold.loop, fold.loop_power, fold.alternation,
                                                   fold.alternation_power)}
    reflecting = []
    loops = []
    for i, kind in enumerate(sc.kinds):
        if kind in REFLECTING:
            # (every angle from either side: the left-out light's directions are open; each bound times the most
            # meetings between two reflections)
            r = sc.coefficients(i, EITHER, 0.0, _HALF_PI)[0]
            inside = in_solid.get(ids[i])
            if inside is not None:
                r = max(r, inside[0])
            if r > 0.0:
                reflecting.append((r, 1 if i in once else legs * per_line[i]))
                loops.append((1.0, 1.0, 1.0, 1.0) if inside is None else (inside[1], inside[2], inside[3], inside[4]))
    from .closed_returns import closed_returns
    if folded:
        cycles, series = closed_returns(sc)
        # (a replaced screen that leaves no record of its search: what its answer and the counter say)
        search = getattr(sc, "search", None) or ClosedReturnSearch(
            termination=BUDGET_EXHAUSTED if counter.stage is not None else RETURN_FOUND if cycles else SEARCH_COMPLETED)
    else:
        cycles, series, search = (), None, ClosedReturnSearch(termination=NOT_STARTED)
    # (a fold solid whose internal returns have no certified sum is a closed return with no series: a finished screen is then
    # POTENTIAL_RETURN, never COMPLETED_NO_RETURN, and every bound that reads the open paths is withheld. A screen that
    # reached its work limit stays INCOMPLETE with no cycle -- it already withholds those bounds (``screened``) and names no
    # return -- so the solid joins the cycles only where the screen finished.)
    held = tuple(fold.surfaces for fold in folds if not fold.bounded)
    if held and counter.stage is None:
        cycles, series = tuple(cycles) + held, None
    # (a step an air-equivalent record does not hold for: the face's exposure upper bound is withheld, as for the first run of
    # the left-out light; every factor reads the record's stated value)
    inside_refused = tuple(use for fold in folds for use in fold.refused
                           if (use.surface_id, use.side) not in {(u.surface_id, u.side) for u in refused})
    if inside_refused:
        refused = refused + inside_refused
        withheld = withheld | frozenset(use.surface_id for use in inside_refused)
    # The all-orders series replaces the open-path count. Retaining a
    # finite pair factor is conservative when that series is available;
    # an unsupported series must never fall back to the open-path bound
    # (nor may a screen that ran out of work: ``ReturnGains.screened``).
    return ReturnGains(wavelength, by_surface, tuple(pairs), len(reflecting), _sum_squares(reflecting), legs,
                       frozenset(ids[i] for i in once), per_leg, tuple(reflecting), cycles, series, refused, refused_pairs, withheld,
                       counter.record(), search, loops=tuple(loops) if folds else (), solid_folds=folds)


def _line_meetings(s: SurfaceG) -> int:
    """The most points a straight line that does not run along surface ``s`` meets it at (module docstring, "Meetings per
    leg"): 1 on a plane, 2 on a sphere; on a sag surface the degree of its profile's equation along the line -- ``e = 2p``
    (``p`` the highest power of ``s = rho^2`` with a non-zero asphere coefficient; 1 without one), ``2e`` where
    ``c (1 + k) != 0``, ``max(e, 2)`` on a paraboloid, ``e`` on a flat base -- doubled by an axicon term. A two-variable
    shape: its degree (``freeform.line_meetings``)."""
    if s.geometry == "plane":
        return 1
    if s.geometry == "sphere":
        return 2
    if s.freeform is not None:
        return line_meetings(s)
    highest = 0
    for j, coefficient in enumerate(s.coefficients):
        if coefficient != 0.0:
            highest = j + 2
    e = 2 * highest if highest else 1
    if s.curvature != 0.0 and 1.0 + s.conic != 0.0:
        base = 2 * e
    elif s.curvature != 0.0:
        base = max(e, 2)
    else:
        base = e
    return 2 * base if s.linear != 0.0 else base


def _sum_squares(values: list[tuple[float, int]]) -> float:
    """``sum m v^2`` over ``(v, m)`` in the order given (plain additions: the same bits in every runtime)."""
    total = 0.0
    for v, m in values:
        total += (v * v) * m
    return total


__all__ = ["RETURN_ANGLE_SIGMAS", "ReturnAnalysis", "ReturnGains", "ReturnPair", "return_gains"]
