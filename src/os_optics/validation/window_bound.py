"""A lower bound of a surface's exposure that light of unknown place cannot undo (``validation/exposures.py``).

Light the trace leaves out as negligible and untraced light of unknown size are bounded by their diffraction limit: their
field is at most an amplitude ``A`` anywhere on the surface. Taken at the centre of a traced beam at that amplitude, as if
it were there in antiphase, such light cancels the beam whenever ``A`` exceeds the beam's own amplitude -- in a
multi-element system, where every ghost of unknown size counts with the diffraction limit of its power, almost always.
Yet that light has a small power: it cannot be strong over the whole area the beam covers. This module bounds the
exposure from below by the beam's field averaged against a window over that area.

The window bound. Let ``E`` be the summed field of one class of light (one coherence group and line, time-averaged
fields in irradiance units, ``|E|^2`` the exposure) on the surface, ``E_k`` a traced beam of it, ``psi`` a window on the
surface. For any ``psi``, ``max |E| >= |integral psi* E dA| / integral |psi| dA`` (the largest value is at least the
weighted mean). Choose ``psi = chi e^(i phi_k)`` with ``phi_k`` the beam's own phase and ``chi >= 0``, so that
``integral psi* E_k dA = integral chi |E_k| dA``. Every other part of ``E`` takes away at most:

* another beam of known Gaussian profile, of whatever phase: ``integral chi |E_j| dA`` (closed form);
* a beam beside the surface (its tail bounded on the whole surface): its bound times ``integral chi dA``;
* light of unknown place -- untraced light of unknown size, of power ``P_i``, and the light left out as negligible, whose
  field is at most ``c f`` times one crossing of its in-phase power ``P*`` (``c`` its open paths' crossing field,
  ``f`` its returns between surfaces facing each other) -- writing it as plane waves of amplitudes ``a`` on each side of
  the surface's tangent plane at the beam's centre, ``|integral psi* E_U dA| = |integral a G dk| <= H sqrt(flux)``
  (Cauchy-Schwarz with the weight ``cos``: the flux through the plane is ``(2 pi)^2 integral |a|^2 cos dk``), where
  ``G(k)`` is the overlap of the window with the plane wave ``k`` and ``H^2 = (2 pi)^-2 integral |G|^2 / cos dk`` over
  each side's propagating waves. The flux of light of power ``p`` crossing the plane from one side is at most ``p``
  (the same premise as its diffraction limit), so the light of unknown place takes away at most
  ``H (sum_i sqrt(P_i) + c f sqrt(P*))``, whatever its directions -- grazing light included, since the weight that
  makes the integral finite sits on the window's side.

So ``max |E|^2 >= ((integral chi |E_k| - sum_j integral chi |E_j| - B integral chi - H U) / integral chi)^2`` with
``U = sum_i sqrt(P_i) + c f sqrt(P*)`` and ``B`` the beside beams' summed tail amplitudes; classes of other lines or
groups add to the exposure and are left out (>= 0). Peak fluence: the same with pulse energies. ``P / d`` (``W/cm``):
"P / d" below.

The window. On the tangent plane (coordinates ``x``, unit normal ``n``) the beam's amplitude is
``sqrt(V) exp(-x^T S x)`` (``S`` its footprint matrix: ``x^T S x`` is ``rho^2``, the squared distance from its chief ray
in units of its 1/e^2 radii; ``V`` its peak value) and its phase ``k_T . x`` plus its wavefront's quadratic phase
``x^T C x`` (``|C| <= k_T (|Re 1/q_x| + |Re 1/q_y|) / 2``, ``q`` the beam's complex parameters). The window is
``chi = exp(-s x^T S x)`` over the clear aperture (beyond its edge it is 0), for a grid of ``s`` (``WINDOW_SCALES``,
the largest bound kept):
``integral chi |E_k| = sqrt(V) pi / ((1 + s) sqrt(det S))``, ``integral chi = pi / (s sqrt(det S))``,
``integral chi^2 = pi / (2 s sqrt(det S))``. Then ``G`` is the Fourier transform of a Gaussian of complex form
``Q - i B`` (``Q = s S``, ``B`` the quadratic phase), ``|G(K)|^2 <= pi^2 / det Q exp(-|K|^2 / (2 Lambda))`` with
``Lambda = lambda_max(Q) + |B|^2 / lambda_min(Q)`` and ``K`` the plane wave's tangential wavevector less the beam's
(``K^T Re((Q - iB)^-1) K`` is at least ``|K|^2 / Lambda``, ``|det(Q - iB)| >= det Q``). Split at ``|K| = kappa``
(``TAIL_SIGMAS`` times ``sqrt(Lambda)``; the best kept): inside, ``cos >= c_A`` -- the cosine of the steepest wave
there, ``sqrt(1 - ((k_T sin(theta) + kappa) / k_side)^2)`` with ``k_side`` the wavenumber on that side --, so that
part is at most ``integral chi^2 / c_A`` on a plane (Parseval: the phase is fixed) and ``pi Lambda / (2 det Q c_A)`` on a
sphere (the phase depends on the wave); outside, ``|G|^2`` is below its bound at ``kappa`` and the weight integrates to
``2 pi k_side^2`` over the disk of propagating waves: ``pi k_side^2 / (2 det Q) exp(-kappa^2 / (2 Lambda))``. Where the
beam's tangential wavevector lies beyond a side's waves altogether (light totally reflected there), only that tail
counts. Both sides: the traced beam's medium and the other one (the larger index of the two media when the beam is in
the other; otherwise any index from 1 up to it, the smallest wavenumber for ``c_A`` and the largest for the tail).
``H`` is the larger of the two sides'. On a sphere, light that meets the surface from one side may cross its tangent
plane in either direction: Cauchy-Schwarz over both hemispheres, with the light's flux through the plane in either
direction at most its power (each ray crosses it once), gives ``sqrt(H_1^2 + H_2^2)`` per side (the bound on ``|G|``
holds for either sign of ``K_z``). A wave of the second hemisphere travels away from the surface at the window's centre,
yet meets the surface from the light's side somewhere in its clear aperture (the premise): it rises above the tangent
plane by less than the largest difference ``sigma`` of two normals there (``k . n0 < |n0 - n(y)|``), so its tangential
wavevector is at least ``k sqrt(1 - sigma^2)``. Where that lies beyond the beam's own by ``kappa > 0``, ``H_2`` is the
tail bound there, ``sqrt(pi k^2 / (2 det Q) exp(-kappa^2 / (2 Lambda)))`` with the truncation and sag terms; otherwise
``H_2 = H_1``. ``sigma = 2 (a / R) sqrt(2 / (1 + sqrt(1 - a^2 / R^2)))`` for an aperture of radius ``a`` (twice the
chord between the vertex's normal and the rim's).

Truncation and a sphere's sag. The window ends at the clear aperture, ``D`` from the beam's centre: what lies beyond
carries at most ``tau = pi / (s lambda_min(S)) exp(-s lambda_min(S) D^2)`` of ``integral chi`` (and of
``integral chi |E_k|`` over ``sqrt(V)``), and changes ``G`` by at most ``tau`` everywhere: ``H`` grows by at most
``tau k_side / sqrt(2 pi)``. On a sphere of radius ``R`` the surface lies ``h(x) = R - sqrt(R^2 - |x|^2)`` off its
tangent plane, within ``|x|^2 / R``; its quadratic part ``|x|^2 / (2 R)`` adds ``|K_z| / (2 R) <= (k_max + k_T) / (2 R)``
to ``|B|``, the rest, at most ``|x|^4 / (8 R^3 (1 - D^2 / R^2))``, and the beam phase's terms beyond quadratic,
``(k_T / 2) (|Re 1/q_x| + |Re 1/q_y|) (2 |x| h + h^2)``, change ``G`` by at most their integral against ``chi``; the
beam's profile at the surface points (``rho^2`` grows by at most ``(2 |x| h + h^2) (1 / w_x^2 + 1 / w_y^2)``) lowers
``integral chi |E_k|`` by at most ``sqrt(V)`` times that against ``chi exp(-x^T S x)`` (``e^-e >= 1 - e``); the window
ends within ``R / 2`` and within the cap. Another beam on a sphere is bounded with half its exponent and ``e^(eta^2)``,
``eta`` its normalized offset over the sag (``(rho - eta)^2 >= rho^2 / 2 - eta^2``). The moments
``integral exp(-a |x|^2) |x|^3 = 3 pi^(3/2) / (4 a^(5/2))`` and ``|x|^4``: ``2 pi / a^3``.

P / d. A beam's ``P / d`` is its power over its narrower transverse diameter (``W/cm``), weighted by its profile;
beams combine like fields with open phases, and the window uses no phase: pointwise the class's amplitude is at least
``sqrt(V) exp(-x^T S x) - sum_j |E_j| - B - sum_m |u_m|`` (the magnitudes, in amplitudes of ``P / d``), so its largest
value is at least the mean of that against ``chi`` -- the traced and the other beams' terms as above, and for each
piece ``m`` of the light of unknown place ``integral chi |u_m|``. That light counts as the Gaussian beams it is made of,
each at the ``P / d`` of its own size: ``|u_m| <= sqrt(kappa p_m / w) exp(-(u / w)^2)``, ``w`` its narrower radius at
the surface (at least the diffraction-limited waist ``w_0 = lambda / (pi n)``: at most the limit ``P / (2 w_0)``),
``u`` the distance across that axis ``b``, ``kappa`` the ``P / d`` of 1 W over a narrower radius of 1 mm. The premise:
each such beam is no more focused than the diffraction limit (its far-field 1/e^2 slopes at most 1 in every direction)
and crosses the surface from one side -- its far-field 1/e^2 cone lies on one side of the surface's tangent plane at
every point of the window's region. Its spectrum is at least ``theta = w_0 / w`` wide across ``b`` (``sigma_x sigma_k
>= 1/2``), so the cone condition gives ``|b x n| >= theta / sqrt(2 + theta^2)`` at every point: the level curves of
``u`` cross the surface at least that steeply, and by the coarea formula ``integral chi exp(-(u / w)^2) <= sqrt(pi) w
sqrt(2 + theta^2) / theta`` times the most ``chi`` along a level curve -- on a plane a line, ``sqrt(pi / a)`` with
``a = s lambda_min(S)`` (a line meets a disk of radius ``r`` along at most ``2 r``); on a sphere the arc of a circle of
radius at least ``rho = R theta / sqrt(2 + theta^2)``, the smaller of ``gamma sqrt(pi / a) (1 + CIRCLE_TERM gamma^2 /
(a rho^2))`` (a circle meets a ball of radius ``r`` along at most ``2 r + 2 (pi - 1) r^3 / rho^2``; ``gamma =
sqrt(1 + D^2 / R^2)``) and ``pi^(3/2) / sqrt(a)`` (any convex curve meets a disk along at most ``2 pi r``). So a piece
of power ``p`` takes away at most ``sqrt(p) F``, ``F`` the largest over ``w >= w_0`` of ``sqrt(kappa / w)`` times the
smaller of that and ``integral chi`` (bracketed by doubling and halving ``w``: the first grows with ``w``, the second
falls; ``F <= sqrt(kappa / w_0) integral chi``, the bound at the centres), and the light of unknown place at most
``F (sum_i sqrt(P_i) + c f sqrt(P*))`` (the smallest ``w_0`` of its lines). A thin sheet of that light meeting the
surface at a shallow angle keeps its ``P / d`` along a strip about ``w^2 / w_0`` wide, so in ``P / d`` the window lowers
that light's share far less than in irradiance (in amplitude to about the fourth root of ``w_0`` over the beam's radius,
there to about its first power). On a sphere the ``P / d`` window ends within ``LINEAR_REACH`` of its own 1/e widths.

Scope: in irradiance and fluence a traced beam of ``M^2 = 1`` (one Gaussian mode: its phase is known), in ``P / d`` any
traced beam, on a plane or a sphere; a sag surface keeps the bound at the traced beams' centres. The media's indices are
at least 1.

Arithmetic: ``+ - x /`` and ``sqrt`` on Python floats and elementwise NumPy, and the libm-free exponential: identical bits
in every runtime.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..math.exact_linalg import symmetric_eigenvalues_2x2
from ..math.exact_trig import exp as exact_exp
from ..optics import apertures
from ..units import NM_TO_MM

_PI = 3.141592653589793
_TWO_PI = 6.283185307179586
_SQRT_TWO_PI = 2.5066282746310002
#: ``3 pi^(3/2) / 4``: ``integral exp(-a |x|^2) |x|^3 d^2x = _M3 / a^(5/2)``.
_M3 = 4.176245997623781
#: ``sqrt(mm^2 per cm^2)``: a value's amplitude in sqrt(W/mm^2) over its amplitude in sqrt(W/cm^2).
SQRT_MM2_PER_CM2 = 10.0
#: The widest and the narrowest window searched, ``s`` in ``exp(-s x^T S x)``: from 1/16 (a window that falls to 1/e at
#: four times the beam's 1/e^2 radius) to at least 1.5e6 (at about 1/1200 of it).
SMALLEST_WINDOW_SCALE = 0.0625
LARGEST_WINDOW_SCALE = 1.5e6
#: Relative difference below which the surface's two refractive indices count as one medium.
SAME_INDEX = 1e-9
#: A sphere of radius ``R`` leaves its parabola by at most ``r^4 / (8 R^3 (1 - r^2 / R^2))`` at ``r`` from its vertex.
SAG_QUARTIC_DIVISOR = 8.0
_SQRT_PI = 1.7724538509055159
#: ``pi^(3/2)``: a convex curve meets a disk of radius ``r`` along at most ``2 pi r``, so it carries at most
#: ``pi^(3/2) / sqrt(a)`` of ``exp(-a |x|^2)``.
_PI_3_2 = 5.568327996831708
#: ``3 (pi - 1) / 2``: a circle of radius ``rho`` meets a ball of radius ``r`` along at most ``2 r + 2 (pi - 1) r^3 /
#: rho^2``, so it carries at most ``sqrt(pi / a) (1 + CIRCLE_TERM / (a rho^2))`` of ``exp(-a r^2)``.
CIRCLE_TERM = 3.2123889803846897
#: The doublings and halvings of the narrower radius at which a beam's strip on the window covers the window (``P / d``).
STRIP_DOUBLINGS = 200
STRIP_HALVINGS = 60
#: On a sphere the ``P / d`` window ends within this many of its own 1/e widths, ``1 / sqrt(s lambda_min(S))``: what lies
#: beyond carries ``exp(-16)`` of it (``tau``).
LINEAR_REACH = 4.0


def _scales() -> tuple[float, ...]:
    out = [SMALLEST_WINDOW_SCALE]
    step = math.sqrt(2.0)
    while out[-1] < LARGEST_WINDOW_SCALE:
        out.append(out[-1] * step)
    return tuple(out)


#: The window widths searched: ``chi = exp(-s x^T S x)`` for ``s`` from 1/16 to about 1.5e6, by factors of sqrt(2).
WINDOW_SCALES = _scales()
_S = np.array(WINDOW_SCALES)
#: Where the window's overlap with plane waves is split (multiples of ``sqrt(Lambda)``).
TAIL_SIGMAS = (6.0, 8.0, 10.0, 12.0)


@dataclass(frozen=True)
class WindowBeam:
    """A beam of known Gaussian profile on the surface: its peak value (the quantity's unit), chief-ray hit (mm), beam
    axes (unit, global) and their 1/e^2 radii (mm)."""

    value: float
    position: tuple[float, float, float]
    bx: tuple[float, float, float]
    by: tuple[float, float, float]
    wx: float
    wy: float


@dataclass(frozen=True)
class Anchor:
    """The traced beam the window follows: its profile (``beam``), unit surface normal at its hit, the cosine of its angle
    of incidence, its medium's index, vacuum wavelength (nm) and ``|Re 1/q|`` of its two axes (1/mm)."""

    beam: WindowBeam
    normal: tuple[float, float, float]
    cos_aoi: float
    n: float
    wavelength_nm: float
    curvature_x: float
    curvature_y: float


def _dot(a, b) -> float:
    return float(a[0]) * float(b[0]) + float(a[1]) * float(b[1]) + float(a[2]) * float(b[2])


def _unit(a) -> tuple[float, float, float]:
    length = math.sqrt(_dot(a, a))
    return (float(a[0]) / length, float(a[1]) / length, float(a[2]) / length)


def _basis(n: tuple[float, float, float]) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """An orthonormal basis of the plane normal to ``n`` (the helper axis the least aligned with it, first on ties)."""
    a = [abs(n[0]), abs(n[1]), abs(n[2])]
    index = a.index(min(a))
    helper = [0.0, 0.0, 0.0]
    helper[index] = 1.0
    cx = helper[1] * n[2] - helper[2] * n[1]
    cy = helper[2] * n[0] - helper[0] * n[2]
    cz = helper[0] * n[1] - helper[1] * n[0]
    e1 = _unit((cx, cy, cz))
    e2 = (n[1] * e1[2] - n[2] * e1[1], n[2] * e1[0] - n[0] * e1[2], n[0] * e1[1] - n[1] * e1[0])
    return e1, e2


def _form(bx, by, wx: float, wy: float, e1, e2) -> tuple[float, float, float]:
    """``x^T S x`` on the plane (basis ``e1``, ``e2``) of a beam's ``rho^2``: (S11, S12, S22)."""
    bxu, bxv, byu, byv = _dot(bx, e1), _dot(bx, e2), _dot(by, e1), _dot(by, e2)
    wx2, wy2 = wx * wx, wy * wy
    return (bxu * bxu / wx2 + byu * byu / wy2, bxu * bxv / wx2 + byu * byv / wy2, bxv * bxv / wx2 + byv * byv / wy2)


def _overlap(s: np.ndarray, form: tuple[float, float, float], beam: WindowBeam, origin, e1, e2, n0,
             sphere: float | None, sag: float) -> np.ndarray:
    """``integral chi |E_j|`` over ``sqrt(V_j)`` for each window scale: ``chi = exp(-s x^T S x)`` against the beam ``j``'s
    amplitude ``exp(-rho_j^2)``; on a sphere ``exp(-rho_j^2 / 2 + eta^2)`` (``sag``: the most the surface leaves its
    tangent plane within the window)."""
    m = (beam.bx, beam.by)
    w = (beam.wx, beam.wy)
    delta = (float(origin[0]) - beam.position[0], float(origin[1]) - beam.position[1], float(origin[2]) - beam.position[2])
    # rho_j(x)^2 = x^T Sj x + 2 bj . x + cj for x on the tangent plane
    s11 = s12 = s22 = b1 = b2 = c0 = nn = 0.0
    for axis, radius in zip(m, w):
        a1, a2, an, ad = _dot(axis, e1), _dot(axis, e2), _dot(axis, n0), _dot(axis, delta)
        inv = 1.0 / (radius * radius)
        s11 += a1 * a1 * inv
        s12 += a1 * a2 * inv
        s22 += a2 * a2 * inv
        b1 += a1 * ad * inv
        b2 += a2 * ad * inv
        c0 += ad * ad * inv
        nn += an * an * inv
    f = 1.0 if sphere is None else 0.5
    eta2 = 0.0 if sphere is None else sag * sag * nn
    a11 = s * form[0] + f * s11
    a12 = s * form[1] + f * s12
    a22 = s * form[2] + f * s22
    det = a11 * a22 - a12 * a12
    fb1, fb2 = f * b1, f * b2
    # b^T A^-1 b with b = f bj
    quad = (fb1 * fb1 * a22 - 2.0 * fb1 * fb2 * a12 + fb2 * fb2 * a11) / det
    exponent = np.minimum(quad - f * c0, 0.0) + eta2
    return _PI / np.sqrt(det) * exact_exp(exponent)


@dataclass(frozen=True)
class _Frame:
    """What the field window and the ``P / d`` window share for one traced beam, per window scale ``s`` (``_S``): the
    tangent-plane basis, the beam's footprint form ``S`` (its eigenvalues ``low`` <= ``high``, ``sqrt(det S)``), where the
    window ends (``distance``; a sphere's sag there), ``integral chi |E_k|`` over ``sqrt(V)`` (``traced``),
    ``integral chi`` (``window``), ``integral chi^2`` (``squared``), the truncation's share ``tau``, a sphere's profile loss
    (``loss``) and the other beams' overlaps (``taken``, in amplitudes of the value unit)."""

    n0: tuple[float, float, float]
    e1: tuple[float, float, float]
    e2: tuple[float, float, float]
    form: tuple[float, float, float]
    det_s: float
    low: float
    high: float
    root_det: float
    distance: float
    sag: float
    traced: np.ndarray
    window: np.ndarray
    squared: np.ndarray
    tau: np.ndarray
    loss: np.ndarray
    taken: np.ndarray


def _frame(beam: WindowBeam, normal, others: list[WindowBeam], sphere: float | None, room: float) -> _Frame | None:
    """The shared geometry (``_Frame``); ``None`` when the beam has no value, the window no room or the footprint no
    extent."""
    if not beam.value > 0.0 or not room > 0.0:
        return None
    n0 = _unit(normal)
    e1, e2 = _basis(n0)
    form = _form(beam.bx, beam.by, beam.wx, beam.wy, e1, e2)
    det_s = form[0] * form[2] - form[1] * form[1]
    if not det_s > 0.0:
        return None
    low, high = symmetric_eigenvalues_2x2(*form)
    if not low > 0.0:
        return None
    root_det = math.sqrt(det_s)
    distance = room
    sag = 0.0
    if sphere is not None:
        distance = min(room / math.sqrt(1.0 + room * room / (sphere * sphere)), 0.5 * sphere)
        sag = distance * distance / sphere
    s = _S
    q_low = s * low
    traced = _PI / ((1.0 + s) * root_det)
    window = _PI / (s * root_det)
    squared = _PI / (2.0 * s * root_det)
    tau = np.zeros_like(s) if not math.isfinite(distance) else _PI / q_low * exact_exp(-(q_low * (distance * distance)))
    loss = np.zeros_like(s)
    if sphere is not None:
        a1 = (1.0 + s) * low
        m3b = _M3 / (a1 * a1 * np.sqrt(a1))
        m4b = _TWO_PI / (a1 * a1 * a1)
        loss = (1.0 / (beam.wx * beam.wx) + 1.0 / (beam.wy * beam.wy)) * (2.0 * m3b / sphere + m4b / (sphere * sphere))
    taken = 0.0 * s
    for other in others:
        if not other.value > 0.0:
            continue
        taken = taken + math.sqrt(other.value) * np.minimum(_overlap(s, form, other, beam.position, e1, e2, n0, sphere, sag),
                                                            window)
    return _Frame(n0, e1, e2, form, det_s, low, high, root_det, distance, sag, traced, window, squared, tau, loss, taken)


def _amplitude(numerator: np.ndarray, window: np.ndarray) -> float:
    """The largest positive ``numerator / window`` over the window scales, squared (0 when none is)."""
    amplitude = numerator / window
    amplitude = np.where(np.isfinite(amplitude) & (amplitude > 0.0), amplitude, 0.0)
    best_amplitude = float(np.max(amplitude))
    return best_amplitude * best_amplitude


def window_lower(anchor: Anchor, others: list[WindowBeam], beside_root: float, unknown_root: float, n_max: float,
                 sphere: float | None, room: float, spread: float | None = None) -> float:
    """The class's exposure lower bound from the window (module docstring), in the anchor's value unit; 0 when it gives
    none. ``others``: the class's other beams of known profile; ``beside_root``: the beside beams' summed tail amplitudes
    (sqrt of the value unit); ``unknown_root``: ``sum_i sqrt(P_i) + c f sqrt(P*)`` of its light of unknown place (sqrt W,
    or sqrt J for a fluence); ``n_max``: the larger index of the surface's media; ``sphere``: the surface's radius
    (``None``: a plane); ``room``: the distance from the beam's hit to the edge of the clear aperture (mm); ``spread``: on a
    sphere the largest difference of two normals in its clear aperture (``surface_room``; ``None``: not known, the second
    hemisphere counts in full)."""
    beam = anchor.beam
    frame = _frame(beam, anchor.normal, others, sphere, room)
    if frame is None:
        return 0.0
    lam = anchor.wavelength_nm * NM_TO_MM
    k_t = _TWO_PI * anchor.n / lam
    k_max = _TWO_PI * max(n_max, anchor.n) / lam
    cos = min(1.0, abs(anchor.cos_aoi))
    tangential = k_t * math.sqrt(max(0.0, 1.0 - cos * cos))
    # the two sides: the beam's own medium, and the other (the larger index when the beam is in the other; else 1 to it)
    if anchor.n < max(n_max, anchor.n) * (1.0 - SAME_INDEX):
        sides = ((k_t, k_t), (k_max, k_max))
    else:
        sides = ((k_t, k_t), (_TWO_PI / lam, k_max))
    curvature = anchor.curvature_x + anchor.curvature_y
    beta = 0.5 * k_t * curvature
    distance = frame.distance
    if sphere is not None:
        beta += (k_max + k_t) / (2.0 * sphere)
    s = _S
    low, high, det_s = frame.low, frame.high, frame.det_s
    q_low, q_high = s * low, s * high
    det_q = s * s * det_s
    squared, tau = frame.squared, frame.tau
    lam_big = q_high + beta * beta / q_low
    remainder = np.zeros_like(s)
    if sphere is not None:
        a = q_low
        m3 = _M3 / (a * a * np.sqrt(a))
        m4 = _TWO_PI / (a * a * a)
        u = distance * distance / (sphere * sphere)
        r3 = sphere * sphere * sphere
        remainder = ((k_max + k_t) * m4 / (SAG_QUARTIC_DIVISOR * r3 * (1.0 - u))
                     + 0.5 * k_t * curvature * (2.0 * m3 / sphere + m4 / (sphere * sphere)))
    h = np.zeros_like(s)
    for k_low, k_high in sides:
        best = np.full_like(s, np.inf)
        for sigma in TAIL_SIGMAS:
            kappa = sigma * np.sqrt(lam_big)
            tail_scale = _PI * (k_high * k_high) / (2.0 * det_q)
            reach = tangential + kappa
            inside = reach < k_low
            ratio = np.where(inside, reach / k_low, 0.0)
            c_a = np.sqrt(np.maximum(1.0 - ratio * ratio, 0.0))
            safe = np.where(inside & (c_a > 0.0), c_a, 1.0)
            main = squared / safe if sphere is None else _PI * lam_big / (2.0 * det_q * safe)
            tail = tail_scale * exact_exp(-(kappa * kappa) / (2.0 * lam_big))
            value = np.where(inside & (c_a > 0.0), main + tail, np.inf)
            gap = tangential - k_high
            if gap > 0.0:
                # (the beam's tangential wavevector beyond this side's waves: only the tail of G reaches them)
                beyond = tail_scale * exact_exp(-(gap * gap) / (2.0 * lam_big))
                value = np.minimum(value, beyond)
            best = np.minimum(best, value)
        side = np.sqrt(best) + (tau + remainder) * k_high / _SQRT_TWO_PI
        if sphere is not None:
            # (light meeting a sphere may cross its tangent plane in either direction: both hemispheres count, the second
            # only with waves rising less than the clear aperture's normals differ, beyond a tangential wavevector)
            second = side
            if spread is not None and spread < 1.0:
                gap = k_low * math.sqrt(1.0 - spread * spread) - tangential
                if gap > 0.0:
                    tail_scale = _PI * (k_high * k_high) / (2.0 * det_q)
                    tail = np.sqrt(tail_scale * exact_exp(-(gap * gap) / (2.0 * lam_big))) + (tau + remainder) * k_high / _SQRT_TWO_PI
                    second = np.minimum(side, tail)
            side = np.sqrt(side * side + second * second)
        h = np.maximum(h, side)
    numerator = math.sqrt(beam.value) * (frame.traced - tau - frame.loss) - frame.taken - beside_root * frame.window
    if unknown_root > 0.0:
        numerator = numerator - SQRT_MM2_PER_CM2 * h * unknown_root
    return _amplitude(numerator, frame.window)


def _strip(w: np.ndarray, waist: float, a: np.ndarray, line: np.ndarray, sphere: float | None, gamma: np.ndarray) -> np.ndarray:
    """``integral chi exp(-(u / w)^2) dx`` over the window's region for a beam of narrower radius ``w`` crossing the
    surface from one side (module docstring, "P / d"), at most: ``sqrt(pi) w`` over the least slope
    ``theta / sqrt(2 + theta^2)`` (``theta = waist / w``) times the most ``chi`` along a level curve -- a line on a
    plane, ``sqrt(pi / a)``; on a sphere the arc of a circle of radius at least ``R`` times that slope, the smaller of
    ``gamma sqrt(pi / a) (1 + CIRCLE_TERM gamma^2 / (a rho^2))`` and ``pi^(3/2) / sqrt(a)``."""
    stretch = np.sqrt(2.0 * w * w + waist * waist) / waist
    along = line
    if sphere is not None:
        inverse_rho2 = (2.0 * w * w + waist * waist) / (sphere * sphere * waist * waist)
        along = np.minimum(_PI_3_2 / np.sqrt(a), gamma * line * (1.0 + CIRCLE_TERM * gamma * gamma * inverse_rho2 / a))
    return _SQRT_PI * w * stretch * along


def _unknown_linear(frame: _Frame, kappa: float, waist: float, sphere: float | None) -> tuple[np.ndarray, np.ndarray]:
    """Per window scale: ``integral chi |u|`` over the window's region, at most, for light of unknown place of 1 W (sqrt W/cm
    times mm^2; module docstring, "P / d") -- the largest over the narrower radius ``w >= waist`` of ``sqrt(kappa / w)``
    times the smaller of ``_strip(w)`` and ``integral chi``, bracketed by doubling and halving (the first factor of the
    smaller grows with ``w``, the second falls); and that region's ``tau`` (a sphere's window ends within
    ``LINEAR_REACH`` of its own 1/e widths)."""
    s = _S
    window = frame.window
    a = s * frame.low
    line = np.sqrt(_PI / a)
    tau = frame.tau
    gamma = np.ones_like(s)
    if sphere is not None:
        distance = np.minimum(frame.distance, LINEAR_REACH / np.sqrt(a))
        tau = _PI / a * exact_exp(-(a * (distance * distance)))
        gamma = np.sqrt(1.0 + distance * distance / (sphere * sphere))
    lo = np.full_like(s, waist)
    trivial = np.sqrt(kappa / lo) * window
    open_ = _strip(lo, waist, a, line, sphere, gamma) < window
    hi = lo
    for _ in range(STRIP_DOUBLINGS):
        grow = open_ & (_strip(hi, waist, a, line, sphere, gamma) < window)
        if not bool(np.any(grow)):
            break
        hi = np.where(grow, 2.0 * hi, hi)
    reached = open_ & (_strip(hi, waist, a, line, sphere, gamma) >= window)
    lo = np.where(reached, np.maximum(lo, 0.5 * hi), lo)
    for _ in range(STRIP_HALVINGS):
        mid = 0.5 * (lo + hi)
        below = _strip(mid, waist, a, line, sphere, gamma) < window
        lo = np.where(reached & below, mid, lo)
        hi = np.where(reached & ~below, mid, hi)
    # (the largest value of min(sqrt(kappa / w) strip(w), sqrt(kappa / w) window) is at most its two ends' at the bracket)
    bracket = np.minimum(np.sqrt(kappa / hi) * _strip(hi, waist, a, line, sphere, gamma), np.sqrt(kappa / lo) * window)
    return np.where(reached, np.minimum(bracket, trivial), trivial), tau


def linear_lower(beam: WindowBeam, normal, others: list[WindowBeam], beside_root: float, unknown_root: float, kappa: float,
                 waist: float, sphere: float | None, room: float) -> float:
    """The class's ``P / d`` lower bound from the window (module docstring, "P / d"), in W/cm; 0 when it gives none.
    ``beam``, ``others``, ``beside_root`` as ``window_lower`` in ``P / d`` (sqrt W/cm); ``unknown_root``: ``sum_i
    sqrt(P_i) + c f sqrt(P*)`` of its light of unknown place (sqrt W); ``kappa``: ``P / d`` in W/cm of 1 W over a narrower
    radius of 1 mm (``MM_PER_CM / 2``, or ``MM_PER_CM / FWHM_PER_RADIUS`` for the FWHM diameter); ``waist``: the smallest
    diffraction-limited waist of that light (mm); ``sphere``, ``room``: as ``window_lower``."""
    frame = _frame(beam, normal, others, sphere, room)
    if frame is None or not waist > 0.0:
        return 0.0
    window = frame.window
    tau = frame.tau
    unknown = np.zeros_like(window)
    if unknown_root > 0.0:
        bound, tau = _unknown_linear(frame, kappa, waist, sphere)
        unknown = bound * unknown_root
    numerator = math.sqrt(beam.value) * (frame.traced - tau - frame.loss) - frame.taken - beside_root * window - unknown
    return _amplitude(numerator, window)


def _plane_room(shape: str, p1: float, p2: float, offset_u: float, offset_v: float, obstruction: float, u: float,
                v: float) -> float:
    """The distance from ``(u, v)`` to the edge of a clear aperture, inward (negative outside it; ``inf`` without one)."""
    if shape == "none":
        return math.inf
    du, dv = u - offset_u, v - offset_v
    if shape == "circle":
        r = math.sqrt(du * du + dv * dv)
        room = p1 - r
        return min(room, r - obstruction) if obstruction > 0.0 else room
    return min(p1 - abs(du), p2 - abs(dv))


def normal_spread(aperture_radius: float, radius: float) -> float:
    """The largest difference ``|n(x) - n(y)|`` of two unit normals of a sphere of ``radius`` within a cap of lateral
    ``aperture_radius``, at most: twice the chord from the vertex's normal to the rim's, ``2 (a / R) sqrt(2 / (1 +
    sqrt(1 - a^2 / R^2)))`` (2 for a hemisphere and beyond)."""
    u = aperture_radius / radius
    if not u < 1.0:
        return 2.0
    return 2.0 * u * math.sqrt(2.0 / (1.0 + math.sqrt(1.0 - u * u)))


def surface_room(scene) -> Callable[[str, object], tuple[float | None, float, float | None] | None]:
    """(surface id, a point of it) -> (the sphere's radius, ``None`` for a plane; the distance from the point to its clear
    aperture's edge; on a sphere the largest difference of two normals in its clear aperture, ``normal_spread``) for the
    surfaces the window bound applies to; ``None`` for any other (a sag surface)."""
    by_id = {s.surface_id: s for s in scene.surfaces}

    def of(surface_id: str, point) -> tuple[float | None, float, float | None] | None:
        s = by_id.get(surface_id)
        if s is None:
            return None
        if s.geometry == "plane":
            rel = (float(point[0]) - float(s.origin[0]), float(point[1]) - float(s.origin[1]), float(point[2]) - float(s.origin[2]))
            ap = s.aperture
            if ap.shape not in ("none", "circle", "rectangle") or ap.holes:  # (a lower bound of the room for any other region)
                return None, apertures.room(ap, _dot(rel, s.u_axis), _dot(rel, s.v_axis)), None
            return None, _plane_room(ap.shape, ap.p1, ap.p2, ap.offset_u, ap.offset_v, ap.obstruction, _dot(rel, s.u_axis),
                                     _dot(rel, s.v_axis)), None
        if s.geometry == "sphere":
            rel = (float(point[0]) - float(s.center[0]), float(point[1]) - float(s.center[1]), float(point[2]) - float(s.center[2]))
            axial = _dot(rel, s.cap_axis)
            tx = rel[0] - axial * float(s.cap_axis[0])
            ty = rel[1] - axial * float(s.cap_axis[1])
            tz = rel[2] - axial * float(s.cap_axis[2])
            return (float(s.radius), float(s.aperture_radius) - math.sqrt(tx * tx + ty * ty + tz * tz),
                    normal_spread(float(s.aperture_radius), float(s.radius)))
        return None

    return of


__all__ = ["Anchor", "SQRT_MM2_PER_CM2", "TAIL_SIGMAS", "WINDOW_SCALES", "WindowBeam", "linear_lower", "normal_spread",
           "surface_room", "window_lower"]
