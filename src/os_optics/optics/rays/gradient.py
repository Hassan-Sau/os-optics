"""Rays in a radial gradient-index medium (a GRIN rod): closed-form propagation between the rod's two end faces.

The medium (``contracts.materials.GradientIndexMaterial``) has ``n(r)^2 = n0^2 (1 - (g r)^2)``, ``r`` the distance from its axis.
With ``z`` along the axis and ``beta = n dz/ds`` (conserved: the medium does not vary along ``z``), the ray equation
``d/ds (n dr/ds) = grad n`` becomes, in ``z``,

    x'' = (1 / (2 beta^2)) d(n^2)/dx = -W^2 x,    y'' = -W^2 y,    W = n0 g / beta,

so every ray -- meridional or skew -- is exactly ``x(z) = x0 cos(W z) + (p / W) sin(W z)``, ``y`` alike with ``q``, ``p`` and
``q`` its slopes ``dx/dz``, ``dy/dz`` at the start (E. W. Marchand, *Gradient Index Optics*, Academic Press 1978, ch. 3; for
paraxial rays ``beta = n0`` and the period along the axis is ``2 pi / g``). The local index follows from the invariant:
``n^2 = beta^2 (1 + x'^2 + y'^2)`` (``x'^2 + W^2 x^2`` is constant on a ray). Along the ray ``ds = (n / beta) dz``, so

* the optical path is ``int n ds = (n0^2 / beta) (Z - g^2 int r^2 dz)`` with ``r^2 = A + B cos(2 W z) + C sin(2 W z)``,
  ``A = (x0^2 + (p/W)^2 + y0^2 + (q/W)^2) / 2``, ``B = (x0^2 - (p/W)^2 + y0^2 - (q/W)^2) / 2``, ``C = x0 p / W + y0 q / W``:
  ``int_0^Z r^2 dz = A Z + (B sin(2 W Z) + C (1 - cos(2 W Z))) / (2 W)`` (closed form);
* the geometric length ``int (n / beta) dz`` and the group path ``int n_g (n / beta) dz`` are Gauss-Legendre sums over
  ``GL_NODES``-point panels of at most a quarter period of ``r^2`` each (smooth integrands: exact to rounding);
* the largest ``r^2`` on the way is ``A + sqrt(B^2 + C^2)`` where the phase ``atan2(C, B)`` of the cosine's crest falls on
  the way (else the larger end): a ray beyond the rod's radius leaves through its side.

The polarization of a ray in an isotropic medium is carried along the ray without turning about it (parallel transport, the
Rytov law): ``transported`` applies the double-reflection rotation-minimizing frame (W. Wang, B. Juttler, D. Zheng, Y. Liu,
ACM Trans. Graph. 27 (2008) 2) on ``points_per_period`` points per period of the path.

Trigonometry is ``math.exact_trig`` (no library transcendentals): the same bits in every runtime.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from ...math.exact_trig import atan2, cos_sin
from ...scene.model import GradientFaceG
from ..gaussian.truncation import gauss_legendre_rule

Array = NDArray[np.float64]

#: Gauss-Legendre nodes per panel of the path integrals, and the longest panel (rad of the phase 2 W z of r^2).
GL_NODES = 8
PANEL_PHASE_RAD = 0.5 * math.pi
#: Points per period of the path (2 pi / W along the axis) of the polyline that carries the polarization and on which the
#: chords are checked against the scene's other surfaces; at least ``MIN_POINTS`` and at most ``MAX_POINTS`` per ray.
POINTS_PER_PERIOD = 256
MIN_POINTS = 16
MAX_POINTS = 8192
TWO_PI = 2.0 * math.pi
#: Below this phase ``W t`` (rad) ``sin(W t) / W`` is taken from its series ``t (1 - (W t)^2 / 6)``; below this span ``2 W Z`` the
#: integral of ``r^2`` from its series; a ray whose axial direction component is at most ``ACROSS_AXIS`` runs across the axis.
SERIES_PHASE_RAD = 1e-8
SINE_SERIES_DIVISOR = 6.0
SERIES_SPAN_RAD = 1e-6
ACROSS_AXIS = 1e-12


@dataclass(frozen=True)
class Profile:
    """The medium at one wavelength: ``n0`` on the axis, ``g`` (1/mm), and the derivatives the group path needs (per nm:
    ``dn0``, ``d2n0``, ``dg``; the gradient is linear between its rows, so ``d2g`` is 0)."""

    n0: float
    g: float
    dn0: float = 0.0
    d2n0: float = 0.0
    dg: float = 0.0

    def index(self, r2: Array) -> Array:
        return self.n0 * np.sqrt(1.0 - (self.g * self.g) * r2)


@dataclass(frozen=True)
class Legs:
    """Closed-form legs of a set of rays from where they are to the plane ``z_end`` (local frame: transverse ``x``, ``y``,
    axial ``z``). Per ray: ``beta``, ``w``, the start ``x0``, ``y0``, slopes ``p``, ``q``, the axial length ``z_len``, the
    direction ``sign`` (+1 / -1 along the axis), and ``r2`` coefficients ``a``, ``b``, ``c``."""

    beta: Array
    w: Array
    x0: Array
    y0: Array
    p: Array
    q: Array
    z0: Array
    z_len: Array
    sign: Array
    a: Array
    b: Array
    c: Array

    def rows(self, index: int) -> Legs:
        """The leg of one ray (arrays of one element)."""

        k = slice(index, index + 1)
        return Legs(self.beta[k], self.w[k], self.x0[k], self.y0[k], self.p[k], self.q[k], self.z0[k], self.z_len[k], self.sign[k],
                    self.a[k], self.b[k], self.c[k])

    def between(self, t0: float, t1: float) -> Legs:
        """The stretch of every ray from the axial distance ``t0`` to ``t1`` as legs of their own (the ray equation does not
        depend on ``z``: the stretch starts where the ray is at ``t0``, with its slopes there)."""

        x, y, _z, dx, dy = self.at(np.full(self.w.shape, t0))
        w = self.w
        px, py = dx / w, dy / w
        a = 0.5 * (x * x + px * px + y * y + py * py)
        b = 0.5 * (x * x - px * px + y * y - py * py)
        c = x * px + y * py
        return Legs(self.beta, w, x, y, dx, dy, self.z0 + self.sign * t0, np.full(w.shape, t1 - t0), self.sign, a, b, c)

    def at(self, t: Array) -> tuple[Array, Array, Array, Array, Array]:
        """``(x, y, z, x', y')`` at the axial distance ``t`` (per ray, same shape as ``t`` broadcast against the rays)."""

        phase = self.w * t
        c, s = cos_sin(phase)
        sw = _sin_over_w(s, phase, self.w, t)
        x = self.x0 * c + self.p * sw
        y = self.y0 * c + self.q * sw
        dx = -self.x0 * self.w * s + self.p * c
        dy = -self.y0 * self.w * s + self.q * c
        return x, y, self.z0 + self.sign * t, dx, dy

    def r2(self, t: Array) -> Array:
        c, s = cos_sin(2.0 * self.w * t)
        return self.a + self.b * c + self.c * s

    def largest_r2(self) -> Array:
        """The largest ``r^2`` on each ray's way (module docstring)."""

        span = 2.0 * self.w * self.z_len
        crest = atan2(self.c, self.b)
        crest = np.where(crest < 0.0, crest + TWO_PI, crest)
        ends = np.maximum(self.a + self.b, self.r2(self.z_len))
        amplitude = np.sqrt(self.b * self.b + self.c * self.c)
        return np.where((span >= TWO_PI) | (crest <= span), self.a + amplitude, ends)

    def r2_integral(self) -> Array:
        """``int_0^z_len r^2 dz`` (closed form)."""

        span = 2.0 * self.w * self.z_len
        c, s = cos_sin(span)
        two_w = 2.0 * self.w
        small = span < SERIES_SPAN_RAD
        # (a ray with W Z -> 0: r^2 is nearly constant, the series of (B sin + C (1 - cos)) / 2W)
        series = self.z_len * (self.b + self.c * self.w * self.z_len)
        part = np.where(small, series, (self.b * s + self.c * (1.0 - c)) / np.where(small, 1.0, two_w))
        return self.a * self.z_len + part

    def optical_path(self, profile: Profile) -> Array:
        return (profile.n0 * profile.n0 / self.beta) * (self.z_len - (profile.g * profile.g) * self.r2_integral())

    def quadrature(self) -> tuple[Array, Array]:
        """Nodes ``t`` (shape ``(k, rays)``) and weights (same shape) of the path integrals over each ray's ``[0, z_len]``."""

        span = 2.0 * self.w * self.z_len
        panels = max(1, int(math.ceil(float(np.max(span, initial=0.0)) / PANEL_PHASE_RAD)))
        nodes, weights = gauss_legendre_rule(GL_NODES)
        u = ((np.arange(panels)[:, None] + 0.5 * (nodes[None, :] + 1.0)) / panels).ravel()  # in [0, 1]
        wu = np.tile(0.5 * weights / panels, panels)
        return u[:, None] * self.z_len[None, :], wu[:, None] * self.z_len[None, :]

    def lengths(self, profile: Profile) -> Array:
        """The geometric length ``int (n / beta) dz`` of each ray's leg."""

        t, wt = self.quadrature()
        n = profile.index(self.r2(t))
        return np.sum(wt * n, axis=0) / self.beta

    def group(self, profile: Profile, wavelength_nm: float, nm3_per_nm2_to_mm: float) -> tuple[float, float]:
        """The chief ray's (row 0) group path ``int n_g ds`` and its group-delay dispersion ``int k2 ds`` (mm^2), with
        ``n_g = n - lambda dn/dlambda`` and ``k2 = lambda^3 / (2 pi) d2n/dlambda2`` of ``n = n0 S``, ``S = sqrt(1 - g^2 r^2)``,
        at fixed geometry (``optics/materials/group.py``'s convention)."""

        chief = self.rows(0)
        t, wt = chief.quadrature()
        r2 = chief.r2(t)
        g, n0 = profile.g, profile.n0
        root = np.sqrt(1.0 - g * g * r2)
        ds = (n0 * root / chief.beta) * wt
        d_root = -g * profile.dg * r2 / root
        d2_root = -(profile.dg * profile.dg) * r2 / root - (g * profile.dg * r2) ** 2 / (root * root * root)
        n = n0 * root
        dn = profile.dn0 * root + n0 * d_root
        d2n = profile.d2n0 * root + 2.0 * profile.dn0 * d_root + n0 * d2_root
        lam = wavelength_nm
        ng = n - lam * dn
        k2 = lam * lam * lam * d2n * nm3_per_nm2_to_mm / TWO_PI
        return float(np.sum(ng * ds)), float(np.sum(k2 * ds))

    def points(self, count: int) -> tuple[Array, Array]:
        """``count + 1`` points along each ray's leg (axial steps of ``z_len / count``) and the unit tangents there, in the
        local frame: arrays of shape ``(count + 1, rays, 3)``."""

        t = (np.arange(count + 1, dtype=np.float64)[:, None] / count) * self.z_len[None, :]
        x, y, z, dx, dy = self.at(t)
        pts = np.stack([x, y, z], axis=-1)
        tan = np.stack([dx, dy, np.broadcast_to(self.sign, dx.shape)], axis=-1)
        tan = tan / np.sqrt(np.sum(tan * tan, axis=-1, keepdims=True))
        return pts, tan

    def point_count(self) -> int:
        periods = float(np.max(self.w * self.z_len, initial=0.0)) / TWO_PI
        return int(min(MAX_POINTS, max(MIN_POINTS, math.ceil(periods * POINTS_PER_PERIOD))))

    def with_length(self, z_len: Array) -> Legs:
        """The same rays' legs to the axial distances ``z_len`` (where each meets an end face that is not a plane across the
        axis: ``to_face``)."""

        return replace(self, z_len=z_len)

    def exit_index(self, profile: Profile) -> Array:
        """The index where each leg ends."""

        return profile.index(self.r2(self.z_len))

    def slope_bound(self, _t0: Array, _t1: Array) -> Array:
        """A bound on each ray's slope ``|(x', y')|`` across the axis anywhere on it: ``x'^2 + W^2 x^2`` is constant per axis,
        so ``|(x', y')|^2 <= p^2 + q^2 + W^2 (x0^2 + y0^2)``."""

        return np.sqrt(self.p * self.p + self.q * self.q + self.w * self.w * (self.x0 * self.x0 + self.y0 * self.y0))

    def turns(self, _profile: Profile | None = None) -> NDArray[np.bool_]:
        """Rays that turn back along the axis on their leg: none in a radial medium (``beta`` is conserved)."""

        return np.zeros(self.w.shape, dtype=bool)


def _sin_over_w(s: Array, phase: Array, w: Array, t: Array) -> Array:
    """``sin(W t) / W`` (``t`` where ``W t`` is tiny)."""

    tiny = np.abs(phase) < SERIES_PHASE_RAD
    return np.where(tiny, t * (1.0 - phase * phase / SINE_SERIES_DIVISOR), s / np.where(tiny, 1.0, w))


def legs(profile: Profile | AxialProfile, x0: Array, y0: Array, z0: Array, d_local: Array, z_lo: float, z_hi: float
         ) -> tuple[Legs | AxialLegs, Array]:
    """The legs of rays at ``(x0, y0, z0)`` travelling along ``d_local`` (unit, local frame) to the end face ahead
    (``z_hi`` travelling +z, ``z_lo`` travelling -z). Returns the legs and a mask of rays that have one (a ray across
    the axis, ``d_z = 0``, has none: it leaves through the side). An axial medium's legs are ``AxialLegs``."""

    if isinstance(profile, AxialProfile):
        return _axial_legs(profile, x0, y0, z0, d_local, z_lo, z_hi)
    dz = d_local[:, 2]
    along = np.abs(dz) > ACROSS_AXIS
    sign = np.where(dz >= 0.0, 1.0, -1.0)
    safe = np.where(along, np.abs(dz), 1.0)
    r2_0 = x0 * x0 + y0 * y0
    n_start = profile.index(r2_0)
    beta = n_start * safe
    w = profile.n0 * profile.g / beta
    p, q = d_local[:, 0] / safe, d_local[:, 1] / safe
    z_end = np.where(sign > 0.0, z_hi, z_lo)
    z_len = np.maximum(0.0, (z_end - z0) * sign)
    px, py = p / w, q / w
    a = 0.5 * (x0 * x0 + px * px + y0 * y0 + py * py)
    b = 0.5 * (x0 * x0 - px * px + y0 * y0 - py * py)
    c = x0 * px + y0 * py
    return Legs(beta, w, x0, y0, p, q, z0, z_len, sign, a, b, c), along


def transported(points: Array, tangents: Array, r0: Array) -> Array:
    """``r0`` (unit, perpendicular to the first tangent; shape ``(rays, 3)``) carried along the polylines ``points`` with
    unit ``tangents`` (shape ``(m + 1, rays, 3)``) by the double-reflection rotation-minimizing frame."""

    r = r0
    for i in range(points.shape[0] - 1):
        v1 = points[i + 1] - points[i]
        c1 = np.sum(v1 * v1, axis=-1, keepdims=True)
        c1 = np.where(c1 > 0.0, c1, 1.0)
        r_l = r - (2.0 / c1) * np.sum(v1 * r, axis=-1, keepdims=True) * v1
        t_l = tangents[i] - (2.0 / c1) * np.sum(v1 * tangents[i], axis=-1, keepdims=True) * v1
        v2 = tangents[i + 1] - t_l
        c2 = np.sum(v2 * v2, axis=-1, keepdims=True)
        r = np.where(c2 > 0.0, r_l - (2.0 / np.where(c2 > 0.0, c2, 1.0)) * np.sum(v2 * r_l, axis=-1, keepdims=True) * v2, r_l)
    # (the frame's rounding: back onto the plane across the last tangent, unit length)
    last = tangents[-1]
    r = r - np.sum(r * last, axis=-1, keepdims=True) * last
    return r / np.sqrt(np.sum(r * r, axis=-1, keepdims=True))




# ------------------------------------------------------------------------------------------------ axial gradient-index media

#: Gauss-Legendre panels (``GL_NODES`` points each) over a whole leg in an axial medium and over one piece of it, and the
#: points of its polyline (the check against the scene's other surfaces, the recorded path): the ray lies in one plane, so its
#: polarization is carried in closed form (``planar_transport``).
AXIAL_PANELS = 16
AXIAL_PIECE_PANELS = 2
AXIAL_POINTS = 32
#: A ray in an axial medium is followed while the cosine of its angle to the axis stays at least this; nearer the turning
#: point (``n(z) = |k|``, where the ray would turn back) it is out of model scope.
AXIAL_MIN_COS = 0.05


@dataclass(frozen=True)
class AxialProfile:
    """An axial medium at one wavelength: ``n(z) = n0 + a1 z + a2 z^2 + a3 z^3`` (``terms``), ``z`` (mm) along the axis from
    its axis point, and the derivatives of ``n0`` the group path needs (per nm; the terms do not vary with wavelength)."""

    n0: float
    terms: tuple[float, ...]
    dn0: float = 0.0
    d2n0: float = 0.0

    def index_z(self, z: Array) -> Array:
        """``n(z)`` (Horner's rule)."""

        acc = np.zeros(np.shape(z))
        for a in reversed(self.terms):
            acc = (acc + a) * z
        return self.n0 + acc

    def least(self, z_a: Array, z_b: Array) -> Array:
        """The least ``n`` between ``z_a`` and ``z_b`` (per entry): at an end or where ``dn/dz = a1 + 2 a2 z + 3 a3 z^2``
        vanishes between them."""

        lo, hi = np.minimum(z_a, z_b), np.maximum(z_a, z_b)
        best = np.minimum(self.index_z(lo), self.index_z(hi))
        a1, a2, a3 = (*self.terms, 0.0, 0.0)[:3]
        # (dn/dz = d1 + d2 z + d3 z^2)
        d1, d2, d3 = a1, a2 + a2, a3 + a3 + a3
        roots: list[float] = []
        if d3 != 0.0:
            disc = d2 * d2 - 4.0 * d3 * d1
            if disc >= 0.0:
                root = math.sqrt(disc)
                roots = [(-d2 + root) / (d3 + d3), (-d2 - root) / (d3 + d3)]
        elif d2 != 0.0:
            roots = [-d1 / d2]
        for z in roots:
            best = np.minimum(best, self.index_z(np.clip(z, lo, hi)))
        return best


@dataclass(frozen=True)
class AxialLegs:
    """Legs of rays in an axial medium (local frame: transverse ``x``, ``y``, axial ``z``). The index does not vary across
    the axis, so each ray keeps its transverse optical direction ``k = (kx, ky) = n (t_x, t_y)`` and, with ``tau`` the axial
    distance travelled and ``R(z) = sqrt(n(z)^2 - |k|^2)``:

    * ``x(tau) = x0 + kx int_0^tau dtau' / R``, ``y`` alike, ``z = z0 + sign tau`` (``dx / dtau = kx / R``);
    * the length ``int n / R dtau`` and the optical path ``int n^2 / R dtau``;

    integrals of smooth functions summed by Gauss-Legendre panels (``AXIAL_PANELS``). Per ray: the start ``x0``, ``y0``,
    ``z0``, ``kx``, ``ky``, the axial length ``z_len`` and the direction ``sign``."""

    profile: AxialProfile
    x0: Array
    y0: Array
    z0: Array
    kx: Array
    ky: Array
    z_len: Array
    sign: Array

    def rows(self, index: int) -> AxialLegs:
        k = slice(index, index + 1)
        return AxialLegs(self.profile, self.x0[k], self.y0[k], self.z0[k], self.kx[k], self.ky[k], self.z_len[k], self.sign[k])

    def with_length(self, z_len: Array) -> AxialLegs:
        return replace(self, z_len=z_len)

    def _k2(self) -> Array:
        return self.kx * self.kx + self.ky * self.ky

    def _integrals(self, t0: Array, t1: Array, panels: int) -> tuple[Array, Array, Array]:
        """``int dtau / R``, ``int n dtau / R`` and ``int n^2 dtau / R`` from ``t0`` to ``t1`` (arrays shaped like the rays, or
        ``(k, rays)``)."""

        nodes, weights = gauss_legendre_rule(GL_NODES)
        u = ((np.arange(panels)[:, None] + 0.5 * (nodes[None, :] + 1.0)) / panels).ravel()
        wu = np.tile(0.5 * weights / panels, panels)
        span = t1 - t0
        shape = (1,) * (np.ndim(span) - 1) + (u.size, 1)
        tau = np.expand_dims(t0, -2) + u.reshape(shape) * np.expand_dims(span, -2)
        n = self.profile.index_z(self.z0 + self.sign * tau)
        with np.errstate(invalid="ignore"):
            inv = 1.0 / np.sqrt(n * n - self._k2())
        w = wu.reshape(shape) * np.expand_dims(span, -2)
        return np.sum(w * inv, axis=-2), np.sum(w * n * inv, axis=-2), np.sum(w * n * n * inv, axis=-2)

    def _slopes(self, z: Array) -> tuple[Array, Array]:
        n = self.profile.index_z(z)
        with np.errstate(invalid="ignore", divide="ignore"):
            root = np.sqrt(n * n - self._k2())
            return self.kx / root, self.ky / root

    def at(self, t: Array) -> tuple[Array, Array, Array, Array, Array]:
        """``(x, y, z, x', y')`` at the axial distance ``t`` (``x' = dx / dtau``)."""

        t = np.asarray(t, dtype=np.float64)
        i1, _i2, _i3 = self._integrals(np.zeros(t.shape), t, AXIAL_PANELS)
        z = self.z0 + self.sign * t
        dx, dy = self._slopes(z)
        return self.x0 + self.kx * i1, self.y0 + self.ky * i1, z, dx, dy

    def largest_r2(self) -> Array:
        """The largest ``r^2`` on each ray's way: its path across the axis is a straight line, so at an end."""

        x, y, _z, _dx, _dy = self.at(self.z_len)
        return np.maximum(self.x0 * self.x0 + self.y0 * self.y0, x * x + y * y)

    def exit_index(self, _profile: AxialProfile | None = None) -> Array:
        return self.profile.index_z(self.z0 + self.sign * self.z_len)

    def optical_path(self, _profile: AxialProfile | None = None) -> Array:
        return self._integrals(np.zeros(self.z_len.shape), self.z_len, AXIAL_PANELS)[2]

    def lengths(self, _profile: AxialProfile | None = None) -> Array:
        return self._integrals(np.zeros(self.z_len.shape), self.z_len, AXIAL_PANELS)[1]

    def group(self, profile: AxialProfile, wavelength_nm: float, nm3_per_nm2_to_mm: float) -> tuple[float, float]:
        """The chief ray's group path ``int (n - lambda dn0/dlambda) ds`` and its group-delay dispersion ``int k2 ds`` (the
        terms do not vary with wavelength, so ``dn / dlambda = dn0 / dlambda`` everywhere)."""

        chief = self.rows(0)
        _i1, length, opl = chief._integrals(np.zeros(1), chief.z_len, AXIAL_PANELS)
        lam = wavelength_nm
        group = float(opl[0]) - lam * profile.dn0 * float(length[0])
        k2 = lam * lam * lam * profile.d2n0 * nm3_per_nm2_to_mm / TWO_PI
        return group, k2 * float(length[0])

    def points(self, count: int) -> tuple[Array, Array]:
        """``count + 1`` points along each leg (axial steps of ``z_len / count``) and the unit tangents there (local frame)."""

        t = (np.arange(count + 1, dtype=np.float64)[:, None] / count) * self.z_len[None, :]
        pieces = self._integrals(t[:-1], t[1:], AXIAL_PIECE_PANELS)[0]
        i1 = np.concatenate([np.zeros((1, t.shape[1])), np.cumsum(pieces, axis=0)], axis=0)
        z = self.z0 + self.sign * t
        x, y = self.x0 + self.kx * i1, self.y0 + self.ky * i1
        dx, dy = self._slopes(z)
        tan = np.stack([dx, dy, np.broadcast_to(self.sign, dx.shape)], axis=-1)
        tan = tan / np.sqrt(np.sum(tan * tan, axis=-1, keepdims=True))
        return np.stack([x, y, z], axis=-1), tan

    def point_count(self) -> int:
        return AXIAL_POINTS

    def between(self, t0: float, t1: float) -> AxialLegs:
        """The stretch from the axial distance ``t0`` to ``t1`` as legs of their own."""

        x, y, z, _dx, _dy = self.at(np.full(self.z_len.shape, t0))
        return AxialLegs(self.profile, x, y, z, self.kx, self.ky, np.full(self.z_len.shape, t1 - t0), self.sign)

    def slope_bound(self, t0: Array, t1: Array) -> Array:
        """The largest slope ``|k| / R`` across the axis between the axial distances ``t0`` and ``t1`` (at the least index)."""

        n = self.profile.least(self.z0 + self.sign * t0, self.z0 + self.sign * t1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.sqrt(self._k2()) / np.sqrt(n * n - self._k2())

    def turns(self, _profile: AxialProfile | None = None) -> NDArray[np.bool_]:
        """Rays whose angle to the axis reaches ``acos(AXIAL_MIN_COS)`` on their leg (near the turning point ``n = |k|``)."""

        n = self.profile.least(self.z0, self.z0 + self.sign * self.z_len)
        return ~(n * n * (1.0 - AXIAL_MIN_COS * AXIAL_MIN_COS) > self._k2())


def _axial_legs(profile: AxialProfile, x0: Array, y0: Array, z0: Array, d_local: Array, z_lo: float, z_hi: float
                ) -> tuple[AxialLegs, Array]:
    dz = d_local[:, 2]
    along = np.abs(dz) > ACROSS_AXIS
    sign = np.where(dz >= 0.0, 1.0, -1.0)
    n = profile.index_z(z0)
    z_end = np.where(sign > 0.0, z_hi, z_lo)
    z_len = np.maximum(0.0, (z_end - z0) * sign)
    return AxialLegs(profile, x0, y0, z0, n * d_local[:, 0], n * d_local[:, 1], z_len, sign), along


def planar_transport(e: Array, d_start: Array, d_end: Array, axis: Array) -> Array:
    """``e`` (unit, across ``d_start``) carried along rays of an axial medium from the directions ``d_start`` to ``d_end`` (scene
    frame, shape ``(rays, 3)``). Each ray lies in the plane of the axis and its start direction and turns in it without
    torsion, so parallel transport keeps the component across that plane and turns the one in it with the ray:
    ``e' = (e . b) b + (e . (d x b)) (d' x b)``, ``b`` the plane's unit normal. A ray along the axis stays straight."""

    b = np.cross(np.broadcast_to(axis, d_start.shape), d_start)
    norm = np.sqrt(np.sum(b * b, axis=-1, keepdims=True))
    straight = norm[:, 0] <= ACROSS_AXIS
    b = b / np.where(straight[:, None], 1.0, norm)
    across = np.sum(e * b, axis=-1, keepdims=True)
    within = np.sum(e * np.cross(d_start, b), axis=-1, keepdims=True)
    out = across * b + within * np.cross(d_end, b)
    out = np.where(straight[:, None], e, out)
    out = out - np.sum(out * d_end, axis=-1, keepdims=True) * d_end
    return out / np.sqrt(np.sum(out * out, axis=-1, keepdims=True))


# --------------------------------------------------------------------------------------- end faces across the axis or not

#: Bisection steps of the axial distance at which a ray meets an end face that is not a plane across the axis (a slanted
#: plane, a curved face): from a bracket of a few mm to the last bits of a double.
FACE_BISECTIONS = 64
#: How far (mm) the face's axial span is widened on both sides for the bracket.
FACE_BRACKET_MM = 1e-6


def face_z(face: GradientFaceG, x: Array, y: Array) -> Array:
    """The axial position of the end face at ``(x, y)`` (local frame): a plane ``z_p - (m_x (x - x_p) + m_y (y - y_p)) / m_z``, or
    a conic surface of revolution about the axis ``z_v + c s / (1 + sqrt(1 - (1 + k) c^2 s))``, ``s = x^2 + y^2``."""

    px, py, pz = face.point
    if face.kind == "plane":
        mx, my, mz = face.normal
        return pz - (mx * (x - px) + my * (y - py)) / mz
    s = x * x + y * y
    c = face.curvature
    with np.errstate(invalid="ignore"):
        root = np.sqrt(1.0 - (1.0 + face.conic) * c * c * s)
    return pz + c * s / (1.0 + root)


def face_normal(face: GradientFaceG, x: float, y: float) -> tuple[float, float, float]:
    """The unit normal (with a positive axial component) of the end face at ``(x, y)`` (local frame)."""

    if face.kind == "plane":
        return face.normal
    c, s = face.curvature, x * x + y * y
    root = math.sqrt(max(0.0, 1.0 - (1.0 + face.conic) * c * c * s))
    # (dz/dx = c x / root, dz/dy = c y / root: the normal (-dz/dx, -dz/dy, 1), times root)
    nx, ny, nz = -c * x, -c * y, root
    norm = math.sqrt(nx * nx + ny * ny + nz * nz)
    return nx / norm, ny / norm, nz / norm


def to_face(leg: Legs | AxialLegs, face: GradientFaceG) -> tuple[Legs | AxialLegs, NDArray[np.bool_]]:
    """Each ray's leg to where it meets ``face`` (``GradientFaceG``) ahead of it, and a mask of the rays for which that point
    is found. The face spans the axial positions ``[z_min, z_max]`` within the rod's radius, so a ray inside the rod lies
    before it at the near end of that span and beyond it at the far end; between them ``sign (z - face_z(x, y))`` grows
    by at least ``1 - slope s`` per mm of axial travel (``slope``: the face's largest ``|dz/dr|``; ``s``: a bound on the
    ray's slope across the axis, ``slope_bound``), so for ``slope s < 1`` it crosses the face exactly once and bisection
    (``FACE_BISECTIONS``) finds that crossing. A ray for which ``slope s >= 1`` (too oblique to the face to cross it once)
    or that is not before / beyond the face at the span's ends (it has left the rod's radius) is not found."""

    sign, z0 = leg.sign, leg.z0
    if face.kind == "plane" and face.slope == 0.0:  # (a plane across the axis: the axial distance to it, as ``legs`` takes it)
        return leg.with_length(np.maximum(0.0, (face.point[2] - z0) * sign)), np.ones(sign.shape, dtype=bool)
    # (the span widened by ``FACE_BRACKET_MM`` on both sides: a ray at its ends is not lost to rounding)
    near = np.where(sign > 0.0, face.z_min, -face.z_max) - FACE_BRACKET_MM
    far = np.where(sign > 0.0, face.z_max, -face.z_min) + FACE_BRACKET_MM
    lo = np.maximum(0.0, near - sign * z0)
    hi = np.maximum(lo, far - sign * z0)

    def beyond(t: Array) -> Array:
        x, y, z, _dx, _dy = leg.at(t)
        return sign * (z - face_z(face, x, y))

    with np.errstate(invalid="ignore"):
        ok = (beyond(lo) <= 0.0) & (beyond(hi) >= 0.0) & (leg.slope_bound(lo, hi) * face.slope < 1.0)
        for _ in range(FACE_BISECTIONS):
            mid = 0.5 * (lo + hi)
            before = beyond(mid) < 0.0
            lo, hi = np.where(before, mid, lo), np.where(before, hi, mid)
    return leg.with_length(np.where(ok, hi, leg.z_len)), ok


__all__ = ["GL_NODES", "AxialLegs", "AxialProfile", "Legs", "POINTS_PER_PERIOD", "Profile", "TWO_PI", "face_normal", "face_z", "legs",
           "planar_transport", "to_face", "transported"]
