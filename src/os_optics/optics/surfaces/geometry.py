"""Ray / surface intersection and aperture tests (vectorized, NumPy).

Every projection is ``math.vectors.dot`` (separately rounded products summed in index order): an intersection has the
same bits in every runtime (no ``einsum`` / ``matmul``, whose fused multiply-adds differ between native builds and
Pyodide)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ...config import NumericalConfig
from ...math.vectors import dot
from ...scene.model import ApertureG, SurfaceG
from .. import apertures
from . import freeform as ff
from . import sag as sg

Array = NDArray[np.float64]


@dataclass(frozen=True)
class Hits:
    """Per-ray intersection with one surface.

    ``t`` is ``inf`` where the ray does not hit the surface support; ``inside`` says whether the hit point lies
    within the clear aperture. Hits on the mount ring have ``inside=False``; rays beyond the ring miss (``t=inf``).
    """

    t: Array
    point: Array
    normal: Array  # unit, geometric normal (not oriented against the ray)
    inside: NDArray[np.bool_]
    u: Array
    v: Array


def aperture_inside(ap: ApertureG, u: Array, v: Array, tol: float) -> NDArray[np.bool_]:
    """Inclusive (within ``tol``) aperture membership for in-surface coordinates ``(u, v)`` (``apertures.inside``)."""
    return apertures.inside(ap, u, v, tol)


def aperture_miss_distance(ap: ApertureG, u: float, v: float) -> float:
    """Distance (mm) from the point ``(u, v)`` to the edge of an aperture (positive outside, negative inside;
    ``apertures.miss_distance``)."""
    return apertures.miss_distance(ap, u, v)


def _dilate(ap: ApertureG, margin: float) -> ApertureG:
    """Outer boundary of the mount: the aperture grown by ``margin`` (``apertures.dilate``; any central obstruction is kept as
    an obstruction hit, holes stay openings)."""
    return apertures.dilate(ap, margin)


def support_of(s: SurfaceG) -> ApertureG:
    """The surface's support: its mount outline when given, else the clear aperture grown by the mount ring."""
    return s.outline if s.outline is not None else _dilate(s.aperture, s.mount_margin)


def intersect(surface: SurfaceG, o: Array, d: Array, cfg: NumericalConfig) -> Hits:
    """Nearest valid intersection (``t > ray_min_advance_mm``) of each ray ``o + t d`` with ``surface``."""
    if surface.geometry == "plane":
        return _intersect_plane(surface, o, d, cfg)
    if surface.geometry == "sag":
        return _intersect_sag(surface, o, d, cfg)
    return _intersect_sphere(surface, o, d, cfg)


_SHAPE_CODE = {"none": 0, "circle": 1, "rectangle": 2}
#: Any other bounded support (ellipse, d_shape, polygon, or a support with holes): tested by ``apertures.inside`` per plane.
_OTHER_SHAPE = 3


def _support_code(ap: ApertureG) -> int:
    return _SHAPE_CODE[ap.shape] if ap.shape in _SHAPE_CODE and not ap.holes else _OTHER_SHAPE


class PlaneBatch:
    """All plane surfaces of a scene stacked into arrays: ray parameters for every (plane, ray) pair in one vectorized pass.

    Applies exactly the same support rule as ``_intersect_plane`` (clear aperture dilated by the mount ring); the tracer uses the
    resulting times only to choose the next surface and then evaluates the chosen surface with ``intersect``."""

    def __init__(self, surfaces: list[SurfaceG]) -> None:
        self.origin = np.array([s.origin for s in surfaces], dtype=np.float64).reshape(-1, 3)
        self.normal = np.array([s.normal for s in surfaces], dtype=np.float64).reshape(-1, 3)
        self.u_axis = np.array([s.u_axis for s in surfaces], dtype=np.float64).reshape(-1, 3)
        self.v_axis = np.array([s.v_axis for s in surfaces], dtype=np.float64).reshape(-1, 3)
        self.origin_u = dot(self.origin, self.u_axis)
        self.origin_v = dot(self.origin, self.v_axis)
        outer = [support_of(s) for s in surfaces]
        self._outer = outer
        self.shape = np.array([_support_code(a) for a in outer], dtype=np.int64)
        self.p1 = np.array([a.p1 for a in outer], dtype=np.float64)
        self.p2 = np.array([a.p2 for a in outer], dtype=np.float64)
        self.off_u = np.array([a.offset_u for a in outer], dtype=np.float64)
        self.off_v = np.array([a.offset_v for a in outer], dtype=np.float64)
        # (P, N) layout: every plane quantity as a contiguous (P, 1) column, so each product runs along the rays
        self._origin_cols = _columns(self.origin)
        self._normal_cols = _columns(self.normal)
        # the bounded supports, by shape: their in-plane coordinates are needed (an unbounded plane's support is everything)
        self._groups = [_SupportGroup(self, np.flatnonzero(self.shape == code), code) for code in (1, 2, _OTHER_SHAPE)]
        self._groups = [g for g in self._groups if g.rows.size]
        self._direction_key: tuple | None = None
        self._direction_terms: tuple[Array, list[tuple[Array, Array]]] | None = None

    def times(self, o: Array, d: Array, cfg: NumericalConfig) -> Array:
        """``(P, N)`` ray parameters, ``inf`` where the ray misses the plane's support.

        Every entry is computed with the operations, in the order, of ``_intersect_plane``: ``(origin - o) . n / (d . n)``
        and the in-plane coordinates ``u = ((o . u - origin . u) + t (d . u)) - offset`` as index-order sums of separately
        rounded products. The products of the directions alone (``d . n``, ``d . u``, ``d . v``) are kept while the
        directions are unchanged (a packet crossing observation planes): the same inputs give the same bits."""
        o_cols = [np.ascontiguousarray(o[:, k]) for k in range(3)]  # (N,): broadcast against the (P, 1) plane columns
        key = (d.shape, d.dtype.str, d.tobytes())
        if key != self._direction_key or self._direction_terms is None:
            d_cols = [np.ascontiguousarray(d[:, k]) for k in range(3)]
            self._direction_terms = (_column_dot(d_cols, self._normal_cols),
                                     [(_column_dot(d_cols, g.u_cols), _column_dot(d_cols, g.v_cols)) for g in self._groups])
            self._direction_key = key
        denom, direction_uv = self._direction_terms
        n0, n1, n2 = self._normal_cols
        c0, c1, c2 = self._origin_cols
        numer = (c0 - o_cols[0]) * n0  # (origin - o) . n, as _intersect_plane
        numer += (c1 - o_cols[1]) * n1
        numer += (c2 - o_cols[2]) * n2
        ok = np.abs(denom) > cfg.parallel_dot_tol
        t = np.where(ok, numer / np.where(ok, denom, 1.0), np.inf)
        valid = ok & (t > cfg.ray_min_advance_mm)
        t = np.where(valid, t, np.inf)
        if not self._groups:
            return t
        # in-plane coordinates of o + t d without materializing a (P, N, 3) array: u = (o - origin).u + t (d.u)
        ts = np.where(valid, t, 0.0)
        tol = cfg.aperture_edge_tol_mm
        for group, (du, dv) in zip(self._groups, direction_uv):
            rows = group.rows
            whole = rows.size == t.shape[0]
            ts_g = ts if whole else ts[rows]
            if group.code == _OTHER_SHAPE:  # (u, v) in each plane's frame: the regions answer about their own offsets
                u = (_column_dot(o_cols, group.u_cols) - group.origin_u) + ts_g * du
                v = (_column_dot(o_cols, group.v_cols) - group.origin_v) + ts_g * dv
                support = np.stack([apertures.inside(self._outer[int(row)], u[k], v[k], tol) for k, row in enumerate(rows)])
                if whole:
                    return np.where(valid & support, t, np.inf)
                t[rows] = np.where(valid[rows] & support, t[rows], np.inf)
                continue
            u = (_column_dot(o_cols, group.u_cols) - group.origin_u) + ts_g * du - group.off_u
            v = (_column_dot(o_cols, group.v_cols) - group.origin_v) + ts_g * dv - group.off_v
            if group.code == 1:
                support = (u * u + v * v) <= (group.p1 + tol) ** 2
            else:
                support = (np.abs(u) <= group.p1 + tol) & (np.abs(v) <= group.p2 + tol)
            if whole:
                return np.where(valid & support, t, np.inf)
            t[rows] = np.where(valid[rows] & support, t[rows], np.inf)
        return t


def _columns(rows: Array) -> list[Array]:
    """The three components of ``(P, 3)`` rows as contiguous ``(P, 1)`` columns."""
    return [np.ascontiguousarray(rows[:, k])[:, None] for k in range(3)]


def _column_dot(ray_cols: list[Array], plane_cols: list[Array]) -> Array:
    """``(P, N)`` dot products of N ray vectors (3 columns ``(N,)``) with P plane vectors (3 columns ``(P, 1)``): the
    index-order sum of separately rounded products, as ``math.vectors.dot``."""
    total = ray_cols[0] * plane_cols[0]
    total += ray_cols[1] * plane_cols[1]
    total += ray_cols[2] * plane_cols[2]
    return total


class _SupportGroup:
    """The planes of a ``PlaneBatch`` whose support has one bounded shape (1 circle, 2 rectangle, 3 any other region), as
    ``(P', 1)`` columns."""

    def __init__(self, batch: PlaneBatch, rows: NDArray[np.int64], code: int) -> None:
        self.rows, self.code = rows, code
        self.u_cols = _columns(batch.u_axis[rows])
        self.v_cols = _columns(batch.v_axis[rows])
        self.origin_u = batch.origin_u[rows][:, None]
        self.origin_v = batch.origin_v[rows][:, None]
        self.off_u = batch.off_u[rows][:, None]
        self.off_v = batch.off_v[rows][:, None]
        self.p1 = batch.p1[rows][:, None]
        self.p2 = batch.p2[rows][:, None]


class SphereBatch:
    """All spherical surfaces of a scene stacked into arrays: the ray parameter of every (sphere, ray) pair in one vectorized
    pass.

    Applies exactly the rule of ``_intersect_sphere`` (the nearer of the two roots that lies on the cap within the clear
    aperture grown by the mount ring); the tracer uses the resulting times only to choose the next surface and then
    evaluates the chosen surface with ``intersect``."""

    def __init__(self, surfaces: list[SurfaceG]) -> None:
        self.center = np.array([s.center for s in surfaces], dtype=np.float64).reshape(-1, 3)
        self.cap_axis = np.array([s.cap_axis for s in surfaces], dtype=np.float64).reshape(-1, 3)
        # (S, N) layout: every sphere quantity as a contiguous (S, 1) column, so each product runs along the rays
        self._center_cols = _columns(self.center)
        self._axis_cols = _columns(self.cap_axis)
        self._radius_sq = np.array([s.radius * s.radius for s in surfaces], dtype=np.float64)[:, None]
        # the support's radius without the edge tolerance: ``(aperture_radius + mount_margin) + tol`` is summed per call
        self._ring = np.array([s.aperture_radius + s.mount_margin for s in surfaces], dtype=np.float64)[:, None]

    def times(self, o: Array, d: Array, cfg: NumericalConfig) -> Array:
        """``(S, N)`` ray parameters, ``inf`` where the ray misses the sphere's cap or its support.

        Every entry is computed with the operations, in the order, of ``_intersect_sphere``: ``b = d . (o - c)`` and
        ``(o - c) . (o - c) - R R`` as index-order sums of separately rounded products, the cancellation-free roots
        ``q = -(b + sign(b) sqrt(b b - cc))`` and ``cc / q``, and for each root the hit point ``o + t d``, its axial
        coordinate on the cap axis and its distance from that axis, the first root tried first."""
        o0, o1, o2 = (np.ascontiguousarray(o[:, k]) for k in range(3))  # (N,): broadcast against the (S, 1) sphere columns
        d0, d1, d2 = (np.ascontiguousarray(d[:, k]) for k in range(3))
        c0, c1, c2 = self._center_cols
        a0, a1, a2 = self._axis_cols
        oc0, oc1, oc2 = o0 - c0, o1 - c1, o2 - c2
        b = d0 * oc0  # d . (o - c), as math.vectors.dot
        b += d1 * oc1
        b += d2 * oc2
        cc = oc0 * oc0
        cc += oc1 * oc1
        cc += oc2 * oc2
        cc -= self._radius_sq
        disc = b * b - cc
        has = disc >= 0.0
        sq = np.sqrt(np.where(has, disc, 0.0))
        q = -(b + np.where(b >= 0.0, sq, -sq))
        nonzero = q != 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            roots = (np.where(has, q, np.inf), np.where(has & nonzero, cc / np.where(nonzero, q, 1.0), np.inf))
        reach = self._ring + cfg.aperture_edge_tol_mm
        best: Array | None = None
        for t in roots:
            t = np.where(np.isfinite(t) & (t > cfg.ray_min_advance_mm), t, np.inf)
            finite = np.isfinite(t)
            ts = np.where(finite, t, 0.0)
            r0, r1, r2 = (o0 + ts * d0) - c0, (o1 + ts * d1) - c1, (o2 + ts * d2) - c2  # the hit point from the centre
            axial = r0 * a0
            axial += r1 * a1
            axial += r2 * a2
            x0, x1, x2 = r0 - axial * a0, r1 - axial * a1, r2 - axial * a2
            rho_sq = x0 * x0
            rho_sq += x1 * x1
            rho_sq += x2 * x2
            valid = finite & (axial >= 0.0) & (np.sqrt(rho_sq) <= reach)
            best = np.where(valid, t, np.inf) if best is None else np.where(valid & (t < best), t, best)
        assert best is not None
        return best


def _intersect_plane(s: SurfaceG, o: Array, d: Array, cfg: NumericalConfig) -> Hits:
    n = s.normal
    denom = dot(d, n)
    numer = dot(s.origin - o, n)
    ok = np.abs(denom) > cfg.parallel_dot_tol
    safe = np.where(ok, denom, 1.0)
    t = np.where(ok, numer / safe, np.inf)
    valid = ok & (t > cfg.ray_min_advance_mm)
    t = np.where(valid, t, np.inf)
    point = o + np.where(valid, t, 0.0)[:, None] * d
    rel = point - s.origin
    u, v = dot(rel, s.u_axis), dot(rel, s.v_axis)
    inside = aperture_inside(s.aperture, u, v, cfg.aperture_edge_tol_mm)
    if apertures.bounded(s.aperture):
        # support = clear aperture dilated by the mount ring; rays beyond it miss the element and pass by
        support = aperture_inside(support_of(s), u, v, cfg.aperture_edge_tol_mm)
        valid &= support
        t = np.where(valid, t, np.inf)
        inside = np.where(valid, inside, True)
    normals = np.broadcast_to(n, o.shape).copy()
    return Hits(t, point, normals, inside, u, v)


def _intersect_sphere(s: SurfaceG, o: Array, d: Array, cfg: NumericalConfig) -> Hits:
    oc = o - s.center
    b = dot(d, oc)
    cc = dot(oc, oc) - s.radius * s.radius
    disc = b * b - cc
    has = disc >= 0.0
    sq = np.sqrt(np.where(has, disc, 0.0))
    # cancellation-free roots: q = -(b + sign(b) sq); t1 = q, t2 = cc / q
    q = -(b + np.where(b >= 0.0, sq, -sq))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_a = np.where(has, q, np.inf)
        t_b = np.where(has & (q != 0.0), cc / np.where(q != 0.0, q, 1.0), np.inf)
    best_t = np.full(o.shape[0], np.inf)
    best_pt = np.zeros_like(o)
    best_in = np.ones(o.shape[0], dtype=bool)
    best_u = np.zeros(o.shape[0])
    best_v = np.zeros(o.shape[0])
    for t in (t_a, t_b):
        t = np.where(np.isfinite(t) & (t > cfg.ray_min_advance_mm), t, np.inf)
        pt = o + np.where(np.isfinite(t), t, 0.0)[:, None] * d
        rel = pt - s.center
        axial = dot(rel, s.cap_axis)
        on_cap = axial >= 0.0
        trans = rel - axial[:, None] * s.cap_axis
        rho = np.sqrt(dot(trans, trans))
        inside = rho <= s.aperture_radius + cfg.aperture_edge_tol_mm
        valid = np.isfinite(t) & on_cap & (rho <= s.aperture_radius + s.mount_margin + cfg.aperture_edge_tol_mm)
        better = valid & (t < best_t)
        best_t = np.where(better, t, best_t)
        best_pt = np.where(better[:, None], pt, best_pt)
        best_in = np.where(better, inside, best_in)
        best_u = np.where(better, dot(rel, s.u_axis), best_u)
        best_v = np.where(better, dot(rel, s.v_axis), best_v)
    rel = best_pt - s.center
    # A ray that misses the cap keeps best_pt at the origin, so (best_pt - center) / R would be an arbitrary non-unit
    # vector; such a row gets the cap axis instead: a finite unit normal (its t is inf, so it is never used for physics).
    hit = np.isfinite(best_t)
    nrm = np.where(hit[:, None], rel / s.radius, s.cap_axis)
    return Hits(best_t, best_pt, nrm, best_in, best_u, best_v)


def _intersect_sag(s: SurfaceG, o: Array, d: Array, cfg: NumericalConfig) -> Hits:
    """Nearest valid intersection with a conic / even-asphere sag surface (``sag.py``).

    The rays are expressed in the surface frame from their point of closest approach to the vertex (small local
    coordinates: no cancellation from a distant ray origin), ``r = p + tau d``. A conic-only profile is solved in closed
    form (the quadric roots on the vertex branch, nearest admissible first). With even-asphere, axicon or two-variable
    terms, ``F(tau) = z(tau) - sag(s(tau))`` (``sag(a(tau), b(tau))``) is sampled at ``sag_bracket_samples`` points across the
    ray's crossing of the support's bounding box, and the sign changes, nearest first, are refined by Newton steps kept
    inside the bracket (bisection otherwise) until one lies inside the support. Hits outside the support (clear aperture
    dilated by the mount ring, measured on the transverse coordinates ``(a, b)``) or outside the profile's domain are misses
    (``t = inf``). On a two-variable surface a ray that grazes it between two samples is split at its closest approach
    (``_split_grazing_roots``), and a ray whose bracket does not converge is counted (``SolverStats``) and reported
    (``SAG_INTERSECTION_NOT_CONVERGED``), not taken for a real miss."""
    rel = o - s.origin
    shift = -dot(rel, d)
    p = rel + shift[:, None] * d
    a0, b0, z0 = dot(p, s.u_axis), dot(p, s.v_axis), dot(p, s.normal)
    da, db, dz = dot(d, s.u_axis), dot(d, s.v_axis), dot(d, s.normal)
    tol = cfg.aperture_edge_tol_mm
    support = support_of(s)
    tau_min = cfg.ray_min_advance_mm - shift
    if s.freeform is not None or s.coefficients or s.linear != 0.0:
        tau = _sag_polynomial_root(s, (a0, b0, z0), (da, db, dz), tau_min, support, cfg)
    else:
        tau = np.full(o.shape[0], np.inf)
        for root in sg.conic_roots(s, a0, b0, z0, da, db, dz):
            finite = np.isfinite(root)
            rs = np.where(finite, root, 0.0)
            ok = (finite & (root > tau_min) & sg.on_conic_branch(s, z0 + rs * dz)
                  & aperture_inside(support, a0 + rs * da, b0 + rs * db, tol))
            tau = np.where(ok & (root < tau), root, tau)
    hit = np.isfinite(tau)
    ts = np.where(hit, tau, 0.0)
    t = np.where(hit, shift + ts, np.inf)
    a = np.where(hit, a0 + ts * da, 0.0)
    b = np.where(hit, b0 + ts * db, 0.0)
    point = o + np.where(hit, t, 0.0)[:, None] * d
    inside = np.where(hit, aperture_inside(s.aperture, a, b, tol), True)
    # a miss keeps a finite unit normal (the axis), never used for physics
    normals = np.where(hit[:, None], sg.local_normals(s, a, b), s.normal)
    return Hits(t, point, normals, inside, a, b)


def _slab(x0: Array, dx: Array, lo: float, hi: float) -> tuple[Array, Array]:
    """Ray-parameter interval in which ``x0 + tau dx`` lies in ``[lo, hi]`` (empty: ``enter > leave``)."""
    moving = dx != 0.0
    step = np.where(moving, dx, 1.0)
    t1, t2 = (lo - x0) / step, (hi - x0) / step
    within = (x0 >= lo) & (x0 <= hi)
    enter = np.where(moving, np.minimum(t1, t2), np.where(within, -np.inf, np.inf))
    leave = np.where(moving, np.maximum(t1, t2), np.where(within, np.inf, -np.inf))
    return enter, leave


def _sag_residual(s: SurfaceG, a: Array, b: Array, z: Array) -> Array:
    """``F = z - sag(s(a, b))`` (NaN outside the profile's domain)."""
    if s.freeform is not None:
        return z - ff.sag_at(s, a, b)
    zs, _ = sg.sag(s, sg.transverse_s(s, a, b))
    return z - zs


def _sag_polynomial_root(s: SurfaceG, p: tuple[Array, Array, Array], d: tuple[Array, Array, Array], tau_min: Array,
                         support: ApertureG, cfg: NumericalConfig) -> Array:
    """Nearest root ``tau > tau_min`` of ``F`` inside the support (``inf``: none), see ``_intersect_sag``."""
    a0, b0, z0 = p
    da, db, dz = d
    tol = cfg.aperture_edge_tol_mm
    u_lo, u_hi, v_lo, v_hi = apertures.bounds(support)
    a_lo, a_hi = u_lo - tol, u_hi + tol
    b_lo, b_hi = v_lo - tol, v_hi + tol
    a_far, b_far = max(abs(a_lo), abs(a_hi)), max(abs(b_lo), abs(b_hi))
    z_lo, z_hi = sg.sag_bounds(s, a_far if s.cylindrical else float(np.sqrt(a_far * a_far + b_far * b_far)))
    enter, leave = np.full(a0.shape, -np.inf), np.full(a0.shape, np.inf)
    for x0, dx, lo, hi in ((a0, da, a_lo, a_hi), (b0, db, b_lo, b_hi), (z0, dz, z_lo - tol, z_hi + tol)):
        e, lv = _slab(x0, dx, lo, hi)
        enter, leave = np.maximum(enter, e), np.minimum(leave, lv)
    start = np.maximum(enter, tau_min)
    has = np.isfinite(start) & np.isfinite(leave) & (start < leave)
    count = cfg.sag_bracket_samples
    frac = np.linspace(0.0, 1.0, count)
    start = np.where(has, start, 0.0)
    span = np.where(has, leave - start, 0.0)
    taus = start[:, None] + span[:, None] * frac[None, :]
    f = _sag_residual(s, a0[:, None] + taus * da[:, None], b0[:, None] + taus * db[:, None], z0[:, None] + taus * dz[:, None])
    if s.freeform is not None:
        taus, f = _split_grazing_roots(s, p, d, taus, f, has, cfg)
    f0, f1 = f[:, :-1], f[:, 1:]
    change = has[:, None] & np.isfinite(f0) & np.isfinite(f1) & (f0 * f1 <= 0.0)
    result = np.full(a0.shape, np.inf)
    first = np.zeros(a0.shape, dtype=np.int64)
    columns = np.arange(taus.shape[1] - 1)
    pending = change.any(axis=1)
    while True:
        candidates = change & (columns[None, :] >= first[:, None]) & pending[:, None]
        pending &= candidates.any(axis=1)
        rows = np.flatnonzero(pending)
        if rows.size == 0:
            return result
        j = np.argmax(candidates[rows], axis=1)
        root = _refine_sag_root(s, (a0[rows], b0[rows], z0[rows]), (da[rows], db[rows], dz[rows]),
                                (taus[rows, j], taus[rows, j + 1]), (f[rows, j], f[rows, j + 1]), cfg)
        rs = np.where(np.isfinite(root), root, 0.0)
        ok = (np.isfinite(root) & (root > tau_min[rows])
              & aperture_inside(support, a0[rows] + rs * da[rows], b0[rows] + rs * db[rows], tol))
        result[rows[ok]] = root[ok]
        pending[rows[ok]] = False
        first[rows] = j + 1


def _split_grazing_roots(s: SurfaceG, p: tuple[Array, Array, Array], d: tuple[Array, Array, Array], taus: Array, f: Array,
                         has: NDArray[np.bool_], cfg: NumericalConfig) -> tuple[Array, Array]:
    """``(taus, f)`` with a sample added where a ray crosses a two-variable surface twice between two samples: ``F`` keeps its
    sign at both but turns back toward 0 from both ends, so its extremum between them (bisection on ``dF/dtau``) is
    evaluated, and a sign change there adds a sample at it. A row with fewer such samples repeats its last sample."""
    a0, b0, z0 = p
    da, db, dz = d
    slopes = ff.residual_slope(s, a0[:, None] + taus * da[:, None], b0[:, None] + taus * db[:, None], z0[:, None] + taus * dz[:, None],
                               da[:, None], db[:, None], dz[:, None])[1]
    f0, f1 = f[:, :-1], f[:, 1:]
    with np.errstate(invalid="ignore"):
        turning = has[:, None] & (f0 * f1 > 0.0) & (f0 * slopes[:, :-1] < 0.0) & (f1 * slopes[:, 1:] > 0.0)
    added: dict[int, list[tuple[int, float, float]]] = {}
    for row, column in zip(*np.nonzero(turning), strict=True):
        low, high = float(taus[row, column]), float(taus[row, column + 1])
        sign = float(f[row, column])
        rs = np.array([row])
        for _ in range(cfg.sag_max_iterations):
            middle = 0.5 * (low + high)
            value, slope = ff.residual_slope(s, a0[rs] + middle * da[rs], b0[rs] + middle * db[rs], z0[rs] + middle * dz[rs], da[rs], db[rs], dz[rs])
            if not (math.isfinite(float(value[0])) and math.isfinite(float(slope[0]))):
                break
            if float(value[0]) * sign <= 0.0:
                added.setdefault(int(row), []).append((int(column), middle, float(value[0])))
                break
            if float(slope[0]) * sign < 0.0:
                low = middle
            else:
                high = middle
            if high - low <= cfg.sag_intersection_tol_mm:
                break
    if not added:
        return taus, f
    columns = taus.shape[1] + max(len(v) for v in added.values())
    grown_t, grown_f = np.empty((taus.shape[0], columns)), np.empty((taus.shape[0], columns))
    grown_t[:, :taus.shape[1]], grown_f[:, :taus.shape[1]] = taus, f
    grown_t[:, taus.shape[1]:], grown_f[:, taus.shape[1]:] = taus[:, -1:], f[:, -1:]
    for row, splits in added.items():
        where = [column + 1 for column, _, _ in splits]
        t_row = np.insert(taus[row], where, [t for _, t, _ in splits])
        f_row = np.insert(f[row], where, [v for _, _, v in splits])
        grown_t[row, :t_row.size], grown_f[row, :f_row.size] = t_row, f_row
    return grown_t, grown_f


def _refine_sag_root(s: SurfaceG, p: tuple[Array, Array, Array], d: tuple[Array, Array, Array], bracket: tuple[Array, Array],
                     values: tuple[Array, Array], cfg: NumericalConfig) -> Array:
    """Root of ``F`` in ``[lo, hi]`` (``F(lo) F(hi) <= 0``): Newton steps that stay strictly inside the shrinking bracket,
    bisection otherwise; converged when a step moves ``tau`` by at most ``sag_intersection_tol_mm`` (``inf``: not
    converged within ``sag_max_iterations``)."""
    a0, b0, z0 = p
    da, db, dz = d
    lo, hi = bracket
    f_lo, f_hi = values
    exact_lo, exact_hi = f_lo == 0.0, f_hi == 0.0
    denominator = f_hi - f_lo
    secant = np.where(denominator != 0.0, lo - f_lo * (hi - lo) / np.where(denominator != 0.0, denominator, 1.0), 0.5 * (lo + hi))
    tau = np.where(exact_lo, lo, np.where(exact_hi, hi, np.clip(secant, lo, hi)))
    done = exact_lo | exact_hi
    iterations = 0
    for _ in range(cfg.sag_max_iterations):
        if bool(done.all()):
            break
        iterations += 1
        a, b, z = a0 + tau * da, b0 + tau * db, z0 + tau * dz
        if s.freeform is not None:
            f, slope = ff.residual_slope(s, a, b, z, da, db, dz)
        else:
            ss = sg.transverse_s(s, a, b)
            zs, _ = sg.sag(s, ss)
            f = z - zs
            big_s, _ = sg.sqrt_term(s, ss)
            slope = dz - sg.sag_dtau(s, a, b, da, db, ss, big_s)
        finite = np.isfinite(f)
        same = finite & ~done & (np.sign(f) == np.sign(f_lo))
        other = finite & ~done & ~same
        lo, f_lo = np.where(same, tau, lo), np.where(same, f, f_lo)
        hi, f_hi = np.where(other, tau, hi), np.where(other, f, f_hi)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            newton = tau - f / slope
        use_newton = finite & np.isfinite(newton) & (newton > lo) & (newton < hi)
        nxt = np.where(use_newton, newton, 0.5 * (lo + hi))
        exact = finite & (f == 0.0)
        nxt = np.where(exact, tau, nxt)
        converged = exact | (np.abs(nxt - tau) <= cfg.sag_intersection_tol_mm)
        tau = np.where(done, tau, nxt)
        done = done | converged
    if s.freeform is not None:
        ff.record_solve(s, a0 + tau * da, b0 + tau * db, z0 + tau * dz, done, iterations)
    return np.where(done, tau, np.inf)
