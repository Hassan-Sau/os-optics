"""A lower bound of the share of a beam that a stopping surface certainly takes before a plane its chief ray crosses.

``decisions.taken_before`` bounds from above the share of a beam that the surface the chief ray meets next takes before an
observation plane (``[P - S, P]``). Where that surface stops every light it meets from the side the beam arrives from (the
back of an uncoated mirror, an absorber), the plane cannot receive the lines of the beam that certainly meet it first, so
the plane's power is at most ``P - L`` with ``L`` the share of those lines (``certain_meeting_share``): a minimum the beam
provably misses is then unmet, not undecided.

The beam's energy flows along lines that keep their place in its profile: the line ``(xi, eta)`` (density
``exp(-2 (xi^2 + eta^2))``) lies at ``xi w_x(s) b_x + eta w_y(s) b_y`` off the chief ray ``start + s d``. For a cell of the
profile and a stretch ``[s0, s1]`` of the chief ray every point of its lines lies in the box of ``s``, ``x = xi w`` and
``y = eta w`` (``w`` between the axis's narrowest and widest radius on the stretch), on which the plane's signed distance and
the surface's coordinates are linear: their exact ranges (``tracer._linear_ranges_of``). A surface function ``F`` vanishes on
the surface and has the sign of the side it is met from: a plane's signed distance, ``z - sag`` of a sag surface (``+`` its
``+normal`` side), ``sign(R) (|p - C|^2 - R^2)`` of a sphere (the side its normal ``(p - C) / R`` points to). Every line of
the cell certainly meets the surface first on the stopping side, before the plane, when on stretches cut along the chief ray
* every stretch before some stretch ``j`` holds no point where a line may meet the surface (``F`` keeps its sign over the
  box, or the box misses its support or the profile's domain), and the stretch just before ``j`` lies where every line of
  the beam already left its start surface and has ``F`` strictly of the stopping side's sign;
* from ``j`` on, every box up to one with ``F`` strictly of the other sign lies wholly on the support (and the profile's
  domain) and wholly before the plane.
Then ``F`` changes sign along each line between those boxes, on the support: its first meeting with the surface is there,
from the stopping side, before the plane. A cell that is not certain is cut in four up to ``LOWER_CELL_DEPTH`` times; the
certain cells' Gaussian masses (products of the axes' exact ``erfc`` shares) add up to ``L``. Lines beyond ``K`` radii are
never counted. Every box is weighed element by element (the same bits in every runtime)."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from ...config import NumericalConfig
from ...math.exact_erf import erfc_float
from ...math.exact_trig import exp_float
from ...math.vectors import dot
from ...scene.model import SurfaceG
from ...units import NM_TO_MM
from ..surfaces.geometry import support_of
from .. import apertures
from .decisions import Leg, flow_stretch

Array = NDArray[np.float64]

#: the profile within ``K`` radii in LOWER_GRID x LOWER_GRID cells, each uncertain one cut in four up to LOWER_CELL_DEPTH times
LOWER_GRID = 16
LOWER_CELL_DEPTH = 3
#: the chief ray's stretch from the lines' start to the plane, cut into LOWER_ZONE_STRETCHES equal stretches; the part of it
#: where a line may meet the surface (with a stretch on either side) cut again into LOWER_STRETCHES
LOWER_ZONE_STRETCHES = 256
LOWER_STRETCHES = 1024
#: ``possible_meeting_share`` cuts a cell again only while that weighs at most this many boxes
LOWER_BOX_BUDGET = 1_500_000
#: the stopping side of a surface: either side (an absorber)
EITHER_SIDE = 0


def _below(z: float) -> float:
    """The share of a Gaussian axis profile (density ``exp(-2 x^2)``) below ``z`` (radii)."""
    tail = 0.5 * erfc_float(math.sqrt(2.0) * abs(z))
    return tail if z <= 0.0 else 1.0 - tail


def _mass(below: list[float], lo: NDArray[np.intp], hi: NDArray[np.intp]) -> Array:
    """The share of a Gaussian axis profile between the grid edges ``lo`` and ``hi`` (indices; ``below``: ``_below`` of
    every edge, computed once per grid)."""
    return np.array([max(below[b] - below[a], 0.0) for a, b in zip(lo.tolist(), hi.tolist())])


def certain_meeting_share(surf: SurfaceG, leg: Leg, k_sigma: float | None, stop_side: int, cfg: NumericalConfig) -> float:
    """A lower bound of the fraction of the beam of ``leg`` (the leg to the observation plane: its end plane) whose
    energy-flow lines certainly meet ``surf`` first, from its side ``stop_side`` (+1 / -1: the sign of ``F`` there;
    ``EITHER_SIDE``: any), before the plane (module docstring). 0 where nothing is certain (no stated ``K``, a leg
    without a finite end plane, a support other than a circle or a rectangle, a surface with holes, no stretch)."""
    share = _meeting_share(surf, leg, k_sigma, stop_side, cfg, False)
    return 0.0 if share is None else share


def possible_meeting_share(surf: SurfaceG, leg: Leg, k_sigma: float | None, cfg: NumericalConfig) -> float | None:
    """An upper bound of the fraction of the beam of ``leg`` whose energy-flow lines may meet ``surf`` before the plane the
    leg ends at: the mass of the cells of the profile within ``K`` radii one of whose boxes, on a stretch of the chief ray
    from where the lines may leave their start surface to where they may still lie before the plane, may lie before the
    plane, on the surface's support (and the profile's domain) with ``F`` able to vanish (cut in four up to
    ``LOWER_CELL_DEPTH`` times; the cells left count whole), plus the profile beyond the ``K`` square, ``2 exp(-2 K^2)``.
    A surface beside the chief ray that a pupil ray meets first takes at most that. ``None``: not bounded this way (no
    stated ``K``, a leg without a finite end plane, a support other than a circle or a rectangle, no stretch)."""
    return _meeting_share(surf, leg, k_sigma, EITHER_SIDE, cfg, True)


def _meeting_share(surf: SurfaceG, leg: Leg, k_sigma: float | None, stop_side: int, cfg: NumericalConfig, possible: bool) -> float | None:
    """``certain_meeting_share`` (``possible`` false) or ``possible_meeting_share``; ``None``: not computed."""
    from .tracer import _box_terms, _linear_ranges_of, _radius_ranges, _sag_ranges_between  # noqa: PLC0415 (the tracer imports this module)

    if k_sigma is None or not math.isfinite(leg.length) or leg.end_offset is None or surf.geometry not in ("plane", "sag", "sphere"):
        return None
    support = support_of(surf) if surf.geometry != "sphere" else None
    if support is not None and support.shape not in ("circle", "rectangle"):
        if not possible or not apertures.bounded(support):
            return None
        # (a share from above may take a larger support: the circle about the region's offset that holds it)
        support = replace(support, shape="circle", p1=apertures.reach(support), p2=0.0, obstruction=0.0, holes=())
    if not possible and (surf.aperture.holes or (support is not None and support.holes)):
        return None  # (a hole passes the lines that reach it: no line on the support is certain to meet the surface)
    d, g = leg.d, leg.beam
    dn_t = float(dot(d, leg.end_normal))
    if abs(dn_t) <= cfg.parallel_dot_tol:
        return None
    start = flow_stretch(leg, leg.start_normal, leg.start_offset, k_sigma, cfg)
    end = flow_stretch(leg, leg.end_normal, leg.end_offset, k_sigma, cfg)
    if start is None or end is None or not end[1] > start[0]:
        return None
    outside = 2.0 * exp_float(-2.0 * k_sigma * k_sigma) if possible else 0.0
    s_from, s_lo, s_hi = start[1], start[0], end[1]
    sigma = 1.0 if dn_t > 0.0 else -1.0
    scale = g.wavelength_nm * NM_TO_MM / (math.pi * g.n)
    tol = cfg.aperture_edge_tol_mm
    base = leg.start

    def coefficients(axis: Array, offset: float) -> tuple[float, float, float, float]:
        return (offset, float(dot(axis, d)), float(dot(axis, g.bx)), float(dot(axis, g.by)))

    plane_c = coefficients(leg.end_normal, float(dot(leg.end_normal, base - (leg.start + leg.end_offset * d))))
    if surf.geometry == "sphere":
        rel = base - surf.center
        comps = [coefficients(np.eye(3)[k], float(rel[k])) for k in range(3)]
        axial_c = coefficients(surf.cap_axis, float(dot(surf.cap_axis, rel)))
        radius = abs(surf.radius)
        rim = min(surf.aperture_radius + surf.mount_margin, radius)
        orientation = 1.0 if surf.radius > 0.0 else -1.0
    else:
        rel = base - surf.origin
        a_c = coefficients(surf.u_axis, float(dot(surf.u_axis, rel)))
        b_c = coefficients(surf.v_axis, float(dot(surf.v_axis, rel)))
        z_c = coefficients(surf.normal, float(dot(surf.normal, rel)))
        k1c2 = (1.0 + surf.conic) * surf.curvature * surf.curvature
        domain = 1.0 / math.sqrt(k1c2) if surf.geometry == "sag" and k1c2 > 0.0 and surf.freeform is None else math.inf

    fine = LOWER_GRID * (1 << LOWER_CELL_DEPTH)
    edges = -k_sigma + np.arange(fine + 1) * (2.0 * k_sigma / fine)
    cuts = s_lo + np.arange(LOWER_ZONE_STRETCHES + 1) * ((s_hi - s_lo) / LOWER_ZONE_STRETCHES)

    def spans(lo: Array, hi: Array, narrow_r: Array, wide_r: Array) -> tuple[Array, Array]:
        corners = (lo * narrow_r, lo * wide_r, hi * narrow_r, hi * wide_r)
        return (np.minimum(np.minimum(corners[0], corners[1]), np.minimum(corners[2], corners[3])),
                np.maximum(np.maximum(corners[0], corners[1]), np.maximum(corners[2], corners[3])))

    def squares(lo: Array, hi: Array) -> tuple[Array, Array]:
        least = np.where((lo <= 0.0) & (hi >= 0.0), 0.0, np.minimum(lo * lo, hi * hi))
        return least, np.maximum(lo * lo, hi * hi)

    def boxes(cells: NDArray[np.intp]):
        """Per cell and stretch of ``cuts``: whether a line may meet the surface there, ``F`` strictly positive, strictly
        negative, and the box wholly on the support (and domain) and before the plane (each ``(cells, stretches)``).
        The cells' quantities are columns ``(cells, 1)`` and the stretches' rows ``(1, stretches)``: every box value is the
        same elementwise expression of its cell's and its stretch's, broadcast to ``(cells, stretches)``."""
        n_c, n_s = len(cells), len(cuts) - 1
        s0_all, s1_all = cuts[:-1], cuts[1:]
        narrow_x, wide_x = _radius_ranges(g.qx, g.m2x, scale, s0_all - leg.first, s1_all - leg.first)
        narrow_y, wide_y = _radius_ranges(g.qy, g.m2y, scale, s0_all - leg.first, s1_all - leg.first)
        s0, s1 = s0_all[None, :], s1_all[None, :]
        x = spans(edges[cells[:, 0]][:, None], edges[cells[:, 1]][:, None], narrow_x[None, :], wide_x[None, :])
        y = spans(edges[cells[:, 2]][:, None], edges[cells[:, 3]][:, None], narrow_y[None, :], wide_y[None, :])
        terms = _box_terms((s0, s1), x, y)
        p_lo, p_hi = _linear_ranges_of(plane_c, terms)
        before = (p_hi if sigma > 0.0 else -p_lo) < -tol
        ahead = (p_lo if sigma > 0.0 else -p_hi) <= tol  # (part of the box may lie before the plane)
        if surf.geometry == "sphere":
            r2_lo, r2_hi = np.zeros_like(s0), np.zeros_like(s0)
            for c in comps:
                lo, hi = _linear_ranges_of(c, terms)
                q_lo, q_hi = squares(lo, hi)
                r2_lo, r2_hi = r2_lo + q_lo, r2_hi + q_hi
            ax_lo, ax_hi = _linear_ranges_of(axial_c, terms)
            f_lo, f_hi = r2_lo - radius * radius, r2_hi - radius * radius
            if orientation < 0.0:
                f_lo, f_hi = -f_hi, -f_lo
            ax2_lo, ax2_hi = squares(ax_lo, ax_hi)
            on = (ax_lo >= tol) & (r2_hi - ax2_lo <= (rim - tol) * (rim - tol))
            maybe = (ax_hi >= -tol) & (r2_lo - ax2_hi <= (rim + tol) * (rim + tol))
        else:
            a_lo, a_hi = _linear_ranges_of(a_c, terms)
            b_lo, b_hi = _linear_ranges_of(b_c, terms)
            du_far = np.maximum(np.abs(a_lo - support.offset_u), np.abs(a_hi - support.offset_u))
            dv_far = np.maximum(np.abs(b_lo - support.offset_v), np.abs(b_hi - support.offset_v))
            if support.shape == "circle":
                on = du_far * du_far + dv_far * dv_far <= (support.p1 - tol) * (support.p1 - tol)
                du = np.maximum(np.maximum(support.offset_u - a_hi, a_lo - support.offset_u), 0.0)
                dv = np.maximum(np.maximum(support.offset_v - b_hi, b_lo - support.offset_v), 0.0)
                maybe = du * du + dv * dv <= (support.p1 + tol) * (support.p1 + tol)
            else:
                on = (du_far <= support.p1 - tol) & (dv_far <= support.p2 - tol)
                maybe = ((a_hi >= support.offset_u - support.p1 - tol) & (a_lo <= support.offset_u + support.p1 + tol)
                         & (b_hi >= support.offset_v - support.p2 - tol) & (b_lo <= support.offset_v + support.p2 + tol))
            z_lo, z_hi = _linear_ranges_of(z_c, terms)
            if surf.geometry == "plane":
                f_lo, f_hi = z_lo, z_hi
            else:
                near_a = np.where((a_lo <= 0.0) & (a_hi >= 0.0), 0.0, np.minimum(np.abs(a_lo), np.abs(a_hi)))
                far_a = np.maximum(np.abs(a_lo), np.abs(a_hi))
                if surf.cylindrical:
                    rho_lo, rho_hi = near_a, far_a
                else:
                    near_b = np.where((b_lo <= 0.0) & (b_hi >= 0.0), 0.0, np.minimum(np.abs(b_lo), np.abs(b_hi)))
                    far_b = np.maximum(np.abs(b_lo), np.abs(b_hi))
                    rho_lo, rho_hi = np.sqrt(near_a * near_a + near_b * near_b), np.sqrt(far_a * far_a + far_b * far_b)
                sag_lo, sag_hi = (v.reshape(rho_lo.shape) for v in _sag_ranges_between(surf, rho_lo.ravel(), rho_hi.ravel()))
                f_lo, f_hi = z_lo - sag_hi, z_hi - sag_lo
                on = on & (rho_hi < domain - tol) & np.isfinite(f_lo) & np.isfinite(f_hi)
                maybe = maybe & (rho_lo <= domain + tol)
        meet = maybe & (f_lo <= tol) & (f_hi >= -tol)
        plus, minus = f_lo > tol, f_hi < -tol
        return tuple(v.reshape(n_c, n_s) for v in (meet, plus, minus, on & before, meet & ahead))

    def certain(cells: NDArray[np.intp]) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
        """Whether every line of each cell certainly meets the surface first, from the stopping side, before the plane; and
        whether a line of it may meet the surface at all (a cell where none may is never cut)."""
        meet, plus, minus, on, _ = boxes(cells)
        n_c, n_s = meet.shape
        rows = np.arange(n_c)
        has = meet.any(axis=1)
        first = np.argmax(meet, axis=1)  # (the first stretch where a line may meet the surface)
        entry = np.maximum(first - 1, 0)
        ok = has & (first >= 1) & (cuts[entry] >= s_from)  # (every line already left its start surface there)
        if stop_side > 0:
            side = np.where(plus[rows, entry], 1, 0)
        elif stop_side < 0:
            side = np.where(minus[rows, entry], -1, 0)
        else:
            side = np.where(plus[rows, entry], 1, np.where(minus[rows, entry], -1, 0))
        ok &= side != 0
        index = np.arange(n_s)[None, :]
        leave = np.where(side[:, None] > 0, minus, plus) & (index >= first[:, None])
        ok &= leave.any(axis=1)
        last = np.argmax(leave, axis=1)  # (the first box from there on with the other side's sign throughout)
        bad = np.cumsum(~on, axis=1)
        before_first = np.where(first >= 1, bad[rows, entry], 0)
        return ok & (bad[rows, last] - before_first == 0), has

    step = (1 << LOWER_CELL_DEPTH)
    ii, jj = np.meshgrid(np.arange(LOWER_GRID) * step, np.arange(LOWER_GRID) * step, indexing="ij")
    cells = np.stack([ii.ravel(), ii.ravel() + step, jj.ravel(), jj.ravel() + step], axis=1).astype(np.intp)
    # the zone of the chief ray where a line of the profile may meet the surface, from a coarse cut, then cut finely there
    zone = boxes(cells)[0].any(axis=0)
    if not bool(zone.any()):
        return outside
    if not possible:  # (a share from above needs no fine cut: a coarse box is only wider)
        first_z = int(np.argmax(zone))
        last_z = len(zone) - 1 - int(np.argmax(zone[::-1]))
        z0, z1 = cuts[max(first_z - 1, 0)], cuts[min(last_z + 2, len(cuts) - 1)]
        fine_cuts = z0 + np.arange(LOWER_STRETCHES + 1) * ((z1 - z0) / LOWER_STRETCHES)
        cuts = np.concatenate([cuts[cuts < z0], fine_cuts, cuts[cuts > z1]])
    total = 0.0
    below: list[float] | None = None  # (``_below`` of every grid edge, once the first cell is weighed)
    for level in range(LOWER_CELL_DEPTH + 1):
        if possible:
            may = boxes(cells)[4].any(axis=1)
            # (the cells that may meet it count whole at the last level, or once cutting them again would weigh more than
            # LOWER_BOX_BUDGET boxes)
            last = level == LOWER_CELL_DEPTH or 4 * int(may.sum()) * (len(cuts) - 1) > LOWER_BOX_BUDGET
            sure = may if last else np.zeros(len(cells), dtype=bool)
        else:
            last = False
            sure, may = certain(cells)
        if bool(sure.any()):
            held = cells[sure]
            if below is None:
                below = [_below(e) for e in edges.tolist()]
            weights = _mass(below, held[:, 0], held[:, 1]) * _mass(below, held[:, 2], held[:, 3])
            for w in weights:  # (in order: the same sum in every runtime)
                total += float(w)
        cells = cells[~sure & may]
        if level == LOWER_CELL_DEPTH or last or not len(cells):
            break
        half = (cells[:, 1] - cells[:, 0]) // 2
        i_mid, j_mid = cells[:, 0] + half, cells[:, 2] + half
        cells = np.concatenate([np.stack([cells[:, 0], i_mid, cells[:, 2], j_mid], axis=1), np.stack([i_mid, cells[:, 1], cells[:, 2], j_mid], axis=1),
                                np.stack([cells[:, 0], i_mid, j_mid, cells[:, 3]], axis=1), np.stack([i_mid, cells[:, 1], j_mid, cells[:, 3]], axis=1)])
    return min(total + outside, 1.0)
