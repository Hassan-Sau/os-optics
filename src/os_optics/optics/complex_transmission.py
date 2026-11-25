"""Thin complex-transmission elements (``ComplexTransmissionInteraction``): ``t(u, v) = A(u, v) exp(i phi(u, v))`` on a plane.

Coordinates: ``(u, v)`` in mm from the surface origin along its in-plane axes (``SurfaceG.u_axis``, ``v_axis``). The phase is the
uniform phase plus the linear ramps, steps and Zernike terms (evaluated with the sag-shape polynomial code: the same
arithmetic as a Zernike sag, in rad instead of mm) plus the sampled map's phase; the amplitude is the uniform factor times the
map's amplitude. The map is bilinear between its nodes (value and the patch's gradient).

What the tracer reads (``Tracer._do_complex_transmission``):

* ``evaluate``: ``A``, ``phi`` and ``grad phi`` at each ray's hit point: the ray's optical path grows by ``phi / k0`` and its
  tangential wavevector by ``grad phi`` (the local grating equation, ``n d_t' = n d_t + grad phi / k0``);
* ``footprint``: nodes and weights of the Gaussian beam's power over its footprint on the plane -- in the beam's whitened frame,
  a Gauss-Legendre rule in the radius over ``PANELS`` equal panels out to ``aperture_quad_extent_sigmas`` beam radii, times a
  uniform azimuth rule (exact for the azimuthal harmonics below its node count); each node projected along the chief ray onto
  the plane, with the Gaussian beam's local ray direction there (its wavefront normal: the chief ray's direction tilted by
  the offset over the wavefront's radius of curvature on each axis);
* ``evanescent``: the footprint nodes whose local plane wave the element turns beyond grazing (``|n d_t + grad phi / k0| >=
  n``): light there is evanescent, outside the model of propagating rays;
* ``evanescent_share_bound``: the beam's power share beyond the outermost node ring inside the innermost turned node,
  ``exp(-2 rho^2)`` in the whitened frame -- the share the turned nodes may stand for, at the footprint's resolution (every
  node on and within that ring propagates); the tracer leaves a share at or below its line's negligible level out (bounded) and keeps the Gaussian beam;
* ``beam_power_fraction``: the Gaussian-weighted mean of ``A^2`` over the footprint, with the same mean from a rule of half the
  nodes per axis (their difference: the quadrature's convergence);
* ``phase_residual``: the weighted RMS of the phase less its best-fitting plane over the footprint (a linear phase steers a
  Gaussian beam and keeps it Gaussian; anything else does not).

Every step is ``+ - x /`` and ``sqrt`` (``math.exact_trig`` for the azimuths and the logarithm-free radial rule): the same bits
in every runtime.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .._vendor.shared_contracts.common.sag_shapes import sag_form, sag_slopes
from ..math.exact_trig import cos_sin
from ..math.exact_trig import exp as exact_exp
from ..math.exact_trig import exp_float
from ..math.vectors import dot

Array = NDArray[np.float64]

#: Equal radial panels of the footprint rule (each with the Gauss-Legendre rule's nodes) out to the extent.
PANELS = 3
#: The Gaussian-weighted mean of A^2 is the element's power factor when the full and the half rule agree within this
#: (absolute, of the factor).
POWER_FACTOR_TOL = 1e-9
#: A beam stays Gaussian behind the element when its phase less the best-fitting plane varies by at most this RMS (rad) over
#: its footprint (a Strehl ratio above 1 - 1e-4).
PHASE_RESIDUAL_TOL_RAD = 0.01
#: A beam without a Gaussian footprint keeps its chief ray's A^2 when its live rays' A^2 differ by at most this (relative).
RAY_POWER_SPREAD_TOL = 1e-6
TWO_PI = 6.283185307179586


@dataclass(frozen=True)
class _ZernikeTerm:
    """A Zernike phase term as the sag-shape code reads it (``value_mm`` holds the phase in rad)."""

    index: int
    value_mm: float


@dataclass(frozen=True)
class _ZernikeShape:
    kind: str
    indexing: str
    normalization: str
    normalization_radius_mm: float
    terms: tuple[_ZernikeTerm, ...]


@dataclass(frozen=True)
class GridG:
    """A sampled map: first node ``(u0, v0)``, pitch ``(du, dv)``, node counts and the node values (``(nodes_v, nodes_u)``;
    ``None``: not stated)."""

    u0: float
    v0: float
    du: float
    dv: float
    nodes_u: int
    nodes_v: int
    amplitude: Array | None
    phase: Array | None


@dataclass(frozen=True)
class TransmissionMapG:
    """A lowered ``ComplexTransmissionInteraction`` (module docstring)."""

    amplitude: float
    phase0: float
    ramp: tuple[float, float]
    steps: tuple[tuple[float, float, float, float], ...]  # (phase, edge normal u, v, offset)
    zernike: tuple[Any, ...]  # SagForm of each Zernike phase (rad)
    grid: GridG | None
    #: the reach of the Zernike terms' normalization (the smallest radius; ``None``: no Zernike term)
    zernike_radius_mm: float | None = None

    @property
    def phase_structured(self) -> bool:
        """The phase is more than a constant and a linear ramp somewhere."""
        return bool(self.steps or self.zernike or (self.grid is not None and self.grid.phase is not None))


def lower_map(inter: Any) -> TransmissionMapG:
    """The lowered map of a ``ComplexTransmissionInteraction``."""

    gu = gv = 0.0
    steps, forms, radii = [], [], []
    for term in inter.phase_terms:
        if term.kind == "linear_ramp":
            gu, gv = gu + term.gradient_rad_per_mm[0], gv + term.gradient_rad_per_mm[1]
        elif term.kind == "step":
            steps.append((term.phase_rad, term.edge_normal[0], term.edge_normal[1], term.edge_offset_mm))
        else:
            shape = _ZernikeShape(kind="ZERNIKE", indexing=term.indexing, normalization=term.normalization,
                                  normalization_radius_mm=term.normalization_radius_mm,
                                  terms=tuple(_ZernikeTerm(index=t.index, value_mm=t.value_rad) for t in term.terms))
            forms.append(sag_form(shape))
            radii.append(term.normalization_radius_mm)
    grid = None
    if inter.sampled_map is not None:
        m = inter.sampled_map
        shape = (m.nodes_v, m.nodes_u)
        grid = GridG(u0=m.center_mm[0] - 0.5 * (m.nodes_u - 1) * m.pitch_mm[0], v0=m.center_mm[1] - 0.5 * (m.nodes_v - 1) * m.pitch_mm[1],
                     du=m.pitch_mm[0], dv=m.pitch_mm[1], nodes_u=m.nodes_u, nodes_v=m.nodes_v,
                     amplitude=None if m.amplitude is None else np.array(m.amplitude, dtype=np.float64).reshape(shape),
                     phase=None if m.phase_rad is None else np.array(m.phase_rad, dtype=np.float64).reshape(shape))
    return TransmissionMapG(amplitude=inter.amplitude, phase0=inter.phase_rad, ramp=(gu, gv), steps=tuple(steps), zernike=tuple(forms),
                            grid=grid, zernike_radius_mm=min(radii) if radii else None)


def grid_extent(grid: GridG) -> tuple[float, float, float, float]:
    """``(u_min, u_max, v_min, v_max)`` of the map's nodes."""

    return grid.u0, grid.u0 + (grid.nodes_u - 1) * grid.du, grid.v0, grid.v0 + (grid.nodes_v - 1) * grid.dv


def _bilinear(values: Array, i: NDArray[np.intp], j: NDArray[np.intp], fx: Array, fy: Array, du: float, dv: float) -> tuple[Array, Array, Array]:
    f00, f10, f01, f11 = values[j, i], values[j, i + 1], values[j + 1, i], values[j + 1, i + 1]
    value = (1.0 - fy) * ((1.0 - fx) * f00 + fx * f10) + fy * ((1.0 - fx) * f01 + fx * f11)
    d_u = ((1.0 - fy) * (f10 - f00) + fy * (f11 - f01)) / du
    d_v = ((1.0 - fx) * (f01 - f00) + fx * (f11 - f10)) / dv
    return value, d_u, d_v


def evaluate(m: TransmissionMapG, u: Array, v: Array) -> tuple[Array, Array, Array, Array]:
    """``(A, phi, d phi / d u, d phi / d v)`` at the points ``(u, v)`` (mm); a point beyond the map takes its border's value."""

    u, v = np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64)
    amp = np.full(u.shape, m.amplitude)
    phi = np.full(u.shape, m.phase0) + m.ramp[0] * u + m.ramp[1] * v
    gu, gv = np.full(u.shape, m.ramp[0]), np.full(u.shape, m.ramp[1])
    for phase, a, b, offset in m.steps:
        phi = phi + np.where(a * u + b * v >= offset, phase, 0.0)
    for form in m.zernike:
        z, zx, zy, _w, _ok = sag_slopes(form, u, v)
        phi, gu, gv = phi + z, gu + zx, gv + zy
    g = m.grid
    if g is not None:
        x = np.clip((u - g.u0) / g.du, 0.0, g.nodes_u - 1.0)
        y = np.clip((v - g.v0) / g.dv, 0.0, g.nodes_v - 1.0)
        i = np.minimum(np.floor(x).astype(np.intp), g.nodes_u - 2)
        j = np.minimum(np.floor(y).astype(np.intp), g.nodes_v - 2)
        fx, fy = x - i, y - j
        if g.amplitude is not None:
            a_map, _, _ = _bilinear(g.amplitude, i, j, fx, fy, g.du, g.dv)
            amp = amp * a_map
        if g.phase is not None:
            p_map, p_u, p_v = _bilinear(g.phase, i, j, fx, fy, g.du, g.dv)
            phi, gu, gv = phi + p_map, gu + p_u, gv + p_v
    return amp, phi, gu, gv


@dataclass(frozen=True)
class Footprint:
    """Nodes of a beam's footprint on the plane: ``(u, v)`` (mm) and weights (the share of the beam's power each stands for;
    their sum is the power within the extent)."""

    u: Array
    v: Array
    weight: Array
    #: the Gaussian beam's local ray direction at each node (unit; ``None``: not formed)
    directions: Array | None = None
    #: each node's radius in the beam's whitened frame (beam radii; ``None``: not formed)
    radius: Array | None = None


def footprint(rule: tuple[Array, Array], azimuths: int, extent: float, hit: Array, d: Array, normal: Array, origin: Array, u_axis: Array,
              v_axis: Array, bx: Array, by: Array, wx: float, wy: float, cx: float = 0.0, cy: float = 0.0) -> Footprint:
    """The nodes of a Gaussian beam's footprint (module docstring): ``rule`` the Gauss-Legendre nodes / weights on [-1, 1],
    ``azimuths`` the azimuth nodes, ``extent`` the radius in beam radii; ``hit`` the chief ray's hit point, ``d`` its direction,
    ``bx``, ``by`` and ``wx``, ``wy`` the beam's axes and 1/e^2 radii there, ``cx``, ``cy`` its wavefront curvatures (1 / the
    radius of curvature, positive diverging; 0: flat) on those axes."""

    x, w = rule
    radii, weights = [], []
    width = extent / PANELS
    for k in range(PANELS):
        rho = width * (k + 0.5 * (x + 1.0))
        radii.append(rho)
        # dP = 4 rho exp(-2 rho^2) d rho (d theta / 2 pi): the power within rho of a beam in its whitened frame
        weights.append(0.5 * width * w * 4.0 * rho * exact_exp(-2.0 * rho * rho))
    rho, wr = np.concatenate(radii), np.concatenate(weights)
    cos_t, sin_t = cos_sin(TWO_PI * np.arange(azimuths) / azimuths)
    xi = (rho[:, None] * cos_t[None, :]).ravel()
    eta = (rho[:, None] * sin_t[None, :]).ravel()
    weight = np.repeat(wr / azimuths, azimuths)
    points = hit[None, :] + (xi * wx)[:, None] * bx[None, :] + (eta * wy)[:, None] * by[None, :]
    along = float(dot(d, normal))
    points = points - (dot(points - hit[None, :], normal) / along)[:, None] * d[None, :]
    rel = points - origin[None, :]
    tilted = d[None, :] + (xi * wx * cx)[:, None] * bx[None, :] + (eta * wy * cy)[:, None] * by[None, :]
    directions = tilted / np.sqrt(dot(tilted, tilted))[:, None]
    return Footprint(dot(rel, u_axis), dot(rel, v_axis), weight, directions, np.repeat(rho, azimuths))


def evanescent(m: TransmissionMapG, fp: Footprint, normal: Array, u_axis: Array, v_axis: Array, scale: float) -> NDArray[np.bool_]:
    """Per footprint node: whether the element turns the beam's local plane wave there beyond grazing, ``|d_t + scale grad phi|
    >= 1`` (``d_t`` the tangential part of the node's local direction, ``scale = lambda / (2 pi n)``)."""
    assert fp.directions is not None
    _amp, _phi, gu, gv = evaluate(m, fp.u, fp.v)
    d = fp.directions
    along = dot(d, normal)
    tangential = d - along[:, None] * normal[None, :] + scale * (gu[:, None] * u_axis[None, :] + gv[:, None] * v_axis[None, :])
    return dot(tangential, tangential) >= 1.0


def evanescent_share_bound(fp: Footprint, turned: NDArray[np.bool_]) -> float:
    """An upper bound of the beam's power share that the ``turned`` footprint nodes (``evanescent`` within the clear
    aperture) stand for: the Gaussian beam's power beyond the outermost node ring inside the innermost turned node,
    ``exp(-2 rho^2)`` (whitened frame): every node on and within that ring propagates, so the turned region lies beyond it as
    far as the footprint's rings and azimuths resolve it; 1 when a node of the innermost ring is turned, 0 when none is."""
    if not bool(turned.any()):
        return 0.0
    assert fp.radius is not None
    inner = float(np.min(fp.radius[turned]))
    clear = fp.radius[fp.radius < inner]
    if clear.size == 0:
        return 1.0
    rho = float(np.max(clear))
    return exp_float(-2.0 * rho * rho)


def _weighted_mean(values: Array, weight: Array) -> float:
    total = 0.0
    norm = 0.0
    for value, w in zip(values.tolist(), weight.tolist()):  # (an ordered sum: the same bits in every runtime)
        total += w * value
        norm += w
    return total / norm


def beam_power_fraction(m: TransmissionMapG, fp: Footprint) -> float:
    """The Gaussian-weighted mean of ``A^2`` over the footprint."""

    amp, _phi, _gu, _gv = evaluate(m, fp.u, fp.v)
    return _weighted_mean(amp * amp, fp.weight)


def phase_residual(m: TransmissionMapG, fp: Footprint) -> float:
    """The weighted RMS (rad) of the phase less its best-fitting plane ``c + a u + b v`` over the footprint."""

    _amp, phi, _gu, _gv = evaluate(m, fp.u, fp.v)
    mu, mv, mp = (_weighted_mean(values, fp.weight) for values in (fp.u, fp.v, phi))
    du, dv, dp = fp.u - mu, fp.v - mv, phi - mp
    suu, svv, suv = _weighted_mean(du * du, fp.weight), _weighted_mean(dv * dv, fp.weight), _weighted_mean(du * dv, fp.weight)
    sup, svp = _weighted_mean(du * dp, fp.weight), _weighted_mean(dv * dp, fp.weight)
    det = suu * svv - suv * suv
    if det > 0.0:
        a, b = (svv * sup - suv * svp) / det, (suu * svp - suv * sup) / det
    else:  # (a footprint along a line: the fit along it)
        a, b = (sup / suu if suu > 0.0 else 0.0), 0.0
    resid = dp - a * du - b * dv
    return math.sqrt(max(_weighted_mean(resid * resid, fp.weight), 0.0))


__all__ = ["Footprint", "GridG", "POWER_FACTOR_TOL", "PHASE_RESIDUAL_TOL_RAD", "RAY_POWER_SPREAD_TOL", "TransmissionMapG",
           "beam_power_fraction", "evaluate", "evanescent", "evanescent_share_bound", "footprint", "grid_extent", "lower_map",
           "phase_residual"]
