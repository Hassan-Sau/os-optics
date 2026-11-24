"""Sampled source patterns: an emitter's intensity map, its sampled complex field or its seeded speckle field, in the
emitter's frame (the Project's ``SampledIntensitySourceV1``, ``SampledFieldSourceV1``, ``SpeckleSourceV1``).

Grid: ``nx`` points along x at pitch ``px`` and ``ny`` along y at ``py``, centred on the origin (point ``(r, c)`` at
``x = (c - (nx - 1) / 2) px``, ``y = (r - (ny - 1) / 2) py``); arrays are ``(ny, nx)``, row 0 at the most negative y.

What this module gives:

* ``sample_positions``: points distributed as the pattern's irradiance by the inverse of its cumulative distribution -- the
  row (or the row of cells) from the first uniform coordinate, the column within the row from the second, and the place in
  the cell from what remains of each: uniform in a cell of a piecewise-constant pattern, by the inverse of the bilinear
  surface's marginal and conditional distributions between the grid points of a ``BILINEAR`` map. A monotone map of each
  coordinate, so a stratified sequence stays stratified, and a uniform pattern gives the uniform rectangle's points.
* ``realization_phase``: the field's phase on the grid (a sampled field's stated phase; a speckle field's screen of
  realization ``m``, ``speckle_screen``).
* ``node_gradient`` / ``gradient_at`` / ``phase_at``: the phase gradient at the grid points (the wrapped differences to the
  neighbours, central inside, one-sided at an edge), bilinear between them, and the phase near a point along it.
* ``field_validity``: the largest wrapped phase step between neighbouring lit grid points and the largest transverse sine
  ``|grad phi| / (n k0)`` at a grid point any ray's direction is interpolated from, the two conditions of a field's ray
  decomposition.
* ``complex_field``: ``U`` in sqrt(W/mm^2) on the grid, scaled to the stated power.

Speckle screens are integer arithmetic (SplitMix64) and ``+ - x / sqrt`` with ``math.exact_trig.exp``: the same bits in every
runtime. Lit: ``|U|^2`` at least ``LIT_SHARE`` of the pattern's peak, and the brightest grid points holding all but
``UNCHECKED_POWER_SHARE`` of the power (so at most that share has unchecked phase steps).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from ...math.exact_trig import exp as exact_exp

Array = NDArray[np.float64]
CArray = NDArray[np.complex128]

#: A grid point is lit at or above this share of the pattern's peak irradiance.
LIT_SHARE = 1e-6
#: At most this share of a field's power may lie on grid points whose phase steps are not checked (the dimmest; ``lit``).
UNCHECKED_POWER_SHARE = 1e-3
#: The largest wrapped phase step between neighbouring lit grid points of a field the ray decomposition takes (four grid
#: points per 2 pi).
MAX_PHASE_STEP_RAD = 0.5 * math.pi
#: The screen kernel reaches this many correlation lengths each side.
KERNEL_REACH = 3.0
#: Uniform deviates summed per white deviate of a screen (centred and scaled to unit variance).
UNIFORMS_PER_DEVIATE = 4
_MASK_64 = (1 << 64) - 1
_GAMMA = 0x9E3779B97F4A7C15
_MUL_1 = 0xBF58476D1CE4E5B9
_MUL_2 = 0x94D049BB133111EB
_TWO_PI = 6.283185307179586
_PI = 3.141592653589793
_TWO_TO_MINUS_53 = 2.0 ** -53
# SplitMix64's three xor-shifts, the shift keeping a word's top 53 bits, and the variance of a uniform deviate on [0, 1) is 1 / 12
_SHIFT_1 = 30
_SHIFT_2 = 27
_SHIFT_3 = 31
_MANTISSA_SHIFT = 11
_UNIFORM_VARIANCE_INVERSE = 12.0


@dataclass(frozen=True)
class SourceGrid:
    """The grid of a pattern (module docstring)."""

    nx: int
    ny: int
    px: float
    py: float

    def x(self) -> Array:
        return (np.arange(self.nx, dtype=np.float64) - 0.5 * (self.nx - 1)) * self.px

    def y(self) -> Array:
        return (np.arange(self.ny, dtype=np.float64) - 0.5 * (self.ny - 1)) * self.py

    @property
    def cell_area(self) -> float:
        return self.px * self.py


@dataclass(frozen=True)
class SpeckleParams:
    """A speckle source's illumination and phase screen (the Project's ``SpeckleSourceV1``)."""

    illumination: Literal["UNIFORM_DISC", "UNIFORM_RECTANGLE", "GAUSSIAN"]
    size_mm: tuple[float, float]
    phase_rms_rad: float
    correlation_length_mm: float
    seed: int


@dataclass(frozen=True)
class PatternG:
    """A lowered sampled pattern. ``kind`` ``map``: an intensity map (``weights`` the relative exitance at the grid points,
    ``interpolation`` its rule); ``field``: a sampled complex field (``weights`` = ``|U|^2``, ``phase`` its phase);
    ``speckle``: an illumination (``weights``) through ``realizations`` screens (``speckle``). ``jones``: the field's
    polarization in the emitter frame (``None`` for a map: unpolarized). ``out_of_scope``: the issue of a coherent field
    whose rays the model does not take (undersampled phase, evanescent light; ``scene/lower.py``): no ray of it is launched
    and no field of it is propagated."""

    kind: Literal["map", "field", "speckle"]
    grid: SourceGrid
    interpolation: Literal["PIECEWISE_CONSTANT", "BILINEAR"]
    weights: Array
    phase: Array | None = None
    jones: CArray | None = None
    speckle: SpeckleParams | None = None
    realizations: int = 1
    out_of_scope: Any = field(default=None, compare=False)
    # the cumulative distributions of sample_positions, computed once
    _cdf: dict = field(default_factory=dict, compare=False, repr=False)

    @property
    def coherent(self) -> bool:
        return self.kind != "map"


# ------------------------------------------------------------------------------------------------------- positions


def _inverse(cdf: Array, u: Array) -> tuple[NDArray[np.intp], Array]:
    """The bin of each ``u`` under the cumulative ``cdf`` (``cdf[-1] = 1``) and the fraction of the bin below it."""

    index = np.minimum(np.searchsorted(cdf, u, side="right"), cdf.size - 1)
    low = np.where(index > 0, cdf[np.maximum(index - 1, 0)], 0.0)
    width = cdf[index] - low
    fraction = np.where(width > 0.0, (u - low) / np.where(width > 0.0, width, 1.0), 0.5)
    return index, np.clip(fraction, 0.0, 1.0)


def _cell_masses(pattern: PatternG) -> Array:
    """The probability mass of each cell: a grid point's cell, or (``BILINEAR``) the cell between four grid points."""

    w = pattern.weights
    if pattern.interpolation == "BILINEAR":
        return 0.25 * (w[:-1, :-1] + w[:-1, 1:] + w[1:, :-1] + w[1:, 1:])
    return w


def _distributions(pattern: PatternG) -> tuple[Array, Array]:
    """The rows' cumulative distribution and each row's cumulative distribution over its columns."""

    if "rows" not in pattern._cdf:
        mass = _cell_masses(pattern)
        rows = mass.sum(axis=1)
        row_cdf = np.cumsum(rows) / rows.sum()
        row_cdf[-1] = 1.0
        totals = np.where(rows > 0.0, rows, 1.0)
        col_cdf = np.cumsum(mass, axis=1) / totals[:, None]
        col_cdf[:, -1] = 1.0
        pattern._cdf["rows"], pattern._cdf["columns"] = row_cdf, col_cdf
    return pattern._cdf["rows"], pattern._cdf["columns"]


def _linear_inverse(a: Array, b: Array, t: Array) -> Array:
    """The ``s`` in ``[0, 1]`` where the integral of the density ``a + b s`` from 0 reaches ``t`` of its total (``a``, ``a + b``
    at least 0, not both 0): ``s = 2 t (a + b / 2) / (a + sqrt(a^2 + 2 b t (a + b / 2)))``, without cancellation."""

    half = a + 0.5 * b
    root = np.sqrt(np.maximum(a * a + 2.0 * b * t * half, 0.0))
    denominator = a + root
    s = np.where(denominator > 0.0, 2.0 * t * half / np.where(denominator > 0.0, denominator, 1.0), t)
    return np.clip(s, 0.0, 1.0)


def sample_positions(pattern: PatternG, u_x: Array, u_y: Array) -> tuple[Array, Array]:
    """Points ``(x, y)`` distributed as the pattern's irradiance (module docstring): ``u_y`` picks the row, ``u_x`` the column."""

    grid = pattern.grid
    row_cdf, col_cdf = _distributions(pattern)
    row, t_y = _inverse(row_cdf, u_y)
    column = np.empty(row.shape, dtype=np.intp)
    t_x = np.empty(row.shape)
    for r in np.unique(row):
        chosen = row == r
        column[chosen], t_x[chosen] = _inverse(col_cdf[r], u_x[chosen])
    if pattern.interpolation == "BILINEAR":
        w = pattern.weights
        f00, f10 = w[row, column], w[row, column + 1]
        f01, f11 = w[row + 1, column], w[row + 1, column + 1]
        s_y = _linear_inverse(0.5 * (f00 + f10), 0.5 * (f01 + f11) - 0.5 * (f00 + f10), t_y)
        c0 = (1.0 - s_y) * f00 + s_y * f01
        c1 = (1.0 - s_y) * f10 + s_y * f11
        s_x = _linear_inverse(c0, c1 - c0, t_x)
        x0 = grid.x()[0]
        y0 = grid.y()[0]
        return x0 + (column + s_x) * grid.px, y0 + (row + s_y) * grid.py
    x_edge = -0.5 * grid.nx * grid.px
    y_edge = -0.5 * grid.ny * grid.py
    return x_edge + (column + t_x) * grid.px, y_edge + (row + t_y) * grid.py


# ------------------------------------------------------------------------------------------------------- phase


def _wrap(values: Array) -> Array:
    """Each value wrapped into ``(-pi, pi]``."""

    wrapped = values - _TWO_PI * np.round(values / _TWO_PI)
    return np.where(wrapped <= -_PI, wrapped + _TWO_PI, wrapped)


def node_gradient(phase: Array, px: float, py: float) -> tuple[Array, Array]:
    """``(d phi / d x, d phi / d y)`` at the grid points (rad/mm): the wrapped differences to the neighbours, their mean inside,
    the one difference at an edge; 0 along an axis of one point."""

    def along(values: Array, axis: int, pitch: float) -> Array:
        n = values.shape[axis]
        if n < 2:
            return np.zeros(values.shape)
        steps = _wrap(np.diff(values, axis=axis)) / pitch
        out = np.empty(values.shape)
        lead = [slice(None)] * values.ndim
        tail = [slice(None)] * values.ndim
        inner = [slice(None)] * values.ndim
        lead[axis], tail[axis], inner[axis] = 0, n - 1, slice(1, n - 1)
        first = [slice(None)] * values.ndim
        last = [slice(None)] * values.ndim
        first[axis], last[axis] = 0, n - 2
        out[tuple(lead)] = steps[tuple(first)]
        out[tuple(tail)] = steps[tuple(last)]
        if n > 2:
            left = [slice(None)] * values.ndim
            right = [slice(None)] * values.ndim
            left[axis], right[axis] = slice(0, n - 2), slice(1, n - 1)
            out[tuple(inner)] = 0.5 * (steps[tuple(left)] + steps[tuple(right)])
        return out

    return along(phase, 1, px), along(phase, 0, py)


def _bilinear_nodes(values: Array, grid: SourceGrid, x: Array, y: Array) -> Array:
    """``values`` (at the grid points) bilinear at ``(x, y)``, the border's value beyond the outer points."""

    fx = np.clip((x - grid.x()[0]) / grid.px, 0.0, grid.nx - 1.0)
    fy = np.clip((y - grid.y()[0]) / grid.py, 0.0, grid.ny - 1.0)
    i = np.minimum(np.floor(fx).astype(np.intp), max(grid.nx - 2, 0))
    j = np.minimum(np.floor(fy).astype(np.intp), max(grid.ny - 2, 0))
    i1 = np.minimum(i + 1, grid.nx - 1)
    j1 = np.minimum(j + 1, grid.ny - 1)
    ax, ay = fx - i, fy - j
    return ((1.0 - ay) * ((1.0 - ax) * values[j, i] + ax * values[j, i1])
            + ay * ((1.0 - ax) * values[j1, i] + ax * values[j1, i1]))


def gradient_at(gx: Array, gy: Array, grid: SourceGrid, x: Array, y: Array) -> tuple[Array, Array]:
    """The phase gradient at ``(x, y)``: the grid points' gradients, bilinear between them."""

    return _bilinear_nodes(gx, grid, x, y), _bilinear_nodes(gy, grid, x, y)


def phase_at(phase: Array, gx: Array, gy: Array, grid: SourceGrid, x: Array, y: Array) -> Array:
    """The phase at ``(x, y)``: the nearest grid point's phase plus its gradient times the offset from it."""

    i = np.clip(np.rint((x - grid.x()[0]) / grid.px).astype(np.intp), 0, grid.nx - 1)
    j = np.clip(np.rint((y - grid.y()[0]) / grid.py).astype(np.intp), 0, grid.ny - 1)
    return phase[j, i] + gx[j, i] * (x - grid.x()[i]) + gy[j, i] * (y - grid.y()[j])


def lit(weights: Array) -> NDArray[np.bool_]:
    """The lit grid points (module docstring): at least ``LIT_SHARE`` of the peak, and the brightest points that hold all but
    ``UNCHECKED_POWER_SHARE`` of the power (a dim floor that holds more is lit too)."""

    on = weights >= LIT_SHARE * float(weights.max())
    flat = np.sort(weights, axis=None)[::-1]
    held = np.cumsum(flat)
    index = int(np.searchsorted(held, (1.0 - UNCHECKED_POWER_SHARE) * float(held[-1])))
    floor = float(flat[min(index, flat.size - 1)])
    return on | ((weights >= floor) & (weights > 0.0))


def emitting(weights: Array) -> NDArray[np.bool_]:
    """The grid points whose phase gradient a ray can carry: the points of every cell that emits (weight above 0) and their
    neighbours (a ray's gradient is bilinear between the four grid points around it, up to half a pitch from its cell)."""

    on = np.pad(weights > 0.0, 1)
    near = np.zeros(on.shape, dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            near |= np.roll(np.roll(on, dy, axis=0), dx, axis=1)
    return near[1:-1, 1:-1]


def field_validity(phase: Array, weights: Array, grid: SourceGrid, k_medium_per_mm: float) -> tuple[float, float]:
    """The largest wrapped phase step between neighbouring lit grid points (rad) and the largest transverse sine
    ``|grad phi| / k_medium`` at a grid point a ray's direction is interpolated from (``emitting``; ``k_medium = n k0``).
    Below a sine of 1 at those points, every ray's sine is below 1 (bilinear interpolation stays within the disc of its four
    points)."""

    on = lit(weights)
    step = 0.0
    if grid.nx > 1:
        both = on[:, 1:] & on[:, :-1]
        if both.any():
            step = max(step, float(np.abs(_wrap(np.diff(phase, axis=1)))[both].max()))
    if grid.ny > 1:
        both = on[1:, :] & on[:-1, :]
        if both.any():
            step = max(step, float(np.abs(_wrap(np.diff(phase, axis=0)))[both].max()))
    gx, gy = node_gradient(phase, grid.px, grid.py)
    sine = float(np.sqrt(gx * gx + gy * gy)[emitting(weights)].max()) / k_medium_per_mm
    return step, sine


# ------------------------------------------------------------------------------------------------------- speckle


def _splitmix_words(seed: int, count: int) -> NDArray[np.uint64]:
    """``count`` consecutive SplitMix64 outputs of ``seed`` (Steele, Lea, Flood 2014)."""

    with np.errstate(over="ignore"):
        index = np.arange(1, count + 1, dtype=np.uint64)
        z = np.uint64(seed & _MASK_64) + index * np.uint64(_GAMMA)
        z = (z ^ (z >> np.uint64(_SHIFT_1))) * np.uint64(_MUL_1)
        z = (z ^ (z >> np.uint64(_SHIFT_2))) * np.uint64(_MUL_2)
        return z ^ (z >> np.uint64(_SHIFT_3))


def _deviates(seed: int, shape: tuple[int, int]) -> Array:
    """Unit-variance deviates: the centred sum of ``UNIFORMS_PER_DEVIATE`` uniform deviates on [0, 1) (53-bit, exact),
    scaled by ``sqrt(12 / UNIFORMS_PER_DEVIATE)``."""

    count = shape[0] * shape[1]
    words = _splitmix_words(seed, count * UNIFORMS_PER_DEVIATE)
    uniform = (words >> np.uint64(_MANTISSA_SHIFT)).astype(np.float64) * _TWO_TO_MINUS_53
    total = uniform.reshape(count, UNIFORMS_PER_DEVIATE).sum(axis=1) - 0.5 * UNIFORMS_PER_DEVIATE
    return (total * math.sqrt(_UNIFORM_VARIANCE_INVERSE / UNIFORMS_PER_DEVIATE)).reshape(shape)


def _kernel(pitch: float, correlation_length: float) -> Array:
    reach = int(math.ceil(KERNEL_REACH * correlation_length / pitch))
    t = np.arange(-reach, reach + 1, dtype=np.float64) * pitch
    return exact_exp(-2.0 * t * t / (correlation_length * correlation_length))


def speckle_screen(grid: SourceGrid, phase_rms_rad: float, correlation_length_mm: float, seed: int) -> Array:
    """One phase screen (the Project's ``PhaseScreenV1``): unit deviates on the extended grid, convolved along x and then y
    with the truncated Gaussian kernels, scaled to ``phase_rms_rad``."""

    kx, ky = _kernel(grid.px, correlation_length_mm), _kernel(grid.py, correlation_length_mm)
    hx, hy = (kx.size - 1) // 2, (ky.size - 1) // 2
    white = _deviates(seed, (grid.ny + 2 * hy, grid.nx + 2 * hx))
    along_x = np.zeros((white.shape[0], grid.nx))
    for t, weight in enumerate(kx):
        along_x = along_x + weight * white[:, t:t + grid.nx]
    screen = np.zeros((grid.ny, grid.nx))
    for t, weight in enumerate(ky):
        screen = screen + weight * along_x[t:t + grid.ny, :]
    scale = phase_rms_rad / math.sqrt(float(np.sum(kx * kx)) * float(np.sum(ky * ky)))
    return screen * scale


def illumination_weights(params: SpeckleParams, grid: SourceGrid) -> Array:
    """The relative irradiance of a speckle source's illumination at the grid points."""

    xx, yy = np.meshgrid(grid.x(), grid.y())
    a, b = params.size_mm
    if params.illumination == "UNIFORM_DISC":
        return np.where(xx * xx + yy * yy <= 0.25 * a * a, 1.0, 0.0)
    if params.illumination == "UNIFORM_RECTANGLE":
        return np.where((np.abs(xx) <= 0.5 * a) & (np.abs(yy) <= 0.5 * b), 1.0, 0.0)
    return exact_exp(-2.0 * (xx * xx + yy * yy) / (a * a))


def realization_phase(pattern: PatternG, m: int) -> Array:
    """The field's phase on the grid in realization ``m`` (a sampled field has one)."""

    if pattern.kind == "field":
        assert pattern.phase is not None
        return pattern.phase
    params = pattern.speckle
    assert params is not None
    return speckle_screen(pattern.grid, params.phase_rms_rad, params.correlation_length_mm, params.seed + m)


def complex_field(pattern: PatternG, m: int, power_W: float) -> CArray:
    """``U`` in sqrt(W/mm^2) at the grid points of realization ``m``: ``sum |U|^2`` times the cell area is ``power_W``."""

    total = float(pattern.weights.sum()) * pattern.grid.cell_area
    amplitude = np.sqrt(pattern.weights * (power_W / total))
    phase = realization_phase(pattern, m)
    return amplitude * np.exp(1j * phase)


__all__ = ["LIT_SHARE", "MAX_PHASE_STEP_RAD", "UNCHECKED_POWER_SHARE", "emitting", "PatternG", "SourceGrid", "SpeckleParams", "complex_field", "field_validity",
           "gradient_at", "illumination_weights", "lit", "node_gradient", "phase_at", "realization_phase", "sample_positions",
           "speckle_screen"]
