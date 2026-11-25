"""The rays that sample an extended emitter: position, direction and polarization of each sample, in the emitter's frame.

Sequence: the first ``N`` points (``N`` a power of two) of the five-dimensional Sobol sequence in Gray-code order, each
coordinate XOR-ed with a 32-bit shift derived from the seed (a digital shift, which keeps the net structure). Every step is
integer arithmetic or an exact conversion, so the points are the same bits on every platform. Each power-of-two prefix is
itself stratified: the first ``N / 2`` points are the estimate the results compare with the full ``N``.

Domain (one coordinate pair each, so each pair's strata are equal-measure cells of its domain):

* position, uniform in area: a rectangle ``x = (u0 - 1/2) w, y = (u1 - 1/2) h``; a disc ``r = R sqrt(u0), phi = 2 pi u1``;
  a sampled pattern (``EmitterG.pattern``: an intensity map, or the irradiance ``|U|^2`` of a coherent field) by the inverse of
  its cumulative distribution, the row from ``u1`` and the column from ``u0`` (``optics/fields/source_fields.sample_positions``);
* direction: ``LAMBERTIAN`` uniform in projected solid angle over the hemisphere around ``+z``, ``sin(theta)^2 = u2,
  phi = 2 pi u3`` (the projected solid angle of a cell is proportional to its area in ``(u2, u3)``); ``ISOTROPIC`` uniform in
  solid angle over the sphere, ``cos(theta) = 1 - 2 u2, phi = 2 pi u3``; ``COSINE_POWER`` (intensity ``cos^n(theta)`` over the
  hemisphere, the scattered light of a surface) ``cos(theta)^(n + 1) = 1 - u2, phi = 2 pi u3`` (the share of the power within
  ``theta`` is ``1 - cos(theta)^(n + 1)``; ``n = 1`` is the Lambertian law); ``UNIFORM_CONE`` uniform in projected solid angle
  within the half angle ``theta_c``, ``sin(theta)^2 = u2 sin(theta_c)^2, phi = 2 pi u3``; ``COLLIMATED`` along ``+z``. A coherent
  field's ray leaves along the field's local wavevector instead: its transverse direction is ``grad(phi) / (n k0)`` at its
  start (the gradient of realization ``k mod realizations`` for ray ``k``), its optical path starts at ``phi / k0``;
* polarization: the fifth coordinate picks one of two orthogonal linear states, so the ensemble is unpolarized; a coherent
  field's rays carry its Jones vector.

Every sample of a line therefore has the same weight ``1 / N`` of the line's power. Trigonometry is ``math/exact_trig``'s
(no libm).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ...math.exact_trig import cos_sin, log
from ...math.exact_trig import exp as exact_exp
from ...math.vectors import matvec
from ...scene.model import EmitterG
from ...units import NM_TO_MM
from ..fields.source_fields import gradient_at, node_gradient, phase_at, realization_phase, sample_positions

Array = NDArray[np.float64]

SEQUENCE_ID = "sobol-joe-kuo-5d-digital-shift-v1"
_BITS = 32
_TWO_TO_BITS = 2.0 ** _BITS
_HALF_CELL = 0.5
_HALF = 0.5
_TWO_PI = 6.283185307179586
_WORD_MASK = (1 << 32) - 1
_MASK_64 = (1 << 64) - 1
# Joe-Kuo direction numbers of dimensions 2 to 5: (degree s, polynomial a, initial m_1 .. m_s). Dimension 1 is the van der
# Corput sequence.
_DIRECTIONS = ((1, 0, (1,)), (2, 1, (1, 3)), (3, 1, (1, 3, 1)), (3, 2, (1, 1, 1)))
DIMENSIONS = 5
# SplitMix64 (Steele, Lea, Flood 2014): one 64-bit word per dimension from the seed.
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_MUL_1 = 0xBF58476D1CE4E5B9
_SPLITMIX_MUL_2 = 0x94D049BB133111EB
_SPLITMIX_SHIFT_1 = 30
_SPLITMIX_SHIFT_2 = 27
_SPLITMIX_SHIFT_3 = 31
_WORD_BITS_64 = 64


def _direction_integers() -> tuple[tuple[int, ...], ...]:
    """``V[d][j]``: the 32-bit direction integer of dimension ``d`` for bit ``j`` of the Gray-code index."""
    table = [tuple(1 << (_BITS - 1 - j) for j in range(_BITS))]
    for degree, polynomial, initial in _DIRECTIONS:
        v = [m << (_BITS - 1 - j) for j, m in enumerate(initial)]
        for j in range(degree, _BITS):
            value = v[j - degree] ^ (v[j - degree] >> degree)
            for k in range(1, degree):
                if (polynomial >> (degree - 1 - k)) & 1:
                    value ^= v[j - k]
            v.append(value)
        table.append(tuple(v))
    return tuple(table)


_V = _direction_integers()


def _shifts(seed: int) -> tuple[int, ...]:
    """One 32-bit digital shift per dimension (the high words of SplitMix64 from ``seed``)."""
    words = []
    state = seed & _MASK_64
    for _ in range(DIMENSIONS):
        state = (state + _SPLITMIX_GAMMA) & _MASK_64
        z = state
        z = ((z ^ (z >> _SPLITMIX_SHIFT_1)) * _SPLITMIX_MUL_1) & _MASK_64
        z = ((z ^ (z >> _SPLITMIX_SHIFT_2)) * _SPLITMIX_MUL_2) & _MASK_64
        z = z ^ (z >> _SPLITMIX_SHIFT_3)
        words.append((z >> (_WORD_BITS_64 - _BITS)) & _WORD_MASK)
    return tuple(words)


def sobol_points(count: int, seed: int) -> Array:
    """The first ``count`` shifted points, shape ``(count, DIMENSIONS)``, each coordinate in ``(0, 1)``."""
    index = np.arange(count, dtype=np.uint64)
    gray = index ^ (index >> np.uint64(1))
    bits = int(count).bit_length()
    points = np.zeros((count, DIMENSIONS))
    for d, shift in enumerate(_shifts(seed)):
        x = np.full(count, shift, dtype=np.uint64)
        for j in range(bits):
            x = x ^ (((gray >> np.uint64(j)) & np.uint64(1)) * np.uint64(_V[d][j]))
        points[:, d] = (x.astype(np.float64) + _HALF_CELL) / _TWO_TO_BITS  # (below 2^53: exact)
    return points


@dataclass(frozen=True)
class EmitterSamples:
    """The samples of one emitter, global frame: launch point and unit direction, and which of the two orthogonal linear
    states (``True``: along the second transverse axis) each carries; a coherent field's samples carry its Jones vector
    (``jones``, in the emitter frame's x / y projected on each ray) and their starting optical paths (``opl``, mm)."""

    points: Array
    directions: Array
    second_state: NDArray[np.bool_]
    jones: NDArray[np.complex128] | None = None
    opl: Array | None = None


def _field_directions(emitter: EmitterG, x: Array, y: Array, n_medium: float) -> tuple[Array, Array, Array]:
    """The transverse direction sines and starting optical paths of a coherent field's rays (module docstring)."""
    pattern = emitter.pattern
    assert pattern is not None
    k0 = _TWO_PI / (emitter.spectrum[0][0] * NM_TO_MM)
    count = x.size
    sx, sy, opl = np.zeros(count), np.zeros(count), np.zeros(count)
    for m in range(min(pattern.realizations, count)):
        rays = np.arange(m, count, pattern.realizations)
        phase = realization_phase(pattern, m)
        gx, gy = node_gradient(phase, pattern.grid.px, pattern.grid.py)
        ax, ay = gradient_at(gx, gy, pattern.grid, x[rays], y[rays])
        sx[rays], sy[rays] = ax / (n_medium * k0), ay / (n_medium * k0)
        opl[rays] = phase_at(phase, gx, gy, pattern.grid, x[rays], y[rays]) / k0
    return sx, sy, opl


def local_samples(emitter: EmitterG, n_medium: float | None = None) -> tuple[Array, Array, NDArray[np.bool_]]:
    """Launch points and unit directions in the emitter's own frame (module docstring) and the polarization state picks
    (``n_medium``: the medium's index at the field's wavelength, which a coherent field needs)."""
    return _local(emitter, n_medium)[:3]


def field_samples(emitter: EmitterG, n_medium: float) -> tuple[Array, Array, Array]:
    """A coherent field's launch points, unit directions (the emitter's own frame) and starting optical paths (mm)."""
    points, directions, _state, opl = _local(emitter, n_medium)
    if opl is None:
        raise ValueError("field_samples needs a coherent field's emitter")
    return points, directions, opl


def _local(emitter: EmitterG, n_medium: float | None) -> tuple[Array, Array, NDArray[np.bool_], Array | None]:
    """``local_samples`` and the starting optical paths of a coherent field's rays (``None`` otherwise)."""
    u = sobol_points(emitter.sample_count, emitter.seed)
    pattern = emitter.pattern
    if pattern is not None:
        x, y = sample_positions(pattern, u[:, 0], u[:, 1])
    elif emitter.shape == "disc":
        radius = emitter.half_x * np.sqrt(u[:, 0])
        cos_p, sin_p = cos_sin(_TWO_PI * u[:, 1])
        x, y = radius * cos_p, radius * sin_p
    else:
        x, y = (u[:, 0] - _HALF) * 2.0 * emitter.half_x, (u[:, 1] - _HALF) * 2.0 * emitter.half_y
    points = np.column_stack([x, y, np.zeros(x.shape)])
    if pattern is not None and pattern.coherent:
        if n_medium is None:
            raise ValueError("a coherent field's rays need the medium's index")
        sx, sy, opl = _field_directions(emitter, x, y, n_medium)
        transverse = sx * sx + sy * sy
        if not bool(np.all(transverse < 1.0)):  # (the scene's field validity keeps every ray's sine below 1)
            raise AssertionError(f"a coherent field's ray has a transverse sine of {float(np.sqrt(transverse.max())):.6g}: no unit direction")
        directions = np.column_stack([sx, sy, np.sqrt(np.maximum((1.0 - sx * sx) - sy * sy, 0.0))])
        return points, directions, np.zeros(x.shape, dtype=bool), opl
    if emitter.law == "COLLIMATED":
        directions = np.column_stack([np.zeros(x.shape), np.zeros(x.shape), np.ones(x.shape)])
        return points, directions, u[:, 4] >= _HALF, None
    cos_a, sin_a = cos_sin(_TWO_PI * u[:, 3])
    if emitter.law == "UNIFORM_CONE":
        sin_t = np.sqrt(u[:, 2]) * emitter.cone_sine
        cos_t = np.sqrt(np.maximum((1.0 - sin_t) * (1.0 + sin_t), 0.0))
    elif emitter.law == "ISOTROPIC":
        cos_t = 1.0 - 2.0 * u[:, 2]
        sin_t = np.sqrt(4.0 * u[:, 2] * (1.0 - u[:, 2]))  # (1 - cos^2, without the cancellation)
    elif emitter.law == "COSINE_POWER":
        # cos(theta) = (1 - u2)^(1 / (n + 1)), libm-free: exp(log(1 - u2) / (n + 1)) (u2 < 1: the cells' centres)
        cos_t = exact_exp(np.array([log(1.0 - float(v)) for v in u[:, 2]]) / (emitter.exponent + 1.0))
        sin_t = np.sqrt(np.maximum((1.0 - cos_t) * (1.0 + cos_t), 0.0))
    else:
        sin_t, cos_t = np.sqrt(u[:, 2]), np.sqrt(1.0 - u[:, 2])
    directions = np.column_stack([sin_t * cos_a, sin_t * sin_a, cos_t])
    return points, directions, u[:, 4] >= _HALF, None


def global_samples(emitter: EmitterG, n_medium: float | None = None) -> EmitterSamples:
    """The samples of ``emitter`` placed in the design frame (``n_medium``: see ``local_samples``)."""
    points, directions, state, opl = _local(emitter, n_medium)
    axes = np.column_stack([emitter.e1, emitter.e2, emitter.direction])
    jones = None
    if emitter.pattern is not None and emitter.pattern.jones is not None:
        jones = np.tile(emitter.pattern.jones, (points.shape[0], 1))
    # (each component the index-order sum of separately rounded products: the same bits in every runtime)
    return EmitterSamples(emitter.position + matvec(axes, points), matvec(axes, directions), state, jones, opl)
