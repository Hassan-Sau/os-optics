"""Deterministic equal-power pupil sampling of a Gaussian intensity distribution.

Ring ``k`` (of ``K``) sits at the radius enclosing cumulative power ``(k + 1/2)/K`` of a unit Gaussian,
``rho_k = sqrt(-ln(1 - (k+1/2)/K) / 2)`` (in units of the 1/e^2 radius), with ``M`` equally spaced points per ring
(alternate rings are offset by half a step). Every sample carries the same power weight ``1/(K M)``. Elliptical
beams scale ``x`` and ``y`` by ``w_x`` and ``w_y``.
"""

from __future__ import annotations

import math

import numpy as np

from ...math.exact_trig import cos_sin, log
from numpy.typing import NDArray

Array = NDArray[np.float64]


def pupil_offsets(w_x: float, w_y: float, ring_count: int, points_per_ring: int) -> Array:
    """``(K*M, 2)`` transverse offsets ``(x, y)`` in the beam frame (mm)."""
    rho = [math.sqrt(-0.5 * log(1.0 - (k + 0.5) / ring_count)) for k in range(ring_count)]
    offset = [0.5 if (k % 2) else 0.0 for k in range(ring_count)]
    # every sample at once, ring after ring: phi = 2 pi (m + offset_k) / M, radius rho_k
    steps = np.tile(np.arange(points_per_ring), ring_count) + np.repeat(np.array(offset, dtype=np.float64), points_per_ring)
    phi = 2.0 * math.pi * steps / points_per_ring
    cos_phi, sin_phi = cos_sin(phi)  # libm-free (the bits of ``cis`` per angle): the same samples in every runtime
    radius = np.repeat(np.array(rho, dtype=np.float64), points_per_ring)
    return np.stack((w_x * radius * cos_phi, w_y * radius * sin_phi), axis=1)
