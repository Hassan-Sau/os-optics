"""Deterministic, version-stable pseudo random numbers.

Only the raw 64-bit stream of ``numpy.random.PCG64`` is used (stream-stable across NumPy versions and
platforms); the transforms to uniform / normal are implemented here with the standard library so no
NumPy ``Generator`` distribution algorithm (which may change between releases) is involved.
"""

from __future__ import annotations

import math

import numpy as np

_MANTISSA_BITS = 53  # IEEE-754 binary64 significand
_WORD_BITS = 64  # PCG64 raw output width
_DISCARD_BITS = _WORD_BITS - _MANTISSA_BITS
_TWO_POW_MANTISSA = float(2**_MANTISSA_BITS)


class DeterministicRng:
    """PCG64 raw stream + own transforms. Identifier recorded in the manifest as ``PRNG_ID``."""

    def __init__(self, seed: int) -> None:
        if seed < 0:
            raise ValueError("seed must be a non-negative integer")
        self._bg = np.random.PCG64(seed)
        self._spare: float | None = None

    def uniform(self) -> float:
        """Uniform in ``[0, 1)`` with 53 random bits."""
        return float(int(self._bg.random_raw()) >> _DISCARD_BITS) / _TWO_POW_MANTISSA

    def uniform_open(self) -> float:
        """Uniform in ``(0, 1)`` (never exactly zero), for logarithms."""
        return (float(int(self._bg.random_raw()) >> _DISCARD_BITS) + 0.5) / _TWO_POW_MANTISSA

    def normal(self) -> float:
        """Standard normal via Box-Muller (both variates used, in order)."""
        if self._spare is not None:
            value, self._spare = self._spare, None
            return value
        u1, u2 = self.uniform_open(), self.uniform()
        radius = math.sqrt(-2.0 * math.log(u1))
        angle = 2.0 * math.pi * u2
        self._spare = radius * math.sin(angle)
        return radius * math.cos(angle)

    def uniform_between(self, low: float, high: float) -> float:
        return low + (high - low) * self.uniform()
