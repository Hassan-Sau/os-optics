"""Samplings of the reference sphere: a polar rule (Gauss-Legendre in ``theta``, uniform in ``phi``) and a Cartesian grid in
the direction sines ``(u, v) = (sin(theta) cos(phi), sin(theta) sin(phi))`` (``objective.py``: the pupil point's own
direction, opposite to the transverse part of ``s``).

Both weight a sample by the solid angle it stands for. The polar rule integrates a pupil that is smooth inside its annulus
``theta_min <= theta <= theta_max`` to the precision of its Gauss-Legendre order (the hard rim is a node boundary, not a
sampled edge). The Cartesian grid is what the matrix Fourier transform needs; a cell the rim crosses is weighted by the share
of its area inside the annulus (``RIM_SUBSAMPLES`` per axis) and takes the field at its centre.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import NDArray

Array = NDArray[np.float64]

#: Sub-cells per axis that measure the share of a rim cell inside the annulus.
RIM_SUBSAMPLES = 16
#: The largest direction sine a cell centre is evaluated at (a rim cell of a pupil reaching almost to grazing).
MAX_SINE = 0.9999


def gauss_legendre(count: int, low: float, high: float) -> tuple[Array, Array]:
    """``count`` Gauss-Legendre nodes and weights on ``[low, high]``."""

    nodes, weights = leggauss(count)
    half = 0.5 * (high - low)
    return low + half * (nodes + 1.0), half * weights


@dataclass(frozen=True)
class PolarRule:
    """Nodes ``(theta, phi)`` and solid-angle weights (``sin(theta) dtheta dphi``) of the sphere, flattened."""

    theta: Array
    phi: Array
    weight: Array
    theta_count: int
    phi_count: int


def polar_rule(theta_min: float, theta_max: float, theta_count: int, phi_count: int) -> PolarRule:
    theta, w_theta = gauss_legendre(theta_count, theta_min, theta_max)
    phi = 2.0 * math.pi * (np.arange(phi_count) + 0.5) / phi_count
    tt, pp = np.meshgrid(theta, phi, indexing="ij")
    weight = np.outer(w_theta * np.sin(theta), np.full(phi_count, 2.0 * math.pi / phi_count))
    return PolarRule(tt.ravel(), pp.ravel(), weight.ravel(), theta_count, phi_count)


@dataclass(frozen=True)
class CartesianGrid:
    """Cells of the direction-sine plane: ``sines`` the cell centres along ``u`` (columns) and ``v`` (rows), ``step`` their
    pitch; per cell the share ``fraction`` inside the annulus, the ``(theta, phi)`` of the point whose field it takes and its
    solid-angle ``weight`` (``fraction step^2 / cos(theta)``)."""

    sines: Array
    step: float
    fraction: Array
    theta: Array
    phi: Array
    weight: Array


def cartesian_grid(sine_min: float, sine_max: float, count: int) -> CartesianGrid:
    step = 2.0 * sine_max / count
    sines = (np.arange(count) - 0.5 * (count - 1)) * step
    uu, vv = np.meshgrid(sines, sines)
    radius = np.sqrt(uu * uu + vv * vv)
    half = 0.5 * step
    reach = half * math.sqrt(2.0)
    fraction = ((radius <= sine_max) & (radius >= sine_min)).astype(np.float64)
    rim = (np.abs(radius - sine_max) < reach) | ((sine_min > 0.0) & (np.abs(radius - sine_min) < reach))
    if rim.any():
        offsets = (np.arange(RIM_SUBSAMPLES) + 0.5) / RIM_SUBSAMPLES * step - half
        su = uu[rim][:, None, None] + offsets[None, None, :]
        sv = vv[rim][:, None, None] + offsets[None, :, None]
        sub = np.sqrt(su * su + sv * sv)
        fraction[rim] = ((sub <= sine_max) & (sub >= sine_min)).mean(axis=(1, 2))
    # a rim cell takes the field at its own centre, just outside the rim at most (the pupil's phase stays continuous there)
    theta = np.arcsin(np.minimum(radius, MAX_SINE))
    phi = np.arctan2(vv, uu)
    weight = fraction * step * step / np.cos(theta)
    return CartesianGrid(sines, step, fraction, theta, phi, weight)


__all__ = ["CartesianGrid", "PolarRule", "RIM_SUBSAMPLES", "cartesian_grid", "gauss_legendre", "polar_rule"]
