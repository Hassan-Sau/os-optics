"""Measures of a focal field or a point spread function that do not presume its shape: the peak, the full width at half
maximum, the half width at ``1/e^2`` of the peak, encircled energy; Zernike polynomials for a stated pupil phase.

Every search is deterministic: a fixed sampling brackets the answer and a fixed number of golden-section or bisection steps
narrows it, so the same field gives the same numbers.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

#: Golden-section ratio of the peak search.
GOLDEN = 0.5 * (math.sqrt(5.0) - 1.0)
#: Steps of the golden-section peak search and of the bisection of a level crossing (each narrows the interval by 0.618 / 0.5).
REFINE_STEPS = 48
#: Bisection steps inside an interval already narrowed by a table (2^-32 of it).
BRACKETED_STEPS = 32
#: Points of the bracketing sample of a peak search.
BRACKET_SAMPLES = 41
#: Most outward steps a crossing search takes before it reports that the level is not crossed.
MAX_WALK_STEPS = 400


def maximize(profile: Callable[[Array], Array], centre: float, half_width: float) -> tuple[float, float]:
    """``(position, value)`` of the largest value of ``profile`` on ``[centre - half_width, centre + half_width]``."""

    grid = centre + np.linspace(-half_width, half_width, BRACKET_SAMPLES)
    values = profile(grid)
    best = int(np.argmax(values))
    low, high = float(grid[max(best - 1, 0)]), float(grid[min(best + 1, grid.size - 1)])
    a = high - GOLDEN * (high - low)
    b = low + GOLDEN * (high - low)
    fa, fb = (float(v) for v in profile(np.array([a, b])))
    for _ in range(REFINE_STEPS):
        if fa >= fb:
            high, b, fb = b, a, fa
            a = high - GOLDEN * (high - low)
            fa = float(profile(np.array([a]))[0])
        else:
            low, a, fa = a, b, fb
            b = low + GOLDEN * (high - low)
            fb = float(profile(np.array([b]))[0])
    position = 0.5 * (low + high)
    value = float(profile(np.array([position]))[0])
    if value < float(values[best]):  # the sampled best point holds a higher value than the refined one: keep it
        return float(grid[best]), float(values[best])
    return position, value


def crossing(profile: Callable[[Array], Array], start: float, step: float, level: float) -> float | None:
    """The first position from ``start`` in the direction of ``step`` where ``profile`` falls to ``level``; ``None`` when it
    does not within ``MAX_WALK_STEPS`` steps."""

    inside = start
    for index in range(1, MAX_WALK_STEPS + 1):
        outside = start + index * step
        if float(profile(np.array([outside]))[0]) <= level:
            break
        inside = outside
    else:
        return None
    for _ in range(REFINE_STEPS):
        middle = 0.5 * (inside + outside)
        if float(profile(np.array([middle]))[0]) > level:
            inside = middle
        else:
            outside = middle
    return 0.5 * (inside + outside)


def widths(profile: Callable[[Array], Array], peak_position: float, peak_value: float, step: float,
           level_share: float) -> tuple[float, float] | None:
    """``(lower, upper)`` crossings of ``level_share * peak_value`` on both sides of the peak, or ``None``."""

    level = level_share * peak_value
    upper = crossing(profile, peak_position, step, level)
    lower = crossing(profile, peak_position, -step, level)
    if upper is None or lower is None:
        return None
    return lower, upper


def zernike_radial(n: int, m: int, rho: Array) -> Array:
    """``R_n^|m|(rho)``."""

    m = abs(m)
    total = np.zeros(np.shape(rho))
    for k in range((n - m) // 2 + 1):
        coefficient = ((-1) ** k * math.factorial(n - k)
                       / (math.factorial(k) * math.factorial((n + m) // 2 - k) * math.factorial((n - m) // 2 - k)))
        total = total + coefficient * rho ** (n - 2 * k)
    return total


def zernike(n: int, m: int, rho: Array, phi: Array) -> Array:
    """The Zernike polynomial normalized to unit RMS over the unit disc (``cos(m phi)``, ``sin(|m| phi)`` for ``m < 0``)."""

    norm = math.sqrt(2.0 * (n + 1)) if m != 0 else math.sqrt(n + 1.0)
    angular = np.cos(m * phi) if m > 0 else (np.sin(-m * phi) if m < 0 else np.ones(np.shape(phi)))
    return norm * zernike_radial(n, m, rho) * angular


def radius_for_fraction(cumulative: Callable[[float], float], fraction: float, low: float, high: float) -> float | None:
    """The radius in ``[low, high]`` where the increasing ``cumulative`` reaches ``fraction``, by bisection (``low`` and
    ``high`` bracket it); ``None`` when it does not."""

    if cumulative(high) < fraction:
        return None
    for _ in range(BRACKETED_STEPS):
        middle = 0.5 * (low + high)
        if cumulative(middle) < fraction:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


__all__ = ["crossing", "maximize", "radius_for_fraction", "widths", "zernike", "zernike_radial"]
