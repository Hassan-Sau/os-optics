"""How much a beam's envelope delay can vary across it: the spread the exposures add to two beams' relative delay.

Two fields of one source line are coherent according to the delay of their envelopes, their group optical paths ``sum n_g
L`` (``optics/interference/coherence.py``). Across a beam the group path of a ray differs from its phase path ``sum n L`` by
``sum (n_g - n) L`` over the dispersive media it crossed. Where the beam crossed none, the envelope delay follows its
wavefront (the exposures bound that by the wavefront's tilt, offset and curvature: ``validation/exposures.py``). A segment
through a dispersive medium between two parallel planes (a window, a plate) changes it with the rays' angles only: a ray
at angle ``theta`` to the planes' normal crosses ``L cos(theta_c) / cos(theta)`` of the medium (``theta_c`` and ``L`` the
chief ray's), whatever its position. The rays of a beam's family of size ``K`` (``COHERENCE_K``: the rays within ``sqrt(K)``
1/e^2 radii and ``sqrt(K)`` slope envelopes ``w |1/q|`` of its chief ray -- the paraxial rays ``a x_A + b x_B``, ``a^2 + b^2
<= K``, of the beam's complex-ray pair; outside them its profile amplitude is below ``exp(-K)``) deviate from the chief ray
by at most ``sqrt(K) w |1/q|``, so the segment changes the excess ``(n_g - n) L`` across them by at most ``|n_g - n| L
max(cos(theta_c) / cos(theta_c + sqrt(K) w |1/q|) - 1, 1 - cos(theta_c))`` (``plate_spread``), and the group-delay dispersion
``k2 L`` likewise. A segment through a dispersive medium that is not between two parallel planes (behind a lens surface, in
a prism, between a face and a cube's diagonal) leaves the spread unbounded (``None``). Arithmetic: ``+ - x /``, ``sqrt`` and
the libm-free ``cos_sin`` (identical bits in every runtime).
"""

from __future__ import annotations

import math

import numpy as np

from ...math.exact_trig import cos_sin

#: Sizes of a beam's ray family (module docstring) at which the spread is kept, and their square roots.
COHERENCE_K = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
COHERENCE_SQRT_K = tuple(math.sqrt(k) for k in COHERENCE_K)
#: Per ``COHERENCE_K``: (the group path's spread beyond the phase path's, mm; the group-delay dispersion's spread, mm^2).
Spread = tuple[tuple[float, ...], tuple[float, ...]]
#: No spread: a beam that crossed no dispersive medium.
NO_SPREAD: Spread = (tuple(0.0 for _ in COHERENCE_K), tuple(0.0 for _ in COHERENCE_K))
#: Two planes whose unit normals' ``|dot|`` is at least ``1 -`` this bound a plate.
PARALLEL_PLATE_TOL = 1e-12


def plate_spread(excess: float, dispersion: float, length_mm: float, cos_chief: float, slope: float) -> Spread | None:
    """The spread of a segment of ``length_mm`` (the chief ray's) through a dispersive medium between two parallel planes
    (module docstring): ``excess`` ``= |n_g - n|``, ``dispersion`` ``= |k2|`` (mm^2 per mm), ``cos_chief`` the cosine of the
    chief ray's angle to the planes' normal, ``slope`` the beam's slope envelope ``w |1/q|`` in the medium. ``None`` where a
    family's rays may meet the planes at grazing incidence."""
    sin_chief = math.sqrt(max(0.0, (1.0 - cos_chief) * (1.0 + cos_chief)))
    cos_d, sin_d = cos_sin(np.array([root * slope for root in COHERENCE_SQRT_K], dtype=np.float64))
    group: list[float] = []
    spread: list[float] = []
    for c, s in zip(cos_d, sin_d):
        widest = cos_chief * float(c) - sin_chief * float(s)
        if widest <= 0.0 or float(c) <= 0.0:
            return None
        factor = max(cos_chief / widest - 1.0, 1.0 - cos_chief)
        group.append(excess * length_mm * factor)
        spread.append(dispersion * length_mm * factor)
    return tuple(group), tuple(spread)


def add_spread(a: Spread | None, b: Spread | None) -> Spread | None:
    """The sum of two spreads (``None``: not bounded)."""
    if a is None or b is None:
        return None
    return (tuple(x + y for x, y in zip(a[0], b[0])), tuple(x + y for x, y in zip(a[1], b[1])))


def spread_at(spread: Spread, k: float) -> tuple[float, float] | None:
    """(group, dispersion) spread of the smallest kept family at least ``k`` (``None`` beyond the largest)."""
    for index, size in enumerate(COHERENCE_K):
        if k <= size:
            return spread[0][index], spread[1][index]
    return None


__all__ = ["COHERENCE_K", "COHERENCE_SQRT_K", "NO_SPREAD", "PARALLEL_PLATE_TOL", "Spread", "add_spread", "plate_spread", "spread_at"]
