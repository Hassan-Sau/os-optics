"""The fixed lab polarization basis of a beam direction (the H / V of every input and output mode).

``V`` is design up (+Z) projected transverse to the beam; for a beam within 45 degrees of vertical, design +Y projected
instead. A direction field without a discontinuity does not exist on the sphere (any rule must switch somewhere); at 45
degrees both references stay well conditioned (their transverse projections are at least 1/sqrt(2) long), unlike a switch
near vertical, where +Z projected is tiny and turns with the smallest tilt. Where the rule switches, ``V`` (and ``H``) turn
about the beam; a direction within ``SWITCH_MARGIN`` of the switch is reported by ``near_switch`` so the export can warn that
another runtime's last-bit difference could pick the other reference.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

#: Design up (+Z) and the reference (+Y) for beams within 45 degrees of vertical.
UP = np.array([0.0, 0.0, 1.0])
FALLBACK = np.array([0.0, 1.0, 0.0])
#: A beam with (d . Z)^2 above this (within 45 degrees of vertical) uses FALLBACK.
SWITCH_COS2 = 0.5
#: |(d . Z)^2 - SWITCH_COS2| below this is "at the switch" (orders of magnitude above float64 roundoff of a unit vector).
SWITCH_MARGIN = 1e-9


def uses_fallback(direction) -> bool:
    """The beam is within 45 degrees of vertical: its ``V`` is design +Y projected."""
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    return float(d[2]) ** 2 > SWITCH_COS2


def near_switch(direction) -> bool:
    """The beam is at the 45 degree switch of the reference within ``SWITCH_MARGIN``."""
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    return abs(float(d[2]) ** 2 - SWITCH_COS2) < SWITCH_MARGIN


def lab_basis(direction) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """``(H, V)`` for a beam along unit ``direction``: ``V`` = design +Z projected transverse to the beam (normalized),
    ``H = V x d``, so ``H x V = d`` (the handedness of the Engine's ``e1 x e2 = d``). For a beam within 45 degrees of vertical
    ``V`` is design +Y projected instead."""
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    ref = FALLBACK if float(d[2]) ** 2 > SWITCH_COS2 else UP
    v = ref - float(ref @ d) * d
    v = v / np.linalg.norm(v)
    h = np.cross(v, d)
    return h / np.linalg.norm(h), v
