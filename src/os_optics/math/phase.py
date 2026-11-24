"""The resolution at which wrapped optical phases are reported.

A phase ``k0 * OPL + arg`` wrapped to ``[-pi, pi)`` loses the relative precision of its unwrapped value: ``k0 * OPL`` of a
200 mm path at 633 nm is about 2e6 rad, so one unit in the last place of the optical path length (3e-14 mm) is already
3e-10 rad of phase, and the rounding of a path through several surfaces (each intersection, each ``OPL += n t``) adds a
few of those; a 10 m path at 200 nm reaches about 1e-7 rad. The wrapped value has no meaning below that, so it is reported
rounded to ``PHASE_RESOLUTION_RAD`` (1e-6 rad, lambda / 6e6 of optical path): the reported digits are digits the optical
path length actually determines. The reported interval is ``(-pi, pi]``: a wrapped value within half a step of -pi reads
+3.141593, so a phase on the seam (a balanced pair whose interface phase is pi, the dark port of an interferometer) has
one sign whatever the sign of the last-bit noise of its path difference. The Engine computes the path lengths themselves
with the same bits in every runtime (``math.vectors.dot``: no fused multiply-add), so the rounding does not depend on the
runtime either.
"""

from __future__ import annotations

import math

#: Wrapped phases (``phase_rad``, ``phase_difference_rad``) are reported in steps of 1 / PHASE_STEPS_PER_RAD rad.
PHASE_STEPS_PER_RAD = 1e6
PHASE_RESOLUTION_RAD = 1.0 / PHASE_STEPS_PER_RAD
#: -pi on the reporting grid (-3.141593): the one step that reads as +pi instead.
_MINUS_PI_STEP = -round(math.pi * PHASE_STEPS_PER_RAD) / PHASE_STEPS_PER_RAD


def reported_phase(phase_rad: float) -> float:
    """A wrapped phase (rad, in ``[-pi, pi)``) rounded to ``PHASE_RESOLUTION_RAD`` (the nearest step, ties to even) and
    reported in ``(-pi, pi]``: a value that rounds to -3.141593 reads +3.141593, and -0.0 reads 0.0."""

    value = round(float(phase_rad) * PHASE_STEPS_PER_RAD) / PHASE_STEPS_PER_RAD + 0.0
    return -value if value == _MINUS_PI_STEP else value


__all__ = ["PHASE_RESOLUTION_RAD", "PHASE_STEPS_PER_RAD", "reported_phase"]
