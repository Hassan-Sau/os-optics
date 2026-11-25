"""Engagement allowances of expected contacts: which overlap an interface permits by design.

A supported part touches its support; the two part meshes may also overlap where the parts engage (a threaded stud
inside a mount body modelled solid, a post in a bore modelled solid, an optic in its seat). The
overlap is split by an engagement zone -- a cylinder about the parent datum's +Z axis -- and only the residual
overlap outside it is judged. The zone is derived from the support relation and the mate's standard interface
designation only (never from part identity):

=================  ==========================================  =============================================
relation           zone radius                                  zone axial range (parent datum frame, +Z)
=================  ==========================================  =============================================
STANDS_ON          none: planar contact on the board top        --
INSERTED_INTO      nominal radius of the designation            between the two mate datums (insertion depth)
SEATED_IN          nominal radius of the designation            the child's own extent along the seat axis
MOUNTED_ON,        nominal radius of the designation            +- ``fastener_engagement_diameters`` x diameter
FASTENED_TO, and
a non-support mate
=================  ==========================================  =============================================

Radius and range are widened by the recorded allowances. Designations understood: metric threads ``M<d>`` /
``M<d>x<pitch>`` (d), outer diameters ``OD<d>`` (d), and lens-tube threads ``SM05``/``SM1``/``SM2``/``SM3`` (their
nominal major diameters), each optionally followed by ``-<qualifier>``. Any other designation gives no zone: then an
overlap beyond contact cannot be judged and is reported as not bounded (UNKNOWN), never silently accepted.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from .._vendor.shared_contracts.assembly_preview import CheckerTolerancesV1, SupportRelationV1
from .overlap import EngagementZone

MM_PER_INCH = 25.4
LENS_TUBE_THREAD_INCHES = {"05": 0.535, "1": 1.035, "2": 2.035, "3": 3.035}

#: The interface families the checker knows, in match order: (designation pattern, nominal diameter in mm of a match).
#: A family is added as a row.
DESIGNATION_FAMILIES: tuple[tuple[re.Pattern[str], Callable[[re.Match[str]], float]], ...] = (
    (re.compile(r"^M([0-9]+(?:\.[0-9]+)?)(?:x[0-9]+(?:\.[0-9]+)?)?(?:-[A-Za-z0-9.]+)*$"), lambda m: float(m.group(1))),  # metric thread
    (re.compile(r"^OD([0-9]+(?:\.[0-9]+)?)(?:-[A-Za-z0-9.]+)*$"), lambda m: float(m.group(1))),  # outer diameter
    (re.compile(r"^SM(05|1|2|3)(?:-[A-Za-z0-9.]+)*$"), lambda m: LENS_TUBE_THREAD_INCHES[m.group(1)] * MM_PER_INCH),  # lens tube
)


def nominal_diameter_mm(designation: str | None) -> float | None:
    """Nominal diameter of a standard interface designation, or ``None`` if it is not one the checker knows."""

    if designation is None:
        return None
    for pattern, diameter in DESIGNATION_FAMILIES:
        if match := pattern.match(designation):
            return diameter(match)
    return None


def needs_zone(relation: SupportRelationV1 | None) -> bool:
    """Whether the relation engages volumetrically (so an unknown designation leaves the overlap unbounded)."""

    return relation is not SupportRelationV1.STANDS_ON


def engagement_zone(relation: SupportRelationV1 | None, designation: str | None, *, child_datum_z: float,
                    child_z_range: tuple[float, float], tolerances: CheckerTolerancesV1) -> EngagementZone | None:
    """The zone in the parent datum frame; ``child_datum_z`` is the child's mate datum origin z in that frame."""

    if not needs_zone(relation):
        return None
    diameter = nominal_diameter_mm(designation)
    if diameter is None:
        return None
    radius = diameter / 2.0 + tolerances.engagement_radial_allowance_mm
    axial = tolerances.engagement_axial_allowance_mm
    if relation is SupportRelationV1.INSERTED_INTO:
        lo, hi = min(0.0, child_datum_z), max(0.0, child_datum_z)
    elif relation is SupportRelationV1.SEATED_IN:
        lo, hi = child_z_range
    else:
        reach = tolerances.fastener_engagement_diameters * diameter
        lo, hi = -reach, reach
    return EngagementZone(radius=radius, z_min=lo - axial, z_max=hi + axial)
