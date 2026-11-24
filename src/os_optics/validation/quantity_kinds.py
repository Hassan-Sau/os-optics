"""The kinds of quantity a target metric is computed from.

A finding about a published model's data (``os_optics.local.validity``) touches some kinds of quantity and leaves the
others as the trace computed them: a coating used outside its bands has no published reflectance there, and the beam
it reflects still leaves the same surface in the same direction. A target is affected by a finding only when its metric
reads a kind the finding touches (``validity.MODEL_SCOPE_AFFECTS``).

``METRIC_QUANTITY_KINDS`` holds every metric explicitly (a test enumerates them); a metric without an entry reads every
kind (``metric_quantity_kinds``). A metric is entered with fewer than all kinds only when its value is the same number
whatever power and polarization the beams carry:

* ``GEOMETRY`` alone: the chief ray's position and direction on the plane, the Gaussian beam's radii, waists, Rayleigh
  ranges and divergences, the geometric and optical path lengths and the phase delay (the sum of index times length; a
  coating's phase is not part of it), the wavelength, the first-order image quantities, the alignment metrics (centroids,
  directions and footprints of two beams) and the two-beam geometry (crossing angle, fringe period and spacing, optical
  path difference);
* no kind: the declared limits of a retained degree of freedom and the layout checks, which read no light.

Every other metric reads all three: a power, an irradiance, a polarization state, a phase (it includes the coatings'
phases), a group delay or relative delay (its interval rests on the coatings' stated group-delay bounds, which hold
where the coating's data hold), a spot or capture quantity of the sampled pupil, a detector, rating or conversion
quantity, a control quantity.
"""

from __future__ import annotations

from enum import StrEnum

from .alignment import ALIGNMENT_METRICS
from .metrics import ALL_SUPPORTED, LAYOUT_METRICS, REFERENCE_RAY_METRICS
from .mechanical import MECHANICAL_METRICS


class QuantityKind(StrEnum):
    #: Where the light goes: the surfaces a beam meets, its points, directions, path lengths and Gaussian beam.
    GEOMETRY = "GEOMETRY"
    #: How much light: a beam's power and everything computed from powers.
    POWER = "POWER"
    #: The polarization state of the light.
    POLARIZATION = "POLARIZATION"


#: Every kind: what a metric without an entry reads and what a finding code without an entry touches.
ALL_QUANTITY_KINDS: frozenset[QuantityKind] = frozenset(QuantityKind)
_GEOMETRY = frozenset({QuantityKind.GEOMETRY})
_NONE: frozenset[QuantityKind] = frozenset()

#: Metrics whose value is fixed by the traced geometry alone (module docstring).
GEOMETRY_ONLY_METRICS: frozenset[str] = frozenset({
    *REFERENCE_RAY_METRICS,
    "beam_radius_x_mm", "beam_radius_y_mm", "beam_radius_mm", "waist_radius_x_mm", "waist_radius_y_mm",
    "waist_distance_x_mm", "waist_distance_y_mm", "rayleigh_range_x_mm", "rayleigh_range_y_mm",
    "divergence_half_angle_x_rad", "divergence_half_angle_y_rad",
    "geometric_path_length_mm", "optical_path_length_mm", "phase_delay_ps", "wavelength_nm",
    "image_parity", "image_rotation_deg", "transverse_magnification",
    *ALIGNMENT_METRICS,
    "beam_crossing_angle_deg", "half_angle_deg", "fringe_period_nm", "fringe_spacing_3d_nm",
    "optical_path_difference_mm", "optical_path_difference_abs_mm",
})
#: Metrics that read no light: declared limits and layout checks.
LIGHT_INDEPENDENT_METRICS: frozenset[str] = frozenset({*MECHANICAL_METRICS, *LAYOUT_METRICS})

#: metric -> the kinds of quantity its value is computed from.
METRIC_QUANTITY_KINDS: dict[str, frozenset[QuantityKind]] = {
    metric: _NONE if metric in LIGHT_INDEPENDENT_METRICS else _GEOMETRY if metric in GEOMETRY_ONLY_METRICS else ALL_QUANTITY_KINDS
    for metric in sorted(ALL_SUPPORTED)
}


def metric_quantity_kinds(metric: str) -> frozenset[QuantityKind]:
    """The kinds of quantity ``metric`` reads; every kind for a metric the table does not hold."""

    return METRIC_QUANTITY_KINDS.get(metric, ALL_QUANTITY_KINDS)


__all__ = ["ALL_QUANTITY_KINDS", "GEOMETRY_ONLY_METRICS", "LIGHT_INDEPENDENT_METRICS", "METRIC_QUANTITY_KINDS", "QuantityKind",
           "metric_quantity_kinds"]
