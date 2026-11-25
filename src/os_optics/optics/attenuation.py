"""Position-dependent thin attenuators (``NeutralDensityInteraction``): the optical density at each point of the plane.

A lowered attenuator (``AttenuatorG``) holds its density law in plane coordinates: origin at the surface origin, ``x``
along the law's axis, ``y = normal x x``; the azimuth of a point is measured from ``+x`` toward ``+y`` in [0, 360). A
rotary law reads the azimuth minus the wheel rotation ``position`` (degrees), a gradient the x coordinate minus the shift
``position`` (mm). A coated annulus (``inner_radius``) leaves the hub uncoated: a point closer than that radius to the
law's centre (the origin; for a gradient, the origin shifted by ``position`` along ``x`` with the filter) is uncoated.
``density(points)`` returns the optical density at each point and whether the point is coated;
``transmittance(points)`` the power transmittance ``10^-OD`` (``uncoated_transmittance`` on uncoated parts, NaN there
when the element states none: out of model scope).

``footprint_density_range`` samples the beam's 1/e^2 footprint on the plane (``FOOTPRINT_SAMPLES`` points of the
Gaussian ellipse projected along the chief ray, plus its centre; with a coated annulus also the footprint point nearest the
hub centre along the line from the footprint centre, or the hub centre itself when the footprint covers it, so a beam
across the hub's edge is seen even when the hub lies inside the footprint) and returns the smallest and largest density
there: the
Engine carries one power per beam (the chief ray's value), so the run reports a beam whose density varies by more than
``ND_FOOTPRINT_OD_TOLERANCE`` across that footprint as out of model scope. For a linear gradient of ``Delta OD`` across the
1/e^2 diameter, the beam-averaged transmittance differs from the chief-ray value by ``exp((ln 10 Delta OD / 2)^2 / 8) - 1``
(0.66 % at the tolerance 0.2) and the transmitted beam's centroid moves by ``ln 10 Delta OD / 8`` beam radii (6 %).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..math.exact_trig import atan2_deg, cis, exp, log10, pow10
from ..math.vectors import dot

Array = NDArray[np.float64]

#: The largest change of optical density across a beam's 1/e^2 footprint for which the chief-ray value represents the beam.
ND_FOOTPRINT_OD_TOLERANCE = 0.2
#: Points of the 1/e^2 footprint ellipse sampled for the density variation across a beam.
FOOTPRINT_SAMPLES = 32
FULL_TURN_DEG = 360.0
#: Optical density is a base-10 logarithm: T = DECADE^-OD.
DECADE = 10.0
LN10 = 2.302585092994046  # ln 10
#: A Gaussian profile exp(-2 x^2 / w^2) has the standard deviation w / 2, so its average of exp(-k x) is exp(k^2 w^2 / 8):
#: the divisor of ``transmission_error_estimate``'s exponent.
GAUSSIAN_AVERAGE_DIVISOR = 8.0
#: The unit directions (cos, sin) of the footprint samples, from libm-free trigonometry (identical in every runtime: the
#: sampled density range is reported).
_RING = np.array([[z.real, z.imag] for z in (cis(k * (2.0 * math.pi / FOOTPRINT_SAMPLES)) for k in range(FOOTPRINT_SAMPLES))])


@dataclass(frozen=True)
class AttenuatorG:
    """A lowered ``NeutralDensityInteraction`` (global frame): ``law`` kind and parameters, plane origin / normal / axes,
    wheel rotation or gradient shift ``position``, ``uncoated_transmittance`` (``None``: not stated) and ``reflectance``."""

    kind: str
    origin: Array
    normal: Array
    x_axis: Array
    y_axis: Array
    position: float
    reflectance: float
    uncoated_transmittance: float | None
    optical_density: float = 0.0
    od_start: float = 0.0
    od_end: float = 0.0
    start: float = 0.0  # arc start (deg) or x start (mm)
    end: float = 0.0
    steps: tuple[tuple[float, float, float], ...] = ()  # (start_deg, end_deg, od)
    inner_radius: float | None = None  # coated annulus: the uncoated hub's radius (mm)

    def hub_center(self) -> tuple[float, float]:
        """The centre of the coated annulus in plane coordinates: the origin, or for a gradient the origin shifted with the
        filter by ``position``."""
        return (self.position, 0.0) if self.kind == "linear_gradient" else (0.0, 0.0)

    def _outside_hub(self, x: Array, y: Array) -> NDArray[np.bool_]:
        if self.inner_radius is None:
            return np.ones(x.shape, dtype=bool)
        cx, cy = self.hub_center()
        return np.sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy)) >= self.inner_radius

    def plane_coordinates(self, points: Array) -> tuple[Array, Array]:
        rel = np.atleast_2d(points) - self.origin
        return dot(rel, self.x_axis), dot(rel, self.y_axis)

    def density(self, points: Array) -> tuple[Array, NDArray[np.bool_]]:
        """Optical density at each point (0 where uncoated) and the coated mask."""
        x, y = self.plane_coordinates(points)
        od, coated = self._law_density(x, y)
        if self.inner_radius is None:
            return od, coated
        coated = coated & self._outside_hub(x, y)
        return np.where(coated, od, 0.0), coated

    def _law_density(self, x: Array, y: Array) -> tuple[Array, NDArray[np.bool_]]:
        if self.kind == "fixed":
            return np.full(x.shape, self.optical_density), np.ones(x.shape, dtype=bool)
        if self.kind == "linear_gradient":
            s = x - self.position
            coated = (s >= self.start) & (s <= self.end)
            od = self.od_start + (self.od_end - self.od_start) * (s - self.start) / (self.end - self.start)
            return np.where(coated, od, 0.0), coated
        azimuth = np.mod(atan2_deg(y, x) - self.position, FULL_TURN_DEG)  # libm-free: the bits decide each ray's T
        if self.kind == "linear_rotary":
            coated = (azimuth >= self.start) & (azimuth <= self.end)
            od = self.od_start + (self.od_end - self.od_start) * (azimuth - self.start) / (self.end - self.start)
            return np.where(coated, od, 0.0), coated
        od = np.zeros(x.shape)
        coated = np.zeros(x.shape, dtype=bool)
        for start, end, value in self.steps:  # the first step containing the azimuth (steps may touch)
            here = ~coated & (azimuth >= start) & (azimuth <= end)
            od = np.where(here, value, od)
            coated = coated | here
        return od, coated

    def transmittance(self, points: Array) -> Array:
        """Power transmittance at each point: ``10^-OD`` on the coating, ``uncoated_transmittance`` elsewhere (NaN when the
        element states none)."""
        od, coated = self.density(points)
        uncoated = math.nan if self.uncoated_transmittance is None else self.uncoated_transmittance
        return np.where(coated, pow10(-od), uncoated)

    def footprint_points(self, center: Array, direction: Array, bx: Array, by: Array, wx: float, wy: float) -> Array:
        """The chief-ray hit ``center`` and ``FOOTPRINT_SAMPLES`` points of the beam's 1/e^2 ellipse (radii ``wx`` / ``wy``
        along its axes ``bx`` / ``by``) projected along ``direction`` onto the plane."""
        offsets = _RING[:, 0:1] * (wx * bx) + _RING[:, 1:2] * (wy * by)
        dn = float(dot(direction, self.normal))
        along = -dot(offsets, self.normal) / dn if abs(dn) > 0.0 else np.zeros(FOOTPRINT_SAMPLES)
        ring = center + offsets + along[:, None] * direction
        points = np.vstack([center[None, :], ring])
        if self.inner_radius is None:
            return points
        # the footprint on the plane is the ellipse c + cos(t) p1 + sin(t) p2 (p1, p2 the projected radii): solve for the hub
        # centre's (a, b) in that basis; inside (a^2 + b^2 <= 1) the hub centre is a footprint point, else the boundary point
        # c + (h - c) / sqrt(a^2 + b^2) on the line toward it
        p1, p2 = ring[0] - center, ring[FOOTPRINT_SAMPLES // 4] - center
        cx, cy = self.hub_center()
        hub = self.origin + cx * self.x_axis + cy * self.y_axis
        m11, m12 = float(dot(p1, self.x_axis)), float(dot(p2, self.x_axis))
        m21, m22 = float(dot(p1, self.y_axis)), float(dot(p2, self.y_axis))
        rel = hub - center
        r1, r2 = float(dot(rel, self.x_axis)), float(dot(rel, self.y_axis))
        det = m11 * m22 - m12 * m21
        if det == 0.0:  # a degenerate (edge-on) footprint: the ring samples stand alone
            return points
        # Cramer's rule (+ - x / only: identical in every runtime, unlike LAPACK)
        a, b = (r1 * m22 - m12 * r2) / det, (m11 * r2 - r1 * m21) / det
        reach = math.sqrt(a * a + b * b)
        nearest = hub if reach <= 1.0 else center + rel / reach
        return np.vstack([points, nearest[None, :]])

    def footprint_density_range(self, points: Array) -> tuple[float, float, bool]:
        """``(min OD, max OD, every point coated or uncoated-transmittance stated)`` over footprint ``points``; an uncoated
        point counts with the density of the uncoated transmittance (``-log10 T``; infinite for T = 0)."""
        od, coated = self.density(points)
        if not bool(coated.all()):
            if self.uncoated_transmittance is None:
                return float(od[coated].min()) if coated.any() else 0.0, float(od[coated].max()) if coated.any() else 0.0, False
            t = self.uncoated_transmittance
            od = np.where(coated, od, math.inf if t <= 0.0 else -log10(t))
        return float(od.min()), float(od.max()), True


def transmission_error_estimate(delta_od: float) -> float:
    """Relative difference between a Gaussian beam's averaged transmittance and its chief-ray transmittance behind a linear
    density gradient of ``delta_od`` across the beam's 1/e^2 diameter: ``exp((ln 10 delta_od / 2)^2 / 8) - 1``."""
    k = LN10 * delta_od / 2.0
    return float(exp(k * k / GAUSSIAN_AVERAGE_DIVISOR)) - 1.0 if math.isfinite(k) else math.inf
