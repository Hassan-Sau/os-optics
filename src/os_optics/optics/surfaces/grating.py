"""A diffraction grating in the lowered scene (``contracts.optics.DiffractionGratingInteraction``): the direction of each
order from the general vector grating equation, and the stated response per order.

Direction. With ``n`` the plane's normal, ``K`` the unit grating vector in the plane, ``L`` the period and ``lambda`` the
vacuum wavelength, an order ``m`` leaves with the tangential part ``n2 t2 = n1 t1 + m (lambda / L) K``, ``t = d - (d . n) n``
(in-plane and conical mounts alike), and the normal part that makes it a unit vector: on the far side of the plane for a
transmission order, back on the near side for a reflection one. ``|n2 t2| >= n2`` is a nonpropagating order: it has no
direction. Everything here uses ``+ - * /`` and ``sqrt`` only, so every runtime computes the same bits.

Response. The table (``GratingResponse``) answers a wavelength and a signed angle of incidence inside its domain: bilinear
over the grid's span with ``linear``, the grid's points with ``points_only``; a dimension of one value answers that value
only. Nothing is extrapolated: outside the domain there is no answer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ...math.vectors import dot

Array = NDArray[np.float64]

#: The tangential part must stay below the exit index by this relative margin to be a propagating order: at the limit the
#: order would run along the plane (a Rayleigh anomaly) and has no usable direction.
PROPAGATION_MARGIN = 1e-12
#: How close a chief ray's angle of incidence must be to a grid point that answers one angle (deg): a traced angle is the
#: stated one up to rounding.
AOI_MATCH_DEG = 1e-6


def _bracket(axis: tuple[float, ...], value: float, exact: bool, tolerance: float) -> tuple[int, int, float] | None:
    """The two grid indices around ``value`` and the weight of the second (``None``: outside the axis)."""

    if len(axis) == 1 or exact:
        for i, point in enumerate(axis):
            if abs(value - point) <= tolerance:
                return i, i, 0.0
        return None
    if value < axis[0] - tolerance or value > axis[-1] + tolerance:
        return None
    if value <= axis[0]:
        return 0, 0, 0.0
    if value >= axis[-1]:
        last = len(axis) - 1
        return last, last, 0.0
    i = next(k for k in range(len(axis) - 1) if value < axis[k + 1])
    return i, i + 1, (value - axis[i]) / (axis[i + 1] - axis[i])


@dataclass(frozen=True)
class GratingReading:
    """The response at one wavelength and angle: each order's efficiency per polarization and the absorption per polarization."""

    efficiency_te: dict[int, float]
    efficiency_tm: dict[int, float]
    absorption_te: float
    absorption_tm: float


@dataclass(frozen=True)
class GratingResponse:
    """The table of a grating in the lowered scene (``contracts.optics.GratingOrderTable``)."""

    normalization: str  # "absolute" | "relative"
    polarization_basis: str  # "te_tm" | "unpolarized"
    interpolation: str  # "linear" | "points_only"
    data_kind: str
    wavelength_nm: tuple[float, ...]
    aoi_deg: tuple[float, ...]
    orders: tuple[int, ...]
    efficiency_te: tuple[float, ...]
    efficiency_tm: tuple[float, ...]
    absorption_te: tuple[float, ...] | None
    absorption_tm: tuple[float, ...] | None
    max_conical_angle_deg: float

    def domain_text(self) -> str:
        """The wavelengths and angles the response is stated for, in words."""

        def span(axis: tuple[float, ...], unit: str) -> str:
            if self.interpolation == "linear" and len(axis) > 1:
                return f"{axis[0]!r}-{axis[-1]!r} {unit}"
            return ", ".join(repr(v) for v in axis) + f" {unit} exactly"

        return f"{span(self.wavelength_nm, 'nm')} at {span(self.aoi_deg, 'deg')}"

    def at(self, wavelength_nm: float, aoi_deg: float) -> GratingReading | None:
        """The response at one traced line and chief-ray angle, ``None`` outside the domain."""

        exact = self.interpolation != "linear"
        w = _bracket(self.wavelength_nm, wavelength_nm, exact, 0.0)
        a = _bracket(self.aoi_deg, aoi_deg, exact, AOI_MATCH_DEG)
        if w is None or a is None:
            return None
        count_w, count_a = len(self.wavelength_nm), len(self.aoi_deg)

        def blend(values: tuple[float, ...], start: int) -> float:
            def cell(i: int, j: int) -> float:
                return values[start + i * count_a + j]

            low = cell(w[0], a[0]) + (cell(w[0], a[1]) - cell(w[0], a[0])) * a[2]
            high = cell(w[1], a[0]) + (cell(w[1], a[1]) - cell(w[1], a[0])) * a[2]
            return low + (high - low) * w[2]

        block = count_w * count_a
        return GratingReading(
            efficiency_te={m: blend(self.efficiency_te, k * block) for k, m in enumerate(self.orders)},
            efficiency_tm={m: blend(self.efficiency_tm, k * block) for k, m in enumerate(self.orders)},
            absorption_te=0.0 if self.absorption_te is None else blend(self.absorption_te, 0),
            absorption_tm=0.0 if self.absorption_tm is None else blend(self.absorption_tm, 0))


@dataclass(frozen=True)
class GratingG:
    """A grating in the lowered scene: ``k_axis`` and ``groove_axis`` are global unit vectors in the plane (``groove_axis``
    is ``normal x k_axis``), ``orders`` the orders followed (with ``all_propagating`` the range's, each followed where it
    propagates)."""

    reflection: bool
    period_nm: float
    k_axis: Array
    groove_axis: Array
    orders: tuple[int, ...]
    all_propagating: bool
    response: GratingResponse | None


def diffracted(d: Array, normal: Array, k_axis: Array, m: int, ratio: float, n1: float, n2: float, reflection: bool) -> tuple[Array, NDArray[np.bool_]]:
    """Per ray, the direction of order ``m`` (rows of ``d``; a nonpropagating ray keeps ``d``) and whether it propagates.
    ``ratio`` is ``lambda / L``, ``n1`` the incident index and ``n2`` the exit index (``n1`` for a reflection order)."""

    along = dot(d, normal)
    tangent = n1 * (d - along[:, None] * normal)
    beta = tangent + (m * ratio) * k_axis
    square = dot(beta, beta)
    limit = n2 * n2 * (1.0 - PROPAGATION_MARGIN)
    propagates = square < limit
    part = np.sqrt(np.where(propagates, n2 * n2 - square, 1.0))
    side = np.where(along < 0.0, -1.0, 1.0)
    if reflection:
        side = -side
    out = (beta + (side * part)[:, None] * normal) / n2
    return np.where(propagates[:, None], out, d), propagates


def propagating_orders(along_k: float, across_k: float, ratio: float, n1: float, n2: float) -> tuple[int, ...]:
    """Every order that propagates for a ray of direction components ``along_k`` and ``across_k`` in the plane (``k_axis`` and
    ``groove_axis``): those with ``(n1 along_k + m ratio)^2 + (n1 across_k)^2 < n2^2``."""

    limit = n2 * n2 * (1.0 - PROPAGATION_MARGIN)
    square = limit - (n1 * across_k) * (n1 * across_k)
    if square <= 0.0:
        return ()
    reach = float(np.sqrt(square))
    centre = n1 * along_k
    low = int(np.ceil((-reach - centre) / ratio))
    high = int(np.floor((reach - centre) / ratio))
    return tuple(m for m in range(low, high + 1) if (centre + m * ratio) * (centre + m * ratio) < square)


__all__ = ["AOI_MATCH_DEG", "GratingG", "GratingReading", "GratingResponse", "PROPAGATION_MARGIN", "diffracted", "propagating_orders"]
