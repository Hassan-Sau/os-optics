"""The per-wavelength response of a tabulated thin lens (``TabulatedThinLensInteraction``): focal length and transmittance
of one traced line, whether the line lies in the table's domain, and whether an arriving polarization is one the response
was validated for.

Interpolation uses ``+ - * /`` only, so every runtime computes the same bits:

* ``linear_in_optical_power``: ``P = P0 + (P1 - P0) (lambda - lambda0) / (lambda1 - lambda0)`` with ``P = 1 / f`` of the two
  rows around ``lambda``, then ``f = 1 / P`` (the rows share one sign, so ``P`` never crosses 0);
* ``linear`` (transmittance): the same linear form on ``T``;
* ``points_only``: the row whose wavelength lies within ``point_tolerance`` (relative) of ``lambda``.

A one-row table is a single design wavelength: matched within ``point_tolerance``. Outside the domain (beyond the rows'
span, or between rows for ``points_only``) the nearest row's value is used and ``in_domain`` is ``False``; nothing is
extrapolated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ...math.exact_linalg import cvdot
from ...math.vectors import dot

Array = NDArray[np.float64]
CArray = NDArray[np.complex128]
PERCENT = 100.0


#: A computed beam radius or field angle counts as beyond the validated maximum only when it exceeds it by more than this
#: relative amount plus the absolute amounts below: the computation's floating-point rounding (a 30 um waist computed as
#: 0.030000000000000002 mm) never takes a beam out of the validated domain, while any physical excess does.
DOMAIN_LIMIT_REL_TOLERANCE = 1e-9
DOMAIN_LIMIT_ABS_TOLERANCE_MM = 1e-12
DOMAIN_LIMIT_ABS_TOLERANCE_DEG = 1e-12


def beyond_limit(value: float, limit: float, absolute: float) -> bool:
    """Whether ``value`` exceeds the validated ``limit`` by more than the rounding tolerance (``DOMAIN_LIMIT_*``)."""

    return value > limit + DOMAIN_LIMIT_REL_TOLERANCE * abs(limit) + absolute


@dataclass(frozen=True)
class ThinLensTable:
    """A tabulated thin lens in the lowered scene (see ``contracts.optics.TabulatedThinLensInteraction``)."""

    wavelength_nm: tuple[float, ...]
    focal_length_mm: tuple[float, ...]
    transmittance: tuple[float, ...]
    focal_interpolation: str  # "linear_in_optical_power" | "points_only"
    transmittance_interpolation: str  # "linear" | "points_only"
    point_tolerance: float
    max_field_angle_deg: float | None = None
    max_beam_radius_mm: float | None = None
    validated_polarizations: tuple[str, ...] | None = None  # "linear_u" / "linear_v"; None: any
    supported_metrics: frozenset[str] | None = None  # None: every metric
    u_axis: Array | None = None  # global in-plane axes of the element plane (polarization check)
    v_axis: Array | None = None

    @property
    def span_nm(self) -> tuple[float, float]:
        return self.wavelength_nm[0], self.wavelength_nm[-1]

    def domain_text(self) -> str:
        """The table's wavelength domain in words."""

        tolerance = f"{self.point_tolerance * PERCENT:g} %"
        if len(self.wavelength_nm) == 1:
            return f"{self.wavelength_nm[0]!r} nm (a single wavelength, matched within +-{tolerance})"
        rows = ", ".join(f"{w!r}" for w in self.wavelength_nm)
        if "points_only" in (self.focal_interpolation, self.transmittance_interpolation):
            return f"its rows only ({rows} nm, each matched within +-{tolerance})"
        return f"{self.wavelength_nm[0]!r}-{self.wavelength_nm[-1]!r} nm"


def _matched_row(table: ThinLensTable, wavelength_nm: float) -> int | None:
    for i, w in enumerate(table.wavelength_nm):
        if abs(wavelength_nm - w) <= table.point_tolerance * w:
            return i
    return None


def _nearest_row(table: ThinLensTable, wavelength_nm: float) -> int:
    return min(range(len(table.wavelength_nm)), key=lambda i: abs(wavelength_nm - table.wavelength_nm[i]))


def _bracket(table: ThinLensTable, wavelength_nm: float) -> tuple[int, float] | None:
    """``(i, s)`` with ``lambda = lambda_i + s (lambda_(i+1) - lambda_i)``, ``0 <= s <= 1`` (``None`` outside the span)."""

    w = table.wavelength_nm
    if wavelength_nm < w[0] or wavelength_nm > w[-1]:
        return None
    for i in range(len(w) - 1):
        if wavelength_nm <= w[i + 1]:
            return i, (wavelength_nm - w[i]) / (w[i + 1] - w[i])
    return len(w) - 1, 0.0  # wavelength_nm == w[-1] of a one-row span


def _value(table: ThinLensTable, wavelength_nm: float, values: tuple[float, ...], linear: bool, power: bool) -> tuple[float, bool]:
    if len(values) == 1 or not linear:
        row = _matched_row(table, wavelength_nm)
        if row is not None:
            return values[row], True
        return values[_nearest_row(table, wavelength_nm)], False
    bracket = _bracket(table, wavelength_nm)
    if bracket is None:
        return values[_nearest_row(table, wavelength_nm)], False
    i, s = bracket
    if s == 0.0 or i == len(values) - 1:
        return values[i], True
    if s == 1.0:
        return values[i + 1], True
    if power:
        p0, p1 = 1.0 / values[i], 1.0 / values[i + 1]
        return 1.0 / (p0 + (p1 - p0) * s), True
    return values[i] + (values[i + 1] - values[i]) * s, True


def _response(table: ThinLensTable, wavelength_nm: float) -> tuple[float, bool, float, bool]:
    focal, focal_ok = _value(table, wavelength_nm, table.focal_length_mm, table.focal_interpolation == "linear_in_optical_power", True)
    transmittance, t_ok = _value(table, wavelength_nm, table.transmittance, table.transmittance_interpolation == "linear", False)
    return focal, focal_ok, transmittance, t_ok


def response_at(table: ThinLensTable, wavelength_nm: float) -> tuple[float, float, bool]:
    """``(focal_length_mm, transmittance, in_domain)`` of one traced line (module docstring)."""

    focal, focal_ok, transmittance, t_ok = _response(table, wavelength_nm)
    return focal, transmittance, focal_ok and t_ok


def nearest_row_quantities(table: ThinLensTable, wavelength_nm: float) -> tuple[str, ...]:
    """Which of the line's values ``response_at`` took from the nearest row because the table gives none at that
    wavelength: ``"focal length"`` and / or ``"transmittance"`` (empty inside the domain). A table interpolated in one
    quantity and given at its rows only in the other takes only the latter from the nearest row inside its span."""

    _, focal_ok, _, t_ok = _response(table, wavelength_nm)
    return tuple(name for name, ok in (("focal length", focal_ok), ("transmittance", t_ok)) if not ok)


def orthogonal_fractions(jones: CArray, e1: Array, e2: Array, d: Array, u_axis: Array, v_axis: Array) -> dict[str, float]:
    """For each validated linear polarization (``linear_u`` / ``linear_v``), the fraction of the arriving power in the
    linear polarization orthogonal to it. The validated direction is the element's in-plane axis projected onto the
    ray's transverse plane; the field is ``E = j1 e1 + j2 e2`` (Jones vector in the ray's transverse basis)."""

    field = jones[0] * e1 + jones[1] * e2
    total = cvdot(field, field).real
    out: dict[str, float] = {}
    for name, axis in (("linear_u", u_axis), ("linear_v", v_axis)):
        along = axis - float(dot(axis, d)) * d
        norm = float(np.sqrt(dot(along, along)))
        if norm == 0.0 or total == 0.0:
            out[name] = 1.0
            continue
        along = along / norm
        across = np.cross(d, along)
        component = cvdot(across, field)
        out[name] = (component.real * component.real + component.imag * component.imag) / total
    return out
