"""Coating power coefficients R/T (s, p) as functions of (wavelength, AOI). Outside its data a coating gives its values at
the nearest stated wavelength and angle together with a ``COATING_OUT_OF_RANGE`` issue (``CoatingCoefficients.issue``,
``.held``): the caller decides what the held values may be used for.

A ``TabulatedCoating`` table whose values do not depend on the angle of incidence (every AOI column of a wavelength row
equal) is evaluated as a 1-D linear interpolation over wavelength only (``np.interp``): its result then cannot depend on the
AOI at all, so a last-ulp difference of the AOI (an ``arccos`` evaluated by another libm, e.g. native CPython versus
Pyodide) cannot change R / T. Tables that do depend on the AOI are interpolated bilinearly by explicit arithmetic
(``bilinear``: cell search, normalized distances, the four weighted corners summed in a fixed order -- only ``+ - x /``),
so the same AOI gives the same bits in every runtime; the tracer reads such a table at an AOI computed without libm
(``math.exact_trig.acos_deg``), so the value itself is the same in every runtime.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ...config import NumericalConfig
from ...contracts.materials import CoatingModel, ConstantCoating, DeclaredCoatingValidity, TabulatedCoating
from ...errors import ErrorCode, Status, ValidationIssue, make_issue

Array = NDArray[np.float64]
COEFFICIENT_NAMES = ("reflectance_s", "reflectance_p", "transmittance_s", "transmittance_p")


@dataclass(frozen=True)
class HeldPoint:
    """Where a coating's data was read for a wavelength or angle of incidence it does not cover: the nearest stated point
    (``wavelength_nm`` / ``aoi_deg`` of the first row asked for) and the range the data covers."""

    wavelength_nm: float
    aoi_deg: float
    asked_wavelength_nm: float
    asked_aoi_deg: float
    valid_wavelength_nm: tuple[float, float]
    valid_aoi_deg: tuple[float, float]


@dataclass(frozen=True)
class CoatingCoefficients:
    """Power coefficients per ray (arrays). ``issue`` / ``held``: some row is outside the coating's data, and the values of
    such a row are the coating's at the nearest stated wavelength and angle."""

    rs: Array
    rp: Array
    ts: Array
    tp: Array
    issue: ValidationIssue | None = None
    held: HeldPoint | None = None


class CoatingEvaluator:
    def __init__(self, model: CoatingModel, cfg: NumericalConfig) -> None:
        self._model = model
        self._cfg = cfg
        self._tables: dict[str, Array] = {}  # AOI-dependent tables, [wavelength][aoi]
        self._columns: dict[str, Array] = {}  # AOI-independent tables: the value per wavelength node
        if isinstance(model, TabulatedCoating):
            wavelengths = np.array(model.wavelength_nm, dtype=np.float64)
            self._wavelengths = wavelengths
            self._angles = np.array(model.aoi_deg, dtype=np.float64)
            for name in COEFFICIENT_NAMES:
                table = np.array(getattr(model, name), dtype=np.float64)
                if bool(np.all(table == table[:, :1])):
                    self._columns[name] = table[:, 0].copy()
                else:
                    self._tables[name] = table

    @property
    def group_delay_bound_fs(self) -> float | None:
        """The coating's group-delay bound (fs; ``CoatingModel.group_delay_bound_fs``), or None when it states none."""
        return self._model.group_delay_bound_fs

    @property
    def declared_validity(self) -> DeclaredCoatingValidity | None:
        """The angle range and wavelengths the coating's published record states (``CoatingModel.declared_validity``), or None without a record."""
        return self._model.declared_validity

    @property
    def angle_dependent(self) -> bool:
        """Whether some value of the table depends on the angle of incidence (the tracer then reads it at a libm-free AOI)."""
        return bool(self._tables)

    def aoi_range(self) -> tuple[float, float]:
        """The angles of incidence (deg) the coating's data covers."""
        m = self._model
        if isinstance(m, ConstantCoating):
            return 0.0, float(m.max_aoi_deg)
        return float(m.aoi_deg[0]), float(m.aoi_deg[-1])

    def aoi_nodes(self) -> tuple[float, ...]:
        """The table's angles of incidence (deg), between which its values are linear in the angle (none when they do not
        depend on it)."""
        if isinstance(self._model, ConstantCoating) or not self._tables:
            return ()
        return tuple(float(a) for a in self._model.aoi_deg)

    def in_range(self, wavelength_nm: float, aoi_deg: Array) -> NDArray[np.bool_]:
        """Per ray: is ``(wavelength_nm, aoi_deg)`` inside the coating's declared validity range (inclusive, with the config slack)?"""
        aoi = np.asarray(aoi_deg, dtype=np.float64)
        cfg = self._cfg
        m = self._model
        if isinstance(m, ConstantCoating):
            lo, hi = m.valid_wavelength_nm
            wl_ok = lo - cfg.tabulated_range_tol_nm <= wavelength_nm <= hi + cfg.tabulated_range_tol_nm
            return np.logical_and(wl_ok, aoi <= m.max_aoi_deg + cfg.tabulated_aoi_tol_deg)
        wl_lo, wl_hi = m.wavelength_nm[0], m.wavelength_nm[-1]
        a_lo, a_hi = m.aoi_deg[0], m.aoi_deg[-1]
        wl_ok = wl_lo - cfg.tabulated_range_tol_nm <= wavelength_nm <= wl_hi + cfg.tabulated_range_tol_nm
        return np.logical_and(wl_ok, (aoi >= a_lo - cfg.tabulated_aoi_tol_deg) & (aoi <= a_hi + cfg.tabulated_aoi_tol_deg))

    def coefficients(self, wavelength_nm: float, aoi_deg: Array, *, location: str) -> CoatingCoefficients:
        """Vectorized over ``aoi_deg``. A row outside the coating's data takes the values at the nearest stated wavelength and
        angle (a constant coating: its values), and the result carries the ``COATING_OUT_OF_RANGE`` issue and ``held``: the
        values are the coating's data, not a statement about the coating there.

        Every row of ``aoi_deg`` is checked; a caller that traces a bundle of rays passes only rows it still follows
        (``Tracer`` evaluates coatings on its live rays and holds a live pupil ray that leaves the range itself).
        """
        aoi = np.asarray(aoi_deg, dtype=np.float64)
        m = self._model
        issue, held = None, None
        if not bool(np.all(self.in_range(wavelength_nm, aoi))):
            if isinstance(m, ConstantCoating):
                bands = (m.valid_wavelength_nm, (0.0, m.max_aoi_deg))
            else:
                bands = ((m.wavelength_nm[0], m.wavelength_nm[-1]), (m.aoi_deg[0], m.aoi_deg[-1]))
            issue = _range_issue(location, wavelength_nm, aoi, bands[0], bands[1])
            first = float(aoi[0]) if aoi.size else 0.0
            held = HeldPoint(
                wavelength_nm=min(max(wavelength_nm, bands[0][0]), bands[0][1]), aoi_deg=min(max(first, bands[1][0]), bands[1][1]),
                asked_wavelength_nm=wavelength_nm, asked_aoi_deg=first,
                valid_wavelength_nm=(float(bands[0][0]), float(bands[0][1])), valid_aoi_deg=(float(bands[1][0]), float(bands[1][1])))
        if isinstance(m, ConstantCoating):
            full = lambda v: np.full_like(aoi, v)  # noqa: E731
            return CoatingCoefficients(full(m.reflectance_s), full(m.reflectance_p), full(m.transmittance_s), full(m.transmittance_p),
                                       issue, held)
        wl_lo, wl_hi = m.wavelength_nm[0], m.wavelength_nm[-1]
        a_lo, a_hi = m.aoi_deg[0], m.aoi_deg[-1]
        wl = min(max(wavelength_nm, wl_lo), wl_hi)
        out: dict[str, Array] = {}
        for name, column in self._columns.items():
            out[name] = np.full_like(aoi, min(max(float(np.interp(wl, self._wavelengths, column)), 0.0), 1.0))
        if self._tables:
            cell = _cell(self._wavelengths, np.array([wl]))
            cells = _cell(self._angles, np.clip(aoi, a_lo, a_hi))
            out.update({name: np.clip(bilinear(table, cell, cells), 0.0, 1.0) for name, table in self._tables.items()})
        return CoatingCoefficients(out["reflectance_s"], out["reflectance_p"], out["transmittance_s"], out["transmittance_p"],
                                   issue, held)

    def coefficients_held(self, wavelength_nm: float, aoi_deg: Array, *, location: str) -> CoatingCoefficients:
        """``coefficients`` with every angle of incidence held at the nearest angle the data covers (``aoi_range``): for a
        ray beside the chief ray whose angle leaves the range while the chief ray's is inside it. The wavelength is not
        held here: outside its range the result is ``coefficients``' (the values at the nearest stated point and the issue)."""
        lo, hi = self.aoi_range()
        return self.coefficients(wavelength_nm, np.clip(np.asarray(aoi_deg, dtype=np.float64), lo, hi), location=location)


def _cell(grid: Array, x: Array) -> tuple[NDArray[np.intp], Array]:
    """Per point ``x`` (inside ``[grid[0], grid[-1]]``): the index ``i`` of its grid cell (``grid[i] <= x < grid[i + 1]``,
    the last cell for ``x == grid[-1]``) and its normalized distance ``(x - grid[i]) / (grid[i + 1] - grid[i])``."""

    index = np.clip(np.searchsorted(grid, x, side="right") - 1, 0, grid.size - 2)
    return index, (x - grid[index]) / (grid[index + 1] - grid[index])


def bilinear(table: Array, rows: tuple[NDArray[np.intp], Array], columns: tuple[NDArray[np.intp], Array]) -> Array:
    """Bilinear interpolation of ``table[i][j]`` (``rows``: the cell / distance of the row coordinate, one point or one per
    column point; ``columns``: per point), summed as ``v00 (1-u)(1-v) + v01 (1-u) v + v10 u (1-v) + v11 u v``."""

    (i, u), (j, v) = rows, columns
    return (table[i, j] * (1.0 - u) * (1.0 - v) + table[i, j + 1] * (1.0 - u) * v
            + table[i + 1, j] * u * (1.0 - v) + table[i + 1, j + 1] * u * v)


def _range_issue(location: str, wavelength_nm: float, aoi: Array, wl_band: tuple[float, float], aoi_band: tuple[float, float]) -> ValidationIssue:
    return make_issue(
        ErrorCode.COATING_OUT_OF_RANGE,
        Status.OUT_OF_MODEL_SCOPE,
        "the coating's data do not cover this wavelength / angle of incidence",
        location=location,
        wavelength_nm=wavelength_nm,
        aoi_max_deg=float(np.max(aoi)) if aoi.size else 0.0,
        valid_wavelength_nm=list(wl_band),
        valid_aoi_deg=list(aoi_band),
        suggested_action="restrict the operating wavelength/AOI or provide a coating table covering them",
    )
