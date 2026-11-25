"""Gaussian beam state (per transverse axis complex q, x/y independent).

``q = z + i z_R`` in millimetres in the local medium of index ``n``; ``1/q = 1/R - i M^2 lambda /(pi n w^2)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from ...config import DEFAULT_NUMERICAL_CONFIG, NumericalConfig
from ...contracts.basis import WHOLE_RECORD, ValidityCheck, ValueBasis
from ...contracts.optics import GaussianBeamState
from ...errors import ErrorCode, Status
from ...errors.status import ValueKind
from ...units import NM_TO_MM

Array = NDArray[np.float64]

#: The resolution of a reported waist distance ``z = Re q`` (mm): the rounding of the path lengths it is the difference of,
#: the coordinate floor of a system about 500 mm across (128 units in the last place of its extent).
WAIST_DISTANCE_RESOLUTION_MM = 1e-11
#: The waist distance within which a curvature radius ``|q|^2 / z`` is not resolved to the cross-runtime relative tolerance
#: (1e-9): ``WAIST_DISTANCE_RESOLUTION_MM / 1e-9`` = 0.01 mm. A beam observed that close to its waist reads flat
#: (``GState.reported_curvature_radius``).
FLAT_WAIST_DISTANCE_MM = WAIST_DISTANCE_RESOLUTION_MM / DEFAULT_NUMERICAL_CONFIG.cross_runtime_rel_tol
#: The departure from a flat wavefront across the beam (``|z| / z_R`` rad at its radius) above which a curvature radius is
#: stated even within ``FLAT_WAIST_DISTANCE_MM`` of the waist: a beam whose Rayleigh range is below 1000 ``|z|``, whose
#: wavefront there is curved, not flat (its radius then carries up to ``WAIST_DISTANCE_RESOLUTION_MM / |z|`` of rounding).
FLAT_WAVEFRONT_PHASE_RAD = 1e-3


@dataclass(frozen=True)
class Limit:
    """One accuracy limit of the first-order model checked for a state: the caller setting that holds it (a
    ``NumericalConfig`` field), the quantity compared with it and its unit, the value observed and the limit in force, the
    code of the finding when it is exceeded, and where (a surface id or the source's id; ``None``: not placed yet)."""

    setting: str
    parameter: str
    unit: str
    observed: float
    maximum: float
    code: ErrorCode
    at: str | None = None

    def check(self, within: bool) -> ValidityCheck:
        return ValidityCheck(parameter=self.parameter, unit=self.unit, maximum=self.maximum, observed=self.observed, within=within,
                             setting=self.setting, at=() if self.at is None else (self.at,), code=None if within else self.code.value)


@dataclass(frozen=True)
class GState:
    valid: bool
    status: Status
    reason: ErrorCode | None
    reason_text: str
    wavelength_nm: float
    n: float
    qx: complex
    qy: complex
    m2x: float
    m2y: float
    bx: Array  # beam frame axis carrying qx (unit, global)
    by: Array
    # The accuracy limit of the first-order model the state first exceeded (``limited``): ``valid`` is False, no first-order
    # number is carried or published from there on (every step and consumer is keyed on ``valid``), and ``to_contract``
    # states this failed check (setting, quantity, observed value, limit, where). ``None`` with ``valid`` False: no
    # first-order state exists for another reason (``invalidated``: its code and text say why).
    limit: Limit | None = None
    # With a limit set by the caller (not the Engine default): the checks the state passed, the largest value observed per
    # setting (``to_contract`` states them with a valid state: the validity holds under those settings).
    passed: tuple[Limit, ...] = ()

    # --- derived quantities (mm, rad) ------------------------------------------------------------
    def _lam_mm(self) -> float:
        return self.wavelength_nm * NM_TO_MM

    def radius(self, axis: str) -> float:
        q, m2 = (self.qx, self.m2x) if axis == "x" else (self.qy, self.m2y)
        # |q|^2 as re^2 + im^2 (no hypot: libm's may differ in the last bit between runtimes, and the radius is reported)
        w2 = (m2 * self._lam_mm() / (math.pi * self.n)) * (q.real * q.real + q.imag * q.imag) / q.imag
        return math.sqrt(w2)

    def rayleigh(self, axis: str) -> float:
        return (self.qx if axis == "x" else self.qy).imag

    def waist_radius(self, axis: str) -> float:
        q, m2 = (self.qx, self.m2x) if axis == "x" else (self.qy, self.m2y)
        return math.sqrt(m2 * self._lam_mm() * q.imag / (math.pi * self.n))

    def waist_distance(self, axis: str) -> float:
        return (self.qx if axis == "x" else self.qy).real

    def far_field_half_angle(self, axis: str) -> float:
        return self.waist_radius(axis) / self.rayleigh(axis)

    def local_divergence(self, axis: str) -> float:
        q = self.qx if axis == "x" else self.qy
        zr = q.imag
        ratio = q.real / zr
        return self.far_field_half_angle(axis) * ratio / math.sqrt(1.0 + ratio * ratio)

    def curvature_radius(self, axis: str) -> float | None:
        q = self.qx if axis == "x" else self.qy
        if q.real == 0.0:
            return None
        return (q.real * q.real + q.imag * q.imag) / q.real

    def reported_curvature_radius(self, axis: str) -> float | None:
        """The curvature radius a result states (``None``: flat). ``|q|^2 / z`` carries the relative error of its waist
        distance ``z``, which is resolved to ``WAIST_DISTANCE_RESOLUTION_MM``; where ``z`` is within
        ``FLAT_WAIST_DISTANCE_MM`` of the waist that error exceeds the cross-runtime tolerance and the digits (and the sign)
        of the radius are rounding, so the wavefront reads flat -- unless it departs from flat by more than
        ``FLAT_WAVEFRONT_PHASE_RAD`` across the beam (``|z| / z_R``, a tight focus), where the radius is stated."""

        q = self.qx if axis == "x" else self.qy
        if abs(q.real) < FLAT_WAIST_DISTANCE_MM and abs(q.real) <= FLAT_WAVEFRONT_PHASE_RAD * q.imag:
            return None
        return self.curvature_radius(axis)

    def with_(self, **changes: object) -> GState:
        return replace(self, **changes)  # type: ignore[arg-type]

    def invalidated(self, code: ErrorCode, text: str, status: Status = Status.OUT_OF_MODEL_SCOPE) -> GState:
        """No first-order state exists from here on (twisted astigmatism, a surface without usable curvature, an
        anisotropic medium, a profile that is not a beam's)."""
        return replace(self, valid=False, status=status, reason=code, reason_text=text, limit=None, passed=())

    def limited(self, limit: Limit, text: str) -> GState:
        """An accuracy limit of the first-order model is exceeded: the state leaves the model's scope with the limit's code and
        records the failed check. A state that is already not valid keeps its own reason."""
        if not self.valid:
            return self
        return replace(self, valid=False, status=Status.OUT_OF_MODEL_SCOPE, reason=limit.code, reason_text=text, limit=limit, passed=())

    def located(self, at: str) -> GState:
        """Place the limit it exceeded (the first place only)."""
        if self.limit is None or self.limit.at is not None:
            return self
        return replace(self, limit=replace(self.limit, at=at))

    def within(self, limit: Limit, cfg: NumericalConfig) -> GState:
        """Note a check the valid state passed against a limit the caller set (not the Engine default)."""
        if not self.valid or getattr(DEFAULT_NUMERICAL_CONFIG, limit.setting) == getattr(cfg, limit.setting):
            return self
        before = next((p for p in self.passed if p.setting == limit.setting), None)
        if before is not None and before.observed >= limit.observed:
            return self
        return replace(self, passed=(*(p for p in self.passed if p.setting != limit.setting), limit))

    def to_contract(self) -> GaussianBeamState:
        if not self.valid:
            # (no number of a state outside the model's scope is published; the failed accuracy check is)
            return GaussianBeamState(valid=False, status=self.status, reason=self.reason.value if self.reason else None,
                                     validity=() if self.limit is None else (self.limit.check(False),))
        basis = None
        if self.passed:
            basis = {WHOLE_RECORD: ValueBasis(kind=ValueKind.MODEL, validity=tuple(
                p.check(True) for p in sorted(self.passed, key=lambda p: p.setting)))}
        return GaussianBeamState(
            valid=True,
            status=self.status,
            reason=None,
            value_basis=basis,
            e1=tuple(float(v) for v in self.bx),  # type: ignore[arg-type]
            e2=tuple(float(v) for v in self.by),  # type: ignore[arg-type]
            wavelength_nm=self.wavelength_nm,
            medium_index=self.n,
            q_x_mm=(self.qx.real, self.qx.imag),
            q_y_mm=(self.qy.real, self.qy.imag),
            m2_x=self.m2x,
            m2_y=self.m2y,
            waist_radius_x_mm=self.waist_radius("x"),
            waist_radius_y_mm=self.waist_radius("y"),
            waist_distance_x_mm=self.waist_distance("x"),
            waist_distance_y_mm=self.waist_distance("y"),
            rayleigh_range_x_mm=self.rayleigh("x"),
            rayleigh_range_y_mm=self.rayleigh("y"),
            beam_radius_x_mm=self.radius("x"),
            beam_radius_y_mm=self.radius("y"),
            far_field_half_angle_x_rad=self.far_field_half_angle("x"),
            far_field_half_angle_y_rad=self.far_field_half_angle("y"),
            local_divergence_x_rad=self.local_divergence("x"),
            local_divergence_y_rad=self.local_divergence("y"),
            curvature_radius_x_mm=self.reported_curvature_radius("x"),
            curvature_radius_y_mm=self.reported_curvature_radius("y"),
        )


def initial_state(
    *,
    wavelength_nm: float,
    n: float,
    w0x: float,
    w0y: float,
    z_waist: float,
    m2x: float,
    m2y: float,
    bx: Array,
    by: Array,
    cfg: NumericalConfig,
) -> GState:
    """State at the source plane. ``z_waist`` = waist position ahead of the plane (mm, may be negative)."""
    lam = wavelength_nm * NM_TO_MM
    zrx = math.pi * n * (w0x * w0x) / (m2x * lam)
    zry = math.pi * n * (w0y * w0y) / (m2y * lam)
    state = GState(True, Status.PASS, None, "", wavelength_nm, n, complex(-z_waist, zrx), complex(-z_waist, zry), m2x, m2y, bx, by)
    return check_divergence(state, cfg)


def check_divergence(state: GState, cfg: NumericalConfig) -> GState:
    """Paraxial applicability by far-field half angle: WARNING above the warn limit, out of scope above the scope limit
    (``GState.limited``: the failed check is recorded)."""
    if not state.valid:
        return state
    theta = max(state.far_field_half_angle("x"), state.far_field_half_angle("y"))
    limit = Limit("gaussian_scope_divergence_rad", "far_field_half_angle_rad", "rad", theta, cfg.gaussian_scope_divergence_rad,
                  ErrorCode.GAUSSIAN_HIGH_NA)
    if theta > cfg.gaussian_scope_divergence_rad:
        return state.limited(
            limit,
            f"far-field half angle {theta:.4g} rad exceeds the paraxial Gaussian scope limit {cfg.gaussian_scope_divergence_rad} rad",
        )
    state = state.within(limit, cfg)
    if theta > cfg.gaussian_warn_divergence_rad and state.status is Status.PASS:
        return state.with_(status=Status.WARNING, reason=ErrorCode.GAUSSIAN_NEAR_HIGH_NA,
                           reason_text=f"far-field half angle {theta:.4g} rad is above the warning limit")
    return state
