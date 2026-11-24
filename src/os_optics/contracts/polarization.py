"""JonesVector and PolarizationBasis contracts."""

from __future__ import annotations

import math

from pydantic import ValidationInfo, field_validator, model_validator

from ..config import DEFAULT_NUMERICAL_CONFIG
from .base import Complex, StrictModel, Vector3, unit_direction

_BASIS_TOL = DEFAULT_NUMERICAL_CONFIG.basis_orthonormality_tol


class JonesVector(StrictModel):
    """Jones vector ``(Ex, Ey)`` expressed in the ray's transverse basis ``(e1, e2)``, ``e1 x e2 = d``.

    Field convention: ``E ~ exp[i(k.r - w t)]``. A source's vector states its polarization only (power is carried
    separately): any non-zero vector, divided by its norm at lowering; propagated states may have norm <= 1 (amplitude loss).
    """

    ex: Complex
    ey: Complex

    @property
    def norm(self) -> float:
        return math.sqrt(self.ex[0] ** 2 + self.ex[1] ** 2 + self.ey[0] ** 2 + self.ey[1] ** 2)

    @classmethod
    def linear(cls, angle_deg: float) -> JonesVector:
        """Linear polarization at ``angle_deg`` from ``e1`` toward ``e2`` (right-handed about ``d``)."""
        a = math.radians(angle_deg)
        return cls(ex=(math.cos(a), 0.0), ey=(math.sin(a), 0.0))

    @classmethod
    def right_circular(cls) -> JonesVector:
        """``(1, -i)/sqrt2``: clockwise when viewed facing the source (Hecht convention for exp[i(kz-wt)])."""
        s = 1.0 / math.sqrt(2.0)
        return cls(ex=(s, 0.0), ey=(0.0, -s))

    @classmethod
    def left_circular(cls) -> JonesVector:
        s = 1.0 / math.sqrt(2.0)
        return cls(ex=(s, 0.0), ey=(0.0, s))

    @classmethod
    def elliptical(cls, azimuth_deg: float, ellipticity_deg: float) -> JonesVector:
        """Ellipse with major axis at ``azimuth_deg`` and ellipticity angle ``chi`` (``tan chi`` = minor/major).

        ``chi = 0`` linear, ``chi = -45 deg`` right circular, ``+45 deg`` left circular (same handedness
        convention as ``right_circular``)."""
        psi, chi = math.radians(azimuth_deg), math.radians(ellipticity_deg)
        c, s = math.cos(chi), math.sin(chi)
        # major axis component (c) real, minor axis component i*s; then rotate by psi
        ex = (c * math.cos(psi), -s * math.sin(psi))
        ey = (c * math.sin(psi), s * math.cos(psi))
        return cls(ex=ex, ey=ey)


class PolarizationBasis(StrictModel):
    """Orthonormal transverse basis of a ray: ``e1 x e2 = direction`` (right-handed)."""

    direction: Vector3
    e1: Vector3
    e2: Vector3

    @field_validator("direction", "e1", "e2")
    @classmethod
    def _unit(cls, v: Vector3, info: ValidationInfo) -> Vector3:
        return unit_direction(v, _BASIS_TOL, str(info.field_name))  # type: ignore[return-value]

    @model_validator(mode="after")
    def _ortho(self) -> PolarizationBasis:
        def dot(a: Vector3, b: Vector3) -> float:
            return sum(x * y for x, y in zip(a, b))

        if abs(dot(self.direction, self.e1)) > _BASIS_TOL or abs(dot(self.direction, self.e2)) > _BASIS_TOL or abs(dot(self.e1, self.e2)) > _BASIS_TOL:
            raise ValueError("basis vectors must be mutually orthogonal")
        return self
