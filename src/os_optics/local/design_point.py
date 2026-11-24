"""The tolerances of published design points in force for one lowered Project (``DesignPointTolerances``).

A coating or retarder that states a single design wavelength or a single design angle of incidence is used as published
within a tolerance of that point (``validity.py``). The tolerances are caller settings: the Project states them in
``numerical_settings.design_point`` and lowering resolves them once, with the Engine defaults for what the Project does
not state, into the run's ``NumericalConfig`` (``design_point_*`` fields, recorded in ``manifest.numerical_tolerances``)
and into this frozen value kept on ``LoweredProject.design_point``. Code that applies a design-point tolerance reads it
from here, never from a module constant; a finding or a basis entry that names one gives its ``NumericalConfig`` field
name (``setting``) and whether the Project stated it (``source``).
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from .._vendor.shared_contracts.design_catalog.enums import CoatingKind
from ..config import DEFAULT_NUMERICAL_CONFIG, NumericalConfig

#: The ``NumericalConfig`` field of each coating kind's single-angle tolerance (kinds not listed keep their point exactly).
ANGLE_SETTINGS: dict[CoatingKind, str] = {
    CoatingKind.METALLIC_REFLECTOR: "design_point_angle_metallic_reflector_deg",
    CoatingKind.DIELECTRIC_HIGH_REFLECTOR: "design_point_angle_dielectric_high_reflector_deg",
    CoatingKind.ANTI_REFLECTION: "design_point_angle_anti_reflection_deg",
    CoatingKind.BEAMSPLITTER: "design_point_angle_beamsplitter_deg",
}
WAVELENGTH_SETTING = "design_point_wavelength_rel"
RETARDER_WAVELENGTH_SETTING = "design_point_retarder_wavelength_rel"
INTERFERENCE_SHIFT_SETTING = "design_point_interference_shift_fraction"
INTERFERENCE_ANGLE_MAX_SETTING = "design_point_interference_angle_max_deg"
#: Every design-point setting, in ``numerical_settings.design_point`` order.
DESIGN_POINT_SETTINGS: tuple[str, ...] = (WAVELENGTH_SETTING, RETARDER_WAVELENGTH_SETTING, *ANGLE_SETTINGS.values(),
                                          INTERFERENCE_SHIFT_SETTING, INTERFERENCE_ANGLE_MAX_SETTING)
PROJECT = "PROJECT"
ENGINE_DEFAULT = "ENGINE_DEFAULT"


def _decimal(value: float) -> Decimal:
    # (the shortest repr of the float: 0.005 -> Decimal("0.005"), the exact decimal validity.py compares with)
    return Decimal(repr(float(value)))


@dataclass(frozen=True)
class DesignPointTolerances:
    """The design-point tolerances in force (module docstring). Hashable, so it can key a cache of compiled elements."""

    wavelength_rel: Decimal
    retarder_wavelength_rel: Decimal
    #: ``(coating kind, deg)`` per kind with a single-angle tolerance, in ``ANGLE_SETTINGS`` order.
    angles_deg: tuple[tuple[CoatingKind, float], ...]
    interference_shift_fraction: float
    interference_angle_max_deg: float
    #: The ``NumericalConfig`` field names of the tolerances the Project stated (the others are the Engine defaults).
    stated: frozenset[str] = frozenset()

    @classmethod
    def of(cls, cfg: NumericalConfig, stated: frozenset[str] | tuple[str, ...] = frozenset()) -> DesignPointTolerances:
        """The tolerances of a run's configuration; ``stated``: the setting names the Project gave."""

        return cls(
            wavelength_rel=_decimal(cfg.design_point_wavelength_rel),
            retarder_wavelength_rel=_decimal(cfg.design_point_retarder_wavelength_rel),
            angles_deg=tuple((kind, float(getattr(cfg, name))) for kind, name in ANGLE_SETTINGS.items()),
            interference_shift_fraction=float(cfg.design_point_interference_shift_fraction),
            interference_angle_max_deg=float(cfg.design_point_interference_angle_max_deg),
            stated=frozenset(name for name in stated if name in DESIGN_POINT_SETTINGS),
        )

    def angle_deg(self, kind: CoatingKind) -> float:
        """The single-angle tolerance of coating kind ``kind`` (0 for a kind without one)."""

        return dict(self.angles_deg).get(kind, 0.0)

    def source(self, setting: str) -> str:
        """``PROJECT`` when the Project stated ``setting``, else ``ENGINE_DEFAULT``."""

        return PROJECT if setting in self.stated else ENGINE_DEFAULT


#: The Engine defaults (a Project that states no design-point tolerance).
DEFAULT_DESIGN_POINT = DesignPointTolerances.of(DEFAULT_NUMERICAL_CONFIG)

__all__ = ["ANGLE_SETTINGS", "DEFAULT_DESIGN_POINT", "DESIGN_POINT_SETTINGS", "ENGINE_DEFAULT", "INTERFERENCE_ANGLE_MAX_SETTING",
           "INTERFERENCE_SHIFT_SETTING", "PROJECT", "RETARDER_WAVELENGTH_SETTING", "WAVELENGTH_SETTING", "DesignPointTolerances"]
