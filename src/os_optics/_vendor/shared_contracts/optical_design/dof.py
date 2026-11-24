"""Design-stage six-degree-of-freedom intent.

A retained DOF says: *after this optical system is built, this degree of freedom of this optical component remains
adjustable*. It is design intent expressed on the optical component; the hardware that provides the adjustment is
configured by OpenSci for the submitted design, and the commercial resolution states how each retained DOF is
delivered.

Frame semantics (fixed, not configurable -- ``DOF_REFERENCE_FRAME``):

* every axis is expressed in the optical component's own LOCAL frame, i.e. the frame whose pose is
  the component's ``nominal_pose`` (``x_global = R(q) x_local + position_mm``, ``PoseV1``);
* local +Z is the component's principal optical / reference axis; local +X and +Y are the two
  orthogonal transverse axes;
* ``Tx``/``Ty``/``Tz`` translate along local +X/+Y/+Z (unit: mm);
* ``Rx``/``Ry``/``Rz`` rotate right-handed about local +X/+Y/+Z through the component's local
  origin (unit: degrees). Mirror-mount tip/tilt is therefore ``Rx`` + ``Ry``; rotation about the
  optical axis is ``Rz``.

Control intent: ``MANUAL`` (default; adjusted by hand when the system is aligned) or ``NUMERICAL`` (intended to be
adjusted by a numerically controlled actuator, e.g. by software or an Agent during operation).

Stated limits (optional ``DOF_LIMIT_FIELDS`` of ``RetainedDofV1`` and ``RetainedDofStateV1``): the hardware that
provides the adjustment as the design states it, in the axis's unit (``dof_unit``: mm or deg), relative to the nominal
pose -- ``travel`` (the interval the stage reaches, ``min <= 0 <= max``), ``hard_limits`` (the end stops, containing the
travel), ``repeatability`` (+/-, positive) and ``drift`` (magnitudes per kelvin and / or per hour, positive), with an
optional ``unit`` that must equal ``dof_unit(axis)``. ``hard_limits``, ``repeatability``, ``drift`` and ``unit`` describe
the stage that provides the travel and come with ``travel``. Every field defaults to absent, so a Project or record that states
none keeps its exact payload and hash. A run lowers a DOF with ``travel`` into an Engine actuator; targets on the
``dof_*`` metrics compare the stated limits with what the design needs, and a DOF without them leaves such a target
undetermined.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from ..base import StrictContractModel
from ..common.geometry import Vector3


DOF_REFERENCE_FRAME: Literal["COMPONENT_LOCAL"] = "COMPONENT_LOCAL"


class DofAxisV1(StrEnum):
    """One of the six component-local design DOFs: translations ``Tx``/``Ty``/``Tz`` along and rotations
    ``Rx``/``Ry``/``Rz`` about the component's local x, y, z axes (local +Z is the optical axis)."""

    TX = "Tx"
    TY = "Ty"
    TZ = "Tz"
    RX = "Rx"
    RY = "Ry"
    RZ = "Rz"


class DofMotionV1(StrEnum):
    TRANSLATION = "TRANSLATION"
    ROTATION = "ROTATION"


class DofControlIntentV1(StrEnum):
    """How a retained DOF is intended to be operated after the system is built."""

    MANUAL = "MANUAL"
    NUMERICAL = "NUMERICAL"


# Canonical wire order of a component's retained DOFs (translations first, then rotations, X/Y/Z).
DOF_AXIS_ORDER: tuple[DofAxisV1, ...] = (DofAxisV1.TX, DofAxisV1.TY, DofAxisV1.TZ, DofAxisV1.RX, DofAxisV1.RY, DofAxisV1.RZ)

_LOCAL_AXIS: dict[DofAxisV1, Vector3] = {
    DofAxisV1.TX: (1.0, 0.0, 0.0),
    DofAxisV1.TY: (0.0, 1.0, 0.0),
    DofAxisV1.TZ: (0.0, 0.0, 1.0),
    DofAxisV1.RX: (1.0, 0.0, 0.0),
    DofAxisV1.RY: (0.0, 1.0, 0.0),
    DofAxisV1.RZ: (0.0, 0.0, 1.0),
}


def dof_motion(axis: DofAxisV1) -> DofMotionV1:
    """TRANSLATION for ``T*``, ROTATION for ``R*``."""

    return DofMotionV1.TRANSLATION if axis.value.startswith("T") else DofMotionV1.ROTATION


def dof_local_axis(axis: DofAxisV1) -> Vector3:
    """The unit vector, in the component's local frame, that the DOF translates along / rotates about."""

    return _LOCAL_AXIS[axis]


def dof_unit(axis: DofAxisV1) -> Literal["mm", "deg"]:
    """The fixed unit of any offset/range expressed for ``axis``: mm for translations, degrees for rotations."""

    return "mm" if dof_motion(axis) is DofMotionV1.TRANSLATION else "deg"


def dof_sort_index(axis: DofAxisV1 | str) -> int:
    """Position of ``axis`` in ``DOF_AXIS_ORDER``; unknown spellings sort last (their own validation rejects them)."""

    try:
        return DOF_AXIS_ORDER.index(DofAxisV1(axis))
    except ValueError:
        return len(DOF_AXIS_ORDER)


class DofIntervalV1(StrictContractModel):
    """A closed interval of a retained DOF's offset from the nominal pose, in the axis's unit (``dof_unit``). The nominal
    pose lies inside it: ``min <= 0 <= max``."""

    min: float = Field(le=0.0, description="Lowest offset from the nominal pose (<= 0), in the axis's unit.")
    max: float = Field(ge=0.0, description="Highest offset from the nominal pose (>= 0), in the axis's unit.")


class DofDriftV1(StrictContractModel):
    """How far a retained DOF's position drifts, as magnitudes in the axis's unit: per kelvin of temperature change and / or
    per hour. At least one is stated."""

    per_kelvin: float | None = Field(None, gt=0.0, description="Drift magnitude per kelvin (axis unit / K).")
    per_hour: float | None = Field(None, gt=0.0, description="Drift magnitude per hour (axis unit / h).")

    @model_validator(mode="after")
    def _stated(self) -> DofDriftV1:
        if self.per_kelvin is None and self.per_hour is None:
            raise ValueError("a drift states per_kelvin and / or per_hour")
        return self


#: The optional stated-limit fields of a retained DOF (``RetainedDofV1`` and ``RetainedDofStateV1`` carry them; module
#: docstring), each in the axis's unit.
DOF_LIMIT_FIELDS: tuple[str, ...] = ("travel", "hard_limits", "repeatability", "drift", "unit")
TRAVEL_DESCRIPTION = "The interval the stage reaches, relative to the nominal pose, in the axis's unit."
HARD_LIMITS_DESCRIPTION = "The mechanical end stops, relative to the nominal pose; they contain the travel."
REPEATABILITY_DESCRIPTION = "+/- spread of returning to a commanded position, in the axis's unit."
DRIFT_DESCRIPTION = "Drift magnitudes per kelvin and / or per hour, in the axis's unit."
UNIT_DESCRIPTION = "The axis's unit, stated for reading: mm for Tx/Ty/Tz, deg for Rx/Ry/Rz."


def check_dof_limits(dof: object, axis: DofAxisV1) -> None:
    """Refuse stated limits that contradict each other or the axis (module docstring): a ``unit`` other than
    ``dof_unit(axis)``, ``hard_limits`` / ``repeatability`` / ``drift`` / ``unit`` without ``travel``, hard limits not
    containing the travel. (The DOF's nominal configuration has no offset: the limits are the stage's, not a position.)"""

    unit, travel, hard = getattr(dof, "unit"), getattr(dof, "travel"), getattr(dof, "hard_limits")
    if unit is not None and unit != dof_unit(axis):
        raise ValueError(f"retained DOF {axis.value} is in {dof_unit(axis)}, not {unit}")
    if travel is None:
        if any(getattr(dof, name) is not None for name in ("hard_limits", "repeatability", "drift", "unit")):
            raise ValueError(f"retained DOF {axis.value}: hard_limits, repeatability, drift and unit describe the stage that "
                             "provides its travel; state travel with them")
        return
    if hard is not None and not (hard.min <= travel.min and travel.max <= hard.max):
        raise ValueError(f"retained DOF {axis.value}: the hard limits must contain the travel")


class RetainedDofV1(StrictContractModel):
    """One retained design DOF of one optical component: which local axis, how it is meant to be controlled, and optionally
    the stated limits of the stage that provides it (module docstring)."""

    axis: DofAxisV1
    frame: Literal["COMPONENT_LOCAL"] = Field(
        DOF_REFERENCE_FRAME,
        description="Always the owning optical component's local frame (local +Z = principal optical/reference axis).",
    )
    control: DofControlIntentV1 = DofControlIntentV1.MANUAL
    travel: DofIntervalV1 | None = Field(None, description=TRAVEL_DESCRIPTION)
    hard_limits: DofIntervalV1 | None = Field(None, description=HARD_LIMITS_DESCRIPTION)
    repeatability: float | None = Field(None, gt=0.0, description=REPEATABILITY_DESCRIPTION)
    drift: DofDriftV1 | None = Field(None, description=DRIFT_DESCRIPTION)
    unit: Literal["mm", "deg"] | None = Field(None, description=UNIT_DESCRIPTION)

    @model_validator(mode="after")
    def _limits(self) -> RetainedDofV1:
        check_dof_limits(self, self.axis)
        return self
