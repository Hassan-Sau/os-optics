"""Design content of a Project: customer-supplied components, selected parameters, observation planes, targets,
materials and clear apertures."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Annotated, Any, ClassVar, Literal

from pydantic import Field, StrictBool, StrictFloat, StringConstraints, ValidationInfo, field_validator, model_validator

from ..base import OmitAbsentFields, StrictContractModel
from ..common.decimal_string import DECIMAL_STRING_PATTERN, DecimalString
from ..common.geometry import PoseV1
from ..common.identifiers import InstanceKey, OpenSciId
from ..common.ordering import canonicalize_by_key, duplicate_keys
from .detection import DetectionV1


class ParameterValueKind(StrEnum):
    """Value kind of a parameter. No semantic key vocabulary is implied."""

    DECIMAL = "DECIMAL"
    RANGE = "RANGE"
    VECTOR = "VECTOR"
    BOOLEAN = "BOOLEAN"
    ENUM = "ENUM"
    TEXT = "TEXT"
    SERIES = "SERIES"


UnitCode = Annotated[
    str,
    StringConstraints(strict=True, min_length=1, max_length=40),
    Field(description="Free-form unit token, as stated. No unit conversion and no fixed unit vocabulary."),
]

ParameterTextValue = Annotated[
    str,
    StringConstraints(strict=True, min_length=1, max_length=4096),
    Field(description="ENUM member or TEXT value. No enum vocabulary is fixed by this contract."),
]

_DECIMAL_PATTERN = re.compile(DECIMAL_STRING_PATTERN)
_NO_SELECTED_VALUE_SHAPE = frozenset({ParameterValueKind.RANGE, ParameterValueKind.VECTOR, ParameterValueKind.SERIES})


def check_value_matches_kind(value_kind: ParameterValueKind, value: object) -> None:
    """Raise ``ValueError`` if ``value``'s Python representation does not match ``value_kind`` (for example
    ``DECIMAL`` + ``"hello"``, ``BOOLEAN`` + ``"true"``, ``TEXT`` + ``True``)."""

    if value_kind is ParameterValueKind.DECIMAL:
        if not (isinstance(value, str) and _DECIMAL_PATTERN.fullmatch(value) is not None):
            raise ValueError(f"DECIMAL value must be a canonical DecimalString, got {value!r}")
    elif value_kind is ParameterValueKind.BOOLEAN:
        if not isinstance(value, bool):
            raise ValueError(f"BOOLEAN value must be an actual bool, got {value!r}")
    elif value_kind in (ParameterValueKind.ENUM, ParameterValueKind.TEXT):
        if not isinstance(value, str) or isinstance(value, bool):
            raise ValueError(f"{value_kind} value must be a string, got {value!r}")
    else:
        raise ValueError(f"{value_kind} has no v1 value shape")


class CustomPartRefV1(StrictContractModel):
    """A user-defined component that is not a catalog product (for example the user's light source): a stable key and
    a free-text description."""

    custom_part_key: OpenSciId
    description: str = Field(min_length=1, max_length=1000)


class SelectedParameterV1(StrictContractModel):
    """Design intent: a proposed value for one named parameter of a component.

    ``value`` must be a well-formed instance of its declared ``value_kind`` (a ``DECIMAL`` selection is a canonical
    ``DecimalString``). ``RANGE`` / ``VECTOR`` / ``SERIES`` have no selected-value shape and are rejected.
    """

    semantic_key: OpenSciId
    value_kind: ParameterValueKind
    value: DecimalString | StrictBool | ParameterTextValue
    unit_code: UnitCode | None = None

    @model_validator(mode="after")
    def _value_matches_kind(self) -> SelectedParameterV1:
        if self.value_kind in _NO_SELECTED_VALUE_SHAPE:
            raise ValueError(f"{self.value_kind} has no v1 selected-value shape and cannot be selected")
        check_value_matches_kind(self.value_kind, self.value)
        return self


UnitLabel = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=40)]
Vector2V1 = Annotated[tuple[float, float], Field(description="Cartesian 2D vector (x, y) or (u, v).")]


class CircularApertureV1(StrictContractModel):
    """Circular clear aperture of a plane or surface."""

    shape: Literal["circle"] = "circle"
    radius_mm: float = Field(gt=0, description="Clear aperture radius (not diameter).")
    center_offset_mm: Vector2V1 = (0.0, 0.0)
    obstruction_radius_mm: float = Field(0.0, ge=0, description="Central circular obstruction radius.")

    @model_validator(mode="after")
    def _obstruction(self) -> CircularApertureV1:
        if self.obstruction_radius_mm >= self.radius_mm:
            raise ValueError("obstruction_radius_mm must be smaller than radius_mm")
        return self


class RectangularApertureV1(StrictContractModel):
    """Rectangular clear aperture of a plane or surface."""

    shape: Literal["rectangle"] = "rectangle"
    half_width_mm: float = Field(gt=0, description="Half extent along the surface u axis.")
    half_height_mm: float = Field(gt=0, description="Half extent along the surface v axis.")
    center_offset_mm: Vector2V1 = (0.0, 0.0)


ApertureV1 = Annotated[CircularApertureV1 | RectangularApertureV1, Field(discriminator="shape")]


class ObservationV1(OmitAbsentFields, StrictContractModel):
    """A virtual observation plane: a detector, screen or measurement plane where the beams are evaluated.

    ``aperture``, ``acceptance_half_angle_deg`` and ``medium`` are the plane's own geometry and acceptance, stated
    explicitly by the design.

    Frame: ``pose`` is in the design frame, unless the plane is attached to a component (``device_instance_id``): then
    ``pose`` is in that component's local frame and the plane's design-frame pose is the component's ``nominal_pose``
    composed with it (the identity ``pose`` puts the plane at the component's origin, normal along its local +Z).
    The local runtime, the assembly resolution and the Assembly Preview all place the plane there.
    """

    # Serialized only when given: a document without it keeps its earlier payload and hash.
    _omit_when_none: ClassVar[tuple[str, ...]] = ("detection",)

    observation_id: OpenSciId
    device_instance_id: InstanceKey | None = Field(
        None, description="Component this plane is attached to, if any (e.g. the sensor of a camera). When set, `pose` is "
                          "relative to that component: it is in the component's local frame (identity pose = at the "
                          "component's origin, normal along its local +Z), not in the design frame.")
    pose: PoseV1 = Field(PoseV1(), description="Plane pose (local +Z = plane normal): in the design frame, or in the "
                                                "attached component's local frame when `device_instance_id` is set.")
    aperture: ApertureV1 | None = Field(None, description="Active area in the plane (u = frame x, v = frame y).")
    acceptance_half_angle_deg: float | None = Field(None, gt=0, le=90, description="Maximum AOI accepted by the detector.")
    absorbing: bool = Field(False, description="An absorbing plane (a detector face, a beam dump) stops the beams that reach its "
                                               "active area: the chief ray inside `aperture`, or at least 0.1 % of the beam's power "
                                               "inside it. Beams crossing the plane outside the active area pass on unchanged (they "
                                               "are reported as crossings outside the active area). Without an `aperture` the whole "
                                               "infinite plane absorbs.")
    medium: OpenSciId = "air"
    detection: DetectionV1 | None = Field(
        None, description="A detector on this plane (its sensitive face) and, optionally, a lock-in measurement of modulated "
                          "light: photocurrent, noise and SNR of the light the plane counts (detection.DetectionV1).")


class TargetParameterEntryV1(StrictContractModel):
    """One entry of a target's metric parameter map: a JSON number or string, never a boolean or a nested object.

    Which keys a target takes and the type of each depend on its metric; the local runtime checks them
    (``os-optics results metrics`` lists each metric's ``selectors``)."""

    key: OpenSciId
    value: StrictFloat | str = Field(description="A JSON number or string; a boolean is refused.")

    @field_validator("value", mode="before")
    @classmethod
    def _no_boolean(cls, value: Any, info: ValidationInfo) -> Any:
        if isinstance(value, bool):
            raise ValueError(f"target parameter {info.data.get('key')!r}: a boolean is not a parameter value (a JSON number or string)")
        return value


class TargetV1(StrictContractModel):
    """A verifiable requirement of the design: a metric compared with ``target_value`` (within ``tolerance_abs`` /
    ``tolerance_rel``) and/or ``minimum`` / ``maximum``.

    ``source_id`` / ``source_id_b`` / ``wavelength_nm`` / ``wavelength_nm_b`` / ``parameters`` / ``source_revision``
    select which beam(s), spectral component(s) or metric-specific scalar the requirement applies to; leave them unset
    when the metric does not use them. ``wavelength_nm_b`` selects the spectral line of a pair metric's second operand
    (``wavelength_nm`` the first's), so a pair metric can compare two lines. Every metric's definition, sign convention, operand order and frame is published with the local runtime
    (``os_optics.local.metric_descriptors()``, ``os-optics results metrics``).
    """

    target_id: OpenSciId
    metric: OpenSciId
    device_instance_id: InstanceKey | None = None
    observation_id: OpenSciId | None = None
    source_id: OpenSciId | None = None
    source_id_b: OpenSciId | None = Field(None, description="Second beam for two-beam metrics.")
    wavelength_nm: float | None = Field(None, gt=0, description="Select one spectral component; default: the first.")
    wavelength_nm_b: float | None = Field(
        None, gt=0, description="Spectral component of the second operand of a pair metric (wavelength_nm selects the first's); "
        "used by pair metrics only.")
    target_value: float | None = None
    tolerance_abs: float | None = Field(None, ge=0)
    tolerance_rel: float | None = Field(None, ge=0)
    minimum: float | None = None
    maximum: float | None = None
    unit: UnitLabel | None = None
    parameters: tuple[TargetParameterEntryV1, ...] = Field(
        (), description="Metric parameters, one entry per `key`, sorted by key.")
    source_revision: str | None = Field(None, max_length=128, description="Free-form label of the beam or run artifact this target selects.")
    configuration_ids: tuple[OpenSciId, ...] | None = Field(
        None, min_length=1, description="The Project configurations this target applies to (ProjectDocumentV2.configurations); "
        "absent: every configuration, and a Project without configurations.")

    @field_validator("parameters", mode="before")
    @classmethod
    def _canonicalize_parameters(cls, value: Any) -> Any:
        return canonicalize_by_key(value, "key")

    @model_validator(mode="after")
    def _comparison(self) -> TargetV1:
        if self.target_value is None and self.minimum is None and self.maximum is None:
            raise ValueError("a target needs target_value and/or minimum/maximum")
        if self.minimum is not None and self.maximum is not None and self.maximum < self.minimum:
            raise ValueError("maximum must be >= minimum")
        return self

    @model_validator(mode="after")
    def _unique_parameter_keys(self) -> TargetV1:
        duplicates = duplicate_keys(self.parameters, "key")
        if duplicates:
            raise ValueError(f"duplicate parameters key in target {self.target_id!r}: {duplicates}")
        return self

    @model_validator(mode="after")
    def _unique_configuration_ids(self) -> TargetV1:
        if self.configuration_ids is not None and len(set(self.configuration_ids)) != len(self.configuration_ids):
            raise ValueError(f"duplicate configuration_ids in target {self.target_id!r}")
        return self


class ProjectMaterialV1(StrictContractModel):
    """A project-defined material identity: ``material_id`` and a free ``kind`` identifier (e.g. "constant")."""

    material_id: OpenSciId
    kind: OpenSciId


__all__ = [
    "ApertureV1",
    "CircularApertureV1",
    "CustomPartRefV1",
    "ObservationV1",
    "ParameterTextValue",
    "ParameterValueKind",
    "ProjectMaterialV1",
    "RectangularApertureV1",
    "SelectedParameterV1",
    "TargetParameterEntryV1",
    "TargetV1",
    "UnitCode",
    "UnitLabel",
    "Vector2V1",
    "check_value_matches_kind",
]
