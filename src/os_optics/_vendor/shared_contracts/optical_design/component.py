"""A pure-optical component instance: OpenSci optical product identity + optical parameters + nominal pose
+ retained DOF intent.

A component is an optical element of the design (a lens, mirror, splitter, filter, window, waveplate, polarizer) or a
user-defined element such as the light source. A user-defined optical element names its physics through
``optical_model_id``: a model of the Project's ``optical_models`` (``optical_model``). Mounting hardware is not part of a design component
(``extra="forbid"`` rejects unknown fields): OpenSci configures the mechanics of a design when it is submitted.

Identity: ``OpticalProductRefV1.optical_sku`` is the OpenSci optical product identity of the public optical catalog
(``shared_contracts.design_catalog``, ``OS-OPT-...``), typed by that catalog's own ``OpticalSku``.
``catalog_version`` optionally pins the public catalog release the component was designed against. What the local
runtime actually simulated (catalog release + simulation model) is recorded per component in
``LocalValidationRecordV1.simulated_parts``.

Design frame: ``nominal_pose`` places the component's local frame in the Project's design frame, which follows the
OpenSci world convention (x right, y forward, **z up**, i.e. gravity along -z). Its origin is free: ``z = 0`` is a
design reference, not a table surface, and no beam height is implied. The quaternion must be unit-norm (within
``UNIT_QUATERNION_TOLERANCE``, the Engine's 1e-6); it fixes the local +Z (optical axis) *and* the roll that defines
local +X/+Y, hence which DOF is ``Tx`` vs ``Ty``.
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Annotated, Any

from pydantic import Field, StringConstraints, field_validator, model_validator

from ..base import StrictContractModel
from ..common.geometry import PoseV1
from ..common.identifiers import InstanceKey
from ..common.ordering import duplicate_keys
from ..design_catalog.optical_catalog import CatalogVersion, OpticalSku
from .design_content import CustomPartRefV1, SelectedParameterV1
from ._ordering import sort_raw
from .dof import RetainedDofV1, dof_sort_index
from .source import SourceModelV1


ModelId = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$"),
    Field(description="Project-scoped optical model id: a letter, then letters, digits, '_' or '-' (at most 64)."),
]


#: The allowed ``|1 - |q||`` of a pose quaternion: the Engine's own tolerance for user-supplied quaternions
#: (``NumericalConfig.quaternion_norm_tol``), so a quaternion written with six decimals (e.g. ``[0.707107, 0, 0.707107,
#: 0]``) is accepted by the contract exactly when the Engine accepts it.
UNIT_QUATERNION_TOLERANCE = 1e-6


def require_unit_quaternion(pose: PoseV1, where: str) -> None:
    norm = math.sqrt(sum(c * c for c in pose.quaternion_wxyz))
    if abs(norm - 1.0) > UNIT_QUATERNION_TOLERANCE:
        normalized = f"; normalized: {[round(c / norm, 9) for c in pose.quaternion_wxyz]}" if norm > 0.0 else ""
        raise ValueError(f"{where}: quaternion_wxyz must be unit-norm within {UNIT_QUATERNION_TOLERANCE:g} "
                         f"(|q| = {norm!r}){normalized}")


class OpticalRoleV1(StrEnum):
    """The optical function of a component (the Engine's component-role spelling).

    ``other_optical`` covers any other element that directly takes part in the optical system.
    """

    SOURCE = "source"
    LENS = "lens"
    MIRROR = "mirror"
    SPLITTER = "splitter"
    FILTER = "filter"
    WINDOW = "window"
    POLARIZER = "polarizer"
    WAVEPLATE = "waveplate"
    APERTURE = "aperture"
    PRISM = "prism"
    DETECTOR = "detector"
    DUMP = "dump"
    OTHER_OPTICAL = "other_optical"


#: Roles whose user-defined components are not optical elements and never carry an ``optical_model_id``: the light
#: source (``source_model``) and terminal instruments (an absorbing observation plane).
NON_OPTICAL_MODEL_ROLES = frozenset({OpticalRoleV1.SOURCE, OpticalRoleV1.DETECTOR, OpticalRoleV1.DUMP})


class OpticalProductRefV1(StrictContractModel):
    """Reference to a product of the public optical catalog (``PublicOpticalCatalogV2``)."""

    optical_sku: OpticalSku
    catalog_version: CatalogVersion | None = Field(
        None, description="Optional pin of the public optical catalog release this component was designed against."
    )


class OpticalComponentInstanceV1(StrictContractModel):
    """One placed optical component at the design stage.

    Exactly one of ``product`` (a public OpenSci optical product) or ``custom_part`` (a user-defined component that
    is not a catalog product, such as the light source) is set.
    ``selected_parameters`` are keyed by ``semantic_key`` and canonicalized; ``retained_dofs`` are keyed by
    ``axis`` and canonicalized into ``DOF_AXIS_ORDER`` (Tx, Ty, Tz, Rx, Ry, Rz). An empty
    ``retained_dofs`` means every DOF is fixed at the nominal pose once the system is built.
    ``source_model`` (``source.SourceModelV1``: a Gaussian beam, an extended emitter, a sampled intensity map, a sampled
    complex field or a speckle field) is the explicit emission model a user-defined
    source needs for simulation; every other component leaves it unset. ``optical_model_id`` names the declared optical
    model (``ProjectDocumentV2.optical_models``) of a user-defined optical element; only a ``custom_part`` that is not a
    source, detector or dump carries one.
    """

    instance_id: InstanceKey
    optical_role: OpticalRoleV1
    product: OpticalProductRefV1 | None = None
    custom_part: CustomPartRefV1 | None = None
    nominal_pose: PoseV1 = Field(description="Designed pose of the component's local frame in the design (global) frame.")
    selected_parameters: tuple[SelectedParameterV1, ...] = ()
    retained_dofs: tuple[RetainedDofV1, ...] = ()
    source_model: SourceModelV1 | None = Field(
        None,
        description="Pure-optical emission model of a user-defined source; only a component with optical_role 'source' may carry one.",
    )
    optical_model_id: ModelId | None = Field(
        None, description="Declared optical model of a user-defined optical element (ProjectDocumentV2.optical_models)."
    )

    @field_validator("selected_parameters", mode="before")
    @classmethod
    def _canonicalize_selected_parameters(cls, value: Any) -> Any:
        return sort_raw(value, lambda get, item: get(item, "semantic_key"))

    @field_validator("retained_dofs", mode="before")
    @classmethod
    def _canonicalize_retained_dofs(cls, value: Any) -> Any:
        return sort_raw(value, lambda get, item: dof_sort_index(get(item, "axis")))

    @model_validator(mode="after")
    def _check(self) -> OpticalComponentInstanceV1:
        if (self.product is None) == (self.custom_part is None):
            raise ValueError(f"component {self.instance_id!r}: exactly one of product or custom_part must be set")
        require_unit_quaternion(self.nominal_pose, f"component {self.instance_id!r} nominal_pose")
        if self.source_model is not None and self.optical_role is not OpticalRoleV1.SOURCE:
            raise ValueError(f"component {self.instance_id!r}: only an optical_role 'source' component may carry a source_model")
        if self.optical_model_id is not None:
            if self.custom_part is None:
                raise ValueError(f"component {self.instance_id!r}: optical_model_id belongs to a custom_part component, not a catalog product")
            if self.optical_role in NON_OPTICAL_MODEL_ROLES:
                raise ValueError(f"component {self.instance_id!r}: a {self.optical_role.value} component has no optical_model_id")
        param_duplicates = duplicate_keys(self.selected_parameters, "semantic_key")
        if param_duplicates:
            raise ValueError(f"duplicate selected_parameters semantic_key in component {self.instance_id!r}: {param_duplicates}")
        dof_duplicates = duplicate_keys(self.retained_dofs, "axis")
        if dof_duplicates:
            raise ValueError(f"duplicate retained_dofs axis in component {self.instance_id!r}: {dof_duplicates}")
        return self
