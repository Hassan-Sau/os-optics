"""Optical Layout Package: the relative optical layout of one commercially resolved design.

An optics-only purchase buys the optical elements the validated design selected and no mounting hardware. Its
deliverable is an Optical Layout Drawing that lets the customer build their own mechanics around the optics: every
element's identity, its position relative to the other elements, its orientation, the nominal beam path through the
elements and the spacing along it. The package is the data of that drawing; an OpenSci runtime renders it locally
(``opensci_engine.optical_layout``) in Simplified Chinese or English.

It carries only what the customer's own design and the public catalog already state -- design positions and
orientations, OpenSci product identities and names, the public outline of each catalog product, the nominal beam path
-- plus the commercial binding (resolution, Order Number and customer name once an Order exists). No supplier,
mechanical realization or price is in it.

Frame (``FRAME``): the axes are the design frame of the Project (+Z up as designed). Positions are design positions
minus the design position of the ``datum`` element (the customer's light source when the design has one, else the first
component), so the layout is relative: no height above a table or breadboard is implied. Elements drawn at different
design heights keep their relative heights.

``axis_direction`` is an element's local +Z in the design frame (a lens's optical axis, a mirror's or plate's face
normal, the emission direction of a light source); ``up_direction`` its local +Y. Lengths are millimetres rounded to
0.001 mm; directions are unit vectors rounded to 1e-6.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, StringConstraints, model_validator

from ..base import StrictContractModel
from ..common.hashes import Sha256Digest, osc1_digest
from ..common.identifiers import InstanceKey, OpenSciId

OPTICAL_LAYOUT_SCHEMA_ID: Literal["opensci.optical_layout.package"] = "opensci.optical_layout.package"
OPTICAL_LAYOUT_SCHEMA_VERSION: Literal["1.0"] = "1.0"
OPTICAL_LAYOUT_FILE_NAME = "optical-layout.json"
FRAME: Literal["DESIGN_FRAME_RELATIVE_TO_DATUM"] = "DESIGN_FRAME_RELATIVE_TO_DATUM"
MAX_ELEMENTS = 256
MAX_BEAM_PATHS = 1024
MAX_PATH_POINTS = 64

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]
DisplayName = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=200)]
OpticalRole = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=64, pattern=r"^[a-z_]+$")]
Hex64 = Annotated[str, StringConstraints(strict=True, pattern=r"^[0-9a-f]{64}$")]

#: Why a design component is supplied by the customer (the resolution's ``customer_supplied_components[].reason_code``).
SupplyReason = Literal["CUSTOMER_LIGHT_SOURCE", "NOT_IN_CATALOG", "NOT_ORDERABLE", "TEMPORARILY_UNAVAILABLE", "CUSTOMER_OWNED"]
DofAxis = Literal["Tx", "Ty", "Tz", "Rx", "Ry", "Rz"]


class OpticalLayoutBindingV1(StrictContractModel):
    """What the layout belongs to: the validated design and its commercial resolution; ``order_number`` and
    ``customer_display_name`` once an Order of that resolution exists (``None`` for the layout of a resolution)."""

    project_hash: Hex64
    local_validation_hash: Hex64
    public_catalog_version: OpenSciId
    commercial_resolution_id: OpenSciId
    purchase_scope: Literal["COMPLETE_SYSTEM", "OPTICS_ONLY"]
    order_number: OpenSciId | None = None
    customer_display_name: DisplayName | None = None


class OpticalLayoutOutlineV1(StrictContractModel):
    """The outline of an element in its own design frame -- a catalog product's public design geometry, or the geometry
    of the optical model the Project declares for a component of its own -- round (``diameter_mm``) or rectangular
    (``width_mm`` across local X, ``height_mm`` across local Y), from ``z_min_mm`` to ``z_max_mm`` along local +Z."""

    shape: Literal["ROUND", "RECTANGULAR"]
    diameter_mm: float | None = Field(default=None, gt=0)
    width_mm: float | None = Field(default=None, gt=0)
    height_mm: float | None = Field(default=None, gt=0)
    z_min_mm: float
    z_max_mm: float

    @model_validator(mode="after")
    def _shape(self) -> OpticalLayoutOutlineV1:
        if self.z_max_mm <= self.z_min_mm:
            raise ValueError("an outline needs z_min_mm < z_max_mm")
        if self.shape == "ROUND" and (self.diameter_mm is None or self.width_mm is not None or self.height_mm is not None):
            raise ValueError("a round outline has diameter_mm only")
        if self.shape == "RECTANGULAR" and (self.diameter_mm is not None or self.width_mm is None or self.height_mm is None):
            raise ValueError("a rectangular outline has width_mm and height_mm only")
        return self


class OpticalLayoutDofV1(StrictContractModel):
    """A retained design DOF of the element (along or about its own local axis), as the design requests it."""

    axis: DofAxis
    control: Literal["MANUAL", "NUMERICAL"]


class OpticalLayoutElementV1(StrictContractModel):
    """One design component at its relative design position.

    ``supplied_by``: ``OPENSCI`` -- an OpenSci optical product on the Order (``opensci_sku``, its names and its Purchase
    BOM ``line_key``); ``CUSTOMER`` -- supplied by the customer (``supply_reason``). A customer-supplied catalog product
    (``opensci_sku`` set) is the catalog model the design used for it: the customer's own part may differ, and only
    the catalog model's outline is known. A customer-supplied user-defined component has ``custom_component_key``; it
    has an outline when the Project declares its optical model with a body of known extent (the model's design
    geometry, a COMPOUND unit's envelope, or the outline and surfaces of the model)."""

    instance_id: InstanceKey
    optical_role: OpticalRole
    supplied_by: Literal["OPENSCI", "CUSTOMER"]
    supply_reason: SupplyReason | None = None
    opensci_sku: OpenSciId | None = None
    display_name: DisplayName | None = None
    display_name_zh_cn: DisplayName | None = None
    custom_component_key: OpenSciId | None = None
    line_key: OpenSciId | None = None
    position_mm: Vec3
    quaternion_wxyz: Quat
    axis_direction: Vec3
    up_direction: Vec3
    outline: OpticalLayoutOutlineV1 | None = None
    retained_dofs: tuple[OpticalLayoutDofV1, ...] = ()

    @model_validator(mode="after")
    def _supply(self) -> OpticalLayoutElementV1:
        if self.supplied_by == "OPENSCI":
            if self.supply_reason is not None or self.opensci_sku is None or self.line_key is None:
                raise ValueError("an OpenSci-supplied element names its product and its BOM line, and no supply reason")
        elif self.supply_reason is None or self.line_key is not None:
            raise ValueError("a customer-supplied element has a supply reason and no BOM line")
        if self.opensci_sku is not None and (self.display_name is None or self.display_name_zh_cn is None):
            raise ValueError("a catalog element carries its English and Simplified Chinese names")
        return self


class OpticalLayoutBeamPathV1(StrictContractModel):
    """One nominal beam path (one per light source and splitter branch): the elements it visits in order (``through``,
    element and observation ids) at ``points_mm`` (relative), and the length of each straight segment between
    consecutive points (``segment_lengths_mm``, one fewer than the points)."""

    path_key: OpenSciId
    wavelength_nm: float | None = Field(default=None, gt=0)
    through: tuple[OpenSciId, ...] = Field(min_length=2, max_length=MAX_PATH_POINTS)
    points_mm: tuple[Vec3, ...] = Field(min_length=2, max_length=MAX_PATH_POINTS)
    segment_lengths_mm: tuple[float, ...] = Field(min_length=1, max_length=MAX_PATH_POINTS - 1)

    @model_validator(mode="after")
    def _shape(self) -> OpticalLayoutBeamPathV1:
        if len(self.points_mm) != len(self.through) or len(self.segment_lengths_mm) != len(self.points_mm) - 1:
            raise ValueError("a beam path has one point per visited element and one segment between consecutive points")
        return self


class OpticalLayoutObservationV1(StrictContractModel):
    """An observation plane of the design (design intent, not hardware) at its relative position."""

    observation_id: OpenSciId
    position_mm: Vec3
    axis_direction: Vec3


class OpticalLayoutContentV1(StrictContractModel):
    """The content of an Optical Layout Package (everything but its ``layout_content_hash``)."""

    schema_id: Literal["opensci.optical_layout.package"] = Field(default=OPTICAL_LAYOUT_SCHEMA_ID, alias="schema")
    schema_version: Literal["1.0"] = OPTICAL_LAYOUT_SCHEMA_VERSION
    binding: OpticalLayoutBindingV1
    frame: Literal["DESIGN_FRAME_RELATIVE_TO_DATUM"] = FRAME
    datum: InstanceKey
    elements: tuple[OpticalLayoutElementV1, ...] = Field(min_length=1, max_length=MAX_ELEMENTS)
    beam_paths: tuple[OpticalLayoutBeamPathV1, ...] = Field(default=(), max_length=MAX_BEAM_PATHS)
    observations: tuple[OpticalLayoutObservationV1, ...] = ()

    @model_validator(mode="after")
    def _references(self) -> OpticalLayoutContentV1:
        ids = [e.instance_id for e in self.elements]
        if len(set(ids)) != len(ids):
            raise ValueError("element instance ids are unique")
        if self.datum not in ids:
            raise ValueError("the datum is one of the elements")
        known = set(ids) | {o.observation_id for o in self.observations}
        for path in self.beam_paths:
            unknown = sorted(set(path.through) - known)
            if unknown:
                raise ValueError(f"beam path {path.path_key} visits unknown ids {unknown}")
        return self


class OpticalLayoutPackageV1(OpticalLayoutContentV1):
    """The Optical Layout Package: its content and the osc1 digest of that content."""

    layout_content_hash: Sha256Digest = Field(description="osc1 digest of this document without layout_content_hash.")

    @model_validator(mode="after")
    def _hash(self) -> OpticalLayoutPackageV1:
        if self.layout_content_hash != osc1_digest(self.model_dump(mode="json", by_alias=True, exclude={"layout_content_hash"})):
            raise ValueError("layout_content_hash does not match the document")
        return self


def layout_content_hash(document: dict) -> str:
    """The osc1 digest of a layout document's content (its canonical form, without ``layout_content_hash``)."""

    content = OpticalLayoutContentV1.model_validate({k: v for k, v in document.items() if k != "layout_content_hash"})
    return osc1_digest(content.model_dump(mode="json", by_alias=True))


def seal_layout(document: dict) -> OpticalLayoutPackageV1:
    """The package of a layout content document, with its ``layout_content_hash``."""

    return OpticalLayoutPackageV1.model_validate({**document, "layout_content_hash": layout_content_hash(document)})


__all__ = [
    "FRAME",
    "OPTICAL_LAYOUT_FILE_NAME",
    "OPTICAL_LAYOUT_SCHEMA_ID",
    "OPTICAL_LAYOUT_SCHEMA_VERSION",
    "OpticalLayoutBeamPathV1",
    "OpticalLayoutBindingV1",
    "OpticalLayoutContentV1",
    "OpticalLayoutDofV1",
    "OpticalLayoutElementV1",
    "OpticalLayoutObservationV1",
    "OpticalLayoutOutlineV1",
    "OpticalLayoutPackageV1",
    "layout_content_hash",
    "seal_layout",
]
