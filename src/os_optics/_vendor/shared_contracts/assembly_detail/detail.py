"""Assembly Detail Package: the dimensioning supplement to exactly ONE verified Assembly Preview Package.

The Assembly Preview Package (``assembly_preview``) already carries everything geometric a Detailed Assembly Drawing
needs: product identities and BOM line refs, world poses, the support tree, mates and datums, settings (with their
basis), fixations, manual operations, the breadboard outline and hole grid, the beam reference and optical-axis height,
DOF delivery, assembly notes and part geometry. This document does NOT repeat any of it.

It adds the *dimension semantics*: which assembly-controlling distances the drawing prints, between which two points of
the resolved scene they are measured, and the basis of each one -- the optic's axis height and board position are
resolved placements, a post's height follows from its nominal insertion depth, the base's position along its slot is a
reference dimension computed from the resolved placements, and the stand-off to the customer's own light source comes
from the customer's design. So a ``DimensionV1`` is a *reference* (two scene points, one world axis, one basis): the
drawing measures the value from the preview's own poses, so there is one pose authority and a moved part moves its
dimension.

Everything else in the document is binding, release and usage:

* ``binding`` ties the supplement to exactly one preview package (its ``package_content_hash`` and revision) and,
  through the same values the preview manifest declares, to one Project, Local Validation, commercial resolution and
  Customer Purchase BOM. A supplement of preview package A verifies against package A only;
* ``release`` names the release under which the detail was issued (``ReleaseBasisV1``) and, for a confirmed Order,
  the customer's own references printed in the drawing's title block: the Order Number and the customer's display name
  (schema 3.1);
* ``line_display_names_zh_cn`` (schema 3.1) gives the Simplified Chinese name of each Customer Purchase BOM line of
  the bound scene (the parts legend of a zh-CN drawing; the preview scene carries the English names only);
* ``customer_supplied_models`` (schema 3.1) names the OpenSci catalog product the design used as the model of each
  customer-supplied catalog component (its English / Simplified Chinese product names), so the drawing can say "your
  own window (modelled as <name> <SKU>)" -- the customer's own part may differ from that model;
* ``usage`` states the terms of use (``DetailUsageTermsV1``).

Every value is an enum, a package-local reference that must resolve in the bound preview scene, a small integer, a
hash, or one of the customer-facing names above.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import Field, StringConstraints, field_validator, model_validator

from ..assembly_preview.manifest import LOCAL_VALIDATION_HASH_SUBJECT, CommercialBomRefV1, PermittedUseV1
from ..assembly_preview.scene import DatumKey, LineRef, PartRef, SettingKindV1
from ..base import StrictContractModel
from ..common.hashes import HashProfile, HashRef, Sha256Digest, osc1_digest, require_hash_profile
from ..common.identifiers import InstanceKey, OpenSciId
from ..optical_design.local_validation import ValidatedProjectRefV1


ASSEMBLY_DETAIL_SCHEMA_ID: Literal["opensci.assembly_detail.package"] = "opensci.assembly_detail.package"
#: 3.1: ``release.order_number`` / ``release.customer_display_name`` / ``release.delivery_mode``,
#: ``line_display_names_zh_cn`` and ``customer_supplied_models``.
ASSEMBLY_DETAIL_SCHEMA_VERSION: Literal["3.1"] = "3.1"
ASSEMBLY_DETAIL_FILE_NAME = "assembly-detail.json"
ASSEMBLY_DETAIL_MEDIA_TYPE: Literal["application/json"] = "application/json"
DETAIL_USAGE_STATEMENT: Literal[
    "Assembly detail data of this Project's resolved assembly, for generating its Detailed Assembly Drawing. Not for "
    "redistribution. The drawing files generated from it are yours to keep and use."
] = (
    "Assembly detail data of this Project's resolved assembly, for generating its Detailed Assembly Drawing. Not for "
    "redistribution. The drawing files generated from it are yours to keep and use."
)
#: Upper bound on dimension references (a multi-optic assembly carries about nine per support chain).
MAX_DIMENSIONS = 512
MAX_DETAIL_BYTES = 256 * 1024
MAX_BOARD_HOLE_INDEX = 4096

#: A customer-facing name (the customer's organization, an OpenSci product name in English or Simplified Chinese): any
#: printable text, no control character.
CustomerFacingName = Annotated[
    str,
    StringConstraints(strict=True, min_length=1, max_length=200, pattern=r"^[^\x00-\x1f\x7f]+$"),
    Field(description="Customer-facing name (no control character)."),
]
OpticalCatalogSku = Annotated[str, StringConstraints(strict=True, pattern=r"^OS-OPT(-[A-Z0-9]+)+$", max_length=120)]

DimensionRef = Annotated[
    str,
    StringConstraints(strict=True, min_length=1, max_length=32, pattern=r"^dim-[0-9]{1,3}$"),
    Field(description="Document-local dimension reference: 'dim-' and a number."),
]


# ------------------------------------------------------------------------------------------------ usage


DETAIL_PERMITTED_USES: tuple[PermittedUseV1, ...] = (PermittedUseV1.GENERATE_ASSEMBLY_DRAWING,)


class DetailUsageTermsV1(StrictContractModel):
    """Terms of use of an Assembly Detail Package: this Project's Detailed Assembly Drawing, no redistribution.

    The drawing files a runtime generates from it (SVG, PDF, result, manifest) are the customer's deliverable."""

    scope: Literal["THIS_PROJECT"] = "THIS_PROJECT"
    permitted_uses: tuple[PermittedUseV1, ...] = DETAIL_PERMITTED_USES
    redistribution: Literal["NOT_PERMITTED"] = "NOT_PERMITTED"
    generated_drawings: Literal["CUSTOMER_DELIVERABLE"] = "CUSTOMER_DELIVERABLE"
    statement: Literal[
        "Assembly detail data of this Project's resolved assembly, for generating its Detailed Assembly Drawing. Not for "
        "redistribution. The drawing files generated from it are yours to keep and use."
    ] = DETAIL_USAGE_STATEMENT

    @model_validator(mode="after")
    def _exact(self) -> DetailUsageTermsV1:
        if self.permitted_uses != DETAIL_PERMITTED_USES:
            raise ValueError("permitted_uses of an assembly detail package are exactly (GENERATE_ASSEMBLY_DRAWING,)")
        return self


# ---------------------------------------------------------------------------------------------- release


class ReleaseBasisV1(StrEnum):
    """Under which release the detail was issued.

    * ``CONFIRMED_ORDER`` -- the Order for this resolution and Project, whose corporate bank-transfer payment OpenSci
      confirmed as received. The document names only the basis; the Order itself is the customer's commercial record;
    * ``DIRECT_RELEASE`` -- released by OpenSci for this Project."""

    CONFIRMED_ORDER = "CONFIRMED_ORDER"
    DIRECT_RELEASE = "DIRECT_RELEASE"


class DetailReleaseRefV1(StrictContractModel):
    """The release under which the detail was issued.

    A ``CONFIRMED_ORDER`` release names the confirmed Order's ``order_number`` and the customer's
    ``customer_display_name`` (the organization display name of the Order commitment): the customer's own references,
    printed in the drawing's title block; and the Order's ``delivery_mode`` (``SELF_ASSEMBLY``: the customer assembles
    the system; ``OPENSCI_ASSEMBLY``: OpenSci assembles and aligns it before delivery, so the drawing's steps document how
    it is built). A ``DIRECT_RELEASE`` names none of them."""

    basis: ReleaseBasisV1
    order_number: OpenSciId | None = None
    customer_display_name: CustomerFacingName | None = None
    delivery_mode: Literal["SELF_ASSEMBLY", "OPENSCI_ASSEMBLY"] | None = None

    @model_validator(mode="after")
    def _order(self) -> DetailReleaseRefV1:
        if self.basis is not ReleaseBasisV1.CONFIRMED_ORDER and (self.order_number or self.customer_display_name
                                                                 or self.delivery_mode):
            raise ValueError("only a CONFIRMED_ORDER release names an Order Number, a customer and a delivery mode")
        if self.customer_display_name is not None and self.order_number is None:
            raise ValueError("a release names its customer together with the Order Number")
        return self


class CustomerSuppliedModelV1(StrictContractModel):
    """The OpenSci catalog product the design used as the model of one customer-supplied component (a design anchor of
    the bound scene whose ``customer_supply.opensci_sku`` is ``opensci_sku``). The customer supplies their own part; the
    drawing names this product only as the model the design was simulated and resolved with."""

    instance_id: InstanceKey
    opensci_sku: OpticalCatalogSku
    display_name: CustomerFacingName
    display_name_zh_cn: CustomerFacingName | None = None


# ---------------------------------------------------------------------------------------------- binding


class DetailBindingV1(StrictContractModel):
    """The one preview package (and through it the Project / Local Validation / resolution / BOM) this supplements.

    Every value except ``preview_package_*`` must equal what the bound preview manifest declares; the verifier
    compares them field by field."""

    project: ValidatedProjectRefV1
    local_validation_hash: HashRef | None = None
    commercial_resolution_id: OpenSciId
    commercial_bom: CommercialBomRefV1
    preview_package_content_hash: Sha256Digest
    preview_package_revision: int = Field(strict=True, ge=1)

    @field_validator("local_validation_hash")
    @classmethod
    def _lv(cls, value: HashRef | None) -> HashRef | None:
        if value is None:
            return None
        require_hash_profile(value, HashProfile.OSC1, field_name="local_validation_hash")
        if value.subject != LOCAL_VALIDATION_HASH_SUBJECT:
            raise ValueError(f"local_validation_hash subject must be {LOCAL_VALIDATION_HASH_SUBJECT!r}")
        return value


# -------------------------------------------------------------------------------------------- dimensions


class PointKindV1(StrEnum):
    """A point of the bound preview scene, in ``ASSEMBLY_WORLD`` millimetres.

    * ``PART_ORIGIN`` -- ``pose_in_world.position_mm`` of ``part_ref``;
    * ``PART_DATUM`` -- datum ``datum_key`` of ``part_ref`` placed by the part's pose;
    * ``BOARD_CORNER`` -- the breadboard outline's minimum corner on the board top (``part_ref`` = the breadboard);
    * ``BOARD_HOLE`` -- centre of grid hole ``hole = (i, j)`` on the board top: ``origin_xy + (i, j) * pitch``
      (``part_ref`` = the breadboard; zero-based indices, as ``FixationV1.board_holes``);
    * ``DESIGN_ANCHOR`` -- ``beam_reference.design_anchors[instance_id].position_mm``.
    """

    PART_ORIGIN = "PART_ORIGIN"
    PART_DATUM = "PART_DATUM"
    BOARD_CORNER = "BOARD_CORNER"
    BOARD_HOLE = "BOARD_HOLE"
    DESIGN_ANCHOR = "DESIGN_ANCHOR"


_POINT_FIELDS: dict[PointKindV1, frozenset[str]] = {
    PointKindV1.PART_ORIGIN: frozenset({"part_ref"}),
    PointKindV1.PART_DATUM: frozenset({"part_ref", "datum_key"}),
    PointKindV1.BOARD_CORNER: frozenset({"part_ref"}),
    PointKindV1.BOARD_HOLE: frozenset({"part_ref", "hole"}),
    PointKindV1.DESIGN_ANCHOR: frozenset({"instance_id"}),
}


class PointRefV1(StrictContractModel):
    kind: PointKindV1
    part_ref: PartRef | None = None
    datum_key: DatumKey | None = None
    hole: tuple[Annotated[int, Field(strict=True, ge=0, le=MAX_BOARD_HOLE_INDEX)],
                Annotated[int, Field(strict=True, ge=0, le=MAX_BOARD_HOLE_INDEX)]] | None = None
    instance_id: InstanceKey | None = None

    @model_validator(mode="after")
    def _fields(self) -> PointRefV1:
        present = {name for name in ("part_ref", "datum_key", "hole", "instance_id") if getattr(self, name) is not None}
        if present != _POINT_FIELDS[self.kind]:
            raise ValueError(f"a {self.kind.value} point takes exactly {sorted(_POINT_FIELDS[self.kind])}")
        return self


class MeasureAxisV1(StrEnum):
    """The world axis a dimension is measured along: value = coordinate(to_point) - coordinate(from_point)."""

    WORLD_X = "WORLD_X"
    WORLD_Y = "WORLD_Y"
    WORLD_Z = "WORLD_Z"


AXIS_INDEX: dict[MeasureAxisV1, int] = {MeasureAxisV1.WORLD_X: 0, MeasureAxisV1.WORLD_Y: 1, MeasureAxisV1.WORLD_Z: 2}


class DimensionKindV1(StrEnum):
    """What an assembly dimension controls. The verifier enforces each kind's shape against the bound scene."""

    OPTICAL_AXIS_HEIGHT = "OPTICAL_AXIS_HEIGHT"          # board top -> optic origin, along +Z
    INSERTION_DEPTH = "INSERTION_DEPTH"                  # receiving datum -> inserted part's bottom datum; shows a setting
    OPTIC_BOARD_POSITION = "OPTIC_BOARD_POSITION"        # board corner -> optic origin or optical centre, along X or Y
    FIXATION_HOLE_POSITION = "FIXATION_HOLE_POSITION"    # board corner -> the fixation hole, along X or Y
    FIXATION_OFFSET = "FIXATION_OFFSET"                  # fixed part origin -> its fixation hole, along X or Y
    EXTERNAL_SOURCE_STANDOFF = "EXTERNAL_SOURCE_STANDOFF"  # customer-supplied source anchor -> optic origin or optical centre


class DimensionBasisV1(StrEnum):
    """Basis of a printed dimension.

    * ``RESOLVED`` -- fixed by the resolved placement (optic position, selected hole, resolved setting);
    * ``NOMINAL`` -- the value is, or follows from, a nominal set-up value of a part (for example a post's nominal
      insertion depth), set at assembly;
    * ``REFERENCE`` -- a reference dimension computed from the resolved placements (for example a base's position
      along its slot), printed in parentheses;
    * ``CUSTOMER_DESIGN`` -- taken from the customer's own Project (for example where their own light source sits
      relative to the optic).
    """

    RESOLVED = "RESOLVED"
    NOMINAL = "NOMINAL"
    REFERENCE = "REFERENCE"
    CUSTOMER_DESIGN = "CUSTOMER_DESIGN"


class SettingRefV1(StrictContractModel):
    """A setting of the bound scene whose value this dimension shows (the measured value must equal it)."""

    part_ref: PartRef
    kind: SettingKindV1


class DimensionV1(StrictContractModel):
    dim_ref: DimensionRef
    kind: DimensionKindV1
    subject_part_ref: PartRef = Field(description="The part this dimension belongs to: its balloon, and the part whose pose moves it.")
    from_point: PointRefV1
    to_point: PointRefV1
    axis: MeasureAxisV1
    basis: DimensionBasisV1
    setting: SettingRefV1 | None = None


# -------------------------------------------------------------------------------------------- document


class AssemblyDetailPackageV1(StrictContractModel):
    schema_id: Literal["opensci.assembly_detail.package"] = Field(ASSEMBLY_DETAIL_SCHEMA_ID, alias="schema")
    schema_version: Literal["3.1"] = ASSEMBLY_DETAIL_SCHEMA_VERSION
    usage: DetailUsageTermsV1 = DetailUsageTermsV1()
    binding: DetailBindingV1
    release: DetailReleaseRefV1
    dimensions: tuple[DimensionV1, ...] = Field(min_length=1, max_length=MAX_DIMENSIONS)
    customer_supplied_models: tuple[CustomerSuppliedModelV1, ...] = Field(
        (), max_length=MAX_DIMENSIONS, description="The catalog model of each customer-supplied catalog component, by instance id.")
    line_display_names_zh_cn: dict[LineRef, CustomerFacingName] = Field(
        default_factory=dict, max_length=MAX_DIMENSIONS,
        description="The Simplified Chinese name of Customer Purchase BOM lines of the bound scene, by line key.")
    detail_content_hash: Sha256Digest = Field(description="osc1 digest of this document without detail_content_hash.")

    @model_validator(mode="after")
    def _check(self) -> AssemblyDetailPackageV1:
        refs = [d.dim_ref for d in self.dimensions]
        if len(set(refs)) != len(refs):
            raise ValueError("dimension refs must be unique")
        ids = [m.instance_id for m in self.customer_supplied_models]
        if ids != sorted(set(ids)):
            raise ValueError("customer_supplied_models are unique and ordered by instance_id")
        if self.detail_content_hash != detail_content_hash_of(self.content_payload()):
            raise ValueError("detail_content_hash does not match the document")
        return self

    def payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", by_alias=True, exclude_none=True)

    def content_payload(self) -> dict[str, Any]:
        payload = self.payload()
        del payload["detail_content_hash"]
        return payload

    def dimension(self, dim_ref: str) -> DimensionV1:
        for dim in self.dimensions:
            if dim.dim_ref == dim_ref:
                return dim
        raise KeyError(dim_ref)


def detail_content_hash_of(content_payload: dict[str, Any]) -> str:
    """``detail_content_hash`` = osc1 digest of the document payload without that field."""

    return osc1_digest(content_payload)
