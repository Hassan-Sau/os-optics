"""Assembly Preview scene: the resolved assembly of ONE project/resolution, for the Experimental Lab Assembly View.

After Assembly Resolution the server sends the user's device the Assembly Preview Package so the device can render the
Experimental Lab Assembly View and check the assembly locally. This module is the scene part of that package
(``scene.json``); the manifest, file layout and integrity checks live in ``manifest.py`` / ``package.py``.

What the scene contains:

* the resolved parts with their OpenSci product identities, world poses, the support tree, the mates used, settings,
  fixations, manual operations, the breadboard, a beam reference bound to the user's own local optical run, and how
  each retained design DOF is delivered;
* the layout checks the resolution passed (``ResolutionSummaryV1``) and assembly notes with fixed wording chosen by
  code (``NOTE_TEMPLATES``);
* the part geometry of every part (``geometry/<part_ref>.glb``), described by ``GeometryFidelityV1``;
* for every part, the Customer Purchase BOM line it belongs to (``commercial_line_ref``): quantities and prices are
  those of that BOM;
* when the customer uses their own optical table instead of the OpenSci breadboard (``breadboard.supplied_by =
  CUSTOMER``), the scene root is that table's mounting area on the resolved hole grid: a ``BREADBOARD`` part with the
  ``CUSTOMER_SUPPLIED`` identity ``OS-CUSTOMER-OPTICAL-TABLE`` and no BOM line. A renderer can show it as a mounting area
  rather than a product;
* every Project component and observation lands at a design anchor (``beam_reference.design_anchors``). A
  customer-supplied component -- the customer's light source, or an optical component OpenSci does not supply (not
  orderable, not in the catalog, temporarily unavailable, or already owned by the customer) -- is an unrealized anchor
  with ``customer_supply`` (its role, the reason, its OpenSci product if it is one, its orientation and,
  when its geometry is known, the space the resolution reserved for it) and a fixed note (``EXTERNAL_LIGHT_SOURCE`` /
  ``CUSTOMER_SUPPLIED_COMPONENT``). It has no part and no BOM line: the customer places it. An optic OpenSci supplies
  without mounting hardware (reason ``OPENSCI_SUPPLIED_UNMOUNTED``: OpenSci has no mount for it) is such an anchor too:
  it is a line of the purchase but has no part in the scene, and the customer mounts it at the anchor.

Conventions a renderer needs:

* frame ``ASSEMBLY_WORLD``, millimetres, +Z up; every pose is ``x_world = R(q) x_local + position_mm``
  with ``quaternion_wxyz`` a unit Hamilton quaternion (w first) of an ACTIVE rotation (``PoseV1``);
* part geometry (``geometry/<part_ref>.glb``) is in the part's own frame, in METRES, +Z up, with no node
  transform: scale by 1000 and apply ``pose_in_world``. A stock glTF viewer (+Y up) shows it lying on its
  side unless the whole assembly gets one root rotation of -90 deg about X;
* the scene graph is ``PreviewPartV1.support.parent_ref`` (rooted at the breadboard). A mate joins two
  datums (``b_in_a`` = datum B in datum A); use ``support`` to know which side carries the other;
* the Project design frame sits in the world at ``beam_reference.design_frame_in_world``: transform the
  client's own local optical run (Optical Simulation View, ``PROJECT_DESIGN_FRAME``, mm) by it to overlay
  the beam;
* a DOF arrow starts at the optic part's origin along ``world_direction``; the part that moves is
  ``moved_part_ref`` together with everything it carries;
* ``bounds`` covers the placed geometry, the design anchors and the beam axes, so a camera framed on it
  shows the whole beam;
* a beam axis's ``segment_roles`` say how prominently to draw each stretch, as the Optical Simulation View does: a
  ``SECONDARY`` stretch carries only weak light (a leak through a coating, the weak arm of a polarizing splitter) and is
  drawn thinner and fainter; its ``notes`` say when another input polarization would route it differently, or where
  its share is not a polarization-resolved value (``BeamAxisNoteV1``).

Versions: ``schema_version`` ``"3.0"`` or ``"3.1"``. Scene 3.1 adds a beam axis's ``segment_roles`` and ``notes``; a
scene uses 3.1 exactly when one of its beam axes carries them (``required_scene_schema_version``), so every other scene
stays a 3.0 document, byte for byte, that readers of 3.0 (opensci-engine before 0.8.0) open. A 3.0 document with a 3.1
field is refused, naming the fields. A reader of 3.1 reads both.
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import Field, StringConstraints, field_validator, model_validator

from ..base import StrictContractModel
from ..common.geometry import PoseV1, Vector3
from ..common.hashes import HashProfile, HashRef, osc1_digest, require_engine_quantized_params, require_hash_profile
from ..common.identifiers import InstanceKey
from ..optical_design.dof import DofAxisV1, DofControlIntentV1
from ..optical_design.local_validation import ENGINE_INPUT_HASH_SUBJECT, ENGINE_RESULT_HASH_SUBJECT


ASSEMBLY_PREVIEW_SCENE_SCHEMA_ID: Literal["opensci.assembly_preview.scene"] = "opensci.assembly_preview.scene"
#: The scene version of a scene without 3.1 fields (module docstring); documents of a version not in
#: ``ASSEMBLY_PREVIEW_SCENE_SCHEMA_VERSIONS`` are refused.
ASSEMBLY_PREVIEW_SCENE_SCHEMA_VERSION: Literal["3.0"] = "3.0"
#: The scene version that adds a beam axis's ``segment_roles`` and ``notes``.
ASSEMBLY_PREVIEW_SCENE_SCHEMA_VERSION_3_1: Literal["3.1"] = "3.1"
ASSEMBLY_PREVIEW_SCENE_SCHEMA_VERSIONS: tuple[str, ...] = (ASSEMBLY_PREVIEW_SCENE_SCHEMA_VERSION, ASSEMBLY_PREVIEW_SCENE_SCHEMA_VERSION_3_1)
ASSEMBLY_PREVIEW_VIEW_KIND: Literal["EXPERIMENTAL_LAB_ASSEMBLY"] = "EXPERIMENTAL_LAB_ASSEMBLY"
ASSEMBLY_PREVIEW_DESCRIPTION: Literal[
    "Experimental Lab Assembly of one resolved OpenSci project: every supplied part with its OpenSci product "
    "identity, world pose, support and mates on the resolved breadboard. Each part names its Customer Purchase BOM "
    "line in commercial_line_ref; quantities and prices are those of that BOM."
] = (
    "Experimental Lab Assembly of one resolved OpenSci project: every supplied part with its OpenSci product "
    "identity, world pose, support and mates on the resolved breadboard. Each part names its Customer Purchase BOM "
    "line in commercial_line_ref; quantities and prices are those of that BOM."
)

GEOMETRY_DIR = "geometry/"
GEOMETRY_SUFFIX = ".glb"
MAX_PARTS = 256
MAX_BEAM_AXIS_POINTS = 64
#: The datum of an optical element at its optical centre, the point its spacing along the beam and its position on the
#: board refer to: a transmissive optic's body centre on its axis (the middle of its thickness); a reflector, a plate
#: splitter or an element placed by its ports at its frame origin (the coated face centre, the body or aperture centre).
OPTICAL_CENTRE_DATUM = "optical_centre"
#: The design roles of the optics the beam passes through: their optical centre is the body centre on the axis.
TRANSMISSIVE_ROLES = frozenset({"lens", "window", "filter", "waveplate", "polarizer"})
UNIT_TOLERANCE = 1e-9

PartRef = Annotated[
    str,
    StringConstraints(strict=True, min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9]*(-[a-z0-9]+)*$"),
    Field(description="Package-local part reference (lowercase words joined by '-'); also the geometry file stem."),
]
ObjectRef = Annotated[
    str,
    StringConstraints(strict=True, min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9]*(-[a-z0-9]+)*$"),
    Field(description="Package-local reference of a mate or manual operation."),
]
DatumKey = Annotated[
    str,
    StringConstraints(strict=True, min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$"),
    Field(description="Datum name inside one part (for example 'seat', 'top', 'board_top')."),
]
OpenSciSku = Annotated[
    str,
    StringConstraints(strict=True, min_length=4, max_length=120, pattern=r"^OS(-[A-Z0-9]+)+$"),
    Field(description="OpenSci product number: 'OS' followed by uppercase letter and digit segments."),
]
DisplayText = Annotated[
    str,
    StringConstraints(strict=True, min_length=1, max_length=160, pattern=r"^[A-Za-z0-9 ,.()/+=\-x]+$"),
    Field(description="Customer-facing display name."),
]
InterfaceDesignation = Annotated[
    str,
    StringConstraints(strict=True, min_length=1, max_length=40, pattern=r"^[A-Za-z0-9.\-x]+$"),
    Field(description="Standard interface designation used by a mate (for example 'SM1', 'M4x0.7')."),
]
LineRef = Annotated[
    str,
    StringConstraints(strict=True, min_length=1, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$"),
    Field(description="Key of the authoritative Customer Purchase BOM line this part belongs to."),
]
SettingValue = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=40, pattern=r"^[A-Za-z0-9.\-x]+$")]
Rgb = tuple[float, float, float]


def _duplicates(values: list[Any]) -> list[Any]:
    seen: set[Any] = set()
    return sorted({v for v in values if v in seen or seen.add(v)}, key=str)  # type: ignore[func-returns-value]


def _finite(values: tuple[float, ...], where: str) -> None:
    if not all(math.isfinite(v) for v in values):
        raise ValueError(f"{where}: values must be finite")


# --------------------------------------------------------------------------------------------------- parts


class PartCategoryV1(StrEnum):
    """What a resolved part does in the assembly."""

    OPTICAL_ELEMENT = "OPTICAL_ELEMENT"
    OPTIC_MOUNT = "OPTIC_MOUNT"
    ADJUSTMENT_STAGE = "ADJUSTMENT_STAGE"
    POST = "POST"
    POST_HOLDER = "POST_HOLDER"
    BASE = "BASE"
    ADAPTER = "ADAPTER"
    BREADBOARD = "BREADBOARD"
    OTHER_SUPPORT = "OTHER_SUPPORT"


class IdentityKindV1(StrEnum):
    """Which identity namespace ``opensci_sku`` comes from."""

    PUBLIC_OPTICAL_SKU = "PUBLIC_OPTICAL_SKU"  # 'OS-OPT-*' optical product identity
    COMMERCIAL_SKU = "COMMERCIAL_SKU"  # OpenSci mechanical and other product identity
    CUSTOMER_SUPPLIED = "CUSTOMER_SUPPLIED"  # not an OpenSci product: the customer's own optical table


#: The identity of the customer's own optical table when it stands in for the OpenSci breadboard.
CUSTOMER_OPTICAL_TABLE_SKU = "OS-CUSTOMER-OPTICAL-TABLE"


class PartIdentityV1(StrictContractModel):
    """The OpenSci identity of a resolved part; the same identity the commercial BOM line shows."""

    opensci_sku: OpenSciSku
    identity_kind: IdentityKindV1
    display_name: DisplayText

    @model_validator(mode="after")
    def _namespace(self) -> PartIdentityV1:
        sku, kind = self.opensci_sku, self.identity_kind
        if kind is IdentityKindV1.PUBLIC_OPTICAL_SKU and not sku.startswith("OS-OPT-"):
            raise ValueError("PUBLIC_OPTICAL_SKU identities start with 'OS-OPT-'")
        if (kind is IdentityKindV1.CUSTOMER_SUPPLIED) != (sku == CUSTOMER_OPTICAL_TABLE_SKU):
            raise ValueError(f"the CUSTOMER_SUPPLIED identity is exactly {CUSTOMER_OPTICAL_TABLE_SKU!r}")
        return self


class DesignBindingV1(StrictContractModel):
    """Binds an optical part to the Project component it realizes (the user's own instance id)."""

    instance_id: InstanceKey
    optical_sku: Annotated[str, StringConstraints(strict=True, pattern=r"^OS-OPT(-[A-Z0-9]+)+$")]
    catalog_version: Annotated[str, StringConstraints(strict=True, pattern=r"^[0-9]{4}-[0-9]{2}-[0-9]{2}\.[1-9][0-9]*$")]


class DatumFrameV1(StrictContractModel):
    datum_key: DatumKey
    pose_in_part: PoseV1


class SupportRelationV1(StrEnum):
    SEATED_IN = "SEATED_IN"  # optic seated in its mount
    MOUNTED_ON = "MOUNTED_ON"  # mount on a post / stage / adapter
    INSERTED_INTO = "INSERTED_INTO"  # post inserted into a holder bore
    STANDS_ON = "STANDS_ON"  # holder / base standing on the breadboard
    FASTENED_TO = "FASTENED_TO"


class SupportV1(StrictContractModel):
    """Scene-graph edge: this part is carried by ``parent_ref`` through mate ``mate_ref``."""

    parent_ref: PartRef
    mate_ref: ObjectRef
    relation: SupportRelationV1


class GeometryFidelityV1(StrEnum):
    """How a part's geometry is built."""

    NOMINAL_SHAPE = "NOMINAL_SHAPE"  # boxes / cylinders from the part's nominal dimensions
    SIMPLIFIED_MESH = "SIMPLIFIED_MESH"  # simplified external shape


class FinishV1(StrEnum):
    BODY = "BODY"
    DARK = "DARK"
    GLASS = "GLASS"
    METAL = "METAL"


class PartGeometryV1(StrictContractModel):
    """Visual geometry of one part: a glTF 2.0 binary in the package, authored in the part's own frame.

    The mesh carries no transform: a client places it with ``PreviewPartV1.pose_in_world``. Coordinates are
    metres (glTF units) on the part's own axes, +Z up like the scene (no Y-up conversion is baked in).
    """

    asset_path: Annotated[str, StringConstraints(strict=True, pattern=r"^geometry/[a-z][a-z0-9]*(-[a-z0-9]+)*\.glb$")]
    format: Literal["GLB_2_0"] = "GLB_2_0"
    length_unit: Literal["m"] = "m"
    frame: Literal["PART_LOCAL"] = "PART_LOCAL"
    up_axis: Literal["+Z"] = "+Z"
    fidelity: GeometryFidelityV1
    bounds_min_mm: Vector3
    bounds_max_mm: Vector3
    triangle_count: int = Field(ge=1, le=200_000)

    @model_validator(mode="after")
    def _bounds(self) -> PartGeometryV1:
        _finite(self.bounds_min_mm + self.bounds_max_mm, "geometry bounds")
        if any(lo > hi for lo, hi in zip(self.bounds_min_mm, self.bounds_max_mm)):
            raise ValueError("geometry bounds_min_mm must not exceed bounds_max_mm")
        return self


class AppearanceV1(StrictContractModel):
    base_color_rgb: Rgb
    finish: FinishV1

    @field_validator("base_color_rgb")
    @classmethod
    def _unit_rgb(cls, value: Rgb) -> Rgb:
        if not all(0.0 <= c <= 1.0 for c in value):
            raise ValueError("base_color_rgb components must lie in [0, 1]")
        return value


class OpticFaceShapeV1(StrEnum):
    """The shape of an optic's outer face as seen from outside the optic."""

    PLANO = "PLANO"
    CONVEX = "CONVEX"
    CONCAVE = "CONCAVE"


class OpticFaceV1(StrictContractModel):
    """One outer face of a transmissive optic: its shape and, for a curved face, its vertex radius (magnitude)."""

    shape: OpticFaceShapeV1
    radius_mm: float | None = Field(None, gt=0, description="The vertex radius of a curved face (absent for a plano face).")

    @model_validator(mode="after")
    def _check(self) -> OpticFaceV1:
        if (self.shape is OpticFaceShapeV1.PLANO) != (self.radius_mm is None):
            raise ValueError("a curved face has a radius_mm and a plano face none")
        if self.radius_mm is not None and not math.isfinite(self.radius_mm):
            raise ValueError("radius_mm must be finite")
        return self


class OpticFacesV1(StrictContractModel):
    """The two outer faces of a transmissive optic whose faces differ (a plano-convex lens, an achromat), in the optic's
    own frame (local +Z is its axis): ``front`` is the face at its origin, the one light travelling along local +Z meets
    first; ``back`` is its last face, the one light travelling against local +Z meets first. Which face the beam meets
    first follows from the part's pose and the beam's direction; an optic with alike faces carries none."""

    front: OpticFaceV1
    back: OpticFaceV1

    @model_validator(mode="after")
    def _check(self) -> OpticFacesV1:
        if self.front == self.back:
            raise ValueError("faces are given only for an optic whose two faces differ")
        return self


class PreviewPartV1(StrictContractModel):
    """One resolved physical part of the assembly, quantity one."""

    part_ref: PartRef
    category: PartCategoryV1
    identity: PartIdentityV1
    design_binding: DesignBindingV1 | None = None
    commercial_line_ref: LineRef | None = None
    pose_in_world: PoseV1
    support: SupportV1 | None = None
    assembly_order: int = Field(ge=1, le=MAX_PARTS)
    datums: tuple[DatumFrameV1, ...] = ()
    geometry: PartGeometryV1
    appearance: AppearanceV1
    actuated: bool = Field(False, description="True only for a part that physically carries an actuator.")
    faces: OpticFacesV1 | None = Field(
        None, description="An OPTICAL_ELEMENT with dissimilar faces: its two outer faces (absent: alike faces, or not stated).")

    @model_validator(mode="after")
    def _check(self) -> PreviewPartV1:
        if self.geometry.asset_path != f"{GEOMETRY_DIR}{self.part_ref}{GEOMETRY_SUFFIX}":
            raise ValueError(f"part {self.part_ref!r}: geometry.asset_path must be 'geometry/{self.part_ref}.glb'")
        if dup := _duplicates([d.datum_key for d in self.datums]):
            raise ValueError(f"part {self.part_ref!r}: duplicate datum(s) {dup}")
        if (self.category is PartCategoryV1.OPTICAL_ELEMENT) != (self.design_binding is not None):
            raise ValueError(f"part {self.part_ref!r}: design_binding is required for, and only for, OPTICAL_ELEMENT parts")
        if (self.category is PartCategoryV1.BREADBOARD) != (self.support is None):
            raise ValueError(f"part {self.part_ref!r}: only the BREADBOARD is the unsupported scene root")
        if self.faces is not None and self.category is not PartCategoryV1.OPTICAL_ELEMENT:
            raise ValueError(f"part {self.part_ref!r}: only an OPTICAL_ELEMENT has optic faces")
        return self


class MateFastenerKindV1(StrEnum):
    #: A cap screw passes part A's base hole into part B's top thread; a stud standing in that thread is taken out
    #: first. The screw is supplied with part A (no separate item line).
    CAP_SCREW_THROUGH_BASE = "CAP_SCREW_THROUGH_BASE"
    #: Part A has base holes on more than one face; only the face with the ``thread_designation`` tapped hole takes
    #: part B's stud: part A stands on that face.
    STUD_INTO_TAPPED_FACE = "STUD_INTO_TAPPED_FACE"


class MateFastenerV1(StrictContractModel):
    """How a mate is fastened when it is not part B's stud entering part A's only tapped hole."""

    kind: MateFastenerKindV1
    thread_designation: InterfaceDesignation


class MateV1(StrictContractModel):
    """A realized datum-to-datum mate: ``b_in_a`` is datum B expressed in datum A. ``fastener``: how it is fastened
    when that is not part B's stud entering part A's tapped hole (absent: that plain stud joint)."""

    mate_ref: ObjectRef
    part_a_ref: PartRef
    datum_a: DatumKey
    part_b_ref: PartRef
    datum_b: DatumKey
    b_in_a: PoseV1
    interface_designation: InterfaceDesignation | None = None
    fastener: MateFastenerV1 | None = None


# ------------------------------------------------------------------------------------ settings / fixations


class SettingKindV1(StrEnum):
    INSERTION_DEPTH = "INSERTION_DEPTH"
    BASE_YAW = "BASE_YAW"
    MOUNT_ORIENTATION_YAW = "MOUNT_ORIENTATION_YAW"
    FIXATION_THREAD = "FIXATION_THREAD"


class SettingBasisV1(StrEnum):
    """Where a setting value comes from."""

    RESOLVED = "RESOLVED"  # fixed by the resolved placement
    NOMINAL = "NOMINAL"  # the nominal set-up value of the part, set at assembly


class SettingV1(StrictContractModel):
    part_ref: PartRef
    kind: SettingKindV1
    value: SettingValue
    unit: Literal["mm", "deg", "NONE"]
    basis: SettingBasisV1


class FixationModeV1(StrEnum):
    FIXED_HOLES = "FIXED_HOLES"
    SLOT = "SLOT"
    MANUAL_BASE_FIXATION = "MANUAL_BASE_FIXATION"


class FastenerNoteV1(StrictContractModel):
    thread_designation: InterfaceDesignation | None = None
    on_commercial_bom: bool


class FixationV1(StrictContractModel):
    part_ref: PartRef
    mode: FixationModeV1
    board_holes: tuple[tuple[int, int], ...] = ()
    fastener: FastenerNoteV1 | None = None
    manual_operation_ref: ObjectRef | None = None

    @model_validator(mode="after")
    def _check(self) -> FixationV1:
        if (self.mode is FixationModeV1.MANUAL_BASE_FIXATION) != (self.manual_operation_ref is not None):
            raise ValueError(f"fixation of {self.part_ref!r}: manual_operation_ref is required for, and only for, MANUAL_BASE_FIXATION")
        return self


class ManualOperationKindV1(StrEnum):
    MANUAL_BASE_FIXATION = "MANUAL_BASE_FIXATION"
    MANUAL_ALIGNMENT = "MANUAL_ALIGNMENT"
    MANUAL_ADJUSTMENT = "MANUAL_ADJUSTMENT"


class ManualOperationV1(StrictContractModel):
    operation_ref: ObjectRef
    part_ref: PartRef
    kind: ManualOperationKindV1


# ------------------------------------------------------------------------------------------- breadboard


class BoardHoleGridV1(StrictContractModel):
    origin_xy_mm: tuple[float, float] = Field(description="World XY of hole (0, 0).")
    pitch_mm: tuple[float, float]
    count: tuple[int, int]
    thread_designation: InterfaceDesignation

    @model_validator(mode="after")
    def _check(self) -> BoardHoleGridV1:
        _finite(self.origin_xy_mm + self.pitch_mm, "hole grid")
        if min(self.pitch_mm) <= 0 or min(self.count) < 1:
            raise ValueError("hole grid pitch must be positive and count at least 1")
        return self


class BreadboardV1(StrictContractModel):
    """The resolved breadboard (post-resolution; the optical Project never selects it). ``supplied_by = CUSTOMER``:
    the customer mounts the layout on their own optical table; this is then the mounting area the layout needs on
    that table, on the same hole grid, and its part is not on the Customer Purchase BOM (absent: the OpenSci
    breadboard)."""

    part_ref: PartRef
    top_z_mm: float
    thickness_mm: float = Field(gt=0)
    outline_min_xy_mm: tuple[float, float]
    outline_max_xy_mm: tuple[float, float]
    hole_grid: BoardHoleGridV1 | None = None
    supplied_by: Literal["CUSTOMER"] | None = None

    @model_validator(mode="after")
    def _check(self) -> BreadboardV1:
        _finite((self.top_z_mm, *self.outline_min_xy_mm, *self.outline_max_xy_mm), "breadboard")
        if any(lo >= hi for lo, hi in zip(self.outline_min_xy_mm, self.outline_max_xy_mm)):
            raise ValueError("breadboard outline min must be below max")
        return self


# ---------------------------------------------------------------------------------------- beam reference


class BeamAxisBasisV1(StrEnum):
    DESIGN_NOMINAL_AXIS = "DESIGN_NOMINAL_AXIS"  # straight lines between design element positions
    ENGINE_CHIEF_RAY = "ENGINE_CHIEF_RAY"  # chief ray of an Engine run of the same Project


class BeamAxisRoleV1(StrEnum):
    """How prominently to draw a stretch of a beam axis (the Optical Simulation View's display roles): ``PRIMARY`` a
    main beam; ``SECONDARY`` a stretch only weak light reaches, drawn thinner and fainter, never over a primary beam."""

    PRIMARY = "PRIMARY"
    SECONDARY = "SECONDARY"


class BeamAxisNoteV1(StrEnum):
    """What a reader should know about how a beam axis was routed.

    * ``POLARIZATION_DEPENDENT`` -- the line reaches an element whose share depends on polarization (a polarizing
      splitter, a polarizer) with a polarization the design does not determine there; it is drawn as unpolarized light
      would go, and another input polarization would route it differently.
    * ``POLARIZATION_NOT_RESOLVED`` -- at an element on this axis the line's share is not a polarization-resolved value
      within the element's characterized data: the routing could not follow the line through the element (for example
      light that meets one of its surfaces a second time) and used the element's unpolarized published share, or it met
      the element outside its characterized angles of incidence, wavelengths or polarizations (where the run reports the
      element out of model scope). The drawing there may differ from the polarization-resolved run."""

    POLARIZATION_DEPENDENT = "POLARIZATION_DEPENDENT"
    POLARIZATION_NOT_RESOLVED = "POLARIZATION_NOT_RESOLVED"


class BeamAxisV1(StrictContractModel):
    """One nominal beam line: its points in path order and, in ``through``, the design element at each point -- the
    ``instance_id`` of a design anchor (a component or an observation plane). An element entered and left at two
    points (an offset return) is named at both. A reader names a point by ``through``, never by matching coordinates:
    two elements may share a position (two light sources at one point), and an element on a laterally shifted axis lies
    off the nominal point.

    ``segment_roles`` gives the display role of each stretch (points k -> k+1), with the meaning of the Optical
    Simulation View's roles: ``PRIMARY`` a main beam, ``SECONDARY`` a stretch only weak light reaches (drawn thinner and
    fainter, never over a primary beam); absent: every stretch is primary. ``notes``: ``BeamAxisNoteV1``."""

    path_key: ObjectRef
    basis: BeamAxisBasisV1
    points_mm: tuple[Vector3, ...] = Field(min_length=2, max_length=MAX_BEAM_AXIS_POINTS)
    wavelength_nm: float | None = Field(None, gt=0)
    through: tuple[InstanceKey, ...] | None = Field(
        None, description="The design anchor (instance_id) at each point of points_mm, in path order (absent: not named).")
    segment_roles: tuple[BeamAxisRoleV1, ...] | None = Field(
        None, description="The display role of each stretch (points k -> k+1), as in the Optical Simulation View; absent: "
        "every stretch is PRIMARY.")
    notes: tuple[BeamAxisNoteV1, ...] | None = Field(None, description="How the axis was routed (BeamAxisNoteV1); absent: nothing to note.")

    @field_validator("points_mm")
    @classmethod
    def _finite_points(cls, points: tuple[Vector3, ...]) -> tuple[Vector3, ...]:
        for p in points:
            _finite(p, "beam axis point")
        return points

    @model_validator(mode="after")
    def _named_points(self) -> BeamAxisV1:
        if self.through is not None and len(self.through) != len(self.points_mm):
            raise ValueError(f"beam axis {self.path_key!r}: through names {len(self.through)} elements for {len(self.points_mm)} points")
        if self.segment_roles is not None and len(self.segment_roles) != len(self.points_mm) - 1:
            raise ValueError(f"beam axis {self.path_key!r}: segment_roles gives {len(self.segment_roles)} roles for "
                             f"{len(self.points_mm) - 1} stretches")
        if self.notes is not None and (not self.notes or len(set(self.notes)) != len(self.notes)):
            raise ValueError(f"beam axis {self.path_key!r}: notes must be distinct and not empty (omit them instead)")
        return self


def scene_schema_3_1_features(beam_axes: tuple[BeamAxisV1, ...]) -> tuple[str, ...]:
    """The scene 3.1 fields ``beam_axes`` use, as ``"<path_key>.segment_roles"`` / ``"<path_key>.notes"``."""

    features = []
    for axis in beam_axes:
        if axis.segment_roles is not None:
            features.append(f"{axis.path_key}.segment_roles")
        if axis.notes is not None:
            features.append(f"{axis.path_key}.notes")
    return tuple(features)


def required_scene_schema_version(beam_axes: tuple[BeamAxisV1, ...]) -> Literal["3.0", "3.1"]:
    """The scene version a writer states for ``beam_axes``: ``"3.1"`` when an axis carries 3.1 fields, else ``"3.0"``."""

    return ASSEMBLY_PREVIEW_SCENE_SCHEMA_VERSION_3_1 if scene_schema_3_1_features(beam_axes) else ASSEMBLY_PREVIEW_SCENE_SCHEMA_VERSION


class DesignAnchorKindV1(StrEnum):
    COMPONENT = "COMPONENT"
    OBSERVATION = "OBSERVATION"


class CustomerSupplyReasonV1(StrEnum):
    """Why a design component is supplied by the customer rather than by OpenSci.

    * ``CUSTOMER_LIGHT_SOURCE`` -- the customer's own light source (a user-defined source component);
    * ``NOT_IN_CATALOG`` -- a user-defined component that is not an OpenSci catalog product;
    * ``NOT_ORDERABLE`` -- an OpenSci catalog product available for design and simulation only (or not available in a
      commercial assembly);
    * ``TEMPORARILY_UNAVAILABLE`` -- an orderable OpenSci product that OpenSci cannot supply at the moment;
    * ``CUSTOMER_OWNED`` -- an orderable OpenSci product the customer chose to supply themselves (already owned);
    * ``OPENSCI_SUPPLIED_UNMOUNTED`` -- an OpenSci product OpenSci supplies without mounting hardware (it has no mount for
      it): a line of the purchase, mounted by the customer at its design position."""

    CUSTOMER_LIGHT_SOURCE = "CUSTOMER_LIGHT_SOURCE"
    NOT_IN_CATALOG = "NOT_IN_CATALOG"
    NOT_ORDERABLE = "NOT_ORDERABLE"
    TEMPORARILY_UNAVAILABLE = "TEMPORARILY_UNAVAILABLE"
    CUSTOMER_OWNED = "CUSTOMER_OWNED"
    OPENSCI_SUPPLIED_UNMOUNTED = "OPENSCI_SUPPLIED_UNMOUNTED"


#: The reasons of an anchor that is an OpenSci catalog product (it names its ``opensci_sku``).
CATALOG_SUPPLY_REASONS = frozenset({CustomerSupplyReasonV1.NOT_ORDERABLE, CustomerSupplyReasonV1.TEMPORARILY_UNAVAILABLE,
                                    CustomerSupplyReasonV1.CUSTOMER_OWNED, CustomerSupplyReasonV1.OPENSCI_SUPPLIED_UNMOUNTED})


#: The optical role of a design component (the Project's ``OpticalRoleV1`` spelling).
DesignRole = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^(source|lens|mirror|splitter|filter|window|polarizer|waveplate|aperture|prism|"
                                          r"detector|dump|other_optical)$"),
    Field(description="The optical role of the design component (the Project's optical_role)."),
]


#: Where a reserved envelope's geometry comes from: the public catalog's design geometry of the component's OpenSci
#: product, or the optical model the Project declares for a component of its own (``ProjectOpticalModelV1``: its
#: ``design_geometry``, a COMPOUND unit's ``envelope``, else the outline and surfaces of the model).
ReservedEnvelopeBasis = Literal["CATALOG_DESIGN_GEOMETRY", "PROJECT_DECLARED_GEOMETRY"]


class ReservedEnvelopeV1(StrictContractModel):
    """The space the resolution reserved for a customer-supplied component of known geometry: the cylinder about the
    component's local +Z (its optical axis or face normal), ``diameter_mm`` wide, from ``z_min_mm`` to ``z_max_mm`` in the
    component's own frame. ``basis`` says where the geometry comes from: ``CATALOG_DESIGN_GEOMETRY`` (the public
    catalog's design geometry of its OpenSci product) or ``PROJECT_DECLARED_GEOMETRY`` (the optical model the Project
    declares for a component of its own). OpenSci hardware keeps clear of it and of the column below it down to the
    breadboard top (where the customer's own support stands), and does not block the beam to or from it."""

    shape: Literal["CYLINDER"] = "CYLINDER"
    basis: ReservedEnvelopeBasis = "CATALOG_DESIGN_GEOMETRY"
    diameter_mm: float = Field(gt=0)
    z_min_mm: float
    z_max_mm: float

    @model_validator(mode="after")
    def _extent(self) -> ReservedEnvelopeV1:
        _finite((self.diameter_mm, self.z_min_mm, self.z_max_mm), "reserved envelope")
        if not self.z_min_mm < self.z_max_mm:
            raise ValueError("reserved envelope z_min_mm must be below z_max_mm")
        return self


class CustomerSupplyV1(StrictContractModel):
    """What the customer supplies at a customer-supplied design anchor: its role, why the customer supplies it, the
    OpenSci product it is (a catalog product; absent for the customer's own light source or another user-defined
    component), how it is oriented in the assembly world (its design orientation carried by the design frame) and, when
    its geometry is known (a catalog product's design geometry, or the optical model the Project declares for a component
    of its own), the space the resolution reserved for it. OpenSci does not mount it -- nor supply it, except an optic of
    reason ``OPENSCI_SUPPLIED_UNMOUNTED`` (supplied without mounting hardware); the customer places it at the anchor
    position."""

    reason: CustomerSupplyReasonV1
    optical_role: DesignRole
    opensci_sku: Annotated[str, StringConstraints(strict=True, pattern=r"^OS-OPT(-[A-Z0-9]+)+$")] | None = None
    orientation_wxyz: tuple[float, float, float, float] = Field(
        description="The component frame's orientation in the assembly world (unit quaternion w, x, y, z).")
    reserved_envelope: ReservedEnvelopeV1 | None = None

    @model_validator(mode="after")
    def _consistent(self) -> CustomerSupplyV1:
        _finite(self.orientation_wxyz, "customer supply orientation")
        if abs(math.sqrt(sum(c * c for c in self.orientation_wxyz)) - 1.0) > UNIT_TOLERANCE:
            raise ValueError("customer supply orientation_wxyz must be a unit quaternion")
        catalog = self.reason in CATALOG_SUPPLY_REASONS
        if catalog != (self.opensci_sku is not None):
            raise ValueError(f"a customer-supplied {self.reason.value} component {'names' if catalog else 'has no'} OpenSci product")
        if (self.reason is CustomerSupplyReasonV1.CUSTOMER_LIGHT_SOURCE) != (self.optical_role == "source"):
            raise ValueError("exactly the customer's light sources have the reason CUSTOMER_LIGHT_SOURCE")
        if self.reserved_envelope is not None:
            declared = self.reason is CustomerSupplyReasonV1.NOT_IN_CATALOG
            if not (catalog or declared):
                raise ValueError("only a catalog product or a component with a declared optical model has a reserved envelope")
            expected = "CATALOG_DESIGN_GEOMETRY" if catalog else "PROJECT_DECLARED_GEOMETRY"
            if self.reserved_envelope.basis != expected:
                raise ValueError(f"a customer-supplied {self.reason.value} component's reserved envelope has basis {expected}")
        return self


class DesignAnchorV1(StrictContractModel):
    """Where a Project component or observation lands in the assembly world."""

    instance_id: InstanceKey
    anchor_kind: DesignAnchorKindV1
    position_mm: Vector3
    realized_by_part_ref: PartRef | None = None
    customer_supplied: bool = Field(
        False, description="Placed and mounted by the customer: their own hardware (their light source, or an optical "
                           "component OpenSci does not supply), or an optic OpenSci supplies without mounting hardware "
                           "(customer_supply.reason OPENSCI_SUPPLIED_UNMOUNTED). OpenSci does not place or mount it.")
    customer_supply: CustomerSupplyV1 | None = Field(
        None, description="What the customer supplies here: present exactly for a customer-supplied anchor.")

    @model_validator(mode="after")
    def _supply(self) -> DesignAnchorV1:
        if self.customer_supplied and (self.realized_by_part_ref is not None or self.anchor_kind is not DesignAnchorKindV1.COMPONENT):
            raise ValueError(f"design anchor {self.instance_id!r}: a customer-supplied anchor is an unrealized COMPONENT")
        if self.customer_supplied != (self.customer_supply is not None):
            raise ValueError(f"design anchor {self.instance_id!r}: customer_supply is given for, and only for, a customer-supplied anchor")
        return self


class BeamReferenceV1(StrictContractModel):
    """How to overlay the optical beam on the resolved mechanics, without copying the optical run.

    The client transforms its own local run of the bound Project (optical scene / ray paths, matched by the
    Engine input/result hashes) with ``design_frame_in_world``. ``beam_axes`` are the nominal beam lines, for a client without its own run;
    each names the design anchor at every point (``BeamAxisV1.through``).
    """

    overlay_source: Literal["CLIENT_LOCAL_RUN"] = "CLIENT_LOCAL_RUN"
    design_frame_in_world: PoseV1
    engine_input_hash: HashRef | None = None
    engine_result_hash: HashRef | None = None
    optical_axis_height_mm: float | None = Field(None, description="Height of the realized optics above the breadboard top.")
    design_anchors: tuple[DesignAnchorV1, ...] = ()
    beam_axes: tuple[BeamAxisV1, ...] = ()

    @field_validator("engine_input_hash")
    @classmethod
    def _input(cls, value: HashRef | None) -> HashRef | None:
        if value is not None:
            require_hash_profile(value, HashProfile.ENGINE_EXACT_PY1, field_name="engine_input_hash")
            if value.subject != ENGINE_INPUT_HASH_SUBJECT:
                raise ValueError(f"engine_input_hash subject must be {ENGINE_INPUT_HASH_SUBJECT!r}")
        return value

    @field_validator("engine_result_hash")
    @classmethod
    def _result(cls, value: HashRef | None) -> HashRef | None:
        if value is not None:
            require_engine_quantized_params(value, field_name="engine_result_hash")
            if value.subject != ENGINE_RESULT_HASH_SUBJECT:
                raise ValueError(f"engine_result_hash subject must be {ENGINE_RESULT_HASH_SUBJECT!r}")
        return value

    @model_validator(mode="after")
    def _check(self) -> BeamReferenceV1:
        if dup := _duplicates([a.instance_id for a in self.design_anchors]):
            raise ValueError(f"duplicate design anchor(s) {dup}")
        if dup := _duplicates([a.path_key for a in self.beam_axes]):
            raise ValueError(f"duplicate beam axis path_key(s) {dup}")
        return self


# ------------------------------------------------------------------------------------------ DOF delivery


class DeliveredControlV1(StrEnum):
    MANUAL = "MANUAL"
    NUMERICAL = "NUMERICAL"
    NONE = "NONE"


class DofDeliveryStatusV1(StrEnum):
    DELIVERED = "DELIVERED"  # delivered with the requested (or a stronger) control
    DELIVERED_AS_MANUAL = "DELIVERED_AS_MANUAL"  # requested NUMERICAL, delivered as a manual adjustment
    NOT_DELIVERED = "NOT_DELIVERED"  # the axis is fixed in the assembly (or belongs to customer-supplied hardware)


class DofMechanismV1(StrEnum):
    POST_HEIGHT_CLAMP = "POST_HEIGHT_CLAMP"
    POST_ROTATION_CLAMP = "POST_ROTATION_CLAMP"
    BASE_SLOT_TRAVEL = "BASE_SLOT_TRAVEL"
    LINEAR_STAGE = "LINEAR_STAGE"
    ROTATION_STAGE = "ROTATION_STAGE"
    TIP_TILT_MOUNT = "TIP_TILT_MOUNT"
    MOTORIZED_STAGE = "MOTORIZED_STAGE"
    NONE = "NONE"
    TRANSLATION_MOUNT = "TRANSLATION_MOUNT"  # transverse translation adjusters of the optic's own mount
    ROTATION_MOUNT = "ROTATION_MOUNT"  # rotation of the optic about its optical axis in its own mount


class AdjustmentCharacterV1(StrEnum):
    CONTINUOUS_CLAMPED = "CONTINUOUS_CLAMPED"  # loosen a clamp, move continuously, re-clamp
    COARSE_REPOSITION = "COARSE_REPOSITION"  # unfasten, slide/reseat, re-fasten
    FINE_ADJUSTER = "FINE_ADJUSTER"  # screw / micrometer adjuster
    ACTUATED = "ACTUATED"


class TravelBasisV1(StrEnum):
    """Where an adjustment travel range comes from."""

    RESOLVED = "RESOLVED"  # the range the resolved parts give at this placement
    NOMINAL = "NOMINAL"  # the nominal travel of the delivering mechanism


class DofImplementationV1(StrictContractModel):
    """How one retained design DOF (Project, component-local) was physically delivered.

    ``axis`` is the Project's component-local label (``Tx``..``Rz``), not a world axis: show a viewer
    ``world_direction``, the component-local +axis expressed in the assembly world (translation direction,
    or rotation axis through the optic part's origin). ``travel_min``/``travel_max`` are relative to the
    nominal pose along that +axis (mm or deg).

    ``implemented_by_part_ref`` carries the mechanism (clamp, slot, stage). ``moved_part_ref`` is the part
    that moves relative to its support when the DOF is used; it moves together with every part it carries
    (its support subtree), so it is the optic itself or one of the optic's supports. NUMERICAL can only be
    delivered by an actuated part; the scene validator enforces that.
    """

    design_instance_id: InstanceKey
    optic_part_ref: PartRef | None = Field(
        None, description="The part realizing the design component; absent only for a customer-supplied component (never delivered).")
    axis: DofAxisV1
    requested_control: DofControlIntentV1
    delivered_control: DeliveredControlV1
    status: DofDeliveryStatusV1
    mechanism: DofMechanismV1
    implemented_by_part_ref: PartRef | None = None
    moved_part_ref: PartRef | None = None
    world_direction: Vector3
    travel_min: float | None = None
    travel_max: float | None = None
    travel_basis: TravelBasisV1 | None = None
    adjustment: AdjustmentCharacterV1 | None = None

    @model_validator(mode="after")
    def _check(self) -> DofImplementationV1:
        where = f"DOF {self.design_instance_id}.{self.axis.value}"
        _finite(self.world_direction, where)
        if abs(math.sqrt(sum(c * c for c in self.world_direction)) - 1.0) > UNIT_TOLERANCE:
            raise ValueError(f"{where}: world_direction must be a unit vector")
        none_fields = (self.implemented_by_part_ref, self.moved_part_ref, self.travel_min, self.travel_max, self.travel_basis, self.adjustment)
        if self.delivered_control is DeliveredControlV1.NONE:
            if self.status is not DofDeliveryStatusV1.NOT_DELIVERED or self.mechanism is not DofMechanismV1.NONE:
                raise ValueError(f"{where}: delivered_control NONE means status NOT_DELIVERED and mechanism NONE")
            if any(v is not None for v in none_fields):
                raise ValueError(f"{where}: an undelivered DOF has no implementing or moved part, travel or adjustment")
            return self
        if (self.mechanism is DofMechanismV1.NONE or self.implemented_by_part_ref is None or self.moved_part_ref is None
                or self.adjustment is None):
            raise ValueError(f"{where}: a delivered DOF names its mechanism, implementing part, moved part and adjustment character")
        numerical = self.delivered_control is DeliveredControlV1.NUMERICAL
        if numerical != (self.mechanism is DofMechanismV1.MOTORIZED_STAGE) or numerical != (self.adjustment is AdjustmentCharacterV1.ACTUATED):
            raise ValueError(f"{where}: NUMERICAL delivery requires, and only it may use, an actuated MOTORIZED_STAGE")
        as_manual = self.requested_control is DofControlIntentV1.NUMERICAL and not numerical
        expected = DofDeliveryStatusV1.DELIVERED_AS_MANUAL if as_manual else DofDeliveryStatusV1.DELIVERED
        if self.status is not expected:
            raise ValueError(f"{where}: requested {self.requested_control.value}, delivered {self.delivered_control.value} means status {expected.value}")
        if (self.travel_min is None) != (self.travel_max is None) or (self.travel_min is None) != (self.travel_basis is None):
            raise ValueError(f"{where}: travel_min, travel_max and travel_basis are given together")
        if self.travel_min is not None and not (self.travel_min <= 0.0 <= self.travel_max):  # type: ignore[operator]
            raise ValueError(f"{where}: travel must contain the nominal pose (travel_min <= 0 <= travel_max)")
        return self


# ------------------------------------------------------------------------------------------------ notes


class AssemblyNoteCodeV1(StrEnum):
    MANUAL_OPERATION_REQUIRED = "MANUAL_OPERATION_REQUIRED"
    DESIGN_COMPONENT_NOT_SUPPLIED = "DESIGN_COMPONENT_NOT_SUPPLIED"
    DESIGN_COMPONENT_OUTSIDE_BREADBOARD = "DESIGN_COMPONENT_OUTSIDE_BREADBOARD"
    EXTERNAL_LIGHT_SOURCE = "EXTERNAL_LIGHT_SOURCE"
    CUSTOMER_SUPPLIED_COMPONENT = "CUSTOMER_SUPPLIED_COMPONENT"
    DOF_MANUAL_ADJUSTMENT = "DOF_MANUAL_ADJUSTMENT"
    DOF_FIXED = "DOF_FIXED"
    DOF_ON_CUSTOMER_HARDWARE = "DOF_ON_CUSTOMER_HARDWARE"


class NoteSubjectV1(StrEnum):
    ASSEMBLY = "ASSEMBLY"
    PART = "PART"
    DESIGN_COMPONENT = "DESIGN_COMPONENT"
    DOF = "DOF"


NOTE_SUBJECTS: dict[AssemblyNoteCodeV1, NoteSubjectV1] = {
    AssemblyNoteCodeV1.MANUAL_OPERATION_REQUIRED: NoteSubjectV1.PART,
    AssemblyNoteCodeV1.DESIGN_COMPONENT_NOT_SUPPLIED: NoteSubjectV1.DESIGN_COMPONENT,
    AssemblyNoteCodeV1.DESIGN_COMPONENT_OUTSIDE_BREADBOARD: NoteSubjectV1.DESIGN_COMPONENT,
    AssemblyNoteCodeV1.EXTERNAL_LIGHT_SOURCE: NoteSubjectV1.DESIGN_COMPONENT,
    AssemblyNoteCodeV1.CUSTOMER_SUPPLIED_COMPONENT: NoteSubjectV1.DESIGN_COMPONENT,
    AssemblyNoteCodeV1.DOF_MANUAL_ADJUSTMENT: NoteSubjectV1.DOF,
    AssemblyNoteCodeV1.DOF_FIXED: NoteSubjectV1.DOF,
    AssemblyNoteCodeV1.DOF_ON_CUSTOMER_HARDWARE: NoteSubjectV1.DOF,
}

#: Fixed user-facing wording, chosen by code. ``{axis}``, ``{instance}``, ``{requested}`` are the only fields.
NOTE_TEMPLATES: dict[AssemblyNoteCodeV1, str] = {
    AssemblyNoteCodeV1.MANUAL_OPERATION_REQUIRED: "A manual assembly operation is required for this part.",
    AssemblyNoteCodeV1.DESIGN_COMPONENT_NOT_SUPPLIED: "This design component is not part of the supplied assembly.",
    AssemblyNoteCodeV1.DESIGN_COMPONENT_OUTSIDE_BREADBOARD: "This design component lies outside the breadboard outline.",
    AssemblyNoteCodeV1.EXTERNAL_LIGHT_SOURCE: (
        "This light source is customer-supplied hardware. Place it so that its beam enters the assembly along the "
        "designed path."
    ),
    # the wording of a component whose shape is unknown (no reserved envelope); see CUSTOMER_SUPPLIED_RESERVED_MESSAGE
    AssemblyNoteCodeV1.CUSTOMER_SUPPLIED_COMPONENT: (
        "This component is customer-supplied: OpenSci does not supply or mount it. You set it up at its design position "
        "and orientation, on your own support; its shape is unknown, so only its position is marked and no space is "
        "reserved for it."
    ),
    AssemblyNoteCodeV1.DOF_MANUAL_ADJUSTMENT: (
        "{axis} of {instance} is a manual adjustment; the design specifies numerical control."
    ),
    AssemblyNoteCodeV1.DOF_FIXED: "{axis} of {instance} is fixed in this assembly; the design specifies {requested} adjustment.",
    AssemblyNoteCodeV1.DOF_ON_CUSTOMER_HARDWARE: "{axis} of {instance} is adjusted on your own hardware.",
}

#: The CUSTOMER_SUPPLIED_COMPONENT wording when its anchor carries a reserved envelope (the space is said to
#: be kept free only when it was). Neither wording assumes who assembles the OpenSci parts.
CUSTOMER_SUPPLIED_RESERVED_MESSAGE = (
    "This component is customer-supplied: OpenSci does not supply or mount it. You set it up at its design position and "
    "orientation, on your own support; the space it needs is kept free."
)

#: The CUSTOMER_SUPPLIED_COMPONENT wording of an optic OpenSci supplies without mounting hardware (its anchor's reason
#: is OPENSCI_SUPPLIED_UNMOUNTED; its catalog design geometry is always reserved).
SUPPLIED_UNMOUNTED_MESSAGE = (
    "OpenSci supplies this optic without mounting hardware (it has no mount for it). You mount it at its design position "
    "and orientation, on your own support; the space it needs is kept free."
)

#: Notes a viewer highlights: they change what the customer can adjust or where a component sits.
WARNING_NOTE_CODES: frozenset[AssemblyNoteCodeV1] = frozenset({
    AssemblyNoteCodeV1.MANUAL_OPERATION_REQUIRED,
    AssemblyNoteCodeV1.DESIGN_COMPONENT_OUTSIDE_BREADBOARD,
    AssemblyNoteCodeV1.DOF_MANUAL_ADJUSTMENT,
    AssemblyNoteCodeV1.DOF_FIXED,
})


def note_message(code: AssemblyNoteCodeV1, *, axis: DofAxisV1 | None = None, instance: str | None = None,
                 requested: DofControlIntentV1 | None = None, reserved_space: bool = False,
                 supplied_unmounted: bool = False) -> str:
    """The one allowed message text for a note code (and, for DOF codes, its DOF; for CUSTOMER_SUPPLIED_COMPONENT,
    whether its anchor is an optic OpenSci supplies unmounted, else whether it carries a reserved envelope)."""

    if code is AssemblyNoteCodeV1.CUSTOMER_SUPPLIED_COMPONENT and supplied_unmounted:
        return SUPPLIED_UNMOUNTED_MESSAGE
    if code is AssemblyNoteCodeV1.CUSTOMER_SUPPLIED_COMPONENT and reserved_space:
        return CUSTOMER_SUPPLIED_RESERVED_MESSAGE
    template = NOTE_TEMPLATES[code]
    if NOTE_SUBJECTS[code] is not NoteSubjectV1.DOF:
        return template
    if axis is None or instance is None or requested is None:
        raise ValueError(f"{code.value}: a DOF note needs axis, instance and requested control")
    return template.format(axis=axis.value, instance=instance, requested=requested.value.lower())


def dof_note_code(dof: DofImplementationV1, *, customer_supplied: bool) -> AssemblyNoteCodeV1 | None:
    """The note one DOF delivery carries, if any."""

    if dof.status is DofDeliveryStatusV1.DELIVERED_AS_MANUAL:
        return AssemblyNoteCodeV1.DOF_MANUAL_ADJUSTMENT
    if dof.status is DofDeliveryStatusV1.NOT_DELIVERED:
        return AssemblyNoteCodeV1.DOF_ON_CUSTOMER_HARDWARE if customer_supplied else AssemblyNoteCodeV1.DOF_FIXED
    return None


class AssemblyNoteV1(StrictContractModel):
    code: AssemblyNoteCodeV1
    severity: Literal["INFO", "WARNING"]
    part_ref: PartRef | None = None
    design_instance_id: InstanceKey | None = None
    dof_axis: DofAxisV1 | None = None
    message: str


# -------------------------------------------------------------------------------------------- resolution


class ResolutionCheckKindV1(StrEnum):
    HEIGHT_CLOSURE = "HEIGHT_CLOSURE"  # each optic axis height closes over its mount, post, holder and insertion depth
    INTERFACE_FIT = "INTERFACE_FIT"  # mount, post and holder interfaces match
    INTRA_CHAIN_ENVELOPE = "INTRA_CHAIN_ENVELOPE"  # bodies of one support chain meet only at their support contacts
    BASE_FIXATION = "BASE_FIXATION"  # each base is fixed on its own breadboard hole
    PART_DIMENSIONS = "PART_DIMENSIONS"  # each part's envelope follows its dimensional data
    ENGAGEMENT_RANGE = "ENGAGEMENT_RANGE"  # each post insertion lies within its holder's insertion range
    ASSEMBLED_LAYOUT = "ASSEMBLED_LAYOUT"  # the assembled optical layout reproduces the design
    INTER_CHAIN_CLEARANCE = "INTER_CHAIN_CLEARANCE"  # bodies of different support chains keep clear
    BEAM_PATH_CLEARANCE = "BEAM_PATH_CLEARANCE"  # the nominal beam passes through no foreign hardware


class ResolutionSummaryV1(StrictContractModel):
    """The layout checks the resolution of this assembly passed."""

    passed_checks: tuple[ResolutionCheckKindV1, ...] = ()

    @model_validator(mode="after")
    def _check(self) -> ResolutionSummaryV1:
        if list(self.passed_checks) != sorted(set(self.passed_checks), key=lambda c: c.value):
            raise ValueError("passed_checks must be unique and sorted")
        return self


class SceneBoundsV1(StrictContractModel):
    """World AABB of the placed part geometry, the design anchors and the beam axes."""

    min_mm: Vector3
    max_mm: Vector3


# ------------------------------------------------------------------------------------------------- scene

class AssemblyPreviewSceneV1(StrictContractModel):
    schema_id: Literal["opensci.assembly_preview.scene"] = Field(ASSEMBLY_PREVIEW_SCENE_SCHEMA_ID, alias="schema")
    schema_version: Literal["3.0", "3.1"] = Field(
        ASSEMBLY_PREVIEW_SCENE_SCHEMA_VERSION,
        description="3.0, or 3.1 when a beam axis carries segment_roles or notes (required_scene_schema_version).")
    view_kind: Literal["EXPERIMENTAL_LAB_ASSEMBLY"] = ASSEMBLY_PREVIEW_VIEW_KIND
    description: Literal[
        "Experimental Lab Assembly of one resolved OpenSci project: every supplied part with its OpenSci product "
        "identity, world pose, support and mates on the resolved breadboard. Each part names its Customer Purchase BOM "
        "line in commercial_line_ref; quantities and prices are those of that BOM."
    ] = ASSEMBLY_PREVIEW_DESCRIPTION
    frame: Literal["ASSEMBLY_WORLD"] = "ASSEMBLY_WORLD"
    length_unit: Literal["mm"] = "mm"
    up_axis: Literal["+Z"] = "+Z"
    resolution: ResolutionSummaryV1
    parts: tuple[PreviewPartV1, ...] = Field(min_length=1, max_length=MAX_PARTS)
    mates: tuple[MateV1, ...] = ()
    settings: tuple[SettingV1, ...] = ()
    fixations: tuple[FixationV1, ...] = ()
    manual_operations: tuple[ManualOperationV1, ...] = ()
    breadboard: BreadboardV1
    beam_reference: BeamReferenceV1
    dof_implementations: tuple[DofImplementationV1, ...] = ()
    notes: tuple[AssemblyNoteV1, ...] = ()
    bounds: SceneBoundsV1

    @model_validator(mode="after")
    def _check(self) -> AssemblyPreviewSceneV1:  # noqa: C901 - one flat list of cross-reference rules
        if self.schema_version == ASSEMBLY_PREVIEW_SCENE_SCHEMA_VERSION and (
                features := scene_schema_3_1_features(self.beam_reference.beam_axes)):
            raise ValueError(f"a scene 3.0 document uses scene 3.1 fields {list(features)}: state schema_version 3.1")
        parts = {p.part_ref: p for p in self.parts}
        if dup := _duplicates([p.part_ref for p in self.parts]):
            raise ValueError(f"duplicate part_ref(s) {dup}")
        mates = {m.mate_ref: m for m in self.mates}
        if dup := _duplicates([m.mate_ref for m in self.mates]):
            raise ValueError(f"duplicate mate_ref(s) {dup}")
        operations = {o.operation_ref: o for o in self.manual_operations}
        if dup := _duplicates([o.operation_ref for o in self.manual_operations]):
            raise ValueError(f"duplicate operation_ref(s) {dup}")

        def need_part(ref: str | None, where: str) -> PreviewPartV1 | None:
            if ref is None:
                return None
            if ref not in parts:
                raise ValueError(f"{where}: unknown part_ref {ref!r}")
            return parts[ref]

        for mate in self.mates:
            for ref, datum in ((mate.part_a_ref, mate.datum_a), (mate.part_b_ref, mate.datum_b)):
                part = need_part(ref, f"mate {mate.mate_ref!r}")
                if datum not in {d.datum_key for d in part.datums}:  # type: ignore[union-attr]
                    raise ValueError(f"mate {mate.mate_ref!r}: part {ref!r} has no datum {datum!r}")
            if mate.part_a_ref == mate.part_b_ref:
                raise ValueError(f"mate {mate.mate_ref!r} joins a part to itself")

        roots = [p.part_ref for p in self.parts if p.support is None]
        if roots != [self.breadboard.part_ref]:
            raise ValueError(f"the scene graph root must be the breadboard {self.breadboard.part_ref!r}, found {roots}")
        board_part = need_part(self.breadboard.part_ref, "breadboard")
        if board_part.category is not PartCategoryV1.BREADBOARD:  # type: ignore[union-attr]
            raise ValueError("breadboard.part_ref must name a BREADBOARD part")
        customer_table = self.breadboard.supplied_by == "CUSTOMER"
        if customer_table and (board_part.commercial_line_ref is not None  # type: ignore[union-attr]
                               or board_part.identity.identity_kind is not IdentityKindV1.CUSTOMER_SUPPLIED):  # type: ignore[union-attr]
            raise ValueError("the customer's own optical table has the CUSTOMER_SUPPLIED identity and no Customer Purchase "
                             "BOM line")
        if customer_supplied_parts := [p.part_ref for p in self.parts if p.identity.identity_kind is IdentityKindV1.CUSTOMER_SUPPLIED
                                       and not (customer_table and p.part_ref == self.breadboard.part_ref)]:
            raise ValueError(f"only the customer's own optical table is a CUSTOMER_SUPPLIED part, not {customer_supplied_parts}")
        for part in self.parts:
            if part.support is None:
                continue
            need_part(part.support.parent_ref, f"support of {part.part_ref!r}")
            mate = mates.get(part.support.mate_ref)
            if mate is None or {mate.part_a_ref, mate.part_b_ref} != {part.part_ref, part.support.parent_ref}:
                raise ValueError(f"support of {part.part_ref!r}: mate {part.support.mate_ref!r} must join it to its parent")
            seen, ref = set(), part.part_ref
            while ref is not None:
                if ref in seen:
                    raise ValueError(f"support cycle through {part.part_ref!r}")
                seen.add(ref)
                support = parts[ref].support
                ref = support.parent_ref if support is not None else None
            if part.assembly_order <= parts[part.support.parent_ref].assembly_order:
                raise ValueError(f"part {part.part_ref!r} must be assembled after its support {part.support.parent_ref!r}")

        if dup := _duplicates([(p.design_binding.instance_id) for p in self.parts if p.design_binding is not None]):
            raise ValueError(f"design instance(s) realized by more than one part: {dup}")
        if dup := _duplicates([(s.part_ref, s.kind) for s in self.settings]):
            raise ValueError(f"duplicate setting(s) {dup}")
        for setting in self.settings:
            need_part(setting.part_ref, "setting")
        if dup := _duplicates([f.part_ref for f in self.fixations]):
            raise ValueError(f"part(s) with more than one fixation: {dup}")
        grid = self.breadboard.hole_grid
        for fixation in self.fixations:
            need_part(fixation.part_ref, "fixation")
            if fixation.manual_operation_ref is not None and fixation.manual_operation_ref not in operations:
                raise ValueError(f"fixation of {fixation.part_ref!r}: unknown manual operation {fixation.manual_operation_ref!r}")
            for i, j in fixation.board_holes:
                if grid is None or not (0 <= i < grid.count[0] and 0 <= j < grid.count[1]):
                    raise ValueError(f"fixation of {fixation.part_ref!r}: hole ({i}, {j}) is not on the breadboard grid")
        for operation in self.manual_operations:
            need_part(operation.part_ref, f"manual operation {operation.operation_ref!r}")

        anchors = {a.instance_id: a for a in self.beam_reference.design_anchors}
        for anchor in self.beam_reference.design_anchors:
            part = need_part(anchor.realized_by_part_ref, f"design anchor {anchor.instance_id!r}")
            if part is not None and (part.design_binding is None or part.design_binding.instance_id != anchor.instance_id):
                raise ValueError(f"design anchor {anchor.instance_id!r}: part {part.part_ref!r} does not realize it")
        for axis in self.beam_reference.beam_axes:
            if unknown := sorted({i for i in axis.through or () if i not in anchors}):
                raise ValueError(f"beam axis {axis.path_key!r} names unknown design anchor(s) {unknown}")

        dofs = {}
        for dof in self.dof_implementations:
            key = (dof.design_instance_id, dof.axis)
            if key in dofs:
                raise ValueError(f"duplicate DOF implementation {dof.design_instance_id}.{dof.axis.value}")
            dofs[key] = dof
            if dof.optic_part_ref is None:
                anchor = anchors.get(dof.design_instance_id)
                if anchor is None or not anchor.customer_supplied or dof.status is not DofDeliveryStatusV1.NOT_DELIVERED:
                    raise ValueError(f"DOF {dof.design_instance_id}.{dof.axis.value}: without an optic part it must belong to a "
                                     "customer-supplied design anchor and be NOT_DELIVERED")
                continue
            optic = need_part(dof.optic_part_ref, f"DOF {key}")
            if optic.design_binding is None or optic.design_binding.instance_id != dof.design_instance_id:  # type: ignore[union-attr]
                raise ValueError(f"DOF {dof.design_instance_id}.{dof.axis.value}: {dof.optic_part_ref!r} does not realize that component")
            actor = need_part(dof.implemented_by_part_ref, f"DOF {key}")
            if dof.moved_part_ref is not None:
                need_part(dof.moved_part_ref, f"DOF {key}")
                carriers, ref = [], dof.optic_part_ref
                while ref is not None:
                    carriers.append(ref)
                    support = parts[ref].support
                    ref = support.parent_ref if support is not None else None
                if dof.moved_part_ref not in carriers[:-1]:
                    raise ValueError(f"DOF {dof.design_instance_id}.{dof.axis.value}: moved part {dof.moved_part_ref!r} must be the optic or "
                                     "one of its supports (not the breadboard)")
            if dof.delivered_control is DeliveredControlV1.NUMERICAL and not (actor is not None and actor.actuated):
                raise ValueError(f"DOF {dof.design_instance_id}.{dof.axis.value}: NUMERICAL delivery needs an actuated part")

        customer_supplied = {a.instance_id for a in self.beam_reference.design_anchors if a.customer_supplied}
        sources = {a.instance_id: a.customer_supply.optical_role == "source"  # type: ignore[union-attr]
                   for a in self.beam_reference.design_anchors if a.customer_supplied}
        required: set[tuple[AssemblyNoteCodeV1, str, DofAxisV1]] = set()
        for dof in self.dof_implementations:
            code = dof_note_code(dof, customer_supplied=dof.design_instance_id in customer_supplied)
            if code is not None:
                required.add((code, dof.design_instance_id, dof.axis))
        present: set[tuple[AssemblyNoteCodeV1, str, DofAxisV1]] = set()
        reserved = {a.instance_id for a in self.beam_reference.design_anchors
                    if a.customer_supplied and a.customer_supply.reserved_envelope is not None}  # type: ignore[union-attr]
        unmounted = {a.instance_id for a in self.beam_reference.design_anchors
                     if a.customer_supplied and a.customer_supply.reason is CustomerSupplyReasonV1.OPENSCI_SUPPLIED_UNMOUNTED}  # type: ignore[union-attr]
        for item in self.notes:
            self._check_note(item, parts, dofs, reserved, unmounted)
            if item.dof_axis is not None:
                present.add((item.code, item.design_instance_id, item.dof_axis))  # type: ignore[arg-type]
        if present != required:
            raise ValueError(f"DOF notes must be exactly one per manual, fixed or customer-hardware DOF: missing "
                             f"{sorted((c.value, i, a.value) for c, i, a in required - present)}, unexpected "
                             f"{sorted((c.value, i, a.value) for c, i, a in present - required)}")
        if dup := _duplicates([(d.code, d.part_ref, d.design_instance_id, d.dof_axis) for d in self.notes]):
            raise ValueError(f"duplicate note(s) {dup}")
        for item in self.notes:
            external = (item.design_instance_id or "") in customer_supplied
            if item.code is AssemblyNoteCodeV1.EXTERNAL_LIGHT_SOURCE and not (external and sources.get(item.design_instance_id)):
                raise ValueError(f"EXTERNAL_LIGHT_SOURCE names {item.design_instance_id!r}, which is not a customer-supplied light source")
            if item.code is AssemblyNoteCodeV1.CUSTOMER_SUPPLIED_COMPONENT and not (external and not sources.get(item.design_instance_id)):
                raise ValueError(f"CUSTOMER_SUPPLIED_COMPONENT names {item.design_instance_id!r}, which is not a customer-supplied "
                                 "optical component")
            if item.code is AssemblyNoteCodeV1.DESIGN_COMPONENT_OUTSIDE_BREADBOARD and external:
                raise ValueError(f"{item.design_instance_id!r} is customer-supplied hardware; it is not placed on the breadboard")
            if item.severity != ("WARNING" if item.code in WARNING_NOTE_CODES else "INFO"):
                raise ValueError(f"note {item.code.value}: severity must be {'WARNING' if item.code in WARNING_NOTE_CODES else 'INFO'}")
        _finite(self.bounds.min_mm + self.bounds.max_mm, "scene bounds")
        return self

    @staticmethod
    def _check_note(item: AssemblyNoteV1, parts: dict[str, PreviewPartV1],
                    dofs: dict[tuple[str, DofAxisV1], DofImplementationV1], reserved: set[str],
                    unmounted: set[str] = frozenset()) -> None:  # type: ignore[assignment]
        subject = NOTE_SUBJECTS[item.code]
        given = (item.part_ref is not None, item.design_instance_id is not None, item.dof_axis is not None)
        expected = {
            NoteSubjectV1.ASSEMBLY: (False, False, False),
            NoteSubjectV1.PART: (True, False, False),
            NoteSubjectV1.DESIGN_COMPONENT: (False, True, False),
            NoteSubjectV1.DOF: (False, True, True),
        }[subject]
        if given != expected:
            raise ValueError(f"note {item.code.value}: subject fields do not match a {subject.value} subject")
        if item.part_ref is not None and item.part_ref not in parts:
            raise ValueError(f"note {item.code.value}: unknown part_ref {item.part_ref!r}")
        requested = None
        if subject is NoteSubjectV1.DOF:
            dof = dofs.get((item.design_instance_id, item.dof_axis))  # type: ignore[arg-type]
            if dof is None:
                raise ValueError(f"note {item.code.value}: no DOF implementation for {item.design_instance_id}.{item.dof_axis}")
            requested = dof.requested_control
        expected_text = note_message(item.code, axis=item.dof_axis, instance=item.design_instance_id, requested=requested,
                                     reserved_space=item.design_instance_id in reserved,
                                     supplied_unmounted=item.design_instance_id in unmounted)
        if item.message != expected_text:
            raise ValueError(f"note {item.code.value}: message must be the fixed template text")

    def payload(self) -> dict[str, Any]:
        """Canonical serialized form (``None`` fields omitted)."""

        return self.model_dump(mode="json", by_alias=True, exclude_none=True)

    def content_hash(self) -> str:
        return osc1_digest(self.payload())

    def part(self, part_ref: str) -> PreviewPartV1:
        for part in self.parts:
            if part.part_ref == part_ref:
                return part
        raise KeyError(part_ref)
